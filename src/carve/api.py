"""Public CARVE API."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from sklearn.cluster import AgglomerativeClustering

from ._types import GridSpec, PreprocOption, RunMode, resolve_mode
from ._output import _print_run_footer, _print_run_header
from ._runner import run_validation
from ._consensus import compute_consensus_metrics
from ._selection import select_best_estimator, select_best_k, select_best_row_by_rule

from ._plotting import (
    _get_annotation,
    plot_metric_over_n_clusters as _plot_metric_over_n_clusters,
    plot_consensus_matrix as _plot_consensus_matrix,
    plot_cluster_boxplot as _plot_cluster_boxplot,
    plot_cluster_violin as _plot_cluster_violin,
    plot_cluster_scatter as _plot_cluster_scatter,
    plot_diagnostic_scatter as _plot_diagnostic_scatter,
)

from ._grids import (
    default_estimator_grids,
    default_normalization_options,
    default_dim_reduction_options,
)

from ._utils import (
    align_cluster_labels,
    ensure_2d_array,
    summarize_preprocessing_records,
    _coerce_n_clusters,
)


@dataclass
class CARVE(BaseEstimator):
    """Stability and generalizability validator for clustering methods.

    CARVE evaluates clustering robustness by repeatedly subsampling the
    input data, running clustering algorithms on each subsample, building
    consensus matrices, and computing stability and generalizability metrics.
    It is compatible with the scikit-learn estimator interface.

    Parameters
    ----------
    n_clusters : int or np.ndarray, default=10
        Number(s) of clusters to evaluate. If an integer *K* is provided,
        all values from 2 to *K* (inclusive) are evaluated.
    n_resamples : int, default=100
        Number of resampling iterations per estimator configuration.
    subsample_ratio : float, default=0.618
        Fraction of samples drawn without replacement per resample.
        Must be in (0, 1).
    estimator_param_grids : list of (Estimator, param_grid) tuples, or {"light", "full"}, default="light"
        Clustering estimators and their parameter grids. ``"light"`` uses
        KMeans, Ward-linkage agglomerative, and self-tuning spectral
        clustering. ``"full"`` additionally includes average/single-linkage
        agglomerative and RBF-kernel spectral clustering. A custom list
        of (EstimatorClass, param_grid) tuples may also be passed.
    normalization_options : list of preprocessing specs, optional
        Normalization preprocessing options. If None, defaults include
        identity, StandardScaler, and log1p.
    dim_reduction_options : list of dimensionality reduction specs, optional
        Dimensionality reduction preprocessing options. If None, defaults
        include identity, PCA, t-SNE, and UMAP.
    classifier : sklearn classifier instance, optional
        Classifier used to score generalizability. If None (default), a
        ``RandomForestClassifier`` is built with ``n_trees`` trees. Must
        implement the sklearn classifier interface (``fit``/``predict``).
        Cloned per resample to avoid state leakage.
    n_trees : int, default=100
        Number of trees in the default random-forest classifier. Ignored
        when ``classifier`` is provided.
    reference_labels : array-like of shape (n_samples,), optional
        Reference labels used to align cluster assignments across
        successive ``get_labels`` calls so that cluster indices remain
        consistent.
    n_jobs : int, default=1
        Number of parallel jobs for resampling. ``-1`` uses all cores.
    random_state : int, optional
        Seed for the random number generator, ensuring reproducibility.
    verbose : int, default=0
        Verbosity level for console output during fitting.
        ``0`` suppresses all output. ``1`` prints per-configuration
        progress messages. ``2`` prints the full header, per-configuration
        progress, and footer.

    Attributes
    ----------
    estimator_results_ : pandas.DataFrame
        Per-configuration aggregate metrics populated by ``fit``.
    estimator_param_grids_ : list of tuple
        Resolved estimator grids used during fitting.
    preprocessing_results_ : pandas.DataFrame or None
        Preprocessing summary when ``randomize_preprocessing=True``.
    consensus_matrices_ : list of ndarray
        Stability consensus matrices, one per configuration.
    consensus_generalizability_matrices_ : list of ndarray
        Generalizability consensus matrices, one per configuration.
    stability_gini_scores_ : ndarray or None
        Per-sample Gini stability scores for each configuration.
    stability_ce_scores_ : ndarray or None
        Per-sample classification entropy stability scores for each
        configuration.
    generalizability_scores_ : list of ndarray or None
        Per-sample generalizability scores for each configuration.
    X_ : ndarray or None
        Input data stored after fitting.

    Notes
    -----
    The validation pipeline proceeds as follows: for each estimator
    configuration (estimator type x hyperparameters x *k*), CARVE draws
    ``n_resamples`` random subsamples, clusters each subsample, and
    measures stability (intra-subsample ARI and consensus-matrix metrics)
    and generalizability (held-out prediction via a random forest).

    See Also
    --------
    SpectralClusteringCARVE : Custom spectral clustering variant included
        in the default estimator grid.

    Examples
    --------
    >>> from carve import CARVE
    >>> carve = CARVE(n_clusters=10, n_resamples=200, subsample_ratio=0.6)
    >>> carve.fit(X)
    >>> labels = carve.get_labels(measure="stability", rule="1se")
    >>> k = carve.get_k(measure="generalizability", rule="1se")
    """

    # --- Constructor parameters ---
    n_clusters: int | np.ndarray = field(
        default_factory=lambda: np.arange(2, 10 + 1, dtype=int)
    )
    n_resamples: int = 100
    subsample_ratio: float = 0.618

    estimator_param_grids: list[GridSpec] | Literal["light", "full"] = "light"
    normalization_options: list[PreprocOption] | None = None
    dim_reduction_options: list[PreprocOption] | None = None

    classifier: ClassifierMixin | None = None
    n_trees: int = 100

    X_: np.ndarray | None = field(init=False, default=None)
    reference_labels: np.ndarray | None = None

    n_jobs: int = 1
    random_state: int | None = None
    verbose: int = 0

    # --- Fitted attributes (set by fit()) ---
    estimator_results_: pd.DataFrame | None = field(init=False, default=None)
    estimator_param_grids_: list[GridSpec] | None = field(init=False, default=None)
    preprocessing_results_: pd.DataFrame | None = field(init=False, default=None)

    # --- Consensus matrices ---
    consensus_matrices_: list[np.ndarray] | None = field(init=False, default=None)
    consensus_generalizability_matrices_: list[np.ndarray] | None = field(
        init=False, default=None
    )

    # --- Sample-level scores ---
    stability_gini_scores_: np.ndarray | None = field(init=False, default=None)
    stability_ce_scores_: np.ndarray | None = field(init=False, default=None)
    generalizability_scores_: list[np.ndarray] | None = field(init=False, default=None)

    # ------------------------------------------------------------------ #
    #  Core Methods                                                      #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        reference_labels: np.ndarray | None = None,
        randomize_preprocessing: bool = False,
        show_progress: bool = False,
        mode: RunMode = "default",
        random_state: int | None = None,
    ) -> "CARVE":
        """Run CARVE validation on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : ignored
            Included for sklearn compatibility.
        reference_labels : array-like of shape (n_samples,), optional
            Reference labels used for generalizability metrics.
            Overrides the ``reference_labels`` passed at __init__ if given.
        randomize_preprocessing : bool, default=False
            Whether to randomize preprocessing pipelines. When True, a
            random normalization and dimensionality reduction combination
            is sampled independently for each resample iteration.
        show_progress : bool, default=False
            Display a tqdm progress bar over grid configurations.
        mode : {"default", "stability", "generalizability"}, default="default"
            ``"default"`` runs both stability and generalizability
            analyses. ``"stability"`` skips generalizability (no
            held-out prediction). ``"generalizability"`` skips stability
            (no second subsample or consensus metrics).
        random_state : int, optional
            Per-call RNG seed. If None, uses ``self.random_state``.

        Returns
        -------
        self : CARVE
            Fitted instance.

        Raises
        ------
        ValueError
            If estimator parameter grids have inconsistent ``n_clusters``
            values.
        """
        policy = resolve_mode(mode)
        if policy.mode != "default":
            warnings.warn(
                "Non-default mode is experimental and may break downstream functionality.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.classifier is not None and self.n_trees != 100:
            warnings.warn(
                "n_trees is ignored when a custom classifier is provided.",
                RuntimeWarning,
                stacklevel=2,
            )

        # --- Resolve X and reference labels ---
        X = ensure_2d_array(X)
        self.X_ = X

        if reference_labels is not None:
            ref_arr = np.asarray(reference_labels)

            if not np.issubdtype(ref_arr.dtype, np.integer):
                ref_arr, _ = pd.factorize(ref_arr)

            self.reference_labels = ref_arr

        # --- Resolve default grids including n_clusters ---
        if (
            self.estimator_param_grids == "light"
            or self.estimator_param_grids == "full"
        ):  # Default estimator grids
            n_clusters_arr = _coerce_n_clusters(self.n_clusters)
            estimator_param_grids = default_estimator_grids(
                X, n_clusters_arr, preset=self.estimator_param_grids
            )

        else:  # User-provided estimator grids (verify consistency of n_clusters)
            estimator_param_grids = self.estimator_param_grids

            # Extract n_clusters from first grid
            n_clusters_arr = estimator_param_grids[0][1].get("n_clusters", None)

            # Verify all grids have the same n_clusters
            for _, grid in estimator_param_grids[1:]:
                grid_n_clusters = grid.get("n_clusters", None)

                if not np.array_equal(n_clusters_arr, grid_n_clusters):
                    raise ValueError(
                        "All estimator parameter grids must contain the same n_clusters values."
                    )

        self.estimator_param_grids_ = estimator_param_grids

        # --- Resolve preprocessing options ---
        norm_options = self.normalization_options or default_normalization_options()
        dr_options = self.dim_reduction_options or default_dim_reduction_options(
            X, self.subsample_ratio
        )

        # --- Print run header ---
        _print_run_header(
            X=X,
            n_clusters=n_clusters_arr,
            n_resamples=self.n_resamples,
            subsample_ratio=self.subsample_ratio,
            estimator_grids=self.estimator_param_grids_,
            n_jobs=self.n_jobs,
            randomize_preprocessing=randomize_preprocessing,
            random_state=self.random_state if random_state is None else random_state,
            verbose=self.verbose,
        )

        # --- Run validation loop ---
        (
            estimator_records,
            pipeline_records,
            self.consensus_matrices_,
            self.consensus_generalizability_matrices_,
            self.generalizability_scores_,
        ) = run_validation(
            X=X,
            estimator_grids=estimator_param_grids,
            n_resamples=self.n_resamples,
            subsample_ratio=self.subsample_ratio,
            normalization_options=norm_options,
            dim_reduction_options=dr_options,
            classifier=self.classifier,
            n_trees=self.n_trees,
            randomize_preprocessing=randomize_preprocessing,
            n_jobs=self.n_jobs,
            random_state=self.random_state if random_state is None else random_state,
            mode=policy.mode,
            show_progress=show_progress,
            verbose=self.verbose,
        )

        self.estimator_results_ = pd.DataFrame.from_records(estimator_records)

        self.preprocessing_results_ = (
            None
            if not randomize_preprocessing
            else summarize_preprocessing_records(pipeline_records)
        )

        n_rows = int(self.estimator_results_.shape[0])

        # --- Stability-derived metrics ---
        if (
            policy.run_stability and self.consensus_matrices_ is not None
        ):  # Default route
            gini_list, ce_list, pac_list = compute_consensus_metrics(
                self.consensus_matrices_
            )

            self.stability_gini_scores_ = np.vstack(gini_list)
            self.stability_ce_scores_ = np.vstack(ce_list)

            self.estimator_results_["consensus_pac_stability"] = pac_list
            self.estimator_results_["consensus_gini_stability"] = (
                self.stability_gini_scores_.mean(axis=1)
            )
            self.estimator_results_["consensus_ce_stability"] = (
                self.stability_ce_scores_.mean(axis=1)
            )

        else:  # If not running stability, set these attributes to None/NaN
            self.stability_gini_scores_ = None
            self.stability_ce_scores_ = None

            self.estimator_results_["consensus_pac_stability"] = np.full(n_rows, np.nan)
            self.estimator_results_["consensus_gini_stability"] = np.full(
                n_rows, np.nan
            )
            self.estimator_results_["consensus_ce_stability"] = np.full(n_rows, np.nan)

        # --- Generalizability-derived metrics ---
        if (
            policy.run_generalizability and self.generalizability_scores_ is not None
        ):  # Default route
            gen_arr = np.vstack(self.generalizability_scores_)
            self.estimator_results_["accuracy_generalizability"] = gen_arr.mean(axis=1)

        else:  # If not running generalizability, set these attributes to None/NaN
            self.estimator_results_["accuracy_generalizability"] = np.full(
                n_rows, np.nan
            )

        _print_run_footer(estimator_df=self.estimator_results_, verbose=self.verbose)

        return self

    def get_labels(
        self,
        *,
        measure: str = "stability",
        rule: str = "1se",
        k: int | None = None,
        not_two: bool = False,
        mode: Literal["default", "generalizability"] = "default",
        estimator: ClusterMixin | None = None,
    ) -> np.ndarray:
        """Return clustering labels from the selected consensus matrix.

        Parameters
        ----------
        measure : str, default="stability"
            Metric key used to select the best configuration. Common
            aliases: ``"stability"`` / ``"s"``, ``"generalizability"`` /
            ``"g"``, ``"average"`` / ``"avg"``, ``"pac"``, ``"gini"``,
            ``"ce"``, ``"accuracy"``.
        rule : str, default="1se"
            Selection rule. ``"max"`` picks the configuration with the
            highest score. ``"1se"`` picks the largest *k* within one
            standard error of the best score. ``"quantile"`` picks the
            largest *k* within the best score's quantile bounds.
        k : int or None, default=None
            Optional fixed number of clusters to select. If None, uses the
            value selected by ``measure`` and ``rule``.
        not_two : bool, default=False
            If True, exclude k=2 configurations during selection. Ignored
            when ``k`` is explicitly provided.
        mode : Literal['default', 'generalizability'], default='default'
            Determines which consensus matrix is used to return labels.
        estimator : ClusterMixin or None, default=None
            If provided, uses this estimator to cluster the consensus
            distance matrix; otherwise defaults to average-linkage
            ``AgglomerativeClustering`` with precomputed distances.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Clustering labels derived from the selected consensus matrix.

        Raises
        ------
        RuntimeError
            If the instance has not been fitted yet or if the required
            consensus matrix is not available.
        ValueError
            If no configurations match the given *k*.
        """
        policy = resolve_mode(mode)

        if (
            (self.consensus_matrices_ is None and policy.run_stability)
            or (
                self.consensus_generalizability_matrices_ is None
                and policy.run_generalizability
            )
            or self.estimator_results_ is None
        ):
            raise RuntimeError("Call fit() first.")

        df = self.estimator_results_

        # --- Select best configuration ---
        if k is None:
            row = select_best_row_by_rule(
                df, measure=measure, rule=rule, not_two=not_two
            )
            k = int(row["n_clusters"])
            best_idx = int(row.name)

        else:
            df_k = df[df["n_clusters"] == k]

            if df_k.empty:
                raise ValueError(f"No configurations found for k={k}.")

            row = select_best_row_by_rule(df_k, measure=measure, rule=rule)
            best_idx = int(row.name)

        # --- Retrieve the consensus matrix ---
        if policy.run_stability and self.consensus_matrices_ is not None:
            M_raw = self.consensus_matrices_[best_idx]
        elif (
            policy.run_generalizability
            and self.consensus_generalizability_matrices_ is not None
        ):
            M_raw = self.consensus_generalizability_matrices_[best_idx]
        else:
            raise ValueError("Mode must be 'default' or 'generalizability'.")

        if M_raw is None:
            raise RuntimeError(
                f"Consensus matrix not available for mode={mode!r}."
                "This run likely used split-mode and skipped building that artifact."
            )

        M = np.asarray(M_raw, dtype=float)

        # Symmetrize and clean up
        S = 0.5 * (M + M.T)
        np.fill_diagonal(S, 1.0)
        S = np.clip(S, 0.0, 1.0)

        if np.isnan(S).any():
            S = np.nan_to_num(S, nan=0.5)
            np.fill_diagonal(S, 1.0)

        # Convert similarity to distance
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)

        if estimator is None:
            estimator = AgglomerativeClustering(
                n_clusters=k,
                linkage="average",
                metric="precomputed",
            )

        labels = estimator.fit_predict(D)

        # --- Align with reference labels if available ---
        cur_k = int(np.unique(labels).size)
        ref = self.reference_labels
        ref_k = int(np.unique(ref).size) if ref is not None else None

        if (ref is None) or (ref_k != cur_k):
            self.reference_labels = labels
        else:
            labels = align_cluster_labels(ref, labels)

        return np.asarray(labels, dtype=np.int32)

    def get_k(
        self,
        *,
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
    ) -> int:
        """Return the best number of clusters.

        Parameters
        ----------
        measure : str, default="stability"
            Metric key used to select the best configuration. Common
            aliases: ``"stability"`` / ``"s"``, ``"generalizability"`` /
            ``"g"``, ``"average"`` / ``"avg"``, ``"pac"``, ``"gini"``,
            ``"ce"``, ``"accuracy"``.
        rule : str, default="1se"
            Selection rule. ``"max"`` picks the configuration with the
            highest score. ``"1se"`` picks the largest *k* within one
            standard error of the best score. ``"quantile"`` picks the
            largest *k* within the best score's quantile bounds.
        not_two : bool, default=False
            If True, exclude k=2 configurations during selection.

        Returns
        -------
        k : int
            Selected number of clusters.

        Raises
        ------
        RuntimeError
            If the instance has not been fitted yet.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        return select_best_k(
            self.estimator_results_, measure=measure, rule=rule, not_two=not_two
        )

    def get_estimator(
        self,
        *,
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
    ) -> ClusterMixin:
        """Return the best estimator.

        Parameters
        ----------
        measure : str, default="stability"
            Metric key used to select the best configuration. Common
            aliases: ``"stability"`` / ``"s"``, ``"generalizability"`` /
            ``"g"``, ``"average"`` / ``"avg"``, ``"pac"``, ``"gini"``,
            ``"ce"``, ``"accuracy"``.
        rule : str, default="1se"
            Selection rule. ``"max"`` picks the configuration with the
            highest score. ``"1se"`` picks the largest *k* within one
            standard error of the best score. ``"quantile"`` picks the
            largest *k* within the best score's quantile bounds.
        not_two : bool, default=False
            If True, exclude k=2 configurations during selection.

        Returns
        -------
        estimator : ClusterMixin
            Instantiated estimator with parameters from the best row.

        Raises
        ------
        RuntimeError
            If the instance has not been fitted yet.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        return select_best_estimator(
            self.estimator_results_,
            self.estimator_param_grids_,
            measure=measure,
            rule=rule,
            not_two=not_two,
        )

    # ------------------------------------------------------------------ #
    #  Plotting                                                          #
    # ------------------------------------------------------------------ #

    def plot_metric_over_n_clusters(
        self,
        *,
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
        ax=None,
        figsize: tuple | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        palette: str = "Accent",
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
        **kwargs,
    ):
        """Plot clustering validation metrics across cluster numbers.

        Creates a line plot showing one line for each unique estimator
        configuration (estimator name + hyperparameters, excluding n_clusters).
        Error bars represent +/-1 standard error. A vertical dashed line
        indicates the selected k according to the specified rule.

        Parameters
        ----------
        measure : str, default="stability"
            Metric to plot. Options include: "stability", "ari_stability",
            "generalizability", "ari_generalizability", "average", "ari_average",
            "pac", "consensus_pac_stability", "gini", "consensus_gini_stability",
            "ce", "consensus_ce_stability", "accuracy", etc.
        rule : str, default="1se"
            Selection rule for choosing the best k. Options: "max", "1se",
            "quantile".
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (9, 5.5).
        title : str, optional
            Figure title.
        xlabel : str, optional
            X-axis label. Default is "Number of Clusters (k)".
        ylabel : str, optional
            Y-axis label. If None, auto-generated from metric name.
        legend : bool, default=True
            Whether to display a legend showing estimator labels.
        legend_loc : str, default="best"
            Legend location (passed to matplotlib's ax.legend).
        palette : str, default="Accent"
            Matplotlib colormap name for line colors. Default is "Accent".
        show : bool, default=False
            Whether to call plt.show() before returning.
        save : str or Path, optional
            Path to save the figure. If provided, the figure is saved and
            None is returned instead of an Axes object.
        dpi : int, default=300
            Dots per inch for saved figures.
        **kwargs
            Additional keyword arguments passed to matplotlib's errorbar
            function (e.g., linewidth, marker, alpha).

        Returns
        -------
        ax : matplotlib.axes.Axes or None
            The Axes object, or None if save was used.

        Raises
        ------
        RuntimeError
            If the instance has not been fitted yet.
        ValueError
            If measure is not found in the results.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> carve = CARVE().fit(X)
        >>> ax = carve.plot_metric_over_n_clusters(measure="stability", rule="1se")
        >>> plt.show()

        >>> carve.plot_metric_over_n_clusters(measure="generalizability", save="gen.pdf")
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        return _plot_metric_over_n_clusters(
            self.estimator_results_,
            measure=measure,
            rule=rule,
            not_two=not_two,
            ax=ax,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
            legend_loc=legend_loc,
            palette=palette,
            show=show,
            save=save,
            dpi=dpi,
            **kwargs,
        )

    def plot_consensus_matrix(
        self,
        *,
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
        mode: Literal["default", "stability", "generalizability"] = "default",
        k: int | None = None,
        ax=None,
        figsize: tuple | None = None,
        cmap: str = "viridis",
        palette: str = "Accent",
        colorbar: bool = True,
        colorbar_label: str = "Consensus",
        title: str | None = None,
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
    ):
        """Plot the selected consensus matrix with a flush top cluster band.

        The best configuration is chosen according to ``measure`` and
        ``rule``.

        Parameters
        ----------
        measure : str, default="stability"
            Metric key used for model selection.
        rule : str, default="1se"
            Selection rule ("max", "1se", "quantile").
        mode : Literal["default", "stability", "generalizability"], default="default"
            Which consensus matrix family to plot.
        k : int, optional
            If given, restrict selection to this number of clusters.
        ax : matplotlib.axes.Axes, optional
            Axis for the heatmap; if None a new figure is created.
        figsize : tuple, optional
            Figure size in inches.
        cmap : str, default="viridis"
            Heatmap colormap.
        palette : str, default="Accent"
            Discrete palette used for the top cluster band.
        colorbar : bool, default=True
            Whether to draw the heatmap colorbar.
        colorbar_label : str, default="Consensus"
            Label for the heatmap colorbar.
        title : str, optional
            Plot title.
        show : bool, default=False
            Whether to call ``plt.show()`` before returning.
        save : str or Path, optional
            Path to save the figure.
        dpi : int, default=300
            Dots per inch for saved figures.

        Returns
        -------
        ax : matplotlib.axes.Axes or None
            Heatmap axis, or None if ``save`` is provided.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        if mode in ("default", "stability"):
            matrices = self.consensus_matrices_
            labels_mode = "default"
        elif mode == "generalizability":
            matrices = self.consensus_generalizability_matrices_
            labels_mode = "generalizability"
        else:
            raise ValueError(
                "mode must be one of: 'default', 'stability', 'generalizability'."
            )

        if matrices is None:
            raise RuntimeError(
                f"Consensus matrices for mode={mode!r} are not available."
            )

        df = self.estimator_results_
        if k is None:
            row = select_best_row_by_rule(
                df, measure=measure, rule=rule, not_two=not_two
            )
        else:
            df_k = df[df["n_clusters"] == k]
            if df_k.empty:
                raise ValueError(f"No configurations found for k={k}.")
            row = select_best_row_by_rule(df_k, measure=measure, rule=rule)

        best_idx = int(row.name)
        selected_k = int(row["n_clusters"])
        matrix = matrices[best_idx]
        if matrix is None:
            raise RuntimeError(
                f"Selected consensus matrix is not available for mode={mode!r}."
            )

        labels = self.get_labels(
            measure=measure,
            rule=rule,
            k=selected_k,
            mode=labels_mode,
        )

        return _plot_consensus_matrix(
            matrix,
            labels,
            ax=ax,
            figsize=figsize,
            cmap=cmap,
            palette=palette,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            title=title,
            show=show,
            save=save,
            dpi=dpi,
        )

    def plot_cluster_boxplot(
        self,
        *,
        source: Literal["gini", "ce", "accuracy"] = "gini",
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
        mode: Literal["default", "stability", "generalizability"] = "default",
        k: int | None = None,
        ax=None,
        figsize: tuple | None = None,
        order: list[int | str] | None = None,
        palette: str = "Accent",
        showfliers: bool = False,
        width: float = 0.75,
        title: str | None = None,
        xlabel: str = "Cluster",
        ylabel: str | None = None,
        annotation: bool | str = True,
        rotation: float | None = None,
        ylim: tuple[float, float] = (-0.02, 1.02),
        fit_ylim: bool = True,
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
    ):
        """Plot cluster-level uncertainty as a boxplot.

        The best configuration is chosen via ``measure`` and ``rule``,
        consistent with other plotting methods.

        Parameters
        ----------
        source : {"gini", "ce", "accuracy"}, default="gini"
            Score source for per-sample values.
        measure : str, default="stability"
            Metric key used for model selection.
        rule : str, default="1se"
            Selection rule.
        mode : Literal["default", "stability", "generalizability"], default="default"
            Consensus matrix mode for label extraction.
        k : int, optional
            If given, restrict selection to this number of clusters.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        figsize : tuple, optional
            Figure size in inches.
        order : list of int or str, optional
            Explicit cluster ordering for the x-axis.
        palette : str, default="Accent"
            Discrete colormap for box colors.
        showfliers : bool, default=False
            Whether to show outlier points.
        width : float, default=0.75
            Box width.
        title : str, optional
            Figure title.
        xlabel : str, default="Cluster"
            X-axis label.
        ylabel : str, optional
            Y-axis label. If None, auto-generated from source.
        annotation : bool or str, optional
            Text for adaptive annotation.
        rotation : float, optional
            Tick label rotation angle.
        ylim : tuple, default=(-0.02, 1.02)
            Y-axis limits. Default is slightly beyond [0, 1] for stability scores.
        fit_ylim : bool, default=True
            Whether to automatically fit y-limits to the data range.
        show : bool, default=False
            Whether to call plt.show() before returning.
        save : str or Path, optional
            Path to save the figure.
        dpi : int, default=300
            Dots per inch for saved figures.

        Returns
        -------
        ax : matplotlib.axes.Axes or None
            The Axes object, or None if ``save`` is provided.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        df = self.estimator_results_
        if k is None:
            row = select_best_row_by_rule(
                df, measure=measure, rule=rule, not_two=not_two
            )
        else:
            df_k = df[df["n_clusters"] == k]
            if df_k.empty:
                raise ValueError(f"No configurations found for k={k}.")
            row = select_best_row_by_rule(df_k, measure=measure, rule=rule)

        best_idx = int(row.name)
        selected_k = int(row["n_clusters"])

        # --- Resolve score source ---
        if source == "gini":
            if self.stability_gini_scores_ is None:
                raise RuntimeError(
                    "Gini stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_gini_scores_[best_idx], dtype=float)
            default_ylabel = "Cluster Stability (Gini)"

        elif source == "ce":
            if self.stability_ce_scores_ is None:
                raise RuntimeError(
                    "CE stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_ce_scores_[best_idx], dtype=float)
            default_ylabel = "Cluster Stability (CE)"

        elif source == "accuracy":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[best_idx], dtype=float)
            default_ylabel = "Cluster Generalizability"

        else:
            raise ValueError("source must be one of: 'accuracy', 'gini', 'ce'.")

        # --- Resolve labels mode and get labels ---
        labels_mode: Literal["default", "generalizability"]
        if mode in ("default", "stability"):
            labels_mode = "default"
        elif mode == "generalizability":
            labels_mode = "generalizability"
        else:
            raise ValueError(
                "mode must be one of: 'default', 'stability', 'generalizability'."
            )

        labels = self.get_labels(
            measure=measure,
            rule=rule,
            k=selected_k,
            mode=labels_mode,
        )

        if ylabel is None:
            ylabel = default_ylabel

        # --- Build annotation ---
        if annotation is True:
            annotation_text = _get_annotation(
                measure=measure,
                rule=rule,
                k=k,
                estimator_results=df,
                row=row,
                selected_k=selected_k,
            )

        elif isinstance(annotation, str):
            annotation_text = annotation
        else:
            annotation_text = None

        # --- Plot ---
        return _plot_cluster_boxplot(
            scores,
            labels,
            ax=ax,
            figsize=figsize,
            order=order,
            palette=palette,
            showfliers=showfliers,
            width=width,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            annotation=annotation_text,
            rotation=rotation,
            ylim=ylim,
            fit_ylim=fit_ylim,
            show=show,
            save=save,
            dpi=dpi,
        )

    def plot_cluster_violin(
        self,
        *,
        source: Literal["gini", "ce", "accuracy"] = "gini",
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
        mode: Literal["default", "stability", "generalizability"] = "default",
        k: int | None = None,
        ax=None,
        figsize: tuple | None = None,
        order: list[int | str] | None = None,
        palette: str = "Accent",
        density_norm: Literal["width", "area", "count"] = "width",
        stripplot: bool = True,
        jitter: bool | float = True,
        size: float = 8.0,
        alpha: float = 0.22,
        inner: Literal["box", "quartile", "none"] = "box",
        title: str | None = None,
        xlabel: str = "Cluster",
        ylabel: str | None = None,
        annotation: bool | str = True,
        rotation: float | None = None,
        ylim: tuple[float, float] = (-0.02, 1.02),
        fit_ylim: bool = True,
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
    ):
        """Plot cluster-level uncertainty as a violin plot.

        The API mirrors common scanpy arguments (``stripplot``, ``jitter``,
        ``density_norm``, ``show``, ``ax``, ``save``). The best
        configuration is chosen via ``measure`` and ``rule``, consistent
        with other plotting methods.

        Parameters
        ----------
        source : {"gini", "ce", "accuracy"}, default="gini"
            Score source for per-sample values.
        measure : str, default="stability"
            Metric key used for model selection.
        rule : str, default="1se"
            Selection rule.
        mode : Literal["default", "stability", "generalizability"], default="default"
            Consensus matrix mode for label extraction.
        k : int, optional
            If given, restrict selection to this number of clusters.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        figsize : tuple, optional
            Figure size in inches.
        order : list of int or str, optional
            Explicit cluster ordering for the x-axis.
        palette : str, default="Accent"
            Discrete colormap for violin colors.
        density_norm : {"width", "area", "count"}, default="width"
            How to normalize violin widths.
        stripplot : bool, default=True
            Whether to overlay individual data points.
        jitter : bool or float, default=True
            Jitter width for the strip plot.
        size : float, default=8.0
            Marker size for strip plot points.
        alpha : float, default=0.22
            Marker alpha for strip plot points.
        inner : {"box", "quartile", "none"}, default="box"
            Inner annotation style.
        title : str, optional
            Figure title.
        xlabel : str, default="Cluster"
            X-axis label.
        ylabel : str, optional
            Y-axis label. If None, auto-generated from source.
        annotation : bool or str, optional
            Text for adaptive annotation.
        rotation : float, optional
            Tick label rotation angle.
        ylim : tuple, default=(-0.02, 1.02)
            Y-axis limits. Default is slightly beyond [0, 1] for stability scores.
        fit_ylim : bool, default=True
            Whether to automatically fit y-limits to the data range.
        show : bool, default=False
            Whether to call plt.show() before returning.
        save : str or Path, optional
            Path to save the figure.
        dpi : int, default=300
            Dots per inch for saved figures.

        Returns
        -------
        ax : matplotlib.axes.Axes or None
            The Axes object, or None if ``save`` is provided.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        df = self.estimator_results_
        if k is None:
            row = select_best_row_by_rule(
                df, measure=measure, rule=rule, not_two=not_two
            )
        else:
            df_k = df[df["n_clusters"] == k]
            if df_k.empty:
                raise ValueError(f"No configurations found for k={k}.")
            row = select_best_row_by_rule(df_k, measure=measure, rule=rule)

        best_idx = int(row.name)
        selected_k = int(row["n_clusters"])

        # --- Resolve score source ---
        if source == "gini":
            if self.stability_gini_scores_ is None:
                raise RuntimeError(
                    "Gini stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_gini_scores_[best_idx], dtype=float)
            default_ylabel = "Cluster Stability (Gini)"

        elif source == "ce":
            if self.stability_ce_scores_ is None:
                raise RuntimeError(
                    "CE stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_ce_scores_[best_idx], dtype=float)
            default_ylabel = "Cluster Stability (CE)"

        elif source == "accuracy":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[best_idx], dtype=float)
            default_ylabel = "Cluster Generalizability"

        else:
            raise ValueError("source must be one of: 'accuracy', 'gini', 'ce'.")

        # --- Resolve labels mode and get labels ---
        labels_mode: Literal["default", "generalizability"]
        if mode in ("default", "stability"):
            labels_mode = "default"
        elif mode == "generalizability":
            labels_mode = "generalizability"
        else:
            raise ValueError(
                "mode must be one of: 'default', 'stability', 'generalizability'."
            )

        labels = self.get_labels(
            measure=measure,
            rule=rule,
            k=selected_k,
            mode=labels_mode,
        )

        if ylabel is None:
            ylabel = default_ylabel

        # --- Build annotation ---
        if annotation is True:
            annotation_text = _get_annotation(
                measure=measure,
                rule=rule,
                k=k,
                estimator_results=df,
                row=row,
                selected_k=selected_k,
            )

        elif isinstance(annotation, str):
            annotation_text = annotation
        else:
            annotation_text = None

        # --- Plot ---
        return _plot_cluster_violin(
            scores,
            labels,
            ax=ax,
            figsize=figsize,
            order=order,
            palette=palette,
            density_norm=density_norm,
            stripplot=stripplot,
            jitter=jitter,
            size=size,
            alpha=alpha,
            inner=inner,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            annotation=annotation_text,
            rotation=rotation,
            ylim=ylim,
            fit_ylim=fit_ylim,
            show=show,
            save=save,
            dpi=dpi,
        )

    def plot_cluster_scatter(
        self,
        *,
        source: Literal["gini", "ce", "accuracy"] = "gini",
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
        mode: Literal["default", "stability", "generalizability"] = "default",
        k: int | None = None,
        X: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
        ax=None,
        figsize: tuple | None = None,
        palette: str = "Accent",
        alpha_range: tuple[float, float] = (0.45, 0.9),
        size_range: tuple[float, float] = (15.0, 60.0),
        sort_order: bool = True,
        legend: bool = True,
        legend_loc: str = "right margin",
        annotation: bool | str = True,
        annotation_style: Literal["legend", "box"] = "legend",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        show_ticks: bool = False,
        frameon: bool = False,
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
    ):
        """Plot data in 2D with score-encoded opacity and point size.

        The best configuration is chosen via ``measure`` and ``rule``,
        consistent with other plotting methods.

        Visual encoding:
        - cluster-level mean score -> opacity (alpha)
        - sample-level score -> marker size

        Parameters
        ----------
        source : {"gini", "ce", "accuracy"}, default="gini"
            Score source for per-sample values.
        measure : str, default="stability"
            Metric key used for model selection.
        rule : str, default="1se"
            Selection rule.
        mode : Literal["default", "stability", "generalizability"], default="default"
            Consensus matrix mode for label extraction.
        k : int, optional
            If given, restrict selection to this number of clusters.
        X : ndarray, optional
            Data array to use. If None, uses ``self.X_``.
        embedding : ndarray of shape (n_samples, 2), optional
            Pre-computed 2D embedding.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        figsize : tuple, optional
            Figure size in inches.
        palette : str, default="Accent"
            Colormap for cluster colors.
        alpha_range : tuple of float, default=(0.45, 0.9)
            ``(alpha_high_score, alpha_low_score)``.  Stable samples
            get the first value (faint); unstable samples the second (opaque).
        size_range : tuple of float, default=(15.0, 60.0)
            ``(size_high_score, size_low_score)``.  Stable samples get
            the first value (small); unstable samples get the second (large).
        sort_order : bool, default=True
            Whether to sort points by alpha so transparent points are drawn first.
        legend : bool, default=True
            Whether to display a legend.
        legend_loc : str, default="right margin"
            Legend location.
        annotation : bool or str, default=True
            Annotation text. ``True`` auto-generates from the selected
            model/measure/rule.  A string is used verbatim.
            ``False`` disables the annotation.
        annotation_style : {"legend", "box"}, default="legend"
            ``"legend"`` appends the annotation to the cluster legend.
            ``"box"`` places a free-floating annotation box.
        title : str, optional
            Figure title.
        xlabel : str, optional
            X-axis label. Default is "Component 1".
        ylabel : str, optional
            Y-axis label. Default is "Component 2".
        show_ticks : bool, default=False
            Whether to show axis ticks.
        frameon : bool, default=False
            Whether to draw axis spines.
        show : bool, default=False
            Whether to call plt.show() before returning.
        save : str or Path, optional
            Path to save the figure.
        dpi : int, default=300
            Dots per inch for saved figures.

        Returns
        -------
        ax : matplotlib.axes.Axes or None
            The Axes object, or None if ``save`` is provided.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        df = self.estimator_results_
        if k is None:
            row = select_best_row_by_rule(
                df, measure=measure, rule=rule, not_two=not_two
            )
        else:
            df_k = df[df["n_clusters"] == k]
            if df_k.empty:
                raise ValueError(f"No configurations found for k={k}.")
            row = select_best_row_by_rule(df_k, measure=measure, rule=rule)

        best_idx = int(row.name)
        selected_k = int(row["n_clusters"])

        # --- Resolve score source ---
        if source == "gini":
            if self.stability_gini_scores_ is None:
                raise RuntimeError(
                    "Gini stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_gini_scores_[best_idx], dtype=float)
            scores_name = "Gini Stability"
        elif source == "ce":
            if self.stability_ce_scores_ is None:
                raise RuntimeError(
                    "CE stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_ce_scores_[best_idx], dtype=float)
            scores_name = "CE Stability"
        elif source == "accuracy":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[best_idx], dtype=float)
            scores_name = "Generalizability"
        else:
            raise ValueError("source must be one of: 'accuracy', 'gini', 'ce'.")

        # --- Resolve labels mode ---
        labels_mode: Literal["default", "generalizability"]
        if mode in ("default", "stability"):
            labels_mode = "default"
        elif mode == "generalizability":
            labels_mode = "generalizability"
        else:
            raise ValueError(
                "mode must be one of: 'default', 'stability', 'generalizability'."
            )

        labels = self.get_labels(
            measure=measure,
            rule=rule,
            k=selected_k,
            mode=labels_mode,
        )

        data = self.X_ if X is None else ensure_2d_array(X)
        if data is None:
            raise RuntimeError(
                "Raw data are not available on this instance. "
                "Pass X=... explicitly (e.g., after loading a model saved with include_data=False)."
            )

        # --- Build annotation ---
        tight_layout = annotation_style == "legend"
        if annotation is True:
            annotation_text = _get_annotation(
                measure=measure,
                rule=rule,
                k=k,
                estimator_results=df,
                row=row,
                selected_k=selected_k,
                tight_layout=tight_layout,
            )
        elif isinstance(annotation, str):
            annotation_text = annotation
        else:
            annotation_text = None

        if xlabel is None:
            xlabel = "Component 1"
        if ylabel is None:
            ylabel = "Component 2"

        return _plot_cluster_scatter(
            data,
            labels,
            scores,
            embedding=embedding,
            ax=ax,
            figsize=figsize,
            palette=palette,
            alpha_range=alpha_range,
            size_range=size_range,
            sort_order=sort_order,
            legend=legend,
            legend_loc=legend_loc,
            annotation=annotation_text,
            annotation_style=annotation_style,
            title=title,
            scores_name=scores_name,
            xlabel=xlabel,
            ylabel=ylabel,
            show_ticks=show_ticks,
            frameon=frameon,
            show=show,
            save=save,
            dpi=dpi,
        )

    def plot_diagnostic_scatter(
        self,
        *,
        source: Literal["gini", "ce", "accuracy"] = "gini",
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
        mode: Literal["default", "stability", "generalizability"] = "default",
        k: int | None = None,
        X: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
        ax=None,
        figsize: tuple | None = None,
        cmap: str = "Greens_r",
        alpha_encoding: bool = True,
        alpha_range: tuple[float, float] = (0.3, 1.0),
        marker_size: float = 30.0,
        marker_linewidth: float = 0.2,
        markers: list[str] | None = None,
        sort_order: bool = True,
        legend: bool = True,
        legend_loc: str = "right margin",
        colorbar: bool = True,
        colorbar_label: str | None = None,
        annotation: bool | str = True,
        annotation_style: Literal["legend", "box"] = "legend",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        show_ticks: bool = False,
        frameon: bool = False,
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
    ):
        """Diagnostic scatter plot with shape-per-cluster and color-per-score.

        Cluster membership is encoded via marker shapes, while per-sample
        scores are mapped to a sequential colormap.  Unstable samples (low
        scores) are visually prominent; stable samples fade into the
        background.

        The best configuration is chosen via ``measure`` and ``rule``,
        consistent with other plotting methods.

        Parameters
        ----------
        source : {"gini", "ce", "accuracy"}, default="gini"
            Score source for per-sample values.
        measure : str, default="stability"
            Metric key used for model selection.
        rule : str, default="1se"
            Selection rule.
        mode : Literal["default", "stability", "generalizability"], default="default"
            Consensus matrix mode for label extraction.
        k : int, optional
            If given, restrict selection to this number of clusters.
        X : ndarray, optional
            Data array to use. If None, uses ``self.X_``.
        embedding : ndarray of shape (n_samples, 2), optional
            Pre-computed 2D embedding.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        figsize : tuple, optional
            Figure size in inches.
        cmap : str, default="Greens_r"
            Sequential matplotlib colormap.
        alpha_encoding : bool, default=True
            Whether to also vary transparency with score.
        alpha_range : tuple of float, default=(0.3, 1.0)
            ``(alpha_high_score, alpha_low_score)``.
        marker_size : float, default=30.0
            Fixed marker size for all points.
        marker_linewidth : float, default=0.2
            Line width for marker edges.
        markers : list of str, optional
            Marker codes for each cluster.
        sort_order : bool, default=True
            Draw stable points first so unstable points render on top.
        legend : bool, default=True
            Whether to display a shape legend for clusters.
        legend_loc : str, default="right margin"
            Legend location.
        colorbar : bool, default=True
            Whether to draw a colorbar for the score encoding.
        colorbar_label : str, optional
            Label for the colorbar.
        annotation : bool or str, default=True
            ``True`` auto-generates from the selected model/measure/rule.
            A string is used verbatim. ``False`` disables.
        annotation_style : {"legend", "box"}, default="legend"
            How to display the annotation.
        title : str, optional
            Figure title.
        xlabel : str, optional
            X-axis label. Default is "Component 1".
        ylabel : str, optional
            Y-axis label. Default is "Component 2".
        show_ticks : bool, default=False
            Whether to show axis ticks.
        frameon : bool, default=False
            Whether to draw axis spines.
        show : bool, default=False
            Whether to call plt.show() before returning.
        save : str or Path, optional
            Path to save the figure.
        dpi : int, default=300
            Dots per inch for saved figures.

        Returns
        -------
        ax : matplotlib.axes.Axes or None
            The Axes object, or None if ``save`` is provided.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        df = self.estimator_results_
        if k is None:
            row = select_best_row_by_rule(
                df, measure=measure, rule=rule, not_two=not_two
            )
        else:
            df_k = df[df["n_clusters"] == k]
            if df_k.empty:
                raise ValueError(f"No configurations found for k={k}.")
            row = select_best_row_by_rule(df_k, measure=measure, rule=rule)

        best_idx = int(row.name)
        selected_k = int(row["n_clusters"])

        # --- Resolve score source ---
        if source == "gini":
            if self.stability_gini_scores_ is None:
                raise RuntimeError(
                    "Gini stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_gini_scores_[best_idx], dtype=float)
            scores_name = "Gini Stability"
        elif source == "ce":
            if self.stability_ce_scores_ is None:
                raise RuntimeError(
                    "CE stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_ce_scores_[best_idx], dtype=float)
            scores_name = "CE Stability"
        elif source == "accuracy":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[best_idx], dtype=float)
            scores_name = "Generalizability"
        else:
            raise ValueError("source must be one of: 'accuracy', 'gini', 'ce'.")

        # --- Resolve labels mode ---
        labels_mode: Literal["default", "generalizability"]
        if mode in ("default", "stability"):
            labels_mode = "default"
        elif mode == "generalizability":
            labels_mode = "generalizability"
        else:
            raise ValueError(
                "mode must be one of: 'default', 'stability', 'generalizability'."
            )

        labels = self.get_labels(
            measure=measure,
            rule=rule,
            k=selected_k,
            mode=labels_mode,
        )

        data = self.X_ if X is None else ensure_2d_array(X)
        if data is None:
            raise RuntimeError(
                "Raw data are not available on this instance. "
                "Pass X=... explicitly (e.g., after loading a model saved with include_data=False)."
            )

        # --- Build annotation ---
        tight_layout = annotation_style == "legend"
        if annotation is True:
            annotation_text = _get_annotation(
                measure=measure,
                rule=rule,
                k=k,
                estimator_results=df,
                row=row,
                selected_k=selected_k,
                tight_layout=tight_layout,
            )
        elif isinstance(annotation, str):
            annotation_text = annotation
        else:
            annotation_text = None

        if xlabel is None:
            xlabel = "Component 1"
        if ylabel is None:
            ylabel = "Component 2"

        return _plot_diagnostic_scatter(
            data,
            labels,
            scores,
            embedding=embedding,
            ax=ax,
            figsize=figsize,
            cmap=cmap,
            alpha_encoding=alpha_encoding,
            alpha_range=alpha_range,
            marker_size=marker_size,
            marker_linewidth=marker_linewidth,
            markers=markers,
            sort_order=sort_order,
            legend=legend,
            legend_loc=legend_loc,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            annotation=annotation_text,
            annotation_style=annotation_style,
            title=title,
            scores_name=scores_name,
            xlabel=xlabel,
            ylabel=ylabel,
            show_ticks=show_ticks,
            frameon=frameon,
            show=show,
            save=save,
            dpi=dpi,
        )

    # ------------------------------------------------------------------ #
    #  Persistence                                                       #
    # ------------------------------------------------------------------ #

    def save(
        self,
        path: str | Path,
        *,
        include_data: bool = False,
        compress: int = 3,
    ) -> None:
        """Save a fitted CARVE instance to disk.

        Parameters
        ----------
        path : str or Path
            Destination file path. The recommended extension is ``.carve``.
        include_data : bool, default=False
            If True, the input array ``X_`` is included in the file.
            When False (default), ``X_`` is excluded to reduce file size;
            methods that need the raw data will require that ``X`` is
            re-supplied after loading.
        compress : int, default=3
            Compression level passed to :func:`joblib.dump` (0-9, where
            0 disables compression and 9 is maximum).

        Raises
        ------
        RuntimeError
            If the instance has not been fitted yet.

        Examples
        --------
        >>> carve = CARVE().fit(X)
        >>> carve.save("results.carve")
        >>> loaded = CARVE.load("results.carve")
        """
        if self.estimator_results_ is None:
            raise RuntimeError(
                "This CARVE instance has not been fitted yet. "
                "Call .fit(X) before saving."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if include_data:
            joblib.dump(self, path, compress=compress)
        else:
            X_backup = self.X_
            self.X_ = None
            try:
                joblib.dump(self, path, compress=compress)
            finally:
                self.X_ = X_backup

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "CARVE":
        """Load a previously saved CARVE instance from disk.

        Parameters
        ----------
        path : str or Path
            Path to the saved ``.carve`` file.

        Returns
        -------
        instance : CARVE
            The deserialized, fitted CARVE instance.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        TypeError
            If the loaded object is not a ``CARVE`` instance.

        Examples
        --------
        >>> loaded = CARVE.load("results.carve")
        >>> loaded.get_labels()
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: '{path}'")

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected a CARVE instance, got {type(obj).__name__!r}.")
        return obj
