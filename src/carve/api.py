"""Public CARVE API."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering

from ._types import GridSpec, PreprocOption, RunMode, resolve_mode
from ._output import _print_run_footer, _print_run_header
from ._runner import run_validation
from ._consensus import compute_consensus_metrics
from ._selection import select_best_estimator, select_best_k, select_best_row_by_rule

from ._plotting import (
    plot_metric_over_n_clusters as _plot_metric_over_n_clusters,
    plot_consensus_matrix as _plot_consensus_matrix,
    plot_cluster_boxplot as _plot_cluster_boxplot,
    plot_cluster_violin as _plot_cluster_violin,
    plot_cluster_scatter as _plot_cluster_scatter,
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
    _coerce_n_clusters
)


@dataclass
class CARVE(BaseEstimator):
    """CARVE validator.

    Parameters
    ----------
    n_clusters : int or np.ndarray, default=10
        Number(s) of clusters to evaluate.
    n_resamples : int, default=100
        Number of resampling iterations.
    subsample_ratio : float, default=0.8
        Subsampling proportion.
    estimator_param_grids : list of (Estimator, param_grid) tuples, optional
        Clustering estimators and their parameter grids.
    normalization_options : list of preprocessing specs, optional
        Normalization preprocessing options.
    dim_reduction_options : list of dimensionality reduction specs, optional
        Dimensionality reduction preprocessing options.
    reference_labels : array-like of shape (n_samples,), optional
        Reference labels for consistent plots.
    n_jobs : int, default=1
        Parallelism.
    random_state : int, optional
        RNG seed.
    verbose : int, default=1
        Verbosity level for console output.
    """

    # --- Constructor parameters ---
    n_clusters: int | np.ndarray = field(default_factory=lambda: np.arange(2, 10 + 1, dtype=int))
    n_resamples: int = 100
    subsample_ratio: float = 0.8
    
    estimator_param_grids: list[GridSpec] | None = None
    normalization_options: list[PreprocOption] | None = None
    dim_reduction_options: list[PreprocOption] | None = None
    
    X_: np.ndarray | None = field(init=False, default=None)
    reference_labels: np.ndarray | None = None
    
    n_jobs: int = 1
    random_state: int | None = None
    verbose: int = 1

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

    # --- Global rates ---
    misclassification_rates_: np.ndarray | None = field(init=False, default=None)

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
            Whether to randomize preprocessing pipelines.
        show_progress : bool, default=False
            Show progress bar.
        mode : Literal['default', 'stability', 'generalizability'], default='default'
            Determines whether to run CARVE regularly ('default') or whether
            to only run stability analysis ('stability'),
            or generalizability analysis ('generalizability').
        random_state : int, optional
            Per-call RNG seed. If None, uses self.random_state.

        Returns
        -------
        self : CARVE
            Fitted instance.
        """
        policy = resolve_mode(mode)
        if policy.mode != "default":
            warnings.warn(
                "Non-default mode is experimental and may break downstream functionality.",
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
        if self.estimator_param_grids is None:  # Default estimator grids
            n_clusters_arr = _coerce_n_clusters(self.n_clusters)
            estimator_grids = default_estimator_grids(X, n_clusters_arr)
        
        else:  # User-provided estimator grids (verify consistency of n_clusters)
            estimator_grids = self.estimator_param_grids
            
            # Extract n_clusters from first grid
            n_clusters_arr = estimator_grids[0][1].get("n_clusters", None)
            
            # Verify all grids have the same n_clusters
            for _, grid in estimator_grids[1: ]:
                grid_n_clusters = grid.get("n_clusters", None)
                
                if not np.array_equal(n_clusters_arr, grid_n_clusters):
                    raise ValueError(
                        "All estimator parameter grids must contain the same n_clusters values."
                    )
            
        self.estimator_param_grids_ = estimator_grids
        
        # --- Resolve preprocessing options ---
        norm_options = self.normalization_options or default_normalization_options()
        dr_options = self.dim_reduction_options or default_dim_reduction_options(X, self.subsample_ratio)

        # --- Print run header ---
        _print_run_header(
            X=X,
            n_clusters=n_clusters_arr,
            n_resamples=self.n_resamples,
            subsample_ratio=self.subsample_ratio,
            estimator_grids=estimator_grids,
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
            estimator_grids=estimator_grids,
            n_resamples=self.n_resamples,
            subsample_ratio=self.subsample_ratio,
            normalization_options=norm_options,
            dim_reduction_options=dr_options,
            randomize_preprocessing=randomize_preprocessing,
            n_jobs=self.n_jobs,
            random_state=self.random_state if random_state is None else random_state,
            mode=mode,
            show_progress=show_progress,
        )

        self.estimator_results_ = pd.DataFrame.from_records(estimator_records)
        self.preprocessing_results_ = (
            None
            if not randomize_preprocessing
            else summarize_preprocessing_records(pipeline_records)
        )

        n_rows = int(self.estimator_results_.shape[0])

        # --- Stability-derived metrics ---
        if policy.run_stability and self.consensus_matrices_ is not None:
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
        else:
            self.stability_gini_scores_ = None
            self.stability_ce_scores_ = None
            self.estimator_results_["consensus_pac_stability"] = np.full(n_rows, np.nan)
            self.estimator_results_["consensus_gini_stability"] = np.full(
                n_rows, np.nan
            )
            self.estimator_results_["consensus_ce_stability"] = np.full(n_rows, np.nan)

        # --- Generalizability-derived metrics ---
        if policy.run_generalizability and self.generalizability_scores_ is not None:
            gen_arr = np.vstack(self.generalizability_scores_)
            self.misclassification_rates_ = np.clip(gen_arr, 0.0, 1.0)

            self.estimator_results_["misclassification_generalizability"] = (
                self.misclassification_rates_.mean(axis=1)
            )
        else:
            self.misclassification_rates_ = None
            self.estimator_results_["misclassification_generalizability"] = np.full(
                n_rows, np.nan
            )

        _print_run_footer(estimator_df=self.estimator_results_, verbose=self.verbose)

        return self

    def get_labels(
        self,
        *,
        measure: str = "stability",
        rule: str = "max",
        k: int | None = None,
        mode: Literal["default", "generalizability"] = "default",
        estimator: ClusterMixin | None = None,
    ) -> np.ndarray:
        """Return clustering labels from the selected consensus matrix.

        Parameters
        ----------
        measure : str, default="stability"
            Metric key used to select the best configuration.
        rule : str, default="max"
            Selection rule ("max", "1se", or "quantile").
        k : int or None, default=None
            Optional fixed number of clusters to select. If None, uses the
            value selected by ``measure`` and ``rule``.
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
        """
        policy = resolve_mode(mode)
        
        if (
            (self.consensus_matrices_ is None and policy.run_stability)
            or (self.consensus_generalizability_matrices_ is None and policy.run_generalizability)
            or self.estimator_results_ is None
        ):
            raise RuntimeError("Call fit() first.")

        df = self.estimator_results_

        # --- Select best configuration ---
        if k is None:
            row = select_best_row_by_rule(df, measure=measure, rule=rule)
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
        elif policy.run_generalizability and self.consensus_generalizability_matrices_ is not None:
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
        rule: str = "max",
    ) -> int:
        """Return the best number of clusters.

        Parameters
        ----------
        measure : str, default="stability"
            Metric key used to select the best configuration.
        rule : str, default="max"
            Selection rule ("max", "1se", or "quantile").

        Returns
        -------
        k : int
            Selected number of clusters.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        return select_best_k(self.estimator_results_, measure=measure, rule=rule)

    def get_estimator(
        self,
        *,
        measure: str = "stability",
        rule: str = "max",
    ) -> ClusterMixin:
        """Return the best estimator.

        Parameters
        ----------
        measure : str, default="stability"
            Metric key used to select the best configuration.
        rule : str, default="max"
            Selection rule ("max", "1se", or "quantile").

        Returns
        -------
        estimator : ClusterMixin
            Instantiated estimator with parameters from the best row.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        return select_best_estimator(
            self.estimator_results_,
            self.estimator_param_grids_,
            measure=measure,
            rule=rule,
        )

    # ------------------------------------------------------------------ #
    #  Plotting                                                          #
    # ------------------------------------------------------------------ #

    def plot_metric_over_n_clusters(
        self,
        *,
        measure: str = "stability",
        rule: str = "1se",
        ax=None,
        figsize: tuple | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        palette: str | None = None,
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
            "ce", "consensus_ce_stability", "misclassification", etc.
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
        palette : str, optional
            Matplotlib colormap name for line colors. Default is "tab10".
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
        mode: Literal["default", "stability", "generalizability"] = "default",
        k: int | None = None,
        ax=None,
        figsize: tuple | None = None,
        cmap: str = "viridis",
        palette: str = "tab20",
        colorbar: bool = True,
        colorbar_label: str = "Consensus",
        title: str | None = None,
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
    ):
        """Plot the selected consensus matrix with a flush top cluster band.

        Selection of the matrix is handled via ``_selection.py`` using
        ``measure`` and ``rule`` (defaults: stability + 1se).

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
        palette : str, default="tab20"
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
            row = select_best_row_by_rule(df, measure=measure, rule=rule)
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
        source: Literal["gini", "ce", "misclassification"] = "gini",
        measure: str = "stability",
        rule: str = "1se",
        mode: Literal["default", "stability", "generalizability"] = "default",
        k: int | None = None,
        ax=None,
        figsize: tuple | None = None,
        order: list[int | str] | None = None,
        palette: str = "tab20",
        showfliers: bool = False,
        width: float = 0.75,
        title: str | None = None,
        xlabel: str = "Cluster",
        ylabel: str | None = None,
        rotation: float | None = None,
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
    ):
        """Plot cluster-level uncertainty as a boxplot.

        The model row is selected with the same single source of truth
        used by ``plot_consensus_matrix``: ``select_best_row_by_rule``
        from ``_selection.py``.

        Parameters
        ----------
        source : {"gini", "ce", "misclassification"}, default="gini"
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
        palette : str, default="tab20"
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
        rotation : float, optional
            Tick label rotation angle.
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
            row = select_best_row_by_rule(df, measure=measure, rule=rule)
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

        elif source == "misclassification":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[best_idx], dtype=float)
            default_ylabel = "Cluster Generalizability"

        else:
            raise ValueError(
                "source must be one of: 'misclassification', 'gini', 'ce'."
            )

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

        if ylabel is None:
            ylabel = default_ylabel

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
            rotation=rotation,
            show=show,
            save=save,
            dpi=dpi,
        )

    def plot_cluster_violin(
        self,
        *,
        source: Literal["gini", "ce", "misclassification"] = "gini",
        measure: str = "stability",
        rule: str = "1se",
        mode: Literal["default", "stability", "generalizability"] = "default",
        k: int | None = None,
        ax=None,
        figsize: tuple | None = None,
        order: list[int | str] | None = None,
        palette: str = "tab20",
        density_norm: Literal["width", "area", "count"] = "width",
        stripplot: bool = True,
        jitter: bool | float = True,
        size: float = 8.0,
        alpha: float = 0.22,
        inner: Literal["box", "quartile", "none"] = "box",
        title: str | None = None,
        xlabel: str = "Cluster",
        ylabel: str | None = None,
        rotation: float | None = None,
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
    ):
        """Plot cluster-level uncertainty as a violin plot.

        The API mirrors common scanpy arguments (``stripplot``, ``jitter``,
        ``density_norm``, ``show``, ``ax``, ``save``) and uses the same
        model row selection path as ``plot_consensus_matrix``.

        Parameters
        ----------
        source : {"gini", "ce", "misclassification"}, default="gini"
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
        palette : str, default="tab20"
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
        rotation : float, optional
            Tick label rotation angle.
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
            row = select_best_row_by_rule(df, measure=measure, rule=rule)
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

        elif source == "misclassification":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[best_idx], dtype=float)
            default_ylabel = "Cluster Generalizability"

        else:
            raise ValueError(
                "source must be one of: 'misclassification', 'gini', 'ce'."
            )

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

        if ylabel is None:
            ylabel = default_ylabel

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
            rotation=rotation,
            show=show,
            save=save,
            dpi=dpi,
        )

    def plot_cluster_scatter(
        self,
        *,
        source: Literal["gini", "ce", "misclassification"] = "gini",
        measure: str = "stability",
        rule: str = "1se",
        mode: Literal["default", "stability", "generalizability"] = "default",
        k: int | None = None,
        X: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
        ax=None,
        figsize: tuple | None = None,
        palette: str = "tab20",
        alpha_range: tuple[float, float] | None = None,
        size_range: tuple[float, float] = (20.0, 100.0),
        sort_order: bool = True,
        legend: bool = True,
        legend_loc: str = "right margin",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        frameon: bool = False,
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
    ):
        """Plot data in 2D with score-encoded opacity and point size.

        Selection of the model row is identical to the consensus/box/violin
        plotting workflow via ``select_best_row_by_rule`` in ``_selection.py``.

        Visual encoding:
        - cluster-level mean score -> opacity (alpha)
        - sample-level score -> marker size

        Parameters
        ----------
        source : {"gini", "ce", "misclassification"}, default="gini"
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
        palette : str, default="tab20"
            Colormap for cluster colors.
        alpha_range : tuple of float, optional
            Min/max alpha for cluster-level score mapping.
        size_range : tuple of float, default=(20.0, 100.0)
            Min/max marker size for sample-level score mapping.
        sort_order : bool, default=True
            Whether to sort points by alpha so transparent points are drawn first.
        legend : bool, default=True
            Whether to display a legend.
        legend_loc : str, default="right margin"
            Legend location.
        title : str, optional
            Figure title.
        xlabel : str, optional
            X-axis label. Default is "Component 1".
        ylabel : str, optional
            Y-axis label. Default is "Component 2".
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
            row = select_best_row_by_rule(df, measure=measure, rule=rule)
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
        elif source == "ce":
            if self.stability_ce_scores_ is None:
                raise RuntimeError(
                    "CE stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_ce_scores_[best_idx], dtype=float)
        elif source == "misclassification":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[best_idx], dtype=float)
        else:
            raise ValueError(
                "source must be one of: 'misclassification', 'gini', 'ce'."
            )

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
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
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
