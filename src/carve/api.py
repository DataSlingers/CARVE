"""Public CARVE API."""

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier

from . import _anndata
from ._grids import (
    default_dim_reduction_options,
    default_estimator_grids,
    default_normalization_options,
)
from ._output import _print_run_footer, _print_run_header
from ._plotting import (
    _get_annotation,
)
from ._plotting import (
    plot_cluster_boxplot as _plot_cluster_boxplot,
)
from ._plotting import (
    plot_cluster_scatter as _plot_cluster_scatter,
)
from ._plotting import (
    plot_cluster_violin as _plot_cluster_violin,
)
from ._plotting import (
    plot_consensus_matrix as _plot_consensus_matrix,
)
from ._plotting import (
    plot_diagnostic_scatter as _plot_diagnostic_scatter,
)
from ._plotting import (
    plot_metric_over_n_clusters as _plot_metric_over_n_clusters,
)
from ._runner import run_validation
from ._selection import select_best_estimator, select_best_k, select_best_row_by_rule
from ._sweep import (
    SweepSpec,
    config_id_of,
    grid_sweep_values,
    infer_sweep_param,
    observed_k,
    resolve_sweep,
    sweep_param_name,
    validate_grids,
)
from ._types import GridSpec, NoisePolicy, PreprocOption, RunMode, resolve_mode
from ._utils import (
    align_cluster_labels,
    ensure_2d_array,
    resolve_anchors,
    summarize_preprocessing_records,
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
    resolution : float or ndarray, optional
        Resolution values for graph-community estimators (Leiden, Louvain).
        Supplying this switches the run to **resolution mode**: granularity
        is swept over ``resolution`` and the number of clusters becomes an
        observed outcome rather than an input. Mutually exclusive with a
        conflicting ``sweep``.
    sweep : str, optional
        Name of the hyperparameter to sweep. Defaults to ``"resolution"``
        when ``resolution`` is given, otherwise ``"n_clusters"``. When
        custom ``estimator_param_grids`` are supplied, CARVE infers this
        from the grids. A single run sweeps exactly one parameter;
        k-based and resolution-based estimators cannot be compared in the
        same run.
    sweep_values : array-like, optional
        Values for ``sweep`` when it is neither ``"n_clusters"`` nor
        ``"resolution"`` (e.g. HDBSCAN's ``min_cluster_size``).
    finer_is_larger : bool, optional
        Whether larger values of ``sweep`` yield more clusters. Inferred
        for known parameters (True for ``n_clusters`` and ``resolution``,
        False for ``min_cluster_size``); required for unknown ones.
    noise_policy : {"drop", "as_cluster", "singleton"}, default="drop"
        How to resolve the ``-1`` labels emitted by density-based methods
        such as HDBSCAN. ``"drop"`` treats noise points as un-sampled for
        that resample, so they contribute nothing to the consensus matrix,
        the ARI, or the classifier.
    n_resamples : int, default=100
        Number of resampling iterations per estimator configuration.
    subsample_ratio : float, default=0.618
        Fraction of samples drawn without replacement per resample.
        Must be in (0, 1).
    anchor_threshold : int, default=5000
        Runs with ``n_samples <= anchor_threshold`` build full consensus
        matrices exactly as before. Above it, consensus quantities are
        computed over a fixed anchor subset, because a dense n-by-n matrix
        per configuration is not feasible at large n. The comparison is
        inclusive.
    consensus_anchors : int, float, or None, default=None
        Number of anchors, or a fraction of ``n_samples`` as a float in
        (0, 1]. None resolves to ``min(n_samples, anchor_threshold)``, which
        keeps the effective anchor count continuous across the threshold.
        Lower it to reduce the memory the retained blocks occupy: each block
        is 4 * m**2 bytes and there are two per configuration.
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
    sweep_ : SweepSpec or None
        The resolved sweep axis used during fitting.
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
    consensus_anchors_ : ndarray or None
        Anchor indices used for this fit, or None when the exact path ran.

    Notes
    -----
    The validation pipeline proceeds as follows: for each estimator
    configuration (estimator type x hyperparameters x *k*), CARVE draws
    ``n_resamples`` random subsamples, clusters each subsample, and
    measures stability (intra-subsample ARI and consensus-matrix metrics)
    and generalizability (held-out prediction via a random forest).

    See Also
    --------
    SpectralClustering : Custom spectral clustering variant included
        in the default estimator grid.

    Examples
    --------
    >>> from carve import CARVE
    >>> carve = CARVE(n_clusters=10, n_resamples=80, subsample_ratio=0.6)
    >>> carve.fit(X)
    >>> labels = carve.get_labels(measure="stability", rule="1se")
    >>> k = carve.get_k(measure="generalizability", rule="1se")

    >>> import numpy as np
    >>> carve = CARVE(resolution=np.arange(0.2, 2.01, 0.2)).fit(X)
    >>> carve.get_sweep_value(), carve.get_k()
    """

    # --- Constructor parameters ---
    n_clusters: int | np.ndarray = field(
        default_factory=lambda: np.arange(2, 10 + 1, dtype=int)
    )
    resolution: float | np.ndarray | None = None
    sweep: str | None = None
    sweep_values: np.ndarray | None = None
    finer_is_larger: bool | None = None
    noise_policy: NoisePolicy = "drop"
    n_resamples: int = 100
    subsample_ratio: float = 0.618
    anchor_threshold: int = 5000
    consensus_anchors: int | float | None = None

    estimator_param_grids: list[GridSpec] | Literal["light", "full"] = "light"
    normalization_options: list[PreprocOption] | None = None
    dim_reduction_options: list[PreprocOption] | None = None

    classifier: ClassifierMixin | None = None
    n_trees: int = 100

    X_: np.ndarray | None = field(init=False, default=None)
    consensus_anchors_: np.ndarray | None = field(init=False, default=None)
    reference_labels: np.ndarray | None = None

    n_jobs: int = 1
    random_state: int | None = None
    verbose: int = 0

    # --- Fitted attributes (set by fit()) ---
    estimator_results_: pd.DataFrame | None = field(init=False, default=None)
    estimator_param_grids_: list[GridSpec] | None = field(init=False, default=None)
    preprocessing_results_: pd.DataFrame | None = field(init=False, default=None)
    sweep_: SweepSpec | None = field(init=False, default=None)

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
        use_rep: str | None = None,
        layer: str | None = None,
        n_pcs: int | None = None,
        reference_labels: np.ndarray | None = None,
        randomize_preprocessing: bool = False,
        show_progress: bool = False,
        mode: RunMode = "default",
        random_state: int | None = None,
    ) -> "CARVE":
        """Run CARVE validation on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or AnnData
            Input data. Accepts NumPy arrays, pandas DataFrames, SciPy
            sparse matrices, and :class:`~anndata.AnnData` objects. Sparse
            input is densified; see :func:`carve._utils.ensure_2d_array`.
        y : ignored
            Included for sklearn compatibility.
        use_rep : str, optional
            Only valid when ``X`` is an AnnData. Key in ``adata.obsm`` to
            cluster, or ``"X"`` for ``adata.X``. Defaults to ``"X_pca"``
            when present, else ``adata.X``.
        layer : str, optional
            Only valid when ``X`` is an AnnData. Key in ``adata.layers`` to
            cluster. Mutually exclusive with ``use_rep``.
        n_pcs : int, optional
            Only valid when ``X`` is an AnnData. Restrict the chosen
            representation to its first ``n_pcs`` columns.
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
            If estimator parameter grids have inconsistent sweep values, or if
            they mix more than one sweep parameter (k-based and
            resolution-based estimators cannot be compared in one run). Also
            if ``use_rep``, ``layer`` or ``n_pcs`` is given for a non-AnnData
            ``X``.

        See Also
        --------
        carve.tl.carve : AnnData-native entry point that also writes the
            results back into the object.
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
        if _anndata.is_anndata(X):
            X = _anndata.get_representation(
                X, use_rep=use_rep, layer=layer, n_pcs=n_pcs
            )
        elif use_rep is not None or layer is not None or n_pcs is not None:
            raise ValueError(
                "use_rep=, layer= and n_pcs= select a representation out of "
                "an AnnData; they are not valid when X is an array. Pass the "
                "array you want to cluster directly."
            )
        X = ensure_2d_array(X)
        self.X_ = X

        seed = self.random_state if random_state is None else random_state
        self.consensus_anchors_ = resolve_anchors(
            X.shape[0],
            consensus_anchors=self.consensus_anchors,
            anchor_threshold=self.anchor_threshold,
            random_state=seed,
        )
        if self.consensus_anchors_ is not None:
            warnings.warn(
                f"n={X.shape[0]} exceeds anchor_threshold="
                f"{self.anchor_threshold}, so CARVE is using anchored "
                f"consensus over {self.consensus_anchors_.size} anchors. "
                "Per-sample scores and labels still cover every sample; "
                "consensus matrices and PAC are computed over the anchors.",
                RuntimeWarning,
                stacklevel=2,
            )

        if reference_labels is not None:
            ref_arr = np.asarray(reference_labels)

            if not np.issubdtype(ref_arr.dtype, np.integer):
                ref_arr, _ = pd.factorize(ref_arr)

            self.reference_labels = ref_arr

        # --- Resolve the sweep axis ---
        custom_grids = not isinstance(self.estimator_param_grids, str)

        sweep_arg = self.sweep
        sweep_values = self.sweep_values

        if custom_grids:
            # Let the grids determine the sweep parameter, e.g.,
            # estimator_param_grids=[(LeidenClustering, {"resolution": [...]})]
            # works without also passing resolution=.
            if sweep_arg is None and self.resolution is None:
                sweep_arg = infer_sweep_param(self.estimator_param_grids)

            if sweep_values is None and sweep_arg is not None:
                sweep_values = grid_sweep_values(self.estimator_param_grids, sweep_arg)

        sweep_spec = resolve_sweep(
            n_clusters=self.n_clusters,
            resolution=self.resolution,
            sweep=sweep_arg,
            sweep_values=sweep_values,
            finer_is_larger=self.finer_is_larger,
        )

        # --- Resolve estimator grids along that axis ---
        if custom_grids:
            estimator_param_grids = self.estimator_param_grids
            sweep_spec = replace(
                sweep_spec,
                values=validate_grids(estimator_param_grids, sweep_spec),
            )

        elif self.estimator_param_grids in ("light", "full"):
            estimator_param_grids = default_estimator_grids(
                X, preset=self.estimator_param_grids, sweep=sweep_spec
            )

        else:
            raise ValueError(
                f"Unknown estimator_param_grids preset "
                f"{self.estimator_param_grids!r}. Expected 'light', 'full', "
                "or a list of (EstimatorClass, param_grid) tuples."
            )

        self.estimator_param_grids_ = estimator_param_grids
        self.sweep_ = sweep_spec

        # --- Resolve preprocessing options ---
        # The default option lists are only consumed when a random pipeline is
        # sampled per resample (see _pipeline.build_preprocessing_pipeline), so
        # resolving them otherwise would import UMAP for nothing.
        norm_options = self.normalization_options or default_normalization_options()
        if self.dim_reduction_options is not None:
            dr_options = self.dim_reduction_options
        elif randomize_preprocessing:
            dr_options = default_dim_reduction_options(X, self.subsample_ratio)
        else:
            dr_options = []

        # --- Print run header ---
        _print_run_header(
            X=X,
            sweep=sweep_spec,
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
            consensus_summaries,
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
            sweep=sweep_spec,
            noise_policy=self.noise_policy,
            show_progress=show_progress,
            mode=policy.mode,
            verbose=self.verbose,
            anchors=self.consensus_anchors_,
        )

        self.estimator_results_ = pd.DataFrame.from_records(estimator_records)

        self.preprocessing_results_ = (
            None
            if not randomize_preprocessing
            else summarize_preprocessing_records(
                pipeline_records, sweep_param=sweep_spec.param
            )
        )

        n_rows = int(self.estimator_results_.shape[0])

        # --- Sanity check: config_id alignment ---
        if not np.array_equal(
            self.estimator_results_["config_id"].to_numpy(), np.arange(n_rows)
        ):
            raise RuntimeError(
                "config_id is misaligned with the per-configuration artifact "
                "containers. This is an internal CARVE error."
            )

        # --- Stability-derived metrics ---
        if policy.run_stability and consensus_summaries is not None:
            self.stability_gini_scores_ = np.vstack(
                [s.gini for s in consensus_summaries]
            )
            self.stability_ce_scores_ = np.vstack([s.ce for s in consensus_summaries])

            self.estimator_results_["consensus_pac_stability"] = [
                s.pac for s in consensus_summaries
            ]
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

    def _select_row(
        self,
        *,
        measure: str,
        rule: str,
        not_two: bool = False,
        k: int | None = None,
        sweep_value: float | None = None,
    ) -> tuple[pd.Series, int, int, bool]:
        """Resolve selection criteria to a single configuration row.

        Parameters
        ----------
        measure : str
            Metric key used to select the best configuration.
        rule : str
            Selection rule ("max", "1se", "quantile").
        not_two : bool, default=False
            Whether to exclude two-cluster configurations.
        k : int, optional
            Restrict selection to configurations with this number of
            clusters. Only valid when the run swept ``n_clusters``.
        sweep_value : float, optional
            Restrict selection to configurations at this value of the
            swept hyperparameter (e.g. a specific Leiden ``resolution``).

        Returns
        -------
        row : pandas.Series
            The selected configuration.
        config_id : int
            Key into the per-configuration artifact containers
            (``consensus_matrices_``,
            ``consensus_generalizability_matrices_``,
            ``generalizability_scores_``, and the sample-level score
            arrays). Read from the row's ``config_id`` column. This is a
            *join*, not a positional slice: it stays correct after the
            caller sorts, filters or reindexes ``estimator_results_``.
        n_clusters : int
            Number of clusters for this row: the requested ``n_clusters``
            in k mode, otherwise the rounded mean observed count.
        pinned : bool
            Whether the user pinned the configuration.

        Raises
        ------
        RuntimeError
            If the instance has not been fitted yet.
        ValueError
            If both pins are given, if ``k`` is used outside k mode, or if
            no configuration matches the pin.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        df = self.estimator_results_
        param = sweep_param_name(df)

        if k is not None and sweep_value is not None:
            raise ValueError("Pass at most one of k= and sweep_value=.")

        if k is not None and param != "n_clusters":
            raise ValueError(
                f"This run swept {param!r}, not 'n_clusters'. Use "
                f"sweep_value=... to pin a {param}, and consensus_k=... to "
                "fix the number of clusters used to cut the consensus matrix."
            )

        pin = k if k is not None else sweep_value

        if pin is None:
            row = select_best_row_by_rule(
                df, measure=measure, rule=rule, not_two=not_two
            )
        else:
            col = "sweep_value"
            df_pin = df[np.isclose(df[col].astype(float), float(pin))]

            if df_pin.empty:
                pin_name = "k" if param == "n_clusters" else param
                raise ValueError(f"No configurations found for {pin_name}={pin}.")

            row = select_best_row_by_rule(df_pin, measure=measure, rule=rule)

        return row, config_id_of(row), observed_k(row), pin is not None

    def get_labels(
        self,
        *,
        measure: str = "stability",
        rule: str = "1se",
        k: int | None = None,
        sweep_value: float | None = None,
        consensus_k: int | None = None,
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
            Optional fixed number of clusters to select. Only valid when
            the run swept ``n_clusters``.
        sweep_value : float or None, default=None
            Optional fixed value of the swept hyperparameter (e.g. a
            specific Leiden ``resolution``).
        consensus_k : int or None, default=None
            Number of clusters used to cut the consensus matrix. Defaults
            to the requested ``n_clusters`` in k mode, or to the rounded
            mean number of clusters observed at the selected sweep value.
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

        # --- Select best configuration ---
        row, config_id, selected_k, _ = self._select_row(
            measure=measure,
            rule=rule,
            not_two=not_two,
            k=k,
            sweep_value=sweep_value,
        )

        # In resolution mode number of clusters is an outcome;
        # consensus dendrogram is cut at count actually observed.
        cut_k = int(consensus_k) if consensus_k is not None else int(selected_k)

        if cut_k < 2:
            raise ValueError(
                f"Cannot cut the consensus matrix at {cut_k} cluster(s). "
                "The selected configuration is degenerate; pass consensus_k= "
                "explicitly or exclude that end of the sweep."
            )

        # --- Retrieve the consensus matrix ---
        if policy.run_stability and self.consensus_matrices_ is not None:
            M_raw = self.consensus_matrices_[config_id]
        elif (
            policy.run_generalizability
            and self.consensus_generalizability_matrices_ is not None
        ):
            M_raw = self.consensus_generalizability_matrices_[config_id]
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
                n_clusters=cut_k,
                linkage="average",
                metric="precomputed",
            )

        labels = estimator.fit_predict(D)

        if self.consensus_anchors_ is not None:
            labels = self._extend_anchor_labels(labels)

        # --- Align with reference labels if available ---
        cur_k = int(np.unique(labels).size)
        ref = self.reference_labels
        ref_k = int(np.unique(ref).size) if ref is not None else None

        if (ref is None) or (ref_k != cur_k):
            self.reference_labels = labels
        else:
            labels = align_cluster_labels(ref, labels)

        return np.asarray(labels, dtype=np.int32)

    def _extend_anchor_labels(self, anchor_labels: np.ndarray) -> np.ndarray:
        """Label every sample from a cut taken over the anchor subset.

        The anchor block yields labels for m anchors. The remaining samples
        are assigned by the same classifier CARVE clones per resample to
        score generalizability, fitted on the anchors and their cut labels.
        Anchors keep the labels the cut gave them rather than the
        classifier's prediction for them.
        """
        if self.X_ is None:
            raise RuntimeError(
                "X_ is not available, so anchored labels cannot be extended to "
                "every sample. Call fit() first, or restore X_ after load()."
            )

        anchors = self.consensus_anchors_
        n_samples = self.X_.shape[0]
        anchor_labels = np.asarray(anchor_labels)

        labels = np.empty(n_samples, dtype=np.int64)
        labels[anchors] = anchor_labels

        rest = np.setdiff1d(np.arange(n_samples), anchors, assume_unique=False)
        if rest.size == 0:
            return labels

        if np.unique(anchor_labels).size < 2:
            # A degenerate cut gives the classifier a single class; every
            # remaining sample belongs to it by construction.
            labels[rest] = anchor_labels[0]
            return labels

        # Mirrors the default/clone construction in _runner.py's
        # _compute_generalizability_ari (the generalizability path run on
        # every resample); the two must stay in step.
        if self.classifier is None:
            n_features = self.X_.shape[1]
            classifier = RandomForestClassifier(
                n_estimators=self.n_trees,
                max_depth=n_features,
                max_features=int(np.sqrt(n_features)),
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            classifier = clone(self.classifier)
            if "random_state" in classifier.get_params():
                classifier.set_params(random_state=self.random_state)

        classifier.fit(self.X_[anchors], anchor_labels)
        labels[rest] = classifier.predict(self.X_[rest])
        return labels

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
            Selected number of clusters. In resolution mode this is the
            rounded mean number of clusters observed across resamples at the
            selected sweep value.

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

    def get_sweep_value(
        self,
        *,
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
    ) -> float:
        """Return the selected value of the swept hyperparameter.

        In the default k mode this is the selected number of clusters; in
        resolution mode it is the selected ``resolution``.

        Parameters
        ----------
        measure : str, default="stability"
            Metric key used to select the best configuration.
        rule : str, default="1se"
            Selection rule ("max", "1se", "quantile").
        not_two : bool, default=False
            If True, exclude two-cluster configurations from selection.

        Returns
        -------
        value : float
            Selected sweep value.

        Raises
        ------
        RuntimeError
            If the instance has not been fitted yet.
        """
        row, _, _, _ = self._select_row(measure=measure, rule=rule, not_two=not_two)

        return float(row["sweep_value"])

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
    ) -> Axes | None:
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
        sweep_value: float | None = None,
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
    ) -> Axes | None:
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
            Only valid when the run swept ``n_clusters``.
        sweep_value : float, optional
            If given, restrict selection to this value of the swept
            hyperparameter (e.g. a specific Leiden ``resolution``).
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

        row, config_id, selected_k, _ = self._select_row(
            measure=measure,
            rule=rule,
            not_two=not_two,
            k=k,
            sweep_value=sweep_value,
        )

        matrix = matrices[config_id]
        if matrix is None:
            raise RuntimeError(
                f"Selected consensus matrix is not available for mode={mode!r}."
            )

        labels = self.get_labels(
            measure=measure,
            rule=rule,
            k=k,
            sweep_value=sweep_value,
            not_two=not_two,
            consensus_k=selected_k,
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
        sweep_value: float | None = None,
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
    ) -> Axes | None:
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
            Only valid when the run swept ``n_clusters``.
        sweep_value : float, optional
            If given, restrict selection to this value of the swept
            hyperparameter (e.g. a specific Leiden ``resolution``).
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
        row, config_id, selected_k, pinned = self._select_row(
            measure=measure,
            rule=rule,
            not_two=not_two,
            k=k,
            sweep_value=sweep_value,
        )
        df = self.estimator_results_

        # --- Resolve score source ---
        if source == "gini":
            if self.stability_gini_scores_ is None:
                raise RuntimeError(
                    "Gini stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_gini_scores_[config_id], dtype=float)
            default_ylabel = "Cluster Stability (Gini)"

        elif source == "ce":
            if self.stability_ce_scores_ is None:
                raise RuntimeError(
                    "CE stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_ce_scores_[config_id], dtype=float)
            default_ylabel = "Cluster Stability (CE)"

        elif source == "accuracy":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[config_id], dtype=float)
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
            k=k,
            sweep_value=sweep_value,
            not_two=not_two,
            consensus_k=selected_k,
            mode=labels_mode,
        )

        if ylabel is None:
            ylabel = default_ylabel

        # --- Build annotation ---
        if annotation is True:
            annotation_text = _get_annotation(
                measure=measure,
                rule=rule,
                estimator_results=df,
                row=row,
                selected_k=selected_k,
                pinned=pinned,
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
        sweep_value: float | None = None,
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
    ) -> Axes | None:
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
            Only valid when the run swept ``n_clusters``.
        sweep_value : float, optional
            If given, restrict selection to this value of the swept
            hyperparameter (e.g. a specific Leiden ``resolution``).
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
        row, config_id, selected_k, pinned = self._select_row(
            measure=measure,
            rule=rule,
            not_two=not_two,
            k=k,
            sweep_value=sweep_value,
        )
        df = self.estimator_results_

        # --- Resolve score source ---
        if source == "gini":
            if self.stability_gini_scores_ is None:
                raise RuntimeError(
                    "Gini stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_gini_scores_[config_id], dtype=float)
            default_ylabel = "Cluster Stability (Gini)"

        elif source == "ce":
            if self.stability_ce_scores_ is None:
                raise RuntimeError(
                    "CE stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_ce_scores_[config_id], dtype=float)
            default_ylabel = "Cluster Stability (CE)"

        elif source == "accuracy":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[config_id], dtype=float)
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
            k=k,
            sweep_value=sweep_value,
            not_two=not_two,
            consensus_k=selected_k,
            mode=labels_mode,
        )

        if ylabel is None:
            ylabel = default_ylabel

        # --- Build annotation ---
        if annotation is True:
            annotation_text = _get_annotation(
                measure=measure,
                rule=rule,
                estimator_results=df,
                row=row,
                selected_k=selected_k,
                pinned=pinned,
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
        sweep_value: float | None = None,
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
    ) -> Axes | None:
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
            Only valid when the run swept ``n_clusters``.
        sweep_value : float, optional
            If given, restrict selection to this value of the swept
            hyperparameter (e.g. a specific Leiden ``resolution``).
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
        row, config_id, selected_k, pinned = self._select_row(
            measure=measure,
            rule=rule,
            not_two=not_two,
            k=k,
            sweep_value=sweep_value,
        )
        df = self.estimator_results_

        # --- Resolve score source ---
        if source == "gini":
            if self.stability_gini_scores_ is None:
                raise RuntimeError(
                    "Gini stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_gini_scores_[config_id], dtype=float)
            scores_name = "Gini Stability"
        elif source == "ce":
            if self.stability_ce_scores_ is None:
                raise RuntimeError(
                    "CE stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_ce_scores_[config_id], dtype=float)
            scores_name = "CE Stability"
        elif source == "accuracy":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[config_id], dtype=float)
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
            k=k,
            sweep_value=sweep_value,
            not_two=not_two,
            consensus_k=selected_k,
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
                estimator_results=df,
                row=row,
                selected_k=selected_k,
                pinned=pinned,
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
        sweep_value: float | None = None,
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
    ) -> Axes | None:
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
            Only valid when the run swept ``n_clusters``.
        sweep_value : float, optional
            If given, restrict selection to this value of the swept
            hyperparameter (e.g. a specific Leiden ``resolution``).
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
        row, config_id, selected_k, pinned = self._select_row(
            measure=measure,
            rule=rule,
            not_two=not_two,
            k=k,
            sweep_value=sweep_value,
        )
        df = self.estimator_results_

        # --- Resolve score source ---
        if source == "gini":
            if self.stability_gini_scores_ is None:
                raise RuntimeError(
                    "Gini stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_gini_scores_[config_id], dtype=float)
            scores_name = "Gini Stability"
        elif source == "ce":
            if self.stability_ce_scores_ is None:
                raise RuntimeError(
                    "CE stability scores are not available for this run."
                )
            scores = np.asarray(self.stability_ce_scores_[config_id], dtype=float)
            scores_name = "CE Stability"
        elif source == "accuracy":
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            scores = np.asarray(self.generalizability_scores_[config_id], dtype=float)
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
            k=k,
            sweep_value=sweep_value,
            not_two=not_two,
            consensus_k=selected_k,
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
                estimator_results=df,
                row=row,
                selected_k=selected_k,
                pinned=pinned,
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
