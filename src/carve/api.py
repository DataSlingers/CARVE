"""Public CARVE API."""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import matplotlib as mpl

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering

from ._grids import default_estimator_grids, default_norm_options, default_dr_options
from ._output import _print_run_footer, _print_run_header
from ._runner import run_validation
from ._consensus import compute_consensus_metrics
from ._misclassification import compute_mean_generalizability
from ._selection import select_best_estimator, select_best_k, select_best_row_by_rule
from ._plotting import (
    screen_spec,
    plos_spec,
    save_figure,
    add_method_columns,
    plot_global_metric_over_k,
    plot_clustering
)
from ._utils import align_cluster_labels, ensure_2d_array, summarize_preprocessing_records, cluster_labels

GridSpec = Tuple[Type[ClusterMixin], Dict[str, List[Any]]]
PreprocSpec = Union[
    Tuple[Callable, Dict[str, Any]],        # (Method, params)
    Tuple[Callable, str, Dict[str, Any]],   # (Method, name, params)
]

@dataclass
class CARVE(BaseEstimator):
    """
    CARVE validator.

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
    dim_reduction_options : list of dimensionality reduction specs, optional
    reference_labels : array-like of shape (n_samples,), optional
        Reference labels for consistent plots.
    n_jobs : int, default=1
        Parallelism.
    random_state : int, optional
        RNG seed.
    verbose : int, default=1
        Verbosity level for console output.
    """
    n_clusters: Union[int, np.ndarray] = 10
    n_resamples: int = 100
    subsample_ratio: float = 0.8
    estimator_param_grids: Optional[List[GridSpec]] = None
    normalization_options: Optional[List[PreprocSpec]] = None
    dim_reduction_options: Optional[List[PreprocSpec]] = None
    reference_labels: Optional[np.ndarray] = None
    n_jobs: int = 1
    random_state: Optional[int] = None
    verbose: int = 1

    # Attributes set after calling fit().
    estimator_results_: Optional[pd.DataFrame] = field(init=False, default=None)
    preprocessing_results_: Optional[pd.DataFrame] = field(init=False, default=None)
    consensus_matrices_: Optional[List[np.ndarray]] = field(init=False, default=None)
    consensus_generalizability_matrices_: Optional[List[np.ndarray]] = field(init=False, default=None)
    stability_gini_scores_: Optional[np.ndarray] = field(init=False, default=None)
    stability_ce_scores_: Optional[np.ndarray] = field(init=False, default=None)
    generalizability_scores_: Optional[List[np.ndarray]] = field(init=False, default=None)
    misclassification_rates_: Optional[np.ndarray] = field(init=False, default=None)
    X_: Optional[np.ndarray] = field(init=False, default=None)
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        reference_labels: Optional[np.ndarray] = None,
        randomize_preprocessing: bool = False,
        show_progress: bool = False,
        mode: Literal['default', 'stability', 'generalizability'] = 'default',
        random_state: Optional[int] = None,
    ) -> 'CARVE':
        """
        Run CARVE validation on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored
            Included for sklearn compatibility.
        reference_labels : array-like of shape (n_samples,), optional
            Reference labels used for generalizability metrics.
            Overrides the `reference_labels` passed at __init__ if given.
        randomize_preprocessing : bool, default=False
            Whether to randomize preprocessing pipelines.
        show_progress : bool, default=False
            Show progress bar.
        mode : Literal['default', 'stability', 'generalizability'], default='default'
            Determines whether to run CARVE regularly ('default') or whether
            to only run stability analysis ('stability'), 
            or generalizability analysis ('generalizability).
        random_state : int, optional
            Per-call RNG seed. If None, uses self.random_state.

        Returns
        -------
        self
        """
        if mode != "default":
            warnings.warn(
                "Non-default mode is experimental and may break downstream functionality.",
                RuntimeWarning,
                stacklevel=2,
            )
        
        X = ensure_2d_array(X)
        self.X_ = X

        if reference_labels is not None:
            ref_arr = np.asarray(reference_labels)
            if not np.issubdtype(ref_arr.dtype, np.integer):
                ref_arr, _ = pd.factorize(ref_arr)
            self.reference_labels = ref_arr

        estimator_grids = self.estimator_param_grids or default_estimator_grids(X, self.n_clusters)
        norm_options = self.normalization_options or default_norm_options()
        dr_options = self.dim_reduction_options or default_dr_options(X, self.subsample_ratio)
        
        # Emit run header to stdout.
        _print_run_header(
            X=X,
            n_clusters=self.n_clusters,
            n_resamples=self.n_resamples,
            subsample_ratio=self.subsample_ratio,
            estimator_grids=estimator_grids,
            n_jobs=self.n_jobs,
            random_preprocess=randomize_preprocessing,
            random_state=self.random_state if random_state is None else random_state,
            verbose=self.verbose
        )

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
            norm_options=norm_options,
            dr_options=dr_options,
            random_preprocess=randomize_preprocessing,
            n_jobs=self.n_jobs,
            random_state=self.random_state if random_state is None else random_state,
            mode=mode,
            prog_bar=show_progress
        )

        self.estimator_results_ = pd.DataFrame.from_records(estimator_records)
        self.preprocessing_results_ = (
            None if not randomize_preprocessing else summarize_preprocessing_records(pipeline_records)
        )

        n_rows = int(self.estimator_results_.shape[0])

        # --- Stability-Derived Metrics
        if mode != "generalizability" and self.consensus_matrices_ is not None:
            gini_list, ce_list, pac_list = compute_consensus_metrics(self.consensus_matrices_)
            
            self.stability_gini_scores_ = np.vstack(gini_list)
            self.stability_ce_scores_ = np.vstack(ce_list)
            self.estimator_results_["consensus_pac_stability"] = pac_list
            self.estimator_results_["consensus_gini_stability"] = (self.stability_gini_scores_.mean(axis=1))
            self.estimator_results_["consensus_ce_stability"] = (self.stability_ce_scores_.mean(axis=1))
        else:
            self.stability_gini_scores_ = None
            self.stability_ce_scores_ = None
            self.estimator_results_["consensus_pac_stability"] = np.full(n_rows, np.nan)
            self.estimator_results_["consensus_gini_stability"] = np.full(n_rows, np.nan)
            self.estimator_results_["consensus_ce_stability"] = np.full(n_rows, np.nan)

        # --- Generalizability-Derived Metrics
        if mode != "stability" and self.generalizability_scores_ is not None:
            misclassification_arrs = compute_mean_generalizability(self.generalizability_scores_)
            
            self.misclassification_rates_ = misclassification_arrs
            self.estimator_results_["misclassification_generalizability"] = misclassification_arrs
        else:
            self.misclassification_rates_ = None
            self.estimator_results_["misclassification_generalizability"] = np.full(n_rows, np.nan)
        
        # output footer
        _print_run_footer(estimator_df=self.estimator_results_, verbose=self.verbose)

        return self

    def get_labels(
        self,
        *,
        measure: str = "stability",
        rule: str = 'max',
        k: Optional[int] = None, 
        mode: Literal['default', 'generalizability'] = 'default',
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
            value selected by `measure` and `rule`.
        mode : Literal['default', 'generalizability'], default='default'
            Determines which consensus matrix is used to return labels.
        estimator : Type[ClusterMixin] | None, default=None
            If provided, uses this estimator to cluster the consensus
            distance matrix; otherwise defaults to average-linkage
            `AgglomerativeClustering` with precomputed distances.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Clustering labels derived from the selected consensus matrix.
        """
        if (self.consensus_matrices_ is None and mode == 'default') or (self.consensus_generalizability_matrices_ is None and mode == 'generalizability') or self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")
        
        df = self.estimator_results_

        # Pick best row index (df index matches consensus_matrices_ order; subsetting by k does not change that)
        if k is None:
            row = select_best_row_by_rule(df, measure=measure, rule=rule)  # returns a row/series
            k = int(row["n_clusters"])
            best_idx = int(row.name)  # keep original df index
        else:
            df_k = df[df["n_clusters"] == k]
            if df_k.empty:
                raise ValueError(f"No configurations found for k={k}.")
            row = select_best_row_by_rule(df_k, measure=measure, rule=rule)
            best_idx = int(row.name)

        if mode == 'default':
            M_raw = self.consensus_matrices_[best_idx]
        elif mode == 'generalizability':
            M_raw = self.consensus_generalizability_matrices_[best_idx]
        else:
            raise ValueError("mode must be 'default' or 'generalizability'.")
        
        if M_raw is None:
            raise RuntimeError(
                f"Consensus matrix not available for mode={mode!r}."
                "This run likely used split-mode and skipped building that artifact."
            )

        M = np.asarray(M_raw, dtype=float)
        
        S = 0.5 * (M + M.T)                    # enforce symmetry
        np.fill_diagonal(S, 1.0)
        S = np.clip(S, 0.0, 1.0)

        # Handle NaNs (we use 0.5 as neutral-esque fill)
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

        # Align with reference-labels if available
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
        rule: str = 'max',
    ) -> int:
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
            Fitted estimator.
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        k = select_best_k(
            self.estimator_results_, 
            measure=measure, rule=rule
        )
        
        return k
    
    def get_estimator(
        self,
        *,
        measure: str = "stability",
        rule: str = 'max',
    ) -> ClusterMixin:
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

        estimator = select_best_estimator(
            self.estimator_results_, 
            measure=measure, rule=rule
        )
        
        return estimator
    
    # Plotting API wrappers (static backend).
    def plot_metric_over_k(
        self,
        measure: str = "stability",
        *,
        rule: str = "1se",
        mode: str = "screen",
        figsize: Tuple[int, int] = (10, 8),
        dpi: Optional[int] = None,
        decimals: int = 4,
        show_grid: bool = True,
        legend_outside: bool = True,
        save_path: Optional[str] = None,
        # width: int = 1000,
        # height: int = 800,
        interactive: bool = False,
        # plotting parameters
        ax: mpl.axes.Axes | None = None,
        show_1se: bool = True,
        show_quant: bool = False,
        # legend/key strategy
        show_legend_panel: bool | Literal["auto"] = "auto",
        max_full_annotation: int = 7,
        key_label_wrap: int = 20,
        # aesthetics
        marker: str = "o",
        linewidth: float = 1.6,
        alpha_band: float = 0.18,
        highlight_linewidth: float = 2.8,
        grid_alpha: float = 0.22,
        title: str | None = None,
        y_label: str | None = None,
        x_label: str = "Number of Clusters (k)",
        annotate_selection: bool = True,
    ) -> None:
        """Plot a global metric over k for all configurations.

        Parameters
        ----------
        measure : str, default="stability"
            Metric key to plot.
        rule : str, default="1se"
            Selection rule used to highlight the best configuration.
        mode : {"screen", "plos"}, default="screen"
            Plotting mode.
        figsize : tuple of int, default=(10, 8)
            Figure size in inches.
        dpi : int or None, default=None
            Override DPI for export.
        decimals : int, default=4
            Decimal precision for method labels.
        show_grid : bool, default=True
            Whether to show grid lines.
        legend_outside : bool, default=True
            Place legend outside the plot when shown.
        save_path : str or None, default=None
            Optional file path to save the figure.
        interactive : bool, default=False
            Interactive plotting is currently disabled (falls back to static).
        ax : matplotlib.axes.Axes or None, default=None
            Optional axes to draw into.
        show_1se : bool, default=True
            Show 1-SE bands if available.
        show_quant : bool, default=False
            Show quantile bands if available.
        show_legend_panel : bool or "auto", default="auto"
            Whether to draw a legend panel or annotate endpoints.
        max_full_annotation : int, default=7
            Max methods for full annotation before using a legend.
        key_label_wrap : int, default=20
            Wrap width for labels.
        marker : str, default="o"
            Marker style.
        linewidth : float, default=1.6
            Line width for non-selected methods.
        alpha_band : float, default=0.18
            Opacity for uncertainty bands.
        highlight_linewidth : float, default=2.8
            Line width for the selected method.
        grid_alpha : float, default=0.22
            Grid line opacity.
        title : str or None, default=None
            Plot title.
        y_label : str or None, default=None
            Y-axis label.
        x_label : str, default="Number of Clusters (k)"
            X-axis label.
        annotate_selection : bool, default=True
            Whether to annotate the selected configuration.
        """
        if self.estimator_results_ is None or self.estimator_results_.empty:
            warnings.warn("estimator_results_ is empty; nothing to plot. Run fit() first.", RuntimeWarning, stacklevel=2)
            return

        if rule not in {"max", "1se", "quantile"}:
            raise ValueError("rule must be 'max', '1se', or 'quantile'")

        if mode not in {"screen", "plos"}:
            raise ValueError("mode must be 'screen' or 'plos'")

        if interactive:
            warnings.warn(
                "Interactive plotting is currently disabled in the new plotting backend; falling back to static.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Build PlotSpec from figsize (+ mode), with optional dpi override
        if mode == "plos":
            spec = plos_spec(width_in=float(figsize[0]), height_in=float(figsize[1]), dpi=(dpi or 600))
        else:
            spec = screen_spec(width_in=float(figsize[0]), height_in=float(figsize[1]), dpi=(dpi or 120))

        # Ensure method labels exist and honor `decimals`
        df_plot = self.estimator_results_
        if "_method_id" not in df_plot.columns or "_method_label" not in df_plot.columns:
            df_plot = add_method_columns(df_plot, decimals=decimals)

        # Plot (new static function)
        out = plot_global_metric_over_k(
            df_plot,
            measure=measure,
            rule=rule,
            spec=spec,
            ax=ax,
            show_1se=show_1se,
            show_quant=show_quant,
            show_legend_panel=show_legend_panel,
            max_full_annotation=max_full_annotation,
            key_label_wrap=key_label_wrap,
            marker=marker,
            linewidth=linewidth,
            alpha_band=alpha_band,
            highlight_linewidth=highlight_linewidth,
            grid_alpha=(grid_alpha if show_grid else 0.0),
            title=title,
            y_label=y_label,
            x_label=x_label,
            annotate_selection=annotate_selection,
        )

        ax = out["ax"]
        fig = out["fig"]

        # If show_grid is False, disable it
        if not show_grid:
            ax.grid(False)

        # Legend placement preference 
        # (only applies when we used real legend; with little models, we use key panel instead)
        if not legend_outside:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
                ax.legend(loc="best", frameon=False)

        if save_path is not None:
            save_figure(fig, save_path, spec=spec)
            
    def plot_cluster_summary(
        self,
        measure: str = "stability",
        *,
        figsize: Tuple[int, int] = (10, 6),
        dpi: Optional[int] = None,
        decimals: int = 4,
        save_path: Optional[str] = None,
        interactive: bool = False,
        # selection + which cluster-wise thing to show
        rule: str = "1se",
        mode: str = "screen",
        k: Optional[int] = None,
        cluster_metric: Literal["gini", "ce", "misclassification"] = "gini",
        # DR + plot aesthetics
        dr: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        title: Optional[str] = None,
        point_size: float = 40.0,
        min_point_size: float = 10.0,
        point_alpha: float = 0.85,
        min_point_alpha: float = 0.35,
        grid_alpha: float = 0.15,
        show_scatter_grid: bool = False,
        show_scatter_axes: bool = False,
        show_boxplot_axes: bool = False,
        annotate_selection: bool = True,
    ) -> None:
        """Plot clustering scatter + cluster-wise boxplots.

        Parameters
        ----------
        measure : str, default="stability"
            Metric key used to select the best configuration.
        figsize : tuple of int, default=(10, 6)
            Figure size in inches.
        dpi : int or None, default=None
            Override DPI for export.
        decimals : int, default=4
            Decimal precision for method labels.
        save_path : str or None, default=None
            Optional file path to save the figure.
        interactive : bool, default=False
            Interactive plotting is currently disabled (falls back to static).
        rule : str, default="1se"
            Selection rule used to highlight the best configuration.
        mode : {"screen", "plos"}, default="screen"
            Plotting mode.
        k : int or None, default=None
            Optional fixed number of clusters to select.
        cluster_metric : {"gini", "ce", "misclassification"}, default="gini"
            Per-sample metric to display in boxplots.
        dr : callable or None, default=None
            Dimensionality reduction function or transformer.
        title : str or None, default=None
            Figure title.
        point_size : float, default=40.0
            Base point size for scatter.
        min_point_size : float, default=10.0
            Minimum point size for scatter.
        point_alpha : float, default=0.85
            Maximum alpha for scatter.
        min_point_alpha : float, default=0.35
            Minimum alpha for scatter.
        grid_alpha : float, default=0.15
            Grid line opacity.
        show_scatter_grid : bool, default=False
            Whether to show grid lines in the boxplot panel.
        show_scatter_axes : bool, default=False
            Whether to show axes on the scatter plot.
        show_boxplot_axes : bool, default=False
            Whether to show axes on the boxplot.
        annotate_selection : bool, default=True
            Whether to annotate the selected configuration.
        """
        if self.estimator_results_ is None or self.estimator_results_.empty:
            warnings.warn("estimator_results_ is empty; nothing to plot. Run fit() first.", RuntimeWarning, stacklevel=2)
            return

        if rule not in {"max", "1se", "quantile"}:
            raise ValueError("rule must be 'max', '1se', or 'quantile'.")

        if mode not in {"screen", "plos"}:
            raise ValueError("mode must be 'screen' or 'plos'.")

        if cluster_metric not in {"gini", "ce", "misclassification"}:
            raise ValueError("cluster_metric must be 'gini', 'ce', or 'misclassification'.")

        if interactive:
            warnings.warn(
                "Interactive plotting is currently disabled in the new plotting backend; falling back to static.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Build PlotSpec from figsize (+ mode), with optional dpi override
        if mode == "plos":
            spec = plos_spec(width_in=float(figsize[0]), height_in=float(figsize[1]), dpi=(dpi or 600))
        else:
            spec = screen_spec(width_in=float(figsize[0]), height_in=float(figsize[1]), dpi=(dpi or 120))

        # Ensure method labels exist and handle `decimals`
        df_plot = self.estimator_results_
        if "_method_id" not in df_plot.columns or "_method_label" not in df_plot.columns:
            df_plot = add_method_columns(df_plot, decimals=decimals)

        # Delegate to plotting backend
        out = plot_clustering(
            carve=self,
            measure=measure, 
            rule=rule,
            k=k,
            cluster_metric=cluster_metric,
            dr=dr,
            spec=spec,
            title=title,
            point_size=point_size,
            min_point_size=min_point_size,
            point_alpha=point_alpha,
            min_point_alpha=min_point_alpha,
            grid_alpha=grid_alpha,
            show_scatter_grid=show_scatter_grid,
            show_scatter_axes=show_scatter_axes,
            show_boxplot_axes=show_boxplot_axes,
            annotate_selection=annotate_selection,
        )

        fig = out["fig"]
        ax_box = out["ax_box"]

        # If show_grid=False, fully disable (backend also respects it, but keep consistent)
        if not show_scatter_grid:
            ax_box.grid(False)

        if save_path is not None:
            save_figure(fig, save_path, spec=spec)
        
    # def plot_preprocessing_results(
    #     self,
    #     measure: str = "stability",
    #     *,
    #     rule: str = "max",
    #     figsize: Tuple[int, int] = (10, 8),
    #     decimals: int = 4,
    #     show_grid: bool = True,
    #     legend_outside: bool = True,
    #     interactive: bool = True,
    # ) -> None:
    #     if self.pipeline_df_ is None or self.pipeline_df_.empty:
    #         warnings.warn("pipeline_df_ is empty; nothing to plot. Run validate() with random pre-processing first.", RuntimeWarning, stacklevel=2)
    #         return
        
    #     if rule not in {"max", "1se", "quantile"}:
    #         raise ValueError("rule must be 'max', '1se', or 'quantile'")
        
    #     config = PlotConfig(
    #         figsize=figsize,
    #         decimals=decimals,
    #         show_grid=show_grid,
    #         legend_outside=legend_outside
    #     )

    #     plot_pipeline_vs_k(self.pipeline_df_, measure=measure, config=config)
    
    # def plot_consensus_matrix(
    #     self,
    #     measure: str = "stability",
    #     *,
    #     rule: str = "max",
    #     k: int = None,
    #     figsize: Tuple[int, int] = (10, 8),
    #     decimals: int = 4,
    #     show_grid: bool = True,
    #     legend_outside: bool = True,
    # ) -> None:
    #     if self.model_df_ is None or self.model_df_.empty:
    #         warnings.warn("model_df_ is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
    #         return
        
    #     if rule not in {"max", "1se", "quantile"}:
    #         raise ValueError("rule must be 'max', '1se', or 'quantile'")
        
    #     config = PlotConfig(
    #         figsize=figsize,
    #         decimals=decimals,
    #         show_grid=show_grid,
    #         legend_outside=legend_outside
    #     )
        
    #     plot_consensus_matrix(
    #         model_df=self.model_df_, 
    #         consensus_mats_raw=self.consensus_mats_raw_, 
    #         measure=measure, 
    #         rule=rule, 
    #         k=k, 
    #         config=config
    #     )    
        
    # def plot_clustering(
    #     self,
    #     measure: str = "stability",
    #     sample_metric: str = "gini",
    #     *,
    #     k: Optional[int] = None,
    #     rule: str = "max",
    #     figsize: Tuple[int, int] = (10, 8),
    #     decimals: int = 4,
    #     show_grid: bool = True,
    #     legend_outside: bool = True,
    #     width: int = 1000,
    #     height: int = 800,
    #     min_size: float = 20.0,
    #     max_size: float = 180.0,
    #     min_alpha: float = 0.30,
    #     max_alpha: float = 1.00,
    #     interactive: bool = True,
    #     auto_display: bool = True
    # ) -> None:
    #     if self.model_df_ is None or self.model_df_.empty:
    #         warnings.warn("model_df_ is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
    #         return
        
    #     if rule not in {"max", "1se", "quantile"}:
    #         raise ValueError("rule must be 'max', '1se', or 'quantile'")
        
    #     config = PlotConfig(
    #         figsize=figsize,
    #         decimals=decimals,
    #         show_grid=show_grid,
    #         legend_outside=legend_outside
    #     )
        
    #     model_df_copy = self.model_df_.copy()
    #     if k is not None:
    #         model_df_copy = model_df_copy[model_df_copy['n_clusters'] == k]

    #     y_col = MEASURE_MAP[measure]
    #     se_col = f"{y_col}_se"
    #     if rule == "1se" and se_col not in model_df_copy.columns:
    #         warnings.warn(f"{se_col!r} not found; falling back to 'max' rule.", RuntimeWarning, stacklevel=2)
    #         rule = "max"

    #     # pick best row
    #     idx = get_best_row(model_df_copy, measure=measure, rule=rule, return_idx=True)
    #     row = model_df_copy.loc[idx]

    #     pos = self.model_df_.index.get_loc(idx)

    #     # labels
    #     labels = self.get_optimal_labels(measure=measure, rule=rule, k=k)
    #     n_clusters = int(row['n_clusters'])
    #     assert len(np.unique(labels)) == n_clusters, \
    #         f"labels has {len(np.unique(labels))} clusters, expected {n_clusters}"

    #     # sample-level vectors
    #     if MEASURE_MAP[measure] == "ari_stability":
    #         if sample_metric == "gini":
    #             sample_level_measures = self.stab_gini_arrs_[pos]
    #         elif sample_metric == "ce":
    #             sample_level_measures = self.stab_ce_arrs_[pos]
    #         else:
    #             raise ValueError("stab_measure must be 'gini' or 'ce'")
    #     else:
    #         sample_level_measures = self.generalizability_arrs_[pos]

    #     if interactive:
    #         fig = plot_clustering_interactive(
    #             X=self.X_, 
    #             row=row,
    #             labels=labels,
    #             stab_gini_vec=self.stab_gini_arrs_[pos],
    #             stab_ce_vec=self.stab_ce_arrs_[pos],
    #             gen_vec=self.generalizability_arrs_[pos],
    #             measure=measure,
    #             width=width,
    #             height=height,
    #             min_size=min_size,
    #             max_size=max_size,
    #             min_alpha=min_alpha,
    #             max_alpha=max_alpha,
    #             auto_display=auto_display
    #         )
    #         return fig

    #     else:
    #         plot_clustering(
    #             X=self.X_, 
    #             row=row,
    #             labels=labels,
    #             sample_level_measures=sample_level_measures,
    #             measure=measure,
    #             config=config,
    #             min_size=min_size,
    #             max_size=max_size,
    #             min_alpha=min_alpha,
    #             max_alpha=max_alpha
    #         )
    
    # def plot_consensus_clustering(
    #     self,
    #     measure: str = "stability",
    #     sample_metric: str = "gini",
    #     *,
    #     k: Optional[int] = None,
    #     rule: str = "max",
    #     figsize: Tuple[int, int] = (10, 8),
    #     decimals: int = 4,
    #     show_grid: bool = True,
    #     legend_outside: bool = True,
    #     min_size: float = 20.0,
    #     max_size: float = 180.0,
    #     min_alpha: float = 0.30,
    #     max_alpha: float = 1.00
    # ) -> None:
    #     if self.model_df_ is None or self.model_df_.empty:
    #         warnings.warn("model_df_ is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
    #         return
        
    #     if rule not in {"max", "1se", "quantile"}:
    #         raise ValueError("rule must be 'max', '1se', or 'quantile'")
        
    #     config = PlotConfig(
    #         figsize=figsize,
    #         decimals=decimals,
    #         show_grid=show_grid,
    #         legend_outside=legend_outside
    #     )
        
    #     model_df_copy = self.model_df_.copy()
    #     if k is not None:
    #         model_df_copy = model_df_copy[model_df_copy['n_clusters'] == k]

    #     y_col = MEASURE_MAP[measure]
    #     se_col = f"{y_col}_se"
    #     if rule == "1se" and se_col not in model_df_copy.columns:
    #         warnings.warn(f"{se_col!r} not found; falling back to 'max' rule.", RuntimeWarning, stacklevel=2)
    #         rule = "max"

    #     # pick best row
    #     idx = get_best_row(model_df_copy, measure=measure, rule=rule, return_idx=True)
    #     row = model_df_copy.loc[idx]

    #     pos = self.model_df_.index.get_loc(idx)

    #     # labels
    #     labels = self.get_optimal_labels(measure=measure, rule=rule, k=k)
    #     n_clusters = int(row['n_clusters'])
    #     assert len(np.unique(labels)) == n_clusters, \
    #         f"labels has {len(np.unique(labels))} clusters, expected {n_clusters}"

    #     # sample-level vectors
    #     if MEASURE_MAP[measure] == "ari_stability":
    #         if sample_metric == "gini":
    #             sample_level_measures = self.stab_gini_arrs_[pos]
    #         elif sample_metric == "ce":
    #             sample_level_measures = self.stab_ce_arrs_[pos]
    #         else:
    #             raise ValueError("stab_measure must be 'gini' or 'ce'")
    #     else:
    #         sample_level_measures = self.generalizability_arrs_[pos]
        
    #     plot_consensus_clustering(
    #             X=self.X_, 
    #             row=row,
    #             labels=labels,
    #             sample_level_measures=sample_level_measures,
    #             consensus_mat_raw=self.consensus_mats_raw_[pos],
    #             measure=measure,
    #             config=config,
    #             min_size=min_size,
    #             max_size=max_size,
    #             min_alpha=min_alpha,
    #             max_alpha=max_alpha
    #     )