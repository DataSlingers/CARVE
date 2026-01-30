from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import multiprocessing as mp

import matplotlib as mpl

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.model_selection import ParameterGrid

from .grids import default_model_grids, default_norm_options, default_dr_options
from ._output import _print_run_footer, _print_run_header
from ._runner import run_validation
from ._consensus import compute_consensus_metrics_batch
from ._misclassification import compute_global_misclassification_arrays
from ._selection import select_best_estimator, select_best_k, get_best_row, MEASURE_MAP
from ._plotting import (
    screen_spec,
    plos_spec,
    save_figure,
    add_method_columns,
    plot_global_metric_over_k,
    plot_clustering
)
from ._utils import align_labels, ensure_array2d, wrangle_pipeline_records, clustering_pipeline

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
    K : int or np.ndarray, default=10
        Number(s) of clusters to evaluate.
    B : int, default=100
        Number of resampling iterations.
    rho : float, default=0.6
        Subsampling proportion.
    model_grids : list of (Estimator, param_grid) tuples, optional
        Clustering models and their parameter grids.
    norm_options : list of preprocessing specs, optional
    dr_options : list of dimensionality reduction specs, optional
    ref_labels : array-like of shape (n_samples,), optional
        Reference labels for consistent plots.
    n_jobs : int, default=1
        Parallelism.
    random_state : int, optional
        RNG seed.
    """
    K: Union[int, np.ndarray] = 10
    B: int = 100
    rho: float = 0.6
    model_grids: Optional[List[GridSpec]] = None
    norm_options: Optional[List[PreprocSpec]] = None
    dr_options: Optional[List[PreprocSpec]] = None
    ref_labels: Optional[np.ndarray] = None
    n_jobs: int = 1
    random_state: Optional[int] = None
    verbose: int = 1

    # learned attributes (set by fit)
    model_df_: Optional[pd.DataFrame] = field(init=False, default=None)
    pipeline_df_: Optional[pd.DataFrame] = field(init=False, default=None)
    consensus_mats_raw_: Optional[List[np.ndarray]] = field(init=False, default=None)
    stab_gini_arrs_: Optional[np.ndarray] = field(init=False, default=None)
    stab_ce_arrs_: Optional[np.ndarray] = field(init=False, default=None)
    generalizability_arrs_: Optional[List[np.ndarray]] = field(init=False, default=None)
    misclassification_arrs_: Optional[np.ndarray] = field(init=False, default=None)
    X_: Optional[np.ndarray] = field(init=False, default=None)
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        ref_labels: Optional[np.ndarray] = None,
        random_preprocess: bool = False,
        prog_bar: bool = False,
        random_state: Optional[int] = None,
    ) -> 'CARVE':
        """
        Run CARVE validation on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored
            Included for sklearn compatibility.
        ref_labels : array-like of shape (n_samples,), optional
            Reference labels used for generalizability metrics.
            Overrides the `ref_labels` passed at __init__ if given.
        random_preprocess : bool, default=False
            Whether to randomize preprocessing pipelines.
        prog_bar : bool, default=False
            Show progress bar.
        random_state : int, optional
            Per-call RNG seed. If None, uses self.random_state.

        Returns
        -------
        self
        """
        X = ensure_array2d(X)
        self.X_ = X

        if ref_labels is not None:
            self.ref_labels = np.asarray(ref_labels)

        model_grids = self.model_grids or default_model_grids(X, self.K)
        norm_options = self.norm_options or default_norm_options()
        dr_options = self.dr_options or default_dr_options(X, self.rho)
        
        # output header
        _print_run_header(
            X=X,
            K=self.K,
            B=self.B,
            rho=self.rho,
            model_grids=model_grids,
            n_jobs=self.n_jobs,
            random_preprocess=random_preprocess,
            random_state=self.random_state if random_state is None else random_state,
            verbose=self.verbose
        )

        (
            model_records,
            pipeline_records,
            self.consensus_mats_raw_,
            self.generalizability_arrs_,
        ) = run_validation(
            X=X,
            model_grids=model_grids,
            B=self.B,
            rho=self.rho,
            norm_options=norm_options,
            dr_options=dr_options,
            random_preprocess=random_preprocess,
            n_jobs=self.n_jobs,
            random_state=self.random_state if random_state is None else random_state,
            prog_bar=prog_bar
        )

        self.model_df_ = pd.DataFrame.from_records(model_records)
        self.pipeline_df_ = (
            None if not random_preprocess else wrangle_pipeline_records(pipeline_records)
        )

        # compute stability vectors from consensus matrices
        gini_list, ce_list, pac_list = compute_consensus_metrics_batch(
            self.consensus_mats_raw_
        )
        misclassification_arrs = compute_global_misclassification_arrays(
            self.generalizability_arrs_
        )

        self.stab_gini_arrs_ = np.vstack(gini_list)
        self.stab_ce_arrs_ = np.vstack(ce_list)
        self.misclassification_arrs_ = misclassification_arrs

        self.model_df_["consensus_pac_stability"] = pac_list
        self.model_df_["consensus_gini_stability"] = self.stab_gini_arrs_.mean(axis=1)
        self.model_df_["consensus_ce_stability"] = self.stab_ce_arrs_.mean(axis=1)
        self.model_df_["misclassification_generalizability"] = misclassification_arrs
        
        # output footer
        _print_run_footer(model_df=self.model_df_, verbose=self.verbose)

        return self

    def get_optimal_labels(
        self,
        *,
        measure: str = "stability",
        rule: str = 'max',
        k: Optional[int] = None,
        return_estimator: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ClusterMixin]]:
        model_grids = self.model_grids or default_model_grids(self.X_, self.K)
        if self.model_df_ is None:
            raise RuntimeError("Call fit() first.")

        estimator = select_best_estimator(self.model_df_, model_grids=model_grids, measure=measure, rule=rule, k=k)
        
        labels = clustering_pipeline(self.X_, type(estimator), **estimator.get_params())

        cur_k = int(np.unique(labels).size)
        ref = self.ref_labels
        ref_k = int(np.unique(ref).size) if ref is not None else None

        if (ref is None) or (ref_k != cur_k):
            # store a fresh reference for this k
            self.ref_labels = labels
        else:
            labels = align_labels(ref, labels)

        return (labels, estimator) if return_estimator else labels
    
    def get_optimal_k(
        self,
        *,
        measure: str = "stability",
        rule: str = 'max',
    ) -> int:
        if self.model_df_ is None:
            raise RuntimeError("Call fit() first.")

        k = select_best_k(
            self.model_df_, 
            measure=measure, rule=rule
        )
        
        return k
    
    # ––––– Plotting Funcitons ––––– 
    def plot_results(
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
        if self.model_df_ is None or self.model_df_.empty:
            warnings.warn("model_df_ is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
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
        df_plot = self.model_df_
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
            
    def plot_clustering(
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
        """
        Wrapper around _plotting.plot_clustering.

        - Selects the optimal configuration via (measure, rule) unless k is provided.
        - Uses CARVE.get_optimal_labels(...) for the chosen config.
        - Draws scatter (DR; default PCA) + cluster-wise boxplots (gini/ce/misclassification).
        """
        if self.model_df_ is None or self.model_df_.empty:
            warnings.warn("model_df_ is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
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
        df_plot = self.model_df_
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