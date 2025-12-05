from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import multiprocessing as mp

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
from ._plotting import PlotConfig, plot_measure_vs_k, plot_measure_vs_k_interactive, plot_pipeline_vs_k, plot_consensus_matrix, plot_clustering, plot_clustering_interactive, plot_consensus_clustering
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
    n_jobs : int, default=cpu_count() - 1
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
    n_jobs: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
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
        rule: str = "max",
        figsize: Tuple[int, int] = (10, 8),
        decimals: int = 4,
        show_grid: bool = True,
        legend_outside: bool = True,
        width: int = 1000,
        height: int = 800,
        interactive: bool = True
    ) -> None:
        if self.model_df_ is None or self.model_df_.empty:
            warnings.warn("model_df_ is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
            return
        
        if rule not in {"max", "1se", "quantile"}:
            raise ValueError("rule must be 'max', '1se', or 'quantile'")
        
        config = PlotConfig(
            figsize=figsize,
            decimals=decimals,
            show_grid=show_grid,
            legend_outside=legend_outside
        )

        if interactive:
            plot_measure_vs_k_interactive(
                self.model_df_, 
                measure=measure, 
                rule=rule, 
                width=width,
                height=height
            )
            
        else:
            plot_measure_vs_k(
                self.model_df_, 
                measure=measure, 
                rule=rule, 
                config=config
            )
        
    def plot_preprocessing_results(
        self,
        measure: str = "stability",
        *,
        rule: str = "max",
        figsize: Tuple[int, int] = (10, 8),
        decimals: int = 4,
        show_grid: bool = True,
        legend_outside: bool = True,
        interactive: bool = True,
    ) -> None:
        if self.pipeline_df_ is None or self.pipeline_df_.empty:
            warnings.warn("pipeline_df_ is empty; nothing to plot. Run validate() with random pre-processing first.", RuntimeWarning, stacklevel=2)
            return
        
        if rule not in {"max", "1se", "quantile"}:
            raise ValueError("rule must be 'max', '1se', or 'quantile'")
        
        config = PlotConfig(
            figsize=figsize,
            decimals=decimals,
            show_grid=show_grid,
            legend_outside=legend_outside
        )

        plot_pipeline_vs_k(self.pipeline_df_, measure=measure, config=config)
    
    def plot_consensus_matrix(
        self,
        measure: str = "stability",
        *,
        rule: str = "max",
        k: int = None,
        figsize: Tuple[int, int] = (10, 8),
        decimals: int = 4,
        show_grid: bool = True,
        legend_outside: bool = True,
    ) -> None:
        if self.model_df_ is None or self.model_df_.empty:
            warnings.warn("model_df_ is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
            return
        
        if rule not in {"max", "1se", "quantile"}:
            raise ValueError("rule must be 'max', '1se', or 'quantile'")
        
        config = PlotConfig(
            figsize=figsize,
            decimals=decimals,
            show_grid=show_grid,
            legend_outside=legend_outside
        )
        
        plot_consensus_matrix(
            model_df=self.model_df_, 
            consensus_mats_raw=self.consensus_mats_raw_, 
            measure=measure, 
            rule=rule, 
            k=k, 
            config=config
        )    
        
    def plot_clustering(
        self,
        measure: str = "stability",
        sample_metric: str = "gini",
        *,
        k: Optional[int] = None,
        rule: str = "max",
        figsize: Tuple[int, int] = (10, 8),
        decimals: int = 4,
        show_grid: bool = True,
        legend_outside: bool = True,
        width: int = 1000,
        height: int = 800,
        min_size: float = 20.0,
        max_size: float = 180.0,
        min_alpha: float = 0.30,
        max_alpha: float = 1.00,
        interactive: bool = True,
        auto_display: bool = True
    ) -> None:
        if self.model_df_ is None or self.model_df_.empty:
            warnings.warn("model_df_ is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
            return
        
        if rule not in {"max", "1se", "quantile"}:
            raise ValueError("rule must be 'max', '1se', or 'quantile'")
        
        config = PlotConfig(
            figsize=figsize,
            decimals=decimals,
            show_grid=show_grid,
            legend_outside=legend_outside
        )
        
        model_df_copy = self.model_df_.copy()
        if k is not None:
            model_df_copy = model_df_copy[model_df_copy['n_clusters'] == k]

        y_col = MEASURE_MAP[measure]
        se_col = f"{y_col}_se"
        if rule == "1se" and se_col not in model_df_copy.columns:
            warnings.warn(f"{se_col!r} not found; falling back to 'max' rule.", RuntimeWarning, stacklevel=2)
            rule = "max"

        # pick best row
        idx = get_best_row(model_df_copy, measure=measure, rule=rule, return_idx=True)
        row = model_df_copy.loc[idx]

        pos = self.model_df_.index.get_loc(idx)

        # labels
        labels = self.get_optimal_labels(measure=measure, rule=rule, k=k)
        n_clusters = int(row['n_clusters'])
        assert len(np.unique(labels)) == n_clusters, \
            f"labels has {len(np.unique(labels))} clusters, expected {n_clusters}"

        # sample-level vectors
        if MEASURE_MAP[measure] == "ari_stability":
            if sample_metric == "gini":
                sample_level_measures = self.stab_gini_arrs_[pos]
            elif sample_metric == "ce":
                sample_level_measures = self.stab_ce_arrs_[pos]
            else:
                raise ValueError("stab_measure must be 'gini' or 'ce'")
        else:
            sample_level_measures = self.generalizability_arrs_[pos]

        if interactive:
            fig = plot_clustering_interactive(
                X=self.X_, 
                row=row,
                labels=labels,
                stab_gini_vec=self.stab_gini_arrs_[pos],
                stab_ce_vec=self.stab_ce_arrs_[pos],
                gen_vec=self.generalizability_arrs_[pos],
                measure=measure,
                width=width,
                height=height,
                min_size=min_size,
                max_size=max_size,
                min_alpha=min_alpha,
                max_alpha=max_alpha,
                auto_display=auto_display
            )
            return fig

        else:
            plot_clustering(
                X=self.X_, 
                row=row,
                labels=labels,
                sample_level_measures=sample_level_measures,
                measure=measure,
                config=config,
                min_size=min_size,
                max_size=max_size,
                min_alpha=min_alpha,
                max_alpha=max_alpha
            )
    
    def plot_consensus_clustering(
        self,
        measure: str = "stability",
        sample_metric: str = "gini",
        *,
        k: Optional[int] = None,
        rule: str = "max",
        figsize: Tuple[int, int] = (10, 8),
        decimals: int = 4,
        show_grid: bool = True,
        legend_outside: bool = True,
        min_size: float = 20.0,
        max_size: float = 180.0,
        min_alpha: float = 0.30,
        max_alpha: float = 1.00
    ) -> None:
        if self.model_df_ is None or self.model_df_.empty:
            warnings.warn("model_df_ is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
            return
        
        if rule not in {"max", "1se", "quantile"}:
            raise ValueError("rule must be 'max', '1se', or 'quantile'")
        
        config = PlotConfig(
            figsize=figsize,
            decimals=decimals,
            show_grid=show_grid,
            legend_outside=legend_outside
        )
        
        model_df_copy = self.model_df_.copy()
        if k is not None:
            model_df_copy = model_df_copy[model_df_copy['n_clusters'] == k]

        y_col = MEASURE_MAP[measure]
        se_col = f"{y_col}_se"
        if rule == "1se" and se_col not in model_df_copy.columns:
            warnings.warn(f"{se_col!r} not found; falling back to 'max' rule.", RuntimeWarning, stacklevel=2)
            rule = "max"

        # pick best row
        idx = get_best_row(model_df_copy, measure=measure, rule=rule, return_idx=True)
        row = model_df_copy.loc[idx]

        pos = self.model_df_.index.get_loc(idx)

        # labels
        labels = self.get_optimal_labels(measure=measure, rule=rule, k=k)
        n_clusters = int(row['n_clusters'])
        assert len(np.unique(labels)) == n_clusters, \
            f"labels has {len(np.unique(labels))} clusters, expected {n_clusters}"

        # sample-level vectors
        if MEASURE_MAP[measure] == "ari_stability":
            if sample_metric == "gini":
                sample_level_measures = self.stab_gini_arrs_[pos]
            elif sample_metric == "ce":
                sample_level_measures = self.stab_ce_arrs_[pos]
            else:
                raise ValueError("stab_measure must be 'gini' or 'ce'")
        else:
            sample_level_measures = self.generalizability_arrs_[pos]
        
        plot_consensus_clustering(
                X=self.X_, 
                row=row,
                labels=labels,
                sample_level_measures=sample_level_measures,
                consensus_mat_raw=self.consensus_mats_raw_[pos],
                measure=measure,
                config=config,
                min_size=min_size,
                max_size=max_size,
                min_alpha=min_alpha,
                max_alpha=max_alpha
        )    