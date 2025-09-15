from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin

from .config import ValidatorConfig
from .grids import default_model_grids, default_norm_options, default_dr_options
from ._runner import run_validation
from ._consensus import compute_consensus_metrics_batch
from ._selection import select_best_estimator, select_best_row, select_best_row_1se, select_best_k, MEASURE_MAP
from ._plotting import plot_measure_vs_k, plot_consensus_matrix, plot_clustering
from ._utils import align_labels, ensure_array2d, wrangle_pipeline_records

@dataclass
class CARVE:
    config: ValidatorConfig

    # outputs (populated by validate)
    model_df: Optional[pd.DataFrame] = field(default=None, init=False)
    pipeline_df: Optional[pd.DataFrame] = field(default=None, init=False)
    consensus_mats_raw: Optional[List[np.ndarray]] = field(default=None, init=False)
    stab_gini_arrs: Optional[List[np.ndarray]] = field(default=None, init=False)
    stab_ce_arrs: Optional[List[np.ndarray]] = field(default=None, init=False)
    generalizability_arrs: Optional[List[np.ndarray]] = field(default=None, init=False)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self.config.to_params(deep)
    
    def set_params(self, **params: Any) -> CARVE:
        # returns a new instance with updated immutable config
        new_config = self.config.update(**params)
        obj = type(self)(new_config)
        return obj
    
    def validate(
        self, 
        *, 
        random_preprocess: bool = False, 
        prog_bar: bool = True, 
        random_state: Optional[int] = None
    ) -> None:
        X = ensure_array2d(self.config.X)
        model_grids = self.config.model_grids or default_model_grids(X, self.config.K)
        norm_options = self.config.norm_options or default_norm_options()
        dr_options = self.config.dr_options or default_dr_options(X, self.config.rho)
        
        (
            model_records, pipeline_records,
            self.consensus_mats_raw,
            self.generalizability_arrs
        ) = run_validation(
            X=X,
            model_grids=model_grids,
            B=self.config.B,
            rho=self.config.rho,
            norm_options=norm_options,
            dr_options=dr_options,
            random_preprocess=random_preprocess,
            n_jobs=self.config.n_jobs,
            random_state=random_state,
            prog_bar=prog_bar
        )
        
        self.model_df = pd.DataFrame.from_records(model_records)
        self.pipeline_df = None if not random_preprocess else wrangle_pipeline_records(pipeline_records)

        # compute stability vectors from consensus matrices
        gini_list, ce_list, pac_list = compute_consensus_metrics_batch(self.consensus_mats_raw)

        self.stab_gini_arr = np.vstack(gini_list)
        self.stab_ce_arr = np.vstack(ce_list)
        
        self.model_df["consensus_pac_stability"] = pac_list
        self.model_df["consensus_gini_stability"] = np.array([np.mean(arr) for arr in self.stab_gini_arr])
        self.model_df["consensus_ce_stability"] = np.array([np.mean(arr) for arr in self.stab_ce_arr])

    def get_optimal_labels(
        self,
        *,
        measure: str = "stability",
        rule: str = 'max',
        k: Optional[int] = None,
        return_estimator: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ClusterMixin]]:
        model_grids = self.config.model_grids or default_model_grids(self.config.X, self.config.K)
        
        if self.model_df is None:
            raise RuntimeError("Call fit() first.")

        estimator = select_best_estimator(
            self.model_df, 
            model_grids=model_grids, 
            measure=measure, rule=rule,
            k=k
        )
        
        if hasattr(estimator, "fit_predict"):
            labels = estimator.fit_predict(self.config.X)
        else:
            estimator.fit(self.config.X)
            labels = getattr(estimator, "labels_")

        if self.config.ref_labels is None:
            self.config = self.config.update(ref_labels=labels)
        else:
            labels = align_labels(self.config.ref_labels, labels)

        if return_estimator:
            return labels, estimator
        
        return labels
    
    def get_optimal_k(
        self,
        *,
        measure: str = "stability",
        rule: str = 'max',
    ) -> int:
        if self.model_df is None:
            raise RuntimeError("Call fit() first.")

        k = select_best_k(
            self.model_df, 
            measure=measure, rule=rule
        )
        
        return k
    
    def plot_global(
        self,
        measure: str = "stability",
        rule: str = "max",
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        if self.model_df is None or self.model_df.empty:
            warnings.warn("model_df is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
            return
        if measure not in MEASURE_MAP:
            raise ValueError(f"Invalid measure {measure!r}. Options: {list(MEASURE_MAP)}")
        if rule not in {"max", "1se"}:
            raise ValueError("rule must be 'max' or '1se'")
        
        y_col = MEASURE_MAP[measure]
        se_col = f"{y_col}_se"
        has_se = se_col in self.model_df.columns
        
        if rule == "1se" and not has_se:
            warnings.warn(
                f"Column {se_col!r} not found; falling back to 'max' rule.",
                RuntimeWarning,
                stacklevel=2,
            )
            rule = "max"
        
        plot_measure_vs_k(
            self.model_df,
            measure=measure,
            rule=rule,
            figsize=figsize
        )
    
    def plot_consensus_matrix(
        self,
        measure: str = "stability",
        rule: str = "max",
        k: int = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        if self.model_df is None or self.model_df.empty:
            warnings.warn("model_df is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
            return
        if measure not in MEASURE_MAP:
            raise ValueError(f"Invalid measure {measure!r}. Options: {list(MEASURE_MAP)}")
        if rule not in {"max", "1se"}:
            raise ValueError("rule must be 'max' or '1se'")
        
        y_col = MEASURE_MAP[measure]
        se_col = f"{y_col}_se"
        has_se = se_col in self.model_df.columns
        
        if rule == "1se" and not has_se:
            warnings.warn(
                f"Column {se_col!r} not found; falling back to 'max' rule.",
                RuntimeWarning,
                stacklevel=2,
            )
            rule = "max"
        
        plot_consensus_matrix(
            self.model_df,
            self.consensus_mats_raw,
            measure=measure, 
            rule=rule,
            k=k,
            figsize=figsize
        )
    
    def plot_clustering(
        self,
        measure: str = "stability",
        stab_measure: str = "gini",
        rule: str = "max",
        k: int = None,
        figsize: Tuple[int, int] = (20, 8),
        min_size: float = 20.0,
        max_size: float = 180.0, 
        min_alpha: float = 0.3, 
        max_alpha: float = 1.0
    ) -> None:
        if self.model_df is None or self.model_df.empty:
            warnings.warn("model_df is empty; nothing to plot. Run validate() first.", RuntimeWarning, stacklevel=2)
            return
        if measure not in MEASURE_MAP:
            raise ValueError(f"Invalid measure {measure!r}. Options: {list(MEASURE_MAP)}")
        if rule not in {"max", "1se"}:
            raise ValueError("rule must be 'max' or '1se'")
        
        y_col = MEASURE_MAP[measure]
        se_col = f"{y_col}_se"
        has_se = se_col in self.model_df.columns
        
        if rule == "1se" and not has_se:
            warnings.warn(
                f"Column {se_col!r} not found; falling back to 'max' rule.",
                RuntimeWarning,
                stacklevel=2,
            )
            rule = "max"
        
        model_df_copy = self.model_df.copy()
        if k is not None:  # subset dataframe if k specified
            model_df_copy = model_df_copy[model_df_copy['n_clusters'] == k]

        # pick best method
        idx = select_best_row(model_df_copy, measure=measure, return_idx=True) if rule == "max" else select_best_row_1se(model_df_copy, measure=measure, return_idx=True)
        row = model_df_copy.loc[idx]
        
        labels = self.get_optimal_labels(
            measure=measure,
            rule=rule,
            k=k
        )
        
        n_clusters = row['n_clusters']
        assert len(np.unique(labels)) == n_clusters, f"labels has {len(np.unique(labels))} clusters, expected {n_clusters}"
        
        if MEASURE_MAP[measure] == "ari_stability":
            if stab_measure == "gini":
                sample_level_measures = self.stab_gini_arr[idx]
            elif stab_measure == "ce":
                sample_level_measures = self.stab_ce_arr[idx]
            else:
                raise ValueError("stab_measure must be 'gini' or 'ce'")
        else:
            sample_level_measures = self.generalizability_arrs[idx]

        plot_clustering(
            X=self.config.X,
            row=row, 
            labels=labels, 
            sample_level_measures=sample_level_measures,
            measure=measure, 
            figsize=figsize, 
            min_size=min_size,
            max_size=max_size,
            min_alpha=min_alpha,
            max_alpha=max_alpha
        )

