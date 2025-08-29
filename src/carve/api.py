from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin

from .config import ValidatorConfig
from .grids import default_model_grids, default_norm_options, default_dr_options
from ._runner import run_validation
from ._consensus import compute_consensus_metrics_batch
from ._selection import select_best_estimator
from ._plotting import plot_measure_vs_k
from ._utils import align_labels, ensure_array2d, wrangle_pipeline_records

@dataclass
class CARVE:
    config: ValidatorConfig

    # outputs (populated by validate)
    model_df: Optional[pd.DataFrame] = field(default=None, init=False)
    pipeline_df: Optional[pd.DataFrame] = field(default=None, init=False)
    consensus_mats: Optional[List[np.ndarray]] = field(default=None, init=False)  
    consensus_mats_raw: Optional[List[np.ndarray]] = field(default=None, init=False)
    stab_gini_arr: Optional[np.ndarray] = field(default=None, init=False)
    stab_ce_arr: Optional[np.ndarray] = field(default=None, init=False)
    mis_arrs: Optional[List[np.ndarray]] = field(default=None, init=False)
    
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
            self.consensus_mats, self.consensus_mats_raw, self.mis_arrs
        ) = run_validation(
            X=X,
            model_grids=model_grids,
            B=self.config.B,
            rho=self.config.rho,
            norm_options=norm_options,
            dr_options=dr_options,
            ref_labels=self.config.ref_labels,
            random_preprocess=random_preprocess,
            n_jobs=self.config.n_jobs,
            random_state=random_state,
            prog_bar=prog_bar
        )
        
        self.model_df = pd.DataFrame.from_records(model_records)
        self.pipeline_df = None if not random_preprocess else wrangle_pipeline_records(pipeline_records)

        # compute stability vectors from consensus matrices (pure, vectorized)
        gini_list, ce_list, pac_list = compute_consensus_metrics_batch(self.consensus_mats_raw)

        self.stab_gini_arr = np.vstack(gini_list)
        self.stab_ce_arr = np.vstack(ce_list)
        self.model_df["consensus_pac_stability"] = pac_list
    
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
            self.config = self.config.updated(ref_labels=labels)
        else:
            labels = align_labels(self.config.ref_labels, labels)

        if return_estimator:
            return labels, estimator
        
        return labels
    
    def plot_global(
        self,
        measure: str = "stability",
        rule: str = "max",
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        plot_measure_vs_k(
            model_df=self.model_df,
            measure=measure,
            rule=rule,
            figsize=figsize
        )
        