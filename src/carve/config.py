from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import multiprocessing as mp
import numpy as np
from sklearn.base import ClusterMixin, TransformerMixin

GridSpec = Tuple[Type[ClusterMixin], Dict[str, List[Any]]]
PreprocSpec = Tuple[Callable[..., TransformerMixin], Dict[str, List[Any]]]

@dataclass(frozen=True)
class ValidatorConfig:
    X: np.ndarray
    K: Union[int, np.ndarray] = 10
    B: int = 100
    rho: float = 0.6
    model_grids: Optional[List[GridSpec]] = None
    norm_options: Optional[List[PreprocSpec]] = None
    dr_options: Optional[List[PreprocSpec]] = None
    ref_labels: Optional[np.ndarray] = None
    n_jobs: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    random_state: Optional[int] = None
    
    def to_params(self, deep: bool = True) -> Dict[str, Any]:
        params = {
            "K": self.K,
            "B": self.B,
            "rho": self.rho,
            "model_grids": self.model_grids,
            "norm_options": self.norm_options,
            "dr_options": self.dr_options,
            "ref_labels": self.ref_labels,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state
        }
        
        if deep:
            # Handle nested objects for model_grids
            if self.model_grids:
                params["model_grids"] = [
                    {
                        "model": grid[0].__name__,
                        "params": grid[1]
                    }
                    for grid in self.model_grids
                ]
            
            # Handle nested objects for norm_options
            if self.norm_options:
                params["norm_options"] = [
                    {
                        "transformer": norm[0].__name__,
                        "params": norm[1]
                    }
                    for norm in self.norm_options
                ]
            
            # Handle nested objects for dr_options
            if self.dr_options:
                params["dr_options"] = [
                    {
                        "transformer": dr[0].__name__,
                        "params": dr[1]
                    }
                    for dr in self.dr_options
            ]
    
        return params
    
    def update(self, **params: Any) -> ValidatorConfig:
        return replace(self, **params)