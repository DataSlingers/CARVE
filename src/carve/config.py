"""Configuration objects for CARVE validation."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin, TransformerMixin

GridSpec = Tuple[Type[ClusterMixin], Dict[str, List[Any]]]
PreprocSpec = Union[
    Tuple[Callable, Dict[str, Any]],        # (Method, params)
    Tuple[Callable, str, Dict[str, Any]],   # (Method, name, params)
]

@dataclass(frozen=True)
class ValidatorConfig:
    """Immutable configuration for CARVE validation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_clusters : int or ndarray, default=10
        Number(s) of clusters to evaluate.
    n_resamples : int, default=100
        Number of resampling iterations.
    subsample_ratio : float, default=0.6
        Subsampling proportion.
    estimator_param_grids : list of tuple, optional
        Estimator classes and parameter grids.
    normalization_options : list, optional
        Normalization options for preprocessing.
    dim_reduction_options : list, optional
        Dimensionality reduction options for preprocessing.
    reference_labels : ndarray or None, optional
        Reference labels for aligning clustering outputs.
    n_jobs : int, default=cpu_count()-1
        Parallelism for resampling.
    random_state : int or None, default=None
        Random seed.
    """
    X: np.ndarray
    n_clusters: Union[int, np.ndarray] = 10
    n_resamples: int = 100
    subsample_ratio: float = 0.6
    estimator_param_grids: Optional[List[GridSpec]] = None
    normalization_options: Optional[List[PreprocSpec]] = None
    dim_reduction_options: Optional[List[PreprocSpec]] = None
    reference_labels: Optional[np.ndarray] = None
    n_jobs: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    random_state: Optional[int] = None
    
    def to_params(self, deep: bool = True) -> Dict[str, Any]:
        """Serialize configuration to a parameter dictionary.

        Parameters
        ----------
        deep : bool, default=True
            If True, include nested details for estimator and preprocessor grids.

        Returns
        -------
        params : dict
            Dictionary representation of the configuration.
        """
        params = {
            "n_clusters": self.n_clusters,
            "n_resamples": self.n_resamples,
            "subsample_ratio": self.subsample_ratio,
            "estimator_param_grids": self.estimator_param_grids,
            "normalization_options": self.normalization_options,
            "dim_reduction_options": self.dim_reduction_options,
            "reference_labels": self.reference_labels,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state
        }
        
        if deep:
            # Handle nested objects for estimator_param_grids
            if self.estimator_param_grids:
                params["estimator_param_grids"] = [
                    {
                        "model": grid[0].__name__,
                        "params": grid[1]
                    }
                    for grid in self.estimator_param_grids
                ]
            
            # Handle nested objects for normalization_options
            if self.normalization_options:
                params["normalization_options"] = [
                    {
                        "transformer": norm[0].__name__,
                        "params": norm[1]
                    }
                    for norm in self.normalization_options
                ]
            
            # Handle nested objects for dim_reduction_options
            if self.dim_reduction_options:
                params["dim_reduction_options"] = [
                    {
                        "transformer": dr[0].__name__,
                        "params": dr[1]
                    }
                    for dr in self.dim_reduction_options
            ]
    
        return params
    
    def update(self, **params: Any) -> ValidatorConfig:
        """Return a new configuration with updated fields.

        Parameters
        ----------
        **params : dict
            Fields to update.

        Returns
        -------
        config : ValidatorConfig
            New configuration instance with updated values.
        """
        return replace(self, **params)