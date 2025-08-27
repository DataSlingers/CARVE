import inspect
from typing import Any, Dict, List, Optional, Tuple, Type
import pandas as pd
import numpy as np 
from sklearn.base import ClusterMixin

GridSpec = Tuple[Type[ClusterMixin], Dict[str, List[Any]]]

MEASURE_MAP = {
        "stability": "ari_stability",
        "generalizability": "ari_generalizability",
    }

def select_best_estimator(
    method_df: pd.DataFrame,
    model_grids: List[GridSpec],
    measure: str = "stability",
    k: Optional[int] = None,
) -> ClusterMixin:
    if k is not None:  # subset dataframe is k specified
        method_df = method_df[method_df['n_clusters'] == k]
        
    if measure not in MEASURE_MAP:
        raise ValueError(f"Invalid measure {measure!r}. Options: {list(MEASURE_MAP)}")
    measure_col = MEASURE_MAP[measure]
    
    # Select the row with the highest value for the specified measure
    best_row = method_df.loc[method_df[measure_col].idxmax()]
    
    # Reconstruct the best estimator
    estimator = instantiate_estimator(model_grids, best_row)
    return estimator

def select_best_estimator_1se(
    method_df: pd.DataFrame,
    model_grids: List[GridSpec],
    measure: str = "stability",
    k: Optional[int] = None,
) -> ClusterMixin:
    if k is not None:  # subset dataframe is k specified
        method_df = method_df[method_df['n_clusters'] == k]
        
    if measure not in MEASURE_MAP:
        raise ValueError(f"Invalid measure {measure!r}. Options: {list(MEASURE_MAP)}")
    measure_col = MEASURE_MAP[measure]
    
    best_row = method_df.loc[method_df[measure_col].idxmax()]
    best_score = best_row[measure_col]
    se = best_row[f"{measure_col}_se"]  # Standard error column for the measure
    
    # Define 1 SE threshold
    threshold = best_score - se
    
    # Filter models within 1 SE of best score
    within_1se = method_df[method_df[measure] >= threshold]
    
    # Select model (most clusters) within 1 SE
    complex_row = within_1se.loc[within_1se["n_clusters"].idxmax()]
    
    # Reconstruct the best estimator
    estimator = instantiate_estimator(model_grids, complex_row)
    return estimator

def instantiate_estimator(
    model_grids: List[GridSpec], 
    row: pd.Series,
) -> ClusterMixin:
    est_name = row["estimator"]
    
    est_class = next(cls for cls, _ in model_grids if cls.__name__ == est_name)
    
    valid_keys = valid_param_names(est_class)
    params = row_to_params(row, valid_keys)

    return est_class(**params)

def valid_param_names(est_class: Type[ClusterMixin]) -> List[str]:
    try:
        est = est_class()
        return set(est.get_params(deep=False).keys())
    
    except Exception:
        # Fallback: inspect the __init__ signature
        sig = inspect.signature(est_class.__init__)
        names = {
            p.name
            for p in sig.parameters.values()
            if p.name != "self"
            and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY
            )
        }
        
        return names
    
def row_to_params(
    row: pd.Series, 
    valid_keys: List[str]
) -> Dict[str, Any]:
    params = {}
    
    for k in valid_keys:
        if k in row.index:
            v = row[k]
            
            if v is None:                               # filter out None ...
                continue
            if isinstance(v, float) and np.isnan(v):    # ... and NaNs
                continue
            
            params[k] = v
            
    return params
