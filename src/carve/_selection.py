import inspect
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Type
import pandas as pd
import numpy as np 
from sklearn.base import ClusterMixin

GridSpec = Tuple[Type[ClusterMixin], Dict[str, List[Any]]]
Measure = Literal[
    "s",
    "stab",
    "stability",
    "ari_stability",
    
    "g",
    "gen",
    "generalizability",
    "ari_generalizability",
    
    "avg",
    "average",
    "ari_average",
    
    "pac",
    "consensus_pac_stability",
    "consensus_pac_stability",

    "gini",
    "consensus_gini_stability",
    "consensus_gini_stability",

    "ce",
    "consensus_ce_stability",
    "consensus_ce_stability",

    "misclass",
    "misclassification",
    "misclassification_generalizability",
    "misclassification_generalizability",
]

Rule = Literal["max", "1se", "quantile"]

MEASURE_MAP = {
    "s": "ari_stability",
    "stab": "ari_stability",
    "stability": "ari_stability",
    "ari_stability": "ari_stability",
    
    "g": "ari_generalizability",
    "gen": "ari_generalizability",
    "generalizability": "ari_generalizability",
    "ari_generalizability": "ari_generalizability",
    
    "avg": "ari_average",
    "average": "ari_average",
    "ari_average": "ari_average",
    
    "pac": "consensus_pac_stability",
    "consensus_pac_stability": "consensus_pac_stability",
    
    "gini": "consensus_gini_stability",
    "consensus_gini_stability": "consensus_gini_stability",
    
    "ce": "consensus_ce_stability",
    "consensus_ce_stability": "consensus_ce_stability",
    
    "misclass": "misclassification_generalizability",
    "misclassification": "misclassification_generalizability",
    "misclassification_generalizability": "misclassification_generalizability",
}


def get_best_row(
    model_df: pd.DataFrame, 
    measure: Measure, 
    rule: Rule, 
    return_idx: bool = False
) -> pd.Series:
    y_col = MEASURE_MAP[measure]
    se_col = f"{y_col}_se"
    has_se = se_col in model_df.columns
    quant_col = f"{y_col}_upper"
    has_quant = quant_col in model_df.columns
    
    if rule == "1se" and not has_se:
        warnings.warn(
            f"Column {se_col!r} not found; falling back to 'max' rule.",
            RuntimeWarning,
            stacklevel=2,
        )
        rule = "max"
    elif rule == "quantile" and not has_quant:
        warnings.warn(
            f"Column {quant_col!r} not found; falling back to 'max' rule.",
            RuntimeWarning,
            stacklevel=2,
        )
        rule = "max"
            
    if rule == "max":
        return select_best_row(model_df, measure=measure, return_idx=return_idx)
    if rule == "1se":
        return select_best_row_1se(model_df, measure=measure, return_idx=return_idx)
    if rule == "quantile":
        return select_best_row_quantile(model_df, measure=measure, return_idx=return_idx)

    raise ValueError(f"Unknown rule: {rule}")

def select_best_estimator(
    model_df: pd.DataFrame,
    model_grids: List[GridSpec],
    measure: Measure = "stability",
    rule: str = "max",
    k: Optional[int] = None,
) -> ClusterMixin:
    if k is not None:  # subset dataframe if k specified
        model_df = model_df[model_df['n_clusters'] == k]
    
    # Select the row with the highest value for the specified measure
    row = get_best_row(model_df, measure=measure, rule=rule, return_idx=False)

    # Reconstruct the best estimator
    estimator = instantiate_estimator(model_grids, row)
    return estimator

def select_best_k(
    model_df: pd.DataFrame,
    measure: Measure = "stability",
    rule: str = "max"
) -> ClusterMixin:
    # Select the row with the highest value for the specified measure
    row = get_best_row(model_df, measure=measure, rule=rule, return_idx=False)

    # Reconstruct the best estimator
    return row['n_clusters']

def select_best_row(
    model_df: pd.DataFrame,
    *,
    measure: Measure = "stability",
    return_idx: bool = False,
) -> pd.Series:
    measure_col = MEASURE_MAP[measure]
    
    if return_idx:
        return model_df[measure_col].idxmax()
    
    return model_df.loc[model_df[measure_col].idxmax()]

def select_best_row_1se(
    model_df: pd.DataFrame,
    *,
    measure: Measure = "stability",
    return_idx: bool = False,
) -> pd.Series:
    if measure not in MEASURE_MAP:
        raise ValueError(f"Invalid measure {measure!r}. Options: {list(MEASURE_MAP)}")
    measure_col = MEASURE_MAP[measure]
    
    best_row = model_df.loc[model_df[measure_col].idxmax()]
    best_score = best_row[measure_col]
    se = best_row[f"{measure_col}_se"]  # Standard error column for the measure
    
    # Define 1 SE threshold
    threshold = best_score - se
    
    # Filter models within 1SE of best score
    within_1se = model_df[model_df[measure_col] >= threshold]
    
    if return_idx:
        return within_1se["n_clusters"].idxmax()
        
    return within_1se.loc[within_1se["n_clusters"].idxmax()]

def select_best_row_quantile(
    model_df: pd.DataFrame,
    *,
    measure: Measure = "stability",
    return_idx: bool = False,
) -> pd.Series:
    if measure not in MEASURE_MAP:
        raise ValueError(f"Invalid measure {measure!r}. Options: {list(MEASURE_MAP)}")
    measure_col = MEASURE_MAP[measure]
    
    best_row = model_df.loc[model_df[measure_col].idxmax()]
    threshold_upper = best_row[f"{measure_col}_upper"]
    threshold_lower = best_row[f"{measure_col}_lower"]
    
    # Filter models within 1SE of best score
    within_quantiles = model_df.loc[
        (model_df[measure_col] >= threshold_lower) & (model_df[measure_col] <= threshold_upper)
    ]
    
    # Fallback if no rows are found
    if within_quantiles.empty:
        warnings.warn("No models within quantile thresholds; falling back to max.", RuntimeWarning, stacklevel=2)
        if return_idx:
            return model_df[measure_col].idxmax()
        return model_df.loc[model_df[measure_col].idxmax()]
    
    if return_idx:
        return within_quantiles["n_clusters"].idxmax()
        
    return within_quantiles.loc[within_quantiles["n_clusters"].idxmax()]

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
