"""Model selection utilities for CARVE results."""

import inspect
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin

from ._types import GridSpec, Measure, Rule

# Maps short measure aliases to canonical column names in estimator_results_
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


def select_best_row_by_rule(
    results_df: pd.DataFrame,
    measure: Measure,
    rule: Rule,
    return_idx: bool = False,
) -> pd.Series:
    """Select the best row according to a selection rule.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results table with metric columns.
    measure : Measure
        Metric key (see ``MEASURE_MAP``).
    rule : Rule
        Selection rule. ``"max"`` picks the row with the highest score.
        ``"1se"`` picks the largest *k* within one standard error of the
        best score. ``"quantile"`` picks the largest *k* within the best
        score's quantile bounds.
    return_idx : bool, default=False
        If True, return the selected row index instead of the row itself.

    Returns
    -------
    row : pandas.Series or int
        Selected row (or row index if ``return_idx=True``).
    """
    y_col = MEASURE_MAP[measure]
    se_col = f"{y_col}_se"
    has_se = se_col in results_df.columns
    quant_col = f"{y_col}_upper"
    has_quant = quant_col in results_df.columns

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
        return select_best_row_max(results_df, measure=measure, return_idx=return_idx)
    if rule == "1se":
        return select_best_row_1se(results_df, measure=measure, return_idx=return_idx)
    if rule == "quantile":
        return select_best_row_quantile(
            results_df, measure=measure, return_idx=return_idx
        )

    raise ValueError(f"Unknown rule: {rule}")


def select_best_estimator(
    results_df: pd.DataFrame,
    estimator_param_grids: list[GridSpec],
    measure: Measure = "stability",
    rule: Rule = "max",
    k: int | None = None,
) -> ClusterMixin:
    """Reconstruct the best estimator from a results table.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results table.
    estimator_param_grids : list of tuple
        Estimator classes and parameter grids.
    measure : Measure, default="stability"
        Metric key used for selection. Common aliases: ``"stability"`` /
        ``"s"``, ``"generalizability"`` / ``"g"``, ``"average"`` /
        ``"avg"``, ``"pac"``, ``"gini"``, ``"ce"``,
        ``"misclassification"``.
    rule : Rule, default="max"
        Selection rule.
    k : int or None, default=None
        Optional fixed number of clusters to filter by.

    Returns
    -------
    estimator : ClusterMixin
        Instantiated estimator with parameters from the selected row.
    """
    if k is not None:
        results_df = results_df[results_df["n_clusters"] == k]

    row = select_best_row_by_rule(
        results_df, measure=measure, rule=rule, return_idx=False
    )
    return build_estimator_from_row(estimator_param_grids, row)


def select_best_k(
    results_df: pd.DataFrame,
    measure: Measure = "stability",
    rule: Rule = "max",
) -> int:
    """Select the best number of clusters from results.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results table.
    measure : Measure, default="stability"
        Metric key used for selection. Common aliases: ``"stability"`` /
        ``"s"``, ``"generalizability"`` / ``"g"``, ``"average"`` /
        ``"avg"``, ``"pac"``, ``"gini"``, ``"ce"``,
        ``"misclassification"``.
    rule : Rule, default="max"
        Selection rule.

    Returns
    -------
    k : int
        Selected number of clusters.
    """
    row = select_best_row_by_rule(
        results_df, measure=measure, rule=rule, return_idx=False
    )
    return row["n_clusters"]


def select_best_row_max(
    results_df: pd.DataFrame,
    *,
    measure: Measure = "stability",
    return_idx: bool = False,
) -> pd.Series:
    """Select the row with the maximum score for a measure.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results table.
    measure : Measure, default="stability"
        Metric key.
    return_idx : bool, default=False
        If True, return the row index instead of the row.

    Returns
    -------
    row : pandas.Series or int
        Selected row (or row index if ``return_idx=True``).
    """
    measure_col = MEASURE_MAP[measure]

    if return_idx:
        return results_df[measure_col].idxmax()

    return results_df.loc[results_df[measure_col].idxmax()]


def select_best_row_1se(
    results_df: pd.DataFrame,
    *,
    measure: Measure = "stability",
    return_idx: bool = False,
) -> pd.Series:
    """Select the model with the largest k within 1 SE of the best score.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results table.
    measure : Measure, default="stability"
        Metric key.
    return_idx : bool, default=False
        If True, return the row index instead of the row.

    Returns
    -------
    row : pandas.Series or int
        Selected row (or row index if ``return_idx=True``).
    """
    if measure not in MEASURE_MAP:
        raise ValueError(f"Invalid measure {measure!r}. Options: {list(MEASURE_MAP)}")
    measure_col = MEASURE_MAP[measure]

    best_row = results_df.loc[results_df[measure_col].idxmax()]
    best_score = best_row[measure_col]
    se = best_row[f"{measure_col}_se"]
    threshold = best_score - se

    within_1se = results_df[results_df[measure_col] >= threshold]

    if return_idx:
        return within_1se["n_clusters"].idxmax()

    return within_1se.loc[within_1se["n_clusters"].idxmax()]


def select_best_row_quantile(
    results_df: pd.DataFrame,
    *,
    measure: Measure = "stability",
    return_idx: bool = False,
) -> pd.Series:
    """Select the model with the largest k within the best score's quantile bounds.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results table.
    measure : Measure, default="stability"
        Metric key.
    return_idx : bool, default=False
        If True, return the row index instead of the row.

    Returns
    -------
    row : pandas.Series or int
        Selected row (or row index if ``return_idx=True``).
    """
    if measure not in MEASURE_MAP:
        raise ValueError(f"Invalid measure {measure!r}. Options: {list(MEASURE_MAP)}")
    measure_col = MEASURE_MAP[measure]

    best_row = results_df.loc[results_df[measure_col].idxmax()]
    threshold_upper = best_row[f"{measure_col}_upper"]
    threshold_lower = best_row[f"{measure_col}_lower"]

    within_quantiles = results_df.loc[
        (results_df[measure_col] >= threshold_lower)
        & (results_df[measure_col] <= threshold_upper)
    ]

    if within_quantiles.empty:
        warnings.warn(
            "No estimators within quantile thresholds; falling back to max.",
            RuntimeWarning,
            stacklevel=2,
        )
        if return_idx:
            return results_df[measure_col].idxmax()
        return results_df.loc[results_df[measure_col].idxmax()]

    if return_idx:
        return within_quantiles["n_clusters"].idxmax()

    return within_quantiles.loc[within_quantiles["n_clusters"].idxmax()]


def build_estimator_from_row(
    estimator_param_grids: list[GridSpec],
    row: pd.Series,
) -> ClusterMixin:
    """Instantiate an estimator from a results table row.

    Parameters
    ----------
    estimator_param_grids : list of tuple
        Estimator classes and parameter grids.
    row : pandas.Series
        Selected configuration row.

    Returns
    -------
    estimator : ClusterMixin
        Instantiated estimator.
    """
    est_name = row["estimator"]

    est_class = next(
        cls for cls, _ in estimator_param_grids if cls.__name__ == est_name
    )

    valid_keys = get_estimator_param_names(est_class)
    params = row_to_estimator_params(row, valid_keys)

    return est_class(**params)


def get_estimator_param_names(est_class: type[ClusterMixin]) -> set[str]:
    """Return valid parameter names for an estimator class.

    Parameters
    ----------
    est_class : type
        Estimator class.

    Returns
    -------
    names : set of str
        Valid parameter names.
    """
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
            and p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        return names


def row_to_estimator_params(
    row: pd.Series,
    valid_keys: set[str],
) -> dict[str, Any]:
    """Extract estimator parameters from a results row.

    Parameters
    ----------
    row : pandas.Series
        Selected configuration row.
    valid_keys : set of str
        Parameter names allowed for the estimator.

    Returns
    -------
    params : dict
        Parameter dictionary with None/NaN filtered out.
    """
    params = {}

    for k in valid_keys:
        if k in row.index:
            v = row[k]

            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue

            params[k] = v

    return params
