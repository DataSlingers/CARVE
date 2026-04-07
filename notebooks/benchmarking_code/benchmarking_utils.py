import inspect

from typing import Type, Iterable, Any

import numpy as np
import pandas as pd

from carve.cluster import SpectralClusteringCARVE

from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import confusion_matrix

from scipy.optimize import linear_sum_assignment


def align_labels(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """
    Aligns predicted cluster labels to true labels by maximizing agreement using the Hungarian algorithm.

    Args:
        - true_labels (np.ndarray): Ground-truth labels for each sample (integer-coded).
        - pred_labels (np.ndarray): Predicted cluster labels for each sample (arbitrary IDs).

    Returns:
        np.ndarray: Aligned predicted labels with IDs permuted to best match true_labels.
    """
    cm = confusion_matrix(true_labels, pred_labels)

    row_ind, col_ind = linear_sum_assignment(-cm)

    label_map = {old: new for old, new in zip(col_ind, row_ind)}
    return np.array(
        [label_map[lab] if lab in label_map else lab for lab in pred_labels]
    )


def _wilson_ci(k_success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k_success / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def _pick_first(value_or_sequence: Any) -> Any:
    """
    Returns the first element if v is a list/tuple; otherwise returns v unchanged.

    Args:
        - value_or_sequence: Parameter value or list/tuple of candidate values.

    Returns:
        The first element if v is list/tuple; otherwise v.
    """
    return (
        value_or_sequence[0]
        if isinstance(value_or_sequence, (list, tuple))
        else value_or_sequence
    )


def _summary_stats(x: pd.Series, q=(0.05, 0.25, 0.50, 0.75, 0.95)) -> dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if x.empty:
        return {k: np.nan for k in ["mean", "sd", "median", "q05", "q25", "q75", "q95"]}
    qs = x.quantile(list(q))
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "median": float(qs.loc[0.50]),
        "q05": float(qs.loc[0.05]),
        "q25": float(qs.loc[0.25]),
        "q75": float(qs.loc[0.75]),
        "q95": float(qs.loc[0.95]),
    }


def _build_estimator(
    estimator_cls: Type[ClusterMixin],
    n_clusters: int,
    estimator_params: dict[str, list[Any]] | None,
    random_seed: int,
) -> ClusterMixin:
    """
    Builds a clustering estimator with fixed parameters and a specified number of clusters.

    Args:
        - estimator_cls (Type[ClusterMixin]): Estimator class to instantiate.
        - n_clusters (int): Number of clusters for this estimator instance.
        - estimator_params (dict[str, list[Any]] | None): Parameter grid or fixed parameters.
        - random_seed (int): Random seed used when the estimator supports random_state.

    Returns:
        ClusterMixin: Instantiated clustering estimator.
    """
    # start with the fixed params coming from estimator_grids[0][1]
    params = {}
    if estimator_params:
        for key, val in estimator_params.items():
            if key == "n_clusters":
                continue

            params[key] = _pick_first(val)

    # set k for this call
    params["n_clusters"] = n_clusters

    # set random_state when supported
    sig = inspect.signature(estimator_cls.__init__)
    if "random_state" in sig.parameters:
        params["random_state"] = random_seed

    # pin kmeans defaults across sklearn versions
    if estimator_cls.__name__ == "KMeans" or estimator_cls is KMeans:
        if "n_init" in sig.parameters and "n_init" not in params:
            params["n_init"] = 10

    return estimator_cls(**params)


def get_rule(carve_metric: str) -> str:
    """
    Determines the carving rule suffix for a metric name.

    Args:
        - carve_metric (str): Metric name with optional suffixes like '_quant' or '_1se'.

    Returns:
        str: Rule identifier ('quantile', '1se', or 'max').
    """
    if carve_metric.endswith("_quant"):
        rule = "quantile"
    elif carve_metric.endswith("_1se"):
        rule = "1se"
    else:
        rule = "max"

    return rule


def get_measure(carve_metric: str) -> str:
    """
    Extracts the base measure name from a carved metric string.

    Args:
        - carve_metric (str): Metric name with optional suffixes like '_quant' or '_1se'.

    Returns:
        str: Base measure name without suffix.
    """
    if carve_metric.endswith("_1se"):
        measure = carve_metric[:-4]
    elif carve_metric.endswith("_quant"):
        measure = carve_metric[:-6]
    else:
        measure = carve_metric

    return measure


def make_estimator_grids(
    estimator: str,
    candidate_clusters: Iterable[int],
) -> list[tuple[Type[ClusterMixin], dict[str, list[Any]]]]:
    """
    Creates parameter grids for supported clustering estimators.

    Args:
        - estimator (str): Estimator key ('agglomerative', 'spectral', or default to kmeans).
        - candidate_clusters (Iterable[int]): Candidate cluster counts.

    Returns:
        list[tuple]: List of (EstimatorClass, param_grid) tuples.
    """
    if estimator == "agglomerative":
        return [
            (
                AgglomerativeClustering,
                {"n_clusters": list(candidate_clusters), "linkage": ["ward"]},
            )
        ]
    if estimator == "spectral":
        return [
            (
                SpectralClusteringCARVE,
                {"n_clusters": list(candidate_clusters), "affinity": ["self_tuning"]},
            )
        ]
    return [(KMeans, {"n_clusters": list(candidate_clusters), "n_init": [10]})]
