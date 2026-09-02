"""Classical cluster validation indices.

The gap statistic returns its standard error alongside its value so that
selection can apply Tibshirani's rule. The routine this replaces returned
only Gap(k) and the caller took a plain argmax, which gave CARVE a 1SE rule
and its main competitor none.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from ._estimators import build_estimator
from ._registry import N_REFERENCE_DATASETS
from ._types import EstimatorSpec


def compute_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster dispersion W_k for squared Euclidean distance.

    Uses W_k = sum_c sum_{i in C_c} ||x_i - mu_c||^2, which is equivalent to
    the pairwise-distance form and runs in O(n*d) rather than O(n^2*d).
    """
    total = 0.0
    for label in np.unique(labels):
        points = X[labels == label]
        if len(points) <= 1:
            continue
        centroid = points.mean(axis=0)
        total += float(np.sum((points - centroid) ** 2))
    return total


def _uniform_reference(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from the axis-aligned bounding box of X."""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return rng.random(size=X.shape) * (maxs - mins) + mins


def gap_statistic(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    spec: EstimatorSpec,
    n_reference_datasets: int = N_REFERENCE_DATASETS,
    random_state: int = 0,
) -> tuple[float, float]:
    """Gap(k) and its standard error s_k.

    Gap(k) = mean_b log W*_kb - log W_k, and
    s_k = sd(log W*_kb) * sqrt(1 + 1 / n_reference_datasets),
    following Tibshirani, Walther and Hastie (2001). The standard deviation
    is the population form (ddof=0), as in the original.

    Returns
    -------
    (gap, s_k), both nan when W_k is non-positive or non-finite.
    """
    rng = np.random.default_rng(int(random_state))

    W_k = compute_dispersion(X, labels)
    if not np.isfinite(W_k) or W_k <= 0:
        return float("nan"), float("nan")

    k = int(np.unique(labels).size)

    log_dispersions = np.empty(n_reference_datasets, dtype=float)
    for b in range(n_reference_datasets):
        X_ref = _uniform_reference(X, rng)
        seed_b = int(rng.integers(0, 2**32 - 1))
        estimator = build_estimator(spec, n_clusters=k, random_state=seed_b)
        ref_labels = estimator.fit_predict(X_ref)
        log_dispersions[b] = np.log(compute_dispersion(X_ref, ref_labels))

    gap = float(np.mean(log_dispersions) - np.log(W_k))
    s_k = float(
        np.std(log_dispersions, ddof=0) * np.sqrt(1.0 + 1.0 / n_reference_datasets)
    )
    return gap, s_k


def davies_bouldin_inv(X: np.ndarray, labels: np.ndarray) -> float:
    """Inverse Davies-Bouldin index, 1 / (1 + DB), so higher is better."""
    db = davies_bouldin_score(X, labels)
    return 1.0 / (1.0 + db) if np.isfinite(db) else float("nan")


def calculate_cvi(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str,
    *,
    spec: EstimatorSpec,
    random_state: int = 0,
) -> tuple[float, float]:
    """Evaluate one classical index, returning (value, standard_error).

    Only the gap statistic has a meaningful standard error; the others
    report 0.0 so callers can treat every index uniformly.
    """
    if metric == "gap":
        return gap_statistic(X, labels, spec=spec, random_state=random_state)
    if metric == "silhouette":
        return float(silhouette_score(X, labels, random_state=random_state)), 0.0
    if metric in ("davies_bouldin", "DB"):
        return davies_bouldin_inv(X, labels), 0.0
    if metric in ("calinski_harabasz", "CH"):
        return float(calinski_harabasz_score(X, labels)), 0.0
    raise ValueError(f"Unknown metric: {metric!r}")


def select_k(
    metric: str,
    ks: Sequence[int],
    values: Sequence[float],
    errors: Sequence[float],
) -> int:
    """Choose k for one classical index.

    Every index but the gap statistic selects the argmax. The gap statistic
    uses Tibshirani's rule: the smallest k with
    Gap(k) >= Gap(k+1) - s_{k+1}. When no k satisfies it, the largest
    candidate is returned, which is the conventional fallback.
    """
    ks = list(ks)
    values = np.asarray(values, dtype=float)

    if metric != "gap":
        if np.all(np.isnan(values)):
            return ks[-1]
        return ks[int(np.nanargmax(values))]

    errors = np.asarray(errors, dtype=float)
    for i in range(len(ks) - 1):
        if not np.isfinite(values[i]) or not np.isfinite(values[i + 1]):
            continue
        if values[i] >= values[i + 1] - errors[i + 1]:
            return ks[i]
    return ks[-1]
