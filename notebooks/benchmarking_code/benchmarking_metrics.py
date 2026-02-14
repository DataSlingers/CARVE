from typing import Type

import numpy as np

from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, pairwise_distances, silhouette_score

from benchmarking_utils import _build_estimator


def compute_dispersion(X: np.ndarray, labels: np.ndarray, metric: str = 'euclidean') -> float:
    """
    Computes the within-cluster dispersion measure W_k for a clustering solution.

    Args:
        - X (np.ndarray): Data matrix with samples as rows and features as columns.
        - labels (np.ndarray): Cluster labels for each observation.
        - metric (str): Distance metric used to compute pairwise distances within clusters (default: 'euclidean').

    Returns:
        float: Total within-cluster dispersion W_k.
    """
    unique_labels = np.unique(labels)
    W_k = 0.0
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        
        if len(cluster_points) <= 1:
            continue
        
        distances = pairwise_distances(cluster_points, metric=metric, squared=True)
        W_k += np.sum(distances) / (2 * cluster_points.shape[0])
    
    return W_k


def gen_null_box(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a uniform reference dataset over the axis-aligned bounding box of X.

    Args:
        - X (np.ndarray): Data matrix with samples as rows and features as columns.
        - rng (np.random.Generator): Random number generator for reproducible sampling.

    Returns:
        np.ndarray: Reference dataset with the same shape as X.
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    span = maxs - mins
    return rng.random(size=X.shape) * span + mins


def gap_statistic(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    n_reference_datasets: int = 10,
    estimator_cls: Type[ClusterMixin] = KMeans,
    estimator_params: dict | None = None,
    random_state: int = 0,
) -> float:
    """
    Computes the Gap statistic: Gap(k) = E[log W*_k] - log W_k.

    W_k is computed from the provided labels on X, while W*_k is computed from
    reference datasets sampled uniformly from the bounding box of X.

    Args:
        - X (np.ndarray): Data matrix with samples as rows and features as columns.
        - labels (np.ndarray): Cluster labels for each observation.
        - metric (str): Distance metric for within-cluster dispersion (default: 'euclidean').
        - n_reference_datasets (int): Number of reference datasets to generate (default: 10).
        - estimator_cls (Type[ClusterMixin]): Clustering estimator class for reference clustering.
        - estimator_params (dict | None): Parameters for the estimator (default: None).
        - random_state (int): Seed for reproducible reference sampling (default: 0).

    Returns:
        float: Gap statistic value; np.nan if W_k is non-positive or non-finite.
    """
    rng = np.random.default_rng(int(random_state))

    W_k = compute_dispersion(X, labels, metric)
    if not np.isfinite(W_k) or W_k <= 0:
        return np.nan

    k = int(np.unique(labels).size)

    ref_log_disps = np.empty(n_reference_datasets, dtype=float)
    for b in range(n_reference_datasets):
        X_ref = gen_null_box(X, rng)
        seed_b = int(rng.integers(0, 2**32 - 1))

        est = _build_estimator(estimator_cls, k, estimator_params, seed_b)
        ref_labels = est.fit_predict(X_ref)

        W_ref = compute_dispersion(X_ref, ref_labels, metric)
        ref_log_disps[b] = np.log(W_ref)

    return float(np.mean(ref_log_disps) - np.log(W_k))


def davies_bouldin_inv(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the inverse Davies-Bouldin index for a clustering solution.

    Args:
        - X (np.ndarray): Data matrix with samples as rows and features as columns.
        - labels (np.ndarray): Cluster labels for each observation.

    Returns:
        float: Inverse Davies-Bouldin score (1 / (1 + db)), higher is better. Returns np.nan if db is not finite.
    """
    db = davies_bouldin_score(X, labels)
    return 1.0 / (1.0 + db) if np.isfinite(db) else np.nan


def calculate_metric(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str,
    estimator_cls: Type[ClusterMixin],
    estimator_params: dict | None = None,
    random_state: int = 0,
) -> float:
    """
    Calculates a clustering validation metric for a given clustering solution.

    Args:
        - X (np.ndarray): Data matrix with samples as rows and features as columns.
        - labels (np.ndarray): Cluster labels for each observation.
        - metric (str): Metric to compute ('gap', 'silhouette', 'davies_bouldin', 'DB', 'calinski_harabasz', 'CH').
        - estimator_cls (Type[ClusterMixin]): Clustering estimator class (used for Gap Statistic reference clustering).
        - estimator_params (dict | None): Parameters for the estimator (default: None).
        - random_state (int): Seed for reproducible reference sampling (default: 0).

    Returns:
        float: Value of the requested clustering metric.
    """
    if metric == 'gap':
        return gap_statistic(
            X,
            labels,
            estimator_cls=estimator_cls,
            estimator_params=estimator_params,
            random_state=random_state,
        )

    elif metric == 'silhouette':
        return silhouette_score(X, labels, random_state=random_state)

    elif metric == 'davies_bouldin' or metric == 'DB':
        return davies_bouldin_inv(X, labels)

    elif metric == 'calinski_harabasz' or metric == 'CH':
        return calinski_harabasz_score(X, labels)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")

