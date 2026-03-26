import inspect

from typing import Type, Iterable, Any

import numpy as np
import pandas as pd

import igraph as ig

import leidenalg

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import confusion_matrix, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def gamma_quantile_approx(
    X: np.ndarray,
    q: float = 0.50,
    max_points: int = 500,
    random_state: int = 0,
) -> float:
    """
    Approximates the gamma hyperparameter for spectral clustering using a quantile of pairwise squared distances.

    Args:
        - X (np.ndarray): Data matrix with samples as rows and features as columns.
        - q (float): Quantile of pairwise squared distances to use (default: 0.50).
        - max_points (int): Maximum number of points sampled to estimate distances (default: 500).
        - random_state (int): Seed for reproducible sampling (default: 0).

    Returns:
        float: Gamma approximation based on the selected distance quantile.
    """
    rng = check_random_state(random_state)
    n = X.shape[0]
    Xs = X[rng.choice(n, size=max_points, replace=False)] if n > max_points else X

    D2 = pairwise_distances(Xs, metric="sqeuclidean")
    d2 = D2[np.triu_indices_from(D2, k=1)]
    d2 = d2[np.isfinite(d2)]
    if d2.size == 0:
        return 1.0
    return float(1.0 / (2.0 * np.quantile(d2, q)))


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
    return np.array([label_map[l] if l in label_map else l for l in pred_labels])


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
    spectral_quant: float = 0.5,
    X: np.ndarray | None = None,
    random_state: int = 0,
) -> list[tuple[Type[ClusterMixin], dict[str, list[Any]]]]:
    """
    Creates parameter grids for supported clustering estimators.

    Args:
        - estimator (str): Estimator key ('agglomerative', 'spectral', or default to kmeans).
        - candidate_clusters (Iterable[int]): Candidate cluster counts.
        - spectral_quant (float): Quantile used for spectral gamma estimation (default: 0.5).
        - X (np.ndarray | None): Data matrix used to estimate spectral gamma when needed.
        - random_state (int): Random seed used for reproducibility.

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
        gamma = gamma_quantile_approx(X, q=spectral_quant, random_state=random_state)
        return [
            (
                SpectralClustering,
                {"n_clusters": list(candidate_clusters), "gamma": [gamma]},
            )
        ]
    return [(KMeans, {"n_clusters": list(candidate_clusters), "n_init": [10]})]


class LeidenClustering(BaseEstimator, ClusterMixin):
    """
    Wrapper for Leiden community detection algorithm with sklearn interface.
    Clusters tabular data by building a k-nearest neighbor graph and running Leiden.

    Parameters:
        n_clusters (int): Desired number of clusters (approximate, Leiden may not match exactly).
        n_neighbors (int): Number of neighbors for graph construction (default: 10).
        resolution (float): Resolution parameter for Leiden (higher = more clusters, default: 1.0).
        linkage (str): Linkage method for hierarchical clustering ('ward', 'single', 'complete', 'average', etc.).
        random_state (int or None): Random seed.
    """

    def __init__(
        self,
        n_clusters=None,
        n_neighbors=10,
        resolution=1.0,
        linkage="ward",
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.linkage = (
            linkage  # may be set to 'ward', 'single', 'complete', 'average', etc.
        )
        self.random_state = random_state

    def fit(self, X, y=None):
        # Build kNN graph
        knn = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric="euclidean")
        knn.fit(X)
        knn_graph = knn.kneighbors_graph(X, mode="connectivity")
        sources, targets = knn_graph.nonzero()
        edges = list(zip(sources.tolist(), targets.tolist()))

        # Create igraph graph
        g = ig.Graph(n=X.shape[0], edges=edges, directed=False)
        g.simplify()

        # Run Leiden
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            seed=self.random_state,
        )
        self.labels_ = np.array(partition.membership)

        # Optionally, relabel to match n_clusters if specified
        if (
            self.n_clusters is not None
            and len(np.unique(self.labels_)) != self.n_clusters
        ):
            # Map largest clusters to 0..n_clusters-1, rest to -1
            # counts = np.bincount(self.labels_)
            # top = np.argsort(counts)[::-1][:self.n_clusters]
            # mapping = {old: new for new, old in enumerate(top)}
            # self.labels_ = np.array([mapping.get(l, -1) for l in self.labels_])

            # Merge clusters using Ward's linkage to match n_clusters

            # Compute pairwise distances between cluster centroids
            unique_labels = np.unique(self.labels_)
            centroids = np.array(
                [X[self.labels_ == label].mean(axis=0) for label in unique_labels]
            )
            distances = pdist(centroids, metric="euclidean")
            Z = linkage(distances, method=self.linkage)

            # Cut dendrogram to get n_clusters
            cluster_mapping = fcluster(Z, self.n_clusters, criterion="maxclust") - 1
            label_map = {old: new for old, new in zip(unique_labels, cluster_mapping)}
            self.labels_ = np.array([label_map[l] for l in self.labels_])

        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
