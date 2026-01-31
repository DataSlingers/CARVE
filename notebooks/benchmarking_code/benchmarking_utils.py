import inspect

from typing import Type, Iterable, Any

import numpy as np

from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import confusion_matrix, pairwise_distances
from sklearn.utils import check_random_state

from scipy.optimize import linear_sum_assignment


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


def _pick_first(v: Any) -> Any:
    """
    Returns the first element if v is a list/tuple; otherwise returns v unchanged.

    Args:
        - v: Parameter value or list/tuple of candidate values.

    Returns:
        The first element if v is list/tuple; otherwise v.
    """
    return v[0] if isinstance(v, (list, tuple)) else v


def _build_estimator(
    algorithm: Type[ClusterMixin],
    k: int,
    algorithm_params: dict[str, list[Any]] | None,
    seed: int,
) -> ClusterMixin:
    """
    Builds a clustering estimator with fixed parameters and a specified number of clusters.

    Args:
        - algorithm (Type[ClusterMixin]): Estimator class to instantiate.
        - k (int): Number of clusters for this estimator instance.
        - algorithm_params (dict[str, list[Any]] | None): Parameter grid or fixed parameters.
        - seed (int): Random seed used when the estimator supports random_state.

    Returns:
        ClusterMixin: Instantiated clustering estimator.
    """
    # start with the fixed params coming from model_grids[0][1]
    params = {}
    if algorithm_params:
        for key, val in algorithm_params.items():
            if key == "n_clusters":
                continue
            
            params[key] = _pick_first(val)

    # set k for this call
    params["n_clusters"] = k

    # set random_state when supported
    sig = inspect.signature(algorithm.__init__)
    if "random_state" in sig.parameters:
        params["random_state"] = seed

    # pin kmeans defaults across sklearn versions
    if algorithm.__name__ == "KMeans" or algorithm is KMeans:
        if "n_init" in sig.parameters and "n_init" not in params:
            params["n_init"] = 10

    return algorithm(**params)


def make_model_grids(
    model: str,
    test_ks: Iterable[int],
    spectral_quant: float = 0.5,
    X: np.ndarray | None = None,
) -> list[tuple[Type[ClusterMixin], dict[str, list[Any]]]]:
    """
    Creates parameter grids for supported clustering models.

    Args:
        - model (str): Model key ('agglomerative', 'spectral', or default to kmeans).
        - test_ks (Iterable[int]): Candidate cluster counts.
        - spectral_quant (float): Quantile used for spectral gamma estimation (default: 0.5).
        - X (np.ndarray | None): Data matrix used to estimate spectral gamma when needed.

    Returns:
        list[tuple]: List of (EstimatorClass, param_grid) tuples.
    """
    if model == "agglomerative":
        return [(AgglomerativeClustering, {"n_clusters": list(test_ks), "linkage": ["ward"]})]
    if model == "spectral":
        gamma = gamma_quantile_approx(X, q=spectral_quant)
        return [(SpectralClustering, {"n_clusters": list(test_ks), "affinity": ["rbf"], "gamma": [gamma]})]
    return [(KMeans, {"n_clusters": list(test_ks), "n_init": [10]})]

