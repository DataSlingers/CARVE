"""Default estimator and preprocessing grids for CARVE."""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from umap import UMAP

from .cluster import SpectralClusteringCARVE
from ._types import GridSpec, PreprocSpec


def estimate_knn_gamma(
    X: np.ndarray,
    n_neighbors: int = 7,
    multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> list[float]:
    """Estimate RBF gamma values using a k-NN median heuristic.

    Fits a NearestNeighbors model, takes the k-th neighbor distance for
    each point, computes sigma = median(kth_distances), and returns
    gamma = 1 / (2 * (m * sigma)^2) for each multiplier m.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_neighbors : int, default=7
        Number of neighbors.
    multipliers : tuple of float, default=(0.5, 1.0, 2.0)
        Scale multipliers applied to sigma.

    Returns
    -------
    gammas : list of float
        Gamma values spanning a range around the local scale.
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    kth_dists = dists[:, -1]
    sigma = float(np.median(kth_dists))
    if sigma <= 0:
        sigma = float(np.mean(kth_dists[kth_dists > 0])) if np.any(kth_dists > 0) else 1.0
    return [float(1.0 / (2.0 * (m * sigma) ** 2)) for m in multipliers]


def default_estimator_grids(
    X: np.ndarray,
    n_clusters: int | np.ndarray = 10,
) -> list[GridSpec]:
    """Return default clustering estimator grids.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data used to derive data-driven hyperparameters (e.g., gamma).
    n_clusters : int or ndarray, default=10
        Number(s) of clusters to evaluate.

    Returns
    -------
    grids : list of tuple
        List of (EstimatorClass, param_grid) tuples suitable for
        ``sklearn.model_selection.ParameterGrid``.
    """
    ks = list(np.asarray(n_clusters).tolist())

    return [
        (KMeans, {"n_clusters": ks}),
        (
            AgglomerativeClustering,
            {"n_clusters": ks, "linkage": ["ward", "average", "single"]},
        ),
        (
            SpectralClusteringCARVE,
            {"n_clusters": ks, "affinity": ["self_tuning"]},
        ),
        (
            SpectralClusteringCARVE,
            {"n_clusters": ks, "affinity": ["rbf"], "gamma": estimate_knn_gamma(X)},
        ),
    ]


def default_normalization_options() -> list[PreprocSpec]:
    """Return default normalization preprocessing options.

    Returns
    -------
    options : list of tuple
        List of (TransformerClass, param_grid) pairs.
    """
    return [
        (FunctionTransformer, {}),
        (StandardScaler, {}),
        (FunctionTransformer, {"func": [np.log1p]}),
    ]


def default_dim_reduction_options(
    X: np.ndarray,
    subsample_ratio: float = 0.6,
) -> list[PreprocSpec]:
    """Return default dimensionality reduction options.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data used to determine feasible component counts.
    subsample_ratio : float, default=0.6
        Subsample ratio used in validation; constrains DR hyperparameters.

    Returns
    -------
    options : list of tuple
        List of (TransformerClass, param_grid) pairs.
    """
    n_samples, p = X.shape
    min_n = int(round(n_samples * (1 - subsample_ratio))) - 1

    return [
        (FunctionTransformer, {}),
        (PCA, {"n_components": list(range(2, min(min_n, p)))}),
        (TSNE, {"n_components": [2], "perplexity": list(range(5, min(min_n, 51)))}),
        (
            UMAP,
            {
                "n_components": list(range(2, min(min_n, p))),
                "n_neighbors": list(range(5, 51)),
                "min_dist": [0.1],
            },
        ),
    ]
