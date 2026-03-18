"""Default estimator and preprocessing grids for CARVE."""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils import check_random_state
from umap import UMAP

from ._types import GridSpec, PreprocSpec


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

    def estimate_rbf_gamma(
        X: np.ndarray,
        quantiles: tuple[float, ...] = (0.05, 0.10, 0.50),
        max_points: int = 500,
        random_state: int = 0,
    ) -> list[float]:
        """Approximate RBF gamma values from distance quantiles.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        quantiles : tuple of float, default=(0.05, 0.10, 0.50)
            Quantiles over squared distances used to set gamma values.
        max_points : int, default=500
            Maximum number of points used to approximate distance quantiles.
        random_state : int, default=0
            Random seed for subsampling.

        Returns
        -------
        gammas : list of float
            Gamma values corresponding to the requested quantiles.
        """
        rng = check_random_state(random_state)
        n = X.shape[0]
        Xs = X[rng.choice(n, size=max_points, replace=False)] if n > max_points else X

        D2 = pairwise_distances(Xs, metric="sqeuclidean")
        d2 = D2[np.triu_indices_from(D2, k=1)]
        d2 = d2[np.isfinite(d2)]
        if d2.size == 0:
            return [1.0 for _ in quantiles]
        return [float(1.0 / (2.0 * np.quantile(d2, q))) for q in quantiles]

    ks = list(np.asarray(n_clusters).tolist())

    return [
        (KMeans, {"n_clusters": ks}),
        (
            AgglomerativeClustering,
            {"n_clusters": ks, "linkage": ["ward", "average", "single"]},
        ),
        (SpectralClustering, {"n_clusters": ks, "gamma": estimate_rbf_gamma(X)}),
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
