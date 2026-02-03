"""Default estimator and preprocessing grids for CARVE."""

from typing import Any, Callable, Dict, List, Tuple, Type, Union
import numpy as np
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils import check_random_state
from umap import UMAP

from .cluster import SpectralClusteringCARVE

GridSpec = Tuple[Type[ClusterMixin], Dict[str, List[Any]]]
PreprocSpec = Tuple[Callable[..., TransformerMixin], Dict[str, List[Any]]]

def default_estimator_grids(
    X: np.ndarray, 
    n_clusters: Union[int, np.ndarray] = 10
) -> List[GridSpec]:
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
    def gamma_quantiles_approx(
        X: np.ndarray,
        qs: Tuple[float, ...] = (0.05, 0.10, 0.50),
        max_points: int = 500,
        random_state: int = 0,
    ) -> List[float]:
        """Approximate RBF gamma values from distance quantiles.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        qs : tuple of float, default=(0.05, 0.10, 0.50)
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
            return [1.0 for _ in qs]
        return [float(1.0 / (2.0 * np.quantile(d2, q))) for q in qs]
    
    if isinstance(n_clusters, int):
        ks = list(range(2, n_clusters + 1))
    else:
        ks = list(np.asarray(n_clusters).tolist())
        
    return [
        (KMeans, {"n_clusters": ks}),
        (AgglomerativeClustering, {"n_clusters": ks, "linkage": ["ward", "average", "single"]}),
        (SpectralClustering, {"n_clusters": ks, "gamma": gamma_quantiles_approx(X)}),
        # (SpectralClustering, {"n_clusters": ks, "affinity": ['nearest_neighbors'], "n_neighbors": [5, 10, 15, 20]}),
        # (SpectralClusteringCARVE, {"n_clusters": ks, "gamma": gamma_quantiles(X)}),
        # (SpectralClusteringCARVE, {"n_clusters": ks, "affinity": ['knn'], "n_neighbors": [5, 10, 15, 20]}),
    ]
    
def default_norm_options() -> List[PreprocSpec]:
    """Return default normalization preprocessing options.

    Returns
    -------
    options : list of tuple
        List of (TransformerClass, param_grid) pairs.
    """
    norm_options = [
        (FunctionTransformer, {}),                      # identity
        (StandardScaler, {}),                           # normalize to zero mean, unit variance
        (FunctionTransformer, {'func': [np.log1p]}),    # log1p transform
    ]

    return norm_options

def default_dr_options(
    X: np.ndarray, 
    subsample_ratio: float = 0.6
) -> List[PreprocSpec]:
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

    dr_options = [
        (FunctionTransformer, {}),                      # identity
        (PCA, {
            'n_components': list(range(2, min(min_n, p)))
        }),
        (TSNE, {
            'n_components': [2], 
            'perplexity': list(range(5, min(min_n, 51)))
        }),
        (UMAP, {
            'n_components': list(range(2, min(min_n, p))),
            'n_neighbors': list(range(5, 51)), 
            'min_dist': [0.1]
        }),
        # (TSNE, {
        #     'n_components': [2, 3], 
        #     'perplexity': list(range(5, min(min_n, 51)))
        # }),
        # (UMAP, {
        #     'n_components': list(range(2, min(min_n, p) + 1)),
        #     'n_neighbors': list(range(5, 51)), 
        #     'min_dist': [0.1]
        # }),
    ]

    return dr_options