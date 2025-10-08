from typing import Any, Callable, Dict, List, Tuple, Type, Union
import numpy as np
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from umap import UMAP

from .cluster import SpectralClusteringCARVE

GridSpec = Tuple[Type[ClusterMixin], Dict[str, List[Any]]]
PreprocSpec = Tuple[Callable[..., TransformerMixin], Dict[str, List[Any]]]

def default_model_grids(
    X: np.ndarray, 
    K: Union[int, np.ndarray] = 10
) -> List[GridSpec]:
    def gamma_quantiles(X, qs=(0.05, 0.10, 0.25, 0.50)):
        D2 = pairwise_distances(X, metric='sqeuclidean')
        d2 = D2[np.triu_indices_from(D2, k=1)]
        return [1.0 / (2.0 * np.quantile(d2, q)) for q in qs]
    
    if isinstance(K, int):
        ks = list(range(2, K + 1))
    else:
        ks = list(np.asarray(K).tolist())
        
    return [
        (KMeans, {"n_clusters": ks}),
        (AgglomerativeClustering, {"n_clusters": ks, "linkage": ["ward", "complete", "average", "single"]}),
        # (SpectralClusteringCARVE, {"n_clusters": ks, "gamma": gamma_quantiles(X)}),
        # (SpectralClusteringCARVE, {"n_clusters": ks, "affinity": ['knn'], "n_neighbors": [5, 10, 15, 20]}),
        (SpectralClustering, {"n_clusters": ks, "gamma": gamma_quantiles(X)}),
        # (SpectralClustering, {"n_clusters": ks, "affinity": ['nearest_neighbors'], "n_neighbors": [5, 10, 15, 20]}),
    ]
    
def default_norm_options() -> List[PreprocSpec]:
    norm_options = [
        (FunctionTransformer, {}),                      # identity
        (StandardScaler, {}),                           # normalize to zero mean, unit variance
        (FunctionTransformer, {'func': [np.log1p]}),    # log1p transform
    ]

    return norm_options

def default_dr_options(
    X: np.ndarray, 
    rho: float = 0.6
) -> List[PreprocSpec]:
    n_samples, p = X.shape
    min_n = int(round(n_samples * (1 - rho))) - 1

    dr_options = [
        (FunctionTransformer, {}),                      # identity
        (PCA, {
            'n_components': list(range(2, min(min_n, p) + 1))
        }),
        (TSNE, {
            'n_components': [2, 3], 
            'perplexity': list(range(5, min(min_n, 51)))
        }),
        (UMAP, {
            'n_components': list(range(2, min(min_n, p) + 1)),
            'n_neighbors': list(range(5, 51)), 
            'min_dist': [0.1]
        }),
    ]

    return dr_options