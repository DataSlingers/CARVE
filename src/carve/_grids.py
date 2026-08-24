"""Default estimator and preprocessing grids for CARVE."""

from typing import Literal

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from umap import UMAP

from .cluster import LeidenClustering, LouvainClustering, SpectralClustering
from ._sweep import SweepSpec, resolve_sweep
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
        sigma = (
            float(np.mean(kth_dists[kth_dists > 0])) if np.any(kth_dists > 0) else 1.0
        )
    return [float(1.0 / (2.0 * (m * sigma) ** 2)) for m in multipliers]


def default_estimator_grids(
    X: np.ndarray,
    n_clusters: int | np.ndarray = 10,
    preset: Literal["light", "full"] = "light",
    *,
    sweep: SweepSpec | None = None,
) -> list[GridSpec]:
    """Return default clustering estimator grids for a run's sweep axis.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data used to derive data-driven hyperparameters (e.g., gamma).
    n_clusters : int or ndarray, default=10
        Number(s) of clusters to evaluate. Ignored when ``sweep`` is given.
    preset : {"light", "full"}, default="light"
        ``"light"`` includes KMeans, Ward-linkage agglomerative, and
        self-tuning spectral clustering. ``"full"`` adds average- and
        single-linkage agglomerative clustering and RBF-kernel spectral
        clustering with data-driven gamma.
    sweep : SweepSpec, optional
        Resolved sweep axis. Defaults to an ``n_clusters`` sweep built
        from n_clusters.

    Returns
    -------
    grids : list of tuple
        List of (EstimatorClass, param_grid) tuples suitable for
        ``sklearn.model_selection.ParameterGrid``.

    Raises
    ------
    ValueError
        If no default grids exist for the sweep parameter.
    """
    if sweep is None:
        sweep = resolve_sweep(n_clusters=n_clusters)

    if sweep.param == "n_clusters":
        return _default_k_grids(X, sweep.values, preset)
    if sweep.param == "resolution":
        return _default_resolution_grids(sweep.values, preset)
    if sweep.param == "min_cluster_size":
        return _default_min_cluster_size_grids(sweep.values, preset)

    raise ValueError(
        f"No default estimator grids for sweep parameter {sweep.param!r}. "
        "Pass estimator_param_grids=... explicitly."
    )


def _default_k_grids(
    X: np.ndarray,
    n_clusters: np.ndarray,
    preset: Literal["light", "full"] = "light",
) -> list[GridSpec]:
    """Grids for estimators that take an explicit number of clusters."""
    ks = list(np.asarray(n_clusters).tolist())

    grids: list[GridSpec] = [
        (KMeans, {"n_clusters": ks}),
        (AgglomerativeClustering, {"n_clusters": ks, "linkage": ["ward"]}),
        (SpectralClustering, {"n_clusters": ks, "affinity": ["self_tuning"]}),
    ]

    if preset == "full":
        grids.append(
            (
                AgglomerativeClustering,
                {"n_clusters": ks, "linkage": ["average", "single", "complete"]},
            ),
        )

        grids.append(
            (
                SpectralClustering,
                {"n_clusters": ks, "affinity": ["rbf"], "gamma": estimate_knn_gamma(X)},
            ),
        )

    return grids


def _default_resolution_grids(
    resolutions: np.ndarray,
    preset: Literal["light", "full"] = "light",
) -> list[GridSpec]:
    """Grids for graph-community estimators swept over ``resolution``.

    Parameters
    ----------
    resolutions : ndarray
        Resolution values to evaluate.
    preset : {"light", "full"}, default="light"
        ``"light"`` uses Leiden and Louvain on a 15-neighbor graph.
        ``"full"`` additionally varies the neighborhood size.

    Returns
    -------
    grids : list of tuple
        List of (EstimatorClass, param_grid) tuples.
    """
    res = [float(r) for r in np.asarray(resolutions).tolist()]

    grids: list[GridSpec] = [
        (LeidenClustering, {"resolution": res, "n_neighbors": [15]}),
        (LouvainClustering, {"resolution": res, "n_neighbors": [15]}),
    ]

    if preset == "full":
        grids.append((LeidenClustering, {"resolution": res, "n_neighbors": [10, 30]}))
        grids.append((LouvainClustering, {"resolution": res, "n_neighbors": [10, 30]}))

    return grids


def _default_min_cluster_size_grids(
    min_cluster_sizes: np.ndarray,
    preset: Literal["light", "full"] = "light",
) -> list[GridSpec]:
    """Grids for HDBSCAN swept over ``min_cluster_size``.

    Parameters
    ----------
    min_cluster_sizes : ndarray
        Minimum cluster sizes to evaluate.
    preset : {"light", "full"}, default="light"
        ``"full"`` additionally evaluates leaf cluster selection.

    Returns
    -------
    grids : list of tuple
        List of (EstimatorClass, param_grid) tuples.
    """
    sizes = [int(s) for s in np.asarray(min_cluster_sizes).tolist()]

    grids: list[GridSpec] = [
        (
            HDBSCAN,
            {"min_cluster_size": sizes, "cluster_selection_method": ["eom"]},
        ),
    ]

    if preset == "full":
        grids.append(
            (
                HDBSCAN,
                {"min_cluster_size": sizes, "cluster_selection_method": ["leaf"]},
            )
        )

    return grids


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
