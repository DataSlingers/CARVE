"""Construction of the base clustering estimators used by the benchmarks.

The literals that used to be scattered through the old estimator factory
(n_init=10, ward linkage, self-tuning affinity) live here as one table.
"""

import inspect
from collections.abc import Sequence
from typing import Any

from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans

from carve.cluster import LeidenClustering, SpectralClustering

from ._types import EstimatorSpec

ESTIMATOR_CLASSES: dict[str, type[ClusterMixin]] = {
    "kmeans": KMeans,
    "minibatch_kmeans": MiniBatchKMeans,
    "agglomerative": AgglomerativeClustering,
    "spectral": SpectralClustering,
    "leiden": LeidenClustering,
}

ESTIMATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    # n_init is pinned so results do not move when scikit-learn changes its
    # default, which it has done before.
    "kmeans": {"n_init": 10},
    "minibatch_kmeans": {"n_init": 10},
    "agglomerative": {"linkage": "ward"},
    "spectral": {"affinity": "self_tuning"},
    # 15 neighbors is the scanpy convention practitioners will recognize;
    # modularity is Leiden's default objective. The two objective functions
    # use different resolution scales and must not share a grid.
    "leiden": {"n_neighbors": 15, "objective_function": "modularity"},
}

# Estimators whose granularity is swept through resolution rather than
# n_clusters. A single CARVE run sweeps exactly one parameter, so these
# cannot appear in the same grid as a k-based estimator.
RESOLUTION_ESTIMATORS: frozenset[str] = frozenset({"leiden"})

# Estimators that build a dense n-by-n affinity or distance matrix, so their
# memory cost is quadratic in the sample count regardless of k or resolution.
# SpectralClustering computes a full affinity matrix; AgglomerativeClustering
# with ward linkage is likewise quadratic. See _studies._check_dense_fit,
# which is where this matters: at n=50,000 a single such matrix is 20 GB.
DENSE_PAIRWISE_ESTIMATORS: frozenset[str] = frozenset({"spectral", "agglomerative"})


def apply_random_state(
    estimator_cls: type[ClusterMixin], params: dict[str, Any], random_state: int
) -> dict[str, Any]:
    """Set random_state on params, but only when the estimator accepts it.

    random_state is passed only when the estimator's signature accepts it,
    probed rather than hardcoded so a swapped estimator class does not raise.
    Mutates params in place and returns it for convenience at the call site.
    """
    signature = inspect.signature(estimator_cls.__init__)
    if "random_state" in signature.parameters:
        params["random_state"] = int(random_state)
    return params


def build_estimator(
    spec: EstimatorSpec, n_clusters: int, random_state: int
) -> ClusterMixin:
    """Instantiate the estimator for one scenario at one k."""
    if spec.name in RESOLUTION_ESTIMATORS:
        raise ValueError(
            f"Estimator {spec.name!r} sweeps resolution, not n_clusters. Use "
            "resolution_grids and CARVE's resolution mode."
        )
    estimator_cls = ESTIMATOR_CLASSES[spec.name]
    params: dict[str, Any] = dict(ESTIMATOR_DEFAULTS[spec.name])
    params["n_clusters"] = int(n_clusters)
    apply_random_state(estimator_cls, params, random_state)

    return estimator_cls(**params)


def param_grids(
    spec: EstimatorSpec, candidate_k: Sequence[int]
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """Build the estimator_param_grids structure CARVE takes.

    One entry, sweeping n_clusters over the candidate values, with the
    estimator's fixed defaults alongside.
    """
    if spec.name in RESOLUTION_ESTIMATORS:
        raise ValueError(
            f"Estimator {spec.name!r} sweeps resolution, not n_clusters. Use "
            "resolution_grids and CARVE's resolution mode."
        )
    if not candidate_k:
        raise ValueError("candidate_k must not be empty.")

    grid: dict[str, list[Any]] = {"n_clusters": [int(k) for k in candidate_k]}
    for key, value in ESTIMATOR_DEFAULTS[spec.name].items():
        grid[key] = [value]

    return [(ESTIMATOR_CLASSES[spec.name], grid)]


def resolution_grids(
    spec: EstimatorSpec, resolutions: Sequence[float]
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """Build the estimator_param_grids structure for a resolution sweep.

    The number of clusters is an outcome here, not an input, so CARVE is run
    in resolution mode and reports the observed cluster count per value.
    """
    if spec.name not in RESOLUTION_ESTIMATORS:
        raise ValueError(
            f"Estimator {spec.name!r} does not sweep resolution. Use "
            "param_grids for k-based estimators."
        )
    if len(resolutions) == 0:
        raise ValueError("resolutions must not be empty.")

    grid: dict[str, list[Any]] = {"resolution": [float(r) for r in resolutions]}
    for key, value in ESTIMATOR_DEFAULTS[spec.name].items():
        grid[key] = [value]

    return [(ESTIMATOR_CLASSES[spec.name], grid)]
