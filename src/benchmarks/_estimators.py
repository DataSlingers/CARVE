"""Construction of the base clustering estimators and the generalizability
classifiers used by the benchmarks.

The literals that used to be scattered through the old estimator factory
(n_init=10, ward linkage, self-tuning affinity) live here as one table.
"""

import inspect
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from sklearn.base import ClassifierMixin, ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from carve.cluster import LeidenClustering, LouvainClustering, SpectralClustering

from ._types import EstimatorSpec

ESTIMATOR_CLASSES: dict[str, type[ClusterMixin]] = {
    "kmeans": KMeans,
    "kmeans_n_init_100": KMeans,
    "minibatch_kmeans": MiniBatchKMeans,
    "agglomerative": AgglomerativeClustering,
    "agglomerative_single": AgglomerativeClustering,
    "spectral": SpectralClustering,
    "leiden": LeidenClustering,
    "louvain": LouvainClustering,
}

ESTIMATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    # n_init is pinned so results do not move when scikit-learn changes its
    # default, which it has done before.
    "kmeans": {"n_init": 10},
    # For data whose KMeans optimum 10 restarts do not reliably reach, so
    # that the partition does not depend on the seed. Registered under its
    # own name, like single linkage, because the gap statistic and the
    # figures map a fit's parameters back to its estimator by name.
    "kmeans_n_init_100": {"n_init": 100},
    "minibatch_kmeans": {"n_init": 10},
    "agglomerative": {"linkage": "ward"},
    "agglomerative_single": {"linkage": "single"},
    "spectral": {"affinity": "self_tuning"},
    # 15 neighbors is the scanpy convention practitioners will recognize;
    # modularity is Leiden's default objective. The two objective functions
    # use different resolution scales and must not share a grid.
    "leiden": {"n_neighbors": 15, "objective_function": "modularity"},
    # Louvain optimizes modularity only and takes no objective_function; the
    # same 15-neighbor graph as Leiden.
    "louvain": {"n_neighbors": 15},
}

# Estimators whose granularity is swept through resolution rather than
# n_clusters. A single CARVE run sweeps exactly one parameter, so these
# cannot appear in the same grid as a k-based estimator.
RESOLUTION_ESTIMATORS: frozenset[str] = frozenset({"leiden", "louvain"})

# Estimators that build a dense n-by-n affinity or distance matrix, so their
# memory cost is quadratic in the sample count regardless of k or resolution.
# SpectralClustering computes a full affinity matrix; AgglomerativeClustering
# is likewise quadratic under ward and single linkage alike (single linkage
# builds the full pairwise distance matrix without a connectivity graph).
# See _studies._check_dense_fit, which is where this matters: at n=50,000 a
# single such matrix is 20 GB.
DENSE_PAIRWISE_ESTIMATORS: frozenset[str] = frozenset(
    {"spectral", "agglomerative", "agglomerative_single"}
)


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


# The generalizability classifier, by name. None hands CARVE no classifier,
# so it builds its default forest from the scenario's n_trees. Shrinkage LDA
# suits k-means on Gaussian mixtures: k-means draws hyperplane boundaries, and
# at large p the separation between two clusters is spread across every
# feature, which a forest of axis-aligned splits cannot assemble from one
# subsample. LDA takes neither random_state nor n_jobs, so CARVE's classifier
# builder leaves it as constructed.
CLASSIFIER_FACTORIES: dict[str, Callable[[], ClassifierMixin] | None] = {
    "random_forest": None,
    "lda": partial(LinearDiscriminantAnalysis, solver="lsqr", shrinkage="auto"),
}


def build_classifier(name: str) -> ClassifierMixin | None:
    """Instantiate a scenario's generalizability classifier.

    Returns None for "random_forest", which is what CARVE takes to mean its
    own default forest.
    """
    factory = CLASSIFIER_FACTORIES[name]
    return None if factory is None else factory()
