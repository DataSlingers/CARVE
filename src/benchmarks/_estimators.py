"""Construction of the base clustering estimators used by the benchmarks.

The literals that used to be scattered through the old estimator factory
(n_init=10, ward linkage, self-tuning affinity) live here as one table.
"""

import inspect
from collections.abc import Sequence
from typing import Any

from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans

from carve.cluster import SpectralClustering

from ._types import EstimatorSpec

ESTIMATOR_CLASSES: dict[str, type[ClusterMixin]] = {
    "kmeans": KMeans,
    "agglomerative": AgglomerativeClustering,
    "spectral": SpectralClustering,
}

ESTIMATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    # n_init is pinned so results do not move when scikit-learn changes its
    # default, which it has done before.
    "kmeans": {"n_init": 10},
    "agglomerative": {"linkage": "ward"},
    "spectral": {"affinity": "self_tuning"},
}


def build_estimator(
    spec: EstimatorSpec, n_clusters: int, random_state: int
) -> ClusterMixin:
    """Instantiate the estimator for one scenario at one k.

    random_state is passed only when the estimator's signature accepts it,
    probed rather than hardcoded so a swapped estimator class does not raise.
    """
    estimator_cls = ESTIMATOR_CLASSES[spec.name]
    params: dict[str, Any] = dict(ESTIMATOR_DEFAULTS[spec.name])
    params["n_clusters"] = int(n_clusters)

    signature = inspect.signature(estimator_cls.__init__)
    if "random_state" in signature.parameters:
        params["random_state"] = int(random_state)

    return estimator_cls(**params)


def param_grids(
    spec: EstimatorSpec, candidate_k: Sequence[int]
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """Build the estimator_param_grids structure CARVE takes.

    One entry, sweeping n_clusters over the candidate values, with the
    estimator's fixed defaults alongside.
    """
    if not candidate_k:
        raise ValueError("candidate_k must not be empty.")

    grid: dict[str, list[Any]] = {"n_clusters": [int(k) for k in candidate_k]}
    for key, value in ESTIMATOR_DEFAULTS[spec.name].items():
        grid[key] = [value]

    return [(ESTIMATOR_CLASSES[spec.name], grid)]
