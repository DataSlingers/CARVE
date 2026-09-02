"""Tests for estimator construction."""

import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans

from benchmarks._estimators import (
    ESTIMATOR_CLASSES,
    build_estimator,
    param_grids,
)
from benchmarks._types import KNOWN_ESTIMATORS, EstimatorSpec


def test_every_known_estimator_has_a_class():
    assert set(ESTIMATOR_CLASSES) == set(KNOWN_ESTIMATORS)


class TestBuildEstimator:
    def test_kmeans_gets_the_pinned_n_init(self):
        est = build_estimator(EstimatorSpec(name="kmeans"), n_clusters=5, random_state=7)
        assert isinstance(est, KMeans)
        assert est.n_clusters == 5
        assert est.n_init == 10

    def test_random_state_is_set_when_the_estimator_supports_it(self):
        est = build_estimator(EstimatorSpec(name="kmeans"), n_clusters=3, random_state=42)
        assert est.random_state == 42

    def test_random_state_is_omitted_when_unsupported(self):
        est = build_estimator(
            EstimatorSpec(name="agglomerative"), n_clusters=3, random_state=42
        )
        assert isinstance(est, AgglomerativeClustering)
        assert not hasattr(est, "random_state")

    def test_agglomerative_uses_ward_linkage(self):
        est = build_estimator(EstimatorSpec(name="agglomerative"), n_clusters=3, random_state=0)
        assert est.linkage == "ward"

    def test_spectral_uses_self_tuning_affinity(self):
        est = build_estimator(EstimatorSpec(name="spectral"), n_clusters=3, random_state=0)
        assert est.affinity == "self_tuning"


class TestParamGrids:
    def test_shape_matches_what_carve_expects(self):
        grids = param_grids(EstimatorSpec(name="kmeans"), candidate_k=(3, 4, 5))
        assert len(grids) == 1
        cls, grid = grids[0]
        assert cls is KMeans
        assert grid["n_clusters"] == [3, 4, 5]
        assert grid["n_init"] == [10]

    def test_agglomerative_grid_carries_linkage(self):
        _, grid = param_grids(EstimatorSpec(name="agglomerative"), candidate_k=(2, 3))[0]
        assert grid["linkage"] == ["ward"]

    def test_rejects_empty_candidate_k(self):
        with pytest.raises(ValueError, match="candidate_k"):
            param_grids(EstimatorSpec(name="kmeans"), candidate_k=())
