"""Tests for estimator construction."""

import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans

from benchmarks._estimators import (
    ESTIMATOR_CLASSES,
    RESOLUTION_ESTIMATORS,
    build_estimator,
    param_grids,
    resolution_grids,
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


class TestResolutionEstimators:
    def test_leiden_sweeps_resolution_not_n_clusters(self):
        assert "leiden" in RESOLUTION_ESTIMATORS
        assert "kmeans" not in RESOLUTION_ESTIMATORS

    def test_resolution_grid_shape(self):
        from carve.cluster import LeidenClustering

        grids = resolution_grids(EstimatorSpec(name="leiden"), [0.1, 0.2, 0.3])
        assert len(grids) == 1
        cls, grid = grids[0]
        assert cls is LeidenClustering
        assert grid["resolution"] == [0.1, 0.2, 0.3]
        assert "n_clusters" not in grid

    def test_resolution_grid_rejects_a_k_based_estimator(self):
        with pytest.raises(ValueError, match="does not sweep resolution"):
            resolution_grids(EstimatorSpec(name="kmeans"), [0.1])

    def test_param_grids_rejects_a_resolution_estimator(self):
        # SweepSpec is frozen: one run sweeps exactly one parameter, so a
        # resolution estimator must never be handed an n_clusters grid.
        with pytest.raises(ValueError, match="sweeps resolution"):
            param_grids(EstimatorSpec(name="leiden"), [2, 3])

    def test_minibatch_kmeans_builds_with_n_clusters(self):
        est = build_estimator(
            EstimatorSpec(name="minibatch_kmeans"), n_clusters=4, random_state=0
        )
        assert isinstance(est, MiniBatchKMeans)
        assert est.n_clusters == 4
        assert est.random_state == 0

    def test_build_estimator_rejects_a_resolution_estimator(self):
        with pytest.raises(ValueError, match="sweeps resolution"):
            build_estimator(EstimatorSpec(name="leiden"), n_clusters=4, random_state=0)
