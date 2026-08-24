"""Tests for carve._grids module."""

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from carve.cluster import LeidenClustering, LouvainClustering, SpectralClustering
from carve._sweep import resolve_sweep
from carve._grids import (
    default_estimator_grids,
    default_normalization_options,
    default_dim_reduction_options,
)


# -----------------------------------------------------------------------
# default_estimator_grids
# -----------------------------------------------------------------------


class TestDefaultEstimatorGrids:
    def test_basic_structure(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]))
        assert isinstance(grids, list)
        assert (
            len(grids) == 3
        )  # KMeans, Agglomerative (ward), SpectralCARVE (self_tuning)

    def test_estimator_types(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]))
        classes = [cls for cls, _ in grids]
        assert KMeans in classes
        assert AgglomerativeClustering in classes
        assert SpectralClustering in classes

    def test_n_clusters_propagated(self):
        X = np.random.RandomState(0).randn(50, 5)
        ks = np.array([2, 3, 4])
        grids = default_estimator_grids(X, n_clusters=ks)
        for _, grid in grids:
            assert grid["n_clusters"] == [2, 3, 4]

    def test_gamma_values(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]), preset="full")
        # Find the RBF spectral entry (the one with gamma in its grid)
        rbf_grids = [(cls, g) for cls, g in grids if "gamma" in g]
        assert len(rbf_grids) == 1
        cls, spectral_grid = rbf_grids[0]
        assert cls is SpectralClustering
        assert len(spectral_grid["gamma"]) == 3  # 3 multipliers
        assert all(g > 0 for g in spectral_grid["gamma"])

    def test_self_tuning_entry(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]))
        # Find the self-tuning entry (has affinity but no gamma)
        st_grids = [
            (cls, g)
            for cls, g in grids
            if cls is SpectralClustering and "gamma" not in g
        ]
        assert len(st_grids) == 1
        _, grid = st_grids[0]
        assert grid["affinity"] == ["self_tuning"]

    def test_agglomerative_linkages(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]))
        agg_grid = next(g for cls, g in grids if cls is AgglomerativeClustering)
        assert "linkage" in agg_grid
        assert set(agg_grid["linkage"]) == {"ward"}

    def test_param_grids_are_valid(self):
        """Verify each grid can be expanded by ParameterGrid."""
        from sklearn.model_selection import ParameterGrid

        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]))
        for cls, grid in grids:
            configs = list(ParameterGrid(grid))
            assert len(configs) > 0
            for config in configs:
                assert "n_clusters" in config


class TestFullPreset:
    def test_full_preset_structure(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]), preset="full")
        assert len(grids) == 5

    def test_full_preset_agglomerative_linkages(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]), preset="full")
        agg_grids = [g for cls, g in grids if cls is AgglomerativeClustering]
        all_linkages = set()
        for g in agg_grids:
            all_linkages.update(g["linkage"])
        assert all_linkages == {"ward", "average", "single", "complete"}

    def test_full_preset_rbf_spectral(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]), preset="full")
        rbf_grids = [
            (cls, g) for cls, g in grids if cls is SpectralClustering and "gamma" in g
        ]
        assert len(rbf_grids) == 1
        _, grid = rbf_grids[0]
        assert grid["affinity"] == ["rbf"]
        assert len(grid["gamma"]) == 3


# -----------------------------------------------------------------------
# default_normalization_options
# -----------------------------------------------------------------------


class TestDefaultNormalizationOptions:
    def test_structure(self):
        options = default_normalization_options()
        assert isinstance(options, list)
        assert len(options) == 3  # identity, StandardScaler, log1p

    def test_contains_identity(self):
        options = default_normalization_options()
        has_identity = any(
            cls is FunctionTransformer and params == {} for cls, params in options
        )
        assert has_identity

    def test_contains_standard_scaler(self):
        options = default_normalization_options()
        has_scaler = any(cls is StandardScaler for cls, params in options)
        assert has_scaler

    def test_contains_log1p(self):
        options = default_normalization_options()
        has_log = any(
            cls is FunctionTransformer and params.get("func") == [np.log1p]
            for cls, params in options
        )
        assert has_log


# -----------------------------------------------------------------------
# default_dim_reduction_options
# -----------------------------------------------------------------------


class TestDefaultDimReductionOptions:
    def test_structure(self):
        X = np.random.RandomState(0).randn(100, 10)
        options = default_dim_reduction_options(X)
        assert isinstance(options, list)
        assert len(options) == 4  # identity, PCA, t-SNE, UMAP

    def test_contains_identity(self):
        X = np.random.RandomState(0).randn(100, 10)
        options = default_dim_reduction_options(X)
        has_identity = any(
            cls is FunctionTransformer and params == {} for cls, params in options
        )
        assert has_identity

    def test_contains_pca(self):
        X = np.random.RandomState(0).randn(100, 10)
        options = default_dim_reduction_options(X)
        has_pca = any(cls is PCA for cls, params in options)
        assert has_pca

    def test_pca_components_respect_data(self):
        X = np.random.RandomState(0).randn(50, 5)
        options = default_dim_reduction_options(X, subsample_ratio=0.6)
        pca_option = next((cls, params) for cls, params in options if cls is PCA)
        n_components = pca_option[1]["n_components"]
        # Should respect min(min_n, p) where min_n = round(50 * 0.4) - 1
        assert all(c >= 2 for c in n_components)
        assert max(n_components) < 5  # p = 5


# -----------------------------------------------------------------------
# Sweep-aware dispatch
# -----------------------------------------------------------------------


class TestResolutionGrids:
    def _grids(self, preset="light"):
        return default_estimator_grids(
            np.zeros((20, 3)),
            preset=preset,
            sweep=resolve_sweep(resolution=[0.5, 1.0]),
        )

    def test_light_returns_leiden_and_louvain(self):
        grids = self._grids()
        assert len(grids) == 2
        assert {cls for cls, _ in grids} == {LeidenClustering, LouvainClustering}

    def test_every_grid_sweeps_resolution(self):
        for _, grid in self._grids():
            assert "resolution" in grid
            assert "n_clusters" not in grid

    def test_resolution_values_propagated(self):
        for _, grid in self._grids():
            assert grid["resolution"] == [0.5, 1.0]

    def test_values_are_floats(self):
        for _, grid in self._grids():
            assert all(isinstance(r, float) for r in grid["resolution"])

    def test_full_preset_adds_neighborhood_grids(self):
        grids = self._grids(preset="full")
        assert len(grids) == 4
        neighborhoods = sorted(tuple(grid["n_neighbors"]) for _, grid in grids)
        assert neighborhoods == [(10, 30), (10, 30), (15,), (15,)]

    def test_grids_expand_under_parameter_grid(self):
        for _, grid in self._grids():
            configs = list(ParameterGrid(grid))
            assert len(configs) == 2
            assert all("resolution" in c for c in configs)


class TestMinClusterSizeGrids:
    def _grids(self, preset="light"):
        return default_estimator_grids(
            np.zeros((20, 3)),
            preset=preset,
            sweep=resolve_sweep(sweep="min_cluster_size", sweep_values=[5, 10, 25]),
        )

    def test_light_returns_one_hdbscan_grid(self):
        grids = self._grids()
        assert len(grids) == 1
        assert grids[0][0] is HDBSCAN
        assert grids[0][1]["cluster_selection_method"] == ["eom"]

    def test_sizes_propagated_as_ints(self):
        grid = self._grids()[0][1]
        assert grid["min_cluster_size"] == [5, 10, 25]
        assert all(isinstance(s, int) for s in grid["min_cluster_size"])

    def test_full_preset_adds_leaf_selection(self):
        grids = self._grids(preset="full")
        assert len(grids) == 2
        methods = sorted(grid["cluster_selection_method"][0] for _, grid in grids)
        assert methods == ["eom", "leaf"]

    def test_no_n_clusters_key(self):
        assert "n_clusters" not in self._grids()[0][1]


class TestSweepDispatch:
    def test_sweep_overrides_n_clusters_argument(self):
        grids = default_estimator_grids(
            np.zeros((20, 3)),
            n_clusters=np.array([2, 3, 4]),
            sweep=resolve_sweep(resolution=[0.5, 1.0]),
        )
        assert {cls for cls, _ in grids} == {LeidenClustering, LouvainClustering}

    def test_defaults_to_k_mode_without_sweep(self):
        grids = default_estimator_grids(np.zeros((20, 3)), n_clusters=np.array([2, 3]))
        assert {cls for cls, _ in grids} == {
            KMeans,
            AgglomerativeClustering,
            SpectralClustering,
        }

    def test_k_mode_via_explicit_sweep_matches_default(self):
        by_arg = default_estimator_grids(np.zeros((20, 3)), n_clusters=np.array([2, 3]))
        by_sweep = default_estimator_grids(
            np.zeros((20, 3)),
            sweep=resolve_sweep(n_clusters=np.array([2, 3])),
        )
        assert [cls for cls, _ in by_arg] == [cls for cls, _ in by_sweep]

    def test_unknown_sweep_param_raises(self):
        sweep = resolve_sweep(
            sweep="min_samples", sweep_values=[3, 5], finer_is_larger=False
        )
        with pytest.raises(ValueError, match="No default estimator grids"):
            default_estimator_grids(np.zeros((20, 3)), sweep=sweep)
