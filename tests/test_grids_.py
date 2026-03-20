"""Tests for carve._grids module."""

import numpy as np
import pytest
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from carve.cluster import SpectralClusteringCARVE
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
        assert len(grids) == 4  # KMeans, Agglomerative, SpectralCARVE x2

    def test_estimator_types(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]))
        classes = [cls for cls, _ in grids]
        assert KMeans in classes
        assert AgglomerativeClustering in classes
        assert SpectralClusteringCARVE in classes

    def test_n_clusters_propagated(self):
        X = np.random.RandomState(0).randn(50, 5)
        ks = np.array([2, 3, 4])
        grids = default_estimator_grids(X, n_clusters=ks)
        for _, grid in grids:
            assert grid["n_clusters"] == [2, 3, 4]

    def test_gamma_values(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]))
        # Find the RBF spectral entry (the one with gamma in its grid)
        rbf_grids = [(cls, g) for cls, g in grids if "gamma" in g]
        assert len(rbf_grids) == 1
        cls, spectral_grid = rbf_grids[0]
        assert cls is SpectralClusteringCARVE
        assert len(spectral_grid["gamma"]) == 3  # 3 multipliers
        assert all(g > 0 for g in spectral_grid["gamma"])

    def test_self_tuning_entry(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]))
        # Find the self-tuning entry (has affinity but no gamma)
        st_grids = [
            (cls, g) for cls, g in grids
            if cls is SpectralClusteringCARVE and "gamma" not in g
        ]
        assert len(st_grids) == 1
        _, grid = st_grids[0]
        assert grid["affinity"] == ["self_tuning"]

    def test_agglomerative_linkages(self):
        X = np.random.RandomState(0).randn(50, 5)
        grids = default_estimator_grids(X, n_clusters=np.array([2, 3]))
        agg_grid = next(g for cls, g in grids if cls is AgglomerativeClustering)
        assert "linkage" in agg_grid
        assert set(agg_grid["linkage"]) == {"ward", "average", "single"}

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
            cls is FunctionTransformer and params == {}
            for cls, params in options
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
            cls is FunctionTransformer and params == {}
            for cls, params in options
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
        pca_option = next(
            (cls, params) for cls, params in options if cls is PCA
        )
        n_components = pca_option[1]["n_components"]
        # Should respect min(min_n, p) where min_n = round(50 * 0.4) - 1
        assert all(c >= 2 for c in n_components)
        assert max(n_components) < 5  # p = 5
