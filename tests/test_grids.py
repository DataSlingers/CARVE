"""Tests for carve._grids module."""

import importlib.util

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from carve._grids import (
    default_dim_reduction_options,
    default_estimator_grids,
    default_normalization_options,
)
from carve._sweep import resolve_sweep
from carve.cluster import LeidenClustering, LouvainClustering, SpectralClustering

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
    def test_nonnegative_input_offers_log1p(self):
        X = np.abs(np.random.RandomState(0).randn(20, 3))
        assert default_normalization_options(X) == [
            (FunctionTransformer, {}),
            (StandardScaler, {}),
            (FunctionTransformer, {"func": [np.log1p]}),
        ]

    def test_zero_minimum_still_offers_log1p(self):
        X = np.zeros((5, 2))
        X[0, 0] = 3.0
        options = default_normalization_options(X)
        assert (FunctionTransformer, {"func": [np.log1p]}) in options

    def test_negative_input_omits_log1p_with_a_warning(self):
        X = np.random.RandomState(0).randn(20, 3)
        with pytest.warns(
            UserWarning, match=r"negative values \(minimum -[0-9.]+\), so log1p is omitted"
        ):
            options = default_normalization_options(X)
        assert options == [(FunctionTransformer, {}), (StandardScaler, {})]


# -----------------------------------------------------------------------
# default_dim_reduction_options
# -----------------------------------------------------------------------


@pytest.fixture()
def no_umap(monkeypatch):
    """Make umap-learn look absent, so the grids do not depend on the extra."""
    monkeypatch.setattr("carve._grids.importlib.util.find_spec", lambda name: None)


class TestDefaultDimReductionOptions:
    def test_structure(self):
        X = np.random.RandomState(0).randn(100, 10)
        options = default_dim_reduction_options(X)
        assert isinstance(options, list)
        # identity, PCA, t-SNE, and UMAP only when the [umap] extra is present.
        expected = 4 if importlib.util.find_spec("umap") is not None else 3
        assert len(options) == expected

    def test_contains_umap_when_installed(self):
        pytest.importorskip("umap")
        from umap import UMAP

        X = np.random.RandomState(0).randn(100, 10)
        options = default_dim_reduction_options(X)
        assert any(cls is UMAP for cls, _ in options)

    def test_warns_when_umap_missing(self, no_umap):
        X = np.random.RandomState(0).randn(100, 10)
        with pytest.warns(UserWarning, match="umap-learn is not installed"):
            options = default_dim_reduction_options(X)
        assert len(options) == 3

    def test_grids_are_discrete_and_filtered(self, no_umap):
        # n=60 at ratio 0.618: 37 training rows and 23 held out, so n_min=23
        # and the PCA limit is min(23, p=10) = 10.
        X = np.random.RandomState(0).randn(60, 10)
        with pytest.warns(UserWarning, match="umap-learn is not installed"):
            options = default_dim_reduction_options(X, subsample_ratio=0.618)
        assert options == [
            (FunctionTransformer, {}),
            (PCA, {"n_components": [2, 5]}),
            (TSNE, {"n_components": [2], "perplexity": [15]}),
        ]

    def test_wide_margin_keeps_every_candidate(self, no_umap):
        # n=400 at ratio 0.618: n_min=153, p=60.
        X = np.random.RandomState(0).randn(400, 60)
        with pytest.warns(UserWarning, match="umap-learn is not installed"):
            grids = dict(default_dim_reduction_options(X, subsample_ratio=0.618))
        assert grids[PCA] == {"n_components": [2, 5, 10, 20, 50]}
        assert grids[TSNE] == {"n_components": [2], "perplexity": [15, 30, 50]}

    def test_limit_is_the_smaller_subsample_at_low_ratios(self, no_umap):
        # Ratio 0.3 on n=100: 30 training rows and 70 held out, so the limit
        # is 30, not the held-out 70.
        X = np.random.RandomState(0).randn(100, 60)
        with pytest.warns(UserWarning, match="umap-learn is not installed"):
            grids = dict(default_dim_reduction_options(X, subsample_ratio=0.3))
        assert grids[PCA] == {"n_components": [2, 5, 10, 20]}
        assert grids[TSNE]["perplexity"] == [15]

    def test_umap_grid_is_filtered(self):
        pytest.importorskip("umap")
        from umap import UMAP

        X = np.random.RandomState(0).randn(60, 10)
        grids = dict(default_dim_reduction_options(X, subsample_ratio=0.618))
        assert grids[UMAP] == {
            "n_components": [2],
            "n_neighbors": [15],
            "min_dist": [0.1],
        }

    def test_tsne_with_an_empty_grid_is_omitted_with_a_warning(self, no_umap):
        # n=30 at ratio 0.618: 18 training rows and 12 held out; no candidate
        # perplexity is below 12.
        X = np.random.RandomState(0).randn(30, 10)
        with pytest.warns(UserWarning) as record:
            options = default_dim_reduction_options(X, subsample_ratio=0.618)
        messages = [str(w.message) for w in record]
        assert any(
            "TSNE is omitted" in m and "no candidate perplexity is below 12" in m
            for m in messages
        )
        assert [cls for cls, _ in options] == [FunctionTransformer, PCA]

    def test_pca_with_an_empty_grid_is_omitted_with_a_warning(self, no_umap):
        X = np.random.RandomState(0).randn(100, 2)
        with pytest.warns(UserWarning) as record:
            options = default_dim_reduction_options(X, subsample_ratio=0.618)
        assert any("PCA is omitted" in str(w.message) for w in record)
        assert [cls for cls, _ in options] == [FunctionTransformer, TSNE]

    def test_umap_with_an_empty_grid_is_omitted_with_a_warning(self):
        pytest.importorskip("umap")
        X = np.random.RandomState(0).randn(30, 10)
        with pytest.warns(UserWarning) as record:
            options = default_dim_reduction_options(X, subsample_ratio=0.618)
        assert any("UMAP is omitted" in str(w.message) for w in record)
        assert [cls for cls, _ in options] == [FunctionTransformer, PCA]

    def test_contains_identity(self):
        X = np.random.RandomState(0).randn(100, 10)
        options = default_dim_reduction_options(X)
        has_identity = any(
            cls is FunctionTransformer and params == {} for cls, params in options
        )
        assert has_identity

    def test_pca_components_respect_data(self):
        X = np.random.RandomState(0).randn(50, 5)
        options = default_dim_reduction_options(X, subsample_ratio=0.6)
        pca_option = next((cls, params) for cls, params in options if cls is PCA)
        n_components = pca_option[1]["n_components"]
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
