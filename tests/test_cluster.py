"""Tests for carve.cluster (SpectralClustering, Leiden, Louvain)."""

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

from carve.cluster import (
    LeidenClustering,
    LouvainClustering,
    SpectralClustering,
    build_knn_graph,
)

from conftest import requires_graph


@pytest.fixture()
def X_blobs():
    """Well-separated blobs for spectral clustering."""
    rng = np.random.RandomState(42)
    return np.vstack(
        [
            rng.randn(30, 3) + [5, 0, 0],
            rng.randn(30, 3) + [0, 5, 0],
            rng.randn(30, 3) + [0, 0, 5],
        ]
    )


# -----------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------


class TestInterface:
    def test_is_sklearn_estimator(self):
        est = SpectralClustering()
        assert isinstance(est, BaseEstimator)
        assert isinstance(est, ClusterMixin)

    def test_get_params(self):
        est = SpectralClustering(n_clusters=3, affinity="knn")
        params = est.get_params()
        assert params["n_clusters"] == 3
        assert params["affinity"] == "knn"

    def test_set_params(self):
        est = SpectralClustering()
        est.set_params(n_clusters=5)
        assert est.n_clusters == 5

    def test_default_params(self):
        est = SpectralClustering()
        assert est.n_clusters == 2
        assert est.affinity == "self_tuning"
        assert est.n_neighbors == 7
        assert est.scale is True


# -----------------------------------------------------------------------
# fit / fit_predict
# -----------------------------------------------------------------------


class TestFitPredict:
    def test_fit_returns_self(self, X_blobs):
        est = SpectralClustering(n_clusters=3, random_state=0)
        result = est.fit(X_blobs)
        assert result is est

    def test_fit_sets_labels(self, X_blobs):
        est = SpectralClustering(n_clusters=3, random_state=0)
        est.fit(X_blobs)
        assert hasattr(est, "labels_")
        assert est.labels_.shape == (90,)
        assert len(np.unique(est.labels_)) == 3

    def test_fit_predict_returns_labels(self, X_blobs):
        est = SpectralClustering(n_clusters=3, random_state=0)
        labels = est.fit_predict(X_blobs)
        assert labels.shape == (90,)
        assert len(np.unique(labels)) == 3

    def test_fit_predict_matches_fit(self, X_blobs):
        est1 = SpectralClustering(n_clusters=3, random_state=0)
        est1.fit(X_blobs)

        est2 = SpectralClustering(n_clusters=3, random_state=0)
        labels2 = est2.fit_predict(X_blobs)

        np.testing.assert_array_equal(est1.labels_, labels2)

    def test_fit_sets_attributes(self, X_blobs):
        est = SpectralClustering(n_clusters=3, random_state=0)
        est.fit(X_blobs)
        assert est.embedding_ is not None
        assert est.embedding_.shape == (90, 3)
        assert est.affinity_ is not None
        assert est.evals_ is not None

    def test_reproducibility(self, X_blobs):
        l1 = SpectralClustering(n_clusters=3, random_state=42).fit_predict(X_blobs)
        l2 = SpectralClustering(n_clusters=3, random_state=42).fit_predict(X_blobs)
        np.testing.assert_array_equal(l1, l2)

    def test_two_clusters(self):
        rng = np.random.RandomState(0)
        X = np.vstack(
            [
                rng.randn(25, 2) + [5, 0],
                rng.randn(25, 2) + [0, 5],
            ]
        )
        labels = SpectralClustering(n_clusters=2, random_state=0).fit_predict(X)
        assert len(np.unique(labels)) == 2

    def test_y_ignored(self, X_blobs):
        est = SpectralClustering(n_clusters=3, random_state=0)
        labels_no_y = est.fit_predict(X_blobs)
        est2 = SpectralClustering(n_clusters=3, random_state=0)
        labels_with_y = est2.fit_predict(X_blobs, y=np.ones(90))
        np.testing.assert_array_equal(labels_no_y, labels_with_y)


# -----------------------------------------------------------------------
# Affinities
# -----------------------------------------------------------------------


class TestAffinities:
    def test_rbf_affinity(self, X_blobs):
        est = SpectralClustering(n_clusters=3, affinity="rbf", random_state=0)
        est.fit(X_blobs)
        assert est.affinity_.shape == (90, 90)

    def test_knn_affinity(self, X_blobs):
        est = SpectralClustering(n_clusters=3, affinity="knn", random_state=0)
        est.fit(X_blobs)
        assert issparse(est.affinity_)
        assert est.affinity_.shape == (90, 90)

    def test_self_tuning_affinity(self, X_blobs):
        est = SpectralClustering(
            n_clusters=3,
            affinity="self_tuning",
            random_state=0,
        )
        est.fit(X_blobs)
        assert est.affinity_.shape == (90, 90)

    def test_invalid_affinity(self, X_blobs):
        est = SpectralClustering(n_clusters=3, affinity="invalid")
        with pytest.raises(ValueError, match="Unknown affinity"):
            est.fit(X_blobs)

    def test_custom_gamma(self, X_blobs):
        est = SpectralClustering(
            n_clusters=3,
            affinity="rbf",
            gamma=0.1,
            random_state=0,
        )
        est.fit(X_blobs)
        assert est.labels_.shape == (90,)

    def test_no_scaling(self, X_blobs):
        est = SpectralClustering(n_clusters=3, scale=False, random_state=0)
        est.fit(X_blobs)
        assert est.labels_.shape == (90,)


# -----------------------------------------------------------------------
# k-NN gamma heuristic
# -----------------------------------------------------------------------


class TestKnnGammaHeuristic:
    def test_gamma_auto_rbf(self, X_blobs):
        """When gamma=None with rbf affinity, gamma_ should be set."""
        est = SpectralClustering(
            n_clusters=3,
            affinity="rbf",
            gamma=None,
            random_state=0,
        )
        est.fit(X_blobs)
        assert hasattr(est, "gamma_")
        assert est.gamma_ is not None
        assert est.gamma_ > 0

    def test_gamma_auto_knn(self, X_blobs):
        """When gamma=None with knn affinity, gamma_ should be set."""
        est = SpectralClustering(
            n_clusters=3,
            affinity="knn",
            gamma=None,
            random_state=0,
        )
        est.fit(X_blobs)
        assert hasattr(est, "gamma_")
        assert est.gamma_ is not None
        assert est.gamma_ > 0

    def test_gamma_explicit(self, X_blobs):
        """When gamma is explicitly set, gamma_ should match."""
        est = SpectralClustering(
            n_clusters=3,
            affinity="rbf",
            gamma=0.5,
            random_state=0,
        )
        est.fit(X_blobs)
        assert est.gamma_ == 0.5

    def test_gamma_self_tuning_is_none(self, X_blobs):
        """Self-tuning affinity doesn't use a global gamma."""
        est = SpectralClustering(
            n_clusters=3,
            affinity="self_tuning",
            random_state=0,
        )
        est.fit(X_blobs)
        assert est.gamma_ is None


# -----------------------------------------------------------------------
# Non-convex clusters
# -----------------------------------------------------------------------


class TestNonConvex:
    def test_moons_self_tuning(self):
        """Self-tuning spectral clustering should handle moons well."""
        X, y_true = make_moons(500, noise=0.08, random_state=42)
        labels = SpectralClustering(
            n_clusters=2,
            affinity="self_tuning",
            random_state=0,
        ).fit_predict(X)
        ari = adjusted_rand_score(y_true, labels)
        assert ari > 0.95

    def test_moons_rbf_auto_gamma(self):
        """RBF with k-NN gamma heuristic should also handle moons."""
        X, y_true = make_moons(500, noise=0.08, random_state=42)
        labels = SpectralClustering(
            n_clusters=2,
            affinity="rbf",
            gamma=None,
            random_state=0,
        ).fit_predict(X)
        ari = adjusted_rand_score(y_true, labels)
        assert ari > 0.90


# -----------------------------------------------------------------------
# build_knn_graph
# -----------------------------------------------------------------------


@requires_graph
class TestBuildKnnGraph:
    def test_vertex_count(self, X_blobs):
        graph = build_knn_graph(X_blobs, n_neighbors=10)
        assert graph.vcount() == X_blobs.shape[0]

    def test_undirected(self, X_blobs):
        assert not build_knn_graph(X_blobs, n_neighbors=10).is_directed()

    def test_every_edge_is_weighted(self, X_blobs):
        for weighting in ("connectivity", "jaccard"):
            graph = build_knn_graph(X_blobs, n_neighbors=10, weighting=weighting)
            weights = graph.es["weight"]
            assert len(weights) == graph.ecount()
            assert graph.ecount() > 0
            assert all(w > 0 for w in weights)

    def test_connectivity_weights_are_unit(self, X_blobs):
        graph = build_knn_graph(X_blobs, n_neighbors=10, weighting="connectivity")
        assert all(w == pytest.approx(1.0) for w in graph.es["weight"])

    def test_jaccard_weights_are_fractions(self, X_blobs):
        graph = build_knn_graph(X_blobs, n_neighbors=10, weighting="jaccard")
        assert all(0.0 < w <= 1.0 for w in graph.es["weight"])

    def test_n_neighbors_clipped_to_n_minus_one(self):
        X = np.random.RandomState(0).randn(6, 2)
        graph = build_knn_graph(X, n_neighbors=100)
        assert graph.vcount() == 6

    def test_unknown_weighting_raises(self, X_blobs):
        with pytest.raises(ValueError, match="Unknown weighting"):
            build_knn_graph(X_blobs, n_neighbors=10, weighting="bogus")

    def test_more_neighbors_means_more_edges(self, X_blobs):
        few = build_knn_graph(X_blobs, n_neighbors=5).ecount()
        many = build_knn_graph(X_blobs, n_neighbors=20).ecount()
        assert many > few


# -----------------------------------------------------------------------
# LeidenClustering
# -----------------------------------------------------------------------


@requires_graph
class TestLeidenClustering:
    def test_interface(self):
        est = LeidenClustering()
        assert isinstance(est, BaseEstimator)
        assert isinstance(est, ClusterMixin)

    def test_defaults(self):
        est = LeidenClustering()
        assert est.resolution == 1.0
        assert est.n_neighbors == 15
        assert est.metric == "euclidean"
        assert est.weighting == "connectivity"
        assert est.objective_function == "modularity"
        assert est.n_iterations == -1
        assert est.random_state is None
        assert est.scale is False

    def test_get_params_round_trips_through_clone(self):
        est = LeidenClustering(resolution=0.7, n_neighbors=12, scale=True)
        cloned = clone(est)
        assert cloned.get_params() == est.get_params()

    def test_fit_returns_self(self, X_blobs):
        est = LeidenClustering(random_state=0)
        assert est.fit(X_blobs) is est

    def test_fit_sets_attributes(self, X_blobs):
        est = LeidenClustering(random_state=0).fit(X_blobs)
        assert est.labels_.shape == (X_blobs.shape[0],)
        assert est.n_clusters_ == np.unique(est.labels_).size
        assert isinstance(est.quality_, float)
        assert est.graph_.vcount() == X_blobs.shape[0]

    def test_recovers_three_blobs(self, X_blobs):
        truth = np.repeat([0, 1, 2], 30)
        labels = LeidenClustering(random_state=0).fit_predict(X_blobs)
        assert adjusted_rand_score(truth, labels) > 0.9

    def test_fit_predict_matches_labels(self, X_blobs):
        est = LeidenClustering(random_state=0)
        np.testing.assert_array_equal(
            est.fit_predict(X_blobs), est.fit(X_blobs).labels_
        )

    def test_no_negative_labels(self, X_blobs):
        """Leiden assigns every point; noise handling is not needed."""
        labels = LeidenClustering(random_state=0).fit_predict(X_blobs)
        assert (labels >= 0).all()

    def test_resolution_is_monotone(self, X_blobs):
        counts = [
            LeidenClustering(resolution=r, random_state=0).fit(X_blobs).n_clusters_
            for r in (0.1, 0.5, 1.0, 3.0)
        ]
        assert counts == sorted(counts)
        assert counts[-1] > counts[0]

    def test_reproducible(self, X_blobs):
        a = LeidenClustering(resolution=1.5, random_state=7).fit_predict(X_blobs)
        b = LeidenClustering(resolution=1.5, random_state=7).fit_predict(X_blobs)
        np.testing.assert_array_equal(a, b)

    def test_cpm_objective_runs(self, X_blobs):
        est = LeidenClustering(
            objective_function="cpm", resolution=0.1, random_state=0
        ).fit(X_blobs)
        assert est.n_clusters_ >= 1

    def test_unknown_objective_raises_at_fit(self, X_blobs):
        est = LeidenClustering(objective_function="bogus")
        with pytest.raises(ValueError, match="Unknown objective_function"):
            est.fit(X_blobs)

    def test_scale_option_runs(self, X_blobs):
        est = LeidenClustering(scale=True, random_state=0).fit(X_blobs)
        assert est.labels_.shape == (X_blobs.shape[0],)

    def test_jaccard_weighting_runs(self, X_blobs):
        est = LeidenClustering(weighting="jaccard", random_state=0).fit(X_blobs)
        assert est.n_clusters_ >= 1

    def test_y_argument_ignored(self, X_blobs):
        est = LeidenClustering(random_state=0)
        np.testing.assert_array_equal(
            est.fit_predict(X_blobs, np.zeros(X_blobs.shape[0])),
            est.fit_predict(X_blobs),
        )


# -----------------------------------------------------------------------
# LouvainClustering
# -----------------------------------------------------------------------


@requires_graph
class TestLouvainClustering:
    def test_interface(self):
        est = LouvainClustering()
        assert isinstance(est, BaseEstimator)
        assert isinstance(est, ClusterMixin)

    def test_defaults(self):
        est = LouvainClustering()
        assert est.resolution == 1.0
        assert est.n_neighbors == 15
        assert est.metric == "euclidean"
        assert est.weighting == "connectivity"
        assert est.scale is False

    def test_get_params_round_trips_through_clone(self):
        est = LouvainClustering(resolution=0.7, n_neighbors=12)
        assert clone(est).get_params() == est.get_params()

    def test_has_no_random_state(self):
        """igraph's community_multilevel takes no seed."""
        assert "random_state" not in LouvainClustering().get_params()

    def test_fit_returns_self(self, X_blobs):
        est = LouvainClustering()
        assert est.fit(X_blobs) is est

    def test_fit_sets_attributes(self, X_blobs):
        est = LouvainClustering().fit(X_blobs)
        assert est.labels_.shape == (X_blobs.shape[0],)
        assert est.n_clusters_ == np.unique(est.labels_).size
        assert isinstance(est.modularity_, float)
        assert est.graph_.vcount() == X_blobs.shape[0]

    def test_recovers_three_blobs(self, X_blobs):
        truth = np.repeat([0, 1, 2], 30)
        labels = LouvainClustering().fit_predict(X_blobs)
        assert adjusted_rand_score(truth, labels) > 0.9

    def test_fit_predict_matches_labels(self, X_blobs):
        est = LouvainClustering()
        np.testing.assert_array_equal(
            est.fit_predict(X_blobs), est.fit(X_blobs).labels_
        )

    def test_no_negative_labels(self, X_blobs):
        assert (LouvainClustering().fit_predict(X_blobs) >= 0).all()

    def test_resolution_is_monotone(self, X_blobs):
        counts = [
            LouvainClustering(resolution=r).fit(X_blobs).n_clusters_
            for r in (0.1, 0.5, 1.0, 3.0)
        ]
        assert counts == sorted(counts)

    def test_deterministic(self, X_blobs):
        """No seed, but the graph is deterministic given X."""
        np.testing.assert_array_equal(
            LouvainClustering(resolution=1.5).fit_predict(X_blobs),
            LouvainClustering(resolution=1.5).fit_predict(X_blobs),
        )

    def test_jaccard_weighting_runs(self, X_blobs):
        est = LouvainClustering(weighting="jaccard").fit(X_blobs)
        assert est.n_clusters_ >= 1

    def test_scale_option_runs(self, X_blobs):
        est = LouvainClustering(scale=True).fit(X_blobs)
        assert est.labels_.shape == (X_blobs.shape[0],)
