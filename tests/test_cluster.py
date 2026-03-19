"""Tests for carve.cluster (SpectralClusteringCARVE)."""

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClusterMixin

from carve.cluster import SpectralClusteringCARVE


@pytest.fixture()
def X_blobs():
    """Well-separated blobs for spectral clustering."""
    rng = np.random.RandomState(42)
    return np.vstack([
        rng.randn(30, 3) + [5, 0, 0],
        rng.randn(30, 3) + [0, 5, 0],
        rng.randn(30, 3) + [0, 0, 5],
    ])


# -----------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------

class TestInterface:
    def test_is_sklearn_estimator(self):
        est = SpectralClusteringCARVE()
        assert isinstance(est, BaseEstimator)
        assert isinstance(est, ClusterMixin)

    def test_get_params(self):
        est = SpectralClusteringCARVE(n_clusters=3, affinity="knn")
        params = est.get_params()
        assert params["n_clusters"] == 3
        assert params["affinity"] == "knn"

    def test_set_params(self):
        est = SpectralClusteringCARVE()
        est.set_params(n_clusters=5)
        assert est.n_clusters == 5

    def test_default_params(self):
        est = SpectralClusteringCARVE()
        assert est.n_clusters == 2
        assert est.affinity == "rbf"
        assert est.normalized is True
        assert est.scale is True
        assert est.converged_ is None
        assert est.used_dense_fallback_ is False


# -----------------------------------------------------------------------
# fit / fit_predict
# -----------------------------------------------------------------------

class TestFitPredict:
    def test_fit_returns_self(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, random_state=0)
        result = est.fit(X_blobs)
        assert result is est

    def test_fit_sets_labels(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, random_state=0)
        est.fit(X_blobs)
        assert hasattr(est, "labels_")
        assert est.labels_.shape == (90,)
        assert len(np.unique(est.labels_)) == 3

    def test_fit_predict_returns_labels(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, random_state=0)
        labels = est.fit_predict(X_blobs)
        assert labels.shape == (90,)
        assert len(np.unique(labels)) == 3

    def test_fit_predict_matches_fit(self, X_blobs):
        est1 = SpectralClusteringCARVE(n_clusters=3, random_state=0)
        est1.fit(X_blobs)

        est2 = SpectralClusteringCARVE(n_clusters=3, random_state=0)
        labels2 = est2.fit_predict(X_blobs)

        np.testing.assert_array_equal(est1.labels_, labels2)

    def test_fit_sets_attributes(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, random_state=0)
        est.fit(X_blobs)
        assert est.embedding_ is not None
        assert est.embedding_.shape == (90, 3)
        assert est.affinity_ is not None
        assert est.evals_ is not None
        assert est.converged_ is not None

    def test_reproducibility(self, X_blobs):
        l1 = SpectralClusteringCARVE(n_clusters=3, random_state=42).fit_predict(X_blobs)
        l2 = SpectralClusteringCARVE(n_clusters=3, random_state=42).fit_predict(X_blobs)
        np.testing.assert_array_equal(l1, l2)

    def test_two_clusters(self):
        rng = np.random.RandomState(0)
        X = np.vstack([
            rng.randn(25, 2) + [5, 0],
            rng.randn(25, 2) + [0, 5],
        ])
        labels = SpectralClusteringCARVE(n_clusters=2, random_state=0).fit_predict(X)
        assert len(np.unique(labels)) == 2

    def test_y_ignored(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, random_state=0)
        labels_no_y = est.fit_predict(X_blobs)
        est2 = SpectralClusteringCARVE(n_clusters=3, random_state=0)
        labels_with_y = est2.fit_predict(X_blobs, y=np.ones(90))
        np.testing.assert_array_equal(labels_no_y, labels_with_y)


# -----------------------------------------------------------------------
# Affinities
# -----------------------------------------------------------------------

class TestAffinities:
    def test_rbf_affinity(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, affinity="rbf", random_state=0)
        est.fit(X_blobs)
        assert issparse(est.affinity_)
        assert est.affinity_.shape == (90, 90)

    def test_knn_affinity(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, affinity="knn", random_state=0)
        est.fit(X_blobs)
        assert issparse(est.affinity_)
        assert est.affinity_.shape == (90, 90)

    def test_self_tuning_affinity(self, X_blobs):
        est = SpectralClusteringCARVE(
            n_clusters=3, affinity="self_tuning", random_state=0,
        )
        est.fit(X_blobs)
        assert issparse(est.affinity_)
        assert est.affinity_.shape == (90, 90)

    def test_invalid_affinity(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, affinity="invalid")
        with pytest.raises(ValueError, match="Unknown affinity"):
            est.fit(X_blobs)

    def test_custom_gamma(self, X_blobs):
        est = SpectralClusteringCARVE(
            n_clusters=3, affinity="rbf", gamma=0.1, random_state=0,
        )
        est.fit(X_blobs)
        assert est.labels_.shape == (90,)


# -----------------------------------------------------------------------
# Normalized vs unnormalized Laplacian
# -----------------------------------------------------------------------

class TestLaplacian:
    def test_normalized(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, normalized=True, random_state=0)
        est.fit(X_blobs)
        assert est.labels_.shape == (90,)

    def test_unnormalized(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, normalized=False, random_state=0)
        est.fit(X_blobs)
        assert est.labels_.shape == (90,)

    def test_no_scaling(self, X_blobs):
        est = SpectralClusteringCARVE(n_clusters=3, scale=False, random_state=0)
        est.fit(X_blobs)
        assert est.labels_.shape == (90,)
