"""Tests for the AnnData bridge and sparse input handling."""

import warnings

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from carve import CARVE
from carve._anndata import (
    get_representation,
    is_anndata,
    labels_to_categorical,
    resolve_basis,
)
from carve._utils import ensure_2d_array

SPARSE_MATRIX_FORMATS = ["csr", "csc", "coo", "lil", "dok"]


def _as_sparse(X, fmt):
    return getattr(sparse, f"{fmt}_matrix")(X)


# -----------------------------------------------------------------------
# ensure_2d_array: sparse support
# -----------------------------------------------------------------------


class TestEnsure2dArraySparse:
    @pytest.fixture
    def X(self):
        return np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])

    @pytest.mark.parametrize("fmt", SPARSE_MATRIX_FORMATS)
    def test_spmatrix_densified(self, X, fmt):
        out = ensure_2d_array(_as_sparse(X, fmt))
        assert isinstance(out, np.ndarray)
        assert not sparse.issparse(out)
        np.testing.assert_allclose(out, X)

    @pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
    def test_sparray_densified(self, X, fmt):
        """The newer sparse *array* classes must work too, not just matrices."""
        arr = getattr(sparse, f"{fmt}_array")(X)
        np.testing.assert_allclose(ensure_2d_array(arr), X)

    def test_shape_preserved_non_square(self):
        X = np.arange(12, dtype=float).reshape(3, 4)
        assert ensure_2d_array(sparse.csr_matrix(X)).shape == (3, 4)

    def test_warns_above_threshold(self, X):
        with pytest.warns(UserWarning, match="Densifying a sparse matrix"):
            ensure_2d_array(sparse.csr_matrix(X), dense_warn_elements=2)

    def test_silent_below_threshold(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ensure_2d_array(sparse.csr_matrix(X), dense_warn_elements=10**9)

    def test_1d_sparse_rejected(self):
        arr = sparse.csr_array(np.array([1.0, 0.0, 2.0]))
        if arr.ndim == 2:  # older scipy keeps this 2D
            pytest.skip("this scipy builds a 2D array from a 1D input")
        with pytest.raises(ValueError, match="must be 2D"):
            ensure_2d_array(arr)

    def test_invalid_type_still_rejected(self):
        """A bare string must not sneak through as a 0-d array."""
        with pytest.raises(ValueError, match="Input must be"):
            ensure_2d_array("not an array")

    def test_error_message_mentions_sparse(self):
        with pytest.raises(ValueError, match="sparse"):
            ensure_2d_array(object())


# -----------------------------------------------------------------------
# get_representation
# -----------------------------------------------------------------------


class TestGetRepresentation:
    def test_use_rep_obsm(self, adata_three_clusters):
        out = get_representation(adata_three_clusters, use_rep="X_pca")
        np.testing.assert_allclose(out, adata_three_clusters.obsm["X_pca"])

    def test_use_rep_x(self, adata_three_clusters):
        out = get_representation(adata_three_clusters, use_rep="X")
        np.testing.assert_allclose(out, adata_three_clusters.X)

    def test_layer(self, adata_three_clusters):
        out = get_representation(adata_three_clusters, layer="counts")
        np.testing.assert_allclose(out, adata_three_clusters.layers["counts"])

    def test_defaults_to_x_pca(self, adata_three_clusters):
        out = get_representation(adata_three_clusters)
        np.testing.assert_allclose(out, adata_three_clusters.obsm["X_pca"])

    def test_defaults_to_x_when_no_pca(self, adata_three_clusters):
        del adata_three_clusters.obsm["X_pca"]
        out = get_representation(adata_three_clusters)
        np.testing.assert_allclose(out, adata_three_clusters.X)

    def test_warns_on_wide_x_without_pca(self):
        adata = ad.AnnData(np.random.RandomState(0).randn(10, 80))
        with pytest.warns(UserWarning, match="sc.pp.pca"):
            get_representation(adata)

    def test_no_warning_on_narrow_x(self):
        adata = ad.AnnData(np.random.RandomState(0).randn(10, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            get_representation(adata)

    def test_sparse_x_densified(self, adata_sparse):
        out = get_representation(adata_sparse, use_rep="X")
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, adata_sparse.X.toarray())

    def test_n_pcs_truncates(self, adata_three_clusters):
        out = get_representation(adata_three_clusters, use_rep="X_pca", n_pcs=2)
        assert out.shape[1] == 2

    def test_never_computes_pca_implicitly(self, adata_three_clusters):
        """Asking for X must give X, not a silently reduced version of it."""
        out = get_representation(adata_three_clusters, use_rep="X")
        assert out.shape[1] == adata_three_clusters.n_vars

    def test_use_rep_and_layer_conflict(self, adata_three_clusters):
        with pytest.raises(ValueError, match="at most one"):
            get_representation(adata_three_clusters, use_rep="X_pca", layer="counts")

    def test_unknown_use_rep_lists_available(self, adata_three_clusters):
        with pytest.raises(ValueError, match="X_pca"):
            get_representation(adata_three_clusters, use_rep="X_nope")

    def test_unknown_layer_lists_available(self, adata_three_clusters):
        with pytest.raises(ValueError, match="counts"):
            get_representation(adata_three_clusters, layer="nope")

    def test_n_pcs_too_large(self, adata_three_clusters):
        with pytest.raises(ValueError, match="exceeds"):
            get_representation(adata_three_clusters, use_rep="X_pca", n_pcs=999)


# -----------------------------------------------------------------------
# resolve_basis
# -----------------------------------------------------------------------


class TestResolveBasis:
    @pytest.mark.parametrize("basis", ["umap", "X_umap"])
    def test_prefix_optional(self, adata_three_clusters, basis):
        out = resolve_basis(adata_three_clusters, basis)
        np.testing.assert_allclose(out, adata_three_clusters.obsm["X_umap"][:, :2])

    def test_default_prefers_umap(self, adata_three_clusters):
        out = resolve_basis(adata_three_clusters)
        np.testing.assert_allclose(out, adata_three_clusters.obsm["X_umap"][:, :2])

    def test_default_falls_back_to_pca(self, adata_three_clusters):
        del adata_three_clusters.obsm["X_umap"]
        out = resolve_basis(adata_three_clusters)
        np.testing.assert_allclose(out, adata_three_clusters.obsm["X_pca"][:, :2])

    def test_returns_none_when_nothing_available(self):
        adata = ad.AnnData(np.zeros((4, 3), dtype=np.float32))
        assert resolve_basis(adata) is None

    def test_explicit_fallback_used(self, adata_three_clusters):
        del adata_three_clusters.obsm["X_umap"]
        del adata_three_clusters.obsm["X_pca"]
        adata_three_clusters.obsm["custom"] = np.zeros(
            (adata_three_clusters.n_obs, 2), dtype=np.float32
        )
        out = resolve_basis(adata_three_clusters, fallback="custom")
        assert out.shape == (adata_three_clusters.n_obs, 2)

    def test_truncates_to_two_columns(self, adata_three_clusters):
        assert resolve_basis(adata_three_clusters, "pca").shape[1] == 2

    def test_missing_basis_raises(self, adata_three_clusters):
        with pytest.raises(ValueError, match="X_nope"):
            resolve_basis(adata_three_clusters, "nope")

    def test_one_column_basis_raises(self, adata_three_clusters):
        adata_three_clusters.obsm["X_thin"] = np.zeros(
            (adata_three_clusters.n_obs, 1), dtype=np.float32
        )
        with pytest.raises(ValueError, match="at least two columns"):
            resolve_basis(adata_three_clusters, "thin")


# -----------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------


class TestHelpers:
    def test_is_anndata(self, adata_three_clusters):
        assert is_anndata(adata_three_clusters)
        assert not is_anndata(np.zeros((3, 3)))

    def test_labels_sorted_numerically(self):
        """ "10" must follow "9", as it does for sc.tl.leiden."""
        cat = labels_to_categorical(np.arange(12))
        assert list(cat.categories) == [str(i) for i in range(12)]

    def test_labels_are_strings(self):
        cat = labels_to_categorical(np.array([0, 1, 1, 0]))
        assert all(isinstance(c, str) for c in cat.categories)


# -----------------------------------------------------------------------
# CARVE.fit accepting AnnData
# -----------------------------------------------------------------------


class TestFitAnnData:
    def _model(self):
        return CARVE(n_clusters=np.arange(2, 5), n_resamples=4, random_state=0)

    def test_matches_raw_array(self, adata_three_clusters):
        """fit(adata, use_rep=...) must equal fit on the extracted array."""
        via_adata = self._model().fit(adata_three_clusters, use_rep="X_pca")
        via_array = self._model().fit(adata_three_clusters.obsm["X_pca"])
        assert via_adata.estimator_results_.equals(via_array.estimator_results_)

    def test_sparse_x_matches_dense(self, adata_sparse):
        via_sparse = self._model().fit(adata_sparse, use_rep="X")
        via_dense = self._model().fit(adata_sparse.X.toarray())
        assert via_sparse.estimator_results_.equals(via_dense.estimator_results_)

    def test_layer_supported(self, adata_three_clusters):
        model = self._model().fit(adata_three_clusters, layer="counts")
        assert model.estimator_results_ is not None

    def test_n_pcs_applied(self, adata_three_clusters):
        model = self._model().fit(adata_three_clusters, use_rep="X_pca", n_pcs=2)
        assert model.X_.shape[1] == 2

    @pytest.mark.parametrize("kwarg", ["use_rep", "layer", "n_pcs"])
    def test_rejects_selectors_for_array_input(self, X_three_clusters, kwarg):
        value = 2 if kwarg == "n_pcs" else "X_pca"
        with pytest.raises(ValueError, match="not valid when X is an array"):
            self._model().fit(X_three_clusters, **{kwarg: value})
