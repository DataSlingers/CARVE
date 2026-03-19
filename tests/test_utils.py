"""Tests for carve._utils module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans, AgglomerativeClustering

from carve._utils import (
    split_subsample_indices,
    _coerce_n_clusters,
    _summarize_ari_scores,
    cluster_labels,
    align_cluster_labels,
    ensure_2d_array,
    summarize_preprocessing_records,
)


# -----------------------------------------------------------------------
# split_subsample_indices
# -----------------------------------------------------------------------

class TestSplitSubsampleIndices:
    def test_basic_split(self):
        train, test = split_subsample_indices(100, subsample_ratio=0.8, random_state=0)
        assert len(train) == 80
        assert len(test) == 20
        assert len(np.intersect1d(train, test)) == 0

    def test_all_indices_covered(self):
        train, test = split_subsample_indices(50, subsample_ratio=0.6, random_state=1)
        combined = np.sort(np.concatenate([train, test]))
        np.testing.assert_array_equal(combined, np.arange(50))

    def test_reproducibility(self):
        t1, _ = split_subsample_indices(100, subsample_ratio=0.8, random_state=42)
        t2, _ = split_subsample_indices(100, subsample_ratio=0.8, random_state=42)
        np.testing.assert_array_equal(t1, t2)

    def test_different_seeds_differ(self):
        t1, _ = split_subsample_indices(100, subsample_ratio=0.8, random_state=0)
        t2, _ = split_subsample_indices(100, subsample_ratio=0.8, random_state=1)
        assert not np.array_equal(t1, t2)

    def test_various_ratios(self):
        for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            train, test = split_subsample_indices(
                100, subsample_ratio=ratio, random_state=0,
            )
            assert len(train) == int(np.float64(ratio * 100))
            assert len(test) == 100 - len(train)

    def test_small_n(self):
        train, test = split_subsample_indices(5, subsample_ratio=0.6, random_state=0)
        assert len(train) + len(test) == 5


# -----------------------------------------------------------------------
# _coerce_n_clusters
# -----------------------------------------------------------------------

class TestCoerceNClusters:
    def test_int_input(self):
        result = _coerce_n_clusters(5)
        np.testing.assert_array_equal(result, np.array([2, 3, 4, 5]))

    def test_int_min(self):
        result = _coerce_n_clusters(2)
        np.testing.assert_array_equal(result, np.array([2]))

    def test_numpy_int(self):
        result = _coerce_n_clusters(np.int64(4))
        np.testing.assert_array_equal(result, np.array([2, 3, 4]))

    def test_array_input(self):
        arr = np.array([3, 5, 7])
        result = _coerce_n_clusters(arr)
        np.testing.assert_array_equal(result, arr)

    def test_int_too_small(self):
        with pytest.raises(ValueError, match="n_clusters int must be >= 2"):
            _coerce_n_clusters(1)

    def test_array_too_small(self):
        with pytest.raises(ValueError, match="All n_clusters values must be >= 2"):
            _coerce_n_clusters(np.array([1, 2, 3]))

    def test_array_not_1d(self):
        with pytest.raises(ValueError, match="1D array"):
            _coerce_n_clusters(np.array([[2, 3]]))

    def test_array_not_integer(self):
        with pytest.raises(TypeError, match="integers"):
            _coerce_n_clusters(np.array([2.5, 3.5]))

    def test_output_dtype(self):
        result = _coerce_n_clusters(5)
        assert result.dtype == int


# -----------------------------------------------------------------------
# _summarize_ari_scores
# -----------------------------------------------------------------------

class TestSummarizeAriScores:
    def test_normal(self):
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        mean, se, q95, q05 = _summarize_ari_scores(scores, 5)
        assert abs(mean - np.mean(scores)) < 1e-10
        assert se > 0
        assert q95 >= q05

    def test_all_nan(self):
        mean, se, q95, q05 = _summarize_ari_scores([np.nan, np.nan], 2)
        assert np.isnan(mean)
        assert np.isnan(se)

    def test_single_value(self):
        mean, se, q95, q05 = _summarize_ari_scores([0.5], 1)
        assert mean == 0.5
        assert np.isnan(se)  # can't compute SE with 1 value

    def test_with_nans(self):
        scores = [0.8, np.nan, 0.9, np.nan, 0.85]
        mean, se, q95, q05 = _summarize_ari_scores(scores, 5)
        assert abs(mean - np.nanmean(scores)) < 1e-10
        assert se > 0

    def test_identical_values(self):
        mean, se, q95, q05 = _summarize_ari_scores([0.5, 0.5, 0.5], 3)
        assert mean == 0.5
        assert abs(se) < 1e-10
        assert q95 == 0.5
        assert q05 == 0.5


# -----------------------------------------------------------------------
# cluster_labels
# -----------------------------------------------------------------------

class TestClusterLabels:
    def test_kmeans(self, X_two_clusters):
        labels = cluster_labels(X_two_clusters, KMeans, random_state=0, n_clusters=2)
        assert labels.shape == (60,)
        assert len(np.unique(labels)) == 2

    def test_agglomerative(self, X_two_clusters):
        labels = cluster_labels(
            X_two_clusters, AgglomerativeClustering, n_clusters=2,
        )
        assert labels.shape == (60,)
        assert len(np.unique(labels)) == 2

    def test_reproducibility(self, X_two_clusters):
        l1 = cluster_labels(X_two_clusters, KMeans, random_state=42, n_clusters=2)
        l2 = cluster_labels(X_two_clusters, KMeans, random_state=42, n_clusters=2)
        np.testing.assert_array_equal(l1, l2)


# -----------------------------------------------------------------------
# align_cluster_labels
# -----------------------------------------------------------------------

class TestAlignClusterLabels:
    def test_already_aligned(self):
        ref = np.array([0, 0, 1, 1, 2, 2])
        labels = np.array([0, 0, 1, 1, 2, 2])
        aligned = align_cluster_labels(ref, labels)
        np.testing.assert_array_equal(aligned, ref)

    def test_permuted(self):
        ref = np.array([0, 0, 0, 1, 1, 1])
        labels = np.array([1, 1, 1, 0, 0, 0])  # swapped
        aligned = align_cluster_labels(ref, labels)
        np.testing.assert_array_equal(aligned, ref)

    def test_three_clusters_permuted(self):
        ref = np.array([0, 0, 1, 1, 2, 2])
        labels = np.array([2, 2, 0, 0, 1, 1])  # rotated
        aligned = align_cluster_labels(ref, labels)
        np.testing.assert_array_equal(aligned, ref)

    def test_output_dtype_matches_reference(self):
        ref = np.array([0, 0, 1, 1], dtype=np.int32)
        labels = np.array([1, 1, 0, 0], dtype=np.int64)
        aligned = align_cluster_labels(ref, labels)
        assert aligned.dtype == ref.dtype


# -----------------------------------------------------------------------
# ensure_2d_array
# -----------------------------------------------------------------------

class TestEnsure2dArray:
    def test_2d_ndarray(self):
        X = np.array([[1, 2], [3, 4]])
        result = ensure_2d_array(X)
        np.testing.assert_array_equal(result, X)

    def test_1d_ndarray(self):
        X = np.array([1, 2, 3])
        result = ensure_2d_array(X)
        assert result.shape == (3, 1)

    def test_3d_ndarray_raises(self):
        X = np.array([[[1, 2], [3, 4]]])
        with pytest.raises(ValueError, match="1D or 2D"):
            ensure_2d_array(X)

    def test_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = ensure_2d_array(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_list(self):
        X = [[1, 2], [3, 4]]
        result = ensure_2d_array(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            ensure_2d_array("not an array")
