"""Tests for carve._utils module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans, AgglomerativeClustering

from carve._runner import ResampleResult
from carve._utils import (
    split_subsample_indices,
    _coerce_n_clusters,
    _summarize_ari_scores,
    align_cluster_labels,
    apply_noise_policy,
    cluster_labels,
    count_clusters,
    ensure_2d_array,
    resolve_anchors,
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
                100,
                subsample_ratio=ratio,
                random_state=0,
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
            X_two_clusters,
            AgglomerativeClustering,
            n_clusters=2,
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


# -----------------------------------------------------------------------
# count_clusters
# -----------------------------------------------------------------------


class TestCountClusters:
    def test_basic(self):
        assert count_clusters(np.array([0, 0, 1, 1, 2])) == 3

    def test_ignores_noise(self):
        """Density-based methods mark unassigned points with -1."""
        assert count_clusters(np.array([-1, -1, 0, 0, 1])) == 2

    def test_all_noise(self):
        assert count_clusters(np.array([-1, -1, -1])) == 0

    def test_none(self):
        assert count_clusters(None) == 0

    def test_empty(self):
        assert count_clusters(np.array([])) == 0

    def test_non_contiguous_labels(self):
        assert count_clusters(np.array([0, 5, 5, 9])) == 3


# -----------------------------------------------------------------------
# apply_noise_policy
# -----------------------------------------------------------------------


class TestApplyNoisePolicy:
    def _noisy(self):
        indices = np.array([10, 11, 12, 13, 14])
        labels = np.array([0, -1, 1, -1, 1])
        return indices, labels

    def test_drop_shrinks_both_arrays(self):
        indices, labels = self._noisy()
        idx, lab, frac = apply_noise_policy(indices, labels, "drop")
        np.testing.assert_array_equal(idx, [10, 12, 14])
        np.testing.assert_array_equal(lab, [0, 1, 1])
        assert frac == pytest.approx(0.4)
        assert len(idx) == len(lab)

    def test_drop_is_the_default(self):
        indices, labels = self._noisy()
        assert np.array_equal(
            apply_noise_policy(indices, labels)[1],
            apply_noise_policy(indices, labels, "drop")[1],
        )

    def test_as_cluster_is_a_no_op(self):
        indices, labels = self._noisy()
        idx, lab, frac = apply_noise_policy(indices, labels, "as_cluster")
        np.testing.assert_array_equal(idx, indices)
        np.testing.assert_array_equal(lab, labels)
        assert frac == pytest.approx(0.4)

    def test_singleton_preserves_length(self):
        indices, labels = self._noisy()
        idx, lab, frac = apply_noise_policy(indices, labels, "singleton")
        np.testing.assert_array_equal(idx, indices)
        assert len(lab) == len(labels)
        assert frac == pytest.approx(0.4)
        # Two noise points become two new, distinct, non-negative clusters.
        assert (lab >= 0).all()
        assert np.unique(lab).size == 4

    def test_singleton_does_not_collide_with_existing_labels(self):
        indices = np.array([0, 1, 2])
        labels = np.array([0, 3, -1])
        _, lab, _ = apply_noise_policy(indices, labels, "singleton")
        assert lab[2] == 4
        assert np.unique(lab).size == 3

    def test_singleton_all_noise_starts_at_zero(self):
        indices = np.array([0, 1])
        labels = np.array([-1, -1])
        _, lab, frac = apply_noise_policy(indices, labels, "singleton")
        np.testing.assert_array_equal(lab, [0, 1])
        assert frac == pytest.approx(1.0)

    def test_no_noise_passes_through(self):
        indices = np.array([0, 1, 2])
        labels = np.array([0, 1, 1])
        for policy in ("drop", "as_cluster", "singleton"):
            idx, lab, frac = apply_noise_policy(indices, labels, policy)
            np.testing.assert_array_equal(idx, indices)
            np.testing.assert_array_equal(lab, labels)
            assert frac == 0.0

    def test_all_noise_dropped_leaves_empty_arrays(self):
        indices = np.array([0, 1, 2])
        labels = np.array([-1, -1, -1])
        idx, lab, frac = apply_noise_policy(indices, labels, "drop")
        assert idx.size == 0
        assert lab.size == 0
        assert frac == pytest.approx(1.0)

    def test_empty_input(self):
        idx, lab, frac = apply_noise_policy(
            np.array([], dtype=int), np.array([], dtype=int), "drop"
        )
        assert idx.size == 0
        assert lab.size == 0
        assert frac == 0.0

    def test_unknown_policy_raises(self):
        indices, labels = self._noisy()
        with pytest.raises(ValueError, match="Unknown noise_policy"):
            apply_noise_policy(indices, labels, "bogus")


# -----------------------------------------------------------------------
# summarize_preprocessing_records
# -----------------------------------------------------------------------


def _pipeline_record(sweep_param, sweep_value):
    """One randomized-preprocessing record with two identity-pipeline runs."""
    blank = dict.fromkeys(
        (
            "labels_train",
            "labels_test",
            "labels_predicted",
            "labels_stability",
            "train_indices",
            "test_indices",
            "stability_indices",
        ),
        None,
    )
    results = [
        ResampleResult(
            ari_stability=s,
            ari_generalizability=g,
            normalization_params={},
            dim_reduction_params={},
            normalization_name="StandardScaler",
            dim_reduction_name="PCA",
            n_clusters_train=3,
            n_clusters_test=3,
            n_clusters_stability=3,
            noise_fraction=0.0,
            **blank,
        )
        for s, g in [(0.8, 0.7), (0.6, 0.5)]
    ]
    return {
        "estimator": "KMeans",
        "params": {sweep_param: sweep_value},
        "results": results,
    }


class TestSummarizePreprocessingRecords:
    def test_groups_on_n_clusters_by_default(self):
        records = [_pipeline_record("n_clusters", 2), _pipeline_record("n_clusters", 3)]
        out = summarize_preprocessing_records(records)
        assert "n_clusters" in out.columns
        assert set(out["n_clusters"]) == {2, 3}
        assert set(out["norm__func"]) == {"StandardScaler"}
        assert set(out["dr__method"]) == {"PCA"}

    def test_averages_within_a_group(self):
        out = summarize_preprocessing_records([_pipeline_record("n_clusters", 2)])
        assert out["ari_stability"].iloc[0] == pytest.approx(0.7)
        assert out["ari_generalizability"].iloc[0] == pytest.approx(0.6)

    def test_groups_on_sweep_param(self):
        records = [
            _pipeline_record("resolution", 0.5),
            _pipeline_record("resolution", 1.0),
        ]
        out = summarize_preprocessing_records(records, sweep_param="resolution")
        assert "resolution" in out.columns
        assert "n_clusters" not in out.columns
        assert set(out["resolution"]) == {0.5, 1.0}


# -----------------------------------------------------------------------
# resolve_anchors
# -----------------------------------------------------------------------


class TestResolveAnchors:
    def test_below_threshold_is_exact(self):
        assert resolve_anchors(
            4000, consensus_anchors=None, anchor_threshold=5000, random_state=0
        ) is None

    def test_threshold_is_inclusive(self):
        # Levine is exactly 5000 cells; its published numbers must not move.
        assert resolve_anchors(
            5000, consensus_anchors=None, anchor_threshold=5000, random_state=0
        ) is None

    def test_above_threshold_uses_threshold_many_anchors(self):
        idx = resolve_anchors(
            6000, consensus_anchors=None, anchor_threshold=5000, random_state=0
        )
        assert idx is not None
        assert idx.shape == (5000,)
        assert idx.dtype == np.int64
        assert np.all(np.diff(idx) > 0)
        assert idx.min() >= 0 and idx.max() < 6000

    def test_no_discontinuity_across_the_threshold(self):
        # 4000 -> 4000 effective anchors (exact); 6000 -> 5000. The count must
        # not fall as n grows, which a flat default would cause.
        below = resolve_anchors(
            4000, consensus_anchors=None, anchor_threshold=5000, random_state=0
        )
        above = resolve_anchors(
            6000, consensus_anchors=None, anchor_threshold=5000, random_state=0
        )
        below_count = 4000 if below is None else below.size
        assert above.size >= below_count

    def test_explicit_int_opts_in_below_threshold(self):
        idx = resolve_anchors(
            5000, consensus_anchors=2000, anchor_threshold=5000, random_state=0
        )
        assert idx is not None and idx.size == 2000

    def test_float_is_a_fraction(self):
        idx = resolve_anchors(
            10_000, consensus_anchors=0.1, anchor_threshold=5000, random_state=0
        )
        assert idx.size == 1000

    def test_fraction_of_one_is_exact(self):
        assert resolve_anchors(
            10_000, consensus_anchors=1.0, anchor_threshold=5000, random_state=0
        ) is None

    def test_deterministic_in_random_state(self):
        a = resolve_anchors(
            9000, consensus_anchors=100, anchor_threshold=5000, random_state=7
        )
        b = resolve_anchors(
            9000, consensus_anchors=100, anchor_threshold=5000, random_state=7
        )
        c = resolve_anchors(
            9000, consensus_anchors=100, anchor_threshold=5000, random_state=8
        )
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    @pytest.mark.parametrize("bad", [0, 1, -5, 0.0, -0.2, 1.5])
    def test_rejects_degenerate_counts(self, bad):
        with pytest.raises(ValueError):
            resolve_anchors(
                9000, consensus_anchors=bad, anchor_threshold=5000, random_state=0
            )
