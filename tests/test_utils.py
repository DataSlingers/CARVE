"""Tests for carve._utils module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer

import carve._utils as carve_utils
from carve._pipeline import PipelineSpec, PipelineStep
from carve._runner import ResampleResult
from carve._utils import (
    _coerce_n_clusters,
    _summarize_ari_scores,
    align_cluster_labels,
    apply_noise_policy,
    cluster_labels,
    count_clusters,
    default_generalizability_classifier,
    ensure_2d_array,
    resolve_anchors,
    resolve_core_budget,
    split_subsample_indices,
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
        assert mean == pytest.approx(np.mean(scores))
        assert se == pytest.approx(np.std(scores, ddof=1) / np.sqrt(5))
        assert q95 == pytest.approx(np.quantile(scores, 0.95))
        assert q05 == pytest.approx(np.quantile(scores, 0.05))

    def test_all_nan(self):
        mean, se, q95, q05 = _summarize_ari_scores([np.nan, np.nan], 2)
        assert np.isnan(mean)
        assert np.isnan(se)

    def test_single_value(self):
        mean, se, q95, q05 = _summarize_ari_scores([0.5], 1)
        assert mean == 0.5
        assert np.isnan(se)  # can't compute SE with 1 value
        assert q95 == 0.5
        assert q05 == 0.5

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
        with pytest.raises(ValueError, match="Input must be a NumPy array"):
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


def _spec(dr_step):
    return PipelineSpec(PipelineStep(FunctionTransformer, {}, None), dr_step)


PCA2 = _spec(PipelineStep(PCA, {"n_components": 2}, None))
PCA3 = _spec(PipelineStep(PCA, {"n_components": 3}, None))


def _result(stability, generalizability, spec, k=3):
    """One ResampleResult carrying only what the summary reads."""
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
    return ResampleResult(
        ari_stability=stability,
        ari_generalizability=generalizability,
        pipeline=spec,
        n_clusters_train=k,
        n_clusters_test=k,
        n_clusters_stability=k,
        noise_fraction=0.0,
        **blank,
    )


def _record(method_id, sweep_value, results, sweep_rank=0):
    """One configuration's record, shaped as run_validation returns it."""
    return {
        "method_id": method_id,
        "method_label": f"KMeans {method_id}",
        "sweep_value": sweep_value,
        "sweep_rank": sweep_rank,
        "results": results,
    }


class TestSummarizePreprocessingRecords:
    def test_columns(self):
        out, _ = summarize_preprocessing_records(
            [_record("m0", 2, [_result(0.8, 0.7, PCA2)])]
        )
        assert list(out.columns) == [
            "method_id",
            "method_label",
            "pipeline",
            "normalization",
            "dim_reduction",
            "n_clusters",
            "n_resamples",
            "ari_stability",
            "ari_stability_se",
            "ari_generalizability",
            "ari_generalizability_se",
            "n_clusters_observed",
            "sweep_param",
            "sweep_value",
            "sweep_rank",
        ]

    def test_hyperparameters_split_rows(self):
        out, pipelines = summarize_preprocessing_records(
            [_record("m0", 2, [_result(0.8, 0.7, PCA2), _result(0.4, 0.3, PCA3)])]
        )
        assert list(out["pipeline"]) == [
            "identity | PCA(n_components=2)",
            "identity | PCA(n_components=3)",
        ]
        assert list(out["normalization"]) == ["identity", "identity"]
        assert list(out["dim_reduction"]) == [
            "PCA(n_components=2)",
            "PCA(n_components=3)",
        ]
        assert pipelines == {PCA2.label: PCA2, PCA3.label: PCA3}

    def test_estimators_split_rows_and_sort_numerically(self):
        records = [
            _record("m10", 2, [_result(0.5, 0.5, PCA2)]),
            _record("m2", 2, [_result(0.9, 0.9, PCA2)]),
        ]
        out, _ = summarize_preprocessing_records(records)
        assert list(out["method_id"]) == ["m2", "m10"]
        assert list(out["ari_stability"]) == [0.9, 0.5]

    def test_counts_sum_to_the_run_within_each_configuration(self):
        records = [
            _record(
                "m0",
                2,
                [_result(0.8, 0.7, PCA2), _result(0.6, 0.5, PCA2), _result(0.4, 0.3, PCA3)],
            ),
            _record(
                "m0",
                3,
                [_result(0.8, 0.7, PCA3), _result(0.6, 0.5, PCA2), _result(0.4, 0.3, PCA3)],
                sweep_rank=1,
            ),
        ]
        out, _ = summarize_preprocessing_records(records)
        assert out.groupby("n_clusters")["n_resamples"].sum().to_dict() == {2: 3, 3: 3}
        # Sorted by pipeline, then sweep value.
        assert list(out["n_resamples"]) == [2, 1, 1, 2]

    def test_standard_error_uses_only_the_rows_resamples(self):
        out, _ = summarize_preprocessing_records(
            [
                _record(
                    "m0",
                    2,
                    [
                        _result(0.8, 0.7, PCA2),
                        _result(0.6, 0.5, PCA2),
                        _result(0.1, 0.0, PCA3),
                        _result(0.3, 0.2, PCA3),
                    ],
                )
            ]
        )
        pca2, pca3 = out.iloc[0], out.iloc[1]
        assert pca2["ari_stability"] == pytest.approx(0.7)
        assert pca2["ari_stability_se"] == pytest.approx(0.1)
        assert pca2["ari_generalizability_se"] == pytest.approx(0.1)
        assert pca3["ari_stability"] == pytest.approx(0.2)
        assert pca3["ari_stability_se"] == pytest.approx(0.1)

    def test_single_resample_has_no_standard_error(self):
        out, _ = summarize_preprocessing_records(
            [_record("m0", 2, [_result(0.8, 0.7, PCA2)])]
        )
        assert np.isnan(out.loc[0, "ari_stability_se"])

    def test_skipped_criterion_is_nan(self):
        out, _ = summarize_preprocessing_records(
            [_record("m0", 2, [_result(0.8, np.nan, PCA2), _result(0.6, np.nan, PCA2)])]
        )
        assert out.loc[0, "ari_stability"] == pytest.approx(0.7)
        assert np.isnan(out.loc[0, "ari_generalizability"])
        assert np.isnan(out.loc[0, "ari_generalizability_se"])

    def test_observed_cluster_count_is_the_rows_mean(self):
        out, _ = summarize_preprocessing_records(
            [
                _record(
                    "m0",
                    2,
                    [
                        _result(0.8, 0.7, PCA2, k=2),
                        _result(0.6, 0.5, PCA2, k=3),
                        _result(0.4, 0.3, PCA3, k=5),
                    ],
                )
            ]
        )
        assert list(out["n_clusters_observed"]) == [2.5, 5.0]

    def test_sweep_column_follows_the_sweep_param(self):
        out, _ = summarize_preprocessing_records(
            [_record("m0", 0.5, [_result(0.8, 0.7, PCA2)])], sweep_param="resolution"
        )
        assert "resolution" in out.columns
        assert "n_clusters" not in out.columns
        assert out.loc[0, "resolution"] == 0.5
        assert out.loc[0, "sweep_param"] == "resolution"

    def test_empty(self):
        out, pipelines = summarize_preprocessing_records([])
        assert out.empty
        assert "pipeline" in out.columns
        assert pipelines == {}


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

    @pytest.mark.parametrize(
        ("bad", "message"),
        [
            (0, "must be at least 2"),
            (1, "must be at least 2"),
            (-5, "must be at least 2"),
            (0.0, r"must be in \(0, 1\]"),
            (-0.2, r"must be in \(0, 1\]"),
            (1.5, r"must be in \(0, 1\]"),
        ],
    )
    def test_rejects_degenerate_counts(self, bad, message):
        with pytest.raises(ValueError, match=message):
            resolve_anchors(
                9000, consensus_anchors=bad, anchor_threshold=5000, random_state=0
            )

    def test_tiny_n_is_exact_rather_than_an_error(self):
        # The default resolves to m = n_samples here, so the run is exact. It
        # used to raise about consensus_anchors, a parameter the caller in
        # this case never set.
        assert resolve_anchors(
            1, consensus_anchors=None, anchor_threshold=5000, random_state=0
        ) is None

    def test_degenerate_default_message_names_anchor_threshold(self):
        # A threshold below 2 is the caller's own doing, but the old message
        # blamed consensus_anchors for it.
        with pytest.raises(ValueError, match=r"anchor_threshold=1"):
            resolve_anchors(
                9000, consensus_anchors=None, anchor_threshold=1, random_state=0
            )

    def test_degenerate_explicit_message_names_consensus_anchors(self):
        with pytest.raises(ValueError, match=r"consensus_anchors=1"):
            resolve_anchors(
                9000, consensus_anchors=1, anchor_threshold=5000, random_state=0
            )


# -----------------------------------------------------------------------
# default_generalizability_classifier
# -----------------------------------------------------------------------


class TestDefaultGeneralizabilityClassifier:
    """Pin the construction both call sites share.

    _runner's generalizability path and CARVE.get_labels' anchored label
    extension must build the same classifier. They diverged once, on
    max_depth, while a comment was the only thing asking them not to. These
    assertions replace the comment.
    """

    def test_default_is_a_random_forest_with_the_documented_parameters(self):
        clf = default_generalizability_classifier(
            classifier=None, n_features=16, n_trees=37, random_state=5, n_jobs=3
        )
        assert isinstance(clf, RandomForestClassifier)
        assert clf.n_estimators == 37
        assert clf.max_depth == 16
        assert clf.max_features == 4  # int(sqrt(16))
        assert clf.random_state == 5
        assert clf.n_jobs == 3

    def test_n_jobs_is_required(self):
        # The forest used to hardcode n_jobs=-1, which oversubscribed under
        # any outer parallelism. Both call sites must now pass the thread
        # count they were budgeted, so there is no default to fall back on.
        with pytest.raises(TypeError, match="n_jobs"):
            default_generalizability_classifier(
                classifier=None, n_features=16, n_trees=37, random_state=5
            )

    def test_default_forest_n_jobs_is_not_hardcoded(self):
        for n_jobs in (1, 2, 5):
            clf = default_generalizability_classifier(
                classifier=None, n_features=4, n_trees=5, random_state=0, n_jobs=n_jobs
            )
            assert clf.n_jobs == n_jobs

    def test_max_features_is_the_integer_square_root(self):
        clf = default_generalizability_classifier(
            classifier=None, n_features=10, n_trees=5, random_state=0, n_jobs=1
        )
        assert clf.max_depth == 10
        assert clf.max_features == 3  # int(sqrt(10)) == 3, not 3.16

    def test_none_random_state_is_passed_through_unchanged(self):
        # The "0 if None" fallback belongs to the callers, not here.
        clf = default_generalizability_classifier(
            classifier=None, n_features=4, n_trees=5, random_state=None, n_jobs=1
        )
        assert clf.random_state is None

    def test_supplied_classifier_is_cloned_and_seeded(self):
        supplied = KNeighborsClassifier(n_neighbors=3)
        clf = default_generalizability_classifier(
            classifier=supplied, n_features=4, n_trees=100, random_state=9, n_jobs=1
        )
        assert clf is not supplied
        assert isinstance(clf, KNeighborsClassifier)
        assert clf.n_neighbors == 3
        # KNeighborsClassifier has no random_state, so none is injected.
        assert "random_state" not in clf.get_params()

    def test_supplied_classifier_with_random_state_receives_the_seed(self):
        supplied = DummyClassifier(strategy="stratified", random_state=None)
        clf = default_generalizability_classifier(
            classifier=supplied, n_features=4, n_trees=100, random_state=13, n_jobs=1
        )
        assert clf is not supplied
        assert clf.random_state == 13
        assert supplied.random_state is None  # the original is untouched

    def test_supplied_classifier_ignores_n_trees(self):
        supplied = RandomForestClassifier(n_estimators=7)
        clf = default_generalizability_classifier(
            classifier=supplied, n_features=4, n_trees=500, random_state=0, n_jobs=1
        )
        assert clf.n_estimators == 7

    def test_supplied_classifier_with_n_jobs_receives_the_budget(self):
        # A user's forest built with n_jobs=-1 would oversubscribe exactly as
        # the old default did, so the budget overrides what it was built with.
        supplied = RandomForestClassifier(n_estimators=7, n_jobs=-1)
        clf = default_generalizability_classifier(
            classifier=supplied, n_features=4, n_trees=500, random_state=0, n_jobs=2
        )
        assert clf.n_jobs == 2
        assert supplied.n_jobs == -1  # the original is untouched

    def test_supplied_classifier_without_n_jobs_is_left_alone(self):
        supplied = DummyClassifier(strategy="most_frequent")
        clf = default_generalizability_classifier(
            classifier=supplied, n_features=4, n_trees=100, random_state=0, n_jobs=2
        )
        assert "n_jobs" not in clf.get_params()

    def test_matches_the_runner_construction_it_replaced(self):
        # The literal parameters the generalizability path built before the
        # extraction, spelled out so a change here has to be deliberate.
        n_features, n_trees, seed = 12, 100, 3
        clf = default_generalizability_classifier(
            classifier=None,
            n_features=n_features,
            n_trees=n_trees,
            random_state=seed,
            n_jobs=4,
        )
        expected = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=n_features,
            max_features=int(np.sqrt(n_features)),
            random_state=seed,
            n_jobs=4,
        )
        assert clf.get_params() == expected.get_params()


# -----------------------------------------------------------------------
# resolve_core_budget
# -----------------------------------------------------------------------


class TestResolveCoreBudget:
    """Pin the split of n_jobs into resample workers and threads per worker.

    Measured before this rule existed: inside loky workers the default
    forest's hardcoded n_jobs=-1 still started one thread per core whatever
    the outer n_jobs, so CARVE(n_jobs=-1) on 11 cores ran 11 x 14 threads.
    The budget makes outer x inner the machine, never more.
    """

    @pytest.fixture(autouse=True)
    def eleven_cores(self, monkeypatch):
        monkeypatch.setattr(carve_utils, "cpu_count", lambda: 11)

    def test_one_worker_gives_the_classifier_every_core(self):
        # n_jobs=1 is today's default configuration: one worker, forest on
        # every core. The budget must not slow that down.
        assert resolve_core_budget(1, n_resamples=100) == (1, 11)

    def test_all_cores_gives_one_thread_per_worker(self):
        assert resolve_core_budget(-1, n_resamples=100) == (11, 1)

    def test_partial_budget_splits_the_remainder(self):
        assert resolve_core_budget(4, n_resamples=100) == (4, 2)

    def test_negative_counts_follow_joblib(self):
        # joblib: -2 means all cores but one.
        assert resolve_core_budget(-2, n_resamples=100) == (10, 1)

    def test_none_means_one_worker(self):
        assert resolve_core_budget(None, n_resamples=100) == (1, 11)

    def test_more_workers_than_cores_gives_one_thread_each(self):
        assert resolve_core_budget(20, n_resamples=100) == (20, 1)

    def test_workers_are_capped_by_resamples(self):
        # Parallel never runs more workers than there are resamples, so the
        # threads the missing workers would have held go to the ones that run.
        assert resolve_core_budget(-1, n_resamples=4) == (4, 2)
        assert resolve_core_budget(8, n_resamples=3) == (3, 3)

    def test_zero_is_rejected(self):
        with pytest.raises(ValueError, match="n_jobs"):
            resolve_core_budget(0, n_resamples=100)

    def test_product_never_exceeds_the_machine_when_workers_fit(self):
        for n_jobs in (1, 2, 3, 5, 7, 11, -1, -3):
            outer, inner = resolve_core_budget(n_jobs, n_resamples=100)
            assert outer * inner <= 11
            assert inner >= 1
