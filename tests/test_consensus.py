"""Tests for carve._consensus module."""

import warnings

import numpy as np
import pytest

from carve._consensus import (
    _default_anchor_chunk_size,
    _row_nanmean,
    compute_consensus_matrix,
    compute_consensus_metrics,
    compute_consensus_pac,
    consensus_anchor_block,
    reorder_consensus_matrix,
    stability_from_consensus,
    stability_from_runs_anchored,
)

# -----------------------------------------------------------------------
# compute_consensus_matrix
# -----------------------------------------------------------------------


class TestComputeConsensusMatrix:
    def test_basic(self):
        runs = [
            (np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1])),
            (np.array([1, 2, 3, 4]), np.array([0, 1, 1, 1])),
        ]
        M = compute_consensus_matrix(5, runs)
        assert M.shape == (5, 5)
        # Sample 0 and sample 4 never co-sampled
        assert np.isnan(M[0, 4])

    def test_symmetric(self):
        runs = [
            (np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1])),
            (np.array([0, 1, 2, 3]), np.array([0, 1, 1, 0])),
        ]
        M = compute_consensus_matrix(4, runs)
        np.testing.assert_array_equal(M, M.T)

    def test_return_counts(self):
        runs = [
            (np.array([0, 1, 2]), np.array([0, 0, 1])),
            (np.array([0, 1, 2]), np.array([0, 0, 1])),
        ]
        M, co_cluster, co_sample = compute_consensus_matrix(
            3,
            runs,
            return_counts=True,
        )
        assert co_sample[0, 1] == 2
        assert co_cluster[0, 1] == 2
        assert M[0, 1] == 1.0

    def test_single_run(self):
        runs = [(np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1]))]
        M = compute_consensus_matrix(4, runs)
        assert M[0, 1] == 1.0
        assert M[2, 3] == 1.0
        assert M[0, 2] == 0.0

    def test_never_co_sampled(self):
        runs = [
            (np.array([0, 1]), np.array([0, 0])),
            (np.array([2, 3]), np.array([1, 1])),
        ]
        M = compute_consensus_matrix(4, runs)
        assert np.isnan(M[0, 2])
        assert np.isnan(M[1, 3])

    def test_perfect_consensus(self):
        runs = [(np.arange(10), np.array([0] * 5 + [1] * 5))] * 5
        M = compute_consensus_matrix(10, runs)
        for i in range(5):
            for j in range(5):
                assert M[i, j] == 1.0
        for i in range(5):
            for j in range(5, 10):
                assert M[i, j] == 0.0

    def test_values_in_range(self):
        rng = np.random.RandomState(0)
        runs = []
        for _ in range(10):
            idx = rng.choice(20, 15, replace=False)
            labels = rng.randint(0, 3, 15)
            runs.append((idx, labels))
        M = compute_consensus_matrix(20, runs)
        valid = ~np.isnan(M)
        assert np.all(M[valid] >= 0.0)
        assert np.all(M[valid] <= 1.0)

    def test_diagonal_is_one(self):
        runs = [
            (np.array([0, 1, 2]), np.array([0, 1, 0])),
            (np.array([0, 1, 2]), np.array([1, 0, 1])),
        ]
        M = compute_consensus_matrix(3, runs)
        for i in range(3):
            assert M[i, i] == 1.0


# -----------------------------------------------------------------------
# reorder_consensus_matrix
# -----------------------------------------------------------------------


class TestReorderConsensusMatrix:
    def _interleaved(self):
        # Two perfect blocks, {0, 2, 4} and {1, 3, 5}, interleaved by index.
        M = np.zeros((6, 6))
        for group in ([0, 2, 4], [1, 3, 5]):
            for i in group:
                for j in group:
                    M[i, j] = 1.0
        return M

    def test_groups_the_interleaved_blocks(self):
        reordered, order = reorder_consensus_matrix(self._interleaved())
        assert sorted(order) == list(range(6))
        # Each block ends up contiguous: the leaf order lists one group's
        # three members, then the other's.
        first, second = frozenset(order[:3]), frozenset(order[3:])
        assert {first, second} == {frozenset({0, 2, 4}), frozenset({1, 3, 5})}
        # And the reordered matrix is block diagonal.
        np.testing.assert_array_equal(reordered[:3, :3], 1.0)
        np.testing.assert_array_equal(reordered[3:, 3:], 1.0)
        np.testing.assert_array_equal(reordered[:3, 3:], 0.0)

    def test_reordering_is_a_permutation_of_the_input(self):
        M = self._interleaved()
        M[0, 2] = M[2, 0] = 0.7
        reordered, order = reorder_consensus_matrix(M)
        np.testing.assert_array_equal(reordered, M[np.ix_(order, order)])
        assert sorted(reordered.ravel()) == sorted(M.ravel())

    def test_nan_handling(self):
        M = np.array([[1.0, np.nan], [np.nan, 1.0]])
        reordered, order = reorder_consensus_matrix(M, fill_nan_for_order=0.0)
        assert reordered.shape == (2, 2)
        assert sorted(order) == [0, 1]
        # The NaN is passed through untouched; fill_nan_for_order only
        # affects the ordering, not the returned values.
        assert np.isnan(reordered[0, 1])


# -----------------------------------------------------------------------
# _row_nanmean
# -----------------------------------------------------------------------


class TestRowNanmean:
    def test_matches_nanmean_wherever_nanmean_is_defined(self):
        rng = np.random.RandomState(0)
        a = rng.rand(50, 40)
        a[rng.rand(50, 40) < 0.3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmean(a, axis=1)
        np.testing.assert_array_equal(_row_nanmean(a), expected)

    def test_all_nan_row_is_nan_and_does_not_warn(self):
        a = np.array([[0.2, np.nan, 0.6], [np.nan, np.nan, np.nan]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = _row_nanmean(a)
        assert out[0] == pytest.approx(0.4)
        assert np.isnan(out[1])


# -----------------------------------------------------------------------
# stability_from_consensus
# -----------------------------------------------------------------------


class TestStabilityFromConsensus:
    def test_perfect_consensus(self):
        M = np.zeros((10, 10))
        M[:5, :5] = 1.0
        M[5:, 5:] = 1.0
        gini, ce = stability_from_consensus(M)
        assert gini.shape == (10,)
        assert ce.shape == (10,)
        assert np.all(gini > 0.9)
        assert np.all(ce > 0.9)

    def test_random_consensus(self):
        rng = np.random.RandomState(0)
        M = rng.rand(20, 20)
        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 1.0)
        gini, ce = stability_from_consensus(M)
        assert np.mean(gini) < 0.9
        assert np.mean(ce) < 0.9

    def test_output_range(self):
        rng = np.random.RandomState(1)
        M = rng.rand(10, 10)
        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 1.0)
        gini, ce = stability_from_consensus(M)
        assert np.all(gini >= 0.0) and np.all(gini <= 1.0)
        assert np.all(ce >= 0.0) and np.all(ce <= 1.0)

    def test_nan_entries(self):
        M = np.array(
            [
                [1.0, 0.8, np.nan],
                [0.8, 1.0, 0.2],
                [np.nan, 0.2, 1.0],
            ]
        )
        gini, ce = stability_from_consensus(M)
        assert gini.shape == (3,)
        assert not np.any(np.isnan(gini))

    def test_isolated_sample_scores_nan_without_warning(self):
        # Sample 2 was never co-sampled with anyone: its off-diagonal row is
        # all NaN, so it has no partners to average over. NaN is the right
        # score; the old nanmean call warned "Mean of empty slice" here.
        M = np.array(
            [
                [1.0, 0.8, np.nan],
                [0.8, 1.0, np.nan],
                [np.nan, np.nan, 1.0],
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gini, ce = stability_from_consensus(M)
        assert np.isfinite(gini[:2]).all()
        assert np.isfinite(ce[:2]).all()
        assert np.isnan(gini[2])
        assert np.isnan(ce[2])


# -----------------------------------------------------------------------
# compute_consensus_pac
# -----------------------------------------------------------------------


class TestComputeConsensusPac:
    def test_perfect_matrix(self):
        M = np.zeros((6, 6))
        M[:3, :3] = 1.0
        M[3:, 3:] = 1.0
        score = compute_consensus_pac(M)
        assert score == 1.0

    def test_ambiguous_matrix(self):
        M = np.full((4, 4), 0.5)
        np.fill_diagonal(M, 1.0)
        score = compute_consensus_pac(M)
        assert score == pytest.approx(0.0)

    def test_custom_tau(self):
        M = np.array(
            [
                [1.0, 0.03, 0.97],
                [0.03, 1.0, 0.03],
                [0.97, 0.03, 1.0],
            ]
        )
        score = compute_consensus_pac(M, tau=0.05)
        assert score == 1.0

    def test_nan_only_matrix(self):
        M = np.full((3, 3), np.nan)
        np.fill_diagonal(M, 1.0)
        score = compute_consensus_pac(M)
        assert np.isnan(score)

    def test_output_range(self):
        rng = np.random.RandomState(0)
        M = rng.rand(10, 10)
        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 1.0)
        score = compute_consensus_pac(M)
        assert 0.0 <= score <= 1.0


# -----------------------------------------------------------------------
# compute_consensus_metrics
# -----------------------------------------------------------------------


class TestComputeConsensusMetrics:
    def test_basic(self):
        M1 = np.zeros((6, 6))
        M1[:3, :3] = 1.0
        M1[3:, 3:] = 1.0

        M2 = np.full((6, 6), 0.5)
        np.fill_diagonal(M2, 1.0)

        gini_list, ce_list, pac_list = compute_consensus_metrics([M1, M2])
        assert len(gini_list) == 2
        assert len(ce_list) == 2
        assert len(pac_list) == 2
        # Perfect matrix should have higher stability
        assert np.mean(gini_list[0]) > np.mean(gini_list[1])
        assert pac_list[0] > pac_list[1]

    def test_single_matrix(self):
        M = np.eye(4)
        gini_list, ce_list, pac_list = compute_consensus_metrics([M])
        assert len(gini_list) == 1
        assert len(ce_list) == 1
        assert len(pac_list) == 1

    def test_output_shapes(self):
        rng = np.random.RandomState(0)
        matrices = [rng.rand(8, 8) for _ in range(3)]
        gini_list, ce_list, pac_list = compute_consensus_metrics(matrices)
        for g in gini_list:
            assert g.shape == (8,)
        for c in ce_list:
            assert c.shape == (8,)
        assert len(pac_list) == 3


# -----------------------------------------------------------------------
# consensus_anchor_block
# -----------------------------------------------------------------------


def _make_runs(n=200, n_runs=12, k=4, seed=0):
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_runs):
        idx = np.sort(rng.choice(n, size=int(0.618 * n), replace=False))
        runs.append((idx, rng.integers(0, k, idx.size)))
    return runs


class TestConsensusAnchorBlock:
    def test_block_equals_the_exact_submatrix(self):
        n = 200
        runs = _make_runs(n)
        anchors = np.sort(np.random.default_rng(1).choice(n, 40, replace=False))

        exact = compute_consensus_matrix(n, runs)[np.ix_(anchors, anchors)]
        block = consensus_anchor_block(n, runs, anchors)

        # Exact equality, not a tolerance: both sides are correctly rounded
        # float32 divisions of exactly representable integer counts, so the
        # promise the block makes is identity, not closeness. equal_nan
        # because both carry NaN for never-co-sampled pairs.
        assert np.array_equal(block, exact, equal_nan=True)

    def test_shape_and_dtype(self):
        n = 200
        runs = _make_runs(n)
        anchors = np.arange(0, n, 5)
        block = consensus_anchor_block(n, runs, anchors)
        assert block.shape == (anchors.size, anchors.size)
        assert block.dtype == np.float32

    def test_all_anchors_reproduces_the_full_matrix(self):
        n = 120
        runs = _make_runs(n, n_runs=8, k=3, seed=2)
        anchors = np.arange(n)
        assert np.allclose(
            consensus_anchor_block(n, runs, anchors),
            compute_consensus_matrix(n, runs),
            equal_nan=True,
        )

    def test_values_are_not_all_identical(self):
        # A block of constant values would satisfy a shape-only assertion,
        # so pin that the fixture actually varies.
        n = 200
        runs = _make_runs(n)
        anchors = np.arange(0, n, 5)
        block = consensus_anchor_block(n, runs, anchors)
        off = block[~np.eye(anchors.size, dtype=bool)]
        assert np.nanstd(off) > 0.01

    def test_never_cosampled_pairs_are_nan(self):
        n = 50
        # Two disjoint halves are never co-sampled, so every cross pair is NaN.
        left = np.arange(0, 25)
        right = np.arange(25, 50)
        runs = [
            (left, np.zeros(left.size, dtype=int)),
            (right, np.zeros(right.size, dtype=int)),
        ]
        anchors = np.array([0, 1, 30, 31])
        block = consensus_anchor_block(n, runs, anchors)
        assert np.isnan(block[0, 2])
        assert not np.isnan(block[0, 1])


# -----------------------------------------------------------------------
# stability_from_runs_anchored
# -----------------------------------------------------------------------


class TestStabilityFromRunsAnchored:
    def test_all_anchors_matches_the_exact_scores(self):
        n = 150
        runs = _make_runs(n, n_runs=10, k=3, seed=5)
        exact_gini, exact_ce = stability_from_consensus(
            compute_consensus_matrix(n, runs)
        )
        gini, ce = stability_from_runs_anchored(n, runs, np.arange(n))

        assert np.allclose(gini, exact_gini, atol=1e-6)
        assert np.allclose(ce, exact_ce, atol=1e-6)

    def test_scores_cover_every_sample(self):
        n = 300
        runs = _make_runs(n, seed=6)
        anchors = np.sort(np.random.default_rng(3).choice(n, 50, replace=False))
        gini, ce = stability_from_runs_anchored(n, runs, anchors)

        assert gini.shape == (n,)
        assert ce.shape == (n,)
        assert np.isfinite(gini).all()
        assert np.isfinite(ce).all()

    def test_chunking_does_not_change_the_result(self):
        n = 300
        runs = _make_runs(n, seed=7)
        anchors = np.sort(np.random.default_rng(4).choice(n, 60, replace=False))
        a = stability_from_runs_anchored(n, runs, anchors, chunk_size=17)
        b = stability_from_runs_anchored(n, runs, anchors, chunk_size=100_000)
        assert np.allclose(a[0], b[0])
        assert np.allclose(a[1], b[1])

    def test_default_chunk_size_matches_an_explicit_one(self):
        # The default row count is derived from the anchor count so the
        # float64 temporaries stay bounded. Deriving it must not move a value.
        n = 300
        runs = _make_runs(n, seed=7)
        anchors = np.sort(np.random.default_rng(4).choice(n, 60, replace=False))
        default = stability_from_runs_anchored(n, runs, anchors)
        explicit = stability_from_runs_anchored(n, runs, anchors, chunk_size=8192)
        assert np.allclose(default[0], explicit[0])
        assert np.allclose(default[1], explicit[1])

    def test_default_chunk_size_binds_at_a_large_anchor_count(self):
        # At m = 60 (the test above) the 8192 ceiling wins and the
        # derivation never engages the 2**23-element budget that is the
        # actual memory fix -- that test would pass unchanged even if the
        # budget were relaxed to something far larger. Pin the derivation
        # directly at an m large enough for the budget to bind instead.
        #
        # 2**23 // 5000 = 1677, which is below the 8192 ceiling, so this is
        # the concrete value the documented rule produces at m = 5000; there
        # is no way to state the expectation here without restating that
        # same arithmetic, so the value is pinned outright.
        assert _default_anchor_chunk_size(5000) == 1677

        # And, regardless of the exact constant, the per-chunk (chunk_size,
        # m) float64 transient this rule bounds stays within the intended
        # element budget rather than growing with m unchecked.
        assert _default_anchor_chunk_size(5000) * 5000 <= 2**23

    def test_scores_vary_across_samples(self):
        # Guards against an implementation that returns a constant vector,
        # which every shape assertion above would still accept.
        n = 300
        runs = _make_runs(n, seed=8)
        anchors = np.arange(0, n, 4)
        gini, ce = stability_from_runs_anchored(n, runs, anchors)
        assert gini.std() > 1e-3
        assert ce.std() > 1e-3

    def test_anchor_subset_approximates_the_exact_scores(self):
        # The estimator is a row mean over a random column subset, so a
        # large anchor set should track the exact answer closely.
        n = 400
        runs = _make_runs(n, n_runs=20, k=4, seed=9)
        exact_gini, _ = stability_from_consensus(compute_consensus_matrix(n, runs))
        gini, _ = stability_from_runs_anchored(
            n, runs, np.sort(np.random.default_rng(5).choice(n, 300, replace=False))
        )
        assert np.corrcoef(gini, exact_gini)[0, 1] > 0.95

    def test_unsorted_run_indices_match_sorted_indices(self):
        # Regression test: the runner's per-resample sample_idx arrays are
        # unsorted permutations (rng.choice without sorting), not the sorted
        # arrays _make_runs happens to produce. Shuffle each run's indices
        # (permuting its labels the same way, so the pairing is preserved)
        # and require a multi-chunk pass to match the sorted-input result.
        n = 200
        sorted_runs = _make_runs(n, n_runs=10, k=3, seed=11)

        rng = np.random.default_rng(12)
        shuffled_runs = []
        for sample_idx, labels in sorted_runs:
            perm = rng.permutation(sample_idx.size)
            shuffled_runs.append((sample_idx[perm], labels[perm]))

        anchors = np.sort(np.random.default_rng(13).choice(n, 40, replace=False))

        sorted_gini, sorted_ce = stability_from_runs_anchored(
            n, sorted_runs, anchors, chunk_size=23
        )
        shuffled_gini, shuffled_ce = stability_from_runs_anchored(
            n, shuffled_runs, anchors, chunk_size=23
        )

        assert np.allclose(shuffled_gini, sorted_gini, equal_nan=True)
        assert np.allclose(shuffled_ce, sorted_ce, equal_nan=True)

    def test_sample_never_cosampled_with_an_anchor_is_nan_without_warning(self):
        # Every run draws from the first 29 samples, so sample 29 shares no
        # run with any anchor and its n-by-m row is entirely NaN.
        n = 30
        rng = np.random.default_rng(0)
        runs = []
        for _ in range(6):
            idx = np.sort(rng.choice(n - 1, size=18, replace=False))
            runs.append((idx, rng.integers(0, 3, idx.size)))
        anchors = np.arange(10)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gini, ce = stability_from_runs_anchored(n, runs, anchors)
        assert np.isnan(gini[n - 1])
        assert np.isnan(ce[n - 1])
        assert np.isfinite(gini[:10]).all()
