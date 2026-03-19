"""Tests for carve._consensus module."""

import numpy as np
import pytest

from carve._consensus import (
    compute_consensus_matrix,
    compute_consensus_metrics,
    compute_consensus_pac,
    reorder_consensus_matrix,
    stability_from_consensus,
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
            3, runs, return_counts=True,
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
        runs = [(np.arange(10), np.array([0]*5 + [1]*5))] * 5
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
    def test_basic(self):
        M = np.zeros((6, 6))
        for i in [0, 2, 4]:
            for j in [0, 2, 4]:
                M[i, j] = 1.0
        for i in [1, 3, 5]:
            for j in [1, 3, 5]:
                M[i, j] = 1.0
        reordered, order = reorder_consensus_matrix(M)
        assert reordered.shape == M.shape
        assert set(order) == set(range(6))

    def test_preserves_values(self):
        M = np.array([[1.0, 0.5], [0.5, 1.0]])
        reordered, order = reorder_consensus_matrix(M)
        assert reordered[0, 0] == 1.0
        assert reordered[1, 1] == 1.0

    def test_nan_handling(self):
        M = np.array([[1.0, np.nan], [np.nan, 1.0]])
        reordered, order = reorder_consensus_matrix(M, fill_nan_for_order=0.0)
        assert reordered.shape == (2, 2)


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
        M = np.array([
            [1.0, 0.8, np.nan],
            [0.8, 1.0, 0.2],
            [np.nan, 0.2, 1.0],
        ])
        gini, ce = stability_from_consensus(M)
        assert gini.shape == (3,)
        assert not np.any(np.isnan(gini))


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
        M = np.array([
            [1.0, 0.03, 0.97],
            [0.03, 1.0, 0.03],
            [0.97, 0.03, 1.0],
        ])
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
