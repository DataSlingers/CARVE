"""Tests for carve._accuracy module."""

import numpy as np
import pytest

from carve._accuracy import compute_generalizability_scores


class TestComputeGeneralizabilityScores:
    def test_perfect_predictions(self):
        """All predictions match true labels => accuracy 1.0 for all."""
        runs = [
            (np.array([0, 1, 2]), np.array([0, 1, 0]), np.array([0, 1, 0])),
            (np.array([1, 2, 3]), np.array([1, 0, 1]), np.array([1, 0, 1])),
        ]
        scores = compute_generalizability_scores(4, runs)
        assert scores.shape == (4,)
        # All evaluated samples should have accuracy 1.0
        assert scores[1] == 1.0
        assert scores[2] == 1.0

    def test_wrong_predictions(self):
        """Predictions with a genuinely wrong assignment (not just permuted)."""
        # true: [0, 0, 1, 1], pred: [0, 1, 0, 1] — after alignment,
        # some samples are still wrong
        runs = [
            (np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])),
        ]
        scores = compute_generalizability_scores(4, runs)
        # After alignment, 2 of 4 wrong
        assert np.any(scores < 1.0)

    def test_never_evaluated(self):
        """Samples never in any run get score 0."""
        runs = [
            (np.array([0, 1]), np.array([0, 1]), np.array([0, 1])),
        ]
        scores = compute_generalizability_scores(5, runs)
        assert scores[0] == 1.0
        assert scores[1] == 1.0
        assert scores[2] == 0.0
        assert scores[3] == 0.0
        assert scores[4] == 0.0

    def test_partial_accuracy(self):
        """Misclassification not fixed via aligning.

        Alignment maps predicted labels to maximize agreement, so samples
        that are structurally wrong (not just permuted) stay wrong.
        """
        # true: 2 samples in cluster 0, 2 in cluster 1
        # pred: sample 2 (true=1) predicted as 0 — a real error
        true_labels = np.array([0, 0, 1, 1])
        pred_labels = np.array([0, 0, 0, 1])  # sample 2 wrong
        runs = [
            (np.array([0, 1, 2, 3]), true_labels, pred_labels),
        ]
        scores = compute_generalizability_scores(4, runs)
        # Alignment maps pred_0→0, pred_1→1 (majority correct)
        # Sample 2 (true=1, pred=0) is wrong => score 0
        assert scores[2] == pytest.approx(0.0)
        # Others are correct
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(1.0)
        assert scores[3] == pytest.approx(1.0)

    def test_output_shape(self):
        runs = [
            (np.array([0, 5, 9]), np.array([0, 1, 0]), np.array([0, 1, 0])),
        ]
        scores = compute_generalizability_scores(10, runs)
        assert scores.shape == (10,)

    def test_multiple_runs_accumulate(self):
        """Scores should average across runs with structural errors."""
        true_labels = np.array([0, 0, 1, 1])
        pred_correct = np.array([0, 0, 1, 1])
        pred_wrong_2 = np.array([0, 0, 0, 1])  # sample 2 wrong

        runs = [
            (np.array([0, 1, 2, 3]), true_labels, pred_correct),
            (np.array([0, 1, 2, 3]), true_labels, pred_correct),
            (np.array([0, 1, 2, 3]), true_labels, pred_wrong_2),
        ]
        scores = compute_generalizability_scores(4, runs)
        # Sample 2 correct in 2 of 3 runs
        assert scores[2] == pytest.approx(2.0 / 3.0)
        # Others correct in all 3
        assert scores[0] == pytest.approx(1.0)
        assert scores[3] == pytest.approx(1.0)
