"""Misclassification-based generalizability metrics."""

from typing import List, Tuple
import numpy as np

from ._utils import align_cluster_labels

PredictionRun = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (sample_indices, true_labels, predicted_labels)

def compute_generalizability_scores(
    n_samples: int,
    runs: List[PredictionRun]
) -> np.ndarray:
    """Compute per-sample generalizability from prediction runs.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the original dataset.
    runs : list of tuple
        Each run is (sample_indices, true_labels, predicted_labels).

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        Per-sample accuracy across available runs. Samples never evaluated
        receive score 0.
    """
    correct = np.zeros(n_samples, int)
    total = np.zeros(n_samples, int)
    
    for sample_idx, true_labels, predicted_labels in runs:
        aligned_predictions = align_cluster_labels(true_labels, predicted_labels)
        correct_mask = (true_labels == aligned_predictions).astype(int)
        np.add.at(correct, sample_idx, correct_mask)
        np.add.at(total, sample_idx, 1)
        
    out = np.zeros(n_samples, float)
    valid = total > 0
    out[valid] = (correct[valid] / total[valid])
    return out

def compute_mean_generalizability(
    generalizability_scores: List[np.ndarray]
) -> List[float]:
    """Compute mean generalizability for each score vector.

    Parameters
    ----------
    generalizability_scores : list of ndarray
        Per-sample generalizability arrays.

    Returns
    -------
    means : list of float
        Mean generalizability per input array.
    """
    mean_generalizability = [np.mean(scores) for scores in generalizability_scores]
    return mean_generalizability