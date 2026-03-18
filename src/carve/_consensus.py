"""Consensus matrix utilities and stability metrics.

This module builds consensus matrices from repeated clustering runs and
derives stability statistics such as Gini, cross-entropy, and PAC.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# (sample_indices, labels) pair from a single clustering run
SampledLabels = tuple[np.ndarray, np.ndarray]


def compute_consensus_matrix(
    n_samples: int,
    runs: list[SampledLabels],
    *,
    return_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a consensus matrix from resampled clustering runs.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the original dataset.
    runs : list of tuple of (ndarray, ndarray)
        Each run is (sample_indices, labels), where ``sample_indices`` are
        integer indices into the original dataset and ``labels`` are cluster
        assignments for those indices.
    return_counts : bool, default=False
        If True, also return raw co-cluster and co-sample counts.

    Returns
    -------
    consensus_matrix : ndarray of shape (n_samples, n_samples)
        Fraction of times each sample pair co-clustered when co-sampled.
        Pairs never co-sampled are set to NaN.
    co_cluster_counts : ndarray of shape (n_samples, n_samples)
        Raw counts of co-clustering across runs (only if ``return_counts``).
    co_sample_counts : ndarray of shape (n_samples, n_samples)
        Raw counts of co-sampling across runs (only if ``return_counts``).
    """
    co_cluster_counts = np.zeros((n_samples, n_samples), dtype=float)
    co_sample_counts = np.zeros((n_samples, n_samples), dtype=float)

    for sample_idx, labels in runs:
        co_sample_counts[np.ix_(sample_idx, sample_idx)] += 1

        for label in np.unique(labels):
            label_idx = sample_idx[labels == label]
            co_cluster_counts[np.ix_(label_idx, label_idx)] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus_matrix = co_cluster_counts / co_sample_counts
    consensus_matrix[co_sample_counts == 0] = np.nan

    if return_counts:
        return consensus_matrix, co_cluster_counts, co_sample_counts
    else:
        return consensus_matrix


def reorder_consensus_matrix(
    consensus_matrix: np.ndarray,
    *,
    fill_nan_for_order: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder a consensus matrix by hierarchical clustering of distances.

    Parameters
    ----------
    consensus_matrix : ndarray of shape (n_samples, n_samples)
        Consensus matrix with values in [0, 1] and NaNs for never-sampled pairs.
    fill_nan_for_order : float, default=0.0
        Value used to replace NaNs when computing the clustering order only.

    Returns
    -------
    reordered : ndarray of shape (n_samples, n_samples)
        Consensus matrix reordered by dendrogram leaves.
    order : ndarray of shape (n_samples,)
        The permutation of indices applied to rows and columns.
    """
    matrix_for_order = np.nan_to_num(consensus_matrix, nan=fill_nan_for_order)
    distances = squareform(1.0 - matrix_for_order, checks=False)
    linkage_matrix = linkage(distances, method="average")
    order = leaves_list(linkage_matrix)

    return consensus_matrix[np.ix_(order, order)], order


def compute_consensus_metrics(
    consensus_matrices: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Compute stability metrics for multiple consensus matrices.

    Parameters
    ----------
    consensus_matrices : list of ndarray
        Consensus matrices to evaluate.

    Returns
    -------
    gini_list : list of ndarray
        Per-sample stability scores based on the Gini-style statistic.
    ce_list : list of ndarray
        Per-sample stability scores based on cross-entropy.
    pac_list : list of float
        Global PAC-based stability scores (1 - PAC).
    """
    gini_list = []
    ce_list = []
    pac_list = []

    for consensus_matrix in consensus_matrices:
        s_gini, s_ce = stability_from_consensus(consensus_matrix)
        pac = compute_consensus_pac(consensus_matrix)

        gini_list.append(s_gini)
        ce_list.append(s_ce)
        pac_list.append(pac)

    return gini_list, ce_list, pac_list


def stability_from_consensus(
    consensus_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample stability scores from a consensus matrix.

    Parameters
    ----------
    consensus_matrix : ndarray of shape (n_samples, n_samples)
        Consensus matrix with values in [0, 1] and NaNs for never-sampled pairs.

    Returns
    -------
    stability_gini : ndarray of shape (n_samples,)
        Gini-based stability score per sample in [0, 1].
    stability_ce : ndarray of shape (n_samples,)
        Cross-entropy-based stability score per sample in [0, 1].
    """
    consensus_probs = np.array(consensus_matrix, dtype=float, copy=True)
    np.fill_diagonal(consensus_probs, np.nan)

    term = consensus_probs * (1.0 - consensus_probs)

    clipped = np.clip(consensus_probs, 1e-12, 1.0 - 1e-12)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))

    # Gini uncertainty in [0, 0.5]; CE uncertainty in [0, log2]
    uncertainty_gini = 2.0 * np.nanmean(term, axis=1)
    uncertainty_ce = np.nanmean(entropy, axis=1)

    # Rescale to [0, 1] stability scores
    stability_gini = 1.0 - np.clip(2.0 * uncertainty_gini, 0.0, 1.0)
    stability_ce = 1.0 - np.clip(uncertainty_ce / np.log(2.0), 0.0, 1.0)

    return stability_gini, stability_ce


def compute_consensus_pac(
    consensus_matrix: np.ndarray,
    *,
    tau: float = 0.05,
) -> float:
    """Compute PAC-based stability score from a consensus matrix.

    PAC measures the proportion of ambiguous consensus values in
    (tau, 1-tau). This function returns 1 - PAC so that larger values
    indicate greater stability.

    Parameters
    ----------
    consensus_matrix : ndarray of shape (n_samples, n_samples)
        Consensus matrix with values in [0, 1] and NaNs for never-sampled pairs.
    tau : float, default=0.05
        Ambiguity threshold defining the PAC interval.

    Returns
    -------
    score : float
        Stability score in [0, 1], or NaN if no valid pairs exist.
    """
    consensus_probs = np.array(consensus_matrix, dtype=float, copy=True)
    n_samples = consensus_probs.shape[0]
    mask = ~np.eye(n_samples, dtype=bool)
    values = consensus_probs[mask]
    values = values[~np.isnan(values)]

    if values.size == 0:
        return np.nan

    ambiguous = ((values > tau) & (values < (1 - tau))).sum()
    pac = ambiguous / values.size
    return 1.0 - pac
