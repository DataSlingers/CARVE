"""Shared utility helpers for CARVE.

Provides subsample splitting, cluster label alignment via the Hungarian
algorithm, ARI score summarization, and array coercion utilities.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.base import ClusterMixin
from sklearn.metrics.cluster import contingency_matrix

from ._types import NoisePolicy


def split_subsample_indices(
    n_samples: int,
    *,
    subsample_ratio: float = 0.8,
    random_state: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into a random subsample and its complement.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    subsample_ratio : float, default=0.8
        Proportion of samples in the subsample.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    train_idx : ndarray
        Indices for the subsample.
    test_idx : ndarray
        Indices for the remaining samples.
    """
    rng = np.random.RandomState(random_state)
    all_idx = np.arange(n_samples)
    train_size = int(np.float64(subsample_ratio * n_samples))

    train_idx = rng.choice(all_idx, size=train_size, replace=False)
    test_idx = np.setdiff1d(all_idx, train_idx)

    return train_idx, test_idx


def count_clusters(labels: np.ndarray | None) -> int:
    """Count distinct non-noise clusters in a label vector.

    Density-based methods such as HDBSCAN mark unassigned points with a
    negative label; those are not counted as a cluster.

    Parameters
    ----------
    labels : ndarray or None
        Cluster labels.

    Returns
    -------
    n_clusters : int
        Number of distinct non-negative labels, or 0 for empty input.
    """
    if labels is None:
        return 0

    labels = np.asarray(labels)
    if labels.size == 0:
        return 0

    return int(np.unique(labels[labels >= 0]).size)


def apply_noise_policy(
    indices: np.ndarray,
    labels: np.ndarray,
    policy: NoisePolicy = "drop",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resolve negative (noise) labels according to policy.

    Density-based methods such as HDBSCAN mark unassigned points with
    ``-1``. Left unhandled, those samples are treated as a cluster by
    the consensus matrix and by ARI, which inflates stability.

    Parameters
    ----------
    indices : ndarray of shape (n_subsample,)
        Row indices of these points in the original dataset.
    labels : ndarray of shape (n_subsample,)
        Cluster labels, possibly containing negative noise markers.
    policy : {"drop", "as_cluster", "singleton"}, default="drop"
        ``"drop"`` removes noise points from this iteration entirely, so
        they count as un-sampled and contribute nothing to the consensus
        matrix, the ARI, or the classifier. ``"as_cluster"`` keeps ``-1``
        as an ordinary label. ``"singleton"`` gives each noise point its
        own cluster.

    Returns
    -------
    indices : ndarray
        Possibly filtered indices, aligned with labels.
    labels : ndarray
        Labels with noise resolved.
    noise_fraction : float
        Fraction of the input points that carried a negative label.

    Raises
    ------
    ValueError
        If policy is not a recognized noise policy.
    """
    if policy not in ("drop", "as_cluster", "singleton"):
        raise ValueError(
            f"Unknown noise_policy {policy!r}. Expected 'drop', 'as_cluster', "
            "or 'singleton'."
        )

    labels = np.asarray(labels)
    indices = np.asarray(indices)

    if labels.size == 0:
        return indices, labels, 0.0

    noise_mask = labels < 0
    noise_fraction = float(noise_mask.mean())

    if not noise_mask.any() or policy == "as_cluster":
        return indices, labels, noise_fraction

    if policy == "drop":
        keep = ~noise_mask
        return indices[keep], labels[keep], noise_fraction

    # policy == "singleton"
    out = labels.copy()
    assigned = labels[~noise_mask]
    start = int(assigned.max()) + 1 if assigned.size else 0
    out[noise_mask] = np.arange(start, start + int(noise_mask.sum()))
    return indices, out, noise_fraction


def _coerce_n_clusters(value: int | np.ndarray) -> np.ndarray:
    """Coerce a cluster-count specification to a 1D integer NumPy array.

    Parameters
    ----------
    value : int or ndarray
        Cluster-count input. If an integer K is provided, this function
        returns the range 2..K (inclusive). If an array-like is provided,
        it is validated and converted to an integer ndarray.

    Returns
    -------
    n_clusters : ndarray of shape (n_values,)
        One-dimensional integer array of cluster counts, where all values
        are greater than or equal to 2.

    Raises
    ------
    ValueError
        If an integer input is smaller than 2, if the provided array is not
        one-dimensional, or if any cluster count is smaller than 2.
    TypeError
        If the provided array does not contain integer values.
    """
    if isinstance(value, (int, np.integer)):
        k = int(value)

        if k < 2:
            raise ValueError("n_clusters int must be >= 2.")

        return np.arange(2, k + 1, dtype=int)

    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError("n_clusters must be a 1D array.")

    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("n_clusters array must contain integers.")

    if np.any(arr < 2):
        raise ValueError("All n_clusters values must be >= 2.")

    return arr.astype(int, copy=False)


def _summarize_ari_scores(
    x: list[float] | np.ndarray,
    n_resamples: int,
) -> tuple[float, float, float, float]:
    """Summarize ARI scores by computing mean, standard error, and quantiles.

    Parameters
    ----------
    x : list of float or ndarray
        ARI scores, possibly containing NaN values.
    n_resamples : int
        Number of resamples (unused in computation; kept for interface
        consistency).

    Returns
    -------
    mean : float
        Mean of the ARI scores, ignoring NaN values.
    se : float
        Standard error, ignoring NaN values. Returns NaN if fewer than 2
        valid entries.
    q95 : float
        95th percentile of the ARI scores, ignoring NaN values.
    q05 : float
        5th percentile of the ARI scores, ignoring NaN values.
    """
    arr = np.asarray(x, dtype=float)
    if np.all(np.isnan(arr)):
        return (np.nan, np.nan, np.nan, np.nan)
    mean = float(np.nanmean(arr))

    m = np.sum(~np.isnan(arr))
    se = float(np.nanstd(arr, ddof=1) / np.sqrt(m)) if m > 1 else np.nan
    q95 = float(np.nanquantile(arr, 0.95))
    q05 = float(np.nanquantile(arr, 0.05))
    return (mean, se, q95, q05)


def cluster_labels(
    X: np.ndarray,
    estimator_cls: type[ClusterMixin],
    *,
    random_state: int = None,
    **params: Any,
) -> np.ndarray:
    """Fit a clustering estimator and return labels.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    estimator_cls : type
        Estimator class implementing clustering.
    random_state : int or None, default=None
        Random seed passed to the estimator if supported.
    **params : dict
        Estimator parameters.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels.

    Raises
    ------
    TypeError
        If the estimator provides no valid way to extract labels.
    """
    try:
        estimator = estimator_cls(random_state=random_state, **params)
    except Exception:
        if "random_state" in params:
            estimator = estimator_cls(**params)
        elif "random_state" in estimator_cls.__init__.__code__.co_varnames:
            estimator = estimator_cls(random_state=random_state, **params)
        else:
            estimator = estimator_cls(**params)

    if hasattr(estimator, "fit_predict"):
        return estimator.fit_predict(X)

    estimator.fit(X)

    try:
        return estimator.labels_
    except Exception:
        return estimator.predict(X)


def summarize_preprocessing_records(
    pipeline_records: list[dict[str, Any]],
    sweep_param: str = "n_clusters",
) -> pd.DataFrame:
    """Summarize randomized preprocessing records.

    Parameters
    ----------
    pipeline_records : list of dict
        Records from randomized preprocessing runs.
    sweep_param : str, default="n_clusters"
        Name of the swept hyperparameter used as the grouping key.

    Returns
    -------
    summary : pandas.DataFrame
        Mean ARI metrics grouped by normalization, DR, and the sweep value.
    """
    rows = []
    for record in pipeline_records:
        params = record["params"]
        sweep_value = params[sweep_param]

        for r in record["results"]:
            ari_s = r.ari_stability
            ari_g = r.ari_generalizability
            norm_p = r.normalization_params
            dr_p = r.dim_reduction_params
            norm_name = r.normalization_name
            dr_name = r.dim_reduction_name

            if norm_name != "FunctionTransformer":
                norm_label = norm_name
            else:
                func = norm_p.get("func", None)
                norm_label = func.__name__ if func is not None else "identity"

            if dr_name != "FunctionTransformer":
                dr_label = dr_name
            else:
                func = dr_p.get("func", None)
                dr_label = func.__name__ if func is not None else "identity"

            rows.append(
                {
                    sweep_param: sweep_value,
                    "norm__func": norm_label,
                    "dr__method": dr_label,
                    "ari_stability": ari_s,
                    "ari_generalizability": ari_g,
                }
            )

    dfp = pd.DataFrame(rows)

    return dfp.groupby(["norm__func", "dr__method", sweep_param], as_index=False).mean()


def align_cluster_labels(
    reference_labels: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Align labels to reference labels using Hungarian assignment.

    Parameters
    ----------
    reference_labels : ndarray of shape (n_samples,)
        Reference clustering labels.
    labels : ndarray of shape (n_samples,)
        Labels to align.

    Returns
    -------
    aligned : ndarray of shape (n_samples,)
        Aligned labels with best matching permutation.
    """
    cont = contingency_matrix(reference_labels, labels)
    row_ind, col_ind = linear_sum_assignment(-cont)

    true_classes = np.unique(reference_labels)
    pred_classes = np.unique(labels)

    mapping = {
        pred_classes[col]: true_classes[row] for row, col in zip(row_ind, col_ind)
    }
    for pc in pred_classes:
        mapping.setdefault(pc, pc)

    aligned = np.array([mapping[lbl] for lbl in labels], dtype=reference_labels.dtype)
    return aligned


def ensure_2d_array(
    X: ArrayLike, *, dense_warn_elements: int = 50_000_000
) -> np.ndarray:
    """Ensure input is a 2D NumPy array.

    Sparse input is densified rather than passed through. Several estimators
    on the default grid (agglomerative clustering, t-SNE, and the RBF affinity
    of :class:`~carve.SpectralClustering`) require dense input, so keeping the
    matrix sparse would only move the failure into a parallel worker.

    Parameters
    ----------
    X : array-like
        Input data. Accepts NumPy arrays, pandas DataFrames, lists, and
        SciPy sparse matrices and arrays.
    dense_warn_elements : int, default=50_000_000
        Warn when densifying a sparse matrix with more entries than this
        (50M float64 entries is roughly 400 MB).

    Returns
    -------
    array : ndarray of shape (n_samples, n_features)
        2D NumPy array representation.

    Raises
    ------
    ValueError
        If the input cannot be converted to a 2D array.
    """
    if isinstance(X, pd.DataFrame):
        return X.values
    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            return X.reshape(-1, 1)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError("Input NumPy array must be 1D or 2D.")
    elif sparse.issparse(X):
        if X.ndim != 2:
            raise ValueError("Input sparse matrix must be 2D.")
        n_elements = int(X.shape[0]) * int(X.shape[1])
        if n_elements > dense_warn_elements:
            warnings.warn(
                f"Densifying a sparse matrix with {n_elements:,} entries "
                f"(~{n_elements * 8 / 1e9:.1f} GB dense). Pass a reduced "
                "representation instead, e.g. use_rep='X_pca'.",
                UserWarning,
                stacklevel=2,
            )
        return np.asarray(X.todense())
    elif isinstance(X, list):
        return np.atleast_2d(np.array(X))
    else:
        raise ValueError(
            "Input must be a NumPy array, SciPy sparse matrix, Pandas "
            "DataFrame, or a list."
        )
