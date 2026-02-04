"""Shared utility helpers for CARVE."""

from typing import Any, Dict, List, Tuple, Type
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import ClusterMixin
from sklearn.metrics.cluster import contingency_matrix
from numpy.typing import ArrayLike


def split_subsample_indices(
    n_samples: int, 
    *,
    subsample_ratio: float = 0.8, 
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
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


def _summarise_ari_scores(x: List[float] | np.ndarray, n_resamples: int) -> Tuple[float, float, float, float]:
    """
    Summarizes a collection of ARI (Adjusted Rand Index) scores by computing mean, standard error, and quantiles.
    
    Parameters
    ----------
    x : list of float or np.ndarray
        A list or array of ARI scores, possibly containing NaN values.
    n_resamples : int
        The number of resamples used to generate the ARI scores. (Unused in computation, but kept for interface consistency.)
        
    Returns
    -------
    mean : float
        The mean of the ARI scores, ignoring NaN values.
    se : float
        The standard error of the ARI scores, ignoring NaN values. Returns NaN if fewer than 2 valid entries.
    q95 : float
        The 95th percentile (quantile) of the ARI scores, ignoring NaN values.
    q05 : float
        The 5th percentile (quantile) of the ARI scores, ignoring NaN values.
    """
    
    arr = np.asarray(x, dtype=float)
    if np.all(np.isnan(arr)):
        return (np.nan, np.nan, np.nan, np.nan)
    mean = float(np.nanmean(arr))
    
    # SE over non-NaN entries
    m = np.sum(~np.isnan(arr))
    se = float(np.nanstd(arr, ddof=1) / np.sqrt(m)) if m > 1 else np.nan
    q95 = float(np.nanquantile(arr, 0.95))
    q05 = float(np.nanquantile(arr, 0.05))
    return (mean, se, q95, q05)


def cluster_labels(
    X: np.ndarray,
    estimator_cls: Type[ClusterMixin],
    *,
    random_state: int = None,
    **params: Any
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
        if 'random_state' in params:
            estimator = estimator_cls(**params)
        elif 'random_state' in estimator_cls.__init__.__code__.co_varnames:
            estimator = estimator_cls(random_state=random_state, **params)
        else:
            estimator = estimator_cls(**params)


    # estimator implements fit_predict
    if hasattr(estimator, "fit_predict"):
        return estimator.fit_predict(X)

    # otherwise: fit first
    estimator.fit(X)

    # standard sklearn behavior: labels_ attribute
    try:
        return estimator.labels_
    except Exception:
        # fallback: predict(X) if available
        return estimator.predict(X)  


def summarize_preprocessing_records(
    pipeline_records: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Summarize randomized preprocessing records.

    Parameters
    ----------
    pipeline_records : list of dict
        Records from randomized preprocessing runs.

    Returns
    -------
    summary : pandas.DataFrame
        Mean ARI metrics grouped by normalization, DR, and k.
    """
    rows = []
    for record in pipeline_records:
        params = record['params']
        n_clusters = params['n_clusters']
        
        for r in record['results']:
            ari_s, ari_g, *_, norm_p, dr_p, norm_name, dr_name = r
            
            if norm_name != 'FunctionTransformer':
                norm_label = norm_name
            else:
                func = norm_p.get('func', None)
                norm_label = func.__name__ if func is not None else 'identity'
            
            if dr_name != 'FunctionTransformer':
                dr_label = dr_name
            else:
                func = dr_p.get('func', None)
                dr_label = func.__name__ if func is not None else 'identity'
                
            rows.append({
                'n_clusters': n_clusters, 
                'norm__func': norm_label,
                'dr__method': dr_label, 
                'ari_stability': ari_s, 
                'ari_generalizability': ari_g 
            })
        
    dfp = pd.DataFrame(rows)
    
    return (
        dfp
        .groupby(['norm__func', 'dr__method', 'n_clusters'], as_index=False)
        .mean()
    )
    
    
def align_cluster_labels(
    reference_labels: np.ndarray, 
    labels: np.ndarray
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
    # get contingency matrix
    cont = contingency_matrix(reference_labels, labels)
    
    # solve assignment on -cont to max matches
    row_ind, col_ind = linear_sum_assignment(-cont)
    
    # align order
    true_classes = np.unique(reference_labels)
    pred_classes = np.unique(labels)
    
    # build mapping
    mapping = {
        pred_classes[col]: true_classes[row]
        for row, col in zip(row_ind, col_ind)
    }
    
    for pc in pred_classes:
        mapping.setdefault(pc, pc)

    # apply mapping
    aligned = np.array([mapping[lbl] for lbl in labels], dtype=reference_labels.dtype)
    return aligned


def ensure_2d_array(X: ArrayLike) -> np.ndarray:
    """Ensure input is a 2D NumPy array.

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    array : ndarray of shape (n_samples, n_features)
        2D NumPy array representation.

    Raises
    ------
    ValueError
        If the input cannot be converted to a 2D array.
    """
    if isinstance(X, pd.DataFrame):     # Convert Pandas DataFrame to NumPy array
        return X.values
    elif isinstance(X, np.ndarray):     # Ensure the array is 2D
        if X.ndim == 1:
            return X.reshape(-1, 1)     
        elif X.ndim == 2:
            return X
        else:
            raise ValueError("Input NumPy array must be 1D or 2D.")
    elif isinstance(X, list):           # Convert list to NumPy array and ensure it's 2D
        return np.atleast_2d(np.array(X))
    else:
        raise ValueError("Input must be a NumPy array, Pandas DataFrame, or a list.")