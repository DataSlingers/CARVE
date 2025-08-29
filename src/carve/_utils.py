from typing import Any, Dict, List, Tuple, Type
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import ClusterMixin
from sklearn.metrics.cluster import contingency_matrix
from numpy.typing import ArrayLike

def subsample_indices(
    n_samples: int, 
    *,
    ratio: float = 0.6, 
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    all_idx = np.arange(n_samples)
    train_size = int(np.float64(ratio * n_samples))
    
    train_idx = rng.choice(all_idx, size=train_size, replace=False)
    test_idx = np.setdiff1d(all_idx, train_idx)
    
    return train_idx, test_idx

def clustering_pipeline(
    X: np.ndarray,
    est_cls: Type[ClusterMixin],
    *,
    random_state: int = None,
    **params: Any
) -> np.ndarray:
    try:
        estimator = est_cls(random_state=random_state, **params)
    
    except Exception:
        estimator = est_cls(**params)

    if hasattr(estimator, "fit_predict"):
        return estimator.fit_predict(X)
    
    estimator.fit(X)
    return getattr(estimator, "labels_")

def wrangle_pipeline_records(
    pipeline_records: List[Dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    for record in pipeline_records:
        params = record['params']
        k = params['n_clusters']
        
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
                'n_clusters': k, 
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
    
def align_labels(
    ref_labels: np.ndarray, 
    labels: np.ndarray
) -> np.ndarray:
    # get contingency matrix
    cont = contingency_matrix(ref_labels, labels)
    
    # solve assignment on -cont to max matches
    row_ind, col_ind = linear_sum_assignment(-cont)
    
    # align order
    true_classes = np.unique(ref_labels)
    pred_classes = np.unique(labels)
    
    # build mapping
    mapping = {
        pred_classes[col]: true_classes[row]
        for row, col in zip(row_ind, col_ind)
    }
    
    for pc in pred_classes:
        mapping.setdefault(pc, pc)

    # apply mapping
    aligned = np.array([mapping[lbl] for lbl in labels], dtype=ref_labels.dtype)
    return aligned

def ensure_array2d(X: ArrayLike) -> np.ndarray:
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