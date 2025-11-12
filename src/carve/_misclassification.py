from typing import List, Tuple
import numpy as np

from ._utils import align_labels

PredLabels = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (idx, true, pred)

def build_generalizability_array(
    n: int,
    runs: List[PredLabels]
) -> np.ndarray:
    correct = np.zeros(n, int)
    total = np.zeros(n, int)
    
    for idx, true, pred in runs:
        pred_aligned = align_labels(true, pred)
        corr = (true == pred_aligned).astype(int)
        np.add.at(correct, idx, corr)
        np.add.at(total, idx, 1)
        
    out = np.zeros(n, float)
    valid = total > 0
    out[valid] = (correct[valid] / total[valid])
    return out

def compute_global_misclassification_arrays(
    generalizability_arrs: List[np.ndarray]
) -> List[float]:
    avg_misclassification = [np.mean(arr) for arr in generalizability_arrs]
    return avg_misclassification