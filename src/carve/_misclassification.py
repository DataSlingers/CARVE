from typing import List, Tuple
import numpy as np

PredLabels = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (idx, true, pred)

def build_generalizability_array(
    n: int,
    runs: List[PredLabels]
) -> np.ndarray:
    correct = np.zeros(n, int)
    total = np.zeros(n, int)
    
    for idx, true, pred in runs:
        corr = (true == pred)
        np.add.at(correct, idx, corr.astype(int))
        np.add.at(total, idx, 1)
        
    generalizability_array = np.zeros(n, float)
    valid = total > 0
    generalizability_array[valid] = (correct[valid] / total[valid])
    return generalizability_array