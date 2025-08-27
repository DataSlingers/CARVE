from typing import List, Tuple
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

IndexLabels = Tuple[np.ndarray, np.ndarray]  # (indices, labels)

def build_consensus_matrix(
    n: int,
    runs: List[IndexLabels], 
    *,
    return_counts: bool = False,
    fill_nan_for_order: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    M_sum = np.zeros((n, n), dtype=float)
    I_sum = np.zeros((n, n), dtype=float)
    
    for idx, labels in runs:
        I_sum[np.ix_(idx, idx)] += 1        # count which pairs were ever co-sampled
        
        for lab in np.unique(labels):
            pts = idx[labels == lab]
            M_sum[np.ix_(pts, pts)] += 1    # increment same-cluster counts
        
    # build consensus fraction
    with np.errstate(divide='ignore', invalid='ignore'):
        M = M_sum / I_sum
    M[I_sum == 0] = np.nan  # set never-sampled pairs to np.nan
    
    # reorder by average-linkage dendrogram
    M_for_order = np.nan_to_num(M, nan=fill_nan_for_order)
    dists = squareform(1.0 - M_for_order, checks=False)
    Z = linkage(dists, method='average')
    order = leaves_list(Z)

    if return_counts:
        return M[np.ix_(order, order)], M, M_sum, I_sum, order
    else:
        return M[np.ix_(order, order)], M

def compute_consensus_metrics_batch(M_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    gini_list = []
    ce_list = []
    pac_list = []
    
    for M in M_list:
        s_gini, s_ce = stab_from_consensus(M)
        pac = consensus_pac(M)
        
        gini_list.append(s_gini)
        ce_list.append(s_ce)
        pac_list.append(pac)
        
    return gini_list, ce_list, pac_list

def stab_from_consensus(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = np.array(M, dtype=float, copy=True)
    np.fill_diagonal(P, np.nan)
    
    term = P * (1.0 - P)
    
    S = np.clip(P, 1e-12, 1.0 - 1e-12)
    H = -(S * np.log(S) + (1.0 - S) * np.log(1.0 - S))

    u_gini = 2.0 * np.nanmean(term, axis=1)     # [0, 0.5]           
    u_ce = np.nanmean(H, axis=1)                # [0, log2]

    s_gini = 1.0 - np.clip(2.0 * u_gini, 0.0, 1.0)
    s_ce = 1.0 - np.clip(u_ce / np.log(2.0), 0.0, 1.0)
    
    return s_gini, s_ce
    
def consensus_pac(
    M: np.ndarray, 
    *, 
    lower: float = 0.1, 
    upper: float = 0.9
) -> float:
    P = np.array(M, dtype=float, copy=True)
    n = P.shape[0]
    mask = ~np.eye(n, dtype=bool)
    V = P[mask]
    V = V[~np.isnan(V)]
    
    if V.size == 0:
        return np.nan
    
    amb = ((V > lower) & (V < upper)).sum()
    return amb / V.size