from typing import List, Tuple, Union
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

IndexLabels = Tuple[np.ndarray, np.ndarray]     # (P_1_idx, labels_1)
# IndexLabelsIntersect = Tuple[
#     np.ndarray, np.ndarray,                     # P_1_idx, P_2_idx
#     np.ndarray, np.ndarray                      # labels_1, labels_2
# ]

def build_consensus_matrix(
    n: int,
    runs: List[IndexLabels], 
    *,
    return_counts: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
    M[I_sum == 0] = np.nan                  # set never-sampled pairs to np.nan

    if return_counts:
        return M, M_sum, I_sum
    else:
        return M
    
def order_consensus_matrix(
    raw_cons_mat: np.ndarray, 
    *,
    fill_nan_for_order: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    M_for_order = np.nan_to_num(raw_cons_mat, nan=fill_nan_for_order)
    dists = squareform(1.0 - M_for_order, checks=False)
    Z = linkage(dists, method='average')
    order = leaves_list(Z)
    
    return raw_cons_mat[np.ix_(order, order)], order
    
def compute_consensus_metrics_batch(
    cons_mats_raw: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    gini_list = []
    ce_list = []
    pac_list = []
    
    for M in cons_mats_raw:
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
    tau: float = 0.05, 
) -> float:
    P = np.array(M, dtype=float, copy=True)
    n = P.shape[0]
    mask = ~np.eye(n, dtype=bool)
    V = P[mask]
    V = V[~np.isnan(V)]
    
    if V.size == 0:
        return np.nan
    
    amb = ((V > tau) & (V < (1 - tau))).sum()
    pac = amb / V.size
    return 1.0 - pac

# def build_consensus_and_flip(
#     n: int,
#     runs: List[IndexLabelsIntersect], 
#     *,
#     return_counts: bool = False
# ):
#     M_sum = np.zeros((n,n), float)  # AND
#     F_sum = np.zeros((n,n), float)  # XOR
#     U_sum = np.zeros((n,n), float)  # OR = AND + XOR
#     I_sum = np.zeros((n,n), float)  # co-observed (overlap)

#     for idx1, idx2, l1, l2 in runs:
#         O, i1, i2 = np.intersect1d(idx1, idx2, return_indices=True)
#         if O.size == 0:
#             continue
#         a1 = l1[i1]
#         a2 = l2[i2]

#         # build per-cluster blocks once per partition
#         # A^(1): union over clusters of rows x rows; A^(2): same for cols x cols (both on O)
#         A1 = np.zeros((O.size, O.size), dtype=bool)
#         A2 = np.zeros((O.size, O.size), dtype=bool)
#         for labs, A in ((a1, A1), (a2, A2)):
#             u, inv = np.unique(labs, return_inverse=True)
#             for k in range(u.size):
#                 m = (inv == k)
#                 if m.sum() > 1:
#                     A[np.ix_(m, m)] = True
#                 else:
#                     A[m, m] = True  # keep diagonals consistent

#         # update counts on the overlap block
#         S_block = A1 & A2               # AND
#         U_block = A1 | A2               # OR
#         F_block = U_block ^ S_block     # XOR

#         I_sum[np.ix_(O, O)] += 1
#         M_sum[np.ix_(O, O)] += S_block
#         U_sum[np.ix_(O, O)] += U_block
#         F_sum[np.ix_(O, O)] += F_block

#     with np.errstate(divide='ignore', invalid='ignore'):
#         M = M_sum / I_sum
#         U = U_sum / I_sum
#         F = F_sum / I_sum
    
#     for A in (M, U, F):
#         A[I_sum == 0] = np.nan

#     if return_counts:
#         return (M, U, F, M_sum, U_sum, F_sum, I_sum)
#     return (M, U, F)

# def build_contingency_entropy(
#     runs: List[IndexLabelsIntersect],
#     *,
#     return_counts: bool = False
# ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
#     R = len(runs)
#     H_weighted = np.full(R, np.nan, dtype=float)
#     H_unweighted = np.full(R, np.nan, dtype=float)
#     overlap_sizes = np.zeros(R, dtype=int)

#     for r, (idx1, idx2, l1, l2) in enumerate(runs):
#         # overlap and alignment of labels on the overlap
#         O, i1, i2 = np.intersect1d(idx1, idx2, return_indices=True)
#         m = O.size
#         overlap_sizes[r] = m
#         if m <= 1:
#             # empty or singleton overlap: no uncertainty measurable
#             H_weighted[r] = np.nan
#             H_unweighted[r] = np.nan
#             continue

#         a = l1[i1].astype(int)
#         b = l2[i2].astype(int)

#         # relabel to compact 0..k-1 indices per side (only labels present on O)
#         au, a_inv = np.unique(a, return_inverse=True)   # k1 present rows
#         bu, b_inv = np.unique(b, return_inverse=True)   # k2 present cols
#         k1, k2 = au.size, bu.size

#         # build contingency N (k1 x k2)
#         N = np.zeros((k1, k2), dtype=float)
#         np.add.at(N, (a_inv, b_inv), 1.0)

#         # row sums and row-normalized probabilities
#         row_sum = N.sum(axis=1, keepdims=True)  # shape (k1,1)
#         # rows with at least one item:
#         valid_rows = (row_sum[:, 0] > 0)

#         P = np.divide(N, row_sum, out=np.zeros_like(N), where=row_sum > 0)

#         # row entropies: -∑ p log p (0*log0 := 0)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             logP = np.where(P > 0, np.log(P), 0.0)
#         H_rows = -np.sum(P * logP, axis=1)  # shape (k1,)

#         # normalize to [0,1] by log(k2) (max entropy occurs at uniform over k2)
#         if k2 >= 2:
#             norm = np.log(k2)
#             H_rows_norm = H_rows / norm
#         else:
#             # if C2 has only one label on O, the mapping is degenerate -> entropy 0
#             H_rows_norm = np.zeros_like(H_rows)

#         # size-weighted average across rows (weights = row sizes)
#         weights = row_sum[:, 0]
#         if valid_rows.any():
#             w = weights[valid_rows]
#             h = H_rows_norm[valid_rows]
#             H_weighted[r] = (h @ w) / w.sum()
#             H_unweighted[r] = h.mean()
#         else:
#             H_weighted[r] = np.nan
#             H_unweighted[r] = np.nan

#     if return_counts:
#         return np.mean(H_weighted), H_unweighted, overlap_sizes
#     return np.mean(H_weighted)
    
# def compute_flip_metric_batch(
#     n: int,
#     instability_fields: List[np.ndarray]
# ) -> Tuple[List[float]]:
#     flip_stability_list = []
    
#     for F in instability_fields:
#         flip_rate = np.nanmean(F[~np.eye(n, dtype=bool)])
#         flip_stability_list.append(flip_rate)
        
#     return flip_stability_list