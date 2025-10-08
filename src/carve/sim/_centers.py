import numpy as np
import warnings
from typing import Literal

def _sample_centers(
    *, k: int, p: int, center_box: float, rng: np.random.Generator,
    method: Literal["none", "lhs", "best_candidate", "min_dist"] = "best_candidate",
    n_candidates: int, min_center_dist: float | None
) -> np.ndarray:
    center_box = _validate_center_box(center_box)
    if method == "none":
        return rng.uniform(-center_box, center_box, size=(k, p))
    
    elif method == "best_candidate":
        return _best_candidate_centers(k, p, center_box, rng, n_candidates=n_candidates)
    
    elif method == "lhs":
        return _lhs_centers(k, p, center_box, rng)
    
    elif method == "min_dist":
        if min_center_dist is None:
            diag = 2 * center_box * np.sqrt(p)
            min_center_dist = 0.05 * diag
        return _min_dist_centers(k, p, center_box, rng, min_center_dist=min_center_dist)
    
    else:
        raise ValueError(f"unknown centroid_method: {method}")
    
def _validate_center_box(center_box: float) -> float:
    if center_box < 0:
        warnings.warn('`center_box` should be positive; using abs(center_box).')
        center_box = -center_box
    if center_box == 0.0:
        warnings.warn('`center_box` was 0; defaulting to 10.0.')
        center_box = 10.0
    return center_box

def _best_candidate_centers(k: int, p: int, center_box: float, rng: np.random.Generator, n_candidates: int = 64) -> np.ndarray:
    centers = np.empty((k, p), dtype=float)
    centers[0] = rng.uniform(-center_box, center_box, size=p)
    
    for i in range(1, k):
        cands = rng.uniform(-center_box, center_box, size=(n_candidates, p))
        
        # compute squared distances to existing centers
        diffs = cands[:, None, :] - centers[None, :i, :]
        d2 = np.sum(diffs**2, axis=2)
        min_d2 = d2.min(axis=1)
        centers[i] = cands[np.argmax(min_d2)]
        
    return centers

def _lhs_centers(k: int, p: int, center_box: float, rng: np.random.Generator) -> np.ndarray:
    # latin hypercube: each dim gets a random permutation of k bins, with jitter inside each bin
    X = np.empty((k, p), dtype=float)
    
    for j in range(p):
        perm = rng.permutation(k)
        u = (perm + rng.random(k)) / k  # jitter inside each bin -> U(0,1)
        X[:, j] = -center_box + (2 * center_box) * u
        
    return X

def _min_dist_centers(k: int, p: int, center_box: float, rng: np.random.Generator, min_center_dist: float, max_tries: int = 200000) -> np.ndarray:
    centers = []
    tries = 0
    min2 = min_center_dist ** 2
    
    while len(centers) < k and tries < max_tries:
        c = rng.uniform(-center_box, center_box, size=p)
        if not centers:
            centers.append(c)
        else:
            cc = np.vstack(centers)
            if np.all(np.sum((cc - c) ** 2, axis=1) >= min2):
                centers.append(c)
        tries += 1
        
    if len(centers) < k:
        raise ValueError(f"could not place {k} centers with min_center_dist={min_center_dist}. try reducing it or enlarging center_box.")
    
    return np.vstack(centers)