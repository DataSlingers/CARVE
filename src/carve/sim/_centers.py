"""Cluster center sampling utilities for simulations."""

import numpy as np
from typing import Literal

def _sample_centers(
    *, k: int, p: int, center_box: float, rng: np.random.Generator,
    method: Literal["none", "lhs", "best_candidate", "min_dist"] = "best_candidate",
    n_candidates: int
) -> np.ndarray:
    """Sample cluster centers within a hypercube.

    Parameters
    ----------
    k : int
        Number of centers to sample.
    p : int
        Dimension of each center (feature count).
    center_box : float
        Half-width of the sampling hypercube; centers lie in
        [-center_box, +center_box]^p.
    rng : numpy.random.Generator
        Random generator for reproducibility.
    method : {"none", "lhs", "best_candidate", "min_dist"}, default="best_candidate"
        Center placement strategy.
    n_candidates : int
        Number of candidates evaluated per center for "best_candidate".

    Returns
    -------
    centers : ndarray of shape (k, p)
        Center coordinates.
    """
    center_box = _validate_center_box(center_box)
    if method == "none":
        return rng.uniform(-center_box, center_box, size=(k, p))

    if method == "best_candidate":
        return _best_candidate_centers(k, p, center_box, rng, n_candidates=n_candidates)

    if method == "lhs":
        return _lhs_centers(k, p, center_box, rng)

    if method == "min_dist":
        diag = 2 * center_box * np.sqrt(p)
        min_center_dist = 0.05 * diag
        return _min_dist_centers(k, p, center_box, rng, min_center_dist=min_center_dist)

    raise ValueError(f"unknown centroid_method: {method}")

def _validate_center_box(center_box: float) -> float:
    """Validate and normalize the center_box parameter.

    Parameters
    ----------
    center_box : float
            Half-width of the sampling hypercube.

    Returns
    -------
    center_box : float
            Positive, finite center_box.
    """
    if not np.isfinite(center_box):
        raise ValueError("`center_box` must be finite.")
    if center_box <= 0:
        raise ValueError("`center_box` must be positive.")
    return float(center_box)

def _best_candidate_centers(
    k: int,
    p: int,
    center_box: float,
    rng: np.random.Generator,
    n_candidates: int = 64
) -> np.ndarray:
    """Sample centers via the best-candidate heuristic.

    Parameters
    ----------
    k : int
        Number of centers.
    p : int
        Dimension of each center.
    center_box : float
        Half-width of the sampling hypercube.
    rng : numpy.random.Generator
        Random generator.
    n_candidates : int, default=64
        Number of candidate points considered per new center.

    Returns
    -------
    centers : ndarray of shape (k, p)
        Center coordinates.
    """
    if n_candidates <= 0:
        raise ValueError("`n_candidates` must be positive.")
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
    """Sample centers using Latin hypercube stratification.

    Parameters
    ----------
    k : int
        Number of centers.
    p : int
        Dimension of each center.
    center_box : float
        Half-width of the sampling hypercube.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    centers : ndarray of shape (k, p)
        Center coordinates.
    """
    # latin hypercube: each dim gets a random permutation of k bins, with jitter inside each bin
    X = np.empty((k, p), dtype=float)

    for j in range(p):
        perm = rng.permutation(k)
        u = (perm + rng.random(k)) / k  # jitter inside each bin -> U(0,1)
        X[:, j] = -center_box + (2 * center_box) * u

    return X

def _min_dist_centers(
    k: int,
    p: int,
    center_box: float,
    rng: np.random.Generator,
    min_center_dist: float,
    max_tries: int = 200000
) -> np.ndarray:
    """Sample centers with a minimum spacing constraint.

    Parameters
    ----------
    k : int
        Number of centers.
    p : int
        Dimension of each center.
    center_box : float
        Half-width of the sampling hypercube.
    rng : numpy.random.Generator
        Random generator.
    min_center_dist : float
        Minimum Euclidean distance between any pair of centers.
    max_tries : int, default=200000
        Maximum random draws before giving up.

    Returns
    -------
    centers : ndarray of shape (k, p)
        Center coordinates.
    """
    centers = []
    tries = 0
    if min_center_dist <= 0:
        raise ValueError("`min_center_dist` must be positive.")
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
        raise ValueError(
            f"could not place {k} centers with min_center_dist={min_center_dist}. "
            "try reducing it or enlarging center_box."
        )

    return np.vstack(centers)