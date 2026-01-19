import numpy as np
from typing import Literal

def _chol_spd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Robust Cholesky with tiny jitter if needed."""
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(A + eps * np.eye(A.shape[0], dtype=A.dtype))

def _sample_cluster_points(
    *, rng: np.random.Generator, size: int, mean: np.ndarray, cov: np.ndarray,
    distribution: Literal["gaussian", "t", "uniform_ball", "circles", "moons", "swiss_roll"],
    t_df: int
) -> np.ndarray:
    p = mean.shape[0]
    if size == 0:
        return np.empty((0, p), dtype=float)

    if distribution == "gaussian":
        return rng.multivariate_normal(mean=mean, cov=cov, size=size)

    elif distribution == "t":
        L = _chol_spd(cov)
        z = rng.standard_normal((size, p))
        v = rng.chisquare(t_df, size=(size, 1))
        return mean + (z @ L.T) * np.sqrt(t_df / v)

    elif distribution == "uniform_ball":
        U = rng.standard_normal((size, p))
        norms = np.linalg.norm(U, axis=1, keepdims=True)

        norms[norms == 0] = 1.0
        U /= norms
        radii = rng.random(size) ** (1.0 / p)
        Z = U * radii.reshape(-1, 1)

        L = _chol_spd(cov)
        Z = (Z @ L.T) * np.sqrt(p + 2)

        return mean + Z

    elif distribution == "circles":
        if p < 2:
            raise ValueError("`circles` distribution requires p >= 2.")
        
        theta = rng.uniform(0.0, 2.0 * np.pi, size=size)
        thickness = 0.08
        r = 1.0 + rng.normal(scale=thickness, size=size)
        Z2 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        
        A = rng.standard_normal((p, 2))
        Q, _ = np.linalg.qr(A) 
        Z = Z2 @ Q.T 

        mu = Z.mean(axis=0, keepdims=True)
        sd = Z.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (Z - mu) / sd
        
        L = _chol_spd(cov)
        Z = Z @ L.T

        return mean + Z

    elif distribution == "moons":
        if p < 2:
            raise ValueError("`moons` distribution requires p >= 2.")

        theta = rng.uniform(0.0, np.pi, size=size)
        r = 1.0 + rng.normal(scale=0.04, size=size)
        x = r * np.cos(theta) + 0.4
        y = r * np.sin(theta) + rng.normal(scale=0.04, size=size)
        Z2 = np.stack([x, y], axis=1)
        
        A = rng.standard_normal((p, 2))
        Q, _ = np.linalg.qr(A) 
        Z = Z2 @ Q.T 

        mu = Z.mean(axis=0, keepdims=True)
        sd = Z.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (Z - mu) / sd
        
        L = _chol_spd(cov)
        Z = Z @ L.T

        return mean + Z

    elif distribution == "swiss_roll":
        if p < 2:
            raise ValueError("`swiss_roll` requires p >= 2.")
        
        t = rng.uniform(0.6 * np.pi, 4.5 * np.pi, size=size)
        a, b, band = 0.3, 0.15, 0.06
        r = a + b * t + rng.normal(scale=band, size=size)
        x = r * np.cos(t)
        y = r * np.sin(t)
        Z2 = np.stack([x, y], axis=1)
        
        A = rng.standard_normal((p, 2))
        Q, _ = np.linalg.qr(A) 
        Z = Z2 @ Q.T 
        
        mu = Z.mean(axis=0, keepdims=True)
        sd = Z.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (Z - mu) / sd
        
        L = _chol_spd(cov)
        Z = Z @ L.T

        return mean + Z

    else:
        raise ValueError("`distribution` must be one of: {'gaussian','t','uniform_ball','circles','moons','swiss_roll'}.")
