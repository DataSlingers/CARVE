"""Noise feature generation for simulations."""

import numpy as np
from typing import Literal


def _sample_noise(
    rng: np.random.Generator,
    n: int,
    q: int,
    dist: Literal["gaussian", "uniform", "laplace", "t"] = "gaussian",
    scale: float = 1.0,
    t_df: int = 3,
) -> np.ndarray:
    """Sample independent noise features.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    n : int
        Number of samples.
    q : int
        Number of noise dimensions (rounded to nearest int).
    dist : {"gaussian", "uniform", "laplace", "t"}, default="gaussian"
        Noise distribution.
    scale : float, default=1.0
        Multiplicative scale for noise.
    t_df : int, default=3
        Degrees of freedom for t noise.

    Returns
    -------
    Z : ndarray of shape (n_samples, q)
        Noise feature matrix.
    """
    q_int = int(np.floor(q + 0.5))
    if q_int <= 0:
        return np.empty((n, 0), dtype=float)

    if dist == "gaussian":
        Z = rng.standard_normal(size=(n, q_int))

        return Z * scale

    elif dist == "uniform":
        # centered uniform with variance ~ scale^2
        # U ~ Unif(-sqrt(3), sqrt(3)) has Var=1, so multiply by scale
        Z = rng.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(n, q_int))

        return Z * scale

    elif dist == "laplace":
        # Laplace(0, b) has Var=2 b^2; choose b = scale / sqrt(2)
        b = scale / np.sqrt(2)

        return rng.laplace(loc=0.0, scale=b, size=(n, q_int))

    elif dist == "t":
        # Student t scaled to roughly unit variance when df>2: Var = df/(df-2)
        if t_df <= 0:
            raise ValueError("`t_df` must be positive for t noise.")
        Z = rng.standard_t(df=t_df, size=(n, q_int))
        if t_df > 2:
            Z = Z / np.sqrt(t_df / (t_df - 2))

        return Z * scale

    else:
        raise ValueError(f"Unknown noise dist: {dist}")
