"""Nonlinear embedding helpers for simulations."""

import numpy as np
from typing import Literal


def _apply_embedding(
    X: np.ndarray,
    *,
    method: Literal["random_fourier", "poly", "rbf"],
    embed_dim: int | None,
    embed_param: float,
    rng: np.random.Generator,
    standardize: bool = True,
) -> np.ndarray:
    """Apply a nonlinear embedding to the data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input array.
    method : {"random_fourier", "poly", "rbf"}
        Embedding method.
    embed_dim : int or None
        Output dimension for random Fourier features (required for random_fourier).
    embed_param : float
        Kernel lengthscale for random_fourier/rbf or degree for poly.
    rng : numpy.random.Generator
        Random generator.
    standardize : bool, default=True
        Standardize input before random Fourier features.

    Returns
    -------
    Z : ndarray of shape (n_samples, d)
        Embedded array, where d depends on ``method``.
    """
    if method == "random_fourier":
        if embed_dim is None:
            raise ValueError("embed_dim must be provided for random_fourier")
        if not np.isfinite(embed_param) or embed_param <= 0:
            raise ValueError(
                "`embed_param` must be positive for random_fourier (kernel lengthscale)."
            )

        X = np.asarray(X)

        if standardize:
            X = (X - X.mean(axis=0, keepdims=True)) / (
                X.std(axis=0, keepdims=True) + 1e-12
            )

        D = int(embed_dim)
        if D <= 0:
            raise ValueError("embed_dim must be positive")

        # For k(x,y)=exp(-||x-y||^2/(2*sigma^2)), sample w ~ N(0, sigma^{-2} I)
        W = rng.normal(size=(X.shape[1], D)) / embed_param
        b = rng.uniform(0.0, 2.0 * np.pi, size=(D,))

        Z = np.sqrt(2.0 / D) * np.cos(X @ W + b)
        return Z

    elif method == "poly":
        if not np.isfinite(embed_param):
            raise ValueError("`embed_param` must be finite for poly.")
        d = int(round(embed_param))
        if not np.isclose(d, embed_param):
            raise ValueError(
                "`embed_param` must be an integer degree for poly embedding."
            )
        if d <= 0:
            raise ValueError(
                "`embed_param` must be a positive integer for poly embedding."
            )

        cols = [X, X**d]
        if X.shape[1] >= 2:
            cols.append((X[:, [0]] * X[:, [1]]))

        return np.hstack(cols)

    elif method == "rbf":
        sigma_rbf = embed_param
        if not np.isfinite(sigma_rbf) or sigma_rbf <= 0:
            raise ValueError("`embed_param` must be positive for rbf.")
        m = min(50, len(X))
        if m == 0:
            return np.empty((len(X), 0), dtype=float)
        prototypes = X[rng.choice(len(X), size=m, replace=False)]
        gamma = 1.0 / (2.0 * sigma_rbf**2)

        return np.exp(-gamma * ((X[:, None, :] - prototypes[None, :, :]) ** 2).sum(-1))

    else:
        raise ValueError(f"Unknown embed_method `{method}`")
