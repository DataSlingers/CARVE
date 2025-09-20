import numpy as np
from typing import Literal

def _apply_embedding(
    X: np.ndarray, *, nonlinear: bool, method: Literal["random_fourier", "poly", "rbf"],
    embed_dim: int | None, embed_param: float, rng: np.random.Generator
) -> np.ndarray:
    if not nonlinear:
        return X

    if method == "random_fourier" or method == "fourier":
        if embed_dim is None:
            raise ValueError("embed_dim must be provided for random_fourier")
        freq_scale = 0.1
        W = rng.normal(scale=freq_scale, size=(X.shape[1], embed_dim))
        b = rng.uniform(0, 2 * np.pi, size=(embed_dim,))
        return np.sin(X @ W + b)

    elif method == "poly":
        d = embed_param
        # simple example: original + power + pairwise product of first two dims (if present)
        cols = [X, X**d]
        if X.shape[1] >= 2:
            cols.append((X[:, [0]] * X[:, [1]]))
        return np.hstack(cols)

    elif method == "rbf":
        sigma_rbf = embed_param
        m = min(50, len(X))
        prototypes = X[rng.choice(len(X), size=m, replace=False)]
        gamma = 1.0 / (2.0 * sigma_rbf**2)
        return np.exp(-gamma * ((X[:, None, :] - prototypes[None, :, :]) ** 2).sum(-1))

    else:
        raise ValueError(f"Unknown embed_method `{method}`")