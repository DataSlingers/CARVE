import numpy as np
from typing import Literal

def _apply_embedding(
    X: np.ndarray, *, 
    method: Literal["random_fourier", "poly", "rbf"],
    embed_dim: int | None, 
    embed_param: float, 
    rng: np.random.Generator,
    standardize: bool = True
) -> np.ndarray:
    """
    Random Fourier Features for the Gaussian/RBF kernel.
    sigma: kernel lengthscale (bigger = smoother/less nonlinear).
    Output: Z shape (n, embed_dim).
    """
    if method in ("random_fourier", "fourier"):
        if embed_dim is None:
            raise ValueError("embed_dim must be provided for random_fourier")
        
        X = np.asarray(X)

        if standardize:
            X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)

        D = int(embed_dim)
        if D <= 0:
            raise ValueError("embed_dim must be positive")

        # For k(x,y)=exp(-||x-y||^2/(2*sigma^2)), sample w ~ N(0, sigma^{-2} I)
        W = rng.normal(size=(X.shape[1], D)) / embed_param
        b = rng.uniform(0.0, 2.0 * np.pi, size=(D,))

        Z = np.sqrt(2.0 / D) * np.cos(X @ W + b)
        return Z

    elif method == "poly":
        d = embed_param

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