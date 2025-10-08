import numpy as np
from typing import Literal

def _apply_embedding(
    X: np.ndarray, *, 
    method: Literal["random_fourier", "poly", "rbf"],
    embed_dim: int | None, 
    embed_param: float, 
    rng: np.random.Generator,
    scale_by_dim: bool = True
) -> np.ndarray:
    if method in ("random_fourier", "fourier"):
        if embed_dim is None:
            raise ValueError("embed_dim must be provided for random_fourier")
        
        embed_dim_f = int(round(embed_dim / 2))
        
        freq_scale = embed_param
        W = rng.normal(scale=freq_scale, size=(X.shape[1], embed_dim_f))
        b = rng.uniform(0, 2 * np.pi, size=(embed_dim_f,))
        Z = np.hstack([np.sin(X @ W + b), np.cos(X @ W + b)])
    
        if scale_by_dim:
            Z /= np.sqrt(embed_dim_f)
            
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