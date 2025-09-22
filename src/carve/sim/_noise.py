import numpy as np
from typing import Literal

def _sample_noise(
    rng: np.random.Generator, n: int, q: int,
    dist: Literal["gaussian", "uniform", "laplace", "t"] = "gaussian",
    scale: float = 1.0, t_df: int = 3
) -> np.ndarray:
    if q <= 0:
        return np.empty((n, 0), dtype=float)
    
    if dist == "gaussian":
        Z = rng.standard_normal(size=(n, q))
        
        return Z * scale
    
    elif dist == "uniform":
        # centered uniform with variance ~ scale^2
        # U ~ Unif(-sqrt(3), sqrt(3)) has Var=1, so multiply by scale
        Z = rng.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(n, q))
        
        return Z * scale
    
    elif dist == "laplace":
        # Laplace(0, b) has Var=2 b^2; choose b = scale / sqrt(2)
        b = scale / np.sqrt(2)
        
        return rng.laplace(loc=0.0, scale=b, size=(n, q))
    
    elif dist == "t":
        # Student t scaled to roughly unit variance when df>2: Var = df/(df-2)
        Z = rng.standard_t(df=t_df, size=(n, q))
        if t_df > 2:
            Z = Z / np.sqrt(t_df / (t_df - 2))
            
        return Z * scale
    
    else:
        raise ValueError(f"Unknown noise dist: {dist}")
