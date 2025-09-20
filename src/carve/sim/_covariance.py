import numpy as np
from typing import Literal

def _build_correlation_matrix(
    *, p: int, corr_type: Literal["none", "ar1", "block"], corr_strength: float, block_size: int | None
) -> np.ndarray:
    if corr_type == "none":
        return np.eye(p)
    
    elif corr_type == "ar1":
        idx = np.arange(p)
        return corr_strength ** np.abs(idx[:, None] - idx[None, :])
    
    elif corr_type == "block":
        if block_size is None:
            raise ValueError("`block_size` must be provided when corr_type='block'")
        
        R = np.zeros((p, p))
        for start in range(0, p, block_size):
            end = min(start + block_size, p)
            R[start:end, start:end] = corr_strength
        np.fill_diagonal(R, 1.0)
        
        return R
    
    else:
        raise ValueError(f"Unknown corr_type `{corr_type}`")

def _cluster_covariances(scales: list[float], R: np.ndarray) -> list[np.ndarray]:
    return [s * R for s in scales]