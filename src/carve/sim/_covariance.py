import numpy as np
from typing import Literal

def _build_correlation_matrix(
    *, p: int, corr_type: Literal["none", "ar1", "block"], corr_strength: float, block_size: int | None
) -> np.ndarray:
    """
    Construct a correlation matrix for the specified correlation structure.

    Parameters:
        - `p`: number of dimensions.
        - `corr_type`: "none" (identity), "ar1" (toeplitz), or "block" (block-constant).
        - `corr_strength`: correlation strength (constraints depend on `corr_type`).
        - `block_size`: block size for block correlation (required when corr_type="block").

    Returns:
        - (p, p) correlation matrix.
    """
    if corr_type == "none":
        return np.eye(p)
    
    elif corr_type == "ar1":
        if not np.isfinite(corr_strength) or abs(corr_strength) >= 1:
            raise ValueError("`corr_strength` for ar1 must satisfy |corr_strength| < 1.")
        idx = np.arange(p)
        return corr_strength ** np.abs(idx[:, None] - idx[None, :])
    
    elif corr_type == "block":
        if block_size is None:
            raise ValueError("`block_size` must be provided when corr_type='block'")
        if block_size <= 0:
            raise ValueError("`block_size` must be positive.")
        if not np.isfinite(corr_strength):
            raise ValueError("`corr_strength` must be finite.")
        if block_size > 1:
            min_allowed = -1.0 / (block_size - 1)
            if not (min_allowed < corr_strength < 1.0):
                raise ValueError(
                    f"`corr_strength` must be in ({min_allowed:.3f}, 1.0) for block_size={block_size}."
                )
        
        R = np.zeros((p, p))
        for start in range(0, p, block_size):
            end = min(start + block_size, p)
            R[start:end, start:end] = corr_strength
        np.fill_diagonal(R, 1.0)
        
        return R
    
    else:
        raise ValueError(f"Unknown corr_type `{corr_type}`")

def _cluster_covariances(scales: list[float], R: np.ndarray) -> list[np.ndarray]:
    """
    Convert per-cluster scales into covariance matrices using a shared correlation matrix.

    Parameters:
        - `scales`: list of per-cluster scale values (interpreted as standard deviations).
        - `R`: base correlation matrix.

    Returns:
        - list of (p, p) covariance matrices, one per cluster.
    """
    covs = []
    for s in scales:
        if s < 0 or not np.isfinite(s):
            raise ValueError("`cluster_scale` values must be nonnegative and finite.")
        covs.append((s ** 2) * R)
    return covs