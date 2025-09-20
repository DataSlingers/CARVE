import numpy as np
import warnings
from typing import Literal

from ._centers import _validate_center_box

def _parse_outliers(outliers: int | float, n_total: int) -> int:
    if isinstance(outliers, float) and 0 < outliers < 1:
        return int(np.floor(n_total * outliers))
    elif isinstance(outliers, int) and outliers >= 0:
        return outliers
    else:
        warnings.warn("`outliers` must be a non-negative int or a float in (0, 1). Defaulting to 0.")
        return 0
    
def _sample_outliers(
    *, rng: np.random.Generator, n_outliers: int, p: int,
    centers: np.ndarray, cluster_sizes: np.ndarray, covs: list[np.ndarray],
    center_box: float, outlier_mode: Literal["far_gaussian", "uniform_box"], outlier_scale: float
) -> np.ndarray:
    if n_outliers <= 0:
        return np.empty((0, p), dtype=float)

    # weighted centroid + avg covariance
    weights = cluster_sizes.astype(float)
    global_centroid = np.average(centers, axis=0, weights=weights)
    avg_cov = sum(w * C for w, C in zip(weights, covs)) / weights.sum()
    global_sigma = float(np.sqrt(np.mean(np.diag(avg_cov))))

    if outlier_mode == "far_gaussian":
        U = rng.standard_normal(size=(n_outliers, p))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        return global_centroid + outlier_scale * global_sigma * U

    elif outlier_mode == "uniform_box":
        box = _validate_center_box(center_box) * outlier_scale
        return rng.uniform(-box, +box, size=(n_outliers, p))

    else:
        raise ValueError(f"unknown outlier_mode `{outlier_mode}`")