from typing import Callable, Literal, NamedTuple, Tuple
import numpy as np

from ._outliers import _parse_outliers, _sample_outliers
from ._centers import _sample_centers
from ._sizes import _compute_cluster_sizes, _get_cluster_scales
from ._covariance import _build_correlation_matrix, _cluster_covariances
from ._distributions import _sample_cluster_points
from ._embed import _apply_embedding
from ._plot import _plot_simulation

class SimulationMeta(NamedTuple):
    centers: np.ndarray
    cluster_sizes: np.ndarray
    cluster_scales: list[float]
    correlation: np.ndarray
    covariances: list[np.ndarray]
    outliers: int

def simulate_clusters(
    n_total: int,
    p: int,
    k: int,
    cluster_scale: float | list[float] | Callable[[], float] = 1.0,
    balanced: bool = True,
    cluster_sizes_frac: list[float] | None = None,
    min_cluster_size_abs: int = 5,
    min_cluster_size_frac: float = 0.1,
    cluster_size_dirichlet_alpha: float | np.ndarray = 0.3,
    corr_type: Literal["none", "ar1", "block"] = "none",
    corr_strength: float = 1.0,
    block_size: int | None = None,
    outliers: int | float = 0,
    outlier_scale: float = 5.0,
    outlier_mode: Literal["far_gaussian", "uniform_box"] = "far_gaussian",
    distribution: Literal["gaussian", "t", "uniform_ball"] = "gaussian",
    t_df: int = 3,
    nonlinear: bool = False,
    embed_dim: int | None = None,
    embed_method: Literal["random_fourier", "poly", "rbf"] = "random_fourier",
    embed_param: float = 2.0,
    center_box: float = 10.0,
    centroid_method: Literal["none", "lhs", "best_candidate", "min_dist"] = "best_candidate",
    n_candidates: int = 64,
    min_center_dist: float | None = None,
    plotting: bool = True,
    random_state: int | None = None
) -> Tuple[np.ndarray, np.ndarray, SimulationMeta]:
    """
    Generate synthetic clustered data with optional correlation, outliers, and embedding.

    Returns: (X, y, meta)
      - X: (n_total, p) array
      - y: (n_total,) labels, 0..k-1 for clusters, -1 for outliers
      - meta: SimulationMeta with centers, sizes, scales, correlation, covariances, outliers
    """
    rng = np.random.default_rng(seed=random_state)

    # Outliers
    n_outliers = _parse_outliers(outliers, n_total)
    n_total_clusters = n_total - n_outliers
    if n_total_clusters < 1:
        raise ValueError("`outliers` too large, no points left for clusters.")

    # Sizes
    cluster_sizes = _compute_cluster_sizes(
        n_total_clusters=n_total_clusters,
        k=k,
        balanced=balanced,
        cluster_sizes_frac=cluster_sizes_frac,
        rng=rng,
        min_abs=min_cluster_size_abs,
        min_frac=min_cluster_size_frac,
        alpha=cluster_size_dirichlet_alpha
    )
    assert int(cluster_sizes.sum()) == n_total_clusters

    # Centers and covariances
    centers = _sample_centers(
        k=k, p=p, center_box=center_box, rng=rng,
        method=centroid_method, n_candidates=n_candidates, min_center_dist=min_center_dist
    )
    scales = _get_cluster_scales(cluster_scale, k)
    R = _build_correlation_matrix(p=p, corr_type=corr_type, corr_strength=corr_strength, block_size=block_size)
    covs = _cluster_covariances(scales, R)

    # Sample clusters
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for c in range(k):
        size = int(cluster_sizes[c])
        mean = centers[c]
        cov = covs[c]
        X_c = _sample_cluster_points(
            rng=rng, size=size, mean=mean, cov=cov,
            distribution=distribution, t_df=t_df
        )
        X_parts.append(X_c)
        y_parts.append(np.full(size, c, dtype=int))

    # Outliers
    if n_outliers > 0:
        X_out = _sample_outliers(
            rng=rng, n_outliers=int(n_outliers), p=p,
            centers=centers, cluster_sizes=cluster_sizes, covs=covs,
            center_box=center_box, outlier_mode=outlier_mode, outlier_scale=outlier_scale
        )
        X_parts.append(X_out)
        y_parts.append(np.full(int(n_outliers), -1, dtype=int))

    # Concatenate and shuffle
    X = np.vstack(X_parts) if X_parts else np.empty((0, p), dtype=float)
    y = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=int)

    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    # Optional embedding
    X = _apply_embedding(
        X, nonlinear=nonlinear, method=embed_method, embed_dim=embed_dim,
        embed_param=embed_param, rng=rng
    )

    # Optional plotting
    if plotting:
        _plot_simulation(X, y, random_state=random_state)

    meta = SimulationMeta(
        centers=centers, cluster_sizes=cluster_sizes, cluster_scales=scales,
        correlation=R, covariances=covs, outliers=int(n_outliers)
    )
    return X, y, meta