"""Public API for synthetic cluster simulations."""

from typing import Callable, Literal, Tuple
import numpy as np

from ._outliers import _parse_outliers, _sample_outliers
from ._centers import _sample_centers
from ._sizes import _compute_cluster_sizes, _get_cluster_scales, _post_embed_scaling
from ._covariance import _build_correlation_matrix, _cluster_covariances
from ._distributions import _sample_cluster_points
from ._embed import _apply_embedding
from ._plot import _plot_simulation
from ._noise import _sample_noise


def simulate_clusters(
    n_total: int,
    p: int,
    k: int,
    cluster_scale: float | list[float] | Callable[[], float] = 1.0,
    balanced: bool = False,
    cluster_size_frac: list[float] | None = None,
    min_cluster_size_abs: int = 5,
    min_cluster_size_floor_frac: float = 0.1,
    cluster_size_dirichlet_alpha: float | np.ndarray = 0.5,
    center_box: float = 3.0,
    centroid_method: Literal["none", "lhs", "best_candidate", "min_dist"] = "best_candidate",
    n_candidates: int = 64,
    corr_type: Literal["none", "ar1", "block"] = "none",
    corr_strength: float = 1.0,
    block_size: int | None = None,
    outliers: int | float = 0,
    outlier_scale: float = 5.0,
    outlier_mode: Literal["far_gaussian", "uniform_box"] = "far_gaussian",
    distribution: Literal["gaussian", "t", "uniform_ball", "circles", "moons", "swiss_roll"] = "gaussian",
    t_df: int = 3,
    nonlinear: bool = False,
    embed_dim: int | None = None,
    embed_method: Literal["random_fourier", "poly", "rbf"] = "random_fourier",
    embed_param: float = 2.0,
    embed_standardize: bool = True,
    post_embed_mode: Literal["none", "standardize", "preserve_global", "standardize_preserve"] = "standardize",
    post_embed_scale: float = 1.0,
    noise_dims: int = 0,
    noise_dist: Literal["gaussian", "uniform", "laplace", "t"] = "gaussian",
    noise_scale: float | Literal["match"] = "match",
    plotting: bool = False,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic clustered data with optional correlation and embedding.

    Parameters
    ----------
    n_total : int
        Total samples (clusters + outliers).
    p : int
        Base feature count before embedding/noise.
    k : int
        Number of clusters.
    cluster_scale : float or list of float or callable, default=1.0
        Per-cluster scale(s) (interpreted as standard deviation).
    balanced : bool, default=False
        If True and ``cluster_size_frac`` is None, produce near-equal sizes.
    cluster_size_frac : list of float or None, default=None
        Explicit per-cluster proportions (length k; will be normalized).
    min_cluster_size_abs : int, default=5
        Minimum samples per cluster for unbalanced sizing.
    min_cluster_size_floor_frac : float, default=0.1
        Minimum fraction per cluster for unbalanced sizing.
    cluster_size_dirichlet_alpha : float or ndarray, default=0.5
        Dirichlet concentration for unbalanced sizing.
    center_box : float, default=3.0
        Half-width of the hypercube for center placement.
    centroid_method : {"none", "lhs", "best_candidate", "min_dist"}, default="best_candidate"
        Center placement strategy.
    n_candidates : int, default=64
        Candidates per center for the best-candidate method.
    corr_type : {"none", "ar1", "block"}, default="none"
        Correlation structure.
    corr_strength : float, default=1.0
        Correlation strength for ar1/block structures.
    block_size : int or None, default=None
        Block size for block correlation.
    outliers : int or float, default=0
        Outlier count or fraction in (0, 1).
    outlier_scale : float, default=5.0
        Scale multiplier for outlier distance/box size.
    outlier_mode : {"far_gaussian", "uniform_box"}, default="far_gaussian"
        Outlier sampling mode.
    distribution : {"gaussian", "t", "uniform_ball", "circles", "moons", "swiss_roll"}, default="gaussian"
        Distribution inside each cluster.
    t_df : int, default=3
        Degrees of freedom for t-distributed samples.
    nonlinear : bool, default=False
        Apply nonlinear embedding to generated clusters.
    embed_dim : int or None, default=None
        Output dimension for random Fourier features.
    embed_method : {"random_fourier", "poly", "rbf"}, default="random_fourier"
        Embedding method.
    embed_param : float, default=2.0
        Kernel lengthscale or polynomial degree.
    embed_standardize : bool, default=True
        Standardize before random Fourier embedding.
    post_embed_mode : {"none", "standardize", "preserve_global", "standardize_preserve"}, default="standardize"
        Post-embedding scaling mode.
    post_embed_scale : float, default=1.0
        Global scale multiplier after embedding.
    noise_dims : int, default=0
        Number of extra noise dimensions.
    noise_dist : {"gaussian", "uniform", "laplace", "t"}, default="gaussian"
        Noise distribution.
    noise_scale : float or {"match"}, default="match"
        Noise scale or "match" to signal standard deviation.
    plotting : bool, default=False
        If True, show a PCA plot of the simulated data.
    random_state : int or None, default=None
        RNG seed.

    Returns
    -------
    X : ndarray of shape (n_total, p + noise_dims)
        Simulated data matrix.
    y : ndarray of shape (n_total,)
        Labels (0..k-1 for clusters, -1 for outliers).

    Notes
    -----
    - If ``cluster_size_frac`` is set, it overrides ``balanced`` and Dirichlet settings.
    - Post-embedding transforms apply only when ``nonlinear=True``.
    """
    rng = np.random.default_rng(seed=random_state)
    p = int(np.floor(p + 0.5))
    n_total = int(np.floor(n_total + 0.5))
    k = int(np.floor(k + 0.5))
    noise_dims = int(np.floor(noise_dims + 0.5))
    t_df = int(np.floor(t_df + 0.5))
    embed_dim = int(np.floor(embed_dim + 0.5)) if embed_dim is not None else None
    
    if p <= 0:
        raise ValueError("`p` must be positive.")
    if n_total <= 0:
        raise ValueError("`n_total` must be positive.")
    if k <= 0:
        raise ValueError("`k` must be positive.")
    if min_cluster_size_abs < 0:
        raise ValueError("`min_cluster_size_abs` must be nonnegative.")
    if min_cluster_size_floor_frac < 0:
        raise ValueError("`min_cluster_size_floor_frac` must be nonnegative.")
    if not np.isfinite(corr_strength):
        raise ValueError("`corr_strength` must be finite.")
    if not np.isfinite(embed_param):
        raise ValueError("`embed_param` must be finite.")
    if (distribution == "t" or noise_dist == "t") and t_df <= 0:
        raise ValueError("`t_df` must be positive for t distributions.")
    noise_dims = int(np.floor(noise_dims + 0.5))
    if noise_dims < 0:
        raise ValueError("`noise_dims` must be nonnegative.")

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
        cluster_size_frac=cluster_size_frac,
        rng=rng,
        min_abs=min_cluster_size_abs,
        min_frac=min_cluster_size_floor_frac,
        alpha=cluster_size_dirichlet_alpha
    )
    assert int(cluster_sizes.sum()) == n_total_clusters

    # Centers and covariances
    centers = _sample_centers(
        k=k, p=p, center_box=center_box, rng=rng,
        method=centroid_method, n_candidates=n_candidates
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

    # Concatenate
    X = np.vstack(X_parts) if X_parts else np.empty((0, p), dtype=float)
    y = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=int)

    # Optional embedding
    X_pre_embed = X
    if nonlinear:
        X = _apply_embedding(
            X, method=embed_method, embed_dim=embed_dim,
            embed_param=embed_param, standardize=embed_standardize, rng=rng
        )
        
        X = _post_embed_scaling(
            X,
            X_pre_embed=X_pre_embed,
            mode=post_embed_mode,
            scale=post_embed_scale
        )

    # Add noise dimensions
    if noise_dims > 0:
        if noise_scale == "match":
            stds = X.std(axis=0, ddof=1)
            base = float(np.nanmean(np.where(stds > 0, stds, np.nan))) if X.shape[1] > 0 else 1.0
            if not np.isfinite(base) or base == 0.0:
                base = 1.0
            scale_val = base
        else:
            scale_val = float(noise_scale)

        Z = _sample_noise(
            rng=rng, n=X.shape[0], q=noise_dims,
            dist=noise_dist, scale=scale_val, t_df=t_df
        )
        X = np.hstack([X, Z])

    # Shuffle
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    # Optional plotting
    if plotting:
        _plot_simulation(X, y, random_state=random_state)
    
    return X, y