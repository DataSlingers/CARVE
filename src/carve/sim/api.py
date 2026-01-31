from typing import Callable, Literal, NamedTuple, Tuple
import numpy as np

from ._outliers import _parse_outliers, _sample_outliers
from ._centers import _sample_centers
from ._sizes import _compute_cluster_sizes, _get_cluster_scales, _post_embed_scaling
from ._covariance import _build_correlation_matrix, _cluster_covariances
from ._distributions import _sample_cluster_points
from ._embed import _apply_embedding
from ._plot import _plot_simulation
from ._noise import _sample_noise

class SimulationMeta(NamedTuple):
        """
        Metadata for a simulated dataset.

        Fields:
            - `centers`: (k, p) cluster centers in the original signal space.
            - `cluster_sizes`: (k,) integer sizes per cluster.
            - `cluster_scales`: list of per-cluster scales (std-devs).
            - `correlation`: (p, p) base correlation matrix.
            - `covariances`: list of (p, p) covariance matrices per cluster.
            - `outliers`: integer number of outliers sampled.
            - `signal_dims`: number of signal dimensions (post-embedding, pre-noise).
            - `noise_dims`: number of appended noise dimensions.
            - `noise_mask`: boolean mask over final dimensions indicating noise features.
        """
        centers: np.ndarray
        cluster_sizes: np.ndarray
        cluster_scales: list[float]
        correlation: np.ndarray
        covariances: list[np.ndarray]
        outliers: int
        signal_dims: int
        noise_dims: int
        noise_mask: np.ndarray

def simulate_clusters(
    n_total: int,
    p: int,
    k: int,
    cluster_scale: float | list[float] | Callable[[], float] = 1.0,
    balanced: bool = True,
    cluster_size_frac: list[float] | None = None,
    min_cluster_size_abs: int = 5,
    min_cluster_size_floor_frac: float = 0.1,
    cluster_size_dirichlet_alpha: float | np.ndarray = 0.3,
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
    center_box: float = 3.0,
    centroid_method: Literal["none", "lhs", "best_candidate", "min_dist"] = "best_candidate",
    n_candidates: int = 64,
    embed_standardize: bool = True,
    post_embed_mode: Literal["none", "standardize", "preserve_global", "standardize_preserve"] = "standardize",
    post_embed_scale: float = 1.0,
    noise_dims: int = 0,
    noise_dist: Literal["gaussian", "uniform", "laplace", "t"] = "gaussian",
    noise_scale: float | Literal["match"] = "match",
    plotting: bool = False,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, SimulationMeta]:
    """
    Generate synthetic clustered data with optional correlation, outliers, and embedding.

        Parameter legend (grouped):
            Core:
                - `n_total`: total samples (clusters + outliers).
                - `p`: base feature count before embedding/noise.
                - `k`: number of clusters.

            Cluster sizes:
                - `cluster_size_frac`: explicit per-cluster proportions (length k, sums to 1).
                - `balanced`: if True, equal sizes when `cluster_size_frac` is None.
                - `min_cluster_size_abs`: floor in absolute counts (unbalanced only).
                - `min_cluster_size_floor_frac`: floor as fraction of n_total (unbalanced only).
                - `cluster_size_dirichlet_alpha`: Dirichlet concentration for unbalanced sizes.

            Cluster shapes:
                - `cluster_scale`: per-cluster scale (std-dev); covariances are scale^2 * correlation.
                - `corr_type`: "none", "ar1", or "block" correlation structure.
                - `corr_strength`: correlation strength for ar1/block.
                - `block_size`: block size for block correlation.

            Center layout:
                - `center_box`: half-width of the hypercube for center placement.
                - `centroid_method`: "none", "lhs", "best_candidate", "min_dist".
                - `n_candidates`: candidates for best-candidate center selection.

            Distributions:
                - `distribution`: point distribution inside each cluster.
                - `t_df`: degrees of freedom for t distributions.

            Outliers:
                - `outliers`: integer count or fraction in (0,1).
                - `outlier_mode`: "far_gaussian" or "uniform_box".
                - `outlier_scale`: multiplier for outlier distance/box size.

            Nonlinear embedding:
                - `nonlinear`: apply embedding to generated clusters.
                - `embed_method`: "random_fourier", "poly", or "rbf".
                - `embed_dim`: output dimension for random Fourier features.
                - `embed_param`: kernel lengthscale or polynomial degree.
                - `embed_standardize`: standardize before embedding (random_fourier only).
                - `post_embed_mode`: "none", "standardize", "preserve_global", "standardize_preserve".
                - `post_embed_scale`: global scale multiplier after embedding.

            Noise:
                - `noise_dims`: number of extra noise dimensions.
                - `noise_dist`: noise distribution.
                - `noise_scale`: noise scale or "match" to signal std.

            Misc:
                - `plotting`: show PCA plot of the simulated data.
                - `random_state`: RNG seed.

        Precedence rules:
            - If `cluster_size_frac` is set, it overrides `balanced` and the Dirichlet settings.
            - Post-embedding transforms apply only when `nonlinear=True`.

    Returns: (X, y, meta)
      - X: (n_total, p) array
      - y: (n_total,) labels, 0..k-1 for clusters, -1 for outliers
      - meta: SimulationMeta with centers, sizes, scales, correlation, covariances, outliers
    """
    rng = np.random.default_rng(seed=random_state)
    p = int(np.floor(p + 0.5))
    n_total = int(np.floor(n_total + 0.5))
    k = int(np.floor(k + 0.5))
    noise_dims = int(np.floor(noise_dims + 0.5))
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

    # build meta data
    total_dims = X.shape[1]
    noise_dims_int = int(noise_dims)
    sig_dims = total_dims - noise_dims_int if noise_dims_int > 0 else total_dims
    noise_mask = np.zeros(total_dims, dtype=bool)
    
    if noise_dims_int > 0:
        noise_mask[sig_dims: total_dims] = True

    meta = SimulationMeta(
        centers=centers, cluster_sizes=cluster_sizes, cluster_scales=scales,
        correlation=R, covariances=covs, outliers=int(n_outliers),
        signal_dims=sig_dims, noise_dims=noise_dims_int, noise_mask=noise_mask
    )
    
    return X, y, meta