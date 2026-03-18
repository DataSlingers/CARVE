"""Cluster size and scaling utilities for simulations."""

import numpy as np
import warnings


def _compute_cluster_sizes(
    *,
    n_total_clusters: int,
    k: int,
    balanced: bool,
    cluster_size_frac: list[float] | None,
    rng: np.random.Generator,
    min_abs: int,
    min_frac: float,
    alpha: float | np.ndarray,
) -> np.ndarray:
    """Compute per-cluster sample sizes.

    Parameters
    ----------
    n_total_clusters : int
        Total number of non-outlier samples.
    k : int
        Number of clusters.
    balanced : bool
        If True and ``cluster_size_frac`` is None, produce near-equal sizes.
    cluster_size_frac : list of float or None
        Explicit proportions per cluster (length k; will be normalized).
    rng : numpy.random.Generator
        Random generator.
    min_abs : int
        Minimum samples per cluster for unbalanced sizing.
    min_frac : float
        Minimum fraction per cluster for unbalanced sizing.
    alpha : float or ndarray
        Dirichlet concentration for unbalanced sizes.

    Returns
    -------
    sizes : ndarray of shape (k,)
        Integer array of cluster sizes summing to ``n_total_clusters``.
    """
    if cluster_size_frac is not None:
        if len(cluster_size_frac) != k:
            raise ValueError(f"cluster_size_frac must have length k={k}.")

        arr = np.asarray(cluster_size_frac, dtype=float)
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("`cluster_size_frac` must be nonnegative and finite.")
        total = float(np.sum(arr))
        if total <= 0:
            raise ValueError("`cluster_size_frac` must sum to a positive value.")
        if abs(total - 1.0) > 1e-8:
            warnings.warn(
                "`cluster_size_frac` does not add up to 1. Rescaling cluster_size_frac."
            )
            arr = arr / total
        sizes = [int(np.floor(n_total_clusters * frac)) for frac in arr]

        # distribute remainder
        for i in range(n_total_clusters - int(np.sum(sizes))):
            sizes[i % k] += 1

        return np.asarray(sizes, dtype=int)

    elif balanced:
        base = n_total_clusters // k
        sizes = [base] * k
        for i in range(n_total_clusters - base * k):
            sizes[i] += 1
        return np.asarray(sizes, dtype=int)

    else:  # unbalanced with constraints
        return _sample_cluster_sizes(
            n_total=n_total_clusters,
            k=k,
            rng=rng,
            min_abs=min_abs,
            min_frac=min_frac,
            alpha=alpha,
        )


def _get_cluster_scales(cluster_scale, k: int) -> list[float]:
    """Resolve per-cluster scale values.

    Parameters
    ----------
    cluster_scale : float, sequence, or callable
        Scalar scale, iterable of length k, or callable returning a float.
    k : int
        Number of clusters.

    Returns
    -------
    scales : list of float
        Per-cluster scales.
    """
    if callable(cluster_scale):
        return [float(cluster_scale()) for _ in range(k)]

    elif isinstance(cluster_scale, (list, tuple, np.ndarray)):
        if len(cluster_scale) != k:
            raise ValueError("Passed `cluster_scale` parameter must be of size k.")
        return [float(x) for x in list(cluster_scale)]

    else:
        return [float(cluster_scale)] * k


def _sample_cluster_sizes(
    n_total: int,
    k: int,
    rng: np.random.Generator,
    min_abs: int = 5,
    min_frac: float = 0.1,
    alpha: float | np.ndarray = 0.3,
    ensure_nonempty: bool = True,
) -> np.ndarray:
    """Sample integer cluster sizes with floor constraints.

    Parameters
    ----------
    n_total : int
        Total samples to allocate.
    k : int
        Number of clusters.
    rng : numpy.random.Generator
        Random generator.
    min_abs : int, default=5
        Minimum absolute size per cluster.
    min_frac : float, default=0.1
        Minimum fraction per cluster.
    alpha : float or ndarray, default=0.3
        Dirichlet concentration (scalar or length-k array).
    ensure_nonempty : bool, default=True
        Enforce at least one sample per cluster.

    Returns
    -------
    sizes : ndarray of shape (k,)
        Cluster sizes summing to ``n_total``.
    """
    if n_total <= 0 or k <= 0:
        raise ValueError("n_total and k must be positive.")
    if min_abs < 0 or min_frac < 0:
        raise ValueError("min_abs and min_frac must be nonnegative.")
    if np.isscalar(alpha):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a positive scalar.")
    else:
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (k,):
            raise ValueError("alpha must be scalar or shape (k,).")
        if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0):
            raise ValueError("alpha entries must be positive and finite.")

    # per-cluster floor
    floor_each = max(min_abs, int(np.ceil(min_frac * n_total)))
    if ensure_nonempty:
        floor_each = max(floor_each, 1)

    total_floor = floor_each * k
    if total_floor > n_total:
        raise ValueError(
            f"Infeasible: k * floor_each = {k} * {floor_each} = {total_floor} > n_total = {n_total}. "
            "Reduce k / floors or increase n_total."
        )

    # assign floors, then distribute the remainder
    sizes = np.full(k, floor_each, dtype=int)
    remaining = n_total - total_floor
    if remaining == 0:
        return sizes

    # Dirichlet probs -> Multinomial integer split
    a = np.full(k, alpha, dtype=float) if np.isscalar(alpha) else alpha.astype(float)
    probs = rng.dirichlet(a)
    sizes += rng.multinomial(remaining, probs)
    return sizes


def _post_embed_scaling(
    X: np.ndarray,
    X_pre_embed: np.ndarray | None = None,
    mode: str = "standardize",
    scale: float = 1.0,
) -> np.ndarray:
    """Apply post-embedding scaling operations.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Embedded data array.
    X_pre_embed : ndarray or None, default=None
        Original data before embedding, used for scale preservation.
    mode : {"none", "standardize", "preserve_global", "standardize_preserve"}, default="standardize"
        Post-embedding scaling mode.
    scale : float, default=1.0
        Global multiplicative scale applied after other transforms.

    Returns
    -------
    X_scaled : ndarray of shape (n_samples, n_features)
        Scaled embedded data array.
    """
    if mode not in {"none", "standardize", "preserve_global", "standardize_preserve"}:
        raise ValueError(
            "`post_embed_mode` must be one of {'none','standardize','preserve_global','standardize_preserve'}."
        )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("`post_embed_scale` must be positive and finite.")

    if mode in {"standardize", "standardize_preserve"}:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
    if mode in {"preserve_global", "standardize_preserve"} and X_pre_embed is not None:
        # global RMS scatter (sqrt mean squared distance to mean) pre vs post
        def _rms_scatter(Z):
            Zc = Z - Z.mean(axis=0, keepdims=True)
            return np.sqrt(np.mean(np.sum(Zc**2, axis=1)))

        s0 = _rms_scatter(X_pre_embed)
        s1 = _rms_scatter(X)

        if s1 > 0:
            X = X * (s0 / s1)

    if scale != 1.0:
        X *= scale

    return X
