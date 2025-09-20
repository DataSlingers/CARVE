import numpy as np
import warnings

def _compute_cluster_sizes(
    *, n_total_clusters: int, k: int, balanced: bool,
    cluster_sizes_frac: list[float] | None,
    rng: np.random.Generator,
    min_abs: int, min_frac: float, alpha: float | np.ndarray
) -> np.ndarray:
    if cluster_sizes_frac is not None:
        if len(cluster_sizes_frac) != k:
            raise ValueError(f"cluster_sizes_frac must have length k={k}.")
        
        total = float(np.sum(cluster_sizes_frac))
        if abs(total - 1.0) > 1e-8:
            warnings.warn("`cluster_sizes_frac` does not add up to 1. Rescaling cluster_sizes_frac.")
            cluster_sizes_frac = [frac / total for frac in cluster_sizes_frac]
        sizes = [int(np.floor(n_total_clusters * frac)) for frac in cluster_sizes_frac]
        
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
            alpha=alpha
        )

def _get_cluster_scales(cluster_scale, k: int) -> list[float]:
    if callable(cluster_scale):
        return [float(cluster_scale()) for _ in range(k)]
    
    elif isinstance(cluster_scale, list):
        if len(cluster_scale) != k:
            raise ValueError("Passed `cluster_scale` parameter must be of size k.")
        return [float(x) for x in cluster_scale]
    
    else:
        return [float(cluster_scale)] * k
    
def _sample_cluster_sizes(
    n_total: int,
    k: int,
    rng: np.random.Generator,
    min_abs: int = 5,
    min_frac: float = 0.1,
    alpha: float | np.ndarray = 0.3,
    ensure_nonempty: bool = True
) -> np.ndarray:
    """
    Returns integer sizes summing to n_total with floors enforced.
    Strategy: allocate the guaranteed minimum first, then distribute the remainder
    via Multinomial with probabilities from a Dirichlet(alpha).
    """
    if n_total <= 0 or k <= 0:
        raise ValueError("n_total and k must be positive.")
    if min_abs < 0 or min_frac < 0:
        raise ValueError("min_abs and min_frac must be nonnegative.")
    if isinstance(alpha, np.ndarray) and alpha.shape != (k,):
        raise ValueError("alpha must be scalar or shape (k,).")

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