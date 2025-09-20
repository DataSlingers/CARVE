import numpy as np
from typing import Literal

def _sample_cluster_points(
    *, rng: np.random.Generator, size: int, mean: np.ndarray, cov: np.ndarray,
    distribution: Literal["gaussian", "t", "uniform_ball", "circles", "moons", "swiss_roll"], t_df: int
) -> np.ndarray:
    p = mean.shape[0]
    if size == 0:
        return np.empty((0, p), dtype=float)

    if distribution == "gaussian":
        return rng.multivariate_normal(mean=mean, cov=cov, size=size)
    
    elif distribution == "t":
        # X = μ + L z * sqrt(df / v), z~N(0,I), v~χ²(df)
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((size, p))
        v = rng.chisquare(t_df, size=(size, 1))
        return mean + (z @ L.T) * np.sqrt(t_df / v)
    
    elif distribution == "uniform_ball":
        # uniform in p-ball, radius tied to covariance scale
        r_max = np.sqrt(np.trace(cov) / p)
        U = rng.standard_normal((size, p))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        radii = rng.random(size) ** (1.0 / p) * r_max
        return mean + U * radii.reshape(-1, 1)
    
    elif distribution == "circles":
        # Single annulus in 2D, then warp by cov and shift by mean
        if p < 2:
            raise ValueError("`circles` distribution requires p >= 2.")
        theta = rng.uniform(0.0, 2.0 * np.pi, size=size)
        thickness = 0.08  # relative radial thickness
        r = 1.0 + rng.normal(scale=thickness, size=size)
        Z2 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

        Z = np.zeros((size, p), dtype=float)
        Z[:, :2] = Z2

        # center and standardize to roughly unit scale before applying cov
        Z -= Z.mean(axis=0, keepdims=True)
        std = Z.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        Z /= std

        L = np.linalg.cholesky(cov)
        return mean + Z @ L.T

    elif distribution == "moons":
        # Single crescent (half circle) in 2D, then (optionally) rotate randomly,
        # embed into p dims, standardize, warp by cov, and shift by mean.
        if p < 2:
            raise ValueError("`moons` distribution requires p >= 2.")

        # base crescent in canonical orientation
        theta = rng.uniform(0.0, np.pi, size=size)
        r = 1.0 + rng.normal(scale=0.04, size=size)                 # thin band
        x = r * np.cos(theta) + 0.4                                 # shift to create a crescent-like arc
        y = r * np.sin(theta) + rng.normal(scale=0.04, size=size)   # thickness
        Z2 = np.stack([x, y], axis=1)

        # random in-plane rotation (and optional mirror) 
        phi = rng.uniform(0.0, 2.0 * np.pi)                         # random orientation
        c, s = np.cos(phi), np.sin(phi)
        R2 = np.array(
            [[c, -s],
             [s,  c]]
        )
        if rng.random() < 0.5:                                      # random mirror to diversify shapes
            R2[0, :] *= -1
        Z2 = Z2 @ R2.T

        # embed into p dims (optionally on random axes)
        Z = np.zeros((size, p), dtype=float)
        
        # pick two random feature indices so different clusters can live on different planes
        axes = rng.choice(p, size=2, replace=False)
        Z[:, axes] = Z2

        # center and standardize
        Z -= Z.mean(axis=0, keepdims=True)
        std = Z.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        Z /= std

        L = np.linalg.cholesky(cov)
        return mean + Z @ L.T

    elif distribution == "swiss_roll":
        # simple 2D "swiss roll" (spiral band), then drop into p dims and (optionally) warp on those 2 dims
        if p < 2:
            raise ValueError("`swiss_roll` requires p >= 2.")

        # 1) parametric spiral with thickness
        #    r(t) = a + b t, (x,y) = r(cos t, sin t)
        t = rng.uniform(0.6 * np.pi, 4.5 * np.pi, size=size)   # ~1 to ~2.25 turns
        a = 0.3                                                # inner radius offset
        b = 0.15                                               # spacing between coils
        band = 0.06                                            # radial thickness

        r = a + b * t + rng.normal(scale=band, size=size)
        x = r * np.cos(t)
        y = r * np.sin(t)

        Z2 = np.stack([x, y], axis=1)

        # 2) random in-plane pose (keeps it a roll, just rotates / mirrors)
        phi = rng.uniform(0, 2*np.pi)
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s],
                    [s,  c]])
        if rng.random() < 0.5:  # optional mirror for variety
            R[0, :] *= -1
        Z2 = Z2 @ R.T

        # 3) embed into a random 2-D coordinate plane of R^p
        axes = rng.choice(p, size=2, replace=False)
        Z = np.zeros((size, p), dtype=float)
        Z[:, axes] = Z2

        # 4) OPTIONAL: warp only within that 2D subspace (preserves the roll)
        #    (using full p×p Cholesky smears the pattern across unused dims)
        L_sub = np.linalg.cholesky(cov[np.ix_(axes, axes)])
        Z[:, axes] = Z[:, axes] @ L_sub.T

        # 5) shift by mean. no re-standardization (keeps scale + cluster size effects visible)
        return Z + mean

    else:
        raise ValueError("`distribution` must be one of: {'gaussian', 't', 'uniform_ball', 'circles', 'moons', 'swiss_roll'}.")