"""Custom spectral clustering implementation for CARVE."""

import numpy as np
from typing import Literal
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


class SpectralClusteringCARVE(BaseEstimator, ClusterMixin):
    """Spectral clustering with self-tuning, RBF, or kNN affinity.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters.
    affinity : {"rbf", "knn", "self_tuning"}, default="self_tuning"
        Affinity type. ``"self_tuning"`` uses per-point local scales
        (Zelnik-Manor & Perona 2004). ``"rbf"`` uses a single global
        gamma. ``"knn"`` builds a sparse kNN graph with RBF-weighted edges.
    gamma : float or None, default=None
        RBF kernel scale for ``"rbf"`` and ``"knn"`` affinities. When None,
        a k-NN median heuristic is used: sigma = median of k-th neighbor
        distances, gamma = 1 / (2 * sigma^2). Ignored for ``"self_tuning"``.
    n_neighbors : int, default=7
        Number of neighbors for kNN graph construction and the self-tuning
        local scale. Zelnik-Manor & Perona recommend 7.
    random_state : int or None, default=None
        Random seed for k-means.
    n_init : "auto" or int, default="auto"
        Number of k-means initializations.
    scale : bool, default=True
        Whether to standardize X before computing affinities.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels assigned after fitting.
    embedding_ : ndarray of shape (n_samples, n_clusters)
        Spectral embedding (row-normalized eigenvectors of the normalized
        graph Laplacian).
    affinity_ : ndarray or scipy.sparse.csr_matrix
        Computed affinity matrix.
    evals_ : ndarray
        Eigenvalues from the Laplacian decomposition.
    gamma_ : float or None
        Computed gamma value (set when ``affinity`` is ``"rbf"`` or ``"knn"``
        and ``gamma=None``). None for ``"self_tuning"``.

    Notes
    -----
    Always uses the normalized Laplacian (L_sym = I - D^{-1/2} W D^{-1/2})
    with Ng-Jordan-Weiss row-normalized eigenvectors, which is strictly
    better than the unnormalized Laplacian (von Luxburg 2007).

    See Also
    --------
    CARVE : Main validation class that uses this estimator in its default
        grid.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        affinity: Literal["rbf", "knn", "self_tuning"] = "self_tuning",
        gamma: float | None = None,
        n_neighbors: int = 7,
        random_state: int | None = None,
        n_init: Literal["auto"] | int = "auto",
        scale: bool = True,
    ):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.n_init = n_init
        self.scale = scale

    def _knn_sigma(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute k-th neighbor distances and median sigma.

        Returns
        -------
        kth_dists : ndarray of shape (n_samples,)
            Distance from each point to its k-th nearest neighbor.
        sigma : float
            Median of k-th neighbor distances.
        """
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        
        # dists[:, 0] is distance to self (0), dists[:, -1] is k-th neighbor
        kth_dists = dists[:, -1]
        sigma = float(np.median(kth_dists))
        if sigma == 0:
            sigma = float(np.mean(kth_dists[kth_dists > 0])) if np.any(kth_dists > 0) else 1.0
        return kth_dists, sigma

    def _compute_rbf_affinity(self, X: np.ndarray) -> np.ndarray:
        """Full pairwise RBF affinity matrix.

        When gamma is None, uses k-NN median heuristic.
        """
        D2 = pairwise_distances(X, metric="sqeuclidean")
        if self.gamma is None:
            _, sigma = self._knn_sigma(X)
            gamma = 1.0 / (2.0 * sigma**2)
            self.gamma_ = gamma
        else:
            gamma = self.gamma
            self.gamma_ = gamma
        W = np.exp(-gamma * D2)
        np.fill_diagonal(W, 0.0)
        return W

    def _compute_knn_affinity(self, X: np.ndarray) -> csr_matrix:
        """Sparse kNN graph with RBF-weighted edges, symmetrized.

        When gamma is None, uses k-NN median heuristic.
        """
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(X)
        dists, indices = nn.kneighbors(X)

        if self.gamma is None:
            kth_dists = dists[:, -1]
            sigma = float(np.median(kth_dists))
            if sigma == 0:
                sigma = float(np.mean(kth_dists[kth_dists > 0])) if np.any(kth_dists > 0) else 1.0
            gamma = 1.0 / (2.0 * sigma**2)
            self.gamma_ = gamma
        else:
            gamma = self.gamma
            self.gamma_ = gamma

        n = X.shape[0]
        rows, cols, vals = [], [], []
        for i in range(n):
            for j_idx in range(self.n_neighbors):
                j = indices[i, j_idx]
                if i == j:
                    continue
                d2 = dists[i, j_idx] ** 2
                w = np.exp(-gamma * d2)
                rows.append(i)
                cols.append(j)
                vals.append(w)

        W = csr_matrix((vals, (rows, cols)), shape=(n, n))
        # Symmetrize: W = (W + W^T) / 2
        W = 0.5 * (W + W.T)
        return W

    def _compute_self_tuning_affinity(self, X: np.ndarray):
        """Self-tuning affinity (Zelnik-Manor & Perona, 2004).

        sigma_i = distance from point i to its k-th nearest neighbor.
        W(i,j) = exp(-d(i,j)^2 / (sigma_i * sigma_j)).

        For n > 5000, uses sparse kNN-only edges for efficiency.
        """
        self.gamma_ = None
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(X)
        dists, indices = nn.kneighbors(X)
        
        # k-th neighbor is at index -1 when n_neighbors=k
        sigma = dists[:, -1].copy()
        
        # Handle zero sigmas
        if np.any(sigma == 0):
            median_nonzero = float(np.median(sigma[sigma > 0])) if np.any(sigma > 0) else 1.0
            sigma[sigma == 0] = median_nonzero

        n = X.shape[0]
        if n > 5000:
            # Sparse: only compute affinities for kNN edges
            rows, cols, vals = [], [], []
            for i in range(n):
                for j_idx in range(self.n_neighbors):
                    j = indices[i, j_idx]
                    if i == j:
                        continue
                    d2 = dists[i, j_idx] ** 2
                    w = np.exp(-d2 / (sigma[i] * sigma[j]))
                    rows.append(i)
                    cols.append(j)
                    vals.append(w)
            W = csr_matrix((vals, (rows, cols)), shape=(n, n))
            W = W.maximum(W.T)  # symmetrize by taking max
            return W
        else:
            # Dense: full pairwise distances
            D2 = pairwise_distances(X, metric="sqeuclidean")
            S = sigma[:, None] * sigma[None, :]
            W = np.exp(-D2 / S)
            np.fill_diagonal(W, 0.0)
            return W

    def _compute_affinity(self, X: np.ndarray):
        """Dispatch to the selected affinity computation."""
        if self.affinity == "rbf":
            return self._compute_rbf_affinity(X)
        elif self.affinity == "knn":
            return self._compute_knn_affinity(X)
        elif self.affinity == "self_tuning":
            return self._compute_self_tuning_affinity(X)
        else:
            raise ValueError(f"Unknown affinity: {self.affinity!r}")

    def _spectral_embedding(self, W, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute spectral embedding from affinity matrix W.

        Always uses normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}
        and row-normalizes eigenvectors (Ng-Jordan-Weiss).

        Uses dense eigh for n < 1000, sparse eigsh otherwise with dense
        fallback on convergence failure.
        """
        from scipy.sparse import issparse as _issparse

        if _issparse(W):
            d = np.asarray(W.sum(axis=1)).ravel()
        else:
            d = np.asarray(W.sum(axis=1)).ravel()

        n = W.shape[0]
        dinvsqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))

        if n < 1000:
            # Dense path
            Wd = W.toarray() if _issparse(W) else W
            Lsym = np.eye(n) - (dinvsqrt[:, None] * Wd * dinvsqrt[None, :])
            # Ensure symmetry
            Lsym = 0.5 * (Lsym + Lsym.T)
            vals, vecs = eigh(Lsym)
            vals, vecs = vals[:k], vecs[:, :k]
        else:
            # Sparse path
            Dinv = diags(dinvsqrt)
            Ws = csr_matrix(W) if not _issparse(W) else W
            Lsym = diags(np.ones(n)) - Dinv @ Ws @ Dinv
            try:
                vals, vecs = eigsh(Lsym, k=k, which="SM", tol=1e-4, maxiter=5000)
            except ArpackNoConvergence as e:
                ev = getattr(e, "eigenvectors", None)
                ew = getattr(e, "eigenvalues", None)
                if ev is not None and ev.shape[1] >= k:
                    vals = ew[:k] if ew is not None else np.full(k, np.nan)
                    vecs = ev[:, :k]
                else:
                    # Dense fallback
                    Ld = Lsym.toarray() if _issparse(Lsym) else np.asarray(Lsym)
                    vals, vecs = eigh(Ld)
                    vals, vecs = vals[:k], vecs[:, :k]

        # Sort by eigenvalue
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]

        # Row-normalize (Ng-Jordan-Weiss)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        U = vecs / (norms + 1e-12)

        return vals, U

    def fit(self, X, y=None):
        """Fit the spectral clustering model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored (sklearn compatibility).

        Returns
        -------
        self : SpectralClusteringCARVE
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        Xp = StandardScaler().fit_transform(X) if self.scale else X

        W = self._compute_affinity(Xp)
        self.affinity_ = W

        vals, U = self._spectral_embedding(W, self.n_clusters)
        self.evals_ = vals
        self.embedding_ = U

        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.labels_ = km.fit_predict(U)
        return self

    def fit_predict(self, X, y=None):
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored (sklearn compatibility).

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return self.fit(X, y).labels_
