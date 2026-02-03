"""Custom spectral clustering implementation for CARVE."""

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph, csr_matrix, diags
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

class SpectralClusteringCARVE(BaseEstimator, ClusterMixin):
    """Spectral clustering with optional self-tuning affinity.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters.
    affinity : {"rbf", "knn", "self_tuning"}, default="rbf"
        Affinity type.
    gamma : float or None, default=None
        RBF scale; if None, uses a median heuristic.
    n_neighbors : int, default=10
        Number of neighbors for kNN or self-tuning affinity.
    normalized : bool, default=True
        Whether to use the normalized graph Laplacian.
    random_state : int or None, default=None
        Random seed for k-means.
    n_init : "auto" or int, default="auto"
        Number of k-means initializations.
    scale : bool, default=True
        Whether to standardize X before computing affinities.
    eig_tol : float, default=1e-4
        Tolerance for eigen solver convergence.
    eig_maxiter : int, default=5000
        Maximum iterations for eigen solver.
    dense_fallback_n : int, default=2000
        Threshold for dense fallback in eigen decomposition.
    """
    def __init__(
        self,
        n_clusters: int = 2,                                        # number of clusters
        affinity: Literal["rbf", "knn", "self_tuning"] = "rbf",     # distance type
        gamma: Optional[float] = None,                              # rbf scale; if None -> median heuristic
        n_neighbors: int = 10,                                      # graph sparsity / locality for knn/self_tuning
        normalized: bool = True,                                    # use L_sym (recommended) vs unnormalized L
        random_state: Optional[int] = None,                         # controls k-means seeding
        n_init: Literal["auto"] | int = "auto",                     # k-means restarts for stability
        scale: bool = True,                                         # standardize X first
        eig_tol: float = 1e-4, 
        eig_maxiter: int = 5000, 
        dense_fallback_n: int = 2000
    ):
        """Initialize the estimator with spectral clustering parameters."""
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.normalized = normalized
        self.random_state = random_state
        self.n_init = n_init
        self.scale = scale
        self.eig_tol = eig_tol
        self.eig_maxiter = eig_maxiter
        self.dense_fallback_n = dense_fallback_n
        self.converged_ = None
        self.used_dense_fallback_ = False
    
    def _rbf_affinity(self, X: np.ndarray) -> csr_matrix:
        """Compute an RBF affinity matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        W : scipy.sparse.csr_matrix
            RBF affinity matrix.
        """
        D2 = pairwise_distances(X, metric="sqeuclidean")
        if self.gamma is None:
            # median heuristic
            tri = D2[np.triu_indices_from(D2, k=1)]
            sigma2 = np.median(tri)
            gamma = 1.0 / (2.0 * sigma2) if sigma2 > 0 else 1.0
        else:
            gamma = self.gamma
        W = np.exp(-gamma * D2)
        np.fill_diagonal(W, 0.0)
        return csr_matrix(W)
    
    def _knn_affinity(self, X: np.ndarray) -> csr_matrix:
        """Compute a mutual kNN affinity matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        W : scipy.sparse.csr_matrix
            Symmetrized kNN affinity matrix.
        """
        # mutual k-NN (symmetrize)
        G = kneighbors_graph(X, self.n_neighbors, mode="distance", include_self=False)
        # convert distances to weights with a global gamma via median heuristic
        if self.gamma is None:
            d = G.data
            sigma2 = np.median(d**2) if d.size else 1.0
            gamma = 1.0 / (2.0 * sigma2) if sigma2 > 0 else 1.0
        else:
            gamma = self.gamma
        G.data = np.exp(-(G.data**2) * gamma)
        W = 0.5 * (G + G.T)  # symmetrize
        return W
    
    def _self_tuning_affinity(self, X: np.ndarray) -> csr_matrix:
        """Compute self-tuning affinity (Zelnik-Manor & Perona, 2004).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        W : scipy.sparse.csr_matrix
            Self-tuning affinity matrix.
        """
        # Zelnik-Manor & Perona (2004)
        G = kneighbors_graph(X, self.n_neighbors, mode="distance", include_self=True)
        dists = G.toarray()  # small n ok; for large n, implement sparse loop
        # local scales sigma_i = distance to k-th neighbor (last in each row after sorting nonzeros)
        # here we’ll take the max along row ignoring zeros on diagonal
        sigma = np.partition(dists, self.n_neighbors, axis=1)[:, self.n_neighbors]
        sigma[sigma == 0] = np.median(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0
        n = X.shape[0]
        W = np.zeros((n, n))
        # build weights only on kNN edges for efficiency
        idx = (dists > 0)
        S = sigma[:, None] * sigma[None, :]
        W[idx] = np.exp(-(dists[idx]**2) / S[idx])
        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0.0)
        return csr_matrix(W)
    
    def _affinity(self, X: np.ndarray) -> csr_matrix:
        """Dispatch to the selected affinity computation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        W : scipy.sparse.csr_matrix
            Affinity matrix.
        """
        if self.affinity == "rbf":
            return self._rbf_affinity(X)
        elif self.affinity == "knn":
            return self._knn_affinity(X)
        elif self.affinity == "self_tuning":
            return self._self_tuning_affinity(X)
        else:
            raise ValueError("Unknown affinity")

    def _smallest_k(self, L, k):
        """Compute the k smallest eigenpairs for a symmetric PSD matrix.

        Parameters
        ----------
        L : array-like or sparse matrix
            Graph Laplacian.
        k : int
            Number of eigenpairs to compute.

        Returns
        -------
        vals : ndarray
            Eigenvalues.
        vecs : ndarray
            Eigenvectors.
        """
        try:
            vals, vecs = eigsh(L, k=k, which="SM", tol=self.eig_tol, maxiter=self.eig_maxiter)
            self.converged_ = True
            return vals, vecs
        except ArpackNoConvergence as e:
            # try to salvage partial eigenspace
            self.converged_ = False
            ev = getattr(e, "eigenvectors", None)
            ew = getattr(e, "eigenvalues", None)
            if ev is not None and ev.shape[1] >= k:
                return (ew[:k] if ew is not None else np.full(k, np.nan)), ev[:, :k]
            # dense fallback (only if not too big)
            Ld = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
            self.used_dense_fallback_ = True
            vals, vecs = eigh(Ld)
            return vals[:k], vecs[:, :k]
        
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
        X = np.asarray(X)
        Xp = StandardScaler().fit_transform(X) if self.scale else X
        W = self._affinity(Xp)
        self.affinity_ = W
        d = np.asarray(W.sum(axis=1)).ravel()
        n = W.shape[0]

        if self.normalized:
            dinvsqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
            Dinv = diags(dinvsqrt)
            Lsym = csr_matrix(np.eye(n)) - (Dinv @ W @ Dinv)
            vals, vecs = self._smallest_k(Lsym, self.n_clusters)
            order = np.argsort(vals); vals, vecs = vals[order], vecs[:, order]
            U = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        else:
            L = csgraph.laplacian(W, normed=False)
            vals, vecs = self._smallest_k(L, self.n_clusters + 1)
            order = np.argsort(vals); vals, vecs = vals[order], vecs[:, order]
            U = vecs[:, 1:self.n_clusters+1]

        self.evals_ = vals
        self.embedding_ = U
        km = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, random_state=self.random_state)
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