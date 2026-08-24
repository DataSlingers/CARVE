"""Custom spectral clustering implementation for CARVE."""

import importlib

import numpy as np
from typing import Literal
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse import coo_matrix, csr_matrix, diags, triu
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


class SpectralClustering(BaseEstimator, ClusterMixin):
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
            sigma = (
                float(np.mean(kth_dists[kth_dists > 0]))
                if np.any(kth_dists > 0)
                else 1.0
            )
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
                sigma = (
                    float(np.mean(kth_dists[kth_dists > 0]))
                    if np.any(kth_dists > 0)
                    else 1.0
                )
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
            median_nonzero = (
                float(np.median(sigma[sigma > 0])) if np.any(sigma > 0) else 1.0
            )
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
        self : SpectralClustering
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


# ---------------------------------------------------------------------- #
#  Graph-community clustering                                            #
# ---------------------------------------------------------------------- #


def _require(module: str, package: str):
    """Import an optional dependency or raise an ImportError."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            f"{package} is required for graph-community clustering. "
            'Install it with: pip install "carve-validate[graph]"'
        ) from exc


def build_knn_graph(
    X: np.ndarray,
    *,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    weighting: Literal["connectivity", "jaccard"] = "connectivity",
):
    """Build a symmetric weighted k-nearest-neighbor graph.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_neighbors : int, default=15
        Number of nearest neighbors per point. Clipped to ``n_samples - 1``.
    metric : str, default="euclidean"
        Distance metric passed to ``sklearn.neighbors.kneighbors_graph``.
    weighting : {"connectivity", "jaccard"}, default="connectivity"
        ``"connectivity"`` gives every kNN edge weight 1 after
        symmetrization. ``"jaccard"`` weights each edge by the Jaccard
        index of its endpoints' neighbor sets (a shared-nearest-neighbor
        graph, as used by Seurat), which suppresses spurious edges between
        clusters.

    Returns
    -------
    graph : igraph.Graph
        Undirected graph on ``n_samples`` vertices with edge weights in the
        ``"weight"`` attribute.

    Raises
    ------
    ImportError
        If ``igraph`` is not installed.
    ValueError
        If weighting is not recognized.
    """
    ig = _require("igraph", "igraph")

    n = X.shape[0]
    k = int(min(n_neighbors, max(n - 1, 1)))

    A = kneighbors_graph(
        X, n_neighbors=k, metric=metric, mode="connectivity", include_self=False
    )

    if weighting == "connectivity":
        W = A.maximum(A.T)

    elif weighting == "jaccard":
        shared = (A @ A.T).tocsr()
        shared.setdiag(0)
        shared.eliminate_zeros()
        S = shared.tocoo()
        jaccard = S.data / (2.0 * k - S.data)
        W = coo_matrix((jaccard, (S.row, S.col)), shape=A.shape)
        W = W.multiply(A.maximum(A.T))

    else:
        raise ValueError(
            f"Unknown weighting: {weighting!r}. Expected 'connectivity' or 'jaccard'."
        )

    W = triu(csr_matrix(W), k=1).tocoo()

    graph = ig.Graph(
        n=n,
        edges=list(zip(W.row.tolist(), W.col.tolist())),
        directed=False,
    )
    graph.es["weight"] = W.data.astype(float).tolist()
    return graph


class LeidenClustering(BaseEstimator, ClusterMixin):
    """Leiden community detection on a k-nearest-neighbor graph.

    Unlike k-means or agglomerative clustering, Leiden takes no number of
    clusters. Granularity is controlled by ``resolution``: larger values
    yield more, smaller communities. Use CARVE's resolution sweep mode
    (``CARVE(resolution=...)``) to validate across that axis.

    Parameters
    ----------
    resolution : float, default=1.0
        Resolution parameter of the partition quality function. Larger
        values produce more clusters.
    n_neighbors : int, default=15
        Number of nearest neighbors used to build the graph.
    metric : str, default="euclidean"
        Distance metric for neighbor search.
    weighting : {"connectivity", "jaccard"}, default="connectivity"
        Edge weighting scheme; see :func:`build_knn_graph`.
    objective_function : {"modularity", "cpm"}, default="modularity"
        Partition quality function. ``"modularity"`` uses the RB
        configuration model; ``"cpm"`` uses the constant Potts model.
        The two use different resolution scales and must not be compared
        on a shared resolution grid.
    n_iterations : int, default=-1
        Number of Leiden iterations. ``-1`` iterates until stable.
    random_state : int or None, default=None
        Seed for the Leiden optimizer.
    scale : bool, default=False
        Whether to standardize X before building the graph.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Community assignment per sample.
    n_clusters_ : int
        Number of communities found.
    graph_ : igraph.Graph
        The kNN graph the partition was computed on.
    quality_ : float
        Quality of the final partition.

    Notes
    -----
    Requires ``igraph`` and ``leidenalg``::

        pip install "carve-validate[graph]"

    References
    ----------
    Traag, Waltman & van Eck (2019). From Louvain to Leiden: guaranteeing
    well-connected communities. Scientific Reports 9:5233.

    See Also
    --------
    LouvainClustering : Modularity-based alternative on the same graph.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        weighting: Literal["connectivity", "jaccard"] = "connectivity",
        objective_function: Literal["modularity", "cpm"] = "modularity",
        n_iterations: int = -1,
        random_state: int | None = None,
        scale: bool = False,
    ):
        self.resolution = resolution
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weighting = weighting
        self.objective_function = objective_function
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.scale = scale

    def fit(self, X, y=None):
        """Fit the Leiden model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored (sklearn compatibility).

        Returns
        -------
        self : LeidenClustering
            Fitted estimator.
        """
        la = _require("leidenalg", "leidenalg")

        partition_types = {
            "modularity": "RBConfigurationVertexPartition",
            "cpm": "CPMVertexPartition",
        }
        if self.objective_function not in partition_types:
            raise ValueError(
                f"Unknown objective_function: {self.objective_function!r}. "
                "Expected 'modularity' or 'cpm'."
            )
        partition_type = getattr(la, partition_types[self.objective_function])

        X = np.asarray(X, dtype=np.float64)
        Xp = StandardScaler().fit_transform(X) if self.scale else X

        graph = build_knn_graph(
            Xp,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            weighting=self.weighting,
        )
        self.graph_ = graph

        partition = la.find_partition(
            graph,
            partition_type,
            resolution_parameter=float(self.resolution),
            weights="weight",
            n_iterations=self.n_iterations,
            seed=self.random_state,
        )

        self.labels_ = np.asarray(partition.membership, dtype=np.int32)
        self.n_clusters_ = int(np.unique(self.labels_).size)
        self.quality_ = float(partition.quality())
        return self

    def fit_predict(self, X, y=None):
        """Fit the model and return community labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored (sklearn compatibility).

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Community labels.
        """
        return self.fit(X, y).labels_


class LouvainClustering(BaseEstimator, ClusterMixin):
    """Louvain community detection on a k-nearest-neighbor graph.

    Provided as a baseline alongside :class:`LeidenClustering`. Louvain can
    produce internally disconnected communities, which Leiden fixes; prefer
    Leiden unless you specifically need Louvain for comparison.

    Parameters
    ----------
    resolution : float, default=1.0
        Resolution of the modularity objective. Larger values produce more
        clusters.
    n_neighbors : int, default=15
        Number of nearest neighbors used to build the graph.
    metric : str, default="euclidean"
        Distance metric for neighbor search.
    weighting : {"connectivity", "jaccard"}, default="connectivity"
        Edge weighting scheme; see :func:`build_knn_graph`.
    scale : bool, default=False
        Whether to standardize X before building the graph.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Community assignment per sample.
    n_clusters_ : int
        Number of communities found.
    graph_ : igraph.Graph
        The kNN graph the partition was computed on.
    modularity_ : float
        Modularity of the final partition.

    Notes
    -----
    Requires ``igraph``::

        pip install "carve-validate[graph]"

    ``igraph.Graph.community_multilevel`` takes no seed; results are
    deterministic given the graph, and the graph is deterministic given X.

    References
    ----------
    Blondel, Guillaume, Lambiotte & Lefebvre (2008). Fast unfolding of
    communities in large networks. J. Stat. Mech. P10008.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        weighting: Literal["connectivity", "jaccard"] = "connectivity",
        scale: bool = False,
    ):
        self.resolution = resolution
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weighting = weighting
        self.scale = scale

    def fit(self, X, y=None):
        """Fit the Louvain model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored (sklearn compatibility).

        Returns
        -------
        self : LouvainClustering
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        Xp = StandardScaler().fit_transform(X) if self.scale else X

        graph = build_knn_graph(
            Xp,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            weighting=self.weighting,
        )
        self.graph_ = graph

        partition = graph.community_multilevel(
            weights="weight", resolution=float(self.resolution)
        )

        self.labels_ = np.asarray(partition.membership, dtype=np.int32)
        self.n_clusters_ = int(np.unique(self.labels_).size)
        self.modularity_ = float(
            graph.modularity(partition.membership, weights="weight")
        )
        return self

    def fit_predict(self, X, y=None):
        """Fit the model and return community labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored (sklearn compatibility).

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Community labels.
        """
        return self.fit(X, y).labels_
