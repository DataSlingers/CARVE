"""Leiden community detection with an sklearn-compatible interface.

Used by case study notebooks, not by the benchmarking pipeline itself.
"""

import numpy as np

import igraph as ig
import leidenalg

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class LeidenClustering(BaseEstimator, ClusterMixin):
    """Wrapper for Leiden community detection algorithm with sklearn interface.

    Clusters tabular data by building a k-nearest neighbor graph and running Leiden.

    Parameters:
        n_clusters (int): Desired number of clusters (approximate, Leiden may not match exactly).
        n_neighbors (int): Number of neighbors for graph construction (default: 10).
        resolution (float): Resolution parameter for Leiden (higher = more clusters, default: 1.0).
        linkage (str): Linkage method for hierarchical merging when Leiden returns
            more clusters than requested ('ward', 'single', 'complete', 'average').
        random_state (int or None): Random seed.
    """

    def __init__(
        self,
        n_clusters=None,
        n_neighbors=10,
        resolution=1.0,
        linkage="ward",
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.linkage = linkage
        self.random_state = random_state

    def fit(self, X, y=None):
        # Build kNN graph
        knn = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric="euclidean")
        knn.fit(X)
        knn_graph = knn.kneighbors_graph(X, mode="connectivity")
        sources, targets = knn_graph.nonzero()
        edges = list(zip(sources.tolist(), targets.tolist()))

        # Create igraph graph
        g = ig.Graph(n=X.shape[0], edges=edges, directed=False)
        g.simplify()

        # Run Leiden
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            seed=self.random_state,
        )
        self.labels_ = np.array(partition.membership)

        # Merge clusters using hierarchical linkage if Leiden returned too many
        if (
            self.n_clusters is not None
            and len(np.unique(self.labels_)) != self.n_clusters
        ):
            unique_labels = np.unique(self.labels_)
            centroids = np.array(
                [X[self.labels_ == label].mean(axis=0) for label in unique_labels]
            )
            distances = pdist(centroids, metric="euclidean")
            Z = linkage(distances, method=self.linkage)

            cluster_mapping = fcluster(Z, self.n_clusters, criterion="maxclust") - 1
            label_map = {old: new for old, new in zip(unique_labels, cluster_mapping)}
            self.labels_ = np.array([label_map[lab] for lab in self.labels_])

        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
