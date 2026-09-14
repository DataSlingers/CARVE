"""Plotting functions that read CARVE results from an :class:`~anndata.AnnData`."""

from ._plots import (
    cluster_boxplot,
    cluster_scatter,
    cluster_violin,
    consensus_matrix,
    diagnostic_scatter,
    metric_by_pipeline,
    metric_over_n_clusters,
)

__all__ = [
    "cluster_boxplot",
    "cluster_scatter",
    "cluster_violin",
    "consensus_matrix",
    "diagnostic_scatter",
    "metric_by_pipeline",
    "metric_over_n_clusters",
]
