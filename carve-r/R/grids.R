# Default estimator grids. Mirrors _grids.py.

# RBF gammas around the local scale: sigma is the median distance to the
# (n_neighbors - 1)-th nearest other sample, and gamma = 1 / (2 (m sigma)^2)
# for each multiplier m.
estimate_knn_gamma <- function(X, n_neighbors = 7L, multipliers = c(0.5, 1, 2)) {
  kth <- kth_distance(X, n_neighbors)
  sigma <- stats::median(kth)
  if (sigma <= 0) {
    sigma <- if (any(kth > 0)) mean(kth[kth > 0]) else 1
  }
  1 / (2 * (multipliers * sigma)^2)
}

default_estimator_grids <- function(X, preset, sweep) {
  switch(sweep@param,
    n_clusters = default_k_grids(X, sweep@values, preset),
    resolution = default_resolution_grids(sweep@values, preset),
    min_cluster_size = default_min_cluster_size_grids(sweep@values, preset),
    stop(sprintf(
      "No default estimator grids for sweep parameter %s. Pass estimator_param_grids=... explicitly.",
      format_repr(sweep@param)
    ), call. = FALSE)
  )
}

default_k_grids <- function(X, n_clusters, preset) {
  grids <- list(
    estimator_grid(KMeans, n_clusters = n_clusters),
    estimator_grid(AgglomerativeClustering, n_clusters = n_clusters, linkage = "ward"),
    estimator_grid(SpectralClustering, n_clusters = n_clusters, affinity = "self_tuning")
  )
  if (identical(preset, "full")) {
    grids <- c(grids, list(
      estimator_grid(
        AgglomerativeClustering,
        n_clusters = n_clusters,
        linkage = c("average", "single", "complete")
      ),
      estimator_grid(
        SpectralClustering,
        n_clusters = n_clusters,
        affinity = "rbf",
        gamma = estimate_knn_gamma(X)
      )
    ))
  }
  grids
}

# Graph communities swept over resolution on a 15-neighbor graph; the full
# preset adds 10- and 30-neighbor graphs.
default_resolution_grids <- function(resolutions, preset) {
  resolutions <- as.numeric(resolutions)
  grids <- list(
    estimator_grid(LeidenClustering, resolution = resolutions, n_neighbors = 15L),
    estimator_grid(LouvainClustering, resolution = resolutions, n_neighbors = 15L)
  )
  if (identical(preset, "full")) {
    grids <- c(grids, list(
      estimator_grid(LeidenClustering, resolution = resolutions, n_neighbors = c(10L, 30L)),
      estimator_grid(LouvainClustering, resolution = resolutions, n_neighbors = c(10L, 30L))
    ))
  }
  grids
}

# HDBSCAN swept over min_cluster_size with excess-of-mass selection; the full
# preset adds leaf selection.
default_min_cluster_size_grids <- function(sizes, preset) {
  require_package("dbscan", "HDBSCAN")
  sizes <- as.integer(sizes)
  grids <- list(
    estimator_grid(HDBSCAN, min_cluster_size = sizes, cluster_selection_method = "eom")
  )
  if (identical(preset, "full")) {
    grids <- c(grids, list(
      estimator_grid(HDBSCAN, min_cluster_size = sizes, cluster_selection_method = "leaf")
    ))
  }
  grids
}
