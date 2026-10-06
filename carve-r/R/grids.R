# Default estimator grids. Mirrors _grids.py for the k axis; stage 2 adds
# the resolution and min_cluster_size presets.

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
  if (sweep@param == "n_clusters") {
    return(default_k_grids(X, sweep@values, preset))
  }
  stop(sprintf(
    "No default estimator grids for sweep parameter %s. Pass estimator_param_grids=... explicitly.",
    format_repr(sweep@param)
  ), call. = FALSE)
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
