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

# Candidate values of the default reduction grids, before they are filtered
# to what the smallest subsample supports.
PCA_COMPONENTS <- c(2L, 5L, 10L, 20L, 50L)
TSNE_PERPLEXITIES <- c(15, 30, 50)
UMAP_NEIGHBORS <- c(15L, 30L)

# Identity and standardization always; log1p only for data without negative
# values, since it is NaN below -1 and meant for counts.
default_normalization_options <- function(X) {
  options <- list(preprocessing_option(Identity), preprocessing_option(StandardScaler))
  x_min <- min(X)
  if (x_min >= 0) {
    return(c(options, list(preprocessing_option(Log1p))))
  }
  warning(sprintf(
    "X has negative values (minimum %s), so log1p is omitted from the default normalization options.",
    sprintf("%.3g", x_min)
  ), call. = FALSE)
  options
}

warn_omitted <- function(name, param, limit) {
  warning(sprintf(
    "%s is omitted from the default dimensionality reduction options: no candidate %s is below %s, the limit the smallest subsample sets.",
    name, param, format_param_value(limit)
  ), call. = FALSE)
}

# Identity, PCA, t-SNE and, with uwot, UMAP, over discrete grids filtered to
# the smaller of a resample's training subsample and held-out set, the
# smallest set a pipeline is fit on. Rtsne needs 3 * perplexity <= n - 1,
# stricter than scikit-learn's perplexity < n, so the t-SNE limit is
# floor((n_min - 1) / 3) + 1.
default_dim_reduction_options <- function(X, subsample_ratio = 0.6) {
  n_samples <- nrow(X)
  n_train <- floor(subsample_ratio * n_samples)
  n_min <- min(n_train, n_samples - n_train)
  options <- list(preprocessing_option(Identity))

  pca_limit <- min(n_min, ncol(X))
  components <- PCA_COMPONENTS[PCA_COMPONENTS < pca_limit]
  if (length(components) > 0L) {
    options <- c(options, list(preprocessing_option(PCA, n_components = components)))
  } else {
    warn_omitted("PCA", "n_components", pca_limit)
  }

  tsne_limit <- floor((n_min - 1) / 3) + 1
  perplexities <- TSNE_PERPLEXITIES[TSNE_PERPLEXITIES < tsne_limit]
  if (length(perplexities) > 0L) {
    options <- c(options, list(preprocessing_option(TSNE, n_components = 2L, perplexity = perplexities)))
  } else {
    warn_omitted("TSNE", "perplexity", tsne_limit)
  }

  if (!has_package("uwot")) {
    warning(
      'uwot is not installed; UMAP is omitted from the default dimensionality reduction options. Install it with install.packages("uwot").',
      call. = FALSE
    )
    return(options)
  }
  neighbors <- UMAP_NEIGHBORS[UMAP_NEIGHBORS < n_min]
  if (length(neighbors) == 0L) {
    warn_omitted("UMAP", "n_neighbors", n_min)
    return(options)
  }
  c(options, list(preprocessing_option(UMAP, n_components = 2L, n_neighbors = neighbors, min_dist = 0.1)))
}
