# Default estimator and preprocessing grids. Mirrors _grids.py.
#
# Grid-spec form in R:
#
#   list(type = "<backend>", grid = list(param = values, ...))
#
# `type` is forwarded to `cluster_labels`. `grid` is a named list of
# parameter *values*, each either a scalar or a vector; `expand.grid`
# materializes one config per combination.
#
# Preprocessor-spec form (normalization / dimensionality reduction):
#
#   list(type = "<kind>", grid = list(param = values), name = "<display>")
#
# Currently supported types: "identity", "standardize", "log1p", "pca".


#' Estimate RBF gamma values from a k-NN median heuristic
#'
#' Mirrors `_grids.estimate_knn_gamma`. Computes the median k-th neighbor
#' distance and returns `gamma = 1 / (2 * (m * sigma)^2)` for each `m` in
#' `multipliers`.
#'
#' @param X Numeric matrix of shape `(n_samples, n_features)`.
#' @param n_neighbors Integer; number of neighbors (default `7`).
#' @param multipliers Numeric vector of scale multipliers applied to sigma.
#'
#' @return Numeric vector of gamma values.
#'
#' @export
estimate_knn_gamma <- function(X, n_neighbors = 7L,
                               multipliers = c(0.5, 1.0, 2.0)) {
  X <- ensure_2d_matrix(X)
  k_excl <- max(1L, as.integer(n_neighbors) - 1L)
  knn <- FNN::get.knn(X, k = k_excl, algorithm = "kd_tree")
  kth <- knn$nn.dist[, k_excl]

  sigma <- stats::median(kth)
  if (sigma <= 0) {
    pos <- kth[kth > 0]
    sigma <- if (length(pos) > 0L) mean(pos) else 1
  }
  as.numeric(1 / (2 * (multipliers * sigma)^2))
}


#' Default estimator grids
#'
#' Mirrors `_grids.default_estimator_grids`. Returns a list of grid specs
#' suitable for iteration in [run_validation()].
#'
#' @param X Numeric matrix used to derive data-driven hyperparameters
#'   (currently unused for `"light"`).
#' @param n_clusters Integer vector or scalar; candidate cluster counts.
#' @param preset `"light"` (KMeans + Ward agglomerative + self-tuning
#'   spectral) or `"full"` (adds average/single/complete linkage and
#'   RBF-kernel spectral with data-driven `gamma`).
#'
#' @return A list of grid specs.
#'
#' @export
default_estimator_grids <- function(X, n_clusters = 10L,
                                    preset = c("light", "full")) {
  preset <- match.arg(preset)
  ks <- coerce_n_clusters(n_clusters)

  grids <- list(
    list(type = "kmeans", grid = list(n_clusters = ks)),
    list(type = "agglomerative",
         grid = list(n_clusters = ks, linkage = "ward")),
    list(type = "spectral",
         grid = list(n_clusters = ks, affinity = "self_tuning"))
  )

  if (identical(preset, "full")) {
    grids <- c(
      grids,
      list(
        list(type = "agglomerative",
             grid = list(n_clusters = ks,
                         linkage = c("average", "single", "complete"))),
        list(type = "spectral",
             grid = list(n_clusters = ks, affinity = "rbf",
                         gamma = estimate_knn_gamma(X)))
      )
    )
  }

  grids
}


#' Default normalization options
#'
#' Mirrors `_grids.default_normalization_options`. Returns identity,
#' standardization, and `log1p`.
#'
#' @return A list of preprocessor specs.
#'
#' @export
default_normalization_options <- function() {
  list(
    list(type = "identity", grid = list(), name = "Identity"),
    list(type = "standardize", grid = list(), name = "StandardScaler"),
    list(type = "log1p", grid = list(), name = "Log1p")
  )
}


#' Default dimensionality reduction options
#'
#' Mirrors `_grids.default_dim_reduction_options` but only supports
#' identity and PCA in this phase; t-SNE and UMAP are reserved for a
#' later milestone. The PCA grid spans `2:min(min_n, p)` components,
#' where `min_n = round(n * (1 - subsample_ratio)) - 1` is the smallest
#' subsample size CARVE will use downstream.
#'
#' @param X Numeric matrix.
#' @param subsample_ratio Numeric in `(0, 1)`.
#'
#' @return A list of preprocessor specs.
#'
#' @export
default_dim_reduction_options <- function(X, subsample_ratio = 0.6) {
  X <- ensure_2d_matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  min_n <- as.integer(round(n * (1 - subsample_ratio))) - 1L
  upper <- min(min_n, p)
  components <- if (upper >= 2L) seq.int(2L, upper) else integer(0L)

  opts <- list(
    list(type = "identity", grid = list(), name = "Identity")
  )
  if (length(components) > 0L) {
    opts <- c(opts, list(
      list(type = "pca",
           grid = list(n_components = components),
           name = "PCA")
    ))
  }
  opts
}


# Expand a grid-spec `grid` into a list of parameter lists. Mirrors
# sklearn's `ParameterGrid`. Returns list() for an empty grid (one empty
# config, matching ParameterGrid semantics).
expand_grid_spec <- function(grid) {
  if (length(grid) == 0L) return(list(list()))
  # Preserve argument order; stringsAsFactors=FALSE keeps character values.
  df <- do.call(
    expand.grid,
    c(grid, list(stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE))
  )
  lapply(seq_len(nrow(df)), function(i) as.list(df[i, , drop = FALSE]))
}
