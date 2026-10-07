test_that("estimate_knn_gamma matches Python", {
  fx <- read_fixture("spectral")
  expect_equal(estimate_knn_gamma(fixture_matrix(fx$X)), fixture_vector(fx$knn_gamma), tolerance = 1e-10)
})

test_that("the light preset holds KMeans, Ward and self-tuning spectral clustering", {
  X <- make_blobs()$X
  grids <- default_estimator_grids(X, "light", resolve_sweep(n_clusters = 4))
  expect_identical(
    vapply(grids, function(g) g$name, character(1)),
    c("KMeans", "AgglomerativeClustering", "SpectralClustering")
  )
  expect_identical(grids[[1]]$estimator, KMeans)
  expect_identical(grids[[1]]$grid, list(n_clusters = 2:4))
  expect_identical(grids[[2]]$grid, list(n_clusters = 2:4, linkage = "ward"))
  expect_identical(grids[[3]]$grid, list(n_clusters = 2:4, affinity = "self_tuning"))
})

test_that("the full preset adds other linkages and RBF spectral clustering", {
  X <- make_blobs()$X
  grids <- default_estimator_grids(X, "full", resolve_sweep(n_clusters = 4))
  expect_length(grids, 5L)
  expect_identical(grids[[4]]$grid$linkage, c("average", "single", "complete"))
  expect_identical(grids[[5]]$grid$affinity, "rbf")
  expect_identical(grids[[5]]$grid$gamma, estimate_knn_gamma(X))
})

test_that("an axis without default grids is an error", {
  sweep <- resolve_sweep(sweep = "max_eps", sweep_values = 0.1, finer_is_larger = FALSE)
  expect_error(
    default_estimator_grids(make_blobs()$X, "light", sweep),
    "No default estimator grids for sweep parameter 'max_eps'. Pass estimator_param_grids=... explicitly.",
    fixed = TRUE
  )
})

test_that("the light resolution preset runs Leiden and Louvain on 15 neighbors", {
  grids <- default_estimator_grids(matrix(0, 10, 2), "light", resolve_sweep(resolution = c(1, 0.5)))
  expect_identical(vapply(grids, function(g) g$name, character(1)), c("LeidenClustering", "LouvainClustering"))
  expect_identical(grids[[1L]]$estimator, LeidenClustering)
  expect_identical(grids[[2L]]$estimator, LouvainClustering)
  for (g in grids) {
    expect_identical(g$grid$resolution, c(0.5, 1))
    expect_identical(g$grid$n_neighbors, 15L)
  }
})

test_that("the full resolution preset adds 10 and 30 neighbors", {
  grids <- default_estimator_grids(matrix(0, 10, 2), "full", resolve_sweep(resolution = c(0.5, 1)))
  expect_identical(
    vapply(grids, function(g) g$name, character(1)),
    c("LeidenClustering", "LouvainClustering", "LeidenClustering", "LouvainClustering")
  )
  expect_identical(grids[[3L]]$grid$n_neighbors, c(10L, 30L))
  expect_identical(grids[[4L]]$grid$n_neighbors, c(10L, 30L))
  expect_identical(sum(vapply(grids, function(g) length(expand_param_grid(g$grid)), integer(1))), 12L)
})

test_that("the min_cluster_size presets run HDBSCAN with eom, and leaf in full", {
  skip_if_not_installed("dbscan")
  sweep <- resolve_sweep(sweep = "min_cluster_size", sweep_values = c(10, 5))
  light <- default_estimator_grids(matrix(0, 10, 2), "light", sweep)
  expect_length(light, 1L)
  expect_identical(light[[1L]]$name, "HDBSCAN")
  expect_identical(light[[1L]]$grid$min_cluster_size, c(5L, 10L))
  expect_identical(light[[1L]]$grid$cluster_selection_method, "eom")
  expect_false("n_clusters" %in% names(light[[1L]]$grid))
  full <- default_estimator_grids(matrix(0, 10, 2), "full", sweep)
  expect_identical(vapply(full, function(g) g$grid$cluster_selection_method, character(1)), c("eom", "leaf"))
})

test_that("the min_cluster_size presets need dbscan", {
  local_mocked_bindings(has_package = function(package) FALSE)
  sweep <- resolve_sweep(sweep = "min_cluster_size", sweep_values = c(5, 10))
  expect_error(
    default_estimator_grids(matrix(0, 10, 2), "light", sweep),
    "dbscan is required for HDBSCAN.",
    fixed = TRUE
  )
})

test_that("a sweep without default grids is an error", {
  sweep <- resolve_sweep(sweep = "eps", sweep_values = c(0.1, 0.2), finer_is_larger = FALSE)
  expect_error(
    default_estimator_grids(matrix(0, 10, 2), "light", sweep),
    "No default estimator grids for sweep parameter 'eps'. Pass estimator_param_grids=... explicitly.",
    fixed = TRUE
  )
})

option_names <- function(options) vapply(options, function(o) o$name, character(1))

test_that("data without negative values get log1p among the normalizations", {
  expect_identical(option_names(default_normalization_options(matrix(c(0, 1, 2, 3), 2))), c("identity", "StandardScaler", "log1p"))
})

test_that("negative data leave log1p out with a warning", {
  options <- NULL
  expect_warning(
    options <- default_normalization_options(matrix(c(-1.5, 1, 2, 3), 2)),
    "X has negative values (minimum -1.5), so log1p is omitted from the default normalization options.",
    fixed = TRUE
  )
  expect_identical(option_names(options), c("identity", "StandardScaler"))
})

test_that("the reduction grids are filtered to the smallest subsample", {
  skip_if_not_installed("uwot")
  # 1,000 samples at ratio 0.618: subsamples of 618 and 382, so every
  # candidate fits.
  options <- default_dim_reduction_options(matrix(0, 1000, 100), 0.618)
  expect_identical(option_names(options), c("identity", "PCA", "TSNE", "UMAP"))
  expect_identical(options[[2L]]$grid$n_components, c(2L, 5L, 10L, 20L, 50L))
  expect_identical(options[[3L]]$grid, list(n_components = 2L, perplexity = c(15, 30, 50)))
  expect_identical(options[[4L]]$grid, list(n_components = 2L, n_neighbors = c(15L, 30L), min_dist = 0.1))
})

test_that("the limits come from the smaller subsample at low ratios", {
  skip_if_not_installed("uwot")
  # 200 samples at ratio 0.75: subsamples of 150 and 50. The held-out set
  # sets every limit: PCA below 50, perplexities up to (50 - 1) / 3.
  options <- default_dim_reduction_options(matrix(0, 200, 100), 0.75)
  expect_identical(options[[2L]]$grid$n_components, c(2L, 5L, 10L, 20L))
  expect_identical(options[[3L]]$grid$perplexity, 15)
  expect_identical(options[[4L]]$grid$n_neighbors, c(15L, 30L))
})

test_that("an option whose grid filters to nothing is left out with a warning", {
  skip_if_not_installed("uwot")
  out <- collect_warnings(default_dim_reduction_options(matrix(0, 30, 2), 0.618))
  # 30 samples: subsamples of 18 and 12.
  expect_identical(option_names(out$value), "identity")
  expect_identical(out$warnings, c(
    "PCA is omitted from the default dimensionality reduction options: no candidate n_components is below 2, the limit the smallest subsample sets.",
    "TSNE is omitted from the default dimensionality reduction options: no candidate perplexity is below 4, the limit the smallest subsample sets.",
    "UMAP is omitted from the default dimensionality reduction options: no candidate n_neighbors is below 12, the limit the smallest subsample sets."
  ))
})

test_that("without uwot, UMAP is left out with a warning", {
  local_mocked_bindings(has_package = function(package) package != "uwot")
  options <- NULL
  expect_warning(
    options <- default_dim_reduction_options(matrix(0, 1000, 100), 0.618),
    'uwot is not installed; UMAP is omitted from the default dimensionality reduction options. Install it with install.packages("uwot").',
    fixed = TRUE
  )
  expect_identical(option_names(options), c("identity", "PCA", "TSNE"))
})
