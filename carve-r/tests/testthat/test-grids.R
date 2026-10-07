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
