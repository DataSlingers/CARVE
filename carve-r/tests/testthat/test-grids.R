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
