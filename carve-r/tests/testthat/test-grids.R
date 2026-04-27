make_X <- function(n = 100L, p = 5L, seed = 1L) {
  set.seed(seed)
  matrix(stats::rnorm(n * p), n, p)
}

test_that("default_estimator_grids 'light' returns 3 backends", {
  X <- make_X()
  grids <- default_estimator_grids(X, n_clusters = 2:4, preset = "light")
  expect_length(grids, 3L)
  types <- vapply(grids, function(g) g$type, character(1L))
  expect_setequal(types, c("kmeans", "agglomerative", "spectral"))
  for (g in grids) {
    expect_equal(g$grid$n_clusters, 2:4)
  }
})

test_that("default_estimator_grids 'full' adds linkage variants and RBF spectral", {
  X <- make_X()
  grids <- default_estimator_grids(X, n_clusters = 2:3, preset = "full")
  expect_length(grids, 5L)
  types <- vapply(grids, function(g) g$type, character(1L))
  expect_equal(sum(types == "agglomerative"), 2L)
  expect_equal(sum(types == "spectral"), 2L)
  rbf <- Filter(function(g) g$type == "spectral" &&
                 identical(g$grid$affinity, "rbf"), grids)
  expect_length(rbf, 1L)
  expect_length(rbf[[1]]$grid$gamma, 3L)
})

test_that("expand_grid_spec expands the Cartesian product", {
  configs <- expand_grid_spec(list(n_clusters = 2:3, linkage = c("ward", "avg")))
  expect_length(configs, 4L)
  expect_equal(configs[[1]]$n_clusters, 2)
  expect_equal(configs[[1]]$linkage, "ward")
})

test_that("expand_grid_spec of an empty grid yields one empty config", {
  configs <- expand_grid_spec(list())
  expect_length(configs, 1L)
  expect_length(configs[[1]], 0L)
})

test_that("default_normalization_options returns identity + standardize + log1p", {
  opts <- default_normalization_options()
  expect_length(opts, 3L)
  types <- vapply(opts, function(o) o$type, character(1L))
  expect_setequal(types, c("identity", "standardize", "log1p"))
})

test_that("default_dim_reduction_options constrains components to the subsample size", {
  X <- make_X(n = 50L, p = 10L)
  opts <- default_dim_reduction_options(X, subsample_ratio = 0.6)
  # min_n = round(50 * 0.4) - 1 = 19; min(19, 10) = 10 → components 2:10.
  pca <- Filter(function(o) o$type == "pca", opts)[[1]]
  expect_equal(pca$grid$n_components, 2:10)
})

test_that("estimate_knn_gamma returns positive gammas for each multiplier", {
  X <- make_X(n = 50L, p = 5L)
  g <- estimate_knn_gamma(X, n_neighbors = 7L, multipliers = c(0.5, 1, 2))
  expect_length(g, 3L)
  expect_true(all(g > 0))
  # Larger multiplier → smaller gamma (gamma ∝ 1 / m^2).
  expect_true(g[1] > g[2] && g[2] > g[3])
})
