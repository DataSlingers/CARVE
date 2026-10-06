toy <- function(X, n_clusters = 2L, resolution = 1, linkage = "ward") rep(1L, nrow(X))

test_that("resolve_sweep defaults to n_clusters", {
  s <- resolve_sweep(n_clusters = 4)
  expect_s4_class(s, "SweepSpec")
  expect_identical(s@param, "n_clusters")
  expect_identical(s@values, 2:4)
  expect_true(s@finer_is_larger)
  expect_true(s@fixes_k)
  expect_identical(s@label, "Number of Clusters (k)")
})

test_that("resolution switches the axis and sorts the values", {
  s <- resolve_sweep(resolution = c(1, 0.5))
  expect_identical(s@param, "resolution")
  expect_identical(s@values, c(0.5, 1))
  expect_false(s@fixes_k)
})

test_that("min_cluster_size runs from fine to coarse", {
  s <- resolve_sweep(sweep = "min_cluster_size", sweep_values = c(20, 5, 10))
  expect_identical(s@values, c(5L, 10L, 20L))
  expect_false(s@finer_is_larger)
  expect_identical(sweep_ranks(s), c(2L, 1L, 0L))
})

test_that("sweep ranks match Python", {
  for (case in read_fixture("sweep")$ranks) {
    spec <- switch(case$param,
      n_clusters = resolve_sweep(n_clusters = case$values),
      resolution = resolve_sweep(resolution = case$values),
      min_cluster_size = resolve_sweep(sweep = "min_cluster_size", sweep_values = case$values)
    )
    expect_equal(spec@values, case$values, info = case$param)
    expect_identical(sweep_ranks(spec), as.integer(case$ranks), info = case$param)
  }
})

test_that("resolve_sweep rejects inconsistent arguments", {
  expect_error(
    resolve_sweep(resolution = 0.5, sweep = "n_clusters"),
    "resolution= was given but sweep='n_clusters'. Pass sweep_values= instead, or drop resolution=.",
    fixed = TRUE
  )
  expect_error(
    resolve_sweep(sweep = "resolution"),
    "No values supplied for sweep parameter 'resolution'. Pass sweep_values=, or supply estimator grids that contain it.",
    fixed = TRUE
  )
  expect_error(
    resolve_sweep(sweep = "eps", sweep_values = c(0.1, 0.2)),
    "Unknown sweep parameter 'eps'. Pass finer_is_larger= to declare whether larger values yield more clusters.",
    fixed = TRUE
  )
})

test_that("an unknown parameter gets a title-case label and the given direction", {
  s <- resolve_sweep(sweep = "max_eps", sweep_values = c(0.2, 0.1), finer_is_larger = FALSE)
  expect_identical(s@label, "Max Eps")
  expect_false(s@finer_is_larger)
  expect_false(s@fixes_k)
})

test_that("finer_is_larger overrides the registry", {
  expect_false(resolve_sweep(resolution = 1, finer_is_larger = FALSE)@finer_is_larger)
})

test_that("a single value passed as sweep_values is not expanded", {
  expect_identical(resolve_sweep(n_clusters = 2:10, sweep_values = 3L)@values, 3L)
  expect_identical(resolve_sweep(n_clusters = 3)@values, 2:3)
})

test_that("coerce_sweep_values validates each axis", {
  expect_error(coerce_sweep_values(c(0, 1), "resolution"), "All resolution values must be > 0.", fixed = TRUE)
  expect_error(coerce_sweep_values(c(2.5, 3), "min_cluster_size"), "min_cluster_size values must be integers.", fixed = TRUE)
  expect_error(coerce_sweep_values(c(1, 3), "min_cluster_size"), "All min_cluster_size values must be >= 2.", fixed = TRUE)
  expect_error(coerce_sweep_values(matrix(1:4, 2), "resolution"), "resolution must be a numeric vector.", fixed = TRUE)
  expect_identical(coerce_sweep_values(c(20, 5), "min_cluster_size"), c(5L, 20L))
})

test_that("sweep_rank_of matches values up to rounding and rejects others", {
  s <- resolve_sweep(resolution = c(0.1, 0.2, 0.3))
  expect_identical(sweep_rank_of(s, 0.1 + 0.2), 2L)
  expect_error(
    sweep_rank_of(s, 0.5),
    "0.5 is not among the swept resolution values [0.1, 0.2, 0.3].",
    fixed = TRUE
  )
})

test_that("sweep_rank_of returns the rank, not the position, on a descending axis", {
  s <- resolve_sweep(sweep = "min_cluster_size", sweep_values = c(5, 10, 20))
  expect_identical(sweep_rank_of(s, 10), 1L)
  expect_identical(sweep_rank_of(s, 20), 0L)
  expect_identical(sweep_rank_of(s, 5), 2L)
})

test_that("infer_sweep_param finds the one swept registry parameter", {
  expect_identical(infer_sweep_param(list(estimator_grid(toy, n_clusters = 2:4, linkage = "ward"))), "n_clusters")
  expect_null(infer_sweep_param(list(estimator_grid(toy, n_clusters = 3L))))
  expect_error(
    infer_sweep_param(list(estimator_grid(toy, n_clusters = 2:3), estimator_grid(toy, resolution = c(0.5, 1)))),
    "Estimator grids sweep more than one parameter (['n_clusters', 'resolution']). CARVE evaluates exactly one sweep axis per run; k-based and resolution-based estimators cannot be compared in the same run.",
    fixed = TRUE
  )
})

test_that("grid_sweep_values reads the first grid that has the parameter", {
  grids <- list(estimator_grid(toy, linkage = "ward"), estimator_grid(toy, n_clusters = 2:3))
  expect_identical(grid_sweep_values(grids, "n_clusters"), 2:3)
  expect_null(grid_sweep_values(grids, "resolution"))
})

test_that("validate_grids checks every grid against the axis", {
  s <- resolve_sweep(n_clusters = 2:3)
  expect_identical(validate_grids(list(estimator_grid(toy, n_clusters = 2:3)), s), 2:3)
  expect_identical(validate_grids(list(estimator_grid(toy, n_clusters = 3L)), s), 3L)
  expect_error(validate_grids(list(), s), "estimator_param_grids must not be empty.", fixed = TRUE)
  expect_error(
    validate_grids(list(estimator_grid(toy, linkage = "ward")), s),
    "Estimator grid for toy does not contain the sweep parameter 'n_clusters'. All grids in a run must sweep the same parameter.",
    fixed = TRUE
  )
  expect_error(
    validate_grids(list(estimator_grid(toy, n_clusters = 2:3, resolution = c(0.5, 1))), s),
    "Estimator grid for toy mixes sweep parameters 'n_clusters' and ['resolution']. CARVE evaluates exactly one sweep axis per run; k-based and resolution-based estimators cannot be compared in the same run.",
    fixed = TRUE
  )
  expect_error(
    validate_grids(list(estimator_grid(toy, n_clusters = 2:3), estimator_grid(toy, n_clusters = 2:4)), s),
    "All estimator parameter grids must contain the same n_clusters values.",
    fixed = TRUE
  )
})

test_that("format_method_label matches Python", {
  for (case in read_fixture("sweep")$labels) {
    expect_identical(format_method_label(case$estimator, case$params, case$sweep_param), case$label)
  }
})

test_that("method ids are shared along the sweep and numbered in first-seen order", {
  assign_id <- method_id_assigner("n_clusters")
  expect_identical(assign_id("KMeans", list(n_clusters = 2L)), c("m0", "KMeans"))
  expect_identical(assign_id("KMeans", list(n_clusters = 3L)), c("m0", "KMeans"))
  expect_identical(
    assign_id("AgglomerativeClustering", list(linkage = "ward", n_clusters = 2L)),
    c("m1", "AgglomerativeClustering, linkage=ward")
  )
  expect_identical(
    assign_id("AgglomerativeClustering", list(linkage = "average", n_clusters = 2L)),
    c("m2", "AgglomerativeClustering, linkage=average")
  )
  expect_identical(assign_id("KMeans", list(n_clusters = 4L)), c("m0", "KMeans"))
})

test_that("observed_k rounds half to even, like Python", {
  expect_identical(observed_k(data.frame(n_clusters_observed = 2.5)), 2L)
  expect_identical(observed_k(data.frame(n_clusters_observed = 3.5)), 4L)
  expect_identical(observed_k_series(data.frame(n_clusters_observed = c(2.4, 2.6))), c(2, 3))
})

test_that("the results-table helpers name the bookkeeping columns and the axis", {
  df <- data.frame(sweep_param = "resolution", config_id = 4L)
  expect_setequal(
    sweep_exclude_cols(df),
    c("sweep_param", "sweep_value", "sweep_rank", "n_clusters_observed", "n_clusters_observed_se",
      "noise_fraction", "config_id", "method_id", "method_label", "resolution")
  )
  expect_identical(sweep_param_name(df), "resolution")
  expect_identical(sweep_axis_label(df), "Resolution")
  expect_identical(sweep_axis_label(data.frame(sweep_param = "max_eps")), "Max Eps")
  expect_identical(config_id_of(df), 4L)
})

test_that("show prints the axis", {
  expect_output(
    show(resolve_sweep(n_clusters = 3)),
    "SweepSpec: n_clusters over 2, 3 (larger values give more clusters)",
    fixed = TRUE
  )
})
