test_that("resolve_mode maps each mode to the stages it runs", {
  expect_identical(
    resolve_mode("default"),
    list(mode = "default", run_stability = TRUE, run_generalizability = TRUE, compute_average_ari = TRUE)
  )
  expect_identical(
    resolve_mode("stability"),
    list(mode = "stability", run_stability = TRUE, run_generalizability = FALSE, compute_average_ari = FALSE)
  )
  expect_identical(
    resolve_mode("generalizability"),
    list(mode = "generalizability", run_stability = FALSE, run_generalizability = TRUE, compute_average_ari = FALSE)
  )
  expect_error(
    resolve_mode("both"),
    "Unknown mode: 'both'. Expected one of 'default', 'stability', 'generalizability'.",
    fixed = TRUE
  )
})

test_that("estimator_grid records the function, its name and the grid", {
  toy <- function(X, n_clusters = 2L, linkage = "ward") rep(1L, nrow(X))
  g <- estimator_grid(toy, n_clusters = 2:4, linkage = c("ward", "average"))
  expect_s3_class(g, "carve_estimator_grid")
  expect_identical(g$name, "toy")
  expect_identical(g$estimator, toy)
  expect_identical(g$grid, list(n_clusters = 2:4, linkage = c("ward", "average")))
  expect_identical(estimator_grid(toy, linkage = factor("ward"))$grid$linkage, "ward")
})

test_that("estimator_grid takes the name after :: and accepts name=", {
  expect_identical(estimator_grid(stats::median)$name, "median")
  expect_identical(estimator_grid(function(X) NULL, name = "anon")$name, "anon")
})

test_that("estimator_grid rejects what it cannot use", {
  toy <- function(X, n_clusters = 2L) NULL
  expect_error(estimator_grid("KMeans"), "estimator must be a function.", fixed = TRUE)
  expect_error(
    estimator_grid(function(X) NULL),
    "Pass name= for an estimator that is not a named function.",
    fixed = TRUE
  )
  expect_error(
    estimator_grid(toy, 2:4),
    "Every grid entry must be named after an argument of the estimator.",
    fixed = TRUE
  )
  expect_error(estimator_grid(toy, k = 2:4), "toy has no argument 'k'.", fixed = TRUE)
  expect_error(
    estimator_grid(toy, n_clusters = integer()),
    "Every grid entry needs at least one value.",
    fixed = TRUE
  )
})

test_that("printing a grid lists its values", {
  toy <- function(X, n_clusters = 2L, linkage = "ward") NULL
  expect_output(
    print(estimator_grid(toy, n_clusters = 2:3, linkage = "ward")),
    "Estimator grid for toy\n  n_clusters: 2, 3\n  linkage: ward",
    fixed = TRUE
  )
})

test_that("expand_param_grid follows sklearn's ParameterGrid order", {
  fx <- read_fixture("sweep")
  expect_identical(expand_param_grid(fx$grid), fx$grid_order)
  expect_identical(expand_param_grid(list()), list(list()))
})
