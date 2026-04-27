skip_if_not_installed("ranger")

make_blobs <- function(n_per = 50L, seed = 1L) {
  set.seed(seed)
  centers <- rbind(c(0, 0), c(6, 0), c(3, 5))
  X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i) {
    matrix(stats::rnorm(n_per * 2L, 0, 0.4), ncol = 2L) +
      matrix(centers[i, ], n_per, 2L, byrow = TRUE)
  }))
  y <- rep(seq_len(nrow(centers)), each = n_per)
  list(X = X, y = y)
}

test_that("CARVE$new + $fit populates results frame and consensus matrices", {
  bl <- make_blobs()
  fit <- CARVE$new(
    n_clusters = 2:4, n_resamples = 5L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:4))
    ),
    random_state = 1L, verbose = 0L
  )
  fit$fit(bl$X)

  expect_s3_class(fit, "CARVE")
  expect_equal(nrow(fit$estimator_results_), 3L)
  expect_true("consensus_pac_stability" %in% names(fit$estimator_results_))
  expect_true("consensus_gini_stability" %in% names(fit$estimator_results_))
  expect_true("consensus_ce_stability" %in% names(fit$estimator_results_))
  expect_true("accuracy_generalizability" %in% names(fit$estimator_results_))
  expect_length(fit$consensus_matrices_, 3L)
})

test_that("CARVE picks k=3 on 3 well-separated blobs", {
  bl <- make_blobs()
  fit <- CARVE$new(
    n_clusters = 2:5, n_resamples = 10L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:5))
    ),
    random_state = 2L
  )
  fit$fit(bl$X)
  expect_equal(fit$get_k(measure = "stability", rule = "max"), 3L)
})

test_that("$get_labels returns an integer vector of length nrow(X)", {
  bl <- make_blobs()
  fit <- CARVE$new(
    n_clusters = 2:3, n_resamples = 5L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:3))
    ),
    random_state = 1L
  )
  fit$fit(bl$X)
  labels <- fit$get_labels(measure = "stability", rule = "max")
  expect_type(labels, "integer")
  expect_length(labels, nrow(bl$X))
  expect_true(length(unique(labels)) <= 3L)
})

test_that("$save() / CARVE$load() roundtrip preserves $get_labels output", {
  bl <- make_blobs()
  fit <- CARVE$new(
    n_clusters = 2:3, n_resamples = 5L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:3))
    ),
    random_state = 1L
  )
  fit$fit(bl$X)
  labels_before <- fit$get_labels(measure = "stability", rule = "max")

  path <- tempfile(fileext = ".carve")
  fit$save(path)
  on.exit(unlink(path), add = TRUE)

  loaded <- CARVE$load(path)
  expect_s3_class(loaded, "CARVE")
  expect_null(loaded$X_)  # data excluded by default

  labels_after <- loaded$get_labels(measure = "stability", rule = "max")
  expect_identical(labels_before, labels_after)
  expect_equal(loaded$estimator_results_, fit$estimator_results_)
})

test_that("$save(include_data = TRUE) keeps X_", {
  bl <- make_blobs(n_per = 30L)
  fit <- CARVE$new(
    n_clusters = 2:3, n_resamples = 3L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:3))
    ),
    random_state = 1L
  )
  fit$fit(bl$X)
  path <- tempfile(fileext = ".carve")
  fit$save(path, include_data = TRUE)
  on.exit(unlink(path), add = TRUE)
  loaded <- CARVE$load(path)
  expect_equal(loaded$X_, fit$X_)
})

test_that("$save() errors before $fit()", {
  fit <- CARVE$new(n_clusters = 2:3, n_resamples = 3L)
  expect_error(fit$save(tempfile()), "has not been fitted")
})

test_that("CARVE$load errors on missing file", {
  expect_error(CARVE$load(tempfile(fileext = ".nope")), "No such file")
})

test_that("$get_k / $get_labels error before $fit()", {
  fit <- CARVE$new(n_clusters = 2:3, n_resamples = 3L)
  expect_error(fit$get_k(), "has not been fitted")
  expect_error(fit$get_labels(), "has not been fitted")
})

test_that("plot methods require a fitted instance", {
  fit <- CARVE$new(n_clusters = 2:3)
  expect_error(fit$plot_metric_over_n_clusters(), "has not been fitted")
  expect_error(fit$plot_consensus_matrix(), "has not been fitted")
  expect_error(fit$plot_cluster_boxplot(), "has not been fitted")
})

test_that("carve() S3 generic on a matrix returns a fitted CARVE", {
  bl <- make_blobs()
  fit <- carve(
    bl$X, n_clusters = 2:3, n_resamples = 5L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:3))
    ),
    random_state = 1L
  )
  expect_s3_class(fit, "CARVE")
  expect_equal(nrow(fit$estimator_results_), 2L)
})

test_that("carve() on a data.frame coerces numeric columns", {
  bl <- make_blobs()
  df <- as.data.frame(bl$X)
  fit <- carve(
    df, n_clusters = 2:3, n_resamples = 5L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:3))
    ),
    random_state = 1L
  )
  expect_s3_class(fit, "CARVE")
})

test_that("carve() on non-numeric data.frame errors informatively", {
  df <- data.frame(x = 1:3, label = letters[1:3])
  expect_error(carve(df), "all columns must be numeric")
})

test_that("mode='stability' leaves accuracy_generalizability as NaN", {
  bl <- make_blobs()
  fit <- CARVE$new(
    n_clusters = 2:3, n_resamples = 5L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:3))
    ),
    random_state = 1L
  )
  suppressWarnings(fit$fit(bl$X, mode = "stability"))
  expect_true(all(is.nan(fit$estimator_results_$accuracy_generalizability)))
  expect_true(all(is.nan(fit$estimator_results_$ari_generalizability)))
})
