blobs <- make_blobs()

test_that("validation_iter returns one resample's labels, indices and scores", {
  r <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L),
                       subsample_ratio = 0.618, n_resamples = 5L, seed = 0L, random_state = 0L)
  expect_length(r$train_indices, 55L)
  expect_length(r$test_indices, 35L)
  expect_length(r$stability_indices, 55L)
  expect_length(r$labels_train, 55L)
  expect_length(r$labels_predicted, 35L)
  expect_identical(r$ari_stability, 1)
  expect_identical(r$ari_generalizability, 1)
  expect_identical(c(r$n_clusters_train, r$n_clusters_test, r$n_clusters_stability), c(3L, 3L, 3L))
  expect_identical(r$noise_fraction, 0)
})

test_that("the subsamples come from random_state + seed and random_state + seed + n_resamples", {
  r <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L,
                       seed = 2L, random_state = 10L)
  expect_identical(r$train_indices, split_subsample_indices(90L, 0.618, 12L)$train)
  expect_identical(r$test_indices, split_subsample_indices(90L, 0.618, 12L)$test)
  expect_identical(r$stability_indices, split_subsample_indices(90L, 0.618, 17L)$train)
})

test_that("mode = 'stability' skips the held-out prediction", {
  r <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L, 0L,
                       mode = "stability", random_state = 0L)
  expect_null(r$labels_test)
  expect_null(r$labels_predicted)
  expect_true(is.nan(r$ari_generalizability))
  expect_identical(r$n_clusters_test, 0L)
  expect_identical(r$ari_stability, 1)
})

test_that("mode = 'generalizability' skips the second subsample", {
  r <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L, 0L,
                       mode = "generalizability", random_state = 0L)
  expect_null(r$stability_indices)
  expect_null(r$labels_stability)
  expect_true(is.nan(r$ari_stability))
  expect_identical(r$ari_generalizability, 1)
})

test_that("a k-axis fit with the wrong cluster count warns for each subsample", {
  two <- function(X, n_clusters = 2L) rep_len(1:2, nrow(X))
  out <- collect_warnings(validation_iter(blobs$X, two, "two", list(n_clusters = 3L), 0.618, 5L, 0L,
                                          mode = "stability", random_state = 0L))
  expect_identical(
    out$warnings,
    c("labels_1 has 2 clusters, expected 3", "labels_2 has 2 clusters, expected 3")
  )
})

test_that("off the k axis, a single-cluster subsample warns", {
  one <- function(X, resolution = 1) rep(1L, nrow(X))
  out <- collect_warnings(validation_iter(blobs$X, one, "one", list(resolution = 0.5), 0.618, 5L, 0L,
                                          sweep_param = "resolution", mode = "stability", random_state = 0L))
  expect_identical(
    out$warnings,
    "one with {'resolution': 0.5} produced 1 cluster(s) on a subsample; stability and generalizability are degenerate at this point on the sweep axis."
  )
})

test_that("each fit gets the neighbor count scaled to its rows", {
  seen <- new.env()
  seen$fits <- list()
  recorder <- function(X, n_clusters = 2L, n_neighbors = 10L) {
    seen$fits[[length(seen$fits) + 1L]] <- c(nrow(X), n_neighbors)
    rep_len(seq_len(n_clusters), nrow(X))
  }
  validation_iter(blobs$X, recorder, "recorder", list(n_clusters = 2L, n_neighbors = 10L),
                  0.618, 3L, 0L, random_state = 0L)
  fits <- do.call(rbind, seen$fits)
  # Training and stability subsamples have 55 rows, the held-out set 35:
  # 10 * 55 / 90 rounds to 6 and 10 * 35 / 90 to 4.
  expect_identical(fits[, 1], c(55L, 35L, 55L))
  expect_identical(fits[, 2], c(6L, 4L, 6L))
})

test_that("the classifier gets the thread count it is given", {
  seen <- new.env()
  clf <- function(x_train, y_train, x_test, n_threads) {
    seen$threads <- n_threads
    rep(y_train[1], nrow(x_test))
  }
  validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L, 0L,
                  classifier = clf, random_state = 0L, classifier_threads = 3L)
  expect_identical(seen$threads, 3L)
})
