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

k_grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
k_sweep <- resolve_sweep(n_clusters = 2:4)

test_that("run_validation returns one record and one artifact per configuration", {
  run <- run_validation(blobs$X, k_grid, n_resamples = 4L, subsample_ratio = 0.618,
                        random_state = 0L, sweep = k_sweep)
  expect_identical(run$records$config_id, 0:2)
  for (container in run[c("consensus_matrices", "consensus_generalizability_matrices",
                          "generalizability_scores", "summaries")]) {
    expect_identical(names(container), c("0", "1", "2"))
  }
  expect_identical(dim(run$consensus_matrices[["1"]]), c(90L, 90L))
  expect_identical(
    colnames(run$records),
    c("config_id", "method_id", "method_label", "estimator", "n_clusters",
      "sweep_param", "sweep_value", "sweep_rank", "n_clusters_observed",
      "n_clusters_observed_se", "noise_fraction",
      "ari_stability", "ari_stability_se", "ari_stability_upper", "ari_stability_lower",
      "ari_generalizability", "ari_generalizability_se", "ari_generalizability_upper",
      "ari_generalizability_lower",
      "ari_average", "ari_average_se", "ari_average_upper", "ari_average_lower")
  )
})

test_that("records carry the sweep bookkeeping and one method id per curve", {
  grids <- list(
    estimator_grid(KMeans, n_clusters = 2:3),
    estimator_grid(AgglomerativeClustering, n_clusters = 2:3, linkage = c("ward", "average"))
  )
  run <- run_validation(blobs$X, grids, n_resamples = 3L, subsample_ratio = 0.618,
                        random_state = 0L, sweep = resolve_sweep(n_clusters = 2:3))
  r <- run$records
  expect_identical(r$method_id, c("m0", "m0", "m1", "m1", "m2", "m2"))
  expect_identical(r$method_label, c(
    "KMeans", "KMeans",
    "AgglomerativeClustering, linkage=ward", "AgglomerativeClustering, linkage=ward",
    "AgglomerativeClustering, linkage=average", "AgglomerativeClustering, linkage=average"
  ))
  expect_identical(r$linkage, c(NA, NA, "ward", "ward", "average", "average"))
  expect_identical(r$sweep_rank, rep(0:1, 3))
  expect_identical(r$sweep_value, rep(2:3, 3))
  expect_identical(r$n_clusters_observed, rep(c(2, 3), 3))
  expect_identical(r$sweep_param, rep("n_clusters", 6))
})

test_that("the summaries come from the stored consensus matrix", {
  run <- run_validation(blobs$X, k_grid, 3L, 0.618, random_state = 0L, sweep = k_sweep)
  M <- run$consensus_matrices[["1"]]
  expect_identical(run$summaries[["1"]]$gini, stability_from_consensus(M)$gini)
  expect_identical(run$summaries[["1"]]$pac, compute_consensus_pac(M))
})

test_that("run_validation is reproducible and leaves the caller's RNG state alone", {
  set.seed(3)
  before <- .Random.seed
  a <- run_validation(blobs$X, k_grid, 3L, 0.618, random_state = 1L, sweep = k_sweep)
  expect_identical(.Random.seed, before)
  expect_identical(a, run_validation(blobs$X, k_grid, 3L, 0.618, random_state = 1L, sweep = k_sweep))
})

test_that("results do not depend on the parallel backend", {
  skip_on_os("windows")
  serial <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep)
  forked <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep,
                           BPPARAM = BiocParallel::MulticoreParam(2L))
  expect_identical(serial, forked)
})

test_that("what a worker receives does not grow with the configuration index", {
  # A SnowParam serializes the function and its arguments for every
  # resample. If the function's enclosure were run_validation's frame, every
  # consensus matrix computed so far would travel with it.
  local_mocked_bindings(resample_backend = function(...) {
    function(X, FUN, ...) {
      sizes <<- c(sizes, length(serialize(list(FUN, list(...)), NULL)))
      lapply(X, FUN, ...)
    }
  })
  # The first run lets R's just-in-time compiler compile the functions it
  # calls, which changes their serialized size once. The second is measured.
  for (run in 1:2) {
    sizes <- numeric()
    run_validation(blobs$X, list(estimator_grid(KMeans, n_clusters = 2:6)), 2L, 0.618,
                   random_state = 0L, sweep = resolve_sweep(n_clusters = 2:6))
  }
  expect_length(sizes, 5L)
  expect_identical(sizes, rep(sizes[[1L]], 5L))
})

test_that("the backend run_validation starts serves the whole run", {
  skip_on_os("windows")
  # Each resample reports the process it ran in. A backend started once has
  # the same two workers for every configuration.
  report_pid <- function(X, n_clusters = 2L) {
    warning(sprintf("process %d", Sys.getpid()))
    rep_len(seq_len(n_clusters), nrow(X))
  }
  out <- collect_warnings(run_validation(
    blobs$X, list(estimator_grid(report_pid, n_clusters = 2:4)), 4L, 0.618,
    n_jobs = 2L, random_state = 0L, sweep = resolve_sweep(n_clusters = 2:4)
  ))
  expect_lte(length(unique(out$warnings)), 2L)
})

test_that("a BPPARAM the caller started stays up, and one run_validation started is stopped", {
  skip_on_os("windows")
  started <- BiocParallel::MulticoreParam(2L)
  BiocParallel::bpstart(started)
  withr::defer(BiocParallel::bpstop(started))
  run_validation(blobs$X, k_grid, 2L, 0.618, random_state = 0L, sweep = k_sweep, BPPARAM = started)
  expect_true(BiocParallel::bpisup(started))
  idle <- BiocParallel::MulticoreParam(2L)
  run_validation(blobs$X, k_grid, 2L, 0.618, random_state = 0L, sweep = k_sweep, BPPARAM = idle)
  expect_false(BiocParallel::bpisup(idle))
})

# The classifier stops on an unexpected thread count. An error, unlike a
# value recorded into an environment, gets back from any backend.
threads_must_be <- function(expected) {
  function(x_train, y_train, x_test, n_threads) {
    if (!identical(n_threads, expected)) {
      stop(sprintf("classifier got %s threads", format(n_threads)))
    }
    rep(y_train[1], nrow(x_test))
  }
}

test_that("with one worker the classifier gets every core", {
  local_mocked_bindings(n_cores = function() 8L)
  expect_no_error(run_validation(blobs$X, k_grid, 2L, 0.618, classifier = threads_must_be(8L),
                                 n_jobs = 1L, random_state = 0L, sweep = k_sweep))
})

test_that("a BPPARAM sets the worker count and the classifier gets the rest", {
  local_mocked_bindings(n_cores = function() 8L)
  expect_no_error(run_validation(blobs$X, k_grid, 2L, 0.618, classifier = threads_must_be(8L),
                                 n_jobs = 4L, BPPARAM = BiocParallel::SerialParam(),
                                 random_state = 0L, sweep = k_sweep))
})

test_that("warnings raised in resamples reach the caller once per configuration", {
  noisy <- function(X, n_clusters = 2L) {
    warning("noisy estimator")
    rep_len(seq_len(n_clusters), nrow(X))
  }
  out <- collect_warnings(run_validation(
    blobs$X, list(estimator_grid(noisy, n_clusters = 2:3)), 3L, 0.618,
    random_state = 0L, sweep = resolve_sweep(n_clusters = 2:3)
  ))
  expect_identical(out$warnings, c("noisy estimator", "noisy estimator"))
})

test_that("mode = 'stability' builds no generalizability artifacts", {
  run <- run_validation(blobs$X, k_grid, 3L, 0.618, random_state = 0L, sweep = k_sweep, mode = "stability")
  expect_true(all(vapply(run$consensus_generalizability_matrices, is.null, logical(1))))
  expect_true(all(vapply(run$generalizability_scores, is.null, logical(1))))
  expect_true(all(is.nan(run$records$ari_generalizability)))
  expect_true(all(is.nan(run$records$ari_average)))
  expect_false(anyNA(run$records$ari_stability))
})

test_that("mode = 'generalizability' builds no stability artifacts", {
  run <- run_validation(blobs$X, k_grid, 3L, 0.618, random_state = 0L, sweep = k_sweep, mode = "generalizability")
  expect_null(run$summaries)
  expect_true(all(vapply(run$consensus_matrices, is.null, logical(1))))
  expect_true(all(is.nan(run$records$ari_stability)))
})

test_that("show_progress draws a bar over the configurations", {
  out <- capture.output(
    invisible(run_validation(blobs$X, k_grid, 2L, 0.618, random_state = 0L, sweep = k_sweep, show_progress = TRUE)),
    type = "message"
  )
  expect_match(paste(out, collapse = ""), "100%", fixed = TRUE)
})
