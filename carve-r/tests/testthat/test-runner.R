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

test_that("resample b gives the estimator and the classifier the seed random_state + b", {
  seen <- new.env()
  seen$estimator <- integer()
  seen$classifier <- integer()
  record_estimator <- function(X, n_clusters = 2L, random_state = NULL) {
    seen$estimator <- c(seen$estimator, random_state)
    rep_len(seq_len(n_clusters), nrow(X))
  }
  record_classifier <- function(x_train, y_train, x_test, random_state) {
    seen$classifier <- c(seen$classifier, random_state)
    rep(y_train[1], nrow(x_test))
  }
  run_validation(blobs$X, list(estimator_grid(record_estimator, n_clusters = 2L)), 3L, 0.618,
                 classifier = record_classifier, random_state = 10L,
                 sweep = resolve_sweep(n_clusters = 2L))
  # Three fits per resample: the first subsample, the held-out samples and
  # the second subsample.
  expect_identical(seen$estimator, rep(10:12, each = 3L))
  expect_identical(seen$classifier, 10:12)
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

# Custom functions without a random_state argument that draw from the
# session's random number stream.
session_kmeans <- function(X, n_clusters) as.integer(stats::kmeans(X, n_clusters)$cluster)
random_guess <- function(x_train, y_train, x_test) sample(y_train, nrow(x_test), replace = TRUE)
session_kmeans_grid <- list(estimator_grid(session_kmeans, n_clusters = 2:4))

test_that("an estimator that draws without random_state is reproducible", {
  set.seed(7)
  before <- .Random.seed
  a <- run_validation(blobs$X, session_kmeans_grid, 4L, 0.618, random_state = 5L, sweep = k_sweep)
  expect_identical(.Random.seed, before)
  expect_identical(a, run_validation(blobs$X, session_kmeans_grid, 4L, 0.618, random_state = 5L, sweep = k_sweep))
})

test_that("a classifier that draws without random_state is reproducible", {
  set.seed(7)
  before <- .Random.seed
  a <- run_validation(blobs$X, k_grid, 4L, 0.618, classifier = random_guess, random_state = 5L, sweep = k_sweep)
  expect_identical(.Random.seed, before)
  expect_identical(
    a,
    run_validation(blobs$X, k_grid, 4L, 0.618, classifier = random_guess, random_state = 5L, sweep = k_sweep)
  )
})

test_that("custom functions that draw give the same results on a parallel backend", {
  skip_on_os("windows")
  serial <- run_validation(blobs$X, session_kmeans_grid, 4L, 0.618, classifier = random_guess,
                           random_state = 5L, sweep = k_sweep)
  forked <- run_validation(blobs$X, session_kmeans_grid, 4L, 0.618, classifier = random_guess,
                           random_state = 5L, sweep = k_sweep, BPPARAM = BiocParallel::MulticoreParam(2L))
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

# Labels the first five rows of every subsample noise and clusters the rest.
noisy_kmeans <- function(X, n_clusters = 3L, random_state = NULL) {
  labels <- KMeans(X, n_clusters, random_state = random_state)
  labels[1:5] <- -1L
  labels
}

identity_spec <- function() {
  allocate_pipelines(list(preprocessing_option(Identity)), list(preprocessing_option(Identity)), 1L, 0L)[[1L]]
}

test_that("drop removes noise from the indices the scores use", {
  r <- validation_iter(blobs$X, noisy_kmeans, "noisy", list(n_clusters = 3L), 0.618, 5L, 0L, random_state = 0L)
  expect_length(r$train_indices, 50L)
  expect_length(r$labels_train, 50L)
  expect_length(r$stability_indices, 50L)
  expect_length(r$test_indices, 30L)
  expect_length(r$labels_predicted, 30L)
  expect_false(any(r$labels_train < 0L))
  expect_equal(r$noise_fraction, 5 / 55)
})

test_that("the samples drop removes are the ones labeled noise", {
  kept <- validation_iter(blobs$X, noisy_kmeans, "noisy", list(n_clusters = 3L), 0.618, 5L, 0L,
                          noise_policy = "as_cluster", random_state = 0L)
  dropped <- validation_iter(blobs$X, noisy_kmeans, "noisy", list(n_clusters = 3L), 0.618, 5L, 0L,
                             random_state = 0L)
  expect_identical(dropped$train_indices, kept$train_indices[kept$labels_train >= 0L])
  expect_identical(sum(kept$labels_train == -1L), 5L)
  expect_identical(kept$n_clusters_train, 3L)
})

test_that("the noise fraction is the same under every policy", {
  # Under "singleton" each noise sample is a cluster of its own, so the
  # k-axis count check warns; those warnings are not what this test is about.
  fractions <- vapply(NOISE_POLICIES, function(policy) {
    suppressWarnings(validation_iter(blobs$X, noisy_kmeans, "noisy", list(n_clusters = 3L), 0.618, 5L, 0L,
                                     noise_policy = policy, random_state = 0L))$noise_fraction
  }, numeric(1))
  expect_equal(unname(fractions), rep(5 / 55, 3))
})

test_that("a subsample that is all noise warns and scores NaN", {
  all_noise <- function(X, min_cluster_size = 5L) rep(-1L, nrow(X))
  out <- collect_warnings(validation_iter(
    blobs$X, all_noise, "all_noise", list(min_cluster_size = 5L), 0.618, 5L, 0L,
    sweep_param = "min_cluster_size", random_state = 0L
  ))
  expect_identical(out$warnings, c(
    "all_noise with {'min_cluster_size': 5} produced 0 cluster(s) on a subsample; stability and generalizability are degenerate at this point on the sweep axis.",
    "All points in a subsample were labelled as noise and dropped (noise_policy='drop'); stability ARI is undefined for this resample. Consider relaxing the clustering parameters or switching to noise_policy='as_cluster'.",
    "All points in a subsample were labelled as noise and dropped (noise_policy='drop'); generalizability ARI is undefined for this resample. Consider relaxing the clustering parameters or switching to noise_policy='as_cluster'."
  ))
  r <- out$value
  expect_true(is.nan(r$ari_stability))
  expect_true(is.nan(r$ari_generalizability))
  expect_null(r$labels_predicted)
  expect_length(r$labels_train, 0L)
  expect_identical(r$noise_fraction, 1)
})

test_that("an unknown noise policy is an error", {
  expect_error(
    validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L, 0L,
                    noise_policy = "keep", random_state = 0L),
    "Unknown noise_policy 'keep'.",
    fixed = TRUE
  )
})

test_that("identity embeddings reproduce the raw resample", {
  emb <- embed_resample(blobs$X, identity_spec(), seed = 2L, n_resamples = 5L,
                        subsample_ratio = 0.618, random_state = 10L)
  raw <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L,
                         seed = 2L, random_state = 10L)
  embedded <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L,
                              seed = 2L, random_state = 10L, embeddings = emb)
  expect_null(raw$pipeline)
  expect_identical(embedded$pipeline, "identity | identity")
  raw$pipeline <- NULL
  embedded$pipeline <- NULL
  expect_identical(embedded, raw)
})

test_that("each subsample is embedded with its own seed", {
  seen <- new.env()
  seen$calls <- list()
  record <- function(X, random_state = NULL) {
    seen$calls[[length(seen$calls) + 1L]] <- c(nrow(X), random_state)
    X
  }
  spec <- allocate_pipelines(list(preprocessing_option(Identity)), list(preprocessing_option(record)), 1L, 0L)[[1L]]
  embed_resample(blobs$X, spec, seed = 2L, n_resamples = 5L, subsample_ratio = 0.618, random_state = 10L)
  calls <- do.call(rbind, seen$calls)
  # The first subsample, the second and the held-out set, in that order.
  expect_identical(calls[, 1], c(55L, 55L, 35L))
  expect_identical(calls[, 2], c(12L, 17L, 22L))
})

test_that("a mode embeds only the subsamples it uses", {
  emb <- embed_resample(blobs$X, identity_spec(), 0L, 5L, 0.618, 0L, mode = "stability")
  expect_null(emb$X_test)
  expect_identical(nrow(emb$X_2), 55L)
  emb <- embed_resample(blobs$X, identity_spec(), 0L, 5L, 0.618, 0L, mode = "generalizability")
  expect_null(emb$X_2)
  expect_identical(nrow(emb$X_test), 35L)
})

test_that("the estimator clusters the embedding and the classifier reads raw features", {
  seen <- new.env()
  first_column <- function(X) X[, 1L, drop = FALSE]
  spec <- allocate_pipelines(list(preprocessing_option(Identity)), list(preprocessing_option(first_column)), 1L, 0L)[[1L]]
  emb <- embed_resample(blobs$X, spec, 0L, 5L, 0.618, 0L)
  width <- function(X, n_clusters = 3L) {
    seen$width <- c(seen$width, ncol(X))
    rep_len(seq_len(n_clusters), nrow(X))
  }
  clf <- function(x_train, y_train, x_test) {
    seen$classifier <- ncol(x_train)
    rep(y_train[1], nrow(x_test))
  }
  validation_iter(blobs$X, width, "width", list(n_clusters = 3L), 0.618, 5L, 0L,
                  classifier = clf, random_state = 0L, embeddings = emb)
  expect_identical(seen$width, c(1L, 1L, 1L))
  expect_identical(seen$classifier, 2L)
})

test_that("embeddings that do not fit the resample are an error", {
  emb <- embed_resample(blobs$X, identity_spec(), 0L, 5L, 0.618, 0L)
  emb$X_test <- emb$X_test[-1L, , drop = FALSE]
  expect_error(
    validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L, 0L,
                    random_state = 0L, embeddings = emb),
    "Precomputed embedding X_test has 34 rows but this resample's subsample has 35. This is an internal CARVE error.",
    fixed = TRUE
  )
})

test_that("an anchored run keeps anchor blocks and full-length scores", {
  anchors <- seq(1L, 90L, by = 3L)
  run <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep, anchors = anchors)
  expect_identical(dim(run$consensus_matrices[["1"]]), c(30L, 30L))
  expect_identical(dim(run$consensus_generalizability_matrices[["1"]]), c(30L, 30L))
  expect_length(run$summaries[["1"]]$gini, 90L)
  expect_length(run$summaries[["1"]]$ce, 90L)
  expect_length(run$generalizability_scores[["1"]], 90L)
  expect_identical(run$summaries[["1"]]$pac, compute_consensus_pac(run$consensus_matrices[["1"]]))
})

test_that("an anchored block is the exact matrix over the anchors", {
  anchors <- seq(2L, 90L, by = 4L)
  exact <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep)
  anchored <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep, anchors = anchors)
  expect_equal(anchored$consensus_matrices[["2"]], exact$consensus_matrices[["2"]][anchors, anchors])
  expect_identical(anchored$records, exact$records)
})

test_that("with every sample an anchor the run equals the exact run", {
  exact <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep)
  everything <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep, anchors = 1:90)
  expect_equal(everything$summaries, exact$summaries)
  expect_equal(everything$consensus_matrices, exact$consensus_matrices)
})

test_that("the noise policy reaches every resample", {
  grid <- list(estimator_grid(noisy_kmeans, n_clusters = 3L))
  axis <- resolve_sweep(n_clusters = 3L)
  dropped <- run_validation(blobs$X, grid, 3L, 0.618, random_state = 0L, sweep = axis)
  kept <- run_validation(blobs$X, grid, 3L, 0.618, random_state = 0L, sweep = axis, noise_policy = "as_cluster")
  expect_equal(dropped$records$noise_fraction, 5 / 55)
  expect_equal(kept$records$noise_fraction, 5 / 55)
  expect_false(identical(dropped$consensus_matrices, kept$consensus_matrices))
})

test_that("resamples that are all noise are left out of the aggregates", {
  # Resample 1 labels every sample noise; resamples 0 and 2 cluster.
  sometimes_noise <- function(X, min_cluster_size = 5L, random_state = NULL) {
    if (identical(random_state, 1L)) {
      return(rep(-1L, nrow(X)))
    }
    KMeans(X, 3L, random_state = random_state)
  }
  out <- collect_warnings(run_validation(
    blobs$X, list(estimator_grid(sometimes_noise, min_cluster_size = 5L)), 3L, 0.618,
    random_state = 0L, sweep = resolve_sweep(sweep = "min_cluster_size", sweep_values = 5L)
  ))
  run <- out$value
  expect_length(out$warnings, 3L)
  expect_equal(run$records$noise_fraction, 1 / 3)
  expect_false(is.nan(run$records$ari_stability))
  expect_false(all(is.nan(run$consensus_matrices[["0"]])))
  expect_false(anyNA(run$generalizability_scores[["0"]]))
})

test_that("with a BPPARAM the thread split counts only workers that get a resample", {
  local_mocked_bindings(n_cores = function() 8L)
  expect_identical(run_core_budget(1L, 2L, BiocParallel::SnowParam(4L)), c(outer = 4L, inner = 4L))
  expect_identical(run_core_budget(1L, 10L, BiocParallel::SnowParam(4L)), c(outer = 4L, inner = 2L))
  expect_identical(run_core_budget(4L, 10L), resolve_core_budget(4L, 10L))
})

test_that("an anchored run with noise does not depend on the backend", {
  skip_on_os("windows")
  grid <- list(estimator_grid(noisy_kmeans, n_clusters = 3L))
  axis <- resolve_sweep(n_clusters = 3L)
  serial <- run_validation(blobs$X, grid, 4L, 0.618, random_state = 0L, sweep = axis, anchors = seq(1L, 90L, by = 2L))
  forked <- run_validation(blobs$X, grid, 4L, 0.618, random_state = 0L, sweep = axis, anchors = seq(1L, 90L, by = 2L),
                           BPPARAM = BiocParallel::MulticoreParam(2L))
  expect_identical(serial, forked)
})
