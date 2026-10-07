blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:4))

small_fit <- function(...) {
  carve(blobs$X, n_clusters = 2:4, n_resamples = 4, estimator_param_grids = k_grid,
        random_state = 0, ...)
}

small_fit_n <- function(n_resamples) {
  carve(blobs$X, n_clusters = 2:4, n_resamples = n_resamples, estimator_param_grids = k_grid,
        random_state = 0)
}

test_that("carve returns a CARVE object whose containers follow config_id", {
  fit <- small_fit()
  expect_s4_class(fit, "CARVE")
  expect_no_error(validObject(fit))
  results <- fit@estimator_results
  expect_identical(results$config_id, 0:2)
  expect_identical(names(fit@consensus_matrices), c("0", "1", "2"))
  expect_identical(names(fit@stability_gini_scores), c("0", "1", "2"))
  expect_identical(names(fit@generalizability_scores), c("0", "1", "2"))
  expect_identical(fit@input_data, blobs$X)
  expect_identical(fit@sweep@values, 2:4)
  expect_identical(fit@run_params$random_state, 0L)
  expect_identical(
    tail(names(results), 4),
    c("consensus_pac_stability", "consensus_gini_stability", "consensus_ce_stability", "accuracy_generalizability")
  )
  expect_equal(results$consensus_gini_stability[2], mean(fit@stability_gini_scores[["1"]]))
  expect_equal(results$consensus_ce_stability[2], mean(fit@stability_ce_scores[["1"]]))
  expect_equal(results$accuracy_generalizability[3], mean(fit@generalizability_scores[["2"]]))
})

test_that("the stability columns are the plain means of the per-sample scores", {
  # With 4 resamples some sample is never drawn with a partner and the means
  # are NaN, so the first test cannot tell a mean from a median. Thirty
  # resamples give finite scores whose mean and median differ.
  fit <- small_fit_n(30)
  results <- fit@estimator_results
  gini <- fit@stability_gini_scores
  ce <- fit@stability_ce_scores
  expect_true(all(is.finite(results$consensus_gini_stability)))
  expect_false(isTRUE(all.equal(mean(gini[["0"]]), stats::median(gini[["0"]]))))
  expect_identical(results$consensus_gini_stability[1], mean(gini[["0"]]))
  expect_identical(results$consensus_gini_stability[3], mean(gini[["2"]]))
  expect_identical(results$consensus_ce_stability[1], mean(ce[["0"]]))
  expect_identical(results$consensus_ce_stability[3], mean(ce[["2"]]))
})

test_that("the stability summaries are joined to the results by config_id", {
  real_run <- run_validation
  local_mocked_bindings(run_validation = function(...) {
    run <- real_run(...)
    run$summaries <- rev(run$summaries)
    run
  })
  fit <- small_fit_n(30)
  results <- fit@estimator_results
  keys <- as.character(results$config_id)
  expect_identical(names(fit@stability_gini_scores), keys)
  expect_identical(
    results$consensus_pac_stability,
    unname(vapply(keys, function(k) compute_consensus_pac(fit@consensus_matrices[[k]]), numeric(1)))
  )
  expect_identical(
    results$consensus_gini_stability,
    unname(vapply(keys, function(k) mean(fit@stability_gini_scores[[k]]), numeric(1)))
  )
  expect_identical(
    results$consensus_ce_stability,
    unname(vapply(keys, function(k) mean(fit@stability_ce_scores[[k]]), numeric(1)))
  )
})

test_that("the light preset runs three estimators per k", {
  fit <- carve(blobs$X, n_clusters = 3, n_resamples = 2, random_state = 0)
  expect_identical(
    unique(fit@estimator_results$estimator),
    c("KMeans", "AgglomerativeClustering", "SpectralClustering")
  )
  expect_identical(fit@estimator_results$n_clusters, rep(2:3, 3))
})

test_that("carve validates the run settings before running", {
  X <- blobs$X
  expect_error(
    carve(X, subsample_ratio = 1),
    "subsample_ratio must be in (0, 1), got 1. Each resample needs both a training and a held-out split.",
    fixed = TRUE
  )
  expect_error(carve(X, n_resamples = 0), "n_resamples must be at least 1, got 0.", fixed = TRUE)
  expect_error(carve(X, n_trees = 0), "n_trees must be at least 1, got 0.", fixed = TRUE)
  expect_error(
    carve(X, estimator_param_grids = "medium"),
    "Unknown estimator_param_grids preset 'medium'. Expected 'light', 'full', or a list of estimator_grid() specifications.",
    fixed = TRUE
  )
  expect_error(carve(X, estimator_param_grids = list()), "estimator_param_grids must not be empty.", fixed = TRUE)
  expect_error(carve(X, classifier = "rf"), "classifier must be NULL or a function.", fixed = TRUE)
  expect_error(carve(X, k = 3), "Unknown argument: k.", fixed = TRUE)
  expect_error(carve(X, random_state = -1), "random_state must be NULL or a non-negative whole number.", fixed = TRUE)
  expect_error(carve(X, reference_labels = 1:3), "reference_labels has 3 entries but X has 90 rows.", fixed = TRUE)
  expect_error(carve(list(1, 2)), "X must be a numeric matrix", fixed = TRUE)
})

test_that("a non-default mode warns and skips the other criterion", {
  fit <- NULL
  expect_warning(
    fit <- small_fit(mode = "stability"),
    "Non-default mode is experimental and may break downstream functionality.",
    fixed = TRUE
  )
  expect_true(all(vapply(fit@generalizability_scores, is.null, logical(1))))
  expect_true(all(is.nan(fit@estimator_results$accuracy_generalizability)))
  expect_warning(fit <- small_fit(mode = "generalizability"), "Non-default mode is experimental", fixed = TRUE)
  expect_null(fit@stability_gini_scores)
  expect_null(fit@stability_ce_scores)
  expect_true(all(is.nan(fit@estimator_results$consensus_pac_stability)))
})

test_that("n_trees with a custom classifier warns", {
  clf <- function(x_train, y_train, x_test) rep(y_train[1], nrow(x_test))
  expect_warning(
    small_fit(classifier = clf, n_trees = 50),
    "n_trees is ignored when a custom classifier is provided.",
    fixed = TRUE
  )
})

test_that("a data frame gives the same results as the matrix", {
  from_frame <- carve(as.data.frame(blobs$X), n_clusters = 2:4, n_resamples = 4,
                      estimator_param_grids = k_grid, random_state = 0)
  expect_identical(from_frame@estimator_results, small_fit()@estimator_results)
})

test_that("reference labels are coded as integers", {
  fit <- small_fit(reference_labels = rep(c("b", "a", NA), each = 30))
  expect_identical(fit@reference_labels, rep(c(1L, 2L, -1L), each = 30))
  fit <- small_fit(reference_labels = rep(c(3, 1, 2), each = 30))
  expect_identical(fit@reference_labels, rep(c(3L, 1L, 2L), each = 30))
})

test_that("whole numbers with an NA are coded like any other labels", {
  # Python factorizes here too: a numpy array with a NaN is not an integer array.
  expect_identical(coerce_reference_labels(c(3, 1, NA, 3), 4L), c(1L, 2L, -1L, 1L))
})

test_that("a fixed random_state reproduces the fit and NULL means 0", {
  expect_identical(small_fit(), small_fit())
  null_seed <- carve(blobs$X, n_clusters = 2:4, n_resamples = 4, estimator_param_grids = k_grid)
  expect_identical(null_seed@estimator_results, small_fit()@estimator_results)
})

test_that("a custom grid with one k value is kept as given", {
  fit <- carve(blobs$X, n_resamples = 2, random_state = 0,
               estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 3L)))
  expect_identical(fit@sweep@values, 3L)
  expect_identical(fit@estimator_results$n_clusters, 3L)
})

test_that("custom grids may sweep another registered parameter", {
  cut_at <- function(X, resolution = 1) {
    as.integer(stats::cutree(stats::hclust(stats::dist(X)), k = round(2 * resolution)))
  }
  fit <- carve(blobs$X, n_resamples = 2, random_state = 0,
               estimator_param_grids = list(estimator_grid(cut_at, resolution = c(1, 1.5))))
  expect_identical(fit@sweep@param, "resolution")
  expect_identical(fit@estimator_results$n_clusters_observed, c(2, 3))
  expect_identical(fit@estimator_results$sweep_rank, 0:1)
})

test_that("grids that mix sweep axes are rejected", {
  toy <- function(X, n_clusters = 2L, resolution = 1) rep(1L, nrow(X))
  expect_error(
    carve(blobs$X, estimator_param_grids = list(
      estimator_grid(toy, n_clusters = 2:3),
      estimator_grid(toy, resolution = c(0.5, 1))
    )),
    "Estimator grids sweep more than one parameter",
    fixed = TRUE
  )
})

test_that("a misaligned container fails validation", {
  fit <- small_fit()
  fit@consensus_matrices <- rev(fit@consensus_matrices)
  expect_error(
    validObject(fit),
    "config_id is misaligned with the per-configuration artifact containers. This is an internal CARVE error.",
    fixed = TRUE
  )
})

test_that("config_id must count from 0 in table order", {
  fit <- small_fit()
  fit@estimator_results$config_id <- c(1L, 0L, 2L)
  expect_error(validObject(fit), "config_id is misaligned", fixed = TRUE)
})

test_that("random_state must leave room for every derived seed", {
  # The held-out embedding of the last resample uses
  # random_state + 3 * n_resamples - 1.
  limit <- .Machine$integer.max - 12L
  expect_error(
    carve(blobs$X, n_resamples = 4, random_state = limit + 1),
    sprintf("random_state must be at most %.0f so that every derived seed fits in an R integer.", limit),
    fixed = TRUE
  )
  expect_identical(resolve_seed(limit, 4L), as.integer(limit))
  expect_error(resolve_seed(limit + 1, 4L), "random_state must be at most", fixed = TRUE)
})

blobs3d <- make_blobs(n_per = 30L, centers = diag(5, 3), sd = 1, seed = 42L)
resolution_fit <- carve(blobs3d$X, resolution = c(0.2, 0.5, 1), n_resamples = 4, random_state = 0)

test_that("a resolution sweep records its axis and the observed cluster counts", {
  axis <- resolution_fit@sweep
  expect_identical(axis@param, "resolution")
  expect_identical(axis@values, c(0.2, 0.5, 1))
  expect_true(axis@finer_is_larger)
  expect_false(axis@fixes_k)
  results <- resolution_fit@estimator_results
  expect_false("n_clusters" %in% names(results))
  expect_identical(results$estimator, rep(c("LeidenClustering", "LouvainClustering"), each = 3))
  expect_identical(results$method_label, rep(c("LeidenClustering, n_neighbors=15", "LouvainClustering, n_neighbors=15"), each = 3))
  expect_identical(results$sweep_value, rep(c(0.2, 0.5, 1), 2))
  expect_identical(results$sweep_rank, rep(0:2, 2))
  expect_identical(results$noise_fraction, rep(0, 6))
  expect_true(get_sweep_value(resolution_fit) %in% c(0.2, 0.5, 1))
  expect_true(get_k(resolution_fit) %in% round(results$n_clusters_observed))
})

test_that("labels on a resolution sweep pin sweep values, not k", {
  expect_length(get_labels(resolution_fit), 90L)
  expect_error(get_labels(resolution_fit, k = 3), "This run swept 'resolution', not 'n_clusters'.", fixed = TRUE)
  expect_length(get_labels(resolution_fit, sweep_value = 0.5), 90L)
  expect_error(get_labels(resolution_fit, sweep_value = 0.7), "No configurations found for resolution=0.7.", fixed = TRUE)
  expect_identical(length(unique(get_labels(resolution_fit, consensus_k = 4))), 4L)
})

test_that("resolution and a different sweep cannot be combined", {
  expect_error(
    carve(blobs$X, resolution = 1, sweep = "n_clusters"),
    "resolution= was given but sweep='n_clusters'. Pass sweep_values= instead, or drop resolution=.",
    fixed = TRUE
  )
})

test_that("a min_cluster_size sweep runs HDBSCAN with ranks running backwards", {
  skip_if_not_installed("dbscan")
  fit <- carve(blobs3d$X, sweep = "min_cluster_size", sweep_values = c(5, 8, 10),
               n_resamples = 4, random_state = 0)
  expect_false(fit@sweep@finer_is_larger)
  results <- fit@estimator_results
  expect_identical(unique(results$estimator), "HDBSCAN")
  expect_identical(results$min_cluster_size, c(5L, 8L, 10L))
  expect_identical(results$sweep_rank, c(2L, 1L, 0L))
  expect_true(all(results$noise_fraction >= 0 & results$noise_fraction < 1))
  expect_length(get_labels(fit, sweep_value = 10), 90L)
  selected <- get_estimator(fit)
  expect_identical(attr(selected, "estimator"), "HDBSCAN")
  expect_identical(attr(selected, "params")$cluster_selection_method, "eom")
})

# Labels the first five rows of every subsample noise.
first_five_noise <- function(X, n_clusters = 3L, random_state = NULL) {
  labels <- KMeans(X, n_clusters, random_state = random_state)
  labels[1:5] <- -1L
  labels
}

test_that("every noise policy records the same noise fraction", {
  grid <- list(estimator_grid(first_five_noise, n_clusters = 3L))
  fractions <- vapply(c("drop", "as_cluster", "singleton"), function(policy) {
    # "singleton" makes each noise sample a cluster, so the k-axis check warns.
    fit <- suppressWarnings(carve(blobs$X, n_resamples = 3, random_state = 0,
                                  estimator_param_grids = grid, noise_policy = policy))
    fit@estimator_results$noise_fraction
  }, numeric(1))
  expect_equal(unname(fractions), rep(5 / 55, 3))
})

test_that("the new settings are checked before the run", {
  X <- blobs$X
  expect_error(
    carve(X, noise_policy = "keep"),
    "Unknown noise_policy 'keep'. Expected 'drop', 'as_cluster', or 'singleton'.",
    fixed = TRUE
  )
  expect_error(carve(X, anchor_threshold = -1), "anchor_threshold must be a non-negative whole number.", fixed = TRUE)
  expect_error(carve(X, randomize_preprocessing = NA), "randomize_preprocessing must be TRUE or FALSE.", fixed = TRUE)
  expect_error(
    carve(X, normalization_options = list(Identity)),
    "normalization_options must be NULL or a list of preprocessing_option() specifications.",
    fixed = TRUE
  )
  expect_error(carve(X, consensus_anchors = 2.5), "consensus_anchors given as a fraction must be in (0, 1], got 2.5.", fixed = TRUE)
})

two <- make_blobs(n_per = 30L, centers = rbind(c(0, 0), c(6, 0)), seed = 4L)
anchored_fit <- function(...) {
  carve(two$X, n_resamples = 6, random_state = 0,
        estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3)), ...)
}

test_that("above anchor_threshold the run is anchored and says why", {
  fit <- NULL
  expect_warning(
    fit <- anchored_fit(anchor_threshold = 30),
    "n=60 exceeds anchor_threshold=30, so CARVE is using anchored consensus over 30 anchors. Per-sample scores and labels still cover every sample; consensus matrices and PAC are computed over the anchors.",
    fixed = TRUE
  )
  expect_no_error(validObject(fit))
  expect_length(fit@consensus_anchors, 30L)
  expect_false(is.unsorted(fit@consensus_anchors, strictly = TRUE))
  expect_identical(dim(fit@consensus_matrices[["0"]]), c(30L, 30L))
  expect_identical(dim(fit@consensus_generalizability_matrices[["1"]]), c(30L, 30L))
  expect_length(fit@stability_gini_scores[["0"]], 60L)
  expect_length(fit@generalizability_scores[["0"]], 60L)
  # One anchor draw serves every configuration.
  expect_identical(is.nan(fit@consensus_matrices[["0"]]), is.nan(fit@consensus_matrices[["1"]]))
  expect_output(show(fit), "Consensus: anchored over 30 anchors", fixed = TRUE)
})

test_that("consensus_anchors anchors a run below the threshold and the warning says so", {
  fit <- NULL
  expect_warning(
    fit <- anchored_fit(anchor_threshold = 1000, consensus_anchors = 25),
    "consensus_anchors=25 opts this run in regardless of anchor_threshold, so CARVE is using anchored consensus over 25 anchors.",
    fixed = TRUE
  )
  expect_length(fit@consensus_anchors, 25L)
})

test_that("at or below the threshold the run is exact", {
  fit <- anchored_fit(anchor_threshold = 100)
  expect_null(fit@consensus_anchors)
  expect_identical(dim(fit@consensus_matrices[["0"]]), c(60L, 60L))
  expect_output(show(fit), "Consensus: exact", fixed = TRUE)
})

test_that("the fit stores the threads its whole budget allows", {
  local_mocked_bindings(n_cores = function() 11L)
  grid <- list(estimator_grid(KMeans, n_clusters = 2L))
  fit <- carve(blobs$X, n_resamples = 4, random_state = 0, estimator_param_grids = grid,
               BPPARAM = BiocParallel::SerialParam())
  expect_identical(fit@run_params$n_threads, 11L)
  skip_on_os("windows")
  # Four workers with two threads each.
  fit <- carve(blobs$X, n_resamples = 4, random_state = 0, estimator_param_grids = grid, n_jobs = 4)
  expect_identical(fit@run_params$n_threads, 8L)
})

randomized_fit <- function(...) {
  carve(blobs$X, n_resamples = 8, random_state = 0, estimator_param_grids = k_grid,
        randomize_preprocessing = TRUE,
        normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
        dim_reduction_options = list(preprocessing_option(Identity), preprocessing_option(PCA, n_components = c(1L, 2L))),
        ...)
}

test_that("a randomized fit splits each configuration's scores by pipeline", {
  fit <- randomized_fit()
  table <- fit@preprocessing_results
  expect_identical(names(table), c(
    "method_id", "method_label", "pipeline", "normalization", "dim_reduction", "n_clusters",
    "n_resamples", "ari_stability", "ari_stability_se", "ari_generalizability",
    "ari_generalizability_se", "n_clusters_observed", "sweep_param", "sweep_value", "sweep_rank"
  ))
  expect_identical(as.vector(tapply(table$n_resamples, table$n_clusters, sum)), rep(8L, 3))
  expect_setequal(unique(table$pipeline), names(fit@preprocessing_pipelines))
  expect_identical(table$pipeline, paste(table$normalization, table$dim_reduction, sep = " | "))
  joined <- merge(table, fit@estimator_results[, c("method_id", "n_clusters", "config_id")],
                  by = c("method_id", "n_clusters"))
  expect_identical(nrow(joined), nrow(table))
  expect_true(all(vapply(fit@preprocessing_pipelines, inherits, logical(1), "carve_pipeline_spec")))
  expect_true(fit@run_params$randomize_preprocessing)
  expect_output(show(fit), sprintf("Preprocessing: randomized over %d pipelines", length(fit@preprocessing_pipelines)), fixed = TRUE)
})

test_that("a fit without randomization has no preprocessing results", {
  fit <- small_fit()
  expect_null(fit@preprocessing_results)
  expect_null(fit@preprocessing_pipelines)
})

test_that("the default options are resolved only for a randomized fit", {
  local_mocked_bindings(
    default_normalization_options = function(X) stop("resolved"),
    default_dim_reduction_options = function(X, subsample_ratio) stop("resolved")
  )
  expect_no_error(small_fit())
  expect_error(small_fit(randomize_preprocessing = TRUE), "resolved", fixed = TRUE)
})

test_that("an empty reduction list under randomization is an error", {
  expect_error(
    small_fit(randomize_preprocessing = TRUE, dim_reduction_options = list(),
              normalization_options = list(preprocessing_option(Identity))),
    "randomize_preprocessing=TRUE needs at least one normalization option and one dimensionality reduction option.",
    fixed = TRUE
  )
})

test_that("the verbose header names the resolved options", {
  messages <- capture_messages(randomized_fit(verbose = 2))
  text <- paste(messages, collapse = "")
  expect_match(text, "[CARVE] normalization      : identity, StandardScaler", fixed = TRUE)
  expect_match(text, "[CARVE] dim_reduction      : identity, PCA", fixed = TRUE)
})
