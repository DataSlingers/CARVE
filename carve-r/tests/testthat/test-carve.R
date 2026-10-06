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
  limit <- .Machine$integer.max - 8L
  expect_error(
    carve(blobs$X, n_resamples = 4, random_state = limit + 1),
    sprintf("random_state must be at most %.0f so that every derived seed fits in an R integer.", limit),
    fixed = TRUE
  )
  expect_identical(resolve_seed(limit, 4L), as.integer(limit))
  expect_error(resolve_seed(limit + 1, 4L), "random_state must be at most", fixed = TRUE)
})
