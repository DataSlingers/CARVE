skip_if_not_installed("ranger")

make_blobs <- function(n_per = 60L, seed = 1L) {
  set.seed(seed)
  centers <- rbind(c(0, 0), c(6, 0), c(3, 5))
  X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i) {
    matrix(stats::rnorm(n_per * 2L, 0, 0.4), ncol = 2L) +
      matrix(centers[i, ], n_per, 2L, byrow = TRUE)
  }))
  y <- rep(seq_len(nrow(centers)), each = n_per)
  list(X = X, y = y)
}

expected_cols <- c(
  "estimator", "n_clusters",
  "ari_stability", "ari_stability_se",
  "ari_stability_upper", "ari_stability_lower",
  "ari_generalizability", "ari_generalizability_se",
  "ari_generalizability_upper", "ari_generalizability_lower",
  "ari_average", "ari_average_se",
  "ari_average_upper", "ari_average_lower"
)

test_that("run_validation produces an estimator_results frame with all expected columns", {
  bl <- make_blobs()
  grids <- list(list(type = "kmeans", grid = list(n_clusters = 2:4)))

  res <- run_validation(
    X = bl$X, estimator_grids = grids,
    n_resamples = 5L, subsample_ratio = 0.8,
    random_state = 1L, mode = "default", verbose = 0L
  )

  expect_true(all(expected_cols %in% names(res$estimator_results)))
  expect_equal(nrow(res$estimator_results), 3L)
  expect_equal(sort(res$estimator_results$n_clusters), c(2, 3, 4))
  expect_equal(res$estimator_results$estimator, rep("KMeans", 3L))
})

test_that("run_validation picks k=3 on 3 well-separated blobs (max-rule)", {
  bl <- make_blobs()
  grids <- list(list(type = "kmeans", grid = list(n_clusters = 2:5)))

  res <- run_validation(
    X = bl$X, estimator_grids = grids,
    n_resamples = 10L, subsample_ratio = 0.8,
    random_state = 2L, mode = "default", verbose = 0L
  )

  k_best <- select_best_k(res$estimator_results,
                          measure = "stability", rule = "max")
  expect_equal(k_best, 3L)
})

test_that("run_validation returns per-config consensus matrices of shape n x n", {
  bl <- make_blobs()
  grids <- list(list(type = "kmeans", grid = list(n_clusters = 2:3)))
  n <- nrow(bl$X)

  res <- run_validation(
    X = bl$X, estimator_grids = grids,
    n_resamples = 5L, subsample_ratio = 0.8,
    random_state = 1L, verbose = 0L
  )

  expect_length(res$consensus_matrices, 2L)
  for (M in res$consensus_matrices) {
    expect_equal(dim(M), c(n, n))
  }
  expect_length(res$generalizability_scores, 2L)
  expect_equal(length(res$generalizability_scores[[1]]), n)
})

test_that("mode='stability' skips generalizability (NaN columns, NULL outputs)", {
  bl <- make_blobs()
  grids <- list(list(type = "kmeans", grid = list(n_clusters = 2:3)))

  res <- run_validation(
    X = bl$X, estimator_grids = grids,
    n_resamples = 5L, subsample_ratio = 0.8,
    random_state = 1L, mode = "stability", verbose = 0L
  )
  expect_true(all(is.nan(res$estimator_results$ari_generalizability)))
  expect_true(all(vapply(res$consensus_generalizability_matrices,
                         is.null, logical(1L))))
  expect_true(all(vapply(res$generalizability_scores, is.null, logical(1L))))
})

test_that("mode='generalizability' skips stability", {
  bl <- make_blobs()
  grids <- list(list(type = "kmeans", grid = list(n_clusters = 2:3)))

  res <- run_validation(
    X = bl$X, estimator_grids = grids,
    n_resamples = 5L, subsample_ratio = 0.8,
    random_state = 1L, mode = "generalizability", verbose = 0L
  )
  expect_true(all(is.nan(res$estimator_results$ari_stability)))
  expect_true(all(vapply(res$consensus_matrices, is.null, logical(1L))))
})

test_that("run_validation is reproducible given random_state", {
  bl <- make_blobs(n_per = 40L)
  grids <- list(list(type = "kmeans", grid = list(n_clusters = 2:3)))

  r1 <- run_validation(bl$X, grids, n_resamples = 5L,
                       subsample_ratio = 0.8, random_state = 7L, verbose = 0L)
  r2 <- run_validation(bl$X, grids, n_resamples = 5L,
                       subsample_ratio = 0.8, random_state = 7L, verbose = 0L)
  expect_equal(r1$estimator_results, r2$estimator_results)
})

test_that("validation_iter produces well-formed labels and indices", {
  bl <- make_blobs(n_per = 30L)
  # Give it a tiny pipeline & seed and verify structural invariants.
  res <- validation_iter(
    X = bl$X, type = "kmeans", params = list(n_clusters = 3L),
    subsample_ratio = 0.8, n_resamples = 5L, seed = 1L,
    normalization_options = default_normalization_options(),
    dim_reduction_options = list(list(type = "identity", grid = list())),
    random_state = 1L
  )
  expect_length(res$labels_train, length(res$train_indices))
  expect_length(res$labels_test, length(res$test_indices))
  expect_length(res$labels_predicted, length(res$test_indices))
  expect_true(all(res$train_indices %in% seq_len(nrow(bl$X))))
  expect_true(length(intersect(res$train_indices, res$test_indices)) == 0L)
})
