test_that("split_subsample_indices produces a valid train/test partition", {
  res <- split_subsample_indices(10L, subsample_ratio = 0.8, random_state = 42L)
  expect_length(res$train, 8L)
  expect_length(res$test, 2L)
  expect_setequal(sort(c(res$train, res$test)), seq_len(10L))
  expect_true(length(intersect(res$train, res$test)) == 0L)
})

test_that("split_subsample_indices is reproducible for a fixed seed", {
  a <- split_subsample_indices(20L, subsample_ratio = 0.7, random_state = 7L)
  b <- split_subsample_indices(20L, subsample_ratio = 0.7, random_state = 7L)
  expect_equal(a, b)
})

test_that("split_subsample_indices does not leak RNG state to the caller", {
  set.seed(123L)
  before <- .Random.seed
  split_subsample_indices(50L, random_state = 999L)
  after <- .Random.seed
  expect_equal(before, after)
})

test_that("coerce_n_clusters accepts an integer scalar", {
  expect_equal(coerce_n_clusters(5L), 2:5)
  expect_equal(coerce_n_clusters(10), 2:10)
})

test_that("coerce_n_clusters accepts a vector and validates", {
  expect_equal(coerce_n_clusters(c(3L, 4L, 5L)), c(3L, 4L, 5L))
  expect_error(coerce_n_clusters(1L), ">= 2")
  expect_error(coerce_n_clusters(c(2L, 1L, 3L)), ">= 2")
  expect_error(coerce_n_clusters(c(2.5, 3.5)), "integers")
})

test_that("summarize_ari_scores handles NaNs and single-observation input", {
  out <- summarize_ari_scores(c(0.8, 0.85, 0.9, NA, 0.95))
  expect_equal(unname(out["mean"]), mean(c(0.8, 0.85, 0.9, 0.95)))
  expect_equal(unname(out["se"]),
               stats::sd(c(0.8, 0.85, 0.9, 0.95)) / sqrt(4),
               tolerance = 1e-12)
  # All-NaN input returns all NaN.
  nan_out <- summarize_ari_scores(c(NA, NA))
  expect_true(all(is.nan(nan_out)))
  # Single observation -> se is NaN.
  one_out <- summarize_ari_scores(c(0.5, NA))
  expect_true(is.nan(unname(one_out["se"])))
  expect_equal(unname(one_out["mean"]), 0.5)
})

test_that("ensure_2d_matrix accepts matrix, data.frame, and numeric vector", {
  M <- matrix(1:12, 3, 4)
  expect_equal(ensure_2d_matrix(M), M)
  expect_equal(dim(ensure_2d_matrix(as.data.frame(M))), c(3L, 4L))
  expect_equal(dim(ensure_2d_matrix(1:5)), c(5L, 1L))
  expect_error(ensure_2d_matrix("not a matrix"), "matrix")
})

test_that("resolve_mode returns the expected policy flags", {
  p <- resolve_mode("default")
  expect_true(p$run_stability)
  expect_true(p$run_generalizability)
  expect_true(p$compute_average_ari)

  p <- resolve_mode("stability")
  expect_true(p$run_stability)
  expect_false(p$run_generalizability)
  expect_false(p$compute_average_ari)

  p <- resolve_mode("generalizability")
  expect_false(p$run_stability)
  expect_true(p$run_generalizability)
  expect_false(p$compute_average_ari)

  expect_error(resolve_mode("bogus"), "should be one of")
})
