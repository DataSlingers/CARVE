test_that("compute_generalizability_scores averages per-sample correct counts", {
  # 4 samples, 2 runs. True labels match predictions for samples {1,2,4}
  # across both runs but diverge for sample 3.
  runs <- list(
    list(sample_idx = c(1L, 2L, 3L),
         labels = c(1L, 1L, 2L),
         predicted = c(1L, 1L, 1L)),  # sample 3 wrong (after alignment: both 1's, so sample 3 mislabelled)
    list(sample_idx = c(2L, 3L, 4L),
         labels = c(1L, 2L, 2L),
         predicted = c(1L, 2L, 2L))  # all correct
  )
  scores <- compute_generalizability_scores(n_samples = 4L, runs = runs)
  # Sample 1: 1/1, Sample 2: 2/2, Sample 3: splits → depends on alignment.
  expect_equal(scores[1], 1)
  expect_equal(scores[2], 1)
  expect_equal(scores[4], 1)
  # Sample 3 evaluated in both runs; first run mis-predicts, second is right.
  expect_equal(scores[3], 0.5)
})

test_that("samples never evaluated get score 0", {
  runs <- list(
    list(sample_idx = c(1L, 2L),
         labels = c(1L, 2L),
         predicted = c(1L, 2L))
  )
  scores <- compute_generalizability_scores(n_samples = 5L, runs = runs)
  expect_equal(scores[3:5], c(0, 0, 0))
})

test_that("adjusted_rand_index is 1 on perfect matches", {
  a <- c(1L, 1L, 2L, 2L, 3L, 3L)
  expect_equal(adjusted_rand_index(a, a), 1)
  # Permuted labels still give ARI = 1.
  expect_equal(adjusted_rand_index(a, c(2L, 2L, 3L, 3L, 1L, 1L)), 1)
})

test_that("adjusted_rand_index is ~0 for a random vs fixed partition", {
  set.seed(1L)
  a <- rep(1:3, each = 30L)
  b <- sample(1:3, 90L, replace = TRUE)
  expect_lt(abs(adjusted_rand_index(a, b)), 0.2)
})

test_that("adjusted_rand_index errors on length mismatch", {
  expect_error(adjusted_rand_index(1:3, 1:4), "same length")
})
