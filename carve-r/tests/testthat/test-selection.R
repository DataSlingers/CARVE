make_results <- function() {
  data.frame(
    estimator = rep("KMeans", 4L),
    n_clusters = 2:5,
    ari_stability = c(0.90, 0.88, 0.85, 0.50),
    ari_stability_se = c(0.02, 0.02, 0.02, 0.02),
    ari_stability_upper = c(0.92, 0.90, 0.87, 0.52),
    ari_stability_lower = c(0.88, 0.86, 0.83, 0.48),
    stringsAsFactors = FALSE
  )
}

test_that("max rule picks the highest-score row", {
  df <- make_results()
  row <- select_best_row_max(df, measure = "stability")
  expect_equal(row$n_clusters, 2L)
  expect_equal(select_best_k(df, measure = "stability", rule = "max"), 2L)
})

test_that("1se rule picks the largest k within one SE of the best", {
  df <- make_results()
  # best = 0.90 at k=2, se = 0.02, threshold = 0.88.
  # Rows above threshold: k=2 (0.90), k=3 (0.88). Largest k = 3.
  row <- select_best_row_1se(df, measure = "stability")
  expect_equal(row$n_clusters, 3L)
  expect_equal(select_best_k(df, measure = "stability", rule = "1se"), 3L)
})

test_that("quantile rule picks the largest k within best-score bounds", {
  df <- make_results()
  # best row k=2 has upper=0.92, lower=0.88.
  # Rows within [0.88, 0.92]: k=2 (0.90), k=3 (0.88). Largest k = 3.
  row <- select_best_row_quantile(df, measure = "stability")
  expect_equal(row$n_clusters, 3L)
})

test_that("not_two excludes k=2 before selecting", {
  df <- make_results()
  row <- select_best_row_by_rule(df, measure = "stability", rule = "max", not_two = TRUE)
  expect_equal(row$n_clusters, 3L)
})

test_that("measure aliases dispatch to the same column", {
  df <- make_results()
  expect_equal(
    select_best_row_max(df, measure = "s")$n_clusters,
    select_best_row_max(df, measure = "ari_stability")$n_clusters
  )
  expect_equal(
    select_best_row_max(df, measure = "stab")$n_clusters,
    select_best_row_max(df, measure = "stability")$n_clusters
  )
})

test_that("invalid measure raises an informative error", {
  df <- make_results()
  expect_error(select_best_row_max(df, measure = "not_a_real_measure"), "Invalid measure")
})

test_that("1se falls back to max when SE column is missing", {
  df <- make_results()
  df$ari_stability_se <- NULL
  expect_warning(
    row <- select_best_row_by_rule(df, measure = "stability", rule = "1se"),
    "Column 'ari_stability_se' not found"
  )
  expect_equal(row$n_clusters, 2L)
})

test_that("quantile falls back to max when upper column is missing", {
  df <- make_results()
  df$ari_stability_upper <- NULL
  expect_warning(
    row <- select_best_row_by_rule(df, measure = "stability", rule = "quantile"),
    "Column 'ari_stability_upper' not found"
  )
  expect_equal(row$n_clusters, 2L)
})

test_that("select_best_estimator returns name, params, and k", {
  df <- make_results()
  df$linkage <- c("ward", "ward", "ward", "ward")
  out <- select_best_estimator(df, measure = "stability", rule = "1se")
  expect_equal(out$estimator, "KMeans")
  expect_equal(out$n_clusters, 3L)
  expect_equal(out$params$linkage, "ward")
})

test_that("select_best_estimator filters by k when k is given", {
  df <- make_results()
  out <- select_best_estimator(df, measure = "stability", rule = "max", k = 4L)
  expect_equal(out$n_clusters, 4L)
})

test_that("not_two raises when no rows remain", {
  df <- data.frame(
    estimator = "KMeans",
    n_clusters = 2L,
    ari_stability = 0.9,
    ari_stability_se = 0.01,
    stringsAsFactors = FALSE
  )
  expect_error(
    select_best_row_by_rule(df, measure = "stability", rule = "max", not_two = TRUE),
    "No configurations remain"
  )
})
