test_that("selection matches Python for every measure, rule and not_two", {
  fx <- read_fixture("selection")
  table <- as.data.frame(fx$table)
  for (case in fx$expected) {
    info <- paste(case$measure, case$rule, case$not_two)
    row <- NULL
    if (is.null(case$warning)) {
      expect_no_warning(
        row <- select_best_row_by_rule(table, case$measure, case$rule, not_two = case$not_two)
      )
    } else {
      expect_warning(
        row <- select_best_row_by_rule(table, case$measure, case$rule, not_two = case$not_two),
        case$warning,
        fixed = TRUE
      )
    }
    expect_identical(row$config_id, as.integer(case$config_id), info = info)
    k <- suppressWarnings(select_best_k(table, case$measure, case$rule, not_two = case$not_two))
    expect_identical(k, as.integer(case$k), info = info)
  }
})

quantile_table <- data.frame(
  estimator = "KMeans",
  n_clusters = 2:3,
  sweep_rank = 0:1,
  n_clusters_observed = c(2, 3),
  ari_stability = c(0.5, 0.3),
  ari_stability_se = 0.01,
  ari_stability_upper = c(0.4, 0.2),
  ari_stability_lower = c(0.6, 0.4)
)

test_that("quantile falls back to max when no row lies within the bounds", {
  row <- NULL
  expect_warning(
    row <- select_best_row_quantile(quantile_table, "stability"),
    "No estimators within quantile thresholds; falling back to max.",
    fixed = TRUE
  )
  expect_identical(row$n_clusters, 2L)
})

test_that("1se picks the finest rank within one standard error", {
  df <- data.frame(
    n_clusters = 2:5, sweep_rank = 0:3, n_clusters_observed = 2:5,
    ari_stability = c(0.90, 0.89, 0.88, 0.50),
    ari_stability_se = c(0.05, 0.04, 0.03, 0.02)
  )
  expect_identical(select_best_row_1se(df, "stability")$n_clusters, 4L)
})

test_that("the rules order by sweep_rank, not by the observed cluster count", {
  # A min_cluster_size axis: smaller values are finer and rank higher. The
  # observed counts do not follow the ranks, as can happen with noise.
  df <- data.frame(
    sweep_value = c(5, 10, 20, 40), sweep_rank = 3:0, n_clusters_observed = c(4, 6, 5, 2),
    ari_stability = c(0.87, 0.86, 0.90, 0.50), ari_stability_se = 0.05,
    ari_stability_upper = 0.95, ari_stability_lower = 0.84
  )
  # The best row has rank 1. Ranks 3, 2 and 1 lie within one standard error
  # and within the percentiles; rank 3 is the finest, and rank 2 has the
  # most clusters.
  expect_identical(select_best_row_max(df, "stability")$sweep_value, 20)
  expect_identical(select_best_row_1se(df, "stability")$sweep_value, 5)
  expect_identical(select_best_row_quantile(df, "stability")$sweep_value, 5)
})

test_that("1se with a NaN standard error returns the best row", {
  df <- data.frame(
    n_clusters = 2:3, sweep_rank = 0:1, n_clusters_observed = 2:3,
    ari_stability = c(0.9, 0.8), ari_stability_se = NaN
  )
  expect_identical(select_best_row_1se(df, "stability")$n_clusters, 2L)
})

test_that("selection rejects unknown measures and rules", {
  expect_error(
    select_best_row_by_rule(quantile_table, "silhouette", "max"),
    "Invalid measure 'silhouette'. Options: ['s', 'stab', ",
    fixed = TRUE
  )
  expect_error(select_best_row_by_rule(quantile_table, "stability", "median"), "Unknown rule: median", fixed = TRUE)
})

test_that("not_two removes two-cluster rows and fails when none remain", {
  expect_identical(
    select_best_row_by_rule(quantile_table, "stability", "max", not_two = TRUE)$n_clusters,
    3L
  )
  expect_error(
    select_best_row_by_rule(quantile_table[1, ], "stability", "max", not_two = TRUE),
    "No configurations remain after excluding k=2.",
    fixed = TRUE
  )
})

test_that("a measure without any value is an error", {
  df <- data.frame(n_clusters = 2:3, sweep_rank = 0:1, n_clusters_observed = 2:3, ari_stability = NaN)
  expect_error(
    select_best_row_max(df, "stability"),
    "No configuration has a value for ari_stability.",
    fixed = TRUE
  )
})

test_that("every measure alias resolves to a results column", {
  expect_setequal(
    unique(unname(MEASURE_MAP)),
    c("ari_stability", "ari_generalizability", "ari_average", "consensus_pac_stability",
      "consensus_gini_stability", "consensus_ce_stability", "accuracy_generalizability")
  )
  for (alias in names(MEASURE_MAP)) {
    expect_identical(measure_column(alias), MEASURE_MAP[[alias]])
  }
})

test_that("row_to_estimator_params keeps the parameters that have values", {
  row <- data.frame(n_clusters = 3L, linkage = NA, gamma = NaN, method_id = "m0")
  expect_identical(
    row_to_estimator_params(row, c("n_clusters", "linkage", "gamma", "tol")),
    list(n_clusters = 3L)
  )
})
