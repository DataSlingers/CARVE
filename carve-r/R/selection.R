# Choosing a configuration from the results table. Mirrors _selection.py.
# The rules order configurations by sweep_rank, so "finest" means the same
# thing on every sweep axis.

MEASURE_MAP <- c(
  s = "ari_stability",
  stab = "ari_stability",
  stability = "ari_stability",
  ari_stability = "ari_stability",
  g = "ari_generalizability",
  gen = "ari_generalizability",
  generalizability = "ari_generalizability",
  ari_generalizability = "ari_generalizability",
  avg = "ari_average",
  average = "ari_average",
  ari_average = "ari_average",
  pac = "consensus_pac_stability",
  consensus_pac_stability = "consensus_pac_stability",
  gini = "consensus_gini_stability",
  consensus_gini_stability = "consensus_gini_stability",
  ce = "consensus_ce_stability",
  consensus_ce_stability = "consensus_ce_stability",
  acc = "accuracy_generalizability",
  accuracy = "accuracy_generalizability",
  accuracy_generalizability = "accuracy_generalizability"
)

measure_column <- function(measure) {
  if (!is.character(measure) || length(measure) != 1L || !measure %in% names(MEASURE_MAP)) {
    stop(sprintf(
      "Invalid measure %s. Options: %s",
      format_repr(measure),
      python_list(names(MEASURE_MAP))
    ), call. = FALSE)
  }
  MEASURE_MAP[[measure]]
}

# The first row with the highest value, as pandas' idxmax picks it.
best_row <- function(results, column) {
  values <- results[[column]]
  if (all(is.na(values))) {
    stop(sprintf("No configuration has a value for %s.", column), call. = FALSE)
  }
  results[which.max(values), , drop = FALSE]
}

finest_row <- function(rows) {
  rows[which.max(rows$sweep_rank), , drop = FALSE]
}

select_best_row_max <- function(results, measure = "stability") {
  best_row(results, measure_column(measure))
}

select_best_row_1se <- function(results, measure = "stability") {
  column <- measure_column(measure)
  best <- best_row(results, column)
  threshold <- best[[column]] - best[[paste0(column, "_se")]]
  within <- results[which(results[[column]] >= threshold), , drop = FALSE]
  # A NaN standard error (one resample) leaves no row within one SE; the
  # best row is the answer then.
  if (nrow(within) == 0L) {
    return(best)
  }
  finest_row(within)
}

select_best_row_quantile <- function(results, measure = "stability") {
  column <- measure_column(measure)
  best <- best_row(results, column)
  upper <- best[[paste0(column, "_upper")]]
  lower <- best[[paste0(column, "_lower")]]
  values <- results[[column]]
  within <- results[which(values >= lower & values <= upper), , drop = FALSE]
  if (nrow(within) == 0L) {
    warning("No estimators within quantile thresholds; falling back to max.", call. = FALSE)
    return(best)
  }
  finest_row(within)
}

select_best_row_by_rule <- function(results, measure, rule, not_two = FALSE) {
  if (not_two) {
    results <- results[!(observed_k_series(results) %in% 2), , drop = FALSE]
    if (nrow(results) == 0L) {
      stop("No configurations remain after excluding k=2.", call. = FALSE)
    }
  }
  column <- measure_column(measure)
  if (identical(rule, "1se") && !paste0(column, "_se") %in% names(results)) {
    warning(sprintf("Column '%s_se' not found; falling back to 'max' rule.", column), call. = FALSE)
    rule <- "max"
  } else if (identical(rule, "quantile") && !paste0(column, "_upper") %in% names(results)) {
    warning(sprintf("Column '%s_upper' not found; falling back to 'max' rule.", column), call. = FALSE)
    rule <- "max"
  }
  switch(rule,
    max = select_best_row_max(results, measure),
    `1se` = select_best_row_1se(results, measure),
    quantile = select_best_row_quantile(results, measure),
    stop(sprintf("Unknown rule: %s", rule), call. = FALSE)
  )
}

# The number of clusters at the selected row: the requested k in k mode,
# the rounded mean observed count on other axes.
select_best_k <- function(results, measure, rule, not_two = FALSE) {
  observed_k(select_best_row_by_rule(results, measure, rule, not_two = not_two))
}

row_to_estimator_params <- function(row, valid_keys) {
  keys <- intersect(valid_keys, names(row))
  params <- lapply(keys, function(key) row[[key]][[1L]])
  names(params) <- keys
  missing <- vapply(params, function(v) length(v) == 1L && is.atomic(v) && is.na(v), logical(1))
  params[!missing]
}
