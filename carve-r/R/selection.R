# Mirrors _selection.py. The MEASURE_MAP gives short aliases for the canonical
# metric column names found in `estimator_results_`.

MEASURE_MAP <- c(
  "s" = "ari_stability",
  "stab" = "ari_stability",
  "stability" = "ari_stability",
  "ari_stability" = "ari_stability",
  "g" = "ari_generalizability",
  "gen" = "ari_generalizability",
  "generalizability" = "ari_generalizability",
  "ari_generalizability" = "ari_generalizability",
  "avg" = "ari_average",
  "average" = "ari_average",
  "ari_average" = "ari_average",
  "pac" = "consensus_pac_stability",
  "consensus_pac_stability" = "consensus_pac_stability",
  "gini" = "consensus_gini_stability",
  "consensus_gini_stability" = "consensus_gini_stability",
  "ce" = "consensus_ce_stability",
  "consensus_ce_stability" = "consensus_ce_stability",
  "acc" = "accuracy_generalizability",
  "accuracy" = "accuracy_generalizability",
  "accuracy_generalizability" = "accuracy_generalizability"
)


.resolve_measure <- function(measure) {
  if (!measure %in% names(MEASURE_MAP)) {
    stop(sprintf(
      "Invalid measure '%s'. Options: %s",
      measure,
      paste(names(MEASURE_MAP), collapse = ", ")
    ), call. = FALSE)
  }
  unname(MEASURE_MAP[[measure]])
}


#' Select the best row of a CARVE results table by a selection rule
#'
#' Mirrors `carve._selection.select_best_row_by_rule`. Dispatches to
#' [select_best_row_max()], [select_best_row_1se()], or
#' [select_best_row_quantile()].
#'
#' @param results_df A data frame or tibble produced by CARVE's `fit()`. Must
#'   have a `n_clusters` column and the metric column corresponding to
#'   `measure` (plus `_se` / `_upper` / `_lower` columns for 1se / quantile).
#' @param measure Metric key. See [MEASURE_MAP] for accepted aliases.
#' @param rule One of `"max"`, `"1se"`, or `"quantile"`.
#' @param return_idx Logical; if `TRUE` return the selected row's integer
#'   position instead of the row itself.
#' @param not_two Logical; if `TRUE`, rows with `n_clusters == 2` are excluded
#'   from selection.
#'
#' @return A one-row data frame (or the integer position if `return_idx`).
#'
#' @export
select_best_row_by_rule <- function(results_df,
                                    measure,
                                    rule = c("max", "1se", "quantile"),
                                    return_idx = FALSE,
                                    not_two = FALSE) {
  rule <- match.arg(rule)
  if (not_two) {
    kept <- results_df$n_clusters != 2
    results_df <- results_df[kept, , drop = FALSE]
    if (nrow(results_df) == 0L) {
      stop("No configurations remain after excluding k=2.", call. = FALSE)
    }
  }

  y_col <- .resolve_measure(measure)
  se_col <- paste0(y_col, "_se")
  up_col <- paste0(y_col, "_upper")

  if (rule == "1se" && !(se_col %in% names(results_df))) {
    warning(sprintf("Column '%s' not found; falling back to 'max' rule.", se_col),
            call. = FALSE)
    rule <- "max"
  } else if (rule == "quantile" && !(up_col %in% names(results_df))) {
    warning(sprintf("Column '%s' not found; falling back to 'max' rule.", up_col),
            call. = FALSE)
    rule <- "max"
  }

  switch(
    rule,
    "max" = select_best_row_max(results_df, measure = measure, return_idx = return_idx),
    "1se" = select_best_row_1se(results_df, measure = measure, return_idx = return_idx),
    "quantile" = select_best_row_quantile(results_df, measure = measure, return_idx = return_idx)
  )
}


#' Select the best number of clusters from a CARVE results table
#'
#' Convenience wrapper that returns `n_clusters` from the row selected by
#' [select_best_row_by_rule()].
#'
#' @inheritParams select_best_row_by_rule
#'
#' @return Integer number of clusters.
#'
#' @export
select_best_k <- function(results_df,
                          measure = "stability",
                          rule = c("max", "1se", "quantile"),
                          not_two = FALSE) {
  rule <- match.arg(rule)
  row <- select_best_row_by_rule(
    results_df, measure = measure, rule = rule, return_idx = FALSE, not_two = not_two
  )
  as.integer(row$n_clusters)
}


#' Select the best estimator configuration as a list
#'
#' Returns the selected row's estimator name and parameter set. Estimator
#' instantiation (mirroring Python's `build_estimator_from_row`) is deferred
#' to the R6 `CARVE$get_estimator()` method in later phases.
#'
#' @inheritParams select_best_row_by_rule
#' @param k Optional integer. If supplied, filter `results_df` to that
#'   `n_clusters` before selection.
#'
#' @return A list with components `estimator` (character) and `params` (named
#'   list of hyperparameters).
#'
#' @export
select_best_estimator <- function(results_df,
                                  measure = "stability",
                                  rule = c("max", "1se", "quantile"),
                                  k = NULL,
                                  not_two = FALSE) {
  rule <- match.arg(rule)
  if (!is.null(k)) {
    results_df <- results_df[results_df$n_clusters == k, , drop = FALSE]
    if (nrow(results_df) == 0L) {
      stop(sprintf("No configurations with n_clusters == %d.", k), call. = FALSE)
    }
  }
  row <- select_best_row_by_rule(
    results_df, measure = measure, rule = rule, return_idx = FALSE, not_two = not_two
  )
  est_name <- as.character(row$estimator)
  # Strip metric / bookkeeping columns to recover the pure hyperparameter set.
  metric_cols <- c(
    "estimator", "n_clusters",
    outer(
      c("ari_stability", "ari_generalizability", "ari_average",
        "consensus_pac_stability", "consensus_gini_stability",
        "consensus_ce_stability", "accuracy_generalizability"),
      c("", "_se", "_upper", "_lower"),
      paste0
    )
  )
  param_cols <- setdiff(names(row), metric_cols)
  params <- as.list(row[, param_cols, drop = FALSE])
  # Drop NA entries (parameters not applicable to this estimator type).
  params <- params[!vapply(params, function(v) {
    length(v) == 1L && (is.na(v) || is.null(v))
  }, logical(1))]
  list(estimator = est_name, params = params, n_clusters = as.integer(row$n_clusters))
}


#' Select the row with the maximum score for a measure
#'
#' @inheritParams select_best_row_by_rule
#'
#' @return A one-row data frame (or integer position if `return_idx`).
#'
#' @export
select_best_row_max <- function(results_df, measure = "stability", return_idx = FALSE) {
  col <- .resolve_measure(measure)
  idx <- which.max(results_df[[col]])
  if (return_idx) return(idx)
  results_df[idx, , drop = FALSE]
}


#' Select the largest-k row within one standard error of the best score
#'
#' @inheritParams select_best_row_by_rule
#'
#' @return A one-row data frame (or integer position if `return_idx`).
#'
#' @export
select_best_row_1se <- function(results_df, measure = "stability", return_idx = FALSE) {
  col <- .resolve_measure(measure)
  best_i <- which.max(results_df[[col]])
  best_score <- results_df[[col]][best_i]
  se <- results_df[[paste0(col, "_se")]][best_i]
  threshold <- best_score - se

  keep <- which(results_df[[col]] >= threshold)
  ks <- results_df$n_clusters[keep]
  chosen <- keep[which.max(ks)]

  if (return_idx) return(chosen)
  results_df[chosen, , drop = FALSE]
}


#' Select the largest-k row within the best score's quantile bounds
#'
#' @inheritParams select_best_row_by_rule
#'
#' @return A one-row data frame (or integer position if `return_idx`).
#'
#' @export
select_best_row_quantile <- function(results_df, measure = "stability", return_idx = FALSE) {
  col <- .resolve_measure(measure)
  best_i <- which.max(results_df[[col]])
  up <- results_df[[paste0(col, "_upper")]][best_i]
  lo <- results_df[[paste0(col, "_lower")]][best_i]

  keep <- which(results_df[[col]] >= lo & results_df[[col]] <= up)
  if (length(keep) == 0L) {
    warning("No estimators within quantile thresholds; falling back to max.",
            call. = FALSE)
    if (return_idx) return(best_i)
    return(results_df[best_i, , drop = FALSE])
  }

  ks <- results_df$n_clusters[keep]
  chosen <- keep[which.max(ks)]
  if (return_idx) return(chosen)
  results_df[chosen, , drop = FALSE]
}
