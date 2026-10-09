#' @include AllClasses.R
NULL

# The sweep axis of a run. Mirrors _sweep.py. A run sweeps exactly one
# parameter: metrics on a k axis and on a resolution axis are not
# comparable, so the two kinds of estimator cannot share a run.

# Bookkeeping and identity columns of the results table. They are not
# estimator parameters and stay out of labels and plot groupings.
SWEEP_META_COLS <- c(
  "sweep_param", "sweep_value", "sweep_rank",
  "n_clusters_observed", "n_clusters_observed_se", "noise_fraction"
)
IDENTITY_COLS <- c("config_id", "method_id", "method_label")

SWEEP_REGISTRY <- list(
  n_clusters = list(finer_is_larger = TRUE, fixes_k = TRUE, label = "Number of Clusters (k)"),
  resolution = list(finer_is_larger = TRUE, fixes_k = FALSE, label = "Resolution"),
  min_cluster_size = list(finer_is_larger = FALSE, fixes_k = FALSE, label = "Minimum Cluster Size")
)

coerce_sweep_values <- function(value, param, expand_scalar = TRUE) {
  if (param == "n_clusters") {
    return(coerce_n_clusters(value, expand_scalar = expand_scalar))
  }
  if (param == "min_cluster_size") {
    if (!is.numeric(value) || !is.null(dim(value)) || anyNA(value) || any(value != round(value))) {
      stop("min_cluster_size values must be integers.", call. = FALSE)
    }
    if (any(value < 2)) {
      stop("All min_cluster_size values must be >= 2.", call. = FALSE)
    }
    return(sort(as.integer(value)))
  }
  if (!is.numeric(value) || !is.null(dim(value))) {
    stop(sprintf("%s must be a numeric vector.", param), call. = FALSE)
  }
  if (anyNA(value) || any(value <= 0)) {
    stop(sprintf("All %s values must be > 0.", param), call. = FALSE)
  }
  sort(as.numeric(value))
}

resolve_sweep <- function(n_clusters = NULL, resolution = NULL, sweep = NULL,
                          sweep_values = NULL, finer_is_larger = NULL) {
  if (!is.null(resolution) && !is.null(sweep) && sweep != "resolution") {
    stop(sprintf(
      "resolution= was given but sweep=%s. Pass sweep_values= instead, or drop resolution=.",
      format_repr(sweep)
    ), call. = FALSE)
  }
  if (is.null(sweep)) {
    sweep <- if (!is.null(resolution)) "resolution" else "n_clusters"
  }
  # A single n_clusters value K means 2:K; a value read from sweep_values
  # (an estimator grid) is kept as it is.
  expand_scalar <- is.null(sweep_values)
  if (is.null(sweep_values)) {
    if (sweep == "n_clusters") {
      sweep_values <- n_clusters
    } else if (sweep == "resolution") {
      sweep_values <- resolution
    }
  }
  if (is.null(sweep_values)) {
    stop(sprintf(
      "No values supplied for sweep parameter %s. Pass sweep_values=, or supply estimator grids that contain it.",
      format_repr(sweep)
    ), call. = FALSE)
  }
  values <- coerce_sweep_values(sweep_values, sweep, expand_scalar = expand_scalar)
  known <- SWEEP_REGISTRY[[sweep]]
  if (is.null(known)) {
    if (is.null(finer_is_larger)) {
      stop(sprintf(
        "Unknown sweep parameter %s. Pass finer_is_larger= to declare whether larger values yield more clusters.",
        format_repr(sweep)
      ), call. = FALSE)
    }
    known <- list(
      finer_is_larger = isTRUE(finer_is_larger),
      fixes_k = FALSE,
      label = title_case(gsub("_", " ", sweep, fixed = TRUE))
    )
  }
  methods::new(
    "SweepSpec",
    param = sweep,
    values = values,
    finer_is_larger = if (is.null(finer_is_larger)) known$finer_is_larger else isTRUE(finer_is_larger),
    fixes_k = known$fixes_k,
    label = known$label
  )
}

# Coarse-to-fine rank of each value, 0-based as in Python.
sweep_ranks <- function(spec) {
  keys <- if (spec@finer_is_larger) spec@values else -spec@values
  ranks <- integer(length(keys))
  ranks[order(keys)] <- seq_along(keys) - 1L
  ranks
}

# numpy.isclose with its default tolerances.
isclose <- function(a, b, rtol = 1e-5, atol = 1e-8) {
  abs(a - b) <= atol + rtol * abs(b)
}

sweep_rank_of <- function(spec, value) {
  matches <- which(isclose(as.numeric(spec@values), as.numeric(value)))
  if (length(matches) == 0L) {
    stop(sprintf(
      "%s is not among the swept %s values [%s].",
      format_repr(value),
      spec@param,
      paste(vapply(spec@values, format_repr, character(1)), collapse = ", ")
    ), call. = FALSE)
  }
  sweep_ranks(spec)[matches[1L]]
}

infer_sweep_param <- function(estimator_grids) {
  swept <- unique(unlist(lapply(estimator_grids, function(g) {
    params <- intersect(names(g$grid), names(SWEEP_REGISTRY))
    params[lengths(g$grid[params]) > 1L]
  })))
  if (length(swept) == 0L) {
    return(NULL)
  }
  if (length(swept) > 1L) {
    stop(sprintf(
      "Estimator grids sweep more than one parameter (%s). CARVE evaluates exactly one sweep axis per run; k-based and resolution-based estimators cannot be compared in the same run.",
      python_list(sort(swept, method = "radix"))
    ), call. = FALSE)
  }
  swept
}

grid_sweep_values <- function(estimator_grids, param) {
  for (g in estimator_grids) {
    if (param %in% names(g$grid)) {
      return(g$grid[[param]])
    }
  }
  NULL
}

validate_grids <- function(estimator_grids, sweep) {
  if (length(estimator_grids) == 0L) {
    stop("estimator_param_grids must not be empty.", call. = FALSE)
  }
  others <- setdiff(names(SWEEP_REGISTRY), sweep@param)
  for (g in estimator_grids) {
    if (!sweep@param %in% names(g$grid)) {
      stop(sprintf(
        "Estimator grid for %s does not contain the sweep parameter %s. All grids in a run must sweep the same parameter.",
        g$name, format_repr(sweep@param)
      ), call. = FALSE)
    }
    clash <- intersect(others, names(g$grid))
    clash <- clash[lengths(g$grid[clash]) > 1L]
    if (length(clash) > 0L) {
      stop(sprintf(
        "Estimator grid for %s mixes sweep parameters %s and %s. CARVE evaluates exactly one sweep axis per run; k-based and resolution-based estimators cannot be compared in the same run.",
        g$name, format_repr(sweep@param), python_list(sort(clash, method = "radix"))
      ), call. = FALSE)
    }
  }
  reference <- estimator_grids[[1L]]$grid[[sweep@param]]
  for (g in estimator_grids[-1L]) {
    if (!identical(as.numeric(reference), as.numeric(g$grid[[sweep@param]]))) {
      stop(sprintf(
        "All estimator parameter grids must contain the same %s values.",
        sweep@param
      ), call. = FALSE)
    }
  }
  coerce_sweep_values(reference, sweep@param, expand_scalar = FALSE)
}

# Results-table helpers.
sweep_param_name <- function(results) {
  as.character(results$sweep_param[[1L]])
}

sweep_exclude_cols <- function(results) {
  c(SWEEP_META_COLS, IDENTITY_COLS, sweep_param_name(results))
}

sweep_axis_label <- function(results) {
  param <- sweep_param_name(results)
  known <- SWEEP_REGISTRY[[param]]
  if (is.null(known)) title_case(gsub("_", " ", param, fixed = TRUE)) else known$label
}

observed_k_series <- function(results) {
  round(results$n_clusters_observed)
}

observed_k <- function(row) {
  as.integer(round(as.numeric(row$n_clusters_observed[[1L]])))
}

config_id_of <- function(row) {
  as.integer(row$config_id[[1L]])
}

# A method is an estimator with every parameter except the swept one: one
# curve in a metric-over-sweep plot.
format_method_label <- function(est_name, params, sweep_param) {
  keys <- setdiff(sort(as.character(names(params)), method = "radix"), sweep_param)
  parts <- vapply(
    keys,
    function(key) paste0(key, "=", format_param_value(params[[key]])),
    character(1),
    USE.NAMES = FALSE
  )
  paste(c(est_name, parts), collapse = ", ")
}

# Two configurations that differ only in the swept parameter share a method
# id. Ids are numbered m0, m1, ... in first-seen order.
method_id_assigner <- function(sweep_param) {
  seen <- new.env(parent = emptyenv())
  count <- 0L
  function(est_name, params) {
    keys <- setdiff(sort(as.character(names(params)), method = "radix"), sweep_param)
    values <- vapply(
      keys,
      function(key) paste(deparse(params[[key]]), collapse = ""),
      character(1),
      USE.NAMES = FALSE
    )
    key <- paste(est_name, paste(keys, values, sep = "=", collapse = ";"), sep = "|")
    if (is.null(seen[[key]])) {
      seen[[key]] <- c(sprintf("m%d", count), format_method_label(est_name, params, sweep_param))
      count <<- count + 1L
    }
    seen[[key]]
  }
}

#' @rdname SweepSpec-class
#' @param object A `SweepSpec` object.
#' @export
setMethod("show", "SweepSpec", function(object) {
  direction <- if (object@finer_is_larger) "larger" else "smaller"
  cat(
    "SweepSpec: ", object@param, " over ",
    paste(vapply(object@values, format_param_value, character(1)), collapse = ", "),
    " (", direction, " values give more clusters)\n",
    sep = ""
  )
  invisible(object)
})
