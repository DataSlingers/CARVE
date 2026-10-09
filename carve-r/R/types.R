# Run modes and estimator grid specifications. Mirrors _types.py; the grid
# helpers are the R forms of Python's (EstimatorClass, param_grid) tuples
# and sklearn's ParameterGrid.

resolve_mode <- function(mode) {
  if (!is.character(mode) || length(mode) != 1L ||
      !mode %in% c("default", "stability", "generalizability")) {
    stop(sprintf(
      "Unknown mode: %s. Expected one of 'default', 'stability', 'generalizability'.",
      format_repr(mode)
    ), call. = FALSE)
  }
  list(
    mode = mode,
    run_stability = mode != "generalizability",
    run_generalizability = mode != "stability",
    compute_average_ari = mode == "default"
  )
}

#' Estimator parameter grid
#'
#' Pairs a clustering function with the values to try for each of its
#' arguments. [carve()] evaluates every combination of the values. This is the
#' R form of the `(EstimatorClass, param_grid)` tuples the Python package
#' takes.
#'
#' @param estimator A clustering function. Its first argument is the data
#'   matrix, one row per sample; its other arguments are the parameters. It
#'   returns one integer label per row, with -1 for noise. If it has a
#'   `random_state` argument, CARVE passes each resample's seed to it.
#' @param ... Values to try, each named after an argument of `estimator`:
#'   a vector, or a list for values that are not scalars.
#' @param name Name recorded in the results table. Defaults to the name the
#'   function was passed under, so an anonymous function needs one.
#' @param x A `carve_estimator_grid` object.
#'
#' @return An object of class `carve_estimator_grid`: a list with the
#'   function, its name and the grid.
#' @seealso The built-in estimators: [KMeans()], [AgglomerativeClustering()],
#'   [SpectralClustering()], [LeidenClustering()], [LouvainClustering()] and
#'   [HDBSCAN()].
#' @examples
#' estimator_grid(KMeans, n_clusters = 2:6)
#' estimator_grid(AgglomerativeClustering, n_clusters = 2:6,
#'                linkage = c("ward", "average"))
#' @export
estimator_grid <- function(estimator, ..., name = NULL) {
  if (!is.function(estimator)) {
    stop("estimator must be a function.", call. = FALSE)
  }
  if (is.null(name)) {
    expr <- substitute(estimator)
    if (is.symbol(expr)) {
      name <- as.character(expr)
    } else if (is.call(expr) && as.character(expr[[1L]]) %in% c("::", ":::")) {
      name <- as.character(expr[[3L]])
    } else {
      stop("Pass name= for an estimator that is not a named function.", call. = FALSE)
    }
  }
  grid <- list(...)
  if (length(grid) > 0L && (is.null(names(grid)) || any(names(grid) == ""))) {
    stop("Every grid entry must be named after an argument of the estimator.", call. = FALSE)
  }
  arguments <- names(formals(estimator))
  if (!"..." %in% arguments) {
    unknown <- setdiff(names(grid), arguments[-1L])
    if (length(unknown) > 0L) {
      stop(sprintf(
        "%s has no argument %s.",
        name,
        paste0("'", unknown, "'", collapse = ", ")
      ), call. = FALSE)
    }
  }
  if (any(lengths(grid) == 0L)) {
    stop("Every grid entry needs at least one value.", call. = FALSE)
  }
  grid <- lapply(grid, function(values) if (is.factor(values)) as.character(values) else values)
  structure(list(estimator = estimator, name = name, grid = grid), class = "carve_estimator_grid")
}

#' @rdname estimator_grid
#' @export
print.carve_estimator_grid <- function(x, ...) {
  cat("Estimator grid for ", x$name, "\n", sep = "")
  for (param in names(x$grid)) {
    values <- vapply(as.list(x$grid[[param]]), format_param_value, character(1))
    cat("  ", param, ": ", paste(values, collapse = ", "), "\n", sep = "")
  }
  invisible(x)
}

# sklearn's ParameterGrid order: keys sorted, the last key varying fastest.
# config_id follows this order, so it must match the Python package.
expand_param_grid <- function(grid) {
  if (length(grid) == 0L) {
    return(list(list()))
  }
  keys <- sort(names(grid), method = "radix")
  sizes <- lengths(grid[keys])
  combos <- expand.grid(lapply(rev(sizes), seq_len), KEEP.OUT.ATTRS = FALSE)
  combos <- combos[, rev(seq_along(keys)), drop = FALSE]
  lapply(seq_len(nrow(combos)), function(r) {
    params <- lapply(seq_along(keys), function(j) grid[[keys[j]]][[combos[r, j]]])
    names(params) <- keys
    params
  })
}
