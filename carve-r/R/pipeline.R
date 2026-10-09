# Preprocessing options and pipelines: allocation, application and labels.
# Mirrors _pipeline.py. A randomized fit draws one pipeline per resample; a
# pipeline records the two transforms and the values drawn for them, so it
# can be applied to any subset of the data with its own seed.

#' Preprocessing option
#'
#' Pairs a preprocessing transform with the values to draw from for each of
#' its arguments. Under `carve(randomize_preprocessing = TRUE)` each resample
#' gets one normalization option and one dimensionality reduction option, with
#' one value drawn for each argument. This is the R form of the
#' `(TransformerClass, param_grid)` options of the Python package.
#'
#' @param transform A function whose first argument is the data matrix, one
#'   row per sample, and which returns a numeric matrix with one row per
#'   sample, such as [PCA()]. If it has a `random_state` argument, CARVE
#'   passes it a seed derived from the resample.
#' @param ... Values to draw from, each named after an argument of
#'   `transform`: a vector, or a list for values that are not scalars.
#' @param name Name used in pipeline labels. Defaults to the name `transform`
#'   was passed under, with [Identity()] and [Log1p()] written `identity` and
#'   `log1p`, as in the Python package's labels. An anonymous function needs
#'   one.
#' @param x A `carve_preprocessing_option` object.
#' @return An object of class `carve_preprocessing_option`: a list with the
#'   function, its name and the values to draw from.
#' @seealso [transforms], [carve()], [preprocessing_results()]
#' @examples
#' preprocessing_option(PCA, n_components = c(2, 5, 10))
#' preprocessing_option(TSNE, n_components = 2, perplexity = c(15, 30))
#' @export
preprocessing_option <- function(transform, ..., name = NULL) {
  if (!is.function(transform)) {
    stop("transform must be a function.", call. = FALSE)
  }
  if (is.null(name)) {
    expr <- substitute(transform)
    if (is.symbol(expr)) {
      name <- as.character(expr)
    } else if (is.call(expr) && (identical(expr[[1L]], quote(`::`)) || identical(expr[[1L]], quote(`:::`)))) {
      name <- as.character(expr[[3L]])
    } else {
      stop("Pass name= for a transform that is not a named function.", call. = FALSE)
    }
    name <- switch(name, Identity = "identity", Log1p = "log1p", name)
  }
  grid <- list(...)
  if (length(grid) > 0L && (is.null(names(grid)) || any(names(grid) == ""))) {
    stop("Every grid entry must be named after an argument of the transform.", call. = FALSE)
  }
  arguments <- names(formals(transform))
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
  empty <- names(grid)[lengths(grid) == 0L]
  if (length(empty) > 0L) {
    stop(sprintf(
      "The candidate list for %s of %s is empty.",
      format_repr(empty[[1L]]), name
    ), call. = FALSE)
  }
  structure(list(transform = transform, name = name, grid = grid), class = "carve_preprocessing_option")
}

#' @rdname preprocessing_option
#' @export
print.carve_preprocessing_option <- function(x, ...) {
  cat("Preprocessing option ", x$name, "\n", sep = "")
  for (param in names(x$grid)) {
    values <- vapply(as.list(x$grid[[param]]), format_param_value, character(1))
    cat("  ", param, ": ", paste(values, collapse = ", "), "\n", sep = "")
  }
  invisible(x)
}

# The label of a drawn step, as Python's PipelineStep.label renders it: the
# name, then the drawn values in grid order.
step_label <- function(name, params) {
  if (length(params) == 0L) {
    return(name)
  }
  values <- vapply(params, format_param_value, character(1))
  paste0(name, "(", paste0(names(params), "=", values, collapse = ", "), ")")
}

new_pipeline_spec <- function(normalization, dim_reduction) {
  structure(
    list(
      normalization = normalization,
      dim_reduction = dim_reduction,
      label = paste(normalization$label, dim_reduction$label, sep = " | ")
    ),
    class = "carve_pipeline_spec"
  )
}

draw_step <- function(option) {
  params <- lapply(option$grid, function(candidates) {
    candidates <- as.list(candidates)
    candidates[[sample.int(length(candidates), 1L)]]
  })
  list(
    transform = option$transform,
    name = option$name,
    params = params,
    label = step_label(option$name, params)
  )
}

# One pipeline per resample. Every pair of a normalization and a reduction
# option gets floor or ceiling of n_resamples / n_pairs resamples, in a
# shuffled order, and each resample draws its own values. Everything comes
# from one stream seeded with random_state, so the allocation is the same
# for every configuration of a run.
allocate_pipelines <- function(normalization_options, dim_reduction_options, n_resamples,
                               random_state) {
  if (length(normalization_options) == 0L || length(dim_reduction_options) == 0L) {
    stop(
      "randomize_preprocessing=TRUE needs at least one normalization option and one dimensionality reduction option.",
      call. = FALSE
    )
  }
  n_reductions <- length(dim_reduction_options)
  n_pairs <- length(normalization_options) * n_reductions
  specs <- seeded(random_state, kind = "L'Ecuyer-CMRG", code = {
    # Pair p (0-based) is normalization p %/% n_reductions with reduction
    # p %% n_reductions, the order of Python's itertools.product.
    cycle <- (seq_len(n_resamples) - 1L) %% n_pairs
    assigned <- cycle[sample.int(length(cycle))]
    lapply(assigned, function(pair) {
      normalization <- draw_step(normalization_options[[pair %/% n_reductions + 1L]])
      dim_reduction <- draw_step(dim_reduction_options[[pair %% n_reductions + 1L]])
      new_pipeline_spec(normalization, dim_reduction)
    })
  })
  # The label keys the results table, so two different pipelines with one
  # label would pool into a single row.
  labels <- vapply(specs, function(s) s$label, character(1))
  for (label in unique(labels)) {
    same <- specs[labels == label]
    if (!all(vapply(same, identical, logical(1), same[[1L]]))) {
      stop(sprintf(
        "Two different pipelines render as the same label %s. Give the options distinct names with preprocessing_option(name = ).",
        format_repr(label)
      ), call. = FALSE)
    }
  }
  specs
}

#' Apply a stored preprocessing pipeline
#'
#' Returns a function that runs one pipeline of a randomized fit on new data:
#' the normalization step, then the dimensionality reduction step, each with
#' the values drawn for it and with `random_state` when it has that argument.
#' Use it, for example, to embed the full data with the pipeline that scored
#' best in [preprocessing_results()].
#'
#' @param spec A pipeline from [preprocessing_pipelines()].
#' @param random_state Seed for both steps.
#' @param x A `carve_pipeline_spec` object.
#' @param ... Not used.
#' @return A function of one argument, the data matrix, that returns the
#'   transformed matrix.
#' @seealso [preprocessing_pipelines()], [preprocessing_option()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_resamples = 4, random_state = 0, randomize_preprocessing = TRUE,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3)),
#'              normalization_options = list(preprocessing_option(Identity)),
#'              dim_reduction_options = list(preprocessing_option(PCA, n_components = 1)))
#' spec <- preprocessing_pipelines(fit)[[1]]
#' spec
#' embed <- pipeline_from_spec(spec, random_state = 0)
#' dim(embed(X))
#' @export
pipeline_from_spec <- function(spec, random_state) {
  if (!inherits(spec, "carve_pipeline_spec")) {
    stop("spec must be a pipeline from preprocessing_pipelines().", call. = FALSE)
  }
  force(random_state)
  function(X) {
    seeded(random_state, {
      normalized <- apply_step(spec$normalization, as_data_matrix(X), random_state)
      apply_step(spec$dim_reduction, normalized, random_state)
    })
  }
}

#' @rdname pipeline_from_spec
#' @export
print.carve_pipeline_spec <- function(x, ...) {
  cat("Preprocessing pipeline: ", x$label, "\n", sep = "")
  invisible(x)
}

# CARVE's derived seed replaces a seed drawn from the grid, so every fit of
# a pipeline is seeded from its resample, as in Python.
apply_step <- function(step, X, random_state) {
  args <- c(list(X), step$params)
  if ("random_state" %in% names(formals(step$transform))) {
    args["random_state"] <- list(random_state)
  }
  out <- do.call(step$transform, args)
  if (is.data.frame(out)) {
    out <- as.matrix(out)
  }
  if (!is.matrix(out) || !is.numeric(out) || nrow(out) != nrow(X)) {
    stop(sprintf(
      "A preprocessing function must return a numeric matrix with one row per row of X (%d).",
      nrow(X)
    ), call. = FALSE)
  }
  out
}
