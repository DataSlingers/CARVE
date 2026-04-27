# Preprocessing pipeline construction. Mirrors _pipeline.py.
#
# A "pipeline" in R is a closure `function(X) X_preprocessed`. It is
# fitted implicitly when called: on first invocation the closure estimates
# parameters (e.g. PCA components) on the full X and returns the
# transformed X. This matches sklearn's `Pipeline.fit_transform(X)`.


#' Build a preprocessing pipeline
#'
#' Mirrors `_pipeline.build_preprocessing_pipeline`. When `randomize =
#' FALSE`, returns an identity pipeline. When `TRUE`, samples one
#' normalization and one dimensionality-reduction option and composes
#' them.
#'
#' @param randomize Logical; randomize preprocessing per call?
#' @param normalization_options List of normalization preprocessor specs.
#' @param dim_reduction_options List of DR preprocessor specs.
#' @param seed Integer seed for sampling.
#'
#' @return A list with elements
#'   * `pipeline` — function(X) → preprocessed X.
#'   * `normalization_params`, `dim_reduction_params` — named lists of
#'     sampled parameter values.
#'   * `normalization_name`, `dim_reduction_name` — display strings.
#'
#' @export
build_preprocessing_pipeline <- function(randomize,
                                         normalization_options,
                                         dim_reduction_options,
                                         seed) {
  if (!isTRUE(randomize)) {
    return(list(
      pipeline = function(X) X,
      normalization_params = list(),
      dim_reduction_params = list(),
      normalization_name = "Identity",
      dim_reduction_name = "Identity"
    ))
  }
  sample_preprocessing_pipeline(
    normalization_options, dim_reduction_options, seed
  )
}


#' Randomly sample a normalization + DR pipeline
#'
#' @param normalization_options,dim_reduction_options Lists of
#'   preprocessor specs.
#' @param seed Integer seed.
#'
#' @return Same shape as [build_preprocessing_pipeline()].
#'
#' @export
sample_preprocessing_pipeline <- function(normalization_options,
                                          dim_reduction_options,
                                          seed) {
  chosen_n <- .with_seed(seed, choose_preprocessor(normalization_options))
  chosen_d <- .with_seed(seed + 1L, choose_preprocessor(dim_reduction_options))

  pipeline <- function(X) {
    X <- ensure_2d_matrix(X)
    X <- chosen_n$fn(X)
    chosen_d$fn(X)
  }

  list(
    pipeline = pipeline,
    normalization_params = chosen_n$params,
    dim_reduction_params = chosen_d$params,
    normalization_name = chosen_n$name,
    dim_reduction_name = chosen_d$name
  )
}


# Randomly pick one option and instantiate it.
# Returns list(fn = function(X), params = list(...), name = "<display>").
choose_preprocessor <- function(options) {
  if (length(options) == 0L) {
    stop("options must be non-empty.", call. = FALSE)
  }
  option <- options[[sample.int(length(options), 1L)]]
  params <- sample_from_grid(option$grid %||% list())
  name <- option$name %||% option$type
  list(
    fn = build_preprocessor(option$type, params),
    params = params,
    name = name
  )
}


sample_from_grid <- function(grid) {
  if (length(grid) == 0L) return(list())
  lapply(grid, function(values) {
    if (length(values) == 1L) values
    else values[[sample.int(length(values), 1L)]]
  })
}


# Dispatch a preprocessor kind + params to a closure.
build_preprocessor <- function(type, params) {
  switch(type,
    identity = function(X) X,
    standardize = function(X) {
      sc <- base::scale(X, center = TRUE, scale = TRUE)
      # Zero-variance columns produce NaN; replace with zeros.
      sc[!is.finite(sc)] <- 0
      attributes(sc)[c("scaled:center", "scaled:scale")] <- NULL
      unclass(sc)
    },
    log1p = function(X) log1p(X),
    pca = {
      n_components <- as.integer(params$n_components %||% 2L)
      function(X) {
        pc <- stats::prcomp(X, center = TRUE, scale. = FALSE,
                            rank. = n_components)
        pc$x
      }
    },
    stop("Unknown preprocessor type: ", sQuote(type), call. = FALSE)
  )
}
