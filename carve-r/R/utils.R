# Shared helpers. Mirrors _utils.py.
#
# Note on indexing: R uses 1-based indexing throughout. Functions that take
# `sample_idx` expect 1-based integer vectors. Python callers bridging via
# reticulate must add 1 to numpy indices before handing them to these
# functions.


#' Temporarily set the RNG seed inside an expression
#'
#' Saves `.Random.seed` on entry, calls `set.seed(seed)`, evaluates `expr`,
#' then restores the original state on exit. If `seed` is `NULL`, `expr` is
#' evaluated under the caller's current RNG state.
#'
#' @param seed Integer or `NULL`.
#' @param expr Expression to evaluate.
#'
#' @return The value of `expr`.
#'
#' @noRd
.with_seed <- function(seed, expr) {
  if (is.null(seed)) {
    return(eval.parent(substitute(expr)))
  }
  had_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (had_seed) {
    old_seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
    on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
  } else {
    on.exit(
      {
        if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
          rm(".Random.seed", envir = globalenv())
        }
      },
      add = TRUE
    )
  }
  set.seed(seed)
  eval.parent(substitute(expr))
}


#' Split indices into a random subsample and its complement
#'
#' Mirrors `_utils.split_subsample_indices`. Returns 1-based integer indices.
#'
#' @param n_samples Total number of samples.
#' @param subsample_ratio Proportion in the subsample. Default `0.8`.
#' @param random_state Integer seed, or `NULL` for the current RNG state.
#'
#' @return A list with components `train` (subsample) and `test` (complement),
#'   each an integer vector of 1-based indices.
#'
#' @export
split_subsample_indices <- function(n_samples,
                                    subsample_ratio = 0.8,
                                    random_state = NULL) {
  train_size <- as.integer(subsample_ratio * n_samples)
  .with_seed(random_state, {
    train <- sample.int(n_samples, size = train_size, replace = FALSE)
  })
  test <- setdiff(seq_len(n_samples), train)
  list(train = train, test = test)
}


#' Coerce a cluster-count specification to an integer vector
#'
#' Mirrors `_utils._coerce_n_clusters`. If `value` is a single integer `K`,
#' returns `2:K`. Otherwise validates the supplied vector.
#'
#' @param value Integer scalar or integer vector.
#'
#' @return An integer vector with all values `>= 2`.
#'
#' @export
coerce_n_clusters <- function(value) {
  if (length(value) == 1L && is.numeric(value)) {
    k <- as.integer(value)
    if (k < 2L) stop("n_clusters int must be >= 2.", call. = FALSE)
    return(seq.int(2L, k))
  }
  if (!is.numeric(value) || !is.null(dim(value))) {
    stop("n_clusters must be a 1D integer vector.", call. = FALSE)
  }
  arr <- as.integer(value)
  if (any(arr != as.numeric(value))) {
    stop("n_clusters array must contain integers.", call. = FALSE)
  }
  if (any(arr < 2L)) stop("All n_clusters values must be >= 2.", call. = FALSE)
  arr
}


#' Summarize ARI scores: mean, standard error, 95th and 5th percentiles
#'
#' Mirrors `_utils._summarize_ari_scores`. NaN values are ignored. The
#' standard error uses `ddof = 1`.
#'
#' @param x Numeric vector, possibly containing `NaN` or `NA`.
#' @param n_resamples Unused; kept for signature parity with Python.
#'
#' @return A numeric vector of length 4: `c(mean, se, q95, q05)`.
#'
#' @export
summarize_ari_scores <- function(x, n_resamples = NULL) {
  arr <- as.numeric(x)
  if (all(is.na(arr))) {
    return(c(mean = NaN, se = NaN, q95 = NaN, q05 = NaN))
  }
  m <- sum(!is.na(arr))
  mu <- mean(arr, na.rm = TRUE)
  se <- if (m > 1L) stats::sd(arr, na.rm = TRUE) / sqrt(m) else NaN
  q95 <- unname(stats::quantile(arr, 0.95, na.rm = TRUE))
  q05 <- unname(stats::quantile(arr, 0.05, na.rm = TRUE))
  c(mean = mu, se = se, q95 = q95, q05 = q05)
}


#' Align cluster labels to a reference via Hungarian assignment
#'
#' Mirrors `_utils.align_cluster_labels`. Builds a contingency matrix, solves
#' the assignment problem with [clue::solve_LSAP()] to maximize agreement,
#' and returns `labels` with the permutation applied. Predicted classes
#' that were not matched (when the predicted label space is larger than the
#' reference) are left unchanged.
#'
#' @param reference_labels Integer vector of reference labels.
#' @param labels Integer vector of labels to align.
#'
#' @return Integer vector of aligned labels with the same length as `labels`.
#'
#' @export
align_cluster_labels <- function(reference_labels, labels) {
  if (length(reference_labels) != length(labels)) {
    stop("reference_labels and labels must have the same length.", call. = FALSE)
  }

  true_classes <- sort(unique(reference_labels))
  pred_classes <- sort(unique(labels))
  n_true <- length(true_classes)
  n_pred <- length(pred_classes)

  ref_f <- factor(reference_labels, levels = true_classes)
  lab_f <- factor(labels, levels = pred_classes)
  cont <- unclass(table(ref_f, lab_f))
  storage.mode(cont) <- "double"
  # Pad to square for clue::solve_LSAP.
  n <- max(n_true, n_pred)
  padded <- matrix(0, n, n)
  padded[seq_len(n_true), seq_len(n_pred)] <- cont

  assignment <- as.integer(clue::solve_LSAP(padded, maximum = TRUE))

  # mapping: a vector indexed by pred_class (as integer position) giving the
  # aligned label. Default to identity for pred_classes unmatched to any true.
  mapping <- setNames(pred_classes, as.character(pred_classes))
  for (i in seq_len(n_true)) {
    j <- assignment[i]
    if (j <= n_pred) {
      mapping[as.character(pred_classes[j])] <- true_classes[i]
    }
  }

  aligned <- unname(mapping[as.character(labels)])
  if (is.integer(reference_labels)) aligned <- as.integer(aligned)
  else if (is.numeric(reference_labels)) aligned <- as.numeric(aligned)
  aligned
}


#' Fit a clustering estimator and return labels
#'
#' Dispatches to a concrete backend by `spec$type`:
#' `"kmeans"` (via [stats::kmeans]), `"agglomerative"` (via [stats::hclust]
#' + [stats::cutree]), or `"spectral"` (via [SpectralClusteringCARVE]).
#'
#' This is the R analogue of `_utils.cluster_labels`. In Python, the
#' estimator *class* is passed; R instead takes a named list of parameters,
#' which is more idiomatic and more reflective-friendly.
#'
#' @param X Numeric matrix of shape `(n_samples, n_features)`.
#' @param spec Named list with a `type` element. All other entries are
#'   parameters forwarded to the backend.
#' @param random_state Integer seed, or `NULL` for the current RNG state.
#'   Passed to the backend where applicable.
#'
#' @return Integer vector of cluster labels (1-based).
#'
#' @examples
#' set.seed(1)
#' X <- rbind(
#'   matrix(rnorm(40, 0, 0.3), ncol = 2),
#'   matrix(rnorm(40, 3, 0.3), ncol = 2)
#' )
#' cluster_labels(X, list(type = "kmeans", n_clusters = 2L))
#'
#' @export
cluster_labels <- function(X, spec, random_state = NULL) {
  X <- ensure_2d_matrix(X)
  if (!is.list(spec) || is.null(spec$type)) {
    stop("spec must be a named list with a 'type' element.", call. = FALSE)
  }
  type <- spec$type
  params <- spec[setdiff(names(spec), "type")]
  k <- params$n_clusters %||% params$k
  if (is.null(k)) {
    stop("spec must include 'n_clusters'.", call. = FALSE)
  }
  k <- as.integer(k)

  switch(type,
    kmeans = {
      nstart <- params$nstart %||% 10L
      iter_max <- params$iter.max %||% 100L
      algorithm <- params$algorithm %||% "Hartigan-Wong"
      km <- .with_seed(random_state, {
        stats::kmeans(
          X, centers = k, nstart = nstart,
          iter.max = iter_max, algorithm = algorithm
        )
      })
      as.integer(km$cluster)
    },
    agglomerative = {
      # sklearn's "ward" linkage = R's "ward.D2" (squared-distance Ward).
      method <- params$linkage %||% "ward.D2"
      if (identical(method, "ward")) method <- "ward.D2"
      metric <- params$metric %||% "euclidean"
      d <- stats::dist(X, method = metric)
      hc <- stats::hclust(d, method = method)
      as.integer(stats::cutree(hc, k = k))
    },
    spectral = {
      args <- params[setdiff(
        names(params),
        c("n_clusters", "k")
      )]
      args$n_clusters <- k
      if (is.null(args$random_state)) args$random_state <- random_state
      sc <- do.call(SpectralClusteringCARVE$new, args)
      sc$fit_predict(X)
    },
    stop("Unknown cluster spec type: ", sQuote(type), call. = FALSE)
  )
}


# Null-coalescing operator (Python's x or y).
`%||%` <- function(a, b) if (is.null(a)) b else a


#' Ensure input is a 2D numeric matrix
#'
#' Mirrors `_utils.ensure_2d_array`. Accepts a matrix, a data frame (converted
#' via `as.matrix`), a sparse `Matrix::dgCMatrix` (densified), or a numeric
#' vector (reshaped to a single-column matrix).
#'
#' @param X Input data.
#'
#' @return A 2D numeric matrix.
#'
#' @export
ensure_2d_matrix <- function(X) {
  if (is.matrix(X)) return(X)
  if (is.data.frame(X)) return(as.matrix(X))
  if (inherits(X, "Matrix")) return(as.matrix(X))
  if (is.numeric(X) && is.null(dim(X))) return(matrix(X, ncol = 1L))
  stop("X must be a matrix, data frame, Matrix, or numeric vector.", call. = FALSE)
}
