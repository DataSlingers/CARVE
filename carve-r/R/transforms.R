# Preprocessing transforms for randomized preprocessing. They stand in for
# the scikit-learn and umap-learn classes the Python package uses: each takes
# the data matrix, its settings and, when it draws, random_state, and returns
# the transformed matrix.

#' Preprocessing transforms
#'
#' Functions that transform a data matrix before clustering, for randomized
#' preprocessing in [carve()]. Each takes the samples as rows and returns a
#' numeric matrix with one row per sample. Pass them to
#' [preprocessing_option()] to build the option lists.
#'
#' `Identity()` returns the data unchanged. `StandardScaler()` centers each
#' column and divides it by its standard deviation, computed with denominator
#' n as scikit-learn does; a constant column becomes zero. `Log1p()` returns
#' `log(1 + X)`, for count data.
#'
#' `PCA()` returns principal component scores of the centered data. It uses
#' [irlba::prcomp_irlba()] when it keeps fewer than half of `min(dim(X))`
#' components and [stats::prcomp()] otherwise.
#'
#' `TSNE()` runs Barnes-Hut t-SNE through [Rtsne::Rtsne()] with the defaults
#' of scikit-learn's `TSNE`: no PCA step before the embedding, a start from
#' the first principal components scaled to a standard deviation of 1e-4,
#' early exaggeration 12 for the first 250 of 1,000 iterations, and the
#' learning rate scikit-learn chooses with `learning_rate = "auto"`, which is
#' `max(n / 12, 200)` in Rtsne's units. Rtsne needs
#' `3 * perplexity <= nrow(X) - 1`.
#'
#' `UMAP()` runs [uwot::umap()] with the defaults of umap-learn's `UMAP`,
#' which include `min_dist = 0.1` (uwot's own default is 0.01). It runs on
#' one thread, so that a seed reproduces the embedding, and it needs the uwot
#' package, which CARVE suggests but does not install.
#'
#' `PCA()` gives scikit-learn's scores up to the sign of each component, and
#' up to the approximation error when a truncated solver computes them
#' (irlba in R, the randomized solver in scikit-learn). `TSNE()` and `UMAP()`
#' embeddings differ from Python's at the same seed: R and Python have
#' different random number generators, and Rtsne and uwot implement the
#' methods separately from scikit-learn and umap-learn.
#'
#' @param X Numeric matrix or data frame, one row per sample.
#' @param n_components Number of dimensions to keep. For `PCA()`, `NULL`
#'   keeps `min(dim(X))`.
#' @param perplexity Perplexity of the t-SNE neighborhoods.
#' @param n_neighbors Neighbors per sample in the UMAP graph. It is not scaled
#'   to the subsample size in a [carve()] run.
#' @param min_dist Smallest distance between embedded points in UMAP.
#' @param random_state Seed, or `NULL` to draw from the session's random
#'   number stream.
#' @return A numeric matrix with one row per row of `X`.
#' @seealso [preprocessing_option()], [preprocessing_results()]
#' @examples
#' X <- rbind(matrix(rnorm(120, 0, 0.3), ncol = 3), matrix(rnorm(120, 3, 0.3), ncol = 3))
#' dim(PCA(X, n_components = 2))
#' dim(TSNE(X, perplexity = 10, random_state = 1))
#' @name transforms
NULL

#' @rdname transforms
#' @export
Identity <- function(X) {
  as_data_matrix(X)
}

#' @rdname transforms
#' @export
StandardScaler <- function(X) {
  standard_scale(as_data_matrix(X))
}

#' @rdname transforms
#' @export
Log1p <- function(X) {
  log1p(as_data_matrix(X))
}

#' @rdname transforms
#' @export
PCA <- function(X, n_components = NULL, random_state = NULL) {
  X <- as_data_matrix(X)
  limit <- min(dim(X))
  k <- if (is.null(n_components)) limit else as.integer(n_components)
  if (k < 1L || k > limit) {
    stop(sprintf(
      "n_components=%d must be between 1 and min(n_samples, n_features)=%d.",
      k, limit
    ), call. = FALSE)
  }
  seeded(random_state, pca_scores(X, k))
}

# Scores of the k leading principal components of the centered data. irlba
# draws its start vector, so callers seed it. Above half of min(dim(X))
# components irlba warns and is slower than a full decomposition.
pca_scores <- function(X, k) {
  if (k < min(dim(X)) / 2) {
    scores <- irlba::prcomp_irlba(X, n = k, center = TRUE, scale. = FALSE)$x
  } else {
    scores <- stats::prcomp(X, center = TRUE, scale. = FALSE, rank. = k)$x
  }
  unname(scores[, seq_len(k), drop = FALSE])
}

#' @rdname transforms
#' @export
TSNE <- function(X, n_components = 2L, perplexity = 30, random_state = NULL) {
  X <- as_data_matrix(X)
  n <- nrow(X)
  if (3 * perplexity > n - 1) {
    stop(sprintf(
      "perplexity=%s is too large for %d samples; Rtsne needs 3 * perplexity <= n_samples - 1.",
      format_param_value(perplexity), n
    ), call. = FALSE)
  }
  seeded(random_state, {
    # scikit-learn starts from the PCA scores scaled so that the first
    # coordinate has standard deviation 1e-4. Rtsne skips early exaggeration
    # when given a start unless stop_lying_iter and mom_switch_iter are set.
    start <- pca_scores(X, n_components)
    start <- start / sqrt(mean((start[, 1L] - mean(start[, 1L]))^2)) * 1e-4
    Rtsne::Rtsne(
      X,
      dims = n_components,
      perplexity = perplexity,
      theta = 0.5,
      pca = FALSE,
      normalize = FALSE,
      check_duplicates = FALSE,
      max_iter = 1000L,
      eta = max(n / 12, 200),
      Y_init = start,
      stop_lying_iter = 250L,
      mom_switch_iter = 250L,
      exaggeration_factor = 12,
      num_threads = 1L,
      verbose = FALSE
    )$Y
  })
}

#' @rdname transforms
#' @export
UMAP <- function(X, n_components = 2L, n_neighbors = 15L, min_dist = 0.1, random_state = NULL) {
  require_package("uwot", "UMAP")
  X <- as_data_matrix(X)
  embedding <- seeded(random_state, uwot::umap(
    X,
    n_neighbors = n_neighbors,
    n_components = n_components,
    min_dist = min_dist,
    n_threads = 1L,
    n_sgd_threads = 0L,
    verbose = FALSE
  ))
  unname(embedding)
}
