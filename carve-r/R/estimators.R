# Clustering estimators. KMeans and AgglomerativeClustering stand in for the
# scikit-learn classes of the same names; SpectralClustering ports
# cluster.py. Each takes the data matrix first and returns integer labels
# from 1 to n_clusters.

sq_dists <- function(X, Y) {
  D2 <- outer(rowSums(X^2), rowSums(Y^2), "+") - 2 * tcrossprod(X, Y)
  D2[D2 < 0] <- 0
  D2
}

nearest_center <- function(X, centers) {
  max.col(-sq_dists(X, centers), ties.method = "first")
}

# Greedy k-means++, as sklearn's _kmeans_plusplus: the first center is a
# uniform draw; each later one is the best of 2 + floor(log(k)) candidates
# drawn with probability proportional to the squared distance to the
# nearest center so far.
kmeans_plusplus <- function(X, n_clusters) {
  n <- nrow(X)
  n_local_trials <- 2L + as.integer(floor(log(n_clusters)))
  centers <- matrix(0, n_clusters, ncol(X))
  first <- sample.int(n, 1L)
  centers[1L, ] <- X[first, ]
  closest <- sq_dists(X, X[first, , drop = FALSE])[, 1L]
  potential <- sum(closest)
  for (c in seq_len(n_clusters)[-1L]) {
    draws <- stats::runif(n_local_trials) * potential
    candidates <- pmin(findInterval(draws, cumsum(closest), left.open = TRUE) + 1L, n)
    distances <- pmin(sq_dists(X, X[candidates, , drop = FALSE]), closest)
    potentials <- colSums(distances)
    best <- which.min(potentials)
    potential <- potentials[[best]]
    closest <- distances[, best]
    centers[c, ] <- X[candidates[best], ]
  }
  centers
}

# Lloyd's algorithm as sklearn's _kmeans_single_lloyd runs it: stop when the
# labels do not change, or when the summed squared center shift is at most
# tol. A cluster left empty takes the point farthest from its own center. X
# and centers are already centered on the column means.
kmeans_lloyd <- function(X, centers, max_iter, tol) {
  n <- nrow(X)
  k <- nrow(centers)
  labels_old <- rep(-1L, n)
  strict <- FALSE
  for (iteration in seq_len(max_iter)) {
    labels <- nearest_center(X, centers)
    sums <- matrix(0, k, ncol(X))
    present <- sort(unique(labels))
    sums[present, ] <- rowsum(X, labels, reorder = TRUE)
    counts <- tabulate(labels, nbins = k)
    empty <- which(counts == 0L)
    if (length(empty) > 0L) {
      own_distance <- rowSums((X - centers[labels, , drop = FALSE])^2)
      if (max(own_distance) > 0) {
        far <- order(own_distance, decreasing = TRUE)[seq_along(empty)]
        for (e in seq_along(empty)) {
          j <- empty[e]
          p <- far[e]
          old <- labels[p]
          sums[old, ] <- sums[old, ] - X[p, ]
          sums[j, ] <- X[p, ]
          counts[old] <- counts[old] - 1L
          counts[j] <- 1L
        }
      }
    }
    new_centers <- sums
    filled <- counts > 0L
    new_centers[filled, ] <- sums[filled, , drop = FALSE] / counts[filled]
    shift <- sum((new_centers - centers)^2)
    centers <- new_centers
    if (identical(labels, labels_old)) {
      strict <- TRUE
      break
    }
    if (shift <= tol) {
      break
    }
    labels_old <- labels
  }
  if (!strict) {
    labels <- nearest_center(X, centers)
  }
  list(
    labels = labels,
    centers = centers,
    inertia = sum((X - centers[labels, , drop = FALSE])^2),
    n_iter = iteration
  )
}

#' K-means clustering
#'
#' Lloyd's algorithm started from greedy k-means++ seeds, as in
#' scikit-learn's `KMeans` with its default settings. The seeding and the
#' iterations follow scikit-learn's implementation; R draws different random
#' numbers, so the labels are not the ones Python would return.
#'
#' @param X Numeric matrix or data frame, one row per sample.
#' @param n_clusters Number of clusters.
#' @param n_init Number of k-means++ starts. The start with the lowest
#'   within-cluster sum of squares is kept. scikit-learn's default, `"auto"`,
#'   is one start.
#' @param max_iter Maximum number of iterations per start.
#' @param tol A start stops when the summed squared shift of the centers is at
#'   most `tol` times the mean column variance.
#' @param random_state Seed, or `NULL` to draw from the session's random
#'   number stream.
#' @return Integer labels from 1 to `n_clusters`, one per row of `X`.
#' @seealso [estimator_grid()] to use it in [carve()].
#' @examples
#' X <- rbind(matrix(rnorm(40, 0, 0.3), ncol = 2), matrix(rnorm(40, 3, 0.3), ncol = 2))
#' KMeans(X, n_clusters = 2, random_state = 1)
#' @export
KMeans <- function(X, n_clusters = 8L, n_init = 1L, max_iter = 300L, tol = 1e-4,
                   random_state = NULL) {
  X <- as_data_matrix(X)
  n_clusters <- as.integer(n_clusters)
  if (nrow(X) < n_clusters) {
    stop(sprintf("n_samples=%d should be >= n_clusters=%d.", nrow(X), n_clusters), call. = FALSE)
  }
  Xc <- sweep(X, 2L, colMeans(X))
  tol_abs <- if (tol == 0) 0 else mean(colMeans(Xc^2)) * tol
  best <- NULL
  seeded(random_state, {
    for (start in seq_len(n_init)) {
      fit <- kmeans_lloyd(Xc, kmeans_plusplus(Xc, n_clusters), max_iter = max_iter, tol = tol_abs)
      if (is.null(best) || fit$inertia < best$inertia) {
        best <- fit
      }
    }
  })
  best$labels
}

#' Agglomerative clustering
#'
#' Hierarchical clustering of the Euclidean distances with [stats::hclust()],
#' cut into `n_clusters` groups. `linkage = "ward"` runs `"ward.D2"`, the Ward
#' criterion scikit-learn's `AgglomerativeClustering` uses.
#'
#' @param X Numeric matrix or data frame, one row per sample.
#' @param n_clusters Number of clusters.
#' @param linkage `"ward"`, `"average"`, `"single"` or `"complete"`.
#' @return Integer labels from 1 to `n_clusters`, one per row of `X`.
#' @seealso [estimator_grid()] to use it in [carve()].
#' @examples
#' X <- rbind(matrix(rnorm(40, 0, 0.3), ncol = 2), matrix(rnorm(40, 3, 0.3), ncol = 2))
#' AgglomerativeClustering(X, n_clusters = 2)
#' @export
AgglomerativeClustering <- function(X, n_clusters = 2L, linkage = "ward") {
  X <- as_data_matrix(X)
  method <- switch(linkage,
    ward = "ward.D2",
    average = "average",
    single = "single",
    complete = "complete",
    stop(sprintf(
      "Unknown linkage: %s. Expected 'ward', 'average', 'single' or 'complete'.",
      format_repr(linkage)
    ), call. = FALSE)
  )
  tree <- stats::hclust(stats::dist(X), method = method)
  as.integer(stats::cutree(tree, k = n_clusters))
}
