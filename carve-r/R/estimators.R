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

# StandardScaler: centered columns divided by the population standard
# deviation; a constant column is left at zero.
standard_scale <- function(X) {
  X <- sweep(X, 2L, colMeans(X))
  sds <- sqrt(colMeans(X^2))
  sds[sds == 0] <- 1
  sweep(X, 2L, sds, "/")
}

# sklearn's NearestNeighbors(n_neighbors = k) counts each sample as its own
# first neighbor, so the Python package's k-th neighbor is the (k - 1)-th
# nearest other sample, which is what get.knn returns.
nearest_neighbors <- function(X, n_neighbors) {
  n_neighbors <- as.integer(n_neighbors)
  if (n_neighbors < 2L) {
    stop("n_neighbors must be at least 2.", call. = FALSE)
  }
  if (n_neighbors > nrow(X)) {
    stop(sprintf("n_neighbors=%d is larger than the %d samples.", n_neighbors, nrow(X)), call. = FALSE)
  }
  FNN::get.knn(X, k = n_neighbors - 1L)
}

kth_distance <- function(X, n_neighbors) {
  nearest_neighbors(X, n_neighbors)$nn.dist[, as.integer(n_neighbors) - 1L]
}

knn_sigma <- function(kth) {
  sigma <- stats::median(kth)
  if (sigma == 0) {
    sigma <- if (any(kth > 0)) mean(kth[kth > 0]) else 1
  }
  sigma
}

squared_distances <- function(X) {
  D2 <- sq_dists(X, X)
  diag(D2) <- 0
  D2
}

# Elementwise maximum of two non-negative sparse matrices, keeping sparsity.
pmax_sparse <- function(A, B) {
  (A + B + abs(A - B)) / 2
}

spectral_affinity <- function(X, affinity = "self_tuning", gamma = NULL, n_neighbors = 7L) {
  n <- nrow(X)
  if (identical(affinity, "rbf")) {
    if (is.null(gamma)) {
      gamma <- 1 / (2 * knn_sigma(kth_distance(X, n_neighbors))^2)
    }
    W <- exp(-gamma * squared_distances(X))
    diag(W) <- 0
    return(list(W = W, gamma = gamma))
  }
  if (identical(affinity, "knn")) {
    nn <- nearest_neighbors(X, n_neighbors)
    k <- ncol(nn$nn.index)
    if (is.null(gamma)) {
      gamma <- 1 / (2 * knn_sigma(nn$nn.dist[, k])^2)
    }
    W <- Matrix::sparseMatrix(
      i = rep(seq_len(n), times = k),
      j = as.vector(nn$nn.index),
      x = exp(-gamma * as.vector(nn$nn.dist)^2),
      dims = c(n, n)
    )
    return(list(W = 0.5 * (W + Matrix::t(W)), gamma = gamma))
  }
  if (identical(affinity, "self_tuning")) {
    # Zelnik-Manor and Perona (2004): each sample's scale is its distance to
    # the k-th neighbor, and W(i, j) = exp(-d(i, j)^2 / (sigma_i sigma_j)).
    nn <- nearest_neighbors(X, n_neighbors)
    k <- ncol(nn$nn.index)
    sigma <- nn$nn.dist[, k]
    if (any(sigma == 0)) {
      sigma[sigma == 0] <- if (any(sigma > 0)) stats::median(sigma[sigma > 0]) else 1
    }
    if (n > 5000L) {
      i <- rep(seq_len(n), times = k)
      j <- as.vector(nn$nn.index)
      W <- Matrix::sparseMatrix(
        i = i,
        j = j,
        x = exp(-as.vector(nn$nn.dist)^2 / (sigma[i] * sigma[j])),
        dims = c(n, n)
      )
      return(list(W = pmax_sparse(W, Matrix::t(W)), gamma = NULL))
    }
    W <- exp(-squared_distances(X) / outer(sigma, sigma))
    diag(W) <- 0
    return(list(W = W, gamma = NULL))
  }
  stop(sprintf("Unknown affinity: %s", format_repr(affinity)), call. = FALSE)
}

dense_eigs <- function(L, k) {
  e <- eigen(L, symmetric = TRUE)
  idx <- seq.int(nrow(L) - k + 1L, nrow(L))
  list(values = e$values[idx], vectors = e$vectors[, idx, drop = FALSE])
}

# Shift-invert around 0. Python falls back to a dense solve when ARPACK
# does not converge or the factorization is singular. RSpectra reports
# non-convergence through a warning and fewer converged values, and a
# singular dense factorization can come back as non-finite values without
# an error; each of these stops here and sends spectral_embedding() to the
# dense fallback.
sparse_eigs <- function(L, k) {
  result <- withCallingHandlers(
    RSpectra::eigs_sym(L, k = k, which = "LM", sigma = 0, opts = list(tol = 1e-4, maxitr = 5000L)),
    warning = function(w) invokeRestart("muffleWarning")
  )
  if (is.null(result$nconv) || result$nconv < k ||
      any(!is.finite(result$values)) || any(!is.finite(result$vectors))) {
    stop("The sparse eigensolver did not converge.", call. = FALSE)
  }
  list(values = result$values, vectors = result$vectors)
}

# Eigenvectors of the normalized Laplacian L = I - D^-1/2 W D^-1/2 for the k
# smallest eigenvalues, rows scaled to unit length (Ng, Jordan and Weiss).
spectral_embedding <- function(W, k) {
  n <- nrow(W)
  degree <- if (inherits(W, "Matrix")) Matrix::rowSums(W) else rowSums(W)
  dinv <- 1 / sqrt(pmax(as.numeric(degree), 1e-12))
  if (n < 1000L) {
    Wd <- as.matrix(W)
    L <- diag(n) - dinv * Wd * rep(dinv, each = n)
    L <- 0.5 * (L + t(L))
    eig <- dense_eigs(L, k)
  } else {
    L <- if (inherits(W, "Matrix")) {
      Dinv <- Matrix::Diagonal(x = dinv)
      Matrix::Diagonal(n) - Dinv %*% W %*% Dinv
    } else {
      diag(n) - dinv * W * rep(dinv, each = n)
    }
    eig <- tryCatch(sparse_eigs(L, k), error = function(e) dense_eigs(as.matrix(L), k))
  }
  ord <- order(eig$values)
  vectors <- eig$vectors[, ord, drop = FALSE]
  norms <- sqrt(rowSums(vectors^2))
  list(values = eig$values[ord], vectors = vectors / (norms + 1e-12))
}

#' Spectral clustering
#'
#' Builds an affinity matrix, embeds the samples with the eigenvectors of the
#' normalized graph Laplacian that belong to its `n_clusters` smallest
#' eigenvalues, scales each row to unit length (Ng, Jordan and Weiss), and
#' clusters the embedding with [KMeans()]. It ports the Python package's
#' `SpectralClustering`.
#'
#' `affinity = "self_tuning"` weighs each pair by the two samples' local
#' scales, the distance to their `(n_neighbors - 1)`-th nearest other sample
#' (Zelnik-Manor and Perona, 2004). Above 5,000 samples it keeps only the
#' nearest-neighbor edges. `"rbf"` uses one global `gamma`, and `"knn"`
#' keeps the nearest-neighbor edges with RBF weights. Without a `gamma`, both
#' set it to `1 / (2 * sigma^2)`, with sigma the median distance to the
#' `(n_neighbors - 1)`-th nearest other sample.
#'
#' Below 1,000 samples the eigenvectors come from a dense decomposition. From
#' 1,000 samples on, [RSpectra::eigs_sym()] finds them by shift-invert around
#' zero, and the dense decomposition takes over if that fails.
#'
#' @param X Numeric matrix or data frame, one row per sample.
#' @param n_clusters Number of clusters.
#' @param affinity `"self_tuning"`, `"rbf"` or `"knn"`.
#' @param gamma RBF scale for `"rbf"` and `"knn"`, or `NULL` to set it from the
#'   nearest-neighbor distances. Not used by `"self_tuning"`.
#' @param n_neighbors Neighborhood size for the local scales and the kNN graph,
#'   at least 2. In a [carve()] run it is scaled to each subsample's size.
#' @param n_init Starts of the final k-means.
#' @param scale Standardize each column before computing affinities.
#' @param random_state Seed for the final k-means, or `NULL`.
#' @return Integer labels from 1 to `n_clusters`, one per row of `X`.
#' @seealso [estimator_grid()] to use it in [carve()].
#' @examples
#' t <- seq(0, pi, length.out = 50)
#' X <- rbind(cbind(cos(t), sin(t)), cbind(1 - cos(t), 0.5 - sin(t)))
#' SpectralClustering(X, n_clusters = 2, random_state = 1)
#' @export
SpectralClustering <- function(X, n_clusters = 2L, affinity = "self_tuning", gamma = NULL,
                               n_neighbors = 7L, n_init = 1L, scale = TRUE,
                               random_state = NULL) {
  X <- as_data_matrix(X)
  Xp <- if (isTRUE(scale)) standard_scale(X) else X
  affinity_matrix <- spectral_affinity(Xp, affinity = affinity, gamma = gamma, n_neighbors = n_neighbors)
  embedding <- spectral_embedding(affinity_matrix$W, as.integer(n_clusters))
  KMeans(embedding$vectors, n_clusters = n_clusters, n_init = n_init, random_state = random_state)
}
