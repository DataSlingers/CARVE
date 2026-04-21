# Custom spectral clustering for CARVE. Mirrors cluster.py.
#
# Uses FNN for k-NN, Matrix for sparse structures, RSpectra for sparse
# eigendecomposition (falls back to base::eigen for small n or on sparse
# convergence failure). Labels are 1-based integers.


#' Spectral clustering with self-tuning, RBF, or kNN affinity
#'
#' An R6 estimator that mirrors the Python `SpectralClusteringCARVE`.
#' Builds an affinity matrix, computes the spectral embedding from the
#' normalized graph Laplacian (Ng-Jordan-Weiss), and runs k-means on the
#' embedding.
#'
#' @section Public fields:
#' After `$fit()`:
#' * `labels_` — integer vector of cluster labels (1-based).
#' * `embedding_` — numeric matrix, rows are row-normalized eigenvectors.
#' * `affinity_` — dense or sparse (`Matrix::dgCMatrix`) affinity matrix.
#' * `evals_` — numeric vector of the `n_clusters` smallest eigenvalues.
#' * `gamma_` — resolved RBF gamma (for `"rbf"` / `"knn"`), or `NULL`
#'   (for `"self_tuning"`).
#'
#' @section References:
#' * Zelnik-Manor, L. & Perona, P. (2004). Self-Tuning Spectral Clustering.
#' * Ng, A., Jordan, M. & Weiss, Y. (2002). On spectral clustering.
#' * von Luxburg, U. (2007). A tutorial on spectral clustering.
#'
#' @examples
#' X <- rbind(
#'   matrix(rnorm(40, 0, 0.3), ncol = 2),
#'   matrix(rnorm(40, 3, 0.3), ncol = 2)
#' )
#' sc <- SpectralClusteringCARVE$new(n_clusters = 2L, random_state = 1L)
#' sc$fit(X)
#' sc$labels_
#'
#' @export
SpectralClusteringCARVE <- R6::R6Class(
  "SpectralClusteringCARVE",
  public = list(
    #' @field n_clusters Number of clusters.
    n_clusters = NULL,
    #' @field affinity Affinity type: `"rbf"`, `"knn"`, or `"self_tuning"`.
    affinity = NULL,
    #' @field gamma User-supplied RBF gamma (`NULL` to auto-select).
    gamma = NULL,
    #' @field n_neighbors Number of neighbors for kNN graph / local scale.
    n_neighbors = NULL,
    #' @field random_state Seed for k-means.
    random_state = NULL,
    #' @field n_init Number of k-means initializations.
    n_init = NULL,
    #' @field scale If `TRUE`, standardize `X` before computing affinities.
    scale = NULL,
    #' @field labels_ Integer cluster labels after `$fit()`.
    labels_ = NULL,
    #' @field embedding_ Row-normalized spectral embedding matrix.
    embedding_ = NULL,
    #' @field affinity_ Affinity matrix actually used.
    affinity_ = NULL,
    #' @field evals_ Eigenvalues of the normalized Laplacian.
    evals_ = NULL,
    #' @field gamma_ Resolved gamma value (or `NULL` for self-tuning).
    gamma_ = NULL,

    #' @description Construct a new spectral clustering estimator.
    #' @param n_clusters Integer; number of clusters (default `2`).
    #' @param affinity One of `"rbf"`, `"knn"`, `"self_tuning"`
    #'   (default `"self_tuning"`).
    #' @param gamma RBF kernel scale for `"rbf"`/`"knn"`, or `NULL` for a
    #'   k-NN median heuristic. Ignored for `"self_tuning"`.
    #' @param n_neighbors Integer; number of neighbors (default `7`,
    #'   following Zelnik-Manor & Perona).
    #' @param random_state Integer seed for k-means, or `NULL`.
    #' @param n_init Number of k-means initializations (default `10`).
    #' @param scale If `TRUE` (default), standardize `X` before fitting.
    initialize = function(n_clusters = 2L,
                          affinity = c("self_tuning", "rbf", "knn"),
                          gamma = NULL,
                          n_neighbors = 7L,
                          random_state = NULL,
                          n_init = 10L,
                          scale = TRUE) {
      self$n_clusters <- as.integer(n_clusters)
      self$affinity <- match.arg(affinity)
      self$gamma <- gamma
      self$n_neighbors <- as.integer(n_neighbors)
      self$random_state <- random_state
      self$n_init <- n_init
      self$scale <- isTRUE(scale)
    },

    #' @description Fit the estimator.
    #' @param X Numeric matrix of shape `(n_samples, n_features)`.
    #' @return `self`, invisibly.
    fit = function(X) {
      X <- ensure_2d_matrix(X)
      storage.mode(X) <- "double"
      Xp <- if (self$scale) {
        sc <- base::scale(X, center = TRUE, scale = TRUE)
        # scale() produces NaN columns for zero-variance features; fall back
        # to uncentered/unscaled columns in that case.
        na_cols <- apply(sc, 2L, function(v) any(!is.finite(v)))
        if (any(na_cols)) sc[, na_cols] <- 0
        attributes(sc)[c("scaled:center", "scaled:scale")] <- NULL
        unclass(sc)
      } else {
        X
      }

      W <- private$compute_affinity(Xp)
      self$affinity_ <- W

      spec <- private$spectral_embedding(W, self$n_clusters)
      self$evals_ <- spec$values
      self$embedding_ <- spec$vectors

      km <- .with_seed(self$random_state, {
        stats::kmeans(
          self$embedding_,
          centers = self$n_clusters,
          nstart = self$n_init,
          algorithm = "Hartigan-Wong",
          iter.max = 100L
        )
      })
      self$labels_ <- as.integer(km$cluster)
      invisible(self)
    },

    #' @description Fit the estimator and return cluster labels.
    #' @param X Numeric matrix.
    #' @return Integer vector of cluster labels.
    fit_predict = function(X) {
      self$fit(X)
      self$labels_
    }
  ),

  private = list(
    # k-th neighbor distances + median sigma, with zero-safe fallback.
    knn_sigma = function(X) {
      # FNN::get.knn returns k neighbors *excluding* self; we want the k-th,
      # which corresponds to `k = n_neighbors - 1` to mirror sklearn's
      # NearestNeighbors(n_neighbors = k) semantics (where index 0 is self).
      k_excl <- max(1L, self$n_neighbors - 1L)
      knn <- FNN::get.knn(X, k = k_excl, algorithm = "kd_tree")
      kth <- knn$nn.dist[, k_excl]

      sigma <- stats::median(kth)
      if (sigma == 0) {
        pos <- kth[kth > 0]
        sigma <- if (length(pos) > 0L) mean(pos) else 1
      }
      list(knn = knn, kth = kth, sigma = sigma)
    },

    compute_rbf_affinity = function(X) {
      D2 <- as.matrix(stats::dist(X))^2
      gamma <- if (is.null(self$gamma)) {
        ks <- private$knn_sigma(X)
        1 / (2 * ks$sigma^2)
      } else {
        self$gamma
      }
      self$gamma_ <- gamma
      W <- exp(-gamma * D2)
      diag(W) <- 0
      W
    },

    compute_knn_affinity = function(X) {
      k_excl <- max(1L, self$n_neighbors - 1L)
      knn <- FNN::get.knn(X, k = k_excl, algorithm = "kd_tree")

      gamma <- if (is.null(self$gamma)) {
        kth <- knn$nn.dist[, k_excl]
        sigma <- stats::median(kth)
        if (sigma == 0) {
          pos <- kth[kth > 0]
          sigma <- if (length(pos) > 0L) mean(pos) else 1
        }
        1 / (2 * sigma^2)
      } else {
        self$gamma
      }
      self$gamma_ <- gamma

      n <- nrow(X)
      rows <- rep(seq_len(n), each = k_excl)
      cols <- as.integer(t(knn$nn.index))
      d2 <- as.numeric(t(knn$nn.dist))^2
      vals <- exp(-gamma * d2)

      W <- Matrix::sparseMatrix(
        i = rows, j = cols, x = vals, dims = c(n, n)
      )
      # Symmetrize: (W + W^T) / 2 — matches Python.
      0.5 * (W + Matrix::t(W))
    },

    compute_self_tuning_affinity = function(X) {
      self$gamma_ <- NULL
      k_excl <- max(1L, self$n_neighbors - 1L)
      knn <- FNN::get.knn(X, k = k_excl, algorithm = "kd_tree")

      sigma <- knn$nn.dist[, k_excl]
      if (any(sigma == 0)) {
        pos <- sigma[sigma > 0]
        fill <- if (length(pos) > 0L) stats::median(pos) else 1
        sigma[sigma == 0] <- fill
      }

      n <- nrow(X)
      if (n > 5000L) {
        # Sparse path: kNN edges only, symmetrize by elementwise max.
        rows <- rep(seq_len(n), each = k_excl)
        cols <- as.integer(t(knn$nn.index))
        d2 <- as.numeric(t(knn$nn.dist))^2
        s_i <- sigma[rows]
        s_j <- sigma[cols]
        vals <- exp(-d2 / (s_i * s_j))
        W <- Matrix::sparseMatrix(
          i = rows, j = cols, x = vals, dims = c(n, n)
        )
        pmax_sparse(W, Matrix::t(W))
      } else {
        D2 <- as.matrix(stats::dist(X))^2
        S <- outer(sigma, sigma, `*`)
        W <- exp(-D2 / S)
        diag(W) <- 0
        W
      }
    },

    compute_affinity = function(X) {
      switch(self$affinity,
        rbf = private$compute_rbf_affinity(X),
        knn = private$compute_knn_affinity(X),
        self_tuning = private$compute_self_tuning_affinity(X),
        stop("Unknown affinity: ", sQuote(self$affinity), call. = FALSE)
      )
    },

    spectral_embedding = function(W, k) {
      is_sparse <- inherits(W, "sparseMatrix")
      n <- nrow(W)
      d <- as.numeric(if (is_sparse) Matrix::rowSums(W) else rowSums(W))
      dinvsqrt <- 1 / sqrt(pmax(d, 1e-12))

      # Sparse path is only beneficial when W itself is sparse AND n is large
      # enough for ARPACK to win over dense eigh. Otherwise go dense.
      use_dense <- n < 1000L || !is_sparse

      if (use_dense) {
        Wd <- if (is_sparse) as.matrix(W) else W
        Lsym <- diag(n) - (dinvsqrt * Wd) * rep(dinvsqrt, each = n)
        Lsym <- 0.5 * (Lsym + t(Lsym))
        eig <- base::eigen(Lsym, symmetric = TRUE)
        # eigen(..., symmetric=TRUE) returns values in decreasing order.
        idx <- seq.int(n - k + 1L, n)
        vals <- eig$values[idx]
        vecs <- eig$vectors[, idx, drop = FALSE]
      } else {
        Dinv <- Matrix::Diagonal(x = dinvsqrt)
        Lsym <- Matrix::Diagonal(n) - Dinv %*% W %*% Dinv
        Lsym <- 0.5 * (Lsym + Matrix::t(Lsym))
        res <- tryCatch(
          RSpectra::eigs_sym(
            Lsym, k = k, which = "LM", sigma = 0,
            opts = list(tol = 1e-4, maxitr = 5000L)
          ),
          error = function(e) NULL
        )
        if (is.null(res)) {
          Ld <- as.matrix(Lsym)
          eig <- base::eigen(Ld, symmetric = TRUE)
          idx <- seq.int(n - k + 1L, n)
          vals <- eig$values[idx]
          vecs <- eig$vectors[, idx, drop = FALSE]
        } else {
          vals <- res$values
          vecs <- res$vectors
        }
      }

      order <- order(vals)
      vals <- vals[order]
      vecs <- vecs[, order, drop = FALSE]

      norms <- sqrt(rowSums(vecs^2))
      U <- vecs / (norms + 1e-12)
      list(values = vals, vectors = U)
    }
  )
)


# Elementwise max for two non-negative sparse matrices (affinity weights).
# Uses max(a, b) = (a + b + |a - b|) / 2, which keeps sparsity.
pmax_sparse <- function(A, B) {
  (A + B + abs(A - B)) / 2
}
