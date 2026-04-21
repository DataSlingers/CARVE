skip_if_not_installed("FNN")
skip_if_not_installed("RSpectra")

make_blobs <- function(n_per = 40L, seed = 1L) {
  set.seed(seed)
  centers <- rbind(c(0, 0), c(6, 0), c(3, 5))
  X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i) {
    matrix(stats::rnorm(n_per * 2L, 0, 0.4), ncol = 2L) +
      matrix(centers[i, ], n_per, 2L, byrow = TRUE)
  }))
  y <- rep(seq_len(nrow(centers)), each = n_per)
  list(X = X, y = y)
}

make_two_moons <- function(n_per = 60L, noise = 0.06, seed = 1L) {
  set.seed(seed)
  theta1 <- stats::runif(n_per, 0, pi)
  theta2 <- stats::runif(n_per, 0, pi)
  X1 <- cbind(cos(theta1), sin(theta1)) +
    stats::rnorm(2L * n_per, 0, noise)
  X2 <- cbind(1 - cos(theta2), 0.5 - sin(theta2)) +
    stats::rnorm(2L * n_per, 0, noise)
  list(
    X = rbind(X1, X2),
    y = rep(c(1L, 2L), each = n_per)
  )
}

ari <- function(a, b) {
  ct <- table(a, b)
  n <- sum(ct)
  sum_ct2 <- sum(choose(ct, 2L))
  sa <- sum(choose(rowSums(ct), 2L))
  sb <- sum(choose(colSums(ct), 2L))
  expected <- sa * sb / choose(n, 2L)
  maxi <- 0.5 * (sa + sb)
  if (maxi == expected) return(1)
  (sum_ct2 - expected) / (maxi - expected)
}

test_that("SpectralClusteringCARVE recovers 3 Gaussian blobs (self_tuning)", {
  for (seed in 1:3) {
    bl <- make_blobs(seed = seed)
    sc <- SpectralClusteringCARVE$new(
      n_clusters = 3L, affinity = "self_tuning", random_state = seed
    )
    sc$fit(bl$X)
    expect_gte(ari(sc$labels_, bl$y), 0.95)
  }
})

test_that("SpectralClusteringCARVE recovers 3 Gaussian blobs (rbf)", {
  for (seed in 1:3) {
    bl <- make_blobs(seed = seed)
    sc <- SpectralClusteringCARVE$new(
      n_clusters = 3L, affinity = "rbf", random_state = seed
    )
    sc$fit(bl$X)
    expect_gte(ari(sc$labels_, bl$y), 0.95)
    expect_true(!is.null(sc$gamma_) && sc$gamma_ > 0)
  }
})

test_that("SpectralClusteringCARVE recovers 3 Gaussian blobs (knn)", {
  for (seed in 1:3) {
    bl <- make_blobs(seed = seed)
    sc <- SpectralClusteringCARVE$new(
      n_clusters = 3L, affinity = "knn", n_neighbors = 10L,
      random_state = seed
    )
    sc$fit(bl$X)
    expect_gte(ari(sc$labels_, bl$y), 0.95)
  }
})

test_that("self_tuning separates two-moons", {
  mn <- make_two_moons(n_per = 100L, noise = 0.05, seed = 42L)
  sc <- SpectralClusteringCARVE$new(
    n_clusters = 2L, affinity = "self_tuning", n_neighbors = 7L,
    random_state = 42L
  )
  sc$fit(mn$X)
  expect_gte(ari(sc$labels_, mn$y), 0.9)
})

test_that("gamma_ is NULL for self_tuning and a positive number otherwise", {
  bl <- make_blobs(seed = 1L)
  sc1 <- SpectralClusteringCARVE$new(n_clusters = 3L, affinity = "self_tuning")
  sc1$fit(bl$X)
  expect_null(sc1$gamma_)

  sc2 <- SpectralClusteringCARVE$new(n_clusters = 3L, affinity = "rbf")
  sc2$fit(bl$X)
  expect_true(is.numeric(sc2$gamma_) && sc2$gamma_ > 0)
})

test_that("explicit gamma overrides the heuristic", {
  bl <- make_blobs(seed = 1L)
  sc <- SpectralClusteringCARVE$new(
    n_clusters = 3L, affinity = "rbf", gamma = 0.5
  )
  sc$fit(bl$X)
  expect_equal(sc$gamma_, 0.5)
})

test_that("fit is reproducible for a fixed random_state", {
  bl <- make_blobs(seed = 1L)
  sc1 <- SpectralClusteringCARVE$new(n_clusters = 3L, random_state = 7L)
  sc1$fit(bl$X)
  sc2 <- SpectralClusteringCARVE$new(n_clusters = 3L, random_state = 7L)
  sc2$fit(bl$X)
  expect_equal(sc1$labels_, sc2$labels_)
})

test_that("fit_predict returns the same labels as fit()$labels_", {
  bl <- make_blobs(seed = 1L)
  sc <- SpectralClusteringCARVE$new(n_clusters = 3L, random_state = 1L)
  labels <- sc$fit_predict(bl$X)
  expect_equal(labels, sc$labels_)
  expect_type(labels, "integer")
})

test_that("cluster_labels dispatches to kmeans / agglomerative / spectral", {
  bl <- make_blobs(seed = 1L)

  lab_km <- cluster_labels(
    bl$X, list(type = "kmeans", n_clusters = 3L, nstart = 10L),
    random_state = 1L
  )
  expect_gte(ari(lab_km, bl$y), 0.95)
  expect_type(lab_km, "integer")

  lab_ag <- cluster_labels(
    bl$X, list(type = "agglomerative", n_clusters = 3L, linkage = "ward")
  )
  expect_gte(ari(lab_ag, bl$y), 0.95)

  lab_sc <- cluster_labels(
    bl$X, list(type = "spectral", n_clusters = 3L, affinity = "self_tuning"),
    random_state = 1L
  )
  expect_gte(ari(lab_sc, bl$y), 0.95)
})

test_that("cluster_labels maps sklearn 'ward' to R 'ward.D2'", {
  bl <- make_blobs(seed = 1L)
  a <- cluster_labels(bl$X, list(type = "agglomerative", n_clusters = 3L,
                                 linkage = "ward"))
  b <- cluster_labels(bl$X, list(type = "agglomerative", n_clusters = 3L,
                                 linkage = "ward.D2"))
  expect_equal(a, b)
})

test_that("cluster_labels errors informatively on bad input", {
  expect_error(cluster_labels(matrix(0, 3, 2), list()), "type")
  expect_error(cluster_labels(matrix(0, 3, 2), list(type = "kmeans")),
               "n_clusters")
  expect_error(cluster_labels(matrix(0, 3, 2),
                              list(type = "bogus", n_clusters = 2L)),
               "Unknown")
})
