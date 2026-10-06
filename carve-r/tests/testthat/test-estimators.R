test_that("kmeans_lloyd reproduces sklearn from the same starting centers", {
  fx <- read_fixture("kmeans")
  X <- fixture_matrix(fx$X)
  # sklearn centers the data before iterating and adds the mean back.
  center <- colMeans(X)
  Xc <- sweep(X, 2L, center)
  tol <- mean(colMeans(Xc^2)) * 1e-4
  for (case in fx$cases) {
    init <- sweep(fixture_matrix(case$init), 2L, center)
    fit <- kmeans_lloyd(Xc, init, max_iter = 300L, tol = tol)
    expect_identical(fit$labels, as.integer(case$labels) + 1L, info = case$name)
    expect_equal(sweep(fit$centers, 2L, center, "+"), fixture_matrix(case$centers), tolerance = 1e-8, info = case$name)
    expect_identical(fit$n_iter, as.integer(case$n_iter), info = case$name)
    expect_equal(fit$inertia, case$inertia, tolerance = 1e-8, info = case$name)
  }
})

test_that("KMeans recovers well-separated blobs and is reproducible", {
  d <- make_blobs()
  labels <- KMeans(d$X, n_clusters = 3L, random_state = 0L)
  expect_identical(adjusted_rand_index(labels, d$y), 1)
  expect_identical(labels, KMeans(d$X, n_clusters = 3L, random_state = 0L))
  expect_setequal(labels, 1:3)
})

test_that("KMeans leaves the caller's RNG state alone", {
  d <- make_blobs()
  set.seed(5)
  before <- .Random.seed
  KMeans(d$X, 3L, random_state = 1L)
  expect_identical(.Random.seed, before)
})

test_that("KMeans needs at least as many samples as clusters", {
  expect_error(
    KMeans(matrix(as.numeric(1:4), 2), n_clusters = 3L),
    "n_samples=2 should be >= n_clusters=3.",
    fixed = TRUE
  )
})

test_that("kmeans_plusplus picks distinct data points as centers", {
  d <- make_blobs()
  centers <- seeded(1L, kmeans_plusplus(d$X, 3L))
  expect_identical(nrow(unique(centers)), 3L)
  is_data_point <- apply(centers, 1L, function(center) any(rowSums(abs(sweep(d$X, 2L, center))) == 0))
  expect_true(all(is_data_point))
})

test_that("AgglomerativeClustering matches sklearn up to label names", {
  fx <- read_fixture("agglomerative")
  X <- fixture_matrix(fx$X)
  for (linkage in names(fx$labels)) {
    labels <- AgglomerativeClustering(X, n_clusters = 4L, linkage = linkage)
    expect_identical(adjusted_rand_index(labels, fx$labels[[linkage]]), 1, info = linkage)
  }
})

test_that("AgglomerativeClustering rejects an unknown linkage", {
  expect_error(
    AgglomerativeClustering(matrix(as.numeric(1:10), 5), linkage = "median"),
    "Unknown linkage: 'median'. Expected 'ward', 'average', 'single' or 'complete'.",
    fixed = TRUE
  )
})

test_that("standard_scale matches StandardScaler", {
  fx <- read_fixture("spectral")
  expect_equal(standard_scale(fixture_matrix(fx$X)), fixture_matrix(fx$scaled), tolerance = 1e-10)
  expect_identical(standard_scale(cbind(c(1, 2, 3), 5))[, 2], c(0, 0, 0))
})

test_that("each affinity and its spectrum match Python", {
  fx <- read_fixture("spectral")
  Xp <- fixture_matrix(fx$scaled)
  settings <- list(
    self_tuning = list(affinity = "self_tuning", gamma = NULL),
    rbf = list(affinity = "rbf", gamma = NULL),
    rbf_gamma = list(affinity = "rbf", gamma = 0.3),
    knn = list(affinity = "knn", gamma = NULL)
  )
  for (name in names(settings)) {
    a <- spectral_affinity(Xp, affinity = settings[[name]]$affinity, gamma = settings[[name]]$gamma, n_neighbors = 7L)
    expect_equal(as.matrix(a$W), fixture_matrix(fx[[name]]$W), tolerance = 1e-10, info = name)
    expect_equal(a$gamma, fx[[name]]$gamma, tolerance = 1e-10, info = name)
    expect_equal(
      spectral_embedding(a$W, 3L)$values,
      fixture_vector(fx[[name]]$evals),
      tolerance = 1e-8,
      info = name
    )
  }
})

test_that("SpectralClustering separates two moons", {
  d <- make_moons(200L)
  expect_gt(adjusted_rand_index(SpectralClustering(d$X, n_clusters = 2L, random_state = 0L), d$y), 0.9)
})

test_that("from 1,000 samples on the sparse solver is used, reproducibly", {
  d <- make_moons(1200L)
  a <- SpectralClustering(d$X, n_clusters = 2L, random_state = 0L)
  expect_identical(a, SpectralClustering(d$X, n_clusters = 2L, random_state = 0L))
  expect_gt(adjusted_rand_index(a, d$y), 0.9)
})

test_that("a failing sparse solve falls back to the dense one", {
  calls <- 0L
  local_mocked_bindings(sparse_eigs = function(L, k) {
    calls <<- calls + 1L
    stop("Factor is exactly singular")
  })
  d <- make_moons(1200L)
  expect_gt(adjusted_rand_index(SpectralClustering(d$X, n_clusters = 2L, random_state = 0L), d$y), 0.9)
  expect_identical(calls, 1L)
})

test_that("the spectral helpers reject impossible settings", {
  X <- matrix(as.numeric(1:20), 10)
  expect_error(spectral_affinity(X, affinity = "cosine"), "Unknown affinity: 'cosine'", fixed = TRUE)
  expect_error(spectral_affinity(X, n_neighbors = 1L), "n_neighbors must be at least 2.", fixed = TRUE)
  expect_error(spectral_affinity(X, n_neighbors = 11L), "n_neighbors=11 is larger than the 10 samples.", fixed = TRUE)
})
