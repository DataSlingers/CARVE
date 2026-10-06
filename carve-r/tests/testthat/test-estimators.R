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
