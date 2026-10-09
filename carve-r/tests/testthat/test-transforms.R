two <- make_blobs(n_per = 40L, centers = rbind(c(0, 0, 0), c(8, 8, 8)), seed = 3L)

test_that("Identity, StandardScaler and Log1p transform as their names say", {
  X <- cbind(c(1, 2, 3, 6), c(5, 5, 5, 5))
  expect_identical(Identity(X), X)
  Z <- StandardScaler(X)
  expect_equal(colMeans(Z), c(0, 0))
  # The population standard deviation, as scikit-learn divides by n.
  expect_equal(sqrt(mean(Z[, 1]^2)), 1)
  expect_identical(Z[, 2], rep(0, 4))
  expect_identical(Log1p(X), log1p(X))
})

test_that("PCA returns the leading principal component scores", {
  X <- withr::with_seed(2, matrix(stats::rnorm(400), 100) %*% diag(c(5, 3, 1, 0.5)))
  reference <- unname(stats::prcomp(X)$x)
  # One component goes through irlba, three through prcomp.
  expect_equal(abs(PCA(X, n_components = 1L, random_state = 0L)[, 1]), abs(reference[, 1]), tolerance = 1e-4)
  expect_equal(abs(PCA(X, n_components = 3L, random_state = 0L)), abs(reference[, 1:3]), tolerance = 1e-8)
  expect_identical(dim(PCA(X)), c(100L, 4L))
})

test_that("PCA rejects a component count the data cannot give", {
  expect_error(
    PCA(matrix(stats::runif(20), 10), n_components = 3L),
    "n_components=3 must be between 1 and min(n_samples, n_features)=2.",
    fixed = TRUE
  )
})

test_that("TSNE runs Rtsne with scikit-learn's defaults", {
  seen <- NULL
  local_mocked_bindings(
    Rtsne = function(X, ...) {
      seen <<- list(...)
      list(Y = matrix(0, nrow(X), 2L))
    },
    .package = "Rtsne"
  )
  X <- withr::with_seed(1, matrix(stats::rnorm(6000), 3000))
  TSNE(X, perplexity = 10, random_state = 0L)
  expect_identical(seen$dims, 2L)
  expect_identical(seen$perplexity, 10)
  expect_false(seen$pca)
  expect_false(seen$normalize)
  expect_false(seen$check_duplicates)
  expect_identical(seen$max_iter, 1000L)
  # scikit-learn's learning_rate = "auto" is max(n / 48, 50); Rtsne's eta is
  # four times scikit-learn's learning rate.
  expect_identical(seen$eta, 250)
  expect_identical(c(seen$stop_lying_iter, seen$mom_switch_iter), c(250L, 250L))
  expect_identical(dim(seen$Y_init), c(3000L, 2L))
  expect_equal(sqrt(mean(seen$Y_init[, 1]^2)), 1e-4)
})

test_that("TSNE separates two groups and is reproducible", {
  set.seed(1)
  before <- .Random.seed
  Y <- TSNE(two$X, perplexity = 10, random_state = 0L)
  expect_identical(.Random.seed, before)
  expect_identical(dim(Y), c(80L, 2L))
  expect_identical(Y, TSNE(two$X, perplexity = 10, random_state = 0L))
  expect_identical(adjusted_rand_index(KMeans(Y, 2L, random_state = 0L), two$y), 1)
})

test_that("a perplexity too large for the samples is an error", {
  expect_error(
    TSNE(matrix(seq_len(40) / 40, 20), perplexity = 7),
    "perplexity=7 is too large for 20 samples; Rtsne needs 3 * perplexity <= n_samples - 1.",
    fixed = TRUE
  )
})

test_that("UMAP runs uwot with umap-learn's defaults on one thread", {
  skip_if_not_installed("uwot")
  seen <- NULL
  local_mocked_bindings(
    umap = function(X, ...) {
      seen <<- list(...)
      matrix(0, nrow(X), 2L)
    },
    .package = "uwot"
  )
  UMAP(two$X, random_state = 0L)
  expect_identical(seen$n_neighbors, 15L)
  expect_identical(seen$n_components, 2L)
  expect_identical(seen$min_dist, 0.1)
  expect_identical(seen$n_threads, 1L)
  expect_identical(seen$n_sgd_threads, 0L)
  expect_false(seen$verbose)
})

test_that("UMAP is reproducible and leaves the session's stream alone", {
  skip_if_not_installed("uwot")
  set.seed(1)
  before <- .Random.seed
  Y <- UMAP(two$X, n_neighbors = 10L, random_state = 0L)
  expect_identical(.Random.seed, before)
  expect_identical(dim(Y), c(80L, 2L))
  expect_identical(Y, UMAP(two$X, n_neighbors = 10L, random_state = 0L))
})

test_that("UMAP needs the uwot package", {
  local_mocked_bindings(has_package = function(package) FALSE)
  expect_error(
    UMAP(two$X),
    'uwot is required for UMAP. Install it with: install.packages("uwot")',
    fixed = TRUE
  )
})
