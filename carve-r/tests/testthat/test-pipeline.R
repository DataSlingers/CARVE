test_that("build_preprocessing_pipeline with randomize=FALSE is identity", {
  X <- matrix(stats::rnorm(30), 10L, 3L)
  pp <- build_preprocessing_pipeline(
    randomize = FALSE,
    normalization_options = default_normalization_options(),
    dim_reduction_options = list(list(type = "identity", grid = list())),
    seed = 1L
  )
  expect_identical(pp$normalization_name, "Identity")
  expect_identical(pp$dim_reduction_name, "Identity")
  expect_equal(pp$pipeline(X), X)
})

test_that("build_preprocessing_pipeline(randomize=TRUE) samples and applies a pipeline", {
  set.seed(1L)
  X <- matrix(stats::runif(60, 0, 5), 20L, 3L)
  pp <- build_preprocessing_pipeline(
    randomize = TRUE,
    normalization_options = default_normalization_options(),
    dim_reduction_options = list(
      list(type = "identity", grid = list(), name = "Identity"),
      list(type = "pca", grid = list(n_components = 2L), name = "PCA")
    ),
    seed = 42L
  )
  out <- pp$pipeline(X)
  expect_true(is.matrix(out))
  expect_true(pp$normalization_name %in%
              c("Identity", "StandardScaler", "Log1p"))
  expect_true(pp$dim_reduction_name %in% c("Identity", "PCA"))
})

test_that("standardize preprocessor produces zero-mean, unit-variance columns", {
  X <- matrix(stats::rnorm(100, mean = 5, sd = 3), 50L, 2L)
  fn <- build_preprocessor("standardize", list())
  Z <- fn(X)
  expect_equal(colMeans(Z), c(0, 0), tolerance = 1e-10)
  expect_equal(apply(Z, 2L, sd), c(1, 1), tolerance = 1e-10)
})

test_that("pca preprocessor reduces to n_components dimensions", {
  set.seed(2L)
  X <- matrix(stats::rnorm(200), 50L, 4L)
  fn <- build_preprocessor("pca", list(n_components = 2L))
  Z <- fn(X)
  expect_equal(dim(Z), c(50L, 2L))
})

test_that("log1p preprocessor elementwise matches log1p", {
  X <- matrix(c(0, 1, 2, 3), 2L, 2L)
  fn <- build_preprocessor("log1p", list())
  expect_equal(fn(X), log1p(X))
})

test_that("unknown preprocessor type errors informatively", {
  expect_error(build_preprocessor("bogus", list()), "Unknown preprocessor")
})
