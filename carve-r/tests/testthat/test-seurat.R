skip_if_not_installed("Seurat")
skip_if_not_installed("ranger")

# pbmc_small is a tiny fixture (80 cells x 230 genes). It actually
# lives in the SeuratObject package (re-exported via Seurat). Keep
# the search package-agnostic so this keeps working across Seurat /
# SeuratObject repackagings.
get_pbmc_small <- function() {
  pkg <- if (requireNamespace("SeuratObject", quietly = TRUE)) {
    "SeuratObject"
  } else {
    "Seurat"
  }
  data_env <- new.env(parent = emptyenv())
  utils::data("pbmc_small", package = pkg, envir = data_env)
  data_env$pbmc_small
}

test_that("carve.Seurat extracts a cells x features matrix and fits", {
  obj <- get_pbmc_small()
  fit <- carve(
    obj,
    assay = "RNA", slot = "data",
    n_clusters = 2:3, n_resamples = 3L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:3))
    ),
    random_state = 1L
  )
  expect_s3_class(fit, "CARVE")
  expect_equal(nrow(fit$X_), ncol(obj))  # CARVE stores cells in rows
})

test_that("carve.Seurat with reduction='pca' uses the embedding", {
  obj <- get_pbmc_small()
  # pbmc_small ships with a PCA reduction already computed.
  fit <- carve(
    obj,
    reduction = "pca", n_dims = 5L,
    n_clusters = 2:3, n_resamples = 3L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:3))
    ),
    random_state = 1L
  )
  expect_s3_class(fit, "CARVE")
  expect_equal(ncol(fit$X_), 5L)
})

test_that("RunCARVE stores the fit on misc$CARVE", {
  obj <- get_pbmc_small()
  obj <- RunCARVE(
    obj, reduction = "pca", n_dims = 5L,
    n_clusters = 2:3, n_resamples = 3L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:3))
    ),
    random_state = 1L
  )
  expect_true(inherits(obj@misc$CARVE, "CARVE"))
})

test_that("AddCarveLabels writes labels to metadata and sets ident", {
  obj <- get_pbmc_small()
  obj <- RunCARVE(
    obj, reduction = "pca", n_dims = 5L,
    n_clusters = 2:3, n_resamples = 3L, subsample_ratio = 0.8,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:3))
    ),
    random_state = 1L
  )
  obj <- AddCarveLabels(obj, measure = "stability", rule = "max",
                        label_col = "carve_k")
  expect_true("carve_k" %in% colnames(obj@meta.data))
  expect_s3_class(obj@meta.data$carve_k, "factor")
  expect_equal(length(obj@meta.data$carve_k), ncol(obj))
})

test_that("AddCarveLabels errors when RunCARVE has not been called", {
  obj <- get_pbmc_small()
  expect_error(AddCarveLabels(obj), "no fitted CARVE")
})
