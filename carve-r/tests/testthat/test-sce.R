blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:3))
# The PCA and the logcounts differ, so a test can tell which one was read.
sce <- make_sce(blobs$X, assays = list(logcounts = t(blobs$X * 10)))

test_that("sce_matrix reads the PCA, then the logcounts", {
  expect_equal(unname(sce_matrix(sce)), blobs$X)
  no_pca <- make_sce(blobs$X, reductions = list(), assays = list(logcounts = t(blobs$X * 10)))
  expect_equal(unname(sce_matrix(no_pca)), blobs$X * 10)
  expect_equal(unname(sce_matrix(sce, assay = "logcounts")), blobs$X * 10)
  with_umap <- make_sce(blobs$X, reductions = list(PCA = blobs$X, UMAP = blobs$X * 2))
  expect_equal(unname(sce_matrix(with_umap, reduction = "UMAP")), blobs$X * 2)
  expect_equal(unname(sce_matrix(sce, n_dims = 1)), blobs$X[, 1L, drop = FALSE])
})

test_that("sce_matrix densifies a sparse assay", {
  sparse <- make_sce(blobs$X, reductions = list(), assays = list(logcounts = Matrix::Matrix(t(blobs$X), sparse = TRUE)))
  X <- sce_matrix(sparse)
  expect_true(is.matrix(X))
  expect_equal(unname(X), blobs$X)
})

test_that("sce_matrix warns before clustering many logcounts features", {
  wide <- withr::with_seed(1L, matrix(stats::rnorm(90L * 51L), 90L, 51L))
  no_pca <- make_sce(wide, reductions = list())
  expect_warning(
    X <- sce_matrix(no_pca),
    "No representation given and reducedDim(object, 'PCA') is absent, so CARVE is clustering the 'logcounts' assay directly (51 features).",
    fixed = TRUE
  )
  expect_identical(dim(X), c(90L, 51L))
  expect_no_warning(sce_matrix(no_pca, assay = "logcounts"))
})

test_that("sce_matrix names what it cannot find", {
  expect_error(
    sce_matrix(sce, assay = "logcounts", reduction = "PCA"),
    "Pass at most one of reduction= and assay=; they select the same thing from different places.",
    fixed = TRUE
  )
  expect_error(sce_matrix(sce, assay = "counts"), "assay='counts' not found. Available assays: ['logcounts']", fixed = TRUE)
  expect_error(
    sce_matrix(sce, reduction = "UMAP"),
    "reduction='UMAP' not found in reducedDimNames(object). Available names: ['PCA']. Pass assay=... to cluster an assay.",
    fixed = TRUE
  )
  expect_error(sce_matrix(sce, n_dims = 3), "n_dims=3 exceeds the 2 available components", fixed = TRUE)
  bare <- make_sce(blobs$X, reductions = list(), assays = list(counts = t(blobs$X)))
  expect_error(
    sce_matrix(bare),
    "No representation given, reducedDim(object, 'PCA') is absent and there is no 'logcounts' assay. Pass reduction= or assay=.",
    fixed = TRUE
  )
})

test_that("carve() fits a SingleCellExperiment as it fits the matrix", {
  fit <- carve(sce, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  reference <- carve(blobs$X, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  expect_identical(estimator_results(fit), estimator_results(reference))
  expect_equal(unname(input_data(fit)), blobs$X)
  logcounts <- carve(sce, assay = "logcounts", n_dims = 1, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  expect_equal(unname(input_data(logcounts)), blobs$X[, 1L, drop = FALSE] * 10)
  expect_error(carve(sce, nope = 1, estimator_param_grids = k_grid), "Unknown argument: nope.", fixed = TRUE)
})

test_that("sce columns and records round-trip", {
  labels <- factor(rep(c("1", "2"), 45))
  updated <- sce_set_columns(sce, list(carve = labels, carve_stability = seq_len(90) / 90))
  expect_identical(sce_column(updated, "carve"), labels)
  expect_equal(sce_column(updated, "carve_stability"), seq_len(90) / 90)
  expect_null(sce_column(updated, "nope"))
  expect_true(all(c("carve", "carve_stability") %in% sce_column_names(updated)))
  record <- list(params = list(measure = "stability"))
  stored <- sce_set_record(updated, "carve", record)
  expect_identical(sce_record(stored, "carve"), record)
  expect_null(sce_record(stored, "other"))
})

test_that("sce_basis prefers UMAP, then TSNE, then PCA, then the fallback", {
  X <- blobs$X
  all_three <- make_sce(X, reductions = list(PCA = X, TSNE = X * 3, UMAP = X * 2))
  expect_equal(sce_basis(all_three), X * 2)
  expect_equal(sce_basis(make_sce(X, reductions = list(PCA = X, TSNE = X * 3))), X * 3)
  expect_equal(sce_basis(sce), X)
  custom <- make_sce(X, reductions = list(embedding = X * 4))
  expect_null(sce_basis(custom))
  expect_equal(sce_basis(custom, fallback = "embedding"), X * 4)
  expect_equal(sce_basis(all_three, basis = "TSNE"), X * 3)
  expect_error(
    sce_basis(all_three, basis = "tsne"),
    "basis='tsne' not found in reducedDimNames(object). Available names: ['PCA', 'TSNE', 'UMAP']",
    fixed = TRUE
  )
  one <- make_sce(X, reductions = list(one = X[, 1L, drop = FALSE]))
  expect_error(sce_basis(one, basis = "one"), "reducedDim(object, 'one') has shape (90, 1); a basis needs at least two columns.", fixed = TRUE)
})
