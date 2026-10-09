blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:3))

test_that("seurat_matrix reads the pca reduction, then the data layer", {
  object <- make_seurat(blobs$X, data = t(blobs$X * 10))
  expect_equal(unname(seurat_matrix(object)), blobs$X)
  expect_equal(unname(seurat_matrix(object, assay = "RNA")), blobs$X * 10)
  no_pca <- make_seurat(blobs$X, reductions = list(), data = t(blobs$X * 10))
  expect_equal(unname(seurat_matrix(no_pca)), blobs$X * 10)
  with_umap <- make_seurat(blobs$X, reductions = list(pca = blobs$X, umap = blobs$X * 2))
  expect_equal(unname(seurat_matrix(with_umap, reduction = "umap")), blobs$X * 2)
  expect_equal(unname(seurat_matrix(object, n_dims = 1)), blobs$X[, 1L, drop = FALSE])
})

test_that("seurat_matrix warns before clustering many features", {
  wide <- withr::with_seed(1L, matrix(stats::rnorm(90L * 51L), 90L, 51L))
  object <- make_seurat(wide, reductions = list())
  expect_warning(
    X <- seurat_matrix(object),
    "No representation given and the 'pca' reduction is absent, so CARVE is clustering the data layer of assay 'RNA' directly (51 features).",
    fixed = TRUE
  )
  expect_identical(dim(X), c(90L, 51L))
})

test_that("seurat_matrix names what it cannot find", {
  object <- make_seurat(blobs$X)
  expect_error(seurat_matrix(object, assay = "RNA", reduction = "pca"), "Pass at most one of reduction= and assay=", fixed = TRUE)
  expect_error(seurat_matrix(object, assay = "ADT"), "assay='ADT' not found. Available assays: ['RNA']", fixed = TRUE)
  expect_error(
    seurat_matrix(object, reduction = "umap"),
    "reduction='umap' not found in Reductions(object). Available names: ['pca']. Pass assay=... to cluster an assay.",
    fixed = TRUE
  )
  unnormalized <- make_seurat(blobs$X, reductions = list(), data = NULL)
  expect_error(
    seurat_matrix(unnormalized),
    "Assay 'RNA' has no 'data' layer. Normalize it first, for example with Seurat::NormalizeData(), or pass reduction=.",
    fixed = TRUE
  )
})

test_that("carve() fits a Seurat object as it fits the matrix", {
  object <- make_seurat(blobs$X)
  fit <- carve(object, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  reference <- carve(blobs$X, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  expect_identical(estimator_results(fit), estimator_results(reference))
  expect_equal(unname(input_data(fit)), blobs$X)
  one <- carve(object, reduction = "pca", n_dims = 1, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  expect_equal(unname(input_data(one)), blobs$X[, 1L, drop = FALSE])
  expect_error(carve(object, nope = 1, estimator_param_grids = k_grid), "Unknown argument: nope.", fixed = TRUE)
  expect_error(carve(blobs$X, assay = "RNA", estimator_param_grids = k_grid), "Unknown argument: assay.", fixed = TRUE)
})

test_that("Seurat columns and records round-trip, and a record is replaced without a warning", {
  object <- make_seurat(blobs$X)
  labels <- factor(rep(c("1", "2"), 45))
  updated <- seurat_set_columns(object, list(carve = labels, carve_stability = seq_len(90) / 90))
  expect_identical(seurat_column(updated, "carve"), labels)
  expect_equal(seurat_column(updated, "carve_stability"), seq_len(90) / 90)
  expect_null(seurat_column(updated, "nope"))
  expect_true(all(c("carve", "carve_stability") %in% seurat_column_names(updated)))
  record <- list(params = list(measure = "stability"))
  stored <- seurat_set_record(updated, "carve", record)
  expect_identical(seurat_record(stored, "carve"), record)
  expect_null(seurat_record(stored, "other"))
  expect_no_warning(replaced <- seurat_set_record(stored, "carve", list(params = list(measure = "g"))))
  expect_identical(seurat_record(replaced, "carve")$params$measure, "g")
})

test_that("seurat_basis prefers umap, then tsne, then pca, then the fallback", {
  X <- blobs$X
  all_three <- make_seurat(X, reductions = list(pca = X, tsne = X * 3, umap = X * 2))
  expect_equal(seurat_basis(all_three), X * 2)
  expect_equal(seurat_basis(make_seurat(X, reductions = list(pca = X, tsne = X * 3))), X * 3)
  expect_equal(seurat_basis(make_seurat(X)), X)
  custom <- make_seurat(X, reductions = list(harmony = X * 4))
  expect_null(seurat_basis(custom))
  expect_equal(seurat_basis(custom, fallback = "harmony"), X * 4)
  expect_error(
    seurat_basis(all_three, basis = "UMAP"),
    "basis='UMAP' not found in Reductions(object). Available names: ['pca', 'tsne', 'umap']",
    fixed = TRUE
  )
})

test_that("the lists in the error messages are in code point order", {
  local_collation()
  X <- blobs$X
  object <- make_seurat(X, reductions = list(harmony = X, UMAP = X))
  object[["integrated"]] <- SeuratObject::CreateAssay5Object(
    counts = SeuratObject::LayerData(object, assay = "RNA", layer = "counts")
  )
  expect_error(seurat_matrix(object, assay = "ADT"), "Available assays: ['RNA', 'integrated']", fixed = TRUE)
  expect_error(seurat_matrix(object, reduction = "tsne"), "Available names: ['UMAP', 'harmony']. Pass assay=", fixed = TRUE)
  expect_error(seurat_basis(object, basis = "tsne"), "Available names: ['UMAP', 'harmony']", fixed = TRUE)
})

test_that("a data layer split across layers gets the JoinLayers() hint", {
  object <- make_seurat(blobs$X, reductions = list())
  object[["RNA"]] <- split(object[["RNA"]], f = rep(c("a", "b"), length.out = ncol(object)))
  expect_error(
    seurat_matrix(object),
    "Assay 'RNA' keeps its normalized data in split layers ['data.a', 'data.b']. Join them first with SeuratObject::JoinLayers(), or pass reduction=.",
    fixed = TRUE
  )
  expect_error(seurat_matrix(object, assay = "RNA"), "Join them first with SeuratObject::JoinLayers()", fixed = TRUE)
  joined <- SeuratObject::JoinLayers(object)
  expect_equal(unname(seurat_matrix(joined)), blobs$X)
})
