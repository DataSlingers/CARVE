test_that("pbmc3k_subset holds 1,000 cells in the documented layout", {
  data <- pbmc3k_subset
  expect_named(data, c("pca", "umap", "seurat_clusters"))
  expect_identical(dim(data$pca), c(1000L, 30L))
  expect_identical(dim(data$umap), c(1000L, 2L))
  expect_identical(colnames(data$pca), paste0("PC_", 1:30))
  expect_identical(colnames(data$umap), c("UMAP_1", "UMAP_2"))
  expect_identical(rownames(data$umap), rownames(data$pca))
  expect_false(anyDuplicated(rownames(data$pca)) > 0L)
  expect_s3_class(data$seurat_clusters, "factor")
  expect_length(data$seurat_clusters, 1000L)
  expect_gt(nlevels(droplevels(data$seurat_clusters)), 1L)
  expect_true(all(is.finite(data$pca)) && all(is.finite(data$umap)))
})

test_that("pbmc3k_subset's scores are rounded to four decimals", {
  expect_identical(pbmc3k_subset$pca, round(pbmc3k_subset$pca, 4))
  expect_identical(pbmc3k_subset$umap, round(pbmc3k_subset$umap, 4))
})
