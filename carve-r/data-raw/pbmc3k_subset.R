# Builds data/pbmc3k_subset.rda: 1,000 cells of the PBMC 3k data set from
# 10x Genomics, after the steps of Seurat's PBMC 3k tutorial. Run from
# code/carve-r:
#
#   Rscript data-raw/pbmc3k_subset.R
#
# It needs Seurat and uwot, which the package itself does not use. The
# download, about 7 MB, is kept in data-raw/cache/, which git ignores.

url <- "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
cache <- file.path("data-raw", "cache")
dir.create(cache, showWarnings = FALSE, recursive = TRUE)
archive <- file.path(cache, basename(url))
if (!file.exists(archive)) {
  utils::download.file(url, archive, mode = "wb")
}
matrix_dir <- file.path(cache, "filtered_gene_bc_matrices", "hg19")
if (!dir.exists(matrix_dir)) {
  utils::untar(archive, exdir = cache)
}

# The tutorial's steps, on all cells.
object <- SeuratObject::CreateSeuratObject(
  counts = Seurat::Read10X(matrix_dir),
  project = "pbmc3k", min.cells = 3, min.features = 200
)
object[["percent.mt"]] <- Seurat::PercentageFeatureSet(object, pattern = "^MT-")
meta <- object[[]]
keep <- meta$nFeature_RNA > 200 & meta$nFeature_RNA < 2500 & meta$percent.mt < 5
object <- subset(object, cells = colnames(object)[keep])
object <- Seurat::NormalizeData(object, verbose = FALSE)
object <- Seurat::FindVariableFeatures(object, nfeatures = 2000, verbose = FALSE)
object <- Seurat::ScaleData(object, verbose = FALSE)
object <- Seurat::RunPCA(object, npcs = 30, verbose = FALSE)
object <- Seurat::FindNeighbors(object, dims = 1:10, verbose = FALSE)
object <- Seurat::FindClusters(object, resolution = 0.5, verbose = FALSE)
object <- Seurat::RunUMAP(object, dims = 1:10, verbose = FALSE)

# 1,000 of the cells, in their original order. Four decimals keep the file
# small and change no clustering.
cells <- colnames(object)[sort(withr::with_seed(0L, sample.int(ncol(object), 1000L)))]
pca <- round(SeuratObject::Embeddings(object, reduction = "pca")[cells, ], 4)
umap <- round(SeuratObject::Embeddings(object, reduction = "umap")[cells, ], 4)
colnames(pca) <- paste0("PC_", seq_len(ncol(pca)))
colnames(umap) <- c("UMAP_1", "UMAP_2")
pbmc3k_subset <- list(
  pca = pca,
  umap = umap,
  seurat_clusters = object[[]][cells, "seurat_clusters"]
)
save(pbmc3k_subset, file = file.path("data", "pbmc3k_subset.rda"), compress = "xz")
