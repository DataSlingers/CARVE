#' PBMC 3k cells
#'
#' 1,000 cells of the PBMC 3k data set from 10x Genomics, peripheral blood
#' mononuclear cells from a healthy donor, after the steps of Seurat's PBMC
#' 3k tutorial. It is small enough for examples that fit CARVE to
#' single-cell data without a download.
#'
#' The steps ran on all 2,700 cells. A cell was kept if it had more than 200
#' and fewer than 2,500 detected genes, and less than 5 percent of its counts
#' came from mitochondrial genes. The counts were log-normalized, the 2,000
#' most variable genes were scaled, and 30 principal components were
#' computed. The clusters come from Seurat's Louvain clustering at resolution
#' 0.5 on a neighbor graph of the first 10 components, and the UMAP from the
#' same 10 components. Then 1,000 cells were drawn at random, and the
#' principal component scores and UMAP coordinates were rounded to four
#' decimals.
#' The script is `data-raw/pbmc3k_subset.R` in the package's source
#' repository.
#'
#' @format A list of three elements, with the cells in the same order in
#'   each:
#' \describe{
#'   \item{pca}{A 1,000 by 30 matrix of principal component scores, with the
#'     cell barcodes as row names.}
#'   \item{umap}{A 1,000 by 2 matrix of UMAP coordinates, with the same row
#'     names.}
#'   \item{seurat_clusters}{A factor, the Seurat cluster of each cell.}
#' }
#' @source 10x Genomics,
#'   <https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz>
#' @examples
#' dim(pbmc3k_subset$pca)
#' table(pbmc3k_subset$seurat_clusters)
"pbmc3k_subset"
