#' CARVE: Cluster Analysis with Resampling for Validation and Exploration
#'
#' Stability- and generalizability-based validation of clustering methods via
#' repeated subsampling, consensus matrices, and rule-based selection.
#'
#' The entry point is [CARVE], an R6 estimator class, with S3 dispatch via
#' [carve()] for matrix, Seurat, and SingleCellExperiment inputs.
#'
#' @keywords internal
#' @importFrom stats hclust as.dist quantile sd var prcomp kmeans cutree dist
#'   rnorm runif setNames median
#' @importFrom utils packageVersion head tail
#' @importFrom R6 R6Class
#' @importFrom FNN get.knn
#' @importFrom Matrix sparseMatrix Diagonal t
#' @importFrom RSpectra eigs_sym
#' @importFrom ranger ranger
#' @importFrom future plan multisession sequential
#' @importFrom furrr future_map furrr_options
#' @importFrom progressr with_progress progressor
#' @importFrom ggplot2 .data
#' @importFrom patchwork wrap_plots
"_PACKAGE"
