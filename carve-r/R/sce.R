#' @include AllGenerics.R
NULL

# SingleCellExperiment input and storage. With seurat.R, mirrors
# _anndata.py: which matrix CARVE clusters, and where run_carve() and
# attach_results() write their results.

#' @importClassesFrom SingleCellExperiment SingleCellExperiment
#' @importFrom SummarizedExperiment colData colData<-
#' @importFrom S4Vectors metadata metadata<-
NULL

# Reduced dimensions the scatter plots look for, in order, when no basis is
# given. scater and scran name them so.
SCE_BASES <- c("UMAP", "TSNE", "PCA")

sce_matrix <- function(object, assay = NULL, reduction = NULL, n_dims = NULL) {
  check_one_source(assay, reduction)
  assays <- as.character(SummarizedExperiment::assayNames(object))
  reductions <- as.character(SingleCellExperiment::reducedDimNames(object))
  if (!is.null(assay)) {
    if (!assay %in% assays) {
      stop(sprintf(
        "assay=%s not found. Available assays: %s",
        format_repr(assay), python_list(sort(assays))
      ), call. = FALSE)
    }
    X <- cells_by_features(SummarizedExperiment::assay(object, assay))
  } else if (!is.null(reduction)) {
    if (!reduction %in% reductions) {
      stop(sprintf(
        "reduction=%s not found in reducedDimNames(object). Available names: %s. Pass assay=... to cluster an assay.",
        format_repr(reduction), python_list(sort(reductions))
      ), call. = FALSE)
    }
    X <- SingleCellExperiment::reducedDim(object, reduction)
  } else if ("PCA" %in% reductions) {
    X <- SingleCellExperiment::reducedDim(object, "PCA")
  } else if ("logcounts" %in% assays) {
    X <- cells_by_features(SummarizedExperiment::assay(object, "logcounts"))
    if (ncol(X) > 50L) {
      warning(sprintf(
        "No representation given and reducedDim(object, 'PCA') is absent, so CARVE is clustering the 'logcounts' assay directly (%d features). Clustering validation on high-dimensional raw data is slow and usually not what you want; compute a PCA first, for example with scater::runPCA(), and pass reduction='PCA'.",
        ncol(X)
      ), call. = FALSE)
    }
  } else {
    stop(
      "No representation given, reducedDim(object, 'PCA') is absent and there is no 'logcounts' assay. Pass reduction= or assay=.",
      call. = FALSE
    )
  }
  select_dims(as_data_matrix(X), n_dims)
}

sce_column_names <- function(object) {
  colnames(colData(object))
}

sce_column <- function(object, name) {
  data <- colData(object)
  if (name %in% colnames(data)) data[[name]] else NULL
}

sce_set_columns <- function(object, columns) {
  data <- colData(object)
  for (name in names(columns)) {
    data[[name]] <- columns[[name]]
  }
  colData(object) <- data
  object
}

sce_record <- function(object, key) {
  metadata(object)[[key]]
}

sce_set_record <- function(object, key, record) {
  metadata(object)[[key]] <- record
  object
}

sce_basis <- function(object, basis = NULL, fallback = NULL) {
  reductions <- as.character(SingleCellExperiment::reducedDimNames(object))
  read <- function(name) {
    embedding_2d(SingleCellExperiment::reducedDim(object, name), sprintf("reducedDim(object, '%s')", name))
  }
  if (!is.null(basis)) {
    if (!basis %in% reductions) {
      stop(sprintf(
        "basis=%s not found in reducedDimNames(object). Available names: %s",
        format_repr(basis), python_list(sort(reductions))
      ), call. = FALSE)
    }
    return(read(basis))
  }
  for (name in c(SCE_BASES, fallback)) {
    if (name %in% reductions) {
      return(read(name))
    }
  }
  NULL
}

#' @rdname carve
#' @param assay,reduction,n_dims For a SingleCellExperiment, which data to
#'   cluster: the assay named `assay` or the reduced dimensions named
#'   `reduction`, not both. By default CARVE clusters the `"PCA"` reduced
#'   dimensions, or the `"logcounts"` assay when there are none. `n_dims`
#'   keeps the first `n_dims` columns.
#' @export
setMethod("carve", "SingleCellExperiment", function(x, assay = NULL, reduction = NULL,
                                                    n_dims = NULL, ...) {
  carve(sce_matrix(x, assay = assay, reduction = reduction, n_dims = n_dims), ...)
})
