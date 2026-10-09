#' @include AllGenerics.R
NULL

# Seurat input and storage, through SeuratObject, which is suggested. With
# sce.R, mirrors _anndata.py. A Seurat object keeps its cell metadata in
# the meta.data slot, and CARVE keeps its record in the misc slot.

# Reductions the scatter plots look for, in order, when no basis is given.
# Seurat names them so.
SEURAT_BASES <- c("umap", "tsne", "pca")

# methods::is() matches the class name even before SeuratObject is loaded,
# so a Seurat object read with readRDS() is recognized.
is_seurat <- function(object) {
  methods::is(object, "Seurat")
}

# The arguments carve() takes for a Seurat object in its ... .
seurat_selection <- function(assay = NULL, reduction = NULL, n_dims = NULL, ...) {
  check_dots(...)
  list(assay = assay, reduction = reduction, n_dims = n_dims)
}

seurat_data_layer <- function(object, assay) {
  layers <- SeuratObject::Layers(object[[assay]])
  if (!"data" %in% layers) {
    split <- grep("^data[.]", layers, value = TRUE)
    if (length(split) > 0L) {
      stop(sprintf(
        "Assay '%s' keeps its normalized data in split layers %s. Join them first with SeuratObject::JoinLayers(), or pass reduction=.",
        assay, python_list(split)
      ), call. = FALSE)
    }
    stop(sprintf(
      "Assay '%s' has no 'data' layer. Normalize it first, for example with Seurat::NormalizeData(), or pass reduction=.",
      assay
    ), call. = FALSE)
  }
  cells_by_features(SeuratObject::LayerData(object, assay = assay, layer = "data"))
}

seurat_matrix <- function(object, assay = NULL, reduction = NULL, n_dims = NULL) {
  require_package("SeuratObject", "Seurat objects")
  check_one_source(assay, reduction)
  reductions <- as.character(SeuratObject::Reductions(object))
  if (!is.null(assay)) {
    assays <- as.character(SeuratObject::Assays(object))
    if (!assay %in% assays) {
      stop(sprintf(
        "assay=%s not found. Available assays: %s",
        format_repr(assay), python_list(sort(assays, method = "radix"))
      ), call. = FALSE)
    }
    X <- seurat_data_layer(object, assay)
  } else if (!is.null(reduction)) {
    if (!reduction %in% reductions) {
      stop(sprintf(
        "reduction=%s not found in Reductions(object). Available names: %s. Pass assay=... to cluster an assay.",
        format_repr(reduction), python_list(sort(reductions, method = "radix"))
      ), call. = FALSE)
    }
    X <- SeuratObject::Embeddings(object, reduction = reduction)
  } else if ("pca" %in% reductions) {
    X <- SeuratObject::Embeddings(object, reduction = "pca")
  } else {
    default <- SeuratObject::DefaultAssay(object)
    X <- seurat_data_layer(object, default)
    if (ncol(X) > 50L) {
      warning(sprintf(
        "No representation given and the 'pca' reduction is absent, so CARVE is clustering the data layer of assay '%s' directly (%d features). Clustering validation on high-dimensional raw data is slow and usually not what you want; run Seurat::RunPCA() first and pass reduction='pca'.",
        default, ncol(X)
      ), call. = FALSE)
    }
  }
  select_dims(as_data_matrix(X), n_dims)
}

seurat_column_names <- function(object) {
  colnames(methods::slot(object, "meta.data"))
}

seurat_column <- function(object, name) {
  data <- methods::slot(object, "meta.data")
  if (name %in% colnames(data)) data[[name]] else NULL
}

seurat_set_columns <- function(object, columns) {
  data <- data.frame(columns, row.names = colnames(object), check.names = FALSE)
  SeuratObject::AddMetaData(object, metadata = data)
}

seurat_record <- function(object, key) {
  methods::slot(object, "misc")[[key]]
}

# SeuratObject's Misc<- warns whenever it replaces an entry, and
# run_carve() replaces its entry on every call, so the slot is set directly.
seurat_set_record <- function(object, key, record) {
  misc <- methods::slot(object, "misc")
  misc[[key]] <- record
  methods::slot(object, "misc") <- misc
  object
}

seurat_basis <- function(object, basis = NULL, fallback = NULL) {
  reductions <- as.character(SeuratObject::Reductions(object))
  read <- function(name) {
    embedding_2d(SeuratObject::Embeddings(object, reduction = name), sprintf("Embeddings(object, '%s')", name))
  }
  if (!is.null(basis)) {
    if (!basis %in% reductions) {
      stop(sprintf(
        "basis=%s not found in Reductions(object). Available names: %s",
        format_repr(basis), python_list(sort(reductions, method = "radix"))
      ), call. = FALSE)
    }
    return(read(basis))
  }
  for (name in c(SEURAT_BASES, fallback)) {
    if (name %in% reductions) {
      return(read(name))
    }
  }
  NULL
}
