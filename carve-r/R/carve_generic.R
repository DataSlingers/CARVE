# S3 generic carve() + Seurat / SingleCellExperiment adapters.
#
# R users coming from base ML tools expect a verb-style function call
# (`kmeans(X, 3)`, `hclust(d)`, `ranger(y ~ ., data)`). We mirror that
# with an S3 generic so `carve(X, ...)` returns a fitted CARVE. Seurat
# users additionally expect a Run*() verb and a downstream AddLabels()
# helper, so we provide both.


#' Fit a CARVE validator
#'
#' Convenience wrapper around `CARVE$new(...)$fit(x, ...)`. Dispatches
#' on the class of `x` so `carve()` works uniformly on matrices, data
#' frames, `Seurat` objects, and `SingleCellExperiment` objects.
#'
#' @param x A numeric matrix, data frame, `Seurat` object, or
#'   `SingleCellExperiment` object.
#' @param ... Passed to the method.
#'
#' @return A fitted [CARVE] instance.
#'
#' @examples
#' set.seed(1)
#' X <- rbind(
#'   matrix(rnorm(60, 0, 0.4), ncol = 2),
#'   matrix(rnorm(60, 5, 0.4), ncol = 2)
#' )
#' fit <- carve(X, n_clusters = 2:3, n_resamples = 5,
#'              estimator_param_grids = list(
#'                list(type = "kmeans", grid = list(n_clusters = 2:3))
#'              ),
#'              random_state = 1L)
#' fit$get_k(measure = "stability", rule = "max")
#'
#' @export
carve <- function(x, ...) {
  UseMethod("carve")
}

#' @rdname carve
#' @param n_clusters,n_resamples,subsample_ratio,estimator_param_grids,normalization_options,dim_reduction_options,classifier,n_trees,reference_labels,n_jobs,random_state,verbose
#'   Forwarded to [`CARVE$new()`][CARVE].
#' @param reference_labels_fit Optional reference labels passed to `$fit()`.
#' @param randomize_preprocessing,show_progress,mode,fit_random_state
#'   Forwarded to [`$fit()`][CARVE].
#' @export
carve.default <- function(x,
                          n_clusters = 2:10,
                          n_resamples = 100L,
                          subsample_ratio = 0.8,
                          estimator_param_grids = "light",
                          normalization_options = NULL,
                          dim_reduction_options = NULL,
                          classifier = NULL,
                          n_trees = 100L,
                          reference_labels = NULL,
                          n_jobs = 1L,
                          random_state = NULL,
                          verbose = 0L,
                          reference_labels_fit = NULL,
                          randomize_preprocessing = FALSE,
                          show_progress = FALSE,
                          mode = "default",
                          fit_random_state = NULL,
                          ...) {
  X <- ensure_2d_matrix(x)
  obj <- CARVE$new(
    n_clusters = n_clusters,
    n_resamples = n_resamples,
    subsample_ratio = subsample_ratio,
    estimator_param_grids = estimator_param_grids,
    normalization_options = normalization_options,
    dim_reduction_options = dim_reduction_options,
    classifier = classifier,
    n_trees = n_trees,
    reference_labels = reference_labels,
    n_jobs = n_jobs,
    random_state = random_state,
    verbose = verbose
  )
  obj$fit(
    X = X,
    reference_labels = reference_labels_fit,
    randomize_preprocessing = randomize_preprocessing,
    show_progress = show_progress,
    mode = mode,
    random_state = fit_random_state
  )
  obj
}

#' @rdname carve
#' @export
carve.matrix <- carve.default

#' @rdname carve
#' @export
carve.data.frame <- function(x, ...) {
  numeric_cols <- vapply(x, is.numeric, logical(1L))
  if (!all(numeric_cols)) {
    stop("carve.data.frame: all columns must be numeric.", call. = FALSE)
  }
  carve.default(as.matrix(x), ...)
}

#' @rdname carve
#' @param assay For `Seurat` / `SingleCellExperiment` inputs, the assay
#'   to extract (default `"RNA"` for Seurat, `"logcounts"` for SCE).
#' @param slot For `Seurat` inputs, the slot/layer (default `"data"`).
#'   Gated on `packageVersion("Seurat")` so both v4 (`slot`) and v5
#'   (`layer`) APIs work.
#' @param features Optional character vector of features/rows to use.
#' @param reduction Optional reduction name (e.g. `"pca"`); when set,
#'   the embedding matrix is used instead of the assay data.
#' @param n_dims Optional integer; number of leading components of the
#'   reduction to keep.
#' @export
carve.Seurat <- function(x,
                         assay = "RNA",
                         slot = "data",
                         features = NULL,
                         reduction = NULL,
                         n_dims = NULL,
                         ...) {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("Package 'Seurat' is required for carve.Seurat(). ",
         "Install it from CRAN with install.packages(\"Seurat\").",
         call. = FALSE)
  }
  X <- .seurat_extract_matrix(x, assay = assay, slot = slot,
                              features = features, reduction = reduction,
                              n_dims = n_dims)
  carve.default(X, ...)
}

#' @rdname carve
#' @export
carve.SingleCellExperiment <- function(x,
                                        assay = "logcounts",
                                        reduction = NULL,
                                        n_dims = NULL,
                                        features = NULL,
                                        ...) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("Package 'SingleCellExperiment' is required for ",
         "carve.SingleCellExperiment(). Install from Bioconductor.",
         call. = FALSE)
  }
  X <- .sce_extract_matrix(x, assay = assay, reduction = reduction,
                           n_dims = n_dims, features = features)
  carve.default(X, ...)
}


#' Run CARVE on a Seurat object (verb-style alias)
#'
#' Seurat users expect `Run*()` verbs that store results on the object.
#' `RunCARVE()` fits a [CARVE] validator and attaches it to
#' `object@misc$CARVE` for later retrieval by [AddCarveLabels()].
#'
#' @param object A `Seurat` object.
#' @param assay,slot,features,reduction,n_dims See [carve.Seurat()].
#' @param ... Forwarded to [carve.default()].
#'
#' @return The input Seurat object with the fitted CARVE stored in
#'   `object@misc$CARVE`.
#'
#' @export
RunCARVE <- function(object,
                     assay = "RNA",
                     slot = "data",
                     features = NULL,
                     reduction = NULL,
                     n_dims = NULL,
                     ...) {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("Package 'Seurat' is required for RunCARVE().", call. = FALSE)
  }
  if (!inherits(object, "Seurat")) {
    stop("RunCARVE(): 'object' must be a Seurat object.", call. = FALSE)
  }
  fit <- carve.Seurat(object, assay = assay, slot = slot,
                      features = features, reduction = reduction,
                      n_dims = n_dims, ...)
  object@misc$CARVE <- fit
  object
}


#' Add CARVE cluster labels to a Seurat object
#'
#' Retrieves the fitted CARVE stored in `object@misc$CARVE` (via
#' [RunCARVE()]) and writes the selected labels to
#' `object@meta.data[[label_col]]` as a factor. Sets the active ident
#' unless `set_ident = FALSE`.
#'
#' @param object A Seurat object previously processed by [RunCARVE()].
#' @param measure,rule,k,not_two,mode See [`CARVE$get_labels()`][CARVE].
#' @param label_col Name of the metadata column to write (default
#'   `"carve_labels"`).
#' @param set_ident Logical; if `TRUE`, sets `label_col` as the active
#'   ident in the Seurat object.
#'
#' @return The input Seurat object with `label_col` added to metadata.
#'
#' @export
AddCarveLabels <- function(object,
                           measure = "generalizability",
                           rule = "1se",
                           k = NULL,
                           not_two = FALSE,
                           mode = "default",
                           label_col = "carve_labels",
                           set_ident = TRUE) {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("Package 'Seurat' is required for AddCarveLabels().", call. = FALSE)
  }
  fit <- object@misc$CARVE
  if (is.null(fit) || !inherits(fit, "CARVE")) {
    stop("AddCarveLabels(): no fitted CARVE found in object@misc$CARVE. ",
         "Call RunCARVE(object, ...) first.", call. = FALSE)
  }
  labels <- fit$get_labels(measure = measure, rule = rule, k = k,
                           not_two = not_two, mode = mode)
  if (length(labels) != ncol(object)) {
    stop(sprintf(
      "AddCarveLabels(): label length (%d) does not match the number ",
      length(labels)),
      sprintf("of cells in the object (%d).", ncol(object)),
      call. = FALSE)
  }
  object@meta.data[[label_col]] <- factor(labels)
  if (isTRUE(set_ident)) {
    Seurat::Idents(object) <- label_col
  }
  object
}


# ---- Internal extractors ----------------------------------------------

# Extract a (cells x features) numeric matrix from a Seurat object.
#
# Seurat v4 stores assay layers under `@data`/`@counts`/`@scale.data`
# via `Seurat::GetAssayData(obj, slot = ...)`. Seurat v5 uses `layer`
# in place of `slot`. We dispatch on the installed version.
.seurat_extract_matrix <- function(obj, assay, slot, features,
                                   reduction, n_dims) {
  if (!is.null(reduction)) {
    emb <- Seurat::Embeddings(obj, reduction = reduction)
    if (!is.null(n_dims)) {
      n_dims <- as.integer(n_dims)
      emb <- emb[, seq_len(min(n_dims, ncol(emb))), drop = FALSE]
    }
    return(as.matrix(emb))
  }
  sv <- utils::packageVersion("Seurat")
  mat <- if (sv >= "5.0.0") {
    tryCatch(
      Seurat::GetAssayData(obj, assay = assay, layer = slot),
      error = function(e) Seurat::GetAssayData(obj, assay = assay, slot = slot)
    )
  } else {
    Seurat::GetAssayData(obj, assay = assay, slot = slot)
  }
  if (!is.null(features)) {
    keep <- intersect(features, rownames(mat))
    if (!length(keep)) {
      stop("None of the requested features were found in assay ",
           sQuote(assay), ".", call. = FALSE)
    }
    mat <- mat[keep, , drop = FALSE]
  }
  # Seurat stores features in rows, cells in columns; CARVE expects
  # rows = samples (cells), so transpose.
  as.matrix(Matrix::t(mat))
}

.sce_extract_matrix <- function(obj, assay, reduction, n_dims, features) {
  if (!is.null(reduction)) {
    emb <- SingleCellExperiment::reducedDim(obj, reduction)
    if (!is.null(n_dims)) {
      n_dims <- as.integer(n_dims)
      emb <- emb[, seq_len(min(n_dims, ncol(emb))), drop = FALSE]
    }
    return(as.matrix(emb))
  }
  mat <- SummarizedExperiment::assay(obj, assay)
  if (!is.null(features)) {
    keep <- intersect(features, rownames(mat))
    if (!length(keep)) {
      stop("None of the requested features were found in assay ",
           sQuote(assay), ".", call. = FALSE)
    }
    mat <- mat[keep, , drop = FALSE]
  }
  as.matrix(Matrix::t(mat))
}
