#' @include AllGenerics.R
NULL

# run_carve() and attach_results(). Mirrors tl/_carve.py. The functions
# below read and write a SingleCellExperiment through sce.R and a Seurat
# object through seurat.R, so the plot methods share them too.

cells_kind <- function(object) {
  if (methods::is(object, "SingleCellExperiment")) {
    return("sce")
  }
  if (is_seurat(object)) {
    require_package("SeuratObject", "Seurat objects")
    return("seurat")
  }
  stop(sprintf(
    "object must be a SingleCellExperiment or a Seurat object, not an object of class '%s'.",
    class(object)[[1L]]
  ), call. = FALSE)
}

cells_matrix <- function(object, assay = NULL, reduction = NULL, n_dims = NULL) {
  switch(cells_kind(object),
    sce = sce_matrix(object, assay = assay, reduction = reduction, n_dims = n_dims),
    seurat = seurat_matrix(object, assay = assay, reduction = reduction, n_dims = n_dims)
  )
}

cells_column_names <- function(object) {
  switch(cells_kind(object), sce = sce_column_names(object), seurat = seurat_column_names(object))
}

cells_get_column <- function(object, name) {
  switch(cells_kind(object), sce = sce_column(object, name), seurat = seurat_column(object, name))
}

cells_set_columns <- function(object, columns) {
  switch(cells_kind(object),
    sce = sce_set_columns(object, columns),
    seurat = seurat_set_columns(object, columns)
  )
}

cells_get_record <- function(object, key) {
  switch(cells_kind(object), sce = sce_record(object, key), seurat = seurat_record(object, key))
}

cells_set_record <- function(object, key, record) {
  switch(cells_kind(object),
    sce = sce_set_record(object, key, record),
    seurat = seurat_set_record(object, key, record)
  )
}

cells_basis <- function(object, basis = NULL, fallback = NULL) {
  switch(cells_kind(object),
    sce = sce_basis(object, basis = basis, fallback = fallback),
    seurat = seurat_basis(object, basis = basis, fallback = fallback)
  )
}

# Where the results live, as R code a user can type, for messages.
cells_columns_where <- function(object) {
  switch(cells_kind(object), sce = "colData(object)", seurat = "object@meta.data")
}

cells_column_where <- function(object, name) {
  sprintf("%s$%s", cells_columns_where(object), name)
}

cells_record_where <- function(object, key) {
  switch(cells_kind(object),
    sce = sprintf("metadata(object)$%s", key),
    seurat = sprintf("Misc(object, '%s')", key)
  )
}

# A dense matrix this large is worth a warning: 10,000 cells take 800 MB.
CONSENSUS_WARN_CELLS <- 10000L

check_cell_count <- function(values, n_cells, what) {
  if (length(values) != n_cells) {
    stop(sprintf(
      "The model produced %d %s but object has %d observations. The model was fitted on different data.",
      length(values), what, n_cells
    ), call. = FALSE)
  }
}

# The settings stored as the record's params. Entries without a value are
# left out, as Python leaves out None.
record_params <- function(fit, selected, measure, rule, not_two, consensus_k, assay, reduction,
                          n_dims, mode, n_cells) {
  settings <- fit@run_params
  param <- fit@sweep@param
  row <- selected$row
  params <- list(
    measure = measure,
    rule = rule,
    not_two = isTRUE(not_two),
    pinned = isTRUE(selected$pinned),
    sweep_param = param,
    n_resamples = settings$n_resamples,
    subsample_ratio = settings$subsample_ratio,
    n_consensus_anchors = if (is.null(fit@consensus_anchors)) NULL else length(fit@consensus_anchors),
    noise_policy = settings$noise_policy,
    mode = mode,
    random_state = settings$random_state,
    assay = assay,
    reduction = reduction,
    n_dims = n_dims,
    consensus_k = consensus_k,
    estimator_param_grids = settings$estimator_param_grids,
    selected_config_id = selected$config_id,
    selected_k = selected$n_clusters,
    n_obs = n_cells,
    sweep_values = as.numeric(fit@sweep@values)
  )
  params <- params[!vapply(params, is.null, logical(1))]
  if (param %in% names(row)) {
    params[[paste0("selected_", param)]] <- row[[param]][[1L]]
  }
  for (name in c("estimator", "method_label", "method_id")) {
    if (name %in% names(row)) {
      params[[paste0("selected_", name)]] <- as.character(row[[name]][[1L]])
    }
  }
  params$carve_version <- as.character(utils::packageVersion("CARVE"))
  params
}

#' Run CARVE on a single-cell object
#'
#' `run_carve()` fits [carve()] to a SingleCellExperiment or a Seurat object
#' and stores the results of the selected configuration in it.
#' `attach_results()` stores them from a fit you already have, for example
#' one you keep to look at other configurations.
#'
#' With the default `key = "carve"`, these cell columns are written, to
#' `colData()` of a SingleCellExperiment or to the cell metadata of a Seurat
#' object:
#'
#' - `carve`: the labels [get_labels()] returns, as a factor with its levels
#'   in numeric order.
#' - `carve_stability` and `carve_stability_ce`: per-cell stability in its
#'   Gini and cross-entropy forms, as [sample_scores()] returns them.
#' - `carve_generalizability`: per-cell held-out accuracy.
#'
#' A column is missing when `mode` skipped its criterion. A call leaves the
#' columns of an earlier call with the same key that it does not write.
#'
#' The call also stores a record: `metadata(object)$carve` for a
#' SingleCellExperiment, and the `carve` entry of the `misc` slot of a Seurat
#' object. The record is a list:
#'
#' - `params`: the run and selection settings, among them the selected
#'   `config_id`, `k` and `method_id`.
#' - `results`: the table [estimator_results()] returns, unless
#'   `store_results = FALSE`.
#' - `preprocessing_results`: the table [preprocessing_results()] returns,
#'   for a run with `randomize_preprocessing = TRUE` and unless
#'   `store_results = FALSE`.
#' - `consensus`: the consensus matrix of the selected configuration, unless
#'   `store_consensus = FALSE`. An anchored run has only the block over its
#'   anchors, which is not stored; a warning says so.
#'
#' Each call replaces the record. The plot functions read the columns and
#' the record, so `plot_cluster_violin(object)` draws the stored
#' configuration without fitting again. The fit itself is not stored,
#' because it holds the consensus matrices of every configuration. Keep it
#' when you need those, and save it with [saveRDS()].
#'
#' @param object A SingleCellExperiment or a Seurat object.
#' @param ... Arguments of [carve()], such as `n_clusters`, `resolution` or
#'   `n_resamples`.
#' @param assay,reduction,n_dims Which data to cluster; see [carve()]. They
#'   are recorded, so the scatter plots can find the same data.
#' @param key Name of the label column and of the record, and prefix of the
#'   score columns.
#' @param measure,rule,not_two,k,sweep_value,consensus_k Which configuration
#'   to store and how to cut its consensus matrix; see [get_labels()].
#' @param reference_key Name of a cell metadata column with reference
#'   labels, which the stored labels are renamed to match; see
#'   `reference_labels` in [carve()].
#' @param store_consensus Store the consensus matrix in the record.
#' @param store_results Store the results tables in the record.
#' @param mode The `mode` of [carve()]. For `attach_results()`, the mode the
#'   fit was run with: under `"generalizability"` the labels and the stored
#'   matrix come from the consensus of the held-out predictions.
#' @param random_state Seed of the run.
#' @param fit A [CARVE-class] fit on the cells of `object`, in the same
#'   order.
#' @return The object, with the results added.
#' @seealso [carve()], [get_labels()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' sce <- run_carve(sce, n_resamples = 10, estimator_param_grids = grid)
#' table(sce$carve)
#' names(S4Vectors::metadata(sce)$carve)
#'
#' fit <- carve(sce, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' sce <- attach_results(sce, fit, key = "carve_k3", k = 3)
#' table(sce$carve_k3)
#' @rdname run_carve
#' @export
run_carve <- function(object, ..., assay = NULL, reduction = NULL, n_dims = NULL, key = "carve",
                      measure = "stability", rule = "1se", not_two = FALSE, k = NULL,
                      sweep_value = NULL, consensus_k = NULL, reference_key = NULL,
                      store_consensus = TRUE, store_results = TRUE, mode = "default",
                      random_state = 0) {
  reference_labels <- NULL
  if (!is.null(reference_key)) {
    reference_labels <- cells_get_column(object, reference_key)
    if (is.null(reference_labels)) {
      stop(sprintf(
        "reference_key=%s not found in %s. Available columns: %s",
        format_repr(reference_key),
        cells_columns_where(object),
        python_list(sort(cells_column_names(object)))
      ), call. = FALSE)
    }
  }
  X <- cells_matrix(object, assay = assay, reduction = reduction, n_dims = n_dims)
  fit <- carve(X, ..., reference_labels = reference_labels, mode = mode, random_state = random_state)
  attach_results(
    object, fit,
    key = key, measure = measure, rule = rule, not_two = not_two, k = k,
    sweep_value = sweep_value, consensus_k = consensus_k, store_consensus = store_consensus,
    store_results = store_results, assay = assay, reduction = reduction, n_dims = n_dims,
    mode = mode
  )
}

#' @rdname run_carve
#' @export
attach_results <- function(object, fit, key = "carve", measure = "stability", rule = "1se",
                           not_two = FALSE, k = NULL, sweep_value = NULL, consensus_k = NULL,
                           store_consensus = TRUE, store_results = TRUE, assay = NULL,
                           reduction = NULL, n_dims = NULL, mode = "default") {
  cells_kind(object)
  if (!methods::is(fit, "CARVE")) {
    stop("fit must be a CARVE object from carve().", call. = FALSE)
  }
  selected <- select_row(fit, measure, rule, not_two = not_two, k = k, sweep_value = sweep_value)
  # A generalizability-only run builds no stability consensus, so the cut
  # and the stored matrix both come from the held-out consensus, as in
  # get_labels().
  label_mode <- if (identical(mode, "generalizability")) "generalizability" else "default"
  labels <- get_labels(
    fit, measure = measure, rule = rule, k = k, sweep_value = sweep_value,
    consensus_k = consensus_k, not_two = not_two, mode = label_mode
  )
  n_cells <- as.integer(ncol(object))
  check_cell_count(labels, n_cells, "consensus labels")

  columns <- list()
  columns[[key]] <- factor(as.character(labels), levels = as.character(sort(unique(labels))))
  id <- as.character(selected$config_id)
  # A criterion the run skipped has no scores, and its column is not
  # written.
  scores <- list(
    stability = fit@stability_gini_scores[[id]],
    stability_ce = fit@stability_ce_scores[[id]],
    generalizability = fit@generalizability_scores[[id]]
  )
  for (suffix in names(scores)) {
    values <- scores[[suffix]]
    if (is.null(values)) {
      next
    }
    check_cell_count(values, n_cells, paste(suffix, "scores"))
    columns[[paste0(key, "_", suffix)]] <- as.numeric(values)
  }

  where <- cells_record_where(object, key)
  consensus <- if (label_mode == "generalizability") {
    fit@consensus_generalizability_matrices[[id]]
  } else {
    fit@consensus_matrices[[id]]
  }
  stored <- NULL
  if (isTRUE(store_consensus) && !is.null(consensus)) {
    anchors <- fit@consensus_anchors
    m <- length(anchors)
    if (!is.null(anchors) && identical(dim(consensus), c(m, m))) {
      warning(sprintf(
        "Anchored consensus is active: the consensus matrix for '%s' is a %dx%d anchor block, not a cell-by-cell matrix. %s holds only the latter, so nothing was written to %s$consensus. The block itself is still available with consensus_matrix(fit, config_id = %d%s), and the anchor count is recorded in %s$params$n_consensus_anchors.",
        key, m, m, where, where, selected$config_id,
        if (label_mode == "generalizability") ", type = 'generalizability'" else "",
        where
      ), call. = FALSE)
    } else if (!identical(dim(consensus), c(n_cells, n_cells))) {
      stop(sprintf(
        "Consensus matrix has shape (%d, %d), but object has %d observations. The model was fitted on different data.",
        nrow(consensus), ncol(consensus), n_cells
      ), call. = FALSE)
    } else {
      if (n_cells > CONSENSUS_WARN_CELLS) {
        warning(sprintf(
          "Storing a dense %dx%d consensus matrix in %s$consensus (~%.1f GB). Pass store_consensus=FALSE if you do not need it.",
          n_cells, n_cells, where, 8 * n_cells^2 / 1e9
        ), call. = FALSE)
      }
      stored <- consensus
    }
  }

  record <- list(params = record_params(
    fit, selected, measure, rule, not_two, consensus_k, assay, reduction, n_dims, mode, n_cells
  ))
  if (isTRUE(store_results)) {
    record$results <- fit@estimator_results
    if (!is.null(fit@preprocessing_results)) {
      record$preprocessing_results <- fit@preprocessing_results
    }
  }
  if (!is.null(stored)) {
    record$consensus <- stored
  }
  object <- cells_set_columns(object, columns)
  cells_set_record(object, key, record)
}
