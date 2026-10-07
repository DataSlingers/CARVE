#' @include AllGenerics.R
NULL

# The plot methods. A CARVE method mirrors the plot method of api.CARVE of
# the same name: it selects a configuration and passes its results to the
# drawing in plotting.R. A SingleCellExperiment method mirrors the carve.pl
# function: it reads what run_carve() or attach_results() stored and selects
# nothing again, because the stored scores belong to the configuration
# selected then. Seurat objects reach the ANY methods, which pass them to
# the same functions as SingleCellExperiment objects.

# The plot methods of a fit take three modes; "stability" cuts the same
# matrix as "default".
labels_mode <- function(mode) {
  if (identical(mode, "default") || identical(mode, "stability")) {
    return("default")
  }
  if (identical(mode, "generalizability")) {
    return("generalizability")
  }
  stop("mode must be one of: 'default', 'stability', 'generalizability'.", call. = FALSE)
}

check_seurat_object <- function(object) {
  if (!is_seurat(object)) {
    stop(sprintf(
      "object must be a CARVE fit, a SingleCellExperiment or a Seurat object, not an object of class '%s'.",
      class(object)[[1L]]
    ), call. = FALSE)
  }
  require_package("SeuratObject", "Seurat objects")
}

# The record run_carve() stored under key.
cells_entry <- function(object, key) {
  record <- cells_get_record(object, key)
  where <- cells_record_where(object, key)
  if (is.null(record)) {
    stop(sprintf(
      "%s not found. Run `run_carve(object, key = %s)` first, or pass the key you used.",
      where, format_repr(key)
    ), call. = FALSE)
  }
  if (!is.list(record) || !"params" %in% names(record)) {
    stop(sprintf(
      "%s does not look like a CARVE result; expected a list with a 'params' entry.",
      where
    ), call. = FALSE)
  }
  record
}

# The stored labels as integers. A factor whose levels are all whole
# numbers, such as the column attach_results() writes, gives its level
# values, so the plots name the clusters as the column and the fit's plots
# do, also when the labels are not 1 to k. Any other factor gives its codes,
# 1, 2, ..., and a numeric column is read as it is. Python reads the codes
# of every categorical column.
cells_labels <- function(object, key) {
  column <- cells_get_column(object, key)
  if (is.null(column)) {
    stop(sprintf(
      "%s not found. Run `run_carve(object, key = %s)` first.",
      cells_column_where(object, key), format_repr(key)
    ), call. = FALSE)
  }
  if (is.factor(column) && all(grepl("^-?[0-9]+$", levels(column)))) {
    return(as.integer(levels(column))[column])
  }
  as.integer(column)
}

cells_results <- function(object, key) {
  results <- cells_entry(object, key)[["results"]]
  if (is.null(results)) {
    stop(sprintf(
      "%s$results not found. It is omitted when run_carve runs with store_results=FALSE.",
      cells_record_where(object, key)
    ), call. = FALSE)
  }
  results
}

cells_preprocessing_results <- function(object, key) {
  results <- cells_entry(object, key)[["preprocessing_results"]]
  if (is.null(results)) {
    stop(sprintf(
      "%s$preprocessing_results not found. It is written only for a randomized fit, so either run_carve ran without randomize_preprocessing=TRUE or it ran with store_results=FALSE.",
      cells_record_where(object, key)
    ), call. = FALSE)
  }
  results
}

# A single-cell plot argument left NULL takes the value run_carve()
# recorded, or Python's default when the record lacks it.
recorded <- function(value, params, name, default) {
  if (!is.null(value)) {
    return(value)
  }
  if (is.null(params[[name]])) default else params[[name]]
}

#' Plot a criterion over the swept parameter
#'
#' Draws the value of `measure` at each value of the swept parameter, one
#' line per estimator configuration. A dashed line marks the value `rule`
#' selects.
#'
#' For a fit, the plot reads [estimator_results()]. For a SingleCellExperiment
#' or a Seurat object, it reads the results table [run_carve()] or
#' [attach_results()] stored under `key`, and `measure`, `rule` and `not_two`
#' default to the values recorded there, so the dashed line marks the
#' configuration whose labels are stored.
#'
#' The three ARI criteria get error bars of one standard error; the other
#' criteria have no standard errors. On a run over `resolution` or
#' `min_cluster_size`, the x axis is that parameter.
#'
#' @param object A [CARVE-class] fit, or a SingleCellExperiment or Seurat
#'   object with results stored by [run_carve()] or [attach_results()].
#' @param measure The criterion to plot and select on, as in [get_k()]. For a
#'   single-cell object, `NULL` uses the recorded one.
#' @param rule The rule that selects the value the dashed line marks:
#'   `"max"`, `"1se"` or `"quantile"`; see [get_k()]. For a single-cell
#'   object, `NULL` uses the recorded one.
#' @param not_two Leave out configurations with two clusters when selecting.
#'   For a single-cell object, `NULL` uses the recorded value.
#' @param title,xlabel,ylabel Title and axis labels. By default there is no
#'   title, and the axes are named after the swept parameter and the
#'   criterion.
#' @param legend Show the legends.
#' @param legend_loc Where the legends go: `"right"`, `"left"`, `"top"`,
#'   `"bottom"`, or two numbers between 0 and 1 that place them inside the
#'   panel.
#' @param palette Line colors: `"Accent"`, another name from
#'   [grDevices::palette.pals()], a viridis option such as `"viridis"` or
#'   `"magma"`, or a vector of colors.
#' @param key For a single-cell object, the `key` the results were stored
#'   under.
#' @param ... For a Seurat object, the arguments of the SingleCellExperiment
#'   method. For a fit or a SingleCellExperiment, not used: an argument name
#'   the method does not know is an error.
#' @return A ggplot object.
#' @seealso [get_k()], [plot_n_clusters_over_sweep()], [run_carve()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_metric_over_n_clusters(fit)
#' plot_metric_over_n_clusters(fit, measure = "generalizability", rule = "max")
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_metric_over_n_clusters(sce)
#' @rdname plot_metric_over_n_clusters
#' @export
setMethod("plot_metric_over_n_clusters", "CARVE", function(object, measure = "stability",
                                                           rule = "1se", not_two = FALSE,
                                                           title = NULL, xlabel = NULL,
                                                           ylabel = NULL, legend = TRUE,
                                                           legend_loc = "right",
                                                           palette = "Accent", ...) {
  check_dots(...)
  metric_over_sweep_plot(
    object@estimator_results,
    measure = measure, rule = rule, not_two = not_two, title = title, xlabel = xlabel,
    ylabel = ylabel, legend = legend, legend_loc = legend_loc, palette = palette
  )
})

cells_plot_metric_over_n_clusters <- function(object, key = "carve", measure = NULL, rule = NULL,
                                              not_two = NULL, title = NULL, xlabel = NULL,
                                              ylabel = NULL, legend = TRUE, legend_loc = "right",
                                              palette = "Accent", ...) {
  check_dots(...)
  params <- cells_entry(object, key)$params
  metric_over_sweep_plot(
    cells_results(object, key),
    measure = recorded(measure, params, "measure", "stability"),
    rule = recorded(rule, params, "rule", "1se"),
    not_two = recorded(not_two, params, "not_two", FALSE),
    title = title, xlabel = xlabel, ylabel = ylabel, legend = legend, legend_loc = legend_loc,
    palette = palette
  )
}

#' @rdname plot_metric_over_n_clusters
#' @export
setMethod("plot_metric_over_n_clusters", "SingleCellExperiment", cells_plot_metric_over_n_clusters)

#' @rdname plot_metric_over_n_clusters
#' @export
setMethod("plot_metric_over_n_clusters", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_metric_over_n_clusters(object, ...)
})

#' Plot a criterion for each preprocessing pipeline
#'
#' For a run with `randomize_preprocessing = TRUE`: draws the rows of
#' [preprocessing_results()] for one configuration, one line per pipeline,
#' with error bars of one standard error over the resamples each pipeline
#' received. The dashed line marks the value CARVE selects from
#' [estimator_results()], which pools the pipelines. Each pipeline's line
#' crosses the dashed line at that pipeline's score for the selected
#' configuration. For a
#' `method_id` that CARVE did not select, the dashed line marks the value
#' `rule` picks among that configuration's rows.
#'
#' For a SingleCellExperiment or a Seurat object, the plot reads the two
#' tables [run_carve()] or [attach_results()] stored under `key`; they are
#' stored for a randomized run unless `store_results = FALSE`. `method_id`,
#' `measure`, `rule` and `not_two` default to the recorded values. A recorded
#' measure that the per-pipeline table does not have is replaced by
#' `"stability"`, with a warning.
#'
#' @inheritParams plot_metric_over_n_clusters
#' @param method_id The configuration to draw, a value of the `method_id`
#'   column. `NULL` takes the configuration CARVE selects under `measure`,
#'   `rule` and `not_two`, or for a single-cell object the recorded one.
#' @param measure `"stability"` or `"generalizability"`, or an alias of
#'   either; the per-pipeline table has only these two criteria. For a
#'   single-cell object, `NULL` uses the recorded measure.
#' @return A ggplot object.
#' @seealso [preprocessing_results()], [plot_metric_over_n_clusters()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' fit <- carve(
#'   X, n_resamples = 20, random_state = 0, randomize_preprocessing = TRUE,
#'   estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)),
#'   normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
#'   dim_reduction_options = list(preprocessing_option(Identity))
#' )
#' plot_metric_by_pipeline(fit)
#' plot_metric_by_pipeline(fit, measure = "generalizability")
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_metric_by_pipeline(sce)
#' @rdname plot_metric_by_pipeline
#' @export
setMethod("plot_metric_by_pipeline", "CARVE", function(object, method_id = NULL,
                                                       measure = "stability", rule = "1se",
                                                       not_two = FALSE, title = NULL,
                                                       xlabel = NULL, ylabel = NULL, legend = TRUE,
                                                       legend_loc = "right", palette = "Accent",
                                                       ...) {
  check_dots(...)
  if (is.null(method_id)) {
    selected <- select_row(object, measure, rule, not_two = not_two)
    method_id <- as.character(selected$row$method_id[[1L]])
  }
  metric_by_pipeline_plot(
    object@preprocessing_results, object@estimator_results,
    method_id = method_id, measure = measure, rule = rule, not_two = not_two, title = title,
    xlabel = xlabel, ylabel = ylabel, legend = legend, legend_loc = legend_loc, palette = palette
  )
})

cells_plot_metric_by_pipeline <- function(object, key = "carve", method_id = NULL, measure = NULL,
                                          rule = NULL, not_two = NULL, title = NULL, xlabel = NULL,
                                          ylabel = NULL, legend = TRUE, legend_loc = "right",
                                          palette = "Accent", ...) {
  check_dots(...)
  params <- cells_entry(object, key)$params
  preprocessing <- cells_preprocessing_results(object, key)
  if (is.null(measure)) {
    measure <- recorded(NULL, params, "measure", "stability")
    if (!measure %in% names(MEASURE_MAP) || !MEASURE_MAP[[measure]] %in% names(preprocessing)) {
      warning(sprintf(
        "run_carve recorded measure %s, which the per-pipeline table does not carry; plotting 'stability'. Pass measure='generalizability' for the other ARI criterion.",
        format_repr(measure)
      ), call. = FALSE)
      measure <- "stability"
    }
  }
  metric_by_pipeline_plot(
    preprocessing, cells_results(object, key),
    method_id = recorded(method_id, params, "selected_method_id", NULL),
    measure = measure,
    rule = recorded(rule, params, "rule", "1se"),
    not_two = recorded(not_two, params, "not_two", FALSE),
    title = title, xlabel = xlabel, ylabel = ylabel, legend = legend, legend_loc = legend_loc,
    palette = palette
  )
}

#' @rdname plot_metric_by_pipeline
#' @export
setMethod("plot_metric_by_pipeline", "SingleCellExperiment", cells_plot_metric_by_pipeline)

#' @rdname plot_metric_by_pipeline
#' @export
setMethod("plot_metric_by_pipeline", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_metric_by_pipeline(object, ...)
})

#' Plot the number of clusters over the swept parameter
#'
#' For a run over `resolution`, `min_cluster_size` or another parameter that
#' does not fix the number of clusters: draws the mean number of clusters
#' observed at each value of the swept parameter, one line per estimator
#' configuration, with error bars of one standard error. The dashed line
#' marks the value selected under `measure`, `rule` and `not_two`, and its
#' legend entry gives the number of clusters there, the number [get_k()]
#' returns for the same arguments. On a run over `n_clusters` the count is
#' the swept value itself, and the function stops with an error.
#'
#' The count is the mean, over resamples, of the number of clusters in the
#' clustering of each resample's subsample, counted after the `noise_policy`
#' of [carve()]: `"drop"` leaves noise points out, `"as_cluster"` counts the
#' noise as one cluster, and `"singleton"` counts each noise point as its own
#' cluster. It is not the count of one clustering of all the samples.
#' [get_labels()] cuts the consensus matrix at the rounded count unless
#' `consensus_k` is given. The error bars show the standard error of the
#' mean, not the spread of the counts.
#'
#' For a SingleCellExperiment or a Seurat object, the plot reads the results
#' table stored under `key`, and `measure`, `rule` and `not_two` default to
#' the recorded values.
#'
#' @inheritParams plot_metric_over_n_clusters
#' @param measure The criterion that selects the marked value, as in
#'   [get_k()]. For a single-cell object, `NULL` uses the recorded one.
#' @param title,xlabel,ylabel Title and axis labels. By default there is no
#'   title, the x axis is named after the swept parameter, and the y axis is
#'   "Mean Observed Number of Clusters".
#' @return A ggplot object.
#' @seealso [get_k()], [plot_metric_over_n_clusters()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(LeidenClustering, resolution = c(0.25, 0.5, 1, 2)))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_n_clusters_over_sweep(fit)
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_n_clusters_over_sweep(sce)
#' @rdname plot_n_clusters_over_sweep
#' @export
setMethod("plot_n_clusters_over_sweep", "CARVE", function(object, measure = "stability",
                                                          rule = "1se", not_two = FALSE,
                                                          title = NULL, xlabel = NULL,
                                                          ylabel = NULL, legend = TRUE,
                                                          legend_loc = "right",
                                                          palette = "Accent", ...) {
  check_dots(...)
  n_clusters_over_sweep_plot(
    object@estimator_results,
    measure = measure, rule = rule, not_two = not_two, title = title, xlabel = xlabel,
    ylabel = ylabel, legend = legend, legend_loc = legend_loc, palette = palette
  )
})

cells_plot_n_clusters_over_sweep <- function(object, key = "carve", measure = NULL, rule = NULL,
                                             not_two = NULL, title = NULL, xlabel = NULL,
                                             ylabel = NULL, legend = TRUE, legend_loc = "right",
                                             palette = "Accent", ...) {
  check_dots(...)
  params <- cells_entry(object, key)$params
  n_clusters_over_sweep_plot(
    cells_results(object, key),
    measure = recorded(measure, params, "measure", "stability"),
    rule = recorded(rule, params, "rule", "1se"),
    not_two = recorded(not_two, params, "not_two", FALSE),
    title = title, xlabel = xlabel, ylabel = ylabel, legend = legend, legend_loc = legend_loc,
    palette = palette
  )
}

#' @rdname plot_n_clusters_over_sweep
#' @export
setMethod("plot_n_clusters_over_sweep", "SingleCellExperiment", cells_plot_n_clusters_over_sweep)

#' @rdname plot_n_clusters_over_sweep
#' @export
setMethod("plot_n_clusters_over_sweep", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_n_clusters_over_sweep(object, ...)
})

#' Plot a consensus matrix
#'
#' Draws the consensus matrix of one configuration as a heatmap, with the
#' samples ordered by cluster, a band of cluster colors along the top and
#' white lines between the clusters. Pairs of samples never drawn together
#' are shown at 0.5.
#'
#' For a fit, the plot selects a configuration with `measure`, `rule`,
#' `not_two`, `k` and `sweep_value`, as [get_labels()] does, and colors the
#' band with the labels of the cut at the selected number of clusters. On an
#' anchored run it draws the block over the anchors, with the anchors'
#' labels, because an anchored run computes the consensus only among the
#' anchors and never builds the matrix over every sample.
#'
#' For a SingleCellExperiment or a Seurat object, the plot draws the matrix
#' stored under `key` with the stored labels. [run_carve()] and
#' [attach_results()] store the matrix unless the run was anchored or
#' `store_consensus = FALSE`.
#'
#' @inheritParams plot_metric_over_n_clusters
#' @param measure,rule,not_two For a fit, how the configuration is selected,
#'   as in [get_labels()].
#' @param mode For a fit, `"default"` or `"stability"` draws the stability
#'   consensus, and `"generalizability"` the consensus of the classifier's
#'   predictions for the held-out samples.
#' @param k,sweep_value For a fit, pin the configuration, as in
#'   [get_labels()].
#' @param cmap Colors of the heatmap: a palette name as for `palette`, such
#'   as `"viridis"` or `"Greens"`, or a vector of at least two colors.
#' @param palette Colors of the cluster band, as in
#'   [plot_metric_over_n_clusters()].
#' @param colorbar Show the color bar.
#' @param colorbar_label Title of the color bar.
#' @param title Plot title.
#' @return A ggplot object.
#' @seealso [consensus_matrix()], [get_labels()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_consensus_matrix(fit)
#' plot_consensus_matrix(fit, k = 3, cmap = "Greens")
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_consensus_matrix(sce)
#' @rdname plot_consensus_matrix
#' @export
setMethod("plot_consensus_matrix", "CARVE", function(object, measure = "stability", rule = "1se",
                                                     not_two = FALSE, mode = "default", k = NULL,
                                                     sweep_value = NULL, cmap = "viridis",
                                                     palette = "Accent", colorbar = TRUE,
                                                     colorbar_label = "Consensus", title = NULL,
                                                     ...) {
  check_dots(...)
  label_mode <- labels_mode(mode)
  selected <- select_row(object, measure, rule, not_two = not_two, k = k, sweep_value = sweep_value)
  id <- as.character(selected$config_id)
  consensus <- if (label_mode == "default") {
    object@consensus_matrices[[id]]
  } else {
    object@consensus_generalizability_matrices[[id]]
  }
  if (is.null(consensus)) {
    stop(sprintf(
      "Selected consensus matrix is not available for mode=%s.",
      format_repr(mode)
    ), call. = FALSE)
  }
  labels <- get_labels(
    object, measure = measure, rule = rule, k = k, sweep_value = sweep_value,
    consensus_k = selected$n_clusters, not_two = not_two, mode = label_mode
  )
  # On an anchored run the stored matrix is the block over the anchors, so
  # the band shows the anchors' labels.
  if (!is.null(object@consensus_anchors)) {
    labels <- labels[object@consensus_anchors]
  }
  consensus_plot(
    consensus, labels,
    cmap = cmap, palette = palette, colorbar = colorbar, colorbar_label = colorbar_label,
    title = title
  )
})

cells_plot_consensus_matrix <- function(object, key = "carve", cmap = "viridis", palette = "Accent",
                                        colorbar = TRUE, colorbar_label = "Consensus",
                                        title = NULL, ...) {
  check_dots(...)
  consensus <- cells_entry(object, key)[["consensus"]]
  if (is.null(consensus)) {
    where <- cells_record_where(object, key)
    stop(sprintf(
      "%s$consensus not found. It is omitted when run_carve runs with store_consensus=FALSE, or when anchored consensus was active: an anchor block is m-by-m rather than n_obs-by-n_obs and %s holds only the latter.",
      where, where
    ), call. = FALSE)
  }
  consensus_plot(
    consensus, cells_labels(object, key),
    cmap = cmap, palette = palette, colorbar = colorbar, colorbar_label = colorbar_label,
    title = title
  )
}

#' @rdname plot_consensus_matrix
#' @export
setMethod("plot_consensus_matrix", "SingleCellExperiment", cells_plot_consensus_matrix)

#' @rdname plot_consensus_matrix
#' @export
setMethod("plot_consensus_matrix", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_consensus_matrix(object, ...)
})

# Per-sample scores by source: the suffix of the column run_carve() writes,
# the name the box and violin plots and the single-cell scatter plots give
# the score, the name the scatter plots of a fit give it, and the mode that
# writes the column.
SOURCE_SUFFIXES <- c(gini = "stability", ce = "stability_ce", accuracy = "generalizability")
SOURCE_LABELS <- c(
  gini = "Cluster Stability (Gini)",
  ce = "Cluster Stability (CE)",
  accuracy = "Cluster Generalizability"
)
SOURCE_NAMES <- c(gini = "Gini Stability", ce = "CE Stability", accuracy = "Generalizability")
SOURCE_MODES <- c(gini = "stability", ce = "stability", accuracy = "generalizability")

# An annotation argument: TRUE builds the text, a string is used as it is,
# and anything else turns the annotation off.
resolve_annotation <- function(annotation, build) {
  if (is.character(annotation)) {
    return(annotation)
  }
  if (isTRUE(annotation)) build() else NULL
}

# What a per-sample plot of a fit draws: the scores of the selected
# configuration, the labels of its cut at the selected number of clusters,
# and the annotation.
fit_sample_plot_inputs <- function(fit, source, measure, rule, not_two, mode, k, sweep_value,
                                   annotation, tight_layout = FALSE) {
  selected <- select_row(fit, measure, rule, not_two = not_two, k = k, sweep_value = sweep_value)
  scores <- sample_scores(fit, selected$config_id, source = source)
  labels <- get_labels(
    fit, measure = measure, rule = rule, k = k, sweep_value = sweep_value,
    consensus_k = selected$n_clusters, not_two = not_two, mode = labels_mode(mode)
  )
  list(
    scores = scores,
    labels = labels,
    annotation = resolve_annotation(annotation, function() {
      annotation_text(
        measure, rule, fit@estimator_results, selected$row, selected$n_clusters,
        pinned = selected$pinned, tight_layout = tight_layout
      )
    })
  )
}

cells_scores <- function(object, key, source) {
  if (!is.character(source) || length(source) != 1L || !source %in% names(SOURCE_SUFFIXES)) {
    stop(sprintf(
      "source must be one of %s; got %s.",
      python_list(sort(names(SOURCE_SUFFIXES))), format_repr(source)
    ), call. = FALSE)
  }
  column <- paste0(key, "_", SOURCE_SUFFIXES[[source]])
  values <- cells_get_column(object, column)
  if (is.null(values)) {
    stop(sprintf(
      "%s not found, so source=%s is unavailable. It is written when run_carve runs with mode='default' or mode=%s.",
      cells_column_where(object, column), format_repr(source), format_repr(SOURCE_MODES[[source]])
    ), call. = FALSE)
  }
  as.numeric(values)
}

# The annotation of a single-cell plot, built from the record as a fit's
# plots build theirs. The row is found by config_id. Without a stored
# results table there is no annotation, and a recorded selected_k of 0
# means no count, as in Python.
cells_annotation <- function(object, key, annotation) {
  resolve_annotation(annotation, function() {
    record <- cells_entry(object, key)
    params <- record$params
    results <- record[["results"]]
    if (is.null(results)) {
      return(NULL)
    }
    id <- as.integer(recorded(NULL, params, "selected_config_id", 0))
    rows <- results[as.integer(results$config_id) == id, , drop = FALSE]
    if (nrow(rows) == 0L) {
      return(NULL)
    }
    selected_k <- as.integer(recorded(NULL, params, "selected_k", 0))
    annotation_text(
      recorded(NULL, params, "measure", "stability"),
      recorded(NULL, params, "rule", "1se"),
      results, rows[1L, , drop = FALSE],
      if (selected_k == 0L) NULL else selected_k,
      pinned = isTRUE(params[["pinned"]])
    )
  })
}

# The matrix a single-cell scatter plot draws from: the reduced dimensions
# named basis, or the first of the usual embeddings, or the reduction CARVE
# clustered. Failing those, the data CARVE clustered, read again from the
# object only then; the drawing reduces it to two principal components.
cells_scatter_data <- function(object, key, basis) {
  params <- cells_entry(object, key)$params
  embedding <- cells_basis(object, basis = basis, fallback = params[["reduction"]])
  if (!is.null(embedding)) {
    return(embedding)
  }
  cells_matrix(
    object,
    assay = params[["assay"]], reduction = params[["reduction"]], n_dims = params[["n_dims"]]
  )
}

#' Box plot of per-sample scores by cluster
#'
#' Draws one box per cluster of the per-sample scores that
#' [sample_scores()] returns: stability in its Gini or cross-entropy form,
#' or held-out accuracy. Samples without a finite score are left out.
#'
#' For a fit, the plot selects a configuration with `measure`, `rule`,
#' `not_two`, `k` and `sweep_value`, takes its scores, and groups them by
#' the labels of its cut at the selected number of clusters, the labels
#' [get_labels()] returns. For a SingleCellExperiment or a Seurat object, it
#' draws the score and label columns [run_carve()] or [attach_results()]
#' wrote under `key`. They belong to the configuration selected then, and
#' nothing is selected again.
#'
#' @inheritParams plot_consensus_matrix
#' @param source The score: `"gini"` or `"ce"` for stability in its Gini or
#'   cross-entropy form, or `"accuracy"` for held-out accuracy.
#' @param measure,rule,not_two,mode,k,sweep_value For a fit, which
#'   configuration to plot and which consensus to cut, as in
#'   [plot_consensus_matrix()]. The single-cell methods plot the stored
#'   configuration and do not take these arguments.
#' @param order Cluster labels in the order to draw them. Clusters not named
#'   are not drawn.
#' @param palette Colors of the clusters, as in
#'   [plot_metric_over_n_clusters()].
#' @param showfliers Draw the points beyond the whiskers.
#' @param width Width of the boxes; the clusters are 1 apart.
#' @param title,xlabel,ylabel Title and axis labels. The default y label
#'   names the score.
#' @param annotation `TRUE` names the selected configuration and how it was
#'   selected, a string is shown as it is, and `FALSE` shows nothing. The
#'   text goes in the caption, below the panel.
#' @param rotation Angle of the cluster labels, in degrees.
#' @param ylim Limits of the y axis, used when `fit_ylim = FALSE`. `NULL`
#'   leaves them to ggplot2.
#' @param fit_ylim Fit the y axis to the scores, with a margin of 5 percent
#'   of their range and at least 0.02.
#' @return A ggplot object.
#' @seealso [plot_cluster_violin()], [sample_scores()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_cluster_boxplot(fit)
#' plot_cluster_boxplot(fit, source = "accuracy", k = 3)
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_cluster_boxplot(sce)
#' @rdname plot_cluster_boxplot
#' @export
setMethod("plot_cluster_boxplot", "CARVE", function(object, source = "gini", measure = "stability",
                                                    rule = "1se", not_two = FALSE,
                                                    mode = "default", k = NULL,
                                                    sweep_value = NULL, order = NULL,
                                                    palette = "Accent", showfliers = FALSE,
                                                    width = 0.75, title = NULL,
                                                    xlabel = "Cluster", ylabel = NULL,
                                                    annotation = TRUE, rotation = NULL,
                                                    ylim = c(-0.02, 1.02), fit_ylim = TRUE, ...) {
  check_dots(...)
  inputs <- fit_sample_plot_inputs(
    object, source, measure, rule, not_two, mode, k, sweep_value, annotation
  )
  cluster_boxplot_plot(
    inputs$scores, inputs$labels,
    order = order, palette = palette, showfliers = showfliers, width = width, title = title,
    xlabel = xlabel, ylabel = if (is.null(ylabel)) SOURCE_LABELS[[source]] else ylabel,
    annotation = inputs$annotation, rotation = rotation, ylim = ylim, fit_ylim = fit_ylim
  )
})

cells_plot_cluster_boxplot <- function(object, key = "carve", source = "gini", order = NULL,
                                       palette = "Accent", showfliers = FALSE, width = 0.75,
                                       title = NULL, xlabel = "Cluster", ylabel = NULL,
                                       annotation = TRUE, rotation = NULL,
                                       ylim = c(-0.02, 1.02), fit_ylim = TRUE, ...) {
  check_dots(...)
  scores <- cells_scores(object, key, source)
  cluster_boxplot_plot(
    scores, cells_labels(object, key),
    order = order, palette = palette, showfliers = showfliers, width = width, title = title,
    xlabel = xlabel, ylabel = if (is.null(ylabel)) SOURCE_LABELS[[source]] else ylabel,
    annotation = cells_annotation(object, key, annotation), rotation = rotation, ylim = ylim,
    fit_ylim = fit_ylim
  )
}

#' @rdname plot_cluster_boxplot
#' @export
setMethod("plot_cluster_boxplot", "SingleCellExperiment", cells_plot_cluster_boxplot)

#' @rdname plot_cluster_boxplot
#' @export
setMethod("plot_cluster_boxplot", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_cluster_boxplot(object, ...)
})

#' Violin plot of per-sample scores by cluster
#'
#' Draws one violin per cluster of the per-sample scores, with the
#' quartiles marked inside and the samples drawn as points on top. A cluster
#' with one finite score gets its point and marks but no violin. The
#' configuration and the scores are chosen as in [plot_cluster_boxplot()].
#'
#' The points are jittered with a fixed seed, so a plot comes out the same
#' each time and leaves the session's random numbers alone.
#'
#' @inheritParams plot_cluster_boxplot
#' @param density_norm How the violins are scaled: `"width"` gives them the
#'   same width, `"area"` the same area, and `"count"` areas in proportion
#'   to the number of samples.
#' @param stripplot Draw the samples as points.
#' @param jitter Spread the points sideways: `TRUE` for the default spread,
#'   `FALSE` for none, or the spread on either side, where the clusters are
#'   1 apart.
#' @param size Area of the points, in square points.
#' @param alpha Opacity of the points.
#' @param inner Marks inside each violin: `"box"` for the median and the
#'   range between the quartiles, `"quartile"` for a line at each quartile,
#'   or `"none"`.
#' @return A ggplot object.
#' @seealso [plot_cluster_boxplot()], [sample_scores()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_cluster_violin(fit)
#' plot_cluster_violin(fit, source = "ce", inner = "quartile", stripplot = FALSE)
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_cluster_violin(sce)
#' @rdname plot_cluster_violin
#' @export
setMethod("plot_cluster_violin", "CARVE", function(object, source = "gini", measure = "stability",
                                                   rule = "1se", not_two = FALSE,
                                                   mode = "default", k = NULL, sweep_value = NULL,
                                                   order = NULL, palette = "Accent",
                                                   density_norm = "width", stripplot = TRUE,
                                                   jitter = TRUE, size = 8, alpha = 0.22,
                                                   inner = "box", title = NULL,
                                                   xlabel = "Cluster", ylabel = NULL,
                                                   annotation = TRUE, rotation = NULL,
                                                   ylim = c(-0.02, 1.02), fit_ylim = TRUE, ...) {
  check_dots(...)
  inputs <- fit_sample_plot_inputs(
    object, source, measure, rule, not_two, mode, k, sweep_value, annotation
  )
  cluster_violin_plot(
    inputs$scores, inputs$labels,
    order = order, palette = palette, density_norm = density_norm, stripplot = stripplot,
    jitter = jitter, size = size, alpha = alpha, inner = inner, title = title, xlabel = xlabel,
    ylabel = if (is.null(ylabel)) SOURCE_LABELS[[source]] else ylabel,
    annotation = inputs$annotation, rotation = rotation, ylim = ylim, fit_ylim = fit_ylim
  )
})

cells_plot_cluster_violin <- function(object, key = "carve", source = "gini", order = NULL,
                                      palette = "Accent", density_norm = "width",
                                      stripplot = TRUE, jitter = TRUE, size = 8, alpha = 0.22,
                                      inner = "box", title = NULL, xlabel = "Cluster",
                                      ylabel = NULL, annotation = TRUE, rotation = NULL,
                                      ylim = c(-0.02, 1.02), fit_ylim = TRUE, ...) {
  check_dots(...)
  scores <- cells_scores(object, key, source)
  cluster_violin_plot(
    scores, cells_labels(object, key),
    order = order, palette = palette, density_norm = density_norm, stripplot = stripplot,
    jitter = jitter, size = size, alpha = alpha, inner = inner, title = title, xlabel = xlabel,
    ylabel = if (is.null(ylabel)) SOURCE_LABELS[[source]] else ylabel,
    annotation = cells_annotation(object, key, annotation), rotation = rotation, ylim = ylim,
    fit_ylim = fit_ylim
  )
}

#' @rdname plot_cluster_violin
#' @export
setMethod("plot_cluster_violin", "SingleCellExperiment", cells_plot_cluster_violin)

#' @rdname plot_cluster_violin
#' @export
setMethod("plot_cluster_violin", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_cluster_violin(object, ...)
})

#' Scatter plot of samples sized by their scores
#'
#' Draws the samples in two dimensions, colored by cluster. Samples with low
#' scores get large, opaque markers and samples with high scores small,
#' faint ones. The legend gives the mean score of each cluster. Samples
#' without a finite score are left out.
#'
#' For a fit, the plot draws the first two columns of `embedding` when it is
#' given. Otherwise it draws `X`, by default the data the fit clustered,
#' reduced to its first two principal components when it has more than two
#' columns. The configuration and the scores are chosen as in
#' [plot_cluster_boxplot()].
#'
#' For a SingleCellExperiment or a Seurat object, the plot draws the reduced
#' dimensions named `basis`. By default it takes the first it finds of
#' `"UMAP"`, `"TSNE"` and `"PCA"` (`"umap"`, `"tsne"` and `"pca"` for a
#' Seurat object), then the reduction CARVE clustered. Without any of these
#' it reads the data CARVE clustered from the object again, and reduces it as
#' for a fit.
#'
#' @inheritParams plot_cluster_boxplot
#' @inheritParams plot_metric_over_n_clusters
#' @param X For a fit, the data to draw, one row per sample. `NULL` uses
#'   [input_data()].
#' @param embedding For a fit, a matrix with one row per sample whose first
#'   two columns are drawn instead of `X`.
#' @param basis For a single-cell object, the name of the reduced dimensions
#'   to draw.
#' @param annotation `TRUE` names the selected configuration and how it was
#'   selected, a string is shown as it is, and `FALSE` shows nothing.
#'   `annotation_style` sets where the text goes.
#' @param alpha_range Opacity of the markers: the value for the highest score
#'   first, then the one for the lowest.
#' @param size_range Area of the markers in square points: the value for the
#'   highest score first, then the one for the lowest.
#' @param sort_order Draw the faint markers first, so the opaque ones are on
#'   top.
#' @param annotation_style Where the annotation goes: `"legend"` above the
#'   legend title, `"box"` in the caption below the panel.
#' @param title,xlabel,ylabel Title and axis labels.
#' @param show_ticks Draw the axis ticks and their labels.
#' @param frameon Draw the border of the panel.
#' @return A ggplot object.
#' @seealso [plot_diagnostic_scatter()], [sample_scores()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_cluster_scatter(fit)
#' plot_cluster_scatter(fit, annotation_style = "box", frameon = TRUE)
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_cluster_scatter(sce, basis = "PCA")
#' @rdname plot_cluster_scatter
#' @export
setMethod("plot_cluster_scatter", "CARVE", function(object, source = "gini", measure = "stability",
                                                    rule = "1se", not_two = FALSE,
                                                    mode = "default", k = NULL,
                                                    sweep_value = NULL, X = NULL,
                                                    embedding = NULL, palette = "Accent",
                                                    alpha_range = c(0.45, 0.9),
                                                    size_range = c(15, 60), sort_order = TRUE,
                                                    legend = TRUE, legend_loc = "right",
                                                    annotation = TRUE,
                                                    annotation_style = "legend", title = NULL,
                                                    xlabel = "Component 1",
                                                    ylabel = "Component 2", show_ticks = FALSE,
                                                    frameon = FALSE, ...) {
  check_dots(...)
  inputs <- fit_sample_plot_inputs(
    object, source, measure, rule, not_two, mode, k, sweep_value, annotation,
    tight_layout = identical(annotation_style, "legend")
  )
  cluster_scatter_plot(
    if (is.null(X)) object@input_data else X,
    inputs$labels, inputs$scores,
    embedding = embedding, palette = palette, alpha_range = alpha_range, size_range = size_range,
    sort_order = sort_order, legend = legend, legend_loc = legend_loc,
    annotation = inputs$annotation, annotation_style = annotation_style, title = title,
    scores_name = SOURCE_NAMES[[source]], xlabel = xlabel, ylabel = ylabel,
    show_ticks = show_ticks, frameon = frameon
  )
})

cells_plot_cluster_scatter <- function(object, key = "carve", source = "gini", basis = NULL,
                                       palette = "Accent", alpha_range = c(0.45, 0.9),
                                       size_range = c(15, 60), sort_order = TRUE, legend = TRUE,
                                       legend_loc = "right", annotation = TRUE,
                                       annotation_style = "legend", title = NULL,
                                       xlabel = "Component 1", ylabel = "Component 2",
                                       show_ticks = FALSE, frameon = FALSE, ...) {
  check_dots(...)
  scores <- cells_scores(object, key, source)
  cluster_scatter_plot(
    cells_scatter_data(object, key, basis), cells_labels(object, key), scores,
    palette = palette, alpha_range = alpha_range, size_range = size_range,
    sort_order = sort_order, legend = legend, legend_loc = legend_loc,
    annotation = cells_annotation(object, key, annotation), annotation_style = annotation_style,
    title = title, scores_name = SOURCE_LABELS[[source]], xlabel = xlabel, ylabel = ylabel,
    show_ticks = show_ticks, frameon = frameon
  )
}

#' @rdname plot_cluster_scatter
#' @export
setMethod("plot_cluster_scatter", "SingleCellExperiment", cells_plot_cluster_scatter)

#' @rdname plot_cluster_scatter
#' @export
setMethod("plot_cluster_scatter", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_cluster_scatter(object, ...)
})

#' Scatter plot of samples colored by their scores
#'
#' Draws the samples in two dimensions with one marker shape per cluster and
#' the per-sample score as the fill color. With the default `cmap` low
#' scores are dark green and high scores nearly white. Samples without a
#' finite score are drawn in faint gray. The data, the configuration and the
#' scores are chosen as in [plot_cluster_scatter()].
#'
#' The clusters are drawn in order of their mean score, highest first. With
#' `sort_order = TRUE` the samples of each cluster are drawn from the highest
#' score to the lowest, so the lowest scores end up on top.
#'
#' @inheritParams plot_cluster_scatter
#' @param cmap Fill colors of the scores, as in [plot_consensus_matrix()].
#' @param alpha_encoding Show the score as opacity too.
#' @param alpha_range Opacity when `alpha_encoding = TRUE`: the value for the
#'   highest score first, then the one for the lowest.
#' @param marker_size Area of the markers, in square points.
#' @param marker_linewidth Width of the marker outlines, in points.
#' @param markers Marker shapes, one per cluster in label order: ggplot2
#'   shapes 21 to 25, the shapes with a fill. `NULL` uses 21, 22, 24, 23 and
#'   25 (circle, square, triangle, diamond and inverted triangle). The shapes
#'   repeat, with a warning, when there are more clusters than shapes.
#' @param sort_order Draw the samples of each cluster from the highest score
#'   to the lowest.
#' @param legend Show the legend of the marker shapes.
#' @param colorbar Show the color bar, below the panel.
#' @param colorbar_label Title of the color bar. `NULL` names the score.
#' @return A ggplot object.
#' @seealso [plot_cluster_scatter()], [sample_scores()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_diagnostic_scatter(fit)
#' plot_diagnostic_scatter(fit, source = "accuracy", cmap = "magma")
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_diagnostic_scatter(sce)
#' @rdname plot_diagnostic_scatter
#' @export
setMethod("plot_diagnostic_scatter", "CARVE", function(object, source = "gini",
                                                       measure = "stability", rule = "1se",
                                                       not_two = FALSE, mode = "default",
                                                       k = NULL, sweep_value = NULL, X = NULL,
                                                       embedding = NULL, cmap = "Greens_r",
                                                       alpha_encoding = TRUE,
                                                       alpha_range = c(0.3, 1), marker_size = 30,
                                                       marker_linewidth = 0.2, markers = NULL,
                                                       sort_order = TRUE, legend = TRUE,
                                                       legend_loc = "right", colorbar = TRUE,
                                                       colorbar_label = NULL, annotation = TRUE,
                                                       annotation_style = "legend",
                                                       title = NULL, xlabel = "Component 1",
                                                       ylabel = "Component 2",
                                                       show_ticks = FALSE, frameon = FALSE, ...) {
  check_dots(...)
  inputs <- fit_sample_plot_inputs(
    object, source, measure, rule, not_two, mode, k, sweep_value, annotation,
    tight_layout = identical(annotation_style, "legend")
  )
  diagnostic_scatter_plot(
    if (is.null(X)) object@input_data else X,
    inputs$labels, inputs$scores,
    embedding = embedding, cmap = cmap, alpha_encoding = alpha_encoding,
    alpha_range = alpha_range, marker_size = marker_size, marker_linewidth = marker_linewidth,
    markers = markers, sort_order = sort_order, legend = legend, legend_loc = legend_loc,
    colorbar = colorbar, colorbar_label = colorbar_label, annotation = inputs$annotation,
    annotation_style = annotation_style, title = title, scores_name = SOURCE_NAMES[[source]],
    xlabel = xlabel, ylabel = ylabel, show_ticks = show_ticks, frameon = frameon
  )
})

cells_plot_diagnostic_scatter <- function(object, key = "carve", source = "gini", basis = NULL,
                                          cmap = "Greens_r", alpha_encoding = TRUE,
                                          alpha_range = c(0.3, 1), marker_size = 30,
                                          marker_linewidth = 0.2, markers = NULL,
                                          sort_order = TRUE, legend = TRUE,
                                          legend_loc = "right", colorbar = TRUE,
                                          colorbar_label = NULL, annotation = TRUE,
                                          annotation_style = "legend", title = NULL,
                                          xlabel = "Component 1", ylabel = "Component 2",
                                          show_ticks = FALSE, frameon = FALSE, ...) {
  check_dots(...)
  scores <- cells_scores(object, key, source)
  diagnostic_scatter_plot(
    cells_scatter_data(object, key, basis), cells_labels(object, key), scores,
    cmap = cmap, alpha_encoding = alpha_encoding, alpha_range = alpha_range,
    marker_size = marker_size, marker_linewidth = marker_linewidth, markers = markers,
    sort_order = sort_order, legend = legend, legend_loc = legend_loc, colorbar = colorbar,
    colorbar_label = if (is.null(colorbar_label)) SOURCE_LABELS[[source]] else colorbar_label,
    annotation = cells_annotation(object, key, annotation), annotation_style = annotation_style,
    title = title, scores_name = SOURCE_LABELS[[source]], xlabel = xlabel, ylabel = ylabel,
    show_ticks = show_ticks, frameon = frameon
  )
}

#' @rdname plot_diagnostic_scatter
#' @export
setMethod("plot_diagnostic_scatter", "SingleCellExperiment", cells_plot_diagnostic_scatter)

#' @rdname plot_diagnostic_scatter
#' @export
setMethod("plot_diagnostic_scatter", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_diagnostic_scatter(object, ...)
})
