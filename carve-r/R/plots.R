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

# The stored labels as integers. A factor gives its codes, 1, 2, ..., as
# Python reads the codes of a categorical column.
cells_labels <- function(object, key) {
  column <- cells_get_column(object, key)
  if (is.null(column)) {
    stop(sprintf(
      "%s not found. Run `run_carve(object, key = %s)` first.",
      cells_column_where(object, key), format_repr(key)
    ), call. = FALSE)
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
#' [estimator_results()], pooled over the pipelines, so the lines that cross
#' it compare the pipelines at the selected configuration. For a `method_id`
#' that CARVE did not select, it marks the value `rule` picks among that
#' configuration's rows.
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
#' labels; a matrix over every sample would be too large to draw at the
#' sizes where anchoring applies.
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
