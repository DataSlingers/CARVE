# Fits shared by the plot tests. The k-axis fits have three configurations,
# KMeans at k = 2, 3 and 4.
blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
fit_blobs <- function(...) {
  carve(blobs$X, n_resamples = 6, estimator_param_grids = k_grid, random_state = 0, ...)
}
fit <- fit_blobs()
stability_fit <- muffle(fit_blobs(mode = "stability"), "Non-default mode is experimental")
anchored <- muffle(fit_blobs(anchor_threshold = 30), "anchored consensus")
randomized_fit <- fit_blobs(
  randomize_preprocessing = TRUE,
  normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
  dim_reduction_options = list(preprocessing_option(Identity))
)
resolution_fit <- carve(
  blobs$X, n_resamples = 4, random_state = 0,
  estimator_param_grids = list(estimator_grid(LeidenClustering, resolution = c(0.5, 1)))
)
sce <- make_sce(blobs$X)
written <- attach_results(sce, fit)
# Values that differ from every default and from each other, so a dropped
# or swapped argument shows.
style <- list(title = "T", xlabel = "X", ylabel = "Y", legend = FALSE, legend_loc = "bottom", palette = "viridis")

test_that("labels_mode maps a plot's mode to a get_labels() mode", {
  expect_identical(labels_mode("default"), "default")
  expect_identical(labels_mode("stability"), "default")
  expect_identical(labels_mode("generalizability"), "generalizability")
  expect_error(labels_mode("nope"), "mode must be one of: 'default', 'stability', 'generalizability'.", fixed = TRUE)
})

test_that("recorded() takes the argument, then the record, then the default", {
  params <- list(rule = "max", not_two = FALSE)
  expect_identical(recorded("quantile", params, "rule", "1se"), "quantile")
  expect_identical(recorded(NULL, params, "rule", "1se"), "max")
  expect_identical(recorded(NULL, params, "measure", "stability"), "stability")
  expect_false(recorded(NULL, params, "not_two", TRUE))
})

test_that("the fit's line plots pass every argument to the drawing", {
  local_mocked_bindings(
    metric_over_sweep_plot = function(...) list(...),
    n_clusters_over_sweep_plot = function(...) list(...),
    metric_by_pipeline_plot = function(...) list(...)
  )
  selection <- list(measure = "generalizability", rule = "max", not_two = TRUE)
  over <- do.call(plot_metric_over_n_clusters, c(list(fit), selection, style))
  expect_identical(over[[1L]], estimator_results(fit))
  expect_args(over, c(selection, style))
  counts <- do.call(plot_n_clusters_over_sweep, c(list(resolution_fit), selection, style))
  expect_identical(counts[[1L]], estimator_results(resolution_fit))
  expect_args(counts, c(selection, style))
  pipelines <- do.call(plot_metric_by_pipeline, c(list(randomized_fit, method_id = "m9"), selection, style))
  expect_identical(pipelines[[1L]], preprocessing_results(randomized_fit))
  expect_identical(pipelines[[2L]], estimator_results(randomized_fit))
  expect_args(pipelines, c(list(method_id = "m9"), selection, style))
  expect_error(plot_metric_over_n_clusters(fit, key = "carve"), "Unknown argument: key.", fixed = TRUE)
})

test_that("plot_metric_by_pipeline draws the configuration the selection picks", {
  seen <- NULL
  local_mocked_bindings(
    select_row = function(fit, measure, rule, not_two = FALSE, k = NULL, sweep_value = NULL) {
      seen <<- list(measure, rule, not_two)
      list(row = data.frame(method_id = "m7"))
    },
    metric_by_pipeline_plot = function(...) list(...)
  )
  args <- plot_metric_by_pipeline(randomized_fit, measure = "generalizability", rule = "max", not_two = TRUE)
  expect_identical(args$method_id, "m7")
  expect_identical(seen, list("generalizability", "max", TRUE))
})

test_that("the fit's heatmap passes the selection and the style through", {
  calls <- list()
  local_mocked_bindings(
    select_row = function(fit, measure, rule, not_two = FALSE, k = NULL, sweep_value = NULL) {
      calls$select <<- list(measure = measure, rule = rule, not_two = not_two, k = k, sweep_value = sweep_value)
      list(row = estimator_results(fit)[3L, ], config_id = 2L, n_clusters = 7L, pinned = TRUE)
    },
    get_labels = function(fit, ...) {
      calls$labels <<- list(...)
      rep(1:3, each = 30L)
    },
    consensus_plot = function(...) list(...)
  )
  heatmap_style <- list(cmap = "magma", palette = "Set 1", colorbar = FALSE, colorbar_label = "Share", title = "T")
  args <- do.call(plot_consensus_matrix, c(
    list(fit, measure = "generalizability", rule = "max", not_two = TRUE, sweep_value = 4),
    heatmap_style
  ))
  expect_identical(calls$select, list(measure = "generalizability", rule = "max", not_two = TRUE, k = NULL, sweep_value = 4))
  expect_identical(
    calls$labels[c("measure", "rule", "not_two", "k", "sweep_value", "consensus_k", "mode")],
    list(measure = "generalizability", rule = "max", not_two = TRUE, k = NULL, sweep_value = 4, consensus_k = 7L, mode = "default")
  )
  expect_identical(args[[1L]], fit@consensus_matrices[["2"]])
  expect_identical(args[[2L]], rep(1:3, each = 30L))
  expect_args(args, heatmap_style)
})

test_that("the fit's heatmap draws the selected matrix with the labels of its cut", {
  id <- select_row(fit, "stability", "1se", k = 4)$config_id
  expect_same_plot(
    plot_consensus_matrix(fit, k = 4),
    consensus_plot(consensus_matrix(fit, id), get_labels(fit, k = 4))
  )
})

test_that("mode chooses the consensus the heatmap draws", {
  id <- select_row(fit, "stability", "1se")$config_id
  expect_same_plot(
    plot_consensus_matrix(fit, mode = "generalizability"),
    consensus_plot(consensus_matrix(fit, id, type = "generalizability"), get_labels(fit, mode = "generalizability"))
  )
  expect_same_plot(plot_consensus_matrix(fit, mode = "stability"), plot_consensus_matrix(fit))
  expect_error(plot_consensus_matrix(fit, mode = "nope"), "mode must be one of: 'default', 'stability', 'generalizability'.", fixed = TRUE)
  expect_error(
    plot_consensus_matrix(stability_fit, mode = "generalizability"),
    "Selected consensus matrix is not available for mode='generalizability'.",
    fixed = TRUE
  )
})

test_that("an anchored fit draws the anchor block with the anchors' labels", {
  plot <- plot_consensus_matrix(anchored)
  expect_identical(dim(raster_colours(raster_layers(plot)[[1L]])), c(30L, 30L))
  id <- select_row(anchored, "stability", "1se")$config_id
  expect_same_plot(
    plot,
    consensus_plot(consensus_matrix(anchored, id), get_labels(anchored)[consensus_anchors(anchored)])
  )
})

test_that("the single-cell line plots default to the recorded selection", {
  local_mocked_bindings(
    metric_over_sweep_plot = function(...) list(...),
    n_clusters_over_sweep_plot = function(...) list(...),
    metric_by_pipeline_plot = function(...) list(...)
  )
  stored <- attach_results(sce, randomized_fit, measure = "generalizability", rule = "max", not_two = TRUE)
  record <- S4Vectors::metadata(stored)$carve
  from_record <- list(measure = "generalizability", rule = "max", not_two = TRUE)
  defaults <- list(title = NULL, xlabel = NULL, ylabel = NULL, legend = TRUE, legend_loc = "right", palette = "Accent")
  over <- plot_metric_over_n_clusters(stored)
  expect_identical(over[[1L]], record$results)
  expect_args(over, c(from_record, defaults))
  expect_args(plot_n_clusters_over_sweep(stored), c(from_record, defaults))
  pipelines <- plot_metric_by_pipeline(stored)
  expect_identical(pipelines[[1L]], record$preprocessing_results)
  expect_identical(pipelines[[2L]], record$results)
  expect_args(pipelines, c(list(method_id = record$params$selected_method_id), from_record, defaults))
  # An argument given, even FALSE, wins over the record.
  chosen <- list(measure = "average", rule = "quantile", not_two = FALSE)
  expect_args(do.call(plot_metric_over_n_clusters, c(list(stored), chosen, style)), c(chosen, style))
  expect_args(do.call(plot_n_clusters_over_sweep, c(list(stored), chosen, style)), c(chosen, style))
  expect_args(
    do.call(plot_metric_by_pipeline, c(list(stored, method_id = "m9"), chosen, style)),
    c(list(method_id = "m9"), chosen, style)
  )
})

test_that("a single-cell plot_metric_by_pipeline plots stability when the recorded measure is not in the table", {
  pac <- attach_results(sce, randomized_fit, measure = "pac", rule = "max")
  method_id <- S4Vectors::metadata(pac)$carve$params$selected_method_id
  expect_warning(
    plot <- plot_metric_by_pipeline(pac),
    "run_carve recorded measure 'pac', which the per-pipeline table does not carry; plotting 'stability'. Pass measure='generalizability' for the other ARI criterion.",
    fixed = TRUE
  )
  expect_same_plot(plot, plot_metric_by_pipeline(randomized_fit, method_id = method_id, rule = "max"))
  expect_error(plot_metric_by_pipeline(pac, measure = "pac"), "The per-pipeline table carries only the ARI criteria", fixed = TRUE)
})

test_that("a SingleCellExperiment draws what the fit draws", {
  expect_same_plot(plot_metric_over_n_clusters(written), plot_metric_over_n_clusters(fit))
  expect_same_plot(plot_consensus_matrix(written), plot_consensus_matrix(fit))
  expect_same_plot(
    plot_n_clusters_over_sweep(attach_results(sce, resolution_fit)),
    plot_n_clusters_over_sweep(resolution_fit)
  )
  expect_same_plot(
    plot_metric_by_pipeline(attach_results(sce, randomized_fit)),
    plot_metric_by_pipeline(randomized_fit)
  )
  expect_error(plot_consensus_matrix(written, nope = 1), "Unknown argument: nope.", fixed = TRUE)
})

test_that("a Seurat object draws what the fit draws", {
  object <- attach_results(make_seurat(blobs$X), fit)
  expect_same_plot(plot_metric_over_n_clusters(object), plot_metric_over_n_clusters(fit))
  expect_same_plot(plot_consensus_matrix(object), plot_consensus_matrix(fit))
  expect_error(
    plot_metric_over_n_clusters(object, key = "other"),
    "Misc(object, 'other') not found. Run `run_carve(object, key = 'other')` first, or pass the key you used.",
    fixed = TRUE
  )
  expect_error(plot_consensus_matrix(object, nope = 1), "Unknown argument: nope.", fixed = TRUE)
})

test_that("stored labels are read as their factor codes", {
  relabeled <- written
  SummarizedExperiment::colData(relabeled)$carve <- factor(rep(c("b", "a", "c"), each = 30L))
  expect_identical(cells_labels(relabeled, "carve"), rep(c(2L, 1L, 3L), each = 30L))
  SummarizedExperiment::colData(relabeled)$carve <- rep(c(3, 1, 2), each = 30L)
  expect_identical(cells_labels(relabeled, "carve"), rep(c(3L, 1L, 2L), each = 30L))
})

test_that("the single-cell plots name what is missing from the object", {
  expect_error(
    plot_metric_over_n_clusters(sce),
    "metadata(object)$carve not found. Run `run_carve(object, key = 'carve')` first, or pass the key you used.",
    fixed = TRUE
  )
  no_params <- sce
  S4Vectors::metadata(no_params)$carve <- list(results = estimator_results(fit))
  expect_error(
    plot_metric_over_n_clusters(no_params),
    "metadata(object)$carve does not look like a CARVE result; expected a list with a 'params' entry.",
    fixed = TRUE
  )
  expect_error(
    plot_n_clusters_over_sweep(attach_results(sce, fit, store_results = FALSE)),
    "metadata(object)$carve$results not found. It is omitted when run_carve runs with store_results=FALSE.",
    fixed = TRUE
  )
  expect_error(
    plot_metric_by_pipeline(written),
    "metadata(object)$carve$preprocessing_results not found. It is written only for a randomized fit, so either run_carve ran without randomize_preprocessing=TRUE or it ran with store_results=FALSE.",
    fixed = TRUE
  )
  expect_error(
    plot_consensus_matrix(attach_results(sce, fit, store_consensus = FALSE)),
    "metadata(object)$carve$consensus not found. It is omitted when run_carve runs with store_consensus=FALSE, or when anchored consensus was active: an anchor block is m-by-m rather than n_obs-by-n_obs and metadata(object)$carve holds only the latter.",
    fixed = TRUE
  )
  unlabeled <- written
  SummarizedExperiment::colData(unlabeled)$carve <- NULL
  expect_error(
    plot_consensus_matrix(unlabeled),
    "colData(object)$carve not found. Run `run_carve(object, key = 'carve')` first.",
    fixed = TRUE
  )
})

test_that("a plot of anything else names the classes the plots take", {
  expect_error(
    plot_metric_over_n_clusters(blobs$X),
    "object must be a CARVE fit, a SingleCellExperiment or a Seurat object, not an object of class 'matrix'.",
    fixed = TRUE
  )
  expect_error(plot_consensus_matrix(list()), "not an object of class 'list'.", fixed = TRUE)
})
