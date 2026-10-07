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

# Values for every drawing argument of the per-sample plots, each different
# from its default and from the others.
box_style <- list(
  order = c(3, 1, 2), palette = "Set 1", showfliers = TRUE, width = 0.5, title = "T", xlabel = "X",
  ylabel = "Y", annotation = "note", rotation = 45, ylim = c(0, 1), fit_ylim = FALSE
)
violin_style <- list(
  order = c(3, 1, 2), palette = "Set 1", density_norm = "count", stripplot = FALSE, jitter = 0.05,
  size = 20, alpha = 0.5, inner = "quartile", title = "T", xlabel = "X", ylabel = "Y",
  annotation = "note", rotation = 45, ylim = c(0, 1), fit_ylim = FALSE
)
scatter_style <- list(
  palette = "Set 1", alpha_range = c(0.2, 0.7), size_range = c(5, 50), sort_order = FALSE,
  legend = FALSE, legend_loc = "bottom", annotation = "note", annotation_style = "box", title = "T",
  xlabel = "X", ylabel = "Y", show_ticks = TRUE, frameon = TRUE
)
diagnostic_style <- list(
  cmap = "magma", alpha_encoding = FALSE, alpha_range = c(0.1, 0.6), marker_size = 50,
  marker_linewidth = 1, markers = c(22, 21, 23), sort_order = FALSE, legend = FALSE,
  legend_loc = "bottom", colorbar = FALSE, colorbar_label = "Bar", annotation = "note",
  annotation_style = "box", title = "T", xlabel = "X", ylabel = "Y", show_ticks = TRUE,
  frameon = TRUE
)

test_that("fit_sample_plot_inputs selects once and reads the scores, labels and annotation", {
  calls <- list()
  local_mocked_bindings(
    select_row = function(fit, measure, rule, not_two = FALSE, k = NULL, sweep_value = NULL) {
      calls$select <<- list(measure = measure, rule = rule, not_two = not_two, k = k, sweep_value = sweep_value)
      list(row = estimator_results(fit)[3L, ], config_id = 2L, n_clusters = 7L, pinned = TRUE)
    },
    get_labels = function(fit, ...) {
      calls$labels <<- list(...)
      rep(1:3, each = 30L)
    }
  )
  inputs <- fit_sample_plot_inputs(
    fit, source = "ce", measure = "generalizability", rule = "max", not_two = TRUE,
    mode = "generalizability", k = NULL, sweep_value = 4, annotation = TRUE, tight_layout = TRUE
  )
  expect_identical(calls$select, list(measure = "generalizability", rule = "max", not_two = TRUE, k = NULL, sweep_value = 4))
  expect_identical(
    calls$labels[c("measure", "rule", "not_two", "k", "sweep_value", "consensus_k", "mode")],
    list(measure = "generalizability", rule = "max", not_two = TRUE, k = NULL, sweep_value = 4, consensus_k = 7L, mode = "generalizability")
  )
  expect_identical(inputs$scores, fit@stability_ce_scores[["2"]])
  expect_identical(inputs$labels, rep(1:3, each = 30L))
  expect_identical(
    inputs$annotation,
    annotation_text("generalizability", "max", estimator_results(fit), estimator_results(fit)[3L, ], 7L, pinned = TRUE, tight_layout = TRUE)
  )
})

test_that("an annotation is built, taken as given or left off", {
  inputs <- function(annotation) {
    fit_sample_plot_inputs(fit, "gini", "stability", "1se", FALSE, "default", NULL, NULL, annotation)
  }
  selected <- select_row(fit, "stability", "1se")
  expect_identical(
    inputs(TRUE)$annotation,
    annotation_text("stability", "1se", estimator_results(fit), selected$row, selected$n_clusters)
  )
  expect_identical(inputs("note")$annotation, "note")
  expect_null(inputs(FALSE)$annotation)
})

test_that("the fit's box and violin plots pass every argument to the drawing", {
  local_mocked_bindings(
    cluster_boxplot_plot = function(...) list(...),
    cluster_violin_plot = function(...) list(...)
  )
  id <- select_row(fit, "stability", "1se", k = 4)$config_id
  box <- do.call(plot_cluster_boxplot, c(list(fit, source = "ce", k = 4), box_style))
  expect_identical(box[[1L]], sample_scores(fit, id, "ce"))
  expect_identical(box[[2L]], get_labels(fit, k = 4))
  expect_args(box, box_style)
  violin <- do.call(plot_cluster_violin, c(list(fit, source = "accuracy", k = 4), violin_style))
  expect_identical(violin[[1L]], sample_scores(fit, id, "accuracy"))
  expect_identical(violin[[2L]], get_labels(fit, k = 4))
  expect_args(violin, violin_style)
  expect_identical(plot_cluster_boxplot(fit, source = "accuracy")$ylabel, "Cluster Generalizability")
  expect_identical(plot_cluster_violin(fit, source = "ce")$ylabel, "Cluster Stability (CE)")
  expect_identical(plot_cluster_violin(fit)$ylabel, "Cluster Stability (Gini)")
})

test_that("the fit's scatter plots pass every argument to the drawing", {
  local_mocked_bindings(
    cluster_scatter_plot = function(...) list(...),
    diagnostic_scatter_plot = function(...) list(...)
  )
  id <- select_row(fit, "stability", "1se", k = 4)$config_id
  other_X <- blobs$X * 2
  embedding <- list(embedding = blobs$X[, 2:1])
  scatter <- do.call(plot_cluster_scatter, c(list(fit, source = "ce", k = 4, X = other_X), embedding, scatter_style))
  expect_identical(scatter[[1L]], other_X)
  expect_identical(scatter[[2L]], get_labels(fit, k = 4))
  expect_identical(scatter[[3L]], sample_scores(fit, id, "ce"))
  expect_args(scatter, c(embedding, scatter_style, list(scores_name = "CE Stability")))
  diagnostic <- do.call(plot_diagnostic_scatter, c(list(fit, source = "accuracy", k = 4, X = other_X), embedding, diagnostic_style))
  expect_identical(diagnostic[[1L]], other_X)
  expect_identical(diagnostic[[3L]], sample_scores(fit, id, "accuracy"))
  expect_args(diagnostic, c(embedding, diagnostic_style, list(scores_name = "Generalizability")))
  defaults <- plot_cluster_scatter(fit)
  expect_identical(defaults[[1L]], input_data(fit))
  expect_null(defaults$embedding)
  expect_identical(defaults$scores_name, "Gini Stability")
  expect_null(plot_diagnostic_scatter(fit)$colorbar_label)
})

test_that("the fit's scatter annotation is tight in the legend and plain in the caption", {
  local_mocked_bindings(
    cluster_scatter_plot = function(...) list(...),
    diagnostic_scatter_plot = function(...) list(...)
  )
  selected <- select_row(fit, "stability", "1se")
  text <- function(tight) {
    annotation_text("stability", "1se", estimator_results(fit), selected$row, selected$n_clusters, tight_layout = tight)
  }
  expect_identical(plot_cluster_scatter(fit)$annotation, text(TRUE))
  expect_identical(plot_cluster_scatter(fit, annotation_style = "box")$annotation, text(FALSE))
  expect_identical(plot_diagnostic_scatter(fit)$annotation, text(TRUE))
  expect_identical(plot_diagnostic_scatter(fit, annotation_style = "box")$annotation, text(FALSE))
})

test_that("the single-cell sample plots pass every argument and read the stored columns", {
  local_mocked_bindings(
    cluster_boxplot_plot = function(...) list(...),
    cluster_violin_plot = function(...) list(...),
    cluster_scatter_plot = function(...) list(...),
    diagnostic_scatter_plot = function(...) list(...)
  )
  pinned <- attach_results(make_sce(blobs$X, reductions = list(PCA = blobs$X, UMAP = blobs$X * 2)), fit, k = 4)
  id <- select_row(fit, "stability", "1se", k = 4)$config_id
  labels <- get_labels(fit, k = 4)
  box <- do.call(plot_cluster_boxplot, c(list(pinned, source = "ce"), box_style))
  expect_equal(box[[1L]], sample_scores(fit, id, "ce"))
  expect_identical(box[[2L]], labels)
  expect_args(box, box_style)
  violin <- do.call(plot_cluster_violin, c(list(pinned, source = "accuracy"), violin_style))
  expect_equal(violin[[1L]], sample_scores(fit, id, "accuracy"))
  expect_identical(violin[[2L]], labels)
  expect_args(violin, violin_style)
  defaults <- plot_cluster_violin(pinned)
  expect_equal(defaults[[1L]], sample_scores(fit, id, "gini"))
  expect_identical(defaults$ylabel, "Cluster Stability (Gini)")
  expect_identical(defaults$annotation, cells_annotation(pinned, "carve", TRUE))
  scatter <- do.call(plot_cluster_scatter, c(list(pinned, source = "ce", basis = "PCA"), scatter_style))
  expect_equal(scatter[[1L]], blobs$X)
  expect_identical(scatter[[2L]], labels)
  expect_equal(scatter[[3L]], sample_scores(fit, id, "ce"))
  expect_args(scatter, c(scatter_style, list(scores_name = "Cluster Stability (CE)")))
  expect_equal(plot_cluster_scatter(pinned)[[1L]], blobs$X * 2)
  diagnostic <- do.call(plot_diagnostic_scatter, c(list(pinned, source = "accuracy", basis = "PCA"), diagnostic_style))
  expect_equal(diagnostic[[1L]], blobs$X)
  expect_equal(diagnostic[[3L]], sample_scores(fit, id, "accuracy"))
  expect_args(diagnostic, c(diagnostic_style, list(scores_name = "Cluster Generalizability")))
  diagnostic_defaults <- plot_diagnostic_scatter(pinned)
  expect_identical(diagnostic_defaults$colorbar_label, "Cluster Stability (Gini)")
  expect_identical(diagnostic_defaults$annotation, cells_annotation(pinned, "carve", TRUE))
})

test_that("the single-cell annotation reads the recorded configuration by its config_id", {
  stored <- attach_results(sce, resolution_fit, sweep_value = 1)
  params <- S4Vectors::metadata(stored)$carve$params
  results <- estimator_results(resolution_fit)
  # Reversed, a row's position no longer equals its config_id.
  S4Vectors::metadata(stored)$carve$results <- results[rev(seq_len(nrow(results))), ]
  row <- results[results$config_id == params$selected_config_id, , drop = FALSE]
  text <- cells_annotation(stored, "carve", TRUE)
  expect_identical(text, annotation_text("stability", "1se", results, row, params$selected_k, pinned = TRUE))
  expect_match(text, "resolution = 1, k ~ ", fixed = TRUE)
  expect_match(text, ", fixed)", fixed = TRUE)
  expect_identical(cells_annotation(stored, "carve", "note"), "note")
  expect_null(cells_annotation(stored, "carve", FALSE))
  S4Vectors::metadata(stored)$carve$params$selected_k <- 0L
  expect_match(cells_annotation(stored, "carve", TRUE), "k ~ None", fixed = TRUE)
  S4Vectors::metadata(stored)$carve$results <- NULL
  expect_null(cells_annotation(stored, "carve", TRUE))
})

test_that("the single-cell scatter plots draw a basis, the clustered reduction or the clustered data", {
  with_umap <- attach_results(make_sce(blobs$X, reductions = list(PCA = blobs$X, UMAP = blobs$X * 2)), fit)
  expect_equal(cells_scatter_data(with_umap, "carve", NULL), blobs$X * 2)
  expect_equal(cells_scatter_data(with_umap, "carve", "PCA"), blobs$X)
  # Read again from the object, the three-column reduction would be reduced
  # to principal components; as the basis it gives its first two columns.
  harmony <- cbind(blobs$X * 4, 1)
  custom <- attach_results(make_sce(blobs$X, reductions = list(harmony = harmony)), fit, reduction = "harmony")
  expect_equal(cells_scatter_data(custom, "carve", NULL), blobs$X * 4)
  wide <- cbind(blobs$X, blobs$X * 3)
  assay_only <- attach_results(
    make_sce(wide, reductions = list(), assays = list(logcounts = t(wide))), fit,
    assay = "logcounts", n_dims = 3
  )
  expect_equal(unname(cells_scatter_data(assay_only, "carve", NULL)), wide[, 1:3])
  expect_error(
    cells_scatter_data(with_umap, "carve", "tsne"),
    "basis='tsne' not found in reducedDimNames(object). Available names: ['PCA', 'UMAP']",
    fixed = TRUE
  )
})

test_that("a SingleCellExperiment draws the sample plots the fit draws", {
  expect_same_plot(plot_cluster_boxplot(written), plot_cluster_boxplot(fit))
  expect_same_plot(plot_cluster_violin(written, source = "accuracy"), plot_cluster_violin(fit, source = "accuracy"))
  expect_same_plot(plot_cluster_scatter(written), plot_cluster_scatter(fit))
  expect_same_plot(plot_diagnostic_scatter(written), plot_diagnostic_scatter(fit))
  # The legend titles differ as they do in Python: the single-cell plots
  # name the score with SOURCE_LABELS and do not tighten the annotation.
  selected <- select_row(fit, "stability", "1se")
  text <- function(tight) {
    annotation_text("stability", "1se", estimator_results(fit), selected$row, selected$n_clusters, tight_layout = tight)
  }
  fill_name <- function(plot) plot$scales$get_scales("fill")$name
  expect_identical(fill_name(plot_cluster_scatter(written)), paste0(text(FALSE), "\nCluster, Cluster Stability (Gini)"))
  expect_identical(fill_name(plot_cluster_scatter(fit)), paste0(text(TRUE), "\nCluster, Gini Stability"))
  expect_identical(fill_name(plot_diagnostic_scatter(written)), "Cluster Stability (Gini)")
  expect_identical(fill_name(plot_diagnostic_scatter(fit)), "Gini Stability")
  expect_null(plot_cluster_violin(attach_results(sce, fit, store_results = FALSE))$labels$caption)
})

test_that("a Seurat object draws the sample plots a SingleCellExperiment draws", {
  object <- attach_results(make_seurat(blobs$X), fit)
  expect_same_plot(plot_cluster_violin(object), plot_cluster_violin(written))
  expect_same_plot(plot_diagnostic_scatter(object), plot_diagnostic_scatter(written))
  stability_only <- attach_results(make_seurat(blobs$X), stability_fit, mode = "stability")
  expect_error(
    plot_cluster_boxplot(stability_only, source = "accuracy"),
    "object@meta.data$carve_generalizability not found, so source='accuracy' is unavailable.",
    fixed = TRUE
  )
})

test_that("the per-sample plots name a missing score, an unknown source or mode, and an unknown argument", {
  stability_only <- attach_results(sce, stability_fit, mode = "stability")
  expect_error(
    plot_cluster_violin(stability_only, source = "accuracy"),
    "colData(object)$carve_generalizability not found, so source='accuracy' is unavailable. It is written when run_carve runs with mode='default' or mode='generalizability'.",
    fixed = TRUE
  )
  expect_error(plot_cluster_boxplot(written, source = "nope"), "source must be one of ['accuracy', 'ce', 'gini']; got 'nope'.", fixed = TRUE)
  expect_error(plot_cluster_boxplot(fit, source = "nope"), "source must be one of: 'accuracy', 'gini', 'ce'.", fixed = TRUE)
  expect_error(plot_cluster_violin(fit, mode = "nope"), "mode must be one of: 'default', 'stability', 'generalizability'.", fixed = TRUE)
  expect_error(
    plot_cluster_violin(stability_fit, source = "accuracy"),
    "Generalizability scores are not available for this run.",
    fixed = TRUE
  )
  expect_error(plot_diagnostic_scatter(written, nope = 1), "Unknown argument: nope.", fixed = TRUE)
  expect_error(plot_cluster_scatter(fit, basis = "UMAP"), "Unknown argument: basis.", fixed = TRUE)
  expect_error(
    plot_cluster_boxplot(blobs$X),
    "object must be a CARVE fit, a SingleCellExperiment or a Seurat object, not an object of class 'matrix'.",
    fixed = TRUE
  )
})

test_that("a Seurat object reaches the ANY method of every plot generic, and a matrix is refused", {
  skip_if_not_installed("SeuratObject")
  generics <- list(
    plot_metric_over_n_clusters = fit,
    plot_metric_by_pipeline = randomized_fit,
    plot_n_clusters_over_sweep = resolution_fit,
    plot_consensus_matrix = fit,
    plot_cluster_boxplot = fit,
    plot_cluster_violin = fit,
    plot_cluster_scatter = fit,
    plot_diagnostic_scatter = fit
  )
  for (name in names(generics)) {
    plot_function <- get(name)
    object <- attach_results(make_seurat(blobs$X), generics[[name]])
    expect_s3_class(plot_function(object), "ggplot")
    # The same plot as the SingleCellExperiment method draws, so an ANY
    # method that calls another generic's function is caught.
    expect_same_plot(plot_function(object), plot_function(attach_results(sce, generics[[name]])))
    expect_error(
      plot_function(blobs$X),
      "object must be a CARVE fit, a SingleCellExperiment or a Seurat object, not an object of class 'matrix'.",
      fixed = TRUE
    )
  }
})
