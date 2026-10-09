test_that("palette_colors samples a palette as matplotlib samples a colormap", {
  f <- read_fixture("plotting")
  for (n in seq_along(f$palette)) {
    expect_identical(palette_colors("Accent", n), toupper(unlist(f$palette[[n]])), info = paste("n =", n))
  }
  expect_identical(palette_colors(c("red", "blue"), 3L), c("red", "blue", "blue"))
  expect_identical(palette_colors("Accent", 0L), character())
})

test_that("a palette is a palette name or a vector of colors", {
  expect_identical(colormap_values("Accent"), ACCENT)
  expect_identical(colormap_values("Set 1"), unname(grDevices::palette.colors(palette = "Set 1")))
  expect_identical(colormap_values("viridis"), scales::viridis_pal(option = "viridis")(256L))
  expect_identical(colormap_values("Greens"), GREENS)
  expect_identical(colormap_values("Greens_r"), rev(GREENS))
  expect_identical(colormap_values(c(a = "red", b = "#000000")), c("red", "#000000"))
  expect_error(
    colormap_values("nope"),
    "palette must be a palette name such as 'Accent' or 'viridis', or a vector of colors.",
    fixed = TRUE
  )
  expect_error(colormap_values(3), "palette must be a palette name", fixed = TRUE)
  expect_identical(continuous_colours("Greens_r"), rev(GREENS))
  expect_error(continuous_colours("red"), "cmap must name a colormap or give at least two colors.", fixed = TRUE)
  expect_error(continuous_colours("nope"), "cmap must be a palette name such as 'Accent'", fixed = TRUE)
})

test_that("values map to colors through a 256-level table, as in matplotlib", {
  lut <- colour_lut(c("#000000", "#FFFFFF"))
  expect_length(lut, 256L)
  expect_identical(lut[[1L]], "#000000")
  expect_identical(lut[[256L]], "#FFFFFF")
  expect_identical(lut_colours(c(0, 0.5, 1), lut), lut[c(1L, 129L, 256L)])
  expect_identical(lut_colours(c(1 / 256 - 1e-9, 1 / 256), lut), lut[c(1L, 2L)])
})

test_that("axis, selection and annotation text match Python's", {
  f <- read_fixture("plotting")
  for (column in names(f$ylabels)) {
    expect_identical(measure_ylabel(column), f$ylabels[[column]], info = column)
  }
  for (case in f$selections) {
    row <- data.frame(sweep_value = case$value)
    expect_identical(selected_value_label(row, case$param, case$rule), case$label)
  }
  for (case in f$annotations) {
    results <- data.frame(sweep_param = case$param, sweep_value = case$value, method_label = case$method_label)
    text <- annotation_text(
      case$measure, case$rule, results, results[1L, , drop = FALSE], case$selected_k,
      pinned = case$pinned, tight_layout = case$tight_layout
    )
    # Python writes the approximate cluster count with U+2248, R with "~".
    expect_identical(text, gsub("\u2248", "~", case$text, fixed = TRUE))
  }
})

test_that("the estimator label is the row's method label", {
  row <- data.frame(estimator = "KMeans", method_label = "KMeans, linkage=ward", gamma = 0.123456)
  expect_identical(build_estimator_label(row), "KMeans, linkage=ward")
  wide <- data.frame(method_label = "AgglomerativeClustering, linkage=ward")
  expect_identical(build_estimator_label(wide, tight_layout = TRUE), "AgglomerativeClustering\nlinkage=ward")
  expect_identical(build_estimator_label(data.frame(method_label = "KMeans"), tight_layout = TRUE), "KMeans")
})

test_that("plot_measure_column names Python's plotting errors", {
  results <- data.frame(ari_stability = 1)
  expect_identical(plot_measure_column(results, "s"), "ari_stability")
  expect_error(
    plot_measure_column(results, "nonexistent"),
    "Measure 'nonexistent' not found. Valid options: ['s', 'stab',",
    fixed = TRUE
  )
  expect_error(
    plot_measure_column(results, "pac"),
    "Metric column 'consensus_pac_stability' not found in the results table.",
    fixed = TRUE
  )
})

test_that("integer_breaks keeps whole numbers inside the limits", {
  expect_identical(integer_breaks(c(1.8, 3.2)), c(2, 3))
  expect_identical(integer_breaks(c(3, 11)), c(2, 4, 6, 8, 10, 12))
  expect_identical(integer_breaks(c(2.2, 2.8)), numeric())
})

test_that("sizes and widths convert from matplotlib's units", {
  # A matplotlib marker of area 36 square points is 6 points across; a
  # ggplot2 circle is 0.75 * size * .pt points across.
  expect_equal(0.75 * point_size(36) * ggplot2::.pt, 6)
  # 1 point is 96 / 72 grid line-width units; ggplot2 draws a stroke s with
  # s * .stroke / 2 units and a line width w with w * .pt units.
  expect_equal(edge_stroke(0.6) * ggplot2::.stroke / 2, 0.6 * 96 / 72)
  expect_equal(line_width(1.5) * ggplot2::.pt, 1.5 * 96 / 72)
})

test_that("legend_position takes ggplot2 positions", {
  expect_identical(legend_position("bottom")$legend.position, "bottom")
  inside <- legend_position(c(0.9, 0.1))
  expect_identical(inside$legend.position, "inside")
  expect_identical(inside$legend.position.inside, c(0.9, 0.1))
  expect_error(
    legend_position("best"),
    "legend_loc must be 'right', 'left', 'top', 'bottom', or two numbers giving a position inside the panel.",
    fixed = TRUE
  )
})

test_that("each method is one line, labeled by its method label", {
  plot <- metric_over_sweep_plot(two_method_results())
  lines <- layer_with(plot, "GeomLine")
  expect_identical(length(unique(lines$group)), 2L)
  expect_identical(guide_labels(plot, "colour"), c("KMeans", "SpectralClustering, affinity=self_tuning"))
  expect_identical(sort(unique(lines$colour)), sort(palette_colors("Accent", 2L)))
  expect_identical(plot$scales$get_scales("colour")$name, "Estimators")
})

test_that("the lines carry the metric and its standard error", {
  plot <- metric_over_sweep_plot(metric_results(), measure = "generalizability")
  lines <- layer_with(plot, "GeomLine")
  expect_equal(lines$x, c(2, 3, 4))
  expect_equal(lines$y, c(0.85, 0.80, 0.65))
  bars <- layer_with(plot, "GeomErrorbar")
  expect_equal(bars$ymin, c(0.85, 0.80, 0.65) - c(0.03, 0.04, 0.06))
  expect_equal(bars$ymax, c(0.85, 0.80, 0.65) + c(0.03, 0.04, 0.06))
  stability <- layer_with(metric_over_sweep_plot(metric_results(), measure = "stability"), "GeomErrorbar")
  expect_equal(stability$ymin, c(0.9, 0.85, 0.7) - c(0.02, 0.03, 0.05))
  expect_equal(stability$ymax, c(0.9, 0.85, 0.7) + c(0.02, 0.03, 0.05))
})

test_that("a table without standard errors draws no error bars", {
  results <- metric_results()
  results$ari_stability_se <- NULL
  plot <- suppressWarnings(metric_over_sweep_plot(results))
  expect_false(has_layer(plot, "GeomErrorbar"))
  expect_true(has_layer(plot, "GeomLine"))
})

test_that("the dashed line marks the selected sweep value", {
  plot <- metric_over_sweep_plot(metric_results(), rule = "1se")
  dashed <- layer_with(plot, "GeomVline")
  # 0.9 - 0.02 admits only k = 2.
  expect_equal(dashed$xintercept, 2)
  expect_identical(dashed$linetype, "dashed")
  expect_identical(guide_labels(plot, "linetype"), "Selected k (1-SE rule): 2")
  resolution <- metric_over_sweep_plot(resolution_results(), rule = "max")
  expect_identical(guide_labels(resolution, "linetype"), "Selected resolution (Max rule): 0.25")
  expect_identical(resolution$labels$y, "ARI Stability")
})

test_that("the dashed line is left off when the selection fails", {
  plot <- metric_over_sweep_plot(metric_results()[1L, ], not_two = TRUE)
  expect_false(has_layer(plot, "GeomVline"))
  expect_true(has_layer(plot, "GeomLine"))
})

test_that("axis labels follow the sweep and can be replaced", {
  plot <- metric_over_sweep_plot(metric_results())
  expect_identical(plot$labels$x, "Number of Clusters (k)")
  expect_identical(plot$labels$y, "ARI Stability")
  expect_null(plot$labels$title)
  expect_equal(x_breaks(plot), c(2, 3, 4))
  resolution <- metric_over_sweep_plot(resolution_results())
  expect_identical(resolution$labels$x, "Resolution")
  expect_equal(x_breaks(resolution), c(0.25, 0.5, 1, 2))
  custom <- metric_over_sweep_plot(metric_results(), title = "T", xlabel = "X", ylabel = "Y")
  expect_identical(c(custom$labels$title, custom$labels$x, custom$labels$y), c("T", "X", "Y"))
})

test_that("legend = FALSE drops both legends", {
  plot <- metric_over_sweep_plot(metric_results(), legend = FALSE)
  expect_null(ggplot2::get_guide_data(plot, "colour"))
  expect_null(ggplot2::get_guide_data(plot, "linetype"))
  expect_identical(plot$theme$legend.position, "right")
})

test_that("metric_over_sweep_plot reports a bad measure and an empty table", {
  expect_error(metric_over_sweep_plot(metric_results(), measure = "nonexistent"), "Measure 'nonexistent' not found", fixed = TRUE)
  expect_error(metric_over_sweep_plot(metric_results()[0L, ]), "Results DataFrame is empty.", fixed = TRUE)
})

test_that("plot_metric_by_pipeline draws one line per pipeline of the configuration", {
  plot <- metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0", measure = "generalizability")
  expect_identical(guide_labels(plot, "colour"), c("identity | PCA(n_components=2)", "identity | identity"))
  lines <- layer_with(plot, "GeomLine")
  first <- lines[lines$colour == palette_colors("Accent", 2L)[[2L]], ]
  expect_equal(first$y, c(0.60, 0.80, 0.50))
  second <- lines[lines$colour == palette_colors("Accent", 2L)[[1L]], ]
  expect_equal(second$y, c(0.55, 0.70, 0.40))
  expect_equal(first$x, c(2, 3, 4))
  expect_identical(plot$scales$get_scales("colour")$name, "Pipelines")
  expect_identical(plot$labels$x, "Number of Clusters (k)")
  expect_identical(plot$labels$y, "ARI Generalizability")
})

test_that("plot_metric_by_pipeline marks the value CARVE selects from the pooled table", {
  selected <- metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0", rule = "1se")
  expect_equal(layer_with(selected, "GeomVline")$xintercept, 4)
  expect_identical(guide_labels(selected, "linetype"), "Selected k (1-SE rule): 4")
  other <- metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m1", rule = "1se")
  expect_equal(layer_with(other, "GeomVline")$xintercept, 2)
})

test_that("plot_metric_by_pipeline follows a resolution axis and the palette", {
  as_resolution <- function(table) {
    names(table)[names(table) == "n_clusters"] <- "resolution"
    table$sweep_param <- "resolution"
    table$sweep_value <- table$sweep_value / 4
    table
  }
  plot <- metric_by_pipeline_plot(as_resolution(pipeline_results()), as_resolution(pooled_results()), "m0")
  expect_identical(plot$labels$x, "Resolution")
  expect_equal(x_breaks(plot), c(0.5, 0.75, 1))
  expect_equal(layer_with(plot, "GeomVline")$xintercept, 1)
  viridis <- metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0", palette = "viridis")
  expect_identical(sort(unique(layer_with(viridis, "GeomLine")$colour)), sort(palette_colors("viridis", 2L)))
})

test_that("plot_metric_by_pipeline names what is missing", {
  expect_error(
    metric_by_pipeline_plot(NULL, pooled_results(), "m0"),
    "There is no preprocessing table to plot: the fit was not randomized. Fit with randomize_preprocessing=TRUE.",
    fixed = TRUE
  )
  expect_error(metric_by_pipeline_plot(pipeline_results()[0L, ], pooled_results(), "m0"), "Preprocessing DataFrame is empty.", fixed = TRUE)
  expect_error(
    metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m9"),
    "method_id 'm9' not found in the preprocessing table. Available: ['m0', 'm1'].",
    fixed = TRUE
  )
  expect_error(
    metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0", measure = "pac"),
    "The per-pipeline table carries only the ARI criteria, so measure must be 'stability' or 'generalizability' (or an alias of either); got 'pac'.",
    fixed = TRUE
  )
  expect_error(
    metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0", measure = "average"),
    "got 'average'.",
    fixed = TRUE
  )
})

test_that("plot_n_clusters_over_sweep draws the observed counts and their standard errors", {
  plot <- n_clusters_over_sweep_plot(realized_count_results())
  expect_identical(guide_labels(plot, "colour"), c("LeidenClustering", "LouvainClustering"))
  lines <- layer_with(plot, "GeomLine")
  bars <- layer_with(plot, "GeomErrorbar")
  leiden <- palette_colors("Accent", 2L)[[1L]]
  expect_equal(lines$x[lines$colour == leiden], c(0.5, 1, 2))
  expect_equal(lines$y[lines$colour == leiden], c(3.2, 6.6, 11.0))
  expect_equal(lines$y[lines$colour != leiden], c(2.2, 5.0, 9.4))
  expect_equal((bars$ymax - bars$ymin)[bars$colour == leiden] / 2, c(0.2, 0.4, 0.0))
  expect_equal((bars$ymax - bars$ymin)[bars$colour != leiden] / 2, c(0.1, 0.3, 0.6))
})

test_that("plot_n_clusters_over_sweep marks the selected value and its count", {
  cases <- list(
    list("stability", "max", FALSE, 1, "Selected resolution (Max rule): 1, 7 clusters"),
    list("stability", "1se", FALSE, 2, "Selected resolution (1-SE rule): 2, 11 clusters"),
    list("generalizability", "max", FALSE, 0.5, "Selected resolution (Max rule): 0.5, 2 clusters"),
    list("generalizability", "max", TRUE, 1, "Selected resolution (Max rule): 1, 7 clusters")
  )
  for (case in cases) {
    plot <- n_clusters_over_sweep_plot(realized_count_results(), measure = case[[1L]], rule = case[[2L]], not_two = case[[3L]])
    expect_equal(layer_with(plot, "GeomVline")$xintercept, case[[4L]])
    expect_identical(guide_labels(plot, "linetype"), case[[5L]])
  }
})

test_that("plot_n_clusters_over_sweep labels its axes and ticks whole numbers", {
  plot <- n_clusters_over_sweep_plot(realized_count_results())
  expect_identical(plot$labels$x, "Resolution")
  expect_identical(plot$labels$y, "Mean Observed Number of Clusters")
  expect_identical(n_clusters_over_sweep_plot(realized_count_results(), ylabel = "Clusters")$labels$y, "Clusters")
  # Between 2 and 3 clusters the default breaks are 2, 2.25, 2.5, ...
  narrow <- resolution_results()[1:3, ]
  narrow$n_clusters_observed <- c(2, 2.5, 3)
  narrow$n_clusters_observed_se <- 0
  breaks <- y_breaks(n_clusters_over_sweep_plot(narrow))
  expect_true(length(breaks) >= 2L)
  expect_identical(breaks, round(breaks))
})

test_that("plot_n_clusters_over_sweep rejects a k sweep, a bad measure and an empty table", {
  expect_error(n_clusters_over_sweep_plot(metric_results()), "other than n_clusters", fixed = TRUE)
  expect_error(n_clusters_over_sweep_plot(realized_count_results(), measure = "nonexistent"), "not found", fixed = TRUE)
  expect_error(n_clusters_over_sweep_plot(data.frame()), "Results DataFrame is empty.", fixed = TRUE)
})

test_that("the three line plots draw through one helper", {
  calls <- list()
  local_mocked_bindings(draw_metric_lines = function(results, ...) {
    args <- list(...)
    calls[[length(calls) + 1L]] <<- list(args$group_col, args$legend_title, args$y_col, nrow(results))
    "drawn"
  })
  expect_identical(metric_over_sweep_plot(metric_results()), "drawn")
  expect_identical(metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0"), "drawn")
  expect_identical(n_clusters_over_sweep_plot(realized_count_results()), "drawn")
  expect_identical(calls, list(
    list("method_id", "Estimators", "ari_stability", 3L),
    list("pipeline", "Pipelines", "ari_stability", 6L),
    list("method_id", "Estimators", "n_clusters_observed", 6L)
  ))
})

test_that("consensus_display symmetrizes, fills unsampled pairs and orders by cluster", {
  M <- rbind(c(1, NaN, 0.2), c(NaN, 1, 0.6), c(0.4, 0.8, 1))
  display <- consensus_display(M, c(2, 1, 2))
  expect_equal(display$matrix, rbind(c(1, 0.5, 0.7), c(0.5, 1, 0.3), c(0.7, 0.3, 1)))
  expect_identical(display$labels, c(1, 2, 2))
  # Python's test_nan_handling: never co-sampled pairs show 0.5, the value
  # get_labels() also uses, and the diagonal is 1.
  expect_equal(consensus_display(rbind(c(1, NaN), c(NaN, 1)), c(1, 2))$matrix, rbind(c(1, 0.5), c(0.5, 1)))
  expect_equal(consensus_display(rbind(c(0.3, -0.2), c(-0.2, 1.4)), c(1, 1))$matrix, rbind(c(1, 0), c(0, 1)))
})

test_that("consensus_display checks its input", {
  expect_error(consensus_display(matrix(0, 3, 4), c(1, 1, 1)), "consensus_matrix must be a square 2D array.", fixed = TRUE)
  expect_error(consensus_display(diag(3), c(1, 2)), "consensus_matrix and labels must have matching first dimension.", fixed = TRUE)
  expect_error(consensus_display(diag(3), matrix(1:3, 1)), "labels must be a 1D array.", fixed = TRUE)
})

test_that("the heatmap shows the ordered matrix in the cmap's colors under a band of cluster colors", {
  M <- matrix(0, 6, 6)
  M[1:3, 1:3] <- 1
  M[4:6, 4:6] <- 1
  M[1, 4] <- 0.3
  labels <- c(9, 9, 9, 4, 4, 4)
  plot <- consensus_plot(M, labels)
  rasters <- raster_layers(plot)
  expect_length(rasters, 2L)
  display <- consensus_display(M, labels)
  expect_identical(
    raster_colours(rasters[[1L]]),
    matrix(lut_colours(display$matrix, colour_lut(continuous_colours("viridis"))), 6L, 6L)
  )
  expect_identical(as.vector(raster_colours(rasters[[2L]])), rep(palette_colors("Accent", 2L), each = 3L))
  greens <- consensus_plot(M, labels, cmap = "Greens")
  expect_identical(
    raster_colours(raster_layers(greens)[[1L]]),
    matrix(lut_colours(display$matrix, colour_lut(GREENS)), 6L, 6L)
  )
})

test_that("white lines separate the clusters", {
  # Row 1 is drawn at the top, at y = n, so the line between rows 1 and 2
  # is at y = n - 0.5.
  plot <- consensus_plot(diag(4), c(1, 2, 2, 2))
  expect_equal(layer_with(plot, "GeomVline")$xintercept, 1.5)
  expect_equal(layer_with(plot, "GeomHline")$yintercept, 3.5)
  expect_false(has_layer(consensus_plot(diag(3), c(1, 1, 1)), "GeomVline"))
})

test_that("the color bar, labels and frame of the heatmap", {
  plot <- consensus_plot(diag(4), c(1, 1, 2, 2), title = "T")
  expect_identical(plot$scales$get_scales("fill")$name, "Consensus")
  expect_false(is.null(ggplot2::get_guide_data(plot, "fill")))
  expect_identical(plot$labels$x, "Samples (ordered by cluster)")
  expect_identical(plot$labels$title, "T")
  ranges <- ggplot2::ggplot_build(plot)$layout$panel_params[[1L]]
  expect_equal(ranges$x.range, c(0.5, 4.5))
  expect_equal(ranges$y.range, c(0.5, 4.5 + 0.04 * 4))
  expect_identical(consensus_plot(diag(4), c(1, 1, 2, 2), colorbar_label = "Share")$scales$get_scales("fill")$name, "Share")
  expect_null(ggplot2::get_guide_data(consensus_plot(diag(4), c(1, 1, 2, 2), colorbar = FALSE), "fill"))
})

test_that("cluster_score_groups matches Python's grouping", {
  f <- read_fixture("plotting")$score_groups
  scores <- fixture_vector(f$scores)
  # R labels count from 1 and are shown as they are; Python's count from 0
  # and are shown plus one.
  labels <- fixture_vector(f$labels) + 1
  for (case in f$cases) {
    order <- if (is.null(case$order)) NULL else fixture_vector(case$order) + 1
    prepared <- cluster_score_groups(scores, labels, order)
    expect_identical(prepared$order, fixture_vector(case$kept_order))
    expect_identical(length(prepared$groups), length(case$groups))
    for (i in seq_along(case$groups)) {
      expect_equal(prepared$groups[[i]], fixture_vector(case$groups[[i]]))
    }
  }
})

test_that("cluster_score_groups drops non-finite scores and checks its input", {
  prepared <- cluster_score_groups(c(0.9, NaN, 0.7, 0.6), c(1, 1, 2, 2))
  expect_identical(lengths(prepared$groups), c(1L, 2L))
  expect_identical(cluster_score_groups(c(0.9, 0.8, 0.7, 0.6), c(1, 1, 2, 2), order = c(2, 1))$order, c(2, 1))
  expect_error(cluster_score_groups(c(NaN, NaN), c(1, 2)), "No finite scores available for plotting.", fixed = TRUE)
  expect_error(cluster_score_groups(0.5, c(1, 2)), "scores and labels must have matching length.", fixed = TRUE)
  expect_error(cluster_score_groups(matrix(0.5), 1), "scores must be a 1D array.", fixed = TRUE)
  expect_error(cluster_score_groups(0.5, matrix(1)), "labels must be a 1D array.", fixed = TRUE)
  expect_error(cluster_score_groups(c(0.5, 0.6), c(1, 2), order = 7), "No cluster values found for the provided order.", fixed = TRUE)
})

test_that("score_ylim fits the scores with a margin or keeps ylim", {
  # A range of 0.1 gives a margin of 0.005, below the 0.02 floor.
  expect_equal(score_ylim(list(c(0.5, 0.6), 0.55), c(-0.02, 1.02), TRUE), c(0.48, 0.62))
  expect_equal(score_ylim(list(c(0, 1)), NULL, TRUE), c(-0.05, 1.05))
  expect_identical(score_ylim(list(0.5), c(0, 1), FALSE), c(0, 1))
  expect_null(score_ylim(list(0.5), NULL, FALSE))
})

test_that("the box plot draws one box per cluster in the requested order", {
  scores <- c(0.9, 0.85, 0.7, 0.65, 0.5, 0.45)
  labels <- c(1, 1, 1, 2, 2, 2)
  plot <- cluster_boxplot_plot(scores, labels, order = c(2, 1), annotation = "note", title = "T")
  boxes <- layer_with(plot, "GeomBoxplot")
  expect_identical(nrow(boxes), 2L)
  expect_equal(boxes$middle, c(0.5, 0.85))
  expect_identical(boxes$fill, palette_colors("Accent", 2L))
  expect_identical(plot$scales$get_scales("x")$labels, c("2", "1"))
  expect_identical(plot$labels$caption, "note")
  expect_identical(plot$labels$title, "T")
  expect_identical(plot$labels$x, "Cluster")
  expect_identical(plot$labels$y, "Uncertainty")
  expect_equal(ggplot2::ggplot_build(plot)$layout$panel_params[[1L]]$y.range, c(0.45 - 0.0225, 0.9 + 0.0225))
  expect_null(cluster_boxplot_plot(scores, labels)$labels$caption)
})

test_that("showfliers and rotation reach the box plot", {
  scores <- c(0.5, 0.52, 0.51, 0.53, 0.5, 0.95, 0.1, 0.2)
  labels <- c(1, 1, 1, 1, 1, 1, 2, 2)
  hidden <- cluster_boxplot_plot(scores, labels)
  shown <- cluster_boxplot_plot(scores, labels, showfliers = TRUE)
  expect_true(is.na(hidden$layers[[layer_index(hidden, "GeomBoxplot")]]$geom_params$outlier_gp$shape))
  expect_identical(shown$layers[[layer_index(shown, "GeomBoxplot")]]$geom_params$outlier_gp$shape, 19)
  rotated <- cluster_boxplot_plot(scores, labels, rotation = 45)
  expect_identical(rotated$theme$axis.text.x$angle, 45)
})

test_that("the violin plot draws bodies, inner marks and seeded points", {
  scores <- c(0.9, 0.85, 0.7, 0.65, 0.5, 0.45)
  labels <- c(1, 1, 1, 2, 2, 2)
  plot <- cluster_violin_plot(scores, labels)
  bodies <- layer_with(plot, "GeomViolin")
  expect_identical(sort(unique(bodies$fill)), sort(palette_colors("Accent", 2L)))
  marks <- layer_with(plot, "GeomSegment")
  # "box": a median line, a quartile line and two quartile caps per cluster.
  expect_identical(nrow(marks), 8L)
  expect_equal(sort(marks$y[marks$x == marks$xend]), sort(c(0.775, 0.475)))
  points <- layer_with(plot, "GeomPoint")
  expect_equal(points$y, scores)
  expect_true(all(abs(points$x - rep(1:2, each = 3)) <= 0.11))
  expect_identical(points$x, layer_with(cluster_violin_plot(scores, labels), "GeomPoint")$x)
  quartile <- layer_with(cluster_violin_plot(scores, labels, inner = "quartile"), "GeomSegment")
  expect_identical(nrow(quartile), 6L)
  expect_false(has_layer(cluster_violin_plot(scores, labels, inner = "none"), "GeomSegment"))
  expect_false(has_layer(cluster_violin_plot(scores, labels, stripplot = FALSE), "GeomPoint"))
  still <- layer_with(cluster_violin_plot(scores, labels, jitter = FALSE), "GeomPoint")
  expect_equal(still$x, rep(c(1, 2), each = 3))
})

test_that("the violin plot scales the bodies and limits the axis", {
  scores <- c(0.95, 0.96, 0.97, 0.98, 0.99, 0.60, 0.61, 0.62)
  labels <- c(1, 1, 1, 1, 1, 2, 2, 2)
  plot <- cluster_violin_plot(scores, labels, density_norm = "count", ylim = c(0, 0.9), fit_ylim = FALSE)
  violin <- plot$layers[[layer_index(plot, "GeomViolin")]]
  expect_identical(violin$stat_params$scale, "count")
  expect_equal(ggplot2::ggplot_build(plot)$layout$panel_params[[1L]]$y.range, c(0, 0.9))
  expect_warning(
    unknown <- cluster_violin_plot(scores, labels, density_norm = "nope"),
    "Unknown density_norm='nope'; using 'width'.",
    fixed = TRUE
  )
  expect_identical(unknown$layers[[layer_index(unknown, "GeomViolin")]]$stat_params$scale, "width")
  expect_error(cluster_violin_plot(scores, labels, inner = "nope"), "inner must be one of: 'box', 'quartile', 'none'.", fixed = TRUE)
})

test_that("a cluster with one score gets points but no violin body", {
  plot <- cluster_violin_plot(c(0.9, 0.8, 0.7, 0.4), c(1, 1, 1, 2))
  bodies <- layer_with(plot, "GeomViolin")
  expect_identical(unique(bodies$fill), palette_colors("Accent", 2L)[[1L]])
  expect_identical(nrow(layer_with(plot, "GeomPoint")), 4L)
})

scatter_case <- function(n = 30L, p = 4L, k = 3L) {
  withr::with_seed(0L, list(
    X = matrix(stats::rnorm(n * p), n, p),
    labels = rep(seq_len(k), each = n %/% k),
    scores = stats::runif(n)
  ))
}

test_that("scatter sizes, opacities and legend match Python's", {
  f <- read_fixture("plotting")$scatter
  X <- fixture_matrix(f$X)
  labels <- fixture_vector(f$labels) + 1
  scores <- fixture_vector(f$scores)
  encoding <- scatter_encoding(scores, c(15, 60), c(0.45, 0.9))
  expect_equal(encoding$size, fixture_vector(f$sizes))
  expect_equal(encoding$alpha, fixture_vector(f$alphas))
  flat <- scatter_encoding(rep(0.8, 12L), c(15, 60), c(0.45, 0.9))
  expect_equal(flat$size, fixture_vector(f$flat_sizes))
  expect_equal(flat$alpha, fixture_vector(f$flat_alphas))
  plot <- cluster_scatter_plot(X, labels, scores, sort_order = FALSE)
  expect_identical(guide_labels(plot, "fill"), unlist(f$legend))
  points <- layer_with(plot, "GeomPoint")
  expect_equal(points$size, point_size(fixture_vector(f$sizes)))
  expect_equal(points$alpha, fixture_vector(f$alphas))
  expect_equal(cbind(points$x, points$y), unname(X))
  expect_identical(points$fill, palette_colors("Accent", 3L)[labels])
})

test_that("scatter coordinates come from the embedding, the data or its principal components", {
  case <- scatter_case(n = 20L, p = 10L, k = 2L)
  expect_equal(scatter_coordinates(case$X[, 1:2]), unname(case$X[, 1:2]))
  expect_equal(scatter_coordinates(case$X[, 1L, drop = FALSE]), cbind(case$X[, 1L], 0))
  reduced <- scatter_coordinates(case$X)
  expect_equal(reduced, PCA(case$X, n_components = 2L, random_state = 0L))
  # Two of ten components go through irlba, hence test-transforms.R's tolerance.
  expect_equal(abs(reduced), abs(unname(stats::prcomp(case$X)$x[, 1:2])), tolerance = 1e-4)
  embedding <- matrix(seq_len(60), 20, 3)
  expect_equal(scatter_coordinates(case$X, embedding), embedding[, 1:2] + 0)
  expect_error(scatter_coordinates(case$X, embedding[, 1L, drop = FALSE]), "embedding must be a 2D array with at least 2 columns.", fixed = TRUE)
  expect_error(scatter_coordinates(case$X, embedding[1:5, ]), "embedding must have the same number of rows as X.", fixed = TRUE)
})

test_that("the scatter plot orders, bounds and drops points by score", {
  case <- scatter_case()
  sorted <- layer_with(cluster_scatter_plot(case$X, case$labels, case$scores, alpha_range = c(0.3, 0.9)), "GeomPoint")
  expect_equal(min(sorted$alpha), 0.3)
  expect_equal(max(sorted$alpha), 0.9)
  expect_false(is.unsorted(sorted$alpha))
  unsorted <- layer_with(cluster_scatter_plot(case$X, case$labels, case$scores, sort_order = FALSE), "GeomPoint")
  expect_true(is.unsorted(unsorted$alpha))
  scores <- case$scores
  scores[4] <- NaN
  expect_identical(nrow(layer_with(cluster_scatter_plot(case$X, case$labels, scores), "GeomPoint")), 29L)
  # A cluster without a finite score has no points but keeps its legend entry.
  scores <- case$scores
  scores[case$labels == 2] <- NaN
  empty <- cluster_scatter_plot(case$X, case$labels, scores)
  expect_identical(nrow(layer_with(empty, "GeomPoint")), 20L)
  expect_identical(
    guide_labels(empty, "fill"),
    c(sprintf("1 (Mean = %.2f)", mean(scores[case$labels == 1])), "2 (Mean = nan)", sprintf("3 (Mean = %.2f)", mean(scores[case$labels == 3])))
  )
})

test_that("the scatter annotation goes into the legend title or the caption", {
  case <- scatter_case()
  in_legend <- cluster_scatter_plot(case$X, case$labels, case$scores, annotation = "note", scores_name = "Gini Stability")
  expect_identical(in_legend$scales$get_scales("fill")$name, "note\nCluster, Gini Stability")
  expect_null(in_legend$labels$caption)
  boxed <- cluster_scatter_plot(case$X, case$labels, case$scores, annotation = "note", annotation_style = "box")
  expect_identical(boxed$labels$caption, "note")
  expect_identical(boxed$scales$get_scales("fill")$name, "Cluster, Score")
  expect_error(
    cluster_scatter_plot(case$X, case$labels, case$scores, annotation_style = "nope"),
    "annotation_style must be 'legend' or 'box'.",
    fixed = TRUE
  )
})

test_that("the scatter legend, ticks and frame can be turned off and on", {
  case <- scatter_case()
  plot <- cluster_scatter_plot(case$X, case$labels, case$scores)
  expect_true(inherits(plot$theme$axis.ticks, "element_blank"))
  expect_true(inherits(plot$theme$panel.border, "element_blank"))
  expect_identical(plot$labels$x, "Component 1")
  framed <- cluster_scatter_plot(case$X, case$labels, case$scores, show_ticks = TRUE, frameon = TRUE)
  expect_false(inherits(framed$theme$axis.ticks, "element_blank"))
  expect_false(inherits(framed$theme$panel.border, "element_blank"))
  expect_null(ggplot2::get_guide_data(cluster_scatter_plot(case$X, case$labels, case$scores, legend = FALSE), "fill"))
})

test_that("the scatter plots check their input as Python does", {
  case <- scatter_case(n = 20L, p = 2L, k = 2L)
  expect_error(cluster_scatter_plot(case$X, case$labels, case$scores[1:10]), "X, labels, and scores must have matching n_samples.", fixed = TRUE)
  expect_error(cluster_scatter_plot(case$X, case$labels, rep(NaN, 20L)), "No finite scores available for plotting.", fixed = TRUE)
  expect_error(diagnostic_scatter_plot(case$X, case$labels, case$scores[-1L]), "matching n_samples", fixed = TRUE)
  expect_error(diagnostic_scatter_plot(case$X, case$labels, rep(NaN, 20L)), "No finite scores", fixed = TRUE)
})

test_that("diagnostic drawing order and opacities match Python's", {
  f <- read_fixture("plotting")
  X <- fixture_matrix(f$scatter$X)
  labels <- fixture_vector(f$scatter$labels) + 1
  scores <- fixture_vector(f$diagnostic$scores)
  encoding <- diagnostic_encoding(scores, TRUE, c(0.3, 1))
  for (name in c("unsorted", "sorted")) {
    rows <- diagnostic_draw_order(labels, scores, encoding$norm, sort_order = name == "sorted")
    expect_identical(rows, as.integer(fixture_vector(f$diagnostic[[paste0("order_", name)]])) + 1L, info = name)
    expect_equal(encoding$alpha[rows], fixture_vector(f$diagnostic[[paste0("alpha_", name)]]), info = name)
  }
  points <- layer_with(diagnostic_scatter_plot(X, labels, scores), "GeomPoint")
  rows <- diagnostic_draw_order(labels, scores, encoding$norm, sort_order = TRUE)
  expect_equal(cbind(points$x, points$y), unname(X[rows, ]))
  expect_equal(points$alpha, encoding$alpha[rows])
})

test_that("diagnostic fills run from dark for low scores, with NaN in faint gray", {
  case <- scatter_case()
  scores <- case$scores
  scores[5] <- NaN
  points <- layer_with(diagnostic_scatter_plot(case$X, case$labels, scores, sort_order = FALSE), "GeomPoint")
  original <- diagnostic_draw_order(case$labels, scores, rep(0.5, 30L), sort_order = FALSE)
  at <- function(i) which(original == i)
  expect_identical(points$fill[at(which.min(scores))], "#00441B")
  expect_identical(points$fill[at(which.max(scores))], "#F7FCF5")
  expect_identical(points$fill[at(5L)], "#808080")
  expect_equal(points$alpha[at(5L)], 0.2)
  bounded <- layer_with(diagnostic_scatter_plot(case$X, case$labels, case$scores, alpha_range = c(0.4, 0.9)), "GeomPoint")
  expect_equal(range(bounded$alpha), c(0.4, 0.9))
  plain <- layer_with(diagnostic_scatter_plot(case$X, case$labels, case$scores, alpha_encoding = FALSE), "GeomPoint")
  expect_true(all(plain$alpha == 1))
})

test_that("each cluster gets a marker shape, in label order", {
  case <- scatter_case()
  plot <- diagnostic_scatter_plot(case$X, case$labels, case$scores, markers = c(21, 22, 24))
  expect_identical(ggplot2::get_guide_data(plot, "shape")$shape, c(21, 22, 24))
  points <- layer_with(plot, "GeomPoint")
  expect_identical(sort(unique(points$shape)), c(21, 22, 24))
  expect_warning(
    cycled <- diagnostic_scatter_plot(case$X, case$labels, case$scores, markers = c(21, 22)),
    "Number of clusters (3) exceeds available markers (2). Markers will cycle.",
    fixed = TRUE
  )
  expect_identical(ggplot2::get_guide_data(cycled, "shape")$shape, c(21, 22, 21))
  expect_error(
    diagnostic_scatter_plot(case$X, case$labels, case$scores, markers = c(1, 2)),
    "markers must be ggplot2 shapes 21 to 25, the shapes with a fill.",
    fixed = TRUE
  )
})

test_that("the diagnostic color bar and legend titles", {
  case <- scatter_case()
  named <- diagnostic_scatter_plot(case$X, case$labels, case$scores, scores_name = "Foo")
  expect_identical(named$scales$get_scales("fill")$name, "Foo")
  expect_false(is.null(ggplot2::get_guide_data(named, "fill")))
  relabeled <- diagnostic_scatter_plot(case$X, case$labels, case$scores, scores_name = "Foo", colorbar_label = "Bar")
  expect_identical(relabeled$scales$get_scales("fill")$name, "Bar")
  expect_null(ggplot2::get_guide_data(diagnostic_scatter_plot(case$X, case$labels, case$scores, colorbar = FALSE), "fill"))
  in_legend <- diagnostic_scatter_plot(case$X, case$labels, case$scores, annotation = "note")
  expect_identical(in_legend$scales$get_scales("shape")$name, "note\nCluster")
  boxed <- diagnostic_scatter_plot(case$X, case$labels, case$scores, annotation = "note", annotation_style = "box")
  expect_identical(boxed$scales$get_scales("shape")$name, "Cluster")
  expect_identical(boxed$labels$caption, "note")
  expect_null(ggplot2::get_guide_data(diagnostic_scatter_plot(case$X, case$labels, case$scores, legend = FALSE), "shape"))
})
