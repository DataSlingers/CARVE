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
