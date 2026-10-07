#' @include AllGenerics.R
NULL

# Drawing for the plots of a CARVE fit. Mirrors _plotting.py: the functions
# here draw from a results table or from per-sample vectors, and plots.R
# picks those inputs from a fit or from a single-cell object. Every plot is
# a ggplot object.

# The colors of the Python package's defaults: matplotlib's Accent and
# Greens colormaps, which are ColorBrewer's. viridis comes from scales.
ACCENT <- c("#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17", "#666666")
GREENS <- c(
  "#F7FCF5", "#E5F5E0", "#C7E9C0", "#A1D99B", "#74C476",
  "#41AB5D", "#238B45", "#006D2C", "#00441B"
)
VIRIDIS_OPTIONS <- c("magma", "inferno", "plasma", "viridis", "cividis", "rocket", "mako", "turbo")
# The ggplot2 shapes with a fill, in the order of Python's first five
# markers: circle, square, up triangle, diamond, down triangle.
DIAGNOSTIC_MARKERS <- c(21L, 22L, 24L, 23L, 25L)

is_colour <- function(x) {
  vapply(x, function(value) {
    tryCatch({
      grDevices::col2rgb(value)
      TRUE
    }, error = function(e) FALSE)
  }, logical(1), USE.NAMES = FALSE)
}

# The colors a palette or cmap argument names: "Accent", a name from
# grDevices::palette.pals(), a viridis option, "Greens" or "Greens_r", or a
# vector of colors.
colormap_values <- function(palette, argument = "palette") {
  if (is.character(palette) && length(palette) == 1L && !is.na(palette)) {
    if (identical(palette, "Accent")) {
      return(ACCENT)
    }
    if (palette %in% VIRIDIS_OPTIONS) {
      return(scales::viridis_pal(option = palette)(256L))
    }
    if (identical(palette, "Greens")) {
      return(GREENS)
    }
    if (identical(palette, "Greens_r")) {
      return(rev(GREENS))
    }
    if (palette %in% grDevices::palette.pals()) {
      return(unname(grDevices::palette.colors(palette = palette)))
    }
  }
  if (is.character(palette) && length(palette) >= 1L && !anyNA(palette) && all(is_colour(palette))) {
    return(unname(palette))
  }
  stop(sprintf(
    "%s must be a palette name such as 'Accent' or 'viridis', or a vector of colors.",
    argument
  ), call. = FALSE)
}

# n colors sampled as matplotlib samples a listed colormap at
# numpy.linspace(0, 1, n): truncation to an index, with 1 mapped to the last
# color. An 8-color palette gives its first and last colors for n = 2 and
# repeats colors for n > 8.
palette_colors <- function(palette, n) {
  values <- colormap_values(palette)
  m <- length(values)
  index <- pmin(floor(seq(0, 1, length.out = n) * m), m - 1) + 1
  values[index]
}

continuous_colours <- function(cmap) {
  values <- colormap_values(cmap, argument = "cmap")
  if (length(values) < 2L) {
    stop("cmap must name a colormap or give at least two colors.", call. = FALSE)
  }
  values
}

# matplotlib maps values in [0, 1] through a 256-entry table; the heatmap
# does the same, so its colors match the color bar ggplot2 draws from the
# same colors.
colour_lut <- function(colours, n = 256L) {
  scales::gradient_n_pal(colours)(seq(0, 1, length.out = n))
}

lut_colours <- function(values, lut) {
  n <- length(lut)
  lut[pmin(floor(values * n), n - 1) + 1]
}

# Python's f"{x:g}", which is C's %g.
format_g <- function(x) {
  sprintf("%.6g", x)
}

measure_ylabel <- function(column) {
  gsub("Ari", "ARI", title_case(gsub("_", " ", column, fixed = TRUE)), fixed = TRUE)
}

rule_title <- function(rule) {
  if (identical(rule, "1se")) "1-SE" else title_case(rule)
}

selected_value_label <- function(row, param, rule) {
  sprintf(
    "Selected %s (%s rule): %s",
    if (param == "n_clusters") "k" else param,
    rule_title(rule),
    format_g(as.numeric(row$sweep_value[[1L]]))
  )
}

build_estimator_label <- function(row, tight_layout = FALSE) {
  label <- as.character(row$method_label[[1L]])
  if (tight_layout) gsub(", ", "\n", label, fixed = TRUE) else label
}

# Two lines naming the selected configuration and how it was selected. Off
# the n_clusters axis the detail gives the swept value and the rounded mean
# cluster count. Python writes that count with an approximately-equal sign,
# which R's pdf() device cannot draw, so R writes "k ~ 7".
annotation_text <- function(measure, rule, results, row, selected_k, pinned = FALSE,
                            tight_layout = FALSE) {
  k_text <- if (is.null(selected_k)) "None" else as.character(selected_k)
  param <- sweep_param_name(results)
  detail <- if (param == "n_clusters") {
    paste0("k = ", k_text)
  } else {
    sprintf("%s = %s, k ~ %s", param, format_g(as.numeric(row$sweep_value[[1L]])), k_text)
  }
  if (isTRUE(pinned)) {
    detail <- paste0(detail, ", fixed")
  }
  text <- sprintf(
    "%s (%s)\n%s, %s rule",
    build_estimator_label(row, tight_layout = tight_layout),
    detail,
    title_case(gsub("_", " ", measure, fixed = TRUE)),
    rule_title(rule)
  )
  if (tight_layout) paste0(text, "\n") else text
}

# Plots check the measure before drawing: an unknown one would otherwise
# make the selection fail inside the tryCatch() around the dashed line, and
# the line would be left off without an error.
plot_measure_column <- function(results, measure) {
  if (!is.character(measure) || length(measure) != 1L || !measure %in% names(MEASURE_MAP)) {
    stop(sprintf(
      "Measure %s not found. Valid options: %s",
      format_repr(measure),
      python_list(names(MEASURE_MAP))
    ), call. = FALSE)
  }
  column <- MEASURE_MAP[[measure]]
  if (!column %in% names(results)) {
    stop(sprintf("Metric column '%s' not found in the results table.", column), call. = FALSE)
  }
  column
}

# Breaks on whole numbers, for counts.
integer_breaks <- function(limits) {
  lo <- ceiling(limits[[1L]])
  hi <- floor(limits[[2L]])
  if (!is.finite(lo) || !is.finite(hi) || hi < lo) {
    return(numeric())
  }
  unique(round(pretty(c(lo, hi))))
}

# matplotlib sizes markers by area in square points, so a marker of area s
# is sqrt(s) points across; ggplot2 draws a circle 0.75 * size * .pt points
# across.
point_size <- function(s) {
  sqrt(s) / (0.75 * ggplot2::.pt)
}

# matplotlib line widths are in points, 96 / 72 grid line-width units each.
# ggplot2 draws a point's outline with stroke * .stroke / 2 units and a line
# with linewidth * .pt units.
edge_stroke <- function(lw) {
  lw * 96 / 72 * 2 / ggplot2::.stroke
}

line_width <- function(lw) {
  lw * 96 / 72 / ggplot2::.pt
}

carve_theme <- function() {
  ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
}

legend_position <- function(legend_loc) {
  if (is.numeric(legend_loc) && length(legend_loc) == 2L && !anyNA(legend_loc)) {
    return(ggplot2::theme(legend.position = "inside", legend.position.inside = legend_loc))
  }
  if (!is.character(legend_loc) || length(legend_loc) != 1L ||
      !legend_loc %in% c("right", "left", "top", "bottom")) {
    stop(
      "legend_loc must be 'right', 'left', 'top', 'bottom', or two numbers giving a position inside the panel.",
      call. = FALSE
    )
  }
  ggplot2::theme(legend.position = legend_loc)
}

# The drawing behind the three line plots over the sweep axis, so they
# cannot drift apart. results is non-empty and has sweep_value, group_col
# and y_col; error bars come from <y_col>_se when the table has it.
# select_row() returns the row whose sweep_value the dashed line marks, and
# selection_label() builds its legend text from that row. If either fails,
# for example because not_two leaves no rows, the line is left off, as in
# Python.
draw_metric_lines <- function(results, y_col, group_col, label_of, legend_title, select_row,
                              selection_label, integer_y = FALSE, title = NULL, xlabel = NULL,
                              ylabel, legend = TRUE, legend_loc = "right", palette = "Accent") {
  keys <- as.character(results[[group_col]])
  # pandas' groupby sorts the group keys in byte order, which radix sorting
  # reproduces whatever the locale.
  groups <- sort(unique(keys), method = "radix")
  colours <- stats::setNames(palette_colors(palette, length(groups)), groups)
  labels <- stats::setNames(
    vapply(match(groups, keys), function(i) label_of(results[i, , drop = FALSE]), character(1)),
    groups
  )
  se_col <- paste0(y_col, "_se")
  data <- data.frame(
    group = factor(keys, levels = groups),
    x = as.numeric(results$sweep_value),
    y = as.numeric(results[[y_col]]),
    se = if (se_col %in% names(results)) as.numeric(results[[se_col]]) else NA_real_
  )
  data <- data[order(data$group, data$x), , drop = FALSE]
  x_values <- sort(unique(data$x))
  selected <- tryCatch(
    {
      row <- select_row()
      list(x = as.numeric(row$sweep_value[[1L]]), label = selection_label(row))
    },
    error = function(e) NULL
  )

  plot <- ggplot2::ggplot(
    data,
    ggplot2::aes(x = .data$x, y = .data$y, colour = .data$group, group = .data$group)
  )
  if (!is.null(selected)) {
    plot <- plot +
      ggplot2::geom_vline(
        data = data.frame(x = selected$x, label = selected$label),
        ggplot2::aes(xintercept = .data$x, linetype = .data$label),
        colour = "gray50", linewidth = line_width(2), alpha = 0.6
      ) +
      ggplot2::scale_linetype_manual(values = "dashed", name = NULL)
  }
  if (any(!is.na(data$se))) {
    span <- if (length(x_values) > 1L) diff(range(x_values)) else 1
    plot <- plot + ggplot2::geom_errorbar(
      ggplot2::aes(ymin = .data$y - .data$se, ymax = .data$y + .data$se),
      width = 0.02 * span, linewidth = line_width(1.5), alpha = 0.8, na.rm = TRUE
    )
  }
  plot <- plot +
    ggplot2::geom_line(linewidth = line_width(2), alpha = 0.8, na.rm = TRUE) +
    ggplot2::geom_point(size = point_size(36), alpha = 0.8, na.rm = TRUE) +
    ggplot2::scale_colour_manual(values = colours, breaks = groups, labels = labels, name = legend_title)
  if (length(x_values) <= 20L) {
    plot <- plot + ggplot2::scale_x_continuous(
      breaks = x_values,
      labels = vapply(x_values, format_g, character(1))
    )
  }
  if (integer_y) {
    plot <- plot + ggplot2::scale_y_continuous(breaks = integer_breaks)
  }
  plot <- plot +
    ggplot2::labs(
      x = if (is.null(xlabel)) sweep_axis_label(results) else xlabel,
      y = ylabel,
      title = title
    ) +
    carve_theme() +
    legend_position(legend_loc)
  if (!isTRUE(legend)) {
    plot <- plot + ggplot2::guides(colour = "none", linetype = "none")
  }
  plot
}

metric_over_sweep_plot <- function(results, measure = "stability", rule = "1se", not_two = FALSE,
                                   title = NULL, xlabel = NULL, ylabel = NULL, legend = TRUE,
                                   legend_loc = "right", palette = "Accent") {
  if (nrow(results) == 0L) {
    stop("Results DataFrame is empty.", call. = FALSE)
  }
  column <- plot_measure_column(results, measure)
  param <- sweep_param_name(results)
  draw_metric_lines(
    results,
    y_col = column,
    group_col = "method_id",
    label_of = build_estimator_label,
    legend_title = "Estimators",
    select_row = function() select_best_row_by_rule(results, measure, rule, not_two = not_two),
    selection_label = function(row) selected_value_label(row, param, rule),
    title = title,
    xlabel = xlabel,
    ylabel = if (is.null(ylabel)) measure_ylabel(column) else ylabel,
    legend = legend,
    legend_loc = legend_loc,
    palette = palette
  )
}

# The estimator_results() row whose sweep value marks method_id's
# selection: CARVE's selection over the pooled table when it lands on
# method_id, otherwise the rule's choice among method_id's rows. Restricting
# to method_id first would not do for the selected configuration: under
# "1se" and "quantile" the tolerance comes from the best row of the whole
# table, which can belong to another configuration.
selected_row_for <- function(estimator_results, method_id, measure, rule, not_two) {
  row <- select_best_row_by_rule(estimator_results, measure, rule, not_two = not_two)
  if (identical(as.character(row$method_id[[1L]]), as.character(method_id))) {
    return(row)
  }
  own <- estimator_results[as.character(estimator_results$method_id) == as.character(method_id), , drop = FALSE]
  select_best_row_by_rule(own, measure, rule, not_two = not_two)
}

metric_by_pipeline_plot <- function(preprocessing, estimator_results, method_id,
                                    measure = "stability", rule = "1se", not_two = FALSE,
                                    title = NULL, xlabel = NULL, ylabel = NULL, legend = TRUE,
                                    legend_loc = "right", palette = "Accent") {
  if (is.null(preprocessing)) {
    stop(
      "There is no preprocessing table to plot: the fit was not randomized. Fit with randomize_preprocessing=TRUE.",
      call. = FALSE
    )
  }
  if (nrow(preprocessing) == 0L) {
    stop("Preprocessing DataFrame is empty.", call. = FALSE)
  }
  ids <- as.character(preprocessing$method_id)
  rows <- preprocessing[ids == as.character(method_id), , drop = FALSE]
  if (nrow(rows) == 0L) {
    stop(sprintf(
      "method_id %s not found in the preprocessing table. Available: %s.",
      format_repr(as.character(method_id)),
      python_list(sort(unique(ids), method = "radix"))
    ), call. = FALSE)
  }
  carried <- is.character(measure) && length(measure) == 1L && measure %in% names(MEASURE_MAP) &&
    MEASURE_MAP[[measure]] %in% names(rows)
  if (!carried) {
    stop(sprintf(
      "The per-pipeline table carries only the ARI criteria, so measure must be 'stability' or 'generalizability' (or an alias of either); got %s.",
      format_repr(measure)
    ), call. = FALSE)
  }
  column <- MEASURE_MAP[[measure]]
  param <- sweep_param_name(rows)
  draw_metric_lines(
    rows,
    y_col = column,
    group_col = "pipeline",
    label_of = function(row) as.character(row$pipeline[[1L]]),
    legend_title = "Pipelines",
    select_row = function() selected_row_for(estimator_results, method_id, measure, rule, not_two),
    selection_label = function(row) selected_value_label(row, param, rule),
    title = title,
    xlabel = xlabel,
    ylabel = if (is.null(ylabel)) measure_ylabel(column) else ylabel,
    legend = legend,
    legend_loc = legend_loc,
    palette = palette
  )
}

n_clusters_over_sweep_plot <- function(results, measure = "stability", rule = "1se",
                                       not_two = FALSE, title = NULL, xlabel = NULL, ylabel = NULL,
                                       legend = TRUE, legend_loc = "right", palette = "Accent") {
  if (nrow(results) == 0L) {
    stop("Results DataFrame is empty.", call. = FALSE)
  }
  param <- sweep_param_name(results)
  if (param == "n_clusters") {
    stop(
      "plot_n_clusters_over_sweep needs a sweep over a parameter other than n_clusters. This table sweeps n_clusters, where the realized count is the swept value itself; use plot_metric_over_n_clusters.",
      call. = FALSE
    )
  }
  plot_measure_column(results, measure)
  draw_metric_lines(
    results,
    y_col = "n_clusters_observed",
    group_col = "method_id",
    label_of = build_estimator_label,
    legend_title = "Estimators",
    select_row = function() select_best_row_by_rule(results, measure, rule, not_two = not_two),
    selection_label = function(row) {
      sprintf("%s, %d clusters", selected_value_label(row, param, rule), observed_k(row))
    },
    integer_y = TRUE,
    title = title,
    xlabel = xlabel,
    ylabel = if (is.null(ylabel)) "Mean Observed Number of Clusters" else ylabel,
    legend = legend,
    legend_loc = legend_loc,
    palette = palette
  )
}

# The matrix as drawn: symmetric, pairs never drawn together at 0.5 (the
# value get_labels() uses), clipped to [0, 1], diagonal 1, and ordered by
# cluster so that each cluster is a block.
consensus_display <- function(consensus_matrix, labels) {
  M <- consensus_matrix
  if (!is.matrix(M) || nrow(M) != ncol(M)) {
    stop("consensus_matrix must be a square 2D array.", call. = FALSE)
  }
  if (!is.null(dim(labels)) && length(dim(labels)) > 1L) {
    stop("labels must be a 1D array.", call. = FALSE)
  }
  labels <- as.vector(labels)
  if (nrow(M) != length(labels)) {
    stop("consensus_matrix and labels must have matching first dimension.", call. = FALSE)
  }
  storage.mode(M) <- "double"
  M <- 0.5 * (M + t(M))
  M[is.na(M)] <- 0.5
  M <- pmin(pmax(M, 0), 1)
  diag(M) <- 1
  ord <- order(labels)
  list(matrix = unname(M[ord, ord, drop = FALSE]), labels = labels[ord])
}

# The matrix and the band of cluster colors are rasters in one panel, the
# band above the matrix. A raster is far cheaper than one tile per pair: at
# 5,000 anchors, about a second and 1 GB against 20 seconds and 5 GB. The
# invisible tile layer carries the fill scale, so the color bar shows.
consensus_plot <- function(consensus_matrix, labels, cmap = "viridis", palette = "Accent",
                           colorbar = TRUE, colorbar_label = "Consensus", title = NULL) {
  display <- consensus_display(consensus_matrix, labels)
  n <- length(display$labels)
  colours <- continuous_colours(cmap)
  cells <- matrix(lut_colours(display$matrix, colour_lut(colours)), n, n)
  clusters <- sort(unique(display$labels))
  band <- matrix(palette_colors(palette, length(clusters))[match(display$labels, clusters)], nrow = 1L)
  band_height <- 0.04 * n
  # Row 1 of the matrix is drawn at the top, at y = n, so the line between
  # rows b and b + 1 is at y = n + 0.5 - b.
  boundaries <- which(diff(display$labels) != 0) + 0.5
  key <- data.frame(x = 1, y = 1, value = c(0, 1))

  plot <- ggplot2::ggplot(key, ggplot2::aes(x = .data$x, y = .data$y, fill = .data$value)) +
    ggplot2::geom_tile(width = 0, height = 0, alpha = 0) +
    ggplot2::annotation_raster(
      cells, xmin = 0.5, xmax = n + 0.5, ymin = 0.5, ymax = n + 0.5, interpolate = FALSE
    ) +
    ggplot2::annotation_raster(
      band, xmin = 0.5, xmax = n + 0.5, ymin = n + 0.5, ymax = n + 0.5 + band_height,
      interpolate = FALSE
    )
  if (length(boundaries) > 0L) {
    plot <- plot +
      ggplot2::geom_vline(xintercept = boundaries, colour = "white", linewidth = line_width(0.6), alpha = 0.8) +
      ggplot2::geom_hline(yintercept = n + 1 - boundaries, colour = "white", linewidth = line_width(0.6), alpha = 0.8)
  }
  plot <- plot +
    ggplot2::scale_fill_gradientn(colours = colours, limits = c(0, 1), name = colorbar_label) +
    ggplot2::scale_x_continuous(breaks = NULL) +
    ggplot2::scale_y_continuous(breaks = NULL) +
    ggplot2::coord_fixed(xlim = c(0.5, n + 0.5), ylim = c(0.5, n + 0.5 + band_height), expand = FALSE) +
    ggplot2::labs(x = "Samples (ordered by cluster)", y = NULL, title = title) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(panel.grid = ggplot2::element_blank(), panel.border = ggplot2::element_blank())
  if (!isTRUE(colorbar)) {
    plot <- plot + ggplot2::guides(fill = "none")
  }
  plot
}

# The finite scores of each cluster in plotting order, and the clusters
# kept: those with at least one finite score.
cluster_score_groups <- function(scores, labels, order = NULL) {
  if (!is.null(dim(scores)) && length(dim(scores)) > 1L) {
    stop("scores must be a 1D array.", call. = FALSE)
  }
  if (!is.null(dim(labels)) && length(dim(labels)) > 1L) {
    stop("labels must be a 1D array.", call. = FALSE)
  }
  scores <- as.numeric(scores)
  labels <- as.vector(labels)
  if (length(scores) != length(labels)) {
    stop("scores and labels must have matching length.", call. = FALSE)
  }
  finite <- is.finite(scores)
  scores <- scores[finite]
  labels <- labels[finite]
  if (length(scores) == 0L) {
    stop("No finite scores available for plotting.", call. = FALSE)
  }
  wanted <- if (is.null(order)) sort(unique(labels)) else order
  groups <- lapply(wanted, function(label) scores[labels == label])
  kept <- lengths(groups) > 0L
  if (!any(kept)) {
    stop("No cluster values found for the provided order.", call. = FALSE)
  }
  list(groups = groups[kept], order = wanted[kept])
}

score_frame <- function(prepared) {
  data.frame(
    position = rep(seq_along(prepared$groups), lengths(prepared$groups)),
    score = unlist(prepared$groups, use.names = FALSE)
  )
}

# With fit_ylim the axis runs over the scores with a margin of 5% of their
# range, at least 0.02; otherwise it is ylim, which may be NULL.
score_ylim <- function(groups, ylim, fit_ylim) {
  if (!isTRUE(fit_ylim)) {
    return(ylim)
  }
  values <- unlist(groups, use.names = FALSE)
  margin <- max(0.02, 0.05 * (max(values) - min(values)))
  c(min(values) - margin, max(values) + margin)
}

# The axis, limits, labels and theme the box and violin plots share. The
# clusters sit at x = 1, 2, ... and the axis shows their labels.
score_plot_frame <- function(plot, prepared, ylim, fit_ylim, title, xlabel, ylabel, annotation,
                             rotation) {
  plot <- plot +
    ggplot2::scale_x_continuous(
      breaks = seq_along(prepared$order),
      labels = as.character(prepared$order),
      minor_breaks = NULL
    ) +
    ggplot2::labs(x = xlabel, y = ylabel, title = title, caption = annotation) +
    carve_theme() +
    ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())
  limits <- score_ylim(prepared$groups, ylim, fit_ylim)
  if (!is.null(limits)) {
    plot <- plot +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(0)) +
      ggplot2::coord_cartesian(ylim = limits)
  }
  if (!is.null(rotation)) {
    plot <- plot + ggplot2::theme(axis.text.x = ggplot2::element_text(angle = rotation))
  }
  plot
}

cluster_fills <- function(prepared, palette) {
  stats::setNames(palette_colors(palette, length(prepared$groups)), seq_along(prepared$groups))
}

cluster_boxplot_plot <- function(scores, labels, order = NULL, palette = "Accent",
                                 showfliers = FALSE, width = 0.75, title = NULL,
                                 xlabel = "Cluster", ylabel = "Uncertainty", annotation = NULL,
                                 rotation = NULL, ylim = c(-0.02, 1.02), fit_ylim = TRUE) {
  prepared <- cluster_score_groups(scores, labels, order)
  data <- score_frame(prepared)
  plot <- ggplot2::ggplot(data, ggplot2::aes(x = .data$position, y = .data$score, group = .data$position)) +
    ggplot2::geom_boxplot(
      ggplot2::aes(fill = factor(.data$position)),
      width = width, alpha = 0.8, linewidth = line_width(1.2),
      outlier.shape = if (isTRUE(showfliers)) 19 else NA
    ) +
    ggplot2::scale_fill_manual(values = cluster_fills(prepared, palette), guide = "none")
  score_plot_frame(plot, prepared, ylim, fit_ylim, title, xlabel, ylabel, annotation, rotation)
}

cluster_violin_plot <- function(scores, labels, order = NULL, palette = "Accent",
                                density_norm = "width", stripplot = TRUE, jitter = TRUE,
                                size = 8, alpha = 0.22, inner = "box", title = NULL,
                                xlabel = "Cluster", ylabel = "Uncertainty", annotation = NULL,
                                rotation = NULL, ylim = c(-0.02, 1.02), fit_ylim = TRUE) {
  if (!is.character(inner) || length(inner) != 1L || !inner %in% c("box", "quartile", "none")) {
    stop("inner must be one of: 'box', 'quartile', 'none'.", call. = FALSE)
  }
  if (!is.character(density_norm) || length(density_norm) != 1L ||
      !density_norm %in% c("width", "area", "count")) {
    warning(sprintf("Unknown density_norm=%s; using 'width'.", format_repr(density_norm)), call. = FALSE)
    density_norm <- "width"
  }
  prepared <- cluster_score_groups(scores, labels, order)
  data <- score_frame(prepared)
  # ggplot2 drops a group with fewer than two values from the violins, with
  # a warning. Such a cluster keeps its points and inner marks.
  dense <- data[data$position %in% which(lengths(prepared$groups) >= 2L), , drop = FALSE]
  plot <- ggplot2::ggplot(data, ggplot2::aes(x = .data$position, y = .data$score)) +
    ggplot2::geom_violin(
      data = dense,
      ggplot2::aes(group = .data$position, fill = factor(.data$position)),
      scale = density_norm, width = 0.8, colour = "black", linewidth = line_width(0.8),
      alpha = 0.8
    ) +
    ggplot2::scale_fill_manual(values = cluster_fills(prepared, palette), guide = "none")

  if (inner != "none") {
    quartiles <- t(vapply(
      prepared$groups, stats::quantile, numeric(3),
      probs = c(0.25, 0.5, 0.75), names = FALSE
    ))
    at <- seq_along(prepared$groups)
    segment <- function(half, from, to, width) {
      data.frame(x = at - half, xend = at + half, y = from, yend = to, width = line_width(width))
    }
    marks <- if (inner == "box") {
      rbind(
        segment(0.13, quartiles[, 2], quartiles[, 2], 1.5),
        data.frame(x = at, xend = at, y = quartiles[, 1], yend = quartiles[, 3], width = line_width(1.2)),
        segment(0.08, quartiles[, 1], quartiles[, 1], 1.0),
        segment(0.08, quartiles[, 3], quartiles[, 3], 1.0)
      )
    } else {
      rbind(
        segment(0.13, quartiles[, 1], quartiles[, 1], 1.0),
        segment(0.13, quartiles[, 2], quartiles[, 2], 1.5),
        segment(0.13, quartiles[, 3], quartiles[, 3], 1.0)
      )
    }
    plot <- plot +
      ggplot2::geom_segment(
        data = marks,
        ggplot2::aes(x = .data$x, xend = .data$xend, y = .data$y, yend = .data$yend, linewidth = .data$width),
        colour = "black", inherit.aes = FALSE
      ) +
      ggplot2::scale_linewidth_identity()
  }

  if (isTRUE(stripplot)) {
    jitter_width <- if (isTRUE(jitter)) 0.11 else if (isFALSE(jitter)) 0 else as.numeric(jitter)
    # A fixed seed makes the plot the same each time and leaves the
    # session's random number stream alone; Python's jitter is unseeded.
    plot <- plot + ggplot2::geom_point(
      position = ggplot2::position_jitter(width = jitter_width, height = 0, seed = 0L),
      shape = 16, size = point_size(size), alpha = alpha, colour = "black", stroke = 0
    )
  }
  score_plot_frame(plot, prepared, ylim, fit_ylim, title, xlabel, ylabel, annotation, rotation)
}

check_scatter_inputs <- function(X, labels, scores) {
  if (!is.null(dim(labels)) && length(dim(labels)) > 1L) {
    stop("labels must be a 1D array.", call. = FALSE)
  }
  if (!is.null(dim(scores)) && length(dim(scores)) > 1L) {
    stop("scores must be a 1D array.", call. = FALSE)
  }
  X <- as_data_matrix(X)
  labels <- as.vector(labels)
  scores <- as.numeric(scores)
  if (nrow(X) != length(labels) || nrow(X) != length(scores)) {
    stop("X, labels, and scores must have matching n_samples.", call. = FALSE)
  }
  if (!any(is.finite(scores))) {
    stop("No finite scores available for plotting.", call. = FALSE)
  }
  list(X = X, labels = labels, scores = scores)
}

check_annotation_style <- function(annotation_style) {
  if (!identical(annotation_style, "legend") && !identical(annotation_style, "box")) {
    stop("annotation_style must be 'legend' or 'box'.", call. = FALSE)
  }
}

# Two columns to draw: the embedding's first two, or the data's, or the
# first two principal components of data with more than two columns.
scatter_coordinates <- function(X, embedding = NULL) {
  if (is.null(embedding)) {
    if (ncol(X) > 2L) {
      return(PCA(X, n_components = 2L, random_state = 0L))
    }
    if (ncol(X) == 2L) {
      return(unname(X))
    }
    if (ncol(X) == 1L) {
      return(unname(cbind(X[, 1L], 0)))
    }
    stop("X must have at least 1 feature for scatter plotting.", call. = FALSE)
  }
  coords <- as.matrix(embedding)
  if (length(dim(coords)) != 2L || ncol(coords) < 2L) {
    stop("embedding must be a 2D array with at least 2 columns.", call. = FALSE)
  }
  if (nrow(coords) != nrow(X)) {
    stop("embedding must have the same number of rows as X.", call. = FALSE)
  }
  coords <- unname(coords[, 1:2, drop = FALSE])
  storage.mode(coords) <- "double"
  coords
}

# Marker area and opacity from the score. size_range and alpha_range give
# the value for the highest score first, so stable samples are small and
# faint, and unstable ones large and opaque. Samples without a finite score
# get NA and are not drawn.
scatter_encoding <- function(scores, size_range, alpha_range) {
  finite <- is.finite(scores)
  lo <- min(scores[finite])
  hi <- max(scores[finite])
  if (isclose(lo, hi)) {
    size <- rep(mean(size_range), length(scores))
    alpha <- rep(mean(alpha_range), length(scores))
  } else {
    norm <- (scores - lo) / (hi - lo)
    size <- size_range[[2L]] + norm * (size_range[[1L]] - size_range[[2L]])
    alpha <- alpha_range[[2L]] + norm * (alpha_range[[1L]] - alpha_range[[2L]])
  }
  size[!finite] <- NA_real_
  alpha[!finite] <- NA_real_
  list(size = size, alpha = pmin(pmax(alpha, 0), 1))
}

cluster_means <- function(scores, labels, clusters, empty) {
  vapply(clusters, function(label) {
    values <- scores[labels == label & is.finite(scores)]
    if (length(values) > 0L) mean(values) else empty
  }, numeric(1), USE.NAMES = FALSE)
}

# Labels, theme and legend position the two scatter plots share.
scatter_frame <- function(plot, title, xlabel, ylabel, caption, legend_loc, show_ticks, frameon) {
  plot <- plot +
    ggplot2::labs(x = xlabel, y = ylabel, title = title, caption = caption) +
    carve_theme() +
    ggplot2::theme(panel.grid = ggplot2::element_blank()) +
    legend_position(legend_loc)
  if (!isTRUE(show_ticks)) {
    plot <- plot + ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank())
  }
  if (!isTRUE(frameon)) {
    plot <- plot + ggplot2::theme(panel.border = ggplot2::element_blank())
  }
  plot
}

cluster_scatter_plot <- function(X, labels, scores, embedding = NULL, palette = "Accent",
                                 alpha_range = c(0.45, 0.9), size_range = c(15, 60),
                                 sort_order = TRUE, legend = TRUE, legend_loc = "right",
                                 annotation = NULL, annotation_style = "legend", title = NULL,
                                 scores_name = "Score", xlabel = "Component 1",
                                 ylabel = "Component 2", show_ticks = FALSE, frameon = FALSE) {
  check_annotation_style(annotation_style)
  inputs <- check_scatter_inputs(X, labels, scores)
  coords <- scatter_coordinates(inputs$X, embedding)
  labels <- inputs$labels
  scores <- inputs$scores
  encoding <- scatter_encoding(scores, size_range, alpha_range)
  clusters <- sort(unique(labels))
  keys <- as.character(clusters)
  means <- cluster_means(scores, labels, clusters, NaN)
  data <- data.frame(
    x = coords[, 1L],
    y = coords[, 2L],
    cluster = factor(as.character(labels), levels = keys),
    size = point_size(encoding$size),
    alpha = encoding$alpha
  )
  # Samples without a finite score are not drawn, so they are not in the data.
  data <- data[is.finite(scores), , drop = FALSE]
  if (isTRUE(sort_order)) {
    # Faint markers first, so the opaque ones are drawn on top.
    data <- data[order(data$alpha), , drop = FALSE]
  }
  legend_title <- if (nzchar(scores_name)) paste0("Cluster, ", scores_name) else "Cluster"
  if (!is.null(annotation) && annotation_style == "legend") {
    legend_title <- paste0(annotation, "\n", legend_title)
  }
  plot <- ggplot2::ggplot(data, ggplot2::aes(x = .data$x, y = .data$y)) +
    ggplot2::geom_point(
      ggplot2::aes(fill = .data$cluster, size = .data$size, alpha = .data$alpha),
      shape = 21, colour = "black", stroke = edge_stroke(0.2), na.rm = TRUE
    ) +
    ggplot2::scale_fill_manual(
      values = stats::setNames(palette_colors(palette, length(clusters)), keys),
      breaks = keys,
      labels = sprintf("%s (Mean = %.2f)", keys, means),
      name = legend_title
    ) +
    ggplot2::scale_size_identity() +
    ggplot2::scale_alpha_identity() +
    ggplot2::guides(fill = if (isTRUE(legend)) {
      ggplot2::guide_legend(override.aes = list(size = point_size(49), alpha = 0.8, colour = NA))
    } else {
      "none"
    })
  caption <- if (!is.null(annotation) && annotation_style == "box") annotation else NULL
  scatter_frame(plot, title, xlabel, ylabel, caption, legend_loc, show_ticks, frameon)
}

# The score as a position between the lowest and the highest finite score
# (0.5 when they are equal, or for a non-finite score), and the opacity:
# alpha_range gives the value for the highest score first. A sample without
# a finite score is faint.
diagnostic_encoding <- function(scores, alpha_encoding, alpha_range) {
  finite <- is.finite(scores)
  lo <- min(scores[finite])
  hi <- max(scores[finite])
  norm <- rep(0.5, length(scores))
  if (!isclose(lo, hi)) {
    norm[finite] <- (scores[finite] - lo) / (hi - lo)
  }
  alpha <- rep(1, length(scores))
  if (isTRUE(alpha_encoding)) {
    if (isclose(alpha_range[[1L]], alpha_range[[2L]])) {
      alpha[] <- mean(alpha_range)
    } else {
      alpha[finite] <- alpha_range[[2L]] + norm[finite] * (alpha_range[[1L]] - alpha_range[[2L]])
    }
  }
  alpha[!finite] <- 0.2
  list(norm = norm, alpha = alpha, limits = c(lo, hi))
}

# Clusters with the highest mean score first, and within a cluster the
# highest scores first, so unstable samples are drawn on top.
diagnostic_draw_order <- function(labels, scores, norm, sort_order) {
  clusters <- sort(unique(labels))
  means <- cluster_means(scores, labels, clusters, 0)
  rows <- lapply(clusters[order(-means)], function(label) {
    members <- which(labels == label)
    if (isTRUE(sort_order)) members[order(-norm[members])] else members
  })
  unlist(rows, use.names = FALSE)
}

diagnostic_scatter_plot <- function(X, labels, scores, embedding = NULL, cmap = "Greens_r",
                                    alpha_encoding = TRUE, alpha_range = c(0.3, 1),
                                    marker_size = 30, marker_linewidth = 0.2, markers = NULL,
                                    sort_order = TRUE, legend = TRUE, legend_loc = "right",
                                    colorbar = TRUE, colorbar_label = NULL, annotation = NULL,
                                    annotation_style = "legend", title = NULL,
                                    scores_name = "Score", xlabel = "Component 1",
                                    ylabel = "Component 2", show_ticks = FALSE, frameon = FALSE) {
  check_annotation_style(annotation_style)
  inputs <- check_scatter_inputs(X, labels, scores)
  coords <- scatter_coordinates(inputs$X, embedding)
  labels <- inputs$labels
  scores <- inputs$scores
  if (is.null(markers)) {
    markers <- DIAGNOSTIC_MARKERS
  }
  if (!is.numeric(markers) || length(markers) == 0L || !all(markers %in% 21:25)) {
    stop("markers must be ggplot2 shapes 21 to 25, the shapes with a fill.", call. = FALSE)
  }
  clusters <- sort(unique(labels))
  keys <- as.character(clusters)
  if (length(clusters) > length(markers)) {
    warning(sprintf(
      "Number of clusters (%d) exceeds available markers (%d). Markers will cycle.",
      length(clusters), length(markers)
    ), call. = FALSE)
  }
  shapes <- stats::setNames(markers[(seq_along(clusters) - 1L) %% length(markers) + 1L], keys)
  encoding <- diagnostic_encoding(scores, alpha_encoding, alpha_range)
  rows <- diagnostic_draw_order(labels, scores, encoding$norm, sort_order)
  data <- data.frame(
    x = coords[rows, 1L],
    y = coords[rows, 2L],
    score = scores[rows],
    cluster = factor(as.character(labels[rows]), levels = keys),
    alpha = encoding$alpha[rows]
  )
  legend_title <- "Cluster"
  if (!is.null(annotation) && annotation_style == "legend") {
    legend_title <- paste0(annotation, "\n", legend_title)
  }
  plot <- ggplot2::ggplot(data, ggplot2::aes(x = .data$x, y = .data$y)) +
    ggplot2::geom_point(
      ggplot2::aes(fill = .data$score, shape = .data$cluster, alpha = .data$alpha),
      size = point_size(marker_size), colour = "black", stroke = edge_stroke(marker_linewidth)
    ) +
    ggplot2::scale_fill_gradientn(
      colours = continuous_colours(cmap),
      limits = encoding$limits,
      na.value = "#808080",
      name = if (is.null(colorbar_label)) scores_name else colorbar_label
    ) +
    ggplot2::scale_shape_manual(values = shapes, breaks = keys, name = legend_title) +
    ggplot2::scale_alpha_identity() +
    ggplot2::guides(
      fill = if (isTRUE(colorbar)) {
        ggplot2::guide_colourbar(direction = "horizontal", position = "bottom")
      } else {
        "none"
      },
      shape = if (isTRUE(legend)) {
        ggplot2::guide_legend(override.aes = list(fill = "gray50", size = point_size(49), alpha = 1))
      } else {
        "none"
      }
    )
  caption <- if (!is.null(annotation) && annotation_style == "box") annotation else NULL
  scatter_frame(plot, title, xlabel, ylabel, caption, legend_loc, show_ticks, frameon)
}
