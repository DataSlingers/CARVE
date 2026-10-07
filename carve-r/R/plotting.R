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
