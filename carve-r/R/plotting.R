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
