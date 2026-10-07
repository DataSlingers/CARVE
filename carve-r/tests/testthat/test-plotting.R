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
