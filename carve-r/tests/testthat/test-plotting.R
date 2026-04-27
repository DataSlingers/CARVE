skip_if_not_installed("ggplot2")
skip_if_not_installed("patchwork")
skip_if_not_installed("ranger")

# Shared blob fixture: 3 well-separated 2D Gaussian blobs.
make_blobs <- function(n_per = 30L, seed = 1L) {
  set.seed(seed)
  centers <- rbind(c(0, 0), c(6, 0), c(3, 5))
  X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i) {
    matrix(stats::rnorm(n_per * 2L, 0, 0.4), ncol = 2L) +
      matrix(centers[i, ], n_per, 2L, byrow = TRUE)
  }))
  list(X = X, y = rep(seq_len(nrow(centers)), each = n_per))
}

fitted_carve <- function() {
  bl <- make_blobs(n_per = 25L)
  cv <- CARVE$new(
    n_clusters = 2:4,
    n_resamples = 6L,
    subsample_ratio = 0.8,
    random_state = 1L,
    verbose = 0L,
    estimator_param_grids = list(
      list(type = "kmeans", grid = list(n_clusters = 2:4))
    )
  )
  cv$fit(bl$X)
  cv
}


test_that(".discrete_palette returns Accent colors and cycles", {
  pal <- CARVE:::.discrete_palette(4, "Accent")
  expect_length(pal, 4L)
  expect_match(pal, "^#[0-9A-Fa-f]{6}$")

  # n > 8 cycles back to the first 8 colors.
  pal9 <- CARVE:::.discrete_palette(9L, "Accent")
  expect_equal(pal9[9L], pal9[1L])
})


test_that(".build_estimator_label formats hyperparameters", {
  row <- list(estimator = "KMeans", n_clusters = 3L,
              gamma = 0.7654, foo = NA)
  lab <- CARVE:::.build_estimator_label(row, exclude_cols = "n_clusters")
  expect_match(lab, "^KMeans")
  expect_match(lab, "gamma=0.765")
  # Excluded columns and NAs are dropped.
  expect_false(grepl("n_clusters", lab))
  expect_false(grepl("foo", lab))
})


test_that(".prepare_cluster_score_groups orders clusters numerically", {
  scores <- c(0.1, 0.2, 0.5, 0.6, 0.9)
  labels <- c(2L, 2L, 1L, 1L, 3L)
  prep <- CARVE:::.prepare_cluster_score_groups(scores, labels)
  expect_equal(prep$order, c(1L, 2L, 3L))
  expect_equal(prep$groups[[1L]], c(0.5, 0.6))
  expect_equal(prep$groups[[3L]], 0.9)
})


test_that("plot_metric_over_n_clusters returns a ggplot", {
  cv <- fitted_carve()
  p <- cv$plot_metric_over_n_clusters(measure = "stability", rule = "max")
  expect_s3_class(p, "ggplot")
  # Sanity check on labels.
  expect_match(p$labels$x, "Number of Clusters")
})


test_that("plot_consensus_matrix returns a patchwork object", {
  cv <- fitted_carve()
  p <- cv$plot_consensus_matrix(measure = "stability", rule = "max")
  expect_true(inherits(p, "patchwork"))
})


test_that("plot_consensus_matrix accepts an explicit k", {
  cv <- fitted_carve()
  p <- cv$plot_consensus_matrix(measure = "stability", rule = "max", k = 3L)
  expect_true(inherits(p, "patchwork"))
})


test_that("plot_consensus_matrix errors when k is not in results", {
  cv <- fitted_carve()
  expect_error(cv$plot_consensus_matrix(k = 99L), "No configurations")
})


test_that("plot_cluster_boxplot returns a ggplot for each source", {
  cv <- fitted_carve()
  for (src in c("gini", "ce", "accuracy")) {
    p <- cv$plot_cluster_boxplot(source = src, measure = "stability",
                                  rule = "max")
    expect_s3_class(p, "ggplot")
  }
})


test_that("plot_cluster_violin returns a ggplot with strip overlay", {
  cv <- fitted_carve()
  p <- cv$plot_cluster_violin(source = "gini", measure = "stability",
                               rule = "max", stripplot = TRUE)
  expect_s3_class(p, "ggplot")
  # Stripplot adds a geom_jitter layer in addition to violin + box.
  geoms <- vapply(p$layers, function(l) class(l$geom)[1L], character(1L))
  expect_true("GeomPoint" %in% geoms || "GeomJitter" %in% geoms)
})


test_that("plot_cluster_scatter and plot_diagnostic_scatter return ggplots", {
  cv <- fitted_carve()
  ps <- cv$plot_cluster_scatter(source = "gini", measure = "stability",
                                 rule = "max")
  pd <- cv$plot_diagnostic_scatter(source = "gini", measure = "stability",
                                    rule = "max")
  expect_s3_class(ps, "ggplot")
  expect_s3_class(pd, "ggplot")
})


test_that("scatter methods accept an explicit X after include_data=FALSE", {
  cv <- fitted_carve()
  X <- cv$X_
  cv$X_ <- NULL  # simulate save(include_data=FALSE)
  expect_error(cv$plot_cluster_scatter(source = "gini", measure = "stability",
                                        rule = "max"),
               "Raw data are not available")
  p <- cv$plot_cluster_scatter(source = "gini", measure = "stability",
                                rule = "max", X = X)
  expect_s3_class(p, "ggplot")
})


test_that("annotation = FALSE suppresses the label layer", {
  cv <- fitted_carve()
  p_with <- cv$plot_cluster_boxplot(source = "gini", measure = "stability",
                                     rule = "max", annotation = TRUE)
  p_without <- cv$plot_cluster_boxplot(source = "gini", measure = "stability",
                                        rule = "max", annotation = FALSE)
  has_label <- function(p) any(vapply(p$layers,
    function(l) inherits(l$geom, "GeomLabel"), logical(1L)))
  expect_true(has_label(p_with))
  expect_false(has_label(p_without))
})


test_that("plot methods error before fit()", {
  cv <- CARVE$new(n_clusters = 2:3, n_resamples = 4L)
  expect_error(cv$plot_metric_over_n_clusters(),
               "has not been fitted")
  expect_error(cv$plot_consensus_matrix(),
               "has not been fitted")
  expect_error(cv$plot_cluster_boxplot(),
               "has not been fitted")
  expect_error(cv$plot_cluster_violin(),
               "has not been fitted")
  expect_error(cv$plot_cluster_scatter(),
               "has not been fitted")
  expect_error(cv$plot_diagnostic_scatter(),
               "has not been fitted")
})
