# Results tables of the Python plotting tests (tests/test_plotting.py and
# tests/conftest.py), with the columns the R runner writes, and helpers that
# read a ggplot object.

metric_results <- function() {
  data.frame(
    config_id = 0:2,
    method_id = "m0",
    method_label = "KMeans, linkage=ward",
    estimator = "KMeans",
    n_clusters = 2:4,
    linkage = "ward",
    ari_stability = c(0.9, 0.85, 0.7),
    ari_stability_se = c(0.02, 0.03, 0.05),
    ari_stability_upper = c(0.92, 0.88, 0.75),
    ari_stability_lower = c(0.88, 0.82, 0.65),
    ari_generalizability = c(0.85, 0.80, 0.65),
    ari_generalizability_se = c(0.03, 0.04, 0.06),
    ari_generalizability_upper = c(0.88, 0.84, 0.71),
    ari_generalizability_lower = c(0.82, 0.76, 0.59),
    sweep_param = "n_clusters",
    sweep_value = c(2, 3, 4),
    sweep_rank = 0:2,
    n_clusters_observed = c(2, 3, 4),
    n_clusters_observed_se = 0,
    noise_fraction = 0
  )
}

# Two curves, one with a parameter the other lacks.
two_method_results <- function() {
  data.frame(
    config_id = 0:5,
    method_id = rep(c("m0", "m1"), each = 3),
    method_label = rep(c("KMeans", "SpectralClustering, affinity=self_tuning"), each = 3),
    estimator = rep(c("KMeans", "SpectralClustering"), each = 3),
    n_clusters = c(2:4, 2:4),
    affinity = rep(c(NA, "self_tuning"), each = 3),
    ari_stability = c(0.9, 0.85, 0.7, 0.88, 0.82, 0.68),
    ari_stability_se = c(0.02, 0.03, 0.05, 0.02, 0.03, 0.05),
    sweep_param = "n_clusters",
    sweep_value = c(2, 3, 4, 2, 3, 4),
    sweep_rank = c(0:2, 0:2),
    n_clusters_observed = c(2, 3, 4, 2, 3, 4),
    n_clusters_observed_se = 0,
    noise_fraction = 0
  )
}

resolution_results <- function() {
  data.frame(
    config_id = 0:3,
    method_id = "m0",
    method_label = "LeidenClustering, n_neighbors=15",
    estimator = "LeidenClustering",
    resolution = c(0.25, 0.5, 1, 2),
    n_neighbors = 15L,
    ari_stability = c(0.9, 0.85, 0.7, 0.6),
    ari_stability_se = c(0.02, 0.03, 0.05, 0.06),
    ari_stability_upper = c(0.92, 0.88, 0.75, 0.66),
    ari_stability_lower = c(0.88, 0.82, 0.65, 0.54),
    sweep_param = "resolution",
    sweep_value = c(0.25, 0.5, 1, 2),
    sweep_rank = 0:3,
    n_clusters_observed = c(2, 3, 5, 9),
    n_clusters_observed_se = c(0, 0.1, 0.3, 0.5),
    noise_fraction = 0
  )
}

# A preprocessing_results() table over two configurations. m0 has two
# pipelines whose values peak at k = 3. m1 has a third pipeline, a decoy
# peaking at k = 4 above anything in m0, so a plot or a selection that reads
# the whole table instead of m0's rows is visibly wrong.
pipeline_results <- function() {
  curves <- list(
    list("m0", "identity | identity", c(0.70, 0.90, 0.60)),
    list("m0", "identity | PCA(n_components=2)", c(0.65, 0.80, 0.50)),
    list("m1", "StandardScaler | identity", c(0.20, 0.30, 0.99))
  )
  rows <- lapply(curves, function(curve) {
    steps <- strsplit(curve[[2L]], " | ", fixed = TRUE)[[1L]]
    data.frame(
      method_id = curve[[1L]],
      method_label = paste("KMeans", curve[[1L]]),
      pipeline = curve[[2L]],
      normalization = steps[[1L]],
      dim_reduction = steps[[2L]],
      n_clusters = 2:4,
      n_resamples = 5L,
      ari_stability = curve[[3L]],
      ari_stability_se = 0.01,
      ari_generalizability = curve[[3L]] - 0.1,
      ari_generalizability_se = 0.02,
      n_clusters_observed = c(2, 3, 4),
      sweep_param = "n_clusters",
      sweep_value = c(2, 3, 4),
      sweep_rank = 0:2
    )
  })
  do.call(rbind, rows)
}

# The estimator_results() table behind pipeline_results(), pooled. It
# disagrees with the per-pipeline rows on purpose. Under "1se" the best row,
# m1 at k = 2 with SE 0.10, admits m0 at k = 4, so CARVE selects (m0, 4); the
# rule over m0's rows alone stops at k = 3. For m1 the rule over its own
# pooled rows stops at k = 2, where its per-pipeline rows peak at k = 4.
pooled_results <- function() {
  data.frame(
    config_id = 0:5,
    method_id = rep(c("m0", "m1"), each = 3),
    method_label = rep(c("KMeans m0", "KMeans m1"), each = 3),
    n_clusters = c(2:4, 2:4),
    ari_stability = c(0.80, 0.90, 0.86, 0.95, 0.40, 0.30),
    ari_stability_se = c(0.01, 0.01, 0.01, 0.10, 0.01, 0.01),
    ari_generalizability = c(0.80, 0.90, 0.86, 0.95, 0.40, 0.30) - 0.1,
    ari_generalizability_se = c(0.01, 0.01, 0.01, 0.10, 0.01, 0.01),
    n_clusters_observed = c(2, 3, 4, 2, 3, 4),
    sweep_param = "n_clusters",
    sweep_value = c(2, 3, 4, 2, 3, 4),
    sweep_rank = c(0:2, 0:2)
  )
}

# A resolution table over two methods. The selections land on different
# rows, and two counts differ between truncation and rounding (6.6 and 2.2):
# stability max is Leiden at 1 (6.6 clusters); stability 1se admits Leiden
# at 2 (11 clusters); generalizability max is Louvain at 0.5 (2.2); with
# not_two, Louvain at 0.5 rounds to two clusters and is left out, which
# leaves Leiden at 1.
realized_count_results <- function() {
  data.frame(
    config_id = 0:5,
    method_id = rep(c("m0", "m1"), each = 3),
    method_label = rep(c("LeidenClustering", "LouvainClustering"), each = 3),
    estimator = rep(c("LeidenClustering", "LouvainClustering"), each = 3),
    resolution = c(0.5, 1, 2, 0.5, 1, 2),
    ari_stability = c(0.80, 0.90, 0.88, 0.70, 0.75, 0.60),
    ari_stability_se = c(0.01, 0.03, 0.01, 0.01, 0.01, 0.01),
    ari_generalizability = c(0.60, 0.70, 0.65, 0.95, 0.50, 0.40),
    ari_generalizability_se = 0.01,
    sweep_param = "resolution",
    sweep_value = c(0.5, 1, 2, 0.5, 1, 2),
    sweep_rank = c(0:2, 0:2),
    n_clusters_observed = c(3.2, 6.6, 11.0, 2.2, 5.0, 9.4),
    n_clusters_observed_se = c(0.2, 0.4, 0.0, 0.1, 0.3, 0.6),
    noise_fraction = 0
  )
}

layer_index <- function(plot, geom) {
  which(vapply(plot$layers, function(layer) inherits(layer$geom, geom), logical(1)))
}

has_layer <- function(plot, geom) {
  length(layer_index(plot, geom)) > 0L
}

layer_with <- function(plot, geom) {
  index <- layer_index(plot, geom)
  if (length(index) == 0L) {
    return(NULL)
  }
  ggplot2::layer_data(plot, index[[1L]])
}

guide_labels <- function(plot, aesthetic) {
  as.character(ggplot2::get_guide_data(plot, aesthetic)$.label)
}

panel_breaks <- function(plot, axis) {
  breaks <- ggplot2::ggplot_build(plot)$layout$panel_params[[1L]][[axis]]$breaks
  as.numeric(breaks[!is.na(breaks)])
}

x_breaks <- function(plot) panel_breaks(plot, "x")
y_breaks <- function(plot) panel_breaks(plot, "y")
