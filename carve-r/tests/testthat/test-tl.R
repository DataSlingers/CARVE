blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
run <- function(object, ...) run_carve(object, n_resamples = 6, estimator_param_grids = k_grid, ...)
direct_fit <- function(...) carve(blobs$X, n_resamples = 6, estimator_param_grids = k_grid, random_state = 0, ...)
sce <- make_sce(blobs$X)
written <- run(sce)
fit <- direct_fit()

test_that("run_carve writes the labels, the scores and the record", {
  data <- SummarizedExperiment::colData(written)
  expect_true(all(c("carve", "carve_stability", "carve_stability_ce", "carve_generalizability") %in% colnames(data)))
  expect_s3_class(data$carve, "factor")
  expect_identical(levels(data$carve), c("1", "2", "3"))
  for (column in c("carve_stability", "carve_stability_ce", "carve_generalizability")) {
    expect_type(data[[column]], "double")
    expect_length(data[[column]], 90L)
  }
  record <- S4Vectors::metadata(written)$carve
  expect_named(record, c("params", "results", "consensus"))
  expect_identical(dim(record$consensus), c(90L, 90L))
  expect_false(any(vapply(record$params, is.null, logical(1))))
})

test_that("the stored labels, scores, table and matrix are the fit's", {
  id <- select_row(fit, "stability", "1se")$config_id
  data <- SummarizedExperiment::colData(written)
  expect_identical(as.integer(as.character(data$carve)), get_labels(fit))
  expect_equal(data$carve_stability, sample_scores(fit, id, "gini"))
  expect_equal(data$carve_stability_ce, sample_scores(fit, id, "ce"))
  expect_equal(data$carve_generalizability, sample_scores(fit, id, "accuracy"))
  # At k = 3 the Gini and CE scores are both 1 on separated blobs; at k = 2
  # they differ, so a swapped column shows here.
  id2 <- select_row(fit, "stability", "1se", k = 2)$config_id
  data2 <- SummarizedExperiment::colData(attach_results(sce, fit, k = 2))
  expect_equal(data2$carve_stability, sample_scores(fit, id2, "gini"))
  expect_equal(data2$carve_stability_ce, sample_scores(fit, id2, "ce"))
  record <- S4Vectors::metadata(written)$carve
  expect_equal(record$consensus, consensus_matrix(fit, id))
  expect_identical(record$results, estimator_results(fit))
})

test_that("the record's params name the run and the selection", {
  params <- S4Vectors::metadata(written)$carve$params
  selected <- select_row(fit, "stability", "1se")
  expect_identical(params$measure, "stability")
  expect_identical(params$rule, "1se")
  expect_false(params$not_two)
  expect_false(params$pinned)
  expect_identical(params$sweep_param, "n_clusters")
  expect_equal(params$n_resamples, 6)
  expect_equal(params$subsample_ratio, 0.618)
  expect_identical(params$noise_policy, "drop")
  expect_identical(params$mode, "default")
  expect_equal(params$random_state, 0)
  expect_identical(params$estimator_param_grids, "custom")
  expect_equal(params$selected_config_id, selected$config_id)
  expect_equal(params$selected_k, get_k(fit))
  expect_equal(params$n_obs, 90)
  expect_equal(params$sweep_values, c(2, 3, 4))
  expect_equal(params$selected_n_clusters, selected$row$n_clusters[[1L]])
  expect_identical(params$selected_estimator, "KMeans")
  expect_identical(params$selected_method_label, as.character(selected$row$method_label))
  expect_identical(params$selected_method_id, as.character(selected$row$method_id))
  expect_identical(params$carve_version, as.character(utils::packageVersion("CARVE")))
  expect_false(any(c("n_consensus_anchors", "assay", "reduction", "n_dims", "consensus_k") %in% names(params)))
  recorded <- S4Vectors::metadata(run(sce, assay = "logcounts", n_dims = 1, consensus_k = 2))$carve$params
  expect_identical(recorded$assay, "logcounts")
  expect_equal(recorded$n_dims, 1)
  expect_equal(recorded$consensus_k, 2)
})

test_that("label levels are in numeric order", {
  many <- attach_results(sce, fit, consensus_k = 11)
  expect_identical(levels(SummarizedExperiment::colData(many)$carve), as.character(1:11))
})

test_that("key names every column and the record, and two keys coexist", {
  other <- run(sce, key = "other")
  data <- SummarizedExperiment::colData(other)
  expect_true(all(c("other", "other_stability", "other_stability_ce", "other_generalizability") %in% colnames(data)))
  expect_false("carve" %in% colnames(data))
  expect_named(S4Vectors::metadata(other), "other")
  both <- run(written, key = "second", rule = "max")
  expect_true(all(c("carve", "second") %in% names(S4Vectors::metadata(both))))
  expect_identical(S4Vectors::metadata(both)$second$params$rule, "max")
})

test_that("store_consensus and store_results leave entries out, and a call replaces the record", {
  no_matrix <- run(sce, store_consensus = FALSE)
  expect_named(S4Vectors::metadata(no_matrix)$carve, c("params", "results"))
  no_table <- run(sce, store_results = FALSE)
  expect_named(S4Vectors::metadata(no_table)$carve, c("params", "consensus"))
  replaced <- run(written, store_consensus = FALSE)
  expect_null(S4Vectors::metadata(replaced)$carve$consensus)
})

test_that("a mode skips its columns, and an earlier column stays", {
  stability <- muffle(run(sce, mode = "stability"), "Non-default mode is experimental")
  data <- SummarizedExperiment::colData(stability)
  expect_true("carve_stability" %in% colnames(data))
  expect_false("carve_generalizability" %in% colnames(data))
  generalizability <- muffle(run(sce, mode = "generalizability", measure = "generalizability"), "Non-default mode is experimental")
  data <- SummarizedExperiment::colData(generalizability)
  expect_false("carve_stability" %in% colnames(data))
  general_fit <- muffle(direct_fit(mode = "generalizability"), "Non-default mode is experimental")
  id <- select_row(general_fit, "generalizability", "1se")$config_id
  expect_equal(S4Vectors::metadata(generalizability)$carve$consensus, consensus_matrix(general_fit, id, type = "generalizability"))
  expect_identical(as.integer(as.character(data$carve)), get_labels(general_fit, measure = "generalizability", mode = "generalizability"))
  rerun <- muffle(run(written, mode = "stability"), "Non-default mode is experimental")
  expect_identical(
    SummarizedExperiment::colData(rerun)$carve_generalizability,
    SummarizedExperiment::colData(written)$carve_generalizability
  )
  expect_error(
    muffle(run(sce, mode = "stability", measure = "generalizability"), "Non-default mode is experimental"),
    "No configuration has a value for ari_generalizability.",
    fixed = TRUE
  )
})

test_that("run_carve under mode = 'generalizability' needs a generalizability measure", {
  # As in Python, the fit runs and the selection on the default measure,
  # stability, then finds no stability scores.
  expect_error(
    muffle(run(sce, mode = "generalizability"), "Non-default mode is experimental"),
    "No configuration has a value for ari_stability.",
    fixed = TRUE
  )
})

test_that("reference_key names a column whose labels the stored ones match", {
  with_truth <- sce
  SummarizedExperiment::colData(with_truth)$truth <- c(3, 1, 2)[blobs$y]
  matched <- run(with_truth, reference_key = "truth")
  expect_identical(as.integer(as.character(SummarizedExperiment::colData(matched)$carve)), as.integer(c(3, 1, 2)[blobs$y]))
  expect_error(run(with_truth, reference_key = "nope"), "reference_key='nope' not found in colData(object). Available columns: ['truth']", fixed = TRUE)
})

test_that("k pins the configuration and a resolution run records its axis", {
  pinned <- S4Vectors::metadata(run(sce, k = 2))$carve$params
  expect_true(pinned$pinned)
  expect_equal(pinned$selected_k, 2)
  graph <- run_carve(sce, n_resamples = 4, estimator_param_grids = list(estimator_grid(LeidenClustering, resolution = c(0.5, 1))))
  params <- S4Vectors::metadata(graph)$carve$params
  expect_identical(params$sweep_param, "resolution")
  expect_true(params$selected_resolution %in% c(0.5, 1))
})

test_that("attach_results stores what run_carve stores", {
  attached <- attach_results(sce, fit)
  expect_identical(SummarizedExperiment::colData(attached), SummarizedExperiment::colData(written))
  expect_identical(S4Vectors::metadata(attached), S4Vectors::metadata(written))
})

test_that("attach_results checks its inputs", {
  expect_error(attach_results(sce, list()), "fit must be a CARVE object from carve().", fixed = TRUE)
  expect_error(
    attach_results(blobs$X, fit),
    "object must be a SingleCellExperiment or a Seurat object, not an object of class 'matrix'.",
    fixed = TRUE
  )
  small <- make_sce(blobs$X[1:60, ])
  expect_error(
    attach_results(small, fit),
    "The model produced 90 consensus labels but object has 60 observations. The model was fitted on different data.",
    fixed = TRUE
  )
  id <- as.character(select_row(fit, "stability", "1se")$config_id)
  broken <- fit
  broken@consensus_matrices[[id]] <- diag(5)
  local_mocked_bindings(get_labels = function(fit, ...) rep(1:3, each = 30L))
  expect_error(
    attach_results(sce, broken),
    "Consensus matrix has shape (5, 5), but object has 90 observations. The model was fitted on different data.",
    fixed = TRUE
  )
})

test_that("an anchored run records its anchor count and stores no block", {
  anchored <- muffle(direct_fit(anchor_threshold = 30), "anchored consensus")
  id <- select_row(anchored, "stability", "1se")$config_id
  expect_warning(
    stored <- attach_results(sce, anchored),
    sprintf(
      "Anchored consensus is active: the consensus matrix for 'carve' is a 30x30 anchor block, not a cell-by-cell matrix. metadata(object)$carve holds only the latter, so nothing was written to metadata(object)$carve$consensus. The block itself is still available with consensus_matrix(fit, config_id = %d), and the anchor count is recorded in metadata(object)$carve$params$n_consensus_anchors.",
      id
    ),
    fixed = TRUE
  )
  record <- S4Vectors::metadata(stored)$carve
  expect_null(record$consensus)
  expect_equal(record$params$n_consensus_anchors, 30)
  chosen <- muffle(direct_fit(consensus_anchors = 20), "anchored consensus")
  expect_equal(
    S4Vectors::metadata(muffle(attach_results(sce, chosen), "Anchored consensus is active"))$carve$params$n_consensus_anchors,
    20
  )
})

test_that("a large consensus matrix is stored with a warning", {
  local_mocked_bindings(CONSENSUS_WARN_CELLS = 50L)
  expect_warning(
    stored <- attach_results(sce, fit),
    "Storing a dense 90x90 consensus matrix in metadata(object)$carve$consensus (~0.0 GB). Pass store_consensus=FALSE if you do not need it.",
    fixed = TRUE
  )
  expect_identical(dim(S4Vectors::metadata(stored)$carve$consensus), c(90L, 90L))
})

test_that("a randomized run stores its per-pipeline table", {
  options <- list(
    normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
    dim_reduction_options = list(preprocessing_option(Identity))
  )
  randomized_fit <- do.call(direct_fit, c(list(randomize_preprocessing = TRUE), options))
  stored <- attach_results(sce, randomized_fit)
  record <- S4Vectors::metadata(stored)$carve
  expect_identical(record$preprocessing_results, preprocessing_results(randomized_fit))
  expect_identical(record$params$selected_method_id, as.character(select_row(randomized_fit, "stability", "1se")$row$method_id))
  expect_null(S4Vectors::metadata(written)$carve$preprocessing_results)
  expect_null(S4Vectors::metadata(attach_results(sce, randomized_fit, store_results = FALSE))$carve$preprocessing_results)
})

test_that("run_carve writes the same results into a Seurat object", {
  object <- make_seurat(blobs$X)
  stored <- run(object)
  meta <- methods::slot(stored, "meta.data")
  data <- SummarizedExperiment::colData(written)
  expect_identical(meta$carve, data$carve)
  expect_equal(meta$carve_stability, data$carve_stability)
  expect_equal(meta$carve_generalizability, data$carve_generalizability)
  record <- methods::slot(stored, "misc")$carve
  expect_identical(record$results, S4Vectors::metadata(written)$carve$results)
  expect_equal(record$consensus, S4Vectors::metadata(written)$carve$consensus)
  expect_no_warning(again <- run(stored, rule = "max"))
  expect_identical(methods::slot(again, "misc")$carve$params$rule, "max")
  anchored <- muffle(direct_fit(anchor_threshold = 30), "anchored consensus")
  expect_warning(attach_results(object, anchored), "Misc(object, 'carve') holds only the latter", fixed = TRUE)
})
