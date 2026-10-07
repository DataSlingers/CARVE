test_that("the header prints only at verbose 2 and reports the consensus memory", {
  grids <- list(estimator_grid(KMeans, n_clusters = 2:3))
  sweep <- resolve_sweep(n_clusters = 2:3)
  X <- matrix(0, 1000, 2)
  expect_silent(print_run_header(X, sweep, 10L, 0.618, grids, 1L, 0L, verbose = 1L))
  text <- paste(capture_messages(print_run_header(X, sweep, 10L, 0.618, grids, 1L, 0L, verbose = 2L)), collapse = "")
  expect_match(text, "[CARVE] Validation Settings:", fixed = TRUE)
  expect_match(text, "[CARVE] n_samples          : 1000", fixed = TRUE)
  expect_match(text, "[CARVE] n_clusters         : [2 3]", fixed = TRUE)
  expect_match(text, "[CARVE] total configs      : 2", fixed = TRUE)
  expect_match(text, "[CARVE] consensus memory   : 0.03 GB", fixed = TRUE)
  expect_match(text, "[CARVE] Starting validation ...", fixed = TRUE)
})

test_that("a single-criterion mode halves the projected memory", {
  grids <- list(estimator_grid(KMeans, n_clusters = 2:3))
  text <- paste(capture_messages(print_run_header(
    matrix(0, 1000, 2), resolve_sweep(n_clusters = 2:3), 10L, 0.618, grids, 1L, 0L,
    verbose = 2L, mode = "stability"
  )), collapse = "")
  expect_match(text, "[CARVE] consensus memory   : 0.02 GB", fixed = TRUE)
})

test_that("per-configuration progress prints from verbose 1", {
  record <- list(ari_stability = 0.9, ari_stability_se = 0.01,
                 ari_generalizability = 0.8, ari_generalizability_se = 0.02)
  expect_silent(log_config_progress(1L, 4L, "KMeans", list(n_clusters = 2L), record, "n_clusters", verbose = 0L))
  expect_message(
    log_config_progress(1L, 4L, "KMeans", list(n_clusters = 2L), record, "n_clusters", verbose = 1L),
    "[CARVE] [1/4] est=KMeans n_clusters=2 | ARI_stab=0.900\u00b10.010  ARI_gen=0.800\u00b10.020",
    fixed = TRUE
  )
})

test_that("the footer prints only at verbose 2", {
  expect_silent(print_run_footer(data.frame(a = 1:3), verbose = 1L))
  expect_message(
    print_run_footer(data.frame(a = 1:3), verbose = 2L),
    "[CARVE] finished. evaluated 3 estimator configurations.",
    fixed = TRUE
  )
})

header_text <- function(...) {
  grids <- list(estimator_grid(KMeans, n_clusters = 2:3))
  paste(capture_messages(print_run_header(
    ..., sweep = resolve_sweep(n_clusters = 2:3), n_resamples = 10L, subsample_ratio = 0.618,
    estimator_grids = grids, n_jobs = 1L, random_state = 0L, verbose = 2L
  )), collapse = "")
}

test_that("a randomized header names the options", {
  text <- header_text(
    X = matrix(0, 100, 2),
    randomize_preprocessing = TRUE,
    normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
    dim_reduction_options = list(preprocessing_option(PCA, n_components = 2L))
  )
  expect_match(text, "[CARVE] randomize_preproc  : TRUE", fixed = TRUE)
  expect_match(text, "[CARVE] normalization      : identity, StandardScaler", fixed = TRUE)
  expect_match(text, "[CARVE] dim_reduction      : PCA", fixed = TRUE)
})

test_that("without randomization the header lists no options", {
  text <- header_text(X = matrix(0, 100, 2))
  expect_match(text, "[CARVE] randomize_preproc  : FALSE", fixed = TRUE)
  expect_match(text, "[CARVE] consensus anchors  : none", fixed = TRUE)
  expect_false(grepl("normalization", text, fixed = TRUE))
})

test_that("an anchored header projects the memory of the anchor blocks", {
  # Two configurations, two matrices each, of 5,000 by 5,000 doubles; the
  # 20,000-sample matrices of an exact run would take 12.80 GB.
  text <- header_text(X = matrix(0, 20000, 1), anchors = seq_len(5000L))
  expect_match(text, "[CARVE] consensus anchors  : 5000", fixed = TRUE)
  expect_match(text, "[CARVE] consensus memory   : 0.80 GB", fixed = TRUE)
})
