blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
# Ten resamples: with five, one sample is never drawn with a partner and its consensus row is
# all 0.5, so the cut places it arbitrarily.
fit <- carve(blobs$X, n_clusters = 2:4, n_resamples = 10, estimator_param_grids = k_grid, random_state = 0)

single_mode_fit <- function(mode) {
  suppressWarnings(carve(blobs$X, n_clusters = 2:3, n_resamples = 2, random_state = 0, mode = mode,
                         estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3))))
}

test_that("get_k and get_sweep_value pick k = 3 on three blobs", {
  expect_identical(get_k(fit), 3L)
  expect_identical(get_sweep_value(fit), 3)
  expect_identical(get_k(fit, measure = "generalizability", rule = "max"), 3L)
})

test_that("a measure without standard errors falls back to max with a warning", {
  k <- NULL
  expect_warning(
    k <- get_k(fit, measure = "pac"),
    "Column 'consensus_pac_stability_se' not found; falling back to 'max' rule.",
    fixed = TRUE
  )
  expect_identical(k, get_k(fit, measure = "pac", rule = "max"))
})

test_that("get_labels recovers the blobs, pins k and overrides the cut", {
  labels <- get_labels(fit)
  expect_type(labels, "integer")
  expect_length(labels, 90L)
  expect_identical(adjusted_rand_index(labels, blobs$y), 1)
  expect_identical(count_clusters(get_labels(fit, k = 2)), 2L)
  expect_identical(count_clusters(get_labels(fit, k = 4)), 4L)
  expect_identical(count_clusters(get_labels(fit, sweep_value = 4)), 4L)
  expect_identical(count_clusters(get_labels(fit, consensus_k = 4)), 4L)
})

test_that("get_labels leaves the fit unchanged", {
  before <- fit
  get_labels(fit, k = 2)
  expect_identical(fit, before)
})

test_that("get_labels aligns to reference labels with the same number of clusters", {
  natural <- get_labels(fit, k = 3)
  permuted <- c(2L, 3L, 1L)[natural]
  expect_identical(get_labels(fit, k = 3, reference_labels = permuted), permuted)
  expect_identical(get_labels(fit, k = 2, reference_labels = permuted), get_labels(fit, k = 2))
})

test_that("the reference given to carve() is the default", {
  reference <- c(10L, 20L, 30L)[blobs$y]
  ref_fit <- carve(blobs$X, n_clusters = 2:4, n_resamples = 10, estimator_param_grids = k_grid,
                   random_state = 0, reference_labels = reference)
  expect_identical(get_labels(ref_fit, k = 3), reference)
})

test_that("reference entries of -1 take no part in the matching", {
  natural <- get_labels(fit, k = 3)
  reference <- c(2L, 3L, 1L)[natural]
  reference[c(1, 31, 61)] <- -1L
  expect_identical(get_labels(fit, k = 3, reference_labels = reference), c(2L, 3L, 1L)[natural])
})

test_that("mode = 'generalizability' cuts the held-out consensus", {
  expect_identical(
    get_labels(fit, k = 3, mode = "generalizability"),
    cut_consensus(consensus_matrix(fit, 1, type = "generalizability"), 3L)
  )
})

test_that("a custom estimator cuts the consensus distances", {
  first_rows <- function(D, n_clusters) rep_len(seq_len(n_clusters), nrow(D))
  expect_identical(get_labels(fit, k = 3, estimator = first_rows), rep_len(1:3, 90L))
})

test_that("get_labels reports requests it cannot meet", {
  expect_error(get_labels(fit, k = 7), "No configurations found for k=7.", fixed = TRUE)
  expect_error(get_labels(fit, k = 3, sweep_value = 3), "Pass at most one of k= and sweep_value=.", fixed = TRUE)
  expect_error(
    get_labels(fit, consensus_k = 1),
    "Cannot cut the consensus matrix at 1 cluster(s). The selected configuration is degenerate; pass consensus_k= explicitly or exclude that end of the sweep.",
    fixed = TRUE
  )
  expect_error(get_labels(fit, mode = "both"), "Unknown mode: 'both'.", fixed = TRUE)
  expect_error(get_labels(fit, kk = 3), "Unknown argument: kk.", fixed = TRUE)
})

test_that("k is only valid on the k axis", {
  cut_at <- function(X, resolution = 1) {
    as.integer(stats::cutree(stats::hclust(stats::dist(X)), k = round(2 * resolution)))
  }
  res_fit <- carve(blobs$X, n_resamples = 2, random_state = 0,
                   estimator_param_grids = list(estimator_grid(cut_at, resolution = c(1, 1.5))))
  expect_error(
    get_labels(res_fit, k = 3),
    "This run swept 'resolution', not 'n_clusters'. Use sweep_value=... to pin a resolution, and consensus_k=... to fix the number of clusters used to cut the consensus matrix.",
    fixed = TRUE
  )
  expect_identical(count_clusters(get_labels(res_fit, sweep_value = 1.5)), 3L)
})

test_that("a stability-only fit has no held-out consensus to cut", {
  expect_error(
    get_labels(single_mode_fit("stability"), mode = "generalizability"),
    "Consensus matrix not available for mode='generalizability'. This run likely used split-mode and skipped building that artifact.",
    fixed = TRUE
  )
})

test_that("not_two leaves out the two-cluster configuration", {
  two <- make_blobs(centers = rbind(c(0, 0), c(8, 0)))
  fit2 <- carve(two$X, n_clusters = 2:3, n_resamples = 5, random_state = 0,
                estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3)))
  expect_identical(get_k(fit2), 2L)
  expect_identical(get_k(fit2, not_two = TRUE), 3L)
  expect_identical(count_clusters(get_labels(fit2, not_two = TRUE)), 3L)
  expect_identical(attr(get_estimator(fit2, not_two = TRUE), "params"), list(n_clusters = 3L))
})

test_that("get_estimator rebuilds the selected configuration", {
  selected <- get_estimator(fit)
  expect_identical(attr(selected, "estimator"), "KMeans")
  expect_identical(attr(selected, "params"), list(n_clusters = 3L))
  expect_identical(selected(blobs$X, random_state = 0L), KMeans(blobs$X, n_clusters = 3L, random_state = 0L))
})

test_that("the accessors return the fit's contents", {
  expect_identical(estimator_results(fit), fit@estimator_results)
  expect_identical(estimator_param_grids(fit), fit@estimator_param_grids)
  expect_identical(sweep_spec(fit), fit@sweep)
  expect_identical(input_data(fit), blobs$X)
  expect_identical(consensus_matrix(fit, 1), fit@consensus_matrices[["1"]])
  expect_identical(consensus_matrix(fit, 1, type = "generalizability"), fit@consensus_generalizability_matrices[["1"]])
  expect_identical(sample_scores(fit, 2), fit@stability_gini_scores[["2"]])
  expect_identical(sample_scores(fit, 2, "ce"), fit@stability_ce_scores[["2"]])
  expect_identical(sample_scores(fit, 2, "accuracy"), fit@generalizability_scores[["2"]])
})

test_that("an unknown config_id or source is an error", {
  expect_error(consensus_matrix(fit, 5), "config_id 5 is not in estimator_results(fit).", fixed = TRUE)
  expect_error(sample_scores(fit, 0, "silhouette"), "source must be one of: 'accuracy', 'gini', 'ce'.", fixed = TRUE)
})

test_that("scores and matrices a mode skipped are reported missing", {
  stab <- single_mode_fit("stability")
  expect_error(sample_scores(stab, 0, "accuracy"), "Generalizability scores are not available for this run.", fixed = TRUE)
  expect_error(
    consensus_matrix(stab, 0, "generalizability"),
    "The generalizability consensus matrix is not available for this fit (mode = 'stability').",
    fixed = TRUE
  )
  gen <- single_mode_fit("generalizability")
  expect_error(sample_scores(gen, 0, "gini"), "Gini stability scores are not available for this run.", fixed = TRUE)
  expect_error(sample_scores(gen, 0, "ce"), "CE stability scores are not available for this run.", fixed = TRUE)
})

test_that("cut_consensus matches Python's consensus cut", {
  for (case in read_fixture("consensus_cut")$cases) {
    labels <- cut_consensus(fixture_matrix(case$consensus), case$k)
    expect_identical(adjusted_rand_index(labels, case$labels), 1, info = paste("k =", case$k))
  }
})

test_that("cut_consensus gives pairs never drawn together a distance of 0.5", {
  M <- matrix(c(1, 0.9, NaN, 0.9, 1, 0.2, NaN, 0.2, 1), 3)
  seen <- NULL
  cut_consensus(M, 2L, estimator = function(D, n_clusters) {
    seen <<- D
    c(1L, 1L, 2L)
  })
  expect_equal(seen[1, 3], 0.5)
  expect_equal(seen[3, 1], 0.5)
  expect_equal(seen[2, 3], 0.8)
  expect_equal(unname(diag(seen)), c(0, 0, 0))
})

test_that("show summarizes the fit", {
  expect_output(show(fit), "CARVE fit on 90 samples and 2 features", fixed = TRUE)
  expect_output(show(fit), "Sweep: n_clusters over 2, 3, 4", fixed = TRUE)
  expect_output(show(fit), "Estimators: KMeans (3 configurations)", fixed = TRUE)
})
