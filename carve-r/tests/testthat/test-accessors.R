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

test_that("get_estimator keeps the parameters of an estimator that takes ...", {
  dots <- function(X, ...) KMeans(X, ...)
  dots_fit <- carve(blobs$X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
                    estimator_param_grids = list(estimator_grid(dots, n_clusters = 2:4)))
  expect_identical(get_k(dots_fit), 3L)
  selected <- get_estimator(dots_fit)
  expect_identical(attr(selected, "params"), list(n_clusters = 3L))
  expect_identical(count_clusters(selected(blobs$X)), 3L)
})

test_that("get_estimator keeps a random_state the grid sets", {
  grid_seed_fit <- carve(blobs$X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
                         estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4, random_state = 5L)))
  selected <- get_estimator(grid_seed_fit)
  expect_identical(attr(selected, "params"), list(n_clusters = 3L, random_state = 5L))
  # The grid's seed wins over the one passed to the returned function.
  expect_identical(selected(blobs$X, random_state = 0L), KMeans(blobs$X, n_clusters = 3L, random_state = 5L))
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

# Distinct scores on [0.5, 1] whose lowest length(low) values sit at low. At
# quantile 0.05 exactly those samples are flagged when length(low) is
# ceiling(0.05 * (n - 1)): 5 for 90 samples, 3 for 60.
known_scores <- function(n, low) {
  values <- seq(0.5, 1, length.out = n)
  scores <- numeric(n)
  scores[low] <- values[seq_along(low)]
  scores[-low] <- values[-seq_along(low)]
  scores
}

# A copy of the fit with scores at config_id and the reversed scores at
# every other configuration, so reading the wrong one moves the noise.
with_scores <- function(fit, source, config_id, scores) {
  name <- switch(source,
    gini = "stability_gini_scores",
    ce = "stability_ce_scores",
    accuracy = "generalizability_scores"
  )
  values <- methods::slot(fit, name)
  for (key in names(values)) {
    values[[key]] <- if (key == as.character(config_id)) scores else rev(scores)
  }
  methods::slot(fit, name) <- values
  fit
}

config_at <- function(fit, k) {
  results <- estimator_results(fit)
  results$config_id[results$sweep_value == k]
}

# The Python suite's noise positions, plus one: spread over the three blobs.
noise_low <- c(4L, 18L, 42L, 59L, 85L)

test_that("noise labels are off by default", {
  plain <- get_labels(fit, k = 3)
  expect_identical(get_labels(fit, k = 3, noise_labels = FALSE), plain)
  expect_false(any(plain == -1L))
})

test_that("noise lands on the samples with the lowest scores", {
  for (source in c("gini", "ce", "accuracy")) {
    noisy_fit <- with_scores(fit, source, config_at(fit, 3), known_scores(90L, noise_low))
    plain <- get_labels(noisy_fit, k = 3)
    noisy <- get_labels(noisy_fit, k = 3, noise_labels = TRUE, noise_score = source)
    expect_identical(which(noisy == -1L), noise_low, info = source)
    expect_identical(noisy[noisy != -1L], plain[noisy != -1L], info = source)
  }
})

test_that("noise scores are joined on config_id", {
  noisy_fit <- with_scores(fit, "gini", config_at(fit, 3), known_scores(90L, noise_low))
  noisy_fit@estimator_results <- noisy_fit@estimator_results[c(3L, 1L, 2L), ]
  # The k = 3 row no longer sits at position config_id + 1.
  expect_false(which(noisy_fit@estimator_results$n_clusters == 3) == config_at(fit, 3) + 1L)
  expect_identical(which(get_labels(noisy_fit, k = 3, noise_labels = TRUE) == -1L), noise_low)
})

test_that("ties at the cutoff warn with the counts", {
  scores <- rep(1, 90)
  scores[c(4L, 42L, 85L)] <- 0.5
  scores[c(11L, 26L, 36L, 51L, 66L, 81L)] <- 0.9
  noisy_fit <- with_scores(fit, "gini", config_at(fit, 3), scores)
  noisy <- NULL
  expect_warning(
    noisy <- get_labels(noisy_fit, k = 3, noise_labels = TRUE),
    "noise_quantile=0.05 asks for about 5 of 90 samples by gini; 3 were flagged. Samples tied at the cutoff (0.900) or within 0.05 of the median (1.000) stay labeled.",
    fixed = TRUE
  )
  expect_identical(which(noisy == -1L), c(4L, 42L, 85L))
})

test_that("NaN scores are noise and stay out of the count", {
  scores <- rep(1, 90)
  scores[c(4L, 42L, 85L)] <- 0.5
  scores[c(11L, 26L, 36L, 51L, 66L, 81L)] <- 0.9
  scores[c(21L, 61L)] <- NaN
  noisy_fit <- with_scores(fit, "gini", config_at(fit, 3), scores)
  noisy <- NULL
  expect_warning(
    noisy <- get_labels(noisy_fit, k = 3, noise_labels = TRUE),
    "noise_quantile=0.05 asks for about 5 of 88 samples by gini; 3 were flagged.",
    fixed = TRUE
  )
  expect_identical(which(noisy == -1L), c(4L, 21L, 42L, 61L, 85L))
})

test_that("the noise settings are checked on every call", {
  for (flag in c(FALSE, TRUE)) {
    expect_error(
      get_labels(fit, noise_labels = flag, noise_score = "nope"),
      "noise_score must be one of: 'gini', 'ce', 'accuracy'.",
      fixed = TRUE
    )
    for (quantile in c(0, 1, 1.5)) {
      expect_error(
        get_labels(fit, noise_labels = flag, noise_quantile = quantile),
        "noise_quantile must be strictly between 0 and 1, got",
        fixed = TRUE
      )
    }
  }
})

test_that("noise needs the score it ranks by", {
  gen <- single_mode_fit("generalizability")
  expect_error(
    get_labels(gen, measure = "generalizability", rule = "max", mode = "generalizability",
               noise_labels = TRUE, noise_score = "gini"),
    "Gini stability scores are not available for this run.",
    fixed = TRUE
  )
  stab <- single_mode_fit("stability")
  expect_error(
    get_labels(stab, noise_labels = TRUE, noise_score = "accuracy"),
    "Generalizability scores are not available for this run.",
    fixed = TRUE
  )
})

two <- make_blobs(n_per = 30L, centers = rbind(c(0, 0), c(6, 0)), seed = 4L)
# The anchored warning is tested in test-carve.R.
anchored <- suppressWarnings(carve(
  two$X, n_resamples = 6, random_state = 0, anchor_threshold = 30,
  estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3))
))

test_that("anchored labels cover every sample and recover the groups", {
  labels <- get_labels(anchored, k = 2)
  expect_length(labels, 60L)
  expect_gt(adjusted_rand_index(labels, two$y), 0.9)
  expect_identical(length(unique(get_labels(anchored, k = 3))), 3L)
  expect_length(get_labels(anchored, k = 2, mode = "generalizability"), 60L)
})

test_that("anchors keep the labels of the cut", {
  # A constant classifier cannot reproduce an alternating cut, so labels
  # that the classifier overwrote would show.
  constant <- anchored
  constant@run_params$classifier <- function(x_train, y_train, x_test) rep(1L, nrow(x_test))
  anchors <- consensus_anchors(constant)
  planted <- rep_len(1:2, length(anchors))
  extended <- extend_anchor_labels(constant, planted)
  expect_identical(extended[anchors], planted)
  expect_true(all(extended[-anchors] == 1L))
})

test_that("a one-cluster cut labels every sample with that cluster", {
  expect_identical(extend_anchor_labels(anchored, rep(2L, 30L)), rep(2L, 60L))
})

test_that("anchored labels are reproducible and use the fit's seed and budget", {
  set.seed(2)
  before <- .Random.seed
  first <- get_labels(anchored, k = 2)
  expect_identical(.Random.seed, before)
  expect_identical(first, get_labels(anchored, k = 2))
  seen <- new.env()
  spy <- anchored
  spy@run_params$classifier <- function(x_train, y_train, x_test, random_state, n_threads) {
    seen$seed <- random_state
    seen$threads <- n_threads
    rep(y_train[1], nrow(x_test))
  }
  spy@run_params$random_state <- 7L
  spy@run_params$n_threads <- 5L
  extend_anchor_labels(spy, rep_len(1:2, 30L))
  expect_identical(seen$seed, 7L)
  expect_identical(seen$threads, 5L)
})

test_that("an anchored run flags samples outside the anchors as noise", {
  low <- setdiff(seq_len(60L), consensus_anchors(anchored))[1:3]
  noisy_fit <- with_scores(anchored, "gini", config_at(anchored, 2), known_scores(60L, low))
  expect_identical(which(get_labels(noisy_fit, k = 2, noise_labels = TRUE) == -1L), low)
})

test_that("an exact fit has no anchors and no preprocessing results", {
  expect_null(consensus_anchors(fit))
  expect_null(preprocessing_results(fit))
  expect_null(preprocessing_pipelines(fit))
  expect_identical(consensus_anchors(anchored), anchored@consensus_anchors)
})

test_that("a randomized fit returns its table and pipelines", {
  randomized <- carve(
    blobs$X, n_resamples = 4, random_state = 0, estimator_param_grids = k_grid,
    randomize_preprocessing = TRUE,
    normalization_options = list(preprocessing_option(Identity)),
    dim_reduction_options = list(preprocessing_option(Identity), preprocessing_option(PCA, n_components = 1L))
  )
  expect_identical(preprocessing_results(randomized), randomized@preprocessing_results)
  expect_setequal(names(preprocessing_pipelines(randomized)), c("identity | identity", "identity | PCA(n_components=1)"))
  spec <- preprocessing_pipelines(randomized)[["identity | PCA(n_components=1)"]]
  expect_identical(dim(pipeline_from_spec(spec, 0L)(blobs$X)), c(90L, 1L))
})
