norm_options <- list(preprocessing_option(Identity), preprocessing_option(StandardScaler))
dr_options <- list(preprocessing_option(Identity), preprocessing_option(PCA, n_components = c(1L, 2L)))

test_that("step labels match Python's", {
  f <- read_fixture("pipeline_labels")
  for (case in f$steps) {
    expect_identical(step_label(case$name, case$params), case$label, info = case$label)
  }
  spec <- new_pipeline_spec(
    list(transform = Identity, name = "identity", params = list(), label = step_label("identity", list())),
    list(transform = PCA, name = "PCA", params = list(n_components = 10L), label = step_label("PCA", list(n_components = 10L)))
  )
  expect_identical(spec$label, f$spec$label)
})

test_that("an option is named after its transform", {
  expect_identical(preprocessing_option(Identity)$name, "identity")
  expect_identical(preprocessing_option(Log1p)$name, "log1p")
  expect_identical(preprocessing_option(StandardScaler)$name, "StandardScaler")
  expect_identical(preprocessing_option(CARVE::PCA, n_components = 2L)$name, "PCA")
  expect_identical(preprocessing_option(function(X) X, name = "mine")$name, "mine")
  expect_identical(names(preprocessing_option(TSNE, perplexity = 30, n_components = 2L)$grid), c("perplexity", "n_components"))
})

test_that("an option rejects what the transform cannot take", {
  expect_error(preprocessing_option(function(X) X), "Pass name= for a transform that is not a named function.", fixed = TRUE)
  expect_error(preprocessing_option(PCA, components = 2), "PCA has no argument 'components'.", fixed = TRUE)
  expect_error(preprocessing_option(PCA, n_components = integer(0)), "The candidate list for 'n_components' of PCA is empty.", fixed = TRUE)
  expect_error(preprocessing_option("PCA"), "transform must be a function.", fixed = TRUE)
})

test_that("every resample gets one pipeline, and pairs differ in count by at most one", {
  specs <- allocate_pipelines(norm_options, dr_options, 10L, 0L)
  expect_length(specs, 10L)
  expect_true(all(vapply(specs, inherits, logical(1), "carve_pipeline_spec")))
  pairs <- vapply(specs, function(s) paste(s$normalization$name, s$dim_reduction$name), character(1))
  counts <- table(factor(pairs, levels = c("identity identity", "identity PCA", "StandardScaler identity", "StandardScaler PCA")))
  expect_true(all(counts %in% c(2L, 3L)))
})

test_that("the allocation is shuffled, not cycled", {
  specs <- allocate_pipelines(norm_options, dr_options, 12L, 0L)
  pairs <- vapply(specs, function(s) paste(s$normalization$name, s$dim_reduction$name), character(1))
  cycled <- rep(c("identity identity", "identity PCA", "StandardScaler identity", "StandardScaler PCA"), 3)
  expect_false(identical(pairs, cycled))
})

test_that("hyperparameters are drawn from the candidate lists", {
  specs <- allocate_pipelines(norm_options, dr_options, 40L, 0L)
  drawn <- unlist(lapply(specs, function(s) s$dim_reduction$params$n_components))
  expect_setequal(drawn, c(1L, 2L))
  labels <- vapply(specs, function(s) s$label, character(1))
  expect_true("identity | PCA(n_components=1)" %in% labels)
  expect_true("StandardScaler | PCA(n_components=2)" %in% labels)
})

test_that("the allocation depends on the seed and works for one resample", {
  labels <- function(seed, n) vapply(allocate_pipelines(norm_options, dr_options, n, seed), function(s) s$label, character(1))
  expect_identical(labels(3L, 20L), labels(3L, 20L))
  expect_false(identical(labels(3L, 20L), labels(4L, 20L)))
  expect_length(labels(0L, 1L), 1L)
  set.seed(9)
  before <- .Random.seed
  labels(0L, 5L)
  expect_identical(.Random.seed, before)
})

test_that("allocation needs options on both sides and distinct labels", {
  expect_error(
    allocate_pipelines(list(), dr_options, 4L, 0L),
    "randomize_preprocessing=TRUE needs at least one normalization option and one dimensionality reduction option.",
    fixed = TRUE
  )
  twin <- list(preprocessing_option(function(X) X, name = "same"), preprocessing_option(function(X) X * 2, name = "same"))
  expect_error(
    allocate_pipelines(twin, list(preprocessing_option(Identity)), 8L, 0L),
    "Two different pipelines render as the same label 'same | identity'. Give the options distinct names with preprocessing_option(name = ).",
    fixed = TRUE
  )
})

test_that("a pipeline applies its normalization, then its reduction, with the seed", {
  seen <- new.env()
  seen$calls <- character()
  first <- function(X, random_state = NULL) {
    seen$calls <- c(seen$calls, paste("first", random_state))
    X * 2
  }
  second <- function(X, width = 1L) {
    seen$calls <- c(seen$calls, "second")
    X[, seq_len(width), drop = FALSE]
  }
  spec <- allocate_pipelines(
    list(preprocessing_option(first, random_state = 99L)),
    list(preprocessing_option(second, width = 1L)),
    1L, 0L
  )[[1L]]
  out <- pipeline_from_spec(spec, 7L)(matrix(1:6, 3))
  expect_identical(out, matrix(c(2, 4, 6), ncol = 1L))
  # CARVE's seed replaces the one drawn from the grid; a step without
  # random_state is not given one.
  expect_identical(seen$calls, c("first 7", "second"))
})

test_that("a pipeline is reproducible and leaves the session's stream alone", {
  noisy <- function(X, random_state = NULL) X + stats::runif(length(X))
  spec <- allocate_pipelines(list(preprocessing_option(Identity)), list(preprocessing_option(noisy)), 1L, 0L)[[1L]]
  set.seed(4)
  before <- .Random.seed
  a <- pipeline_from_spec(spec, 3L)(matrix(0, 4, 2))
  expect_identical(.Random.seed, before)
  expect_identical(a, pipeline_from_spec(spec, 3L)(matrix(0, 4, 2)))
  expect_false(identical(a, pipeline_from_spec(spec, 4L)(matrix(0, 4, 2))))
})

test_that("a step that returns the wrong shape is an error", {
  drop_row <- function(X) X[-1, , drop = FALSE]
  spec <- allocate_pipelines(list(preprocessing_option(Identity)), list(preprocessing_option(drop_row)), 1L, 0L)[[1L]]
  expect_error(
    pipeline_from_spec(spec, 0L)(matrix(0, 4, 2)),
    "A preprocessing function must return a numeric matrix with one row per row of X (4).",
    fixed = TRUE
  )
})

test_that("options and pipelines print their names", {
  expect_output(print(preprocessing_option(PCA, n_components = c(2L, 5L))), "Preprocessing option PCA", fixed = TRUE)
  expect_output(print(preprocessing_option(PCA, n_components = c(2L, 5L))), "n_components: 2, 5", fixed = TRUE)
  spec <- allocate_pipelines(norm_options[1L], dr_options[1L], 1L, 0L)[[1L]]
  expect_output(print(spec), "Preprocessing pipeline: identity | identity", fixed = TRUE)
})
