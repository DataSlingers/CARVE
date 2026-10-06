test_that("seeded() fixes the draw and restores the caller's RNG state", {
  set.seed(42)
  before <- .Random.seed
  a <- seeded(7L, stats::runif(3))
  expect_identical(.Random.seed, before)
  expect_identical(a, seeded(7L, stats::runif(3)))
})

test_that("seeded(NULL) draws from the caller's stream", {
  set.seed(1)
  a <- seeded(NULL, stats::runif(1))
  set.seed(1)
  expect_identical(a, stats::runif(1))
})

test_that("seeded() gives the same draw whatever the caller's RNG kind", {
  a <- seeded(3L, sample.int(100L, 5L))
  old <- RNGkind("L'Ecuyer-CMRG")
  withr::defer(RNGkind(old[1]))
  expect_identical(seeded(3L, sample.int(100L, 5L)), a)
})

test_that("split_subsample_indices draws floor(ratio * n) rows and holds out the rest", {
  s <- split_subsample_indices(100L, 0.618, random_state = 1L)
  expect_length(s$train, 61L)
  expect_length(s$test, 39L)
  expect_setequal(c(s$train, s$test), 1:100)
  expect_length(intersect(s$train, s$test), 0L)
  expect_false(is.unsorted(s$test))
})

test_that("split_subsample_indices is reproducible and depends on the seed", {
  expect_identical(split_subsample_indices(50L, 0.5, 3L), split_subsample_indices(50L, 0.5, 3L))
  expect_false(identical(
    split_subsample_indices(50L, 0.5, 3L)$train,
    split_subsample_indices(50L, 0.5, 4L)$train
  ))
})

test_that("count_clusters counts distinct non-negative labels", {
  expect_identical(count_clusters(c(1L, 1L, 2L, 3L)), 3L)
  expect_identical(count_clusters(c(-1L, 0L, 0L, 4L)), 2L)
  expect_identical(count_clusters(c(-1L, -1L)), 0L)
  expect_identical(count_clusters(NULL), 0L)
  expect_identical(count_clusters(integer()), 0L)
})

test_that("a single number K expands to 2:K", {
  expect_identical(coerce_n_clusters(5), 2:5)
  expect_identical(coerce_n_clusters(2L), 2L)
})

test_that("a vector of cluster counts is kept in its own order", {
  expect_identical(coerce_n_clusters(c(5, 3, 2)), c(5L, 3L, 2L))
  expect_identical(coerce_n_clusters(3, expand_scalar = FALSE), 3L)
})

test_that("invalid cluster counts are rejected", {
  expect_error(coerce_n_clusters(1), "n_clusters int must be >= 2.", fixed = TRUE)
  expect_error(coerce_n_clusters(c(1, 3)), "All n_clusters values must be >= 2.", fixed = TRUE)
  expect_error(coerce_n_clusters(c(2.5, 3)), "n_clusters values must be whole numbers.", fixed = TRUE)
  expect_error(coerce_n_clusters("3"), "n_clusters must be a numeric vector.", fixed = TRUE)
  expect_error(coerce_n_clusters(matrix(2:5, 2)), "n_clusters must be a numeric vector.", fixed = TRUE)
})

test_that("summarize_ari_scores matches Python", {
  for (case in read_fixture("summarize_ari")$cases) {
    s <- summarize_ari_scores(fixture_vector(case$x))
    expected <- vapply(case[c("mean", "se", "upper", "lower")], fixture_number, numeric(1))
    expect_equal(nan_to_na(unname(s)), unname(expected), tolerance = 1e-12)
  }
})

test_that("as_data_matrix returns a double matrix for each supported input", {
  m <- matrix(1:6, 3)
  expect_identical(as_data_matrix(m), matrix(as.numeric(1:6), 3))
  expect_identical(
    as_data_matrix(data.frame(a = 1:3, b = c(4, 5, 6))),
    cbind(a = c(1, 2, 3), b = c(4, 5, 6))
  )
  expect_identical(dim(as_data_matrix(c(1, 2, 3))), c(3L, 1L))
  expect_equal(
    as_data_matrix(Matrix::Matrix(m, sparse = TRUE)),
    matrix(as.numeric(1:6), 3),
    ignore_attr = "dimnames"
  )
})

test_that("as_data_matrix rejects other inputs", {
  expect_error(
    as_data_matrix(data.frame(a = 1:2, b = c("x", "y"))),
    "X has non-numeric columns: b.",
    fixed = TRUE
  )
  expect_error(as_data_matrix(list(1, 2)), "X must be a numeric matrix", fixed = TRUE)
  expect_error(as_data_matrix(c(1, NA)), "X contains missing or infinite values.", fixed = TRUE)
})

test_that("densifying a large sparse matrix warns", {
  s <- Matrix::Matrix(diag(4), sparse = TRUE)
  expect_warning(
    as_data_matrix(s, dense_warn_elements = 10),
    "Densifying a sparse matrix with 16 entries",
    fixed = TRUE
  )
})

test_that("collect_warnings returns the value and the muffled warnings", {
  out <- collect_warnings({
    warning("first")
    warning("second")
    3
  })
  expect_identical(out, list(value = 3, warnings = c("first", "second")))
})

test_that("check_dots names arguments no method knows", {
  expect_silent(check_dots())
  expect_error(check_dots(k = 3), "Unknown argument: k.", fixed = TRUE)
  expect_error(check_dots(k = 3, 4), "Unknown arguments: k, <unnamed>.", fixed = TRUE)
})

test_that("format_param_value matches the Python method labels", {
  expect_identical(format_param_value(7L), "7")
  expect_identical(format_param_value(1), "1")
  expect_identical(format_param_value(0.25), "0.25")
  expect_identical(format_param_value(1e-5), "1e-05")
  expect_identical(format_param_value(1234.5), "1.23e+03")
  expect_identical(format_param_value(TRUE), "True")
  expect_identical(format_param_value("ward"), "ward")
})

test_that("format_repr, python_list and format_params follow Python's repr", {
  expect_identical(format_repr("x"), "'x'")
  expect_identical(format_repr(3L), "3")
  expect_identical(format_repr(0.5), "0.5")
  expect_identical(format_repr(FALSE), "False")
  expect_identical(format_repr(NULL), "None")
  expect_identical(python_list(c("a", "b")), "['a', 'b']")
  expect_identical(format_params(list(resolution = 0.5, linkage = "ward")), "{'resolution': 0.5, 'linkage': 'ward'}")
  expect_identical(format_params(list()), "{}")
  expect_identical(title_case("max eps"), "Max Eps")
})

test_that("adjusted_rand_index matches sklearn", {
  for (case in read_fixture("ari")$cases) {
    expect_equal(adjusted_rand_index(case$a, case$b), case$ari, tolerance = 1e-12)
  }
})

test_that("align_cluster_labels matches Python", {
  for (case in read_fixture("align")$cases) {
    expect_identical(
      align_cluster_labels(case$reference, case$labels),
      case$aligned,
      info = case$name
    )
  }
})

test_that("reference entries outside keep take no part in the matching", {
  reference <- c(-1L, -1L, 0L, 0L, 1L, 1L)
  labels <- c(1L, 1L, 2L, 2L, 3L, 3L)
  # Without keep, label 1 is matched onto -1. With it, label 1 has no
  # partner and takes the fresh id 2.
  expect_identical(align_cluster_labels(reference, labels)[1:2], c(-1L, -1L))
  expect_identical(
    align_cluster_labels(reference, labels, keep = reference >= 0),
    c(2L, 2L, 0L, 0L, 1L, 1L)
  )
})

test_that("resolve_core_budget matches Python", {
  for (case in read_fixture("core_budget")$cases) {
    local_mocked_bindings(n_cores = function() as.integer(case$cores))
    expect_identical(
      resolve_core_budget(case$n_jobs, case$n_resamples),
      c(outer = as.integer(case$outer), inner = as.integer(case$inner)),
      info = paste(case$cores, format_repr(case$n_jobs), case$n_resamples)
    )
  }
})

test_that("n_jobs = 0 is rejected", {
  expect_error(
    resolve_core_budget(0, 10L),
    "n_jobs == 0 has no meaning; use 1 or a negative count",
    fixed = TRUE
  )
})

test_that("n_cores honors R CMD check's core limit", {
  withr::local_envvar(c("_R_CHECK_LIMIT_CORES_" = "TRUE"))
  expect_lte(n_cores(), 2L)
})

test_that("scale_neighbor_count matches Python", {
  estimator <- function(X, n_neighbors = 7L, n_clusters = 2L) NULL
  for (case in read_fixture("scale_neighbor")$cases) {
    scaled <- scale_neighbor_count(estimator, case$params, case$n_fit, case$n_full)
    expect_equal(scaled$n_neighbors, case$n_neighbors)
  }
})

test_that("an estimator without n_neighbors passes through scaling", {
  estimator <- function(X, n_clusters = 2L) NULL
  expect_identical(
    scale_neighbor_count(estimator, list(n_clusters = 3L), 10, 100),
    list(n_clusters = 3L)
  )
})

test_that("call_estimator passes random_state only to estimators that take it", {
  with_seed_arg <- function(X, n_clusters, random_state = NULL) rep(random_state, nrow(X))
  without <- function(X, n_clusters) rep(n_clusters, nrow(X))
  X <- matrix(0, 3, 1)
  expect_identical(call_estimator(with_seed_arg, X, list(n_clusters = 2L), random_state = 9L), rep(9L, 3))
  expect_identical(call_estimator(without, X, list(n_clusters = 2L), random_state = 9L), rep(2L, 3))
})

test_that("call_estimator checks what the estimator returns", {
  X <- matrix(0, 3, 1)
  expect_error(
    call_estimator(function(X) 1:2, X, list()),
    "A clustering function must return one integer label per row of X (3).",
    fixed = TRUE
  )
  expect_error(
    call_estimator(function(X) c(1.5, 1, 1), X, list()),
    "A clustering function must return one integer label per row of X (3).",
    fixed = TRUE
  )
  expect_identical(call_estimator(function(X) factor(c("a", "b", "a")), X, list()), c(1L, 2L, 1L))
})

test_that("the default classifier predicts separable classes", {
  d <- make_blobs()
  train <- seq(1L, 90L, by = 2L)
  test <- seq(2L, 90L, by = 2L)
  predict_fn <- default_generalizability_classifier(NULL, n_features = 2L, n_trees = 50L, random_state = 1L, n_threads = 1L)
  expect_identical(predict_fn(d$X[train, ], d$y[train], d$X[test, ]), d$y[test])
})

test_that("the default classifier gives the same predictions at any thread count", {
  d <- make_blobs(sd = 2)
  train <- seq(1L, 90L, by = 2L)
  test <- seq(2L, 90L, by = 2L)
  one <- default_generalizability_classifier(NULL, 2L, 50L, random_state = 1L, n_threads = 1L)
  two <- default_generalizability_classifier(NULL, 2L, 50L, random_state = 1L, n_threads = 2L)
  expect_identical(
    one(d$X[train, ], d$y[train], d$X[test, ]),
    two(d$X[train, ], d$y[train], d$X[test, ])
  )
})

test_that("a single training class predicts that class", {
  predict_fn <- default_generalizability_classifier(NULL, 2L, 10L, 1L, 1L)
  expect_identical(predict_fn(matrix(0, 3, 2), c(4L, 4L, 4L), matrix(1, 2, 2)), c(4L, 4L))
})

test_that("a custom classifier gets n_threads and random_state when it takes them", {
  seen <- new.env()
  clf <- function(x_train, y_train, x_test, n_threads, random_state) {
    seen$args <- c(n_threads, random_state)
    rep(y_train[1], nrow(x_test))
  }
  predict_fn <- default_generalizability_classifier(clf, 2L, 100L, random_state = 5L, n_threads = 3L)
  expect_identical(predict_fn(matrix(0, 2, 2), c(1L, 2L), matrix(0, 4, 2)), rep(1L, 4))
  expect_identical(seen$args, c(3L, 5L))
  plain <- function(x_train, y_train, x_test) rep(y_train[1], nrow(x_test))
  plain_fn <- default_generalizability_classifier(plain, 2L, 100L, 5L, 3L)
  expect_identical(plain_fn(matrix(0, 2, 2), c(1L, 2L), matrix(0, 4, 2)), rep(1L, 4))
})

test_that("a custom classifier must return one label per test row", {
  bad <- function(x_train, y_train, x_test) 1L
  predict_fn <- default_generalizability_classifier(bad, 2L, 100L, 5L, 1L)
  expect_error(
    predict_fn(matrix(0, 2, 2), c(1L, 2L), matrix(0, 4, 2)),
    "A classifier must return one label per row of x_test (4).",
    fixed = TRUE
  )
})

test_that("the default classifier passes the forest settings to ranger", {
  seen <- NULL
  local_mocked_bindings(ranger_predict = function(x_train, y_train, x_test, n_trees,
                                                  mtry, max_depth, seed, n_threads) {
    seen <<- list(n_trees = n_trees, mtry = mtry, max_depth = max_depth,
                  seed = seed, n_threads = n_threads)
    rep(1L, nrow(x_test))
  })
  expected_mtry <- c("1" = 1L, "2" = 1L, "5" = 2L, "16" = 4L)
  for (p in c(1L, 2L, 5L, 16L)) {
    predict_fn <- default_generalizability_classifier(NULL, p, 37L, random_state = 11L, n_threads = 3L)
    predict_fn(matrix(0, 2, p), c(1L, 2L), matrix(0, 2, p))
    expect_identical(seen$mtry, expected_mtry[[as.character(p)]], info = p)
    expect_identical(seen$max_depth, p, info = p)
    expect_identical(seen$n_trees, 37L)
    expect_identical(seen$seed, 11L)
    expect_identical(seen$n_threads, 3L)
  }
})

test_that("the default classifier is deterministic at seed 0 and leaves the caller's RNG alone", {
  d <- make_blobs(sd = 3)
  train <- seq(1L, 90L, by = 2L)
  test <- seq(2L, 90L, by = 2L)
  predict_fn <- default_generalizability_classifier(NULL, 2L, 25L, random_state = 0L, n_threads = 1L)
  set.seed(5)
  before <- .Random.seed
  first <- predict_fn(d$X[train, ], d$y[train], d$X[test, ])
  expect_identical(.Random.seed, before)
  for (i in 1:4) {
    expect_identical(predict_fn(d$X[train, ], d$y[train], d$X[test, ]), first)
  }
})

test_that("vote ties in the default classifier do not depend on the session RNG", {
  versicolor_virginica <- which(iris$Species != "setosa")
  X <- as.matrix(iris[versicolor_virginica, 1:4])
  y <- as.integer(iris$Species[versicolor_virginica]) - 1L
  train <- seq(1L, 100L, by = 2L)
  test <- seq(2L, 100L, by = 2L)
  predict_fn <- default_generalizability_classifier(NULL, 4L, 2L, random_state = 2L, n_threads = 1L)
  predictions <- lapply(1:8, function(s) {
    set.seed(s)
    before <- .Random.seed
    out <- predict_fn(X[train, ], y[train], X[test, ])
    expect_identical(.Random.seed, before)
    out
  })
  for (p in predictions[-1]) expect_identical(p, predictions[[1]])
})

test_that("largest_probability_class returns the class of the largest probability", {
  probabilities <- matrix(
    c(0.2, 0.7, 0.1, 0.5, 0.3, 0.2, 0.1, 0.1, 0.8),
    nrow = 3, byrow = TRUE, dimnames = list(NULL, c("1", "2", "3"))
  )
  expect_identical(largest_probability_class(probabilities), c(2L, 1L, 3L))
})

test_that("largest_probability_class sends ties to the first class in sorted order", {
  probabilities <- matrix(
    c(0.25, 0.25, 0.5, 0.5, 0.5, 0),
    nrow = 2, byrow = TRUE, dimnames = list(NULL, c("3", "1", "2"))
  )
  # Columns arrive unsorted; the tie in row 2 is between classes 3 and 1.
  expect_identical(largest_probability_class(probabilities), c(2L, 1L))
  tied <- matrix(c(0.5, 0.5), nrow = 1, dimnames = list(NULL, c("7", "4")))
  expect_identical(largest_probability_class(tied), 4L)
})

test_that("the default forest grows probability trees with leaves of one sample", {
  seen <- NULL
  local_mocked_bindings(
    ranger = function(...) {
      seen <<- list(...)
      structure(list(), class = "fake_forest")
    },
    .package = "ranger"
  )
  local_mocked_bindings(
    predict = function(object, data, ...) {
      list(predictions = matrix(c(0.1, 0.9), nrow = nrow(data), ncol = 2, byrow = TRUE,
                                dimnames = list(NULL, c("1", "2"))))
    },
    .package = "stats"
  )
  out <- ranger_predict(
    matrix(c(0, 1, 2, 3), 4, 1), c(1L, 1L, 2L, 2L), matrix(0, 3, 1),
    n_trees = 7L, mtry = 1L, max_depth = 1L, seed = 4L, n_threads = 1L
  )
  expect_true(seen$probability)
  expect_identical(seen$min.node.size, 1L)
  expect_identical(seen$num.trees, 7L)
  expect_identical(seen$seed, 5L)
  expect_identical(out, rep(2L, 3))
})

test_that("a depth-limited forest predicts the largest mean probability, not the tree vote", {
  set.seed(1)
  angle <- runif(300, 0, 2 * pi)
  radius <- rep(c(1, 0.5), each = 150)
  X <- cbind(radius * cos(angle), radius * sin(angle)) + matrix(rnorm(600, sd = 0.05), ncol = 2)
  y <- rep(1:2, each = 150)
  test_rows <- seq(2L, 300L, by = 2L)
  train_rows <- seq(1L, 300L, by = 2L)
  soft <- ranger_predict(
    X[train_rows, ], y[train_rows], X[test_rows, ],
    n_trees = 100L, mtry = 1L, max_depth = 2L, seed = 0L, n_threads = 1L
  )
  colnames(X) <- c("x1", "x2")
  forest <- ranger::ranger(
    x = X[train_rows, ], y = factor(y[train_rows]), num.trees = 100L, mtry = 1L,
    max.depth = 2L, min.node.size = 1L, probability = TRUE, seed = 1L, num.threads = 1L
  )
  probabilities <- predict(forest, data = X[test_rows, ], seed = 1L, num.threads = 1L)$predictions
  expect_identical(soft, as.integer(colnames(probabilities)[max.col(probabilities, ties.method = "first")]))
  hard <- ranger::ranger(
    x = X[train_rows, ], y = factor(y[train_rows]), num.trees = 100L, mtry = 1L,
    max.depth = 2L, seed = 1L, num.threads = 1L
  )
  tree_vote <- as.integer(as.character(predict(hard, data = X[test_rows, ], seed = 1L, num.threads = 1L)$predictions))
  expect_gt(sum(soft != tree_vote), 0L)
})
