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
