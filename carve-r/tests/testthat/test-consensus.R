fixture_runs <- function(runs) {
  lapply(runs, function(r) list(indices = as.integer(r$indices) + 1L, labels = as.integer(r$labels)))
}

test_that("compute_consensus_matrix matches Python", {
  fx <- read_fixture("consensus")
  M <- compute_consensus_matrix(fx$n_samples, fixture_runs(fx$runs))
  expect_equal(nan_to_na(M), fixture_matrix(fx$consensus), tolerance = 1e-6)
})

test_that("stability scores and PAC match Python", {
  fx <- read_fixture("consensus")
  M <- compute_consensus_matrix(fx$n_samples, fixture_runs(fx$runs))
  s <- stability_from_consensus(M)
  expect_equal(nan_to_na(s$gini), fixture_vector(fx$gini), tolerance = 1e-6)
  expect_equal(nan_to_na(s$ce), fixture_vector(fx$ce), tolerance = 1e-6)
  expect_equal(compute_consensus_pac(M), fx$pac_005, tolerance = 1e-6)
  expect_equal(compute_consensus_pac(M, tau = 0.1), fx$pac_010, tolerance = 1e-6)
})

test_that("pairs never drawn together are NaN and so is a sample never drawn", {
  runs <- list(
    list(indices = 1:2, labels = c(1L, 1L)),
    list(indices = 2:3, labels = c(1L, 2L))
  )
  M <- compute_consensus_matrix(4L, runs)
  expect_true(is.nan(M[1, 3]))
  expect_true(all(is.nan(M[4, ])))
  expect_identical(M[1, 2], 1)
  expect_identical(M[2, 3], 0)
  expect_identical(diag(M)[1:3], c(1, 1, 1))
  expect_true(is.na(stability_from_consensus(M)$gini[4]))
})

test_that("perfect consensus scores 1 everywhere", {
  M <- matrix(c(1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1), 4)
  s <- stability_from_consensus(M)
  expect_equal(s$gini, rep(1, 4))
  expect_equal(s$ce, rep(1, 4), tolerance = 1e-9)
  expect_identical(compute_consensus_pac(M), 1)
})

test_that("PAC is NaN when no pair was drawn together", {
  expect_true(is.nan(compute_consensus_pac(matrix(NaN, 3, 3))))
})
