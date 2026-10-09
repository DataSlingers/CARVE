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

test_that("PAC uses tau and counts only values strictly between tau and 1 - tau", {
  # Six off-diagonal pairs: 0.15, 0.85, 0.2, 0.8, 0.5 and 1.
  # tau = 0.1: 0.15, 0.85, 0.2, 0.8 and 0.5 are ambiguous, 5 of 6, PAC = 1/6.
  # tau = 0.2: only 0.5 is ambiguous; 0.2 and 0.8 sit on the bounds and are
  # excluded by the strict inequalities, so PAC = 5/6.
  M <- diag(4)
  pairs <- list(c(1, 2, 0.15), c(1, 3, 0.85), c(1, 4, 0.2), c(2, 3, 0.8), c(2, 4, 0.5), c(3, 4, 1))
  for (p in pairs) {
    M[p[1], p[2]] <- p[3]
    M[p[2], p[1]] <- p[3]
  }
  expect_equal(compute_consensus_pac(M, tau = 0.1), 1 / 6, tolerance = 1e-12)
  expect_equal(compute_consensus_pac(M, tau = 0.2), 5 / 6, tolerance = 1e-12)
})

anchored_fixture <- function() {
  f <- read_fixture("anchored")
  list(
    n = as.integer(f$n),
    runs = lapply(f$runs, function(r) {
      list(indices = as.integer(r$indices) + 1L, labels = as.integer(r$labels))
    }),
    anchors = as.integer(f$anchors) + 1L,
    block = fixture_matrix(f$block),
    gini = fixture_vector(f$gini),
    ce = fixture_vector(f$ce)
  )
}

test_that("the anchor block matches Python", {
  f <- anchored_fixture()
  expect_equal(nan_to_na(consensus_anchor_block(f$n, f$runs, f$anchors)), f$block, tolerance = 1e-6)
})

test_that("the anchor block is the exact matrix restricted to the anchors", {
  f <- anchored_fixture()
  full <- compute_consensus_matrix(f$n, f$runs)
  expect_equal(consensus_anchor_block(f$n, f$runs, f$anchors), full[f$anchors, f$anchors], tolerance = 1e-12)
})

test_that("anchored stability matches Python with and without chunks", {
  f <- anchored_fixture()
  for (chunk in list(NULL, 4L)) {
    scores <- stability_from_runs_anchored(f$n, f$runs, f$anchors, chunk_size = chunk)
    expect_equal(nan_to_na(scores$gini), f$gini, tolerance = 1e-6)
    expect_equal(nan_to_na(scores$ce), f$ce, tolerance = 1e-6)
  }
})

test_that("with every sample an anchor, anchored stability equals the exact scores", {
  f <- anchored_fixture()
  exact <- stability_from_consensus(compute_consensus_matrix(f$n, f$runs))
  anchored <- stability_from_runs_anchored(f$n, f$runs, seq_len(f$n), chunk_size = 4L)
  expect_equal(anchored, exact, tolerance = 1e-12)
})

test_that("the chunk size does not change the scores", {
  f <- anchored_fixture()
  one <- stability_from_runs_anchored(f$n, f$runs, f$anchors, chunk_size = 1L)
  for (chunk in c(3L, 100L)) {
    expect_equal(stability_from_runs_anchored(f$n, f$runs, f$anchors, chunk_size = chunk), one, tolerance = 1e-12)
  }
})

test_that("unsorted run indices give the same scores as sorted ones", {
  f <- anchored_fixture()
  sorted <- lapply(f$runs, function(r) {
    o <- order(r$indices)
    list(indices = r$indices[o], labels = r$labels[o])
  })
  expect_equal(
    stability_from_runs_anchored(f$n, sorted, f$anchors),
    stability_from_runs_anchored(f$n, f$runs, f$anchors),
    tolerance = 1e-12
  )
})

test_that("a sample never drawn with an anchor scores NaN without a warning", {
  f <- anchored_fixture()
  expect_no_warning(scores <- stability_from_runs_anchored(f$n, f$runs, f$anchors))
  expect_true(is.nan(scores$gini[15]))
  expect_true(is.nan(scores$ce[15]))
})

test_that("a run without stability runs gets an all-NaN block", {
  block <- consensus_anchor_block(10L, list(), c(2L, 5L))
  expect_true(all(is.nan(block)))
  expect_identical(dim(block), c(2L, 2L))
})

test_that("the default chunk size keeps a chunk near 2^23 entries", {
  expect_identical(default_anchor_chunk_size(1000L), 8192L)
  expect_identical(default_anchor_chunk_size(5000L), 1677L)
  expect_identical(default_anchor_chunk_size(100000L), 256L)
})
