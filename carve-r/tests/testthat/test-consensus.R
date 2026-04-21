# Exact pins for consensus-matrix math. The values below were hand-computed
# from the definition and match the Python reference implementation.

test_that("compute_consensus_matrix matches the hand-computed 4-sample example", {
  # Run 1: samples {1,2,3}, labels {1,1,2}  -> {1,2} co-cluster, 3 alone
  # Run 2: samples {2,3,4}, labels {1,2,2}  -> {3,4} co-cluster, 2 alone
  runs <- list(
    list(sample_idx = c(1L, 2L, 3L), labels = c(1L, 1L, 2L)),
    list(sample_idx = c(2L, 3L, 4L), labels = c(1L, 2L, 2L))
  )

  m <- compute_consensus_matrix(n_samples = 4L, runs = runs)

  expected <- matrix(
    c(
      1, 1, 0, NaN,
      1, 1, 0, 0,
      0, 0, 1, 1,
      NaN, 0, 1, 1
    ),
    nrow = 4L, byrow = TRUE
  )

  expect_true(all(is.nan(m) == is.nan(expected)))
  # Compare the non-NaN entries exactly.
  mask <- !is.nan(expected)
  expect_equal(m[mask], expected[mask])
})

test_that("compute_consensus_matrix can return raw counts", {
  runs <- list(
    list(sample_idx = c(1L, 2L, 3L), labels = c(1L, 1L, 2L)),
    list(sample_idx = c(2L, 3L, 4L), labels = c(1L, 2L, 2L))
  )
  res <- compute_consensus_matrix(4L, runs, return_counts = TRUE)

  expect_named(res, c("consensus_matrix", "co_cluster_counts", "co_sample_counts"))
  expect_equal(res$co_sample_counts[1, ], c(1, 1, 1, 0))
  expect_equal(res$co_sample_counts[4, ], c(0, 1, 1, 1))
  expect_equal(res$co_cluster_counts[2, 2], 2)
  expect_equal(res$co_cluster_counts[3, 4], 1)
})

test_that("stability_from_consensus is 1 on a perfect consensus", {
  # All off-diagonal values are either 0 or 1 -> perfect stability.
  cm <- matrix(c(
    1, 1, 0, 0,
    1, 1, 0, 0,
    0, 0, 1, 1,
    0, 0, 1, 1
  ), 4L, 4L, byrow = TRUE)

  s <- stability_from_consensus(cm)
  expect_equal(unname(s$stability_gini), rep(1, 4L), tolerance = 1e-10)
  # CE has a tiny floor due to epsilon-clipping at p=0 or p=1; still ~1.
  expect_true(all(s$stability_ce > 1 - 1e-8))
})

test_that("stability_from_consensus is 0 when all off-diagonal values are 0.5", {
  cm <- matrix(0.5, 4L, 4L)
  diag(cm) <- 1  # diag is ignored in the computation anyway

  s <- stability_from_consensus(cm)
  # p*(1-p) = 0.25, 2 * 0.25 = 0.5, stability = 1 - clip(2*0.5,0,1) = 0
  expect_equal(unname(s$stability_gini), rep(0, 4L), tolerance = 1e-10)
  # Binary entropy at 0.5 = log(2); stability = 1 - clip(log(2)/log(2)) = 0
  expect_equal(unname(s$stability_ce), rep(0, 4L), tolerance = 1e-10)
})

test_that("compute_consensus_pac returns 1 on unambiguous consensus", {
  cm <- matrix(c(
    1, 1, 0, 0,
    1, 1, 0, 0,
    0, 0, 1, 1,
    0, 0, 1, 1
  ), 4L, 4L, byrow = TRUE)
  expect_equal(compute_consensus_pac(cm), 1)
})

test_that("compute_consensus_pac returns 0 when all off-diagonal values are ambiguous", {
  cm <- matrix(0.5, 4L, 4L)
  diag(cm) <- 1
  expect_equal(compute_consensus_pac(cm), 0)
})

test_that("compute_consensus_pac respects tau threshold", {
  # 6 off-diagonal entries (3x3 symmetric minus diag = 6).
  # Values: 0.02, 0.5, 0.98 -> with tau=0.05, only 0.5 is ambiguous (0.5 in (0.05,0.95)).
  cm <- matrix(c(
    1, 0.02, 0.5,
    0.02, 1, 0.98,
    0.5, 0.98, 1
  ), 3L, 3L, byrow = TRUE)
  # 6 off-diagonal entries; 2 are 0.5 (ambiguous pair i<->j, j<->i).
  # 0.02 appears twice (0.02 < tau so not ambiguous at tau=0.05).
  # 0.98 appears twice (0.98 > 1-tau so not ambiguous at tau=0.05).
  expect_equal(compute_consensus_pac(cm, tau = 0.05), 1 - (2 / 6))

  # At tau = 0.01, 0.02 and 0.98 also become ambiguous (2 more each side) -> 6/6
  expect_equal(compute_consensus_pac(cm, tau = 0.01), 0)
})

test_that("compute_consensus_pac returns NaN when no valid off-diagonal pairs exist", {
  cm <- matrix(NaN, 3L, 3L)
  expect_true(is.nan(compute_consensus_pac(cm)))
})

test_that("compute_consensus_metrics aggregates across matrices", {
  cm1 <- matrix(c(
    1, 1, 0, 0,
    1, 1, 0, 0,
    0, 0, 1, 1,
    0, 0, 1, 1
  ), 4L, 4L, byrow = TRUE)
  cm2 <- matrix(0.5, 4L, 4L); diag(cm2) <- 1

  res <- compute_consensus_metrics(list(cm1, cm2))
  expect_length(res$gini_list, 2L)
  expect_length(res$ce_list, 2L)
  expect_equal(res$pac_list, c(1, 0))
})

test_that("reorder_consensus_matrix groups co-clustered samples", {
  cm <- matrix(c(
    1, 1, 0, 0,
    1, 1, 0, 0,
    0, 0, 1, 1,
    0, 0, 1, 1
  ), 4L, 4L, byrow = TRUE)

  out <- reorder_consensus_matrix(cm)
  expect_named(out, c("reordered", "order"))
  # The two clusters {1,2} and {3,4} should appear as contiguous blocks.
  # After hclust on 1-cm, the order is either (1,2,3,4) or (3,4,1,2) etc.
  ord <- out$order
  # Adjacent pairs in the ordering must all have high consensus.
  expect_true(all(cm[cbind(ord[-4], ord[-1])] %in% c(0, 1)))
  # The reordered block structure: top-left 2x2 and bottom-right 2x2 are 1s.
  expect_equal(out$reordered[1:2, 1:2], matrix(1, 2, 2))
  expect_equal(out$reordered[3:4, 3:4], matrix(1, 2, 2))
})
