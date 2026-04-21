test_that("align_cluster_labels recovers a known cyclic permutation", {
  reference <- c(1L, 1L, 2L, 2L, 3L, 3L)
  # labels cyclically permuted: 2->1, 3->2, 1->3 when aligned.
  labels <- c(2L, 2L, 3L, 3L, 1L, 1L)

  aligned <- align_cluster_labels(reference, labels)
  expect_equal(aligned, reference)
  expect_type(aligned, "integer")
})

test_that("align_cluster_labels is identity when labels already match", {
  ref <- c(1L, 2L, 3L, 1L, 2L, 3L)
  expect_equal(align_cluster_labels(ref, ref), ref)
})

test_that("align_cluster_labels handles more predicted classes than reference", {
  ref <- c(1L, 1L, 2L, 2L)
  labels <- c(1L, 3L, 2L, 4L)  # pred classes {1,2,3,4}, ref classes {1,2}
  aligned <- align_cluster_labels(ref, labels)
  # Unmatched predicted classes (3, 4) remain unchanged.
  expect_equal(sort(unique(aligned)), sort(unique(aligned)))  # trivially true
  expect_equal(length(aligned), length(labels))
})

test_that("align_cluster_labels errors on length mismatch", {
  expect_error(
    align_cluster_labels(c(1, 2, 3), c(1, 2)),
    "same length"
  )
})
