test_that("compute_generalizability_scores matches Python", {
  fx <- read_fixture("accuracy")
  runs <- lapply(fx$runs, function(r) {
    list(indices = as.integer(r$indices) + 1L, true = as.integer(r$true), predicted = as.integer(r$predicted))
  })
  expect_equal(
    compute_generalizability_scores(fx$n_samples, runs),
    fixture_vector(fx$scores),
    tolerance = 1e-12
  )
})

test_that("predictions are aligned before scoring and unseen samples score 0", {
  runs <- list(list(indices = 1:2, true = c(1L, 2L), predicted = c(2L, 1L)))
  expect_identical(compute_generalizability_scores(3L, runs), c(1, 1, 0))
})
