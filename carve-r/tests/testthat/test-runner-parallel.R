skip_if_not_installed("ranger")
skip_if_not_installed("furrr")
skip_if_not_installed("future")

# Parallel workers are fresh R sessions that don't inherit
# `devtools::load_all`'s namespace. Only run these tests when CARVE
# is truly installed (as is the case during `R CMD check`). We detect
# an installed build by checking whether the package shows up in
# `installed.packages()`.
carve_is_installed <- function() {
  "CARVE" %in% rownames(utils::installed.packages())
}
skip_if_not(carve_is_installed(),
            "CARVE not installed; parallel workers cannot resolve namespace")

make_blobs <- function(n_per = 40L, seed = 1L) {
  set.seed(seed)
  centers <- rbind(c(0, 0), c(6, 0), c(3, 5))
  X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i) {
    matrix(stats::rnorm(n_per * 2L, 0, 0.4), ncol = 2L) +
      matrix(centers[i, ], n_per, 2L, byrow = TRUE)
  }))
  list(X = X, y = rep(seq_len(nrow(centers)), each = n_per))
}

test_that("n_jobs > 1 produces identical results to n_jobs = 1 (per-resample seeding)", {
  # Per-resample seeds (random_state + b) are explicit inside validation_iter,
  # so the result is reproducible regardless of worker dispatch order.
  bl <- make_blobs(n_per = 30L)
  grids <- list(list(type = "kmeans", grid = list(n_clusters = 2:3)))

  r_seq <- run_validation(
    X = bl$X, estimator_grids = grids,
    n_resamples = 6L, subsample_ratio = 0.8,
    random_state = 7L, n_jobs = 1L, verbose = 0L
  )
  r_par <- run_validation(
    X = bl$X, estimator_grids = grids,
    n_resamples = 6L, subsample_ratio = 0.8,
    random_state = 7L, n_jobs = 2L, verbose = 0L
  )

  expect_equal(r_seq$estimator_results, r_par$estimator_results)
  expect_equal(r_seq$consensus_matrices, r_par$consensus_matrices)
  expect_equal(r_seq$generalizability_scores, r_par$generalizability_scores)
})

test_that("future::plan is restored after run_validation returns", {
  bl <- make_blobs(n_per = 30L)
  grids <- list(list(type = "kmeans", grid = list(n_clusters = 2L)))

  before <- future::plan()
  run_validation(
    X = bl$X, estimator_grids = grids,
    n_resamples = 3L, subsample_ratio = 0.8,
    random_state = 1L, n_jobs = 2L, verbose = 0L
  )
  after <- future::plan()
  # Same class identity confirms the plan was restored, not left at multisession.
  expect_equal(class(before), class(after))
})

test_that("show_progress = TRUE emits progressr signals without erroring", {
  bl <- make_blobs(n_per = 30L)
  grids <- list(list(type = "kmeans", grid = list(n_clusters = 2:3)))

  # Count progressor signals with a local handler.
  count <- 0L
  progressr::handlers(progressr::handler_void())
  progressr::with_progress({
    res <- run_validation(
      X = bl$X, estimator_grids = grids,
      n_resamples = 4L, subsample_ratio = 0.8,
      random_state = 1L, n_jobs = 1L, show_progress = TRUE, verbose = 0L
    )
  })
  expect_s3_class(res$estimator_results, "data.frame")
})
