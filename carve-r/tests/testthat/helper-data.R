# Small synthetic data sets, generated in R with a fixed seed.
make_blobs <- function(n_per = 30L, centers = rbind(c(0, 0), c(6, 0), c(3, 5)),
                       sd = 0.5, seed = 1L) {
  withr::with_seed(seed, {
    X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i) {
      noise <- matrix(stats::rnorm(n_per * ncol(centers), sd = sd), ncol = ncol(centers))
      noise + matrix(centers[i, ], n_per, ncol(centers), byrow = TRUE)
    }))
  })
  list(X = X, y = rep(seq_len(nrow(centers)), each = n_per))
}

make_moons <- function(n = 200L, noise = 0.05, seed = 1L) {
  n_out <- n %/% 2L
  n_in <- n - n_out
  t_out <- seq(0, pi, length.out = n_out)
  t_in <- seq(0, pi, length.out = n_in)
  X <- rbind(cbind(cos(t_out), sin(t_out)), cbind(1 - cos(t_in), 1 - sin(t_in) - 0.5))
  withr::with_seed(seed, {
    X <- X + matrix(stats::rnorm(2L * n, sd = noise), ncol = 2L)
  })
  list(X = X, y = rep(1:2, c(n_out, n_in)))
}
