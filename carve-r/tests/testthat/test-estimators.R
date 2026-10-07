test_that("kmeans_lloyd reproduces sklearn from the same starting centers", {
  fx <- read_fixture("kmeans")
  X <- fixture_matrix(fx$X)
  # sklearn centers the data before iterating and adds the mean back.
  center <- colMeans(X)
  Xc <- sweep(X, 2L, center)
  tol <- mean(colMeans(Xc^2)) * 1e-4
  for (case in fx$cases) {
    init <- sweep(fixture_matrix(case$init), 2L, center)
    fit <- kmeans_lloyd(Xc, init, max_iter = 300L, tol = tol)
    expect_identical(fit$labels, as.integer(case$labels) + 1L, info = case$name)
    expect_equal(sweep(fit$centers, 2L, center, "+"), fixture_matrix(case$centers), tolerance = 1e-8, info = case$name)
    expect_identical(fit$n_iter, as.integer(case$n_iter), info = case$name)
    expect_equal(fit$inertia, case$inertia, tolerance = 1e-8, info = case$name)
  }
})

test_that("KMeans recovers well-separated blobs and is reproducible", {
  d <- make_blobs()
  labels <- KMeans(d$X, n_clusters = 3L, random_state = 0L)
  expect_identical(adjusted_rand_index(labels, d$y), 1)
  expect_identical(labels, KMeans(d$X, n_clusters = 3L, random_state = 0L))
  expect_setequal(labels, 1:3)
})

test_that("KMeans leaves the caller's RNG state alone", {
  d <- make_blobs()
  set.seed(5)
  before <- .Random.seed
  KMeans(d$X, 3L, random_state = 1L)
  expect_identical(.Random.seed, before)
})

test_that("KMeans needs at least as many samples as clusters", {
  expect_error(
    KMeans(matrix(as.numeric(1:4), 2), n_clusters = 3L),
    "n_samples=2 should be >= n_clusters=3.",
    fixed = TRUE
  )
})

test_that("kmeans_plusplus picks distinct data points as centers", {
  d <- make_blobs()
  centers <- seeded(1L, kmeans_plusplus(d$X, 3L))
  expect_identical(nrow(unique(centers)), 3L)
  is_data_point <- apply(centers, 1L, function(center) any(rowSums(abs(sweep(d$X, 2L, center))) == 0))
  expect_true(all(is_data_point))
})

test_that("kmeans_plusplus keeps the candidate with the lowest potential", {
  # Points at 0, 1 and 10, and draws that pick the first and the last point
  # that can be drawn. With the first center at 0 or 1, the candidates are
  # the other of the two and the point at 10, which leaves a potential of 1
  # instead of 81. With the first center at 10 the two candidates tie.
  local_mocked_bindings(runif = function(n, ...) c(1e-6, 0.999)[seq_len(n)], .package = "stats")
  X <- matrix(c(0, 1, 10), ncol = 1L)
  first <- numeric()
  for (seed in 1:6) {
    centers <- seeded(seed, kmeans_plusplus(X, 2L))
    first <- c(first, centers[1L, 1L])
    expect_true(10 %in% centers[, 1L], info = paste("seed", seed))
  }
  # Some first center is not the point at 10, so a wrong choice shows.
  expect_true(any(first != 10))
})

test_that("KMeans keeps the start with the lowest inertia", {
  d <- make_blobs(n_per = 20L, centers = rbind(c(0, 0), c(3, 0), c(0, 3), c(3, 3), c(9, 9)), sd = 0.9)
  Xc <- sweep(d$X, 2L, colMeans(d$X))
  tol <- mean(colMeans(Xc^2)) * 1e-4
  # The ten starts KMeans(n_init = 10) runs under this seed, in order.
  starts <- seeded(1L, lapply(1:10, function(i) {
    kmeans_lloyd(Xc, kmeans_plusplus(Xc, 4L), max_iter = 300L, tol = tol)
  }))
  inertias <- vapply(starts, function(s) s$inertia, numeric(1))
  expect_gt(max(inertias) - min(inertias), 1)
  expect_gt(which.min(inertias), 1L)
  expect_identical(KMeans(d$X, 4L, n_init = 10L, random_state = 1L), starts[[which.min(inertias)]]$labels)
})

test_that("AgglomerativeClustering matches sklearn up to label names", {
  fx <- read_fixture("agglomerative")
  X <- fixture_matrix(fx$X)
  for (linkage in names(fx$labels)) {
    labels <- AgglomerativeClustering(X, n_clusters = 4L, linkage = linkage)
    expect_identical(adjusted_rand_index(labels, fx$labels[[linkage]]), 1, info = linkage)
  }
})

test_that("AgglomerativeClustering rejects an unknown linkage", {
  expect_error(
    AgglomerativeClustering(matrix(as.numeric(1:10), 5), linkage = "median"),
    "Unknown linkage: 'median'. Expected 'ward', 'average', 'single' or 'complete'.",
    fixed = TRUE
  )
})

test_that("standard_scale matches StandardScaler", {
  fx <- read_fixture("spectral")
  expect_equal(standard_scale(fixture_matrix(fx$X)), fixture_matrix(fx$scaled), tolerance = 1e-10)
  expect_identical(standard_scale(cbind(c(1, 2, 3), 5))[, 2], c(0, 0, 0))
})

test_that("each affinity and its spectrum match Python", {
  fx <- read_fixture("spectral")
  Xp <- fixture_matrix(fx$scaled)
  settings <- list(
    self_tuning = list(affinity = "self_tuning", gamma = NULL),
    rbf = list(affinity = "rbf", gamma = NULL),
    rbf_gamma = list(affinity = "rbf", gamma = 0.3),
    knn = list(affinity = "knn", gamma = NULL)
  )
  for (name in names(settings)) {
    a <- spectral_affinity(Xp, affinity = settings[[name]]$affinity, gamma = settings[[name]]$gamma, n_neighbors = 7L)
    expect_equal(as.matrix(a$W), fixture_matrix(fx[[name]]$W), tolerance = 1e-10, info = name)
    expect_equal(a$gamma, fx[[name]]$gamma, tolerance = 1e-10, info = name)
    expect_equal(
      spectral_embedding(a$W, 3L)$values,
      fixture_vector(fx[[name]]$evals),
      tolerance = 1e-8,
      info = name
    )
  }
})

test_that("SpectralClustering separates two moons", {
  d <- make_moons(200L)
  expect_gt(adjusted_rand_index(SpectralClustering(d$X, n_clusters = 2L, random_state = 0L), d$y), 0.9)
})

test_that("from 1,000 samples on the sparse solver is used, reproducibly", {
  d <- make_moons(1200L)
  a <- SpectralClustering(d$X, n_clusters = 2L, random_state = 0L)
  expect_identical(a, SpectralClustering(d$X, n_clusters = 2L, random_state = 0L))
  expect_gt(adjusted_rand_index(a, d$y), 0.9)
})

test_that("the sparse and dense solvers find the same smallest eigenvalues", {
  # The Python package once used an eigensolver call that returned
  # unconverged values without failing, so the two solvers are compared on
  # the Laplacian SpectralClustering builds for 1,200 moons.
  d <- make_moons(1200L)
  W <- spectral_affinity(standard_scale(d$X), affinity = "self_tuning", n_neighbors = 7L)$W
  n <- nrow(W)
  dinv <- 1 / sqrt(rowSums(W))
  L <- diag(n) - dinv * W * rep(dinv, each = n)
  sparse <- sort(sparse_eigs(L, 3L)$values)
  dense <- sort(dense_eigs(L, 3L)$values)
  # Each value within the sparse solver's relative tolerance, 1e-4, with an
  # absolute floor for the eigenvalue at 0.
  expect_true(all(isclose(sparse, dense, rtol = 1e-4, atol = 1e-10)),
              info = paste(format(sparse), "vs", format(dense), collapse = "; "))
})

test_that("a failing sparse solve falls back to the dense one", {
  calls <- 0L
  local_mocked_bindings(sparse_eigs = function(L, k) {
    calls <<- calls + 1L
    stop("Factor is exactly singular")
  })
  d <- make_moons(1200L)
  expect_gt(adjusted_rand_index(SpectralClustering(d$X, n_clusters = 2L, random_state = 0L), d$y), 0.9)
  expect_identical(calls, 1L)
})

test_that("the spectral helpers reject impossible settings", {
  X <- matrix(as.numeric(1:20), 10)
  expect_error(spectral_affinity(X, affinity = "cosine"), "Unknown affinity: 'cosine'", fixed = TRUE)
  expect_error(spectral_affinity(X, n_neighbors = 1L), "n_neighbors must be at least 2.", fixed = TRUE)
  expect_error(spectral_affinity(X, n_neighbors = 11L), "n_neighbors=11 is larger than the 10 samples.", fixed = TRUE)
})

blobs3d <- make_blobs(n_per = 30L, centers = diag(5, 3), sd = 1, seed = 42L)

edge_table <- function(from, to, weight) {
  out <- data.frame(i = as.integer(pmin(from, to)), j = as.integer(pmax(from, to)), weight = as.numeric(weight))
  out <- out[order(out$i, out$j), ]
  rownames(out) <- NULL
  out
}

test_that("the kNN graph matches Python's for both weightings", {
  f <- read_fixture("knn_graph")
  X <- fixture_matrix(f$X)
  for (case in f$cases) {
    info <- paste(case$weighting, case$n_neighbors)
    graph <- knn_graph(X, n_neighbors = case$n_neighbors, weighting = case$weighting)
    edges <- igraph::as_edgelist(graph, names = FALSE)
    ours <- edge_table(edges[, 1], edges[, 2], igraph::E(graph)$weight)
    theirs <- edge_table(case$edges[, 1] + 1L, case$edges[, 2] + 1L, case$weights)
    expect_identical(ours[, c("i", "j")], theirs[, c("i", "j")], info = info)
    expect_equal(ours$weight, theirs$weight, tolerance = 1e-12, info = info)
  }
})

test_that("the graph has a vertex per sample and clips n_neighbors to n - 1", {
  X <- withr::with_seed(0, matrix(stats::rnorm(12), 6))
  graph <- knn_graph(X, n_neighbors = 100L)
  expect_equal(igraph::vcount(graph), 6)
  expect_equal(igraph::ecount(graph), 15)
  expect_false(igraph::is_directed(graph))
})

test_that("connectivity weights are 1 and Jaccard weights are fractions", {
  expect_true(all(igraph::E(knn_graph(blobs3d$X, 10L))$weight == 1))
  jaccard <- igraph::E(knn_graph(blobs3d$X, 10L, "jaccard"))$weight
  expect_true(all(jaccard > 0 & jaccard <= 1))
  expect_true(any(jaccard < 1))
})

test_that("an unknown weighting is an error", {
  expect_error(
    knn_graph(blobs3d$X, 10L, "bogus"),
    "Unknown weighting: 'bogus'. Expected 'connectivity' or 'jaccard'.",
    fixed = TRUE
  )
})

test_that("LeidenClustering recovers three blobs with labels counting from 1", {
  labels <- LeidenClustering(blobs3d$X, random_state = 0L)
  expect_gt(adjusted_rand_index(labels, blobs3d$y), 0.9)
  expect_identical(sort(unique(labels)), seq_len(max(labels)))
})

test_that("a higher resolution gives at least as many Leiden communities", {
  counts <- vapply(c(0.1, 0.5, 1, 3), function(r) {
    count_clusters(LeidenClustering(blobs3d$X, resolution = r, random_state = 0L))
  }, integer(1))
  expect_false(is.unsorted(counts))
  expect_gt(counts[4], counts[1])
})

test_that("LeidenClustering is reproducible and leaves the session's stream alone", {
  set.seed(5)
  before <- .Random.seed
  a <- LeidenClustering(blobs3d$X, resolution = 1.5, random_state = 7L)
  expect_identical(.Random.seed, before)
  expect_identical(a, LeidenClustering(blobs3d$X, resolution = 1.5, random_state = 7L))
})

test_that("LeidenClustering passes the objective, the weighting and the scaling on", {
  cpm <- LeidenClustering(blobs3d$X, objective_function = "cpm", resolution = 0.5, random_state = 0L)
  modularity <- LeidenClustering(blobs3d$X, resolution = 0.5, random_state = 0L)
  expect_length(cpm, 90L)
  expect_false(identical(cpm, modularity))
  expect_gt(adjusted_rand_index(LeidenClustering(blobs3d$X, weighting = "jaccard", random_state = 0L), blobs3d$y), 0.9)
  expect_gt(adjusted_rand_index(LeidenClustering(blobs3d$X, scale = TRUE, random_state = 0L), blobs3d$y), 0.9)
})

test_that("an unknown Leiden objective is an error", {
  expect_error(
    LeidenClustering(blobs3d$X, objective_function = "surprise"),
    "Unknown objective_function: 'surprise'. Expected 'modularity' or 'cpm'.",
    fixed = TRUE
  )
})

test_that("LouvainClustering recovers three blobs and follows the resolution", {
  labels <- LouvainClustering(blobs3d$X, random_state = 0L)
  expect_gt(adjusted_rand_index(labels, blobs3d$y), 0.9)
  expect_identical(sort(unique(labels)), seq_len(max(labels)))
  counts <- vapply(c(0.1, 0.5, 1, 3), function(r) {
    count_clusters(LouvainClustering(blobs3d$X, resolution = r, random_state = 0L))
  }, integer(1))
  expect_false(is.unsorted(counts))
  expect_gt(counts[4], counts[1])
})

test_that("LouvainClustering is reproducible and runs with Jaccard weights and scaling", {
  set.seed(5)
  before <- .Random.seed
  a <- LouvainClustering(blobs3d$X, resolution = 1.5, random_state = 7L)
  expect_identical(.Random.seed, before)
  expect_identical(a, LouvainClustering(blobs3d$X, resolution = 1.5, random_state = 7L))
  expect_gt(adjusted_rand_index(LouvainClustering(blobs3d$X, weighting = "jaccard", random_state = 0L), blobs3d$y), 0.9)
  expect_gt(adjusted_rand_index(LouvainClustering(blobs3d$X, scale = TRUE, random_state = 0L), blobs3d$y), 0.9)
})

test_that("HDBSCAN matches scikit-learn's eom and leaf selections", {
  skip_if_not_installed("dbscan")
  f <- read_fixture("hdbscan")
  X <- fixture_matrix(f$X)
  for (m in c(5L, 10L)) {
    for (method in c("eom", "leaf")) {
      expected <- as.integer(f$labels[[paste0(method, "_", m)]])
      labels <- HDBSCAN(X, min_cluster_size = m, cluster_selection_method = method)
      expect_true(same_partition(labels, expected), info = paste(method, m))
    }
  }
  # The fixture tests leaf selection only if the two selections differ on it,
  # and tests the noise label only if some sample is noise.
  differ <- vapply(c(5L, 10L), function(m) {
    !same_partition(as.integer(f$labels[[paste0("eom_", m)]]), as.integer(f$labels[[paste0("leaf_", m)]]))
  }, logical(1))
  expect_true(any(differ))
  expect_true(any(unlist(f$labels) == -1L))
})

test_that("HDBSCAN numbers clusters from 1 and marks noise -1", {
  skip_if_not_installed("dbscan")
  X <- fixture_matrix(read_fixture("hdbscan")$X)
  for (method in c("eom", "leaf")) {
    labels <- HDBSCAN(X, min_cluster_size = 5L, cluster_selection_method = method)
    clusters <- sort(unique(labels[labels >= 0L]))
    expect_identical(clusters, seq_along(clusters), info = method)
    expect_true(all(labels == -1L | labels >= 1L), info = method)
  }
})

test_that("an unknown cluster selection method is an error", {
  skip_if_not_installed("dbscan")
  expect_error(
    HDBSCAN(blobs3d$X, cluster_selection_method = "tree"),
    "Unknown cluster_selection_method: 'tree'. Expected 'eom' or 'leaf'.",
    fixed = TRUE
  )
})

test_that("HDBSCAN needs the dbscan package", {
  local_mocked_bindings(has_package = function(package) FALSE)
  expect_error(
    HDBSCAN(blobs3d$X),
    'dbscan is required for HDBSCAN. Install it with: install.packages("dbscan")',
    fixed = TRUE
  )
})
