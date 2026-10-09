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

# Muffles the warnings whose message contains pattern, so a fit that warns
# on purpose (an anchored run, a non-default mode) can set up a test under
# warn = 2. Any other warning still fails the test.
muffle <- function(expr, pattern) {
  withCallingHandlers(expr, warning = function(w) {
    if (grepl(pattern, conditionMessage(w), fixed = TRUE)) {
      invokeRestart("muffleWarning")
    }
  })
}

cell_names <- function(n) sprintf("cell%03d", seq_len(n))

# A SingleCellExperiment over the rows of X, which become its cells: by
# default X is its "PCA" reduced dimensions and t(X) its "logcounts" assay.
make_sce <- function(X, reductions = list(PCA = X), assays = list(logcounts = t(X))) {
  cells <- cell_names(nrow(X))
  assays <- lapply(assays, function(values) {
    dimnames(values) <- list(sprintf("gene%d", seq_len(nrow(values))), cells)
    values
  })
  reductions <- lapply(reductions, function(values) {
    rownames(values) <- cells
    values
  })
  args <- list(reducedDims = reductions, colData = S4Vectors::DataFrame(row.names = cells))
  if (length(assays) > 0L) {
    args$assays <- assays
  }
  do.call(SingleCellExperiment::SingleCellExperiment, args)
}

# A Seurat object over the rows of X, which become its cells: Poisson
# counts, data as the "data" layer of assay "RNA" (none when data is NULL),
# and X as the "pca" reduction by default.
make_seurat <- function(X, reductions = list(pca = X), data = t(X)) {
  testthat::skip_if_not_installed("SeuratObject")
  cells <- cell_names(nrow(X))
  n_genes <- if (is.null(data)) ncol(X) else nrow(data)
  genes <- sprintf("gene%d", seq_len(n_genes))
  counts <- withr::with_seed(1L, matrix(
    stats::rpois(n_genes * nrow(X), 10), n_genes, nrow(X),
    dimnames = list(genes, cells)
  ))
  object <- SeuratObject::CreateSeuratObject(counts = Matrix::Matrix(counts, sparse = TRUE))
  if (!is.null(data)) {
    dimnames(data) <- list(genes, cells)
    SeuratObject::LayerData(object, layer = "data") <- Matrix::Matrix(data, sparse = TRUE)
  }
  for (name in names(reductions)) {
    embeddings <- reductions[[name]]
    key <- paste0(gsub("[^A-Za-z0-9]", "", name), "_")
    dimnames(embeddings) <- list(cells, paste0(key, seq_len(ncol(embeddings))))
    object[[name]] <- SeuratObject::CreateDimReducObject(embeddings = embeddings, key = key, assay = "RNA")
  }
  object
}
