
<!-- README.md is generated from README.Rmd. Edit the .Rmd and run -->
<!-- devtools::build_readme() to regenerate. -->

# CARVE

<!-- badges: start -->
<!-- badges: end -->

**Cluster Analysis with Resampling for Validation and Exploration** —
the R companion to the Python
[`carve`](https://github.com/DataSlingers/CARVE) package.

Choosing the number of clusters is a recurring challenge in unsupervised
learning, and standard internal cluster validity indices can be brittle
in high-dimensional, noisy, or nonlinear settings. CARVE quantifies
clustering robustness via two resampling-based concepts:

- **Stability** — the reproducibility of cluster assignments under data
  perturbation, summarised through ARI, consensus PAC, Gini, and
  cross-entropy.
- **Generalizability** — the agreement between held-out clusterings and
  the predictions of a classifier trained on the in-sample clustering.

Both metrics are produced at global, per-cluster, and per-sample
resolutions, and all of it lives behind a familiar fit / get-labels
workflow with `ggplot2` visualisations and first-class `Seurat` /
`SingleCellExperiment` integration.

## Installation

``` r
# install.packages("remotes")
remotes::install_github("DataSlingers/CARVE", subdir = "code/carve-r")
```

CARVE depends only on CRAN packages (`R6`, `clue`, `ranger`, `RSpectra`,
`FNN`, `Matrix`, `ggplot2`, `patchwork`, `furrr`, `progressr`).
`Seurat`, `SingleCellExperiment`, and `reticulate` are optional and only
loaded when used.

## Quick start

``` r
library(CARVE)

set.seed(1)
centers <- rbind(c(0, 0), c(6, 0), c(3, 5))
X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i) {
  matrix(stats::rnorm(60, 0, 0.4), ncol = 2) +
    matrix(centers[i, ], 30, 2, byrow = TRUE)
}))

fit <- CARVE$new(
  n_clusters = 2:6,
  n_resamples = 50,
  subsample_ratio = 0.8,
  random_state = 1L
)
fit$fit(X)

fit$get_k(measure = "stability", rule = "max")
#> [1] 3
labels <- fit$get_labels(measure = "stability", rule = "max")
```

The S3 entry point `carve()` accepts a matrix, a numeric data frame, or
(when the optional packages are installed) a `Seurat` or
`SingleCellExperiment` object:

``` r
fit <- carve(X, n_clusters = 2:6, n_resamples = 50, random_state = 1L)
```

## Visualisation

All plot methods return `ggplot` objects (the consensus heatmap is a
`patchwork`):

``` r
fit$plot_metric_over_n_clusters(measure = "stability", rule = "max")
fit$plot_consensus_matrix(measure = "stability", rule = "max")
fit$plot_cluster_violin(source = "gini", measure = "stability", rule = "max")
fit$plot_cluster_scatter(source = "gini", measure = "stability", rule = "max")
```

## Seurat workflow

``` r
library(Seurat)
data("pbmc_small")

pbmc_small <- RunCARVE(pbmc_small, reduction = "pca", n_dims = 10,
                       n_clusters = 2:6, n_resamples = 30, random_state = 1L)
pbmc_small <- AddCarveLabels(pbmc_small,
                              measure = "stability", rule = "max")
DimPlot(pbmc_small, group.by = "carve_labels")
```

## Parallel resampling

``` r
fit <- CARVE$new(n_clusters = 2:10, n_resamples = 100,
                 n_jobs = 4L, random_state = 1L)
progressr::with_progress(fit$fit(X, show_progress = TRUE))
```

`n_jobs > 1` opens a scoped `future::multisession()` plan and pins
worker RNGs to Mersenne-Twister so results are bit-for-bit reproducible
across `n_jobs` values when the seed is fixed.

## Vignettes

- `vignette("getting-started", package = "CARVE")` — five-minute tour.
- `vignette("seurat-workflow", package = "CARVE")` — end-to-end Seurat
  pipeline on `pbmc_small`.
- `vignette("cross-validation", package = "CARVE")` — Python ↔ R
  numerical equivalence on a fixed set of toy datasets, driven via
  `reticulate`.

## Citation
If you use CARVE in your research, please cite:


## License

MIT (see `LICENSE`).
