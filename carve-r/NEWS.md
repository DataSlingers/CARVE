# CARVE 0.1.0 (unreleased)

Initial release. R port of the Python
[`carve`](https://github.com/DataSlingers/CARVE) package
accompanying the manuscript *"CARVE: Cluster Analysis with Resampling
for Validation and Exploration"* (under review at PLoS Computational
Biology).

## Features

* `CARVE` R6 estimator with `$fit()`, `$get_k()`, `$get_labels()`,
  `$get_estimator()`, `$save()` / `CARVE$load()`.
* `carve()` S3 generic with methods for matrix, numeric data frame,
  `Seurat`, and `SingleCellExperiment`.
* Seurat verb-style helpers: `RunCARVE()` and `AddCarveLabels()`.
* Built-in clustering primitives:
  * `stats::kmeans` with multiple random starts;
  * `stats::hclust(method = "ward.D2")` (matches sklearn's squared-
    distance Ward);
  * `SpectralClusteringCARVE` R6 class with rbf, kNN, and self-tuning
    affinities, dispatching dense / sparse eigen solvers.
* Stability metrics: ARI, consensus PAC, Gini, cross-entropy.
* Generalizability metrics: held-out RandomForest accuracy and ARI
  via `ranger`.
* Selection rules: `max`, `1se`, `quantile`.
* Six `ggplot2` / `patchwork` plot methods mirroring the Python
  figures.
* Reproducible parallel resampling via `future` + `furrr` with seed
  pinning so results are bit-for-bit identical across `n_jobs` values
  for a fixed `random_state`.
* `progressr`-based progress bars (off by default).

## Numerical parity with the Python package

Pure-math primitives (consensus formulas, Hungarian alignment,
selection rules, ARI) are pinned to byte-exact agreement in the test
suite. End-to-end fits agree statistically (±0.05 ARI, ±0.03 PAC /
Gini / CE) on the fixed-seed datasets in
`vignette("cross-validation")`; bit-exact agreement is infeasible
because the underlying RNGs differ between languages.
