# CARVE 2.0.0

Version 2.0.0 is a rewrite with the same scores, selection rules and
defaults as the Python package, carve-validate;
`vignette("python-users")` lists where the two differ. Its interface is
not compatible with version 1.0.0.

## Replaced functions

| 1.0.0 | 2.0.0 |
|---|---|
| `CARVE$new(...)` and `$fit(X)` | `carve(X, ...)`, which returns an S4 object of class `CARVE` |
| `$get_k()`, `$get_estimator()`, `$get_labels()` | `get_k(fit)`, `get_estimator(fit)`, `get_labels(fit)` |
| `$save()` and `CARVE$load()` | `saveRDS()` and `readRDS()` |
| the six `$plot_*()` methods | eight `plot_*()` functions, which take a fit, a SingleCellExperiment or a Seurat object |
| `RunCARVE()` | `run_carve()` |
| `AddCarveLabels()` | `attach_results()` |
| `SpectralClusteringCARVE` | `SpectralClustering()` |
| S3 methods of `carve()` | S4 methods of `carve()` |

The 28 internal functions that 1.0.0 exported, from
`adjusted_rand_index()` to `validation_iter()`, are no longer exported.
The fields of a fitted 1.0.0 object, such as `estimator_results_`, are now
read through accessor functions such as `estimator_results()`,
`consensus_matrix()` and `sample_scores()`.

## Changed defaults

- `get_k()`, `get_labels()` and the other selection functions select on
  `measure = "stability"`, as the Python package does. Version 1.0.0 used
  `"generalizability"`.
- `subsample_ratio` is 0.618, as in Python. It was 0.8.

## New features

- Resolution sweeps with `LeidenClustering()` and `LouvainClustering()`,
  and minimum cluster size sweeps with `HDBSCAN()`, with a noise policy
  for HDBSCAN's -1 labels.
- Anchored consensus above `anchor_threshold` samples, 5,000 by default,
  which keeps memory bounded on large data.
- Randomized preprocessing, with `preprocessing_option()`,
  `preprocessing_results()`, `preprocessing_pipelines()`,
  `pipeline_from_spec()` and `plot_metric_by_pipeline()`.
- Custom clustering functions through `estimator_grid()`, and custom
  classifiers.
- `n_jobs` as a core budget, run through BiocParallel, and `BPPARAM` to
  choose the backend. Results do not depend on either.
- `get_sweep_value()`, and the `k`, `sweep_value`, `consensus_k` and
  `noise_labels` arguments of `get_labels()`.
- `plot_n_clusters_over_sweep()`.
- `run_carve()` and `attach_results()` store the selected configuration
  in SingleCellExperiment and Seurat objects with the Python package's
  layout.
- The data set `pbmc3k_subset`, 1,000 cells of 10x Genomics' PBMC 3k.
- Four vignettes: `vignette("CARVE")`, `vignette("single-cell")`,
  `vignette("customizing")` and `vignette("python-users")`.

## Dependencies

R6, furrr, future, progressr and patchwork are no longer used.
BiocParallel, igraph, irlba, Rtsne, S4Vectors, scales,
SingleCellExperiment, SummarizedExperiment and withr are new imports.
dbscan, SeuratObject and uwot are suggested, for HDBSCAN, Seurat objects
and UMAP.
