# R package rework: design

Date: 2026-10-06
Branch: `r-package-rework`, off `main` at d5e46eb
Status: design approved in brainstorming on 2026-10-06 and implemented in four stages, each planned
in its own conversation. Edited on 2026-10-08 to describe the package as built; the Decisions
sections of the four stage plans give the reasons for each change.

## Goal

Rebuild the R package in `carve-r/` so that it matches the Python package feature for feature,
follows R and Bioconductor conventions, and has documentation that reads as if a person wrote it.
The author asked for two deliverables: the public documentation (reference pages, README, NEWS and
four vignettes) and a tutorial, `notebooks/R_Tutorial.Rmd`, that walks through every essential
function. The package rework is what those documents describe, so it comes first.

## Decisions

Settled with the author on 2026-10-06.

| Question | Decision |
|---|---|
| Parity scope | Full parity with `carve.CARVE`, `carve.cluster`, `carve.tl` and `carve.pl`. `carve.sim` and `benchmarks` stay Python-only. |
| API shape | One fitting call, `carve()`, returns a fit object. Each Python method becomes a function of the same name that takes the fit as its first argument. |
| Class system | S4, so a Bioconductor submission later needs no class rewrite. |
| Distribution | GitHub only for now (`remotes::install_github`), kept ready for Bioconductor. |
| Approach | Rewrite on a branch with the Python package as the executable spec. Version 2.0.0, a clean break with no shims for the old R6 API. |
| Consensus memory | Store consensus blocks as doubles, which takes twice the memory of Python's float32 blocks. Document it and print the projected size in the verbose header. |
| Tutorial format | `notebooks/R_Tutorial.Rmd`, committed with its rendered HTML, outside `R CMD check`. |
| Tutorial data | Simulated data first, then PBMC 3k through the standard Seurat workflow. |
| Writing | The `de-ai-writing` skill governs every text a user reads. The author named this the most important requirement of the project. |

## The package before the rework

Before the rework, the R package was last changed on 2026-08-21. It has about 3,600 lines of R,
against about 10,000 in `src/carve/`. There are 103 testthat tests, and one fails locally
(`test-plotting.R:148`, the annotation label). The API is an R6 class `CARVE` plus an S3 generic
`carve()`, and there are three vignettes.

The R package sweeps only k, with KMeans, Ward and spectral clustering. Compared with Python it
lacks:

- the `resolution` and `min_cluster_size` sweeps, with Leiden, Louvain and HDBSCAN
- the noise policy for HDBSCAN's -1 labels
- anchored consensus above 5,000 samples
- randomized preprocessing fit per subsample, and `preprocessing_results_`
- user-supplied classifiers, and `n_jobs` as a core budget (R pins `ranger` to one thread)
- `get_sweep_value`, and the `k`, `sweep_value`, `consensus_k` and `noise_labels` arguments of
  `get_labels`
- `plot_metric_by_pipeline` and `plot_n_clusters_over_sweep`
- neighbor-count scaling on subsamples, and fit-time validation of the run parameters

Its default `subsample_ratio` is 0.8, where Python's is 0.618. It exports about 30 internal
functions (`validation_iter`, `run_validation` and others). The README and vignettes use British
spelling, bold text and em dashes throughout.

The manuscript (V2, Software Availability and Methods) says the R package has an interface
analogous to Python's, with native Seurat and SingleCellExperiment support, and gives the install
line `remotes::install_github("DataSlingers/CARVE", subdir = "code/carve-r")`. That line names a
subdirectory the repository does not have: its root holds `carve-r/` directly, so the README gives
`subdir = "carve-r"`. The manuscript's line needs the same correction, which is the author's.

## Architecture

### Package and fit class

The package keeps the name `CARVE` and becomes version 2.0.0. `carve()` returns an S4 object of
class `CARVE`. Its slots hold the resolved run parameters and the fitted attributes of the Python
class.

A validity method checks that every per-configuration list carries exactly the `config_id` values
of the results table. It is the R form of Python's `RuntimeError("config_id is misaligned ...")`.
Per-configuration matrices and scores are stored in lists named by `config_id` and looked up by
name. No code indexes them by position.

`show()` prints a short summary: data size, sweep axis and values, estimators, number of
configurations, resamples per configuration, and whether the run was anchored or randomized.

### Module map

One R file per Python module. Tests mirror the R files one to one (`sweep.R` is tested in
`test-sweep.R`).

| R file | Python source |
|---|---|
| `AllClasses.R` | fitted fields of `api.CARVE`, `_types.ModePolicy`, `_sweep.SweepSpec`, `_pipeline.PipelineSpec` |
| `AllGenerics.R` | method names of `api.CARVE` and `carve.pl` |
| `carve.R` | `api.CARVE.fit` |
| `accessors.R` | `get_k`, `get_sweep_value`, `get_labels`, `get_estimator`, and the fitted attributes |
| `sweep.R` | `_sweep.py` |
| `selection.R` | `_selection.py` |
| `runner.R` | `_runner.py` |
| `consensus.R` | `_consensus.py` |
| `accuracy.R` | `_accuracy.py` |
| `utils.R` | `_utils.py` |
| `pipeline.R` | `_pipeline.py` |
| `grids.R` | `_grids.py` |
| `output.R` | `_output.py` |
| `estimators.R` | `cluster.py`, plus the KMeans, AgglomerativeClustering and HDBSCAN wrappers |
| `plotting.R` | `_plotting.py`: drawing from a results table or per-sample vectors |
| `plots.R` | the plot methods of `api.CARVE` and `pl/_plots.py` |
| `sce.R`, `seurat.R` | `_anndata.py`, one file per object class |
| `tl.R` | `tl/_carve.py` |
| `transforms.R` | scikit-learn's `StandardScaler`, `PCA`, `TSNE` and `FunctionTransformer` options, and umap-learn's `UMAP` |
| `types.R` | `_types.py`, and `estimator_grid()` with the grid expansion |
| `data.R` | the help page of `pbmc3k_subset` |

Dependency direction follows CLAUDE.md. `utils.R`, `sweep.R` and `estimators.R` are leaves, and
nothing called by `carve.R` calls back into `carve.R` or `accessors.R`.

### Dependencies

Imports: `methods`, `stats`, `utils`, `parallel`, `Matrix`, `BiocParallel`,
`SingleCellExperiment`, `SummarizedExperiment`, `S4Vectors`, `igraph (>= 1.3.0)`, `ranger`,
`RSpectra`, `irlba`, `Rtsne (>= 0.15)`, `FNN`, `clue`, `ggplot2 (>= 3.5.0)`, `scales`,
`withr (>= 2.4.2)`.

Suggests: `SeuratObject (>= 5.0.0)`, `dbscan`, `uwot`, `testthat (>= 3.1.8)`, `jsonlite`, `knitr`,
`rmarkdown`. The package reads Seurat objects through SeuratObject, so Seurat itself is not needed;
only the tutorial runs Seurat's workflow. `reticulate` is used only by `data-raw/parity_check.R`,
outside the built package, and is not declared.

This mirrors Python's split between core dependencies and optional extras: HDBSCAN and UMAP are
optional in both. DESCRIPTION gains a `biocViews:` field so `remotes::install_github` resolves the
Bioconductor imports. `R6`, `furrr`, `future`, `progressr` and `patchwork` are dropped; the
consensus heatmap is a single ggplot.

### Conventions

- Seeds. Resample `i` of a configuration uses `random_state + i` for its subsample and
  `random_state + i + n_resamples` for its held-out split, as `validation_iter` does. Each resample
  runs under a local seed, and the caller's RNG state is restored afterwards. Results are identical
  for every `n_jobs`.
- Messages. Diagnostics go through `warning()`. Verbose output goes through `message()`, gated on
  `verbose`, which is also what Bioconductor requires. There is no logging package.
- Parallelism. A serial loop over configurations, with `BiocParallel::bplapply` over resamples
  inside it. `n_jobs` is a core budget resolved by the port of `resolve_core_budget`: the outer
  worker count, and `ranger` threads per worker equal to the core count divided by the workers.
  The classifier that extends an anchored cut in `get_labels` gets the whole budget. An optional
  `BPPARAM` argument lets users choose the backend; its worker count then sets the outer count.
- Persistence. `saveRDS` and `readRDS`. There are no `save()` or `load()` wrappers.

## Public API

The exported surface matches Python's public surface, plus the few R constructs Python gets from
its classes (`estimator_grid`, `preprocessing_option`, the accessors). Internals are not exported;
tests reach them from the package namespace.

### Fitting

`carve(x, ...)` is an S4 generic with methods for `matrix`, `data.frame` and
`SingleCellExperiment`, and an `ANY` method that handles Seurat objects. Its arguments, in
Python's order and with Python's defaults:

```r
carve(x, n_clusters = 2:10, resolution = NULL, sweep = NULL, sweep_values = NULL,
      finer_is_larger = NULL, noise_policy = "drop", n_resamples = 100,
      subsample_ratio = 0.618, anchor_threshold = 5000, consensus_anchors = NULL,
      estimator_param_grids = "light", normalization_options = NULL,
      dim_reduction_options = NULL, randomize_preprocessing = FALSE, classifier = NULL,
      n_trees = 100, reference_labels = NULL, mode = "default", n_jobs = 1,
      BPPARAM = NULL, random_state = NULL, show_progress = FALSE, verbose = 0)
```

The `SingleCellExperiment` method adds `assay`, `reduction` and `n_dims`, which correspond to
Python's `layer`, `use_rep` and `n_pcs`. By default it uses the reduced dimension `"PCA"` when
present and the `"logcounts"` assay otherwise. Seurat objects go through the `ANY` method, which
takes the same three arguments and defaults to the `"pca"` reduction, then to the data layer of the
default assay. Cells are columns in both classes, so the matrix is transposed before fitting.

### Querying

```r
get_k(fit, measure = "stability", rule = "1se", not_two = FALSE)
get_sweep_value(fit, measure = "stability", rule = "1se", not_two = FALSE)
get_estimator(fit, measure = "stability", rule = "1se", not_two = FALSE)
get_labels(fit, measure = "stability", rule = "1se", k = NULL, sweep_value = NULL,
           consensus_k = NULL, not_two = FALSE, mode = "default", estimator = NULL,
           noise_labels = FALSE, noise_quantile = 0.05, noise_score = "gini",
           reference_labels = NULL)
```

`get_estimator` returns the selected estimator as a function of `X` and `random_state`, with the
selected parameters already set. Its `"estimator"` and `"params"` attributes record the name and
the parameters.

`get_labels()` keeps no state. Python's `get_labels` stores its result as the reference for later
calls; R takes earlier labels through `reference_labels`.

Accessors for the fitted attributes:

| R accessor | Python attribute |
|---|---|
| `estimator_results(fit)` | `estimator_results_` |
| `estimator_param_grids(fit)` | `estimator_param_grids_` |
| `preprocessing_results(fit)` | `preprocessing_results_` |
| `preprocessing_pipelines(fit)` | `preprocessing_pipelines_` |
| `sweep_spec(fit)` | `sweep_` (not `sweep()`, which would mask `base::sweep`) |
| `consensus_matrix(fit, config_id, type = c("stability", "generalizability"))` | `consensus_matrices_`, `consensus_generalizability_matrices_` |
| `sample_scores(fit, config_id, source = c("gini", "ce", "accuracy"))` | `stability_gini_scores_`, `stability_ce_scores_`, `generalizability_scores_` |
| `consensus_anchors(fit)` | `consensus_anchors_` |
| `input_data(fit)` | `X_` |

### Estimators

Estimators keep Python's class names and are plain functions that take a data matrix, their
parameters and `random_state`, and return integer labels from 1 to k, with -1 for noise:

- `KMeans(X, n_clusters, ...)`: k-means++ seeding and Lloyd iterations through `stats::kmeans`,
  matching sklearn's algorithm and its `n_init` default.
- `AgglomerativeClustering(X, n_clusters, linkage = "ward", ...)`: `stats::hclust` on Euclidean
  distances, with `ward.D2` for Ward (sklearn's Ward), and `average`, `single` and `complete`.
- `SpectralClustering(X, n_clusters, affinity = "self_tuning", ...)`: a port of
  `cluster.SpectralClustering`, with the shift-invert eigensolver and the dense fallback.
- `LeidenClustering(X, resolution, n_neighbors = 15, ...)` and `LouvainClustering(...)`:
  `igraph::cluster_leiden` and `igraph::cluster_louvain`, seeded, on a port of `build_knn_graph`
  with both edge weightings.
- `HDBSCAN(X, min_cluster_size, cluster_selection_method = "eom")`: the single-linkage tree of
  `dbscan::hdbscan()`, condensed and selected by a port of scikit-learn's `_tree.pyx`, credited in
  `inst/COPYRIGHTS`.

The parameter defaults of each function are read from `cluster.py` and from the sklearn classes it
wraps. `estimator_grid(estimator, ..., name = NULL)` is Python's `(EstimatorClass, param_grid)`
tuple; parameter names are checked against the function's formals, as `get_estimator_param_names`
checks a class's parameters. Any function with the estimator signature works as a custom
estimator.

### Classifier

`classifier = NULL` builds a `ranger` forest with `n_trees` trees, the settings of
`default_generalizability_classifier`, and the per-worker thread count. A custom classifier is a
function `(x_train, y_train, x_test)` that returns predicted labels for `x_test`. CARVE also
passes `n_threads` and `random_state` when the function has those arguments, the rule Python
applies to `n_jobs`.

### Preprocessing

`Identity`, `StandardScaler`, `Log1p`, `PCA` (through `irlba`), `TSNE` (`Rtsne`) and `UMAP`
(`uwot`) are functions that take a subsample, their parameters and `random_state`, and return the
transformed subsample. `preprocessing_option(transform, ..., name = NULL)` is Python's
`PreprocOption`. `pipeline_from_spec(spec, random_state)` returns a function that applies a stored
pipeline to new data, for example to embed the full data with the best pipeline. Option, pipeline
and method labels are the same strings that Python's `option_label` and `format_method_label`
produce, so the R and Python results tables can be compared row by row.

### Single-cell integration

```r
run_carve(object, ..., assay = NULL, reduction = NULL, n_dims = NULL, key = "carve",
          measure = "stability", rule = "1se", not_two = FALSE, k = NULL, sweep_value = NULL,
          consensus_k = NULL, reference_key = NULL, store_consensus = TRUE,
          store_results = TRUE, mode = "default", random_state = 0)
attach_results(object, fit, key = "carve", measure = "stability", rule = "1se",
               not_two = FALSE, k = NULL, sweep_value = NULL, consensus_k = NULL,
               store_consensus = TRUE, store_results = TRUE, assay = NULL, reduction = NULL,
               n_dims = NULL, mode = "default")
```

`run_carve` is `tl.carve` (fit, then attach) and returns the modified object. `attach_results` is
`tl.attach_results`. Both replace the old `RunCARVE` and `AddCarveLabels`. Storage follows
Python's key layout, with `colData(sce)` or `obj@meta.data` in place of `obs`:

- column `carve`: the consensus labels, as a factor
- columns `carve_stability` (Gini), `carve_stability_ce` and `carve_generalizability`: the
  per-cell scores
- a record in `metadata(sce)$carve` or `obj@misc$carve`: a list with `params`, `results`,
  `preprocessing_results` (randomized runs only), and `consensus`, the selected configuration's
  matrix. As in Python, the matrix is skipped with a warning when the run was anchored, and is
  left out when `store_consensus = FALSE`.

The fit object itself is not stored, because it holds the consensus matrices of every
configuration. Users who want those keep the fit, which is also what Python's documentation
advises. This replaces the wording of the approved section 2, which said the fit would be stored.

`attach_results()` records `assay`, `reduction` and `n_dims`, so that the scatter plots can rebuild
the data. Each call replaces the whole record.

### Plots

There are eight plots: `plot_metric_over_n_clusters`, `plot_metric_by_pipeline`,
`plot_n_clusters_over_sweep`, `plot_consensus_matrix`, `plot_cluster_boxplot`,
`plot_cluster_violin`, `plot_cluster_scatter`, `plot_diagnostic_scatter`.

Each is an S4 generic. The method for `CARVE` corresponds to the Python method of the same name.
The `SingleCellExperiment` methods, and the `ANY` methods, which handle Seurat objects, correspond
to the `carve.pl` functions and read the stored record; the scatter plots take `basis`, a
reduced-dimension name. Every plot returns a ggplot object. Python's `ax`, `figsize`, `show`,
`save` and `dpi` arguments are dropped, because plots are saved with `ggsave` and styled with `+`.
Selection and content arguments keep Python's names and defaults. The palette uses the color
values of `_plotting.py`, defined once in `plotting.R`.

### Bundled data

`pbmc3k_subset`: a list with the 30 principal components, a two-dimensional UMAP and the Seurat
cluster labels of 1,000 PBMC 3k cells, about 150 KB. `data-raw/pbmc3k_subset.R` builds it and
records the download URL.

## Engine behavior

The engine ports `_runner.py`, `_sweep.py`, `_consensus.py`, `_selection.py`, `_utils.py`,
`_pipeline.py` and `_grids.py` behavior for behavior:

- Sweep axis: `n_clusters`, `resolution` or `min_cluster_size`, through a frozen sweep spec with
  `finer_is_larger` inferred for the known parameters. A run sweeps exactly one parameter, and
  mixing k-based and resolution-based estimators is an error.
- Selection: `max`, `1se` and `quantile` on `sweep_rank`, with Python's measure aliases and
  `not_two`.
- Noise policy: `drop`, `as_cluster` and `singleton`. Under `drop` the training rows are re-sliced
  after the policy is applied, and resamples that are all noise are removed before aggregation.
- Anchored consensus: above `anchor_threshold`, consensus quantities are computed over an anchor
  subset of `consensus_anchors` samples, and `get_labels` extends the anchor cut to every sample
  with the classifier.
- Neighbor scaling: `n_neighbors` of spectral, Leiden and Louvain is scaled to each subsample's
  share of the rows, rounded and floored at 2. Preprocessing neighbor parameters are not scaled.
- Randomized preprocessing: pipelines are allocated across resamples as `allocate_pipelines`
  does, each is fit on its subsample, and the classifier trains on raw features.
  `preprocessing_results()` has Python's columns.
- `get_labels` noise labels: the quantile cutoff, the per-sample margin guard (`NOISE_MARGIN`),
  NaN scores counted as noise, and the warning when fewer samples than the quantile's target are
  flagged.
- Validation: `subsample_ratio`, `n_resamples` and `n_trees` are checked at fit time, and the
  `source = "accuracy"` guard is ported.
- Messages: error and warning text is copied from Python, so a user moving between the two
  packages sees the same wording.
- Verbose output: ports of `_print_run_header`, `_log_config_progress` and `_print_run_footer`.
  The header also prints the projected memory of the retained consensus blocks.
- Default grids: the `light` and `full` presets of `_grids.py` for each sweep axis, with the same
  warnings when an option is omitted.

### Known differences from Python

- Results agree statistically, not bit for bit. R's RNG differs from numpy's, so the subsamples
  differ.
- KMeans seeding is an R implementation of k-means++. It follows sklearn's algorithm but draws
  different random numbers.
- Leiden runs in igraph's C implementation, Python uses `leidenalg`. Both optimize the same
  objective, but their partitions differ, and so do CARVE's scores: on `pbmc3k_subset`, R's Leiden
  scores at resolutions 0.75 and 1 are 0.008 to 0.029 higher (stage 3 report).
- Consensus blocks take twice the memory (decision above). For example, 20 configurations at
  5,000 anchors hold about 8 GB in R and 4 GB in Python.
- HDBSCAN: `dbscan::hdbscan` builds the single-linkage tree, and R condenses it and selects
  clusters with a port of scikit-learn's code, so `eom` and `leaf` both follow scikit-learn. Labels
  can still differ where mutual reachability distances tie, because dbscan merges tied samples in a
  different order.
- Reference labels containing -1: Python counts -1 as a reference cluster and can map a real
  cluster onto it (not yet fixed in Python; see the open API issues note). R ports the fixed
  behavior: reference clusters are counted with `count_clusters`, and labels are matched only onto
  classes 0 and up.
- `vignette("python-users")` lists every difference a user can see, the plot differences among
  them.

## Testing and parity

- Unit tests: testthat edition 3, one file per R file. Where behavior is shared, assertions are
  ported from the matching `tests/test_<module>.py`. Each new assertion is checked once by breaking
  the code it covers and confirming that the test fails. An unexpected warning fails the suite,
  the R counterpart of `filterwarnings = error`.
- Golden fixtures: `data-raw/make_fixtures.py` runs deterministic Python functions on fixed inputs
  and writes the outputs as JSON or CSV to `tests/testthat/fixtures/`. The functions covered are:
  consensus matrices from given runs; Gini, CE and PAC; anchored consensus from given anchors; ARI;
  label alignment; `noise_mask`; the selection rules on a fixed results table;
  `resolve_core_budget`; `scale_neighbor_count`; sweep resolution; generalizability scores. R
  compares to 1e-10, or 1e-6 where Python computes in float32. The fixtures are committed, so R
  CI never needs Python. Re-running the script after a Python change shows where R has fallen
  behind.
- End-to-end parity: `data-raw/parity_check.R` uses reticulate to fit both packages on the same
  data and writes a short report. The cases are easy and hard blobs, circles, a resolution sweep on
  PBMC 3k principal components, and one randomized-preprocessing run. The report compares the
  selected k or resolution, the metric curves within stated tolerances, and the ARI between the R
  and Python labels. It runs at the end of each stage and is not part of CI. It replaces the public
  `cross-validation` vignette.
- Gates: CI runs `R CMD check --no-manual` with warnings as errors (`r-ci.yml`); stage 4 added a
  pandoc step so that the vignettes build. The final stage installs BiocCheck and runs it once as
  a report; it gates nothing until the author decides to submit.

## Documentation

### Reference and package files

- Roxygen pages for every export, generated with roxygen2.
- `README.Rmd`, rendered to `README.md`: installation, a ten-line example, links to the vignettes.
- `NEWS.md`: a 2.0.0 entry listing the API break and what replaced each old function.

### Vignettes

Each builds inside `R CMD check` in under a minute.

1. `CARVE.Rmd` (named after the package, as Bioconductor expects): matrix input, stability and
   generalizability, the results table, the selection rules, the `get_*` functions, the main
   plots.
2. `single-cell.Rmd`: SingleCellExperiment and Seurat, resolution sweeps with Leiden and Louvain,
   `run_carve` and `attach_results`, per-cell scores, plots on objects, anchors at large n, noise
   labels. It uses `pbmc3k_subset`, and `SeuratObject::pbmc_small` for Seurat objects; those chunks
   run only when SeuratObject is installed.
3. `customizing.Rmd`: custom grids and estimator functions, the HDBSCAN sweep with a noise policy,
   a custom classifier, randomized preprocessing with `plot_metric_by_pipeline`, `n_jobs`,
   `BPPARAM` and memory, reference labels.
4. `python-users.Rmd`: a table from Python names to R names, the defaults, and where and why
   results differ.

### Tutorial

`notebooks/R_Tutorial.Rmd`, committed with its rendered HTML, as the Python notebooks are committed
with their outputs. It is not built by `R CMD check`, so it can run full-size fits.

- Part 1 follows `Tutorial.ipynb` on simulated data: easy and hard blobs, every `get_*` function,
  seven of the eight plots (`plot_n_clusters_over_sweep()` stops with an error on an `n_clusters`
  sweep, so part 2 shows it on the resolution sweep), the selection rules and `not_two`, the
  fixed-k override, custom grids, `saveRDS`, reference labels, parallelism.
- Part 2 follows `Resolution_Tutorial.ipynb` on PBMC 3k. The 10x download is cached in
  `notebooks/data/` and processed with the standard Seurat workflow. It covers the Leiden and
  Louvain sweep, `sweep_value`, `consensus_k`, HDBSCAN, `run_carve` and `attach_results` on Seurat
  and SingleCellExperiment objects, randomized preprocessing and noise labels.

A check script, `data-raw/check_exports_used.R`, fails if any export is used in neither the
tutorial nor a vignette.

## Writing standard

The author ranks this above every other requirement. It covers roxygen text, README, NEWS,
vignettes, the tutorial and code comments.

- Use the `de-ai-writing` skill, installed at `~/.claude/skills/de-ai-writing/` with its
  catalogue in `references/signs.md`. Load it before writing any user-facing text. Writing plainly
  from the first draft is the goal; the cleanup pass is a check.
- Follow CLAUDE.md: American spelling, no bold or italics, plain prose, sentence-case headings.
- Every document gets a final pass. The scanner script the skill refers to
  (`scripts/check_ai_signs.py`) was not supplied, so the pass is a full read against the catalogue
  plus a grep check for the section 2.1 vocabulary, em dashes, emphasis markup and Title Case
  headings. The target is zero hits; any hit kept on purpose is listed in the stage report.
- Error and warning messages copied from Python keep Python's wording.
- Nothing from the old README, NEWS or vignettes is carried over without being rewritten.

## Staging

All work happened on `r-package-rework`. There were four plans, each written in its own
conversation, and each ended with `R CMD check` green and the parity script run:

1. Core on the k axis: the S4 class and validity, `carve()` for `matrix` and `data.frame`, the
   runner with seeds and the core budget, exact consensus, selection, `get_*` and the accessors,
   KMeans, AgglomerativeClustering and SpectralClustering, the classifier, the fixture script. The
   old R code and tests were removed at the start.
2. Other axes and scale: the `resolution` and `min_cluster_size` sweeps, LeidenClustering,
   LouvainClustering and HDBSCAN, the noise policy and noise labels, anchored consensus,
   randomized preprocessing.
3. Plots and single-cell: the eight plots, the `SingleCellExperiment` methods and the `ANY` methods
   for Seurat objects, `run_carve`, `attach_results`, `pbmc3k_subset`.
4. Documentation and tutorial: roxygen text, README, NEWS, the four vignettes, the tutorial and its
   rendered HTML, the writing pass, the BiocCheck report, the final parity report.

`main` receives the branch only after stage 4, so it never carries a half-ported package.

## Out of scope

- `carve.sim` and `benchmarks`.
- Manuscript edits. The manuscript's install line needs the correction described under the package
  before the rework; that edit is the author's.
- The Python fix for reference labels containing -1. It is a separate change; until it lands, R and
  Python differ on such inputs.
- A Bioconductor or CRAN submission.
- The nightly notebook `OMP_NUM_THREADS` pin.
- The DESCRIPTION maintainer address, which stays as it is unless the author changes it.

## Open items for the plans, and how they were settled

1. HDBSCAN leaf selection (stage 2): R ports scikit-learn's condensed tree and selection onto
   dbscan's tree, so the full preset keeps `leaf`.
2. Warnings as test failures (stage 1): `tests/testthat/setup.R` sets `warn = 2`, and
   `expect_warning()` still works under it.
3. Seurat dispatch (stage 3): `ANY` methods that check `methods::is(x, "Seurat")`.
4. k-means++ (stage 1): implemented and checked against scikit-learn on fixtures.
5. Code worth keeping (stage 1): the old R code was removed at the start of stage 1, and the new
   ports were checked against the Python fixtures.
6. PBMC 3k download (stage 3): the 10x Genomics URL in `data-raw/pbmc3k_subset.R`, cached in
   `data-raw/cache/` for the data script and `notebooks/data/` for the tutorial, both ignored by
   git.
7. Estimator defaults (stages 1 and 2): read from `cluster.py` and the scikit-learn classes, and
   given on the help pages.
