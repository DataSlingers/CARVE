# Open API issues in the Python and R packages

Date: 2026-10-08. These came up while porting the package to R (branch
`r-package-rework`). The first three behave the same way in both packages
today. The author chose to keep them aligned and to fix each issue later in
both. The fourth and fifth are specific to one package.

## An invalid rule removes the selection line without an error

The metric plots draw a dashed line at the selected configuration. Python
computes it inside `try: ... except Exception: pass` in
`src/carve/_plotting.py` (about line 631), and R inside `tryCatch()` in
`draw_metric_lines()`, `carve-r/R/plotting.R` (about line 237). An invalid
`rule`, such as `"1SE"`, fails inside that block, so the plot is drawn
without the line and without an error. R already checks `measure` before
drawing for the same reason.

Fix in both: check `rule` before drawing, with the message the selection
functions give for an unknown rule.

## run_carve() fits before it finds that measure and mode disagree

`tl.carve(adata, mode="generalizability")` and
`run_carve(object, mode = "generalizability")` keep the default
`measure="stability"`. They run the whole fit and then fail in the
selection with "No configuration has a value for ari_stability." The R
help page of `run_carve()` says to pass a generalizability measure with
that mode.

Fix in both: before fitting, check that `measure` names a criterion that
`mode` computes, and fail with a message that says so.

## attach_results() does not take its mode from the fit

`tl.attach_results(adata, model)` and `attach_results(object, fit)`
default `mode` to `"default"`, whatever mode the fit ran with, so a fit
run with `mode="stability"` or `"generalizability"` needs the mode passed
again.

Checked in R on 2026-10-08, with a five-resample fit of three Gaussian
blobs and n_clusters 2 to 4. For a fit run with `mode = "stability"`,
`attach_results(object, fit)` does not fail. It writes the same columns as
`mode = "stability"` (`carve`, `carve_stability`, `carve_stability_ce`),
but the stored record has `params$mode` equal to `"default"`, which does
not describe the fit. For a fit run with `mode = "generalizability"`, the
call fails with "No configuration has a value for ari_stability." The call
with `measure = "generalizability"` and the default `mode` fails with
"Consensus matrix not available for mode='default'. This run likely used
split-mode and skipped building that artifact." Passing both
`mode = "generalizability"` and a generalizability measure is the working
form.

Fix in both: default `mode` to the mode the fit ran with.

## Reference labels containing -1 (Python only)

A `reference_labels` array that contains -1 makes Python count -1 as a
cluster, and the alignment can then map a real cluster to -1, even with
`noise_labels=False`. Examples are a reference with NaN values factorized by
pandas, a `tl.carve` `reference_key` column with NaN, categorical codes, or a
user estimator that emits -1. In `CARVE.get_labels`, `src/carve/api.py`
line 1009 sets `ref_k = int(np.unique(ref).size)`, and line 1014 calls
`align_cluster_labels(ref, labels)`, which takes every value of the
reference, -1 included, as a target class.

R counts reference clusters with `count_clusters()` (`carve-r/R/utils.R`),
which skips negative labels, and `get_labels()` in `carve-r/R/accessors.R`
aligns with `keep = reference >= 0`, so labels are matched only onto
reference labels 0 and up. The two packages differ on such input until
Python is fixed.

This was found during the review of the noise labels work (2026-10-06) and
left for a separate change.

Fix in Python: count `ref_k` with `count_clusters` and match only onto
classes 0 and up.

## Forked workers crash after Seurat's PCA (R only)

With the default backend, BiocParallel's `MulticoreParam`, which forks the
R session on macOS and Linux, the tutorial's randomized-preprocessing fit
on the 2,638 by 2,000 PBMC expression matrix crashed after
`Seurat::RunPCA()` had run in the same session. The error was "wrong args
for environment subassignment" in BiocParallel's `.manager_recv`, followed
by a segfault.

What was observed while writing the R tutorial:

- The same fit ran in a clean session with `n_jobs` 4 and 10.
- It ran in a session with Seurat loaded but without `ScaleData()` and
  `RunPCA()`.
- Dropping `ScaleData()` and `RunPCA()` fixed it. Dropping
  `FindNeighbors()`, `FindClusters()` and `RunUMAP()` did not.

The cause is not isolated. The tutorial passes
`BPPARAM = BiocParallel::SnowParam(workers = n_jobs)` on that fit and says
why.

Question for the author: should a fix document the workaround on the help
page of `carve()`, or choose a socket backend by default on macOS, or is
the tutorial's note enough?
