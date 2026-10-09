# Open API issues in the Python and R packages

Date: 2026-10-08. These came up while porting the package to R (branch
`r-package-rework`). The first three behave the same way in both packages
today. The author chose to keep them aligned and to fix each issue later in
both. The fourth is specific to R and was found while writing the tutorial.

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
help page now says to pass a generalizability measure with that mode.

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
form. I did not run the Python function.

Fix in both: default `mode` to the mode the fit ran with.

## Forked workers crash after Seurat's PCA (R only)

With the default backend, BiocParallel's `MulticoreParam`, which forks the
R session on macOS and Linux, the tutorial's randomized-preprocessing fit
on the 2,638 by 2,000 PBMC expression matrix crashed after
`Seurat::RunPCA()` had run in the same session. The error was "wrong args
for environment subassignment" in BiocParallel's `.manager_recv`, followed
by a segfault.

What was observed in Task 11:

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
