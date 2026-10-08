# R Package Rework, Stage 4 (Documentation and Tutorial) Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: finish the R package for the merge to `main`: close the code items the earlier stages left for this one, write the four vignettes, the README and NEWS, the tutorial `notebooks/R_Tutorial.Rmd` with its rendered HTML, bring the spec up to date, and run the final writing pass, `R CMD check`, the BiocCheck report and the parity script.

Architecture: no new module. Three small code tasks touch `sce.R`, `seurat.R`, `tl.R`, `estimators.R`, `plots.R` and `data.R`, each with tests. The vignettes live in `carve-r/vignettes/` and build inside `R CMD check`; the tutorial lives in `notebooks/` beside the Python notebooks and is rendered once, by hand, with the installed package. A script in `data-raw/` fails when an export appears in neither a vignette nor the tutorial.

Tech stack: R 4.5.1 (DESCRIPTION floor 4.1.0), knitr 1.51, rmarkdown 2.31, pandoc 3.2 from RStudio's bundle, roxygen2 8.0.0, testthat 3.3.2 (edition 3), devtools 2.5.2, SingleCellExperiment 1.30.1, SeuratObject 5.4.0 and Seurat 5.5.0 (tutorial only), mclust (tutorial only), BiocManager 1.30.27 for BiocCheck; Python 3.12 in `code/.venv` with carve 1.0.0 for the parity script.

Spec: `docs/superpowers/specs/2026-10-06-r-package-rework-design.md` (commit 0b8d5b4). This plan implements stage 4 of its four stages and builds on stages 1 to 3 as executed (plans `docs/superpowers/plans/2026-10-06-r-package-rework-stage1.md`, `-stage2.md` and `2026-10-07-r-package-rework-stage3.md`; the branch at c19fbf3). Executors read the spec and this plan. Where this plan departs from the spec, the departure is listed under Decisions, and Task 12 edits the spec to match.

## Global Constraints

- Branch: `r-package-rework` in the `code/` repository, at c19fbf3 plus this plan's commit. The author's checkout at `/Users/kaiwycik/GitHub/CARVE/code` is on `main` and stays there: all work happens in the worktree the Workspace section creates. Run `git -C "$REPO" branch --show-current` immediately before every commit and confirm it prints `r-package-rework`. Commit after every task. Stage files by explicit path only; never `git add -A`, `git add .` or `git commit -a`.
- Files outside `carve-r/` this plan may change: `notebooks/R_Tutorial.Rmd` and `notebooks/R_Tutorial.html` (created), `.github/workflows/r-ci.yml`, the spec, and `docs/superpowers/notes/2026-10-08-open-api-issues.md` (created). Nothing else: not `src/`, `tests/`, the other notebooks, or `../overleaf/`.
- Every shell block sets the variables it uses, from the preamble of the Workspace section, written out in full. Shell state does not carry from one command to the next.
- Pandoc is not installed system-wide. RStudio's copy works when `RSTUDIO_PANDOC` points at it (checked while writing this plan: `rmarkdown::pandoc_available()` is `TRUE`, version 3.2). Every build, render or check sets it.
- Vignettes, the README and the tutorial load the installed package from the private library `$RLIB`, never the CARVE 1.0.0 in the system library. After any change under `R/`, reinstall: `R CMD INSTALL --no-test-load --library="$RLIB" .` from `carve-r/`.
- Dependencies changed in this stage: floors `igraph (>= 1.3.0)`, `Rtsne (>= 0.15)` and `SeuratObject (>= 5.0.0)` (Task 4); Suggests `knitr` and `rmarkdown` and `VignetteBuilder: knitr` (Task 5); a `cph` entry and a `Copyright` field (Task 2). Nothing else. Seurat is not added (Decision 3). Keep Imports and Suggests in the case-insensitive alphabetical order stage 1 used.
- Exports: none added or removed.
- The engine does not change. The only code changes are those of Tasks 1 to 3, and none alters a fit's numbers (Decision 15).
- Writing: load the `de-ai-writing` skill before writing or changing any text a user reads (roxygen, comments, vignettes, README, NEWS, tutorial, the spec, the notes file). The text in this plan was written to that standard; keep it as written unless a rendered output contradicts it. Rules: American spelling; no bold or italics (no `**`, `__`, or `*word*` and `_word_` emphasis); sentence-case headings; no em or en dashes; none of the section 2.1 vocabulary of `~/.claude/skills/de-ai-writing/references/signs.md`; straight quotes only. Error and warning messages copied from Python keep Python's wording.
- ASCII only in `R/`, `tests/`, `vignettes/` and `notebooks/R_Tutorial.Rmd` (an `R CMD check` warning for the first two, and curly quotes are an AI sign in the others). Write a non-ASCII character in an R string as an escape such as `"\u2248"`.
- Vignettes: output `rmarkdown::html_vignette`; each renders in under 60 seconds (measured in the task that writes it); chunks that need a suggested package carry `eval = requireNamespace("<package>", quietly = TRUE)`; only `library(CARVE)` is attached, and every other package is called with `::` (the ggplot2, SeuratObject and SingleCellExperiment binaries here were built under R 4.5.2, and `library()` warns about that on R 4.5.1); each ends with a "Session information" section calling `sessionInfo()`.
- Prose that describes output: every sentence in a vignette, the README or the tutorial that states what an output shows must hold in the rendered output. Numbers come from inline R (`` `r get_k(fit)` ``) wherever they can. After rendering, knit the document to Markdown in a scratch directory, read it, and rewrite any sentence the output contradicts, keeping the writing rules.
- Tests: unexpected warnings fail the suite (`tests/testthat/setup.R` sets `warn = 2`). Every task report shows `SKIP 0`. Assertions must be able to fail: after a task's tests pass, apply the mutations its last step names, one at a time, confirm the named test fails, and restore with `git checkout -- <file>`. Report each result.
- No `print()` or `cat()` in `R/` outside `show` and `print` methods; diagnostics use `warning(..., call. = FALSE)`; no logging package.
- The plan's code and text were written without being run, except the checks recorded under Decisions. Treat them as a detailed proposal. When plan code or a plan test is wrong, fix it, say so in the report with the failing output and why, and keep the assertion's intent. Never loosen an assertion to make it pass against code that disagrees with it.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Decisions this plan makes that the spec left open or changes

Items 4 to 7 are the author's rulings of 2026-10-08. Items 1, 3 and 8 change the spec's text and are carried into it by Task 12.

1. Pandoc. Locally, RStudio's bundled pandoc through `RSTUDIO_PANDOC`. In CI, `r-lib/actions/setup-pandoc@v2` is added to `r-ci.yml`; without it the vignettes cannot build there. The spec said CI stays unchanged.
2. Vignette format: `rmarkdown::html_vignette`, which needs no new dependency. Bioconductor packages often use BiocStyle; the BiocCheck report (Task 13) shows whether it asks for it, and the author decides at submission time.
3. Seurat stays out of Suggests, reversing stage 3's Decision 16. No package code and no vignette calls it: the single-cell vignette uses `SeuratObject::pbmc_small` (80 cells with `pca` and `tsne` reductions; `run_carve()` on it takes about 1 s, checked while writing this plan), and SeuratObject is already suggested. The tutorial, outside the build, uses Seurat for the standard workflow.
4. LeidenClustering help text (author): "igraph's Leiden and leidenalg are separate implementations, so the partitions, and the scores CARVE derives from them, differ somewhat from the Python package's." followed by a pointer to `vignette("python-users")`, which gives the measured differences.
5. HDBSCAN port credit (author): `person("The scikit-learn developers", role = "cph", ...)` in Authors@R, a `Copyright: See file inst/COPYRIGHTS.` field, `inst/COPYRIGHTS` with scikit-learn's BSD-3 notice copied from `code/.venv/lib/python3.12/site-packages/scikit_learn-1.7.2.dist-info/licenses/COPYING`, and a comment above the ported functions.
6. Three behaviors stay as Python has them (author): plots still drop the dashed selection line silently when `rule` is invalid, `run_carve()` still fits before finding that `measure` needs a criterion `mode` skipped, and `attach_results(mode =)` still defaults to `"default"`. Task 12 records them in `docs/superpowers/notes/2026-10-08-open-api-issues.md` as open issues to fix later in both packages.
7. The spec is edited in place (author), not amended with an appendix (Task 12).
8. Version floors, checked while writing this plan against the CRAN archive sources: igraph 1.3.0's `cluster_louvain()` has `resolution` and processes vertices in random order, so seeds take effect; its `cluster_leiden()` names the argument `resolution_parameter`, which `resolution =` reaches by partial matching because that version's signature has no `...`; Rtsne 0.15 has every argument `TSNE()` passes (`Y_init`, `stop_lying_iter`, `mom_switch_iter`, `exaggeration_factor`, `eta`, `num_threads`); SeuratObject 5.0.0 exports `Layers()`, `LayerData()` and `JoinLayers()`. The floors are declared, not tested, as the Python package's are.
9. Lists in error messages are sorted in code point order, `sort(x, method = "radix")`, as Python's `sorted()` sorts them; `sort()` follows the collation locale. The seven sites are in `sce.R`, `seurat.R` and `tl.R`.
10. A Seurat v5 assay whose normalized data are split across layers (`data.a`, `data.b` after `split()`) gets its own error, which names the layers and points to `SeuratObject::JoinLayers()`. Before, it got "has no 'data' layer. Normalize it first", which is wrong for that case (checked while writing this plan).
11. The per-sample help pages and the consensus page draw from two blobs two units apart with standard deviation 0.6. Measured while writing this plan (10 resamples, KMeans k = 2 to 4): k = 2 is selected, the Gini scores have mean 0.979, SD 0.082 and minimum 0.36, and 40 of 60 samples score below 0.99. The old blobs, three units apart with SD 0.3, give every sample a Gini score of 1.
12. The export check is `carve-r/data-raw/check_exports_used.R`, run by hand. It is not in CI, because the tutorial is outside the package.
13. The tutorial caches the 10x download in `notebooks/data/`, which `code/.gitignore` already ignores (its `data/` line). It fits at full size with `n_jobs` and is rendered once in Task 11.
14. BiocCheck is installed into the private library `$RLIB`, not the user library, and its report is committed as `carve-r/data-raw/bioccheck-report.md`, beside the parity report. It gates nothing.
15. Parity. Tasks 1 and 2 change only error text, an error path and a comparison that dbscan's trees never reach, so the parity script must reproduce `data-raw/parity-report.md` exactly apart from its date line. Any other change is a defect to find before the merge.
16. Labels in the documents. `get_labels()` returns 1 to k, and the vignettes and tutorial say so where Python users would expect 0 to k - 1.

## File structure

| File | Task | Change |
|---|---|---|
| `carve-r/tests/testthat/helper-locale.R` | 1 | Create: `local_collation()` for tests of sorted lists |
| `carve-r/R/sce.R`, `R/seurat.R`, `R/tl.R` | 1 | Radix sort in error lists; split-layer error |
| `carve-r/tests/testthat/test-sce.R`, `test-seurat.R`, `test-tl.R` | 1 | Tests for the above; the pinned-resolution test |
| `carve-r/R/estimators.R` | 2 | NaN guard in `select_hdbscan_clusters()`; credit comment; Leiden and HDBSCAN help text |
| `carve-r/tests/testthat/test-estimators.R` | 2 | Test of the NaN guard |
| `carve-r/inst/COPYRIGHTS` | 2 | Create: scikit-learn's BSD-3 notice |
| `carve-r/DESCRIPTION` | 2, 4, 5 | cph and Copyright; floors; knitr, rmarkdown, VignetteBuilder |
| `carve-r/R/plots.R`, `R/data.R` | 3 | Example data; pbmc3k_subset page |
| `carve-r/.gitignore` | 4 | Ignore rendered vignette output |
| `.github/workflows/r-ci.yml` | 4 | Pandoc step |
| `carve-r/data-raw/check_exports_used.R` | 4 | Create: the export check |
| `carve-r/vignettes/CARVE.Rmd` | 5 | Create |
| `carve-r/vignettes/single-cell.Rmd` | 6 | Create |
| `carve-r/vignettes/customizing.Rmd` | 7 | Create |
| `carve-r/vignettes/python-users.Rmd` | 8 | Create |
| `carve-r/README.Rmd`, `README.md`, `NEWS.md`, `R/CARVE-package.R` | 9 | Rewrite |
| `notebooks/R_Tutorial.Rmd` | 10, 11 | Create |
| `notebooks/R_Tutorial.html` | 11 | Create (rendered) |
| the spec, `docs/superpowers/notes/2026-10-08-open-api-issues.md` | 12 | Edit; create |
| `carve-r/data-raw/parity-report.md`, `carve-r/data-raw/bioccheck-report.md` | 13 | Regenerate; create |
| `carve-r/man/*.Rd`, `NAMESPACE` | 1 to 3, 9 | Regenerated by `devtools::document()` only |

## Workspace

Run once, before Task 1.

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
git branch --show-current        # main; this checkout is left alone
git worktree list                # r-package-rework must not be checked out anywhere yet
grep -qx ".worktrees/" .git/info/exclude || echo ".worktrees/" >> .git/info/exclude
git worktree add .worktrees/r-package-rework r-package-rework
mkdir -p .worktrees/r-package-rework/.Rlib
grep -qx ".Rlib/" .git/info/exclude || echo ".Rlib/" >> .git/info/exclude
```

`.git/info/exclude` is the repository's local ignore file, shared by all its worktrees, so neither directory shows up in `git status` and no tracked `.gitignore` changes. Then install the package into the private library and check that it loads from there:

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
R CMD INSTALL --no-test-load --library="$RLIB" .
Rscript -e 'cat(format(packageVersion("CARVE")), find.package("CARVE"), rmarkdown::pandoc_available(), "\n")'
```

Expected: `2.0.0 /Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework/.Rlib/CARVE TRUE`. The first five lines are the preamble every later shell block repeats.

---

### Task 1: Sorted error lists, the split-layer error and the pinned-resolution test

Files:
- Create: `carve-r/tests/testthat/helper-locale.R`
- Modify: `carve-r/R/sce.R` (lines 25, 33, 92), `carve-r/R/seurat.R` (`seurat_data_layer()` at lines 23-31; lines 43, 51, 106), `carve-r/R/tl.R` (line 236)
- Test: `carve-r/tests/testthat/test-sce.R`, `test-seurat.R`, `test-tl.R`

Interfaces:
- Consumes: `sce_matrix()`, `sce_basis()`, `seurat_matrix()`, `seurat_basis()`, `run_carve()`, `python_list()`, and the test helpers `make_sce()`, `make_seurat()`, `blobs`, `sce` and `run()` that the three test files already define.
- Produces: `local_collation(locale = "en_US.UTF-8", envir = parent.frame())` in the test helpers. No change to any exported signature.

- [ ] Step 1: Write the locale helper

Create `carve-r/tests/testthat/helper-locale.R`:

```r
# Sets the collation locale until the calling test ends. R's sort() follows
# it and Python's sorted() does not. Under the C locale the two orders agree,
# so tests of the lists in error messages set a locale where they differ.
# The test is skipped when the locale is not installed.
local_collation <- function(locale = "en_US.UTF-8", envir = parent.frame()) {
  old <- Sys.getlocale("LC_COLLATE")
  if (!nzchar(suppressWarnings(Sys.setlocale("LC_COLLATE", locale)))) {
    testthat::skip(sprintf("Collation locale %s is not installed.", locale))
  }
  withr::defer(Sys.setlocale("LC_COLLATE", old), envir = envir)
}
```

- [ ] Step 2: Write the failing tests

Append to `carve-r/tests/testthat/test-sce.R`:

```r
test_that("the lists in the error messages are in code point order", {
  local_collation()
  X <- blobs$X
  # Inserted in the locale's order, so neither an unsorted nor a
  # locale-sorted list matches.
  mixed <- make_sce(X, reductions = list(harmony = X, PCA = X), assays = list(logcounts = t(X), SCT = t(X)))
  expect_error(sce_matrix(mixed, assay = "counts"), "Available assays: ['SCT', 'logcounts']", fixed = TRUE)
  expect_error(sce_matrix(mixed, reduction = "UMAP"), "Available names: ['PCA', 'harmony']. Pass assay=", fixed = TRUE)
  expect_error(sce_basis(mixed, basis = "UMAP"), "Available names: ['PCA', 'harmony']", fixed = TRUE)
})
```

Append to `carve-r/tests/testthat/test-seurat.R`:

```r
test_that("the lists in the error messages are in code point order", {
  local_collation()
  X <- blobs$X
  object <- make_seurat(X, reductions = list(harmony = X, UMAP = X))
  object[["integrated"]] <- SeuratObject::CreateAssay5Object(
    counts = SeuratObject::LayerData(object, assay = "RNA", layer = "counts")
  )
  expect_error(seurat_matrix(object, assay = "ADT"), "Available assays: ['RNA', 'integrated']", fixed = TRUE)
  expect_error(seurat_matrix(object, reduction = "tsne"), "Available names: ['UMAP', 'harmony']. Pass assay=", fixed = TRUE)
  expect_error(seurat_basis(object, basis = "tsne"), "Available names: ['UMAP', 'harmony']", fixed = TRUE)
})

test_that("a data layer split across layers gets the JoinLayers() hint", {
  object <- make_seurat(blobs$X, reductions = list())
  object[["RNA"]] <- split(object[["RNA"]], f = rep(c("a", "b"), length.out = ncol(object)))
  expect_error(
    seurat_matrix(object),
    "Assay 'RNA' keeps its normalized data in split layers ['data.a', 'data.b']. Join them first with SeuratObject::JoinLayers(), or pass reduction=.",
    fixed = TRUE
  )
  expect_error(seurat_matrix(object, assay = "RNA"), "Join them first with SeuratObject::JoinLayers()", fixed = TRUE)
  joined <- SeuratObject::JoinLayers(object)
  expect_equal(unname(seurat_matrix(joined)), blobs$X)
})
```

`split()` dispatches to SeuratObject's method for an `Assay5`; checked while writing this plan, the layers become `counts.a counts.b data.a data.b`, and `JoinLayers()` brings back `data` and `counts`. If the joined matrix comes back in another cell order, compare after reordering its rows by `colnames(object)` and say so in the report.

In `carve-r/tests/testthat/test-tl.R`, append:

```r
test_that("reference_key's error lists the columns in code point order", {
  local_collation()
  mixed <- sce
  SummarizedExperiment::colData(mixed)$batch <- 1
  SummarizedExperiment::colData(mixed)$Truth <- blobs$y
  expect_error(run(mixed, reference_key = "nope"), "Available columns: ['Truth', 'batch']", fixed = TRUE)
})
```

and in the test "k pins the configuration and a resolution run records its axis", replace the last three lines

```r
  graph <- run_carve(sce, n_resamples = 4, estimator_param_grids = list(estimator_grid(LeidenClustering, resolution = c(0.5, 1))))
  params <- S4Vectors::metadata(graph)$carve$params
  expect_identical(params$sweep_param, "resolution")
  expect_true(params$selected_resolution %in% c(0.5, 1))
```

with

```r
  graph_grid <- list(estimator_grid(LeidenClustering, resolution = c(0.5, 1)))
  for (value in c(0.5, 1)) {
    graph <- run_carve(sce, n_resamples = 4, estimator_param_grids = graph_grid, sweep_value = value)
    params <- S4Vectors::metadata(graph)$carve$params
    expect_identical(params$sweep_param, "resolution")
    expect_equal(params$selected_resolution, value)
  }
```

The old line passed whichever resolution was recorded. Pinning each value in turn ties the record to the selected row.

- [ ] Step 3: Run the new tests and watch them fail

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'devtools::test(filter = "sce|seurat|tl")' 2>&1 | tail -40
```

Expected: the three "code point order" tests fail, with the lists in the locale's order (for example `['logcounts', 'SCT']`), and the split-layer test fails with "Assay 'RNA' has no 'data' layer. Normalize it first". The pinned-resolution test passes already; Step 6 shows it can fail.

- [ ] Step 4: Sort in code point order and add the split-layer error

In `carve-r/R/sce.R`, change the three calls `python_list(sort(assays))`, `python_list(sort(reductions))` (lines 25, 33) and `python_list(sort(reductions))` (line 92) to `python_list(sort(assays, method = "radix"))` and `python_list(sort(reductions, method = "radix"))`. In `carve-r/R/seurat.R`, change the same three calls at lines 43, 51 and 106. In `carve-r/R/tl.R` line 236, change `python_list(sort(cells_column_names(object)))` to `python_list(sort(cells_column_names(object), method = "radix"))`.

In `carve-r/R/seurat.R`, replace `seurat_data_layer()` with:

```r
seurat_data_layer <- function(object, assay) {
  layers <- SeuratObject::Layers(object[[assay]])
  if (!"data" %in% layers) {
    split <- grep("^data[.]", layers, value = TRUE)
    if (length(split) > 0L) {
      stop(sprintf(
        "Assay '%s' keeps its normalized data in split layers %s. Join them first with SeuratObject::JoinLayers(), or pass reduction=.",
        assay, python_list(split)
      ), call. = FALSE)
    }
    stop(sprintf(
      "Assay '%s' has no 'data' layer. Normalize it first, for example with Seurat::NormalizeData(), or pass reduction=.",
      assay
    ), call. = FALSE)
  }
  cells_by_features(SeuratObject::LayerData(object, assay = assay, layer = "data"))
}
```

The new message is R's own; Python has no Seurat path.

- [ ] Step 5: Run the tests

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'devtools::test(filter = "sce|seurat|tl")' 2>&1 | tail -5
Rscript -e 'devtools::test()' 2>&1 | tail -3
```

Expected: both runs `FAIL 0 | WARN 0 | SKIP 0`. The full suite was 1925 passing expectations at c19fbf3; it is now higher by the new expectations.

- [ ] Step 6: Mutations

Apply one at a time, run the named test file, confirm the named test fails, then restore with `git checkout -- <file>`:

1. `R/sce.R` line 25 back to `sort(assays)`: "the lists in the error messages are in code point order" in test-sce.R fails.
2. `R/sce.R` line 92 to `python_list(reductions)` (no sort): the same test fails.
3. `R/seurat.R` line 106 back to `sort(reductions)`: the code point test in test-seurat.R fails.
4. `R/seurat.R` line 43 back to `sort(assays)`: the same test fails.
5. `R/tl.R` line 236 back to `sort(cells_column_names(object))`: "reference_key's error lists the columns in code point order" fails.
6. In `seurat_data_layer()`, change `"^data[.]"` to `"^counts[.]"`: the split-layer test fails (the message names the counts layers).
7. In `R/tl.R`, change `params[[paste0("selected_", param)]] <- row[[param]][[1L]]` to `params[[paste0("selected_", param)]] <- fit@sweep@values[[1L]]`: "k pins the configuration and a resolution run records its axis" fails at `value = 1`.

- [ ] Step 7: Commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO"
git branch --show-current
git add carve-r/tests/testthat/helper-locale.R carve-r/R/sce.R carve-r/R/seurat.R carve-r/R/tl.R \
  carve-r/tests/testthat/test-sce.R carve-r/tests/testthat/test-seurat.R carve-r/tests/testthat/test-tl.R
git commit -m "$(cat <<'MSG'
fix(carve-r): list names in code point order and name split Seurat layers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: The HDBSCAN port's NaN comparison and credit, and the estimator help text

Files:
- Modify: `carve-r/R/estimators.R` (the Leiden help paragraph at lines 419-424; the HDBSCAN help paragraph at lines 527-533; the comment above `condense_tree()` at line 562; the comment above `select_hdbscan_clusters()` at lines 642-650; the comparison at line 672)
- Modify: `carve-r/DESCRIPTION` (Authors@R, a new Copyright field)
- Create: `carve-r/inst/COPYRIGHTS`
- Test: `carve-r/tests/testthat/test-estimators.R`
- Regenerated: `carve-r/man/LeidenClustering.Rd`, `carve-r/man/HDBSCAN.Rd`

Interfaces:
- Consumes: `select_hdbscan_clusters(tree, method)`, where `tree` is the list `condense_tree()` returns: `n_samples`, and the parallel vectors `parent`, `child`, `lambda`, `size`.
- Produces: no change to any signature.

- [ ] Step 1: Write the failing test

Append to `carve-r/tests/testthat/test-estimators.R`, after the HDBSCAN tests:

```r
test_that("a NaN stability compares as false in the eom selection, as in scikit-learn", {
  # The root, cluster 5, splits at distance 0 into clusters 6 and 7 of two
  # samples each. Their samples leave at the same lambda, Inf, at which the
  # clusters were born, so both stabilities are Inf - Inf = NaN. scikit-learn
  # compares NaN as false and keeps both clusters.
  tree <- list(
    n_samples = 4L,
    parent = c(5L, 5L, 6L, 6L, 7L, 7L),
    child = c(6L, 7L, 1L, 2L, 3L, 4L),
    lambda = rep(Inf, 6L),
    size = c(2L, 2L, 1L, 1L, 1L, 1L)
  )
  expect_identical(select_hdbscan_clusters(tree, "eom"), c(1L, 1L, 2L, 2L))
  expect_identical(select_hdbscan_clusters(tree, "leaf"), c(1L, 1L, 2L, 2L))
})
```

The tree is built by hand because dbscan's trees did not produce this case in 1,500 trials (stage 2 review). The leaf line passes before the fix; it pins that both selections agree here.

- [ ] Step 2: Run it and watch it fail

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'devtools::test(filter = "estimators")' 2>&1 | tail -15
```

Expected: the new test errors with "missing value where TRUE/FALSE needed".

- [ ] Step 3: Fix the comparison and credit the port

In `select_hdbscan_clusters()`, change

```r
      if (kids_stability > stability[cl]) {
```

to

```r
      if (isTRUE(kids_stability > stability[cl])) {
```

and add one sentence at the end of the comment block above the function (after "...or -1 when there is none."):

```r
# A NaN stability, from a split at distance 0, compares as false, as in
# scikit-learn.
```

Replace the first line of the comment above `condense_tree()`,

```r
# The condensed tree of scikit-learn's HDBSCAN, built from the single-linkage
# tree dbscan::hdbscan() returns; a port of _condense_tree() in
# sklearn/cluster/_hdbscan/_tree.pyx. dbscan condenses the same tree
```

with

```r
# condense_tree() and select_hdbscan_clusters() are ported from
# sklearn/cluster/_hdbscan/_tree.pyx in scikit-learn 1.7.2, copyright the
# scikit-learn developers, under the BSD 3-Clause license; inst/COPYRIGHTS
# has the notice.
#
# The condensed tree of scikit-learn's HDBSCAN, built from the single-linkage
# tree dbscan::hdbscan() returns; a port of _condense_tree() in
# sklearn/cluster/_hdbscan/_tree.pyx. dbscan condenses the same tree
```

Create `carve-r/inst/COPYRIGHTS`. Copy the license text exactly from `/Users/kaiwycik/GitHub/CARVE/code/.venv/lib/python3.12/site-packages/scikit_learn-1.7.2.dist-info/licenses/COPYING` (its first lines are `BSD 3-Clause License`, a blank line, `Copyright (c) 2007-2024 The scikit-learn developers.`, `All rights reserved.`). The file is:

```text
CARVE is distributed under the MIT license; see the file LICENSE.

The functions condense_tree() and select_hdbscan_clusters() in
R/estimators.R are ported from sklearn/cluster/_hdbscan/_tree.pyx in
scikit-learn 1.7.2. scikit-learn is distributed under the following
license.

<the full text of COPYING, unchanged>
```

Check the copied part with `diff <(tail -n +7 carve-r/inst/COPYRIGHTS) /Users/kaiwycik/GitHub/CARVE/code/.venv/lib/python3.12/site-packages/scikit_learn-1.7.2.dist-info/licenses/COPYING` (adjust the `+7` to the line where COPYING starts); expected: no output.

In `carve-r/DESCRIPTION`, replace the Authors@R field with

```
Authors@R: c(
    person("Kai", "Wycik", email = "kai.wycik@columbia.edu", role = c("aut", "cre")),
    person("The scikit-learn developers", role = "cph",
           comment = "HDBSCAN cluster selection in R/estimators.R, ported from scikit-learn")
  )
```

and add, after the `License:` line,

```
Copyright: See file inst/COPYRIGHTS.
```

The maintainer address stays as it is (spec, out of scope).

- [ ] Step 4: Correct the Leiden and HDBSCAN help text

Load the `de-ai-writing` skill first. In the LeidenClustering roxygen block, replace

```r
#' scale, so the two should not share a resolution grid. igraph's Leiden
#' implementation differs from leidenalg's in its random choices, so the
#' partitions agree with the Python package's in quality, not label by label.
```

with (author's wording, Decision 4)

```r
#' scale, so the two should not share a resolution grid. igraph's Leiden and
#' leidenalg are separate implementations, so the partitions, and the scores
#' CARVE derives from them, differ somewhat from the Python package's.
#' `vignette("python-users", package = "CARVE")` gives the measured
#' differences.
```

In the HDBSCAN roxygen block, replace

```r
#' tree the selection works on. How often this happens depends on the data:
#' agreement is common when few distances tie and can be rare when many do,
#' as with small integer counts. dbscan also computes all pairwise
#' distances, so memory grows with the square of the number of samples.
```

with

```r
#' tree the selection works on. How often this happens depends on the data.
#' dbscan also computes all pairwise distances, so memory grows with the
#' square of the number of samples.
```

The removed clause had no measurement behind it (stage 2 review). Then regenerate the pages:

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'devtools::document()'
git status --short
```

Expected: `R/estimators.R`, `DESCRIPTION`, `man/LeidenClustering.Rd`, `man/HDBSCAN.Rd`, `tests/testthat/test-estimators.R` and the new `inst/COPYRIGHTS`. If `document()` rewrote DESCRIPTION's layout, keep what it wrote (stage 1 ruling). `man/CARVE-package.Rd` changes too if roxygen lists the authors there; stage it.

- [ ] Step 5: Run the tests and the package metadata check

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'devtools::test()' 2>&1 | tail -3
Rscript -e 'a <- eval(parse(text = read.dcf("DESCRIPTION", "Authors@R")[1, 1])); print(a); cat(read.dcf("DESCRIPTION", "Copyright")[1, 1], "\n")'
R CMD INSTALL --no-test-load --library="$RLIB" .
Rscript -e 'cat(file.exists(system.file("COPYRIGHTS", package = "CARVE")), "\n")'
```

Expected: `FAIL 0 | WARN 0 | SKIP 0`; the two persons print, the second with `[cph]`; `See file inst/COPYRIGHTS.`; and `TRUE` (R installs `inst/COPYRIGHTS` at the package's top level).

- [ ] Step 6: Mutation

Apply one at a time and restore after each:

1. Remove the `isTRUE()` around the comparison: the new test fails with the error of Step 2.
2. Change `isTRUE(kids_stability > stability[cl])` to `!isFALSE(kids_stability > stability[cl])`, which treats the NaN comparison as true: both clusters are dropped for their (empty) children, and the eom expectation fails with `c(-1L, -1L, -1L, -1L)`.

- [ ] Step 7: Commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO"
git branch --show-current
git add carve-r/R/estimators.R carve-r/DESCRIPTION carve-r/inst/COPYRIGHTS carve-r/man/LeidenClustering.Rd \
  carve-r/man/HDBSCAN.Rd carve-r/tests/testthat/test-estimators.R
git status --short   # stage man/CARVE-package.Rd too if document() changed it
git commit -m "$(cat <<'MSG'
fix(carve-r): compare a NaN HDBSCAN stability as false and credit scikit-learn

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: Help-page examples that show a spread, and the pbmc3k_subset page

Files:
- Modify: `carve-r/R/plots.R` (the data line of five `@examples` blocks: lines 437, 667, 757, 870 and 976), `carve-r/R/data.R` (the description)
- Regenerated: `carve-r/man/plot_consensus_matrix.Rd`, `plot_cluster_boxplot.Rd`, `plot_cluster_violin.Rd`, `plot_cluster_scatter.Rd`, `plot_diagnostic_scatter.Rd`, `pbmc3k_subset.Rd`

Interfaces:
- Consumes: the plot generics and `carve()`, unchanged.
- Produces: nothing later tasks call.

- [ ] Step 1: Change the example data

In `carve-r/R/plots.R`, the examples of `plot_consensus_matrix` (line 437), `plot_cluster_boxplot` (667), `plot_cluster_violin` (757), `plot_cluster_scatter` (870) and `plot_diagnostic_scatter` (976) each start with

```r
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
```

In those five blocks only, change the second line to

```r
#' X <- rbind(matrix(rnorm(60, 0, 0.6), ncol = 2), matrix(rnorm(60, 2, 0.6), ncol = 2))
```

The other three plot pages draw curves over the sweep, which the old data show well; leave them. Confirm the line numbers with `grep -n "rnorm(60, 0, 0.3)" R/plots.R` first; eight lines match, and the five above are the ones under the five named functions' `@examples`.

- [ ] Step 2: Correct the pbmc3k_subset page

Load the `de-ai-writing` skill first. In `carve-r/R/data.R`, replace

```r
#' The steps ran on all 2,700 cells. A cell was kept if it had more than 200
#' and fewer than 2,500 detected genes, and less than 5 percent of its counts
#' came from mitochondrial genes. The counts were log-normalized, the 2,000
```

with

```r
#' The steps ran on all 2,700 cells. Genes detected in fewer than 3 cells
#' were dropped, and a cell was kept if it had more than 200 and fewer than
#' 2,500 detected genes and less than 5 percent of its counts came from
#' mitochondrial genes. The counts were log-normalized, the 2,000
```

and replace

```r
#' decimals.
#' The script is `data-raw/pbmc3k_subset.R` in the package's source
```

with

```r
#' decimals. Clusters 7 and 8 have 12 and 5 cells in the subset.
#' The script is `data-raw/pbmc3k_subset.R` in the package's source
```

Reflow the paragraph afterwards so that each line stays under 80 characters. The script creates the object with `min.cells = 3, min.features = 200`; `min.features` is implied by the stricter filter that follows, so the page names only the gene filter. The cluster sizes were checked while writing this plan: `table(pbmc3k_subset$seurat_clusters)` gives 233, 194, 194, 140, 103, 61, 58, 12 and 5 for clusters 0 to 8.

- [ ] Step 3: Regenerate and run the examples

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'devtools::document()'
git status --short
Rscript -e 'devtools::load_all(); devtools::run_examples(start = "pbmc3k_subset", document = FALSE)' 2>&1 | tail -20
```

Expected: `document()` changes the six Rd files named under Files and nothing else. `run_examples()` runs the topics from `pbmc3k_subset` on, in alphabetical order, which covers the six changed pages; none errors or warns. It needs the `load_all()` before it (stage 3 found this).

- [ ] Step 4: Look at the plots

Save the five plots of the new data to the session scratchpad and view each PNG with the Read tool:

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
OUT=$(mktemp -d)
Rscript -e "
devtools::load_all(quiet = TRUE)
set.seed(1)
X <- rbind(matrix(rnorm(60, 0, 0.6), ncol = 2), matrix(rnorm(60, 2, 0.6), ncol = 2))
rownames(X) <- sprintf('cell%02d', seq_len(nrow(X)))
grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
res <- estimator_results(fit)
id <- res\$config_id[res\$n_clusters == get_k(fit)]
print(summary(sample_scores(fit, id, 'gini')))
plots <- list(consensus = plot_consensus_matrix(fit), box = plot_cluster_boxplot(fit), violin = plot_cluster_violin(fit),
              scatter = plot_cluster_scatter(fit), diagnostic = plot_diagnostic_scatter(fit))
for (n in names(plots)) ggplot2::ggsave(file.path('$OUT', paste0(n, '.png')), plots[[n]], width = 6, height = 4, dpi = 80)
"
ls "$OUT"
```

Expected: the Gini summary has a minimum near 0.36 and a mean near 0.98 (Decision 11). In the images, the box and violin plots have visible spread below 1 in at least one cluster, the scatter plots show some larger, more opaque points between the two blobs, and the consensus heatmap has off-diagonal values that are not all dark. Describe what each image shows in the report, then delete the directory. If a plot is still flat, report it with the summary; do not change the data again without saying why.

- [ ] Step 5: Run the suite and commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'devtools::test()' 2>&1 | tail -3
cd "$REPO"
git branch --show-current
git add carve-r/R/plots.R carve-r/R/data.R carve-r/man/plot_consensus_matrix.Rd carve-r/man/plot_cluster_boxplot.Rd \
  carve-r/man/plot_cluster_violin.Rd carve-r/man/plot_cluster_scatter.Rd carve-r/man/plot_diagnostic_scatter.Rd \
  carve-r/man/pbmc3k_subset.Rd
git commit -m "$(cat <<'MSG'
docs(carve-r): draw the per-sample examples from overlapping blobs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

Expected: `FAIL 0 | WARN 0 | SKIP 0`. This task has no test of its own; the examples run inside `R CMD check` (Task 13), and Step 4 is the check that they show what they are meant to.

---

### Task 4: Version floors, the CI pandoc step and the export check

Files:
- Modify: `carve-r/DESCRIPTION` (Imports and Suggests floors), `carve-r/.gitignore`, `.github/workflows/r-ci.yml`
- Create: `carve-r/data-raw/check_exports_used.R`

Interfaces:
- Consumes: `carve-r/NAMESPACE`.
- Produces: `Rscript data-raw/check_exports_used.R [extra .Rmd files]`, run from `carve-r/`. It exits with status 0 and prints "Every export is used." when every `export()` in NAMESPACE appears as a symbol in an R chunk of `vignettes/*.Rmd` or `../notebooks/R_Tutorial.Rmd` (or of the extra files), and otherwise prints the missing names and exits with status 1. Tasks 5 to 11 run it to see what is still missing; Task 13 requires it to pass.

- [ ] Step 1: Write the export check

Create `carve-r/data-raw/check_exports_used.R`:

```r
# Fails when an exported function of CARVE is used in neither a vignette nor
# the tutorial. Run from code/carve-r:
#
#   Rscript data-raw/check_exports_used.R [extra .Rmd files]
#
# A function counts as used when its name is a symbol in an R chunk, whether
# it is called or passed as a value, as KMeans is in
# estimator_grid(KMeans, ...). Prose and inline code do not count.

chunk_code <- function(path) {
  lines <- readLines(path, warn = FALSE)
  starts <- grep("^\\s*```\\{r", lines)
  ends <- grep("^\\s*```\\s*$", lines)
  code <- character(0)
  for (start in starts) {
    end <- ends[ends > start][1L]
    if (!is.na(end) && end > start + 1L) {
      code <- c(code, lines[(start + 1L):(end - 1L)])
    }
  }
  code
}

chunk_symbols <- function(path) {
  code <- chunk_code(path)
  if (length(code) == 0L) {
    return(character(0))
  }
  data <- utils::getParseData(parse(text = code, keep.source = TRUE))
  unique(data$text[data$token %in% c("SYMBOL_FUNCTION_CALL", "SYMBOL")])
}

namespace <- readLines("NAMESPACE")
exports <- sub("^export\\((.*)\\)$", "\\1", grep("^export\\(", namespace, value = TRUE))

documents <- c(
  list.files("vignettes", pattern = "[.]Rmd$", full.names = TRUE),
  file.path("..", "notebooks", "R_Tutorial.Rmd"),
  commandArgs(trailingOnly = TRUE)
)
documents <- documents[file.exists(documents)]

used <- unique(unlist(lapply(documents, chunk_symbols)))
missing <- sort(setdiff(exports, used), method = "radix")
cat(sprintf("%d documents, %d exports, %d not used\n", length(documents), length(exports), length(missing)))
if (length(missing) > 0L) {
  cat("Not used in any vignette or the tutorial:\n", paste0("  ", missing, "\n"), sep = "")
  quit(status = 1L)
}
cat("Every export is used.\n")
```

- [ ] Step 2: Run it, then show that it sees a chunk and ignores prose

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript data-raw/check_exports_used.R; echo "status $?"
PROBE=$(mktemp -d)/probe.Rmd
printf '%s\n' 'Prose naming get_labels() and `r get_sweep_value(fit)` does not count.' '' '```{r}' 'fit <- carve(X, estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)))' 'get_k(fit)' '```' > "$PROBE"
Rscript data-raw/check_exports_used.R "$PROBE" | head -3; echo "status $?"
Rscript data-raw/check_exports_used.R "$PROBE" | grep -c -E "^  (carve|get_k|estimator_grid|KMeans)$"
Rscript data-raw/check_exports_used.R "$PROBE" | grep -c -E "^  (get_labels|get_sweep_value)$"
```

Expected: the first run prints `0 documents, 39 exports, 39 not used`, the list, and `status 1`. With the probe: `1 documents, 39 exports, 35 not used`; the first grep counts 0 (the four names in the chunk are no longer missing) and the second counts 2 (prose and inline code do not count). The `status` after the `head` pipe is the pipe's status; ignore it there.

- [ ] Step 3: Declare the floors

In `carve-r/DESCRIPTION`, change `igraph,` to `igraph (>= 1.3.0),` and `Rtsne,` to `Rtsne (>= 0.15),` under Imports, and `SeuratObject,` to `SeuratObject (>= 5.0.0),` under Suggests. Decision 8 gives the reasons; nothing else in DESCRIPTION changes in this task.

- [ ] Step 4: Ignore rendered vignette output and add pandoc to CI

Append to `carve-r/.gitignore`:

```
/vignettes/*.html
/vignettes/*.R
/vignettes/*_files/
```

In `.github/workflows/r-ci.yml`, after the step

```yaml
      - uses: r-lib/actions/setup-r@v2
        with:
          use-public-rspm: true
```

insert

```yaml
      - uses: r-lib/actions/setup-pandoc@v2
```

with the same indentation. Without pandoc, `R CMD build` cannot build the vignettes Tasks 5 to 8 add (Decision 1).

- [ ] Step 5: Check and commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'd <- read.dcf("DESCRIPTION"); cat(d[, "Imports"], "\n", d[, "Suggests"], "\n")' | grep -o -E "(igraph|Rtsne|SeuratObject) \(>= [0-9.]+\)"
Rscript -e 'devtools::test()' 2>&1 | tail -3
python3 -c "import yaml, sys; yaml.safe_load(open('$REPO/.github/workflows/r-ci.yml')); print('yaml ok')"
cd "$REPO"
git branch --show-current
git add carve-r/DESCRIPTION carve-r/.gitignore carve-r/data-raw/check_exports_used.R .github/workflows/r-ci.yml
git commit -m "$(cat <<'MSG'
build(carve-r): declare version floors, add pandoc to CI and an export check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

Expected: the three floors print; `FAIL 0 | WARN 0 | SKIP 0`; `yaml ok` (if the system `python3` lacks PyYAML, use `/Users/kaiwycik/GitHub/CARVE/code/.venv/bin/python`).

---

### Task 5: The introductory vignette, `CARVE.Rmd`

Files:
- Create: `carve-r/vignettes/CARVE.Rmd`
- Modify: `carve-r/DESCRIPTION` (Suggests `knitr` and `rmarkdown`; `VignetteBuilder: knitr`)

Interfaces:
- Consumes: the exported API as it stands after Task 4.
- Produces: `vignette("CARVE", package = "CARVE")`, which Tasks 6 to 9 link to by that name.

- [ ] Step 1: Declare the vignette builder

In `carve-r/DESCRIPTION`, add `knitr,` and `rmarkdown,` to Suggests in alphabetical order (between `jsonlite,` and `SeuratObject (>= 5.0.0),`), and after the Suggests field add the line

```
VignetteBuilder: knitr
```

- [ ] Step 2: Write the vignette

Load the `de-ai-writing` skill first. Create `carve-r/vignettes/CARVE.Rmd`:

````markdown
---
title: "Introduction to CARVE"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{Introduction to CARVE}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

```{r, include = FALSE}
knitr::opts_chunk$set(collapse = TRUE, comment = "#>", fig.width = 6, fig.height = 4)
```

CARVE chooses the number of clusters, and the clustering method, by
resampling. Each candidate configuration (a clustering method, its
parameters and a number of clusters) clusters many subsamples of the data
and gets two scores. Stability is the agreement between the clusterings of
two overlapping subsamples on the samples they share. Generalizability is
how well a random forest trained on one subsample's clusters predicts the
clusters of the samples held out of it. Both are adjusted Rand indices
(ARI), averaged over the resamples. A selection rule picks a configuration
from the scores, and the labels come from the consensus of that
configuration's resamples.

This vignette fits CARVE to simulated data and goes through the results.
`vignette("single-cell", package = "CARVE")` covers SingleCellExperiment
and Seurat objects, `vignette("customizing", package = "CARVE")` the
options of a run, and `vignette("python-users", package = "CARVE")` the
differences from the Python package.

## Simulated data

Four groups of 50 points in two dimensions:

```{r data, fig.width = 4.5, fig.height = 4}
library(CARVE)

set.seed(1)
centers <- rbind(c(0, 0), c(4, 0), c(2, 3.5), c(6, 3.5))
truth <- rep(1:4, each = 50)
X <- centers[truth, ] + matrix(rnorm(400, sd = 0.8), ncol = 2)
plot(X, col = truth, pch = 19, asp = 1, xlab = "", ylab = "")
```

## Fitting

```{r fit}
fit <- carve(X, n_clusters = 2:8, n_resamples = 30, random_state = 1)
fit
```

`carve()` takes a numeric matrix or data frame with one row per sample.
`n_clusters` lists the numbers of clusters to compare. By default CARVE
compares three methods, k-means, Ward's agglomerative clustering and
spectral clustering, so this run has
`r nrow(estimator_results(fit))` configurations. Each configuration is
clustered on `n_resamples` resamples. The default is 100; fewer resamples
give noisier scores. With `random_state` set, resample `b` draws its
subsamples with the seeds `random_state + b` and
`random_state + b + n_resamples`, so a run gives the same results on any
number of cores.

## The results table

```{r results}
results <- estimator_results(fit)
head(results[, c("config_id", "method_label", "n_clusters", "ari_stability",
                 "ari_stability_se", "ari_generalizability")])
```

Each row is one configuration, identified by its `config_id`.
`ari_stability` and `ari_generalizability` are means over the resamples,
with their standard errors in the `_se` columns and their 5th and 95th
percentiles in the `_lower` and `_upper` columns. `ari_average` is the
mean of the two. The table also has three scores computed from each
configuration's consensus matrix (`consensus_pac_stability`,
`consensus_gini_stability` and `consensus_ce_stability`) and the
classifier's accuracy on the held-out samples
(`accuracy_generalizability`).

## Selecting a configuration

```{r select}
get_k(fit)
get_k(fit, measure = "generalizability", rule = "max")
get_k(fit, measure = "gini", rule = "max")
```

`measure` names the score: `"stability"` (the default),
`"generalizability"`, `"average"`, or one of the consensus and accuracy
scores, `"pac"`, `"gini"`, `"ce"` and `"accuracy"`. `rule` says how to
pick a configuration:

- `"max"` takes the one with the highest mean score.
- `"1se"`, the default, takes the largest number of clusters among the
  configurations whose score is within one standard error of the highest.
  When several numbers of clusters score about the same, it prefers the
  finer clustering.
- `"quantile"` does the same with another band: the 5th to 95th
  percentiles of the best configuration's resample scores.

The consensus and accuracy scores have no standard errors or percentiles.
With them, `"1se"` and `"quantile"` fall back to `"max"` with a warning, so
pass `rule = "max"`.

`get_sweep_value()` returns the selected value of the swept parameter,
here the number of clusters again. `get_estimator()` returns the selected
method as a function of the data and a seed, with its parameters set:

```{r estimator}
get_sweep_value(fit)
estimator <- get_estimator(fit)
attr(estimator, "estimator")
attr(estimator, "params")
table(estimator(X, random_state = 1))
```

## Two clusters

Stability tends to be high at two clusters even when the data have more
structure, because a split into two halves is easy to repeat.
`not_two = TRUE` leaves two clusters out of the selection:

```{r not-two}
get_k(fit, not_two = TRUE)
```

## Labels

```{r labels}
labels <- get_labels(fit)
table(labels, truth)
```

`get_labels()` does not fit the method again to all the data. It takes
the selected configuration's consensus matrix, whose entry for two samples
is the share of the subsamples drawing both in which they fell in the same
cluster, and cuts an average-linkage tree built from it. The labels run
from 1 to the number of clusters. `k` asks for the best configuration at
another number of clusters:

```{r labels-k}
table(get_labels(fit, k = 3), truth)
```

## Plots

Every plot is a ggplot object, so `+` changes it and `ggplot2::ggsave()`
saves it.

```{r metric, fig.width = 7}
plot_metric_over_n_clusters(fit)
plot_metric_over_n_clusters(fit, measure = "generalizability") +
  ggplot2::labs(title = "Four simulated groups")
```

Each line is one method, with standard error bars, and the dashed line
marks the selected configuration.

```{r consensus, fig.width = 5, fig.height = 4.5}
plot_consensus_matrix(fit)
```

The heatmap is the selected configuration's consensus matrix, with the
samples ordered by cluster and the clusters shown in the band above it. A
cluster that the resamples reproduce is a block of values near 1 on the
diagonal.

Four plots show per-sample scores. `source` picks the score: `"gini"`
(the default) or `"ce"` for stability, which is 1 when every subsample
drawing the sample agreed on its partners, and `"accuracy"` for
generalizability, the share of the draws holding the sample out in which
the classifier predicted its cluster.

```{r per-sample}
plot_cluster_violin(fit)
plot_cluster_boxplot(fit, source = "accuracy")
```

The two scatter plots draw the samples over the data the fit was given,
or over the first two columns of `embedding`, such as a UMAP. Samples with
low scores are drawn larger and more opaque. In the diagnostic scatter the
shape gives the cluster and the fill the score.

```{r scatter, fig.width = 6, fig.height = 4.5}
plot_cluster_scatter(fit)
plot_diagnostic_scatter(fit, source = "ce")
```

## Scores and matrices of one configuration

`config_id` looks up a configuration's per-sample scores and consensus
matrix. Here it is the selected configuration's:

```{r scores}
id <- results$config_id[results$estimator == attr(estimator, "estimator") &
                          results$n_clusters == get_k(fit)]
id
summary(sample_scores(fit, id, source = "gini"))
dim(consensus_matrix(fit, id))
```

The fit also keeps the sweep, the grids it ran and the data:

```{r accessors}
sweep_spec(fit)
estimator_param_grids(fit)[[1]]
dim(input_data(fit))
```

## Saving a fit

A fit is an ordinary R object, saved with `saveRDS()` and read back with
`readRDS()`. It holds two consensus matrices of `n` by `n` doubles for
each configuration, so the file grows with the square of the number of
samples.

```{r save}
path <- tempfile(fileext = ".rds")
saveRDS(fit, path)
identical(get_labels(readRDS(path)), labels)
```

## Session information

```{r session}
sessionInfo()
```
````

- [ ] Step 3: Install, render and time it

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
R CMD INSTALL --no-test-load --library="$RLIB" .
OUT=$(mktemp -d)
Rscript -e "t <- system.time(rmarkdown::render('vignettes/CARVE.Rmd', output_dir = '$OUT', intermediates_dir = '$OUT', quiet = TRUE)); print(t[['elapsed']])"
Rscript -e "knitr::opts_knit\$set(base.dir = '$OUT'); knitr::knit('vignettes/CARVE.Rmd', output = '$OUT/CARVE.md', quiet = TRUE)"
grep -n -E "^## (Warning|Error)|#> (Warning|Error)" "$OUT/CARVE.md" || echo "no warnings or errors"
git status --short
echo "$OUT"
```

Expected: the elapsed time is under 60 seconds (the fit took about 4 s while writing this plan); `no warnings or errors`; `git status` shows only the new Rmd and DESCRIPTION (rendering wrote nothing into `vignettes/`). Read `$OUT/CARVE.md` from top to bottom and check every sentence that describes an output against the output under it: the number of configurations, which number of clusters each `get_k()` call gives, whether `not_two = TRUE` changed anything (on this data it should not, since two clusters score low), the label table, and what the plots show. View the PNGs under `$OUT/figure/` (or the `fig.path` knitr reports) for the metric plot, the heatmap and one scatter plot. Rewrite any sentence the output contradicts, keeping the writing rules, and re-render. Then delete `$OUT`.

- [ ] Step 4: Writing check

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO/carve-r"
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|highlight|it's important|it is important|in conclusion|overall,|\*\*|__" vignettes/CARVE.Rmd || echo "no hits"
LC_ALL=C grep -n -P "[^\x00-\x7F]" vignettes/CARVE.Rmd || echo "ascii only"
```

Expected: `no hits` and `ascii only`. Then read the prose once against `~/.claude/skills/de-ai-writing/references/signs.md` for the signs a grep cannot find (rule of three used for rhythm, elegant variation, contrasts nobody raised, a closing summary).

- [ ] Step 5: Build, run the export check, commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
R CMD build . 2>&1 | tail -5
tar -tzf CARVE_2.0.0.tar.gz | grep inst/doc
rm -f CARVE_2.0.0.tar.gz
Rscript data-raw/check_exports_used.R
cd "$REPO"
git branch --show-current
git add carve-r/DESCRIPTION carve-r/vignettes/CARVE.Rmd
git commit -m "$(cat <<'MSG'
docs(carve-r): add the introductory vignette

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

Expected: the build reports `creating vignettes ... OK`; the tarball lists `CARVE/inst/doc/CARVE.Rmd`, `CARVE.R` and `CARVE.html`. The export check still fails (status 1) and lists what this vignette does not use: the estimator and transform functions, `estimator_grid`, the preprocessing functions, `run_carve`, `attach_results`, `consensus_anchors`, `plot_metric_by_pipeline` and `plot_n_clusters_over_sweep`. Put its output in the report.

---

### Task 6: The single-cell vignette, `single-cell.Rmd`

Files:
- Create: `carve-r/vignettes/single-cell.Rmd`

Interfaces:
- Consumes: `pbmc3k_subset` (a list: `pca`, 1,000 by 30 with cell barcodes as row names; `umap`, 1,000 by 2; `seurat_clusters`, a factor), `SeuratObject::pbmc_small` (80 cells, an `Assay` with `pca` and `tsne` reductions), `run_carve()`, `attach_results()`, `carve()`, `get_labels()`, `consensus_anchors()`.
- Produces: `vignette("single-cell", package = "CARVE")`.

- [ ] Step 1: Write the vignette

Load the `de-ai-writing` skill first. Create `carve-r/vignettes/single-cell.Rmd`:

````markdown
---
title: "CARVE for single-cell data"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{CARVE for single-cell data}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

```{r, include = FALSE}
knitr::opts_chunk$set(collapse = TRUE, comment = "#>", fig.width = 6, fig.height = 4)
```

Single-cell data are usually clustered with a community detection method,
Leiden or Louvain, on a nearest-neighbor graph of the cells' principal
components. These methods take a resolution instead of a number of
clusters, and CARVE compares resolutions the way it compares numbers of
clusters: the number of clusters becomes an outcome of each resolution.
This vignette fits CARVE to a SingleCellExperiment and a Seurat object.
`vignette("CARVE", package = "CARVE")` explains the scores and the
selection rules.

The data are `pbmc3k_subset`: 1,000 cells of 10x Genomics' PBMC 3k data
set, with 30 principal components, a UMAP and the clusters of Seurat's
tutorial, all computed on the full data.

```{r sce}
library(CARVE)

cells <- rownames(pbmc3k_subset$pca)
sce <- SingleCellExperiment::SingleCellExperiment(
  reducedDims = list(PCA = pbmc3k_subset$pca, UMAP = pbmc3k_subset$umap),
  colData = S4Vectors::DataFrame(seurat_clusters = pbmc3k_subset$seurat_clusters, row.names = cells)
)
sce
```

## Comparing resolutions

```{r run}
sce <- run_carve(sce, resolution = c(0.25, 0.5, 1, 1.5), n_resamples = 20)
```

`run_carve()` reads the cells from the `"PCA"` reduced dimensions, fits
CARVE and writes the result into the object. `reduction`, `assay` and
`n_dims` choose other input; without a `"PCA"` entry it reads the
`"logcounts"` assay. With `resolution` given, CARVE compares Leiden and
Louvain clustering, each on a graph of the 15 nearest neighbors, at every
resolution. Each subsample is clustered on a graph with the neighbor count
scaled to its size, so that the graph covers the same neighborhood. This
run uses 20 resamples to keep the vignette fast; use the default, 100, for
an analysis.

The selected configuration's results go into the object:

```{r stored}
head(SummarizedExperiment::colData(sce))
names(S4Vectors::metadata(sce)$carve)
S4Vectors::metadata(sce)$carve$params$selected_resolution
table(CARVE = sce$carve, Seurat = sce$seurat_clusters)
```

The column `carve` holds the labels, as a factor. `carve_stability` and
`carve_stability_ce` hold each cell's stability in its Gini and
cross-entropy forms, and `carve_generalizability` the share of the draws
holding the cell out in which the classifier predicted its cluster.
`metadata(sce)$carve` holds the run's parameters, the results table and
the selected configuration's consensus matrix. The `key` argument changes
the name `carve` in all of them, so the results of several runs can sit in
one object.

## Plots of the object

Every plot function also takes the object and reads what `run_carve()`
stored:

```{r sce-metric, fig.width = 7}
plot_metric_over_n_clusters(sce)
plot_n_clusters_over_sweep(sce)
```

The first plot has the resolution on the x axis. The second shows how
many clusters each method found at each resolution, averaged over the
resamples, with standard error bars.

```{r sce-cells, fig.width = 6, fig.height = 4.5}
plot_cluster_scatter(sce, basis = "UMAP")
plot_cluster_violin(sce)
```

`basis` names the reduced dimensions to draw. Cells with low stability are
drawn larger and more opaque.

## Keeping the fit

`run_carve()` stores only the selected configuration, because a fit holds
the consensus matrices of all of them. To look at the others, fit with
`carve()` and keep the fit. With the same arguments and seed it gives the
same result, and `attach_results()` writes it into the object as
`run_carve()` does:

```{r fit}
fit <- carve(sce, resolution = c(0.25, 0.5, 1, 1.5), n_resamples = 20, random_state = 0)
get_sweep_value(fit)
get_k(fit)
identical(attach_results(sce, fit)$carve, sce$carve)
```

On a resolution sweep, `get_sweep_value()` returns the selected
resolution, and `get_k()` the number of clusters at it, rounded from the
mean over the resamples. `sweep_value` asks for another resolution, and
`consensus_k` cuts the selected consensus matrix into another number of
clusters:

```{r other-resolution}
table(get_labels(fit, sweep_value = 1))
table(get_labels(fit, consensus_k = 4))
sce <- attach_results(sce, fit, sweep_value = 1, key = "carve_res1")
table(sce$carve_res1)
```

## Ambiguous cells

`noise_labels = TRUE` labels -1 the cells whose stability is low compared
with the rest: those strictly below the `noise_quantile` of the selected
configuration's per-cell scores and more than 0.05 below their median.
`noise_score` picks the score.

```{r noise}
table(get_labels(fit, noise_labels = TRUE, noise_quantile = 0.05))
```

If fewer cells pass both tests than the quantile asks for, a warning gives
the counts.

## Larger data sets

A consensus matrix has one entry per pair of cells, and the fit keeps two
per configuration, `8 * n^2` bytes each. Above `anchor_threshold` cells,
5,000 by default, CARVE draws `consensus_anchors` anchor cells once and
keeps the consensus over them only. Every cell still gets its scores, and
`get_labels()` labels the cells that are not anchors with the classifier.
Setting `consensus_anchors` anchors a smaller run too, with a warning,
which is how it is shown here:

```{r anchored}
anchored <- carve(sce, resolution = c(0.5, 1), n_resamples = 20,
                  consensus_anchors = 300, random_state = 0)
length(consensus_anchors(anchored))
dim(consensus_matrix(anchored, estimator_results(anchored)$config_id[1]))
table(get_labels(anchored))
```

## Seurat objects

The same functions take a Seurat object. Its default input is the `"pca"`
reduction, then the `"data"` layer of the default assay; `assay` names a
Seurat assay, whose `"data"` layer is clustered. The labels and scores go
into the cell metadata, and the record into the `misc` slot.
`SeuratObject::pbmc_small` has 80 cells:

```{r seurat, eval = requireNamespace("SeuratObject", quietly = TRUE)}
pbmc <- SeuratObject::pbmc_small
pbmc <- run_carve(pbmc, resolution = c(0.5, 1, 2), n_resamples = 20)
head(pbmc@meta.data[, c("carve", "carve_stability", "carve_generalizability")])
names(pbmc@misc$carve)
```

```{r seurat-plot, eval = requireNamespace("SeuratObject", quietly = TRUE), fig.width = 6, fig.height = 4.5}
plot_cluster_scatter(pbmc, basis = "tsne")
```

## Session information

```{r session}
sessionInfo()
```
````

- [ ] Step 2: Install, render and time it

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
R CMD INSTALL --no-test-load --library="$RLIB" .
OUT=$(mktemp -d)
Rscript -e "t <- system.time(rmarkdown::render('vignettes/single-cell.Rmd', output_dir = '$OUT', intermediates_dir = '$OUT', quiet = TRUE)); print(t[['elapsed']])"
Rscript -e "knitr::opts_knit\$set(base.dir = '$OUT'); knitr::knit('vignettes/single-cell.Rmd', output = '$OUT/single-cell.md', quiet = TRUE)"
grep -n -E "^## (Warning|Error)|#> (Warning|Error)" "$OUT/single-cell.md" || echo "no warnings or errors"
git status --short
echo "$OUT"
```

Expected: under 60 seconds (while writing this plan, `run_carve()` on 1,000 cells at three resolutions and 20 resamples took about 7 s, and on `pbmc_small` about 1 s; this vignette fits three times on 1,000 cells). The only warning in the Markdown is the anchored run's "consensus_anchors=300 opts this run in regardless of anchor_threshold, ...", plus possibly the noise-label count warning; anything else is a defect to report. Read the Markdown against the prose: the selected resolution, whether `identical(...)` printed `TRUE` (if `FALSE`, find out why before going on; the two calls must agree), the cross-table with Seurat's clusters, the noise table, the anchor count (300) and the 300 by 300 block, and the column names of `pbmc@meta.data`. View the scatter and violin PNGs. Rewrite any sentence the output contradicts, re-render, then delete `$OUT`.

If the time is over 60 seconds, lower `n_resamples` in the three 1,000-cell fits to 15 and say so in the report.

- [ ] Step 3: Writing check

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO/carve-r"
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|highlight|it's important|it is important|in conclusion|overall,|\*\*|__" vignettes/single-cell.Rmd || echo "no hits"
LC_ALL=C grep -n -P "[^\x00-\x7F]" vignettes/single-cell.Rmd || echo "ascii only"
```

Expected: `no hits` and `ascii only`. Then read the prose against `~/.claude/skills/de-ai-writing/references/signs.md`.

- [ ] Step 4: Run the export check and commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript data-raw/check_exports_used.R
cd "$REPO"
git branch --show-current
git add carve-r/vignettes/single-cell.Rmd
git commit -m "$(cat <<'MSG'
docs(carve-r): add the single-cell vignette

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

Expected: the check still fails; `run_carve`, `attach_results`, `consensus_anchors` and `plot_n_clusters_over_sweep` are no longer listed.

---

### Task 7: The customizing vignette, `customizing.Rmd`

Files:
- Create: `carve-r/vignettes/customizing.Rmd`

Interfaces:
- Consumes: `estimator_grid()`, `KMeans()`, `AgglomerativeClustering()`, `SpectralClustering()`, `HDBSCAN()` (through the `min_cluster_size` preset), the six transforms, `preprocessing_option()`, `preprocessing_results()`, `preprocessing_pipelines()` (a list named by pipeline label), `pipeline_from_spec(spec, random_state)` (returns a function of the data), `plot_metric_by_pipeline()`, `carve(classifier =, reference_labels =, n_jobs =, BPPARAM =, verbose =)`.
- Produces: `vignette("customizing", package = "CARVE")`.

- [ ] Step 1: Write the vignette

Load the `de-ai-writing` skill first. Create `carve-r/vignettes/customizing.Rmd`:

````markdown
---
title: "Customizing a CARVE run"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{Customizing a CARVE run}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

```{r, include = FALSE}
knitr::opts_chunk$set(collapse = TRUE, comment = "#>", fig.width = 6, fig.height = 4)
```

This vignette goes through the arguments of `carve()` that change what a
run compares and how it runs: the methods and their parameters, a
clustering function of your own, density-based clustering with noise, the
classifier, randomized preprocessing, reference labels, and parallel runs.
`vignette("CARVE", package = "CARVE")` introduces the scores and the
selection rules.

The data are four groups of 60 points in ten dimensions:

```{r data}
library(CARVE)

set.seed(2)
truth <- rep(1:4, each = 60)
centers <- matrix(rnorm(40, sd = 2), nrow = 4)
X <- centers[truth, ] + matrix(rnorm(240 * 10), ncol = 10)
```

## Methods and parameters

The clustering methods are plain R functions that take the data and their
parameters and return one label per row:

```{r kmeans}
table(KMeans(X, n_clusters = 4, random_state = 0), truth)
```

`estimator_grid()` pairs a function with the values to try for each of its
arguments, and `carve()` runs every combination:

```{r grids}
grids <- list(
  estimator_grid(KMeans, n_clusters = 2:6, n_init = c(1, 10)),
  estimator_grid(AgglomerativeClustering, n_clusters = 2:6, linkage = c("ward", "average")),
  estimator_grid(SpectralClustering, n_clusters = 2:6)
)
fit <- carve(X, estimator_param_grids = grids, n_resamples = 20, random_state = 0)
unique(estimator_results(fit)$method_label)
get_k(fit)
```

The grids set the numbers of clusters, so `n_clusters` is not needed.
Every argument with several values multiplies the configurations: this run
has `r nrow(estimator_results(fit))`. Without custom grids,
`estimator_param_grids = "light"` (the default) or `"full"` chooses a
preset; `carve()`'s help page lists what each holds.

## A clustering function of your own

Any function with the same shape works: the data matrix first, one row per
sample, then its parameters, returning one integer label per row and -1
for noise. If it has a `random_state` argument, CARVE passes it each
resample's seed. This one clusters on correlation distance:

```{r custom-estimator}
average_correlation <- function(X, n_clusters = 2L) {
  distances <- stats::as.dist(1 - stats::cor(t(X)))
  stats::cutree(stats::hclust(distances, method = "average"), k = n_clusters)
}
fit_custom <- carve(
  X, n_resamples = 20, random_state = 0,
  estimator_param_grids = list(estimator_grid(average_correlation, n_clusters = 2:6))
)
estimator_results(fit_custom)[, c("method_label", "n_clusters", "ari_stability")]
```

The method takes the name the function was passed under. An anonymous
function needs `estimator_grid(..., name = )`.

## Density-based clustering and noise

HDBSCAN takes a minimum cluster size instead of a number of clusters, and
labels the samples in sparse regions -1, for noise. It needs the dbscan
package, which CARVE suggests but does not install.

```{r hdbscan, eval = requireNamespace("dbscan", quietly = TRUE)}
fit_hdbscan <- carve(X, sweep = "min_cluster_size", sweep_values = c(5, 10, 20),
                     n_resamples = 20, random_state = 0)
estimator_results(fit_hdbscan)[, c("min_cluster_size", "ari_stability", "noise_fraction",
                                   "n_clusters_observed")]
get_sweep_value(fit_hdbscan)
```

A larger minimum size gives fewer clusters, so this sweep runs from fine
to coarse, and among sizes that score about the same, `"1se"` prefers the
smaller one. `HDBSCAN()`
can also be used in a grid, with `estimator_grid(HDBSCAN, min_cluster_size
= ...)`. `noise_policy` sets what happens to the -1 labels in each
resample. `"drop"`, the default, leaves those samples out, so the
consensus matrix, the ARIs and the classifier treat them as not drawn.
`"as_cluster"` keeps -1 as an ordinary label, and `"singleton"` puts each
noise sample in a cluster of its own. `noise_fraction` is the share of the
samples labeled -1, averaged over the resamples.

```{r hdbscan-policy, eval = requireNamespace("dbscan", quietly = TRUE)}
fit_kept <- carve(X, sweep = "min_cluster_size", sweep_values = c(5, 10, 20),
                  noise_policy = "as_cluster", n_resamples = 20, random_state = 0)
get_sweep_value(fit_kept)
```

## The classifier

Generalizability uses a random forest of `n_trees` trees by default.
`classifier` takes another one: a function of `x_train`, `y_train` and
`x_test` that returns one predicted label per row of `x_test`. If it has
`n_threads` or `random_state` arguments, CARVE fills them in. A
five-nearest-neighbor classifier, with the FNN package that CARVE already
uses:

```{r classifier}
nearest_neighbors <- function(x_train, y_train, x_test) {
  FNN::knn(x_train, x_test, y_train, k = 5)
}
fit_knn <- carve(
  X, classifier = nearest_neighbors, n_resamples = 20, random_state = 0,
  estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:6))
)
get_k(fit_knn, measure = "generalizability")
```

## Randomized preprocessing

Clustering often runs on transformed data: scaled, reduced with PCA, or
embedded with t-SNE or UMAP. With `randomize_preprocessing = TRUE`, each
resample draws a pipeline of one normalization option and one
dimensionality reduction option, fits it on its subsamples and clusters
the result, so a configuration that only works after one transform scores
lower. The classifier trains on the raw features. The transforms are
plain functions too:

```{r transforms}
dim(PCA(X, n_components = 2, random_state = 0))
round(colMeans(StandardScaler(X))[1:3], 10)
counts <- matrix(rpois(20, lambda = 5), nrow = 4)
Log1p(counts)
dim(TSNE(X, perplexity = 10, random_state = 0))
```

```{r umap, eval = requireNamespace("uwot", quietly = TRUE)}
dim(UMAP(X, n_neighbors = 10, random_state = 0))
```

`preprocessing_option()` pairs a transform with the values to draw from.
Without options, CARVE uses identity, standardization and, for data
without negative values, log1p, then identity, PCA, t-SNE and, with uwot
installed, UMAP. Here the options are smaller, to keep the run short:

```{r randomized}
fit_random <- carve(
  X, randomize_preprocessing = TRUE, n_resamples = 24, random_state = 0,
  estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:6)),
  normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
  dim_reduction_options = list(preprocessing_option(Identity), preprocessing_option(PCA, n_components = c(2, 5)))
)
head(preprocessing_results(fit_random)[, c("pipeline", "n_clusters", "n_resamples", "ari_stability")])
```

```{r pipeline-plot, fig.width = 7}
plot_metric_by_pipeline(fit_random)
```

`preprocessing_results()` splits each configuration's scores by the
pipeline its resamples used, and `plot_metric_by_pipeline()` draws one
line per pipeline for the selected method. `preprocessing_pipelines()`
returns the pipelines by label, and `pipeline_from_spec()` turns one into
a function that applies it to data, for example to embed all the samples:

```{r pipelines}
pipelines <- preprocessing_pipelines(fit_random)
names(pipelines)
embed <- pipeline_from_spec(pipelines[[1]], random_state = 0)
dim(embed(X))
```

## Reference labels

The cluster numbers `get_labels()` returns are arbitrary. `reference_labels`
renames the clusters to match another labeling with the same number of
clusters, by maximum overlap. Given to `carve()`, it applies to every
`get_labels()` call on the fit:

```{r reference}
fit_reference <- carve(
  X, reference_labels = truth, n_resamples = 20, random_state = 0,
  estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:6))
)
table(get_labels(fit_reference), truth)
```

`get_labels()` keeps nothing between calls. To keep the names of one
call's clusters in a later call, pass its labels:

```{r reference-call}
first <- get_labels(fit)
table(get_labels(fit, measure = "generalizability", reference_labels = first), first)
```

## Parallel runs and memory

`n_jobs` is a core budget. The resamples of each configuration are spread
over up to `n_jobs` worker processes, and each worker's random forest gets
the cores divided by the number of workers. `n_jobs = -1` uses every core.
The results do not depend on it:

```{r parallel}
grid <- list(estimator_grid(KMeans, n_clusters = 2:6))
fit_two <- carve(X, estimator_param_grids = grid, n_jobs = 2, n_resamples = 20, random_state = 0)
fit_one <- carve(X, estimator_param_grids = grid, n_resamples = 20, random_state = 0,
                 BPPARAM = BiocParallel::SerialParam())
all.equal(estimator_results(fit_two), estimator_results(fit_one))
```

`BPPARAM` chooses the BiocParallel backend; its worker count then replaces
the one `n_jobs` would set. `show_progress = TRUE` shows a progress bar
over the configurations.

Each worker holds its own copy of the data. The fit keeps two consensus
matrices of `n` by `n` doubles per configuration, `8 * n^2` bytes each, so
memory grows with the square of the number of samples. `verbose = 2`
prints the projected total in its header:

```{r verbose}
fit_verbose <- carve(X, estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3)),
                     n_resamples = 5, random_state = 0, verbose = 2)
```

`vignette("single-cell", package = "CARVE")` shows how anchors keep that
memory bounded on large data.

## Session information

```{r session}
sessionInfo()
```
````

- [ ] Step 2: Install, render and time it

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
R CMD INSTALL --no-test-load --library="$RLIB" .
OUT=$(mktemp -d)
Rscript -e "t <- system.time(rmarkdown::render('vignettes/customizing.Rmd', output_dir = '$OUT', intermediates_dir = '$OUT', quiet = TRUE)); print(t[['elapsed']])"
Rscript -e "knitr::opts_knit\$set(base.dir = '$OUT'); knitr::knit('vignettes/customizing.Rmd', output = '$OUT/customizing.md', quiet = TRUE)"
grep -n -E "^## (Warning|Error)|#> (Warning|Error)" "$OUT/customizing.md" || echo "no warnings or errors"
git status --short
echo "$OUT"
```

Expected: under 60 seconds and `no warnings or errors`. The data and fits of this vignette were not run while writing this plan, so read the Markdown with care:

- The KMeans table should put each true group in one cluster. If the groups overlap so much that it does not, change the `set.seed(2)` value and say so in the report; the rest of the vignette assumes four separable groups.
- The HDBSCAN fits must not warn. If a size gives all-noise subsamples (a warning names it), drop that size from both `sweep_values` vectors and say so.
- `all.equal(...)` must print `TRUE`. If it does not, stop and report it: the spec promises results that do not depend on `n_jobs`.
- `names(pipelines)` and the `head()` of `preprocessing_results()` show labels such as `identity | PCA(n_components=2)`; check that the prose's description of the pipelines matches them.
- The `verbose = 2` output appears as messages, including the `consensus memory` line the prose names.
- `Log1p(counts)` prints a 4 by 5 matrix.

Rewrite any sentence the output contradicts, re-render, view the pipeline plot PNG, then delete `$OUT`.

- [ ] Step 3: Writing check

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO/carve-r"
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|highlight|it's important|it is important|in conclusion|overall,|\*\*|__" vignettes/customizing.Rmd || echo "no hits"
LC_ALL=C grep -n -P "[^\x00-\x7F]" vignettes/customizing.Rmd || echo "ascii only"
```

Expected: `no hits` and `ascii only`. Then read the prose against `~/.claude/skills/de-ai-writing/references/signs.md`. The opening sentence lists seven topics; it is the vignette's table of contents, not a rhetorical list, so keep it.

- [ ] Step 4: Run the export check and commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript data-raw/check_exports_used.R
cd "$REPO"
git branch --show-current
git add carve-r/vignettes/customizing.Rmd
git commit -m "$(cat <<'MSG'
docs(carve-r): add the vignette on customizing a run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

Expected: the check still fails, listing at most `HDBSCAN`, `LeidenClustering` and `LouvainClustering` (the presets use them, but no chunk names them). The tutorial (Tasks 10 and 11) names all three.

---

### Task 8: The vignette for Python users, `python-users.Rmd`

Files:
- Create: `carve-r/vignettes/python-users.Rmd`

Interfaces:
- Consumes: nothing at run time; it evaluates only `sessionInfo()`. Its content comes from the spec, the three earlier plans' Decisions, the stage reports and `data-raw/parity-report.md`.
- Produces: `vignette("python-users", package = "CARVE")`, which the LeidenClustering help page (Task 2) and the README (Task 9) link to.

- [ ] Step 1: Check the facts this vignette states

Before writing, confirm each of these against the current branch, and correct the vignette text below where one does not hold:

1. `git grep -n "random_state = 0" carve-r/R/tl.R` shows `run_carve()`'s default, and `carve()`'s `@param random_state` says "`NULL` means 0".
2. `data-raw/parity-report.md`, HDBSCAN section: the largest score difference is 0.007 (`consensus_pac_stability`).
3. `_claude_playground/r-rework-stage3-records/stage3-report.md`, "Seed study" and "Diagnosis of the Leiden gap": modularity higher in R by less than 0.002; Leiden scores at 0.75 and 1 higher in R by 0.008 to 0.029; Louvain within the seed spread.
4. The same report, "Stage 2 residuals", and `_claude_playground/r-rework-stage2-records/stage2-report.md`: randomized SpectralClustering at k = 3 and 4, consensus columns lower in R by about 0.015 to 0.033 over ten seeds; anchored SpectralClustering at k = 5, accuracy 0.759 in R against 0.781, over five seeds.
5. The stage 3 plan's Decisions 5 to 9 and 12 (the plot and record differences listed below), and stage 1's Decisions 1 and 2 (stateless `get_labels()`, labels 1 to k).

- [ ] Step 2: Write the vignette

Load the `de-ai-writing` skill first. Create `carve-r/vignettes/python-users.Rmd`:

````markdown
---
title: "CARVE for Python users"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{CARVE for Python users}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

The R package follows the Python package, `carve-validate` (imported as
`carve`), function for function: the same scores, selection rules and
defaults, and the same error and warning messages. This page maps the
Python names to the R ones and lists where the two packages differ.

## Names

A fit is an S4 object of class `CARVE`. Each method of the Python class is
a function that takes the fit first, and each fitted attribute has an
accessor.

| Python | R |
|---|---|
| `CARVE(...).fit(X)` | `carve(X, ...)` |
| arguments of `fit()`: `reference_labels`, `randomize_preprocessing`, `show_progress`, `mode`, `random_state` | arguments of `carve()` |
| `model.get_k()`, `get_sweep_value()`, `get_labels()` | `get_k(fit)`, `get_sweep_value(fit)`, `get_labels(fit)` |
| `model.get_estimator()`, an unfitted estimator | `get_estimator(fit)`, a function of the data and a seed |
| `model.estimator_results_` | `estimator_results(fit)` |
| `model.estimator_param_grids_` | `estimator_param_grids(fit)` |
| `model.preprocessing_results_`, `preprocessing_pipelines_` | `preprocessing_results(fit)`, `preprocessing_pipelines(fit)` |
| `model.sweep_` | `sweep_spec(fit)` |
| `model.consensus_matrices_`, `consensus_generalizability_matrices_` | `consensus_matrix(fit, config_id, type = "stability")` or `"generalizability"` |
| `model.stability_gini_scores_`, `stability_ce_scores_`, `generalizability_scores_` | `sample_scores(fit, config_id, source = "gini")`, `"ce"` or `"accuracy"` |
| `model.consensus_anchors_` | `consensus_anchors(fit)` |
| `model.X_` | `input_data(fit)` |
| `model.save(path)`, `CARVE.load(path)` | `saveRDS(fit, path)`, `readRDS(path)` |
| `model.plot_cluster_violin()` and the other plot methods | `plot_cluster_violin(fit)` and so on |
| `carve.tl.carve(adata)`, `carve.tl.attach_results(adata, model)` | `run_carve(object)`, `attach_results(object, fit)` |
| `carve.pl.cluster_violin(adata)` and the other `pl` functions | `plot_cluster_violin(object)` and so on |
| `key_added=` | `key =` |
| `use_rep=`, `layer=`, `n_pcs=` | `reduction =`, `assay =`, `n_dims =` |
| `(KMeans, {"n_clusters": [2, 3, 4]})` | `estimator_grid(KMeans, n_clusters = 2:4)` |
| `(PCA, {"n_components": [5, 10]})` | `preprocessing_option(PCA, n_components = c(5, 10))` |
| `carve.LeidenClustering(resolution=1).fit_predict(X)` | `LeidenClustering(X, resolution = 1)` |
| a scikit-learn classifier for `classifier=` | a function of `x_train`, `y_train` and `x_test` |

The estimators keep their Python and scikit-learn names: `KMeans()`,
`AgglomerativeClustering()`, `SpectralClustering()`,
`LeidenClustering()`, `LouvainClustering()` and `HDBSCAN()`. So do the
transforms `StandardScaler()`, `PCA()`, `TSNE()` and `UMAP()`, with
`Identity()` and `Log1p()` for the two `FunctionTransformer` options.
Each takes the data matrix first and returns labels or a transformed
matrix.

On a SingleCellExperiment, `obs` becomes `colData(object)` and the `uns`
record `metadata(object)$carve`; on a Seurat object they are the cell
metadata and `object@misc$carve`. The consensus matrix that Python keeps
in `obsp` is part of the record. Plots are saved with `ggplot2::ggsave()`
and changed with `+`, so the `ax`, `figsize`, `show`, `save` and `dpi`
arguments are gone, and `tl.carve()`'s `copy` is not needed because R
returns the changed object.

## Defaults

The defaults are Python's: `n_clusters = 2:10`, `n_resamples = 100`,
`subsample_ratio = 0.618`, `anchor_threshold = 5000`,
`estimator_param_grids = "light"`, `n_trees = 100`,
`noise_policy = "drop"`, `measure = "stability"` and `rule = "1se"`.
`run_carve()` sets `random_state = 0`, following scanpy as `tl.carve()`
does. `carve(random_state = NULL)` also runs with seed 0, as the Python
runner does, and `run_carve()` records 0 in the stored parameters where
Python records `None`.

## Behavior that differs

- Labels run from 1 to k, as `kmeans()` and `cutree()` number them, where
  Python's run from 0 to k - 1. Noise is -1 in both. `config_id`,
  `sweep_rank` and the method ids keep Python's values, starting at 0.
- `get_labels()` keeps no state. Python's `get_labels()` stores its result
  as the reference that later calls are matched to; in R, pass earlier
  labels as `reference_labels` to keep cluster names stable.
- `consensus_anchors = 1` means every sample, so the run stays exact.
  Python tells a share from a count by type and rejects the integer 1. In
  R, a number in (0, 1] is a share and a whole number above 1 a count.
- Consensus matrices are doubles, so they take twice the memory of
  Python's float32 matrices: 20 configurations at 5,000 anchors hold about
  8 GB in R and 4 GB in Python.
- Each `run_carve()` or `attach_results()` call replaces the whole record.
  Python leaves an earlier consensus matrix in `obsp` when a later call
  has `store_consensus=False`.
- For a Seurat object, `assay` names a Seurat assay, and CARVE clusters its
  `"data"` layer.
- R checks some input Python accepts: data with missing or infinite
  values, `noise_policy` at fit time, an unknown `measure` before a plot
  is drawn, and the plot arguments `inner`, `annotation_style`,
  `legend_loc` and `markers`.

## Plots

- `legend_loc` takes ggplot2 positions: `"right"` (the default), `"left"`,
  `"top"`, `"bottom"`, or two numbers inside the panel. matplotlib's
  `"best"` has no equivalent.
- `palette` and `cmap` take R palette names, such as `"Accent"`,
  `"viridis"` and `"Greens_r"`, or a vector of colors.
- Sizes and widths keep matplotlib's units, marker areas in square points
  and line widths in points, so Python's values give marks of about the
  same size.
- The annotation goes in the caption below the panel or, on the scatter
  plots with `annotation_style = "legend"`, at the top of the legend
  title. ggplot2 cannot place a box where it hides the least data, as
  matplotlib does.
- Off the `n_clusters` axis the annotation writes `k ~ 7` where Python
  writes k with an approximately-equal sign, which R's PDF device cannot
  print.
- `density_norm = "area"` gives violins of equal area, and `"count"`
  scales them by ggplot2's rule. Python draws `"area"` like `"width"`,
  and scales `"count"` widths as `0.2 + 0.6 * n / max(n)`.
- The diagnostic scatter uses ggplot2's five filled shapes, so the shapes
  repeat after five clusters, with Python's warning.
- The violins' jittered points are seeded, so a plot is the same each
  time.
- The scatter plots label their axes "Component 1" and "Component 2" by
  default, where Python's default is no label; `xlabel = NULL` drops the
  label.
- Samples whose score is `NaN` are left out of `plot_cluster_scatter()`.
  Python leaves them out too, except when all the other scores are equal.

## Where the results differ

R and numpy draw different random numbers, so the subsamples differ, and
the two packages agree statistically, not number for number. The source
repository's `carve-r/data-raw/parity_check.R` fits both packages on the
same data, and `carve-r/data-raw/parity-report.md` gives the differences
with their tolerances. Beyond the random numbers, the methods differ in
these ways.

- KMeans uses an R implementation of k-means++ seeding, which follows
  scikit-learn's algorithm with different random numbers.
- Leiden runs igraph's implementation, and Python runs leidenalg. On the
  same graphs from `pbmc3k_subset`, igraph's partitions reached a slightly
  higher modularity on average, by less than 0.002, and over five seeds
  CARVE's Leiden scores at resolutions 0.75 and 1 were 0.008 to 0.029
  higher in R. Louvain runs igraph's code in both packages, and its scores
  agreed within the seed-to-seed spread.
- HDBSCAN builds its tree with the dbscan package and selects clusters
  with a port of scikit-learn's selection. Where mutual reachability
  distances tie, dbscan can merge samples in a different order, which
  changes the tree and so the clusters. In the parity check's HDBSCAN case
  the scores agreed within 0.007. dbscan computes all pairwise distances,
  so its memory grows with the square of the number of samples.
- The default classifier is a ranger probability forest with scikit-learn's
  settings, which predicts the class with the highest mean leaf
  probability, as scikit-learn's random forest does.
- `PCA()` gives scikit-learn's scores up to the sign of each component,
  and up to the approximation error of the truncated solver. `TSNE()` runs
  Rtsne and `UMAP()` runs uwot, which implement the methods separately from
  scikit-learn and umap-learn, so the embeddings differ.

Two differences in the parity report have no explanation yet. With
randomized preprocessing, spectral clustering's consensus scores at 3 and
4 clusters are lower in R by about 0.015 to 0.033. On an anchored run,
spectral clustering's accuracy at 5 clusters is 0.759 in R and 0.781 in
Python. The first was measured over ten seeds and the second over five,
and both lie outside the seed-to-seed spread.

## Session information

```{r session}
sessionInfo()
```
````

- [ ] Step 3: Install, render and time it

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
R CMD INSTALL --no-test-load --library="$RLIB" .
OUT=$(mktemp -d)
Rscript -e "t <- system.time(rmarkdown::render('vignettes/python-users.Rmd', output_dir = '$OUT', intermediates_dir = '$OUT', quiet = TRUE)); print(t[['elapsed']])"
grep -c "<table" "$OUT/python-users.html"
git status --short
rm -rf "$OUT"
```

Expected: a few seconds; one table in the HTML (the names table); only the new Rmd in `git status`.

- [ ] Step 4: Writing check

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO/carve-r"
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|highlight|it's important|it is important|in conclusion|overall,|\*\*|__" vignettes/python-users.Rmd || echo "no hits"
LC_ALL=C grep -n -P "[^\x00-\x7F]" vignettes/python-users.Rmd || echo "ascii only"
```

Expected: `no hits` and `ascii only`. Read the prose against `~/.claude/skills/de-ai-writing/references/signs.md`. The bullet lists here are lists of separate facts, each a full sentence, which is what the catalogue allows; keep them unlabeled.

- [ ] Step 5: Commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO"
git branch --show-current
git add carve-r/vignettes/python-users.Rmd
git commit -m "$(cat <<'MSG'
docs(carve-r): add the vignette for Python users

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 9: README, NEWS and the package help page

Files:
- Modify: `carve-r/README.Rmd`, `carve-r/NEWS.md`, `carve-r/R/CARVE-package.R`
- Regenerated: `carve-r/README.md` (by `devtools::build_readme()`), `carve-r/man/CARVE-package.Rd`

Interfaces:
- Consumes: the four vignette names of Tasks 5 to 8.
- Produces: nothing later tasks call.

- [ ] Step 1: Collect the facts NEWS states

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
git show main:carve-r/NAMESPACE | grep -E "^export|^S3method"
git show main:carve-r/DESCRIPTION | sed -n '/^Imports:/,/^Suggests:/p'
sed -n '/^Imports:/,/^Suggests:/p' carve-r/DESCRIPTION
git show main:carve-r/R/api.R | grep -n -E "^\s+[a-z_]+ = function" | head -20
```

Expected, as found while writing this plan: version 1.0.0 exported the R6 class `CARVE`, `RunCARVE()`, `AddCarveLabels()`, `SpectralClusteringCARVE`, an S3 `carve()` with methods for matrix, data frame, Seurat and SingleCellExperiment, and 28 internal functions (`adjusted_rand_index` to `validation_iter`); its R6 methods were `fit`, `get_k`, `get_estimator`, `get_labels`, `save`, six `plot_*` methods and `CARVE$load()`, and its selection functions defaulted to `measure = "generalizability"`. Its Imports were clue, FNN, furrr, future, ggplot2, Matrix, patchwork, progressr, R6, ranger, RSpectra, stats and utils. If any of this differs, correct the NEWS text below.

- [ ] Step 2: Write NEWS.md

Load the `de-ai-writing` skill first. Replace `carve-r/NEWS.md` with:

```markdown
# CARVE 2.0.0

Version 2.0.0 is a rewrite. The package now follows the Python package,
carve-validate, function for function, and its interface is not
compatible with version 1.0.0.

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
The accessors of a fit, such as `estimator_results()`,
`consensus_matrix()` and `sample_scores()`, return what they computed.

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
```

Check the "new imports" sentence against the two Imports lists from Step 1 (methods and parallel are base packages and need no mention), and fix it if they disagree.

- [ ] Step 3: Write README.Rmd

Replace `carve-r/README.Rmd` with:

````markdown
---
output: github_document
---

<!-- README.md is generated from README.Rmd. Edit the .Rmd and run -->
<!-- devtools::build_readme() to regenerate. -->

```{r, include = FALSE}
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")
```

# CARVE

CARVE (cluster analysis with resampling for validation and exploration)
chooses the number of clusters, and the clustering method, by resampling.
Each candidate configuration clusters many subsamples of the data and is
scored for stability, the agreement between the clusterings of two
overlapping subsamples, and for generalizability, how well a classifier
trained on one subsample predicts the clusters of the held-out samples.
CARVE compares numbers of clusters, Leiden and Louvain resolutions, or
HDBSCAN's minimum cluster size, on a matrix or on a SingleCellExperiment
or Seurat object.

This is the R package. The Python package, carve-validate, is in the same
repository; both compute the same scores with the same defaults.

## Installation

```r
# install.packages("remotes")
remotes::install_github("DataSlingers/CARVE", subdir = "code/carve-r")
```

Add `build_vignettes = TRUE` to build the vignettes. HDBSCAN needs the
dbscan package, UMAP the uwot package and Seurat objects the SeuratObject
package; CARVE suggests them but does not install them.

## Example

```{r example}
library(CARVE)

set.seed(1)
centers <- rbind(c(0, 0), c(4, 0), c(2, 3.5), c(6, 3.5))
X <- centers[rep(1:4, each = 50), ] + matrix(rnorm(400, sd = 0.8), ncol = 2)

fit <- carve(X, n_clusters = 2:8, n_resamples = 30, random_state = 1)
get_k(fit)
table(get_labels(fit))
```

`plot_metric_over_n_clusters(fit)` draws the scores against the number of
clusters, and `estimator_results(fit)` holds them as a table.

## Documentation

- `vignette("CARVE", package = "CARVE")`: the scores, the selection rules,
  the results and the plots, on simulated data.
- `vignette("single-cell", package = "CARVE")`: SingleCellExperiment and
  Seurat objects, Leiden and Louvain resolutions, and large data sets.
- `vignette("customizing", package = "CARVE")`: methods and parameters,
  HDBSCAN, the classifier, randomized preprocessing, reference labels and
  parallel runs.
- `vignette("python-users", package = "CARVE")`: the Python names and
  their R equivalents, and where the results differ.
- `notebooks/R_Tutorial.Rmd` in the repository goes through every function
  on full-size data, and `notebooks/R_Tutorial.html` is its rendered
  output.

## Citation

If you use CARVE, please cite:

> Wycik, K. R., Tang, T. M., Zikry, T. M., & Allen, G. I. (2026). CARVE:
> Cluster Analysis with Resampling for Validation and Exploration. Zenodo.
> https://doi.org/10.5281/zenodo.20448965

```bibtex
@software{wycik2026carve,
  author    = {Wycik, Kai R. and Tang, Tiffany M. and Zikry, Tarek M. and Allen, Genevera I.},
  title     = {{CARVE}: Cluster Analysis with Resampling for Validation and Exploration},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20448965},
  url       = {https://doi.org/10.5281/zenodo.20448965}
}
```

## License

MIT. The HDBSCAN cluster selection is ported from scikit-learn and keeps
its BSD 3-Clause notice; see `inst/COPYRIGHTS`.
````

The citation and BibTeX entry are copied from the old README; only the italics around the title were removed (CLAUDE.md).

- [ ] Step 4: Point the package help page at the vignettes

In `carve-r/R/CARVE-package.R`, after the first paragraph (ending "...pick a configuration with a selection rule."), add:

```r
#'
#' Start with `vignette("CARVE", package = "CARVE")`. The other vignettes
#' cover single-cell data (`"single-cell"`), the options of a run
#' (`"customizing"`) and the differences from the Python package
#' (`"python-users"`).
```

- [ ] Step 5: Render the README and regenerate the help page

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'devtools::document()'
Rscript -e 'devtools::build_readme()'
sed -n '/## Example/,/## Documentation/p' README.md
git status --short
```

Expected: `README.md` shows the example with its output: `get_k(fit)` and the label table under `#>` lines. `man/CARVE-package.Rd` has the new paragraph. `build_readme()` installs the package into a temporary library, so the output is the current code's. Check that the README's sentence about `plot_metric_over_n_clusters()` and `estimator_results()` is true; it does not depend on the output.

- [ ] Step 6: Writing check

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO/carve-r"
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|highlight|it's important|it is important|in conclusion|overall,|\*\*|__" README.Rmd NEWS.md R/CARVE-package.R || echo "no hits"
LC_ALL=C grep -n -P "[^\x00-\x7F]" README.Rmd README.md NEWS.md || echo "ascii only"
```

Expected: `no hits` and `ascii only`. Read README.Rmd and NEWS.md against the catalogue. The "New features" list in NEWS is a list of separate facts; keep it plain, with no labels before colons.

- [ ] Step 7: Commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO"
git branch --show-current
git add carve-r/README.Rmd carve-r/README.md carve-r/NEWS.md carve-r/R/CARVE-package.R carve-r/man/CARVE-package.Rd
git commit -m "$(cat <<'MSG'
docs(carve-r): rewrite the README and NEWS for version 2.0.0

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 10: The tutorial, part 1 (simulated data)

Files:
- Create: `notebooks/R_Tutorial.Rmd` (header, setup and part 1)

Interfaces:
- Consumes: the installed package in `$RLIB`; mclust from the user library (for `adjustedRandIndex()`; it is installed, and the tutorial is outside the package, so it is not declared).
- Produces: `notebooks/R_Tutorial.Rmd` with a setup chunk that defines `n_jobs` and `tutorial_start`, which part 2 (Task 11) uses, and the objects `make_blobs()` and `fit_moons`. Task 11 inserts part 2 before the closing level-one section, "# Session information".

- [ ] Step 1: Write the header and part 1

Load the `de-ai-writing` skill first. Create `notebooks/R_Tutorial.Rmd`:

````markdown
---
title: "CARVE tutorial for R"
output:
  html_document:
    toc: true
    toc_depth: 2
---

```{r setup, include = FALSE}
knitr::opts_chunk$set(collapse = TRUE, comment = "#>", fig.width = 7, fig.height = 4.5)
tutorial_start <- Sys.time()
```

This tutorial goes through the CARVE R package, on simulated data in
part 1 and on the PBMC 3k single-cell data set in part 2. It follows the
Python notebooks `Tutorial.ipynb` and `Resolution_Tutorial.ipynb` in this
folder and fits at full size, with the default 100 resamples. The
package's vignettes explain the same steps on smaller data:
`vignette("CARVE", package = "CARVE")` for the scores and the selection
rules, `vignette("single-cell", package = "CARVE")`,
`vignette("customizing", package = "CARVE")` and
`vignette("python-users", package = "CARVE")`.

Part 2 needs the Seurat, SeuratObject, dbscan and mclust packages, and
downloads 7 MB from 10x Genomics into `data/` on its first run.

```{r packages}
library(CARVE)

n_jobs <- max(1L, parallel::detectCores() - 1L)
n_jobs
```

`n_jobs` is the number of cores each fit may use.

# Part 1: simulated data

## Quick start

Two pairs of interleaved half-moons, 1,000 points in four clusters. The
clusters are not convex, which k-means handles poorly and spectral
clustering handles well.

```{r moons, fig.width = 6, fig.height = 4}
make_moons <- function(n, noise) {
  n_outer <- n %/% 2
  n_inner <- n - n_outer
  t_outer <- seq(0, pi, length.out = n_outer)
  t_inner <- seq(0, pi, length.out = n_inner)
  X <- rbind(cbind(cos(t_outer), sin(t_outer)),
             cbind(1 - cos(t_inner), 1 - sin(t_inner) - 0.5))
  X + matrix(rnorm(2 * n, sd = noise), ncol = 2)
}

set.seed(42)
moons <- rbind(make_moons(500, noise = 0.12),
               make_moons(500, noise = 0.10) + matrix(c(2, 1.5), 500, 2, byrow = TRUE))
moons_truth <- rep(1:4, each = 250)
plot(moons, col = moons_truth, pch = 19, cex = 0.5, asp = 1, xlab = "", ylab = "")
```

```{r moons-fit}
fit_moons <- carve(moons, n_clusters = 2:8, n_jobs = n_jobs, random_state = 42)
fit_moons
get_k(fit_moons)
moons_labels <- get_labels(fit_moons)
table(moons_labels, moons_truth)
mclust::adjustedRandIndex(moons_labels, moons_truth)
```

```{r moons-plots}
plot_metric_over_n_clusters(fit_moons)
plot_cluster_scatter(fit_moons)
```

## How CARVE scores a configuration

A configuration is a clustering method, its parameters and a value of the
swept parameter, here the number of clusters. For each of `n_resamples`
resamples, CARVE draws two overlapping subsamples of `subsample_ratio` of
the samples (0.618 by default) and clusters both. Stability is the
adjusted Rand index (ARI) between the two clusterings on the samples they
share. For generalizability, a random forest trained on one subsample's
clusters predicts the clusters of the samples held out of it, and the ARI
compares the prediction with the clusters those samples were given. After
the resamples, CARVE builds each configuration's consensus matrix, the
share of the subsamples drawing two samples in which they fell in the
same cluster, and computes per-sample scores from it.

```{r results}
results <- estimator_results(fit_moons)
names(results)
results[results$n_clusters == 4,
        c("method_label", "ari_stability", "ari_generalizability",
          "consensus_pac_stability", "consensus_gini_stability", "accuracy_generalizability")]
```

`ari_stability` and `ari_generalizability` are means over the resamples,
with standard errors (`_se`) and 5th and 95th percentiles (`_lower`,
`_upper`). `consensus_pac_stability` is one minus the proportion of
ambiguous clustering, the share of consensus entries between 0.1 and 0.9,
so higher is better, as for every score in the table.
`consensus_gini_stability` and `consensus_ce_stability` average the
per-sample stability scores, and `accuracy_generalizability` the
per-sample accuracy of the classifier on held-out samples.

## Plots

The eight plots return ggplot objects. The metric plot draws one line per
method over the swept values, and the dashed line marks the selection:

```{r plot-metric}
plot_metric_over_n_clusters(fit_moons, measure = "generalizability")
```

`plot_n_clusters_over_sweep()` draws the number of clusters each method
returned, averaged over the resamples. On the `n_clusters` axis it is
usually the number asked for; part 2 uses it on a resolution sweep, where
the count is an outcome.

```{r plot-n-clusters}
plot_n_clusters_over_sweep(fit_moons)
```

The consensus heatmap orders the samples by cluster. `k` shows another
number of clusters:

```{r plot-consensus, fig.width = 5.5, fig.height = 5}
plot_consensus_matrix(fit_moons)
plot_consensus_matrix(fit_moons, k = 6)
```

The per-sample plots show the stability of each sample (`source = "gini"`
or `"ce"`) or the share of the draws holding it out in which the
classifier predicted its cluster (`source = "accuracy"`):

```{r plot-per-sample}
plot_cluster_boxplot(fit_moons)
plot_cluster_violin(fit_moons, source = "accuracy")
plot_diagnostic_scatter(fit_moons)
```

`plot_metric_by_pipeline()`, the eighth, is in the section on randomized
preprocessing below.

## Selecting a configuration and labeling the samples

```{r selecting}
get_k(fit_moons, measure = "generalizability")
get_sweep_value(fit_moons)
moons_estimator <- get_estimator(fit_moons)
attributes(moons_estimator)[c("estimator", "params")]
table(get_labels(fit_moons, k = 2), moons_truth)
```

`get_estimator()` returns the selected method as a function of the data
and a seed, and `get_labels(k = 2)` cuts the best configuration at two
clusters instead of the selected one. Labels run from 1 to the number of
clusters.

## Easy and hard data

Three well-separated groups, and five that overlap:

```{r easy-hard}
make_blobs <- function(n_per, centers, sd) {
  truth <- rep(seq_len(nrow(centers)), each = n_per)
  noise <- matrix(rnorm(length(truth) * ncol(centers), sd = sd), ncol = ncol(centers))
  list(X = centers[truth, ] + noise, truth = truth)
}

set.seed(0)
easy <- make_blobs(100, rbind(c(0, 0), c(8, 0), c(4, 7)), sd = 1)
hard <- make_blobs(100, rbind(c(0, 0), c(3, 0), c(1.5, 2.6), c(4.5, 2.6), c(3, 5.2)), sd = 1.2)
fit_easy <- carve(easy$X, n_clusters = 2:8, n_jobs = n_jobs, random_state = 0)
fit_hard <- carve(hard$X, n_clusters = 2:8, n_jobs = n_jobs, random_state = 0)
c(easy = get_k(fit_easy), hard = get_k(fit_hard))
```

```{r easy-hard-plots}
plot_metric_over_n_clusters(fit_easy) + ggplot2::labs(title = "Three separated groups")
plot_metric_over_n_clusters(fit_hard) + ggplot2::labs(title = "Five overlapping groups")
```

On the easy data the scores are high and peak at the true number of
groups. On the hard data they are lower and flatter, and stability and
generalizability can favor different numbers of clusters.

## The selection rules

The same hard fit under each measure and rule:

```{r rules}
measures <- c("stability", "generalizability", "average")
rules <- c("max", "1se", "quantile")
sapply(rules, function(rule) {
  sapply(measures, function(measure) get_k(fit_hard, measure = measure, rule = rule))
})
```

`"max"` takes the highest mean score. `"1se"` takes the largest number of
clusters whose score is within one standard error of the highest, and
`"quantile"` the largest within the 5th to 95th percentiles of the best
configuration's resample scores, so both lean toward finer clusterings
when scores are close.

## Two clusters

Four groups in a row can be split into two halves the same way in every
resample, so two clusters can score as well as four:

```{r not-two}
set.seed(1)
line <- make_blobs(75, cbind(c(0, 3, 6, 9), 0), sd = 0.7)
fit_line <- carve(line$X, n_clusters = 2:8, n_jobs = n_jobs, random_state = 1)
c(max = get_k(fit_line, rule = "max"),
  max_not_two = get_k(fit_line, rule = "max", not_two = TRUE),
  one_se = get_k(fit_line))
```

```{r not-two-plots, fig.width = 5.5, fig.height = 5}
plot_consensus_matrix(fit_line, k = 2)
plot_consensus_matrix(fit_line, rule = "max", not_two = TRUE)
```

The two-cluster consensus matrix looks clean, but it says little about
the four groups. `not_two = TRUE` leaves two clusters out of the
selection.

## Custom grids

`estimator_grid()` sets the methods and the values of their parameters:

```{r grids}
grids <- list(
  estimator_grid(KMeans, n_clusters = 2:8, n_init = 10),
  estimator_grid(AgglomerativeClustering, n_clusters = 2:8, linkage = c("ward", "single")),
  estimator_grid(SpectralClustering, n_clusters = 2:8, n_neighbors = c(5, 15))
)
fit_grid <- carve(moons, estimator_param_grids = grids, n_jobs = n_jobs, random_state = 42)
unique(estimator_results(fit_grid)$method_label)
get_k(fit_grid)
plot_metric_over_n_clusters(fit_grid)
```

## Randomized preprocessing

k-means cannot separate the moons in their own coordinates. With
`randomize_preprocessing = TRUE`, each resample draws a normalization and
a dimensionality reduction and clusters the transformed subsamples; the
classifier still trains on the raw coordinates.

```{r randomized}
fit_random <- carve(
  moons, randomize_preprocessing = TRUE, n_resamples = 60, n_jobs = n_jobs, random_state = 42,
  estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:8)),
  normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
  dim_reduction_options = list(preprocessing_option(Identity), preprocessing_option(TSNE, perplexity = c(30, 50)))
)
plot_metric_by_pipeline(fit_random)
pipeline_results <- preprocessing_results(fit_random)
pipeline_results[pipeline_results$n_clusters == 4,
                 c("pipeline", "n_resamples", "ari_stability", "ari_generalizability")]
```

`preprocessing_results()` splits each configuration's scores by pipeline.
This run uses 60 resamples, because every resample runs t-SNE on its
subsamples.

## Saving a fit

```{r save}
path <- file.path(tempdir(), "fit_moons.rds")
saveRDS(fit_moons, path)
reloaded <- readRDS(path)
identical(get_labels(reloaded), moons_labels)
round(file.size(path) / 2^20, 1)
```

The size is in MB. A fit keeps the data, so the scatter plots work on a
fit read back from disk; Python's `save()` leaves the data out unless
`include_data=True`.

## Reference labels

Cluster numbers are arbitrary. `reference_labels` renames the clusters to
match another labeling with the same number of clusters, here the truth:

```{r reference}
table(get_labels(fit_moons, reference_labels = moons_truth), moons_truth)
```

`carve(reference_labels = )` sets the labeling for every `get_labels()`
call on the fit.

## Parallel runs

A run clusters every configuration on every resample, so its cost grows
with the number of configurations times `n_resamples`. `n_jobs` spreads
each configuration's resamples over worker processes; the results do not
depend on it.

```{r timing}
blob_grid <- list(estimator_grid(KMeans, n_clusters = 2:8))
time_serial <- system.time(fit_serial <- carve(hard$X, estimator_param_grids = blob_grid, random_state = 0))
time_workers <- system.time(fit_workers <- carve(hard$X, estimator_param_grids = blob_grid, n_jobs = n_jobs, random_state = 0))
c(serial = time_serial[["elapsed"]], workers = time_workers[["elapsed"]])
all.equal(estimator_results(fit_serial), estimator_results(fit_workers))
```

Starting the workers has a cost of its own, so the gain grows with the
work in each resample.

# Session information

This rendering took
`r round(as.numeric(difftime(Sys.time(), tutorial_start, units = "mins")))`
minutes on `r n_jobs` cores.

```{r session}
sessionInfo()
```
````

- [ ] Step 2: Render part 1 and read it

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/notebooks"
OUT=$(mktemp -d)
Rscript -e "t <- system.time(rmarkdown::render('R_Tutorial.Rmd', output_dir = '$OUT', intermediates_dir = '$OUT', quiet = TRUE)); print(t[['elapsed']])"
Rscript -e "knitr::opts_knit\$set(base.dir = '$OUT'); knitr::knit('R_Tutorial.Rmd', output = '$OUT/R_Tutorial.md', quiet = TRUE)"
grep -n -E "^## (Warning|Error)|#> (Warning|Error)" "$OUT/R_Tutorial.md" || echo "no warnings or errors"
git -C "$REPO" status --short
echo "$OUT"
```

This renders part 1 twice (once to HTML, once to Markdown for reading); expect several minutes each. Nothing is written into `notebooks/`. Then read `$OUT/R_Tutorial.md` and check each sentence that describes an output:

- The quick start should select four clusters with a high ARI. If it does not, say so in the report and rewrite the sentences that assume it; do not tune the data to force it.
- "On the easy data the scores are high and peak at the true number of groups. On the hard data they are lower and flatter, and stability and generalizability can favor different numbers of clusters." Check against the two plots and the rules table.
- The not_two section should show `max = 2` and `max_not_two = 4`. If `max` is not 2, the section's first sentence is wrong for this data: try `sd = 0.5` in `make_blobs(75, ...)` once, and if that does not give 2 either, rewrite the section to say what the output shows.
- "k-means cannot separate the moons in their own coordinates": check the pipeline plot and table; an identity pipeline should score lower at four clusters than a t-SNE one. If not, rewrite.
- `all.equal(...)` prints `TRUE`. If not, stop and report it.
- "Starting the workers has a cost of its own, so the gain grows with the work in each resample." Keep it whatever the two times are; it explains them either way.
- `plot_n_clusters_over_sweep()`: check that the counts are close to the diagonal.

View the PNGs of the quick-start scatter, one consensus heatmap and the pipeline plot. Rewrite any sentence the output contradicts, keeping the writing rules, and re-render. Delete `$OUT` at the end.

- [ ] Step 3: Writing check

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO/notebooks"
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|highlight|it's important|it is important|in conclusion|overall,|\*\*|__" R_Tutorial.Rmd || echo "no hits"
LC_ALL=C grep -n -P "[^\x00-\x7F]" R_Tutorial.Rmd || echo "ascii only"
```

Expected: `no hits` and `ascii only`. Read the prose against `~/.claude/skills/de-ai-writing/references/signs.md`.

- [ ] Step 4: Commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO"
git branch --show-current
git add notebooks/R_Tutorial.Rmd
git commit -m "$(cat <<'MSG'
docs(notebooks): add part 1 of the R tutorial

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

The rendered HTML is committed in Task 11, once, from the finished file.

---

### Task 11: The tutorial, part 2 (PBMC 3k), and the rendered HTML

Files:
- Modify: `notebooks/R_Tutorial.Rmd` (insert part 2 before `# Session information`)
- Create: `notebooks/R_Tutorial.html` (rendered)

Interfaces:
- Consumes: `n_jobs` from the setup of Task 10; Seurat 5.5.0 and SeuratObject 5.4.0, dbscan 1.2.7 and mclust from the user library; the 10x Genomics archive at the URL `data-raw/pbmc3k_subset.R` uses.
- Produces: the finished tutorial and its HTML.

- [ ] Step 1: Check the cache directory is ignored

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
git check-ignore -v notebooks/data/pbmc3k_filtered_gene_bc_matrices.tar.gz
```

Expected: a line naming `.gitignore` and its `data/` pattern. If nothing prints, stop and report: the download must not be committed.

- [ ] Step 2: Write part 2

Load the `de-ai-writing` skill first. In `notebooks/R_Tutorial.Rmd`, insert the following immediately before the line `# Session information`:

````markdown
# Part 2: PBMC 3k

Single-cell data are usually clustered with Leiden or Louvain community
detection on a nearest-neighbor graph of principal components. These
methods take a resolution, and CARVE compares resolutions the way it
compares numbers of clusters; the number of clusters becomes an outcome.

## Data and the Seurat workflow

The 10x Genomics PBMC 3k data, processed with the steps of Seurat's
tutorial:

```{r pbmc-download}
url <- "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
dir.create("data", showWarnings = FALSE)
archive <- file.path("data", basename(url))
if (!file.exists(archive)) {
  utils::download.file(url, archive, mode = "wb")
}
matrix_dir <- file.path("data", "filtered_gene_bc_matrices", "hg19")
if (!dir.exists(matrix_dir)) {
  utils::untar(archive, exdir = "data")
}
```

```{r pbmc-seurat, message = FALSE, warning = FALSE}
pbmc <- SeuratObject::CreateSeuratObject(
  counts = Seurat::Read10X(matrix_dir), project = "pbmc3k", min.cells = 3, min.features = 200
)
pbmc[["percent.mt"]] <- Seurat::PercentageFeatureSet(pbmc, pattern = "^MT-")
keep <- pbmc$nFeature_RNA > 200 & pbmc$nFeature_RNA < 2500 & pbmc$percent.mt < 5
pbmc <- subset(pbmc, cells = colnames(pbmc)[keep])
pbmc <- Seurat::NormalizeData(pbmc, verbose = FALSE)
pbmc <- Seurat::FindVariableFeatures(pbmc, nfeatures = 2000, verbose = FALSE)
pbmc <- Seurat::ScaleData(pbmc, verbose = FALSE)
pbmc <- Seurat::RunPCA(pbmc, npcs = 30, verbose = FALSE)
pbmc <- Seurat::FindNeighbors(pbmc, dims = 1:10, verbose = FALSE)
pbmc <- Seurat::FindClusters(pbmc, resolution = 0.5, verbose = FALSE)
pbmc <- Seurat::RunUMAP(pbmc, dims = 1:10, verbose = FALSE)
pbmc
```

The chunk hides Seurat's own messages and warnings, such as its notice
about the UMAP method; CARVE's output is shown everywhere else.

## Comparing resolutions

```{r pbmc-fit}
resolutions <- c(0.25, 0.5, 0.75, 1, 1.5)
fit_pbmc <- carve(pbmc, resolution = resolutions, n_dims = 10, n_jobs = n_jobs, random_state = 0)
fit_pbmc
```

`carve()` reads the first 10 principal components of the `"pca"`
reduction, the ones Seurat's graph used. With `resolution` given, it
compares Leiden and Louvain at each resolution, each on a graph of the 15
nearest neighbors scaled to the subsample's size.

```{r pbmc-sweep}
sweep_spec(fit_pbmc)
get_sweep_value(fit_pbmc)
get_k(fit_pbmc)
```

`sweep_spec()` describes the sweep: the parameter, its values, and that
larger values give more clusters. `get_sweep_value()` returns the selected
resolution, and `get_k()` the number of clusters at it, rounded from the
mean over the resamples.

```{r pbmc-curves}
plot_metric_over_n_clusters(fit_pbmc)
plot_n_clusters_over_sweep(fit_pbmc)
```

The metric plot has the resolution on its x axis, one line per method.
The second plot shows how many clusters each method found at each
resolution.

```{r pbmc-consensus, fig.width = 5.5, fig.height = 5}
plot_consensus_matrix(fit_pbmc)
plot_consensus_matrix(fit_pbmc, sweep_value = 1.5)
```

`sweep_value` pins a resolution in every plot and in `get_labels()`. On
the UMAP, at the selected resolution and at the finest one:

```{r pbmc-scatter}
umap <- SeuratObject::Embeddings(pbmc, reduction = "umap")
plot_cluster_scatter(fit_pbmc, embedding = umap)
plot_cluster_scatter(fit_pbmc, embedding = umap, sweep_value = 1.5)
plot_cluster_violin(fit_pbmc, sweep_value = 0.25)
plot_cluster_violin(fit_pbmc, sweep_value = 1.5)
```

The results table gains the sweep columns:

```{r pbmc-results}
results_pbmc <- estimator_results(fit_pbmc)
results_pbmc[, c("method_label", "resolution", "sweep_rank", "n_clusters_observed",
                 "ari_stability", "ari_generalizability")]
```

`sweep_rank` orders the values from coarse to fine, and the selection
rules work on it, so they behave the same on every axis.
`n_clusters_observed` is the mean number of clusters over the resamples.

## Labels and Seurat's clusters

```{r pbmc-labels}
labels_pbmc <- get_labels(fit_pbmc)
table(CARVE = labels_pbmc, Seurat = pbmc$seurat_clusters)
mclust::adjustedRandIndex(labels_pbmc, pbmc$seurat_clusters)
table(get_labels(fit_pbmc, consensus_k = 4))
```

`consensus_k` cuts the selected consensus matrix into another number of
clusters, to see how the cells group at a coarser level.

## Storing the results in the object

`attach_results()` writes the selected configuration into the Seurat
object: the labels and the per-cell scores into the cell metadata, and the
parameters, the results table and the consensus matrix into
`pbmc@misc$carve`.

```{r pbmc-attach}
pbmc <- attach_results(pbmc, fit_pbmc)
head(pbmc@meta.data[, c("seurat_clusters", "carve", "carve_stability", "carve_generalizability")])
Seurat::DimPlot(pbmc, group.by = "carve")
plot_diagnostic_scatter(pbmc, basis = "umap")
```

`run_carve()` does both steps in one call. On a SingleCellExperiment made
from the same object, with the same seed, it gives the same labels:

```{r pbmc-sce}
sce <- Seurat::as.SingleCellExperiment(pbmc)
SingleCellExperiment::reducedDimNames(sce)
sce <- run_carve(sce, resolution = resolutions, n_dims = 10, n_jobs = n_jobs)
identical(as.character(sce$carve), as.character(pbmc$carve))
plot_cluster_scatter(sce, basis = "UMAP")
```

## Ambiguous cells

```{r pbmc-noise}
noisy <- get_labels(fit_pbmc, noise_labels = TRUE, noise_quantile = 0.05)
table(noisy)
pbmc$carve_noise <- factor(noisy)
Seurat::DimPlot(pbmc, group.by = "carve_noise")
```

`noise_labels = TRUE` labels -1 the cells whose stability is in the
lowest 5 percent and more than 0.05 below the median.

## HDBSCAN

HDBSCAN takes a minimum cluster size and labels cells in sparse regions
-1. It needs the dbscan package.

```{r pbmc-hdbscan}
pcs <- SeuratObject::Embeddings(pbmc, reduction = "pca")[, 1:10]
hdbscan_grid <- list(estimator_grid(HDBSCAN, min_cluster_size = c(10, 20, 40, 80)))
fit_hdbscan <- carve(pcs, estimator_param_grids = hdbscan_grid, n_jobs = n_jobs, random_state = 0)
sweep_spec(fit_hdbscan)
estimator_results(fit_hdbscan)[, c("min_cluster_size", "n_clusters_observed", "noise_fraction", "ari_stability")]
get_sweep_value(fit_hdbscan)
```

From the grid, CARVE infers that `min_cluster_size` is swept and that
larger values give fewer clusters. `noise_fraction` is the share of cells
labeled noise, averaged over the resamples. By default, `noise_policy =
"drop"` leaves noise cells out of each resample's scores.

## Graph methods on their own, and neighborhood sizes

`LeidenClustering()` and `LouvainClustering()` are plain functions:

```{r pbmc-graph}
table(LeidenClustering(pcs, resolution = 0.5, random_state = 0))
table(LouvainClustering(pcs, resolution = 0.5, random_state = 0))
```

A grid can vary the graph as well as the resolution. CARVE infers the
resolution sweep from the grid:

```{r pbmc-neighbors}
neighbor_grid <- list(estimator_grid(LeidenClustering, resolution = resolutions, n_neighbors = c(10, 15, 30)))
fit_neighbors <- carve(pcs, estimator_param_grids = neighbor_grid, n_jobs = n_jobs, random_state = 0)
sweep_spec(fit_neighbors)
plot_metric_over_n_clusters(fit_neighbors)
```

## Randomized preprocessing

The principal components above are one choice of preprocessing. With
`randomize_preprocessing = TRUE`, CARVE draws a pipeline for each resample
and fits it on the subsamples, here on the log-normalized expression of
the 2,000 variable genes:

```{r pbmc-randomized}
genes <- SeuratObject::VariableFeatures(pbmc)
expression <- t(as.matrix(SeuratObject::LayerData(pbmc, layer = "data")[genes, ]))
dim(expression)
fit_random_pbmc <- carve(
  expression, randomize_preprocessing = TRUE, n_resamples = 60, n_jobs = n_jobs, random_state = 0,
  estimator_param_grids = list(estimator_grid(LeidenClustering, resolution = c(0.25, 0.5, 1))),
  normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
  dim_reduction_options = list(preprocessing_option(PCA, n_components = c(10, 20, 30)))
)
plot_metric_by_pipeline(fit_random_pbmc)
```

`pipeline_from_spec()` applies a pipeline to all the cells, for example
the one with the highest stability:

```{r pbmc-pipeline}
pipeline_results <- preprocessing_results(fit_random_pbmc)
best <- pipeline_results$pipeline[which.max(pipeline_results$ari_stability)]
best
embedding <- pipeline_from_spec(preprocessing_pipelines(fit_random_pbmc)[[best]], random_state = 0)(expression)
dim(embedding)
```

````

- [ ] Step 3: Render the whole tutorial

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/notebooks"
Rscript -e "t <- system.time(rmarkdown::render('R_Tutorial.Rmd', quiet = TRUE)); print(t[['elapsed']])"
ls -la R_Tutorial.html
git -C "$REPO" status --short
```

Run it in the background if the shell would time out (the whole tutorial fits the moons and PBMC data several times; expect 15 to 45 minutes). Expected: `R_Tutorial.html` in `notebooks/`, self-contained (no `R_Tutorial_files/` directory), and `git status` showing the Rmd change and the HTML only; `data/` is ignored. If the file is over 25 MB, report its size before committing.

- [ ] Step 4: Read the rendered tutorial

Knit part 2's output to Markdown for reading, or read the HTML's text:

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO/notebooks"
TEXT=$(mktemp -d)/R_Tutorial.txt
python3 -c "import html, re; t = open('R_Tutorial.html').read(); t = re.sub(r'<img[^>]*>', '[image]', t); t = re.sub(r'<[^>]+>', '', t); print(html.unescape(t))" > "$TEXT"
grep -n -E "Warning|Error" "$TEXT" || echo "no warnings or errors"
echo "$TEXT"
```

Check every sentence of part 2 that describes an output:

- the selected resolution and `get_k()`, against the inline claims and the stage 3 report (resolution 0.5 was selected on the 1,000-cell subset; the full data may differ, and the text makes no claim about which);
- `identical(...)` after `run_carve()` prints `TRUE`; if not, stop and find out why;
- the HDBSCAN table: if every size gives all noise or one cluster, or a warning appears, report it and change the sizes once (for example to 5, 10, 20, 40);
- the noise sentence (lowest 5 percent, more than 0.05 below the median) matches `?get_labels`;
- `sweep_spec(fit_hdbscan)` says larger values give fewer clusters;
- the session information section names the number of minutes and cores.

View the PNGs embedded for the resolution metric plot, one consensus heatmap and one UMAP scatter (save them from the HTML, or re-knit the chunks to Markdown in a scratch directory). Rewrite any sentence the output contradicts and render again.

- [ ] Step 5: Writing check and the export check

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export R_LIBS=$RLIB
cd "$REPO/notebooks"
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|highlight|it's important|it is important|in conclusion|overall,|\*\*|__" R_Tutorial.Rmd || echo "no hits"
LC_ALL=C grep -n -P "[^\x00-\x7F]" R_Tutorial.Rmd || echo "ascii only"
cd "$REPO/carve-r"
Rscript data-raw/check_exports_used.R; echo "status $?"
```

Expected: `no hits`, `ascii only`, and `5 documents, 39 exports, 0 not used`, `Every export is used.`, `status 0`. If an export is still missing, add a line that uses it where it fits, in the vignette or tutorial section on its topic, and render that document again.

- [ ] Step 6: Commit

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO"
git branch --show-current
git add notebooks/R_Tutorial.Rmd notebooks/R_Tutorial.html
git commit -m "$(cat <<'MSG'
docs(notebooks): add part 2 of the R tutorial and its rendered HTML

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 12: The spec as built, and the open API issues

Files:
- Modify: `docs/superpowers/specs/2026-10-06-r-package-rework-design.md`
- Create: `docs/superpowers/notes/2026-10-08-open-api-issues.md`

Interfaces:
- Consumes: the Decisions sections of the four stage plans, and this plan's.
- Produces: nothing code uses.

- [ ] Step 1: Edit the spec in place

Load the `de-ai-writing` skill first. The author asked for the spec to be edited in place so that it describes the package as built (Decision 7). Make these replacements; each "old" text is quoted from the spec at 0b8d5b4, and line breaks inside a paragraph may differ. Reflow edited paragraphs to the spec's line width of about 100 characters.

1. Status line. Old: "Status: design approved in brainstorming. Each of the four plans below is written in its own conversation." New: "Status: design approved in brainstorming on 2026-10-06 and implemented in four stages, each planned in its own conversation. Edited on 2026-10-08 to describe the package as built; the Decisions sections of the four stage plans give the reasons for each change."
2. Heading "## Current state" becomes "## The package before the rework", and its first sentence "The R package was last changed on 2026-08-21." becomes "Before the rework, the R package was last changed on 2026-08-21."
3. Module map. Replace the two rows `| plotting.R | _plotting.py and pl/_plots.py |` and `| sce.R, seurat.R | tl/_carve.py and _anndata.py |` (with their backticks) by these rows, in the table's style:

   | R file | Python source |
   |---|---|
   | `plotting.R` | `_plotting.py`: drawing from a results table or per-sample vectors |
   | `plots.R` | the plot methods of `api.CARVE` and `pl/_plots.py` |
   | `sce.R`, `seurat.R` | `_anndata.py`, one file per object class |
   | `tl.R` | `tl/_carve.py` |
   | `transforms.R` | scikit-learn's `StandardScaler`, `PCA`, `TSNE` and `FunctionTransformer` options, and umap-learn's `UMAP` |
   | `types.R` | `_types.py`, and `estimator_grid()` with the grid expansion |
   | `data.R` | the help page of `pbmc3k_subset` |

   (Only the seven rows go into the spec's table; the header lines above show the columns.)
4. Dependencies. Replace the Imports and Suggests paragraphs with: "Imports: `methods`, `stats`, `utils`, `parallel`, `Matrix`, `BiocParallel`, `SingleCellExperiment`, `SummarizedExperiment`, `S4Vectors`, `igraph (>= 1.3.0)`, `ranger`, `RSpectra`, `irlba`, `Rtsne (>= 0.15)`, `FNN`, `clue`, `ggplot2 (>= 3.5.0)`, `scales`, `withr (>= 2.4.2)`." and "Suggests: `SeuratObject (>= 5.0.0)`, `dbscan`, `uwot`, `testthat (>= 3.1.8)`, `jsonlite`, `knitr`, `rmarkdown`. The package reads Seurat objects through SeuratObject, so Seurat itself is not needed; only the tutorial runs Seurat's workflow. `reticulate` is used only by `data-raw/parity_check.R`, outside the built package, and is not declared." In the next paragraph, change "`R6`, `furrr`, `future` and `progressr` are dropped." to "`R6`, `furrr`, `future`, `progressr` and `patchwork` are dropped; the consensus heatmap is a single ggplot."
5. Querying. Add `, reference_labels = NULL` as the last argument of the `get_labels()` signature, and after the paragraph on `get_estimator` add: "`get_labels()` keeps no state. Python's `get_labels` stores its result as the reference for later calls; R takes earlier labels through `reference_labels`."
6. Estimators. In "and return integer labels with -1 for noise:", write "and return integer labels from 1 to k, with -1 for noise:". Replace the HDBSCAN bullet with: "- `HDBSCAN(X, min_cluster_size, cluster_selection_method = "eom")`: the single-linkage tree of `dbscan::hdbscan()`, condensed and selected by a port of scikit-learn's `_tree.pyx`, credited in `inst/COPYRIGHTS`."
7. Classifier. Replace "A custom classifier is a function `(x_train, y_train, x_test, n_threads)` that returns predicted labels for `x_test`; `n_threads` is passed only when the function has that argument, the same rule Python applies to `n_jobs`." with "A custom classifier is a function `(x_train, y_train, x_test)` that returns predicted labels for `x_test`. CARVE also passes `n_threads` and `random_state` when the function has those arguments, the rule Python applies to `n_jobs`."
8. Fitting. Replace "The `Seurat` method takes the same three arguments" with "Seurat objects go through the `ANY` method, which takes the same three arguments".
9. Single-cell integration. Replace the code block with

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

   and after the paragraph that follows it add: "`attach_results()` records `assay`, `reduction` and `n_dims`, so that the scatter plots can rebuild the data. Each call replaces the whole record."
10. Plots. Replace "The methods for `SingleCellExperiment` and `Seurat` correspond to the `carve.pl` functions" with "The `SingleCellExperiment` methods, and the `ANY` methods, which handle Seurat objects, correspond to the `carve.pl` functions", and "Every plot returns a ggplot or patchwork object." with "Every plot returns a ggplot object."
11. Bundled data. Replace the sentence with: "`pbmc3k_subset`: a list with the 30 principal components, a two-dimensional UMAP and the Seurat cluster labels of 1,000 PBMC 3k cells, about 150 KB. `data-raw/pbmc3k_subset.R` builds it and records the download URL."
12. Known differences. Replace the Leiden bullet with: "- Leiden runs in igraph's C implementation, Python uses `leidenalg`. Both optimize the same objective, but their partitions differ, and so do CARVE's scores: on `pbmc3k_subset`, R's Leiden scores at resolutions 0.75 and 1 are 0.008 to 0.029 higher (stage 3 report)." Replace the HDBSCAN bullet with: "- HDBSCAN: `dbscan::hdbscan` builds the single-linkage tree, and R condenses it and selects clusters with a port of scikit-learn's code, so `eom` and `leaf` both follow scikit-learn. Labels can still differ where mutual reachability distances tie, because dbscan merges tied samples in a different order." Add a last bullet: "- `vignette("python-users")` lists every difference a user can see, the plot differences among them."
13. Testing and parity, Gates. Replace "CI is unchanged, `R CMD check --no-manual` with warnings as errors (`r-ci.yml`)." with "CI runs `R CMD check --no-manual` with warnings as errors (`r-ci.yml`); stage 4 added a pandoc step so that the vignettes build."
14. Vignettes, item 2. Replace "It uses `pbmc3k_subset`. Seurat chunks run only when Seurat is installed." with "It uses `pbmc3k_subset`, and `SeuratObject::pbmc_small` for Seurat objects; those chunks run only when SeuratObject is installed." In the Tutorial section, replace "A check script fails if any export is used in neither the tutorial nor a vignette." with "A check script, `data-raw/check_exports_used.R`, fails if any export is used in neither the tutorial nor a vignette."
15. Open items. Change the heading to "## Open items for the plans, and how they were settled" and replace the seven items with:
    1. HDBSCAN leaf selection (stage 2): R ports scikit-learn's condensed tree and selection onto dbscan's tree, so the full preset keeps `leaf`.
    2. Warnings as test failures (stage 1): `tests/testthat/setup.R` sets `warn = 2`, and `expect_warning()` still works under it.
    3. Seurat dispatch (stage 3): `ANY` methods that check `methods::is(x, "Seurat")`.
    4. k-means++ (stage 1): implemented and checked against scikit-learn on fixtures.
    5. Code worth keeping (stage 1): the old R code was removed at the start of stage 1, and the new ports were checked against the Python fixtures.
    6. PBMC 3k download (stage 3): the 10x Genomics URL in `data-raw/pbmc3k_subset.R`, cached in `data-raw/cache/` for the data script and `notebooks/data/` for the tutorial, both ignored by git.
    7. Estimator defaults (stages 1 and 2): read from `cluster.py` and the scikit-learn classes, and given on the help pages.

Then check:

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
grep -n -E "patchwork|CI is unchanged|Seurat chunks run only when Seurat|dbscan::hdbscan\` has no leaf|or patchwork object|\(x_train, y_train, x_test, n_threads\)" docs/superpowers/specs/2026-10-06-r-package-rework-design.md
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|highlight|it's important|it is important|in conclusion|overall,|\*\*|__" docs/superpowers/specs/2026-10-06-r-package-rework-design.md || echo "no hits"
```

Expected: the first grep prints only the Dependencies sentence that says patchwork is dropped; the second prints `no hits`. Read the edited sections once more against the catalogue.

- [ ] Step 2: Write the open-issues note

Create `docs/superpowers/notes/2026-10-08-open-api-issues.md`:

```markdown
# Open API issues in the Python and R packages

Date: 2026-10-08. These came up while porting the package to R (branch
`r-package-rework`). Both packages behave the same way today. The author
chose to keep them aligned and to fix each issue later in both.

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

Fix in both: default `mode` to the mode the fit ran with.

## Reference labels containing -1 (Python only)

Python counts -1 in `reference_labels` as a reference cluster and can
match a real cluster to it. R counts reference clusters with
`count_clusters()` and matches only onto labels 0 and up. The noise labels
spec parked the Python fix; until it lands, the two packages differ on
such input.
```

Before writing the third section, check what `attach_results()` does with a `mode = "stability"` fit and the default `mode`: if it fails, add the error message to that paragraph; if it silently stores something, say what. Use a five-resample fit on `make_blobs()` data in a scratch R session. Before writing the fourth, confirm the sentence about Python against the noise labels spec (`docs/superpowers/specs/`, the file whose name contains `noise-labels`); drop the section if that spec does not say so.

- [ ] Step 3: Writing check and commit

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|highlight|it's important|it is important|in conclusion|overall,|\*\*|__" docs/superpowers/notes/2026-10-08-open-api-issues.md || echo "no hits"
git branch --show-current
git add docs/superpowers/specs/2026-10-06-r-package-rework-design.md docs/superpowers/notes/2026-10-08-open-api-issues.md
git commit -m "$(cat <<'MSG'
docs(spec): describe the R package as built and record open API issues

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 13: The final writing pass, the package check, BiocCheck and parity

Files:
- Modify, if the writing pass finds anything: roxygen blocks and comments in `carve-r/R/*.R`, the vignettes, README.Rmd, NEWS.md, the tutorial (and its HTML, rendered again)
- Create (generated): `carve-r/data-raw/bioccheck-report.md`
- Modify (generated): `carve-r/data-raw/parity-report.md`
- Create (outside the repository): `/Users/kaiwycik/GitHub/CARVE/_claude_playground/r-rework-stage4-records/stage4-report.md`

Interfaces:
- Consumes: everything above.
- Produces: the state the branch is merged from.

- [ ] Step 1: The writing pass over the whole package

This is the last pass over every text a user reads, not only this stage's. Load the `de-ai-writing` skill first.

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO/carve-r"
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|highlight|it's important|it is important|in conclusion|overall,|\*\*|__" \
  R/*.R vignettes/*.Rmd README.Rmd NEWS.md ../notebooks/R_Tutorial.Rmd data-raw/*.R || echo "no hits"
Rscript -e 'for (f in c(list.files("R", full.names = TRUE), list.files("tests/testthat", pattern = "[.]R$", full.names = TRUE), list.files("vignettes", full.names = TRUE), "../notebooks/R_Tutorial.Rmd")) tools::showNonASCIIfile(f)'
grep -n -E "^#+ +[A-Z][a-z]+ [A-Z][a-z]+" vignettes/*.Rmd README.Rmd NEWS.md ../notebooks/R_Tutorial.Rmd || echo "no title case headings"
```

Expected: `no hits`, no output from `showNonASCIIfile()`, and `no title case headings` (a heading that starts with a proper noun, such as "CARVE 2.0.0" or "Part 2: PBMC 3k", is fine; list any such hit as kept). Error and warning messages copied from Python keep their wording and are not counted. Then read, against `~/.claude/skills/de-ai-writing/references/signs.md`:

- every roxygen block in `R/` (29 help pages; stages 1 to 3 each read their own, so look for what crosses pages: the same thing described in different words on two pages, a term defined one way here and another way there, a page that promises what another page contradicts);
- the four vignettes, the README, NEWS and the tutorial, top to bottom, as one set: the same scores, rules and defaults must be described the same way everywhere.

Fix each hit by rewriting the sentence, not by swapping a word. Run `devtools::document()` after roxygen edits, reinstall into `$RLIB` after any change under `R/`, and render a changed vignette again. If the tutorial changed, render it again (Task 11 Step 3) so that the HTML matches. Commit the fixes:

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO"
git branch --show-current
git status --short
# stage each changed file by path, then:
git commit -m "$(cat <<'MSG'
docs(carve-r): final writing pass over the help pages and documents

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

If nothing changed, skip the commit and say so in the report.

- [ ] Step 2: The suite, the export check and R CMD check

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'devtools::document()'
git status --short
Rscript -e 'devtools::test()' 2>&1 | tail -3
Rscript data-raw/check_exports_used.R; echo "status $?"
R CMD build .
_R_CHECK_FORCE_SUGGESTS_=false R CMD check --no-manual CARVE_2.0.0.tar.gz 2>&1 | tee "$(mktemp -d)/check.log" | tail -30
grep -A3 "re-building of vignette outputs" CARVE.Rcheck/00check.log
```

Expected: `document()` leaves the tree clean; `FAIL 0 | WARN 0 | SKIP 0`; `Every export is used.` and `status 0`; `Status: OK`. A WARNING or ERROR fails the stage, as in CI; record any NOTE. Keep the tarball for Step 3, and remove `CARVE.Rcheck` afterwards.

- [ ] Step 3: BiocCheck

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript -e 'BiocManager::install("BiocCheck", lib = Sys.getenv("R_LIBS"), update = FALSE, ask = FALSE)'
Rscript -e 'cat(format(packageVersion("BiocCheck")), BiocManager::version() |> format(), "\n")'
LOG=$(mktemp -d)/bioccheck.log
Rscript -e 'BiocCheck::BiocCheck("CARVE_2.0.0.tar.gz", `new-package` = TRUE)' > "$LOG" 2>&1
tail -60 "$LOG"
rm -rf CARVE.BiocCheck CARVE_2.0.0.tar.gz
```

Installing BiocCheck into `$RLIB` keeps the user library unchanged (Decision 14); it pulls in several Bioconductor packages. If the `new-package` argument name differs in the installed BiocCheck version, check `?BiocCheck::BiocCheck` and use its spelling. Write `carve-r/data-raw/bioccheck-report.md`:

```markdown
# BiocCheck report

Run on 2026-10-DD with BiocCheck X.Y.Z (Bioconductor X.Y) on CARVE 2.0.0, branch
r-package-rework at <short SHA>, from the built tarball, as a new package. It gates nothing; it is
here for the decision to submit to Bioconductor.

## Errors

<each ERROR line of the log, as a list item, or "None.">

## Warnings

<each WARNING line, or "None.">

## Notes

<each NOTE line, or "None.">
```

The angle-bracket lines are filled from the log, item for item, in the log's wording; the date, versions and SHA from the run. Do not fix anything BiocCheck reports in this stage; list the items in the stage report for the author.

- [ ] Step 4: The parity script

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
RLIB=$REPO/.Rlib
export R_LIBS=$RLIB
cd "$REPO/carve-r"
Rscript data-raw/parity_check.R
git diff --stat data-raw/parity-report.md
git diff data-raw/parity-report.md | head -40
```

Run it in the background if the shell would time out; stage 3's run took 990.8 s. Expected (Decision 15): the only changed line is the date in the first paragraph. Any other change is a defect: find the commit of this stage that changed a fit, with `git bisect` over this stage's commits if needed, before going on. The script uses the Python in `/Users/kaiwycik/GitHub/CARVE/code/.venv` through reticulate, as in stages 1 to 3.

- [ ] Step 5: Commit the reports

```bash
REPO=/Users/kaiwycik/GitHub/CARVE/code/.worktrees/r-package-rework
cd "$REPO"
git branch --show-current
git add carve-r/data-raw/bioccheck-report.md carve-r/data-raw/parity-report.md
git commit -m "$(cat <<'MSG'
test(carve-r): rerun the parity check and record the BiocCheck report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
git status --short
```

Expected: a clean tree afterwards (the ignored `.Rlib/`, `notebooks/data/` and rendered vignette files do not show).

- [ ] Step 6: Write the stage report

Write `/Users/kaiwycik/GitHub/CARVE/_claude_playground/r-rework-stage4-records/stage4-report.md`, outside the repository as the earlier stages' records are, and copy the SDD ledger and the final review's findings and fix report beside it at the end. It covers:

- the final test count and `SKIP 0`; the `R CMD check` status and any NOTE; the export check's line;
- the render time of each vignette and of the tutorial, and the HTML's size;
- the BiocCheck summary (counts of errors, warnings and notes, and the items the author should read first);
- the parity diff (date only, or what else changed and why);
- every plan defect found and how it was fixed, with the failing output; every mutation that did not fail as the plan predicted;
- every sentence of the documents that a rendered output contradicted, and how it was rewritten;
- the writing pass: what the grep found, what the reading found and how each was fixed, and any hit kept on purpose;
- what is left for the author, from "After stage 4" below.

---

## After stage 4

The R package is complete: it fits, queries and plots everything the Python package does, and its documentation, tutorial and spec describe it as built. What remains belongs to the author:

- The merge. `main` has moved since the branch was cut (at least six hECA commits, from 15b1b86 to a164ab5), so the merge is not a fast-forward. Use the superpowers:finishing-a-development-branch skill when the author asks for it. CI has never run on this branch; it runs on the first push or pull request, and it now needs the pandoc step of Task 4 and the Bioconductor imports, which `r-lib/actions/setup-r-dependencies` installs through pak.
- The BiocCheck items and the BiocStyle question (Decision 2), at submission time.
- The two parity residuals of stage 2 and the Leiden gap of stage 3, documented in `python-users.Rmd` and not diagnosed further.
- The three open API issues and the Python reference-label fix, in `docs/superpowers/notes/2026-10-08-open-api-issues.md`.
- The DESCRIPTION maintainer address, which the spec left as it is.
- The worktree at `code/.worktrees/r-package-rework` and the private library in it: remove them with `git worktree remove` after the merge. The two lines this plan added to `.git/info/exclude` can stay.
