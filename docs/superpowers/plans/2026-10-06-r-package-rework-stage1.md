# R Package Rework, Stage 1 (Core on the k Axis) Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: replace the R6 package in `carve-r/` with an S4 package whose engine reproduces the Python package's behavior on the k axis: resampling, consensus, stability and generalizability scoring, selection, `carve()`, the `get_*` functions and the accessors, with KMeans, AgglomerativeClustering and SpectralClustering.

Architecture: one R file per Python module, tests one to one with the R files. Deterministic pieces are checked against JSON fixtures written by a Python script from the real Python functions; stochastic pieces are checked by behavior (recovery, reproducibility, seed derivation). `carve()` returns an S4 object of class `CARVE` whose per-configuration lists are named by `config_id`, and every Python method becomes an S4 generic that takes the fit first.

Tech stack: R 4.5 (DESCRIPTION floor 4.1.0), methods (S4), BiocParallel, ranger, RSpectra, FNN, clue, Matrix, withr, testthat 3e, roxygen2 with markdown, devtools; Python 3.12 in `code/.venv` for the fixture and parity scripts.

Spec: `docs/superpowers/specs/2026-10-06-r-package-rework-design.md` (commit 0b8d5b4). This plan implements stage 1 of its four stages. Executors read both.

## Global Constraints

- Branch: `r-package-rework` in the `code/` repository. Commit after every task. Stage files by explicit path only; never `git add -A`, `git add .` or `git commit -a`.
- Run R commands from `code/carve-r/`. Run the fixture script from `code/`. Commands use the R on PATH (4.5.1 here) and the Python in `code/.venv`.
- Do not touch anything outside `carve-r/` except this plan's own commit; in particular not `src/`, `tests/`, `notebooks/` or `../overleaf/`.
- Package name `CARVE`, version `2.0.0`. Imports declared in this stage: `BiocParallel`, `clue`, `FNN`, `Matrix`, `methods`, `parallel`, `ranger`, `RSpectra`, `stats`, `utils`, `withr`. Suggests: `jsonlite`, `testthat (>= 3.0.0)`. Later stages add the rest of the spec's list; an import declared before it is used gives an `R CMD check` NOTE.
- Exports in this stage: `carve`, classes `CARVE` and `SweepSpec`, `get_k`, `get_sweep_value`, `get_estimator`, `get_labels`, `estimator_results`, `estimator_param_grids`, `sweep_spec`, `input_data`, `consensus_matrix`, `sample_scores`, `estimator_grid`, `KMeans`, `AgglomerativeClustering`, `SpectralClustering`. Nothing else is exported; tests reach internals through the package namespace that `devtools::test()` loads.
- Defaults copied from Python: `n_clusters = 2:10`, `n_resamples = 100`, `subsample_ratio = 0.618`, `n_trees = 100`, `n_jobs = 1`, `random_state = NULL` (resolved to 0), `estimator_param_grids = "light"`, `verbose = 0`, `measure = "stability"`, `rule = "1se"`.
- Seeds are derived, never shared: resample `b` (0-based) draws its first subsample with `random_state + b`, its second with `random_state + b + n_resamples`, and fits its estimators and classifier with `random_state + b`. All seeding goes through `seeded()`, which pins the RNG kind and restores the caller's RNG state.
- `config_id` is a join key: 0-based, as in Python, stored in the results table, and used as the names of every per-configuration list. Never index those lists by position.
- Labels: built-in estimators return 1-based integer labels; -1 marks noise; a label is a cluster when it is non-negative. `sweep_rank` and `config_id` keep Python's 0-based values so tables can be compared across languages. Sample indices inside R are 1-based; fixtures store Python's 0-based indices and tests add 1.
- Diagnostics use `warning(..., call. = FALSE)`; verbose output uses `message()` gated on `verbose`. No logging package, no `print()` or `cat()` outside `show` and `print` methods and the progress bar.
- Error and warning text copies Python's wording where Python has the message; the exact strings are in the tasks.
- Unexpected warnings fail the suite: `tests/testthat/setup.R` sets `options(warn = 2)`. A test that expects warnings captures all of them, with `expect_warning` (first match only, the rest still fail), `collect_warnings()` or `suppressWarnings()`.
- Assertions must be able to fail. After a task's tests pass, apply the mutations its last step names, one at a time, confirm the named test fails, and restore the code. Report the result in the task report.
- Writing: every roxygen text and code comment follows the spec's writing standard. The roxygen text in this plan is already written to it; keep it as written. If you add or change any user-facing text, load the `de-ai-writing` skill first. American spelling, plain prose, no bold or italics, sentence-case Rd titles.
- No non-ASCII characters in `R/` or `tests/` files (an `R CMD check` warning); write `±` for the plus-minus sign.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Decisions this plan makes that the spec left open

Each is reflected in the tasks. Items 1 and 2 change behavior a Python user would notice and are flagged to the author with the plan.

1. `get_labels()` keeps no state. Python's `get_labels` writes its output into `self.reference_labels` and aligns later calls to it. A fit returned by value cannot do that without a hidden mutable slot, so R's `get_labels()` takes a `reference_labels =` argument and otherwise aligns to the labels given to `carve(reference_labels = )`. The spec's fixed alignment rule applies: reference clusters are counted with `count_clusters()`, and only reference entries of 0 or more take part in the matching.
2. Built-in estimators return labels 1 to k, the R convention (`kmeans`, `cutree`), where Python returns 0 to k-1. `config_id`, `sweep_rank` and method ids keep Python's values.
3. KMeans runs its own Lloyd loop. The spec named `stats::kmeans(algorithm = "Lloyd")`, but that function stops with an error when a cluster empties and stops on a different rule. The R loop follows sklearn's `_kmeans_single_lloyd` (tolerance on the summed squared center shift, strict convergence on unchanged labels, empty clusters moved to the farthest points) and is checked against sklearn from fixed starting centers.
4. `R/types.R` mirrors `_types.py` (`resolve_mode`) and also holds `estimator_grid()` and `expand_param_grid()`, the R forms of Python's `GridSpec` and sklearn's `ParameterGrid`. Only the S4 classes live in `AllClasses.R`.
5. `withr` (for `with_seed`) and `parallel` (for `detectCores`) move to Imports; `jsonlite` joins Suggests for the fixtures. `reticulate` is used only by `data-raw/parity_check.R`, which is not part of the built package, so it is not declared.
6. Warnings raised inside a resample are collected in the worker and re-emitted by the main process once per configuration, without duplicates. Warnings from forked workers would otherwise be lost or repeated per resample; Python's default warning filter also prints each message once.
7. `carve()` in this stage exposes only the k-axis arguments. Custom grids already go through Python's sweep inference, so a custom estimator may sweep `resolution` or `min_cluster_size`; stage 2 adds `resolution`, `sweep`, `sweep_values`, `finer_is_larger`, `noise_policy`, the anchor arguments and the preprocessing arguments.
8. Small guards Python does not have: `X` with missing or infinite values is an error; an all-NaN measure is an error that names the column; `1se` with a NaN standard error (a single resample) returns the best row, where Python raises an empty-argmax error; `random_state` must leave room for the derived seeds in an R integer; `reference_labels` must have one entry per sample; an argument name no method knows is an error; `n_cores()` honors `_R_CHECK_LIMIT_CORES_`.
9. Python's message "Consensus matrix not available for mode=...{mode!r}.This run" lacks a space after the period; R adds it.

## File structure

```
carve-r/
  DESCRIPTION                      rewritten (Task 1); Collate added by roxygen (Task 6 on)
  NAMESPACE                        generated by roxygen
  R/CARVE-package.R                package doc and methods imports (Task 1)
  R/utils.R                        _utils.py: seeding, splits, counts, ARI, alignment,
                                   core budget, neighbor scaling, estimator and
                                   classifier calls, formatting helpers (Tasks 3-4)
  R/types.R                        _types.py: resolve_mode; estimator_grid,
                                   expand_param_grid (Task 5)
  R/AllClasses.R                   SweepSpec (Task 6), CARVE (Task 14)
  R/sweep.R                        _sweep.py (Task 6)
  R/selection.R                    _selection.py (Task 7)
  R/consensus.R                    _consensus.py, exact path (Task 8)
  R/accuracy.R                     _accuracy.py (Task 8)
  R/estimators.R                   KMeans, AgglomerativeClustering (Task 9),
                                   SpectralClustering (Task 10)
  R/grids.R                        _grids.py, k presets (Task 11)
  R/runner.R                       _runner.py: validation_iter (Task 12),
                                   run_validation (Task 13)
  R/output.R                       _output.py (Task 13)
  R/AllGenerics.R                  carve generic (Task 14), the rest (Task 15)
  R/carve.R                        carve() and fit_carve() (Task 14)
  R/accessors.R                    get_*, accessors, show (Task 15)
  tests/testthat.R                 (Task 1)
  tests/testthat/setup.R           warn = 2 (Task 1)
  tests/testthat/helper-fixtures.R read_fixture and NaN helpers (Task 1)
  tests/testthat/helper-data.R     make_blobs, make_moons (Task 1)
  tests/testthat/test-*.R          one per R file
  tests/testthat/fixtures/*.json   written by data-raw/make_fixtures.py (Task 2)
  data-raw/make_fixtures.py        (Task 2)
  data-raw/parity_check.R          (Task 16)
  data-raw/parity-report.md        written by the parity script (Task 16)
```

Removed in Task 1: `R/` (all 15 files), `tests/`, `man/`, `vignettes/`, `inst/`, `NAMESPACE`. `README.Rmd`, `README.md`, `NEWS.md`, `LICENSE`, `.Rbuildignore` and `.gitignore` stay until stage 4 rewrites them.

---

### Task 1: Clear the old package and set up the 2.0.0 skeleton

Files:
- Delete: `carve-r/R/`, `carve-r/tests/`, `carve-r/man/`, `carve-r/vignettes/`, `carve-r/inst/`, `carve-r/NAMESPACE`
- Rewrite: `carve-r/DESCRIPTION`
- Create: `carve-r/R/CARVE-package.R`, `carve-r/tests/testthat.R`, `carve-r/tests/testthat/setup.R`, `carve-r/tests/testthat/helper-fixtures.R`, `carve-r/tests/testthat/helper-data.R`
- Test: `carve-r/tests/testthat/test-package.R`

Interfaces:
- Consumes: nothing.
- Produces: test helpers available to every later test file: `read_fixture(name)` returns the parsed JSON (objects as named lists, arrays as vectors or matrices, `null` as `NULL` for a scalar field and `NA` inside an array); `fixture_matrix(x)` returns a double matrix from a parsed JSON matrix; `fixture_vector(x)` returns a double vector with `NULL` entries as `NA`; `fixture_number(x)` returns a double scalar, `NA` for `NULL`; `nan_to_na(x)` replaces `NaN` with `NA`; `make_blobs(n_per = 30L, centers = rbind(c(0, 0), c(6, 0), c(3, 5)), sd = 0.5, seed = 1L)` returns `list(X, y)` with `y` the 1-based blob index; `make_moons(n = 200L, noise = 0.05, seed = 1L)` returns `list(X, y)`.

- [ ] Step 1: Remove the old package content

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
git rm -r -q R tests man vignettes inst NAMESPACE
ls
```

Expected: `DESCRIPTION  LICENSE  NEWS.md  README.Rmd  README.md` remain (plus the hidden `.Rbuildignore` and `.gitignore`).

- [ ] Step 2: Write `DESCRIPTION`

```
Package: CARVE
Title: Cluster Analysis with Resampling for Validation and Exploration
Version: 2.0.0
Authors@R: c(
    person("Kai", "Wycik", email = "kai.wycik@columbia.edu", role = c("aut", "cre"))
  )
Description: Chooses the number of clusters and the clustering method by
    resampling. Every candidate configuration clusters many subsamples of
    the data and is scored for stability, the agreement between the
    clusterings of two overlapping subsamples, and for generalizability,
    how well a classifier trained on one subsample predicts the clusters
    of the held-out samples.
License: MIT + file LICENSE
Encoding: UTF-8
Depends: R (>= 4.1.0)
Imports:
    BiocParallel,
    clue,
    FNN,
    Matrix,
    methods,
    parallel,
    ranger,
    RSpectra,
    stats,
    utils,
    withr
Suggests:
    jsonlite,
    testthat (>= 3.0.0)
biocViews: Clustering, SingleCell, Software
Config/testthat/edition: 3
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.3.3
URL: https://github.com/DataSlingers/CARVE
BugReports: https://github.com/DataSlingers/CARVE/issues
```

- [ ] Step 3: Write `R/CARVE-package.R`

```r
#' CARVE: cluster analysis with resampling for validation and exploration
#'
#' CARVE chooses the number of clusters, and the clustering method, by
#' resampling. [carve()] clusters many subsamples of the data with every
#' candidate configuration and scores each one for stability and
#' generalizability. [get_k()] and [get_labels()] then pick a configuration
#' with a selection rule.
#'
#' @keywords internal
#' @importFrom methods new setClass setClassUnion setGeneric setMethod
#'   setValidity show standardGeneric validObject
"_PACKAGE"
```

- [ ] Step 4: Write the test harness

`tests/testthat.R`:

```r
library(testthat)
library(CARVE)

test_check("CARVE")
```

`tests/testthat/setup.R`:

```r
# Any warning a test does not expect fails it, the counterpart of
# filterwarnings = error in the Python suite. expect_warning() and
# expect_no_warning() still work under warn = 2.
withr::local_options(list(warn = 2), .local_envir = testthat::teardown_env())
```

`tests/testthat/helper-fixtures.R`:

```r
# Fixtures are written by data-raw/make_fixtures.py from the Python package.
# JSON has no NaN, so the fixtures store NaN as null. Inside an array null
# reads back as NA; a null scalar field reads back as NULL.
read_fixture <- function(name) {
  testthat::skip_if_not_installed("jsonlite")
  jsonlite::fromJSON(
    testthat::test_path("fixtures", paste0(name, ".json")),
    simplifyDataFrame = FALSE
  )
}

fixture_vector <- function(x) {
  if (is.list(x)) {
    x <- unlist(lapply(x, function(v) if (is.null(v)) NA else v))
  }
  as.numeric(x)
}

fixture_matrix <- function(x) {
  if (is.list(x)) {
    x <- do.call(rbind, lapply(x, fixture_vector))
  }
  storage.mode(x) <- "double"
  x
}

fixture_number <- function(x) {
  if (is.null(x)) NA_real_ else as.numeric(x)
}

nan_to_na <- function(x) {
  x[is.nan(x)] <- NA
  x
}
```

`tests/testthat/helper-data.R`:

```r
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
```

`tests/testthat/test-package.R`:

```r
test_that("the package is version 2.0.0", {
  expect_identical(as.character(utils::packageVersion("CARVE")), "2.0.0")
})

test_that("an unexpected warning fails a test", {
  expect_identical(getOption("warn"), 2)
})
```

- [ ] Step 5: Generate the namespace and run the tests

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(stop_on_failure = TRUE)'
```

Expected: `NAMESPACE` holds only the `importFrom(methods, ...)` lines; `man/CARVE-package.Rd` exists; 2 tests pass.

- [ ] Step 6: Commit

```bash
git add DESCRIPTION NAMESPACE R/CARVE-package.R man/CARVE-package.Rd tests/testthat.R tests/testthat/setup.R tests/testthat/helper-fixtures.R tests/testthat/helper-data.R tests/testthat/test-package.R
git commit -m "$(cat <<'EOF'
refactor(carve-r): clear the R6 package and set up the 2.0.0 skeleton

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

The `git rm` from Step 1 is already staged and goes into the same commit.

---

### Task 2: Python fixture generator

Files:
- Create: `carve-r/data-raw/make_fixtures.py`
- Create (generated): `carve-r/tests/testthat/fixtures/{consensus,ari,align,accuracy,selection,core_budget,summarize_ari,scale_neighbor,sweep,kmeans,agglomerative,spectral,consensus_cut}.json`

Interfaces:
- Consumes: the Python package installed in `code/.venv` (editable).
- Produces: the JSON files below. Shapes, as later tasks read them:
  - `consensus`: `n_samples`; `runs`, a list of `{indices (0-based), labels}`; `consensus` (12 by 12 matrix, null where NaN); `gini`, `ce` (vectors); `pac_005`, `pac_010`.
  - `ari`: `cases`, a list of `{a, b, ari}`.
  - `align`: `cases`, a list of `{name, reference, labels, aligned}`.
  - `accuracy`: `n_samples`; `runs`, a list of `{indices (0-based), true, predicted}`; `scores`.
  - `selection`: `table` (columns as named vectors); `expected`, a list of `{measure, rule, not_two, config_id, k, warning}` where `warning` is the first warning message or null.
  - `core_budget`: `cases`, a list of `{cores, n_jobs (null for None), n_resamples, outer, inner}`.
  - `summarize_ari`: `cases`, a list of `{x, mean, se, upper, lower}`.
  - `scale_neighbor`: `cases`, a list of `{params, n_fit, n_full, n_neighbors}`; the estimator's default `n_neighbors` is 7.
  - `sweep`: `labels`, a list of `{estimator, params, sweep_param, label}`; `grid` and `grid_order` (sklearn `ParameterGrid` output); `ranks`, a list of `{param, values, ranks}`.
  - `kmeans`: `X` (60 by 2); `cases`, a list of `{name, init, labels (0-based), centers, n_iter, inertia}`.
  - `agglomerative`: `X` (40 by 2); `labels`, named by linkage.
  - `spectral`: `X` (40 by 2); `scaled`; `knn_gamma`; and `self_tuning`, `rbf`, `rbf_gamma`, `knn`, each `{W, gamma, evals}`.
  - `consensus_cut`: `cases`, a list of `{k, consensus, labels}` from a real Python fit.

- [ ] Step 1: Write `carve-r/data-raw/make_fixtures.py`

```python
"""Write the Python reference outputs that the R tests compare against.

Run from code/ with the repository's virtual environment:

    .venv/bin/python carve-r/data-raw/make_fixtures.py

Each fixture is a JSON file in carve-r/tests/testthat/fixtures/. NaN is
written as null. Sample indices and labels are 0-based, as in Python; the
R tests add 1 to indices before using them.
"""

import json
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from carve import CARVE, SpectralClustering, _utils
from carve._accuracy import compute_generalizability_scores
from carve._consensus import (
    compute_consensus_matrix,
    compute_consensus_pac,
    stability_from_consensus,
)
from carve._grids import estimate_knn_gamma
from carve._selection import select_best_k, select_best_row_by_rule
from carve._sweep import format_method_label, resolve_sweep
from carve._utils import (
    _summarize_ari_scores,
    align_cluster_labels,
    resolve_core_budget,
    scale_neighbor_count,
)

OUT = Path(__file__).resolve().parents[1] / "tests" / "testthat" / "fixtures"


def clean(x):
    """Convert numpy values to JSON types, with NaN as None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    return x


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(clean(obj), f, indent=1, allow_nan=False)
        f.write("\n")


def consensus():
    rng = np.random.default_rng(0)
    n = 12
    runs = []
    for _ in range(6):
        # Sample 11 is never drawn, so its row of the matrix is NaN.
        idx = rng.choice(n - 1, size=8, replace=False)
        labels = rng.integers(0, 3, size=8)
        runs.append((idx, labels))
    M = compute_consensus_matrix(n, runs)
    gini, ce = stability_from_consensus(M)
    write(
        "consensus",
        {
            "n_samples": n,
            "runs": [{"indices": i, "labels": lab} for i, lab in runs],
            "consensus": M,
            "gini": gini,
            "ce": ce,
            "pac_005": compute_consensus_pac(M),
            "pac_010": compute_consensus_pac(M, tau=0.1),
        },
    )


def ari():
    rng = np.random.default_rng(1)
    cases = [
        ([0, 0, 1, 1], [0, 0, 1, 1]),
        ([0, 0, 1, 1], [1, 1, 0, 0]),
        ([0, 0, 0, 0], [0, 1, 2, 3]),
        ([0, 0, 1, 1], [0, 1, 0, 1]),
        ([0, 0, 0], [0, 0, 0]),
        ([0, 1, 2, 3], [0, 1, 2, 3]),
    ]
    for _ in range(5):
        cases.append((rng.integers(0, 4, size=40), rng.integers(0, 3, size=40)))
    a = rng.integers(0, 4, size=60)
    b = a.copy()
    flip = rng.choice(60, size=10, replace=False)
    b[flip] = rng.integers(0, 4, size=10)
    cases.append((a, b))
    write(
        "ari",
        {"cases": [{"a": a, "b": b, "ari": adjusted_rand_score(a, b)} for a, b in cases]},
    )


def align():
    # Every case has a unique best matching, so scipy and clue must agree.
    ref = np.repeat([0, 1, 2], 10)
    noisy = (ref + 2) % 3
    noisy[[0, 25]] = [1, 2]
    cases = [
        ("permuted", ref, (ref + 1) % 3),
        ("noisy", ref, noisy),
        ("more_clusters", ref, (np.repeat([0, 1, 2, 3], [10, 10, 6, 4]) + 1) % 4),
        ("fewer_clusters", ref, np.repeat([0, 1], [18, 12])),
        ("sparse_reference_values", np.repeat([3, 7, 9], 10), np.repeat([2, 0, 1], 10)),
    ]
    write(
        "align",
        {
            "cases": [
                {"name": n, "reference": r, "labels": lab, "aligned": align_cluster_labels(r, lab)}
                for n, r, lab in cases
            ]
        },
    )


def accuracy():
    rng = np.random.default_rng(2)
    n = 20
    runs = []
    for _ in range(5):
        # Sample 19 is never held out, so its score is 0.
        idx = rng.choice(n - 1, size=12, replace=False)
        true = rng.permutation(np.repeat([0, 1, 2], 4))
        pred = rng.permutation(3)[true]
        j = int(rng.integers(0, 12))
        pred[j] = (pred[j] + 1) % 3
        runs.append((idx, true, pred))
    write(
        "accuracy",
        {
            "n_samples": n,
            "runs": [{"indices": i, "true": t, "predicted": p} for i, t, p in runs],
            "scores": compute_generalizability_scores(n, runs),
        },
    )


def selection():
    rng = np.random.default_rng(3)
    rows = []
    for m, estimator in enumerate(["KMeans", "AgglomerativeClustering"]):
        for rank, k in enumerate(range(2, 7)):
            row = {
                "config_id": 5 * m + rank,
                "method_id": f"m{m}",
                "method_label": estimator,
                "estimator": estimator,
                "n_clusters": k,
                "sweep_param": "n_clusters",
                "sweep_value": k,
                "sweep_rank": rank,
                "n_clusters_observed": float(k),
                "n_clusters_observed_se": 0.0,
                "noise_fraction": 0.0,
            }
            for prefix in ("ari_stability", "ari_generalizability", "ari_average"):
                mean = rng.uniform(0.6, 0.95)
                row[prefix] = mean
                row[f"{prefix}_se"] = rng.uniform(0.005, 0.05)
                row[f"{prefix}_upper"] = mean + rng.uniform(0.01, 0.08)
                row[f"{prefix}_lower"] = mean - rng.uniform(0.01, 0.08)
            for col in (
                "consensus_pac_stability",
                "consensus_gini_stability",
                "consensus_ce_stability",
                "accuracy_generalizability",
            ):
                row[col] = rng.uniform(0.5, 1.0)
            rows.append(row)
    # Shuffled, so table order differs from config_id order.
    table = pd.DataFrame(rows).sample(frac=1, random_state=4).reset_index(drop=True)
    expected = []
    for measure in ("stability", "generalizability", "average", "pac", "gini", "ce", "accuracy"):
        for rule in ("max", "1se", "quantile"):
            for not_two in (False, True):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    row = select_best_row_by_rule(
                        table, measure=measure, rule=rule, not_two=not_two
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    k = select_best_k(table, measure=measure, rule=rule, not_two=not_two)
                expected.append(
                    {
                        "measure": measure,
                        "rule": rule,
                        "not_two": not_two,
                        "config_id": int(row["config_id"]),
                        "k": int(k),
                        "warning": str(caught[0].message) if caught else None,
                    }
                )
    write("selection", {"table": table.to_dict(orient="list"), "expected": expected})


def core_budget():
    cases = []
    for cores in (1, 4, 11):
        for n_jobs in (None, 1, 2, 4, -1, -2, 16):
            for n_resamples in (3, 100):
                with mock.patch.object(_utils, "cpu_count", return_value=cores):
                    outer, inner = resolve_core_budget(n_jobs, n_resamples=n_resamples)
                cases.append(
                    {
                        "cores": cores,
                        "n_jobs": n_jobs,
                        "n_resamples": n_resamples,
                        "outer": outer,
                        "inner": inner,
                    }
                )
    write("core_budget", {"cases": cases})


def summarize_ari():
    arrays = [
        [0.8, 0.9, 1.0, 0.7],
        [0.5],
        [np.nan, 0.6, 0.8],
        [np.nan, np.nan],
        [0.3, 0.3, 0.3],
    ]
    cases = []
    for a in arrays:
        mean, se, upper, lower = _summarize_ari_scores(a, len(a))
        cases.append({"x": a, "mean": mean, "se": se, "upper": upper, "lower": lower})
    write("summarize_ari", {"cases": cases})


class _Neighbors:
    def __init__(self, n_neighbors=7, n_clusters=2):
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters


def scale_neighbor():
    cases = []
    for params, n_fit, n_full in [
        ({}, 618, 1000),
        ({"n_neighbors": 15}, 618, 1000),
        ({"n_neighbors": 3}, 10, 1000),
        ({"n_neighbors": 15}, 1000, 1000),
        ({"n_neighbors": 9}, 500, 1000),
        ({"n_neighbors": 15}, 1, 1000),
    ]:
        scaled = scale_neighbor_count(_Neighbors, params, n_fit=n_fit, n_full=n_full)
        cases.append(
            {
                "params": params,
                "n_fit": n_fit,
                "n_full": n_full,
                "n_neighbors": scaled.get("n_neighbors"),
            }
        )
    write("scale_neighbor", {"cases": cases})


def sweep():
    labels = [
        ("KMeans", {"n_clusters": 3}, "n_clusters"),
        ("AgglomerativeClustering", {"n_clusters": 3, "linkage": "ward"}, "n_clusters"),
        ("SpectralClustering", {"n_clusters": 3, "affinity": "rbf", "gamma": 0.5}, "n_clusters"),
        ("SpectralClustering", {"n_clusters": 3, "gamma": 1.0, "n_neighbors": 7}, "n_clusters"),
        ("Custom", {"tol": 1e-5, "scale": True, "b": 1234.5, "resolution": 0.25}, "resolution"),
    ]
    grid = {"n_clusters": [2, 3], "linkage": ["ward", "average"], "affinity": ["x"]}
    specs = [
        resolve_sweep(n_clusters=np.array([5, 2, 3])),
        resolve_sweep(resolution=[0.5, 0.1, 1.0]),
        resolve_sweep(sweep="min_cluster_size", sweep_values=[10, 5, 20]),
    ]
    write(
        "sweep",
        {
            "labels": [
                {"estimator": e, "params": p, "sweep_param": s, "label": format_method_label(e, p, s)}
                for e, p, s in labels
            ],
            "grid": grid,
            "grid_order": list(ParameterGrid(grid)),
            "ranks": [{"param": s.param, "values": s.values, "ranks": s.ranks()} for s in specs],
        },
    )


def kmeans():
    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=1.5, random_state=5)
    starts = {
        "data_points": X[[0, 1, 2]],
        # No point is closer to the third center than to the other two, so
        # Lloyd's first step leaves it empty and relocates it.
        "empty_cluster": np.vstack([X[0], X[1], [100.0, 100.0]]),
    }
    cases = []
    for name, init in starts.items():
        km = KMeans(
            n_clusters=3, init=init, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd"
        ).fit(X)
        cases.append(
            {
                "name": name,
                "init": init,
                "labels": km.labels_,
                "centers": km.cluster_centers_,
                "n_iter": km.n_iter_,
                "inertia": km.inertia_,
            }
        )
    write("kmeans", {"X": X, "cases": cases})


def agglomerative():
    X, _ = make_blobs(n_samples=40, centers=4, cluster_std=2.0, random_state=6)
    labels = {
        linkage: AgglomerativeClustering(n_clusters=4, linkage=linkage).fit_predict(X)
        for linkage in ("ward", "average", "single", "complete")
    }
    write("agglomerative", {"X": X, "labels": labels})


def spectral():
    X, _ = make_blobs(n_samples=40, centers=3, cluster_std=1.0, random_state=7)
    out = {
        "X": X,
        "scaled": StandardScaler().fit_transform(X),
        "knn_gamma": estimate_knn_gamma(X),
    }
    settings = {
        "self_tuning": {"affinity": "self_tuning"},
        "rbf": {"affinity": "rbf"},
        "rbf_gamma": {"affinity": "rbf", "gamma": 0.3},
        "knn": {"affinity": "knn"},
    }
    for name, kwargs in settings.items():
        sc = SpectralClustering(n_clusters=3, random_state=0, **kwargs).fit(X)
        W = sc.affinity_.toarray() if sparse.issparse(sc.affinity_) else sc.affinity_
        out[name] = {"W": W, "gamma": sc.gamma_, "evals": sc.evals_}
    write("spectral", out)


def consensus_cut():
    X, _ = make_blobs(n_samples=45, centers=3, cluster_std=2.5, random_state=8)
    model = CARVE(
        n_clusters=np.array([2, 3, 4]),
        n_resamples=5,
        estimator_param_grids=[(KMeans, {"n_clusters": [2, 3, 4]})],
        random_state=0,
    ).fit(X)
    results = model.estimator_results_
    cases = []
    for k in (2, 3, 4):
        config_id = int(results.loc[results["n_clusters"] == k, "config_id"].iloc[0])
        cases.append(
            {
                "k": k,
                "consensus": model.consensus_matrices_[config_id],
                "labels": model.get_labels(k=k),
            }
        )
    write("consensus_cut", {"cases": cases})


if __name__ == "__main__":
    for make in (
        consensus,
        ari,
        align,
        accuracy,
        selection,
        core_budget,
        summarize_ari,
        scale_neighbor,
        sweep,
        kmeans,
        agglomerative,
        spectral,
        consensus_cut,
    ):
        make()
        print(f"wrote {make.__name__}.json")
```

- [ ] Step 2: Run it

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
.venv/bin/python carve-r/data-raw/make_fixtures.py
ls carve-r/tests/testthat/fixtures
```

Expected: 13 lines `wrote <name>.json` and 13 JSON files. A Python warning or traceback is a failure; fix the script, not the package.

- [ ] Step 3: Check that every fixture parses in R

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript -e 'for (f in list.files("tests/testthat/fixtures", full.names = TRUE)) { x <- jsonlite::fromJSON(f, simplifyDataFrame = FALSE); cat(basename(f), length(x), "\n") }'
```

Expected: 13 lines, no error. Spot-check one value: `Rscript -e 'x <- jsonlite::fromJSON("tests/testthat/fixtures/consensus.json", simplifyDataFrame = FALSE); print(dim(x$consensus)); print(x$gini)'` prints `12 12` and a vector whose last entry is `NA`.

- [ ] Step 4: Commit

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
git add carve-r/data-raw/make_fixtures.py carve-r/tests/testthat/fixtures
git commit -m "$(cat <<'EOF'
test(carve-r): write Python reference fixtures for the R port

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Utilities, part 1 (seeding, splits, counts, coercion, formatting)

Files:
- Create: `carve-r/R/utils.R`
- Test: `carve-r/tests/testthat/test-utils.R`

Interfaces:
- Consumes: the test helpers from Task 1; the `summarize_ari` fixture.
- Produces (all internal):
  - `seeded(seed, code)`: evaluates `code` under `seed` with the RNG kind pinned to Mersenne-Twister, Inversion, Rejection, and restores the caller's RNG state; `seed = NULL` evaluates `code` on the caller's stream.
  - `split_subsample_indices(n_samples, subsample_ratio, random_state)` returns `list(train, test)`: `floor(subsample_ratio * n_samples)` 1-based training indices in draw order, and the remaining indices sorted.
  - `count_clusters(labels)` returns an integer: distinct labels of 0 or more.
  - `coerce_n_clusters(value, expand_scalar = TRUE)` returns an integer vector.
  - `summarize_ari_scores(x)` returns the named double vector `c(mean, se, upper, lower)`.
  - `as_data_matrix(X, dense_warn_elements = 5e7)` returns a double matrix.
  - `collect_warnings(expr)` returns `list(value, warnings)` with the warnings muffled.
  - `check_dots(...)` errors on any argument passed through `...`.
  - Formatting: `format_repr(x)` (Python `repr` of a scalar), `python_list(x)` (`['a', 'b']`), `format_params(params)` (Python dict repr), `format_param_value(val)` (the method-label value format), `title_case(x)`.

- [ ] Step 1: Write the failing tests in `tests/testthat/test-utils.R`

```r
test_that("seeded() fixes the draw and restores the caller's RNG state", {
  set.seed(42)
  before <- .Random.seed
  a <- seeded(7L, stats::runif(3))
  expect_identical(.Random.seed, before)
  expect_identical(a, seeded(7L, stats::runif(3)))
})

test_that("seeded(NULL) draws from the caller's stream", {
  set.seed(1)
  a <- seeded(NULL, stats::runif(1))
  set.seed(1)
  expect_identical(a, stats::runif(1))
})

test_that("seeded() gives the same draw whatever the caller's RNG kind", {
  a <- seeded(3L, sample.int(100L, 5L))
  old <- RNGkind("L'Ecuyer-CMRG")
  withr::defer(RNGkind(old[1]))
  expect_identical(seeded(3L, sample.int(100L, 5L)), a)
})

test_that("split_subsample_indices draws floor(ratio * n) rows and holds out the rest", {
  s <- split_subsample_indices(100L, 0.618, random_state = 1L)
  expect_length(s$train, 61L)
  expect_length(s$test, 39L)
  expect_setequal(c(s$train, s$test), 1:100)
  expect_length(intersect(s$train, s$test), 0L)
  expect_false(is.unsorted(s$test))
})

test_that("split_subsample_indices is reproducible and depends on the seed", {
  expect_identical(split_subsample_indices(50L, 0.5, 3L), split_subsample_indices(50L, 0.5, 3L))
  expect_false(identical(
    split_subsample_indices(50L, 0.5, 3L)$train,
    split_subsample_indices(50L, 0.5, 4L)$train
  ))
})

test_that("count_clusters counts distinct non-negative labels", {
  expect_identical(count_clusters(c(1L, 1L, 2L, 3L)), 3L)
  expect_identical(count_clusters(c(-1L, 0L, 0L, 4L)), 2L)
  expect_identical(count_clusters(c(-1L, -1L)), 0L)
  expect_identical(count_clusters(NULL), 0L)
  expect_identical(count_clusters(integer()), 0L)
})

test_that("a single number K expands to 2:K", {
  expect_identical(coerce_n_clusters(5), 2:5)
  expect_identical(coerce_n_clusters(2L), 2L)
})

test_that("a vector of cluster counts is kept in its own order", {
  expect_identical(coerce_n_clusters(c(5, 3, 2)), c(5L, 3L, 2L))
  expect_identical(coerce_n_clusters(3, expand_scalar = FALSE), 3L)
})

test_that("invalid cluster counts are rejected", {
  expect_error(coerce_n_clusters(1), "n_clusters int must be >= 2.", fixed = TRUE)
  expect_error(coerce_n_clusters(c(1, 3)), "All n_clusters values must be >= 2.", fixed = TRUE)
  expect_error(coerce_n_clusters(c(2.5, 3)), "n_clusters values must be whole numbers.", fixed = TRUE)
  expect_error(coerce_n_clusters("3"), "n_clusters must be a numeric vector.", fixed = TRUE)
  expect_error(coerce_n_clusters(matrix(2:5, 2)), "n_clusters must be a numeric vector.", fixed = TRUE)
})

test_that("summarize_ari_scores matches Python", {
  for (case in read_fixture("summarize_ari")$cases) {
    s <- summarize_ari_scores(fixture_vector(case$x))
    expected <- vapply(case[c("mean", "se", "upper", "lower")], fixture_number, numeric(1))
    expect_equal(nan_to_na(unname(s)), unname(expected), tolerance = 1e-12)
  }
})

test_that("as_data_matrix returns a double matrix for each supported input", {
  m <- matrix(1:6, 3)
  expect_identical(as_data_matrix(m), matrix(as.numeric(1:6), 3))
  expect_identical(
    as_data_matrix(data.frame(a = 1:3, b = c(4, 5, 6))),
    cbind(a = c(1, 2, 3), b = c(4, 5, 6))
  )
  expect_identical(dim(as_data_matrix(c(1, 2, 3))), c(3L, 1L))
  expect_equal(
    as_data_matrix(Matrix::Matrix(m, sparse = TRUE)),
    matrix(as.numeric(1:6), 3),
    ignore_attr = "dimnames"
  )
})

test_that("as_data_matrix rejects other inputs", {
  expect_error(
    as_data_matrix(data.frame(a = 1:2, b = c("x", "y"))),
    "X has non-numeric columns: b.",
    fixed = TRUE
  )
  expect_error(as_data_matrix(list(1, 2)), "X must be a numeric matrix", fixed = TRUE)
  expect_error(as_data_matrix(c(1, NA)), "X contains missing or infinite values.", fixed = TRUE)
})

test_that("densifying a large sparse matrix warns", {
  s <- Matrix::Matrix(diag(4), sparse = TRUE)
  expect_warning(
    as_data_matrix(s, dense_warn_elements = 10),
    "Densifying a sparse matrix with 16 entries",
    fixed = TRUE
  )
})

test_that("collect_warnings returns the value and the muffled warnings", {
  out <- collect_warnings({
    warning("first")
    warning("second")
    3
  })
  expect_identical(out, list(value = 3, warnings = c("first", "second")))
})

test_that("check_dots names arguments no method knows", {
  expect_silent(check_dots())
  expect_error(check_dots(k = 3), "Unknown argument: k.", fixed = TRUE)
  expect_error(check_dots(k = 3, 4), "Unknown arguments: k, <unnamed>.", fixed = TRUE)
})

test_that("format_param_value matches the Python method labels", {
  expect_identical(format_param_value(7L), "7")
  expect_identical(format_param_value(1), "1")
  expect_identical(format_param_value(0.25), "0.25")
  expect_identical(format_param_value(1e-5), "1e-05")
  expect_identical(format_param_value(1234.5), "1.23e+03")
  expect_identical(format_param_value(TRUE), "True")
  expect_identical(format_param_value("ward"), "ward")
})

test_that("format_repr, python_list and format_params follow Python's repr", {
  expect_identical(format_repr("x"), "'x'")
  expect_identical(format_repr(3L), "3")
  expect_identical(format_repr(0.5), "0.5")
  expect_identical(format_repr(FALSE), "False")
  expect_identical(format_repr(NULL), "None")
  expect_identical(python_list(c("a", "b")), "['a', 'b']")
  expect_identical(format_params(list(resolution = 0.5, linkage = "ward")), "{'resolution': 0.5, 'linkage': 'ward'}")
  expect_identical(format_params(list()), "{}")
  expect_identical(title_case("max eps"), "Max Eps")
})
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "utils", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "seeded"` (and the same for the other helpers).

- [ ] Step 3: Write `R/utils.R`

```r
# Shared helpers. Mirrors _utils.py. A leaf: nothing here calls into the
# rest of the package.

# Evaluates code under a fixed seed and restores the caller's RNG state
# afterwards. The RNG kind is pinned, so a user's RNGkind() setting cannot
# change the draws. With seed = NULL the code draws from the caller's stream.
seeded <- function(seed, code) {
  if (is.null(seed)) {
    return(code)
  }
  withr::with_seed(
    seed,
    code,
    .rng_kind = "Mersenne-Twister",
    .rng_normal_kind = "Inversion",
    .rng_sample_kind = "Rejection"
  )
}

split_subsample_indices <- function(n_samples, subsample_ratio, random_state) {
  train_size <- floor(subsample_ratio * n_samples)
  train <- seeded(random_state, sample.int(n_samples, train_size))
  list(train = train, test = setdiff(seq_len(n_samples), train))
}

# Density-based methods mark unassigned points with a negative label; those
# are not counted as a cluster.
count_clusters <- function(labels) {
  if (is.null(labels) || length(labels) == 0L) {
    return(0L)
  }
  length(unique(labels[labels >= 0]))
}

# A single number K means 2:K, as in Python. expand_scalar = FALSE keeps a
# single value as it is, for values read from an estimator grid.
coerce_n_clusters <- function(value, expand_scalar = TRUE) {
  if (!is.numeric(value) || !is.null(dim(value))) {
    stop("n_clusters must be a numeric vector.", call. = FALSE)
  }
  if (anyNA(value) || any(value != round(value))) {
    stop("n_clusters values must be whole numbers.", call. = FALSE)
  }
  if (expand_scalar && length(value) == 1L) {
    if (value < 2) {
      stop("n_clusters int must be >= 2.", call. = FALSE)
    }
    return(seq.int(2L, as.integer(value)))
  }
  if (any(value < 2)) {
    stop("All n_clusters values must be >= 2.", call. = FALSE)
  }
  as.integer(value)
}

# Mean, standard error and the 95th and 5th percentiles of the resampled
# ARI scores, ignoring NaN. Quantile type 7 is numpy's default.
summarize_ari_scores <- function(x) {
  x <- as.numeric(x)
  ok <- x[!is.na(x)]
  if (length(ok) == 0L) {
    return(c(mean = NaN, se = NaN, upper = NaN, lower = NaN))
  }
  se <- if (length(ok) > 1L) stats::sd(ok) / sqrt(length(ok)) else NaN
  c(
    mean = mean(ok),
    se = se,
    upper = unname(stats::quantile(ok, 0.95, type = 7)),
    lower = unname(stats::quantile(ok, 0.05, type = 7))
  )
}

# Sparse input is densified: several estimators need dense input, and a
# sparse matrix would only move the failure into a worker.
as_data_matrix <- function(X, dense_warn_elements = 5e7) {
  if (is.data.frame(X)) {
    numeric_cols <- vapply(X, is.numeric, logical(1))
    if (!all(numeric_cols)) {
      stop(sprintf(
        "X has non-numeric columns: %s.",
        paste(names(X)[!numeric_cols], collapse = ", ")
      ), call. = FALSE)
    }
    X <- as.matrix(X)
  } else if (inherits(X, "Matrix")) {
    n_elements <- as.numeric(nrow(X)) * ncol(X)
    if (n_elements > dense_warn_elements) {
      warning(sprintf(
        "Densifying a sparse matrix with %s entries (~%.1f GB dense). Pass a reduced representation instead, such as principal components.",
        format(n_elements, big.mark = ",", scientific = FALSE),
        n_elements * 8 / 1e9
      ), call. = FALSE)
    }
    X <- as.matrix(X)
  } else if (is.numeric(X) && is.null(dim(X))) {
    X <- matrix(X, ncol = 1L)
  } else if (!is.matrix(X) || !is.numeric(X)) {
    stop(
      "X must be a numeric matrix, a numeric data frame, a sparse Matrix or a numeric vector.",
      call. = FALSE
    )
  }
  storage.mode(X) <- "double"
  if (any(!is.finite(X))) {
    stop("X contains missing or infinite values.", call. = FALSE)
  }
  X
}

# Runs expr, muffles its warnings and returns them with the value. The
# runner uses it inside each resample, so warnings from parallel workers
# reach the main process.
collect_warnings <- function(expr) {
  messages <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    messages <<- c(messages, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = messages)
}

check_dots <- function(...) {
  if (...length() == 0L) {
    return(invisible(NULL))
  }
  arg_names <- names(list(...))
  if (is.null(arg_names)) {
    arg_names <- rep("", ...length())
  }
  arg_names[arg_names == ""] <- "<unnamed>"
  stop(sprintf(
    "Unknown argument%s: %s.",
    if (length(arg_names) > 1L) "s" else "",
    paste(arg_names, collapse = ", ")
  ), call. = FALSE)
}

# Formatting that reproduces the Python package's labels and messages.
format_repr <- function(x) {
  if (is.null(x)) {
    return("None")
  }
  if (is.character(x) && length(x) == 1L) {
    return(paste0("'", x, "'"))
  }
  if (is.logical(x) && length(x) == 1L && !is.na(x)) {
    return(if (x) "True" else "False")
  }
  if (is.numeric(x) && length(x) == 1L) {
    return(format(x, digits = 15))
  }
  paste(deparse(x), collapse = "")
}

python_list <- function(x) {
  paste0("[", paste0("'", x, "'", collapse = ", "), "]")
}

format_params <- function(params) {
  if (length(params) == 0L) {
    return("{}")
  }
  parts <- vapply(
    names(params),
    function(name) paste0("'", name, "': ", format_repr(params[[name]])),
    character(1)
  )
  paste0("{", paste(parts, collapse = ", "), "}")
}

format_param_value <- function(val) {
  if (is.logical(val) && length(val) == 1L && !is.na(val)) {
    return(if (val) "True" else "False")
  }
  if (is.numeric(val) && length(val) == 1L && !is.na(val)) {
    if (val == round(val)) {
      return(sprintf("%.0f", val))
    }
    return(sprintf("%.3g", val))
  }
  if (is.character(val) && length(val) == 1L) {
    return(val)
  }
  paste(deparse(val), collapse = "")
}

title_case <- function(x) {
  words <- strsplit(x, " ", fixed = TRUE)[[1L]]
  paste(
    paste0(toupper(substring(words, 1L, 1L)), tolower(substring(words, 2L))),
    collapse = " "
  )
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "utils", stop_on_failure = TRUE)'`
Expected: PASS, no warnings.

- [ ] Step 5: Mutation check

Apply each, confirm the named test fails, restore:
- In `seeded()`, drop the three `.rng_*` arguments: "seeded() gives the same draw whatever the caller's RNG kind" fails.
- In `split_subsample_indices()`, use `ceiling` for `train_size`: the floor test fails.
- In `count_clusters()`, drop `[labels >= 0]`: "count_clusters counts distinct non-negative labels" fails.
- In `summarize_ari_scores()`, use `type = 6`: the fixture test fails.

- [ ] Step 6: Commit

```bash
git add R/utils.R tests/testthat/test-utils.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add seeding, subsample splits and formatting helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Utilities, part 2 (ARI, alignment, core budget, neighbor scaling, estimator and classifier calls)

Files:
- Modify: `carve-r/R/utils.R` (append)
- Test: `carve-r/tests/testthat/test-utils.R` (append)

Interfaces:
- Consumes: `seeded`, `as_data_matrix` (Task 3); fixtures `ari`, `align`, `core_budget`, `scale_neighbor`.
- Produces (all internal):
  - `adjusted_rand_index(a, b)` returns a double; exactly 1 when the two partitions agree on every pair (sklearn's special case).
  - `align_cluster_labels(reference_labels, labels, keep = NULL)` returns `labels` renamed to the reference values by maximum-overlap matching; the matching uses only positions where `keep` is `TRUE` (all by default); unmatched labels take fresh ids above the largest matched reference value, in sorted order. Integer reference gives integer output.
  - `n_cores()` returns the logical core count, at most 2 when `_R_CHECK_LIMIT_CORES_` is set to anything but `"false"`.
  - `resolve_core_budget(n_jobs, n_resamples)` returns `c(outer = <int>, inner = <int>)`.
  - `scale_neighbor_count(estimator, params, n_fit, n_full)` returns `params`, with `n_neighbors` scaled when the estimator takes it.
  - `call_estimator(estimator, X, params, random_state = NULL)` returns integer labels, one per row.
  - `default_generalizability_classifier(classifier, n_features, n_trees, random_state, n_threads)` returns `function(x_train, y_train, x_test)` that gives one integer label per row of `x_test`.

- [ ] Step 1: Append the failing tests to `tests/testthat/test-utils.R`

```r
test_that("adjusted_rand_index matches sklearn", {
  for (case in read_fixture("ari")$cases) {
    expect_equal(adjusted_rand_index(case$a, case$b), case$ari, tolerance = 1e-12)
  }
})

test_that("align_cluster_labels matches Python", {
  for (case in read_fixture("align")$cases) {
    expect_identical(
      align_cluster_labels(case$reference, case$labels),
      case$aligned,
      info = case$name
    )
  }
})

test_that("reference entries outside keep take no part in the matching", {
  reference <- c(-1L, -1L, 0L, 0L, 1L, 1L)
  labels <- c(1L, 1L, 2L, 2L, 3L, 3L)
  # Without keep, label 1 is matched onto -1. With it, label 1 has no
  # partner and takes the fresh id 2.
  expect_identical(align_cluster_labels(reference, labels)[1:2], c(-1L, -1L))
  expect_identical(
    align_cluster_labels(reference, labels, keep = reference >= 0),
    c(2L, 2L, 0L, 0L, 1L, 1L)
  )
})

test_that("resolve_core_budget matches Python", {
  for (case in read_fixture("core_budget")$cases) {
    local_mocked_bindings(n_cores = function() as.integer(case$cores))
    expect_identical(
      resolve_core_budget(case$n_jobs, case$n_resamples),
      c(outer = as.integer(case$outer), inner = as.integer(case$inner)),
      info = paste(case$cores, format_repr(case$n_jobs), case$n_resamples)
    )
  }
})

test_that("n_jobs = 0 is rejected", {
  expect_error(
    resolve_core_budget(0, 10L),
    "n_jobs == 0 has no meaning; use 1 or a negative count",
    fixed = TRUE
  )
})

test_that("n_cores honors R CMD check's core limit", {
  withr::local_envvar(c("_R_CHECK_LIMIT_CORES_" = "TRUE"))
  expect_lte(n_cores(), 2L)
})

test_that("scale_neighbor_count matches Python", {
  estimator <- function(X, n_neighbors = 7L, n_clusters = 2L) NULL
  for (case in read_fixture("scale_neighbor")$cases) {
    scaled <- scale_neighbor_count(estimator, case$params, case$n_fit, case$n_full)
    expect_equal(scaled$n_neighbors, case$n_neighbors)
  }
})

test_that("an estimator without n_neighbors passes through scaling", {
  estimator <- function(X, n_clusters = 2L) NULL
  expect_identical(
    scale_neighbor_count(estimator, list(n_clusters = 3L), 10, 100),
    list(n_clusters = 3L)
  )
})

test_that("call_estimator passes random_state only to estimators that take it", {
  with_seed_arg <- function(X, n_clusters, random_state = NULL) rep(random_state, nrow(X))
  without <- function(X, n_clusters) rep(n_clusters, nrow(X))
  X <- matrix(0, 3, 1)
  expect_identical(call_estimator(with_seed_arg, X, list(n_clusters = 2L), random_state = 9L), rep(9L, 3))
  expect_identical(call_estimator(without, X, list(n_clusters = 2L), random_state = 9L), rep(2L, 3))
})

test_that("call_estimator checks what the estimator returns", {
  X <- matrix(0, 3, 1)
  expect_error(
    call_estimator(function(X) 1:2, X, list()),
    "A clustering function must return one integer label per row of X (3).",
    fixed = TRUE
  )
  expect_error(
    call_estimator(function(X) c(1.5, 1, 1), X, list()),
    "A clustering function must return one integer label per row of X (3).",
    fixed = TRUE
  )
  expect_identical(call_estimator(function(X) factor(c("a", "b", "a")), X, list()), c(1L, 2L, 1L))
})

test_that("the default classifier predicts separable classes", {
  d <- make_blobs()
  train <- seq(1L, 90L, by = 2L)
  test <- seq(2L, 90L, by = 2L)
  predict_fn <- default_generalizability_classifier(NULL, n_features = 2L, n_trees = 50L, random_state = 1L, n_threads = 1L)
  expect_identical(predict_fn(d$X[train, ], d$y[train], d$X[test, ]), d$y[test])
})

test_that("the default classifier gives the same predictions at any thread count", {
  d <- make_blobs(sd = 2)
  train <- seq(1L, 90L, by = 2L)
  test <- seq(2L, 90L, by = 2L)
  one <- default_generalizability_classifier(NULL, 2L, 50L, random_state = 1L, n_threads = 1L)
  two <- default_generalizability_classifier(NULL, 2L, 50L, random_state = 1L, n_threads = 2L)
  expect_identical(
    one(d$X[train, ], d$y[train], d$X[test, ]),
    two(d$X[train, ], d$y[train], d$X[test, ])
  )
})

test_that("a single training class predicts that class", {
  predict_fn <- default_generalizability_classifier(NULL, 2L, 10L, 1L, 1L)
  expect_identical(predict_fn(matrix(0, 3, 2), c(4L, 4L, 4L), matrix(1, 2, 2)), c(4L, 4L))
})

test_that("a custom classifier gets n_threads and random_state when it takes them", {
  seen <- new.env()
  clf <- function(x_train, y_train, x_test, n_threads, random_state) {
    seen$args <- c(n_threads, random_state)
    rep(y_train[1], nrow(x_test))
  }
  predict_fn <- default_generalizability_classifier(clf, 2L, 100L, random_state = 5L, n_threads = 3L)
  expect_identical(predict_fn(matrix(0, 2, 2), c(1L, 2L), matrix(0, 4, 2)), rep(1L, 4))
  expect_identical(seen$args, c(3L, 5L))
  plain <- function(x_train, y_train, x_test) rep(y_train[1], nrow(x_test))
  plain_fn <- default_generalizability_classifier(plain, 2L, 100L, 5L, 3L)
  expect_identical(plain_fn(matrix(0, 2, 2), c(1L, 2L), matrix(0, 4, 2)), rep(1L, 4))
})

test_that("a custom classifier must return one label per test row", {
  bad <- function(x_train, y_train, x_test) 1L
  predict_fn <- default_generalizability_classifier(bad, 2L, 100L, 5L, 1L)
  expect_error(
    predict_fn(matrix(0, 2, 2), c(1L, 2L), matrix(0, 4, 2)),
    "A classifier must return one label per row of x_test (4).",
    fixed = TRUE
  )
})
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "utils", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "adjusted_rand_index"` and the like.

- [ ] Step 3: Append to `R/utils.R`

```r
# sklearn's adjusted_rand_score, through the pair confusion matrix. Counts
# are doubles, so very large n loses exactness but not accuracy.
adjusted_rand_index <- function(a, b) {
  n <- length(a)
  counts <- as.numeric(table(a, b))
  sum_squares <- sum(counts^2)
  n_rows <- as.numeric(table(a))
  n_cols <- as.numeric(table(b))
  tp <- sum_squares - n
  fp <- sum(n_cols^2) - sum_squares
  fn <- sum(n_rows^2) - sum_squares
  tn <- n^2 - fp - fn - sum_squares
  if (fn == 0 && fp == 0) {
    return(1)
  }
  2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
}

# Renames labels to the reference values by maximum-overlap (Hungarian)
# matching. Labels with no partner, when there are more clusters than
# reference values, take fresh ids past the largest matched reference
# value, so they never merge with a matched cluster. Only positions where
# keep is TRUE take part in the matching.
align_cluster_labels <- function(reference_labels, labels, keep = NULL) {
  if (is.null(keep)) {
    keep <- rep(TRUE, length(labels))
  }
  ref <- reference_labels[keep]
  lab <- labels[keep]
  ref_classes <- sort(unique(ref))
  pred_classes <- sort(unique(lab))
  overlap <- table(factor(ref, levels = ref_classes), factor(lab, levels = pred_classes))
  overlap <- matrix(as.numeric(overlap), nrow = length(ref_classes))

  mapping <- rep(NA_real_, length(pred_classes))
  if (nrow(overlap) <= ncol(overlap)) {
    cols <- as.integer(clue::solve_LSAP(overlap, maximum = TRUE))
    mapping[cols] <- ref_classes
  } else {
    rows <- as.integer(clue::solve_LSAP(t(overlap), maximum = TRUE))
    mapping <- as.numeric(ref_classes[rows])
  }

  all_pred <- sort(unique(labels))
  full <- mapping[match(all_pred, pred_classes)]
  next_id <- max(ref_classes) + 1
  for (i in which(is.na(full))) {
    full[i] <- next_id
    next_id <- next_id + 1
  }
  aligned <- full[match(labels, all_pred)]
  if (is.integer(reference_labels)) as.integer(aligned) else aligned
}

n_cores <- function() {
  n <- parallel::detectCores(logical = TRUE)
  if (is.na(n) || n < 1L) {
    n <- 1L
  }
  limit <- Sys.getenv("_R_CHECK_LIMIT_CORES_")
  if (nzchar(limit) && !identical(tolower(limit), "false")) {
    n <- min(n, 2L)
  }
  as.integer(n)
}

# Splits n_jobs into resample workers (outer) and classifier threads per
# worker (inner), following joblib's convention for n_jobs: a positive
# count, -1 for every core, -2 for all but one, NULL for one worker. The
# worker count is capped at n_resamples.
resolve_core_budget <- function(n_jobs, n_resamples) {
  cores <- n_cores()
  if (is.null(n_jobs)) {
    outer <- 1L
  } else if (n_jobs == 0) {
    stop("n_jobs == 0 has no meaning; use 1 or a negative count", call. = FALSE)
  } else if (n_jobs < 0) {
    outer <- max(cores + 1L + as.integer(n_jobs), 1L)
  } else {
    outer <- as.integer(n_jobs)
  }
  outer <- max(1L, min(outer, as.integer(n_resamples)))
  c(outer = outer, inner = max(1L, cores %/% outer))
}

# A neighbor count chosen for the full data reaches further on a subsample.
# Scaling it by n_fit / n_full keeps the neighborhood the same size. The
# count comes from params or else from the estimator's default; it is
# rounded half to even, as Python's round() does, never set below 2 and
# never raised.
scale_neighbor_count <- function(estimator, params, n_fit, n_full) {
  if (n_fit >= n_full) {
    return(params)
  }
  base <- if ("n_neighbors" %in% names(params)) {
    params$n_neighbors
  } else if ("n_neighbors" %in% names(formals(estimator))) {
    formals(estimator)$n_neighbors
  } else {
    NULL
  }
  if (!is.numeric(base) || length(base) != 1L || is.na(base) || base != round(base)) {
    return(params)
  }
  scaled <- max(2, round(base * n_fit / n_full))
  params$n_neighbors <- as.integer(min(base, scaled))
  params
}

# Runs a clustering function on X. random_state is passed only when the
# function has that argument and the grid does not set it.
call_estimator <- function(estimator, X, params, random_state = NULL) {
  args <- c(list(X), params)
  if ("random_state" %in% names(formals(estimator)) && !"random_state" %in% names(params)) {
    args["random_state"] <- list(random_state)
  }
  labels <- do.call(estimator, args)
  if (is.factor(labels)) {
    labels <- as.integer(labels)
  }
  if (!is.numeric(labels) || length(labels) != nrow(X) || anyNA(labels) ||
      any(labels != round(labels))) {
    stop(sprintf(
      "A clustering function must return one integer label per row of X (%d).",
      nrow(X)
    ), call. = FALSE)
  }
  as.integer(labels)
}

# Builds the function that predicts held-out labels. The default is a
# ranger forest with the settings of Python's default classifier: n_trees
# trees, depth at most n_features, floor(sqrt(n_features)) candidate
# features per split. A custom classifier gets n_threads and random_state
# only when it has those arguments.
default_generalizability_classifier <- function(classifier, n_features, n_trees,
                                                random_state, n_threads) {
  check <- function(predicted, x_test) {
    if (length(predicted) != nrow(x_test)) {
      stop(sprintf(
        "A classifier must return one label per row of x_test (%d).",
        nrow(x_test)
      ), call. = FALSE)
    }
    as.integer(predicted)
  }
  if (is.null(classifier)) {
    return(function(x_train, y_train, x_test) {
      check(ranger_predict(
        x_train, y_train, x_test,
        n_trees = n_trees,
        mtry = max(1L, as.integer(floor(sqrt(n_features)))),
        max_depth = as.integer(n_features),
        seed = random_state,
        n_threads = n_threads
      ), x_test)
    })
  }
  arguments <- names(formals(classifier))
  function(x_train, y_train, x_test) {
    args <- list(x_train = x_train, y_train = y_train, x_test = x_test)
    if ("n_threads" %in% arguments) {
      args$n_threads <- n_threads
    }
    if ("random_state" %in% arguments) {
      args["random_state"] <- list(random_state)
    }
    check(do.call(classifier, args), x_test)
  }
}

ranger_predict <- function(x_train, y_train, x_test, n_trees, mtry, max_depth,
                           seed, n_threads) {
  classes <- sort(unique(y_train))
  if (length(classes) == 1L) {
    return(rep(classes, nrow(x_test)))
  }
  # ranger finds no covariates in a matrix without column names.
  feature_names <- paste0("x", seq_len(ncol(x_train)))
  colnames(x_train) <- feature_names
  colnames(x_test) <- feature_names
  forest <- ranger::ranger(
    x = x_train,
    y = factor(y_train, levels = classes),
    num.trees = n_trees,
    mtry = mtry,
    max.depth = max_depth,
    seed = seed,
    num.threads = n_threads,
    verbose = FALSE
  )
  predicted <- stats::predict(forest, data = x_test, num.threads = n_threads)$predictions
  as.integer(as.character(predicted))
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "utils", stop_on_failure = TRUE)'`
Expected: PASS, no warnings.

- [ ] Step 5: Mutation check

- In `adjusted_rand_index()`, drop the `fn == 0 && fp == 0` branch: the fixture case `[0, 0, 0]` against `[0, 0, 0]` gives NaN and "adjusted_rand_index matches sklearn" fails.
- In `align_cluster_labels()`, set `next_id <- max(ref_classes)`: the `more_clusters` fixture case fails.
- In `resolve_core_budget()`, drop the `min(outer, n_resamples)` cap: the fixture test fails on the `n_resamples = 3` cases.
- In `scale_neighbor_count()`, replace `round` with `ceiling`: the fixture test fails (the 4.5 case).
- In `ranger_predict()`, drop `seed = seed`: "the default classifier gives the same predictions at any thread count" fails, or passes by chance; if it passes, report it.

- [ ] Step 6: Commit

```bash
git add R/utils.R tests/testthat/test-utils.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add ARI, label alignment, the core budget and classifier calls

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Run modes and estimator grids

Files:
- Create: `carve-r/R/types.R`
- Test: `carve-r/tests/testthat/test-types.R`

Interfaces:
- Consumes: `format_repr`, `format_param_value` (Task 3); the `sweep` fixture.
- Produces:
  - `resolve_mode(mode)` (internal) returns `list(mode, run_stability, run_generalizability, compute_average_ari)`.
  - `estimator_grid(estimator, ..., name = NULL)` (exported) returns `structure(list(estimator = <function>, name = <chr>, grid = <named list>), class = "carve_estimator_grid")`. Factor grid values are stored as character.
  - `print.carve_estimator_grid(x, ...)` (registered S3 method).
  - `expand_param_grid(grid)` (internal) returns a list of named parameter lists in sklearn's `ParameterGrid` order: keys sorted in C-locale order, the last key varying fastest. An empty grid gives `list(list())`.

- [ ] Step 1: Write the failing tests in `tests/testthat/test-types.R`

```r
test_that("resolve_mode maps each mode to the stages it runs", {
  expect_identical(
    resolve_mode("default"),
    list(mode = "default", run_stability = TRUE, run_generalizability = TRUE, compute_average_ari = TRUE)
  )
  expect_identical(
    resolve_mode("stability"),
    list(mode = "stability", run_stability = TRUE, run_generalizability = FALSE, compute_average_ari = FALSE)
  )
  expect_identical(
    resolve_mode("generalizability"),
    list(mode = "generalizability", run_stability = FALSE, run_generalizability = TRUE, compute_average_ari = FALSE)
  )
  expect_error(
    resolve_mode("both"),
    "Unknown mode: 'both'. Expected one of 'default', 'stability', 'generalizability'.",
    fixed = TRUE
  )
})

test_that("estimator_grid records the function, its name and the grid", {
  toy <- function(X, n_clusters = 2L, linkage = "ward") rep(1L, nrow(X))
  g <- estimator_grid(toy, n_clusters = 2:4, linkage = c("ward", "average"))
  expect_s3_class(g, "carve_estimator_grid")
  expect_identical(g$name, "toy")
  expect_identical(g$estimator, toy)
  expect_identical(g$grid, list(n_clusters = 2:4, linkage = c("ward", "average")))
  expect_identical(estimator_grid(toy, linkage = factor("ward"))$grid$linkage, "ward")
})

test_that("estimator_grid takes the name after :: and accepts name=", {
  expect_identical(estimator_grid(stats::median)$name, "median")
  expect_identical(estimator_grid(function(X) NULL, name = "anon")$name, "anon")
})

test_that("estimator_grid rejects what it cannot use", {
  toy <- function(X, n_clusters = 2L) NULL
  expect_error(estimator_grid("KMeans"), "estimator must be a function.", fixed = TRUE)
  expect_error(
    estimator_grid(function(X) NULL),
    "Pass name= for an estimator that is not a named function.",
    fixed = TRUE
  )
  expect_error(
    estimator_grid(toy, 2:4),
    "Every grid entry must be named after an argument of the estimator.",
    fixed = TRUE
  )
  expect_error(estimator_grid(toy, k = 2:4), "toy has no argument 'k'.", fixed = TRUE)
  expect_error(
    estimator_grid(toy, n_clusters = integer()),
    "Every grid entry needs at least one value.",
    fixed = TRUE
  )
})

test_that("printing a grid lists its values", {
  toy <- function(X, n_clusters = 2L, linkage = "ward") NULL
  expect_output(
    print(estimator_grid(toy, n_clusters = 2:3, linkage = "ward")),
    "Estimator grid for toy\n  n_clusters: 2, 3\n  linkage: ward",
    fixed = TRUE
  )
})

test_that("expand_param_grid follows sklearn's ParameterGrid order", {
  fx <- read_fixture("sweep")
  expect_identical(expand_param_grid(fx$grid), fx$grid_order)
  expect_identical(expand_param_grid(list()), list(list()))
})
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "types", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "resolve_mode"`.

- [ ] Step 3: Write `R/types.R`

```r
# Run modes and estimator grid specifications. Mirrors _types.py; the grid
# helpers are the R forms of Python's (EstimatorClass, param_grid) tuples
# and sklearn's ParameterGrid.

resolve_mode <- function(mode) {
  if (!is.character(mode) || length(mode) != 1L ||
      !mode %in% c("default", "stability", "generalizability")) {
    stop(sprintf(
      "Unknown mode: %s. Expected one of 'default', 'stability', 'generalizability'.",
      format_repr(mode)
    ), call. = FALSE)
  }
  list(
    mode = mode,
    run_stability = mode != "generalizability",
    run_generalizability = mode != "stability",
    compute_average_ari = mode == "default"
  )
}

#' Estimator parameter grid
#'
#' Pairs a clustering function with the values to try for each of its
#' arguments. [carve()] evaluates every combination of the values. This is the
#' R form of the `(EstimatorClass, param_grid)` tuples the Python package
#' takes.
#'
#' @param estimator A clustering function. Its first argument is the data
#'   matrix, one row per sample; its other arguments are the parameters. It
#'   returns one integer label per row, with -1 for noise. If it has a
#'   `random_state` argument, CARVE passes each resample's seed to it.
#' @param ... Values to try, each named after an argument of `estimator`:
#'   a vector, or a list for values that are not scalars.
#' @param name Name recorded in the results table. Defaults to the name the
#'   function was passed under, so an anonymous function needs one.
#' @param x A `carve_estimator_grid` object.
#'
#' @return An object of class `carve_estimator_grid`: a list with the
#'   function, its name and the grid.
#' @seealso [KMeans()], [AgglomerativeClustering()] and [SpectralClustering()]
#'   for the built-in estimators.
#' @examples
#' estimator_grid(KMeans, n_clusters = 2:6)
#' estimator_grid(AgglomerativeClustering, n_clusters = 2:6,
#'                linkage = c("ward", "average"))
#' @export
estimator_grid <- function(estimator, ..., name = NULL) {
  if (!is.function(estimator)) {
    stop("estimator must be a function.", call. = FALSE)
  }
  if (is.null(name)) {
    expr <- substitute(estimator)
    if (is.symbol(expr)) {
      name <- as.character(expr)
    } else if (is.call(expr) && as.character(expr[[1L]]) %in% c("::", ":::")) {
      name <- as.character(expr[[3L]])
    } else {
      stop("Pass name= for an estimator that is not a named function.", call. = FALSE)
    }
  }
  grid <- list(...)
  if (length(grid) > 0L && (is.null(names(grid)) || any(names(grid) == ""))) {
    stop("Every grid entry must be named after an argument of the estimator.", call. = FALSE)
  }
  arguments <- names(formals(estimator))
  if (!"..." %in% arguments) {
    unknown <- setdiff(names(grid), arguments[-1L])
    if (length(unknown) > 0L) {
      stop(sprintf(
        "%s has no argument %s.",
        name,
        paste0("'", unknown, "'", collapse = ", ")
      ), call. = FALSE)
    }
  }
  if (any(lengths(grid) == 0L)) {
    stop("Every grid entry needs at least one value.", call. = FALSE)
  }
  grid <- lapply(grid, function(values) if (is.factor(values)) as.character(values) else values)
  structure(list(estimator = estimator, name = name, grid = grid), class = "carve_estimator_grid")
}

#' @rdname estimator_grid
#' @export
print.carve_estimator_grid <- function(x, ...) {
  cat("Estimator grid for ", x$name, "\n", sep = "")
  for (param in names(x$grid)) {
    values <- vapply(as.list(x$grid[[param]]), format_param_value, character(1))
    cat("  ", param, ": ", paste(values, collapse = ", "), "\n", sep = "")
  }
  invisible(x)
}

# sklearn's ParameterGrid order: keys sorted, the last key varying fastest.
# config_id follows this order, so it must match the Python package.
expand_param_grid <- function(grid) {
  if (length(grid) == 0L) {
    return(list(list()))
  }
  keys <- sort(names(grid), method = "radix")
  sizes <- lengths(grid[keys])
  combos <- expand.grid(lapply(rev(sizes), seq_len), KEEP.OUT.ATTRS = FALSE)
  combos <- combos[, rev(seq_along(keys)), drop = FALSE]
  lapply(seq_len(nrow(combos)), function(r) {
    params <- lapply(seq_along(keys), function(j) grid[[keys[j]]][[combos[r, j]]])
    names(params) <- keys
    params
  })
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::document(); devtools::test(filter = "types", stop_on_failure = TRUE)'`
Expected: `man/estimator_grid.Rd` written; tests PASS.

- [ ] Step 5: Mutation check

- In `expand_param_grid()`, drop the `rev(...)` reordering of `combos` (so the first key varies fastest): the ParameterGrid test fails.
- In `estimator_grid()`, drop the `"::"` branch: "estimator_grid takes the name after ::" fails.

- [ ] Step 6: Commit

```bash
git add R/types.R tests/testthat/test-types.R NAMESPACE man/estimator_grid.Rd
git commit -m "$(cat <<'EOF'
feat(carve-r): add run modes and estimator grids

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: The sweep axis

Files:
- Create: `carve-r/R/AllClasses.R`, `carve-r/R/sweep.R`
- Test: `carve-r/tests/testthat/test-sweep.R`

Interfaces:
- Consumes: `coerce_n_clusters`, `format_repr`, `python_list`, `format_param_value`, `title_case` (Task 3); `estimator_grid` (Task 5, in tests); the `sweep` fixture.
- Produces:
  - S4 class `SweepSpec` (exported) with slots `param` (character), `values` (numeric), `finer_is_larger` (logical), `fixes_k` (logical), `label` (character), and a `show` method.
  - Class unions `carveListOrNULL` and `carveIntegerOrNULL` (used by Task 14).
  - Internal: `SWEEP_REGISTRY`, `SWEEP_META_COLS`, `IDENTITY_COLS`; `coerce_sweep_values(value, param, expand_scalar = TRUE)`; `resolve_sweep(n_clusters = NULL, resolution = NULL, sweep = NULL, sweep_values = NULL, finer_is_larger = NULL)` returns a `SweepSpec`; `sweep_ranks(spec)` returns 0-based integer ranks, coarse to fine; `sweep_rank_of(spec, value)` returns one rank; `isclose(a, b)`; `infer_sweep_param(grids)` returns a parameter name or `NULL`; `grid_sweep_values(grids, param)`; `validate_grids(grids, spec)` returns the sweep values in the grids; `sweep_param_name(results)`, `sweep_exclude_cols(results)`, `sweep_axis_label(results)`, `observed_k_series(results)` (rounded doubles), `observed_k(row)` (integer); `format_method_label(est_name, params, sweep_param)`; `method_id_assigner(sweep_param)` returns `function(est_name, params)` giving `c(method_id, method_label)`; `config_id_of(row)` returns an integer.

- [ ] Step 1: Write the failing tests in `tests/testthat/test-sweep.R`

```r
toy <- function(X, n_clusters = 2L, resolution = 1, linkage = "ward") rep(1L, nrow(X))

test_that("resolve_sweep defaults to n_clusters", {
  s <- resolve_sweep(n_clusters = 4)
  expect_s4_class(s, "SweepSpec")
  expect_identical(s@param, "n_clusters")
  expect_identical(s@values, 2:4)
  expect_true(s@finer_is_larger)
  expect_true(s@fixes_k)
  expect_identical(s@label, "Number of Clusters (k)")
})

test_that("resolution switches the axis and sorts the values", {
  s <- resolve_sweep(resolution = c(1, 0.5))
  expect_identical(s@param, "resolution")
  expect_identical(s@values, c(0.5, 1))
  expect_false(s@fixes_k)
})

test_that("min_cluster_size runs from fine to coarse", {
  s <- resolve_sweep(sweep = "min_cluster_size", sweep_values = c(20, 5, 10))
  expect_identical(s@values, c(5L, 10L, 20L))
  expect_false(s@finer_is_larger)
  expect_identical(sweep_ranks(s), c(2L, 1L, 0L))
})

test_that("sweep ranks match Python", {
  for (case in read_fixture("sweep")$ranks) {
    spec <- switch(case$param,
      n_clusters = resolve_sweep(n_clusters = case$values),
      resolution = resolve_sweep(resolution = case$values),
      min_cluster_size = resolve_sweep(sweep = "min_cluster_size", sweep_values = case$values)
    )
    expect_equal(spec@values, case$values, info = case$param)
    expect_identical(sweep_ranks(spec), as.integer(case$ranks), info = case$param)
  }
})

test_that("resolve_sweep rejects inconsistent arguments", {
  expect_error(
    resolve_sweep(resolution = 0.5, sweep = "n_clusters"),
    "resolution= was given but sweep='n_clusters'. Pass sweep_values= instead, or drop resolution=.",
    fixed = TRUE
  )
  expect_error(
    resolve_sweep(sweep = "resolution"),
    "No values supplied for sweep parameter 'resolution'. Pass sweep_values=, or supply estimator grids that contain it.",
    fixed = TRUE
  )
  expect_error(
    resolve_sweep(sweep = "eps", sweep_values = c(0.1, 0.2)),
    "Unknown sweep parameter 'eps'. Pass finer_is_larger= to declare whether larger values yield more clusters.",
    fixed = TRUE
  )
})

test_that("an unknown parameter gets a title-case label and the given direction", {
  s <- resolve_sweep(sweep = "max_eps", sweep_values = c(0.2, 0.1), finer_is_larger = FALSE)
  expect_identical(s@label, "Max Eps")
  expect_false(s@finer_is_larger)
  expect_false(s@fixes_k)
})

test_that("finer_is_larger overrides the registry", {
  expect_false(resolve_sweep(resolution = 1, finer_is_larger = FALSE)@finer_is_larger)
})

test_that("a single value passed as sweep_values is not expanded", {
  expect_identical(resolve_sweep(n_clusters = 2:10, sweep_values = 3L)@values, 3L)
  expect_identical(resolve_sweep(n_clusters = 3)@values, 2:3)
})

test_that("coerce_sweep_values validates each axis", {
  expect_error(coerce_sweep_values(c(0, 1), "resolution"), "All resolution values must be > 0.", fixed = TRUE)
  expect_error(coerce_sweep_values(c(2.5, 3), "min_cluster_size"), "min_cluster_size values must be integers.", fixed = TRUE)
  expect_error(coerce_sweep_values(c(1, 3), "min_cluster_size"), "All min_cluster_size values must be >= 2.", fixed = TRUE)
  expect_error(coerce_sweep_values(matrix(1:4, 2), "resolution"), "resolution must be a numeric vector.", fixed = TRUE)
  expect_identical(coerce_sweep_values(c(20, 5), "min_cluster_size"), c(5L, 20L))
})

test_that("sweep_rank_of matches values up to rounding and rejects others", {
  s <- resolve_sweep(resolution = c(0.1, 0.2, 0.3))
  expect_identical(sweep_rank_of(s, 0.1 + 0.2), 2L)
  expect_error(
    sweep_rank_of(s, 0.5),
    "0.5 is not among the swept resolution values [0.1, 0.2, 0.3].",
    fixed = TRUE
  )
})

test_that("infer_sweep_param finds the one swept registry parameter", {
  expect_identical(infer_sweep_param(list(estimator_grid(toy, n_clusters = 2:4, linkage = "ward"))), "n_clusters")
  expect_null(infer_sweep_param(list(estimator_grid(toy, n_clusters = 3L))))
  expect_error(
    infer_sweep_param(list(estimator_grid(toy, n_clusters = 2:3), estimator_grid(toy, resolution = c(0.5, 1)))),
    "Estimator grids sweep more than one parameter (['n_clusters', 'resolution']). CARVE evaluates exactly one sweep axis per run; k-based and resolution-based estimators cannot be compared in the same run.",
    fixed = TRUE
  )
})

test_that("grid_sweep_values reads the first grid that has the parameter", {
  grids <- list(estimator_grid(toy, linkage = "ward"), estimator_grid(toy, n_clusters = 2:3))
  expect_identical(grid_sweep_values(grids, "n_clusters"), 2:3)
  expect_null(grid_sweep_values(grids, "resolution"))
})

test_that("validate_grids checks every grid against the axis", {
  s <- resolve_sweep(n_clusters = 2:3)
  expect_identical(validate_grids(list(estimator_grid(toy, n_clusters = 2:3)), s), 2:3)
  expect_identical(validate_grids(list(estimator_grid(toy, n_clusters = 3L)), s), 3L)
  expect_error(validate_grids(list(), s), "estimator_param_grids must not be empty.", fixed = TRUE)
  expect_error(
    validate_grids(list(estimator_grid(toy, linkage = "ward")), s),
    "Estimator grid for toy does not contain the sweep parameter 'n_clusters'. All grids in a run must sweep the same parameter.",
    fixed = TRUE
  )
  expect_error(
    validate_grids(list(estimator_grid(toy, n_clusters = 2:3, resolution = c(0.5, 1))), s),
    "Estimator grid for toy mixes sweep parameters 'n_clusters' and ['resolution']. CARVE evaluates exactly one sweep axis per run; k-based and resolution-based estimators cannot be compared in the same run.",
    fixed = TRUE
  )
  expect_error(
    validate_grids(list(estimator_grid(toy, n_clusters = 2:3), estimator_grid(toy, n_clusters = 2:4)), s),
    "All estimator parameter grids must contain the same n_clusters values.",
    fixed = TRUE
  )
})

test_that("format_method_label matches Python", {
  for (case in read_fixture("sweep")$labels) {
    expect_identical(format_method_label(case$estimator, case$params, case$sweep_param), case$label)
  }
})

test_that("method ids are shared along the sweep and numbered in first-seen order", {
  assign_id <- method_id_assigner("n_clusters")
  expect_identical(assign_id("KMeans", list(n_clusters = 2L)), c("m0", "KMeans"))
  expect_identical(assign_id("KMeans", list(n_clusters = 3L)), c("m0", "KMeans"))
  expect_identical(
    assign_id("AgglomerativeClustering", list(linkage = "ward", n_clusters = 2L)),
    c("m1", "AgglomerativeClustering, linkage=ward")
  )
  expect_identical(
    assign_id("AgglomerativeClustering", list(linkage = "average", n_clusters = 2L)),
    c("m2", "AgglomerativeClustering, linkage=average")
  )
  expect_identical(assign_id("KMeans", list(n_clusters = 4L)), c("m0", "KMeans"))
})

test_that("observed_k rounds half to even, like Python", {
  expect_identical(observed_k(data.frame(n_clusters_observed = 2.5)), 2L)
  expect_identical(observed_k(data.frame(n_clusters_observed = 3.5)), 4L)
  expect_identical(observed_k_series(data.frame(n_clusters_observed = c(2.4, 2.6))), c(2, 3))
})

test_that("the results-table helpers name the bookkeeping columns and the axis", {
  df <- data.frame(sweep_param = "resolution", config_id = 4L)
  expect_setequal(
    sweep_exclude_cols(df),
    c("sweep_param", "sweep_value", "sweep_rank", "n_clusters_observed", "n_clusters_observed_se",
      "noise_fraction", "config_id", "method_id", "method_label", "resolution")
  )
  expect_identical(sweep_param_name(df), "resolution")
  expect_identical(sweep_axis_label(df), "Resolution")
  expect_identical(sweep_axis_label(data.frame(sweep_param = "max_eps")), "Max Eps")
  expect_identical(config_id_of(df), 4L)
})

test_that("show prints the axis", {
  expect_output(
    show(resolve_sweep(n_clusters = 3)),
    "SweepSpec: n_clusters over 2, 3 (larger values give more clusters)",
    fixed = TRUE
  )
})
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "sweep", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "resolve_sweep"`.

- [ ] Step 3: Write `R/AllClasses.R`

```r
# S4 classes of the package.

setClassUnion("carveListOrNULL", c("list", "NULL"))
setClassUnion("carveIntegerOrNULL", c("integer", "NULL"))

#' Sweep axis of a CARVE run
#'
#' Records the hyperparameter a [carve()] run swept, the values it tried and
#' whether larger values give more clusters. [sweep_spec()] returns it.
#'
#' @slot param Name of the swept parameter, such as `"n_clusters"`.
#' @slot values The values tried.
#' @slot finer_is_larger `TRUE` when larger values give more clusters.
#' @slot fixes_k `TRUE` when the parameter sets the number of clusters
#'   directly, which only `n_clusters` does.
#' @slot label Axis label for plots.
#' @export
setClass(
  "SweepSpec",
  slots = c(
    param = "character",
    values = "numeric",
    finer_is_larger = "logical",
    fixes_k = "logical",
    label = "character"
  )
)
```

- [ ] Step 4: Write `R/sweep.R`

```r
#' @include AllClasses.R
NULL

# The sweep axis of a run. Mirrors _sweep.py. A run sweeps exactly one
# parameter: metrics on a k axis and on a resolution axis are not
# comparable, so the two kinds of estimator cannot share a run.

# Bookkeeping and identity columns of the results table. They are not
# estimator parameters and stay out of labels and plot groupings.
SWEEP_META_COLS <- c(
  "sweep_param", "sweep_value", "sweep_rank",
  "n_clusters_observed", "n_clusters_observed_se", "noise_fraction"
)
IDENTITY_COLS <- c("config_id", "method_id", "method_label")

SWEEP_REGISTRY <- list(
  n_clusters = list(finer_is_larger = TRUE, fixes_k = TRUE, label = "Number of Clusters (k)"),
  resolution = list(finer_is_larger = TRUE, fixes_k = FALSE, label = "Resolution"),
  min_cluster_size = list(finer_is_larger = FALSE, fixes_k = FALSE, label = "Minimum Cluster Size")
)

coerce_sweep_values <- function(value, param, expand_scalar = TRUE) {
  if (param == "n_clusters") {
    return(coerce_n_clusters(value, expand_scalar = expand_scalar))
  }
  if (param == "min_cluster_size") {
    if (!is.numeric(value) || !is.null(dim(value)) || anyNA(value) || any(value != round(value))) {
      stop("min_cluster_size values must be integers.", call. = FALSE)
    }
    if (any(value < 2)) {
      stop("All min_cluster_size values must be >= 2.", call. = FALSE)
    }
    return(sort(as.integer(value)))
  }
  if (!is.numeric(value) || !is.null(dim(value))) {
    stop(sprintf("%s must be a numeric vector.", param), call. = FALSE)
  }
  if (anyNA(value) || any(value <= 0)) {
    stop(sprintf("All %s values must be > 0.", param), call. = FALSE)
  }
  sort(as.numeric(value))
}

resolve_sweep <- function(n_clusters = NULL, resolution = NULL, sweep = NULL,
                          sweep_values = NULL, finer_is_larger = NULL) {
  if (!is.null(resolution) && !is.null(sweep) && sweep != "resolution") {
    stop(sprintf(
      "resolution= was given but sweep=%s. Pass sweep_values= instead, or drop resolution=.",
      format_repr(sweep)
    ), call. = FALSE)
  }
  if (is.null(sweep)) {
    sweep <- if (!is.null(resolution)) "resolution" else "n_clusters"
  }
  # A single n_clusters value K means 2:K; a value read from sweep_values
  # (an estimator grid) is kept as it is.
  expand_scalar <- is.null(sweep_values)
  if (is.null(sweep_values)) {
    if (sweep == "n_clusters") {
      sweep_values <- n_clusters
    } else if (sweep == "resolution") {
      sweep_values <- resolution
    }
  }
  if (is.null(sweep_values)) {
    stop(sprintf(
      "No values supplied for sweep parameter %s. Pass sweep_values=, or supply estimator grids that contain it.",
      format_repr(sweep)
    ), call. = FALSE)
  }
  values <- coerce_sweep_values(sweep_values, sweep, expand_scalar = expand_scalar)
  known <- SWEEP_REGISTRY[[sweep]]
  if (is.null(known)) {
    if (is.null(finer_is_larger)) {
      stop(sprintf(
        "Unknown sweep parameter %s. Pass finer_is_larger= to declare whether larger values yield more clusters.",
        format_repr(sweep)
      ), call. = FALSE)
    }
    known <- list(
      finer_is_larger = isTRUE(finer_is_larger),
      fixes_k = FALSE,
      label = title_case(gsub("_", " ", sweep, fixed = TRUE))
    )
  }
  methods::new(
    "SweepSpec",
    param = sweep,
    values = values,
    finer_is_larger = if (is.null(finer_is_larger)) known$finer_is_larger else isTRUE(finer_is_larger),
    fixes_k = known$fixes_k,
    label = known$label
  )
}

# Coarse-to-fine rank of each value, 0-based as in Python.
sweep_ranks <- function(spec) {
  keys <- if (spec@finer_is_larger) spec@values else -spec@values
  ranks <- integer(length(keys))
  ranks[order(keys)] <- seq_along(keys) - 1L
  ranks
}

# numpy.isclose with its default tolerances.
isclose <- function(a, b, rtol = 1e-5, atol = 1e-8) {
  abs(a - b) <= atol + rtol * abs(b)
}

sweep_rank_of <- function(spec, value) {
  matches <- which(isclose(as.numeric(spec@values), as.numeric(value)))
  if (length(matches) == 0L) {
    stop(sprintf(
      "%s is not among the swept %s values [%s].",
      format_repr(value),
      spec@param,
      paste(vapply(spec@values, format_repr, character(1)), collapse = ", ")
    ), call. = FALSE)
  }
  sweep_ranks(spec)[matches[1L]]
}

infer_sweep_param <- function(estimator_grids) {
  swept <- unique(unlist(lapply(estimator_grids, function(g) {
    params <- intersect(names(g$grid), names(SWEEP_REGISTRY))
    params[lengths(g$grid[params]) > 1L]
  })))
  if (length(swept) == 0L) {
    return(NULL)
  }
  if (length(swept) > 1L) {
    stop(sprintf(
      "Estimator grids sweep more than one parameter (%s). CARVE evaluates exactly one sweep axis per run; k-based and resolution-based estimators cannot be compared in the same run.",
      python_list(sort(swept, method = "radix"))
    ), call. = FALSE)
  }
  swept
}

grid_sweep_values <- function(estimator_grids, param) {
  for (g in estimator_grids) {
    if (param %in% names(g$grid)) {
      return(g$grid[[param]])
    }
  }
  NULL
}

validate_grids <- function(estimator_grids, sweep) {
  if (length(estimator_grids) == 0L) {
    stop("estimator_param_grids must not be empty.", call. = FALSE)
  }
  others <- setdiff(names(SWEEP_REGISTRY), sweep@param)
  for (g in estimator_grids) {
    if (!sweep@param %in% names(g$grid)) {
      stop(sprintf(
        "Estimator grid for %s does not contain the sweep parameter %s. All grids in a run must sweep the same parameter.",
        g$name, format_repr(sweep@param)
      ), call. = FALSE)
    }
    clash <- intersect(others, names(g$grid))
    clash <- clash[lengths(g$grid[clash]) > 1L]
    if (length(clash) > 0L) {
      stop(sprintf(
        "Estimator grid for %s mixes sweep parameters %s and %s. CARVE evaluates exactly one sweep axis per run; k-based and resolution-based estimators cannot be compared in the same run.",
        g$name, format_repr(sweep@param), python_list(sort(clash, method = "radix"))
      ), call. = FALSE)
    }
  }
  reference <- estimator_grids[[1L]]$grid[[sweep@param]]
  for (g in estimator_grids[-1L]) {
    if (!identical(as.numeric(reference), as.numeric(g$grid[[sweep@param]]))) {
      stop(sprintf(
        "All estimator parameter grids must contain the same %s values.",
        sweep@param
      ), call. = FALSE)
    }
  }
  coerce_sweep_values(reference, sweep@param, expand_scalar = FALSE)
}

# Results-table helpers.
sweep_param_name <- function(results) {
  as.character(results$sweep_param[[1L]])
}

sweep_exclude_cols <- function(results) {
  c(SWEEP_META_COLS, IDENTITY_COLS, sweep_param_name(results))
}

sweep_axis_label <- function(results) {
  param <- sweep_param_name(results)
  known <- SWEEP_REGISTRY[[param]]
  if (is.null(known)) title_case(gsub("_", " ", param, fixed = TRUE)) else known$label
}

observed_k_series <- function(results) {
  round(results$n_clusters_observed)
}

observed_k <- function(row) {
  as.integer(round(as.numeric(row$n_clusters_observed[[1L]])))
}

config_id_of <- function(row) {
  as.integer(row$config_id[[1L]])
}

# A method is an estimator with every parameter except the swept one: one
# curve in a metric-over-sweep plot.
format_method_label <- function(est_name, params, sweep_param) {
  keys <- setdiff(sort(as.character(names(params)), method = "radix"), sweep_param)
  parts <- vapply(
    keys,
    function(key) paste0(key, "=", format_param_value(params[[key]])),
    character(1),
    USE.NAMES = FALSE
  )
  paste(c(est_name, parts), collapse = ", ")
}

# Two configurations that differ only in the swept parameter share a method
# id. Ids are numbered m0, m1, ... in first-seen order.
method_id_assigner <- function(sweep_param) {
  seen <- new.env(parent = emptyenv())
  count <- 0L
  function(est_name, params) {
    keys <- setdiff(sort(as.character(names(params)), method = "radix"), sweep_param)
    values <- vapply(
      keys,
      function(key) paste(deparse(params[[key]]), collapse = ""),
      character(1),
      USE.NAMES = FALSE
    )
    key <- paste(est_name, paste(keys, values, sep = "=", collapse = ";"), sep = "|")
    if (is.null(seen[[key]])) {
      seen[[key]] <- c(sprintf("m%d", count), format_method_label(est_name, params, sweep_param))
      count <<- count + 1L
    }
    seen[[key]]
  }
}

#' @rdname SweepSpec-class
#' @param object A `SweepSpec` object.
#' @export
setMethod("show", "SweepSpec", function(object) {
  direction <- if (object@finer_is_larger) "larger" else "smaller"
  cat(
    "SweepSpec: ", object@param, " over ",
    paste(vapply(object@values, format_param_value, character(1)), collapse = ", "),
    " (", direction, " values give more clusters)\n",
    sep = ""
  )
  invisible(object)
})
```

- [ ] Step 5: Run the tests and see them pass

Run: `Rscript -e 'devtools::document(); devtools::test(filter = "sweep", stop_on_failure = TRUE)'`
Expected: `man/SweepSpec-class.Rd` written, `DESCRIPTION` gains a `Collate:` field; tests PASS.

- [ ] Step 6: Mutation check

- In `sweep_ranks()`, ignore `finer_is_larger` (always use `spec@values`): "min_cluster_size runs from fine to coarse" and the fixture test fail.
- In `resolve_sweep()`, set `expand_scalar <- TRUE`: "a single value passed as sweep_values is not expanded" fails.
- In `method_id_assigner()`, include the swept parameter in `keys`: the method-id test fails.

- [ ] Step 7: Commit

```bash
git add R/AllClasses.R R/sweep.R tests/testthat/test-sweep.R NAMESPACE DESCRIPTION man/SweepSpec-class.Rd
git commit -m "$(cat <<'EOF'
feat(carve-r): add the sweep axis and method ids

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Selection rules

Files:
- Create: `carve-r/R/selection.R`
- Test: `carve-r/tests/testthat/test-selection.R`

Interfaces:
- Consumes: `observed_k`, `observed_k_series` (Task 6); `format_repr`, `python_list` (Task 3); the `selection` fixture.
- Produces (all internal):
  - `MEASURE_MAP`, a named character vector from alias to results column.
  - `measure_column(measure)` returns the column name or errors.
  - `select_best_row_max(results, measure)`, `select_best_row_1se(results, measure)`, `select_best_row_quantile(results, measure)`, `select_best_row_by_rule(results, measure, rule, not_two = FALSE)`: each returns a one-row data frame taken from `results`, with all its columns.
  - `select_best_k(results, measure, rule, not_two = FALSE)` returns an integer.
  - `row_to_estimator_params(row, valid_keys)` returns a named list of the row's non-missing values for `valid_keys`.

- [ ] Step 1: Write the failing tests in `tests/testthat/test-selection.R`

```r
test_that("selection matches Python for every measure, rule and not_two", {
  fx <- read_fixture("selection")
  table <- as.data.frame(fx$table)
  for (case in fx$expected) {
    info <- paste(case$measure, case$rule, case$not_two)
    row <- NULL
    if (is.null(case$warning)) {
      expect_no_warning(
        row <- select_best_row_by_rule(table, case$measure, case$rule, not_two = case$not_two)
      )
    } else {
      expect_warning(
        row <- select_best_row_by_rule(table, case$measure, case$rule, not_two = case$not_two),
        case$warning,
        fixed = TRUE
      )
    }
    expect_identical(row$config_id, as.integer(case$config_id), info = info)
    k <- suppressWarnings(select_best_k(table, case$measure, case$rule, not_two = case$not_two))
    expect_identical(k, as.integer(case$k), info = info)
  }
})

quantile_table <- data.frame(
  estimator = "KMeans",
  n_clusters = 2:3,
  sweep_rank = 0:1,
  n_clusters_observed = c(2, 3),
  ari_stability = c(0.5, 0.3),
  ari_stability_se = 0.01,
  ari_stability_upper = c(0.4, 0.2),
  ari_stability_lower = c(0.6, 0.4)
)

test_that("quantile falls back to max when no row lies within the bounds", {
  row <- NULL
  expect_warning(
    row <- select_best_row_quantile(quantile_table, "stability"),
    "No estimators within quantile thresholds; falling back to max.",
    fixed = TRUE
  )
  expect_identical(row$n_clusters, 2L)
})

test_that("1se picks the finest rank within one standard error", {
  df <- data.frame(
    n_clusters = 2:5, sweep_rank = 0:3, n_clusters_observed = 2:5,
    ari_stability = c(0.90, 0.89, 0.88, 0.50),
    ari_stability_se = c(0.05, 0.04, 0.03, 0.02)
  )
  expect_identical(select_best_row_1se(df, "stability")$n_clusters, 4L)
})

test_that("1se with a NaN standard error returns the best row", {
  df <- data.frame(
    n_clusters = 2:3, sweep_rank = 0:1, n_clusters_observed = 2:3,
    ari_stability = c(0.9, 0.8), ari_stability_se = NaN
  )
  expect_identical(select_best_row_1se(df, "stability")$n_clusters, 2L)
})

test_that("selection rejects unknown measures and rules", {
  expect_error(
    select_best_row_by_rule(quantile_table, "silhouette", "max"),
    "Invalid measure 'silhouette'. Options: ['s', 'stab', ",
    fixed = TRUE
  )
  expect_error(select_best_row_by_rule(quantile_table, "stability", "median"), "Unknown rule: median", fixed = TRUE)
})

test_that("not_two removes two-cluster rows and fails when none remain", {
  expect_identical(
    select_best_row_by_rule(quantile_table, "stability", "max", not_two = TRUE)$n_clusters,
    3L
  )
  expect_error(
    select_best_row_by_rule(quantile_table[1, ], "stability", "max", not_two = TRUE),
    "No configurations remain after excluding k=2.",
    fixed = TRUE
  )
})

test_that("a measure without any value is an error", {
  df <- data.frame(n_clusters = 2:3, sweep_rank = 0:1, n_clusters_observed = 2:3, ari_stability = NaN)
  expect_error(
    select_best_row_max(df, "stability"),
    "No configuration has a value for ari_stability.",
    fixed = TRUE
  )
})

test_that("every measure alias resolves to a results column", {
  expect_setequal(
    unique(unname(MEASURE_MAP)),
    c("ari_stability", "ari_generalizability", "ari_average", "consensus_pac_stability",
      "consensus_gini_stability", "consensus_ce_stability", "accuracy_generalizability")
  )
  for (alias in names(MEASURE_MAP)) {
    expect_identical(measure_column(alias), MEASURE_MAP[[alias]])
  }
})

test_that("row_to_estimator_params keeps the parameters that have values", {
  row <- data.frame(n_clusters = 3L, linkage = NA, gamma = NaN, method_id = "m0")
  expect_identical(
    row_to_estimator_params(row, c("n_clusters", "linkage", "gamma", "tol")),
    list(n_clusters = 3L)
  )
})
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "selection", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "select_best_row_by_rule"`.

- [ ] Step 3: Write `R/selection.R`

```r
# Choosing a configuration from the results table. Mirrors _selection.py.
# The rules order configurations by sweep_rank, so "finest" means the same
# thing on every sweep axis.

MEASURE_MAP <- c(
  s = "ari_stability",
  stab = "ari_stability",
  stability = "ari_stability",
  ari_stability = "ari_stability",
  g = "ari_generalizability",
  gen = "ari_generalizability",
  generalizability = "ari_generalizability",
  ari_generalizability = "ari_generalizability",
  avg = "ari_average",
  average = "ari_average",
  ari_average = "ari_average",
  pac = "consensus_pac_stability",
  consensus_pac_stability = "consensus_pac_stability",
  gini = "consensus_gini_stability",
  consensus_gini_stability = "consensus_gini_stability",
  ce = "consensus_ce_stability",
  consensus_ce_stability = "consensus_ce_stability",
  acc = "accuracy_generalizability",
  accuracy = "accuracy_generalizability",
  accuracy_generalizability = "accuracy_generalizability"
)

measure_column <- function(measure) {
  if (!is.character(measure) || length(measure) != 1L || !measure %in% names(MEASURE_MAP)) {
    stop(sprintf(
      "Invalid measure %s. Options: %s",
      format_repr(measure),
      python_list(names(MEASURE_MAP))
    ), call. = FALSE)
  }
  MEASURE_MAP[[measure]]
}

# The first row with the highest value, as pandas' idxmax picks it.
best_row <- function(results, column) {
  values <- results[[column]]
  if (all(is.na(values))) {
    stop(sprintf("No configuration has a value for %s.", column), call. = FALSE)
  }
  results[which.max(values), , drop = FALSE]
}

finest_row <- function(rows) {
  rows[which.max(rows$sweep_rank), , drop = FALSE]
}

select_best_row_max <- function(results, measure = "stability") {
  best_row(results, measure_column(measure))
}

select_best_row_1se <- function(results, measure = "stability") {
  column <- measure_column(measure)
  best <- best_row(results, column)
  threshold <- best[[column]] - best[[paste0(column, "_se")]]
  within <- results[which(results[[column]] >= threshold), , drop = FALSE]
  # A NaN standard error (one resample) leaves no row within one SE; the
  # best row is the answer then.
  if (nrow(within) == 0L) {
    return(best)
  }
  finest_row(within)
}

select_best_row_quantile <- function(results, measure = "stability") {
  column <- measure_column(measure)
  best <- best_row(results, column)
  upper <- best[[paste0(column, "_upper")]]
  lower <- best[[paste0(column, "_lower")]]
  values <- results[[column]]
  within <- results[which(values >= lower & values <= upper), , drop = FALSE]
  if (nrow(within) == 0L) {
    warning("No estimators within quantile thresholds; falling back to max.", call. = FALSE)
    return(best)
  }
  finest_row(within)
}

select_best_row_by_rule <- function(results, measure, rule, not_two = FALSE) {
  if (not_two) {
    results <- results[!(observed_k_series(results) %in% 2), , drop = FALSE]
    if (nrow(results) == 0L) {
      stop("No configurations remain after excluding k=2.", call. = FALSE)
    }
  }
  column <- measure_column(measure)
  if (identical(rule, "1se") && !paste0(column, "_se") %in% names(results)) {
    warning(sprintf("Column '%s_se' not found; falling back to 'max' rule.", column), call. = FALSE)
    rule <- "max"
  } else if (identical(rule, "quantile") && !paste0(column, "_upper") %in% names(results)) {
    warning(sprintf("Column '%s_upper' not found; falling back to 'max' rule.", column), call. = FALSE)
    rule <- "max"
  }
  switch(rule,
    max = select_best_row_max(results, measure),
    `1se` = select_best_row_1se(results, measure),
    quantile = select_best_row_quantile(results, measure),
    stop(sprintf("Unknown rule: %s", rule), call. = FALSE)
  )
}

# The number of clusters at the selected row: the requested k in k mode,
# the rounded mean observed count on other axes.
select_best_k <- function(results, measure, rule, not_two = FALSE) {
  observed_k(select_best_row_by_rule(results, measure, rule, not_two = not_two))
}

row_to_estimator_params <- function(row, valid_keys) {
  keys <- intersect(valid_keys, names(row))
  params <- lapply(keys, function(key) row[[key]][[1L]])
  names(params) <- keys
  missing <- vapply(params, function(v) length(v) == 1L && is.atomic(v) && is.na(v), logical(1))
  params[!missing]
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "selection", stop_on_failure = TRUE)'`
Expected: PASS.

- [ ] Step 5: Mutation check

- In `finest_row()`, use `which.min`: the fixture test and "1se picks the finest rank" fail.
- In `select_best_row_by_rule()`, drop the `_se` fallback (go straight to `select_best_row_1se`): the fixture test fails on `pac` with `1se`.
- In `select_best_row_quantile()`, swap `lower` and `upper`: the fixture test fails.

- [ ] Step 6: Commit

```bash
git add R/selection.R tests/testthat/test-selection.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add the max, 1se and quantile selection rules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Consensus matrices and held-out accuracy

Files:
- Create: `carve-r/R/consensus.R`, `carve-r/R/accuracy.R`
- Test: `carve-r/tests/testthat/test-consensus.R`, `carve-r/tests/testthat/test-accuracy.R`

Interfaces:
- Consumes: `align_cluster_labels` (Task 4); fixtures `consensus`, `accuracy`.
- Produces (all internal):
  - `compute_consensus_matrix(n_samples, runs)` where each run is `list(indices = <1-based int>, labels = <int>)`; returns an `n_samples` by `n_samples` double matrix, `NaN` for pairs never drawn together.
  - `stability_from_consensus(consensus_matrix)` returns `list(gini, ce)`, per-sample scores on [0, 1], `NaN` for a sample never drawn with any partner.
  - `compute_consensus_pac(consensus_matrix, tau = 0.05)` returns 1 minus the share of ambiguous off-diagonal values, `NaN` when there are none.
  - `compute_generalizability_scores(n_samples, runs)` where each run is `list(indices, true, predicted)`; returns per-sample accuracy, 0 for a sample never held out.

- [ ] Step 1: Write the failing tests

`tests/testthat/test-consensus.R`:

```r
fixture_runs <- function(runs) {
  lapply(runs, function(r) list(indices = as.integer(r$indices) + 1L, labels = as.integer(r$labels)))
}

test_that("compute_consensus_matrix matches Python", {
  fx <- read_fixture("consensus")
  M <- compute_consensus_matrix(fx$n_samples, fixture_runs(fx$runs))
  expect_equal(nan_to_na(M), fixture_matrix(fx$consensus), tolerance = 1e-6)
})

test_that("stability scores and PAC match Python", {
  fx <- read_fixture("consensus")
  M <- compute_consensus_matrix(fx$n_samples, fixture_runs(fx$runs))
  s <- stability_from_consensus(M)
  expect_equal(nan_to_na(s$gini), fixture_vector(fx$gini), tolerance = 1e-6)
  expect_equal(nan_to_na(s$ce), fixture_vector(fx$ce), tolerance = 1e-6)
  expect_equal(compute_consensus_pac(M), fx$pac_005, tolerance = 1e-6)
  expect_equal(compute_consensus_pac(M, tau = 0.1), fx$pac_010, tolerance = 1e-6)
})

test_that("pairs never drawn together are NaN and so is a sample never drawn", {
  runs <- list(
    list(indices = 1:2, labels = c(1L, 1L)),
    list(indices = 2:3, labels = c(1L, 2L))
  )
  M <- compute_consensus_matrix(4L, runs)
  expect_true(is.nan(M[1, 3]))
  expect_true(all(is.nan(M[4, ])))
  expect_identical(M[1, 2], 1)
  expect_identical(M[2, 3], 0)
  expect_identical(diag(M)[1:3], c(1, 1, 1))
  expect_true(is.na(stability_from_consensus(M)$gini[4]))
})

test_that("perfect consensus scores 1 everywhere", {
  M <- matrix(c(1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1), 4)
  s <- stability_from_consensus(M)
  expect_equal(s$gini, rep(1, 4))
  expect_equal(s$ce, rep(1, 4), tolerance = 1e-9)
  expect_identical(compute_consensus_pac(M), 1)
})

test_that("PAC is NaN when no pair was drawn together", {
  expect_true(is.nan(compute_consensus_pac(matrix(NaN, 3, 3))))
})
```

`tests/testthat/test-accuracy.R`:

```r
test_that("compute_generalizability_scores matches Python", {
  fx <- read_fixture("accuracy")
  runs <- lapply(fx$runs, function(r) {
    list(indices = as.integer(r$indices) + 1L, true = as.integer(r$true), predicted = as.integer(r$predicted))
  })
  expect_equal(
    compute_generalizability_scores(fx$n_samples, runs),
    fixture_vector(fx$scores),
    tolerance = 1e-12
  )
})

test_that("predictions are aligned before scoring and unseen samples score 0", {
  runs <- list(list(indices = 1:2, true = c(1L, 2L), predicted = c(2L, 1L)))
  expect_identical(compute_generalizability_scores(3L, runs), c(1, 1, 0))
})
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "consensus|accuracy", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "compute_consensus_matrix"`.

- [ ] Step 3: Write `R/consensus.R`

```r
# Consensus matrices and the stability scores derived from them. Mirrors
# the exact path of _consensus.py; the anchored path comes in stage 2.

# Entry (i, j) is the share of the runs that drew both i and j in which they
# landed in the same cluster; NaN when no run drew both. S marks which runs
# drew each sample and B which run-cluster each sample fell in, so the two
# counts are tcrossprod(S) and tcrossprod(B).
compute_consensus_matrix <- function(n_samples, runs) {
  n_labels <- vapply(runs, function(r) length(unique(r$labels)), integer(1))
  S <- matrix(0, n_samples, length(runs))
  B <- matrix(0, n_samples, sum(n_labels))
  col <- 0L
  for (r in seq_along(runs)) {
    indices <- runs[[r]]$indices
    labels <- runs[[r]]$labels
    S[indices, r] <- 1
    for (label in unique(labels)) {
      col <- col + 1L
      B[indices[labels == label], col] <- 1
    }
  }
  co_sampled <- tcrossprod(S)
  co_clustered <- tcrossprod(B)
  consensus <- co_clustered / co_sampled
  consensus[co_sampled == 0] <- NaN
  consensus
}

# Per-sample stability from the off-diagonal consensus values of each row.
# Gini uncertainty p(1 - p) and binary entropy are averaged over the row
# and rescaled so that 1 means every pair always agreed.
stability_from_consensus <- function(consensus_matrix) {
  p <- consensus_matrix
  diag(p) <- NaN
  gini_term <- p * (1 - p)
  clipped <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  entropy <- -(clipped * log(clipped) + (1 - clipped) * log(1 - clipped))
  uncertainty_gini <- 2 * rowMeans(gini_term, na.rm = TRUE)
  uncertainty_ce <- rowMeans(entropy, na.rm = TRUE)
  list(
    gini = 1 - pmin(pmax(2 * uncertainty_gini, 0), 1),
    ce = 1 - pmin(pmax(uncertainty_ce / log(2), 0), 1)
  )
}

# 1 minus the proportion of ambiguous consensus values, those strictly
# between tau and 1 - tau, over the off-diagonal pairs drawn together.
compute_consensus_pac <- function(consensus_matrix, tau = 0.05) {
  off_diagonal <- row(consensus_matrix) != col(consensus_matrix)
  values <- consensus_matrix[off_diagonal]
  values <- values[!is.na(values)]
  if (length(values) == 0L) {
    return(NaN)
  }
  1 - mean(values > tau & values < 1 - tau)
}
```

- [ ] Step 4: Write `R/accuracy.R`

```r
# Per-sample held-out accuracy. Mirrors _accuracy.py.

# For every held-out sample, the share of the runs that held it out in which
# the classifier, after its labels were aligned to the held-out clustering,
# predicted the sample's cluster. A sample never held out scores 0.
compute_generalizability_scores <- function(n_samples, runs) {
  correct <- numeric(n_samples)
  total <- numeric(n_samples)
  for (run in runs) {
    aligned <- align_cluster_labels(run$true, run$predicted)
    correct[run$indices] <- correct[run$indices] + (run$true == aligned)
    total[run$indices] <- total[run$indices] + 1
  }
  out <- numeric(n_samples)
  held_out <- total > 0
  out[held_out] <- correct[held_out] / total[held_out]
  out
}
```

- [ ] Step 5: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "consensus|accuracy", stop_on_failure = TRUE)'`
Expected: PASS.

- [ ] Step 6: Mutation check

- In `stability_from_consensus()`, drop `diag(p) <- NaN`: the fixture test fails.
- In `compute_consensus_pac()`, return `mean(...)` without the `1 -`: "stability scores and PAC match Python" and "perfect consensus scores 1 everywhere" fail.
- In `compute_generalizability_scores()`, skip the alignment (compare `run$true == run$predicted`): "predictions are aligned before scoring" fails.

- [ ] Step 7: Commit

```bash
git add R/consensus.R R/accuracy.R tests/testthat/test-consensus.R tests/testthat/test-accuracy.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add consensus matrices, stability scores and held-out accuracy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: KMeans and AgglomerativeClustering

Files:
- Create: `carve-r/R/estimators.R`
- Test: `carve-r/tests/testthat/test-estimators.R`

Interfaces:
- Consumes: `seeded`, `as_data_matrix`, `adjusted_rand_index`, `format_repr` (Tasks 3-4); fixtures `kmeans`, `agglomerative`.
- Produces:
  - Exported `KMeans(X, n_clusters = 8L, n_init = 1L, max_iter = 300L, tol = 1e-4, random_state = NULL)` and `AgglomerativeClustering(X, n_clusters = 2L, linkage = "ward")`, each returning integer labels 1 to `n_clusters`.
  - Internal `sq_dists(X, Y)` (squared Euclidean distances, rows of `X` by rows of `Y`), `nearest_center(X, centers)`, `kmeans_plusplus(X, n_clusters)` (draws from the current RNG stream; call it inside `seeded()`), `kmeans_lloyd(X, centers, max_iter, tol)` returning `list(labels, centers, inertia, n_iter)` with `tol` already absolute.

- [ ] Step 1: Write the failing tests in `tests/testthat/test-estimators.R`

```r
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
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "estimators", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "kmeans_lloyd"`.

- [ ] Step 3: Write `R/estimators.R`

```r
# Clustering estimators. KMeans and AgglomerativeClustering stand in for the
# scikit-learn classes of the same names; SpectralClustering ports
# cluster.py. Each takes the data matrix first and returns integer labels
# from 1 to n_clusters.

sq_dists <- function(X, Y) {
  D2 <- outer(rowSums(X^2), rowSums(Y^2), "+") - 2 * tcrossprod(X, Y)
  D2[D2 < 0] <- 0
  D2
}

nearest_center <- function(X, centers) {
  max.col(-sq_dists(X, centers), ties.method = "first")
}

# Greedy k-means++, as sklearn's _kmeans_plusplus: the first center is a
# uniform draw; each later one is the best of 2 + floor(log(k)) candidates
# drawn with probability proportional to the squared distance to the
# nearest center so far.
kmeans_plusplus <- function(X, n_clusters) {
  n <- nrow(X)
  n_local_trials <- 2L + as.integer(floor(log(n_clusters)))
  centers <- matrix(0, n_clusters, ncol(X))
  first <- sample.int(n, 1L)
  centers[1L, ] <- X[first, ]
  closest <- sq_dists(X, X[first, , drop = FALSE])[, 1L]
  potential <- sum(closest)
  for (c in seq_len(n_clusters)[-1L]) {
    draws <- stats::runif(n_local_trials) * potential
    candidates <- pmin(findInterval(draws, cumsum(closest), left.open = TRUE) + 1L, n)
    distances <- pmin(sq_dists(X, X[candidates, , drop = FALSE]), closest)
    potentials <- colSums(distances)
    best <- which.min(potentials)
    potential <- potentials[[best]]
    closest <- distances[, best]
    centers[c, ] <- X[candidates[best], ]
  }
  centers
}

# Lloyd's algorithm as sklearn's _kmeans_single_lloyd runs it: stop when the
# labels do not change, or when the summed squared center shift is at most
# tol. A cluster left empty takes the point farthest from its own center. X
# and centers are already centered on the column means.
kmeans_lloyd <- function(X, centers, max_iter, tol) {
  n <- nrow(X)
  k <- nrow(centers)
  labels_old <- rep(-1L, n)
  strict <- FALSE
  for (iteration in seq_len(max_iter)) {
    labels <- nearest_center(X, centers)
    sums <- matrix(0, k, ncol(X))
    present <- sort(unique(labels))
    sums[present, ] <- rowsum(X, labels, reorder = TRUE)
    counts <- tabulate(labels, nbins = k)
    empty <- which(counts == 0L)
    if (length(empty) > 0L) {
      own_distance <- rowSums((X - centers[labels, , drop = FALSE])^2)
      if (max(own_distance) > 0) {
        far <- order(own_distance, decreasing = TRUE)[seq_along(empty)]
        for (e in seq_along(empty)) {
          j <- empty[e]
          p <- far[e]
          old <- labels[p]
          sums[old, ] <- sums[old, ] - X[p, ]
          sums[j, ] <- X[p, ]
          counts[old] <- counts[old] - 1L
          counts[j] <- 1L
        }
      }
    }
    new_centers <- sums
    filled <- counts > 0L
    new_centers[filled, ] <- sums[filled, , drop = FALSE] / counts[filled]
    shift <- sum((new_centers - centers)^2)
    centers <- new_centers
    if (identical(labels, labels_old)) {
      strict <- TRUE
      break
    }
    if (shift <= tol) {
      break
    }
    labels_old <- labels
  }
  if (!strict) {
    labels <- nearest_center(X, centers)
  }
  list(
    labels = labels,
    centers = centers,
    inertia = sum((X - centers[labels, , drop = FALSE])^2),
    n_iter = iteration
  )
}

#' K-means clustering
#'
#' Lloyd's algorithm started from greedy k-means++ seeds, as in
#' scikit-learn's `KMeans` with its default settings. The seeding and the
#' iterations follow scikit-learn's implementation; R draws different random
#' numbers, so the labels are not the ones Python would return.
#'
#' @param X Numeric matrix or data frame, one row per sample.
#' @param n_clusters Number of clusters.
#' @param n_init Number of k-means++ starts. The start with the lowest
#'   within-cluster sum of squares is kept. scikit-learn's default, `"auto"`,
#'   is one start.
#' @param max_iter Maximum number of iterations per start.
#' @param tol A start stops when the summed squared shift of the centers is at
#'   most `tol` times the mean column variance.
#' @param random_state Seed, or `NULL` to draw from the session's random
#'   number stream.
#' @return Integer labels from 1 to `n_clusters`, one per row of `X`.
#' @seealso [estimator_grid()] to use it in [carve()].
#' @examples
#' X <- rbind(matrix(rnorm(40, 0, 0.3), ncol = 2), matrix(rnorm(40, 3, 0.3), ncol = 2))
#' KMeans(X, n_clusters = 2, random_state = 1)
#' @export
KMeans <- function(X, n_clusters = 8L, n_init = 1L, max_iter = 300L, tol = 1e-4,
                   random_state = NULL) {
  X <- as_data_matrix(X)
  n_clusters <- as.integer(n_clusters)
  if (nrow(X) < n_clusters) {
    stop(sprintf("n_samples=%d should be >= n_clusters=%d.", nrow(X), n_clusters), call. = FALSE)
  }
  Xc <- sweep(X, 2L, colMeans(X))
  tol_abs <- if (tol == 0) 0 else mean(colMeans(Xc^2)) * tol
  best <- NULL
  seeded(random_state, {
    for (start in seq_len(n_init)) {
      fit <- kmeans_lloyd(Xc, kmeans_plusplus(Xc, n_clusters), max_iter = max_iter, tol = tol_abs)
      if (is.null(best) || fit$inertia < best$inertia) {
        best <- fit
      }
    }
  })
  best$labels
}

#' Agglomerative clustering
#'
#' Hierarchical clustering of the Euclidean distances with [stats::hclust()],
#' cut into `n_clusters` groups. `linkage = "ward"` runs `"ward.D2"`, the Ward
#' criterion scikit-learn's `AgglomerativeClustering` uses.
#'
#' @param X Numeric matrix or data frame, one row per sample.
#' @param n_clusters Number of clusters.
#' @param linkage `"ward"`, `"average"`, `"single"` or `"complete"`.
#' @return Integer labels from 1 to `n_clusters`, one per row of `X`.
#' @seealso [estimator_grid()] to use it in [carve()].
#' @examples
#' X <- rbind(matrix(rnorm(40, 0, 0.3), ncol = 2), matrix(rnorm(40, 3, 0.3), ncol = 2))
#' AgglomerativeClustering(X, n_clusters = 2)
#' @export
AgglomerativeClustering <- function(X, n_clusters = 2L, linkage = "ward") {
  X <- as_data_matrix(X)
  method <- switch(linkage,
    ward = "ward.D2",
    average = "average",
    single = "single",
    complete = "complete",
    stop(sprintf(
      "Unknown linkage: %s. Expected 'ward', 'average', 'single' or 'complete'.",
      format_repr(linkage)
    ), call. = FALSE)
  )
  tree <- stats::hclust(stats::dist(X), method = method)
  as.integer(stats::cutree(tree, k = n_clusters))
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::document(); devtools::test(filter = "estimators", stop_on_failure = TRUE)'`
Expected: `man/KMeans.Rd` and `man/AgglomerativeClustering.Rd` written; tests PASS.

- [ ] Step 5: Mutation check

- In `kmeans_lloyd()`, remove the empty-cluster relocation block: the `empty_cluster` fixture case fails.
- In `kmeans_lloyd()`, drop the `identical(labels, labels_old)` check: the `n_iter` assertion fails for at least one case.
- In `AgglomerativeClustering()`, map `ward` to `"ward.D"`: the fixture test fails for `ward`, or passes by chance; report which.

- [ ] Step 6: Commit

```bash
git add R/estimators.R tests/testthat/test-estimators.R NAMESPACE man/KMeans.Rd man/AgglomerativeClustering.Rd
git commit -m "$(cat <<'EOF'
feat(carve-r): add KMeans and AgglomerativeClustering

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: SpectralClustering

Files:
- Modify: `carve-r/R/estimators.R` (append)
- Test: `carve-r/tests/testthat/test-estimators.R` (append)

Interfaces:
- Consumes: `KMeans`, `sq_dists` (Task 9); `as_data_matrix`, `format_repr` (Task 3); the `spectral` fixture; `make_moons` (Task 1).
- Produces:
  - Exported `SpectralClustering(X, n_clusters = 2L, affinity = "self_tuning", gamma = NULL, n_neighbors = 7L, n_init = 1L, scale = TRUE, random_state = NULL)`.
  - Internal `standard_scale(X)`, `nearest_neighbors(X, n_neighbors)` (the `FNN::get.knn` result for the `n_neighbors - 1` nearest other samples), `kth_distance(X, n_neighbors)`, `knn_sigma(kth)`, `squared_distances(X)`, `pmax_sparse(A, B)`, `spectral_affinity(X, affinity = "self_tuning", gamma = NULL, n_neighbors = 7L)` returning `list(W, gamma)`, `spectral_embedding(W, k)` returning `list(values, vectors)` with values ascending, `dense_eigs(L, k)`, `sparse_eigs(L, k)`.

- [ ] Step 1: Append the failing tests to `tests/testthat/test-estimators.R`

```r
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
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "estimators", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "standard_scale"`.

- [ ] Step 3: Append to `R/estimators.R`

```r
# StandardScaler: centered columns divided by the population standard
# deviation; a constant column is left at zero.
standard_scale <- function(X) {
  X <- sweep(X, 2L, colMeans(X))
  sds <- sqrt(colMeans(X^2))
  sds[sds == 0] <- 1
  sweep(X, 2L, sds, "/")
}

# sklearn's NearestNeighbors(n_neighbors = k) counts each sample as its own
# first neighbor, so the Python package's k-th neighbor is the (k - 1)-th
# nearest other sample, which is what get.knn returns.
nearest_neighbors <- function(X, n_neighbors) {
  n_neighbors <- as.integer(n_neighbors)
  if (n_neighbors < 2L) {
    stop("n_neighbors must be at least 2.", call. = FALSE)
  }
  if (n_neighbors > nrow(X)) {
    stop(sprintf("n_neighbors=%d is larger than the %d samples.", n_neighbors, nrow(X)), call. = FALSE)
  }
  FNN::get.knn(X, k = n_neighbors - 1L)
}

kth_distance <- function(X, n_neighbors) {
  nearest_neighbors(X, n_neighbors)$nn.dist[, as.integer(n_neighbors) - 1L]
}

knn_sigma <- function(kth) {
  sigma <- stats::median(kth)
  if (sigma == 0) {
    sigma <- if (any(kth > 0)) mean(kth[kth > 0]) else 1
  }
  sigma
}

squared_distances <- function(X) {
  D2 <- sq_dists(X, X)
  diag(D2) <- 0
  D2
}

# Elementwise maximum of two non-negative sparse matrices, keeping sparsity.
pmax_sparse <- function(A, B) {
  (A + B + abs(A - B)) / 2
}

spectral_affinity <- function(X, affinity = "self_tuning", gamma = NULL, n_neighbors = 7L) {
  n <- nrow(X)
  if (identical(affinity, "rbf")) {
    if (is.null(gamma)) {
      gamma <- 1 / (2 * knn_sigma(kth_distance(X, n_neighbors))^2)
    }
    W <- exp(-gamma * squared_distances(X))
    diag(W) <- 0
    return(list(W = W, gamma = gamma))
  }
  if (identical(affinity, "knn")) {
    nn <- nearest_neighbors(X, n_neighbors)
    k <- ncol(nn$nn.index)
    if (is.null(gamma)) {
      gamma <- 1 / (2 * knn_sigma(nn$nn.dist[, k])^2)
    }
    W <- Matrix::sparseMatrix(
      i = rep(seq_len(n), times = k),
      j = as.vector(nn$nn.index),
      x = exp(-gamma * as.vector(nn$nn.dist)^2),
      dims = c(n, n)
    )
    return(list(W = 0.5 * (W + Matrix::t(W)), gamma = gamma))
  }
  if (identical(affinity, "self_tuning")) {
    # Zelnik-Manor and Perona (2004): each sample's scale is its distance to
    # the k-th neighbor, and W(i, j) = exp(-d(i, j)^2 / (sigma_i sigma_j)).
    nn <- nearest_neighbors(X, n_neighbors)
    k <- ncol(nn$nn.index)
    sigma <- nn$nn.dist[, k]
    if (any(sigma == 0)) {
      sigma[sigma == 0] <- if (any(sigma > 0)) stats::median(sigma[sigma > 0]) else 1
    }
    if (n > 5000L) {
      i <- rep(seq_len(n), times = k)
      j <- as.vector(nn$nn.index)
      W <- Matrix::sparseMatrix(
        i = i,
        j = j,
        x = exp(-as.vector(nn$nn.dist)^2 / (sigma[i] * sigma[j])),
        dims = c(n, n)
      )
      return(list(W = pmax_sparse(W, Matrix::t(W)), gamma = NULL))
    }
    W <- exp(-squared_distances(X) / outer(sigma, sigma))
    diag(W) <- 0
    return(list(W = W, gamma = NULL))
  }
  stop(sprintf("Unknown affinity: %s", format_repr(affinity)), call. = FALSE)
}

dense_eigs <- function(L, k) {
  e <- eigen(L, symmetric = TRUE)
  idx <- seq.int(nrow(L) - k + 1L, nrow(L))
  list(values = e$values[idx], vectors = e$vectors[, idx, drop = FALSE])
}

# Shift-invert around 0. Python falls back to a dense solve when ARPACK
# does not converge or the factorization is singular. RSpectra reports
# non-convergence through a warning and fewer converged values, and a
# singular dense factorization can come back as non-finite values without
# an error; each of these stops here and sends spectral_embedding() to the
# dense fallback.
sparse_eigs <- function(L, k) {
  result <- withCallingHandlers(
    RSpectra::eigs_sym(L, k = k, which = "LM", sigma = 0, opts = list(tol = 1e-4, maxitr = 5000L)),
    warning = function(w) invokeRestart("muffleWarning")
  )
  if (is.null(result$nconv) || result$nconv < k ||
      any(!is.finite(result$values)) || any(!is.finite(result$vectors))) {
    stop("The sparse eigensolver did not converge.", call. = FALSE)
  }
  list(values = result$values, vectors = result$vectors)
}

# Eigenvectors of the normalized Laplacian L = I - D^-1/2 W D^-1/2 for the k
# smallest eigenvalues, rows scaled to unit length (Ng, Jordan and Weiss).
spectral_embedding <- function(W, k) {
  n <- nrow(W)
  degree <- if (inherits(W, "Matrix")) Matrix::rowSums(W) else rowSums(W)
  dinv <- 1 / sqrt(pmax(as.numeric(degree), 1e-12))
  if (n < 1000L) {
    Wd <- as.matrix(W)
    L <- diag(n) - dinv * Wd * rep(dinv, each = n)
    L <- 0.5 * (L + t(L))
    eig <- dense_eigs(L, k)
  } else {
    L <- if (inherits(W, "Matrix")) {
      Dinv <- Matrix::Diagonal(x = dinv)
      Matrix::Diagonal(n) - Dinv %*% W %*% Dinv
    } else {
      diag(n) - dinv * W * rep(dinv, each = n)
    }
    eig <- tryCatch(sparse_eigs(L, k), error = function(e) dense_eigs(as.matrix(L), k))
  }
  ord <- order(eig$values)
  vectors <- eig$vectors[, ord, drop = FALSE]
  norms <- sqrt(rowSums(vectors^2))
  list(values = eig$values[ord], vectors = vectors / (norms + 1e-12))
}

#' Spectral clustering
#'
#' Builds an affinity matrix, embeds the samples with the eigenvectors of the
#' normalized graph Laplacian that belong to its `n_clusters` smallest
#' eigenvalues, scales each row to unit length (Ng, Jordan and Weiss), and
#' clusters the embedding with [KMeans()]. It ports the Python package's
#' `SpectralClustering`.
#'
#' `affinity = "self_tuning"` weighs each pair by the two samples' local
#' scales, the distance to their `(n_neighbors - 1)`-th nearest other sample
#' (Zelnik-Manor and Perona, 2004). Above 5,000 samples it keeps only the
#' nearest-neighbor edges. `"rbf"` uses one global `gamma`, and `"knn"`
#' keeps the nearest-neighbor edges with RBF weights. Without a `gamma`, both
#' set it to `1 / (2 * sigma^2)`, with sigma the median distance to the
#' `(n_neighbors - 1)`-th nearest other sample.
#'
#' Below 1,000 samples the eigenvectors come from a dense decomposition. From
#' 1,000 samples on, [RSpectra::eigs_sym()] finds them by shift-invert around
#' zero, and the dense decomposition takes over if that fails.
#'
#' @param X Numeric matrix or data frame, one row per sample.
#' @param n_clusters Number of clusters.
#' @param affinity `"self_tuning"`, `"rbf"` or `"knn"`.
#' @param gamma RBF scale for `"rbf"` and `"knn"`, or `NULL` to set it from the
#'   nearest-neighbor distances. Not used by `"self_tuning"`.
#' @param n_neighbors Neighborhood size for the local scales and the kNN graph,
#'   at least 2. In a [carve()] run it is scaled to each subsample's size.
#' @param n_init Starts of the final k-means.
#' @param scale Standardize each column before computing affinities.
#' @param random_state Seed for the final k-means, or `NULL`.
#' @return Integer labels from 1 to `n_clusters`, one per row of `X`.
#' @seealso [estimator_grid()] to use it in [carve()].
#' @examples
#' t <- seq(0, pi, length.out = 50)
#' X <- rbind(cbind(cos(t), sin(t)), cbind(1 - cos(t), 0.5 - sin(t)))
#' SpectralClustering(X, n_clusters = 2, random_state = 1)
#' @export
SpectralClustering <- function(X, n_clusters = 2L, affinity = "self_tuning", gamma = NULL,
                               n_neighbors = 7L, n_init = 1L, scale = TRUE,
                               random_state = NULL) {
  X <- as_data_matrix(X)
  Xp <- if (isTRUE(scale)) standard_scale(X) else X
  affinity_matrix <- spectral_affinity(Xp, affinity = affinity, gamma = gamma, n_neighbors = n_neighbors)
  embedding <- spectral_embedding(affinity_matrix$W, as.integer(n_clusters))
  KMeans(embedding$vectors, n_clusters = n_clusters, n_init = n_init, random_state = random_state)
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::document(); devtools::test(filter = "estimators", stop_on_failure = TRUE)'`
Expected: `man/SpectralClustering.Rd` written; tests PASS. The two 1,200-sample tests take a few seconds each.

- [ ] Step 5: Mutation check

- In `nearest_neighbors()`, call `FNN::get.knn(X, k = n_neighbors)`: the affinity fixture test fails for `self_tuning`, `rbf` and `knn`.
- In `standard_scale()`, use `apply(X, 2, sd)` (the sample standard deviation): "standard_scale matches StandardScaler" fails.
- In `spectral_embedding()`, change the threshold to `n < 2000L`: "a failing sparse solve falls back to the dense one" fails on `calls`.

- [ ] Step 6: Commit

```bash
git add R/estimators.R tests/testthat/test-estimators.R NAMESPACE man/SpectralClustering.Rd
git commit -m "$(cat <<'EOF'
feat(carve-r): add SpectralClustering with the self-tuning, RBF and kNN affinities

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Default estimator grids

Files:
- Create: `carve-r/R/grids.R`
- Test: `carve-r/tests/testthat/test-grids.R`

Interfaces:
- Consumes: `estimator_grid` (Task 5), `resolve_sweep` (Task 6), `KMeans`, `AgglomerativeClustering`, `SpectralClustering`, `kth_distance` (Tasks 9-10), `format_repr` (Task 3); the `spectral` fixture.
- Produces (internal): `estimate_knn_gamma(X, n_neighbors = 7L, multipliers = c(0.5, 1, 2))` returning three doubles; `default_estimator_grids(X, preset, sweep)` returning a list of `carve_estimator_grid` objects; `default_k_grids(X, n_clusters, preset)`.

- [ ] Step 1: Write the failing tests in `tests/testthat/test-grids.R`

```r
test_that("estimate_knn_gamma matches Python", {
  fx <- read_fixture("spectral")
  expect_equal(estimate_knn_gamma(fixture_matrix(fx$X)), fixture_vector(fx$knn_gamma), tolerance = 1e-10)
})

test_that("the light preset holds KMeans, Ward and self-tuning spectral clustering", {
  X <- make_blobs()$X
  grids <- default_estimator_grids(X, "light", resolve_sweep(n_clusters = 4))
  expect_identical(
    vapply(grids, function(g) g$name, character(1)),
    c("KMeans", "AgglomerativeClustering", "SpectralClustering")
  )
  expect_identical(grids[[1]]$estimator, KMeans)
  expect_identical(grids[[1]]$grid, list(n_clusters = 2:4))
  expect_identical(grids[[2]]$grid, list(n_clusters = 2:4, linkage = "ward"))
  expect_identical(grids[[3]]$grid, list(n_clusters = 2:4, affinity = "self_tuning"))
})

test_that("the full preset adds other linkages and RBF spectral clustering", {
  X <- make_blobs()$X
  grids <- default_estimator_grids(X, "full", resolve_sweep(n_clusters = 4))
  expect_length(grids, 5L)
  expect_identical(grids[[4]]$grid$linkage, c("average", "single", "complete"))
  expect_identical(grids[[5]]$grid$affinity, "rbf")
  expect_identical(grids[[5]]$grid$gamma, estimate_knn_gamma(X))
})

test_that("an axis without default grids is an error", {
  sweep <- resolve_sweep(sweep = "max_eps", sweep_values = 0.1, finer_is_larger = FALSE)
  expect_error(
    default_estimator_grids(make_blobs()$X, "light", sweep),
    "No default estimator grids for sweep parameter 'max_eps'. Pass estimator_param_grids=... explicitly.",
    fixed = TRUE
  )
})
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "grids", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "estimate_knn_gamma"`.

- [ ] Step 3: Write `R/grids.R`

```r
# Default estimator grids. Mirrors _grids.py for the k axis; stage 2 adds
# the resolution and min_cluster_size presets.

# RBF gammas around the local scale: sigma is the median distance to the
# (n_neighbors - 1)-th nearest other sample, and gamma = 1 / (2 (m sigma)^2)
# for each multiplier m.
estimate_knn_gamma <- function(X, n_neighbors = 7L, multipliers = c(0.5, 1, 2)) {
  kth <- kth_distance(X, n_neighbors)
  sigma <- stats::median(kth)
  if (sigma <= 0) {
    sigma <- if (any(kth > 0)) mean(kth[kth > 0]) else 1
  }
  1 / (2 * (multipliers * sigma)^2)
}

default_estimator_grids <- function(X, preset, sweep) {
  if (sweep@param == "n_clusters") {
    return(default_k_grids(X, sweep@values, preset))
  }
  stop(sprintf(
    "No default estimator grids for sweep parameter %s. Pass estimator_param_grids=... explicitly.",
    format_repr(sweep@param)
  ), call. = FALSE)
}

default_k_grids <- function(X, n_clusters, preset) {
  grids <- list(
    estimator_grid(KMeans, n_clusters = n_clusters),
    estimator_grid(AgglomerativeClustering, n_clusters = n_clusters, linkage = "ward"),
    estimator_grid(SpectralClustering, n_clusters = n_clusters, affinity = "self_tuning")
  )
  if (identical(preset, "full")) {
    grids <- c(grids, list(
      estimator_grid(
        AgglomerativeClustering,
        n_clusters = n_clusters,
        linkage = c("average", "single", "complete")
      ),
      estimator_grid(
        SpectralClustering,
        n_clusters = n_clusters,
        affinity = "rbf",
        gamma = estimate_knn_gamma(X)
      )
    ))
  }
  grids
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "grids", stop_on_failure = TRUE)'`
Expected: PASS.

- [ ] Step 5: Mutation check

- In `estimate_knn_gamma()`, use `kth_distance(X, n_neighbors + 1L)`: the fixture test fails.
- In `default_k_grids()`, drop `"complete"` from the full preset: the full-preset test fails.

- [ ] Step 6: Commit

```bash
git add R/grids.R tests/testthat/test-grids.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add the light and full estimator presets for the k axis

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: One resample (validation_iter)

Files:
- Create: `carve-r/R/runner.R`
- Test: `carve-r/tests/testthat/test-runner.R`

Interfaces:
- Consumes: `resolve_mode` (Task 5); `split_subsample_indices`, `count_clusters`, `scale_neighbor_count`, `call_estimator`, `adjusted_rand_index`, `default_generalizability_classifier`, `format_params`, `collect_warnings` (Tasks 3-4); `KMeans` (Task 9, in tests).
- Produces (internal):
  - `resample_indices(n_samples, seed, n_resamples, subsample_ratio, random_state, run_stability)` returns `list(train, test, stability)`; `stability` is `NULL` when `run_stability` is `FALSE`.
  - `validation_iter(X, estimator, estimator_name, params, subsample_ratio, n_resamples, seed, classifier = NULL, n_trees = 100L, sweep_param = "n_clusters", mode = "default", random_state = 0L, classifier_threads = 1L)` returns a list with `ari_stability`, `ari_generalizability` (`NaN` when skipped), `labels_train`, `labels_test`, `labels_predicted`, `labels_stability` (`NULL` when skipped), `train_indices`, `test_indices`, `stability_indices`, `n_clusters_train`, `n_clusters_test`, `n_clusters_stability` (0 when skipped) and `noise_fraction`. `random_state` must already be an integer (the runner resolves `NULL` to 0).

- [ ] Step 1: Write the failing tests in `tests/testthat/test-runner.R`

```r
blobs <- make_blobs()

test_that("validation_iter returns one resample's labels, indices and scores", {
  r <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L),
                       subsample_ratio = 0.618, n_resamples = 5L, seed = 0L, random_state = 0L)
  expect_length(r$train_indices, 55L)
  expect_length(r$test_indices, 35L)
  expect_length(r$stability_indices, 55L)
  expect_length(r$labels_train, 55L)
  expect_length(r$labels_predicted, 35L)
  expect_identical(r$ari_stability, 1)
  expect_identical(r$ari_generalizability, 1)
  expect_identical(c(r$n_clusters_train, r$n_clusters_test, r$n_clusters_stability), c(3L, 3L, 3L))
  expect_identical(r$noise_fraction, 0)
})

test_that("the subsamples come from random_state + seed and random_state + seed + n_resamples", {
  r <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L,
                       seed = 2L, random_state = 10L)
  expect_identical(r$train_indices, split_subsample_indices(90L, 0.618, 12L)$train)
  expect_identical(r$test_indices, split_subsample_indices(90L, 0.618, 12L)$test)
  expect_identical(r$stability_indices, split_subsample_indices(90L, 0.618, 17L)$train)
})

test_that("mode = 'stability' skips the held-out prediction", {
  r <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L, 0L,
                       mode = "stability", random_state = 0L)
  expect_null(r$labels_test)
  expect_null(r$labels_predicted)
  expect_true(is.nan(r$ari_generalizability))
  expect_identical(r$n_clusters_test, 0L)
  expect_identical(r$ari_stability, 1)
})

test_that("mode = 'generalizability' skips the second subsample", {
  r <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L, 0L,
                       mode = "generalizability", random_state = 0L)
  expect_null(r$stability_indices)
  expect_null(r$labels_stability)
  expect_true(is.nan(r$ari_stability))
  expect_identical(r$ari_generalizability, 1)
})

test_that("a k-axis fit with the wrong cluster count warns for each subsample", {
  two <- function(X, n_clusters = 2L) rep_len(1:2, nrow(X))
  out <- collect_warnings(validation_iter(blobs$X, two, "two", list(n_clusters = 3L), 0.618, 5L, 0L,
                                          mode = "stability", random_state = 0L))
  expect_identical(
    out$warnings,
    c("labels_1 has 2 clusters, expected 3", "labels_2 has 2 clusters, expected 3")
  )
})

test_that("off the k axis, a single-cluster subsample warns", {
  one <- function(X, resolution = 1) rep(1L, nrow(X))
  out <- collect_warnings(validation_iter(blobs$X, one, "one", list(resolution = 0.5), 0.618, 5L, 0L,
                                          sweep_param = "resolution", mode = "stability", random_state = 0L))
  expect_identical(
    out$warnings,
    "one with {'resolution': 0.5} produced 1 cluster(s) on a subsample; stability and generalizability are degenerate at this point on the sweep axis."
  )
})

test_that("each fit gets the neighbor count scaled to its rows", {
  seen <- new.env()
  seen$fits <- list()
  recorder <- function(X, n_clusters = 2L, n_neighbors = 10L) {
    seen$fits[[length(seen$fits) + 1L]] <- c(nrow(X), n_neighbors)
    rep_len(seq_len(n_clusters), nrow(X))
  }
  validation_iter(blobs$X, recorder, "recorder", list(n_clusters = 2L, n_neighbors = 10L),
                  0.618, 3L, 0L, random_state = 0L)
  fits <- do.call(rbind, seen$fits)
  # Training and stability subsamples have 55 rows, the held-out set 35:
  # 10 * 55 / 90 rounds to 6 and 10 * 35 / 90 to 4.
  expect_identical(fits[, 1], c(55L, 35L, 55L))
  expect_identical(fits[, 2], c(6L, 4L, 6L))
})

test_that("the classifier gets the thread count it is given", {
  seen <- new.env()
  clf <- function(x_train, y_train, x_test, n_threads) {
    seen$threads <- n_threads
    rep(y_train[1], nrow(x_test))
  }
  validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L, 0L,
                  classifier = clf, random_state = 0L, classifier_threads = 3L)
  expect_identical(seen$threads, 3L)
})
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "runner", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "validation_iter"`.

- [ ] Step 3: Write `R/runner.R`

```r
# The resampling loop. Mirrors _runner.py: a serial loop over
# configurations with the resamples inside it, run in this process or
# through BiocParallel. Every resample derives its own seeds from
# random_state, so the results do not depend on the backend.

# The only place the subsample seeds are derived.
resample_indices <- function(n_samples, seed, n_resamples, subsample_ratio,
                             random_state, run_stability) {
  first <- split_subsample_indices(n_samples, subsample_ratio, random_state + seed)
  second <- NULL
  if (run_stability) {
    second <- split_subsample_indices(n_samples, subsample_ratio, random_state + seed + n_resamples)$train
  }
  list(train = first$train, test = first$test, stability = second)
}

validation_iter <- function(X, estimator, estimator_name, params, subsample_ratio,
                            n_resamples, seed, classifier = NULL, n_trees = 100L,
                            sweep_param = "n_clusters", mode = "default",
                            random_state = 0L, classifier_threads = 1L) {
  policy <- resolve_mode(mode)
  n <- nrow(X)
  idx <- resample_indices(n, seed, n_resamples, subsample_ratio, random_state, policy$run_stability)

  # Each subsample is clustered on its own, with the neighbor count scaled
  # to its own rows.
  fit_labels <- function(rows) {
    scaled <- scale_neighbor_count(estimator, params, n_fit = length(rows), n_full = n)
    call_estimator(estimator, X[rows, , drop = FALSE], scaled, random_state = random_state + seed)
  }
  labels_train <- fit_labels(idx$train)
  labels_test <- if (policy$run_generalizability) fit_labels(idx$test) else NULL
  labels_stability <- if (policy$run_stability) fit_labels(idx$stability) else NULL

  k_train <- count_clusters(labels_train)
  k_test <- if (policy$run_generalizability) count_clusters(labels_test) else 0L
  k_stability <- if (policy$run_stability) count_clusters(labels_stability) else 0L

  if (sweep_param == "n_clusters") {
    expected <- as.integer(params$n_clusters)
    if (k_train != expected) {
      warning(sprintf("labels_1 has %d clusters, expected %d", k_train, expected), call. = FALSE)
    }
    if (policy$run_generalizability && k_test != expected) {
      warning(sprintf("labels_test has %d clusters, expected %d", k_test, expected), call. = FALSE)
    }
    if (policy$run_stability && k_stability != expected) {
      warning(sprintf("labels_2 has %d clusters, expected %d", k_stability, expected), call. = FALSE)
    }
  } else if (k_train < 2L) {
    warning(sprintf(
      "%s with %s produced %d cluster(s) on a subsample; stability and generalizability are degenerate at this point on the sweep axis.",
      estimator_name, format_params(params), k_train
    ), call. = FALSE)
  }

  # Stability: ARI on the samples the two subsamples share.
  ari_stability <- NaN
  if (policy$run_stability) {
    shared <- intersect(idx$train, idx$stability)
    ari_stability <- adjusted_rand_index(
      labels_train[match(shared, idx$train)],
      labels_stability[match(shared, idx$stability)]
    )
  }

  # Generalizability: the classifier learns the first subsample's clusters
  # from its raw features and predicts the held-out samples.
  labels_predicted <- NULL
  ari_generalizability <- NaN
  if (policy$run_generalizability) {
    predict_labels <- default_generalizability_classifier(
      classifier,
      n_features = ncol(X),
      n_trees = n_trees,
      random_state = random_state + seed,
      n_threads = classifier_threads
    )
    labels_predicted <- predict_labels(
      X[idx$train, , drop = FALSE],
      labels_train,
      X[idx$test, , drop = FALSE]
    )
    ari_generalizability <- adjusted_rand_index(labels_test, labels_predicted)
  }

  list(
    ari_stability = ari_stability,
    ari_generalizability = ari_generalizability,
    labels_train = labels_train,
    labels_test = labels_test,
    labels_predicted = labels_predicted,
    labels_stability = labels_stability,
    train_indices = idx$train,
    test_indices = idx$test,
    stability_indices = idx$stability,
    n_clusters_train = k_train,
    n_clusters_test = k_test,
    n_clusters_stability = k_stability,
    noise_fraction = mean(labels_train < 0)
  )
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "runner", stop_on_failure = TRUE)'`
Expected: PASS.

- [ ] Step 5: Mutation check

- In `resample_indices()`, drop `+ n_resamples` from the second seed: the seed-derivation test fails.
- In `validation_iter()`, fit the held-out set with `n_full = length(rows)` (no scaling): the neighbor-count test fails.
- In `validation_iter()`, compute the stability ARI on `labels_train[seq_along(shared)]`: the first test fails (ARI below 1), or report if it does not.

- [ ] Step 6: Commit

```bash
git add R/runner.R tests/testthat/test-runner.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add one resampling iteration with derived seeds

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: The configuration loop (run_validation) and console output

Files:
- Modify: `carve-r/R/runner.R` (append)
- Create: `carve-r/R/output.R`
- Test: `carve-r/tests/testthat/test-runner.R` (append), `carve-r/tests/testthat/test-output.R`

Interfaces:
- Consumes: `validation_iter` (Task 12); `expand_param_grid`, `estimator_grid` (Task 5); `method_id_assigner`, `sweep_rank_of`, `resolve_sweep` (Task 6); `compute_consensus_matrix`, `stability_from_consensus`, `compute_consensus_pac`, `compute_generalizability_scores` (Task 8); `resolve_core_budget`, `n_cores`, `collect_warnings`, `summarize_ari_scores`, `format_param_value` (Tasks 3-4).
- Produces (internal):
  - `run_validation(X, estimator_grids, n_resamples, subsample_ratio, classifier = NULL, n_trees = 100L, n_jobs = 1L, BPPARAM = NULL, random_state = 0L, sweep, mode = "default", show_progress = FALSE, verbose = 0L)` returns `list(records, consensus_matrices, consensus_generalizability_matrices, generalizability_scores, summaries)`. `records` is a data frame with the columns of Python's records, in Python's order (identity columns, then each estimator's parameters in first-seen order, then the sweep and score columns). The four lists are named by `config_id` (`"0"`, `"1"`, ...); entries are `NULL` for the criterion `mode` skips; `summaries` entries are `list(gini, ce, pac)`, and `summaries` is `NULL` when stability is skipped.
  - `resample_backend(outer, BPPARAM = NULL)` returns `function(X, FUN)`: `lapply` for one worker without a `BPPARAM`, else `BiocParallel::bplapply`.
  - `bind_records(records)`, `summary_columns(prefix, scores)`.
  - `print_run_header(X, sweep, n_resamples, subsample_ratio, estimator_grids, n_jobs, random_state, verbose, mode = "default")`, `print_run_footer(results, verbose)`, `log_config_progress(config_idx, total_configs, estimator_name, params, record, sweep_param, verbose)`.

- [ ] Step 1: Append the failing tests to `tests/testthat/test-runner.R`

```r
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
k_sweep <- resolve_sweep(n_clusters = 2:4)

test_that("run_validation returns one record and one artifact per configuration", {
  run <- run_validation(blobs$X, k_grid, n_resamples = 4L, subsample_ratio = 0.618,
                        random_state = 0L, sweep = k_sweep)
  expect_identical(run$records$config_id, 0:2)
  for (container in run[c("consensus_matrices", "consensus_generalizability_matrices",
                          "generalizability_scores", "summaries")]) {
    expect_identical(names(container), c("0", "1", "2"))
  }
  expect_identical(dim(run$consensus_matrices[["1"]]), c(90L, 90L))
  expect_identical(
    colnames(run$records),
    c("config_id", "method_id", "method_label", "estimator", "n_clusters",
      "sweep_param", "sweep_value", "sweep_rank", "n_clusters_observed",
      "n_clusters_observed_se", "noise_fraction",
      "ari_stability", "ari_stability_se", "ari_stability_upper", "ari_stability_lower",
      "ari_generalizability", "ari_generalizability_se", "ari_generalizability_upper",
      "ari_generalizability_lower",
      "ari_average", "ari_average_se", "ari_average_upper", "ari_average_lower")
  )
})

test_that("records carry the sweep bookkeeping and one method id per curve", {
  grids <- list(
    estimator_grid(KMeans, n_clusters = 2:3),
    estimator_grid(AgglomerativeClustering, n_clusters = 2:3, linkage = c("ward", "average"))
  )
  run <- run_validation(blobs$X, grids, n_resamples = 3L, subsample_ratio = 0.618,
                        random_state = 0L, sweep = resolve_sweep(n_clusters = 2:3))
  r <- run$records
  expect_identical(r$method_id, c("m0", "m0", "m1", "m1", "m2", "m2"))
  expect_identical(r$method_label, c(
    "KMeans", "KMeans",
    "AgglomerativeClustering, linkage=ward", "AgglomerativeClustering, linkage=ward",
    "AgglomerativeClustering, linkage=average", "AgglomerativeClustering, linkage=average"
  ))
  expect_identical(r$linkage, c(NA, NA, "ward", "ward", "average", "average"))
  expect_identical(r$sweep_rank, rep(0:1, 3))
  expect_identical(r$sweep_value, rep(2:3, 3))
  expect_identical(r$n_clusters_observed, rep(c(2, 3), 3))
  expect_identical(r$sweep_param, rep("n_clusters", 6))
})

test_that("the summaries come from the stored consensus matrix", {
  run <- run_validation(blobs$X, k_grid, 3L, 0.618, random_state = 0L, sweep = k_sweep)
  M <- run$consensus_matrices[["1"]]
  expect_identical(run$summaries[["1"]]$gini, stability_from_consensus(M)$gini)
  expect_identical(run$summaries[["1"]]$pac, compute_consensus_pac(M))
})

test_that("run_validation is reproducible and leaves the caller's RNG state alone", {
  set.seed(3)
  before <- .Random.seed
  a <- run_validation(blobs$X, k_grid, 3L, 0.618, random_state = 1L, sweep = k_sweep)
  expect_identical(.Random.seed, before)
  expect_identical(a, run_validation(blobs$X, k_grid, 3L, 0.618, random_state = 1L, sweep = k_sweep))
})

test_that("results do not depend on the parallel backend", {
  skip_on_os("windows")
  serial <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep)
  forked <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep,
                           BPPARAM = BiocParallel::MulticoreParam(2L))
  expect_identical(serial, forked)
})

# The classifier stops on an unexpected thread count. An error, unlike a
# value recorded into an environment, gets back from any backend.
threads_must_be <- function(expected) {
  function(x_train, y_train, x_test, n_threads) {
    if (!identical(n_threads, expected)) {
      stop(sprintf("classifier got %s threads", format(n_threads)))
    }
    rep(y_train[1], nrow(x_test))
  }
}

test_that("with one worker the classifier gets every core", {
  local_mocked_bindings(n_cores = function() 8L)
  expect_no_error(run_validation(blobs$X, k_grid, 2L, 0.618, classifier = threads_must_be(8L),
                                 n_jobs = 1L, random_state = 0L, sweep = k_sweep))
})

test_that("a BPPARAM sets the worker count and the classifier gets the rest", {
  local_mocked_bindings(n_cores = function() 8L)
  expect_no_error(run_validation(blobs$X, k_grid, 2L, 0.618, classifier = threads_must_be(8L),
                                 n_jobs = 4L, BPPARAM = BiocParallel::SerialParam(),
                                 random_state = 0L, sweep = k_sweep))
})

test_that("warnings raised in resamples reach the caller once per configuration", {
  noisy <- function(X, n_clusters = 2L) {
    warning("noisy estimator")
    rep_len(seq_len(n_clusters), nrow(X))
  }
  out <- collect_warnings(run_validation(
    blobs$X, list(estimator_grid(noisy, n_clusters = 2:3)), 3L, 0.618,
    random_state = 0L, sweep = resolve_sweep(n_clusters = 2:3)
  ))
  expect_identical(out$warnings, c("noisy estimator", "noisy estimator"))
})

test_that("mode = 'stability' builds no generalizability artifacts", {
  run <- run_validation(blobs$X, k_grid, 3L, 0.618, random_state = 0L, sweep = k_sweep, mode = "stability")
  expect_true(all(vapply(run$consensus_generalizability_matrices, is.null, logical(1))))
  expect_true(all(vapply(run$generalizability_scores, is.null, logical(1))))
  expect_true(all(is.nan(run$records$ari_generalizability)))
  expect_true(all(is.nan(run$records$ari_average)))
  expect_false(anyNA(run$records$ari_stability))
})

test_that("mode = 'generalizability' builds no stability artifacts", {
  run <- run_validation(blobs$X, k_grid, 3L, 0.618, random_state = 0L, sweep = k_sweep, mode = "generalizability")
  expect_null(run$summaries)
  expect_true(all(vapply(run$consensus_matrices, is.null, logical(1))))
  expect_true(all(is.nan(run$records$ari_stability)))
})

test_that("show_progress draws a bar over the configurations", {
  out <- capture.output(
    invisible(run_validation(blobs$X, k_grid, 2L, 0.618, random_state = 0L, sweep = k_sweep, show_progress = TRUE)),
    type = "message"
  )
  expect_match(paste(out, collapse = ""), "100%", fixed = TRUE)
})
```

`tests/testthat/test-output.R`:

```r
test_that("the header prints only at verbose 2 and reports the consensus memory", {
  grids <- list(estimator_grid(KMeans, n_clusters = 2:3))
  sweep <- resolve_sweep(n_clusters = 2:3)
  X <- matrix(0, 1000, 2)
  expect_silent(print_run_header(X, sweep, 10L, 0.618, grids, 1L, 0L, verbose = 1L))
  text <- paste(capture_messages(print_run_header(X, sweep, 10L, 0.618, grids, 1L, 0L, verbose = 2L)), collapse = "")
  expect_match(text, "[CARVE] Validation Settings:", fixed = TRUE)
  expect_match(text, "[CARVE] n_samples          : 1000", fixed = TRUE)
  expect_match(text, "[CARVE] n_clusters         : [2 3]", fixed = TRUE)
  expect_match(text, "[CARVE] total configs      : 2", fixed = TRUE)
  expect_match(text, "[CARVE] consensus memory   : 0.03 GB", fixed = TRUE)
  expect_match(text, "[CARVE] Starting validation ...", fixed = TRUE)
})

test_that("a single-criterion mode halves the projected memory", {
  grids <- list(estimator_grid(KMeans, n_clusters = 2:3))
  text <- paste(capture_messages(print_run_header(
    matrix(0, 1000, 2), resolve_sweep(n_clusters = 2:3), 10L, 0.618, grids, 1L, 0L,
    verbose = 2L, mode = "stability"
  )), collapse = "")
  expect_match(text, "[CARVE] consensus memory   : 0.02 GB", fixed = TRUE)
})

test_that("per-configuration progress prints from verbose 1", {
  record <- list(ari_stability = 0.9, ari_stability_se = 0.01,
                 ari_generalizability = 0.8, ari_generalizability_se = 0.02)
  expect_silent(log_config_progress(1L, 4L, "KMeans", list(n_clusters = 2L), record, "n_clusters", verbose = 0L))
  expect_message(
    log_config_progress(1L, 4L, "KMeans", list(n_clusters = 2L), record, "n_clusters", verbose = 1L),
    "[CARVE] [1/4] est=KMeans n_clusters=2 | ARI_stab=0.900±0.010  ARI_gen=0.800±0.020",
    fixed = TRUE
  )
})

test_that("the footer prints only at verbose 2", {
  expect_silent(print_run_footer(data.frame(a = 1:3), verbose = 1L))
  expect_message(
    print_run_footer(data.frame(a = 1:3), verbose = 2L),
    "[CARVE] finished. evaluated 3 estimator configurations.",
    fixed = TRUE
  )
})
```

The memory figures: two configurations, 1000 samples, 8 bytes per entry. Both criteria: 2 x 2 x 1000^2 x 8 = 32 MB, printed `0.03 GB`; stability only: 16 MB, printed `0.02 GB`.

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "runner|output", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "run_validation"` and `"print_run_header"`.

- [ ] Step 3: Write `R/output.R`

```r
# Console output of a run. Mirrors _output.py, through message() so that
# suppressMessages() silences it.

print_run_header <- function(X, sweep, n_resamples, subsample_ratio, estimator_grids,
                             n_jobs, random_state, verbose, mode = "default") {
  if (verbose < 2) {
    return(invisible(NULL))
  }
  total <- sum(vapply(estimator_grids, function(g) length(expand_param_grid(g$grid)), integer(1)))
  policy <- resolve_mode(mode)
  n_matrices <- total * (policy$run_stability + policy$run_generalizability)
  memory_gb <- n_matrices * as.numeric(nrow(X))^2 * 8 / 1e9
  line <- strrep("=", 60)
  message(paste(c(
    paste("[CARVE]", line),
    "[CARVE] Validation Settings:",
    sprintf("[CARVE] n_samples          : %d", nrow(X)),
    sprintf("[CARVE] n_features         : %d", ncol(X)),
    sprintf("[CARVE] sweep parameter    : %s", sweep@param),
    sprintf(
      "[CARVE] %-19s: [%s]",
      sweep@param,
      paste(vapply(sweep@values, format_param_value, character(1)), collapse = " ")
    ),
    sprintf("[CARVE] n_resamples        : %d", as.integer(n_resamples)),
    sprintf("[CARVE] subsample_ratio    : %s", format(subsample_ratio)),
    sprintf("[CARVE] n_jobs             : %s", if (is.null(n_jobs)) "NULL" else format(n_jobs)),
    sprintf("[CARVE] total configs      : %d", total),
    sprintf("[CARVE] consensus memory   : %.2f GB", memory_gb),
    sprintf("[CARVE] random_state       : %d", as.integer(random_state)),
    paste("[CARVE]", line),
    "",
    "[CARVE] Starting validation ...",
    ""
  ), collapse = "\n"))
}

print_run_footer <- function(results, verbose) {
  if (verbose < 2) {
    return(invisible(NULL))
  }
  message(sprintf("\n[CARVE] finished. evaluated %d estimator configurations.", nrow(results)))
}

log_config_progress <- function(config_idx, total_configs, estimator_name, params, record,
                                sweep_param, verbose) {
  if (verbose <= 0) {
    return(invisible(NULL))
  }
  value <- params[[sweep_param]]
  message(sprintf(
    "[CARVE] [%d/%d] est=%s %s=%s | ARI_stab=%.3f±%.3f  ARI_gen=%.3f±%.3f  ",
    as.integer(config_idx),
    as.integer(total_configs),
    estimator_name,
    sweep_param,
    if (is.null(value)) "?" else format(value),
    record$ari_stability,
    record$ari_stability_se,
    record$ari_generalizability,
    record$ari_generalizability_se
  ))
}
```

- [ ] Step 4: Append to `R/runner.R`

```r
resample_backend <- function(outer, BPPARAM = NULL) {
  if (is.null(BPPARAM) && outer == 1L) {
    return(function(X, FUN) lapply(X, FUN))
  }
  if (is.null(BPPARAM)) {
    BPPARAM <- if (.Platform$OS.type == "windows") {
      BiocParallel::SnowParam(workers = outer)
    } else {
      BiocParallel::MulticoreParam(workers = outer)
    }
  }
  function(X, FUN) BiocParallel::bplapply(X, FUN, BPPARAM = BPPARAM)
}

summary_columns <- function(prefix, scores) {
  s <- summarize_ari_scores(scores)
  stats::setNames(as.list(unname(s)), paste0(prefix, c("", "_se", "_upper", "_lower")))
}

# One row per record. Columns appear in first-seen order and are NA where a
# record lacks them (another estimator's parameters), as in pandas'
# DataFrame.from_records. A parameter whose values are not scalars becomes a
# list column.
bind_records <- function(records) {
  columns <- unique(unlist(lapply(records, names)))
  out <- lapply(columns, function(column) {
    values <- lapply(records, function(record) {
      value <- record[[column]]
      if (is.null(value)) NA else value
    })
    scalar <- all(lengths(values) == 1L) && all(vapply(values, is.atomic, logical(1)))
    if (scalar) unlist(values) else I(values)
  })
  names(out) <- columns
  as.data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
}

run_validation <- function(X, estimator_grids, n_resamples, subsample_ratio, classifier = NULL,
                           n_trees = 100L, n_jobs = 1L, BPPARAM = NULL, random_state = 0L,
                           sweep, mode = "default", show_progress = FALSE, verbose = 0L) {
  policy <- resolve_mode(mode)
  n <- nrow(X)
  # Workers over resamples (outer), and threads for each worker's forest
  # (inner). A BPPARAM brings its own worker count.
  budget <- resolve_core_budget(n_jobs, n_resamples)
  if (!is.null(BPPARAM)) {
    workers <- as.integer(BiocParallel::bpnworkers(BPPARAM))
    budget <- c(outer = workers, inner = max(1L, n_cores() %/% workers))
  }
  apply_resamples <- resample_backend(budget[["outer"]], BPPARAM)

  configs <- unlist(lapply(estimator_grids, function(g) {
    lapply(expand_param_grid(g$grid), function(params) list(grid = g, params = params))
  }), recursive = FALSE)
  total <- length(configs)
  ids <- as.character(seq_len(total) - 1L)
  assign_method <- method_id_assigner(sweep@param)

  records <- vector("list", total)
  consensus <- stats::setNames(vector("list", total), ids)
  consensus_generalizability <- stats::setNames(vector("list", total), ids)
  generalizability_scores <- stats::setNames(vector("list", total), ids)
  summaries <- stats::setNames(vector("list", total), ids)

  bar <- NULL
  if (show_progress) {
    bar <- utils::txtProgressBar(min = 0, max = total, style = 3, file = stderr())
    on.exit(close(bar), add = TRUE)
  }

  for (i in seq_len(total)) {
    g <- configs[[i]]$grid
    params <- configs[[i]]$params
    outputs <- apply_resamples(seq_len(n_resamples) - 1L, function(b) {
      collect_warnings(validation_iter(
        X, g$estimator, g$name, params,
        subsample_ratio = subsample_ratio,
        n_resamples = n_resamples,
        seed = b,
        classifier = classifier,
        n_trees = n_trees,
        sweep_param = sweep@param,
        mode = mode,
        random_state = random_state,
        classifier_threads = budget[["inner"]]
      ))
    })
    for (text in unique(unlist(lapply(outputs, function(o) o$warnings)))) {
      warning(text, call. = FALSE)
    }
    results <- lapply(outputs, function(o) o$value)

    stability_runs <- Filter(function(r) length(r$labels_train) > 0L, results)
    held_out_runs <- Filter(
      function(r) length(r$labels_predicted) > 0L && length(r$labels_test) > 0L,
      results
    )
    if (policy$run_stability) {
      M <- compute_consensus_matrix(n, lapply(stability_runs, function(r) {
        list(indices = r$train_indices, labels = r$labels_train)
      }))
      scores <- stability_from_consensus(M)
      consensus[i] <- list(M)
      summaries[i] <- list(list(gini = scores$gini, ce = scores$ce, pac = compute_consensus_pac(M)))
    }
    if (policy$run_generalizability) {
      consensus_generalizability[i] <- list(compute_consensus_matrix(n, lapply(held_out_runs, function(r) {
        list(indices = r$test_indices, labels = r$labels_predicted)
      })))
      generalizability_scores[i] <- list(compute_generalizability_scores(n, lapply(held_out_runs, function(r) {
        list(indices = r$test_indices, true = r$labels_test, predicted = r$labels_predicted)
      })))
    }

    aris_stability <- vapply(results, function(r) r$ari_stability, numeric(1))
    aris_generalizability <- vapply(results, function(r) r$ari_generalizability, numeric(1))
    aris_average <- if (policy$compute_average_ari) {
      (aris_stability + aris_generalizability) / 2
    } else {
      rep(NaN, length(results))
    }
    k_observed <- vapply(results, function(r) as.numeric(r$n_clusters_train), numeric(1))
    noise <- vapply(results, function(r) r$noise_fraction, numeric(1))
    sweep_value <- params[[sweep@param]]
    method <- assign_method(g$name, params)

    record <- c(
      list(config_id = i - 1L, method_id = method[[1L]], method_label = method[[2L]], estimator = g$name),
      params,
      list(
        sweep_param = sweep@param,
        sweep_value = sweep_value,
        sweep_rank = sweep_rank_of(sweep, sweep_value),
        n_clusters_observed = mean(k_observed),
        n_clusters_observed_se = if (length(k_observed) > 1L) {
          stats::sd(k_observed) / sqrt(length(k_observed))
        } else {
          NaN
        },
        noise_fraction = mean(noise)
      ),
      summary_columns("ari_stability", aris_stability),
      summary_columns("ari_generalizability", aris_generalizability),
      summary_columns("ari_average", aris_average)
    )
    records[[i]] <- record
    log_config_progress(i, total, g$name, params, record, sweep@param, verbose)
    if (!is.null(bar)) {
      utils::setTxtProgressBar(bar, i)
    }
  }

  list(
    records = bind_records(records),
    consensus_matrices = consensus,
    consensus_generalizability_matrices = consensus_generalizability,
    generalizability_scores = generalizability_scores,
    summaries = if (policy$run_stability) summaries else NULL
  )
}
```

- [ ] Step 5: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "runner|output", stop_on_failure = TRUE)'`
Expected: PASS, no warnings. The backend test forks two workers.

- [ ] Step 6: Mutation check

- In `run_validation()`, drop `unique()` around the re-emitted warnings: "warnings raised in resamples reach the caller once per configuration" fails (six warnings instead of two).
- In `run_validation()`, ignore `BPPARAM`'s worker count (keep `budget` from `n_jobs = 4`, capped at the 2 resamples): "a BPPARAM sets the worker count" fails with "classifier got 4 threads".
- In `bind_records()`, build columns from the first record's names only: the `linkage` assertion fails.
- In `print_run_header()`, count one matrix per configuration: the memory assertions fail.

- [ ] Step 7: Commit

```bash
git add R/runner.R R/output.R tests/testthat/test-runner.R tests/testthat/test-output.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add the configuration loop, the parallel backend and run output

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: The CARVE class and carve()

Files:
- Modify: `carve-r/R/AllClasses.R` (append the `CARVE` class and its validity method)
- Create: `carve-r/R/AllGenerics.R`, `carve-r/R/carve.R`
- Test: `carve-r/tests/testthat/test-carve.R`

Interfaces:
- Consumes: everything from Tasks 3-13: `as_data_matrix`, `check_dots`, `format_repr` (utils); `resolve_mode`, `estimator_grid` (types); `resolve_sweep`, `infer_sweep_param`, `grid_sweep_values`, `validate_grids` (sweep); `default_estimator_grids` (grids); `run_validation`, `print_run_header`, `print_run_footer` (runner, output).
- Produces:
  - S4 class `CARVE` (exported) with slots `input_data` (matrix), `reference_labels` (`carveIntegerOrNULL`), `run_params` (list: `n_resamples`, `subsample_ratio`, `n_trees`, `n_jobs`, `mode`, `random_state`, `classifier`), `estimator_results` (data.frame), `estimator_param_grids` (list), `sweep` (`SweepSpec`), `consensus_matrices` (list), `consensus_generalizability_matrices` (list), `stability_gini_scores` (`carveListOrNULL`), `stability_ce_scores` (`carveListOrNULL`), `generalizability_scores` (list). Validity: `config_id` is `0:(n - 1)` in table order and every non-NULL container is named by it.
  - Generic `carve(x, ...)` with one method for `"ANY"` (exported). Signature: `carve(x, n_clusters = 2:10, n_resamples = 100, subsample_ratio = 0.618, estimator_param_grids = "light", classifier = NULL, n_trees = 100, reference_labels = NULL, mode = "default", n_jobs = 1, BPPARAM = NULL, random_state = NULL, show_progress = FALSE, verbose = 0, ...)`.
  - Internal `fit_carve(X, ...)` (same arguments without `x` and `...`), `resolve_seed(random_state, n_resamples)` returning an integer, `coerce_reference_labels(reference_labels, n_samples)` returning an integer vector or `NULL`.

- [ ] Step 1: Write the failing tests in `tests/testthat/test-carve.R`

```r
blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:4))

small_fit <- function(...) {
  carve(blobs$X, n_clusters = 2:4, n_resamples = 4, estimator_param_grids = k_grid,
        random_state = 0, ...)
}

test_that("carve returns a CARVE object whose containers follow config_id", {
  fit <- small_fit()
  expect_s4_class(fit, "CARVE")
  expect_no_error(validObject(fit))
  results <- fit@estimator_results
  expect_identical(results$config_id, 0:2)
  expect_identical(names(fit@consensus_matrices), c("0", "1", "2"))
  expect_identical(names(fit@stability_gini_scores), c("0", "1", "2"))
  expect_identical(names(fit@generalizability_scores), c("0", "1", "2"))
  expect_identical(fit@input_data, blobs$X)
  expect_identical(fit@sweep@values, 2:4)
  expect_identical(fit@run_params$random_state, 0L)
  expect_identical(
    tail(names(results), 4),
    c("consensus_pac_stability", "consensus_gini_stability", "consensus_ce_stability", "accuracy_generalizability")
  )
  expect_equal(results$consensus_gini_stability[2], mean(fit@stability_gini_scores[["1"]]))
  expect_equal(results$consensus_ce_stability[2], mean(fit@stability_ce_scores[["1"]]))
  expect_equal(results$accuracy_generalizability[3], mean(fit@generalizability_scores[["2"]]))
})

test_that("the light preset runs three estimators per k", {
  fit <- carve(blobs$X, n_clusters = 3, n_resamples = 2, random_state = 0)
  expect_identical(
    unique(fit@estimator_results$estimator),
    c("KMeans", "AgglomerativeClustering", "SpectralClustering")
  )
  expect_identical(fit@estimator_results$n_clusters, rep(2:3, 3))
})

test_that("carve validates the run settings before running", {
  X <- blobs$X
  expect_error(
    carve(X, subsample_ratio = 1),
    "subsample_ratio must be in (0, 1), got 1. Each resample needs both a training and a held-out split.",
    fixed = TRUE
  )
  expect_error(carve(X, n_resamples = 0), "n_resamples must be at least 1, got 0.", fixed = TRUE)
  expect_error(carve(X, n_trees = 0), "n_trees must be at least 1, got 0.", fixed = TRUE)
  expect_error(
    carve(X, estimator_param_grids = "medium"),
    "Unknown estimator_param_grids preset 'medium'. Expected 'light', 'full', or a list of estimator_grid() specifications.",
    fixed = TRUE
  )
  expect_error(carve(X, estimator_param_grids = list()), "estimator_param_grids must not be empty.", fixed = TRUE)
  expect_error(carve(X, classifier = "rf"), "classifier must be NULL or a function.", fixed = TRUE)
  expect_error(carve(X, k = 3), "Unknown argument: k.", fixed = TRUE)
  expect_error(carve(X, random_state = -1), "random_state must be NULL or a non-negative whole number.", fixed = TRUE)
  expect_error(carve(X, reference_labels = 1:3), "reference_labels has 3 entries but X has 90 rows.", fixed = TRUE)
  expect_error(carve(list(1, 2)), "X must be a numeric matrix", fixed = TRUE)
})

test_that("a non-default mode warns and skips the other criterion", {
  fit <- NULL
  expect_warning(
    fit <- small_fit(mode = "stability"),
    "Non-default mode is experimental and may break downstream functionality.",
    fixed = TRUE
  )
  expect_true(all(vapply(fit@generalizability_scores, is.null, logical(1))))
  expect_true(all(is.nan(fit@estimator_results$accuracy_generalizability)))
  expect_warning(fit <- small_fit(mode = "generalizability"), "Non-default mode is experimental", fixed = TRUE)
  expect_null(fit@stability_gini_scores)
  expect_null(fit@stability_ce_scores)
  expect_true(all(is.nan(fit@estimator_results$consensus_pac_stability)))
})

test_that("n_trees with a custom classifier warns", {
  clf <- function(x_train, y_train, x_test) rep(y_train[1], nrow(x_test))
  expect_warning(
    small_fit(classifier = clf, n_trees = 50),
    "n_trees is ignored when a custom classifier is provided.",
    fixed = TRUE
  )
})

test_that("a data frame gives the same results as the matrix", {
  from_frame <- carve(as.data.frame(blobs$X), n_clusters = 2:4, n_resamples = 4,
                      estimator_param_grids = k_grid, random_state = 0)
  expect_identical(from_frame@estimator_results, small_fit()@estimator_results)
})

test_that("reference labels are coded as integers", {
  fit <- small_fit(reference_labels = rep(c("b", "a", NA), each = 30))
  expect_identical(fit@reference_labels, rep(c(1L, 2L, -1L), each = 30))
  fit <- small_fit(reference_labels = rep(c(3, 1, 2), each = 30))
  expect_identical(fit@reference_labels, rep(c(3L, 1L, 2L), each = 30))
})

test_that("a fixed random_state reproduces the fit and NULL means 0", {
  expect_identical(small_fit(), small_fit())
  null_seed <- carve(blobs$X, n_clusters = 2:4, n_resamples = 4, estimator_param_grids = k_grid)
  expect_identical(null_seed@estimator_results, small_fit()@estimator_results)
})

test_that("a custom grid with one k value is kept as given", {
  fit <- carve(blobs$X, n_resamples = 2, random_state = 0,
               estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 3L)))
  expect_identical(fit@sweep@values, 3L)
  expect_identical(fit@estimator_results$n_clusters, 3L)
})

test_that("custom grids may sweep another registered parameter", {
  cut_at <- function(X, resolution = 1) {
    as.integer(stats::cutree(stats::hclust(stats::dist(X)), k = round(2 * resolution)))
  }
  fit <- carve(blobs$X, n_resamples = 2, random_state = 0,
               estimator_param_grids = list(estimator_grid(cut_at, resolution = c(1, 1.5))))
  expect_identical(fit@sweep@param, "resolution")
  expect_identical(fit@estimator_results$n_clusters_observed, c(2, 3))
  expect_identical(fit@estimator_results$sweep_rank, 0:1)
})

test_that("grids that mix sweep axes are rejected", {
  toy <- function(X, n_clusters = 2L, resolution = 1) rep(1L, nrow(X))
  expect_error(
    carve(blobs$X, estimator_param_grids = list(
      estimator_grid(toy, n_clusters = 2:3),
      estimator_grid(toy, resolution = c(0.5, 1))
    )),
    "Estimator grids sweep more than one parameter",
    fixed = TRUE
  )
})

test_that("a misaligned container fails validation", {
  fit <- small_fit()
  fit@consensus_matrices <- rev(fit@consensus_matrices)
  expect_error(
    validObject(fit),
    "config_id is misaligned with the per-configuration artifact containers. This is an internal CARVE error.",
    fixed = TRUE
  )
})

test_that("config_id must count from 0 in table order", {
  fit <- small_fit()
  fit@estimator_results$config_id <- c(1L, 0L, 2L)
  expect_error(validObject(fit), "config_id is misaligned", fixed = TRUE)
})
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "carve", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "carve"`.

- [ ] Step 3: Append the class to `R/AllClasses.R`

```r
#' Fitted CARVE object
#'
#' [carve()] returns an object of this class. Read it with the accessor
#' functions; the slots are listed here for reference.
#'
#' @slot input_data The data matrix of the fit, one row per sample.
#' @slot reference_labels Reference labels from `carve(reference_labels = )`,
#'   coded as integers, or `NULL`.
#' @slot run_params List of the run settings: `n_resamples`,
#'   `subsample_ratio`, `n_trees`, `n_jobs`, `mode`, `random_state` (the seed
#'   the run used) and `classifier`.
#' @slot estimator_results Data frame with one row per configuration; see
#'   [estimator_results()].
#' @slot estimator_param_grids The estimator grids the run evaluated.
#' @slot sweep The run's [SweepSpec-class] object.
#' @slot consensus_matrices Stability consensus matrices, a list named by
#'   `config_id`. Entries are `NULL` when the run skipped stability.
#' @slot consensus_generalizability_matrices Consensus matrices of the
#'   classifier's held-out predictions, a list named by `config_id`. Entries
#'   are `NULL` when the run skipped generalizability.
#' @slot stability_gini_scores Per-sample Gini stability, a list named by
#'   `config_id`, or `NULL` when the run skipped stability.
#' @slot stability_ce_scores Per-sample cross-entropy stability, a list named
#'   by `config_id`, or `NULL` when the run skipped stability.
#' @slot generalizability_scores Per-sample held-out accuracy, a list named by
#'   `config_id`. Entries are `NULL` when the run skipped generalizability.
#' @seealso [carve()], [estimator_results()], [get_labels()]
#' @export
setClass(
  "CARVE",
  slots = c(
    input_data = "matrix",
    reference_labels = "carveIntegerOrNULL",
    run_params = "list",
    estimator_results = "data.frame",
    estimator_param_grids = "list",
    sweep = "SweepSpec",
    consensus_matrices = "list",
    consensus_generalizability_matrices = "list",
    stability_gini_scores = "carveListOrNULL",
    stability_ce_scores = "carveListOrNULL",
    generalizability_scores = "list"
  )
)

# config_id joins a results row to its matrices and scores. It must count
# from 0 in table order, and every container must be named by it.
setValidity("CARVE", function(object) {
  expected <- seq_len(nrow(object@estimator_results)) - 1L
  containers <- list(
    object@consensus_matrices,
    object@consensus_generalizability_matrices,
    object@generalizability_scores,
    object@stability_gini_scores,
    object@stability_ce_scores
  )
  named_by_id <- vapply(
    containers,
    function(x) is.null(x) || identical(as.character(names(x)), as.character(expected)),
    logical(1)
  )
  if (identical(as.integer(object@estimator_results$config_id), expected) && all(named_by_id)) {
    return(TRUE)
  }
  "config_id is misaligned with the per-configuration artifact containers. This is an internal CARVE error."
})
```

- [ ] Step 4: Write `R/AllGenerics.R`

```r
#' @include AllClasses.R
NULL

#' @rdname carve
#' @export
setGeneric("carve", function(x, ...) standardGeneric("carve"))
```

- [ ] Step 5: Write `R/carve.R`

```r
#' @include AllGenerics.R
NULL

#' Validate clusterings by resampling
#'
#' `carve()` clusters many subsamples of the data with every configuration in
#' the estimator grids and scores each configuration on two criteria.
#' Stability is the adjusted Rand index (ARI) between the clusterings of two
#' overlapping subsamples, computed on the samples they share.
#' Generalizability is the ARI between the clustering of the held-out samples
#' and the labels a random forest predicts for them after training on the
#' first subsample and its clustering. Each configuration also gets consensus
#' matrices: for every pair of samples, the share of the subsamples drawing
#' both in which the two fell in the same cluster.
#'
#' Resample `b`, counting from 0, draws its first subsample with seed
#' `random_state + b` and its second with `random_state + b + n_resamples`,
#' and its estimator and classifier use `random_state + b`. The results are
#' therefore the same for every `n_jobs` and `BPPARAM`, and the session's
#' random number stream is left as it was.
#'
#' `n_jobs` is a core budget. The resamples are spread over `n_jobs` worker
#' processes, at most `n_resamples` of them, and each worker's random forest
#' gets the core count divided by the number of workers. With `n_jobs = 1`
#' one process runs every resample and the forest uses every core; with
#' `n_jobs = -1` there is one worker per core, each with a single-threaded
#' forest. Every worker holds its own copy of the data and of its forest, so
#' memory grows with the worker count.
#'
#' The fit keeps two `n` by `n` consensus matrices per configuration, stored
#' as doubles: `8 * n^2` bytes each, twice what the Python package needs for
#' the same run. With `verbose = 2` the header reports the total.
#'
#' @param x The data, one row per sample: a numeric matrix, a numeric data
#'   frame, a sparse matrix from the Matrix package, or a numeric vector.
#' @param ... Not used. An argument name `carve()` does not know is an error.
#' @param n_clusters Numbers of clusters to evaluate. A single number `K`
#'   means `2:K`. Custom grids set their own values.
#' @param n_resamples Resamples per configuration.
#' @param subsample_ratio Share of the samples drawn, without replacement,
#'   into each subsample. Must lie strictly between 0 and 1.
#' @param estimator_param_grids `"light"` (KMeans, Ward-linkage agglomerative
#'   and self-tuning spectral clustering), `"full"` (adds average, single and
#'   complete linkage and RBF spectral clustering), or a list of
#'   [estimator_grid()] specifications. All grids must sweep the same
#'   parameter over the same values.
#' @param classifier `NULL` for the default random forest, or a function
#'   `(x_train, y_train, x_test)` that returns one predicted label per row of
#'   `x_test`. If it also has `n_threads` or `random_state` arguments, CARVE
#'   fills them in.
#' @param n_trees Trees in the default random forest. Not used when
#'   `classifier` is given.
#' @param reference_labels Labels, one per sample, that [get_labels()]
#'   renames its clusters to match. Whole numbers keep their values; other
#'   vectors are coded 1, 2, ... in order of first appearance, and `NA`
#'   becomes -1.
#' @param mode `"default"` scores both criteria, `"stability"` skips
#'   generalizability and `"generalizability"` skips stability. The last two
#'   are experimental.
#' @param n_jobs Core budget; see Details. `-1` means every core and `-2` all
#'   but one.
#' @param BPPARAM Optional [BiocParallel::BiocParallelParam-class] backend for
#'   the resamples. Its worker count replaces the one `n_jobs` would set.
#' @param random_state Seed of the run. `NULL` means 0.
#' @param show_progress Show a progress bar over the configurations.
#' @param verbose 0 prints nothing, 1 prints a line per configuration, and 2
#'   also prints a header and a footer.
#'
#' @return A [CARVE-class] object.
#' @seealso [get_k()], [get_labels()], [estimator_results()]
#' @examples
#' set.seed(1)
#' X <- rbind(
#'   matrix(rnorm(60, mean = 0, sd = 0.3), ncol = 2),
#'   matrix(rnorm(60, mean = 3, sd = 0.3), ncol = 2)
#' )
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0)
#' fit
#' estimator_results(fit)[, c("method_label", "n_clusters", "ari_stability")]
#' @export
setMethod("carve", "ANY", function(x, n_clusters = 2:10, n_resamples = 100,
                                   subsample_ratio = 0.618, estimator_param_grids = "light",
                                   classifier = NULL, n_trees = 100, reference_labels = NULL,
                                   mode = "default", n_jobs = 1, BPPARAM = NULL,
                                   random_state = NULL, show_progress = FALSE, verbose = 0,
                                   ...) {
  check_dots(...)
  fit_carve(
    as_data_matrix(x),
    n_clusters = n_clusters,
    n_resamples = n_resamples,
    subsample_ratio = subsample_ratio,
    estimator_param_grids = estimator_param_grids,
    classifier = classifier,
    n_trees = n_trees,
    reference_labels = reference_labels,
    mode = mode,
    n_jobs = n_jobs,
    BPPARAM = BPPARAM,
    random_state = random_state,
    show_progress = show_progress,
    verbose = verbose
  )
})

fit_carve <- function(X, n_clusters, n_resamples, subsample_ratio, estimator_param_grids,
                      classifier, n_trees, reference_labels, mode, n_jobs, BPPARAM,
                      random_state, show_progress, verbose) {
  if (!is.numeric(subsample_ratio) || length(subsample_ratio) != 1L ||
      is.na(subsample_ratio) || subsample_ratio <= 0 || subsample_ratio >= 1) {
    stop(sprintf(
      "subsample_ratio must be in (0, 1), got %s. Each resample needs both a training and a held-out split.",
      format(subsample_ratio)
    ), call. = FALSE)
  }
  if (n_resamples < 1) {
    stop(sprintf("n_resamples must be at least 1, got %s.", format(n_resamples)), call. = FALSE)
  }
  if (!is.null(classifier) && !is.function(classifier)) {
    stop("classifier must be NULL or a function.", call. = FALSE)
  }
  if (is.null(classifier) && n_trees < 1) {
    stop(sprintf("n_trees must be at least 1, got %s.", format(n_trees)), call. = FALSE)
  }
  policy <- resolve_mode(mode)
  if (policy$mode != "default") {
    warning("Non-default mode is experimental and may break downstream functionality.", call. = FALSE)
  }
  if (!is.null(classifier) && n_trees != 100) {
    warning("n_trees is ignored when a custom classifier is provided.", call. = FALSE)
  }
  n_resamples <- as.integer(n_resamples)
  seed <- resolve_seed(random_state, n_resamples)
  reference <- coerce_reference_labels(reference_labels, nrow(X))

  # The sweep axis, and the grids along it. Custom grids decide the swept
  # parameter, so a grid sweeping resolution works without resolution=.
  if (is.character(estimator_param_grids) && length(estimator_param_grids) == 1L &&
      estimator_param_grids %in% c("light", "full")) {
    sweep <- resolve_sweep(n_clusters = n_clusters)
    grids <- default_estimator_grids(X, preset = estimator_param_grids, sweep = sweep)
  } else if (is.list(estimator_param_grids) && !is.object(estimator_param_grids) &&
             all(vapply(estimator_param_grids, inherits, logical(1), "carve_estimator_grid"))) {
    grids <- estimator_param_grids
    param <- infer_sweep_param(grids)
    values <- if (is.null(param)) NULL else grid_sweep_values(grids, param)
    sweep <- resolve_sweep(n_clusters = n_clusters, sweep = param, sweep_values = values)
    sweep@values <- validate_grids(grids, sweep)
  } else {
    stop(sprintf(
      "Unknown estimator_param_grids preset %s. Expected 'light', 'full', or a list of estimator_grid() specifications.",
      format_repr(estimator_param_grids)
    ), call. = FALSE)
  }

  print_run_header(X, sweep, n_resamples, subsample_ratio, grids, n_jobs, seed, verbose, policy$mode)
  run <- run_validation(
    X, grids,
    n_resamples = n_resamples,
    subsample_ratio = subsample_ratio,
    classifier = classifier,
    n_trees = as.integer(n_trees),
    n_jobs = n_jobs,
    BPPARAM = BPPARAM,
    random_state = seed,
    sweep = sweep,
    mode = policy$mode,
    show_progress = show_progress,
    verbose = verbose
  )

  results <- run$records
  gini <- NULL
  ce <- NULL
  if (policy$run_stability) {
    gini <- lapply(run$summaries, function(s) s$gini)
    ce <- lapply(run$summaries, function(s) s$ce)
    results$consensus_pac_stability <- unname(vapply(run$summaries, function(s) s$pac, numeric(1)))
    # A plain mean, as in Python: one sample never drawn with a partner
    # makes the configuration's mean NaN.
    results$consensus_gini_stability <- unname(vapply(gini, mean, numeric(1)))
    results$consensus_ce_stability <- unname(vapply(ce, mean, numeric(1)))
  } else {
    results$consensus_pac_stability <- NaN
    results$consensus_gini_stability <- NaN
    results$consensus_ce_stability <- NaN
  }
  results$accuracy_generalizability <- if (policy$run_generalizability) {
    unname(vapply(run$generalizability_scores, mean, numeric(1)))
  } else {
    NaN
  }

  fit <- methods::new(
    "CARVE",
    input_data = X,
    reference_labels = reference,
    run_params = list(
      n_resamples = n_resamples,
      subsample_ratio = subsample_ratio,
      n_trees = as.integer(n_trees),
      n_jobs = n_jobs,
      mode = policy$mode,
      random_state = seed,
      classifier = classifier
    ),
    estimator_results = results,
    estimator_param_grids = grids,
    sweep = sweep,
    consensus_matrices = run$consensus_matrices,
    consensus_generalizability_matrices = run$consensus_generalizability_matrices,
    stability_gini_scores = gini,
    stability_ce_scores = ce,
    generalizability_scores = run$generalizability_scores
  )
  print_run_footer(results, verbose)
  fit
}

# NULL means 0, as in Python. Every derived seed, up to
# random_state + 2 * n_resamples, must fit in an R integer.
resolve_seed <- function(random_state, n_resamples) {
  if (is.null(random_state)) {
    return(0L)
  }
  if (!is.numeric(random_state) || length(random_state) != 1L || is.na(random_state) ||
      random_state != round(random_state) || random_state < 0) {
    stop("random_state must be NULL or a non-negative whole number.", call. = FALSE)
  }
  limit <- .Machine$integer.max - 2 * n_resamples
  if (random_state > limit) {
    stop(sprintf(
      "random_state must be at most %.0f so that every derived seed fits in an R integer.",
      limit
    ), call. = FALSE)
  }
  as.integer(random_state)
}

# Whole numbers keep their values. Anything else is coded 1, 2, ... in order
# of first appearance, and NA becomes -1, a sample without a reference label.
coerce_reference_labels <- function(reference_labels, n_samples) {
  if (is.null(reference_labels)) {
    return(NULL)
  }
  if (length(reference_labels) != n_samples) {
    stop(sprintf(
      "reference_labels has %d entries but X has %d rows.",
      length(reference_labels), n_samples
    ), call. = FALSE)
  }
  if (is.numeric(reference_labels) && !anyNA(reference_labels) &&
      all(reference_labels == round(reference_labels))) {
    return(as.integer(reference_labels))
  }
  values <- as.character(reference_labels)
  codes <- match(values, unique(values[!is.na(values)]))
  codes[is.na(values)] <- -1L
  as.integer(codes)
}
```

- [ ] Step 6: Run the tests and see them pass

Run: `Rscript -e 'devtools::document(); devtools::test(filter = "carve", stop_on_failure = TRUE)'`
Expected: `man/carve.Rd` and `man/CARVE-class.Rd` written; tests PASS. Roxygen warns about links to topics that Task 15 adds (`estimator_results`, `get_k`, `get_labels`); that is expected until then.

- [ ] Step 7: Mutation check

- In the validity method, drop the `identical(as.integer(...config_id), expected)` part: "config_id must count from 0 in table order" fails.
- In `fit_carve()`, compute `consensus_gini_stability` with `median` in place of `mean`: the `consensus_gini_stability` assertion fails.
- In `fit_carve()`, skip `sweep@values <- validate_grids(...)`: "a custom grid with one k value is kept as given" fails.
- In `coerce_reference_labels()`, code NA as `0L`: "reference labels are coded as integers" fails.

- [ ] Step 8: Commit

```bash
git add R/AllClasses.R R/AllGenerics.R R/carve.R tests/testthat/test-carve.R NAMESPACE DESCRIPTION man/carve.Rd man/CARVE-class.Rd
git commit -m "$(cat <<'EOF'
feat(carve-r): add the CARVE class and carve()

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Selection, labels and accessors on the fit

Files:
- Modify: `carve-r/R/AllGenerics.R` (append)
- Create: `carve-r/R/accessors.R`
- Test: `carve-r/tests/testthat/test-accessors.R`

Interfaces:
- Consumes: the `CARVE` class and `carve()` (Task 14); `coerce_reference_labels` (Task 14); `select_best_row_by_rule`, `select_best_k`, `row_to_estimator_params` (Task 7); `sweep_param_name`, `config_id_of`, `observed_k`, `isclose` (Task 6); `align_cluster_labels`, `count_clusters`, `call_estimator`, `as_data_matrix`, `check_dots`, `format_param_value`, `format_repr` (Tasks 3-4); `resolve_mode` (Task 5); the `consensus_cut` fixture.
- Produces:
  - Exported generics with `CARVE` methods: `get_k(fit, measure = "stability", rule = "1se", not_two = FALSE, ...)` returning an integer; `get_sweep_value(...)` (same arguments) returning a double; `get_estimator(...)` (same arguments) returning `function(X, random_state = NULL)` with attributes `"estimator"` and `"params"`; `get_labels(fit, measure = "stability", rule = "1se", k = NULL, sweep_value = NULL, consensus_k = NULL, not_two = FALSE, mode = "default", estimator = NULL, reference_labels = NULL, ...)` returning integer labels; `estimator_results(fit, ...)`, `estimator_param_grids(fit, ...)`, `sweep_spec(fit, ...)`, `input_data(fit, ...)`; `consensus_matrix(fit, config_id, type = c("stability", "generalizability"), ...)`; `sample_scores(fit, config_id, source = c("gini", "ce", "accuracy"), ...)`; a `show` method for `CARVE`.
  - Internal `select_row(fit, measure, rule, not_two = FALSE, k = NULL, sweep_value = NULL)` returning `list(row, config_id, n_clusters, pinned)`; `config_key(fit, config_id)`; `cut_consensus(consensus_matrix, n_clusters, estimator = NULL)`.

- [ ] Step 1: Write the failing tests in `tests/testthat/test-accessors.R`

```r
blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
fit <- carve(blobs$X, n_clusters = 2:4, n_resamples = 5, estimator_param_grids = k_grid, random_state = 0)

single_mode_fit <- function(mode) {
  suppressWarnings(carve(blobs$X, n_clusters = 2:3, n_resamples = 2, random_state = 0, mode = mode,
                         estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3))))
}

test_that("get_k and get_sweep_value pick k = 3 on three blobs", {
  expect_identical(get_k(fit), 3L)
  expect_identical(get_sweep_value(fit), 3)
  expect_identical(get_k(fit, measure = "generalizability", rule = "max"), 3L)
})

test_that("a measure without standard errors falls back to max with a warning", {
  k <- NULL
  expect_warning(
    k <- get_k(fit, measure = "pac"),
    "Column 'consensus_pac_stability_se' not found; falling back to 'max' rule.",
    fixed = TRUE
  )
  expect_identical(k, get_k(fit, measure = "pac", rule = "max"))
})

test_that("get_labels recovers the blobs, pins k and overrides the cut", {
  labels <- get_labels(fit)
  expect_type(labels, "integer")
  expect_length(labels, 90L)
  expect_identical(adjusted_rand_index(labels, blobs$y), 1)
  expect_identical(count_clusters(get_labels(fit, k = 2)), 2L)
  expect_identical(count_clusters(get_labels(fit, k = 4)), 4L)
  expect_identical(count_clusters(get_labels(fit, sweep_value = 4)), 4L)
  expect_identical(count_clusters(get_labels(fit, consensus_k = 4)), 4L)
})

test_that("get_labels leaves the fit unchanged", {
  before <- fit
  get_labels(fit, k = 2)
  expect_identical(fit, before)
})

test_that("get_labels aligns to reference labels with the same number of clusters", {
  natural <- get_labels(fit, k = 3)
  permuted <- c(2L, 3L, 1L)[natural]
  expect_identical(get_labels(fit, k = 3, reference_labels = permuted), permuted)
  expect_identical(get_labels(fit, k = 2, reference_labels = permuted), get_labels(fit, k = 2))
})

test_that("the reference given to carve() is the default", {
  reference <- c(10L, 20L, 30L)[blobs$y]
  ref_fit <- carve(blobs$X, n_clusters = 2:4, n_resamples = 5, estimator_param_grids = k_grid,
                   random_state = 0, reference_labels = reference)
  expect_identical(get_labels(ref_fit, k = 3), reference)
})

test_that("reference entries of -1 take no part in the matching", {
  natural <- get_labels(fit, k = 3)
  reference <- c(2L, 3L, 1L)[natural]
  reference[c(1, 31, 61)] <- -1L
  expect_identical(get_labels(fit, k = 3, reference_labels = reference), c(2L, 3L, 1L)[natural])
})

test_that("mode = 'generalizability' cuts the held-out consensus", {
  expect_identical(
    get_labels(fit, k = 3, mode = "generalizability"),
    cut_consensus(consensus_matrix(fit, 1, type = "generalizability"), 3L)
  )
})

test_that("a custom estimator cuts the consensus distances", {
  first_rows <- function(D, n_clusters) rep_len(seq_len(n_clusters), nrow(D))
  expect_identical(get_labels(fit, k = 3, estimator = first_rows), rep_len(1:3, 90L))
})

test_that("get_labels reports requests it cannot meet", {
  expect_error(get_labels(fit, k = 7), "No configurations found for k=7.", fixed = TRUE)
  expect_error(get_labels(fit, k = 3, sweep_value = 3), "Pass at most one of k= and sweep_value=.", fixed = TRUE)
  expect_error(
    get_labels(fit, consensus_k = 1),
    "Cannot cut the consensus matrix at 1 cluster(s). The selected configuration is degenerate; pass consensus_k= explicitly or exclude that end of the sweep.",
    fixed = TRUE
  )
  expect_error(get_labels(fit, mode = "both"), "Unknown mode: 'both'.", fixed = TRUE)
  expect_error(get_labels(fit, kk = 3), "Unknown argument: kk.", fixed = TRUE)
})

test_that("k is only valid on the k axis", {
  cut_at <- function(X, resolution = 1) {
    as.integer(stats::cutree(stats::hclust(stats::dist(X)), k = round(2 * resolution)))
  }
  res_fit <- carve(blobs$X, n_resamples = 2, random_state = 0,
                   estimator_param_grids = list(estimator_grid(cut_at, resolution = c(1, 1.5))))
  expect_error(
    get_labels(res_fit, k = 3),
    "This run swept 'resolution', not 'n_clusters'. Use sweep_value=... to pin a resolution, and consensus_k=... to fix the number of clusters used to cut the consensus matrix.",
    fixed = TRUE
  )
  expect_identical(count_clusters(get_labels(res_fit, sweep_value = 1.5)), 3L)
})

test_that("a stability-only fit has no held-out consensus to cut", {
  expect_error(
    get_labels(single_mode_fit("stability"), mode = "generalizability"),
    "Consensus matrix not available for mode='generalizability'. This run likely used split-mode and skipped building that artifact.",
    fixed = TRUE
  )
})

test_that("not_two leaves out the two-cluster configuration", {
  two <- make_blobs(centers = rbind(c(0, 0), c(8, 0)))
  fit2 <- carve(two$X, n_clusters = 2:3, n_resamples = 5, random_state = 0,
                estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3)))
  expect_identical(get_k(fit2), 2L)
  expect_identical(get_k(fit2, not_two = TRUE), 3L)
  expect_identical(count_clusters(get_labels(fit2, not_two = TRUE)), 3L)
  expect_identical(attr(get_estimator(fit2, not_two = TRUE), "params"), list(n_clusters = 3L))
})

test_that("get_estimator rebuilds the selected configuration", {
  selected <- get_estimator(fit)
  expect_identical(attr(selected, "estimator"), "KMeans")
  expect_identical(attr(selected, "params"), list(n_clusters = 3L))
  expect_identical(selected(blobs$X, random_state = 0L), KMeans(blobs$X, n_clusters = 3L, random_state = 0L))
})

test_that("the accessors return the fit's contents", {
  expect_identical(estimator_results(fit), fit@estimator_results)
  expect_identical(estimator_param_grids(fit), fit@estimator_param_grids)
  expect_identical(sweep_spec(fit), fit@sweep)
  expect_identical(input_data(fit), blobs$X)
  expect_identical(consensus_matrix(fit, 1), fit@consensus_matrices[["1"]])
  expect_identical(consensus_matrix(fit, 1, type = "generalizability"), fit@consensus_generalizability_matrices[["1"]])
  expect_identical(sample_scores(fit, 2), fit@stability_gini_scores[["2"]])
  expect_identical(sample_scores(fit, 2, "ce"), fit@stability_ce_scores[["2"]])
  expect_identical(sample_scores(fit, 2, "accuracy"), fit@generalizability_scores[["2"]])
})

test_that("an unknown config_id or source is an error", {
  expect_error(consensus_matrix(fit, 5), "config_id 5 is not in estimator_results(fit).", fixed = TRUE)
  expect_error(sample_scores(fit, 0, "silhouette"), "source must be one of: 'accuracy', 'gini', 'ce'.", fixed = TRUE)
})

test_that("scores and matrices a mode skipped are reported missing", {
  stab <- single_mode_fit("stability")
  expect_error(sample_scores(stab, 0, "accuracy"), "Generalizability scores are not available for this run.", fixed = TRUE)
  expect_error(
    consensus_matrix(stab, 0, "generalizability"),
    "The generalizability consensus matrix is not available for this fit (mode = 'stability').",
    fixed = TRUE
  )
  gen <- single_mode_fit("generalizability")
  expect_error(sample_scores(gen, 0, "gini"), "Gini stability scores are not available for this run.", fixed = TRUE)
  expect_error(sample_scores(gen, 0, "ce"), "CE stability scores are not available for this run.", fixed = TRUE)
})

test_that("cut_consensus matches Python's consensus cut", {
  for (case in read_fixture("consensus_cut")$cases) {
    labels <- cut_consensus(fixture_matrix(case$consensus), case$k)
    expect_identical(adjusted_rand_index(labels, case$labels), 1, info = paste("k =", case$k))
  }
})

test_that("show summarizes the fit", {
  expect_output(show(fit), "CARVE fit on 90 samples and 2 features", fixed = TRUE)
  expect_output(show(fit), "Sweep: n_clusters over 2, 3, 4", fixed = TRUE)
  expect_output(show(fit), "Estimators: KMeans (3 configurations)", fixed = TRUE)
})
```

- [ ] Step 2: Run the tests and see them fail

Run: `Rscript -e 'devtools::test(filter = "accessors", stop_on_failure = TRUE)'`
Expected: FAIL, `could not find function "get_k"`.

- [ ] Step 3: Append the generics to `R/AllGenerics.R`

```r
#' @rdname get_k
#' @export
setGeneric("get_k", function(fit, ...) standardGeneric("get_k"))

#' @rdname get_k
#' @export
setGeneric("get_sweep_value", function(fit, ...) standardGeneric("get_sweep_value"))

#' @rdname get_k
#' @export
setGeneric("get_estimator", function(fit, ...) standardGeneric("get_estimator"))

#' @rdname get_labels
#' @export
setGeneric("get_labels", function(fit, ...) standardGeneric("get_labels"))

#' @rdname estimator_results
#' @export
setGeneric("estimator_results", function(fit, ...) standardGeneric("estimator_results"))

#' @rdname estimator_results
#' @export
setGeneric("estimator_param_grids", function(fit, ...) standardGeneric("estimator_param_grids"))

#' @rdname estimator_results
#' @export
setGeneric("sweep_spec", function(fit, ...) standardGeneric("sweep_spec"))

#' @rdname estimator_results
#' @export
setGeneric("input_data", function(fit, ...) standardGeneric("input_data"))

#' @rdname consensus_matrix
#' @export
setGeneric("consensus_matrix", function(fit, ...) standardGeneric("consensus_matrix"))

#' @rdname consensus_matrix
#' @export
setGeneric("sample_scores", function(fit, ...) standardGeneric("sample_scores"))
```

- [ ] Step 4: Write `R/accessors.R`

```r
#' @include AllGenerics.R
NULL

# Selection, labels and accessors of a fitted CARVE object. Mirrors the
# query methods of api.CARVE.

select_row <- function(fit, measure, rule, not_two = FALSE, k = NULL, sweep_value = NULL) {
  results <- fit@estimator_results
  param <- sweep_param_name(results)
  if (!is.null(k) && !is.null(sweep_value)) {
    stop("Pass at most one of k= and sweep_value=.", call. = FALSE)
  }
  if (!is.null(k) && param != "n_clusters") {
    stop(sprintf(
      "This run swept '%s', not 'n_clusters'. Use sweep_value=... to pin a %s, and consensus_k=... to fix the number of clusters used to cut the consensus matrix.",
      param, param
    ), call. = FALSE)
  }
  pin <- if (!is.null(k)) k else sweep_value
  if (is.null(pin)) {
    row <- select_best_row_by_rule(results, measure, rule, not_two = not_two)
  } else {
    pinned <- results[which(isclose(as.numeric(results$sweep_value), as.numeric(pin))), , drop = FALSE]
    if (nrow(pinned) == 0L) {
      stop(sprintf(
        "No configurations found for %s=%s.",
        if (param == "n_clusters") "k" else param,
        format_param_value(pin)
      ), call. = FALSE)
    }
    row <- select_best_row_by_rule(pinned, measure, rule)
  }
  list(row = row, config_id = config_id_of(row), n_clusters = observed_k(row), pinned = !is.null(pin))
}

config_key <- function(fit, config_id) {
  key <- if (is.numeric(config_id) && length(config_id) == 1L) as.character(as.integer(config_id)) else NA_character_
  if (is.na(key) || !key %in% names(fit@consensus_matrices)) {
    stop(sprintf("config_id %s is not in estimator_results(fit).", format_repr(config_id)), call. = FALSE)
  }
  key
}

# Average-linkage clustering of 1 - consensus. The matrix is symmetrized
# and clipped to [0, 1], and pairs never drawn together count as 0.5.
cut_consensus <- function(consensus_matrix, n_clusters, estimator = NULL) {
  S <- 0.5 * (consensus_matrix + t(consensus_matrix))
  diag(S) <- 1
  S <- pmin(pmax(S, 0), 1)
  if (anyNA(S)) {
    S[is.na(S)] <- 0.5
    diag(S) <- 1
  }
  D <- 1 - S
  diag(D) <- 0
  if (is.null(estimator)) {
    tree <- stats::hclust(stats::as.dist(D), method = "average")
    return(as.integer(stats::cutree(tree, k = n_clusters)))
  }
  args <- list(D)
  if ("n_clusters" %in% names(formals(estimator))) {
    args$n_clusters <- n_clusters
  }
  labels <- do.call(estimator, args)
  if (length(labels) != nrow(D)) {
    stop(sprintf(
      "The estimator must return one label per sample (%d).",
      nrow(D)
    ), call. = FALSE)
  }
  as.integer(labels)
}

#' Select a configuration
#'
#' `get_k()` returns the number of clusters of the configuration a selection
#' rule picks, `get_sweep_value()` the value of the swept parameter there,
#' and `get_estimator()` the estimator with that configuration's parameters
#' filled in.
#'
#' `rule = "max"` takes the configuration with the highest value of
#' `measure`. `"1se"` takes the finest configuration, the one with the highest
#' `sweep_rank`, whose value is within one standard error of the highest.
#' `"quantile"` takes the finest configuration whose value lies between the
#' 5th and 95th percentiles of the resampled values at the best one. Only the
#' three ARI measures have standard errors and percentiles; for the others,
#' `"1se"` and `"quantile"` fall back to `"max"` with a warning.
#'
#' On an axis other than `n_clusters`, the number of clusters is the rounded
#' mean of the counts observed across resamples.
#'
#' @param fit A [CARVE-class] object from [carve()].
#' @param measure The criterion to select on: `"stability"` (or `"s"`),
#'   `"generalizability"` (`"g"`), `"average"` (the mean of the two),
#'   `"pac"`, `"gini"`, `"ce"` or `"accuracy"`, or the name of the matching
#'   column of [estimator_results()], such as `"ari_stability"`.
#' @param rule `"max"`, `"1se"` or `"quantile"`.
#' @param not_two Leave out configurations with two clusters.
#' @param ... Not used. An argument name these functions do not know is an
#'   error.
#' @return `get_k()` returns an integer and `get_sweep_value()` a number.
#'   `get_estimator()` returns a function `(X, random_state = NULL)` that
#'   clusters `X` with the selected configuration; its attributes
#'   `"estimator"` and `"params"` hold the estimator's name and parameters.
#' @seealso [get_labels()], [estimator_results()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)))
#' get_k(fit)
#' get_k(fit, measure = "generalizability", rule = "max")
#' selected <- get_estimator(fit)
#' attr(selected, "params")
#' @export
setMethod("get_k", "CARVE", function(fit, measure = "stability", rule = "1se",
                                     not_two = FALSE, ...) {
  check_dots(...)
  select_best_k(fit@estimator_results, measure, rule, not_two = not_two)
})

#' @rdname get_k
#' @export
setMethod("get_sweep_value", "CARVE", function(fit, measure = "stability", rule = "1se",
                                               not_two = FALSE, ...) {
  check_dots(...)
  as.numeric(select_row(fit, measure, rule, not_two = not_two)$row$sweep_value)
})

#' @rdname get_k
#' @export
setMethod("get_estimator", "CARVE", function(fit, measure = "stability", rule = "1se",
                                             not_two = FALSE, ...) {
  check_dots(...)
  row <- select_best_row_by_rule(fit@estimator_results, measure, rule, not_two = not_two)
  grid <- Find(function(g) identical(g$name, row$estimator[[1L]]), fit@estimator_param_grids)
  estimator <- grid$estimator
  arguments <- setdiff(names(formals(estimator))[-1L], c("random_state", "..."))
  params <- row_to_estimator_params(row, arguments)
  selected <- function(X, random_state = NULL) {
    call_estimator(estimator, as_data_matrix(X), params, random_state = random_state)
  }
  attr(selected, "estimator") <- grid$name
  attr(selected, "params") <- params
  selected
})

#' Consensus cluster labels
#'
#' `get_labels()` selects a configuration, as [get_k()] does, and cuts its
#' consensus matrix into clusters by average-linkage hierarchical clustering
#' of `1 - consensus`, at the configuration's number of clusters. Pairs of
#' samples never drawn together count as 0.5.
#'
#' When the reference labels have as many clusters as the cut, the clusters
#' are renamed to match them by maximum overlap; samples whose reference
#' label is -1 take no part in the matching. `get_labels()` keeps no state
#' between calls, so to keep cluster names stable across calls, pass earlier
#' labels as `reference_labels`.
#'
#' @inheritParams get_k
#' @param k Use a configuration with this number of clusters, the one `rule`
#'   picks among those with that `k`. Only for runs that sweep `n_clusters`.
#' @param sweep_value Use a configuration at this value of the swept
#'   parameter. Give at most one of `k` and `sweep_value`; with either,
#'   `not_two` is not used.
#' @param consensus_k Cut into this many clusters instead.
#' @param mode `"default"` and `"stability"` cut the stability consensus
#'   matrix; `"generalizability"` cuts the consensus of the classifier's
#'   held-out predictions.
#' @param estimator `NULL` for the average-linkage cut, or a function of the
#'   distance matrix `1 - consensus` that returns one label per sample. If it
#'   has an `n_clusters` argument, it gets the number of clusters.
#' @param reference_labels Labels to match, one per sample, coded as in
#'   [carve()]. `NULL` uses the labels given to [carve()], if any.
#' @return An integer vector of cluster labels, one per sample.
#' @seealso [get_k()], [consensus_matrix()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)))
#' labels <- get_labels(fit)
#' table(labels)
#' table(get_labels(fit, k = 3, reference_labels = labels))
#' @export
setMethod("get_labels", "CARVE", function(fit, measure = "stability", rule = "1se", k = NULL,
                                          sweep_value = NULL, consensus_k = NULL,
                                          not_two = FALSE, mode = "default", estimator = NULL,
                                          reference_labels = NULL, ...) {
  check_dots(...)
  policy <- resolve_mode(mode)
  selected <- select_row(fit, measure, rule, not_two = not_two, k = k, sweep_value = sweep_value)
  # Off the k axis the cluster count is an outcome, so the consensus is cut
  # at the count observed.
  cut_k <- if (is.null(consensus_k)) selected$n_clusters else as.integer(consensus_k)
  if (cut_k < 2L) {
    stop(sprintf(
      "Cannot cut the consensus matrix at %d cluster(s). The selected configuration is degenerate; pass consensus_k= explicitly or exclude that end of the sweep.",
      cut_k
    ), call. = FALSE)
  }
  key <- as.character(selected$config_id)
  consensus <- if (policy$run_stability) {
    fit@consensus_matrices[[key]]
  } else {
    fit@consensus_generalizability_matrices[[key]]
  }
  if (is.null(consensus)) {
    stop(sprintf(
      "Consensus matrix not available for mode='%s'. This run likely used split-mode and skipped building that artifact.",
      policy$mode
    ), call. = FALSE)
  }
  labels <- cut_consensus(consensus, cut_k, estimator)
  reference <- if (is.null(reference_labels)) {
    fit@reference_labels
  } else {
    coerce_reference_labels(reference_labels, length(labels))
  }
  if (!is.null(reference) && count_clusters(reference) == length(unique(labels))) {
    labels <- align_cluster_labels(reference, labels, keep = reference >= 0)
  }
  as.integer(labels)
})

#' Contents of a CARVE fit
#'
#' `estimator_results()` returns the results table, one row per
#' configuration. `estimator_param_grids()` returns the estimator grids the
#' run evaluated, `sweep_spec()` the run's [SweepSpec-class] object, and
#' `input_data()` the data matrix.
#'
#' The columns of the results table:
#'
#' - `config_id` joins a row to its consensus matrices and per-sample
#'   scores. It counts from 0.
#' - `method_id` and `method_label` name the curve a row belongs to: an
#'   estimator with all its parameters except the swept one.
#' - `estimator` is followed by one column per estimator parameter, `NA`
#'   where an estimator does not have the parameter.
#' - `sweep_param`, `sweep_value` and `sweep_rank` place the row on the
#'   sweep axis; rank 0 is the coarsest.
#' - `n_clusters_observed` is the mean number of clusters over the
#'   resamples, with standard error `n_clusters_observed_se`, and
#'   `noise_fraction` the mean share of samples labeled noise.
#' - `ari_stability`, `ari_generalizability` and `ari_average` are means
#'   over the resamples. Each comes with `_se` (its standard error), `_upper`
#'   (the 95th percentile) and `_lower` (the 5th percentile).
#' - `consensus_pac_stability` is one minus the share of consensus values
#'   strictly between 0.05 and 0.95. `consensus_gini_stability` and
#'   `consensus_ce_stability` are the means of the per-sample stability
#'   scores, and `accuracy_generalizability` the mean held-out accuracy.
#'
#' @inheritParams get_k
#' @return `estimator_results()` returns a data frame,
#'   `estimator_param_grids()` a list of [estimator_grid()] objects,
#'   `sweep_spec()` a [SweepSpec-class] object and `input_data()` a matrix.
#' @seealso [consensus_matrix()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)))
#' estimator_results(fit)[, c("config_id", "n_clusters", "ari_stability", "ari_stability_se")]
#' sweep_spec(fit)
#' @export
setMethod("estimator_results", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@estimator_results
})

#' @rdname estimator_results
#' @export
setMethod("estimator_param_grids", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@estimator_param_grids
})

#' @rdname estimator_results
#' @export
setMethod("sweep_spec", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@sweep
})

#' @rdname estimator_results
#' @export
setMethod("input_data", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@input_data
})

#' Consensus matrices and per-sample scores
#'
#' `consensus_matrix()` returns a configuration's consensus matrix. Entry
#' `(i, j)` is the share of the subsamples drawing both samples in which the
#' two fell in the same cluster, and `NaN` when no subsample drew both.
#' `sample_scores()` returns a configuration's per-sample scores.
#'
#' @inheritParams get_k
#' @param config_id The configuration, a value of the `config_id` column of
#'   [estimator_results()].
#' @param type `"stability"` for the clusterings of the subsamples, or
#'   `"generalizability"` for the classifier's predictions on the held-out
#'   samples.
#' @param source `"gini"` or `"ce"` for per-sample stability, in its Gini or
#'   cross-entropy form; both run from 0 to 1, and 1 means every pair the
#'   sample was drawn with always agreed. `"accuracy"` is the share of the
#'   draws holding the sample out in which the classifier predicted its
#'   cluster.
#' @return `consensus_matrix()` returns an `n` by `n` matrix and
#'   `sample_scores()` a vector of length `n`. Stability scores are `NaN` for
#'   a sample never drawn with a partner; accuracy is 0 for a sample never
#'   held out.
#' @seealso [estimator_results()], [get_labels()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)))
#' M <- consensus_matrix(fit, config_id = 0)
#' dim(M)
#' summary(sample_scores(fit, config_id = 0, source = "gini"))
#' @export
setMethod("consensus_matrix", "CARVE", function(fit, config_id,
                                                type = c("stability", "generalizability"), ...) {
  check_dots(...)
  type <- match.arg(type)
  key <- config_key(fit, config_id)
  consensus <- if (type == "stability") {
    fit@consensus_matrices[[key]]
  } else {
    fit@consensus_generalizability_matrices[[key]]
  }
  if (is.null(consensus)) {
    stop(sprintf(
      "The %s consensus matrix is not available for this fit (mode = '%s').",
      type, fit@run_params$mode
    ), call. = FALSE)
  }
  consensus
})

#' @rdname consensus_matrix
#' @export
setMethod("sample_scores", "CARVE", function(fit, config_id, source = c("gini", "ce", "accuracy"),
                                             ...) {
  check_dots(...)
  if (identical(source, c("gini", "ce", "accuracy"))) {
    source <- "gini"
  }
  if (!is.character(source) || length(source) != 1L || !source %in% c("gini", "ce", "accuracy")) {
    stop("source must be one of: 'accuracy', 'gini', 'ce'.", call. = FALSE)
  }
  key <- config_key(fit, config_id)
  if (source == "gini") {
    if (is.null(fit@stability_gini_scores)) {
      stop("Gini stability scores are not available for this run.", call. = FALSE)
    }
    return(fit@stability_gini_scores[[key]])
  }
  if (source == "ce") {
    if (is.null(fit@stability_ce_scores)) {
      stop("CE stability scores are not available for this run.", call. = FALSE)
    }
    return(fit@stability_ce_scores[[key]])
  }
  scores <- fit@generalizability_scores[[key]]
  if (is.null(scores)) {
    stop("Generalizability scores are not available for this run.", call. = FALSE)
  }
  scores
})

#' @rdname CARVE-class
#' @param object A `CARVE` object.
#' @export
setMethod("show", "CARVE", function(object) {
  results <- object@estimator_results
  settings <- object@run_params
  sweep <- object@sweep
  cat("CARVE fit on ", nrow(object@input_data), " samples and ", ncol(object@input_data),
      " features\n", sep = "")
  cat("Sweep: ", sweep@param, " over ",
      paste(vapply(sweep@values, format_param_value, character(1)), collapse = ", "), "\n", sep = "")
  cat("Estimators: ", paste(unique(results$estimator), collapse = ", "), " (", nrow(results),
      " configurations)\n", sep = "")
  cat("Resamples: ", settings$n_resamples, " per configuration, subsample_ratio ",
      format(settings$subsample_ratio), "\n", sep = "")
  cat("Mode: ", settings$mode, "\n", sep = "")
  invisible(object)
})
```

- [ ] Step 5: Run the tests and see them pass

Run: `Rscript -e 'devtools::document(); devtools::test(filter = "accessors", stop_on_failure = TRUE)'`
Expected: `man/get_k.Rd`, `man/get_labels.Rd`, `man/estimator_results.Rd`, `man/consensus_matrix.Rd` written and `man/CARVE-class.Rd` updated; tests PASS.

If "get_k and get_sweep_value pick k = 3 on three blobs" or "not_two leaves out the two-cluster configuration" fails on the selected k, print `estimator_results(fit)[, c("n_clusters", "ari_stability", "ari_stability_se")]` and report it rather than changing the data or the seed: these tests state what the method should find on clearly separated blobs.

- [ ] Step 6: Mutation check

- In `get_labels()`, align without `keep` (`align_cluster_labels(reference, labels)`) and count the reference with `length(unique(reference))`: "reference entries of -1 take no part in the matching" fails.
- In `cut_consensus()`, replace NaN with 0 instead of 0.5: the `consensus_cut` fixture test fails for at least one k, or report if it does not.
- In `select_row()`, ignore `not_two` (pass `FALSE`): the `not_two` test fails.
- In `consensus_matrix()`, index by position (`fit@consensus_matrices[[config_id]]`, so id 1 reads the first matrix, which belongs to id 0): "the accessors return the fit's contents" fails.

- [ ] Step 7: Run the whole suite

Run: `Rscript -e 'devtools::test(stop_on_failure = TRUE)'`
Expected: every file passes, no warnings. Note the elapsed time in the report.

- [ ] Step 8: Commit

```bash
git add R/AllGenerics.R R/accessors.R tests/testthat/test-accessors.R NAMESPACE DESCRIPTION man/get_k.Rd man/get_labels.Rd man/estimator_results.Rd man/consensus_matrix.Rd man/CARVE-class.Rd
git commit -m "$(cat <<'EOF'
feat(carve-r): add get_k, get_labels, get_estimator and the accessors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Parity with Python and the package check

Files:
- Create: `carve-r/data-raw/parity_check.R`
- Create (generated): `carve-r/data-raw/parity-report.md`

Interfaces:
- Consumes: the whole package; `reticulate` and `devtools` from the R library; the Python package in `code/.venv`.
- Produces: the parity report, and a clean `R CMD check`. This task changes no package code unless the check or the report finds a defect; such a fix goes in its own commit with a test that fails without it.

- [ ] Step 1: Write `carve-r/data-raw/parity_check.R`

```r
# Compares the R package with the Python package on the same data and
# writes data-raw/parity-report.md. Run from code/carve-r:
#
#   Rscript data-raw/parity_check.R
#
# It loads the package from source with devtools, and needs reticulate and
# the Python environment in ../.venv, where carve is installed. The two
# languages draw different subsamples, so the results agree within
# tolerances, not exactly.

devtools::load_all(".", quiet = TRUE)
reticulate::use_python(normalizePath("../.venv/bin/python"), required = TRUE)
datasets <- reticulate::import("sklearn.datasets")
np <- reticulate::import("numpy", convert = FALSE)
pycarve <- reticulate::import("carve")

ks <- 2:6
n_resamples <- 100L

# Largest allowed absolute difference between the R and the Python value
# of one configuration.
tolerance <- c(
  ari_stability = 0.05,
  ari_generalizability = 0.05,
  ari_average = 0.05,
  consensus_pac_stability = 0.03,
  consensus_gini_stability = 0.03,
  consensus_ce_stability = 0.03,
  accuracy_generalizability = 0.03
)

cases <- list(
  "easy blobs" = datasets$make_blobs(
    n_samples = 500L, n_features = 2L, centers = 3L, cluster_std = 0.8, random_state = 3L
  ),
  "hard blobs" = datasets$make_blobs(
    n_samples = 500L, n_features = 2L, centers = 3L, cluster_std = 2.3, random_state = 3L
  ),
  "circles" = datasets$make_circles(n_samples = 400L, factor = 0.5, noise = 0.05, random_state = 0L)
)

compare_case <- function(name, data) {
  X <- data[[1L]]
  r_fit <- carve(X, n_clusters = ks, n_resamples = n_resamples, random_state = 0)
  py_fit <- pycarve$CARVE(n_clusters = np$array(ks), n_resamples = n_resamples, random_state = 0L)$fit(X)
  columns <- c("method_label", "n_clusters", names(tolerance))
  joined <- merge(
    estimator_results(r_fit)[, columns],
    py_fit$estimator_results_[, columns],
    by = c("method_label", "n_clusters"),
    suffixes = c("_r", "_py")
  )
  differences <- vapply(names(tolerance), function(column) {
    max(abs(joined[[paste0(column, "_r")]] - joined[[paste0(column, "_py")]]))
  }, numeric(1))
  selections <- expand.grid(
    measure = c("stability", "generalizability", "average"),
    rule = c("max", "1se"),
    stringsAsFactors = FALSE
  )
  selections$k_r <- mapply(function(m, r) get_k(r_fit, measure = m, rule = r), selections$measure, selections$rule)
  selections$k_py <- mapply(
    function(m, r) as.integer(py_fit$get_k(measure = m, rule = r)),
    selections$measure, selections$rule
  )
  list(
    name = name,
    rows = nrow(joined),
    differences = differences,
    selections = selections,
    label_ari = adjusted_rand_index(get_labels(r_fit), as.integer(py_fit$get_labels()))
  )
}

format_case <- function(result) {
  over <- result$differences > tolerance
  c(
    paste("##", result$name),
    "",
    sprintf("Configurations compared: %d of %d.", result$rows, 3L * length(ks)),
    "",
    "| Column | Largest difference | Tolerance |",
    "|---|---|---|",
    sprintf(
      "| %s | %.3f%s | %.2f |",
      names(tolerance), result$differences, ifelse(over, " (over)", ""), tolerance
    ),
    "",
    "| Measure | Rule | k in R | k in Python |",
    "|---|---|---|---|",
    sprintf(
      "| %s | %s | %d | %d |",
      result$selections$measure, result$selections$rule,
      result$selections$k_r, result$selections$k_py
    ),
    "",
    sprintf(
      "ARI between the R and Python labels at the default selection (stability, 1se): %.3f",
      result$label_ari
    ),
    ""
  )
}

results <- Map(compare_case, names(cases), cases)
report <- c(
  "# R and Python parity report",
  "",
  sprintf(
    "Generated on %s with CARVE %s in R and carve %s in Python: the light preset, k from %d to %d, %d resamples, random_state 0.",
    format(Sys.Date()), as.character(utils::packageVersion("CARVE")), pycarve$`__version__`,
    min(ks), max(ks), n_resamples
  ),
  "",
  unlist(lapply(results, format_case))
)
writeLines(report, "data-raw/parity-report.md")
writeLines(report)
```

- [ ] Step 2: Run the parity check

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript data-raw/parity_check.R
```

Expected: the report prints and is written to `data-raw/parity-report.md`, with 15 of 15 configurations compared in each case and no "(over)" marks. It takes several minutes.

If a case compares fewer than 15 configurations, the method labels differ between the languages; fix the label code. If a difference is over its tolerance, do not raise the tolerance: compare the per-configuration rows of that column, find which estimator and k differ, and report it to the author before going on. A selected k that differs between the languages is acceptable only when the curves agree within the tolerances; list each such case in the task report.

- [ ] Step 3: Check the writing in the package sources

```bash
grep -n -i -E "—|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|it's important|in conclusion" R/*.R data-raw/parity_check.R || echo "no hits"
Rscript -e 'for (f in c(list.files("R", full.names = TRUE), list.files("tests/testthat", pattern = "[.]R$", full.names = TRUE))) tools::showNonASCIIfile(f)'
```

Expected: `no hits`, and no lines from `showNonASCIIfile`. Fix any hit by rewriting the sentence (load the `de-ai-writing` skill first), not by swapping a word.

- [ ] Step 4: Run R CMD check

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript -e 'devtools::document()'
R CMD build .
_R_CHECK_FORCE_SUGGESTS_=false R CMD check --no-manual CARVE_2.0.0.tar.gz
```

Expected: `Status: OK`, or NOTEs only. A WARNING or an ERROR fails the stage, as in CI (`error-on: '"warning"'`). Record any NOTE in the task report. Then remove the build output, which is not tracked:

```bash
rm -rf CARVE.Rcheck CARVE_2.0.0.tar.gz
git status --short
```

Expected: only `data-raw/parity_check.R` and `data-raw/parity-report.md` untracked, plus any `man/` or `NAMESPACE` change from `document()` (there should be none).

- [ ] Step 5: Commit

```bash
git add data-raw/parity_check.R data-raw/parity-report.md
git commit -m "$(cat <<'EOF'
test(carve-r): compare the k-axis engine with Python and record the report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

## After stage 1

The package can fit and query the k axis. The next plan (stage 2, written in its own conversation) adds the `resolution` and `min_cluster_size` sweeps with `LeidenClustering`, `LouvainClustering` and `HDBSCAN`, the noise policy and noise labels, anchored consensus and randomized preprocessing, and extends `make_fixtures.py` and the parity script accordingly. The branch is merged into `main` only after stage 4.
