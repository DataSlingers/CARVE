# R Package Rework, Stage 2 (Other Axes and Scale) Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: extend the stage 1 package so that it sweeps `resolution` and `min_cluster_size` with `LeidenClustering`, `LouvainClustering` and `HDBSCAN`, resolves HDBSCAN noise with Python's three noise policies, labels ambiguous samples as noise in `get_labels()`, computes anchored consensus above 5,000 samples, and runs randomized preprocessing with `preprocessing_results()`, all with the Python package's behavior and messages.

Architecture: the stage 1 structure stays: one R file per Python module, tests one to one with the R files, deterministic pieces checked against JSON fixtures written from the Python functions, stochastic pieces checked by behavior. Two files are new: `R/pipeline.R` (the port of `_pipeline.py`) and `R/transforms.R` (the preprocessing transforms, R stand-ins for the scikit-learn and umap-learn classes). The runner gains the noise policy, anchor blocks and a precompute pass for embeddings; the fit object gains three slots.

Tech stack: R 4.5.1 (DESCRIPTION floor 4.1.0), methods (S4), BiocParallel, igraph (Leiden and Louvain), dbscan (HDBSCAN, suggested), irlba and Rtsne (PCA and t-SNE), uwot (UMAP, suggested), Matrix, ranger, withr, testthat 3e, roxygen2 8.0.0 with markdown, devtools; Python 3.12 in `code/.venv` (carve 1.0.0, scikit-learn 1.7.2, igraph 1.0.0, leidenalg 0.12.0, umap-learn 0.5.12) for the fixture and parity scripts.

Spec: `docs/superpowers/specs/2026-10-06-r-package-rework-design.md` (commit 0b8d5b4). This plan implements stage 2 of its four stages and builds on stage 1 as executed (`docs/superpowers/plans/2026-10-06-r-package-rework-stage1.md`, commits 17cac7e..3c6abd6). Executors read the spec and this plan.

## Global Constraints

- Branch: `r-package-rework` in the `code/` repository, which is at 3c6abd6 (stage 1 complete) when this plan starts. Commit after every task. Stage files by explicit path only; never `git add -A`, `git add .` or `git commit -a`.
- Run R commands from `code/carve-r/`. Run the fixture script from `code/`. Commands use the R on PATH (4.5.1) and the Python in `code/.venv`.
- Do not touch anything outside `carve-r/` except this plan's own commit; in particular not `src/`, `tests/`, `notebooks/` or `../overleaf/`.
- Dependencies added in this stage, each in the task that first uses it: Imports `igraph` (Task 4), `irlba` and `Rtsne` (Task 7); Suggests `dbscan` (Task 5) and `uwot` (Task 7). An import declared before it is used gives an `R CMD check` NOTE. Keep both fields in the case-insensitive alphabetical order stage 1 used.
- Exports added in this stage: `LeidenClustering`, `LouvainClustering`, `HDBSCAN`, `Identity`, `StandardScaler`, `Log1p`, `PCA`, `TSNE`, `UMAP`, `preprocessing_option`, `pipeline_from_spec`, `consensus_anchors`, `preprocessing_results`, `preprocessing_pipelines`, and the S3 methods `print.carve_preprocessing_option` and `print.carve_pipeline_spec`. Nothing else; tests reach internals through the namespace `devtools::test()` loads.
- Defaults copied from Python: `noise_policy = "drop"`, `anchor_threshold = 5000`, `consensus_anchors = NULL`, `randomize_preprocessing = FALSE`; `LeidenClustering(resolution = 1, n_neighbors = 15L, weighting = "connectivity", objective_function = "modularity", n_iterations = -1L, random_state = NULL, scale = FALSE)`; `LouvainClustering(resolution = 1, n_neighbors = 15L, weighting = "connectivity", scale = FALSE, random_state = NULL)`; `HDBSCAN(min_cluster_size = 5L, cluster_selection_method = "eom")`; `get_labels(noise_labels = FALSE, noise_quantile = 0.05, noise_score = "gini")`; `NOISE_MARGIN = 0.05`; `PCA(n_components = NULL)`; `TSNE(n_components = 2L, perplexity = 30)`; `UMAP(n_components = 2L, n_neighbors = 15L, min_dist = 0.1)`.
- Seeds are derived, never shared. Stage 1's rule stands: resample `b` (0-based) draws its first subsample with `random_state + b`, its second with `random_state + b + n_resamples`, and fits its estimators and classifier under `random_state + b`. New in this stage: under randomized preprocessing, resample `b` fits its pipeline on the first subsample with `random_state + b`, on the second with `random_state + b + n_resamples` and on the held-out samples with `random_state + b + 2 * n_resamples`. The anchor draw and the pipeline allocation use `seeded(random_state, ..., kind = "L'Ecuyer-CMRG")`. All seeding goes through `seeded()`.
- `config_id` is a join key: 0-based, as in Python, the names of every per-configuration list, never a position. Anchors are sorted 1-based integer row indices.
- Labels: built-in estimators return 1-based integer labels; -1 marks noise; a label is a cluster when it is non-negative. `get_labels(noise_labels = TRUE)` marks noise with `-1L`.
- Diagnostics use `warning(..., call. = FALSE)`; verbose output uses `message()` gated on `verbose`. No logging package, no `print()` or `cat()` outside `show` and `print` methods and the progress bars.
- Error and warning text copies Python's wording where Python has the message; the exact strings are in the tasks. Two copied warnings keep Python's British "labelled" (the all-noise warnings of Task 10); that is deliberate.
- Unexpected warnings fail the suite (`tests/testthat/setup.R` sets `warn = 2`). A test that expects warnings captures all of them, with `expect_warning` (first match only, the rest still fail), `collect_warnings()` or `suppressWarnings()`.
- Suggested packages: tests that need dbscan or uwot start with `skip_if_not_installed()`. Both are installed here (dbscan 1.2.7, installed on 2026-10-06 while this plan was written; uwot 0.2.4), so every task report shows `SKIP 0`. A skip here means the environment changed; stop and report it.
- Assertions must be able to fail. After a task's tests pass, apply the mutations its last step names, one at a time, confirm the named test fails, and restore the code with `git checkout -- <file>`. Report each result.
- Writing: every roxygen text and code comment follows the spec's writing standard. The roxygen text in this plan is written to it; keep it as written. If you add or change any user-facing text, load the `de-ai-writing` skill first. American spelling, plain prose, no bold or italics, sentence-case Rd titles.
- No non-ASCII characters in `R/` or `tests/` files (an `R CMD check` warning).
- Stage 1's rulings still apply: commit DESCRIPTION as `devtools::document()` leaves it (RoxygenNote); every task that adds a file under `R/` runs `devtools::document()` before committing and stages DESCRIPTION, NAMESPACE and any changed `man/` files (the Collate field); a generic declared with `@rdname X` in `AllGenerics.R` and its method documented elsewhere both carry `@rdname X`, so each topic is one page.
- The plan's code was written without being run, except for the HDBSCAN and igraph checks recorded under Decisions. Treat it as a detailed proposal. When plan code or a plan test is wrong, fix it, say so in the report with the failing output and why, and keep the assertion's intent. Never loosen an assertion to make it pass against code that disagrees with it.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Decisions this plan makes that the spec left open

Items 1, 5, 6 and 8 change behavior a Python user would notice and are flagged to the author with the plan.

1. HDBSCAN leaf selection (spec open item 1). It can be computed from the cluster hierarchy that `dbscan::hdbscan()` stores in its `"hdbscan"` attribute: each cluster's `contains` lists the samples that leave it, and the `cl_hierarchy` attribute maps each parent to its two children, so the leaves are the clusters without children and a leaf's `contains` is its membership. The full preset keeps `leaf`. Measured while writing this plan (dbscan 1.2.7, scikit-learn 1.7.2, 25 data sets of three overlapping blobs, `min_cluster_size` 5 and 10): dbscan's eom selection matched scikit-learn's exactly in 46 of 50 cases, and the leaf extraction in 25 of 50. Running scikit-learn's own extraction (`tree_to_labels`) on dbscan's trees gave the same 46 and 25, and the merge heights of the two single-linkage trees were identical, so the extraction is equivalent and the gaps come from the order in which the two packages merge tied mutual reachability distances. Porting scikit-learn's extraction would not close them. The `HDBSCAN` help page says so, and stage 4's `python-users.Rmd` repeats it. The Task 1 fixture uses data on which both selections match exactly.
2. `HDBSCAN()` takes `min_cluster_size` and `cluster_selection_method` only. dbscan's `minPts` sets both the minimum cluster size and the core-distance neighbor count, which is scikit-learn's behavior when `min_samples` is left at its default; the check above confirmed that dbscan's core distances equal scikit-learn's.
3. The graph estimators have no `metric` argument: the neighbor search is Euclidean (FNN), scikit-learn's default. The kNN graph links each sample to its `k = min(n_neighbors, n - 1)` nearest other samples, as `kneighbors_graph(include_self = False)` does. This differs from stage 1's `nearest_neighbors()`, which counts the sample itself because the spectral code follows `NearestNeighbors`.
4. Leiden runs `igraph::cluster_leiden()`. Python's `"modularity"` maps to igraph's `"modularity"`, which sets the vertex weights to the degrees and divides the resolution by twice the total edge weight, the RB configuration model leidenalg optimizes; `"cpm"` maps to igraph's `"CPM"`. igraph's `beta` stays at its default (0.01). `n_iterations = -1` iterates until the quality stops improving, in igraph as in leidenalg. Louvain runs `igraph::cluster_louvain()`. Both draw from R's generator, so `seeded(random_state, ...)` makes them reproducible (checked while writing this plan).
5. The anchor draw and the pipeline allocation use a L'Ecuyer-CMRG stream seeded with `random_state`. Python draws both from a numpy `default_rng`, a different generator from the `RandomState` that draws the subsamples. A Mersenne-Twister draw in R with the same seed would make the anchors the first `m` samples of resample 0's first subsample, because R's `sample.int()` draws without replacement as prefixes of one another.
6. `consensus_anchors`: a number in (0, 1] is a share of the samples and a whole number above 1 is a count. Python tells the two apart by type (float or int); R numbers are doubles, so `consensus_anchors = 1` means every sample (the exact path) where Python's integer 1 is an error. Anything else gets Python's fraction message.
7. Anchored stability is computed from sparse indicator matrices (Matrix) in row chunks of Python's default size (`max(256, min(8192, 2^23 %/% m))`), so the `n` by `m` slab never exists in full. Anchor blocks are doubles, like every consensus matrix in R.
8. Core budget and the anchored `get_labels()` (both stage 1 carries). With a `BPPARAM`, the classifier threads per worker are the cores divided by `min(workers, n_resamples)`, since only that many workers can be busy. The fit stores `run_params$n_threads = min(n_cores(), outer * inner)`, and the classifier that extends an anchored cut gets that many threads: the whole budget, as in Python, also when a `BPPARAM` set the worker count. The extension runs under `seeded(random_state)`, as each resample does.
9. Workers receive one task per resample, `list(b, embeddings)`, so a worker gets only its own resample's embeddings, as in Python.
10. A randomized run's pipeline records keep each resample's pipeline label, its two ARIs and its cluster count. Python keeps the whole per-resample result, labels and indices included; the summary uses only these four.
11. `R/transforms.R` is a file with no Python module of its own: it holds the R stand-ins for scikit-learn's `FunctionTransformer`, `StandardScaler`, `PCA`, `TSNE` and umap-learn's `UMAP`, as `estimators.R` holds the stand-ins for the scikit-learn clusterers. `R/pipeline.R` mirrors `_pipeline.py`.
12. `TSNE()` passes Rtsne scikit-learn's defaults: no PCA preprocessing and no normalization of the input, initialization from the first principal components scaled to a standard deviation of 1e-4 in the first coordinate, early exaggeration 12 for 250 iterations (`stop_lying_iter` and `mom_switch_iter` must be set, because Rtsne sets them to 0 when `Y_init` is given), 1,000 iterations, and the learning rate of `learning_rate = "auto"`, which is `max(n / 48, 50)` in scikit-learn's units and `max(n / 12, 200)` in Rtsne's (Rtsne's `eta` is four times scikit-learn's learning rate; the Cusanovich case study found and fixed the same factor on 2026-10-03). Rtsne needs `3 * perplexity <= n - 1`, stricter than scikit-learn's `perplexity < n`, so the default perplexity filter uses Rtsne's limit and its warning prints `floor((n_min - 1) / 3) + 1` as the limit.
13. `UMAP()` passes uwot umap-learn's defaults where they differ from uwot's (`min_dist = 0.1`; uwot's own is 0.01), one thread for the neighbor search and the optimization (`n_threads = 1L`, `n_sgd_threads = 0L`) so a seed reproduces the embedding, and `verbose = FALSE` (uwot logs by default). Without uwot, the default options omit UMAP with Python's warning and an R install hint.
14. `PCA()` uses `irlba::prcomp_irlba()` when it keeps fewer than half of `min(dim(X))` components and `stats::prcomp()` otherwise; irlba warns and is slower above that share.
15. `noise_policy` is validated at fit time with Python's message; Python checks it only inside the first resample.
16. `random_state` must be at most `.Machine$integer.max - 3 * n_resamples`, since the held-out embedding seed reaches `random_state + 3 * n_resamples - 1`. Stage 1's limit was `2 * n_resamples`.
17. Console output: the header prints `randomize_preproc` as `TRUE` or `FALSE`, the option names of a randomized run, a `consensus anchors` line, and the projected consensus memory over the anchors when the run is anchored. `show()` adds a consensus line and, for randomized runs, a preprocessing line.
18. The precompute pass shows its own progress bar before the configuration bar when `show_progress = TRUE`. It reaches 100% when the pass ends; with parallel workers it does not advance in between.
19. `align_cluster_labels()` with a `keep` that selects nothing returns the labels unchanged (stage 1 carry).
20. The parity script adds a resolution sweep on synthetic 10-dimensional blobs. The spec's PBMC 3k case needs `pbmc3k_subset`, which stage 3 builds; stage 3 adds it to the parity script.
21. `get_labels()` takes Python's arguments in Python's order, followed by stage 1's `reference_labels`.

Stage 1 leftovers this plan does not touch: `format_repr()` prints whole doubles without ".0"; R keeps whole-number double reference labels; README and NEWS still describe the R6 package until stage 4; a stale CARVE 1.0.0 in the system R library is what SnowParam workers load under `devtools::load_all()`, so no test here uses a SnowParam to run resamples.

## File structure

```
carve-r/
  DESCRIPTION                      Imports igraph (Task 4), irlba, Rtsne (Task 7);
                                   Suggests dbscan (Task 5), uwot (Task 7)
  R/utils.R                        + seeded(kind =), has_package, require_package,
                                   noise policy, noise_mask, resolve_anchors,
                                   summarize_preprocessing_records (Tasks 2, 3, 12)
  R/consensus.R                    + anchored consensus (Task 3)
  R/estimators.R                   + kNN graph, LeidenClustering, LouvainClustering
                                   (Task 4), HDBSCAN (Task 5)
  R/grids.R                        + resolution and min_cluster_size presets (Task 6),
                                   default preprocessing options (Task 9)
  R/transforms.R                   new: Identity, StandardScaler, Log1p, PCA, TSNE,
                                   UMAP (Task 7)
  R/pipeline.R                     new: _pipeline.py (Task 8)
  R/runner.R                       validation_iter with noise and embeddings (Task 10);
                                   run_validation with anchors, noise policy and the
                                   core budget (Task 11), randomized runs (Task 12)
  R/output.R                       header lines for anchors and preprocessing (Task 13)
  R/AllClasses.R                   three new CARVE slots (Task 14)
  R/carve.R                        the full carve() signature (Task 14)
  R/AllGenerics.R                  consensus_anchors, preprocessing_results,
                                   preprocessing_pipelines (Task 15)
  R/accessors.R                    show() lines (Task 14); new accessors, anchored and
                                   noise labels in get_labels() (Task 15)
  tests/testthat/helper-fixtures.R + same_partition() (Task 1)
  tests/testthat/test-*.R          one per R file; test-transforms.R and
                                   test-pipeline.R are new
  tests/testthat/fixtures/         + anchored, noise_mask, knn_graph, hdbscan,
                                   pipeline_labels (Task 1)
  data-raw/make_fixtures.py        + five fixture writers (Task 1)
  data-raw/parity_check.R          + resolution, HDBSCAN, anchored and randomized
                                   cases (Task 16)
  data-raw/parity-report.md        regenerated (Task 16)
```

---

### Task 1: Python fixtures for stage 2

Files:
- Modify: `carve-r/data-raw/make_fixtures.py`
- Create (generated): `carve-r/tests/testthat/fixtures/anchored.json`, `noise_mask.json`, `knn_graph.json`, `hdbscan.json`, `pipeline_labels.json`
- Modify: `carve-r/tests/testthat/helper-fixtures.R`

Interfaces:
- Consumes: the Python package in `code/.venv`.
- Produces (fixture shapes, as `read_fixture()` returns them with `simplifyDataFrame = FALSE`; indices and labels are 0-based):
  - `anchored`: `n` (15); `runs`, a list of `list(indices, labels)`, six runs of 9 unsorted indices, sample 14 never drawn; `anchors` (6 values, 14 among them); `block` (6 by 6 matrix, `NA` for never co-sampled pairs); `gini`, `ce` (length 15, `NA` where Python has NaN).
  - `noise_mask`: `cases`, each `list(name, scores, quantile, mask, n_target, cutoff, median)`; scores may hold `NA` (Python NaN).
  - `knn_graph`: `X` (25 by 2); `cases`, each `list(weighting, n_neighbors, edges, weights)`, `edges` an integer matrix with two columns.
  - `hdbscan`: `X` (120 by 2); `labels`, a named list `eom_5`, `leaf_5`, `eom_10`, `leaf_10` of integer vectors with -1 for noise.
  - `pipeline_labels`: `steps`, each `list(name, params, label)` with `params` a named list in Python's key order; `spec`, `list(label)`.
  - Test helper `same_partition(a, b)`: `TRUE` when `a` and `b` put -1 on the same samples and group the others identically, whatever the label values.

- [ ] Step 1: Check the environment

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
Rscript -e 'cat(as.character(packageVersion("dbscan")), as.character(packageVersion("uwot")), as.character(packageVersion("igraph")), "\n")'
.venv/bin/python -c "import igraph, leidenalg, sklearn; print(sklearn.__version__)"
git status --short
git log --oneline -1
```

Expected: `1.2.7 0.2.4 2.3.2`, `1.7.2`, a clean tree, and HEAD at 3c6abd6. If dbscan is missing, install it with `Rscript -e 'install.packages("dbscan", repos = "https://cloud.r-project.org")'` and say so in the report.

- [ ] Step 2: Add the imports to `data-raw/make_fixtures.py`

Change the scikit-learn cluster import and add four imports beside the existing `carve` ones:

```python
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
```

```python
from carve._consensus import (
    compute_consensus_matrix,
    compute_consensus_pac,
    consensus_anchor_block,
    stability_from_consensus,
    stability_from_runs_anchored,
)
from carve._pipeline import PipelineSpec, PipelineStep
from carve._utils import noise_mask
from carve.cluster import build_knn_graph
```

Keep the existing `from carve._utils import (...)` block; `noise_mask` may join it instead of a separate line.

- [ ] Step 3: Add the five fixture writers before `if __name__ == "__main__":`

```python
def anchored():
    rng = np.random.default_rng(1)
    n = 15
    runs = []
    for _ in range(6):
        # Sample 14 is never drawn, so its scores are NaN. The indices stay
        # unsorted, as the runner passes them.
        idx = rng.choice(n - 1, size=9, replace=False)
        labels = rng.integers(0, 3, size=9)
        runs.append((idx, labels))
    anchors = np.array([0, 2, 5, 9, 13, 14])
    gini, ce = stability_from_runs_anchored(n, runs, anchors, chunk_size=4)
    write(
        "anchored",
        {
            "n": n,
            "runs": [{"indices": idx, "labels": labels} for idx, labels in runs],
            "anchors": anchors,
            "block": consensus_anchor_block(n, runs, anchors),
            "gini": gini,
            "ce": ce,
        },
    )


def noise_masks():
    ties = np.ones(90)
    ties[[3, 41, 84]] = 0.5
    ties[[10, 25, 35, 50, 65, 80]] = 0.9
    with_nan = ties.copy()
    with_nan[[20, 60]] = np.nan
    spread = np.random.default_rng(2).uniform(0.0, 1.0, 40)
    # Every score is within 0.03 of 1, so the margin flags nothing even
    # though some scores lie below the quantile.
    near_one = np.linspace(0.97, 1.0, 40)
    flat = np.full(30, 0.8)
    cases = []
    for name, scores, quantile in [
        ("ties", ties, 0.05),
        ("with_nan", with_nan, 0.05),
        ("spread", spread, 0.1),
        ("spread_quarter", spread, 0.25),
        ("near_one", near_one, 0.1),
        ("flat", flat, 0.05),
    ]:
        cut = noise_mask(scores, quantile=quantile)
        cases.append(
            {
                "name": name,
                "scores": scores,
                "quantile": quantile,
                "mask": cut.mask,
                "n_target": cut.n_target,
                "cutoff": cut.cutoff,
                "median": cut.median,
            }
        )
    write("noise_mask", {"cases": cases})


def knn_graph():
    X = np.random.RandomState(3).randn(25, 2)
    cases = []
    for weighting, n_neighbors in [("connectivity", 4), ("jaccard", 4), ("jaccard", 7)]:
        graph = build_knn_graph(X, n_neighbors=n_neighbors, weighting=weighting)
        cases.append(
            {
                "weighting": weighting,
                "n_neighbors": n_neighbors,
                "edges": np.array(graph.get_edgelist()),
                "weights": graph.es["weight"],
            }
        )
    write("knn_graph", {"X": X, "cases": cases})


def hdbscan():
    # Two close groups and a far one. On these data scikit-learn's eom and
    # leaf selections differ, and dbscan::hdbscan() orders its tied merges so
    # that both of its selections match scikit-learn's exactly (checked with
    # dbscan 1.2.7 when this fixture was written). Other data need not match
    # exactly; the HDBSCAN help page explains why.
    X, _ = make_blobs(
        n_samples=[40, 40, 40],
        centers=[[0, 0], [2.0, 0], [8, 0]],
        cluster_std=[0.4, 0.4, 0.8],
        random_state=0,
    )
    labels = {}
    for m in (5, 10):
        for method in ("eom", "leaf"):
            labels[f"{method}_{m}"] = HDBSCAN(
                min_cluster_size=m, cluster_selection_method=method, copy=True
            ).fit_predict(X)
    write("hdbscan", {"X": X, "labels": labels})


def pipeline_labels():
    # A user-supplied name renders like a class name, so each step is built
    # with a name and a stand-in class; the label depends only on the name
    # and the sampled values.
    steps = [
        ("identity", {}),
        ("log1p", {}),
        ("StandardScaler", {}),
        ("PCA", {"n_components": 10}),
        ("TSNE", {"n_components": 2, "perplexity": 30}),
        ("TSNE", {"perplexity": 12.5, "n_components": 2}),
        ("UMAP", {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1}),
        ("Scaled", {"factor": 0.000123}),
        ("Flag", {"center": True}),
    ]
    cases = [
        {
            "name": name,
            "params": params,
            "label": PipelineStep(cls=object, params=params, name=name).label,
        }
        for name, params in steps
    ]
    spec = PipelineSpec(
        normalization=PipelineStep(cls=object, params={}, name="identity"),
        dim_reduction=PipelineStep(cls=object, params={"n_components": 10}, name="PCA"),
    )
    write("pipeline_labels", {"steps": cases, "spec": {"label": spec.label}})
```

Add `anchored, noise_masks, knn_graph, hdbscan, pipeline_labels` to the end of the tuple in the `__main__` loop.

- [ ] Step 4: Run the script and check that only the new files appear

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
.venv/bin/python carve-r/data-raw/make_fixtures.py
git status --short carve-r/
```

Expected: the script prints a `wrote ...` line for each of the 18 fixtures, and `git status` shows `make_fixtures.py` modified and the five new JSON files untracked. If an existing fixture changed, stop and report: the stage 1 fixtures must reproduce byte for byte.

- [ ] Step 5: Check the fixtures' content

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
.venv/bin/python - <<'EOF'
import json
from pathlib import Path
F = Path("carve-r/tests/testthat/fixtures")
h = json.load(open(F / "hdbscan.json"))["labels"]
for key, labels in h.items():
    print(key, "clusters", len(set(labels) - {-1}), "noise", labels.count(-1))
a = json.load(open(F / "anchored.json"))
print("anchored gini nulls at", [i for i, v in enumerate(a["gini"]) if v is None])
for case in json.load(open(F / "noise_mask.json"))["cases"]:
    print(case["name"], "flagged", sum(case["mask"]), "target", case["n_target"])
print([c["label"] for c in json.load(open(F / "pipeline_labels.json"))["steps"]])
EOF
```

Expected: for at least one `m`, the eom and leaf rows differ in cluster count or noise count; some row has noise; sample 14 is among the null `gini` entries; `ties` flags 3 against a target of 5, `with_nan` flags 5 (3 low scores and 2 NaN) against 5, `near_one` flags 0, `flat` flags 0; the labels include `TSNE(perplexity=12.5, n_components=2)`, `Scaled(factor=0.000123)` and `Flag(center=True)`. Record the output in the report.

- [ ] Step 6: Add the comparison helper to `tests/testthat/helper-fixtures.R`

Append:

```r
# TRUE when two label vectors put -1 on the same samples and group the other
# samples the same way, whatever the label values. Used to compare clusterings
# with Python's, which number clusters differently.
same_partition <- function(a, b) {
  if (!identical(which(a < 0), which(b < 0))) {
    return(FALSE)
  }
  keep <- a >= 0
  if (!any(keep)) {
    return(TRUE)
  }
  counts <- table(a[keep], b[keep])
  all(rowSums(counts > 0) == 1L) && all(colSums(counts > 0) == 1L)
}
```

- [ ] Step 7: Run the existing suite

Run: `cd carve-r && Rscript -e 'devtools::test(stop_on_failure = TRUE)'`
Expected: `[ FAIL 0 | WARN 0 | SKIP 0 | PASS 729 ]`, as at the end of stage 1.

- [ ] Step 8: Commit

```bash
git add carve-r/data-raw/make_fixtures.py carve-r/tests/testthat/helper-fixtures.R \
  carve-r/tests/testthat/fixtures/anchored.json carve-r/tests/testthat/fixtures/noise_mask.json \
  carve-r/tests/testthat/fixtures/knn_graph.json carve-r/tests/testthat/fixtures/hdbscan.json \
  carve-r/tests/testthat/fixtures/pipeline_labels.json
git commit -m "$(cat <<'EOF'
test(carve-r): add the Python fixtures for anchors, noise, graphs, HDBSCAN and pipeline labels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---
### Task 2: Noise policy, noise mask and other utilities

Files:
- Modify: `carve-r/R/utils.R`
- Test: `carve-r/tests/testthat/test-utils.R`

Interfaces:
- Consumes: `noise_mask` fixture and `read_fixture`, `fixture_vector` (Task 1 and stage 1); `format_repr` (stage 1).
- Produces (internal):
  - `seeded(seed, code, kind = "Mersenne-Twister")`: as in stage 1, with the generator kind as an argument.
  - `has_package(package)` returns `TRUE` when `package` can be loaded. Tests mock it.
  - `require_package(package, needed_for)` stops with an install hint when `has_package(package)` is `FALSE`, and returns `invisible(TRUE)` otherwise.
  - `NOISE_POLICIES` (`c("drop", "as_cluster", "singleton")`) and `check_noise_policy(policy)`.
  - `apply_noise_policy(indices, labels, policy = "drop")` returns `list(indices, labels, noise_fraction)`.
  - `NOISE_MARGIN` (0.05) and `noise_mask(scores, quantile, margin = NOISE_MARGIN)` returns `list(mask, n_target, cutoff, median)`.
  - `align_cluster_labels()` returns `labels` unchanged when `keep` selects nothing.

- [ ] Step 1: Write the failing tests at the end of `tests/testthat/test-utils.R`

```r
test_that("seeded can use a second generator and restores the session's kind", {
  kind <- RNGkind()
  mersenne <- seeded(1L, stats::runif(3))
  lecuyer <- seeded(1L, stats::runif(3), kind = "L'Ecuyer-CMRG")
  expect_false(isTRUE(all.equal(mersenne, lecuyer)))
  expect_identical(lecuyer, seeded(1L, stats::runif(3), kind = "L'Ecuyer-CMRG"))
  expect_identical(RNGkind(), kind)
})

test_that("require_package names the missing package and how to install it", {
  expect_error(
    require_package("notAnInstalledPackage", "testing"),
    'notAnInstalledPackage is required for testing. Install it with: install.packages("notAnInstalledPackage")',
    fixed = TRUE
  )
  expect_true(require_package("stats", "testing"))
})

test_that("drop removes noise from the indices and the labels", {
  out <- apply_noise_policy(c(10L, 11L, 12L, 13L), c(1L, -1L, 2L, -1L), "drop")
  expect_identical(out$indices, c(10L, 12L))
  expect_identical(out$labels, c(1L, 2L))
  expect_identical(out$noise_fraction, 0.5)
})

test_that("drop is the default policy", {
  out <- apply_noise_policy(1:3, c(1L, -1L, 1L))
  expect_identical(out$indices, c(1L, 3L))
})

test_that("as_cluster keeps -1 as an ordinary label", {
  out <- apply_noise_policy(1:4, c(1L, -1L, 2L, 2L), "as_cluster")
  expect_identical(out$indices, 1:4)
  expect_identical(out$labels, c(1L, -1L, 2L, 2L))
  expect_identical(out$noise_fraction, 0.25)
})

test_that("singleton gives each noise sample a new cluster past the largest label", {
  out <- apply_noise_policy(1:5, c(2L, -1L, 1L, -1L, -1L), "singleton")
  expect_identical(out$indices, 1:5)
  expect_identical(out$labels, c(2L, 3L, 1L, 4L, 5L))
  expect_identical(out$noise_fraction, 0.6)
})

test_that("singleton numbering starts at 0 when every sample is noise", {
  out <- apply_noise_policy(1:3, c(-1L, -1L, -1L), "singleton")
  expect_identical(out$labels, 0:2)
})

test_that("labels without noise pass through every policy", {
  for (policy in NOISE_POLICIES) {
    out <- apply_noise_policy(4:6, c(1L, 2L, 1L), policy)
    expect_identical(out$indices, 4:6, info = policy)
    expect_identical(out$labels, c(1L, 2L, 1L), info = policy)
    expect_identical(out$noise_fraction, 0, info = policy)
  }
})

test_that("dropping a subsample that is all noise leaves empty vectors", {
  out <- apply_noise_policy(1:3, c(-1L, -1L, -1L), "drop")
  expect_identical(out$indices, integer(0))
  expect_identical(out$labels, integer(0))
  expect_identical(out$noise_fraction, 1)
})

test_that("an empty subsample has a noise fraction of 0", {
  out <- apply_noise_policy(integer(0), integer(0), "drop")
  expect_identical(out$noise_fraction, 0)
})

test_that("an unknown noise policy is an error", {
  expect_error(
    apply_noise_policy(1:2, c(1L, 1L), "ignore"),
    "Unknown noise_policy 'ignore'. Expected 'drop', 'as_cluster', or 'singleton'.",
    fixed = TRUE
  )
})

test_that("noise_mask matches Python", {
  for (case in read_fixture("noise_mask")$cases) {
    cut <- noise_mask(fixture_vector(case$scores), case$quantile)
    expect_identical(cut$mask, as.logical(case$mask), info = case$name)
    expect_identical(cut$n_target, as.integer(case$n_target), info = case$name)
    expect_equal(cut$cutoff, case$cutoff, tolerance = 1e-12, info = case$name)
    expect_equal(cut$median, case$median, tolerance = 1e-12, info = case$name)
  }
})

test_that("noise_mask needs a finite score", {
  expect_error(noise_mask(c(NaN, NA), 0.05), "scores contains no finite value.", fixed = TRUE)
})

test_that("a keep that selects nothing leaves the labels as they are", {
  expect_identical(
    align_cluster_labels(c(-1L, -1L, -1L), c(2L, 1L, 2L), keep = c(FALSE, FALSE, FALSE)),
    c(2L, 1L, 2L)
  )
})
```

- [ ] Step 2: Run them and see them fail

Run: `cd carve-r && Rscript -e 'devtools::test(filter = "utils")'`
Expected: failures for `seeded` (unused argument `kind`), the missing functions, and an error from `max()` of an empty vector in the last test.

- [ ] Step 3: Implement in `R/utils.R`

Replace `seeded()` and its comment with:

```r
# Evaluates code under a fixed seed and restores the caller's RNG state
# afterwards. The RNG kind is pinned, so a user's RNGkind() setting cannot
# change the draws. With seed = NULL the code draws from the caller's stream.
# The anchor draw and the pipeline allocation pass kind = "L'Ecuyer-CMRG":
# Python takes them from a numpy Generator, apart from the subsample draws,
# and a Mersenne-Twister draw from the same seed would repeat the start of
# resample 0's first subsample.
seeded <- function(seed, code, kind = "Mersenne-Twister") {
  if (is.null(seed)) {
    return(code)
  }
  withr::with_seed(
    seed,
    code,
    .rng_kind = kind,
    .rng_normal_kind = "Inversion",
    .rng_sample_kind = "Rejection"
  )
}
```

Append after `count_clusters()`:

```r
# A suggested package is loaded only when a function needs it. has_package()
# is its own function so that tests can pretend a package is missing.
has_package <- function(package) {
  requireNamespace(package, quietly = TRUE)
}

require_package <- function(package, needed_for) {
  if (!has_package(package)) {
    stop(sprintf(
      "%s is required for %s. Install it with: install.packages(\"%s\")",
      package, needed_for, package
    ), call. = FALSE)
  }
  invisible(TRUE)
}

NOISE_POLICIES <- c("drop", "as_cluster", "singleton")

check_noise_policy <- function(policy) {
  if (!is.character(policy) || length(policy) != 1L || !policy %in% NOISE_POLICIES) {
    stop(sprintf(
      "Unknown noise_policy %s. Expected 'drop', 'as_cluster', or 'singleton'.",
      format_repr(policy)
    ), call. = FALSE)
  }
  invisible(policy)
}

# Resolves the negative labels that density-based methods give unassigned
# samples. "drop" removes them from the resample, so they count as not
# drawn; "as_cluster" keeps -1 as an ordinary label; "singleton" gives each
# its own cluster. noise_fraction is the share of noise before the policy.
apply_noise_policy <- function(indices, labels, policy = "drop") {
  check_noise_policy(policy)
  if (length(labels) == 0L) {
    return(list(indices = indices, labels = labels, noise_fraction = 0))
  }
  noise <- labels < 0
  fraction <- mean(noise)
  if (!any(noise) || policy == "as_cluster") {
    return(list(indices = indices, labels = labels, noise_fraction = fraction))
  }
  if (policy == "drop") {
    return(list(indices = indices[!noise], labels = labels[!noise], noise_fraction = fraction))
  }
  assigned <- labels[!noise]
  start <- if (length(assigned) > 0L) max(assigned) + 1L else 0L
  labels[noise] <- seq.int(start, length.out = sum(noise))
  list(indices = indices, labels = labels, noise_fraction = fraction)
}

# How far below the median sample's score, on the [0, 1] score scale, a
# sample must lie before noise_mask() may flag it.
NOISE_MARGIN <- 0.05

# Flags the samples whose score marks them as ambiguous: a NaN score, or a
# score strictly below the quantile of the finite scores and more than
# margin below their median, so that tied scores flag nothing. n_target is
# how many finite samples the quantile would flag if every score were
# distinct, counted through the quantile itself so that it agrees with the
# cutoff. Quantile type 7 is numpy's default.
noise_mask <- function(scores, quantile, margin = NOISE_MARGIN) {
  finite <- is.finite(scores)
  if (!any(finite)) {
    stop("scores contains no finite value.", call. = FALSE)
  }
  values <- scores[finite]
  cutoff <- stats::quantile(values, quantile, type = 7, names = FALSE)
  median <- stats::median(values)
  mask <- !finite
  mask[finite] <- values < cutoff & values < median - margin
  ranks <- seq_along(values) - 1
  n_target <- sum(ranks < stats::quantile(ranks, quantile, type = 7, names = FALSE))
  list(mask = mask, n_target = as.integer(n_target), cutoff = cutoff, median = median)
}
```

In `align_cluster_labels()`, add a guard right after `keep` is defaulted, and extend the comment above the function by one sentence:

```r
  if (!any(keep)) {
    return(labels)
  }
```

Comment addition (append to the existing comment): `With no position kept there is nothing to match, and the labels come back unchanged.`

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "utils", stop_on_failure = TRUE)'`
Expected: PASS. Then run the whole suite (`Rscript -e 'devtools::test(stop_on_failure = TRUE)'`); it must stay green, since `seeded()` keeps its default.

- [ ] Step 5: Mutation check

- In `seeded()`, pass `.rng_kind = "Mersenne-Twister"` regardless of `kind`: the `seeded` test fails.
- In `noise_mask()`, compare `values <= cutoff`: the `ties` case of the fixture test fails.
- In `noise_mask()`, drop `& values < median - margin`: the `near_one` case fails.
- In `apply_noise_policy()`, compute `fraction` after dropping (`mean(labels[!noise] < 0)`): the drop test fails.
- In `apply_noise_policy()`, start singletons at `max(assigned)`: the singleton test fails.

- [ ] Step 6: Commit

```bash
git add carve-r/R/utils.R carve-r/tests/testthat/test-utils.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add the noise policy, the noise mask and a second seeded generator

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Anchored consensus

Files:
- Modify: `carve-r/R/consensus.R`, `carve-r/R/utils.R`
- Test: `carve-r/tests/testthat/test-consensus.R`, `carve-r/tests/testthat/test-utils.R`

Interfaces:
- Consumes: `anchored` fixture (Task 1); `compute_consensus_matrix`, `stability_from_consensus`, `split_subsample_indices`, `format_repr`, `format_param_value` (stage 1); `seeded(kind =)` (Task 2).
- Produces (internal):
  - `run_indicators(n_samples, runs)` returns `list(S, B)`, sparse `n_samples` by `n_runs` and `n_samples` by (total clusters) indicator matrices. `runs` is a list of `list(indices, labels)` with 1-based indices, as `compute_consensus_matrix()` takes.
  - `row_stability(p)` returns `list(gini, ce)` for a matrix whose excluded entries (the self pair) are already `NaN`; `stability_from_consensus()` now calls it.
  - `consensus_anchor_block(n_samples, runs, anchors)` returns the `m` by `m` block, equal to `compute_consensus_matrix(n_samples, runs)[anchors, anchors]`.
  - `default_anchor_chunk_size(m)` returns an integer.
  - `stability_from_runs_anchored(n_samples, runs, anchors, chunk_size = NULL)` returns `list(gini, ce)`, each of length `n_samples`.
  - `resolve_anchors(n_samples, consensus_anchors, anchor_threshold, random_state)` returns sorted 1-based integer anchors, or `NULL` for the exact path. `random_state` is the run's resolved integer seed.

- [ ] Step 1: Write the failing tests

At the end of `tests/testthat/test-consensus.R`:

```r
anchored_fixture <- function() {
  f <- read_fixture("anchored")
  list(
    n = as.integer(f$n),
    runs = lapply(f$runs, function(r) {
      list(indices = as.integer(r$indices) + 1L, labels = as.integer(r$labels))
    }),
    anchors = as.integer(f$anchors) + 1L,
    block = fixture_matrix(f$block),
    gini = fixture_vector(f$gini),
    ce = fixture_vector(f$ce)
  )
}

test_that("the anchor block matches Python", {
  f <- anchored_fixture()
  expect_equal(nan_to_na(consensus_anchor_block(f$n, f$runs, f$anchors)), f$block, tolerance = 1e-6)
})

test_that("the anchor block is the exact matrix restricted to the anchors", {
  f <- anchored_fixture()
  full <- compute_consensus_matrix(f$n, f$runs)
  expect_equal(consensus_anchor_block(f$n, f$runs, f$anchors), full[f$anchors, f$anchors], tolerance = 1e-12)
})

test_that("anchored stability matches Python with and without chunks", {
  f <- anchored_fixture()
  for (chunk in list(NULL, 4L)) {
    scores <- stability_from_runs_anchored(f$n, f$runs, f$anchors, chunk_size = chunk)
    expect_equal(nan_to_na(scores$gini), f$gini, tolerance = 1e-6)
    expect_equal(nan_to_na(scores$ce), f$ce, tolerance = 1e-6)
  }
})

test_that("with every sample an anchor, anchored stability equals the exact scores", {
  f <- anchored_fixture()
  exact <- stability_from_consensus(compute_consensus_matrix(f$n, f$runs))
  anchored <- stability_from_runs_anchored(f$n, f$runs, seq_len(f$n), chunk_size = 4L)
  expect_equal(anchored, exact, tolerance = 1e-12)
})

test_that("the chunk size does not change the scores", {
  f <- anchored_fixture()
  one <- stability_from_runs_anchored(f$n, f$runs, f$anchors, chunk_size = 1L)
  for (chunk in c(3L, 100L)) {
    expect_equal(stability_from_runs_anchored(f$n, f$runs, f$anchors, chunk_size = chunk), one, tolerance = 1e-12)
  }
})

test_that("unsorted run indices give the same scores as sorted ones", {
  f <- anchored_fixture()
  sorted <- lapply(f$runs, function(r) {
    o <- order(r$indices)
    list(indices = r$indices[o], labels = r$labels[o])
  })
  expect_equal(
    stability_from_runs_anchored(f$n, sorted, f$anchors),
    stability_from_runs_anchored(f$n, f$runs, f$anchors),
    tolerance = 1e-12
  )
})

test_that("a sample never drawn with an anchor scores NaN without a warning", {
  f <- anchored_fixture()
  expect_no_warning(scores <- stability_from_runs_anchored(f$n, f$runs, f$anchors))
  expect_true(is.nan(scores$gini[15]))
  expect_true(is.nan(scores$ce[15]))
})

test_that("a run without stability runs gets an all-NaN block", {
  block <- consensus_anchor_block(10L, list(), c(2L, 5L))
  expect_true(all(is.nan(block)))
  expect_identical(dim(block), c(2L, 2L))
})

test_that("the default chunk size keeps a chunk near 2^23 entries", {
  expect_identical(default_anchor_chunk_size(1000L), 8192L)
  expect_identical(default_anchor_chunk_size(5000L), 1677L)
  expect_identical(default_anchor_chunk_size(100000L), 256L)
})
```

At the end of `tests/testthat/test-utils.R`:

```r
test_that("runs at or below the anchor threshold take the exact path", {
  expect_null(resolve_anchors(100L, NULL, 100, 0L))
  expect_null(resolve_anchors(100L, NULL, 5000, 0L))
  # One sample resolves to one anchor, which is the exact path, not an error.
  expect_null(resolve_anchors(1L, NULL, 5000, 0L))
})

test_that("above the threshold the anchor count is the threshold", {
  anchors <- resolve_anchors(200L, NULL, 50, 0L)
  expect_true(is.integer(anchors))
  expect_length(anchors, 50L)
  expect_false(is.unsorted(anchors, strictly = TRUE))
  expect_true(all(anchors >= 1L & anchors <= 200L))
})

test_that("consensus_anchors is a count above 1 and a share up to 1", {
  expect_length(resolve_anchors(100L, 30, 1000, 0L), 30L)
  expect_length(resolve_anchors(100L, 0.25, 1000, 0L), 25L)
  expect_null(resolve_anchors(100L, 1, 1000, 0L))
  expect_null(resolve_anchors(100L, 500, 1000, 0L))
})

test_that("the anchors depend on random_state", {
  expect_identical(resolve_anchors(500L, NULL, 50, 3L), resolve_anchors(500L, NULL, 50, 3L))
  expect_false(identical(resolve_anchors(500L, NULL, 50, 3L), resolve_anchors(500L, NULL, 50, 4L)))
})

test_that("the anchors are not the start of resample 0's first subsample", {
  anchors <- resolve_anchors(1000L, NULL, 100, 0L)
  first <- split_subsample_indices(1000L, 0.618, 0L)$train
  expect_false(all(anchors %in% first))
})

test_that("an invalid consensus_anchors value is an error", {
  for (value in list(0, -3, 2.5, "a", c(10, 20), NA_real_)) {
    expect_error(
      resolve_anchors(100L, value, 1000, 0L),
      "consensus_anchors given as a fraction must be in (0, 1], got",
      fixed = TRUE
    )
  }
})

test_that("too few anchors name the setting they came from", {
  expect_error(
    resolve_anchors(100L, NULL, 1, 0L),
    "The consensus anchor count must be at least 2, got 1. It comes from consensus_anchors=None when that is set, and otherwise from min(n_samples, anchor_threshold=1).",
    fixed = TRUE
  )
  expect_error(
    resolve_anchors(100L, 0.01, 1000, 0L),
    "got 1. It comes from consensus_anchors=0.01 when that is set",
    fixed = TRUE
  )
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "consensus|utils")'`
Expected: failures, the new functions not found.

- [ ] Step 3: Implement the anchored path in `R/consensus.R`

Change the first comment line of the file to `# Consensus matrices and the stability scores derived from them. Mirrors` / `# _consensus.py, the exact path and the anchored one.` (two lines, replacing the stage 1 wording about stage 2). Replace `stability_from_consensus()` with the version below and add the rest after `compute_consensus_pac()`:

```r
# Per-sample stability from the off-diagonal consensus values of each row.
stability_from_consensus <- function(consensus_matrix) {
  p <- consensus_matrix
  diag(p) <- NaN
  row_stability(p)
}

# Gini uncertainty p(1 - p) and binary entropy averaged over each row's
# entries that are not NaN, and rescaled so that 1 means every pair always
# agreed. A row without such an entry scores NaN.
row_stability <- function(p) {
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

# Sparse indicator factors of the runs: S[i, r] is 1 when run r drew sample
# i, and B[i, c] is 1 when sample i fell in run-cluster c. The consensus
# matrix is tcrossprod(B) / tcrossprod(S); the anchored functions use only
# the rows they need.
run_indicators <- function(n_samples, runs) {
  indices <- lapply(runs, function(r) r$indices)
  codes <- lapply(runs, function(r) match(r$labels, unique(r$labels)))
  n_clusters <- vapply(codes, function(code) length(unique(code)), integer(1))
  offsets <- cumsum(c(0L, n_clusters))[seq_along(runs)]
  # as.integer() keeps an empty set of runs, possible when every resample
  # was all noise, from turning into NULL.
  rows <- as.integer(unlist(indices))
  S <- Matrix::sparseMatrix(
    i = rows,
    j = rep(seq_along(runs), lengths(indices)),
    x = 1,
    dims = c(n_samples, length(runs))
  )
  B <- Matrix::sparseMatrix(
    i = rows,
    j = as.integer(unlist(Map(`+`, codes, offsets))),
    x = 1,
    dims = c(n_samples, sum(n_clusters))
  )
  list(S = S, B = B)
}

# The consensus matrix restricted to anchor-by-anchor pairs. Anchoring
# changes which pairs are computed, not any pair's value.
consensus_anchor_block <- function(n_samples, runs, anchors) {
  factors <- run_indicators(n_samples, runs)
  Sa <- as.matrix(factors$S[anchors, , drop = FALSE])
  Ba <- as.matrix(factors$B[anchors, , drop = FALSE])
  co_sampled <- tcrossprod(Sa)
  block <- tcrossprod(Ba) / co_sampled
  block[co_sampled == 0] <- NaN
  unname(block)
}

# Rows per chunk of stability_from_runs_anchored(). Each chunk holds a few
# chunk-by-m double matrices at once; about 2^23 entries each keeps them
# near 64 MB whatever the anchor count.
default_anchor_chunk_size <- function(m) {
  max(256L, min(8192L, as.integer(2^23 %/% max(m, 1))))
}

# Per-sample Gini and cross-entropy stability against the anchors: the row
# statistic of stability_from_consensus() over the m anchor partners instead
# of all n - 1. Rows are processed in chunks, so the n by m slab never
# exists in full, and the result does not depend on the chunk size.
stability_from_runs_anchored <- function(n_samples, runs, anchors, chunk_size = NULL) {
  factors <- run_indicators(n_samples, runs)
  Sa_t <- t(as.matrix(factors$S[anchors, , drop = FALSE]))
  Ba_t <- t(as.matrix(factors$B[anchors, , drop = FALSE]))
  position <- integer(n_samples)
  position[anchors] <- seq_along(anchors)
  if (is.null(chunk_size)) {
    chunk_size <- default_anchor_chunk_size(length(anchors))
  }
  gini <- numeric(n_samples)
  ce <- numeric(n_samples)
  for (start in seq.int(1L, n_samples, by = chunk_size)) {
    rows <- start:min(start + chunk_size - 1L, n_samples)
    co_sampled <- as.matrix(factors$S[rows, , drop = FALSE] %*% Sa_t)
    probs <- as.matrix(factors$B[rows, , drop = FALSE] %*% Ba_t) / co_sampled
    probs[co_sampled == 0] <- NaN
    # An anchor does not count its own pair, as the diagonal is left out of
    # stability_from_consensus().
    own <- position[rows]
    self <- which(own > 0L)
    probs[cbind(self, own[self])] <- NaN
    scores <- row_stability(probs)
    gini[rows] <- scores$gini
    ce[rows] <- scores$ce
  }
  list(gini = gini, ce = ce)
}
```

- [ ] Step 4: Implement `resolve_anchors()` in `R/utils.R`

Append:

```r
# The anchors of a run, sorted, or NULL when the exact path applies. With
# consensus_anchors = NULL there are min(n_samples, anchor_threshold)
# anchors, so the anchor count does not jump as n crosses the threshold. A
# value in (0, 1] is a share of the samples and a whole number above 1 a
# count. Python tells the two apart by type; R numbers are doubles.
resolve_anchors <- function(n_samples, consensus_anchors, anchor_threshold, random_state) {
  if (is.null(consensus_anchors)) {
    m <- min(n_samples, anchor_threshold)
  } else {
    valid <- is.numeric(consensus_anchors) && length(consensus_anchors) == 1L &&
      !is.na(consensus_anchors)
    if (valid && consensus_anchors > 1 && consensus_anchors == round(consensus_anchors)) {
      m <- consensus_anchors
    } else if (valid && consensus_anchors > 0 && consensus_anchors <= 1) {
      m <- round(consensus_anchors * n_samples)
    } else {
      stop(sprintf(
        "consensus_anchors given as a fraction must be in (0, 1], got %s.",
        format_repr(consensus_anchors)
      ), call. = FALSE)
    }
  }
  # The exact path comes first: a run this small is exact, not misconfigured.
  if (m >= n_samples) {
    return(NULL)
  }
  if (m < 2) {
    stop(sprintf(
      "The consensus anchor count must be at least 2, got %d. It comes from consensus_anchors=%s when that is set, and otherwise from min(n_samples, anchor_threshold=%s).",
      as.integer(m), format_repr(consensus_anchors), format_param_value(anchor_threshold)
    ), call. = FALSE)
  }
  sort(seeded(random_state, sample.int(n_samples, m), kind = "L'Ecuyer-CMRG"))
}
```

- [ ] Step 5: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "consensus|utils", stop_on_failure = TRUE)'`
Expected: PASS. Run the whole suite as well; the exact-path tests of stage 1 must stay green after the `row_stability()` refactor.

- [ ] Step 6: Mutation check

- In `stability_from_runs_anchored()`, delete the line that blanks the self pair: the every-sample-an-anchor test fails.
- In `run_indicators()`, use `offsets <- cumsum(n_clusters)` (no leading 0): the fixture tests fail.
- In `default_anchor_chunk_size()`, use `2^20`: the chunk size test fails.
- In `resolve_anchors()`, drop `kind = "L'Ecuyer-CMRG"`: the resample-0 test fails.
- In `resolve_anchors()`, move the `m < 2` check above the exact-path return: the exact-path test fails on its one-sample case.
- In `run_indicators()`, drop the `as.integer()` around the cluster columns: the all-NaN block test fails.

- [ ] Step 7: Commit

```bash
git add carve-r/R/consensus.R carve-r/R/utils.R carve-r/tests/testthat/test-consensus.R carve-r/tests/testthat/test-utils.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add anchored consensus blocks, anchored stability and the anchor draw

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---
### Task 4: The kNN graph, LeidenClustering and LouvainClustering

Files:
- Modify: `carve-r/R/estimators.R`, `carve-r/DESCRIPTION`
- Generated: `carve-r/NAMESPACE`, `carve-r/man/LeidenClustering.Rd`, `carve-r/man/LouvainClustering.Rd`
- Test: `carve-r/tests/testthat/test-estimators.R`

Interfaces:
- Consumes: `knn_graph` fixture (Task 1); `as_data_matrix`, `standard_scale`, `pmax_sparse`, `seeded`, `count_clusters`, `adjusted_rand_index`, `format_repr` (stage 1); `make_blobs` (test helper).
- Produces:
  - internal `knn_graph_weights(X, n_neighbors = 15L, weighting = "connectivity")` returns the symmetric sparse weight matrix; `knn_graph(X, n_neighbors = 15L, weighting = "connectivity")` returns the undirected weighted igraph graph on `nrow(X)` vertices.
  - exported `LeidenClustering(X, resolution = 1, n_neighbors = 15L, weighting = "connectivity", objective_function = "modularity", n_iterations = -1L, random_state = NULL, scale = FALSE)` and `LouvainClustering(X, resolution = 1, n_neighbors = 15L, weighting = "connectivity", scale = FALSE, random_state = NULL)`, both returning integer labels from 1 to the number of communities.

- [ ] Step 1: Write the failing tests at the end of `tests/testthat/test-estimators.R`

```r
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
```

If the recovery or monotonicity assertions fail on these R-generated blobs, report the observed ARI or counts before changing the data; the Python suite makes the same assertions on blobs of the same shape.

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "estimators")'`
Expected: failures, `knn_graph` and the estimators not found.

- [ ] Step 3: Add igraph to Imports in `DESCRIPTION`

```
Imports:
    BiocParallel,
    clue,
    FNN,
    igraph,
    Matrix,
    methods,
    parallel,
    ranger,
    RSpectra,
    stats,
    utils,
    withr (>= 2.4.2)
```

- [ ] Step 4: Implement in `R/estimators.R`

Replace the first comment of the file with:

```r
# Clustering estimators. KMeans and AgglomerativeClustering stand in for the
# scikit-learn classes of the same names; SpectralClustering,
# LeidenClustering and LouvainClustering port cluster.py. Each takes the data
# matrix first and returns integer labels counting from 1.
```

Append:

```r
# The symmetric kNN weight matrix the graph estimators cluster. Each sample
# links to its k = min(n_neighbors, n - 1) nearest other samples, as
# scikit-learn's kneighbors_graph(include_self = False) does, and an edge is
# kept when either endpoint chose the other. "jaccard" weights an edge by
# the Jaccard index of the endpoints' neighbor sets, |shared| / (2k - |shared|),
# as in a shared-nearest-neighbor graph.
knn_graph_weights <- function(X, n_neighbors = 15L, weighting = "connectivity") {
  if (!is.character(weighting) || length(weighting) != 1L ||
      !weighting %in% c("connectivity", "jaccard")) {
    stop(sprintf(
      "Unknown weighting: %s. Expected 'connectivity' or 'jaccard'.",
      format_repr(weighting)
    ), call. = FALSE)
  }
  n <- nrow(X)
  k <- as.integer(min(n_neighbors, max(n - 1L, 1L)))
  nn <- FNN::get.knn(X, k = k)
  A <- Matrix::sparseMatrix(
    i = rep(seq_len(n), times = k),
    j = as.vector(nn$nn.index),
    x = 1,
    dims = c(n, n)
  )
  W <- pmax_sparse(A, Matrix::t(A))
  if (weighting == "jaccard") {
    shared <- methods::as(Matrix::tcrossprod(A), "generalMatrix")
    Matrix::diag(shared) <- 0
    shared <- Matrix::drop0(shared)
    shared@x <- shared@x / (2 * k - shared@x)
    W <- shared * W
  }
  Matrix::drop0(W)
}

knn_graph <- function(X, n_neighbors = 15L, weighting = "connectivity") {
  W <- knn_graph_weights(X, n_neighbors = n_neighbors, weighting = weighting)
  igraph::graph_from_adjacency_matrix(W, mode = "undirected", weighted = TRUE, diag = FALSE)
}

#' Leiden clustering on a nearest-neighbor graph
#'
#' Links every sample to its nearest neighbors and splits the resulting graph
#' into communities with the Leiden algorithm, through
#' [igraph::cluster_leiden()]. There is no number of clusters to set:
#' `resolution` controls the granularity, and larger values give more,
#' smaller communities. Use `carve(resolution = )` to compare resolutions.
#'
#' Each sample is linked to its `n_neighbors` nearest other samples, and an
#' edge is kept when either sample chose the other.
#' `weighting = "connectivity"` gives every edge weight 1. `"jaccard"` weights
#' an edge by the Jaccard index of the two samples' neighbor sets, as a
#' shared-nearest-neighbor graph does, which weakens edges between clusters.
#'
#' `objective_function = "modularity"` optimizes modularity at the given
#' resolution, the objective the Python package optimizes with leidenalg.
#' `"cpm"` uses the constant Potts model, whose resolution is on a different
#' scale, so the two should not share a resolution grid. igraph's Leiden
#' implementation differs from leidenalg's in its random choices, so the
#' partitions agree with the Python package's in quality, not label by label.
#'
#' @param X Numeric matrix or data frame, one row per sample.
#' @param resolution Resolution of the quality function. Larger values give
#'   more clusters.
#' @param n_neighbors Neighbors per sample in the graph, at most
#'   `nrow(X) - 1`. In a [carve()] run it is scaled to each subsample's size.
#' @param weighting `"connectivity"` or `"jaccard"`.
#' @param objective_function `"modularity"` or `"cpm"`.
#' @param n_iterations Number of Leiden iterations. A negative value iterates
#'   until the partition stops improving.
#' @param random_state Seed, or `NULL` to draw from the session's random
#'   number stream.
#' @param scale Standardize each column before building the graph.
#' @return Integer labels from 1 to the number of communities, one per row of
#'   `X`.
#' @references Traag, V. A., Waltman, L. and van Eck, N. J. (2019). From
#'   Louvain to Leiden: guaranteeing well-connected communities. Scientific
#'   Reports 9, 5233.
#' @seealso [LouvainClustering()], and [estimator_grid()] to use it in
#'   [carve()].
#' @examples
#' X <- rbind(matrix(rnorm(80, 0, 0.3), ncol = 2), matrix(rnorm(80, 3, 0.3), ncol = 2))
#' table(LeidenClustering(X, resolution = 0.5, random_state = 1))
#' @export
LeidenClustering <- function(X, resolution = 1, n_neighbors = 15L, weighting = "connectivity",
                             objective_function = "modularity", n_iterations = -1L,
                             random_state = NULL, scale = FALSE) {
  if (identical(objective_function, "modularity")) {
    objective <- "modularity"
  } else if (identical(objective_function, "cpm")) {
    objective <- "CPM"
  } else {
    stop(sprintf(
      "Unknown objective_function: %s. Expected 'modularity' or 'cpm'.",
      format_repr(objective_function)
    ), call. = FALSE)
  }
  X <- as_data_matrix(X)
  Xp <- if (isTRUE(scale)) standard_scale(X) else X
  graph <- knn_graph(Xp, n_neighbors = n_neighbors, weighting = weighting)
  # igraph draws from R's generator, so seeded() fixes the result.
  partition <- seeded(random_state, igraph::cluster_leiden(
    graph,
    objective_function = objective,
    weights = igraph::E(graph)$weight,
    resolution = resolution,
    n_iterations = n_iterations
  ))
  as.integer(igraph::membership(partition))
}

#' Louvain clustering on a nearest-neighbor graph
#'
#' Builds the same nearest-neighbor graph as [LeidenClustering()] and splits
#' it into communities by multi-level modularity optimization, through
#' [igraph::cluster_louvain()]. Louvain can return communities that are not
#' connected inside, which Leiden prevents; it is offered for comparison.
#'
#' @inheritParams LeidenClustering
#' @param resolution Resolution of the modularity. Larger values give more
#'   clusters.
#' @return Integer labels from 1 to the number of communities, one per row of
#'   `X`.
#' @references Blondel, V. D., Guillaume, J.-L., Lambiotte, R. and Lefebvre,
#'   E. (2008). Fast unfolding of communities in large networks. Journal of
#'   Statistical Mechanics, P10008.
#' @seealso [LeidenClustering()], and [estimator_grid()] to use it in
#'   [carve()].
#' @examples
#' X <- rbind(matrix(rnorm(80, 0, 0.3), ncol = 2), matrix(rnorm(80, 3, 0.3), ncol = 2))
#' table(LouvainClustering(X, resolution = 0.5, random_state = 1))
#' @export
LouvainClustering <- function(X, resolution = 1, n_neighbors = 15L, weighting = "connectivity",
                              scale = FALSE, random_state = NULL) {
  X <- as_data_matrix(X)
  Xp <- if (isTRUE(scale)) standard_scale(X) else X
  graph <- knn_graph(Xp, n_neighbors = n_neighbors, weighting = weighting)
  partition <- seeded(random_state, igraph::cluster_louvain(
    graph,
    weights = igraph::E(graph)$weight,
    resolution = resolution
  ))
  as.integer(igraph::membership(partition))
}
```

- [ ] Step 5: Document, then run the tests and see them pass

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "estimators", stop_on_failure = TRUE)'
```

Expected: `document()` writes the two Rd files and adds two `export()` lines; the tests pass. Then run the whole suite.

- [ ] Step 6: Mutation check

- In `knn_graph_weights()`, use `k <- n_neighbors - 1L` before the clip (the self-counting convention): the fixture test fails.
- In `knn_graph_weights()`, divide by `(2 * k + shared@x)`: the Jaccard cases of the fixture test fail.
- In `LeidenClustering()`, drop `resolution = resolution`: the monotonicity test fails.
- In `LeidenClustering()`, map `"cpm"` to `"modularity"`: the objective test fails.
- In `LouvainClustering()`, call igraph without `seeded()`: the reproducibility test fails.

- [ ] Step 7: Commit

```bash
git add carve-r/DESCRIPTION carve-r/NAMESPACE carve-r/R/estimators.R \
  carve-r/man/LeidenClustering.Rd carve-r/man/LouvainClustering.Rd carve-r/tests/testthat/test-estimators.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add LeidenClustering and LouvainClustering on a kNN graph

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: HDBSCAN

Files:
- Modify: `carve-r/R/estimators.R`, `carve-r/DESCRIPTION`
- Generated: `carve-r/NAMESPACE`, `carve-r/man/HDBSCAN.Rd`
- Test: `carve-r/tests/testthat/test-estimators.R`

Interfaces:
- Consumes: `hdbscan` fixture and `same_partition` (Task 1); `require_package`, `has_package` (Task 2); `as_data_matrix`, `format_repr` (stage 1).
- Produces:
  - exported `HDBSCAN(X, min_cluster_size = 5L, cluster_selection_method = "eom")` returns integer labels from 1 to the number of clusters, and -1 for noise.
  - internal `hdbscan_leaf_labels(fit, n_samples)`.

- [ ] Step 1: Write the failing tests at the end of `tests/testthat/test-estimators.R`

```r
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
    labels <- HDBSCAN(X, min_cluster_size = 10L, cluster_selection_method = method)
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
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "estimators")'`
Expected: failures, `HDBSCAN` not found.

- [ ] Step 3: Add dbscan to Suggests in `DESCRIPTION`

```
Suggests:
    dbscan,
    jsonlite,
    testthat (>= 3.1.8)
```

- [ ] Step 4: Implement in `R/estimators.R`

Change the first comment of the file to name HDBSCAN among the stand-ins: `# Clustering estimators. KMeans, AgglomerativeClustering and HDBSCAN stand in` / `# for the scikit-learn classes of the same names; SpectralClustering,` / `# LeidenClustering and LouvainClustering port cluster.py. Each takes the data` / `# matrix first and returns integer labels counting from 1, with -1 for noise.` (four lines). Append:

```r
#' HDBSCAN clustering
#'
#' Density-based clustering through [dbscan::hdbscan()]. Samples in regions
#' too sparse to belong to a cluster are labeled -1, and [carve()] handles
#' them according to its `noise_policy`. It needs the dbscan package, which
#' CARVE suggests but does not install.
#'
#' `cluster_selection_method = "eom"` keeps the clusters with the largest
#' excess of mass. `"leaf"` keeps the leaves of the cluster tree, which gives
#' more and smaller clusters and more noise. `min_cluster_size` also sets the
#' number of neighbors of the core distances, as in scikit-learn's `HDBSCAN`
#' with `min_samples` left at its default.
#'
#' The labels can differ from scikit-learn's on the same data. Mutual
#' reachability distances are often tied, and the two packages merge tied
#' samples in a different order, which can change the clusters selected. On
#' 25 simulated data sets of three overlapping groups, at minimum cluster
#' sizes 5 and 10, the eom selections were identical in 46 of 50 cases and
#' the leaf selections in 25 of 50. dbscan also computes all pairwise
#' distances, so memory grows with the square of the number of samples.
#'
#' @param X Numeric matrix or data frame, one row per sample.
#' @param min_cluster_size Smallest cluster, at least 2. Larger values give
#'   fewer clusters, so a [carve()] sweep over it runs from fine to coarse.
#' @param cluster_selection_method `"eom"` or `"leaf"`.
#' @return Integer labels from 1 to the number of clusters, and -1 for noise,
#'   one per row of `X`.
#' @seealso [estimator_grid()] to use it in [carve()].
#' @examples
#' X <- rbind(matrix(rnorm(80, 0, 0.3), ncol = 2), matrix(rnorm(80, 3, 0.3), ncol = 2))
#' if (requireNamespace("dbscan", quietly = TRUE)) {
#'   table(HDBSCAN(X, min_cluster_size = 10))
#' }
#' @export
HDBSCAN <- function(X, min_cluster_size = 5L, cluster_selection_method = "eom") {
  require_package("dbscan", "HDBSCAN")
  if (!identical(cluster_selection_method, "eom") && !identical(cluster_selection_method, "leaf")) {
    stop(sprintf(
      "Unknown cluster_selection_method: %s. Expected 'eom' or 'leaf'.",
      format_repr(cluster_selection_method)
    ), call. = FALSE)
  }
  X <- as_data_matrix(X)
  fit <- dbscan::hdbscan(X, minPts = as.integer(min_cluster_size))
  if (cluster_selection_method == "leaf") {
    return(hdbscan_leaf_labels(fit, nrow(X)))
  }
  labels <- as.integer(fit$cluster)
  labels[labels == 0L] <- -1L
  labels
}

# Leaf selection from the cluster hierarchy dbscan::hdbscan() keeps in its
# "hdbscan" attribute. Each entry is a cluster whose "contains" lists the
# samples that leave it, and the "cl_hierarchy" attribute maps each parent
# to its two children. A leaf has no children, so all its samples leave it
# and "contains" is its membership. Cluster "0" is the root, which is never
# selected. Samples outside every leaf are noise, as in scikit-learn.
hdbscan_leaf_labels <- function(fit, n_samples) {
  hierarchy <- attr(fit, "hdbscan")
  parents <- names(attr(hierarchy, "cl_hierarchy"))
  leaves <- setdiff(names(hierarchy), c("0", parents))
  leaves <- leaves[order(as.integer(leaves))]
  labels <- rep(-1L, n_samples)
  for (i in seq_along(leaves)) {
    labels[hierarchy[[leaves[i]]]$contains] <- i
  }
  labels
}
```

- [ ] Step 5: Document, then run the tests and see them pass

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "estimators", stop_on_failure = TRUE)'
```

Expected: PASS with `SKIP 0`. Then run the whole suite.

- [ ] Step 6: Mutation check

- In `HDBSCAN()`, return the eom labels for `"leaf"` too: the fixture test fails.
- In `HDBSCAN()`, leave dbscan's 0 for noise: both tests fail.
- In `hdbscan_leaf_labels()`, drop `parents` from the `setdiff()` (every cluster but the root counts as a leaf): the fixture test fails on a leaf case.

- [ ] Step 7: Commit

```bash
git add carve-r/DESCRIPTION carve-r/NAMESPACE carve-r/R/estimators.R carve-r/man/HDBSCAN.Rd \
  carve-r/tests/testthat/test-estimators.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add HDBSCAN with eom and leaf selection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Default grids for the resolution and min_cluster_size axes

Files:
- Modify: `carve-r/R/grids.R`
- Test: `carve-r/tests/testthat/test-grids.R`

Interfaces:
- Consumes: `LeidenClustering`, `LouvainClustering` (Task 4); `HDBSCAN` (Task 5); `require_package`, `has_package` (Task 2); `estimator_grid`, `expand_param_grid`, `resolve_sweep`, `default_k_grids` (stage 1).
- Produces (internal): `default_estimator_grids(X, preset, sweep)` dispatches on `sweep@param` to `default_k_grids()`, `default_resolution_grids(resolutions, preset)` and `default_min_cluster_size_grids(sizes, preset)`; any other parameter is an error.

- [ ] Step 1: Write the failing tests at the end of `tests/testthat/test-grids.R`

```r
test_that("the light resolution preset runs Leiden and Louvain on 15 neighbors", {
  grids <- default_estimator_grids(matrix(0, 10, 2), "light", resolve_sweep(resolution = c(1, 0.5)))
  expect_identical(vapply(grids, function(g) g$name, character(1)), c("LeidenClustering", "LouvainClustering"))
  expect_identical(grids[[1L]]$estimator, LeidenClustering)
  expect_identical(grids[[2L]]$estimator, LouvainClustering)
  for (g in grids) {
    expect_identical(g$grid$resolution, c(0.5, 1))
    expect_identical(g$grid$n_neighbors, 15L)
  }
})

test_that("the full resolution preset adds 10 and 30 neighbors", {
  grids <- default_estimator_grids(matrix(0, 10, 2), "full", resolve_sweep(resolution = c(0.5, 1)))
  expect_identical(
    vapply(grids, function(g) g$name, character(1)),
    c("LeidenClustering", "LouvainClustering", "LeidenClustering", "LouvainClustering")
  )
  expect_identical(grids[[3L]]$grid$n_neighbors, c(10L, 30L))
  expect_identical(grids[[4L]]$grid$n_neighbors, c(10L, 30L))
  expect_identical(sum(vapply(grids, function(g) length(expand_param_grid(g$grid)), integer(1))), 12L)
})

test_that("the min_cluster_size presets run HDBSCAN with eom, and leaf in full", {
  skip_if_not_installed("dbscan")
  sweep <- resolve_sweep(sweep = "min_cluster_size", sweep_values = c(10, 5))
  light <- default_estimator_grids(matrix(0, 10, 2), "light", sweep)
  expect_length(light, 1L)
  expect_identical(light[[1L]]$name, "HDBSCAN")
  expect_identical(light[[1L]]$grid$min_cluster_size, c(5L, 10L))
  expect_identical(light[[1L]]$grid$cluster_selection_method, "eom")
  expect_false("n_clusters" %in% names(light[[1L]]$grid))
  full <- default_estimator_grids(matrix(0, 10, 2), "full", sweep)
  expect_identical(vapply(full, function(g) g$grid$cluster_selection_method, character(1)), c("eom", "leaf"))
})

test_that("the min_cluster_size presets need dbscan", {
  local_mocked_bindings(has_package = function(package) FALSE)
  sweep <- resolve_sweep(sweep = "min_cluster_size", sweep_values = c(5, 10))
  expect_error(
    default_estimator_grids(matrix(0, 10, 2), "light", sweep),
    "dbscan is required for HDBSCAN.",
    fixed = TRUE
  )
})

test_that("a sweep without default grids is an error", {
  sweep <- resolve_sweep(sweep = "eps", sweep_values = c(0.1, 0.2), finer_is_larger = FALSE)
  expect_error(
    default_estimator_grids(matrix(0, 10, 2), "light", sweep),
    "No default estimator grids for sweep parameter 'eps'. Pass estimator_param_grids=... explicitly.",
    fixed = TRUE
  )
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "grids")'`
Expected: failures; the resolution and min_cluster_size calls reach the stage 1 error.

- [ ] Step 3: Implement in `R/grids.R`

Change the file's first comment to `# Default estimator grids. Mirrors _grids.py.` (one line) and replace `default_estimator_grids()` with:

```r
default_estimator_grids <- function(X, preset, sweep) {
  switch(sweep@param,
    n_clusters = default_k_grids(X, sweep@values, preset),
    resolution = default_resolution_grids(sweep@values, preset),
    min_cluster_size = default_min_cluster_size_grids(sweep@values, preset),
    stop(sprintf(
      "No default estimator grids for sweep parameter %s. Pass estimator_param_grids=... explicitly.",
      format_repr(sweep@param)
    ), call. = FALSE)
  )
}

# Graph communities swept over resolution on a 15-neighbor graph; the full
# preset adds 10- and 30-neighbor graphs.
default_resolution_grids <- function(resolutions, preset) {
  resolutions <- as.numeric(resolutions)
  grids <- list(
    estimator_grid(LeidenClustering, resolution = resolutions, n_neighbors = 15L),
    estimator_grid(LouvainClustering, resolution = resolutions, n_neighbors = 15L)
  )
  if (identical(preset, "full")) {
    grids <- c(grids, list(
      estimator_grid(LeidenClustering, resolution = resolutions, n_neighbors = c(10L, 30L)),
      estimator_grid(LouvainClustering, resolution = resolutions, n_neighbors = c(10L, 30L))
    ))
  }
  grids
}

# HDBSCAN swept over min_cluster_size with excess-of-mass selection; the full
# preset adds leaf selection.
default_min_cluster_size_grids <- function(sizes, preset) {
  require_package("dbscan", "HDBSCAN")
  sizes <- as.integer(sizes)
  grids <- list(
    estimator_grid(HDBSCAN, min_cluster_size = sizes, cluster_selection_method = "eom")
  )
  if (identical(preset, "full")) {
    grids <- c(grids, list(
      estimator_grid(HDBSCAN, min_cluster_size = sizes, cluster_selection_method = "leaf")
    ))
  }
  grids
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "grids", stop_on_failure = TRUE)'`
Expected: PASS. Then run the whole suite.

- [ ] Step 5: Mutation check

- In `default_resolution_grids()`, give the full preset `n_neighbors = c(10L, 15L)`: the full-preset test fails.
- In `default_min_cluster_size_grids()`, use `"eom"` in the full preset's second grid: the HDBSCAN preset test fails.
- In `default_min_cluster_size_grids()`, drop the `require_package()` line: the dbscan test fails (HDBSCAN is not called while building grids).

- [ ] Step 6: Commit

```bash
git add carve-r/R/grids.R carve-r/tests/testthat/test-grids.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add the default grids for the resolution and min_cluster_size axes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---
### Task 7: Preprocessing transforms

Files:
- Create: `carve-r/R/transforms.R`
- Modify: `carve-r/DESCRIPTION`
- Generated: `carve-r/NAMESPACE`, `carve-r/man/transforms.Rd`, the Collate field
- Test: `carve-r/tests/testthat/test-transforms.R` (new)

Interfaces:
- Consumes: `as_data_matrix`, `seeded`, `format_param_value`, `standard_scale` (stage 1); `require_package`, `has_package` (Task 2); `make_blobs`, `KMeans`, `adjusted_rand_index` (tests).
- Produces:
  - exported `Identity(X)`, `StandardScaler(X)`, `Log1p(X)`, `PCA(X, n_components = NULL, random_state = NULL)`, `TSNE(X, n_components = 2L, perplexity = 30, random_state = NULL)`, `UMAP(X, n_components = 2L, n_neighbors = 15L, min_dist = 0.1, random_state = NULL)`, each returning a numeric matrix with one row per row of `X`.
  - internal `pca_scores(X, k)`.

- [ ] Step 1: Write the failing tests in `tests/testthat/test-transforms.R`

```r
two <- make_blobs(n_per = 40L, centers = rbind(c(0, 0, 0), c(8, 8, 8)), seed = 3L)

test_that("Identity, StandardScaler and Log1p transform as their names say", {
  X <- cbind(c(1, 2, 3, 6), c(5, 5, 5, 5))
  expect_identical(Identity(X), X)
  Z <- StandardScaler(X)
  expect_equal(colMeans(Z), c(0, 0))
  # The population standard deviation, as scikit-learn divides by n.
  expect_equal(sqrt(mean(Z[, 1]^2)), 1)
  expect_identical(Z[, 2], rep(0, 4))
  expect_identical(Log1p(X), log1p(X))
})

test_that("PCA returns the leading principal component scores", {
  X <- withr::with_seed(2, matrix(stats::rnorm(400), 100) %*% diag(c(5, 3, 1, 0.5)))
  reference <- unname(stats::prcomp(X)$x)
  # One component goes through irlba, three through prcomp.
  expect_equal(abs(PCA(X, n_components = 1L, random_state = 0L)[, 1]), abs(reference[, 1]), tolerance = 1e-4)
  expect_equal(abs(PCA(X, n_components = 3L, random_state = 0L)), abs(reference[, 1:3]), tolerance = 1e-8)
  expect_identical(dim(PCA(X)), c(100L, 4L))
})

test_that("PCA rejects a component count the data cannot give", {
  expect_error(
    PCA(matrix(stats::runif(20), 10), n_components = 3L),
    "n_components=3 must be between 1 and min(n_samples, n_features)=2.",
    fixed = TRUE
  )
})

test_that("TSNE runs Rtsne with scikit-learn's defaults", {
  seen <- NULL
  local_mocked_bindings(
    Rtsne = function(X, ...) {
      seen <<- list(...)
      list(Y = matrix(0, nrow(X), 2L))
    },
    .package = "Rtsne"
  )
  X <- withr::with_seed(1, matrix(stats::rnorm(6000), 3000))
  TSNE(X, perplexity = 10, random_state = 0L)
  expect_identical(seen$dims, 2L)
  expect_identical(seen$perplexity, 10)
  expect_false(seen$pca)
  expect_false(seen$normalize)
  expect_false(seen$check_duplicates)
  expect_identical(seen$max_iter, 1000L)
  # scikit-learn's learning_rate = "auto" is max(n / 48, 50); Rtsne's eta is
  # four times scikit-learn's learning rate.
  expect_identical(seen$eta, 250)
  expect_identical(c(seen$stop_lying_iter, seen$mom_switch_iter), c(250L, 250L))
  expect_identical(dim(seen$Y_init), c(3000L, 2L))
  expect_equal(sqrt(mean(seen$Y_init[, 1]^2)), 1e-4)
})

test_that("TSNE separates two groups and is reproducible", {
  set.seed(1)
  before <- .Random.seed
  Y <- TSNE(two$X, perplexity = 10, random_state = 0L)
  expect_identical(.Random.seed, before)
  expect_identical(dim(Y), c(80L, 2L))
  expect_identical(Y, TSNE(two$X, perplexity = 10, random_state = 0L))
  expect_identical(adjusted_rand_index(KMeans(Y, 2L, random_state = 0L), two$y), 1)
})

test_that("a perplexity too large for the samples is an error", {
  expect_error(
    TSNE(matrix(seq_len(40) / 40, 20), perplexity = 7),
    "perplexity=7 is too large for 20 samples; Rtsne needs 3 * perplexity <= n_samples - 1.",
    fixed = TRUE
  )
})

test_that("UMAP runs uwot with umap-learn's defaults on one thread", {
  skip_if_not_installed("uwot")
  seen <- NULL
  local_mocked_bindings(
    umap = function(X, ...) {
      seen <<- list(...)
      matrix(0, nrow(X), 2L)
    },
    .package = "uwot"
  )
  UMAP(two$X, random_state = 0L)
  expect_identical(seen$n_neighbors, 15L)
  expect_identical(seen$n_components, 2L)
  expect_identical(seen$min_dist, 0.1)
  expect_identical(seen$n_threads, 1L)
  expect_identical(seen$n_sgd_threads, 0L)
  expect_false(seen$verbose)
})

test_that("UMAP is reproducible and leaves the session's stream alone", {
  skip_if_not_installed("uwot")
  set.seed(1)
  before <- .Random.seed
  Y <- UMAP(two$X, n_neighbors = 10L, random_state = 0L)
  expect_identical(.Random.seed, before)
  expect_identical(dim(Y), c(80L, 2L))
  expect_identical(Y, UMAP(two$X, n_neighbors = 10L, random_state = 0L))
})

test_that("UMAP needs the uwot package", {
  local_mocked_bindings(has_package = function(package) FALSE)
  expect_error(
    UMAP(two$X),
    'uwot is required for UMAP. Install it with: install.packages("uwot")',
    fixed = TRUE
  )
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "transforms")'`
Expected: failures, the functions not found.

- [ ] Step 3: Add the dependencies to `DESCRIPTION`

Imports gains `irlba` after `igraph` and `Rtsne` after `ranger` (case-insensitive order: `ranger`, `RSpectra`, `Rtsne`); Suggests gains `uwot` after `testthat`:

```
Imports:
    BiocParallel,
    clue,
    FNN,
    igraph,
    irlba,
    Matrix,
    methods,
    parallel,
    ranger,
    RSpectra,
    Rtsne,
    stats,
    utils,
    withr (>= 2.4.2)
Suggests:
    dbscan,
    jsonlite,
    testthat (>= 3.1.8),
    uwot
```

- [ ] Step 4: Write `R/transforms.R`

```r
# Preprocessing transforms for randomized preprocessing. They stand in for
# the scikit-learn and umap-learn classes the Python package uses: each takes
# the data matrix, its settings and, when it draws, random_state, and returns
# the transformed matrix.

#' Preprocessing transforms
#'
#' Functions that transform a data matrix before clustering, for randomized
#' preprocessing in [carve()]. Each takes the samples as rows and returns a
#' numeric matrix with one row per sample. Pass them to
#' [preprocessing_option()] to build the option lists.
#'
#' `Identity()` returns the data unchanged. `StandardScaler()` centers each
#' column and divides it by its standard deviation, computed with denominator
#' n as scikit-learn does; a constant column becomes zero. `Log1p()` returns
#' `log(1 + X)`, for count data.
#'
#' `PCA()` returns principal component scores of the centered data. It uses
#' [irlba::prcomp_irlba()] when it keeps fewer than half of `min(dim(X))`
#' components and [stats::prcomp()] otherwise.
#'
#' `TSNE()` runs Barnes-Hut t-SNE through [Rtsne::Rtsne()] with the defaults
#' of scikit-learn's `TSNE`: no PCA step before the embedding, a start from
#' the first principal components scaled to a standard deviation of 1e-4,
#' early exaggeration 12 for the first 250 of 1,000 iterations, and the
#' learning rate scikit-learn chooses with `learning_rate = "auto"`, which is
#' `max(n / 12, 200)` in Rtsne's units. Rtsne needs
#' `3 * perplexity <= nrow(X) - 1`.
#'
#' `UMAP()` runs [uwot::umap()] with the defaults of umap-learn's `UMAP`,
#' which include `min_dist = 0.1` (uwot's own default is 0.01). It runs on
#' one thread, so that a seed reproduces the embedding, and it needs the uwot
#' package, which CARVE suggests but does not install.
#'
#' R and Python draw different random numbers, so `PCA()` signs aside, the
#' embeddings are not the ones the Python package computes.
#'
#' @param X Numeric matrix or data frame, one row per sample.
#' @param n_components Number of dimensions to keep. For `PCA()`, `NULL`
#'   keeps `min(dim(X))`.
#' @param perplexity Perplexity of the t-SNE neighborhoods.
#' @param n_neighbors Neighbors per sample in the UMAP graph. It is not scaled
#'   to the subsample size in a [carve()] run.
#' @param min_dist Smallest distance between embedded points in UMAP.
#' @param random_state Seed, or `NULL` to draw from the session's random
#'   number stream.
#' @return A numeric matrix with one row per row of `X`.
#' @seealso [preprocessing_option()], [preprocessing_results()]
#' @examples
#' X <- rbind(matrix(rnorm(120, 0, 0.3), ncol = 3), matrix(rnorm(120, 3, 0.3), ncol = 3))
#' dim(PCA(X, n_components = 2))
#' dim(TSNE(X, perplexity = 10, random_state = 1))
#' @name transforms
NULL

#' @rdname transforms
#' @export
Identity <- function(X) {
  as_data_matrix(X)
}

#' @rdname transforms
#' @export
StandardScaler <- function(X) {
  standard_scale(as_data_matrix(X))
}

#' @rdname transforms
#' @export
Log1p <- function(X) {
  log1p(as_data_matrix(X))
}

#' @rdname transforms
#' @export
PCA <- function(X, n_components = NULL, random_state = NULL) {
  X <- as_data_matrix(X)
  limit <- min(dim(X))
  k <- if (is.null(n_components)) limit else as.integer(n_components)
  if (k < 1L || k > limit) {
    stop(sprintf(
      "n_components=%d must be between 1 and min(n_samples, n_features)=%d.",
      k, limit
    ), call. = FALSE)
  }
  seeded(random_state, pca_scores(X, k))
}

# Scores of the k leading principal components of the centered data. irlba
# draws its start vector, so callers seed it. Above half of min(dim(X))
# components irlba warns and is slower than a full decomposition.
pca_scores <- function(X, k) {
  if (k < min(dim(X)) / 2) {
    scores <- irlba::prcomp_irlba(X, n = k, center = TRUE, scale. = FALSE)$x
  } else {
    scores <- stats::prcomp(X, center = TRUE, scale. = FALSE, rank. = k)$x
  }
  unname(scores[, seq_len(k), drop = FALSE])
}

#' @rdname transforms
#' @export
TSNE <- function(X, n_components = 2L, perplexity = 30, random_state = NULL) {
  X <- as_data_matrix(X)
  n <- nrow(X)
  if (3 * perplexity > n - 1) {
    stop(sprintf(
      "perplexity=%s is too large for %d samples; Rtsne needs 3 * perplexity <= n_samples - 1.",
      format_param_value(perplexity), n
    ), call. = FALSE)
  }
  seeded(random_state, {
    # scikit-learn starts from the PCA scores scaled so that the first
    # coordinate has standard deviation 1e-4. Rtsne skips early exaggeration
    # when given a start unless stop_lying_iter and mom_switch_iter are set.
    start <- pca_scores(X, n_components)
    start <- start / sqrt(mean((start[, 1L] - mean(start[, 1L]))^2)) * 1e-4
    Rtsne::Rtsne(
      X,
      dims = n_components,
      perplexity = perplexity,
      theta = 0.5,
      pca = FALSE,
      normalize = FALSE,
      check_duplicates = FALSE,
      max_iter = 1000L,
      eta = max(n / 12, 200),
      Y_init = start,
      stop_lying_iter = 250L,
      mom_switch_iter = 250L,
      exaggeration_factor = 12,
      num_threads = 1L,
      verbose = FALSE
    )$Y
  })
}

#' @rdname transforms
#' @export
UMAP <- function(X, n_components = 2L, n_neighbors = 15L, min_dist = 0.1, random_state = NULL) {
  require_package("uwot", "UMAP")
  X <- as_data_matrix(X)
  embedding <- seeded(random_state, uwot::umap(
    X,
    n_neighbors = n_neighbors,
    n_components = n_components,
    min_dist = min_dist,
    n_threads = 1L,
    n_sgd_threads = 0L,
    verbose = FALSE
  ))
  unname(embedding)
}
```

- [ ] Step 5: Document, then run the tests and see them pass

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "transforms", stop_on_failure = TRUE)'
```

Expected: `document()` adds six `export()` lines, `man/transforms.Rd`, and `transforms.R` to Collate; the tests pass with `SKIP 0`. If `local_mocked_bindings(.package = "Rtsne")` or `(.package = "uwot")` cannot rebind the function, report the error; stage 1 mocked `ranger::ranger` and `stats::predict` the same way. Then run the whole suite.

- [ ] Step 6: Mutation check

- In `StandardScaler()`, divide by `stats::sd()` columns (denominator n - 1): the first test fails.
- In `pca_scores()`, pass `center = FALSE` to both branches: the PCA test fails.
- In `TSNE()`, use `eta = max(n / 48, 50)`: the Rtsne settings test fails.
- In `TSNE()`, drop `stop_lying_iter = 250L`: the Rtsne settings test fails.
- In `UMAP()`, leave out `min_dist = min_dist`: the uwot settings test fails.

- [ ] Step 7: Commit

```bash
git add carve-r/DESCRIPTION carve-r/NAMESPACE carve-r/R/transforms.R carve-r/man/transforms.Rd \
  carve-r/tests/testthat/test-transforms.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add the preprocessing transforms Identity, StandardScaler, Log1p, PCA, TSNE and UMAP

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Preprocessing options, allocation and pipelines

Files:
- Create: `carve-r/R/pipeline.R`
- Generated: `carve-r/NAMESPACE`, `carve-r/man/preprocessing_option.Rd`, `carve-r/man/pipeline_from_spec.Rd`, the Collate field
- Test: `carve-r/tests/testthat/test-pipeline.R` (new)

Interfaces:
- Consumes: `pipeline_labels` fixture (Task 1); `Identity`, `StandardScaler`, `PCA` (Task 7); `seeded(kind =)` (Task 2); `as_data_matrix`, `format_param_value`, `format_repr` (stage 1).
- Produces:
  - exported `preprocessing_option(transform, ..., name = NULL)` returns a `carve_preprocessing_option`: `list(transform, name, grid)`, the grid in the order given.
  - exported `pipeline_from_spec(spec, random_state)` returns `function(X)`.
  - internal `step_label(name, params)`; `new_pipeline_spec(normalization, dim_reduction)` returns a `carve_pipeline_spec`: `list(normalization, dim_reduction, label)`, each step `list(transform, name, params, label)`; `allocate_pipelines(normalization_options, dim_reduction_options, n_resamples, random_state)` returns a list of `n_resamples` specs, entry `b + 1` for resample `b`.
  - S3 `print` methods for both classes.

- [ ] Step 1: Write the failing tests in `tests/testthat/test-pipeline.R`

```r
norm_options <- list(preprocessing_option(Identity), preprocessing_option(StandardScaler))
dr_options <- list(preprocessing_option(Identity), preprocessing_option(PCA, n_components = c(1L, 2L)))

test_that("step labels match Python's", {
  f <- read_fixture("pipeline_labels")
  for (case in f$steps) {
    expect_identical(step_label(case$name, case$params), case$label, info = case$label)
  }
  spec <- new_pipeline_spec(
    list(transform = Identity, name = "identity", params = list(), label = step_label("identity", list())),
    list(transform = PCA, name = "PCA", params = list(n_components = 10L), label = step_label("PCA", list(n_components = 10L)))
  )
  expect_identical(spec$label, f$spec$label)
})

test_that("an option is named after its transform", {
  expect_identical(preprocessing_option(Identity)$name, "identity")
  expect_identical(preprocessing_option(Log1p)$name, "log1p")
  expect_identical(preprocessing_option(StandardScaler)$name, "StandardScaler")
  expect_identical(preprocessing_option(CARVE::PCA, n_components = 2L)$name, "PCA")
  expect_identical(preprocessing_option(function(X) X, name = "mine")$name, "mine")
  expect_identical(names(preprocessing_option(TSNE, perplexity = 30, n_components = 2L)$grid), c("perplexity", "n_components"))
})

test_that("an option rejects what the transform cannot take", {
  expect_error(preprocessing_option(function(X) X), "Pass name= for a transform that is not a named function.", fixed = TRUE)
  expect_error(preprocessing_option(PCA, components = 2), "PCA has no argument 'components'.", fixed = TRUE)
  expect_error(preprocessing_option(PCA, n_components = integer(0)), "The candidate list for 'n_components' of PCA is empty.", fixed = TRUE)
  expect_error(preprocessing_option("PCA"), "transform must be a function.", fixed = TRUE)
})

test_that("every resample gets one pipeline, and pairs differ in count by at most one", {
  specs <- allocate_pipelines(norm_options, dr_options, 10L, 0L)
  expect_length(specs, 10L)
  expect_true(all(vapply(specs, inherits, logical(1), "carve_pipeline_spec")))
  pairs <- vapply(specs, function(s) paste(s$normalization$name, s$dim_reduction$name), character(1))
  counts <- table(factor(pairs, levels = c("identity identity", "identity PCA", "StandardScaler identity", "StandardScaler PCA")))
  expect_true(all(counts %in% c(2L, 3L)))
})

test_that("the allocation is shuffled, not cycled", {
  specs <- allocate_pipelines(norm_options, dr_options, 12L, 0L)
  pairs <- vapply(specs, function(s) paste(s$normalization$name, s$dim_reduction$name), character(1))
  cycled <- rep(c("identity identity", "identity PCA", "StandardScaler identity", "StandardScaler PCA"), 3)
  expect_false(identical(pairs, cycled))
})

test_that("hyperparameters are drawn from the candidate lists", {
  specs <- allocate_pipelines(norm_options, dr_options, 40L, 0L)
  drawn <- unlist(lapply(specs, function(s) s$dim_reduction$params$n_components))
  expect_setequal(drawn, c(1L, 2L))
  labels <- vapply(specs, function(s) s$label, character(1))
  expect_true("identity | PCA(n_components=1)" %in% labels)
  expect_true("StandardScaler | PCA(n_components=2)" %in% labels)
})

test_that("the allocation depends on the seed and works for one resample", {
  labels <- function(seed, n) vapply(allocate_pipelines(norm_options, dr_options, n, seed), function(s) s$label, character(1))
  expect_identical(labels(3L, 20L), labels(3L, 20L))
  expect_false(identical(labels(3L, 20L), labels(4L, 20L)))
  expect_length(labels(0L, 1L), 1L)
  set.seed(9)
  before <- .Random.seed
  labels(0L, 5L)
  expect_identical(.Random.seed, before)
})

test_that("allocation needs options on both sides and distinct labels", {
  expect_error(
    allocate_pipelines(list(), dr_options, 4L, 0L),
    "randomize_preprocessing=TRUE needs at least one normalization option and one dimensionality reduction option.",
    fixed = TRUE
  )
  twin <- list(preprocessing_option(function(X) X, name = "same"), preprocessing_option(function(X) X * 2, name = "same"))
  expect_error(
    allocate_pipelines(twin, list(preprocessing_option(Identity)), 8L, 0L),
    "Two different pipelines render as the same label 'same | identity'. Give the options distinct names with preprocessing_option(name = ).",
    fixed = TRUE
  )
})

test_that("a pipeline applies its normalization, then its reduction, with the seed", {
  seen <- new.env()
  seen$calls <- character()
  first <- function(X, random_state = NULL) {
    seen$calls <- c(seen$calls, paste("first", random_state))
    X * 2
  }
  second <- function(X, width = 1L) {
    seen$calls <- c(seen$calls, "second")
    X[, seq_len(width), drop = FALSE]
  }
  spec <- allocate_pipelines(
    list(preprocessing_option(first, random_state = 99L)),
    list(preprocessing_option(second, width = 1L)),
    1L, 0L
  )[[1L]]
  out <- pipeline_from_spec(spec, 7L)(matrix(1:6, 3))
  expect_identical(out, matrix(c(2, 4, 6), ncol = 1L))
  # CARVE's seed replaces the one drawn from the grid; a step without
  # random_state is not given one.
  expect_identical(seen$calls, c("first 7", "second"))
})

test_that("a pipeline is reproducible and leaves the session's stream alone", {
  noisy <- function(X, random_state = NULL) X + stats::runif(length(X))
  spec <- allocate_pipelines(list(preprocessing_option(Identity)), list(preprocessing_option(noisy)), 1L, 0L)[[1L]]
  set.seed(4)
  before <- .Random.seed
  a <- pipeline_from_spec(spec, 3L)(matrix(0, 4, 2))
  expect_identical(.Random.seed, before)
  expect_identical(a, pipeline_from_spec(spec, 3L)(matrix(0, 4, 2)))
  expect_false(identical(a, pipeline_from_spec(spec, 4L)(matrix(0, 4, 2))))
})

test_that("a step that returns the wrong shape is an error", {
  drop_row <- function(X) X[-1, , drop = FALSE]
  spec <- allocate_pipelines(list(preprocessing_option(Identity)), list(preprocessing_option(drop_row)), 1L, 0L)[[1L]]
  expect_error(
    pipeline_from_spec(spec, 0L)(matrix(0, 4, 2)),
    "A preprocessing function must return a numeric matrix with one row per row of X (4).",
    fixed = TRUE
  )
})

test_that("options and pipelines print their names", {
  expect_output(print(preprocessing_option(PCA, n_components = c(2L, 5L))), "Preprocessing option PCA", fixed = TRUE)
  expect_output(print(preprocessing_option(PCA, n_components = c(2L, 5L))), "n_components: 2, 5", fixed = TRUE)
  spec <- allocate_pipelines(norm_options[1L], dr_options[1L], 1L, 0L)[[1L]]
  expect_output(print(spec), "Preprocessing pipeline: identity | identity", fixed = TRUE)
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "pipeline")'`
Expected: failures, the functions not found.

- [ ] Step 3: Write `R/pipeline.R`

```r
# Preprocessing options and pipelines: allocation, application and labels.
# Mirrors _pipeline.py. A randomized fit draws one pipeline per resample; a
# pipeline records the two transforms and the values drawn for them, so it
# can be applied to any subset of the data with its own seed.

#' Preprocessing option
#'
#' Pairs a preprocessing transform with the values to draw from for each of
#' its arguments. Under `carve(randomize_preprocessing = TRUE)` each resample
#' gets one normalization option and one dimensionality reduction option, with
#' one value drawn for each argument. This is the R form of the
#' `(TransformerClass, param_grid)` options of the Python package.
#'
#' @param transform A function whose first argument is the data matrix, one
#'   row per sample, and which returns a numeric matrix with one row per
#'   sample, such as [PCA()]. If it has a `random_state` argument, CARVE
#'   passes it a seed derived from the resample.
#' @param ... Values to draw from, each named after an argument of
#'   `transform`: a vector, or a list for values that are not scalars.
#' @param name Name used in pipeline labels. Defaults to the name `transform`
#'   was passed under, with [Identity()] and [Log1p()] written `identity` and
#'   `log1p`, as in the Python package's labels. An anonymous function needs
#'   one.
#' @param x A `carve_preprocessing_option` object.
#' @return An object of class `carve_preprocessing_option`: a list with the
#'   function, its name and the values to draw from.
#' @seealso [transforms], [carve()], [preprocessing_results()]
#' @examples
#' preprocessing_option(PCA, n_components = c(2, 5, 10))
#' preprocessing_option(TSNE, n_components = 2, perplexity = c(15, 30))
#' @export
preprocessing_option <- function(transform, ..., name = NULL) {
  if (!is.function(transform)) {
    stop("transform must be a function.", call. = FALSE)
  }
  if (is.null(name)) {
    expr <- substitute(transform)
    if (is.symbol(expr)) {
      name <- as.character(expr)
    } else if (is.call(expr) && (identical(expr[[1L]], quote(`::`)) || identical(expr[[1L]], quote(`:::`)))) {
      name <- as.character(expr[[3L]])
    } else {
      stop("Pass name= for a transform that is not a named function.", call. = FALSE)
    }
    name <- switch(name, Identity = "identity", Log1p = "log1p", name)
  }
  grid <- list(...)
  if (length(grid) > 0L && (is.null(names(grid)) || any(names(grid) == ""))) {
    stop("Every grid entry must be named after an argument of the transform.", call. = FALSE)
  }
  arguments <- names(formals(transform))
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
  empty <- names(grid)[lengths(grid) == 0L]
  if (length(empty) > 0L) {
    stop(sprintf(
      "The candidate list for %s of %s is empty.",
      format_repr(empty[[1L]]), name
    ), call. = FALSE)
  }
  structure(list(transform = transform, name = name, grid = grid), class = "carve_preprocessing_option")
}

#' @rdname preprocessing_option
#' @export
print.carve_preprocessing_option <- function(x, ...) {
  cat("Preprocessing option ", x$name, "\n", sep = "")
  for (param in names(x$grid)) {
    values <- vapply(as.list(x$grid[[param]]), format_param_value, character(1))
    cat("  ", param, ": ", paste(values, collapse = ", "), "\n", sep = "")
  }
  invisible(x)
}

# The label of a drawn step, as Python's PipelineStep.label renders it: the
# name, then the drawn values in grid order.
step_label <- function(name, params) {
  if (length(params) == 0L) {
    return(name)
  }
  values <- vapply(params, format_param_value, character(1))
  paste0(name, "(", paste0(names(params), "=", values, collapse = ", "), ")")
}

new_pipeline_spec <- function(normalization, dim_reduction) {
  structure(
    list(
      normalization = normalization,
      dim_reduction = dim_reduction,
      label = paste(normalization$label, dim_reduction$label, sep = " | ")
    ),
    class = "carve_pipeline_spec"
  )
}

draw_step <- function(option) {
  params <- lapply(option$grid, function(candidates) {
    candidates <- as.list(candidates)
    candidates[[sample.int(length(candidates), 1L)]]
  })
  list(
    transform = option$transform,
    name = option$name,
    params = params,
    label = step_label(option$name, params)
  )
}

# One pipeline per resample. Every pair of a normalization and a reduction
# option gets floor or ceiling of n_resamples / n_pairs resamples, in a
# shuffled order, and each resample draws its own values. Everything comes
# from one stream seeded with random_state, so the allocation is the same
# for every configuration of a run.
allocate_pipelines <- function(normalization_options, dim_reduction_options, n_resamples,
                               random_state) {
  if (length(normalization_options) == 0L || length(dim_reduction_options) == 0L) {
    stop(
      "randomize_preprocessing=TRUE needs at least one normalization option and one dimensionality reduction option.",
      call. = FALSE
    )
  }
  n_reductions <- length(dim_reduction_options)
  n_pairs <- length(normalization_options) * n_reductions
  specs <- seeded(random_state, kind = "L'Ecuyer-CMRG", code = {
    # Pair p (0-based) is normalization p %/% n_reductions with reduction
    # p %% n_reductions, the order of Python's itertools.product. sample()
    # on a single number would permute 1:n, so the vector is indexed.
    cycle <- (seq_len(n_resamples) - 1L) %% n_pairs
    assigned <- cycle[sample.int(length(cycle))]
    lapply(assigned, function(pair) {
      normalization <- draw_step(normalization_options[[pair %/% n_reductions + 1L]])
      dim_reduction <- draw_step(dim_reduction_options[[pair %% n_reductions + 1L]])
      new_pipeline_spec(normalization, dim_reduction)
    })
  })
  # The label keys the results table, so two different pipelines with one
  # label would pool into a single row.
  labels <- vapply(specs, function(s) s$label, character(1))
  for (label in unique(labels)) {
    same <- specs[labels == label]
    if (!all(vapply(same, identical, logical(1), same[[1L]]))) {
      stop(sprintf(
        "Two different pipelines render as the same label %s. Give the options distinct names with preprocessing_option(name = ).",
        format_repr(label)
      ), call. = FALSE)
    }
  }
  specs
}

#' Apply a stored preprocessing pipeline
#'
#' Returns a function that runs one pipeline of a randomized fit on new data:
#' the normalization step, then the dimensionality reduction step, each with
#' the values drawn for it and with `random_state` when it has that argument.
#' Use it, for example, to embed the full data with the pipeline that scored
#' best in [preprocessing_results()].
#'
#' @param spec A pipeline from [preprocessing_pipelines()].
#' @param random_state Seed for both steps.
#' @param x A `carve_pipeline_spec` object.
#' @param ... Not used.
#' @return A function of one argument, the data matrix, that returns the
#'   transformed matrix.
#' @seealso [preprocessing_pipelines()], [preprocessing_option()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_resamples = 4, random_state = 0, randomize_preprocessing = TRUE,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3)),
#'              normalization_options = list(preprocessing_option(Identity)),
#'              dim_reduction_options = list(preprocessing_option(PCA, n_components = 1)))
#' spec <- preprocessing_pipelines(fit)[[1]]
#' spec
#' embed <- pipeline_from_spec(spec, random_state = 0)
#' dim(embed(X))
#' @export
pipeline_from_spec <- function(spec, random_state) {
  if (!inherits(spec, "carve_pipeline_spec")) {
    stop("spec must be a pipeline from preprocessing_pipelines().", call. = FALSE)
  }
  force(random_state)
  function(X) {
    seeded(random_state, {
      normalized <- apply_step(spec$normalization, as_data_matrix(X), random_state)
      apply_step(spec$dim_reduction, normalized, random_state)
    })
  }
}

#' @rdname pipeline_from_spec
#' @export
print.carve_pipeline_spec <- function(x, ...) {
  cat("Preprocessing pipeline: ", x$label, "\n", sep = "")
  invisible(x)
}

# CARVE's derived seed replaces a seed drawn from the grid, so every fit of
# a pipeline is seeded from its resample, as in Python.
apply_step <- function(step, X, random_state) {
  args <- c(list(X), step$params)
  if ("random_state" %in% names(formals(step$transform))) {
    args["random_state"] <- list(random_state)
  }
  out <- do.call(step$transform, args)
  if (is.data.frame(out)) {
    out <- as.matrix(out)
  }
  if (!is.matrix(out) || !is.numeric(out) || nrow(out) != nrow(X)) {
    stop(sprintf(
      "A preprocessing function must return a numeric matrix with one row per row of X (%d).",
      nrow(X)
    ), call. = FALSE)
  }
  out
}
```

The `pipeline_from_spec()` example calls `carve()` with arguments that Task 14 adds. Examples run only under `R CMD check` in Task 16, after Task 14, so the example stays as written.

- [ ] Step 4: Document, then run the tests and see them pass

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "pipeline", stop_on_failure = TRUE)'
```

Expected: PASS. NAMESPACE gains `export(pipeline_from_spec)`, `export(preprocessing_option)` and two `S3method(print, ...)` lines. Then run the whole suite.

- [ ] Step 5: Mutation check

- In `allocate_pipelines()`, use `assigned <- sample(cycle)`: the one-resample case of the seed test fails (`sample(0)` is empty).
- In `allocate_pipelines()`, use `cycle` without shuffling: the shuffle test fails.
- In `allocate_pipelines()`, drop `kind = "L'Ecuyer-CMRG"`: no test here fails; that is expected, since the kind only matters for its independence from the subsample stream, which Task 3 tests for the anchors. Report it.
- In `apply_step()`, keep a `random_state` drawn from the grid: the seed test fails (`first 99`).
- In `step_label()`, sort the parameter names: the fixture test fails on `TSNE(perplexity=12.5, n_components=2)`.
- In `preprocessing_option()`, skip the `Identity`/`Log1p` renaming: the naming test fails.

- [ ] Step 6: Commit

```bash
git add carve-r/DESCRIPTION carve-r/NAMESPACE carve-r/R/pipeline.R carve-r/man/preprocessing_option.Rd \
  carve-r/man/pipeline_from_spec.Rd carve-r/tests/testthat/test-pipeline.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add preprocessing options, pipeline allocation and pipeline_from_spec

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Default preprocessing options

Files:
- Modify: `carve-r/R/grids.R`
- Test: `carve-r/tests/testthat/test-grids.R`

Interfaces:
- Consumes: `preprocessing_option` (Task 8); `Identity`, `StandardScaler`, `Log1p`, `PCA`, `TSNE`, `UMAP` (Task 7); `has_package` (Task 2); `format_param_value` (stage 1).
- Produces (internal): `default_normalization_options(X)` and `default_dim_reduction_options(X, subsample_ratio = 0.6)`, each a list of `carve_preprocessing_option` objects; constants `PCA_COMPONENTS`, `TSNE_PERPLEXITIES`, `UMAP_NEIGHBORS`.

- [ ] Step 1: Write the failing tests at the end of `tests/testthat/test-grids.R`

```r
option_names <- function(options) vapply(options, function(o) o$name, character(1))

test_that("data without negative values get log1p among the normalizations", {
  expect_identical(option_names(default_normalization_options(matrix(c(0, 1, 2, 3), 2))), c("identity", "StandardScaler", "log1p"))
})

test_that("negative data leave log1p out with a warning", {
  options <- NULL
  expect_warning(
    options <- default_normalization_options(matrix(c(-1.5, 1, 2, 3), 2)),
    "X has negative values (minimum -1.5), so log1p is omitted from the default normalization options.",
    fixed = TRUE
  )
  expect_identical(option_names(options), c("identity", "StandardScaler"))
})

test_that("the reduction grids are filtered to the smallest subsample", {
  skip_if_not_installed("uwot")
  # 1,000 samples at ratio 0.618: subsamples of 618 and 382, so every
  # candidate fits.
  options <- default_dim_reduction_options(matrix(0, 1000, 100), 0.618)
  expect_identical(option_names(options), c("identity", "PCA", "TSNE", "UMAP"))
  expect_identical(options[[2L]]$grid$n_components, c(2L, 5L, 10L, 20L, 50L))
  expect_identical(options[[3L]]$grid, list(n_components = 2L, perplexity = c(15, 30, 50)))
  expect_identical(options[[4L]]$grid, list(n_components = 2L, n_neighbors = c(15L, 30L), min_dist = 0.1))
})

test_that("the limits come from the smaller subsample at low ratios", {
  skip_if_not_installed("uwot")
  # 200 samples at ratio 0.75: subsamples of 150 and 50. The held-out set
  # sets every limit: PCA below 50, perplexities up to (50 - 1) / 3.
  options <- default_dim_reduction_options(matrix(0, 200, 100), 0.75)
  expect_identical(options[[2L]]$grid$n_components, c(2L, 5L, 10L, 20L))
  expect_identical(options[[3L]]$grid$perplexity, 15)
  expect_identical(options[[4L]]$grid$n_neighbors, c(15L, 30L))
})

test_that("an option whose grid filters to nothing is left out with a warning", {
  skip_if_not_installed("uwot")
  out <- collect_warnings(default_dim_reduction_options(matrix(0, 30, 2), 0.618))
  # 30 samples: subsamples of 18 and 12.
  expect_identical(option_names(out$value), "identity")
  expect_identical(out$warnings, c(
    "PCA is omitted from the default dimensionality reduction options: no candidate n_components is below 2, the limit the smallest subsample sets.",
    "TSNE is omitted from the default dimensionality reduction options: no candidate perplexity is below 4, the limit the smallest subsample sets.",
    "UMAP is omitted from the default dimensionality reduction options: no candidate n_neighbors is below 12, the limit the smallest subsample sets."
  ))
})

test_that("without uwot, UMAP is left out with a warning", {
  local_mocked_bindings(has_package = function(package) package != "uwot")
  options <- NULL
  expect_warning(
    options <- default_dim_reduction_options(matrix(0, 1000, 100), 0.618),
    'uwot is not installed; UMAP is omitted from the default dimensionality reduction options. Install it with install.packages("uwot").',
    fixed = TRUE
  )
  expect_identical(option_names(options), c("identity", "PCA", "TSNE"))
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "grids")'`
Expected: failures, the functions not found.

- [ ] Step 3: Implement in `R/grids.R`

Append:

```r
# Candidate values of the default reduction grids, before they are filtered
# to what the smallest subsample supports.
PCA_COMPONENTS <- c(2L, 5L, 10L, 20L, 50L)
TSNE_PERPLEXITIES <- c(15, 30, 50)
UMAP_NEIGHBORS <- c(15L, 30L)

# Identity and standardization always; log1p only for data without negative
# values, since it is NaN below -1 and meant for counts.
default_normalization_options <- function(X) {
  options <- list(preprocessing_option(Identity), preprocessing_option(StandardScaler))
  x_min <- min(X)
  if (x_min >= 0) {
    return(c(options, list(preprocessing_option(Log1p))))
  }
  warning(sprintf(
    "X has negative values (minimum %s), so log1p is omitted from the default normalization options.",
    sprintf("%.3g", x_min)
  ), call. = FALSE)
  options
}

warn_omitted <- function(name, param, limit) {
  warning(sprintf(
    "%s is omitted from the default dimensionality reduction options: no candidate %s is below %s, the limit the smallest subsample sets.",
    name, param, format_param_value(limit)
  ), call. = FALSE)
}

# Identity, PCA, t-SNE and, with uwot, UMAP, over discrete grids filtered to
# the smaller of a resample's training subsample and held-out set, the
# smallest set a pipeline is fit on. Rtsne needs 3 * perplexity <= n - 1,
# stricter than scikit-learn's perplexity < n, so the t-SNE limit is
# floor((n_min - 1) / 3) + 1.
default_dim_reduction_options <- function(X, subsample_ratio = 0.6) {
  n_samples <- nrow(X)
  n_train <- floor(subsample_ratio * n_samples)
  n_min <- min(n_train, n_samples - n_train)
  options <- list(preprocessing_option(Identity))

  pca_limit <- min(n_min, ncol(X))
  components <- PCA_COMPONENTS[PCA_COMPONENTS < pca_limit]
  if (length(components) > 0L) {
    options <- c(options, list(preprocessing_option(PCA, n_components = components)))
  } else {
    warn_omitted("PCA", "n_components", pca_limit)
  }

  tsne_limit <- floor((n_min - 1) / 3) + 1
  perplexities <- TSNE_PERPLEXITIES[TSNE_PERPLEXITIES < tsne_limit]
  if (length(perplexities) > 0L) {
    options <- c(options, list(preprocessing_option(TSNE, n_components = 2L, perplexity = perplexities)))
  } else {
    warn_omitted("TSNE", "perplexity", tsne_limit)
  }

  if (!has_package("uwot")) {
    warning(
      'uwot is not installed; UMAP is omitted from the default dimensionality reduction options. Install it with install.packages("uwot").',
      call. = FALSE
    )
    return(options)
  }
  neighbors <- UMAP_NEIGHBORS[UMAP_NEIGHBORS < n_min]
  if (length(neighbors) == 0L) {
    warn_omitted("UMAP", "n_neighbors", n_min)
    return(options)
  }
  c(options, list(preprocessing_option(UMAP, n_components = 2L, n_neighbors = neighbors, min_dist = 0.1)))
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "grids", stop_on_failure = TRUE)'`
Expected: PASS with `SKIP 0`. Then run the whole suite.

- [ ] Step 5: Mutation check

- In `default_dim_reduction_options()`, use `n_min <- n_train`: the low-ratio test fails.
- In `default_dim_reduction_options()`, use scikit-learn's limit (`tsne_limit <- n_min`): the low-ratio test fails (it keeps 30 and 50).
- In `default_normalization_options()`, test `x_min > 0`: the first test fails, since its minimum is 0.

- [ ] Step 6: Commit

```bash
git add carve-r/R/grids.R carve-r/tests/testthat/test-grids.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add the default normalization and dimensionality reduction options

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---
### Task 10: One resample with the noise policy and embeddings

Files:
- Modify: `carve-r/R/runner.R`
- Test: `carve-r/tests/testthat/test-runner.R`

Interfaces:
- Consumes: `apply_noise_policy` (Task 2); `allocate_pipelines`, `pipeline_from_spec`, `preprocessing_option` (Task 8); `Identity` (Task 7); stage 1's `resample_indices`, `scale_neighbor_count`, `call_estimator`, `count_clusters`, `adjusted_rand_index`, `default_generalizability_classifier`, `format_params`, `collect_warnings`, `KMeans`.
- Produces (internal):
  - `validation_iter(X, estimator, estimator_name, params, subsample_ratio, n_resamples, seed, classifier = NULL, n_trees = 100L, sweep_param = "n_clusters", noise_policy = "drop", mode = "default", random_state = 0L, classifier_threads = 1L, embeddings = NULL)` returns stage 1's list plus `pipeline` (the pipeline label, or `NULL` without embeddings). Under `"drop"`, `train_indices`, `test_indices`, `stability_indices` and the labels exclude noise; `noise_fraction` is the share of noise in the first subsample before the policy.
  - `embed_resample(X, spec, seed, n_resamples, subsample_ratio, random_state, mode = "default")` returns `list(spec, X_1, X_2, X_test)`; `X_2` is `NULL` when stability is skipped and `X_test` when generalizability is.
  - `check_embeddings(policy, embeddings, idx)` stops when an embedding's row count differs from its subsample's.

- [ ] Step 1: Write the failing tests at the end of `tests/testthat/test-runner.R`

```r
# Labels the first five rows of every subsample noise and clusters the rest.
noisy_kmeans <- function(X, n_clusters = 3L, random_state = NULL) {
  labels <- KMeans(X, n_clusters, random_state = random_state)
  labels[1:5] <- -1L
  labels
}

identity_spec <- function() {
  allocate_pipelines(list(preprocessing_option(Identity)), list(preprocessing_option(Identity)), 1L, 0L)[[1L]]
}

test_that("drop removes noise from the indices the scores use", {
  r <- validation_iter(blobs$X, noisy_kmeans, "noisy", list(n_clusters = 3L), 0.618, 5L, 0L, random_state = 0L)
  expect_length(r$train_indices, 50L)
  expect_length(r$labels_train, 50L)
  expect_length(r$stability_indices, 50L)
  expect_length(r$test_indices, 30L)
  expect_length(r$labels_predicted, 30L)
  expect_false(any(r$labels_train < 0L))
  expect_equal(r$noise_fraction, 5 / 55)
})

test_that("the samples drop removes are the ones labeled noise", {
  kept <- validation_iter(blobs$X, noisy_kmeans, "noisy", list(n_clusters = 3L), 0.618, 5L, 0L,
                          noise_policy = "as_cluster", random_state = 0L)
  dropped <- validation_iter(blobs$X, noisy_kmeans, "noisy", list(n_clusters = 3L), 0.618, 5L, 0L,
                             random_state = 0L)
  expect_identical(dropped$train_indices, kept$train_indices[kept$labels_train >= 0L])
  expect_identical(sum(kept$labels_train == -1L), 5L)
  expect_identical(kept$n_clusters_train, 3L)
})

test_that("the noise fraction is the same under every policy", {
  # Under "singleton" each noise sample is a cluster of its own, so the
  # k-axis count check warns; those warnings are not what this test is about.
  fractions <- vapply(NOISE_POLICIES, function(policy) {
    suppressWarnings(validation_iter(blobs$X, noisy_kmeans, "noisy", list(n_clusters = 3L), 0.618, 5L, 0L,
                                     noise_policy = policy, random_state = 0L))$noise_fraction
  }, numeric(1))
  expect_equal(unname(fractions), rep(5 / 55, 3))
})

test_that("a subsample that is all noise warns and scores NaN", {
  all_noise <- function(X, min_cluster_size = 5L) rep(-1L, nrow(X))
  out <- collect_warnings(validation_iter(
    blobs$X, all_noise, "all_noise", list(min_cluster_size = 5L), 0.618, 5L, 0L,
    sweep_param = "min_cluster_size", random_state = 0L
  ))
  expect_identical(out$warnings, c(
    "all_noise with {'min_cluster_size': 5} produced 0 cluster(s) on a subsample; stability and generalizability are degenerate at this point on the sweep axis.",
    "All points in a subsample were labelled as noise and dropped (noise_policy='drop'); stability ARI is undefined for this resample. Consider relaxing the clustering parameters or switching to noise_policy='as_cluster'.",
    "All points in a subsample were labelled as noise and dropped (noise_policy='drop'); generalizability ARI is undefined for this resample. Consider relaxing the clustering parameters or switching to noise_policy='as_cluster'."
  ))
  r <- out$value
  expect_true(is.nan(r$ari_stability))
  expect_true(is.nan(r$ari_generalizability))
  expect_null(r$labels_predicted)
  expect_length(r$labels_train, 0L)
  expect_identical(r$noise_fraction, 1)
})

test_that("an unknown noise policy is an error", {
  expect_error(
    validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L, 0L,
                    noise_policy = "keep", random_state = 0L),
    "Unknown noise_policy 'keep'.",
    fixed = TRUE
  )
})

test_that("identity embeddings reproduce the raw resample", {
  emb <- embed_resample(blobs$X, identity_spec(), seed = 2L, n_resamples = 5L,
                        subsample_ratio = 0.618, random_state = 10L)
  raw <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L,
                         seed = 2L, random_state = 10L)
  embedded <- validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L,
                              seed = 2L, random_state = 10L, embeddings = emb)
  expect_null(raw$pipeline)
  expect_identical(embedded$pipeline, "identity | identity")
  raw$pipeline <- NULL
  embedded$pipeline <- NULL
  expect_identical(embedded, raw)
})

test_that("each subsample is embedded with its own seed", {
  seen <- new.env()
  seen$calls <- list()
  record <- function(X, random_state = NULL) {
    seen$calls[[length(seen$calls) + 1L]] <- c(nrow(X), random_state)
    X
  }
  spec <- allocate_pipelines(list(preprocessing_option(Identity)), list(preprocessing_option(record)), 1L, 0L)[[1L]]
  embed_resample(blobs$X, spec, seed = 2L, n_resamples = 5L, subsample_ratio = 0.618, random_state = 10L)
  calls <- do.call(rbind, seen$calls)
  # The first subsample, the second and the held-out set, in that order.
  expect_identical(calls[, 1], c(55L, 55L, 35L))
  expect_identical(calls[, 2], c(12L, 17L, 22L))
})

test_that("a mode embeds only the subsamples it uses", {
  emb <- embed_resample(blobs$X, identity_spec(), 0L, 5L, 0.618, 0L, mode = "stability")
  expect_null(emb$X_test)
  expect_identical(nrow(emb$X_2), 55L)
  emb <- embed_resample(blobs$X, identity_spec(), 0L, 5L, 0.618, 0L, mode = "generalizability")
  expect_null(emb$X_2)
  expect_identical(nrow(emb$X_test), 35L)
})

test_that("the estimator clusters the embedding and the classifier reads raw features", {
  seen <- new.env()
  first_column <- function(X) X[, 1L, drop = FALSE]
  spec <- allocate_pipelines(list(preprocessing_option(Identity)), list(preprocessing_option(first_column)), 1L, 0L)[[1L]]
  emb <- embed_resample(blobs$X, spec, 0L, 5L, 0.618, 0L)
  width <- function(X, n_clusters = 3L) {
    seen$width <- c(seen$width, ncol(X))
    rep_len(seq_len(n_clusters), nrow(X))
  }
  clf <- function(x_train, y_train, x_test) {
    seen$classifier <- ncol(x_train)
    rep(y_train[1], nrow(x_test))
  }
  validation_iter(blobs$X, width, "width", list(n_clusters = 3L), 0.618, 5L, 0L,
                  classifier = clf, random_state = 0L, embeddings = emb)
  expect_identical(seen$width, c(1L, 1L, 1L))
  expect_identical(seen$classifier, 2L)
})

test_that("embeddings that do not fit the resample are an error", {
  emb <- embed_resample(blobs$X, identity_spec(), 0L, 5L, 0.618, 0L)
  emb$X_test <- emb$X_test[-1L, , drop = FALSE]
  expect_error(
    validation_iter(blobs$X, KMeans, "KMeans", list(n_clusters = 3L), 0.618, 5L, 0L,
                    random_state = 0L, embeddings = emb),
    "Precomputed embedding X_test has 34 rows but this resample's subsample has 35. This is an internal CARVE error.",
    fixed = TRUE
  )
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "runner")'`
Expected: failures: unused arguments `noise_policy` and `embeddings`, `embed_resample` not found.

- [ ] Step 3: Implement in `R/runner.R`

Replace `validation_iter()` with the version below and add `ALL_NOISE_WARNING`, `check_embeddings()` and `embed_resample()` after it. Extend the comment above `resample_indices()` to `# The only place the subsample seeds are derived, so the embedding pass and` / `# the configuration loop draw the same subsamples.` (two lines).

```r
# Python's warning when the noise policy leaves a subsample empty.
ALL_NOISE_WARNING <- "All points in a subsample were labelled as noise and dropped (noise_policy='drop'); %s ARI is undefined for this resample. Consider relaxing the clustering parameters or switching to noise_policy='as_cluster'."

validation_iter <- function(X, estimator, estimator_name, params, subsample_ratio,
                            n_resamples, seed, classifier = NULL, n_trees = 100L,
                            sweep_param = "n_clusters", noise_policy = "drop",
                            mode = "default", random_state = 0L, classifier_threads = 1L,
                            embeddings = NULL) {
  policy <- resolve_mode(mode)
  n <- nrow(X)
  idx <- resample_indices(n, seed, n_resamples, subsample_ratio, random_state, policy$run_stability)

  # What the estimator clusters: the raw subsamples, or under randomized
  # preprocessing this resample's embeddings, one independent fit per
  # subsample.
  pipeline <- NULL
  if (is.null(embeddings)) {
    X_1 <- X[idx$train, , drop = FALSE]
    X_test <- if (policy$run_generalizability) X[idx$test, , drop = FALSE] else NULL
    X_2 <- if (policy$run_stability) X[idx$stability, , drop = FALSE] else NULL
  } else {
    check_embeddings(policy, embeddings, idx)
    pipeline <- embeddings$spec$label
    X_1 <- embeddings$X_1
    X_test <- embeddings$X_test
    X_2 <- embeddings$X_2
  }

  # Each set is clustered on its own, with the neighbor count scaled to its
  # own rows.
  fit_labels <- function(X_fit) {
    scaled <- scale_neighbor_count(estimator, params, n_fit = nrow(X_fit), n_full = n)
    call_estimator(estimator, X_fit, scaled, random_state = random_state + seed)
  }
  labels_train <- fit_labels(X_1)
  labels_test <- if (policy$run_generalizability) fit_labels(X_test) else NULL
  labels_stability <- if (policy$run_stability) fit_labels(X_2) else NULL

  # Noise labels. Under "drop" the index vectors shrink with the labels. The
  # classifier reads raw features through these indices and the embeddings
  # are not used again, so nothing else needs re-slicing.
  train <- apply_noise_policy(idx$train, labels_train, noise_policy)
  idx$train <- train$indices
  labels_train <- train$labels
  if (policy$run_generalizability) {
    test <- apply_noise_policy(idx$test, labels_test, noise_policy)
    idx$test <- test$indices
    labels_test <- test$labels
  }
  if (policy$run_stability) {
    second <- apply_noise_policy(idx$stability, labels_stability, noise_policy)
    idx$stability <- second$indices
    labels_stability <- second$labels
  }

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
    if (length(idx$train) == 0L || length(idx$stability) == 0L) {
      warning(sprintf(ALL_NOISE_WARNING, "stability"), call. = FALSE)
    } else {
      shared <- intersect(idx$train, idx$stability)
      ari_stability <- adjusted_rand_index(
        labels_train[match(shared, idx$train)],
        labels_stability[match(shared, idx$stability)]
      )
    }
  }

  # Generalizability: the classifier learns the first subsample's clusters
  # from its raw features and predicts the held-out samples. It reads raw
  # features also under randomized preprocessing: a cluster generalizes only
  # if it can be learned from the data, and a transform such as t-SNE cannot
  # embed new samples anyway.
  labels_predicted <- NULL
  ari_generalizability <- NaN
  if (policy$run_generalizability) {
    if (length(idx$train) == 0L || length(idx$test) == 0L) {
      warning(sprintf(ALL_NOISE_WARNING, "generalizability"), call. = FALSE)
    } else {
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
    pipeline = pipeline,
    n_clusters_train = k_train,
    n_clusters_test = k_test,
    n_clusters_stability = k_stability,
    noise_fraction = train$noise_fraction
  )
}

# The embedding pass and validation_iter() derive the same indices from
# resample_indices(); a row count that differs means that has broken.
check_embeddings <- function(policy, embeddings, idx) {
  expected <- list(X_1 = idx$train)
  if (policy$run_stability) {
    expected$X_2 <- idx$stability
  }
  if (policy$run_generalizability) {
    expected$X_test <- idx$test
  }
  for (name in names(expected)) {
    rows <- if (is.null(embeddings[[name]])) "None" else nrow(embeddings[[name]])
    if (!identical(rows, length(expected[[name]]))) {
      stop(sprintf(
        "Precomputed embedding %s has %s rows but this resample's subsample has %d. This is an internal CARVE error.",
        name, rows, length(expected[[name]])
      ), call. = FALSE)
    }
  }
  invisible(TRUE)
}

# Fits resample `seed`'s pipeline on each subsample its mode uses: the first
# with random_state + seed, the second with random_state + seed +
# n_resamples and the held-out set with random_state + seed + 2 *
# n_resamples, so no two fits share a seed.
embed_resample <- function(X, spec, seed, n_resamples, subsample_ratio, random_state,
                           mode = "default") {
  policy <- resolve_mode(mode)
  idx <- resample_indices(nrow(X), seed, n_resamples, subsample_ratio, random_state, policy$run_stability)
  base <- random_state + seed
  embed <- function(rows, offset) {
    pipeline_from_spec(spec, base + offset)(X[rows, , drop = FALSE])
  }
  list(
    spec = spec,
    X_1 = embed(idx$train, 0L),
    X_2 = if (policy$run_stability) embed(idx$stability, n_resamples) else NULL,
    X_test = if (policy$run_generalizability) embed(idx$test, 2L * n_resamples) else NULL
  )
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "runner", stop_on_failure = TRUE)'`
Expected: PASS, the stage 1 runner tests included. Then run the whole suite.

- [ ] Step 5: Mutation check

- In `validation_iter()`, train the classifier on `X_1` and let it predict `X_test` (the clustered matrices instead of the raw features): the raw-features test fails.
- In `validation_iter()`, return `noise_fraction = mean(labels_train < 0)` after the policy: the drop test fails.
- In `validation_iter()`, skip the policy on the held-out set: the drop test fails on `test_indices`.
- In `embed_resample()`, embed the held-out set with offset `n_resamples`: the seed test fails.
- In `check_embeddings()`, check only `X_1`: the mismatch test fails.

- [ ] Step 6: Commit

```bash
git add carve-r/R/runner.R carve-r/tests/testthat/test-runner.R
git commit -m "$(cat <<'EOF'
feat(carve-r): resolve noise labels and cluster precomputed embeddings in each resample

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: The configuration loop with anchors, the noise policy and the core budget

Files:
- Modify: `carve-r/R/runner.R`
- Test: `carve-r/tests/testthat/test-runner.R`

Interfaces:
- Consumes: `validation_iter` (Task 10); `consensus_anchor_block`, `stability_from_runs_anchored` (Task 3); stage 1's `resolve_core_budget`, `n_cores`, `resample_bpparam`, `resample_backend`, `compute_consensus_matrix`, `stability_from_consensus`, `compute_consensus_pac`, `compute_generalizability_scores`.
- Produces (internal):
  - `run_core_budget(n_jobs, n_resamples, BPPARAM = NULL)` returns `c(outer =, inner =)`.
  - `run_resample(task, data, estimator, estimator_name, params, subsample_ratio, n_resamples, classifier, n_trees, sweep_param, noise_policy, mode, random_state, classifier_threads)`, where `task` is `list(b, embeddings)`.
  - `run_validation(X, estimator_grids, n_resamples, subsample_ratio, classifier = NULL, n_trees = 100L, n_jobs = 1L, BPPARAM = NULL, random_state = 0L, sweep, noise_policy = "drop", mode = "default", anchors = NULL, show_progress = FALSE, verbose = 0L)`: stage 1's return value. Under `anchors`, the consensus matrices are `m` by `m` anchor blocks and the summaries' `gini` and `ce` still have one entry per sample.

- [ ] Step 1: Write the failing tests at the end of `tests/testthat/test-runner.R`

```r
test_that("an anchored run keeps anchor blocks and full-length scores", {
  anchors <- seq(1L, 90L, by = 3L)
  run <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep, anchors = anchors)
  expect_identical(dim(run$consensus_matrices[["1"]]), c(30L, 30L))
  expect_identical(dim(run$consensus_generalizability_matrices[["1"]]), c(30L, 30L))
  expect_length(run$summaries[["1"]]$gini, 90L)
  expect_length(run$summaries[["1"]]$ce, 90L)
  expect_length(run$generalizability_scores[["1"]], 90L)
  expect_identical(run$summaries[["1"]]$pac, compute_consensus_pac(run$consensus_matrices[["1"]]))
})

test_that("an anchored block is the exact matrix over the anchors", {
  anchors <- seq(2L, 90L, by = 4L)
  exact <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep)
  anchored <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep, anchors = anchors)
  expect_equal(anchored$consensus_matrices[["2"]], exact$consensus_matrices[["2"]][anchors, anchors])
  expect_identical(anchored$records, exact$records)
})

test_that("with every sample an anchor the run equals the exact run", {
  exact <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep)
  everything <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep, anchors = 1:90)
  expect_equal(everything$summaries, exact$summaries)
  expect_equal(everything$consensus_matrices, exact$consensus_matrices)
})

test_that("the noise policy reaches every resample", {
  grid <- list(estimator_grid(noisy_kmeans, n_clusters = 3L))
  axis <- resolve_sweep(n_clusters = 3L)
  dropped <- run_validation(blobs$X, grid, 3L, 0.618, random_state = 0L, sweep = axis)
  kept <- run_validation(blobs$X, grid, 3L, 0.618, random_state = 0L, sweep = axis, noise_policy = "as_cluster")
  expect_equal(dropped$records$noise_fraction, 5 / 55)
  expect_equal(kept$records$noise_fraction, 5 / 55)
  expect_false(identical(dropped$consensus_matrices, kept$consensus_matrices))
})

test_that("resamples that are all noise are left out of the aggregates", {
  # Resample 1 labels every sample noise; resamples 0 and 2 cluster.
  sometimes_noise <- function(X, min_cluster_size = 5L, random_state = NULL) {
    if (identical(random_state, 1L)) {
      return(rep(-1L, nrow(X)))
    }
    KMeans(X, 3L, random_state = random_state)
  }
  out <- collect_warnings(run_validation(
    blobs$X, list(estimator_grid(sometimes_noise, min_cluster_size = 5L)), 3L, 0.618,
    random_state = 0L, sweep = resolve_sweep(sweep = "min_cluster_size", sweep_values = 5L)
  ))
  run <- out$value
  expect_length(out$warnings, 3L)
  expect_equal(run$records$noise_fraction, 1 / 3)
  expect_false(is.nan(run$records$ari_stability))
  expect_false(all(is.nan(run$consensus_matrices[["0"]])))
  expect_false(anyNA(run$generalizability_scores[["0"]]))
})

test_that("with a BPPARAM the thread split counts only workers that get a resample", {
  local_mocked_bindings(n_cores = function() 8L)
  expect_identical(run_core_budget(1L, 2L, BiocParallel::SnowParam(4L)), c(outer = 4L, inner = 4L))
  expect_identical(run_core_budget(1L, 10L, BiocParallel::SnowParam(4L)), c(outer = 4L, inner = 2L))
  expect_identical(run_core_budget(4L, 10L), resolve_core_budget(4L, 10L))
})

test_that("an anchored run with noise does not depend on the backend", {
  skip_on_os("windows")
  grid <- list(estimator_grid(noisy_kmeans, n_clusters = 3L))
  axis <- resolve_sweep(n_clusters = 3L)
  serial <- run_validation(blobs$X, grid, 4L, 0.618, random_state = 0L, sweep = axis, anchors = seq(1L, 90L, by = 2L))
  forked <- run_validation(blobs$X, grid, 4L, 0.618, random_state = 0L, sweep = axis, anchors = seq(1L, 90L, by = 2L),
                           BPPARAM = BiocParallel::MulticoreParam(2L))
  expect_identical(serial, forked)
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "runner")'`
Expected: failures: unused arguments `anchors` and `noise_policy`, `run_core_budget` not found.

- [ ] Step 3: Implement in `R/runner.R`

Replace `run_resample()` and its comment:

```r
# One resample of one configuration, described by its task: the resample
# index b and, under randomized preprocessing, that resample's embeddings.
# It is a namespace function, so a worker receives these arguments and
# nothing else; a closure made inside run_validation would carry that frame,
# with every consensus matrix computed so far, to every worker. The data
# argument is not called X, which would clash with the X of lapply and
# bplapply.
# The resample runs under the seed random_state + b, so an estimator or
# classifier that draws without a random_state argument gives the same
# results on every backend and leaves the caller's stream alone. The
# built-in functions reseed themselves inside it.
run_resample <- function(task, data, estimator, estimator_name, params, subsample_ratio,
                         n_resamples, classifier, n_trees, sweep_param, noise_policy, mode,
                         random_state, classifier_threads) {
  seeded(random_state + task$b, collect_warnings(validation_iter(
    data, estimator, estimator_name, params,
    subsample_ratio = subsample_ratio,
    n_resamples = n_resamples,
    seed = task$b,
    classifier = classifier,
    n_trees = n_trees,
    sweep_param = sweep_param,
    noise_policy = noise_policy,
    mode = mode,
    random_state = random_state,
    classifier_threads = classifier_threads,
    embeddings = task$embeddings
  )))
}
```

Add before `resample_bpparam()`:

```r
# Workers over resamples (outer) and classifier threads per worker (inner).
# A BPPARAM brings its own worker count; as with n_jobs, at most
# n_resamples of them have a resample, so the threads are shared among
# those.
run_core_budget <- function(n_jobs, n_resamples, BPPARAM = NULL) {
  if (is.null(BPPARAM)) {
    return(resolve_core_budget(n_jobs, n_resamples))
  }
  workers <- as.integer(BiocParallel::bpnworkers(BPPARAM))
  busy <- max(1L, min(workers, as.integer(n_resamples)))
  c(outer = workers, inner = max(1L, n_cores() %/% busy))
}
```

Replace `run_validation()` with:

```r
run_validation <- function(X, estimator_grids, n_resamples, subsample_ratio, classifier = NULL,
                           n_trees = 100L, n_jobs = 1L, BPPARAM = NULL, random_state = 0L,
                           sweep, noise_policy = "drop", mode = "default", anchors = NULL,
                           show_progress = FALSE, verbose = 0L) {
  policy <- resolve_mode(mode)
  n <- nrow(X)
  budget <- run_core_budget(n_jobs, n_resamples, BPPARAM)
  # The backend starts once for the whole run. Left to bplapply, a backend
  # that is not running starts and stops for every configuration, and a
  # SnowParam launches new R processes each time. A backend the caller
  # already started stays up.
  BPPARAM <- resample_bpparam(budget[["outer"]], BPPARAM)
  if (!is.null(BPPARAM) && !BiocParallel::bpisup(BPPARAM)) {
    BiocParallel::bpstart(BPPARAM)
    on.exit(BiocParallel::bpstop(BPPARAM), add = TRUE)
  }
  apply_resamples <- resample_backend(BPPARAM)

  # One task per resample; a worker receives its own tasks only.
  tasks <- lapply(seq_len(n_resamples) - 1L, function(b) list(b = b, embeddings = NULL))

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
    outputs <- apply_resamples(
      tasks, run_resample,
      data = X,
      estimator = g$estimator,
      estimator_name = g$name,
      params = params,
      subsample_ratio = subsample_ratio,
      n_resamples = n_resamples,
      classifier = classifier,
      n_trees = n_trees,
      sweep_param = sweep@param,
      noise_policy = noise_policy,
      mode = mode,
      random_state = random_state,
      classifier_threads = budget[["inner"]]
    )
    for (text in unique(unlist(lapply(outputs, function(o) o$warnings)))) {
      warning(text, call. = FALSE)
    }
    results <- lapply(outputs, function(o) o$value)

    # A resample whose subsample was all noise under "drop" has nothing to
    # aggregate; it already warned.
    stability_runs <- Filter(function(r) length(r$labels_train) > 0L, results)
    held_out_runs <- Filter(
      function(r) length(r$labels_predicted) > 0L && length(r$labels_test) > 0L,
      results
    )
    stability_pairs <- lapply(stability_runs, function(r) {
      list(indices = r$train_indices, labels = r$labels_train)
    })
    held_out_pairs <- lapply(held_out_runs, function(r) {
      list(indices = r$test_indices, labels = r$labels_predicted)
    })
    if (policy$run_stability) {
      # Anchored, the stored matrix is the anchor block, and PAC comes from
      # it; the per-sample scores still cover every sample.
      if (is.null(anchors)) {
        M <- compute_consensus_matrix(n, stability_pairs)
        scores <- stability_from_consensus(M)
      } else {
        M <- consensus_anchor_block(n, stability_pairs, anchors)
        scores <- stability_from_runs_anchored(n, stability_pairs, anchors)
      }
      consensus[i] <- list(M)
      summaries[i] <- list(list(gini = scores$gini, ce = scores$ce, pac = compute_consensus_pac(M)))
    }
    if (policy$run_generalizability) {
      consensus_generalizability[i] <- list(if (is.null(anchors)) {
        compute_consensus_matrix(n, held_out_pairs)
      } else {
        consensus_anchor_block(n, held_out_pairs, anchors)
      })
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

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "runner", stop_on_failure = TRUE)'`
Expected: PASS, the stage 1 tests included (the worker-size test still measures the dots of each call). Then run the whole suite: `fit_carve()` calls `run_validation()` by name and is unaffected.

- [ ] Step 5: Mutation check

- In `run_validation()`, compute anchored scores with `stability_from_consensus(M)`: the full-length test fails.
- In `run_validation()`, leave `noise_policy` out of the `apply_resamples()` call: the noise-policy test fails.
- In `run_validation()`, build the generalizability matrix with `compute_consensus_matrix()` on anchored runs too: the first anchored test fails on its dimensions.
- In `run_core_budget()`, use `busy <- workers`: the budget test fails.

The two `Filter()` calls cannot be caught this way: an empty run adds nothing to the counts, and Task 2's guard in `align_cluster_labels()` makes an empty held-out run harmless. The all-noise test checks that such a run completes with finite aggregates, as Python's test does.

- [ ] Step 6: Commit

```bash
git add carve-r/R/runner.R carve-r/tests/testthat/test-runner.R
git commit -m "$(cat <<'EOF'
feat(carve-r): run anchored consensus and the noise policy in the configuration loop

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Randomized preprocessing in the configuration loop

Files:
- Modify: `carve-r/R/runner.R`, `carve-r/R/utils.R`
- Test: `carve-r/tests/testthat/test-runner.R`, `carve-r/tests/testthat/test-utils.R`

Interfaces:
- Consumes: `allocate_pipelines`, `new_pipeline_spec`, `preprocessing_option` (Task 8); `embed_resample` (Task 10); `run_validation` (Task 11); `Identity`, `StandardScaler`, `PCA` (Task 7); `summarize_ari_scores` (stage 1).
- Produces (internal):
  - `run_validation()` gains `randomize_preprocessing = FALSE, normalization_options = list(), dim_reduction_options = list()` (after `anchors`, before `show_progress`) and returns two more entries: `pipeline_records`, a list with one entry per configuration (`method_id`, `method_label`, `sweep_value`, `sweep_rank`, and `runs`, a data frame with columns `pipeline`, `ari_stability`, `ari_generalizability`, `n_clusters`, one row per resample), and `pipelines`, the allocated specs named by label in order of first use. Both are `NULL` without randomization.
  - `precompute_embeddings(X, pipelines, subsample_ratio, random_state, mode, apply_resamples, show_progress = FALSE)` returns one `embed_resample()` result per resample; `run_embedding(task, data, n_resamples, subsample_ratio, random_state, mode)`.
  - `summarize_preprocessing_records(pipeline_records, pipelines, sweep_param = "n_clusters")` in `utils.R` returns the `preprocessing_results` data frame.

- [ ] Step 1: Write the failing tests

At the end of `tests/testthat/test-runner.R`:

```r
norm_opts <- list(preprocessing_option(Identity), preprocessing_option(StandardScaler))
dr_opts <- list(preprocessing_option(Identity), preprocessing_option(PCA, n_components = 1L))

randomized_run <- function(...) {
  run_validation(blobs$X, k_grid, 8L, 0.618, random_state = 0L, sweep = k_sweep,
                 randomize_preprocessing = TRUE, normalization_options = norm_opts,
                 dim_reduction_options = dr_opts, ...)
}

test_that("a randomized run records each resample's pipeline", {
  run <- randomized_run()
  expect_length(run$pipeline_records, 3L)
  record <- run$pipeline_records[[2L]]
  expect_identical(record$method_id, "m0")
  expect_identical(record$sweep_value, 3L)
  expect_identical(record$sweep_rank, 1L)
  expect_identical(names(record$runs), c("pipeline", "ari_stability", "ari_generalizability", "n_clusters"))
  expect_identical(nrow(record$runs), 8L)
  expect_identical(
    sort(names(run$pipelines), method = "radix"),
    c("StandardScaler | PCA(n_components=1)", "StandardScaler | identity",
      "identity | PCA(n_components=1)", "identity | identity")
  )
  # Eight resamples over four pairs: two each, in every configuration alike.
  expect_identical(as.vector(table(record$runs$pipeline)), rep(2L, 4L))
  expect_identical(run$pipeline_records[[1L]]$runs$pipeline, run$pipeline_records[[3L]]$runs$pipeline)
})

test_that("each resample's task carries that resample's embeddings", {
  seen <- list()
  local_mocked_bindings(resample_backend = function(...) {
    function(X, FUN, ...) {
      seen[[length(seen) + 1L]] <<- X
      lapply(X, FUN, ...)
    }
  })
  run <- randomized_run()
  # The first call is the embedding pass; the configuration calls follow.
  tasks <- seen[[2L]]
  expect_identical(vapply(tasks, function(t) t$b, integer(1)), 0:7)
  expect_identical(
    vapply(tasks, function(t) t$embeddings$spec$label, character(1)),
    run$pipeline_records[[1L]]$runs$pipeline
  )
})

test_that("an identity pipeline gives the records of a run without preprocessing", {
  identity <- list(preprocessing_option(Identity))
  raw <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep)
  randomized <- run_validation(blobs$X, k_grid, 4L, 0.618, random_state = 0L, sweep = k_sweep,
                               randomize_preprocessing = TRUE, normalization_options = identity,
                               dim_reduction_options = identity)
  expect_identical(randomized$records, raw$records)
  expect_identical(randomized$consensus_matrices, raw$consensus_matrices)
})

test_that("a run without randomization has no pipeline records", {
  run <- run_validation(blobs$X, k_grid, 2L, 0.618, random_state = 0L, sweep = k_sweep)
  expect_null(run$pipeline_records)
  expect_null(run$pipelines)
})

test_that("a randomized run does not depend on the backend", {
  skip_on_os("windows")
  expect_identical(randomized_run(), randomized_run(BPPARAM = BiocParallel::MulticoreParam(2L)))
})

test_that("show_progress draws a bar for the embedding pass too", {
  count_full <- function(run) {
    text <- paste(capture.output(invisible(run), type = "message"), collapse = "")
    lengths(regmatches(text, gregexpr("100%", text, fixed = TRUE)))
  }
  expect_identical(count_full(run_validation(blobs$X, k_grid, 2L, 0.618, random_state = 0L,
                                             sweep = k_sweep, show_progress = TRUE)), 1L)
  expect_identical(count_full(randomized_run(show_progress = TRUE)), 2L)
})
```

At the end of `tests/testthat/test-utils.R`:

```r
test_that("the preprocessing summary splits each configuration by pipeline", {
  step <- function(label) list(transform = Identity, name = label, params = list(), label = label)
  pipelines <- list(
    "a | x" = new_pipeline_spec(step("a"), step("x")),
    "b | x" = new_pipeline_spec(step("b"), step("x"))
  )
  records <- list(
    list(method_id = "m10", method_label = "KMeans", sweep_value = 3L, sweep_rank = 1L,
         runs = data.frame(pipeline = c("b | x", "a | x", "b | x"), ari_stability = c(0.5, 1, 0.7),
                           ari_generalizability = c(0.2, 0.4, NaN), n_clusters = c(3, 3, 2),
                           stringsAsFactors = FALSE)),
    list(method_id = "m2", method_label = "Ward", sweep_value = 2L, sweep_rank = 0L,
         runs = data.frame(pipeline = "a | x", ari_stability = 0.9, ari_generalizability = 0.8,
                           n_clusters = 2, stringsAsFactors = FALSE))
  )
  out <- summarize_preprocessing_records(records, pipelines, "n_clusters")
  expect_identical(names(out), c(
    "method_id", "method_label", "pipeline", "normalization", "dim_reduction", "n_clusters",
    "n_resamples", "ari_stability", "ari_stability_se", "ari_generalizability",
    "ari_generalizability_se", "n_clusters_observed", "sweep_param", "sweep_value", "sweep_rank"
  ))
  # m2 sorts before m10 by its number.
  expect_identical(out$method_id, c("m2", "m10", "m10"))
  expect_identical(out$pipeline, c("a | x", "a | x", "b | x"))
  expect_identical(out$normalization, c("a", "a", "b"))
  expect_identical(out$n_clusters, c(2L, 3L, 3L))
  expect_identical(out$n_resamples, c(1L, 1L, 2L))
  expect_equal(out$ari_stability, c(0.9, 1, 0.6))
  expect_equal(out$ari_stability_se[3], stats::sd(c(0.5, 0.7)) / sqrt(2))
  expect_true(is.nan(out$ari_stability_se[1]))
  expect_equal(out$ari_generalizability[3], 0.2)
  expect_equal(out$n_clusters_observed, c(2, 3, 2.5))
  expect_identical(out$sweep_param, rep("n_clusters", 3))
  expect_identical(rownames(out), c("1", "2", "3"))
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "runner|utils")'`
Expected: failures: unused argument `randomize_preprocessing`, `summarize_preprocessing_records` not found.

- [ ] Step 3: Implement the summary in `R/utils.R`

Append:

```r
# One row per configuration, pipeline and sweep value of a randomized run.
# The scores are over the resamples that used the row's pipeline at the
# row's configuration only, and n_resamples counts them. Rows are sorted by
# the number in method_id (m10 after m9), then pipeline, then sweep value.
summarize_preprocessing_records <- function(pipeline_records, pipelines, sweep_param = "n_clusters") {
  rows <- list()
  for (record in pipeline_records) {
    runs <- record$runs
    for (label in unique(runs$pipeline)) {
      own <- runs[runs$pipeline == label, , drop = FALSE]
      spec <- pipelines[[label]]
      stability <- summarize_ari_scores(own$ari_stability)
      generalizability <- summarize_ari_scores(own$ari_generalizability)
      row <- list(
        method_id = record$method_id,
        method_label = record$method_label,
        pipeline = label,
        normalization = spec$normalization$label,
        dim_reduction = spec$dim_reduction$label
      )
      row[[sweep_param]] <- record$sweep_value
      rows[[length(rows) + 1L]] <- c(row, list(
        n_resamples = nrow(own),
        ari_stability = stability[["mean"]],
        ari_stability_se = stability[["se"]],
        ari_generalizability = generalizability[["mean"]],
        ari_generalizability_se = generalizability[["se"]],
        n_clusters_observed = mean(own$n_clusters),
        sweep_param = sweep_param,
        sweep_value = record$sweep_value,
        sweep_rank = record$sweep_rank
      ))
    }
  }
  summary <- do.call(rbind, lapply(rows, function(row) {
    data.frame(row, check.names = FALSE, stringsAsFactors = FALSE)
  }))
  order_by <- order(
    as.integer(substring(summary$method_id, 2L)),
    summary$pipeline,
    summary$sweep_value,
    method = "radix"
  )
  summary <- summary[order_by, , drop = FALSE]
  rownames(summary) <- NULL
  summary
}
```

- [ ] Step 4: Implement the randomized run in `R/runner.R`

Add after `embed_resample()`:

```r
# Embeds every resample's subsamples once, before any configuration runs,
# with the backend the configurations use. A worker receives one resample's
# pipeline and returns its embeddings, which are kept for the whole run.
precompute_embeddings <- function(X, pipelines, subsample_ratio, random_state, mode,
                                  apply_resamples, show_progress = FALSE) {
  n_resamples <- length(pipelines)
  tasks <- lapply(seq_len(n_resamples), function(i) list(b = i - 1L, spec = pipelines[[i]]))
  bar <- NULL
  if (show_progress) {
    bar <- utils::txtProgressBar(min = 0, max = n_resamples, style = 3, file = stderr())
    on.exit(close(bar), add = TRUE)
  }
  outputs <- apply_resamples(
    tasks, run_embedding,
    data = X,
    n_resamples = n_resamples,
    subsample_ratio = subsample_ratio,
    random_state = random_state,
    mode = mode
  )
  if (!is.null(bar)) {
    utils::setTxtProgressBar(bar, n_resamples)
  }
  for (text in unique(unlist(lapply(outputs, function(o) o$warnings)))) {
    warning(text, call. = FALSE)
  }
  lapply(outputs, function(o) o$value)
}

run_embedding <- function(task, data, n_resamples, subsample_ratio, random_state, mode) {
  collect_warnings(embed_resample(
    data, task$spec, task$b, n_resamples, subsample_ratio, random_state, mode = mode
  ))
}
```

In `run_validation()`:

1. Add the three arguments to the signature, after `anchors = NULL`: `randomize_preprocessing = FALSE, normalization_options = list(), dim_reduction_options = list(),`.
2. Replace the `tasks <- ...` line and its comment with:

```r
  # Under randomized preprocessing one pipeline is allocated per resample
  # and embedded once, before the configurations. Nothing here is per
  # configuration, so config_id is unaffected.
  embeddings <- NULL
  registry <- NULL
  if (randomize_preprocessing) {
    pipelines <- allocate_pipelines(normalization_options, dim_reduction_options, n_resamples, random_state)
    labels <- vapply(pipelines, function(s) s$label, character(1))
    first_use <- !duplicated(labels)
    registry <- stats::setNames(pipelines[first_use], labels[first_use])
    embeddings <- precompute_embeddings(
      X, pipelines, subsample_ratio, random_state, mode, apply_resamples,
      show_progress = show_progress
    )
  }
  # One task per resample; a worker receives its own tasks only, with that
  # resample's embeddings.
  tasks <- lapply(seq_len(n_resamples), function(i) {
    list(b = i - 1L, embeddings = embeddings[[i]])
  })
```

3. After `summaries <- ...` add `pipeline_records <- vector("list", total)`.
4. After `records[[i]] <- record`, add:

```r
    if (randomize_preprocessing) {
      pipeline_records[[i]] <- list(
        method_id = method[[1L]],
        method_label = method[[2L]],
        sweep_value = sweep_value,
        sweep_rank = record$sweep_rank,
        runs = data.frame(
          pipeline = vapply(results, function(r) r$pipeline, character(1)),
          ari_stability = aris_stability,
          ari_generalizability = aris_generalizability,
          n_clusters = k_observed,
          stringsAsFactors = FALSE
        )
      )
    }
```

5. Add two entries to the returned list:

```r
    pipeline_records = if (randomize_preprocessing) pipeline_records else NULL,
    pipelines = registry
```

`embeddings[[i]]` is `NULL` when `embeddings` is `NULL`, so a run without randomization builds the same tasks as before.

- [ ] Step 5: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "runner|utils", stop_on_failure = TRUE)'`
Expected: PASS. Then run the whole suite.

- [ ] Step 6: Mutation check

- In `run_validation()`, give every task `embeddings[[1L]]`: the task test fails.
- In `run_validation()`, call `allocate_pipelines()` inside the configuration loop with `random_state + i`: the same-allocation assertion of the first test fails.
- In `summarize_preprocessing_records()`, sort `method_id` as text: the summary test fails.
- In `summarize_preprocessing_records()`, average over all of the configuration's resamples (`runs` instead of `own`): the summary test fails.
- In `precompute_embeddings()`, skip `setTxtProgressBar()`: the progress test fails.

- [ ] Step 7: Commit

```bash
git add carve-r/R/runner.R carve-r/R/utils.R carve-r/tests/testthat/test-runner.R carve-r/tests/testthat/test-utils.R
git commit -m "$(cat <<'EOF'
feat(carve-r): embed each resample once and record the pipeline scores of randomized runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---
### Task 13: Console output for anchored and randomized runs

Files:
- Modify: `carve-r/R/output.R`
- Test: `carve-r/tests/testthat/test-output.R`

Interfaces:
- Consumes: `preprocessing_option`, `Identity`, `StandardScaler`, `PCA` (Tasks 7 and 8, in tests); stage 1's `print_run_header`.
- Produces (internal): `print_run_header(X, sweep, n_resamples, subsample_ratio, estimator_grids, n_jobs, random_state, verbose, mode = "default", anchors = NULL, randomize_preprocessing = FALSE, normalization_options = list(), dim_reduction_options = list())`.

- [ ] Step 1: Write the failing tests at the end of `tests/testthat/test-output.R`

```r
header_text <- function(...) {
  grids <- list(estimator_grid(KMeans, n_clusters = 2:3))
  paste(capture_messages(print_run_header(
    ..., sweep = resolve_sweep(n_clusters = 2:3), n_resamples = 10L, subsample_ratio = 0.618,
    estimator_grids = grids, n_jobs = 1L, random_state = 0L, verbose = 2L
  )), collapse = "")
}

test_that("a randomized header names the options", {
  text <- header_text(
    X = matrix(0, 100, 2),
    randomize_preprocessing = TRUE,
    normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
    dim_reduction_options = list(preprocessing_option(PCA, n_components = 2L))
  )
  expect_match(text, "[CARVE] randomize_preproc  : TRUE", fixed = TRUE)
  expect_match(text, "[CARVE] normalization      : identity, StandardScaler", fixed = TRUE)
  expect_match(text, "[CARVE] dim_reduction      : PCA", fixed = TRUE)
})

test_that("without randomization the header lists no options", {
  text <- header_text(X = matrix(0, 100, 2))
  expect_match(text, "[CARVE] randomize_preproc  : FALSE", fixed = TRUE)
  expect_match(text, "[CARVE] consensus anchors  : none", fixed = TRUE)
  expect_false(grepl("normalization", text, fixed = TRUE))
})

test_that("an anchored header projects the memory of the anchor blocks", {
  # Two configurations, two matrices each, of 5,000 by 5,000 doubles; the
  # 20,000-sample matrices of an exact run would take 12.80 GB.
  text <- header_text(X = matrix(0, 20000, 1), anchors = seq_len(5000L))
  expect_match(text, "[CARVE] consensus anchors  : 5000", fixed = TRUE)
  expect_match(text, "[CARVE] consensus memory   : 0.80 GB", fixed = TRUE)
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "output")'`
Expected: failures, unused arguments.

- [ ] Step 3: Implement in `R/output.R`

Replace `print_run_header()` with:

```r
print_run_header <- function(X, sweep, n_resamples, subsample_ratio, estimator_grids,
                             n_jobs, random_state, verbose, mode = "default", anchors = NULL,
                             randomize_preprocessing = FALSE, normalization_options = list(),
                             dim_reduction_options = list()) {
  if (verbose < 2) {
    return(invisible(NULL))
  }
  total <- sum(vapply(estimator_grids, function(g) length(expand_param_grid(g$grid)), integer(1)))
  policy <- resolve_mode(mode)
  n_matrices <- total * (policy$run_stability + policy$run_generalizability)
  # An anchored run keeps m by m blocks instead of n by n matrices.
  side <- if (is.null(anchors)) nrow(X) else length(anchors)
  memory_gb <- n_matrices * as.numeric(side)^2 * 8 / 1e9
  option_names <- function(options) {
    paste(vapply(options, function(o) o$name, character(1)), collapse = ", ")
  }
  preprocessing <- if (isTRUE(randomize_preprocessing)) {
    c(
      sprintf("[CARVE] normalization      : %s", option_names(normalization_options)),
      sprintf("[CARVE] dim_reduction      : %s", option_names(dim_reduction_options))
    )
  }
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
    sprintf("[CARVE] randomize_preproc  : %s", if (isTRUE(randomize_preprocessing)) "TRUE" else "FALSE"),
    preprocessing,
    sprintf("[CARVE] consensus anchors  : %s", if (is.null(anchors)) "none" else length(anchors)),
    sprintf("[CARVE] consensus memory   : %.2f GB", memory_gb),
    sprintf("[CARVE] random_state       : %d", as.integer(random_state)),
    paste("[CARVE]", line),
    "",
    "[CARVE] Starting validation ...",
    ""
  ), collapse = "\n"))
}
```

- [ ] Step 4: Run the tests and see them pass

Run: `Rscript -e 'devtools::test(filter = "output", stop_on_failure = TRUE)'`
Expected: PASS, the stage 1 header tests included.

- [ ] Step 5: Mutation check

- Compute the memory from `nrow(X)` regardless of `anchors`: the anchored test fails (12.80 GB).
- Print the option lines whatever `randomize_preprocessing` is: the second test fails.

- [ ] Step 6: Commit

```bash
git add carve-r/R/output.R carve-r/tests/testthat/test-output.R
git commit -m "$(cat <<'EOF'
feat(carve-r): report anchors and preprocessing options in the run header

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: The fit object and the full carve() signature

Files:
- Modify: `carve-r/R/AllClasses.R`, `carve-r/R/carve.R`, `carve-r/R/accessors.R` (the `show()` method only)
- Generated: `carve-r/man/carve.Rd`, `carve-r/man/CARVE-class.Rd`
- Test: `carve-r/tests/testthat/test-carve.R`

Interfaces:
- Consumes: everything above: `check_noise_policy`, `resolve_anchors`, `run_core_budget`, `default_estimator_grids`, `default_normalization_options`, `default_dim_reduction_options`, `summarize_preprocessing_records`, `run_validation`, `print_run_header`; stage 1's `resolve_sweep`, `infer_sweep_param`, `grid_sweep_values`, `validate_grids`, `coerce_reference_labels`, `n_cores`.
- Produces:
  - class union `carveDataFrameOrNULL`; `CARVE` slots `consensus_anchors` (`carveIntegerOrNULL`), `preprocessing_results` (`carveDataFrameOrNULL`), `preprocessing_pipelines` (`carveListOrNULL`).
  - `carve(x, n_clusters = 2:10, resolution = NULL, sweep = NULL, sweep_values = NULL, finer_is_larger = NULL, noise_policy = "drop", n_resamples = 100, subsample_ratio = 0.618, anchor_threshold = 5000, consensus_anchors = NULL, estimator_param_grids = "light", normalization_options = NULL, dim_reduction_options = NULL, randomize_preprocessing = FALSE, classifier = NULL, n_trees = 100, reference_labels = NULL, mode = "default", n_jobs = 1, BPPARAM = NULL, random_state = NULL, show_progress = FALSE, verbose = 0, ...)`, the spec's signature.
  - `run_params` gains `noise_policy`, `anchor_threshold`, `consensus_anchors` (the argument as given), `randomize_preprocessing` and `n_threads` (integer), which Task 15 reads.
  - internal `check_preprocessing_options(options, argument)`; `resolve_seed()` leaves room for `3 * n_resamples`.

- [ ] Step 1: Write the failing tests

In `tests/testthat/test-carve.R`, replace the body of the test `"random_state must leave room for every derived seed"`:

```r
test_that("random_state must leave room for every derived seed", {
  # The held-out embedding of the last resample uses
  # random_state + 3 * n_resamples - 1.
  limit <- .Machine$integer.max - 12L
  expect_error(
    carve(blobs$X, n_resamples = 4, random_state = limit + 1),
    sprintf("random_state must be at most %.0f so that every derived seed fits in an R integer.", limit),
    fixed = TRUE
  )
  expect_identical(resolve_seed(limit, 4L), as.integer(limit))
  expect_error(resolve_seed(limit + 1, 4L), "random_state must be at most", fixed = TRUE)
})
```

Append:

```r
blobs3d <- make_blobs(n_per = 30L, centers = diag(5, 3), sd = 1, seed = 42L)
resolution_fit <- carve(blobs3d$X, resolution = c(0.2, 0.5, 1), n_resamples = 4, random_state = 0)

test_that("a resolution sweep records its axis and the observed cluster counts", {
  axis <- resolution_fit@sweep
  expect_identical(axis@param, "resolution")
  expect_identical(axis@values, c(0.2, 0.5, 1))
  expect_true(axis@finer_is_larger)
  expect_false(axis@fixes_k)
  results <- resolution_fit@estimator_results
  expect_false("n_clusters" %in% names(results))
  expect_identical(results$estimator, rep(c("LeidenClustering", "LouvainClustering"), each = 3))
  expect_identical(results$method_label, rep(c("LeidenClustering, n_neighbors=15", "LouvainClustering, n_neighbors=15"), each = 3))
  expect_identical(results$sweep_value, rep(c(0.2, 0.5, 1), 2))
  expect_identical(results$sweep_rank, rep(0:2, 2))
  expect_identical(results$noise_fraction, rep(0, 6))
  expect_true(get_sweep_value(resolution_fit) %in% c(0.2, 0.5, 1))
  expect_true(get_k(resolution_fit) %in% round(results$n_clusters_observed))
})

test_that("labels on a resolution sweep pin sweep values, not k", {
  expect_length(get_labels(resolution_fit), 90L)
  expect_error(get_labels(resolution_fit, k = 3), "This run swept 'resolution', not 'n_clusters'.", fixed = TRUE)
  expect_length(get_labels(resolution_fit, sweep_value = 0.5), 90L)
  expect_error(get_labels(resolution_fit, sweep_value = 0.7), "No configurations found for resolution=0.7.", fixed = TRUE)
  expect_identical(length(unique(get_labels(resolution_fit, consensus_k = 4))), 4L)
})

test_that("resolution and a different sweep cannot be combined", {
  expect_error(
    carve(blobs$X, resolution = 1, sweep = "n_clusters"),
    "resolution= was given but sweep='n_clusters'. Pass sweep_values= instead, or drop resolution=.",
    fixed = TRUE
  )
})

test_that("a min_cluster_size sweep runs HDBSCAN with ranks running backwards", {
  skip_if_not_installed("dbscan")
  fit <- carve(blobs3d$X, sweep = "min_cluster_size", sweep_values = c(5, 10, 20),
               n_resamples = 4, random_state = 0)
  expect_false(fit@sweep@finer_is_larger)
  results <- fit@estimator_results
  expect_identical(unique(results$estimator), "HDBSCAN")
  expect_identical(results$min_cluster_size, c(5L, 10L, 20L))
  expect_identical(results$sweep_rank, c(2L, 1L, 0L))
  expect_true(all(results$noise_fraction >= 0 & results$noise_fraction < 1))
  expect_length(get_labels(fit, sweep_value = 10), 90L)
  selected <- get_estimator(fit)
  expect_identical(attr(selected, "estimator"), "HDBSCAN")
  expect_identical(attr(selected, "params")$cluster_selection_method, "eom")
})

# Labels the first five rows of every subsample noise.
first_five_noise <- function(X, n_clusters = 3L, random_state = NULL) {
  labels <- KMeans(X, n_clusters, random_state = random_state)
  labels[1:5] <- -1L
  labels
}

test_that("every noise policy records the same noise fraction", {
  grid <- list(estimator_grid(first_five_noise, n_clusters = 3L))
  fractions <- vapply(c("drop", "as_cluster", "singleton"), function(policy) {
    # "singleton" makes each noise sample a cluster, so the k-axis check warns.
    fit <- suppressWarnings(carve(blobs$X, n_resamples = 3, random_state = 0,
                                  estimator_param_grids = grid, noise_policy = policy))
    fit@estimator_results$noise_fraction
  }, numeric(1))
  expect_equal(unname(fractions), rep(5 / 55, 3))
})

test_that("the new settings are checked before the run", {
  X <- blobs$X
  expect_error(
    carve(X, noise_policy = "keep"),
    "Unknown noise_policy 'keep'. Expected 'drop', 'as_cluster', or 'singleton'.",
    fixed = TRUE
  )
  expect_error(carve(X, anchor_threshold = -1), "anchor_threshold must be a non-negative whole number.", fixed = TRUE)
  expect_error(carve(X, randomize_preprocessing = NA), "randomize_preprocessing must be TRUE or FALSE.", fixed = TRUE)
  expect_error(
    carve(X, normalization_options = list(Identity)),
    "normalization_options must be NULL or a list of preprocessing_option() specifications.",
    fixed = TRUE
  )
  expect_error(carve(X, consensus_anchors = 2.5), "consensus_anchors given as a fraction must be in (0, 1], got 2.5.", fixed = TRUE)
})

two <- make_blobs(n_per = 30L, centers = rbind(c(0, 0), c(6, 0)), seed = 4L)
anchored_fit <- function(...) {
  carve(two$X, n_resamples = 6, random_state = 0,
        estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3)), ...)
}

test_that("above anchor_threshold the run is anchored and says why", {
  fit <- NULL
  expect_warning(
    fit <- anchored_fit(anchor_threshold = 30),
    "n=60 exceeds anchor_threshold=30, so CARVE is using anchored consensus over 30 anchors. Per-sample scores and labels still cover every sample; consensus matrices and PAC are computed over the anchors.",
    fixed = TRUE
  )
  expect_no_error(validObject(fit))
  expect_length(fit@consensus_anchors, 30L)
  expect_false(is.unsorted(fit@consensus_anchors, strictly = TRUE))
  expect_identical(dim(fit@consensus_matrices[["0"]]), c(30L, 30L))
  expect_identical(dim(fit@consensus_generalizability_matrices[["1"]]), c(30L, 30L))
  expect_length(fit@stability_gini_scores[["0"]], 60L)
  expect_length(fit@generalizability_scores[["0"]], 60L)
  # One anchor draw serves every configuration.
  expect_identical(is.nan(fit@consensus_matrices[["0"]]), is.nan(fit@consensus_matrices[["1"]]))
  expect_output(show(fit), "Consensus: anchored over 30 anchors", fixed = TRUE)
})

test_that("consensus_anchors anchors a run below the threshold and the warning says so", {
  fit <- NULL
  expect_warning(
    fit <- anchored_fit(anchor_threshold = 1000, consensus_anchors = 25),
    "consensus_anchors=25 opts this run in regardless of anchor_threshold, so CARVE is using anchored consensus over 25 anchors.",
    fixed = TRUE
  )
  expect_length(fit@consensus_anchors, 25L)
})

test_that("at or below the threshold the run is exact", {
  fit <- anchored_fit(anchor_threshold = 100)
  expect_null(fit@consensus_anchors)
  expect_identical(dim(fit@consensus_matrices[["0"]]), c(60L, 60L))
  expect_output(show(fit), "Consensus: exact", fixed = TRUE)
})

test_that("the fit stores the threads its whole budget allows", {
  local_mocked_bindings(n_cores = function() 11L)
  grid <- list(estimator_grid(KMeans, n_clusters = 2L))
  fit <- carve(blobs$X, n_resamples = 4, random_state = 0, estimator_param_grids = grid,
               BPPARAM = BiocParallel::SerialParam())
  expect_identical(fit@run_params$n_threads, 11L)
  skip_on_os("windows")
  # Four workers with two threads each.
  fit <- carve(blobs$X, n_resamples = 4, random_state = 0, estimator_param_grids = grid, n_jobs = 4)
  expect_identical(fit@run_params$n_threads, 8L)
})

randomized_fit <- function(...) {
  carve(blobs$X, n_resamples = 8, random_state = 0, estimator_param_grids = k_grid,
        randomize_preprocessing = TRUE,
        normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
        dim_reduction_options = list(preprocessing_option(Identity), preprocessing_option(PCA, n_components = c(1L, 2L))),
        ...)
}

test_that("a randomized fit splits each configuration's scores by pipeline", {
  fit <- randomized_fit()
  table <- fit@preprocessing_results
  expect_identical(names(table), c(
    "method_id", "method_label", "pipeline", "normalization", "dim_reduction", "n_clusters",
    "n_resamples", "ari_stability", "ari_stability_se", "ari_generalizability",
    "ari_generalizability_se", "n_clusters_observed", "sweep_param", "sweep_value", "sweep_rank"
  ))
  expect_identical(as.vector(tapply(table$n_resamples, table$n_clusters, sum)), rep(8L, 3))
  expect_setequal(unique(table$pipeline), names(fit@preprocessing_pipelines))
  expect_identical(table$pipeline, paste(table$normalization, table$dim_reduction, sep = " | "))
  joined <- merge(table, fit@estimator_results[, c("method_id", "n_clusters", "config_id")],
                  by = c("method_id", "n_clusters"))
  expect_identical(nrow(joined), nrow(table))
  expect_true(all(vapply(fit@preprocessing_pipelines, inherits, logical(1), "carve_pipeline_spec")))
  expect_true(fit@run_params$randomize_preprocessing)
  expect_output(show(fit), sprintf("Preprocessing: randomized over %d pipelines", length(fit@preprocessing_pipelines)), fixed = TRUE)
})

test_that("a fit without randomization has no preprocessing results", {
  fit <- small_fit()
  expect_null(fit@preprocessing_results)
  expect_null(fit@preprocessing_pipelines)
})

test_that("the default options are resolved only for a randomized fit", {
  local_mocked_bindings(
    default_normalization_options = function(X) stop("resolved"),
    default_dim_reduction_options = function(X, subsample_ratio) stop("resolved")
  )
  expect_no_error(small_fit())
  expect_error(small_fit(randomize_preprocessing = TRUE), "resolved", fixed = TRUE)
})

test_that("an empty reduction list under randomization is an error", {
  expect_error(
    small_fit(randomize_preprocessing = TRUE, dim_reduction_options = list()),
    "randomize_preprocessing=TRUE needs at least one normalization option and one dimensionality reduction option.",
    fixed = TRUE
  )
})

test_that("the verbose header names the resolved options", {
  messages <- capture_messages(randomized_fit(verbose = 2))
  text <- paste(messages, collapse = "")
  expect_match(text, "[CARVE] normalization      : identity, StandardScaler", fixed = TRUE)
  expect_match(text, "[CARVE] dim_reduction      : identity, PCA", fixed = TRUE)
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "carve")'`
Expected: failures, unknown arguments (`Unknown argument: resolution.` from `check_dots()`), and the limit test.

- [ ] Step 3: Add the slots in `R/AllClasses.R`

After the two existing class unions add:

```r
setClassUnion("carveDataFrameOrNULL", c("data.frame", "NULL"))
```

In the `CARVE` class roxygen block, replace the `@slot run_params` and `@slot consensus_matrices` entries and add three slots after `@slot generalizability_scores`:

```r
#' @slot run_params List of the run settings: `n_resamples`,
#'   `subsample_ratio`, `n_trees`, `n_jobs`, `mode`, `random_state` (the seed
#'   the run used), `classifier`, `noise_policy`, `anchor_threshold`,
#'   `consensus_anchors`, `randomize_preprocessing`, and `n_threads`, the
#'   threads of the classifier [get_labels()] trains on an anchored run.
```

```r
#' @slot consensus_matrices Stability consensus matrices, a list named by
#'   `config_id`. Entries are `NULL` when the run skipped stability. On an
#'   anchored run each is the block over the anchors.
```

```r
#' @slot consensus_anchors Sorted row indices of the anchors of an anchored
#'   run, or `NULL` for an exact run.
#' @slot preprocessing_results Scores split by preprocessing pipeline, or
#'   `NULL` for a run without randomized preprocessing.
#' @slot preprocessing_pipelines The pipelines of a randomized run, named by
#'   their labels, or `NULL`.
```

Add the slots to `setClass("CARVE", ...)` after `generalizability_scores = "list"`:

```r
    consensus_anchors = "carveIntegerOrNULL",
    preprocessing_results = "carveDataFrameOrNULL",
    preprocessing_pipelines = "carveListOrNULL"
```

- [ ] Step 4: Rewrite `R/carve.R`

Replace the roxygen block and the `setMethod("carve", "ANY", ...)` call with:

```r
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
#' A run sweeps one parameter. By default it is `n_clusters`. Giving
#' `resolution` sweeps the resolution of [LeidenClustering()] and
#' [LouvainClustering()], and `sweep = "min_cluster_size"` with
#' `sweep_values` sweeps the minimum cluster size of [HDBSCAN()]. Off the
#' `n_clusters` axis the number of clusters is an outcome, recorded for each
#' configuration as its mean over the resamples.
#'
#' Resample `b`, counting from 0, draws its first subsample with seed
#' `random_state + b` and its second with `random_state + b + n_resamples`,
#' and its estimator and classifier use `random_state + b`. The results are
#' therefore the same for every `n_jobs` and `BPPARAM`, and the session's
#' random number stream is left as it was.
#'
#' The `n_neighbors` of [SpectralClustering()], [LeidenClustering()] and
#' [LouvainClustering()] is a count at the density of the full data. Each
#' subsample is clustered with the count scaled by its share of the samples,
#' rounded and at least 2, so that it covers the same neighborhood. The
#' neighbor settings of preprocessing transforms are not scaled.
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
#' Above `anchor_threshold` samples those matrices would not fit in memory,
#' and the run is anchored: it draws `m` anchor samples once and keeps only
#' the `m` by `m` blocks over them. Per-sample stability is computed against
#' the anchors, so every sample still gets a score, and [get_labels()] cuts
#' the anchor block and labels the other samples with the classifier,
#' trained on the anchors. PAC then covers anchor pairs only and cannot be
#' compared with the PAC of an exact run. A warning says when a run is
#' anchored, and [consensus_anchors()] returns the anchors.
#'
#' With `randomize_preprocessing = TRUE`, each resample draws a pipeline: one
#' normalization option and one dimensionality reduction option, with one
#' value drawn for each of their arguments. Every pair of options gets
#' `n_resamples` divided by the number of pairs, rounded up or down, in a
#' seeded random order. The pipeline is fit separately on each of the
#' resample's subsamples, with seeds `random_state + b`,
#' `random_state + b + n_resamples` and `random_state + b + 2 * n_resamples`,
#' so an embedding that does not reproduce across fits lowers both criteria.
#' Each resample is embedded once, before the configurations run, and the
#' embeddings stay in memory during the fit. The classifier trains on the
#' raw features of the first subsample with the labels clustered from its
#' embedding, so a cluster that exists only in the embedding does not
#' generalize, and t-SNE, which cannot embed new samples, can take part.
#' [preprocessing_results()] splits the scores by pipeline.
#'
#' @param x The data, one row per sample: a numeric matrix, a numeric data
#'   frame, a sparse matrix from the Matrix package, or a numeric vector.
#' @param ... Not used. An argument name `carve()` does not know is an error.
#' @param n_clusters Numbers of clusters to evaluate. A single number `K`
#'   means `2:K`. Custom grids set their own values.
#' @param resolution Resolutions to evaluate with the graph estimators.
#'   Giving it makes `resolution` the swept parameter.
#' @param sweep Name of the swept parameter. Defaults to `"resolution"` when
#'   `resolution` is given and to `"n_clusters"` otherwise; custom grids set
#'   it themselves. Estimators that take a number of clusters and estimators
#'   that take a resolution cannot share a run.
#' @param sweep_values Values of `sweep` when it is neither `"n_clusters"`
#'   nor `"resolution"`, such as minimum cluster sizes for [HDBSCAN()].
#' @param finer_is_larger Whether larger values of `sweep` give more
#'   clusters. It is known for `n_clusters` and `resolution` (`TRUE`) and for
#'   `min_cluster_size` (`FALSE`), and required for any other parameter.
#' @param noise_policy What to do with the -1 labels of density-based
#'   methods such as [HDBSCAN()]. `"drop"` leaves those samples out of the
#'   resample, so the consensus matrix, the ARI and the classifier treat them
#'   as not drawn. `"as_cluster"` keeps -1 as an ordinary label, and
#'   `"singleton"` puts each of them in a cluster of its own.
#' @param n_resamples Resamples per configuration.
#' @param subsample_ratio Share of the samples drawn, without replacement,
#'   into each subsample. Must lie strictly between 0 and 1.
#' @param anchor_threshold Runs with at most this many samples build full
#'   consensus matrices; larger runs are anchored. See Details.
#' @param consensus_anchors Number of anchors, or a share of the samples in
#'   (0, 1]. `NULL` means `min(nrow(x), anchor_threshold)`. Setting it anchors
#'   a run of any size; lowering it shrinks the blocks, `8 * m^2` bytes each.
#' @param estimator_param_grids `"light"`, `"full"`, or a list of
#'   [estimator_grid()] specifications. On the `n_clusters` axis `"light"`
#'   runs KMeans, Ward-linkage agglomerative and self-tuning spectral
#'   clustering, and `"full"` adds average, single and complete linkage and
#'   RBF spectral clustering. On the `resolution` axis `"light"` runs Leiden
#'   and Louvain on 15-neighbor graphs, and `"full"` adds 10- and 30-neighbor
#'   graphs. On the `min_cluster_size` axis `"light"` runs HDBSCAN with
#'   excess-of-mass selection, and `"full"` adds leaf selection. All grids
#'   must sweep the same parameter over the same values.
#' @param normalization_options,dim_reduction_options Lists of
#'   [preprocessing_option()] objects for randomized preprocessing. `NULL`
#'   uses the defaults: identity, standardization and, for data without
#'   negative values, log1p; identity, PCA, t-SNE and, when uwot is
#'   installed, UMAP, over the values the smallest subsample supports. Not
#'   used unless `randomize_preprocessing = TRUE`.
#' @param randomize_preprocessing Draw a preprocessing pipeline for each
#'   resample. See Details.
#' @param classifier `NULL` for the default random forest, or a function
#'   `(x_train, y_train, x_test)` that returns one predicted label per row of
#'   `x_test`. If it also has `n_threads` or `random_state` arguments, CARVE
#'   fills them in.
#' @param n_trees Trees in the default random forest. Not used when
#'   `classifier` is given.
#' @param reference_labels Labels, one per sample, that [get_labels()]
#'   renames its clusters to match. Whole numbers without `NA` keep their
#'   values. Any other vector, including whole numbers with an `NA`, is
#'   coded 1, 2, ... in order of first appearance, and `NA` becomes -1.
#' @param mode `"default"` scores both criteria, `"stability"` skips
#'   generalizability and `"generalizability"` skips stability. The last two
#'   are experimental.
#' @param n_jobs Core budget; see Details. `-1` means every core and `-2` all
#'   but one.
#' @param BPPARAM Optional [BiocParallel::BiocParallelParam-class] backend for
#'   the resamples. Its worker count replaces the one `n_jobs` would set.
#' @param random_state Seed of the run. `NULL` means 0.
#' @param show_progress Show a progress bar over the configurations and,
#'   under randomized preprocessing, one before it over the embedding pass.
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
#'
#' graph_fit <- carve(X, resolution = c(0.25, 0.5, 1), n_resamples = 10, random_state = 0)
#' estimator_results(graph_fit)[, c("method_label", "resolution", "n_clusters_observed")]
#' @rdname carve
#' @export
setMethod("carve", "ANY", function(x, n_clusters = 2:10, resolution = NULL, sweep = NULL,
                                   sweep_values = NULL, finer_is_larger = NULL,
                                   noise_policy = "drop", n_resamples = 100,
                                   subsample_ratio = 0.618, anchor_threshold = 5000,
                                   consensus_anchors = NULL, estimator_param_grids = "light",
                                   normalization_options = NULL, dim_reduction_options = NULL,
                                   randomize_preprocessing = FALSE, classifier = NULL,
                                   n_trees = 100, reference_labels = NULL, mode = "default",
                                   n_jobs = 1, BPPARAM = NULL, random_state = NULL,
                                   show_progress = FALSE, verbose = 0, ...) {
  check_dots(...)
  fit_carve(
    as_data_matrix(x),
    n_clusters = n_clusters,
    resolution = resolution,
    sweep = sweep,
    sweep_values = sweep_values,
    finer_is_larger = finer_is_larger,
    noise_policy = noise_policy,
    n_resamples = n_resamples,
    subsample_ratio = subsample_ratio,
    anchor_threshold = anchor_threshold,
    consensus_anchors = consensus_anchors,
    estimator_param_grids = estimator_param_grids,
    normalization_options = normalization_options,
    dim_reduction_options = dim_reduction_options,
    randomize_preprocessing = randomize_preprocessing,
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
```

Replace `fit_carve()` with:

```r
fit_carve <- function(X, n_clusters, resolution, sweep, sweep_values, finer_is_larger,
                      noise_policy, n_resamples, subsample_ratio, anchor_threshold,
                      consensus_anchors, estimator_param_grids, normalization_options,
                      dim_reduction_options, randomize_preprocessing, classifier, n_trees,
                      reference_labels, mode, n_jobs, BPPARAM, random_state, show_progress,
                      verbose) {
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
  # Python checks the noise policy only inside the first resample.
  check_noise_policy(noise_policy)
  if (!is.numeric(anchor_threshold) || length(anchor_threshold) != 1L || is.na(anchor_threshold) ||
      anchor_threshold < 0 || anchor_threshold != round(anchor_threshold)) {
    stop("anchor_threshold must be a non-negative whole number.", call. = FALSE)
  }
  if (!isTRUE(randomize_preprocessing) && !isFALSE(randomize_preprocessing)) {
    stop("randomize_preprocessing must be TRUE or FALSE.", call. = FALSE)
  }
  check_preprocessing_options(normalization_options, "normalization_options")
  check_preprocessing_options(dim_reduction_options, "dim_reduction_options")
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

  anchors <- resolve_anchors(nrow(X), consensus_anchors, anchor_threshold, seed)
  if (!is.null(anchors)) {
    # resolve_anchors() ignores anchor_threshold once consensus_anchors is
    # set, so the message names the setting that applied.
    reason <- if (is.null(consensus_anchors)) {
      sprintf("n=%d exceeds anchor_threshold=%s", nrow(X), format_param_value(anchor_threshold))
    } else {
      sprintf(
        "consensus_anchors=%s opts this run in regardless of anchor_threshold",
        format_repr(consensus_anchors)
      )
    }
    warning(sprintf(
      "%s, so CARVE is using anchored consensus over %d anchors. Per-sample scores and labels still cover every sample; consensus matrices and PAC are computed over the anchors.",
      reason, length(anchors)
    ), call. = FALSE)
  }

  # The sweep axis, and the grids along it. Custom grids decide the swept
  # parameter, so a grid sweeping resolution works without resolution=.
  is_preset <- is.character(estimator_param_grids) && length(estimator_param_grids) == 1L &&
    estimator_param_grids %in% c("light", "full")
  is_custom <- is.list(estimator_param_grids) && !is.object(estimator_param_grids) &&
    all(vapply(estimator_param_grids, inherits, logical(1), "carve_estimator_grid"))
  if (!is_preset && !is_custom) {
    stop(sprintf(
      "Unknown estimator_param_grids preset %s. Expected 'light', 'full', or a list of estimator_grid() specifications.",
      format_repr(estimator_param_grids)
    ), call. = FALSE)
  }
  swept <- sweep
  values <- sweep_values
  if (is_custom) {
    if (is.null(swept) && is.null(resolution)) {
      swept <- infer_sweep_param(estimator_param_grids)
    }
    if (is.null(values) && !is.null(swept)) {
      values <- grid_sweep_values(estimator_param_grids, swept)
    }
  }
  axis <- resolve_sweep(
    n_clusters = n_clusters,
    resolution = resolution,
    sweep = swept,
    sweep_values = values,
    finer_is_larger = finer_is_larger
  )
  if (is_custom) {
    grids <- estimator_param_grids
    axis@values <- validate_grids(grids, axis)
  } else {
    grids <- default_estimator_grids(X, preset = estimator_param_grids, sweep = axis)
  }

  # Only a randomized fit uses the preprocessing options. Resolving the
  # defaults otherwise would warn about log1p on any data with negative
  # values. An empty normalization list means the defaults, as in Python;
  # an empty reduction list is kept and rejected by the allocation.
  if (randomize_preprocessing) {
    norm_options <- if (length(normalization_options) > 0L) {
      normalization_options
    } else {
      default_normalization_options(X)
    }
    dr_options <- if (!is.null(dim_reduction_options)) {
      dim_reduction_options
    } else {
      default_dim_reduction_options(X, subsample_ratio)
    }
  } else {
    norm_options <- if (is.null(normalization_options)) list() else normalization_options
    dr_options <- if (is.null(dim_reduction_options)) list() else dim_reduction_options
  }

  print_run_header(
    X, axis, n_resamples, subsample_ratio, grids, n_jobs, seed, verbose, policy$mode,
    anchors = anchors,
    randomize_preprocessing = randomize_preprocessing,
    normalization_options = norm_options,
    dim_reduction_options = dr_options
  )
  run <- run_validation(
    X, grids,
    n_resamples = n_resamples,
    subsample_ratio = subsample_ratio,
    classifier = classifier,
    n_trees = as.integer(n_trees),
    n_jobs = n_jobs,
    BPPARAM = BPPARAM,
    random_state = seed,
    sweep = axis,
    noise_policy = noise_policy,
    mode = policy$mode,
    anchors = anchors,
    randomize_preprocessing = randomize_preprocessing,
    normalization_options = norm_options,
    dim_reduction_options = dr_options,
    show_progress = show_progress,
    verbose = verbose
  )

  results <- run$records
  # config_id is the join key between the table and the per-configuration
  # lists.
  keys <- as.character(results$config_id)
  gini <- NULL
  ce <- NULL
  if (policy$run_stability) {
    summaries <- run$summaries[keys]
    gini <- lapply(summaries, function(s) s$gini)
    ce <- lapply(summaries, function(s) s$ce)
    results$consensus_pac_stability <- unname(vapply(summaries, function(s) s$pac, numeric(1)))
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
    unname(vapply(run$generalizability_scores[keys], mean, numeric(1)))
  } else {
    NaN
  }

  preprocessing <- NULL
  if (randomize_preprocessing) {
    preprocessing <- summarize_preprocessing_records(run$pipeline_records, run$pipelines, axis@param)
  }
  # The classifier that extends an anchored cut in get_labels() is one fit
  # outside the resample loop, so it gets the run's whole budget.
  budget <- run_core_budget(n_jobs, n_resamples, BPPARAM)

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
      classifier = classifier,
      noise_policy = noise_policy,
      anchor_threshold = anchor_threshold,
      consensus_anchors = consensus_anchors,
      randomize_preprocessing = randomize_preprocessing,
      n_threads = as.integer(min(n_cores(), budget[["outer"]] * budget[["inner"]]))
    ),
    estimator_results = results,
    estimator_param_grids = grids,
    sweep = axis,
    consensus_matrices = run$consensus_matrices,
    consensus_generalizability_matrices = run$consensus_generalizability_matrices,
    stability_gini_scores = gini,
    stability_ce_scores = ce,
    generalizability_scores = run$generalizability_scores,
    consensus_anchors = anchors,
    preprocessing_results = preprocessing,
    preprocessing_pipelines = if (randomize_preprocessing) run$pipelines else NULL
  )
  print_run_footer(results, verbose)
  fit
}

check_preprocessing_options <- function(options, argument) {
  valid <- is.null(options) || (is.list(options) && !is.object(options) &&
    all(vapply(options, inherits, logical(1), "carve_preprocessing_option")))
  if (!valid) {
    stop(sprintf(
      "%s must be NULL or a list of preprocessing_option() specifications.",
      argument
    ), call. = FALSE)
  }
  invisible(options)
}
```

In `resolve_seed()`, change the comment to `# NULL means 0, as in Python. Every derived seed, up to` / `# random_state + 3 * n_resamples - 1 (the held-out embedding of the last` / `# resample under randomized preprocessing), must fit in an R integer.` and the limit line to `limit <- .Machine$integer.max - 3 * n_resamples`.

- [ ] Step 5: Extend `show()` in `R/accessors.R`

Replace the last two lines of the `show` method for `CARVE` (`cat("Mode: ", ...)` and `invisible(object)`) with:

```r
  cat("Mode: ", settings$mode, "\n", sep = "")
  anchors <- object@consensus_anchors
  cat("Consensus: ", if (is.null(anchors)) "exact" else sprintf("anchored over %d anchors", length(anchors)),
      "\n", sep = "")
  if (!is.null(object@preprocessing_pipelines)) {
    cat("Preprocessing: randomized over ", length(object@preprocessing_pipelines), " pipelines\n", sep = "")
  }
  invisible(object)
```

- [ ] Step 6: Document, then run the tests and see them pass

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "carve", stop_on_failure = TRUE)'
Rscript -e 'devtools::test(stop_on_failure = TRUE)'
```

Expected: `document()` rewrites `man/carve.Rd` and `man/CARVE-class.Rd`; the links to `consensus_anchors()` and `preprocessing_results()` resolve once Task 15 adds those topics, and `R CMD check` runs only after that. All tests pass with `SKIP 0` (the `n_jobs = 4` part of the threads test runs here, on macOS).

- [ ] Step 7: Mutation check

- In `fit_carve()`, pass `resolution = NULL` to `resolve_sweep()`: the resolution tests fail.
- In `fit_carve()`, store `n_threads = budget[["inner"]]`: the threads test fails.
- In `fit_carve()`, name the threshold in the warning when `consensus_anchors` was set: the opt-in test fails.
- In `fit_carve()`, resolve the default options whatever `randomize_preprocessing` is: the default-options test fails.
- In `resolve_seed()`, keep `2 * n_resamples`: the limit test fails.

- [ ] Step 8: Commit

```bash
git add carve-r/R/AllClasses.R carve-r/R/carve.R carve-r/R/accessors.R carve-r/man/carve.Rd \
  carve-r/man/CARVE-class.Rd carve-r/tests/testthat/test-carve.R
git commit -m "$(cat <<'EOF'
feat(carve-r): give carve() the sweep, noise, anchor and preprocessing arguments

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Anchored and noise labels, and the new accessors

Files:
- Modify: `carve-r/R/AllGenerics.R`, `carve-r/R/accessors.R`
- Generated: `carve-r/NAMESPACE`, `carve-r/man/get_labels.Rd`, `carve-r/man/consensus_matrix.Rd`, `carve-r/man/estimator_results.Rd`, `carve-r/man/preprocessing_results.Rd`
- Test: `carve-r/tests/testthat/test-accessors.R`

Interfaces:
- Consumes: `noise_mask`, `NOISE_MARGIN` (Task 2); the Task 14 slots and `run_params$n_threads`; stage 1's `select_row`, `cut_consensus`, `align_cluster_labels`, `coerce_reference_labels`, `count_clusters`, `default_generalizability_classifier`, `sample_scores`.
- Produces:
  - exported generics and `CARVE` methods `consensus_anchors(fit, ...)`, `preprocessing_results(fit, ...)`, `preprocessing_pipelines(fit, ...)`.
  - `get_labels(fit, measure = "stability", rule = "1se", k = NULL, sweep_value = NULL, consensus_k = NULL, not_two = FALSE, mode = "default", estimator = NULL, noise_labels = FALSE, noise_quantile = 0.05, noise_score = "gini", reference_labels = NULL, ...)`.
  - internal `extend_anchor_labels(fit, anchor_labels)` returns one label per sample.

- [ ] Step 1: Write the failing tests at the end of `tests/testthat/test-accessors.R`

```r
# Distinct scores on [0.5, 1] whose lowest length(low) values sit at low. At
# quantile 0.05 exactly those samples are flagged when length(low) is
# ceiling(0.05 * (n - 1)): 5 for 90 samples, 3 for 60.
known_scores <- function(n, low) {
  values <- seq(0.5, 1, length.out = n)
  scores <- numeric(n)
  scores[low] <- values[seq_along(low)]
  scores[-low] <- values[-seq_along(low)]
  scores
}

# A copy of the fit with scores at config_id and the reversed scores at
# every other configuration, so reading the wrong one moves the noise.
with_scores <- function(fit, source, config_id, scores) {
  name <- switch(source,
    gini = "stability_gini_scores",
    ce = "stability_ce_scores",
    accuracy = "generalizability_scores"
  )
  values <- methods::slot(fit, name)
  for (key in names(values)) {
    values[[key]] <- if (key == as.character(config_id)) scores else rev(scores)
  }
  methods::slot(fit, name) <- values
  fit
}

config_at <- function(fit, k) {
  results <- estimator_results(fit)
  results$config_id[results$sweep_value == k]
}

# The Python suite's noise positions, plus one: spread over the three blobs.
noise_low <- c(4L, 18L, 42L, 59L, 85L)

test_that("noise labels are off by default", {
  plain <- get_labels(fit, k = 3)
  expect_identical(get_labels(fit, k = 3, noise_labels = FALSE), plain)
  expect_false(any(plain == -1L))
})

test_that("noise lands on the samples with the lowest scores", {
  for (source in c("gini", "ce", "accuracy")) {
    noisy_fit <- with_scores(fit, source, config_at(fit, 3), known_scores(90L, noise_low))
    plain <- get_labels(noisy_fit, k = 3)
    noisy <- get_labels(noisy_fit, k = 3, noise_labels = TRUE, noise_score = source)
    expect_identical(which(noisy == -1L), noise_low, info = source)
    expect_identical(noisy[noisy != -1L], plain[noisy != -1L], info = source)
  }
})

test_that("noise scores are joined on config_id", {
  noisy_fit <- with_scores(fit, "gini", config_at(fit, 3), known_scores(90L, noise_low))
  noisy_fit@estimator_results <- noisy_fit@estimator_results[c(3L, 1L, 2L), ]
  # The k = 3 row no longer sits at position config_id + 1.
  expect_false(which(noisy_fit@estimator_results$n_clusters == 3) == config_at(fit, 3) + 1L)
  expect_identical(which(get_labels(noisy_fit, k = 3, noise_labels = TRUE) == -1L), noise_low)
})

test_that("ties at the cutoff warn with the counts", {
  scores <- rep(1, 90)
  scores[c(4L, 42L, 85L)] <- 0.5
  scores[c(11L, 26L, 36L, 51L, 66L, 81L)] <- 0.9
  noisy_fit <- with_scores(fit, "gini", config_at(fit, 3), scores)
  noisy <- NULL
  expect_warning(
    noisy <- get_labels(noisy_fit, k = 3, noise_labels = TRUE),
    "noise_quantile=0.05 asks for about 5 of 90 samples by gini; 3 were flagged. Samples tied at the cutoff (0.900) or within 0.05 of the median (1.000) stay labeled.",
    fixed = TRUE
  )
  expect_identical(which(noisy == -1L), c(4L, 42L, 85L))
})

test_that("NaN scores are noise and stay out of the count", {
  scores <- rep(1, 90)
  scores[c(4L, 42L, 85L)] <- 0.5
  scores[c(11L, 26L, 36L, 51L, 66L, 81L)] <- 0.9
  scores[c(21L, 61L)] <- NaN
  noisy_fit <- with_scores(fit, "gini", config_at(fit, 3), scores)
  noisy <- NULL
  expect_warning(
    noisy <- get_labels(noisy_fit, k = 3, noise_labels = TRUE),
    "noise_quantile=0.05 asks for about 5 of 88 samples by gini; 3 were flagged.",
    fixed = TRUE
  )
  expect_identical(which(noisy == -1L), c(4L, 21L, 42L, 61L, 85L))
})

test_that("the noise settings are checked on every call", {
  for (flag in c(FALSE, TRUE)) {
    expect_error(
      get_labels(fit, noise_labels = flag, noise_score = "nope"),
      "noise_score must be one of: 'gini', 'ce', 'accuracy'.",
      fixed = TRUE
    )
    for (quantile in c(0, 1, 1.5)) {
      expect_error(
        get_labels(fit, noise_labels = flag, noise_quantile = quantile),
        "noise_quantile must be strictly between 0 and 1, got",
        fixed = TRUE
      )
    }
  }
})

test_that("noise needs the score it ranks by", {
  gen <- single_mode_fit("generalizability")
  expect_error(
    get_labels(gen, measure = "generalizability", rule = "max", mode = "generalizability",
               noise_labels = TRUE, noise_score = "gini"),
    "Gini stability scores are not available for this run.",
    fixed = TRUE
  )
  stab <- single_mode_fit("stability")
  expect_error(
    get_labels(stab, noise_labels = TRUE, noise_score = "accuracy"),
    "Generalizability scores are not available for this run.",
    fixed = TRUE
  )
})

two <- make_blobs(n_per = 30L, centers = rbind(c(0, 0), c(6, 0)), seed = 4L)
# The anchored warning is tested in test-carve.R.
anchored <- suppressWarnings(carve(
  two$X, n_resamples = 6, random_state = 0, anchor_threshold = 30,
  estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3))
))

test_that("anchored labels cover every sample and recover the groups", {
  labels <- get_labels(anchored, k = 2)
  expect_length(labels, 60L)
  expect_gt(adjusted_rand_index(labels, two$y), 0.9)
  expect_identical(length(unique(get_labels(anchored, k = 3))), 3L)
  expect_length(get_labels(anchored, k = 2, mode = "generalizability"), 60L)
})

test_that("anchors keep the labels of the cut", {
  # A constant classifier cannot reproduce an alternating cut, so labels
  # that the classifier overwrote would show.
  constant <- anchored
  constant@run_params$classifier <- function(x_train, y_train, x_test) rep(1L, nrow(x_test))
  anchors <- consensus_anchors(constant)
  planted <- rep_len(1:2, length(anchors))
  extended <- extend_anchor_labels(constant, planted)
  expect_identical(extended[anchors], planted)
  expect_true(all(extended[-anchors] == 1L))
})

test_that("a one-cluster cut labels every sample with that cluster", {
  expect_identical(extend_anchor_labels(anchored, rep(2L, 30L)), rep(2L, 60L))
})

test_that("anchored labels are reproducible and use the fit's seed and budget", {
  set.seed(2)
  before <- .Random.seed
  first <- get_labels(anchored, k = 2)
  expect_identical(.Random.seed, before)
  expect_identical(first, get_labels(anchored, k = 2))
  seen <- new.env()
  spy <- anchored
  spy@run_params$classifier <- function(x_train, y_train, x_test, random_state, n_threads) {
    seen$seed <- random_state
    seen$threads <- n_threads
    rep(y_train[1], nrow(x_test))
  }
  spy@run_params$random_state <- 7L
  spy@run_params$n_threads <- 5L
  extend_anchor_labels(spy, rep_len(1:2, 30L))
  expect_identical(seen$seed, 7L)
  expect_identical(seen$threads, 5L)
})

test_that("an anchored run flags samples outside the anchors as noise", {
  low <- setdiff(seq_len(60L), consensus_anchors(anchored))[1:3]
  noisy_fit <- with_scores(anchored, "gini", config_at(anchored, 2), known_scores(60L, low))
  expect_identical(which(get_labels(noisy_fit, k = 2, noise_labels = TRUE) == -1L), low)
})

test_that("an exact fit has no anchors and no preprocessing results", {
  expect_null(consensus_anchors(fit))
  expect_null(preprocessing_results(fit))
  expect_null(preprocessing_pipelines(fit))
  expect_identical(consensus_anchors(anchored), anchored@consensus_anchors)
})

test_that("a randomized fit returns its table and pipelines", {
  randomized <- carve(
    blobs$X, n_resamples = 4, random_state = 0, estimator_param_grids = k_grid,
    randomize_preprocessing = TRUE,
    normalization_options = list(preprocessing_option(Identity)),
    dim_reduction_options = list(preprocessing_option(Identity), preprocessing_option(PCA, n_components = 1L))
  )
  expect_identical(preprocessing_results(randomized), randomized@preprocessing_results)
  expect_setequal(names(preprocessing_pipelines(randomized)), c("identity | identity", "identity | PCA(n_components=1)"))
  spec <- preprocessing_pipelines(randomized)[["identity | PCA(n_components=1)"]]
  expect_identical(dim(pipeline_from_spec(spec, 0L)(blobs$X)), c(90L, 1L))
})
```

- [ ] Step 2: Run them and see them fail

Run: `Rscript -e 'devtools::test(filter = "accessors")'`
Expected: failures: `Unknown argument: noise_labels.`, the accessors and `extend_anchor_labels` not found.

- [ ] Step 3: Add the generics to `R/AllGenerics.R`

```r
#' @rdname consensus_matrix
#' @export
setGeneric("consensus_anchors", function(fit, ...) standardGeneric("consensus_anchors"))

#' @rdname preprocessing_results
#' @export
setGeneric("preprocessing_results", function(fit, ...) standardGeneric("preprocessing_results"))

#' @rdname preprocessing_results
#' @export
setGeneric("preprocessing_pipelines", function(fit, ...) standardGeneric("preprocessing_pipelines"))
```

- [ ] Step 4: Update `get_labels()` in `R/accessors.R`

Replace the roxygen block above `setMethod("get_labels", ...)` and the method with:

```r
#' Consensus cluster labels
#'
#' `get_labels()` selects a configuration, as [get_k()] does, and cuts its
#' consensus matrix into clusters by average-linkage hierarchical clustering
#' of `1 - consensus`, at the configuration's number of clusters. Pairs of
#' samples never drawn together count as 0.5.
#'
#' On an anchored run the cut covers the anchors. The classifier the run used
#' for generalizability, trained on the anchors and their labels, then labels
#' the other samples; the anchors keep the labels of the cut.
#'
#' When the reference labels have as many clusters as the cut, the clusters
#' are renamed to match them by maximum overlap; samples whose reference
#' label is -1 take no part in the matching. `get_labels()` keeps no state
#' between calls, so to keep cluster names stable across calls, pass earlier
#' labels as `reference_labels`.
#'
#' With `noise_labels = TRUE`, ambiguous samples are labeled -1 after the
#' matching. A sample is ambiguous when its `noise_score` at the selected
#' configuration is `NaN`, or lies strictly below the `noise_quantile` of the
#' scores and more than 0.05 below their median. Samples tied with the
#' quantile keep their labels, so identical scores flag nothing, and the
#' margin keeps samples that score about as well as the median one. A warning
#' gives the counts when fewer samples are flagged than the quantile asks
#' for. The scores belong to the selected configuration and do not depend on
#' `consensus_k` or `estimator`.
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
#' @param noise_labels Label ambiguous samples -1; see Details.
#' @param noise_quantile Share of the samples considered for noise, strictly
#'   between 0 and 1. Checked on every call.
#' @param noise_score The per-sample score that ranks the samples: `"gini"`,
#'   `"ce"` or `"accuracy"`, as [sample_scores()] returns them. Checked on
#'   every call.
#' @param reference_labels Labels to match, one per sample, coded as in
#'   [carve()]. `NULL` uses the labels given to [carve()], if any.
#' @return An integer vector of cluster labels, one per sample, with -1 for
#'   the samples `noise_labels` flags. An `estimator` that labels samples -1
#'   itself also gives -1.
#' @seealso [get_k()], [consensus_matrix()], [sample_scores()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)))
#' labels <- get_labels(fit)
#' table(labels)
#' table(get_labels(fit, k = 3, reference_labels = labels))
#' table(get_labels(fit, noise_labels = TRUE, noise_quantile = 0.1))
#' @rdname get_labels
#' @export
setMethod("get_labels", "CARVE", function(fit, measure = "stability", rule = "1se", k = NULL,
                                          sweep_value = NULL, consensus_k = NULL,
                                          not_two = FALSE, mode = "default", estimator = NULL,
                                          noise_labels = FALSE, noise_quantile = 0.05,
                                          noise_score = "gini", reference_labels = NULL, ...) {
  check_dots(...)
  if (!is.character(noise_score) || length(noise_score) != 1L ||
      !noise_score %in% c("gini", "ce", "accuracy")) {
    stop("noise_score must be one of: 'gini', 'ce', 'accuracy'.", call. = FALSE)
  }
  if (!is.numeric(noise_quantile) || length(noise_quantile) != 1L || is.na(noise_quantile) ||
      noise_quantile <= 0 || noise_quantile >= 1) {
    stop(sprintf(
      "noise_quantile must be strictly between 0 and 1, got %s.",
      format_repr(noise_quantile)
    ), call. = FALSE)
  }
  policy <- resolve_mode(mode)
  selected <- select_row(fit, measure, rule, not_two = not_two, k = k, sweep_value = sweep_value)
  # The noise scores are read before the cut, so a missing score fails first.
  noise <- NULL
  if (isTRUE(noise_labels)) {
    scores <- sample_scores(fit, selected$config_id, source = noise_score)
    noise <- noise_mask(scores, noise_quantile)
  }
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
  if (!is.null(fit@consensus_anchors)) {
    labels <- extend_anchor_labels(fit, labels)
  }
  reference <- if (is.null(reference_labels)) {
    fit@reference_labels
  } else {
    coerce_reference_labels(reference_labels, length(labels))
  }
  if (!is.null(reference) && count_clusters(reference) == length(unique(labels))) {
    labels <- align_cluster_labels(reference, labels, keep = reference >= 0)
  }
  labels <- as.integer(labels)
  if (is.null(noise)) {
    return(labels)
  }
  labels[noise$mask] <- -1L
  scored <- is.finite(scores)
  n_flagged <- sum(noise$mask & scored)
  if (n_flagged < noise$n_target) {
    warning(sprintf(
      "noise_quantile=%s asks for about %d of %d samples by %s; %d were flagged. Samples tied at the cutoff (%.3f) or within %s of the median (%.3f) stay labeled.",
      format(noise_quantile), noise$n_target, sum(scored), noise_score, n_flagged,
      noise$cutoff, format(NOISE_MARGIN), noise$median
    ), call. = FALSE)
  }
  labels
})

# Labels every sample from a cut over the anchors. The anchors keep their
# cut labels, and the classifier the run used for generalizability, trained
# on the anchors, labels the rest. It is one fit outside the resample loop,
# so it gets the run's whole core budget, and it runs under the run's seed,
# as each resample does.
extend_anchor_labels <- function(fit, anchor_labels) {
  anchors <- fit@consensus_anchors
  X <- fit@input_data
  labels <- integer(nrow(X))
  labels[anchors] <- anchor_labels
  rest <- setdiff(seq_len(nrow(X)), anchors)
  if (length(rest) == 0L) {
    return(labels)
  }
  if (length(unique(anchor_labels)) < 2L) {
    # A one-cluster cut gives the classifier a single class.
    labels[rest] <- anchor_labels[[1L]]
    return(labels)
  }
  settings <- fit@run_params
  predict_labels <- default_generalizability_classifier(
    settings$classifier,
    n_features = ncol(X),
    n_trees = settings$n_trees,
    random_state = settings$random_state,
    n_threads = settings$n_threads
  )
  labels[rest] <- seeded(settings$random_state, predict_labels(
    X[anchors, , drop = FALSE],
    anchor_labels,
    X[rest, , drop = FALSE]
  ))
  labels
}
```

- [ ] Step 5: Add the accessors and update two topics in `R/accessors.R`

In the `estimator_results` roxygen block, extend the bullet on `consensus_pac_stability` with one sentence: `On an anchored run it covers anchor pairs only.`

In the `consensus_matrix` roxygen block, add a paragraph after the first one and extend `@return`:

```r
#' On an anchored run (see [carve()]) `consensus_matrix()` returns the block
#' over the anchors, and `consensus_anchors()` their row indices in the data,
#' sorted. For an exact run `consensus_anchors()` returns `NULL`.
```

```r
#' @return `consensus_matrix()` returns an `n` by `n` matrix, or `m` by `m` on
#'   an anchored run, and `sample_scores()` a vector of length `n`. Stability
#'   scores are `NaN` for a sample never drawn with a partner; accuracy is 0
#'   for a sample never held out. `consensus_anchors()` returns an integer
#'   vector or `NULL`.
```

Append after the `sample_scores` method:

```r
#' @rdname consensus_matrix
#' @export
setMethod("consensus_anchors", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@consensus_anchors
})

#' Results of randomized preprocessing
#'
#' After `carve(randomize_preprocessing = TRUE)`, `preprocessing_results()`
#' splits each configuration's scores by the pipeline its resamples used, and
#' `preprocessing_pipelines()` returns those pipelines. Both return `NULL` for
#' a run without randomized preprocessing.
#'
#' The table has one row per configuration, pipeline and sweep value, and
#' joins [estimator_results()] on `method_id` and the sweep column. Its
#' columns:
#'
#' - `method_id` and `method_label`, as in [estimator_results()].
#' - `pipeline`, the pipeline's label and its name in
#'   `preprocessing_pipelines()`, and `normalization` and `dim_reduction`, the
#'   labels of its two steps with the values drawn for them.
#' - The sweep column, such as `n_clusters` or `resolution`.
#' - `n_resamples`, the resamples of this configuration that used the
#'   pipeline.
#' - `ari_stability` and `ari_generalizability`, with standard errors
#'   `ari_stability_se` and `ari_generalizability_se`, over those resamples
#'   only. A criterion the run's `mode` skipped is `NaN`.
#' - `n_clusters_observed`, the mean number of clusters over those
#'   resamples.
#' - `sweep_param`, `sweep_value` and `sweep_rank`, as in
#'   [estimator_results()].
#'
#' @inheritParams get_k
#' @return `preprocessing_results()` returns a data frame and
#'   `preprocessing_pipelines()` a list of pipelines named by label, which
#'   [pipeline_from_spec()] applies to new data; `NULL` for a run without
#'   randomized preprocessing.
#' @seealso [carve()], [preprocessing_option()], [pipeline_from_spec()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_resamples = 8, random_state = 0, randomize_preprocessing = TRUE,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3)),
#'              normalization_options = list(preprocessing_option(Identity),
#'                                           preprocessing_option(StandardScaler)),
#'              dim_reduction_options = list(preprocessing_option(Identity)))
#' preprocessing_results(fit)[, c("pipeline", "n_clusters", "n_resamples", "ari_stability")]
#' names(preprocessing_pipelines(fit))
#' @rdname preprocessing_results
#' @export
setMethod("preprocessing_results", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@preprocessing_results
})

#' @rdname preprocessing_results
#' @export
setMethod("preprocessing_pipelines", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@preprocessing_pipelines
})
```

- [ ] Step 6: Document, then run the tests and see them pass

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "accessors", stop_on_failure = TRUE)'
Rscript -e 'devtools::test(stop_on_failure = TRUE)'
```

Expected: `document()` adds three `export()` and three `exportMethods()` lines and `man/preprocessing_results.Rd`, and rewrites the other three pages. All tests pass with `SKIP 0`.

- [ ] Step 7: Mutation check

- In `get_labels()`, apply the noise mask before the anchored extension (to the anchor labels): the anchored noise test fails.
- In `get_labels()`, read the scores at `selected$config_id + 1L`: the config_id test fails.
- In `get_labels()`, count `sum(noise$mask)` as flagged: the NaN test fails (5 flagged, no warning).
- In `extend_anchor_labels()`, predict every sample and keep the predictions for the anchors too: the anchors-keep-their-labels test fails.
- In `extend_anchor_labels()`, call the classifier outside `seeded()`: the reproducibility test fails if the default forest draws from the session; if it does not, report that and confirm the spy test still passes.
- In `extend_anchor_labels()`, pass `n_threads = 1L`: the budget test fails.

- [ ] Step 8: Commit

```bash
git add carve-r/NAMESPACE carve-r/R/AllGenerics.R carve-r/R/accessors.R carve-r/man/get_labels.Rd \
  carve-r/man/consensus_matrix.Rd carve-r/man/estimator_results.Rd carve-r/man/preprocessing_results.Rd \
  carve-r/tests/testthat/test-accessors.R
git commit -m "$(cat <<'EOF'
feat(carve-r): label anchored runs and noise in get_labels, and add the new accessors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---
### Task 16: Parity with Python, the package check and the writing pass

Files:
- Modify: `carve-r/data-raw/parity_check.R`
- Create (generated): `carve-r/data-raw/parity-report.md` (overwritten)

Interfaces:
- Consumes: the whole package; `reticulate` and `devtools` from the R library; the Python package in `code/.venv`.
- Produces: the parity report, a clean `R CMD check`, the writing pass, and the stage report. This task changes no package code unless the check or the report finds a defect; such a fix goes in its own commit with a test that fails without it.

- [ ] Step 1: Rewrite `carve-r/data-raw/parity_check.R`

```r
# Compares the R package with the Python package on the same data and
# writes data-raw/parity-report.md. Run from code/carve-r:
#
#   Rscript data-raw/parity_check.R
#
# It loads the package from source with devtools, and needs reticulate, the
# dbscan and uwot packages, and the Python environment in ../.venv, where
# carve is installed. The two languages draw different subsamples, so the
# results agree within tolerances, not exactly.

devtools::load_all(".", quiet = TRUE)
# normalizePath() on the interpreter would follow the symlink out of the
# virtual environment, so resolve the environment directory instead.
reticulate::use_python(file.path(normalizePath("../.venv"), "bin", "python"), required = TRUE)
datasets <- reticulate::import("sklearn.datasets")
preprocessing <- reticulate::import("sklearn.preprocessing")
decomposition <- reticulate::import("sklearn.decomposition")
np <- reticulate::import("numpy", convert = FALSE)
pycarve <- reticulate::import("carve")

n_resamples <- 100L
ks <- 2:6

# Largest allowed absolute difference between the R and the Python value of
# one configuration.
tolerance <- c(
  ari_stability = 0.05,
  ari_generalizability = 0.05,
  ari_average = 0.05,
  consensus_pac_stability = 0.03,
  consensus_gini_stability = 0.03,
  consensus_ce_stability = 0.03,
  accuracy_generalizability = 0.03,
  n_clusters_observed = 0.5,
  noise_fraction = 0.02
)

# reticulate converts numeric pandas columns to R vectors but leaves string
# columns as Python objects, so those are rebuilt with tolist().
py_table <- function(frame, columns) {
  out <- frame[, columns]
  for (column in columns) {
    if (!is.atomic(out[[column]])) {
      out[[column]] <- unlist(out[[column]]$tolist())
    }
  }
  out
}

largest_difference <- function(a, b) {
  difference <- abs(a - b)
  if (all(is.na(difference))) NA_real_ else max(difference, na.rm = TRUE)
}

compare_tables <- function(r_table, python_table, keys, columns) {
  joined <- merge(
    r_table[, c(keys, columns)],
    python_table[, c(keys, columns)],
    by = keys,
    suffixes = c("_r", "_py")
  )
  list(
    rows = nrow(joined),
    expected = nrow(r_table),
    differences = vapply(columns, function(column) {
      largest_difference(joined[[paste0(column, "_r")]], joined[[paste0(column, "_py")]])
    }, numeric(1))
  )
}

# The selected k on the k axis, the selected sweep value elsewhere.
compare_selections <- function(r_fit, py_fit, by_value) {
  out <- expand.grid(
    measure = c("stability", "generalizability", "average"),
    rule = c("max", "1se"),
    stringsAsFactors = FALSE
  )
  out$r <- mapply(function(m, r) {
    if (by_value) get_sweep_value(r_fit, measure = m, rule = r) else get_k(r_fit, measure = m, rule = r)
  }, out$measure, out$rule)
  out$py <- mapply(function(m, r) {
    as.numeric(if (by_value) py_fit$get_sweep_value(measure = m, rule = r) else py_fit$get_k(measure = m, rule = r))
  }, out$measure, out$rule)
  out
}

compare_fits <- function(name, r_fit, py_fit, keys, by_value = FALSE, note = NULL,
                         pipelines = FALSE) {
  columns <- names(tolerance)
  out <- list(
    name = name,
    note = note,
    results = compare_tables(estimator_results(r_fit), py_table(py_fit$estimator_results_, c(keys, columns)), keys, columns),
    selections = compare_selections(r_fit, py_fit, by_value),
    label_ari = adjusted_rand_index(get_labels(r_fit), as.integer(py_fit$get_labels()))
  )
  if (pipelines) {
    pipeline_keys <- c("method_label", "pipeline", setdiff(keys, "method_label"))
    pipeline_columns <- c("n_resamples", "ari_stability", "ari_generalizability")
    out$pipelines <- compare_tables(
      preprocessing_results(r_fit),
      py_table(py_fit$preprocessing_results_, c(pipeline_keys, pipeline_columns)),
      pipeline_keys,
      pipeline_columns
    )
  }
  out
}

blobs <- function(n, std, seed, features = 2L, centers = 3L) {
  datasets$make_blobs(
    n_samples = as.integer(n), n_features = as.integer(features), centers = as.integer(centers),
    cluster_std = std, random_state = as.integer(seed)
  )[[1L]]
}

k_case <- function(name, X) {
  r_fit <- carve(X, n_clusters = ks, n_resamples = n_resamples, random_state = 0)
  py_fit <- pycarve$CARVE(n_clusters = np$array(ks), n_resamples = n_resamples, random_state = 0L)$fit(X)
  compare_fits(name, r_fit, py_fit, c("method_label", "n_clusters"))
}

resolution_case <- function() {
  X <- blobs(600, 2.0, 4, features = 10L, centers = 5L)
  resolutions <- c(0.1, 0.25, 0.5, 1, 2)
  r_fit <- carve(X, resolution = resolutions, n_resamples = n_resamples, random_state = 0)
  py_fit <- pycarve$CARVE(resolution = np$array(resolutions), n_resamples = n_resamples, random_state = 0L)$fit(X)
  compare_fits(
    "resolution sweep, 10-dimensional blobs", r_fit, py_fit, c("method_label", "resolution"),
    by_value = TRUE,
    note = "Leiden runs in igraph in R and in leidenalg in Python. Both optimize modularity at each resolution."
  )
}

hdbscan_case <- function() {
  X <- blobs(500, 1.0, 7, centers = 4L)
  sizes <- c(5L, 10L, 20L, 40L)
  r_fit <- carve(X, sweep = "min_cluster_size", sweep_values = sizes, n_resamples = n_resamples, random_state = 0)
  py_fit <- pycarve$CARVE(sweep = "min_cluster_size", sweep_values = np$array(sizes),
                          n_resamples = n_resamples, random_state = 0L)$fit(X)
  compare_fits(
    "min_cluster_size sweep, HDBSCAN", r_fit, py_fit, c("method_label", "min_cluster_size"),
    by_value = TRUE,
    note = "dbscan and scikit-learn can merge tied distances in a different order, so HDBSCAN can select different clusters on some subsamples (see the HDBSCAN help page)."
  )
}

anchored_case <- function(X) {
  # The anchored warning is expected.
  r_fit <- suppressWarnings(carve(X, n_clusters = ks, n_resamples = n_resamples, anchor_threshold = 200,
                                  random_state = 0))
  py_fit <- pycarve$CARVE(n_clusters = np$array(ks), n_resamples = n_resamples, anchor_threshold = 200L,
                          random_state = 0L)$fit(X)
  compare_fits(
    "easy blobs, anchored over 200 of 500 samples", r_fit, py_fit, c("method_label", "n_clusters"),
    note = "The two languages draw different anchors. PAC covers anchor pairs only, so it can differ more than in the exact case."
  )
}

randomized_case <- function() {
  X <- blobs(300, 0.8, 3)
  r_fit <- carve(
    X, n_clusters = 2:4, n_resamples = n_resamples, random_state = 0, randomize_preprocessing = TRUE,
    normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
    dim_reduction_options = list(preprocessing_option(Identity), preprocessing_option(PCA, n_components = 1L))
  )
  identity <- reticulate::tuple(preprocessing$FunctionTransformer, reticulate::dict())
  py_fit <- pycarve$CARVE(
    n_clusters = np$array(2:4), n_resamples = n_resamples, random_state = 0L,
    normalization_options = list(identity, reticulate::tuple(preprocessing$StandardScaler, reticulate::dict())),
    dim_reduction_options = list(identity, reticulate::tuple(decomposition$PCA, reticulate::dict(n_components = list(1L))))
  )$fit(X, randomize_preprocessing = TRUE)
  compare_fits(
    "easy blobs, randomized preprocessing", r_fit, py_fit, c("method_label", "n_clusters"),
    pipelines = TRUE,
    note = "Each of the four pipelines gets 25 of the 100 resamples in both languages, so n_resamples must agree exactly."
  )
}

format_table <- function(comparison, limits) {
  over <- !is.na(comparison$differences) & comparison$differences > limits
  shown <- ifelse(is.na(comparison$differences), "NA", sprintf("%.3f", comparison$differences))
  c(
    sprintf("Rows compared: %d of %d.", comparison$rows, comparison$expected),
    "",
    "| Column | Largest difference | Tolerance |",
    "|---|---|---|",
    sprintf("| %s | %s%s | %s |", names(limits), shown, ifelse(over, " (over)", ""), format(limits))
  )
}

format_case <- function(result) {
  s <- result$selections
  value <- function(x) vapply(x, format_param_value, character(1))
  lines <- c(
    paste("##", result$name),
    "",
    if (!is.null(result$note)) c(result$note, ""),
    format_table(result$results, tolerance),
    "",
    "| Measure | Rule | R | Python |",
    "|---|---|---|---|",
    sprintf("| %s | %s | %s | %s |", s$measure, s$rule, value(s$r), value(s$py)),
    "",
    sprintf(
      "ARI between the R and Python labels at the default selection (stability, 1se): %.3f",
      result$label_ari
    ),
    ""
  )
  if (!is.null(result$pipelines)) {
    lines <- c(
      lines,
      "Rows of preprocessing_results():",
      "",
      format_table(result$pipelines, c(n_resamples = 0, ari_stability = 0.05, ari_generalizability = 0.05)),
      ""
    )
  }
  lines
}

easy <- blobs(500, 0.8, 3)
results <- list(
  k_case("easy blobs", easy),
  k_case("hard blobs", blobs(500, 2.3, 3)),
  k_case("circles", datasets$make_circles(n_samples = 400L, factor = 0.5, noise = 0.05, random_state = 0L)[[1L]]),
  resolution_case(),
  hdbscan_case(),
  anchored_case(easy),
  randomized_case()
)
report <- c(
  "# R and Python parity report",
  "",
  sprintf(
    "Generated on %s with CARVE %s in R and carve %s in Python, %d resamples per configuration and random_state 0. The k cases use the light preset over k = %d to %d.",
    format(Sys.Date()), as.character(utils::packageVersion("CARVE")), pycarve$`__version__`,
    n_resamples, min(ks), max(ks)
  ),
  "",
  unlist(lapply(results, format_case))
)
writeLines(report, "data-raw/parity-report.md")
writeLines(report)
```

- [ ] Step 2: Run it

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript data-raw/parity_check.R
```

Expected: it finishes (stage 1's three cases took about 7 minutes; allow about 20 for all seven) and writes the report. Read every case:

- k cases: the stage 1 report is the baseline; the columns should be within their stage 1 differences.
- `n_resamples` in the per-pipeline table must be 0.000. Anything else is a defect in the allocation.
- Every selected k must match, and selected sweep values should match or sit next to each other on the grid.
- For any column marked `(over)`, run the stage 1 seed study before calling it a defect: rerun the configuration in both languages with `random_state` 0, 1000, 2000, 3000 and 4000 (spaced, because nearby seeds reuse resample streams) and compare the means with the seed-to-seed spread. A gap outside both spreads is a defect to diagnose, not to tolerate.

- [ ] Step 3: Re-check the circles spectral gap stage 1 parked

Stage 1 found circles `SpectralClustering` at k = 2 with Gini and CE stability below Python's on average over five spaced seeds (0.982 against 0.998, 0.969 against 0.996), with Python's mean inside R's spread but not the reverse, and asked for a re-check in this stage. Run this in a scratch file (not committed) from `code/carve-r`:

```r
devtools::load_all(".", quiet = TRUE)
reticulate::use_python(file.path(normalizePath("../.venv"), "bin", "python"), required = TRUE)
pycarve <- reticulate::import("carve")
X <- reticulate::import("sklearn.datasets")$make_circles(n_samples = 400L, factor = 0.5, noise = 0.05, random_state = 0L)[[1L]]
seeds <- c(0, 1000, 2000, 3000, 4000)
columns <- c("consensus_gini_stability", "consensus_ce_stability")
r_grid <- list(estimator_grid(SpectralClustering, n_clusters = 2L, affinity = "self_tuning"))
py_grid <- list(reticulate::tuple(pycarve$SpectralClustering, reticulate::dict(n_clusters = list(2L), affinity = list("self_tuning"))))
r <- t(vapply(seeds, function(s) {
  unlist(estimator_results(carve(X, n_resamples = 100, random_state = s, estimator_param_grids = r_grid))[columns])
}, numeric(2)))
py <- t(vapply(seeds, function(s) {
  results <- pycarve$CARVE(n_resamples = 100L, random_state = as.integer(s), estimator_param_grids = py_grid)$fit(X)$estimator_results_
  c(results$consensus_gini_stability, results$consensus_ce_stability)
}, numeric(2)))
print(rbind(R_mean = colMeans(r), R_sd = apply(r, 2, stats::sd), Py_mean = colMeans(py), Py_sd = apply(py, 2, stats::sd)))
```

Put the printed table in the stage report with a one-line reading. If the gap persists outside both spreads, compare the R and Python self-tuning affinities and eigenvectors on one fixed subsample of the circles before changing any code.

- [ ] Step 4: Check the writing in the package sources

Load the `de-ai-writing` skill first. Then:

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|it's important|in conclusion|\*\*" R/*.R data-raw/parity_check.R || echo "no hits"
Rscript -e 'for (f in c(list.files("R", full.names = TRUE), list.files("tests/testthat", pattern = "[.]R$", full.names = TRUE))) tools::showNonASCIIfile(f)'
```

Expected: `no hits`, and no lines from `showNonASCIIfile`. Then read every roxygen block this stage added or changed (in `estimators.R`, `transforms.R`, `pipeline.R`, `carve.R`, `AllClasses.R` and `accessors.R`) against `~/.claude/skills/de-ai-writing/references/signs.md`, including the signs a grep cannot find: rule of three, elegant variation, contrasts nobody raised, summary endings. Fix any hit by rewriting the sentence, not by swapping a word. The two copied Python warnings with "labelled" (Task 10) stay as they are. List any other hit kept on purpose in the report.

- [ ] Step 5: Run R CMD check

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript -e 'devtools::document()'
R CMD build .
_R_CHECK_FORCE_SUGGESTS_=false R CMD check --no-manual CARVE_2.0.0.tar.gz
```

Expected: `Status: OK`, or NOTEs only. A WARNING or an ERROR fails the stage, as in CI (`error-on: '"warning"'`). Record any NOTE in the report. The HDBSCAN and UMAP examples and tests are guarded, so the check passes with or without dbscan and uwot. Then remove the build output, which is not tracked:

```bash
rm -rf CARVE.Rcheck CARVE_2.0.0.tar.gz
git status --short
```

Expected: only `data-raw/parity_check.R` and `data-raw/parity-report.md` changed, plus any `man/` or `NAMESPACE` change from `document()` (there should be none).

- [ ] Step 6: Commit

```bash
git add carve-r/data-raw/parity_check.R carve-r/data-raw/parity-report.md
git commit -m "$(cat <<'EOF'
test(carve-r): compare the resolution, HDBSCAN, anchored and randomized runs with Python

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 7: Write the stage report

Write `_claude_playground/r-rework-stage2-records/stage2-report.md` (outside the repository, as stage 1's records were): the final test count and `SKIP 0`, the `R CMD check` status and NOTEs, the parity report's over-tolerance rows with their seed-study reading, the circles re-check table, every plan defect found and how it was fixed (with the failing output), every mutation that did not fail as the plan predicted, and the writing pass result.

---

## After stage 2

The package fits and queries all three sweep axes, handles noise, anchors large runs and randomizes preprocessing. The next plan (stage 3, written in its own conversation) adds the eight plots, the `SingleCellExperiment` and `Seurat` methods of `carve()`, `run_carve()`, `attach_results()` and the `pbmc3k_subset` data set, and adds the PBMC 3k resolution case to the parity script. Stage 3 also decides the Seurat dispatch (spec open item 3) and the PBMC 3k download (open item 6). The branch is merged into `main` only after stage 4.

Notes for stage 3 from this plan:

- `plot_metric_by_pipeline()` reads `preprocessing_results()` and selects with `method_id`; `plot_n_clusters_over_sweep()` reads `n_clusters_observed` and `n_clusters_observed_se`, both present on every axis.
- `attach_results()` skips the consensus matrix with a warning on an anchored run (the spec's storage section); `consensus_anchors()` tells it when.
- The scatter plots take `basis`; under randomized preprocessing, `pipeline_from_spec()` gives the embedding of the best pipeline.
