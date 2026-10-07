# R Package Rework, Stage 3 (Plots and Single-Cell Objects) Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: add the eight plots of the Python package to the R package, as ggplot2 plots of a `CARVE` fit and of a `SingleCellExperiment` or `Seurat` object; let `carve()` take those two classes; add `run_carve()` and `attach_results()`, which store the selected configuration's results in the object; and ship `pbmc3k_subset`, 1,000 PBMC 3k cells, which the parity script fits in both languages.

Architecture: the stage 1 and 2 structure stays: one R file per Python module, tests one to one with the R files, deterministic pieces checked against JSON fixtures written from the Python functions. `R/plotting.R` ports `_plotting.py`: functions that draw from a results table or from per-sample vectors. `R/plots.R` holds the eight S4 generics' methods: the `CARVE` methods port the plot methods of `api.CARVE`, and the `SingleCellExperiment` methods, with `ANY` methods that take Seurat objects, port `carve.pl`. `R/sce.R` and `R/seurat.R` read and write the two classes, and `R/tl.R` ports `tl/_carve.py` on top of them.

Tech stack: R 4.5.1 (DESCRIPTION floor 4.1.0), methods (S4), ggplot2 4.0.3 (floor 3.5.0), scales 1.4.0, SingleCellExperiment 1.30.1, SummarizedExperiment 1.38.1, S4Vectors 0.46.0, SeuratObject 5.4.0 (suggested), Seurat 5.5.0 (only for `data-raw/pbmc3k_subset.R`), testthat 3.3.2 (edition 3), roxygen2 8.0.0 with markdown, devtools 2.5.2; Python 3.12 in `code/.venv` (carve 1.0.0, matplotlib) for the fixture and parity scripts.

Spec: `docs/superpowers/specs/2026-10-06-r-package-rework-design.md` (commit 0b8d5b4). This plan implements stage 3 of its four stages and builds on stages 1 and 2 as executed (`docs/superpowers/plans/2026-10-06-r-package-rework-stage1.md` and `-stage2.md`, commits 17cac7e..d5092b2). Executors read the spec and this plan. Where this plan departs from the spec, the departure is listed under Decisions.

## Global Constraints

- Branch: `r-package-rework` in the `code/` repository, at d5092b2 (stage 2 complete) plus this plan's commit when the plan starts. Commit after every task. Stage files by explicit path only; never `git add -A`, `git add .` or `git commit -a`.
- Run R commands from `code/carve-r/`. Run the fixture script from `code/`. Commands use the R on PATH (4.5.1) and the Python in `code/.venv`.
- Do not touch anything outside `carve-r/` except this plan's own commit: not `src/`, `tests/`, `notebooks/`, the spec, or `../overleaf/`.
- Dependencies added in this stage, each in the task that first uses it: Imports `ggplot2 (>= 3.5.0)` and `scales` (Task 2), `S4Vectors`, `SingleCellExperiment` and `SummarizedExperiment` (Task 7); Suggests `SeuratObject` (Task 8). `LazyData: true` (Task 12). Keep Imports and Suggests in the case-insensitive alphabetical order stage 1 used. An import declared before any code uses it gives an `R CMD check` NOTE.
- Exports added in this stage: the generics and their methods `plot_metric_over_n_clusters`, `plot_metric_by_pipeline`, `plot_n_clusters_over_sweep`, `plot_consensus_matrix`, `plot_cluster_boxplot`, `plot_cluster_violin`, `plot_cluster_scatter`, `plot_diagnostic_scatter` (methods for `CARVE`, `SingleCellExperiment` and `ANY`); the functions `run_carve` and `attach_results`; the `carve()` method for `SingleCellExperiment`; and the data set `pbmc3k_subset`. Nothing else. Tests reach internals through the namespace `devtools::test()` loads.
- Defaults copied from Python: every plot argument keeps its Python name and default, except the ones Decision 5 changes; `run_carve(key = "carve", measure = "stability", rule = "1se", not_two = FALSE, store_consensus = TRUE, store_results = TRUE, mode = "default", random_state = 0)`; `attach_results(key = "carve", measure = "stability", rule = "1se", not_two = FALSE, store_consensus = TRUE, store_results = TRUE, mode = "default")`.
- Never call `library()` or `require()` on a dependency in `R/`, `tests/` or examples; write `ggplot2::`, `SingleCellExperiment::` and so on. The ggplot2, SeuratObject and Seurat binaries here were built under R 4.5.2, and `library()` warns about that on R 4.5.1, which fails the suite under `warn = 2`. `loadNamespace()` does not warn (checked while writing this plan).
- Plot tests read the plot object: `ggplot2::layer_data()`, `ggplot2::get_guide_data()` (its legend text is in the `.label` column, which carries a `pos` attribute; compare `as.character(...)`), `plot$scales$get_scales()`, `plot$labels`, and the raster of an `annotation_raster()` layer at `layer$geom_params$raster`. All of these work in ggplot2 4.0.3 (checked while writing this plan). There are no image snapshot tests; vdiffr is not installed.
- Labels: `get_labels()` returns 1-based labels, and the plots show label values as they are. A stored label column is a factor; the plots read its codes (1, 2, ...), as Python reads the categorical codes and adds one.
- `config_id` is a join key, never a position.
- Diagnostics use `warning(..., call. = FALSE)`. No logging package, and no `print()` or `cat()` outside `show` and `print` methods.
- Error and warning text copies Python's wording. Where Python names an AnnData object, a Python function or a Python value, the R text names the R object path (`colData(object)$carve`, `metadata(object)$carve`, `object@meta.data$carve`, `Misc(object, 'carve')`), the R function (`run_carve`) and the R value (`TRUE`, `FALSE`, `NULL`), and keeps the rest of the sentence. The exact strings are in the tasks.
- Unexpected warnings fail the suite (`tests/testthat/setup.R` sets `warn = 2`). A test that expects warnings captures all of them, with `expect_warning` (first match only, the rest still fail), `collect_warnings()`, `suppressWarnings()` or the `muffle()` helper of Task 7.
- Suggested packages: tests that need SeuratObject start with `skip_if_not_installed("SeuratObject")`. It is installed here (5.4.0), as are dbscan and uwot, so every task report shows `SKIP 0`. A skip means the environment changed; stop and report it.
- Assertions must be able to fail. After a task's tests pass, apply the mutations its last step names, one at a time, confirm the named test fails, and restore the code with `git checkout -- <file>`. Report each result.
- Writing: every roxygen text and code comment follows the spec's writing standard. The roxygen text in this plan is written to it; keep it as written. If you add or change any user-facing text, load the `de-ai-writing` skill first. American spelling, plain prose, no bold or italics, sentence-case Rd titles.
- No non-ASCII characters in `R/` or `tests/` files (an `R CMD check` warning). Write a non-ASCII character in a string as an escape such as `"\u2248"`.
- Stage 1's rulings still apply: commit DESCRIPTION as `devtools::document()` leaves it; every task that adds a file under `R/` runs `devtools::document()` before committing and stages DESCRIPTION, NAMESPACE and the changed `man/` files (the Collate field); a generic declared with `@rdname X` in `AllGenerics.R` and its methods documented elsewhere all carry `@rdname X`, so each topic is one page.
- The plan's code was written without being run, except for the checks recorded under Decisions. Treat it as a detailed proposal. When plan code or a plan test is wrong, fix it, say so in the report with the failing output and why, and keep the assertion's intent. Never loosen an assertion to make it pass against code that disagrees with it.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Decisions this plan makes that the spec left open or changes

Items 2, 4, 5, 6, 7, 8, 11 and 16 change what a user sees or the spec's text, and are flagged to the author with the plan.

1. Seurat dispatch (spec open item 3). `Seurat` is an S4 class of a suggested package, so no method can name it. The `ANY` method of `carve()` and of each plot generic checks `methods::is(x, "Seurat")`, loads SeuratObject with `require_package()`, and handles the object; any other class gets an error naming the classes the function takes. `run_carve()` and `attach_results()` are plain functions that call `cells_kind()`, which tells the two classes apart. `methods::is()` matches the class name even when SeuratObject is not loaded yet.
2. For a Seurat object, `assay` names a Seurat assay (`"RNA"`), and CARVE clusters that assay's `"data"` layer; the spec gives the Seurat method the same three arguments as the SingleCellExperiment one and does not say how a layer is chosen. With no `assay` and no `"pca"` reduction, the default assay's `"data"` layer is used.
3. Plot files. The spec maps `_plotting.py` and `pl/_plots.py` to one `plotting.R`. This plan splits the drawing (`plotting.R`, about 650 lines) from the generic methods (`plots.R`, about 700 lines), adds `tl.R` for `run_carve()`, `attach_results()` and the dispatch between the two single-cell classes, and `data.R` for the data set's help page. `sce.R` and `seurat.R` hold what is particular to each class.
4. No patchwork. The consensus heatmap and its cluster band are one ggplot: the matrix and the band are `annotation_raster()` layers, one above the other in the same panel, and an invisible tile layer carries the fill scale so the color bar shows. Measured while writing this plan on a 5,000 by 5,000 matrix (the default anchor count): `annotation_raster()` took 1.2 s and 1.1 GB, `geom_raster()` on the long table 21 s and 5 GB. Nothing else needs patchwork, so it is not imported, although the spec lists it. The plot returns a ggplot object, which the spec allows.
5. Plot arguments keep Python's names and defaults, with these changes. `ax`, `figsize`, `show`, `save`, `dpi` and `**kwargs` are dropped (spec). `legend_loc` takes ggplot2 positions, `"right"` (the default everywhere), `"left"`, `"top"`, `"bottom"` or two numbers inside the panel; matplotlib's `"best"` and `"right margin"` have no ggplot2 equivalent. `palette` and `cmap` take a palette name (`"Accent"` and the other `grDevices::palette.pals()` names, the viridis options, `"Greens"`, `"Greens_r"`) or a vector of colors. Sizes and widths keep matplotlib's units (marker area in square points, line widths in points) and are converted, so Python's defaults give marks of about the same size.
6. Annotations. matplotlib places the annotation box where it hides the least data; ggplot2 cannot, so the annotation is the plot caption, below the panel, or with `annotation_style = "legend"` on the scatter plots the first lines of the legend title, as in Python. On a sweep other than `n_clusters` the cluster count is written `k ~ 7`; Python writes `k ≈ 7` (U+2248), which R's `pdf()` device, and so `ggsave("plot.pdf")`, prints as dots.
7. Diagnostic scatter markers. The plot shows the score as the marker fill, and ggplot2 has five shapes with a fill (21 to 25), so the default `markers` are those five in Python's order (circle, square, up triangle, diamond, down triangle). Python's default list has 14. With more than five clusters the shapes repeat, with Python's warning.
8. `density_norm` maps to `geom_violin(scale = )`: `"width"` and `"count"` as in ggplot2, and `"area"` gives equal areas. Python draws `"area"` like `"width"` and scales `"count"` by width (`0.2 + 0.6 n / max(n)`). Stage 4's `python-users.Rmd` lists the difference.
9. Plot validation that Python lacks: `inner`, `annotation_style`, `legend_loc` and `markers` are checked, with new messages, where Python ignores an unknown value. A cluster with one finite score gets no violin body (ggplot2 would drop it with a warning) and keeps its point and inner marks. The strip plot's jitter is seeded (`position_jitter(seed = 0)`), so a plot is the same each time and leaves the session's random stream alone; Python's jitter is unseeded. Samples whose score is `NaN` are left out of `plot_cluster_scatter()` (Python leaves them out too, except when all other scores are equal).
10. Plot generics take `object` first, a fit or a single-cell object, where the accessors take `fit`.
11. `attach_results()` gains `assay`, `reduction` and `n_dims` (Python's `use_rep`, `layer` and `n_pcs`), recorded so that the scatter plots can rebuild the data; the spec's signature leaves them out. `run_carve()` takes `assay`, `reduction`, `n_dims` and `mode` as named arguments after `...`, because it passes them to both `carve()` and `attach_results()`.
12. The record. Each call replaces the whole record, so a matrix from an earlier call never outlives a call with `store_consensus = FALSE`; Python leaves an old `.obsp` matrix in place. Score columns from an earlier call with the same key stay, as in Python. Seurat records are written to the `misc` slot directly, because SeuratObject's `Misc<-` warns whenever it replaces an entry (checked while writing this plan).
13. `run_params$estimator_param_grids` records `"light"`, `"full"` or `"custom"`, which `attach_results()` stores as Python stores `model.estimator_param_grids`.
14. `pbmc3k_subset` is a list (`pca`, `umap`, `seurat_clusters`), not a SingleCellExperiment or Seurat object, so stage 4 can build either object from it in two lines. Its scores are rounded to four decimals to keep the file near the spec's 150 KB.
15. PBMC 3k download (spec open item 6): the 10x Genomics URL of Seurat's tutorial, `https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz`, cached in `carve-r/data-raw/cache/` (git-ignored, and outside the build because `data-raw` is in `.Rbuildignore`). Stage 4's tutorial caches in `notebooks/data/`, as the spec says.
16. Seurat (the full package) goes into Suggests in stage 4, with the vignette that first uses it; this stage needs only SeuratObject.
17. `code/.gitignore` ignores every `data/` directory, which would hide `carve-r/data/`. `carve-r/.gitignore` re-includes it with `!/data/`.
18. Parity. The engine does not change in this stage, so the seven stage 2 cases must reproduce the stage 2 report exactly, apart from the date line; any other change is a defect. The PBMC 3k resolution case is added. The two stage 2 residuals (randomized SpectralClustering consensus at k = 3 and 4; anchored SpectralClustering k = 5 accuracy) are re-read in the new report and stay open for the author; this plan does not diagnose them.

Stage 2 items this plan does not touch: the BSD-3 attribution of the HDBSCAN port and the stale HDBSCAN text of the spec (author decisions before the merge); the hedged HDBSCAN help sentence, version floors, `isTRUE()` for a NaN stability and the `python-users.Rmd` items (stage 4).

## File structure

```
carve-r/
  DESCRIPTION                      Imports ggplot2, scales (Task 2); S4Vectors,
                                   SingleCellExperiment, SummarizedExperiment (Task 7);
                                   Suggests SeuratObject (Task 8); LazyData (Task 12)
  .gitignore                       !/data/ and /data-raw/cache/ (Task 12)
  R/CARVE-package.R                @importFrom ggplot2 .data (Task 2)
  R/plotting.R                     new: _plotting.py (Tasks 2 to 6)
  R/AllGenerics.R                  + the eight plot generics (Tasks 10, 11)
  R/plots.R                        new: plot methods of api.CARVE and carve.pl
                                   (Tasks 10, 11)
  R/utils.R                        + check_one_source, cells_by_features, select_dims,
                                   embedding_2d (Task 7)
  R/sce.R                          new: SingleCellExperiment input and storage,
                                   carve() method (Task 7)
  R/seurat.R                       new: Seurat input and storage (Task 8)
  R/carve.R                        ANY method routes Seurat objects (Task 8);
                                   run_params$estimator_param_grids (Task 9)
  R/AllClasses.R                   run_params doc line (Task 9)
  R/tl.R                           new: tl/_carve.py and the class dispatch (Task 9)
  R/data.R                         new: pbmc3k_subset help page (Task 12)
  data/pbmc3k_subset.rda           new (Task 12)
  data-raw/pbmc3k_subset.R         new (Task 12)
  data-raw/make_fixtures.py        + plotting() (Task 1)
  data-raw/parity_check.R          + PBMC 3k case (Task 13)
  data-raw/parity-report.md        regenerated (Task 13)
  tests/testthat/fixtures/         + plotting.json (Task 1)
  tests/testthat/helper-plots.R    new: Python test tables, plot readers (Tasks 3, 4,
                                   10)
  tests/testthat/helper-data.R     + make_sce, make_seurat, muffle (Tasks 7, 8)
  tests/testthat/test-*.R          one per R file; test-plotting.R, test-plots.R,
                                   test-sce.R, test-seurat.R, test-tl.R and
                                   test-data.R are new
```

---

### Task 1: Python fixtures for the plots

Files:
- Modify: `carve-r/data-raw/make_fixtures.py`
- Create (generated): `carve-r/tests/testthat/fixtures/plotting.json`

Interfaces:
- Consumes: the Python package in `code/.venv`, with matplotlib.
- Produces: `plotting.json`, read with `read_fixture("plotting")` (`simplifyDataFrame = FALSE`; a JSON `null` field reads back as `NULL`). Labels are 0-based, as in Python. Fields:
  - `palette`: a list of 12 character vectors; element `n` holds the `n` colors `plt.get_cmap("Accent")(np.linspace(0, 1, n))` gives, as lowercase hex.
  - `ylabels`: a named list from each results column of `MEASURE_MAP` to `_measure_ylabel(column)`.
  - `selections`: a list of `list(param, rule, value, label)`, `label` from `_selection_label()`.
  - `annotations`: a list of `list(param, value, method_label, measure, rule, selected_k, pinned, tight_layout, text)`; `selected_k` may be `NULL`; `text` from `_get_annotation()` on a one-row table.
  - `score_groups`: `scores` (6 values, one `NA`), `labels` (6 values), and `cases`, each `list(order, groups, kept_order)`; `order` may be `NULL`; `groups` is a list of numeric vectors; `kept_order` is Python's 1-based display order.
  - `scatter`: `X` (12 by 2), `labels` (12), `scores` (12), and from `plot_cluster_scatter(sort_order=False)`: `sizes`, `alphas`, `legend` (the legend texts), `flat_sizes` and `flat_alphas` (all scores 0.8).
  - `diagnostic`: `scores` (the scatter scores with element 6, 0-based 5, set to `NA`), and from `plot_diagnostic_scatter()`: `order_unsorted`, `alpha_unsorted`, `order_sorted`, `alpha_sorted`, the 0-based sample index and the alpha of every marker in drawing order, with `sort_order` `False` and `True`.

- [ ] Step 1: Check the environment

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
.venv/bin/python -c "import matplotlib, carve; print(matplotlib.__version__)"
Rscript -e 'for (p in c("ggplot2", "scales", "SingleCellExperiment", "SeuratObject")) cat(p, as.character(packageVersion(p)), "\n")'
git status --short
git log --oneline -2
```

Expected: a matplotlib version; `ggplot2 4.0.3`, `scales 1.4.0`, `SingleCellExperiment 1.30.1`, `SeuratObject 5.4.0`; a clean tree; HEAD is this plan's commit, with d5092b2 below it.

- [ ] Step 2: Add the fixture writer before `if __name__ == "__main__":` in `data-raw/make_fixtures.py`

```python
def plotting():
    # The plotting functions draw with matplotlib, which needs a backend
    # that opens no window.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from carve._plotting import (
        _get_annotation,
        _measure_ylabel,
        _prepare_cluster_score_groups,
        _selection_label,
        plot_cluster_scatter,
        plot_diagnostic_scatter,
    )
    from carve._selection import MEASURE_MAP

    accent = plt.get_cmap("Accent")
    palette = [[to_hex(c) for c in accent(np.linspace(0, 1, n))] for n in range(1, 13)]
    ylabels = {col: _measure_ylabel(col) for col in sorted(set(MEASURE_MAP.values()))}

    selections = []
    for param, rule, value in [
        ("n_clusters", "1se", 4),
        ("n_clusters", "max", 10),
        ("resolution", "max", 0.25),
        ("resolution", "quantile", 1.0),
        ("min_cluster_size", "1se", 40),
        ("resolution", "max", 1e-05),
        ("resolution", "max", 100000.0),
        ("resolution", "max", 1234567.0),
    ]:
        row = pd.Series({"sweep_value": value})
        label = _selection_label(row, param=param, rule=rule)
        selections.append({"param": param, "rule": rule, "value": value, "label": label})

    annotations = []
    for param, value, method_label, cases in [
        (
            "n_clusters",
            3,
            "AgglomerativeClustering, linkage=ward",
            [
                ("stability", "1se", 3, False, False),
                ("ari_generalizability", "max", 3, True, False),
                ("average", "quantile", 3, False, True),
            ],
        ),
        (
            "resolution",
            0.5,
            "LeidenClustering, n_neighbors=15, weighting=connectivity",
            [
                ("stability", "1se", 7, False, False),
                ("generalizability", "max", 7, True, True),
                ("gini", "max", None, False, False),
            ],
        ),
    ]:
        results = pd.DataFrame(
            {"sweep_param": [param], "sweep_value": [value], "method_label": [method_label]}
        )
        for measure, rule, selected_k, pinned, tight in cases:
            text = _get_annotation(
                measure=measure,
                rule=rule,
                estimator_results=results,
                row=results.iloc[0],
                selected_k=selected_k,
                pinned=pinned,
                tight_layout=tight,
            )
            annotations.append(
                {
                    "param": param,
                    "value": value,
                    "method_label": method_label,
                    "measure": measure,
                    "rule": rule,
                    "selected_k": selected_k,
                    "pinned": pinned,
                    "tight_layout": tight,
                    "text": text,
                }
            )

    group_scores = np.array([0.9, np.nan, 0.7, 0.6, 0.95, 0.3])
    group_labels = np.array([2, 0, 0, 2, 1, 1])
    group_cases = []
    for order in (None, [2, 0]):
        groups, kept = _prepare_cluster_score_groups(group_scores, group_labels, order=order)
        group_cases.append({"order": order, "groups": [list(g) for g in groups], "kept_order": kept})

    X = np.random.RandomState(0).randn(12, 2)
    labels = np.repeat([0, 1, 2], 4)
    scores = np.random.RandomState(1).rand(12)
    ax = plot_cluster_scatter(X, labels, scores, sort_order=False)
    points = ax.collections[0]
    scatter = {
        "X": X,
        "labels": labels,
        "scores": scores,
        "sizes": points.get_sizes(),
        "alphas": points.get_facecolors()[:, 3],
        "legend": [t.get_text() for t in ax.get_legend().get_texts()],
    }
    ax = plot_cluster_scatter(X, labels, np.full(12, 0.8), sort_order=False)
    scatter["flat_sizes"] = ax.collections[0].get_sizes()
    scatter["flat_alphas"] = ax.collections[0].get_facecolors()[:, 3]

    diagnostic_scores = scores.copy()
    diagnostic_scores[5] = np.nan
    diagnostic = {"scores": diagnostic_scores}
    for sort_order, name in ((False, "unsorted"), (True, "sorted")):
        ax = plot_diagnostic_scatter(X, labels, diagnostic_scores, sort_order=sort_order)
        order, alpha = [], []
        # One collection per cluster, in drawing order. Each marker is found
        # in X by its coordinates, which plot_diagnostic_scatter passes
        # through unchanged for two-column data.
        for collection in ax.collections:
            for offset, face in zip(collection.get_offsets(), collection.get_facecolors()):
                order.append(int(np.flatnonzero((X == offset).all(axis=1))[0]))
                alpha.append(face[3])
        diagnostic[f"order_{name}"] = order
        diagnostic[f"alpha_{name}"] = alpha
    plt.close("all")

    write(
        "plotting",
        {
            "palette": palette,
            "ylabels": ylabels,
            "selections": selections,
            "annotations": annotations,
            "score_groups": {
                "scores": group_scores,
                "labels": group_labels,
                "cases": group_cases,
            },
            "scatter": scatter,
            "diagnostic": diagnostic,
        },
    )
```

Add `plotting,` at the end of the tuple under `if __name__ == "__main__":`, after `pipeline_labels,`.

- [ ] Step 3: Run the script

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
.venv/bin/python carve-r/data-raw/make_fixtures.py
git status --short carve-r/
```

Expected: `wrote plotting.json` among the lines, and `git status` shows only `make_fixtures.py` modified and `fixtures/plotting.json` new. Any other fixture changing means the Python package moved since stage 2; stop and report it.

- [ ] Step 4: Read the fixture back in R

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript -e 'f <- jsonlite::fromJSON("tests/testthat/fixtures/plotting.json", simplifyDataFrame = FALSE); str(f$palette[[3]]); str(f$annotations[[6]]$selected_k); cat(f$annotations[[4]]$text, "\n"); str(f$diagnostic$order_sorted)'
```

Expected: `chr [1:3] "#7fc97f" "#386cb0" "#666666"`, `NULL`, the two-line annotation of the resolution case ending in `Stability, 1-SE rule` with `k ≈ 7` in it, and 12 indices.

- [ ] Step 5: Commit

```bash
git add carve-r/data-raw/make_fixtures.py carve-r/tests/testthat/fixtures/plotting.json
git commit -m "$(cat <<'EOF'
test(carve-r): write Python fixtures for the plot colors, text and encodings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Colors, text and sizes of the plots

Files:
- Create: `carve-r/R/plotting.R`
- Modify: `carve-r/DESCRIPTION` (Imports `ggplot2 (>= 3.5.0)` and `scales`)
- Modify: `carve-r/R/CARVE-package.R` (`@importFrom ggplot2 .data`)
- Create: `carve-r/tests/testthat/test-plotting.R`

Interfaces:
- Consumes: `title_case()`, `format_repr()`, `python_list()` (`utils.R`); `MEASURE_MAP` (`selection.R`); `sweep_param_name()` (`sweep.R`); the `plotting` fixture.
- Produces (all internal, in `R/plotting.R`):
  - `ACCENT` (8 colors), `GREENS` (9 colors), `VIRIDIS_OPTIONS` (8 names), `DIAGNOSTIC_MARKERS` (`c(21L, 22L, 24L, 23L, 25L)`).
  - `colormap_values(palette, argument = "palette")` returns the colors a palette argument names.
  - `palette_colors(palette, n)` returns `n` colors sampled as matplotlib samples a colormap.
  - `continuous_colours(cmap)` returns at least two colors for a continuous scale.
  - `colour_lut(colours, n = 256L)` and `lut_colours(values, lut)`: a lookup table over [0, 1] and the colors of values under it.
  - `format_g(x)` formats a number as Python's `f"{x:g}"`.
  - `measure_ylabel(column)`, `rule_title(rule)`, `selected_value_label(row, param, rule)`, `build_estimator_label(row, tight_layout = FALSE)`, `annotation_text(measure, rule, results, row, selected_k, pinned = FALSE, tight_layout = FALSE)`.
  - `plot_measure_column(results, measure)` returns the column behind `measure`, with Python's plotting messages.
  - `integer_breaks(limits)`, `point_size(s)`, `edge_stroke(lw)`, `line_width(lw)`, `carve_theme()`, `legend_position(legend_loc)`.

- [ ] Step 1: Write the failing tests

Create `carve-r/tests/testthat/test-plotting.R`:

```r
test_that("palette_colors samples a palette as matplotlib samples a colormap", {
  f <- read_fixture("plotting")
  for (n in seq_along(f$palette)) {
    expect_identical(palette_colors("Accent", n), toupper(unlist(f$palette[[n]])), info = paste("n =", n))
  }
  expect_identical(palette_colors(c("red", "blue"), 3L), c("red", "blue", "blue"))
  expect_identical(palette_colors("Accent", 0L), character())
})

test_that("a palette is a palette name or a vector of colors", {
  expect_identical(colormap_values("Accent"), ACCENT)
  expect_identical(colormap_values("Set 1"), unname(grDevices::palette.colors(palette = "Set 1")))
  expect_identical(colormap_values("viridis"), scales::viridis_pal(option = "viridis")(256L))
  expect_identical(colormap_values("Greens"), GREENS)
  expect_identical(colormap_values("Greens_r"), rev(GREENS))
  expect_identical(colormap_values(c(a = "red", b = "#000000")), c("red", "#000000"))
  expect_error(
    colormap_values("nope"),
    "palette must be a palette name such as 'Accent' or 'viridis', or a vector of colors.",
    fixed = TRUE
  )
  expect_error(colormap_values(3), "palette must be a palette name", fixed = TRUE)
  expect_identical(continuous_colours("Greens_r"), rev(GREENS))
  expect_error(continuous_colours("red"), "cmap must name a colormap or give at least two colors.", fixed = TRUE)
  expect_error(continuous_colours("nope"), "cmap must be a palette name such as 'Accent'", fixed = TRUE)
})

test_that("values map to colors through a 256-level table, as in matplotlib", {
  lut <- colour_lut(c("#000000", "#FFFFFF"))
  expect_length(lut, 256L)
  expect_identical(lut[[1L]], "#000000")
  expect_identical(lut[[256L]], "#FFFFFF")
  expect_identical(lut_colours(c(0, 0.5, 1), lut), lut[c(1L, 129L, 256L)])
  expect_identical(lut_colours(c(1 / 256 - 1e-9, 1 / 256), lut), lut[c(1L, 2L)])
})

test_that("axis, selection and annotation text match Python's", {
  f <- read_fixture("plotting")
  for (column in names(f$ylabels)) {
    expect_identical(measure_ylabel(column), f$ylabels[[column]], info = column)
  }
  for (case in f$selections) {
    row <- data.frame(sweep_value = case$value)
    expect_identical(selected_value_label(row, case$param, case$rule), case$label)
  }
  for (case in f$annotations) {
    results <- data.frame(sweep_param = case$param, sweep_value = case$value, method_label = case$method_label)
    text <- annotation_text(
      case$measure, case$rule, results, results[1L, , drop = FALSE], case$selected_k,
      pinned = case$pinned, tight_layout = case$tight_layout
    )
    # Python writes the approximate cluster count with U+2248, R with "~".
    expect_identical(text, gsub("\u2248", "~", case$text, fixed = TRUE))
  }
})

test_that("the estimator label is the row's method label", {
  row <- data.frame(estimator = "KMeans", method_label = "KMeans, linkage=ward", gamma = 0.123456)
  expect_identical(build_estimator_label(row), "KMeans, linkage=ward")
  wide <- data.frame(method_label = "AgglomerativeClustering, linkage=ward")
  expect_identical(build_estimator_label(wide, tight_layout = TRUE), "AgglomerativeClustering\nlinkage=ward")
  expect_identical(build_estimator_label(data.frame(method_label = "KMeans"), tight_layout = TRUE), "KMeans")
})

test_that("plot_measure_column names Python's plotting errors", {
  results <- data.frame(ari_stability = 1)
  expect_identical(plot_measure_column(results, "s"), "ari_stability")
  expect_error(
    plot_measure_column(results, "nonexistent"),
    "Measure 'nonexistent' not found. Valid options: ['s', 'stab',",
    fixed = TRUE
  )
  expect_error(
    plot_measure_column(results, "pac"),
    "Metric column 'consensus_pac_stability' not found in the results table.",
    fixed = TRUE
  )
})

test_that("integer_breaks keeps whole numbers inside the limits", {
  expect_identical(integer_breaks(c(1.8, 3.2)), c(2, 3))
  expect_identical(integer_breaks(c(3, 11)), c(2, 4, 6, 8, 10, 12))
  expect_identical(integer_breaks(c(2.2, 2.8)), numeric())
})

test_that("sizes and widths convert from matplotlib's units", {
  # A matplotlib marker of area 36 square points is 6 points across; a
  # ggplot2 circle is 0.75 * size * .pt points across.
  expect_equal(0.75 * point_size(36) * ggplot2::.pt, 6)
  # 1 point is 96 / 72 grid line-width units; ggplot2 draws a stroke s with
  # s * .stroke / 2 units and a line width w with w * .pt units.
  expect_equal(edge_stroke(0.6) * ggplot2::.stroke / 2, 0.6 * 96 / 72)
  expect_equal(line_width(1.5) * ggplot2::.pt, 1.5 * 96 / 72)
})

test_that("legend_position takes ggplot2 positions", {
  expect_identical(legend_position("bottom")$legend.position, "bottom")
  inside <- legend_position(c(0.9, 0.1))
  expect_identical(inside$legend.position, "inside")
  expect_identical(inside$legend.position.inside, c(0.9, 0.1))
  expect_error(
    legend_position("best"),
    "legend_loc must be 'right', 'left', 'top', 'bottom', or two numbers giving a position inside the panel.",
    fixed = TRUE
  )
})
```

- [ ] Step 2: Run the tests to see them fail

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript -e 'devtools::test(filter = "plotting")'
```

Expected: FAIL, `could not find function "palette_colors"` and the like.

- [ ] Step 3: Add the dependencies

In `DESCRIPTION`, add `ggplot2 (>= 3.5.0),` to Imports after `FNN,`, and `scales,` after `Rtsne,`. The floor is the first ggplot2 with `legend.position.inside`, guide positions and `get_guide_data()`.

In `R/CARVE-package.R`, add a line to the roxygen block, after the two `@importFrom methods` lines:

```r
#' @importFrom ggplot2 .data
```

`.data` is the pronoun the plots use inside `ggplot2::aes()`; without the import, `R CMD check` reports it as an undefined global variable.

- [ ] Step 4: Create `R/plotting.R`

```r
#' @include AllGenerics.R
NULL

# Drawing for the plots of a CARVE fit. Mirrors _plotting.py: the functions
# here draw from a results table or from per-sample vectors, and plots.R
# picks those inputs from a fit or from a single-cell object. Every plot is
# a ggplot object.

# The colors of the Python package's defaults: matplotlib's Accent and
# Greens colormaps, which are ColorBrewer's. viridis comes from scales.
ACCENT <- c("#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17", "#666666")
GREENS <- c(
  "#F7FCF5", "#E5F5E0", "#C7E9C0", "#A1D99B", "#74C476",
  "#41AB5D", "#238B45", "#006D2C", "#00441B"
)
VIRIDIS_OPTIONS <- c("magma", "inferno", "plasma", "viridis", "cividis", "rocket", "mako", "turbo")
# The ggplot2 shapes with a fill, in the order of Python's first five
# markers: circle, square, up triangle, diamond, down triangle.
DIAGNOSTIC_MARKERS <- c(21L, 22L, 24L, 23L, 25L)

is_colour <- function(x) {
  vapply(x, function(value) {
    tryCatch({
      grDevices::col2rgb(value)
      TRUE
    }, error = function(e) FALSE)
  }, logical(1), USE.NAMES = FALSE)
}

# The colors a palette or cmap argument names: "Accent", a name from
# grDevices::palette.pals(), a viridis option, "Greens" or "Greens_r", or a
# vector of colors.
colormap_values <- function(palette, argument = "palette") {
  if (is.character(palette) && length(palette) == 1L && !is.na(palette)) {
    if (identical(palette, "Accent")) {
      return(ACCENT)
    }
    if (palette %in% VIRIDIS_OPTIONS) {
      return(scales::viridis_pal(option = palette)(256L))
    }
    if (identical(palette, "Greens")) {
      return(GREENS)
    }
    if (identical(palette, "Greens_r")) {
      return(rev(GREENS))
    }
    if (palette %in% grDevices::palette.pals()) {
      return(unname(grDevices::palette.colors(palette = palette)))
    }
  }
  if (is.character(palette) && length(palette) >= 1L && !anyNA(palette) && all(is_colour(palette))) {
    return(unname(palette))
  }
  stop(sprintf(
    "%s must be a palette name such as 'Accent' or 'viridis', or a vector of colors.",
    argument
  ), call. = FALSE)
}

# n colors sampled as matplotlib samples a listed colormap at
# numpy.linspace(0, 1, n): truncation to an index, with 1 mapped to the last
# color. An 8-color palette gives its first and last colors for n = 2 and
# repeats colors for n > 8.
palette_colors <- function(palette, n) {
  values <- colormap_values(palette)
  m <- length(values)
  index <- pmin(floor(seq(0, 1, length.out = n) * m), m - 1) + 1
  values[index]
}

continuous_colours <- function(cmap) {
  values <- colormap_values(cmap, argument = "cmap")
  if (length(values) < 2L) {
    stop("cmap must name a colormap or give at least two colors.", call. = FALSE)
  }
  values
}

# matplotlib maps values in [0, 1] through a 256-entry table; the heatmap
# does the same, so its colors match the color bar ggplot2 draws from the
# same colors.
colour_lut <- function(colours, n = 256L) {
  scales::gradient_n_pal(colours)(seq(0, 1, length.out = n))
}

lut_colours <- function(values, lut) {
  n <- length(lut)
  lut[pmin(floor(values * n), n - 1) + 1]
}

# Python's f"{x:g}", which is C's %g.
format_g <- function(x) {
  formatC(x, digits = 6L, format = "g")
}

measure_ylabel <- function(column) {
  gsub("Ari", "ARI", title_case(gsub("_", " ", column, fixed = TRUE)), fixed = TRUE)
}

rule_title <- function(rule) {
  if (identical(rule, "1se")) "1-SE" else title_case(rule)
}

selected_value_label <- function(row, param, rule) {
  sprintf(
    "Selected %s (%s rule): %s",
    if (param == "n_clusters") "k" else param,
    rule_title(rule),
    format_g(as.numeric(row$sweep_value[[1L]]))
  )
}

build_estimator_label <- function(row, tight_layout = FALSE) {
  label <- as.character(row$method_label[[1L]])
  if (tight_layout) gsub(", ", "\n", label, fixed = TRUE) else label
}

# Two lines naming the selected configuration and how it was selected. Off
# the n_clusters axis the detail gives the swept value and the rounded mean
# cluster count. Python writes that count with an approximately-equal sign,
# which R's pdf() device cannot draw, so R writes "k ~ 7".
annotation_text <- function(measure, rule, results, row, selected_k, pinned = FALSE,
                            tight_layout = FALSE) {
  k_text <- if (is.null(selected_k)) "None" else as.character(selected_k)
  param <- sweep_param_name(results)
  detail <- if (param == "n_clusters") {
    paste0("k = ", k_text)
  } else {
    sprintf("%s = %s, k ~ %s", param, format_g(as.numeric(row$sweep_value[[1L]])), k_text)
  }
  if (isTRUE(pinned)) {
    detail <- paste0(detail, ", fixed")
  }
  text <- sprintf(
    "%s (%s)\n%s, %s rule",
    build_estimator_label(row, tight_layout = tight_layout),
    detail,
    title_case(gsub("_", " ", measure, fixed = TRUE)),
    rule_title(rule)
  )
  if (tight_layout) paste0(text, "\n") else text
}

# Plots check the measure before drawing: an unknown one would otherwise
# make the selection fail inside the tryCatch() around the dashed line, and
# the line would be left off without an error.
plot_measure_column <- function(results, measure) {
  if (!is.character(measure) || length(measure) != 1L || !measure %in% names(MEASURE_MAP)) {
    stop(sprintf(
      "Measure %s not found. Valid options: %s",
      format_repr(measure),
      python_list(names(MEASURE_MAP))
    ), call. = FALSE)
  }
  column <- MEASURE_MAP[[measure]]
  if (!column %in% names(results)) {
    stop(sprintf("Metric column '%s' not found in the results table.", column), call. = FALSE)
  }
  column
}

# Breaks on whole numbers, for counts.
integer_breaks <- function(limits) {
  lo <- ceiling(limits[[1L]])
  hi <- floor(limits[[2L]])
  if (!is.finite(lo) || !is.finite(hi) || hi < lo) {
    return(numeric())
  }
  unique(round(pretty(c(lo, hi))))
}

# matplotlib sizes markers by area in square points, so a marker of area s
# is sqrt(s) points across; ggplot2 draws a circle 0.75 * size * .pt points
# across.
point_size <- function(s) {
  sqrt(s) / (0.75 * ggplot2::.pt)
}

# matplotlib line widths are in points, 96 / 72 grid line-width units each.
# ggplot2 draws a point's outline with stroke * .stroke / 2 units and a line
# with linewidth * .pt units.
edge_stroke <- function(lw) {
  lw * 96 / 72 * 2 / ggplot2::.stroke
}

line_width <- function(lw) {
  lw * 96 / 72 / ggplot2::.pt
}

carve_theme <- function() {
  ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
}

legend_position <- function(legend_loc) {
  if (is.numeric(legend_loc) && length(legend_loc) == 2L && !anyNA(legend_loc)) {
    return(ggplot2::theme(legend.position = "inside", legend.position.inside = legend_loc))
  }
  if (!is.character(legend_loc) || length(legend_loc) != 1L ||
      !legend_loc %in% c("right", "left", "top", "bottom")) {
    stop(
      "legend_loc must be 'right', 'left', 'top', 'bottom', or two numbers giving a position inside the panel.",
      call. = FALSE
    )
  }
  ggplot2::theme(legend.position = legend_loc)
}
```

- [ ] Step 5: Run the tests

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "plotting")'
Rscript -e 'devtools::test()'
```

Expected: `document()` adds `importFrom(ggplot2,.data)` to NAMESPACE and `'plotting.R'` to the Collate field. The plotting tests pass; the full suite passes with `SKIP 0` and 1194 plus this task's expectations.

If `legend_position("bottom")$legend.position` does not return the string in ggplot2 4.0.3 (a theme is an S7 object there), read it with `ggplot2::calc_element("legend.position", ggplot2::theme_bw() + legend_position("bottom"))` instead, and say so in the report.

- [ ] Step 6: Commit

```bash
git add carve-r/DESCRIPTION carve-r/NAMESPACE carve-r/R/CARVE-package.R carve-r/R/plotting.R carve-r/tests/testthat/test-plotting.R
git commit -m "$(cat <<'EOF'
feat(carve-r): add the colors, text and sizes the plots share

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 7: Mutations

Apply each, run `Rscript -e 'devtools::test(filter = "plotting")'`, confirm the named test fails, and restore with `git checkout -- R/plotting.R`:

1. In `palette_colors()`, `round()` instead of `floor()`: fails "palette_colors samples a palette".
2. In `palette_colors()`, drop `pmin(..., m - 1)`: fails "palette_colors samples a palette" (index 9 of 8 is `NA`).
3. In `measure_ylabel()`, drop the `gsub("Ari", "ARI", ...)`: fails "axis, selection and annotation text".
4. In `annotation_text()`, drop the `", fixed"` branch: fails "axis, selection and annotation text".
5. In `rule_title()`, return `toupper(rule)`: fails "axis, selection and annotation text".
6. In `lut_colours()`, `round()` instead of `floor()`: fails "values map to colors".

---

### Task 3: The three line plots over the sweep axis

Files:
- Modify: `carve-r/R/plotting.R`
- Create: `carve-r/tests/testthat/helper-plots.R`
- Modify: `carve-r/tests/testthat/test-plotting.R`

Interfaces:
- Consumes: Task 2's helpers; `select_best_row_by_rule()` (`selection.R`); `sweep_param_name()`, `sweep_axis_label()`, `observed_k()` (`sweep.R`).
- Produces (internal, `R/plotting.R`):
  - `draw_metric_lines(results, y_col, group_col, label_of, legend_title, select_row, selection_label, integer_y = FALSE, title = NULL, xlabel = NULL, ylabel, legend = TRUE, legend_loc = "right", palette = "Accent")` returns a ggplot. Its layers, in order: a `geom_vline` (only when the selection succeeds), a `geom_errorbar` (only when the table has `<y_col>_se`), a `geom_line` and a `geom_point`.
  - `metric_over_sweep_plot(results, measure = "stability", rule = "1se", not_two = FALSE, title = NULL, xlabel = NULL, ylabel = NULL, legend = TRUE, legend_loc = "right", palette = "Accent")`.
  - `selected_row_for(estimator_results, method_id, measure, rule, not_two)`.
  - `metric_by_pipeline_plot(preprocessing, estimator_results, method_id, measure = "stability", rule = "1se", not_two = FALSE, title = NULL, xlabel = NULL, ylabel = NULL, legend = TRUE, legend_loc = "right", palette = "Accent")`.
  - `n_clusters_over_sweep_plot(results, measure = "stability", rule = "1se", not_two = FALSE, title = NULL, xlabel = NULL, ylabel = NULL, legend = TRUE, legend_loc = "right", palette = "Accent")`.
  - Test helpers (`helper-plots.R`): the tables `metric_results()`, `two_method_results()`, `resolution_results()`, `pipeline_results()`, `pooled_results()`, `realized_count_results()`; `layer_with(plot, geom)` (the `layer_data()` of the first layer whose geom inherits from `geom`, or `NULL`), `has_layer(plot, geom)`, `guide_labels(plot, aesthetic)`, `x_breaks(plot)`, `y_breaks(plot)`.

- [ ] Step 1: Create `tests/testthat/helper-plots.R`

The tables are the ones in `tests/test_plotting.py` and `tests/conftest.py`, with the columns the R runner writes.

```r
# Results tables of the Python plotting tests (tests/test_plotting.py and
# tests/conftest.py), with the columns the R runner writes, and helpers that
# read a ggplot object.

metric_results <- function() {
  data.frame(
    config_id = 0:2,
    method_id = "m0",
    method_label = "KMeans, linkage=ward",
    estimator = "KMeans",
    n_clusters = 2:4,
    linkage = "ward",
    ari_stability = c(0.9, 0.85, 0.7),
    ari_stability_se = c(0.02, 0.03, 0.05),
    ari_stability_upper = c(0.92, 0.88, 0.75),
    ari_stability_lower = c(0.88, 0.82, 0.65),
    ari_generalizability = c(0.85, 0.80, 0.65),
    ari_generalizability_se = c(0.03, 0.04, 0.06),
    ari_generalizability_upper = c(0.88, 0.84, 0.71),
    ari_generalizability_lower = c(0.82, 0.76, 0.59),
    sweep_param = "n_clusters",
    sweep_value = c(2, 3, 4),
    sweep_rank = 0:2,
    n_clusters_observed = c(2, 3, 4),
    n_clusters_observed_se = 0,
    noise_fraction = 0
  )
}

# Two curves, one with a parameter the other lacks.
two_method_results <- function() {
  data.frame(
    config_id = 0:5,
    method_id = rep(c("m0", "m1"), each = 3),
    method_label = rep(c("KMeans", "SpectralClustering, affinity=self_tuning"), each = 3),
    estimator = rep(c("KMeans", "SpectralClustering"), each = 3),
    n_clusters = c(2:4, 2:4),
    affinity = rep(c(NA, "self_tuning"), each = 3),
    ari_stability = c(0.9, 0.85, 0.7, 0.88, 0.82, 0.68),
    ari_stability_se = c(0.02, 0.03, 0.05, 0.02, 0.03, 0.05),
    sweep_param = "n_clusters",
    sweep_value = c(2, 3, 4, 2, 3, 4),
    sweep_rank = c(0:2, 0:2),
    n_clusters_observed = c(2, 3, 4, 2, 3, 4),
    n_clusters_observed_se = 0,
    noise_fraction = 0
  )
}

resolution_results <- function() {
  data.frame(
    config_id = 0:3,
    method_id = "m0",
    method_label = "LeidenClustering, n_neighbors=15",
    estimator = "LeidenClustering",
    resolution = c(0.25, 0.5, 1, 2),
    n_neighbors = 15L,
    ari_stability = c(0.9, 0.85, 0.7, 0.6),
    ari_stability_se = c(0.02, 0.03, 0.05, 0.06),
    ari_stability_upper = c(0.92, 0.88, 0.75, 0.66),
    ari_stability_lower = c(0.88, 0.82, 0.65, 0.54),
    sweep_param = "resolution",
    sweep_value = c(0.25, 0.5, 1, 2),
    sweep_rank = 0:3,
    n_clusters_observed = c(2, 3, 5, 9),
    n_clusters_observed_se = c(0, 0.1, 0.3, 0.5),
    noise_fraction = 0
  )
}

# A preprocessing_results() table over two configurations. m0 has two
# pipelines whose values peak at k = 3. m1 has a third pipeline, a decoy
# peaking at k = 4 above anything in m0, so a plot or a selection that reads
# the whole table instead of m0's rows is visibly wrong.
pipeline_results <- function() {
  curves <- list(
    list("m0", "identity | identity", c(0.70, 0.90, 0.60)),
    list("m0", "identity | PCA(n_components=2)", c(0.65, 0.80, 0.50)),
    list("m1", "StandardScaler | identity", c(0.20, 0.30, 0.99))
  )
  rows <- lapply(curves, function(curve) {
    steps <- strsplit(curve[[2L]], " | ", fixed = TRUE)[[1L]]
    data.frame(
      method_id = curve[[1L]],
      method_label = paste("KMeans", curve[[1L]]),
      pipeline = curve[[2L]],
      normalization = steps[[1L]],
      dim_reduction = steps[[2L]],
      n_clusters = 2:4,
      n_resamples = 5L,
      ari_stability = curve[[3L]],
      ari_stability_se = 0.01,
      ari_generalizability = curve[[3L]] - 0.1,
      ari_generalizability_se = 0.02,
      n_clusters_observed = c(2, 3, 4),
      sweep_param = "n_clusters",
      sweep_value = c(2, 3, 4),
      sweep_rank = 0:2
    )
  })
  do.call(rbind, rows)
}

# The estimator_results() table behind pipeline_results(), pooled. It
# disagrees with the per-pipeline rows on purpose. Under "1se" the best row,
# m1 at k = 2 with SE 0.10, admits m0 at k = 4, so CARVE selects (m0, 4); the
# rule over m0's rows alone stops at k = 3. For m1 the rule over its own
# pooled rows stops at k = 2, where its per-pipeline rows peak at k = 4.
pooled_results <- function() {
  data.frame(
    config_id = 0:5,
    method_id = rep(c("m0", "m1"), each = 3),
    method_label = rep(c("KMeans m0", "KMeans m1"), each = 3),
    n_clusters = c(2:4, 2:4),
    ari_stability = c(0.80, 0.90, 0.86, 0.95, 0.40, 0.30),
    ari_stability_se = c(0.01, 0.01, 0.01, 0.10, 0.01, 0.01),
    ari_generalizability = c(0.80, 0.90, 0.86, 0.95, 0.40, 0.30) - 0.1,
    ari_generalizability_se = c(0.01, 0.01, 0.01, 0.10, 0.01, 0.01),
    n_clusters_observed = c(2, 3, 4, 2, 3, 4),
    sweep_param = "n_clusters",
    sweep_value = c(2, 3, 4, 2, 3, 4),
    sweep_rank = c(0:2, 0:2)
  )
}

# A resolution table over two methods. The selections land on different
# rows, and two counts differ between truncation and rounding (6.6 and 2.2):
# stability max is Leiden at 1 (6.6 clusters); stability 1se admits Leiden
# at 2 (11 clusters); generalizability max is Louvain at 0.5 (2.2); with
# not_two, Louvain at 0.5 rounds to two clusters and is left out, which
# leaves Leiden at 1.
realized_count_results <- function() {
  data.frame(
    config_id = 0:5,
    method_id = rep(c("m0", "m1"), each = 3),
    method_label = rep(c("LeidenClustering", "LouvainClustering"), each = 3),
    estimator = rep(c("LeidenClustering", "LouvainClustering"), each = 3),
    resolution = c(0.5, 1, 2, 0.5, 1, 2),
    ari_stability = c(0.80, 0.90, 0.88, 0.70, 0.75, 0.60),
    ari_stability_se = c(0.01, 0.03, 0.01, 0.01, 0.01, 0.01),
    ari_generalizability = c(0.60, 0.70, 0.65, 0.95, 0.50, 0.40),
    ari_generalizability_se = 0.01,
    sweep_param = "resolution",
    sweep_value = c(0.5, 1, 2, 0.5, 1, 2),
    sweep_rank = c(0:2, 0:2),
    n_clusters_observed = c(3.2, 6.6, 11.0, 2.2, 5.0, 9.4),
    n_clusters_observed_se = c(0.2, 0.4, 0.0, 0.1, 0.3, 0.6),
    noise_fraction = 0
  )
}

layer_index <- function(plot, geom) {
  which(vapply(plot$layers, function(layer) inherits(layer$geom, geom), logical(1)))
}

has_layer <- function(plot, geom) {
  length(layer_index(plot, geom)) > 0L
}

layer_with <- function(plot, geom) {
  index <- layer_index(plot, geom)
  if (length(index) == 0L) {
    return(NULL)
  }
  ggplot2::layer_data(plot, index[[1L]])
}

guide_labels <- function(plot, aesthetic) {
  as.character(ggplot2::get_guide_data(plot, aesthetic)$.label)
}

panel_breaks <- function(plot, axis) {
  breaks <- ggplot2::ggplot_build(plot)$layout$panel_params[[1L]][[axis]]$breaks
  as.numeric(breaks[!is.na(breaks)])
}

x_breaks <- function(plot) panel_breaks(plot, "x")
y_breaks <- function(plot) panel_breaks(plot, "y")
```

- [ ] Step 2: Write the failing tests

Append to `tests/testthat/test-plotting.R`:

```r
test_that("each method is one line, labeled by its method label", {
  plot <- metric_over_sweep_plot(two_method_results())
  lines <- layer_with(plot, "GeomLine")
  expect_identical(length(unique(lines$group)), 2L)
  expect_identical(guide_labels(plot, "colour"), c("KMeans", "SpectralClustering, affinity=self_tuning"))
  expect_identical(sort(unique(lines$colour)), sort(palette_colors("Accent", 2L)))
  expect_identical(plot$scales$get_scales("colour")$name, "Estimators")
})

test_that("the lines carry the metric and its standard error", {
  plot <- metric_over_sweep_plot(metric_results(), measure = "generalizability")
  lines <- layer_with(plot, "GeomLine")
  expect_equal(lines$x, c(2, 3, 4))
  expect_equal(lines$y, c(0.85, 0.80, 0.65))
  bars <- layer_with(plot, "GeomErrorbar")
  expect_equal(bars$ymin, c(0.85, 0.80, 0.65) - c(0.03, 0.04, 0.06))
  expect_equal(bars$ymax, c(0.85, 0.80, 0.65) + c(0.03, 0.04, 0.06))
})

test_that("a table without standard errors draws no error bars", {
  results <- metric_results()
  results$ari_stability_se <- NULL
  plot <- suppressWarnings(metric_over_sweep_plot(results))
  expect_false(has_layer(plot, "GeomErrorbar"))
  expect_true(has_layer(plot, "GeomLine"))
})

test_that("the dashed line marks the selected sweep value", {
  plot <- metric_over_sweep_plot(metric_results(), rule = "1se")
  dashed <- layer_with(plot, "GeomVline")
  # 0.9 - 0.02 admits only k = 2.
  expect_equal(dashed$xintercept, 2)
  expect_identical(dashed$linetype, "dashed")
  expect_identical(guide_labels(plot, "linetype"), "Selected k (1-SE rule): 2")
  resolution <- metric_over_sweep_plot(resolution_results(), rule = "max")
  expect_identical(guide_labels(resolution, "linetype"), "Selected resolution (Max rule): 0.25")
  expect_identical(resolution$labels$y, "ARI Stability")
})

test_that("the dashed line is left off when the selection fails", {
  plot <- metric_over_sweep_plot(metric_results()[1L, ], not_two = TRUE)
  expect_false(has_layer(plot, "GeomVline"))
  expect_true(has_layer(plot, "GeomLine"))
})

test_that("axis labels follow the sweep and can be replaced", {
  plot <- metric_over_sweep_plot(metric_results())
  expect_identical(plot$labels$x, "Number of Clusters (k)")
  expect_identical(plot$labels$y, "ARI Stability")
  expect_null(plot$labels$title)
  expect_equal(x_breaks(plot), c(2, 3, 4))
  resolution <- metric_over_sweep_plot(resolution_results())
  expect_identical(resolution$labels$x, "Resolution")
  expect_equal(x_breaks(resolution), c(0.25, 0.5, 1, 2))
  custom <- metric_over_sweep_plot(metric_results(), title = "T", xlabel = "X", ylabel = "Y")
  expect_identical(c(custom$labels$title, custom$labels$x, custom$labels$y), c("T", "X", "Y"))
})

test_that("legend = FALSE drops both legends", {
  plot <- metric_over_sweep_plot(metric_results(), legend = FALSE)
  expect_null(ggplot2::get_guide_data(plot, "colour"))
  expect_null(ggplot2::get_guide_data(plot, "linetype"))
  expect_identical(plot$theme$legend.position, "right")
})

test_that("metric_over_sweep_plot reports a bad measure and an empty table", {
  expect_error(metric_over_sweep_plot(metric_results(), measure = "nonexistent"), "Measure 'nonexistent' not found", fixed = TRUE)
  expect_error(metric_over_sweep_plot(metric_results()[0L, ]), "Results DataFrame is empty.", fixed = TRUE)
})

test_that("plot_metric_by_pipeline draws one line per pipeline of the configuration", {
  plot <- metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0", measure = "generalizability")
  expect_identical(guide_labels(plot, "colour"), c("identity | PCA(n_components=2)", "identity | identity"))
  lines <- layer_with(plot, "GeomLine")
  first <- lines[lines$colour == palette_colors("Accent", 2L)[[2L]], ]
  expect_equal(first$y, c(0.60, 0.80, 0.50))
  second <- lines[lines$colour == palette_colors("Accent", 2L)[[1L]], ]
  expect_equal(second$y, c(0.55, 0.70, 0.40))
  expect_equal(first$x, c(2, 3, 4))
  expect_identical(plot$scales$get_scales("colour")$name, "Pipelines")
  expect_identical(plot$labels$x, "Number of Clusters (k)")
  expect_identical(plot$labels$y, "ARI Generalizability")
})

test_that("plot_metric_by_pipeline marks the value CARVE selects from the pooled table", {
  selected <- metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0", rule = "1se")
  expect_equal(layer_with(selected, "GeomVline")$xintercept, 4)
  expect_identical(guide_labels(selected, "linetype"), "Selected k (1-SE rule): 4")
  other <- metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m1", rule = "1se")
  expect_equal(layer_with(other, "GeomVline")$xintercept, 2)
})

test_that("plot_metric_by_pipeline follows a resolution axis and the palette", {
  as_resolution <- function(table) {
    names(table)[names(table) == "n_clusters"] <- "resolution"
    table$sweep_param <- "resolution"
    table$sweep_value <- table$sweep_value / 4
    table
  }
  plot <- metric_by_pipeline_plot(as_resolution(pipeline_results()), as_resolution(pooled_results()), "m0")
  expect_identical(plot$labels$x, "Resolution")
  expect_equal(x_breaks(plot), c(0.5, 0.75, 1))
  expect_equal(layer_with(plot, "GeomVline")$xintercept, 1)
  viridis <- metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0", palette = "viridis")
  expect_identical(sort(unique(layer_with(viridis, "GeomLine")$colour)), sort(palette_colors("viridis", 2L)))
})

test_that("plot_metric_by_pipeline names what is missing", {
  expect_error(
    metric_by_pipeline_plot(NULL, pooled_results(), "m0"),
    "There is no preprocessing table to plot: the fit was not randomized. Fit with randomize_preprocessing=TRUE.",
    fixed = TRUE
  )
  expect_error(metric_by_pipeline_plot(pipeline_results()[0L, ], pooled_results(), "m0"), "Preprocessing DataFrame is empty.", fixed = TRUE)
  expect_error(
    metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m9"),
    "method_id 'm9' not found in the preprocessing table. Available: ['m0', 'm1'].",
    fixed = TRUE
  )
  expect_error(
    metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0", measure = "pac"),
    "The per-pipeline table carries only the ARI criteria, so measure must be 'stability' or 'generalizability' (or an alias of either); got 'pac'.",
    fixed = TRUE
  )
  expect_error(
    metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0", measure = "average"),
    "got 'average'.",
    fixed = TRUE
  )
})

test_that("plot_n_clusters_over_sweep draws the observed counts and their standard errors", {
  plot <- n_clusters_over_sweep_plot(realized_count_results())
  expect_identical(guide_labels(plot, "colour"), c("LeidenClustering", "LouvainClustering"))
  lines <- layer_with(plot, "GeomLine")
  bars <- layer_with(plot, "GeomErrorbar")
  leiden <- palette_colors("Accent", 2L)[[1L]]
  expect_equal(lines$x[lines$colour == leiden], c(0.5, 1, 2))
  expect_equal(lines$y[lines$colour == leiden], c(3.2, 6.6, 11.0))
  expect_equal(lines$y[lines$colour != leiden], c(2.2, 5.0, 9.4))
  expect_equal((bars$ymax - bars$ymin)[bars$colour == leiden] / 2, c(0.2, 0.4, 0.0))
  expect_equal((bars$ymax - bars$ymin)[bars$colour != leiden] / 2, c(0.1, 0.3, 0.6))
})

test_that("plot_n_clusters_over_sweep marks the selected value and its count", {
  cases <- list(
    list("stability", "max", FALSE, 1, "Selected resolution (Max rule): 1, 7 clusters"),
    list("stability", "1se", FALSE, 2, "Selected resolution (1-SE rule): 2, 11 clusters"),
    list("generalizability", "max", FALSE, 0.5, "Selected resolution (Max rule): 0.5, 2 clusters"),
    list("generalizability", "max", TRUE, 1, "Selected resolution (Max rule): 1, 7 clusters")
  )
  for (case in cases) {
    plot <- n_clusters_over_sweep_plot(realized_count_results(), measure = case[[1L]], rule = case[[2L]], not_two = case[[3L]])
    expect_equal(layer_with(plot, "GeomVline")$xintercept, case[[4L]])
    expect_identical(guide_labels(plot, "linetype"), case[[5L]])
  }
})

test_that("plot_n_clusters_over_sweep labels its axes and ticks whole numbers", {
  plot <- n_clusters_over_sweep_plot(realized_count_results())
  expect_identical(plot$labels$x, "Resolution")
  expect_identical(plot$labels$y, "Mean Observed Number of Clusters")
  expect_identical(n_clusters_over_sweep_plot(realized_count_results(), ylabel = "Clusters")$labels$y, "Clusters")
  # Between 2 and 3 clusters the default breaks are 2, 2.25, 2.5, ...
  narrow <- resolution_results()[1:3, ]
  narrow$n_clusters_observed <- c(2, 2.5, 3)
  narrow$n_clusters_observed_se <- 0
  breaks <- y_breaks(n_clusters_over_sweep_plot(narrow))
  expect_true(length(breaks) >= 2L)
  expect_identical(breaks, round(breaks))
})

test_that("plot_n_clusters_over_sweep rejects a k sweep, a bad measure and an empty table", {
  expect_error(n_clusters_over_sweep_plot(metric_results()), "other than n_clusters", fixed = TRUE)
  expect_error(n_clusters_over_sweep_plot(realized_count_results(), measure = "nonexistent"), "not found", fixed = TRUE)
  expect_error(n_clusters_over_sweep_plot(data.frame()), "Results DataFrame is empty.", fixed = TRUE)
})

test_that("the three line plots draw through one helper", {
  calls <- list()
  local_mocked_bindings(draw_metric_lines = function(results, ...) {
    args <- list(...)
    calls[[length(calls) + 1L]] <<- list(args$group_col, args$legend_title, args$y_col, nrow(results))
    "drawn"
  })
  expect_identical(metric_over_sweep_plot(metric_results()), "drawn")
  expect_identical(metric_by_pipeline_plot(pipeline_results(), pooled_results(), "m0"), "drawn")
  expect_identical(n_clusters_over_sweep_plot(realized_count_results()), "drawn")
  expect_identical(calls, list(
    list("method_id", "Estimators", "ari_stability", 3L),
    list("pipeline", "Pipelines", "ari_stability", 6L),
    list("method_id", "Estimators", "n_clusters_observed", 6L)
  ))
})
```

Two notes on these tests. The pipeline legend sorts in byte order, as pandas' `groupby` does, so `"identity | PCA(n_components=2)"` (with `P`) comes before `"identity | identity"`; the first color goes to the PCA pipeline. testthat runs tests with the C collation, where the default sort is byte order too, so no test can tell the `"radix"` sort from the default; it matters in a user's locale, where the default sort would put `identity` first. The no-standard-error test uses `suppressWarnings()` because `"1se"` falls back to `"max"` with a warning when the `_se` column is missing.

- [ ] Step 3: Run the tests to see them fail

```bash
Rscript -e 'devtools::test(filter = "plotting")'
```

Expected: the Task 2 tests pass; the new ones fail with `could not find function "metric_over_sweep_plot"` and the like.

- [ ] Step 4: Add the line plots to `R/plotting.R`

```r
# The drawing behind the three line plots over the sweep axis, so they
# cannot drift apart. results is non-empty and has sweep_value, group_col
# and y_col; error bars come from <y_col>_se when the table has it.
# select_row() returns the row whose sweep_value the dashed line marks, and
# selection_label() builds its legend text from that row. If either fails,
# for example because not_two leaves no rows, the line is left off, as in
# Python.
draw_metric_lines <- function(results, y_col, group_col, label_of, legend_title, select_row,
                              selection_label, integer_y = FALSE, title = NULL, xlabel = NULL,
                              ylabel, legend = TRUE, legend_loc = "right", palette = "Accent") {
  keys <- as.character(results[[group_col]])
  # pandas' groupby sorts the group keys in byte order, which radix sorting
  # reproduces whatever the locale.
  groups <- sort(unique(keys), method = "radix")
  colours <- stats::setNames(palette_colors(palette, length(groups)), groups)
  labels <- stats::setNames(
    vapply(match(groups, keys), function(i) label_of(results[i, , drop = FALSE]), character(1)),
    groups
  )
  se_col <- paste0(y_col, "_se")
  data <- data.frame(
    group = factor(keys, levels = groups),
    x = as.numeric(results$sweep_value),
    y = as.numeric(results[[y_col]]),
    se = if (se_col %in% names(results)) as.numeric(results[[se_col]]) else NA_real_
  )
  data <- data[order(data$group, data$x), , drop = FALSE]
  x_values <- sort(unique(data$x))
  selected <- tryCatch(
    {
      row <- select_row()
      list(x = as.numeric(row$sweep_value[[1L]]), label = selection_label(row))
    },
    error = function(e) NULL
  )

  plot <- ggplot2::ggplot(
    data,
    ggplot2::aes(x = .data$x, y = .data$y, colour = .data$group, group = .data$group)
  )
  if (!is.null(selected)) {
    plot <- plot +
      ggplot2::geom_vline(
        data = data.frame(x = selected$x, label = selected$label),
        ggplot2::aes(xintercept = .data$x, linetype = .data$label),
        colour = "gray50", linewidth = line_width(2), alpha = 0.6
      ) +
      ggplot2::scale_linetype_manual(values = "dashed", name = NULL)
  }
  if (any(!is.na(data$se))) {
    span <- if (length(x_values) > 1L) diff(range(x_values)) else 1
    plot <- plot + ggplot2::geom_errorbar(
      ggplot2::aes(ymin = .data$y - .data$se, ymax = .data$y + .data$se),
      width = 0.02 * span, linewidth = line_width(1.5), alpha = 0.8, na.rm = TRUE
    )
  }
  plot <- plot +
    ggplot2::geom_line(linewidth = line_width(2), alpha = 0.8, na.rm = TRUE) +
    ggplot2::geom_point(size = point_size(36), alpha = 0.8, na.rm = TRUE) +
    ggplot2::scale_colour_manual(values = colours, breaks = groups, labels = labels, name = legend_title)
  if (length(x_values) <= 20L) {
    plot <- plot + ggplot2::scale_x_continuous(
      breaks = x_values,
      labels = vapply(x_values, format_g, character(1))
    )
  }
  if (integer_y) {
    plot <- plot + ggplot2::scale_y_continuous(breaks = integer_breaks)
  }
  plot <- plot +
    ggplot2::labs(
      x = if (is.null(xlabel)) sweep_axis_label(results) else xlabel,
      y = ylabel,
      title = title
    ) +
    carve_theme() +
    legend_position(legend_loc)
  if (!isTRUE(legend)) {
    plot <- plot + ggplot2::guides(colour = "none", linetype = "none")
  }
  plot
}

metric_over_sweep_plot <- function(results, measure = "stability", rule = "1se", not_two = FALSE,
                                   title = NULL, xlabel = NULL, ylabel = NULL, legend = TRUE,
                                   legend_loc = "right", palette = "Accent") {
  if (nrow(results) == 0L) {
    stop("Results DataFrame is empty.", call. = FALSE)
  }
  column <- plot_measure_column(results, measure)
  param <- sweep_param_name(results)
  draw_metric_lines(
    results,
    y_col = column,
    group_col = "method_id",
    label_of = build_estimator_label,
    legend_title = "Estimators",
    select_row = function() select_best_row_by_rule(results, measure, rule, not_two = not_two),
    selection_label = function(row) selected_value_label(row, param, rule),
    title = title,
    xlabel = xlabel,
    ylabel = if (is.null(ylabel)) measure_ylabel(column) else ylabel,
    legend = legend,
    legend_loc = legend_loc,
    palette = palette
  )
}

# The estimator_results() row whose sweep value marks method_id's
# selection: CARVE's selection over the pooled table when it lands on
# method_id, otherwise the rule's choice among method_id's rows. Restricting
# to method_id first would not do for the selected configuration: under
# "1se" and "quantile" the tolerance comes from the best row of the whole
# table, which can belong to another configuration.
selected_row_for <- function(estimator_results, method_id, measure, rule, not_two) {
  row <- select_best_row_by_rule(estimator_results, measure, rule, not_two = not_two)
  if (identical(as.character(row$method_id[[1L]]), as.character(method_id))) {
    return(row)
  }
  own <- estimator_results[as.character(estimator_results$method_id) == as.character(method_id), , drop = FALSE]
  select_best_row_by_rule(own, measure, rule, not_two = not_two)
}

metric_by_pipeline_plot <- function(preprocessing, estimator_results, method_id,
                                    measure = "stability", rule = "1se", not_two = FALSE,
                                    title = NULL, xlabel = NULL, ylabel = NULL, legend = TRUE,
                                    legend_loc = "right", palette = "Accent") {
  if (is.null(preprocessing)) {
    stop(
      "There is no preprocessing table to plot: the fit was not randomized. Fit with randomize_preprocessing=TRUE.",
      call. = FALSE
    )
  }
  if (nrow(preprocessing) == 0L) {
    stop("Preprocessing DataFrame is empty.", call. = FALSE)
  }
  ids <- as.character(preprocessing$method_id)
  rows <- preprocessing[ids == as.character(method_id), , drop = FALSE]
  if (nrow(rows) == 0L) {
    stop(sprintf(
      "method_id %s not found in the preprocessing table. Available: %s.",
      format_repr(as.character(method_id)),
      python_list(sort(unique(ids), method = "radix"))
    ), call. = FALSE)
  }
  carried <- is.character(measure) && length(measure) == 1L && measure %in% names(MEASURE_MAP) &&
    MEASURE_MAP[[measure]] %in% names(rows)
  if (!carried) {
    stop(sprintf(
      "The per-pipeline table carries only the ARI criteria, so measure must be 'stability' or 'generalizability' (or an alias of either); got %s.",
      format_repr(measure)
    ), call. = FALSE)
  }
  column <- MEASURE_MAP[[measure]]
  param <- sweep_param_name(rows)
  draw_metric_lines(
    rows,
    y_col = column,
    group_col = "pipeline",
    label_of = function(row) as.character(row$pipeline[[1L]]),
    legend_title = "Pipelines",
    select_row = function() selected_row_for(estimator_results, method_id, measure, rule, not_two),
    selection_label = function(row) selected_value_label(row, param, rule),
    title = title,
    xlabel = xlabel,
    ylabel = if (is.null(ylabel)) measure_ylabel(column) else ylabel,
    legend = legend,
    legend_loc = legend_loc,
    palette = palette
  )
}

n_clusters_over_sweep_plot <- function(results, measure = "stability", rule = "1se",
                                       not_two = FALSE, title = NULL, xlabel = NULL, ylabel = NULL,
                                       legend = TRUE, legend_loc = "right", palette = "Accent") {
  if (nrow(results) == 0L) {
    stop("Results DataFrame is empty.", call. = FALSE)
  }
  param <- sweep_param_name(results)
  if (param == "n_clusters") {
    stop(
      "plot_n_clusters_over_sweep needs a sweep over a parameter other than n_clusters. This table sweeps n_clusters, where the realized count is the swept value itself; use plot_metric_over_n_clusters.",
      call. = FALSE
    )
  }
  plot_measure_column(results, measure)
  draw_metric_lines(
    results,
    y_col = "n_clusters_observed",
    group_col = "method_id",
    label_of = build_estimator_label,
    legend_title = "Estimators",
    select_row = function() select_best_row_by_rule(results, measure, rule, not_two = not_two),
    selection_label = function(row) {
      sprintf("%s, %d clusters", selected_value_label(row, param, rule), observed_k(row))
    },
    integer_y = TRUE,
    title = title,
    xlabel = xlabel,
    ylabel = if (is.null(ylabel)) "Mean Observed Number of Clusters" else ylabel,
    legend = legend,
    legend_loc = legend_loc,
    palette = palette
  )
}
```

- [ ] Step 5: Run the tests

```bash
Rscript -e 'devtools::test(filter = "plotting")'
Rscript -e 'devtools::test()'
```

Expected: all pass; the suite passes with `SKIP 0`.

- [ ] Step 6: Commit

```bash
git add carve-r/R/plotting.R carve-r/tests/testthat/helper-plots.R carve-r/tests/testthat/test-plotting.R
git commit -m "$(cat <<'EOF'
feat(carve-r): draw metrics and cluster counts over the sweep axis

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 7: Mutations

Apply each, run `Rscript -e 'devtools::test(filter = "plotting")'`, confirm the named test fails, and restore with `git checkout -- R/plotting.R`:

1. In `metric_by_pipeline_plot()`, pass `select_row = function() select_best_row_by_rule(estimator_results, measure, rule, not_two = not_two)`: fails "marks the value CARVE selects from the pooled table" (m1's line moves from 2 to 4).
2. In `selected_row_for()`, always select among `method_id`'s own rows: fails "marks the value CARVE selects from the pooled table" (4 becomes 3).
3. In `n_clusters_over_sweep_plot()`, `integer_y = FALSE`: fails "labels its axes and ticks whole numbers".
4. In `n_clusters_over_sweep_plot()`, `floor()` the count in the label instead of `observed_k()`: fails "marks the selected value and its count" (6.6 gives 6).
5. In `metric_by_pipeline_plot()`, draw `preprocessing` instead of `rows`: fails "draws one line per pipeline" (three legend entries).
6. In `draw_metric_lines()`, take the error bars from `y_col` instead of `<y_col>_se`: fails "carry the metric and its standard error".
7. In `draw_metric_lines()`, drop the `guides(colour = "none", linetype = "none")` line: fails "legend = FALSE drops both legends".

---

### Task 4: The consensus heatmap

Files:
- Modify: `carve-r/R/plotting.R`
- Modify: `carve-r/tests/testthat/helper-plots.R`
- Modify: `carve-r/tests/testthat/test-plotting.R`

Interfaces:
- Consumes: Task 2's `continuous_colours()`, `colour_lut()`, `lut_colours()`, `palette_colors()`.
- Produces (internal, `R/plotting.R`):
  - `consensus_display(consensus_matrix, labels)` returns `list(matrix, labels)`: the matrix symmetrized, with `NaN` as 0.5, clipped to [0, 1], diagonal 1, rows and columns ordered by label (ties keep their order); and the labels in that order.
  - `consensus_plot(consensus_matrix, labels, cmap = "viridis", palette = "Accent", colorbar = TRUE, colorbar_label = "Consensus", title = NULL)` returns a ggplot whose first two `annotation_raster()` layers are the heatmap and the cluster band.
  - Test helper `raster_layers(plot)` returns the `annotation_raster()` layers in order.

- [ ] Step 1: Add the raster helper to `tests/testthat/helper-plots.R`

```r
raster_layers <- function(plot) {
  Filter(function(layer) inherits(layer$geom, "GeomRasterAnn"), plot$layers)
}

raster_colours <- function(layer) {
  as.matrix(layer$geom_params$raster)
}
```

- [ ] Step 2: Write the failing tests

Append to `tests/testthat/test-plotting.R`:

```r
test_that("consensus_display symmetrizes, fills unsampled pairs and orders by cluster", {
  M <- rbind(c(1, NaN, 0.2), c(NaN, 1, 0.6), c(0.4, 0.8, 1))
  display <- consensus_display(M, c(2, 1, 2))
  expect_equal(display$matrix, rbind(c(1, 0.5, 0.7), c(0.5, 1, 0.3), c(0.7, 0.3, 1)))
  expect_identical(display$labels, c(1, 2, 2))
  # Python's test_nan_handling: never co-sampled pairs show 0.5, the value
  # get_labels() also uses, and the diagonal is 1.
  expect_equal(consensus_display(rbind(c(1, NaN), c(NaN, 1)), c(1, 2))$matrix, rbind(c(1, 0.5), c(0.5, 1)))
  expect_equal(consensus_display(rbind(c(0.3, -0.2), c(-0.2, 1.4)), c(1, 1))$matrix, rbind(c(1, 0), c(0, 1)))
})

test_that("consensus_display checks its input", {
  expect_error(consensus_display(matrix(0, 3, 4), c(1, 1, 1)), "consensus_matrix must be a square 2D array.", fixed = TRUE)
  expect_error(consensus_display(diag(3), c(1, 2)), "consensus_matrix and labels must have matching first dimension.", fixed = TRUE)
  expect_error(consensus_display(diag(3), matrix(1:3, 1)), "labels must be a 1D array.", fixed = TRUE)
})

test_that("the heatmap shows the ordered matrix in the cmap's colors under a band of cluster colors", {
  M <- matrix(0, 6, 6)
  M[1:3, 1:3] <- 1
  M[4:6, 4:6] <- 1
  M[1, 4] <- 0.3
  labels <- c(9, 9, 9, 4, 4, 4)
  plot <- consensus_plot(M, labels)
  rasters <- raster_layers(plot)
  expect_length(rasters, 2L)
  display <- consensus_display(M, labels)
  expect_identical(
    raster_colours(rasters[[1L]]),
    matrix(lut_colours(display$matrix, colour_lut(continuous_colours("viridis"))), 6L, 6L)
  )
  expect_identical(as.vector(raster_colours(rasters[[2L]])), rep(palette_colors("Accent", 2L), each = 3L))
  greens <- consensus_plot(M, labels, cmap = "Greens")
  expect_identical(
    raster_colours(raster_layers(greens)[[1L]]),
    matrix(lut_colours(display$matrix, colour_lut(GREENS)), 6L, 6L)
  )
})

test_that("white lines separate the clusters", {
  # Row 1 is drawn at the top, at y = n, so the line between rows 1 and 2
  # is at y = n - 0.5.
  plot <- consensus_plot(diag(4), c(1, 2, 2, 2))
  expect_equal(layer_with(plot, "GeomVline")$xintercept, 1.5)
  expect_equal(layer_with(plot, "GeomHline")$yintercept, 3.5)
  expect_false(has_layer(consensus_plot(diag(3), c(1, 1, 1)), "GeomVline"))
})

test_that("the color bar, labels and frame of the heatmap", {
  plot <- consensus_plot(diag(4), c(1, 1, 2, 2), title = "T")
  expect_identical(plot$scales$get_scales("fill")$name, "Consensus")
  expect_false(is.null(ggplot2::get_guide_data(plot, "fill")))
  expect_identical(plot$labels$x, "Samples (ordered by cluster)")
  expect_identical(plot$labels$title, "T")
  ranges <- ggplot2::ggplot_build(plot)$layout$panel_params[[1L]]
  expect_equal(ranges$x.range, c(0.5, 4.5))
  expect_equal(ranges$y.range, c(0.5, 4.5 + 0.04 * 4))
  expect_identical(consensus_plot(diag(4), c(1, 1, 2, 2), colorbar_label = "Share")$scales$get_scales("fill")$name, "Share")
  expect_null(ggplot2::get_guide_data(consensus_plot(diag(4), c(1, 1, 2, 2), colorbar = FALSE), "fill"))
})
```

- [ ] Step 3: Run the tests to see them fail

```bash
Rscript -e 'devtools::test(filter = "plotting")'
```

Expected: the new tests fail with `could not find function "consensus_display"`.

- [ ] Step 4: Add the heatmap to `R/plotting.R`

```r
# The matrix as drawn: symmetric, pairs never drawn together at 0.5 (the
# value get_labels() uses), clipped to [0, 1], diagonal 1, and ordered by
# cluster so that each cluster is a block.
consensus_display <- function(consensus_matrix, labels) {
  M <- consensus_matrix
  if (!is.matrix(M) || nrow(M) != ncol(M)) {
    stop("consensus_matrix must be a square 2D array.", call. = FALSE)
  }
  if (!is.null(dim(labels)) && length(dim(labels)) > 1L) {
    stop("labels must be a 1D array.", call. = FALSE)
  }
  labels <- as.vector(labels)
  if (nrow(M) != length(labels)) {
    stop("consensus_matrix and labels must have matching first dimension.", call. = FALSE)
  }
  storage.mode(M) <- "double"
  M <- 0.5 * (M + t(M))
  M[is.na(M)] <- 0.5
  M <- pmin(pmax(M, 0), 1)
  diag(M) <- 1
  ord <- order(labels)
  list(matrix = unname(M[ord, ord, drop = FALSE]), labels = labels[ord])
}

# The matrix and the band of cluster colors are rasters in one panel, the
# band above the matrix. A raster is far cheaper than one tile per pair: at
# 5,000 anchors, about a second and 1 GB against 20 seconds and 5 GB. The
# invisible tile layer carries the fill scale, so the color bar shows.
consensus_plot <- function(consensus_matrix, labels, cmap = "viridis", palette = "Accent",
                           colorbar = TRUE, colorbar_label = "Consensus", title = NULL) {
  display <- consensus_display(consensus_matrix, labels)
  n <- length(display$labels)
  colours <- continuous_colours(cmap)
  cells <- matrix(lut_colours(display$matrix, colour_lut(colours)), n, n)
  clusters <- sort(unique(display$labels))
  band <- matrix(palette_colors(palette, length(clusters))[match(display$labels, clusters)], nrow = 1L)
  band_height <- 0.04 * n
  # Row 1 of the matrix is drawn at the top, at y = n, so the line between
  # rows b and b + 1 is at y = n + 0.5 - b.
  boundaries <- which(diff(display$labels) != 0) + 0.5
  key <- data.frame(x = 1, y = 1, value = c(0, 1))

  plot <- ggplot2::ggplot(key, ggplot2::aes(x = .data$x, y = .data$y, fill = .data$value)) +
    ggplot2::geom_tile(width = 0, height = 0, alpha = 0) +
    ggplot2::annotation_raster(
      cells, xmin = 0.5, xmax = n + 0.5, ymin = 0.5, ymax = n + 0.5, interpolate = FALSE
    ) +
    ggplot2::annotation_raster(
      band, xmin = 0.5, xmax = n + 0.5, ymin = n + 0.5, ymax = n + 0.5 + band_height,
      interpolate = FALSE
    )
  if (length(boundaries) > 0L) {
    plot <- plot +
      ggplot2::geom_vline(xintercept = boundaries, colour = "white", linewidth = line_width(0.6), alpha = 0.8) +
      ggplot2::geom_hline(yintercept = n + 1 - boundaries, colour = "white", linewidth = line_width(0.6), alpha = 0.8)
  }
  plot <- plot +
    ggplot2::scale_fill_gradientn(colours = colours, limits = c(0, 1), name = colorbar_label) +
    ggplot2::scale_x_continuous(breaks = NULL) +
    ggplot2::scale_y_continuous(breaks = NULL) +
    ggplot2::coord_fixed(xlim = c(0.5, n + 0.5), ylim = c(0.5, n + 0.5 + band_height), expand = FALSE) +
    ggplot2::labs(x = "Samples (ordered by cluster)", y = NULL, title = title) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(panel.grid = ggplot2::element_blank(), panel.border = ggplot2::element_blank())
  if (!isTRUE(colorbar)) {
    plot <- plot + ggplot2::guides(fill = "none")
  }
  plot
}
```

- [ ] Step 5: Run the tests

```bash
Rscript -e 'devtools::test(filter = "plotting")'
Rscript -e 'devtools::test()'
```

Expected: all pass, `SKIP 0`. Then look at one plot once, to check what the tests cannot (the band sits on top of the matrix, the color bar is on the right, the lines are white):

```bash
Rscript -e 'devtools::load_all(quiet = TRUE); M <- matrix(0.1, 40, 40); M[1:15, 1:15] <- 0.9; M[16:40, 16:40] <- 0.8; ggplot2::ggsave("/tmp/carve-consensus.png", consensus_plot(M, rep(c(2, 1), c(15, 25)), title = "check"), width = 5, height = 5)'
```

Open `/tmp/carve-consensus.png` (the Read tool shows images) and describe it in the report. Delete it afterwards.

- [ ] Step 6: Commit

```bash
git add carve-r/R/plotting.R carve-r/tests/testthat/helper-plots.R carve-r/tests/testthat/test-plotting.R
git commit -m "$(cat <<'EOF'
feat(carve-r): draw the consensus matrix as a heatmap with a cluster band

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 7: Mutations

Apply each, run `Rscript -e 'devtools::test(filter = "plotting")'`, confirm the named test fails, and restore with `git checkout -- R/plotting.R`:

1. In `consensus_display()`, `M[is.na(M)] <- 0`: fails "symmetrizes, fills unsampled pairs".
2. In `consensus_display()`, drop the symmetrizing line: fails "symmetrizes, fills unsampled pairs".
3. In `consensus_display()`, `ord <- seq_along(labels)`: fails "symmetrizes, fills unsampled pairs" and "shows the ordered matrix".
4. In `consensus_plot()`, index the band colors by the label value instead of `match(display$labels, clusters)`: fails "shows the ordered matrix" (label 9 has no color).
5. In `consensus_plot()`, `yintercept = boundaries`: fails "white lines separate the clusters".
6. In `consensus_plot()`, ignore `cmap` (always `continuous_colours("viridis")`): fails "shows the ordered matrix" (the Greens case).

---

### Task 5: Box plots and violin plots of per-sample scores

Files:
- Modify: `carve-r/R/plotting.R`
- Modify: `carve-r/tests/testthat/test-plotting.R`

Interfaces:
- Consumes: Task 2's helpers; the `plotting` fixture's `score_groups`.
- Produces (internal, `R/plotting.R`):
  - `cluster_score_groups(scores, labels, order = NULL)` returns `list(groups, order)`: the finite scores of each cluster, in `order` (default: the sorted labels), and the labels kept, which are the ones with at least one finite score. R shows label values as they are, so `order` is not shifted by one as Python's is.
  - `score_frame(prepared)` returns a data frame with `position` (1, 2, ...) and `score`.
  - `score_ylim(groups, ylim, fit_ylim)` returns the y limits or `NULL`.
  - `cluster_boxplot_plot(scores, labels, order = NULL, palette = "Accent", showfliers = FALSE, width = 0.75, title = NULL, xlabel = "Cluster", ylabel = "Uncertainty", annotation = NULL, rotation = NULL, ylim = c(-0.02, 1.02), fit_ylim = TRUE)`.
  - `cluster_violin_plot(scores, labels, order = NULL, palette = "Accent", density_norm = "width", stripplot = TRUE, jitter = TRUE, size = 8, alpha = 0.22, inner = "box", title = NULL, xlabel = "Cluster", ylabel = "Uncertainty", annotation = NULL, rotation = NULL, ylim = c(-0.02, 1.02), fit_ylim = TRUE)`.
  - Both put the clusters at x = 1, 2, ... on a continuous axis labeled with the label values, and `annotation` (a string or `NULL`) in the caption.

- [ ] Step 1: Write the failing tests

Append to `tests/testthat/test-plotting.R`:

```r
test_that("cluster_score_groups matches Python's grouping", {
  f <- read_fixture("plotting")$score_groups
  scores <- fixture_vector(f$scores)
  # R labels count from 1 and are shown as they are; Python's count from 0
  # and are shown plus one.
  labels <- fixture_vector(f$labels) + 1
  for (case in f$cases) {
    order <- if (is.null(case$order)) NULL else fixture_vector(case$order) + 1
    prepared <- cluster_score_groups(scores, labels, order)
    expect_identical(prepared$order, fixture_vector(case$kept_order))
    expect_identical(length(prepared$groups), length(case$groups))
    for (i in seq_along(case$groups)) {
      expect_equal(prepared$groups[[i]], fixture_vector(case$groups[[i]]))
    }
  }
})

test_that("cluster_score_groups drops non-finite scores and checks its input", {
  prepared <- cluster_score_groups(c(0.9, NaN, 0.7, 0.6), c(1, 1, 2, 2))
  expect_identical(lengths(prepared$groups), c(1L, 2L))
  expect_identical(cluster_score_groups(c(0.9, 0.8, 0.7, 0.6), c(1, 1, 2, 2), order = c(2, 1))$order, c(2, 1))
  expect_error(cluster_score_groups(c(NaN, NaN), c(1, 2)), "No finite scores available for plotting.", fixed = TRUE)
  expect_error(cluster_score_groups(0.5, c(1, 2)), "scores and labels must have matching length.", fixed = TRUE)
  expect_error(cluster_score_groups(matrix(0.5), 1), "scores must be a 1D array.", fixed = TRUE)
  expect_error(cluster_score_groups(0.5, matrix(1)), "labels must be a 1D array.", fixed = TRUE)
  expect_error(cluster_score_groups(c(0.5, 0.6), c(1, 2), order = 7), "No cluster values found for the provided order.", fixed = TRUE)
})

test_that("score_ylim fits the scores with a margin or keeps ylim", {
  # A range of 0.1 gives a margin of 0.005, below the 0.02 floor.
  expect_equal(score_ylim(list(c(0.5, 0.6), 0.55), c(-0.02, 1.02), TRUE), c(0.48, 0.62))
  expect_equal(score_ylim(list(c(0, 1)), NULL, TRUE), c(-0.05, 1.05))
  expect_identical(score_ylim(list(0.5), c(0, 1), FALSE), c(0, 1))
  expect_null(score_ylim(list(0.5), NULL, FALSE))
})

test_that("the box plot draws one box per cluster in the requested order", {
  scores <- c(0.9, 0.85, 0.7, 0.65, 0.5, 0.45)
  labels <- c(1, 1, 1, 2, 2, 2)
  plot <- cluster_boxplot_plot(scores, labels, order = c(2, 1), annotation = "note", title = "T")
  boxes <- layer_with(plot, "GeomBoxplot")
  expect_identical(nrow(boxes), 2L)
  expect_equal(boxes$middle, c(0.5, 0.85))
  expect_identical(boxes$fill, palette_colors("Accent", 2L))
  expect_identical(plot$scales$get_scales("x")$labels, c("2", "1"))
  expect_identical(plot$labels$caption, "note")
  expect_identical(plot$labels$title, "T")
  expect_identical(plot$labels$x, "Cluster")
  expect_identical(plot$labels$y, "Uncertainty")
  expect_equal(ggplot2::ggplot_build(plot)$layout$panel_params[[1L]]$y.range, c(0.45 - 0.0225, 0.9 + 0.0225))
  expect_null(cluster_boxplot_plot(scores, labels)$labels$caption)
})

test_that("showfliers and rotation reach the box plot", {
  scores <- c(0.5, 0.52, 0.51, 0.53, 0.5, 0.95, 0.1, 0.2)
  labels <- c(1, 1, 1, 1, 1, 1, 2, 2)
  hidden <- cluster_boxplot_plot(scores, labels)
  shown <- cluster_boxplot_plot(scores, labels, showfliers = TRUE)
  expect_true(is.na(hidden$layers[[layer_index(hidden, "GeomBoxplot")]]$geom_params$outlier.shape))
  expect_identical(shown$layers[[layer_index(shown, "GeomBoxplot")]]$geom_params$outlier.shape, 19)
  rotated <- cluster_boxplot_plot(scores, labels, rotation = 45)
  expect_identical(rotated$theme$axis.text.x$angle, 45)
})

test_that("the violin plot draws bodies, inner marks and seeded points", {
  scores <- c(0.9, 0.85, 0.7, 0.65, 0.5, 0.45)
  labels <- c(1, 1, 1, 2, 2, 2)
  plot <- cluster_violin_plot(scores, labels)
  bodies <- layer_with(plot, "GeomViolin")
  expect_identical(sort(unique(bodies$fill)), sort(palette_colors("Accent", 2L)))
  marks <- layer_with(plot, "GeomSegment")
  # "box": a median line, a quartile line and two quartile caps per cluster.
  expect_identical(nrow(marks), 8L)
  expect_equal(sort(marks$y[marks$x == marks$xend]), sort(c(0.775, 0.475)))
  points <- layer_with(plot, "GeomPoint")
  expect_equal(points$y, scores)
  expect_true(all(abs(points$x - rep(1:2, each = 3)) <= 0.11))
  expect_identical(points$x, layer_with(cluster_violin_plot(scores, labels), "GeomPoint")$x)
  quartile <- layer_with(cluster_violin_plot(scores, labels, inner = "quartile"), "GeomSegment")
  expect_identical(nrow(quartile), 6L)
  expect_false(has_layer(cluster_violin_plot(scores, labels, inner = "none"), "GeomSegment"))
  expect_false(has_layer(cluster_violin_plot(scores, labels, stripplot = FALSE), "GeomPoint"))
  still <- layer_with(cluster_violin_plot(scores, labels, jitter = FALSE), "GeomPoint")
  expect_equal(still$x, rep(c(1, 2), each = 3))
})

test_that("the violin plot scales the bodies and limits the axis", {
  scores <- c(0.95, 0.96, 0.97, 0.98, 0.99, 0.60, 0.61, 0.62)
  labels <- c(1, 1, 1, 1, 1, 2, 2, 2)
  plot <- cluster_violin_plot(scores, labels, density_norm = "count", ylim = c(0, 0.9), fit_ylim = FALSE)
  violin <- plot$layers[[layer_index(plot, "GeomViolin")]]
  expect_identical(violin$stat_params$scale, "count")
  expect_equal(ggplot2::ggplot_build(plot)$layout$panel_params[[1L]]$y.range, c(0, 0.9))
  expect_warning(
    unknown <- cluster_violin_plot(scores, labels, density_norm = "nope"),
    "Unknown density_norm='nope'; using 'width'.",
    fixed = TRUE
  )
  expect_identical(unknown$layers[[layer_index(unknown, "GeomViolin")]]$stat_params$scale, "width")
  expect_error(cluster_violin_plot(scores, labels, inner = "nope"), "inner must be one of: 'box', 'quartile', 'none'.", fixed = TRUE)
})

test_that("a cluster with one score gets points but no violin body", {
  plot <- cluster_violin_plot(c(0.9, 0.8, 0.7, 0.4), c(1, 1, 1, 2))
  bodies <- layer_with(plot, "GeomViolin")
  expect_identical(unique(bodies$fill), palette_colors("Accent", 2L)[[1L]])
  expect_identical(nrow(layer_with(plot, "GeomPoint")), 4L)
})
```

The box test's y range: the scores run from 0.45 to 0.9, so the margin is `max(0.02, 0.05 * 0.45) = 0.0225`. The violin's quartile line runs from the first to the third quartile of each cluster (numpy's default percentiles are R's type 7), so the segment ends are 0.775 and 0.875 for cluster 1 and 0.475 and 0.575 for cluster 2; the test checks the lower ends.

- [ ] Step 2: Run the tests to see them fail

```bash
Rscript -e 'devtools::test(filter = "plotting")'
```

Expected: the new tests fail with `could not find function "cluster_score_groups"`.

- [ ] Step 3: Add the score plots to `R/plotting.R`

```r
# The finite scores of each cluster in plotting order, and the clusters
# kept: those with at least one finite score.
cluster_score_groups <- function(scores, labels, order = NULL) {
  if (!is.null(dim(scores)) && length(dim(scores)) > 1L) {
    stop("scores must be a 1D array.", call. = FALSE)
  }
  if (!is.null(dim(labels)) && length(dim(labels)) > 1L) {
    stop("labels must be a 1D array.", call. = FALSE)
  }
  scores <- as.numeric(scores)
  labels <- as.vector(labels)
  if (length(scores) != length(labels)) {
    stop("scores and labels must have matching length.", call. = FALSE)
  }
  finite <- is.finite(scores)
  scores <- scores[finite]
  labels <- labels[finite]
  if (length(scores) == 0L) {
    stop("No finite scores available for plotting.", call. = FALSE)
  }
  wanted <- if (is.null(order)) sort(unique(labels)) else order
  groups <- lapply(wanted, function(label) scores[labels == label])
  kept <- lengths(groups) > 0L
  if (!any(kept)) {
    stop("No cluster values found for the provided order.", call. = FALSE)
  }
  list(groups = groups[kept], order = wanted[kept])
}

score_frame <- function(prepared) {
  data.frame(
    position = rep(seq_along(prepared$groups), lengths(prepared$groups)),
    score = unlist(prepared$groups, use.names = FALSE)
  )
}

# With fit_ylim the axis runs over the scores with a margin of 5% of their
# range, at least 0.02; otherwise it is ylim, which may be NULL.
score_ylim <- function(groups, ylim, fit_ylim) {
  if (!isTRUE(fit_ylim)) {
    return(ylim)
  }
  values <- unlist(groups, use.names = FALSE)
  margin <- max(0.02, 0.05 * (max(values) - min(values)))
  c(min(values) - margin, max(values) + margin)
}

# The axis, limits, labels and theme the box and violin plots share. The
# clusters sit at x = 1, 2, ... and the axis shows their labels.
score_plot_frame <- function(plot, prepared, ylim, fit_ylim, title, xlabel, ylabel, annotation,
                             rotation) {
  plot <- plot +
    ggplot2::scale_x_continuous(
      breaks = seq_along(prepared$order),
      labels = as.character(prepared$order),
      minor_breaks = NULL
    ) +
    ggplot2::labs(x = xlabel, y = ylabel, title = title, caption = annotation) +
    carve_theme() +
    ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())
  limits <- score_ylim(prepared$groups, ylim, fit_ylim)
  if (!is.null(limits)) {
    plot <- plot +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(0)) +
      ggplot2::coord_cartesian(ylim = limits)
  }
  if (!is.null(rotation)) {
    plot <- plot + ggplot2::theme(axis.text.x = ggplot2::element_text(angle = rotation))
  }
  plot
}

cluster_fills <- function(prepared, palette) {
  stats::setNames(palette_colors(palette, length(prepared$groups)), seq_along(prepared$groups))
}

cluster_boxplot_plot <- function(scores, labels, order = NULL, palette = "Accent",
                                 showfliers = FALSE, width = 0.75, title = NULL,
                                 xlabel = "Cluster", ylabel = "Uncertainty", annotation = NULL,
                                 rotation = NULL, ylim = c(-0.02, 1.02), fit_ylim = TRUE) {
  prepared <- cluster_score_groups(scores, labels, order)
  data <- score_frame(prepared)
  plot <- ggplot2::ggplot(data, ggplot2::aes(x = .data$position, y = .data$score, group = .data$position)) +
    ggplot2::geom_boxplot(
      ggplot2::aes(fill = factor(.data$position)),
      width = width, alpha = 0.8, linewidth = line_width(1.2),
      outlier.shape = if (isTRUE(showfliers)) 19 else NA
    ) +
    ggplot2::scale_fill_manual(values = cluster_fills(prepared, palette), guide = "none")
  score_plot_frame(plot, prepared, ylim, fit_ylim, title, xlabel, ylabel, annotation, rotation)
}

cluster_violin_plot <- function(scores, labels, order = NULL, palette = "Accent",
                                density_norm = "width", stripplot = TRUE, jitter = TRUE,
                                size = 8, alpha = 0.22, inner = "box", title = NULL,
                                xlabel = "Cluster", ylabel = "Uncertainty", annotation = NULL,
                                rotation = NULL, ylim = c(-0.02, 1.02), fit_ylim = TRUE) {
  if (!is.character(inner) || length(inner) != 1L || !inner %in% c("box", "quartile", "none")) {
    stop("inner must be one of: 'box', 'quartile', 'none'.", call. = FALSE)
  }
  if (!is.character(density_norm) || length(density_norm) != 1L ||
      !density_norm %in% c("width", "area", "count")) {
    warning(sprintf("Unknown density_norm=%s; using 'width'.", format_repr(density_norm)), call. = FALSE)
    density_norm <- "width"
  }
  prepared <- cluster_score_groups(scores, labels, order)
  data <- score_frame(prepared)
  # ggplot2 drops a group with fewer than two values from the violins, with
  # a warning. Such a cluster keeps its points and inner marks.
  dense <- data[data$position %in% which(lengths(prepared$groups) >= 2L), , drop = FALSE]
  plot <- ggplot2::ggplot(data, ggplot2::aes(x = .data$position, y = .data$score)) +
    ggplot2::geom_violin(
      data = dense,
      ggplot2::aes(group = .data$position, fill = factor(.data$position)),
      scale = density_norm, width = 0.8, colour = "black", linewidth = line_width(0.8),
      alpha = 0.8
    ) +
    ggplot2::scale_fill_manual(values = cluster_fills(prepared, palette), guide = "none")

  if (inner != "none") {
    quartiles <- t(vapply(
      prepared$groups, stats::quantile, numeric(3),
      probs = c(0.25, 0.5, 0.75), names = FALSE
    ))
    at <- seq_along(prepared$groups)
    segment <- function(half, from, to, width) {
      data.frame(x = at - half, xend = at + half, y = from, yend = to, width = line_width(width))
    }
    marks <- if (inner == "box") {
      rbind(
        segment(0.13, quartiles[, 2], quartiles[, 2], 1.5),
        data.frame(x = at, xend = at, y = quartiles[, 1], yend = quartiles[, 3], width = line_width(1.2)),
        segment(0.08, quartiles[, 1], quartiles[, 1], 1.0),
        segment(0.08, quartiles[, 3], quartiles[, 3], 1.0)
      )
    } else {
      rbind(
        segment(0.13, quartiles[, 1], quartiles[, 1], 1.0),
        segment(0.13, quartiles[, 2], quartiles[, 2], 1.5),
        segment(0.13, quartiles[, 3], quartiles[, 3], 1.0)
      )
    }
    plot <- plot +
      ggplot2::geom_segment(
        data = marks,
        ggplot2::aes(x = .data$x, xend = .data$xend, y = .data$y, yend = .data$yend, linewidth = .data$width),
        colour = "black", inherit.aes = FALSE
      ) +
      ggplot2::scale_linewidth_identity()
  }

  if (isTRUE(stripplot)) {
    jitter_width <- if (isTRUE(jitter)) 0.11 else if (isFALSE(jitter)) 0 else as.numeric(jitter)
    # A fixed seed makes the plot the same each time and leaves the
    # session's random number stream alone; Python's jitter is unseeded.
    plot <- plot + ggplot2::geom_point(
      position = ggplot2::position_jitter(width = jitter_width, height = 0, seed = 0L),
      shape = 16, size = point_size(size), alpha = alpha, colour = "black", stroke = 0
    )
  }
  score_plot_frame(plot, prepared, ylim, fit_ylim, title, xlabel, ylabel, annotation, rotation)
}
```

- [ ] Step 4: Run the tests

```bash
Rscript -e 'devtools::test(filter = "plotting")'
Rscript -e 'devtools::test()'
```

Expected: all pass, `SKIP 0`. Three tests read layer and theme internals: the violin's `scale` at `layer$stat_params$scale`, the box plot's `outlier.shape` at `layer$geom_params$outlier.shape`, and the label angle at `plot$theme$axis.text.x$angle`. If ggplot2 4.0.3 keeps one of them elsewhere, find it with `str(layer, max.level = 1)` (or `@angle` on the S7 element) and change the test lines that read it; the assertion's intent (the argument reaches the plot) stays. Say so in the report.

Then look at one violin plot and one box plot, as in Task 4 Step 5:

```bash
Rscript -e 'devtools::load_all(quiet = TRUE); set.seed(1); s <- c(runif(40, 0.8, 1), runif(30, 0.4, 0.9), 0.5); l <- c(rep(1, 40), rep(2, 30), 3); ggplot2::ggsave("/tmp/carve-violin.png", cluster_violin_plot(s, l, annotation = "KMeans (k = 3)\nStability, 1-SE rule"), width = 5, height = 4); ggplot2::ggsave("/tmp/carve-box.png", cluster_boxplot_plot(s, l), width = 5, height = 4)'
```

Describe both in the report (violins filled per cluster with black outlines, inner box marks, gray-black points, the caption below, cluster 3 with a point and no body), then delete the files.

- [ ] Step 5: Commit

```bash
git add carve-r/R/plotting.R carve-r/tests/testthat/test-plotting.R
git commit -m "$(cat <<'EOF'
feat(carve-r): draw per-sample scores by cluster as box and violin plots

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 6: Mutations

Apply each, run `Rscript -e 'devtools::test(filter = "plotting")'`, confirm the named test fails, and restore with `git checkout -- R/plotting.R`:

1. In `cluster_score_groups()`, keep non-finite scores: fails "matches Python's grouping" and "drops non-finite scores".
2. In `cluster_score_groups()`, ignore `order`: fails "matches Python's grouping" and "one box per cluster in the requested order".
3. In `score_ylim()`, `margin <- 0.05 * (max(values) - min(values))`: fails "fits the scores with a margin" (the 0.02 floor).
4. In `cluster_violin_plot()`, drop the `dense` filter (draw every group): fails "a cluster with one score" (ggplot2's drop warning becomes an error under `warn = 2`).
5. In `cluster_violin_plot()`, `seed = NA` in `position_jitter()`: fails "draws bodies, inner marks and seeded points" (two plots, different jitter).
6. In `cluster_violin_plot()`, pass `scale = "width"` always: fails "scales the bodies and limits the axis".
7. In `score_plot_frame()`, drop `caption = annotation`: fails "one box per cluster in the requested order".

---

### Task 6: The two scatter plots

Files:
- Modify: `carve-r/R/plotting.R`
- Modify: `carve-r/tests/testthat/test-plotting.R`

Interfaces:
- Consumes: Task 2's helpers; `PCA()` (`transforms.R`); `as_data_matrix()` (`utils.R`); `isclose()` (`sweep.R`); the `plotting` fixture's `scatter` and `diagnostic`.
- Produces (internal, `R/plotting.R`):
  - `check_scatter_inputs(X, labels, scores)` returns `list(X, labels, scores)` after Python's checks.
  - `scatter_coordinates(X, embedding = NULL)` returns an unnamed n by 2 matrix: `embedding`'s first two columns, or `X` itself with two columns, `(X, 0)` with one, and `PCA(X, n_components = 2L, random_state = 0L)` with more.
  - `scatter_encoding(scores, size_range, alpha_range)` returns `list(size, alpha)`, matplotlib's marker areas and opacities; `NA` for non-finite scores.
  - `diagnostic_encoding(scores, alpha_encoding, alpha_range)` returns `list(norm, alpha, limits)`.
  - `diagnostic_draw_order(labels, scores, norm, sort_order)` returns the sample indices in drawing order.
  - `cluster_scatter_plot(X, labels, scores, embedding = NULL, palette = "Accent", alpha_range = c(0.45, 0.9), size_range = c(15, 60), sort_order = TRUE, legend = TRUE, legend_loc = "right", annotation = NULL, annotation_style = "legend", title = NULL, scores_name = "Score", xlabel = "Component 1", ylabel = "Component 2", show_ticks = FALSE, frameon = FALSE)`.
  - `diagnostic_scatter_plot(X, labels, scores, embedding = NULL, cmap = "Greens_r", alpha_encoding = TRUE, alpha_range = c(0.3, 1), marker_size = 30, marker_linewidth = 0.2, markers = NULL, sort_order = TRUE, legend = TRUE, legend_loc = "right", colorbar = TRUE, colorbar_label = NULL, annotation = NULL, annotation_style = "legend", title = NULL, scores_name = "Score", xlabel = "Component 1", ylabel = "Component 2", show_ticks = FALSE, frameon = FALSE)`.
  - In both, the point layer's data rows are in drawing order.

- [ ] Step 1: Write the failing tests

Append to `tests/testthat/test-plotting.R`:

```r
scatter_case <- function(n = 30L, p = 4L, k = 3L) {
  withr::with_seed(0L, list(
    X = matrix(stats::rnorm(n * p), n, p),
    labels = rep(seq_len(k), each = n %/% k),
    scores = stats::runif(n)
  ))
}

test_that("scatter sizes, opacities and legend match Python's", {
  f <- read_fixture("plotting")$scatter
  X <- fixture_matrix(f$X)
  labels <- fixture_vector(f$labels) + 1
  scores <- fixture_vector(f$scores)
  encoding <- scatter_encoding(scores, c(15, 60), c(0.45, 0.9))
  expect_equal(encoding$size, fixture_vector(f$sizes))
  expect_equal(encoding$alpha, fixture_vector(f$alphas))
  flat <- scatter_encoding(rep(0.8, 12L), c(15, 60), c(0.45, 0.9))
  expect_equal(flat$size, fixture_vector(f$flat_sizes))
  expect_equal(flat$alpha, fixture_vector(f$flat_alphas))
  plot <- cluster_scatter_plot(X, labels, scores, sort_order = FALSE)
  expect_identical(guide_labels(plot, "fill"), unlist(f$legend))
  points <- layer_with(plot, "GeomPoint")
  expect_equal(points$size, point_size(fixture_vector(f$sizes)))
  expect_equal(points$alpha, fixture_vector(f$alphas))
  expect_equal(cbind(points$x, points$y), unname(X))
  expect_identical(points$fill, palette_colors("Accent", 3L)[labels])
})

test_that("scatter coordinates come from the embedding, the data or its principal components", {
  case <- scatter_case(n = 20L, p = 10L, k = 2L)
  expect_equal(scatter_coordinates(case$X[, 1:2]), unname(case$X[, 1:2]))
  expect_equal(scatter_coordinates(case$X[, 1L, drop = FALSE]), cbind(case$X[, 1L], 0))
  reduced <- scatter_coordinates(case$X)
  expect_equal(reduced, PCA(case$X, n_components = 2L, random_state = 0L))
  # Two of ten components go through irlba, hence test-transforms.R's tolerance.
  expect_equal(abs(reduced), abs(unname(stats::prcomp(case$X)$x[, 1:2])), tolerance = 1e-4)
  embedding <- matrix(seq_len(60), 20, 3)
  expect_equal(scatter_coordinates(case$X, embedding), embedding[, 1:2] + 0)
  expect_error(scatter_coordinates(case$X, embedding[, 1L, drop = FALSE]), "embedding must be a 2D array with at least 2 columns.", fixed = TRUE)
  expect_error(scatter_coordinates(case$X, embedding[1:5, ]), "embedding must have the same number of rows as X.", fixed = TRUE)
})

test_that("the scatter plot orders, bounds and drops points by score", {
  case <- scatter_case()
  sorted <- layer_with(cluster_scatter_plot(case$X, case$labels, case$scores, alpha_range = c(0.3, 0.9)), "GeomPoint")
  expect_equal(min(sorted$alpha), 0.3)
  expect_equal(max(sorted$alpha), 0.9)
  expect_false(is.unsorted(sorted$alpha))
  unsorted <- layer_with(cluster_scatter_plot(case$X, case$labels, case$scores, sort_order = FALSE), "GeomPoint")
  expect_true(is.unsorted(unsorted$alpha))
  scores <- case$scores
  scores[4] <- NaN
  expect_identical(nrow(layer_with(cluster_scatter_plot(case$X, case$labels, scores), "GeomPoint")), 29L)
})

test_that("the scatter annotation goes into the legend title or the caption", {
  case <- scatter_case()
  in_legend <- cluster_scatter_plot(case$X, case$labels, case$scores, annotation = "note", scores_name = "Gini Stability")
  expect_identical(in_legend$scales$get_scales("fill")$name, "note\nCluster, Gini Stability")
  expect_null(in_legend$labels$caption)
  boxed <- cluster_scatter_plot(case$X, case$labels, case$scores, annotation = "note", annotation_style = "box")
  expect_identical(boxed$labels$caption, "note")
  expect_identical(boxed$scales$get_scales("fill")$name, "Cluster, Score")
  expect_error(
    cluster_scatter_plot(case$X, case$labels, case$scores, annotation_style = "nope"),
    "annotation_style must be 'legend' or 'box'.",
    fixed = TRUE
  )
})

test_that("the scatter legend, ticks and frame can be turned off and on", {
  case <- scatter_case()
  plot <- cluster_scatter_plot(case$X, case$labels, case$scores)
  expect_true(inherits(plot$theme$axis.ticks, "element_blank"))
  expect_true(inherits(plot$theme$panel.border, "element_blank"))
  expect_identical(plot$labels$x, "Component 1")
  framed <- cluster_scatter_plot(case$X, case$labels, case$scores, show_ticks = TRUE, frameon = TRUE)
  expect_false(inherits(framed$theme$axis.ticks, "element_blank"))
  expect_false(inherits(framed$theme$panel.border, "element_blank"))
  expect_null(ggplot2::get_guide_data(cluster_scatter_plot(case$X, case$labels, case$scores, legend = FALSE), "fill"))
})

test_that("the scatter plots check their input as Python does", {
  case <- scatter_case(n = 20L, p = 2L, k = 2L)
  expect_error(cluster_scatter_plot(case$X, case$labels, case$scores[1:10]), "X, labels, and scores must have matching n_samples.", fixed = TRUE)
  expect_error(cluster_scatter_plot(case$X, case$labels, rep(NaN, 20L)), "No finite scores available for plotting.", fixed = TRUE)
  expect_error(diagnostic_scatter_plot(case$X, case$labels, case$scores[-1L]), "matching n_samples", fixed = TRUE)
  expect_error(diagnostic_scatter_plot(case$X, case$labels, rep(NaN, 20L)), "No finite scores", fixed = TRUE)
})

test_that("diagnostic drawing order and opacities match Python's", {
  f <- read_fixture("plotting")
  X <- fixture_matrix(f$scatter$X)
  labels <- fixture_vector(f$scatter$labels) + 1
  scores <- fixture_vector(f$diagnostic$scores)
  encoding <- diagnostic_encoding(scores, TRUE, c(0.3, 1))
  for (name in c("unsorted", "sorted")) {
    rows <- diagnostic_draw_order(labels, scores, encoding$norm, sort_order = name == "sorted")
    expect_identical(rows, as.integer(fixture_vector(f$diagnostic[[paste0("order_", name)]])) + 1L, info = name)
    expect_equal(encoding$alpha[rows], fixture_vector(f$diagnostic[[paste0("alpha_", name)]]), info = name)
  }
  points <- layer_with(diagnostic_scatter_plot(X, labels, scores), "GeomPoint")
  rows <- diagnostic_draw_order(labels, scores, encoding$norm, sort_order = TRUE)
  expect_equal(cbind(points$x, points$y), unname(X[rows, ]))
  expect_equal(points$alpha, encoding$alpha[rows])
})

test_that("diagnostic fills run from dark for low scores, with NaN in faint gray", {
  case <- scatter_case()
  scores <- case$scores
  scores[5] <- NaN
  points <- layer_with(diagnostic_scatter_plot(case$X, case$labels, scores, sort_order = FALSE), "GeomPoint")
  original <- diagnostic_draw_order(case$labels, scores, rep(0.5, 30L), sort_order = FALSE)
  at <- function(i) which(original == i)
  expect_identical(points$fill[at(which.min(scores))], "#00441B")
  expect_identical(points$fill[at(which.max(scores))], "#F7FCF5")
  expect_identical(points$fill[at(5L)], "#808080")
  expect_equal(points$alpha[at(5L)], 0.2)
  bounded <- layer_with(diagnostic_scatter_plot(case$X, case$labels, case$scores, alpha_range = c(0.4, 0.9)), "GeomPoint")
  expect_equal(range(bounded$alpha), c(0.4, 0.9))
  plain <- layer_with(diagnostic_scatter_plot(case$X, case$labels, case$scores, alpha_encoding = FALSE), "GeomPoint")
  expect_true(all(plain$alpha == 1))
})

test_that("each cluster gets a marker shape, in label order", {
  case <- scatter_case()
  plot <- diagnostic_scatter_plot(case$X, case$labels, case$scores, markers = c(21, 22, 24))
  expect_identical(ggplot2::get_guide_data(plot, "shape")$shape, c(21, 22, 24))
  points <- layer_with(plot, "GeomPoint")
  expect_identical(sort(unique(points$shape)), c(21, 22, 24))
  expect_warning(
    cycled <- diagnostic_scatter_plot(case$X, case$labels, case$scores, markers = c(21, 22)),
    "Number of clusters (3) exceeds available markers (2). Markers will cycle.",
    fixed = TRUE
  )
  expect_identical(ggplot2::get_guide_data(cycled, "shape")$shape, c(21, 22, 21))
  expect_error(
    diagnostic_scatter_plot(case$X, case$labels, case$scores, markers = c(1, 2)),
    "markers must be ggplot2 shapes 21 to 25, the shapes with a fill.",
    fixed = TRUE
  )
})

test_that("the diagnostic color bar and legend titles", {
  case <- scatter_case()
  named <- diagnostic_scatter_plot(case$X, case$labels, case$scores, scores_name = "Foo")
  expect_identical(named$scales$get_scales("fill")$name, "Foo")
  expect_false(is.null(ggplot2::get_guide_data(named, "fill")))
  relabeled <- diagnostic_scatter_plot(case$X, case$labels, case$scores, scores_name = "Foo", colorbar_label = "Bar")
  expect_identical(relabeled$scales$get_scales("fill")$name, "Bar")
  expect_null(ggplot2::get_guide_data(diagnostic_scatter_plot(case$X, case$labels, case$scores, colorbar = FALSE), "fill"))
  in_legend <- diagnostic_scatter_plot(case$X, case$labels, case$scores, annotation = "note")
  expect_identical(in_legend$scales$get_scales("shape")$name, "note\nCluster")
  boxed <- diagnostic_scatter_plot(case$X, case$labels, case$scores, annotation = "note", annotation_style = "box")
  expect_identical(boxed$scales$get_scales("shape")$name, "Cluster")
  expect_identical(boxed$labels$caption, "note")
  expect_null(ggplot2::get_guide_data(diagnostic_scatter_plot(case$X, case$labels, case$scores, legend = FALSE), "shape"))
})
```

In the fill test, `sort_order = FALSE` keeps the samples in cluster order with the clusters ordered by mean score, which `diagnostic_draw_order()` with any `norm` reproduces, so `at(i)` finds sample `i`'s row.

- [ ] Step 2: Run the tests to see them fail

```bash
Rscript -e 'devtools::test(filter = "plotting")'
```

Expected: the new tests fail with `could not find function "scatter_encoding"` and the like.

- [ ] Step 3: Add the scatter plots to `R/plotting.R`

```r
check_scatter_inputs <- function(X, labels, scores) {
  if (!is.null(dim(labels)) && length(dim(labels)) > 1L) {
    stop("labels must be a 1D array.", call. = FALSE)
  }
  if (!is.null(dim(scores)) && length(dim(scores)) > 1L) {
    stop("scores must be a 1D array.", call. = FALSE)
  }
  X <- as_data_matrix(X)
  labels <- as.vector(labels)
  scores <- as.numeric(scores)
  if (nrow(X) != length(labels) || nrow(X) != length(scores)) {
    stop("X, labels, and scores must have matching n_samples.", call. = FALSE)
  }
  if (!any(is.finite(scores))) {
    stop("No finite scores available for plotting.", call. = FALSE)
  }
  list(X = X, labels = labels, scores = scores)
}

check_annotation_style <- function(annotation_style) {
  if (!identical(annotation_style, "legend") && !identical(annotation_style, "box")) {
    stop("annotation_style must be 'legend' or 'box'.", call. = FALSE)
  }
}

# Two columns to draw: the embedding's first two, or the data's, or the
# first two principal components of data with more than two columns.
scatter_coordinates <- function(X, embedding = NULL) {
  if (is.null(embedding)) {
    if (ncol(X) > 2L) {
      return(PCA(X, n_components = 2L, random_state = 0L))
    }
    if (ncol(X) == 2L) {
      return(unname(X))
    }
    if (ncol(X) == 1L) {
      return(unname(cbind(X[, 1L], 0)))
    }
    stop("X must have at least 1 feature for scatter plotting.", call. = FALSE)
  }
  coords <- as.matrix(embedding)
  if (length(dim(coords)) != 2L || ncol(coords) < 2L) {
    stop("embedding must be a 2D array with at least 2 columns.", call. = FALSE)
  }
  if (nrow(coords) != nrow(X)) {
    stop("embedding must have the same number of rows as X.", call. = FALSE)
  }
  coords <- unname(coords[, 1:2, drop = FALSE])
  storage.mode(coords) <- "double"
  coords
}

# Marker area and opacity from the score. size_range and alpha_range give
# the value for the highest score first, so stable samples are small and
# faint, and unstable ones large and opaque. Samples without a finite score
# get NA and are not drawn.
scatter_encoding <- function(scores, size_range, alpha_range) {
  finite <- is.finite(scores)
  lo <- min(scores[finite])
  hi <- max(scores[finite])
  if (isclose(lo, hi)) {
    size <- rep(mean(size_range), length(scores))
    alpha <- rep(mean(alpha_range), length(scores))
  } else {
    norm <- (scores - lo) / (hi - lo)
    size <- size_range[[2L]] + norm * (size_range[[1L]] - size_range[[2L]])
    alpha <- alpha_range[[2L]] + norm * (alpha_range[[1L]] - alpha_range[[2L]])
  }
  size[!finite] <- NA_real_
  alpha[!finite] <- NA_real_
  list(size = size, alpha = pmin(pmax(alpha, 0), 1))
}

cluster_means <- function(scores, labels, clusters, empty) {
  vapply(clusters, function(label) {
    values <- scores[labels == label & is.finite(scores)]
    if (length(values) > 0L) mean(values) else empty
  }, numeric(1), USE.NAMES = FALSE)
}

# Labels, theme and legend position the two scatter plots share.
scatter_frame <- function(plot, title, xlabel, ylabel, caption, legend_loc, show_ticks, frameon) {
  plot <- plot +
    ggplot2::labs(x = xlabel, y = ylabel, title = title, caption = caption) +
    carve_theme() +
    ggplot2::theme(panel.grid = ggplot2::element_blank()) +
    legend_position(legend_loc)
  if (!isTRUE(show_ticks)) {
    plot <- plot + ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank())
  }
  if (!isTRUE(frameon)) {
    plot <- plot + ggplot2::theme(panel.border = ggplot2::element_blank())
  }
  plot
}

cluster_scatter_plot <- function(X, labels, scores, embedding = NULL, palette = "Accent",
                                 alpha_range = c(0.45, 0.9), size_range = c(15, 60),
                                 sort_order = TRUE, legend = TRUE, legend_loc = "right",
                                 annotation = NULL, annotation_style = "legend", title = NULL,
                                 scores_name = "Score", xlabel = "Component 1",
                                 ylabel = "Component 2", show_ticks = FALSE, frameon = FALSE) {
  check_annotation_style(annotation_style)
  inputs <- check_scatter_inputs(X, labels, scores)
  coords <- scatter_coordinates(inputs$X, embedding)
  labels <- inputs$labels
  scores <- inputs$scores
  encoding <- scatter_encoding(scores, size_range, alpha_range)
  clusters <- sort(unique(labels))
  keys <- as.character(clusters)
  means <- cluster_means(scores, labels, clusters, NaN)
  data <- data.frame(
    x = coords[, 1L],
    y = coords[, 2L],
    cluster = factor(as.character(labels), levels = keys),
    size = point_size(encoding$size),
    alpha = encoding$alpha
  )
  if (isTRUE(sort_order)) {
    # Faint markers first, so the opaque ones are drawn on top.
    data <- data[order(data$alpha), , drop = FALSE]
  }
  legend_title <- if (nzchar(scores_name)) paste0("Cluster, ", scores_name) else "Cluster"
  if (!is.null(annotation) && annotation_style == "legend") {
    legend_title <- paste0(annotation, "\n", legend_title)
  }
  plot <- ggplot2::ggplot(data, ggplot2::aes(x = .data$x, y = .data$y)) +
    ggplot2::geom_point(
      ggplot2::aes(fill = .data$cluster, size = .data$size, alpha = .data$alpha),
      shape = 21, colour = "black", stroke = edge_stroke(0.2), na.rm = TRUE
    ) +
    ggplot2::scale_fill_manual(
      values = stats::setNames(palette_colors(palette, length(clusters)), keys),
      breaks = keys,
      labels = sprintf("%s (Mean = %.2f)", keys, means),
      name = legend_title
    ) +
    ggplot2::scale_size_identity() +
    ggplot2::scale_alpha_identity() +
    ggplot2::guides(fill = if (isTRUE(legend)) {
      ggplot2::guide_legend(override.aes = list(size = point_size(49), alpha = 0.8, colour = NA))
    } else {
      "none"
    })
  caption <- if (!is.null(annotation) && annotation_style == "box") annotation else NULL
  scatter_frame(plot, title, xlabel, ylabel, caption, legend_loc, show_ticks, frameon)
}

# The score as a position between the lowest and the highest finite score
# (0.5 when they are equal, or for a non-finite score), and the opacity:
# alpha_range gives the value for the highest score first. A sample without
# a finite score is faint.
diagnostic_encoding <- function(scores, alpha_encoding, alpha_range) {
  finite <- is.finite(scores)
  lo <- min(scores[finite])
  hi <- max(scores[finite])
  norm <- rep(0.5, length(scores))
  if (!isclose(lo, hi)) {
    norm[finite] <- (scores[finite] - lo) / (hi - lo)
  }
  alpha <- rep(1, length(scores))
  if (isTRUE(alpha_encoding)) {
    if (isclose(alpha_range[[1L]], alpha_range[[2L]])) {
      alpha[] <- mean(alpha_range)
    } else {
      alpha[finite] <- alpha_range[[2L]] + norm[finite] * (alpha_range[[1L]] - alpha_range[[2L]])
    }
  }
  alpha[!finite] <- 0.2
  list(norm = norm, alpha = alpha, limits = c(lo, hi))
}

# Clusters with the highest mean score first, and within a cluster the
# highest scores first, so unstable samples are drawn on top.
diagnostic_draw_order <- function(labels, scores, norm, sort_order) {
  clusters <- sort(unique(labels))
  means <- cluster_means(scores, labels, clusters, 0)
  rows <- lapply(clusters[order(-means)], function(label) {
    members <- which(labels == label)
    if (isTRUE(sort_order)) members[order(-norm[members])] else members
  })
  unlist(rows, use.names = FALSE)
}

diagnostic_scatter_plot <- function(X, labels, scores, embedding = NULL, cmap = "Greens_r",
                                    alpha_encoding = TRUE, alpha_range = c(0.3, 1),
                                    marker_size = 30, marker_linewidth = 0.2, markers = NULL,
                                    sort_order = TRUE, legend = TRUE, legend_loc = "right",
                                    colorbar = TRUE, colorbar_label = NULL, annotation = NULL,
                                    annotation_style = "legend", title = NULL,
                                    scores_name = "Score", xlabel = "Component 1",
                                    ylabel = "Component 2", show_ticks = FALSE, frameon = FALSE) {
  check_annotation_style(annotation_style)
  inputs <- check_scatter_inputs(X, labels, scores)
  coords <- scatter_coordinates(inputs$X, embedding)
  labels <- inputs$labels
  scores <- inputs$scores
  if (is.null(markers)) {
    markers <- DIAGNOSTIC_MARKERS
  }
  if (!is.numeric(markers) || length(markers) == 0L || !all(markers %in% 21:25)) {
    stop("markers must be ggplot2 shapes 21 to 25, the shapes with a fill.", call. = FALSE)
  }
  clusters <- sort(unique(labels))
  keys <- as.character(clusters)
  if (length(clusters) > length(markers)) {
    warning(sprintf(
      "Number of clusters (%d) exceeds available markers (%d). Markers will cycle.",
      length(clusters), length(markers)
    ), call. = FALSE)
  }
  shapes <- stats::setNames(markers[(seq_along(clusters) - 1L) %% length(markers) + 1L], keys)
  encoding <- diagnostic_encoding(scores, alpha_encoding, alpha_range)
  rows <- diagnostic_draw_order(labels, scores, encoding$norm, sort_order)
  data <- data.frame(
    x = coords[rows, 1L],
    y = coords[rows, 2L],
    score = scores[rows],
    cluster = factor(as.character(labels[rows]), levels = keys),
    alpha = encoding$alpha[rows]
  )
  legend_title <- "Cluster"
  if (!is.null(annotation) && annotation_style == "legend") {
    legend_title <- paste0(annotation, "\n", legend_title)
  }
  plot <- ggplot2::ggplot(data, ggplot2::aes(x = .data$x, y = .data$y)) +
    ggplot2::geom_point(
      ggplot2::aes(fill = .data$score, shape = .data$cluster, alpha = .data$alpha),
      size = point_size(marker_size), colour = "black", stroke = edge_stroke(marker_linewidth)
    ) +
    ggplot2::scale_fill_gradientn(
      colours = continuous_colours(cmap),
      limits = encoding$limits,
      na.value = "#808080",
      name = if (is.null(colorbar_label)) scores_name else colorbar_label
    ) +
    ggplot2::scale_shape_manual(values = shapes, breaks = keys, name = legend_title) +
    ggplot2::scale_alpha_identity() +
    ggplot2::guides(
      fill = if (isTRUE(colorbar)) {
        ggplot2::guide_colourbar(direction = "horizontal", position = "bottom")
      } else {
        "none"
      },
      shape = if (isTRUE(legend)) {
        ggplot2::guide_legend(override.aes = list(fill = "gray50", size = point_size(49), alpha = 1))
      } else {
        "none"
      }
    )
  caption <- if (!is.null(annotation) && annotation_style == "box") annotation else NULL
  scatter_frame(plot, title, xlabel, ylabel, caption, legend_loc, show_ticks, frameon)
}
```

- [ ] Step 4: Run the tests

```bash
Rscript -e 'devtools::test(filter = "plotting")'
Rscript -e 'devtools::test()'
```

Expected: all pass, `SKIP 0`. Then look at both plots once, as in Task 4 Step 5:

```bash
Rscript -e 'devtools::load_all(quiet = TRUE); set.seed(2); X <- rbind(matrix(rnorm(80, 0, 0.5), 40), matrix(rnorm(80, 3, 0.5), 40)); l <- rep(1:2, each = 40); s <- c(runif(40, 0.8, 1), runif(40, 0.3, 1)); s[7] <- NaN; ggplot2::ggsave("/tmp/carve-scatter.png", cluster_scatter_plot(X, l, s, annotation = "KMeans (k = 2)\nStability, 1-SE rule"), width = 6, height = 4.5); ggplot2::ggsave("/tmp/carve-diagnostic.png", diagnostic_scatter_plot(X, l, s, annotation = "KMeans (k = 2)\nStability, 1-SE rule"), width = 6, height = 5)'
```

Describe both in the report (cluster colors with black outlines, large opaque markers for low scores; one shape per cluster, dark green for low scores, a gray point, a horizontal color bar below the panel, the annotation above the legend title), then delete the files.

- [ ] Step 5: Commit

```bash
git add carve-r/R/plotting.R carve-r/tests/testthat/test-plotting.R
git commit -m "$(cat <<'EOF'
feat(carve-r): draw samples in two dimensions with their scores encoded

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 6: Mutations

Apply each, run `Rscript -e 'devtools::test(filter = "plotting")'`, confirm the named test fails, and restore with `git checkout -- R/plotting.R`:

1. In `scatter_encoding()`, swap `size_range[[1L]]` and `size_range[[2L]]`: fails "scatter sizes, opacities and legend match Python's".
2. In `diagnostic_draw_order()`, `order(means)` instead of `order(-means)`: fails "diagnostic drawing order and opacities match Python's".
3. In `diagnostic_draw_order()`, `order(norm[members])`: fails "diagnostic drawing order" (the sorted case).
4. In `diagnostic_encoding()`, drop `alpha[!finite] <- 0.2`: fails "diagnostic drawing order" (the NaN sample's alpha) and "diagnostic fills".
5. In `diagnostic_scatter_plot()`, default `cmap = "Greens"`: fails "diagnostic fills run from dark for low scores".
6. In `diagnostic_scatter_plot()`, drop the cycling warning: fails "each cluster gets a marker shape".
7. In `cluster_scatter_plot()`, ignore `sort_order`: fails "orders, bounds and drops points by score".
8. In `scatter_coordinates()`, return `X[, 1:2]` when there are more than two columns: fails "scatter coordinates come from the embedding".

---

### Task 7: SingleCellExperiment input and storage

Files:
- Create: `carve-r/R/sce.R`
- Modify: `carve-r/R/utils.R` (`check_one_source()`, `cells_by_features()`, `select_dims()`, `embedding_2d()`)
- Modify: `carve-r/R/carve.R` (roxygen of `x` and `...`)
- Modify: `carve-r/DESCRIPTION` (Imports `S4Vectors`, `SingleCellExperiment`, `SummarizedExperiment`)
- Modify: `carve-r/tests/testthat/helper-data.R` (`muffle()`, `cell_names()`, `make_sce()`)
- Create: `carve-r/tests/testthat/test-sce.R`
- Modify: `carve-r/tests/testthat/test-utils.R`

Interfaces:
- Consumes: `as_data_matrix()`, `format_repr()`, `python_list()` (`utils.R`); the `carve()` generic and its `ANY` method.
- Produces:
  - `utils.R`: `check_one_source(assay, reduction)`; `cells_by_features(values)` (cells-as-columns to cells-as-rows, sparse stays sparse); `select_dims(X, n_dims)`; `embedding_2d(values, where)` (first two columns, unnamed, or an error naming `where`).
  - `sce.R`: `sce_matrix(object, assay = NULL, reduction = NULL, n_dims = NULL)`; `sce_column(object, name)` (a column of `colData`, or `NULL`); `sce_column_names(object)`; `sce_set_columns(object, columns)` (a named list of vectors; returns the object); `sce_record(object, key)` and `sce_set_record(object, key, record)` (`metadata(object)[[key]]`); `sce_basis(object, basis = NULL, fallback = NULL)` (an n by 2 matrix or `NULL`); `SCE_BASES` (`c("UMAP", "TSNE", "PCA")`).
  - The exported `carve()` method for `SingleCellExperiment`: `carve(x, assay = NULL, reduction = NULL, n_dims = NULL, ...)`.
  - Test helpers: `muffle(expr, pattern)`; `cell_names(n)`; `make_sce(X, reductions = list(PCA = X), assays = list(logcounts = t(X)))`.

- [ ] Step 1: Add the test helpers to `tests/testthat/helper-data.R`

```r
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
```

- [ ] Step 2: Write the failing tests

Append to `tests/testthat/test-utils.R`:

```r
test_that("cells_by_features turns cells-as-columns into cells-as-rows", {
  values <- matrix(1:6, 2, 3)
  expect_identical(cells_by_features(values), t(values))
  sparse <- Matrix::Matrix(values, sparse = TRUE)
  expect_s4_class(cells_by_features(sparse), "Matrix")
  expect_equal(as.matrix(cells_by_features(sparse)), t(values) + 0)
})

test_that("select_dims keeps the first n_dims columns", {
  X <- matrix(1:6, 2, 3)
  expect_identical(select_dims(X, NULL), X)
  expect_identical(select_dims(X, 2), X[, 1:2])
  expect_error(select_dims(X, 4), "n_dims=4 exceeds the 3 available components in the selected representation.", fixed = TRUE)
  expect_error(select_dims(X, 1.5), "n_dims must be a positive whole number.", fixed = TRUE)
})

test_that("embedding_2d keeps two columns or names what is wrong", {
  values <- matrix(1:6, 2, 3, dimnames = list(c("a", "b"), NULL))
  expect_identical(embedding_2d(values, "here"), unname(values[, 1:2]))
  expect_error(embedding_2d(values[, 1L, drop = FALSE], "here"), "here has shape (2, 1); a basis needs at least two columns.", fixed = TRUE)
})
```

Create `tests/testthat/test-sce.R`:

```r
blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:3))
# The PCA and the logcounts differ, so a test can tell which one was read.
sce <- make_sce(blobs$X, assays = list(logcounts = t(blobs$X * 10)))

test_that("sce_matrix reads the PCA, then the logcounts", {
  expect_equal(unname(sce_matrix(sce)), blobs$X)
  no_pca <- make_sce(blobs$X, reductions = list(), assays = list(logcounts = t(blobs$X * 10)))
  expect_equal(unname(sce_matrix(no_pca)), blobs$X * 10)
  expect_equal(unname(sce_matrix(sce, assay = "logcounts")), blobs$X * 10)
  with_umap <- make_sce(blobs$X, reductions = list(PCA = blobs$X, UMAP = blobs$X * 2))
  expect_equal(unname(sce_matrix(with_umap, reduction = "UMAP")), blobs$X * 2)
  expect_equal(unname(sce_matrix(sce, n_dims = 1)), blobs$X[, 1L, drop = FALSE])
})

test_that("sce_matrix densifies a sparse assay", {
  sparse <- make_sce(blobs$X, reductions = list(), assays = list(logcounts = Matrix::Matrix(t(blobs$X), sparse = TRUE)))
  X <- sce_matrix(sparse)
  expect_true(is.matrix(X))
  expect_equal(unname(X), blobs$X)
})

test_that("sce_matrix warns before clustering many logcounts features", {
  wide <- withr::with_seed(1L, matrix(stats::rnorm(90L * 51L), 90L, 51L))
  no_pca <- make_sce(wide, reductions = list())
  expect_warning(
    X <- sce_matrix(no_pca),
    "No representation given and reducedDim(object, 'PCA') is absent, so CARVE is clustering the 'logcounts' assay directly (51 features).",
    fixed = TRUE
  )
  expect_identical(dim(X), c(90L, 51L))
  expect_no_warning(sce_matrix(no_pca, assay = "logcounts"))
})

test_that("sce_matrix names what it cannot find", {
  expect_error(
    sce_matrix(sce, assay = "logcounts", reduction = "PCA"),
    "Pass at most one of reduction= and assay=; they select the same thing from different places.",
    fixed = TRUE
  )
  expect_error(sce_matrix(sce, assay = "counts"), "assay='counts' not found. Available assays: ['logcounts']", fixed = TRUE)
  expect_error(
    sce_matrix(sce, reduction = "UMAP"),
    "reduction='UMAP' not found in reducedDimNames(object). Available names: ['PCA']. Pass assay=... to cluster an assay.",
    fixed = TRUE
  )
  expect_error(sce_matrix(sce, n_dims = 3), "n_dims=3 exceeds the 2 available components", fixed = TRUE)
  bare <- make_sce(blobs$X, reductions = list(), assays = list(counts = t(blobs$X)))
  expect_error(
    sce_matrix(bare),
    "No representation given, reducedDim(object, 'PCA') is absent and there is no 'logcounts' assay. Pass reduction= or assay=.",
    fixed = TRUE
  )
})

test_that("carve() fits a SingleCellExperiment as it fits the matrix", {
  fit <- carve(sce, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  reference <- carve(blobs$X, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  expect_identical(estimator_results(fit), estimator_results(reference))
  expect_equal(unname(input_data(fit)), blobs$X)
  logcounts <- carve(sce, assay = "logcounts", n_dims = 1, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  expect_equal(unname(input_data(logcounts)), blobs$X[, 1L, drop = FALSE] * 10)
  expect_error(carve(sce, nope = 1, estimator_param_grids = k_grid), "Unknown argument: nope.", fixed = TRUE)
})

test_that("sce columns and records round-trip", {
  labels <- factor(rep(c("1", "2"), 45))
  updated <- sce_set_columns(sce, list(carve = labels, carve_stability = seq_len(90) / 90))
  expect_identical(sce_column(updated, "carve"), labels)
  expect_equal(sce_column(updated, "carve_stability"), seq_len(90) / 90)
  expect_null(sce_column(updated, "nope"))
  expect_true(all(c("carve", "carve_stability") %in% sce_column_names(updated)))
  record <- list(params = list(measure = "stability"))
  stored <- sce_set_record(updated, "carve", record)
  expect_identical(sce_record(stored, "carve"), record)
  expect_null(sce_record(stored, "other"))
})

test_that("sce_basis prefers UMAP, then TSNE, then PCA, then the fallback", {
  X <- blobs$X
  all_three <- make_sce(X, reductions = list(PCA = X, TSNE = X * 3, UMAP = X * 2))
  expect_equal(sce_basis(all_three), X * 2)
  expect_equal(sce_basis(make_sce(X, reductions = list(PCA = X, TSNE = X * 3))), X * 3)
  expect_equal(sce_basis(sce), X)
  custom <- make_sce(X, reductions = list(embedding = X * 4))
  expect_null(sce_basis(custom))
  expect_equal(sce_basis(custom, fallback = "embedding"), X * 4)
  expect_equal(sce_basis(all_three, basis = "TSNE"), X * 3)
  expect_error(
    sce_basis(all_three, basis = "tsne"),
    "basis='tsne' not found in reducedDimNames(object). Available names: ['PCA', 'TSNE', 'UMAP']",
    fixed = TRUE
  )
  one <- make_sce(X, reductions = list(one = X[, 1L, drop = FALSE]))
  expect_error(sce_basis(one, basis = "one"), "reducedDim(object, 'one') has shape (90, 1); a basis needs at least two columns.", fixed = TRUE)
})
```

- [ ] Step 3: Run the tests to see them fail

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript -e 'devtools::test(filter = "sce|utils")'
```

Expected: FAIL, `could not find function "cells_by_features"` and `"sce_matrix"`.

- [ ] Step 4: Add the helpers to `R/utils.R`

Append:

```r
# Single-cell input. SingleCellExperiment and Seurat objects keep cells as
# columns; CARVE clusters a matrix with cells as rows.
check_one_source <- function(assay, reduction) {
  if (!is.null(assay) && !is.null(reduction)) {
    stop(
      "Pass at most one of reduction= and assay=; they select the same thing from different places.",
      call. = FALSE
    )
  }
}

# A sparse matrix stays sparse, so as_data_matrix() can warn before it
# densifies a large one.
cells_by_features <- function(values) {
  if (inherits(values, "Matrix")) Matrix::t(values) else t(as.matrix(values))
}

select_dims <- function(X, n_dims) {
  if (is.null(n_dims)) {
    return(X)
  }
  if (!is.numeric(n_dims) || length(n_dims) != 1L || is.na(n_dims) || n_dims < 1 ||
      n_dims != round(n_dims)) {
    stop("n_dims must be a positive whole number.", call. = FALSE)
  }
  if (n_dims > ncol(X)) {
    stop(sprintf(
      "n_dims=%d exceeds the %d available components in the selected representation.",
      as.integer(n_dims), ncol(X)
    ), call. = FALSE)
  }
  X[, seq_len(n_dims), drop = FALSE]
}

# The first two columns of an embedding, for the scatter plots. where names
# the embedding in the error.
embedding_2d <- function(values, where) {
  values <- as.matrix(values)
  if (ncol(values) < 2L) {
    stop(sprintf(
      "%s has shape (%d, %d); a basis needs at least two columns.",
      where, nrow(values), ncol(values)
    ), call. = FALSE)
  }
  unname(values[, 1:2, drop = FALSE])
}
```

- [ ] Step 5: Create `R/sce.R`

```r
#' @include AllGenerics.R
NULL

# SingleCellExperiment input and storage. With seurat.R, mirrors
# _anndata.py: which matrix CARVE clusters, and where run_carve() and
# attach_results() write their results.

#' @importClassesFrom SingleCellExperiment SingleCellExperiment
#' @importFrom SummarizedExperiment colData colData<-
#' @importFrom S4Vectors metadata metadata<-
NULL

# Reduced dimensions the scatter plots look for, in order, when no basis is
# given. scater and scran name them so.
SCE_BASES <- c("UMAP", "TSNE", "PCA")

sce_matrix <- function(object, assay = NULL, reduction = NULL, n_dims = NULL) {
  check_one_source(assay, reduction)
  assays <- as.character(SummarizedExperiment::assayNames(object))
  reductions <- as.character(SingleCellExperiment::reducedDimNames(object))
  if (!is.null(assay)) {
    if (!assay %in% assays) {
      stop(sprintf(
        "assay=%s not found. Available assays: %s",
        format_repr(assay), python_list(sort(assays))
      ), call. = FALSE)
    }
    X <- cells_by_features(SummarizedExperiment::assay(object, assay))
  } else if (!is.null(reduction)) {
    if (!reduction %in% reductions) {
      stop(sprintf(
        "reduction=%s not found in reducedDimNames(object). Available names: %s. Pass assay=... to cluster an assay.",
        format_repr(reduction), python_list(sort(reductions))
      ), call. = FALSE)
    }
    X <- SingleCellExperiment::reducedDim(object, reduction)
  } else if ("PCA" %in% reductions) {
    X <- SingleCellExperiment::reducedDim(object, "PCA")
  } else if ("logcounts" %in% assays) {
    X <- cells_by_features(SummarizedExperiment::assay(object, "logcounts"))
    if (ncol(X) > 50L) {
      warning(sprintf(
        "No representation given and reducedDim(object, 'PCA') is absent, so CARVE is clustering the 'logcounts' assay directly (%d features). Clustering validation on high-dimensional raw data is slow and usually not what you want; compute a PCA first, for example with scater::runPCA(), and pass reduction='PCA'.",
        ncol(X)
      ), call. = FALSE)
    }
  } else {
    stop(
      "No representation given, reducedDim(object, 'PCA') is absent and there is no 'logcounts' assay. Pass reduction= or assay=.",
      call. = FALSE
    )
  }
  select_dims(as_data_matrix(X), n_dims)
}

sce_column_names <- function(object) {
  colnames(colData(object))
}

sce_column <- function(object, name) {
  data <- colData(object)
  if (name %in% colnames(data)) data[[name]] else NULL
}

sce_set_columns <- function(object, columns) {
  data <- colData(object)
  for (name in names(columns)) {
    data[[name]] <- columns[[name]]
  }
  colData(object) <- data
  object
}

sce_record <- function(object, key) {
  metadata(object)[[key]]
}

sce_set_record <- function(object, key, record) {
  metadata(object)[[key]] <- record
  object
}

sce_basis <- function(object, basis = NULL, fallback = NULL) {
  reductions <- as.character(SingleCellExperiment::reducedDimNames(object))
  read <- function(name) {
    embedding_2d(SingleCellExperiment::reducedDim(object, name), sprintf("reducedDim(object, '%s')", name))
  }
  if (!is.null(basis)) {
    if (!basis %in% reductions) {
      stop(sprintf(
        "basis=%s not found in reducedDimNames(object). Available names: %s",
        format_repr(basis), python_list(sort(reductions))
      ), call. = FALSE)
    }
    return(read(basis))
  }
  for (name in c(SCE_BASES, fallback)) {
    if (name %in% reductions) {
      return(read(name))
    }
  }
  NULL
}

#' @rdname carve
#' @param assay,reduction,n_dims For a SingleCellExperiment, which data to
#'   cluster: the assay named `assay` or the reduced dimensions named
#'   `reduction`, not both. By default CARVE clusters the `"PCA"` reduced
#'   dimensions, or the `"logcounts"` assay when there are none. `n_dims`
#'   keeps the first `n_dims` columns.
#' @export
setMethod("carve", "SingleCellExperiment", function(x, assay = NULL, reduction = NULL,
                                                    n_dims = NULL, ...) {
  carve(sce_matrix(x, assay = assay, reduction = reduction, n_dims = n_dims), ...)
})
```

- [ ] Step 6: Update the `carve()` roxygen in `R/carve.R`

Replace the `@param x` and `@param ...` entries:

```r
#' @param x The data: a numeric matrix with one row per sample, a numeric
#'   data frame, a sparse matrix from the Matrix package, a numeric vector,
#'   or a SingleCellExperiment, whose cells are its columns.
#' @param ... For a SingleCellExperiment, the arguments of the matrix
#'   method, from `n_clusters` on. For other input, not used: an argument
#'   name `carve()` does not know is an error.
```

- [ ] Step 7: Add the dependencies

In `DESCRIPTION`, add `S4Vectors`, `SingleCellExperiment` and `SummarizedExperiment` to Imports. In case-insensitive alphabetical order the field is then `BiocParallel, clue, FNN, ggplot2 (>= 3.5.0), igraph, irlba, Matrix, methods, parallel, ranger, RSpectra, Rtsne, S4Vectors, scales, SingleCellExperiment, stats, SummarizedExperiment, utils, withr (>= 2.4.2)`.

- [ ] Step 8: Run the tests

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "sce|utils")'
Rscript -e 'devtools::test()'
```

Expected: `document()` adds the imports to NAMESPACE, `exportMethods(carve)` stays, `'sce.R'` joins Collate, and `man/carve.Rd` gains the SingleCellExperiment usage and the three parameters. All tests pass, `SKIP 0`. Loading the package now loads SingleCellExperiment's namespace; that should not warn (checked for `loadNamespace()` while writing this plan).

- [ ] Step 9: Commit

```bash
git add carve-r/DESCRIPTION carve-r/NAMESPACE carve-r/R/sce.R carve-r/R/utils.R carve-r/R/carve.R carve-r/man/carve.Rd carve-r/tests/testthat/helper-data.R carve-r/tests/testthat/test-sce.R carve-r/tests/testthat/test-utils.R
git commit -m "$(cat <<'EOF'
feat(carve-r): fit a SingleCellExperiment and read and write its results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 10: Mutations

Apply each, run `Rscript -e 'devtools::test(filter = "sce|utils")'`, confirm the named test fails, and restore with `git checkout -- R/sce.R R/utils.R`:

1. In `sce_matrix()`, check for `"logcounts"` before `"PCA"`: fails "reads the PCA, then the logcounts".
2. In `cells_by_features()`, return `as.matrix(values)` without transposing: fails "turns cells-as-columns into cells-as-rows" and "reads the PCA, then the logcounts".
3. In `select_dims()`, `seq_len(n_dims - 1)`: fails "keeps the first n_dims columns".
4. In `sce_basis()`, loop over `SCE_BASES` only (drop `fallback`): fails "prefers UMAP, then TSNE, then PCA, then the fallback".
5. In `sce_basis()`, `rev(SCE_BASES)`: fails the same test.
6. In `sce_matrix()`, drop the 50-feature warning: fails "warns before clustering many logcounts features".

---

### Task 8: Seurat input and storage

Files:
- Create: `carve-r/R/seurat.R`
- Modify: `carve-r/R/carve.R` (the `ANY` method routes Seurat objects; roxygen of `x` and `...`)
- Modify: `carve-r/R/sce.R` (roxygen of `assay`, `reduction`, `n_dims`)
- Modify: `carve-r/DESCRIPTION` (Suggests `SeuratObject`)
- Modify: `carve-r/tests/testthat/helper-data.R` (`make_seurat()`)
- Create: `carve-r/tests/testthat/test-seurat.R`

Interfaces:
- Consumes: Task 7's `check_one_source()`, `cells_by_features()`, `select_dims()`, `embedding_2d()`; `require_package()`, `check_dots()`, `as_data_matrix()` (`utils.R`); `fit_carve()` (`carve.R`).
- Produces (`R/seurat.R`, internal): `SEURAT_BASES` (`c("umap", "tsne", "pca")`); `is_seurat(object)`; `seurat_selection(assay = NULL, reduction = NULL, n_dims = NULL, ...)`; `seurat_matrix(object, assay = NULL, reduction = NULL, n_dims = NULL)`; `seurat_data_layer(object, assay)`; `seurat_column(object, name)`; `seurat_column_names(object)`; `seurat_set_columns(object, columns)`; `seurat_record(object, key)`; `seurat_set_record(object, key, record)`; `seurat_basis(object, basis = NULL, fallback = NULL)`. The same contracts as Task 7's `sce_*` functions.
- `carve(x)` for a Seurat object takes `assay`, `reduction` and `n_dims` in `...`.
- Test helper: `make_seurat(X, reductions = list(pca = X), data = t(X))`; `data = NULL` leaves the assay without a `"data"` layer. It skips the test when SeuratObject is not installed.

- [ ] Step 1: Add the test helper to `tests/testthat/helper-data.R`

```r
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
```

These calls run under `warn = 2` without a warning in SeuratObject 5.4.0 (checked while writing this plan, with three genes and six cells).

- [ ] Step 2: Write the failing tests

Create `tests/testthat/test-seurat.R`:

```r
blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:3))

test_that("seurat_matrix reads the pca reduction, then the data layer", {
  object <- make_seurat(blobs$X, data = t(blobs$X * 10))
  expect_equal(unname(seurat_matrix(object)), blobs$X)
  expect_equal(unname(seurat_matrix(object, assay = "RNA")), blobs$X * 10)
  no_pca <- make_seurat(blobs$X, reductions = list(), data = t(blobs$X * 10))
  expect_equal(unname(seurat_matrix(no_pca)), blobs$X * 10)
  with_umap <- make_seurat(blobs$X, reductions = list(pca = blobs$X, umap = blobs$X * 2))
  expect_equal(unname(seurat_matrix(with_umap, reduction = "umap")), blobs$X * 2)
  expect_equal(unname(seurat_matrix(object, n_dims = 1)), blobs$X[, 1L, drop = FALSE])
})

test_that("seurat_matrix warns before clustering many features", {
  wide <- withr::with_seed(1L, matrix(stats::rnorm(90L * 51L), 90L, 51L))
  object <- make_seurat(wide, reductions = list())
  expect_warning(
    X <- seurat_matrix(object),
    "No representation given and the 'pca' reduction is absent, so CARVE is clustering the data layer of assay 'RNA' directly (51 features).",
    fixed = TRUE
  )
  expect_identical(dim(X), c(90L, 51L))
})

test_that("seurat_matrix names what it cannot find", {
  object <- make_seurat(blobs$X)
  expect_error(seurat_matrix(object, assay = "RNA", reduction = "pca"), "Pass at most one of reduction= and assay=", fixed = TRUE)
  expect_error(seurat_matrix(object, assay = "ADT"), "assay='ADT' not found. Available assays: ['RNA']", fixed = TRUE)
  expect_error(
    seurat_matrix(object, reduction = "umap"),
    "reduction='umap' not found in Reductions(object). Available names: ['pca']. Pass assay=... to cluster an assay.",
    fixed = TRUE
  )
  unnormalized <- make_seurat(blobs$X, reductions = list(), data = NULL)
  expect_error(
    seurat_matrix(unnormalized),
    "Assay 'RNA' has no 'data' layer. Normalize it first, for example with Seurat::NormalizeData(), or pass reduction=.",
    fixed = TRUE
  )
})

test_that("carve() fits a Seurat object as it fits the matrix", {
  object <- make_seurat(blobs$X)
  fit <- carve(object, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  reference <- carve(blobs$X, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  expect_identical(estimator_results(fit), estimator_results(reference))
  expect_equal(unname(input_data(fit)), blobs$X)
  one <- carve(object, reduction = "pca", n_dims = 1, n_resamples = 3, random_state = 0, estimator_param_grids = k_grid)
  expect_equal(unname(input_data(one)), blobs$X[, 1L, drop = FALSE])
  expect_error(carve(object, nope = 1, estimator_param_grids = k_grid), "Unknown argument: nope.", fixed = TRUE)
  expect_error(carve(blobs$X, assay = "RNA", estimator_param_grids = k_grid), "Unknown argument: assay.", fixed = TRUE)
})

test_that("Seurat columns and records round-trip, and a record is replaced without a warning", {
  object <- make_seurat(blobs$X)
  labels <- factor(rep(c("1", "2"), 45))
  updated <- seurat_set_columns(object, list(carve = labels, carve_stability = seq_len(90) / 90))
  expect_identical(seurat_column(updated, "carve"), labels)
  expect_equal(seurat_column(updated, "carve_stability"), seq_len(90) / 90)
  expect_null(seurat_column(updated, "nope"))
  expect_true(all(c("carve", "carve_stability") %in% seurat_column_names(updated)))
  record <- list(params = list(measure = "stability"))
  stored <- seurat_set_record(updated, "carve", record)
  expect_identical(seurat_record(stored, "carve"), record)
  expect_null(seurat_record(stored, "other"))
  expect_no_warning(replaced <- seurat_set_record(stored, "carve", list(params = list(measure = "g"))))
  expect_identical(seurat_record(replaced, "carve")$params$measure, "g")
})

test_that("seurat_basis prefers umap, then tsne, then pca, then the fallback", {
  X <- blobs$X
  all_three <- make_seurat(X, reductions = list(pca = X, tsne = X * 3, umap = X * 2))
  expect_equal(seurat_basis(all_three), X * 2)
  expect_equal(seurat_basis(make_seurat(X, reductions = list(pca = X, tsne = X * 3))), X * 3)
  expect_equal(seurat_basis(make_seurat(X)), X)
  custom <- make_seurat(X, reductions = list(harmony = X * 4))
  expect_null(seurat_basis(custom))
  expect_equal(seurat_basis(custom, fallback = "harmony"), X * 4)
  expect_error(
    seurat_basis(all_three, basis = "UMAP"),
    "basis='UMAP' not found in Reductions(object). Available names: ['pca', 'tsne', 'umap']",
    fixed = TRUE
  )
})
```

- [ ] Step 3: Run the tests to see them fail

```bash
Rscript -e 'devtools::test(filter = "seurat")'
```

Expected: FAIL, `could not find function "seurat_matrix"`.

- [ ] Step 4: Create `R/seurat.R`

```r
#' @include AllGenerics.R
NULL

# Seurat input and storage, through SeuratObject, which is suggested. With
# sce.R, mirrors _anndata.py. A Seurat object keeps its cell metadata in
# the meta.data slot, and CARVE keeps its record in the misc slot.

# Reductions the scatter plots look for, in order, when no basis is given.
# Seurat names them so.
SEURAT_BASES <- c("umap", "tsne", "pca")

# methods::is() matches the class name even before SeuratObject is loaded,
# so a Seurat object read with readRDS() is recognized.
is_seurat <- function(object) {
  methods::is(object, "Seurat")
}

# The arguments carve() takes for a Seurat object in its ... .
seurat_selection <- function(assay = NULL, reduction = NULL, n_dims = NULL, ...) {
  check_dots(...)
  list(assay = assay, reduction = reduction, n_dims = n_dims)
}

seurat_data_layer <- function(object, assay) {
  if (!"data" %in% SeuratObject::Layers(object[[assay]])) {
    stop(sprintf(
      "Assay '%s' has no 'data' layer. Normalize it first, for example with Seurat::NormalizeData(), or pass reduction=.",
      assay
    ), call. = FALSE)
  }
  cells_by_features(SeuratObject::LayerData(object, assay = assay, layer = "data"))
}

seurat_matrix <- function(object, assay = NULL, reduction = NULL, n_dims = NULL) {
  require_package("SeuratObject", "Seurat objects")
  check_one_source(assay, reduction)
  reductions <- as.character(SeuratObject::Reductions(object))
  if (!is.null(assay)) {
    assays <- as.character(SeuratObject::Assays(object))
    if (!assay %in% assays) {
      stop(sprintf(
        "assay=%s not found. Available assays: %s",
        format_repr(assay), python_list(sort(assays))
      ), call. = FALSE)
    }
    X <- seurat_data_layer(object, assay)
  } else if (!is.null(reduction)) {
    if (!reduction %in% reductions) {
      stop(sprintf(
        "reduction=%s not found in Reductions(object). Available names: %s. Pass assay=... to cluster an assay.",
        format_repr(reduction), python_list(sort(reductions))
      ), call. = FALSE)
    }
    X <- SeuratObject::Embeddings(object, reduction = reduction)
  } else if ("pca" %in% reductions) {
    X <- SeuratObject::Embeddings(object, reduction = "pca")
  } else {
    default <- SeuratObject::DefaultAssay(object)
    X <- seurat_data_layer(object, default)
    if (ncol(X) > 50L) {
      warning(sprintf(
        "No representation given and the 'pca' reduction is absent, so CARVE is clustering the data layer of assay '%s' directly (%d features). Clustering validation on high-dimensional raw data is slow and usually not what you want; run Seurat::RunPCA() first and pass reduction='pca'.",
        default, ncol(X)
      ), call. = FALSE)
    }
  }
  select_dims(as_data_matrix(X), n_dims)
}

seurat_column_names <- function(object) {
  colnames(methods::slot(object, "meta.data"))
}

seurat_column <- function(object, name) {
  data <- methods::slot(object, "meta.data")
  if (name %in% colnames(data)) data[[name]] else NULL
}

seurat_set_columns <- function(object, columns) {
  data <- data.frame(columns, row.names = colnames(object), check.names = FALSE)
  SeuratObject::AddMetaData(object, metadata = data)
}

seurat_record <- function(object, key) {
  methods::slot(object, "misc")[[key]]
}

# SeuratObject's Misc<- warns whenever it replaces an entry, and
# run_carve() replaces its entry on every call, so the slot is set directly.
seurat_set_record <- function(object, key, record) {
  misc <- methods::slot(object, "misc")
  misc[[key]] <- record
  methods::slot(object, "misc") <- misc
  object
}

seurat_basis <- function(object, basis = NULL, fallback = NULL) {
  reductions <- as.character(SeuratObject::Reductions(object))
  read <- function(name) {
    embedding_2d(SeuratObject::Embeddings(object, reduction = name), sprintf("Embeddings(object, '%s')", name))
  }
  if (!is.null(basis)) {
    if (!basis %in% reductions) {
      stop(sprintf(
        "basis=%s not found in Reductions(object). Available names: %s",
        format_repr(basis), python_list(sort(reductions))
      ), call. = FALSE)
    }
    return(read(basis))
  }
  for (name in c(SEURAT_BASES, fallback)) {
    if (name %in% reductions) {
      return(read(name))
    }
  }
  NULL
}
```

- [ ] Step 5: Route Seurat objects in the `ANY` method of `carve()`

In `R/carve.R`, the body of `setMethod("carve", "ANY", ...)` begins with these three lines:

```r
  check_dots(...)
  fit_carve(
    as_data_matrix(x),
```

Replace them with:

```r
  # A Seurat object is an S4 class of a suggested package, so no method can
  # name it; it arrives here, with assay, reduction and n_dims in ... .
  if (is_seurat(x)) {
    selection <- seurat_selection(...)
    data <- seurat_matrix(
      x,
      assay = selection$assay,
      reduction = selection$reduction,
      n_dims = selection$n_dims
    )
  } else {
    check_dots(...)
    data <- as_data_matrix(x)
  }
  fit_carve(
    data,
```

The rest of the `fit_carve()` call stays as it is.

- [ ] Step 6: Update the roxygen for both classes

In `R/carve.R`, replace the `@param x` and `@param ...` entries Task 7 wrote:

```r
#' @param x The data: a numeric matrix with one row per sample, a numeric
#'   data frame, a sparse matrix from the Matrix package, a numeric vector,
#'   or a SingleCellExperiment or Seurat object, whose cells are its columns.
#' @param ... For a SingleCellExperiment, the arguments of the matrix
#'   method, from `n_clusters` on. For a Seurat object, `assay`, `reduction`
#'   and `n_dims`. For other input, not used: an argument name `carve()`
#'   does not know is an error.
```

In `R/sce.R`, replace the `@param assay,reduction,n_dims` entry:

```r
#' @param assay,reduction,n_dims For a SingleCellExperiment or a Seurat
#'   object, which data to cluster: the assay named `assay` or the reduced
#'   dimensions named `reduction`, not both. For a Seurat object, `assay`
#'   names a Seurat assay such as `"RNA"`, and CARVE clusters its `"data"`
#'   layer. By default CARVE clusters the `"PCA"` reduced dimensions of a
#'   SingleCellExperiment, or its `"logcounts"` assay when there are none,
#'   and the `"pca"` reduction of a Seurat object, or the `"data"` layer of
#'   its default assay. `n_dims` keeps the first `n_dims` columns.
```

- [ ] Step 7: Add SeuratObject to Suggests

In `DESCRIPTION`, Suggests becomes `dbscan, jsonlite, SeuratObject, testthat (>= 3.1.8), uwot`.

- [ ] Step 8: Run the tests

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "seurat|sce|carve")'
Rscript -e 'devtools::test()'
```

Expected: `document()` adds `'seurat.R'` to Collate and rewrites `man/carve.Rd`. All tests pass, `SKIP 0`.

- [ ] Step 9: Commit

```bash
git add carve-r/DESCRIPTION carve-r/R/seurat.R carve-r/R/carve.R carve-r/R/sce.R carve-r/man/carve.Rd carve-r/tests/testthat/helper-data.R carve-r/tests/testthat/test-seurat.R
git commit -m "$(cat <<'EOF'
feat(carve-r): fit a Seurat object and read and write its results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 10: Mutations

Apply each, run `Rscript -e 'devtools::test(filter = "seurat")'`, confirm the named test fails, and restore with `git checkout -- R/seurat.R R/carve.R`:

1. In `seurat_matrix()`, check the data layer before the `"pca"` reduction: fails "reads the pca reduction, then the data layer".
2. In `seurat_set_record()`, write with `SeuratObject::Misc(object, slot = key) <- record`: fails "a record is replaced without a warning".
3. In `seurat_selection()`, drop `check_dots(...)`: fails "fits a Seurat object as it fits the matrix" (`nope` is accepted).
4. In `seurat_data_layer()`, drop the layer check: fails "names what it cannot find" (SeuratObject's own error text instead of CARVE's).
5. In `seurat_basis()`, loop over `SEURAT_BASES` only: fails "prefers umap, then tsne, then pca, then the fallback".
6. In the `ANY` method, route on `inherits(x, "SingleCellExperiment")` instead of `is_seurat(x)`: fails "fits a Seurat object as it fits the matrix".

---

### Task 9: `run_carve()` and `attach_results()`

Files:
- Create: `carve-r/R/tl.R`
- Modify: `carve-r/R/carve.R` (`run_params$estimator_param_grids`)
- Modify: `carve-r/R/AllClasses.R` (the `run_params` slot doc)
- Create: `carve-r/tests/testthat/test-tl.R`
- Modify: `carve-r/tests/testthat/test-carve.R`

Interfaces:
- Consumes: the `sce_*` (Task 7) and `seurat_*` (Task 8) functions; `select_row()`, `get_labels()` (`accessors.R`); `format_repr()`, `python_list()` (`utils.R`).
- Produces:
  - `R/tl.R`, internal: `cells_kind(object)` (`"sce"` or `"seurat"`, else an error); the dispatchers `cells_matrix()`, `cells_get_column()`, `cells_column_names()`, `cells_set_columns()`, `cells_get_record()`, `cells_set_record()`, `cells_basis()`, with the arguments of the `sce_*` functions; `cells_columns_where(object)` (`"colData(object)"` or `"object@meta.data"`), `cells_column_where(object, name)` and `cells_record_where(object, key)` (`"metadata(object)$<key>"` or `"Misc(object, '<key>')"`), the R paths that messages name; `CONSENSUS_WARN_CELLS` (10000); `check_cell_count(values, n_cells, what)`; `record_params(fit, selected, measure, rule, not_two, consensus_k, assay, reduction, n_dims, mode, n_cells)`.
  - Exported: `run_carve(object, ..., assay = NULL, reduction = NULL, n_dims = NULL, key = "carve", measure = "stability", rule = "1se", not_two = FALSE, k = NULL, sweep_value = NULL, consensus_k = NULL, reference_key = NULL, store_consensus = TRUE, store_results = TRUE, mode = "default", random_state = 0)` and `attach_results(object, fit, key = "carve", measure = "stability", rule = "1se", not_two = FALSE, k = NULL, sweep_value = NULL, consensus_k = NULL, store_consensus = TRUE, store_results = TRUE, assay = NULL, reduction = NULL, n_dims = NULL, mode = "default")`, both returning the object.
  - The record, a list: `params` (named list, no `NULL` entries), `results` and `preprocessing_results` (data frames, when stored), `consensus` (matrix, when stored). Tasks 10 and 11 read it.
  - `fit@run_params$estimator_param_grids`: `"light"`, `"full"` or `"custom"`.

- [ ] Step 1: Write the failing tests

Append to `tests/testthat/test-carve.R`:

```r
test_that("run_params records which estimator grids the run used", {
  X <- make_blobs()$X
  light <- carve(X, n_clusters = 2:3, n_resamples = 2, random_state = 0)
  expect_identical(light@run_params$estimator_param_grids, "light")
  custom <- carve(X, n_resamples = 2, random_state = 0,
                  estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:3)))
  expect_identical(custom@run_params$estimator_param_grids, "custom")
})
```

Create `tests/testthat/test-tl.R`:

```r
blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
run <- function(object, ...) run_carve(object, n_resamples = 6, estimator_param_grids = k_grid, ...)
direct_fit <- function(...) carve(blobs$X, n_resamples = 6, estimator_param_grids = k_grid, random_state = 0, ...)
sce <- make_sce(blobs$X)
written <- run(sce)
fit <- direct_fit()

test_that("run_carve writes the labels, the scores and the record", {
  data <- SummarizedExperiment::colData(written)
  expect_true(all(c("carve", "carve_stability", "carve_stability_ce", "carve_generalizability") %in% colnames(data)))
  expect_s3_class(data$carve, "factor")
  expect_identical(levels(data$carve), c("1", "2", "3"))
  for (column in c("carve_stability", "carve_stability_ce", "carve_generalizability")) {
    expect_type(data[[column]], "double")
    expect_length(data[[column]], 90L)
  }
  record <- S4Vectors::metadata(written)$carve
  expect_named(record, c("params", "results", "consensus"))
  expect_identical(dim(record$consensus), c(90L, 90L))
  expect_false(any(vapply(record$params, is.null, logical(1))))
})

test_that("the stored labels, scores, table and matrix are the fit's", {
  id <- select_row(fit, "stability", "1se")$config_id
  data <- SummarizedExperiment::colData(written)
  expect_identical(as.integer(as.character(data$carve)), get_labels(fit))
  expect_equal(data$carve_stability, sample_scores(fit, id, "gini"))
  expect_equal(data$carve_stability_ce, sample_scores(fit, id, "ce"))
  expect_equal(data$carve_generalizability, sample_scores(fit, id, "accuracy"))
  record <- S4Vectors::metadata(written)$carve
  expect_equal(record$consensus, consensus_matrix(fit, id))
  expect_identical(record$results, estimator_results(fit))
})

test_that("the record's params name the run and the selection", {
  params <- S4Vectors::metadata(written)$carve$params
  selected <- select_row(fit, "stability", "1se")
  expect_identical(params$measure, "stability")
  expect_identical(params$rule, "1se")
  expect_false(params$not_two)
  expect_false(params$pinned)
  expect_identical(params$sweep_param, "n_clusters")
  expect_equal(params$n_resamples, 6)
  expect_equal(params$subsample_ratio, 0.618)
  expect_identical(params$noise_policy, "drop")
  expect_identical(params$mode, "default")
  expect_equal(params$random_state, 0)
  expect_identical(params$estimator_param_grids, "custom")
  expect_equal(params$selected_config_id, selected$config_id)
  expect_equal(params$selected_k, get_k(fit))
  expect_equal(params$n_obs, 90)
  expect_equal(params$sweep_values, c(2, 3, 4))
  expect_equal(params$selected_n_clusters, selected$row$n_clusters[[1L]])
  expect_identical(params$selected_estimator, "KMeans")
  expect_identical(params$selected_method_label, as.character(selected$row$method_label))
  expect_identical(params$selected_method_id, as.character(selected$row$method_id))
  expect_identical(params$carve_version, as.character(utils::packageVersion("CARVE")))
  expect_false(any(c("n_consensus_anchors", "assay", "reduction", "n_dims", "consensus_k") %in% names(params)))
  recorded <- S4Vectors::metadata(run(sce, assay = "logcounts", n_dims = 1, consensus_k = 2))$carve$params
  expect_identical(recorded$assay, "logcounts")
  expect_equal(recorded$n_dims, 1)
  expect_equal(recorded$consensus_k, 2)
})

test_that("label levels are in numeric order", {
  many <- attach_results(sce, fit, consensus_k = 11)
  expect_identical(levels(SummarizedExperiment::colData(many)$carve), as.character(1:11))
})

test_that("key names every column and the record, and two keys coexist", {
  other <- run(sce, key = "other")
  data <- SummarizedExperiment::colData(other)
  expect_true(all(c("other", "other_stability", "other_stability_ce", "other_generalizability") %in% colnames(data)))
  expect_false("carve" %in% colnames(data))
  expect_named(S4Vectors::metadata(other), "other")
  both <- run(written, key = "second", rule = "max")
  expect_true(all(c("carve", "second") %in% names(S4Vectors::metadata(both))))
  expect_identical(S4Vectors::metadata(both)$second$params$rule, "max")
})

test_that("store_consensus and store_results leave entries out, and a call replaces the record", {
  no_matrix <- run(sce, store_consensus = FALSE)
  expect_named(S4Vectors::metadata(no_matrix)$carve, c("params", "results"))
  no_table <- run(sce, store_results = FALSE)
  expect_named(S4Vectors::metadata(no_table)$carve, c("params", "consensus"))
  replaced <- run(written, store_consensus = FALSE)
  expect_null(S4Vectors::metadata(replaced)$carve$consensus)
})

test_that("a mode skips its columns, and an earlier column stays", {
  stability <- muffle(run(sce, mode = "stability"), "Non-default mode is experimental")
  data <- SummarizedExperiment::colData(stability)
  expect_true("carve_stability" %in% colnames(data))
  expect_false("carve_generalizability" %in% colnames(data))
  generalizability <- muffle(run(sce, mode = "generalizability"), "Non-default mode is experimental")
  data <- SummarizedExperiment::colData(generalizability)
  expect_false("carve_stability" %in% colnames(data))
  general_fit <- muffle(direct_fit(mode = "generalizability"), "Non-default mode is experimental")
  id <- select_row(general_fit, "stability", "1se")$config_id
  expect_equal(S4Vectors::metadata(generalizability)$carve$consensus, consensus_matrix(general_fit, id, type = "generalizability"))
  expect_identical(as.integer(as.character(data$carve)), get_labels(general_fit, mode = "generalizability"))
  rerun <- muffle(run(written, mode = "stability"), "Non-default mode is experimental")
  expect_identical(
    SummarizedExperiment::colData(rerun)$carve_generalizability,
    SummarizedExperiment::colData(written)$carve_generalizability
  )
  expect_error(
    muffle(run(sce, mode = "stability", measure = "generalizability"), "Non-default mode is experimental"),
    "No configuration has a value for ari_generalizability.",
    fixed = TRUE
  )
})

test_that("reference_key names a column whose labels the stored ones match", {
  with_truth <- sce
  SummarizedExperiment::colData(with_truth)$truth <- c(3, 1, 2)[blobs$y]
  matched <- run(with_truth, reference_key = "truth")
  expect_identical(as.integer(as.character(SummarizedExperiment::colData(matched)$carve)), as.integer(c(3, 1, 2)[blobs$y]))
  expect_error(run(with_truth, reference_key = "nope"), "reference_key='nope' not found in colData(object). Available columns: ['truth']", fixed = TRUE)
})

test_that("k pins the configuration and a resolution run records its axis", {
  pinned <- S4Vectors::metadata(run(sce, k = 2))$carve$params
  expect_true(pinned$pinned)
  expect_equal(pinned$selected_k, 2)
  graph <- run_carve(sce, n_resamples = 4, estimator_param_grids = list(estimator_grid(LeidenClustering, resolution = c(0.5, 1))))
  params <- S4Vectors::metadata(graph)$carve$params
  expect_identical(params$sweep_param, "resolution")
  expect_true(params$selected_resolution %in% c(0.5, 1))
})

test_that("attach_results stores what run_carve stores", {
  attached <- attach_results(sce, fit)
  expect_identical(SummarizedExperiment::colData(attached), SummarizedExperiment::colData(written))
  expect_identical(S4Vectors::metadata(attached), S4Vectors::metadata(written))
})

test_that("attach_results checks its inputs", {
  expect_error(attach_results(sce, list()), "fit must be a CARVE object from carve().", fixed = TRUE)
  expect_error(
    attach_results(blobs$X, fit),
    "object must be a SingleCellExperiment or a Seurat object, not an object of class 'matrix'.",
    fixed = TRUE
  )
  small <- make_sce(blobs$X[1:60, ])
  expect_error(
    attach_results(small, fit),
    "The model produced 90 consensus labels but object has 60 observations. The model was fitted on different data.",
    fixed = TRUE
  )
  id <- as.character(select_row(fit, "stability", "1se")$config_id)
  broken <- fit
  broken@consensus_matrices[[id]] <- diag(5)
  local_mocked_bindings(get_labels = function(fit, ...) rep(1:3, each = 30L))
  expect_error(
    attach_results(sce, broken),
    "Consensus matrix has shape (5, 5), but object has 90 observations. The model was fitted on different data.",
    fixed = TRUE
  )
})

test_that("an anchored run records its anchor count and stores no block", {
  anchored <- muffle(direct_fit(anchor_threshold = 30), "anchored consensus")
  id <- select_row(anchored, "stability", "1se")$config_id
  expect_warning(
    stored <- attach_results(sce, anchored),
    sprintf(
      "Anchored consensus is active: the consensus matrix for 'carve' is a 30x30 anchor block, not a cell-by-cell matrix. metadata(object)$carve holds only the latter, so nothing was written to metadata(object)$carve$consensus. The block itself is still available with consensus_matrix(fit, config_id = %d), and the anchor count is recorded in metadata(object)$carve$params$n_consensus_anchors.",
      id
    ),
    fixed = TRUE
  )
  record <- S4Vectors::metadata(stored)$carve
  expect_null(record$consensus)
  expect_equal(record$params$n_consensus_anchors, 30)
  chosen <- muffle(direct_fit(consensus_anchors = 20), "anchored consensus")
  expect_equal(
    S4Vectors::metadata(muffle(attach_results(sce, chosen), "Anchored consensus is active"))$carve$params$n_consensus_anchors,
    20
  )
})

test_that("a large consensus matrix is stored with a warning", {
  local_mocked_bindings(CONSENSUS_WARN_CELLS = 50L)
  expect_warning(
    stored <- attach_results(sce, fit),
    "Storing a dense 90x90 consensus matrix in metadata(object)$carve$consensus (~0.0 GB). Pass store_consensus=FALSE if you do not need it.",
    fixed = TRUE
  )
  expect_identical(dim(S4Vectors::metadata(stored)$carve$consensus), c(90L, 90L))
})

test_that("a randomized run stores its per-pipeline table", {
  options <- list(
    normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
    dim_reduction_options = list(preprocessing_option(Identity))
  )
  randomized_fit <- do.call(direct_fit, c(list(randomize_preprocessing = TRUE), options))
  stored <- attach_results(sce, randomized_fit)
  record <- S4Vectors::metadata(stored)$carve
  expect_identical(record$preprocessing_results, preprocessing_results(randomized_fit))
  expect_identical(record$params$selected_method_id, as.character(select_row(randomized_fit, "stability", "1se")$row$method_id))
  expect_null(S4Vectors::metadata(written)$carve$preprocessing_results)
  expect_null(S4Vectors::metadata(attach_results(sce, randomized_fit, store_results = FALSE))$carve$preprocessing_results)
})

test_that("run_carve writes the same results into a Seurat object", {
  object <- make_seurat(blobs$X)
  stored <- run(object)
  meta <- methods::slot(stored, "meta.data")
  data <- SummarizedExperiment::colData(written)
  expect_identical(meta$carve, data$carve)
  expect_equal(meta$carve_stability, data$carve_stability)
  expect_equal(meta$carve_generalizability, data$carve_generalizability)
  record <- methods::slot(stored, "misc")$carve
  expect_identical(record$results, S4Vectors::metadata(written)$carve$results)
  expect_equal(record$consensus, S4Vectors::metadata(written)$carve$consensus)
  expect_no_warning(again <- run(stored, rule = "max"))
  expect_identical(methods::slot(again, "misc")$carve$params$rule, "max")
  anchored <- muffle(direct_fit(anchor_threshold = 30), "anchored consensus")
  expect_warning(attach_results(object, anchored), "Misc(object, 'carve') holds only the latter", fixed = TRUE)
})
```

The test that mocks `get_labels()` makes the labels the right length, so the check after them, on the matrix's shape, is the one that fails; Python's test reaches that check the same way. The large-matrix test mocks the threshold, a constant: if `local_mocked_bindings()` refuses a binding that is not a function, turn `CONSENSUS_WARN_CELLS` into a function `consensus_warn_cells()` that returns 10000, mock that instead, and say so in the report.

- [ ] Step 2: Run the tests to see them fail

```bash
Rscript -e 'devtools::test(filter = "tl|carve")'
```

Expected: FAIL, `could not find function "run_carve"`, and the `run_params` test fails on `NULL`.

- [ ] Step 3: Record the grid preset in `R/carve.R`

In `fit_carve()`, in the `run_params = list(...)` of `methods::new("CARVE", ...)`, add after `randomize_preprocessing = randomize_preprocessing,`:

```r
      estimator_param_grids = if (is_preset) estimator_param_grids else "custom",
```

In `R/AllClasses.R`, replace the `@slot run_params` entry:

```r
#' @slot run_params List of the run settings: `n_resamples`,
#'   `subsample_ratio`, `n_trees`, `n_jobs`, `mode`, `random_state` (the seed
#'   the run used), `classifier`, `noise_policy`, `anchor_threshold`,
#'   `consensus_anchors`, `randomize_preprocessing`, `estimator_param_grids`
#'   (`"light"`, `"full"` or `"custom"`), and `n_threads`, the threads of the
#'   classifier [get_labels()] trains on an anchored run.
```

- [ ] Step 4: Create `R/tl.R`

```r
#' @include AllGenerics.R
NULL

# run_carve() and attach_results(). Mirrors tl/_carve.py. The functions
# below read and write a SingleCellExperiment through sce.R and a Seurat
# object through seurat.R, so the plot methods share them too.

cells_kind <- function(object) {
  if (methods::is(object, "SingleCellExperiment")) {
    return("sce")
  }
  if (is_seurat(object)) {
    require_package("SeuratObject", "Seurat objects")
    return("seurat")
  }
  stop(sprintf(
    "object must be a SingleCellExperiment or a Seurat object, not an object of class '%s'.",
    class(object)[[1L]]
  ), call. = FALSE)
}

cells_matrix <- function(object, assay = NULL, reduction = NULL, n_dims = NULL) {
  switch(cells_kind(object),
    sce = sce_matrix(object, assay = assay, reduction = reduction, n_dims = n_dims),
    seurat = seurat_matrix(object, assay = assay, reduction = reduction, n_dims = n_dims)
  )
}

cells_column_names <- function(object) {
  switch(cells_kind(object), sce = sce_column_names(object), seurat = seurat_column_names(object))
}

cells_get_column <- function(object, name) {
  switch(cells_kind(object), sce = sce_column(object, name), seurat = seurat_column(object, name))
}

cells_set_columns <- function(object, columns) {
  switch(cells_kind(object),
    sce = sce_set_columns(object, columns),
    seurat = seurat_set_columns(object, columns)
  )
}

cells_get_record <- function(object, key) {
  switch(cells_kind(object), sce = sce_record(object, key), seurat = seurat_record(object, key))
}

cells_set_record <- function(object, key, record) {
  switch(cells_kind(object),
    sce = sce_set_record(object, key, record),
    seurat = seurat_set_record(object, key, record)
  )
}

cells_basis <- function(object, basis = NULL, fallback = NULL) {
  switch(cells_kind(object),
    sce = sce_basis(object, basis = basis, fallback = fallback),
    seurat = seurat_basis(object, basis = basis, fallback = fallback)
  )
}

# Where the results live, as R code a user can type, for messages.
cells_columns_where <- function(object) {
  switch(cells_kind(object), sce = "colData(object)", seurat = "object@meta.data")
}

cells_column_where <- function(object, name) {
  sprintf("%s$%s", cells_columns_where(object), name)
}

cells_record_where <- function(object, key) {
  switch(cells_kind(object),
    sce = sprintf("metadata(object)$%s", key),
    seurat = sprintf("Misc(object, '%s')", key)
  )
}

# A dense matrix this large is worth a warning: 10,000 cells take 800 MB.
CONSENSUS_WARN_CELLS <- 10000L

check_cell_count <- function(values, n_cells, what) {
  if (length(values) != n_cells) {
    stop(sprintf(
      "The model produced %d %s but object has %d observations. The model was fitted on different data.",
      length(values), what, n_cells
    ), call. = FALSE)
  }
}

# The settings stored as the record's params. Entries without a value are
# left out, as Python leaves out None.
record_params <- function(fit, selected, measure, rule, not_two, consensus_k, assay, reduction,
                          n_dims, mode, n_cells) {
  settings <- fit@run_params
  param <- fit@sweep@param
  row <- selected$row
  params <- list(
    measure = measure,
    rule = rule,
    not_two = isTRUE(not_two),
    pinned = isTRUE(selected$pinned),
    sweep_param = param,
    n_resamples = settings$n_resamples,
    subsample_ratio = settings$subsample_ratio,
    n_consensus_anchors = if (is.null(fit@consensus_anchors)) NULL else length(fit@consensus_anchors),
    noise_policy = settings$noise_policy,
    mode = mode,
    random_state = settings$random_state,
    assay = assay,
    reduction = reduction,
    n_dims = n_dims,
    consensus_k = consensus_k,
    estimator_param_grids = settings$estimator_param_grids,
    selected_config_id = selected$config_id,
    selected_k = selected$n_clusters,
    n_obs = n_cells,
    sweep_values = as.numeric(fit@sweep@values)
  )
  params <- params[!vapply(params, is.null, logical(1))]
  if (param %in% names(row)) {
    params[[paste0("selected_", param)]] <- row[[param]][[1L]]
  }
  for (name in c("estimator", "method_label", "method_id")) {
    if (name %in% names(row)) {
      params[[paste0("selected_", name)]] <- as.character(row[[name]][[1L]])
    }
  }
  params$carve_version <- as.character(utils::packageVersion("CARVE"))
  params
}

#' Run CARVE on a single-cell object
#'
#' `run_carve()` fits [carve()] to a SingleCellExperiment or a Seurat object
#' and stores the results of the selected configuration in it.
#' `attach_results()` stores them from a fit you already have, for example
#' one you keep to look at other configurations.
#'
#' With the default `key = "carve"`, these cell columns are written, to
#' `colData()` of a SingleCellExperiment or to the cell metadata of a Seurat
#' object:
#'
#' - `carve`: the labels [get_labels()] returns, as a factor with its levels
#'   in numeric order.
#' - `carve_stability` and `carve_stability_ce`: per-cell stability in its
#'   Gini and cross-entropy forms, as [sample_scores()] returns them.
#' - `carve_generalizability`: per-cell held-out accuracy.
#'
#' A column is missing when `mode` skipped its criterion. A call leaves the
#' columns of an earlier call with the same key that it does not write.
#'
#' The call also stores a record: `metadata(object)$carve` for a
#' SingleCellExperiment, and the `carve` entry of the `misc` slot of a Seurat
#' object. The record is a list:
#'
#' - `params`: the run and selection settings, among them the selected
#'   `config_id`, `k` and `method_id`.
#' - `results`: the table [estimator_results()] returns, unless
#'   `store_results = FALSE`.
#' - `preprocessing_results`: the table [preprocessing_results()] returns,
#'   for a run with `randomize_preprocessing = TRUE` and unless
#'   `store_results = FALSE`.
#' - `consensus`: the consensus matrix of the selected configuration, unless
#'   `store_consensus = FALSE`. An anchored run has only the block over its
#'   anchors, which is not stored; a warning says so.
#'
#' Each call replaces the record. The plot functions read the columns and
#' the record, so `plot_cluster_violin(object)` draws the stored
#' configuration without fitting again. The fit itself is not stored,
#' because it holds the consensus matrices of every configuration. Keep it
#' when you need those, and save it with [saveRDS()].
#'
#' @param object A SingleCellExperiment or a Seurat object.
#' @param ... Arguments of [carve()], such as `n_clusters`, `resolution` or
#'   `n_resamples`.
#' @param assay,reduction,n_dims Which data to cluster; see [carve()]. They
#'   are recorded, so the scatter plots can find the same data.
#' @param key Name of the label column and of the record, and prefix of the
#'   score columns.
#' @param measure,rule,not_two,k,sweep_value,consensus_k Which configuration
#'   to store and how to cut its consensus matrix; see [get_labels()].
#' @param reference_key Name of a cell metadata column with reference
#'   labels, which the stored labels are renamed to match; see
#'   `reference_labels` in [carve()].
#' @param store_consensus Store the consensus matrix in the record.
#' @param store_results Store the results tables in the record.
#' @param mode The `mode` of [carve()]. For `attach_results()`, the mode the
#'   fit was run with: under `"generalizability"` the labels and the stored
#'   matrix come from the consensus of the held-out predictions.
#' @param random_state Seed of the run.
#' @param fit A [CARVE-class] fit on the cells of `object`, in the same
#'   order.
#' @return The object, with the results added.
#' @seealso [carve()], [get_labels()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' sce <- run_carve(sce, n_resamples = 10, estimator_param_grids = grid)
#' table(sce$carve)
#' names(S4Vectors::metadata(sce)$carve)
#'
#' fit <- carve(sce, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' sce <- attach_results(sce, fit, key = "carve_k3", k = 3)
#' table(sce$carve_k3)
#' @rdname run_carve
#' @export
run_carve <- function(object, ..., assay = NULL, reduction = NULL, n_dims = NULL, key = "carve",
                      measure = "stability", rule = "1se", not_two = FALSE, k = NULL,
                      sweep_value = NULL, consensus_k = NULL, reference_key = NULL,
                      store_consensus = TRUE, store_results = TRUE, mode = "default",
                      random_state = 0) {
  reference_labels <- NULL
  if (!is.null(reference_key)) {
    reference_labels <- cells_get_column(object, reference_key)
    if (is.null(reference_labels)) {
      stop(sprintf(
        "reference_key=%s not found in %s. Available columns: %s",
        format_repr(reference_key),
        cells_columns_where(object),
        python_list(sort(cells_column_names(object)))
      ), call. = FALSE)
    }
  }
  X <- cells_matrix(object, assay = assay, reduction = reduction, n_dims = n_dims)
  fit <- carve(X, ..., reference_labels = reference_labels, mode = mode, random_state = random_state)
  attach_results(
    object, fit,
    key = key, measure = measure, rule = rule, not_two = not_two, k = k,
    sweep_value = sweep_value, consensus_k = consensus_k, store_consensus = store_consensus,
    store_results = store_results, assay = assay, reduction = reduction, n_dims = n_dims,
    mode = mode
  )
}

#' @rdname run_carve
#' @export
attach_results <- function(object, fit, key = "carve", measure = "stability", rule = "1se",
                           not_two = FALSE, k = NULL, sweep_value = NULL, consensus_k = NULL,
                           store_consensus = TRUE, store_results = TRUE, assay = NULL,
                           reduction = NULL, n_dims = NULL, mode = "default") {
  cells_kind(object)
  if (!methods::is(fit, "CARVE")) {
    stop("fit must be a CARVE object from carve().", call. = FALSE)
  }
  selected <- select_row(fit, measure, rule, not_two = not_two, k = k, sweep_value = sweep_value)
  # A generalizability-only run builds no stability consensus, so the cut
  # and the stored matrix both come from the held-out consensus, as in
  # get_labels().
  label_mode <- if (identical(mode, "generalizability")) "generalizability" else "default"
  labels <- get_labels(
    fit, measure = measure, rule = rule, k = k, sweep_value = sweep_value,
    consensus_k = consensus_k, not_two = not_two, mode = label_mode
  )
  n_cells <- ncol(object)
  check_cell_count(labels, n_cells, "consensus labels")

  columns <- list()
  columns[[key]] <- factor(as.character(labels), levels = as.character(sort(unique(labels))))
  id <- as.character(selected$config_id)
  # A criterion the run skipped has no scores, and its column is not
  # written.
  scores <- list(
    stability = fit@stability_gini_scores[[id]],
    stability_ce = fit@stability_ce_scores[[id]],
    generalizability = fit@generalizability_scores[[id]]
  )
  for (suffix in names(scores)) {
    values <- scores[[suffix]]
    if (is.null(values)) {
      next
    }
    check_cell_count(values, n_cells, paste(suffix, "scores"))
    columns[[paste0(key, "_", suffix)]] <- as.numeric(values)
  }

  where <- cells_record_where(object, key)
  consensus <- if (label_mode == "generalizability") {
    fit@consensus_generalizability_matrices[[id]]
  } else {
    fit@consensus_matrices[[id]]
  }
  stored <- NULL
  if (isTRUE(store_consensus) && !is.null(consensus)) {
    anchors <- fit@consensus_anchors
    m <- length(anchors)
    if (!is.null(anchors) && identical(dim(consensus), c(m, m))) {
      warning(sprintf(
        "Anchored consensus is active: the consensus matrix for '%s' is a %dx%d anchor block, not a cell-by-cell matrix. %s holds only the latter, so nothing was written to %s$consensus. The block itself is still available with consensus_matrix(fit, config_id = %d%s), and the anchor count is recorded in %s$params$n_consensus_anchors.",
        key, m, m, where, where, selected$config_id,
        if (label_mode == "generalizability") ", type = 'generalizability'" else "",
        where
      ), call. = FALSE)
    } else if (!identical(dim(consensus), c(n_cells, n_cells))) {
      stop(sprintf(
        "Consensus matrix has shape (%d, %d), but object has %d observations. The model was fitted on different data.",
        nrow(consensus), ncol(consensus), n_cells
      ), call. = FALSE)
    } else {
      if (n_cells > CONSENSUS_WARN_CELLS) {
        warning(sprintf(
          "Storing a dense %dx%d consensus matrix in %s$consensus (~%.1f GB). Pass store_consensus=FALSE if you do not need it.",
          n_cells, n_cells, where, 8 * n_cells^2 / 1e9
        ), call. = FALSE)
      }
      stored <- consensus
    }
  }

  record <- list(params = record_params(
    fit, selected, measure, rule, not_two, consensus_k, assay, reduction, n_dims, mode, n_cells
  ))
  if (isTRUE(store_results)) {
    record$results <- fit@estimator_results
    if (!is.null(fit@preprocessing_results)) {
      record$preprocessing_results <- fit@preprocessing_results
    }
  }
  if (!is.null(stored)) {
    record$consensus <- stored
  }
  object <- cells_set_columns(object, columns)
  cells_set_record(object, key, record)
}
```

- [ ] Step 5: Run the tests

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "tl|carve")'
Rscript -e 'devtools::test()'
```

Expected: `document()` exports `run_carve` and `attach_results`, adds `'tl.R'` to Collate, and writes `man/run_carve.Rd` and the changed `man/CARVE-class.Rd`. All tests pass, `SKIP 0`. Run the examples of the new page once:

```bash
Rscript -e 'devtools::run_examples(run_donttest = TRUE, document = FALSE, start = "run_carve")' 2>&1 | head -40
```

Expected: the tables print without a warning or an error. (`start` begins at that page and runs the ones after it; stop reading after `run_carve`.)

- [ ] Step 6: Commit

```bash
git add carve-r/NAMESPACE carve-r/DESCRIPTION carve-r/R/tl.R carve-r/R/carve.R carve-r/R/AllClasses.R carve-r/man/run_carve.Rd carve-r/man/CARVE-class.Rd carve-r/tests/testthat/test-tl.R carve-r/tests/testthat/test-carve.R
git commit -m "$(cat <<'EOF'
feat(carve-r): store the selected configuration in a single-cell object

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 7: Mutations

Apply each, run `Rscript -e 'devtools::test(filter = "tl")'`, confirm the named test fails, and restore with `git checkout -- R/tl.R`:

1. In `attach_results()`, `label_mode <- "default"`: fails "a mode skips its columns" (the generalizability run stores the stability matrix, and `get_labels()` errors on the missing stability consensus).
2. In `attach_results()`, write the Gini scores to `_stability_ce` and the cross-entropy scores to `_stability`: fails "the stored labels, scores, table and matrix are the fit's".
3. In `record_params()`, drop the `NULL` filter: fails "run_carve writes the labels, the scores and the record".
4. In `attach_results()`, merge into the old record (`utils::modifyList(cells_get_record(object, key), record)` when there is one): fails "a call replaces the record".
5. In `attach_results()`, drop the anchored branch: fails "an anchored run records its anchor count" (the shape check stops the call instead).
6. In `run_carve()`, do not pass `reference_labels`: fails "reference_key names a column".
7. In `attach_results()`, `levels = sort(unique(as.character(labels)))`: fails "label levels are in numeric order".
8. In `cells_record_where()`, return `"metadata(object)$<key>"` for both classes: fails "writes the same results into a Seurat object".

---

### Task 10: The sweep plots and the consensus heatmap, for fits and single-cell objects

Files:
- Modify: `carve-r/R/AllGenerics.R` (four generics)
- Create: `carve-r/R/plots.R`
- Modify: `carve-r/tests/testthat/helper-plots.R` (`expect_args()`, `expect_same_plot()`)
- Create: `carve-r/tests/testthat/test-plots.R`
- Generated: `carve-r/man/plot_metric_over_n_clusters.Rd`, `carve-r/man/plot_metric_by_pipeline.Rd`, `carve-r/man/plot_n_clusters_over_sweep.Rd`, `carve-r/man/plot_consensus_matrix.Rd`, `carve-r/NAMESPACE`, `carve-r/DESCRIPTION` (Collate)

Interfaces:
- Consumes: `metric_over_sweep_plot()`, `metric_by_pipeline_plot()`, `n_clusters_over_sweep_plot()` (Task 3); `consensus_plot()` (Task 4); `MEASURE_MAP` (`selection.R`); `select_row()`, `get_labels()` (`accessors.R`); `is_seurat()` (Task 8); `cells_get_record()`, `cells_get_column()`, `cells_record_where()`, `cells_column_where()` (Task 9); `require_package()`, `check_dots()`, `format_repr()` (`utils.R`); the test helpers `make_sce()`, `make_seurat()`, `muffle()` (Tasks 7, 8), `raster_layers()`, `raster_colours()` (Task 4).
- Produces:
  - Exported generics `plot_metric_over_n_clusters(object, ...)`, `plot_metric_by_pipeline(object, ...)`, `plot_n_clusters_over_sweep(object, ...)` and `plot_consensus_matrix(object, ...)`, each with a method for `CARVE`, one for `SingleCellExperiment`, and one for `ANY` that takes Seurat objects and rejects everything else.
  - Internal, in `R/plots.R`: `labels_mode(mode)` returns `"default"` or `"generalizability"`; `check_seurat_object(object)`; `cells_entry(object, key)` returns the stored record or stops; `cells_labels(object, key)` returns the stored labels as integer codes; `cells_results(object, key)` and `cells_preprocessing_results(object, key)` return the stored tables or stop; `recorded(value, params, name, default)`; and `cells_plot_metric_over_n_clusters()`, `cells_plot_metric_by_pipeline()`, `cells_plot_n_clusters_over_sweep()`, `cells_plot_consensus_matrix()`, the SingleCellExperiment methods, whose formals are the documented arguments.
  - Test helpers (`helper-plots.R`): `expect_args(args, expected)` and `expect_same_plot(object, expected)`.
  - Task 11 appends four more generics to `AllGenerics.R`, a section to `plots.R` and tests to `test-plots.R`, and reuses the fits at the top of `test-plots.R`.

The methods follow the two Python sources. A `CARVE` method ports the plot method of `api.CARVE`: it selects a configuration and passes its results to the drawing. A `SingleCellExperiment` method ports the `carve.pl` function: it reads the record `run_carve()` or `attach_results()` wrote and selects nothing again, because the stored scores belong to the configuration selected then. Its `measure`, `rule` and `not_two` default to `NULL`, which means the recorded value. A Seurat object reaches the `ANY` method, which hands it to the same function as a SingleCellExperiment; the `cells_*` functions of Task 9 tell the two classes apart.

The help pages of this task link only to pages that exist when it is committed. Task 11's pages link back to these.

- [ ] Step 1: Add the comparison helpers to `tests/testthat/helper-plots.R`

```r
# The arguments a mocked drawing function received, by name. The unnamed
# ones, the data, are checked by position in the test.
expect_args <- function(args, expected) {
  named <- args[names(args) != ""]
  expect_setequal(names(named), names(expected))
  expect_identical(named[names(expected)], expected)
}

# Two plots that draw the same marks with the same text: the data of every
# layer, the colors of the annotation_raster() layers, and the labels. Scale
# names, such as legend titles, are not compared.
expect_same_plot <- function(object, expected) {
  expect_equal(ggplot2::ggplot_build(object)$data, ggplot2::ggplot_build(expected)$data)
  expect_identical(
    lapply(raster_layers(object), raster_colours),
    lapply(raster_layers(expected), raster_colours)
  )
  expect_identical(object$labels, expected$labels)
}
```

Both work in ggplot2 4.0.3: `ggplot_build(plot)$data` is the list of layer data, and the `labels` of two plots built the same way are identical (checked while writing this plan).

- [ ] Step 2: Write the failing tests

Create `tests/testthat/test-plots.R`:

```r
# Fits shared by the plot tests. The k-axis fits have three configurations,
# KMeans at k = 2, 3 and 4.
blobs <- make_blobs()
k_grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
fit_blobs <- function(...) {
  carve(blobs$X, n_resamples = 6, estimator_param_grids = k_grid, random_state = 0, ...)
}
fit <- fit_blobs()
stability_fit <- muffle(fit_blobs(mode = "stability"), "Non-default mode is experimental")
anchored <- muffle(fit_blobs(anchor_threshold = 30), "anchored consensus")
randomized_fit <- fit_blobs(
  randomize_preprocessing = TRUE,
  normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
  dim_reduction_options = list(preprocessing_option(Identity))
)
resolution_fit <- carve(
  blobs$X, n_resamples = 4, random_state = 0,
  estimator_param_grids = list(estimator_grid(LeidenClustering, resolution = c(0.5, 1)))
)
sce <- make_sce(blobs$X)
written <- attach_results(sce, fit)
# Values that differ from every default and from each other, so a dropped
# or swapped argument shows.
style <- list(title = "T", xlabel = "X", ylabel = "Y", legend = FALSE, legend_loc = "bottom", palette = "viridis")

test_that("labels_mode maps a plot's mode to a get_labels() mode", {
  expect_identical(labels_mode("default"), "default")
  expect_identical(labels_mode("stability"), "default")
  expect_identical(labels_mode("generalizability"), "generalizability")
  expect_error(labels_mode("nope"), "mode must be one of: 'default', 'stability', 'generalizability'.", fixed = TRUE)
})

test_that("recorded() takes the argument, then the record, then the default", {
  params <- list(rule = "max", not_two = FALSE)
  expect_identical(recorded("quantile", params, "rule", "1se"), "quantile")
  expect_identical(recorded(NULL, params, "rule", "1se"), "max")
  expect_identical(recorded(NULL, params, "measure", "stability"), "stability")
  expect_false(recorded(NULL, params, "not_two", TRUE))
})

test_that("the fit's line plots pass every argument to the drawing", {
  local_mocked_bindings(
    metric_over_sweep_plot = function(...) list(...),
    n_clusters_over_sweep_plot = function(...) list(...),
    metric_by_pipeline_plot = function(...) list(...)
  )
  selection <- list(measure = "generalizability", rule = "max", not_two = TRUE)
  over <- do.call(plot_metric_over_n_clusters, c(list(fit), selection, style))
  expect_identical(over[[1L]], estimator_results(fit))
  expect_args(over, c(selection, style))
  counts <- do.call(plot_n_clusters_over_sweep, c(list(resolution_fit), selection, style))
  expect_identical(counts[[1L]], estimator_results(resolution_fit))
  expect_args(counts, c(selection, style))
  pipelines <- do.call(plot_metric_by_pipeline, c(list(randomized_fit, method_id = "m9"), selection, style))
  expect_identical(pipelines[[1L]], preprocessing_results(randomized_fit))
  expect_identical(pipelines[[2L]], estimator_results(randomized_fit))
  expect_args(pipelines, c(list(method_id = "m9"), selection, style))
  expect_error(plot_metric_over_n_clusters(fit, key = "carve"), "Unknown argument: key.", fixed = TRUE)
})

test_that("plot_metric_by_pipeline draws the configuration the selection picks", {
  seen <- NULL
  local_mocked_bindings(
    select_row = function(fit, measure, rule, not_two = FALSE, k = NULL, sweep_value = NULL) {
      seen <<- list(measure, rule, not_two)
      list(row = data.frame(method_id = "m7"))
    },
    metric_by_pipeline_plot = function(...) list(...)
  )
  args <- plot_metric_by_pipeline(randomized_fit, measure = "generalizability", rule = "max", not_two = TRUE)
  expect_identical(args$method_id, "m7")
  expect_identical(seen, list("generalizability", "max", TRUE))
})

test_that("the fit's heatmap passes the selection and the style through", {
  calls <- list()
  local_mocked_bindings(
    select_row = function(fit, measure, rule, not_two = FALSE, k = NULL, sweep_value = NULL) {
      calls$select <<- list(measure = measure, rule = rule, not_two = not_two, k = k, sweep_value = sweep_value)
      list(row = estimator_results(fit)[3L, ], config_id = 2L, n_clusters = 7L, pinned = TRUE)
    },
    get_labels = function(fit, ...) {
      calls$labels <<- list(...)
      rep(1:3, each = 30L)
    },
    consensus_plot = function(...) list(...)
  )
  heatmap_style <- list(cmap = "magma", palette = "Set 1", colorbar = FALSE, colorbar_label = "Share", title = "T")
  args <- do.call(plot_consensus_matrix, c(
    list(fit, measure = "generalizability", rule = "max", not_two = TRUE, sweep_value = 4),
    heatmap_style
  ))
  expect_identical(calls$select, list(measure = "generalizability", rule = "max", not_two = TRUE, k = NULL, sweep_value = 4))
  expect_identical(
    calls$labels[c("measure", "rule", "not_two", "k", "sweep_value", "consensus_k", "mode")],
    list(measure = "generalizability", rule = "max", not_two = TRUE, k = NULL, sweep_value = 4, consensus_k = 7L, mode = "default")
  )
  expect_identical(args[[1L]], fit@consensus_matrices[["2"]])
  expect_identical(args[[2L]], rep(1:3, each = 30L))
  expect_args(args, heatmap_style)
})

test_that("the fit's heatmap draws the selected matrix with the labels of its cut", {
  id <- select_row(fit, "stability", "1se", k = 4)$config_id
  expect_same_plot(
    plot_consensus_matrix(fit, k = 4),
    consensus_plot(consensus_matrix(fit, id), get_labels(fit, k = 4))
  )
})

test_that("mode chooses the consensus the heatmap draws", {
  id <- select_row(fit, "stability", "1se")$config_id
  expect_same_plot(
    plot_consensus_matrix(fit, mode = "generalizability"),
    consensus_plot(consensus_matrix(fit, id, type = "generalizability"), get_labels(fit, mode = "generalizability"))
  )
  expect_same_plot(plot_consensus_matrix(fit, mode = "stability"), plot_consensus_matrix(fit))
  expect_error(plot_consensus_matrix(fit, mode = "nope"), "mode must be one of: 'default', 'stability', 'generalizability'.", fixed = TRUE)
  expect_error(
    plot_consensus_matrix(stability_fit, mode = "generalizability"),
    "Selected consensus matrix is not available for mode='generalizability'.",
    fixed = TRUE
  )
})

test_that("an anchored fit draws the anchor block with the anchors' labels", {
  plot <- plot_consensus_matrix(anchored)
  expect_identical(dim(raster_colours(raster_layers(plot)[[1L]])), c(30L, 30L))
  id <- select_row(anchored, "stability", "1se")$config_id
  expect_same_plot(
    plot,
    consensus_plot(consensus_matrix(anchored, id), get_labels(anchored)[consensus_anchors(anchored)])
  )
})

test_that("the single-cell line plots default to the recorded selection", {
  local_mocked_bindings(
    metric_over_sweep_plot = function(...) list(...),
    n_clusters_over_sweep_plot = function(...) list(...),
    metric_by_pipeline_plot = function(...) list(...)
  )
  stored <- attach_results(sce, randomized_fit, measure = "generalizability", rule = "max", not_two = TRUE)
  record <- S4Vectors::metadata(stored)$carve
  from_record <- list(measure = "generalizability", rule = "max", not_two = TRUE)
  defaults <- list(title = NULL, xlabel = NULL, ylabel = NULL, legend = TRUE, legend_loc = "right", palette = "Accent")
  over <- plot_metric_over_n_clusters(stored)
  expect_identical(over[[1L]], record$results)
  expect_args(over, c(from_record, defaults))
  expect_args(plot_n_clusters_over_sweep(stored), c(from_record, defaults))
  pipelines <- plot_metric_by_pipeline(stored)
  expect_identical(pipelines[[1L]], record$preprocessing_results)
  expect_identical(pipelines[[2L]], record$results)
  expect_args(pipelines, c(list(method_id = record$params$selected_method_id), from_record, defaults))
  # An argument given, even FALSE, wins over the record.
  chosen <- list(measure = "average", rule = "quantile", not_two = FALSE)
  expect_args(do.call(plot_metric_over_n_clusters, c(list(stored), chosen, style)), c(chosen, style))
  expect_args(do.call(plot_n_clusters_over_sweep, c(list(stored), chosen, style)), c(chosen, style))
  expect_args(
    do.call(plot_metric_by_pipeline, c(list(stored, method_id = "m9"), chosen, style)),
    c(list(method_id = "m9"), chosen, style)
  )
})

test_that("a single-cell plot_metric_by_pipeline plots stability when the recorded measure is not in the table", {
  pac <- attach_results(sce, randomized_fit, measure = "pac", rule = "max")
  method_id <- S4Vectors::metadata(pac)$carve$params$selected_method_id
  expect_warning(
    plot <- plot_metric_by_pipeline(pac),
    "run_carve recorded measure 'pac', which the per-pipeline table does not carry; plotting 'stability'. Pass measure='generalizability' for the other ARI criterion.",
    fixed = TRUE
  )
  expect_same_plot(plot, plot_metric_by_pipeline(randomized_fit, method_id = method_id, rule = "max"))
  expect_error(plot_metric_by_pipeline(pac, measure = "pac"), "The per-pipeline table carries only the ARI criteria", fixed = TRUE)
})

test_that("a SingleCellExperiment draws what the fit draws", {
  expect_same_plot(plot_metric_over_n_clusters(written), plot_metric_over_n_clusters(fit))
  expect_same_plot(plot_consensus_matrix(written), plot_consensus_matrix(fit))
  expect_same_plot(
    plot_n_clusters_over_sweep(attach_results(sce, resolution_fit)),
    plot_n_clusters_over_sweep(resolution_fit)
  )
  expect_same_plot(
    plot_metric_by_pipeline(attach_results(sce, randomized_fit)),
    plot_metric_by_pipeline(randomized_fit)
  )
  expect_error(plot_consensus_matrix(written, nope = 1), "Unknown argument: nope.", fixed = TRUE)
})

test_that("a Seurat object draws what the fit draws", {
  object <- attach_results(make_seurat(blobs$X), fit)
  expect_same_plot(plot_metric_over_n_clusters(object), plot_metric_over_n_clusters(fit))
  expect_same_plot(plot_consensus_matrix(object), plot_consensus_matrix(fit))
  expect_error(
    plot_metric_over_n_clusters(object, key = "other"),
    "Misc(object, 'other') not found. Run `run_carve(object, key = 'other')` first, or pass the key you used.",
    fixed = TRUE
  )
  expect_error(plot_consensus_matrix(object, nope = 1), "Unknown argument: nope.", fixed = TRUE)
})

test_that("stored labels are read as their factor codes", {
  relabeled <- written
  SummarizedExperiment::colData(relabeled)$carve <- factor(rep(c("b", "a", "c"), each = 30L))
  expect_identical(cells_labels(relabeled, "carve"), rep(c(2L, 1L, 3L), each = 30L))
  SummarizedExperiment::colData(relabeled)$carve <- rep(c(3, 1, 2), each = 30L)
  expect_identical(cells_labels(relabeled, "carve"), rep(c(3L, 1L, 2L), each = 30L))
})

test_that("the single-cell plots name what is missing from the object", {
  expect_error(
    plot_metric_over_n_clusters(sce),
    "metadata(object)$carve not found. Run `run_carve(object, key = 'carve')` first, or pass the key you used.",
    fixed = TRUE
  )
  no_params <- sce
  S4Vectors::metadata(no_params)$carve <- list(results = estimator_results(fit))
  expect_error(
    plot_metric_over_n_clusters(no_params),
    "metadata(object)$carve does not look like a CARVE result; expected a list with a 'params' entry.",
    fixed = TRUE
  )
  expect_error(
    plot_n_clusters_over_sweep(attach_results(sce, fit, store_results = FALSE)),
    "metadata(object)$carve$results not found. It is omitted when run_carve runs with store_results=FALSE.",
    fixed = TRUE
  )
  expect_error(
    plot_metric_by_pipeline(written),
    "metadata(object)$carve$preprocessing_results not found. It is written only for a randomized fit, so either run_carve ran without randomize_preprocessing=TRUE or it ran with store_results=FALSE.",
    fixed = TRUE
  )
  expect_error(
    plot_consensus_matrix(attach_results(sce, fit, store_consensus = FALSE)),
    "metadata(object)$carve$consensus not found. It is omitted when run_carve runs with store_consensus=FALSE, or when anchored consensus was active: an anchor block is m-by-m rather than n_obs-by-n_obs and metadata(object)$carve holds only the latter.",
    fixed = TRUE
  )
  unlabeled <- written
  SummarizedExperiment::colData(unlabeled)$carve <- NULL
  expect_error(
    plot_consensus_matrix(unlabeled),
    "colData(object)$carve not found. Run `run_carve(object, key = 'carve')` first.",
    fixed = TRUE
  )
})

test_that("a plot of anything else names the classes the plots take", {
  expect_error(
    plot_metric_over_n_clusters(blobs$X),
    "object must be a CARVE fit, a SingleCellExperiment or a Seurat object, not an object of class 'matrix'.",
    fixed = TRUE
  )
  expect_error(plot_consensus_matrix(list()), "not an object of class 'list'.", fixed = TRUE)
})
```

Three notes on these tests. The mocks replace the drawing functions with `function(...) list(...)`, so a method returns the arguments it passed; `expect_args()` compares the named ones, and the test checks the unnamed ones, the data, by position. The heatmap's wiring test also mocks `select_row()` and `get_labels()`, so it can name a selection (`config_id` 2, 7 clusters, pinned) that no real call returns and check that each reaches the right place. The fallback test records `rule = "max"` because `"pac"` has no standard error, and `"1se"` would fall back to `"max"` with a warning when `attach_results()` selects.

- [ ] Step 3: Run the tests to see them fail

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript -e 'devtools::test(filter = "plots")'
```

Expected: FAIL, `could not find function "labels_mode"`, `could not find function "plot_metric_over_n_clusters"` and the like.

- [ ] Step 4: Add the four generics to the end of `R/AllGenerics.R`

```r
#' @rdname plot_metric_over_n_clusters
#' @export
setGeneric("plot_metric_over_n_clusters", function(object, ...) {
  standardGeneric("plot_metric_over_n_clusters")
})

#' @rdname plot_metric_by_pipeline
#' @export
setGeneric("plot_metric_by_pipeline", function(object, ...) standardGeneric("plot_metric_by_pipeline"))

#' @rdname plot_n_clusters_over_sweep
#' @export
setGeneric("plot_n_clusters_over_sweep", function(object, ...) {
  standardGeneric("plot_n_clusters_over_sweep")
})

#' @rdname plot_consensus_matrix
#' @export
setGeneric("plot_consensus_matrix", function(object, ...) standardGeneric("plot_consensus_matrix"))
```

- [ ] Step 5: Create `R/plots.R`

Each `SingleCellExperiment` method is a named function registered with `setMethod()`. roxygen2 writes the method's usage from that function's formals (checked while writing this plan on a scratch package: `setMethod("plot_x", "character", cells_plot_x)` gives `\S4method{plot_x}{character}(object, key = "carve", title = NULL, ...)`). The `ANY` method calls the same function after `check_seurat_object()`.

```r
#' @include AllGenerics.R
NULL

# The plot methods. A CARVE method mirrors the plot method of api.CARVE of
# the same name: it selects a configuration and passes its results to the
# drawing in plotting.R. A SingleCellExperiment method mirrors the carve.pl
# function: it reads what run_carve() or attach_results() stored and selects
# nothing again, because the stored scores belong to the configuration
# selected then. Seurat objects reach the ANY methods, which pass them to
# the same functions as SingleCellExperiment objects.

# The plot methods of a fit take three modes; "stability" cuts the same
# matrix as "default".
labels_mode <- function(mode) {
  if (identical(mode, "default") || identical(mode, "stability")) {
    return("default")
  }
  if (identical(mode, "generalizability")) {
    return("generalizability")
  }
  stop("mode must be one of: 'default', 'stability', 'generalizability'.", call. = FALSE)
}

check_seurat_object <- function(object) {
  if (!is_seurat(object)) {
    stop(sprintf(
      "object must be a CARVE fit, a SingleCellExperiment or a Seurat object, not an object of class '%s'.",
      class(object)[[1L]]
    ), call. = FALSE)
  }
  require_package("SeuratObject", "Seurat objects")
}

# The record run_carve() stored under key.
cells_entry <- function(object, key) {
  record <- cells_get_record(object, key)
  where <- cells_record_where(object, key)
  if (is.null(record)) {
    stop(sprintf(
      "%s not found. Run `run_carve(object, key = %s)` first, or pass the key you used.",
      where, format_repr(key)
    ), call. = FALSE)
  }
  if (!is.list(record) || !"params" %in% names(record)) {
    stop(sprintf(
      "%s does not look like a CARVE result; expected a list with a 'params' entry.",
      where
    ), call. = FALSE)
  }
  record
}

# The stored labels as integers. A factor gives its codes, 1, 2, ..., as
# Python reads the codes of a categorical column.
cells_labels <- function(object, key) {
  column <- cells_get_column(object, key)
  if (is.null(column)) {
    stop(sprintf(
      "%s not found. Run `run_carve(object, key = %s)` first.",
      cells_column_where(object, key), format_repr(key)
    ), call. = FALSE)
  }
  as.integer(column)
}

cells_results <- function(object, key) {
  results <- cells_entry(object, key)[["results"]]
  if (is.null(results)) {
    stop(sprintf(
      "%s$results not found. It is omitted when run_carve runs with store_results=FALSE.",
      cells_record_where(object, key)
    ), call. = FALSE)
  }
  results
}

cells_preprocessing_results <- function(object, key) {
  results <- cells_entry(object, key)[["preprocessing_results"]]
  if (is.null(results)) {
    stop(sprintf(
      "%s$preprocessing_results not found. It is written only for a randomized fit, so either run_carve ran without randomize_preprocessing=TRUE or it ran with store_results=FALSE.",
      cells_record_where(object, key)
    ), call. = FALSE)
  }
  results
}

# A single-cell plot argument left NULL takes the value run_carve()
# recorded, or Python's default when the record lacks it.
recorded <- function(value, params, name, default) {
  if (!is.null(value)) {
    return(value)
  }
  if (is.null(params[[name]])) default else params[[name]]
}

#' Plot a criterion over the swept parameter
#'
#' Draws the value of `measure` at each value of the swept parameter, one
#' line per estimator configuration. A dashed line marks the value `rule`
#' selects.
#'
#' For a fit, the plot reads [estimator_results()]. For a SingleCellExperiment
#' or a Seurat object, it reads the results table [run_carve()] or
#' [attach_results()] stored under `key`, and `measure`, `rule` and `not_two`
#' default to the values recorded there, so the dashed line marks the
#' configuration whose labels are stored.
#'
#' The three ARI criteria get error bars of one standard error; the other
#' criteria have no standard errors. On a run over `resolution` or
#' `min_cluster_size`, the x axis is that parameter.
#'
#' @param object A [CARVE-class] fit, or a SingleCellExperiment or Seurat
#'   object with results stored by [run_carve()] or [attach_results()].
#' @param measure The criterion to plot and select on, as in [get_k()]. For a
#'   single-cell object, `NULL` uses the recorded one.
#' @param rule The rule that selects the value the dashed line marks:
#'   `"max"`, `"1se"` or `"quantile"`; see [get_k()]. For a single-cell
#'   object, `NULL` uses the recorded one.
#' @param not_two Leave out configurations with two clusters when selecting.
#'   For a single-cell object, `NULL` uses the recorded value.
#' @param title,xlabel,ylabel Title and axis labels. By default there is no
#'   title, and the axes are named after the swept parameter and the
#'   criterion.
#' @param legend Show the legends.
#' @param legend_loc Where the legends go: `"right"`, `"left"`, `"top"`,
#'   `"bottom"`, or two numbers between 0 and 1 that place them inside the
#'   panel.
#' @param palette Line colors: `"Accent"`, another name from
#'   [grDevices::palette.pals()], a viridis option such as `"viridis"` or
#'   `"magma"`, or a vector of colors.
#' @param key For a single-cell object, the `key` the results were stored
#'   under.
#' @param ... For a Seurat object, the arguments of the SingleCellExperiment
#'   method. For a fit or a SingleCellExperiment, not used: an argument name
#'   the method does not know is an error.
#' @return A ggplot object.
#' @seealso [get_k()], [plot_n_clusters_over_sweep()], [run_carve()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_metric_over_n_clusters(fit)
#' plot_metric_over_n_clusters(fit, measure = "generalizability", rule = "max")
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_metric_over_n_clusters(sce)
#' @rdname plot_metric_over_n_clusters
#' @export
setMethod("plot_metric_over_n_clusters", "CARVE", function(object, measure = "stability",
                                                           rule = "1se", not_two = FALSE,
                                                           title = NULL, xlabel = NULL,
                                                           ylabel = NULL, legend = TRUE,
                                                           legend_loc = "right",
                                                           palette = "Accent", ...) {
  check_dots(...)
  metric_over_sweep_plot(
    object@estimator_results,
    measure = measure, rule = rule, not_two = not_two, title = title, xlabel = xlabel,
    ylabel = ylabel, legend = legend, legend_loc = legend_loc, palette = palette
  )
})

cells_plot_metric_over_n_clusters <- function(object, key = "carve", measure = NULL, rule = NULL,
                                              not_two = NULL, title = NULL, xlabel = NULL,
                                              ylabel = NULL, legend = TRUE, legend_loc = "right",
                                              palette = "Accent", ...) {
  check_dots(...)
  params <- cells_entry(object, key)$params
  metric_over_sweep_plot(
    cells_results(object, key),
    measure = recorded(measure, params, "measure", "stability"),
    rule = recorded(rule, params, "rule", "1se"),
    not_two = recorded(not_two, params, "not_two", FALSE),
    title = title, xlabel = xlabel, ylabel = ylabel, legend = legend, legend_loc = legend_loc,
    palette = palette
  )
}

#' @rdname plot_metric_over_n_clusters
#' @export
setMethod("plot_metric_over_n_clusters", "SingleCellExperiment", cells_plot_metric_over_n_clusters)

#' @rdname plot_metric_over_n_clusters
#' @export
setMethod("plot_metric_over_n_clusters", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_metric_over_n_clusters(object, ...)
})

#' Plot a criterion for each preprocessing pipeline
#'
#' For a run with `randomize_preprocessing = TRUE`: draws the rows of
#' [preprocessing_results()] for one configuration, one line per pipeline,
#' with error bars of one standard error over the resamples each pipeline
#' received. The dashed line marks the value CARVE selects from
#' [estimator_results()], pooled over the pipelines, so the lines that cross
#' it compare the pipelines at the selected configuration. For a `method_id`
#' that CARVE did not select, it marks the value `rule` picks among that
#' configuration's rows.
#'
#' For a SingleCellExperiment or a Seurat object, the plot reads the two
#' tables [run_carve()] or [attach_results()] stored under `key`; they are
#' stored for a randomized run unless `store_results = FALSE`. `method_id`,
#' `measure`, `rule` and `not_two` default to the recorded values. A recorded
#' measure that the per-pipeline table does not have is replaced by
#' `"stability"`, with a warning.
#'
#' @inheritParams plot_metric_over_n_clusters
#' @param method_id The configuration to draw, a value of the `method_id`
#'   column. `NULL` takes the configuration CARVE selects under `measure`,
#'   `rule` and `not_two`, or for a single-cell object the recorded one.
#' @param measure `"stability"` or `"generalizability"`, or an alias of
#'   either; the per-pipeline table has only these two criteria. For a
#'   single-cell object, `NULL` uses the recorded measure.
#' @return A ggplot object.
#' @seealso [preprocessing_results()], [plot_metric_over_n_clusters()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' fit <- carve(
#'   X, n_resamples = 20, random_state = 0, randomize_preprocessing = TRUE,
#'   estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)),
#'   normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
#'   dim_reduction_options = list(preprocessing_option(Identity))
#' )
#' plot_metric_by_pipeline(fit)
#' plot_metric_by_pipeline(fit, measure = "generalizability")
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_metric_by_pipeline(sce)
#' @rdname plot_metric_by_pipeline
#' @export
setMethod("plot_metric_by_pipeline", "CARVE", function(object, method_id = NULL,
                                                       measure = "stability", rule = "1se",
                                                       not_two = FALSE, title = NULL,
                                                       xlabel = NULL, ylabel = NULL, legend = TRUE,
                                                       legend_loc = "right", palette = "Accent",
                                                       ...) {
  check_dots(...)
  if (is.null(method_id)) {
    selected <- select_row(object, measure, rule, not_two = not_two)
    method_id <- as.character(selected$row$method_id[[1L]])
  }
  metric_by_pipeline_plot(
    object@preprocessing_results, object@estimator_results,
    method_id = method_id, measure = measure, rule = rule, not_two = not_two, title = title,
    xlabel = xlabel, ylabel = ylabel, legend = legend, legend_loc = legend_loc, palette = palette
  )
})

cells_plot_metric_by_pipeline <- function(object, key = "carve", method_id = NULL, measure = NULL,
                                          rule = NULL, not_two = NULL, title = NULL, xlabel = NULL,
                                          ylabel = NULL, legend = TRUE, legend_loc = "right",
                                          palette = "Accent", ...) {
  check_dots(...)
  params <- cells_entry(object, key)$params
  preprocessing <- cells_preprocessing_results(object, key)
  if (is.null(measure)) {
    measure <- recorded(NULL, params, "measure", "stability")
    if (!measure %in% names(MEASURE_MAP) || !MEASURE_MAP[[measure]] %in% names(preprocessing)) {
      warning(sprintf(
        "run_carve recorded measure %s, which the per-pipeline table does not carry; plotting 'stability'. Pass measure='generalizability' for the other ARI criterion.",
        format_repr(measure)
      ), call. = FALSE)
      measure <- "stability"
    }
  }
  metric_by_pipeline_plot(
    preprocessing, cells_results(object, key),
    method_id = recorded(method_id, params, "selected_method_id", NULL),
    measure = measure,
    rule = recorded(rule, params, "rule", "1se"),
    not_two = recorded(not_two, params, "not_two", FALSE),
    title = title, xlabel = xlabel, ylabel = ylabel, legend = legend, legend_loc = legend_loc,
    palette = palette
  )
}

#' @rdname plot_metric_by_pipeline
#' @export
setMethod("plot_metric_by_pipeline", "SingleCellExperiment", cells_plot_metric_by_pipeline)

#' @rdname plot_metric_by_pipeline
#' @export
setMethod("plot_metric_by_pipeline", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_metric_by_pipeline(object, ...)
})

#' Plot the number of clusters over the swept parameter
#'
#' For a run over `resolution`, `min_cluster_size` or another parameter that
#' does not fix the number of clusters: draws the mean number of clusters
#' observed at each value of the swept parameter, one line per estimator
#' configuration, with error bars of one standard error. The dashed line
#' marks the value selected under `measure`, `rule` and `not_two`, and its
#' legend entry gives the number of clusters there, the number [get_k()]
#' returns for the same arguments. On a run over `n_clusters` the count is
#' the swept value itself, and the function stops with an error.
#'
#' The count is the mean, over resamples, of the number of clusters in the
#' clustering of each resample's subsample, counted after the `noise_policy`
#' of [carve()]: `"drop"` leaves noise points out, `"as_cluster"` counts the
#' noise as one cluster, and `"singleton"` counts each noise point as its own
#' cluster. It is not the count of one clustering of all the samples.
#' [get_labels()] cuts the consensus matrix at the rounded count unless
#' `consensus_k` is given. The error bars show the standard error of the
#' mean, not the spread of the counts.
#'
#' For a SingleCellExperiment or a Seurat object, the plot reads the results
#' table stored under `key`, and `measure`, `rule` and `not_two` default to
#' the recorded values.
#'
#' @inheritParams plot_metric_over_n_clusters
#' @param measure The criterion that selects the marked value, as in
#'   [get_k()]. For a single-cell object, `NULL` uses the recorded one.
#' @param title,xlabel,ylabel Title and axis labels. By default there is no
#'   title, the x axis is named after the swept parameter, and the y axis is
#'   "Mean Observed Number of Clusters".
#' @return A ggplot object.
#' @seealso [get_k()], [plot_metric_over_n_clusters()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(LeidenClustering, resolution = c(0.25, 0.5, 1, 2)))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_n_clusters_over_sweep(fit)
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_n_clusters_over_sweep(sce)
#' @rdname plot_n_clusters_over_sweep
#' @export
setMethod("plot_n_clusters_over_sweep", "CARVE", function(object, measure = "stability",
                                                          rule = "1se", not_two = FALSE,
                                                          title = NULL, xlabel = NULL,
                                                          ylabel = NULL, legend = TRUE,
                                                          legend_loc = "right",
                                                          palette = "Accent", ...) {
  check_dots(...)
  n_clusters_over_sweep_plot(
    object@estimator_results,
    measure = measure, rule = rule, not_two = not_two, title = title, xlabel = xlabel,
    ylabel = ylabel, legend = legend, legend_loc = legend_loc, palette = palette
  )
})

cells_plot_n_clusters_over_sweep <- function(object, key = "carve", measure = NULL, rule = NULL,
                                             not_two = NULL, title = NULL, xlabel = NULL,
                                             ylabel = NULL, legend = TRUE, legend_loc = "right",
                                             palette = "Accent", ...) {
  check_dots(...)
  params <- cells_entry(object, key)$params
  n_clusters_over_sweep_plot(
    cells_results(object, key),
    measure = recorded(measure, params, "measure", "stability"),
    rule = recorded(rule, params, "rule", "1se"),
    not_two = recorded(not_two, params, "not_two", FALSE),
    title = title, xlabel = xlabel, ylabel = ylabel, legend = legend, legend_loc = legend_loc,
    palette = palette
  )
}

#' @rdname plot_n_clusters_over_sweep
#' @export
setMethod("plot_n_clusters_over_sweep", "SingleCellExperiment", cells_plot_n_clusters_over_sweep)

#' @rdname plot_n_clusters_over_sweep
#' @export
setMethod("plot_n_clusters_over_sweep", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_n_clusters_over_sweep(object, ...)
})

#' Plot a consensus matrix
#'
#' Draws the consensus matrix of one configuration as a heatmap, with the
#' samples ordered by cluster, a band of cluster colors along the top and
#' white lines between the clusters. Pairs of samples never drawn together
#' are shown at 0.5.
#'
#' For a fit, the plot selects a configuration with `measure`, `rule`,
#' `not_two`, `k` and `sweep_value`, as [get_labels()] does, and colors the
#' band with the labels of the cut at the selected number of clusters. On an
#' anchored run it draws the block over the anchors, with the anchors'
#' labels; a matrix over every sample would be too large to draw at the
#' sizes where anchoring applies.
#'
#' For a SingleCellExperiment or a Seurat object, the plot draws the matrix
#' stored under `key` with the stored labels. [run_carve()] and
#' [attach_results()] store the matrix unless the run was anchored or
#' `store_consensus = FALSE`.
#'
#' @inheritParams plot_metric_over_n_clusters
#' @param measure,rule,not_two For a fit, how the configuration is selected,
#'   as in [get_labels()].
#' @param mode For a fit, `"default"` or `"stability"` draws the stability
#'   consensus, and `"generalizability"` the consensus of the classifier's
#'   predictions for the held-out samples.
#' @param k,sweep_value For a fit, pin the configuration, as in
#'   [get_labels()].
#' @param cmap Colors of the heatmap: a palette name as for `palette`, such
#'   as `"viridis"` or `"Greens"`, or a vector of at least two colors.
#' @param palette Colors of the cluster band, as in
#'   [plot_metric_over_n_clusters()].
#' @param colorbar Show the color bar.
#' @param colorbar_label Title of the color bar.
#' @param title Plot title.
#' @return A ggplot object.
#' @seealso [consensus_matrix()], [get_labels()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_consensus_matrix(fit)
#' plot_consensus_matrix(fit, k = 3, cmap = "Greens")
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_consensus_matrix(sce)
#' @rdname plot_consensus_matrix
#' @export
setMethod("plot_consensus_matrix", "CARVE", function(object, measure = "stability", rule = "1se",
                                                     not_two = FALSE, mode = "default", k = NULL,
                                                     sweep_value = NULL, cmap = "viridis",
                                                     palette = "Accent", colorbar = TRUE,
                                                     colorbar_label = "Consensus", title = NULL,
                                                     ...) {
  check_dots(...)
  label_mode <- labels_mode(mode)
  selected <- select_row(object, measure, rule, not_two = not_two, k = k, sweep_value = sweep_value)
  id <- as.character(selected$config_id)
  consensus <- if (label_mode == "default") {
    object@consensus_matrices[[id]]
  } else {
    object@consensus_generalizability_matrices[[id]]
  }
  if (is.null(consensus)) {
    stop(sprintf(
      "Selected consensus matrix is not available for mode=%s.",
      format_repr(mode)
    ), call. = FALSE)
  }
  labels <- get_labels(
    object, measure = measure, rule = rule, k = k, sweep_value = sweep_value,
    consensus_k = selected$n_clusters, not_two = not_two, mode = label_mode
  )
  # On an anchored run the stored matrix is the block over the anchors, so
  # the band shows the anchors' labels.
  if (!is.null(object@consensus_anchors)) {
    labels <- labels[object@consensus_anchors]
  }
  consensus_plot(
    consensus, labels,
    cmap = cmap, palette = palette, colorbar = colorbar, colorbar_label = colorbar_label,
    title = title
  )
})

cells_plot_consensus_matrix <- function(object, key = "carve", cmap = "viridis", palette = "Accent",
                                        colorbar = TRUE, colorbar_label = "Consensus",
                                        title = NULL, ...) {
  check_dots(...)
  consensus <- cells_entry(object, key)[["consensus"]]
  if (is.null(consensus)) {
    where <- cells_record_where(object, key)
    stop(sprintf(
      "%s$consensus not found. It is omitted when run_carve runs with store_consensus=FALSE, or when anchored consensus was active: an anchor block is m-by-m rather than n_obs-by-n_obs and %s holds only the latter.",
      where, where
    ), call. = FALSE)
  }
  consensus_plot(
    consensus, cells_labels(object, key),
    cmap = cmap, palette = palette, colorbar = colorbar, colorbar_label = colorbar_label,
    title = title
  )
}

#' @rdname plot_consensus_matrix
#' @export
setMethod("plot_consensus_matrix", "SingleCellExperiment", cells_plot_consensus_matrix)

#' @rdname plot_consensus_matrix
#' @export
setMethod("plot_consensus_matrix", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_consensus_matrix(object, ...)
})
```

The fit's heatmap has no check for a missing list of matrices, which Python has: the R fit always holds a list per mode, with `NULL` entries for a skipped mode, so the check after the selection is the one that applies. In Python, `pl.consensus_matrix` looks for the matrix without reading the record first; here the matrix is part of the record, so a missing record gives `cells_entry()`'s message, which names `run_carve()`.

- [ ] Step 6: Run the tests and the examples

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "plots")'
Rscript -e 'devtools::test()'
Rscript -e 'grDevices::pdf(NULL); devtools::run_examples(run_donttest = TRUE, document = FALSE, start = "plot_consensus_matrix")' 2>&1 | head -80
```

Expected: `document()` adds `export()` and `exportMethods()` lines for the four generics to NAMESPACE, `'plots.R'` to the Collate field, and writes the four Rd pages, each with the generic and its `CARVE`, `SingleCellExperiment` and `ANY` methods in the usage. The plot tests pass, and the full suite passes with `SKIP 0`. The examples of the four pages run without a warning or an error (`start` runs the pages after it as well; stop reading after `plot_n_clusters_over_sweep`). `pdf(NULL)` keeps the examples from writing `Rplots.pdf`.

If an assertion fails, find out which side is wrong before changing either, and report it as the Global Constraints say. Two places to look first: `expect_same_plot()` comparing build data that differ only in a column ggplot2 adds per call (compare the two `ggplot_build()$data` lists with `all.equal()` to see which column), and a mocked `select_row()` or `get_labels()` that the method does not reach because it calls a different function.

- [ ] Step 7: Commit

```bash
git add carve-r/R/AllGenerics.R carve-r/R/plots.R carve-r/NAMESPACE carve-r/DESCRIPTION carve-r/man/plot_metric_over_n_clusters.Rd carve-r/man/plot_metric_by_pipeline.Rd carve-r/man/plot_n_clusters_over_sweep.Rd carve-r/man/plot_consensus_matrix.Rd carve-r/tests/testthat/helper-plots.R carve-r/tests/testthat/test-plots.R
git commit -m "$(cat <<'EOF'
feat(carve-r): plot metrics, cluster counts and consensus for fits and single-cell objects

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 8: Mutations

Apply each, run `Rscript -e 'devtools::test(filter = "plots")'`, confirm the named test fails, and restore with `git checkout -- R/plots.R`:

1. In `cells_plot_metric_over_n_clusters()`, pass `rule = rule` instead of the `recorded()` call: fails "the single-cell line plots default to the recorded selection".
2. In `recorded()`, return the recorded value whenever there is one, before looking at `value`: fails "recorded() takes the argument" and "default to the recorded selection" (the `not_two = FALSE` override).
3. In the `CARVE` method of `plot_metric_by_pipeline`, select with `select_row(object, "stability", "1se")`: fails "draws the configuration the selection picks".
4. In the `CARVE` method of `plot_consensus_matrix`, drop `consensus_k = selected$n_clusters` from the `get_labels()` call: fails "passes the selection and the style through".
5. In the same method, read `object@consensus_matrices` whatever the mode: fails "mode chooses the consensus the heatmap draws".
6. In the same method, drop the anchor indexing: fails "an anchored fit draws the anchor block" (`consensus_display()` stops on 90 labels for a 30 by 30 block).
7. In `cells_plot_metric_by_pipeline()`, drop the fallback, so a recorded `"pac"` is passed on: fails "plots stability when the recorded measure is not in the table".
8. In `cells_labels()`, return `as.integer(as.character(column))`: fails "stored labels are read as their factor codes" (the coercion warning fails the test under `warn = 2`).
9. In the `ANY` method of `plot_metric_over_n_clusters`, drop `check_seurat_object(object)`: fails "names the classes the plots take" (Task 9's `cells_kind()` message instead).

---

### Task 11: The per-sample plots, for fits and single-cell objects

Files:
- Modify: `carve-r/R/AllGenerics.R` (four generics)
- Modify: `carve-r/R/plots.R`
- Modify: `carve-r/tests/testthat/test-plots.R`
- Generated: `carve-r/man/plot_cluster_boxplot.Rd`, `carve-r/man/plot_cluster_violin.Rd`, `carve-r/man/plot_cluster_scatter.Rd`, `carve-r/man/plot_diagnostic_scatter.Rd`, `carve-r/NAMESPACE`

Interfaces:
- Consumes: `cluster_boxplot_plot()`, `cluster_violin_plot()` (Task 5); `cluster_scatter_plot()`, `diagnostic_scatter_plot()` (Task 6); `annotation_text()` (Task 2); `cells_get_column()`, `cells_column_where()`, `cells_basis()`, `cells_matrix()` (Task 9); `labels_mode()`, `check_seurat_object()`, `cells_entry()`, `cells_labels()`, `recorded()` (Task 10); `select_row()`, `get_labels()`, `sample_scores()` (`accessors.R`); the fits and the `expect_args()` and `expect_same_plot()` helpers of Task 10's tests.
- Produces:
  - Exported generics `plot_cluster_boxplot(object, ...)`, `plot_cluster_violin(object, ...)`, `plot_cluster_scatter(object, ...)` and `plot_diagnostic_scatter(object, ...)`, each with methods for `CARVE`, `SingleCellExperiment` and `ANY`.
  - Internal, in `R/plots.R`: `SOURCE_SUFFIXES`, `SOURCE_LABELS`, `SOURCE_NAMES`, `SOURCE_MODES` (named by `"gini"`, `"ce"`, `"accuracy"`); `resolve_annotation(annotation, build)`; `fit_sample_plot_inputs(fit, source, measure, rule, not_two, mode, k, sweep_value, annotation, tight_layout = FALSE)` returns `list(scores, labels, annotation)`; `cells_scores(object, key, source)`; `cells_annotation(object, key, annotation)`; `cells_scatter_data(object, key, basis)` returns the matrix the scatter plots draw from; and `cells_plot_cluster_boxplot()`, `cells_plot_cluster_violin()`, `cells_plot_cluster_scatter()`, `cells_plot_diagnostic_scatter()`.

The four plots show per-sample scores. The score names follow Python, which uses two sets: the box and violin plots of both classes and the single-cell scatter plots name a score `"Cluster Stability (Gini)"`, `"Cluster Stability (CE)"` or `"Cluster Generalizability"` (`SOURCE_LABELS`), and the scatter plots of a fit name it `"Gini Stability"`, `"CE Stability"` or `"Generalizability"` (`SOURCE_NAMES`). The scatter plots of a fit build their annotation with `tight_layout = TRUE` when it goes into the legend (`annotation_style = "legend"`), which puts each part of the method label on its own line; the single-cell scatter plots never do. Python does the same in both places.

- [ ] Step 1: Write the failing tests

Append to `tests/testthat/test-plots.R`:

```r
# Values for every drawing argument of the per-sample plots, each different
# from its default and from the others.
box_style <- list(
  order = c(3, 1, 2), palette = "Set 1", showfliers = TRUE, width = 0.5, title = "T", xlabel = "X",
  ylabel = "Y", annotation = "note", rotation = 45, ylim = c(0, 1), fit_ylim = FALSE
)
violin_style <- list(
  order = c(3, 1, 2), palette = "Set 1", density_norm = "count", stripplot = FALSE, jitter = 0.05,
  size = 20, alpha = 0.5, inner = "quartile", title = "T", xlabel = "X", ylabel = "Y",
  annotation = "note", rotation = 45, ylim = c(0, 1), fit_ylim = FALSE
)
scatter_style <- list(
  palette = "Set 1", alpha_range = c(0.2, 0.7), size_range = c(5, 50), sort_order = FALSE,
  legend = FALSE, legend_loc = "bottom", annotation = "note", annotation_style = "box", title = "T",
  xlabel = "X", ylabel = "Y", show_ticks = TRUE, frameon = TRUE
)
diagnostic_style <- list(
  cmap = "magma", alpha_encoding = FALSE, alpha_range = c(0.1, 0.6), marker_size = 50,
  marker_linewidth = 1, markers = c(22, 21, 23), sort_order = FALSE, legend = FALSE,
  legend_loc = "bottom", colorbar = FALSE, colorbar_label = "Bar", annotation = "note",
  annotation_style = "box", title = "T", xlabel = "X", ylabel = "Y", show_ticks = TRUE,
  frameon = TRUE
)

test_that("fit_sample_plot_inputs selects once and reads the scores, labels and annotation", {
  calls <- list()
  local_mocked_bindings(
    select_row = function(fit, measure, rule, not_two = FALSE, k = NULL, sweep_value = NULL) {
      calls$select <<- list(measure = measure, rule = rule, not_two = not_two, k = k, sweep_value = sweep_value)
      list(row = estimator_results(fit)[3L, ], config_id = 2L, n_clusters = 7L, pinned = TRUE)
    },
    get_labels = function(fit, ...) {
      calls$labels <<- list(...)
      rep(1:3, each = 30L)
    }
  )
  inputs <- fit_sample_plot_inputs(
    fit, source = "ce", measure = "generalizability", rule = "max", not_two = TRUE,
    mode = "generalizability", k = NULL, sweep_value = 4, annotation = TRUE, tight_layout = TRUE
  )
  expect_identical(calls$select, list(measure = "generalizability", rule = "max", not_two = TRUE, k = NULL, sweep_value = 4))
  expect_identical(
    calls$labels[c("measure", "rule", "not_two", "k", "sweep_value", "consensus_k", "mode")],
    list(measure = "generalizability", rule = "max", not_two = TRUE, k = NULL, sweep_value = 4, consensus_k = 7L, mode = "generalizability")
  )
  expect_identical(inputs$scores, fit@stability_ce_scores[["2"]])
  expect_identical(inputs$labels, rep(1:3, each = 30L))
  expect_identical(
    inputs$annotation,
    annotation_text("generalizability", "max", estimator_results(fit), estimator_results(fit)[3L, ], 7L, pinned = TRUE, tight_layout = TRUE)
  )
})

test_that("an annotation is built, taken as given or left off", {
  inputs <- function(annotation) {
    fit_sample_plot_inputs(fit, "gini", "stability", "1se", FALSE, "default", NULL, NULL, annotation)
  }
  selected <- select_row(fit, "stability", "1se")
  expect_identical(
    inputs(TRUE)$annotation,
    annotation_text("stability", "1se", estimator_results(fit), selected$row, selected$n_clusters)
  )
  expect_identical(inputs("note")$annotation, "note")
  expect_null(inputs(FALSE)$annotation)
})

test_that("the fit's box and violin plots pass every argument to the drawing", {
  local_mocked_bindings(
    cluster_boxplot_plot = function(...) list(...),
    cluster_violin_plot = function(...) list(...)
  )
  id <- select_row(fit, "stability", "1se", k = 4)$config_id
  box <- do.call(plot_cluster_boxplot, c(list(fit, source = "ce", k = 4), box_style))
  expect_identical(box[[1L]], sample_scores(fit, id, "ce"))
  expect_identical(box[[2L]], get_labels(fit, k = 4))
  expect_args(box, box_style)
  violin <- do.call(plot_cluster_violin, c(list(fit, source = "accuracy", k = 4), violin_style))
  expect_identical(violin[[1L]], sample_scores(fit, id, "accuracy"))
  expect_identical(violin[[2L]], get_labels(fit, k = 4))
  expect_args(violin, violin_style)
  expect_identical(plot_cluster_boxplot(fit, source = "accuracy")$ylabel, "Cluster Generalizability")
  expect_identical(plot_cluster_violin(fit, source = "ce")$ylabel, "Cluster Stability (CE)")
  expect_identical(plot_cluster_violin(fit)$ylabel, "Cluster Stability (Gini)")
})

test_that("the fit's scatter plots pass every argument to the drawing", {
  local_mocked_bindings(
    cluster_scatter_plot = function(...) list(...),
    diagnostic_scatter_plot = function(...) list(...)
  )
  id <- select_row(fit, "stability", "1se", k = 4)$config_id
  other_X <- blobs$X * 2
  embedding <- list(embedding = blobs$X[, 2:1])
  scatter <- do.call(plot_cluster_scatter, c(list(fit, source = "ce", k = 4, X = other_X), embedding, scatter_style))
  expect_identical(scatter[[1L]], other_X)
  expect_identical(scatter[[2L]], get_labels(fit, k = 4))
  expect_identical(scatter[[3L]], sample_scores(fit, id, "ce"))
  expect_args(scatter, c(embedding, scatter_style, list(scores_name = "CE Stability")))
  diagnostic <- do.call(plot_diagnostic_scatter, c(list(fit, source = "accuracy", k = 4, X = other_X), embedding, diagnostic_style))
  expect_identical(diagnostic[[1L]], other_X)
  expect_identical(diagnostic[[3L]], sample_scores(fit, id, "accuracy"))
  expect_args(diagnostic, c(embedding, diagnostic_style, list(scores_name = "Generalizability")))
  defaults <- plot_cluster_scatter(fit)
  expect_identical(defaults[[1L]], input_data(fit))
  expect_null(defaults$embedding)
  expect_identical(defaults$scores_name, "Gini Stability")
  expect_null(plot_diagnostic_scatter(fit)$colorbar_label)
})

test_that("the fit's scatter annotation is tight in the legend and plain in the caption", {
  local_mocked_bindings(
    cluster_scatter_plot = function(...) list(...),
    diagnostic_scatter_plot = function(...) list(...)
  )
  selected <- select_row(fit, "stability", "1se")
  text <- function(tight) {
    annotation_text("stability", "1se", estimator_results(fit), selected$row, selected$n_clusters, tight_layout = tight)
  }
  expect_identical(plot_cluster_scatter(fit)$annotation, text(TRUE))
  expect_identical(plot_cluster_scatter(fit, annotation_style = "box")$annotation, text(FALSE))
  expect_identical(plot_diagnostic_scatter(fit)$annotation, text(TRUE))
  expect_identical(plot_diagnostic_scatter(fit, annotation_style = "box")$annotation, text(FALSE))
})

test_that("the single-cell sample plots pass every argument and read the stored columns", {
  local_mocked_bindings(
    cluster_boxplot_plot = function(...) list(...),
    cluster_violin_plot = function(...) list(...),
    cluster_scatter_plot = function(...) list(...),
    diagnostic_scatter_plot = function(...) list(...)
  )
  pinned <- attach_results(make_sce(blobs$X, reductions = list(PCA = blobs$X, UMAP = blobs$X * 2)), fit, k = 4)
  id <- select_row(fit, "stability", "1se", k = 4)$config_id
  labels <- get_labels(fit, k = 4)
  box <- do.call(plot_cluster_boxplot, c(list(pinned, source = "ce"), box_style))
  expect_equal(box[[1L]], sample_scores(fit, id, "ce"))
  expect_identical(box[[2L]], labels)
  expect_args(box, box_style)
  violin <- do.call(plot_cluster_violin, c(list(pinned, source = "accuracy"), violin_style))
  expect_equal(violin[[1L]], sample_scores(fit, id, "accuracy"))
  expect_identical(violin[[2L]], labels)
  expect_args(violin, violin_style)
  defaults <- plot_cluster_violin(pinned)
  expect_equal(defaults[[1L]], sample_scores(fit, id, "gini"))
  expect_identical(defaults$ylabel, "Cluster Stability (Gini)")
  expect_identical(defaults$annotation, cells_annotation(pinned, "carve", TRUE))
  scatter <- do.call(plot_cluster_scatter, c(list(pinned, source = "ce", basis = "PCA"), scatter_style))
  expect_equal(scatter[[1L]], blobs$X)
  expect_identical(scatter[[2L]], labels)
  expect_equal(scatter[[3L]], sample_scores(fit, id, "ce"))
  expect_args(scatter, c(scatter_style, list(scores_name = "Cluster Stability (CE)")))
  expect_equal(plot_cluster_scatter(pinned)[[1L]], blobs$X * 2)
  diagnostic <- do.call(plot_diagnostic_scatter, c(list(pinned, source = "accuracy", basis = "PCA"), diagnostic_style))
  expect_equal(diagnostic[[1L]], blobs$X)
  expect_equal(diagnostic[[3L]], sample_scores(fit, id, "accuracy"))
  expect_args(diagnostic, c(diagnostic_style, list(scores_name = "Cluster Generalizability")))
  diagnostic_defaults <- plot_diagnostic_scatter(pinned)
  expect_identical(diagnostic_defaults$colorbar_label, "Cluster Stability (Gini)")
  expect_identical(diagnostic_defaults$annotation, cells_annotation(pinned, "carve", TRUE))
})

test_that("the single-cell annotation reads the recorded configuration by its config_id", {
  stored <- attach_results(sce, resolution_fit, sweep_value = 1)
  params <- S4Vectors::metadata(stored)$carve$params
  results <- estimator_results(resolution_fit)
  # Reversed, a row's position no longer equals its config_id.
  S4Vectors::metadata(stored)$carve$results <- results[rev(seq_len(nrow(results))), ]
  row <- results[results$config_id == params$selected_config_id, , drop = FALSE]
  text <- cells_annotation(stored, "carve", TRUE)
  expect_identical(text, annotation_text("stability", "1se", results, row, params$selected_k, pinned = TRUE))
  expect_match(text, "resolution = 1, k ~ ", fixed = TRUE)
  expect_match(text, ", fixed)", fixed = TRUE)
  expect_identical(cells_annotation(stored, "carve", "note"), "note")
  expect_null(cells_annotation(stored, "carve", FALSE))
  S4Vectors::metadata(stored)$carve$params$selected_k <- 0L
  expect_match(cells_annotation(stored, "carve", TRUE), "k ~ None", fixed = TRUE)
  S4Vectors::metadata(stored)$carve$results <- NULL
  expect_null(cells_annotation(stored, "carve", TRUE))
})

test_that("the single-cell scatter plots draw a basis, the clustered reduction or the clustered data", {
  with_umap <- attach_results(make_sce(blobs$X, reductions = list(PCA = blobs$X, UMAP = blobs$X * 2)), fit)
  expect_equal(cells_scatter_data(with_umap, "carve", NULL), blobs$X * 2)
  expect_equal(cells_scatter_data(with_umap, "carve", "PCA"), blobs$X)
  # Read again from the object, the three-column reduction would be reduced
  # to principal components; as the basis it gives its first two columns.
  harmony <- cbind(blobs$X * 4, 1)
  custom <- attach_results(make_sce(blobs$X, reductions = list(harmony = harmony)), fit, reduction = "harmony")
  expect_equal(cells_scatter_data(custom, "carve", NULL), blobs$X * 4)
  wide <- cbind(blobs$X, blobs$X * 3)
  assay_only <- attach_results(
    make_sce(wide, reductions = list(), assays = list(logcounts = t(wide))), fit,
    assay = "logcounts", n_dims = 3
  )
  expect_equal(unname(cells_scatter_data(assay_only, "carve", NULL)), wide[, 1:3])
  expect_error(
    cells_scatter_data(with_umap, "carve", "tsne"),
    "basis='tsne' not found in reducedDimNames(object). Available names: ['PCA', 'UMAP']",
    fixed = TRUE
  )
})

test_that("a SingleCellExperiment draws the sample plots the fit draws", {
  expect_same_plot(plot_cluster_boxplot(written), plot_cluster_boxplot(fit))
  expect_same_plot(plot_cluster_violin(written, source = "accuracy"), plot_cluster_violin(fit, source = "accuracy"))
  expect_same_plot(plot_cluster_scatter(written), plot_cluster_scatter(fit))
  expect_same_plot(plot_diagnostic_scatter(written), plot_diagnostic_scatter(fit))
  # The legend titles differ as they do in Python: the single-cell plots
  # name the score with SOURCE_LABELS and do not tighten the annotation.
  selected <- select_row(fit, "stability", "1se")
  text <- function(tight) {
    annotation_text("stability", "1se", estimator_results(fit), selected$row, selected$n_clusters, tight_layout = tight)
  }
  fill_name <- function(plot) plot$scales$get_scales("fill")$name
  expect_identical(fill_name(plot_cluster_scatter(written)), paste0(text(FALSE), "\nCluster, Cluster Stability (Gini)"))
  expect_identical(fill_name(plot_cluster_scatter(fit)), paste0(text(TRUE), "\nCluster, Gini Stability"))
  expect_identical(fill_name(plot_diagnostic_scatter(written)), "Cluster Stability (Gini)")
  expect_identical(fill_name(plot_diagnostic_scatter(fit)), "Gini Stability")
  expect_null(plot_cluster_violin(attach_results(sce, fit, store_results = FALSE))$labels$caption)
})

test_that("a Seurat object draws the sample plots a SingleCellExperiment draws", {
  object <- attach_results(make_seurat(blobs$X), fit)
  expect_same_plot(plot_cluster_violin(object), plot_cluster_violin(written))
  expect_same_plot(plot_diagnostic_scatter(object), plot_diagnostic_scatter(written))
  stability_only <- attach_results(make_seurat(blobs$X), stability_fit, mode = "stability")
  expect_error(
    plot_cluster_boxplot(stability_only, source = "accuracy"),
    "object@meta.data$carve_generalizability not found, so source='accuracy' is unavailable.",
    fixed = TRUE
  )
})

test_that("the per-sample plots name a missing score, an unknown source or mode, and an unknown argument", {
  stability_only <- attach_results(sce, stability_fit, mode = "stability")
  expect_error(
    plot_cluster_violin(stability_only, source = "accuracy"),
    "colData(object)$carve_generalizability not found, so source='accuracy' is unavailable. It is written when run_carve runs with mode='default' or mode='generalizability'.",
    fixed = TRUE
  )
  expect_error(plot_cluster_boxplot(written, source = "nope"), "source must be one of ['accuracy', 'ce', 'gini']; got 'nope'.", fixed = TRUE)
  expect_error(plot_cluster_boxplot(fit, source = "nope"), "source must be one of: 'accuracy', 'gini', 'ce'.", fixed = TRUE)
  expect_error(plot_cluster_violin(fit, mode = "nope"), "mode must be one of: 'default', 'stability', 'generalizability'.", fixed = TRUE)
  expect_error(
    plot_cluster_violin(stability_fit, source = "accuracy"),
    "Generalizability scores are not available for this run.",
    fixed = TRUE
  )
  expect_error(plot_diagnostic_scatter(written, nope = 1), "Unknown argument: nope.", fixed = TRUE)
  expect_error(plot_cluster_scatter(fit, basis = "UMAP"), "Unknown argument: basis.", fixed = TRUE)
  expect_error(
    plot_cluster_boxplot(blobs$X),
    "object must be a CARVE fit, a SingleCellExperiment or a Seurat object, not an object of class 'matrix'.",
    fixed = TRUE
  )
})
```

Notes on these tests. The scores stored in `colData` are compared with `expect_equal()`, because `attach_results()` stores them with `as.numeric()`, which drops any attributes. In the scatter test of a fit, `embedding` is a one-element list so it can go into both the call and the expected arguments; the single-cell methods take no `embedding` and pass none. The annotation test reverses the stored table, so that looking a row up by position finds the other resolution and changes the text. `"k ~ None"` comes from Python's rule that a recorded `selected_k` of 0 means no count.

- [ ] Step 2: Run the tests to see them fail

```bash
Rscript -e 'devtools::test(filter = "plots")'
```

Expected: the Task 10 tests pass; the new ones fail with `could not find function "fit_sample_plot_inputs"`, `could not find function "plot_cluster_boxplot"` and the like.

- [ ] Step 3: Add the four generics to the end of `R/AllGenerics.R`

```r
#' @rdname plot_cluster_boxplot
#' @export
setGeneric("plot_cluster_boxplot", function(object, ...) standardGeneric("plot_cluster_boxplot"))

#' @rdname plot_cluster_violin
#' @export
setGeneric("plot_cluster_violin", function(object, ...) standardGeneric("plot_cluster_violin"))

#' @rdname plot_cluster_scatter
#' @export
setGeneric("plot_cluster_scatter", function(object, ...) standardGeneric("plot_cluster_scatter"))

#' @rdname plot_diagnostic_scatter
#' @export
setGeneric("plot_diagnostic_scatter", function(object, ...) standardGeneric("plot_diagnostic_scatter"))
```

- [ ] Step 4: Append the per-sample plots to `R/plots.R`

```r
# Per-sample scores by source: the suffix of the column run_carve() writes,
# the name the box and violin plots and the single-cell scatter plots give
# the score, the name the scatter plots of a fit give it, and the mode that
# writes the column.
SOURCE_SUFFIXES <- c(gini = "stability", ce = "stability_ce", accuracy = "generalizability")
SOURCE_LABELS <- c(
  gini = "Cluster Stability (Gini)",
  ce = "Cluster Stability (CE)",
  accuracy = "Cluster Generalizability"
)
SOURCE_NAMES <- c(gini = "Gini Stability", ce = "CE Stability", accuracy = "Generalizability")
SOURCE_MODES <- c(gini = "stability", ce = "stability", accuracy = "generalizability")

# An annotation argument: TRUE builds the text, a string is used as it is,
# and anything else turns the annotation off.
resolve_annotation <- function(annotation, build) {
  if (is.character(annotation)) {
    return(annotation)
  }
  if (isTRUE(annotation)) build() else NULL
}

# What a per-sample plot of a fit draws: the scores of the selected
# configuration, the labels of its cut at the selected number of clusters,
# and the annotation.
fit_sample_plot_inputs <- function(fit, source, measure, rule, not_two, mode, k, sweep_value,
                                   annotation, tight_layout = FALSE) {
  selected <- select_row(fit, measure, rule, not_two = not_two, k = k, sweep_value = sweep_value)
  scores <- sample_scores(fit, selected$config_id, source = source)
  labels <- get_labels(
    fit, measure = measure, rule = rule, k = k, sweep_value = sweep_value,
    consensus_k = selected$n_clusters, not_two = not_two, mode = labels_mode(mode)
  )
  list(
    scores = scores,
    labels = labels,
    annotation = resolve_annotation(annotation, function() {
      annotation_text(
        measure, rule, fit@estimator_results, selected$row, selected$n_clusters,
        pinned = selected$pinned, tight_layout = tight_layout
      )
    })
  )
}

cells_scores <- function(object, key, source) {
  if (!is.character(source) || length(source) != 1L || !source %in% names(SOURCE_SUFFIXES)) {
    stop(sprintf(
      "source must be one of %s; got %s.",
      python_list(sort(names(SOURCE_SUFFIXES))), format_repr(source)
    ), call. = FALSE)
  }
  column <- paste0(key, "_", SOURCE_SUFFIXES[[source]])
  values <- cells_get_column(object, column)
  if (is.null(values)) {
    stop(sprintf(
      "%s not found, so source=%s is unavailable. It is written when run_carve runs with mode='default' or mode=%s.",
      cells_column_where(object, column), format_repr(source), format_repr(SOURCE_MODES[[source]])
    ), call. = FALSE)
  }
  as.numeric(values)
}

# The annotation of a single-cell plot, built from the record as a fit's
# plots build theirs. The row is found by config_id. Without a stored
# results table there is no annotation, and a recorded selected_k of 0
# means no count, as in Python.
cells_annotation <- function(object, key, annotation) {
  resolve_annotation(annotation, function() {
    record <- cells_entry(object, key)
    params <- record$params
    results <- record[["results"]]
    if (is.null(results)) {
      return(NULL)
    }
    id <- as.integer(recorded(NULL, params, "selected_config_id", 0))
    rows <- results[as.integer(results$config_id) == id, , drop = FALSE]
    if (nrow(rows) == 0L) {
      return(NULL)
    }
    selected_k <- as.integer(recorded(NULL, params, "selected_k", 0))
    annotation_text(
      recorded(NULL, params, "measure", "stability"),
      recorded(NULL, params, "rule", "1se"),
      results, rows[1L, , drop = FALSE],
      if (selected_k == 0L) NULL else selected_k,
      pinned = isTRUE(params[["pinned"]])
    )
  })
}

# The matrix a single-cell scatter plot draws from: the reduced dimensions
# named basis, or the first of the usual embeddings, or the reduction CARVE
# clustered. Failing those, the data CARVE clustered, read again from the
# object only then; the drawing reduces it to two principal components.
cells_scatter_data <- function(object, key, basis) {
  params <- cells_entry(object, key)$params
  embedding <- cells_basis(object, basis = basis, fallback = params[["reduction"]])
  if (!is.null(embedding)) {
    return(embedding)
  }
  cells_matrix(
    object,
    assay = params[["assay"]], reduction = params[["reduction"]], n_dims = params[["n_dims"]]
  )
}

#' Box plot of per-sample scores by cluster
#'
#' Draws one box per cluster of the per-sample scores that
#' [sample_scores()] returns: stability in its Gini or cross-entropy form,
#' or held-out accuracy. Samples without a finite score are left out.
#'
#' For a fit, the plot selects a configuration with `measure`, `rule`,
#' `not_two`, `k` and `sweep_value`, takes its scores, and groups them by
#' the labels of its cut at the selected number of clusters, the labels
#' [get_labels()] returns. For a SingleCellExperiment or a Seurat object, it
#' draws the score and label columns [run_carve()] or [attach_results()]
#' wrote under `key`. They belong to the configuration selected then, and
#' nothing is selected again.
#'
#' @inheritParams plot_consensus_matrix
#' @param source The score: `"gini"` or `"ce"` for stability in its Gini or
#'   cross-entropy form, or `"accuracy"` for held-out accuracy.
#' @param measure,rule,not_two,mode,k,sweep_value For a fit, which
#'   configuration to plot and which consensus to cut, as in
#'   [plot_consensus_matrix()]. The single-cell methods plot the stored
#'   configuration and do not take these arguments.
#' @param order Cluster labels in the order to draw them. Clusters not named
#'   are not drawn.
#' @param palette Colors of the clusters, as in
#'   [plot_metric_over_n_clusters()].
#' @param showfliers Draw the points beyond the whiskers.
#' @param width Width of the boxes; the clusters are 1 apart.
#' @param title,xlabel,ylabel Title and axis labels. The default y label
#'   names the score.
#' @param annotation `TRUE` names the selected configuration and how it was
#'   selected, a string is shown as it is, and `FALSE` shows nothing. The
#'   text goes in the caption, below the panel.
#' @param rotation Angle of the cluster labels, in degrees.
#' @param ylim Limits of the y axis, used when `fit_ylim = FALSE`. `NULL`
#'   leaves them to ggplot2.
#' @param fit_ylim Fit the y axis to the scores, with a margin of 5 percent
#'   of their range and at least 0.02.
#' @return A ggplot object.
#' @seealso [plot_cluster_violin()], [sample_scores()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_cluster_boxplot(fit)
#' plot_cluster_boxplot(fit, source = "accuracy", k = 3)
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_cluster_boxplot(sce)
#' @rdname plot_cluster_boxplot
#' @export
setMethod("plot_cluster_boxplot", "CARVE", function(object, source = "gini", measure = "stability",
                                                    rule = "1se", not_two = FALSE,
                                                    mode = "default", k = NULL,
                                                    sweep_value = NULL, order = NULL,
                                                    palette = "Accent", showfliers = FALSE,
                                                    width = 0.75, title = NULL,
                                                    xlabel = "Cluster", ylabel = NULL,
                                                    annotation = TRUE, rotation = NULL,
                                                    ylim = c(-0.02, 1.02), fit_ylim = TRUE, ...) {
  check_dots(...)
  inputs <- fit_sample_plot_inputs(
    object, source, measure, rule, not_two, mode, k, sweep_value, annotation
  )
  cluster_boxplot_plot(
    inputs$scores, inputs$labels,
    order = order, palette = palette, showfliers = showfliers, width = width, title = title,
    xlabel = xlabel, ylabel = if (is.null(ylabel)) SOURCE_LABELS[[source]] else ylabel,
    annotation = inputs$annotation, rotation = rotation, ylim = ylim, fit_ylim = fit_ylim
  )
})

cells_plot_cluster_boxplot <- function(object, key = "carve", source = "gini", order = NULL,
                                       palette = "Accent", showfliers = FALSE, width = 0.75,
                                       title = NULL, xlabel = "Cluster", ylabel = NULL,
                                       annotation = TRUE, rotation = NULL,
                                       ylim = c(-0.02, 1.02), fit_ylim = TRUE, ...) {
  check_dots(...)
  scores <- cells_scores(object, key, source)
  cluster_boxplot_plot(
    scores, cells_labels(object, key),
    order = order, palette = palette, showfliers = showfliers, width = width, title = title,
    xlabel = xlabel, ylabel = if (is.null(ylabel)) SOURCE_LABELS[[source]] else ylabel,
    annotation = cells_annotation(object, key, annotation), rotation = rotation, ylim = ylim,
    fit_ylim = fit_ylim
  )
}

#' @rdname plot_cluster_boxplot
#' @export
setMethod("plot_cluster_boxplot", "SingleCellExperiment", cells_plot_cluster_boxplot)

#' @rdname plot_cluster_boxplot
#' @export
setMethod("plot_cluster_boxplot", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_cluster_boxplot(object, ...)
})

#' Violin plot of per-sample scores by cluster
#'
#' Draws one violin per cluster of the per-sample scores, with the
#' quartiles marked inside and the samples drawn as points on top. A cluster
#' with one finite score gets its point and marks but no violin. The
#' configuration and the scores are chosen as in [plot_cluster_boxplot()].
#'
#' The points are jittered with a fixed seed, so a plot comes out the same
#' each time and leaves the session's random numbers alone.
#'
#' @inheritParams plot_cluster_boxplot
#' @param density_norm How the violins are scaled: `"width"` gives them the
#'   same width, `"area"` the same area, and `"count"` areas in proportion
#'   to the number of samples.
#' @param stripplot Draw the samples as points.
#' @param jitter Spread the points sideways: `TRUE` for the default spread,
#'   `FALSE` for none, or the spread on either side, where the clusters are
#'   1 apart.
#' @param size Area of the points, in square points.
#' @param alpha Opacity of the points.
#' @param inner Marks inside each violin: `"box"` for the median and the
#'   range between the quartiles, `"quartile"` for a line at each quartile,
#'   or `"none"`.
#' @return A ggplot object.
#' @seealso [plot_cluster_boxplot()], [sample_scores()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_cluster_violin(fit)
#' plot_cluster_violin(fit, source = "ce", inner = "quartile", stripplot = FALSE)
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_cluster_violin(sce)
#' @rdname plot_cluster_violin
#' @export
setMethod("plot_cluster_violin", "CARVE", function(object, source = "gini", measure = "stability",
                                                   rule = "1se", not_two = FALSE,
                                                   mode = "default", k = NULL, sweep_value = NULL,
                                                   order = NULL, palette = "Accent",
                                                   density_norm = "width", stripplot = TRUE,
                                                   jitter = TRUE, size = 8, alpha = 0.22,
                                                   inner = "box", title = NULL,
                                                   xlabel = "Cluster", ylabel = NULL,
                                                   annotation = TRUE, rotation = NULL,
                                                   ylim = c(-0.02, 1.02), fit_ylim = TRUE, ...) {
  check_dots(...)
  inputs <- fit_sample_plot_inputs(
    object, source, measure, rule, not_two, mode, k, sweep_value, annotation
  )
  cluster_violin_plot(
    inputs$scores, inputs$labels,
    order = order, palette = palette, density_norm = density_norm, stripplot = stripplot,
    jitter = jitter, size = size, alpha = alpha, inner = inner, title = title, xlabel = xlabel,
    ylabel = if (is.null(ylabel)) SOURCE_LABELS[[source]] else ylabel,
    annotation = inputs$annotation, rotation = rotation, ylim = ylim, fit_ylim = fit_ylim
  )
})

cells_plot_cluster_violin <- function(object, key = "carve", source = "gini", order = NULL,
                                      palette = "Accent", density_norm = "width",
                                      stripplot = TRUE, jitter = TRUE, size = 8, alpha = 0.22,
                                      inner = "box", title = NULL, xlabel = "Cluster",
                                      ylabel = NULL, annotation = TRUE, rotation = NULL,
                                      ylim = c(-0.02, 1.02), fit_ylim = TRUE, ...) {
  check_dots(...)
  scores <- cells_scores(object, key, source)
  cluster_violin_plot(
    scores, cells_labels(object, key),
    order = order, palette = palette, density_norm = density_norm, stripplot = stripplot,
    jitter = jitter, size = size, alpha = alpha, inner = inner, title = title, xlabel = xlabel,
    ylabel = if (is.null(ylabel)) SOURCE_LABELS[[source]] else ylabel,
    annotation = cells_annotation(object, key, annotation), rotation = rotation, ylim = ylim,
    fit_ylim = fit_ylim
  )
}

#' @rdname plot_cluster_violin
#' @export
setMethod("plot_cluster_violin", "SingleCellExperiment", cells_plot_cluster_violin)

#' @rdname plot_cluster_violin
#' @export
setMethod("plot_cluster_violin", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_cluster_violin(object, ...)
})

#' Scatter plot of samples sized by their scores
#'
#' Draws the samples in two dimensions, colored by cluster. Samples with low
#' scores get large, opaque markers and samples with high scores small,
#' faint ones. The legend gives the mean score of each cluster. Samples
#' without a finite score are left out.
#'
#' For a fit, the plot draws the first two columns of `embedding` when it is
#' given. Otherwise it draws `X`, by default the data the fit clustered,
#' reduced to its first two principal components when it has more than two
#' columns. The configuration and the scores are chosen as in
#' [plot_cluster_boxplot()].
#'
#' For a SingleCellExperiment or a Seurat object, the plot draws the reduced
#' dimensions named `basis`. By default it takes the first it finds of
#' `"UMAP"`, `"TSNE"` and `"PCA"` (`"umap"`, `"tsne"` and `"pca"` for a
#' Seurat object), then the reduction CARVE clustered. Without any of these
#' it reads the data CARVE clustered from the object again, and reduces it as
#' for a fit.
#'
#' @inheritParams plot_cluster_boxplot
#' @inheritParams plot_metric_over_n_clusters
#' @param X For a fit, the data to draw, one row per sample. `NULL` uses
#'   [input_data()].
#' @param embedding For a fit, a matrix with one row per sample whose first
#'   two columns are drawn instead of `X`.
#' @param basis For a single-cell object, the name of the reduced dimensions
#'   to draw.
#' @param alpha_range Opacity of the markers: the value for the highest score
#'   first, then the one for the lowest.
#' @param size_range Area of the markers in square points: the value for the
#'   highest score first, then the one for the lowest.
#' @param sort_order Draw the faint markers first, so the opaque ones are on
#'   top.
#' @param annotation_style Where the annotation goes: `"legend"` above the
#'   legend title, `"box"` in the caption below the panel.
#' @param title,xlabel,ylabel Title and axis labels.
#' @param show_ticks Draw the axis ticks and their labels.
#' @param frameon Draw the border of the panel.
#' @return A ggplot object.
#' @seealso [plot_diagnostic_scatter()], [sample_scores()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_cluster_scatter(fit)
#' plot_cluster_scatter(fit, annotation_style = "box", frameon = TRUE)
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_cluster_scatter(sce, basis = "PCA")
#' @rdname plot_cluster_scatter
#' @export
setMethod("plot_cluster_scatter", "CARVE", function(object, source = "gini", measure = "stability",
                                                    rule = "1se", not_two = FALSE,
                                                    mode = "default", k = NULL,
                                                    sweep_value = NULL, X = NULL,
                                                    embedding = NULL, palette = "Accent",
                                                    alpha_range = c(0.45, 0.9),
                                                    size_range = c(15, 60), sort_order = TRUE,
                                                    legend = TRUE, legend_loc = "right",
                                                    annotation = TRUE,
                                                    annotation_style = "legend", title = NULL,
                                                    xlabel = "Component 1",
                                                    ylabel = "Component 2", show_ticks = FALSE,
                                                    frameon = FALSE, ...) {
  check_dots(...)
  inputs <- fit_sample_plot_inputs(
    object, source, measure, rule, not_two, mode, k, sweep_value, annotation,
    tight_layout = identical(annotation_style, "legend")
  )
  cluster_scatter_plot(
    if (is.null(X)) object@input_data else X,
    inputs$labels, inputs$scores,
    embedding = embedding, palette = palette, alpha_range = alpha_range, size_range = size_range,
    sort_order = sort_order, legend = legend, legend_loc = legend_loc,
    annotation = inputs$annotation, annotation_style = annotation_style, title = title,
    scores_name = SOURCE_NAMES[[source]], xlabel = xlabel, ylabel = ylabel,
    show_ticks = show_ticks, frameon = frameon
  )
})

cells_plot_cluster_scatter <- function(object, key = "carve", source = "gini", basis = NULL,
                                       palette = "Accent", alpha_range = c(0.45, 0.9),
                                       size_range = c(15, 60), sort_order = TRUE, legend = TRUE,
                                       legend_loc = "right", annotation = TRUE,
                                       annotation_style = "legend", title = NULL,
                                       xlabel = "Component 1", ylabel = "Component 2",
                                       show_ticks = FALSE, frameon = FALSE, ...) {
  check_dots(...)
  scores <- cells_scores(object, key, source)
  cluster_scatter_plot(
    cells_scatter_data(object, key, basis), cells_labels(object, key), scores,
    palette = palette, alpha_range = alpha_range, size_range = size_range,
    sort_order = sort_order, legend = legend, legend_loc = legend_loc,
    annotation = cells_annotation(object, key, annotation), annotation_style = annotation_style,
    title = title, scores_name = SOURCE_LABELS[[source]], xlabel = xlabel, ylabel = ylabel,
    show_ticks = show_ticks, frameon = frameon
  )
}

#' @rdname plot_cluster_scatter
#' @export
setMethod("plot_cluster_scatter", "SingleCellExperiment", cells_plot_cluster_scatter)

#' @rdname plot_cluster_scatter
#' @export
setMethod("plot_cluster_scatter", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_cluster_scatter(object, ...)
})

#' Scatter plot of samples colored by their scores
#'
#' Draws the samples in two dimensions with one marker shape per cluster and
#' the per-sample score as the fill color. With the default `cmap` low
#' scores are dark green and high scores nearly white. Samples without a
#' finite score are drawn in faint gray. The data, the configuration and the
#' scores are chosen as in [plot_cluster_scatter()].
#'
#' The clusters are drawn in order of their mean score, highest first. With
#' `sort_order = TRUE` the samples of each cluster are drawn from the highest
#' score to the lowest, so the lowest scores end up on top.
#'
#' @inheritParams plot_cluster_scatter
#' @param cmap Fill colors of the scores, as in [plot_consensus_matrix()].
#' @param alpha_encoding Show the score as opacity too.
#' @param alpha_range Opacity when `alpha_encoding = TRUE`: the value for the
#'   highest score first, then the one for the lowest.
#' @param marker_size Area of the markers, in square points.
#' @param marker_linewidth Width of the marker outlines, in points.
#' @param markers Marker shapes, one per cluster in label order: ggplot2
#'   shapes 21 to 25, the shapes with a fill. `NULL` uses 21, 22, 24, 23 and
#'   25 (circle, square, triangle, diamond and inverted triangle). The shapes
#'   repeat, with a warning, when there are more clusters than shapes.
#' @param sort_order Draw the samples of each cluster from the highest score
#'   to the lowest.
#' @param legend Show the legend of the marker shapes.
#' @param colorbar Show the color bar, below the panel.
#' @param colorbar_label Title of the color bar. `NULL` names the score.
#' @return A ggplot object.
#' @seealso [plot_cluster_scatter()], [sample_scores()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' rownames(X) <- sprintf("cell%02d", seq_len(nrow(X)))
#' grid <- list(estimator_grid(KMeans, n_clusters = 2:4))
#' fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = grid)
#' plot_diagnostic_scatter(fit)
#' plot_diagnostic_scatter(fit, source = "accuracy", cmap = "magma")
#'
#' sce <- SingleCellExperiment::SingleCellExperiment(
#'   reducedDims = list(PCA = X),
#'   colData = S4Vectors::DataFrame(row.names = rownames(X))
#' )
#' sce <- attach_results(sce, fit)
#' plot_diagnostic_scatter(sce)
#' @rdname plot_diagnostic_scatter
#' @export
setMethod("plot_diagnostic_scatter", "CARVE", function(object, source = "gini",
                                                       measure = "stability", rule = "1se",
                                                       not_two = FALSE, mode = "default",
                                                       k = NULL, sweep_value = NULL, X = NULL,
                                                       embedding = NULL, cmap = "Greens_r",
                                                       alpha_encoding = TRUE,
                                                       alpha_range = c(0.3, 1), marker_size = 30,
                                                       marker_linewidth = 0.2, markers = NULL,
                                                       sort_order = TRUE, legend = TRUE,
                                                       legend_loc = "right", colorbar = TRUE,
                                                       colorbar_label = NULL, annotation = TRUE,
                                                       annotation_style = "legend",
                                                       title = NULL, xlabel = "Component 1",
                                                       ylabel = "Component 2",
                                                       show_ticks = FALSE, frameon = FALSE, ...) {
  check_dots(...)
  inputs <- fit_sample_plot_inputs(
    object, source, measure, rule, not_two, mode, k, sweep_value, annotation,
    tight_layout = identical(annotation_style, "legend")
  )
  diagnostic_scatter_plot(
    if (is.null(X)) object@input_data else X,
    inputs$labels, inputs$scores,
    embedding = embedding, cmap = cmap, alpha_encoding = alpha_encoding,
    alpha_range = alpha_range, marker_size = marker_size, marker_linewidth = marker_linewidth,
    markers = markers, sort_order = sort_order, legend = legend, legend_loc = legend_loc,
    colorbar = colorbar, colorbar_label = colorbar_label, annotation = inputs$annotation,
    annotation_style = annotation_style, title = title, scores_name = SOURCE_NAMES[[source]],
    xlabel = xlabel, ylabel = ylabel, show_ticks = show_ticks, frameon = frameon
  )
})

cells_plot_diagnostic_scatter <- function(object, key = "carve", source = "gini", basis = NULL,
                                          cmap = "Greens_r", alpha_encoding = TRUE,
                                          alpha_range = c(0.3, 1), marker_size = 30,
                                          marker_linewidth = 0.2, markers = NULL,
                                          sort_order = TRUE, legend = TRUE,
                                          legend_loc = "right", colorbar = TRUE,
                                          colorbar_label = NULL, annotation = TRUE,
                                          annotation_style = "legend", title = NULL,
                                          xlabel = "Component 1", ylabel = "Component 2",
                                          show_ticks = FALSE, frameon = FALSE, ...) {
  check_dots(...)
  scores <- cells_scores(object, key, source)
  diagnostic_scatter_plot(
    cells_scatter_data(object, key, basis), cells_labels(object, key), scores,
    cmap = cmap, alpha_encoding = alpha_encoding, alpha_range = alpha_range,
    marker_size = marker_size, marker_linewidth = marker_linewidth, markers = markers,
    sort_order = sort_order, legend = legend, legend_loc = legend_loc, colorbar = colorbar,
    colorbar_label = if (is.null(colorbar_label)) SOURCE_LABELS[[source]] else colorbar_label,
    annotation = cells_annotation(object, key, annotation), annotation_style = annotation_style,
    title = title, scores_name = SOURCE_LABELS[[source]], xlabel = xlabel, ylabel = ylabel,
    show_ticks = show_ticks, frameon = frameon
  )
}

#' @rdname plot_diagnostic_scatter
#' @export
setMethod("plot_diagnostic_scatter", "SingleCellExperiment", cells_plot_diagnostic_scatter)

#' @rdname plot_diagnostic_scatter
#' @export
setMethod("plot_diagnostic_scatter", "ANY", function(object, ...) {
  check_seurat_object(object)
  cells_plot_diagnostic_scatter(object, ...)
})
```

The single-cell diagnostic plot passes the source's label as `colorbar_label` when none is given, as `pl.diagnostic_scatter` does, although `diagnostic_scatter_plot()` would fall back to the same `scores_name`; the mocked test checks the argument, the plot test the result.

- [ ] Step 5: Run the tests and the examples

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "plots")'
Rscript -e 'devtools::test()'
Rscript -e 'grDevices::pdf(NULL); devtools::run_examples(run_donttest = TRUE, document = FALSE, start = "plot_cluster_boxplot")' 2>&1 | head -80
```

Expected: `document()` adds the four generics to NAMESPACE and writes the four Rd pages, with the inherited arguments filled in. All tests pass, `SKIP 0`. The examples run without a warning or an error; stop reading after `plot_diagnostic_scatter` (the run goes on through the pages after it, which include Task 10's). Then look at the single-cell scatter plot once, as in Task 4 Step 5:

```bash
Rscript -e 'devtools::load_all(quiet = TRUE); set.seed(1); X <- rbind(matrix(rnorm(80, 0, 0.4), ncol = 2), matrix(rnorm(80, 3, 0.4), ncol = 2)); rownames(X) <- sprintf("cell%02d", 1:80); fit <- carve(X, n_resamples = 10, random_state = 0, estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4))); sce <- SingleCellExperiment::SingleCellExperiment(reducedDims = list(PCA = X), colData = S4Vectors::DataFrame(row.names = rownames(X))); sce <- attach_results(sce, fit); ggplot2::ggsave("/tmp/carve-sce-scatter.png", plot_cluster_scatter(sce), width = 6, height = 4.5); ggplot2::ggsave("/tmp/carve-fit-violin.png", plot_cluster_violin(fit), width = 5, height = 4)'
```

Describe both in the report (the annotation above the legend title, the legend naming each cluster's mean score; the violin plot's caption with the configuration and the rule), then delete the files.

- [ ] Step 6: Commit

```bash
git add carve-r/R/AllGenerics.R carve-r/R/plots.R carve-r/NAMESPACE carve-r/man/plot_cluster_boxplot.Rd carve-r/man/plot_cluster_violin.Rd carve-r/man/plot_cluster_scatter.Rd carve-r/man/plot_diagnostic_scatter.Rd carve-r/tests/testthat/test-plots.R
git commit -m "$(cat <<'EOF'
feat(carve-r): plot per-sample scores for fits and single-cell objects

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 7: Mutations

Apply each, run `Rscript -e 'devtools::test(filter = "plots")'`, confirm the named test fails, and restore with `git checkout -- R/plots.R`:

1. In `fit_sample_plot_inputs()`, drop `consensus_k = selected$n_clusters` from the `get_labels()` call: fails "fit_sample_plot_inputs selects once".
2. In `fit_sample_plot_inputs()`, pass `mode = mode` to `get_labels()` instead of `labels_mode(mode)`: fails "name a missing score, an unknown source or mode" (`get_labels()` rejects `"nope"` with its own message).
3. In the `CARVE` method of `plot_cluster_scatter`, pass `tight_layout = FALSE`: fails "tight in the legend and plain in the caption".
4. In `cells_annotation()`, look the row up by position, `results[id + 1L, , drop = FALSE]`: fails "reads the recorded configuration by its config_id".
5. In `cells_annotation()`, pass `selected_k` without the check for 0: fails the same test (`"k ~ 0"` instead of `"k ~ None"`).
6. In `cells_scores()`, read the `"stability"` column for every source: fails "pass every argument and read the stored columns".
7. In `cells_scatter_data()`, drop `fallback = params[["reduction"]]`: fails "draw a basis, the clustered reduction or the clustered data" (the three-column reduction comes back whole).
8. In `cells_plot_cluster_scatter()`, pass `scores_name = SOURCE_NAMES[[source]]`: fails "pass every argument and read the stored columns" and "draws the sample plots the fit draws".
9. In the `CARVE` method of `plot_cluster_boxplot`, default the y label to `SOURCE_NAMES[[source]]`: fails "the fit's box and violin plots pass every argument".

---

### Task 12: The `pbmc3k_subset` data set

Files:
- Create: `carve-r/data-raw/pbmc3k_subset.R`
- Create (generated): `carve-r/data/pbmc3k_subset.rda`
- Create: `carve-r/R/data.R`
- Modify: `carve-r/.gitignore`
- Modify: `carve-r/DESCRIPTION` (`LazyData: true`; Collate through `document()`)
- Create: `carve-r/tests/testthat/test-data.R`
- Generated: `carve-r/man/pbmc3k_subset.Rd`

Interfaces:
- Consumes: Seurat 5.5.0 and SeuratObject from the R library (the script only, not the package); the 10x Genomics download (Decision 15).
- Produces: the exported data set `pbmc3k_subset`, a list of `pca` (1,000 by 30 matrix, rows named by cell barcode, columns `PC_1` to `PC_30`), `umap` (1,000 by 2, columns `UMAP_1` and `UMAP_2`, the same row names) and `seurat_clusters` (a factor of length 1,000), with the scores rounded to four decimals. Task 13's parity case fits `pbmc3k_subset$pca`; stage 4's single-cell vignette uses all three.

The script follows the steps of Seurat's PBMC 3k tutorial on all cells and keeps 1,000 of them at the end, so the clusters and the UMAP come from the whole data set. Seurat's functions seed themselves (`RunPCA()` and `RunUMAP()` with 42, `FindClusters()` with 0), and the cells are drawn under `withr::with_seed(0)`.

- [ ] Step 1: Check the environment

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript -e 'for (p in c("Seurat", "SeuratObject", "uwot")) cat(p, as.character(packageVersion(p)), "\n")'
git check-ignore -v data/pbmc3k_subset.rda; echo "exit $?"
```

Expected: the three versions (Seurat 5.5.0); and `git check-ignore` reports a match on line 1 of `../.gitignore`, the `data/` pattern, with exit 0. That pattern is why Step 2 re-includes the directory (Decision 17).

- [ ] Step 2: Ignore the download cache and re-include `data/`

Append to `carve-r/.gitignore`:

```
/data-raw/cache/
!/data/
```

Then:

```bash
mkdir -p data-raw/cache data && touch data-raw/cache/probe data/probe
git status --short --ignored data data-raw/cache
rm data-raw/cache/probe data/probe
```

Expected: `?? data/` (tracked from now on) and `!! data-raw/cache/` (ignored). `data-raw` is already in `.Rbuildignore`, so the cache never reaches the package build.

- [ ] Step 3: Write the failing test

Create `tests/testthat/test-data.R`:

```r
test_that("pbmc3k_subset holds 1,000 cells in the documented layout", {
  data <- pbmc3k_subset
  expect_named(data, c("pca", "umap", "seurat_clusters"))
  expect_identical(dim(data$pca), c(1000L, 30L))
  expect_identical(dim(data$umap), c(1000L, 2L))
  expect_identical(colnames(data$pca), paste0("PC_", 1:30))
  expect_identical(colnames(data$umap), c("UMAP_1", "UMAP_2"))
  expect_identical(rownames(data$umap), rownames(data$pca))
  expect_false(anyDuplicated(rownames(data$pca)) > 0L)
  expect_s3_class(data$seurat_clusters, "factor")
  expect_length(data$seurat_clusters, 1000L)
  expect_gt(nlevels(droplevels(data$seurat_clusters)), 1L)
  expect_true(all(is.finite(data$pca)) && all(is.finite(data$umap)))
})

test_that("pbmc3k_subset's scores are rounded to four decimals", {
  expect_identical(pbmc3k_subset$pca, round(pbmc3k_subset$pca, 4))
  expect_identical(pbmc3k_subset$umap, round(pbmc3k_subset$umap, 4))
})
```

```bash
Rscript -e 'devtools::test(filter = "data")'
```

Expected: FAIL, `object 'pbmc3k_subset' not found`.

- [ ] Step 4: Create `data-raw/pbmc3k_subset.R`

```r
# Builds data/pbmc3k_subset.rda: 1,000 cells of the PBMC 3k data set from
# 10x Genomics, after the steps of Seurat's PBMC 3k tutorial. Run from
# code/carve-r:
#
#   Rscript data-raw/pbmc3k_subset.R
#
# It needs Seurat and uwot, which the package itself does not use. The
# download, about 7 MB, is kept in data-raw/cache/, which git ignores.

url <- "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
cache <- file.path("data-raw", "cache")
dir.create(cache, showWarnings = FALSE, recursive = TRUE)
archive <- file.path(cache, basename(url))
if (!file.exists(archive)) {
  utils::download.file(url, archive, mode = "wb")
}
matrix_dir <- file.path(cache, "filtered_gene_bc_matrices", "hg19")
if (!dir.exists(matrix_dir)) {
  utils::untar(archive, exdir = cache)
}

# The tutorial's steps, on all cells.
object <- SeuratObject::CreateSeuratObject(
  counts = Seurat::Read10X(matrix_dir),
  project = "pbmc3k", min.cells = 3, min.features = 200
)
object[["percent.mt"]] <- Seurat::PercentageFeatureSet(object, pattern = "^MT-")
meta <- object[[]]
keep <- meta$nFeature_RNA > 200 & meta$nFeature_RNA < 2500 & meta$percent.mt < 5
object <- subset(object, cells = colnames(object)[keep])
object <- Seurat::NormalizeData(object, verbose = FALSE)
object <- Seurat::FindVariableFeatures(object, nfeatures = 2000, verbose = FALSE)
object <- Seurat::ScaleData(object, verbose = FALSE)
object <- Seurat::RunPCA(object, npcs = 30, verbose = FALSE)
object <- Seurat::FindNeighbors(object, dims = 1:10, verbose = FALSE)
object <- Seurat::FindClusters(object, resolution = 0.5, verbose = FALSE)
object <- Seurat::RunUMAP(object, dims = 1:10, verbose = FALSE)

# 1,000 of the cells, in their original order. Four decimals keep the file
# small and change no clustering.
cells <- colnames(object)[sort(withr::with_seed(0L, sample.int(ncol(object), 1000L)))]
pca <- round(SeuratObject::Embeddings(object, reduction = "pca")[cells, ], 4)
umap <- round(SeuratObject::Embeddings(object, reduction = "umap")[cells, ], 4)
colnames(pca) <- paste0("PC_", seq_len(ncol(pca)))
colnames(umap) <- c("UMAP_1", "UMAP_2")
pbmc3k_subset <- list(
  pca = pca,
  umap = umap,
  seurat_clusters = object[[]][cells, "seurat_clusters"]
)
save(pbmc3k_subset, file = file.path("data", "pbmc3k_subset.rda"), compress = "xz")
```

- [ ] Step 5: Run it

```bash
Rscript data-raw/pbmc3k_subset.R
ls -l data/pbmc3k_subset.rda
Rscript -e 'load("data/pbmc3k_subset.rda"); str(pbmc3k_subset); print(table(pbmc3k_subset$seurat_clusters))'
```

Expected: the download once, Seurat's messages, and a file of about 150 KB; above 300 KB, say so in the report. The tutorial keeps 2,638 of the 2,700 cells and finds nine clusters at resolution 0.5, so `table()` should show nine clusters over the 1,000 cells. A different count is not a failure (Seurat versions differ), but report it, because the help page below does not state the count.

- [ ] Step 6: Document the data set and load it lazily

In `DESCRIPTION`, add `LazyData: true` after `Encoding: UTF-8`.

Create `R/data.R`:

```r
#' PBMC 3k cells
#'
#' 1,000 cells of the PBMC 3k data set from 10x Genomics, peripheral blood
#' mononuclear cells from a healthy donor, after the steps of Seurat's PBMC
#' 3k tutorial. It is small enough for examples that fit CARVE to
#' single-cell data without a download.
#'
#' The steps ran on all 2,700 cells. Cells with 200 to 2,500 detected genes
#' and less than 5 percent of their counts from mitochondrial genes were
#' kept. The counts were log-normalized, the 2,000 most variable genes were
#' scaled, and 30 principal components were computed. The clusters come from
#' Seurat's Louvain clustering at resolution 0.5, and the UMAP from the
#' first 10 components. Then 1,000 cells were drawn at random, and the scores
#' were rounded to four decimals. The script is `data-raw/pbmc3k_subset.R`
#' in the package's source repository.
#'
#' @format A list with one entry per cell in each element:
#' \describe{
#'   \item{pca}{A 1,000 by 30 matrix of principal component scores, with the
#'     cell barcodes as row names.}
#'   \item{umap}{A 1,000 by 2 matrix of UMAP coordinates, with the same row
#'     names.}
#'   \item{seurat_clusters}{A factor, the Seurat cluster of each cell.}
#' }
#' @source 10x Genomics,
#'   <https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz>
#' @examples
#' dim(pbmc3k_subset$pca)
#' table(pbmc3k_subset$seurat_clusters)
"pbmc3k_subset"
```

- [ ] Step 7: Run the tests

```bash
Rscript -e 'devtools::document()'
Rscript -e 'devtools::test(filter = "data")'
Rscript -e 'devtools::test()'
```

Expected: `document()` writes `man/pbmc3k_subset.Rd` with `\docType{data}` and adds `'data.R'` to Collate; nothing is added to NAMESPACE, because lazy data is found without an export. The data tests pass, and the full suite passes with `SKIP 0`.

- [ ] Step 8: Commit

```bash
git add carve-r/.gitignore carve-r/DESCRIPTION carve-r/R/data.R carve-r/man/pbmc3k_subset.Rd carve-r/data/pbmc3k_subset.rda carve-r/data-raw/pbmc3k_subset.R carve-r/tests/testthat/test-data.R
git status --short carve-r/
git commit -m "$(cat <<'EOF'
feat(carve-r): ship 1,000 PBMC 3k cells as pbmc3k_subset

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

Expected: `git status` before the commit shows nothing under `carve-r/data-raw/cache/`.

- [ ] Step 9: Mutation

The data set is a file, so the mutation changes the file. Run this, confirm that "rounded to four decimals" fails, and restore with `git checkout -- data/pbmc3k_subset.rda`:

```bash
Rscript -e 'load("data/pbmc3k_subset.rda"); pbmc3k_subset$pca[1, 1] <- pbmc3k_subset$pca[1, 1] + 1e-5; save(pbmc3k_subset, file = "data/pbmc3k_subset.rda", compress = "xz")'
Rscript -e 'devtools::test(filter = "data")'
git checkout -- data/pbmc3k_subset.rda
```

---

### Task 13: Parity with Python, the package check and the writing pass

Files:
- Modify: `carve-r/data-raw/parity_check.R`
- Create (generated): `carve-r/data-raw/parity-report.md` (overwritten)
- Create (outside the repository): `_claude_playground/r-rework-stage3-records/stage3-report.md`

Interfaces:
- Consumes: the whole package, `pbmc3k_subset` (Task 12); `reticulate` and `devtools` from the R library; the Python package in `code/.venv`.
- Produces: the parity report with the PBMC 3k case, a clean `R CMD check`, the writing pass, and the stage report. This task changes no package code unless the check or the report finds a defect; such a fix goes in its own commit with a test that fails without it.

- [ ] Step 1: Add the PBMC 3k case to `data-raw/parity_check.R`

After `randomized_case()`, add:

```r
pbmc_case <- function() {
  X <- pbmc3k_subset$pca
  resolutions <- c(0.25, 0.5, 0.75, 1, 1.5)
  r_fit <- carve(X, resolution = resolutions, n_resamples = n_resamples, random_state = 0)
  py_fit <- pycarve$CARVE(resolution = np$array(resolutions), n_resamples = n_resamples, random_state = 0L)$fit(X)
  compare_fits(
    "resolution sweep, PBMC 3k principal components", r_fit, py_fit, c("method_label", "resolution"),
    by_value = TRUE,
    note = "The 1,000 cells and 30 principal components of pbmc3k_subset. Leiden runs in igraph in R and in leidenalg in Python."
  )
}
```

and add `pbmc_case()` to the `results` list after `randomized_case()`:

```r
results <- list(
  k_case("easy blobs", easy),
  k_case("hard blobs", blobs(500, 2.3, 3)),
  k_case("circles", datasets$make_circles(n_samples = 400L, factor = 0.5, noise = 0.05, random_state = 0L)[[1L]]),
  resolution_case(),
  hdbscan_case(),
  anchored_case(easy),
  randomized_case(),
  pbmc_case()
)
```

`devtools::load_all()` at the top of the script also loads the package's data, so `pbmc3k_subset` is found. reticulate passes the R matrix to Python as a float64 array and drops its row names.

- [ ] Step 2: Run it

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript data-raw/parity_check.R
git diff --stat data-raw/parity-report.md
git diff data-raw/parity-report.md | head -80
```

Expected: it finishes (stage 2's seven cases took about 20 minutes; allow 30) and rewrites the report. The engine has not changed in this stage (Decision 18), so the diff is the date in the first paragraph and the new section at the end, nothing else. Any other changed line is a defect: find which commit of this stage changed the fit before going on. A changed Python result is the other possible cause; Task 1 Step 3 would have shown it as changed fixtures.

- [ ] Step 3: Read the PBMC 3k case

In the new section:

- The selected resolutions should match, or sit next to each other on the grid. They are selected from curves over 1,000 cells, which are flatter than the blob cases'.
- The ARI between the R and Python labels is reported, not tested. Leiden's partitions agree in quality between igraph and leidenalg, not label by label (the stage 2 report says the same of the blob case).
- For any column marked `(over)`, run stage 2's seed study before calling it a defect: refit the case in both languages with `random_state` 0, 1000, 2000, 3000 and 4000 and compare the means with the seed-to-seed spread. Announce the run first; it is ten fits of 1,000 cells. A gap outside both spreads is a defect to diagnose, not to tolerate.

- [ ] Step 4: Re-read the two stage 2 residuals

Stage 2 left two gaps open, measured over five or ten seeds in `_claude_playground/r-rework-stage2-records/stage2-report.md` ("Seed studies"): randomized SpectralClustering at k = 3 and 4, whose consensus columns are lower in R by about 0.02 to 0.03 (k = 4 PAC 0.033 against a tolerance of 0.03), and anchored SpectralClustering at k = 5, whose accuracy_generalizability is 0.759 in R against 0.781 in Python. Check that the "easy blobs, anchored over 200 of 500 samples" and "easy blobs, randomized preprocessing" sections of the new report are unchanged (Step 2), and record in the stage report that both residuals stand as stage 2 measured them. This plan does not diagnose them (Decision 18).

- [ ] Step 5: Check the writing in the package sources

Load the `de-ai-writing` skill first. Then:

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
grep -n -i -E "—|–|additionally|crucial|delve|enhanc|foster|robust|tapestry|pivotal|seamless|serves as|leverag|showcas|underscor|it's important|in conclusion|\*\*" R/*.R data-raw/parity_check.R data-raw/pbmc3k_subset.R || echo "no hits"
Rscript -e 'for (f in c(list.files("R", full.names = TRUE), list.files("tests/testthat", pattern = "[.]R$", full.names = TRUE), list.files("data-raw", pattern = "[.]R$", full.names = TRUE))) tools::showNonASCIIfile(f)'
```

Expected: `no hits`, and no lines from `showNonASCIIfile`. Then read every roxygen block this stage added or changed against `~/.claude/skills/de-ai-writing/references/signs.md`: the eight plot pages in `plots.R`, `run_carve()` in `tl.R`, `pbmc3k_subset` in `data.R`, the `carve()` parameters in `carve.R` and `sce.R`, and the `run_params` slot in `AllClasses.R`; and the code comments of `plotting.R`, `plots.R`, `sce.R`, `seurat.R` and `tl.R`. Look for the signs a grep cannot find: rule of three, elegant variation, contrasts nobody raised, summary endings. Fix any hit by rewriting the sentence, not by swapping a word. The error and warning messages copied from Python keep Python's wording (Global Constraints). List any hit kept on purpose in the report.

- [ ] Step 6: Run R CMD check

```bash
cd /Users/kaiwycik/GitHub/CARVE/code/carve-r
Rscript -e 'devtools::document()'
R CMD build .
_R_CHECK_FORCE_SUGGESTS_=false R CMD check --no-manual CARVE_2.0.0.tar.gz
```

Expected: `Status: OK`, or NOTEs only. A WARNING or an ERROR fails the stage, as in CI (`error-on: '"warning"'`). Record any NOTE in the report. The Seurat tests skip themselves without SeuratObject, so the check passes with or without it. Then remove the build output, which is not tracked:

```bash
rm -rf CARVE.Rcheck CARVE_2.0.0.tar.gz
git status --short
```

Expected: only `data-raw/parity_check.R` and `data-raw/parity-report.md` changed, plus any `man/` or `NAMESPACE` change from `document()` (there should be none).

- [ ] Step 7: Commit

```bash
git add carve-r/data-raw/parity_check.R carve-r/data-raw/parity-report.md
git commit -m "$(cat <<'EOF'
test(carve-r): compare the PBMC 3k resolution sweep with Python

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] Step 8: Write the stage report

Write `_claude_playground/r-rework-stage3-records/stage3-report.md` (outside the repository, as the earlier stages' records were): the final test count and `SKIP 0`; the `R CMD check` status and NOTEs; the parity diff (date and new section only, or what else changed and why); the PBMC 3k section with its reading and any seed study; the two stage 2 residuals as they stand; what the plot images of Tasks 4, 5, 6 and 11 showed; every plan defect found and how it was fixed, with the failing output; every mutation that did not fail as the plan predicted; and the writing pass result.

---

## After stage 3

The package now draws the eight plots for a fit and for a SingleCellExperiment or Seurat object, fits both classes, stores the selected configuration in them with `run_carve()` and `attach_results()`, and ships `pbmc3k_subset`. The next plan (stage 4, written in its own conversation) writes the README, NEWS, the four vignettes, the tutorial and its rendered HTML, does the final writing pass and the BiocCheck report, and reruns the parity script. The branch is merged into `main` only after stage 4.

Notes for stage 4 from this plan:

- Decisions 2, 4, 5, 6, 7, 8, 11 and 16 change what a user sees or the spec's text. The spec still lists patchwork in Imports, one `plotting.R` for both Python plot modules, and `run_carve()` and `attach_results()` without `assay`, `reduction` and `n_dims`; the author decides whether to amend it.
- `python-users.Rmd` should list where the plots differ from Python's: `legend_loc` takes ggplot2 positions, `palette` and `cmap` take R palette names, sizes and widths keep matplotlib's units, the annotation goes in the caption or the legend title, `k ~ 7` stands for `k ≈ 7`, `density_norm = "area"` and `"count"` draw differently, the diagnostic markers are the five filled shapes, the violin jitter is seeded, and `inner`, `annotation_style`, `legend_loc` and `markers` are checked. It should also say that each `run_carve()` call replaces the whole record, where Python leaves an old consensus matrix in `.obsp`, and that a Seurat `assay` means its `"data"` layer.
- Seurat (the full package) goes into Suggests with the first vignette that uses it (Decision 16). The tutorial caches the 10x download in `notebooks/data/`, as the spec says; `data-raw/pbmc3k_subset.R` shows the download and the steps.
- CI has not run on this branch. This stage adds three Bioconductor packages to Imports and SeuratObject to Suggests; check that the R CI job installs them before the merge.
- Open from stage 2 and still open: the BSD-3 attribution of the HDBSCAN port and the stale HDBSCAN text of the spec (author decisions before the merge), the two parity residuals, and stage 2's `python-users.Rmd` items.
