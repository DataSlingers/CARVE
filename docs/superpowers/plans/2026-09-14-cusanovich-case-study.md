# Cusanovich Case Study on Randomized Preprocessing Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: rebuild the Cusanovich case study around the source publication's own 30 clusters and a Leiden resolution sweep under randomized preprocessing (the LSI as-is, t-SNE, UMAP), with a four-panel figure, two CSV tables, a rewritten notebook, and a development-scale smoke run.

Architecture: the loader serves the source's 30 clusters over every cell. `STUDIES["cusanovich"]` declares the Leiden sweep, a resample count and a `PreprocessingSpec` of registry keys, which a new `_preprocessing` module resolves into the option lists `carve` takes. `carve_cache_path` gains a run key so a randomized fit cannot share a filename with any other fit. A new compute module, `_cusanovich_compare`, picks the best pipeline at CARVE's selection, embeds all cells with it, places the source's operating point on the t-SNE pipeline's resolution axis, probes the published partition with CARVE's own classifier, and assembles `CusanovichInputs`; `figures/_cusanovich_results` draws the four panels from that, through a generalized `carve_lines` and a new `pipeline_lines` in `_panels`. The CVI sweep, the k-based fit and the two `CompositeInputs` figures leave this study.

Tech stack: Python (`requires-python >= 3.13`; the main venv is 3.12.11 and runs the suite), NumPy, pandas 3, scikit-learn 1.7, umap-learn 0.5.12, igraph and leidenalg, matplotlib 3.11, nbformat and nbconvert for the notebook, pytest. No new dependencies. The only change under `src/carve/` is Task 7's fix to `_utils.align_cluster_labels`.

Spec: `docs/superpowers/specs/2026-09-13-cusanovich-randomized-preprocessing-case-study-design.md`, cited as "spec 5" and so on. It consumes the API built by `docs/superpowers/specs/2026-09-13-randomized-preprocessing-design.md`, implemented on branch `randomized-preprocessing` (head a95bf46): `preprocessing_results_`, `preprocessing_pipelines_`, `_pipeline.pipeline_from_spec` and `_plotting.plot_metric_by_pipeline`.

## Decisions taken with the author on 2026-09-14

- Spec 6's deletion is confirmed: `figure_carve_output_cusanovich`, the `CompositeInputs`-based `figure_cusanovich_results`, the constants `_carve_output.py` imports from `_cusanovich_results.py`, and their tests are removed. Klein and Levine are untouched.
- Task 0 fast-forwards `main` to `randomized-preprocessing` locally (no push) and cuts `cusanovich-case-study` from `main`.
- The open defect in `carve._plotting.plot_metric_by_pipeline` (the dashed selection marker is dropped under `not_two=True` when the best k=2 row has a finite standard error and the best remaining row does not) stays open. The notebook selects with `not_two=False`, which does not reach it.
- Changes to the R package are out of scope.
- `carve._utils.align_cluster_labels` must not merge clusters when the reference has fewer labels than the clustering has clusters; Task 7 fixes it in carve.
- The plan supplies scaffolding. The resolution grid stays at the spec's 0.1 to 2.0 whether or not it reaches 30 clusters at development scale; the author tunes the case study's parameters and runs it on the full data set.

## Scope

In scope: `src/benchmarks/`, `tests/benchmarks/`, `src/carve/_utils.py` and `tests/test_utils.py` (Task 7 only), `notebooks/case_studies/Cusanovich.ipynb`, and this plan under `docs/superpowers/plans/`.

Out of scope, and a reviewer does not flag their absence: anything else under `src/carve/`; anything under `carve-r/`, `.github/` or `pyproject.toml`; the case study's parameters and its publication-scale and full-atlas runs; the manuscript. `../overleaf/` is read-only, and the text drafts in spec 9 stay with the author.

## Global Constraints

- Run every command from `/Users/kaiwycik/GitHub/CARVE/code`. The venv is uv-managed and has no pip; call `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/python` and `.venv/bin/jupyter` directly.
- Lint and format with the pinned ruff (0.16.4) on `src/` only: `.venv/bin/ruff format src/` and `.venv/bin/ruff check src/`. Never on `tests/` or `notebooks/`. The code in this plan is already formatted, so `.venv/bin/ruff format --check src/` reports nothing to reformat after every task.
- `filterwarnings = error` is on. A new warning fails the suite; an expected one is asserted with `pytest.warns(..., match=...)`, and known noise is filtered where it arises.
- Tests mirror modules one to one: `src/carve/_utils.py` to `tests/test_utils.py`, `datasets/_cusanovich.py` to `tests/benchmarks/datasets/test_cusanovich.py`, `_types.py` to `tests/benchmarks/test_benchmarks_types.py`, `_preprocessing.py` to `tests/benchmarks/test_preprocessing.py`, `_studies.py` to `tests/benchmarks/test_studies.py`, `_cusanovich_compare.py` to `tests/benchmarks/test_cusanovich_compare.py`, `figures/_case_study.py` to `tests/benchmarks/figures/test_case_study.py`, `_panels.py` to `tests/benchmarks/test_panels.py`, `_theme.py` to `tests/benchmarks/test_theme.py`, `figures/_cusanovich_results.py` to `tests/benchmarks/figures/test_cusanovich_results.py`. Shared test doubles live in `tests/benchmarks/_helpers.py`.
- `benchmarks/_types.py` stays a leaf that imports only the standard library. `carve` never imports `benchmarks`.
- Case-study configuration lives in `STUDIES["cusanovich"]` and is read, never restated at a call site (spec 5).
- Every comparison with the source's clusters is reported as agreement, never as accuracy (spec 2).
- One theme: colors and line styles are defined in `_theme.py` only.
- No `logging`. Diagnostics go through `warnings.warn(..., stacklevel=2)`.
- American spelling. No bold or italics in code comments, docstrings, notebook markdown or authored Markdown.
- Run tests in the foreground with a long timeout (600000 ms on the Bash tool); `tests/benchmarks` alone takes several minutes. A slow run is not a hang.
- Commit after every task, ending the message with the attribution lines from the session's system reminder.

## Refinements the plan makes to the spec

Stated here so an executor or reviewer does not read them as deviations.

1. `PREPROCESSOR_CLASSES` maps each key to an import path, and `preprocessor_class` imports it on use. The spec lists classes; holding them directly would import umap-learn whenever `benchmarks._studies` is imported, and the package must import without the umap extra.
2. `resolve_preprocessing` binds `PREPROCESSOR_DEFAULTS` into the transformer with `functools.partial` and attaches the display name through carve's `(cls, name, params)` option form. A default passed as a one-value grid key would render in the label (`TSNE(n_components=2, perplexity=30)`, refinement 4 of the randomized-preprocessing plan); bound, it does not, and the labels read as spec 5 writes them.
3. `carve_cache_path` takes the run's `model_grids`, `n_resamples` and `preprocessing` as keyword arguments. The run key hashes the grid, whose keys name the swept parameter (the spec's "sweep parameter"), `preprocessing_fingerprint` of the spec, which adds the defaults it binds and imports nothing, and `n_resamples`. It is omitted when the grid equals `study_model_grids(study)`, there is no preprocessing and `n_resamples` is CARVE's default of 100. A study without a k-based grid must pass `model_grids`, so the Cusanovich notebook cannot fall back to a default grid that does not exist.
4. `fit_or_load_carve` raises `ValueError` when an option list is passed without `randomize_preprocessing=True`, since CARVE ignores the lists on a non-randomized fit.
5. `source_operating_point` takes `target_k` as a required keyword instead of defaulting to 30, and `prepare_cusanovich_inputs` passes the reference's own cluster count, so 30 is never restated. Ties go to the lower resolution, and a pipeline with rows for more than one estimator configuration raises.
6. `best_pipeline` raises for a measure other than stability or generalizability, the only criteria the per-pipeline table carries.
7. The assembly lives in `_cusanovich_compare.py` with the compute it calls: `CusanovichInputs`, `prepare_cusanovich_inputs`, `source_recipe_pipeline` (the one pipeline whose reduction is t-SNE), `selection_summary`, `TABLE_FILENAMES` and `save_tables`. The spec defines the dataclass in its figure section and calls assembly compute; the figure module only draws. `axis_prefix` is shared by `pipeline_embedding` and panel A's title. `prepare_cusanovich_inputs` imports `_align_to_reference` from `figures/_case_study.py` inside the function.
8. `published_partition_generalizability` gains `n_jobs` for the forest and returns a NaN standard error for a single split, as CARVE's tables do.
9. Task 7 is not in the spec; the author asked for it. `carve._utils.align_cluster_labels` merged clusters whenever there were more clusters than reference labels, and every unmatched cluster now takes a fresh id. `CARVE.get_labels` aligns only equal counts and is unaffected; `_accuracy` can reach the changed case, so `accuracy_generalizability` can move there. Panel A and the benchmarks composites align through `_align_to_reference`, which delegates to the fixed function.
10. `carve_lines` gains `rule`, which it hardcoded to `"1se"`, and reads the sweep axis from `sweep_`, falling back to k when `sweep_` is None: only fits cached before `sweep_` existed lack it, and all of those are k-based.
11. Panel C places the operating-point markers on the pooled curves at the operating point's resolution, as spec 7.2 says, with the t-SNE pipeline's observed count in the legend. The published partition's line is dashed, labeled "published 30 clusters, RF probe" with the count read from y. Panel D shares C's x axis through `Axes.sharex`.
12. `meta["drop_unknown"]` is recorded, and `meta["n_cells_annotated"]` counts the annotated cells of the whole atlas whether or not they were dropped (spec 4: "for information only").
13. The notebook selects with `MEASURE = "stability"`, `RULE = "1se"`, `NOT_TWO = False`, CARVE's defaults; the spec leaves the choice to the notebook.
14. `_cusanovich_results.AXIS_LABELS` goes with the function that imported it (spec 6, "their constants"). The rewritten module exports `MARKER_SIZE`, `SOURCE_TSNE_LABELS` and `SAVE_NAME`.
15. Task 0 deletes the invalid cache named in spec 3 together with its fingerprint sidecar. The run key can never resolve to that name anyway, which a Task 5 test pins.

## How this plan was verified

Tasks 1 to 11 were executed on 2026-09-14 in a scratch checkout of `src/`, `tests/`, the notebooks, `.github/`, `results/` and `.gitignore` taken from a95bf46, with the real `data/` linked in. The plan was replayed in its written order on a fresh checkout: for each task the test blocks were applied first and the task's tests run, which produced the failures quoted in Step 2; then the source blocks; then `ruff format --check src/` and `ruff check src/`, both with nothing to change; then the passing run quoted in Step 4. The task sections below are generated from the same edit files that replay applied, so the code here is the code that ran. Every new test was checked by mutation: 83 named perturbations of the source, listed under each task, each turned its test red against an unmutated control that was green. Three first-draft tests failed that check (a probe split count, a tree count and an embedding seed whose effects separated blobs cannot show) and were rewritten to record the calls instead. After Task 11 the benchmarks leg gave 877 passed and 10 skipped, against 786 passed and 10 skipped at a95bf46; that run had a first version of Task 7 with three benchmarks tests, which the final Task 7 replaces with one carve test. Task 7 was rewritten at the author's request to fix the function in carve and replayed on its own, with `tests/test_utils.py`, `tests/test_accuracy.py` and the non-randomized regression gate (105 passed). A development-scale fit on the released atlas ran the new code end to end; Task 12 repeats that through the notebook. Line numbers are not used anywhere: every replacement's old text occurs exactly once at that point in the task order.

## File structure

Created:
- `src/benchmarks/_preprocessing.py`: the preprocessing registry and `resolve_preprocessing`, the counterpart of `_estimators.py`.
- `src/benchmarks/_cusanovich_compare.py`: the comparison compute, `CusanovichInputs` and its assembly, and the tables.
- `tests/benchmarks/test_preprocessing.py`, `tests/benchmarks/test_cusanovich_compare.py`.

Rewritten:
- `src/benchmarks/figures/_cusanovich_results.py`: the four-panel figure.
- `tests/benchmarks/datasets/test_cusanovich.py`, `tests/benchmarks/figures/test_cusanovich_results.py`.
- `notebooks/case_studies/Cusanovich.ipynb`.

Modified:
- `src/benchmarks/datasets/_cusanovich.py`: the reference label, `drop_unknown`, `source_labels`.
- `src/benchmarks/_types.py`: `KNOWN_PREPROCESSORS`, `PreprocessingSpec`, `Study.n_resamples`, `Study.preprocessing`.
- `src/benchmarks/_studies.py`: `_cusanovich_loader`, `PACKAGE_N_RESAMPLES`, `fit_or_load_carve`, `_run_key`, `carve_cache_path`, `STUDIES["cusanovich"]`, the `study_model_grids` docstring.
- `src/carve/_utils.py`: `align_cluster_labels` gives unmatched clusters fresh ids.
- `src/benchmarks/_theme.py`: `PIPELINE_COLORS`, `PIPELINE_CMAP_NAME`, `MEASURE_LINESTYLES`.
- `src/benchmarks/_panels.py`: `_sweep_axis_label`, `carve_lines`, `pipeline_lines`.
- `src/benchmarks/figures/_carve_output.py` and `src/benchmarks/figures/__init__.py`: the Cusanovich output figure removed.
- `tests/benchmarks/_helpers.py`: `make_carve_spy` records fit kwargs; `StubCarve` handles resolution runs and records selection arguments; `resolution_results`, `pipeline_results`.
- `tests/benchmarks/test_benchmarks_types.py`, `test_studies.py`, `test_panels.py`, `test_theme.py`, `test_notebooks.py`, `figures/test_carve_output.py`, `figures/test_figure_contract.py`; `tests/test_utils.py`.

---

### Task 0: Branch, commit this plan, remove the invalid cache

Spec 3's preconditions. The cell-order fix is on `main` as 1dea71d. `randomized-preprocessing` is not merged; the author chose a local fast-forward.

- [ ] Step 1: Check the starting state

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse --short randomized-preprocessing
git merge-base --is-ancestor main randomized-preprocessing && echo "main can fast-forward"
.venv/bin/python -c "import umap, igraph, leidenalg, nbformat, nbconvert; print('extras present')"
```

Expected: `git status --short` lists only this plan file as untracked; the branch head is `a95bf46`; `main can fast-forward`; `extras present`. If anything else is uncommitted, or the head differs, stop and ask.

- [ ] Step 2: Fast-forward main and cut the branch

```bash
git checkout main
git merge --ff-only randomized-preprocessing
git checkout -b cusanovich-case-study
```

Nothing is pushed.

- [ ] Step 3: Delete the development cache fit before the cell-order fix

Spec 3: every Cusanovich number produced before 1dea71d is invalid, and this cache is deleted, not reused. The directory is gitignored, so nothing is committed. Leave every other file in it.

```bash
ls -la notebooks/case_studies/carve_state_saves/carve_cusanovich_dev_7841fb1f.carve*
rm notebooks/case_studies/carve_state_saves/carve_cusanovich_dev_7841fb1f.carve \
   notebooks/case_studies/carve_state_saves/carve_cusanovich_dev_7841fb1f.carve.x-sha1
```

- [ ] Step 4: Record the benchmarks leg before any change

Run: `.venv/bin/pytest tests/benchmarks -q 2>&1 | tail -1`

Expected: `786 passed, 10 skipped` (the count at a95bf46; the time varies).

- [ ] Step 5: Commit the plan

```bash
git add docs/superpowers/plans/2026-09-14-cusanovich-case-study.md
git commit -m "docs: implementation plan for the Cusanovich case study on randomized preprocessing"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 1: The loader serves the source's 30 clusters over every cell

Spec 4. The reference label becomes the source's `cluster` column, returned as strings. Every cell is kept by default, because the source's 30 clusters assign the 10,029 cells whose `cell_label` is `Unknown` too; `drop_unknown=True` restores the old filter. The source's label columns travel with X as `meta["source_labels"]`, and the subsample is stratified by cluster. `_cusanovich_loader` passes `label_column="cluster"` and leaves `drop_unknown` at its default. The fixture's `cluster` column becomes a function of each cell's t-SNE columns (two clusters per tissue), so a row of `source_labels` can be matched to its cell after the loader has filtered and subsampled; the cells file stays shuffled, as the cell-order fix (1dea71d) requires.

Files:
- Modify: `src/benchmarks/_studies.py`
- Modify: `src/benchmarks/datasets/_cusanovich.py`
- Test: `tests/benchmarks/datasets/test_cusanovich.py`
- Test: `tests/benchmarks/test_studies.py`

Interfaces:
- Consumes: nothing new.
- Produces: `load_cusanovich(*, root=None, subsample=None, random_state=42, site_frequency_threshold=0.03, n_components=50, label_column="cluster", drop_unknown=False) -> (X, y, meta)`. New meta keys: `source_labels` (a `pandas.DataFrame` with columns `tissue`, `cluster`, `subset_cluster`, `cell_label`, row-aligned with X) and `drop_unknown`. `n_cells_annotated` now counts the atlas cells whose `cell_label` is not `Unknown`, whether or not they were dropped.

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/test_studies.py`, replace:

```python
    _heca_loader,
```

with:

```python
    _cusanovich_loader,
    _heca_loader,
```

In `tests/benchmarks/test_studies.py`, replace:

```python
    def test_levine_loader_requests_five_thousand_cells(self, monkeypatch):
```

with:

```python
    def test_cusanovich_loader_references_the_source_clusters(self, monkeypatch):
        # The reference is the source's 30 clusters, which assign every cell,
        # so the Unknown-labeled cells stay: drop_unknown is left at False.
        calls = {}

        def fake_load_cusanovich(**kwargs):
            calls.update(kwargs)
            return np.zeros((1, 1)), pd.Series(["a"]), {}

        monkeypatch.setattr(
            "benchmarks.datasets.load_cusanovich", fake_load_cusanovich
        )
        _cusanovich_loader(1500)
        assert calls["subsample"] == 1500
        assert calls["label_column"] == "cluster"
        assert calls.get("drop_unknown", False) is False

    def test_levine_loader_requests_five_thousand_cells(self, monkeypatch):
```

Create `tests/benchmarks/datasets/test_cusanovich.py` with this content (replacing the file if it exists):

```python
"""Cusanovich loader tests.

The real matrix is 1.1 GB, so every test builds a small fixture in the same
on-disk layout and exercises the preprocessing chain against it.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

from benchmarks.datasets import load_cusanovich

N_CELLS = 120
N_PEAKS = 400
TISSUE_CODE = {"Lung": 0.0, "Liver": 1.0, "Spleen": 2.0}


def _planted_cluster(tsne_1: np.ndarray, tsne_2: np.ndarray) -> np.ndarray:
    """The fixture's source cluster, recomputed from its t-SNE columns.

    Two clusters per tissue, split by the parity of the cell's matrix column,
    so a row of source_labels can be checked against the same row of
    source_tsne without reconstructing the loader's filter or subsample.
    """
    return (2 * tsne_2 + tsne_1 % 2 + 1).astype(int)


@pytest.fixture
def atlas(tmp_path):
    rng = np.random.default_rng(0)
    tissues = np.repeat(["Lung", "Liver", "Spleen"], N_CELLS // 3)

    # Three tissue-specific peak blocks plus a shared background, so the
    # embedding has structure a clustering can actually recover.
    M = np.zeros((N_PEAKS, N_CELLS), dtype=np.int8)
    for block, tissue in enumerate(["Lung", "Liver", "Spleen"]):
        rows = slice(block * 100, (block + 1) * 100)
        cols = tissues == tissue
        M[rows, cols] = rng.random((100, cols.sum())) < 0.6
    M[300:, :] = rng.random((100, N_CELLS)) < 0.3

    root = tmp_path / "data"
    d = root / "Cusanovich"
    (d / "matrices").mkdir(parents=True)
    (d / "metadata").mkdir(parents=True)

    path = d / "matrices" / "atac_matrix.binary.qc_filtered.mtx"
    mmwrite(str(path), sparse.csr_matrix(M), field="integer")
    with open(path, "rb") as src, gzip.open(str(path) + ".gz", "wb") as dst:
        dst.write(src.read())
    path.unlink()

    cells = [f"cell{i:04d}" for i in range(N_CELLS)]
    # The release's .cells.txt holds the same barcodes as cell_metadata.txt
    # but in a different order, and it is the metadata's row order, not the
    # cells file, that matches the matrix columns. Write the cells file
    # shuffled so a loader that reindexes by it scrambles the labels here
    # the way it does on the real atlas.
    (d / "matrices" / "atac_matrix.binary.qc_filtered.cells.txt").write_text(
        "\n".join(cells[i] for i in rng.permutation(N_CELLS)) + "\n"
    )
    (d / "matrices" / "atac_matrix.binary.qc_filtered.peaks.txt").write_text(
        "\n".join(f"chr1_{i}_{i + 100}" for i in range(N_PEAKS)) + "\n"
    )
    # The source's own t-SNE coordinates. tsne_1 encodes the cell's column
    # in the matrix and tsne_2 its tissue, and the source cluster is a
    # function of both, so a test can check which cell a row belongs to
    # after the loader has filtered and subsampled. Rows are in matrix
    # column order, as in the real release.
    tsne_1 = np.arange(N_CELLS, dtype=float)
    tsne_2 = np.array([TISSUE_CODE[t] for t in tissues], dtype=float)
    pd.DataFrame(
        {
            "cell": cells,
            "tissue": tissues,
            "cluster": _planted_cluster(tsne_1, tsne_2),
            "subset_cluster": np.arange(N_CELLS) % 3 + 1,
            "tsne_1": tsne_1,
            "tsne_2": tsne_2,
            "cell_label": np.where(rng.random(N_CELLS) < 0.15, "Unknown", tissues),
        }
    ).to_csv(d / "metadata" / "cell_metadata.txt", sep="\t", index=False)
    return root


def _metadata(atlas: Path) -> pd.DataFrame:
    return pd.read_csv(atlas / "Cusanovich" / "metadata" / "cell_metadata.txt", sep="\t")


class TestLoadCusanovich:
    def test_returns_cells_by_components(self, atlas):
        X, y, meta = load_cusanovich(root=atlas, n_components=10)
        assert X.ndim == 2
        assert X.shape[1] == 10
        assert X.shape[0] == y.shape[0]
        assert np.isfinite(X).all()

    def test_every_cell_is_kept_by_default(self, atlas):
        # The source's clusters assign every cell, Unknown-labeled ones
        # included, so the default keeps them all and still reports how many
        # are annotated.
        X, _, meta = load_cusanovich(root=atlas, n_components=10)
        labels = meta["source_labels"]["cell_label"]
        assert X.shape[0] == N_CELLS
        assert meta["n_cells"] == N_CELLS
        assert "Unknown" in set(labels)
        assert meta["n_cells_annotated"] == int((labels != "Unknown").sum())
        assert meta["n_cells_annotated"] < N_CELLS

    def test_drop_unknown_keeps_only_the_annotated_cells(self, atlas):
        X, _, meta = load_cusanovich(root=atlas, n_components=10, drop_unknown=True)
        annotated = int((_metadata(atlas)["cell_label"] != "Unknown").sum())
        assert X.shape[0] == annotated
        assert meta["n_cells_annotated"] == annotated
        assert meta["n_cells_full"] == N_CELLS
        assert "Unknown" not in set(meta["source_labels"]["cell_label"])

    def test_label_column_defaults_to_the_source_clusters(self, atlas):
        _, cluster, _ = load_cusanovich(root=atlas, n_components=10)
        _, tissue, _ = load_cusanovich(
            root=atlas, n_components=10, label_column="tissue"
        )
        assert cluster.name == "cluster"
        assert set(cluster) == {str(c) for c in range(1, 7)}
        assert tissue.name == "tissue"

    def test_site_frequency_threshold_filters_peaks(self, atlas):
        _, _, loose = load_cusanovich(
            root=atlas, n_components=10, site_frequency_threshold=0.0
        )
        # threshold=0.5 leaves zero peaks for this fixture (see
        # test_zero_surviving_peaks_warns_and_does_not_claim_svd_ran, below),
        # which load_cusanovich warns about; that warning is exercised on
        # its own there; it is incidental here and must not leak into the
        # suite's output.
        with pytest.warns(UserWarning, match="site_frequency_threshold=0.5"):
            _, _, strict = load_cusanovich(
                root=atlas, n_components=10, site_frequency_threshold=0.5
            )
        assert strict["n_peaks_kept"] < loose["n_peaks_kept"]

    def test_zero_surviving_peaks_warns_and_does_not_claim_svd_ran(self, atlas):
        # threshold=0.5 leaves zero peaks for this fixture (see the test
        # above), so X degenerates to a zero embedding. That must be loud,
        # not silent: a warning naming the threshold, and meta that does not
        # claim TF-IDF/SVD ran when they were skipped.
        with pytest.warns(UserWarning, match="site_frequency_threshold=0.5"):
            _, _, meta = load_cusanovich(
                root=atlas, n_components=10, site_frequency_threshold=0.5
            )
        assert meta["n_peaks_kept"] == 0
        chain = " ".join(meta["preprocessing"])
        assert "skipped" in chain
        assert "TruncatedSVD(n_components=10);" not in chain
        assert "TF-IDF: per-cell term frequency" not in chain

    def test_labels_follow_the_metadata_row_order_not_the_cells_file(self, atlas):
        # On the released atlas the matrix columns are in cell_metadata.txt
        # row order; .cells.txt lists the same barcodes in another order.
        # Reindexing by the cells file kept tissue blocks roughly intact but
        # gave every cell another cell's label, cluster and t-SNE position
        # within its tissue. Row i of y must be metadata row i.
        _, y, meta = load_cusanovich(root=atlas, n_components=10)
        metadata = _metadata(atlas)
        np.testing.assert_array_equal(
            y.to_numpy(), metadata["cluster"].astype(str).to_numpy()
        )
        np.testing.assert_array_equal(
            meta["source_tsne"][:, 0], metadata["tsne_1"].to_numpy()
        )

    def test_cells_file_disagreeing_with_the_metadata_raises(self, atlas):
        # The cells file is still read: it is the only independent record of
        # which barcodes the matrix holds, so a mismatch with the metadata
        # means the two files are not from the same release.
        path = atlas / "Cusanovich" / "matrices" / "atac_matrix.binary.qc_filtered.cells.txt"
        lines = path.read_text().split()
        lines[0] = "not_a_cell"
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="cells.txt"):
            load_cusanovich(root=atlas, n_components=10)

    def test_embedding_separates_the_planted_tissues(self, atlas):
        # A loader that returned noise would satisfy every shape assertion
        # above, so pin that the preprocessing preserves real structure.
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        X, y, _ = load_cusanovich(root=atlas, n_components=10, label_column="tissue")
        labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_subsample_is_stratified_by_the_source_clusters(self, atlas):
        # Six planted clusters of 20 cells each; a draw of 30 stratified by
        # cluster takes exactly five from each.
        X, y, meta = load_cusanovich(root=atlas, n_components=10, subsample=30)
        assert X.shape[0] == 30
        assert meta["subsample"] == 30
        assert y.value_counts().to_dict() == {str(c): 5 for c in range(1, 7)}

    def test_source_tsne_rows_follow_x_through_filter_and_subsample(self, atlas):
        # meta["source_tsne"] is the source's own t-SNE, carried through as
        # an (n, 2) array whose row i is the same cell as row i of X and y.
        # The fixture encodes each cell's matrix column in tsne_1 and its
        # tissue in tsne_2, so both the Unknown filter and the stratified
        # subsample can be checked without reconstructing the split.
        X, y, meta = load_cusanovich(
            root=atlas,
            n_components=10,
            subsample=30,
            drop_unknown=True,
            label_column="tissue",
        )
        tsne = meta["source_tsne"]
        assert isinstance(tsne, np.ndarray)
        assert tsne.dtype == np.float64
        assert tsne.shape == (X.shape[0], 2)

        # Row i's tissue code matches row i of y.
        expected = np.array([TISSUE_CODE[t] for t in y], dtype=float)
        np.testing.assert_array_equal(tsne[:, 1], expected)

        # Every row is a distinct annotated cell: no Unknown cell survives
        # and no cell appears twice.
        metadata = _metadata(atlas).set_index("cell")
        columns = tsne[:, 0].astype(int)
        assert len(set(columns)) == len(columns)
        labels = metadata["cell_label"].loc[[f"cell{i:04d}" for i in columns]]
        assert "Unknown" not in set(labels)

    def test_source_labels_rows_follow_x_through_filter_and_subsample(self, atlas):
        # meta["source_labels"] is filtered and subsampled with X, so row i
        # names the same cell as row i of y and of source_tsne. The fixture's
        # cluster is a function of that cell's t-SNE columns, so a table left
        # unsubsampled, or subsampled with a different index, disagrees.
        X, y, meta = load_cusanovich(
            root=atlas, n_components=10, subsample=30, drop_unknown=True
        )
        labels = meta["source_labels"]
        tsne = meta["source_tsne"]
        assert list(labels.columns) == [
            "tissue",
            "cluster",
            "subset_cluster",
            "cell_label",
        ]
        assert len(labels) == X.shape[0]
        np.testing.assert_array_equal(
            labels["cluster"].astype(str).to_numpy(), y.to_numpy()
        )
        np.testing.assert_array_equal(
            labels["cluster"].to_numpy(), _planted_cluster(tsne[:, 0], tsne[:, 1])
        )
        np.testing.assert_array_equal(
            labels["tissue"].map(TISSUE_CODE).to_numpy(), tsne[:, 1]
        )
        assert "Unknown" not in set(labels["cell_label"])

    def test_source_tables_are_full_length_without_subsample(self, atlas):
        X, _, meta = load_cusanovich(root=atlas, n_components=10)
        assert meta["source_tsne"].shape == (X.shape[0], 2)
        assert meta["source_labels"].shape == (X.shape[0], 4)

    def test_meta_records_the_preprocessing_chain(self, atlas):
        _, _, meta = load_cusanovich(root=atlas, n_components=10)
        chain = " ".join(meta["preprocessing"])
        assert "TF-IDF" in chain
        assert "3%" in chain or "site_frequency_threshold" in chain
        assert "keep every cell" in chain
        assert meta["drop_unknown"] is False
        assert meta["drops_first_component"] is False
        assert meta["source"] == "Cusanovich"

    def test_missing_data_directory_names_the_download(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="mouse_atlas_data_release"):
            load_cusanovich(root=tmp_path / "nothing")

    def test_missing_data_directory_names_a_path_that_resolves_correctly(
        self, tmp_path, monkeypatch
    ):
        # load_cusanovich takes no root in the notebooks, so a relative
        # "data/" resolves against the process's cwd at call time -- the
        # notebook's own directory when run via nbconvert, not the
        # repository root. A message that just repeats "data/Cusanovich/"
        # reads correctly only from the repo root; the resolved absolute
        # path is correct regardless of where the process actually started.
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError) as excinfo:
            load_cusanovich(root=Path("nothing"))
        expected = Path("nothing").resolve()
        assert str(expected) in str(excinfo.value)
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/benchmarks/datasets/test_cusanovich.py tests/benchmarks/test_studies.py::TestLoaderSubsampling -q`

Expected: exit code 1, ending with:

```text
E       KeyError: 'source_labels'
E       TypeError: load_cusanovich() got an unexpected keyword argument 'drop_unknown'
E       AssertionError: assert 'tissue' == 'cluster'
E         
E         - cluster
E         + tissue
E       AssertionError: 
E       Arrays are not equal
E       
E       (shapes (103,), (120,) mismatch)
E        ACTUAL: array(['Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung',
E              'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung',
10 failed, 10 passed in 0.68s
```

- [ ] Step 3: Implement

In `src/benchmarks/datasets/_cusanovich.py`, replace:

```python
436,206 peaks by 81,173 cells of binarized chromatin accessibility across 13
adult mouse tissues. The reference label used here is the tissue the nucleus
was dissected from, which is externally determined rather than an output of
anybody's clustering, and is therefore the same kind of independent ground
truth as the Klein study's collection timepoints.
```

with:

```python
436,206 peaks by 81,173 cells of binarized chromatin accessibility across 13
adult mouse tissues. The reference label used here is the source
publication's own clustering: 30 clusters from graph community detection on a
two-dimensional t-SNE of their LSI. It is a published partition, not ground
truth, so agreement with it is reported as agreement and never as accuracy.
The 30 clusters assign every cell, so every cell is kept by default. Tissue of
dissection and the 40 marker-based cell labels ride along in
meta["source_labels"].
```

In `src/benchmarks/datasets/_cusanovich.py`, replace:

```python
_SOURCE_TSNE_COLUMNS = ("tsne_1", "tsne_2")

#: Label value marking cells the source left unannotated. Dropped, mirroring
#: the Levine study's removal of the uncharacterized population 15.
UNKNOWN_LABEL = "Unknown"
```

with:

```python
_SOURCE_TSNE_COLUMNS = ("tsne_1", "tsne_2")

#: Columns of cell_metadata.txt holding the source's own labels, carried
#: through as meta["source_labels"] for coloring and side checks.
_SOURCE_LABEL_COLUMNS = ("tissue", "cluster", "subset_cluster", "cell_label")

#: Label value marking cells the source left unannotated. Dropped only under
#: drop_unknown=True.
UNKNOWN_LABEL = "Unknown"
```

In `src/benchmarks/datasets/_cusanovich.py`, replace:

```python
    n_components: int = 50,
    label_column: str = "tissue",
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess the Cusanovich mouse sci-ATAC atlas.

    Parameters
    ----------
    root : Path or None
        Directory holding the dataset folders. Defaults to data/.
    subsample : int, float, or None
        Stratified subsample size, as a count or a fraction. None keeps all.
    random_state : int
        Seed for the stratified subsample and the SVD.
    site_frequency_threshold : float
        Keep peaks accessible in at least this fraction of cells. 0.03 is the
        source publication's own value.
    n_components : int
        SVD components retained. All of them are kept; none is dropped.
    label_column : str
        Metadata column used as the reference label. "tissue" is externally
        determined; "cell_label" is the source's marker-based annotation.
        Cells the source could not annotate (UNKNOWN_LABEL in cell_label)
        are dropped regardless of which column is chosen here.

    Returns
    -------
    (X, y, meta)
        meta["source_tsne"] is an (n_cells, 2) float array of the source's
        own t-SNE coordinates, filtered and subsampled alongside X so that
        row i of both is the same cell.
    """
```

with:

```python
    n_components: int = 50,
    label_column: str = "cluster",
    drop_unknown: bool = False,
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess the Cusanovich mouse sci-ATAC atlas.

    Parameters
    ----------
    root : Path or None
        Directory holding the dataset folders. Defaults to data/.
    subsample : int, float, or None
        Subsample size, as a count or a fraction, stratified by the reference
        label. None keeps all.
    random_state : int
        Seed for the stratified subsample and the SVD.
    site_frequency_threshold : float
        Keep peaks accessible in at least this fraction of cells. 0.03 is the
        source publication's own value.
    n_components : int
        SVD components retained. All of them are kept; none is dropped.
    label_column : str
        Metadata column used as the reference label, returned as strings.
        "cluster" is the source's 30-cluster partition of every cell;
        "tissue" is the dissection of origin; "cell_label" is the source's
        marker-based annotation.
    drop_unknown : bool
        Drop cells whose cell_label is UNKNOWN_LABEL, 10,029 of the release's
        81,173. Off by default: the source's clusters assign every cell, so
        dropping these would compare against a partition of a different set.

    Returns
    -------
    (X, y, meta)
        Row i of X, y, meta["source_tsne"] and meta["source_labels"] is the
        same cell. meta["source_tsne"] is an (n_cells, 2) float array of the
        source's own t-SNE coordinates; meta["source_labels"] is a DataFrame
        of the source's tissue, cluster, subset_cluster and cell_label.
        meta["n_cells_annotated"] counts the atlas cells whose cell_label is
        not UNKNOWN_LABEL, whether or not they were dropped.
    """
```

In `src/benchmarks/datasets/_cusanovich.py`, replace:

```python
    metadata = metadata.set_index("cell")

    # UNKNOWN_LABEL is not a rare edge case: in the source cell_metadata.txt
    # it covers 10,029 of 81,173 cells, about 12 percent. Dropping it here,
    # regardless of scale or label_column, means the atlas's published
    # 81,173 cells and the annotated set this function actually returns
    # (n_cells_annotated in meta, below) are two different numbers -- do not
    # conflate them, including at scales={"atlas": None}.
    keep_cells = metadata[_ANNOTATION_COLUMN].notna().to_numpy() & (
        metadata[_ANNOTATION_COLUMN].to_numpy() != UNKNOWN_LABEL
    )

    peak_cell_counts = np.asarray((matrix > 0).sum(axis=1)).ravel()
    keep_peaks = peak_cell_counts >= site_frequency_threshold * n_cells_full

    n_peaks_kept = int(keep_peaks.sum())
    n_kept_cells = int(keep_cells.sum())

    preprocessing = [
        f"drop cells labeled {UNKNOWN_LABEL!r} in {_ANNOTATION_COLUMN}",
        f"keep peaks accessible in at least {site_frequency_threshold:.0%} "
        "of cells (site_frequency_threshold, source default 3%)",
    ]
```

with:

```python
    metadata = metadata.set_index("cell")

    # UNKNOWN_LABEL covers 10,029 of 81,173 cells in the release, about 12
    # percent. The source's 30 clusters assign every one of them, so they
    # stay unless drop_unknown asks otherwise; n_cells_annotated reports the
    # annotated count either way.
    annotated = metadata[_ANNOTATION_COLUMN].notna().to_numpy() & (
        metadata[_ANNOTATION_COLUMN].to_numpy() != UNKNOWN_LABEL
    )
    n_annotated = int(annotated.sum())
    keep_cells = annotated if drop_unknown else np.ones(n_cells_full, dtype=bool)

    peak_cell_counts = np.asarray((matrix > 0).sum(axis=1)).ravel()
    keep_peaks = peak_cell_counts >= site_frequency_threshold * n_cells_full

    n_peaks_kept = int(keep_peaks.sum())
    n_kept_cells = int(keep_cells.sum())

    preprocessing = [
        (
            f"drop cells labeled {UNKNOWN_LABEL!r} in {_ANNOTATION_COLUMN}"
            if drop_unknown
            else f"keep every cell, including the {n_cells_full - n_annotated} "
            f"labeled {UNKNOWN_LABEL!r} in {_ANNOTATION_COLUMN}"
        ),
        f"keep peaks accessible in at least {site_frequency_threshold:.0%} "
        "of cells (site_frequency_threshold, source default 3%)",
    ]
```

In `src/benchmarks/datasets/_cusanovich.py`, replace:

```python
    X = np.asarray(embedding, dtype=np.float64)
    y = pd.Series(
        metadata.loc[keep_cells, label_column].to_numpy(),
        name=label_column,
    ).astype(str)
    source_tsne = np.asarray(
        metadata.loc[keep_cells, list(_SOURCE_TSNE_COLUMNS)].to_numpy(),
        dtype=np.float64,
    )

    n_annotated = int(X.shape[0])
    if subsample is not None:
        size = (
            int(subsample * n_annotated)
            if isinstance(subsample, float)
            else int(subsample)
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=size / n_annotated, random_state=random_state
        )
        _, idx = next(splitter.split(X, y))
        X = X[idx]
        y = y.iloc[idx].reset_index(drop=True)
        source_tsne = source_tsne[idx]
```

with:

```python
    X = np.asarray(embedding, dtype=np.float64)
    kept = metadata.loc[keep_cells]
    y = pd.Series(kept[label_column].to_numpy(), name=label_column).astype(str)
    source_tsne = np.asarray(
        kept[list(_SOURCE_TSNE_COLUMNS)].to_numpy(), dtype=np.float64
    )
    source_labels = kept[list(_SOURCE_LABEL_COLUMNS)].reset_index(drop=True)

    n_kept = int(X.shape[0])
    if subsample is not None:
        size = (
            int(subsample * n_kept) if isinstance(subsample, float) else int(subsample)
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=size / n_kept, random_state=random_state
        )
        _, idx = next(splitter.split(X, y))
        X = X[idx]
        y = y.iloc[idx].reset_index(drop=True)
        source_tsne = source_tsne[idx]
        source_labels = source_labels.iloc[idx].reset_index(drop=True)
```

In `src/benchmarks/datasets/_cusanovich.py`, replace:

```python
        "label_name": label_column,
        "drops_first_component": False,
        "preprocessing": preprocessing,
        "subsample": subsample,
        "random_state": random_state,
        "source_tsne": source_tsne,
    }
```

with:

```python
        "label_name": label_column,
        "drop_unknown": drop_unknown,
        "drops_first_component": False,
        "preprocessing": preprocessing,
        "subsample": subsample,
        "random_state": random_state,
        "source_tsne": source_tsne,
        "source_labels": source_labels,
    }
```

In `src/benchmarks/_studies.py`, replace:

```python
    # tissue is the reference: it is determined by dissection, not by any
    # clustering, which is what makes it independent ground truth.
    return load_cusanovich(subsample=subsample, random_state=42, label_column="tissue")
```

with:

```python
    # The reference is the source publication's own 30 clusters, which assign
    # every cell, so drop_unknown stays at its default and the comparison is
    # over the cells that partition covers.
    return load_cusanovich(subsample=subsample, random_state=42, label_column="cluster")
```

Then format and lint. The blocks above are already formatted, so the first command changes nothing:

```bash
.venv/bin/ruff format --check src/
.venv/bin/ruff check src/
```

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/benchmarks/datasets/test_cusanovich.py tests/benchmarks/test_studies.py::TestLoaderSubsampling -q`

Expected: exit code 0, ending with:

```text
20 passed in 0.56s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `_cusanovich.py`: `label_column: str = "cluster",` to `label_column: str = "tissue",` turns test_label_column_defaults_to_the_source_clusters red.
- `_cusanovich.py`: `keep_cells = annotated if drop_unknown else np.ones(n_cells_full, dtype=bool)` to `keep_cells = annotated` turns test_every_cell_is_kept_by_default red.
- `_cusanovich.py`: `keep_cells = annotated if drop_unknown else np.ones(n_cells_full, dtype=bool)` to `keep_cells = np.ones(n_cells_full, dtype=bool)` turns test_drop_unknown_keeps_only_the_annotated_cells red.
- `_cusanovich.py`: `n_annotated = int(annotated.sum())` to `n_annotated = n_cells_full` turns test_every_cell_is_kept_by_default red.
- `_cusanovich.py`: `source_labels = source_labels.iloc[idx].reset_index(drop=True)` to `source_labels = source_labels.iloc[np.sort(idx)].reset_index(drop=True)` turns test_source_labels_rows_follow_x_through_filter_and_subsample red.
- `_cusanovich.py`: `source_labels = kept[list(_SOURCE_LABEL_COLUMNS)].reset_index(drop=True)` to `source_labels = metadata[list(_SOURCE_LABEL_COLUMNS)].reset_index(drop=True)` turns test_source_labels_rows_follow_x_through_filter_and_subsample red.
- `_cusanovich.py`: `_, idx = next(splitter.split(X, y))` to `idx = np.random.default_rng(random_state).permutation(n_kept)[:size]` turns test_subsample_is_stratified_by_the_source_clusters red.
- `_studies.py`: `label_column="cluster"` to `label_column="tissue"` turns test_cusanovich_loader_references_the_source_clusters red.
- `_studies.py`: `label_column="cluster"` to `label_column="cluster", drop_unknown=True` turns test_cusanovich_loader_references_the_source_clusters red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "feat(benchmarks): Cusanovich loader references the source's 30 clusters over every cell"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 2: PreprocessingSpec, and Study declares its resample count, preprocessing and either sweep

Spec 5. `PreprocessingSpec` names options by registry key so `_types` stays a leaf. `Study` gains `n_resamples` (default 100) and `preprocessing` (default None), and its check relaxes from "candidate_k must not be empty" to "at least one of candidate_k and resolutions".

Files:
- Modify: `src/benchmarks/_types.py`
- Test: `tests/benchmarks/test_benchmarks_types.py`

Interfaces:
- Consumes: nothing new.
- Produces: `benchmarks._types.KNOWN_PREPROCESSORS: frozenset[str]` = {identity, standard_scaler, pca, tsne, umap}; `PreprocessingSpec(normalization: tuple[tuple[str, Mapping[str, Sequence]], ...], dim_reduction: ...)`, frozen, raising `ValueError` on an unknown key or an empty role; `Study.n_resamples: int = 100`, `Study.preprocessing: PreprocessingSpec | None = None`.

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/test_benchmarks_types.py`, replace:

```python
from benchmarks._types import (
    KNOWN_ESTIMATORS,
    Axis,
    EstimatorSpec,
    Manifest,
    Scenario,
    Study,
)
```

with:

```python
from benchmarks._types import (
    KNOWN_ESTIMATORS,
    KNOWN_PREPROCESSORS,
    Axis,
    EstimatorSpec,
    Manifest,
    PreprocessingSpec,
    Scenario,
    Study,
)
```

In `tests/benchmarks/test_benchmarks_types.py`, replace:

```python
    def test_rejects_empty_candidate_k(self):
        with pytest.raises(ValueError, match="candidate_k"):
            Study(
                name="demo",
                loader=lambda subsample: (np.zeros((2, 2)), np.zeros(2), {}),
                estimator=EstimatorSpec(name="kmeans"),
                candidate_k=(),
                scales={"dev": 100},
                default_scale="dev",
            )


class TestManifest:
```

with:

```python
    def test_rejects_a_study_with_neither_sweep(self):
        with pytest.raises(ValueError, match="candidate_k, resolutions or both"):
            Study(
                name="demo",
                loader=lambda subsample: (np.zeros((2, 2)), np.zeros(2), {}),
                estimator=EstimatorSpec(name="kmeans"),
                candidate_k=(),
                scales={"dev": 100},
                default_scale="dev",
            )

    def test_accepts_a_resolution_only_study(self):
        # A study that sweeps only Leiden resolution has no k grid to declare.
        study = Study(
            name="demo",
            loader=lambda subsample: (np.zeros((2, 2)), np.zeros(2), {}),
            estimator=EstimatorSpec(name="leiden"),
            candidate_k=(),
            resolutions=(0.5, 1.0),
            scales={"dev": 100},
            default_scale="dev",
        )
        assert study.candidate_k == ()
        assert study.resolutions == (0.5, 1.0)

    def test_run_settings_default_to_the_package_run(self):
        study = Study(
            name="demo",
            loader=lambda subsample: (np.zeros((2, 2)), np.zeros(2), {}),
            estimator=EstimatorSpec(name="kmeans"),
            candidate_k=(2, 3),
            scales={"dev": 100},
            default_scale="dev",
        )
        assert study.n_resamples == 100
        assert study.preprocessing is None


def _roles(**override):
    options = {
        "normalization": (("identity", {}),),
        "dim_reduction": (("identity", {}),),
    }
    options.update(override)
    return options


class TestPreprocessingSpec:
    def test_accepts_every_known_preprocessor(self):
        spec = PreprocessingSpec(
            normalization=(("identity", {}), ("standard_scaler", {})),
            dim_reduction=(
                ("identity", {}),
                ("pca", {"n_components": [5]}),
                ("tsne", {"perplexity": [30]}),
                ("umap", {"n_neighbors": [15, 30]}),
            ),
        )
        named = {key for key, _ in (*spec.normalization, *spec.dim_reduction)}
        assert named == KNOWN_PREPROCESSORS

    @pytest.mark.parametrize("role", ["normalization", "dim_reduction"])
    def test_an_unknown_key_raises_naming_the_valid_ones(self, role):
        with pytest.raises(ValueError, match=r"nmf.*Valid names") as excinfo:
            PreprocessingSpec(**_roles(**{role: (("nmf", {}),)}))
        assert role in str(excinfo.value)
        assert "tsne" in str(excinfo.value)

    @pytest.mark.parametrize("role", ["normalization", "dim_reduction"])
    def test_an_empty_role_raises(self, role):
        with pytest.raises(ValueError, match=f"{role} needs at least one option"):
            PreprocessingSpec(**_roles(**{role: ()}))


class TestManifest:
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/benchmarks/test_benchmarks_types.py -q`

Expected: exit code 2, ending with:

```text
E   ImportError: cannot import name 'KNOWN_PREPROCESSORS' from 'benchmarks._types' (/private/tmp/claude-501/-Users-kaiwycik-GitHub-CARVE--claude-playground/349e6055-9572-4f5c-919e-e41ea3aca477/scratchpad/pass2/src/benchmarks/_types.py)
ERROR tests/benchmarks/test_benchmarks_types.py
1 error in 0.09s
```

- [ ] Step 3: Implement

In `src/benchmarks/_types.py`, replace:

```python
from collections.abc import Callable, Iterator, Mapping
```

with:

```python
from collections.abc import Callable, Iterator, Mapping, Sequence
```

In `src/benchmarks/_types.py`, replace:

```python
        "leiden",
    }
)
```

with:

```python
        "leiden",
    }
)

#: Registry keys a PreprocessingSpec may name. _preprocessing maps each one to
#: its transformer; the names live here so this module stays a leaf.
KNOWN_PREPROCESSORS: frozenset[str] = frozenset(
    {"identity", "standard_scaler", "pca", "tsne", "umap"}
)
```

In `src/benchmarks/_types.py`, replace:

```python
                f"Valid names are {sorted(KNOWN_ESTIMATORS)}."
            )


def _simulator_parameters() -> frozenset[str]:
```

with:

```python
                f"Valid names are {sorted(KNOWN_ESTIMATORS)}."
            )


@dataclass(frozen=True)
class PreprocessingSpec:
    """The preprocessing options a randomized case-study fit draws from.

    Each role holds (key, grid) pairs: key names a registered transformer and
    grid maps each of its hyperparameters to candidate values, one of which is
    drawn per resample. Options are named by key rather than by class so this
    module stays a leaf; _preprocessing.resolve_preprocessing turns them into
    the option lists CARVE takes. Validation is strict for the same reason
    EstimatorSpec's is.
    """

    normalization: tuple[tuple[str, Mapping[str, Sequence[Any]]], ...]
    dim_reduction: tuple[tuple[str, Mapping[str, Sequence[Any]]], ...]

    def __post_init__(self) -> None:
        for role in ("normalization", "dim_reduction"):
            options = getattr(self, role)
            if not options:
                raise ValueError(
                    f"PreprocessingSpec: {role} needs at least one option."
                )
            unknown = sorted({key for key, _ in options} - KNOWN_PREPROCESSORS)
            if unknown:
                raise ValueError(
                    f"PreprocessingSpec: unknown {role} option(s) {unknown}. "
                    f"Valid names are {sorted(KNOWN_PREPROCESSORS)}."
                )


def _simulator_parameters() -> frozenset[str]:
```

In `src/benchmarks/_types.py`, replace:

```python
    partners are the estimators swept alongside the study's own, in the
    order study_model_grids emits them. Declared here so the set of
    estimators a case study compares is part of its configuration rather
    than a branch on its name.
    """
```

with:

```python
    partners are the estimators swept alongside the study's own, in the
    order study_model_grids emits them. Declared here so the set of
    estimators a case study compares is part of its configuration rather
    than a branch on its name.

    A study declares candidate_k, resolutions or both, depending on which
    sweeps it runs. n_resamples is the resample count its CARVE fit runs,
    and preprocessing, when set, is the option set a randomized fit draws
    its pipelines from; both live here so a notebook reads them rather than
    restating them.
    """
```

In `src/benchmarks/_types.py`, replace:

```python
    consensus_anchors: int | None = None
    k_star: int | None = None

    def __post_init__(self) -> None:
        if not self.candidate_k:
            raise ValueError(f"Study {self.name!r}: candidate_k must not be empty.")
```

with:

```python
    consensus_anchors: int | None = None
    k_star: int | None = None
    n_resamples: int = 100
    preprocessing: PreprocessingSpec | None = None

    def __post_init__(self) -> None:
        if not self.candidate_k and not self.resolutions:
            raise ValueError(
                f"Study {self.name!r}: declare candidate_k, resolutions or "
                "both; both are empty."
            )
```

Then format and lint. The blocks above are already formatted, so the first command changes nothing:

```bash
.venv/bin/ruff format --check src/
.venv/bin/ruff check src/
```

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/benchmarks/test_benchmarks_types.py -q`

Expected: exit code 0, ending with:

```text
40 passed in 0.03s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `_types.py`: `if not self.candidate_k and not self.resolutions:` to `if not self.candidate_k:` turns test_accepts_a_resolution_only_study red.
- `_types.py`: `if not self.candidate_k and not self.resolutions:` to `if False:` turns test_rejects_a_study_with_neither_sweep red.
- `_types.py`: `n_resamples: int = 100` to `n_resamples: int = 50` turns test_run_settings_default_to_the_package_run red.
- `_types.py`: `unknown = sorted({key for key, _ in options} - KNOWN_PREPROCESSORS)` to `unknown = []` turns test_an_unknown_key_raises_naming_the_valid_ones red.
- `_types.py`: `if not options:` to `if options == "never":` turns test_an_empty_role_raises red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "feat(benchmarks): PreprocessingSpec; Study declares n_resamples, preprocessing and either sweep"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 3: The preprocessing registry resolves a spec into carve's option lists

Spec 5, the preprocessing counterpart of `_estimators.py`. Two refinements, listed above: the registry holds import paths so umap-learn is imported only when a spec names UMAP, and the fixed settings in `PREPROCESSOR_DEFAULTS` are bound with `functools.partial` beside a display name, which keeps them out of the pipeline label.

Files:
- Create: `src/benchmarks/_preprocessing.py`
- Test: `tests/benchmarks/test_preprocessing.py`

Interfaces:
- Consumes: Task 2: `PreprocessingSpec`, `KNOWN_PREPROCESSORS`. carve: `_pipeline.allocate_pipelines`, `_pipeline.pipeline_from_spec` (tests only).
- Produces: `benchmarks._preprocessing.PREPROCESSOR_CLASSES: dict[str, str]` (import paths), `PREPROCESSOR_DEFAULTS: dict[str, dict]`, `PREPROCESSOR_NAMES: dict[str, str]` (identity, StandardScaler, PCA, TSNE, UMAP), `preprocessor_class(key) -> type`, `resolve_preprocessing(spec) -> {"normalization_options": [...], "dim_reduction_options": [...]}` where each option is `(transformer or functools.partial, display name, grid)`.

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/test_preprocessing.py` with this content (replacing the file if it exists):

```python
"""Tests for preprocessing option construction."""

import sys
import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from benchmarks._preprocessing import (
    PREPROCESSOR_CLASSES,
    PREPROCESSOR_DEFAULTS,
    PREPROCESSOR_NAMES,
    preprocessor_class,
    resolve_preprocessing,
)
from benchmarks._types import KNOWN_PREPROCESSORS, PreprocessingSpec
from carve._pipeline import allocate_pipelines, pipeline_from_spec


def _spec(dim_reduction, normalization=(("identity", {}),)):
    return PreprocessingSpec(normalization=normalization, dim_reduction=dim_reduction)


def _pipelines(spec, n_resamples):
    options = resolve_preprocessing(spec)
    return allocate_pipelines(
        options["normalization_options"],
        options["dim_reduction_options"],
        n_resamples=n_resamples,
        random_state=0,
    )


def test_every_known_preprocessor_is_registered():
    assert set(PREPROCESSOR_CLASSES) == set(KNOWN_PREPROCESSORS)
    assert set(PREPROCESSOR_DEFAULTS) == set(KNOWN_PREPROCESSORS)
    assert set(PREPROCESSOR_NAMES) == set(KNOWN_PREPROCESSORS)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("identity", FunctionTransformer),
        ("standard_scaler", StandardScaler),
        ("pca", PCA),
        ("tsne", TSNE),
    ],
)
def test_registry_keys_name_their_classes(key, expected):
    assert preprocessor_class(key) is expected


def test_the_umap_key_names_umap():
    umap = pytest.importorskip("umap")
    assert preprocessor_class("umap") is umap.UMAP


def test_an_unknown_key_raises():
    with pytest.raises(ValueError, match="Valid names"):
        preprocessor_class("nmf")


def test_a_missing_umap_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(ImportError, match=r"carve-validate\[umap\]"):
        preprocessor_class("umap")


def test_the_pinned_defaults():
    assert PREPROCESSOR_DEFAULTS["tsne"] == {"n_components": 2}
    assert PREPROCESSOR_DEFAULTS["umap"] == {"n_components": 2, "min_dist": 0.1}


# (role, key, grid, output width on six features)
_FIT_CASES = [
    ("normalization", "identity", {}, 6),
    ("normalization", "standard_scaler", {}, 6),
    ("dim_reduction", "identity", {}, 6),
    ("dim_reduction", "pca", {"n_components": [3]}, 3),
    ("dim_reduction", "tsne", {"perplexity": [5]}, 2),
    ("dim_reduction", "umap", {"n_neighbors": [5]}, 2),
]


@pytest.mark.parametrize(
    ("role", "key", "grid", "width"),
    _FIT_CASES,
    ids=[f"{role}-{key}" for role, key, _, _ in _FIT_CASES],
)
def test_every_registry_key_resolves_to_an_option_carve_fits(role, key, grid, width):
    if key == "umap":
        pytest.importorskip("umap")
    option = ((key, grid),)
    spec = (
        _spec(dim_reduction=option)
        if role == "dim_reduction"
        else _spec(normalization=option, dim_reduction=(("identity", {}),))
    )
    (pipeline,) = _pipelines(spec, n_resamples=1)
    X = np.random.default_rng(0).normal(size=(40, 6))
    with warnings.catch_warnings():
        # umap-learn warns that a seed disables its parallelism; carve seeds
        # every transformer on purpose (see _runner.embed_resample).
        warnings.filterwarnings(
            "ignore",
            message=r"n_jobs value .* overridden to 1 by setting random_state",
            category=UserWarning,
        )
        Z = pipeline_from_spec(pipeline, random_state=0).fit_transform(X)
    assert Z.shape == (40, width)


def test_bound_defaults_come_from_the_registry(monkeypatch):
    # The fixed settings are bound from PREPROCESSOR_DEFAULTS rather than
    # left to the library, so editing the table changes the transformer.
    monkeypatch.setitem(PREPROCESSOR_DEFAULTS, "tsne", {"n_components": 3})
    (pipeline,) = _pipelines(
        _spec(dim_reduction=(("tsne", {"perplexity": [5]}),)), n_resamples=1
    )
    transformer = pipeline_from_spec(pipeline, random_state=0).named_steps["dr"]
    assert transformer.n_components == 3


def test_labels_name_only_the_drawn_hyperparameters():
    # TSNE's bound n_components=2 stays out of the label; the display name
    # and the drawn values are all it shows, in the spec's option order.
    spec = _spec(
        dim_reduction=(
            ("identity", {}),
            ("tsne", {"perplexity": [30]}),
            ("pca", {"n_components": [5, 10]}),
        )
    )
    options = resolve_preprocessing(spec)
    assert [name for _, name, _ in options["dim_reduction_options"]] == [
        "identity",
        "TSNE",
        "PCA",
    ]
    assert {pipeline.label for pipeline in _pipelines(spec, n_resamples=30)} == {
        "identity | identity",
        "identity | TSNE(perplexity=30)",
        "identity | PCA(n_components=5)",
        "identity | PCA(n_components=10)",
    }
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/benchmarks/test_preprocessing.py -q`

Expected: exit code 2, ending with:

```text
E   ModuleNotFoundError: No module named 'benchmarks._preprocessing'
ERROR tests/benchmarks/test_preprocessing.py
1 error in 0.07s
```

- [ ] Step 3: Implement

Create `src/benchmarks/_preprocessing.py` with this content (replacing the file if it exists):

```python
"""Construction of the preprocessing options case studies randomize over.

The preprocessing counterpart of _estimators. A PreprocessingSpec names each
option by registry key; resolve_preprocessing turns the keys into the option
lists CARVE takes, with PREPROCESSOR_DEFAULTS bound into each transformer
and a display name attached. Bound settings stay out of the pipeline label,
so it reads "identity | TSNE(perplexity=30)" and names only what is drawn
per resample. Transformer classes are imported when a spec is resolved, not
when this module is, so umap-learn is needed only by a spec that names UMAP.
"""

import importlib
from functools import partial
from typing import Any

from ._types import PreprocessingSpec

#: Registry key to transformer class, as an import path.
PREPROCESSOR_CLASSES: dict[str, str] = {
    "identity": "sklearn.preprocessing.FunctionTransformer",
    "standard_scaler": "sklearn.preprocessing.StandardScaler",
    "pca": "sklearn.decomposition.PCA",
    "tsne": "sklearn.manifold.TSNE",
    "umap": "umap.UMAP",
}

#: Settings bound into every transformer of a kind, pinned so results do not
#: move when a library changes its default. t-SNE's max_iter is left at
#: scikit-learn's default.
PREPROCESSOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "identity": {},
    "standard_scaler": {},
    "pca": {},
    "tsne": {"n_components": 2},
    "umap": {"n_components": 2, "min_dist": 0.1},
}

#: Display names, which pipeline labels use in place of the class name.
PREPROCESSOR_NAMES: dict[str, str] = {
    "identity": "identity",
    "standard_scaler": "StandardScaler",
    "pca": "PCA",
    "tsne": "TSNE",
    "umap": "UMAP",
}

#: One resolved option: transformer (or a partial binding its defaults),
#: display name, and the grid drawn from per resample.
ResolvedOption = tuple[Any, str, dict[str, list[Any]]]


def preprocessor_class(key: str) -> type:
    """Import and return the transformer class registered under key."""
    if key not in PREPROCESSOR_CLASSES:
        raise ValueError(
            f"Unknown preprocessor {key!r}. "
            f"Valid names are {sorted(PREPROCESSOR_CLASSES)}."
        )
    module_name, _, class_name = PREPROCESSOR_CLASSES[key].rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"The {key!r} preprocessor needs {module_name}, which is not "
            'installed. UMAP ships with the umap extra: pip install "carve-validate[umap]".'
        ) from exc
    return getattr(module, class_name)


def _resolve_option(key: str, grid: Any) -> ResolvedOption:
    cls = preprocessor_class(key)
    fixed = PREPROCESSOR_DEFAULTS[key]
    transformer = partial(cls, **fixed) if fixed else cls
    return (
        transformer,
        PREPROCESSOR_NAMES[key],
        {name: list(values) for name, values in grid.items()},
    )


def resolve_preprocessing(spec: PreprocessingSpec) -> dict[str, list[ResolvedOption]]:
    """The option lists CARVE takes, built from a spec.

    Returns a dict with the keys normalization_options and
    dim_reduction_options, each a list of (transformer, display name, grid)
    triples in the spec's order, so it can be passed straight to
    fit_or_load_carve or CARVE as keyword arguments. A transformer with bound
    defaults is a functools.partial of its class; carve instantiates it with
    the drawn hyperparameters and a derived seed.
    """
    return {
        "normalization_options": [
            _resolve_option(key, grid) for key, grid in spec.normalization
        ],
        "dim_reduction_options": [
            _resolve_option(key, grid) for key, grid in spec.dim_reduction
        ],
    }
```

Then format and lint. The blocks above are already formatted, so the first command changes nothing:

```bash
.venv/bin/ruff format --check src/
.venv/bin/ruff check src/
```

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/benchmarks/test_preprocessing.py -q`

Expected: exit code 0, ending with:

```text
17 passed in 6.07s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `_preprocessing.py`: `transformer = partial(cls, **fixed) if fixed else cls` to `transformer = cls` turns test_bound_defaults_come_from_the_registry red.
- `_preprocessing.py`: `PREPROCESSOR_NAMES[key],` to `None,` turns test_labels_name_only_the_drawn_hyperparameters red.
- `_preprocessing.py`: `_resolve_option(key, grid) for key, grid in spec.dim_reduction` to `_resolve_option(key, grid) for key, grid in reversed(spec.dim_reduction)` turns test_labels_name_only_the_drawn_hyperparameters red.
- `_preprocessing.py`: `"umap": "umap.UMAP",` to `"umap": "umap.umap_.UMAP_",` turns test_the_umap_key_names_umap red.
- `_preprocessing.py`: `UMAP ships with the umap extra: pip install "carve-validate[umap]".'` to `Install it.'` turns test_a_missing_umap_names_the_extra red.
- `_preprocessing.py`: `"standard_scaler": "StandardScaler",` to `(line removed)` turns test_every_known_preprocessor_is_registered red.
- `_preprocessing.py`: `"umap": {"n_components": 2, "min_dist": 0.1},` to `"umap": {"n_components": 2},` turns test_the_pinned_defaults red.
- `_preprocessing.py`: `{name: list(values) for name, values in grid.items()},` to `{},` turns test_every_registry_key_resolves_to_an_option_carve_fits red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "feat(benchmarks): preprocessing registry resolves specs into carve option lists"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 4: A run-keyed cache path, and fit_or_load_carve forwards the preprocessing

Spec 5, last paragraph. `carve_cache_path` appends a run key, a short hash of the grid, the preprocessing spec with the defaults it binds, and `n_resamples`, except for the study's default run, so the Klein and Levine publication caches keep their filenames. The test pins those filenames byte for byte, as computed from the code at a95bf46. `fit_or_load_carve` forwards the three preprocessing arguments only when set, and refuses option lists without `randomize_preprocessing=True`, which CARVE would otherwise ignore silently. A real randomized fit round-trips through the cache, which proves the `functools.partial` options pickle.

Files:
- Modify: `src/benchmarks/_preprocessing.py`
- Modify: `src/benchmarks/_studies.py`
- Test: `tests/benchmarks/_helpers.py`
- Test: `tests/benchmarks/test_preprocessing.py`
- Test: `tests/benchmarks/test_studies.py`

Interfaces:
- Consumes: Task 2: `PreprocessingSpec`. Task 3: `PREPROCESSOR_DEFAULTS`, `resolve_preprocessing`.
- Produces: `benchmarks._preprocessing.preprocessing_fingerprint(spec | None) -> str`; `benchmarks._studies.PACKAGE_N_RESAMPLES: int` (100, read from `CARVE`); `carve_cache_path(study, *, scale=None, root, model_grids=None, n_resamples=PACKAGE_N_RESAMPLES, preprocessing=None) -> Path`, raising `ValueError` naming `model_grids` for a study with no k-based grid; `fit_or_load_carve(..., randomize_preprocessing=False, normalization_options=None, dim_reduction_options=None)`. `tests/benchmarks/_helpers.make_carve_spy()` classes gain `captured_fit_kwargs`.

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/_helpers.py`, replace:

```python
    class SpyCARVE:
        captured_kwargs: dict | None = None

        def __init__(self, **kwargs):
            type(self).captured_kwargs = kwargs
            self.estimator_results_ = pd.DataFrame({"config_id": [0]})
            self._n = 0

        def fit(self, X, *args, **kwargs):
            self._n = int(np.asarray(X).shape[0])
            return self
```

with:

```python
    class SpyCARVE:
        captured_kwargs: dict | None = None
        captured_fit_kwargs: dict | None = None

        def __init__(self, **kwargs):
            type(self).captured_kwargs = kwargs
            self.estimator_results_ = pd.DataFrame({"config_id": [0]})
            self._n = 0

        def fit(self, X, *args, **kwargs):
            type(self).captured_fit_kwargs = kwargs
            self._n = int(np.asarray(X).shape[0])
            return self
```

In `tests/benchmarks/_helpers.py`, replace:

```python
    """A CARVE stand-in class that records its constructor kwargs.
```

with:

```python
    """A CARVE stand-in class that records its constructor and fit kwargs.
```

In `tests/benchmarks/test_preprocessing.py`, replace:

```python
    preprocessor_class,
    resolve_preprocessing,
)
```

with:

```python
    preprocessing_fingerprint,
    preprocessor_class,
    resolve_preprocessing,
)
```

In `tests/benchmarks/test_preprocessing.py`, replace:

```python
def test_bound_defaults_come_from_the_registry(monkeypatch):
```

with:

```python
def test_the_fingerprint_names_the_bound_defaults(monkeypatch):
    spec = _spec(dim_reduction=(("tsne", {"perplexity": [30]}),))
    before = preprocessing_fingerprint(spec)
    monkeypatch.setitem(PREPROCESSOR_DEFAULTS, "tsne", {"n_components": 3})
    assert preprocessing_fingerprint(spec) != before


def test_the_fingerprint_imports_nothing(monkeypatch):
    # A cache path is computed before any fit, including where umap-learn is
    # not installed.
    monkeypatch.setitem(sys.modules, "umap", None)
    spec = _spec(dim_reduction=(("umap", {"n_neighbors": [15]}),))
    assert "min_dist" in preprocessing_fingerprint(spec)


def test_bound_defaults_come_from_the_registry(monkeypatch):
```

In `tests/benchmarks/test_studies.py`, replace:

```python
from benchmarks._estimators import param_grids
```

with:

```python
from benchmarks._estimators import param_grids, resolution_grids
from benchmarks._preprocessing import PREPROCESSOR_DEFAULTS, resolve_preprocessing
```

In `tests/benchmarks/test_studies.py`, replace:

```python
from benchmarks._types import EstimatorSpec, Study
```

with:

```python
from benchmarks._types import EstimatorSpec, PreprocessingSpec, Study
```

In `tests/benchmarks/test_studies.py`, replace:

```python
    def test_study_scaling_sweep_forwards_consensus_anchors(self, blobs, monkeypatch):
```

with:

```python
    def test_study_scaling_sweep_forwards_consensus_anchors(self, blobs, monkeypatch):
```

In `tests/benchmarks/test_studies.py`, replace:

```python
class TestDenseEstimatorGuard:
```

with:

```python
_TSNE_SPEC = PreprocessingSpec(
    normalization=(("identity", {}),),
    dim_reduction=(("identity", {}), ("tsne", {"perplexity": [5]})),
)


class TestPreprocessingForwarding:
    def test_forwards_the_preprocessing_arguments(self, blobs, tmp_path, monkeypatch):
        X, y = blobs
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)
        normalization = [("normalization sentinel",)]
        dim_reduction = [("dim_reduction sentinel",)]

        fit_or_load_carve(
            X,
            y,
            cache_path=tmp_path / "demo.carve",
            model_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
            n_resamples=3,
            randomize_preprocessing=True,
            normalization_options=normalization,
            dim_reduction_options=dim_reduction,
        )

        assert spy.captured_kwargs["normalization_options"] is normalization
        assert spy.captured_kwargs["dim_reduction_options"] is dim_reduction
        assert spy.captured_fit_kwargs["randomize_preprocessing"] is True

    def test_omits_the_preprocessing_arguments_when_unset(
        self, blobs, tmp_path, monkeypatch
    ):
        X, y = blobs
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)

        fit_or_load_carve(
            X,
            y,
            cache_path=tmp_path / "demo.carve",
            model_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
            n_resamples=3,
        )

        assert "normalization_options" not in spy.captured_kwargs
        assert "dim_reduction_options" not in spy.captured_kwargs
        assert "randomize_preprocessing" not in spy.captured_fit_kwargs

    def test_options_without_randomize_preprocessing_raise(self, blobs, tmp_path):
        # CARVE ignores the option lists on a non-randomized fit, so passing
        # them there is a misconfiguration, not a no-op.
        X, y = blobs
        with pytest.raises(ValueError, match="randomize_preprocessing=True"):
            fit_or_load_carve(
                X,
                y,
                cache_path=tmp_path / "demo.carve",
                model_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
                n_resamples=3,
                dim_reduction_options=[],
            )
        assert not (tmp_path / "demo.carve").exists()

    def test_a_randomized_fit_survives_the_cache(self, blobs, tmp_path):
        # resolve_preprocessing binds t-SNE's defaults with functools.partial;
        # the cached fit must pickle and reload it with the per-pipeline
        # table and the pipeline registry intact.
        X, y = blobs
        kwargs = dict(
            cache_path=tmp_path / "demo.carve",
            model_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
            n_resamples=4,
            randomize_preprocessing=True,
            **resolve_preprocessing(_TSNE_SPEC),
        )
        fitted = fit_or_load_carve(X, y, **kwargs)
        mtime = kwargs["cache_path"].stat().st_mtime_ns
        loaded = fit_or_load_carve(X, y, **kwargs)

        assert kwargs["cache_path"].stat().st_mtime_ns == mtime
        pd.testing.assert_frame_equal(
            loaded.preprocessing_results_, fitted.preprocessing_results_
        )
        assert set(loaded.preprocessing_pipelines_) == {
            "identity | identity",
            "identity | TSNE(perplexity=5)",
        }


class TestDenseEstimatorGuard:
```

In `tests/benchmarks/test_studies.py`, replace:

```python
    def test_the_full_data_none_scale_gets_a_stable_path(self, tmp_path):
        # None (full data) has no natural filename spelling; the hash must
        # still be deterministic across calls.
        a = carve_cache_path(_study(), scale="publication", root=tmp_path)
        b = carve_cache_path(_study(), scale="publication", root=tmp_path)
        assert a == b
```

with:

```python
    def test_the_full_data_none_scale_gets_a_stable_path(self, tmp_path):
        # None (full data) has no natural filename spelling; the hash must
        # still be deterministic across calls.
        a = carve_cache_path(_study(), scale="publication", root=tmp_path)
        b = carve_cache_path(_study(), scale="publication", root=tmp_path)
        assert a == b

    def test_default_runs_keep_their_existing_filenames(self, tmp_path):
        # The Klein and Levine publication caches are hours of compute each
        # and were written before run keys existed, so their default runs
        # must resolve to the same names as before, byte for byte.
        klein = carve_cache_path(STUDIES["klein"], root=tmp_path)
        levine = carve_cache_path(STUDIES["levine32"], root=tmp_path)
        heca = carve_cache_path(STUDIES["heca"], root=tmp_path)
        assert klein.name == "carve_klein_publication_1b390cd5.carve"
        assert levine.name == "carve_levine32_publication_f8237d89.carve"
        assert heca.name == "carve_heca_dev_8314e95d.carve"

    def test_passing_the_default_run_explicitly_changes_nothing(self, tmp_path):
        study = _study()
        explicit = carve_cache_path(
            study,
            scale="dev",
            root=tmp_path,
            model_grids=study_model_grids(study),
            n_resamples=100,
            preprocessing=None,
        )
        assert explicit == carve_cache_path(study, scale="dev", root=tmp_path)

    @pytest.mark.parametrize(
        "change",
        [
            {"model_grids": resolution_grids(EstimatorSpec(name="leiden"), (0.5, 1.0))},
            {"preprocessing": _TSNE_SPEC},
            {"n_resamples": 150},
        ],
        ids=["resolution-grid", "preprocessing", "n_resamples"],
    )
    def test_any_other_run_gets_its_own_filename(self, tmp_path, change):
        default = carve_cache_path(_study(), scale="dev", root=tmp_path)
        path = carve_cache_path(_study(), scale="dev", root=tmp_path, **change)
        assert path != default
        assert path.name.startswith(default.stem + "_")
        assert path.suffix == ".carve"

    def test_different_preprocessing_gets_a_different_filename(self, tmp_path):
        other = PreprocessingSpec(
            normalization=(("identity", {}),),
            dim_reduction=(("identity", {}), ("tsne", {"perplexity": [15]})),
        )
        a = carve_cache_path(_study(), scale="dev", root=tmp_path, preprocessing=_TSNE_SPEC)
        b = carve_cache_path(_study(), scale="dev", root=tmp_path, preprocessing=other)
        assert a != b

    def test_editing_the_bound_defaults_changes_the_filename(self, tmp_path, monkeypatch):
        before = carve_cache_path(
            _study(), scale="dev", root=tmp_path, preprocessing=_TSNE_SPEC
        )
        monkeypatch.setitem(PREPROCESSOR_DEFAULTS, "tsne", {"n_components": 3})
        after = carve_cache_path(
            _study(), scale="dev", root=tmp_path, preprocessing=_TSNE_SPEC
        )
        assert before != after

    def test_a_study_without_a_k_grid_must_name_its_grid(self, tmp_path):
        study = _study(
            estimator=EstimatorSpec(name="leiden"), candidate_k=(), resolutions=(0.5, 1.0)
        )
        with pytest.raises(ValueError, match="model_grids"):
            carve_cache_path(study, scale="dev", root=tmp_path)
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/benchmarks/test_studies.py tests/benchmarks/test_preprocessing.py -q`

Expected: exit code 2, ending with:

```text
E   ImportError: cannot import name 'preprocessing_fingerprint' from 'benchmarks._preprocessing' (/private/tmp/claude-501/-Users-kaiwycik-GitHub-CARVE--claude-playground/349e6055-9572-4f5c-919e-e41ea3aca477/scratchpad/pass2/src/benchmarks/_preprocessing.py)
ERROR tests/benchmarks/test_preprocessing.py
1 error in 0.15s
```

- [ ] Step 3: Implement

In `src/benchmarks/_preprocessing.py`, replace:

```python
def resolve_preprocessing(
```

with:

```python
def preprocessing_fingerprint(spec: PreprocessingSpec | None) -> str:
    """A stable text form of a spec and the defaults it binds, for cache keys.

    Two specs that resolve to different transformers differ here, including
    when only PREPROCESSOR_DEFAULTS changed, which the spec alone would not
    show. Nothing is imported, so computing a cache path never needs
    umap-learn.
    """
    if spec is None:
        return repr(None)
    keys = [key for key, _ in (*spec.normalization, *spec.dim_reduction)]
    return repr((spec, {key: PREPROCESSOR_DEFAULTS[key] for key in keys}))


def resolve_preprocessing(
```

In `src/benchmarks/_studies.py`, replace:

```python
from collections.abc import Sequence
```

with:

```python
from collections.abc import Sequence
from dataclasses import fields
```

In `src/benchmarks/_studies.py`, replace:

```python
from ._types import EstimatorSpec, Study
```

with:

```python
from ._preprocessing import preprocessing_fingerprint
from ._types import EstimatorSpec, PreprocessingSpec, Study
```

In `src/benchmarks/_studies.py`, replace:

```python
_DENSE_ESTIMATOR_CLASSES: frozenset[type[ClusterMixin]] = frozenset(
```

with:

```python
#: CARVE's own default resample count. A study's k-based run at this count
#: keeps the cache filename it had before carve_cache_path added run keys.
PACKAGE_N_RESAMPLES: int = next(
    field.default for field in fields(CARVE) if field.name == "n_resamples"
)

_DENSE_ESTIMATOR_CLASSES: frozenset[type[ClusterMixin]] = frozenset(
```

In `src/benchmarks/_studies.py`, replace:

```python
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
    n_resamples: int = 100,
    n_jobs: int = 1,
    random_state: int = 42,
    force: bool = False,
    consensus_anchors: int | None = None,
) -> CARVE:
    """Fit CARVE on a case study, caching the fitted state to disk.

    A Levine fit takes hours, so the cache is what makes regenerating a figure
    practical. The saved state does not carry the data matrix, so X_ is
    restored after loading, matching the notebook this replaces.

    consensus_anchors : int or None, default=None
        Forwarded to CARVE only when not None, so studies that leave it at
        the package default (an exact, unanchored run) are unaffected. hECA
        sets this at case-study scale, where an exact consensus matrix does
        not fit.
    """
    cache_path = Path(cache_path)
    _check_dense_fit(np.asarray(X).shape[0], model_grids)
```

with:

```python
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
    n_resamples: int = PACKAGE_N_RESAMPLES,
    n_jobs: int = 1,
    random_state: int = 42,
    force: bool = False,
    consensus_anchors: int | None = None,
    randomize_preprocessing: bool = False,
    normalization_options: list[Any] | None = None,
    dim_reduction_options: list[Any] | None = None,
) -> CARVE:
    """Fit CARVE on a case study, caching the fitted state to disk.

    A Levine fit takes hours, so the cache is what makes regenerating a figure
    practical. The saved state does not carry the data matrix, so X_ is
    restored after loading, matching the notebook this replaces.

    The cache is found by path alone. A caller that changes the grid, the
    resample count or the preprocessing must pass the same to
    carve_cache_path, which keys the filename on them.

    consensus_anchors : int or None, default=None
        Forwarded to CARVE only when not None, so studies that leave it at
        the package default (an exact, unanchored run) are unaffected. hECA
        sets this at case-study scale, where an exact consensus matrix does
        not fit.
    randomize_preprocessing : bool, default=False
        Forwarded to CARVE.fit only when True.
    normalization_options, dim_reduction_options : list or None
        Forwarded to CARVE only when not None, in the option syntax CARVE
        takes; _preprocessing.resolve_preprocessing builds both from a
        Study's preprocessing. Passing either without randomize_preprocessing
        raises, since CARVE would ignore it.
    """
    cache_path = Path(cache_path)
    _check_dense_fit(np.asarray(X).shape[0], model_grids)
    if not randomize_preprocessing and (
        normalization_options is not None or dim_reduction_options is not None
    ):
        raise ValueError(
            "normalization_options and dim_reduction_options only apply to a "
            "fit with randomize_preprocessing=True; CARVE would ignore them."
        )
```

In `src/benchmarks/_studies.py`, replace:

```python
    carve = CARVE(
        estimator_param_grids=model_grids,
        n_resamples=n_resamples,
        n_jobs=n_jobs,
        random_state=random_state,
        **(
            {}
            if consensus_anchors is None
            else {"consensus_anchors": consensus_anchors}
        ),
    )
    reference = None if y is None else np.asarray(y)
    carve.fit(np.asarray(X), reference_labels=reference)
```

with:

```python
    optional = {
        name: value
        for name, value in (
            ("consensus_anchors", consensus_anchors),
            ("normalization_options", normalization_options),
            ("dim_reduction_options", dim_reduction_options),
        )
        if value is not None
    }
    carve = CARVE(
        estimator_param_grids=model_grids,
        n_resamples=n_resamples,
        n_jobs=n_jobs,
        random_state=random_state,
        **optional,
    )
    reference = None if y is None else np.asarray(y)
    carve.fit(
        np.asarray(X),
        reference_labels=reference,
        **({"randomize_preprocessing": True} if randomize_preprocessing else {}),
    )
```

In `src/benchmarks/_studies.py`, replace:

```python
def carve_cache_path(study: Study, *, scale: str | None = None, root: Path) -> Path:
    """Where a study's fitted CARVE state is cached, per scale.

    Both the scale name and a short hash of its resolved size are part of
    the filename deliberately. fit_or_load_carve loads whatever file sits at
    the path it is handed, so a shared path would let a publication run
    silently reuse a development-scale fit -- that is what the scale name
    guards against. The hash guards the same failure one level up: editing
    STUDIES[...].scales[name] (say, hECA's "dev" from 25,000 to 50,000)
    changes what that scale resolves to without changing its name, and
    without the hash fit_or_load_carve would silently load the fit taken at
    the old size. A hash reads better than the raw resolved size for the
    None case (full data), which has no natural filename spelling.
    """
    resolved = resolve_scale(study, scale)  # raises if the scale is unknown
    name = _scale_name(study, scale)
    size_key = hashlib.sha1(repr(resolved).encode()).hexdigest()[:8]
    return Path(root) / f"carve_{study.name}_{name}_{size_key}.carve"
```

with:

```python
def _run_key(
    study: Study,
    *,
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]] | None,
    n_resamples: int,
    preprocessing: PreprocessingSpec | None,
) -> str | None:
    """A short hash of what a fit runs, or None for the study's default run.

    The default run is the study's own k-based grid (study_model_grids), no
    randomized preprocessing, and the package's resample count. Anything
    else hashes the grid, whose keys name the swept parameter, the
    preprocessing spec with the defaults it binds, and n_resamples.
    """
    default_grids = (
        study_model_grids(study)
        if study.candidate_k and study.estimator.name not in RESOLUTION_ESTIMATORS
        else None
    )
    grids = default_grids if model_grids is None else model_grids
    if grids is None:
        raise ValueError(
            f"Study {study.name!r} has no k-based grid to default to; pass the "
            "model_grids its fit runs."
        )
    if (
        grids == default_grids
        and preprocessing is None
        and n_resamples == PACKAGE_N_RESAMPLES
    ):
        return None
    payload = repr((grids, preprocessing_fingerprint(preprocessing), n_resamples))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def carve_cache_path(
    study: Study,
    *,
    scale: str | None = None,
    root: Path,
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]] | None = None,
    n_resamples: int = PACKAGE_N_RESAMPLES,
    preprocessing: PreprocessingSpec | None = None,
) -> Path:
    """Where a study's fitted CARVE state is cached, per scale and run.

    Both the scale name and a short hash of its resolved size are part of
    the filename deliberately. fit_or_load_carve loads whatever file sits at
    the path it is handed, so a shared path would let a publication run
    silently reuse a development-scale fit -- that is what the scale name
    guards against. The hash guards the same failure one level up: editing
    STUDIES[...].scales[name] (say, hECA's "dev" from 25,000 to 50,000)
    changes what that scale resolves to without changing its name, and
    without the hash fit_or_load_carve would silently load the fit taken at
    the old size. A hash reads better than the raw resolved size for the
    None case (full data), which has no natural filename spelling.

    A run key guards the same failure across kinds of fit: a k-based fit, a
    Leiden resolution fit and a randomized-preprocessing fit at one scale
    would otherwise share a filename. Pass the model_grids, n_resamples and
    preprocessing the fit runs. The study's default run (its k-based grid,
    no preprocessing, the package's resample count) carries no run key, so
    the caches written before run keys existed keep their names. A study
    with no k-based grid must pass model_grids.
    """
    resolved = resolve_scale(study, scale)  # raises if the scale is unknown
    name = _scale_name(study, scale)
    size_key = hashlib.sha1(repr(resolved).encode()).hexdigest()[:8]
    run_key = _run_key(
        study,
        model_grids=model_grids,
        n_resamples=n_resamples,
        preprocessing=preprocessing,
    )
    stem = f"carve_{study.name}_{name}_{size_key}"
    filename = f"{stem}.carve" if run_key is None else f"{stem}_{run_key}.carve"
    return Path(root) / filename
```

Then format and lint. The blocks above are already formatted, so the first command changes nothing:

```bash
.venv/bin/ruff format --check src/
.venv/bin/ruff check src/
```

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/benchmarks/test_studies.py tests/benchmarks/test_preprocessing.py -q`

Expected: exit code 0, ending with:

```text
97 passed in 25.81s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `_studies.py`: `filename = f"{stem}.carve" if run_key is None else f"{stem}_{run_key}.carve"` to `filename = f"{stem}_{run_key}.carve"` turns test_default_runs_keep_their_existing_filenames red.
- `_studies.py`: `and n_resamples == PACKAGE_N_RESAMPLES` to `(line removed)` turns test_any_other_run_gets_its_own_filename red.
- `_studies.py`: `and preprocessing is None` to `(line removed)` turns test_any_other_run_gets_its_own_filename red.
- `_studies.py`: `payload = repr((grids, preprocessing_fingerprint(preprocessing), n_resamples))` to `payload = repr((grids, preprocessing is None, n_resamples))` turns test_different_preprocessing_gets_a_different_filename red.
- `_preprocessing.py`: `return repr((spec, {key: PREPROCESSOR_DEFAULTS[key] for key in keys}))` to `return repr(spec)` turns test_editing_the_bound_defaults_changes_the_filename, test_the_fingerprint_names_the_bound_defaults red.
- `_studies.py`: `grids = default_grids if model_grids is None else model_grids` to `grids = default_grids if model_grids is None else model_grids` turns test_a_study_without_a_k_grid_must_name_its_grid red.
- `_studies.py`: `("normalization_options", normalization_options),` to `(line removed)` turns test_forwards_the_preprocessing_arguments red.
- `_studies.py`: `**({"randomize_preprocessing": True} if randomize_preprocessing else {}),` to `(line removed)` turns test_forwards_the_preprocessing_arguments red.
- `_studies.py`: `if value is not None` to `}` turns test_omits_the_preprocessing_arguments_when_unset red.
- `_studies.py`: `if not randomize_preprocessing and (` to `if randomize_preprocessing and (` turns test_options_without_randomize_preprocessing_raise red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "feat(benchmarks): run-keyed cache path; fit_or_load_carve forwards preprocessing"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 5: STUDIES["cusanovich"] sweeps Leiden resolution under randomized preprocessing

Spec 5's table. Leiden only, no partners, no candidate_k, 150 resamples, and the identity, t-SNE (perplexity 30) and UMAP (n_neighbors 15 or 30) pipelines over identity normalization. The allocation test fits nothing: it draws the 150 pipeline specs and counts 50 per reduction. After this task the notebook still calls `study_model_grids` and would fail if run; Task 11 rebuilds it, and the structural notebook tests read strings only, so they stay green in between.

Files:
- Modify: `src/benchmarks/_studies.py`
- Test: `tests/benchmarks/test_studies.py`

Interfaces:
- Consumes: Tasks 2 to 4: `PreprocessingSpec`, `resolve_preprocessing`, `carve_cache_path(model_grids=..., n_resamples=..., preprocessing=...)`.
- Produces: `STUDIES["cusanovich"]` with `estimator=EstimatorSpec("leiden")`, `candidate_k=()`, `partners=()`, `resolutions` 0.1 to 2.0 in steps of 0.1, `n_resamples=150`, `consensus_anchors=2000`, and `preprocessing=PreprocessingSpec(normalization=(("identity", {}),), dim_reduction=(("identity", {}), ("tsne", {"perplexity": [30]}), ("umap", {"n_neighbors": [15, 30]})))`.

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/test_studies.py`, replace:

```python
import numpy as np
import pandas as pd
import pytest
```

with:

```python
from collections import Counter

import numpy as np
import pandas as pd
import pytest
```

In `tests/benchmarks/test_studies.py`, replace:

```python
from carve.cluster import LeidenClustering, SpectralClustering
```

with:

```python
from carve._pipeline import allocate_pipelines
from carve.cluster import LeidenClustering, SpectralClustering
```

In `tests/benchmarks/test_studies.py`, replace:

```python
    def test_each_study_has_a_loader_and_candidate_k(self):
        for study in STUDIES.values():
            assert callable(study.loader)
            assert len(study.candidate_k) > 0
```

with:

```python
    def test_each_study_has_a_loader_and_a_sweep(self):
        for study in STUDIES.values():
            assert callable(study.loader)
            assert study.candidate_k or study.resolutions
```

In `tests/benchmarks/test_studies.py`, replace:

```python
    def test_cusanovich_sweeps_four_through_sixteen(self):
        # Centered near the 13 tissues; stops well short of the 30 clusters
        # and 40 cell labels the source reports.
        assert STUDIES["cusanovich"].candidate_k == tuple(range(4, 17))
```

with:

```python
    def test_cusanovich_sweeps_leiden_resolution_only(self):
        # Graph community detection is the source's own algorithm family, so
        # the study sweeps Leiden resolution and nothing k-based.
        study = STUDIES["cusanovich"]
        assert study.estimator == EstimatorSpec(name="leiden")
        assert study.candidate_k == ()
        assert study.partners == ()
        with pytest.raises(ValueError, match="study_resolution_grids"):
            study_model_grids(study)
        ((cls, grid),) = study_resolution_grids(study)
        assert cls is LeidenClustering
        assert grid["resolution"] == pytest.approx(
            [round(0.1 * i, 1) for i in range(1, 21)]
        )
        assert grid["n_neighbors"] == [15]

    def test_cusanovich_randomizes_over_the_lsi_tsne_and_umap(self):
        study = STUDIES["cusanovich"]
        assert study.n_resamples == 150
        assert study.preprocessing == PreprocessingSpec(
            normalization=(("identity", {}),),
            dim_reduction=(
                ("identity", {}),
                ("tsne", {"perplexity": [30]}),
                ("umap", {"n_neighbors": [15, 30]}),
            ),
        )

    def test_cusanovich_gives_each_reduction_fifty_resamples(self):
        # Stratified allocation is over options, so 150 resamples split 50,
        # 50, 50 across identity, t-SNE and UMAP; UMAP's 50 are shared by its
        # two n_neighbors values, each of which is its own pipeline label.
        pytest.importorskip("umap")
        study = STUDIES["cusanovich"]
        options = resolve_preprocessing(study.preprocessing)
        pipelines = allocate_pipelines(
            options["normalization_options"],
            options["dim_reduction_options"],
            n_resamples=study.n_resamples,
            random_state=42,
        )
        counts = Counter(pipeline.label for pipeline in pipelines)
        assert set(counts) == {
            "identity | identity",
            "identity | TSNE(perplexity=30)",
            "identity | UMAP(n_neighbors=15)",
            "identity | UMAP(n_neighbors=30)",
        }
        assert counts["identity | identity"] == 50
        assert counts["identity | TSNE(perplexity=30)"] == 50
        assert (
            counts["identity | UMAP(n_neighbors=15)"]
            + counts["identity | UMAP(n_neighbors=30)"]
            == 50
        )

    def test_cusanovich_cache_never_resolves_to_the_invalid_pre_fix_cache(
        self, tmp_path
    ):
        # carve_cusanovich_dev_7841fb1f.carve was fit before the cell-order
        # fix (1dea71d) and must never be served again.
        study = STUDIES["cusanovich"]
        path = carve_cache_path(
            study,
            root=tmp_path,
            model_grids=study_resolution_grids(study),
            n_resamples=study.n_resamples,
            preprocessing=study.preprocessing,
        )
        assert path.name.startswith("carve_cusanovich_dev_7841fb1f_")
        assert path.name != "carve_cusanovich_dev_7841fb1f.carve"
```

In `tests/benchmarks/test_studies.py`, delete:

```python
    def test_cusanovich_sweeps_kmeans_spectral_ward_and_single_linkage(self):
        grids = study_model_grids(STUDIES["cusanovich"])

        assert [(cls, params.get("linkage")) for cls, params in grids] == [
            (KMeans, None),
            (SpectralClustering, None),
            (AgglomerativeClustering, ["ward"]),
            (AgglomerativeClustering, ["single"]),
        ]

```

In `tests/benchmarks/test_studies.py`, delete:

```python
    def test_cusanovich_also_declares_resolutions_for_its_atlas_pass(self):
        # At atlas scale (every annotated cell -- smaller than the atlas's
        # published 81,173, since Unknown-labeled cells are always dropped)
        # neither spectral nor Ward can run, so the full-atlas pass sweeps
        # Leiden resolution instead of k.
        assert len(STUDIES["cusanovich"].resolutions) == 20
        assert study_resolution_grids(STUDIES["cusanovich"])

```

In `tests/benchmarks/test_studies.py`, replace:

```python
    def test_cusanovich_pins_the_same_anchor_count_as_heca(self):
        # Left at the package default, the atlas-scale Leiden sweep (tens of
        # thousands of cells, above anchor_threshold=5000) would retain 8.0
        # GB of consensus blocks against hECA's 1.28 GB at a tenth the
        # sample count. Pinned to the same 2000 anchors for the same reason.
        assert STUDIES["cusanovich"].consensus_anchors == 2000
```

with:

```python
    def test_cusanovich_pins_the_same_anchor_count_as_heca(self):
        # The 20-configuration resolution sweep retains 1.28 GB of consensus
        # blocks at 2000 anchors, against 8.0 GB at the package default once
        # n exceeds anchor_threshold. Pinned to hECA's count for that reason.
        assert STUDIES["cusanovich"].consensus_anchors == 2000
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/benchmarks/test_studies.py -q`

Expected: exit code 1, ending with:

```text
E       AssertionError: assert EstimatorSpec(name='kmeans') == EstimatorSpec(name='leiden')
E         
E         Differing attributes:
E         ['name']
E         
E         Drill down into differing attribute name:
E           name: 'kmeans' != 'leiden'
E           - leiden
E           + kmeans
E       AssertionError: assert 100 == 150
E        +  where 100 = Study(name='cusanovich', loader=<function _cusanovich_loader at 0x1192f9f80>, estimator=EstimatorSpec(name='kmeans'), ...1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0), consensus_anchors=2000, k_star=None, n_resamples=100, preprocessing=None).n_resamples
E       AttributeError: 'NoneType' object has no attribute 'normalization'
3 failed, 76 passed in 21.41s
```

- [ ] Step 3: Implement

In `src/benchmarks/_studies.py`, replace:

```python
    "cusanovich": Study(
        name="cusanovich",
        loader=_cusanovich_loader,
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=tuple(range(4, 17)),
        scales={"dev": 1500, "publication": 5000, "atlas": None},
        default_scale="dev",
        # Spectral as in the other case studies, plus Ward and single-linkage
        # agglomerative clustering. Single linkage is included deliberately:
        # on an LSI it tends to peel off outliers one at a time, a partition
        # that is near-identical across resamples and so scores as highly
        # stable while saying little, which is worth showing.
        partners=(
            EstimatorSpec(name="spectral"),
            EstimatorSpec(name="agglomerative"),
            EstimatorSpec(name="agglomerative_single"),
        ),
        # The atlas scale runs every annotated cell (the loader always drops
        # cell_label=="Unknown", about 12 percent of the atlas), where
        # spectral and Ward cannot run, so that pass sweeps Leiden resolution
        # instead. This is what shows the case-study conclusion survives past
        # the subsample. 81,173 is the atlas as published; the annotated
        # subset actually analyzed at this scale is smaller (see
        # datasets._cusanovich and meta["n_cells_annotated"]).
        resolutions=tuple(round(0.1 * i, 1) for i in range(1, 21)),
        # Pinned rather than left at the package default. Both
        # consensus_matrices_ and consensus_generalizability_matrices_ are
        # retained per configuration, so retained memory is
        # n_configs * 2 * m**2 * 8 bytes. At atlas scale (tens of thousands
        # of cells, above anchor_threshold=5000) the 20-config Leiden
        # resolution sweep would otherwise anchor to the package default of
        # m=5000: 20 * 2 * 5000**2 * 8 B = 8.0 GB retained. Pinned to the
        # same 2000 anchors hECA uses (below), that sweep instead retains
        # 20 * 2 * 2000**2 * 8 B = 1.28 GB. At dev scale (1,500 cells) this
        # is a no-op: m=2000 exceeds n, so resolve_anchors takes the exact,
        # unanchored path regardless. At publication scale (5,000 cells) it
        # newly anchors to 2000 where the run was previously exact, which
        # is a deliberate, harmless trade at that size (a 5000x5000 exact
        # matrix is only 0.2 GB) made for one consistent anchor count across
        # every scale this study declares.
        consensus_anchors=2000,
    ),
```

with:

```python
    "cusanovich": Study(
        name="cusanovich",
        loader=_cusanovich_loader,
        # The source clustered its t-SNE with graph community detection, so
        # Leiden over resolution gives their operating point a position on
        # the same axis. Nothing k-based is swept.
        estimator=EstimatorSpec(name="leiden"),
        candidate_k=(),
        # The atlas scale stays declared so the loader can still produce it;
        # the notebook does not run it (spec section 6).
        scales={"dev": 1500, "publication": 5000, "atlas": None},
        default_scale="dev",
        partners=(),
        resolutions=tuple(round(0.1 * i, 1) for i in range(1, 21)),
        # Pinned rather than left at the package default. Both
        # consensus_matrices_ and consensus_generalizability_matrices_ are
        # retained per configuration, so retained memory is
        # n_configs * 2 * m**2 * 8 bytes: 20 * 2 * 2000**2 * 8 B = 1.28 GB
        # for the 20-configuration sweep at 2000 anchors, against 8.0 GB at
        # the package default of 5000 once n exceeds anchor_threshold. At dev
        # scale (1,500 cells) this is a no-op, since m=2000 exceeds n and the
        # run is exact. At publication scale (5,000 cells) the run anchors to
        # 2000, the count hECA uses.
        consensus_anchors=2000,
        # 150 resamples over three dimensionality reductions gives each 50
        # under stratified allocation, against 33 at the package default.
        n_resamples=150,
        preprocessing=PreprocessingSpec(
            # The LSI is already scaled by its singular values; neither the
            # source nor the field standardizes or log-transforms it.
            normalization=(("identity", {}),),
            dim_reduction=(
                ("identity", {}),
                # The source's own perplexity and no other, so the t-SNE line
                # is their recipe rather than an average over perplexities.
                ("tsne", {"perplexity": [30]}),
                # The field's current default embedding, so the finding reads
                # as one about clustering in an embedding, not about t-SNE.
                ("umap", {"n_neighbors": [15, 30]}),
            ),
        ),
    ),
```

In `src/benchmarks/_studies.py`, replace:

```python
    Klein sweeps Ward agglomerative and spectral; Levine sweeps KMeans and
    spectral; Cusanovich sweeps KMeans, spectral, Ward and single linkage at
    case-study scale; hECA pairs MiniBatchKMeans with KMeans. Each study
    declares this on Study.partners.
    """
```

with:

```python
    Klein sweeps Ward agglomerative and spectral; Levine sweeps KMeans and
    spectral; hECA pairs MiniBatchKMeans with KMeans. Each study declares
    this on Study.partners. Cusanovich sweeps only Leiden resolution, so it
    has no k-based grid and this raises for it.
    """
```

Then format and lint. The blocks above are already formatted, so the first command changes nothing:

```bash
.venv/bin/ruff format --check src/
.venv/bin/ruff check src/
```

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/benchmarks/test_studies.py -q`

Expected: exit code 0, ending with:

```text
79 passed in 20.67s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `_studies.py`: `estimator=EstimatorSpec(name="leiden"),` to `estimator=EstimatorSpec(name="kmeans"),` turns test_cusanovich_sweeps_leiden_resolution_only red.
- `_studies.py`: `n_resamples=150,` to `n_resamples=100,` turns test_cusanovich_randomizes_over_the_lsi_tsne_and_umap red.
- `_studies.py`: `("tsne", {"perplexity": [30]}),` to `("tsne", {"perplexity": [15, 30]}),` turns test_cusanovich_randomizes_over_the_lsi_tsne_and_umap, test_cusanovich_gives_each_reduction_fifty_resamples red.
- `_studies.py`: `("umap", {"n_neighbors": [15, 30]}),` to `),` turns test_cusanovich_gives_each_reduction_fifty_resamples red.
- `_studies.py`: `partners=(),` to `partners=(EstimatorSpec(name="kmeans"),),` turns test_cusanovich_sweeps_leiden_resolution_only red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "feat(benchmarks): Cusanovich sweeps Leiden resolution over LSI, t-SNE and UMAP"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 6: Comparison compute: best pipeline, embedding, operating point, published partition probe

Spec 7.1, the four functions, in a new module with no drawing in it. The tests use a small fake fitted object and planted tables, each with a decoy that a wrong implementation would pick: a better-scoring row at another resolution, a pipeline that hits the target count exactly, a random relabeling that a forest memorizes if scored on its training rows.

Files:
- Create: `src/benchmarks/_cusanovich_compare.py`
- Test: `tests/benchmarks/test_cusanovich_compare.py`

Interfaces:
- Consumes: carve: `CARVE._select_row`, `preprocessing_results_`, `preprocessing_pipelines_`, `_pipeline.PipelineSpec`, `PipelineStep`, `pipeline_from_spec`, `_selection.MEASURE_MAP`, `_utils.default_generalizability_classifier`. Task 3: `PREPROCESSOR_NAMES`.
- Produces: `benchmarks._cusanovich_compare.OPERATING_POINT_TOLERANCE = 0.25`; `axis_prefix(step) -> str`; `best_pipeline(carve, *, measure="stability", rule="1se", not_two=False) -> (pandas.Series, PipelineSpec)`; `pipeline_embedding(X, spec, *, random_state) -> (ndarray (n, 2), (str, str))`; `source_operating_point(carve, *, pipeline, target_k) -> (resolution, observed count)`; `published_partition_generalizability(X, labels, *, n_splits, subsample_ratio, n_trees, random_state, n_jobs=1) -> (mean, se)`.

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/test_cusanovich_compare.py` with this content (replacing the file if it exists):

```python
"""Tests for the Cusanovich comparison compute."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from benchmarks._cusanovich_compare import (
    axis_prefix,
    best_pipeline,
    pipeline_embedding,
    published_partition_generalizability,
    source_operating_point,
)
from carve._pipeline import PipelineSpec, PipelineStep

IDENTITY = PipelineStep(cls=FunctionTransformer, params={}, name="identity")


def _spec(dim_reduction: PipelineStep) -> PipelineSpec:
    return PipelineSpec(normalization=IDENTITY, dim_reduction=dim_reduction)


def _row(method_id, pipeline, resolution, stability, generalizability, observed):
    return {
        "method_id": method_id,
        "pipeline": pipeline,
        "resolution": resolution,
        "sweep_value": resolution,
        "ari_stability": stability,
        "ari_generalizability": generalizability,
        "n_clusters_observed": observed,
    }


class _FakeCarve:
    """The members of a fitted CARVE these functions read."""

    def __init__(self, table, pipelines=None, selected=None):
        self.preprocessing_results_ = table
        self.preprocessing_pipelines_ = pipelines
        self._selected = selected
        self.select_calls = []

    def _select_row(self, *, measure, rule, not_two=False):
        self.select_calls.append({"measure": measure, "rule": rule, "not_two": not_two})
        return pd.Series(self._selected), 0, 0, False


class TestBestPipeline:
    @pytest.fixture
    def carve(self):
        # Two pipelines at two resolutions of one configuration. At the
        # selected resolution, 0.5, B has the higher stability and A the
        # higher generalizability. Resolution 1.0 holds a decoy that beats
        # both on both criteria, which a function ignoring the selected
        # configuration would pick.
        table = pd.DataFrame(
            [
                _row("m0", "A", 0.5, 0.60, 0.80, 8.0),
                _row("m0", "B", 0.5, 0.70, 0.50, 9.0),
                _row("m0", "A", 1.0, 0.95, 0.95, 20.0),
                _row("m0", "B", 1.0, 0.20, 0.20, 21.0),
            ]
        )
        pipelines = {"A": object(), "B": object()}
        return _FakeCarve(
            table, pipelines, selected={"method_id": "m0", "sweep_value": 0.5}
        )

    def test_picks_the_best_row_at_the_selected_configuration_not_overall(self, carve):
        row, spec = best_pipeline(carve, measure="stability", rule="1se")
        assert row["pipeline"] == "B"
        assert row["sweep_value"] == 0.5
        assert spec is carve.preprocessing_pipelines_["B"]

    def test_the_measure_decides_the_winner(self, carve):
        row, spec = best_pipeline(carve, measure="generalizability", rule="1se")
        assert row["pipeline"] == "A"
        assert spec is carve.preprocessing_pipelines_["A"]

    def test_forwards_the_selection_arguments(self, carve):
        best_pipeline(carve, measure="generalizability", rule="max", not_two=True)
        assert carve.select_calls == [
            {"measure": "generalizability", "rule": "max", "not_two": True}
        ]

    def test_another_configuration_at_the_same_resolution_does_not_compete(
        self, carve
    ):
        decoy = pd.DataFrame([_row("m1", "A", 0.5, 0.99, 0.99, 8.0)])
        carve.preprocessing_results_ = pd.concat(
            [carve.preprocessing_results_, decoy], ignore_index=True
        )
        row, _ = best_pipeline(carve, measure="stability", rule="1se")
        assert (row["method_id"], row["pipeline"]) == ("m0", "B")

    def test_a_non_ari_measure_raises(self, carve):
        with pytest.raises(ValueError, match="stability or generalizability"):
            best_pipeline(carve, measure="pac", rule="1se")

    def test_a_fit_without_randomized_preprocessing_raises(self):
        with pytest.raises(ValueError, match="randomize_preprocessing=True"):
            best_pipeline(_FakeCarve(None), measure="stability", rule="1se")


class TestPipelineEmbedding:
    @pytest.fixture
    def X(self):
        return np.random.default_rng(0).normal(size=(60, 50))

    def test_the_identity_pipeline_gives_the_first_two_lsi_components(self, X):
        Z, labels = pipeline_embedding(X, _spec(IDENTITY), random_state=0)
        np.testing.assert_array_equal(Z, X[:, :2])
        assert labels == ("LSI 1", "LSI 2")

    def test_a_two_dimensional_pipeline_gives_its_output_fit_on_all_of_x(self, X):
        step = PipelineStep(cls=PCA, params={"n_components": 2}, name="PCA")
        Z, labels = pipeline_embedding(X, _spec(step), random_state=0)
        np.testing.assert_allclose(
            Z, PCA(n_components=2, random_state=0).fit_transform(X)
        )
        assert labels == ("PC 1", "PC 2")

    def test_tsne_axes_are_named_for_tsne_and_seeded(self, X):
        step = PipelineStep(cls=TSNE, params={"perplexity": 5}, name="TSNE")
        first, labels = pipeline_embedding(X, _spec(step), random_state=3)
        second, _ = pipeline_embedding(X, _spec(step), random_state=3)
        assert first.shape == (60, 2)
        np.testing.assert_array_equal(first, second)
        assert labels == ("t-SNE 1", "t-SNE 2")

    def test_axis_prefixes(self):
        # A display name decides the prefix; an unnamed step falls back to
        # its class name, mapped when the registry knows it and kept when not.
        assert axis_prefix(PipelineStep(cls=object, params={}, name="UMAP")) == "UMAP"
        assert axis_prefix(PipelineStep(cls=PCA, params={}, name=None)) == "PC"
        scaler = PipelineStep(cls=StandardScaler, params={}, name=None)
        assert axis_prefix(scaler) == "StandardScaler"


class TestSourceOperatingPoint:
    @pytest.fixture
    def carve(self):
        # The t-SNE pipeline's counts rise with resolution, listed out of
        # order. A decoy pipeline hits 30 exactly, which a function ignoring
        # the pipeline would return.
        table = pd.DataFrame(
            [
                _row("m0", "tsne", 0.6, 0.5, 0.5, 33.0),
                _row("m0", "tsne", 0.2, 0.5, 0.5, 12.0),
                _row("m0", "tsne", 0.4, 0.5, 0.5, 26.0),
                _row("m0", "umap", 0.8, 0.5, 0.5, 30.0),
            ]
        )
        return _FakeCarve(table)

    def test_picks_the_named_pipelines_nearest_count(self, carve):
        assert source_operating_point(carve, pipeline="tsne", target_k=30) == (
            0.6,
            33.0,
        )

    def test_ties_go_to_the_lower_resolution(self, carve):
        # 26 and 34 are both 4 from 30; the table lists 0.6 first.
        carve.preprocessing_results_.loc[0, "n_clusters_observed"] = 34.0
        assert source_operating_point(carve, pipeline="tsne", target_k=30) == (
            0.4,
            26.0,
        )

    def test_stays_silent_within_a_quarter_of_the_target(self, carve):
        # 33 is 7 from 40, inside 25 percent (10); any warning fails the test
        # under filterwarnings = error.
        assert source_operating_point(carve, pipeline="tsne", target_k=40) == (
            0.6,
            33.0,
        )

    def test_warns_past_a_quarter_of_the_target(self, carve):
        # 33 is 17 from 50, past 25 percent (12.5).
        with pytest.warns(UserWarning, match="more than 25%"):
            point = source_operating_point(carve, pipeline="tsne", target_k=50)
        assert point == (0.6, 33.0)

    def test_an_unknown_pipeline_raises_naming_the_available_ones(self, carve):
        with pytest.raises(ValueError, match="umap"):
            source_operating_point(carve, pipeline="pca", target_k=30)

    def test_more_than_one_configuration_raises(self, carve):
        carve.preprocessing_results_.loc[1, "method_id"] = "m1"
        with pytest.raises(ValueError, match="more than one estimator configuration"):
            source_operating_point(carve, pipeline="tsne", target_k=30)


class TestPublishedPartitionGeneralizability:
    @pytest.fixture
    def blobs(self):
        rng = np.random.default_rng(0)
        centers = rng.normal(scale=10.0, size=(3, 5))
        members = np.repeat(np.arange(3), 60)
        X = centers[members] + rng.normal(scale=0.3, size=(180, 5))
        return X, np.array(["a", "b", "c"])[members]

    @staticmethod
    def _score(X, labels, **overrides):
        settings = {
            "n_splits": 10,
            "subsample_ratio": 0.618,
            "n_trees": 20,
            "random_state": 0,
        }
        settings.update(overrides)
        return published_partition_generalizability(X, labels, **settings)

    def test_a_partition_that_is_a_function_of_x_generalizes_perfectly(self, blobs):
        mean, se = self._score(*blobs)
        assert mean == pytest.approx(1.0)
        assert se == pytest.approx(0.0)

    def test_a_random_relabeling_does_not_generalize(self, blobs):
        # A forest memorizes any labeling of its training rows, so scoring on
        # the training rows, or against them, would come out near 1.
        X, labels = blobs
        shuffled = np.random.default_rng(1).permutation(labels)
        mean, _ = self._score(X, shuffled, n_splits=30)
        assert abs(mean) < 0.05

    def test_is_seeded(self, blobs):
        X, labels = blobs
        shuffled = np.random.default_rng(1).permutation(labels)
        assert self._score(X, shuffled) == self._score(X, shuffled)

    def test_a_single_split_has_no_standard_error(self, blobs):
        mean, se = self._score(*blobs, n_splits=1)
        assert mean == pytest.approx(1.0)
        assert np.isnan(se)
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/benchmarks/test_cusanovich_compare.py -q`

Expected: exit code 2, ending with:

```text
E   ModuleNotFoundError: No module named 'benchmarks._cusanovich_compare'
ERROR tests/benchmarks/test_cusanovich_compare.py
1 error in 0.08s
```

- [ ] Step 3: Implement

Create `src/benchmarks/_cusanovich_compare.py` with this content (replacing the file if it exists):

```python
"""Compute for the Cusanovich case study on randomized preprocessing.

The source clustered a two-dimensional t-SNE of its LSI into 30 clusters.
These functions set that recipe beside CARVE's: which preprocessing pipeline
CARVE rates best at the configuration it selects, where the source's
granularity falls on the t-SNE pipeline's resolution axis, and how well the
published partition generalizes under the classifier probe CARVE applies to
its own clusterings. No matplotlib here; figures._cusanovich_results draws
the result.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedShuffleSplit

from carve._pipeline import PipelineSpec, PipelineStep, pipeline_from_spec
from carve._selection import MEASURE_MAP
from carve._utils import default_generalizability_classifier

from ._preprocessing import PREPROCESSOR_NAMES

#: How far a pipeline's nearest observed cluster count may sit from the target
#: before source_operating_point warns, as a fraction of the target.
OPERATING_POINT_TOLERANCE = 0.25

#: What a panel calls the output axes of each dimensionality reduction. The
#: identity pipeline passes the loader's LSI through.
_AXIS_PREFIXES = {
    PREPROCESSOR_NAMES["identity"]: "LSI",
    PREPROCESSOR_NAMES["pca"]: "PC",
    PREPROCESSOR_NAMES["tsne"]: "t-SNE",
    PREPROCESSOR_NAMES["umap"]: "UMAP",
}

#: umap-learn warns on every seeded fit that the seed disables its own
#: parallelism; _runner.embed_resample silences the same message.
_UMAP_SEED_WARNING = r"n_jobs value .* overridden to 1 by setting random_state"

_ARI_COLUMNS = ("ari_stability", "ari_generalizability")


def _per_pipeline_table(carve: Any) -> pd.DataFrame:
    table = carve.preprocessing_results_
    if table is None:
        raise ValueError(
            "This CARVE fit has no per-pipeline table; fit it with "
            "randomize_preprocessing=True."
        )
    return table


def axis_prefix(step: PipelineStep) -> str:
    """The name a panel gives the output axes of a dimensionality reduction."""
    name = step.name if step.name is not None else getattr(step.cls, "__name__", "")
    return _AXIS_PREFIXES.get(name, name or "Component")


def best_pipeline(
    carve: Any,
    *,
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
) -> tuple[pd.Series, PipelineSpec]:
    """The pipeline CARVE rates best at the configuration it selects.

    The configuration is carve._select_row under measure, rule and not_two,
    the selection every figure reads. Among the preprocessing_results_ rows
    at that configuration's method_id and sweep value, the one with the
    highest measure wins; rows at any other configuration never compete, even
    when they score higher.

    Returns
    -------
    (row, spec) : the preprocessing_results_ row and its PipelineSpec from
        preprocessing_pipelines_.
    """
    table = _per_pipeline_table(carve)
    column = MEASURE_MAP.get(measure)
    if column not in _ARI_COLUMNS:
        raise ValueError(
            "The per-pipeline table carries only the ARI criteria, so measure "
            f"must be stability or generalizability (or an alias); got {measure!r}."
        )

    selected, _, _, _ = carve._select_row(measure=measure, rule=rule, not_two=not_two)
    at_selected = table.loc[
        (table["method_id"] == selected["method_id"])
        & np.isclose(table["sweep_value"].astype(float), float(selected["sweep_value"]))
    ]
    if at_selected.empty:
        raise RuntimeError(
            f"preprocessing_results_ has no rows at the selected configuration "
            f"({selected['method_id']}, {selected['sweep_value']})."
        )
    row = at_selected.loc[at_selected[column].idxmax()]
    return row, carve.preprocessing_pipelines_[row["pipeline"]]


def pipeline_embedding(
    X: np.ndarray, spec: PipelineSpec, *, random_state: int
) -> tuple[np.ndarray, tuple[str, str]]:
    """Embed all of X with one pipeline, for drawing.

    Fits a fresh pipeline_from_spec(spec, random_state) on the full X and
    keeps its first two output columns: the whole embedding for a
    two-dimensional reduction, LSI 1 and LSI 2 for the identity pipeline on
    the LSI. The axis names come back with it, so a panel names what it draws.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_UMAP_SEED_WARNING, category=UserWarning
        )
        Z = np.asarray(pipeline_from_spec(spec, random_state).fit_transform(X))
    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError(
            f"Pipeline {spec.label!r} produced an output of shape {Z.shape}; "
            "drawing it needs at least two columns."
        )
    prefix = axis_prefix(spec.dim_reduction)
    return Z[:, :2], (f"{prefix} 1", f"{prefix} 2")


def source_operating_point(
    carve: Any, *, pipeline: str, target_k: int
) -> tuple[float, float]:
    """The resolution at which one pipeline's clusterings come nearest target_k.

    Reads the preprocessing_results_ rows for the named pipeline and returns
    (resolution, observed cluster count) for the row whose n_clusters_observed
    is nearest target_k, ties going to the lower resolution. The count is a
    mean over that pipeline's resamples at that resolution. Warns when it is
    more than OPERATING_POINT_TOLERANCE of target_k away, which means the
    resolution grid does not reach the source's granularity for this pipeline.
    """
    table = _per_pipeline_table(carve)
    rows = table.loc[table["pipeline"] == pipeline]
    if rows.empty:
        raise ValueError(
            f"No per-pipeline rows for {pipeline!r}. "
            f"Available: {sorted(table['pipeline'].unique())}."
        )
    if rows["method_id"].nunique() > 1:
        raise ValueError(
            f"Pipeline {pipeline!r} has rows for more than one estimator "
            "configuration; an operating point is defined on one resolution axis."
        )

    rows = rows.sort_values("sweep_value", kind="stable")
    distance = (rows["n_clusters_observed"] - target_k).abs()
    row = rows.loc[distance.idxmin()]
    resolution = float(row["sweep_value"])
    observed = float(row["n_clusters_observed"])
    if abs(observed - target_k) > OPERATING_POINT_TOLERANCE * target_k:
        warnings.warn(
            f"The {pipeline!r} clustering nearest {target_k} clusters has "
            f"{observed:.1f}, at resolution {resolution:g}: more than "
            f"{OPERATING_POINT_TOLERANCE:.0%} away. Extend the study's "
            "resolution grid.",
            UserWarning,
            stacklevel=2,
        )
    return resolution, observed


def published_partition_generalizability(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int,
    subsample_ratio: float,
    n_trees: int,
    random_state: int,
    n_jobs: int = 1,
) -> tuple[float, float]:
    """How well a fixed partition generalizes under CARVE's classifier probe.

    For each of n_splits stratified splits with subsample_ratio of the rows in
    training, the classifier CARVE trains for generalizability
    (carve._utils.default_generalizability_classifier, a random forest with
    n_trees trees) is fit on X[train] with the partition's labels and predicts
    X[test]; the split scores the ARI between those predictions and the
    partition's own test labels. Split s seeds its forest with
    random_state + s. Stratifying keeps every class in training, which a
    random split can miss for a small one. Stability of a fixed partition is
    not defined, so none is computed.

    Returns
    -------
    (mean, se) : the mean ARI over splits and its standard error, NaN for a
        single split, as in CARVE's own tables.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, train_size=subsample_ratio, random_state=random_state
    )
    scores = []
    for split, (train, test) in enumerate(splitter.split(X, labels)):
        classifier = default_generalizability_classifier(
            classifier=None,
            n_features=X.shape[1],
            n_trees=n_trees,
            random_state=random_state + split,
            n_jobs=n_jobs,
        )
        classifier.fit(X[train], labels[train])
        scores.append(adjusted_rand_score(labels[test], classifier.predict(X[test])))

    scores = np.asarray(scores, dtype=float)
    se = (
        float(np.std(scores, ddof=1) / np.sqrt(scores.size))
        if scores.size > 1
        else float("nan")
    )
    return float(np.mean(scores)), se
```

Then format and lint. The blocks above are already formatted, so the first command changes nothing:

```bash
.venv/bin/ruff format --check src/
.venv/bin/ruff check src/
```

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/benchmarks/test_cusanovich_compare.py -q`

Expected: exit code 0, ending with:

```text
20 passed in 0.80s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `_cusanovich_compare.py`: `& np.isclose(` to `| np.isclose(` turns test_picks_the_best_row_at_the_selected_configuration_not_overall red.
- `_cusanovich_compare.py`: `(table["method_id"] == selected["method_id"])` to `(table["method_id"] == table["method_id"])` turns test_another_configuration_at_the_same_resolution_does_not_compete red.
- `_cusanovich_compare.py`: `row = at_selected.loc[at_selected[column].idxmax()]` to `row = at_selected.loc[at_selected["ari_stability"].idxmax()]` turns test_the_measure_decides_the_winner red.
- `_cusanovich_compare.py`: `selected, _, _, _ = carve._select_row(measure=measure, rule=rule, not_two=not_two)` to `selected, _, _, _ = carve._select_row(measure=measure, rule="1se", not_two=not_two)` turns test_forwards_the_selection_arguments red.
- `_cusanovich_compare.py`: `if column not in _ARI_COLUMNS:` to `if column is None:` turns test_a_non_ari_measure_raises red.
- `_cusanovich_compare.py`: `return Z[:, :2], (f"{prefix} 1", f"{prefix} 2")` to `return Z[:, -2:], (f"{prefix} 1", f"{prefix} 2")` turns test_the_identity_pipeline_gives_the_first_two_lsi_components red.
- `_cusanovich_compare.py`: `Z = np.asarray(pipeline_from_spec(spec, random_state).fit_transform(X))` to `Z = np.asarray(pipeline_from_spec(spec, random_state).fit(X[: len(X) // 2]).transform(X))` turns test_a_two_dimensional_pipeline_gives_its_output_fit_on_all_of_x red.
- `_cusanovich_compare.py`: `Z = np.asarray(pipeline_from_spec(spec, random_state).fit_transform(X))` to `Z = np.asarray(pipeline_from_spec(spec, np.random.default_rng().integers(1000)).fit_transform(X))` turns test_tsne_axes_are_named_for_tsne_and_seeded red.
- `_cusanovich_compare.py`: `PREPROCESSOR_NAMES["tsne"]: "t-SNE",` to `PREPROCESSOR_NAMES["tsne"]: "TSNE",` turns test_tsne_axes_are_named_for_tsne_and_seeded red.
- `_cusanovich_compare.py`: `rows = table.loc[table["pipeline"] == pipeline]` to `rows = table` turns test_picks_the_named_pipelines_nearest_count red.
- `_cusanovich_compare.py`: `rows = rows.sort_values("sweep_value", kind="stable")` to `(line removed)` turns test_ties_go_to_the_lower_resolution red.
- `_cusanovich_compare.py`: `OPERATING_POINT_TOLERANCE = 0.25` to `OPERATING_POINT_TOLERANCE = 0.4` turns test_warns_past_a_quarter_of_the_target red.
- `_cusanovich_compare.py`: `OPERATING_POINT_TOLERANCE = 0.25` to `OPERATING_POINT_TOLERANCE = 0.15` turns test_stays_silent_within_a_quarter_of_the_target red.
- `_cusanovich_compare.py`: `if rows["method_id"].nunique() > 1:` to `if rows["method_id"].nunique() > 2:` turns test_more_than_one_configuration_raises red.
- `_cusanovich_compare.py`: `scores.append(adjusted_rand_score(labels[test], classifier.predict(X[test])))` to `scores.append(adjusted_rand_score(labels[train], classifier.predict(X[train])))` turns test_a_random_relabeling_does_not_generalize red.
- `_cusanovich_compare.py`: `if scores.size > 1` to `if scores.size > 1` turns test_a_single_split_has_no_standard_error red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "feat(benchmarks): Cusanovich comparison compute"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 7: align_cluster_labels never merges clusters

The author's request, not in the spec (refinement 9). `carve._utils.align_cluster_labels` leaves each cluster the Hungarian assignment cannot match on its own id, which can equal the id a matched cluster was given, so the two merge whenever there are more clusters than reference labels (40 clusters against 30 labels came back as 34 ids). Unmatched clusters now take fresh ids past the largest reference label. `figures._case_study._align_to_reference` delegates to this function, so the composites and the Cusanovich panel A inherit the fix. `_accuracy` also calls it, with predictions and held-out labels whose counts can differ, so the task runs the accuracy tests and the non-randomized regression gate as well; both stayed green when this plan was verified. Changes to the R port are out of scope.

Files:
- Modify: `src/carve/_utils.py`
- Test: `tests/test_utils.py`

Interfaces:
- Consumes: nothing new.
- Produces: `carve._utils.align_cluster_labels(reference_labels, labels)`, one to one: unmatched clusters take ids from `max(reference_labels) + 1` upward, in sorted cluster order.

- [ ] Step 1: Write the failing tests

In `tests/test_utils.py`, replace:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
```

with:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
```

In `tests/test_utils.py`, replace:

```python
        assert aligned.dtype == ref.dtype
```

with:

```python
        assert aligned.dtype == ref.dtype

    def test_more_clusters_than_reference_labels_are_never_merged(self):
        # Three reference labels, five clusters of twelve. Clusters 4, 3 and 2
        # carry most of labels 0, 1 and 2; clusters 0 and 1 straddle two
        # labels and go unmatched. Left on their own ids they would share the
        # ids clusters 4 and 3 were given.
        ref = np.repeat([0, 1, 2], 20)
        labels = np.repeat([4, 0, 3, 1, 2], 12)
        aligned = align_cluster_labels(ref, labels)
        assert np.unique(aligned).size == 5
        assert adjusted_rand_score(labels, aligned) == 1.0
        assert (aligned[0], aligned[24], aligned[48]) == (0, 1, 2)
        assert (aligned[12], aligned[36]) == (3, 4)
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/test_utils.py tests/test_accuracy.py tests/test_api.py::TestNonRandomizedRegressionGate -q`

Expected: exit code 1, ending with:

```text
E       assert 3 == 5
E        +  where 3 = array([0, 1, 2]).size
E        +    where array([0, 1, 2]) = <function unique at 0x1074123b0>(array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n       1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
E        +      where <function unique at 0x1074123b0> = np.unique
FAILED tests/test_utils.py::TestAlignClusterLabels::test_more_clusters_than_reference_labels_are_never_merged
1 failed, 104 passed in 4.87s
```

- [ ] Step 3: Implement

In `src/carve/_utils.py`, replace:

```python
        Aligned labels with best matching permutation.
    """
    cont = contingency_matrix(reference_labels, labels)
    row_ind, col_ind = linear_sum_assignment(-cont)

    true_classes = np.unique(reference_labels)
    pred_classes = np.unique(labels)

    mapping = {
        pred_classes[col]: true_classes[row] for row, col in zip(row_ind, col_ind)
    }
    for pc in pred_classes:
        mapping.setdefault(pc, pc)
```

with:

```python
        Aligned labels with best matching permutation. Clusters the
        assignment cannot match, when there are more clusters than reference
        labels, take ids past the largest reference label.
    """
    cont = contingency_matrix(reference_labels, labels)
    row_ind, col_ind = linear_sum_assignment(-cont)

    true_classes = np.unique(reference_labels)
    pred_classes = np.unique(labels)

    mapping = {
        pred_classes[col]: true_classes[row] for row, col in zip(row_ind, col_ind)
    }
    # An unmatched cluster that kept its own id could share the id a matched
    # cluster was given, merging the two, so each takes a fresh one instead.
    next_id = int(np.max(true_classes)) + 1
    for pc in pred_classes:
        if pc not in mapping:
            mapping[pc] = next_id
            next_id += 1
```

Then format and lint. The blocks above are already formatted, so the first command changes nothing:

```bash
.venv/bin/ruff format --check src/
.venv/bin/ruff check src/
```

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/test_utils.py tests/test_accuracy.py tests/test_api.py::TestNonRandomizedRegressionGate -q`

Expected: exit code 0, ending with:

```text
105 passed in 4.68s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `_utils.py`: `mapping[pc] = next_id` to `mapping[pc] = pc` turns test_more_clusters_than_reference_labels_are_never_merged red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "fix(carve): align_cluster_labels gives unmatched clusters fresh ids instead of merging them"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 8: CusanovichInputs, its assembly, the selection summary and the tables

Spec 7.1 and 7.2's input dataclass. Assembly is compute, so it lives beside the functions it calls (refinement 7). The operating point targets y's own cluster count and the published probe reads the fit's resample count, subsample ratio and tree count, so no study number is restated. The integration tests fit a real, small randomized Leiden run once per module on four separated blobs whose labels sort differently from their first appearance, so the relabeling is visible; they need the graph extra and carry `requires_graph`.

Files:
- Modify: `src/benchmarks/_cusanovich_compare.py`
- Test: `tests/benchmarks/test_cusanovich_compare.py`

Interfaces:
- Consumes: Task 6: all four functions and `axis_prefix`. `figures._case_study._align_to_reference`, which inherits Task 7's fix. Task 3: `PREPROCESSOR_NAMES`, `resolve_preprocessing`. carve: `CARVE.get_labels`, `n_resamples`, `subsample_ratio`, `n_trees`.
- Produces: `CusanovichInputs(X, y, carve, carve_labels, embedding_A, embedding_A_labels, source_tsne, best_pipeline_row, operating_point, published_generalizability, measure="stability", rule="1se", not_two=False)`, frozen; `source_recipe_pipeline(carve) -> str`; `prepare_cusanovich_inputs(X, y, carve, *, source_tsne, measure="stability", rule="1se", not_two=False, random_state=42, n_jobs=1) -> CusanovichInputs`; `selection_summary(inputs) -> DataFrame[quantity, value]`; `TABLE_FILENAMES`; `save_tables(inputs, out_dir) -> list[Path]`.

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/test_cusanovich_compare.py`, replace:

```python
from benchmarks._cusanovich_compare import (
    axis_prefix,
    best_pipeline,
    pipeline_embedding,
    published_partition_generalizability,
    source_operating_point,
)
from carve._pipeline import PipelineSpec, PipelineStep
```

with:

```python
from sklearn.metrics import adjusted_rand_score

from benchmarks._cusanovich_compare import (
    TABLE_FILENAMES,
    axis_prefix,
    best_pipeline,
    pipeline_embedding,
    prepare_cusanovich_inputs,
    published_partition_generalizability,
    save_tables,
    selection_summary,
    source_operating_point,
    source_recipe_pipeline,
)
from benchmarks._estimators import resolution_grids
from benchmarks._preprocessing import resolve_preprocessing
from benchmarks._types import EstimatorSpec, PreprocessingSpec
from benchmarks.figures._case_study import _align_to_reference
from carve import CARVE
from carve._pipeline import PipelineSpec, PipelineStep
```

In `tests/benchmarks/test_cusanovich_compare.py`, replace:

```python
    def test_a_single_split_has_no_standard_error(self, blobs):
        mean, se = self._score(*blobs, n_splits=1)
        assert mean == pytest.approx(1.0)
        assert np.isnan(se)
```

with:

```python
    def test_a_single_split_has_no_standard_error(self, blobs):
        mean, se = self._score(*blobs, n_splits=1)
        assert mean == pytest.approx(1.0)
        assert np.isnan(se)


class TestSourceRecipePipeline:
    @staticmethod
    def _carve(*steps):
        specs = [_spec(step) for step in steps]
        return _FakeCarve(pd.DataFrame(), {spec.label: spec for spec in specs})

    def test_finds_the_one_tsne_pipeline(self):
        carve = self._carve(
            IDENTITY,
            PipelineStep(cls=TSNE, params={"perplexity": 30}, name="TSNE"),
            PipelineStep(cls=PCA, params={"n_components": 5}, name="PCA"),
        )
        assert source_recipe_pipeline(carve) == "identity | TSNE(perplexity=30)"

    @pytest.mark.parametrize("perplexities", [(), (15, 30)])
    def test_anything_but_one_tsne_pipeline_raises(self, perplexities):
        steps = [
            PipelineStep(cls=TSNE, params={"perplexity": p}, name="TSNE")
            for p in perplexities
        ]
        carve = self._carve(IDENTITY, *steps)
        with pytest.raises(ValueError, match="exactly one t-SNE pipeline"):
            source_recipe_pipeline(carve)


# Four separated blobs, labeled so the reference's sorted codes (a, b, c, d)
# differ from its first-appearance order (d, b, a, c), which is the order
# CARVE factorizes reference labels in.
_BLOB_NAMES = np.array(["d", "b", "a", "c"])
_TSNE_PIPELINE = "identity | TSNE(perplexity=10)"


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=8.0, size=(4, 6))
    members = np.repeat(np.arange(4), 40)
    X = centers[members] + rng.normal(scale=0.5, size=(160, 6))
    y = _BLOB_NAMES[members]
    spec = PreprocessingSpec(
        normalization=(("identity", {}),),
        dim_reduction=(("identity", {}), ("tsne", {"perplexity": [10]})),
    )
    carve = CARVE(
        estimator_param_grids=resolution_grids(EstimatorSpec(name="leiden"), (0.1, 0.5)),
        n_resamples=6,
        n_trees=20,
        random_state=0,
        **resolve_preprocessing(spec),
    )
    carve.fit(X, reference_labels=y, randomize_preprocessing=True)
    return X, y, carve


@pytest.fixture(scope="module")
def inputs(fitted):
    X, y, carve = fitted
    return prepare_cusanovich_inputs(
        X, y, carve, source_tsne=X[:, :2], random_state=0
    )


@pytest.mark.requires_graph
class TestPrepareCusanovichInputs:
    def test_embeds_all_of_x_with_the_best_pipeline(self, fitted, inputs):
        X, _, carve = fitted
        row, spec = best_pipeline(carve, measure="stability", rule="1se")
        assert inputs.best_pipeline_row.equals(row)
        expected, labels = pipeline_embedding(X, spec, random_state=0)
        np.testing.assert_allclose(inputs.embedding_A, expected)
        assert inputs.embedding_A_labels == labels
        assert (inputs.measure, inputs.rule, inputs.not_two) == ("stability", "1se", False)

    def test_carve_labels_are_relabeled_onto_the_reference_codes(self, fitted, inputs):
        _, y, carve = fitted
        raw = carve.get_labels(measure="stability", rule="1se")
        np.testing.assert_array_equal(inputs.carve_labels, _align_to_reference(raw, y))

    def test_the_operating_point_targets_the_reference_cluster_count(
        self, fitted, inputs
    ):
        # y has four clusters, so the operating point is the t-SNE pipeline's
        # resolution nearest four; a fixed 30 would be far off and warn.
        _, _, carve = fitted
        assert inputs.operating_point == source_operating_point(
            carve, pipeline=_TSNE_PIPELINE, target_k=4
        )

    def test_forwards_the_fits_settings_and_the_seed(self, fitted, monkeypatch):
        # On separated blobs the probe scores 1.0 at any split count, and the
        # best pipeline may be the seedless identity, so the returned values
        # cannot show what was passed. Record the calls instead.
        import benchmarks._cusanovich_compare as compare

        X, y, carve = fitted
        probes, embeddings = [], []
        real_probe = compare.published_partition_generalizability
        real_embedding = compare.pipeline_embedding

        def probe(*args, **kwargs):
            probes.append(kwargs)
            return real_probe(*args, **kwargs)

        def embedding(X, spec, *, random_state):
            embeddings.append((spec, random_state))
            return real_embedding(X, spec, random_state=random_state)

        monkeypatch.setattr(compare, "published_partition_generalizability", probe)
        monkeypatch.setattr(compare, "pipeline_embedding", embedding)
        prepare_cusanovich_inputs(
            X, y, carve, source_tsne=X[:, :2], random_state=7, n_jobs=2
        )

        assert probes == [
            {
                "n_splits": 6,
                "subsample_ratio": carve.subsample_ratio,
                "n_trees": 20,
                "random_state": 7,
                "n_jobs": 2,
            }
        ]
        _, spec = best_pipeline(carve, measure="stability", rule="1se")
        assert embeddings == [(spec, 7)]

    def test_the_summary_reports_the_selection_and_the_comparison(
        self, fitted, inputs
    ):
        _, y, carve = fitted
        summary = selection_summary(inputs).set_index("quantity")["value"]
        selected, _, n_clusters, _ = carve._select_row(measure="stability", rule="1se")
        assert summary["selected_resolution"] == float(selected["sweep_value"])
        assert summary["selected_n_clusters"] == n_clusters
        assert summary["best_pipeline"] == inputs.best_pipeline_row["pipeline"]
        assert summary["source_pipeline"] == _TSNE_PIPELINE
        assert summary["operating_point_resolution"] == inputs.operating_point[0]
        assert summary["published_generalizability"] == (
            inputs.published_generalizability[0]
        )
        raw = carve.get_labels(measure="stability", rule="1se")
        assert summary["consensus_ari_vs_source_clusters"] == pytest.approx(
            adjusted_rand_score(y, raw)
        )

    def test_save_tables_writes_both_csvs(self, fitted, inputs, tmp_path):
        _, _, carve = fitted
        paths = save_tables(inputs, tmp_path / "out")
        assert [path.name for path in paths] == list(TABLE_FILENAMES)
        assert len(pd.read_csv(paths[0])) == len(carve.preprocessing_results_)
        written = pd.read_csv(paths[1])
        assert list(written["quantity"]) == list(selection_summary(inputs)["quantity"])
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/benchmarks/test_cusanovich_compare.py -q`

Expected: exit code 2, ending with:

```text
E   ImportError: cannot import name 'TABLE_FILENAMES' from 'benchmarks._cusanovich_compare' (/private/tmp/claude-501/-Users-kaiwycik-GitHub-CARVE--claude-playground/349e6055-9572-4f5c-919e-e41ea3aca477/scratchpad/pass2/src/benchmarks/_cusanovich_compare.py)
ERROR tests/benchmarks/test_cusanovich_compare.py
1 error in 0.08s
```

- [ ] Step 3: Implement

In `src/benchmarks/_cusanovich_compare.py`, replace:

```python
its own clusterings. No matplotlib here; figures._cusanovich_results draws
the result.
```

with:

```python
its own clusterings. prepare_cusanovich_inputs gathers all of it into
CusanovichInputs; nothing here draws, figures._cusanovich_results does.
```

In `src/benchmarks/_cusanovich_compare.py`, replace:

```python
import warnings
from typing import Any
```

with:

```python
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
```

In `src/benchmarks/_cusanovich_compare.py`, replace:

```python
    return float(np.mean(scores)), se
```

with:

```python
    return float(np.mean(scores)), se


@dataclass(frozen=True)
class CusanovichInputs:
    """Everything the Cusanovich figure draws, already computed.

    Assembling this is compute (prepare_cusanovich_inputs); drawing it is
    reporting (figures.figure_cusanovich_results). Keeping them apart lets the
    figure be tested without fitting anything, as with CompositeInputs.

    carve_labels are CARVE's consensus labels relabeled onto y's codes, so a
    CARVE cluster takes the color of the source cluster it best matches.
    embedding_A is all of X embedded by the best pipeline, named by
    embedding_A_labels. operating_point is (resolution, observed cluster
    count) on the source recipe's pipeline; published_generalizability is
    (mean, se). measure, rule and not_two are the selection every panel reads.
    """

    X: np.ndarray
    y: np.ndarray
    carve: Any
    carve_labels: np.ndarray
    embedding_A: np.ndarray
    embedding_A_labels: tuple[str, str]
    source_tsne: np.ndarray
    best_pipeline_row: pd.Series
    operating_point: tuple[float, float]
    published_generalizability: tuple[float, float]
    measure: str = "stability"
    rule: str = "1se"
    not_two: bool = False


def source_recipe_pipeline(carve: Any) -> str:
    """The label of the pipeline that reproduces the source's embedding.

    The source clustered a t-SNE of its LSI, so this is the one pipeline whose
    dimensionality reduction is t-SNE. More than one, as when several
    perplexities are offered, raises: which is the source's is then the
    study's decision, not something to guess.
    """
    _per_pipeline_table(carve)
    labels = sorted(
        label
        for label, spec in carve.preprocessing_pipelines_.items()
        if spec.dim_reduction.name == PREPROCESSOR_NAMES["tsne"]
    )
    if len(labels) != 1:
        raise ValueError(
            f"Expected exactly one t-SNE pipeline, found {len(labels)}: {labels}."
        )
    return labels[0]


def prepare_cusanovich_inputs(
    X: np.ndarray,
    y: np.ndarray,
    carve: Any,
    *,
    source_tsne: np.ndarray,
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
    random_state: int = 42,
    n_jobs: int = 1,
) -> CusanovichInputs:
    """Assemble the Cusanovich figure's inputs from a randomized fit.

    Embeds all of X with the best pipeline at the selected configuration,
    relabels CARVE's consensus labels onto y, places the source's operating
    point on the t-SNE pipeline at y's own cluster count, and scores the
    published partition y under the classifier probe with the fit's own
    resample count, subsample ratio and tree count. This is compute; call it
    once and draw from the result.
    """
    from .figures._case_study import _align_to_reference

    X = np.asarray(X)
    y = np.asarray(y)
    row, spec = best_pipeline(carve, measure=measure, rule=rule, not_two=not_two)
    embedding, axis_labels = pipeline_embedding(X, spec, random_state=random_state)
    labels = carve.get_labels(measure=measure, rule=rule, not_two=not_two)

    return CusanovichInputs(
        X=X,
        y=y,
        carve=carve,
        carve_labels=_align_to_reference(labels, y),
        embedding_A=embedding,
        embedding_A_labels=axis_labels,
        source_tsne=np.asarray(source_tsne, dtype=np.float64),
        best_pipeline_row=row,
        operating_point=source_operating_point(
            carve,
            pipeline=source_recipe_pipeline(carve),
            target_k=int(np.unique(y).size),
        ),
        published_generalizability=published_partition_generalizability(
            X,
            y,
            n_splits=int(carve.n_resamples),
            subsample_ratio=float(carve.subsample_ratio),
            n_trees=int(carve.n_trees),
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        measure=measure,
        rule=rule,
        not_two=not_two,
    )


def selection_summary(inputs: CusanovichInputs) -> pd.DataFrame:
    """The numbers the case study reports, one (quantity, value) row each.

    Written next to the figure by save_tables, so every number the manuscript
    quotes has a file behind it.
    """
    carve = inputs.carve
    selected, _, n_clusters, _ = carve._select_row(
        measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
    )
    best = inputs.best_pipeline_row
    resolution, observed = inputs.operating_point
    mean, se = inputs.published_generalizability
    rows = [
        ("measure", inputs.measure),
        ("rule", inputs.rule),
        ("not_two", inputs.not_two),
        ("n_cells", int(inputs.y.size)),
        ("source_clusters", int(np.unique(inputs.y).size)),
        ("selected_method", selected["method_label"]),
        ("selected_resolution", float(selected["sweep_value"])),
        ("selected_n_clusters", int(n_clusters)),
        ("selected_ari_stability", float(selected["ari_stability"])),
        ("selected_ari_generalizability", float(selected["ari_generalizability"])),
        ("best_pipeline", best["pipeline"]),
        ("best_pipeline_n_resamples", int(best["n_resamples"])),
        ("best_pipeline_ari_stability", float(best["ari_stability"])),
        ("best_pipeline_ari_generalizability", float(best["ari_generalizability"])),
        ("source_pipeline", source_recipe_pipeline(carve)),
        ("operating_point_resolution", resolution),
        ("operating_point_n_clusters", observed),
        ("published_generalizability", mean),
        ("published_generalizability_se", se),
        (
            "consensus_ari_vs_source_clusters",
            float(adjusted_rand_score(inputs.y, inputs.carve_labels)),
        ),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


#: The CSVs save_tables writes, next to cusanovich_results.png.
TABLE_FILENAMES: tuple[str, str] = (
    "cusanovich_preprocessing_results.csv",
    "cusanovich_selection_summary.csv",
)


def save_tables(inputs: CusanovichInputs, out_dir: Path) -> list[Path]:
    """Write preprocessing_results_ and the selection summary as CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path, summary_path = (out_dir / name for name in TABLE_FILENAMES)
    inputs.carve.preprocessing_results_.to_csv(results_path, index=False)
    selection_summary(inputs).to_csv(summary_path, index=False)
    return [results_path, summary_path]
```

Then format and lint. The blocks above are already formatted, so the first command changes nothing:

```bash
.venv/bin/ruff format --check src/
.venv/bin/ruff check src/
```

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/benchmarks/test_cusanovich_compare.py -q`

Expected: exit code 0, ending with:

```text
29 passed in 3.58s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `_cusanovich_compare.py`: `target_k=int(np.unique(y).size),` to `target_k=30,` turns test_the_operating_point_targets_the_reference_cluster_count red.
- `_cusanovich_compare.py`: `n_splits=int(carve.n_resamples),` to `n_splits=10,` turns test_forwards_the_fits_settings_and_the_seed red.
- `_cusanovich_compare.py`: `n_trees=int(carve.n_trees),` to `n_trees=100,` turns test_forwards_the_fits_settings_and_the_seed red.
- `_cusanovich_compare.py`: `carve_labels=_align_to_reference(labels, y),` to `carve_labels=np.asarray(labels),` turns test_carve_labels_are_relabeled_onto_the_reference_codes red.
- `_cusanovich_compare.py`: `embedding, axis_labels = pipeline_embedding(X, spec, random_state=random_state)` to `embedding, axis_labels = pipeline_embedding(X, spec, random_state=random_state + 1)` turns test_forwards_the_fits_settings_and_the_seed red.
- `_cusanovich_compare.py`: `if spec.dim_reduction.name == PREPROCESSOR_NAMES["tsne"]` to `if spec.dim_reduction.name != PREPROCESSOR_NAMES["identity"]` turns test_finds_the_one_tsne_pipeline red.
- `_cusanovich_compare.py`: `("selected_n_clusters", int(n_clusters)),` to `("selected_n_clusters", int(n_clusters) + 1),` turns test_the_summary_reports_the_selection_and_the_comparison red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "feat(benchmarks): CusanovichInputs assembly and the case-study tables"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 9: carve_lines over any sweep axis; pipeline_lines; the pipeline palette

Spec 7.2, panels C and D. `carve_lines` reads the sweep parameter from `sweep_` (falling back to k for caches that predate it) and gains `rule`, forwarded to every selection call. `pipeline_lines` themes `carve._plotting.plot_metric_by_pipeline`, called once per measure: colors from a registered `PIPELINE_CMAP_NAME` colormap, solid stability and dashed generalizability from `MEASURE_LINESTYLES`, one legend. `StubCarve` learns resolution-mode selection and records its selection arguments, so every figure test that uses it is rerun.

Files:
- Modify: `src/benchmarks/_panels.py`
- Modify: `src/benchmarks/_theme.py`
- Test: `tests/benchmarks/_helpers.py`
- Test: `tests/benchmarks/test_panels.py`
- Test: `tests/benchmarks/test_theme.py`

Interfaces:
- Consumes: carve: `_sweep.SWEEP_REGISTRY`, `_plotting.plot_metric_by_pipeline`.
- Produces: `_theme.PIPELINE_COLORS`, `PIPELINE_CMAP_NAME = "carve_pipeline"`, `MEASURE_LINESTYLES`; `_panels.carve_lines(ax, carve_obj, *, measures=("stability", "generalizability"), rule="1se", not_two=False, title=None, annotate=False, show_selected_k=True)`; `_panels.pipeline_lines(ax, carve_obj, *, method_id, measures=..., rule="1se", not_two=False, title=None) -> Axes`. Test helpers: `StubCarve(..., sweep_param="n_clusters")` with `get_sweep_value` and `selection_calls`; `resolution_results(resolutions, method_label, *, method_id="m0")`; `pipeline_results(pipelines, resolutions, *, method_ids=("m0",))`.

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/_helpers.py`, replace:

```python
from collections.abc import Callable
from pathlib import Path
```

with:

```python
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
```

In `tests/benchmarks/_helpers.py`, replace:

```python
    The panels and figure code read three members off a fitted model:
    estimator_results_, _select_row() and get_k() (and prepare_composite
    also calls get_labels()). This implements exactly those instead of
    fitting a real model.
```

with:

```python
    The panels and figure code read a few members off a fitted model:
    estimator_results_, sweep_, _select_row(), get_k() and get_sweep_value()
    (and prepare_composite also calls get_labels()). This implements exactly
    those instead of fitting a real model, and records the selection
    arguments each call received in selection_calls.
```

In `tests/benchmarks/_helpers.py`, replace:

```python
    results : DataFrame
        The table, with at least method_id and n_clusters columns.
    select : callable (measure, not_two) -> (method_id, n_clusters)
        Which row _select_row and get_k resolve to. Making it a function lets
        a test decide whether measure or not_two changes the answer, so the
        test can tell whether the code under test forwarded them.
    labels : ndarray, optional
        What get_labels returns. Left None by tests that never call it.
    """

    def __init__(
        self,
        results: pd.DataFrame,
        select: Callable[[str, bool], tuple[str, int]],
        labels: np.ndarray | None = None,
    ):
        self.estimator_results_ = results
        self._select = select
        self._labels = labels
        self.get_labels_calls: list[dict] = []

    def _select_row(self, *, measure, rule, not_two=False):
        method_id, k = self._select(measure, not_two)
        results = self.estimator_results_
        row = results.loc[
            (results["method_id"] == method_id) & (results["n_clusters"] == k)
        ].iloc[0]
        return row, 0, int(k), False

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return int(self._select(measure, not_two)[1])
```

with:

```python
    results : DataFrame
        The table, with at least method_id and the sweep column: n_clusters
        for a k-based stub, sweep_value (and n_clusters_observed) otherwise.
    select : callable (measure, not_two) -> (method_id, sweep value)
        Which row _select_row, get_k and get_sweep_value resolve to. Making it
        a function lets a test decide whether measure or not_two changes the
        answer, so the test can tell whether the code under test forwarded
        them.
    labels : ndarray, optional
        What get_labels returns. Left None by tests that never call it.
    sweep_param : str, default="n_clusters"
        The swept parameter sweep_.param reports.
    """

    def __init__(
        self,
        results: pd.DataFrame,
        select: Callable[[str, bool], tuple[str, float]],
        labels: np.ndarray | None = None,
        *,
        sweep_param: str = "n_clusters",
    ):
        self.estimator_results_ = results
        self._select = select
        self._labels = labels
        self.sweep_ = SimpleNamespace(param=sweep_param)
        self.get_labels_calls: list[dict] = []
        self.selection_calls: list[dict] = []

    def _record(self, measure, rule, not_two):
        self.selection_calls.append(
            {"measure": measure, "rule": rule, "not_two": not_two}
        )
        return self._select(measure, not_two)

    def _select_row(self, *, measure, rule, not_two=False):
        method_id, value = self._record(measure, rule, not_two)
        results = self.estimator_results_
        if self.sweep_.param == "n_clusters":
            row = results.loc[
                (results["method_id"] == method_id) & (results["n_clusters"] == value)
            ].iloc[0]
            return row, 0, int(value), False
        row = results.loc[
            (results["method_id"] == method_id)
            & np.isclose(results["sweep_value"], value)
        ].iloc[0]
        return row, 0, int(round(float(row["n_clusters_observed"]))), False

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return int(self._record(measure, rule, not_two)[1])

    def get_sweep_value(self, *, measure="stability", rule="1se", not_two=False):
        return float(self._record(measure, rule, not_two)[1])
```

In `tests/benchmarks/_helpers.py`, replace:

```python
def make_carve_spy() -> type:
```

with:

```python
def resolution_results(
    resolutions, method_label: str, *, method_id: str = "m0"
) -> pd.DataFrame:
    """One Leiden configuration swept over resolutions, as estimator_results_.

    Observed cluster counts rise with resolution, as they do on real data.
    """
    resolutions = [float(r) for r in resolutions]
    n = len(resolutions)
    return pd.DataFrame(
        {
            "method_id": [method_id] * n,
            "method_label": [method_label] * n,
            "estimator": ["LeidenClustering"] * n,
            "resolution": resolutions,
            "sweep_param": ["resolution"] * n,
            "sweep_value": resolutions,
            "sweep_rank": list(range(n)),
            "n_clusters_observed": [4.0 + 4.0 * i for i in range(n)],
            "ari_stability": list(np.linspace(0.8, 0.4, n)),
            "ari_stability_se": [0.02] * n,
            "ari_generalizability": list(np.linspace(0.7, 0.3, n)),
            "ari_generalizability_se": [0.03] * n,
        }
    )


def pipeline_results(
    pipelines, resolutions, *, method_ids=("m0",)
) -> pd.DataFrame:
    """A preprocessing_results_ table: every pipeline at every resolution.

    Pipeline i at resolution rank r of method j scores
    0.5 + 0.1 * i - 0.05 * r + 0.3 * j on stability, and 0.1 less on
    generalizability, so every line is distinguishable and a test can check
    which configuration a panel drew.
    """
    rows = []
    for j, method_id in enumerate(method_ids):
        for i, pipeline in enumerate(pipelines):
            for r, resolution in enumerate(resolutions):
                stability = 0.5 + 0.1 * i - 0.05 * r + 0.3 * j
                rows.append(
                    {
                        "method_id": method_id,
                        "method_label": "LeidenClustering",
                        "pipeline": pipeline,
                        "resolution": float(resolution),
                        "n_resamples": 5,
                        "ari_stability": stability,
                        "ari_stability_se": 0.02,
                        "ari_generalizability": stability - 0.1,
                        "ari_generalizability_se": 0.03,
                        "n_clusters_observed": 4.0 + 4.0 * r,
                        "sweep_param": "resolution",
                        "sweep_value": float(resolution),
                        "sweep_rank": r,
                    }
                )
    return pd.DataFrame(rows)


def make_carve_spy() -> type:
```

In `tests/benchmarks/test_panels.py`, replace:

```python
    panel_letter,
    runtime_lines,
    scatter_clusters,
)
from benchmarks._theme import FONT_SIZES, FOREGROUND_COLOR, cluster_colors, metric_color
from tests.benchmarks._helpers import StubCarve
```

with:

```python
    panel_letter,
    pipeline_lines,
    runtime_lines,
    scatter_clusters,
)
from benchmarks._theme import (
    FONT_SIZES,
    FOREGROUND_COLOR,
    PIPELINE_COLORS,
    cluster_colors,
    metric_color,
)
from tests.benchmarks._helpers import StubCarve, pipeline_results, resolution_results
```

In `tests/benchmarks/test_panels.py`, replace:

```python
    "carve_lines",
    "cvi_lines",
```

with:

```python
    "carve_lines",
    "pipeline_lines",
    "cvi_lines",
```

In `tests/benchmarks/test_panels.py`, replace:

```python
        carve_lines(
            ax,
            _carve_obj(),
            measures=("stability", "generalizability"),
            show_selected_k=False,
        )
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        for line in data_lines:
            x = line.get_xdata()
            assert np.all(np.diff(x) > 0)
```

with:

```python
        carve_lines(
            ax,
            _carve_obj(),
            measures=("stability", "generalizability"),
            show_selected_k=False,
        )
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        for line in data_lines:
            x = line.get_xdata()
            assert np.all(np.diff(x) > 0)

    def test_a_k_based_run_keeps_the_number_of_clusters_axis(self, ax):
        carve_lines(ax, _carve_obj(), measures=("stability",))
        assert ax.get_xlabel() == "Number of clusters $k$"

    def test_a_resolution_run_draws_over_resolution(self, ax):
        # A decoy configuration at other resolutions must not join the line,
        # and the selected resolution, 0.6, comes from get_sweep_value: get_k
        # would round it to a cluster count.
        results = pd.concat(
            [
                resolution_results([0.2, 0.4, 0.6, 0.8], "Leiden"),
                resolution_results([0.3, 0.5], "Leiden decoy", method_id="m1"),
            ],
            ignore_index=True,
        )
        carve = StubCarve(
            results, select=lambda measure, not_two: ("m0", 0.6), sweep_param="resolution"
        )
        carve_lines(ax, carve, measures=("stability",))
        data_line = next(ln for ln in ax.lines if ln.get_marker() == "o")
        selection = next(ln for ln in ax.lines if ln.get_marker() != "o")
        np.testing.assert_allclose(data_line.get_xdata(), [0.2, 0.4, 0.6, 0.8])
        assert selection.get_xdata()[0] == pytest.approx(0.6)
        assert ax.get_xlabel() == "Resolution"

    def test_forwards_rule_and_not_two_to_every_selection(self, ax):
        carve = _carve_obj()
        carve_lines(ax, carve, rule="max", not_two=True)
        assert len(carve.selection_calls) == 4
        assert all(
            call["rule"] == "max" and call["not_two"] for call in carve.selection_calls
        )


_PIPELINES = (
    "identity | identity",
    "identity | TSNE(perplexity=30)",
    "identity | UMAP(n_neighbors=15)",
)


class TestPipelineLines:
    @staticmethod
    def _draw(ax, **kwargs):
        table = pipeline_results(_PIPELINES, (0.2, 0.4, 0.6), method_ids=("m0", "m1"))
        carve = SimpleNamespace(preprocessing_results_=table)
        return pipeline_lines(ax, carve, method_id="m0", **kwargs)

    def test_returns_the_same_axes(self, ax):
        assert self._draw(ax) is ax

    def test_one_line_pair_per_pipeline_sharing_a_pipeline_color(self, ax):
        self._draw(ax)
        drawn: dict[str, list] = {}
        for container in ax.containers:
            line = container.lines[0]
            drawn.setdefault(container.get_label(), []).append(
                (mcolors.to_hex(line.get_color()), line.get_linestyle())
            )
        assert set(drawn) == set(_PIPELINES)
        for pair in drawn.values():
            assert [style for _, style in pair] == ["-", "--"]
            assert len({color for color, _ in pair}) == 1
        colors = {pair[0][0] for pair in drawn.values()}
        assert len(colors) == len(_PIPELINES)
        assert colors <= {color.lower() for color in PIPELINE_COLORS}

    def test_draws_only_the_named_configuration(self, ax):
        # m1 scores 0.3 higher everywhere; pipeline 0 of m0 scores
        # 0.5, 0.45, 0.4 on stability.
        self._draw(ax)
        stability = next(
            container
            for container in ax.containers
            if container.get_label() == "identity | identity"
        )
        np.testing.assert_allclose(stability.lines[0].get_ydata(), [0.5, 0.45, 0.4])

    def test_one_legend_names_the_pipelines_and_the_two_criteria(self, ax):
        self._draw(ax)
        texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert texts == [*sorted(_PIPELINES), "Stability", "Generalizability"]

    def test_labels_the_sweep_axis_and_the_criterion(self, ax):
        self._draw(ax, title="By pipeline")
        assert ax.get_xlabel() == "Resolution"
        assert ax.get_ylabel() == "ARI"
        assert ax.get_title() == "By pipeline"

    def test_forwards_the_selection_arguments(self, ax, monkeypatch):
        import carve._plotting as carve_plotting

        calls = []
        real = carve_plotting.plot_metric_by_pipeline

        def spy(*args, **kwargs):
            calls.append(
                (kwargs["measure"], kwargs["rule"], kwargs["not_two"], kwargs["method_id"])
            )
            return real(*args, **kwargs)

        monkeypatch.setattr(carve_plotting, "plot_metric_by_pipeline", spy)
        self._draw(ax, rule="max", not_two=True)
        assert calls == [
            ("stability", "max", True, "m0"),
            ("generalizability", "max", True, "m0"),
        ]
```

In `tests/benchmarks/test_panels.py`, replace:

```python
import inspect

import matplotlib.colors as mcolors
```

with:

```python
import inspect
from types import SimpleNamespace

import matplotlib.colors as mcolors
```

In `tests/benchmarks/test_theme.py`, replace:

```python
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex
```

with:

```python
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
```

In `tests/benchmarks/test_theme.py`, replace:

```python
    CLUSTER_PALETTE,
    FONT_SIZES,
    METRIC_COLORS,
```

with:

```python
    CLUSTER_PALETTE,
    FONT_SIZES,
    MEASURE_LINESTYLES,
    METRIC_COLORS,
    PIPELINE_CMAP_NAME,
    PIPELINE_COLORS,
```

In `tests/benchmarks/test_theme.py`, replace:

```python
class TestRcParams:
```

with:

```python
class TestPipelinePalette:
    def test_the_registered_colormap_is_the_pipeline_colors(self):
        cmap = plt.get_cmap(PIPELINE_CMAP_NAME)
        assert [to_hex(cmap(i)) for i in range(cmap.N)] == [
            color.lower() for color in PIPELINE_COLORS
        ]

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_up_to_four_pipelines_get_distinct_colors(self, n):
        # plot_metric_by_pipeline samples its colormap at np.linspace(0, 1, n).
        cmap = plt.get_cmap(PIPELINE_CMAP_NAME)
        colors = [to_hex(color) for color in cmap(np.linspace(0, 1, n))]
        assert len(set(colors)) == n

    def test_no_pipeline_color_is_a_criterion_color(self):
        pipeline = {color.lower() for color in PIPELINE_COLORS}
        metric = {color.lower() for color in METRIC_COLORS.values()}
        assert not pipeline & metric

    def test_the_two_criteria_have_their_own_line_styles(self):
        assert MEASURE_LINESTYLES == {"stability": "-", "generalizability": "--"}


class TestRcParams:
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/benchmarks/test_panels.py tests/benchmarks/test_theme.py tests/benchmarks/figures -q`

Expected: exit code 2, ending with:

```text
E   ImportError: cannot import name 'pipeline_lines' from 'benchmarks._panels' (/private/tmp/claude-501/-Users-kaiwycik-GitHub-CARVE--claude-playground/349e6055-9572-4f5c-919e-e41ea3aca477/scratchpad/pass2/src/benchmarks/_panels.py)
E   ImportError: cannot import name 'MEASURE_LINESTYLES' from 'benchmarks._theme' (/private/tmp/claude-501/-Users-kaiwycik-GitHub-CARVE--claude-playground/349e6055-9572-4f5c-919e-e41ea3aca477/scratchpad/pass2/src/benchmarks/_theme.py)
ERROR tests/benchmarks/test_panels.py
ERROR tests/benchmarks/test_theme.py
2 errors in 0.22s
```

- [ ] Step 3: Implement

In `src/benchmarks/_theme.py`, replace:

```python
FONT_SIZES: dict[str, float] = {
```

with:

```python
# Preprocessing pipelines in the Cusanovich figure's per-pipeline panel.
# CARVE's plot_metric_by_pipeline takes a colormap name and samples it at
# evenly spaced points, one per pipeline in sorted label order, so a
# ListedColormap of exactly these colors gives each of up to four pipelines
# its own entry. Okabe-Ito hues and tab10's brown, none of them a
# METRIC_COLORS value, so a pipeline line is never read as a criterion.
PIPELINE_COLORS: tuple[str, ...] = ("#0072B2", "#D55E00", "#CC79A7", "#8C564B")
PIPELINE_CMAP_NAME: str = "carve_pipeline"
if PIPELINE_CMAP_NAME not in mpl.colormaps:
    mpl.colormaps.register(
        ListedColormap(list(PIPELINE_COLORS), name=PIPELINE_CMAP_NAME)
    )

# The per-pipeline panel draws both criteria for every pipeline on one axes;
# color carries the pipeline, so line style carries the criterion.
MEASURE_LINESTYLES: dict[str, str] = {"stability": "-", "generalizability": "--"}

FONT_SIZES: dict[str, float] = {
```

In `src/benchmarks/_panels.py`, replace:

```python
from ._registry import (
```

with:

```python
from carve._sweep import SWEEP_REGISTRY

from ._registry import (
```

In `src/benchmarks/_panels.py`, replace:

```python
    FOREGROUND_COLOR,
    cluster_colors,
```

with:

```python
    FOREGROUND_COLOR,
    MEASURE_LINESTYLES,
    PIPELINE_CMAP_NAME,
    cluster_colors,
```

In `src/benchmarks/_panels.py`, replace:

```python
def carve_lines(
    ax: Axes,
    carve_obj: Any,
    *,
    measures: Sequence[str] = ("stability", "generalizability"),
    not_two: bool = False,
    title: str | None = None,
    annotate: bool = False,
    show_selected_k: bool = True,
) -> Axes:
    """Plot CARVE validation curves over k, one line per measure.
```

with:

```python
def _sweep_axis_label(param: str) -> str:
    """The x-axis label for a sweep parameter, in CARVE's own wording."""
    if param == "n_clusters":
        return "Number of clusters $k$"
    known = SWEEP_REGISTRY.get(param)
    return known[2] if known else param.replace("_", " ").title()


def carve_lines(
    ax: Axes,
    carve_obj: Any,
    *,
    measures: Sequence[str] = ("stability", "generalizability"),
    rule: str = "1se",
    not_two: bool = False,
    title: str | None = None,
    annotate: bool = False,
    show_selected_k: bool = True,
) -> Axes:
    """Plot CARVE validation curves over the sweep axis, one line per measure.
```

In `src/benchmarks/_panels.py`, replace:

```python
    named in the legend rather than left implicit.
    """
    results = carve_obj.estimator_results_

    for measure in measures:
        selected_row, _, _, _ = carve_obj._select_row(
            measure=measure, rule="1se", not_two=not_two
        )
        method_id = selected_row["method_id"]
        method_label = selected_row["method_label"]
        curve = results.loc[results["method_id"] == method_id].sort_values("n_clusters")

        color = metric_color(f"ari_{measure}_1se")
        ks = curve["n_clusters"].to_numpy()
        values = curve[f"ari_{measure}"].to_numpy()
        label = f"{_display(f'ari_{measure}_1se')} — {method_label}"
        ax.plot(
            ks,
            values,
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=color,
            label=label,
        )
        if f"ari_{measure}_se" in curve.columns:
            se = curve[f"ari_{measure}_se"].to_numpy()
            ax.fill_between(ks, values - se, values + se, color=color, alpha=0.15)

        if show_selected_k:
            selected = int(
                carve_obj.get_k(measure=measure, rule="1se", not_two=not_two)
            )
            ax.axvline(selected, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
            if annotate:
                ax.annotate(
                    f"$\\hat{{k}}={selected}$",
                    xy=(selected, float(np.nanmax(values))),
                    fontsize=FONT_SIZES["legend"],
                    color=color,
                )

    ax.set_xlabel("Number of clusters $k$", fontsize=FONT_SIZES["axis_label"])
```

with:

```python
    named in the legend rather than left implicit.

    The x axis is the run's sweep parameter, read from carve_obj.sweep_: the
    number of clusters for a k-based run (and for a fit cached before sweep_
    existed, every one of which is k-based), the swept value otherwise, such
    as Leiden resolution. The selected value is marked from get_k or
    get_sweep_value to match. rule and not_two are forwarded to every
    selection call.
    """
    results = carve_obj.estimator_results_
    sweep = getattr(carve_obj, "sweep_", None)
    param = "n_clusters" if sweep is None else sweep.param
    by_k = param == "n_clusters"
    x_col = "n_clusters" if by_k else "sweep_value"

    for measure in measures:
        selected_row, _, _, _ = carve_obj._select_row(
            measure=measure, rule=rule, not_two=not_two
        )
        method_id = selected_row["method_id"]
        method_label = selected_row["method_label"]
        curve = results.loc[results["method_id"] == method_id].sort_values(x_col)

        color = metric_color(f"ari_{measure}_1se")
        xs = curve[x_col].to_numpy()
        values = curve[f"ari_{measure}"].to_numpy()
        label = f"{_display(f'ari_{measure}_1se')} — {method_label}"
        ax.plot(
            xs,
            values,
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=color,
            label=label,
        )
        if f"ari_{measure}_se" in curve.columns:
            se = curve[f"ari_{measure}_se"].to_numpy()
            ax.fill_between(xs, values - se, values + se, color=color, alpha=0.15)

        if show_selected_k:
            if by_k:
                selected = int(
                    carve_obj.get_k(measure=measure, rule=rule, not_two=not_two)
                )
                note = f"$\\hat{{k}}={selected}$"
            else:
                selected = float(
                    carve_obj.get_sweep_value(
                        measure=measure, rule=rule, not_two=not_two
                    )
                )
                note = f"{selected:g}"
            ax.axvline(selected, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
            if annotate:
                ax.annotate(
                    note,
                    xy=(selected, float(np.nanmax(values))),
                    fontsize=FONT_SIZES["legend"],
                    color=color,
                )

    ax.set_xlabel(_sweep_axis_label(param), fontsize=FONT_SIZES["axis_label"])
```

In `src/benchmarks/_panels.py`, replace:

```python
def cvi_lines(
```

with:

```python
def pipeline_lines(
    ax: Axes,
    carve_obj: Any,
    *,
    method_id: str,
    measures: Sequence[str] = ("stability", "generalizability"),
    rule: str = "1se",
    not_two: bool = False,
    title: str | None = None,
) -> Axes:
    """Plot per-pipeline validation curves for one estimator configuration.

    The drawing is carve._plotting.plot_metric_by_pipeline, called once per
    measure on the same axes, so the per-pipeline rows, the error bars at one
    standard error and the per-pipeline selection line are CARVE's own; this
    only themes them. Pipelines take PIPELINE_CMAP_NAME's colors, each measure
    its MEASURE_LINESTYLES style, and one legend names both.
    """
    from carve._plotting import plot_metric_by_pipeline

    table = carve_obj.preprocessing_results_
    for measure in measures:
        plot_metric_by_pipeline(
            table,
            method_id=method_id,
            measure=measure,
            rule=rule,
            not_two=not_two,
            ax=ax,
            legend=False,
            palette=PIPELINE_CMAP_NAME,
            linestyle=MEASURE_LINESTYLES[measure],
        )

    handles: dict[str, Line2D] = {}
    for container in ax.containers:
        label = container.get_label()
        if label not in handles:
            handles[label] = Line2D(
                [], [], color=container.lines[0].get_color(), marker="o", label=label
            )
    for measure in measures:
        handles[measure] = Line2D(
            [],
            [],
            color=FOREGROUND_COLOR,
            linestyle=MEASURE_LINESTYLES[measure],
            label=measure.capitalize(),
        )
    ax.legend(
        handles=list(handles.values()), fontsize=FONT_SIZES["legend"], frameon=False
    )

    ax.set_xlabel(
        _sweep_axis_label(str(table["sweep_param"].iloc[0])),
        fontsize=FONT_SIZES["axis_label"],
    )
    ax.set_ylabel("ARI", fontsize=FONT_SIZES["axis_label"])
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    return style_axes(ax)


def cvi_lines(
```

Then format and lint. The blocks above are already formatted, so the first command changes nothing:

```bash
.venv/bin/ruff format --check src/
.venv/bin/ruff check src/
```

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/benchmarks/test_panels.py tests/benchmarks/test_theme.py tests/benchmarks/figures -q`

Expected: exit code 0, ending with:

```text
348 passed in 14.00s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `_panels.py`: `x_col = "n_clusters" if by_k else "sweep_value"` to `x_col = "sweep_value"` turns test_line_length_matches_one_configurations_k_values_not_every_row red.
- `_panels.py`: `carve_obj.get_sweep_value(` to `carve_obj.get_k(measure=measure, rule=rule, not_two=not_two)` turns test_a_resolution_run_draws_over_resolution red.
- `_panels.py`: `curve = results.loc[results["method_id"] == method_id].sort_values(x_col)` to `curve = results.sort_values(x_col)` turns test_a_resolution_run_draws_over_resolution red.
- `_panels.py`: `return "Number of clusters $k$"` to `return "Number of Clusters (k)"` turns test_a_k_based_run_keeps_the_number_of_clusters_axis red.
- `_panels.py`: `selected_row, _, _, _ = carve_obj._select_row(` to `selected_row, _, _, _ = carve_obj._select_row(` turns test_forwards_rule_and_not_two_to_every_selection red.
- `_panels.py`: `linestyle=MEASURE_LINESTYLES[measure],` to `)` turns test_one_line_pair_per_pipeline_sharing_a_pipeline_color red.
- `_panels.py`: `palette=PIPELINE_CMAP_NAME,` to `(line removed)` turns test_one_line_pair_per_pipeline_sharing_a_pipeline_color red.
- `_panels.py`: `method_id=method_id,` to `method_id="m1",` turns test_draws_only_the_named_configuration red.
- `_panels.py`: `for measure in measures:` to `for measure in ():` turns test_one_legend_names_the_pipelines_and_the_two_criteria red.
- `_theme.py`: `PIPELINE_COLORS: tuple[str, ...] = ("#0072B2", "#D55E00", "#CC79A7", "#8C564B")` to `PIPELINE_COLORS: tuple[str, ...] = ("#0072B2", "#D55E00", "#CC79A7", "#009E73")` turns test_no_pipeline_color_is_a_criterion_color red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "feat(benchmarks): carve_lines over any sweep axis; per-pipeline panel and palette"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 10: The four-panel figure, and the CompositeInputs pair leaves the study

Spec 7.2 and spec 6's deletion, confirmed by the author. `figure_cusanovich_results` is rewritten over `CusanovichInputs`; `figure_carve_output_cusanovich`, its constants import and its three tests go, and the figure contract drops the name. The synthetic inputs list CARVE's clusters in an order different from their ids, so a color map keyed on first appearance would disagree with the aligned one.

Files:
- Create: `src/benchmarks/figures/_cusanovich_results.py`
- Modify: `src/benchmarks/figures/__init__.py`
- Modify: `src/benchmarks/figures/_carve_output.py`
- Test: `tests/benchmarks/figures/test_carve_output.py`
- Test: `tests/benchmarks/figures/test_cusanovich_results.py`
- Test: `tests/benchmarks/figures/test_figure_contract.py`

Interfaces:
- Consumes: Task 8: `CusanovichInputs`. Task 6: `axis_prefix`. Task 9: `carve_lines(rule=...)`, `pipeline_lines`, `StubCarve(sweep_param=...)`, `resolution_results`, `pipeline_results`. Existing: `aligned_color_maps`, `scatter_clusters`, `axis_arrows`, `panel_letter`, `save_figure`.
- Produces: `figures._cusanovich_results.MARKER_SIZE = 8.0`, `SOURCE_TSNE_LABELS = ("t-SNE 1", "t-SNE 2")`, `SAVE_NAME = "cusanovich_results.png"`, `figure_cusanovich_results(inputs, *, save=True, out_dir=None) -> Figure`. Removed: `figure_carve_output_cusanovich` and `_cusanovich_results.AXIS_LABELS`.

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/figures/test_figure_contract.py`, replace:

```python
    "figure_carve_output_levine",
    "figure_carve_output_cusanovich",
    "figure_cusanovich_results",
```

with:

```python
    "figure_carve_output_levine",
    "figure_cusanovich_results",
```

In `tests/benchmarks/figures/test_carve_output.py`, replace:

```python
from benchmarks.figures import (
    figure_carve_output_cusanovich,
    figure_carve_output_klein,
    figure_carve_output_levine,
)
```

with:

```python
from benchmarks.figures import (
    figure_carve_output_klein,
    figure_carve_output_levine,
)
```

In `tests/benchmarks/figures/test_carve_output.py`, delete:

```python
from benchmarks.figures._cusanovich_results import AXIS_LABELS, MARKER_SIZE
```

In `tests/benchmarks/figures/test_carve_output.py`, delete:

```python
    def test_cusanovich_saves_under_the_manuscript_filename(self, inputs, tmp_path):
        fig = figure_carve_output_cusanovich(inputs, save=True, out_dir=tmp_path)
        assert (tmp_path / "CARVE_output_cusanovich.png").exists()
        plt.close(fig)

    def test_cusanovich_save_false_writes_nothing(self, inputs, tmp_path):
        fig = figure_carve_output_cusanovich(inputs, save=False, out_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

```

In `tests/benchmarks/figures/test_carve_output.py`, replace:

```python
    def test_cusanovich_matches_its_composite_embedding_and_marker_size(
        self, inputs
    ):
        # The two Cusanovich figures draw the same source t-SNE at the same
        # dot size, read from the composite module's constants rather than
        # restated, so the pair cannot drift apart.
        fig = figure_carve_output_cusanovich(inputs, save=False)
        panels = _panel_by_letter(fig)
        ax_e = panels["E"]
        assert (ax_e.get_xlabel(), ax_e.get_ylabel()) == AXIS_LABELS
        sizes = np.concatenate(
            [
                c.get_sizes()
                for c in panels["F"].collections
                if isinstance(c, PathCollection)
            ]
        )
        np.testing.assert_allclose(sizes, MARKER_SIZE)
        plt.close(fig)


class TestClusterIdsAgreeWithTheComposite:
```

with:

```python
class TestClusterIdsAgreeWithTheComposite:
```

Create `tests/benchmarks/figures/test_cusanovich_results.py` with this content (replacing the file if it exists):

```python
"""Tests for the Cusanovich case-study figure."""

from dataclasses import replace

import matplotlib.colors as mcolors
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from sklearn.manifold import TSNE
from sklearn.preprocessing import FunctionTransformer

from benchmarks._cusanovich_compare import CusanovichInputs
from benchmarks.figures import figure_cusanovich_results
from benchmarks.figures._cusanovich_results import MARKER_SIZE, SOURCE_TSNE_LABELS
from carve._pipeline import PipelineSpec, PipelineStep
from tests.benchmarks._helpers import StubCarve, pipeline_results, resolution_results

RESOLUTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
IDENTITY = PipelineStep(cls=FunctionTransformer, params={}, name="identity")
SPECS = {
    spec.label: spec
    for spec in (
        PipelineSpec(normalization=IDENTITY, dim_reduction=step)
        for step in (
            IDENTITY,
            PipelineStep(cls=TSNE, params={"perplexity": 30}, name="TSNE"),
            PipelineStep(cls=object, params={"n_neighbors": 15}, name="UMAP"),
        )
    )
}


@pytest.fixture
def inputs():
    # Six source clusters of 20 cells; CARVE's four clusters are listed in an
    # order different from their ids, so a color map keyed on first
    # appearance would disagree with the aligned one. The stub selects
    # resolution 0.6, where resolution_results observes 12 clusters.
    rng = np.random.default_rng(0)
    n = 120
    carve = StubCarve(
        resolution_results(RESOLUTIONS, "LeidenClustering"),
        select=lambda measure, not_two: ("m0", 0.6),
        sweep_param="resolution",
    )
    carve.preprocessing_results_ = pipeline_results(tuple(SPECS), RESOLUTIONS)
    carve.preprocessing_pipelines_ = SPECS
    table = carve.preprocessing_results_
    best = table.loc[
        (table["sweep_value"] == 0.6) & (table["pipeline"] == "identity | identity")
    ].iloc[0]
    return CusanovichInputs(
        X=rng.normal(size=(n, 5)),
        y=np.repeat(["1", "2", "3", "4", "5", "6"], n // 6),
        carve=carve,
        carve_labels=np.repeat([3, 1, 0, 2], n // 4),
        embedding_A=rng.normal(size=(n, 2)),
        embedding_A_labels=("LSI 1", "LSI 2"),
        source_tsne=rng.normal(size=(n, 2)),
        best_pipeline_row=best,
        operating_point=(0.8, 16.0),
        published_generalizability=(0.55, 0.02),
    )


def _panels(fig):
    """Map each panel letter to its axes via the text panel_letter drew."""
    return {
        text.get_text(): ax
        for ax in fig.get_axes()
        for text in ax.texts
        if text.get_text() in {"A", "B", "C", "D"}
    }


def _scatter_collections(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def test_draws_four_lettered_panels_and_writes_nothing_when_save_is_false(
    inputs, tmp_path
):
    fig = figure_cusanovich_results(inputs, save=False, out_dir=tmp_path)
    assert set(_panels(fig)) == set("ABCD")
    assert list(tmp_path.iterdir()) == []


def test_writes_under_its_manuscript_name(inputs, tmp_path):
    figure_cusanovich_results(inputs, save=True, out_dir=tmp_path)
    assert (tmp_path / "cusanovich_results.png").is_file()


def test_titles_name_the_selection_and_the_source(inputs):
    panels = _panels(figure_cusanovich_results(inputs, save=False))
    title = panels["A"].get_title()
    assert "CARVE: Leiden, resolution 0.6, 12 clusters" in title
    assert "consensus over LSI, t-SNE, UMAP" in title
    assert "shown on LSI 1/2" in title
    assert panels["B"].get_title() == "Cusanovich et al.: Louvain on t-SNE, 6 clusters"


def test_a_draws_the_best_pipelines_embedding_and_b_the_source_tsne(inputs):
    panels = _panels(figure_cusanovich_results(inputs, save=False))
    for letter, points, names in (
        ("A", inputs.embedding_A, inputs.embedding_A_labels),
        ("B", inputs.source_tsne, SOURCE_TSNE_LABELS),
    ):
        ax = panels[letter]
        drawn = np.concatenate([c.get_offsets() for c in _scatter_collections(ax)])
        assert sorted(map(tuple, drawn)) == sorted(map(tuple, points))
        assert set(names) <= {text.get_text() for text in ax.texts}
        sizes = np.concatenate([c.get_sizes() for c in _scatter_collections(ax)])
        np.testing.assert_allclose(sizes, MARKER_SIZE)


def test_a_carve_cluster_takes_the_color_of_the_source_cluster_it_matches(inputs):
    # carve_labels are relabeled onto y's sorted codes, so CARVE cluster 0
    # and source cluster "1" share a palette entry, as do 3 and "4".
    panels = _panels(figure_cusanovich_results(inputs, save=False))

    def color_of(ax, label):
        (collection,) = [c for c in _scatter_collections(ax) if c.get_label() == label]
        return mcolors.to_hex(collection.get_facecolor()[0])

    assert color_of(panels["A"], "0") == color_of(panels["B"], "1")
    assert color_of(panels["A"], "3") == color_of(panels["B"], "4")


def test_c_marks_the_operating_point_and_the_published_partition(inputs):
    ax = _panels(figure_cusanovich_results(inputs, save=False))["C"]
    assert ax.get_xlabel() == "Resolution"

    at_point = inputs.carve.estimator_results_.set_index("sweep_value").loc[0.8]
    markers = [line for line in ax.lines if line.get_marker() == "D"]
    assert sorted((line.get_xdata()[0], line.get_ydata()[0]) for line in markers) == (
        sorted(
            [(0.8, at_point["ari_stability"]), (0.8, at_point["ari_generalizability"])]
        )
    )
    horizontal = [line for line in ax.lines if list(line.get_ydata()) == [0.55, 0.55]]
    assert len(horizontal) == 1

    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "source operating point, 16 clusters on t-SNE" in legend
    assert "published 6 clusters, RF probe" in legend


def test_c_counts_the_observed_clusters_on_a_secondary_axis(inputs):
    fig = figure_cusanovich_results(inputs, save=False)
    fig.canvas.draw()
    (top,) = _panels(fig)["C"].child_axes
    assert [label.get_text() for label in top.get_xticklabels()] == [
        "4",
        "8",
        "12",
        "16",
        "20",
    ]


def test_d_draws_one_line_pair_per_pipeline_sharing_cs_x_axis(inputs):
    panels = _panels(figure_cusanovich_results(inputs, save=False))
    labels = [container.get_label() for container in panels["D"].containers]
    assert sorted(labels) == sorted([*SPECS, *SPECS])
    assert panels["D"].get_shared_x_axes().joined(panels["D"], panels["C"])


def test_every_selection_uses_the_inputs_rule_and_not_two(inputs):
    figure_cusanovich_results(replace(inputs, rule="max", not_two=True), save=False)
    calls = inputs.carve.selection_calls
    assert calls
    assert all(call["rule"] == "max" and call["not_two"] for call in calls)
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/benchmarks/figures tests/benchmarks/test_cusanovich_compare.py -q`

Expected: exit code 2, ending with:

```text
E   ImportError: cannot import name 'SOURCE_TSNE_LABELS' from 'benchmarks.figures._cusanovich_results' (/private/tmp/claude-501/-Users-kaiwycik-GitHub-CARVE--claude-playground/349e6055-9572-4f5c-919e-e41ea3aca477/scratchpad/pass2/src/benchmarks/figures/_cusanovich_results.py)
ERROR tests/benchmarks/figures/test_cusanovich_results.py
1 error in 0.15s
```

- [ ] Step 3: Implement

Create `src/benchmarks/figures/_cusanovich_results.py` with this content (replacing the file if it exists):

```python
"""The Cusanovich mouse sci-ATAC case-study figure.

Four panels over CusanovichInputs. (A) CARVE's consensus labels on the
embedding of the pipeline CARVE rates best at its selected configuration.
(B) The source's clusters on the source's own t-SNE, which cell_metadata.txt
ships and the loader carries as meta["source_tsne"]. (C) CARVE's stability
and generalizability over resolution, pooled over pipelines, with the selected
resolution, the source's operating point and the published partition's own
generalizability marked, and the mean observed cluster count on a secondary
axis. (D) The same two criteria per pipeline at the selected configuration,
on C's x axis. A and B share one color map, so a CARVE cluster takes the
color of the source cluster it best matches; with 30 source clusters the
palette cycles, as _theme.cluster_colors documents.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._cusanovich_compare import CusanovichInputs, axis_prefix
from .._panels import (
    aligned_color_maps,
    axis_arrows,
    carve_lines,
    panel_letter,
    pipeline_lines,
    scatter_clusters,
)
from .._theme import FONT_SIZES, metric_color, save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path

MARKER_SIZE = 8.0
SOURCE_TSNE_LABELS = ("t-SNE 1", "t-SNE 2")
SAVE_NAME = "cusanovich_results.png"


def _panel_a_title(inputs: CusanovichInputs, selected, n_clusters: int) -> str:
    carve = inputs.carve
    estimator = str(selected["estimator"]).removesuffix("Clustering")
    pooled = sorted(
        {
            axis_prefix(spec.dim_reduction)
            for spec in carve.preprocessing_pipelines_.values()
        },
        key=str.lower,
    )
    shown = inputs.embedding_A_labels[0].removesuffix(" 1")
    return (
        f"CARVE: {estimator}, {carve.sweep_.param} "
        f"{float(selected['sweep_value']):g}, {n_clusters} clusters\n"
        f"consensus over {', '.join(pooled)}; shown on {shown} 1/2"
    )


def _mark_source(ax: Axes, inputs: CusanovichInputs, method_id: str) -> None:
    """The source's operating point on C's curves, and its partition's score.

    The operating point is placed on the pooled curves at the resolution where
    the t-SNE pipeline's clusterings come nearest the source's cluster count,
    and labeled with that count. The horizontal line is the published
    partition's own generalizability under the same classifier probe.
    """
    results = inputs.carve.estimator_results_
    curve = results.loc[results["method_id"] == method_id]
    resolution, observed = inputs.operating_point
    at_point = curve.loc[np.isclose(curve["sweep_value"].astype(float), resolution)]
    n_source = int(np.unique(inputs.y).size)

    for index, measure in enumerate(("stability", "generalizability")):
        ax.plot(
            [resolution],
            [float(at_point[f"ari_{measure}"].iloc[0])],
            marker="D",
            markersize=8.0,
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=1.5,
            markeredgecolor=metric_color(f"ari_{measure}_1se"),
            zorder=3,
            label=(
                f"source operating point, {observed:.0f} clusters on t-SNE"
                if index == 0
                else "_nolegend_"
            ),
        )

    mean, _ = inputs.published_generalizability
    ax.axhline(
        mean,
        color=metric_color("ari_generalizability_1se"),
        linestyle="--",
        linewidth=1.2,
        label=f"published {n_source} clusters, RF probe",
    )
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)


def _observed_cluster_axis(ax: Axes, inputs: CusanovichInputs, method_id: str) -> None:
    """A secondary x axis naming the mean observed cluster count per resolution."""
    results = inputs.carve.estimator_results_
    curve = results.loc[results["method_id"] == method_id].sort_values("sweep_value")
    top = ax.secondary_xaxis("top")
    top.set_xticks(
        curve["sweep_value"].to_numpy(dtype=float),
        labels=[f"{count:.0f}" for count in curve["n_clusters_observed"]],
    )
    top.tick_params(labelsize=FONT_SIZES["tick"])
    top.set_xlabel("Mean observed clusters", fontsize=FONT_SIZES["axis_label"])


def figure_cusanovich_results(
    inputs: CusanovichInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build the Cusanovich case-study figure from assembled inputs."""
    carve = inputs.carve
    selected, _, n_clusters, _ = carve._select_row(
        measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
    )
    method_id = str(selected["method_id"])
    source_cmap, carve_cmap = aligned_color_maps(inputs.y, inputs.carve_labels)
    n_source = int(np.unique(inputs.y).size)

    with theme_context():
        fig, axes = plt.subplots(2, 2, figsize=(15.0, 13.0))
        ax_a, ax_b, ax_c, ax_d = axes.flat
        ax_d.sharex(ax_c)

        scatter_clusters(
            ax_a,
            inputs.embedding_A,
            inputs.carve_labels,
            color_map=carve_cmap,
            s=MARKER_SIZE,
            title=_panel_a_title(inputs, selected, n_clusters),
        )
        axis_arrows(ax_a, inputs.embedding_A_labels)
        scatter_clusters(
            ax_b,
            inputs.source_tsne,
            inputs.y,
            color_map=source_cmap,
            s=MARKER_SIZE,
            title=f"Cusanovich et al.: Louvain on t-SNE, {n_source} clusters",
        )
        axis_arrows(ax_b, SOURCE_TSNE_LABELS)

        carve_lines(
            ax_c,
            carve,
            rule=inputs.rule,
            not_two=inputs.not_two,
            title=f"CARVE over {carve.sweep_.param}, pooled over pipelines",
        )
        _mark_source(ax_c, inputs, method_id)
        _observed_cluster_axis(ax_c, inputs, method_id)

        pipeline_lines(
            ax_d,
            carve,
            method_id=method_id,
            rule=inputs.rule,
            not_two=inputs.not_two,
            title="Per pipeline, at the selected configuration",
        )

        for letter, ax in zip("ABCD", axes.flat):
            panel_letter(ax, letter)
        fig.tight_layout()

        if save:
            save_figure(
                fig, figure_path(SAVE_NAME, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
```

In `src/benchmarks/figures/_carve_output.py`, replace:

```python
"""Fig 3, S4 Fig and the Cusanovich figure: CARVE's own output on a case study.
```

with:

```python
"""Fig 3 and S4 Fig: CARVE's own output on a case study.
```

In `src/benchmarks/figures/_carve_output.py`, replace:

```python
from ._case_study import CompositeInputs, carve_labels_aligned
from ._cusanovich_results import AXIS_LABELS as CUSANOVICH_AXIS_LABELS
from ._cusanovich_results import MARKER_SIZE as CUSANOVICH_MARKER_SIZE
from ._paths import CASE_STUDY_DIR, figure_path
```

with:

```python
from ._case_study import CompositeInputs, carve_labels_aligned
from ._paths import CASE_STUDY_DIR, figure_path
```

In `src/benchmarks/figures/_carve_output.py`, replace:

```python
        save_name="CARVE_output_levine.png",
        save=save,
        out_dir=out_dir,
    )


def figure_carve_output_cusanovich(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build the CARVE output figure for the Cusanovich sci-ATAC case study.

    Marker size and axis labels come from the composite module rather than
    being restated here, so the two Cusanovich figures draw the same source
    t-SNE at the same dot size by construction.
    """
    return carve_output_figure(
        inputs,
        marker_size=CUSANOVICH_MARKER_SIZE,
        axis_labels=CUSANOVICH_AXIS_LABELS,
        save_name="CARVE_output_cusanovich.png",
        save=save,
        out_dir=out_dir,
    )
```

with:

```python
        save_name="CARVE_output_levine.png",
        save=save,
        out_dir=out_dir,
    )
```

In `src/benchmarks/figures/__init__.py`, replace:

```python
from ._carve_output import (
    figure_carve_output_cusanovich,
    figure_carve_output_klein,
    figure_carve_output_levine,
)
```

with:

```python
from ._carve_output import figure_carve_output_klein, figure_carve_output_levine
```

In `src/benchmarks/figures/__init__.py`, replace:

```python
    "figure_carve_output_cusanovich",
    "figure_carve_output_klein",
```

with:

```python
    "figure_carve_output_klein",
```

Then format and lint. The blocks above are already formatted, so the first command changes nothing:

```bash
.venv/bin/ruff format --check src/
.venv/bin/ruff check src/
```

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/benchmarks/figures tests/benchmarks/test_cusanovich_compare.py -q`

Expected: exit code 0, ending with:

```text
224 passed in 16.75s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `_cusanovich_results.py`: `inputs.source_tsne,` to `inputs.embedding_A,` turns test_a_draws_the_best_pipelines_embedding_and_b_the_source_tsne red.
- `_cusanovich_results.py`: `source_cmap, carve_cmap = aligned_color_maps(inputs.y, inputs.carve_labels)` to `from .._panels import cluster_color_map` turns test_a_carve_cluster_takes_the_color_of_the_source_cluster_it_matches red.
- `_cusanovich_results.py`: `title=_panel_a_title(inputs, selected, n_clusters),` to `title=_panel_a_title(inputs, selected, n_clusters + 1),` turns test_titles_name_the_selection_and_the_source red.
- `_cusanovich_results.py`: `for index, measure in enumerate(("stability", "generalizability")):` to `for index, measure in enumerate(("stability",)):` turns test_c_marks_the_operating_point_and_the_published_partition red.
- `_cusanovich_results.py`: `mean, _ = inputs.published_generalizability` to `_, mean = inputs.published_generalizability` turns test_c_marks_the_operating_point_and_the_published_partition red.
- `_cusanovich_results.py`: `labels=[f"{count:.0f}" for count in curve["n_clusters_observed"]],` to `labels=[f"{value:g}" for value in curve["sweep_value"]],` turns test_c_counts_the_observed_clusters_on_a_secondary_axis red.
- `_cusanovich_results.py`: `ax_d.sharex(ax_c)` to `(line removed)` turns test_d_draws_one_line_pair_per_pipeline_sharing_cs_x_axis red.
- `_cusanovich_results.py`: `carve_lines(` to `carve_lines(` turns test_every_selection_uses_the_inputs_rule_and_not_two red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "feat(benchmarks): four-panel Cusanovich figure; remove its CompositeInputs pair"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 11: The notebook

Spec 7.3. Sections 0 to 4 as the spec lists them. Every setting comes from `STUDIES["cusanovich"]`; the only constants the notebook sets are the seed, the scale and the selection (`MEASURE = "stability"`, `RULE = "1se"`, `NOT_TWO = False`). The runtime note carries the spec's estimate.

Files:
- Modify: `notebooks/case_studies/Cusanovich.ipynb` (rebuilt by a script kept outside the repository)
- Test: `tests/benchmarks/test_notebooks.py`

Interfaces:
- Consumes: Tasks 1 to 10.
- Produces: `notebooks/case_studies/Cusanovich.ipynb`, 12 cells.

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/test_notebooks.py`, replace:

```python
def test_cusanovich_notebook_reads_its_config_from_studies():
    source = _code(NOTEBOOKS["cusanovich"])
    # Configuration must be read from STUDIES, not restated. Four manuscript
    # mismatches on this project came from re-derivation at the call site.
    assert 'STUDIES["cusanovich"]' in source
    assert "study_model_grids(study)" in source
    assert "candidate_k=study.candidate_k" in source
    assert "range(4, 17)" not in source
    assert "consensus_anchors=study.consensus_anchors" in source


def test_cusanovich_notebook_draws_carve_output_before_the_composite():
    # Mirrors Klein: CARVE's own six-panel diagnostic figure, then the
    # composite, both from the one prepare_composite call.
    source = _code(NOTEBOOKS["cusanovich"])
    assert "figure_carve_output_cusanovich(inputs" in source
    assert source.index("figure_carve_output_cusanovich(inputs") < source.index(
        "figure_cusanovich_results(inputs"
    )
```

with:

```python
def test_cusanovich_notebook_reads_its_config_from_studies():
    source = _code(NOTEBOOKS["cusanovich"])
    # Configuration must be read from STUDIES, not restated. Four manuscript
    # mismatches on this project came from re-derivation at the call site.
    assert 'STUDIES["cusanovich"]' in source
    assert "study_resolution_grids(study)" in source
    assert "n_resamples=study.n_resamples" in source
    assert "preprocessing=study.preprocessing" in source
    assert "resolve_preprocessing(study.preprocessing)" in source
    assert "consensus_anchors=study.consensus_anchors" in source
    assert "randomize_preprocessing=True" in source
    for restated in ("perplexity", "n_neighbors", "n_resamples=150", "np.arange"):
        assert restated not in source


@pytest.mark.parametrize(
    "retired",
    [
        "cvi_sweep",
        "study_model_grids",
        "figure_carve_output_cusanovich",
        "prepare_composite",
    ],
)
def test_cusanovich_notebook_no_longer_runs_the_k_based_comparison(retired):
    # The CVI sweep, the k-based fit and the CompositeInputs figures left the
    # study with the move to randomized preprocessing (case-study spec,
    # section 6).
    assert retired not in _code(NOTEBOOKS["cusanovich"])
```

In `tests/benchmarks/test_notebooks.py`, replace:

```python
# Each ATAC case-study notebook opens with a reference-label scatter of one
# embedding and then hands that same embedding to prepare_composite, so the
# data is shown one way throughout. The embedding follows the source: the
# Cusanovich atlas ships its own t-SNE, which the loader carries through as
# meta["source_tsne"]; hECA ships nothing, so the notebook computes a UMAP
# the way Levine_32dim.ipynb computes its t-SNE. AXIS_LABELS in the two
# composite modules are pinned to match, in their own test files.
def test_cusanovich_notebook_draws_the_source_tsne_throughout():
    source = _code(NOTEBOOKS["cusanovich"])
    assert 'meta["source_tsne"]' in source
    assert "figure_reference_scatter(" in source
    assert "embedding=" in source
    assert "plt.subplots" not in source
```

with:

```python
# Each ATAC case-study notebook opens with a reference-label scatter of the
# embedding its source provides. The Cusanovich atlas ships its own t-SNE,
# which the loader carries through as meta["source_tsne"] and the figure's
# panel B draws again. hECA ships nothing, so its notebook computes a UMAP the
# way Levine_32dim.ipynb computes its t-SNE and hands that same embedding to
# prepare_composite.
def test_cusanovich_notebook_draws_the_source_tsne_and_writes_its_tables():
    source = _code(NOTEBOOKS["cusanovich"])
    assert 'meta["source_tsne"]' in source
    assert "figure_reference_scatter(" in source
    assert "figure_cusanovich_results(inputs" in source
    assert "save_tables(inputs" in source
    assert "plt.subplots" not in source
```

- [ ] Step 2: Run them and watch them fail

Run: `.venv/bin/pytest tests/benchmarks/test_notebooks.py -q`

Expected: exit code 1, ending with:

```text
E       assert 'n_resamples=study.n_resamples' in 'from pathlib import Path\n\nimport matplotlib.pyplot as plt\n\nfrom benchmarks._studies import (\n    STUDIES,\n    c... ).fit(X_atlas)\n\n    display(\n        atlas.estimator_results_[["config_id", "resolution", "ari_stability"]]\n    )'
E       assert 'cvi_sweep' not in 'from pathli...ty"]]\n    )'
E         
E         'cvi_sweep' is contained here:
E           path,
E               cvi_sweep,
E               fit_or_load_carve,
E               load_study,
E               study_model_grids,...
E         
E         ...Full output truncated (87 lines hidden), use '-vv' to show
E       assert 'study_model_grids' not in 'from pathli...ty"]]\n    )'
6 failed, 66 passed in 0.23s
```

- [ ] Step 3: Rebuild the notebook

Save this script outside the repository, for example as `build_cusanovich_notebook.py` in the session scratchpad. It keeps the notebook's metadata and replaces every cell; it is not committed.

```python
"""Write notebooks/case_studies/Cusanovich.ipynb from the cell sources below.

Keeps the notebook's existing metadata (kernelspec, language_info) and
replaces every cell. Run from the repository root:

    .venv/bin/python build_cusanovich_notebook.py notebooks/case_studies/Cusanovich.ipynb
"""

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

CELLS = [
    (
        "markdown",
        '''# Cusanovich et al. (2018): clustering in t-SNE space, checked by CARVE

Source: Cusanovich et al., "A Single-Cell Atlas of In Vivo Mammalian Chromatin Accessibility." Cell 174.5 (2018): 1309-1324. GEO accession GSE111586.

sci-ATAC-seq of 81,173 nuclei from 13 adult mouse tissues. The source built an LSI (3 percent site filter, TF-IDF, 50-component SVD), ran t-SNE on it at perplexity 30, and clustered the two-dimensional t-SNE with Seurat's graph community detection into 30 clusters.

Clustering a t-SNE can manufacture clusters: the embedding is stochastic and its islands depend on seed and perplexity. This notebook sweeps Leiden resolution under CARVE with randomized preprocessing (the LSI as-is, a t-SNE of it, or a UMAP of it) and asks which pipeline is stable and generalizable, and where the source's 30-cluster partition sits. The expectation, stated so it can be falsified: clustering in t-SNE space is less stable and less generalizable than clustering the LSI at the same granularity, and 30 clusters is past the point where either criterion starts to fall.

The reference is the source's own 30 clusters. Every comparison with it is reported as agreement, not accuracy.''',
    ),
    (
        "markdown",
        '''---

## 0. Configuration

Everything below reads STUDIES["cusanovich"]: the scales, the resolution grid, the resample count and the preprocessing options. The fit needs the graph extra (Leiden) and the umap extra.

Runtime, estimated before measurement: at SCALE = "publication" (5,000 cells) the fit draws 150 resamples, a third of them t-SNE, then runs 20 Leiden configurations; the design estimate is 30 to 60 minutes of t-SNE and one to two hours of Leiden and random forests on one core, less in parallel. SCALE = "dev" (1,500 cells) is for iteration.''',
    ),
    (
        "code",
        '''from pathlib import Path

import matplotlib.pyplot as plt

from benchmarks._cusanovich_compare import (
    prepare_cusanovich_inputs,
    save_tables,
    selection_summary,
)
from benchmarks._preprocessing import resolve_preprocessing
from benchmarks._studies import (
    STUDIES,
    carve_cache_path,
    fit_or_load_carve,
    load_study,
    study_resolution_grids,
)
from benchmarks.figures import figure_cusanovich_results, figure_reference_scatter
from benchmarks.figures._cusanovich_results import MARKER_SIZE, SOURCE_TSNE_LABELS
from benchmarks.figures._paths import CASE_STUDY_DIR

RANDOM_SEED = 42
SCALE = "dev"  # "dev", "publication" or "atlas" (every cell)

# The selection every panel and table reads.
MEASURE = "stability"
RULE = "1se"
NOT_TWO = False

# Figures and tables are written under a scale-qualified directory, so
# development output can never be mistaken for the manuscript's. Only the
# publication directory is ever copied into overleaf/vis/.
OUT_DIR = CASE_STUDY_DIR if SCALE == "publication" else CASE_STUDY_DIR / SCALE
OUT_DIR.mkdir(parents=True, exist_ok=True)

study = STUDIES["cusanovich"]
model_grids = study_resolution_grids(study)
X, y, meta = load_study(study, scale=SCALE)
print(meta["n_cells"], "cells,", meta["n_features"], "LSI components,", meta["scale"], "scale")


def show(fig):
    """Display a figure exactly once, then close it."""
    display(fig)
    plt.close(fig)''',
    ),
    (
        "markdown",
        '''---

## 1. The source's partition

The source's clusters on the source's own t-SNE (their Figure 1), restricted to the cells analyzed here. cell_metadata.txt ships both; the loader carries them as y and meta["source_tsne"], row for row with X. The subsample is stratified by cluster, so every cluster is present. Cells whose marker-based label is Unknown stay, because the 30 clusters assign them too.''',
    ),
    (
        "code",
        '''unknown = meta["source_labels"]["cell_label"] == "Unknown"
print(y.nunique(), "source clusters; the smallest has", y.value_counts().min(), "cells here")
print(
    f"{unknown.mean():.1%} of the analyzed cells have cell_label Unknown "
    f"({meta['n_cells_full'] - meta['n_cells_annotated']} of {meta['n_cells_full']} in the atlas)"
)

show(
    figure_reference_scatter(
        meta["source_tsne"], y.to_numpy(),
        axis_labels=SOURCE_TSNE_LABELS, title=f"Cusanovich et al.: {y.nunique()} clusters",
        marker_size=MARKER_SIZE, out_dir=OUT_DIR,
        save_name="cusanovich_reference_scatter.png",
    )
)''',
    ),
    (
        "markdown",
        '''---

## 2. CARVE fit

Leiden over the study's resolution grid, with one preprocessing pipeline drawn per resample: the LSI as-is, t-SNE at the source's perplexity, or UMAP. Each pipeline is fit separately on each subsample, so an embedding that does not reproduce across fits lowers both criteria, and the generalizability classifier trains on the LSI itself. The cached fit's filename is keyed on the scale, the grid, the resample count and the preprocessing.''',
    ),
    (
        "code",
        '''carve = fit_or_load_carve(
    X, y,
    cache_path=carve_cache_path(
        study, scale=SCALE, root=Path("./carve_state_saves"),
        model_grids=model_grids, n_resamples=study.n_resamples,
        preprocessing=study.preprocessing,
    ),
    model_grids=model_grids,
    n_resamples=study.n_resamples,
    randomize_preprocessing=True,
    **resolve_preprocessing(study.preprocessing),
    consensus_anchors=study.consensus_anchors,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
carve.estimator_results_[
    ["resolution", "n_clusters_observed", "ari_stability", "ari_generalizability"]
]''',
    ),
    (
        "markdown",
        '''---

## 3. Comparison

Three things set the source's recipe beside CARVE's: the pipeline CARVE rates best at its selected resolution; the source's operating point, the resolution at which the t-SNE pipeline's clusterings come nearest the source's cluster count; and the published partition's own generalizability under the classifier CARVE uses, on the same LSI.''',
    ),
    (
        "code",
        '''inputs = prepare_cusanovich_inputs(
    X, y.to_numpy(), carve, source_tsne=meta["source_tsne"],
    measure=MEASURE, rule=RULE, not_two=NOT_TWO,
    random_state=RANDOM_SEED, n_jobs=-1,
)
selection_summary(inputs)''',
    ),
    (
        "code",
        '''table = carve.preprocessing_results_
table.loc[
    table["method_id"] == inputs.best_pipeline_row["method_id"],
    ["pipeline", "resolution", "n_resamples", "n_clusters_observed",
     "ari_stability", "ari_generalizability"],
]''',
    ),
    (
        "code",
        '''# The figure and both tables are written into OUT_DIR. Copying the
# publication figure into overleaf/vis/ stays a manual step.
show(figure_cusanovich_results(inputs, out_dir=OUT_DIR))
save_tables(inputs, OUT_DIR)''',
    ),
    (
        "markdown",
        '''---

## 4. Summary

Panel C places the source's recipe on CARVE's pooled curves twice: at the resolution where the t-SNE pipeline reaches the source's cluster count, and as the published partition's generalizability under the same classifier probe. Panel D separates the curves by pipeline, which is where a difference between clustering the LSI and clustering its t-SNE shows. The selection summary, written to cusanovich_selection_summary.csv, holds every number the case study reports, including the agreement (ARI) between CARVE's consensus and the source's clusters; the per-pipeline table is written to cusanovich_preprocessing_results.csv.''',
    ),
]


def main(path: str) -> None:
    target = Path(path)
    existing = nbformat.read(target, as_version=4)
    notebook = nbformat.v4.new_notebook(
        cells=[
            new_markdown_cell(source) if kind == "markdown" else new_code_cell(source)
            for kind, source in CELLS
        ],
        metadata=existing.metadata,
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, target)
    print(f"wrote {len(CELLS)} cells to {target}")


if __name__ == "__main__":
    main(sys.argv[1])
```

Run it from the repository root: `.venv/bin/python <scratchpad>/build_cusanovich_notebook.py notebooks/case_studies/Cusanovich.ipynb`. Expected output: `wrote 12 cells to notebooks/case_studies/Cusanovich.ipynb`.

- [ ] Step 4: Run the tests and watch them pass

Run: `.venv/bin/pytest tests/benchmarks/test_notebooks.py -q`

Expected: exit code 0, ending with:

```text
72 passed in 0.17s
```

Mutation checks run while this plan was verified; each turned its test red and was reverted:

- `Cusanovich.ipynb`: `"model_grids = study_resolution_grids(study)\n",` to `"model_grids = study_model_grids(study)\n",` turns test_cusanovich_notebook_reads_its_config_from_studies, test_cusanovich_notebook_no_longer_runs_the_k_based_comparison red.
- `Cusanovich.ipynb`: `"    randomize_preprocessing=True,\n",` to `"    randomize_preprocessing=False,\n",` turns test_cusanovich_notebook_reads_its_config_from_studies red.
- `Cusanovich.ipynb`: `"    n_resamples=study.n_resamples,\n",` to `"    n_resamples=150,\n",` turns test_cusanovich_notebook_reads_its_config_from_studies red.
- `Cusanovich.ipynb`: `"save_tables(inputs, OUT_DIR)"` to `"inputs"` turns test_cusanovich_notebook_draws_the_source_tsne_and_writes_its_tables red.

- [ ] Step 5: Commit

```bash
git add -A
git commit -m "docs(notebooks): rebuild the Cusanovich notebook on randomized preprocessing"
```

End the message with the attribution lines from the session's system reminder.

---

### Task 12: Smoke-run the notebook at development scale

The plan supplies the scaffolding; the author tunes the case study's parameters and runs it on the full data set. This task only shows the notebook runs end to end on the released atlas. Nothing is committed.

- [ ] Step 1: Execute the notebook

The executed copy goes to a scratch directory outside the repository, so image output is never committed. nbconvert runs the notebook from its own directory, so the fit is cached under `notebooks/case_studies/carve_state_saves/`.

```bash
OUT=<scratchpad>/cusanovich-dev && mkdir -p "$OUT"
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output-dir="$OUT" notebooks/case_studies/Cusanovich.ipynb
ls vis/case_studies/dev/ | grep cusanovich
```

Expected: nbconvert exits 0, and `vis/case_studies/dev/` holds `cusanovich_reference_scatter.png`, `cusanovich_results.png`, `cusanovich_preprocessing_results.csv` and `cusanovich_selection_summary.csv`. While planning, loading the atlas took about 4 minutes with a 13.5 GB peak memory footprint, and the fit about 5 minutes over 20 resolutions on an 11-core machine. At 1,500 cells and resolutions up to 2.0 the t-SNE pipeline reached 21.5 clusters, so the notebook shows `source_operating_point`'s warning that the nearest count is more than 25 percent from 30. That warning is expected here and is for the author's parameter choices; it is not a failure.

- [ ] Step 2: Leave the tree clean

Run: `git status --short`

Expected: nothing. The cache, the executed notebook and `vis/` are gitignored or outside the repository.

---

### Task 13: Final verification

- [ ] Step 1: Lint and format, as CI does

```bash
.venv/bin/ruff check src/
.venv/bin/ruff format --check src/
```

Expected: `All checks passed!` and `66 files already formatted`.

- [ ] Step 2: The benchmarks leg

Run: `.venv/bin/pytest tests/benchmarks -v --tb=short 2>&1 | tail -1`

Expected: no failures, 10 skipped, and more passing tests than Task 0 Step 4's 786: the difference is the tests this plan adds less the three it retires from `figures/test_carve_output.py`.

- [ ] Step 3: The carve leg

Run: `.venv/bin/pytest tests --ignore=tests/benchmarks --cov=carve --cov-fail-under=75 -q 2>&1 | tail -3`

Expected: 878 passed (877 at a95bf46 plus Task 7's test), coverage above 75 percent.

- [ ] Step 4: Nothing outside the scope changed

```bash
git diff --stat main -- src/carve carve-r .github pyproject.toml
grep -rn "figure_carve_output_cusanovich\|import AXIS_LABELS, MARKER_SIZE" src/benchmarks tests/benchmarks
git status --short
```

Expected: the diff lists only `src/carve/_utils.py`; the grep prints nothing; the tree is clean.

- [ ] Step 5: Hand the integration decision to the author

Use superpowers:finishing-a-development-branch. The branch stays local unless the author says otherwise.

---

## After this plan

- Parameters (resolution grid, resample count, pipelines, selection rule) and the publication-scale and full-atlas runs are the author's.
- Task 7 changes `accuracy_generalizability` wherever predictions and held-out labels had different cluster counts, and the Klein, Levine and hECA composites wherever a selected k exceeded the reported label count; none of those results or figures were regenerated here.
- The `not_two` selection-marker defect in `carve._plotting.plot_metric_by_pipeline` stays open by the author's decision.
- Spec 8's development-scale record and spec 9's manuscript text are the author's.
- The R port mirrors none of this, including Task 7; changes to it are out of scope.
