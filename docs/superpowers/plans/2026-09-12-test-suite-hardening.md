# Test Suite Hardening Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: implement every finding of the 2026-09-11 Python test-suite audit so that the carve and benchmarks suites run under `filterwarnings = error`, share one test infrastructure, assert values rather than shapes, cover `carve.pl` and the other untested public behavior, and spend roughly half the wall-clock they do today.

Architecture: warnings hygiene lands first (a two-site library fix in `_consensus.py` unblocks `filterwarnings = error`, and every later task is verified under it). The test tree then becomes a package (`tests/__init__.py`) with shared helpers in `tests/_helpers.py` and `tests/benchmarks/_helpers.py`, so spies and the stub CARVE exist once. Mechanical defect fixes follow, then the missing tests (a new `tests/test_pl.py`, `plot_diagnostic_scatter`, the sparse spectral path, split-mode error contracts, re-fit semantics), then fixture caching in the three slow files. Three library changes are in scope, all Python-only: the nanmean fix, fit-time validation of three constructor parameters, and a defect in the four `source="accuracy"` plot branches that the error-contract tests expose.

Tech Stack: Python 3.12 locally (3.13 in CI), pytest 9, NumPy 2.5, pandas 3, matplotlib 3.10, scikit-learn 1.7, joblib. New dev dependencies: `pytest-xdist` (local convenience only) and `pyyaml` (already imported by a test, never declared).

Spec: `docs/superpowers/specs/2026-09-11-test-suite-audit.md` (copied verbatim from `_claude_playground/test_suite_audit.md`; section numbers below refer to it).

## Global Constraints

- The R port (`carve-r/`) is out of scope. Library changes here are Python-only. The nanmean change has no R counterpart to make: R's `rowMeans(term, na.rm = TRUE)` already returns NaN silently for an all-NA row, so the two ports agree on values today and continue to.
- Dependency direction: `_types`, `_utils`, `_anndata` and `cluster` are leaves. Nothing under `_*` imports `api`.
- Tests mirror modules one to one. `_consensus.py` maps to `tests/test_consensus.py`, `api.py` to `tests/test_api.py`, `pl/_plots.py` to the new `tests/test_pl.py`, `_plotting.py` to `tests/test_plotting.py`, `cluster.py` to `tests/test_cluster.py`.
- `config_id` is a join key, never a positional index. Seeds are derived arithmetically, never shared. `SweepSpec` is frozen. No `logging`; diagnostics are `warnings.warn(..., stacklevel=2)` and `verbose`-gated `print`.
- Lint and format `src/` only, with the pinned ruff (`.venv/bin/ruff`, `==0.16.4`): `ruff check src/` and `ruff format src/`. Task 6 runs ruff on `tests/` once, with an explicit `--select`, because the audit calls for it; that is a one-off, not a new practice.
- The venv is uv-managed and has no pip. Install with `VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"`. Run tests with `.venv/bin/python -m pytest`.
- Always `OMP_NUM_THREADS=1` when running tests locally (the CI notebook job pins it; the default RF classifier sets `n_jobs=-1`).
- Run tests in the foreground with a long timeout. The carve suite is about 3 minutes, the benchmarks suite about 7 minutes before Task 17 and about 4 after. A slow run is not a hang.
- American spelling. No bold or italics in anything written into the repo: docstrings, comments, TOML comments, Markdown.
- Every `pytest.warns(...)` written or touched by this plan carries `match=`. Every `pytest.raises(...)` written by this plan carries `match=`.
- Fixtures must be built so a wrong answer differs from a right one. Where a task says "verify by mutation", perturb the production code as instructed, confirm the test goes red, then revert the perturbation before committing.
- Commit after every task. Commit messages follow the repository's `type(scope): summary` style seen in `git log`.

All commands are run from `/Users/kaiwycik/GitHub/CARVE/code` (or the worktree created for this plan; the executor creates it with `superpowers:using-git-worktrees` before Task 1). Runtime numbers in this plan were measured on the audit machine (M-series, Python 3.12.11).

---

### Task 1: Silence the two nanmean sites in `_consensus.py`

The carve suite emits 738 `RuntimeWarning: Mean of empty slice` from four `np.nanmean(..., axis=1)` calls: two in `stability_from_consensus` (an all-NaN consensus row, i.e. a sample never co-sampled with anyone) and two in `stability_from_runs_anchored` (a sample never co-sampled with any anchor). NaN is the correct answer for those rows; the warning is not. Spec: 3.1.

Files:
- Modify: `src/carve/_consensus.py` (add `_row_nanmean`, replace four calls)
- Test: `tests/test_consensus.py`

Interfaces:
- Produces: `_row_nanmean(a: np.ndarray) -> np.ndarray`, private to `_consensus.py`. Row means over `axis=1`, NaN for a row with no finite entry, bit-identical to `np.nanmean(a, axis=1)` elsewhere, never warns.

- [ ] Step 1: Write the failing tests

Add `import warnings` to the imports of `tests/test_consensus.py`, add `_row_nanmean` to the `from carve._consensus import (...)` block, and add these tests. Put `TestRowNanmean` directly after `TestReorderConsensusMatrix`; put the other two methods at the end of the classes named.

```python
class TestRowNanmean:
    def test_matches_nanmean_wherever_nanmean_is_defined(self):
        rng = np.random.RandomState(0)
        a = rng.rand(50, 40)
        a[rng.rand(50, 40) < 0.3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmean(a, axis=1)
        np.testing.assert_array_equal(_row_nanmean(a), expected)

    def test_all_nan_row_is_nan_and_does_not_warn(self):
        a = np.array([[0.2, np.nan, 0.6], [np.nan, np.nan, np.nan]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = _row_nanmean(a)
        assert out[0] == pytest.approx(0.4)
        assert np.isnan(out[1])
```

In `TestStabilityFromConsensus`:

```python
    def test_isolated_sample_scores_nan_without_warning(self):
        # Sample 2 was never co-sampled with anyone: its off-diagonal row is
        # all NaN, so it has no partners to average over. NaN is the right
        # score; the old nanmean call warned "Mean of empty slice" here.
        M = np.array(
            [
                [1.0, 0.8, np.nan],
                [0.8, 1.0, np.nan],
                [np.nan, np.nan, 1.0],
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gini, ce = stability_from_consensus(M)
        assert np.isfinite(gini[:2]).all()
        assert np.isfinite(ce[:2]).all()
        assert np.isnan(gini[2])
        assert np.isnan(ce[2])
```

In `TestStabilityFromRunsAnchored`:

```python
    def test_sample_never_cosampled_with_an_anchor_is_nan_without_warning(self):
        # Every run draws from the first 29 samples, so sample 29 shares no
        # run with any anchor and its n-by-m row is entirely NaN.
        n = 30
        rng = np.random.default_rng(0)
        runs = []
        for _ in range(6):
            idx = np.sort(rng.choice(n - 1, size=18, replace=False))
            runs.append((idx, rng.integers(0, 3, idx.size)))
        anchors = np.arange(10)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gini, ce = stability_from_runs_anchored(n, runs, anchors)
        assert np.isnan(gini[n - 1])
        assert np.isnan(ce[n - 1])
        assert np.isfinite(gini[:10]).all()
```

- [ ] Step 2: Run the tests to verify they fail

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_consensus.py -k "RowNanmean or isolated_sample or never_cosampled_with_an_anchor" -v`
Expected: `ImportError: cannot import name '_row_nanmean'` at collection (all fail).

- [ ] Step 3: Implement `_row_nanmean` and use it at all four sites

In `src/carve/_consensus.py`, add after `reorder_consensus_matrix`:

```python
def _row_nanmean(a: np.ndarray) -> np.ndarray:
    """Mean of each row ignoring NaN, and NaN for a row with no finite entry.

    Returns exactly what ``np.nanmean(a, axis=1)`` returns, without the
    ``Mean of empty slice`` warning nanmean emits on every all-NaN row. Such
    rows are routine at small ``n_resamples``: a sample that was never
    co-sampled with any partner has nothing to average, and NaN is its score.
    """
    valid = ~np.isnan(a)
    count = valid.sum(axis=1)
    total = np.where(valid, a, 0.0).sum(axis=1)
    out = np.full(a.shape[0], np.nan, dtype=float)
    has_partner = count > 0
    out[has_partner] = total[has_partner] / count[has_partner]
    return out
```

In `stability_from_consensus`, replace

```python
    uncertainty_gini = 2.0 * np.nanmean(term, axis=1)
    uncertainty_ce = np.nanmean(entropy, axis=1)
```

with

```python
    uncertainty_gini = 2.0 * _row_nanmean(term)
    uncertainty_ce = _row_nanmean(entropy)
```

In the chunk loop of `stability_from_runs_anchored`, replace the identical pair the same way.

- [ ] Step 4: Run the consensus tests

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_consensus.py -v`
Expected: all pass, including `test_all_anchors_matches_the_exact_scores` and `test_chunking_does_not_change_the_result` (the helper is value-identical to nanmean).

- [ ] Step 5: Confirm the warning is gone from the carve suite

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests --ignore=tests/benchmarks -q -W "error:Mean of empty slice:RuntimeWarning"`
Expected: 686 passed. (This is the only warning promoted; the full `error` policy is Task 2.)

- [ ] Step 6: Lint and commit

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format src/`

```bash
git add src/carve/_consensus.py tests/test_consensus.py
git commit -m "fix(consensus): return NaN silently for samples with no co-sampled partner"
```

---

### Task 2: Turn on `filterwarnings = error` and fix every site it exposes

Measured on HEAD db4eca7 with only the nanmean warning ignored: the carve suite has 18 failures under `-W error` and the benchmarks suite 13. Every one is listed below with its fix. Spec: 1.3, 3.1, 3.6, part of 1.4, and the `slow` marker decision (removed).

Files:
- Modify: `pyproject.toml`
- Modify: `tests/test_api.py`, `tests/test_tl.py`, `tests/test_runner.py`, `tests/test_plotting.py`, `tests/benchmarks/test_run.py`, `tests/test_anchored_accuracy.py`, `tests/benchmarks/test_regression.py`
- Modify: `/Users/kaiwycik/GitHub/CARVE/_claude_playground/CLAUDE.md` (one line)

- [ ] Step 1: Configure pytest

Replace the `[tool.pytest.ini_options]` table in `pyproject.toml` with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-ra"]
xfail_strict = true
filterwarnings = [
  "error",
  # umap-learn warns at import when tensorflow is absent; the carve CI job
  # installs the [umap] extra without tensorflow.
  "ignore:Tensorflow not installed:ImportWarning",
  # joblib.numpy_pickle sets .shape on arrays, deprecated in NumPy 2.5.
  # Hit by CARVE.save and CARVE.load. Drop once joblib stops doing it.
  "ignore:Setting the shape on a NumPy array:DeprecationWarning:joblib",
]
```

Order matters: pytest applies the last matching filter, so `error` comes first and the ignores after it. The `markers` key is dropped with the `slow` marker; Task 3 adds a new `markers` entry.

Add to the `dev` extra in `[project.optional-dependencies]`:

```toml
  "pytest-xdist>=3",
  # test_ci_config.py parses the nightly workflow with yaml; it reached CI
  # only transitively (anndata -> zarr -> donfig -> pyyaml).
  "pyyaml>=6",
```

Run: `VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"`

- [ ] Step 2: Remove the `slow` marker

Delete the four `@pytest.mark.slow` decorators in `tests/test_anchored_accuracy.py` and the two in `tests/benchmarks/test_regression.py`. In the module docstring of `test_regression.py`, change "Marked slow and skipped unless" to "Skipped unless". The marker deselected nothing anywhere; the regression file is gated by `CARVE_RUN_REGRESSION=1` and stays so.

- [ ] Step 3: Run the carve suite to see the exposed sites

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests --ignore=tests/benchmarks -q --tb=line`
Expected: 8 failures (the 18 measured on HEAD, minus the 9 joblib save/load sites and the 1 umap import site the ignores now cover): `test_api.py::TestGetLabels::test_different_measures`, `test_runner.py::...::test_sweep_rank_inverted_for_min_cluster_size`, the two `test_tl.py::TestOptionalOutputs::test_partial_modes_omit_columns_without_raising` cases, `test_tl.py::...::test_measure_without_matching_mode_raises`, the two `test_tl.py::TestAnchoringProvenance` tests, and `test_tl.py::TestAnchoredConsensusObspGuard::test_default_store_consensus_survives_anchoring`. They are the sites fixed in Steps 4 to 7. If `PytestUnknownMarkWarning` appears, a `@pytest.mark.slow` was missed in Step 2.

- [ ] Step 4: `test_api.py` -- the `1se` fallback on consensus measures

`TestGetLabels.test_different_measures` calls `get_labels(measure="pac"|"gini"|"ce")` with the default `rule="1se"`; those columns have no `_se`, so selection warns and falls back to `max`. Replace the test with two:

```python
    @pytest.mark.parametrize("measure", ["stability", "generalizability"])
    def test_ari_measures_select_under_the_default_rule(self, fitted_carve, measure):
        labels = fitted_carve.get_labels(measure=measure)
        assert labels.shape == (60,)

    @pytest.mark.parametrize("measure", ["pac", "gini", "ce"])
    def test_consensus_measures_select_under_max(self, fitted_carve, measure):
        # These columns carry no standard error, so the default "1se" rule
        # cannot apply to them; "max" is the rule they support.
        labels = fitted_carve.get_labels(measure=measure, rule="max")
        assert labels.shape == (60,)

    def test_consensus_measure_under_1se_warns_and_falls_back(self, fitted_carve):
        with pytest.warns(RuntimeWarning, match="falling back to 'max' rule"):
            labels = fitted_carve.get_labels(measure="pac")
        assert labels.shape == (60,)
```

- [ ] Step 5: `test_runner.py` -- HDBSCAN all-noise resample

`TestRunValidationRecords.test_sweep_rank_inverted_for_min_cluster_size` triggers `UserWarning: All points in a subsample were labelled as noise`. It is a legitimate library warning unrelated to the assertion. Decorate the test:

```python
    @pytest.mark.filterwarnings(
        "ignore:All points in a subsample were labelled as noise:UserWarning"
    )
    def test_sweep_rank_inverted_for_min_cluster_size(self, X_two_clusters):
```

- [ ] Step 6: `test_tl.py` -- split modes and anchoring

In `TestOptionalOutputs.test_partial_modes_omit_columns_without_raising`, wrap the run:

```python
        with pytest.warns(RuntimeWarning, match="Non-default mode is experimental"):
            _run(adata_three_clusters, mode=mode, measure=measure)
```

Replace `test_measure_without_matching_mode_raises` in full (it also pins the exception, spec 1.4; the `ValueError` is pandas' `idxmax` on an all-NaN metric column):

```python
    def test_measure_without_matching_mode_raises(self, adata_three_clusters):
        """Selecting on a metric the run never computed must fail loudly.

        The error is pandas' idxmax refusing an all-NaN column: a
        generalizability-only run leaves every stability metric NaN.
        """
        with (
            pytest.warns(RuntimeWarning, match="Non-default mode is experimental"),
            pytest.raises(ValueError, match="all NA values"),
        ):
            _run(
                adata_three_clusters,
                mode="generalizability",
                measure="stability",
            )
```

The three anchoring tests each emit two warnings: the `RuntimeWarning` from `fit` and the `UserWarning` from `attach_results` about the skipped `obsp` write. pytest 8+ re-emits any warning a `pytest.warns` block did not match, so both must be caught. In `TestAnchoringProvenance.test_resolved_anchor_count_is_recorded_under_anchoring`:

```python
        with (
            pytest.warns(RuntimeWarning, match="anchored consensus"),
            pytest.warns(UserWarning, match="Anchored consensus is active"),
        ):
            _run(adata_three_clusters, anchor_threshold=40)
```

Same two-line change in `test_explicit_anchor_count_is_recorded` (call `_run(adata_three_clusters, consensus_anchors=25)`) and in `TestAnchoredConsensusObspGuard.test_default_store_consensus_survives_anchoring` (call `_run(adata_three_clusters, anchor_threshold=40)`). These were the two vacuous guards the audit verified by mutation (spec 1.3); with `match=` they now assert the anchoring warning.

- [ ] Step 7: Every remaining bare `pytest.warns(RuntimeWarning)`

Run: `grep -rn "pytest.warns(RuntimeWarning)" tests/`
Expected before the edit: 7 sites in `tests/test_api.py` (in `TestAnchoredConsensus`, `TestAnchoredLabels._fitted`, and the three `TestAnchoredLabels` tests that fit inline) and 2 in `tests/test_plotting.py` (the two anchoring tests at the bottom). Change each to:

```python
        with pytest.warns(RuntimeWarning, match="anchored consensus"):
```

Run the grep again. Expected: no output.

- [ ] Step 8: Verify the carve suite is green under `error`

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests --ignore=tests/benchmarks -q`
Expected: 691 passed (686, minus the one test Step 4 replaced, plus its six parametrized replacements), 0 warnings in the summary. The `-ra` summary lists no skips.

- [ ] Step 9: Benchmarks -- the quantile fallback warning in `test_run.py`

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks/test_run.py -q --tb=line`
Expected: 9 failures in `TestRunScenario`, all `RuntimeWarning: No estimators within quantile thresholds; falling back to max.` The tiny scenario's "hard" axis point produces it in the quantile-rule metrics. Add below the imports of `tests/benchmarks/test_run.py`:

```python
# The tiny scenario's hard axis point leaves no configuration inside the
# best score's quantile band for some seeds, so carve's quantile rule warns
# and falls back to max. That is the rule working as documented, not a
# defect in the runner, and it is data-dependent: pin the message, not the
# category, so any other RuntimeWarning still fails the test.
pytestmark = pytest.mark.filterwarnings(
    "ignore:No estimators within quantile thresholds:RuntimeWarning"
)
```

- [ ] Step 10: Verify the benchmarks suite

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks -q`
Expected: 753 passed, 10 skipped (the `-ra` summary now names them: 8 regression, 2 rpy2), 0 warnings. The 4 `test_studies.py` failures measured earlier were joblib sites and are covered by the ignore.

- [ ] Step 11: Update the playground note and commit

In `/Users/kaiwycik/GitHub/CARVE/_claude_playground/CLAUDE.md`, replace the bullet "There are no pytest `addopts`, so `-v` and `--cov` are never implicit." with "`addopts` is `-ra` only; `filterwarnings = error` is on, so a new warning fails the suite and needs either a `match=` assertion or a targeted `filterwarnings` mark. `-v` and `--cov` are never implicit." (The playground is not a git repository; no commit for that file.)

```bash
git add pyproject.toml tests/test_api.py tests/test_tl.py tests/test_runner.py tests/test_plotting.py tests/benchmarks/test_run.py tests/test_anchored_accuracy.py tests/benchmarks/test_regression.py
git commit -m "test: run both suites under filterwarnings=error and drop the decorative slow marker"
```

(`uv.lock` does not track the `dev` extra and is left alone.)

---

### Task 3: Make `tests/` a package and stop importing `conftest`

Spec: 3.4. Decision: `__init__.py` packages, prepend import mode unchanged. With `tests/__init__.py`, pytest inserts the repository root on `sys.path`, test modules become `tests.test_api` and so on, duplicate basenames become legal, and helpers are ordinary absolute imports.

Files:
- Create: `tests/__init__.py`, `tests/benchmarks/__init__.py`, `tests/benchmarks/figures/__init__.py`, `tests/benchmarks/datasets/__init__.py` (each one line: `"""Test package."""`)
- Create: `tests/_helpers.py`
- Modify: `tests/conftest.py`, `pyproject.toml`
- Modify: `tests/test_api.py`, `tests/test_cluster.py`, `tests/test_selection.py`, `tests/test_plotting.py`, `tests/benchmarks/test_benchmarks_types.py`

Interfaces:
- Produces: `tests._helpers.with_sweep_cols` (same signature as today's `conftest.with_sweep_cols`); marker `pytest.mark.requires_graph`, skipped by a `pytest_collection_modifyitems` hook when `igraph` or `leidenalg` is absent.

- [ ] Step 1: Create the packages and the helper module

Create the four `__init__.py` files. Create `tests/_helpers.py` containing the module docstring below and the `with_sweep_cols` function moved verbatim from `tests/conftest.py` (with its `import pandas as pd` and `from carve._sweep import resolve_sweep`):

```python
"""Helpers shared across the carve test files.

Not a conftest: conftest.py holds fixtures and hooks, and pytest discourages
importing it as a module. Anything a test file imports by name lives here.
"""
```

- [ ] Step 2: Rewrite the top of `tests/conftest.py`

Replace everything above the `rng` fixture with:

```python
"""Shared fixtures and hooks for the CARVE test suite."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from tests._helpers import with_sweep_cols

# Leiden and Louvain live behind the optional [graph] extra. Tests that need
# them carry @pytest.mark.requires_graph; the hook below skips them when the
# extra is absent.
_HAS_GRAPH = all(
    importlib.util.find_spec(m) is not None for m in ("igraph", "leidenalg")
)


def pytest_collection_modifyitems(config, items):
    if _HAS_GRAPH:
        return
    skip = pytest.mark.skip(reason="requires the [graph] extra (igraph + leidenalg)")
    for item in items:
        if "requires_graph" in item.keywords:
            item.add_marker(skip)
```

Delete the old `requires_graph` object and the old `with_sweep_cols` definition (and the now-unused `from carve._sweep import resolve_sweep`). The `results_df`, `resolution_results_df` and `min_cluster_size_results_df` fixtures keep calling `with_sweep_cols`, now the imported one.

Register the marker in `pyproject.toml` (a new key under `[tool.pytest.ini_options]`; unregistered marks warn, and warnings are errors now):

```toml
markers = [
  "requires_graph: needs the [graph] extra (igraph + leidenalg); skipped without it",
]
```

- [ ] Step 3: Update the four importing files

In `tests/test_api.py` and `tests/test_cluster.py`: delete `from conftest import requires_graph`; change every `@requires_graph` decorator (one in `test_api.py` on `TestResolutionMode`, three in `test_cluster.py` on `TestBuildKnnGraph`, `TestLeidenClustering`, `TestLouvainClustering`) to `@pytest.mark.requires_graph`.

In `tests/test_selection.py` and `tests/test_plotting.py`: change `from conftest import with_sweep_cols` to `from tests._helpers import with_sweep_cols`.

In `tests/benchmarks/test_benchmarks_types.py`: the module docstring explains the file was named to dodge a basename collision. Replace it with `"""Tests for the leaf dataclasses in benchmarks._types."""` (the file keeps its name; the reason no longer applies).

- [ ] Step 4: Verify collection and both suites

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests --collect-only -q | tail -3`
Expected: the same item count as before (691 carve + 763 benchmarks), no errors. Node ids now read `tests/test_api.py::...` exactly as before (prepend mode with packages keeps paths).

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q -x --ignore=tests/benchmarks` and then `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks -q -x`
Expected: green, same counts as Task 2.

Verify the marker path: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_cluster.py -q -p no:cacheprovider -o markers="requires_graph: x"` passes; then confirm the skip branch by temporarily editing `_HAS_GRAPH = False` in `conftest.py`, running `tests/test_cluster.py -q -ra`, seeing the Leiden and Louvain classes reported as skipped with the reason, and reverting.

- [ ] Step 5: Commit

```bash
git add tests/__init__.py tests/benchmarks/__init__.py tests/benchmarks/figures/__init__.py tests/benchmarks/datasets/__init__.py tests/_helpers.py tests/conftest.py pyproject.toml tests/test_api.py tests/test_cluster.py tests/test_selection.py tests/test_plotting.py tests/benchmarks/test_benchmarks_types.py
git commit -m "test: make tests/ a package; move shared helpers out of conftest; register requires_graph"
```

---

### Task 4: One matplotlib setup, rcParams isolation, and an `ax` fixture

Spec: 3.2 (rcParams leak, verified), 3.3 (16 copies of `matplotlib.use`, 2 duplicate `close_figures`, 60 hand-written `plt.subplots()`/`plt.close(fig)` pairs).

Files:
- Modify: `tests/conftest.py`
- Modify: every test file with a module-level `matplotlib.use("Agg")` (list in Step 2)
- Modify: `tests/test_api.py`, `tests/test_plotting.py` (delete the duplicate `close_figures`)
- Modify: `tests/benchmarks/test_theme.py`, `tests/benchmarks/test_panels.py`

- [ ] Step 1: Write the leak probe test

In `tests/benchmarks/test_theme.py`, remove `CLUSTER_CMAP_NAME` from the import block (unused), and append to `TestRcParams`, after `test_theme_context_restores_previous_state`:

```python
    def test_apply_theme_does_not_leak_into_the_next_test(self):
        # Runs after test_apply_theme_actually_mutates_rcparams in file
        # order. apply_theme sets axes.spines.top to False for the whole
        # process; the autouse rc_context fixture in tests/conftest.py must
        # have undone that before this test started. Before that fixture
        # existed this assertion failed.
        assert plt.rcParams["axes.spines.top"] is True
        assert plt.rcParams["pdf.fonttype"] == 3
```

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks/test_theme.py::TestRcParams -v`
Expected: `test_apply_theme_does_not_leak_into_the_next_test` FAILS on the first assertion (`axes.spines.top` is False).

- [ ] Step 2: One backend selection and per-test rcParams isolation

In `tests/conftest.py`, add directly after the imports (before `_HAS_GRAPH`):

```python
# Select the file-backed backend once, before any test module imports
# pyplot. conftest.py is imported before collection reaches any test file.
matplotlib.use("Agg", force=True)
```

with `import matplotlib` added to the import block. Replace the `_headless_matplotlib` fixture at the bottom with:

```python
@pytest.fixture(autouse=True)
def _isolated_matplotlib_state():
    """Undo every rcParams change a test makes and close its figures.

    rc_context restores each rcParam changed inside the block on exit, so a
    test that calls apply_theme() or assigns plt.rcParams[...] cannot leak
    into the next test.
    """
    import matplotlib.pyplot as plt

    with matplotlib.rc_context():
        yield
    plt.close("all")
```

Delete the module-level `import matplotlib` / `matplotlib.use("Agg")` pair (and the `# non-interactive backend for tests` comment where present) from these files, keeping their `import matplotlib.pyplot as plt` lines:
`tests/test_api.py`, `tests/test_plotting.py`, `tests/benchmarks/test_panels.py`, `tests/benchmarks/test_theme.py`, and all eleven files under `tests/benchmarks/figures/` (`test_benchmarking_examples.py`, `test_benchmarking_results.py`, `test_carve_output.py`, `test_case_study.py`, `test_cusanovich_results.py`, `test_figure_contract.py`, `test_heca_results.py`, `test_reference_scatter.py`, `test_scaling.py`, `test_scenario_overview.py`, `test_study_scaling.py`).

Delete the `close_figures` autouse fixtures from `tests/test_api.py` and `tests/test_plotting.py` (the conftest fixture already closes figures after every test).

Run: `grep -rn "matplotlib.use" tests/`
Expected: exactly one line, in `tests/conftest.py`.

- [ ] Step 3: Verify the probe passes and nothing else changed

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks/test_theme.py tests/benchmarks/test_panels.py tests/test_plotting.py tests/test_api.py -q`
Expected: green, including the new probe.

- [ ] Step 4: An `ax` fixture for `test_panels.py`

Add near the top of `tests/benchmarks/test_panels.py`, after the imports:

```python
@pytest.fixture
def ax():
    """A fresh Axes; the conftest fixture closes every figure afterwards."""
    _, ax = plt.subplots()
    return ax
```

Then apply this transformation to every test in the file:

```python
# before
    def test_returns_the_axes(self):
        fig, ax = plt.subplots()
        Z = np.random.default_rng(0).normal(size=(30, 2))
        labels = np.repeat([0, 1, 2], 10)
        assert scatter_clusters(ax, Z, labels) is ax
        plt.close(fig)

# after
    def test_returns_the_axes(self, ax):
        Z = np.random.default_rng(0).normal(size=(30, 2))
        labels = np.repeat([0, 1, 2], 10)
        assert scatter_clusters(ax, Z, labels) is ax
```

Rules: add `ax` to the test's parameters (after `self`, before other fixtures); delete the `fig, ax = plt.subplots()` line and the trailing `plt.close(fig)`; if the body still references `fig` (for example `fig.canvas.draw()`, `fig.legends`), insert `fig = ax.figure` as the first line of the body. Tests that call `plt.subplots(...)` with arguments (multiple axes, a figsize) are left alone. Tests that create two figures keep their second `plt.subplots()` and drop only its `plt.close`.

Run: `grep -c "plt.subplots()" tests/benchmarks/test_panels.py`
Expected: 1 (the fixture). `grep -c "plt.close(fig)" tests/benchmarks/test_panels.py` is 0.

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks/test_panels.py -q`
Expected: same count as before, green.

- [ ] Step 5: Commit

```bash
git add tests/conftest.py tests/test_api.py tests/test_plotting.py tests/benchmarks/test_panels.py tests/benchmarks/test_theme.py tests/benchmarks/figures/
git commit -m "test: select the Agg backend once and isolate rcParams per test"
```

---

### Task 5: One definition each for the spies and the stub CARVE

Spec: 3.2 (mutable class attributes, hand-rolled `try/finally`), 3.3 (`_NJobsSpy` twice, the stub CARVE five times, `_SpyCARVE`).

The classifier spies cannot record into an instance attribute: `sklearn.base.clone` rebuilds the estimator from `get_params()` and deep-copies non-estimator parameters, so any list handed to the constructor is copied, not shared. A fresh class per test, created by a factory, gives each test its own class-level list with nothing to clear.

Files:
- Modify: `tests/_helpers.py` (add three factories)
- Create: `tests/benchmarks/_helpers.py`
- Modify: `tests/test_api.py`, `tests/test_runner.py`, `tests/benchmarks/test_studies.py`, `tests/benchmarks/test_panels.py`, `tests/benchmarks/figures/test_case_study.py`, `tests/benchmarks/figures/test_heca_results.py`, `tests/benchmarks/figures/test_cusanovich_results.py`

Interfaces:
- Produces in `tests._helpers`: `make_njobs_spy() -> type`, `make_seed_spy() -> type`, `make_parallel_spy() -> type`. Each returns a new class whose `seen: list` is empty at creation.
- Produces in `tests.benchmarks._helpers`: `StubCarve(results: pd.DataFrame, select: Callable[[str, bool], tuple[str, int]], labels: np.ndarray | None = None)` with `_select_row`, `get_k`, `get_labels`, `get_labels_calls`; `make_carve_spy() -> type` whose instances record `captured_kwargs` and stand in for a fitted model.

- [ ] Step 1: Add the spy factories to `tests/_helpers.py`

Append (with `import numpy as np` and `from sklearn.base import BaseEstimator, ClassifierMixin` added to the imports):

```python
def make_njobs_spy() -> type:
    """A classifier class that records the n_jobs CARVE injects at fit time.

    A new class per call: sklearn.clone rebuilds estimators from get_params
    and deep-copies any list passed to the constructor, so the record has to
    live on the class, and a fresh class keeps tests from sharing it.
    """

    class NJobsSpy(BaseEstimator, ClassifierMixin):
        seen: list = []

        def __init__(self, n_jobs=None):
            self.n_jobs = n_jobs

        def fit(self, X, y):
            type(self).seen.append(self.n_jobs)
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.full(X.shape[0], self.classes_[0])

    return NJobsSpy


def make_seed_spy() -> type:
    """A classifier class that records the random_state CARVE injects."""

    class SeedSpy(BaseEstimator, ClassifierMixin):
        seen: list = []

        def __init__(self, random_state=None):
            self.random_state = random_state

        def fit(self, X, y):
            type(self).seen.append(self.random_state)
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.full(X.shape[0], self.classes_[0])

    return SeedSpy


def make_parallel_spy() -> type:
    """A stand-in for joblib.Parallel: records n_jobs, runs the tasks inline."""

    class ParallelSpy:
        seen: list = []

        def __init__(self, n_jobs=None, **kwargs):
            type(self).seen.append(n_jobs)

        def __call__(self, tasks):
            return [func(*args, **kwargs) for func, args, kwargs in tasks]

    return ParallelSpy
```

- [ ] Step 2: Use them in `tests/test_runner.py`

Delete the `_NJobsSpy` and `_ParallelSpy` classes. Add `from tests._helpers import make_njobs_spy, make_parallel_spy` to the imports. In the class whose autouse fixture is `eleven_cores` (the one that monkeypatches `Parallel`), replace the fixture and the class-level references:

```python
    @pytest.fixture(autouse=True)
    def eleven_cores(self, monkeypatch):
        monkeypatch.setattr(carve_utils, "cpu_count", lambda: 11)
        self.parallel_spy = make_parallel_spy()
        self.njobs_spy = make_njobs_spy()
        monkeypatch.setattr(carve_runner, "Parallel", self.parallel_spy)
```

and change every `_ParallelSpy.seen` in that class to `self.parallel_spy.seen`, every `_NJobsSpy()` to `self.njobs_spy()` and every `_NJobsSpy.seen` to `self.njobs_spy.seen`. (Read the class to find them: `grep -n "_ParallelSpy\|_NJobsSpy" tests/test_runner.py` must return nothing afterwards.)

- [ ] Step 3: Use them in `tests/test_api.py`

Delete the `_SeedSpy` and `_NJobsSpy` classes and the now-unused `from sklearn.base import BaseEstimator, ClassifierMixin`. Import `from tests._helpers import make_njobs_spy, make_seed_spy`. Rewrite the two tests that used them:

```python
    def test_fit_time_random_state_is_honored_by_the_extension(self):
        # fit(X, random_state=7) is honored by the anchor draw and by
        # run_validation; the extension must read the same resolved seed
        # rather than self.random_state, which fit() never writes.
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning, match="anchored consensus"):
            c.fit(X, random_state=7)

        spy = make_seed_spy()
        c.classifier = spy()
        c._extend_anchor_labels(np.arange(c.consensus_anchors_.size) % 2)
        assert spy.seen == [7]

    def test_extension_fit_receives_the_whole_core_budget(self, monkeypatch):
        # fit() spreads n_jobs=4 over 4 workers x 2 threads on 11 cores. The
        # extension is one fit outside that loop, so it gets the product, not
        # the per-worker share.
        monkeypatch.setattr(carve_utils, "cpu_count", lambda: 11)
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            anchor_threshold=30,
            n_jobs=4,
            random_state=0,
        )
        with pytest.warns(RuntimeWarning, match="anchored consensus"):
            c.fit(X)

        spy = make_njobs_spy()
        c.classifier = spy()
        c._extend_anchor_labels(np.arange(c.consensus_anchors_.size) % 2)
        assert spy.seen == [8]
```

- [ ] Step 4: `monkeypatch` in `TestRowIdentity` and `test_misaligned_config_id_raises`

Every `try: ... finally: fitted_identity_single.estimator_results_ = original` block in `TestRowIdentity` becomes a `monkeypatch.setattr`. Pattern:

```python
    def test_labels_survive_reindexing(self, fitted_identity_single, monkeypatch):
        before = fitted_identity_single.get_labels()
        monkeypatch.setattr(
            fitted_identity_single,
            "estimator_results_",
            self._reindexed(fitted_identity_single.estimator_results_),
        )
        np.testing.assert_array_equal(fitted_identity_single.get_labels(), before)
```

Apply to `test_config_id_survives_reindexing` (the one that asserts `after_id == before_id`), `test_consensus_matrix_survives_reindexing`, `test_labels_survive_reindexing`, `test_labels_survive_row_reordering` (patched value `original.sample(frac=1, random_state=0)`), and `test_get_estimator_survives_a_shuffle` (`original.sample(frac=1, random_state=1)`). `monkeypatch.setattr` on a dataclass instance sets the attribute and restores the original at teardown even when the assertion fails.

In `test_misaligned_config_id_raises`, replace the manual patch and `try/finally` with:

```python
        monkeypatch.setattr(carve_runner, "run_validation", _corrupt)
        monkeypatch.setattr(carve_api, "run_validation", _corrupt)
        with pytest.raises(RuntimeError, match="config_id is misaligned"):
            carve.fit(X_two_clusters)
```

and add `monkeypatch` to that test's parameters.

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_api.py tests/test_runner.py -q`
Expected: green. `grep -n "finally:" tests/test_api.py` returns nothing.

- [ ] Step 5: Create `tests/benchmarks/_helpers.py`

```python
"""Helpers shared across the benchmarks test files."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


class StubCarve:
    """Minimal stand-in for a fitted CARVE object.

    The panels and figure code read three members off a fitted model:
    estimator_results_, _select_row() and get_k() (and prepare_composite
    also calls get_labels()). This implements exactly those instead of
    fitting a real model.

    Column names are the canonical estimator_results_ names a real fitted
    model uses (ari_stability and ari_generalizability plus their _se
    variants, method_id, method_label, n_clusters). "stability" and
    "generalizability" are only measure aliases, never column names; a stub
    that named its columns after the aliases would let carve_lines index the
    alias directly and still pass.

    Parameters
    ----------
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

    def _select_row(self, *, measure="stability", rule="1se", not_two=False):
        method_id, k = self._select(measure, not_two)
        results = self.estimator_results_
        row = results.loc[
            (results["method_id"] == method_id) & (results["n_clusters"] == k)
        ].iloc[0]
        return row, 0, int(k), False

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return int(self._select(measure, not_two)[1])

    def get_labels(self, *, measure="stability", rule="1se", not_two=False):
        self.get_labels_calls.append(
            {"measure": measure, "rule": rule, "not_two": not_two}
        )
        if self._labels is None:
            raise AssertionError("get_labels called on a StubCarve built without labels")
        return self._labels


def simple_results(ks, method_label: str) -> pd.DataFrame:
    """One method swept over ks with monotone metrics, for composite figures."""
    ks = list(ks)
    n = len(ks)
    return pd.DataFrame(
        {
            "n_clusters": ks,
            "method_id": ["m0"] * n,
            "method_label": [method_label] * n,
            "ari_stability": list(np.linspace(0.1, 0.3, n)),
            "ari_generalizability": list(np.linspace(0.15, 0.35, n)),
        }
    )


def make_carve_spy() -> type:
    """A CARVE stand-in class that records its constructor kwargs.

    Used where a test needs to see what a call site passed to CARVE(...)
    without paying for a fit. A new class per call keeps the record private
    to the test. Instances expose enough of a fitted model for
    fit_or_load_carve and study_scaling_sweep to run to completion.
    """

    class SpyCARVE:
        captured_kwargs: dict | None = None

        def __init__(self, **kwargs):
            type(self).captured_kwargs = kwargs
            self.estimator_results_ = pd.DataFrame({"config_id": [0]})
            self._n = 0

        def fit(self, X, *args, **kwargs):
            self._n = int(np.asarray(X).shape[0])
            return self

        def save(self, path):
            Path(path).write_text("stub")

        def get_labels(self, **kwargs):
            return np.zeros(self._n, dtype=int)

        def get_k(self, **kwargs):
            return 2

    return SpyCARVE
```

- [ ] Step 6: Replace the five stubs and `_SpyCARVE`

`tests/benchmarks/test_panels.py`: delete `_StubCarve`; import `from tests.benchmarks._helpers import StubCarve`; where `_carve_obj()` (or any test) built `_StubCarve(results, selection)` with `selection` a dict `measure -> {"method_id", "n_clusters"}`, build

```python
    StubCarve(
        results,
        select=lambda measure, not_two: (
            selection[measure]["method_id"],
            selection[measure]["n_clusters"],
        ),
    )
```

`tests/benchmarks/figures/test_case_study.py`: delete `_StubCarve`; keep the table it built (two methods `m0`/KMeans and `m1`/AgglomerativeClustering over `ks`) in a small module function `_two_method_results(ks)`, and construct

```python
def _stub(ks, labels=None):
    ks = list(ks)

    def select(measure, not_two):
        candidates = ks[1:] if not_two else ks
        return "m0", candidates[len(candidates) // 2]

    return StubCarve(_two_method_results(ks), select=select, labels=labels)
```

replacing every `_StubCarve(ks, labels=...)` call with `_stub(ks, labels=...)`. `get_labels_calls` keeps its name, so the forwarding assertions do not change.

`tests/benchmarks/figures/test_heca_results.py`: delete the nested `_Carve` class in the `inputs` fixture and the module-level `_CarveForAriPanel`; both become

```python
StubCarve(simple_results([3, 4, 5], "MiniBatchKMeans"), select=lambda m, nt: ("m0", 4))
```

`tests/benchmarks/figures/test_cusanovich_results.py`: the nested `_Carve` becomes

```python
StubCarve(simple_results([4, 5, 6], "KMeans"), select=lambda m, nt: ("m0", 5))
```

`tests/benchmarks/test_studies.py`: delete `_SpyCARVE`; import `make_carve_spy`; in each of the two `TestConsensusAnchorsForwarding` tests:

```python
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)
        ...
        assert spy.captured_kwargs["consensus_anchors"] == 123
```

(and `assert "consensus_anchors" not in spy.captured_kwargs` in the second).

Run: `grep -rn "class _StubCarve\|class _Carve\|class _CarveForAriPanel\|class _SpyCARVE\|class _NJobsSpy\|class _SeedSpy\|class _ParallelSpy" tests/`
Expected: no output.

- [ ] Step 7: Run the touched files and commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks/test_panels.py tests/benchmarks/figures/test_case_study.py tests/benchmarks/figures/test_heca_results.py tests/benchmarks/figures/test_cusanovich_results.py tests/benchmarks/test_studies.py -q`
Expected: green, same counts.

Verify by mutation that the shared stub still detects a forwarding bug: in `src/benchmarks/_panels.py`, find the `carve_lines` call to `carve._select_row(...)` and temporarily drop its `not_two=` argument; run `tests/benchmarks/figures/test_case_study.py -q`; at least one test must fail; revert.

```bash
git add tests/_helpers.py tests/benchmarks/_helpers.py tests/test_api.py tests/test_runner.py tests/benchmarks/test_studies.py tests/benchmarks/test_panels.py tests/benchmarks/figures/test_case_study.py tests/benchmarks/figures/test_heca_results.py tests/benchmarks/figures/test_cusanovich_results.py
git commit -m "test: define the classifier spies and the stub CARVE once"
```

---

### Task 6: Import hygiene and the small things

Spec: 3.4 (mid-file and duplicated imports), 3.7, and the `dataclasses.replace` idiom.

Files:
- Modify: `tests/benchmarks/test_artifacts.py`, `tests/benchmarks/test_benchmarks_types.py`, `tests/benchmarks/test_studies.py`, `tests/benchmarks/figures/test_scaling.py`, `tests/benchmarks/figures/test_benchmarking_results.py`, `tests/benchmarks/test_panels.py`, `tests/test_grids.py`, `tests/test_api.py`, `tests/test_anchored_accuracy.py`, and whichever files the ruff pass touches

- [ ] Step 1: Hoist the mid-file imports by hand

`tests/benchmarks/test_artifacts.py`: move the `from benchmarks._artifacts import (RUNTIME_SCHEMA, read_runtimes, write_runtime_checkpoint)` block at line ~259 into the top-level `from benchmarks._artifacts import (...)` block.

`tests/benchmarks/test_benchmarks_types.py`: move `from benchmarks._types import Scenario` (line ~56), `import numpy as np` and `from benchmarks._types import Manifest, Study` (lines ~145-147) to the top; the single import becomes `from benchmarks._types import KNOWN_ESTIMATORS, Axis, EstimatorSpec, Manifest, Scenario, Study`. While there, delete `test_unknown_estimator_still_raises` (spec 1.1: it duplicates `TestEstimatorSpec.test_raises_on_an_unknown_name`).

`tests/benchmarks/test_studies.py`: the local `from carve.cluster import SpectralClustering` / `from sklearn.cluster import KMeans` / `AgglomerativeClustering` / `LeidenClustering` imports inside the `TestNewStudies` tests (lines ~558-575) are already imported at module level except `LeidenClustering`; add `LeidenClustering` to the module-level `from carve.cluster import ...` and delete the function-local imports.

`tests/benchmarks/figures/test_scaling.py`, `tests/benchmarks/figures/test_benchmarking_results.py`, `tests/benchmarks/test_panels.py`: hoist the function-local `from benchmarks._theme import METRIC_COLORS` / `FONT_SIZES` to the module import block and delete the local lines.

`tests/test_grids.py`: delete the `from sklearn.model_selection import ParameterGrid` inside the test at line ~90 (it is imported at module level).

`tests/test_api.py`: delete `import warnings as _w` and change the two `_w.catch_warnings()` / `_w.simplefilter(...)` uses to `warnings.catch_warnings()` / `warnings.simplefilter(...)`.

`tests/test_anchored_accuracy.py`: move `import warnings` out of `_fit` to the module imports and use `warnings.catch_warnings()` / `warnings.simplefilter("ignore", RuntimeWarning)` there. (Task 16 rewrites this file's fixtures; the import stays.)

- [ ] Step 2: The one-off ruff pass over `tests/`

Run: `.venv/bin/ruff check tests/ --select E402,F401,F811,I --fix`
Expected: the 19 auto-fixable findings (import sorting and the unused `CLUSTER_CMAP_NAME`, if Task 4 did not already remove it) are fixed and nothing remains: `.venv/bin/ruff check tests/ --select E402,F401,F811,I` prints `All checks passed!`. Do not run `ruff format` on `tests/`.

- [ ] Step 3: Small things

`tests/benchmarks/test_artifacts.py::TestConfigHash::test_changes_when_an_anchor_changes`: replace the `type(s)(name=..., axis=..., anchors=anchors, shared=..., estimator=...)` reconstruction with `changed = dataclasses.replace(s, anchors=anchors)` (add `import dataclasses`), which keeps `k_star`, `candidate_k`, `n_seeds` and `n_trees` instead of silently dropping them.

`tests/benchmarks/test_panels.py` and `tests/benchmarks/figures/test_benchmarking_results.py` read `legend._ncols`. Above the first such read in each file add the comment:

```python
    # Legend._ncols is private; it was _ncol before matplotlib 3.6 and the
    # matplotlib>=3.9.4 floor makes the current name safe.
```

- [ ] Step 4: Run both suites and commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q`
Expected: green; the item count drops by one (the deleted duplicate).

```bash
git add tests/
git commit -m "test: hoist mid-file imports, sort import blocks, use dataclasses.replace"
```

---

### Task 7: Carve-side defects -- copy-paste, misnamed, and too-weak tests

Spec: 1.1, 1.2, 1.4 for `tests/test_api.py`, `tests/test_types.py`, `tests/test_plotting.py`, `tests/test_selection.py`, `tests/test_consensus.py`, `tests/test_runner.py`, `tests/test_accuracy.py`, `tests/test_utils.py`. Every replacement below was run against the current code; the expected values are what the code produces today.

Files:
- Modify: the eight test files named above

- [ ] Step 1: `tests/test_api.py`

`test_generalizability_scores_populated`: the second assertion becomes `assert fitted_carve.consensus_generalizability_matrices_ is not None`.

Replace `test_reproducibility` and `test_per_call_random_state`:

```python
    def test_reproducibility(self, X_two_clusters):
        def make_carve():
            return CARVE(
                n_clusters=2,
                n_resamples=3,
                subsample_ratio=0.8,
                estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
                normalization_options=[],
                dim_reduction_options=[],
                verbose=0,
                random_state=42,
            )

        c1 = make_carve().fit(X_two_clusters)
        c2 = make_carve().fit(X_two_clusters)
        pd.testing.assert_frame_equal(c1.estimator_results_, c2.estimator_results_)
        np.testing.assert_array_equal(
            c1.consensus_matrices_[0], c2.consensus_matrices_[0]
        )
        np.testing.assert_array_equal(
            c1.consensus_generalizability_matrices_[0],
            c2.consensus_generalizability_matrices_[0],
        )

    def test_per_call_random_state(self, X_two_clusters):
        def make_carve():
            return CARVE(
                n_clusters=2,
                n_resamples=3,
                subsample_ratio=0.8,
                estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
                normalization_options=[],
                dim_reduction_options=[],
                verbose=0,
            )

        a = make_carve().fit(X_two_clusters, random_state=99)
        b = make_carve().fit(X_two_clusters, random_state=99)
        c = make_carve().fit(X_two_clusters, random_state=0)
        pd.testing.assert_frame_equal(a.estimator_results_, b.estimator_results_)
        # The subsample draws differ between seeds, so the pairs that were
        # ever co-sampled differ and the NaN pattern of the matrix with them.
        assert np.array_equal(
            a.consensus_matrices_[0], b.consensus_matrices_[0], equal_nan=True
        )
        assert not np.array_equal(
            a.consensus_matrices_[0], c.consensus_matrices_[0], equal_nan=True
        )
```

Rename `test_reference_labels_alignment` to `test_get_labels_is_idempotent` with docstring `"""Calling get_labels twice with the same arguments returns the same labels."""` (the real alignment tests are Task 14).

- [ ] Step 2: `tests/test_types.py`

Delete `test_invalid_mode_type` (identical to `test_invalid_mode`).

- [ ] Step 3: `tests/test_plotting.py`

Add `from sklearn.decomposition import PCA` and `from matplotlib.collections import PolyCollection` to the imports. Replace these tests:

```python
    def test_no_legend(self, metric_results_df):
        ax = plot_metric_over_n_clusters(
            metric_results_df,
            measure="stability",
            legend=False,
        )
        assert ax.get_legend() is None
        assert (
            plot_metric_over_n_clusters(
                metric_results_df, measure="stability"
            ).get_legend()
            is not None
        )
```

```python
    def test_nan_handling(self):
        # Never co-sampled pairs are drawn at 0.5, the value get_labels also
        # substitutes; the diagonal is forced to 1.
        M = np.array([[1.0, np.nan], [np.nan, 1.0]])
        labels = np.array([0, 1])
        ax = plot_consensus_matrix(M, labels)
        np.testing.assert_array_equal(
            np.asarray(ax.images[0].get_array()), [[1.0, 0.5], [0.5, 1.0]]
        )
```

```python
    def test_with_existing_ax(self):
        fig, ax = plt.subplots()
        M = np.eye(3)
        labels = np.array([0, 1, 1])
        assert plot_consensus_matrix(M, labels, ax=ax) is ax
        assert len(ax.images) == 1
```

(this is a new test in `TestPlotConsensusMatrix`; the existing `test_with_existing_ax` in `TestPlotMetricOverNClusters` already asserts identity and stays.)

```python
    def test_custom_order(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([0, 0, 1, 1])
        ax = plot_cluster_boxplot(scores, labels, order=[1, 0])
        # Tick labels are one-based cluster numbers, in the requested order.
        assert [t.get_text() for t in ax.get_xticklabels()] == ["2", "1"]
```

```python
    def test_violin_no_clip_when_ylim_none(self):
        """ylim=None with fit_ylim=False leaves the KDE bodies unclipped.

        matplotlib evaluates the KDE on [min, max] of each group, so the top
        vertex of the unclipped bodies is the data maximum; a ylim inside the
        data range clips it.
        """
        scores = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 0.60, 0.61, 0.62])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])

        def top(ax):
            bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
            return max(p.vertices[:, 1].max() for b in bodies for p in b.get_paths())

        clipped = plot_cluster_violin(scores, labels, ylim=(0.0, 0.9), fit_ylim=False)
        assert top(clipped) == pytest.approx(0.9)
        free = plot_cluster_violin(scores, labels, ylim=None, fit_ylim=False)
        assert top(free) == pytest.approx(0.99)
```

```python
    def test_high_dim_uses_pca(self):
        X = np.random.RandomState(0).randn(20, 10)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(20)
        ax = plot_cluster_scatter(X, labels, scores, sort_order=False)
        expected = PCA(n_components=2, random_state=0).fit_transform(X)
        np.testing.assert_allclose(ax.collections[0].get_offsets(), expected)

    def test_alpha_range(self):
        X = np.random.RandomState(0).randn(20, 2)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(20)
        ax = plot_cluster_scatter(X, labels, scores, alpha_range=(0.3, 0.9))
        alphas = ax.collections[0].get_facecolors()[:, 3]
        assert alphas.min() == pytest.approx(0.3)
        assert alphas.max() == pytest.approx(0.9)
```

In `test_plot_consensus_matrix_under_anchoring` (bottom of the file), replace the last two lines with:

```python
    ax = c.plot_consensus_matrix(k=2)
    # The rendered image is the anchor block, not the full 60-by-60 matrix.
    assert np.asarray(ax.images[0].get_array()).shape == (30, 30)
```

(Task 10 later shares the fit between the two anchoring tests.)

- [ ] Step 4: `tests/test_selection.py`

`TestSelectBestRow1se.test_return_idx`: `assert idx == 0` (k=2 is the only row within one SE, and it is index label 0). `TestSelectBestRowQuantile.test_return_idx`: `assert idx == 0` (same reasoning with the quantile band). `TestSelectBestRowByRule.test_return_idx`: `assert idx == 0`. `TestSelectBestRowByRule.test_1se_rule`: `assert row["n_clusters"] == 2`. `test_quantile_rule`: `assert row["n_clusters"] == 2`. Each replaces an `isinstance` assertion; keep the rest of each test.

- [ ] Step 5: `tests/test_consensus.py::TestReorderConsensusMatrix`

Replace the class:

```python
class TestReorderConsensusMatrix:
    def _interleaved(self):
        # Two perfect blocks, {0, 2, 4} and {1, 3, 5}, interleaved by index.
        M = np.zeros((6, 6))
        for group in ([0, 2, 4], [1, 3, 5]):
            for i in group:
                for j in group:
                    M[i, j] = 1.0
        return M

    def test_groups_the_interleaved_blocks(self):
        reordered, order = reorder_consensus_matrix(self._interleaved())
        assert sorted(order) == list(range(6))
        # Each block ends up contiguous: the leaf order lists one group's
        # three members, then the other's.
        first, second = set(order[:3]), set(order[3:])
        assert {first, second} == {frozenset({0, 2, 4}), frozenset({1, 3, 5})}
        # And the reordered matrix is block diagonal.
        np.testing.assert_array_equal(reordered[:3, :3], 1.0)
        np.testing.assert_array_equal(reordered[3:, 3:], 1.0)
        np.testing.assert_array_equal(reordered[:3, 3:], 0.0)

    def test_reordering_is_a_permutation_of_the_input(self):
        M = self._interleaved()
        M[0, 2] = M[2, 0] = 0.7
        reordered, order = reorder_consensus_matrix(M)
        np.testing.assert_array_equal(reordered, M[np.ix_(order, order)])
        assert sorted(reordered.ravel()) == sorted(M.ravel())

    def test_nan_handling(self):
        M = np.array([[1.0, np.nan], [np.nan, 1.0]])
        reordered, order = reorder_consensus_matrix(M, fill_nan_for_order=0.0)
        assert reordered.shape == (2, 2)
        assert sorted(order) == [0, 1]
        # The NaN is passed through untouched; fill_nan_for_order only
        # affects the ordering, not the returned values.
        assert np.isnan(reordered[0, 1])
```

(`first`/`second` are compared as a set of frozensets so either block may come first.)

- [ ] Step 6: `tests/test_runner.py`, `tests/test_accuracy.py`, `tests/test_utils.py`

`TestComputeStabilityAri.test_basic`: the overlap is samples 1 and 2, in two singleton clusters under both labelings, so ARI is exactly 1: replace `assert -0.5 <= ari <= 1.0` with `assert ari == 1.0`.

Delete `TestResampleResult.test_field_count` (a field count is a change detector; the named-field test above it is the contract).

`tests/test_accuracy.py::test_wrong_predictions`: replace `assert np.any(scores < 1.0)` with

```python
        # Hungarian alignment maps the tie to the identity, so samples 0 and
        # 3 are right and 1 and 2 are wrong.
        np.testing.assert_array_equal(scores, [1.0, 0.0, 0.0, 1.0])
```

`tests/test_utils.py::TestEnsure2dArray::test_invalid_type`: `pytest.raises(ValueError, match="Input must be a NumPy array")`.

`tests/test_utils.py::test_rejects_degenerate_counts`: replace the parametrization and body:

```python
    @pytest.mark.parametrize(
        ("bad", "message"),
        [
            (0, "must be at least 2"),
            (1, "must be at least 2"),
            (-5, "must be at least 2"),
            (0.0, r"must be in \(0, 1\]"),
            (-0.2, r"must be in \(0, 1\]"),
            (1.5, r"must be in \(0, 1\]"),
        ],
    )
    def test_rejects_degenerate_counts(self, bad, message):
        with pytest.raises(ValueError, match=message):
            resolve_anchors(
                9000, consensus_anchors=bad, anchor_threshold=5000, random_state=0
            )
```

- [ ] Step 7: Run, mutate, commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_api.py tests/test_types.py tests/test_plotting.py tests/test_selection.py tests/test_consensus.py tests/test_runner.py tests/test_accuracy.py tests/test_utils.py -q`
Expected: green.

Mutation checks (revert each after): in `src/carve/_consensus.py::reorder_consensus_matrix` return `consensus_matrix, np.arange(n)` instead of the reordered pair; `test_groups_the_interleaved_blocks` must fail. In `src/carve/_plotting.py::plot_cluster_boxplot` ignore `order` (pass `order=None` to `_prepare_cluster_score_groups`); `test_custom_order` must fail.

```bash
git add tests/
git commit -m "test: replace shape and isinstance assertions with the values the code produces"
```

---

### Task 8: Benchmarks-side defects and dead guards

Spec: 1.1 (`test_studies.py`), 1.2 (`test_calibrate.py`, `figures/test_scaling.py`), 1.5 (dead skip guards in `test_anndata.py` and `test_h5ad_roundtrip.py`).

Files:
- Modify: `tests/benchmarks/test_studies.py`, `tests/benchmarks/test_calibrate.py`, `tests/benchmarks/figures/test_scaling.py`, `tests/test_anndata.py`, `tests/test_h5ad_roundtrip.py`

- [ ] Step 1: `test_studies.py`

Rename `TestStudies.test_both_case_studies_are_registered` to `test_every_case_study_is_registered` (same body). Delete `TestNewStudies.test_both_new_studies_are_registered` (a subset of the same fact).

- [ ] Step 2: `test_calibrate.py`

Rename `TestMeanOracleAri.test_returns_a_value_in_the_unit_interval` to `test_returns_a_value_in_the_ari_range` and add a comment above the assertion: `# ARI is bounded below by -1, not 0; the oracle fit can land anywhere in it.`

- [ ] Step 3: `figures/test_scaling.py`

Delete `test_no_element_scale_fudge_is_applied_by_default` (a negative source-text search). Replace it with the property it stood for. `figure_scaling_ari` draws through `_panels.metric_lines`, which scales its markers and lines by `element_scale`; the figure must leave that at its default of 1.0, so the drawn geometry equals the unscaled panel values (`markersize=5.0`, `linewidth=metric_linewidth(metric)`, both verified against the current code). Add `from benchmarks._theme import metric_linewidth` to the imports and remove `inspect` and `_scaling` from them if nothing else uses them.

```python
    def test_marker_and_line_geometry_are_unscaled(self, results):
        """One theme means Fig 4 and this figure match without a fudge factor.

        The old test grepped the module source for a literal scale factor,
        which any other literal would have satisfied. This reads the
        errorbar data lines that were actually drawn: metric_lines multiplies
        markersize 5.0 and the theme linewidth by element_scale, so any
        scale other than 1.0 changes both.
        """
        fig = figure_scaling_ari(results, metrics=METRICS, save=False)
        widths = {m: metric_linewidth(m) for m in METRICS}
        drawn = [
            container[0]
            for ax in fig.get_axes()
            for container in ax.containers
        ]
        assert len(drawn) == len(METRICS) * len(results)
        for line in drawn:
            assert line.get_markersize() == 5.0
            assert line.get_linewidth() in widths.values()
        plt.close(fig)
```

- [ ] Step 4: Dead guards

`tests/test_anndata.py::test_1d_sparse_rejected`: delete the two lines `if arr.ndim == 2:` / `pytest.skip(...)`. Under the `scipy>=1.16` floor `csr_array` of a 1D input is 1D (verified: scipy 1.16.3 gives `ndim == 1`).

`tests/test_h5ad_roundtrip.py::TestZarr::test_zarr_round_trip`: delete `pytest.importorskip("zarr")`; zarr is a hard dependency of `anndata>=0.13`.

- [ ] Step 5: Run and commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks/test_studies.py tests/benchmarks/test_calibrate.py tests/benchmarks/figures/test_scaling.py tests/test_anndata.py tests/test_h5ad_roundtrip.py -q -ra`
Expected: green, and the `-ra` summary shows no skips from these files.

```bash
git add tests/
git commit -m "test: drop duplicate and source-grepping tests; remove skip guards below the dependency floors"
```

---

### Task 9: `tests/test_pl.py`

Spec: Part 2, item 1. `carve.pl` is public and has no test file. `tl.carve` is `CARVE(...).fit(...)` followed by `tl.attach_results`, so one module-scoped fit plus `attach_results` into fresh AnnData objects covers every write-side option without refitting. The conftest `fitted_adata` fixture (unused today) is replaced by this file's own fixtures.

Every assertion below was run against the current code.

Files:
- Create: `tests/test_pl.py`
- Modify: `tests/conftest.py` (delete `fitted_adata`)

- [ ] Step 1: Delete the unused conftest fixture

Remove the `fitted_adata` fixture from `tests/conftest.py`.

- [ ] Step 2: Write `tests/test_pl.py`

```python
"""Tests for carve.pl -- plots that read tl.carve's output out of an AnnData.

One CARVE fit is shared by the module. tl.carve is that fit followed by
tl.attach_results, so attach_results into a fresh AnnData reproduces every
write-side option (key_added, store_results, store_consensus, use_rep,
layer, n_pcs, a pinned k) without refitting.
"""

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.legend import Legend
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import carve
from carve import CARVE

N_OBS = 90


def _X():
    rng = np.random.RandomState(0)
    return np.vstack(
        [rng.randn(30, 6) + 8, rng.randn(30, 6) - 8, rng.randn(30, 6) * 0.4]
    ).astype(np.float32)


def _bare_adata():
    """An AnnData with the representations the fit used and no results."""
    X = _X()
    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = X[:, :4]
    adata.obsm["X_umap"] = X[:, :2]
    adata.layers["counts"] = X * 2.0
    return adata


@pytest.fixture(scope="module")
def model():
    """CARVE fitted on X_pca of _bare_adata()."""
    m = CARVE(
        n_clusters=np.arange(2, 5),
        estimator_param_grids=[(KMeans, {"n_clusters": [2, 3, 4], "n_init": [3]})],
        n_resamples=5,
        random_state=0,
    )
    m.fit(_bare_adata(), use_rep="X_pca")
    return m


@pytest.fixture(scope="module")
def written(model):
    """What tl.carve(adata, use_rep="X_pca") leaves behind."""
    adata = _bare_adata()
    carve.tl.attach_results(adata, model, use_rep="X_pca")
    return adata


def _attach(model, **kwargs):
    """A fresh AnnData with attach_results applied under kwargs."""
    adata = _bare_adata()
    carve.tl.attach_results(adata, model, **kwargs)
    return adata


PL_FUNCTIONS = [
    carve.pl.metric_over_n_clusters,
    carve.pl.consensus_matrix,
    carve.pl.cluster_boxplot,
    carve.pl.cluster_violin,
    carve.pl.cluster_scatter,
    carve.pl.diagnostic_scatter,
]


# -----------------------------------------------------------------------
# Each pl function draws what the corresponding CARVE method draws
# -----------------------------------------------------------------------


class TestMatchesTheModelMethods:
    """pl.* reads the stored artifacts; CARVE.plot_* re-selects from the
    fitted model. With the recorded measure and rule they must agree on the
    drawn data, not merely both return an Axes.
    """

    def test_metric_lines(self, written, model):
        a = carve.pl.metric_over_n_clusters(written)
        b = model.plot_metric_over_n_clusters(measure="stability", rule="1se")
        assert len(a.lines) == len(b.lines) > 0
        for la, lb in zip(a.lines, b.lines):
            np.testing.assert_array_equal(la.get_ydata(), lb.get_ydata())

    def test_consensus_image(self, written, model):
        a = carve.pl.consensus_matrix(written)
        b = model.plot_consensus_matrix()
        np.testing.assert_array_equal(
            np.asarray(a.images[0].get_array()), np.asarray(b.images[0].get_array())
        )

    def test_boxplot_medians(self, written, model):
        def medians(ax):
            return sorted(
                line.get_ydata()[0]
                for line in ax.lines
                if line.get_ydata().size == 2
                and line.get_ydata()[0] == line.get_ydata()[1]
            )

        a = carve.pl.cluster_boxplot(written)
        b = model.plot_cluster_boxplot()
        assert medians(a) == medians(b)
        assert len(medians(a)) >= 2

    def test_violin_collections(self, written, model):
        a = carve.pl.cluster_violin(written)
        b = model.plot_cluster_violin()
        assert len(a.collections) == len(b.collections) > 0

    def test_scatter_offsets(self, written, model):
        a = carve.pl.cluster_scatter(written)
        b = model.plot_cluster_scatter(embedding=written.obsm["X_umap"])
        np.testing.assert_array_equal(
            a.collections[0].get_offsets(), b.collections[0].get_offsets()
        )

    def test_diagnostic_offsets_per_cluster(self, written, model):
        a = carve.pl.diagnostic_scatter(written)
        b = model.plot_diagnostic_scatter(embedding=written.obsm["X_umap"])
        assert len(a.collections) == len(b.collections)
        for ca, cb in zip(a.collections, b.collections):
            np.testing.assert_array_equal(ca.get_offsets(), cb.get_offsets())

    @pytest.mark.parametrize("fn", PL_FUNCTIONS, ids=lambda f: f.__name__)
    def test_save_writes_the_file_and_returns_none(self, written, fn, tmp_path):
        path = tmp_path / f"{fn.__name__}.png"
        assert fn(written, save=path) is None
        assert path.exists()


# -----------------------------------------------------------------------
# key= and the recorded selection
# -----------------------------------------------------------------------


class TestKeys:
    def test_non_default_key(self, model):
        adata = _attach(model, use_rep="X_pca", key_added="other")
        assert "other_stability" in adata.obs
        assert "other_consensus" in adata.obsp
        ax = carve.pl.cluster_boxplot(adata, key="other")
        assert len(ax.lines) > 0
        # Under the default key nothing was written: the score column is
        # the first thing cluster_boxplot reads, so that is what it reports.
        with pytest.raises(KeyError, match=r"adata.obs\['carve_stability'\] not found"):
            carve.pl.cluster_boxplot(adata)
        with pytest.raises(KeyError, match=r"adata.uns\['carve'\] not found"):
            carve.pl.metric_over_n_clusters(adata)

    def test_metric_defaults_to_the_recorded_measure(self, written):
        assert written.uns["carve"]["params"]["measure"] == "stability"
        assert "Stability" in carve.pl.metric_over_n_clusters(written).get_ylabel()
        assert (
            "Generalizability"
            in carve.pl.metric_over_n_clusters(
                written, measure="generalizability"
            ).get_ylabel()
        )

    def test_kwargs_reach_the_errorbar_line(self, written):
        # The data line is the ErrorbarContainer's first artist; the
        # selected-k marker is dashed on its own, so it cannot be the probe.
        dashed = carve.pl.metric_over_n_clusters(written, linestyle="--")
        assert dashed.containers[0][0].get_linestyle() == "--"
        plain = carve.pl.metric_over_n_clusters(written)
        assert plain.containers[0][0].get_linestyle() == "-"


# -----------------------------------------------------------------------
# Error paths in _entry, _labels, _scores, _results
# -----------------------------------------------------------------------


class TestErrors:
    def test_missing_uns_entry(self, written):
        with pytest.raises(KeyError, match=r"adata.uns\['nope'\] not found"):
            carve.pl.metric_over_n_clusters(written, key="nope")

    def test_malformed_uns_entry(self, written):
        bad = written.copy()
        bad.uns["carve"] = {"results": 1}
        with pytest.raises(KeyError, match="does not look like a CARVE result"):
            carve.pl.metric_over_n_clusters(bad)

    def test_missing_labels_column(self, written):
        bad = written.copy()
        del bad.obs["carve"]
        with pytest.raises(KeyError, match=r"adata.obs\['carve'\] not found"):
            carve.pl.consensus_matrix(bad)

    def test_bad_source(self, written):
        with pytest.raises(ValueError, match="source must be one of"):
            carve.pl.cluster_boxplot(written, source="nope")

    def test_missing_score_column_names_the_producing_mode(self, written):
        bad = written.copy()
        del bad.obs["carve_generalizability"]
        with pytest.raises(KeyError, match="source='accuracy' is unavailable"):
            carve.pl.cluster_boxplot(bad, source="accuracy")

    def test_store_results_false(self, model):
        adata = _attach(model, use_rep="X_pca", store_results=False)
        with pytest.raises(KeyError, match="store_results=False"):
            carve.pl.metric_over_n_clusters(adata)
        # The annotation needs the results table too, but a missing table
        # drops the annotation instead of failing the plot.
        ax = carve.pl.cluster_boxplot(adata)
        assert ax.get_legend() is None

    def test_store_consensus_false(self, model):
        adata = _attach(model, use_rep="X_pca", store_consensus=False)
        with pytest.raises(KeyError, match="store_consensus=False"):
            carve.pl.consensus_matrix(adata)

    def test_missing_basis(self, written):
        with pytest.raises(ValueError, match="basis='tsne' not found"):
            carve.pl.cluster_scatter(written, basis="tsne")


# -----------------------------------------------------------------------
# _annotation_text
# -----------------------------------------------------------------------


class TestAnnotation:
    def _legend_texts(self, ax):
        legend = ax.get_legend()
        return [] if legend is None else [t.get_text() for t in legend.get_texts()]

    def test_true_describes_the_recorded_selection(self, written):
        texts = self._legend_texts(carve.pl.cluster_boxplot(written, annotation=True))
        assert len(texts) == 1
        assert "KMeans, n_init=3" in texts[0]
        assert "Stability, 1-SE rule" in texts[0]

    def test_string_is_used_verbatim(self, written):
        assert self._legend_texts(
            carve.pl.cluster_boxplot(written, annotation="hello")
        ) == ["hello"]

    def test_false_draws_no_annotation(self, written):
        assert carve.pl.cluster_boxplot(written, annotation=False).get_legend() is None

    def test_pinned_k_is_reported_as_fixed(self, model):
        adata = _attach(model, use_rep="X_pca", k=3)
        assert adata.uns["carve"]["params"]["pinned"] is True
        texts = self._legend_texts(carve.pl.cluster_boxplot(adata))
        assert "(k = 3, fixed)" in texts[0]

    def test_unmatched_config_id_yields_no_annotation(self, model):
        adata = _attach(model, use_rep="X_pca")
        adata.uns["carve"]["params"]["selected_config_id"] = 99
        assert carve.pl.cluster_boxplot(adata).get_legend() is None

    def test_box_style_on_diagnostic_scatter(self, written):
        ax = carve.pl.diagnostic_scatter(
            written, annotation="note", annotation_style="box"
        )
        legends = [c for c in ax.get_children() if isinstance(c, Legend)]
        assert any(
            [t.get_text() for t in legend.get_texts()] == ["note"] for legend in legends
        )
        assert ax.get_legend().get_title().get_text() == "Cluster"


# -----------------------------------------------------------------------
# _representation: use_rep, layer and n_pcs are reconstructed from params
# -----------------------------------------------------------------------


class TestRepresentation:
    """With no embedding in obsm the scatter falls back to the representation
    the params record, so the drawn coordinates reveal which one was used.
    """

    def _no_obsm(self, model, **kwargs):
        adata = _attach(model, **kwargs)
        del adata.obsm["X_pca"]
        del adata.obsm["X_umap"]
        return adata

    def test_use_rep_x_with_two_pcs_draws_the_columns(self, model):
        adata = self._no_obsm(model, use_rep="X", n_pcs=2)
        ax = carve.pl.cluster_scatter(adata, sort_order=False)
        np.testing.assert_allclose(ax.collections[0].get_offsets(), adata.X[:, :2])

    def test_use_rep_x_with_three_pcs_is_reduced_by_pca(self, model):
        adata = self._no_obsm(model, use_rep="X", n_pcs=3)
        ax = carve.pl.cluster_scatter(adata, sort_order=False)
        expected = PCA(n_components=2, random_state=0).fit_transform(
            np.asarray(adata.X[:, :3], dtype=float)
        )
        np.testing.assert_allclose(ax.collections[0].get_offsets(), expected, rtol=1e-5)

    def test_layer_is_read_instead_of_x(self, model):
        adata = self._no_obsm(model, layer="counts")
        ax = carve.pl.cluster_scatter(adata, sort_order=False)
        expected = PCA(n_components=2, random_state=0).fit_transform(
            np.asarray(adata.layers["counts"], dtype=float)
        )
        np.testing.assert_allclose(ax.collections[0].get_offsets(), expected, rtol=1e-5)
```

- [ ] Step 3: Run the file

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_pl.py -v`
Expected: all pass (32 tests), about 6 s.

- [ ] Step 4: Mutation check

In `src/carve/pl/_plots.py::_representation`, temporarily replace `n_pcs=int(n_pcs) if n_pcs is not None else None` with `n_pcs=None`; `test_use_rep_x_with_two_pcs_draws_the_columns` must fail. In `cluster_boxplot`, temporarily pass `annotation=None` instead of `_annotation_text(...)`; `TestAnnotation.test_true_describes_the_recorded_selection` must fail. Revert both.

- [ ] Step 5: Commit

```bash
git add tests/test_pl.py tests/conftest.py
git commit -m "test(pl): cover carve.pl against the model's own plots, every error path, and representation recovery"
```

---

### Task 10: `plot_diagnostic_scatter` unit tests and a shared anchoring fixture

Spec: Part 2 item 2; 1.5 (the two anchoring tests in `test_plotting.py` fit the same model twice).

Files:
- Modify: `tests/test_plotting.py`

- [ ] Step 1: Add the unit tests

Add `plot_diagnostic_scatter` to the `from carve._plotting import (...)` block and `from matplotlib.legend import Legend` if not already imported. Add after `TestPlotClusterScatter`:

```python
# -----------------------------------------------------------------------
# plot_diagnostic_scatter
# -----------------------------------------------------------------------


class TestPlotDiagnosticScatter:
    def _data(self, n=30, p=4, k=3):
        rng = np.random.RandomState(0)
        return rng.randn(n, p), np.repeat(np.arange(k), n // k), rng.rand(n)

    def test_one_collection_per_cluster(self):
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(X, labels, scores)
        assert len(ax.collections) == 3
        assert sorted(c.get_offsets().shape[0] for c in ax.collections) == [10, 10, 10]

    def test_markers_override_in_label_order(self):
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(X, labels, scores, markers=["o", "s", "^"])
        handles = ax.get_legend().legend_handles
        assert [h.get_marker() for h in handles] == ["o", "s", "^"]

    def test_more_clusters_than_markers_warns_and_cycles(self):
        X, labels, scores = self._data()
        with pytest.warns(UserWarning, match="Markers will cycle"):
            ax = plot_diagnostic_scatter(X, labels, scores, markers=["o", "s"])
        handles = ax.get_legend().legend_handles
        assert [h.get_marker() for h in handles] == ["o", "s", "o"]

    def test_colorbar_label_defaults_to_scores_name(self):
        X, labels, scores = self._data()
        fig, ax = plt.subplots()
        plot_diagnostic_scatter(X, labels, scores, ax=ax, scores_name="Foo")
        cbar_axes = [a for a in fig.axes if a is not ax]
        assert len(cbar_axes) == 1
        assert cbar_axes[0].get_xlabel() == "Foo"

    def test_colorbar_label_override(self):
        X, labels, scores = self._data()
        fig, ax = plt.subplots()
        plot_diagnostic_scatter(
            X, labels, scores, ax=ax, scores_name="Foo", colorbar_label="Bar"
        )
        assert [a for a in fig.axes if a is not ax][0].get_xlabel() == "Bar"

    def test_no_colorbar_adds_no_axes(self):
        X, labels, scores = self._data()
        fig, ax = plt.subplots()
        plot_diagnostic_scatter(X, labels, scores, ax=ax, colorbar=False)
        assert fig.axes == [ax]

    def test_sort_order_draws_high_scores_first(self):
        # alpha encodes the score: alpha_range=(alpha_high, alpha_low), so a
        # high score is transparent. Drawing high scores first means the
        # alphas within a cluster's collection are non-decreasing.
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(X, labels, scores, sort_order=True)
        for c in ax.collections:
            assert np.all(np.diff(c.get_facecolors()[:, 3]) >= -1e-12)
        unsorted = plot_diagnostic_scatter(X, labels, scores, sort_order=False)
        assert any(
            np.any(np.diff(c.get_facecolors()[:, 3]) < 0) for c in unsorted.collections
        )

    def test_alpha_range_bounds_the_facecolors(self):
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(X, labels, scores, alpha_range=(0.4, 0.9))
        alphas = np.vstack([c.get_facecolors() for c in ax.collections])[:, 3]
        assert alphas.min() == pytest.approx(0.4)
        assert alphas.max() == pytest.approx(0.9)

    def test_nan_scores_are_drawn_gray(self):
        X, labels, scores = self._data()
        scores = scores.copy()
        scores[0] = np.nan
        ax = plot_diagnostic_scatter(X, labels, scores)
        facecolors = np.vstack([c.get_facecolors() for c in ax.collections])
        assert np.any(np.all(np.isclose(facecolors, (0.5, 0.5, 0.5, 0.2)), axis=1))

    def test_legend_annotation_prefixes_the_title(self):
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(
            X, labels, scores, annotation="note", annotation_style="legend"
        )
        assert ax.get_legend().get_title().get_text() == "note\nCluster"

    def test_box_annotation_keeps_the_cluster_legend(self):
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(
            X, labels, scores, annotation="note", annotation_style="box"
        )
        legends = [c for c in ax.get_children() if isinstance(c, Legend)]
        assert any(
            [t.get_text() for t in legend.get_texts()] == ["note"] for legend in legends
        )
        assert ax.get_legend().get_title().get_text() == "Cluster"

    def test_save(self, tmp_path):
        X, labels, scores = self._data()
        path = tmp_path / "diagnostic.png"
        assert plot_diagnostic_scatter(X, labels, scores, save=path) is None
        assert path.exists()

    def test_no_finite_scores_raises(self):
        X, labels, _ = self._data()
        with pytest.raises(ValueError, match="No finite scores"):
            plot_diagnostic_scatter(X, labels, np.full(30, np.nan))

    def test_mismatched_lengths_raise(self):
        X, labels, scores = self._data()
        with pytest.raises(ValueError, match="matching n_samples"):
            plot_diagnostic_scatter(X, labels, scores[:-1])
```

- [ ] Step 2: Share the anchored fit between the two anchoring tests

Replace the two module-level anchoring tests at the bottom of the file with:

```python
@pytest.fixture(scope="module")
def anchored():
    """Two blobs fitted under anchoring (threshold 30 of 60 samples)."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(6, 1, (30, 4))])
    c = CARVE(
        estimator_param_grids=[(KMeans, {"n_clusters": [2, 3], "n_init": [10]})],
        n_resamples=6,
        random_state=0,
        anchor_threshold=30,
    )
    with pytest.warns(RuntimeWarning, match="anchored consensus"):
        c.fit(X)
    return X, c


def test_plot_consensus_matrix_under_anchoring(anchored):
    _, c = anchored
    ax = c.plot_consensus_matrix(k=2)
    # The rendered image is the anchor block, not the full 60-by-60 matrix.
    assert np.asarray(ax.images[0].get_array()).shape == (30, 30)


def test_sample_level_plots_work_under_anchoring(anchored):
    X, c = anchored
    # The property the whole feature rests on: consensus matrices shrink to
    # the anchor block, but every per-sample array stays length n.
    assert c.stability_gini_scores_.shape[1] == X.shape[0]
    assert c.stability_ce_scores_.shape[1] == X.shape[0]
    assert c.generalizability_scores_[0].shape == (X.shape[0],)

    # Each of these reads a per-sample score array and calls get_labels;
    # a length mismatch between the two would raise here.
    assert c.plot_cluster_boxplot(k=2) is not None
    assert c.plot_cluster_violin(k=2) is not None
    assert c.plot_cluster_scatter(X=X, k=2) is not None
    assert c.plot_diagnostic_scatter(X=X, k=2) is not None
```

- [ ] Step 3: Run and commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_plotting.py -q`
Expected: green; 14 new tests.

Mutation: in `plot_diagnostic_scatter`, temporarily ignore `sort_order` (delete the `if sort_order:` reordering); `test_sort_order_draws_high_scores_first` must fail. Revert.

```bash
git add tests/test_plotting.py
git commit -m "test(plotting): unit tests for plot_diagnostic_scatter; share the anchored fit"
```

---

### Task 11: The sparse spectral eigensolver path

Spec: Part 2 item 3. Every cluster fixture has `n <= 500`, so the `n >= 1000` `eigsh` branch in `SpectralClustering._spectral_embedding` has never run under test. Measured: two fits at n=1000 take 1.4 s together; the dense fit at n=999 takes 0.15 s.

Files:
- Modify: `tests/test_cluster.py`

- [ ] Step 1: Add the tests

Add to `TestNonConvex`:

```python
    def test_sparse_path_is_reproducible_for_a_fixed_seed(self):
        """n=1000 is the first size that takes the eigsh branch.

        test_regression.py records that ARPACK draws its start vector from
        the global NumPy RNG, so two fits could disagree. Under scipy 1.16
        they do not; if this ever fails, mark it xfail(strict=True) with the
        scipy version in the reason rather than loosening it.
        """
        X, _ = make_moons(1000, noise=0.05, random_state=0)
        first = SpectralClustering(n_clusters=2, random_state=0).fit_predict(X)
        second = SpectralClustering(n_clusters=2, random_state=0).fit_predict(X)
        np.testing.assert_array_equal(first, second)

    def test_sparse_path_recovers_moons(self):
        X, y_true = make_moons(1000, noise=0.05, random_state=0)
        labels = SpectralClustering(n_clusters=2, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y_true, labels) > 0.9

    def test_dense_path_just_below_the_sparse_threshold(self):
        # The boundary is n < 1000 dense, n >= 1000 sparse; both must agree
        # on the planted structure so the switch is invisible to a caller.
        X, y_true = make_moons(999, noise=0.05, random_state=0)
        labels = SpectralClustering(n_clusters=2, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y_true, labels) > 0.9
```

- [ ] Step 2: Prove the branch is taken

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_cluster.py -k "sparse_path or just_below" -v --cov=carve.cluster --cov-report=term-missing`
Expected: pass, and the coverage report no longer lists the `eigsh` line range of `cluster.py` (about lines 255-270) as missing. If the `ArpackNoConvergence` fallback lines remain uncovered that is expected; forcing non-convergence is not attempted here.

- [ ] Step 3: Commit

```bash
git add tests/test_cluster.py
git commit -m "test(cluster): cover the sparse eigensolver path at n=1000"
```

---

### Task 12: Error contracts on the fitted model, and the accuracy-branch defect

Spec: Part 2 item 6. Writing these tests exposed a defect: after a `mode="stability"` fit, `generalizability_scores_` is `[None]` (a list of None per configuration), not `None`, so the four plot methods' `source="accuracy"` branches skip their guard, pass a 0-d NaN array down, and fail with `ValueError: scores must be a 1D array.` instead of the intended `RuntimeError("Generalizability scores are not available for this run.")`. The `gini` and `ce` branches are correct because a generalizability-only fit sets those attributes to `None`.

Files:
- Modify: `src/carve/api.py` (four sites)
- Modify: `tests/test_api.py`

- [ ] Step 1: Write the tests

Add to `tests/test_api.py`, after `TestGetLabels`:

```python
# ---------------------------------------------------------------------------
# Error contracts on a fitted model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def split_mode_fits():
    """The same small fit under mode="stability" and mode="generalizability".

    Module-scoped, so it builds its own data rather than taking the
    function-scoped X_two_clusters fixture.
    """
    rng = np.random.RandomState(42)
    X = np.vstack(
        [rng.randn(30, 5) + [4, 0, 0, 0, 0], rng.randn(30, 5) + [0, 4, 0, 0, 0]]
    )

    def make():
        return CARVE(
            n_clusters=2,
            n_resamples=3,
            subsample_ratio=0.8,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            normalization_options=[],
            dim_reduction_options=[],
            random_state=0,
            verbose=0,
        )

    with pytest.warns(RuntimeWarning, match="experimental"):
        stability_only = make().fit(X, mode="stability")
    with pytest.warns(RuntimeWarning, match="experimental"):
        generalizability_only = make().fit(X, mode="generalizability")
    return stability_only, generalizability_only


SCORE_PLOTS = [
    "plot_cluster_boxplot",
    "plot_cluster_violin",
    "plot_cluster_scatter",
    "plot_diagnostic_scatter",
]


class TestErrorContracts:
    def test_unknown_grid_preset(self, X_two_clusters):
        carve = CARVE(estimator_param_grids="nope", n_resamples=2, verbose=0)
        with pytest.raises(
            ValueError, match="Unknown estimator_param_grids preset 'nope'"
        ):
            carve.fit(X_two_clusters)

    def test_cannot_cut_below_two_clusters(self, fitted_carve):
        with pytest.raises(
            ValueError, match="Cannot cut the consensus matrix at 1 cluster"
        ):
            fitted_carve.get_labels(consensus_k=1)

    def test_get_labels_rejects_an_unknown_mode(self, fitted_carve):
        with pytest.raises(ValueError, match="Unknown mode"):
            fitted_carve.get_labels(mode="nope")

    def test_get_labels_names_the_missing_consensus_matrix(self, split_mode_fits):
        stability_only, generalizability_only = split_mode_fits
        with pytest.raises(
            RuntimeError,
            match="Consensus matrix not available for mode='generalizability'",
        ):
            stability_only.get_labels(mode="generalizability")
        with pytest.raises(
            RuntimeError, match="Consensus matrix not available for mode='default'"
        ):
            generalizability_only.get_labels(measure="generalizability", rule="max")

    def test_plot_consensus_matrix_split_modes(self, split_mode_fits):
        stability_only, generalizability_only = split_mode_fits
        with pytest.raises(ValueError, match="mode must be one of"):
            stability_only.plot_consensus_matrix(mode="nope")
        with pytest.raises(
            RuntimeError,
            match="Selected consensus matrix is not available for mode='default'",
        ):
            generalizability_only.plot_consensus_matrix(
                measure="generalizability", rule="max"
            )
        with pytest.raises(
            RuntimeError,
            match="Selected consensus matrix is not available for "
            "mode='generalizability'",
        ):
            stability_only.plot_consensus_matrix(mode="generalizability")

    @pytest.mark.parametrize("method", SCORE_PLOTS)
    def test_source_and_mode_are_validated(self, fitted_carve, method):
        with pytest.raises(ValueError, match="source must be one of"):
            getattr(fitted_carve, method)(source="nope")
        with pytest.raises(ValueError, match="mode must be one of"):
            getattr(fitted_carve, method)(mode="nope")

    @pytest.mark.parametrize("method", SCORE_PLOTS)
    def test_stability_scores_are_reported_missing_after_a_generalizability_fit(
        self, split_mode_fits, method
    ):
        _, generalizability_only = split_mode_fits
        kwargs = dict(measure="generalizability", rule="max", mode="generalizability")
        with pytest.raises(
            RuntimeError, match="Gini stability scores are not available"
        ):
            getattr(generalizability_only, method)(source="gini", **kwargs)
        with pytest.raises(RuntimeError, match="CE stability scores are not available"):
            getattr(generalizability_only, method)(source="ce", **kwargs)

    @pytest.mark.parametrize("method", SCORE_PLOTS)
    def test_generalizability_scores_are_reported_missing_after_a_stability_fit(
        self, split_mode_fits, method
    ):
        # A stability-only fit leaves generalizability_scores_ as a list of
        # None, one per configuration, not as None. The guard has to look
        # inside the list.
        stability_only, _ = split_mode_fits
        with pytest.raises(
            RuntimeError, match="Generalizability scores are not available"
        ):
            getattr(stability_only, method)(source="accuracy")
```

- [ ] Step 2: Run to see the four expected failures

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_api.py -k "ErrorContracts" -v`
Expected: everything passes except the four `test_generalizability_scores_are_reported_missing_after_a_stability_fit[...]` cases, each failing with `ValueError: scores must be a 1D array.`

- [ ] Step 3: Fix the four `accuracy` branches in `src/carve/api.py`

In `plot_cluster_boxplot`, `plot_cluster_violin`, `plot_cluster_scatter` and `plot_diagnostic_scatter`, the `elif source == "accuracy":` branch reads:

```python
            if self.generalizability_scores_ is None:
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
```

Change the condition at all four sites to:

```python
            if (
                self.generalizability_scores_ is None
                or self.generalizability_scores_[config_id] is None
            ):
```

(A stability-only fit stores one `None` per configuration; see `fit`, where `run_validation` returns the list, and `tl._carve._artifact`, which already makes the same distinction.)

- [ ] Step 4: Run again, lint, commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_api.py -q`
Expected: green.

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format src/`

```bash
git add src/carve/api.py tests/test_api.py
git commit -m "fix(api): report missing generalizability scores after a stability-only fit; pin the error contracts"
```

---

### Task 13: Fit-time validation of `subsample_ratio`, `n_resamples`, `n_trees`

Spec: Part 2 item 10. `subsample_ratio=1.0` yields an empty held-out split and fails somewhere inside the runner; `n_resamples=0` and `n_trees=0` fail in sklearn or produce empty frames. Validation lives in `fit()` (CARVE is a dataclass with sklearn-style verbatim parameter storage; `__post_init__` would reject a constructor a user intends to repair before fitting).

Files:
- Modify: `src/carve/api.py` (`fit`, before `resolve_mode`)
- Modify: `tests/test_api.py`

- [ ] Step 1: Write the failing tests

Add to `tests/test_api.py` in `TestFit` (the class holding `test_stability_mode`):

```python
    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.5, 1.5])
    def test_subsample_ratio_must_be_strictly_between_zero_and_one(
        self, X_two_clusters, ratio
    ):
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            subsample_ratio=ratio,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            verbose=0,
        )
        with pytest.raises(
            ValueError, match=rf"subsample_ratio must be in \(0, 1\), got {ratio}"
        ):
            carve.fit(X_two_clusters)

    @pytest.mark.parametrize("n", [0, -1])
    def test_n_resamples_must_be_positive(self, X_two_clusters, n):
        carve = CARVE(
            n_clusters=2,
            n_resamples=n,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            verbose=0,
        )
        with pytest.raises(ValueError, match=f"n_resamples must be at least 1, got {n}"):
            carve.fit(X_two_clusters)

    @pytest.mark.parametrize("n", [0, -3])
    def test_n_trees_must_be_positive(self, X_two_clusters, n):
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            n_trees=n,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            verbose=0,
        )
        with pytest.raises(ValueError, match=f"n_trees must be at least 1, got {n}"):
            carve.fit(X_two_clusters)

    def test_n_trees_is_not_validated_when_a_classifier_is_given(self, X_two_clusters):
        # n_trees is ignored with a custom classifier (fit() already warns
        # about that), so a nonsensical value must not block the fit.
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            n_trees=0,
            classifier=DummyClassifier(strategy="most_frequent"),
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            normalization_options=[],
            dim_reduction_options=[],
            verbose=0,
        )
        with pytest.warns(RuntimeWarning, match="n_trees is ignored"):
            carve.fit(X_two_clusters)
        assert carve.estimator_results_ is not None
```

- [ ] Step 2: Run to verify they fail

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_api.py -k "subsample_ratio_must or n_resamples_must or n_trees" -v`
Expected: the parametrized cases fail (no ValueError, or a different error from deeper in the runner); the DummyClassifier case passes already.

- [ ] Step 3: Implement

In `CARVE.fit`, as the first statements of the body (before `policy = resolve_mode(mode)`):

```python
        if not 0.0 < float(self.subsample_ratio) < 1.0:
            raise ValueError(
                f"subsample_ratio must be in (0, 1), got {self.subsample_ratio}. "
                "Each resample needs both a training and a held-out split."
            )
        if int(self.n_resamples) < 1:
            raise ValueError(f"n_resamples must be at least 1, got {self.n_resamples}.")
        if self.classifier is None and int(self.n_trees) < 1:
            raise ValueError(f"n_trees must be at least 1, got {self.n_trees}.")
```

Add a `Raises` entry to the `fit` docstring's `ValueError` paragraph: "Also if ``subsample_ratio`` is outside (0, 1), ``n_resamples`` is below 1, or ``n_trees`` is below 1 without a custom classifier."

- [ ] Step 4: Run, lint, commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_api.py tests/test_tl.py -q`
Expected: green (the `tl` defaults are all valid).

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format src/`

```bash
git add src/carve/api.py tests/test_api.py
git commit -m "feat(api): reject subsample_ratio outside (0, 1), n_resamples < 1 and n_trees < 1 at fit time"
```

---

### Task 14: Remaining CARVE-level behavior: preprocessing, progress, `not_two`, HDBSCAN estimator, round trips, reference labels, re-fit

Spec: Part 2 items 4, 5, 7, 8, 9, 11, 12. All expected values below were verified against the current code.

Files:
- Modify: `tests/test_api.py`, `tests/test_utils.py`

- [ ] Step 1: `randomize_preprocessing=True` end to end

Add to `tests/test_api.py` (imports: `from sklearn.decomposition import PCA`, `from sklearn.preprocessing import FunctionTransformer, StandardScaler`):

```python
class TestRandomizedPreprocessing:
    """fit(randomize_preprocessing=True) samples a normalization and a
    reduction per resample and summarizes them in preprocessing_results_.
    Explicit option lists keep the defaults' t-SNE and UMAP out of the test.
    """

    def _make(self):
        return CARVE(
            n_clusters=np.array([2, 3]),
            n_resamples=4,
            estimator_param_grids=[(KMeans, {"n_clusters": [2, 3]})],
            normalization_options=[(FunctionTransformer, {}), (StandardScaler, {})],
            dim_reduction_options=[(FunctionTransformer, {}), (PCA, {"n_components": [2, 3]})],
            random_state=0,
            verbose=0,
        )

    def test_summary_groups_on_the_sweep_parameter(self, X_two_clusters):
        carve = self._make().fit(X_two_clusters, randomize_preprocessing=True)
        df = carve.preprocessing_results_
        assert list(df.columns) == [
            "norm__func",
            "dr__method",
            "n_clusters",
            "ari_stability",
            "ari_generalizability",
        ]
        assert set(df["norm__func"]) <= {"identity", "StandardScaler"}
        assert set(df["dr__method"]) <= {"identity", "PCA"}
        assert set(df["n_clusters"]) == {2, 3}
        # More than one pipeline was actually drawn.
        assert len(set(zip(df["norm__func"], df["dr__method"]))) > 1

    def test_summary_is_reproducible_under_a_seed(self, X_two_clusters):
        a = self._make().fit(X_two_clusters, randomize_preprocessing=True)
        b = self._make().fit(X_two_clusters, randomize_preprocessing=True)
        pd.testing.assert_frame_equal(a.preprocessing_results_, b.preprocessing_results_)

    def test_summary_is_none_without_randomization(self, X_two_clusters):
        assert self._make().fit(X_two_clusters).preprocessing_results_ is None


class TestShowProgress:
    def test_progress_bar_and_per_config_lines(self, X_two_clusters, capsys):
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            subsample_ratio=0.8,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            normalization_options=[],
            dim_reduction_options=[],
            random_state=0,
            verbose=1,
        )
        carve.fit(X_two_clusters, show_progress=True)
        captured = capsys.readouterr()
        # tqdm writes the bar to stderr; the per-configuration line goes
        # through pbar.write, which lands on stdout.
        assert "Grid configs" in captured.err
        assert "1/1" in captured.err
        assert "[CARVE] [1/1] est=KMeans n_clusters=2" in captured.out
```

- [ ] Step 2: `not_two`, `get_estimator` on HDBSCAN, and the inverted round trip

Add a module-scoped fixture next to `fitted_carve_multi_k` and a class after `TestGetEstimator`:

```python
@pytest.fixture(scope="module")
def fitted_two_blobs_k23():
    """Two planted blobs swept over k in {2, 3}: k=2 wins on stability, so
    not_two has to change the answer."""
    rng = np.random.RandomState(42)
    X = np.vstack(
        [rng.randn(30, 5) + [4, 0, 0, 0, 0], rng.randn(30, 5) + [0, 4, 0, 0, 0]]
    )
    carve = CARVE(
        n_clusters=np.array([2, 3]),
        n_resamples=3,
        subsample_ratio=0.8,
        estimator_param_grids=[
            (AgglomerativeClustering, {"n_clusters": [2, 3], "linkage": ["ward"]})
        ],
        normalization_options=[],
        dim_reduction_options=[],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    carve.fit(X)
    return carve


class TestNotTwo:
    def test_two_wins_without_the_flag(self, fitted_two_blobs_k23):
        assert fitted_two_blobs_k23.get_k() == 2

    def test_get_k(self, fitted_two_blobs_k23):
        assert fitted_two_blobs_k23.get_k(not_two=True) == 3

    def test_get_labels(self, fitted_two_blobs_k23):
        labels = fitted_two_blobs_k23.get_labels(not_two=True)
        assert np.unique(labels).size == 3

    def test_get_estimator(self, fitted_two_blobs_k23):
        est = fitted_two_blobs_k23.get_estimator(not_two=True)
        assert isinstance(est, AgglomerativeClustering)
        assert est.n_clusters == 3
        assert est.linkage == "ward"
```

Add to `TestMinClusterSizeMode` (which uses the module-scoped `fitted_hdbscan`; add `from sklearn.cluster import HDBSCAN` to the imports):

```python
    def test_get_estimator_rebuilds_an_hdbscan(self, fitted_hdbscan):
        est = fitted_hdbscan.get_estimator()
        assert isinstance(est, HDBSCAN)
        assert est.min_cluster_size == int(fitted_hdbscan.get_sweep_value())
        assert est.cluster_selection_method == "eom"

    def test_save_load_round_trips_the_inverted_sweep(self, fitted_hdbscan, tmp_path):
        # min_cluster_size is the one registry axis whose rank runs
        # backwards; a rank bug after reload would invert selection.
        path = tmp_path / "hdbscan.carve"
        fitted_hdbscan.save(path)
        loaded = CARVE.load(path)
        assert loaded.sweep_.param == "min_cluster_size"
        assert loaded.sweep_.finer_is_larger is False
        assert [loaded.sweep_.rank_of(v) for v in (3, 5, 8)] == [2, 1, 0]
        assert loaded.get_sweep_value() == fitted_hdbscan.get_sweep_value()
        pd.testing.assert_frame_equal(
            loaded.estimator_results_, fitted_hdbscan.estimator_results_
        )
```

- [ ] Step 3: `get_labels` and `reference_labels`

Add after `TestGetLabels`:

```python
class TestReferenceLabels:
    """get_labels aligns to self.reference_labels when the cluster counts
    match and replaces the reference when they differ or it is unset.
    """

    def _fit(self):
        rng = np.random.RandomState(42)
        X = np.vstack(
            [
                rng.randn(30, 5) + [4, 0, 0, 0, 0],
                rng.randn(30, 5) + [0, 4, 0, 0, 0],
                rng.randn(30, 5) + [0, 0, 4, 0, 0],
            ]
        )
        return CARVE(
            n_clusters=np.array([2, 3]),
            n_resamples=3,
            subsample_ratio=0.8,
            estimator_param_grids=[(KMeans, {"n_clusters": [2, 3]})],
            normalization_options=[],
            dim_reduction_options=[],
            random_state=0,
            verbose=0,
        ).fit(X)

    def test_first_call_seeds_the_reference(self):
        carve = self._fit()
        assert carve.reference_labels is None
        first = carve.get_labels(k=2)
        np.testing.assert_array_equal(carve.reference_labels, first)

    def test_a_different_k_replaces_the_reference(self):
        carve = self._fit()
        first = carve.get_labels(k=2)
        carve.get_labels(k=3)
        assert np.unique(carve.reference_labels).size == 3
        again = carve.get_labels(k=2)
        np.testing.assert_array_equal(again, first)
        np.testing.assert_array_equal(carve.reference_labels, first)

    def test_a_user_reference_with_matching_k_is_aligned_to(self):
        carve = self._fit()
        natural = carve.get_labels(k=3)
        permuted = (natural + 1) % 3
        carve.reference_labels = permuted
        np.testing.assert_array_equal(carve.get_labels(k=3), permuted)

    def test_a_user_reference_with_another_k_is_replaced(self):
        # Current contract: the reference is not preserved across k. A user
        # who passed a k=3 reference and cuts at k=2 gets a k=2 reference back.
        carve = self._fit()
        carve.reference_labels = (carve.get_labels(k=3) + 1) % 3
        two = carve.get_labels(k=2)
        np.testing.assert_array_equal(carve.reference_labels, two)
```

- [ ] Step 4: Re-fit semantics

Add after `TestReferenceLabels`:

```python
class TestRefit:
    @pytest.mark.filterwarnings(
        "ignore:All points in a subsample were labelled as noise:UserWarning"
    )
    def test_second_fit_resets_the_per_run_state(self, X_res_blobs):
        """An anchored min_cluster_size fit followed by an exact k-mode fit
        on the same instance leaves nothing of the first behind."""
        carve = CARVE(
            sweep="min_cluster_size",
            sweep_values=np.array([3, 5, 8]),
            consensus_anchors=30,
            n_resamples=3,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=0,
            verbose=0,
        )
        with pytest.warns(RuntimeWarning, match="anchored consensus"):
            carve.fit(X_res_blobs, reference_labels=np.repeat([0, 1, 2], 40))
        assert carve.consensus_anchors_ is not None
        assert "min_cluster_size" in carve.estimator_results_.columns

        carve.sweep = None
        carve.sweep_values = None
        carve.consensus_anchors = None
        carve.n_clusters = np.array([2, 3])
        carve.estimator_param_grids = [(KMeans, {"n_clusters": [2, 3]})]
        carve.fit(X_res_blobs, reference_labels=np.repeat([1, 0, 2], 40))

        assert carve.consensus_anchors_ is None
        assert carve.sweep_.param == "n_clusters"
        assert "min_cluster_size" not in carve.estimator_results_.columns
        assert carve.preprocessing_results_ is None
        assert carve.consensus_matrices_[0].shape == (120, 120)
        np.testing.assert_array_equal(carve.reference_labels, np.repeat([1, 0, 2], 40))
```

(`fit()` only writes `reference_labels` when the argument is given; a re-fit without it keeps whatever was there. That is deliberately not pinned here.)

- [ ] Step 5: `_summarize_ari_scores` quantiles

In `tests/test_utils.py::TestSummarizeAriScores`, replace `test_normal` and `test_single_value`:

```python
    def test_normal(self):
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        mean, se, q95, q05 = _summarize_ari_scores(scores, 5)
        assert mean == pytest.approx(np.mean(scores))
        assert se == pytest.approx(np.std(scores, ddof=1) / np.sqrt(5))
        assert q95 == pytest.approx(np.quantile(scores, 0.95))
        assert q05 == pytest.approx(np.quantile(scores, 0.05))

    def test_single_value(self):
        mean, se, q95, q05 = _summarize_ari_scores([0.5], 1)
        assert mean == 0.5
        assert np.isnan(se)  # can't compute SE with 1 value
        assert q95 == 0.5
        assert q05 == 0.5
```

- [ ] Step 6: Run and commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_api.py tests/test_utils.py -q`
Expected: green.

Mutation: in `src/carve/_selection.py::select_best_row_by_rule`, temporarily delete the `if not_two:` block; `TestNotTwo.test_get_k` must fail. Revert.

```bash
git add tests/test_api.py tests/test_utils.py
git commit -m "test(api): randomized preprocessing, progress output, not_two, HDBSCAN estimator, reference labels, re-fit"
```

---

### Task 15: Benchmarks gaps -- tutorial notebooks, table run-directory choice, scaling sweep anchors

Spec: Part 2 item 13.

Files:
- Modify: `tests/benchmarks/test_notebooks.py`, `tests/benchmarks/test_tables_cli.py`, `tests/benchmarks/test_studies.py`

- [ ] Step 1: Tutorials in the notebook checks

In `tests/benchmarks/test_notebooks.py`, add to the `NOTEBOOKS` dict:

```python
    "tutorial": REPO_ROOT / "notebooks" / "Tutorial.ipynb",
    "resolution_tutorial": REPO_ROOT / "notebooks" / "Resolution_Tutorial.ipynb",
```

The nightly workflow executes both; every `parametrize("name", sorted(NOTEBOOKS))` check now covers them. Run the file; if a tutorial fails a structural check (for example `add_gridspec` or a `sys.path` edit), that is a real finding: fix the notebook cell source (a cell-source edit, not an execution) and note it in the commit message.

- [ ] Step 2: Which run directory `--tables` uses

In `tests/benchmarks/test_tables_cli.py::TestTablesCli`, add:

```python
    def test_uses_the_lexicographically_last_run_directory(self, tmp_path):
        # The chosen frame's k_star is written into the caption, so two runs
        # with different k_star reveal which one was read.
        root = tmp_path / "runs"
        _write_run_dir(root, "gaussians", "aaaaaaaa", _frame("gaussians").assign(k_star=5))
        _write_run_dir(root, "gaussians", "bbbbbbbb", _frame("gaussians").assign(k_star=7))
        out = tmp_path / "tables"

        with pytest.warns(UserWarning, match="using .*bbbbbbbb"):
            main(["--tables", str(out), "--root", str(root)])

        fragment = (out / f"{TABLE_NAMES['gaussians']}.tex").read_text()
        assert "k^\\star = 7" in fragment
        assert "k^\\star = 5" not in fragment
```

- [ ] Step 3: `study_scaling_sweep` forwards `consensus_anchors`

In `tests/benchmarks/test_studies.py::TestConsensusAnchorsForwarding`, add (with `study_scaling_sweep` imported from `benchmarks._studies` if it is not already):

```python
    def test_study_scaling_sweep_forwards_consensus_anchors(self, blobs, monkeypatch):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)

        frame = study_scaling_sweep(
            X, y, sizes=[len(X)], model_grids=grids, n_resamples=3, consensus_anchors=77
        )

        assert spy.captured_kwargs["consensus_anchors"] == 77
        assert list(frame["n"]) == [len(X)]
```

- [ ] Step 4: Run and commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks/test_notebooks.py tests/benchmarks/test_tables_cli.py tests/benchmarks/test_studies.py -q`
Expected: green.

```bash
git add tests/benchmarks/test_notebooks.py tests/benchmarks/test_tables_cli.py tests/benchmarks/test_studies.py
git commit -m "test(benchmarks): check the tutorials, the --tables run choice, and scaling-sweep anchor forwarding"
```

---

### Task 16: Cache the anchored-accuracy fits

Spec: 1.5. `test_accuracy_improves_with_more_anchors` refits `m=500` and `m=2000`, which the parametrized test already fitted. A module-scoped dictionary of the three anchored fits removes two fits (about 15 s). The `print` calls are dropped: they are invisible without `-s`, and the assertions are the contract. If the supplementary-table numbers are wanted, they belong in a script, not in test output.

Files:
- Modify: `tests/test_anchored_accuracy.py`

- [ ] Step 1: Restructure the fixtures

Replace everything from the `exact` fixture to the end of the file with:

```python
ANCHOR_COUNTS = (500, 1000, 2000)


@pytest.fixture(scope="module")
def exact():
    X, y = _data()
    return X, y, _fit(X, anchors=None)


@pytest.fixture(scope="module")
def anchored_fits(exact):
    """One anchored fit per anchor count, on the exact fixture's data."""
    X, _, _ = exact
    return {m: _fit(X, anchors=m) for m in ANCHOR_COUNTS}


def test_exact_fixture_sits_on_the_default_threshold_boundary(exact):
    # N is exactly the default anchor_threshold, so the exact fixture is a
    # real fit at the boundary rather than one held open by a raised
    # threshold. Everything below is measured against it.
    _, _, exact_carve = exact
    assert exact_carve.anchor_threshold == N
    assert exact_carve.consensus_anchors_ is None
    assert exact_carve.consensus_matrices_[0].shape == (N, N)


@pytest.mark.parametrize("m", ANCHOR_COUNTS)
def test_anchored_tracks_exact(exact, anchored_fits, m):
    X, y, exact_carve = exact
    anchored = anchored_fits[m]

    assert anchored.consensus_anchors_.size == m

    # Selected k agrees.
    assert anchored.get_k(measure="stability", rule="1se") == exact_carve.get_k(
        measure="stability", rule="1se"
    )

    # Per-sample stability scores correlate with the exact ones. Same
    # estimator family and 1/m variance argument for gini and ce, so the
    # same bound applies to both.
    for attr in ("stability_gini_scores_", "stability_ce_scores_"):
        corr = np.corrcoef(getattr(anchored, attr)[0], getattr(exact_carve, attr)[0])[
            0, 1
        ]
        assert corr > 0.8, f"{attr} correlation {corr:.3f} at m={m}"

    # PAC is not compared: under anchoring it is computed over the m-by-m
    # anchor block rather than over all pairs, so it is a different
    # quantity from the exact PAC.

    # Labels agree with each other and both recover the planted structure.
    both = adjusted_rand_score(exact_carve.get_labels(k=3), anchored.get_labels(k=3))
    assert both > 0.9, f"label ARI {both:.3f} at m={m}"
    assert adjusted_rand_score(y, anchored.get_labels(k=3)) > 0.9


def test_accuracy_improves_with_more_anchors(exact, anchored_fits):
    _, _, exact_carve = exact
    reference = exact_carve.stability_gini_scores_[0]
    correlations = [
        np.corrcoef(anchored_fits[m].stability_gini_scores_[0], reference)[0, 1]
        for m in (500, 2000)
    ]
    # Variance of the row-mean estimator falls as 1/m, so more anchors must
    # track the exact answer better. A flat result would mean the anchor
    # count is not actually being honored.
    assert correlations[1] > correlations[0]
```

Also update the module docstring's last sentence to: "The assertions are the contract; the correlations are not printed."

- [ ] Step 2: Run, time, commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_anchored_accuracy.py -q --durations=0`
Expected: 5 passed; the module setup (four fits) is the only slow entry; total under 35 s, down from about 45.

```bash
git add tests/test_anchored_accuracy.py
git commit -m "test: fit each anchored-accuracy model once per module"
```

---

### Task 17: Cache the fits in `tests/benchmarks/test_run.py`

Spec: 1.5 and 3.3 (`RecordingCarve` three times). The identical `run_cell(...)` is repeated in 6 tests and `run_scenario(...)` (2 axes x 2 seeds) in 9, all full CARVE fits. Module-scoped results plus `shutil.copytree` for the tests that delete or corrupt files cut the file from about 240 s to about 100 s without weakening an assertion.

Files:
- Modify: `tests/benchmarks/test_run.py`

Interfaces:
- Produces (module-local fixtures): `tiny_scenario` (module scope), `cell` -> `(rows, runtime)` at `n_resamples=20`, `cell10` and `cell10_timed` -> `(rows, runtime)` at `n_resamples=10` without and with `timing_fits`, `completed` -> `(root, rd)` of a finished `run_scenario` at `n_resamples=20`, `run_copy` -> a private `(root, rd)` copy, `recording_carve` -> namespace with `init_kwargs` and `label_calls` lists.

- [ ] Step 1: Fixtures

Add `import shutil` and `from types import SimpleNamespace` to the imports. Change `tiny_scenario` to `@pytest.fixture(scope="module")` (Scenario is a frozen dataclass; tests derive variants with `dataclasses.replace`). Add after it:

```python
CELL_KWARGS = dict(
    axis_idx=0, axis_value=0, axis_label="easy", seed=0, run_id="r1", random_state=0
)


@pytest.fixture(scope="module")
def cell(tiny_scenario):
    """One easy-axis cell at n_resamples=20; every TestRunCell test reads it."""
    return run_cell(tiny_scenario, n_resamples=20, **CELL_KWARGS)


@pytest.fixture(scope="module")
def completed(tiny_scenario, tmp_path_factory):
    """A finished run_scenario at n_resamples=20: (root, run directory).

    Shared read-only. Tests that delete checkpoints or corrupt the manifest
    take run_copy instead.
    """
    root = tmp_path_factory.mktemp("completed")
    rd = run_scenario(tiny_scenario, root=root, n_resamples=20)
    return root, rd


@pytest.fixture
def run_copy(completed, tmp_path):
    """A private copy of the completed run under this test's tmp_path."""
    root, rd = completed
    new_root = tmp_path / "runs"
    shutil.copytree(root, new_root)
    return new_root, new_root / rd.relative_to(root)


@pytest.fixture
def recording_carve(monkeypatch):
    """Patch benchmarks._run.CARVE with a subclass that records constructor
    kwargs and get_labels calls, then delegates to the real class."""
    import benchmarks._run as run_module

    original = run_module.CARVE
    record = SimpleNamespace(init_kwargs=[], label_calls=[])

    class RecordingCarve(original):
        def __init__(self, *args, **kwargs):
            record.init_kwargs.append(kwargs)
            super().__init__(*args, **kwargs)

        def get_labels(self, **kwargs):
            record.label_calls.append(kwargs)
            return super().get_labels(**kwargs)

    monkeypatch.setattr(run_module, "CARVE", RecordingCarve)
    return record
```

- [ ] Step 2: `TestRunCell`

Rewrite the class body so the five identical-call tests take `cell`:

```python
class TestRunCell:
    def test_emits_one_row_per_metric_and_k(self, tiny_scenario, cell):
        rows, _ = cell
        assert len(rows) == N_METRICS * len(tiny_scenario.candidate_k)

    def test_rows_carry_exactly_the_schema(self, cell):
        rows, _ = cell
        assert set(rows[0]) == set(SCHEMA)

    def test_exactly_one_k_is_selected_per_metric(self, cell):
        rows, _ = cell
        for metric in CARVE_METRICS_ALL + CVI_METRICS:
            selected = [r for r in rows if r["metric_name"] == metric and r["is_selected"]]
            assert len(selected) == 1, metric

    def test_selects_true_k_marks_k_star(self, tiny_scenario, cell):
        rows, _ = cell
        for row in rows:
            assert row["selects_true_k"] == (row["k"] == tiny_scenario.k_star)

    def test_oracle_ari_is_constant_within_a_cell(self, cell):
        rows, _ = cell
        assert len({row["oracle_ari"] for row in rows}) == 1

    def test_generalizability_metrics_use_the_generalizability_matrix(
        self, tiny_scenario, recording_carve
    ):
        """(keep the existing docstring)"""
        run_cell(tiny_scenario, n_resamples=20, **CELL_KWARGS)
        seen_modes = [c.get("mode", "default") for c in recording_carve.label_calls]
        assert "generalizability" in seen_modes
        assert "default" in seen_modes

    def test_is_deterministic_for_a_fixed_seed(self, tiny_scenario, cell):
        first, _ = cell
        second, _ = run_cell(tiny_scenario, n_resamples=20, **CELL_KWARGS)
        assert [r["metric_value"] for r in first] == [r["metric_value"] for r in second]

    def test_carve_receives_the_scenario_n_trees(self, tiny_scenario, recording_carve):
        """(keep the existing docstring)"""
        scenario = dataclasses.replace(tiny_scenario, n_trees=500)
        run_cell(scenario, n_resamples=20, **CELL_KWARGS)
        assert [k.get("n_trees") for k in recording_carve.init_kwargs] == [500]

    def test_provenance_columns_record_the_actual_estimator(self, cell):
        (keep the existing body, reading rows from cell)
```

Keep `test_labels_mode_routes_each_metric_family` unchanged (it does no fit). Where "(keep ...)" appears, carry the current text over; the assertions of `test_carve_receives_the_scenario_n_trees` must match what the current test checks (if it asserts a single recorded value, keep that shape).

- [ ] Step 3: `TestRunScenario`

```python
class TestRunScenario:
    def test_writes_a_complete_run(self, tiny_scenario, completed):
        _, rd = completed
        df = read_run(rd)
        expected = (
            len(tiny_scenario.axis)
            * tiny_scenario.n_seeds
            * N_METRICS
            * len(tiny_scenario.candidate_k)
        )
        assert len(df) == expected

    def test_writes_a_manifest(self, completed):
        _, rd = completed
        assert (rd / "manifest.json").exists()

    def test_resumes_without_recomputing_completed_cells(self, tiny_scenario, run_copy):
        root, rd = run_copy
        before = {p: p.stat().st_mtime_ns for p in rd.glob("cell__*.parquet")}
        run_scenario(tiny_scenario, root=root, n_resamples=20, resume=True)
        after = {p: p.stat().st_mtime_ns for p in rd.glob("cell__*.parquet")}
        assert before == after

    def test_the_same_config_reuses_one_directory(self, tiny_scenario, run_copy):
        root, rd = run_copy
        assert run_scenario(tiny_scenario, root=root, n_resamples=20) == rd

    def test_a_changed_config_gets_a_new_directory(self, tiny_scenario, run_copy):
        root, rd = run_copy
        other = run_scenario(tiny_scenario, root=root, n_seeds=1, n_resamples=21)
        assert other != rd

    def test_n_seeds_override_shortens_the_run(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_seeds=1, n_resamples=20)
        assert len(read_run(rd)["seed"].unique()) == 1

    def test_resuming_a_partial_run_keeps_one_run_id(self, tiny_scenario, run_copy):
        """(keep the existing docstring)"""
        root, rd = run_copy
        sorted(rd.glob("cell__*.parquet"))[0].unlink()
        run_scenario(tiny_scenario, root=root, n_resamples=20, resume=True)
        manifest = json.loads((rd / "manifest.json").read_text())
        assert set(read_run(rd)["run_id"].unique()) == {manifest["run_id"]}

    def test_resume_with_a_corrupt_manifest_warns_and_completes(
        self, tiny_scenario, run_copy
    ):
        """(keep the existing docstring)"""
        root, rd = run_copy
        (rd / "manifest.json").write_text("{not valid json")
        with pytest.warns(UserWarning, match="manifest"):
            run_scenario(tiny_scenario, root=root, n_resamples=20, resume=True)
        assert "run_id" in json.loads((rd / "manifest.json").read_text())

    def test_wall_clock_accumulates_across_a_resume(self, tiny_scenario, run_copy):
        """(keep the existing docstring)"""
        root, rd = run_copy
        first = json.loads((rd / "manifest.json").read_text())
        sorted(rd.glob("cell__*.parquet"))[0].unlink()
        run_scenario(tiny_scenario, root=root, n_resamples=20, resume=True)
        second = json.loads((rd / "manifest.json").read_text())
        assert second["wall_clock_s"] >= first["wall_clock_s"]
```

- [ ] Step 4: `TestRuntimeCapture`

Add two module-scoped fixtures after `_cell` (keep `_cell` and its comment about the seven-resample floor):

```python
@pytest.fixture(scope="module")
def cell10(tiny_scenario):
    return _cell(tiny_scenario)


@pytest.fixture(scope="module")
def cell10_timed(tiny_scenario):
    return _cell(tiny_scenario, timing_fits=True)
```

Then: `test_run_cell_returns_metric_rows_and_a_runtime_row`, `test_runtime_records_the_actual_data_shape`, `test_timing_fits_are_off_by_default` read `rows, runtime = cell10`; `test_timing_fits_populate_both_modes_when_requested` and `test_per_k_runtimes_divide_by_the_candidate_count` read `_, runtime = cell10_timed`; `test_timing_fits_do_not_change_the_metric_rows` compares `cell10[0]` with `cell10_timed[0]`; `test_timed_fits_receive_the_scenario_n_trees` uses `recording_carve` and asserts `[k.get("n_trees") for k in recording_carve.init_kwargs] == [500, 500, 500]`; `test_the_experimental_mode_warning_is_suppressed` keeps its own `_cell(..., timing_fits=True)` call (recwarn only records during the test); `test_run_scenario_writes_a_runtime_row_per_cell` reads `completed` instead of running (`read_runtimes(rd)` has one row per axis point and seed regardless of `n_resamples`).

- [ ] Step 5: Run, time, commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks/test_run.py -q --durations=5`
Expected: same test count, green, under 120 s (was about 240). `grep -c "class RecordingCarve" tests/benchmarks/test_run.py` is 1.

```bash
git add tests/benchmarks/test_run.py
git commit -m "test(benchmarks): fit the tiny scenario once per module and copy the run for destructive tests"
```

---

### Task 18: One `gaussians` run for the CLI tests

Spec: 1.5. `test_runs_a_scenario_end_to_end` and `test_promote_publishes_a_finished_run` each run the full `gaussians` scenario (about 40 s each).

Files:
- Modify: `tests/benchmarks/test_cli.py`

- [ ] Step 1: Share the run

Add `import pytest` and a module-scoped fixture; rewrite the two tests:

```python
@pytest.fixture(scope="module")
def gaussians_run(tmp_path_factory):
    """One reduced gaussians run through the CLI: (exit code, root)."""
    root = tmp_path_factory.mktemp("cli")
    code = main(
        [
            "--scenario",
            "gaussians",
            "--root",
            str(root),
            "--n-seeds",
            "1",
            "--n-resamples",
            "20",
        ]
    )
    return code, root


class TestCli:
    ...

    def test_runs_a_scenario_end_to_end(self, gaussians_run):
        code, root = gaussians_run
        assert code == 0
        assert list(root.glob("gaussians/*/manifest.json"))

    def test_promote_publishes_a_finished_run(self, gaussians_run, tmp_path):
        _, root = gaussians_run
        rd = next((root / "gaussians").iterdir())
        code = main(["--promote", str(rd), "--published-root", str(tmp_path / "pub")])
        assert code == 0
        assert (tmp_path / "pub" / "gaussians" / "results.csv").exists()
```

- [ ] Step 2: Run and commit

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks/test_cli.py -q --durations=3`
Expected: 7 passed; one slow setup entry of about 40 s instead of two.

```bash
git add tests/benchmarks/test_cli.py
git commit -m "test(benchmarks): promote the CLI test's own run instead of running gaussians twice"
```

---

### Task 19: Full verification, order independence, and the handoff note

Files:
- Modify: `/Users/kaiwycik/.claude/projects/-Users-kaiwycik-GitHub-CARVE--claude-playground/memory/carve-test-suite-audit.md` (memory, outside the repo)

- [ ] Step 1: Both suites exactly as CI runs them

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests --ignore=tests/benchmarks -v --tb=short --cov=carve --cov-report=term-missing --cov-fail-under=75`
Expected: all pass, coverage at or above the previous 89.9% (the `pl` module and the sparse branch add to it), 0 warnings.

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks -v --tb=short`
Expected: all pass, 10 skipped (8 regression, 2 rpy2) listed by `-ra`, 0 warnings. Record both wall-clock times.

- [ ] Step 2: One randomized run

Run, in order:

```bash
VIRTUAL_ENV=$PWD/.venv uv pip install pytest-randomly
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests --ignore=tests/benchmarks -q
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/benchmarks -q
VIRTUAL_ENV=$PWD/.venv uv pip uninstall pytest-randomly
```

Expected: both green (pytest-randomly is active as soon as it is installed and prints the seed it used). pytest-randomly shuffles modules, classes and functions but keeps module-scoped fixtures grouped. A failure here is an order dependence: fix it in the test (it is almost always a module-level object mutated by one test and read by another) before finishing. Do not leave pytest-randomly installed.

- [ ] Step 3: Lint `src/`

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format --check src/`
Expected: clean.

- [ ] Step 4: Ruff sanity on the tests (one-off)

Run: `.venv/bin/ruff check tests/ --select E402,F401,F811,I`
Expected: `All checks passed!` (the later tasks added imports; this confirms none regressed Task 6).

- [ ] Step 5: Update the memory note

Rewrite the memory file `carve-test-suite-audit.md` so its body says the audit was implemented on branch `<branch>` (the worktree branch name) with the commits from Tasks 1 to 18, records the two suite wall-clock times from Step 1, and lists what remains deliberately open: the R port is untouched (nanmean needs nothing; the fit-time validation and the accuracy-branch guard have no R mirror yet), and `fit()` without `reference_labels=` keeps a stale reference from an earlier fit. Update its one-line description in `MEMORY.md` accordingly.

- [ ] Step 6: Finish the branch

Invoke `superpowers:finishing-a-development-branch` to decide between merging into `main` and opening a PR against `origin`. The diff touches `src/carve/_consensus.py`, `src/carve/api.py`, `pyproject.toml`, and `tests/` throughout; it is one branch, not several, because every task after Task 2 is verified under the warning policy Task 2 turned on.

---

## Self-review

Spec coverage, section by section:

- 1.1: Task 7 (`test_api` duplicate assertion, `test_types` duplicate), Task 6 (`test_benchmarks_types` duplicate, `test_grids` inner import, `test_theme` unused import via Task 4), Task 8 (`test_studies` rename and duplicate), Task 9 (`fitted_adata`).
- 1.2: Task 7 (`test_get_labels_is_idempotent`, per-call seed, whole-frame reproducibility, `test_no_legend`, anchoring image shape), Task 8 (`test_calibrate` rename, `test_scaling` replacement).
- 1.3: Task 2 (every bare `pytest.warns(RuntimeWarning)` gets `match=`; the two vacuous `test_tl` guards now assert both warnings).
- 1.4: Task 7 (selection `return_idx` and rule tests, reorder tests, `_compute_stability_ari`, field count, accuracy vector, `match=` on the two `test_utils` sites), Task 2 (`test_different_measures`, the `test_tl` exception pin).
- 1.5: Task 16 (anchored accuracy), Task 10 (plotting anchoring fixture), Task 17 (`test_run`), Task 18 (`test_cli`), Task 8 (dead guards), Task 2 (`slow` marker removed).
- 2.1: Task 9. 2.2: Task 10. 2.3: Task 11. 2.4 and 2.5: Task 14. 2.6: Task 12. 2.7, 2.8, 2.9, 2.11, 2.12: Task 14. 2.10: Task 13. 2.13: Task 15.
- 3.1: Tasks 1 and 2. 3.2: Task 4 (`rc_context`, leak probe), Task 5 (spies, `monkeypatch`). 3.3: Task 5 (spies, stub), Task 4 (backend, `close_figures`, `ax` fixture), Task 17 (`RecordingCarve`). 3.4: Task 3 (packages, no `conftest` import, marker), Task 6 (mid-file and duplicated imports, ruff pass). 3.5: Task 2 (`pyyaml`). 3.6: Task 2 (`-ra`, `xfail_strict`, `filterwarnings`, marker removal), Task 3 (`requires_graph` registered), Task 2 and Task 19 (xdist declared, one randomized run). 3.7: Task 6.

Placeholder scan: the only "(keep the existing ...)" markers are in Task 17 and refer to docstrings and one test body that exist today in the file being edited; the executor carries them over unchanged. Every other step shows its code.

Type consistency: `with_sweep_cols` keeps its signature across Tasks 3 and later imports. `make_njobs_spy`/`make_seed_spy`/`make_parallel_spy` (Task 5) return classes with `seen: list`; Task 5's rewritten tests read `spy.seen`. `StubCarve(results, select, labels)` (Task 5) is what Task 5's five call sites construct. `make_carve_spy` (Task 5) exposes `captured_kwargs`, used in Tasks 5 and 15; its instances have `fit`, `save`, `get_labels`, `get_k`, `estimator_results_`, which is what `study_scaling_sweep` (Task 15) calls. `split_mode_fits` (Task 12) is only used inside Task 12. `cell`, `completed`, `run_copy`, `recording_carve`, `cell10`, `cell10_timed`, `CELL_KWARGS` (Task 17) are all module-local to `test_run.py`. `_row_nanmean` (Task 1) is private and imported only by `test_consensus.py`.
