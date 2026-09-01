# Benchmarks Reporting Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the presentation half of the benchmarks package so every manuscript figure is produced by a named function from committed artifacts, under one theme, with no notebook doing layout work.

**Architecture:** A single `_theme.py` owns the palette, rcParams, and figure geometry. `_panels.py` holds drawing primitives that all take an `ax` and return it. `figures/` has one module per manuscript figure, each exposing `figure_<manuscript_name>()` that returns a `Figure` and saves under the manuscript filename at 300 dpi. `datasets/` provides loaders returning `(X, y, meta)`. Notebooks become thin drivers.

**Tech Stack:** Python 3.12+, matplotlib, numpy, pandas, scikit-learn, scanpy, anndata, rpy2 (Levine only), pytest, ruff 0.16.4.

**Spec:** `docs/superpowers/specs/2026-09-01-benchmarks-rebuild-design.md`

**Prerequisite:** `docs/superpowers/plans/2026-09-01-benchmarks-compute-foundation.md` must be complete. This plan imports `benchmarks._artifacts`, `benchmarks._registry`, `benchmarks._tables`, and `benchmarks._run` from it.

## Gap found while planning

The spec's artifact schema has no runtime columns, but `paper_fig_scaling_runtime_k5.png` is a
manuscript figure (`CARVE_manuscript.tex:1476`). The old `benchmark_scaling` returned
`(scores_df, runtimes_df)` and the unified schema kept only the first. Task 1 adds a runtime
sidecar to the compute layer before anything tries to plot from it.

The published figure carries two curves, not one. `plot_runtime_over_axis` defaults to
`runtime_cols=("t_carve_sec_s", "t_carve_sec_g")`, labeled "CARVE Stability" and "CARVE
Generalizability", and the notebook does not override them. The old scaling runner could produce
two numbers because it fit two CARVE objects, one per mode, and timed each. The rebuilt runner
does a single default-mode fit, which computes both consensus matrices in one pass and therefore
yields one timing.

Decision taken: reproduce the published figure exactly by running two extra mode-specific fits
whose only purpose is timing. Their clustering results are discarded, so this cannot reintroduce
the `ari_at_k` mode bug that plan 1 fixed. The cost is real, so it is opt-in per scenario through
`TIMED_SCENARIOS` in the registry, and only the two scaling scenarios carry it.

Two consequences to keep in view. `carve.fit` warns
`"Non-default mode is experimental and may break downstream functionality."` (`api.py:310`) on
every mode-specific fit, so the published runtimes come from a path the package itself flags as
experimental; the runner suppresses that one warning deliberately and says why in a comment. And
the two timed fits together cost more than a user actually pays, since one default fit yields
both measures. `t_default_s` is recorded alongside them so the honest single-fit number is
available without a re-run.

A second, smaller correction: the spec says "nine paper figures". There are ten
`\includegraphics` references, of which two (`clustering_problem.png`, `CARVE_schema.png`) are
hand-drawn schematics with no generating code. Eight figures are code-generated, and this plan
covers all eight.

## Global Constraints

- Python `>=3.12`. Ruff pinned to `==0.16.4` from the `dev` extra; lint set exactly `["E4", "E7", "E9", "F", "I", "UP"]`.
- All commands run from `/Users/kaiwycik/GitHub/CARVE/code` using `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`.
- No `logging`. Diagnostics use `warnings.warn(..., stacklevel=2)` and `verbose`-gated `print()`. Progress uses `tqdm.auto`.
- `_types.py` remains a leaf. Nothing under `_*` imports `benchmarks.run`.
- No library code calls `plt.show()`. Every panel primitive takes `ax` as its first parameter and returns it. Every figure function returns a `Figure`.
- Figures save at `dpi=300` with `bbox_inches="tight"` under their exact manuscript filenames.
- American spelling in prose. No bold or italics in authored files, including docstrings and Markdown.
- Tests set the Agg backend before importing pyplot: `matplotlib.use("Agg")`.
- `code/vis/` is gitignored. Copying figures into `overleaf/vis/` stays manual, per the existing convention. Never write to `../overleaf/`.

## Manuscript figure inventory

| Function | Output file | Destination | Manuscript |
|---|---|---|---|
| `figure_benchmarking_examples()` | `benchmarking_examples.png` | `vis/benchmarking/` | S1 Fig |
| `figure_benchmarking_results()` | `benchmarking_results.png` | `vis/benchmarking/` | Fig 4 |
| `figure_scaling_ari()` | `paper_fig_scaling_ari_k5.png` | `vis/benchmarking/` | scaling ARI |
| `figure_scaling_runtime()` | `paper_fig_scaling_runtime_k5.png` | `vis/benchmarking/` | scaling runtime |
| `figure_klein_results()` | `klein_results.png` | `vis/case_studies/` | Fig 5 |
| `figure_levine_results()` | `levine_results.png` | `vis/case_studies/` | Fig 6 |
| `figure_carve_output_klein()` | `CARVE_output_klein.png` | `vis/case_studies/` | Fig 3 |
| `figure_carve_output_levine()` | `CARVE_output_levine.png` | `vis/case_studies/` | S4 Fig |

---

### Task 1: Runtime sidecar in the compute layer

**Files:**
- Modify: `src/benchmarks/_artifacts.py`
- Modify: `src/benchmarks/_run.py`
- Modify: `src/benchmarks/_registry.py`
- Modify: `tests/benchmarks/test_artifacts.py`
- Modify: `tests/benchmarks/test_run.py`

**Interfaces:**
- Consumes: `run_dir`, `read_run` from plan 1 Task 9; `run_cell`, `run_scenario` from plan 1 Task 10.
- Produces:
  - `RUNTIME_SCHEMA: tuple[str, ...]` = `("run_id", "scenario", "axis_name", "axis_value", "axis_label", "seed", "n_samples", "n_features", "n_resamples", "n_jobs", "estimator", "t_default_s", "t_stability_s", "t_generalizability_s", "t_per_k_stability_s", "t_per_k_generalizability_s")`
  - `write_runtime_checkpoint(rd: Path, axis_label: str, seed: int, rows: list[dict]) -> Path`
  - `read_runtimes(rd: Path) -> pd.DataFrame`
  - `TIMED_SCENARIOS: frozenset[str]` in `_registry.py`, equal to `{"gaussians_samples", "gaussians_dimensionality"}`
  - `run_cell(..., timing_fits: bool = False) -> tuple[list[dict], dict]` — now returns `(metric_rows, runtime_row)`
  - `run_scenario(..., timing_fits: bool | None = None)` — `None` consults `TIMED_SCENARIOS`

Column names spell the mode out. The old names were `t_carve_sec_s` and `t_carve_sec_g`, where
`sec` meant seconds and the trailing `_s` meant stability, so `_s` carried two meanings in one
name. Here `_s` is only ever seconds.

This changes `run_cell`'s return type and adds a parameter. `run_scenario` is the only caller
inside the package; update it and the eight tests that bind a single name in the same task, so no
commit leaves the suite red.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarks/test_artifacts.py`:

```python
from benchmarks._artifacts import (
    RUNTIME_SCHEMA,
    read_runtimes,
    write_runtime_checkpoint,
)


def _runtime_row(**overrides):
    row = {
        "run_id": "r1",
        "scenario": "demo",
        "axis_name": "difficulty_level",
        "axis_value": 0,
        "axis_label": "easy",
        "seed": 0,
        "n_samples": 120,
        "n_features": 4,
        "n_resamples": 3,
        "n_jobs": 1,
        "estimator": "kmeans",
        "t_default_s": 1.25,
        "t_stability_s": 0.80,
        "t_generalizability_s": 0.95,
        "t_per_k_stability_s": 0.16,
        "t_per_k_generalizability_s": 0.19,
    }
    row.update(overrides)
    return row


class TestRuntimeSidecar:
    def test_schema_is_the_documented_contract(self):
        assert RUNTIME_SCHEMA == (
            "run_id",
            "scenario",
            "axis_name",
            "axis_value",
            "axis_label",
            "seed",
            "n_samples",
            "n_features",
            "n_resamples",
            "n_jobs",
            "estimator",
            "t_default_s",
            "t_stability_s",
            "t_generalizability_s",
            "t_per_k_stability_s",
            "t_per_k_generalizability_s",
        )

    def test_round_trips_through_parquet(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        write_runtime_checkpoint(rd, "easy", 0, [_runtime_row()])
        df = read_runtimes(rd)
        assert len(df) == 1
        assert list(df.columns) == list(RUNTIME_SCHEMA)

    def test_runtime_files_do_not_pollute_the_metric_run(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row()])
        write_runtime_checkpoint(rd, "easy", 0, [_runtime_row()])
        assert len(read_run(rd)) == 1
        assert len(read_runtimes(rd)) == 1

    def test_an_empty_run_returns_an_empty_runtime_frame(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        assert read_runtimes(rd).empty

    def test_rejects_rows_outside_the_runtime_schema(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        bad = _runtime_row()
        del bad["t_stability_s"]
        with pytest.raises(ValueError, match="t_stability_s"):
            write_runtime_checkpoint(rd, "easy", 0, [bad])
```

Append to `tests/benchmarks/test_run.py`:

```python
def _cell(scenario, **overrides):
    kwargs = dict(
        axis_idx=0,
        axis_value=0,
        axis_label="easy",
        seed=0,
        run_id="r1",
        random_state=0,
        n_resamples=3,
    )
    kwargs.update(overrides)
    return run_cell(scenario, **kwargs)


class TestRuntimeCapture:
    def test_run_cell_returns_metric_rows_and_a_runtime_row(self, tiny_scenario):
        rows, runtime = _cell(tiny_scenario)
        assert isinstance(rows, list)
        assert isinstance(runtime, dict)
        assert runtime["t_default_s"] > 0.0

    def test_runtime_records_the_actual_data_shape(self, tiny_scenario):
        _, runtime = _cell(tiny_scenario)
        assert runtime["n_samples"] == 120
        assert runtime["n_features"] == 4

    def test_timing_fits_are_off_by_default(self, tiny_scenario):
        _, runtime = _cell(tiny_scenario)
        assert np.isnan(runtime["t_stability_s"])
        assert np.isnan(runtime["t_generalizability_s"])

    def test_timing_fits_populate_both_modes_when_requested(self, tiny_scenario):
        _, runtime = _cell(tiny_scenario, timing_fits=True)
        assert runtime["t_stability_s"] > 0.0
        assert runtime["t_generalizability_s"] > 0.0

    def test_per_k_runtimes_divide_by_the_candidate_count(self, tiny_scenario):
        _, runtime = _cell(tiny_scenario, timing_fits=True)
        n_k = len(tiny_scenario.candidate_k)
        assert runtime["t_per_k_stability_s"] == pytest.approx(
            runtime["t_stability_s"] / n_k
        )
        assert runtime["t_per_k_generalizability_s"] == pytest.approx(
            runtime["t_generalizability_s"] / n_k
        )

    def test_timing_fits_do_not_change_the_metric_rows(self, tiny_scenario):
        """The timed fits are discarded; only the default fit feeds metrics."""
        without, _ = _cell(tiny_scenario)
        with_timing, _ = _cell(tiny_scenario, timing_fits=True)
        assert [r["metric_value"] for r in without] == [
            r["metric_value"] for r in with_timing
        ]
        assert [r["ari_at_k"] for r in without] == [
            r["ari_at_k"] for r in with_timing
        ]

    def test_the_experimental_mode_warning_is_suppressed(self, tiny_scenario, recwarn):
        _cell(tiny_scenario, timing_fits=True)
        messages = [str(w.message) for w in recwarn]
        assert not any("Non-default mode is experimental" in m for m in messages)

    def test_run_scenario_writes_a_runtime_row_per_cell(self, tiny_scenario, tmp_path):
        from benchmarks._artifacts import read_runtimes

        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=3)
        runtimes = read_runtimes(rd)
        assert len(runtimes) == len(tiny_scenario.axis) * tiny_scenario.n_seeds

    def test_scaling_scenarios_are_timed_by_default(self):
        from benchmarks._registry import SCENARIOS, TIMED_SCENARIOS

        assert TIMED_SCENARIOS == {"gaussians_samples", "gaussians_dimensionality"}
        assert TIMED_SCENARIOS <= set(SCENARIOS)

    def test_difficulty_scenarios_are_not_timed_by_default(self):
        from benchmarks._registry import TIMED_SCENARIOS

        assert "gaussians" not in TIMED_SCENARIOS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/benchmarks/test_artifacts.py -k Runtime tests/benchmarks/test_run.py -k Runtime -v`
Expected: FAIL with `ImportError: cannot import name 'RUNTIME_SCHEMA'`

- [ ] **Step 3: Add the sidecar to `_artifacts.py`**

Add below the existing `SCHEMA` definition:

```python
RUNTIME_SCHEMA: tuple[str, ...] = (
    "run_id",
    "scenario",
    "axis_name",
    "axis_value",
    "axis_label",
    "seed",
    "n_samples",
    "n_features",
    "n_resamples",
    "n_jobs",
    "estimator",
    # The default-mode fit is the one that produced the metric rows. The two
    # mode-specific timings come from extra fits whose results are discarded;
    # they exist only to reproduce the published two-curve runtime figure.
    "t_default_s",
    "t_stability_s",
    "t_generalizability_s",
    "t_per_k_stability_s",
    "t_per_k_generalizability_s",
)
```

Add these functions after `read_run`:

```python
def _validate_against(frame: pd.DataFrame, schema: tuple[str, ...]) -> pd.DataFrame:
    """Check a frame carries exactly the schema columns, then order them."""
    missing = [column for column in schema if column not in frame.columns]
    if missing:
        raise ValueError(f"Rows are missing schema columns: {missing}.")
    extra = [column for column in frame.columns if column not in schema]
    if extra:
        raise ValueError(f"Rows carry columns outside the schema: {extra}.")
    return frame[list(schema)]


def write_runtime_checkpoint(
    rd: Path, axis_label: str, seed: int, rows: list[dict[str, Any]]
) -> Path:
    """Write one cell's timing row to the runtime sidecar.

    Timing lives beside the metric table rather than inside it: a runtime is
    one value per cell, while the metric table has one row per metric and k,
    so folding them together would repeat the same number a hundred times.
    """
    path = Path(rd) / f"runtime__{axis_label}__{seed:04d}.parquet"
    _validate_against(pd.DataFrame(rows), RUNTIME_SCHEMA).to_parquet(path, index=False)
    return path


def read_runtimes(rd: Path) -> pd.DataFrame:
    """Concatenate every runtime checkpoint in a run directory."""
    paths = sorted(Path(rd).glob("runtime__*.parquet"))
    if not paths:
        return pd.DataFrame(columns=list(RUNTIME_SCHEMA))
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return frame[list(RUNTIME_SCHEMA)]
```

Refactor the existing `write_checkpoint` to reuse the validator, replacing its inline
`missing`/`extra` block with:

```python
    path = _checkpoint_path(rd, axis_label, seed)
    _validate_against(pd.DataFrame(rows), SCHEMA).to_parquet(path, index=False)
    return path
```

- [ ] **Step 4: Register which scenarios get timing fits**

Add to `src/benchmarks/_registry.py`, below `SCENARIOS`:

```python
# Scenarios that additionally run two mode-specific CARVE fits purely to time
# them. This reproduces the published two-curve runtime figure, which was
# produced by a runner that fit each mode separately. The extra fits more than
# double a cell's cost, so only the scaling scenarios carry them.
TIMED_SCENARIOS: frozenset[str] = frozenset(
    {"gaussians_samples", "gaussians_dimensionality"}
)
```

- [ ] **Step 5: Capture timing in `_run.py`**

Add `import time` and `import warnings` if they are not already imported, and add
`timing_fits: bool = False` to `run_cell`'s keyword-only parameters. Wrap the existing fit:

```python
    t0 = time.perf_counter()
    carve.fit(X)
    t_default_s = time.perf_counter() - t0
```

After the metric rows are built and before the `return`, add the timed fits and the runtime row:

```python
    t_stability_s = float("nan")
    t_generalizability_s = float("nan")

    if timing_fits:
        # Two extra fits that exist only to be timed. Their results are never
        # read, so this cannot reintroduce the ari_at_k mode bug: every metric
        # row above comes from the default-mode fit.
        #
        # carve.fit warns that non-default modes are experimental (api.py:310).
        # Exercising that path is deliberate here, because it is what produced
        # the published runtime figure, so the warning is suppressed rather
        # than raised sixty times per scenario.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Non-default mode is experimental",
                category=RuntimeWarning,
            )
            for mode in ("stability", "generalizability"):
                timer = CARVE(
                    estimator_param_grids=grids,
                    n_resamples=n_resamples,
                    n_jobs=1,
                    random_state=benchmark_seed,
                )
                t0 = time.perf_counter()
                timer.fit(X, mode=mode)
                elapsed = time.perf_counter() - t0
                if mode == "stability":
                    t_stability_s = elapsed
                else:
                    t_generalizability_s = elapsed

    n_k = len(candidate_k)
    runtime_row = {
        "run_id": run_id,
        "scenario": scenario.name,
        "axis_name": scenario.axis.name,
        "axis_value": axis_value,
        "axis_label": axis_label,
        "seed": seed,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_resamples": int(n_resamples),
        "n_jobs": 1,
        "estimator": scenario.estimator.name,
        "t_default_s": float(t_default_s),
        "t_stability_s": float(t_stability_s),
        "t_generalizability_s": float(t_generalizability_s),
        "t_per_k_stability_s": float(t_stability_s / n_k),
        "t_per_k_generalizability_s": float(t_generalizability_s / n_k),
    }

    return rows, runtime_row
```

Dividing `nan` by `n_k` yields `nan`, so the per-k columns are absent exactly when the timed fits
did not run. That is intentional: an untimed cell records no timing rather than a misleading zero.

`n_jobs` is recorded as 1 because CARVE is always constructed with `n_jobs=1`; the runner's own
parallelism is in the manifest.

Update the docstring's Returns section to say it returns `(metric_rows, runtime_row)`, and
document `timing_fits`.

In `run_scenario`, add `timing_fits: bool | None = None` to the signature, resolve it, and update
`_one`:

```python
    from ._registry import TIMED_SCENARIOS

    if timing_fits is None:
        timing_fits = scenario.name in TIMED_SCENARIOS

    def _one(axis_idx, axis_value, axis_label, seed):
        rows, runtime_row = run_cell(
            scenario,
            axis_idx=axis_idx,
            axis_value=axis_value,
            axis_label=axis_label,
            seed=seed,
            run_id=run_id,
            random_state=random_state,
            n_resamples=n_resamples,
            timing_fits=timing_fits,
        )
        write_checkpoint(rd, axis_label, seed, rows)
        write_runtime_checkpoint(rd, axis_label, seed, [runtime_row])
```

Add `write_runtime_checkpoint` to the `._artifacts` import block in `_run.py`, and record
`timing_fits` in the manifest by passing it through to `build_manifest`'s `config` dict, so an
artifact says whether its runtimes were collected.

- [ ] **Step 6: Update the existing run tests for the new return type**

Every test in `TestRunCell` that calls `run_cell` and binds one name must now unpack two. Change
each `rows = run_cell(...)` to `rows, _ = run_cell(...)`. There are seven such call sites in
`TestRunCell` plus one in `test_generalizability_metrics_use_the_generalizability_matrix`, which
discards the result entirely and needs no change.

- [ ] **Step 7: Extend `promote()` to carry runtimes**

In `_artifacts.promote`, after the results CSV is written, add:

```python
    runtimes = read_runtimes(rd)
    if not runtimes.empty:
        runtimes.to_csv(out / "runtimes.csv", index=False)
```

- [ ] **Step 8: Run the full compute suite**

Run: `.venv/bin/pytest tests/benchmarks/ -v --tb=short -k "not regression"`
Expected: all passed

- [ ] **Step 9: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_artifacts.py src/benchmarks/_run.py src/benchmarks/_registry.py tests/benchmarks/test_artifacts.py tests/benchmarks/test_run.py
git commit -m "feat(benchmarks): capture per-cell CARVE runtime in a sidecar artifact"
```

---

### Task 2: The theme

**Files:**
- Create: `src/benchmarks/_theme.py`
- Create: `tests/benchmarks/test_theme.py`

**Interfaces:**
- Consumes: `METRIC_DISPLAY_NAMES` from plan 1 Task 6.
- Produces:
  - `METRIC_COLORS: dict[str, str]`
  - `CLUSTER_PALETTE: tuple[str, ...]`
  - `RC_PARAMS: dict[str, Any]`
  - `FONT_SIZES: dict[str, float]`
  - `apply_theme() -> None`
  - `theme_context()` — a context manager applying the theme and restoring afterwards
  - `cluster_colors(n: int) -> list[str]`
  - `metric_color(metric: str) -> str`
  - `style_axes(ax, *, grid: bool = True) -> Axes`
  - `save_figure(fig, path: Path, *, dpi: int = 300) -> Path`

There are currently four uncoordinated palettes. `benchmarking_plotting.py` defines
`cluster_pallette` (Dark2), `lines_pallette_contrastive_carve` starting `#00CD6C`, and
`lines_pallette_contrastive_other`; `benchmarking_config.py` defines `METRIC_COLOR` with
`ari_stability_1se` at `#009E73`; `case_study_plotting.py` defines `CARVE_GREEN = "#009E73"` and
`CLUSTER_PALETTE_NAME = "tab20"`. The result is that CARVE stability is `#00CD6C` in Fig 4 and
`#009E73` in Fig 5, the same method in two greens in one paper.

Resolve to `#009E73`, the Okabe-Ito bluish green: it is colorblind-safe, and it is already what
two of the three definitions use. Fig 4's green shifts slightly as a result; that is the fix
landing, not a regression.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_theme.py`:

```python
"""Tests for the single source of figure styling."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from benchmarks._registry import CARVE_METRICS_ALL, CVI_METRICS
from benchmarks._theme import (
    CLUSTER_PALETTE,
    FONT_SIZES,
    METRIC_COLORS,
    RC_PARAMS,
    apply_theme,
    cluster_colors,
    metric_color,
    save_figure,
    style_axes,
    theme_context,
)

PLOTTED_METRICS = (
    "ari_stability_1se",
    "ari_generalizability_1se",
    "ari_average_1se",
    *CVI_METRICS,
)


class TestPalette:
    def test_every_plotted_metric_has_a_color(self):
        for metric in PLOTTED_METRICS:
            assert metric in METRIC_COLORS

    def test_carve_stability_is_the_okabe_ito_green_everywhere(self):
        assert METRIC_COLORS["ari_stability_1se"] == "#009E73"

    def test_no_two_plotted_metrics_share_a_color(self):
        colors = [METRIC_COLORS[m] for m in PLOTTED_METRICS]
        assert len(set(colors)) == len(colors)

    def test_metric_color_falls_back_without_raising(self):
        assert metric_color("not_a_metric").startswith("#")

    def test_every_carve_metric_resolves_to_a_color(self):
        for metric in CARVE_METRICS_ALL:
            assert metric_color(metric).startswith("#")

    def test_cluster_colors_cycle_beyond_the_palette(self):
        assert len(cluster_colors(3)) == 3
        assert len(cluster_colors(len(CLUSTER_PALETTE) + 5)) == len(CLUSTER_PALETTE) + 5

    def test_cluster_colors_are_stable(self):
        assert cluster_colors(4) == cluster_colors(4)


class TestRcParams:
    def test_pins_pdf_fonttype_42_for_editable_text(self):
        assert RC_PARAMS["pdf.fonttype"] == 42
        assert RC_PARAMS["ps.fonttype"] == 42

    def test_saves_at_publication_resolution(self):
        assert RC_PARAMS["savefig.dpi"] == 300

    def test_hides_the_top_and_right_spines(self):
        assert RC_PARAMS["axes.spines.top"] is False
        assert RC_PARAMS["axes.spines.right"] is False

    def test_apply_theme_actually_mutates_rcparams(self):
        plt.rcParams["pdf.fonttype"] = 3
        apply_theme()
        assert plt.rcParams["pdf.fonttype"] == 42

    def test_theme_context_restores_previous_state(self):
        plt.rcParams["pdf.fonttype"] = 3
        with theme_context():
            assert plt.rcParams["pdf.fonttype"] == 42
        assert plt.rcParams["pdf.fonttype"] == 3


class TestFontSizes:
    def test_named_sizes_replace_inline_literals(self):
        for key in ("panel_letter", "axis_label", "tick", "legend", "title"):
            assert key in FONT_SIZES

    def test_sizes_are_ordered_sensibly(self):
        assert FONT_SIZES["tick"] <= FONT_SIZES["axis_label"] <= FONT_SIZES["title"]
        assert FONT_SIZES["panel_letter"] > FONT_SIZES["title"]


class TestStyleAxes:
    def test_returns_the_same_axes_object(self):
        fig, ax = plt.subplots()
        assert style_axes(ax) is ax
        plt.close(fig)

    def test_removes_the_top_and_right_spines(self):
        fig, ax = plt.subplots()
        style_axes(ax)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close(fig)

    def test_grid_can_be_disabled(self):
        fig, ax = plt.subplots()
        style_axes(ax, grid=False)
        assert not ax.xaxis._major_tick_kw.get("gridOn", False)
        plt.close(fig)


class TestSaveFigure:
    def test_writes_the_file_and_returns_its_path(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = save_figure(fig, tmp_path / "sub" / "demo.png")
        assert path.exists()
        assert path.name == "demo.png"
        plt.close(fig)

    def test_creates_missing_parent_directories(self, tmp_path):
        fig, _ = plt.subplots()
        save_figure(fig, tmp_path / "a" / "b" / "c.png")
        assert (tmp_path / "a" / "b").is_dir()
        plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_theme.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._theme'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_theme.py`:

```python
"""The single source of figure styling.

Everything that decides how a figure looks lives here: palette, rcParams,
spine treatment, font sizes, and saving. The code this replaces had four
uncoordinated palettes, so CARVE stability was drawn in two different greens
in the same paper, and a set_paper_style() that nothing ever called, so every
published figure used stock matplotlib defaults and pdf.fonttype 42 was never
applied.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Okabe-Ito, which is colorblind-safe and already what two of the three
# previous definitions used for CARVE stability.
METRIC_COLORS: dict[str, str] = {
    "ari_stability_1se": "#009E73",
    "ari_stability": "#009E73",
    "ari_stability_quant": "#009E73",
    "ari_generalizability_1se": "#56B4E9",
    "ari_generalizability": "#56B4E9",
    "ari_generalizability_quant": "#56B4E9",
    "ari_average_1se": "#E69F00",
    "ari_average": "#E69F00",
    "ari_average_quant": "#E69F00",
    "consensus_pac_stability": "#8C8C8C",
    "consensus_gini_stability": "#6E6E6E",
    "consensus_ce_stability": "#4F4F4F",
    "accuracy_generalizability": "#7BC8F0",
    "silhouette": "#E8588C",
    "gap": "#0072B2",
    "davies_bouldin": "#D55E00",
    "calinski_harabasz": "#CC79A7",
    "baseline_oracle": "#000000",
}

_FALLBACK_COLOR = "#7F7F7F"

CLUSTER_PALETTE: tuple[str, ...] = (
    "#009ADE",
    "#00CD6C",
    "#FF1F5B",
    "#AF58BA",
    "#F28522",
    "#A6761D",
    "#A0B1BA",
    "#1B9E77",
    "#7570B3",
    "#66A61E",
)

FONT_SIZES: dict[str, float] = {
    "tick": 9.0,
    "legend": 9.0,
    "axis_label": 10.0,
    "title": 11.0,
    "panel_letter": 28.0,
}

RC_PARAMS: dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": FONT_SIZES["title"],
    "axes.labelsize": FONT_SIZES["axis_label"],
    "legend.fontsize": FONT_SIZES["legend"],
    "xtick.labelsize": FONT_SIZES["tick"],
    "ytick.labelsize": FONT_SIZES["tick"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    # Type 42 keeps text editable and selectable in the submitted PDF, which
    # PLOS requires. The previous code set this in a function nothing called.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_theme() -> None:
    """Apply the manuscript style to the global rcParams."""
    plt.rcParams.update(RC_PARAMS)


@contextmanager
def theme_context() -> Iterator[None]:
    """Apply the theme for the duration of a block, then restore."""
    with mpl.rc_context(RC_PARAMS):
        yield


def metric_color(metric: str) -> str:
    """Return the color for a metric, falling back to gray for unknown names."""
    return METRIC_COLORS.get(metric, _FALLBACK_COLOR)


def cluster_colors(n: int) -> list[str]:
    """Return n cluster colors, cycling the palette when n exceeds its length."""
    return [CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(n)]


def style_axes(ax: Axes, *, grid: bool = True) -> Axes:
    """Apply the spine and grid treatment to one axes.

    Replaces the spine block that was retyped in three places with three
    different alpha values.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_alpha(0.6)
    ax.grid(grid, alpha=RC_PARAMS["grid.alpha"], linewidth=RC_PARAMS["grid.linewidth"])
    ax.set_axisbelow(True)
    return ax


def save_figure(fig: Figure, path: Path, *, dpi: int = 300) -> Path:
    """Save a figure under its manuscript filename, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_theme.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_theme.py tests/benchmarks/test_theme.py
git commit -m "feat(benchmarks): add the single theme module"
```

---

### Task 3: Panel primitives

**Files:**
- Create: `src/benchmarks/_panels.py`
- Create: `tests/benchmarks/test_panels.py`
- Read for reference: `notebooks/benchmarking_code/case_study_plotting.py:455-1278`, `notebooks/benchmarking_code/benchmarking_plotting.py:195-343`

**Interfaces:**
- Consumes: `cluster_colors`, `metric_color`, `style_axes`, `FONT_SIZES` from Task 2; `METRIC_DISPLAY_NAMES`, `metric_measure` from plan 1 Task 6.
- Produces, every one taking `ax` first and returning it:
  - `scatter_clusters(ax, Z, labels, *, color_map=None, s=20.0, alpha=0.85, title=None, hide_axes=True, axis_labels=None) -> Axes`
  - `metric_lines(ax, df, *, metrics, x_col="axis_value", x_label=None, element_scale=1.0, show_legend=True) -> Axes`
  - `carve_lines(ax, carve_obj, *, measures=("stability", "generalizability"), not_two=False, title=None, annotate=False) -> Axes`
  - `cvi_lines(ax, curves_df, best_df, *, title=None, normalize=True) -> Axes`
  - `alluvial(ax, y_true, left_labels, right_labels, *, left_cmap, right_cmap, true_cmap, left_title, right_title, true_title) -> Axes`
  - `ari_lollipop(ax, ari_df, *, title=None, annotate_k=True) -> Axes`
  - `runtime_lines(ax, runtimes_df, *, runtime_cols=("t_stability_s", "t_generalizability_s"), runtime_labels=("CARVE Stability", "CARVE Generalizability"), x_col="axis_value", x_label=None, yscale="log", dodge=0.01, element_scale=1.0, show_legend=True) -> Axes`
  - `grouped_legend(fig, axes, *, fontsize=None, y_offset=0.10) -> Legend`
  - `panel_letter(ax, letter, *, x=-0.08, y=1.1) -> None`

The old code had five different return contracts across thirteen plotting functions, and three of
them (`plot_examples`, `plot_dim_red`, `plot_alluvial`) discarded the figure and called
`plt.show()` inside library code. The uniform contract here is the point of the module, and there
is a test that enforces it.

Port the drawing bodies from the modules named above. Drop the `band` and `show_band_for`
parameters, which were threaded through three functions and ignored by all of them, and drop
`_metric_color_map` and `_hex_to_rgba`, which have no callers.

- [ ] **Step 1: Write the contract test first**

Create `tests/benchmarks/test_panels.py`:

```python
"""Tests for the panel drawing primitives.

The uniform ax-in / ax-out contract is the reason this module exists, so it
is asserted directly rather than left to convention.
"""

import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from benchmarks import _panels

AX_FIRST_FUNCTIONS = (
    "scatter_clusters",
    "metric_lines",
    "carve_lines",
    "cvi_lines",
    "alluvial",
    "ari_lollipop",
    "runtime_lines",
)


class TestContract:
    @pytest.mark.parametrize("name", AX_FIRST_FUNCTIONS)
    def test_first_parameter_is_ax(self, name):
        signature = inspect.signature(getattr(_panels, name))
        assert next(iter(signature.parameters)) == "ax"

    def test_no_primitive_calls_plt_show(self):
        source = inspect.getsource(_panels)
        assert "plt.show(" not in source

    def test_no_primitive_creates_its_own_figure(self):
        source = inspect.getsource(_panels)
        assert "plt.subplots(" not in source
        assert "plt.figure(" not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_panels.py -v`
Expected: FAIL with `ImportError: cannot import name '_panels'`

- [ ] **Step 3: Write the scatter and line primitives**

Create `src/benchmarks/_panels.py` with the module docstring and the first three primitives:

```python
"""Drawing primitives. Every one takes an ax and returns it.

Nothing here creates a figure, calls plt.show, or saves anything. Composition
is the job of the figures package; these only draw.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from ._registry import METRIC_DISPLAY_NAMES
from ._theme import FONT_SIZES, cluster_colors, metric_color, style_axes


def _display(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def cluster_color_map(labels: np.ndarray) -> dict[Any, str]:
    """Map each distinct label to a stable color."""
    unique = list(dict.fromkeys(np.asarray(labels).tolist()))
    return dict(zip(unique, cluster_colors(len(unique))))


def scatter_clusters(
    ax: Axes,
    Z: np.ndarray,
    labels: np.ndarray,
    *,
    color_map: Mapping[Any, str] | None = None,
    s: float = 20.0,
    alpha: float = 0.85,
    linewidth: float = 0.3,
    title: str | None = None,
    hide_axes: bool = True,
    axis_labels: Sequence[str] | None = None,
) -> Axes:
    """Scatter a two-dimensional embedding, colored by label."""
    Z = np.asarray(Z)
    labels = np.asarray(labels)
    color_map = color_map or cluster_color_map(labels)

    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=s,
            alpha=alpha,
            linewidth=linewidth,
            edgecolor="none",
            color=color_map.get(label, "#7F7F7F"),
            label=str(label),
        )

    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0], fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel(axis_labels[1], fontsize=FONT_SIZES["axis_label"])
    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    else:
        style_axes(ax)
    return ax


def metric_lines(
    ax: Axes,
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    x_col: str = "axis_value",
    x_label: str | None = None,
    y_label: str | None = None,
    element_scale: float = 1.0,
    show_legend: bool = True,
) -> Axes:
    """Plot mean ARI at the selected k against the axis, one line per metric.

    Consumes the unified artifact schema directly, so there is no axis-column
    sniffing. The function this replaces inferred its x column by inspecting
    which of two possible schemas it had been handed.
    """
    selected = df.loc[df["is_selected"]]

    for metric in metrics:
        sub = selected.loc[selected["metric_name"] == metric]
        if sub.empty:
            continue
        grouped = sub.groupby(x_col)["ari_at_k"]
        centers = grouped.mean()
        errors = grouped.sem()
        ax.errorbar(
            centers.index,
            centers.to_numpy(),
            yerr=errors.to_numpy(),
            marker="o",
            markersize=5.0 * element_scale,
            linewidth=1.8 * element_scale,
            capsize=2.5 * element_scale,
            color=metric_color(metric),
            label=_display(metric),
        )

    ax.set_xlabel(x_label or x_col, fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(
        y_label or r"ARI (selected $\hat{k}$ vs. true labels)",
        fontsize=FONT_SIZES["axis_label"],
    )
    if show_legend:
        ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)


RUNTIME_COLS: tuple[str, str] = ("t_stability_s", "t_generalizability_s")
RUNTIME_LABELS: tuple[str, str] = ("CARVE Stability", "CARVE Generalizability")
_RUNTIME_METRIC_FOR_COLOR = {
    "t_stability_s": "ari_stability_1se",
    "t_generalizability_s": "ari_generalizability_1se",
    "t_default_s": "ari_average_1se",
}


def runtime_lines(
    ax: Axes,
    runtimes_df: pd.DataFrame,
    *,
    runtime_cols: Sequence[str] = RUNTIME_COLS,
    runtime_labels: Sequence[str] = RUNTIME_LABELS,
    x_col: str = "axis_value",
    x_label: str | None = None,
    y_label: str = "Runtime (seconds)",
    yscale: str = "log",
    dodge: float = 0.01,
    element_scale: float = 1.0,
    show_legend: bool = True,
) -> Axes:
    """Plot CARVE wall-clock against the swept axis, one line per timed mode.

    Two series by default, matching the published figure: the stability-mode
    and generalizability-mode fits are timed separately. They are dodged
    horizontally by a fraction of the x-range so the error bars do not
    overlap at each anchor.
    """
    if len(runtime_cols) != len(runtime_labels):
        raise ValueError("runtime_cols and runtime_labels must be the same length.")

    x_values = np.sort(runtimes_df[x_col].unique().astype(float))
    span = float(x_values[-1] - x_values[0]) if len(x_values) > 1 else 1.0
    offsets = np.linspace(
        -dodge * span * (len(runtime_cols) - 1) / 2,
        dodge * span * (len(runtime_cols) - 1) / 2,
        len(runtime_cols),
    )

    for column, label, offset in zip(runtime_cols, runtime_labels, offsets):
        if column not in runtimes_df.columns:
            continue
        grouped = runtimes_df.groupby(x_col)[column]
        centers = grouped.mean().dropna()
        if centers.empty:
            continue
        errors = grouped.sem().reindex(centers.index)

        ax.errorbar(
            centers.index.to_numpy(dtype=float) + offset,
            centers.to_numpy(),
            yerr=errors.to_numpy(),
            marker="o",
            markersize=5.0 * element_scale,
            linewidth=1.8 * element_scale,
            capsize=2.5 * element_scale,
            color=metric_color(_RUNTIME_METRIC_FOR_COLOR.get(column, column)),
            label=label,
        )

    ax.set_yscale(yscale)
    ax.set_xlabel(x_label or x_col, fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(y_label, fontsize=FONT_SIZES["axis_label"])
    if show_legend:
        ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)
```

- [ ] **Step 4: Add the tests for the primitives written so far**

Append to `tests/benchmarks/test_panels.py`:

```python
from benchmarks._panels import (
    cluster_color_map,
    metric_lines,
    runtime_lines,
    scatter_clusters,
)
from benchmarks._artifacts import SCHEMA


def _results_frame():
    rows = []
    for axis_value in (0, 1, 2):
        for seed in range(3):
            for metric in ("ari_stability_1se", "silhouette"):
                for k in (4, 5):
                    rows.append(
                        {
                            "run_id": "r1",
                            "scenario": "demo",
                            "axis_name": "difficulty_level",
                            "axis_value": axis_value,
                            "axis_label": ["easy", "medium", "hard"][axis_value],
                            "seed": seed,
                            "k_star": 5,
                            "estimator": "kmeans",
                            "metric_name": metric,
                            "k": k,
                            "metric_value": 0.1 * k,
                            "is_selected": k == 5,
                            "selects_true_k": k == 5,
                            "ari_at_k": 0.9 - 0.1 * axis_value,
                            "oracle_ari": 0.95,
                        }
                    )
    return pd.DataFrame(rows)[list(SCHEMA)]


class TestScatterClusters:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        Z = np.random.default_rng(0).normal(size=(30, 2))
        labels = np.repeat([0, 1, 2], 10)
        assert scatter_clusters(ax, Z, labels) is ax
        plt.close(fig)

    def test_draws_one_collection_per_label(self):
        fig, ax = plt.subplots()
        Z = np.random.default_rng(0).normal(size=(30, 2))
        labels = np.repeat([0, 1, 2], 10)
        scatter_clusters(ax, Z, labels)
        assert len(ax.collections) == 3
        plt.close(fig)

    def test_hides_axes_when_asked(self):
        fig, ax = plt.subplots()
        Z = np.random.default_rng(0).normal(size=(10, 2))
        scatter_clusters(ax, Z, np.zeros(10, dtype=int), hide_axes=True)
        assert list(ax.get_xticks()) == []
        plt.close(fig)


class TestClusterColorMap:
    def test_assigns_one_color_per_distinct_label(self):
        mapping = cluster_color_map(np.array([2, 2, 0, 1, 1]))
        assert set(mapping) == {0, 1, 2}
        assert len(set(mapping.values())) == 3


class TestMetricLines:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        assert metric_lines(ax, _results_frame(), metrics=("silhouette",)) is ax
        plt.close(fig)

    def test_draws_one_series_per_requested_metric(self):
        fig, ax = plt.subplots()
        metric_lines(ax, _results_frame(), metrics=("ari_stability_1se", "silhouette"))
        assert len(ax.get_legend().get_texts()) == 2
        plt.close(fig)

    def test_uses_only_selected_rows(self):
        fig, ax = plt.subplots()
        metric_lines(ax, _results_frame(), metrics=("silhouette",), show_legend=False)
        line = ax.lines[0]
        # three axis values, one point each, drawn from the k=5 rows only
        assert len(line.get_xdata()) == 3
        plt.close(fig)

    def test_skips_a_metric_with_no_rows(self):
        fig, ax = plt.subplots()
        metric_lines(ax, _results_frame(), metrics=("gap",), show_legend=False)
        assert ax.lines == []
        plt.close(fig)


def _runtime_frame():
    return pd.DataFrame(
        {
            "axis_value": [1000, 1000, 5500, 5500],
            "t_default_s": [1.5, 1.7, 6.0, 6.6],
            "t_stability_s": [1.0, 1.2, 4.0, 4.4],
            "t_generalizability_s": [1.3, 1.4, 5.1, 5.5],
        }
    )


class TestRuntimeLines:
    def test_returns_the_same_axes_and_uses_a_log_scale(self):
        fig, ax = plt.subplots()
        assert runtime_lines(ax, _runtime_frame()) is ax
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_draws_both_mode_curves_by_default(self):
        fig, ax = plt.subplots()
        runtime_lines(ax, _runtime_frame())
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["CARVE Stability", "CARVE Generalizability"]
        plt.close(fig)

    def test_the_two_curves_are_dodged_apart(self):
        fig, ax = plt.subplots()
        runtime_lines(ax, _runtime_frame())
        first, second = ax.lines[0].get_xdata(), ax.lines[1].get_xdata()
        assert not np.allclose(first, second)
        plt.close(fig)

    def test_skips_a_column_that_is_all_nan(self):
        """Untimed cells record nan, and an untimed series must not be drawn."""
        fig, ax = plt.subplots()
        df = _runtime_frame()
        df["t_generalizability_s"] = np.nan
        runtime_lines(ax, df)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["CARVE Stability"]
        plt.close(fig)

    def test_skips_a_column_that_is_absent(self):
        fig, ax = plt.subplots()
        runtime_lines(ax, _runtime_frame()[["axis_value", "t_stability_s"]])
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_rejects_mismatched_columns_and_labels(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="same length"):
            runtime_lines(ax, _runtime_frame(), runtime_cols=("t_stability_s",))
        plt.close(fig)
```

- [ ] **Step 5: Run the tests written so far**

Run: `.venv/bin/pytest tests/benchmarks/test_panels.py -v`
Expected: all passed

- [ ] **Step 6: Commit the first half**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_panels.py tests/benchmarks/test_panels.py
git commit -m "feat(benchmarks): add scatter and line panel primitives"
```

---

### Task 4: Case-study panel primitives

**Files:**
- Modify: `src/benchmarks/_panels.py`
- Modify: `tests/benchmarks/test_panels.py`
- Read for reference: `notebooks/benchmarking_code/case_study_plotting.py:725-1278,1398-1531`

**Interfaces:**
- Consumes: everything from Task 3.
- Produces: `carve_lines`, `cvi_lines`, `alluvial`, `ari_lollipop`, `grouped_legend`, `panel_letter`, with the signatures listed in Task 3.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarks/test_panels.py`:

```python
from benchmarks._panels import (
    alluvial,
    ari_lollipop,
    cvi_lines,
    grouped_legend,
    panel_letter,
)


def _curves_and_best():
    curves = pd.DataFrame(
        {
            "metric": ["silhouette"] * 3 + ["gap"] * 3,
            "k": [3, 4, 5] * 2,
            "score": [0.4, 0.6, 0.5, 0.2, 0.3, 0.35],
            "model": ["KMeans"] * 6,
        }
    )
    best = pd.DataFrame(
        {"metric": ["silhouette", "gap"], "k": [4, 5], "model": ["KMeans", "KMeans"]}
    )
    return curves, best


class TestCviLines:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        assert cvi_lines(ax, curves, best) is ax
        plt.close(fig)

    def test_draws_one_line_per_metric(self):
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        assert len([ln for ln in ax.lines if ln.get_label() != "_nolegend_"]) >= 2
        plt.close(fig)

    def test_marks_the_selected_k_for_each_metric(self):
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        assert len(ax.collections) >= 2
        plt.close(fig)


class TestAlluvial:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        y_true = np.repeat([0, 1], 20)
        left = np.repeat([0, 1], 20)
        right = np.repeat([1, 0], 20)
        result = alluvial(
            ax,
            y_true,
            left,
            right,
            left_cmap=cluster_color_map(left),
            right_cmap=cluster_color_map(right),
            true_cmap=cluster_color_map(y_true),
            left_title="CARVE",
            right_title="CVI",
            true_title="Reported",
        )
        assert result is ax
        plt.close(fig)

    def test_draws_the_three_column_titles(self):
        fig, ax = plt.subplots()
        y_true = np.repeat([0, 1], 20)
        alluvial(
            ax,
            y_true,
            y_true,
            y_true,
            left_cmap=cluster_color_map(y_true),
            right_cmap=cluster_color_map(y_true),
            true_cmap=cluster_color_map(y_true),
            left_title="CARVE",
            right_title="CVI",
            true_title="Reported",
        )
        texts = [t.get_text() for t in ax.texts]
        assert "CARVE" in texts and "CVI" in texts and "Reported" in texts
        plt.close(fig)


class TestAriLollipop:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame(
            {
                "method": ["CARVE (stab)", "Silhouette"],
                "ari": [0.78, 0.63],
                "k": [10, 7],
            }
        )
        assert ari_lollipop(ax, df) is ax
        plt.close(fig)

    def test_draws_one_marker_per_method(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame(
            {"method": ["a", "b", "c"], "ari": [0.1, 0.2, 0.3], "k": [3, 4, 5]}
        )
        ari_lollipop(ax, df, annotate_k=False)
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_annotates_k_when_asked(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame({"method": ["a"], "ari": [0.5], "k": [9]})
        ari_lollipop(ax, df, annotate_k=True)
        assert any("9" in t.get_text() for t in ax.texts)
        plt.close(fig)


class TestGroupedLegend:
    def test_returns_a_legend_attached_to_the_figure(self):
        fig, axes = plt.subplots(1, 2, squeeze=False)
        for ax in axes.flat:
            ax.plot([0, 1], [0, 1], label="CARVE Stability (1SE)")
        legend = grouped_legend(fig, axes)
        assert isinstance(legend, Legend)
        plt.close(fig)

    def test_deduplicates_repeated_labels(self):
        fig, axes = plt.subplots(1, 3, squeeze=False)
        for ax in axes.flat:
            ax.plot([0, 1], [0, 1], label="Silhouette")
        legend = grouped_legend(fig, axes)
        assert len(legend.get_texts()) == 1
        plt.close(fig)


class TestPanelLetter:
    def test_adds_one_text_artist(self):
        fig, ax = plt.subplots()
        panel_letter(ax, "A")
        assert [t.get_text() for t in ax.texts] == ["A"]
        plt.close(fig)

    def test_uses_the_theme_font_size(self):
        from benchmarks._theme import FONT_SIZES

        fig, ax = plt.subplots()
        panel_letter(ax, "B")
        assert ax.texts[0].get_fontsize() == FONT_SIZES["panel_letter"]
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/benchmarks/test_panels.py -k "Cvi or Alluvial or Lollipop or GroupedLegend or PanelLetter" -v`
Expected: FAIL with `ImportError: cannot import name 'alluvial'`

- [ ] **Step 3: Append the remaining primitives**

Add to `src/benchmarks/_panels.py`:

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
    """Plot CARVE validation curves over k, one line per measure."""
    results = carve_obj.estimator_results_
    ks = results["n_clusters"].to_numpy()

    for measure in measures:
        color = metric_color(f"ari_{measure}_1se")
        values = results[measure].to_numpy()
        ax.plot(
            ks,
            values,
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=color,
            label=_display(f"ari_{measure}_1se"),
        )
        if f"{measure}_se" in results.columns:
            se = results[f"{measure}_se"].to_numpy()
            ax.fill_between(ks, values - se, values + se, color=color, alpha=0.15)

        if show_selected_k:
            selected = int(carve_obj.get_k(measure=measure, rule="1se", not_two=not_two))
            ax.axvline(selected, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
            if annotate:
                ax.annotate(
                    f"$\\hat{{k}}={selected}$",
                    xy=(selected, float(np.nanmax(values))),
                    fontsize=FONT_SIZES["legend"],
                    color=color,
                )

    ax.set_xlabel("Number of clusters $k$", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Validation score", fontsize=FONT_SIZES["axis_label"])
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)


def cvi_lines(
    ax: Axes,
    curves_df: pd.DataFrame,
    best_df: pd.DataFrame,
    *,
    title: str | None = None,
    normalize: bool = True,
) -> Axes:
    """Plot classical index curves over k, marking each index's selected k.

    Indices live on incompatible scales, so they are min-max normalized to a
    common axis by default; the selected k is unaffected by that rescaling.
    """
    for metric, sub in curves_df.groupby("metric", sort=False):
        sub = sub.sort_values("k")
        values = sub["score"].to_numpy(dtype=float)
        if normalize:
            span = np.nanmax(values) - np.nanmin(values)
            values = (values - np.nanmin(values)) / span if span > 0 else values * 0.0

        color = metric_color(str(metric))
        ax.plot(
            sub["k"].to_numpy(),
            values,
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            color=color,
            label=_display(str(metric)),
        )

        best = best_df.loc[best_df["metric"] == metric]
        if not best.empty:
            best_k = int(best["k"].iloc[0])
            match = np.where(sub["k"].to_numpy() == best_k)[0]
            if match.size:
                ax.scatter(
                    [best_k],
                    [values[match[0]]],
                    s=90,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.8,
                    zorder=5,
                )

    ax.set_xlabel("Number of clusters $k$", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(
        "Normalized index" if normalize else "Index value",
        fontsize=FONT_SIZES["axis_label"],
    )
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)


def _stack_segments(sizes: Sequence[int], gap_frac: float = 0.015) -> list[tuple[float, float]]:
    """Return (bottom, top) spans for a stacked bar with proportional gaps."""
    total = float(sum(sizes))
    if total <= 0:
        return []
    gap = gap_frac
    usable = 1.0 - gap * max(len(sizes) - 1, 0)
    spans = []
    cursor = 0.0
    for size in sizes:
        height = usable * (size / total)
        spans.append((cursor, cursor + height))
        cursor += height + gap
    return spans


def alluvial(
    ax: Axes,
    y_true: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    *,
    left_cmap: Mapping[Any, str],
    right_cmap: Mapping[Any, str],
    true_cmap: Mapping[Any, str],
    left_title: str,
    right_title: str,
    true_title: str,
    link_alpha: float = 0.35,
    bar_width: float = 0.08,
) -> Axes:
    """Draw a three-column alluvial: left clustering, truth, right clustering."""
    y_true = np.asarray(y_true)
    left_labels = np.asarray(left_labels)
    right_labels = np.asarray(right_labels)

    columns = [
        (0.0, left_labels, left_cmap, left_title),
        (0.5, y_true, true_cmap, true_title),
        (1.0, right_labels, right_cmap, right_title),
    ]

    spans_by_column = []
    for x, labels, cmap, title in columns:
        categories = list(dict.fromkeys(labels.tolist()))
        sizes = [int((labels == c).sum()) for c in categories]
        spans = _stack_segments(sizes)
        for category, (bottom, top) in zip(categories, spans):
            ax.add_patch(
                plt_rectangle(x - bar_width / 2, bottom, bar_width, top - bottom,
                              cmap.get(category, "#7F7F7F"))
            )
        ax.text(x, 1.04, title, ha="center", va="bottom",
                fontsize=FONT_SIZES["axis_label"])
        spans_by_column.append((x, categories, dict(zip(categories, spans))))

    for (x0, cats0, spans0), (x1, cats1, spans1), left_arr, right_arr, flow_cmap in (
        (*spans_by_column[0:2], left_labels, y_true, left_cmap),
        (*spans_by_column[1:3], y_true, right_labels, true_cmap),
    ):
        cursor0 = {c: spans0[c][0] for c in cats0}
        cursor1 = {c: spans1[c][0] for c in cats1}
        for c0 in cats0:
            total0 = max(int((left_arr == c0).sum()), 1)
            height0 = spans0[c0][1] - spans0[c0][0]
            for c1 in cats1:
                overlap = int(((left_arr == c0) & (right_arr == c1)).sum())
                if overlap == 0:
                    continue
                h0 = height0 * overlap / total0
                total1 = max(int((right_arr == c1).sum()), 1)
                h1 = (spans1[c1][1] - spans1[c1][0]) * overlap / total1
                ax.fill_between(
                    np.linspace(x0 + bar_width / 2, x1 - bar_width / 2, 32),
                    np.linspace(cursor0[c0], cursor1[c1], 32),
                    np.linspace(cursor0[c0] + h0, cursor1[c1] + h1, 32),
                    color=flow_cmap.get(c0, "#7F7F7F"),
                    alpha=link_alpha,
                    linewidth=0,
                )
                cursor0[c0] += h0
                cursor1[c1] += h1

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.02, 1.12)
    ax.set_axis_off()
    return ax


def plt_rectangle(x: float, y: float, width: float, height: float, color: str):
    """Build a filled rectangle patch. Split out so alluvial stays readable."""
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), width, height, facecolor=color, edgecolor="none")


def ari_lollipop(
    ax: Axes,
    ari_df: pd.DataFrame,
    *,
    title: str | None = None,
    annotate_k: bool = True,
) -> Axes:
    """Horizontal lollipop of ARI against reported labels, one row per method."""
    ordered = ari_df.sort_values("ari", ascending=True).reset_index(drop=True)
    positions = np.arange(len(ordered))
    colors = [metric_color(str(m)) for m in ordered.get("metric", ordered["method"])]

    ax.hlines(positions, 0, ordered["ari"].to_numpy(), color=colors, linewidth=2.0)
    ax.scatter(ordered["ari"].to_numpy(), positions, color=colors, s=60, zorder=3)

    if annotate_k and "k" in ordered.columns:
        for position, (ari, k) in enumerate(zip(ordered["ari"], ordered["k"])):
            ax.text(
                ari + 0.01,
                position,
                f"k={int(k)}",
                va="center",
                fontsize=FONT_SIZES["legend"],
            )

    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["method"], fontsize=FONT_SIZES["tick"])
    ax.set_xlabel("ARI vs. reported labels", fontsize=FONT_SIZES["axis_label"])
    ax.set_xlim(0, max(1.0, float(ordered["ari"].max()) * 1.15))
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    return style_axes(ax)


def grouped_legend(
    fig: Figure,
    axes: np.ndarray,
    *,
    fontsize: float | None = None,
    y_offset: float = 0.10,
    ncol: int | None = None,
) -> Legend:
    """One deduplicated legend below a grid of axes.

    The old code extracted this helper out of a 408-line figure function and
    then never removed the original, so the same legend was built twice.
    """
    handles: list[Line2D] = []
    labels: list[str] = []
    for ax in np.asarray(axes).flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    return fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -y_offset),
        ncol=ncol or min(len(labels), 4),
        frameon=False,
        fontsize=fontsize or FONT_SIZES["legend"],
    )


def panel_letter(ax: Axes, letter: str, *, x: float = -0.08, y: float = 1.1) -> None:
    """Place a bold panel letter in axes coordinates."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=FONT_SIZES["panel_letter"],
        fontweight="bold",
        va="top",
        ha="right",
    )
```

- [ ] **Step 4: Run the panel tests**

Run: `.venv/bin/pytest tests/benchmarks/test_panels.py -v`
Expected: all passed. If `alluvial` fails on the flow geometry, the failure is in `_stack_segments`
or the cursor bookkeeping, not in the contract; fix it there rather than loosening the test.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_panels.py tests/benchmarks/test_panels.py
git commit -m "feat(benchmarks): add case-study panel primitives"
```

---

## Decision recorded: fitted state is compute, not reporting

`notebooks/case_studies/carve_state_saves/` holds `carve_klein.carve` (48 MB) and
`carve_levine32.carve` (712 MB), both gitignored. The notebooks fit-or-load against those paths
and restore `carve.X_ = X` after loading.

That is a fit cache, and it belongs on the compute side of the split the spec already decided.
So: every `figure_*` function takes an already-fitted `CARVE` object as an argument and never
fits, loads, or caches anything. `_studies.py` (Task 7) owns fit-or-load. This keeps the figure
modules pure and testable with a small stub, and it means regenerating a figure from a cached fit
costs seconds rather than hours.

---

### Task 5: Klein loader

**Files:**
- Create: `src/benchmarks/datasets/__init__.py`
- Create: `src/benchmarks/datasets/_klein.py`
- Create: `tests/benchmarks/datasets/test_klein.py`
- Read for reference: `notebooks/case_studies/Klein.ipynb` cells 9, 11, 13, 15, 17, 18, 20, 21

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `DATA_ROOT: Path` — `Path("data")` relative to the repo root, overridable
  - `resolve_data_dir(name: str, *, root: Path | None = None) -> Path` — case-insensitive lookup
  - `load_klein(*, root=None, subsample=None, random_state=42, n_top_genes=2000) -> tuple[np.ndarray, pd.Series, dict]`

Two fixes land here. The notebook uses `Path("../../data/klein")` while the directory on disk is
`data/Klein`; that works on macOS, whose default filesystem is case-insensitive, and fails on
Linux, which is what CI runs. `resolve_data_dir` looks the name up case-insensitively and raises
with the candidates it saw, so the next mismatch is a clear error rather than a platform-dependent
one.

The loader is also the single home for preprocessing that currently appears in both `Klein.ipynb`
and `Motivation.ipynb`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/datasets/test_klein.py`:

```python
"""Tests for the Klein loader.

The heavy path needs the raw GEO files under data/Klein and is skipped when
they are absent, so the suite still runs on a machine without the data.
"""

from pathlib import Path

import pytest

from benchmarks.datasets import load_klein, resolve_data_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
KLEIN_DIR = REPO_ROOT / "data" / "Klein"
needs_data = pytest.mark.skipif(
    not KLEIN_DIR.is_dir(), reason="data/Klein is not present in this checkout"
)


class TestResolveDataDir:
    def test_finds_a_directory_by_its_exact_name(self, tmp_path):
        (tmp_path / "Klein").mkdir()
        assert resolve_data_dir("Klein", root=tmp_path).name == "Klein"

    def test_finds_a_directory_ignoring_case(self, tmp_path):
        (tmp_path / "Klein").mkdir()
        assert resolve_data_dir("klein", root=tmp_path).name == "Klein"

    def test_raises_with_the_candidates_it_saw(self, tmp_path):
        (tmp_path / "Levine").mkdir()
        with pytest.raises(FileNotFoundError, match="Levine"):
            resolve_data_dir("klein", root=tmp_path)

    def test_error_names_what_was_requested(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="klein"):
            resolve_data_dir("klein", root=tmp_path)


@needs_data
class TestLoadKlein:
    def test_returns_x_y_and_metadata(self):
        X, y, meta = load_klein(subsample=200, random_state=42)
        assert X.ndim == 2
        assert len(y) == X.shape[0]
        assert meta["source"] == "Klein"

    def test_subsampling_is_honored(self):
        X, _, _ = load_klein(subsample=200, random_state=42)
        assert X.shape[0] == 200

    def test_subsampling_is_deterministic(self):
        import numpy as np

        first, _, _ = load_klein(subsample=200, random_state=42)
        second, _, _ = load_klein(subsample=200, random_state=42)
        np.testing.assert_array_equal(first, second)

    def test_labels_are_the_four_collection_days(self):
        _, y, _ = load_klein(subsample=200, random_state=42)
        assert set(y.unique()) <= {"d0", "d2", "d4", "d7"}

    def test_metadata_records_the_preprocessing_chain(self):
        _, _, meta = load_klein(subsample=200, random_state=42)
        assert "normalize_total" in meta["preprocessing"]
        assert "log1p" in meta["preprocessing"]
        assert meta["n_top_genes"] == 2000

    def test_hvg_selection_bounds_the_feature_count(self):
        X, _, _ = load_klein(subsample=200, random_state=42)
        assert X.shape[1] <= 2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/datasets/test_klein.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.datasets'`

- [ ] **Step 3: Write the package init and directory resolver**

Create `src/benchmarks/datasets/__init__.py`:

```python
"""Dataset loaders for the case studies.

Every loader returns (X, y, meta): a float array of shape (n_samples,
n_features), a pandas Series of labels, and a dict recording provenance and
the exact preprocessing applied. The loaders are the single home for
preprocessing that previously appeared in two notebooks each.
"""

from ._klein import DATA_ROOT, load_klein, resolve_data_dir
from ._levine import load_levine32

__all__ = ["DATA_ROOT", "load_klein", "load_levine32", "resolve_data_dir"]
```

Create `src/benchmarks/datasets/_klein.py`:

```python
"""Klein et al. mouse embryonic stem cell scRNA-seq (GSE65525).

Four collection days (d0, d2, d4, d7) are concatenated and the day is used as
the reported label. Preprocessing is scanpy's standard chain: total-count
normalization to 1e4, log1p, then the top 2000 highly variable genes selected
per batch with the seurat flavor.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("data")

_FILES = {
    "d0": "GSM1599494_ES_d0_main.csv.bz2",
    "d2": "GSM1599497_ES_d2_LIFminus.csv.bz2",
    "d4": "GSM1599498_ES_d4_LIFminus.csv.bz2",
    "d7": "GSM1599499_ES_d7_LIFminus.csv.bz2",
}


def resolve_data_dir(name: str, *, root: Path | None = None) -> Path:
    """Find a dataset directory, ignoring case.

    The notebooks referred to data/klein while the directory on disk is
    data/Klein. That works on a case-insensitive macOS filesystem and fails
    on Linux, which is what CI runs.
    """
    root = Path(root) if root is not None else DATA_ROOT
    exact = root / name
    if exact.is_dir():
        return exact

    if root.is_dir():
        lowered = name.lower()
        for candidate in sorted(root.iterdir()):
            if candidate.is_dir() and candidate.name.lower() == lowered:
                return candidate
        seen = sorted(c.name for c in root.iterdir() if c.is_dir())
    else:
        seen = []

    raise FileNotFoundError(
        f"No dataset directory matching {name!r} under {root}. Found: {seen}."
    )


def _read_block(path: Path) -> pd.DataFrame:
    """Read one genes-by-cells block, coercing stray strings to zero."""
    df = pd.read_csv(path, header=None, index_col=0, compression="bz2")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df[~df.index.duplicated(keep="first")]


def load_klein(
    *,
    root: Path | None = None,
    subsample: int | float | None = None,
    random_state: int = 42,
    n_top_genes: int = 2000,
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess the Klein dataset.

    Parameters
    ----------
    root : Path or None
        Directory holding the dataset folders. Defaults to data/.
    subsample : int, float, or None
        Stratified subsample size, as a count or a fraction. None keeps all.
    random_state : int
        Seed for the stratified subsample.
    n_top_genes : int
        Highly variable genes to keep.

    Returns
    -------
    (X, y, meta)
    """
    import anndata as ad
    import scanpy as sc
    from sklearn.model_selection import StratifiedShuffleSplit

    data_dir = resolve_data_dir("Klein", root=root)

    blocks = {key: _read_block(data_dir / filename) for key, filename in _FILES.items()}

    genes = blocks["d0"].index
    for key in ("d2", "d4", "d7"):
        genes = genes.union(blocks[key].index)
    blocks = {key: block.reindex(genes).fillna(0.0) for key, block in blocks.items()}

    for key, block in blocks.items():
        block.columns = [f"{key}_{i + 1:04d}" for i in range(block.shape[1])]

    expression = pd.concat([blocks[k] for k in ("d0", "d2", "d4", "d7")], axis=1)
    labels = [key for key in ("d0", "d2", "d4", "d7") for _ in range(blocks[key].shape[1])]
    cells = expression.columns
    condition = pd.Series(labels, index=cells, name="condition").astype("category")

    adata = ad.AnnData(
        X=expression.T.values.astype(np.float32),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=expression.index.astype(str)),
    )
    adata.obs["condition"] = condition
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes, batch_key="condition", flavor="seurat"
        )
    adata = adata[:, adata.var["highly_variable"]].copy()

    X = np.asarray(adata.X)
    y = pd.Series(adata.obs["condition"].to_numpy(), name="condition")

    n_full = X.shape[0]
    if subsample is not None:
        size = int(subsample * n_full) if isinstance(subsample, float) else int(subsample)
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=size / n_full, random_state=random_state
        )
        _, idx = next(splitter.split(X, y))
        X = X[idx]
        y = y.iloc[idx].reset_index(drop=True)

    meta = {
        "source": "Klein",
        "accession": "GSE65525",
        "citation": "Klein et al. 2015, Cell 161(5):1187-1201",
        "n_cells_full": int(n_full),
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_top_genes": int(n_top_genes),
        "label_name": "condition",
        "preprocessing": [
            "concatenate d0/d2/d4/d7 on the gene union, missing filled with 0",
            "normalize_total(target_sum=1e4)",
            "log1p",
            f"highly_variable_genes(n_top_genes={n_top_genes}, "
            "batch_key='condition', flavor='seurat')",
        ],
        "subsample": subsample,
        "random_state": random_state,
    }
    return X, y, meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/datasets/test_klein.py -v`
Expected: the `TestResolveDataDir` tests pass. The `TestLoadKlein` tests pass if `data/Klein` is
present and skip otherwise.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/datasets/__init__.py src/benchmarks/datasets/_klein.py tests/benchmarks/datasets/test_klein.py
git commit -m "feat(benchmarks): add the Klein loader and case-insensitive data lookup"
```

---

### Task 6: Levine loader with a disk cache

**Files:**
- Create: `src/benchmarks/datasets/_levine.py`
- Create: `tests/benchmarks/datasets/test_levine.py`
- Read for reference: `notebooks/case_studies/Levine_32dim.ipynb` cells 7, 8, 9, 10, 12

**Interfaces:**
- Consumes: `resolve_data_dir`, `DATA_ROOT` from Task 5.
- Produces: `load_levine32(*, cache_dir=None, allow_install=False, subsample=None, random_state=42) -> tuple[np.ndarray, pd.Series, dict]`

Two fixes land here. The current loader runs `BiocManager::install("HDCytoData")` through rpy2 as
a side effect of executing the notebook, which mutates the user's R library without asking; that
becomes opt-in behind `allow_install`. And it caches nothing, so every run pays the full R round
trip; the rebuilt loader writes an `.npz` cache and reads it back on subsequent calls.

One preprocessing detail must be preserved exactly rather than tidied. `RobustScaler` is fit on
the labeled cells including population 15, and population 15 is dropped afterwards from the
already-scaled matrix. The scaler's quantiles therefore include a population that is not in the
returned data. That is what produced the published results, so it stays, with a docstring saying
so.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/datasets/test_levine.py`:

```python
"""Tests for the Levine 32-dimension CyTOF loader.

The R round trip is skipped unless CARVE_ALLOW_R=1, so the suite runs on a
machine without HDCytoData installed. The cache logic is tested with a
synthetic cache file, which needs no R at all.
"""

import os

import numpy as np
import pytest

from benchmarks.datasets import load_levine32
from benchmarks.datasets._levine import _cache_path, _read_cache, _write_cache

needs_r = pytest.mark.skipif(
    os.environ.get("CARVE_ALLOW_R") != "1",
    reason="set CARVE_ALLOW_R=1 to exercise the rpy2 path",
)


class TestCache:
    def test_round_trips_through_npz(self, tmp_path):
        X = np.arange(12, dtype=float).reshape(4, 3)
        y = np.array(["a", "b", "a", "c"])
        markers = ["m1", "m2", "m3"]
        _write_cache(tmp_path, X, y, markers)
        loaded_X, loaded_y, loaded_markers = _read_cache(tmp_path)
        np.testing.assert_array_equal(loaded_X, X)
        np.testing.assert_array_equal(loaded_y, y)
        assert loaded_markers == markers

    def test_reports_a_missing_cache_as_none(self, tmp_path):
        assert _read_cache(tmp_path) is None

    def test_cache_path_is_under_the_cache_dir(self, tmp_path):
        assert _cache_path(tmp_path).parent == tmp_path


class TestInstallGate:
    def test_refuses_to_install_without_explicit_opt_in(self, tmp_path):
        with pytest.raises(RuntimeError, match="allow_install"):
            load_levine32(cache_dir=tmp_path, allow_install=False)

    def test_error_explains_how_to_proceed(self, tmp_path):
        with pytest.raises(RuntimeError, match="HDCytoData"):
            load_levine32(cache_dir=tmp_path, allow_install=False)


class TestLoadFromCache:
    def test_uses_the_cache_without_touching_r(self, tmp_path):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = rng.choice(["1", "2", "3"], size=200)
        _write_cache(tmp_path, X, y, ["m1", "m2", "m3", "m4"])

        loaded_X, loaded_y, meta = load_levine32(cache_dir=tmp_path)
        assert loaded_X.shape == (200, 4)
        assert len(loaded_y) == 200
        assert meta["from_cache"] is True

    def test_subsampling_is_honored_and_deterministic(self, tmp_path):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4))
        y = rng.choice(["1", "2", "3"], size=300)
        _write_cache(tmp_path, X, y, ["m1", "m2", "m3", "m4"])

        first, _, _ = load_levine32(cache_dir=tmp_path, subsample=100, random_state=7)
        second, _, _ = load_levine32(cache_dir=tmp_path, subsample=100, random_state=7)
        assert first.shape[0] == 100
        np.testing.assert_array_equal(first, second)

    def test_metadata_records_the_scaler_ordering_quirk(self, tmp_path):
        rng = np.random.default_rng(0)
        _write_cache(tmp_path, rng.normal(size=(50, 3)), rng.choice(["1", "2"], 50), ["a", "b", "c"])
        _, _, meta = load_levine32(cache_dir=tmp_path)
        assert any("population 15" in step for step in meta["preprocessing"])


@needs_r
class TestLoadFromR:
    def test_populates_the_cache_on_first_load(self, tmp_path):
        X, y, meta = load_levine32(cache_dir=tmp_path, allow_install=True, subsample=500)
        assert X.shape[0] == 500
        assert meta["from_cache"] is False
        assert _cache_path(tmp_path).exists()

    def test_population_15_is_absent(self, tmp_path):
        _, y, _ = load_levine32(cache_dir=tmp_path, allow_install=True)
        assert "15" not in set(np.asarray(y).astype(str))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/datasets/test_levine.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_levine32'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/datasets/_levine.py`:

```python
"""Levine et al. 32-dimension CyTOF bone marrow data, via HDCytoData.

Cells with an assigned population are kept, restricted to the type markers
(marker_class == 2), arcsinh transformed with cofactor 5, robust scaled, and
population 15 is then removed.

The ordering of the last two steps is deliberate and is preserved from the
code that produced the published results: RobustScaler is fit on the labeled
cells while population 15 is still present, and population 15 is dropped from
the already-scaled matrix afterwards. The scaler's quantiles therefore
reflect a population that is not in the returned data. Do not "correct" this
without re-running the case study; it changes the numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ._klein import DATA_ROOT

_CACHE_NAME = "levine32_preprocessed.npz"
_COFACTOR = 5.0
_QUANTILE_RANGE = (10, 90)
_DROPPED_POPULATION = "15"


def _cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / _CACHE_NAME


def _write_cache(
    cache_dir: Path, X: np.ndarray, y: np.ndarray, markers: list[str]
) -> Path:
    path = _cache_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X=X, y=np.asarray(y).astype(str), markers=np.asarray(markers, dtype=object)
    )
    return path


def _read_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    path = _cache_path(cache_dir)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as payload:
        return payload["X"], payload["y"], [str(m) for m in payload["markers"]]


def _load_from_r(allow_install: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch and preprocess the dataset through rpy2.

    Installing an R package is a side effect on the user's machine, so it is
    opt-in. The previous loader ran BiocManager::install unconditionally as a
    side effect of executing a notebook cell.
    """
    if not allow_install:
        raise RuntimeError(
            "The Levine dataset is not cached and fetching it requires the R package "
            "HDCytoData, which may need installing. Re-run with allow_install=True to "
            "permit BiocManager::install, or place a prepared cache at the cache_dir."
        )

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from sklearn.preprocessing import RobustScaler

    ro.r(r"""
    if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
    if (!requireNamespace("HDCytoData", quietly=TRUE)) BiocManager::install("HDCytoData")
    library(HDCytoData)
    library(SummarizedExperiment)
    """)
    ro.r("sce <- Levine_32dim_SE()")

    X_all = np.array(ro.r("assay(sce)"), dtype=float)
    row_df = pandas2ri.rpy2py(ro.r("as.data.frame(rowData(sce))"))
    col_df = pandas2ri.rpy2py(ro.r("as.data.frame(colData(sce))"))

    labeled = (row_df["population_id"] != "unassigned").to_numpy()
    X_labeled = X_all[labeled]
    y_labeled = row_df["population_id"].to_numpy()[labeled].astype(str)

    type_markers = (col_df["marker_class"].astype(int) == 2).to_numpy()
    X_labeled = X_labeled[:, type_markers]
    markers = [str(m) for m in col_df.index[type_markers].tolist()]

    # Scaler is fit while population 15 is still present. See module docstring.
    X_transformed = np.arcsinh(X_labeled.astype(np.float64) / _COFACTOR)
    X_scaled = RobustScaler(quantile_range=_QUANTILE_RANGE).fit_transform(X_transformed)

    keep = y_labeled != _DROPPED_POPULATION
    return X_scaled[keep], y_labeled[keep], markers


def load_levine32(
    *,
    cache_dir: Path | None = None,
    allow_install: bool = False,
    subsample: int | float | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess the Levine 32-dimension dataset.

    Reads a local cache when one exists, and otherwise goes through rpy2,
    which requires allow_install=True.

    Returns
    -------
    (X, y, meta)
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_ROOT / "refs"

    cached = _read_cache(cache_dir)
    from_cache = cached is not None
    if cached is None:
        X, y_arr, markers = _load_from_r(allow_install)
        _write_cache(cache_dir, X, y_arr, markers)
    else:
        X, y_arr, markers = cached

    y = pd.Series(np.asarray(y_arr).astype(str), name="population_id")

    n_full = X.shape[0]
    if subsample is not None:
        size = int(subsample * n_full) if isinstance(subsample, float) else int(subsample)
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=size / n_full, random_state=random_state
        )
        _, idx = next(splitter.split(X, y))
        X = X[idx]
        y = y.iloc[idx].reset_index(drop=True)

    meta = {
        "source": "Levine_32dim",
        "package": "HDCytoData",
        "citation": "Levine et al. 2015, Cell 162(1):184-197",
        "n_cells_full": int(n_full),
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "markers": markers,
        "label_name": "population_id",
        "from_cache": bool(from_cache),
        "preprocessing": [
            "keep cells with an assigned population",
            "keep type markers (marker_class == 2)",
            f"arcsinh(x / {_COFACTOR})",
            f"RobustScaler(quantile_range={_QUANTILE_RANGE}) fit before "
            "population 15 is dropped, deliberately",
            "drop population 15",
        ],
        "subsample": subsample,
        "random_state": random_state,
    }
    return X, y, meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/datasets/test_levine.py -v`
Expected: the cache and install-gate tests pass; the R tests skip.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/datasets/_levine.py tests/benchmarks/datasets/test_levine.py
git commit -m "feat(benchmarks): add the Levine loader with a disk cache and opt-in install"
```

---

### Task 7: Case-study compute — fit cache and CVI sweep

**Files:**
- Create: `src/benchmarks/_studies.py`
- Create: `tests/benchmarks/test_studies.py`
- Read for reference: `notebooks/benchmarking_code/case_study_plotting.py:260-455`, `notebooks/case_studies/Klein.ipynb` cells 24, 25, 28

**Interfaces:**
- Consumes: `Study`, `EstimatorSpec` from plan 1 Task 4 and 2; `param_grids`, `build_estimator` from plan 1 Task 5; `calculate_cvi`, `select_k` from plan 1 Task 8; `load_klein`, `load_levine32` from Tasks 5 and 6.
- Produces:
  - `CVI_SWEEP_METRICS: tuple[str, ...]` = `("silhouette", "gap", "davies_bouldin", "calinski_harabasz")`
  - `cvi_sweep(X, y, *, model_grids, candidate_k, random_state=0, n_jobs=1) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(curves_df, best_df)`; `curves_df` columns are `("metric", "model", "k", "score", "ari")` and `best_df` columns are `("metric", "model", "k", "score", "ari")`
  - `fit_or_load_carve(X, y, *, cache_path, model_grids, n_jobs=1, random_state=42, force=False) -> CARVE`
  - `STUDIES: dict[str, Study]` with keys `"klein"` and `"levine32"`
  - `study_model_grids(study: Study) -> list[tuple[type[ClusterMixin], dict[str, list]]]` — KMeans plus spectral, which is what the case studies sweep

This is compute, so it lives outside `figures/` and imports no matplotlib. The function it
replaces, `baseline_metrics_over_k`, swept the indices and then built a figure and called
`plt.show()` in the same call, which is why the sweep could not be reused without drawing.

`fit_or_load_carve` restores `carve.X_ = X` after loading, matching the notebook, because the
saved state does not carry the data matrix.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_studies.py`:

```python
"""Tests for case-study compute."""

import numpy as np
import pandas as pd
import pytest

from benchmarks._estimators import param_grids
from benchmarks._studies import (
    CVI_SWEEP_METRICS,
    STUDIES,
    cvi_sweep,
    fit_or_load_carve,
)
from benchmarks._types import EstimatorSpec


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=0.0, scale=0.4, size=(40, 3)),
        rng.normal(loc=6.0, scale=0.4, size=(40, 3)),
        rng.normal(loc=(0.0, 6.0, 0.0), scale=0.4, size=(40, 3)),
    ])
    y = np.repeat(["a", "b", "c"], 40)
    return X, y


class TestCviSweep:
    def test_returns_curves_and_best(self, blobs):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3, 4))
        curves, best = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3, 4))
        assert set(curves.columns) == {"metric", "model", "k", "score", "ari"}
        assert set(best.columns) == {"metric", "model", "k", "score", "ari"}

    def test_one_curve_row_per_metric_model_and_k(self, blobs):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3, 4))
        curves, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3, 4))
        assert len(curves) == len(CVI_SWEEP_METRICS) * 3

    def test_one_best_row_per_metric(self, blobs):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3, 4))
        _, best = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3, 4))
        assert len(best) == len(CVI_SWEEP_METRICS)
        assert set(best["metric"]) == set(CVI_SWEEP_METRICS)

    def test_gap_uses_tibshirani_not_argmax(self, blobs):
        """The selected k must come from select_k, not from the score column."""
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3, 4))
        curves, best = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3, 4))
        gap_curve = curves[curves["metric"] == "gap"].sort_values("k")
        argmax_k = int(gap_curve.loc[gap_curve["score"].idxmax(), "k"])
        selected_k = int(best.loc[best["metric"] == "gap", "k"].iloc[0])
        assert selected_k <= argmax_k

    def test_ari_is_recorded_against_the_true_labels(self, blobs):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3, 4))
        curves, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3, 4))
        at_three = curves[(curves["k"] == 3) & (curves["metric"] == "silhouette")]
        assert at_three["ari"].iloc[0] > 0.9

    def test_works_without_labels(self, blobs):
        X, _ = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        curves, _ = cvi_sweep(X, None, model_grids=grids, candidate_k=(2, 3))
        assert curves["ari"].isna().all()

    def test_is_deterministic(self, blobs):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        first, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3), random_state=1)
        second, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3), random_state=1)
        pd.testing.assert_frame_equal(first, second)


class TestFitOrLoadCarve:
    def test_fits_and_writes_the_cache_on_first_call(self, blobs, tmp_path):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        carve = fit_or_load_carve(
            X, y, cache_path=cache, model_grids=grids, n_resamples=3
        )
        assert cache.exists()
        assert carve.estimator_results_ is not None

    def test_second_call_loads_the_cache(self, blobs, tmp_path):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        mtime = cache.stat().st_mtime_ns
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        assert cache.stat().st_mtime_ns == mtime

    def test_loaded_object_has_the_data_matrix_restored(self, blobs, tmp_path):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        carve = fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        np.testing.assert_array_equal(carve.X_, X)

    def test_force_refits_even_with_a_cache(self, blobs, tmp_path):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        mtime = cache.stat().st_mtime_ns
        fit_or_load_carve(
            X, y, cache_path=cache, model_grids=grids, n_resamples=3, force=True
        )
        assert cache.stat().st_mtime_ns != mtime


class TestStudies:
    def test_both_case_studies_are_registered(self):
        assert set(STUDIES) == {"klein", "levine32"}

    def test_each_study_has_a_loader_and_candidate_k(self):
        for study in STUDIES.values():
            assert callable(study.loader)
            assert len(study.candidate_k) > 0

    def test_klein_sweeps_k_two_through_ten(self):
        assert STUDIES["klein"].candidate_k == tuple(range(2, 11))

    def test_levine_sweeps_k_seven_through_seventeen(self):
        assert STUDIES["levine32"].candidate_k == tuple(range(7, 18))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_studies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._studies'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_studies.py`:

```python
"""Case-study compute: the CVI sweep and the CARVE fit cache.

No matplotlib here. The routine this replaces swept the classical indices and
then built a figure and called plt.show() in the same call, so the sweep
could not be reused without also drawing it.
"""

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from carve import CARVE
from joblib import Parallel, delayed
from sklearn.base import ClusterMixin
from sklearn.metrics import adjusted_rand_score

from ._cvi import calculate_cvi, select_k
from ._estimators import param_grids
from ._types import EstimatorSpec, Study

CVI_SWEEP_METRICS: tuple[str, ...] = (
    "silhouette",
    "gap",
    "davies_bouldin",
    "calinski_harabasz",
)


def _model_label(estimator_cls: type[ClusterMixin], params: dict[str, Any]) -> str:
    """A short readable name for one estimator configuration."""
    if not params:
        return estimator_cls.__name__
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{estimator_cls.__name__} ({rendered})"


def _spec_for(estimator_cls: type[ClusterMixin]) -> EstimatorSpec:
    """Map an estimator class back to its spec, for the gap statistic."""
    from ._estimators import ESTIMATOR_CLASSES

    for name, cls in ESTIMATOR_CLASSES.items():
        if cls is estimator_cls:
            return EstimatorSpec(name=name)
    raise ValueError(f"No EstimatorSpec is registered for {estimator_cls.__name__}.")


def _sweep_cell(
    X: np.ndarray,
    y: np.ndarray | None,
    estimator_cls: type[ClusterMixin],
    fixed_params: dict[str, Any],
    k: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Fit one estimator at one k and score every classical index."""
    params = dict(fixed_params)
    params["n_clusters"] = int(k)
    import inspect

    if "random_state" in inspect.signature(estimator_cls.__init__).parameters:
        params["random_state"] = int(random_state)

    labels = np.asarray(estimator_cls(**params).fit_predict(X), dtype=np.int32)
    ari = float(adjusted_rand_score(y, labels)) if y is not None else float("nan")
    spec = _spec_for(estimator_cls)

    rows = []
    for metric in CVI_SWEEP_METRICS:
        score, error = calculate_cvi(
            X, labels, metric, spec=spec, random_state=random_state
        )
        rows.append(
            {
                "metric": metric,
                "model": _model_label(estimator_cls, fixed_params),
                "k": int(k),
                "score": float(score),
                "se": float(error),
                "ari": ari,
            }
        )
    return rows


def cvi_sweep(
    X: np.ndarray,
    y: np.ndarray | None,
    *,
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
    candidate_k: tuple[int, ...],
    random_state: int = 0,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep every classical index over every (model, k) combination.

    Returns
    -------
    curves_df : one row per (metric, model, k) with columns
        (metric, model, k, score, ari)
    best_df : one row per metric, the selected (model, k). The gap statistic
        selects by Tibshirani's rule; the others by argmax. Selection goes
        through _cvi.select_k so the two agree with the simulated benchmarks.
    """
    X = np.asarray(X)
    y_arr = None if y is None else np.asarray(y)

    jobs = []
    for estimator_cls, grid in model_grids:
        other_keys = [key for key in grid if key != "n_clusters"]
        other_values = [
            grid[key] if isinstance(grid[key], list | tuple | np.ndarray) else [grid[key]]
            for key in other_keys
        ]
        combos = list(product(*other_values)) if other_keys else [()]

        for combo in combos:
            fixed = dict(zip(other_keys, combo)) if other_keys else {}
            for k in candidate_k:
                jobs.append(
                    delayed(_sweep_cell)(X, y_arr, estimator_cls, fixed, k, random_state)
                )

    raw = Parallel(n_jobs=n_jobs)(jobs)
    curves = pd.DataFrame([row for batch in raw for row in batch])
    curves = curves.sort_values(["metric", "model", "k"]).reset_index(drop=True)
    # se is an internal selection input, not part of the returned contract.
    returned_curves = curves.drop(columns=["se"])

    best_rows = []
    for metric in CVI_SWEEP_METRICS:
        sub = curves[curves["metric"] == metric]
        candidates = []
        for model, by_model in sub.groupby("model", sort=False):
            by_model = by_model.sort_values("k")
            ks = by_model["k"].tolist()
            scores = by_model["score"].tolist()
            # se is the gap statistic's s_k, carried out of the sweep rather
            # than recomputed, and 0.0 for every other index.
            errors = by_model["se"].tolist()
            chosen = select_k(metric, ks, scores, errors)
            row = by_model[by_model["k"] == chosen].iloc[0]
            candidates.append(row)

        winner = max(candidates, key=lambda r: r["score"])
        best_rows.append(winner.to_dict())

    best = pd.DataFrame(best_rows)[["metric", "model", "k", "score", "ari"]]
    return returned_curves, best.reset_index(drop=True)


def fit_or_load_carve(
    X: np.ndarray,
    y: np.ndarray | pd.Series | None,
    *,
    cache_path: Path,
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
    n_resamples: int = 100,
    n_jobs: int = 1,
    random_state: int = 42,
    force: bool = False,
) -> CARVE:
    """Fit CARVE on a case study, caching the fitted state to disk.

    A Levine fit takes hours, so the cache is what makes regenerating a figure
    practical. The saved state does not carry the data matrix, so X_ is
    restored after loading, matching the notebook this replaces.
    """
    cache_path = Path(cache_path)

    if cache_path.is_file() and not force:
        carve = CARVE.load(str(cache_path))
        carve.X_ = np.asarray(X)
        return carve

    carve = CARVE(
        estimator_param_grids=model_grids,
        n_resamples=n_resamples,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    reference = None if y is None else np.asarray(y)
    carve.fit(np.asarray(X), reference_labels=reference)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    carve.save(str(cache_path))
    return carve


def _klein_loader():
    from .datasets import load_klein

    return load_klein(subsample=None, random_state=42)


def _levine_loader():
    from .datasets import load_levine32

    return load_levine32(subsample=None, random_state=42)


STUDIES: dict[str, Study] = {
    "klein": Study(
        name="klein",
        loader=_klein_loader,
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=tuple(range(2, 11)),
    ),
    "levine32": Study(
        name="levine32",
        loader=_levine_loader,
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=tuple(range(7, 18)),
    ),
}


def study_model_grids(study: Study) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """Both estimators the case studies sweep: KMeans and spectral.

    The manuscript reports running both for the case studies, unlike the
    simulated benchmarks, which fix one estimator per scenario.
    """
    return param_grids(EstimatorSpec(name="kmeans"), study.candidate_k) + param_grids(
        EstimatorSpec(name="spectral"), study.candidate_k
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_studies.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_studies.py tests/benchmarks/test_studies.py
git commit -m "feat(benchmarks): add case-study CVI sweep and CARVE fit cache"
```

---

### Task 8: Figures package and the S1 example figure

**Files:**
- Create: `src/benchmarks/figures/__init__.py`
- Create: `src/benchmarks/figures/_paths.py`
- Create: `src/benchmarks/figures/_benchmarking_examples.py`
- Create: `tests/benchmarks/figures/test_figure_contract.py`
- Create: `tests/benchmarks/figures/test_benchmarking_examples.py`
- Read for reference: `notebooks/benchmarking_code/benchmarking_plotting.py:444-553`

**Interfaces:**
- Consumes: `simulate` from plan 1 Task 7; `build_estimator` from plan 1 Task 5; `SCENARIOS` from plan 1 Task 6; `scatter_clusters`, `panel_letter` from Tasks 3 and 4; `theme_context`, `save_figure` from Task 2.
- Produces:
  - `VIS_ROOT: Path`, `BENCHMARKING_DIR: Path`, `CASE_STUDY_DIR: Path`
  - `figure_path(name: str, *, subdir: Path) -> Path`
  - `figure_benchmarking_examples(*, scenarios=None, seed=0, save=True, out_dir=None) -> Figure`
  - `benchmarks.figures.__all__` listing all eight figure functions

The contract test is written first and applies to every figure module added afterwards, so a later
task cannot quietly reintroduce a `plt.show()` or forget to save.

- [ ] **Step 1: Write the contract test**

Create `tests/benchmarks/figures/test_figure_contract.py`:

```python
"""Contract every manuscript figure function must satisfy."""

import inspect

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.figure import Figure

from benchmarks import figures

EXPECTED = (
    "figure_benchmarking_examples",
    "figure_benchmarking_results",
    "figure_scaling_ari",
    "figure_scaling_runtime",
    "figure_klein_results",
    "figure_levine_results",
    "figure_carve_output_klein",
    "figure_carve_output_levine",
)


def test_all_eight_manuscript_figures_are_exported():
    assert set(figures.__all__) >= set(EXPECTED)


@pytest.mark.parametrize("name", EXPECTED)
def test_every_figure_function_exists(name):
    assert callable(getattr(figures, name))


@pytest.mark.parametrize("name", EXPECTED)
def test_every_figure_function_returns_a_figure(name):
    signature = inspect.signature(getattr(figures, name))
    assert signature.return_annotation in (Figure, "Figure")


@pytest.mark.parametrize("name", EXPECTED)
def test_every_figure_function_accepts_save_and_out_dir(name):
    parameters = inspect.signature(getattr(figures, name)).parameters
    assert "save" in parameters
    assert "out_dir" in parameters


@pytest.mark.parametrize("name", EXPECTED)
def test_no_figure_module_calls_plt_show(name):
    source = inspect.getsource(inspect.getmodule(getattr(figures, name)))
    assert "plt.show(" not in source
```

- [ ] **Step 2: Write the S1 figure test**

Create `tests/benchmarks/figures/test_benchmarking_examples.py`:

```python
"""Tests for the S1 example-scatter figure."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from benchmarks.figures import figure_benchmarking_examples

SCENARIOS = ("gaussians", "moons")


class TestFigureBenchmarkingExamples:
    def test_returns_a_figure_without_saving(self, tmp_path):
        fig = figure_benchmarking_examples(scenarios=SCENARIOS, save=False)
        assert fig.get_axes()
        plt.close(fig)

    def test_one_axes_per_scenario_and_difficulty(self):
        fig = figure_benchmarking_examples(scenarios=SCENARIOS, save=False)
        assert len(fig.get_axes()) == len(SCENARIOS) * 3
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, tmp_path):
        fig = figure_benchmarking_examples(
            scenarios=SCENARIOS, save=True, out_dir=tmp_path
        )
        assert (tmp_path / "benchmarking_examples.png").exists()
        plt.close(fig)

    def test_uses_the_swiss_roll_estimator_the_benchmark_actually_ran(self):
        """S1 was drawn with spectral while the results came from agglomerative."""
        from benchmarks._registry import SCENARIOS as REGISTRY

        assert REGISTRY["swiss_rolls"].estimator.name == "agglomerative"

    def test_is_deterministic_for_a_fixed_seed(self):
        first = figure_benchmarking_examples(scenarios=("gaussians",), seed=3, save=False)
        second = figure_benchmarking_examples(scenarios=("gaussians",), seed=3, save=False)
        a = first.get_axes()[0].collections[0].get_offsets()
        b = second.get_axes()[0].collections[0].get_offsets()
        assert (a == b).all()
        plt.close(first)
        plt.close(second)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/benchmarks/figures/ -v`
Expected: FAIL with `ImportError: cannot import name 'figures'`

- [ ] **Step 4: Write the paths module**

Create `src/benchmarks/figures/_paths.py`:

```python
"""Where figures are written.

code/vis/ is gitignored. Copying a figure into overleaf/vis/ stays a manual
step, per the existing convention, so a regenerated figure never silently
changes the manuscript.
"""

from pathlib import Path

VIS_ROOT = Path("vis")
BENCHMARKING_DIR = VIS_ROOT / "benchmarking"
CASE_STUDY_DIR = VIS_ROOT / "case_studies"


def figure_path(name: str, *, subdir: Path, out_dir: Path | None = None) -> Path:
    """Resolve a figure's output path, honoring an explicit override."""
    base = Path(out_dir) if out_dir is not None else subdir
    return base / name
```

- [ ] **Step 5: Write the S1 figure module**

Create `src/benchmarks/figures/_benchmarking_examples.py`:

```python
"""S1 Fig: example scatters for each scenario at each difficulty.

One row per scenario, one column per difficulty anchor, each panel a PCA
projection of one simulated dataset colored by the true labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from .._panels import cluster_color_map, scatter_clusters
from .._registry import SCENARIOS
from .._simulate import simulate
from .._theme import FONT_SIZES, save_figure, theme_context
from ._paths import BENCHMARKING_DIR, figure_path

DEFAULT_SCENARIOS = (
    "gaussians",
    "t_dist",
    "t_dist_noise",
    "circles",
    "moons",
    "swiss_rolls",
)

SCENARIO_TITLES = {
    "gaussians": "Gaussian Mixtures",
    "t_dist": "t-Distributed",
    "t_dist_noise": "t-Distributed + Nuisance Dims",
    "circles": "RFF Circles",
    "moons": "RFF Moons",
    "swiss_rolls": "RFF Swiss Rolls",
}


def figure_benchmarking_examples(
    *,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
    seed: int = 0,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Build S1 Fig.

    Parameters
    ----------
    scenarios : tuple of str
        Registry keys, one row each.
    seed : int
        Seed index passed through the standard derivation, so the panels show
        the same data the benchmark scored.
    save : bool
        Write the file. False returns the figure without touching disk.
    out_dir : Path or None
        Override the destination directory.
    """
    with theme_context():
        n_rows = len(scenarios)
        fig, axes = plt.subplots(
            n_rows, 3, figsize=(11.0, 3.1 * n_rows), squeeze=False
        )

        for row, scenario_name in enumerate(scenarios):
            scenario = SCENARIOS[scenario_name]
            for axis_idx, axis_value, axis_label in scenario.axis:
                ax = axes[row, axis_idx]
                benchmark_seed = seed + (axis_idx * 10000)
                X, y = simulate(
                    scenario,
                    axis_value=axis_value,
                    axis_label=axis_label,
                    seed=benchmark_seed,
                )
                Z = PCA(n_components=2, random_state=0).fit_transform(np.asarray(X))
                scatter_clusters(
                    ax,
                    Z,
                    np.asarray(y),
                    color_map=cluster_color_map(np.asarray(y)),
                    s=6.0,
                    alpha=0.7,
                    title=axis_label.capitalize() if row == 0 else None,
                )
                if axis_idx == 0:
                    ax.set_ylabel(
                        SCENARIO_TITLES.get(scenario_name, scenario_name),
                        fontsize=FONT_SIZES["axis_label"],
                    )

        fig.tight_layout()

        if save:
            save_figure(
                fig,
                figure_path(
                    "benchmarking_examples.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig
```

- [ ] **Step 6: Write the package init**

Create `src/benchmarks/figures/__init__.py`:

```python
"""One module per manuscript figure.

Every function returns a Figure and, unless save=False, writes it under its
exact manuscript filename at 300 dpi. Four of these figures previously had no
savefig at all and were right-click-saved out of Jupyter output.
"""

from ._benchmarking_examples import figure_benchmarking_examples
from ._paths import BENCHMARKING_DIR, CASE_STUDY_DIR, VIS_ROOT, figure_path

__all__ = [
    "BENCHMARKING_DIR",
    "CASE_STUDY_DIR",
    "VIS_ROOT",
    "figure_benchmarking_examples",
    "figure_path",
]
```

The contract test will fail on the seven missing figure functions until Tasks 9 through 12 add
them. That is intended: it is the checklist. Run it filtered while working:

Run: `.venv/bin/pytest tests/benchmarks/figures/test_benchmarking_examples.py -v`
Expected: all passed

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/figures/ tests/benchmarks/figures/
git commit -m "feat(benchmarks): add the figures package and S1 example figure"
```

---

### Task 9: Fig 4, the main benchmarking results figure

**Files:**
- Create: `src/benchmarks/figures/_benchmarking_results.py`
- Create: `tests/benchmarks/figures/test_benchmarking_results.py`
- Modify: `src/benchmarks/figures/__init__.py`
- Read for reference: `notebooks/benchmarking_code/benchmarking_plotting.py:764-979,1084-1496`

**Interfaces:**
- Consumes: `metric_lines`, `grouped_legend`, `panel_letter` from Tasks 3 and 4; `read_run` from plan 1 Task 9; `theme_context`, `save_figure` from Task 2.
- Produces: `figure_benchmarking_results(results_by_scenario: Mapping[str, pd.DataFrame], *, metrics=None, k_star=5, save=True, out_dir=None) -> Figure`

This replaces `plot_paper_figure`, which is 408 lines with 19 parameters and re-simulates data
inside the plotting call. The rebuilt version takes the artifact frames it is drawing and does
nothing but lay them out, which is what makes it testable without running a benchmark.

`draw_grouped_legend` was extracted from `plot_paper_figure` and then never removed from it, so
the same legend code ran in two places. Only the extracted one survives.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/figures/test_benchmarking_results.py`:

```python
"""Tests for Fig 4."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarks._artifacts import SCHEMA
from benchmarks.figures import figure_benchmarking_results

METRICS = ("ari_stability_1se", "ari_generalizability_1se", "silhouette")


def _frame(scenario: str) -> pd.DataFrame:
    rows = []
    for axis_value, axis_label in enumerate(("easy", "medium", "hard")):
        for seed in range(4):
            for metric in METRICS:
                for k in (4, 5, 6):
                    rows.append(
                        {
                            "run_id": "r1",
                            "scenario": scenario,
                            "axis_name": "difficulty_level",
                            "axis_value": axis_value,
                            "axis_label": axis_label,
                            "seed": seed,
                            "k_star": 5,
                            "estimator": "kmeans",
                            "metric_name": metric,
                            "k": k,
                            "metric_value": 0.1 * k,
                            "is_selected": k == 5,
                            "selects_true_k": k == 5,
                            "ari_at_k": 0.9 - 0.1 * axis_value,
                            "oracle_ari": 0.95,
                        }
                    )
    return pd.DataFrame(rows)[list(SCHEMA)]


@pytest.fixture
def results():
    return {name: _frame(name) for name in ("gaussians", "t_dist", "moons")}


class TestFigureBenchmarkingResults:
    def test_returns_a_figure(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        assert fig.get_axes()
        plt.close(fig)

    def test_one_panel_per_scenario(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        drawn = [ax for ax in fig.get_axes() if ax.lines or ax.collections]
        assert len(drawn) == len(results)
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, results, tmp_path):
        fig = figure_benchmarking_results(
            results, metrics=METRICS, save=True, out_dir=tmp_path
        )
        assert (tmp_path / "benchmarking_results.png").exists()
        plt.close(fig)

    def test_carries_one_deduplicated_figure_legend(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        assert len(fig.legends) == 1
        assert len(fig.legends[0].get_texts()) == len(METRICS)
        plt.close(fig)

    def test_panels_are_lettered(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        letters = {t.get_text() for ax in fig.get_axes() for t in ax.texts}
        assert {"A", "B", "C"} <= letters
        plt.close(fig)

    def test_carve_stability_is_drawn_in_the_theme_green(self, results):
        from benchmarks._theme import METRIC_COLORS

        fig = figure_benchmarking_results(
            results, metrics=("ari_stability_1se",), save=False
        )
        colors = {ln.get_color() for ax in fig.get_axes() for ln in ax.lines}
        assert METRIC_COLORS["ari_stability_1se"] in colors
        plt.close(fig)

    def test_raises_on_an_empty_mapping(self):
        with pytest.raises(ValueError, match="at least one scenario"):
            figure_benchmarking_results({}, save=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/figures/test_benchmarking_results.py -v`
Expected: FAIL with `ImportError: cannot import name 'figure_benchmarking_results'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/figures/_benchmarking_results.py`:

```python
"""Fig 4: ARI against difficulty, one panel per simulated scenario.

Takes the artifact frames it draws. The function this replaces re-simulated
data inside the plotting call, which meant a figure could disagree with the
table beside it, and it took 19 parameters across 408 lines.
"""

import string
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._panels import grouped_legend, metric_lines, panel_letter
from .._theme import FONT_SIZES, save_figure, theme_context
from ._benchmarking_examples import SCENARIO_TITLES
from ._paths import BENCHMARKING_DIR, figure_path

DEFAULT_METRICS: tuple[str, ...] = (
    "ari_stability_1se",
    "ari_generalizability_1se",
    "silhouette",
    "gap",
    "davies_bouldin",
    "calinski_harabasz",
)

_DIFFICULTY_TICKS = ("easy", "medium", "hard")


def figure_benchmarking_results(
    results_by_scenario: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    k_star: int = 5,
    ncols: int = 3,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Build Fig 4.

    Parameters
    ----------
    results_by_scenario : mapping of scenario name to its artifact frame
        Frames follow the unified schema, so no axis-column sniffing is needed.
    metrics : sequence of str
        Metric names to draw, in legend order.
    k_star : int
        True cluster count, used only for the panel subtitle.
    ncols : int
        Panels per row.
    save, out_dir
        As in every figure function.
    """
    if not results_by_scenario:
        raise ValueError("Need at least one scenario to draw.")

    names = list(results_by_scenario)
    n_rows = -(-len(names) // ncols)

    with theme_context():
        fig, axes = plt.subplots(
            n_rows,
            ncols,
            figsize=(4.0 * ncols, 3.2 * n_rows),
            squeeze=False,
            sharey=True,
        )

        for index, name in enumerate(names):
            ax = axes[index // ncols, index % ncols]
            frame = results_by_scenario[name]
            metric_lines(
                ax,
                frame,
                metrics=metrics,
                x_col="axis_value",
                x_label="Difficulty",
                show_legend=False,
            )
            ax.set_xticks(sorted(frame["axis_value"].unique()))
            ax.set_xticklabels(_DIFFICULTY_TICKS[: frame["axis_value"].nunique()])
            ax.set_title(
                SCENARIO_TITLES.get(name, name), fontsize=FONT_SIZES["title"]
            )
            if index % ncols != 0:
                ax.set_ylabel("")
            panel_letter(ax, string.ascii_uppercase[index])

        for index in range(len(names), n_rows * ncols):
            axes[index // ncols, index % ncols].set_visible(False)

        fig.suptitle(f"$k^\\star = {k_star}$", fontsize=FONT_SIZES["title"], y=1.0)
        fig.tight_layout()
        grouped_legend(fig, axes, y_offset=0.06)

        if save:
            save_figure(
                fig,
                figure_path(
                    "benchmarking_results.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig
```

- [ ] **Step 4: Export it**

In `src/benchmarks/figures/__init__.py`, add the import and the `__all__` entry:

```python
from ._benchmarking_results import figure_benchmarking_results
```

```python
    "figure_benchmarking_results",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/figures/test_benchmarking_results.py -v`
Expected: all passed

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/figures/ tests/benchmarks/figures/test_benchmarking_results.py
git commit -m "feat(benchmarks): add Fig 4 from artifacts instead of re-simulation"
```

---

### Task 10: The two scaling figures

**Files:**
- Create: `src/benchmarks/figures/_scaling.py`
- Create: `tests/benchmarks/figures/test_scaling.py`
- Modify: `src/benchmarks/figures/__init__.py`
- Read for reference: `notebooks/Benchmarking.ipynb` cell 60

**Interfaces:**
- Consumes: `metric_lines`, `runtime_lines`, `grouped_legend` from Tasks 3 and 4; `read_run`, `read_runtimes` from plan 1 Task 9 and Task 1; `theme_context`, `save_figure` from Task 2.
- Produces:
  - `SCALING_PANELS: tuple[tuple[str, str], ...]` — `(scenario_name, x_label)` pairs
  - `figure_scaling_ari(results_by_scenario, *, metrics=..., k_star=5, save=True, out_dir=None) -> Figure`
  - `figure_scaling_runtime(runtimes_by_scenario, *, k_star=5, save=True, out_dir=None) -> Figure`

The 118-line notebook cell this replaces carried its own geometry constants, its own
`_scaling_figsize` and `_scaling_adjust` helpers, and `ELEMENT_SCALE = 0.47`, a fudge factor whose
comment says it exists "to scale line/marker widths to match Fig 4 apparent thickness". That
mismatch existed because the two figures used different font and element sizes; with one theme
they match by construction, so `element_scale` defaults to 1.0 and the fudge is gone.

The commented-out moons panels in that cell are not ported. The moons scaling experiment is
dropped.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/figures/test_scaling.py`:

```python
"""Tests for the two scaling figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarks._artifacts import RUNTIME_SCHEMA, SCHEMA
from benchmarks.figures import figure_scaling_ari, figure_scaling_runtime

METRICS = ("ari_stability_1se", "ari_generalizability_1se")


def _results(scenario, axis_name, values):
    rows = []
    for axis_value, axis_label in zip(values, ("start", "middle", "end")):
        for seed in range(3):
            for metric in METRICS:
                for k in (4, 5):
                    rows.append({
                        "run_id": "r1", "scenario": scenario, "axis_name": axis_name,
                        "axis_value": axis_value, "axis_label": axis_label, "seed": seed,
                        "k_star": 5, "estimator": "kmeans", "metric_name": metric, "k": k,
                        "metric_value": 0.1 * k, "is_selected": k == 5,
                        "selects_true_k": k == 5, "ari_at_k": 0.8, "oracle_ari": 0.9,
                    })
    return pd.DataFrame(rows)[list(SCHEMA)]


def _runtimes(scenario, axis_name, values):
    rows = []
    for axis_value, axis_label in zip(values, ("start", "middle", "end")):
        for seed in range(3):
            rows.append({
                "run_id": "r1", "scenario": scenario, "axis_name": axis_name,
                "axis_value": axis_value, "axis_label": axis_label, "seed": seed,
                "n_samples": int(axis_value), "n_features": 50, "n_resamples": 100,
                "n_jobs": 1, "estimator": "kmeans",
                "t_default_s": 1.5 + axis_value / 800.0,
                "t_stability_s": 1.0 + axis_value / 1000.0,
                "t_generalizability_s": 1.2 + axis_value / 900.0,
                "t_per_k_stability_s": (1.0 + axis_value / 1000.0) / 5,
                "t_per_k_generalizability_s": (1.2 + axis_value / 900.0) / 5,
            })
    return pd.DataFrame(rows)[list(RUNTIME_SCHEMA)]


@pytest.fixture
def results():
    return {
        "gaussians_samples": _results("gaussians_samples", "n_total", (1000, 5500, 10000)),
        "gaussians_dimensionality": _results(
            "gaussians_dimensionality", "embed_dim", (10, 255, 500)
        ),
    }


@pytest.fixture
def runtimes():
    return {
        "gaussians_samples": _runtimes("gaussians_samples", "n_total", (1000, 5500, 10000)),
        "gaussians_dimensionality": _runtimes(
            "gaussians_dimensionality", "embed_dim", (10, 255, 500)
        ),
    }


class TestFigureScalingAri:
    def test_returns_a_figure_with_one_panel_per_scenario(self, results):
        fig = figure_scaling_ari(results, metrics=METRICS, save=False)
        drawn = [ax for ax in fig.get_axes() if ax.lines]
        assert len(drawn) == 2
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, results, tmp_path):
        fig = figure_scaling_ari(results, metrics=METRICS, save=True, out_dir=tmp_path)
        assert (tmp_path / "paper_fig_scaling_ari_k5.png").exists()
        plt.close(fig)

    def test_filename_carries_k_star(self, results, tmp_path):
        fig = figure_scaling_ari(
            results, metrics=METRICS, k_star=7, save=True, out_dir=tmp_path
        )
        assert (tmp_path / "paper_fig_scaling_ari_k7.png").exists()
        plt.close(fig)

    def test_x_axis_labels_name_the_swept_quantity(self, results):
        fig = figure_scaling_ari(results, metrics=METRICS, save=False)
        labels = {ax.get_xlabel() for ax in fig.get_axes()}
        assert "Number of samples (n)" in labels
        plt.close(fig)

    def test_has_one_deduplicated_legend(self, results):
        fig = figure_scaling_ari(results, metrics=METRICS, save=False)
        assert len(fig.legends) == 1
        assert len(fig.legends[0].get_texts()) == len(METRICS)
        plt.close(fig)

    def test_no_element_scale_fudge_is_applied_by_default(self, results):
        """One theme means Fig 4 and this figure match without a fudge factor."""
        import inspect

        from benchmarks.figures import _scaling

        source = inspect.getsource(_scaling)
        assert "0.47" not in source


class TestFigureScalingRuntime:
    def test_returns_a_figure_with_a_log_y_axis(self, runtimes):
        fig = figure_scaling_runtime(runtimes, save=False)
        drawn = [ax for ax in fig.get_axes() if ax.lines]
        assert drawn
        assert all(ax.get_yscale() == "log" for ax in drawn)
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, runtimes, tmp_path):
        fig = figure_scaling_runtime(runtimes, save=True, out_dir=tmp_path)
        assert (tmp_path / "paper_fig_scaling_runtime_k5.png").exists()
        plt.close(fig)

    def test_y_axis_is_labeled_in_seconds(self, runtimes):
        fig = figure_scaling_runtime(runtimes, save=False)
        assert any("second" in ax.get_ylabel().lower() for ax in fig.get_axes())
        plt.close(fig)

    def test_draws_two_curves_per_panel_like_the_published_figure(self, runtimes):
        fig = figure_scaling_runtime(runtimes, save=False)
        drawn = [ax for ax in fig.get_axes() if ax.lines]
        for ax in drawn:
            labelled = [ln for ln in ax.lines if not ln.get_label().startswith("_")]
            assert len(labelled) == 2
        plt.close(fig)

    def test_legend_names_both_carve_modes(self, runtimes):
        fig = figure_scaling_runtime(runtimes, save=False)
        labels = {t.get_text() for t in fig.legends[0].get_texts()}
        assert labels == {"CARVE Stability", "CARVE Generalizability"}
        plt.close(fig)

    def test_raises_on_an_empty_mapping(self):
        with pytest.raises(ValueError, match="at least one scenario"):
            figure_scaling_runtime({}, save=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/figures/test_scaling.py -v`
Expected: FAIL with `ImportError: cannot import name 'figure_scaling_ari'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/figures/_scaling.py`:

```python
"""The two scaling figures: ARI and runtime against the swept axis.

The notebook cell this replaces carried its own geometry constants and an
ELEMENT_SCALE = 0.47 fudge whose only purpose was to make line and marker
widths match Fig 4 by eye. With a single theme they match by construction, so
the fudge is gone.

The moons panels commented out in that cell are not ported; the moons scaling
experiment is dropped.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._panels import grouped_legend, metric_lines, runtime_lines
from .._theme import save_figure, theme_context
from ._paths import BENCHMARKING_DIR, figure_path

SCALING_PANELS: tuple[tuple[str, str], ...] = (
    ("gaussians_samples", "Number of samples (n)"),
    ("gaussians_dimensionality", "Number of features (p)"),
)

DEFAULT_METRICS: tuple[str, ...] = (
    "ari_stability_1se",
    "ari_generalizability_1se",
)


def _panel_grid(n_panels: int) -> tuple[Figure, list]:
    """One row of panels sized from the theme, not from hand-tuned margins."""
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.2 * n_panels, 3.0), squeeze=False
    )
    return fig, axes


def _ordered_panels(available: Mapping[str, pd.DataFrame]) -> list[tuple[str, str]]:
    panels = [(name, label) for name, label in SCALING_PANELS if name in available]
    if not panels:
        raise ValueError(
            "Need at least one scenario to draw. Expected keys from "
            f"{[name for name, _ in SCALING_PANELS]}, got {sorted(available)}."
        )
    return panels


def figure_scaling_ari(
    results_by_scenario: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    k_star: int = 5,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """ARI at the selected k against each scaling axis."""
    panels = _ordered_panels(results_by_scenario)

    with theme_context():
        fig, axes = _panel_grid(len(panels))
        for index, (name, x_label) in enumerate(panels):
            ax = axes[0, index]
            metric_lines(
                ax,
                results_by_scenario[name],
                metrics=metrics,
                x_col="axis_value",
                x_label=x_label,
                show_legend=False,
            )
            if index != 0:
                ax.set_ylabel("")

        fig.tight_layout()
        grouped_legend(fig, axes, y_offset=0.10)

        if save:
            save_figure(
                fig,
                figure_path(
                    f"paper_fig_scaling_ari_k{k_star}.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig


def figure_scaling_runtime(
    runtimes_by_scenario: Mapping[str, pd.DataFrame],
    *,
    k_star: int = 5,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """CARVE wall-clock against each scaling axis, on a log y axis.

    Two curves per panel, one per timed CARVE mode, matching the published
    figure. Those timings come from the extra mode-specific fits the runner
    performs for the scenarios listed in TIMED_SCENARIOS; a scenario without
    them records nan and its series is skipped rather than drawn at zero.
    """
    panels = _ordered_panels(runtimes_by_scenario)

    with theme_context():
        fig, axes = _panel_grid(len(panels))
        for index, (name, x_label) in enumerate(panels):
            ax = axes[0, index]
            runtime_lines(
                ax,
                runtimes_by_scenario[name],
                x_col="axis_value",
                x_label=x_label,
                yscale="log",
                show_legend=False,
            )
            if index != 0:
                ax.set_ylabel("")

        fig.tight_layout()
        grouped_legend(fig, axes, y_offset=0.10)

        if save:
            save_figure(
                fig,
                figure_path(
                    f"paper_fig_scaling_runtime_k{k_star}.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig
```

- [ ] **Step 4: Export both**

In `src/benchmarks/figures/__init__.py`:

```python
from ._scaling import figure_scaling_ari, figure_scaling_runtime
```

```python
    "figure_scaling_ari",
    "figure_scaling_runtime",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/figures/test_scaling.py -v`
Expected: all passed

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/figures/ tests/benchmarks/figures/test_scaling.py
git commit -m "feat(benchmarks): add the scaling figures and drop the element-scale fudge"
```

---

### Task 11: The shared case-study composite, Fig 5 and Fig 6

**Files:**
- Create: `src/benchmarks/figures/_case_study.py`
- Create: `src/benchmarks/figures/_klein_results.py`
- Create: `src/benchmarks/figures/_levine_results.py`
- Create: `tests/benchmarks/figures/test_case_study.py`
- Modify: `src/benchmarks/figures/__init__.py`
- Read for reference: `notebooks/case_studies/Klein.ipynb` cell 32, `notebooks/case_studies/Levine_32dim.ipynb` cell 23

**Interfaces:**
- Consumes: `scatter_clusters`, `carve_lines`, `cvi_lines`, `alluvial`, `ari_lollipop`, `panel_letter`, `cluster_color_map` from Tasks 3 and 4; `theme_context`, `save_figure` from Task 2.
- Produces:
  - `CompositeInputs` — a frozen dataclass holding `X`, `y`, `Z`, `carve`, `carve_labels`, `comparison_labels`, `comparison_name`, `comparison_k`, `curves_df`, `best_df`
  - `prepare_composite(X, y, carve, *, curves_df, best_df, comparison_metric="silhouette", embedding=None, measure="stability", rule="1se", not_two=False, random_state=42) -> CompositeInputs`
  - `composite_figure(inputs, *, bottom_panel, marker_size, axis_labels, save_name, out_dir, save) -> Figure`
  - `figure_klein_results(inputs, *, save=True, out_dir=None) -> Figure`
  - `figure_levine_results(inputs, *, save=True, out_dir=None) -> Figure`

The two 178-line notebook cells this replaces are identical except for four things: marker size
(20.0 versus 8.0), the embedding (PCA computed internally versus a precomputed t-SNE), the axis
labels, and panel F (an alluvial versus an ARI lollipop). Those four become parameters and the
other 174 lines become one function.

Two defects in the original are not carried over. The Klein cell assigns `leg_D` twice, so the
first legend is built and immediately discarded. And both cells end in `plt.show()` with no
`savefig`, which is why these two figures were right-click-saved out of Jupyter.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/figures/test_case_study.py`:

```python
"""Tests for the shared case-study composite and its two instantiations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from benchmarks.figures import (
    figure_klein_results,
    figure_levine_results,
)
from benchmarks.figures._case_study import CompositeInputs


class _StubCarve:
    """Minimal stand-in for a fitted CARVE, so figure tests need no fit."""

    def __init__(self, ks):
        self.estimator_results_ = pd.DataFrame(
            {
                "n_clusters": list(ks),
                "stability": np.linspace(0.5, 0.9, len(ks)),
                "generalizability": np.linspace(0.4, 0.85, len(ks)),
                "stability_se": np.full(len(ks), 0.02),
                "generalizability_se": np.full(len(ks), 0.03),
            }
        )
        self._ks = list(ks)

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return self._ks[len(self._ks) // 2]


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 5))
    Z = rng.normal(size=(n, 2))
    y = pd.Series(rng.choice(["a", "b", "c"], size=n))
    carve_labels = rng.integers(0, 3, size=n)
    comparison_labels = rng.integers(0, 2, size=n)
    curves = pd.DataFrame(
        {
            "metric": ["silhouette"] * 3 + ["gap"] * 3,
            "model": ["KMeans"] * 6,
            "k": [3, 4, 5] * 2,
            "score": [0.4, 0.6, 0.5, 0.2, 0.3, 0.35],
            "ari": [0.5] * 6,
        }
    )
    best = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["KMeans", "KMeans"],
            "k": [4, 5],
            "score": [0.6, 0.35],
            "ari": [0.5, 0.5],
        }
    )
    return CompositeInputs(
        X=X,
        y=y.to_numpy(),
        Z=Z,
        carve=_StubCarve([3, 4, 5]),
        carve_labels=carve_labels,
        comparison_labels=comparison_labels,
        comparison_name="Silhouette",
        comparison_k=4,
        curves_df=curves,
        best_df=best,
    )


class TestKleinFigure:
    def test_returns_a_figure_with_six_drawn_panels(self, inputs):
        fig = figure_klein_results(inputs, save=False)
        lettered = {t.get_text() for ax in fig.get_axes() for t in ax.texts}
        assert {"A", "B", "C", "D", "E", "F"} <= lettered
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, inputs, tmp_path):
        fig = figure_klein_results(inputs, save=True, out_dir=tmp_path)
        assert (tmp_path / "klein_results.png").exists()
        plt.close(fig)

    def test_builds_exactly_two_panel_legends(self, inputs):
        """The original built legend D twice and discarded the first."""
        fig = figure_klein_results(inputs, save=False)
        legend_axes = [ax for ax in fig.get_axes() if ax.get_legend() is not None]
        assert len(legend_axes) == 2
        plt.close(fig)

    def test_bottom_panel_is_the_alluvial(self, inputs):
        fig = figure_klein_results(inputs, save=False)
        # The alluvial turns its axes off and carries three column titles.
        alluvial_axes = [ax for ax in fig.get_axes() if not ax.axison and ax.patches]
        assert alluvial_axes
        plt.close(fig)


class TestLevineFigure:
    def test_saves_under_the_manuscript_filename(self, inputs, tmp_path):
        fig = figure_levine_results(inputs, save=True, out_dir=tmp_path)
        assert (tmp_path / "levine_results.png").exists()
        plt.close(fig)

    def test_bottom_panel_is_the_ari_lollipop(self, inputs):
        fig = figure_levine_results(inputs, save=False)
        labels = [ax.get_xlabel() for ax in fig.get_axes()]
        assert any("ARI" in label for label in labels)
        plt.close(fig)

    def test_uses_the_smaller_marker_size(self, inputs):
        klein = figure_klein_results(inputs, save=False)
        levine = figure_levine_results(inputs, save=False)
        klein_sizes = klein.get_axes()[0].collections[0].get_sizes()
        levine_sizes = levine.get_axes()[0].collections[0].get_sizes()
        assert levine_sizes[0] < klein_sizes[0]
        plt.close(klein)
        plt.close(levine)


class TestSharedComposite:
    def test_both_figures_come_from_one_builder(self):
        import inspect

        from benchmarks.figures import _klein_results, _levine_results

        for module in (_klein_results, _levine_results):
            assert "composite_figure" in inspect.getsource(module)

    def test_neither_module_lays_out_its_own_gridspec(self):
        import inspect

        from benchmarks.figures import _klein_results, _levine_results

        for module in (_klein_results, _levine_results):
            assert "add_gridspec" not in inspect.getsource(module)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/figures/test_case_study.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.figures._case_study'`

- [ ] **Step 3: Write the shared composite**

Create `src/benchmarks/figures/_case_study.py`:

```python
"""The composite layout shared by Fig 5 and Fig 6.

The two notebook cells this replaces were 178 lines each and differed in four
things: marker size, the embedding, the axis labels, and panel F. Those are
parameters here; the remaining 174 lines are shared.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._panels import (
    carve_lines,
    cluster_color_map,
    cvi_lines,
    panel_letter,
    scatter_clusters,
)
from .._theme import FONT_SIZES, save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path


@dataclass(frozen=True)
class CompositeInputs:
    """Everything the composite draws, already computed.

    Assembling this is compute; drawing it is reporting. Keeping them apart is
    what lets the figure be tested without fitting anything.
    """

    X: np.ndarray
    y: np.ndarray
    Z: np.ndarray
    carve: Any
    carve_labels: np.ndarray
    comparison_labels: np.ndarray
    comparison_name: str
    comparison_k: int
    curves_df: pd.DataFrame
    best_df: pd.DataFrame


def prepare_composite(
    X: np.ndarray,
    y: np.ndarray,
    carve: Any,
    *,
    curves_df: pd.DataFrame,
    best_df: pd.DataFrame,
    comparison_metric: str = "silhouette",
    embedding: np.ndarray | None = None,
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
    random_state: int = 42,
) -> CompositeInputs:
    """Assemble the composite's inputs, computing a PCA embedding if needed."""
    from sklearn.decomposition import PCA

    from .._estimators import build_estimator
    from .._types import EstimatorSpec

    X = np.asarray(X)
    y = np.asarray(y)

    Z = (
        np.asarray(embedding)
        if embedding is not None
        else PCA(n_components=2, random_state=random_state).fit_transform(X)
    )

    carve_labels = np.asarray(
        carve.get_labels(measure=measure, rule=rule, not_two=not_two)
    )

    best_row = best_df.loc[best_df["metric"] == comparison_metric]
    if best_row.empty:
        raise ValueError(
            f"No row for {comparison_metric!r} in best_df; "
            f"available metrics are {sorted(best_df['metric'].unique())}."
        )
    comparison_k = int(best_row["k"].iloc[0])
    estimator = build_estimator(
        EstimatorSpec(name="kmeans"),
        n_clusters=comparison_k,
        random_state=random_state,
    )
    comparison_labels = np.asarray(estimator.fit_predict(X))

    return CompositeInputs(
        X=X,
        y=y,
        Z=Z,
        carve=carve,
        carve_labels=carve_labels,
        comparison_labels=comparison_labels,
        comparison_name=comparison_metric.title(),
        comparison_k=comparison_k,
        curves_df=curves_df,
        best_df=best_df,
    )


def composite_figure(
    inputs: CompositeInputs,
    *,
    bottom_panel: Callable[[Axes, CompositeInputs], Axes],
    marker_size: float,
    axis_labels: Sequence[str],
    save_name: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Draw the six-panel case-study composite.

    Panels A, B and C are scatters of the reported labels, the CARVE
    clustering, and the CVI clustering; D and E are the CARVE and CVI curves
    over k; F is supplied by the caller.
    """
    true_cmap = cluster_color_map(inputs.y)
    carve_cmap = cluster_color_map(inputs.carve_labels)
    comparison_cmap = cluster_color_map(inputs.comparison_labels)

    with theme_context():
        fig = plt.figure(figsize=(16, 16), constrained_layout=False)
        gs = fig.add_gridspec(
            4, 6, height_ratios=[1.0, 1.0, 0.3, 1.0], hspace=0.2, wspace=0.8
        )
        ax_a = fig.add_subplot(gs[0, 0:2])
        ax_b = fig.add_subplot(gs[0, 2:4])
        ax_c = fig.add_subplot(gs[0, 4:6])
        ax_d = fig.add_subplot(gs[1, 0:3])
        ax_e = fig.add_subplot(gs[1, 3:6])
        ax_f = fig.add_subplot(gs[3, 1:5])

        legend_ax_d = fig.add_subplot(gs[2, 0:3])
        legend_ax_d.set_axis_off()
        legend_ax_e = fig.add_subplot(gs[2, 3:6])
        legend_ax_e.set_axis_off()

        scatter_clusters(
            ax_a, inputs.Z, inputs.y, color_map=true_cmap,
            s=marker_size, title="Reported Labels", axis_labels=axis_labels,
        )
        scatter_clusters(
            ax_b, inputs.Z, inputs.carve_labels, color_map=carve_cmap,
            s=marker_size, title="CARVE clustering", axis_labels=axis_labels,
        )
        scatter_clusters(
            ax_c, inputs.Z, inputs.comparison_labels, color_map=comparison_cmap,
            s=marker_size,
            title=f"CVI ({inputs.comparison_name}, k={inputs.comparison_k})",
            axis_labels=axis_labels,
        )

        carve_lines(ax_d, inputs.carve, title="CARVE ARI over k")
        cvi_lines(ax_e, inputs.curves_df, inputs.best_df, title="CVIs over k")

        bottom_panel(ax_f, inputs)

        # Move the two panel legends into their own strip. Building each one
        # exactly once; the cell this replaces assigned leg_D twice and threw
        # the first away.
        for source_ax, target_ax, title in (
            (ax_d, legend_ax_d, "CARVE"),
            (ax_e, legend_ax_e, "CVIs"),
        ):
            handles, labels = source_ax.get_legend_handles_labels()
            existing = source_ax.get_legend()
            if existing is not None:
                existing.remove()
            target_ax.legend(
                handles, labels, title=title, loc="center", ncol=1,
                frameon=False, fontsize=FONT_SIZES["legend"],
            )

        for letter, ax in zip("ABCDEF", (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f)):
            panel_letter(ax, letter)

        if save:
            save_figure(
                fig,
                figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir),
            )
    return fig
```

- [ ] **Step 4: Write the two instantiations**

Create `src/benchmarks/figures/_klein_results.py`:

```python
"""Fig 5: the Klein case study.

Panel F is an alluvial linking the CARVE clustering, the reported labels, and
the CVI clustering. Panels A to C use a PCA embedding and 20-point markers.
"""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._panels import alluvial, cluster_color_map
from ._case_study import CompositeInputs, composite_figure

MARKER_SIZE = 20.0
AXIS_LABELS = ("PC1", "PC2")


def _alluvial_panel(ax: Axes, inputs: CompositeInputs) -> Axes:
    return alluvial(
        ax,
        inputs.y,
        inputs.carve_labels,
        inputs.comparison_labels,
        left_cmap=cluster_color_map(inputs.carve_labels),
        right_cmap=cluster_color_map(inputs.comparison_labels),
        true_cmap=cluster_color_map(inputs.y),
        left_title="CARVE",
        right_title=f"CVI ({inputs.comparison_name})",
        true_title="Reported Label",
    )


def figure_klein_results(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build Fig 5."""
    return composite_figure(
        inputs,
        bottom_panel=_alluvial_panel,
        marker_size=MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="klein_results.png",
        save=save,
        out_dir=out_dir,
    )
```

Create `src/benchmarks/figures/_levine_results.py`:

```python
"""Fig 6: the Levine 32-dimension case study.

Panel F is an ARI lollipop against the reported labels. Panels A to C use the
precomputed t-SNE embedding carried on CompositeInputs.Z and 8-point markers,
because the dataset has an order of magnitude more cells than Klein.
"""

from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import adjusted_rand_score

from .._panels import ari_lollipop
from ._case_study import CompositeInputs, composite_figure

MARKER_SIZE = 8.0
AXIS_LABELS = ("t-SNE 1", "t-SNE 2")


def _ari_table(inputs: CompositeInputs) -> pd.DataFrame:
    """ARI of each selection against the reported labels."""
    rows = [
        {
            "method": "CARVE",
            "metric": "ari_stability_1se",
            "ari": float(adjusted_rand_score(inputs.y, inputs.carve_labels)),
            "k": int(len(set(inputs.carve_labels.tolist()))),
        }
    ]
    for _, row in inputs.best_df.iterrows():
        rows.append(
            {
                "method": str(row["metric"]).replace("_", " ").title(),
                "metric": str(row["metric"]),
                "ari": float(row["ari"]),
                "k": int(row["k"]),
            }
        )
    return pd.DataFrame(rows)


def _ari_panel(ax: Axes, inputs: CompositeInputs) -> Axes:
    return ari_lollipop(
        ax, _ari_table(inputs), title="Agreement with Reported Labels (ARI)"
    )


def figure_levine_results(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build Fig 6."""
    return composite_figure(
        inputs,
        bottom_panel=_ari_panel,
        marker_size=MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="levine_results.png",
        save=save,
        out_dir=out_dir,
    )
```

- [ ] **Step 5: Export both**

In `src/benchmarks/figures/__init__.py`:

```python
from ._case_study import CompositeInputs, composite_figure, prepare_composite
from ._klein_results import figure_klein_results
from ._levine_results import figure_levine_results
```

```python
    "CompositeInputs",
    "composite_figure",
    "prepare_composite",
    "figure_klein_results",
    "figure_levine_results",
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/figures/test_case_study.py -v`
Expected: all passed

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/figures/ tests/benchmarks/figures/test_case_study.py
git commit -m "feat(benchmarks): collapse the two case-study cells into one composite"
```

---

### Task 12: The two CARVE-output figures

**Files:**
- Create: `src/benchmarks/figures/_carve_output.py`
- Create: `tests/benchmarks/figures/test_carve_output.py`
- Modify: `src/benchmarks/figures/__init__.py`
- Read for reference: `notebooks/case_studies/Klein.ipynb` cell 29, `notebooks/case_studies/Levine_32dim.ipynb` cell 20

**Interfaces:**
- Consumes: `carve_lines`, `scatter_clusters`, `panel_letter`, `cluster_color_map` from Tasks 3 and 4; `theme_context`, `save_figure` from Task 2; `CompositeInputs` from Task 11.
- Produces:
  - `carve_output_figure(inputs, *, marker_size, axis_labels, save_name, save, out_dir) -> Figure`
  - `figure_carve_output_klein(inputs, *, save=True, out_dir=None) -> Figure`
  - `figure_carve_output_levine(inputs, *, save=True, out_dir=None) -> Figure`

These are Fig 3 and S4 Fig, the two figures showing CARVE's own diagnostic output: the validation
curves, the consensus matrix, and the per-sample stability distribution. Both were
right-click-saved from Jupyter, so this is where that stops.

Consensus matrices come off the fitted object as `carve.consensus_matrices_[config_id]`. Retrieve
the id through the join key, never by position: `config_id` links `estimator_results_` rows to the
matrices, and `fit()` raises `RuntimeError("config_id is misaligned ...")` if that invariant
breaks. Selecting a row with a pandas index label and then using it to index the matrices
positionally is the specific mistake to avoid.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/figures/test_carve_output.py`:

```python
"""Tests for the two CARVE-output figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from benchmarks.figures import figure_carve_output_klein, figure_carve_output_levine
from benchmarks.figures._case_study import CompositeInputs


class _StubCarve:
    def __init__(self, ks, n):
        rng = np.random.default_rng(0)
        # config_id is a join key, deliberately not 0..n-1, so a positional
        # lookup would pick the wrong matrix and the test would catch it.
        self.estimator_results_ = pd.DataFrame(
            {
                "config_id": [100 + i for i in range(len(ks))],
                "n_clusters": list(ks),
                "stability": np.linspace(0.5, 0.9, len(ks)),
                "generalizability": np.linspace(0.4, 0.85, len(ks)),
                "stability_se": np.full(len(ks), 0.02),
                "generalizability_se": np.full(len(ks), 0.03),
            }
        )
        base = rng.random((n, n))
        self.consensus_matrices_ = {
            100 + i: (base + base.T) / 2 for i in range(len(ks))
        }
        self.stability_gini_scores_ = {
            100 + i: rng.random(n) for i in range(len(ks))
        }
        self._ks = list(ks)

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return self._ks[len(self._ks) // 2]


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    n = 60
    return CompositeInputs(
        X=rng.normal(size=(n, 4)),
        y=rng.choice(["a", "b"], size=n),
        Z=rng.normal(size=(n, 2)),
        carve=_StubCarve([3, 4, 5], n),
        carve_labels=rng.integers(0, 3, size=n),
        comparison_labels=rng.integers(0, 2, size=n),
        comparison_name="Silhouette",
        comparison_k=4,
        curves_df=pd.DataFrame(
            {"metric": ["silhouette"], "model": ["KMeans"], "k": [4],
             "score": [0.6], "ari": [0.5]}
        ),
        best_df=pd.DataFrame(
            {"metric": ["silhouette"], "model": ["KMeans"], "k": [4],
             "score": [0.6], "ari": [0.5]}
        ),
    )


class TestCarveOutputFigures:
    def test_klein_saves_under_the_manuscript_filename(self, inputs, tmp_path):
        fig = figure_carve_output_klein(inputs, save=True, out_dir=tmp_path)
        assert (tmp_path / "CARVE_output_klein.png").exists()
        plt.close(fig)

    def test_levine_saves_under_the_manuscript_filename(self, inputs, tmp_path):
        fig = figure_carve_output_levine(inputs, save=True, out_dir=tmp_path)
        assert (tmp_path / "CARVE_output_levine.png").exists()
        plt.close(fig)

    def test_draws_three_lettered_panels(self, inputs):
        fig = figure_carve_output_klein(inputs, save=False)
        letters = {t.get_text() for ax in fig.get_axes() for t in ax.texts}
        assert {"A", "B", "C"} <= letters
        plt.close(fig)

    def test_consensus_matrix_is_rendered_as_an_image(self, inputs):
        fig = figure_carve_output_klein(inputs, save=False)
        assert any(ax.images for ax in fig.get_axes())
        plt.close(fig)

    def test_consensus_matrix_is_looked_up_by_join_key_not_position(self, inputs):
        """config_id is 100.. here, so a positional lookup would raise."""
        fig = figure_carve_output_klein(inputs, save=False)
        assert fig.get_axes()
        plt.close(fig)

    def test_levine_uses_the_smaller_marker_size(self, inputs):
        klein = figure_carve_output_klein(inputs, save=False)
        levine = figure_carve_output_levine(inputs, save=False)
        klein_scatter = next(ax for ax in klein.get_axes() if ax.collections)
        levine_scatter = next(ax for ax in levine.get_axes() if ax.collections)
        assert (
            levine_scatter.collections[0].get_sizes()[0]
            < klein_scatter.collections[0].get_sizes()[0]
        )
        plt.close(klein)
        plt.close(levine)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/figures/test_carve_output.py -v`
Expected: FAIL with `ImportError: cannot import name 'figure_carve_output_klein'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/figures/_carve_output.py`:

```python
"""Fig 3 and S4 Fig: CARVE's own diagnostic output on a case study.

Panel A is the validation curves over k, B the consensus matrix at the
selected k, C the per-sample stability distribution. Both figures were
previously right-click-saved out of Jupyter with no savefig anywhere.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .._panels import carve_lines, cluster_color_map, panel_letter, scatter_clusters
from .._theme import FONT_SIZES, save_figure, theme_context
from ._case_study import CompositeInputs
from ._paths import CASE_STUDY_DIR, figure_path


def _config_id_at_k(carve: object, k: int) -> int:
    """Look up the config_id for one k.

    config_id is a join key linking estimator_results_ rows to the consensus
    matrices, never a positional index. Selecting a row and using its pandas
    label to index the matrices positionally is the mistake this guards
    against.
    """
    results = carve.estimator_results_
    match = results.loc[results["n_clusters"] == k, "config_id"]
    if match.empty:
        raise ValueError(
            f"No configuration at k={k}; available: "
            f"{sorted(results['n_clusters'].unique())}."
        )
    return int(match.iloc[0])


def carve_output_figure(
    inputs: CompositeInputs,
    *,
    marker_size: float,
    axis_labels: Sequence[str],
    save_name: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Draw the three-panel CARVE diagnostic figure."""
    carve = inputs.carve
    selected_k = int(carve.get_k(measure="stability", rule="1se"))
    config_id = _config_id_at_k(carve, selected_k)

    with theme_context():
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
        ax_a, ax_b, ax_c = axes

        carve_lines(ax_a, carve, title="Validation over $k$")

        matrix = np.asarray(carve.consensus_matrices_[config_id], dtype=float)
        image = ax_b.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
        ax_b.set_title(
            f"Consensus matrix ($k={selected_k}$)", fontsize=FONT_SIZES["title"]
        )
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        fig.colorbar(image, ax=ax_b, fraction=0.046, pad=0.04)

        gini = np.asarray(carve.stability_gini_scores_[config_id], dtype=float)
        scatter_clusters(
            ax_c,
            inputs.Z,
            inputs.carve_labels,
            color_map=cluster_color_map(inputs.carve_labels),
            s=marker_size,
            title="Sample-level stability",
            axis_labels=axis_labels,
        )
        ax_c.set_xlabel(
            f"median Gini = {np.median(gini):.3f}", fontsize=FONT_SIZES["axis_label"]
        )

        for letter, ax in zip("ABC", (ax_a, ax_b, ax_c)):
            panel_letter(ax, letter)

        fig.tight_layout()

        if save:
            save_figure(
                fig,
                figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir),
            )
    return fig


def figure_carve_output_klein(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build Fig 3."""
    return carve_output_figure(
        inputs,
        marker_size=20.0,
        axis_labels=("PC1", "PC2"),
        save_name="CARVE_output_klein.png",
        save=save,
        out_dir=out_dir,
    )


def figure_carve_output_levine(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build S4 Fig."""
    return carve_output_figure(
        inputs,
        marker_size=8.0,
        axis_labels=("t-SNE 1", "t-SNE 2"),
        save_name="CARVE_output_levine.png",
        save=save,
        out_dir=out_dir,
    )
```

- [ ] **Step 4: Export both and run the full contract test**

In `src/benchmarks/figures/__init__.py`:

```python
from ._carve_output import figure_carve_output_klein, figure_carve_output_levine
```

```python
    "figure_carve_output_klein",
    "figure_carve_output_levine",
```

Run: `.venv/bin/pytest tests/benchmarks/figures/ -v`
Expected: all passed, including `test_figure_contract.py`, which now finds all eight functions.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/figures/ tests/benchmarks/figures/test_carve_output.py
git commit -m "feat(benchmarks): add the two CARVE-output figures with real savefig"
```

---

### Task 13: Table generation under manuscript names

**Files:**
- Create: `src/benchmarks/tables.py`
- Create: `tests/benchmarks/test_tables_cli.py`
- Modify: `src/benchmarks/run.py`

**Interfaces:**
- Consumes: `summarize`, `write_tables` from plan 1 Task 13; `read_run` from plan 1 Task 9; `SCENARIOS` from plan 1 Task 6.
- Produces:
  - `TABLE_NAMES: dict[str, str]` — scenario name to manuscript table name
  - `TABLE_CAPTIONS: dict[str, str]`
  - `write_all_tables(results_by_scenario, out_dir, *, metrics=None) -> list[Path]`
  - `main(argv=None) -> int` extended in `run.py` with `--tables`

The supplementary tables are currently printed for copy-paste, which is why the committed versions
have the generated `$k^\star=5$` header row stripped by hand and have drifted from what the code
produces. Writing `.tex` fragments to disk under their manuscript names ends that.

`EXCLUDE_FROM_TABLES` in the old config listed five metrics kept out of the manuscript tables;
that list is preserved here as `EXCLUDED_METRICS`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_tables_cli.py`:

```python
"""Tests for manuscript table generation."""

import pandas as pd
import pytest

from benchmarks._artifacts import SCHEMA
from benchmarks.tables import (
    EXCLUDED_METRICS,
    TABLE_CAPTIONS,
    TABLE_NAMES,
    write_all_tables,
)

METRICS = ("ari_stability_1se", "ari_average_1se", "silhouette")


def _frame(scenario):
    rows = []
    for axis_value, axis_label in enumerate(("easy", "medium", "hard")):
        for seed in range(3):
            for metric in METRICS:
                for k in (4, 5):
                    rows.append({
                        "run_id": "r1", "scenario": scenario,
                        "axis_name": "difficulty_level", "axis_value": axis_value,
                        "axis_label": axis_label, "seed": seed, "k_star": 5,
                        "estimator": "kmeans", "metric_name": metric, "k": k,
                        "metric_value": 0.1 * k, "is_selected": k == 5,
                        "selects_true_k": k == 5, "ari_at_k": 0.9, "oracle_ari": 0.95,
                    })
    return pd.DataFrame(rows)[list(SCHEMA)]


class TestTableNames:
    def test_every_simulated_scenario_has_a_manuscript_table_name(self):
        from benchmarks._registry import SCENARIOS

        assert set(TABLE_NAMES) == set(SCENARIOS)

    def test_every_table_has_a_caption(self):
        assert set(TABLE_CAPTIONS) == set(TABLE_NAMES)

    def test_names_are_unique(self):
        assert len(set(TABLE_NAMES.values())) == len(TABLE_NAMES)


class TestWriteAllTables:
    def test_writes_one_fragment_per_scenario(self, tmp_path):
        frames = {"gaussians": _frame("gaussians"), "moons": _frame("moons")}
        paths = write_all_tables(frames, tmp_path)
        assert len(paths) == 2
        assert all(p.suffix == ".tex" and p.exists() for p in paths)

    def test_uses_the_manuscript_table_name(self, tmp_path):
        paths = write_all_tables({"gaussians": _frame("gaussians")}, tmp_path)
        assert paths[0].stem == TABLE_NAMES["gaussians"]

    def test_fragments_contain_a_tabular_environment(self, tmp_path):
        paths = write_all_tables({"gaussians": _frame("gaussians")}, tmp_path)
        assert "\\begin{tabular}" in paths[0].read_text()

    def test_excluded_metrics_do_not_appear(self, tmp_path):
        paths = write_all_tables({"gaussians": _frame("gaussians")}, tmp_path)
        text = paths[0].read_text()
        assert "ari_average_1se" not in text
        assert "ARI (avg, 1SE)" not in text

    def test_the_k_star_header_is_generated_not_stripped(self, tmp_path):
        paths = write_all_tables({"gaussians": _frame("gaussians")}, tmp_path)
        assert "k^\\star" in paths[0].read_text()

    def test_skips_a_scenario_with_no_rows(self, tmp_path):
        empty = pd.DataFrame(columns=list(SCHEMA))
        paths = write_all_tables({"gaussians": empty}, tmp_path)
        assert paths == []


class TestExcludedMetrics:
    def test_matches_the_published_exclusion_list(self):
        assert EXCLUDED_METRICS == frozenset({
            "ari_average",
            "ari_average_1se",
            "ari_average_quant",
            "consensus_pac_stability",
            "consensus_ce_stability",
        })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_tables_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.tables'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/tables.py`:

```python
"""Write the supplementary tables as .tex fragments.

Under their manuscript names, to disk, rather than printed for copy-paste.
The committed versions have the generated k-star header row stripped by hand,
which is the drift this ends.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ._registry import CARVE_METRICS_ALL, CVI_METRICS
from ._tables import write_tables

EXCLUDED_METRICS: frozenset[str] = frozenset({
    "ari_average",
    "ari_average_1se",
    "ari_average_quant",
    "consensus_pac_stability",
    "consensus_ce_stability",
})

TABLE_NAMES: dict[str, str] = {
    "gaussians": "S2_table",
    "t_dist": "S3_table",
    "t_dist_noise": "S4_table",
    "circles": "S5_table",
    "moons": "S6_table",
    "swiss_rolls": "S7_table",
    "gaussians_samples": "S8_table",
    "gaussians_dimensionality": "S9_table",
}

TABLE_CAPTIONS: dict[str, str] = {
    "gaussians": "Gaussian mixtures: ARI at the selected k and k-recovery by difficulty.",
    "t_dist": "t-distributed clusters: ARI at the selected k and k-recovery by difficulty.",
    "t_dist_noise": (
        "t-distributed clusters with nuisance dimensions: ARI at the selected k and "
        "k-recovery by difficulty."
    ),
    "circles": "RFF-embedded circles: ARI at the selected k and k-recovery by difficulty.",
    "moons": "RFF-embedded moons: ARI at the selected k and k-recovery by difficulty.",
    "swiss_rolls": (
        "RFF-embedded Swiss rolls: ARI at the selected k and k-recovery by difficulty."
    ),
    "gaussians_samples": (
        "Gaussian mixtures over sample size: ARI at the selected k and k-recovery."
    ),
    "gaussians_dimensionality": (
        "Gaussian mixtures over embedding dimension: ARI at the selected k and k-recovery."
    ),
}


def table_metrics() -> tuple[str, ...]:
    """Metrics that appear in the manuscript tables, in a stable order."""
    return tuple(
        m for m in (*CARVE_METRICS_ALL, *CVI_METRICS) if m not in EXCLUDED_METRICS
    )


def write_all_tables(
    results_by_scenario: Mapping[str, pd.DataFrame],
    out_dir: Path,
    *,
    metrics: Sequence[str] | None = None,
) -> list[Path]:
    """Write one .tex fragment per scenario under its manuscript table name."""
    metrics = tuple(metrics) if metrics is not None else table_metrics()
    written: list[Path] = []

    for name, frame in results_by_scenario.items():
        if frame.empty:
            continue
        present = [m for m in metrics if (frame["metric_name"] == m).any()]
        if not present:
            continue

        k_star = int(frame["k_star"].iloc[0])
        table_name = TABLE_NAMES.get(name, f"{name}_table")
        written.append(
            write_tables(
                frame,
                Path(out_dir),
                table_name,
                caption=(
                    f"{TABLE_CAPTIONS.get(name, name)} $k^\\star = {k_star}$."
                ),
                label=f"tab:{table_name.lower()}",
                metrics=present,
            )
        )
    return written
```

- [ ] **Step 4: Wire the CLI**

In `src/benchmarks/run.py`, add to `_parser()`:

```python
    parser.add_argument(
        "--tables",
        help="Write manuscript .tex table fragments to this directory.",
    )
```

and, in `main`, immediately before the `if args.scenario:` block:

```python
    if args.tables:
        from ._artifacts import read_run
        from .tables import write_all_tables

        frames = {}
        for name in sorted(SCENARIOS):
            for candidate in sorted(Path(args.root).glob(f"{name}/*/")):
                frames[name] = read_run(candidate)
        paths = write_all_tables(frames, Path(args.tables))
        for path in paths:
            print(path)
        return 0
```

Add `from pathlib import Path` to the imports if it is not already there.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_tables_cli.py tests/benchmarks/test_cli.py -v`
Expected: all passed

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/tables.py src/benchmarks/run.py tests/benchmarks/test_tables_cli.py
git commit -m "feat(benchmarks): write manuscript tables to disk under their names"
```

---

### Task 14: Notebooks as thin drivers

**Files:**
- Modify: `notebooks/Benchmarking.ipynb`
- Modify: `notebooks/case_studies/Klein.ipynb`
- Modify: `notebooks/case_studies/Levine_32dim.ipynb`
- Modify: `notebooks/case_studies/Motivation.ipynb`
- Create: `tests/benchmarks/test_notebooks.py`

**Interfaces:**
- Consumes: every public function from Tasks 5 through 13.
- Produces: notebooks containing no `sys.path` manipulation, no loader code, and no gridspec layout.

`Benchmarking.ipynb` keeps its narrative structure. Its per-scenario `anchor_settings` cells
(9, 15, 21, 27, 33, 39, 46, 51) are deleted, because that content now lives in `_registry.py`.
Cell 60, the 118-line scaling composite, becomes two calls. The case-study notebooks lose their
loader cells and their 178-line composite cells.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_notebooks.py`:

```python
"""Structural checks on the notebooks.

These do not execute the notebooks; they assert that the code the rebuild
moved into the package is no longer duplicated in notebook cells.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = {
    "benchmarking": REPO_ROOT / "notebooks" / "Benchmarking.ipynb",
    "klein": REPO_ROOT / "notebooks" / "case_studies" / "Klein.ipynb",
    "levine": REPO_ROOT / "notebooks" / "case_studies" / "Levine_32dim.ipynb",
    "motivation": REPO_ROOT / "notebooks" / "case_studies" / "Motivation.ipynb",
}


def _code(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_no_sys_path_manipulation(name):
    source = _code(NOTEBOOKS[name])
    assert "sys.path.insert" not in source
    assert "sys.path.append" not in source


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_imports_come_from_the_installed_package(name):
    source = _code(NOTEBOOKS[name])
    assert "import benchmarking_code" not in source
    assert "import case_study_plotting" not in source
    assert "import benchmarking_plotting" not in source


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_no_gridspec_layout_in_notebooks(name):
    assert "add_gridspec" not in _code(NOTEBOOKS[name])


def test_anchor_dicts_live_in_the_registry_not_the_notebook():
    assert "anchor_settings_gaussians" not in _code(NOTEBOOKS["benchmarking"])


@pytest.mark.parametrize("name", ["klein", "levine", "motivation"])
def test_loader_code_is_not_duplicated_in_case_study_notebooks(name):
    source = _code(NOTEBOOKS[name])
    assert "BiocManager::install" not in source
    assert "read_block" not in source
    assert "highly_variable_genes" not in source


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_no_get_n_jobs_boilerplate(name):
    assert "def get_n_jobs" not in _code(NOTEBOOKS[name])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_notebooks.py -v`
Expected: multiple failures, one per notebook still carrying moved code.

- [ ] **Step 3: Rewrite the Benchmarking notebook driver cells**

Replace the import cell (cell 6) with:

```python
from pathlib import Path

from benchmarks._artifacts import read_run, read_runtimes
from benchmarks._registry import SCENARIOS
from benchmarks.figures import (
    figure_benchmarking_examples,
    figure_benchmarking_results,
    figure_scaling_ari,
    figure_scaling_runtime,
)
from benchmarks.tables import write_all_tables

RESULTS_ROOT = Path("../results/runs")

def load(name):
    """Read the most recent run for a scenario."""
    runs = sorted((RESULTS_ROOT / name).glob("*/"))
    if not runs:
        raise FileNotFoundError(
            f"No run for {name!r}. Produce one with: "
            f"python -m benchmarks.run --scenario {name}"
        )
    return read_run(runs[-1])
```

Delete cells 9, 15, 21, 27, 33, 39, 46 and 51 entirely; the anchors are in `_registry.py`.

Replace each `plot_examples(...)` cell with a markdown note pointing at the single figure call,
and put one call near the top of the results section:

```python
fig = figure_benchmarking_examples()
```

Replace the `benchmark_cluster_metrics(...)` cells with a markdown cell reading:

> Compute runs headless. Produce artifacts with `python -m benchmarks.run --all`,
> then re-run this notebook, which only reads them.

Replace cell 60 with:

```python
results = {name: load(name) for name in SCENARIOS}

fig_main = figure_benchmarking_results(
    {k: v for k, v in results.items() if v["axis_name"].iloc[0] == "difficulty_level"}
)

scaling = {k: v for k, v in results.items() if v["axis_name"].iloc[0] != "difficulty_level"}
fig_ari = figure_scaling_ari(scaling)

runtimes = {}
for name in scaling:
    runs = sorted((RESULTS_ROOT / name).glob("*/"))
    runtimes[name] = read_runtimes(runs[-1])
fig_runtime = figure_scaling_runtime(runtimes)

write_all_tables(results, Path("../vis/tables"))
```

- [ ] **Step 4: Rewrite the case-study notebooks**

In `Klein.ipynb`, replace cells 4 through 21 (imports through subsampling) with:

```python
from pathlib import Path

from benchmarks._studies import STUDIES, cvi_sweep, fit_or_load_carve, study_model_grids
from benchmarks.datasets import load_klein
from benchmarks.figures import (
    figure_carve_output_klein,
    figure_klein_results,
    prepare_composite,
)

RANDOM_SEED = 42
study = STUDIES["klein"]
model_grids = study_model_grids(study)

X, y, meta = load_klein(root=Path("../../data"), random_state=RANDOM_SEED)
print(f"{meta['n_cells']:,} cells x {meta['n_features']:,} genes")
```

Replace cells 24 through 26 with:

```python
curves_df, best_df = cvi_sweep(
    X, y, model_grids=model_grids, candidate_k=study.candidate_k,
    random_state=RANDOM_SEED, n_jobs=-1,
)
best_df
```

Replace cell 28 with:

```python
carve = fit_or_load_carve(
    X, y,
    cache_path=Path("./carve_state_saves/carve_klein.carve"),
    model_grids=model_grids,
    random_state=RANDOM_SEED,
)
```

Replace cells 29 and 32 with:

```python
inputs = prepare_composite(
    X, y.to_numpy(), carve, curves_df=curves_df, best_df=best_df,
    comparison_metric="silhouette", random_state=RANDOM_SEED,
)
fig_output = figure_carve_output_klein(inputs)
fig_results = figure_klein_results(inputs)
```

Apply the same shape to `Levine_32dim.ipynb`, substituting `load_levine32`,
`STUDIES["levine32"]`, `figure_carve_output_levine`, `figure_levine_results`, and passing the
precomputed t-SNE as `embedding=` to `prepare_composite`.

In `Motivation.ipynb`, delete both duplicated loader blocks (cells 10 through 14 for Levine and
cells 16 through 28 for Klein) and replace them with two loader calls:

```python
X_levine, y_levine, meta_levine = load_levine32()
X_klein, y_klein, meta_klein = load_klein(root=Path("../../data"))
```

- [ ] **Step 5: Run the structural tests**

Run: `.venv/bin/pytest tests/benchmarks/test_notebooks.py -v`
Expected: all passed

- [ ] **Step 6: Execute the notebooks end to end from a clean kernel**

This is spec verification item 7. Run each with no `sys.path` manipulation:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 notebooks/Benchmarking.ipynb
```

Repeat for the three case-study notebooks from within `notebooks/case_studies/`. The Levine
notebook needs its cache present or `CARVE_ALLOW_R=1`; the Klein and Levine notebooks need their
`.carve` state files, or they will fit from scratch, which takes hours.

If a notebook fails because no run artifacts exist, produce them first with
`python -m benchmarks.run --all`.

- [ ] **Step 7: Commit**

```bash
git add notebooks/ tests/benchmarks/test_notebooks.py
git commit -m "refactor(benchmarks): reduce notebooks to thin drivers over the package"
```

---

### Task 15: Deletions and final verification

**Files:**
- Delete: `notebooks/benchmarking_code/` (all ten modules plus `legacy/` and `__pycache__/`)
- Delete: `results/scalability_results/results_moons_dimensionality.csv`, `results/scalability_results/runtimes_moons_dimensionality.csv`
- Delete: `results/results_*.csv`, `results/results_swiss_rolls.csv.zip`, `results/scalability_results/`
- Create: `docs/superpowers/notes/2026-09-01-legacy-design-ideas.md`
- Modify: `/Users/kaiwycik/GitHub/CARVE/_claude_playground/CLAUDE.md`

**Interfaces:**
- Consumes: everything.
- Produces: a tree with one benchmarking codebase in it.

Nothing here runs until the regression harness from plan 1 Task 14 has passed. The old CSVs are
the oracle, so they cannot be deleted in the same step that checks against them, and the old
modules are the reference the port was read from.

- [ ] **Step 1: Confirm the port is verified before deleting anything**

Run: `CARVE_RUN_REGRESSION=1 .venv/bin/pytest tests/benchmarks/test_regression.py -v`
Expected: all passed.

If this has not been run, or did not pass, stop. Do not proceed with any deletion in this task.

- [ ] **Step 2: Extract the two design ideas from legacy before deleting it**

`legacy/` is 1,480 lines with zero importers and two files whose relative imports no longer
resolve. Two capabilities in it were lost in the current design and are worth recording before the
code goes:

Read `notebooks/benchmarking_code/legacy/benchmarking_simulation_helpers copy.py:30-101` and
`notebooks/benchmarking_code/legacy/benchmarking_runners copy.py:30`, then write
`docs/superpowers/notes/2026-09-01-legacy-design-ideas.md` containing:

- A description of the interpolated continuous-difficulty scheme: how it parameterized difficulty
  as a continuous value between anchors rather than three discrete levels, with the interpolation
  formula copied out, and a note that under the rebuilt design this is
  `Axis("difficulty", values=<continuous grid>, ...)` and needs no new runner.
- A description of the multi-k-star sweep: how it varied the true cluster count across a run, with
  the loop structure copied out, and a note that this needs a second axis, which `SweepSpec` being
  frozen to one parameter currently forbids, so it is a design question rather than a small
  addition.

- [ ] **Step 3: Delete the old benchmarking code**

```bash
git rm -r notebooks/benchmarking_code/
```

This removes all ten modules, `legacy/`, and `test_benchmarking.py`, whose still-useful coverage
of `make_scaling_x_values`, `_wilson_ci`, `get_rule`/`get_measure`, and `align_labels` was
reproduced in `tests/benchmarks/` during plan 1.

- [ ] **Step 4: Delete the superseded results**

```bash
git rm results/results_gaussian.csv results/results_t_dist.csv \
       results/results_t_dist_noise.csv results/results_circles.csv \
       results/results_moons.csv results/results_swiss_rolls.csv \
       results/results_swiss_rolls.csv.zip
git rm -r results/scalability_results/
```

The replacement is `results/published/<scenario>/`, produced by `--promote`. Promote each verified
run before committing this step, so the repo never sits in a state with no numbers in it:

```bash
for s in gaussians t_dist t_dist_noise circles moons swiss_rolls \
         gaussians_samples gaussians_dimensionality; do
  .venv/bin/python -m benchmarks.run --promote "$(ls -d results/runs/$s/*/ | tail -1)"
done
git add results/published/
```

- [ ] **Step 5: Delete the remaining dead code references**

Confirm nothing still imports the removed modules:

```bash
grep -rn "benchmarking_code\|case_study_plotting\|benchmarking_plotting\|benchmarking_runners" \
  --include="*.py" --include="*.ipynb" --include="*.yml" --include="*.toml" . \
  | grep -v "^./docs/" | grep -v "^./.venv/"
```

Expected: no output. Matches under `docs/` are plan and spec prose and are fine.

- [ ] **Step 6: Update CLAUDE.md**

In `/Users/kaiwycik/GitHub/CARVE/_claude_playground/CLAUDE.md`:

- Delete the "Known issues" bullet about `notebooks/benchmarking_code/` being covered by neither
  pytest nor ruff; it no longer exists.
- Under "Workspace layout", note that `../code/src/benchmarks/` holds the benchmarking and
  case-study code, is linted and tested in CI, and is excluded from the wheel.
- Under "Key structural rules", add: `Figures come only from benchmarks.figures.figure_*; each
  returns a Figure and saves under its manuscript filename. Copying into overleaf/vis/ stays
  manual.`
- Under "Conventions that are easy to break", add: `One theme. benchmarks._theme is the single
  source for palette, rcParams, and geometry. Do not add a module-level color literal.`
- Update the `results/` line: `results/runs/` is gitignored working output; `results/published/`
  holds promoted CSVs plus manifests and is committed.

- [ ] **Step 7: Run everything**

```bash
.venv/bin/pytest -v --tb=short --cov=carve --cov-report=term-missing --cov-fail-under=75
.venv/bin/ruff check src/
.venv/bin/ruff format --check src/
```

Expected: all pass, ruff clean, coverage at or above 75 percent on `carve`.

- [ ] **Step 8: Regenerate every figure and compare against the manuscript**

This is spec verification item 5.

```bash
.venv/bin/python -m benchmarks.run --all --n-jobs -1
```

then run the four notebooks, which write all eight figures into `vis/`. Compare each against the
corresponding file in `../overleaf/vis/` by eye. The four that were previously screenshot-derived
(`CARVE_output_klein`, `klein_results`, `CARVE_output_levine`, `levine_results`) will differ in
resolution and styling by design; their content must match. Fig 4's CARVE-stability green shifts
from `#00CD6C` to `#009E73`, which is the palette fix landing.

Record the comparison in `docs/superpowers/notes/2026-09-01-figure-comparison.md`, one line per
figure: matches, differs cosmetically, or differs in content. Any content difference is a bug in
the port and must be traced before the branch merges.

- [ ] **Step 9: Regenerate and diff the tables**

This is spec verification item 6.

```bash
.venv/bin/python -m benchmarks.run --tables vis/tables
```

Diff each fragment against the corresponding table in `../overleaf/CARVE_manuscript.tex`. The
generated `$k^\star = 5$` header row now appears in the fragment; the committed manuscript has it
stripped by hand, so that difference is expected and is the drift being closed. Numeric
differences are expected only where plan 1's bug fixes changed them: the gap column everywhere,
and `ari_at_k` for the generalizability metrics.

Record the outcome in `docs/superpowers/notes/2026-09-01-table-comparison.md`.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor(benchmarks): delete the superseded benchmarking code and results"
```

---

## Completion criteria

Plan 2 is done when all of these hold:

1. `.venv/bin/pytest -v --tb=short --cov=carve --cov-fail-under=75` passes.
2. `.venv/bin/ruff check src/` and `.venv/bin/ruff format --check src/` are clean.
3. `tests/benchmarks/figures/test_figure_contract.py` passes, so all eight manuscript figures
   exist, return a `Figure`, accept `save` and `out_dir`, and no figure module calls `plt.show()`.
4. All eight figures regenerate from artifacts into `vis/`, and the comparison against
   `overleaf/vis/` is recorded with no content differences.
5. The eight table fragments regenerate, and the comparison against the manuscript is recorded
   with differences only where the bug fixes explain them.
6. All four notebooks execute end to end from a clean kernel with no `sys.path` manipulation.
7. `notebooks/benchmarking_code/` is gone, nothing references it, and the two legacy design ideas
   are recorded in `docs/superpowers/notes/`.
8. `results/published/` holds a promoted CSV plus manifest for every scenario, and the old
   top-level `results/results_*.csv` files are gone.

## Out of scope

The seven reviewer experiments and the two new case studies remain follow-on plans, per spec
decision 4: large scRNA-seq (R1.1), scATAC/multimodal (R1.7), preprocessing ablation (R1.6), rho
and n_resamples sensitivity (R4.4), SC3/M3C comparison (R2.1), Leiden/Louvain on the case studies
(R1.4), and the resampling-bias investigation (R1.3).

`clustering_problem.png` and `CARVE_schema.png` are hand-drawn schematics with no generating code
and are not part of this plan.
