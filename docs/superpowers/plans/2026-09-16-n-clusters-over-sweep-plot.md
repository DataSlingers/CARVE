# Realized Cluster Count Plot Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: add a native plot of the realized number of clusters (`n_clusters_observed`, plus or minus one standard error) against the swept parameter, for runs that sweep a parameter other than `n_clusters`, at the module, `CARVE` and `pl` levels.

Architecture: `_plotting._draw_metric_lines`, which already draws `plot_metric_over_n_clusters` and `plot_metric_by_pipeline`, is generalized to take the y column, the dashed line's legend text and an integer-tick switch from its callers. The new `plot_n_clusters_over_sweep` is a third caller. `CARVE.plot_n_clusters_over_sweep` forwards `estimator_results_`; `pl.n_clusters_over_sweep` reads the stored results table and the recorded selection from `adata.uns`. Nothing in fitting, the runner, the results tables or `tl.carve` changes.

Tech stack: Python (`requires-python >= 3.13`; the main venv is 3.12.11 and runs the suite), pandas, NumPy, matplotlib 3.11.1 (`Axes.errorbar`, `matplotlib.ticker.MaxNLocator`), anndata, pytest. No new dependencies.

Spec: `docs/superpowers/specs/2026-09-16-n-clusters-over-sweep-plot-design.md` (committed as af51b1c). Sections are cited as "spec 4.1" etc.

## Scope

This plan changes `src/carve/_plotting.py`, `src/carve/api.py`, `src/carve/pl/_plots.py`, `src/carve/pl/__init__.py`, and their mirrored test files. Nothing else.

Out of scope, per spec 3. Reviewers do not flag these as gaps:

- An R port. Nothing under `carve-r/` or `.github/workflows/r-ci.yml` changes.
- A per-pipeline variant reading `preprocessing_results_`.
- `notebooks/`, `README.md`, and a log-scale option.
- `src/benchmarks/`. The Cusanovich figure keeps its own secondary axis.
- `../overleaf/`, which is read-only.

## Global Constraints

- All commands run from `/Users/kaiwycik/GitHub/CARVE/code`. The venv is uv-managed and has no pip; use `.venv/bin/pytest`, `.venv/bin/ruff` and `.venv/bin/python` directly.
- Work on the existing branch `n-clusters-over-sweep`. Its only commit over `main` (87b9f40) is the spec, af51b1c.
- Tests mirror modules one to one: `_plotting.py` to `tests/test_plotting.py`, `api.py` to `tests/test_api.py`, `pl/_plots.py` to `tests/test_pl.py`.
- `filterwarnings = error` is on. A new warning fails the suite. `-ra` is the only implicit addopt; `--cov-fail-under=75` fails a single-file run, so omit it locally.
- Lint and format with the pinned ruff 0.16.4 only: `.venv/bin/ruff check src/` and `.venv/bin/ruff format src/`. Never on `tests/` or `notebooks/`. The lint set includes `I` (import sorting), so new imports go in sorted position.
- Run tests in the foreground with a long timeout (600000 ms on the Bash tool). A slow run is not a hang.
- Dependency direction: `_plotting` imports from `_selection` and `_sweep` only; nothing under `_*` imports `api`.
- No `logging`.
- The only new public names are `carve._plotting.plot_n_clusters_over_sweep`, `CARVE.plot_n_clusters_over_sweep` and `carve.pl.n_clusters_over_sweep`.
- The output of `plot_metric_over_n_clusters` and `plot_metric_by_pipeline` does not change: lines, error bars, labels, ticks and legend text (spec 5.1).
- Default y label of the new plot: `Mean Observed Number of Clusters`. Dashed-line legend text: `Selected {param} ({rule_str} rule): {value:g}, {count} clusters`, where `{count}` is `_sweep.observed_k(row)`. The error for a sweep over `n_clusters` contains the phrase `other than n_clusters` (spec 4.1, 4.2).
- American spelling. No bold or italics in code comments, docstrings or authored Markdown.
- Commit after every task, ending the message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Refinements the plan makes to the spec

Stated here so an executor does not read them as deviations.

1. `integer_y` is added to `_draw_metric_lines` in Task 2, with the test that needs it, rather than in the Task 1 generalization. No Task 1 test could exercise it.
2. `_draw_metric_lines` gets two more private helpers besides `_measure_column` (spec 5.1): `_measure_ylabel(measure_col)` and `_selection_label(row, *, param, rule)`. All three plots build the dashed line's text through `_selection_label`, so the text lives in one place.
3. The spy in `TestSharedMetricDrawing` also records `y_col`. Its recorded tuples have four entries. The new plot's entry is `("method_id", "Estimators", "n_clusters_observed", 6)` rather than spec 6.1's `("method_id", "Estimators", 6)`.
4. Task 1 adds a pinning test for the dashed-line label and default y label of `plot_metric_over_n_clusters` on a resolution table. Spec 5.1 requires that output to stay unchanged, and no existing test pinned the non-k label of that plot.
5. The `CARVE` test in Task 3 goes beyond spec 6.1's values in three ways. All counts other than size 8 are set to 4.0, so `not_two` cannot exclude a row by accident of the fit. `ari_stability_se` and `ari_generalizability_se` are set explicitly. Generalizability at size 5 is 0.98, inside one standard error (0.05) of the peak. With the spec's values, "1se" and "max" would mark the same size, so a method that dropped `rule` would pass.

## How this plan was verified

Checked on 2026-09-16 against `main` at 87b9f40:

- Matplotlib 3.11.1 in the venv. With only `yerr`, `ax.errorbar` returns a container whose third element holds one `LineCollection`; the half-widths of its segments recover the standard errors, including 0.0. For counts 2.0, 2.5 and 3.0, the default locator ticks at 1.8, 2.0, ..., 3.2, and `MaxNLocator(integer=True)` ticks at 1, 2, 3, 4. `MaxNLocator.default_params["min_n_ticks"]` is 2.
- `np.select` accepts boolean pandas Series as conditions. Python's `round` gives `round(6.6) == 7` and `round(2.2) == 2`; `_sweep.observed_k` uses it.
- An earlier draft of the Task 2 to 4 tests was run against unchanged source. It failed for the expected reasons: an `ImportError` for `plot_n_clusters_over_sweep` in `tests/test_plotting.py`, and an `AttributeError` for the `CARVE` method and the `pl` function. The `tests/test_pl.py` fixtures (HDBSCAN fit on `X_pca`, `attach_results`, h5ad round trip) ran without warnings under `filterwarnings = error`. That draft was then discarded. The Task 3 test here differs from it as refinement 5 describes.
- `tests/conftest.py` has an autouse fixture that closes all figures after each test.
- Collected test counts on `main`: 882 for `tests --ignore=tests/benchmarks`, 309 for the three test files this plan edits. `ruff format --check src/` reports 66 files already formatted.
- The edits in Tasks 1 to 4 were applied in memory, in plan order, to the files at 87b9f40. Every old text occurred exactly once at the point it applies, including the Task 2 old texts that come from Task 1's new text. The resulting source and test files compile. The resulting `src/` files pass the pinned `ruff check`, and `ruff format --diff` reports no changes for them. The mutation targets named in Tasks 1 to 3 occur where the steps say: `y_col="n_clusters_observed"`, `{observed_k(row)} clusters`, `integer_y=True`, `_measure_column(results_df, measure)` and `measure=measure` once each inside the new function, and `rule=rule` three times (the steps name the one in `select_row`).

Not verified: the source code in Tasks 1 to 4 has not been executed. The expected failures in each mutation step were derived by hand from the fixture values. Treat the code as a detailed proposal. If a test in this plan cannot pass against correct code, fix the test and say so with evidence. Never loosen an assertion because the code under test disagrees with it.

## File structure

Modified, in task order:

- `src/carve/_plotting.py`: `_measure_column`, `_measure_ylabel`, `_selection_label`; `_draw_metric_lines` takes `y_col`, `selection_label`, a resolved `ylabel` (Task 1) and `integer_y` (Task 2); `plot_metric_over_n_clusters` and `plot_metric_by_pipeline` pass them (Task 1); `plot_n_clusters_over_sweep` (Task 2).
- `src/carve/api.py`: `CARVE.plot_n_clusters_over_sweep` (Task 3).
- `src/carve/pl/_plots.py`, `src/carve/pl/__init__.py`: `n_clusters_over_sweep` (Task 4).
- `tests/test_plotting.py` (Tasks 1, 2), `tests/test_api.py` (Task 3), `tests/test_pl.py` (Task 4).

---

### Task 0: Commit the plan

- [ ] Step 1: Check out the branch and confirm its state

```bash
git checkout n-clusters-over-sweep
git log --oneline main..HEAD
git status --short
```

Expected: one commit, `af51b1c docs(spec): design for a realized cluster count plot over non-k sweeps`, and `?? docs/superpowers/plans/2026-09-16-n-clusters-over-sweep-plot.md` as the only status line.

- [ ] Step 2: Commit the plan

```bash
git add docs/superpowers/plans/2026-09-16-n-clusters-over-sweep-plot.md
git commit -m "docs(plan): implementation plan for the realized cluster count plot" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 1: Generalize the shared line drawing

Spec 5.1, without `integer_y` (refinement 1). `_draw_metric_lines` stops deriving the y column, the default y label and the dashed line's text from `measure`; its callers pass them. Both metric plots keep their output exactly.

Files:
- Modify: `src/carve/_plotting.py`
- Test: `tests/test_plotting.py`

Interfaces:
- Consumes: nothing from earlier tasks.
- Produces:
  - `_plotting._measure_column(results_df: pd.DataFrame, measure: str) -> str`. Raises `ValueError(f"Measure {measure!r} not found. Valid options: [...]")` or `ValueError(f"Metric column {measure_col!r} not found in the results table.")`.
  - `_plotting._measure_ylabel(measure_col: str) -> str`, e.g. `"ari_stability"` to `"ARI Stability"`.
  - `_plotting._selection_label(row: pd.Series, *, param: str, rule: str) -> str`, e.g. `"Selected resolution (Max rule): 0.25"`, `"Selected k (1-SE rule): 4"`.
  - `_plotting._draw_metric_lines(results_df, *, y_col: str, group_col: str, label_of: Callable[[pd.Series], str], legend_title: str, select_row: Callable[[], pd.Series], selection_label: Callable[[pd.Series], str], rule: str, ax, figsize, title, xlabel, ylabel: str, legend, legend_loc, palette, show, save, dpi, **kwargs) -> Axes | None`.

- [ ] Step 1: Write the failing test and the pinning test

In `tests/test_plotting.py`, replace:

```python
        assert ax.get_title() == "My Title"
        assert ax.get_xlabel() == "k"
        assert ax.get_ylabel() == "Score"
```

with:

```python
        assert ax.get_title() == "My Title"
        assert ax.get_xlabel() == "k"
        assert ax.get_ylabel() == "Score"

    def test_marker_label_and_default_ylabel(self, resolution_results_df):
        # Pins the text _draw_metric_lines built itself before its callers
        # passed y_col, ylabel and selection_label in.
        ax = plot_metric_over_n_clusters(
            resolution_results_df, measure="stability", rule="max"
        )
        (dashed,) = _dashed(ax)
        assert dashed.get_label() == "Selected resolution (Max rule): 0.25"
        assert ax.get_ylabel() == "ARI Stability"
```

In `tests/test_plotting.py`, replace:

```python
        def spy(df, **kwargs):
            calls.append((kwargs["group_col"], kwargs["legend_title"], len(df)))
            return "drawn"
```

with:

```python
        def spy(df, **kwargs):
            calls.append(
                (kwargs["group_col"], kwargs["legend_title"], kwargs["y_col"], len(df))
            )
            return "drawn"
```

In `tests/test_plotting.py`, replace:

```python
        assert calls == [("method_id", "Estimators", 3), ("pipeline", "Pipelines", 6)]
```

with:

```python
        assert calls == [
            ("method_id", "Estimators", "ari_stability", 3),
            ("pipeline", "Pipelines", "ari_stability", 6),
        ]
```

- [ ] Step 2: Run the two tests

```bash
.venv/bin/pytest tests/test_plotting.py -q -k "SharedMetricDrawing or test_marker_label_and_default_ylabel"
```

Expected: `1 failed, 1 passed`. The failure is `test_both_metric_plots_draw_through_one_helper` with `KeyError: 'y_col'`. The pinning test passes: it records today's output.

- [ ] Step 3: Generalize `_draw_metric_lines` and update both callers

In `src/carve/_plotting.py`, replace:

```python
    if results_df.empty:
        raise RuntimeError("Results DataFrame is empty.")

    return _draw_metric_lines(
        results_df,
        group_col="method_id",
        label_of=_build_estimator_label,
        legend_title="Estimators",
        select_row=lambda: select_best_row_by_rule(
            results_df, measure=measure, rule=rule, not_two=not_two
        ),
        measure=measure,
        rule=rule,
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
```

with:

```python
    if results_df.empty:
        raise RuntimeError("Results DataFrame is empty.")
    measure_col = _measure_column(results_df, measure)
    param = sweep_param_name(results_df)

    return _draw_metric_lines(
        results_df,
        y_col=measure_col,
        group_col="method_id",
        label_of=_build_estimator_label,
        legend_title="Estimators",
        select_row=lambda: select_best_row_by_rule(
            results_df, measure=measure, rule=rule, not_two=not_two
        ),
        selection_label=lambda row: _selection_label(row, param=param, rule=rule),
        rule=rule,
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel if ylabel is not None else _measure_ylabel(measure_col),
```

In `src/carve/_plotting.py`, replace:

```python
    return _draw_metric_lines(
        rows,
        group_col="pipeline",
        label_of=lambda row: str(row["pipeline"]),
        legend_title="Pipelines",
        select_row=lambda: _selected_row_for(
            estimator_df, method_id, measure=measure, rule=rule, not_two=not_two
        ),
        measure=measure,
        rule=rule,
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
```

with:

```python
    measure_col = MEASURE_MAP[measure]
    param = sweep_param_name(rows)

    return _draw_metric_lines(
        rows,
        y_col=measure_col,
        group_col="pipeline",
        label_of=lambda row: str(row["pipeline"]),
        legend_title="Pipelines",
        select_row=lambda: _selected_row_for(
            estimator_df, method_id, measure=measure, rule=rule, not_two=not_two
        ),
        selection_label=lambda row: _selection_label(row, param=param, rule=rule),
        rule=rule,
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel if ylabel is not None else _measure_ylabel(measure_col),
```

In `src/carve/_plotting.py`, replace the whole of `_draw_metric_lines`: every line from `def _draw_metric_lines(` up to, but not including, `def plot_consensus_matrix(`. Replace it with:

```python
def _measure_column(results_df: pd.DataFrame, measure: str) -> str:
    """Return the results column behind ``measure``.

    Plots call this before drawing. An unknown measure would otherwise make
    the selection raise inside the ``try`` around the dashed line, and the
    line would be left off without an error.

    Raises
    ------
    ValueError
        If ``measure`` is not a key of ``MEASURE_MAP``, or the table does not
        carry its column.
    """
    if measure not in MEASURE_MAP:
        raise ValueError(
            f"Measure {measure!r} not found. Valid options: {list(MEASURE_MAP.keys())}"
        )
    measure_col = MEASURE_MAP[measure]
    if measure_col not in results_df.columns:
        raise ValueError(
            f"Metric column {measure_col!r} not found in the results table."
        )
    return measure_col


def _measure_ylabel(measure_col: str) -> str:
    """Default y-axis label for a metric column, e.g. ``"ARI Stability"``."""
    return measure_col.replace("_", " ").title().replace("Ari", "ARI")


def _selection_label(row: pd.Series, *, param: str, rule: str) -> str:
    """Legend text for the dashed line at the selected row's sweep value."""
    rule_str = "1-SE" if rule == "1se" else rule.title()
    pretty = "k" if param == "n_clusters" else param
    value = float(row["sweep_value"])
    return f"Selected {pretty} ({rule_str} rule): {value:g}"


def _draw_metric_lines(
    results_df: pd.DataFrame,
    *,
    y_col: str,
    group_col: str,
    label_of: Callable[[pd.Series], str],
    legend_title: str,
    select_row: Callable[[], pd.Series],
    selection_label: Callable[[pd.Series], str],
    rule: str,
    ax: Axes | None,
    figsize: tuple | None,
    title: str | None,
    xlabel: str | None,
    ylabel: str,
    legend: bool,
    legend_loc: str,
    palette: str,
    show: bool,
    save: str | Path | None,
    dpi: int,
    **kwargs,
) -> Axes | None:
    """Draw one line of ``y_col`` per group across the sweep axis.

    The drawing behind the line plots over the sweep axis in this module, so
    they cannot drift apart. ``results_df`` is non-empty and carries
    ``sweep_param``, ``sweep_value``, ``sweep_rank``, ``group_col`` and
    ``y_col``. Error bars come from ``f"{y_col}_se"`` when the table carries
    it. Callers validate the measure and resolve the default ``ylabel``.

    ``select_row`` returns the row whose ``sweep_value`` the dashed line
    marks, and ``selection_label`` builds its legend text from that row. If
    either raises, for example because ``not_two`` leaves no rows, the line
    is left off.
    """
    se_col = f"{y_col}_se"
    has_se = se_col in results_df.columns

    # --- Figure setup ---
    if ax is None:
        if figsize is None:
            figsize = (9, 5.5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- Identify grouping and x-axis columns ---
    x_col = "sweep_value"

    results_df = results_df.copy()
    grouped = results_df.groupby([group_col])

    colors = plt.get_cmap(palette)(np.linspace(0, 1, len(grouped)))

    # --- Plot each group ---
    for color_idx, (group_key, group_df) in enumerate(grouped):
        label = label_of(group_df.iloc[0])
        group_df_sorted = group_df.sort_values(x_col)

        x = group_df_sorted[x_col].values
        y = group_df_sorted[y_col].values
        yerr = group_df_sorted[se_col].values if has_se else None
        color = colors[color_idx % len(colors)]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=4,
            capthick=1.5,
            alpha=0.8,
            color=color,
            label=label,
            **kwargs,
        )

    # --- Vertical line at the selected sweep value ---
    try:
        best_row = select_row()
        ax.axvline(
            float(best_row[x_col]),
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label=selection_label(best_row),
            zorder=0,
        )
    except Exception:
        pass

    # --- Labels and formatting ---
    if xlabel is None:
        xlabel = sweep_axis_label(results_df)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if title is not None:
        ax.set_title(title, fontsize=13, pad=15)

    x_unique = sorted(results_df[x_col].unique())
    if len(x_unique) <= 20:
        ax.set_xticks(x_unique)

    if legend:
        ax.legend(
            loc=legend_loc,
            frameon=True,
            framealpha=0.95,
            fontsize=10,
            title=legend_title if rule else None,
        )

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    # --- Save or show ---
    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    if show:
        plt.show()

    return ax


```

The replacement ends with two blank lines, so `def plot_consensus_matrix(` stays separated by two blank lines. The drawing code is unchanged apart from `y_col`, the dashed line's label and the y label. That keeps the metric plots' output the same.

- [ ] Step 4: Run the three test files

```bash
.venv/bin/pytest tests/test_plotting.py tests/test_pl.py tests/test_api.py -q
```

Expected: no failures or errors, 310 tests in total (309 before this task, plus the pinning test). Among them, the existing `TestPlotMetricOverNClusters::test_invalid_measure` (message "not found") and `TestPlotMetricByPipeline::test_marks_the_sweep_value_carve_selected` (label `Selected k (1-SE rule): 4`) pass unchanged.

- [ ] Step 5: Lint and format

```bash
.venv/bin/ruff check src/
.venv/bin/ruff format src/
```

Expected: `All checks passed!` and `66 files left unchanged`. If ruff reformatted a file anyway, rerun Step 4.

- [ ] Step 6: Commit

```bash
git add src/carve/_plotting.py tests/test_plotting.py
git commit -m "refactor(plotting): shared line drawing takes its y column and marker text from callers" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] Step 7: Mutation check

The pinning test must catch a change to the dashed-line text. In `_selection_label`, change `pretty = "k" if param == "n_clusters" else param` to `pretty = "k"`, then run:

```bash
.venv/bin/pytest tests/test_plotting.py -q -k test_marker_label_and_default_ylabel
git checkout -- src/carve/_plotting.py
```

Expected: `1 failed` (the label reads `Selected k (Max rule): 0.25`). After the checkout, `git status --short` prints nothing.

---

### Task 2: `plot_n_clusters_over_sweep` in `_plotting`

Spec 4.1, 4.2, 4.3 and 5.1. The new function validates its input and draws through `_draw_metric_lines`, which gains `integer_y`.

Files:
- Modify: `src/carve/_plotting.py`
- Test: `tests/test_plotting.py`

Interfaces:
- Consumes: `_measure_column`, `_selection_label` and `_draw_metric_lines(..., y_col, selection_label, rule, ylabel, ...)` from Task 1; `_sweep.observed_k(row: pd.Series) -> int`; `_selection.select_best_row_by_rule`.
- Produces:
  - `_plotting.plot_n_clusters_over_sweep(results_df: pd.DataFrame, *, measure: str = "stability", rule: str = "1se", not_two: bool = False, ax: Axes | None = None, figsize: tuple | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, legend: bool = True, legend_loc: str = "best", palette: str = "Accent", show: bool = False, save: str | Path | None = None, dpi: int = 300, **kwargs) -> Axes | None`. Raises `RuntimeError("Results DataFrame is empty.")`, a `ValueError` containing `other than n_clusters`, and the `_measure_column` errors.
  - `_draw_metric_lines` gains `integer_y: bool = False`.

- [ ] Step 1: Write the failing tests

In `tests/test_plotting.py`, replace:

```python
    plot_metric_by_pipeline,
    plot_metric_over_n_clusters,
)
from tests._helpers import with_sweep_cols
```

with:

```python
    plot_metric_by_pipeline,
    plot_metric_over_n_clusters,
    plot_n_clusters_over_sweep,
)
from tests._helpers import with_sweep_cols
```

In `tests/test_plotting.py`, replace:

```python
        assert calls == [
            ("method_id", "Estimators", "ari_stability", 3),
            ("pipeline", "Pipelines", "ari_stability", 6),
        ]
```

with:

```python
        assert plot_n_clusters_over_sweep(_realized_count_df()) == "drawn"
        assert calls == [
            ("method_id", "Estimators", "ari_stability", 3),
            ("pipeline", "Pipelines", "ari_stability", 6),
            ("method_id", "Estimators", "n_clusters_observed", 6),
        ]


# -----------------------------------------------------------------------
# plot_n_clusters_over_sweep
# -----------------------------------------------------------------------


def _realized_count_df():
    """A resolution-mode estimator_results_ table over two methods.

    Each selection below lands on a different row, and two of the counts
    differ between truncation and rounding (6.6 and 2.2 round to 7 and 2):

    - stability, "max": Leiden at 1.0 (0.90), 6.6 observed.
    - stability, "1se": 0.90 - 0.03 admits Leiden at 2.0 (0.88), the finest
      row within tolerance, 11.0 observed.
    - generalizability, "max": Louvain at 0.5 (0.95), 2.2 observed.
    - generalizability, "max", not_two: Louvain at 0.5 rounds to two
      clusters and is excluded, leaving Leiden at 1.0 (0.70).
    """
    curves = {
        ("m0", "LeidenClustering"): (
            [0.80, 0.90, 0.88],
            [0.01, 0.03, 0.01],
            [0.60, 0.70, 0.65],
            [3.2, 6.6, 11.0],
            [0.2, 0.4, 0.0],
        ),
        ("m1", "LouvainClustering"): (
            [0.70, 0.75, 0.60],
            [0.01, 0.01, 0.01],
            [0.95, 0.50, 0.40],
            [2.2, 5.0, 9.4],
            [0.1, 0.3, 0.6],
        ),
    }
    rows = []
    for (method_id, label), values in curves.items():
        stab, stab_se, gen, observed, observed_se = values
        for rank, resolution in enumerate((0.5, 1.0, 2.0)):
            rows.append(
                {
                    "config_id": len(rows),
                    "method_id": method_id,
                    "method_label": label,
                    "estimator": label,
                    "resolution": resolution,
                    "ari_stability": stab[rank],
                    "ari_stability_se": stab_se[rank],
                    "ari_generalizability": gen[rank],
                    "ari_generalizability_se": 0.01,
                    "sweep_param": "resolution",
                    "sweep_value": resolution,
                    "sweep_rank": rank,
                    "n_clusters_observed": observed[rank],
                    "n_clusters_observed_se": observed_se[rank],
                    "noise_fraction": 0.0,
                }
            )
    return pd.DataFrame(rows)


class TestPlotNClustersOverSweep:
    def test_lines_carry_the_observed_counts_and_their_standard_errors(self):
        ax = plot_n_clusters_over_sweep(_realized_count_df())
        curves = {c.get_label(): c for c in ax.containers}
        assert set(curves) == {"LeidenClustering", "LouvainClustering"}

        for label, counts, ses in (
            ("LeidenClustering", [3.2, 6.6, 11.0], [0.2, 0.4, 0.0]),
            ("LouvainClustering", [2.2, 5.0, 9.4], [0.1, 0.3, 0.6]),
        ):
            data_line, _, (bars,) = curves[label]
            np.testing.assert_allclose(data_line.get_xdata(), [0.5, 1.0, 2.0])
            np.testing.assert_allclose(data_line.get_ydata(), counts)
            half_widths = [
                (seg[1][1] - seg[0][1]) / 2 for seg in bars.get_segments()
            ]
            np.testing.assert_allclose(half_widths, ses, atol=1e-12)

    @pytest.mark.parametrize(
        ("measure", "rule", "not_two", "x", "label"),
        [
            (
                "stability",
                "max",
                False,
                1.0,
                "Selected resolution (Max rule): 1, 7 clusters",
            ),
            (
                "stability",
                "1se",
                False,
                2.0,
                "Selected resolution (1-SE rule): 2, 11 clusters",
            ),
            (
                "generalizability",
                "max",
                False,
                0.5,
                "Selected resolution (Max rule): 0.5, 2 clusters",
            ),
            (
                "generalizability",
                "max",
                True,
                1.0,
                "Selected resolution (Max rule): 1, 7 clusters",
            ),
        ],
    )
    def test_marks_the_selected_sweep_value_and_its_count(
        self, measure, rule, not_two, x, label
    ):
        ax = plot_n_clusters_over_sweep(
            _realized_count_df(), measure=measure, rule=rule, not_two=not_two
        )
        (dashed,) = _dashed(ax)
        assert list(dashed.get_xdata()) == [x, x]
        assert dashed.get_label() == label

    def test_axis_labels(self):
        ax = plot_n_clusters_over_sweep(_realized_count_df())
        assert ax.get_xlabel() == "Resolution"
        assert ax.get_ylabel() == "Mean Observed Number of Clusters"
        ax = plot_n_clusters_over_sweep(_realized_count_df(), ylabel="Clusters")
        assert ax.get_ylabel() == "Clusters"

    def test_y_ticks_are_whole_numbers(self):
        # Between 2 and 3 clusters the default locator ticks at 1.8, 2.0, ...
        df = with_sweep_cols(
            pd.DataFrame(
                {
                    "estimator": ["LeidenClustering"] * 3,
                    "resolution": [0.5, 1.0, 2.0],
                    "ari_stability": [0.9, 0.8, 0.7],
                    "ari_stability_se": [0.01, 0.01, 0.01],
                }
            ),
            param="resolution",
            method_label="LeidenClustering",
            observed=[2.0, 2.5, 3.0],
        )
        ticks = plot_n_clusters_over_sweep(df).get_yticks()
        np.testing.assert_array_equal(ticks, np.round(ticks))

    def test_a_sweep_over_n_clusters_raises(self, metric_results_df):
        with pytest.raises(ValueError, match="other than n_clusters"):
            plot_n_clusters_over_sweep(metric_results_df)

    def test_unknown_measure_raises(self):
        with pytest.raises(ValueError, match="not found"):
            plot_n_clusters_over_sweep(_realized_count_df(), measure="nonexistent")

    def test_empty_df(self):
        with pytest.raises(RuntimeError, match="empty"):
            plot_n_clusters_over_sweep(pd.DataFrame())

    def test_save_writes_the_file_and_returns_none(self, tmp_path):
        path = tmp_path / "n_clusters.png"
        assert plot_n_clusters_over_sweep(_realized_count_df(), save=path) is None
        assert path.exists()
```

- [ ] Step 2: Run the tests to verify they fail

```bash
.venv/bin/pytest tests/test_plotting.py -q
```

Expected: an error during collection, `ImportError: cannot import name 'plot_n_clusters_over_sweep' from 'carve._plotting'`.

- [ ] Step 3: Implement

In `src/carve/_plotting.py`, replace:

```python
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
```

with:

```python
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
```

In `src/carve/_plotting.py`, replace:

```python
from ._sweep import (
    sweep_axis_label,
    sweep_param_name,
)
```

with:

```python
from ._sweep import (
    observed_k,
    sweep_axis_label,
    sweep_param_name,
)
```

In `src/carve/_plotting.py`, replace:

```python
    selection_label: Callable[[pd.Series], str],
    rule: str,
    ax: Axes | None,
```

with:

```python
    selection_label: Callable[[pd.Series], str],
    rule: str,
    integer_y: bool = False,
    ax: Axes | None,
```

This is valid Python: after `*`, a keyword-only parameter with a default may come before ones without.

In `src/carve/_plotting.py`, replace:

```python
    is left off.
    """
    se_col = f"{y_col}_se"
```

with:

```python
    is left off.

    ``integer_y`` puts the y ticks on whole numbers, for counts.
    """
    se_col = f"{y_col}_se"
```

In `src/carve/_plotting.py`, replace:

```python
    x_unique = sorted(results_df[x_col].unique())
    if len(x_unique) <= 20:
        ax.set_xticks(x_unique)
```

with:

```python
    x_unique = sorted(results_df[x_col].unique())
    if len(x_unique) <= 20:
        ax.set_xticks(x_unique)

    if integer_y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
```

In `src/carve/_plotting.py`, replace:

```python
        dpi=dpi,
        **kwargs,
    )


def _selected_row_for(
```

with:

```python
        dpi=dpi,
        **kwargs,
    )


def plot_n_clusters_over_sweep(
    results_df: pd.DataFrame,
    *,
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    palette: str = "Accent",
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
    **kwargs,
) -> Axes | None:
    """Plot the realized number of clusters across a sweep that does not fix k.

    When a run sweeps ``resolution``, ``min_cluster_size`` or another
    parameter that is not ``n_clusters``, the number of clusters is an
    outcome of the sweep. This plot draws it against the swept value, one
    line per estimator configuration (``method_id``), with error bars at
    +/-1 standard error. A vertical dashed line marks the sweep value
    selected under ``measure``, ``rule`` and ``not_two``, and its legend
    entry states the count there. The x axis matches
    :func:`plot_metric_over_n_clusters`, so the two plots can share it.

    The count is ``n_clusters_observed``: the mean, over resamples, of the
    number of clusters in the clustering of each resample's training
    subsample, counted after the noise policy. Under ``"drop"`` noise points
    are removed before counting; under ``"as_cluster"`` the noise label
    counts as one cluster; under ``"singleton"`` each noise point counts as
    its own cluster. It is not the count of one fit on the full data. At the
    selected sweep value it rounds to ``CARVE.get_k()``, the count
    ``CARVE.get_labels()`` cuts the consensus at by default. The error bars
    are the standard error of that mean, not the spread of the count across
    resamples.

    Parameters
    ----------
    results_df : pd.DataFrame
        ``estimator_results_`` from a CARVE fit that swept a parameter other
        than ``n_clusters``.
    measure : str, default="stability"
        Metric used to select the marked sweep value. Options as in
        :func:`plot_metric_over_n_clusters`.
    rule : str, default="1se"
        Selection rule for the marked sweep value: "max", "1se", "quantile".
    not_two : bool, default=False
        Exclude configurations whose rounded count is two when selecting.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (9, 5.5).
    title : str, optional
        Figure title.
    xlabel : str, optional
        X-axis label. Default is derived from the sweep parameter.
    ylabel : str, optional
        Y-axis label. Default is "Mean Observed Number of Clusters".
    legend : bool, default=True
        Whether to display a legend.
    legend_loc : str, default="best"
        Legend location (passed to ax.legend).
    palette : str, default="Accent"
        Matplotlib colormap name for line colors.
    show : bool, default=False
        Whether to call plt.show() before returning.
    save : str or Path, optional
        Path to save the figure. If provided, the figure is saved and None
        is returned instead of the Axes object.
    dpi : int, default=300
        Dots per inch for saved figures.
    **kwargs
        Additional keyword arguments passed to ax.errorbar().

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None if save is used.

    Raises
    ------
    RuntimeError
        If results_df is empty.
    ValueError
        If the table sweeps ``n_clusters``, or ``measure`` is unknown or
        names a column the table does not carry.

    Examples
    --------
    >>> from carve.api import CARVE
    >>> carve = CARVE(resolution=[0.25, 0.5, 1.0, 2.0]).fit(X)
    >>> ax = plot_n_clusters_over_sweep(carve.estimator_results_)
    >>> ax.set_xscale("log")
    """
    if results_df.empty:
        raise RuntimeError("Results DataFrame is empty.")

    param = sweep_param_name(results_df)
    if param == "n_clusters":
        raise ValueError(
            "plot_n_clusters_over_sweep needs a sweep over a parameter other "
            "than n_clusters. This table sweeps n_clusters, where the realized "
            "count is the swept value itself; use plot_metric_over_n_clusters."
        )
    _measure_column(results_df, measure)

    def selection_label(row: pd.Series) -> str:
        marked = _selection_label(row, param=param, rule=rule)
        return f"{marked}, {observed_k(row)} clusters"

    return _draw_metric_lines(
        results_df,
        y_col="n_clusters_observed",
        group_col="method_id",
        label_of=_build_estimator_label,
        legend_title="Estimators",
        select_row=lambda: select_best_row_by_rule(
            results_df, measure=measure, rule=rule, not_two=not_two
        ),
        selection_label=selection_label,
        rule=rule,
        integer_y=True,
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel if ylabel is not None else "Mean Observed Number of Clusters",
        legend=legend,
        legend_loc=legend_loc,
        palette=palette,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def _selected_row_for(
```

- [ ] Step 4: Run the tests to verify they pass

```bash
.venv/bin/pytest tests/test_plotting.py -q
```

Expected: no failures or errors.

- [ ] Step 5: Lint and format

```bash
.venv/bin/ruff check src/
.venv/bin/ruff format src/
```

Expected: `All checks passed!`. If ruff reformatted `src/carve/_plotting.py`, rerun Step 4.

- [ ] Step 6: Commit

```bash
git add src/carve/_plotting.py tests/test_plotting.py
git commit -m "feat(plotting): plot the realized cluster count over a non-k sweep" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] Step 7: Mutation checks

Spec 6.2. Apply each mutation on its own, run the command, compare with the expected result, then restore the committed file before the next one. Every change below is inside `plot_n_clusters_over_sweep`, except (b), which is in `_draw_metric_lines`.

```bash
# after each mutation:
.venv/bin/pytest tests/test_plotting.py -q -k "NClustersOverSweep or SharedMetricDrawing"
git checkout -- src/carve/_plotting.py
```

| | Mutation | Expected |
|---|---|---|
| a | `y_col="n_clusters_observed"` to `y_col="ari_stability"` | fails, including `test_lines_carry_the_observed_counts_and_their_standard_errors` and `test_both_metric_plots_draw_through_one_helper` |
| b | in `_draw_metric_lines`, `se_col = f"{y_col}_se"` to `se_col = f"{y_col}_sd"` | fails, including `test_lines_carry_the_observed_counts_and_their_standard_errors` (no error bars to unpack) |
| c | `{observed_k(row)} clusters` to `{int(float(row['n_clusters_observed']))} clusters` | 2 failed: the two marker cases expecting `1, 7 clusters` |
| d | in `select_row`, `measure=measure` to `measure="stability"` | 1 failed: the marker case expecting `0.5, 2 clusters` |
| e | in `select_row`, `rule=rule` to `rule="1se"` | 1 failed: the stability `max` marker case |
| f | in `select_row`, `not_two=not_two` to `not_two=False` | 1 failed: the generalizability `max` case with `not_two=True` |
| g | delete the `if param == "n_clusters": raise ...` block | 1 failed: `test_a_sweep_over_n_clusters_raises` |
| h | `integer_y=True` to `integer_y=False` | 1 failed: `test_y_ticks_are_whole_numbers` |
| i | delete the line `_measure_column(results_df, measure)` | 1 failed: `test_unknown_measure_raises` |

Why (d) and (e) catch only one case each. Under (d), stability `max` with `not_two` also marks Leiden at 1.0. Under (e), generalizability under "1se" marks the same rows as under "max", because the peaks sit more than one standard error above the next row. After the last checkout, `git status --short` prints nothing.

---

### Task 3: `CARVE.plot_n_clusters_over_sweep`

Spec 5.2.

Files:
- Modify: `src/carve/api.py`
- Test: `tests/test_api.py`

Interfaces:
- Consumes: `_plotting.plot_n_clusters_over_sweep` from Task 2, with the signature given there.
- Produces: `CARVE.plot_n_clusters_over_sweep(self, *, measure="stability", rule="1se", not_two=False, ax=None, figsize=None, title=None, xlabel=None, ylabel=None, legend=True, legend_loc="best", palette="Accent", show=False, save=None, dpi=300, **kwargs) -> Axes | None`. Raises `RuntimeError("Call fit() first.")` before `fit()`.

- [ ] Step 1: Write the failing tests

In `tests/test_api.py`, replace:

```python
    def test_plot_xlabel(self, fitted_hdbscan):
        ax = fitted_hdbscan.plot_metric_over_n_clusters()
        assert ax.get_xlabel() == "Minimum Cluster Size"
```

with:

```python
    def test_plot_xlabel(self, fitted_hdbscan):
        ax = fitted_hdbscan.plot_metric_over_n_clusters()
        assert ax.get_xlabel() == "Minimum Cluster Size"

    def test_plot_n_clusters_over_sweep_marks_the_selection_it_is_asked_for(
        self, fitted_hdbscan
    ):
        # min_cluster_size runs coarse to fine as 8, 5, 3. Stability peaks at
        # size 3. Generalizability peaks at size 8 and reaches 0.98 at size 5,
        # inside one standard error of the peak, and size 8's count is two
        # clusters. So the default call marks 3; generalizability under "max"
        # marks 8, where "1se" would mark the finer 5; and not_two=True
        # excludes 8 and marks 5. A method that drops measure, rule or
        # not_two marks the wrong size in one of the three calls.
        carve = copy.deepcopy(fitted_hdbscan)
        results = carve.estimator_results_
        size = results["sweep_value"]
        results["ari_stability"] = np.where(size == 3, 1.0, 0.0)
        results["ari_stability_se"] = 0.01
        results["ari_generalizability"] = np.select(
            [size == 8, size == 5], [1.0, 0.98], default=0.0
        )
        results["ari_generalizability_se"] = 0.05
        results["n_clusters_observed"] = np.where(size == 8, 2.0, 4.0)

        def marked(**kwargs):
            ax = carve.plot_n_clusters_over_sweep(**kwargs)
            (dashed,) = [
                line for line in ax.get_lines() if line.get_linestyle() == "--"
            ]
            return dashed.get_xdata()[0]

        assert marked() == 3
        assert marked(measure="generalizability", rule="max") == 8
        assert marked(measure="generalizability", rule="max", not_two=True) == 5
```

In `tests/test_api.py`, replace:

```python
    def test_plot_metric_unfitted(self):
        carve = CARVE(verbose=0)
        with pytest.raises(RuntimeError, match="fit"):
            carve.plot_metric_over_n_clusters()
```

with:

```python
    def test_plot_metric_unfitted(self):
        carve = CARVE(verbose=0)
        with pytest.raises(RuntimeError, match="fit"):
            carve.plot_metric_over_n_clusters()

    def test_plot_n_clusters_over_sweep_unfitted(self):
        with pytest.raises(RuntimeError, match="Call fit"):
            CARVE(verbose=0).plot_n_clusters_over_sweep()
```

- [ ] Step 2: Run the tests to verify they fail

```bash
.venv/bin/pytest tests/test_api.py -q -k "n_clusters_over_sweep"
```

Expected: `2 failed`, each with `AttributeError: 'CARVE' object has no attribute 'plot_n_clusters_over_sweep'`.

- [ ] Step 3: Implement

In `src/carve/api.py`, replace:

```python
from ._plotting import (
    plot_metric_over_n_clusters as _plot_metric_over_n_clusters,
)
```

with:

```python
from ._plotting import (
    plot_metric_over_n_clusters as _plot_metric_over_n_clusters,
)
from ._plotting import (
    plot_n_clusters_over_sweep as _plot_n_clusters_over_sweep,
)
```

In `src/carve/api.py`, replace:

```python
            dpi=dpi,
            **kwargs,
        )

    def plot_consensus_matrix(
```

with:

```python
            dpi=dpi,
            **kwargs,
        )

    def plot_n_clusters_over_sweep(
        self,
        *,
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
        ax=None,
        figsize: tuple | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        palette: str = "Accent",
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
        **kwargs,
    ) -> Axes | None:
        """Plot the realized number of clusters across a sweep that does not fix k.

        For a run over ``resolution``, ``min_cluster_size`` or another
        parameter that is not ``n_clusters``. Draws ``n_clusters_observed``
        from ``estimator_results_`` against the swept value, one line per
        estimator configuration, with error bars at +/-1 standard error. The
        dashed line marks the sweep value selected under ``measure``,
        ``rule`` and ``not_two``. Its legend entry states the count there,
        which is what ``get_k`` returns for the same arguments.

        The count is the mean, over resamples, of the number of clusters in
        the clustering of each resample's training subsample, counted after
        ``noise_policy``. Under ``"drop"`` noise points are removed before
        counting; under ``"as_cluster"`` the noise label counts as one
        cluster; under ``"singleton"`` each noise point counts as its own
        cluster. It is not the count of one fit on the full data;
        ``get_labels`` cuts the consensus at the rounded count by default.
        The error bars are the standard error of the mean, not the spread of
        the count across resamples.

        Parameters
        ----------
        measure : str, default="stability"
            Metric used to select the marked sweep value. Options as in
            ``plot_metric_over_n_clusters``.
        rule : str, default="1se"
            Selection rule: "max", "1se", "quantile".
        not_two : bool, default=False
            Whether to exclude configurations whose rounded count is two
            when selecting.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (9, 5.5).
        title : str, optional
            Figure title.
        xlabel : str, optional
            X-axis label. Default is derived from the sweep parameter.
        ylabel : str, optional
            Y-axis label. Default is "Mean Observed Number of Clusters".
        legend : bool, default=True
            Whether to display a legend showing estimator labels.
        legend_loc : str, default="best"
            Legend location (passed to matplotlib's ax.legend).
        palette : str, default="Accent"
            Matplotlib colormap name for line colors.
        show : bool, default=False
            Whether to call plt.show() before returning.
        save : str or Path, optional
            Path to save the figure. If provided, the figure is saved and
            None is returned instead of an Axes object.
        dpi : int, default=300
            Dots per inch for saved figures.
        **kwargs
            Additional keyword arguments passed to matplotlib's errorbar
            function.

        Returns
        -------
        ax : matplotlib.axes.Axes or None
            The Axes object, or None if save was used.

        Raises
        ------
        RuntimeError
            If the instance has not been fitted yet.
        ValueError
            If the run swept ``n_clusters``, or ``measure`` is unknown.

        Examples
        --------
        >>> carve = CARVE(resolution=[0.25, 0.5, 1.0, 2.0]).fit(X)
        >>> ax = carve.plot_n_clusters_over_sweep(measure="stability", rule="1se")
        >>> ax.set_xscale("log")
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        return _plot_n_clusters_over_sweep(
            self.estimator_results_,
            measure=measure,
            rule=rule,
            not_two=not_two,
            ax=ax,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
            legend_loc=legend_loc,
            palette=palette,
            show=show,
            save=save,
            dpi=dpi,
            **kwargs,
        )

    def plot_consensus_matrix(
```

- [ ] Step 4: Run the tests to verify they pass

```bash
.venv/bin/pytest tests/test_api.py -q -k "n_clusters_over_sweep or MinClusterSizeMode or TestPlotting"
```

Expected: no failures or errors.

- [ ] Step 5: Lint and format

```bash
.venv/bin/ruff check src/
.venv/bin/ruff format src/
```

Expected: `All checks passed!`. If ruff reformatted `src/carve/api.py`, rerun Step 4.

- [ ] Step 6: Commit

```bash
git add src/carve/api.py tests/test_api.py
git commit -m "feat(api): CARVE.plot_n_clusters_over_sweep" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] Step 7: Mutation checks

Apply each on its own, inside the new method's `_plot_n_clusters_over_sweep(...)` call, then restore:

```bash
# after each mutation:
.venv/bin/pytest tests/test_api.py -q -k "test_plot_n_clusters_over_sweep_marks_the_selection_it_is_asked_for"
git checkout -- src/carve/api.py
```

| | Mutation | Expected, 1 failed on the assertion |
|---|---|---|
| a | `measure=measure` to `measure="stability"` | `marked(measure="generalizability", rule="max")` returns 3, not 8 |
| b | `rule=rule` to `rule="1se"` | `marked(measure="generalizability", rule="max")` returns 5, not 8 |
| c | `not_two=not_two` to `not_two=False` | `marked(..., not_two=True)` returns 8, not 5 |

---

### Task 4: `pl.n_clusters_over_sweep`

Spec 5.3.

Files:
- Modify: `src/carve/pl/_plots.py`, `src/carve/pl/__init__.py`
- Test: `tests/test_pl.py`

Interfaces:
- Consumes: `_plotting.plot_n_clusters_over_sweep` from Task 2, and `CARVE.plot_n_clusters_over_sweep` from Task 3 (in the comparison test).
- Produces: `carve.pl.n_clusters_over_sweep(adata, *, key="carve", measure=None, rule=None, not_two=None, ax=None, figsize=None, title=None, xlabel=None, ylabel=None, legend=True, legend_loc="best", palette="Accent", show=False, save=None, dpi=300, **kwargs) -> Axes | None`, calling the module-level alias `pl._plots._plot_n_clusters_over_sweep`.

- [ ] Step 1: Write the failing tests

In `tests/test_pl.py`, replace:

```python
    path = tmp_path_factory.mktemp("h5ad") / "randomized.h5ad"
    adata.write_h5ad(path)
    return ad.read_h5ad(path)
```

with:

```python
    path = tmp_path_factory.mktemp("h5ad") / "randomized.h5ad"
    adata.write_h5ad(path)
    return ad.read_h5ad(path)


@pytest.fixture(scope="module")
def sweep_model():
    """CARVE fitted over HDBSCAN's min_cluster_size, a sweep that does not fix k."""
    m = CARVE(
        sweep="min_cluster_size",
        sweep_values=np.array([3, 5, 8]),
        n_resamples=5,
        random_state=0,
    )
    m.fit(_bare_adata(), use_rep="X_pca")
    return m


@pytest.fixture(scope="module")
def sweep_written(sweep_model, tmp_path_factory):
    """The min_cluster_size result written into an AnnData and read back from h5ad."""
    adata = _bare_adata()
    carve.tl.attach_results(adata, sweep_model, use_rep="X_pca")
    path = tmp_path_factory.mktemp("h5ad") / "sweep.h5ad"
    adata.write_h5ad(path)
    return ad.read_h5ad(path)
```

In `tests/test_pl.py`, replace:

```python
    def test_save_writes_the_file_and_returns_none(self, randomized_written, tmp_path):
        path = tmp_path / "metric_by_pipeline.png"
        assert carve.pl.metric_by_pipeline(randomized_written, save=path) is None
        assert path.exists()
```

with:

```python
    def test_save_writes_the_file_and_returns_none(self, randomized_written, tmp_path):
        path = tmp_path / "metric_by_pipeline.png"
        assert carve.pl.metric_by_pipeline(randomized_written, save=path) is None
        assert path.exists()


# -----------------------------------------------------------------------
# n_clusters_over_sweep
# -----------------------------------------------------------------------


class TestNClustersOverSweep:
    def test_matches_the_model_method_after_an_h5ad_round_trip(
        self, sweep_written, sweep_model
    ):
        a = carve.pl.n_clusters_over_sweep(sweep_written)
        b = sweep_model.plot_n_clusters_over_sweep(measure="stability", rule="1se")
        assert len(a.containers) == len(b.containers) > 0
        for ca, cb in zip(a.containers, b.containers):
            assert ca.get_label() == cb.get_label()
            np.testing.assert_allclose(ca[0].get_xdata(), cb[0].get_xdata())
            np.testing.assert_allclose(ca[0].get_ydata(), cb[0].get_ydata())
            np.testing.assert_allclose(
                ca[2][0].get_segments(), cb[2][0].get_segments()
            )
        (dashed_a,) = [line for line in a.get_lines() if line.get_linestyle() == "--"]
        (dashed_b,) = [line for line in b.get_lines() if line.get_linestyle() == "--"]
        assert dashed_a.get_label() == dashed_b.get_label()
        assert dashed_a.get_xdata()[0] == sweep_model.get_sweep_value()

    def test_defaults_come_from_the_recorded_selection(
        self, sweep_written, monkeypatch
    ):
        # The recorded params are set to non-default values so that a wrapper
        # hard-coding "stability", "1se" and False could not pass by accident.
        seen = {}

        def spy(table, **kwargs):
            seen.update(kwargs, n_rows=len(table))

        monkeypatch.setattr(pl_plots, "_plot_n_clusters_over_sweep", spy)
        adata = sweep_written.copy()
        adata.uns["carve"]["params"]["measure"] = "generalizability"
        adata.uns["carve"]["params"]["rule"] = "max"
        adata.uns["carve"]["params"]["not_two"] = True
        carve.pl.n_clusters_over_sweep(adata)
        assert (seen["measure"], seen["rule"], seen["not_two"]) == (
            "generalizability",
            "max",
            True,
        )
        assert seen["n_rows"] == len(sweep_written.uns["carve"]["results"])

        carve.pl.n_clusters_over_sweep(
            adata, measure="pac", rule="1se", not_two=False
        )
        assert (seen["measure"], seen["rule"], seen["not_two"]) == (
            "pac",
            "1se",
            False,
        )

    def test_save_writes_the_file_and_returns_none(self, sweep_written, tmp_path):
        path = tmp_path / "n_clusters_over_sweep.png"
        assert carve.pl.n_clusters_over_sweep(sweep_written, save=path) is None
        assert path.exists()
```

- [ ] Step 2: Run the tests to verify they fail

```bash
.venv/bin/pytest tests/test_pl.py -q -k "NClustersOverSweep"
```

Expected: `3 failed`. Two fail with `AttributeError: module 'carve.pl' has no attribute 'n_clusters_over_sweep'`; `test_defaults_come_from_the_recorded_selection` fails at `monkeypatch.setattr` because `carve.pl._plots` has no attribute `_plot_n_clusters_over_sweep`.

- [ ] Step 3: Implement

In `src/carve/pl/_plots.py`, replace:

```python
from .._plotting import (
    plot_metric_over_n_clusters as _plot_metric_over_n_clusters,
)
```

with:

```python
from .._plotting import (
    plot_metric_over_n_clusters as _plot_metric_over_n_clusters,
)
from .._plotting import (
    plot_n_clusters_over_sweep as _plot_n_clusters_over_sweep,
)
```

In `src/carve/pl/_plots.py`, replace:

```python
    "metric_by_pipeline",
    "metric_over_n_clusters",
]
```

with:

```python
    "metric_by_pipeline",
    "metric_over_n_clusters",
    "n_clusters_over_sweep",
]
```

In `src/carve/pl/_plots.py`, replace:

```python
        dpi=dpi,
        **kwargs,
    )


def consensus_matrix(
```

with:

```python
        dpi=dpi,
        **kwargs,
    )


def n_clusters_over_sweep(
    adata: AnnData,
    *,
    key: str = "carve",
    measure: str | None = None,
    rule: str | None = None,
    not_two: bool | None = None,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    palette: str = "Accent",
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
    **kwargs,
) -> Axes | None:
    """Plot the realized number of clusters across a sweep that does not fix k.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix carrying results from a :func:`carve.tl.carve`
        run over a parameter other than ``n_clusters``.
    key : str, default="carve"
        The ``key_added`` used when the results were written.
    measure : str, optional
        Metric used to select the marked sweep value. Defaults to the
        measure recorded at fit time, so the marker matches the labels in
        ``adata.obs``.
    rule : str, optional
        Selection rule for the marked sweep value. Defaults to the recorded
        rule.
    not_two : bool, optional
        Whether two-cluster solutions are excluded when selecting. Defaults
        to the recorded value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when None.
    figsize : tuple, optional
        Figure size, used only when ``ax`` is None.
    title, xlabel, ylabel : str, optional
        Axis text overrides.
    legend : bool, default=True
        Draw the legend.
    legend_loc : str, default="best"
        Legend location.
    palette : str, default="Accent"
        Matplotlib colormap name used for the per-method colors.
    show : bool, default=False
        Call ``plt.show()`` before returning.
    save : str or pathlib.Path, optional
        Write the figure to this path and return None.
    dpi : int, default=300
        Resolution used when saving.
    **kwargs
        Forwarded to the underlying line plot.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None when ``save`` is given.

    Raises
    ------
    KeyError
        If ``adata.uns[key]`` or its results table is absent.
    ValueError
        If the stored results sweep ``n_clusters``.

    Notes
    -----
    The count is the mean over resamples of the number of clusters in the
    clustering of each resample's training subsample, not the count of one
    fit on the full data. See :meth:`carve.CARVE.plot_n_clusters_over_sweep`.
    """
    params = dict(_entry(adata, key)["params"])
    return _plot_n_clusters_over_sweep(
        _results(adata, key),
        measure=(
            measure if measure is not None else str(params.get("measure", "stability"))
        ),
        rule=rule if rule is not None else str(params.get("rule", "1se")),
        not_two=(
            not_two if not_two is not None else bool(params.get("not_two", False))
        ),
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
        legend_loc=legend_loc,
        palette=palette,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def consensus_matrix(
```

In `src/carve/pl/__init__.py`, replace:

```python
    metric_by_pipeline,
    metric_over_n_clusters,
)
```

with:

```python
    metric_by_pipeline,
    metric_over_n_clusters,
    n_clusters_over_sweep,
)
```

In `src/carve/pl/__init__.py`, replace:

```python
    "metric_by_pipeline",
    "metric_over_n_clusters",
]
```

with:

```python
    "metric_by_pipeline",
    "metric_over_n_clusters",
    "n_clusters_over_sweep",
]
```

- [ ] Step 4: Run the tests to verify they pass

```bash
.venv/bin/pytest tests/test_pl.py -q
```

Expected: no failures or errors.

- [ ] Step 5: Lint and format

```bash
.venv/bin/ruff check src/
.venv/bin/ruff format src/
```

Expected: `All checks passed!`. If ruff reformatted a file, rerun Step 4.

- [ ] Step 6: Commit

```bash
git add src/carve/pl/_plots.py src/carve/pl/__init__.py tests/test_pl.py
git commit -m "feat(pl): pl.n_clusters_over_sweep reads the recorded selection" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] Step 7: Mutation checks

Apply each on its own, inside `n_clusters_over_sweep`, then restore:

```bash
# after each mutation:
.venv/bin/pytest tests/test_pl.py -q -k "test_defaults_come_from_the_recorded_selection"
git checkout -- src/carve/pl/_plots.py
```

| | Mutation | Expected |
|---|---|---|
| a | the `measure=(...)` argument to `measure=measure or "stability"` | 1 failed: the recorded `"generalizability"` is not passed |
| b | the `rule=...` argument to `rule=rule if rule is not None else "1se"` | 1 failed |
| c | the `not_two=(...)` argument to `not_two=bool(not_two)` | 1 failed |
| d | the `measure=(...)` argument to `measure=str(params.get("measure", "stability"))` | 1 failed: the explicit `measure="pac"` is not passed |

---

### Task 5: Full verification

No code changes. This task checks the branch the way CI does.

- [ ] Step 1: Lint and format check

```bash
.venv/bin/ruff check src/
.venv/bin/ruff format --check src/
```

Expected: `All checks passed!` and `66 files already formatted`.

- [ ] Step 2: The carve leg, with CI's flags

```bash
.venv/bin/pytest tests --ignore=tests/benchmarks --tb=short --cov=carve --cov-report=term-missing --cov-fail-under=75 -q
```

Expected: no failures or errors, no warnings summary, 899 tests (882 on `main`, plus 17: one in Task 1, eleven in Task 2, two in Task 3, three in Task 4), and `Required test coverage of 75% reached`. It takes several minutes. pytest-xdist is not installed, so do not pass `-n`.

- [ ] Step 3: The benchmarks leg

```bash
.venv/bin/pytest tests/benchmarks --tb=short -q
```

Expected: the same pass and skip counts as on `main`, and no failures. This plan changes no benchmark code, but `src/benchmarks/` calls `plot_metric_over_n_clusters` and `plot_metric_by_pipeline`, whose shared drawing Task 1 changed. About seven minutes.

- [ ] Step 4: Confirm the scope

```bash
git diff --stat main..HEAD
```

Expected: only the spec, this plan, `src/carve/_plotting.py`, `src/carve/api.py`, `src/carve/pl/_plots.py`, `src/carve/pl/__init__.py`, `tests/test_plotting.py`, `tests/test_api.py` and `tests/test_pl.py`. Nothing under `carve-r/`, `src/benchmarks/`, `notebooks/` or `.github/`.

- [ ] Step 5: Finish the branch

Use superpowers:finishing-a-development-branch.

---

## After this plan

Not part of this branch:

- The Cusanovich figure could replace its hand-built secondary axis (`_observed_cluster_axis`) with a panel drawn by `plot_n_clusters_over_sweep`. That is a figure decision for the author.
- A cell in `notebooks/Resolution_Tutorial.ipynb` showing the plot.
- The R port, if the author wants `plotting.R` to follow.
