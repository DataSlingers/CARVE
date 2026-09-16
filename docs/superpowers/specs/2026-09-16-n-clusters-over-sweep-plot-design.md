# Realized cluster count over a non-k sweep: design

Date: 2026-09-16
Status: awaiting author review, then planning with /superpowers:writing-plans
Supersedes: nothing
Related: nothing

## 1. Motivation

When a CARVE run sweeps a parameter other than `n_clusters` (Leiden and Louvain `resolution`,
HDBSCAN `min_cluster_size`, or a custom parameter), the number of clusters is an outcome of the
sweep rather than an input. The usual way to read such an axis is a plot of the realized cluster
count against the swept parameter. CARVE records that count for every configuration but has no plot
for it. The Cusanovich case-study figure builds a partial version by hand, as a secondary x axis
(`src/benchmarks/figures/_cusanovich_results.py::_observed_cluster_axis`).

This design adds that plot as a native CARVE plot, on the same three surfaces as the existing
plots.

## 2. What already exists

Read on 2026-09-16 at `87b9f40`.

- `estimator_results_` carries `n_clusters_observed` and `n_clusters_observed_se` on every row
  (`src/carve/_runner.py`, the "Observed granularity" block). They are the mean and the standard
  error of the mean, over resamples, of `n_clusters_train`: the number of distinct labels in the
  clustering of the resample's training subsample P1, counted after `apply_noise_policy`. It is not
  the count of one fit on the full data.
- In a non-k sweep, `get_k()` returns this mean rounded (`_sweep.observed_k`) at the selected row,
  and `get_labels()` cuts the consensus dendrogram at that count by default.
- `preprocessing_results_` carries `n_clusters_observed` but no standard error.
- `_anndata.results_to_uns` passes numeric columns through untouched, so both columns survive an
  h5ad round trip into `adata.uns[key]["results"]`.
- Each plot exists on three surfaces: a function in `src/carve/_plotting.py` that takes a results
  table; a `CARVE.plot_*` method in `src/carve/api.py` that raises before `fit()` and forwards
  `self.estimator_results_`; and a wrapper in `src/carve/pl/_plots.py` that reads the table with
  `_results(adata, key)` and defaults `measure`, `rule` and `not_two` to the values `tl.carve`
  recorded in `params`. The `api` and `pl` modules import the `_plotting` functions under a
  leading-underscore alias (`plot_metric_by_pipeline as _plot_metric_by_pipeline`).
- `_plotting._draw_metric_lines` draws both `plot_metric_over_n_clusters` and
  `plot_metric_by_pipeline`. `tests/test_plotting.py::TestSharedMetricDrawing` asserts that both go
  through it. No other code calls it.
- `_sweep.py` provides `sweep_param_name`, `sweep_axis_label` and `observed_k`.

## 3. Decisions

Taken in the 2026-09-16 brainstorming session.

Error bars. The bars show plus or minus one standard error, from `n_clusters_observed_se`, the same
statistic the metric plots draw. Two alternatives were rejected. The spread of the count over
resamples (standard deviation, or minimum to maximum) is not stored; showing it would need new
runner columns, a change to the results table and a change to the results stored in AnnData. No
error bars at all was also rejected.

Scope. Plotting only. No change to `fit`, the runner, the results tables or `tl.carve`.

Surfaces and names. All three surfaces:

- `carve._plotting.plot_n_clusters_over_sweep`
- `CARVE.plot_n_clusters_over_sweep`
- `carve.pl.n_clusters_over_sweep`

Shared drawing. The new plot draws through `_draw_metric_lines`, generalized as in section 5.1, so
the three line plots cannot drift apart.

Selection marker. A dashed vertical line at the selected sweep value, as in the metric plots, so the
plot can sit under a metric plot on a shared x axis. Its legend entry also states the count.

A sweep over `n_clusters` raises `ValueError`. There the count is the swept value itself.

Out of scope:

- An R port. The rule that `carve-r/R/*.R` mirrors the Python modules was removed from CLAUDE.md on
  2026-09-16, and `plotting.R` has no counterpart of `plot_metric_by_pipeline` either.
- A per-pipeline variant reading `preprocessing_results_`.
- A new cell in `notebooks/Resolution_Tutorial.ipynb`, and README changes.
- A log-scale option. The Axes is returned; callers can call `ax.set_xscale("log")`.

## 4. Behavior

### 4.1 What is drawn

- x: `sweep_value`. Default label from `sweep_axis_label` ("Resolution", "Minimum Cluster Size", or
  the title-cased custom parameter). Ticks at the sweep values when there are at most 20, as today.
- y: `n_clusters_observed`, with `n_clusters_observed_se` as the error bar when that column is
  present. Default label "Mean Observed Number of Clusters".
- One line per `method_id`, labeled with `method_label`. Colors, marker, line width, caps and alpha
  as in `plot_metric_over_n_clusters`; legend title "Estimators"; `**kwargs` reach
  `ax.errorbar`.
- The dashed line marks the `sweep_value` of
  `select_best_row_by_rule(results_df, measure=measure, rule=rule, not_two=not_two)`. Its label is
  `Selected {param} ({rule_str} rule): {value:g}, {count} clusters`, for example
  "Selected resolution (1-SE rule): 0.75, 12 clusters". `{param}` and `{rule_str}` are formed as
  in the metric plots today. `{count}` is `observed_k(row)`, the rounded mean, which is what
  `get_k()` returns for the same arguments. If the selection raises (for example `not_two` leaves
  no rows), the line is left off, as in the metric plots.
- The y axis uses `MaxNLocator(integer=True)`. Matplotlib relaxes this to fractional ticks when
  fewer than `min_n_ticks` (default 2) integers fall in view, for example when every count lies
  between 2.1 and 2.9. That is accepted.
- Grid, `tight_layout`, `figsize` default (9, 5.5), `title`, `xlabel`, `ylabel`, `legend`,
  `legend_loc`, `palette`, `show`, `save` and `dpi` behave as in the metric plots. With `save`, the
  figure is written and closed and None is returned.
- For `min_cluster_size`, larger values give fewer clusters, so the line falls from left to right.
  The x axis is not inverted.

### 4.2 Errors

- Empty table: `RuntimeError("Results DataFrame is empty.")`, checked first.
- `sweep_param == "n_clusters"`: `ValueError` stating that the plot needs a sweep over a parameter
  other than `n_clusters` and pointing to `plot_metric_over_n_clusters`. The message contains the
  phrase "other than n_clusters".
- `measure` not in `MEASURE_MAP`, or its column absent from the table: `ValueError`, with the
  messages `_draw_metric_lines` raises today ("Measure ... not found. Valid options: ..." and
  "Metric column ... not found in the results table."). The check must run before drawing: an
  unknown measure would otherwise make the selection raise inside the `try` around the marker, and
  the dashed line would disappear without an error.
- `CARVE.plot_n_clusters_over_sweep` before `fit()`: `RuntimeError("Call fit() first.")`.
- `pl.n_clusters_over_sweep`: the existing `_entry` and `_results` errors for a missing key or a
  run written with `store_results=False`.

### 4.3 Docstrings

numpydoc, like the sibling plots, with an Examples section. The `_plotting` function and the method
state:

- what the count is: the mean over resamples of the number of clusters in the clustering of each
  resample's training subsample, after the noise policy. Under `"drop"` noise points are removed
  before counting; under `"as_cluster"` the noise label counts as one cluster; under `"singleton"`
  each noise point counts as its own cluster.
- that it is not the count of one fit on the full data, and that at the selected value it rounds to
  `get_k()`, the count `get_labels()` cuts at by default.
- that the error bars are the standard error of the mean, not the spread of the count.

## 5. Implementation shape

### 5.1 `src/carve/_plotting.py`

New public function, placed after `plot_metric_by_pipeline`:

```python
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
```

`_draw_metric_lines` is generalized. After the change:

- It takes `y_col: str` in place of `measure`. The error-bar column is `f"{y_col}_se"`, which holds
  for both the ARI columns and `n_clusters_observed`.
- Measure validation moves into a small helper, `_measure_column(results_df, measure) -> str`,
  raising the two messages in section 4.2. `plot_metric_over_n_clusters` and the new function call
  it. `plot_metric_by_pipeline` keeps its own stricter check and also passes `y_col`.
- Callers resolve the default y label. The metric plots keep today's derivation from the measure
  column ("ARI Stability"); the new plot uses "Mean Observed Number of Clusters".
- It takes `selection_label: Callable[[pd.Series], str]`, alongside `select_row`, to build the
  dashed line's legend text from the selected row. The metric plots pass a function that produces
  exactly today's text; the new plot appends `, {observed_k(row)} clusters`.
- It takes `integer_y: bool = False`; the new plot passes True. The metric plots are unaffected.
- `rule` stays a parameter; it still controls the legend title. Everything else stays as it is.

No existing plot's output changes: the lines, error bars, labels, ticks and legend text of
`plot_metric_over_n_clusters` and `plot_metric_by_pipeline` stay as they are.

### 5.2 `src/carve/api.py`

`CARVE.plot_n_clusters_over_sweep`, placed after `plot_metric_by_pipeline`, with the parameters of
the `_plotting` function minus `results_df`. It raises `RuntimeError("Call fit() first.")` when
`estimator_results_` is None and otherwise forwards every argument. The function is imported as
`_plot_n_clusters_over_sweep`.

### 5.3 `src/carve/pl/_plots.py` and `src/carve/pl/__init__.py`

`n_clusters_over_sweep(adata, *, key="carve", measure=None, rule=None, not_two=None, ...)`, with
the remaining parameters of the `_plotting` function. It reads `_results(adata, key)`, fills
`measure`, `rule` and `not_two` from the recorded `params` exactly as `metric_over_n_clusters`
does, and calls `_plot_n_clusters_over_sweep` (module-level alias, so tests can patch it). The
function is added to `__all__` in both files.

## 6. Verification

### 6.1 Tests

Tests mirror modules. Expected values are written by hand, not computed with CARVE code.

`tests/test_plotting.py`, a new `TestPlotNClustersOverSweep` class with a two-method
resolution-mode table:

| method_id | method_label | resolution | ari_stability | ari_stability_se | ari_generalizability | n_clusters_observed | n_clusters_observed_se |
|---|---|---|---|---|---|---|---|
| m0 | LeidenClustering | 0.5 | 0.80 | 0.01 | 0.60 | 3.2 | 0.2 |
| m0 | LeidenClustering | 1.0 | 0.90 | 0.03 | 0.70 | 6.6 | 0.4 |
| m0 | LeidenClustering | 2.0 | 0.88 | 0.01 | 0.65 | 11.0 | 0.0 |
| m1 | LouvainClustering | 0.5 | 0.70 | 0.01 | 0.95 | 2.2 | 0.1 |
| m1 | LouvainClustering | 1.0 | 0.75 | 0.01 | 0.50 | 5.0 | 0.3 |
| m1 | LouvainClustering | 2.0 | 0.60 | 0.01 | 0.40 | 9.4 | 0.6 |

(`sweep_rank` 0, 1, 2 by resolution; `ari_generalizability_se` 0.01 throughout; the usual identity
and sweep columns.) Each selection lands on a different row, and two counts differ between
truncation and rounding:

| measure | rule | not_two | marked | label |
|---|---|---|---|---|
| stability | max | False | 1.0 | Selected resolution (Max rule): 1, 7 clusters |
| stability | 1se | False | 2.0 | Selected resolution (1-SE rule): 2, 11 clusters |
| generalizability | max | False | 0.5 | Selected resolution (Max rule): 0.5, 2 clusters |
| generalizability | max | True | 1.0 | Selected resolution (Max rule): 1, 7 clusters |

Under "1se" with stability, the threshold is 0.90 - 0.03 = 0.87, which admits Leiden at 1.0 and
2.0; the finer is 2.0. Under `not_two`, Louvain at 0.5 (2.2, rounded to 2) is excluded.

Tests in the class:

- Each method's line carries the table's x values, counts and error-bar half-widths, and the legend
  names both methods.
- The marker position and label, parametrized over the four rows above.
- The default axis labels ("Resolution", "Mean Observed Number of Clusters") and a `ylabel`
  override.
- The y ticks are whole numbers on a one-method table with counts 2.0, 2.5 and 3.0 and zero
  standard errors. Checked on 2026-09-16: the default locator ticks that range at 1.8, 2.0, ...,
  3.2; `MaxNLocator(integer=True)` at 1, 2, 3, 4.
- A sweep over `n_clusters` raises (use the existing `metric_results_df` fixture), an unknown
  measure raises, an empty table raises.
- `save` writes the file and returns None.

`TestSharedMetricDrawing` is extended so the new plot is the third call recorded by the spy
(`("method_id", "Estimators", 6)`).

`tests/test_api.py`:

- In `TestMinClusterSizeMode`, using the existing `fitted_hdbscan` fixture (sklearn only, no graph
  extra): a deep copy with the results altered so that stability peaks at size 3, generalizability
  peaks at size 8 and then 5, and the size-8 count is set to 2.0. The default call marks 3,
  `measure="generalizability", rule="max"` marks 8, and adding `not_two=True` marks 5. This
  catches a method that drops any of the three arguments.
- In `TestPlotting`: calling the method before `fit()` raises `RuntimeError` matching "Call fit".

`tests/test_pl.py`:

- Module-scoped fixtures: `CARVE(sweep="min_cluster_size", sweep_values=np.array([3, 5, 8]),
  n_resamples=5, random_state=0)` fitted on `_bare_adata()` with `use_rep="X_pca"`, then
  `tl.attach_results` into a fresh AnnData, written to h5ad and read back. During brainstorming this
  fit and round trip were seen to run without warnings under `filterwarnings = error`.
- The wrapper matches the model method after the round trip: the same line labels, x, y and
  error-bar segments, the same dashed label, and the dashed x equal to `get_sweep_value()`.
- Recorded defaults: with a spy on `pl_plots._plot_n_clusters_over_sweep` and the recorded params
  set to non-defaults (`"generalizability"`, `"max"`, True), the wrapper passes those values and
  the full table. Explicit arguments override them.
- `save` writes the file and returns None.

The new wrapper is not added to `PL_FUNCTIONS`: that list runs against the k-sweep `written`
fixture, where the new plot raises.

### 6.2 Mutation checks

Each of the following changes must fail at least one test:

- drawing the measure column, or no column, instead of `n_clusters_observed`
- dropping the standard error from the error bars
- truncating the count in the marker label instead of rounding it
- not forwarding `measure`, `rule` or `not_two` on any of the three surfaces
- removing the `n_clusters` guard
- removing the integer tick locator
- removing the measure validation (the marker then disappears without an error)

### 6.3 Commands

From `code/`, with the pinned toolchain in `.venv`:

```bash
pytest tests/test_plotting.py tests/test_api.py tests/test_pl.py
pytest                      # whole suite, since _draw_metric_lines changes
ruff check src/
ruff format src/
```

The existing tests of `plot_metric_over_n_clusters` and `plot_metric_by_pipeline` must pass
unchanged, apart from the extension of `TestSharedMetricDrawing`.
