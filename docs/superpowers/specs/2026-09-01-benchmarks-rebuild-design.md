# Rebuild the CARVE benchmarking and case-study code

Date: 2026-09-01
Status: approved design, pending implementation plan

## Context

The PLOS Computational Biology submission came back with four reviews
(`overleaf/Review/comments.tex`). Seven of the requested changes land on the benchmarking and
case-study code, and two new case studies are needed: a large scRNA-seq dataset (>500k cells,
R1.1) and a scATAC-seq or multimodal dataset (R1.7). Reviewer 3 also asked directly for the code
to be revised for clarity, so the rebuild is itself part of the response.

The existing code is `notebooks/benchmarking_code/`: 6,200 lines across ten modules in a flat
directory with no `__init__.py`, reached by a `sys.path.insert` hack repeated in four notebooks.
It is neither linted nor tested by CI. The computational logic is sound and is preserved; what
changes is structure, provenance, and the cost of adding a study.

The load-bearing observation from the review: the difficulty benchmark and the scaling benchmark
are the same experiment. `easy/medium/hard` is `axis_name="difficulty_level",
axis_value=0|1|2`; `start/middle/end` is `axis_name="n_total", axis_value=1000|5500|10000`. Not
modeling that is the root cause of two runners with roughly 180 duplicated and since-diverged
lines, two summarizers that are 86 percent identical, a two-headed output schema, and
schema-sniffing (`_infer_axis_cols`) in the plotting layer. Unifying the axis collapses all four,
and makes each remaining reviewer experiment a new axis rather than a new runner.

## Decisions taken

1. Two new case studies: large scRNA-seq (>500k cells) and scATAC-seq/multimodal.
2. Code lives at `src/benchmarks/`, linted and tested in CI, excluded from the wheel.
3. Compute is split from reporting: headless runners write versioned artifacts; notebooks only
   read artifacts and render figures.
4. This plan rebuilds the foundation and ports what exists. The seven reviewer experiments and
   the two new case studies are follow-on plans that slot into the finished framework.
5. Three bugs are fixed and the affected results re-run; the manuscript numbers change.
6. The SNR calibration is rebuilt as a runnable module, with the published anchors retained as a
   committed fallback constant.
7. The moons scaling experiment is dropped.
8. Working runs are gitignored; a published run is promoted to a committed CSV plus its manifest.

## Target layout

```
code/src/benchmarks/
  __init__.py
  _types.py        Axis, SimParams, EstimatorSpec, Scenario, Study, Manifest
  _registry.py     SCENARIOS / STUDIES - the study design, in code
  _calibrate.py    regenerates SNR anchors from target ARI bands
  _simulate.py     Scenario -> (X, y); wraps carve.sim.simulate_clusters
  _estimators.py   EstimatorSpec -> sklearn estimator; strict
  _cvi.py          silhouette, gap, davies_bouldin, calinski_harabasz
  _run.py          the single runner
  _artifacts.py    parquet + manifest.json, content-addressed paths, promote()
  _tables.py       the single summarizer + LaTeX renderer
  _theme.py        palette, rcParams, spines, legends, figure geometry
  _panels.py       ax-in / ax-out drawing primitives
  datasets/        loaders returning (X, y, meta)
  figures/         one module per manuscript figure, each exposing figure_<name>()
  run.py           python -m benchmarks.run
```

Dependency direction follows the package convention already enforced in `src/carve/`: `_types` is
a leaf, nothing under `_*` imports `run`, and the public surface stays small. There is no R
counterpart; the 1:1 mirror rule in CLAUDE.md applies to `src/carve/_*.py` only.

### Why `src/benchmarks/` and not `code/benchmarks/`

The editable install is a single `.pth` file containing `.../code/src` and nothing else. A package
at the repo root would therefore not be importable from a notebook running in `notebooks/`, which
would reintroduce the `sys.path` hack this rebuild sets out to delete. Placing the package under
`src/` and excluding it from `[tool.setuptools.packages.find]` gives both properties at once.

This was verified empirically against a replica project using the same setuptools version, rather
than assumed:

- `pip install -e .` emits a `.pth` pointing at `src`, and `import benchmarks` succeeds even
  though `benchmarks` is listed in `exclude`.
- `python -m build --wheel` produces a wheel containing `carve/` only; `benchmarks/` is absent.

If a future setuptools switches from the static-path strategy to the strict import finder, the
symptom is an `ImportError` on `import benchmarks` immediately after an editable install. Record
this in the benchmarks README next to the install instructions.

Reproducing benchmarks requires a repo checkout with an editable install, because `carve.sim` is
excluded from the wheel. State that in the benchmarks README rather than working around it.

## Core types

`_types.py` holds frozen dataclasses.

```python
@dataclass(frozen=True)
class Axis:
    name: str            # "difficulty" | "n_total" | "p" | "rho" | "n_resamples"
    values: tuple        # (0, 1, 2) | (1000, 5500, 10000)
    labels: tuple        # ("easy", "medium", "hard")

@dataclass(frozen=True)
class Scenario:
    name: str
    axis: Axis
    anchors: Mapping[str, Mapping[str, Any]]   # axis label -> simulate_clusters kwargs
    shared: Mapping[str, Any]
    estimator: EstimatorSpec
    k_star: int = 5
    candidate_k: tuple[int, ...] = (3, 4, 5, 6, 7)
    n_seeds: int = 20
```

`Scenario.__post_init__` validates every anchor and shared key against
`inspect.signature(simulate_clusters)` and rejects keys present in both. This replaces two current
failure modes: a mistyped key is silently ignored today, and a duplicated key raises
`TypeError: got multiple values` from inside the simulator with no indication which dict is at
fault.

`EstimatorSpec` raises on an unknown name. Today `make_estimator_grids`
(`benchmarking_utils.py:163`) has no `else` clause, so `"aglomerative"` silently returns KMeans
while the provenance column records the string that was passed.

`Study` covers case studies: loader, preprocessing, estimator grid, candidate k. It runs through
the same runner with `axis=None`.

Constants that currently live outside the config module move into it: `DIFFICULTY_LABELS` and
`STAGE_LABELS` (`benchmarking_simulation_helpers.py:23`), `n_reference_datasets=10`
(`benchmarking_metrics.py:102`), `n_init=10` and the linkage/affinity literals
(`benchmarking_utils.py:158,181,188`).

## SNR calibration

`benchmarking_config.py:16` points at a `Calibrate_Settings.ipynb` that does not exist in the
repo. Only its output survives, hardcoded as anchor dicts in notebook cells, which makes the
procedure described in S3 Text unreproducible. Reviewer 3 asked for exactly this kind of gap to
be closed.

`_calibrate.py` reimplements the documented procedure: search the SNR parameters of a scenario so
that a base estimator informed with the true `k_star` reaches a target mean ARI over the
calibration seeds, with bands `[0.9, 1.0]` for easy, `[0.8, 0.9]` for medium, and `[0.7, 0.8]`
for hard (`CARVE_manuscript.tex:854`). Calibration uses the same seeds as the benchmark run, as
the manuscript states.

The original search space is not recoverable, and the spec does not pretend otherwise. The
published anchors vary all four parameters at once and do so non-monotonically across difficulty
levels - for moons, `embed_param` runs 10.7, 5.7, 14.5 from easy to hard - which is not the
signature of a single ordered sweep. The rebuilt module therefore defines its own search space
rather than reconstructing the lost one: each scenario declares which kwargs are calibrated and
over what range, the search is documented in the module docstring, and S3 Text is updated to
describe what the code now does. This is a deliberate substitution, not a reproduction.

The module is run offline, not as part of a benchmark run. Its output is written into
`_registry.py` as the active anchors.

`_registry.py` also retains the currently published anchor dicts verbatim as a separate constant,
so a regenerated calibration can be compared against them and reverted to if it moves the results
for reasons unrelated to the three bug fixes. Both sets are committed; the active one is named
explicitly so there is never ambiguity about which produced a given artifact, and the manifest
records which was used.

## Artifact contract

One long-format table per run, one schema for every experiment:

```
run_id, scenario, axis_name, axis_value, axis_label, seed, k_star,
estimator, metric_name, k, metric_value, is_selected, selects_true_k,
ari_at_k, oracle_ari
```

Renames, chosen so no column means two things: `difficulty`/`stage` become `axis_label`;
`is_optimal` becomes `is_selected`; `is_correct` becomes `selects_true_k`; `metric_ari` becomes
`ari_at_k`; `baseline_ari` becomes `oracle_ari`; `dataset_iteration` becomes `seed`. The resample
count is `n_resamples` everywhere and never `B`, retiring the collision with the manuscript's
B=20 datasets.

A sidecar `manifest.json` records the resolved config, seed derivation, which anchor set was
used, package versions, git SHA, CPU model, OS, core count, wall-clock, and peak RSS. This closes
R1.1's memory request and the missing compute-environment gap in one step; today `n_jobs=10` is
recoverable only from a column in one runtime CSV.

Peak RSS comes from `resource.getrusage` over both `RUSAGE_SELF` and `RUSAGE_CHILDREN`, so joblib
workers are counted. No new dependency is needed. The units differ by platform: `ru_maxrss` is
bytes on macOS and kilobytes on Linux. Normalize to bytes at the call site and record the unit in
the manifest, because a benchmark whose headline claim is memory cannot afford a factor-of-1024
ambiguity between the machine the author ran it on and the machine CI ran it on.

### Storage and promotion

Working runs write parquet plus `manifest.json` to `results/runs/<scenario>/<config-hash>/`,
which is gitignored. Content addressing means changing an anchor produces a new directory instead
of a silent stale read. The current scheme is `if not os.path.exists(csv)` against a committed
CSV, which means the nightly workflow has never actually run a benchmark - it always takes the
cache branch and exercises only plotting.

`_artifacts.promote()` copies a chosen run to `results/published/<scenario>/` as CSV plus its
`manifest.json`, and those are committed. This keeps the repo small and diffable, lets a reviewer
read the numbers without pyarrow, and keeps the provenance that answers R1.1 attached to the
numbers it describes. Promotion is deliberate and explicit; a run is never promoted as a side
effect of executing.

Per-seed checkpointing makes runs resumable. Today a crash at seed 55 of 60 loses everything.

Parquet for working artifacts requires pyarrow, which is not currently a dependency. It goes in a
new `benchmarks` optional-dependency group rather than into the `notebooks` extra.

## The runner

`_run.py` has one public entry point. Structure:

```
for axis_value, axis_label in scenario.axis:
    for seed in range(scenario.n_seeds):     # joblib parallelizes HERE
        simulate -> oracle fit at k_star -> CARVE fit -> CARVE metrics at every k
                 -> CVI metrics at every k -> rows
```

Parallelism inverts relative to today. Currently joblib parallelizes 5-element inner loops with
processes while the 60-iteration outer loop runs serially, and the outer `n_jobs` and
`CARVE(n_jobs=...)` are given the same value, so they nest. The rebuild parallelizes the outer
(axis x seed) loop and passes `n_jobs=1` to CARVE, which removes the oversubscription that the CI
currently works around with `OMP_NUM_THREADS=1`.

Reusable pieces to keep: the seed derivation `seed + axis_idx * 10000 + random_state`
(`benchmarking_runners.py:270`), `_build_estimator`'s `inspect.signature` probe for
`random_state` (`benchmarking_utils.py:120`), and `get_rule`/`get_measure`
(`benchmarking_utils.py:77,97`), though those two become fields on a metric record rather than
string-suffix parsing.

The runner does not import matplotlib. Today `benchmarking_runners.py:50` imports the plotting
module and calls `display_benchmark_snapshot` unconditionally on all 60 iterations, each call
rebuilding the whole growing results list into a DataFrame (quadratic over the run), running a
PCA, and building two figures. `get_snapshot` is dropped entirely: it re-simulates at a different
seed than the frame it is drawn beside, so its two panels are inconsistent by construction.

## Tables

`_tables.py` replaces `summarize_regime_tables` and `summarize_scaling_tables`, which share 105
of 122 code lines. With `axis_label` unified they are one function whose only difference was the
column header. Keep `_summarize_single_group`'s statistics (mean/sd/median/IQR ARI,
delta-to-oracle, k-recovery with Wilson CI, k-bias, over/under rates) and `_render_grouped_tex`'s
output, which matches the manuscript's table format. Drop the dead second LaTeX path
(`render_tex_table`, which routes through `df.to_latex` and has no callers).

Tables are written to disk as `.tex` fragments under their manuscript names rather than printed
for copy-paste, so the S2-S10 tables stop drifting from what the code produces. The committed
manuscript tables currently have the generated `$k^*=5$` header row stripped by hand.

## Figures and theme

`_theme.py` is the single source for palette, rcParams, spine treatment, legend kwargs, and
figure geometry. Today there are three uncoordinated module-level palettes plus a fourth in the
config that only a legacy notebook reads, so CARVE-stability is `#00CD6C` in Fig 4 and `#009E73`
in Fig 5 - the same method in two greens in one paper. `set_paper_style()` exists in
`benchmarking_summary_functions.py:23` and is called by nothing, so every published figure uses
stock matplotlib defaults and `pdf.fonttype: 42` was never applied. Also replaces 24 inline
`fontsize=` literals, the spine block retyped with three different alphas, and the notebook's
`ELEMENT_SCALE = 0.47` fudge that exists to make the scaling figure visually match Fig 4.

`_panels.py` holds ax-in/ax-out primitives; there are currently five different return contracts
across thirteen plotting functions, and three of them (`plot_examples`, `plot_dim_red`,
`plot_alluvial`) discard the figure and call `plt.show()` inside library code.

`figures/` is a subpackage with one module per manuscript figure, each exposing a single
`figure_<manuscript_name>()` that returns a `Figure` and saves under the manuscript filename at
300 dpi. A single `_figures.py` would exceed a thousand lines once the two roughly 250-line
composite figures move in from the notebooks, and each follow-on case study adds another. One
module per figure keeps each unit readable and makes adding a study a new file rather than an
edit to a shared one.

This is the largest single reproducibility win: four of nine paper figures
(`CARVE_output_klein`, `klein_results`, `CARVE_output_levine`, `levine_results`) have no
`savefig` at all and were right-click-saved from Jupyter output, and the rest were manually
downscaled to one-third and renamed en route to `overleaf/vis/`. Figures are written to
`code/vis/`; copying into the manuscript stays manual, per the existing convention.

Replace `cm.get_cmap` (`benchmarking_plotting.py:52`), which is deprecated and scheduled for
removal in matplotlib 3.11.

## Datasets

`datasets/` provides `load_klein()`, `load_levine32()`, and `load_motivation_*()`, each returning
`(X, y, meta)` with provenance. This removes roughly 300 lines of loader code duplicated across
three notebooks: the Klein loader appears in Klein.ipynb and Motivation.ipynb, the Levine loader
in Levine_32dim.ipynb and Motivation.ipynb, and `get_n_jobs()` and the seed block appear four
times each.

Two fixes go in here. `data/klein` versus the on-disk `data/Klein` is a case-only mismatch that
works on macOS and breaks on Linux. The Levine loader currently runs `BiocManager::install`
through rpy2 as a side effect of executing the notebook and caches nothing locally; it gains a
disk cache and an explicit opt-in for the install step.

Preserve the documented preprocessing exactly, including the ordering quirk where `RobustScaler`
is fit before population 15 is dropped. Record it in the docstring so the next reader does not
"correct" it.

## Notebooks

Notebooks become thin drivers: load artifacts, call `figure_*`, display. `sys.path` hacks go away
because `benchmarks` is importable from an editable install. `Benchmarking.ipynb` keeps its
narrative structure; its per-scenario `anchor_settings` dicts move into `_registry.py`, which is
where the study design belongs - it currently exists only in notebook cells.

The 120-line raw-matplotlib scaling cell and the two roughly 250-line composite-figure cells in
Klein.ipynb and Levine_32dim.ipynb move into `figures/`. The Klein and Levine cells are
near-identical, differing only in marker size and panel F.

## CI and testing

- `ci.yml` needs no change to its ruff steps: `ruff check src/` and `ruff format --check src/`
  already cover `src/benchmarks/` by virtue of the layout decision. Confirm this rather than
  assume it, and keep the pinned ruff 0.16.4. Expect the first run to flag a large number of
  findings in ported code.
- `pyproject.toml`: add `"benchmarks", "benchmarks.*"` to
  `[tool.setuptools.packages.find] exclude`, and add the `benchmarks` optional-dependency group
  with pyarrow.
- `.gitignore`: add `results/runs/`. It currently ignores only `results/backup*/` and
  `results/legacy/`, so without this the content-addressed working tree would be committed by
  accident on the first run.
- Tests live at `tests/benchmarks/`, mirroring modules 1:1 as the existing convention requires.
  The existing `testpaths = ["tests"]` picks them up with no change.
- Coverage stays scoped to `carve` (`[tool.coverage.run] source = ["carve"]`), so the existing
  `--cov-fail-under=75` gate is unaffected by the new package. Revisit a separate floor for
  `benchmarks` once it has settled.
- Update the CLAUDE.md lines that say ruff runs on `src/` only and that describe
  `notebooks/benchmarking_code/`. The file lives at `_claude_playground/CLAUDE.md`, outside the
  repo.
- `nightly-notebooks.yml`: add a genuine smoke run - one scenario at reduced `n_seeds` and
  `n_resamples` into a temp directory - so the workflow exercises compute rather than always
  hitting the cache.

Tests target the parts that currently have none: the runner, the output schema (one
`is_selected` True per metric/axis/seed group, row counts matching
`n_axis x n_seeds x n_metrics x n_k`), determinism at a fixed seed, `Scenario` validation
rejecting bad and duplicated keys, `EstimatorSpec` raising on unknown names, the gap statistic
against a known case, and `promote()` round-tripping a run to CSV without schema loss. Reuse the
existing coverage of `make_scaling_x_values`, `_wilson_ci`, `get_rule`/`get_measure`, and
`align_labels` from `test_benchmarking.py`; drop the tests for the six dead functions.

## Bug fixes and blast radius

| Fix | Current state | Affects |
|---|---|---|
| Swiss-roll estimator | `regimes_full` passes `"spectral"`; the benchmark ran `"agglomerative"`; S3 Text says Ward | S1 Fig panel; Table S7 unaffected (results were correct, only the illustration was wrong) |
| Gap statistic | Plain `argmax(Gap(k))`, no `s_k`; not Tibshirani's rule, while CARVE gets a 1SE rule. Fix to the published rule: smallest k with `Gap(k) >= Gap(k+1) - s_{k+1}`, where `s_k` is the reference-dispersion standard deviation inflated by `sqrt(1 + 1/n_reference_datasets)` | Gap column in every table; Fig 4; S2-S9 |
| `ari_at_k` fit mode | Difficulty runner computes it from stability-mode labels for every metric because `mode` defaults to `"default"`; scaling runner uses the matching mode | `ari_at_k` across all files; the two files' columns currently disagree in meaning |

Re-run affected scenarios and record the corrections for the response letter. Fixing the gap
statistic may narrow CARVE's reported margin; that is the point, given R2.1 challenges the choice
of competitors.

## Deletions

- `legacy/` - 1,480 lines, zero importers, two files whose relative imports no longer resolve.
  Extract two design ideas first: the interpolated continuous-difficulty scheme
  (`legacy/benchmarking_simulation_helpers copy.py:30-101`) and the multi-`k*` sweep
  (`legacy/benchmarking_runners copy.py:30`), both capabilities the current design lost.
- The moons scaling experiment. `results/scalability_results/results_moons_dimensionality.csv`
  and its runtime file are removed, the commented-out notebook cell goes, and no moons entry is
  added to a scaling axis in `_registry.py`. Moons remains a difficulty scenario; only its
  scaling axis is dropped.
- Six of eight functions in `benchmarking_summary_functions.py`; two of them
  (`summarize_true_k_accuracy`, `best_metric_per_benchmark`) are written against a schema the
  runners never emitted and would raise `KeyError` on any real frame.
- `plot_ari_overview_grid`, `plot_benchmark_snapshot`, `summarize_benchmark_regime`,
  `render_tex_table`, `_metric_color_map`, `_hex_to_rgba` - no callers.
- The unused `plotly` import and the `estimator_grids` parameter, which is documented as an
  override and then unconditionally overwritten at `benchmarking_runners.py:283`.
- The `band` / `show_band_for` parameters threaded through three functions and ignored by all.
- `notebooks/benchmarking_code/` in its entirety, once the port is verified.

## Verification

1. Unit tests pass; `ruff check` and `ruff format --check` clean on `src/`.
2. Regression: re-run each unaffected scenario at its recorded seeds and assert the rebuilt
   pipeline reproduces the existing committed CSVs (`results/results_<scenario>.csv`, old schema)
   within numerical tolerance. This is the primary evidence that the rewrite preserved the logic.
   The comparison needs an explicit old-to-new column mapping, since the rebuild renames six
   columns; that mapping lives in the regression test, not in the library, and is deleted with
   the old files once the port is verified. The old CSVs stay in place until then - they are the
   oracle, so they cannot be removed in the same step that would be checked against them.
3. Affected scenarios: re-run, diff against the old numbers, and record every change with its
   cause.
4. Calibration: run `_calibrate.py` and compare the regenerated anchors against the retained
   published set. Report the difference; do not silently adopt either.
5. Regenerate all nine manuscript figures from code and compare against the current
   `overleaf/vis/` images by eye. The four screenshot-derived figures will differ in resolution
   and styling by design; content must match.
6. Regenerate the S2-S10 tables and diff against the manuscript.
7. Run `Benchmarking.ipynb`, `Klein.ipynb`, `Levine_32dim.ipynb`, and `Motivation.ipynb` end to
   end from a clean kernel with no `sys.path` manipulation.
8. Confirm `manifest.json` captures enough to answer R1.1 without rerunning anything.
9. Confirm `import benchmarks` works from a fresh editable install and that `python -m build`
   produces a wheel with no `benchmarks/` in it.

## Out of scope

Follow-on plans, each of which should be a small uniform addition to the finished framework:
large scRNA-seq case study (R1.1), scATAC/multimodal case study (R1.7), preprocessing ablation
(R1.6), rho and n_resamples sensitivity (R4.4), SC3/M3C comparison (R2.1), Leiden/Louvain
comparison on case studies (R1.4), resampling-bias investigation (R1.3). The six package changes
and the manuscript changes in `comments.tex` are also out of scope here.

R1.6 and R4.4 need no new runner under this design - they are `Axis("n_pcs", ...)` and
`Axis("rho", ...)`. R1.7 may require changes to `carve` itself if the data is not tabular; that
is a package plan, not a benchmarks plan.
