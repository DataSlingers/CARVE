# Publication re-run handoff

Date: 2026-09-23

Branch `benchmark-uniformity` is code-complete against the `2026-09-21
benchmarking-uniformity-and-notebook-design` spec. This note is what the
author runs from next: the remaining dev-scale acceptance check, the staged
validation that separates a runner defect from the intended redesign, the
publication commands with their costs, what to do once the run finishes, and
the manuscript edits it forces.

## What changed since the artifacts on disk were produced

Every one of these moves numbers, so nothing under `results/runs/` or
`results/results_*.csv` from before this branch is comparable to what a
re-run produces:

- The spectral eigensolver in `carve/cluster.py` moved from an unseeded
  `which="SM"` ARPACK call to shift-invert (`sigma=0.0, which="LM"`,
  09b810c). The old solver was nondeterministic and did not reliably
  converge; circles and moons in particular were measuring solver failure
  rate, not clustering difficulty.
- `swiss_rolls` moved from Ward to spectral clustering, as a consequence of
  the eigensolver fix rather than an independent choice, joining circles and
  moons as one self-tuning-spectral family.
- `n_trees` is 500 for every scenario. Previously it was 500 on circles,
  moons and swiss_rolls and 100 elsewhere, an artifact of which notebook
  cell happened to pass the argument.
- The difficulty anchors are calibrated (`ACTIVE_ANCHOR_SET_NAME` is now
  `"CALIBRATED_ANCHORS"`, not `"PUBLISHED_ANCHORS"`). All eighteen anchors
  land inside their documented ARI bands under the fixed solver; see
  `docs/superpowers/notes/2026-09-22-calibration-comparison.md` for the full
  comparison. `PUBLISHED_ANCHORS` stays in the registry for reference and
  revert.

Runner and artifact behavior also changed in ways that affect how a re-run
is operated, though they do not move any number themselves: `manifest.json`
is now written with a `status` field before the pool starts, not only after
it finishes; resuming into a run directory recorded at a different
`git_sha` is refused unless `--allow-code-change` is passed; there is one
shared `widest_run` resolver for both `--tables` and the notebook, which
selects the widest `(n_seeds, n_resamples)` sweep and raises on a tie rather
than picking arbitrarily; and `n_jobs` now resolves per scenario class
inside the runner (`-1` for the six difficulty scenarios, `1` for the two
scaling scenarios, since their timings are only meaningful single-worker),
with an explicit `--n-jobs` still able to override it.

The notebook and manuscript tooling changed too: `notebooks/Benchmarking.ipynb`
now has one dashboard per scenario section (four rows: example scatters, ARI
over the axis, normalized criterion-over-k, and a P(k-hat=k) heatmap) instead
of the old overview-plus-table pair, and the S2-S9 `.tex` fragments are now
the manuscript's caption-free, side-by-side table float rather than the old
shape. The palette was re-stepped at the hues that failed the dataviz
validator (CARVE Generalizability, Davies-Bouldin, Calinski-Harabasz, Gap
Statistic). The failing values and the validator report for the new set are
in commit 26da3a3's message; in `_theme.py`, the comment above the four
classical indices in `METRIC_COLORS` says why each moved, and the comment
above `PIPELINE_COLORS` records Generalizability's move from `#56B4E9`.

## Current state of results/runs/

`results/runs/` is gitignored. The old 20-dataset/5-dataset mixed tree was
moved by hand to `archive/20260923-162048/runs/`, together with the old
`vis/benchmarking/` figures (`archive/20260923-162048/benchmarking/`) and
`vis/tables/` fragments (`archive/20260923-162048/tables/`). The move is not
tracked: `archive/` is gitignored, and 0b58047 only adds that entry to
`.gitignore`. `results/runs/ablation_rho_b` was left in place, untouched by
this branch.

The dev-scale acceptance run (`--n-seeds 3 --n-resamples 10`) was stopped
partway through at the author's request. As of this branch's tip:

- Complete, at 3 datasets x B=10, on this branch's code: `circles`,
  `gaussians`, `gaussians_dimensionality`.
- `gaussians_samples`: manifest status is `"running"` (interrupted). `widest_run`
  skips it, since it requires a complete manifest by default. The directory
  can be left in place or removed before the real dev run; either way,
  `--no-resume` recomputes from scratch and ignores whatever is there.
- Not run at all yet: `t_dist`, `t_dist_noise`, `moons`, `swiss_rolls`.

## Re-deriving the cost from what actually ran

The spec's E4 figures were projected from the pre-branch manifests, before
the eigensolver fix, the `n_trees` unification, swiss_rolls moving to
spectral, or the calibrated anchors. The three completed dev-scale runs give
real per-cell timings on this branch's code instead. The script below skips
any scenario with no complete run, and treats `gaussians_dimensionality`
differently from the other two: it is a `TIMED_SCENARIOS` member, so each of
its cells also runs two extra CARVE fits purely to time them
(`t_stability_s`, `t_generalizability_s`), more than doubling a cell's cost
over `t_default_s` alone. `wall_clock_s`, at this scenario class's `n_jobs=1`,
is the serial sum of all three timing columns plus fixed per-run overhead,
so it is used directly for that row instead of `t_default_s`:

```python
import json
from pathlib import Path

from benchmarks._artifacts import read_runtimes, widest_run
from benchmarks._registry import SCENARIOS, TIMED_SCENARIOS

ROOT = Path("results/runs")
print(
    f"{'scenario':26s} {'s/cell':>8s} {'cells':>6s} {'dev CPU-h':>10s} "
    f"{'20x100 CPU-h':>13s}  basis"
)
total = 0.0
skipped = []
for name in SCENARIOS:
    try:
        rd = widest_run(ROOT, name)
    except FileNotFoundError:
        skipped.append(name)
        continue
    manifest = json.loads((rd / "manifest.json").read_text())
    runtimes = read_runtimes(rd)
    cells = len(runtimes)
    if name in TIMED_SCENARIOS:
        dev_h = manifest["wall_clock_s"] / 3600
        basis = "wall_clock_s (n_jobs=1)"
    else:
        dev_h = float(runtimes["t_default_s"].sum()) / 3600
        basis = "sum(t_default_s)"
    per_cell_s = dev_h * 3600 / cells
    # B and the dataset count both scale the cost linearly.
    scale = (20 / manifest["n_seeds"]) * (100 / manifest["n_resamples"])
    full_h = dev_h * scale
    total += full_h
    print(
        f"{name:26s} {per_cell_s:8.2f} {cells:6d} {dev_h:10.3f} "
        f"{full_h:13.1f}  {basis}"
    )
print(f"{'TOTAL (measured scenarios only)':26s} {'':8s} {'':6s} {'':10s} {total:13.1f}")
if skipped:
    print(f"skipped (no complete run): {', '.join(sorted(skipped))}")
```

Output:

```
scenario                     s/cell  cells  dev CPU-h  20x100 CPU-h  basis
gaussians                     87.17      9      0.218          14.5  sum(t_default_s)
circles                       87.95      9      0.220          14.7  sum(t_default_s)
gaussians_dimensionality      87.34      9      0.218          14.6  wall_clock_s (n_jobs=1)
TOTAL (measured scenarios only)                                     43.7
skipped (no complete run): gaussians_samples, moons, swiss_rolls, t_dist, t_dist_noise
```

The `gaussians_dimensionality` row is not comparable with the other two,
and its 87.3 s/cell landing next to their 87.2 s and 88.0 s is a
coincidence of two different measurements. It is serial wall clock: one
cell at a time, with each cell's fits free to use every core (the
classifier runs with `n_jobs=-1`), so its "CPU-h" columns are wall-clock
hours. The difficulty rows sum `t_default_s` over cells that ran nine at a
time on eleven cores, each competing with the others for them. What the row
does show is that the earlier `t_default_s`-only figure for it (39.3 s/cell,
6.5 h at 20x100) left out the two timed fits.

For `gaussians` and `circles`, which are not `TIMED_SCENARIOS` members and
run no extra timed fits, `t_default_s` alone is the whole per-cell cost.
Checked against parallelism: summed over their 9 cells, `t_default_s` comes
to 784.6 s (gaussians) and 791.5 s (circles), against manifest `wall_clock_s`
of 94.3 s and 114.5 s at `n_jobs=-1` — an 8.3x and 6.9x speedup across 9
concurrently scheduled cells on an 11-core machine, short of 9x because the
classifier's own internal `n_jobs=-1` threading inside each cell
oversubscribes the cores. That gap is expected, not a discrepancy to chase.

The linear scaling in `B` and in the dataset count is an approximation: the
classifier's cost per resample is roughly constant, but consensus matrix
assembly is not linear in `B`, so these are projections, not measurements,
even for the three rows with a real dev-scale number behind them.

For the five scenarios with no complete dev run, the spec's E4 projections
are what is available: roughly 75 CPU-h total for the six difficulty
scenarios (about seven hours wall at eleven workers), and roughly 23 and 34
CPU-h for the two scaling scenarios. These are worth treating with more
caution than the three measured rows above. The measured `gaussians` and
`circles` numbers (14.5 and 14.7 CPU-h) alone already account for close to
40% of the 75 CPU-h difficulty-scenario total, leaving `t_dist`,
`t_dist_noise`, `moons` and `swiss_rolls` to share roughly 46 CPU-h between
them, unsplit by the spec. The corrected `gaussians_dimensionality` number
(14.6 CPU-h) sits below the spec's 23-34 CPU-h range for the scaling pair,
but only by a third to a half, not by the factor of three to five the
uncorrected `t_default_s`-only figure implied. Do not treat 23 and 34 as
calibrated per-scenario figures; `gaussians_samples` sweeps `n_total` up to
10,000, well past this scenario's fixed 1,500, and is itself also a
`TIMED_SCENARIOS` member running the same two extra timed fits per cell, so
its measured cost is plausibly higher than `gaussians_dimensionality`'s, not
lower.

The author's own dev run (below) replaces every projected row in this note
once it completes. Nothing here should be used for capacity planning beyond
"budget the high end and confirm before committing machine time."

## The dev-scale acceptance run still to do

This is the run the previous session stopped partway through. Before moving
to the staged validation or the publication run:

1. Finish the dev-scale acceptance run, all eight scenarios, discarding the
   interrupted `gaussians_samples` attempt:

   ```
   python -m benchmarks.run --all --n-seeds 3 --n-resamples 10 --no-resume --verbose
   ```

   `--verbose` is the correct flag; `-v` is rejected by the parser.
   `--no-resume` means the interrupted `gaussians_samples` directory does not
   need to be removed first, but it can be, since nothing about it is read.

2. Check every manifest: all eight `status: "complete"`, `n_seeds: 3`,
   `n_resamples: 10`, `anchor_set: "CALIBRATED_ANCHORS"`, and `n_jobs` of -1
   on the six difficulty scenarios and 1 on `gaussians_dimensionality` and
   `gaussians_samples`.

3. Regenerate every figure and table by executing the notebook, with the
   executed copy written to a scratch directory rather than over the
   original, under warnings-as-errors. Warnings-as-errors goes into the
   kernel through an IPython startup file:

   ```
   mkdir -p <ipython dir>/profile_default/startup
   printf 'import warnings\nwarnings.simplefilter("error")\n' \
     > <ipython dir>/profile_default/startup/00-warnings-as-errors.py
   IPYTHONDIR=<ipython dir> .venv/bin/jupyter nbconvert --to notebook --execute \
     --ExecutePreprocessor.timeout=1800 --output-dir=<scratch dir> \
     notebooks/Benchmarking.ipynb
   ```

   `<ipython dir>` is any fresh directory, which also keeps your own
   IPython profile out of the run. Do not use `PYTHONWARNINGS=error` instead:
   it applies from interpreter startup, and tornado raises a
   `DeprecationWarning` ("There is no current event loop") while the kernel
   initializes, so the kernel dies before replying to `kernel_info` and no
   cell runs. The startup file runs after the kernel is up and before the
   first cell. Checked on a two-cell scratch notebook: the kernel starts, the
   clean cell runs, and the cell that calls `warnings.warn` fails with
   `CellExecutionError`; with an empty `<ipython dir>` both cells pass.

   The notebook's own second cell sets
   `PYTHONWARNINGS=ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning,ignore::RuntimeWarning`
   via `%env`, for the readability of its own output. That assignment only
   reaches subprocesses started after that cell runs; it does not change the
   warnings filters of the kernel process, so the startup file's filter
   stays in force for the figure- and table-generation code, which runs
   in-process. The gate is that nothing raises, no panel is empty, and every
   generated `.tex` fragment compiles.

4. Look at the `circles` and `gaussians_samples` dashboards and
   `vis/benchmarking/benchmarking_results.png`. These are the two places
   most likely to show a problem: circles is one of the two scenarios whose
   difficulty axis was never verified to separate under the fixed solver
   (see Open item below), and gaussians_samples is a scaling scenario that
   has not run to completion even once on this code yet.

   Only the executed `.ipynb` goes to the scratch directory. The figures go
   to the repository's `vis/benchmarking/` (`VIS_ROOT` in
   `benchmarks/figures/_paths.py` is anchored at the repository root, not
   the working directory) and the S2-S9 fragments to the repository's
   `vis/tables/` (the notebook writes them to `../vis/tables` from
   `notebooks/`), overwriting whatever is there; both trees are gitignored.
   `benchmarking_results.png` is among them, since
   `figure_benchmarking_results` defaults to `save=True`. The two
   scenario dashboards are not: the notebook's `scenario_dashboard(name)`
   wrapper calls `figure_scenario_dashboard(name, frame)` without
   `save=True`, which defaults to `save=False`, so the executed notebook
   only shows them inline in its own output — open the executed `.ipynb` in
   the scratch directory to see them there. To also get them as standalone
   PNG files, call the figure function directly with `save=True`:

   ```python
   from pathlib import Path

   from benchmarks._artifacts import read_run, widest_run
   from benchmarks.figures import figure_scenario_dashboard

   for name in ("circles", "gaussians_samples"):
       frame = read_run(widest_run(Path("results/runs"), name))
       figure_scenario_dashboard(name, frame, save=True)
   ```

   which writes `vis/benchmarking/scenario_dashboard_circles.png` and
   `..._gaussians_samples.png`.

## Staged validation against PUBLISHED_ANCHORS

The committed `results_*.csv` files are the only oracle for the runner's own
correctness, and the calibrated anchors move the numbers out from under
them, so the comparison has to be staged rather than direct. `tests/benchmarks/test_regression.py`
already does this: it pins `PUBLISHED_ANCHORS` and `n_trees=100` (the
published run's own forest size for these three scenarios) via
`dataclasses.replace`, independent of whatever `ACTIVE_ANCHORS` or `_N_TREES`
say in the live registry, and runs `gaussians`, `t_dist` and `t_dist_noise` —
the three scenarios using k-means and Ward, which the eigensolver fix cannot
touch and which A1 (the `n_trees` unification) is the only registry change
that reaches. No registry edit is needed; the staged check is one pytest
invocation:

```
CARVE_RUN_REGRESSION=1 .venv/bin/python -m pytest tests/benchmarks/test_regression.py -v
```

The test pins every input the published run used, `n_trees=100` included,
so a disagreement with the committed CSVs is a defect in the runner or
artifact layer, not a consequence of the redesign; passing here is what
licenses switching to `CALIBRATED_ANCHORS` for the real six-scenario run.
`swiss_rolls` is not part of this check: its committed CSV was produced under Ward, and it
is no longer a comparable oracle now that the scenario runs spectral.

Cost: this is a full 3-anchor x 20-dataset x B=100 run of `gaussians`,
`t_dist` and `t_dist_noise`, at `n_trees=100`. The spec's E4 puts the whole
three-scenario pass at about 7 CPU-h, scaled from the old manifests at that
forest size. The measured `gaussians` dev-scale number above, scaled the
same way, comes to 14.5 CPU-h for `gaussians` alone — well above what a
one-third share of a 7 CPU-h total would suggest. Two factors likely explain
part of the gap, neither of them cleanly separable from the dev-scale
numbers alone: `n_trees` moved from 100 (this staged run's forest size) to
500 in the measured figure, and `t_default_s` bundles fixed per-cell costs —
simulating the dataset, the classical indices, the gap statistic's reference
sets, the oracle fit — that do not scale with `B` at all, so extrapolating
linearly from this dev run's B=10 to the staged run's B=100 overstates
whatever part of the cost actually is B-dependent. Both push in the
direction of the measured number overstating the staged run's true cost, but
not by how much. Treat 7 CPU-h as the spec's own estimate and 14.5 CPU-h as
an upper bound that is not directly comparable to it, rather than reconciling
the two by assumption; the evidence that settles this is the author's own
dev run (which already ran `gaussians` at this branch's actual `n_trees=500`
and `CALIBRATED_ANCHORS`) and the actual wall clock of the first of these
three staged-validation scenarios once it runs.

## Publication run

Six commands, one per difficulty scenario, `--no-resume` so nothing from a
prior partial run is reused. `--n-jobs` is deliberately absent on all eight
commands in this section: the runner resolves it per scenario class, and
passing it by hand on the two scaling scenarios is exactly how their timings
get corrupted.

```
python -m benchmarks.run --scenario gaussians --no-resume       # measured: ~14.5 CPU-h
python -m benchmarks.run --scenario t_dist --no-resume          # projected (spec E4)
python -m benchmarks.run --scenario t_dist_noise --no-resume    # projected (spec E4)
python -m benchmarks.run --scenario circles --no-resume         # measured: ~14.7 CPU-h
python -m benchmarks.run --scenario moons --no-resume           # projected (spec E4); same family as circles
python -m benchmarks.run --scenario swiss_rolls --no-resume     # projected (spec E4): "costs like circles and moons" now that it is spectral
```

Combined, the spec's E4 puts these six at roughly 75 CPU-h, about seven
hours wall at eleven workers. Two of the six now have a measured number
(`gaussians` and `circles`, 14.5 and 14.7 CPU-h); the other four are
unmeasured and share the remaining ~46 CPU-h in the spec's estimate, unsplit.

The two scaling scenarios are costed and run separately from the six above,
one worker each (the runner forces this; not an operator choice), and must
not share the machine with the rho/B ablation's own publication run — both
are long, both want the machine's cores, and running them together corrupts
whichever one is trying to measure wall-clock time.

```
python -m benchmarks.run --scenario gaussians_dimensionality --no-resume   # measured: ~14.6 CPU-h (wall_clock_s; the two extra timed fits dominate)
python -m benchmarks.run --scenario gaussians_samples --no-resume          # projected (spec E4, combined with the above at 23-34 CPU-h for the pair)
```

Both run serially at one worker. `gaussians_dimensionality`'s 14.6 h is
already wall clock (see the cost section), and sits within a third to a
half of the spec's 23-34 CPU-h range for the pair, rather than well under
it, so there is no basis here for shortening "two to three days" (the
spec's wall-clock estimate for the pair). If anything,
`gaussians_samples` is likely to cost at least as much per cell as
`gaussians_dimensionality`, plausibly more: it is also a `TIMED_SCENARIOS`
member running the same two extra timed fits, and it sweeps `n_total` up to
10,000, well past the difficulty scenarios' and `gaussians_dimensionality`'s
fixed 1,500. It has not been measured at all yet; keep the full scheduled
window until it has.

Every command above passes `--no-resume`, which recomputes every cell, so
an interrupted multi-day run restarted with the same command starts from
zero. To resume one instead, rerun it without `--no-resume` at the same
HEAD: the checkpoints already written are kept and only the missing cells
run, under the original `run_id`. If HEAD has moved and the change provably
cannot move a recorded value (a docstring, a figure, a note), add
`--allow-code-change`; if it can, start over with `--no-resume`. The guard
compares HEAD only and does not see uncommitted edits, so leave the working
tree alone while a run is in progress.

## After the run finishes

Re-run `notebooks/Benchmarking.ipynb` (the nbconvert-into-scratch command
above, now against the full-scale artifacts) to regenerate the four
manuscript figures and the eight S2-S9 fragments. Then, by hand — this stays
a manual step, since `../overleaf/` is read-only and there is no build
connecting the two trees:

- Copy the four figures (`benchmarking_examples.png`, `benchmarking_results.png`,
  and the two scaling figures) and the eight `S2_table.tex`-through-`S9_table.tex`
  fragments from `vis/benchmarking/` and `vis/tables/` into `overleaf/vis/`.
- The S2-S9 fragments themselves can be dropped in without hand-editing —
  D1's renderer already reproduces the committed shape. The `\paragraph*`
  caption text immediately above each fragment is the author's own prose and
  is untouched by this branch, but check it: the old captions describe a
  "top" and "bottom" sub-table, which was true of the old stacked layout.
  D1's renderer puts the two tabulars side by side inside one
  `\resizebox`, so "top/bottom" is now stale wording that should read
  something like "left" and "right," independent of anything the numbers
  themselves changed.
- Row order. S2-S7 are generated in the committed S2's order and S8-S9 in
  the committed S8/S9 order (`TABLE_ROW_GROUPS` and
  `SCALING_TABLE_ROW_GROUPS` in `_registry.py`). The committed S3-S7 do not
  share S2's order within the classical-index and remaining-CARVE groups,
  and S4 also lists CARVE Generalizability (1SE) before CARVE Stability
  (1SE), so dropping those five in reorders rows within their groups.
  Nothing needs editing for them to compile.

The manuscript edits the re-run forces, from the spec's "Consequences
outside this repository" section:

- Circles and moons are the load-bearing edit. Their published anchors sat
  in the region where the broken eigensolver failed to converge at different
  rates by anchor — that pattern, not increasing difficulty, produced the
  published gradient (0.875/0.822/0.781 on circles, similarly on moons). The
  sentence "On circles and moons, CARVE's metrics outperformed the CVIs
  consistently, while most CVIs struggled across all SNR levels" cannot be
  carried over as-is; it has to be rewritten against the re-run's numbers,
  whatever those turn out to show.
- Swiss rolls now run spectral clustering, and three places still reflect
  Ward. The estimator sentence in the synthetic benchmarking section
  (`CARVE_manuscript.tex:584`, "Ward agglomerative clustering ... for
  $t$-mixtures and swiss rolls") and S3 Text's shapes 4-6 paragraph (`:902`,
  "Ward agglomerative clustering for Swiss rolls") need to say spectral. The
  findings sentence at `:587` ("For swiss rolls, which are roughly
  spherical, CARVE's metrics provided advantages at easy and hard SNR
  settings while remaining competitive at medium settings") describes
  Ward's results and has to be rewritten against the re-run's numbers.
- `CARVE_manuscript.tex:600` attributes the generalizability collapse on
  `gaussians_dimensionality` to "the default 100-tree random forest" and
  recommends more trees as $p$ grows; `:1451` says the same ("a random
  forest with 100 trees") and makes the same recommendation. The benchmark
  now runs 500 trees, so both the finding and the recommendation have to be
  re-checked against the re-run. The dev-scale run (3 datasets, B=10, 500
  trees) already shows the collapse persisting: CARVE Generalizability
  (1SE) mean ARI 0.676, 0.351 and 0.147 at p = 50, 525 and 1000, against
  the oracle's 0.852, 0.725 and 0.872. Three datasets at B=10 settle
  nothing, but if the publication run agrees, forest size is not the cause
  the text gives.
- S3 Text's anchor values change (all eighteen, since every scenario was
  recalibrated, not only the three the eigensolver fix touched), and the
  calibration procedure it describes should name both calibration knobs
  (`cluster_scale` multiplier for the four scale-driven scenarios,
  `embed_param` for circles and moons) and the direction ARI moves in for
  each — decreasing for the former, increasing for the latter.
- Fig 4's caption uses `B` for the dataset count while
  `overleaf/CARVE_manuscript.tex:775` uses `B` for CARVE's own resample
  count. Line 939 already has the disambiguating form, `B_datasets`; use it
  throughout rather than the bare `B` in both places.
- Fig 4, S1 Fig, S2 Fig and S3 Fig all need re-export after the palette
  re-step and the re-run, even where a panel's data did not change, since
  the four re-stepped series colors (CARVE Generalizability, Davies-Bouldin,
  Calinski-Harabasz, Gap Statistic) appear across all of them.
- `METRIC_COLORS` is global, so every figure drawn through `metric_color`
  changes color when next regenerated, not only the benchmark figures. The
  case-study composites (`_case_study.composite_figure`: `carve_lines`,
  `cvi_lines` and the ARI lollipop) are Fig 5 (`klein_results.png`,
  `:610`) and Fig 6 (`levine_results.png`, `:631`); the Cusanovich figure
  (`cusanovich_results.png`, already in `overleaf/vis/` though the `.tex`
  does not include it yet) and the ablation figures
  (`si_fig_ablation_rho.png`, `si_fig_ablation_b.png`) draw the same
  metric colors, as do `heca_results.png` and `klein_m3c.png` if they are
  added. Regenerate and re-export all of them with Fig 4 and S1-S3 Fig.
  Re-exporting only the benchmark figures leaves Gap, Davies-Bouldin,
  Calinski-Harabasz and Generalizability in two colors across the paper.
  `CARVE_output_klein.png` (Fig 3) and `CARVE_output_levine.png` (S4 Fig)
  are drawn through carve's own plotting and do not change.

## Open item: circles and moons axis separation

Calibration under the fixed solver landed all eighteen anchors inside their
target bands, but circles and moons sit on a steep, high-variance part of
the `embed_param` response — the transition from unsolvable to saturated is
narrow (roughly 2.7 to 3.5), and several anchors landed close to a band
edge: circles/medium at 0.800 mean ARI and t_dist_noise/hard at 0.711, both
right at their band's boundary (see
`docs/superpowers/notes/2026-09-22-calibration-comparison.md` for the full
per-anchor table). The dev-scale acceptance run above is the first place
this will show up concretely: look at whether circles' and moons' three
difficulty levels actually separate in the k-recovery heatmap (dashboard row
4), not just in mean ARI. If they do not separate — if easy, medium and hard
land on indistinguishable k-recovery fractions — the three-point difficulty
axis is not doing its job for these two scenarios, and redesigning it (a
wider `embed_param` spread, a different knob, or dropping the axis) is a new
decision, not something to patch inside this branch's plan.

## Known issue, not fixed here

`_theme.py`'s CARVE Generalizability color is now `#0072B2`, which is also
`PIPELINE_COLORS[0]`, the first entry in the Cusanovich case study's
per-pipeline palette. This branch introduced the collision: Generalizability
was `#56B4E9` before this branch's contrast re-step moved it onto `#0072B2`
to clear the 3:1 contrast floor, and `#0072B2` was already the pipeline
palette's first color. `figures/_cusanovich_results.py` draws CARVE metric
colors in panel C and pipeline colors in panel D of the same composite
figure, so blue now reads as CARVE Generalizability in one panel and as the
first preprocessing pipeline in the other. Recorded in `_theme.py`'s own
comment rather than fixed here, since the pipeline palette belongs to the
Cusanovich case study, outside this spec's scope. If the ambiguity ever
matters for that figure, the fix is re-stepping `PIPELINE_COLORS[0]` and
re-running the five pipeline colors through a CVD check.
