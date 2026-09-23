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
Statistic); see the `_theme.py` `C5` comment block for the before/after
table.

## Current state of results/runs/

`results/runs/` is gitignored. The old 20-dataset/5-dataset mixed tree was
moved to `archive/20260923-162048/` in 0b58047; `results/runs/ablation_rho_b`
was left in place, untouched by this branch.

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
real per-cell timings on this branch's code instead. Script (adapted from
the plan's Step 1 to skip scenarios with no complete run):

```python
import json
from pathlib import Path

from benchmarks._artifacts import read_runtimes, widest_run
from benchmarks._registry import SCENARIOS

ROOT = Path("results/runs")
print(f"{'scenario':26s} {'s/cell':>8s} {'cells':>6s} {'dev CPU-h':>10s} {'20x100 CPU-h':>13s}")
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
    per_cell = float(runtimes["t_default_s"].mean())
    cells = len(runtimes)
    dev_h = per_cell * cells / 3600
    # B and the dataset count both scale the cost linearly.
    scale = (20 / manifest["n_seeds"]) * (100 / manifest["n_resamples"])
    full_h = dev_h * scale
    total += full_h
    print(f"{name:26s} {per_cell:8.2f} {cells:6d} {dev_h:10.3f} {full_h:13.1f}")
print(f"{'TOTAL (measured scenarios only)':26s} {'':8s} {'':6s} {'':10s} {total:13.1f}")
if skipped:
    print(f"skipped (no complete run): {', '.join(sorted(skipped))}")
```

Output:

```
scenario                     s/cell  cells  dev CPU-h  20x100 CPU-h
gaussians                     87.17      9      0.218          14.5
circles                       87.95      9      0.220          14.7
gaussians_dimensionality      39.29      9      0.098           6.5
TOTAL (measured scenarios only)                                     35.7
skipped (no complete run): gaussians_samples, moons, swiss_rolls, t_dist, t_dist_noise
```

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
them, unsplit by the spec. And the measured `gaussians_dimensionality`
number (6.5 CPU-h) sits well under either end of the spec's 23-34 CPU-h
range for the scaling pair, which was projected from the old code's timings
at the old `n_trees` split and without the eigensolver fix. Do not treat 23
and 34 as calibrated per-scenario figures; `gaussians_samples` in particular
could turn out cheaper or more expensive than either number once it is
actually measured.

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

3. Regenerate every figure and table by executing the notebook into a
   scratch directory, not in place, under warnings-as-errors:

   ```
   PYTHONWARNINGS=error .venv/bin/jupyter nbconvert --to notebook --execute \
     --ExecutePreprocessor.timeout=1800 --output-dir=<scratch dir> \
     notebooks/Benchmarking.ipynb
   ```

   One caveat: the notebook's own second cell sets
   `PYTHONWARNINGS=ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning,ignore::RuntimeWarning`
   via `%env`, for the readability of its own output. That assignment only
   reaches subprocesses started after that cell runs; it cannot retroactively
   change the warnings filters already installed in the kernel process at
   startup. Setting `PYTHONWARNINGS=error` in the shell before invoking
   `nbconvert` still enforces warnings-as-errors for the figure- and
   table-generation code that actually runs in-process, which is what this
   check is for. The gate is that nothing raises, no panel is empty, and
   every generated `.tex` fragment compiles.

4. Look at the `circles` and `gaussians_samples` dashboards
   (`vis/benchmarking/scenario_dashboard_circles.png` and
   `..._gaussians_samples.png`) and `vis/benchmarking/benchmarking_results.png`
   in the scratch output. These are the two places most likely to show a
   problem: circles is one of the two scenarios whose difficulty axis was
   never verified to separate under the fixed solver (see Open item below),
   and gaussians_samples is a scaling scenario that has not run to completion
   even once on this code yet.

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

If this passes, a disagreement with the committed CSVs beyond what the
`n_trees` change explains would have been a defect in the runner or artifact
layer, not a consequence of the redesign; passing here is what licenses
switching to `CALIBRATED_ANCHORS` for the real six-scenario run. `swiss_rolls`
is not part of this check: its committed CSV was produced under Ward, and it
is no longer a comparable oracle now that the scenario runs spectral.

Cost: this is a full 3-anchor x 20-dataset x B=100 run of `gaussians`,
`t_dist` and `t_dist_noise`, at `n_trees=100`. The spec's E4 puts the whole
three-scenario pass at about 7 CPU-h, scaled from the old manifests at that
forest size. The measured `gaussians` dev-scale number above, scaled the
same way, comes to 14.5 CPU-h for `gaussians` alone — at `n_trees=500`
rather than the 100 this staged run actually uses, and forest size is not
supposed to be the bottleneck (registry comment on `_N_TREES`), but that gap
is wide enough that budgeting only 7 CPU-h for all three scenarios combined
looks optimistic. Treat 7 CPU-h as the spec's own estimate, not a number this
session re-derived and confirmed; watch the actual wall clock when the
regression suite runs and update this expectation from that, rather than
from either projection.

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
python -m benchmarks.run --scenario gaussians_dimensionality --no-resume   # measured: ~6.5 CPU-h
python -m benchmarks.run --scenario gaussians_samples --no-resume          # projected (spec E4, combined with the above at 23-34 CPU-h for the pair)
```

Since both run serially at one worker, wall clock is close to the CPU-h
figure for each. The spec's combined 23-34 CPU-h estimate for the pair looks
high next to the 6.5 CPU-h `gaussians_dimensionality` measured above, so
"two to three days" (the spec's wall-clock estimate for the pair) may be
pessimistic; do not shorten the scheduled window on that basis alone, since
`gaussians_samples` sweeps a much larger `n_total` (up to 10,000, versus the
difficulty scenarios' fixed 1,500) and has not been measured at all yet.

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
- The swiss rolls estimator sentence in the synthetic benchmarking section
  currently names Ward; it needs to say spectral.
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

`_theme.py`'s CARVE Generalizability color (`#0072B2`) is also
`PIPELINE_COLORS[0]`, the first entry in the Cusanovich case study's
per-pipeline palette. `figures/_cusanovich_results.py` draws CARVE metric
colors in panel C and pipeline colors in panel D of the same composite
figure, so blue reads as CARVE Generalizability in one panel and as the
first preprocessing pipeline in the other. This was already true before this
branch re-stepped Generalizability's color to clear a contrast floor; it is
recorded in `_theme.py`'s own comment rather than fixed, since the pipeline
palette belongs to the Cusanovich case study, outside this spec's scope. If
the ambiguity ever matters for that figure, the fix is re-stepping
`PIPELINE_COLORS[0]` and re-running the five pipeline colors through a CVD
check.
