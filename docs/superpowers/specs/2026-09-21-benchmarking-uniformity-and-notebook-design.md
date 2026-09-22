# Benchmarking uniformity, correctness and notebook redesign

Date: 2026-09-21

## Purpose

Before the rho/B ablation is handed off, bring the simulated benchmarks to a state where
every scenario runs the same experiment, the runner cannot produce a silently wrong
artifact, and the output notebook answers its questions visually rather than in tables.

The spec covers code and verification only. The full re-run is written up here as a
handoff and is not executed as part of the implementation.

## Scope

In scope: `src/benchmarks/`, `src/carve/cluster.py` (one defect), `notebooks/Benchmarking.ipynb`,
and the tests mirroring those modules.

Out of scope: the R port under `code/carve-r/` (stated by the author), the case studies,
the rho/B ablation itself, and `../overleaf/` (read-only; manuscript consequences are
listed as manual follow-ups).

## Findings this spec responds to

Measured or read off the code on 2026-09-21, not assumed.

### Sample counts are already uniform; the artifacts are not

Every scenario simulates `n_total = 1500`. Confirmed by simulating all 24 cells: shapes are
`(1500, p)` everywhere except the two scaling scenarios, which sweep n deliberately.

What is not uniform is the artifact tree. `results/runs/` holds four scenarios at 20
datasets x B=100 (2026-09-07, sha 033768b: gaussians, t_dist, t_dist_noise, swiss_rolls) and
four at 5 datasets x B=20 (2026-09-03: circles, moons, gaussians_samples,
gaussians_dimensionality). The notebook selects the widest sweep per scenario, so the
current `benchmarking_results.png` and the S2-S9 fragments mix two scales with nothing
marking it. Swiss rolls therefore reads as having run on twenty times the resampling budget
of circles and moons, its two siblings in the same family.

The code's defaults are already correct: `Scenario.n_seeds = 20` for all eight scenarios and
the CLI's `--n-resamples` defaults to 100, which is also CARVE's own default and what
`overleaf/CARVE_manuscript.tex:775` states. The 5 x 20 runs came from explicit flags on
exploratory runs. No registry change is needed to make the scale uniform; the artifacts must
be discarded and regenerated.

Terminology, since the manuscript overloads the symbol B: `n_seeds` is the number of
distinct simulated datasets per axis point, each with its own derived seed
(`seed + axis_idx * 10000 + random_state`); `n_resamples` is CARVE's resample count B.
Fig 4's caption uses B for the former and `CARVE_manuscript.tex:775` for the latter;
`CARVE_manuscript.tex:939` disambiguates with `B_datasets = 20`.

### The spectral eigensolver defect reverses an estimator choice

`src/carve/cluster.py:263` calls `eigsh(Lsym, k=k, which="SM", tol=1e-4, maxiter=5000)` with
no start vector. `_spectral_embedding` takes that branch at n >= 1000. The consequences are
nondeterminism, non-convergence, and a 13-20x slowdown, all measured previously and recorded
in the project memory. The R port at `carve-r/R/cluster.R:270` is already correct.

Measured here, oracle ARI at k\*=5 on swiss_rolls over 20 datasets:

```
arm                  easy            medium           hard      s/fit
ward            0.982+/-0.080   0.860+/-0.120   0.752+/-0.154     0.02
spectral-current 0.763+/-0.236  0.761+/-0.130   0.806+/-0.187     5.93
spectral-fixed  1.000+/-0.000   0.900+/-0.143   0.834+/-0.216     0.19
```

Spectral as implemented today loses to Ward on swiss rolls by 0.22 ARI at the easy anchor,
which is why the scenario runs Ward. With the R port's shift-invert solver, spectral wins at
every difficulty and runs thirty times faster. The estimator choice recorded in the registry
is an artifact of the defect, not a property of the data.

### The difficulty anchors are not calibrated to the bands the manuscript claims

The manuscript states the base estimator informed with the correct k reaches ARI of
approximately 0.9-1.0 (easy), 0.8-0.9 (medium), 0.7-0.8 (hard). `_calibrate.py` encodes the
same bands. The published oracle ARI, read from the committed `results_*.csv`:

```
                easy  medium   hard
circles        0.875   0.822  0.781    easy below band
moons          0.891   0.861  0.802    easy below band
gaussians      0.914   0.868  0.724
swiss_rolls    0.982   0.860  0.752
t_dist         0.983   0.888  0.724
t_dist_noise   0.981   0.848  0.721
```

Circles and moons miss the easy band, so "easy" means ARI 0.88 on one scenario and 0.98 on
another and the six panels are not on a common scale.

### Circles and moons have no difficulty gradient under a correct solver

Measured with the shift-invert solver, oracle ARI at k\*=5 over 20 datasets:

```
                             easy           medium           hard
circles  spectral-current   0.861+/-0.162  0.822+/-0.140  0.781+/-0.159
         spectral-fixed     1.000+/-0.000  1.000+/-0.000  1.000+/-0.000
moons    spectral-current   0.901+/-0.146  0.853+/-0.137  0.768+/-0.217
         spectral-fixed     1.000+/-0.000  1.000+/-0.000  1.000+/-0.000
swiss    spectral-current   0.739+/-0.218  0.761+/-0.130  0.817+/-0.180
         spectral-fixed     1.000+/-0.000  0.900+/-0.143  0.834+/-0.216
```

Circles and moons saturate at 1.000 with zero variance at every anchor. Multiplying
`cluster_scale` by 3 does not move them. The apparent difficulty gradient in the published
results — 0.875, 0.822, 0.781 on circles — was the eigensolver failing to converge at
different rates across anchors, not the data becoming harder. Those two scenarios have been
measuring solver behavior rather than difficulty.

The swiss_rolls `spectral-current` row differs from the same arm in the previous table
(0.739 / 0.761 / 0.817 against 0.763 / 0.761 / 0.806) because the two tables come from two
separate invocations. Same data, same seeds, same estimator, different answer: that
disagreement is the nondeterminism defect showing itself directly.

The parameter that does control difficulty on the RFF shapes is `embed_param`, the random
Fourier feature bandwidth, and it behaves as a threshold rather than a gradient:

```
circles/medium, 20 datasets, fixed solver
 embed_param   mean ARI      sd     min     max
        2.00      0.253   0.137   0.130   0.725
        2.25      0.325   0.156   0.132   0.723
        2.50      0.513   0.207   0.132   1.000
        2.75      0.668   0.226   0.293   1.000
        3.00      0.800   0.186   0.386   1.000
        3.25      0.910   0.162   0.532   1.000
        3.50      0.967   0.104   0.617   1.000
        4.00      1.000   0.000   1.000   1.000

moons/medium, 20 datasets, fixed solver
 embed_param   mean ARI      sd     min     max
        2.00      0.300   0.141   0.141   0.527
        2.25      0.490   0.179   0.142   0.765
        2.50      0.604   0.184   0.141   1.000
        2.75      0.782   0.175   0.486   1.000
        3.00      0.842   0.168   0.400   1.000
        3.25      0.965   0.108   0.615   1.000
        3.50      0.981   0.086   0.615   1.000
        4.00      1.000   0.000   1.000   1.000
```

All three target bands fall between roughly 2.7 and 3.5 — hard near 2.7 to 3.0, medium near
2.9 to 3.2, easy from about 3.25 up. Below 2 the problem is unsolvable (ARI near 0, and
non-monotone: 0.034 at 0.5, -0.026 at 1.0); above 4 it is trivial. The published anchors sit
at 12.0, 7.3, 6.0 for circles and 10.7, 5.7, 14.5 for moons, all in the saturated region, and
the moons values are not even ordered.

`corr_strength` has no effect on these scenarios: ARI is 1.000 at 0.23, 0.5, 0.7 and 0.9.

Two properties of this band matter for the calibration design. ARI increases with
`embed_param`, the opposite direction to `cluster_scale`, and `_calibrate.py` hardcodes the
decreasing assumption. And the per-dataset spread inside the band is large — sd 0.10 to 0.23,
individual datasets from 0.39 to 1.000 at the same setting — so a calibrated mean on these
scenarios sits on a much wider distribution than on the Gaussian ones.

### Three runner and artifact defects

1. `run_scenario` writes `manifest.json` only after the joblib pool finishes (`_run.py:520`).
   An interrupted scenario leaves a directory with checkpoints and no manifest, which the
   notebook's run-directory resolver cannot open. The ablation runner received the
   manifest-first fix in e720073; the scenario runner did not.
2. Run directories are content-addressed on configuration only. `git_sha` is written into
   the manifest and never checked, so resuming onto newer code concatenates two code
   versions' checkpoints. This is what left the ablation dev run unreadable.
3. `--tables` selects the lexically last run directory with a warning (`run.py:139`) while
   the notebook selects the widest sweep. They disagree today: S4 regenerated from the CLI
   is the 5-dataset run.

A fourth item is ergonomic rather than a defect: one `--n-jobs` flag serves both the six
difficulty scenarios, which want every worker, and the two scaling scenarios, whose timings
are only meaningful at one worker. `--all` cannot be correct for both, and the current
default of 1 turns a seven-hour job into a three-day one.

### The published palette has one accessibility failure

Fig 4's seven series, checked with the dataviz validator against a light surface:

```
[PASS] Lightness band       all 7 inside L 0.43-0.77
[FAIL] Normal-vision floor  Davies-Bouldin #A8389E <-> Silhouette #E0457B, dE 13.6 (floor 15)
[PASS] CVD separation       worst adjacent dE 10.8 protan / 11.2 tritan
[WARN] Contrast vs surface  #56B4E9 at 2.25:1, #F28522 at 2.51:1 (floor 3:1)
```

The hard failure is visible in the published figure: in the t-distributed panel the
Silhouette and Davies-Bouldin curves are hard to separate. The contrast warning lands on
CARVE generalizability, a headline series drawn as the palest line on the page.

### Two small inconsistencies

`swiss_rolls` sets `center_box = 3.0`, which is `simulate_clusters`'s own default, so the key
reads as meaningful and is not. `TABLE_CAPTIONS["gaussians_dimensionality"]` says "over
embedding dimension" while the axis is `p`, the ambient feature dimension; the registry
comment and the manuscript's S3 Fig both say `p`.

## Design

### A. Registry and study design

A1. Set `n_trees` to 500 for all eight scenarios. Today it is 500 on circles, moons and
swiss rolls and 100 elsewhere, determined by which published notebook cell passed the
argument. 500 is the better generalizability estimate; the cost is linear in a term that is
not the bottleneck.

A2. Fix the eigensolver in `carve/cluster.py`, mirroring the R port:
`eigsh(Lsym, k=k, sigma=0.0, which="LM", tol=1e-4, maxiter=5000)`, keeping the existing
`ArpackNoConvergence` handling and dense fallback. Python only.

A3. Move `swiss_rolls` to the spectral estimator, as a consequence of A2 rather than an
independent choice. This makes the RFF family one family: circles, moons and swiss rolls all
on self-tuning spectral clustering.

A4. Regenerate the difficulty anchors through `_calibrate.py` so every anchor lands inside
its documented band. The module needs three changes, the first two structural:

- `CALIBRATION_SPACE` becomes per-scenario rather than one global entry. Two knobs are
  needed, because no single parameter controls difficulty on every scenario:

  ```
  scenario          knob                       interval        ARI direction
  gaussians         cluster_scale multiplier   [0.1, 3.0]      decreasing
  t_dist            cluster_scale multiplier   [0.1, 3.0]      decreasing
  t_dist_noise      cluster_scale multiplier   [0.1, 3.0]      decreasing
  swiss_rolls       cluster_scale multiplier   [0.1, 3.0]      decreasing
  circles           embed_param                [2.0, 4.0]      increasing
  moons             embed_param                [2.0, 4.0]      increasing
  ```

  The direction is a declared property of the knob, not an assumption in the bisection. Today
  `calibrate_anchor` hardcodes "ARI decreases as cluster_scale grows", which is correct for
  the four scale-driven scenarios and inverted for the two RFF ones.

- Where the knob is `cluster_scale`, bisect a multiplier on the anchor's published vector
  rather than replacing the vector with a scalar. The nonlinear anchors carry unequal
  per-cluster scales, for example `[4.38, 4.08, 4.08, 4.08, 4.08]` on circles/hard, which
  encode the scenario's character; a scalar bisection flattens them. The objective stays
  monotone in the multiplier, so bisection remains valid.

- Record the achieved mean ARI, its standard deviation across the calibration datasets, and
  the resolved knob value alongside each regenerated anchor, so the calibration's provenance
  lives in the code rather than in a notebook that can be lost again. The standard deviation
  is recorded because the RFF transition band is wide in spread even where the mean is on
  target, and a reader of the anchors should see that.

Circles and moons keep their three-point difficulty axis. It is the first time that axis will
correspond to an actual difficulty gradient: their published anchors produce ARI 1.000 at
every level once the solver is correct, so the existing S5 and S6 tables and the circles and
moons panels of Fig 4 are superseded outright rather than shifted.

All six difficulty scenarios are recalibrated, not only the three the eigensolver fix moves.
A half-published, half-calibrated anchor set is the same class of inconsistency this spec
exists to remove. `PUBLISHED_ANCHORS` stays in the file for comparison and revert;
`ACTIVE_ANCHOR_SET_NAME` already reaches every run manifest.

Calibration runs with the fixed eigensolver and each scenario's final estimator, otherwise it
calibrates against a solver that will not be used.

A5. Remove `center_box = 3.0` from the swiss_rolls shared settings.

A6. Correct the `gaussians_dimensionality` table caption to name the feature dimension `p`.

### B. Runner and artifacts

B1. Write `manifest.json` before the pool with a `status` field, flipping it to complete when
the pool finishes, mirroring the ablation runner. Run readers refuse an incomplete run unless
explicitly asked for it.

B2. Refuse to resume into a run directory whose manifest records a different `git_sha`,
with an explicit override flag. The value is already recorded; only the check is missing.

B3. One run-directory resolver in `_artifacts`, shared by `--tables` and the notebook,
selecting the widest sweep by `(n_seeds, n_resamples)` and raising on a tie rather than
picking arbitrarily. The CLI's lexical-last selection and the notebook's private `run_dir`
helper are both deleted in favor of it.

B4. Resolve `n_jobs` per scenario class inside the runner: workers for the six difficulty
scenarios, forced serial for `TIMED_SCENARIOS`, with an explicit `--n-jobs` still overriding
both. `--all` becomes correct and fast simultaneously.

B5. Archive the existing `results/runs/<scenario>/` tree before anything is run. That tree is
gitignored. The tracked `results/results_*.csv` and `results/scalability_results/` stay in
place as the regression oracle.

### C. The notebook

C1. Add `figure_scenario_dashboard(name, frame)`, one figure per scenario section, replacing
the current overview-plus-table pair. Four rows:

- Row 1, example scatters, one panel per axis point. Existing `draw_example_row`, unchanged,
  so the panels show the data the benchmark scored.
- Row 2, ARI at the selected k over the axis. Fig 4's series and legend, reading their colors
  from the same `_theme` entries, so a section and the manuscript figure cannot disagree.
  Per-seed points behind each mean, dodged along x per method rather than overplotted, at low
  alpha.
- Row 3, criterion over k, one panel per axis point. Each criterion is min-max normalized over
  the candidate k within each dataset, then averaged over datasets. Mean ARI-at-k is drawn on
  the same [0, 1] axis as a grey reference; a second y axis is not used. Each method's
  selected k is marked and k\* is ruled. The gap statistic's marker will not sit on its own
  maximum, because Tibshirani's rule is not an argmax; that is the panel showing the rule at
  work, not a defect.
- Row 4, P(k-hat = k) as a method-by-k heatmap, one panel per axis point, on a single-hue
  sequential ramp added to `_theme`. This is the table's k-recovery column made visual.

One shared legend at the foot, through the existing `metric_legend`.

C2. The two scaling scenarios take the same dashboard with the axis relabeled, alongside the
runtime panels they already have.

C3. Delete `scenario_report` from the notebook. `summarize` stays as the engine behind the
manuscript tables.

C4. Section 4 keeps the four manuscript figures and `write_all_tables` unchanged in role.

C5. Re-step the failing hues in `_theme`, keeping blue and green for CARVE and
pink/purple/red/orange for the comparators:

```
series                    published   re-stepped   reason
Baseline (oracle)         #595959     unchanged    neutral grey reference
CARVE Stability           #009E73     unchanged    green
CARVE Generalizability    #56B4E9     #0072B2      stays blue, clears 3:1 (was 2.25)
Silhouette                #E0457B     unchanged    pink
Davies-Bouldin            #A8389E     #6A2C91      deeper purple, clears the dE 15 floor
Calinski-Harabasz         #D6292E     #B01B20      deeper red
Gap Statistic             #F28522     #C96A05      deeper orange, clears 3:1 (was 2.51)
```

That set passes every validator check: lightness band, chroma floor, CVD separation at worst
adjacent dE 8.7 protan and 8.6 tritan, normal-vision floor at worst adjacent dE 16.2, and
contrast at or above 3:1 for all seven. The grey oracle's chroma-floor flag is a deliberate
exception: it is a neutral reference line, not a categorical series.

Each of the four comparators additionally gets its own dash pattern. Four warm hues cannot
all clear the CVD threshold pairwise; hue does the work it can and dash carries the rest.
The validator report goes in the commit message.

Known and not changed: `_theme` colors `ari_average*` orange and the three consensus metrics
grey, both of which sit in the comparator family rather than CARVE's blue and green, and the
greys collide with the oracle's. Those metrics reach the tables only, never a figure drawn by
this spec, so they are recorded rather than re-stepped.

### D. Manuscript tables

D1. Rework `render_grouped_tex` into the published shape: two tabulars side by side inside a
`\resizebox{\textwidth}{!}{...}` separated by `\quad`, the left giving mean ARI(k-hat) and the
right k-recovery, columns `Metric | easy | medium | hard`, booktabs rules, and four row groups
separated by `\midrule`. Bold marks the best value in each column and underline the
second-best; ties for best are all bold with no underline, matching the committed S2 table.
The oracle's k-recovery cells stay empty.

D2. Declare the row order and grouping in the registry rather than deriving it from a sorted
set, so the generated fragment cannot drift from the manuscript's layout. The order the
committed S2 table uses, which is what the renderer must reproduce:

```
group 1  Baseline (Oracle)
group 2  CARVE Stability (1SE), CARVE Generalizability (1SE)
group 3  Davies-Bouldin, Silhouette, Gap Statistic, Calinski-Harabasz
group 4  ARI (stab, quantile), Gini (stab), ARI (stab, max),
         ARI (gen, quantile), ARI (gen, max), Accuracy (gen)
```

`EXCLUDED_METRICS` already drops the three `ari_average*` variants, PAC and CE, which is why
they do not appear.

### E. Verification

E1. A dev-scale acceptance run: all eight scenarios at `--n-seeds 3 --n-resamples 10
--no-resume`, then every figure and every table regenerated from the result. That scale
exercises every code path — three datasets is enough for a standard deviation and a
k-recovery fraction, ten resamples is enough for a consensus matrix — while costing minutes
rather than hours. The gate is that nothing raises, no panel is empty, and every generated
`.tex` fragment compiles. Its numbers are not reported.

E2. A test for the eigensolver fix that fails on the current code: a hard cell at n=1500, k=5,
asserting both determinism across repeated fits and ARI near 1.0. `tests/test_cluster.py:256`
already asserts sparse-path reproducibility and passes on the broken code because it uses
n=1000, k=2 on easy moons, where the flip does not occur; the new test must not repeat that.

E3. Tests for each new panel, the dashboard assembly and the table renderer, in the mirrored
files under `tests/benchmarks/`.

E4. The full re-run is documented as a handoff, not executed. Scaled from the per-cell
timings recorded in the existing manifests, six difficulty scenarios at 20 datasets x B=100
come to roughly 75 CPU-h, about seven hours wall at eleven workers — the previous estimate of
68 plus swiss rolls moving from Ward to spectral, which makes it cost like circles and moons.
The two scaling scenarios are roughly 23 and 34 CPU-h and must run serially, so they are two
to three days and stay parked. These are projections, not measurements; the dev-scale run in
E1 is the place to re-derive them from the new code.

Validation strategy for the re-run. The committed `results_*.csv` files are the only oracle
for the runner's correctness, and A4 moves the anchors out from under them, so the comparison
has to be staged. Run gaussians, t_dist and t_dist_noise once against `PUBLISHED_ANCHORS`
with the new code before switching `ACTIVE_ANCHORS`: those three run k-means and Ward, which
the eigensolver fix cannot touch and A1 changes only through `n_trees`, so any disagreement
with the committed CSVs beyond the forest change is a defect in B1-B4 rather than a
consequence of the redesign. Only then switch to the calibrated anchors and run all six. This
costs one extra pass over the three cheapest scenarios, about 7 CPU-h, and is what separates
"the runner is correct" from "the experiment changed".

## Where this lands

`rho-b-ablation` is 27 commits ahead of `main` and 0 behind, so it merges as a fast-forward.
B4 builds on the `cpu_cap` and `thread_cap_for` work that lives on that branch, so this spec
cannot start from `main` as it stands.

The order is: fast-forward `main` to `rho-b-ablation` and push, then branch this work off
`main`. The ablation's own remaining gates — its dev re-run and the multi-day publication run
— stay open independently of the merge; nothing here depends on them.

## Consequences outside this repository

All manual, since `../overleaf/` is read-only:

- The estimator sentence in the synthetic benchmarking section names Ward for swiss rolls.
- S3 Text's anchor values change, and the calibration procedure it describes should name both
  knobs and the direction each runs in.
- The circles and moons findings in the main text rest on results that were produced by a
  defective eigensolver at anchors where the correct solver is saturated. The sentence
  "On circles and moons, CARVE's metrics outperformed the CVIs consistently, while most CVIs
  struggled across all SNR levels" cannot be carried over; it has to be rewritten against the
  re-run. The same applies to the swiss rolls sentence, whose estimator also changes.
- Fig 4's caption uses B for dataset count while line 775 uses B for CARVE's resample count.
  Line 939's `B_datasets` is the disambiguating form and should be used throughout.
- Fig 4, S1 Fig, S2 Fig and S3 Fig need re-export after the palette re-step and the re-run.
- S2-S9 fragments are regenerated by D1 and can be dropped in without hand-editing.

## Decisions taken

- Sample count is already uniform; the scale mixture is an artifact problem, fixed by
  discarding and re-running, not by changing the registry.
- `n_trees` unifies at 500.
- swiss_rolls moves to spectral, coupled to the eigensolver fix, on measured evidence.
- All six difficulty scenarios are recalibrated rather than only the ones that moved.
- Circles and moons are recalibrated on `embed_param` and keep their three-point axis, rather
  than being widened, redesigned or dropped. Their published results are treated as
  superseded.
- The notebook drops its inline tables entirely in favor of one dashboard per scenario.
- The criterion row normalizes rather than using a second y axis.
- The S2-S9 fragments are fixed to match the manuscript rather than deferred.
- The palette is patched at the failing hues rather than re-stepped wholesale.
- The spec stops at code-complete and verified at dev scale; the full re-run is a handoff.
- `main` is fast-forwarded to `rho-b-ablation` first; this work branches off `main`.

## Open items

- The recalibrated anchors are compared against `PUBLISHED_ANCHORS` before adoption, as
  `_calibrate.py`'s docstring already requires. The previous comparison is at
  `docs/superpowers/notes/2026-09-01-calibration-comparison.md`; this one extends it.
- Circles and moons sit on a steep, high-variance part of the `embed_param` response, so the
  calibrated anchors will be closer together than on the other scenarios and more sensitive
  to the calibration seed set. The dev-scale acceptance run (E1) is the first place this will
  show; if the three anchors do not separate in k-recovery, the axis design for these two
  scenarios has to be revisited, and that is a new decision rather than a plan step.
- The two scaling scenarios' anchors vary `cluster_scale` along the axis to hold difficulty
  roughly constant. They are not covered by `TARGET_ARI_BANDS`, which is keyed on
  easy/medium/hard, so A4 leaves them alone. Whether they should be calibrated against a
  target band of their own is deferred.
- `_theme` colors `ari_average*` orange and the three consensus metrics grey, which sit in
  the comparator family rather than CARVE's blue and green. Recorded under C5; not changed,
  because those metrics reach the tables only.
