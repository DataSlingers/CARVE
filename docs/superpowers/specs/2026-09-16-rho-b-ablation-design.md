# Sensitivity of CARVE to the subsampling proportion and the resample count: design

Date: 2026-09-16
Revised: 2026-09-17 (parallelism: decision 9, sections 7.3, 7.5, 10, 11)
Status: awaiting author review, then planning with /superpowers:writing-plans
Supersedes: nothing
Related: `docs/superpowers/specs/2026-09-01-benchmarks-rebuild-design.md` (scenario runner and
artifacts this design extends)

## 1. Motivation

Two reviewer comments in `overleaf/review/comments.tex` concern the subsampling proportion ρ
(`subsample_ratio`) and the number of resampling iterations B (`n_resamples`).

- R4.4 asks how ρ and B were determined, whether a small ρ affects the results, and for a
  sensitivity analysis on both, to be mentioned where the parameters are first introduced. The
  reviewer adds that ideally both should be insensitive.
- R1.3 asks whether resampling biases the estimated number of clusters or reduces the detection of
  rare populations, and how similar the clustering of each resample is to the clustering of the full
  data.

ρ sets how far each resample is from the full data, so one study answers both comments: a sweep
over ρ addresses R1.3 and the ρ half of R4.4, and a sweep over B addresses the rest of R4.4.

The rationale for ρ = 0.618 is recorded in the comment file: it solves ρ² = 1 − ρ, which makes the
expected size of the stability evaluation set (the overlap of the two subsamples) equal to the
expected size of the generalizability hold-out set. The manuscript does not state it; S1 Table lists
0.618 without justification.

## 2. What already exists

Read on 2026-09-16 at `9d869fa`.

Resampling, in `src/carve/_runner.py` and `src/carve/_utils.py`:

- `_resample_indices` draws P1 with `split_subsample_indices(n, subsample_ratio=ρ,
  random_state=random_state + b)` and P2 with `random_state + b + B`. P_test is the complement of P1.
  P1 and P2 each hold `int(ρ n)` samples. Each of P1, P2 and P_test is clustered separately.
- Stability ARI is computed on P1 ∩ P2, of expected size about ρ²n. Generalizability trains a
  classifier on P1 and scores it on P_test, of size about (1 − ρ)n.
- The consensus matrix accumulates co-clustering over P1 only, so a pair of samples is co-sampled in
  about Bρ² resamples. Pairs never co-sampled are NaN (`_consensus.py`) and are filled with 0 when
  the matrix is ordered.
- The seeds one fit uses span `random_state` to below `random_state + 3B` (P1, P2, and the P_test
  pipeline under randomized preprocessing).
- `estimator_results_` stores the mean, standard error (standard deviation over resamples divided by
  the square root of the resample count) and quantiles of each ARI. Per-resample scores are not
  kept, so measuring the effect of B takes separate fits.
- With identity preprocessing ρ enters only through subsampling. `default_dim_reduction_options(X,
  subsample_ratio)` depends on ρ, but only randomized fits use it, and neither the simulations nor
  Klein randomize.
- The package defaults are `CARVE.subsample_ratio = 0.618` and `CARVE.n_resamples = 100`.

Benchmarks, in `src/benchmarks/`:

- `run_cell` (`_run.py`) simulates one dataset, fits CARVE with the scenario's estimator over
  `candidate_k = (3, ..., 7)`, scores every CARVE metric and every CVI, and returns rows in
  `_artifacts.SCHEMA`. It never passes `subsample_ratio`. One seed, `benchmark_seed = seed +
  axis_idx * 10000 + random_state`, sets both the simulated data and CARVE's `random_state`.
- The CVIs, including the gap statistic with 10 reference datasets, depend only on the data, not on
  ρ or B.
- `labels_by_k` in `run_cell` holds the scenario estimator's full-data fit at each candidate k, with
  `random_state=benchmark_seed`.
- `run_scenario` parallelizes over cells with joblib, gives CARVE `n_jobs=1`, checkpoints one parquet
  file per cell and content-addresses the run directory with `config_hash`, which does not include
  `n_jobs`. `run.py` defaults `--n-jobs` to 1.
- CARVE's `n_jobs` is a core budget (`carve._utils.resolve_core_budget`): `outer` resample workers,
  and `inner = max(1, cpu_count() // outer)` threads for each resample's classifier, where
  `cpu_count` is joblib's. Probed on 2026-09-17 inside an 11-worker loky pool (joblib 1.5.3):
  OpenMP and BLAS are capped at one thread (`OMP_NUM_THREADS=1`), but `cpu_count()` still returns 11,
  so a fit with `n_jobs=1` gives its forest 11 threads and the pool can run 121 forest threads on 11
  cores. Setting `LOKY_MAX_CPU_COUNT=1` inside the worker made `resolve_core_budget(1, ...)` return
  (1, 1); joblib's `cpu_count` takes the minimum of the system count and that variable. The core
  budget work on 2026-09-11 measured nested forest threads as slower (eight forest fits: 2.9 s
  single-threaded, 3.7 s with every forest on all cores) and found `estimator_results_` identical
  across thread counts, so the thread count affects speed only.
- The machine has 5 performance and 6 efficiency cores and 18 GiB. Concurrent single-threaded t-SNE
  fits measured on 2026-09-14 gave 4.3, 4.9 and 5.2 single-fit equivalents at 5, 8 and 11 workers.
- `STUDIES["klein"]` declares Ward agglomerative and self-tuning spectral clustering over k = 2..10,
  with scales `dev` (400 cells) and `publication` (0.5, which is 1,358 cells). The Klein notebook
  selects with `measure="generalizability"`, `rule="1se"` and `not_two=True`, with `RANDOM_SEED =
  42`; `not_two` is not part of the Study.
- `fit_or_load_carve` caches whole fitted states at `carve_cache_path`, whose run key includes the
  grids, the preprocessing and `n_resamples` but not `subsample_ratio` or `random_state`.
- `carve._selection.select_best_row_by_rule` returns the selected results row, including its
  `estimator`. `carve._utils.align_cluster_labels` performs Hungarian matching; the benchmarks
  package already imports it (`figures/_case_study.py`).
- Figure functions satisfy `tests/benchmarks/figures/test_figure_contract.py`: they return a
  `Figure`, accept `save` and `out_dir`, and never call `plt.show`.
  `tests/benchmarks/test_notebooks.py` pins code in the notebooks it registers.
- Hard-difficulty Dirichlet concentrations: 0.1 for gaussians, t_dist, t_dist_noise and moons, 0.35
  for circles, 0.5 for swiss_rolls.

Measured cost, from the manifests in `results/runs/` on this Mac at `n_jobs=-1`: the published
simulation grid (20 datasets per difficulty, B = 100) took 22 min for gaussians, 17 min for t_dist,
59 min for swiss_rolls and 105 min for t_dist_noise. circles and moons have only reduced-scale runs,
which extrapolate to about 5.4 h and 3.8 h. That is about 12.6 min of wall clock per (difficulty,
dataset) pair for all six scenarios together. A Klein fit takes about 6 min at `n_jobs=8`.

## 3. Decisions

Taken in the 2026-09-16 brainstorming session.

1. Data. The six difficulty scenarios and the Klein case study. The simulations carry bias in k and
   rare-cluster recovery, which need ground truth. Klein shows whether a real-data selection moves.
   Levine (no local data), Cusanovich and hECA (randomized preprocessing, hours per fit) are
   excluded.
2. Design. One parameter at a time: ρ in {0.2, 0.3, 0.4, 0.5, 0.618, 0.7, 0.8, 0.9} at B = 100, and
   B in {10, 25, 50, 100, 200} at ρ = 0.618. The grid includes 0.5 and 0.8, proportions used
   elsewhere in the resampling literature, so the default can be placed against them. A full
   factorial was rejected on cost; the Bρ² dependence of the consensus metrics is therefore observed
   only along the two arms.
3. The B arm measures repeatability. Several CARVE fits per dataset, each with its own
   `random_state`, reported as agreement of the selected k across fits and as the spread of the
   score curves, alongside k* recovery and ARI.
4. The ρ arm adds two R1.3 measurements: rare-cluster recovery and subsample-versus-full-data
   similarity.
5. The publication run is sized at about four days (section 11). A dev-scale run comes first.
6. Deliverables. A new SI Text with two figures and one table, all generated by code and shown,
   together with every intermediate plot and output, in a notebook. Drafts of the Methods sentence
   and an S1 Text paragraph are written outside `overleaf/`.
7. Architecture. A dedicated ablation runner with its own configuration, schemas and CLI entry.
   Two alternatives were rejected. New Scenario axes for ρ and B: the axis index feeds the dataset
   seed, so each ρ value would see different data, and there is no replicate dimension. Computation
   inside the notebook: a multi-day run needs checkpointing and resume.
8. No change to `src/carve/`, and no change to any published scenario's output or `config_hash`.
9. Parallelism (author, 2026-09-17). Every ablation run uses the worker count that finishes it
   fastest, and no fit runs more threads than its share of the cores. The runner's `n_jobs` defaults
   to -1, and the publication value is chosen among 5, 6, 7, 8 and -1 by a timing check (section
   10).

## 4. Study design

### 4.1 Cells and arms

A cell is one CARVE fit, keyed by (study, difficulty, dataset, ρ, B, replicate). The study is a
scenario name or `klein`; difficulty and dataset are empty for Klein.

The two arms are views over cells, so a cell that belongs to both is computed once:

- ρ arm: ρ over the ρ grid, B at its default.
- B arm: B over the B grid, ρ at its default.

The defaults are read from `dataclasses.fields(CARVE)`, as `_studies.PACKAGE_N_RESAMPLES` already
does, never restated as literals. Configuration validation fails if a default is missing from its
grid.

Publication scale:

| | Simulations | Klein |
|---|---|---|
| ρ arm | 6 scenarios × 3 difficulties × datasets 0–9 × 1 replicate × 8 ρ = 1,440 cells | 8 ρ × 10 replicates = 80 cells |
| B arm | 6 scenarios × {medium, hard} × datasets 0–4 × 3 replicates × 5 B = 900 cells, 60 shared with the ρ arm | 5 B × 10 replicates = 50 cells, 10 shared |
| Similarity draws | 20 per (dataset, ρ, k) | 20 per (ρ, estimator, k) |

Dev scale uses the same grids with simulations at `n_total = 500` instead of 1,500, medium difficulty
only, datasets 0–1 and 2 replicates in the B arm; Klein at its `dev` scale (400 cells) with 2
replicates in both arms; and 5 similarity draws. Dev exercises every code path and figure. Its
numbers are not reported.

### 4.2 Seeds

- Simulated data: unchanged, `benchmark_seed = dataset + difficulty_index * 10000 +
  PUBLISHED_RANDOM_STATE`. Dataset d at a given difficulty is the same data as seed d in the
  published benchmark.
- CARVE `random_state`: `base + replicate * REPLICATE_SEED_SPACING`, where base is `benchmark_seed`
  for the simulations and `PUBLISHED_RANDOM_STATE` (42, the Klein notebook's `RANDOM_SEED`) for
  Klein, and the spacing is 1,000,000. Replicate 0 therefore uses the seed of the published fits.
  One fit's seeds span less than 3 × 200 = 600; replicates spaced closer than that would share
  subsamples and overstate agreement. Because the spacing also exceeds the largest `benchmark_seed`
  (about 20,061) by more than that span, the seed ranges of cells with different replicate indices
  never intersect, whatever their dataset.
- Cells that differ only in scenario, ρ or B share a seed by design, which pairs their comparisons.
  Cells for different datasets may have overlapping seed ranges, as in the published benchmark;
  they draw subsamples of different data.
- Similarity draw m: `base + SIMILARITY_SEED_OFFSET + m`, with the offset at 500,000, used for both
  the subsample draw and the clustering fit.

### 4.3 CARVE configuration per cell

- Simulations: as in `run_cell` (the scenario's estimator and `n_trees`, its candidate k, `n_jobs=1`),
  plus `subsample_ratio=ρ` and `n_resamples=B`.
- Klein: `study_model_grids(STUDIES["klein"])`, `n_jobs=1`, `subsample_ratio=ρ`, `n_resamples=B`.
  The data is loaded once at the configured scale and shared with the workers. Selection uses the
  Study's `not_two` (section 7.1).
- In both, CARVE's `n_jobs=1` means one resample worker per fit. How many threads its classifier gets
  is set by the thread cap in section 7.3.

## 5. Outcomes

Selectors. Every metric in `CARVE_METRICS_ALL` is recorded. The figures and the table lead with the
manuscript's two: stability with the 1SE rule and generalizability with the 1SE rule. Labels come
from the consensus matrix of the metric's mode (`_run._labels_mode`).

Recorded per cell:

1. Score curves: `metric_value` and `metric_se` for every (metric, estimator, k). `metric_se` is the
   `<measure>_se` column where CARVE provides one and NaN otherwise.
2. Selection per metric: the selected estimator and k, through `select_best_row_by_rule` (the path
   `get_k` takes), and the ARI of the selected labels against the reference labels (simulation
   truth; Klein's time points d0, d2, d4 and d7).
3. Simulations only, for every (mode, k): `ari_at_k` as in `run_cell`, and `rare_recall_at_k`.
4. Diagnostics: fit time; the fraction of never co-sampled pairs (NaN entries) in the consensus
   matrices, maximum over configurations; and the number of CARVE cluster-count warnings, captured
   and not escalated.

Recorded once per simulated dataset: the oracle ARI, and the label and size fraction of the smallest
true cluster. These are NaN for Klein.

Derived in the analysis:

- Bias in k: the mean of k̂ − k*, and the k* recovery rate with Wilson intervals
  (`_tables.wilson_ci`), per setting, per scenario and pooled over scenarios.
- Rare-cluster recovery. The smallest true cluster c, with ties going to the lowest label. Consensus
  labels L are aligned to the truth with `align_cluster_labels(y, L)`; recall is the fraction of
  samples with label c in the truth whose aligned label is c. A true cluster left unmatched, which
  happens whenever k is below the true cluster count, has recall 0. Reported at k̂ and at k*, with
  the hard difficulty foremost.
- Replicate agreement (B arm, and both arms for Klein): per dataset and setting, the fraction of
  replicate pairs that select the same k (for Klein, the same estimator and k), averaged over
  datasets. Three replicates give three pairs per dataset; ten give 45.
- Curve spread: the standard deviation of `metric_value` across replicates at each (metric,
  estimator, k), averaged over k.
- Subsample-versus-full similarity (ρ arm). For each dataset, estimator and candidate k, F is the
  full-data fit with seed base, which for the simulations reproduces `run_cell`'s `labels_by_k`. For
  each draw m, S_m = `split_subsample_indices(n, subsample_ratio=ρ, random_state=seed_m)`; X[S_m] is
  clustered with the same estimator and k, and the ARI is taken against F restricted to S_m. The
  reference at ρ = 1 is M refits on the full data with seeds seed_m, scored against F. It separates
  the estimator's own initialization randomness from the effect of subsampling; for Ward it is 1 by
  construction. Klein uses both estimators at k = 2..10.

## 6. Expected mechanisms

Predictions the analysis checks. None is a manuscript claim until the data shows it.

1. Stability ARI should rise with ρ at every k, because the two subsamples share more samples and
   each resembles the full data more closely. As the curves approach 1 they flatten, which may let
   the 1SE and quantile rules select larger k at high ρ.
2. Generalizability trades a larger training set against a smaller and noisier hold-out clustering
   as ρ grows. The direction is not predicted.
3. The standard error shrinks roughly as one over the square root of B, which narrows the 1SE
   tolerance. As B grows, 1SE selections may move toward the max rule's selection, and agreement
   across replicates should rise.
4. The consensus metrics (PAC, Gini, cross-entropy) rest on about Bρ² co-samples per pair: about 4
   at ρ = 0.2 and B = 100, where about 2% of pairs are never co-sampled. They should degrade at small
   ρ before the ARI-based metrics do.
5. Subsample-versus-full similarity should rise toward the ρ = 1 reference. A gap that persists at
   ρ = 0.9 would indicate structure that subsamples do not recover, which is the R1.3 concern.

## 7. Code design

### 7.1 Configuration

- `_types.py` gains two frozen dataclasses. `AblationScale` holds the difficulties, the datasets,
  the B-arm replicates, the Klein replicates, the similarity draws, an optional simulation `n_total`
  override (None at publication scale) and the Klein scale name. `Ablation` holds the name, the
  scenario names, the Klein study name, the ρ grid, the B grid, the scales and the default scale.
  Checks that need nothing outside `_types` run in `__post_init__`: grids sorted, ρ inside (0, 1),
  B at least 2, the default scale declared. `_types` stays a leaf.
- `_registry.py` defines `ABLATIONS = {"rho_b": Ablation(...)}` and, beside
  `PUBLISHED_RANDOM_STATE`, the constants `REPLICATE_SEED_SPACING` and `SIMILARITY_SEED_OFFSET`. A
  validation function run at definition checks what needs the registry: the scenario names exist in
  `SCENARIOS` and use the difficulty axis; the difficulty labels exist on that axis; the CARVE
  defaults are in the grids; the spacing exceeds 3 × max(B grid) plus the largest `benchmark_seed`;
  the similarity seeds fall outside every replicate's range.
- `Study` gains `not_two: bool = False`, and `STUDIES["klein"]` sets it to True, so the ablation reads
  the published selection setting instead of restating it. A notebook test pins that the Klein
  notebook's `not_two=True` agrees with the Study. The Klein notebook itself is not edited.

### 7.2 Sharing code with `run_cell`

The CARVE fit and its scoring move out of `run_cell` into shared code that the ablation also calls.
It takes the data, the reference labels, the grids, `subsample_ratio`, `n_resamples`, the CARVE
`random_state` and `not_two`, and yields the records of section 5. `run_cell` calls it with the
package defaults and projects to `SCHEMA`, so its output for every published scenario is unchanged.
The CVIs stay in `run_cell` and are not recomputed per ablation cell. Function boundaries are settled
in the plan.

The thread cap of section 7.3 lives in this shared code, so a scenario run with `n_jobs` above 1 gets
the same speedup without moving a result row. With `n_jobs=1` the cap is every core, so a timed
scenario run at the default measures what it measures today.

### 7.3 Runner

A new module `_ablation.py` provides `run_ablation(ablation, *, scale, root, n_jobs, resume,
verbose) -> Path`.

- It enumerates the unique cells for the scale. Klein cells run first. Simulation cells follow,
  grouped by dataset index (every scenario and setting for dataset 0, then dataset 1, and so on), so
  a stopped run holds a balanced design over fewer datasets.
- Parallelism. joblib parallelizes over units of work with the loky backend, and `n_jobs` defaults
  to -1. Units are many and small (at most 1,500 samples), so parallelism across units is used rather
  than within a fit, and no process pool is nested inside another. Each CARVE fit gets `n_jobs=1`.
- Thread cap. For the duration of each fit, a unit caps the CPU count its process reports at
  `max(1, cores // workers)`, where `workers` is the resolved worker count. CARVE's core budget then
  gives the forest that many threads instead of every core. The probe in section 2 shows that
  `LOKY_MAX_CPU_COUNT` does this without a change to `src/carve/`; the plan settles the mechanism,
  including restoring the variable afterwards.
- The manifest records the requested `n_jobs`, the resolved worker count, the thread cap and
  `os.cpu_count()`. The per-cell `fit_seconds` are measured with a full worker pool; they feed the
  cost estimate and the diagnostics, not a runtime result.
- Per-dataset work (oracle ARI, smallest true cluster, full-data fits for similarity) runs once per
  (study, difficulty, dataset) and is checkpointed on its own. Similarity runs once per (study,
  difficulty, dataset, ρ) and is checkpointed separately from the CARVE cells.
- One parquet checkpoint per unit; resume skips completed units, as `_artifacts.completed_cells` does
  for scenarios.
- The run directory is `results/runs/ablation_<name>/<scale>/<hash>/`. The hash covers the ablation
  configuration at that scale, the canonical configuration of every referenced scenario
  (`_artifacts._canonical_config`), the Klein study's grids, resolved scale and `not_two`,
  `PUBLISHED_RANDOM_STATE`, both seed constants and the active anchor set name. The scale is in both
  the path and the hash, so a publication read cannot pick up a dev run.
- The manifest records the provenance scenario runs record (git SHA, package versions, anchor set,
  cumulative wall clock) plus the ablation name and scale.
- No fitted CARVE state is saved. Klein fits do not go through `fit_or_load_carve`: its cache key
  omits ρ and the seed, and 120 saved fits, each carrying its n × n consensus matrices, would cost
  far more disk than the rows the analysis reads.

### 7.4 Artifacts

Schemas live in `_artifacts.py` and are validated on write, as `SCHEMA` is. The cell key is study,
difficulty, dataset, subsample_ratio, n_resamples and replicate.

- `ABLATION_CURVE_SCHEMA`: cell key, metric_name, estimator, k, metric_value, metric_se.
- `ABLATION_SELECTION_SCHEMA`: cell key, metric_name, selected_estimator, selected_k, k_star (NaN for
  Klein), ari_selected.
- `ABLATION_AT_K_SCHEMA` (simulations): cell key, mode, k, ari_at_k, rare_recall_at_k.
- `ABLATION_CELL_SCHEMA`: cell key, carve_random_state, n_samples, fit_seconds,
  consensus_nan_fraction, n_cluster_count_warnings.
- `ABLATION_DATASET_SCHEMA`: study, difficulty, dataset, n_samples, k_star, oracle_ari, rare_label,
  rare_fraction.
- `ABLATION_SIMILARITY_SCHEMA`: study, difficulty, dataset, subsample_ratio (1.0 marks the refit
  reference), estimator, k, draw, ari.

`read_ablation(run_dir)` returns the frames by name, and `arm_view(frames, ablation, scale, arm)`
filters them to one arm's cells.

### 7.5 CLI

`python -m benchmarks.run --ablation rho_b [--scale dev|publication] [--n-jobs N] [--no-resume]`.
`--list` also prints the ablation names. `--ablation` is mutually exclusive with `--scenario` and
`--all`. The scale defaults to the ablation's default scale. With `--ablation`, `--n-jobs` defaults
to -1. With `--scenario` and `--all` it keeps its default of 1 (section 13).

## 8. Figures, table and notebook

Figures live in `figures/_ablation.py`, are exported from `benchmarks.figures`, are added to
`EXPECTED` in the figure contract test and take all styling from `benchmarks._theme`.

`figure_ablation_rho`, with ρ on the x axis and a reference line at the default:

- A: k* recovery rate, per scenario and pooled, one column per headline selector.
- B: mean k̂ − k*.
- C: ARI at k̂.
- D: rare-cluster recall at k̂ and at k*, hard difficulty.
- E: mean stability and generalizability ARI at k*.
- F: subsample-versus-full ARI at k*, per scenario, with the ρ = 1 refit reference.
- Klein row: the distribution of the selected (estimator, k) across replicates; ARI of the selected
  labels against the time points; subsample-versus-full ARI at k = 4 for both estimators.

`figure_ablation_b`, with B on a log axis and a reference line at the default:

- A: replicate agreement on the selected k, per scenario and pooled, one column per headline
  selector.
- B: curve spread across replicates, with a guide proportional to one over the square root of B.
- C: k* recovery rate and mean k̂ − k*.
- D: mean reported standard error at k*.
- Klein row: replicate agreement; ARI of the selected labels against the time points.

Panel layout is settled in the plan.

The table function in `_tables.py` renders two sub-tables in the style of S2–S7 Tables with
`render_grouped_tex`: one with a row per ρ value, one with a row per B value. For each headline
selector the columns are the pooled k* recovery rate with its Wilson interval, the mean ARI at k̂
and, in the B sub-table, replicate agreement. Klein's columns are the modal (estimator, k) and its
share of replicates.

The notebook `notebooks/Resampling_Ablation.ipynb` computes nothing. It reads the run directory for a
scale set in its first cell, in these sections:

1. Setup, with the configuration read from `ABLATIONS["rho_b"]` and no grid literals.
2. ρ arm, simulations: per-scenario panels for every outcome in section 5, at all three
   difficulties.
3. ρ arm, Klein.
4. B arm, simulations and Klein.
5. Subsample-versus-full similarity, per scenario and k.
6. Diagnostics: NaN fraction, cluster-count warnings and fit time against ρ and B.
7. The two SI figures and the table, written to the figure and table output paths.

The notebook is registered in `NOTEBOOKS` in `tests/benchmarks/test_notebooks.py`, with a pin that
it reads `ABLATIONS` and contains no ρ or B grid literal. Its executed outputs from the publication
run are committed with it, as with the Cusanovich notebook.

## 9. Manuscript drafts

Written to `_claude_playground/rho_b_ablation/manuscript_drafts.tex`. Nothing is written to
`overleaf/`.

- Methods, where ρ and B are introduced (manuscript line 499): one or two sentences giving the
  defaults, the ρ² = 1 − ρ rationale and a pointer to the new SI Text. The wording must place the
  balance between the evaluation sets (the overlap of P1 and P2, and the hold-out set), not between
  the clustered subsamples: P1 and P2 each hold ρn samples while the hold-out clustering uses
  (1 − ρ)n.
- S1 Text: a short paragraph with the derivation, ρ = (√5 − 1)/2 ≈ 0.618, and the co-sampling count
  Bρ² behind the consensus matrix.
- The new SI Text: design, results and the answer to R1.3, written after the publication run from
  the notebook's outputs. It states that ρ cannot reach 1 in CARVE, so the ρ = 0.9 end and the
  similarity reference are the closest evidence about the full data.
- Any citation for the 0.5 and 0.8 reference points is checked against its source before it appears
  in a draft.

The first two drafts need no results and are written during implementation.

## 10. Testing and verification

Unit tests, in the files that mirror each module:

- Configuration: each invalid configuration named in section 7.1 raises.
- Seeds: for the publication configuration, the seed ranges [s, s + 3B) of cells with different
  replicate indices never intersect, and no similarity seed falls inside any cell's range. Setting
  the spacing to 100 must make this test fail.
- `run_cell`: output for a small scenario is identical before and after the extraction, through the
  existing `test_regression.py` and a direct comparison.
- Rare-cluster recall on hand-built labels: perfect recovery gives 1, a smallest cluster merged into
  another gives 0, and k below the true cluster count leaves it unmatched and gives 0.
- Similarity: Ward's refit reference is 1; a subsample draw has `int(ρ n)` indices.
- Runner: a tiny configuration (one scenario at a small `n_total`, two ρ values, two B values, two
  replicates, a synthetic Study in place of Klein) runs end to end; a cell in both arms is computed
  once; resume skips completed units; dev and publication run directories differ.
- Artifacts: every schema validates on write; `arm_view` returns exactly its arm's cells.
- CLI: `--ablation` parses and is exclusive with `--scenario` and `--all`; its `n_jobs` defaults to
  -1, and `--scenario` still defaults to 1.
- Thread cap: in a real two-worker loky pool, a recording classifier passed to a fit inside a unit
  sees `n_jobs == max(1, cores // 2)`. With the cap removed it sees every core, so the test fails.
  With one worker it sees every core. The CPU-count variable is restored after the fit.
- Figures: the contract test, and both figures render from the tiny run's frames.
- Notebook pins as in section 8, and the Klein `not_two` pin from section 7.1.

Warnings that small test data raises under `filterwarnings = error`, such as the cluster-count
warnings, get `match=` assertions or targeted marks.

Verification before the publication run, in order:

1. The dev run completes (about one hour); the notebook executes with nbconvert into a scratch
   directory; both figures and the table render.
2. The publication-scale Klein cell at ρ = 0.618, B = 100, replicate 0, a single fit of about 6
   minutes, reproduces the published selection: generalizability with the 1SE rule and `not_two`
   selects Ward agglomerative clustering at k = 4.
3. Where a full-scale scenario run from current code exists in `results/runs/`, the CARVE metric
   rows of the simulation cells at ρ = 0.618, B = 100, replicate 0 equal that run's rows for the
   same scenario, difficulty and dataset.
4. Timing check. One fixed batch of publication-scale cells, including Klein cells, with at least 22
   units so every worker count up to 11 stays busy, runs into a scratch root once at each `n_jobs`
   in {5, 6, 7, 8, -1}. The batch is sized to take about 15 minutes at -1, which puts the check at
   about 1.5 hours. The value with the shortest wall clock is the publication run's `n_jobs`, and
   the publication cost estimate is re-derived from its throughput before the four-day run starts.

## 11. Cost and run order

Estimates from section 2; circles and moons carry about ±30%.

- ρ arm, simulations: 30 (difficulty, dataset) pairs × 8 ρ values × 12.6 min, scaled by the (1 + ρ)n
  samples each resample clusters relative to the default: about 48 h, plus about 12% for
  similarity, about 54 h.
- B arm, simulations: 10 pairs × 3 replicates × (10 + 25 + 50 + 100 + 200)/100 × 12.6 min: about
  24 h, less the 60 shared cells.
- Klein: 120 unique fits. At 6 min per fit on 8 cores, run one after another, that is 12 h. The
  runner fits them concurrently with capped forests, so the timing check re-derives this figure.
- Total: about 90 h.

Run order within the ablation is set by the runner (section 7.3). The Cusanovich 150-resample run
needs the same machine; which runs first is the author's call.

## 12. Risks

- At ρ = 0.2 some consensus entries are NaN (section 6, item 4). CARVE fills them with 0 when
  ordering. The dev run confirms that labels, Gini and cross-entropy come back finite, and the NaN
  fraction is recorded per cell.
- At ρ = 0.9 the hold-out set holds 150 samples in the simulations and 136 cells in Klein, clustered
  into up to 7 and 10 clusters. A clustering with fewer clusters than requested raises a CARVE
  warning; the runner counts these per cell and the notebook reports them.
- The cost of circles and moons is extrapolated from reduced-scale runs.
- The B arm has 5 datasets per difficulty. Per-scenario agreement estimates are coarse; the pooled
  estimates carry the result.

## 13. Out of scope

- A ρ × B factorial.
- Levine, Cusanovich and hECA.
- The R port.
- Changing CARVE's defaults, or storing per-resample scores in CARVE.
- Any change to a published scenario's output.
- Changing the `--n-jobs` default for `--scenario` and `--all`. Their result rows do not depend on
  it, but the scaling scenarios time their fits, and concurrent workers change those timings.
- Edits to `overleaf/`.
