# hECA v2.0 at full scale on Longleaf: runtime design

Date: 2026-10-01
Status: design approved, not implemented
Reviewer comment addressed: R1.1

## Motivation

Reviewer 1, comment 1:

> It would be helpful to report runtime benchmarks, memory usage, computational complexity, and
> parallelization strategies on datasets of increasing size, including large-scale single-cell
> datasets (for example >500,000 cells).

The increasing-size part is covered by the Cusanovich runtime ladder (5,000 to 81,173 cells) and
its parallel check at 10,000 cells, measured on 2026-09-15. What is missing is a run above
500,000 cells. The hECA case study was designed for that on 2026-09-07 but has never run at full
scale: the pooled embedding of all annotated cells needs tens of GB of memory and has never been
computed, and the only local cache is a 5 MB subsample.

This design runs one CARVE fit on every annotated cell of the five pooled hECA organs on UNC's
Longleaf cluster. Runtime and memory are the primary deliverable; the clustering result is
secondary.

## Decisions

Made by the author on 2026-10-01:

- One CARVE fit on all annotated cells, sweeping Leiden resolution for two settings: a
  15-neighbor graph and a 50-neighbor graph.
- No scaling ladder, no k-based fit, no CVI comparison. Code used only by those is removed.
- Runtime reporting: totals for the fit plus a component breakdown (neighbor search, graph
  assembly, Leiden, forest fit, forest predict, CARVE overhead).
- A batch entry point in `src/benchmarks/` with thin sbatch scripts; the notebook draws figures
  locally from copied-back artifacts.
- A calibration stage on the cluster chooses the resolution grid and projects the fit's
  runtime before the fit starts.
- The pooled embedding stays as the loader computes it, without batch correction. Study
  composition per cluster is reported as a caveat.
- Target machine: Longleaf. The author has no account yet.

Added while writing this spec, not discussed: a `status` command that reads the fit's live
timing rows. The fit has no checkpoint, and the calibration projection is a lower bound, so the
first configurations are the earliest check on the real runtime.

## Data

Counted on 2026-10-01 from the local h5ad files' obs. The five default organs hold 764,859
cells; 695,304 remain after dropping `Unclassified`. They carry 54 harmonized cell types and come
from 8 source studies, fetal and adult.

| Organ | Cells | Cell types | Studies |
|---|---|---|---|
| Brain | 261,680 | 18 | 5 |
| Lung | 201,666 | 18 | 3 |
| Heart | 152,355 | 11 | 2 |
| Kidney | 46,537 | 19 | 2 |
| Thymus | 33,066 | 3 | 2 |

Cell-type sizes run from 59 to 66,999 cells (median 4,462); 8 types have fewer than 1,000
cells.

| study_id | Source | Organs | Cells |
|---|---|---|---|
| 10.1016/j.cell.2021.10.024 | Zhang 2021 (CATlas) | Brain, Heart, Lung, Thymus | 141,522 |
| 10.1126/science.aba7612 | Domcke 2020 (fetal atlas) | Brain, Lung, Thymus | 136,522 |
| 10.1038/s41588-021-00894-z | Morabito 2021 | Brain | 129,902 |
| 10.7554/eLife.62522 | Wang 2020 | Lung | 91,499 |
| 10.1126/sciadv.abf1444 | Hocker 2021 | Heart | 79,507 |
| 10.1016/j.celrep.2022.110467 | Zhang 2022 (pituitary) | Brain, Kidney | 78,104 |
| 10.1016/j.cell.2021.07.039 | Trevino 2021 (fetal cortex) | Brain | 29,716 |
| 10.1038/s41467-021-27660-3 | Wang 2022 | Kidney | 8,532 |

## Configuration

`STUDIES["heca"]` becomes a resolution study, like Cusanovich:

- `estimator`: Leiden with `n_neighbors=15`.
- `partners`: Leiden with `n_neighbors=50`.
- `candidate_k`: empty.
- `resolutions`: the provisional grid below until calibration replaces it.
- `n_resamples`: 100. `consensus_anchors`: 2000. Seed 42. `scales`: `{"publication": None}`.

Both settings otherwise use the registered Leiden defaults: modularity, connectivity weights,
Euclidean metric, no scaling.

`EstimatorSpec` holds only a name today, so it gains `params`, a tuple of (name, value) pairs
that override `ESTIMATOR_DEFAULTS` for that spec. It stays frozen and hashable, and an unknown
parameter name raises. `param_grids` and `resolution_grids` apply the overrides.
`study_resolution_grids` returns grids for the estimator and every partner. Cusanovich has no
partners and is unaffected. The Leiden fallback for k-based studies served hECA alone and is
removed; a study whose estimator does not sweep resolution now raises. The two settings must
appear as distinct methods in `estimator_results_`.

Removed with the dropped runs: `study_scaling_sweep`, `figure_study_scaling`
(`figures/_study_scaling.py`), `figure_heca_results` (`figures/_heca_results.py`) and their
tests. The constants the reference scatter uses (`AXIS_LABELS`, `DEFAULT_SCATTER_SUBSAMPLE`,
`MARKER_SIZE`) move to the module that replaces `_heca_results.py`. `cvi_sweep` stays; Klein and
Levine use it.

Preprocessing is unchanged: cPeaks open in at least 0.5% of cells, 50,000 highly variable
cPeaks, log-normalization, 50 principal components. The loader's metadata
(`datasets/_heca.py:662`) gives memory as the reason for 50,000 against the source's 500,000.
That reason no longer holds on the cluster and is restated: hECA's 500,000 belongs to its PeakVI
visualization; its annotation selected peaks per dataset with no fixed count. 50,000 keeps the
embedding cheap and does not enter CARVE's runtime.

`load_heca` returns one label column. Calibration and the notebook score against organ and
against cell type, and the notebook reports study composition, so the loader also returns the
annotated cells' `cell_type`, `organ` and `study_id` in its metadata, aligned to the rows of X.
The CARVE fit itself is unsupervised and uses no labels.

## Resolution grid

### What the sources used

hECA v2.0 published no clustering of the pooled collection. Each source study clustered its own
cells, mostly after batch correction, often in two rounds (major types, then subtypes). Methods
read on 2026-10-01; Trevino 2021 is not open access.

| Source | Graph | Algorithm | Resolution reported |
|---|---|---|---|
| Zhang 2021 (CATlas) | 50 neighbors, SnapATAC embedding | Leiden, modularity in the first round, CPM in the final round | Not stated; swept and chosen by perturbation stability and silhouette |
| Hocker 2021 | 50 neighbors, same pipeline | Leiden, modularity then CPM | Not stated |
| Wang 2020 | 30 neighbors, cosine, after Harmony | Leiden | 1.5 |
| Wang 2022 | 15 neighbors, SnapATAC | Louvain | 1.0 (default) |
| Domcke 2020 | LSI after Harmony, per tissue | Louvain (Seurat v3) | Not stated; 172 clusters summed over tissues |
| Morabito 2021 | LSI after MNN correction | Leiden (monocle3) | Not stated |
| Zhang 2022 | Weighted multimodal neighbor graph | Seurat/Signac | Not stated |
| hECA v2.0 annotation | Per dataset, EpiScanpy | Leiden | Not stated |

The explicit values, 1.0 and 1.5, come from single studies of 8,500 to 91,500 cells. The number
of modularity clusters at a fixed resolution grows with the graph, and each CARVE resample
clusters about 430,000 cells, so those values should give fine partitions here and organ level
needs far smaller ones. On Cusanovich's 50-dimensional LSI at 50,164 cells, Louvain with 15
neighbors gave 13 clusters at 0.2, 20 at 1.0 and 34 at 3.0.

### Provisional grid

Fifteen log-spaced values from 0.005 to 3.0, rounded to two significant figures as for
Cusanovich:

0.005, 0.0079, 0.012, 0.02, 0.031, 0.049, 0.078, 0.12, 0.19, 0.31, 0.48, 0.76, 1.2, 1.9, 3.0

It places the published 0.4 to 1.5 in its upper middle. Both settings share one grid, as one
CARVE run requires. Cost depends on the number of values, not on the span.

### Calibration rule

Calibration scans 40 log-spaced values from 0.001 to 10 for each setting and records the
cluster count at each. The targets come from the data: the lower target is the number of
distinct organs (5), the upper target twice the number of distinct cell types (108).

- For setting s, lo_s is the smallest scan value giving at least the lower target, and hi_s is
  the largest scan value giving at most the upper target.
- The grid runs from min over s of lo_s to max over s of hi_s, so both settings cover organ level
  at the coarse end and about 108 clusters at the fine end.
- Fifteen log-spaced values between those endpoints, rounded to two significant figures.
- Calibration fails with a message naming the setting if a setting never reaches the lower
  target within the scan, exceeds the upper target at 0.001, if the endpoints cross, or if
  rounding produces duplicate values.

The author reviews the proposal and commits the grid to `STUDIES` before the fit. This is the
design's one human gate.

## Stages

`python -m benchmarks.heca {embed,calibrate,fit,status} --run-dir PATH`. The CLI is
`src/benchmarks/heca.py`; the stages live in `src/benchmarks/_heca_stages.py`, and the timing
instrumentation in `src/benchmarks/_timing.py`. A stage refuses to overwrite its own outputs
without `--force`.

Run directory layout:

```
<run-dir>/
  env.json            commit, Python and package versions, host, physical and logical cores,
                      node memory, SLURM job id
  embed.json
  calibration.csv     one row per setting and scan value
  calibration.json    proposed grid, component times, peak memory, projection
  fit/
    <carve_cache_path file name>
    timings/<host>-<pid>.csv
    memory.csv
    runtime.json
    sacct.txt         written by the sbatch script after the fit step
```

Every stage writes `env.json` if it does not exist yet, and appends its SLURM job id.

### embed

Calls `load_study(STUDIES["heca"])`. The loader's pooled path writes its embedding cache into
`data/hECA/`. `embed.json` records cells before and after the `Unclassified` drop, peak counts
(total, open, selected), wall-clock time and the process's peak resident memory. Preprocessing
cost is reported separately from CARVE's.

### calibrate

1. Load the embedding. Draw one training split and its complement with seed 42, at the sizes a
   CARVE resample uses (`subsample_ratio` 0.618: 429,698 and 265,606 cells; the stage computes
   them with CARVE's own rounding).
2. Run single-threaded throughout. The sbatch script pins OMP and BLAS threads to 1, and the
   forest gets `n_jobs=1`, because each worker in the fit runs that way.
3. For each setting, build the graph on the training split and on its complement with
   `carve.cluster.build_knn_graph`, timing the neighbor search and graph assembly separately.
4. On the training-split graph, run Leiden at the 40 scan values using the same partition call
   and arguments as `LeidenClustering.fit`. Record time, cluster count, and ARI against organ and
   against cell type.
5. Fit the default classifier, built by `default_generalizability_classifier` with 100 trees,
   on the training split, using setting one's labels at the scan value nearest the provisional
   grid's geometric middle. Predict the complement. Time both.
6. Record the process's peak resident memory after loading the embedding and at the end. The end
   value bounds what one worker needs.
7. Propose the grid by the calibration rule.
8. Project the fit's runtime. For setting s, the single-threaded cost of one configuration and
   one resample is

   c_s = 2 g_s(train) + g_s(test) + (2 + n_test / n_train) L_s + F + P

   where g is neighbor search plus graph assembly, L_s is setting s's mean Leiden time over the
   scan values inside the proposed grid, F and P are forest fit and predict. Resamples run in
   rounds of n_jobs, so the projected wall-clock is

   R x sum over s of ceil(B / n_jobs) x c_s

   with R the number of resolutions and B the number of resamples. `calibration.json` lists
   total core-hours and the wall-clock for n_jobs in {16, 24, 32, 48, 64, 128}. The wall-clock is
   a lower bound: it leaves out CARVE's own overhead and contention between workers for memory
   bandwidth (on the author's Mac, 11 workers gave 5.2 times one fit's throughput).

Gate: if the projection at the planned node size exceeds 8 days, the author decides before the
fit runs. Fewer resamples (100 to 50) is the first lever, fewer resolutions the second.

### fit

1. Runs the grid currently in `STUDIES` and records it in `runtime.json`.
2. Sets `n_jobs` to the smaller of the node's physical core count and
   floor(0.9 x node memory / the per-worker peak from `calibration.json`). Longleaf has
   hyperthreading enabled, and one worker per hardware thread would distort the timing.
   Physical cores come from `psutil.cpu_count(logical=False)`. psutil is installed but
   undeclared, so it is added to the `benchmarks` extra.
3. Builds instrumented grids: the study's resolution grids with `benchmarks._timing`'s Leiden
   subclass in place of `carve.cluster.LeidenClustering`. The classifier is the timing forest
   with the default's exact settings: 100 trees, `max_depth` equal to the feature count,
   `max_features` the integer square root of it.
4. Fits through `fit_or_load_carve`, which gains a `classifier` passthrough. The cache path is
   computed by `carve_cache_path` from the study's plain grids, never from the instrumented ones.
   The cache key hashes the grids' repr, which names the estimator class's module, so hashing the
   instrumented grids would produce a file name the notebook cannot find.
5. Sets `CARVE_TIMING_DIR` to `fit/timings/` before the fit, so loky workers inherit it.
6. Samples the resident memory of the fit process and all its children every 60 seconds with
   psutil and appends it to `memory.csv`. This is the memory curve and a cross-check on SLURM's
   accounting, whose configuration on Longleaf is unknown.
7. `runtime.json` records start and end timestamps, wall-clock time measured with
   `perf_counter`, n_jobs, physical and logical cores, node memory, grid, resamples, seed and
   package versions.

### status

Reads `fit/timings/` and `calibration.json` while the fit runs. It reports:

- configurations completed (a configuration is complete when it has 3B Leiden rows),
- mean wall-clock time per completed configuration and the projected finish,
- the median cluster count per resolution observed so far.

Configurations run in grid order: the 15-neighbor setting first, then the 50-neighbor setting,
each coarsest first (`ParameterGrid` varies `resolution`, the alphabetically last key, fastest).
The real time per configuration is therefore known within hours. Early-abort rule: cancel and
revisit if the projected finish exceeds 10 days. Cluster counts are informational; calibration
has already placed the grid, and the coarser setting may legitimately give fewer than 5
clusters at the grid's low end.

## Instrumentation

`benchmarks._timing` defines:

- A subclass of `carve.cluster.LeidenClustering`, also named `LeidenClustering`, so the
  `estimator` and method labels in `estimator_results_` are unchanged. Its `fit` calls
  `super().fit` and does not reimplement it. While `super().fit` runs, it temporarily replaces
  `carve.cluster.kneighbors_graph` and `carve.cluster.build_knn_graph` with timing wrappers.
  Graph assembly is `build_knn_graph` time minus neighbor-search time; Leiden is the fit's total
  minus `build_knn_graph` time. One row per fit: setting, resolution, sample count, cluster
  count, neighbor-search, graph-assembly and Leiden seconds.
- `TimedRandomForestClassifier`, a `RandomForestClassifier` subclass without its own
  `__init__`, so `clone` and `get_params` behave as for the parent. One row per `fit` and per
  `predict` call: sample count and seconds.

Rows go to `<CARVE_TIMING_DIR>/<host>-<pid>.csv`, flushed per row. With the variable unset,
both classes behave exactly as their parents and write nothing. The replacement of module
attributes is process-local, and a loky worker runs one task at a time.

The breakdown in the report is component time summed across workers, divided by the fit's total
CPU time. The remainder is CARVE's own overhead: ARI scoring, consensus accumulation and
scheduling.

## Running on Longleaf

The runbook is `slurm/heca/README.md`, beside `embed.sbatch`, `calibrate.sbatch` and
`fit.sbatch`. The author runs them. Longleaf facts as documented on 2026-10-01: SLURM, a
`general` partition with an 11-day job limit, a `datamover` partition for transfers, a 50 GB
home quota, and 8 TB of `/work/users` storage. Current node sizes were not confirmed.

Setup:

1. Request a Longleaf account. Push the branch so the cluster can clone it.
2. Clone into `/work/users/...`, not home, since `code/data/` holds about 42 GB of h5ad files
   plus caches.
3. Install `uv` in home and build a Python 3.13 virtual environment (CI's version) with the
   extras the loader, Leiden and the stages need.
4. In a `datamover` job, download the five organ zips from Zenodo record 15627886 into
   `code/data/hECA/` and unzip them. Check that each h5ad opens with h5py before going further;
   a truncated extraction has happened before.

Jobs:

| Stage | Request | Submission |
|---|---|---|
| embed | 8 cores, at least 200 GB, 24 h | By hand |
| calibrate | 4 cores, 64 GB, 24 h | Chained on embed with `--dependency=afterok` |
| fit | One whole node (`--exclusive`, all memory), up to 11 days, from the projection | By hand, after the grid is committed and pushed |
| status | Login node or a short `interact` session | By hand, as often as wanted |

The fit runs as an `srun` step. When the step ends, the sbatch script writes
`sacct -j <job>.0 --format=JobID,Elapsed,TotalCPU,MaxRSS,AveRSS,NCPUS -P` to `fit/sacct.txt`.
An exclusive node keeps the timing clean: no other job shares its memory bandwidth.

Copy back with rsync: the run directory to the gitignored `results/runs/heca/<date>-<commit>/`,
and the embedding cache (about 280 MB) to `data/hECA/` under its own name, so `load_study`
finds it locally.

The fit saves only at the end. A killed job (time limit, node failure) is rerun from scratch.
The projection gate and the status check exist to keep that from happening.

## Local reporting

`hECA.ipynb` is rewritten to read a run directory. It fits nothing.

0. Configuration from `STUDIES` and the run directory. The reference UMAP colored by organ, as
   the existing cell draws it on a row subsample.
1. Calibration: cluster count and ARI against organ and cell type over the scan, both settings,
   with the committed grid marked.
2. CARVE results: stability and generalizability over resolution for both settings; the
   selected configuration (stability, 1se) and its cluster count; ARI against organ and against
   cell type; study composition per cluster.
3. Runtime: a table of wall-clock time, core-hours, n_jobs, node and peak memory, with the
   embed stage's cost on its own row; the component breakdown per setting; projected against
   actual time; the memory curve.

New figures and tables are functions in `benchmarks.figures` and `benchmarks.tables`, following
the other case-study notebooks; `test_notebooks` keeps its hECA pins (configuration read from
`STUDIES`, one UMAP drawn throughout). Every number comes from an artifact.

hECA stays in the supplement. The S Text needs: cells, organs, cell types and studies; the
embed stage's time and memory; the fit's wall-clock time, core-hours, n_jobs, node and peak
memory; the component shares; the selected setting, resolution and cluster count with its ARI
against organ and cell type; projected against actual runtime. The author writes the text.

## Testing

Tests mirror modules: `tests/benchmarks/test_heca.py`, `test_heca_stages.py`, `test_timing.py`,
plus updates to `test_studies.py`, `test_estimators.py`, `test_benchmarks_types.py` and
`test_notebooks.py`.

- Instrumentation: on toy data, an instrumented fit and a plain fit give identical
  `estimator_results_` and labels. Timing rows arrive from real loky workers at `n_jobs=2`.
  Estimator names in `estimator_results_` are unchanged. With `CARVE_TIMING_DIR` unset, nothing is
  written. Each assertion is mutation-verified.
- Cache path: the fit stage's cache path equals `carve_cache_path` on the plain grids.
- `EstimatorSpec.params`: overrides reach the grid, an unknown name raises, and the two Leiden
  settings are distinct methods in `estimator_results_`.
- Grid rule: synthetic calibration tables give the expected endpoints and values; each failure
  case raises with the setting named.
- Projection: matches a hand-computed case, including the rounding up of resample rounds.
- n_jobs rule: memory-capped and core-capped cases.
- Status: on synthetic timing files, completed configurations, projected finish and cluster
  counts match hand-computed values.
- End to end: embed, calibrate and fit run on a tiny synthetic hECA-style h5ad set with two
  resamples and two resolutions, and the notebook's loaders read the result.
- Removed functions take their tests with them.

The sbatch scripts are not executed in CI.

## Risks and caveats

- Batch effects. No source clustered without correction. Clusters may follow study, notably
  fetal against adult brain, which lowers agreement with organ labels without affecting
  runtime. Reported through study composition per cluster.
- Rare types. Modularity's resolution limit on a graph this size means the 8 types under 1,000
  cells probably will not separate; CATlas and Hocker switched to CPM for that reason.
- Runtime is unknown until calibration, and the projection is optimistic. The status check
  catches the difference within hours.
- No checkpointing. A killed fit is lost.
- Longleaf's node sizes and SLURM accounting configuration are unconfirmed; the memory curve
  covers the second.

## Out of scope

- The R port.
- Checkpointing inside a CARVE fit.
- The CPM objective, which needs its own grid and a separate run.
- Harmony or other batch correction.
- Reusing one neighbor graph across resolutions. It would remove most graph builds, which likely
  dominate the cost, but it is a carve change and needs its own design if the gate fails.
