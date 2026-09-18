# Forest threads oversubscribe the cores when benchmark workers run CARVE fits concurrently

Date: 2026-09-17
Found while: designing the rho/B ablation
(`docs/superpowers/specs/2026-09-16-rho-b-ablation-design.md`, section 2)
Status: fixed in the rho-b-ablation branch (cpu_cap in benchmarks/_run.py, Task 3 of docs/superpowers/plans/2026-09-17-rho-b-ablation.md)

## Issue

`benchmarks._run.run_scenario` runs cells concurrently in a loky pool and gives each CARVE fit
`n_jobs=1`. Under CARVE's core budget (`carve._utils.resolve_core_budget`), `n_jobs=1` means one
resample worker whose classifier gets `cpu_count() // 1` threads, that is, every core. Inside a loky
worker, joblib caps OpenMP and BLAS at one thread but `joblib.cpu_count()` still reports the whole
machine, so every concurrent fit builds its random forest with all cores. At `--n-jobs -1` on the
11-core Mac that is up to 121 forest threads on 11 cores.

## Evidence

Probed on 2026-09-17 with joblib 1.5.3 inside an 11-worker pool: each worker saw
`OMP_NUM_THREADS=1`, `cpu_count() == 11` and `resolve_core_budget(1, n_resamples=100) == (1, 11)`.
Setting `LOKY_MAX_CPU_COUNT=1` inside the worker made the same call return `(1, 1)`; joblib's
`cpu_count` takes the minimum of the system count and that variable. The core budget work on
2026-09-11 measured eight forest fits at 2.9 s single-threaded against 3.7 s with every forest on
all cores.

## Effect

Speed only. `estimator_results_` was identical across thread counts on 2026-09-11, so no published
benchmark number is affected. Scenario runs at `--n-jobs` above 1 are slower than they need to be.
The scaling scenarios' timed fits (`t_default_s` and the two mode timings) were produced under
this contention and describe it.

## Fix

In the shared fit code, cap the CPU count the worker reports to `max(1, cores // workers)` for the
duration of each fit, through `LOKY_MAX_CPU_COUNT`, and restore it afterwards. No change to
`src/carve/`. A test with a recording classifier in a real two-worker pool pins the thread count.
