"""Stages of the full-scale hECA run on Longleaf: embed, fit and status.

Each stage reads and writes one run directory, so the stages run as separate
SLURM jobs and their outputs copy back as a unit. The calibration stage is
in _heca_calibration, the notebook-side reading in _heca_report. Layout:

    <run-dir>/env.json          provenance, then one record per stage run
    <run-dir>/embed.json
    <run-dir>/calibration.csv   one row per setting and scanned resolution
    <run-dir>/calibration.json
    <run-dir>/fit/              the CARVE cache and its fingerprint,
                                timings/, memory.csv, started.json,
                                runtime.json, sacct.txt

Design: docs/superpowers/specs/2026-10-01-heca-longleaf-runtime-design.md.
"""

import csv
import json
import os
import resource
import shutil
import socket
import threading
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from carve import CARVE
from carve._utils import scale_neighbor_count
from carve.cluster import LeidenClustering

from ._artifacts import peak_rss_bytes, provenance
from ._studies import (
    STUDIES,
    carve_cache_path,
    fit_or_load_carve,
    load_study,
    study_resolution_grids,
)
from ._timing import instrumented_grids, read_timings, timed_forest, timing_directory
from ._types import Study

#: The seed every stage draws with, as for the other case studies.
SEED = 42

#: CARVE's own defaults, read rather than restated, so the calibration split
#: and forest are the ones the fit uses.
SUBSAMPLE_RATIO: float = next(
    field.default for field in fields(CARVE) if field.name == "subsample_ratio"
)
CARVE_N_TREES: int = next(
    field.default for field in fields(CARVE) if field.name == "n_trees"
)

#: Versions recorded beside provenance()'s, for the packages this run adds.
EXTRA_PACKAGES: tuple[str, ...] = (
    "igraph",
    "leidenalg",
    "psutil",
    "scanpy",
    "anndata",
    "threadpoolctl",
)

#: Thread-count variables joblib's loky backend keeps from the parent rather
#: than setting per worker. fit.sbatch unsets them; run_fit records them.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


class StageOutputExists(FileExistsError):
    """A stage's output is already in the run directory."""


@dataclass(frozen=True)
class Setting:
    """One graph setting of the sweep: a label and its configured neighbors."""

    label: str
    n_neighbors: int


def resolve_study(study: Study | None) -> Study:
    """The study a stage runs: STUDIES["heca"] unless a test passes another."""
    return STUDIES["heca"] if study is None else study


def sweep_settings(study: Study) -> list[Setting]:
    """The study's settings in grid order, one per resolution grid."""
    settings = []
    for _, grid in study_resolution_grids(study):
        (n_neighbors,) = grid["n_neighbors"]
        settings.append(
            Setting(label=f"leiden_{n_neighbors}", n_neighbors=int(n_neighbors))
        )
    return settings


def split_sizes(n_cells: int) -> tuple[int, int]:
    """Rows in one resample's training split and in its complement.

    The arithmetic of carve._utils.split_subsample_indices, which a test
    holds this to.
    """
    n_train = int(np.float64(SUBSAMPLE_RATIO * n_cells))
    return n_train, n_cells - n_train


def scaled_neighbors(setting: Setting, *, n_fit: int, n_full: int) -> int:
    """The neighbor count CARVE fits setting with on n_fit of n_full rows."""
    params = scale_neighbor_count(
        LeidenClustering,
        {"n_neighbors": setting.n_neighbors},
        n_fit=n_fit,
        n_full=n_full,
    )
    return int(params["n_neighbors"])


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def guard_output(path: Path, *, force: bool) -> None:
    """Refuse to overwrite a stage's output unless forced."""
    if Path(path).exists() and not force:
        raise StageOutputExists(f"{path} exists. Pass --force to overwrite it.")


#: Linux's CPU topology, read to count the physical cores a job runs on.
CPU_TOPOLOGY = Path("/sys/devices/system/cpu")


def physical_cores() -> int:
    """Physical cores this process may run on.

    A SLURM job that does not hold a whole node runs on part of it, so on
    Linux the cores are counted from the process's CPU affinity: hardware
    threads that list the same siblings share one core. Elsewhere, or where
    the topology cannot be read, the machine's physical cores, or logical
    ones where psutil cannot tell.
    """
    import psutil

    try:
        cores = {
            (CPU_TOPOLOGY / f"cpu{cpu}" / "topology" / "thread_siblings_list")
            .read_text()
            .strip()
            for cpu in os.sched_getaffinity(0)
        }
    except (AttributeError, OSError):
        cores = set()
    return len(cores) or int(psutil.cpu_count(logical=False) or os.cpu_count() or 1)


def node_memory_bytes() -> int:
    """Memory the job may use: SLURM's --mem, in megabytes, else the machine's."""
    import psutil

    allocated = os.environ.get("SLURM_MEM_PER_NODE", "")
    if allocated.isdigit() and int(allocated) > 0:
        return int(allocated) * 1024**2
    return int(psutil.virtual_memory().total)


def _extra_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in EXTRA_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def record_stage(run_dir: Path, stage: str) -> dict[str, Any]:
    """Append this stage's machine record to env.json, creating the file first.

    Provenance (commit, package versions, platform) is written once, by the
    first stage to run. Each stage then adds its host, SLURM job id, core
    counts and memory, since the stages run on different nodes. env.json
    records every stage run that passed its overwrite guard, including runs
    that later failed; a stage's own output file (embed.json,
    calibration.json, fit/runtime.json) shows that it completed.
    """
    import psutil

    path = Path(run_dir) / "env.json"
    record = (
        read_json(path)
        if path.exists()
        else {"provenance": provenance(), "packages": _extra_versions(), "stages": []}
    )
    record["stages"].append(
        {
            "stage": stage,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "physical_cores": physical_cores(),
            "logical_cores": psutil.cpu_count(logical=True),
            "memory_bytes": node_memory_bytes(),
        }
    )
    write_json(path, record)
    return record


def run_embed(
    run_dir: Path, *, study: Study | None = None, force: bool = False
) -> dict[str, Any]:
    """Compute or load the pooled embedding and record what it cost.

    The loader caches the embedding under data/hECA/, so the later stages
    load it in seconds. wall_clock_s is the preprocessing cost only when
    cached is False; a run whose cache already existed records a load.
    """
    study = resolve_study(study)
    run_dir = Path(run_dir)
    out = run_dir / "embed.json"
    guard_output(out, force=force)
    record_stage(run_dir, "embed")

    started = time.perf_counter()
    X, _, meta = load_study(study)
    elapsed = time.perf_counter() - started

    obs = meta["obs"]
    record = {
        "cached": bool(meta["cached"]),
        "wall_clock_s": elapsed,
        "peak_rss_bytes": int(peak_rss_bytes()),
        "organs": list(meta["organs"]),
        "n_cells_full": int(meta["n_cells_full"]),
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_peaks_full": int(meta["n_peaks_full"]),
        "n_peaks_open": int(meta["n_peaks_open"]),
        "n_peaks_selected": int(meta["n_peaks_selected"]),
        "n_organs": int(obs["organ"].nunique()),
        "n_cell_types": int(obs["cell_type"].nunique()),
        "n_studies": int(obs["study_id"].nunique()),
    }
    write_json(out, record)
    return record


#: Share of node memory the fit's workers may plan to fill.
MEMORY_FRACTION = 0.9


def choose_n_jobs(
    *,
    cores: int,
    memory_bytes: int,
    worker_peak_bytes: int,
    memory_fraction: float = MEMORY_FRACTION,
) -> int:
    """Workers for the fit: one per physical core, unless memory allows fewer."""
    by_memory = int(memory_fraction * memory_bytes // max(int(worker_peak_bytes), 1))
    return max(1, min(int(cores), by_memory))


class MemorySampler:
    """Sample the resident memory of this process and of its children.

    One row per interval: seconds since entry, this process's RSS, and the
    summed RSS of every descendant (the loky workers). A row is also written
    on entry and on exit, so even a short block leaves two. This is the
    fit's memory curve, and a cross-check on SLURM's accounting.

    Summed RSS counts pages shared between workers, notably joblib's
    memory-mapped copy of X, once per worker, so the curve overstates the
    footprint; SLURM's MaxRSS is the primary figure. A sample that fails
    warns once and does not stop the later ones.
    """

    COLUMNS = ("elapsed_s", "own_rss_bytes", "children_rss_bytes", "n_children")

    def __init__(self, path: Path, *, interval: float = 60.0) -> None:
        self.path = Path(path)
        self.interval = float(interval)

    def __enter__(self) -> "MemorySampler":
        import psutil

        self._process = psutil.Process()
        self._started = time.perf_counter()
        self._stop = threading.Event()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as handle:
            csv.writer(handle).writerow(self.COLUMNS)
        self._sample()
        self._thread = threading.Thread(
            target=self._run, name="memory-sampler", daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        warned = False
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception as error:
                # One failed sample must not end the curve for the rest of a
                # multi-day fit.
                if not warned:
                    warnings.warn(
                        f"Memory sampling failed ({type(error).__name__}: "
                        f"{error}); sampling continues on later intervals.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned = True

    def _sample(self) -> None:
        import psutil

        own = self._process.memory_info().rss
        children_rss, n_children = 0, 0
        for child in self._process.children(recursive=True):
            try:
                children_rss += child.memory_info().rss
                n_children += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        elapsed = time.perf_counter() - self._started
        with self.path.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [f"{elapsed:.3f}", own, children_rss, n_children]
            )

    def __exit__(self, *exc_info: object) -> bool:
        self._stop.set()
        self._thread.join()
        self._sample()
        return False


def _cpu_seconds() -> float:
    """User plus system CPU seconds of this process and its reaped children.

    RUSAGE_CHILDREN counts only children that have exited and been waited
    for, so a worker's time appears in it once its pool has shut down.
    """
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        usage = resource.getrusage(who)
        total += usage.ru_utime + usage.ru_stime
    return total


def run_fit(
    run_dir: Path,
    *,
    study: Study | None = None,
    n_jobs: int | None = None,
    force: bool = False,
    sample_interval: float = 60.0,
) -> dict[str, Any]:
    """Fit CARVE on every cell with the timing classes in place.

    n_jobs defaults to one worker per physical core the job holds, capped
    by the job's memory over the per-worker peak calibration measured.
    LOKY_MAX_CPU_COUNT is set to the physical core count for the fit, so
    CARVE's core budget sees physical cores rather than hyperthreads: with
    one worker per core, each worker's forest and BLAS run single-threaded,
    and the spare cores of a memory-capped run go to forest threads.

    The cache path comes from the study's plain grids, not the instrumented
    ones: the cache key hashes the grids' repr, which names the estimator
    class's module, and the notebook looks the fit up by the plain grids.

    runtime.json records cpu_s, the fit's user plus system CPU time from
    getrusage, read before the fit and after the timing block has shut the
    worker pool down. RUSAGE_CHILDREN counts only reaped processes, so the
    shutdown is what brings the workers' time in. The report prefers
    SLURM's TotalCPU from sacct.txt and falls back to cpu_s. thread_env
    records the thread-count variables the fit inherited (None when unset),
    which joblib would pass on to every worker.
    """
    study = resolve_study(study)
    run_dir = Path(run_dir)
    fit_dir = run_dir / "fit"
    grids = study_resolution_grids(study)
    cache_path = carve_cache_path(
        study, root=fit_dir, model_grids=grids, n_resamples=study.n_resamples
    )
    guard_output(cache_path, force=force)
    guard_output(fit_dir / "runtime.json", force=force)
    # The files the fit and fit.sbatch write last: a rerun that fails partway
    # must not leave the previous run's beside its own started.json.
    (fit_dir / "runtime.json").unlink(missing_ok=True)
    (fit_dir / "sacct.txt").unlink(missing_ok=True)
    record_stage(run_dir, "fit")

    X, y, _ = load_study(study)
    cores = physical_cores()
    if n_jobs is None:
        calibration = read_json(run_dir / "calibration.json")
        n_jobs = choose_n_jobs(
            cores=cores,
            memory_bytes=node_memory_bytes(),
            worker_peak_bytes=int(calibration["worker_peak_bytes"]),
        )

    # A forced rerun starts its timing rows afresh; memory.csv is rewritten.
    shutil.rmtree(fit_dir / "timings", ignore_errors=True)
    started_at = time.time()
    write_json(
        fit_dir / "started.json",
        {
            "started_at": started_at,
            "n_cells": int(X.shape[0]),
            "n_jobs": int(n_jobs),
            "resolutions": [float(r) for r in study.resolutions],
            "n_resamples": int(study.n_resamples),
            "settings": [setting.label for setting in sweep_settings(study)],
        },
    )

    thread_env = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
    previous_cap = os.environ.get("LOKY_MAX_CPU_COUNT")
    os.environ["LOKY_MAX_CPU_COUNT"] = str(cores)
    try:
        with (
            timing_directory(fit_dir / "timings"),
            MemorySampler(fit_dir / "memory.csv", interval=sample_interval),
        ):
            cpu_before = _cpu_seconds()
            started = time.perf_counter()
            fit_or_load_carve(
                X,
                y,
                cache_path=cache_path,
                model_grids=instrumented_grids(grids),
                n_resamples=study.n_resamples,
                n_jobs=int(n_jobs),
                random_state=SEED,
                force=force,
                consensus_anchors=study.consensus_anchors,
                classifier=timed_forest(
                    n_features=int(X.shape[1]), n_trees=CARVE_N_TREES
                ),
                # Settings at the start, then each configuration's stability
                # and generalizability in the job log as it completes.
                verbose=2,
            )
            wall_clock_s = time.perf_counter() - started
        # timing_directory has shut the pool down on exit, so the workers
        # are reaped and counted in RUSAGE_CHILDREN.
        cpu_s = _cpu_seconds() - cpu_before
    finally:
        if previous_cap is None:
            os.environ.pop("LOKY_MAX_CPU_COUNT", None)
        else:
            os.environ["LOKY_MAX_CPU_COUNT"] = previous_cap

    record = {
        "started_at": started_at,
        "finished_at": time.time(),
        "wall_clock_s": wall_clock_s,
        "cpu_s": cpu_s,
        "n_jobs": int(n_jobs),
        "physical_cores": cores,
        "logical_cores": os.cpu_count(),
        "memory_bytes": node_memory_bytes(),
        "loky_max_cpu_count": cores,
        "thread_env": thread_env,
        "resolutions": [float(r) for r in study.resolutions],
        "n_resamples": int(study.n_resamples),
        "seed": SEED,
        "n_cells": int(X.shape[0]),
        "cache_file": cache_path.name,
        "provenance": provenance(),
    }
    write_json(fit_dir / "runtime.json", record)
    return record


#: The early-abort rule: cancel and revisit if the projected total passes this.
ABORT_AFTER_S = 10 * 86_400


def label_settings(leiden: pd.DataFrame, study: Study, *, n_cells: int) -> pd.DataFrame:
    """Leiden timing rows with the setting each came from.

    A row records the neighbor count CARVE actually fitted with, which it
    scales to each subset's size; this maps (scaled count, subset size)
    back to the setting.
    """
    n_train, n_test = split_sizes(n_cells)
    lookup = {}
    for setting in sweep_settings(study):
        for n_fit in (n_train, n_test):
            key = (scaled_neighbors(setting, n_fit=n_fit, n_full=n_cells), n_fit)
            lookup[key] = setting.label
    settings = [
        lookup.get((int(k), int(n)))
        for k, n in zip(leiden["n_neighbors"], leiden["n_samples"], strict=True)
    ]
    return leiden.assign(setting=settings)


def fit_status(
    run_dir: Path, *, study: Study | None = None, now: float | None = None
) -> dict[str, Any]:
    """Progress of a running or finished fit, from its live timing rows.

    Configurations run one after another, each across every resample, so
    the mean time of the completed ones projects the total. A configuration
    is complete once it has 3 x n_resamples Leiden rows: both training
    splits and the complement, per resample.
    """
    study = resolve_study(study)
    fit_dir = Path(run_dir) / "fit"
    started = read_json(fit_dir / "started.json")
    n_cells = int(started["n_cells"])
    n_resamples = int(started["n_resamples"])
    n_train, _ = split_sizes(n_cells)
    leiden = label_settings(
        read_timings(fit_dir / "timings")["leiden"], study, n_cells=n_cells
    )

    total = len(sweep_settings(study)) * len(started["resolutions"])
    counts = leiden.groupby(["setting", "resolution"]).size()
    completed = int((counts >= 3 * n_resamples).sum())

    now = time.time() if now is None else float(now)
    elapsed = now - float(started["started_at"])
    per_configuration = elapsed / completed if completed else None
    projected_total = None if per_configuration is None else per_configuration * total
    training = leiden[leiden["n_samples"] == n_train]
    clusters = (
        pd.DataFrame()
        if training.empty
        else training.groupby(["resolution", "setting"])["n_clusters"]
        .median()
        .unstack("setting")
    )
    return {
        "completed": completed,
        "total": total,
        "elapsed_s": elapsed,
        "seconds_per_configuration": per_configuration,
        "projected_total_s": projected_total,
        "projected_finish": None
        if projected_total is None
        else float(started["started_at"]) + projected_total,
        "abort": projected_total is not None and projected_total > ABORT_AFTER_S,
        "clusters": clusters,
    }


def format_status(status: Mapping[str, Any]) -> str:
    """The lines python -m benchmarks.heca status prints."""
    lines = [
        f"Configurations complete: {status['completed']} of {status['total']}",
        f"Elapsed: {status['elapsed_s'] / 3600:.1f} h",
    ]
    if status["seconds_per_configuration"] is None:
        lines.append("No configuration has completed yet, so there is no projection.")
    else:
        finish = datetime.fromtimestamp(status["projected_finish"], UTC)
        lines.append(
            f"Per configuration: {status['seconds_per_configuration'] / 3600:.2f} h"
        )
        lines.append(
            f"Projected total: {status['projected_total_s'] / 3600:.1f} h, "
            f"finishing {finish:%Y-%m-%d %H:%M} UTC"
        )
    verdict = (
        "triggered: cancel the fit and revisit the grid or the resample count"
        if status["abort"]
        else "not triggered"
    )
    lines.append(
        f"Abort rule (projected total over {ABORT_AFTER_S / 86_400:.0f} days): "
        f"{verdict}"
    )
    clusters = status["clusters"]
    if not clusters.empty:
        lines.append("Median clusters on the training split, by resolution:")
        lines.append(clusters.to_string())
    return "\n".join(lines)
