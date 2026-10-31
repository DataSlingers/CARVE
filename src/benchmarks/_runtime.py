"""Runtime and memory of a study's CARVE fit as cells and workers grow.

The supplementary runtime text and table report how one CARVE fit with a
case study's configuration scales with the number of cells, with and without
randomized preprocessing, and with the number of workers. Each fit runs in a Python
process of its own (``python -m benchmarks._runtime``), so the peak memory
recorded belongs to that fit alone. With one worker every resample runs in
that process, and timers split the wall-clock time into the embedding, the
clustering estimator's fit (its neighbor graph included) and the random
forest. With more workers those calls run in child processes, and the split
is not recorded.

Measurements are stored one row per fit. run_measurements reads the stored
rows back and runs only the fits the file lacks, so a finished set is not
repeated and an interrupted one resumes. Each row keeps the pipelines its
resamples drew and its number of configurations, and reading warns when
either differs from what the study would run now. Times and memory belong to
the machine that measured them.
"""

import functools
import json
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.base import ClusterMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import carve._runner as _runner
from carve._pipeline import allocate_pipelines

from ._artifacts import provenance
from ._estimators import RESOLUTION_ESTIMATORS
from ._preprocessing import resolve_preprocessing
from ._studies import fit_carve, study_model_grids, study_resolution_grids
from ._types import Study

#: Cell counts of the size ladder. The data set's own size is always the last
#: rung; rungs at or above it are skipped.
LADDER_CELLS: tuple[int, ...] = (5_000, 10_000, 25_000, 50_000)

#: Resamples per ladder fit. A fit's cost is linear in its resample count, so
#: four give the time per resample, and projections scale it.
LADDER_RESAMPLES = 4

#: The resample count the ladder table projects each fit to, CARVE's default B.
PROJECTED_RESAMPLES = 100

#: The parallel check: one size, more resamples than the ladder so that every
#: worker count gets at least one, and the worker counts compared.
PARALLEL_CELLS = 10_000
PARALLEL_RESAMPLES = 8
PARALLEL_N_JOBS: tuple[int, ...] = (1, 4, 8)

#: "fixed" clusters the data as given; "randomized" draws one of the study's
#: preprocessing pipelines per resample and fits it on each subsample.
SETTINGS: tuple[str, ...] = ("fixed", "randomized")

COMPONENTS: tuple[str, ...] = (
    "embedding",
    "clustering",
    "forest_fit",
    "forest_predict",
)

COLUMNS: tuple[str, ...] = (
    "label",
    "setting",
    "n_cells",
    "n_resamples",
    "n_jobs",
    "random_state",
    "estimator",
    "n_configs",
    "pipelines",
    "wall_s",
    "embedding_s",
    "clustering_s",
    "forest_fit_s",
    "forest_predict_s",
    "peak_memory_gb",
    "n_anchors",
    "consensus_retained_gb",
    "started",
    "carve_commit",
    "python",
    "sklearn",
    "machine",
    "cpu_count",
    "ram_gb",
)

_KEY_COLUMNS = ("label", "setting", "n_cells", "n_resamples", "n_jobs", "random_state")

_TEXT_COLUMNS = (
    "label",
    "setting",
    "estimator",
    "pipelines",
    "started",
    "carve_commit",
    "python",
    "sklearn",
    "machine",
)

_SETTING_TITLES = {
    "fixed": "Data as given",
    "randomized": "Randomized preprocessing, refit per resample",
}

#: The clock every timer reads. A module attribute so a test can replace it.
_clock = time.perf_counter

_MISSING = object()


@dataclass(frozen=True)
class Measurement:
    """One fit to time: the comparison it belongs to and what it runs."""

    label: str
    setting: str
    n_cells: int
    n_resamples: int
    n_jobs: int


@dataclass(frozen=True)
class RuntimeSpec:
    """What a measured fit runs, read from a Study as its notebook's fit reads it."""

    estimator: str
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]]
    normalization_options: list[Any] = field(default_factory=list)
    dim_reduction_options: list[Any] = field(default_factory=list)
    consensus_anchors: int | None = None

    @classmethod
    def from_study(cls, study: Study) -> "RuntimeSpec":
        if study.estimator.name in RESOLUTION_ESTIMATORS:
            grids = study_resolution_grids(study)
        else:
            grids = study_model_grids(study)
        options = (
            resolve_preprocessing(study.preprocessing)
            if study.preprocessing is not None
            else {"normalization_options": [], "dim_reduction_options": []}
        )
        return cls(
            estimator=study.estimator.name,
            model_grids=grids,
            normalization_options=options["normalization_options"],
            dim_reduction_options=options["dim_reduction_options"],
            consensus_anchors=study.consensus_anchors,
        )

    @property
    def randomizable(self) -> bool:
        return bool(self.normalization_options and self.dim_reduction_options)

    @property
    def n_configs(self) -> int:
        return sum(len(ParameterGrid(grid)) for _, grid in self.model_grids)

    def pipelines(self, n_resamples: int, random_state: int) -> dict[str, int]:
        """Resamples per pipeline label, as a randomized fit allocates them."""
        if not self.randomizable:
            return {}
        specs = allocate_pipelines(
            self.normalization_options,
            self.dim_reduction_options,
            int(n_resamples),
            int(random_state),
        )
        return dict(sorted(Counter(spec.label for spec in specs).items()))


def measurement_plan(
    n_total: int,
    *,
    settings: Sequence[str] = SETTINGS,
    ladder_cells: Sequence[int] = LADDER_CELLS,
    ladder_resamples: int = LADDER_RESAMPLES,
    parallel_cells: int = PARALLEL_CELLS,
    parallel_resamples: int = PARALLEL_RESAMPLES,
    parallel_n_jobs: Sequence[int] = PARALLEL_N_JOBS,
) -> list[Measurement]:
    """The ladder for every setting, then the parallel check.

    The ladder runs one worker, so its timers split every fit. The parallel
    check runs the last setting, the costlier one when both are measured.
    """
    rungs = sorted({int(c) for c in ladder_cells if c < n_total} | {int(n_total)})
    plan = [
        Measurement("ladder", setting, cells, ladder_resamples, 1)
        for setting in settings
        for cells in rungs
    ]
    cells = min(int(parallel_cells), int(n_total))
    plan += [
        Measurement("parallel", settings[-1], cells, parallel_resamples, int(n_jobs))
        for n_jobs in parallel_n_jobs
    ]
    return plan


def subset_indices(n_total: int, n_cells: int, random_state: int) -> np.ndarray:
    """Sorted indices of a random subset, seeded by random_state and its size."""
    if n_cells >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(int(random_state) + int(n_cells))
    return np.sort(rng.choice(n_total, size=int(n_cells), replace=False))


class PeakMemory:
    """Peak resident memory of this process and its descendants, in bytes.

    Sampled every interval seconds on a background thread, and once more on
    exit. Worker processes are counted while they live.
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        process = psutil.Process()
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                # A worker can exit between listing and reading.
                pass
        self.peak = max(self.peak, total)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakMemory":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


@contextmanager
def component_timers(
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
) -> Iterator[dict[str, float]]:
    """Accumulate seconds per component while the block runs in this process.

    Wraps carve._runner.embed_resample (one resample's pipeline fits), each
    clustering estimator's fit and RandomForestClassifier's fit and predict,
    and restores them on exit. A call made inside a call to the same component
    counts once. Calls made in other processes are not seen.
    """
    times = dict.fromkeys(COMPONENTS, 0.0)
    depth = dict.fromkeys(COMPONENTS, 0)

    def timed(func: Any, key: str) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if depth[key]:
                return func(*args, **kwargs)
            depth[key] += 1
            start = _clock()
            try:
                return func(*args, **kwargs)
            finally:
                times[key] += _clock() - start
                depth[key] -= 1

        return wrapper

    estimators = dict.fromkeys(estimator for estimator, _ in model_grids)
    targets: list[tuple[Any, str, str]] = [(_runner, "embed_resample", "embedding")]
    targets += [(estimator, "fit", "clustering") for estimator in estimators]
    targets += [
        (RandomForestClassifier, "fit", "forest_fit"),
        (RandomForestClassifier, "predict", "forest_predict"),
    ]

    saved: list[tuple[Any, str, Any]] = []
    try:
        for owner, name, key in targets:
            # An inherited method is not in the owner's own namespace; on exit
            # it is deleted again rather than left behind as a copy.
            saved.append((owner, name, vars(owner).get(name, _MISSING)))
            setattr(owner, name, timed(getattr(owner, name), key))
        yield times
    finally:
        for owner, name, original in reversed(saved):
            if original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)


def _measure(
    payload: dict[str, Any], measurement: Measurement, random_state: int
) -> dict[str, Any]:
    """Time one fit in this process and return its row."""
    spec: RuntimeSpec = payload["spec"]
    X, y = payload["X"], payload["y"]
    idx = subset_indices(X.shape[0], measurement.n_cells, random_state)
    X = np.ascontiguousarray(X[idx])
    y = None if y is None else y[idx]

    randomized = measurement.setting == "randomized"
    options: dict[str, Any] = (
        {
            "normalization_options": spec.normalization_options,
            "dim_reduction_options": spec.dim_reduction_options,
        }
        if randomized
        else {}
    )
    timers = (
        component_timers(spec.model_grids)
        if measurement.n_jobs == 1
        else nullcontext(None)
    )

    started = datetime.now().isoformat(timespec="seconds")
    with timers as times, PeakMemory() as memory:
        start = _clock()
        carve = fit_carve(
            X,
            y,
            model_grids=spec.model_grids,
            n_resamples=measurement.n_resamples,
            n_jobs=measurement.n_jobs,
            random_state=random_state,
            consensus_anchors=spec.consensus_anchors,
            randomize_preprocessing=randomized,
            **options,
        )
        wall = _clock() - start

    retained = [
        matrix
        for matrix in (
            *(carve.consensus_matrices_ or []),
            *(carve.consensus_generalizability_matrices_ or []),
        )
        if matrix is not None
    ]
    pipelines = ""
    if randomized:
        counts = carve.preprocessing_results_.groupby("pipeline")["n_resamples"].max()
        pipelines = json.dumps(
            {str(k): int(v) for k, v in counts.items()}, sort_keys=True
        )

    record = provenance()
    return {
        **asdict(measurement),
        "random_state": int(random_state),
        "estimator": spec.estimator,
        "n_configs": int(len(carve.estimator_results_)),
        "pipelines": pipelines,
        "wall_s": wall,
        **{
            f"{key}_s": (times[key] if times is not None else float("nan"))
            for key in COMPONENTS
        },
        "peak_memory_gb": memory.peak / 1e9,
        "n_anchors": 0
        if carve.consensus_anchors_ is None
        else int(carve.consensus_anchors_.size),
        "consensus_retained_gb": sum(matrix.nbytes for matrix in retained) / 1e9,
        "started": started,
        "carve_commit": record["git_sha"],
        "python": record["platform"]["python"],
        "sklearn": sklearn.__version__,
        "machine": record["platform"]["machine"],
        "cpu_count": record["platform"]["cores"],
        "ram_gb": psutil.virtual_memory().total / 1e9,
    }


def _run_in_process(
    payload_path: Path, measurement: Measurement, random_state: int, out_path: Path
) -> dict[str, Any]:
    """Time one fit in a fresh Python process and return its row."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks._runtime",
            str(payload_path),
            json.dumps(asdict(measurement)),
            str(random_state),
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"The runtime fit {measurement} failed in its process:\n{result.stderr[-4000:]}"
        )
    return json.loads(out_path.read_text())


def read_runtime(path: Path) -> pd.DataFrame:
    """Stored runtime rows, or an empty frame with the columns if none exist."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=list(COLUMNS))
    frame = pd.read_csv(path, dtype=dict.fromkeys(_TEXT_COLUMNS, str))
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is not a runtime file: missing columns {missing}.")
    frame["pipelines"] = frame["pipelines"].fillna("")
    return frame[list(COLUMNS)]


def _key(measurement: Measurement, random_state: int) -> tuple[Any, ...]:
    return (*asdict(measurement).values(), int(random_state))


def _check_configuration(frame: pd.DataFrame, spec: RuntimeSpec, path: Path) -> None:
    """Warn when stored rows ran another grid or other pipelines than spec."""
    problems = []
    expected_configs = spec.n_configs
    other_grid = frame[frame["n_configs"] != expected_configs]
    if not other_grid.empty:
        counts = sorted(int(n) for n in other_grid["n_configs"].unique())
        problems.append(
            f"{len(other_grid)} of {len(frame)} fits ran {counts} configurations "
            f"instead of the study's {expected_configs}"
        )

    randomized = frame[frame["setting"] == "randomized"]
    observed: set[str] = set()
    expected: set[str] = set()
    stale = 0
    for row in randomized.itertuples(index=False):
        drawn = set(json.loads(row.pipelines)) if row.pipelines else set()
        allocated = set(spec.pipelines(row.n_resamples, row.random_state))
        if drawn != allocated:
            stale += 1
            observed |= drawn
            expected |= allocated
    if stale:
        problems.append(
            f"{stale} of {len(randomized)} randomized fits drew other pipelines "
            f"({', '.join(sorted(observed))}) than the study's "
            f"({', '.join(sorted(expected))})"
        )

    if problems:
        warnings.warn(
            f"The runtime rows in {path} were not measured with the study's current "
            f"configuration: {'; '.join(problems)}. Delete the file to measure it.",
            stacklevel=2,
        )


def run_measurements(
    spec: RuntimeSpec,
    X: np.ndarray,
    y: np.ndarray | pd.Series | None,
    *,
    path: Path,
    plan: Sequence[Measurement],
    random_state: int = 42,
) -> pd.DataFrame:
    """Rows for every measurement in plan, fitting only those path lacks.

    Each missing fit runs in its own process, and its row is written to path
    as soon as it finishes. The returned rows follow plan's order.
    """
    path = Path(path)
    frame = read_runtime(path)
    done = set(frame[list(_KEY_COLUMNS)].itertuples(index=False, name=None))
    missing = [m for m in plan if _key(m, random_state) not in done]

    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = frame.to_dict("records")
        with tempfile.TemporaryDirectory() as tmp:
            payload_path = Path(tmp) / "payload.pkl"
            with payload_path.open("wb") as handle:
                pickle.dump(
                    {
                        "spec": spec,
                        "X": np.asarray(X, dtype=np.float64),
                        "y": None if y is None else np.asarray(y),
                    },
                    handle,
                )
            for i, measurement in enumerate(tqdm(missing, desc="Runtime fits")):
                out_path = Path(tmp) / f"row_{i}.json"
                rows.append(
                    _run_in_process(payload_path, measurement, random_state, out_path)
                )
                pd.DataFrame(rows, columns=list(COLUMNS)).to_csv(path, index=False)
        frame = read_runtime(path)

    keys = [_key(m, random_state) for m in plan]
    indexed = frame.drop_duplicates(list(_KEY_COLUMNS), keep="last").set_index(
        list(_KEY_COLUMNS)
    )
    selected = indexed.loc[keys].reset_index()[list(COLUMNS)]
    _check_configuration(selected, spec, path)
    return selected


def measure_or_load_runtime(
    study: Study,
    X: np.ndarray,
    y: np.ndarray | pd.Series | None,
    *,
    path: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """The study's size ladder and parallel check, measured or read from path."""
    spec = RuntimeSpec.from_study(study)
    settings = SETTINGS if spec.randomizable else ("fixed",)
    plan = measurement_plan(np.asarray(X).shape[0], settings=settings)
    return run_measurements(spec, X, y, path=path, plan=plan, random_state=random_state)


def runtime_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """When, where and with which pipelines the fits were measured."""
    rows = frame.assign(
        measured=frame["started"].str[:10],
        carve_commit=frame["carve_commit"].fillna("").str[:7],
        ram_gb=frame["ram_gb"].round(1),
        pipelines=frame["pipelines"].map(
            lambda text: ", ".join(json.loads(text)) if text else "none"
        ),
    )
    columns = [
        "measured",
        "carve_commit",
        "machine",
        "cpu_count",
        "ram_gb",
        "python",
        "sklearn",
        "setting",
        "pipelines",
    ]
    return rows.groupby(columns, sort=False).size().rename("fits").reset_index()


def runtime_ladder_table(
    frame: pd.DataFrame, *, n_resamples: int = PROJECTED_RESAMPLES
) -> pd.DataFrame:
    """Seconds per resample, hours at n_resamples, shares and peak memory per fit.

    The hours are projected: the time per resample times n_resamples, as a
    fit's cost is linear in its resample count.
    """
    ladder = frame[frame["label"] == "ladder"].copy()
    ladder["order"] = ladder["setting"].map({s: i for i, s in enumerate(SETTINGS)})
    ladder = ladder.sort_values(["order", "n_cells"])
    wall = ladder["wall_s"]
    forest = ladder["forest_fit_s"] + ladder["forest_predict_s"]
    other = wall - ladder["embedding_s"] - ladder["clustering_s"] - forest
    per_resample = wall / ladder["n_resamples"]
    table = pd.DataFrame(
        {
            "setting": ladder["setting"],
            "cells": ladder["n_cells"],
            "per_resample_s": per_resample,
            "projected_h": per_resample * n_resamples / 3600,
            "embedding_pct": 100 * ladder["embedding_s"] / wall,
            "clustering_pct": 100 * ladder["clustering_s"] / wall,
            "forest_pct": 100 * forest / wall,
            "other_pct": 100 * other / wall,
            "peak_memory_gb": ladder["peak_memory_gb"],
        }
    )
    return table.reset_index(drop=True)


def _cells(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def runtime_ladder_latex(
    frame: pd.DataFrame, *, n_resamples: int = PROJECTED_RESAMPLES
) -> str:
    """The ladder as a booktabs tabular, the layout of S14 Table."""
    table = runtime_ladder_table(frame, n_resamples=n_resamples)
    clustering = str(frame["estimator"].iloc[0]).capitalize()
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        rf"Cells & Per resample & $B = {n_resamples}$ & Embedding & {clustering} & Forest"
        r" & Other & Peak memory \\",
        r" & (s) & (h) & (\%) & (\%) & (\%) & (\%) & (GB) \\",
        r"\midrule",
    ]
    for i, (setting, group) in enumerate(table.groupby("setting", sort=False)):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{8}}{{l}}{{{_SETTING_TITLES[setting]}}} \\")
        for row in group.itertuples(index=False):
            embedding = "--" if setting == "fixed" else f"{row.embedding_pct:.0f}"
            lines.append(
                f"{_cells(row.cells)} & {row.per_resample_s:.1f} & "
                f"{row.projected_h:.1f} & {embedding} & {row.clustering_pct:.0f} & "
                f"{row.forest_pct:.0f} & {row.other_pct:.0f} & "
                f"{row.peak_memory_gb:.1f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def runtime_slopes(frame: pd.DataFrame, *, min_cells: int = 10_000) -> pd.DataFrame:
    """Slope of log time against log cells per component, from min_cells up.

    NaN where a component takes no time (the fixed setting embeds nothing) or
    fewer than two rungs qualify.
    """
    ladder = frame[(frame["label"] == "ladder") & (frame["n_cells"] >= min_cells)]
    slopes: dict[str, dict[str, float]] = {}
    for setting in [s for s in SETTINGS if s in set(ladder["setting"])]:
        rows = ladder[ladder["setting"] == setting].sort_values("n_cells")
        forest = rows["forest_fit_s"] + rows["forest_predict_s"]
        other = rows["wall_s"] - rows["embedding_s"] - rows["clustering_s"] - forest
        components = {
            "total": rows["wall_s"],
            "embedding": rows["embedding_s"],
            "clustering": rows["clustering_s"],
            "forest": forest,
            "other": other,
        }
        log_cells = np.log(rows["n_cells"].to_numpy(dtype=float))
        slopes[setting] = {}
        for name, seconds in components.items():
            values = seconds.to_numpy(dtype=float)
            if len(values) < 2 or not np.all(values > 0):
                slopes[setting][name] = float("nan")
            else:
                slopes[setting][name] = float(
                    np.polyfit(log_cells, np.log(values), 1)[0]
                )
    return pd.DataFrame(slopes)


def projected_hours(frame: pd.DataFrame, *, n_resamples: int) -> pd.DataFrame:
    """One worker's hours for n_resamples on the largest rung, per setting."""
    ladder = frame[frame["label"] == "ladder"]
    rows = []
    for setting in [s for s in SETTINGS if s in set(ladder["setting"])]:
        largest = ladder[ladder["setting"] == setting].sort_values("n_cells").iloc[-1]
        per_resample = largest["wall_s"] / largest["n_resamples"]
        rows.append(
            {
                "setting": setting,
                "cells": int(largest["n_cells"]),
                "n_resamples": int(n_resamples),
                "hours": per_resample * n_resamples / 3600,
            }
        )
    return pd.DataFrame(rows)


def runtime_parallel_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Minutes, speedup over one worker and peak memory per worker count."""
    parallel = frame[frame["label"] == "parallel"].sort_values("n_jobs")
    one = parallel.loc[parallel["n_jobs"] == 1, "wall_s"]
    base = float(one.iloc[0]) if len(one) else float("nan")
    table = pd.DataFrame(
        {
            "cells": parallel["n_cells"],
            "n_resamples": parallel["n_resamples"],
            "n_jobs": parallel["n_jobs"],
            "wall_clock_min": parallel["wall_s"] / 60,
            "speedup": base / parallel["wall_s"],
            "peak_memory_gb": parallel["peak_memory_gb"],
        }
    )
    return table.reset_index(drop=True)


def save_runtime_tables(
    frame: pd.DataFrame, out_dir: Path, *, prefix: str
) -> list[Path]:
    """Write the rows as CSV and the ladder as the S14 Table tabular."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{prefix}_runtime.csv"
    latex_path = out_dir / f"{prefix}_runtime_ladder.tex"
    frame.to_csv(rows_path, index=False)
    latex_path.write_text(runtime_ladder_latex(frame))
    return [rows_path, latex_path]


def _main(argv: Sequence[str] | None = None) -> None:
    payload_path, measurement, random_state, out_path = (
        sys.argv[1:] if argv is None else argv
    )
    with open(payload_path, "rb") as handle:
        payload = pickle.load(handle)
    row = _measure(payload, Measurement(**json.loads(measurement)), int(random_state))
    Path(out_path).write_text(json.dumps(row))


if __name__ == "__main__":
    _main()
