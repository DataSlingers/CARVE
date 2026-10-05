"""Timing instrumentation for a CARVE fit's components.

CARVE runs each resample in a loky worker process, where a profiler in the
parent cannot see. The classes here time themselves inside the worker and
append one CSV row per call to the directory named by CARVE_TIMING_DIR, which
loky workers inherit when they are spawned. With the variable unset they
behave exactly as their parents and write nothing.

LeidenClustering keeps its parent's class name, because CARVE labels each
configuration in estimator_results_ by the estimator class's __name__; an
instrumented run's tables then read exactly as an uninstrumented run's.
"""

import csv
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import ClusterMixin
from sklearn.ensemble import RandomForestClassifier

import carve.cluster as _cluster

TIMING_DIR_ENV = "CARVE_TIMING_DIR"

LEIDEN_COLUMNS: tuple[str, ...] = (
    "pid",
    "host",
    "unix_time",
    "n_neighbors",
    "resolution",
    "n_samples",
    "n_clusters",
    "knn_s",
    "graph_s",
    "leiden_s",
    "fit_s",
)
FOREST_COLUMNS: tuple[str, ...] = (
    "pid",
    "host",
    "unix_time",
    "call",
    "n_samples",
    "seconds",
)


@dataclass
class GraphTimes:
    """Seconds spent building neighbor graphs, accumulated across calls.

    knn_s is the neighbor search alone; build_s is all of build_knn_graph,
    the search included, so graph assembly is build_s - knn_s.
    """

    knn_s: float = 0.0
    build_s: float = 0.0


@contextmanager
def graph_timers() -> Iterator[GraphTimes]:
    """Time carve.cluster's neighbor search and graph construction.

    Replaces the two module attributes LeidenClustering.fit reaches through
    for the duration of the block, so the fit itself is not reimplemented.
    The replacement is process-local, and a loky worker runs one task at a
    time.
    """
    times = GraphTimes()
    search = _cluster.kneighbors_graph
    build = _cluster.build_knn_graph

    def timed_search(*args: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return search(*args, **kwargs)
        finally:
            times.knn_s += time.perf_counter() - started

    def timed_build(*args: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return build(*args, **kwargs)
        finally:
            times.build_s += time.perf_counter() - started

    _cluster.kneighbors_graph = timed_search
    _cluster.build_knn_graph = timed_build
    try:
        yield times
    finally:
        _cluster.kneighbors_graph = search
        _cluster.build_knn_graph = build


def _append_row(kind: str, row: dict[str, Any], columns: tuple[str, ...]) -> None:
    """Append one row to this process's file of the given kind."""
    directory = Path(os.environ[TIMING_DIR_ENV])
    host, pid = socket.gethostname(), os.getpid()
    path = directory / f"{kind}-{host}-{pid}.csv"
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow({"pid": pid, "host": host, "unix_time": time.time(), **row})


class LeidenClustering(_cluster.LeidenClustering):
    """carve.cluster.LeidenClustering that times its graph build and Leiden."""

    def fit(self, X, y=None):
        if not os.environ.get(TIMING_DIR_ENV):
            return super().fit(X, y)
        started = time.perf_counter()
        with graph_timers() as times:
            super().fit(X, y)
        total = time.perf_counter() - started
        _append_row(
            "leiden",
            {
                "n_neighbors": int(self.n_neighbors),
                "resolution": float(self.resolution),
                "n_samples": int(len(X)),
                "n_clusters": int(self.n_clusters_),
                "knn_s": times.knn_s,
                "graph_s": times.build_s - times.knn_s,
                "leiden_s": total - times.build_s,
                "fit_s": total,
            },
            LEIDEN_COLUMNS,
        )
        return self


class TimedRandomForestClassifier(RandomForestClassifier):
    """RandomForestClassifier that times fit and predict.

    Defines no __init__, so get_params and clone see the parent's
    parameters, and CARVE clones it and sets random_state and n_jobs as it
    does for any user classifier.
    """

    def fit(self, X, y, sample_weight=None):
        if not os.environ.get(TIMING_DIR_ENV):
            return super().fit(X, y, sample_weight=sample_weight)
        started = time.perf_counter()
        super().fit(X, y, sample_weight=sample_weight)
        seconds = time.perf_counter() - started
        _append_row(
            "forest",
            {"call": "fit", "n_samples": int(len(X)), "seconds": seconds},
            FOREST_COLUMNS,
        )
        return self

    def predict(self, X):
        if not os.environ.get(TIMING_DIR_ENV):
            return super().predict(X)
        started = time.perf_counter()
        labels = super().predict(X)
        seconds = time.perf_counter() - started
        _append_row(
            "forest",
            {"call": "predict", "n_samples": int(len(X)), "seconds": seconds},
            FOREST_COLUMNS,
        )
        return labels


def timed_forest(*, n_features: int, n_trees: int) -> TimedRandomForestClassifier:
    """CARVE's default generalizability forest, timed.

    Built from the parameters of CARVE's own factory rather than restated,
    so an instrumented fit trains exactly the classifier an uninstrumented
    one does. CARVE sets random_state and n_jobs on its clone per resample.
    """
    from carve._utils import default_generalizability_classifier

    plain = default_generalizability_classifier(
        classifier=None,
        n_features=n_features,
        n_trees=n_trees,
        random_state=None,
        n_jobs=1,
    )
    return TimedRandomForestClassifier(**plain.get_params())


def instrumented_grids(
    grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """grids with carve.cluster.LeidenClustering swapped for the timed class."""
    return [
        (LeidenClustering if cls is _cluster.LeidenClustering else cls, dict(grid))
        for cls, grid in grids
    ]


def _shutdown_worker_pool() -> None:
    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True)


@contextmanager
def timing_directory(path: Path) -> Iterator[Path]:
    """Point the timing classes at path for the duration of the block.

    loky workers copy the environment when they are spawned, and joblib
    keeps its worker pool alive between calls, so a pool started before the
    variable was set would never see it. The pool is shut down on entry, so
    the block's first parallel call spawns workers that inherit the
    variable, and again on exit, so no later call reuses workers that still
    carry it.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get(TIMING_DIR_ENV)
    _shutdown_worker_pool()
    os.environ[TIMING_DIR_ENV] = str(path)
    try:
        yield path
    finally:
        if previous is None:
            os.environ.pop(TIMING_DIR_ENV, None)
        else:
            os.environ[TIMING_DIR_ENV] = previous
        _shutdown_worker_pool()


def read_timings(directory: Path) -> dict[str, pd.DataFrame]:
    """Every row the timing classes wrote under directory, by kind.

    Safe to call while a fit is still writing: a row cut off mid-write is
    dropped rather than read as data.
    """
    directory = Path(directory)
    tables: dict[str, pd.DataFrame] = {}
    for kind, columns in (("leiden", LEIDEN_COLUMNS), ("forest", FOREST_COLUMNS)):
        frames = [
            pd.read_csv(path, on_bad_lines="skip").dropna()
            for path in sorted(directory.glob(f"{kind}-*.csv"))
        ]
        frames = [frame for frame in frames if not frame.empty]
        tables[kind] = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=list(columns))
        )
    return tables
