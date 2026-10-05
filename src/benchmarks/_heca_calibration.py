"""The calibration stage: a measured resolution grid and a runtime projection.

Runs once on the cluster before the fit, on the training split and complement
that the fit's first resample draws. For each setting it builds both graphs
with the neighbor counts CARVE scales to each subset, scans Leiden
resolution on the training graph, and times every component the fit
repeats, single-threaded as each fit worker runs. It proposes the grid by
the spec's rule and projects the fit's runtime; the author commits the grid
to STUDIES before submitting the fit. The scan and the costs are written
even when the rule fails, so a failed rule never discards hours of scanning.

The forest is timed at both ends of the grid: once on labels near the
provisional grid's geometric middle, and once on the scan's fine end, the
most clusters within the upper target. A fitted forest's size and fit time
grow steeply with its class count, so a middle-only forest would understate
both the per-worker memory peak the fit's worker count is chosen from and
the forest's share of the projection.
"""

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import carve.cluster as _cluster
from carve._utils import default_generalizability_classifier, split_subsample_indices

from ._artifacts import peak_rss_bytes
from ._heca_stages import (
    CARVE_N_TREES,
    SEED,
    SUBSAMPLE_RATIO,
    guard_output,
    record_stage,
    resolve_study,
    scaled_neighbors,
    sweep_settings,
    write_json,
)
from ._studies import load_study
from ._timing import GraphTimes, graph_timers
from ._types import Study

#: Forty log-spaced values from 0.001 to 10: wide enough for either setting
#: to pass organ level and twice the cell-type count.
SCAN_RESOLUTIONS: tuple[float, ...] = tuple(
    float(value) for value in np.geomspace(0.001, 10.0, 40)
)

#: Values in the proposed grid, as many as the provisional grid holds.
GRID_SIZE = 15

#: Worker counts the projection is tabulated for. The fit node's count is
#: not known when calibration runs.
PROJECTION_N_JOBS: tuple[int, ...] = (16, 24, 32, 48, 64, 128)


class GridRuleError(ValueError):
    """The scan does not support a grid under the calibration rule."""


def propose_grid(
    scan: pd.DataFrame,
    *,
    lower_target: int,
    upper_target: int,
    size: int = GRID_SIZE,
) -> tuple[float, ...]:
    """The grid the calibration rule proposes from a scan.

    For each setting, lo is the smallest scanned resolution giving at least
    lower_target clusters and hi the largest giving at most upper_target.
    The grid runs from the smallest lo to the largest hi, so every setting
    covers organ level at the coarse end and the upper target at the fine
    end, in size log-spaced values rounded to two significant figures.
    """
    lows, highs = [], []
    for setting, rows in scan.groupby("setting", sort=False):
        reaching = rows.loc[rows["n_clusters"] >= lower_target, "resolution"]
        if reaching.empty:
            raise GridRuleError(
                f"Setting {setting!r} never reaches {lower_target} clusters "
                f"(largest scanned resolution {rows['resolution'].max():g}); "
                "widen the scan upward."
            )
        within = rows.loc[rows["n_clusters"] <= upper_target, "resolution"]
        if within.empty:
            raise GridRuleError(
                f"Setting {setting!r} already exceeds {upper_target} clusters at "
                f"{rows['resolution'].min():g}; widen the scan downward."
            )
        lows.append(float(reaching.min()))
        highs.append(float(within.max()))

    lo, hi = min(lows), max(highs)
    if lo >= hi:
        raise GridRuleError(
            f"The endpoints cross: the grid would start at {lo:g} and end at "
            f"{hi:g}. Inspect the scan; the targets may not suit this data."
        )
    grid = tuple(
        float(f"{lo * (hi / lo) ** (i / (size - 1)):.2g}") for i in range(size)
    )
    if len(set(grid)) != size:
        raise GridRuleError(
            f"From {lo:g} to {hi:g}, rounding to two significant figures gives "
            f"duplicate values: {grid}."
        )
    return grid


def fine_end_row(scan: pd.DataFrame, *, upper_target: int) -> tuple[str, float]:
    """The (setting, resolution) of the scan's fine end.

    The row with the most clusters not exceeding upper_target, across every
    setting: the most classes a fit worker's forest trains on inside the
    grid. When no row is within the target, the row with the fewest
    clusters. Ties go to the earlier row.
    """
    within = scan[scan["n_clusters"] <= upper_target]
    if within.empty:
        row = scan.loc[scan["n_clusters"].idxmin()]
    else:
        row = within.loc[within["n_clusters"].idxmax()]
    return str(row["setting"]), float(row["resolution"])


def project_runtime(
    costs: Mapping[str, Mapping[str, float]],
    *,
    n_resolutions: int,
    n_resamples: int,
    n_train: int,
    n_test: int,
    n_jobs_options: Sequence[int] = PROJECTION_N_JOBS,
) -> dict[str, Any]:
    """Project the fit's runtime from single-threaded component costs.

    One configuration and one resample cost two training graphs, one
    complement graph, three Leiden runs (the complement's scaled by its
    size) and one forest fit and predict. Resamples run in rounds of n_jobs
    workers, so wall-clock time counts whole rounds. A lower bound: it
    leaves out CARVE's own overhead and contention between workers.
    """
    per_resample = {
        setting: (
            2 * c["graph_train_s"]
            + c["graph_test_s"]
            + (2 + n_test / n_train) * c["leiden_s"]
            + c["forest_fit_s"]
            + c["forest_predict_s"]
        )
        for setting, c in costs.items()
    }
    core_seconds = n_resolutions * n_resamples * sum(per_resample.values())
    wall_clock_hours = {
        str(n_jobs): n_resolutions
        * sum(math.ceil(n_resamples / n_jobs) * c for c in per_resample.values())
        / 3600
        for n_jobs in n_jobs_options
    }
    return {
        "per_resample_seconds": per_resample,
        "core_hours": core_seconds / 3600,
        "wall_clock_hours": wall_clock_hours,
    }


def calibration_costs(
    calibration: Mapping[str, Any],
    scan: pd.DataFrame,
    *,
    resolutions: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Per-setting costs for project_runtime, for a grid's range.

    Leiden's cost is the mean over the scanned resolutions inside the grid's
    range, or over the whole scan when none falls inside. The forest's cost
    is the mean of the middle and fine-end forests, or the middle forest
    alone for a calibration.json written before the fine end was timed.
    """
    lo, hi = min(resolutions), max(resolutions)
    forest = {
        key: float(
            np.mean([calibration[key], calibration[fine_key]])
            if fine_key in calibration
            else calibration[key]
        )
        for key, fine_key in (
            ("forest_fit_s", "forest_fit_fine_s"),
            ("forest_predict_s", "forest_predict_fine_s"),
        )
    }
    costs: dict[str, dict[str, float]] = {}
    for label, graph in calibration["graph_costs"].items():
        rows = scan[scan["setting"] == label]
        inside = rows[(rows["resolution"] >= lo) & (rows["resolution"] <= hi)]
        leiden_s = float((inside if not inside.empty else rows)["seconds"].mean())
        costs[label] = {
            "graph_train_s": float(graph["graph_train_s"]),
            "graph_test_s": float(graph["graph_test_s"]),
            "leiden_s": leiden_s,
            **forest,
        }
    return costs


def timed_graph(X: np.ndarray, n_neighbors: int) -> tuple[Any, GraphTimes]:
    """Build the graph LeidenClustering.fit builds on X, timing its two parts."""
    with graph_timers() as times:
        graph = _cluster.build_knn_graph(X, n_neighbors=n_neighbors)
    return graph, times


def scan_resolutions(
    graph: Any, resolutions: Sequence[float], *, seed: int
) -> list[tuple[float, np.ndarray, float]]:
    """Leiden on one prebuilt graph at each resolution.

    The partition call and its arguments are LeidenClustering.fit's, so the
    labels match a fit on the same data and seed.
    """
    import leidenalg as la

    results = []
    for resolution in resolutions:
        started = time.perf_counter()
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            resolution_parameter=float(resolution),
            weights="weight",
            n_iterations=-1,
            seed=seed,
        )
        seconds = time.perf_counter() - started
        labels = np.asarray(partition.membership, dtype=np.int32)
        results.append((float(resolution), labels, seconds))
    return results


def _time_forest(
    X_train: np.ndarray, labels: np.ndarray, X_test: np.ndarray
) -> tuple[float, float]:
    """Seconds to fit the default forest on labels and to predict X_test.

    Single-threaded, as in a fit worker. The forest is dropped on return,
    so a second call measures one forest at a time.
    """
    forest = default_generalizability_classifier(
        classifier=None,
        n_features=int(X_train.shape[1]),
        n_trees=CARVE_N_TREES,
        random_state=SEED,
        n_jobs=1,
    )
    started = time.perf_counter()
    forest.fit(X_train, labels)
    fit_s = time.perf_counter() - started
    started = time.perf_counter()
    forest.predict(X_test)
    predict_s = time.perf_counter() - started
    del forest
    return fit_s, predict_s


def run_calibrate(
    run_dir: Path,
    *,
    study: Study | None = None,
    scan: Sequence[float] = SCAN_RESOLUTIONS,
    force: bool = False,
) -> dict[str, Any]:
    """Scan, time and project; write calibration.csv and calibration.json.

    The default forest is fit and timed twice, one forest at a time as in a
    fit worker: on the first setting's labels near the provisional grid's
    geometric middle, then on the scan's fine end (fine_end_row). The fine
    end has the most classes a worker's forest trains on, and a forest's
    size grows steeply with its class count, so worker_peak_bytes, read
    after the second fit, bounds what one fit worker needs.
    """
    study = resolve_study(study)
    run_dir = Path(run_dir)
    csv_path = run_dir / "calibration.csv"
    json_path = run_dir / "calibration.json"
    guard_output(csv_path, force=force)
    guard_output(json_path, force=force)
    record_stage(run_dir, "calibrate")

    X, _, meta = load_study(study)
    baseline_rss = peak_rss_bytes()
    obs = meta["obs"]
    n_cells = int(X.shape[0])
    # The split the fit's first resample draws (seed 42 + resample 0).
    train_idx, test_idx = split_subsample_indices(
        n_cells, subsample_ratio=SUBSAMPLE_RATIO, random_state=SEED
    )
    X_train, X_test = X[train_idx], X[test_idx]
    organ = obs["organ"].to_numpy()[train_idx]
    cell_type = obs["cell_type"].to_numpy()[train_idx]
    middle = math.sqrt(min(study.resolutions) * max(study.resolutions))
    lower_target = int(obs["organ"].nunique())
    upper_target = 2 * int(obs["cell_type"].nunique())

    rows: list[dict[str, Any]] = []
    graph_costs: dict[str, dict[str, float]] = {}
    forest_labels: np.ndarray | None = None
    # The labels of the fine end over the settings scanned so far: only this
    # one candidate's partition is kept, not every scanned one.
    fine_labels: np.ndarray | None = None
    for setting in sweep_settings(study):
        k_train = scaled_neighbors(setting, n_fit=len(train_idx), n_full=n_cells)
        k_test = scaled_neighbors(setting, n_fit=len(test_idx), n_full=n_cells)
        graph, train_times = timed_graph(X_train, k_train)
        _, test_times = timed_graph(X_test, k_test)
        graph_costs[setting.label] = {
            "knn_train_s": train_times.knn_s,
            "graph_train_s": train_times.build_s,
            "knn_test_s": test_times.knn_s,
            "graph_test_s": test_times.build_s,
        }
        scanned = scan_resolutions(graph, scan, seed=SEED)
        del graph
        for resolution, labels, seconds in scanned:
            rows.append(
                {
                    "setting": setting.label,
                    "n_neighbors": setting.n_neighbors,
                    "n_neighbors_scaled": k_train,
                    "resolution": resolution,
                    "n_clusters": int(np.unique(labels).size),
                    "seconds": seconds,
                    "ari_organ": float(adjusted_rand_score(organ, labels)),
                    "ari_cell_type": float(adjusted_rand_score(cell_type, labels)),
                }
            )
        if forest_labels is None:
            # Any realistic partition times the forest: take the first
            # setting's at the scan value nearest the provisional grid's
            # geometric middle.
            _, forest_labels, _ = min(
                scanned, key=lambda item: abs(math.log(item[0] / middle))
            )
        fine_setting, fine_resolution = fine_end_row(
            pd.DataFrame(rows), upper_target=upper_target
        )
        if fine_setting == setting.label:
            fine_labels = next(
                labels for value, labels, _ in scanned if value == fine_resolution
            )
        del scanned

    # After the last setting, fine_setting and fine_resolution are the whole
    # scan's fine end. The middle forest is gone before the fine one is fit.
    forest_fit_s, forest_predict_s = _time_forest(X_train, forest_labels, X_test)
    forest_fit_fine_s, forest_predict_fine_s = _time_forest(
        X_train, fine_labels, X_test
    )
    # ru_maxrss is a running maximum: this covers the graphs and both
    # forests, the fine-end one with the most classes.
    worker_peak_bytes = int(peak_rss_bytes())

    scan_frame = pd.DataFrame(rows)
    scan_frame.to_csv(csv_path, index=False)

    grid: tuple[float, ...] | None
    try:
        grid = propose_grid(
            scan_frame, lower_target=lower_target, upper_target=upper_target
        )
        grid_error = None
    except GridRuleError as error:
        grid, grid_error = None, str(error)

    record: dict[str, Any] = {
        "n_cells": n_cells,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "scan": [float(value) for value in scan],
        "lower_target": lower_target,
        "upper_target": upper_target,
        "proposed_grid": None if grid is None else list(grid),
        "grid_error": grid_error,
        "graph_costs": graph_costs,
        "forest_fit_s": forest_fit_s,
        "forest_predict_s": forest_predict_s,
        "forest_fit_fine_s": forest_fit_fine_s,
        "forest_predict_fine_s": forest_predict_fine_s,
        "forest_fine": {
            "setting": fine_setting,
            "resolution": fine_resolution,
            "n_clusters": int(np.unique(fine_labels).size),
        },
        "baseline_rss_bytes": int(baseline_rss),
        "worker_peak_bytes": worker_peak_bytes,
    }
    costs = calibration_costs(
        record, scan_frame, resolutions=grid if grid is not None else tuple(scan)
    )
    record["projection"] = project_runtime(
        costs,
        n_resolutions=len(grid) if grid is not None else len(study.resolutions),
        n_resamples=study.n_resamples,
        n_train=record["n_train"],
        n_test=record["n_test"],
    )
    write_json(json_path, record)
    return record
