"""Read a copied-back hECA run and summarize it for the notebook.

Nothing here fits: the run directory holds what the cluster computed, and
figures._heca_* draw it. This module plays the part _cusanovich_compare
plays for its study.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from carve import CARVE

from ._artifacts import check_fingerprint
from ._heca_calibration import calibration_costs, project_runtime
from ._heca_stages import label_settings, read_json, resolve_study
from ._studies import carve_cache_path, study_resolution_grids
from ._timing import read_timings
from ._types import Study

_MEMORY_UNITS: dict[str, int] = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


@dataclass
class HecaRun:
    """Everything one run directory holds, with the fitted CARVE loaded."""

    run_dir: Path
    study: Study
    env: dict[str, Any]
    embed: dict[str, Any]
    calibration: dict[str, Any]
    scan: pd.DataFrame
    started: dict[str, Any]
    runtime: dict[str, Any]
    leiden: pd.DataFrame
    forest: pd.DataFrame
    memory: pd.DataFrame
    sacct: pd.DataFrame | None
    carve: CARVE


def slurm_duration_seconds(text: str) -> float:
    """A SLURM duration, [DD-][HH:]MM:SS[.mmm], in seconds."""
    text = str(text).strip()
    if not text:
        return float("nan")
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    parts = [float(part) for part in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts
    return days * 86_400 + hours * 3_600 + minutes * 60 + seconds


def slurm_memory_bytes(text: str) -> float:
    """A SLURM memory field such as 200G or 512K, in bytes."""
    text = str(text).strip()
    if not text:
        return float("nan")
    unit = text[-1]
    if unit in _MEMORY_UNITS:
        return float(text[:-1]) * _MEMORY_UNITS[unit]
    return float(text)


def parse_sacct(text: str) -> pd.DataFrame:
    """sacct -P output (pipe-separated, header first) as a frame."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    header = lines[0].split("|")
    frame = pd.DataFrame(
        [dict(zip(header, line.split("|"), strict=False)) for line in lines[1:]],
        columns=header,
    )
    for column in ("Elapsed", "TotalCPU"):
        if column in frame:
            frame[f"{column}_s"] = frame[column].map(slurm_duration_seconds)
    for column in ("MaxRSS", "AveRSS"):
        if column in frame:
            frame[f"{column}_bytes"] = frame[column].map(slurm_memory_bytes)
    return frame


def load_heca_run(
    run_dir: Path, *, X: np.ndarray, study: Study | None = None
) -> HecaRun:
    """Read a run directory and load its fit.

    The study is rebuilt with the grid and resample count the fit recorded
    in started.json, so the fit is found even after STUDIES moves on. X is
    the embedding the fit ran on, checked against the cache's fingerprint.
    """
    run_dir = Path(run_dir)
    fit_dir = run_dir / "fit"
    started = read_json(fit_dir / "started.json")
    study = replace(
        resolve_study(study),
        resolutions=tuple(float(r) for r in started["resolutions"]),
        n_resamples=int(started["n_resamples"]),
    )
    cache_path = carve_cache_path(
        study,
        root=fit_dir,
        model_grids=study_resolution_grids(study),
        n_resamples=study.n_resamples,
    )
    check_fingerprint(cache_path, X)
    carve = CARVE.load(str(cache_path))
    carve.X_ = np.asarray(X)

    timings = read_timings(fit_dir / "timings")
    sacct_path = fit_dir / "sacct.txt"
    return HecaRun(
        run_dir=run_dir,
        study=study,
        env=read_json(run_dir / "env.json"),
        embed=read_json(run_dir / "embed.json"),
        calibration=read_json(run_dir / "calibration.json"),
        scan=pd.read_csv(run_dir / "calibration.csv"),
        started=started,
        runtime=read_json(fit_dir / "runtime.json"),
        leiden=label_settings(
            timings["leiden"], study, n_cells=int(started["n_cells"])
        ),
        forest=timings["forest"],
        memory=pd.read_csv(fit_dir / "memory.csv"),
        sacct=parse_sacct(sacct_path.read_text()) if sacct_path.exists() else None,
        carve=carve,
    )


def fit_cpu_hours(run: HecaRun) -> float:
    """The fit step's total CPU time from SLURM, or NaN without sacct.txt."""
    if run.sacct is None or run.sacct.empty:
        return float("nan")
    return float(run.sacct["TotalCPU_s"].iloc[0]) / 3600


def runtime_table(run: HecaRun) -> pd.DataFrame:
    """Wall-clock time, CPU time, workers and peak memory: embedding and fit.

    The fit's peak memory is SLURM's MaxRSS when sacct.txt was copied back,
    else the largest sampled total of the fit process and its workers; the
    sampled value is reported beside it either way.
    """
    sampled = float(
        (run.memory["own_rss_bytes"] + run.memory["children_rss_bytes"]).max()
    )
    if run.sacct is not None and not run.sacct.empty:
        fit_memory, fit_source = float(run.sacct["MaxRSS_bytes"].iloc[0]), "slurm"
    else:
        fit_memory, fit_source = sampled, "sampled"
    return pd.DataFrame(
        [
            {
                "stage": "embedding",
                "wall_clock_h": run.embed["wall_clock_s"] / 3600,
                "cpu_h": float("nan"),
                "workers": 1,
                "peak_memory_gb": run.embed["peak_rss_bytes"] / 1e9,
                "memory_source": "process",
                "sampled_peak_memory_gb": float("nan"),
            },
            {
                "stage": "CARVE fit",
                "wall_clock_h": run.runtime["wall_clock_s"] / 3600,
                "cpu_h": fit_cpu_hours(run),
                "workers": int(run.runtime["n_jobs"]),
                "peak_memory_gb": fit_memory / 1e9,
                "memory_source": fit_source,
                "sampled_peak_memory_gb": sampled / 1e9,
            },
        ]
    )


def component_table(run: HecaRun) -> pd.DataFrame:
    """Core-hours per component, and each one's share of the fit's CPU time.

    The graph and Leiden components are per setting; the forest is for the
    whole fit, since the classifier cannot tell which configuration it
    serves. With sacct.txt, CARVE's own overhead (ARI scoring, consensus
    accumulation, scheduling) is the rest of the step's CPU time; without
    it, shares are of the timed components alone. Components are wall-clock
    seconds in single-threaded workers, which equals their CPU time when the
    fit ran one worker per physical core.
    """
    rows = []
    for setting, part in run.leiden.groupby("setting", sort=False):
        for column, name in (
            ("knn_s", "neighbor search"),
            ("graph_s", "graph assembly"),
            ("leiden_s", "Leiden"),
        ):
            rows.append(
                {
                    "component": name,
                    "setting": setting,
                    "core_hours": float(part[column].sum()) / 3600,
                }
            )
    for call, name in (("fit", "forest fit"), ("predict", "forest predict")):
        seconds = run.forest.loc[run.forest["call"] == call, "seconds"].sum()
        rows.append(
            {"component": name, "setting": "all", "core_hours": float(seconds) / 3600}
        )
    table = pd.DataFrame(rows)
    cpu_hours = fit_cpu_hours(run)
    if np.isnan(cpu_hours):
        table["share"] = table["core_hours"] / table["core_hours"].sum()
        return table
    overhead = {
        "component": "CARVE overhead",
        "setting": "all",
        "core_hours": cpu_hours - float(table["core_hours"].sum()),
    }
    table = pd.concat([table, pd.DataFrame([overhead])], ignore_index=True)
    table["share"] = table["core_hours"] / cpu_hours
    return table


def selection_table(
    run: HecaRun, obs: pd.DataFrame, *, rule: str = "1se"
) -> pd.DataFrame:
    """The configuration each criterion selects, and its agreement with the
    two references."""
    results = run.carve.estimator_results_
    rows = []
    for measure in ("stability", "generalizability"):
        estimator = run.carve.get_estimator(measure=measure, rule=rule)
        resolution = float(run.carve.get_sweep_value(measure=measure, rule=rule))
        labels = run.carve.get_labels(measure=measure, rule=rule)
        match = results[
            (results["n_neighbors"] == estimator.n_neighbors)
            & np.isclose(results["resolution"].astype(float), resolution)
        ]
        rows.append(
            {
                "measure": measure,
                "setting": f"leiden_{estimator.n_neighbors}",
                "resolution": resolution,
                "n_clusters_observed": float(match["n_clusters_observed"].iloc[0]),
                "n_clusters_consensus": int(np.unique(labels).size),
                "ari_organ": float(adjusted_rand_score(obs["organ"], labels)),
                "ari_cell_type": float(adjusted_rand_score(obs["cell_type"], labels)),
            }
        )
    return pd.DataFrame(rows)


def study_composition(
    labels: Sequence[int] | np.ndarray, obs: pd.DataFrame
) -> pd.DataFrame:
    """Each cluster's cells by study of origin, as fractions of the cluster.

    A cluster drawn mostly from one study may be recovering its study rather
    than biology: the pooled embedding is not batch-corrected.
    """
    clusters = pd.Series(np.asarray(labels), name="cluster")
    studies = pd.Series(obs["study_id"].to_numpy(), name="study_id")
    table = pd.crosstab(clusters, studies, normalize="index")
    table["n_cells"] = clusters.value_counts().reindex(table.index)
    table["dominant_share"] = table.drop(columns="n_cells").max(axis=1)
    return table.sort_values("n_cells", ascending=False)


def projection_vs_actual(run: HecaRun) -> dict[str, Any]:
    """Calibration's projection at the fit's worker count, beside the fit."""
    costs = calibration_costs(
        run.calibration, run.scan, resolutions=run.study.resolutions
    )
    n_jobs = int(run.runtime["n_jobs"])
    projection = project_runtime(
        costs,
        n_resolutions=len(run.study.resolutions),
        n_resamples=run.study.n_resamples,
        n_train=int(run.calibration["n_train"]),
        n_test=int(run.calibration["n_test"]),
        n_jobs_options=(n_jobs,),
    )
    return {
        "n_jobs": n_jobs,
        "projected_wall_clock_h": projection["wall_clock_hours"][str(n_jobs)],
        "actual_wall_clock_h": run.runtime["wall_clock_s"] / 3600,
    }
