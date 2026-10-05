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

import json
import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from carve import CARVE
from carve._utils import scale_neighbor_count
from carve.cluster import LeidenClustering

from ._artifacts import peak_rss_bytes, provenance
from ._studies import STUDIES, load_study, study_resolution_grids
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


def physical_cores() -> int:
    """Physical cores, or logical ones where psutil cannot tell."""
    import psutil

    return int(psutil.cpu_count(logical=False) or os.cpu_count() or 1)


def node_memory_bytes() -> int:
    import psutil

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
    counts and memory, since the stages run on different nodes.
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
