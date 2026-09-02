"""Reading and writing benchmark artifacts.

Runs are content-addressed: results/runs/<scenario>/<config-hash>/. Changing
an anchor produces a new directory rather than a silent stale read. Each cell
is checkpointed as its own parquet file so a crash at seed 55 of 60 costs one
cell instead of the whole run.
"""

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from ._registry import ACTIVE_ANCHOR_SET_NAME
from ._types import Manifest, Scenario

SCHEMA: tuple[str, ...] = (
    "run_id",
    "scenario",
    "axis_name",
    "axis_value",
    "axis_label",
    "seed",
    "k_star",
    "estimator",
    "metric_name",
    "k",
    "metric_value",
    "is_selected",
    "selects_true_k",
    "ari_at_k",
    "oracle_ari",
)

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "joblib",
    "carve-validate",
)


def _canonical_config(
    scenario: Scenario, *, n_seeds: int, n_resamples: int
) -> dict[str, Any]:
    """The subset of a scenario that changing must invalidate a run."""
    return {
        "name": scenario.name,
        "axis_name": scenario.axis.name,
        "axis_values": list(scenario.axis.values),
        "axis_labels": list(scenario.axis.labels),
        "anchors": {k: dict(v) for k, v in sorted(scenario.anchors.items())},
        "shared": dict(sorted(scenario.shared.items())),
        "estimator": scenario.estimator.name,
        "k_star": scenario.k_star,
        "candidate_k": list(scenario.candidate_k),
        "n_trees": scenario.n_trees,
        "n_seeds": n_seeds,
        "n_resamples": n_resamples,
    }


def config_hash(scenario: Scenario, *, n_seeds: int, n_resamples: int) -> str:
    """Twelve-hex-character digest of everything that defines a run."""
    payload = json.dumps(
        _canonical_config(scenario, n_seeds=n_seeds, n_resamples=n_resamples),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_dir(root: Path, scenario_name: str, cfg_hash: str) -> Path:
    """Create and return the directory for one run."""
    path = Path(root) / scenario_name / cfg_hash
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(rd: Path, axis_label: str, seed: int) -> Path:
    return Path(rd) / f"cell__{axis_label}__{seed:04d}.parquet"


def write_checkpoint(
    rd: Path, axis_label: str, seed: int, rows: list[dict[str, Any]]
) -> Path:
    """Write one cell's rows, validating them against the schema first."""
    frame = pd.DataFrame(rows)
    missing = [column for column in SCHEMA if column not in frame.columns]
    if missing:
        raise ValueError(f"Rows are missing schema columns: {missing}.")
    extra = [column for column in frame.columns if column not in SCHEMA]
    if extra:
        raise ValueError(f"Rows carry columns outside the schema: {extra}.")

    path = _checkpoint_path(rd, axis_label, seed)
    frame[list(SCHEMA)].to_parquet(path, index=False)
    return path


def completed_cells(rd: Path) -> set[tuple[str, int]]:
    """Return the (axis_label, seed) pairs already on disk."""
    cells: set[tuple[str, int]] = set()
    for path in Path(rd).glob("cell__*.parquet"):
        _, axis_label, seed = path.stem.split("__")
        cells.add((axis_label, int(seed)))
    return cells


def read_run(rd: Path) -> pd.DataFrame:
    """Concatenate every checkpoint in a run directory."""
    paths = sorted(Path(rd).glob("cell__*.parquet"))
    if not paths:
        return pd.DataFrame(columns=list(SCHEMA))
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return frame[list(SCHEMA)]


def peak_rss_bytes() -> int:
    """Peak resident set size of this process and its children, in bytes.

    ru_maxrss is reported in bytes on macOS and in kilobytes on Linux. The
    unit is normalized here and recorded in the manifest, because a benchmark
    whose headline claim is memory cannot carry a silent factor-of-1024
    difference between the machine an author ran it on and CI.
    """
    scale = 1 if sys.platform == "darwin" else 1024
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(max(own, children) * scale)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def build_manifest(
    scenario: Scenario,
    *,
    run_id: str,
    config_hash: str,
    n_seeds: int,
    n_resamples: int,
    n_jobs: int,
    random_state: int,
    wall_clock_s: float,
) -> Manifest:
    """Assemble the provenance record for one run."""
    return Manifest(
        run_id=run_id,
        scenario=scenario.name,
        config_hash=config_hash,
        anchor_set=ACTIVE_ANCHOR_SET_NAME,
        n_seeds=n_seeds,
        n_resamples=n_resamples,
        n_jobs=n_jobs,
        random_state=random_state,
        wall_clock_s=float(wall_clock_s),
        peak_rss_bytes=peak_rss_bytes(),
        peak_rss_unit="bytes",
        git_sha=_git_sha(),
        package_versions=_package_versions(),
        platform={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cores": os.cpu_count(),
        },
        config=_canonical_config(scenario, n_seeds=n_seeds, n_resamples=n_resamples),
    )


def write_manifest(rd: Path, manifest: Manifest) -> Path:
    """Write manifest.json beside the checkpoints."""
    path = Path(rd) / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str))
    return path


def promote(rd: Path, published_root: Path) -> Path:
    """Publish a run as committed CSV plus its manifest.

    Deliberate and explicit. A run is never promoted as a side effect of
    executing one.
    """
    rd = Path(rd)
    manifest_path = rd / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. A run without provenance is not publishable."
        )

    manifest = json.loads(manifest_path.read_text())
    out = Path(published_root) / manifest["scenario"]
    out.mkdir(parents=True, exist_ok=True)

    read_run(rd).to_csv(out / "results.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out
