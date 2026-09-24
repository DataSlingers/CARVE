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
import tempfile
import warnings
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
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

RUNTIME_SCHEMA: tuple[str, ...] = (
    "run_id",
    "scenario",
    "axis_name",
    "axis_value",
    "axis_label",
    "seed",
    "n_samples",
    "n_features",
    "n_resamples",
    "n_jobs",
    "estimator",
    # The default-mode fit is the one that produced the metric rows. The two
    # mode-specific timings come from extra fits whose results are discarded;
    # they exist only to reproduce the published two-curve runtime figure.
    "t_default_s",
    "t_stability_s",
    "t_generalizability_s",
    "t_per_k_stability_s",
    "t_per_k_generalizability_s",
)

# --- Ablation frames ---------------------------------------------------------
# A cell is one CARVE fit. Every per-cell frame starts with this key so the
# frames join on it; the study (Klein) has difficulty "" and dataset 0.
CELL_KEY: tuple[str, ...] = (
    "study",
    "difficulty",
    "dataset",
    "subsample_ratio",
    "n_resamples",
    "replicate",
)

ABLATION_CURVE_SCHEMA: tuple[str, ...] = CELL_KEY + (
    "metric_name",
    "estimator",
    "k",
    "metric_value",
    # The <measure>_se column where CARVE provides one, NaN otherwise.
    "metric_se",
)

ABLATION_SELECTION_SCHEMA: tuple[str, ...] = CELL_KEY + (
    "metric_name",
    "selected_estimator",
    "selected_k",
    # NaN for the study, which has no true k.
    "k_star",
    "ari_selected",
)

# Simulations only: labels at every candidate k, per consensus mode.
ABLATION_AT_K_SCHEMA: tuple[str, ...] = CELL_KEY + (
    "mode",
    "k",
    "ari_at_k",
    "rare_recall_at_k",
)

ABLATION_CELL_SCHEMA: tuple[str, ...] = CELL_KEY + (
    "carve_random_state",
    "n_samples",
    "fit_seconds",
    # Largest fraction of never-co-sampled (NaN) entries over the
    # configurations' consensus matrices: the stability one, built from
    # training pairs, and the generalizability one, built from test-set
    # pairs. The second is far sparser at the grid ends (a pair is never
    # co-tested with probability (1 - (1 - rho)^2)^B, 0.37 at rho 0.9 and
    # B 100), and get_labels fills those entries with 0.5 before cutting
    # the generalizability-mode labels that ari_selected scores.
    "consensus_nan_fraction",
    "consensus_generalizability_nan_fraction",
    "n_cluster_count_warnings",
)

# One row per simulated dataset; NaN in every numeric column for the study.
ABLATION_DATASET_SCHEMA: tuple[str, ...] = (
    "study",
    "difficulty",
    "dataset",
    "n_samples",
    "k_star",
    "oracle_ari",
    "rare_label",
    "rare_fraction",
)

# subsample_ratio 1.0 marks the refit reference: the full data clustered
# again with the draw's seed and scored against the base full-data fit.
ABLATION_SIMILARITY_SCHEMA: tuple[str, ...] = (
    "study",
    "difficulty",
    "dataset",
    "subsample_ratio",
    "estimator",
    "k",
    "draw",
    "ari",
)

ABLATION_SCHEMAS: dict[str, tuple[str, ...]] = {
    "curves": ABLATION_CURVE_SCHEMA,
    "selection": ABLATION_SELECTION_SCHEMA,
    "at_k": ABLATION_AT_K_SCHEMA,
    "cells": ABLATION_CELL_SCHEMA,
    "datasets": ABLATION_DATASET_SCHEMA,
    "similarity": ABLATION_SIMILARITY_SCHEMA,
}

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "joblib",
    "carve-validate",
)


def fingerprint(X: np.ndarray) -> str:
    """A digest of X's values, for checking a cache against the data it holds."""
    return hashlib.sha1(
        np.ascontiguousarray(np.asarray(X, dtype=np.float64)).tobytes()
    ).hexdigest()


def fingerprint_path(cache_path: Path) -> Path:
    """Where the digest sidecar for a cache file lives."""
    return cache_path.with_name(cache_path.name + ".x-sha1")


def check_fingerprint(cache_path: Path, X: np.ndarray) -> None:
    """Refuse to serve a cached result against a different X.

    A cached result is only valid for the matrix it was computed on. The hECA
    development embedding changes under one scale name (subsample-first until
    the pooled cache exists, pooled after), which is exactly the case a
    scale-keyed filename cannot catch. A cache written before this check
    existed has no record to compare against; it is served with a warning
    rather than discarded, since a fit can be hours of compute.

    Shared by fit_or_load_carve and run_or_load_m3c. It lives here rather
    than in either caller because a second copy of a correctness check is how
    the two drift apart.
    """
    sidecar = fingerprint_path(cache_path)
    if not sidecar.is_file():
        warnings.warn(
            f"{cache_path} carries no fingerprint of the X it was computed on, "
            "so it cannot be checked against the X passed now. Pass force=True "
            "to recompute if the data has changed since it was cached.",
            stacklevel=3,
        )
        return
    if sidecar.read_text().strip() != fingerprint(X):
        raise ValueError(
            f"{cache_path} was fit on a different X than the one passed now "
            "(the fingerprint differs). Serving it would report results for "
            "data it never saw. Pass force=True to refit on this X, or load "
            "the data the cache was fit on."
        )


def scenario_identity(scenario: Scenario) -> dict[str, Any]:
    """What defines a scenario, independent of how many cells a run draws."""
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
    }


def _canonical_config(
    scenario: Scenario, *, n_seeds: int, n_resamples: int, random_state: int
) -> dict[str, Any]:
    """The subset of a scenario that changing must invalidate a run."""
    return {
        **scenario_identity(scenario),
        "n_seeds": n_seeds,
        "n_resamples": n_resamples,
        "random_state": random_state,
    }


def config_hash(
    scenario: Scenario, *, n_seeds: int, n_resamples: int, random_state: int
) -> str:
    """Twelve-hex-character digest of everything that defines a run.

    random_state is required, not defaulted: seeds derive as
    seed + axis_idx * 10000 + random_state, so two runs differing only in
    the base seed produce entirely different simulated data and must not
    content-address to the same directory.
    """
    payload = json.dumps(
        _canonical_config(
            scenario,
            n_seeds=n_seeds,
            n_resamples=n_resamples,
            random_state=random_state,
        ),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_dir(root: Path, scenario_name: str, cfg_hash: str) -> Path:
    """Create and return the directory for one run."""
    path = Path(root) / scenario_name / cfg_hash
    path.mkdir(parents=True, exist_ok=True)
    return path


def widest_run(
    root: Path, scenario_name: str, *, require_complete: bool = True
) -> Path:
    """The run directory holding the widest sweep for a scenario.

    Run directories are named by configuration hash, so sorting them
    lexically picks an arbitrary one when a scenario has been run at more
    than one scale. Selecting on the manifest's sweep size instead means a
    reduced exploratory run never silently shadows a full one.

    This is the one resolver. The CLI's --tables path took the lexically
    last directory with a warning while the notebook took the widest sweep,
    so the two disagreed: S4 regenerated from the CLI was the 5-dataset run
    and the notebook's panel was the 20-dataset one, with nothing marking
    the difference.

    A tie raises rather than choosing. Two equally wide runs of one scenario
    differ in something the sweep size does not capture -- an anchor set, a
    random_state -- and picking either silently is how a mixture gets into a
    figure.

    require_complete skips runs whose manifest still says "running", which
    is what an interrupted run leaves behind. Pass False to inspect one.
    Manifests written before status was introduced have none at all; they
    are skipped under require_complete, since they cannot be told apart from
    an interrupted run, and are candidates like any other when it is False.
    """
    candidates: list[tuple[tuple[int, int], Path]] = []
    for path in sorted((Path(root) / scenario_name).glob("*/")):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if require_complete and manifest.get("status") != "complete":
            continue
        candidates.append(
            ((int(manifest["n_seeds"]), int(manifest["n_resamples"])), path)
        )

    if not candidates:
        raise FileNotFoundError(
            f"No {'complete ' if require_complete else ''}run for "
            f"{scenario_name!r} under {root}. Produce one with: "
            f"python -m benchmarks.run --scenario {scenario_name}"
        )

    widest = max(size for size, _ in candidates)
    matching = [path for size, path in candidates if size == widest]
    if len(matching) > 1:
        raise RuntimeError(
            f"{scenario_name}: {len(matching)} equally wide runs at "
            f"{widest[0]} datasets x B={widest[1]} "
            f"({', '.join(p.name for p in matching)}). They differ in something "
            "the sweep size does not capture; read their manifests and remove "
            "or archive the one you do not want."
        )
    return matching[0]


def _checkpoint_path(rd: Path, axis_label: str, seed: int) -> Path:
    return Path(rd) / f"cell__{axis_label}__{seed:04d}.parquet"


def _validate_against(frame: pd.DataFrame, schema: tuple[str, ...]) -> pd.DataFrame:
    """Check a frame carries exactly the schema columns, then order them."""
    missing = [column for column in schema if column not in frame.columns]
    if missing:
        raise ValueError(f"Rows are missing schema columns: {missing}.")
    extra = [column for column in frame.columns if column not in schema]
    if extra:
        raise ValueError(f"Rows carry columns outside the schema: {extra}.")
    return frame[list(schema)]


def write_checkpoint(
    rd: Path, axis_label: str, seed: int, rows: list[dict[str, Any]]
) -> Path:
    """Write one cell's rows, validating them against the schema first."""
    path = _checkpoint_path(rd, axis_label, seed)
    _validate_against(pd.DataFrame(rows), SCHEMA).to_parquet(path, index=False)
    return path


def completed_cells(rd: Path) -> set[tuple[str, int]]:
    """Return the (axis_label, seed) pairs already on disk.

    Splits only the trailing seed off the filename, so an axis label that
    itself contains a double underscore (a Study name, for instance) still
    round-trips instead of raising an unpacking error.
    """
    cells: set[tuple[str, int]] = set()
    for path in Path(rd).glob("cell__*.parquet"):
        axis_label, seed = path.stem.removeprefix("cell__").rsplit("__", 1)
        cells.add((axis_label, int(seed)))
    return cells


def read_run(rd: Path) -> pd.DataFrame:
    """Concatenate every checkpoint in a run directory."""
    paths = sorted(Path(rd).glob("cell__*.parquet"))
    if not paths:
        return pd.DataFrame(columns=list(SCHEMA))
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return frame[list(SCHEMA)]


def write_runtime_checkpoint(
    rd: Path, axis_label: str, seed: int, rows: list[dict[str, Any]]
) -> Path:
    """Write one cell's timing row to the runtime sidecar.

    Timing lives beside the metric table rather than inside it: a runtime is
    one value per cell, while the metric table has one row per metric and k,
    so folding them together would repeat the same number a hundred times.
    """
    path = Path(rd) / f"runtime__{axis_label}__{seed:04d}.parquet"
    _validate_against(pd.DataFrame(rows), RUNTIME_SCHEMA).to_parquet(path, index=False)
    return path


def read_runtimes(rd: Path) -> pd.DataFrame:
    """Concatenate every runtime checkpoint in a run directory."""
    paths = sorted(Path(rd).glob("runtime__*.parquet"))
    if not paths:
        return pd.DataFrame(columns=list(RUNTIME_SCHEMA))
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return frame[list(RUNTIME_SCHEMA)]


def ablation_dir(root: Path, name: str, scale: str, cfg_hash: str) -> Path:
    """Create and return results/runs/ablation_<name>/<scale>/<hash>/.

    The scale is in the path as well as in the hash, so a publication read
    cannot pick up a development-scale run by accident.
    """
    path = Path(root) / f"ablation_{name}" / scale / cfg_hash
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_frame(
    path: Path, rows: list[dict[str, Any]], schema: tuple[str, ...]
) -> Path:
    """Write rows as one parquet file after validating them against a schema.

    An empty row list writes an empty frame with the schema's columns; the
    study's at_k checkpoint is empty by design.

    Written atomically, the way write_manifest writes manifest.json: to a
    temporary file in the same directory, then os.replace onto the final
    path. unit_done is existence-only, so a worker killed mid-write must
    never leave a truncated file at a path resume would treat as complete.
    """
    path = Path(path)
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(schema))
    frame = _validate_against(frame, schema)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".parquet.tmp"
    )
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, index=False)
        # mkstemp creates the file owner-read-only; give the checkpoint
        # the mode manifest.json has.
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def read_frames(rd: Path) -> dict[str, pd.DataFrame]:
    """Concatenate every ablation checkpoint in a run directory, per schema.

    Files are named <schema>__<unit>.parquet. A schema with no file yet
    comes back as an empty frame with its columns, so a partial run reads.

    Empty files are left out of the concatenation: their columns come back
    from parquet as nulls, and pandas 3 no longer excludes empty entries
    when it determines a concat result dtype, so the empty at_k file every
    study cell writes would otherwise turn every at_k column object for
    the whole run. When every file is empty the schema-only frame is
    returned, as when there is none.
    """
    frames: dict[str, pd.DataFrame] = {}
    for name, schema in ABLATION_SCHEMAS.items():
        paths = sorted(Path(rd).glob(f"{name}__*.parquet"))
        parts = [part for part in map(pd.read_parquet, paths) if not part.empty]
        if not parts:
            frames[name] = pd.DataFrame(columns=list(schema))
            continue
        frames[name] = pd.concat(parts, ignore_index=True)[list(schema)]
    return frames


def provenance() -> dict[str, Any]:
    """The machine-and-code record every manifest carries."""
    return {
        "git_sha": _git_sha(),
        "package_versions": _package_versions(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cores": os.cpu_count(),
        },
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_rss_unit": "bytes",
    }


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


RUN_STATUSES: frozenset[str] = frozenset({"running", "complete"})


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
    status: str = "complete",
    timing_fits: bool = False,
) -> Manifest:
    """Assemble the provenance record for one run.

    timing_fits is recorded in the config dict, not folded into
    _canonical_config: it says whether this run's cells also timed the two
    mode-specific fits, but it changes no metric a run produces, so it must
    not affect config_hash's content address.
    """
    if status not in RUN_STATUSES:
        raise ValueError(
            f"Unknown run status {status!r}. Valid values are {sorted(RUN_STATUSES)}."
        )
    config = _canonical_config(
        scenario,
        n_seeds=n_seeds,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    config["timing_fits"] = timing_fits
    return Manifest(
        run_id=run_id,
        scenario=scenario.name,
        config_hash=config_hash,
        status=status,
        anchor_set=ACTIVE_ANCHOR_SET_NAME,
        n_seeds=n_seeds,
        n_resamples=n_resamples,
        n_jobs=n_jobs,
        random_state=random_state,
        wall_clock_s=float(wall_clock_s),
        **provenance(),
        config=config,
    )


def write_manifest(rd: Path, manifest: Manifest | Mapping[str, Any]) -> Path:
    """Write manifest.json beside the checkpoints, atomically.

    Accepts a Manifest or any mapping, so the ablation writes the same atomic file.

    Writes to a temporary file in the same directory, then os.replace onto
    manifest.json. os.replace is atomic on POSIX, so a concurrent reader --
    or a process killed mid-write -- never observes a partially written
    file; a resumed run_scenario call would otherwise be able to read a
    truncated manifest and fail. The temporary file lives in the same
    directory as the target deliberately: os.replace is only atomic within
    a single filesystem.
    """
    rd = Path(rd)
    path = rd / "manifest.json"
    fd, tmp_name = tempfile.mkstemp(dir=rd, prefix=".manifest.", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            payload = (
                manifest.to_dict() if isinstance(manifest, Manifest) else dict(manifest)
            )
            f.write(json.dumps(payload, indent=2, default=str))
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def promote(rd: Path, published_root: Path) -> Path:
    """Publish a run as committed CSV plus its manifest.

    Deliberate and explicit. A run is never promoted as a side effect of
    executing one.

    Only a run whose manifest says "complete" is publishable. run_scenario
    writes the manifest with status "running" before its pool starts, so an
    interrupted run has a manifest beside a partial set of checkpoints. A
    manifest with no status predates the field and cannot be told apart
    from an interrupted run, so it is refused too, as widest_run skips it.
    """
    rd = Path(rd)
    manifest_path = rd / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. A run without provenance is not publishable."
        )

    manifest = json.loads(manifest_path.read_text())
    status = manifest.get("status")
    if status != "complete":
        found = "no status" if status is None else f"status {status!r}"
        raise ValueError(
            f"{rd} is not a complete run: its manifest has {found}. Only a run "
            "whose manifest says 'complete' is publishable; an interrupted run's "
            "checkpoints are partial. Resume it with the command that started it, "
            "then promote."
        )

    out = Path(published_root) / manifest["scenario"]
    out.mkdir(parents=True, exist_ok=True)

    read_run(rd).to_csv(out / "results.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    runtimes = read_runtimes(rd)
    if not runtimes.empty:
        runtimes.to_csv(out / "runtimes.csv", index=False)

    return out
