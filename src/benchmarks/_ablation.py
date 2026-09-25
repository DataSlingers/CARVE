"""The rho/B ablation runner: configuration hash, unit executors, run loop.

Each unit runs in a loky worker, which computes the rows for the frames it
owns and writes them itself (_run_and_write); the parent only schedules
units and writes the manifest. Every CARVE fit runs on one worker with its
forest capped at the worker's share of the cores (see _run.cpu_cap).
Nothing here saves a fitted CARVE: 120 Klein fits of n-by-n consensus
matrices would cost far more disk than the rows the analysis reads, and
the fit cache's filename does not distinguish rho or the seed.
"""

import dataclasses
import hashlib
import json
import time
import uuid
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

from carve import CARVE
from carve._selection import select_best_row_by_rule
from carve._utils import split_subsample_indices

from ._ablation_cells import (
    REFERENCE_RATIO,
    Cell,
    Unit,
    carve_seed,
    dataset_seed,
    enumerate_units,
    scenario_at_scale,
    similarity_seed,
    unit_done,
    unit_paths,
)
from ._artifacts import (
    ABLATION_SCHEMAS,
    ablation_dir,
    provenance,
    scenario_identity,
    write_frame,
    write_manifest,
)
from ._estimators import ESTIMATOR_CLASSES, build_estimator, param_grids
from ._registry import (
    ACTIVE_ANCHOR_SET_NAME,
    CARVE_METRICS_ALL,
    PUBLISHED_RANDOM_STATE,
    REPLICATE_SEED_SPACING,
    SCENARIOS,
    SIMILARITY_SEED_OFFSET,
    metric_measure,
    metric_rule,
)
from ._run import (
    _labels_mode,
    fit_carve,
    labels_by_mode,
    smallest_cluster,
    thread_cap_for,
)
from ._simulate import simulate
from ._studies import STUDIES, load_study, resolve_scale, study_model_grids
from ._types import Ablation

#: The study fits with CARVE's own tree count, as fit_or_load_carve does.
CARVE_N_TREES: int = next(
    field.default for field in fields(CARVE) if field.name == "n_trees"
)


# --- Configuration ------------------------------------------------------------
def ablation_config(ablation: Ablation, scale: str) -> dict[str, Any]:
    """Everything that defines a run at one scale.

    Only the chosen scale is included, so adding or editing another scale
    cannot invalidate a publication run.
    """
    sc = ablation.scales[scale]
    study = STUDIES[ablation.study]
    return {
        "ablation": {
            k: v for k, v in dataclasses.asdict(ablation).items() if k != "scales"
        },
        "scale": {scale: dataclasses.asdict(sc)},
        "scenarios": {
            name: scenario_identity(scenario_at_scale(SCENARIOS[name], sc))
            for name in ablation.scenarios
        },
        "study": {
            "name": study.name,
            "grids": repr(study_model_grids(study)),
            "scale": sc.study_scale,
            "resolved": resolve_scale(study, sc.study_scale),
            "not_two": study.not_two,
        },
        "random_state": PUBLISHED_RANDOM_STATE,
        "replicate_seed_spacing": REPLICATE_SEED_SPACING,
        "similarity_seed_offset": SIMILARITY_SEED_OFFSET,
        "anchor_set": ACTIVE_ANCHOR_SET_NAME,
    }


def ablation_hash(ablation: Ablation, scale: str) -> str:
    payload = json.dumps(ablation_config(ablation, scale), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# --- Data ---------------------------------------------------------------------
def _simulated(cell: Cell, ablation: Ablation, scale: str):
    """(X, y, scenario, base seed) of a simulated cell's dataset."""
    scenario = scenario_at_scale(SCENARIOS[cell.study], ablation.scales[scale])
    axis_idx = scenario.axis.labels.index(cell.difficulty)
    base = dataset_seed(cell, ablation=ablation)
    X, y = simulate(
        scenario,
        axis_value=scenario.axis.values[axis_idx],
        axis_label=cell.difficulty,
        seed=base,
    )
    return X, y, scenario, base


def _dataset_fragment(cell: Cell) -> dict[str, Any]:
    return {
        "study": cell.study,
        "difficulty": cell.difficulty,
        "dataset": int(cell.dataset),
    }


# --- Unit executors -----------------------------------------------------------
def dataset_rows(
    unit: Unit, *, ablation: Ablation, scale: str, data: tuple | None
) -> list[dict[str, Any]]:
    """One row: sample count, and for a simulation its truth and oracle."""
    cell = unit.cell
    if cell.study == ablation.study:
        X, _ = data
        return [
            {
                **_dataset_fragment(cell),
                "n_samples": int(X.shape[0]),
                "k_star": np.nan,
                "oracle_ari": np.nan,
                "rare_label": np.nan,
                "rare_fraction": np.nan,
            }
        ]
    X, y, scenario, base = _simulated(cell, ablation, scale)
    oracle = build_estimator(
        scenario.estimator, n_clusters=scenario.k_star, random_state=base
    )
    rare_label, rare_fraction = smallest_cluster(y)
    return [
        {
            **_dataset_fragment(cell),
            "n_samples": int(X.shape[0]),
            "k_star": float(scenario.k_star),
            "oracle_ari": float(adjusted_rand_score(y, oracle.fit_predict(X))),
            "rare_label": float(rare_label),
            "rare_fraction": float(rare_fraction),
        }
    ]


def similarity_rows(
    unit: Unit, *, ablation: Ablation, scale: str, data: tuple | None
) -> list[dict[str, Any]]:
    """Subsample-versus-full ARI at every candidate k for one dataset and rho.

    F is the full-data fit with the base seed (for a simulation, exactly
    run_cell's labels_by_k). Draw m subsamples with similarity_seed(base, m)
    and clusters it with the same estimator, k and seed; its ARI is taken
    against F restricted to the draw. At REFERENCE_RATIO the full data is
    refit with the draw's seed instead and scored against F.
    """
    cell = unit.cell
    sc = ablation.scales[scale]
    if cell.study == ablation.study:
        X, _ = data
        study = STUDIES[ablation.study]
        specs = (study.estimator, *study.partners)
        candidate_k = study.candidate_k
        base = dataset_seed(cell, ablation=ablation)
    else:
        X, _, scenario, base = _simulated(cell, ablation, scale)
        specs = (scenario.estimator,)
        candidate_k = scenario.candidate_k
    X = np.asarray(X)
    n = X.shape[0]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        estimator_name = ESTIMATOR_CLASSES[spec.name].__name__
        for k in candidate_k:
            full = build_estimator(spec, n_clusters=k, random_state=base).fit_predict(X)
            for draw in range(sc.similarity_draws):
                seed = similarity_seed(base, draw)
                estimator = build_estimator(spec, n_clusters=k, random_state=seed)
                if cell.subsample_ratio == REFERENCE_RATIO:
                    ari = adjusted_rand_score(full, estimator.fit_predict(X))
                else:
                    idx, _ = split_subsample_indices(
                        n, subsample_ratio=cell.subsample_ratio, random_state=seed
                    )
                    ari = adjusted_rand_score(full[idx], estimator.fit_predict(X[idx]))
                rows.append(
                    {
                        **_dataset_fragment(cell),
                        "subsample_ratio": float(cell.subsample_ratio),
                        "estimator": estimator_name,
                        "k": int(k),
                        "draw": int(draw),
                        "ari": float(ari),
                    }
                )
    return rows


def cell_rows(
    unit: Unit,
    *,
    ablation: Ablation,
    scale: str,
    data: tuple | None,
    thread_cap: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """Fit CARVE once and return the curves, selection, at_k and cells rows."""
    cell = unit.cell
    if cell.study == ablation.study:
        X, y = data
        study = STUDIES[ablation.study]
        grids = study_model_grids(study)
        candidate_k = tuple(study.candidate_k)
        not_two = study.not_two
        n_trees = CARVE_N_TREES
        k_star = None
        rare_label = None
    else:
        X, y, scenario, _ = _simulated(cell, ablation, scale)
        grids = param_grids(scenario.estimator, list(scenario.candidate_k))
        candidate_k = tuple(scenario.candidate_k)
        not_two = False
        n_trees = scenario.n_trees
        k_star = scenario.k_star
        rare_label, _ = smallest_cluster(y)
    X = np.asarray(X)
    y = np.asarray(y)
    seed = carve_seed(cell, ablation=ablation)

    fit = fit_carve(
        X,
        grids=grids,
        n_resamples=cell.n_resamples,
        n_trees=n_trees,
        random_state=seed,
        subsample_ratio=cell.subsample_ratio,
        thread_cap=thread_cap,
    )
    carve = fit.carve
    results = carve.estimator_results_
    key = cell.key()

    curves: list[dict[str, Any]] = []
    for metric_name in CARVE_METRICS_ALL:
        measure = metric_measure(metric_name)
        se_col = f"{measure}_se"
        for _, row in results.iterrows():
            curves.append(
                {
                    **key,
                    "metric_name": metric_name,
                    "estimator": str(row["estimator"]),
                    "k": int(row["n_clusters"]),
                    "metric_value": float(row[measure]),
                    "metric_se": float(row[se_col])
                    if se_col in results.columns
                    else np.nan,
                }
            )

    selection: list[dict[str, Any]] = []
    for metric_name in CARVE_METRICS_ALL:
        measure = metric_measure(metric_name)
        rule = metric_rule(metric_name)
        if results[measure].isna().all():
            # A degenerate resampling draw (small B, small n, or an unlucky
            # seed) can leave one sample with zero consensus co-occurrences,
            # which makes this measure NaN for every configuration -- see
            # CARVE's stability_gini_scores_/stability_ce_scores_ .mean(),
            # not .nanmean(). select_best_row_by_rule's idxmax then raises on
            # an all-NaN column. That is CARVE's own behavior (src/carve/ is
            # out of scope here), so this cell records an unselected row for
            # the metric instead of crashing the whole run.
            warnings.warn(
                f"Cell {key}: metric {metric_name!r} is NaN for every "
                f"configuration (measure {measure!r} could not be scored); "
                "recording an unselected row instead of raising.",
                stacklevel=2,
            )
            selection.append(
                {
                    **key,
                    "metric_name": metric_name,
                    "selected_estimator": np.nan,
                    "selected_k": np.nan,
                    "k_star": np.nan if k_star is None else float(k_star),
                    "ari_selected": np.nan,
                }
            )
            continue
        row = select_best_row_by_rule(
            results, measure=measure, rule=rule, not_two=not_two
        )
        labels = carve.get_labels(
            measure=measure, rule=rule, not_two=not_two, mode=_labels_mode(metric_name)
        )
        selection.append(
            {
                **key,
                "metric_name": metric_name,
                "selected_estimator": str(row["estimator"]),
                "selected_k": int(row["n_clusters"]),
                "k_star": np.nan if k_star is None else float(k_star),
                "ari_selected": float(adjusted_rand_score(y, labels)),
            }
        )

    at_k: list[dict[str, Any]] = []
    if k_star is not None:
        scores = labels_by_mode(
            carve, y, candidate_k=candidate_k, rare_label=rare_label
        )
        for mode, per_k in scores.items():
            for k, entry in per_k.items():
                at_k.append(
                    {
                        **key,
                        "mode": mode,
                        "k": int(k),
                        "ari_at_k": entry["ari"],
                        "rare_recall_at_k": entry["rare_recall"],
                    }
                )

    # Both consensus matrices, not only the stability one: the
    # generalizability consensus is built from test-set pairs and is far
    # sparser at the grid ends, and it is the matrix the generalizability-
    # mode labels behind ari_selected and ari_at_k are cut from. Nothing
    # fitted is saved, so this cannot be recovered after the run.
    cells = [
        {
            **key,
            "carve_random_state": int(seed),
            "n_samples": int(X.shape[0]),
            "fit_seconds": fit.fit_seconds,
            "consensus_nan_fraction": _max_nan_fraction(carve.consensus_matrices_),
            "consensus_generalizability_nan_fraction": _max_nan_fraction(
                carve.consensus_generalizability_matrices_
            ),
            "n_cluster_count_warnings": int(fit.n_cluster_count_warnings),
        }
    ]
    return {"curves": curves, "selection": selection, "at_k": at_k, "cells": cells}


def _max_nan_fraction(matrices) -> float:
    """Largest NaN fraction over one fit's per-configuration consensus matrices."""
    return max(float(np.isnan(np.asarray(M, dtype=float)).mean()) for M in matrices)


def run_unit(
    unit: Unit,
    *,
    ablation: Ablation,
    scale: str,
    data: tuple | None,
    thread_cap: int | None,
) -> dict[str, list[dict[str, Any]]]:
    if unit.kind == "dataset":
        return {
            "datasets": dataset_rows(unit, ablation=ablation, scale=scale, data=data)
        }
    if unit.kind == "similarity":
        return {
            "similarity": similarity_rows(
                unit, ablation=ablation, scale=scale, data=data
            )
        }
    if unit.kind == "cell":
        return cell_rows(
            unit, ablation=ablation, scale=scale, data=data, thread_cap=thread_cap
        )
    raise ValueError(f"Unknown unit kind {unit.kind!r}.")


def _run_and_write(unit, rd, ablation, scale, data, thread_cap) -> None:
    # Module level, not a closure: joblib decides whether to memmap an
    # array argument by its size (max_nbytes), not by whether the callable
    # wrapping it is a closure. This stays at module scope so it pickles by
    # qualified name for any joblib backend, and so run_unit is a global
    # tests can monkeypatch (see test_ablation.py's resume test).
    frames = run_unit(
        unit, ablation=ablation, scale=scale, data=data, thread_cap=thread_cap
    )
    for name, path in unit_paths(rd, unit).items():
        write_frame(path, frames[name], ABLATION_SCHEMAS[name])


# --- Runner -------------------------------------------------------------------
def run_ablation(
    ablation: Ablation,
    *,
    scale: str | None = None,
    root: Path,
    n_jobs: int = -1,
    resume: bool = True,
    verbose: int = 0,
    units: list[Unit] | None = None,
) -> Path:
    """Run every unit of an ablation at a scale, checkpointing as it goes.

    Returns the run directory, results/runs/ablation_<name>/<scale>/<hash>/.
    Provenance across resumes follows run_scenario: a resumed run keeps the
    manifest's run_id and adds its wall clock; resume=False mints a new id.
    units overrides what runs (the timing batch); resume still applies.

    The manifest is written twice: with status "running" before the pool
    starts, carrying the wall clock accumulated so far, and with status
    "complete" and the cumulative wall clock once every unit is done. An
    interrupted run therefore still has a manifest -- the notebook locates
    a run by it, and the publication cost is derived from its wall clock --
    and a resume reads run_id and the previous wall clock from whichever
    of the two it finds.
    """
    scale = ablation.default_scale if scale is None else scale
    if scale not in ablation.scales:
        raise ValueError(
            f"Ablation {ablation.name!r} has no scale {scale!r}; declared scales "
            f"are {sorted(ablation.scales)}."
        )
    sc = ablation.scales[scale]
    cfg_hash = ablation_hash(ablation, scale)
    rd = ablation_dir(Path(root), ablation.name, scale, cfg_hash)
    workers, thread_cap = thread_cap_for(n_jobs)

    manifest_path = rd / "manifest.json"
    previous = None
    if resume and manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(
                f"Could not read manifest at {manifest_path} ({exc}); "
                "continuing this run with fresh provenance.",
                stacklevel=2,
            )

    all_units = enumerate_units(ablation, scale) if units is None else list(units)
    pending = [u for u in all_units if not (resume and unit_done(rd, u))]

    study_data = None
    if any(u.cell.study == ablation.study for u in pending):
        X, y, _ = load_study(STUDIES[ablation.study], scale=sc.study_scale)
        study_data = (np.asarray(X), np.asarray(y))

    if verbose:
        print(
            f"ablation {ablation.name} ({scale}): {len(pending)} units to run, "
            f"{len(all_units) - len(pending)} already done; {workers} workers, "
            f"{thread_cap} threads per fit."
        )

    run_id = previous["run_id"] if previous else uuid.uuid4().hex[:12]
    carried = float(previous["wall_clock_s"]) if previous is not None else 0.0

    def _manifest(status: str, wall_clock_s: float) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "status": status,
            "ablation": ablation.name,
            "scale": scale,
            "config_hash": cfg_hash,
            "anchor_set": ACTIVE_ANCHOR_SET_NAME,
            "n_jobs": int(n_jobs),
            "workers": int(workers),
            "thread_cap": int(thread_cap),
            "cores": int(cpu_count()),
            "n_units": len(all_units),
            "wall_clock_s": float(wall_clock_s),
            "config": ablation_config(ablation, scale),
            **provenance(),
        }

    write_manifest(rd, _manifest("running", carried))
    started = time.perf_counter()
    if pending:
        Parallel(n_jobs=n_jobs)(
            delayed(_run_and_write)(
                unit,
                rd,
                ablation,
                scale,
                study_data if unit.cell.study == ablation.study else None,
                thread_cap,
            )
            for unit in tqdm(pending, desc=f"ablation {ablation.name}", leave=False)
        )
    elapsed = carried + (time.perf_counter() - started)
    write_manifest(rd, _manifest("complete", elapsed))
    return rd
