"""The single benchmark runner.

One entry point covers every simulated experiment, because a difficulty
sweep and a scaling sweep are the same experiment over different axes.

Two behaviors differ deliberately from the code this replaces.

Parallelism is inverted. The old runners parallelized five-element inner
loops with processes while the sixty-iteration outer loop ran serially, and
passed the same n_jobs to both the outer loop and CARVE, so the two nested.
Here joblib parallelizes the outer (axis x seed) loop and CARVE always gets
n_jobs=1.

reference_labels is not passed to fit. The old scaling runner passed the true
labels and the difficulty runner did not. reference_labels only permutes
cluster ids for consistency across get_labels calls (carve/api.py:728-736);
it never enters the stability or generalizability computation, and ARI is
invariant to label permutation. The divergence was cosmetic.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

from carve import CARVE

from ._artifacts import (
    build_manifest,
    completed_cells,
    config_hash,
    run_dir,
    write_checkpoint,
    write_manifest,
)
from ._cvi import calculate_cvi, select_k
from ._estimators import build_estimator, param_grids
from ._registry import (
    CARVE_METRICS_ALL,
    CVI_METRICS,
    GENERALIZABILITY_METRICS,
    PUBLISHED_RANDOM_STATE,
    metric_measure,
    metric_rule,
)
from ._simulate import simulate
from ._types import Scenario


def _labels_mode(metric_name: str) -> str:
    """Which consensus matrix a metric's labels must be cut from.

    get_labels resolves this argument through resolve_mode, and "default"
    yields run_stability=True, which selects the stability matrix. The old
    difficulty runner never passed it, so every metric's ari_at_k came from
    stability-mode labels.
    """
    return "generalizability" if metric_name in GENERALIZABILITY_METRICS else "default"


def run_cell(
    scenario: Scenario,
    *,
    axis_idx: int,
    axis_value: Any,
    axis_label: str,
    seed: int,
    run_id: str,
    random_state: int,
    n_resamples: int,
) -> list[dict[str, Any]]:
    """Run one (axis point, seed) cell and return its rows.

    Simulates, fits an oracle estimator at k_star, fits CARVE, then scores
    every CARVE metric and every classical index at every candidate k.
    """
    benchmark_seed = seed + (axis_idx * 10000) + random_state
    candidate_k = list(scenario.candidate_k)

    X, y = simulate(
        scenario, axis_value=axis_value, axis_label=axis_label, seed=benchmark_seed
    )

    oracle = build_estimator(
        scenario.estimator, n_clusters=scenario.k_star, random_state=benchmark_seed
    )
    oracle_ari = float(adjusted_rand_score(y, oracle.fit_predict(X)))

    grids = param_grids(scenario.estimator, candidate_k)
    carve = CARVE(
        estimator_param_grids=grids,
        n_resamples=n_resamples,
        n_trees=scenario.n_trees,
        n_jobs=1,
        random_state=benchmark_seed,
    )
    carve.fit(X)

    context = {
        "run_id": run_id,
        "scenario": scenario.name,
        "axis_name": scenario.axis.name,
        "axis_value": axis_value,
        "axis_label": axis_label,
        "seed": seed,
        "k_star": scenario.k_star,
        "estimator": scenario.estimator.name,
        "oracle_ari": oracle_ari,
    }

    rows: list[dict[str, Any]] = []

    # --- CARVE metrics -----------------------------------------------------
    # ari_at_k depends only on the consensus matrix a metric is cut from, so
    # labels are computed once per mode rather than once per metric.
    ari_by_mode: dict[str, dict[int, float]] = {}
    for mode in ("default", "generalizability"):
        ari_by_mode[mode] = {
            k: float(adjusted_rand_score(y, carve.get_labels(k=k, mode=mode)))
            for k in candidate_k
        }

    for metric_name in CARVE_METRICS_ALL:
        measure = metric_measure(metric_name)
        rule = metric_rule(metric_name)
        selected_k = int(carve.get_k(measure=measure, rule=rule))
        aris = ari_by_mode[_labels_mode(metric_name)]

        results = carve.estimator_results_
        for k in candidate_k:
            value = float(results[measure].loc[results["n_clusters"] == k].values[0])
            rows.append(
                {
                    **context,
                    "metric_name": metric_name,
                    "k": k,
                    "metric_value": value,
                    "is_selected": k == selected_k,
                    "selects_true_k": k == scenario.k_star,
                    "ari_at_k": aris[k],
                }
            )

    # --- Classical indices -------------------------------------------------
    labels_by_k = {
        k: np.asarray(
            build_estimator(
                scenario.estimator, n_clusters=k, random_state=benchmark_seed
            ).fit_predict(X),
            dtype=np.int32,
        )
        for k in candidate_k
    }
    cvi_ari = {k: float(adjusted_rand_score(y, labels_by_k[k])) for k in candidate_k}

    for metric_name in CVI_METRICS:
        values: list[float] = []
        errors: list[float] = []
        for k in candidate_k:
            value, error = calculate_cvi(
                X,
                labels_by_k[k],
                metric_name,
                spec=scenario.estimator,
                random_state=benchmark_seed,
            )
            values.append(value)
            errors.append(error)

        selected_k = select_k(metric_name, candidate_k, values, errors)
        for k, value in zip(candidate_k, values):
            rows.append(
                {
                    **context,
                    "metric_name": metric_name,
                    "k": k,
                    "metric_value": value,
                    "is_selected": k == selected_k,
                    "selects_true_k": k == scenario.k_star,
                    "ari_at_k": cvi_ari[k],
                }
            )

    return rows


def run_scenario(
    scenario: Scenario,
    *,
    root: Path,
    n_jobs: int = 1,
    random_state: int = PUBLISHED_RANDOM_STATE,
    n_seeds: int | None = None,
    n_resamples: int = 100,
    resume: bool = True,
    verbose: int = 0,
) -> Path:
    """Run every cell of a scenario, checkpointing as it goes.

    Returns the run directory, which is content-addressed on the scenario
    configuration so a changed anchor cannot read a stale result.

    Provenance across resumes. When resume=True and the run directory
    already has a manifest.json, this invocation reuses the run_id it
    records rather than minting a new one, so every row on disk -- from
    this invocation and every earlier one -- carries the same run_id the
    manifest reports. wall_clock_s is cumulative across resumed
    invocations: each call adds its own elapsed time to whatever the
    existing manifest already recorded, rather than overwriting it, so the
    manifest reflects the total work the run directory represents rather
    than only the most recent increment.

    resume=False recomputes every cell regardless of what is already on
    disk -- write_checkpoint overwrites each cell's file unconditionally --
    so the resulting run directory is a complete, self-consistent output of
    this single invocation. Accordingly resume=False always mints a fresh
    run_id and records only this invocation's wall_clock_s, even if a
    manifest from an earlier run already exists at the same path: that
    manifest describes a run this invocation has now fully superseded.
    """
    n_seeds = scenario.n_seeds if n_seeds is None else int(n_seeds)
    cfg_hash = config_hash(
        scenario, n_seeds=n_seeds, n_resamples=n_resamples, random_state=random_state
    )
    rd = run_dir(Path(root), scenario.name, cfg_hash)

    manifest_path = rd / "manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text())
        if resume and manifest_path.exists()
        else None
    )

    done = completed_cells(rd) if resume else set()
    pending = [
        (axis_idx, axis_value, axis_label, seed)
        for axis_idx, axis_value, axis_label in scenario.axis
        for seed in range(n_seeds)
        if (axis_label, seed) not in done
    ]

    if verbose:
        print(
            f"{scenario.name}: {len(pending)} cells to run, {len(done)} already done."
        )

    run_id = previous_manifest["run_id"] if previous_manifest else uuid.uuid4().hex[:12]
    started = time.perf_counter()

    def _one(axis_idx, axis_value, axis_label, seed):
        rows = run_cell(
            scenario,
            axis_idx=axis_idx,
            axis_value=axis_value,
            axis_label=axis_label,
            seed=seed,
            run_id=run_id,
            random_state=random_state,
            n_resamples=n_resamples,
        )
        write_checkpoint(rd, axis_label, seed, rows)

    if pending:
        Parallel(n_jobs=n_jobs)(
            delayed(_one)(*cell)
            for cell in tqdm(pending, desc=scenario.name, leave=False)
        )

    elapsed = time.perf_counter() - started
    if previous_manifest is not None:
        elapsed += float(previous_manifest["wall_clock_s"])

    write_manifest(
        rd,
        build_manifest(
            scenario,
            run_id=run_id,
            config_hash=cfg_hash,
            n_seeds=n_seeds,
            n_resamples=n_resamples,
            n_jobs=n_jobs,
            random_state=random_state,
            wall_clock_s=elapsed,
        ),
    )
    return rd
