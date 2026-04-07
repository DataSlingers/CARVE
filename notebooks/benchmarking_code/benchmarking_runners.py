"""Core benchmarking functions for difficulty and scaling experiments.

Two entry points:
- ``benchmark_cluster_metrics``  — sweeps difficulty levels (with live plotting)
- ``benchmark_scaling``          — sweeps a scaling axis (n_total, p, or embed_dim)

Both evaluate CARVE metrics and classical (external) metrics, recording
per-k results in a flat DataFrame.
"""

import random
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from sklearn.metrics import adjusted_rand_score

from tqdm.notebook import tqdm

from carve import CARVE

from benchmarking_config import (
    CARVE_METRICS_ALL,
    CARVE_METRICS_STABILITY,
    CARVE_METRICS_GENERALIZABILITY,
    EXTERNAL_METRICS,
    make_scaling_x_values,
)
from benchmarking_simulation_helpers import (
    parse_difficulty_and_simulate,
    parse_range_and_simulate,
)
from benchmarking_utils import (
    make_estimator_grids,
    _build_estimator,
    get_measure,
    get_rule,
)
from benchmarking_metrics import calculate_metric
from benchmarking_plotting import plot_benchmark_snapshot


# ── Shared helper: evaluate external (classical) metrics ──────────────────────


def _evaluate_external_metrics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    external_metrics: Sequence[str],
    candidate_clusters: Sequence[int],
    estimator_cls,
    estimator_param_grid: dict,
    benchmark_seed: int,
) -> tuple[list[dict], dict]:
    """Evaluate external clustering metrics for all candidate k values.

    For each metric, clusters *X* at every candidate k, computes the metric
    value, and identifies the k that maximises it.

    Returns:
        results: List of result dicts (one per metric x k combination).
            Each dict has keys: metric_name, k, metric_value, is_optimal, metric_ari.
        plotting_dict: Mapping metric_name -> {measure, k, labels, ari} for the
            optimal k (used for live snapshot visualisation).
    """
    results: list[dict] = []
    plotting_dict: dict = {}

    for metric in external_metrics:
        metric_values: list[tuple] = []
        best_value, best_k, best_labels, best_ari = -np.inf, None, None, None

        for k in candidate_clusters:
            est = _build_estimator(
                estimator_cls=estimator_cls,
                n_clusters=k,
                estimator_params=estimator_param_grid,
                random_seed=benchmark_seed,
            )
            labels = np.asarray(est.fit_predict(X), dtype=np.int32)
            ari = adjusted_rand_score(y, labels)

            value = calculate_metric(
                X,
                labels,
                metric,
                estimator_cls=estimator_cls,
                estimator_params=estimator_param_grid,
                random_state=benchmark_seed,
            )
            metric_values.append((k, value, ari))

            if value > best_value:
                best_value, best_k, best_labels, best_ari = value, k, labels, ari

        optimal_k = max(metric_values, key=lambda x: x[1])[0]

        for k, value, ari in metric_values:
            results.append(
                {
                    "metric_name": metric,
                    "k": k,
                    "metric_value": value,
                    "is_optimal": k == optimal_k,
                    "metric_ari": ari,
                }
            )

        plotting_dict[metric] = {
            "measure": metric,
            "k": best_k,
            "labels": best_labels,
            "ari": best_ari,
        }

    return results, plotting_dict


# ── Difficulty benchmark ──────────────────────────────────────────────────────


def benchmark_cluster_metrics(
    settings_by_k: Dict,
    other_settings: Dict,
    difficulty_levels: int = 10,
    n_seeds_per_dataset: int = 20,
    estimator: str = "kmeans",
    estimator_grids: Optional[list[tuple[type, dict[str, list[Any]]]]] = None,
    true_cluster_counts: Sequence[int] = (3, 4),
    candidate_clusters: Sequence[int] = range(2, 8),
    external_metrics: Sequence[str] = EXTERNAL_METRICS,
    get_snapshot: bool = False,
    snapshot_df: Optional[pd.DataFrame] = None,
    n_jobs: int = 1,
    random_state: int = 0,
) -> pd.DataFrame:
    """Benchmark clustering metrics across simulated datasets of varying difficulty.

    Sweeps over *difficulty_levels* x *true_cluster_counts* x *n_seeds_per_dataset*
    combinations.  For each, fits CARVE and evaluates all CARVE + external metrics.

    Args:
        settings_by_k: Mapping from true_k to difficulty-anchor settings
            (each containing 'easy', 'medium', 'hard' sub-dicts).
        other_settings: Shared simulation keyword arguments.
        difficulty_levels: Number of difficulty levels (interpolated between anchors).
        n_seeds_per_dataset: Number of random seeds per (difficulty, true_k) pair.
        estimator: Clustering estimator key ('kmeans', 'agglomerative', 'spectral').
        estimator_grids: Optional pre-built estimator grids (overrides *estimator*).
        true_cluster_counts: True cluster counts to simulate.
        candidate_clusters: Candidate k values to evaluate.
        external_metrics: External metrics to evaluate.
        get_snapshot: If True, randomise iteration order and return after one step.
        snapshot_df: If provided with get_snapshot, resume from this DataFrame.
        n_jobs: Number of parallel jobs for CARVE.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with one row per (difficulty, seed, true_k, metric, k).
    """
    if get_snapshot and snapshot_df is not None:
        results = snapshot_df.to_dict(orient="records")
    else:
        results = []

    rng = random.Random(random_state)
    plt.ion()

    total_steps = difficulty_levels * n_seeds_per_dataset * len(true_cluster_counts)
    pbar = tqdm(total=total_steps, desc="Benchmarking", leave=True)

    for difficulty_level in range(difficulty_levels):
        for true_k in true_cluster_counts:
            for seed in range(n_seeds_per_dataset):
                if get_snapshot:
                    difficulty_level = rng.randint(0, difficulty_levels - 1)
                    true_k = rng.choice(list(true_cluster_counts))
                    seed = rng.randint(0, n_seeds_per_dataset - 1)

                # --- 0) Deterministic seed ---
                benchmark_seed = (
                    seed
                    + ((true_k - min(true_cluster_counts)) * 100)
                    + (difficulty_level * 10000)
                    + random_state
                )
                plotting_dict: dict = {}

                # --- 1) Simulate data ---
                X, y = parse_difficulty_and_simulate(
                    settings_by_k=settings_by_k[true_k],
                    other_settings=other_settings,
                    difficulty_levels=difficulty_levels,
                    difficulty_level=difficulty_level,
                    true_cluster_count=true_k,
                    seed=benchmark_seed,
                )

                # --- 2) Baseline ARI (oracle k) ---
                estimator_grids = make_estimator_grids(
                    estimator=estimator,
                    candidate_clusters=candidate_clusters,
                )
                estimator_cls, estimator_param_grid = estimator_grids[0]

                baseline_estimator = _build_estimator(
                    estimator_cls=estimator_cls,
                    n_clusters=true_k,
                    estimator_params=estimator_param_grid,
                    random_seed=benchmark_seed,
                )
                baseline_labels = baseline_estimator.fit_predict(X)
                baseline_ari = adjusted_rand_score(y, baseline_labels)

                # --- 3) Fit CARVE and evaluate CARVE metrics ---
                carve = CARVE(
                    estimator_param_grids=estimator_grids,
                    n_jobs=n_jobs,
                    random_state=benchmark_seed,
                )
                carve.fit(X)

                # Get consensus ARIs for all candidate k
                carve_aris = []
                carve_labels_by_k = []
                for k in candidate_clusters:
                    consensus_labels = carve.get_labels(k=k)
                    carve_aris.append(adjusted_rand_score(y, consensus_labels))
                    carve_labels_by_k.append(consensus_labels)

                # Record CARVE metric results
                for carve_metric in CARVE_METRICS_ALL:
                    rule = get_rule(carve_metric)
                    measure = get_measure(carve_metric)
                    optimal_k = carve.get_k(measure=measure, rule=rule)

                    for k in candidate_clusters:
                        value = (
                            carve.estimator_results_[measure]
                            .loc[carve.estimator_results_["n_clusters"] == k]
                            .values[0]
                        )
                        results.append(
                            {
                                "axis_name": "difficulty_level",
                                "axis_value": difficulty_level,
                                "baseline_ari": baseline_ari,
                                "dataset_iteration": seed,
                                "true_k": true_k,
                                "metric_name": carve_metric,
                                "k": k,
                                "metric_value": value,
                                "is_optimal": k == optimal_k,
                                "is_correct": k == true_k,
                                "metric_ari": carve_aris[
                                    list(candidate_clusters).index(k)
                                ],
                            }
                        )

                    opt_idx = list(candidate_clusters).index(optimal_k)
                    plotting_dict[carve_metric] = {
                        "measure": carve_metric,
                        "k": optimal_k,
                        "labels": carve_labels_by_k[opt_idx],
                        "ari": carve_aris[opt_idx],
                    }

                # --- 4) Evaluate external metrics ---
                context = {
                    "axis_name": "difficulty_level",
                    "axis_value": difficulty_level,
                    "baseline_ari": baseline_ari,
                    "dataset_iteration": seed,
                    "true_k": true_k,
                }
                ext_results, ext_plotting = _evaluate_external_metrics(
                    X,
                    y,
                    external_metrics=external_metrics,
                    candidate_clusters=candidate_clusters,
                    estimator_cls=estimator_cls,
                    estimator_param_grid=estimator_param_grid,
                    benchmark_seed=benchmark_seed,
                )
                for row in ext_results:
                    row["is_correct"] = row["k"] == true_k
                    row.update(context)
                    results.append(row)
                plotting_dict.update(ext_plotting)

                # --- Live snapshot visualisation ---
                clear_output(wait=True)
                fig_pca, fig_sum = plot_benchmark_snapshot(
                    X=X,
                    results_df=pd.DataFrame(results),
                    plotting_dict=plotting_dict,
                    true_labels=y,
                    baseline_labels=baseline_labels,
                    baseline_ari=baseline_ari,
                )
                display(fig_pca)
                display(fig_sum)
                plt.close(fig_pca)
                plt.close(fig_sum)

                pbar.update(1)

                if get_snapshot:
                    return None

    pbar.close()
    return pd.DataFrame(results)


# ── Scaling benchmark ─────────────────────────────────────────────────────────


def benchmark_scaling(
    settings_by_k: Dict,
    other_settings: Dict,
    axis_name: str,
    granularity: int = 10,
    n_seeds_per_value: int = 20,
    estimator: str = "kmeans",
    true_cluster_counts: Sequence[int] = (3, 4),
    candidate_clusters: Sequence[int] = range(2, 8),
    external_metrics: Sequence[str] = EXTERNAL_METRICS,
    n_jobs: int = 1,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark CARVE and external metrics across a scaling axis.

    Sweeps one axis (n_total, p, or embed_dim) over the range defined in
    ``benchmarking_config.SCALING_RANGES`` using linear spacing.

    Stability and generalizability CARVE modes are fitted separately so
    their runtimes can be tracked independently.

    Args:
        settings_by_k: Simulation regime settings keyed by true_k, each
            containing 'start', 'middle', 'end' sub-dicts.
        other_settings: Shared simulation keyword arguments.
        axis_name: Scaling axis name ('n_total', 'p', or 'embed_dim').
        granularity: Number of linearly-spaced points along the axis.
        n_seeds_per_value: Number of seeds per (axis_value, true_k) pair.
        estimator: Clustering estimator key for CARVE grids.
        true_cluster_counts: True cluster counts to simulate.
        candidate_clusters: Candidate k values to evaluate.
        external_metrics: External metrics to evaluate.
        n_jobs: Number of parallel jobs for CARVE.
        random_state: Seed for reproducibility.

    Returns:
        (scores_df, runtimes_df) — results and per-iteration timing info.
    """
    x_values = make_scaling_x_values(axis_name, granularity)

    _carve_metrics_s = set(CARVE_METRICS_STABILITY)

    scores: list[dict] = []
    runtimes: list[dict] = []

    total_steps = granularity * n_seeds_per_value * len(true_cluster_counts)
    pbar = tqdm(
        total=total_steps, desc=f"benchmarking scaling ({axis_name})", leave=True
    )

    for i, x_value in enumerate(x_values):
        for true_k in true_cluster_counts:
            for seed in range(n_seeds_per_value):
                # --- 0) Deterministic seed ---
                benchmark_seed = (
                    seed
                    + ((true_k - min(true_cluster_counts)) * 100)
                    + (i * 10000)
                    + random_state
                )

                # --- 1) Simulate data ---
                X, y = parse_range_and_simulate(
                    settings_by_k=settings_by_k[true_k],
                    other_settings=other_settings,
                    total_stages=granularity,
                    stage=i,
                    true_cluster_count=true_k,
                    axis_name=axis_name,
                    axis_value=x_value,
                    random_state=benchmark_seed,
                )

                # --- 2) Baseline ARI (oracle k) ---
                estimator_grids = make_estimator_grids(
                    estimator=estimator,
                    candidate_clusters=candidate_clusters,
                )
                estimator_cls, estimator_param_grid = estimator_grids[0]

                baseline_estimator = _build_estimator(
                    estimator_cls=estimator_cls,
                    n_clusters=true_k,
                    estimator_params=estimator_param_grid,
                    random_seed=benchmark_seed,
                )
                baseline_labels = baseline_estimator.fit_predict(X)
                baseline_ari = adjusted_rand_score(y, baseline_labels)

                # --- 3) Fit CARVE (stability + generalizability separately) ---
                carve_s = CARVE(
                    estimator_param_grids=estimator_grids,
                    n_jobs=n_jobs,
                    random_state=benchmark_seed,
                )
                carve_g = CARVE(
                    estimator_param_grids=estimator_grids,
                    n_jobs=n_jobs,
                    random_state=benchmark_seed,
                )

                t0_s = time.perf_counter()
                carve_s.fit(X, reference_labels=y, mode="stability")
                t_carve_s = time.perf_counter() - t0_s

                t0_g = time.perf_counter()
                carve_g.fit(X, reference_labels=y, mode="generalizability")
                t_carve_g = time.perf_counter() - t0_g

                # Get consensus ARIs for all candidate k (per mode)
                carve_aris_s, carve_aris_g = [], []
                for k in candidate_clusters:
                    labels_s = carve_s.get_labels(
                        k=k, measure="stability", mode="stability"
                    )
                    labels_g = carve_g.get_labels(
                        k=k, measure="generalizability", mode="generalizability"
                    )
                    carve_aris_s.append(adjusted_rand_score(y, labels_s))
                    carve_aris_g.append(adjusted_rand_score(y, labels_g))

                # Record CARVE metric results
                all_carve = sorted(
                    _carve_metrics_s.union(CARVE_METRICS_GENERALIZABILITY)
                )
                for carve_metric in all_carve:
                    rule = get_rule(carve_metric)
                    measure = get_measure(carve_metric)

                    is_stab = carve_metric in _carve_metrics_s
                    carve = carve_s if is_stab else carve_g
                    aris = carve_aris_s if is_stab else carve_aris_g

                    optimal_k = carve.get_k(measure=measure, rule=rule)

                    for k in candidate_clusters:
                        value = (
                            carve.estimator_results_[measure]
                            .loc[carve.estimator_results_["n_clusters"] == k]
                            .values[0]
                        )
                        scores.append(
                            {
                                "axis_name": axis_name,
                                "axis_value": x_value,
                                "baseline_ari": baseline_ari,
                                "dataset_iteration": seed,
                                "true_k": true_k,
                                "metric_name": carve_metric,
                                "k": k,
                                "metric_value": value,
                                "is_optimal": k == optimal_k,
                                "is_correct": k == true_k,
                                "metric_ari": aris[list(candidate_clusters).index(k)],
                            }
                        )

                # --- 4) Evaluate external metrics ---
                context = {
                    "axis_name": axis_name,
                    "axis_value": x_value,
                    "baseline_ari": baseline_ari,
                    "dataset_iteration": seed,
                    "true_k": true_k,
                }
                ext_results, _ = _evaluate_external_metrics(
                    X,
                    y,
                    external_metrics=external_metrics,
                    candidate_clusters=candidate_clusters,
                    estimator_cls=estimator_cls,
                    estimator_param_grid=estimator_param_grid,
                    benchmark_seed=benchmark_seed,
                )
                for row in ext_results:
                    row["is_correct"] = row["k"] == true_k
                    row.update(context)
                    scores.append(row)

                # --- 5) Record runtimes ---
                n_ks = len(list(candidate_clusters))
                runtimes.append(
                    {
                        "axis_name": axis_name,
                        "axis_value": x_value,
                        "dataset_iteration": seed,
                        "true_k": true_k,
                        "n": X.shape[0],
                        "p": X.shape[1],
                        "B": n_seeds_per_value,
                        "n_jobs": n_jobs,
                        "n_test_ks": n_ks,
                        "estimator": estimator,
                        "t_carve_sec_s": t_carve_s,
                        "t_carve_sec_g": t_carve_g,
                        "t_carve_per_k_sec_s": t_carve_s / n_ks,
                        "t_carve_per_k_sec_g": t_carve_g / n_ks,
                    }
                )

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(scores), pd.DataFrame(runtimes)
