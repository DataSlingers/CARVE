"""Core benchmarking functions for difficulty and scaling experiments.

Two entry points:
- ``benchmark_cluster_metrics``  — sweeps three difficulty anchors (easy/medium/hard)
- ``benchmark_scaling``          — sweeps three stage anchors (start/middle/end) along a scaling axis

Both evaluate CARVE metrics and classical (non_carve) metrics, recording
per-k results in a flat DataFrame.
"""

# =============================================================================
# Imports
# =============================================================================
import random
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from sklearn.metrics import adjusted_rand_score

from tqdm.notebook import tqdm

from carve import CARVE

from benchmarking_config import (
    CARVE_METRICS_ALL,
    CARVE_METRICS_STABILITY,
    CARVE_METRICS_GENERALIZABILITY,
    NON_CARVE_METRICS,
    make_scaling_x_values,
)
from benchmarking_simulation_helpers import (
    DIFFICULTY_LABELS,
    STAGE_LABELS,
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
from benchmarking_plotting import display_benchmark_snapshot


# =============================================================================
# Per-(metric, k) primitives — used by parallel evaluation loops
# =============================================================================
def _evaluate_single_metric_at_k(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    metric: str,
    k: int,
    estimator_cls,
    estimator_param_grid: dict,
    benchmark_seed: int,
) -> tuple[str, int, float, float]:
    """Compute one (metric, k) value given pre-computed labels.

    Returns:
        Tuple (metric, k, value, ari).
    """
    ari = adjusted_rand_score(y, labels)
    value = calculate_metric(
        X,
        labels,
        metric,
        estimator_cls=estimator_cls,
        estimator_params=estimator_param_grid,
        random_state=benchmark_seed,
    )
    return metric, k, value, ari


def _fit_estimator_at_k(
    X: np.ndarray,
    estimator_cls,
    n_clusters: int,
    estimator_param_grid: dict,
    benchmark_seed: int,
) -> tuple[int, np.ndarray]:
    """Fit a single estimator at one k and return (k, labels)."""
    est = _build_estimator(
        estimator_cls=estimator_cls,
        n_clusters=n_clusters,
        estimator_params=estimator_param_grid,
        random_seed=benchmark_seed,
    )
    return n_clusters, np.asarray(est.fit_predict(X), dtype=np.int32)


def _carve_labels_and_ari(
    carve,
    k: int,
    y: np.ndarray,
    **get_labels_kwargs,
) -> tuple[int, np.ndarray, float]:
    """Run ``carve.get_labels`` for one k and return (k, labels, ari)."""
    consensus_labels = carve.get_labels(k=k, **get_labels_kwargs)
    return k, consensus_labels, adjusted_rand_score(y, consensus_labels)


# =============================================================================
# Composite helper — evaluate all non-CARVE metrics across candidate k values
# =============================================================================
def _evaluate_non_carve_metrics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    non_carve_metrics: Sequence[str],
    candidate_clusters: Sequence[int],
    estimator_cls,
    estimator_param_grid: dict,
    benchmark_seed: int,
    n_jobs: int = 1,
) -> tuple[list[dict], dict]:
    """Evaluate non_carve clustering metrics for all candidate k values.

    Clusters X once per candidate k, then evaluates all metrics at each k
    in parallel.

    Returns:
        results: List of result dicts (one per metric x k combination).
            Each dict has keys: metric_name, k, metric_value, is_optimal, metric_ari.
        plotting_dict: Mapping metric_name -> {measure, k, labels, ari} for the
            optimal k (used for live snapshot visualisation).
    """
    # --- Cluster data set (one fit per k, in parallel) ---
    # KMeans uses OpenMP internally; running it under joblib threads can
    # deadlock, so use processes (joblib default) here.
    fit_results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_estimator_at_k)(
            X,
            estimator_cls,
            k,
            estimator_param_grid,
            benchmark_seed,
        )
        for k in candidate_clusters
    )
    labels_by_k: dict[int, np.ndarray] = dict(fit_results)

    # --- Evaluate (metric, k) pairs ---
    # gap_statistic refits the estimator in each task, so use processes here too.
    tasks = [
        delayed(_evaluate_single_metric_at_k)(
            X,
            y,
            labels_by_k[k],
            metric,
            k,
            estimator_cls,
            estimator_param_grid,
            benchmark_seed,
        )
        for metric in non_carve_metrics
        for k in candidate_clusters
    ]
    raw = Parallel(n_jobs=n_jobs)(tasks)

    # --- Group by metric, find optimum, build outputs ---
    by_metric: dict[str, list[tuple[int, float, float]]] = {
        m: [] for m in non_carve_metrics
    }
    for metric, k, value, ari in raw:
        by_metric[metric].append((k, value, ari))

    results: list[dict] = []
    plotting_dict: dict = {}

    for metric in non_carve_metrics:
        metric_values = by_metric[metric]
        optimal_k = max(metric_values, key=lambda x: x[1])[0]

        best_value, best_k, best_ari = -np.inf, None, None
        for k, value, ari in metric_values:
            if value > best_value:
                best_value, best_k, best_ari = value, k, ari
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
            "labels": labels_by_k[best_k],
            "ari": best_ari,
        }

    return results, plotting_dict


# =============================================================================
# Difficulty benchmark
# =============================================================================
def benchmark_cluster_metrics(
    anchor_settings: Dict,
    other_settings: Dict,
    true_cluster_count: int = 5,
    n_seeds_per_dataset: int = 20,
    estimator: str = "kmeans",
    estimator_grids: Optional[list[tuple[type, dict[str, list[Any]]]]] = None,
    n_trees: int = 100,
    candidate_clusters: Sequence[int] = range(3, 8),
    non_carve_metrics: Sequence[str] = NON_CARVE_METRICS,
    get_snapshot: bool = False,
    snapshot_df: Optional[pd.DataFrame] = None,
    n_jobs: int = 1,
    random_state: int = 0,
) -> pd.DataFrame:
    """Benchmark clustering metrics across the three difficulty anchors.

    Sweeps over (easy/medium/hard) x *n_seeds_per_dataset* combinations.
    For each, fits CARVE and evaluates all CARVE + non_carve metrics.

    Args:
        anchor_settings: Mapping ``{"easy": {...}, "medium": {...}, "hard": {...}}``
            of calibrated parameter dicts.
        other_settings: Shared simulation keyword arguments.
        n_seeds_per_dataset: Number of random seeds per difficulty anchor.
        estimator: Clustering estimator key ('kmeans', 'agglomerative', 'spectral').
        estimator_grids: Optional pre-built estimator grids (overrides *estimator*).
        n_trees: Number of trees for CARVE's random forest classifier.
        true_cluster_count: True cluster count to simulate (k*).
        candidate_clusters: Candidate k values to evaluate.
        non_carve_metrics: non_carve metrics to evaluate.
        get_snapshot: If True, randomise iteration order and return after one step.
        snapshot_df: If provided with get_snapshot, resume from this DataFrame.
        n_jobs: Number of parallel jobs for CARVE.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with one row per (difficulty, seed, metric, k).
    """
    if get_snapshot and snapshot_df is not None:
        results = snapshot_df.to_dict(orient="records")
    else:
        results = []

    rng = random.Random(random_state)
    plt.ion()

    n_anchors = len(DIFFICULTY_LABELS)
    total_steps = n_anchors * n_seeds_per_dataset
    pbar = tqdm(total=total_steps, desc="Benchmarking", leave=True)
    snapshot_handles: dict = {"pca": None, "summary": None}

    for difficulty_idx, difficulty in enumerate(DIFFICULTY_LABELS):
        for seed in range(n_seeds_per_dataset):
            if get_snapshot:
                difficulty_idx = rng.randint(0, n_anchors - 1)
                difficulty = DIFFICULTY_LABELS[difficulty_idx]
                seed = rng.randint(0, n_seeds_per_dataset - 1)

            # --- 0) Deterministic seed ---
            benchmark_seed = seed + (difficulty_idx * 10000) + random_state
            plotting_dict: dict = {}

            # --- 1) Simulate data ---
            X, y = parse_difficulty_and_simulate(
                anchor_settings=anchor_settings,
                other_settings=other_settings,
                difficulty=difficulty,
                true_cluster_count=true_cluster_count,
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
                n_clusters=true_cluster_count,
                estimator_params=estimator_param_grid,
                random_seed=benchmark_seed,
            )
            baseline_labels = baseline_estimator.fit_predict(X)
            baseline_ari = adjusted_rand_score(y, baseline_labels)

            # --- 3) Fit CARVE and evaluate CARVE metrics ---
            carve = CARVE(
                estimator_param_grids=estimator_grids,
                n_trees=n_trees,
                n_jobs=n_jobs,
                random_state=benchmark_seed,
            )
            carve.fit(X)

            # Get consensus ARIs for all candidate k (parallel over k)
            carve_results = Parallel(n_jobs=n_jobs)(
                delayed(_carve_labels_and_ari)(carve, k, y) for k in candidate_clusters
            )
            carve_labels_by_k = [labels for _, labels, _ in carve_results]
            carve_aris = [ari for _, _, ari in carve_results]

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
                            "axis_value": difficulty_idx,
                            "difficulty": difficulty,
                            "baseline_ari": baseline_ari,
                            "dataset_iteration": seed,
                            "true_k": true_cluster_count,
                            "metric_name": carve_metric,
                            "k": k,
                            "metric_value": value,
                            "is_optimal": k == optimal_k,
                            "is_correct": k == true_cluster_count,
                            "metric_ari": carve_aris[list(candidate_clusters).index(k)],
                        }
                    )

                opt_idx = list(candidate_clusters).index(optimal_k)
                plotting_dict[carve_metric] = {
                    "measure": carve_metric,
                    "k": optimal_k,
                    "labels": carve_labels_by_k[opt_idx],
                    "ari": carve_aris[opt_idx],
                }

            # --- 4) Evaluate non_carve metrics ---
            context = {
                "axis_name": "difficulty_level",
                "axis_value": difficulty_idx,
                "difficulty": difficulty,
                "baseline_ari": baseline_ari,
                "dataset_iteration": seed,
                "true_k": true_cluster_count,
            }
            ext_results, ext_plotting = _evaluate_non_carve_metrics(
                X,
                y,
                non_carve_metrics=non_carve_metrics,
                candidate_clusters=candidate_clusters,
                estimator_cls=estimator_cls,
                estimator_param_grid=estimator_param_grid,
                benchmark_seed=benchmark_seed,
                n_jobs=n_jobs,
            )
            for row in ext_results:
                row["is_correct"] = row["k"] == true_cluster_count
                row.update(context)
                results.append(row)
            plotting_dict.update(ext_plotting)

            # --- Live snapshot visualisation ---
            display_benchmark_snapshot(
                snapshot_handles,
                X=X,
                results_df=pd.DataFrame(results),
                plotting_dict=plotting_dict,
                true_labels=y,
                baseline_labels=baseline_labels,
                baseline_ari=baseline_ari,
            )

            pbar.update(1)

            if get_snapshot:
                return None

    pbar.close()
    return pd.DataFrame(results)


# =============================================================================
# Scaling benchmark
# =============================================================================
def benchmark_scaling(
    anchor_settings: Dict,
    other_settings: Dict,
    axis_name: str,
    n_seeds_per_value: int = 20,
    estimator: str = "kmeans",
    true_cluster_count: int = 5,
    candidate_clusters: Sequence[int] = range(3, 8),
    non_carve_metrics: Sequence[str] = NON_CARVE_METRICS,
    n_jobs: int = 1,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark CARVE and non_carve metrics across three scaling-axis stages.

    Iterates the (start/middle/end) anchors over the three points (low, mid,
    high) of the requested axis, defined in ``benchmarking_config.SCALING_RANGES``.

    Stability and generalizability CARVE modes are fitted separately so
    their runtimes can be tracked independently.

    Args:
        anchor_settings: Mapping ``{"start": {...}, "middle": {...}, "end": {...}}``
            of calibrated parameter dicts.
        other_settings: Shared simulation keyword arguments.
        axis_name: Scaling axis name ('n_total', 'p', or 'embed_dim').
        n_seeds_per_value: Number of seeds per stage anchor.
        estimator: Clustering estimator key for CARVE grids.
        true_cluster_count: True cluster count to simulate (k*).
        candidate_clusters: Candidate k values to evaluate.
        non_carve_metrics: non_carve metrics to evaluate.
        n_jobs: Number of parallel jobs for CARVE.
        random_state: Seed for reproducibility.

    Returns:
        (scores_df, runtimes_df) — results and per-iteration timing info.
    """
    x_values = make_scaling_x_values(axis_name, granularity=len(STAGE_LABELS))
    if len(x_values) != len(STAGE_LABELS):
        raise ValueError(
            f"Expected {len(STAGE_LABELS)} axis values, got {len(x_values)}"
        )

    _carve_metrics_s = set(CARVE_METRICS_STABILITY)

    scores: list[dict] = []
    runtimes: list[dict] = []

    n_stages = len(STAGE_LABELS)
    total_steps = n_stages * n_seeds_per_value
    pbar = tqdm(
        total=total_steps, desc=f"benchmarking scaling ({axis_name})", leave=True
    )

    for stage_idx, (stage_label, x_value) in enumerate(zip(STAGE_LABELS, x_values)):
        for seed in range(n_seeds_per_value):
            # --- 0) Deterministic seed ---
            benchmark_seed = seed + (stage_idx * 10000) + random_state

            # --- 1) Simulate data ---
            X, y = parse_range_and_simulate(
                anchor_settings=anchor_settings,
                other_settings=other_settings,
                stage_label=stage_label,
                true_cluster_count=true_cluster_count,
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
                n_clusters=true_cluster_count,
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

            # Get consensus ARIs for all candidate k (parallel over k, per mode)
            results_s = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_carve_labels_and_ari)(
                    carve_s, k, y, measure="stability", mode="stability"
                )
                for k in candidate_clusters
            )
            results_g = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_carve_labels_and_ari)(
                    carve_g, k, y, measure="generalizability", mode="generalizability"
                )
                for k in candidate_clusters
            )
            carve_aris_s = [ari for _, _, ari in results_s]
            carve_aris_g = [ari for _, _, ari in results_g]

            # Record CARVE metric results
            all_carve = sorted(_carve_metrics_s.union(CARVE_METRICS_GENERALIZABILITY))
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
                            "stage": stage_label,
                            "baseline_ari": baseline_ari,
                            "dataset_iteration": seed,
                            "true_k": true_cluster_count,
                            "metric_name": carve_metric,
                            "k": k,
                            "metric_value": value,
                            "is_optimal": k == optimal_k,
                            "is_correct": k == true_cluster_count,
                            "metric_ari": aris[list(candidate_clusters).index(k)],
                        }
                    )

            # --- 4) Evaluate non_carve metrics ---
            context = {
                "axis_name": axis_name,
                "axis_value": x_value,
                "stage": stage_label,
                "baseline_ari": baseline_ari,
                "dataset_iteration": seed,
                "true_k": true_cluster_count,
            }
            ext_results, _ = _evaluate_non_carve_metrics(
                X,
                y,
                non_carve_metrics=non_carve_metrics,
                candidate_clusters=candidate_clusters,
                estimator_cls=estimator_cls,
                estimator_param_grid=estimator_param_grid,
                benchmark_seed=benchmark_seed,
                n_jobs=n_jobs,
            )
            for row in ext_results:
                row["is_correct"] = row["k"] == true_cluster_count
                row.update(context)
                scores.append(row)

            # --- 5) Record runtimes ---
            n_ks = len(list(candidate_clusters))
            runtimes.append(
                {
                    "axis_name": axis_name,
                    "axis_value": x_value,
                    "stage": stage_label,
                    "dataset_iteration": seed,
                    "true_k": true_cluster_count,
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
