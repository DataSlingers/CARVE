import time

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from sklearn.metrics import adjusted_rand_score

from tqdm.notebook import tqdm

from carve import CARVE

from benchmarking_simulation_helpers import (
    parse_difficulty_and_simulate,
    parse_range_and_simulate,
)
from benchmarking_utils import (
    make_estimator_grids,
    _build_estimator,
    get_measure,
    get_rule,
    _pick_first,
)
from benchmarking_metrics import calculate_metric
from benchmarking_plotting import plot_benchmark_snapshot
import random


def benchmark_cluster_metrics(
    settings_by_k: Dict,
    other_settings: Dict,
    difficulty_levels: int = 10,
    n_seeds_per_dataset: int = 20,
    estimator: str = "kmeans",
    spectral_quant: float = 0.5,
    estimator_grids: Optional[list[tuple[type, dict[str, list[Any]]]]] = None,
    true_cluster_counts: Sequence[int] = (3, 4),
    candidate_clusters: Sequence[int] = range(2, 8),
    external_metrics: Sequence[str] = (
        "silhouette",
        "gap",
        "davies_bouldin",
        "calinski_harabasz",
    ),
    get_snapshot: bool = False,
    snapshot_df: Optional[pd.DataFrame] = None,
    n_jobs: int = 1,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Benchmarks clustering metrics across simulated datasets and configurations.

    Args:
        - settings_by_k (Dict): Mapping from true_k to simulation settings.
        - other_settings (Dict): Shared simulation settings.
        - difficulty_levels (int): Number of difficulty levels/datasets.
        - n_seeds_per_dataset (int): Number of seeds per dataset family.
        - estimator (str): Clustering estimator key ('kmeans', 'agglomerative', 'spectral').
        - spectral_quant (float): Quantile used for spectral gamma estimation (default: 0.5).
        - estimator_grids (Optional[list[tuple[type, dict]]]): Optional pre-built estimator grids.
        - true_cluster_counts (Sequence[int]): True cluster counts to simulate.
        - candidate_clusters (Sequence[int]): Candidate k values to evaluate.
        - external_metrics (Sequence[str]): External metrics to evaluate.
        - n_jobs (int): Number of jobs for CARVE.
        - random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Benchmarking results per metric, dataset, and configuration.
    """
    if get_snapshot and snapshot_df is not None:
        results = snapshot_df.to_dict(orient="records")
    else:
        results = []

    rng = random.Random(random_state)

    plt.ion()

    CARVE_AVAILABLE_METRICS = [
        "ari_stability",
        "ari_generalizability",
        "ari_average",
        "ari_stability_1se",
        "ari_generalizability_1se",
        "ari_average_1se",
        "ari_stability_quant",
        "ari_generalizability_quant",
        "ari_average_quant",
        "consensus_pac_stability",
        "consensus_gini_stability",
        "consensus_ce_stability",
        "misclassification_generalizability",
    ]

    total_steps = difficulty_levels * n_seeds_per_dataset * len(true_cluster_counts)
    pbar = tqdm(total=total_steps, desc="Benchmarking", leave=True)

    for difficulty_level in range(difficulty_levels):
        for true_k in true_cluster_counts:
            for seed in range(n_seeds_per_dataset):
                # difficulty_level, true_k, seed = rng.randint(0, difficulty_levels - 1), rng.choice(list(true_cluster_counts)), rng.randint(0, n_seeds_per_dataset - 1) if get_snapshot else difficulty_level, true_k, seed  # Pick a random combination for snapshot
                if get_snapshot:
                    difficulty_level = rng.randint(0, difficulty_levels - 1)
                    true_k = rng.choice(list(true_cluster_counts))
                    seed = rng.randint(0, n_seeds_per_dataset - 1)

                # --- 0) Set seed ---
                benchmark_seed = (
                    seed
                    + ((true_k - min(true_cluster_counts)) * 100)
                    + (difficulty_level * 10000)
                    + random_state
                )
                plotting_dict = {}  # Plotting

                # --- 1) Simulate data ---
                X, y = parse_difficulty_and_simulate(
                    settings_by_k=settings_by_k[true_k],
                    other_settings=other_settings,
                    difficulty_levels=difficulty_levels,
                    difficulty_level=difficulty_level,
                    true_cluster_count=true_k,
                    seed=benchmark_seed,
                )

                # --- 2) Get baseline ARI ---
                estimator_grids = make_estimator_grids(
                    estimator=estimator,
                    candidate_clusters=candidate_clusters,
                    spectral_quant=spectral_quant,
                    X=X,
                    random_state=benchmark_seed,
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

                # --- 3) Fit and evaluate CARVE ---
                # 3.1) Fitting CARVE
                carve = CARVE(
                    estimator_param_grids=estimator_grids,
                    n_jobs=n_jobs,
                    random_state=benchmark_seed,
                )

                carve.fit(X)

                # 3.2) Get ARIs for all k
                carve_aris = []
                carve_labels_by_k = []  # Plotting
                for k in candidate_clusters:
                    carve_consensus_labels = carve.get_labels(
                        k=k
                    )  # providing measure and rule is not necessary here as there is only a single option for every k
                    carve_aris.append(adjusted_rand_score(y, carve_consensus_labels))
                    carve_labels_by_k.append(carve_consensus_labels)  # Plotting

                # 3.3) Evaluating CARVE metrics
                for carve_metric in CARVE_AVAILABLE_METRICS:
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
                                "metric_ari": carve_aris[candidate_clusters.index(k)],
                            }
                        )

                    # --- Plotting
                    opt_idx = candidate_clusters.index(optimal_k)
                    plotting_dict[carve_metric] = {
                        "measure": carve_metric,
                        "k": optimal_k,
                        "labels": carve_labels_by_k[opt_idx],
                        "ari": carve_aris[opt_idx],
                    }
                    # ---

                # --- 4) Evaluate external metrics ---
                for metric in external_metrics:
                    metric_values = []

                    # --- Plotting
                    best_value = -np.inf
                    best_k = None
                    best_labels = None
                    best_ari = None
                    # ---

                    # Calculate metric value for each test_k
                    for k in candidate_clusters:
                        # 4.1) Get labels for this k
                        est = _build_estimator(
                            estimator_cls=estimator_cls,
                            n_clusters=k,
                            estimator_params=estimator_param_grid,
                            random_seed=benchmark_seed,
                        )
                        labels = np.asarray(est.fit_predict(X), dtype=np.int32)

                        ari = adjusted_rand_score(y, labels)

                        # 4.2) Calculate metric value
                        value = calculate_metric(
                            X,
                            labels,
                            metric,
                            estimator_cls=estimator_cls,
                            estimator_params=estimator_param_grid,
                            random_state=benchmark_seed,
                        )
                        metric_values.append((k, value, ari))

                        # --- Plotting
                        if value > best_value:
                            best_value = value
                            best_k = k
                            best_labels = labels
                            best_ari = ari
                        # ---

                    # 4.3) Find the k where respective metric is maximized
                    optimal_k = max(metric_values, key=lambda x: x[1])[0]

                    # 4.4) Record all results
                    for k, value, ari in metric_values:
                        results.append(
                            {
                                "axis_name": "difficulty_level",
                                "axis_value": difficulty_level,
                                "baseline_ari": baseline_ari,
                                "dataset_iteration": seed,
                                "true_k": true_k,
                                "metric_name": metric,
                                "k": k,
                                "metric_value": value,
                                "is_optimal": k == optimal_k,
                                "is_correct": k == true_k,
                                "metric_ari": ari,
                            }
                        )

                    # --- Plotting ---
                    plotting_dict[metric] = {
                        "measure": metric,
                        "k": best_k,
                        "labels": best_labels,
                        "ari": best_ari,
                    }
                    # ---

                # --- Plotting ---
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


def benchmark_scaling(
    settings_by_k: Dict,
    other_settings: Dict,
    axis_name: str,
    granularity: int = 10,
    n_seeds_per_value: int = 20,
    estimator: str = "kmeans",
    spectral_quant: float = 0.5,
    true_cluster_counts: Sequence[int] = (3, 4),
    candidate_clusters: Sequence[int] = range(2, 8),
    external_metrics: Sequence[str] = (
        "silhouette",
        "gap",
        "davies_bouldin",
        "calinski_harabasz",
    ),
    n_jobs: int = 1,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Benchmarks CARVE and external metrics across scaling regimes.

    Args:
        - settings_by_k (Dict): Simulation regime settings keyed by true_k.
        - other_settings (Dict): Other simulation settings.
        - axis_name (str): Scaling axis name ('n_total', 'p', or 'embed_dim').
        - n_seeds_per_value (int): Number of seeds per x_value and true_k.
        - estimator (str): Clustering estimator key for CARVE grids.
        - spectral_quant (float): Quantile used for spectral gamma estimation (default: 0.5).
        - true_cluster_counts (Sequence[int]): True cluster counts to simulate.
        - candidate_clusters (Sequence[int]): Candidate k values to evaluate.
        - external_metrics (Sequence[str]): External metrics to evaluate.
        - n_jobs (int): Number of jobs for CARVE.
        - random_state (int): Seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (scores_df, runtimes_df).
    """
    CARVE_AVAILABLE_METRICS_S = [
        "ari_stability",
        "ari_stability_1se",
        "ari_stability_quant",
        "consensus_pac_stability",
        "consensus_gini_stability",
        "consensus_ce_stability",
    ]

    CARVE_AVAILABLE_METRICS_G = [
        "ari_generalizability",
        "ari_generalizability_1se",
        "ari_generalizability_quant",
        "misclassification_generalizability",
    ]

    if axis_name == "n_total":
        x_values = [
            int(x) for x in np.logspace(np.log10(100), np.log10(10000), num=granularity)
        ]
    elif axis_name == "p":
        x_values = [
            int(x) for x in np.logspace(np.log10(10), np.log10(2500), num=granularity)
        ]
    elif axis_name == "embed_dim":
        x_values = [
            int(x) for x in np.logspace(np.log10(10), np.log10(2500), num=granularity)
        ]
    else:
        raise ValueError("axis_name must be 'n_total', 'p', or 'embed_dim'.")

    scores = []
    runtimes = []

    total_steps = granularity * n_seeds_per_value * len(true_cluster_counts)
    pbar = tqdm(
        total=total_steps, desc=f"benchmarking scaling ({axis_name})", leave=True
    )

    for i, x_value in enumerate(x_values):
        for true_k in true_cluster_counts:
            for seed in range(n_seeds_per_value):
                # --- 0) Set seed ---
                benchmark_seed = (
                    seed
                    + ((true_k - min(true_cluster_counts)) * 100)
                    + (i * 10000)
                    + random_state
                )
                # plotting_dict = {}  # Plotting

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

                # --- 2) Get baseline ARI ---
                estimator_grids = make_estimator_grids(
                    estimator=estimator,
                    candidate_clusters=candidate_clusters,
                    spectral_quant=spectral_quant,
                    X=X,
                    random_state=benchmark_seed,
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

                # --- 3) Fit and evaluate CARVE ---
                # 3.1) Fitting CARVE
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

                # 3.1.1) Evaluate stability runtime
                t0_s = time.perf_counter()
                carve_s.fit(X, reference_labels=y, mode="stability")
                t_carve_s = time.perf_counter() - t0_s

                # 3.1.2) Evaluate generalizability runtime
                t0_g = time.perf_counter()
                carve_g.fit(X, reference_labels=y, mode="generalizability")
                t_carve_g = time.perf_counter() - t0_g

                # 3.2) Get ARIs for all k
                carve_aris_s, carve_aris_g = [], []
                # carve_labels_by_k = []  # Plotting
                for k in candidate_clusters:
                    carve_consensus_labels_s = carve_s.get_labels(
                        k=k, measure="stability", mode="stability"
                    )  # providing rule is not necessary here as there is only a single option for every k
                    carve_consensus_labels_g = carve_g.get_labels(
                        k=k, measure="generalizability", mode="generalizability"
                    )
                    carve_aris_s.append(
                        adjusted_rand_score(y, carve_consensus_labels_s)
                    )
                    carve_aris_g.append(
                        adjusted_rand_score(y, carve_consensus_labels_g)
                    )
                    # carve_labels_by_k.append(carve_consensus_labels)  # Plotting

                # 3.3) Evaluating CARVE metrics
                for carve_metric in sorted(
                    set(CARVE_AVAILABLE_METRICS_S).union(CARVE_AVAILABLE_METRICS_G)
                ):
                    rule = get_rule(carve_metric)
                    measure = get_measure(carve_metric)

                    carve = (
                        carve_s
                        if carve_metric in CARVE_AVAILABLE_METRICS_S
                        else carve_g
                    )

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
                                "metric_ari": carve_aris_s[candidate_clusters.index(k)]
                                if carve_metric in CARVE_AVAILABLE_METRICS_S
                                else carve_aris_g[candidate_clusters.index(k)],
                            }
                        )

                    # --- Plotting
                    # opt_idx = candidate_clusters.index(optimal_k)
                    # plotting_dict[carve_metric] = {
                    #     "measure": carve_metric,
                    #     "k": optimal_k,
                    #     "labels": carve_labels_by_k[opt_idx],
                    #     "ari": carve_aris[opt_idx],
                    # }
                    # ---

                # --- 4) Evaluate external metrics ---
                for metric in external_metrics:
                    metric_values = []

                    # --- Plotting
                    # best_value = -np.inf
                    # best_k = None
                    # best_labels = None
                    # best_ari = None
                    # ---

                    # Calculate metric value for each test_k
                    for k in candidate_clusters:
                        # 4.1) Get labels for this k
                        est = _build_estimator(
                            estimator_cls=estimator_cls,
                            n_clusters=k,
                            estimator_params=estimator_param_grid,
                            random_seed=benchmark_seed,
                        )
                        labels = np.asarray(est.fit_predict(X), dtype=np.int32)

                        ari = adjusted_rand_score(y, labels)

                        # 4.2) Calculate metric value
                        value = calculate_metric(
                            X,
                            labels,
                            metric,
                            estimator_cls=estimator_cls,
                            estimator_params=estimator_param_grid,
                            random_state=benchmark_seed,
                        )
                        metric_values.append((k, value, ari))

                        # --- Plotting
                        # if value > best_value:
                        #     best_value = value
                        #     best_k = k
                        #     best_labels = labels
                        #     best_ari = ari
                        # ---

                    # 4.3) Find the k where respective metric is maximized
                    optimal_k = max(metric_values, key=lambda x: x[1])[0]

                    # 4.4) Record all results
                    for k, value, ari in metric_values:
                        scores.append(
                            {
                                "axis_name": axis_name,
                                "axis_value": x_value,
                                "baseline_ari": baseline_ari,
                                "dataset_iteration": seed,
                                "true_k": true_k,
                                "metric_name": metric,
                                "k": k,
                                "metric_value": value,
                                "is_optimal": k == optimal_k,
                                "is_correct": k == true_k,
                                "metric_ari": ari,
                            }
                        )

                    # --- Plotting ---
                    # plotting_dict[metric] = {
                    #     "measure": metric,
                    #     "k": best_k,
                    #     "labels": best_labels,
                    #     "ari": best_ari,
                    # }
                    # ---

                # 5) Record runtimes
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
                        "n_test_ks": len(list(candidate_clusters)),
                        "estimator": estimator,
                        "t_carve_sec_s": t_carve_s,
                        "t_carve_sec_g": t_carve_g,
                        "t_carve_per_k_sec_s": t_carve_s
                        / len(list(candidate_clusters)),
                        "t_carve_per_k_sec_g": t_carve_g
                        / len(list(candidate_clusters)),
                    }
                )

                # --- Plotting ---
                # clear_output(wait=True)

                # fig_pca, fig_sum = plot_benchmark_snapshot(
                #     X=X,
                #     results_df=pd.DataFrame(results),
                #     plotting_dict=plotting_dict,
                #     true_labels=y,
                #     baseline_labels=baseline_labels,
                #     baseline_ari=baseline_ari,
                #     panel_metrics=("ari_stability_1se", "ari_generalizability_1se", "silhouette", "davies_bouldin"),
                # )

                # display(fig_pca)
                # display(fig_sum)

                # plt.close(fig_pca)
                # plt.close(fig_sum)

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(scores), pd.DataFrame(runtimes)
