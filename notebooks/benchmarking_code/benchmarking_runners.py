import time

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from sklearn.metrics import adjusted_rand_score

from tqdm.notebook import tqdm

from carve import CARVE

from .benchmarking_simulation_helpers import simulate_scaling, parse_difficulty_and_simulate
from .benchmarking_utils import make_estimator_grids, _build_estimator, get_measure, get_rule, _pick_first
from .benchmarking_metrics import calculate_metric
from .benchmarking_plotting_reporting import plot_benchmark_snapshot


def benchmark_cluster_metrics(
    settings_by_k: Dict,
    other_settings: Dict,
    difficulty_levels: int = 10,
    n_seeds_per_dataset: int = 20,
    estimator: str = 'kmeans',
    spectral_quant: float = 0.5,
    estimator_grids: Optional[list[tuple[type, dict[str, list[Any]]]]] = None,
    true_cluster_counts: Sequence[int] = (3, 4, 5, 6),
    candidate_clusters: Sequence[int] = range(2, 8),
    external_metrics: Sequence[str] = ("silhouette", "gap", "davies_bouldin", "calinski_harabasz"),
    n_jobs: int = 1,
    random_state: int = 0
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
    results = []
    plt.ion()

    CARVE_AVAILABLE_METRICS = [
        'ari_stability', 'ari_generalizability', 'ari_average',
        'ari_stability_1se', 'ari_generalizability_1se', 'ari_average_1se',
        'ari_stability_quant', 'ari_generalizability_quant', 'ari_average_quant',
        
        'consensus_pac_stability', 'consensus_gini_stability', 'consensus_ce_stability',
        
        'misclassification_generalizability'
    ]

    total_steps = difficulty_levels * n_seeds_per_dataset * len(true_cluster_counts)
    pbar = tqdm(total=total_steps, desc="Benchmarking", leave=True)
    for difficulty_level in range(difficulty_levels):
        for true_k in true_cluster_counts:
            for seed in range(n_seeds_per_dataset): 
                # --- 0) Set seed ---
                benchmark_seed = seed + ((true_k - min(true_cluster_counts)) * 100) + (difficulty_level * 10000) + random_state
                plotting_dict = {}  # plotting
                
                # --- 1) Simulate data ---
                X, y = parse_difficulty_and_simulate(
                    settings_by_k=settings_by_k[true_k],
                    other_settings=other_settings,
                    difficulty_levels=difficulty_levels,
                    difficulty_index=difficulty_level,
                    true_cluster_count=true_k,
                    seed=benchmark_seed
                )
                
                # --- 2) Get baseline ARI ---
                estimator_grids = make_estimator_grids(estimator=estimator, candidate_clusters=candidate_clusters, spectral_quant=spectral_quant, X=X, random_state=benchmark_seed)
                estimator_cls, estimator_param_grid = estimator_grids[0]
                
                baseline_estimator = _build_estimator(
                    estimator_cls=estimator_cls,
                    n_clusters=true_k,
                    estimator_params=estimator_param_grid,
                    random_seed=benchmark_seed
                )
                baseline_labels = baseline_estimator.fit_predict(X)
                
                baseline_ari = adjusted_rand_score(y, baseline_labels)
                
                # --- 3) Fit and evaluate CARVE ---
                # 3.1) Fitting CARVE
                carve = CARVE(
                    estimator_param_grids=estimator_grids,
                    n_jobs=n_jobs,
                    random_state=benchmark_seed
                )

                carve.fit(X)
                
                # 3.2) Get ARIs for all k
                carve_aris = []
                carve_labels_by_k = []  # plotting
                for k in candidate_clusters:
                    carve_consensus_labels = carve.get_labels(k=k)  # providing measure and rule is not necessary here as there is only a single option for every k
                    carve_aris.append(adjusted_rand_score(y, carve_consensus_labels))
                    carve_labels_by_k.append(carve_consensus_labels)  # plotting
                
                # 3.2) Evaluating CARVE metrics
                for carve_metric in CARVE_AVAILABLE_METRICS:
                    rule = get_rule(carve_metric)
                    measure = get_measure(carve_metric)
                    optimal_k = carve.get_k(measure=measure, rule=rule)
                    
                    # --- plotting
                    opt_idx = candidate_clusters.index(optimal_k)
                    plotting_dict[carve_metric] = {
                        "measure": carve_metric,
                        "k": optimal_k,
                        "labels": carve_labels_by_k[opt_idx],
                        "ari": carve_aris[opt_idx],
                    }
                    # ---
                    
                    for k in candidate_clusters:
                        value = carve.estimator_results_[measure].loc[carve.estimator_results_["n_clusters"] == k].values[0]
                        results.append({
                            'difficulty_level': difficulty_level,
                            'baseline_ari': baseline_ari,
                            'dataset_iteration': seed,
                            'true_k': true_k,
                            'metric_name': carve_metric,
                            'k': k,
                            'metric_value': value,
                            'is_optimal': k == optimal_k,
                            'is_correct': k == true_k,
                            'metric_ari': carve_aris[candidate_clusters.index(k)],
                        })
                
                # --- 4) Evaluate external metrics ---
                for metric in external_metrics:
                    metric_values = []
                    
                    # --- plotting
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
                            X, labels, metric,
                            estimator_cls=estimator_cls,
                            estimator_params=estimator_param_grid,
                            random_state=benchmark_seed
                        )
                        metric_values.append((k, value, ari))
                        
                        # --- plotting
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
                        results.append({
                            'difficulty_level': difficulty_level,
                            'baseline_ari': baseline_ari,
                            'dataset_iteration': seed,
                            'true_k': true_k,
                            'metric_name': metric,
                            'k': k,
                            'metric_value': value,
                            'is_optimal': k == optimal_k,
                            'is_correct': k == true_k,
                            'metric_ari': ari,
                        })
                    
                    plotting_dict[metric] = {
                        "measure": metric,
                        "k": best_k,
                        "labels": best_labels,
                        "ari": best_ari,
                    }

                pbar.update(1)
                
                # --- Plotting ---
                clear_output(wait=True)
                
                fig_pca, fig_sum = plot_benchmark_snapshot(
                    X=X,
                    results_df=pd.DataFrame(results),
                    plotting_dict=plotting_dict,
                    true_labels=y,
                    baseline_labels=baseline_labels,
                    baseline_ari=baseline_ari,
                    panel_metrics=("ari_stability_1se", "ari_generalizability_1se", "silhouette", "davies_bouldin"),
                )
                
                display(fig_pca)
                display(fig_sum)

                plt.close(fig_pca)
                plt.close(fig_sum)

    pbar.close()
    return pd.DataFrame(results)

