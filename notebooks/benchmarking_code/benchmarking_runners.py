import time

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from carve import CARVE

from benchmarking_simulation_helpers import simulate_scaling, parse_difficulty_and_simulate
from benchmarking_utils import make_model_grids
from benchmarking_metrics import calculate_metric
from typing import Any, Dict, Optional, Sequence


def benchmark_scaling(
    regime: Dict[int, Dict[str, Any]],
    x_name: str,
    B: int = 20,
    model: str = "kmeans",
    spectral_quant: float = 0.5,
    true_ks: Sequence[int] = (3, 4, 5, 6),
    test_ks: Sequence[int] = range(2, 8),
    external_metrics: Sequence[str] = ("silhouette", "gap", "davies_bouldin", "calinski_harabasz"),
    n_jobs: int = 1,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Benchmarks CARVE and external metrics across scaling regimes.

    Args:
        - regime (Dict): Simulation regime settings keyed by true_k.
        - x_name (str): Scaling axis name ('n_total', 'p', or 'embed_dim').
        - B (int): Number of seeds per x_value and true_k.
        - model (str): Clustering model key for CARVE grids.
        - spectral_quant (float): Quantile used for spectral gamma estimation (default: 0.5).
        - true_ks (Sequence[int]): True cluster counts to simulate.
        - test_ks (Sequence[int]): Candidate k values to evaluate.
        - external_metrics (Sequence[str]): External metrics to evaluate.
        - n_jobs (int): Number of jobs for CARVE.
        - random_state (int): Seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (scores_df, runtimes_df).
    """
    CARVE_AVAILABLE_METRICS = [
        "ari_stability", "ari_generalizability", "ari_average",
        "ari_stability_1se", "ari_generalizability_1se", "ari_average_1se",
        "ari_stability_quant", "ari_generalizability_quant", "ari_average_quant",
        "consensus_pac_stability", "consensus_gini_stability", "consensus_ce_stability",
        "misclassification_generalizability",
    ]

    if x_name == "n_total":
        x_values = [int(x) for x in np.logspace(np.log10(100), np.log10(10000), num=10)]
    elif x_name == "p":
        x_values = [int(x) for x in np.logspace(np.log10(10), np.log10(2500), num=10)]
    elif x_name == "embed_dim":
        x_values = [int(x) for x in np.logspace(np.log10(128), np.log10(2048), num=10)]
    else:
        raise ValueError("x_name must be 'n_total', 'p', or 'embed_dim'.")

    scores = []
    runtimes = []

    total_steps = len(x_values) * B * len(true_ks)
    pbar = tqdm(total=total_steps, desc=f"benchmarking scaling ({x_name})", leave=True)

    for x_value in x_values:
        for seed in range(B):
            for true_k in true_ks:
                X, y = simulate_scaling(
                    regime=regime[true_k],
                    true_k=true_k,
                    x_name=x_name,
                    x_value=x_value,
                    seed=seed,
                    base_random_state=random_state
                )

                model_grids = make_model_grids(model=model, test_ks=test_ks, spectral_quant=spectral_quant, X=X)

                carve = CARVE(
                    model_grids=model_grids,
                    n_jobs=n_jobs,
                    random_state=seed + random_state,
                )

                t0 = time.perf_counter()
                carve.fit(X, ref_labels=y)
                t_carve = time.perf_counter() - t0

                for carve_metric in CARVE_AVAILABLE_METRICS:
                    if carve_metric.endswith("_quant"):
                        rule = "quantile"
                        measure = carve_metric[:-6]
                    elif carve_metric.endswith("_1se"):
                        rule = "1se"
                        measure = carve_metric[:-4]
                    else:
                        rule = "max"
                        measure = carve_metric

                    optimal_k = carve.get_optimal_k(measure=measure, rule=rule)

                    for k in test_ks:
                        value = carve.model_df_.loc[carve.model_df_["n_clusters"] == k, measure].values[0]
                        scores.append({
                            x_name: x_value,
                            "metric_name": carve_metric,
                            "true_k": true_k,
                            "k": k,
                            "metric_value": value,
                            "is_optimal": (k == optimal_k),
                            "is_correct": (k == true_k),
                            "dataset_id": seed,
                        })
                                
                # time external metric pass separately
                t1 = time.perf_counter()
                
                model_cls, model_param_grid = model_grids[0]
                for metric in external_metrics:
                    metric_values = []
                    for k in test_ks:
                        labels = carve.get_optimal_labels(k=k)
                        
                        value = calculate_metric(
                            X, labels, metric,
                            model=model_cls,
                            model_params=model_param_grid,
                            random_state=seed + random_state
                        )
                        metric_values.append((k, value))

                    optimal_k, _ = max(metric_values, key=lambda x: x[1])
                    for k, value in metric_values:
                        scores.append({
                            x_name: x_value,
                            "metric_name": metric,
                            "true_k": true_k,
                            "k": k,
                            "metric_value": value,
                            "is_optimal": (k == optimal_k),
                            "is_correct": (k == true_k),
                            "dataset_id": seed,
                        })

                t_external = time.perf_counter() - t1

                runtimes.append({
                    x_name: x_value,
                    "true_k": true_k,
                    "dataset_id": seed,
                    "n": X.shape[0],
                    "p": X.shape[1],
                    "B": B,
                    "n_jobs": n_jobs,
                    "n_test_ks": len(list(test_ks)),
                    "model": model,
                    "t_carve_sec": t_carve,
                    "t_carve_per_k_sec": t_carve / len(list(test_ks)),
                    "t_external_sec": t_external,  # keep or drop
                })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(scores), pd.DataFrame(runtimes)


def benchmark_cluster_metrics(
    settings_by_k: Dict, 
    other_settings: Dict, 
    n_datasets: int = 10,
    B: int = 20,
    model: str = 'kmeans',
    spectral_quant: float = 0.5,
    model_grids: Optional[list[tuple[type, dict[str, list[Any]]]]] = None,
    true_ks: Sequence[int] = (3, 4, 5, 6),
    test_ks: Sequence[int] = range(2, 8),
    external_metrics: Sequence[str] = ("silhouette", "gap", "davies_bouldin", "calinski_harabasz"),
    n_jobs: int = 1,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Benchmarks clustering metrics across simulated datasets and configurations.

    Args:
        - settings_by_k (Dict): Mapping from true_k to simulation settings.
        - other_settings (Dict): Shared simulation settings.
        - n_datasets (int): Number of difficulty levels/datasets.
        - B (int): Number of seeds per dataset family.
        - model (str): Clustering model key ('kmeans', 'agglomerative', 'spectral').
        - spectral_quant (float): Quantile used for spectral gamma estimation (default: 0.5).
        - model_grids (Optional[list[tuple[type, dict]]]): Optional pre-built model grids.
        - true_ks (Sequence[int]): True cluster counts to simulate.
        - test_ks (Sequence[int]): Candidate k values to evaluate.
        - external_metrics (Sequence[str]): External metrics to evaluate.
        - n_jobs (int): Number of jobs for CARVE.
        - random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Benchmarking results per metric, dataset, and configuration.
    """
    results = []

    CARVE_AVAILABLE_METRICS = [
        'ari_stability', 'ari_generalizability', 'ari_average',
        'ari_stability_1se', 'ari_generalizability_1se', 'ari_average_1se',
        'ari_stability_quant', 'ari_generalizability_quant', 'ari_average_quant',
        
        'consensus_pac_stability', 'consensus_gini_stability', 'consensus_ce_stability',
        'misclassification_generalizability'
    ]

    total_steps = n_datasets * B * len(true_ks)
    pbar = tqdm(total=total_steps, desc="Benchmarking", leave=True)
    for i in range(n_datasets):
        for seed in range(B):
            for true_k in true_ks:
                benchmark_seed = i + true_k * 10 + seed * 100 + random_state
                X, y = parse_difficulty_and_simulate(
                    settings_by_k=settings_by_k[true_k],
                    other_settings=other_settings,
                    n_datasets=n_datasets,
                    difficulty=i,
                    true_k=true_k,
                    random_state=benchmark_seed
                )
                
                # fit and evaluate CARVE
                model_grids = make_model_grids(model, test_ks, spectral_quant=spectral_quant, X=X)
                carve = CARVE(
                    model_grids=model_grids,
                    n_jobs=n_jobs,
                    random_state=benchmark_seed
                )

                carve.fit(X, ref_labels=y)
                
                for carve_metric in CARVE_AVAILABLE_METRICS:
                    if carve_metric.endswith('_quant'):
                        rule = 'quantile'
                    elif carve_metric.endswith('_1se'):
                        rule = '1se'
                    else:
                        rule = 'max'
                    
                    if carve_metric.endswith('_1se'):
                        measure = carve_metric[:-4]
                    elif carve_metric.endswith('_quant'):
                        measure = carve_metric[:-6]
                    else:
                        measure = carve_metric
                            
                    optimal_k = carve.get_optimal_k(measure=measure, rule=rule)
                    
                    for k in test_ks:
                        value = carve.model_df_[measure].loc[carve.model_df_['n_clusters'] == k].values[0]
                        results.append({
                            'difficulty': i,
                            'metric_name': carve_metric,
                            'true_k': true_k,
                            'k': k,
                            'metric_value': value,
                            'is_optimal': k == optimal_k,
                            'is_correct': k == true_k,
                            'dataset_id': seed
                        })
                
                # evaluate external metrics
                model_cls, model_param_grid = model_grids[0]
                for metric in external_metrics:
                    metric_values = []
                    
                    # Calculate metric value for each test_k
                    for k in test_ks:
                        labels = carve.get_optimal_labels(k=k)
                        value = calculate_metric(
                            X, labels, metric,
                            model=model_cls,
                            model_params=model_param_grid,
                            random_state=benchmark_seed
                        )
                        metric_values.append((k, value))
                    
                    # Find the k where respective metric is maximized
                    optimal_k, _ = max(metric_values, key=lambda x: x[1])
                    
                    # Record all results
                    for k, value in metric_values:
                        results.append({
                            'difficulty': i,
                            'metric_name': metric,
                            'true_k': true_k,
                            'k': k,
                            'metric_value': value,
                            'is_optimal': k == optimal_k,
                            'is_correct': k == true_k,
                            'dataset_id': seed
                        })

                pbar.update(1)
    
    pbar.close()
    return pd.DataFrame(results)

