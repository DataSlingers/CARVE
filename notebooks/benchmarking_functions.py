import multiprocessing
import os, random, inspect, time

from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tqdm.notebook import tqdm

from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, confusion_matrix, davies_bouldin_score, pairwise_distances, silhouette_score
from sklearn.utils import check_random_state

from scipy.optimize import linear_sum_assignment

from carve import CARVE
from carve.sim import simulate_clusters

def gamma_quantile_approx(
    X: np.ndarray,
    q: float = 0.50,
    max_points: int = 500,
    random_state: int = 0,
) -> float:
    rng = check_random_state(random_state)
    n = X.shape[0]
    Xs = X[rng.choice(n, size=max_points, replace=False)] if n > max_points else X

    D2 = pairwise_distances(Xs, metric="sqeuclidean")
    d2 = D2[np.triu_indices_from(D2, k=1)]
    d2 = d2[np.isfinite(d2)]
    if d2.size == 0:
        return 1.0
    return float(1.0 / (2.0 * np.quantile(d2, q)))

def align_labels(true_labels, pred_labels):
    """
    Aligns predicted cluster labels to true labels by maximizing agreement using the Hungarian algorithm.

    Args:
        true_labels (np.ndarray): Ground-truth labels for each sample (integer-coded).
        pred_labels (np.ndarray): Predicted cluster labels for each sample (arbitrary IDs).

    Returns:
        np.ndarray: Aligned predicted labels with IDs permuted to best match true_labels.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    label_map = {old: new for old, new in zip(col_ind, row_ind)}
    return np.array([label_map[l] if l in label_map else l for l in pred_labels])

def compute_dispersion(X: np.ndarray, labels: np.ndarray, metric: str = 'euclidean') -> float:
    """
    Computes the within-cluster dispersion measure W_k for a clustering solution.

    Args:
        X (np.ndarray): Data matrix with samples as rows and features as columns.
        labels (np.ndarray): Cluster labels for each observation.
        metric (str): Distance metric used to compute pairwise distances within clusters (default: 'euclidean').

    Returns:
        float: Total within-cluster dispersion W_k.
    """
    unique_labels = np.unique(labels)
    W_k = 0.0
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        
        if len(cluster_points) <= 1:
            continue
        
        distances = pairwise_distances(cluster_points, metric=metric, squared=True)
        W_k += np.sum(distances) / (2 * cluster_points.shape[0])
    
    return W_k

def gen_null_box(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform reference distribution over the axis-aligned bounding box of X."""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    span = maxs - mins
    return rng.random(size=X.shape) * span + mins

def _pick_first(v):
    return v[0] if isinstance(v, (list, tuple)) else v

def _build_estimator(
    algorithm: Type[ClusterMixin],
    k: int,
    algorithm_params: dict | None,
    seed: int,
):
    # start with the fixed params coming from model_grids[0][1]
    params = {}
    if algorithm_params:
        for key, val in algorithm_params.items():
            if key == "n_clusters":
                continue
            
            params[key] = _pick_first(val)

    # set k for this call
    params["n_clusters"] = k

    # set random_state when supported
    sig = inspect.signature(algorithm.__init__)
    if "random_state" in sig.parameters:
        params["random_state"] = seed

    # pin kmeans defaults across sklearn versions
    if algorithm.__name__ == "KMeans" or algorithm is KMeans:
        if "n_init" in sig.parameters and "n_init" not in params:
            params["n_init"] = 10

    return algorithm(**params)

def gap_statistic(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    n_refs: int = 10,
    algorithm: Type[ClusterMixin] = KMeans,
    algorithm_params: dict | None = None,
    random_state: int = 0,
) -> float:
    """
    Gap(k) = E[log W*_k] - log W_k, with W_k computed from provided labels.
    """
    rng = np.random.default_rng(int(random_state))

    W_k = compute_dispersion(X, labels, metric)
    if not np.isfinite(W_k) or W_k <= 0:
        return np.nan

    k = int(np.unique(labels).size)

    ref_log_disps = np.empty(n_refs, dtype=float)
    for b in range(n_refs):
        X_ref = gen_null_box(X, rng)
        seed_b = int(rng.integers(0, 2**32 - 1))

        est = _build_estimator(algorithm, k, algorithm_params, seed_b)
        ref_labels = est.fit_predict(X_ref)

        W_ref = compute_dispersion(X_ref, ref_labels, metric)
        ref_log_disps[b] = np.log(W_ref)

    return float(np.mean(ref_log_disps) - np.log(W_k))

def davies_bouldin_inv(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the inverse Davies-Bouldin index for a clustering solution.

    Args:
        X (np.ndarray): Data matrix with samples as rows and features as columns.
        labels (np.ndarray): Cluster labels for each observation.

    Returns:
        float: Inverse Davies-Bouldin score (1 / (1 + db)), higher is better. Returns np.nan if db is not finite.
    """
    db = davies_bouldin_score(X, labels)
    return 1.0 / (1.0 + db) if np.isfinite(db) else np.nan

def make_model_grids(model, test_ks, spectral_quant=0.5, X=None):
    if model == "agglomerative":
        return [(AgglomerativeClustering, {"n_clusters": list(test_ks), "linkage": ["ward"]})]
    if model == "spectral":
        gamma = gamma_quantile_approx(X, q=spectral_quant)
        return [(SpectralClustering, {"n_clusters": list(test_ks), "affinity": ["rbf"], "gamma": [gamma]})]
    return [(KMeans, {"n_clusters": list(test_ks), "n_init": [10]})]

def calculate_metric(X: np.ndarray,
    labels: np.ndarray,
    metric: str,
    model: Type[ClusterMixin],
    model_params: dict | None = None,
    random_state: int | None = 0,
) -> float:
    """
    Calculates a clustering validation metric for a given clustering solution.

    Args:
        X (np.ndarray): Data matrix with samples as rows and features as columns.
        labels (np.ndarray): Cluster labels for each observation.
        metric (str): Metric to compute ('gap', 'silhouette', 'davies_bouldin', 'DB', 'calinski_harabasz', 'CH').
        model (Type[ClusterMixin]): Clustering estimator class (used for Gap Statistic reference clustering).

    Returns:
        float: Value of the requested clustering metric.
    """
    if metric == 'gap':
        return gap_statistic(X, labels,
            algorithm=model,
            algorithm_params=model_params,
            random_state=random_state
        )

    elif metric == 'silhouette':
        return silhouette_score(X, labels, random_state=random_state)

    elif metric == 'davies_bouldin' or metric == 'DB':
        return davies_bouldin_inv(X, labels)

    elif metric == 'calinski_harabasz' or metric == 'CH':
        return calinski_harabasz_score(X, labels)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
def simulate_scaling(
    *,
    regime: Dict[str, Any],
    true_k: int,
    x_name: str,
    x_value: int,
    seed: int,
    base_random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO
    """
    if x_name not in {"n_total", "p", "embed_dim"}:
        raise ValueError("x_name must be 'n_total', 'p', or 'embed_dim'")

    n_total = int(regime.get("n_total", 500))
    p = int(regime.get("p", 100))

    if x_name == "n_total":
        n_total = int(x_value)
    elif x_name == "embed_dim":
        regime["embed_dim"] = int(x_value)
    else:
        p = int(x_value)

    X, y, _ = simulate_clusters(
        n_total=n_total,
        p=p,
        k=int(true_k),
        plotting=False,
        random_state=int(base_random_state + seed),
        **{k: v for k, v in regime.items() if k not in {"n_total", "p"}},
    )
    return X, y
    
def parse_difficulty_and_simulate(
    settings_by_k: Dict[int, Dict[str, List[float]]],
    other_settings: Dict,
    n_datasets: int,
    difficulty: int,
    true_k: int,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates clustered datasets based on difficulty level by interpolating settings.

    Args:
        settings_by_k (Dict[int, Dict[str, List[float]]]): Dictionary mapping difficulty levels to simulation settings.
        other_settings (Dict): Additional simulation settings.
        n_datasets (int): Total number of difficulty levels/datasets.
        difficulty (int): Difficulty index (0=easy, middle=medium, last=difficult).
        true_k (int): True number of clusters to simulate.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Simulated data matrix X and cluster labels y.
    """
    if difficulty == 0 or difficulty == n_datasets - 1 or difficulty == int(round(n_datasets // 2)):
        if difficulty == 0:
            level = 'easy'
        elif difficulty == n_datasets - 1:
            level = 'difficult'
        elif difficulty == int(round(n_datasets // 2)):
            level = 'medium'
            
        X, y, _ = simulate_clusters(
            k=true_k,
            plotting=False,
            random_state=random_state,
            **{k: v for k, v in settings_by_k[level].items()},
            **{k: v for k, v in other_settings.items()}
        )
        
    else:
        if difficulty > 0 and difficulty < int(round(n_datasets // 2)):
            # Linear interpolation between 'easy' and 'medium' settings
            frac = difficulty / int(round(n_datasets // 2))
            easy_settings = settings_by_k['easy']
            medium_settings = settings_by_k['medium']
            
            interpolated_settings = {}
            for key in easy_settings:
                v_easy = np.array(easy_settings[key])
                v_medium = np.array(medium_settings[key])
                interpolated = (1 - frac) * v_easy + frac * v_medium
                interpolated_settings[key] = interpolated.tolist()
                
        elif difficulty > int(round(n_datasets // 2)) and difficulty < (n_datasets - 1):
            # Linear interpolation between 'medium' and 'difficult' settings
            frac = (difficulty - int(round(n_datasets // 2))) / (n_datasets - 1 - int(round(n_datasets // 2)))
            medium_settings = settings_by_k['medium']
            difficult_settings = settings_by_k['difficult']
            
            interpolated_settings = {}
            for key in medium_settings:
                v_medium = np.array(medium_settings[key])
                v_difficult = np.array(difficult_settings[key])
                interpolated = (1 - frac) * v_medium + frac * v_difficult
                interpolated_settings[key] = interpolated.tolist()
                
        else:
            raise ValueError(f"Got difficulty={difficulty} but expected 0, {n_datasets - 1}, {int(round(n_datasets // 2))}, or a value in between those.")
        
        if "noise_dims" in interpolated_settings:
            noise_dims = interpolated_settings["noise_dims"]
            interpolated_settings["noise_dims"] = int(np.floor(noise_dims + 0.5))
            
        X, y, _ = simulate_clusters(
            k=true_k,
            plotting=False,
            random_state=random_state,
            **{k: v for k, v in interpolated_settings.items()},
            **{k: v for k, v in other_settings.items()}
        )
    
    return X, y

def benchmark_scaling(
    regime,
    x_name,
    B=20,
    model="kmeans",
    spectral_quant=0.5,
    true_ks=(3, 4, 5, 6),
    test_ks=range(2, 8),
    external_metrics=("silhouette", "gap", "davies_bouldin", "calinski_harabasz"),
    n_jobs=1,
    random_state=0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    returns:
      df_scores: long-format metric values + selection correctness (as you already do)
      df_runtime: one row per dataset with carve runtime (+ optional external runtime)
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
    settings_by_k, 
    other_settings, 
    n_datasets=10,
    B=20,
    model='kmeans',
    spectral_quant=0.5,
    model_grids=None,
    true_ks=[3, 4, 5, 6],
    test_ks=range(2, 7+1),
    external_metrics=["silhouette", "gap", "davies_bouldin", "calinski_harabasz"],
    n_jobs=1,
    random_state=0
) -> pd.DataFrame:
    """
    Benchmarks clustering metrics across multiple simulated datasets and clustering configurations.

    Args:
        settings_by_k (Dict): Dictionary mapping true number of clusters to simulation settings.
        other_settings (Dict): Additional simulation settings.
        n_datasets (int): Total number of difficulty levels/datasets.
        B (int): Number of random seeds per dataset family.
        model (str): Clustering model to use ('kmeans', 'agglomerative', 'spectral').
        model_grids (List[Tuple[Type, Dict]], optional): List of model classes and their parameter grids.
        true_ks (List[int]): List of true number of clusters for simulation.
        test_ks (range): Range of cluster numbers to test.
        external_metrics (List[str]): List of external metrics to evaluate.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing benchmarking results for each metric, dataset, and configuration.
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

def plot_scaling_examples(
    regime: Dict,
    x_name: str,
    ks: np.ndarray = np.array([3, 4, 5, 6]),
    B: int = 10,
    random_state: int = 0,
    estimator_type: str = 'hierarchical',
    spectral_quant: float = 0.5,
    name_of_ex: str = 'Scaling Benchmark'
) -> None:
    """
    TODO
    """
    if x_name == "n_total":
        x_values = [int(x) for x in np.logspace(np.log10(100), np.log10(10000), num=3)]
    elif x_name == "p":
        x_values = [int(x) for x in np.logspace(np.log10(10), np.log10(2500), num=3)]
    elif x_name == "embed_dim":
        x_values = [int(x) for x in np.logspace(np.log10(128), np.log10(2048), num=3)]
    else:
        raise ValueError("x_name must be 'n_total', 'p', or 'embed_dim'.")

    _, axes = plt.subplots(len(ks), len(x_values), figsize=(3 * len(x_values), 3 * len(ks)))
    for i, k in enumerate(ks):
        for j, x_value in enumerate(x_values):
            # Simulate dataset to be displayed
            X, y = simulate_scaling(
                regime=regime[k],
                true_k=k,
                x_name=x_name,
                x_value=x_value,
                seed=1 * 10 + j,
                base_random_state=random_state
            )

            # Compute baseline ARI over B replicates
            ari_arr = []
            for b in range(B):
                X_val, y_val = simulate_scaling(
                    regime=regime[k],
                    true_k=k,
                    x_name=x_name,
                    x_value=x_value,
                    seed=i * 100 + j * 10 + b,
                    base_random_state=random_state
                )

                if estimator_type == 'hierarchical':
                    estimator = AgglomerativeClustering(n_clusters=k)
                elif estimator_type == 'kmeans':
                    estimator = KMeans(n_clusters=k, random_state=random_state)
                elif estimator_type == 'spectral':
                    gamma = gamma_quantile_approx(X_val, q=spectral_quant)
                    estimator = SpectralClustering(n_clusters=k, affinity='rbf', gamma=gamma, random_state=random_state)
                else:
                    raise ValueError("estimator_type must be 'hierarchical' or 'spectral'")

                y_hat_val = estimator.fit_predict(X_val)
                ari = adjusted_rand_score(y_val, y_hat_val)
                ari_arr.append(ari)

            ari_mean = np.mean(ari_arr)

            # Plot PCA of dataset
            X_pca = PCA(n_components=2, random_state=0).fit_transform(X)
            ax = axes[i, j]
            ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=y, cmap="tab10", s=20, alpha=0.8, edgecolors="k"
            )
            ax.set_title(f"k={k}, {x_name}={x_value} | ARI={ari_mean:.3f}")
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"Scaling Benchmark: {name_of_ex}", fontsize=16)
    plt.tight_layout(); plt.show()

def plot_examples(
    settings_by_k: Dict, 
    other_settings: Dict, 
    ks: np.ndarray = np.array([3, 4, 5, 6]), 
    anchors: List[str] = ['easy', 'medium', 'difficult'], 
    B: int = 10, 
    estimator_type: str = 'kmeans', 
    spectral_quant: float = 0.5,
    name_of_ex: str = 'Uncorrelated Gaussians'
) -> None:
    """
    Visualizes example clustering datasets for different cluster counts and difficulty levels,
    and computes baseline clustering performance (ARI) for each configuration.

    Args:
        settings_by_k (Dict): Simulation parameters for each k and difficulty anchor.
        other_settings (Dict): Shared/global simulation parameters.
        ks (np.ndarray): Array of cluster counts to visualize.
        anchors (List[str]): List of difficulty levels.
        B (int): Number of replicates for baseline ARI estimation.
        estimator_type (str): Clustering algorithm type ('kmeans', 'agglomerative' or 'spectral').
        name_of_ex (str): Title for the figure.

    Returns:
        None
    """
    
    _, axes = plt.subplots(len(ks), len(anchors), figsize=(12, 14))
    for i, k in enumerate(ks):
        for j, a in enumerate(anchors):
            
            # simulate dataset to be displayed
            X, y, _ = simulate_clusters(
                k=k,
                plotting=False,
                random_state=1*10 + j,
                **settings_by_k[k][a],
                **other_settings
            )
            
            # compute baseline ARI over B replicates
            ari_arr = []
            for b in range(B):
                X_val, y_val, _ = simulate_clusters(
                    k=k,
                    plotting=False,
                    random_state=1*100 + j*10 + b,
                    **settings_by_k[k][a],
                    **other_settings
                )
                
                if estimator_type == 'agglomerative':
                    estimator = AgglomerativeClustering(
                        n_clusters=k
                    )
                elif estimator_type == 'spectral':
                    gamma = gamma_quantile_approx(X_val, q=spectral_quant)
                    estimator = SpectralClustering(
                        n_clusters=k,
                        affinity='rbf',
                        gamma=gamma,
                        random_state=0
                    )
                else:
                    estimator = KMeans(
                        n_clusters=k,
                        random_state=0
                    )
                
                y_hat_val = estimator.fit_predict(X_val)
                ari = adjusted_rand_score(y_val, y_hat_val)
                ari_arr.append(ari)
                
            ari_mean = np.mean(ari_arr)

            # plot PCA of dataset
            X_pca = PCA(n_components=2, random_state=0).fit_transform(X)
            ax = axes[i, j]
            ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=y, cmap="tab10", s=20, alpha=0.8, edgecolors="k"
            )
            ax.set_title(f"k={k}, difficulty={a} | Baseline ARI={ari_mean:.3f}")
            ax.set_xticks([]); ax.set_yticks([])
            
    plt.suptitle(f"Example datasets: {name_of_ex}", fontsize=16)
    plt.tight_layout(); plt.show()
    
def plot_accuracy_vs_difficulty(
    results_df: pd.DataFrame, 
    show_se: bool = False
) -> None:
    """
    Plots the accuracy of clustering metrics as a function of problem difficulty.

    Args:
        results_df (pd.DataFrame): DataFrame containing benchmarking results with columns
            ['difficulty', 'metric_name', 'is_optimal', 'is_correct', 'dataset_id'].
        show_se (bool, optional): If True, displays standard error bands around the mean accuracy.

    Returns:
        None. Displays a line plot of accuracy vs. difficulty for each metric.
    """
    accuracy_df = (
        results_df[results_df['is_optimal']]
            .groupby(['difficulty', 'metric_name', 'dataset_id'])
            .agg({'is_correct': 'mean'})
            .reset_index()
            .groupby(['difficulty', 'metric_name'])
            .agg({'is_correct': ['mean', 'std']})
            .reset_index()
    )
    
    # Plot
    plt.figure(figsize=(12, 8))
    for metric in accuracy_df['metric_name'].unique():
        subset = accuracy_df[accuracy_df['metric_name'] == metric]
        plt.plot(subset['difficulty'], subset[('is_correct', 'mean')], marker='o', label=metric)
        if show_se:
            plt.fill_between(
                subset['difficulty'],
                subset[('is_correct', 'mean')] - subset[('is_correct', 'std')],
                subset[('is_correct', 'mean')] + subset[('is_correct', 'std')],
                alpha=0.2
        )
    
    plt.xlabel('Difficulty (difficulty)')
    plt.ylabel('Accuracy')
    plt.title('Metric Accuracy vs. Problem Difficulty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
def plot_metric_lineplots_grid(
    results_df: pd.DataFrame,
    difficulties: List[float],
    metrics_to_display: List[str] = None,
    show_majority_votes: bool = True,
    figsize: Tuple[float, float] = None
) -> None:
    """
    Plots a grid of line plots for clustering metrics across different difficulties and true_k values.

    Args:
        results_df (pd.DataFrame): DataFrame containing clustering results with columns:
            ['true_k', 'k', 'difficulty', 'metric_name', 'metric_value', ...].
        difficulties (List[float]): List of difficulty values to anchor the grid columns.
        metrics_to_display (Optional[List[str]]): List of metric names to plot. If None, all available metrics are used.
        show_majority_votes (bool): If True, displays majority vote markers (diamond, star, cross) for optimal k selection.
        figsize (Optional[Tuple[float, float]]): Figure size for the plot. If None, size is determined automatically.

    Returns:
        None. Displays the matplotlib figure.
    """
    PAIRED = {
        "ari_stability_1se": "ari_stability",
        "ari_generalizability_1se": "ari_generalizability",
    }
    
    BASE_WITH_PAIR = set(PAIRED.values())  # metrics that have a 1se partner

    # numeric safety
    results_df = results_df.copy()
    if 'is_optimal' in results_df.columns:
        results_df['is_optimal'] = results_df['is_optimal'].astype(int)

    anchors = [difficulties[0], difficulties[len(difficulties)//2], difficulties[-1]]

    all_metrics = sorted(results_df['metric_name'].unique())
    if metrics_to_display is not None:
        metrics = [m for m in metrics_to_display if (m in all_metrics) and (m not in PAIRED.keys())]
        if not metrics:
            raise ValueError(f"None of the provided metrics {metrics_to_display} found in data")
    else:
        metrics = [m for m in all_metrics if m not in PAIRED.keys()]

    true_ks = sorted(results_df['true_k'].unique())

    if figsize is None:
        figsize = (3.5*len(anchors), 2.5*len(true_ks))

    fig, axes = plt.subplots(len(true_ks), len(anchors), figsize=figsize)

    if len(true_ks) == 1:
        axes = np.array([axes])
    if len(anchors) == 1:
        axes = np.array([axes]).T

    def difficulty_mask(series, a):
        return np.isclose(series.to_numpy(dtype=float), float(a), rtol=1e-6, atol=1e-8)

    for i, true_k in enumerate(true_ks):
        for j, difficulty in enumerate(anchors):
            ax = axes[i, j]
            line_y = {}

            # plot each (base) metric line
            for metric in metrics:
                subset_mask = (
                    (results_df['true_k'] == true_k) &
                    (results_df['metric_name'] == metric) &
                    difficulty_mask(results_df['difficulty'], difficulty)
                )
                subset = results_df.loc[subset_mask]
                if subset.empty:
                    continue

                avg_df = (
                    subset.groupby('k')['metric_value']
                        .agg(mean='mean', std='std')
                        .reset_index()
                )
                mean_values = avg_df['mean'].to_numpy()
                
                min_val = np.min(mean_values)
                max_val = np.max(mean_values)
                mean_values = (mean_values - min_val) / (max_val - min_val)

                ax.plot(avg_df['k'], mean_values, marker='o', label=metric)
                line_y[metric] = dict(zip(avg_df['k'].to_numpy(), mean_values))

                if show_majority_votes and 'is_optimal' in subset.columns and metric not in BASE_WITH_PAIR:
                    votes = subset.groupby('k')['is_optimal'].sum().sort_values(ascending=False)
                    if not votes.empty:
                        k_base_vote = int(votes[votes == votes.iloc[0]].index.max())  # if tie, then largest k
                        if k_base_vote in line_y[metric]:
                            y_on_curve = float(line_y[metric][k_base_vote])
                            ax.scatter(
                                [k_base_vote], [y_on_curve], marker='D', s=70,
                                edgecolors='black', facecolors='white',
                                zorder=5, label=None
                            )

            # handle paired markers with agreement logic
            if show_majority_votes:
                for excl_metric, base_metric in PAIRED.items():
                    excl_mask = (
                        (results_df['true_k'] == true_k) &
                        (results_df['metric_name'] == excl_metric) &
                        difficulty_mask(results_df['difficulty'], difficulty)
                    )
                    base_mask = (
                        (results_df['true_k'] == true_k) &
                        (results_df['metric_name'] == base_metric) &
                        difficulty_mask(results_df['difficulty'], difficulty)
                    )
                    excl_subset = results_df.loc[excl_mask]
                    base_subset = results_df.loc[base_mask]
                    if excl_subset.empty or base_subset.empty:
                        continue
                    if 'is_optimal' not in excl_subset.columns or 'is_optimal' not in base_subset.columns:
                        continue

                    # votes (consistent tie rule: largest k)
                    excl_votes = excl_subset.groupby('k')['is_optimal'].sum().sort_values(ascending=False)
                    base_votes = base_subset.groupby('k')['is_optimal'].sum().sort_values(ascending=False)
                    if excl_votes.empty or base_votes.empty:
                        continue

                    k_excl_vote = int(excl_votes[excl_votes == excl_votes.iloc[0]].index.max())
                    k_base_vote = int(base_votes[base_votes == base_votes.iloc[0]].index.max())

                    # y anchors from the base metric curve
                    y_map = line_y.get(base_metric, {})
                    if not y_map:
                        continue
                    # require the y for at least the locations we will plot
                    if (k_base_vote not in y_map) and (k_excl_vote not in y_map):
                        continue

                    if k_base_vote == k_excl_vote:
                        y_star = float(y_map.get(k_base_vote, list(y_map.values())[0]))
                        y_star += 0.02
                        ax.scatter([k_base_vote], [y_star], marker='*', s=220,
                                   facecolors='none', edgecolors='black',
                                   linewidths=2.0, zorder=8, label=None)
                    else:
                        # draw ◇ at base vote
                        if k_base_vote in y_map:
                            y_d = float(y_map[k_base_vote])
                            ax.scatter(
                                [k_base_vote], [y_d], marker='D', s=70,
                                edgecolors='black', facecolors='white',
                                zorder=6, label=None
                            )

                        if k_excl_vote in y_map:
                            y_x = float(y_map[k_excl_vote]) + 0.025
                            ax.scatter(
                                [k_excl_vote], [y_x], marker='x', s=90,
                                color='black', linewidths=2, zorder=7, label=None
                            )

            # guides / labels
            ax.axvline(true_k, color='gray', linestyle='--', alpha=0.5)
            if i == 0:
                ax.set_title(f'α = {difficulty:.3f}')
            if j == 0:
                ax.set_ylabel(f'true_k = {true_k}')
            ax.set_xlabel('k')
            ax.grid(True, alpha=0.3)

            k_values = results_df['k'].unique()
            ax.set_xlim(min(k_values) - 0.5, max(k_values) + 0.5)

    # legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if show_majority_votes:
        handles += [
            Line2D(
                [0],[0], marker='D', linestyle='None',
                markerfacecolor='white', markeredgecolor='black', label='max rule (vote)'
            ),
            Line2D(
                [0],[0], marker='x', linestyle='None',
                color='black', label='1se rule (vote)'
            ),
            Line2D(
                [0],[0], marker='*', linestyle='None',
                markerfacecolor='none', markeredgecolor='black', label='agree (vote)'
            ),
        ]
        labels += ['max rule (vote)', '1se rule (vote)', 'agree (vote)']

    fig.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
        fancybox=True, shadow=True, ncol=min(5, max(1, len(metrics)))
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def print_summary_stats(
    results_df: pd.DataFrame,
    group_col: str = "difficulty"
) -> None:
    """
    Prints a summary table of clustering metric accuracy statistics,
    reporting mean, standard deviation (SD), and standard error (SE) per metric and group_col.

    Args:
        results_df (pd.DataFrame): DataFrame with columns [group_col, 'metric_name', 'is_optimal', 'is_correct', ...].
        group_col (str): Column to group by (e.g., 'difficulty', 'p', 'n_total', 'embed_dim').

    Returns:
        None. Displays formatted accuracy statistics per metric and group_col, and overall ranking.
    """
    if group_col not in results_df.columns:
        raise ValueError(f"Column '{group_col}' not found in results_df.")

    df = results_df.loc[results_df['is_optimal']].copy()

    # Per-(group_col, metric) stats
    g = (
        df.groupby([group_col, 'metric_name'])['is_correct']
            .agg(mean='mean', std='std', n='count')
            .reset_index()
    )
    g['std'] = g['std'].fillna(0.0)
    g['se'] = g['std'] / g['n'].clip(lower=1).pow(0.5)

    # Mark best per group_col
    stars = (
        g.loc[g.groupby(group_col)['mean'].idxmax(), [group_col, 'metric_name']]
        .set_index([group_col, 'metric_name'])
        .assign(star='*')
    )

    # Format: mean (SD, SE)[*]
    def _fmt(row):
        key = (row[group_col], row['metric_name'])
        star = stars.loc[key, 'star'] if key in stars.index else ''
        return f"{row['mean']:.3f} ({row['std']:.3f}, {row['se']:.3f}){star}"

    g['cell'] = g.apply(_fmt, axis=1)
    groups = sorted(g[group_col].unique())

    # Compact grid
    grid = (
        g.pivot(index='metric_name', columns=group_col, values='cell')
         .reindex(columns=groups)
    )

    # Overall stats
    overall = (
        df.groupby('metric_name')['is_correct']
            .agg(mean='mean', std='std', n='count')
    )
    overall['std'] = overall['std'].fillna(0.0)
    overall['se'] = overall['std'] / overall['n'].clip(lower=1).pow(0.5)

    # Print
    print(f"\n=== Accuracy (mean (SD, SE)) by {group_col} ===")
    with pd.option_context('display.max_columns', None, 'display.width', 120):
        print(grid.fillna('').sort_index())

    print(f"\n======= Overall ranking (aggregated over all {group_col} & k) ========")
    ranked = overall.sort_values('mean', ascending=False)
    for i, (metric, r) in enumerate(ranked.iterrows(), start=1):
        print(f"{i:>2}. {metric:<35} {r['mean']:.3f} ({r['std']:.3f}, {r['se']:.3f})  (n={int(r['n'])})")