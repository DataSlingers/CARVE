"""Core validation runner for CARVE."""

from typing import Any, Callable, Dict, List, Tuple, Type
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from ._output import _log_config_progress
from ._consensus import compute_consensus_matrix
from ._misclassification import compute_generalizability_scores
from ._pipeline import build_preprocessing_pipeline
from ._utils import cluster_labels, split_subsample_indices
import warnings


EstimatorRecord = Dict[str, Any]
PipelineRecord = Dict[str, Any]

ValidationReturn = Tuple[
    List[EstimatorRecord],      # model_records | TODO: check whether this type suggestion is correct
    List[PipelineRecord],   # pipeline_records | TODO: check whether this type suggestion is correct
    List[np.ndarray],       # consensus_matrices
    List[np.ndarray],       # generalizability_scores
]

ResultTuple = Tuple[
    float, float,                           # ARI stability, ARI generalizability
    np.ndarray, np.ndarray, np.ndarray,     # labels_1, labels_test, labels_pred
    np.ndarray,                             # labels_2
    np.ndarray, np.ndarray, np.ndarray,     # P_1_idx, P_test_idx, P_2_idx,
    Dict[str, Any], Dict[str, Any],         # norm_params, dr_params
    str, str                                # norm_name, dr_name
]

GridSpec = Tuple[Type[ClusterMixin], Dict[str, List[Any]]]
PreprocSpec = Tuple[Callable[..., TransformerMixin], Dict[str, List[Any]]]

def run_validation(
    X: np.ndarray,
    estimator_grids: List[GridSpec],
    n_resamples: int,
    subsample_ratio: float,
    norm_options: List[PreprocSpec],
    dr_options: List[PreprocSpec], 
    random_preprocess: bool = False, 
    n_jobs: int = 1, 
    random_state: int = None,
    prog_bar: bool = False,
    verbose: int = 1
) -> ValidationReturn:
    """Run CARVE validation over estimator grids and resamples.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    estimator_grids : list of tuple
        Estimator classes and parameter grids to evaluate.
    n_resamples : int
        Number of resampling iterations per configuration.
    subsample_ratio : float
        Proportion of samples used in each subsample.
    norm_options : list
        Normalization preprocessing options.
    dr_options : list
        Dimensionality reduction options.
    random_preprocess : bool, default=False
        Whether to sample preprocessing randomly per resample.
    n_jobs : int, default=1
        Number of parallel jobs for resamples.
    random_state : int or None, default=None
        Random seed for reproducibility.
    prog_bar : bool, default=False
        If True, display a progress bar for grid configurations.
    verbose : int, default=1
        Verbosity for logging.

    Returns
    -------
    estimator_records : list of dict
        One record per estimator configuration with aggregate metrics.
    pipeline_records : list of dict
        Optional per-resample preprocessing records when randomized.
    consensus_matrices : list of ndarray
        Consensus matrices for each configuration.
    generalizability_scores : list of ndarray
        Per-sample generalizability arrays for each configuration.
    """
    n = X.shape[0]

    estimator_records: List[EstimatorRecord] = []
    pipeline_records: List[PipelineRecord] = []
    consensus_matrices: List[np.ndarray] = []
    generalizability_scores: List[np.ndarray] = []
    
    total_configs = sum(len(list(ParameterGrid(g))) for _, g in estimator_grids)
    
    config_idx = 0
    with tqdm(
        total=total_configs, 
        desc="Grid configs", 
        disable=not prog_bar
    ) as pbar:
        for est_class, grid in estimator_grids:
            for params in ParameterGrid(grid):
                config_idx += 1
                
                worker = delayed(validation_iter)
                results = Parallel(n_jobs=n_jobs)(
                    worker(
                        X=X, 
                        est_class=est_class, 
                        params=params, 
                        subsample_ratio=subsample_ratio, 
                        n_resamples=n_resamples, 
                        seed=b, 
                        norm_options=norm_options, 
                        dr_options=dr_options,
                        random_preprocess=random_preprocess, 
                        random_state=random_state
                    ) 
                    for b in range(n_resamples)
                )
                
                aris_stab = [r[0] for r in results]
                aris_gen = [r[1] for r in results]
                aris_avg = [(r[0] + r[1]) / 2 for r in results]
                
                M = compute_consensus_matrix(
                    n_samples=n, 
                    runs=[(r[6], r[2]) for r in results],        # r[6]: P_1_idx, r[2]: labels_1
                )
                
                E = compute_generalizability_scores(
                    n_samples=n, 
                    runs=[(r[7], r[3], r[4]) for r in results]  # r[7]: P_test_idx, r[3]: labels_test, # r[4]: labels_pred
                )                      
                
                consensus_matrices.append(M)
                generalizability_scores.append(E)

                record: EstimatorRecord = {
                    "estimator": est_class.__name__,
                    **params,
                    "ari_stability": float(np.mean(aris_stab)),
                    "ari_stability_se": float(np.std(aris_stab, ddof=1) / np.sqrt(n_resamples)),
                    "ari_stability_upper": float(np.quantile(aris_stab, 0.95)),
                    "ari_stability_lower": float(np.quantile(aris_stab, 0.05)),
                    
                    "ari_generalizability": float(np.mean(aris_gen)),
                    "ari_generalizability_se": float(np.std(aris_gen, ddof=1) / np.sqrt(n_resamples)),
                    "ari_generalizability_upper": float(np.quantile(aris_gen, 0.95)),
                    "ari_generalizability_lower": float(np.quantile(aris_gen, 0.05)),
                    
                    "ari_average": float(np.mean(aris_avg)),
                    "ari_average_se": float(np.std(aris_avg, ddof=1) / np.sqrt(n_resamples)),
                    "ari_average_upper": float(np.quantile(aris_avg, 0.95)),
                    "ari_average_lower": float(np.quantile(aris_avg, 0.05)),
                }
                estimator_records.append(record)
                
                if random_preprocess:
                    pipeline_records.append({
                        'estimator': est_class.__name__, 
                        'params': params, 
                        'results': results
                    })
                    
                _log_config_progress(
                    config_idx=config_idx,
                    total_configs=total_configs,
                    est_class=est_class,
                    params=params,
                    record=record,
                    pbar_obj=pbar if prog_bar else None,
                    verbose=verbose
                )
                
                pbar.update(1)
                
    return estimator_records, pipeline_records, consensus_matrices, generalizability_scores

def validation_iter(
    X: np.ndarray,
    est_class: Type,
    params: Dict[str, Any],
    subsample_ratio: float,
    n_resamples: int,
    seed: int,
    norm_options: List[PreprocSpec],
    dr_options: List[PreprocSpec],
    random_preprocess: bool = False,
    random_state: int = None
) -> ResultTuple:
    """Run a single resampling iteration for one estimator configuration.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    est_class : type
        Clustering estimator class.
    params : dict
        Hyperparameters for the estimator.
    subsample_ratio : float
        Proportion of samples used in each subsample.
    n_resamples : int
        Total resamples, used to offset random seeds.
    seed : int
        Per-resample seed offset.
    norm_options : list
        Normalization options.
    dr_options : list
        Dimensionality reduction options.
    random_preprocess : bool, default=False
        Whether to randomize preprocessing.
    random_state : int or None, default=None
        Base random seed for reproducibility.

    Returns
    -------
    result : tuple
        Tuple containing ARI metrics, labels, sample indices, and preprocessing
        metadata for this resample.
    """
    n_samples = X.shape[0]
    random_state0 = random_state if random_state is not None else 0
    
    P_1_idx, P_test_idx = split_subsample_indices(n_samples, subsample_ratio=subsample_ratio, random_state=random_state0+seed)
    P_2_idx, _ = split_subsample_indices(n_samples, subsample_ratio=subsample_ratio, random_state=random_state0+seed+n_resamples)
    
    pipeline, normalization_params, dim_reduction_params, normalization_name, dim_reduction_name = build_preprocessing_pipeline(
        randomize_preprocessing=random_preprocess, 
        normalization_options=norm_options, 
        dim_reduction_options=dr_options, 
        seed=random_state0 + seed
    )
    
    X_prepocessed = pipeline.fit_transform(X)

    X_1 = X_prepocessed[P_1_idx]
    X_test = X_prepocessed[P_test_idx]
    X_2 = X_prepocessed[P_2_idx]
    
    # clustering 
    labels_1 = cluster_labels(X_1, est_class, random_state=random_state0+seed, **params)
    labels_test = cluster_labels(X_test, est_class, random_state=random_state0+seed, **params)
    labels_2 = cluster_labels(X_2, est_class, random_state=random_state0+seed, **params)
    
    n_clusters = params.get('n_clusters')
    if len(np.unique(labels_1)) != n_clusters:
        warnings.warn(f"labels_1 has {len(np.unique(labels_1))} clusters, expected {n_clusters}")
    if len(np.unique(labels_test)) != n_clusters:
        warnings.warn(f"labels_test has {len(np.unique(labels_test))} clusters, expected {n_clusters}")
    if len(np.unique(labels_2)) != n_clusters:
        warnings.warn(f"labels_2 has {len(np.unique(labels_2))} clusters, expected {n_clusters}")
    
    # model-explorer ARI
    _, i_1, i_2 = np.intersect1d(P_1_idx, P_2_idx, return_indices=True)
    ari_stab = adjusted_rand_score(labels_1[i_1], labels_2[i_2])
    
    # predictive ARI
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=X_1.shape[1],
        max_features=int(np.sqrt(X_1.shape[1])),
        random_state=random_state0+seed, 
        n_jobs=-1
    )
    rf.fit(X_1, labels_1)
    labels_pred = rf.predict(X_test)
    ari_pred = adjusted_rand_score(labels_test, labels_pred)
    
    return [
        ari_stab, ari_pred, 
        labels_1, labels_test, labels_pred, labels_2,
        P_1_idx, P_test_idx, P_2_idx,
        normalization_params, dim_reduction_params, 
        normalization_name, dim_reduction_name
    ]