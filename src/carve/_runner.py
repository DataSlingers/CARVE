from typing import Any, Callable, Dict, List, Tuple, Type
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from ._consensus import build_consensus_matrix
from ._misclassification import build_generalizability_array
from ._pipeline import create_pipeline
from ._utils import clustering_pipeline, subsample_indices
import warnings


ModelRecord = Dict[str, Any]
PipelineRecord = Dict[str, Any]

ValidationReturn = Tuple[
    List[ModelRecord],      # model_records | TODO: check whether this type suggestion is correct
    List[PipelineRecord],   # pipeline_records | TODO: check whether this type suggestion is correct
    List[np.ndarray],       # consensus_mats_raw
    List[np.ndarray],       # generalizability_arrs
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
    model_grids: List[GridSpec],
    B: int,
    rho: float,
    norm_options: List[PreprocSpec],
    dr_options: List[PreprocSpec], 
    random_preprocess: bool = False, 
    n_jobs: int = 1, 
    random_state: int = None,
    prog_bar: bool = True
) -> ValidationReturn:
    n = X.shape[0]
    
    model_records = []
    pipeline_records = []
    cons_mats_raw = []
    generalizability_arrs = []
    
    total_configs = sum(len(list(ParameterGrid(g))) for _, g in model_grids)
    with tqdm(total=total_configs, desc="Grid configs", disable=not prog_bar) as pbar:
        for est_class, grid in model_grids:
            for params in ParameterGrid(grid):
                worker = delayed(validation_iter)
                results = Parallel(n_jobs=n_jobs)(
                    worker(
                        X=X, 
                        est_class=est_class, 
                        params=params, 
                        rho=rho, 
                        B=B, 
                        seed=b, 
                        norm_options=norm_options, 
                        dr_options=dr_options,
                        random_preprocess=random_preprocess, 
                        random_state=random_state
                    ) 
                    for b in range(B)
                )
                
                aris_stab = [r[0] for r in results]
                aris_gen = [r[1] for r in results]
                aris_avg = [(r[0] + r[1]) / 2 for r in results]
                
                M = build_consensus_matrix(
                    n=n, 
                    runs=[(r[6], r[2]) for r in results],        # r[6]: P_1_idx, r[2]: labels_1
                )
                
                E = build_generalizability_array(
                    n=n, 
                    runs=[(r[7], r[3], r[4]) for r in results]  # r[7]: P_test_idx, r[3]: labels_test, # r[4]: labels_pred
                )                      
                
                cons_mats_raw.append(M)
                generalizability_arrs.append(E)
                
                model_records.append({
                    'estimator': est_class.__name__,
                    **params,
                    'ari_stability': np.mean(aris_stab),
                    'ari_stability_se': np.std(aris_stab, ddof=1) / np.sqrt(B),
                    'ari_stability_upper': np.quantile(aris_stab, 0.95),
                    'ari_stability_lower': np.quantile(aris_stab, 0.05),
                    'ari_generalizability': np.mean(aris_gen),
                    'ari_generalizability_se': np.std(aris_gen, ddof=1) / np.sqrt(B),
                    'ari_generalizability_upper': np.quantile(aris_gen, 0.95),
                    'ari_generalizability_lower': np.quantile(aris_gen, 0.05),
                    'ari_average': np.mean(aris_avg),
                    'ari_average_se': np.std(aris_avg, ddof=1) / np.sqrt(B),
                    'ari_average_upper': np.quantile(aris_avg, 0.95),
                    'ari_average_lower': np.quantile(aris_avg, 0.05)
                })
                
                if random_preprocess:
                    pipeline_records.append({
                        'estimator': est_class.__name__, 
                        'params': params, 
                        'results': results
                    })
                
                pbar.update(1)
                
    return model_records, pipeline_records, cons_mats_raw, generalizability_arrs

def validation_iter(
    X: np.ndarray,
    est_class: Type,
    params: Dict[str, Any],
    rho: float,
    B: int,
    seed: int,
    norm_options: List[PreprocSpec],
    dr_options: List[PreprocSpec],
    random_preprocess: bool = False,
    random_state: int = None
) -> ResultTuple:
    n_samples = X.shape[0]
    random_state0 = random_state if random_state is not None else 0
    
    P_1_idx, P_test_idx = subsample_indices(n_samples, ratio=rho, random_state=random_state0+seed)
    P_2_idx, _ = subsample_indices(n_samples, ratio=rho, random_state=random_state0+seed+B)
    
    pipeline, norm_params, dr_params, norm_name, dr_name = create_pipeline(
        random_preprocess=random_preprocess, 
        norm_options=norm_options, 
        dr_options=dr_options, 
        seed=random_state0 + seed
    )
    
    X_prepocessed = pipeline.fit_transform(X)

    X_1 = X_prepocessed[P_1_idx]
    X_test = X_prepocessed[P_test_idx]
    X_2 = X_prepocessed[P_2_idx]
    
    # clustering 
    labels_1 = clustering_pipeline(X_1, est_class, random_state=random_state0+seed, **params)
    labels_test = clustering_pipeline(X_test, est_class, random_state=random_state0+seed, **params)
    labels_2 = clustering_pipeline(X_2, est_class, random_state=random_state0+seed, **params)
    
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
        norm_params, dr_params, 
        norm_name, dr_name
    ]