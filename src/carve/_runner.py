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
from ._utils import clustering_pipeline, subsample_indices, align_labels
from ._plotting import plot_cluster_stability


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
    ref_labels: np.ndarray,
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
                
                if ref_labels is None:
                    ref_clust = clustering_pipeline(X, est_class, random_state, **params)
                else:
                    ref_clust = ref_labels
                
                results = Parallel(n_jobs=n_jobs)(
                    worker(
                        X=X, 
                        est_class=est_class, 
                        params=params, 
                        rho=rho, 
                        B=B, 
                        ref_clust=ref_clust,
                        seed=b, 
                        norm_options=norm_options, 
                        dr_options=dr_options,
                        random_preprocess=random_preprocess, 
                        random_state=random_state
                    ) 
                    for b in range(B)
                )
                
                aris_stab = [r[0] for r in results]
                aris_pred = [r[1] for r in results]
                
                M = build_consensus_matrix(n=n, runs=[(r[6], r[2]) for r in results], return_counts=False)  # r[6]: P_1_idx, r[2]: labels_1
                E = build_generalizability_array(n=n, runs=[(r[7], r[3], r[4]) for r in results])           # r[7]: P_test_idx, r[3]: labels_test, # r[4]: labels_pred

                cons_mats_raw.append(M)
                generalizability_arrs.append(E)
                
                model_records.append({
                    'estimator': est_class.__name__,
                    **params,
                    'ari_stability': np.mean(aris_stab),
                    'ari_stability_se': np.std(aris_stab, ddof=1) / np.sqrt(B),
                    'ari_generalizability': np.mean(aris_pred),
                    'ari_generalizability_se': np.std(aris_pred, ddof=1) / np.sqrt(B)
                })
                
                if random_preprocess:
                    pipeline_records.append({
                        'estimator': est_class.__name__, 
                        'params': params, 
                        'results': results
                    })
                    
                # plot_cluster_stability(
                #     X=X, 
                #     results=results
                # )  # for de-bugging
                
                pbar.update(1)
                
    return model_records, pipeline_records, cons_mats_raw, generalizability_arrs

def validation_iter(
    X: np.ndarray,
    est_class: Type,
    params: Dict[str, Any],
    rho: float,
    B: int,
    ref_clust: np.ndarray,
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
        random_preprocess, norm_options, dr_options, random_state0 + seed
    )

    X_1 = pipeline.fit_transform(X[P_1_idx])
    X_test = pipeline.fit_transform(X[P_test_idx])
    X_2 = pipeline.fit_transform(X[P_2_idx])
    
    # clustering 
    labels_1_raw = clustering_pipeline(X_1, est_class, random_state=random_state0+seed, **params)
    labels_test_raw = clustering_pipeline(X_test, est_class, random_state=random_state0+seed, **params)
    labels_2_raw = clustering_pipeline(X_2, est_class, random_state=random_state0+seed, **params)
    
    # align to reference clustering
    labels_1 = align_labels(ref_clust[P_1_idx], labels_1_raw)
    labels_test = align_labels(ref_clust[P_test_idx], labels_test_raw)
    labels_2 = align_labels(ref_clust[P_2_idx], labels_2_raw)
    
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