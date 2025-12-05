import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Type, Union
from sklearn.base import ClusterMixin
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

ModelRecord = Dict[str, Any]
GridSpec = Tuple[Type[ClusterMixin], Dict[str, List[Any]]]

def _print_run_header(
    X: np.ndarray,
    K: Union[int, np.ndarray],
    B: int,
    rho: float,
    model_grids: List[GridSpec],
    n_jobs: int,
    random_preprocess: bool,
    random_state: int | None,
    verbose: int
) -> None:
    if verbose < 1:
        return

    total_configs = sum(len(ParameterGrid(g)) for _, g in model_grids)
    line = "=" * 60

    print(f"[CARVE] {line}")
    print(f"[CARVE] starting validation run")
    print(f"[CARVE] n_samples         : {X.shape[0]}")
    print(f"[CARVE] n_features        : {X.shape[1]}")
    print(f"[CARVE] K                 : {K}")
    print(f"[CARVE] B (resamples)     : {B}")
    print(f"[CARVE] rho               : {rho}")
    print(f"[CARVE] n_jobs            : {n_jobs}")
    print(f"[CARVE] total configs     : {total_configs}")
    print(f"[CARVE] random_preprocess : {random_preprocess}")
    print(f"[CARVE] random_state      : {random_state}")
    print(f"[CARVE] {line}")
    print(f"[CARVE] Starting validation ...\n\n")
    
def _print_run_footer(
    model_df: pd.DataFrame,
    verbose: int
) -> None:
    if verbose < 1 or verbose is None:
        return
    print(f"\n[CARVE] finished. evaluated {len(model_df)} model configurations.")

def _log_config_progress(
    config_idx: int,
    total_configs: int,
    est_class: Type[ClusterMixin],
    params: Dict[str, Any],
    record: ModelRecord,
    pbar_obj: tqdm | None,
    verbose: int = 1
) -> None:
    if verbose <= 0:
        return

    n_clusters = params.get("n_clusters", "?")
    msg = (
        f"[CARVE] [{config_idx}/{total_configs}] "
        f"est={est_class.__name__} "
        f"n_clusters={n_clusters} | "
        f"ARI_stab={record['ari_stability']:.3f}±{record['ari_stability_se']:.3f} "
        f"ARI_gen={record['ari_generalizability']:.3f}±{record['ari_generalizability_se']:.3f} "
        f"ARI_avg={record['ari_average']:.3f}±{record['ari_average_se']:.3f}"
    )

    if pbar_obj is not None:
        pbar_obj.write(msg)
    else:
        print(msg)