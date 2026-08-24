"""Console output helpers for CARVE runs."""

import numpy as np
import pandas as pd
from typing import Any
from sklearn.base import ClusterMixin
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from ._sweep import SweepSpec
from ._types import EstimatorRecord, GridSpec


def _print_run_header(
    X: np.ndarray,
    sweep: SweepSpec,
    n_resamples: int,
    subsample_ratio: float,
    estimator_grids: list[GridSpec],
    n_jobs: int,
    randomize_preprocessing: bool,
    random_state: int | None,
    verbose: int,
) -> None:
    """Print a standard header describing the validation configuration.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    sweep : SweepSpec
        The swept hyperparameter axis.
    n_resamples : int
        Number of resampling iterations.
    subsample_ratio : float
        Proportion of samples in each subsample.
    estimator_grids : list of tuple
        Estimator classes and parameter grids.
    n_jobs : int
        Parallelism used for runs.
    randomize_preprocessing : bool
        Whether preprocessing is randomized.
    random_state : int or None
        Random seed.
    verbose : int
        Verbosity level; prints only if >= 2.
    """
    if verbose < 2:
        return

    total_configs = sum(len(ParameterGrid(g)) for _, g in estimator_grids)
    line = "=" * 60

    print(f"[CARVE] {line}")
    print("[CARVE] Validation Settings:")
    print(f"[CARVE] n_samples          : {X.shape[0]}")
    print(f"[CARVE] n_features         : {X.shape[1]}")
    print(f"[CARVE] sweep parameter    : {sweep.param}")
    print(f"[CARVE] {sweep.param:<19}: {sweep.values}")
    print(f"[CARVE] n_resamples        : {n_resamples}")
    print(f"[CARVE] subsample_ratio    : {subsample_ratio}")
    print(f"[CARVE] n_jobs             : {n_jobs}")
    print(f"[CARVE] total configs      : {total_configs}")
    print(f"[CARVE] randomize_preproc  : {randomize_preprocessing}")
    print(f"[CARVE] random_state       : {random_state}")
    print(f"[CARVE] {line}")
    print("\n[CARVE] Starting validation ...\n")


def _print_run_footer(
    estimator_df: pd.DataFrame,
    verbose: int,
) -> None:
    """Print a standard footer after validation completes.

    Parameters
    ----------
    estimator_df : pandas.DataFrame
        Results table with one row per estimator configuration.
    verbose : int
        Verbosity level; prints only if >= 2.
    """
    if verbose < 2:
        return
    print(
        f"\n[CARVE] finished. evaluated {len(estimator_df)} estimator configurations."
    )


def _log_config_progress(
    config_idx: int,
    total_configs: int,
    est_class: type[ClusterMixin],
    params: dict[str, Any],
    record: EstimatorRecord,
    pbar_obj: tqdm | None,
    sweep_param: str = "n_clusters",
    verbose: int = 0,
) -> None:
    """Log per-configuration progress during grid evaluation.

    Parameters
    ----------
    config_idx : int
        1-based index of the current configuration.
    total_configs : int
        Total number of configurations in the grid.
    est_class : type
        Estimator class being evaluated.
    params : dict
        Hyperparameters for the current configuration.
    record : dict
        Metrics for the current configuration.
    pbar_obj : tqdm or None
        Optional progress-bar object used to display messages.
    sweep_param : str, default="n_clusters"
        Name of the swept hyperparameter to report.
    verbose : int, default=0
        Verbosity level; logs only if >= 1.
    """
    if verbose <= 0:
        return

    sweep_value = params.get(sweep_param, "?")
    msg = (
        f"[CARVE] [{config_idx}/{total_configs}] "
        f"est={est_class.__name__} "
        f"{sweep_param}={sweep_value} | "
        f"ARI_stab={record['ari_stability']:.3f}±{record['ari_stability_se']:.3f}  "
        f"ARI_gen={record['ari_generalizability']:.3f}±{record['ari_generalizability_se']:.3f}  "
    )

    if pbar_obj is not None:
        pbar_obj.write(msg)
    else:
        print(msg)
