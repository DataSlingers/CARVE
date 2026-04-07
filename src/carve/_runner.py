"""Core validation runner for CARVE.

Coordinates the resampling loop: for each estimator configuration it
draws subsamples, runs clustering, collects ARI scores, builds consensus
matrices, and computes generalizability scores.
"""

import warnings
from typing import Any, NamedTuple

from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from ._output import _log_config_progress
from ._consensus import compute_consensus_matrix
from ._accuracy import compute_generalizability_scores
from ._pipeline import build_preprocessing_pipeline
from ._utils import cluster_labels, split_subsample_indices, _summarize_ari_scores

from ._types import (
    EstimatorRecord,
    GridSpec,
    PipelineRecord,
    PreprocSpec,
    RunMode,
    resolve_mode,
)

# Full return type for run_validation
ValidationReturn = tuple[
    list[EstimatorRecord],
    list[PipelineRecord],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]


class ResampleResult(NamedTuple):
    """Results from a single resampling iteration.

    Attributes
    ----------
    ari_stability : float
        Adjusted Rand Index between overlapping samples of two
        independent subsamples. ``NaN`` when stability is skipped.
    ari_generalizability : float
        ARI between held-out true labels and random-forest predictions.
        ``NaN`` when generalizability is skipped.
    labels_train : ndarray
        Cluster labels for the training subsample.
    labels_test : ndarray or None
        Cluster labels for the held-out test set.
    labels_predicted : ndarray or None
        Random-forest predicted labels for the test set.
    labels_stability : ndarray or None
        Cluster labels for the second independent subsample.
    train_indices : ndarray
        Row indices of the training subsample in the original data.
    test_indices : ndarray
        Row indices of the held-out test set.
    stability_indices : ndarray or None
        Row indices of the second subsample used for stability.
    normalization_params : dict
        Parameters of the normalization transformer used.
    dim_reduction_params : dict
        Parameters of the dimensionality reduction transformer used.
    normalization_name : str
        Class name of the normalization transformer.
    dim_reduction_name : str
        Class name of the dimensionality reduction transformer.
    """

    ari_stability: float
    ari_generalizability: float
    labels_train: np.ndarray
    labels_test: np.ndarray
    labels_predicted: np.ndarray
    labels_stability: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    stability_indices: np.ndarray
    normalization_params: dict[str, Any]
    dim_reduction_params: dict[str, Any]
    normalization_name: str
    dim_reduction_name: str


def run_validation(
    X: np.ndarray,
    estimator_grids: list[GridSpec],
    n_resamples: int,
    subsample_ratio: float,
    normalization_options: list[PreprocSpec],
    dim_reduction_options: list[PreprocSpec],
    randomize_preprocessing: bool = False,
    n_jobs: int = 1,
    random_state: int = None,
    show_progress: bool = False,
    mode: RunMode = "default",
    verbose: int = 0,
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
    normalization_options : list
        Normalization preprocessing options.
    dim_reduction_options : list
        Dimensionality reduction options.
    randomize_preprocessing : bool, default=False
        Whether to sample preprocessing randomly per resample.
    n_jobs : int, default=1
        Number of parallel jobs for resamples.
    random_state : int or None, default=None
        Random seed for reproducibility.
    show_progress : bool, default=False
        If True, display a progress bar for grid configurations.
    mode : Literal['default', 'stability', 'generalizability'], default='default'
        Determines whether to run CARVE regularly ('default') or whether
        to only run stability analysis ('stability'),
        or generalizability analysis ('generalizability').
    verbose : int, default=0
        Verbosity for logging. ``0`` suppresses all output, ``1`` prints
        per-configuration progress, ``2`` includes header and footer.

    Returns
    -------
    estimator_records : list of dict
        One record per estimator configuration with aggregate metrics.
    pipeline_records : list of dict
        Optional per-resample preprocessing records when randomized.
    consensus_matrices : list of ndarray
        Consensus matrices for each configuration.
    consensus_generalizability_matrices : list of ndarray
        Generalizability consensus matrices for each configuration.
    generalizability_scores : list of ndarray
        Per-sample generalizability arrays for each configuration.
    """
    policy = resolve_mode(mode)

    estimator_records: list[EstimatorRecord] = []
    pipeline_records: list[PipelineRecord] = []

    consensus_matrices: list[np.ndarray] = []
    consensus_generalizability_matrices: list[np.ndarray] = []

    generalizability_scores: list[np.ndarray] = []

    total_configs = sum(len(list(ParameterGrid(g))) for _, g in estimator_grids)
    config_idx = 0

    n = X.shape[0]

    with tqdm(
        total=total_configs, desc="Grid configs", disable=not show_progress
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
                        normalization_options=normalization_options,
                        dim_reduction_options=dim_reduction_options,
                        randomize_preprocessing=randomize_preprocessing,
                        mode=mode,
                        random_state=random_state,
                    )
                    for b in range(n_resamples)
                )

                # --- Collect per-resample ARI scores ---
                aris_stab = [r.ari_stability for r in results]
                aris_gen = [r.ari_generalizability for r in results]
                aris_avg = (
                    [(r.ari_stability + r.ari_generalizability) / 2 for r in results]
                    if policy.compute_average_ari
                    else [np.nan] * n_resamples
                )

                # --- Build consensus matrices ---
                M = (
                    compute_consensus_matrix(
                        n_samples=n,
                        runs=[(r.train_indices, r.labels_train) for r in results],
                    )
                    if policy.run_stability
                    else None
                )

                M_g = (
                    compute_consensus_matrix(
                        n_samples=n,
                        runs=[(r.test_indices, r.labels_predicted) for r in results],
                    )
                    if policy.run_generalizability
                    else None
                )

                # --- Compute generalizability scores ---
                E = (
                    compute_generalizability_scores(
                        n_samples=n,
                        runs=[
                            (r.test_indices, r.labels_test, r.labels_predicted)
                            for r in results
                        ],
                    )
                    if policy.run_generalizability
                    else None
                )

                consensus_matrices.append(M)
                consensus_generalizability_matrices.append(M_g)
                generalizability_scores.append(E)

                # --- Summarize ARI statistics ---
                stab_mean, stab_se, stab_q95, stab_q05 = _summarize_ari_scores(
                    aris_stab, n_resamples
                )
                gen_mean, gen_se, gen_q95, gen_q05 = _summarize_ari_scores(
                    aris_gen, n_resamples
                )
                avg_mean, avg_se, avg_q95, avg_q05 = _summarize_ari_scores(
                    aris_avg, n_resamples
                )
                record: EstimatorRecord = {
                    "estimator": est_class.__name__,
                    **params,
                    "ari_stability": stab_mean,
                    "ari_stability_se": stab_se,
                    "ari_stability_upper": stab_q95,
                    "ari_stability_lower": stab_q05,
                    "ari_generalizability": gen_mean,
                    "ari_generalizability_se": gen_se,
                    "ari_generalizability_upper": gen_q95,
                    "ari_generalizability_lower": gen_q05,
                    "ari_average": avg_mean,
                    "ari_average_se": avg_se,
                    "ari_average_upper": avg_q95,
                    "ari_average_lower": avg_q05,
                }
                estimator_records.append(record)

                if randomize_preprocessing:
                    pipeline_records.append(
                        {
                            "estimator": est_class.__name__,
                            "params": params,
                            "results": results,
                        }
                    )

                _log_config_progress(
                    config_idx=config_idx,
                    total_configs=total_configs,
                    est_class=est_class,
                    params=params,
                    record=record,
                    pbar_obj=pbar if show_progress else None,
                    verbose=verbose,
                )

                pbar.update(1)

    return (
        estimator_records,
        pipeline_records,
        consensus_matrices,
        consensus_generalizability_matrices,
        generalizability_scores,
    )


def validation_iter(
    X: np.ndarray,
    est_class: type,
    params: dict[str, Any],
    subsample_ratio: float,
    n_resamples: int,
    seed: int,
    normalization_options: list[PreprocSpec],
    dim_reduction_options: list[PreprocSpec],
    randomize_preprocessing: bool = False,
    mode: RunMode = "default",
    random_state: int = None,
) -> ResampleResult:
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
    normalization_options : list
        Normalization options.
    dim_reduction_options : list
        Dimensionality reduction options.
    randomize_preprocessing : bool, default=False
        Whether to randomize preprocessing.
    mode : Literal['default', 'stability', 'generalizability'], default='default'
        Determines whether to run CARVE regularly ('default') or whether
        to only run stability analysis ('stability'),
        or generalizability analysis ('generalizability').
    random_state : int or None, default=None
        Base random seed for reproducibility.

    Returns
    -------
    result : ResampleResult
        Named tuple containing ARI metrics, labels, sample indices, and
        preprocessing metadata for this resample.
    """
    policy = resolve_mode(mode)

    n_samples = X.shape[0]
    random_state0 = random_state if random_state is not None else 0

    # --- Subsample indices ---
    P_1_idx, P_test_idx = split_subsample_indices(
        n_samples, subsample_ratio=subsample_ratio, random_state=random_state0 + seed
    )

    P_2_idx = None
    if policy.run_stability:
        P_2_idx, _ = split_subsample_indices(
            n_samples,
            subsample_ratio=subsample_ratio,
            random_state=random_state0 + seed + n_resamples,
        )

    # --- Preprocessing ---
    (
        pipeline,
        normalization_params,
        dim_reduction_params,
        normalization_name,
        dim_reduction_name,
    ) = build_preprocessing_pipeline(
        randomize_preprocessing=randomize_preprocessing,
        normalization_options=normalization_options,
        dim_reduction_options=dim_reduction_options,
        seed=random_state0 + seed,
    )

    X_preprocessed = pipeline.fit_transform(X)

    X_1 = X_preprocessed[P_1_idx]
    X_test = X_preprocessed[P_test_idx] if policy.run_generalizability else None
    X_2 = X_preprocessed[P_2_idx] if policy.run_stability else None

    # --- Clustering ---
    labels_1 = cluster_labels(
        X_1, est_class, random_state=random_state0 + seed, **params
    )

    labels_test = (
        cluster_labels(X_test, est_class, random_state=random_state0 + seed, **params)
        if policy.run_generalizability
        else None
    )

    labels_2 = (
        cluster_labels(X_2, est_class, random_state=random_state0 + seed, **params)
        if policy.run_stability
        else None
    )

    # --- Cluster count sanity checks ---
    n_clusters = params.get("n_clusters")

    if len(np.unique(labels_1)) != n_clusters:
        warnings.warn(
            f"labels_1 has {len(np.unique(labels_1))} clusters, expected {n_clusters}"
        )

    if policy.run_generalizability and len(np.unique(labels_test)) != n_clusters:
        warnings.warn(
            f"labels_test has {len(np.unique(labels_test))} clusters, expected {n_clusters}"
        )

    if policy.run_stability and len(np.unique(labels_2)) != n_clusters:
        warnings.warn(
            f"labels_2 has {len(np.unique(labels_2))} clusters, expected {n_clusters}"
        )

    # --- Stability ARI (overlap between two independent subsamples) ---
    ari_stab = _compute_stability_ari(policy, P_1_idx, P_2_idx, labels_1, labels_2)

    # --- Generalizability ARI (RF prediction on held-out set) ---
    labels_pred, ari_pred = _compute_generalizability_ari(
        policy=policy,
        X_1=X_1,
        X_test=X_test,
        labels_1=labels_1,
        labels_test=labels_test,
        seed=random_state0 + seed,
    )

    return ResampleResult(
        ari_stability=ari_stab,
        ari_generalizability=ari_pred,
        labels_train=labels_1,
        labels_test=labels_test,
        labels_predicted=labels_pred,
        labels_stability=labels_2,
        train_indices=P_1_idx,
        test_indices=P_test_idx,
        stability_indices=P_2_idx,
        normalization_params=normalization_params,
        dim_reduction_params=dim_reduction_params,
        normalization_name=normalization_name,
        dim_reduction_name=dim_reduction_name,
    )


def _compute_stability_ari(
    policy,
    P_1_idx,
    P_2_idx,
    labels_1,
    labels_2,
):
    """Compute stability ARI on overlapping samples of two subsamples.

    Parameters
    ----------
    policy : ModePolicy
        Execution policy; stability is skipped when
        ``policy.run_stability`` is False.
    P_1_idx : ndarray
        Indices of the first subsample.
    P_2_idx : ndarray or None
        Indices of the second subsample.
    labels_1 : ndarray
        Cluster labels for the first subsample.
    labels_2 : ndarray or None
        Cluster labels for the second subsample.

    Returns
    -------
    ari : float
        Adjusted Rand Index on overlapping samples, or ``NaN`` when
        stability is skipped.
    """
    if not policy.run_stability:
        return np.nan

    _, i_1, i_2 = np.intersect1d(P_1_idx, P_2_idx, return_indices=True)

    return adjusted_rand_score(labels_1[i_1], labels_2[i_2])


def _compute_generalizability_ari(
    policy,
    X_1,
    X_test,
    labels_1,
    labels_test,
    seed,
):
    """Compute generalizability ARI via random-forest prediction.

    Parameters
    ----------
    policy : ModePolicy
        Execution policy; generalizability is skipped when
        ``policy.run_generalizability`` is False.
    X_1 : ndarray
        Training features.
    X_test : ndarray or None
        Held-out test features.
    labels_1 : ndarray
        Cluster labels for the training set.
    labels_test : ndarray or None
        Cluster labels for the test set.
    seed : int
        Random seed for the random forest.

    Returns
    -------
    labels_pred : ndarray or None
        Predicted labels for the test set, or None when skipped.
    ari : float
        Adjusted Rand Index between test labels and predictions, or
        ``NaN`` when skipped.
    """
    if not policy.run_generalizability:
        return None, np.nan

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=X_1.shape[1],
        max_features=int(np.sqrt(X_1.shape[1])),
        random_state=seed,
        n_jobs=-1,
    )

    rf.fit(X_1, labels_1)

    labels_pred = rf.predict(X_test)
    ari_pred = adjusted_rand_score(labels_test, labels_pred)

    return labels_pred, ari_pred
