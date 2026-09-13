"""Core validation runner for CARVE.

Coordinates the resampling loop: for each estimator configuration it
draws subsamples, runs clustering, collects ARI scores, builds consensus
matrices, and computes generalizability scores. Under randomized
preprocessing a precompute pass first fits each resample's pipeline on each
of its subsamples, once per resample, before any configuration runs.
"""

import warnings
from typing import Any, NamedTuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from ._accuracy import compute_generalizability_scores
from ._consensus import (
    compute_consensus_matrix,
    compute_consensus_pac,
    consensus_anchor_block,
    stability_from_consensus,
    stability_from_runs_anchored,
)
from ._output import _log_config_progress
from ._pipeline import PipelineSpec, allocate_pipelines, pipeline_from_spec
from ._sweep import MethodIds, SweepSpec
from ._sweep import resolve_sweep as _resolve_sweep
from ._types import (
    ConsensusSummary,
    EstimatorRecord,
    GridSpec,
    NoisePolicy,
    PipelineRecord,
    PreprocSpec,
    RunMode,
    resolve_mode,
)
from ._utils import (
    _summarize_ari_scores,
    apply_noise_policy,
    cluster_labels,
    count_clusters,
    default_generalizability_classifier,
    resolve_core_budget,
    split_subsample_indices,
)

# Full return type for run_validation
ValidationReturn = tuple[
    list[EstimatorRecord],
    list[PipelineRecord],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[ConsensusSummary] | None,
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
    pipeline : PipelineSpec or None
        The preprocessing pipeline this resample used, or None when
        preprocessing was not randomized.
    n_clusters_train : int
        Number of non-noise clusters found on the training subsample.
    n_clusters_test : int
        Number of non-noise clusters found on the held-out set.
    n_clusters_stability : int
        Number of non-noise clusters found on the second subsample.
    noise_fraction : float
        Fraction of training-subsample points labelled as noise before the
        noise policy was applied.
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
    pipeline: PipelineSpec | None
    n_clusters_train: int
    n_clusters_test: int
    n_clusters_stability: int
    noise_fraction: float


class ResampleEmbeddings(NamedTuple):
    """One resample's subsamples, each embedded by its own pipeline fit.

    The pipeline is fitted separately on every subsample, so the arrays come
    from independent fits and do not share coordinates.

    Attributes
    ----------
    spec : PipelineSpec
        The pipeline allocated to this resample.
    X_1 : ndarray of shape (n_train, d)
        Embedding of the first subsample.
    X_2 : ndarray of shape (n_train, d) or None
        Embedding of the second subsample; None when stability is skipped.
    X_test : ndarray of shape (n_test, d) or None
        Embedding of the held-out set; None when generalizability is skipped.
    """

    spec: PipelineSpec
    X_1: np.ndarray
    X_2: np.ndarray | None
    X_test: np.ndarray | None


def _resample_indices(
    n_samples: int,
    *,
    seed: int,
    n_resamples: int,
    subsample_ratio: float,
    random_state: int | None,
    run_stability: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return one resample's subsample indices: P_1, P_test and P_2.

    The only place these seeds are derived, so the precompute pass and the
    configuration loop draw the same subsamples by construction. P_2 is None
    when stability is skipped.
    """
    random_state0 = random_state if random_state is not None else 0

    P_1_idx, P_test_idx = split_subsample_indices(
        n_samples, subsample_ratio=subsample_ratio, random_state=random_state0 + seed
    )

    P_2_idx = None
    if run_stability:
        P_2_idx, _ = split_subsample_indices(
            n_samples,
            subsample_ratio=subsample_ratio,
            random_state=random_state0 + seed + n_resamples,
        )

    return P_1_idx, P_test_idx, P_2_idx


def embed_resample(
    X: np.ndarray,
    spec: PipelineSpec,
    *,
    seed: int,
    n_resamples: int,
    subsample_ratio: float,
    random_state: int | None,
    mode: RunMode = "default",
) -> ResampleEmbeddings:
    """Fit one resample's pipeline on each subsample the mode needs.

    The pipeline fitted on P_1 is seeded with ``random_state + seed``, on
    P_2 with ``random_state + seed + n_resamples`` and on P_test with
    ``random_state + seed + 2 * n_resamples``, so no two fits share a seed
    and no RNG state crosses a joblib boundary.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    spec : PipelineSpec
        The pipeline allocated to this resample.
    seed : int
        Resample index, the per-resample seed offset.
    n_resamples : int
        Total resamples, used to offset seeds.
    subsample_ratio : float
        Proportion of samples in each subsample.
    random_state : int or None
        Base seed; None means 0.
    mode : {"default", "stability", "generalizability"}, default="default"
        P_2 is embedded only when stability runs, P_test only when
        generalizability runs.

    Returns
    -------
    embeddings : ResampleEmbeddings
    """
    policy = resolve_mode(mode)
    random_state0 = random_state if random_state is not None else 0

    P_1_idx, P_test_idx, P_2_idx = _resample_indices(
        X.shape[0],
        seed=seed,
        n_resamples=n_resamples,
        subsample_ratio=subsample_ratio,
        random_state=random_state0,
        run_stability=policy.run_stability,
    )

    base = random_state0 + seed
    X_2 = X_test = None

    with warnings.catch_warnings():
        # umap-learn warns on every seeded fit that the seed disables its own
        # parallelism. CARVE seeds every transformer so the fit is
        # reproducible and parallelizes over resamples instead, so the warning
        # carries nothing here, and under warnings-as-errors it aborts the fit.
        warnings.filterwarnings(
            "ignore",
            message=r"n_jobs value .* overridden to 1 by setting random_state",
            category=UserWarning,
        )
        X_1 = np.asarray(pipeline_from_spec(spec, base).fit_transform(X[P_1_idx]))

        if policy.run_stability:
            pipeline = pipeline_from_spec(spec, base + n_resamples)
            X_2 = np.asarray(pipeline.fit_transform(X[P_2_idx]))

        if policy.run_generalizability:
            pipeline = pipeline_from_spec(spec, base + 2 * n_resamples)
            X_test = np.asarray(pipeline.fit_transform(X[P_test_idx]))

    return ResampleEmbeddings(spec=spec, X_1=X_1, X_2=X_2, X_test=X_test)


def precompute_embeddings(
    X: np.ndarray,
    pipelines: list[PipelineSpec],
    *,
    subsample_ratio: float,
    random_state: int | None,
    mode: RunMode,
    n_jobs: int,
    show_progress: bool = False,
) -> list[ResampleEmbeddings]:
    """Embed every resample's subsamples once, before any configuration runs.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    pipelines : list of PipelineSpec
        Entry ``b`` is resample ``b``'s pipeline, from
        ``_pipeline.allocate_pipelines``; its length is the resample count.
    subsample_ratio : float
        Proportion of samples in each subsample.
    random_state : int or None
        Base seed; None means 0.
    mode : {"default", "stability", "generalizability"}
        Which subsamples to embed.
    n_jobs : int
        Worker count over resamples, the same one the configuration loop uses.
    show_progress : bool, default=False
        Display a tqdm bar labeled "Preprocessing".

    Returns
    -------
    embeddings : list of ResampleEmbeddings
        Entry ``b`` belongs to resample ``b``.
    """
    n_resamples = len(pipelines)
    worker = delayed(embed_resample)
    tasks = (
        worker(
            X,
            pipelines[b],
            seed=b,
            n_resamples=n_resamples,
            subsample_ratio=subsample_ratio,
            random_state=random_state,
            mode=mode,
        )
        for b in range(n_resamples)
    )
    results = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
    return list(
        tqdm(
            results,
            total=n_resamples,
            desc="Preprocessing",
            disable=not show_progress,
        )
    )


def run_validation(
    X: np.ndarray,
    estimator_grids: list[GridSpec],
    n_resamples: int,
    subsample_ratio: float,
    normalization_options: list[PreprocSpec],
    dim_reduction_options: list[PreprocSpec],
    classifier: ClassifierMixin | None = None,
    n_trees: int = 100,
    randomize_preprocessing: bool = False,
    n_jobs: int = 1,
    random_state: int = None,
    sweep: SweepSpec | None = None,
    noise_policy: NoisePolicy = "drop",
    show_progress: bool = False,
    mode: RunMode = "default",
    verbose: int = 0,
    anchors: np.ndarray | None = None,
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
    classifier : sklearn classifier instance, optional
        Classifier used to score generalizability.
    n_trees : int, default=100
        Number of trees in the default random-forest classifier.
    randomize_preprocessing : bool, default=False
        Whether to allocate one pipeline per resample from the option lists
        (``_pipeline.allocate_pipelines``) and cluster each subsample's own
        embedding of it. The embeddings are computed once, before the
        configuration loop, by ``precompute_embeddings``.
    n_jobs : int, default=1
        Core budget for the run. Split once by ``resolve_core_budget`` into
        workers over resamples and threads per worker for the classifier.
        The precompute pass uses the same worker count.
    random_state : int or None, default=None
        Random seed for reproducibility.
    sweep : SweepSpec, optional
        The swept hyperparameter axis. Defaults to an ``n_clusters`` sweep
        inferred from the first estimator grid.
    noise_policy : {"drop", "as_cluster", "singleton"}, default="drop"
        How to resolve negative labels emitted by density-based methods.
    show_progress : bool, default=False
        If True, display a progress bar for grid configurations.
    mode : Literal['default', 'stability', 'generalizability'], default='default'
        Determines whether to run CARVE regularly ('default') or whether
        to only run stability analysis ('stability'),
        or generalizability analysis ('generalizability').
    verbose : int, default=0
        Verbosity for logging. ``0`` suppresses all output, ``1`` prints
        per-configuration progress, ``2`` includes header and footer.
    anchors : ndarray or None, default=None
        Indices of anchor samples. When given, consensus matrices are
        restricted to anchor-by-anchor blocks instead of full n-by-n
        matrices, and per-sample stability scores are computed against
        the anchor set rather than the full sample.

    Returns
    -------
    estimator_records : list of dict
        One record per estimator configuration with aggregate metrics.
    pipeline_records : list of dict
        One record per configuration when randomized, empty otherwise:
        ``method_id``, ``method_label``, ``sweep_value``, ``sweep_rank`` and
        ``results``, the configuration's ``ResampleResult`` list.
    consensus_matrices : list of ndarray
        Consensus matrices for each configuration.
    consensus_generalizability_matrices : list of ndarray
        Generalizability consensus matrices for each configuration.
    generalizability_scores : list of ndarray
        Per-sample generalizability arrays for each configuration.
    consensus_summaries : list of ConsensusSummary or None
        Full-length per-configuration stability quantities, or None when
        stability is not run.
    """
    policy = resolve_mode(mode)

    # If sweep is not provided, default to sweeping over n_clusters
    if sweep is None:
        sweep = _resolve_sweep(
            n_clusters=np.asarray(estimator_grids[0][1]["n_clusters"])
        )

    # Method identity is defined relative to the sweep axis, so this is
    # built after the sweep is resolved and shared across the whole run.
    method_ids = MethodIds(sweep.param)

    estimator_records: list[EstimatorRecord] = []
    pipeline_records: list[PipelineRecord] = []

    consensus_matrices: list[np.ndarray] = []
    consensus_generalizability_matrices: list[np.ndarray] = []

    generalizability_scores: list[np.ndarray] = []
    consensus_summaries: list[ConsensusSummary | None] = []

    total_configs = sum(len(list(ParameterGrid(g))) for _, g in estimator_grids)
    config_idx = 0

    n = X.shape[0]

    # Workers over resamples, threads inside each worker's classifier. The
    # loky backend already caps BLAS and OpenMP in its workers at the same
    # cpu_count // outer, and keeps an OMP_NUM_THREADS pin from the parent;
    # the forest's own thread pool is the one thing it cannot reach.
    outer_n_jobs, classifier_n_jobs = resolve_core_budget(
        n_jobs, n_resamples=n_resamples
    )

    # --- Randomized preprocessing: allocate, then embed once per resample ---
    # Nothing here is per configuration, so the config_id alignment fit()
    # asserts is untouched.
    embeddings: list[ResampleEmbeddings] | None = None
    if randomize_preprocessing:
        pipelines = allocate_pipelines(
            normalization_options, dim_reduction_options, n_resamples, random_state
        )
        embeddings = precompute_embeddings(
            X,
            pipelines,
            subsample_ratio=subsample_ratio,
            random_state=random_state,
            mode=mode,
            n_jobs=outer_n_jobs,
            show_progress=show_progress,
        )

    with tqdm(
        total=total_configs, desc="Grid configs", disable=not show_progress
    ) as pbar:
        for est_class, grid in estimator_grids:
            for params in ParameterGrid(grid):
                config_idx += 1

                worker = delayed(validation_iter)
                results = Parallel(n_jobs=outer_n_jobs)(
                    worker(
                        X=X,
                        est_class=est_class,
                        params=params,
                        subsample_ratio=subsample_ratio,
                        n_resamples=n_resamples,
                        seed=b,
                        classifier=classifier,
                        n_trees=n_trees,
                        sweep_param=sweep.param,
                        noise_policy=noise_policy,
                        mode=mode,
                        random_state=random_state,
                        classifier_n_jobs=classifier_n_jobs,
                        # Each worker receives only its own resample's arrays.
                        embeddings=None if embeddings is None else embeddings[b],
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

                # --- Drop degenerate resamples ---
                #
                # Under noise_policy="drop" a subsample can be labelled
                # entirely as noise, leaving nothing to aggregate. Those
                # resamples already warned and carry empty or None labels;
                # they must not reach the consensus/accuracy aggregators.
                stab_runs = [
                    r
                    for r in results
                    if r.labels_train is not None and np.size(r.labels_train)
                ]
                gen_runs = [
                    r
                    for r in results
                    if r.labels_predicted is not None
                    and np.size(r.labels_predicted)
                    and r.labels_test is not None
                    and np.size(r.labels_test)
                ]

                # --- Build consensus matrices ---
                if not policy.run_stability:
                    M = None
                elif anchors is None:
                    M = compute_consensus_matrix(
                        n_samples=n,
                        runs=[(r.train_indices, r.labels_train) for r in stab_runs],
                    )
                else:
                    M = consensus_anchor_block(
                        n_samples=n,
                        runs=[(r.train_indices, r.labels_train) for r in stab_runs],
                        anchors=anchors,
                    )

                if not policy.run_generalizability:
                    M_g = None
                elif anchors is None:
                    M_g = compute_consensus_matrix(
                        n_samples=n,
                        runs=[(r.test_indices, r.labels_predicted) for r in gen_runs],
                    )
                else:
                    M_g = consensus_anchor_block(
                        n_samples=n,
                        runs=[(r.test_indices, r.labels_predicted) for r in gen_runs],
                        anchors=anchors,
                    )

                # --- Per-sample stability scores, always full length ---
                # (pac is not: it comes straight from M, which under
                # anchoring is the m-by-m anchor block, not n-by-n.)
                if not policy.run_stability:
                    summary = None
                else:
                    stab_pairs = [(r.train_indices, r.labels_train) for r in stab_runs]
                    if anchors is None:
                        gini, ce = stability_from_consensus(M)
                    else:
                        gini, ce = stability_from_runs_anchored(
                            n_samples=n, runs=stab_pairs, anchors=anchors
                        )
                    summary = ConsensusSummary(
                        gini=gini, ce=ce, pac=compute_consensus_pac(M)
                    )

                # --- Compute generalizability scores ---
                E = (
                    compute_generalizability_scores(
                        n_samples=n,
                        runs=[
                            (r.test_indices, r.labels_test, r.labels_predicted)
                            for r in gen_runs
                        ],
                    )
                    if policy.run_generalizability
                    else None
                )

                consensus_matrices.append(M)
                consensus_generalizability_matrices.append(M_g)
                generalizability_scores.append(E)
                consensus_summaries.append(summary)

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

                # --- Observed granularity (if sweep parameter does not fix k) ---
                k_obs = np.array([r.n_clusters_train for r in results], dtype=float)

                n_clusters_observed = float(np.mean(k_obs))
                n_clusters_observed_se = (
                    float(np.std(k_obs, ddof=1) / np.sqrt(k_obs.size))
                    if k_obs.size > 1
                    else np.nan
                )

                # --- Observed noise fraction ---
                noise = np.array([r.noise_fraction for r in results], dtype=float)
                noise_fraction = float(np.mean(noise))
                sweep_value = params[sweep.param]

                # --- Assign method ID for this configuration ---
                method_id, method_label = method_ids.assign(est_class.__name__, params)

                record: EstimatorRecord = {
                    # config_id keys the per-configuration artifact
                    # containers appended just above.
                    "config_id": config_idx - 1,
                    "method_id": method_id,
                    "method_label": method_label,
                    "estimator": est_class.__name__,
                    **params,
                    "sweep_param": sweep.param,
                    "sweep_value": sweep_value,
                    "sweep_rank": sweep.rank_of(sweep_value),
                    "n_clusters_observed": n_clusters_observed,
                    "n_clusters_observed_se": n_clusters_observed_se,
                    "noise_fraction": noise_fraction,
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
                            "method_id": method_id,
                            "method_label": method_label,
                            "sweep_value": sweep_value,
                            "sweep_rank": record["sweep_rank"],
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
                    sweep_param=sweep.param,
                    verbose=verbose,
                )

                pbar.update(1)

    return (
        estimator_records,
        pipeline_records,
        consensus_matrices,
        consensus_generalizability_matrices,
        generalizability_scores,
        consensus_summaries if policy.run_stability else None,
    )


def validation_iter(
    X: np.ndarray,
    est_class: type,
    params: dict[str, Any],
    subsample_ratio: float,
    n_resamples: int,
    seed: int,
    classifier: ClassifierMixin | None = None,
    n_trees: int = 100,
    sweep_param: str = "n_clusters",
    noise_policy: NoisePolicy = "drop",
    mode: RunMode = "default",
    random_state: int = None,
    classifier_n_jobs: int = 1,
    embeddings: ResampleEmbeddings | None = None,
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
    classifier : sklearn classifier instance, optional
        Classifier used to score generalizability.
    n_trees : int, default=100
        Number of trees in the default random-forest classifier.
    sweep_param : str, default="n_clusters"
        Name of the swept hyperparameter. Cluster-count sanity checks are
        only meaningful when this is ``"n_clusters"``.
    noise_policy : {"drop", "as_cluster", "singleton"}, default="drop"
        How to resolve negative labels emitted by density-based methods.
    mode : Literal['default', 'stability', 'generalizability'], default='default'
        Determines whether to run CARVE regularly ('default') or whether
        to only run stability analysis ('stability'),
        or generalizability analysis ('generalizability').
    random_state : int or None, default=None
        Base random seed for reproducibility.
    classifier_n_jobs : int, default=1
        Threads for this resample's classifier, its share of the run's core
        budget.
    embeddings : ResampleEmbeddings or None, default=None
        This resample's precomputed embeddings under randomized
        preprocessing. None clusters the raw subsamples. The classifier
        trains on raw features either way.

    Returns
    -------
    result : ResampleResult
        Named tuple containing ARI metrics, labels, sample indices, and
        the pipeline this resample used.

    Raises
    ------
    RuntimeError
        If ``embeddings`` does not match this resample's subsample sizes.
    """
    policy = resolve_mode(mode)

    random_state0 = random_state if random_state is not None else 0

    # --- Subsample indices ---
    P_1_idx, P_test_idx, P_2_idx = _resample_indices(
        X.shape[0],
        seed=seed,
        n_resamples=n_resamples,
        subsample_ratio=subsample_ratio,
        random_state=random_state0,
        run_stability=policy.run_stability,
    )

    # --- Features to cluster ---
    # Raw slices without randomization; with it, the resample's precomputed
    # embeddings, one independent fit per subsample. After clustering the
    # embeddings are not used again, so the noise policy's index shrinkage
    # below needs no re-slicing of them.
    if embeddings is None:
        spec = None
        X_1 = X[P_1_idx]
        X_test = X[P_test_idx] if policy.run_generalizability else None
        X_2 = X[P_2_idx] if policy.run_stability else None
    else:
        _check_embeddings(
            policy,
            embeddings,
            P_1_idx=P_1_idx,
            P_2_idx=P_2_idx,
            P_test_idx=P_test_idx,
        )
        spec = embeddings.spec
        X_1, X_2, X_test = embeddings.X_1, embeddings.X_2, embeddings.X_test

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

    # --- Resolve noise labels (density-based methods emit -1) ---
    #
    # Under "drop" policy, index arrays shrink. The classifier below reads
    # raw features by these global indices, so nothing needs re-slicing.
    P_1_idx, labels_1, noise_fraction = apply_noise_policy(
        P_1_idx, labels_1, noise_policy
    )

    if policy.run_generalizability:
        P_test_idx, labels_test, _ = apply_noise_policy(
            P_test_idx, labels_test, noise_policy
        )

    if policy.run_stability:
        P_2_idx, labels_2, _ = apply_noise_policy(P_2_idx, labels_2, noise_policy)

    # --- Cluster count bookkeeping ---
    k_1 = count_clusters(labels_1)
    k_test = count_clusters(labels_test) if policy.run_generalizability else 0
    k_2 = count_clusters(labels_2) if policy.run_stability else 0

    if sweep_param == "n_clusters":
        expected = params.get("n_clusters")

        if k_1 != expected:
            warnings.warn(f"labels_1 has {k_1} clusters, expected {expected}")

        if policy.run_generalizability and k_test != expected:
            warnings.warn(f"labels_test has {k_test} clusters, expected {expected}")

        if policy.run_stability and k_2 != expected:
            warnings.warn(f"labels_2 has {k_2} clusters, expected {expected}")

    elif k_1 < 2:
        warnings.warn(
            f"{est_class.__name__} with {params!r} produced {k_1} cluster(s) "
            "on a subsample; stability and generalizability are degenerate "
            "at this point on the sweep axis."
        )

    # --- Stability ARI (overlap between two independent subsamples) ---
    ari_stab = _compute_stability_ari(
        policy=policy,
        P_1_idx=P_1_idx,
        P_2_idx=P_2_idx,
        labels_1=labels_1,
        labels_2=labels_2,
    )

    # --- Generalizability ARI (RF prediction on held-out set) ---
    # The classifier trains on raw features, not on the embedding the labels
    # came from: a cluster generalizes only if it can be learned from the
    # data, and a transformer without transform (t-SNE) could not embed
    # unseen rows anyway. Under identity preprocessing these equal X_1, X_test.
    labels_pred, ari_pred = _compute_generalizability_ari(
        policy=policy,
        X_1=X[P_1_idx],
        X_test=X[P_test_idx] if policy.run_generalizability else None,
        labels_1=labels_1,
        labels_test=labels_test,
        classifier=classifier,
        n_trees=n_trees,
        seed=random_state0 + seed,
        classifier_n_jobs=classifier_n_jobs,
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
        pipeline=spec,
        n_clusters_train=k_1,
        n_clusters_test=k_test,
        n_clusters_stability=k_2,
        noise_fraction=noise_fraction,
    )


def _check_embeddings(
    policy,
    embeddings: ResampleEmbeddings,
    *,
    P_1_idx: np.ndarray,
    P_2_idx: np.ndarray | None,
    P_test_idx: np.ndarray,
) -> None:
    """Raise if precomputed embeddings do not match this resample's subsamples.

    The precompute pass and ``validation_iter`` derive the same indices from
    ``_resample_indices``; a row-count mismatch means that invariant broke.
    """
    expected = [("X_1", embeddings.X_1, P_1_idx)]
    if policy.run_stability:
        expected.append(("X_2", embeddings.X_2, P_2_idx))
    if policy.run_generalizability:
        expected.append(("X_test", embeddings.X_test, P_test_idx))

    for name, embedded, idx in expected:
        n_rows = None if embedded is None else embedded.shape[0]
        if n_rows != idx.size:
            raise RuntimeError(
                f"Precomputed embedding {name} has {n_rows} rows but this "
                f"resample's subsample has {idx.size}. This is an internal "
                "CARVE error."
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

    if P_1_idx.size == 0 or P_2_idx.size == 0:
        warnings.warn(
            "All points in a subsample were labelled as noise and dropped "
            "(noise_policy='drop'); stability ARI is undefined for this "
            "resample. Consider relaxing the clustering parameters or "
            "switching to noise_policy='as_cluster'."
        )
        return np.nan

    _, i_1, i_2 = np.intersect1d(P_1_idx, P_2_idx, return_indices=True)

    return adjusted_rand_score(labels_1[i_1], labels_2[i_2])


def _compute_generalizability_ari(
    policy,
    X_1,
    X_test,
    labels_1,
    labels_test,
    classifier,
    n_trees,
    seed,
    classifier_n_jobs=1,
):
    """Compute generalizability ARI via random-forest prediction.

    Parameters
    ----------
    policy : ModePolicy
        Execution policy; generalizability is skipped when
        ``policy.run_generalizability`` is False.
    X_1 : ndarray
        Raw training features, ``X[P_1]`` after the noise policy.
    X_test : ndarray or None
        Raw held-out features, ``X[P_test]`` after the noise policy.
    labels_1 : ndarray
        Cluster labels for the training set.
    labels_test : ndarray or None
        Cluster labels for the test set.
    classifier : sklearn classifier instance, optional
        Classifier used to score generalizability.
    n_trees : int, default=100
        Number of trees in the default random-forest classifier.
    seed : int
        Random seed for the random forest.
    classifier_n_jobs : int, default=1
        Threads for the classifier.

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

    if X_1.shape[0] == 0 or X_test.shape[0] == 0:
        warnings.warn(
            "All points in a subsample were labelled as noise and dropped "
            "(noise_policy='drop'); generalizability ARI is undefined for "
            "this resample. Consider relaxing the clustering parameters or "
            "switching to noise_policy='as_cluster'."
        )
        return None, np.nan

    clf = default_generalizability_classifier(
        classifier=classifier,
        n_features=X_1.shape[1],
        n_trees=n_trees,
        random_state=seed,
        n_jobs=classifier_n_jobs,
    )

    clf.fit(X_1, labels_1)

    labels_pred = clf.predict(X_test)
    ari_pred = adjusted_rand_score(labels_test, labels_pred)

    return labels_pred, ari_pred
