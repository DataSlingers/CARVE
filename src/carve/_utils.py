"""Shared utility helpers for CARVE.

Provides subsample splitting, cluster label alignment via the Hungarian
algorithm, ARI score summarization, and array coercion utilities.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from joblib import cpu_count
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.base import ClassifierMixin, ClusterMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.cluster import contingency_matrix

from ._types import NoisePolicy


def split_subsample_indices(
    n_samples: int,
    *,
    subsample_ratio: float = 0.8,
    random_state: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into a random subsample and its complement.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    subsample_ratio : float, default=0.8
        Proportion of samples in the subsample.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    train_idx : ndarray
        Indices for the subsample.
    test_idx : ndarray
        Indices for the remaining samples.
    """
    rng = np.random.RandomState(random_state)
    all_idx = np.arange(n_samples)
    train_size = int(np.float64(subsample_ratio * n_samples))

    train_idx = rng.choice(all_idx, size=train_size, replace=False)
    test_idx = np.setdiff1d(all_idx, train_idx)

    return train_idx, test_idx


def count_clusters(labels: np.ndarray | None) -> int:
    """Count distinct non-noise clusters in a label vector.

    Density-based methods such as HDBSCAN mark unassigned points with a
    negative label; those are not counted as a cluster.

    Parameters
    ----------
    labels : ndarray or None
        Cluster labels.

    Returns
    -------
    n_clusters : int
        Number of distinct non-negative labels, or 0 for empty input.
    """
    if labels is None:
        return 0

    labels = np.asarray(labels)
    if labels.size == 0:
        return 0

    return int(np.unique(labels[labels >= 0]).size)


def apply_noise_policy(
    indices: np.ndarray,
    labels: np.ndarray,
    policy: NoisePolicy = "drop",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resolve negative (noise) labels according to policy.

    Density-based methods such as HDBSCAN mark unassigned points with
    ``-1``. Left unhandled, those samples are treated as a cluster by
    the consensus matrix and by ARI, which inflates stability.

    Parameters
    ----------
    indices : ndarray of shape (n_subsample,)
        Row indices of these points in the original dataset.
    labels : ndarray of shape (n_subsample,)
        Cluster labels, possibly containing negative noise markers.
    policy : {"drop", "as_cluster", "singleton"}, default="drop"
        ``"drop"`` removes noise points from this iteration entirely, so
        they count as un-sampled and contribute nothing to the consensus
        matrix, the ARI, or the classifier. ``"as_cluster"`` keeps ``-1``
        as an ordinary label. ``"singleton"`` gives each noise point its
        own cluster.

    Returns
    -------
    indices : ndarray
        Possibly filtered indices, aligned with labels.
    labels : ndarray
        Labels with noise resolved.
    noise_fraction : float
        Fraction of the input points that carried a negative label.

    Raises
    ------
    ValueError
        If policy is not a recognized noise policy.
    """
    if policy not in ("drop", "as_cluster", "singleton"):
        raise ValueError(
            f"Unknown noise_policy {policy!r}. Expected 'drop', 'as_cluster', "
            "or 'singleton'."
        )

    labels = np.asarray(labels)
    indices = np.asarray(indices)

    if labels.size == 0:
        return indices, labels, 0.0

    noise_mask = labels < 0
    noise_fraction = float(noise_mask.mean())

    if not noise_mask.any() or policy == "as_cluster":
        return indices, labels, noise_fraction

    if policy == "drop":
        keep = ~noise_mask
        return indices[keep], labels[keep], noise_fraction

    # policy == "singleton"
    out = labels.copy()
    assigned = labels[~noise_mask]
    start = int(assigned.max()) + 1 if assigned.size else 0
    out[noise_mask] = np.arange(start, start + int(noise_mask.sum()))
    return indices, out, noise_fraction


def _coerce_n_clusters(value: int | np.ndarray) -> np.ndarray:
    """Coerce a cluster-count specification to a 1D integer NumPy array.

    Parameters
    ----------
    value : int or ndarray
        Cluster-count input. If an integer K is provided, this function
        returns the range 2..K (inclusive). If an array-like is provided,
        it is validated and converted to an integer ndarray.

    Returns
    -------
    n_clusters : ndarray of shape (n_values,)
        One-dimensional integer array of cluster counts, where all values
        are greater than or equal to 2.

    Raises
    ------
    ValueError
        If an integer input is smaller than 2, if the provided array is not
        one-dimensional, or if any cluster count is smaller than 2.
    TypeError
        If the provided array does not contain integer values.
    """
    if isinstance(value, (int, np.integer)):
        k = int(value)

        if k < 2:
            raise ValueError("n_clusters int must be >= 2.")

        return np.arange(2, k + 1, dtype=int)

    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError("n_clusters must be a 1D array.")

    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("n_clusters array must contain integers.")

    if np.any(arr < 2):
        raise ValueError("All n_clusters values must be >= 2.")

    return arr.astype(int, copy=False)


def _summarize_ari_scores(
    x: list[float] | np.ndarray,
    n_resamples: int,
) -> tuple[float, float, float, float]:
    """Summarize ARI scores by computing mean, standard error, and quantiles.

    Parameters
    ----------
    x : list of float or ndarray
        ARI scores, possibly containing NaN values.
    n_resamples : int
        Number of resamples (unused in computation; kept for interface
        consistency).

    Returns
    -------
    mean : float
        Mean of the ARI scores, ignoring NaN values.
    se : float
        Standard error, ignoring NaN values. Returns NaN if fewer than 2
        valid entries.
    q95 : float
        95th percentile of the ARI scores, ignoring NaN values.
    q05 : float
        5th percentile of the ARI scores, ignoring NaN values.
    """
    arr = np.asarray(x, dtype=float)
    if np.all(np.isnan(arr)):
        return (np.nan, np.nan, np.nan, np.nan)
    mean = float(np.nanmean(arr))

    m = np.sum(~np.isnan(arr))
    se = float(np.nanstd(arr, ddof=1) / np.sqrt(m)) if m > 1 else np.nan
    q95 = float(np.nanquantile(arr, 0.95))
    q05 = float(np.nanquantile(arr, 0.05))
    return (mean, se, q95, q05)


def cluster_labels(
    X: np.ndarray,
    estimator_cls: type[ClusterMixin],
    *,
    random_state: int = None,
    **params: Any,
) -> np.ndarray:
    """Fit a clustering estimator and return labels.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    estimator_cls : type
        Estimator class implementing clustering.
    random_state : int or None, default=None
        Random seed passed to the estimator if supported.
    **params : dict
        Estimator parameters.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels.

    Raises
    ------
    TypeError
        If the estimator provides no valid way to extract labels.
    """
    try:
        estimator = estimator_cls(random_state=random_state, **params)
    except Exception:
        if "random_state" in params:
            estimator = estimator_cls(**params)
        elif "random_state" in estimator_cls.__init__.__code__.co_varnames:
            estimator = estimator_cls(random_state=random_state, **params)
        else:
            estimator = estimator_cls(**params)

    if hasattr(estimator, "fit_predict"):
        return estimator.fit_predict(X)

    estimator.fit(X)

    try:
        return estimator.labels_
    except Exception:
        return estimator.predict(X)


def summarize_preprocessing_records(
    pipeline_records: list[dict[str, Any]],
    sweep_param: str = "n_clusters",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize a randomized run per configuration, pipeline and sweep value.

    Parameters
    ----------
    pipeline_records : list of dict
        One record per configuration, as ``_runner.run_validation`` returns
        them: ``method_id``, ``method_label``, ``sweep_value``,
        ``sweep_rank`` and ``results``, the configuration's per-resample
        results, each carrying the ``pipeline`` it used.
    sweep_param : str, default="n_clusters"
        Name of the swept hyperparameter, used to name the sweep column.

    Returns
    -------
    summary : pandas.DataFrame
        One row per (configuration, pipeline, sweep value), sorted by
        ``method_id``, ``pipeline`` and sweep value. Means and standard
        errors are over the resamples that used the row's pipeline at the
        row's configuration only, and ``n_resamples`` counts them.
    pipelines : dict
        The pipeline specs, keyed by the ``pipeline`` column.
    """
    columns = [
        "method_id",
        "method_label",
        "pipeline",
        "normalization",
        "dim_reduction",
        sweep_param,
        "n_resamples",
        "ari_stability",
        "ari_stability_se",
        "ari_generalizability",
        "ari_generalizability_se",
        "n_clusters_observed",
        "sweep_param",
        "sweep_value",
        "sweep_rank",
    ]
    rows: list[dict[str, Any]] = []
    pipelines: dict[str, Any] = {}

    for record in pipeline_records:
        # Group the configuration's resamples by pipeline. The spec is
        # duck-typed on .label: _utils is a leaf and cannot import _pipeline.
        by_pipeline: dict[str, list[Any]] = {}
        for result in record["results"]:
            label = result.pipeline.label
            pipelines.setdefault(label, result.pipeline)
            by_pipeline.setdefault(label, []).append(result)

        for label, runs in by_pipeline.items():
            spec = pipelines[label]
            stab_mean, stab_se, _, _ = _summarize_ari_scores(
                [r.ari_stability for r in runs], len(runs)
            )
            gen_mean, gen_se, _, _ = _summarize_ari_scores(
                [r.ari_generalizability for r in runs], len(runs)
            )
            rows.append(
                {
                    "method_id": record["method_id"],
                    "method_label": record["method_label"],
                    "pipeline": label,
                    "normalization": spec.normalization.label,
                    "dim_reduction": spec.dim_reduction.label,
                    sweep_param: record["sweep_value"],
                    "n_resamples": len(runs),
                    "ari_stability": stab_mean,
                    "ari_stability_se": stab_se,
                    "ari_generalizability": gen_mean,
                    "ari_generalizability_se": gen_se,
                    "n_clusters_observed": float(
                        np.mean([r.n_clusters_train for r in runs])
                    ),
                    "sweep_param": sweep_param,
                    "sweep_value": record["sweep_value"],
                    "sweep_rank": record["sweep_rank"],
                }
            )

    summary = pd.DataFrame(rows, columns=columns)
    if summary.empty:
        return summary, pipelines

    # method_id is "m<n>"; sort on n so that m10 follows m9.
    summary = summary.sort_values(
        ["method_id", "pipeline", "sweep_value"],
        key=lambda col: col.str[1:].astype(int) if col.name == "method_id" else col,
        kind="stable",
    ).reset_index(drop=True)
    return summary, pipelines


def align_cluster_labels(
    reference_labels: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Align labels to reference labels using Hungarian assignment.

    Parameters
    ----------
    reference_labels : ndarray of shape (n_samples,)
        Reference clustering labels.
    labels : ndarray of shape (n_samples,)
        Labels to align.

    Returns
    -------
    aligned : ndarray of shape (n_samples,)
        Aligned labels with best matching permutation. Clusters the
        assignment cannot match, when there are more clusters than reference
        labels, take ids past the largest reference label.
    """
    cont = contingency_matrix(reference_labels, labels)
    row_ind, col_ind = linear_sum_assignment(-cont)

    true_classes = np.unique(reference_labels)
    pred_classes = np.unique(labels)

    mapping = {
        pred_classes[col]: true_classes[row] for row, col in zip(row_ind, col_ind)
    }
    # An unmatched cluster that kept its own id could share the id a matched
    # cluster was given, merging the two, so each takes a fresh one instead.
    next_id = int(np.max(true_classes)) + 1
    for pc in pred_classes:
        if pc not in mapping:
            mapping[pc] = next_id
            next_id += 1

    aligned = np.array([mapping[lbl] for lbl in labels], dtype=reference_labels.dtype)
    return aligned


def ensure_2d_array(
    X: ArrayLike, *, dense_warn_elements: int = 50_000_000
) -> np.ndarray:
    """Ensure input is a 2D NumPy array.

    Sparse input is densified rather than passed through. Several estimators
    on the default grid (agglomerative clustering, t-SNE, and the RBF affinity
    of :class:`~carve.SpectralClustering`) require dense input, so keeping the
    matrix sparse would only move the failure into a parallel worker.

    Parameters
    ----------
    X : array-like
        Input data. Accepts NumPy arrays, pandas DataFrames, lists, and
        SciPy sparse matrices and arrays.
    dense_warn_elements : int, default=50_000_000
        Warn when densifying a sparse matrix with more entries than this
        (50M float64 entries is roughly 400 MB).

    Returns
    -------
    array : ndarray of shape (n_samples, n_features)
        2D NumPy array representation.

    Raises
    ------
    ValueError
        If the input cannot be converted to a 2D array.
    """
    if isinstance(X, pd.DataFrame):
        return X.values
    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            return X.reshape(-1, 1)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError("Input NumPy array must be 1D or 2D.")
    elif sparse.issparse(X):
        if X.ndim != 2:
            raise ValueError("Input sparse matrix must be 2D.")
        n_elements = int(X.shape[0]) * int(X.shape[1])
        if n_elements > dense_warn_elements:
            warnings.warn(
                f"Densifying a sparse matrix with {n_elements:,} entries "
                f"(~{n_elements * 8 / 1e9:.1f} GB dense). Pass a reduced "
                "representation instead, e.g. use_rep='X_pca'.",
                UserWarning,
                stacklevel=2,
            )
        return np.asarray(X.todense())
    elif isinstance(X, list):
        return np.atleast_2d(np.array(X))
    else:
        raise ValueError(
            "Input must be a NumPy array, SciPy sparse matrix, Pandas "
            "DataFrame, or a list."
        )


def resolve_core_budget(n_jobs: int | None, *, n_resamples: int) -> tuple[int, int]:
    """Split ``n_jobs`` into resample workers and threads per worker.

    ``n_jobs`` follows joblib's convention: a positive count, ``-1`` for
    every core, ``-2`` for all but one, and None for one worker. The worker
    count is capped at ``n_resamples`` because Parallel never runs more
    workers than tasks; the threads those workers would have held go to the
    ones that run. Each worker's classifier gets the cores left over,
    ``cpu_count // workers``, never fewer than one.

    ``n_jobs=1`` therefore keeps the historical default: one worker with the
    forest on every core. ``n_jobs=-1`` is the other end, one resample per
    core with single-threaded forests. In between, the product of the two
    never exceeds the machine when the workers fit on it.

    Parameters
    ----------
    n_jobs : int or None
        The core budget passed to ``CARVE``.
    n_resamples : int
        Resamples per configuration, the number of tasks the workers share.

    Returns
    -------
    outer : int
        Worker count to give ``joblib.Parallel`` over resamples.
    inner : int
        Thread count to give each resample's classifier.
    """
    n_cores = cpu_count()
    if n_jobs is None:
        outer = 1
    elif n_jobs == 0:
        raise ValueError("n_jobs == 0 has no meaning; use 1 or a negative count")
    elif n_jobs < 0:
        outer = max(n_cores + 1 + n_jobs, 1)
    else:
        outer = n_jobs
    outer = max(1, min(outer, n_resamples))
    inner = max(1, n_cores // outer)
    return outer, inner


def default_generalizability_classifier(
    *,
    classifier: ClassifierMixin | None,
    n_features: int,
    n_trees: int,
    random_state: int | None,
    n_jobs: int,
) -> ClassifierMixin:
    """Build the classifier CARVE uses to predict labels for unseen samples.

    Single source for that construction. It is used twice: once per resample
    on the generalizability path, and once by ``CARVE.get_labels`` to extend
    an anchored cut from the anchors to every sample. The two must build the
    same classifier, and a comment saying so is not enforceable, so they call
    this instead.

    Parameters
    ----------
    classifier : sklearn classifier instance or None
        User-supplied classifier. None selects the default random forest.
    n_features : int
        Number of columns in the data the classifier will be fitted on. Caps
        the default forest's depth.
    n_trees : int
        Number of trees in the default forest. Ignored when ``classifier``
        is given.
    random_state : int or None
        Seed. Applied to a user-supplied classifier only when it accepts a
        ``random_state`` parameter.
    n_jobs : int
        Threads the classifier may use, its share of the run's core budget
        from ``resolve_core_budget``. Applied to a user-supplied classifier
        only when it accepts an ``n_jobs`` parameter, overriding whatever it
        was built with.

    Returns
    -------
    classifier : sklearn classifier instance
        A fresh, unfitted classifier. A user-supplied one is cloned, so no
        state leaks between calls.
    """
    if classifier is None:
        return RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=n_features,
            max_features=int(np.sqrt(n_features)),
            random_state=random_state,
            n_jobs=n_jobs,
        )

    clf = clone(classifier)
    params = clf.get_params()
    if "random_state" in params:
        clf.set_params(random_state=random_state)
    if "n_jobs" in params:
        clf.set_params(n_jobs=n_jobs)
    return clf


def resolve_anchors(
    n_samples: int,
    *,
    consensus_anchors: int | float | None,
    anchor_threshold: int,
    random_state: int | None,
) -> np.ndarray | None:
    """Resolve the consensus anchor index for a run.

    Returns None when the exact path applies, meaning the consensus matrix is
    built over every sample as it always has been. Otherwise returns a sorted
    array of anchor indices.

    The default resolves to ``min(n_samples, anchor_threshold)``. A flat
    default would make the effective anchor count fall discontinuously as
    n crosses the threshold -- 5,000 anchors at n=5,000 and 2,000 at
    n=5,001 -- which is an artificial jump in estimator variance at exactly
    the boundary users cross.

    Parameters
    ----------
    n_samples : int
        Number of samples in the run.
    consensus_anchors : int, float, or None
        Anchor count as an integer, a fraction of ``n_samples`` as a float in
        (0, 1], or None for the default.
    anchor_threshold : int
        Runs with ``n_samples <= anchor_threshold`` take the exact path. The
        comparison is inclusive so a dataset sized exactly at the threshold
        keeps the exact path.
    random_state : int or None
        Seed for the anchor draw. Drawn from a local Generator, so no global
        RNG state is touched and joblib workers stay reproducible.

    Returns
    -------
    anchors : ndarray of shape (m,) or None
    """
    n_samples = int(n_samples)

    if consensus_anchors is None:
        m = min(n_samples, int(anchor_threshold))
    elif isinstance(consensus_anchors, float):
        if not 0.0 < consensus_anchors <= 1.0:
            raise ValueError(
                "consensus_anchors given as a fraction must be in (0, 1], got "
                f"{consensus_anchors}."
            )
        m = int(round(consensus_anchors * n_samples))
    else:
        m = int(consensus_anchors)

    # The exact-path return comes first: at n_samples < 2 the default resolves
    # to m = n_samples, and a run that small is simply exact, not misconfigured.
    if m >= n_samples:
        return None

    if m < 2:
        raise ValueError(
            f"The consensus anchor count must be at least 2, got {m}. It comes "
            f"from consensus_anchors={consensus_anchors!r} when that is set, "
            f"and otherwise from min(n_samples, anchor_threshold="
            f"{anchor_threshold})."
        )

    rng = np.random.default_rng(0 if random_state is None else int(random_state))
    return np.sort(rng.choice(n_samples, size=m, replace=False)).astype(np.int64)
