"""Case-study compute: the CVI sweep and the CARVE fit cache.

No matplotlib here. The routine this replaces swept the classical indices and
then built a figure and called plt.show() in the same call, so the sweep
could not be reused without also drawing it.
"""

import hashlib
import warnings
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import ClusterMixin
from sklearn.metrics import adjusted_rand_score

from carve import CARVE

from ._cvi import calculate_cvi, select_k
from ._estimators import (
    DENSE_PAIRWISE_ESTIMATORS,
    ESTIMATOR_CLASSES,
    RESOLUTION_ESTIMATORS,
    apply_random_state,
    param_grids,
    resolution_grids,
)
from ._types import EstimatorSpec, Study

CVI_SWEEP_METRICS: tuple[str, ...] = (
    "silhouette",
    "gap",
    "davies_bouldin",
    "calinski_harabasz",
)

# Above this many samples, a dense n-by-n float64 matrix is no longer a
# reasonable thing to build without asking first: at 5,000 it is 0.2 GB; at
# 50,000 it is 20 GB. 5,000 also matches CARVE's own anchor_threshold
# default, which draws the same "an exact n-by-n matrix stops being
# tractable here" line for the consensus matrix.
DENSE_ESTIMATOR_SAFE_N = 5_000

_DENSE_ESTIMATOR_CLASSES: frozenset[type[ClusterMixin]] = frozenset(
    cls for name, cls in ESTIMATOR_CLASSES.items() if name in DENSE_PAIRWISE_ESTIMATORS
)


def _check_dense_fit(
    n_samples: int,
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
) -> None:
    """Refuse to silently build an O(n^2) affinity or distance matrix.

    study_model_grids is scale-blind by design -- it knows an estimator and
    candidate_k, never n -- so a check placed there could not see this
    coming, and a caller who assembles model_grids by hand instead of going
    through study_model_grids would walk straight past it anyway. cvi_sweep,
    fit_or_load_carve, and study_scaling_sweep are the one place every
    case-study entry point brings X and model_grids together before handing
    them to a real estimator, which is why the check lives here instead.
    """
    offending = sorted(
        {cls.__name__ for cls, _ in model_grids if cls in _DENSE_ESTIMATOR_CLASSES}
    )
    if offending and n_samples > DENSE_ESTIMATOR_SAFE_N:
        gb = 8 * n_samples**2 / 1e9
        raise ValueError(
            f"model_grids includes {', '.join(offending)} at n_samples="
            f"{n_samples}, which would build a dense {n_samples}-by-"
            f"{n_samples} matrix (about {gb:.1f} GB). That is only safe up "
            f"to about {DENSE_ESTIMATOR_SAFE_N} samples here; use a smaller "
            "scale, or restrict model_grids to estimators that scale to "
            "this n (Leiden via study_resolution_grids, for example)."
        )


def _model_label(estimator_cls: type[ClusterMixin], params: dict[str, Any]) -> str:
    """A short readable name for one estimator configuration."""
    if not params:
        return estimator_cls.__name__
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{estimator_cls.__name__} ({rendered})"


def _spec_for(
    estimator_cls: type[ClusterMixin], fixed_params: dict[str, Any]
) -> EstimatorSpec:
    """Map an estimator class and its fixed parameters back to a spec.

    Used by the gap statistic to refit reference datasets. One class can be
    registered under more than one name (AgglomerativeClustering as
    "agglomerative" and "agglomerative_single"), so the fixed parameters
    are part of the match; matching the class alone would refit every
    single-linkage cell's references with Ward.
    """
    from ._estimators import ESTIMATOR_CLASSES, ESTIMATOR_DEFAULTS

    for name, cls in ESTIMATOR_CLASSES.items():
        if cls is estimator_cls and ESTIMATOR_DEFAULTS[name] == fixed_params:
            return EstimatorSpec(name=name)
    raise ValueError(
        f"No EstimatorSpec is registered for {estimator_cls.__name__} with "
        f"parameters {fixed_params!r}."
    )


def _sweep_cell(
    X: np.ndarray,
    y: np.ndarray | None,
    estimator_cls: type[ClusterMixin],
    fixed_params: dict[str, Any],
    k: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Fit one estimator at one k and score every classical index."""
    params = dict(fixed_params)
    params["n_clusters"] = int(k)
    apply_random_state(estimator_cls, params, random_state)

    labels = np.asarray(estimator_cls(**params).fit_predict(X), dtype=np.int32)
    ari = float(adjusted_rand_score(y, labels)) if y is not None else float("nan")
    spec = _spec_for(estimator_cls, fixed_params)

    rows = []
    for metric in CVI_SWEEP_METRICS:
        score, error = calculate_cvi(
            X, labels, metric, spec=spec, random_state=random_state
        )
        rows.append(
            {
                "metric": metric,
                "model": _model_label(estimator_cls, fixed_params),
                "k": int(k),
                "score": float(score),
                "se": float(error),
                "ari": ari,
            }
        )
    return rows


def cvi_sweep(
    X: np.ndarray,
    y: np.ndarray | None,
    *,
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
    candidate_k: tuple[int, ...],
    random_state: int = 0,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep every classical index over every (model, k) combination.

    Returns
    -------
    curves_df : one row per (metric, model, k) with columns
        (metric, model, k, score, ari)
    best_df : one row per metric, the selected (model, k). The gap statistic
        selects by Tibshirani's rule; the others by argmax. Selection goes
        through _cvi.select_k so the two agree with the simulated benchmarks.
    """
    X = np.asarray(X)
    y_arr = None if y is None else np.asarray(y)
    _check_dense_fit(X.shape[0], model_grids)

    jobs = []
    for estimator_cls, grid in model_grids:
        other_keys = [key for key in grid if key != "n_clusters"]
        other_values = [
            grid[key]
            if isinstance(grid[key], list | tuple | np.ndarray)
            else [grid[key]]
            for key in other_keys
        ]
        combos = list(product(*other_values)) if other_keys else [()]

        for combo in combos:
            fixed = dict(zip(other_keys, combo)) if other_keys else {}
            for k in candidate_k:
                jobs.append(
                    delayed(_sweep_cell)(
                        X, y_arr, estimator_cls, fixed, k, random_state
                    )
                )

    raw = Parallel(n_jobs=n_jobs)(jobs)
    curves = pd.DataFrame([row for batch in raw for row in batch])
    curves = curves.sort_values(["metric", "model", "k"]).reset_index(drop=True)
    # se is an internal selection input, not part of the returned contract.
    returned_curves = curves.drop(columns=["se"])

    best_rows = []
    for metric in CVI_SWEEP_METRICS:
        sub = curves[curves["metric"] == metric]
        candidates = []
        for model, by_model in sub.groupby("model", sort=False):
            by_model = by_model.sort_values("k")
            ks = by_model["k"].tolist()
            scores = by_model["score"].tolist()
            # se is the gap statistic's s_k, carried out of the sweep rather
            # than recomputed, and 0.0 for every other index.
            errors = by_model["se"].tolist()
            chosen = select_k(metric, ks, scores, errors)
            row = by_model[by_model["k"] == chosen].iloc[0]
            candidates.append(row)

        winner = max(candidates, key=lambda r: r["score"])
        best_rows.append(winner.to_dict())

    best = pd.DataFrame(best_rows)[["metric", "model", "k", "score", "ari"]]
    return returned_curves, best.reset_index(drop=True)


def _fingerprint(X: np.ndarray) -> str:
    return hashlib.sha1(
        np.ascontiguousarray(np.asarray(X, dtype=np.float64)).tobytes()
    ).hexdigest()


def _fingerprint_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".x-sha1")


def _check_fingerprint(cache_path: Path, X: np.ndarray) -> None:
    """Refuse to serve a cached fit against a different X.

    A cached fit is only valid for the matrix it was fit on, and
    fit_or_load_carve restores X_ onto whatever it loads. The hECA
    development embedding changes under one scale name (subsample-first
    until the pooled cache exists, pooled after), which is exactly the case
    a scale-keyed filename cannot catch. A cache written before this check
    existed has no record to compare against; it is served with a warning
    rather than discarded, since a fit can be hours of compute.
    """
    sidecar = _fingerprint_path(cache_path)
    if not sidecar.is_file():
        warnings.warn(
            f"{cache_path} carries no fingerprint of the X it was fit on, so "
            "it cannot be checked against the X passed now. Pass force=True "
            "to refit if the data has changed since it was cached.",
            stacklevel=3,
        )
        return
    if sidecar.read_text().strip() != _fingerprint(X):
        raise ValueError(
            f"{cache_path} was fit on a different X than the one passed now "
            "(the fingerprint differs). Serving it would report results for "
            "data it never saw. Pass force=True to refit on this X, or load "
            "the data the cache was fit on."
        )


def fit_or_load_carve(
    X: np.ndarray,
    y: np.ndarray | pd.Series | None,
    *,
    cache_path: Path,
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
    n_resamples: int = 100,
    n_jobs: int = 1,
    random_state: int = 42,
    force: bool = False,
    consensus_anchors: int | None = None,
) -> CARVE:
    """Fit CARVE on a case study, caching the fitted state to disk.

    A Levine fit takes hours, so the cache is what makes regenerating a figure
    practical. The saved state does not carry the data matrix, so X_ is
    restored after loading, matching the notebook this replaces.

    consensus_anchors : int or None, default=None
        Forwarded to CARVE only when not None, so studies that leave it at
        the package default (an exact, unanchored run) are unaffected. hECA
        sets this at case-study scale, where an exact consensus matrix does
        not fit.
    """
    cache_path = Path(cache_path)
    _check_dense_fit(np.asarray(X).shape[0], model_grids)

    if cache_path.is_file() and not force:
        _check_fingerprint(cache_path, X)
        carve = CARVE.load(str(cache_path))
        carve.X_ = np.asarray(X)
        return carve

    carve = CARVE(
        estimator_param_grids=model_grids,
        n_resamples=n_resamples,
        n_jobs=n_jobs,
        random_state=random_state,
        **(
            {}
            if consensus_anchors is None
            else {"consensus_anchors": consensus_anchors}
        ),
    )
    reference = None if y is None else np.asarray(y)
    carve.fit(np.asarray(X), reference_labels=reference)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    carve.save(str(cache_path))
    _fingerprint_path(cache_path).write_text(_fingerprint(X))
    return carve


def _klein_loader(subsample: int | float | None):
    from .datasets import load_klein

    # The manuscript's 1,358 cells is a 0.5 subsample of the 2,717-cell
    # preprocessed set (Klein sample size, manuscript line 606).
    return load_klein(subsample=subsample, random_state=42)


def _levine_loader(subsample: int | float | None):
    from .datasets import load_levine32

    # Manuscript line 627: a stratified subsample of 5,000 cells.
    return load_levine32(subsample=subsample, random_state=42)


def _cusanovich_loader(subsample: int | float | None):
    from .datasets import load_cusanovich

    # tissue is the reference: it is determined by dissection, not by any
    # clustering, which is what makes it independent ground truth.
    return load_cusanovich(subsample=subsample, random_state=42, label_column="tissue")


def _heca_loader(subsample: int | float | None):
    from .datasets import load_heca

    # subsample_before_embedding: the pooled embedding of the five organs
    # is tens of GB to compute, so a development-scale load draws its rows
    # from the organ files first. load_heca ignores the flag at publication
    # scale (subsample=None) and whenever the pooled cache is present.
    return load_heca(
        subsample=subsample,
        random_state=42,
        label_column="organ",
        subsample_before_embedding=True,
    )


def _scale_name(study: Study, scale: str | None) -> str:
    """Resolve a scale argument to the scale name it refers to.

    None means the study's own default. This is the one place the
    dev/publication default is chosen; resolve_scale, load_study, and
    carve_cache_path all go through it so the rule cannot drift between them.
    """
    return study.default_scale if scale is None else scale


def resolve_scale(study: Study, scale: str | None) -> int | float | None:
    """Resolve a scale name to the subsample size the loader receives."""
    name = _scale_name(study, scale)
    if name not in study.scales:
        raise ValueError(
            f"Study {study.name!r} has no scale {name!r}. "
            f"Declared scales are {sorted(study.scales)}."
        )
    return study.scales[name]


def load_study(
    study: Study, *, scale: str | None = None
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a study's data at a named scale, recording the scale in meta."""
    subsample = resolve_scale(study, scale)
    name = _scale_name(study, scale)
    X, y, meta = study.loader(subsample)
    meta = dict(meta)
    meta["scale"] = name
    meta["study"] = study.name
    return X, y, meta


def carve_cache_path(study: Study, *, scale: str | None = None, root: Path) -> Path:
    """Where a study's fitted CARVE state is cached, per scale.

    Both the scale name and a short hash of its resolved size are part of
    the filename deliberately. fit_or_load_carve loads whatever file sits at
    the path it is handed, so a shared path would let a publication run
    silently reuse a development-scale fit -- that is what the scale name
    guards against. The hash guards the same failure one level up: editing
    STUDIES[...].scales[name] (say, hECA's "dev" from 25,000 to 50,000)
    changes what that scale resolves to without changing its name, and
    without the hash fit_or_load_carve would silently load the fit taken at
    the old size. A hash reads better than the raw resolved size for the
    None case (full data), which has no natural filename spelling.
    """
    resolved = resolve_scale(study, scale)  # raises if the scale is unknown
    name = _scale_name(study, scale)
    size_key = hashlib.sha1(repr(resolved).encode()).hexdigest()[:8]
    return Path(root) / f"carve_{study.name}_{name}_{size_key}.carve"


STUDIES: dict[str, Study] = {
    "klein": Study(
        name="klein",
        loader=_klein_loader,
        estimator=EstimatorSpec(name="agglomerative"),
        candidate_k=tuple(range(2, 11)),
        scales={"dev": 400, "publication": 0.5},
        default_scale="publication",
        partners=(EstimatorSpec(name="spectral"),),
    ),
    "levine32": Study(
        name="levine32",
        loader=_levine_loader,
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=tuple(range(7, 18)),
        scales={"dev": 800, "publication": 5000},
        default_scale="publication",
        partners=(EstimatorSpec(name="spectral"),),
    ),
    "cusanovich": Study(
        name="cusanovich",
        loader=_cusanovich_loader,
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=tuple(range(4, 17)),
        scales={"dev": 1500, "publication": 5000, "atlas": None},
        default_scale="dev",
        # Spectral as in the other case studies, plus Ward and single-linkage
        # agglomerative clustering. Single linkage is included deliberately:
        # on an LSI it tends to peel off outliers one at a time, a partition
        # that is near-identical across resamples and so scores as highly
        # stable while saying little, which is worth showing.
        partners=(
            EstimatorSpec(name="spectral"),
            EstimatorSpec(name="agglomerative"),
            EstimatorSpec(name="agglomerative_single"),
        ),
        # The atlas scale runs every annotated cell (the loader always drops
        # cell_label=="Unknown", about 12 percent of the atlas), where
        # spectral and Ward cannot run, so that pass sweeps Leiden resolution
        # instead. This is what shows the case-study conclusion survives past
        # the subsample. 81,173 is the atlas as published; the annotated
        # subset actually analyzed at this scale is smaller (see
        # datasets._cusanovich and meta["n_cells_annotated"]).
        resolutions=tuple(round(0.1 * i, 1) for i in range(1, 21)),
        # Pinned rather than left at the package default. Both
        # consensus_matrices_ and consensus_generalizability_matrices_ are
        # retained per configuration, so retained memory is
        # n_configs * 2 * m**2 * 8 bytes. At atlas scale (tens of thousands
        # of cells, above anchor_threshold=5000) the 20-config Leiden
        # resolution sweep would otherwise anchor to the package default of
        # m=5000: 20 * 2 * 5000**2 * 8 B = 8.0 GB retained. Pinned to the
        # same 2000 anchors hECA uses (below), that sweep instead retains
        # 20 * 2 * 2000**2 * 8 B = 1.28 GB. At dev scale (1,500 cells) this
        # is a no-op: m=2000 exceeds n, so resolve_anchors takes the exact,
        # unanchored path regardless. At publication scale (5,000 cells) it
        # newly anchors to 2000 where the run was previously exact, which
        # is a deliberate, harmless trade at that size (a 5000x5000 exact
        # matrix is only 0.2 GB) made for one consistent anchor count across
        # every scale this study declares.
        consensus_anchors=2000,
    ),
    "heca": Study(
        name="heca",
        loader=_heca_loader,
        estimator=EstimatorSpec(name="minibatch_kmeans"),
        candidate_k=tuple(range(4, 16)),
        scales={"dev": 25_000, "publication": None},
        default_scale="dev",
        # Spectral builds a dense n-by-n affinity and cannot run at this
        # scale, so the partner is KMeans.
        partners=(EstimatorSpec(name="kmeans"),),
        resolutions=tuple(round(0.1 * i, 1) for i in range(1, 21)),
        # See the cusanovich entry above for the arithmetic: both
        # consensus_matrices_ and consensus_generalizability_matrices_ are
        # retained per configuration, so the 20-config Leiden resolution
        # sweep retains 20 * 2 * 2000**2 * 8 B = 1.28 GB here, against 8.0 GB
        # at the package default (anchor_threshold=5000) at hECA's own
        # scale, which is always above that threshold.
        consensus_anchors=2000,
    ),
}


def study_model_grids(
    study: Study,
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """The study's own estimator plus its declared partners, over candidate_k.

    Klein sweeps Ward agglomerative and spectral; Levine sweeps KMeans and
    spectral; Cusanovich sweeps KMeans, spectral, Ward and single linkage at
    case-study scale; hECA pairs MiniBatchKMeans with KMeans. Each study
    declares this on Study.partners.
    """
    if study.estimator.name in RESOLUTION_ESTIMATORS:
        raise ValueError(
            f"Study {study.name!r} has a resolution-based estimator; use "
            "study_resolution_grids."
        )

    grids = param_grids(study.estimator, study.candidate_k)
    for partner in study.partners:
        grids += param_grids(partner, study.candidate_k)
    return grids


def study_resolution_grids(
    study: Study,
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """The study's Leiden resolution sweep.

    A separate CARVE run from study_model_grids: SweepSpec is frozen, so a
    k-based and a resolution-based sweep cannot share one run.
    """
    if not study.resolutions:
        raise ValueError(f"Study {study.name!r} declares no resolutions.")
    return resolution_grids(EstimatorSpec(name="leiden"), study.resolutions)


def study_scaling_sweep(
    X: np.ndarray,
    y: np.ndarray | pd.Series | None,
    *,
    sizes: Sequence[int],
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
    n_resamples: int = 100,
    n_jobs: int = 1,
    random_state: int = 42,
    consensus_anchors: int | None = None,
    measure: str = "stability",
    rule: str = "1se",
) -> pd.DataFrame:
    """Fit CARVE at a ladder of subsample sizes and measure the cost.

    Every size subsamples from the same (X, y), so the biology (which
    clusters exist, their proportions and separation) is held constant and n
    is the only thing varying. This is what a scaling sweep needs and a
    per-dataset comparison cannot give: a runtime or memory difference
    between datasets of different sizes could always be attributed to the
    datasets differing in more than size.

    peak_rss_bytes is the process high-water mark at the end of each fit, not
    a per-fit delta: ru_maxrss only ever rises. Because the ladder grows
    monotonically in n, the reported value is still the peak attributable to
    that size, but it must not be read as the memory a single fit would need
    in a fresh process. Callers should pass sizes in increasing order.

    A rung larger than the data actually available is skipped, with a
    warning naming it, rather than raised as an error. hECA's publication
    ladder tops out at a literal 500_000, but the count that matters is the
    pooled total after the Unclassified drop, which is not independently
    verified anywhere; failing on an oversized top rung after already
    running every smaller one would waste hours of prior compute for no
    reason better than a hardcoded target the data may not actually reach.

    Returns
    -------
    DataFrame with columns (n, n_configs, wall_clock_s, peak_rss_bytes,
    selected_k, ari). ari is against y, or NaN when y is None. A row is
    omitted for any size skipped as too large.
    """
    import time
    import warnings

    from sklearn.metrics import adjusted_rand_score

    from ._artifacts import peak_rss_bytes

    X = np.asarray(X)
    y_arr = None if y is None else np.asarray(y)
    n_total = X.shape[0]

    rows: list[dict[str, Any]] = []
    for size in sizes:
        size = int(size)
        if size > n_total:
            warnings.warn(
                f"Skipping scaling-sweep size {size}: it exceeds the "
                f"{n_total} available samples.",
                stacklevel=2,
            )
            continue
        _check_dense_fit(size, model_grids)

        rng = np.random.default_rng(random_state + size)
        idx = np.sort(rng.choice(n_total, size=size, replace=False))

        carve = CARVE(
            estimator_param_grids=model_grids,
            n_resamples=n_resamples,
            n_jobs=n_jobs,
            random_state=random_state,
            consensus_anchors=consensus_anchors,
        )

        started = time.perf_counter()
        carve.fit(X[idx])
        elapsed = time.perf_counter() - started

        labels = carve.get_labels(measure=measure, rule=rule)
        rows.append(
            {
                "n": size,
                "n_configs": int(carve.estimator_results_.shape[0]),
                "wall_clock_s": float(elapsed),
                "peak_rss_bytes": int(peak_rss_bytes()),
                "selected_k": int(carve.get_k(measure=measure, rule=rule)),
                "ari": (
                    float("nan")
                    if y_arr is None
                    else float(adjusted_rand_score(y_arr[idx], labels))
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "n",
            "n_configs",
            "wall_clock_s",
            "peak_rss_bytes",
            "selected_k",
            "ari",
        ],
    )
