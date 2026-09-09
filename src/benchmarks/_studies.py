"""Case-study compute: the CVI sweep and the CARVE fit cache.

No matplotlib here. The routine this replaces swept the classical indices and
then built a figure and called plt.show() in the same call, so the sweep
could not be reused without also drawing it.
"""

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
from ._estimators import apply_random_state, param_grids
from ._types import EstimatorSpec, Study

CVI_SWEEP_METRICS: tuple[str, ...] = (
    "silhouette",
    "gap",
    "davies_bouldin",
    "calinski_harabasz",
)


def _model_label(estimator_cls: type[ClusterMixin], params: dict[str, Any]) -> str:
    """A short readable name for one estimator configuration."""
    if not params:
        return estimator_cls.__name__
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{estimator_cls.__name__} ({rendered})"


def _spec_for(estimator_cls: type[ClusterMixin]) -> EstimatorSpec:
    """Map an estimator class back to its spec, for the gap statistic."""
    from ._estimators import ESTIMATOR_CLASSES

    for name, cls in ESTIMATOR_CLASSES.items():
        if cls is estimator_cls:
            return EstimatorSpec(name=name)
    raise ValueError(f"No EstimatorSpec is registered for {estimator_cls.__name__}.")


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
    spec = _spec_for(estimator_cls)

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
) -> CARVE:
    """Fit CARVE on a case study, caching the fitted state to disk.

    A Levine fit takes hours, so the cache is what makes regenerating a figure
    practical. The saved state does not carry the data matrix, so X_ is
    restored after loading, matching the notebook this replaces.
    """
    cache_path = Path(cache_path)

    if cache_path.is_file() and not force:
        carve = CARVE.load(str(cache_path))
        carve.X_ = np.asarray(X)
        return carve

    carve = CARVE(
        estimator_param_grids=model_grids,
        n_resamples=n_resamples,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    reference = None if y is None else np.asarray(y)
    carve.fit(np.asarray(X), reference_labels=reference)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    carve.save(str(cache_path))
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

    The scale is part of the filename deliberately. fit_or_load_carve loads
    whatever file sits at the path it is handed, so a shared path would let a
    publication run silently reuse a development-scale fit.
    """
    resolve_scale(study, scale)  # validates scale, raising if it is unknown
    name = _scale_name(study, scale)
    return Path(root) / f"carve_{study.name}_{name}.carve"


STUDIES: dict[str, Study] = {
    "klein": Study(
        name="klein",
        loader=_klein_loader,
        estimator=EstimatorSpec(name="agglomerative"),
        candidate_k=tuple(range(2, 11)),
        scales={"dev": 400, "publication": 0.5},
        default_scale="publication",
    ),
    "levine32": Study(
        name="levine32",
        loader=_levine_loader,
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=tuple(range(7, 18)),
        scales={"dev": 800, "publication": 5000},
        default_scale="publication",
    ),
}


def study_model_grids(
    study: Study,
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """The study's own estimator plus spectral clustering.

    The manuscript reports running two estimators per case study, but the
    pair differs by study: Klein sweeps Ward agglomerative clustering and
    spectral clustering with self-tuning affinity; Levine sweeps KMeans and
    spectral. This mirrors the simulated benchmarks fixing one estimator per
    scenario, except a case study adds spectral as a second one.
    """
    return param_grids(study.estimator, study.candidate_k) + param_grids(
        EstimatorSpec(name="spectral"), study.candidate_k
    )
