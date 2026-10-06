"""Case-study compute: the CVI sweep and the CARVE fit cache.

No matplotlib here. The routine this replaces swept the classical indices and
then built a figure and called plt.show() in the same call, so the sweep
could not be reused without also drawing it.
"""

import hashlib
from dataclasses import fields
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, ClusterMixin
from sklearn.metrics import adjusted_rand_score

from carve import CARVE

from ._artifacts import check_fingerprint, fingerprint, fingerprint_path
from ._cvi import calculate_cvi, select_k
from ._estimators import (
    DENSE_PAIRWISE_ESTIMATORS,
    ESTIMATOR_CLASSES,
    RESOLUTION_ESTIMATORS,
    apply_random_state,
    param_grids,
    resolution_grids,
)
from ._preprocessing import preprocessing_fingerprint
from ._types import EstimatorSpec, PreprocessingSpec, Study

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

#: CARVE's own default resample count. A study's k-based run at this count
#: keeps the cache filename it had before carve_cache_path added run keys.
PACKAGE_N_RESAMPLES: int = next(
    field.default for field in fields(CARVE) if field.name == "n_resamples"
)

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
    through study_model_grids would walk straight past it anyway. cvi_sweep
    and fit_or_load_carve are the one place every case-study entry point
    brings X and model_grids together before handing them to a real
    estimator, which is why the check lives here instead.
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


def fit_or_load_carve(
    X: np.ndarray,
    y: np.ndarray | pd.Series | None,
    *,
    cache_path: Path,
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
    n_resamples: int = PACKAGE_N_RESAMPLES,
    n_jobs: int = 1,
    random_state: int = 42,
    force: bool = False,
    consensus_anchors: int | None = None,
    randomize_preprocessing: bool = False,
    normalization_options: list[Any] | None = None,
    dim_reduction_options: list[Any] | None = None,
    classifier: ClassifierMixin | None = None,
) -> CARVE:
    """Fit CARVE on a case study, caching the fitted state to disk.

    A Levine fit takes hours, so the cache is what makes regenerating a figure
    practical. The saved state does not carry the data matrix, so X_ is
    restored after loading, matching the notebook this replaces.

    The cache is found by path alone. A caller that changes the grid, the
    resample count or the preprocessing must pass the same to
    carve_cache_path, which keys the filename on them.

    consensus_anchors : int or None, default=None
        Forwarded to CARVE only when not None, so studies that leave it at
        the package default (an exact, unanchored run) are unaffected. hECA
        sets this at case-study scale, where an exact consensus matrix does
        not fit.
    randomize_preprocessing : bool, default=False
        Forwarded to CARVE.fit only when True.
    normalization_options, dim_reduction_options : list or None
        Forwarded to CARVE only when not None, in the option syntax CARVE
        takes; _preprocessing.resolve_preprocessing builds both from a
        Study's preprocessing. Passing either without randomize_preprocessing
        raises, since CARVE would ignore it.
    classifier : sklearn classifier or None, default=None
        Forwarded to CARVE only when not None. hECA's fit passes a forest
        with the default's settings that times itself (see _timing).
    """
    cache_path = Path(cache_path)
    _check_dense_fit(np.asarray(X).shape[0], model_grids)
    if not randomize_preprocessing and (
        normalization_options is not None or dim_reduction_options is not None
    ):
        raise ValueError(
            "normalization_options and dim_reduction_options only apply to a "
            "fit with randomize_preprocessing=True; CARVE would ignore them."
        )

    if cache_path.is_file() and not force:
        check_fingerprint(cache_path, X)
        carve = CARVE.load(str(cache_path))
        carve.X_ = np.asarray(X)
        return carve

    optional = {
        name: value
        for name, value in (
            ("consensus_anchors", consensus_anchors),
            ("normalization_options", normalization_options),
            ("dim_reduction_options", dim_reduction_options),
            ("classifier", classifier),
        )
        if value is not None
    }
    carve = CARVE(
        estimator_param_grids=model_grids,
        n_resamples=n_resamples,
        n_jobs=n_jobs,
        random_state=random_state,
        **optional,
    )
    reference = None if y is None else np.asarray(y)
    carve.fit(
        np.asarray(X),
        reference_labels=reference,
        **({"randomize_preprocessing": True} if randomize_preprocessing else {}),
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    carve.save(str(cache_path))
    fingerprint_path(cache_path).write_text(fingerprint(X))
    return carve


def _klein_loader(subsample: int | float | None):
    from .datasets import load_klein

    # Publication scale is every one of the 2,717 preprocessed cells.
    return load_klein(subsample=subsample, random_state=42)


def _levine_loader(subsample: int | float | None):
    from .datasets import load_levine32

    # Manuscript line 627: a stratified subsample of 5,000 cells.
    return load_levine32(subsample=subsample, random_state=42)


def _cusanovich_loader(subsample: int | float | None):
    from .datasets import load_cusanovich

    # The reference is the source publication's own 30 clusters, which assign
    # every cell, so drop_unknown stays at its default and the comparison is
    # over the cells that partition covers.
    return load_cusanovich(subsample=subsample, random_state=42, label_column="cluster")


def _heca_loader(subsample: int | float | None):
    from .datasets import load_heca

    return load_heca(subsample=subsample, random_state=42, label_column="organ")


def _scale_name(study: Study, scale: str | None) -> str:
    """Resolve a scale argument to the scale name it refers to.

    None means the study's own default. This is the one place the
    default scale is chosen; resolve_scale, load_study, and
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


def _run_key(
    study: Study,
    *,
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]] | None,
    n_resamples: int,
    preprocessing: PreprocessingSpec | None,
) -> str | None:
    """A short hash of what a fit runs, or None for the study's default run.

    The default run is the study's own k-based grid (study_model_grids), no
    randomized preprocessing, and the package's resample count. Anything
    else hashes the grid, whose keys name the swept parameter, the
    preprocessing spec with the defaults it binds, and n_resamples.
    """
    default_grids = (
        study_model_grids(study)
        if study.candidate_k and study.estimator.name not in RESOLUTION_ESTIMATORS
        else None
    )
    grids = default_grids if model_grids is None else model_grids
    if grids is None:
        raise ValueError(
            f"Study {study.name!r} has no k-based grid to default to; pass the "
            "model_grids its fit runs."
        )
    if (
        grids == default_grids
        and preprocessing is None
        and n_resamples == PACKAGE_N_RESAMPLES
    ):
        return None
    payload = repr((grids, preprocessing_fingerprint(preprocessing), n_resamples))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def carve_cache_path(
    study: Study,
    *,
    scale: str | None = None,
    root: Path,
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]] | None = None,
    n_resamples: int = PACKAGE_N_RESAMPLES,
    preprocessing: PreprocessingSpec | None = None,
) -> Path:
    """Where a study's fitted CARVE state is cached, per scale and run.

    Both the scale name and a short hash of its resolved size are part of
    the filename deliberately. fit_or_load_carve loads whatever file sits at
    the path it is handed, so a shared path would let one scale silently
    reuse another's fit (Cusanovich's atlas run serving its 5,000-cell
    publication fit) -- that is what the scale name guards against. The hash
    guards the same failure one level up: editing STUDIES[...].scales[name]
    (say, Levine's "publication" from 5,000 to 10,000)
    changes what that scale resolves to without changing its name, and
    without the hash fit_or_load_carve would silently load the fit taken at
    the old size. A hash reads better than the raw resolved size for the
    None case (full data), which has no natural filename spelling.

    A run key guards the same failure across kinds of fit: a k-based fit, a
    Leiden resolution fit and a randomized-preprocessing fit at one scale
    would otherwise share a filename. Pass the model_grids, n_resamples and
    preprocessing the fit runs. The study's default run (its k-based grid,
    no preprocessing, the package's resample count) carries no run key, so
    the caches written before run keys existed keep their names. A study
    with no k-based grid must pass model_grids.
    """
    resolved = resolve_scale(study, scale)  # raises if the scale is unknown
    name = _scale_name(study, scale)
    size_key = hashlib.sha1(repr(resolved).encode()).hexdigest()[:8]
    run_key = _run_key(
        study,
        model_grids=model_grids,
        n_resamples=n_resamples,
        preprocessing=preprocessing,
    )
    stem = f"carve_{study.name}_{name}_{size_key}"
    filename = f"{stem}.carve" if run_key is None else f"{stem}_{run_key}.carve"
    return Path(root) / filename


STUDIES: dict[str, Study] = {
    "klein": Study(
        name="klein",
        loader=_klein_loader,
        estimator=EstimatorSpec(name="agglomerative"),
        candidate_k=tuple(range(2, 11)),
        # Every cell at publication scale: 2,717, whose resamples of 1,679
        # put spectral on its sparse eigensolver branch (n >= 1000).
        scales={"publication": None},
        default_scale="publication",
        partners=(EstimatorSpec(name="spectral"),),
        # k=2 stays a candidate: the manuscript reports stability's own
        # selection (spectral at k=2) next to the headline k=4 (Ward,
        # generalizability, 1SE, which k=2 does not affect). The Klein
        # notebook and the ablation read both from here.
        not_two=False,
        reported_k=4,
    ),
    "levine32": Study(
        name="levine32",
        loader=_levine_loader,
        # With 10 restarts, KMeans at k=7 on the 5,000 cells stops at one of
        # two solutions depending on the seed (ARI 0.625 or 0.849); with 100
        # it finds the lower-inertia one from every seed tried (measured
        # 2026-09-30). The CVI sweep and CARVE's fits share this grid.
        estimator=EstimatorSpec(name="kmeans_n_init_100"),
        candidate_k=tuple(range(7, 18)),
        scales={"publication": 5000},
        default_scale="publication",
        partners=(EstimatorSpec(name="spectral"),),
    ),
    "cusanovich": Study(
        name="cusanovich",
        loader=_cusanovich_loader,
        # The source clustered its t-SNE with Seurat's Louvain, so Louvain
        # over resolution gives their operating point a position on the same
        # axis. Nothing k-based is swept.
        estimator=EstimatorSpec(name="louvain"),
        candidate_k=(),
        # Every cell at the atlas scale. With the resample count and
        # pipelines below it takes about 120 core-hours, about 28 hours at
        # five workers on an 11-core, 18 GB Mac (extrapolated from timings
        # measured 2026-09-14). A perplexity-100 fit peaks near 2.2 GB, and
        # workers beyond five add little throughput.
        scales={"publication": 5000, "atlas": None},
        default_scale="atlas",
        partners=(),
        # Log-spaced from 0.02 to 3.0, 15 values. Louvain's cluster count at a
        # given resolution grows with the number of cells and differs by
        # pipeline: on 50,164 atlas cells a converged perplexity-30 t-SNE gives
        # 24 clusters at 0.05, 30 at 0.2 and 127 at 3.0, while the LSI gives 13
        # at 0.2 and 34 at 3.0 (measured 2026-09-14). A log grid reaches the
        # source's 30 clusters for both.
        resolutions=tuple(float(f"{0.02 * 150 ** (i / 14):.2g}") for i in range(15)),
        # Pinned rather than left at the package default. Both
        # consensus_matrices_ and consensus_generalizability_matrices_ are
        # retained per configuration, so retained memory is
        # n_configs * 2 * m**2 * 8 bytes: 15 * 2 * 2000**2 * 8 B = 0.96 GB
        # for the 15-configuration sweep at 2000 anchors, against 6.0 GB at
        # the package default of 5000 once n exceeds anchor_threshold. At
        # both scales the run anchors to 2000, the count hECA uses.
        consensus_anchors=2000,
        # 150 resamples over four pipelines give each 37 or 38 under
        # stratified allocation. The preliminary atlas run used 50.
        n_resamples=150,
        preprocessing=PreprocessingSpec(
            # The LSI is already scaled by its singular values; the source
            # does not standardize or log-transform it.
            normalization=(("identity", {}),),
            dim_reduction=(
                # The source's LSI, all 50 components, and its first 10: the
                # number of components is the LSI's counterpart of the number
                # of principal components. Keeping 30, the Signac and ArchR
                # default, or dropping the depth-correlated first component
                # gave the 50-component partition back up to Louvain's own
                # run-to-run variation on 50,164 atlas cells (2026-10-03), so
                # neither is a pipeline of its own; 10 components differ.
                ("lsi", {"n_components": [50]}),
                ("lsi", {"n_components": [10]}),
                # t-SNE of all 50 components at the source's perplexity, 30,
                # and at 100, the top of the standard range of 10 to 100 for
                # large data (Kobak and Berens 2019) and near the log-scale
                # midpoint between 30 and n/100 of a subsample (310 to 501).
                # The preliminary run's perplexity 300 stayed within about
                # 0.03 of 100 at every resolution, at 1.4 times the cost.
                # Perplexities compared on a small subsample do not transfer:
                # in a 927-cell subsample most source clusters are already
                # smaller than perplexity 30. Each perplexity is its own
                # option so stratified allocation balances resamples across
                # them; one option with two values would split t-SNE's share
                # at random. The source's other Rtsne settings (5,000
                # iterations, random initialization, learning rate) are bound
                # in PREPROCESSOR_DEFAULTS.
                ("tsne", {"perplexity": [30]}),
                ("tsne", {"perplexity": [100]}),
            ),
        ),
        # The figure draws CARVE's partition and the source's on one t-SNE of
        # every cell at perplexity 100, the more stable of the two t-SNE
        # pipelines at the selected resolution of the 150-resample run
        # (stability 0.792 against 0.776 at perplexity 30, 2026-10-04).
        map_option=("tsne", {"perplexity": [100]}),
    ),
    "heca": Study(
        name="heca",
        loader=_heca_loader,
        # Leiden on two graphs: 15 neighbors, the scanpy convention hECA's
        # own EpiScanpy annotation inherits, and 50, the graph CATlas (Zhang
        # 2021) and Hocker 2021 clustered; those two supply 221,029 of the
        # pooled cells. Both optimize modularity. See the 2026-10-01 spec.
        estimator=EstimatorSpec(name="leiden", params=(("n_neighbors", 15),)),
        partners=(EstimatorSpec(name="leiden", params=(("n_neighbors", 50),)),),
        candidate_k=(),
        scales={"publication": None},
        default_scale="publication",
        # Provisional: 15 log-spaced values from 0.005 to 3.0. The single
        # studies the cells come from reported 1.0 and 1.5 on 8,500 to 91,500
        # cells; each resample here clusters about 430,000, where modularity
        # splits further at the same value, so organ level needs far smaller
        # ones. python -m benchmarks.heca calibrate proposes the grid that
        # replaces this one before the fit runs.
        resolutions=tuple(float(f"{0.005 * 600 ** (i / 14):.2g}") for i in range(15)),
        # 30 configurations each retain a stability and a generalizability
        # consensus block of 2000 x 2000 anchors, about 1 GB in all.
        consensus_anchors=2000,
        n_resamples=100,
    ),
}


def study_model_grids(
    study: Study,
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """The study's own estimator plus its declared partners, over candidate_k.

    Klein sweeps Ward agglomerative and spectral; Levine sweeps KMeans and
    spectral. Each study declares this on Study.partners. Cusanovich and
    hECA sweep resolution only, so neither has a k-based grid and this
    raises for both.
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
    """The study's resolution sweep: its own estimator, then every partner.

    Cusanovich sweeps Louvain alone; hECA sweeps Leiden at two neighbor
    counts, declared as its estimator and one partner. A separate CARVE run
    from study_model_grids: SweepSpec is frozen, so a k-based and a
    resolution-based sweep cannot share one run.
    """
    if not study.resolutions:
        raise ValueError(f"Study {study.name!r} declares no resolutions.")
    if study.estimator.name not in RESOLUTION_ESTIMATORS:
        raise ValueError(
            f"Study {study.name!r} sweeps {study.estimator.name!r}, which does "
            "not take a resolution; use study_model_grids."
        )
    grids = resolution_grids(study.estimator, study.resolutions)
    for partner in study.partners:
        grids += resolution_grids(partner, study.resolutions)
    return grids
