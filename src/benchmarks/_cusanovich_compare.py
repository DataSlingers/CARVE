"""Compute for the Cusanovich case study on randomized preprocessing.

The source clustered a two-dimensional t-SNE of its LSI into 30 clusters.
These functions set that recipe beside CARVE's: which preprocessing pipeline
CARVE rates best at the configuration it selects, where the source's
granularity falls on the t-SNE pipeline's resolution axis, and how well the
published partition generalizes under the classifier probe CARVE applies to
its own clusterings. No matplotlib here; figures._cusanovich_results draws
the result.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedShuffleSplit

from carve._pipeline import PipelineSpec, PipelineStep, pipeline_from_spec
from carve._selection import MEASURE_MAP
from carve._utils import default_generalizability_classifier

from ._preprocessing import PREPROCESSOR_NAMES

#: How far a pipeline's nearest observed cluster count may sit from the target
#: before source_operating_point warns, as a fraction of the target.
OPERATING_POINT_TOLERANCE = 0.25

#: What a panel calls the output axes of each dimensionality reduction. The
#: identity pipeline passes the loader's LSI through.
_AXIS_PREFIXES = {
    PREPROCESSOR_NAMES["identity"]: "LSI",
    PREPROCESSOR_NAMES["pca"]: "PC",
    PREPROCESSOR_NAMES["tsne"]: "t-SNE",
    PREPROCESSOR_NAMES["umap"]: "UMAP",
}

#: umap-learn warns on every seeded fit that the seed disables its own
#: parallelism; _runner.embed_resample silences the same message.
_UMAP_SEED_WARNING = r"n_jobs value .* overridden to 1 by setting random_state"

_ARI_COLUMNS = ("ari_stability", "ari_generalizability")


def _per_pipeline_table(carve: Any) -> pd.DataFrame:
    table = carve.preprocessing_results_
    if table is None:
        raise ValueError(
            "This CARVE fit has no per-pipeline table; fit it with "
            "randomize_preprocessing=True."
        )
    return table


def axis_prefix(step: PipelineStep) -> str:
    """The name a panel gives the output axes of a dimensionality reduction."""
    name = step.name if step.name is not None else getattr(step.cls, "__name__", "")
    return _AXIS_PREFIXES.get(name, name or "Component")


def best_pipeline(
    carve: Any,
    *,
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
) -> tuple[pd.Series, PipelineSpec]:
    """The pipeline CARVE rates best at the configuration it selects.

    The configuration is carve._select_row under measure, rule and not_two,
    the selection every figure reads. Among the preprocessing_results_ rows
    at that configuration's method_id and sweep value, the one with the
    highest measure wins; rows at any other configuration never compete, even
    when they score higher.

    Returns
    -------
    (row, spec) : the preprocessing_results_ row and its PipelineSpec from
        preprocessing_pipelines_.
    """
    table = _per_pipeline_table(carve)
    column = MEASURE_MAP.get(measure)
    if column not in _ARI_COLUMNS:
        raise ValueError(
            "The per-pipeline table carries only the ARI criteria, so measure "
            f"must be stability or generalizability (or an alias); got {measure!r}."
        )

    selected, _, _, _ = carve._select_row(measure=measure, rule=rule, not_two=not_two)
    at_selected = table.loc[
        (table["method_id"] == selected["method_id"])
        & np.isclose(table["sweep_value"].astype(float), float(selected["sweep_value"]))
    ]
    if at_selected.empty:
        raise RuntimeError(
            f"preprocessing_results_ has no rows at the selected configuration "
            f"({selected['method_id']}, {selected['sweep_value']})."
        )
    row = at_selected.loc[at_selected[column].idxmax()]
    return row, carve.preprocessing_pipelines_[row["pipeline"]]


def pipeline_embedding(
    X: np.ndarray, spec: PipelineSpec, *, random_state: int
) -> tuple[np.ndarray, tuple[str, str]]:
    """Embed all of X with one pipeline, for drawing.

    Fits a fresh pipeline_from_spec(spec, random_state) on the full X and
    keeps its first two output columns: the whole embedding for a
    two-dimensional reduction, LSI 1 and LSI 2 for the identity pipeline on
    the LSI. The axis names come back with it, so a panel names what it draws.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_UMAP_SEED_WARNING, category=UserWarning
        )
        Z = np.asarray(pipeline_from_spec(spec, random_state).fit_transform(X))
    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError(
            f"Pipeline {spec.label!r} produced an output of shape {Z.shape}; "
            "drawing it needs at least two columns."
        )
    prefix = axis_prefix(spec.dim_reduction)
    return Z[:, :2], (f"{prefix} 1", f"{prefix} 2")


def source_operating_point(
    carve: Any, *, pipeline: str, target_k: int
) -> tuple[float, float]:
    """The resolution at which one pipeline's clusterings come nearest target_k.

    Reads the preprocessing_results_ rows for the named pipeline and returns
    (resolution, observed cluster count) for the row whose n_clusters_observed
    is nearest target_k, ties going to the lower resolution. The count is a
    mean over that pipeline's resamples at that resolution. Warns when it is
    more than OPERATING_POINT_TOLERANCE of target_k away, which means the
    resolution grid does not reach the source's granularity for this pipeline.
    """
    table = _per_pipeline_table(carve)
    rows = table.loc[table["pipeline"] == pipeline]
    if rows.empty:
        raise ValueError(
            f"No per-pipeline rows for {pipeline!r}. "
            f"Available: {sorted(table['pipeline'].unique())}."
        )
    if rows["method_id"].nunique() > 1:
        raise ValueError(
            f"Pipeline {pipeline!r} has rows for more than one estimator "
            "configuration; an operating point is defined on one resolution axis."
        )

    rows = rows.sort_values("sweep_value", kind="stable")
    distance = (rows["n_clusters_observed"] - target_k).abs()
    row = rows.loc[distance.idxmin()]
    resolution = float(row["sweep_value"])
    observed = float(row["n_clusters_observed"])
    if abs(observed - target_k) > OPERATING_POINT_TOLERANCE * target_k:
        warnings.warn(
            f"The {pipeline!r} clustering nearest {target_k} clusters has "
            f"{observed:.1f}, at resolution {resolution:g}: more than "
            f"{OPERATING_POINT_TOLERANCE:.0%} away. Extend the study's "
            "resolution grid.",
            UserWarning,
            stacklevel=2,
        )
    return resolution, observed


def published_partition_generalizability(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int,
    subsample_ratio: float,
    n_trees: int,
    random_state: int,
    n_jobs: int = 1,
) -> tuple[float, float]:
    """How well a fixed partition generalizes under CARVE's classifier probe.

    For each of n_splits stratified splits with subsample_ratio of the rows in
    training, the classifier CARVE trains for generalizability
    (carve._utils.default_generalizability_classifier, a random forest with
    n_trees trees) is fit on X[train] with the partition's labels and predicts
    X[test]; the split scores the ARI between those predictions and the
    partition's own test labels. Split s seeds its forest with
    random_state + s. Stratifying keeps every class in training, which a
    random split can miss for a small one. Stability of a fixed partition is
    not defined, so none is computed.

    Returns
    -------
    (mean, se) : the mean ARI over splits and its standard error, NaN for a
        single split, as in CARVE's own tables.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, train_size=subsample_ratio, random_state=random_state
    )
    scores = []
    for split, (train, test) in enumerate(splitter.split(X, labels)):
        classifier = default_generalizability_classifier(
            classifier=None,
            n_features=X.shape[1],
            n_trees=n_trees,
            random_state=random_state + split,
            n_jobs=n_jobs,
        )
        classifier.fit(X[train], labels[train])
        scores.append(adjusted_rand_score(labels[test], classifier.predict(X[test])))

    scores = np.asarray(scores, dtype=float)
    se = (
        float(np.std(scores, ddof=1) / np.sqrt(scores.size))
        if scores.size > 1
        else float("nan")
    )
    return float(np.mean(scores)), se
