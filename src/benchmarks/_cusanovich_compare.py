"""Compute for the Cusanovich case study on randomized preprocessing.

The source clustered a two-dimensional t-SNE of its LSI into 30 clusters.
These functions set that recipe beside CARVE's: which preprocessing pipeline
CARVE rates best at the configuration it selects, and where the source's
granularity falls on CARVE's pooled resolution axis.
prepare_cusanovich_inputs gathers both into CusanovichInputs; nothing here
draws, figures._cusanovich_results does.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from carve._pipeline import PipelineSpec, PipelineStep, pipeline_from_spec
from carve._selection import MEASURE_MAP

from ._preprocessing import PREPROCESSOR_NAMES

#: How far the nearest pooled observed cluster count may sit from the target
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
    carve: Any, *, method_id: str, target_k: int
) -> tuple[float, float]:
    """The resolution at which one configuration's clusterings come nearest target_k.

    Reads the estimator_results_ rows for method_id and returns (resolution,
    observed cluster count) for the row whose n_clusters_observed is nearest
    target_k, ties going to the lower resolution. The count is the mean over
    all resamples at that resolution, pooled over pipelines, which is the
    count the figure's secondary axis names. Warns when it is more than
    OPERATING_POINT_TOLERANCE of target_k away, which means the resolution
    grid does not reach the source's granularity.
    """
    results = carve.estimator_results_
    rows = results.loc[results["method_id"] == method_id]
    if rows.empty:
        raise ValueError(
            f"No estimator_results_ rows for {method_id!r}. "
            f"Available: {sorted(results['method_id'].unique())}."
        )

    rows = rows.sort_values("sweep_value", kind="stable")
    distance = (rows["n_clusters_observed"] - target_k).abs()
    row = rows.loc[distance.idxmin()]
    resolution = float(row["sweep_value"])
    observed = float(row["n_clusters_observed"])
    if abs(observed - target_k) > OPERATING_POINT_TOLERANCE * target_k:
        warnings.warn(
            f"The {method_id!r} clustering nearest {target_k} clusters has "
            f"{observed:.1f}, at resolution {resolution:g}: more than "
            f"{OPERATING_POINT_TOLERANCE:.0%} away. Extend the study's "
            "resolution grid.",
            UserWarning,
            stacklevel=2,
        )
    return resolution, observed


@dataclass(frozen=True)
class CusanovichInputs:
    """Everything the Cusanovich figure draws, already computed.

    Assembling this is compute (prepare_cusanovich_inputs); drawing it is
    reporting (figures.figure_cusanovich_results). Keeping them apart lets the
    figure be tested without fitting anything, as with CompositeInputs.

    carve_labels are CARVE's consensus labels relabeled onto y's codes, so a
    CARVE cluster takes the color of the source cluster it best matches.
    embedding_A is all of X embedded by the best pipeline, named by
    embedding_A_labels. operating_point is (resolution, observed cluster
    count) on the selected configuration's pooled curve. measure, rule and
    not_two are the selection every panel reads.
    """

    X: np.ndarray
    y: np.ndarray
    carve: Any
    carve_labels: np.ndarray
    embedding_A: np.ndarray
    embedding_A_labels: tuple[str, str]
    source_tsne: np.ndarray
    best_pipeline_row: pd.Series
    operating_point: tuple[float, float]
    measure: str = "stability"
    rule: str = "1se"
    not_two: bool = False


def prepare_cusanovich_inputs(
    X: np.ndarray,
    y: np.ndarray,
    carve: Any,
    *,
    source_tsne: np.ndarray,
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
    random_state: int = 42,
) -> CusanovichInputs:
    """Assemble the Cusanovich figure's inputs from a randomized fit.

    Embeds all of X with the best pipeline at the selected configuration,
    relabels CARVE's consensus labels onto y, and places the source's
    operating point on the selected configuration's pooled curve at y's own
    cluster count. This is compute; call it once and draw from the result.
    """
    from .figures._case_study import _align_to_reference

    X = np.asarray(X)
    y = np.asarray(y)
    row, spec = best_pipeline(carve, measure=measure, rule=rule, not_two=not_two)
    embedding, axis_labels = pipeline_embedding(X, spec, random_state=random_state)
    labels = carve.get_labels(measure=measure, rule=rule, not_two=not_two)

    return CusanovichInputs(
        X=X,
        y=y,
        carve=carve,
        carve_labels=_align_to_reference(labels, y),
        embedding_A=embedding,
        embedding_A_labels=axis_labels,
        source_tsne=np.asarray(source_tsne, dtype=np.float64),
        best_pipeline_row=row,
        operating_point=source_operating_point(
            carve, method_id=str(row["method_id"]), target_k=int(np.unique(y).size)
        ),
        measure=measure,
        rule=rule,
        not_two=not_two,
    )


def selection_summary(inputs: CusanovichInputs) -> pd.DataFrame:
    """The numbers the case study reports, one (quantity, value) row each.

    Written next to the figure by save_tables, so every number the manuscript
    quotes has a file behind it.
    """
    carve = inputs.carve
    selected, _, n_clusters, _ = carve._select_row(
        measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
    )
    best = inputs.best_pipeline_row
    resolution, observed = inputs.operating_point
    rows = [
        ("measure", inputs.measure),
        ("rule", inputs.rule),
        ("not_two", inputs.not_two),
        ("n_cells", int(inputs.y.size)),
        ("source_clusters", int(np.unique(inputs.y).size)),
        ("selected_method", selected["method_label"]),
        ("selected_resolution", float(selected["sweep_value"])),
        ("selected_n_clusters", int(n_clusters)),
        ("selected_ari_stability", float(selected["ari_stability"])),
        ("selected_ari_generalizability", float(selected["ari_generalizability"])),
        ("best_pipeline", best["pipeline"]),
        ("best_pipeline_n_resamples", int(best["n_resamples"])),
        ("best_pipeline_ari_stability", float(best["ari_stability"])),
        ("best_pipeline_ari_generalizability", float(best["ari_generalizability"])),
        ("operating_point_resolution", resolution),
        ("operating_point_n_clusters", observed),
        (
            "consensus_ari_vs_source_clusters",
            float(adjusted_rand_score(inputs.y, inputs.carve_labels)),
        ),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


#: The CSVs save_tables writes, next to cusanovich_results.png.
TABLE_FILENAMES: tuple[str, str] = (
    "cusanovich_preprocessing_results.csv",
    "cusanovich_selection_summary.csv",
)


def save_tables(inputs: CusanovichInputs, out_dir: Path) -> list[Path]:
    """Write preprocessing_results_ and the selection summary as CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path, summary_path = (out_dir / name for name in TABLE_FILENAMES)
    inputs.carve.preprocessing_results_.to_csv(results_path, index=False)
    selection_summary(inputs).to_csv(summary_path, index=False)
    return [results_path, summary_path]
