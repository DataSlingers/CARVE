"""Compute for the Cusanovich case study on randomized preprocessing.

The source clustered a two-dimensional t-SNE of its LSI into 30 clusters.
These functions set that recipe beside CARVE's: which preprocessing pipeline
CARVE rates best at the configuration it selects, and where the source's
granularity falls on CARVE's pooled resolution axis, plus the map both
scatter panels draw on. prepare_cusanovich_inputs gathers them into
CusanovichInputs; nothing here draws, figures._cusanovich_results does.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from carve._pipeline import PipelineSpec, pipeline_from_spec
from carve._selection import MEASURE_MAP

from ._preprocessing import PREPROCESSOR_NAMES
from ._tables import DisplayTable

#: How far the nearest pooled observed cluster count may sit from the target
#: before source_operating_point warns, as a fraction of the target.
OPERATING_POINT_TOLERANCE = 0.25

_ARI_COLUMNS = ("ari_stability", "ari_generalizability")

#: What a panel calls a two-dimensional map's axes, by display name.
_MAP_AXIS_NAMES = {
    PREPROCESSOR_NAMES["tsne"]: "t-SNE",
    PREPROCESSOR_NAMES["umap"]: "UMAP",
}


def _per_pipeline_table(carve: Any) -> pd.DataFrame:
    table = carve.preprocessing_results_
    if table is None:
        raise ValueError(
            "This CARVE fit has no per-pipeline table; fit it with "
            "randomize_preprocessing=True."
        )
    return table


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


def map_pipeline(carve: Any, option: tuple[str, dict[str, list[Any]]]) -> PipelineSpec:
    """The fit's pipeline whose dimensionality reduction is option.

    option is a PreprocessingSpec dim_reduction entry, (registry key, grid),
    with one value per grid key, as Study.map_option holds.
    """
    key, grid = option
    params = {name: values[0] for name, values in grid.items()}
    for spec in carve.preprocessing_pipelines_.values():
        step = spec.dim_reduction
        if step.name == PREPROCESSOR_NAMES.get(key) and step.params == params:
            return spec
    raise ValueError(
        f"map_option {option!r} matches no pipeline this fit drew; its pipelines "
        f"are {sorted(carve.preprocessing_pipelines_)}."
    )


def map_coordinates(
    X: np.ndarray, spec: PipelineSpec, *, random_state: int
) -> tuple[np.ndarray, str, tuple[str, str]]:
    """Fit one two-dimensional pipeline on all of X, for both scatter panels.

    Returns the coordinates, the map's name for a title (a t-SNE with its
    perplexity) and its axis names.
    """
    Z = np.asarray(pipeline_from_spec(spec, random_state).fit_transform(X))
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(
            f"Pipeline {spec.label!r} gives an output of shape {Z.shape}; a map "
            "needs two columns."
        )
    step = spec.dim_reduction
    prefix = _MAP_AXIS_NAMES.get(step.name, step.name or step.label)
    perplexity = step.params.get("perplexity")
    name = prefix if perplexity is None else f"{prefix}, perplexity {perplexity:g}"
    return Z, name, (f"{prefix} 1", f"{prefix} 2")


def _nearest(rows: pd.DataFrame, target_k: int) -> pd.Series:
    """The row whose n_clusters_observed is nearest target_k, ties going to
    the lower resolution."""
    rows = rows.sort_values("sweep_value", kind="stable")
    return rows.loc[(rows["n_clusters_observed"] - target_k).abs().idxmin()]


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

    row = _nearest(rows, target_k)
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
    map_coordinates is the map both A and B draw on, named by map_name and
    map_axis_labels. operating_point is (resolution, observed cluster
    count) on the selected configuration's pooled curve. measure, rule and
    not_two are the selection every panel reads.
    """

    X: np.ndarray
    y: np.ndarray
    carve: Any
    carve_labels: np.ndarray
    map_coordinates: np.ndarray
    map_name: str
    map_axis_labels: tuple[str, str]
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
    map_option: tuple[str, dict[str, list[Any]]],
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
    random_state: int = 42,
) -> CusanovichInputs:
    """Assemble the Cusanovich figure's inputs from a randomized fit.

    Finds the best pipeline at the selected configuration, relabels CARVE's
    consensus labels onto y, places the source's operating point on the
    selected configuration's pooled curve at y's own cluster count, and fits
    the map_option pipeline on all of X, seeded with random_state, as the map
    both scatter panels draw on. This is compute; call it once and draw from
    the result.
    """
    from .figures._case_study import _align_to_reference

    X = np.asarray(X)
    y = np.asarray(y)
    row, _ = best_pipeline(carve, measure=measure, rule=rule, not_two=not_two)
    coordinates, name, axis_labels = map_coordinates(
        X, map_pipeline(carve, map_option), random_state=random_state
    )
    labels = carve.get_labels(measure=measure, rule=rule, not_two=not_two)

    return CusanovichInputs(
        X=X,
        y=y,
        carve=carve,
        carve_labels=_align_to_reference(labels, y),
        map_coordinates=coordinates,
        map_name=name,
        map_axis_labels=axis_labels,
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


CUSANOVICH_SELECTION_CAPTION: str = (
    "S16 Table. CARVE's selection and Cusanovich et al.'s granularity on the "
    "Cusanovich data."
)
CUSANOVICH_PIPELINE_CAPTION: str = (
    "S17 Table. Stability and generalizability by preprocessing pipeline on the "
    "Cusanovich data."
)

_GRANULARITY_COLUMNS: list[str] = [
    "Resolution",
    "Clusters",
    "Stability",
    "Generalizability",
]


def _granularity_cells(row: pd.Series) -> list[str]:
    return [
        f"{float(row['sweep_value']):g}",
        f"{float(row['n_clusters_observed']):.1f}",
        f"{float(row['ari_stability']):.3f}",
        f"{float(row['ari_generalizability']):.3f}",
    ]


def _pipeline_name(label: str) -> str:
    """A pipeline as the tables name it: "LSI, 50 components"."""
    from .figures._cusanovich_results import _short_label

    return _short_label(label)


def _at(rows: pd.DataFrame, sweep_value: float) -> pd.Series:
    return rows.loc[
        np.isclose(rows["sweep_value"].astype(float), float(sweep_value))
    ].iloc[0]


def selection_granularity_table(
    inputs: CusanovichInputs,
    *,
    source_option: tuple[str, dict[str, list[Any]]],
    caption: str = CUSANOVICH_SELECTION_CAPTION,
) -> DisplayTable:
    """S16 Table: CARVE's selection, the best pipeline there, the pooled
    resolution nearest the reference cluster count, and the source's own
    pipeline (source_option) at the resolution nearest that count."""
    carve = inputs.carve
    target = int(np.unique(inputs.y).size)
    selected, _, _, _ = carve._select_row(
        measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
    )
    method_id = selected["method_id"]
    pooled = carve.estimator_results_
    pooled = pooled[pooled["method_id"] == method_id]
    source = map_pipeline(carve, source_option)
    table = _per_pipeline_table(carve)
    source_rows = table[
        (table["method_id"] == method_id) & (table["pipeline"] == source.label)
    ]
    best = inputs.best_pipeline_row
    rows = {
        "CARVE selection, pooled over pipelines": selected,
        "Best pipeline at CARVE-selected resolution: "
        f"{_pipeline_name(best['pipeline'])}": best,
        f"Pooled, nearest Cusanovich et al.'s {target} clusters": _at(
            pooled, inputs.operating_point[0]
        ),
        f"Cusanovich et al.'s pipeline ({_pipeline_name(source.label)}), "
        f"nearest to {target} clusters": _nearest(source_rows, target),
    }
    cells = pd.DataFrame(
        [_granularity_cells(row) for row in rows.values()],
        index=pd.Index(list(rows), name="Configuration"),
        columns=_GRANULARITY_COLUMNS,
    )
    return DisplayTable(caption=caption, cells=cells)


def pipeline_table(
    inputs: CusanovichInputs,
    *,
    options: tuple[tuple[str, dict[str, list[Any]]], ...],
    caption: str = CUSANOVICH_PIPELINE_CAPTION,
) -> DisplayTable:
    """S17 Table: each pipeline of options, in that order, at CARVE's
    selected resolution and at its own resolution nearest the reference
    cluster count, with the number of resamples it received."""
    carve = inputs.carve
    target = int(np.unique(inputs.y).size)
    selected, _, _, _ = carve._select_row(
        measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
    )
    table = _per_pipeline_table(carve)
    at_method = table[table["method_id"] == selected["method_id"]]
    index, rows = [], []
    for option in options:
        label = map_pipeline(carve, option).label
        own = at_method[at_method["pipeline"] == label]
        block = f"{_pipeline_name(label)}, B = {int(own['n_resamples'].iloc[0])}"
        index += [(block, "Selected resolution"), (block, f"Nearest {target} clusters")]
        rows += [
            _granularity_cells(_at(own, selected["sweep_value"])),
            _granularity_cells(_nearest(own, target)),
        ]
    cells = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["Pipeline", "Granularity"]),
        columns=_GRANULARITY_COLUMNS,
    )
    return DisplayTable(caption=caption, cells=cells)
