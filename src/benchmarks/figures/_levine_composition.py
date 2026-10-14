"""The Levine composition figure: what each cluster is made of.

Panel A shows, for CARVE's selected partition and the CVI-selected one, how
each cluster's cells divide among the 14 gated populations. Panel B shows the
median expression of the 32 type markers for the gated populations and for
both partitions' clusters. Rows carry the colors S7 Fig's scatters use, and
CARVE's rows the cluster numbers S8 Fig prints.

The numbers are computed apart from the drawing. composition_tables and
marker_table return what the two panels draw, with rows named as drawn, so
the notebook prints the figure's own numbers. The figure is laid out at the
printed page width under print_theme_context, so its 8-point text is its
printed size.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._panels import _order_by_reference
from ._case_study import CompositeInputs, carve_labels_aligned, composite_color_maps

#: The 14 gated populations by lineage: T, NK, B, myeloid, progenitor.
#: Related populations sit next to each other, so a merge within a lineage
#: reads as a contiguous block. The names are the loader's, which are
#: HDCytoData's factor levels.
POPULATION_ORDER: tuple[str, ...] = (
    "CD4_T_cells",
    "CD8_T_cells",
    "CD16-_NK_cells",
    "CD16+_NK_cells",
    "Pro_B_cells",
    "Pre_B_cells",
    "Mature_B_cells",
    "Plasma_B_cells",
    "Monocytes",
    "pDCs",
    "Basophils",
    "CD34+CD38lo_HSCs",
    "CD34+CD38+CD123-_HSPCs",
    "CD34+CD38+CD123+_HSPCs",
)

RULE_NAMES: dict[str, str] = {"1se": "1-SE"}


@dataclass(frozen=True)
class _Partition:
    """One partition's rows, as both panels and the tables present them.

    counts is cluster_composition's table, rows by cluster id in display
    order; names and colors follow that order. Deciding order, names and
    colors in one place keeps panel A, panel B and the printed tables in
    agreement.
    """

    title: str
    block: str
    labels: np.ndarray
    counts: pd.DataFrame
    names: list[str]
    colors: list[str]


def cluster_composition(
    labels: np.ndarray, y: np.ndarray, *, order: Sequence[str]
) -> pd.DataFrame:
    """Cells per cluster and gated population.

    Rows are cluster ids in _order_by_reference order, which places each
    cluster at the position of the populations its cells come from. Columns
    are order, with zeros for populations a partition never reaches. Every
    population in y must be in order: one left out would drop its cells from
    the table and misstate every fraction in their rows.
    """
    labels = np.asarray(labels)
    y = np.asarray(y)
    missing = sorted(set(y.tolist()) - set(order))
    if missing:
        raise ValueError(f"y has populations that order does not list: {missing}.")
    rows = _order_by_reference(labels, y, order)
    counts = pd.crosstab(pd.Series(labels), pd.Series(y))
    counts = counts.reindex(index=rows, columns=list(order), fill_value=0)
    counts.index.name = "cluster"
    counts.columns.name = "population"
    return counts


def carve_cluster_numbers(inputs: CompositeInputs) -> dict[int, int]:
    """The number S8 Fig prints for each id in inputs.carve_labels.

    S8 Fig draws CARVE's own labels inside carve_labels_aligned and numbers
    each cluster by its id plus one. When the selected k differs from the
    number of reported populations, as on Levine, those ids are CARVE's own
    and not the aligned ids this figure colors by, so the two labelings are
    matched through the cells they share.
    """
    with carve_labels_aligned(inputs.carve, inputs.y):
        own = np.asarray(
            inputs.carve.get_labels(
                measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
            )
        )
    pairs = pd.crosstab(pd.Series(np.asarray(inputs.carve_labels)), pd.Series(own))
    numbers: dict[int, int] = {}
    for aligned_id, row in pairs.iterrows():
        shared = row[row > 0]
        if len(shared) != 1:
            break
        numbers[int(aligned_id)] = int(shared.index[0]) + 1
    else:
        if len(set(numbers.values())) == len(numbers):
            return numbers
    raise ValueError(
        "inputs.carve_labels and CARVE's own labels are not the same partition, "
        "so CARVE's rows cannot carry S8 Fig's cluster numbers."
    )


def _carve_title(inputs: CompositeInputs) -> str:
    """Panel A's title for CARVE: the selection and the selected estimator."""
    row, _, k, _ = inputs.carve._select_row(
        measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
    )
    rule = RULE_NAMES.get(inputs.rule, inputs.rule)
    return f"CARVE ({inputs.measure}, {rule}): {row['method_label']}, k={k}"


def _comparison_title(inputs: CompositeInputs) -> str:
    """Panel A's title for the comparison: the CVI and its selected estimator."""
    best = inputs.best_df
    metric = inputs.comparison_name.lower()
    model = best.loc[best["metric"] == metric, "model"].iloc[0]
    return f"{inputs.comparison_name}: {model}, k={inputs.comparison_k}"


def _partitions(inputs: CompositeInputs) -> tuple[_Partition, _Partition]:
    """CARVE's and the comparison's rows: counts, names, colors and titles."""
    y = np.asarray(inputs.y)
    _, carve_cmap, comparison_cmap = composite_color_maps(inputs)
    carve = cluster_composition(inputs.carve_labels, y, order=POPULATION_ORDER)
    comparison = cluster_composition(
        inputs.comparison_labels, y, order=POPULATION_ORDER
    )
    numbers = carve_cluster_numbers(inputs)
    return (
        _Partition(
            title=_carve_title(inputs),
            block="CARVE",
            labels=np.asarray(inputs.carve_labels),
            counts=carve,
            names=[f"C{numbers[int(c)]}" for c in carve.index],
            colors=[carve_cmap[int(c)] for c in carve.index],
        ),
        _Partition(
            title=_comparison_title(inputs),
            block=inputs.comparison_name,
            labels=np.asarray(inputs.comparison_labels),
            counts=comparison,
            names=[f"C{i}" for i in range(1, len(comparison) + 1)],
            colors=[comparison_cmap[int(c)] for c in comparison.index],
        ),
    )


def _named(partition: _Partition) -> pd.DataFrame:
    """A partition's counts with its rows named as the figure names them."""
    return partition.counts.set_axis(pd.Index(partition.names, name="cluster"), axis=0)


def composition_tables(
    inputs: CompositeInputs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel A's counts for CARVE and for the comparison, rows named as drawn."""
    carve, comparison = _partitions(inputs)
    return _named(carve), _named(comparison)
