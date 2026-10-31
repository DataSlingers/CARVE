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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from scipy.cluster.hierarchy import leaves_list, linkage

from .._panels import _order_by_reference, compact_setting
from .._theme import (
    FOREGROUND_COLOR,
    PRINT_FONT_SIZES,
    PRINT_MAX_HEIGHT_IN,
    PRINT_WIDTH_IN,
    SEQUENTIAL_CMAP_NAME,
    print_theme_context,
    save_figure,
)
from ._case_study import CompositeInputs, carve_cluster_numbers, composite_color_maps
from ._paths import CASE_STUDY_DIR, figure_path

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

# Layout in inches at the printed size. A row fits one 8-point label. The
# label bands fit the longest population name at 45 degrees and the longest
# marker name (CD235ab) upright. With 14 populations and partitions of 10
# and 7 clusters it is 8.62 inches tall, under PRINT_MAX_HEIGHT_IN; fewer
# clusters make it shorter.
ROW_HEIGHT_IN = 0.125
TOP_IN = 0.25
TITLE_GAP_IN = 0.3
POPULATION_LABEL_IN = 1.15
PANEL_GAP_IN = 0.25
BLOCK_GAP_IN = 0.06
MARKER_LABEL_IN = 0.55
LEFT_IN = 1.75
RIGHT_IN = 0.55
SWATCH_IN = 0.12
SWATCH_GAP_IN = 0.04
COLORBAR_GAP_IN = 0.12
COLORBAR_IN = 0.12
LETTER_OFFSET_PT = -4.0
# A count is printed in white on cells above this fraction, in the foreground
# color below it.
TEXT_ON_DARK = 0.6


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


def _carve_title(inputs: CompositeInputs) -> str:
    """Panel A's title for CARVE: the selection and the selected estimator."""
    row, _, k, _ = inputs.carve._select_row(
        measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
    )
    rule = RULE_NAMES.get(inputs.rule, inputs.rule)
    # A resolution run's k is an outcome several resolutions can share, so
    # the resolution names the partition.
    sweep = getattr(inputs.carve, "sweep_", None)
    at = (
        f"resolution {float(row['sweep_value']):g}, "
        if sweep is not None and sweep.param == "resolution"
        else ""
    )
    label = compact_setting(str(row["method_label"]))
    return f"CARVE ({inputs.measure}, {rule}): {label}, {at}k={k}"


def _comparison_title(inputs: CompositeInputs) -> str:
    """Panel A's title for the comparison: the CVI and its selected estimator."""
    best = inputs.best_df
    metric = inputs.comparison_name.lower()
    model = best.loc[best["metric"] == metric, "model"].iloc[0]
    at = (
        ""
        if inputs.comparison_resolution is None
        else f"resolution {inputs.comparison_resolution:g}, "
    )
    model = compact_setting(str(model))
    return f"{inputs.comparison_name}: {model}, {at}k={inputs.comparison_k}"


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


def _display_name(population: str) -> str:
    """A population name as printed, with HDCytoData's underscores as spaces."""
    return population.replace("_", " ")


def _scale_markers(X: np.ndarray) -> np.ndarray:
    """Each column mapped to [0, 1] by its 1st and 99th percentiles, clipped.

    The scaling of the CyTOF workflow of Nowicka et al. (2017) and CATALYST.
    A positive affine map of a column leaves the result unchanged, so the
    loader's RobustScaler output gives the values its arcsinh input would. A
    column whose two percentiles are equal has no contrast and maps to 0.
    """
    X = np.asarray(X, dtype=np.float64)
    low, high = np.percentile(X, [1, 99], axis=0)
    span = high - low
    scaled = np.zeros_like(X)
    varying = span > 0
    scaled[:, varying] = np.clip(
        (X[:, varying] - low[varying]) / span[varying], 0.0, 1.0
    )
    return scaled


def marker_medians(
    X: np.ndarray,
    groups: np.ndarray,
    *,
    markers: Sequence[str],
    group_order: Sequence,
) -> pd.DataFrame:
    """Median scaled expression per group, groups by markers.

    Scaling (_scale_markers) is over every cell in X before grouping, so all
    groups share one scale per marker. group_order must list exactly the
    groups present.
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    if X.shape[1] != len(markers):
        raise ValueError(
            f"X has {X.shape[1]} columns but {len(markers)} markers were named."
        )
    present = set(groups.tolist())
    if present != set(group_order):
        raise ValueError(
            "group_order must list exactly the groups present; "
            f"missing {sorted(map(str, present - set(group_order)))}, "
            f"not present {sorted(map(str, set(group_order) - present))}."
        )
    scaled = pd.DataFrame(_scale_markers(X), columns=list(markers))
    medians = scaled.groupby(groups).median().reindex(index=list(group_order))
    medians.index.name = "group"
    medians.columns.name = "marker"
    return medians


def marker_order(medians: pd.DataFrame) -> list[str]:
    """Markers in the leaf order of an average-linkage tree over their profiles.

    Each marker's profile is its column of medians. Euclidean distance with
    optimal leaf ordering puts markers expressed together next to each other,
    and the order depends only on the values.
    """
    tree = linkage(
        medians.to_numpy().T,
        method="average",
        metric="euclidean",
        optimal_ordering=True,
    )
    return [str(medians.columns[i]) for i in leaves_list(tree)]


def _marker_blocks(
    inputs: CompositeInputs,
    markers: Sequence[str],
    partitions: tuple[_Partition, _Partition],
) -> list[tuple[str, pd.DataFrame, list[str]]]:
    """Panel B's blocks as (label, medians, row colors), rows named as drawn.

    The gated block lists the populations present, in lineage order. Each
    partition's block keeps panel A's row order and names. Every block's
    markers follow the gated block's marker_order.
    """
    y = np.asarray(inputs.y)
    true_cmap, _, _ = composite_color_maps(inputs)
    present = set(y.tolist())
    populations = [p for p in POPULATION_ORDER if p in present]
    gated = marker_medians(inputs.X, y, markers=markers, group_order=populations)
    columns = marker_order(gated)
    blocks = [
        (
            "Gated",
            gated[columns].set_axis(
                pd.Index([_display_name(p) for p in populations], name="row"),
                axis=0,
            ),
            [true_cmap[p] for p in populations],
        )
    ]
    for partition in partitions:
        medians = marker_medians(
            inputs.X,
            partition.labels,
            markers=markers,
            group_order=list(partition.counts.index),
        )
        blocks.append(
            (
                partition.block,
                medians[columns].set_axis(
                    pd.Index(partition.names, name="row"), axis=0
                ),
                partition.colors,
            )
        )
    return blocks


def marker_table(inputs: CompositeInputs, markers: Sequence[str]) -> pd.DataFrame:
    """Panel B's medians, the gated, CARVE and comparison blocks stacked.

    Indexed by (block, row), with rows named as the figure names them and
    markers in the figure's column order.
    """
    blocks = _marker_blocks(inputs, markers, _partitions(inputs))
    return pd.concat(
        {label: frame for label, frame, _ in blocks}, names=["block", "row"]
    )


def _hide_spines(ax: Axes) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _heatmap(ax: Axes, values: np.ndarray) -> AxesImage:
    """values on [0, 1] in the sequential map, one cell per entry, no grid."""
    image = ax.imshow(
        values,
        cmap=SEQUENTIAL_CMAP_NAME,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    ax.grid(False)
    ax.set_yticks([])
    _hide_spines(ax)
    return image


def _swatches(ax: Axes, colors: Sequence[str], names: Sequence[str]) -> None:
    """A one-column strip of row colors, carrying the row names as y ticks."""
    ax.imshow(
        np.array([to_rgba(color) for color in colors])[:, None, :],
        aspect="auto",
        interpolation="nearest",
    )
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks(range(len(names)), labels=list(names))
    ax.tick_params(left=False)
    _hide_spines(ax)


def _column_labels(
    ax: Axes, names: Sequence[str], *, show: bool, rotation: float
) -> None:
    """Column names under a heatmap, or none on a heatmap with another below.

    rotation_mode="anchor" with ha="right" ends each label at its tick at
    any angle. Matplotlib's "xtick" mode does the same but needs 3.11, above
    the 3.9.4 floor.
    """
    if show:
        ax.set_xticks(
            range(len(names)),
            labels=list(names),
            rotation=rotation,
            ha="right",
            va="center" if rotation == 90 else "top",
            rotation_mode="anchor",
        )
    else:
        ax.set_xticks([])
    ax.tick_params(bottom=False)


def _letter(ax: Axes, letter: str) -> None:
    """A bold panel letter left of the axes, on its title's baseline."""
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(LETTER_OFFSET_PT, plt.rcParams["axes.titlepad"]),
        textcoords="offset points",
        ha="right",
        va="baseline",
        fontweight="bold",
        fontsize=PRINT_FONT_SIZES["panel_letter"],
    )


def _counts(ax: Axes, counts: np.ndarray, fractions: np.ndarray) -> None:
    """Each nonzero count in its cell, white on the darker cells."""
    for (i, j), count in np.ndenumerate(counts):
        if count:
            ax.text(
                j,
                i,
                f"{int(count)}",
                ha="center",
                va="center",
                fontsize=PRINT_FONT_SIZES["tick"],
                color="white" if fractions[i, j] > TEXT_ON_DARK else FOREGROUND_COLOR,
            )


def _row_axes(fig: Figure, grid: GridSpec, row: int, label: str) -> tuple[Axes, Axes]:
    """One block's swatch and heatmap axes, labeled so tests can find them."""
    swatch = fig.add_subplot(grid[row, 0])
    heat = fig.add_subplot(grid[row, 2])
    swatch.set_label(f"swatch:{label}")
    heat.set_label(label)
    return swatch, heat


def figure_levine_composition(
    inputs: CompositeInputs,
    markers: Sequence[str],
    *,
    save: bool = True,
    out_dir: Path | None = None,
    save_name: str = "levine_composition.png",
) -> Figure:
    """Draw panel A (cluster composition) and panel B (marker medians).

    markers names the columns of inputs.X, as load_levine32 returns them in
    meta["markers"]. Rows are named and colored as composition_tables and
    marker_table name them; the module docstring describes the panels.
    """
    carve, comparison = _partitions(inputs)
    blocks = _marker_blocks(inputs, markers, (carve, comparison))
    sized = {
        partition.block: [
            f"{name} (n={int(n)})"
            for name, n in zip(partition.names, partition.counts.sum(axis=1))
        ]
        for partition in (carve, comparison)
    }

    heights = [
        len(carve.counts) * ROW_HEIGHT_IN,
        TITLE_GAP_IN,
        len(comparison.counts) * ROW_HEIGHT_IN,
        POPULATION_LABEL_IN + PANEL_GAP_IN,
        len(blocks[0][1]) * ROW_HEIGHT_IN,
        BLOCK_GAP_IN,
        len(carve.counts) * ROW_HEIGHT_IN,
        BLOCK_GAP_IN,
        len(comparison.counts) * ROW_HEIGHT_IN,
        MARKER_LABEL_IN,
    ]
    height = TOP_IN + sum(heights)
    if height > PRINT_MAX_HEIGHT_IN:
        warnings.warn(
            f"The figure is {height:.2f} inches tall for {len(carve.counts)} CARVE "
            f"and {len(comparison.counts)} {comparison.block} clusters, over the "
            f"{PRINT_MAX_HEIGHT_IN} inch print limit. It will not fit the printed "
            "page at this size.",
            UserWarning,
            stacklevel=2,
        )
    heat_in = (
        PRINT_WIDTH_IN
        - LEFT_IN
        - RIGHT_IN
        - SWATCH_IN
        - SWATCH_GAP_IN
        - COLORBAR_GAP_IN
        - COLORBAR_IN
    )
    widths = [SWATCH_IN, SWATCH_GAP_IN, heat_in, COLORBAR_GAP_IN, COLORBAR_IN]
    populations = [_display_name(p) for p in POPULATION_ORDER]

    with print_theme_context():
        fig = plt.figure(figsize=(PRINT_WIDTH_IN, height))
        grid = fig.add_gridspec(
            len(heights),
            len(widths),
            height_ratios=heights,
            width_ratios=widths,
            left=LEFT_IN / PRINT_WIDTH_IN,
            right=1 - RIGHT_IN / PRINT_WIDTH_IN,
            top=1 - TOP_IN / height,
            bottom=0.0,
            hspace=0.0,
            wspace=0.0,
        )

        # Panel A: one heatmap per partition, population names under the lower.
        images = []
        for row, partition, key in ((0, carve, "carve"), (2, comparison, "comparison")):
            swatch, heat = _row_axes(fig, grid, row, f"composition:{key}")
            counts = partition.counts.to_numpy()
            fractions = counts / counts.sum(axis=1, keepdims=True)
            images.append(_heatmap(heat, fractions))
            _counts(heat, counts, fractions)
            _swatches(swatch, partition.colors, sized[partition.block])
            _column_labels(heat, populations, show=key == "comparison", rotation=45)
            heat.set_title(partition.title, loc="left")
            if key == "carve":
                _letter(swatch, "A")
        fig.colorbar(
            images[0], cax=fig.add_subplot(grid[0:3, 4]), label="Fraction of cluster"
        )

        # Panel B: the gated populations, then both partitions; markers upright.
        for row, (label, frame, colors), key in zip(
            (4, 6, 8), blocks, ("gated", "carve", "comparison")
        ):
            swatch, heat = _row_axes(fig, grid, row, f"markers:{key}")
            images.append(_heatmap(heat, frame.to_numpy()))
            names = list(frame.index) if key == "gated" else sized[label]
            _swatches(swatch, colors, names)
            swatch.set_ylabel(label)
            _column_labels(
                heat, list(frame.columns), show=key == "comparison", rotation=90
            )
            if key == "gated":
                heat.set_title("Median marker expression", loc="left")
                _letter(swatch, "B")
        fig.colorbar(
            images[-1],
            cax=fig.add_subplot(grid[4:9, 4]),
            label="Scaled median expression",
        )

        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
