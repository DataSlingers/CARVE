"""The Cusanovich mouse sci-ATAC case-study figure.

Four panels over CusanovichInputs. (A) CARVE's consensus labels and (B) the
source's clusters, both on one map, the study's map_option fit on every cell
(a t-SNE at perplexity 100), so the two partitions line up whichever
pipeline CARVE rates best. (C) CARVE's stability
and generalizability over resolution, pooled over pipelines, with the selected
resolution and the source's operating point marked, and the mean observed
cluster count on a secondary axis. (D) The same two criteria per pipeline at
the selected configuration, on C's x axis. A and B share one color map, so a
CARVE cluster takes the color of the source cluster it best matches.
_theme.CLUSTER_PALETTE holds 64 distinct colors, so each of the 30 source
clusters, and each CARVE cluster left unmatched past them, has its own until 64.
The source clusters CARVE matches take tab10 and then the palette's most
distinct mid-lightness entries (aligned_color_maps' matched_first), so A's
fewer clusters are not left on whichever dark entries their matches had.

The figure is laid out at the PLOS page width under print_theme_context and
saved both as the PNG the manuscript includes and as Fig6.tif.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._cusanovich_compare import CusanovichInputs
from .._panels import (
    aligned_color_maps,
    axis_arrows,
    carve_lines,
    panel_letter,
    pipeline_lines,
    scatter_clusters,
)
from .._theme import (
    PRINT_FONT_SIZES,
    PRINT_WIDTH_IN,
    metric_color,
    print_axes_text,
    print_free_text,
    print_theme_context,
    save_figure,
    save_plos_tiff,
    thin_lines,
)
from ._paths import CASE_STUDY_DIR, figure_path, plos_figure_path

# The manuscript's figure number, which names the PLOS file.
PLOS_FIGURE_NUMBER = 6

# MARKER_SIZE is the marker area for the 15-inch screen layout the
# notebook's reference scatter shares; at the print width it scales with the
# area. C's curves (carve_lines, 1.8-point lines) and D's (CARVE's own,
# 2-point) are drawn at a fraction of their screen weight.
MARKER_SIZE = 8.0
_SCREEN_WIDTH_IN = 15.0
_PRINT_HEIGHT_IN = 6.9
_PRINT_MARKER_SCALE = (PRINT_WIDTH_IN / _SCREEN_WIDTH_IN) ** 2
_CARVE_LINES_SCALE = 0.75
_PIPELINE_LINES_SCALE = 0.6
# scatter_clusters outlines each marker at 0.3 points, which on 81,173 small
# markers would darken every cluster; the outline thins with the figure.
_PRINT_OUTLINE_WIDTH = 0.3 * PRINT_WIDTH_IN / _SCREEN_WIDTH_IN

# At the print width more than this many resolution labels run into each
# other, so every other grid value is labeled; every value keeps its tick.
_MAX_TICK_LABELS = 8

# The axis arrows sit in the maps' lower-left corner, which the t-SNE fills;
# the axes are widened by this fraction of their range to leave it free.
_ARROW_CORNER = 0.12

# Legend entries as the printed figure names them: CARVE's estimator and
# pipeline reprs run past their panel at the print width.
_SHORT_LABELS: tuple[tuple[str, str], ...] = (
    (r"(\w+)Clustering, n_neighbors=(\d+)", r"\1, \2 neighbors"),
    (r"identity \| ", ""),
    (r"LSI\(n_components=(\d+)\)", r"LSI, \1 components"),
    (r"TSNE\(perplexity=(\d+)\)", r"t-SNE, perplexity \1"),
)


def _short_label(text: str) -> str:
    """A legend entry with CARVE's estimator and pipeline reprs shortened."""
    for pattern, replacement in _SHORT_LABELS:
        text = re.sub(pattern, replacement, text)
    return text


def _shorten_legend(ax: Axes) -> None:
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_text(_short_label(text.get_text()))


def resolution_tick_labels(grid: np.ndarray) -> list[str]:
    """The resolution axis labels: every grid value, or every other past eight."""
    step = 1 if len(grid) <= _MAX_TICK_LABELS else 2
    return [
        f"{value:g}" if index % step == 0 else "" for index, value in enumerate(grid)
    ]


def _room_for_legend(ax: Axes) -> None:
    """Extend the y axis down by a fifth of its range.

    C's and D's legends sit in their lower-left corner and take a larger
    share of the panel at the print width; the extension keeps them under
    the curves.
    """
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom - 0.2 * (top - bottom), top)


def _clear_corner(ax: Axes) -> None:
    """Widen the axes left and down, leaving the lower-left corner to the arrows."""
    for get, set_ in ((ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim)):
        low, high = get()
        set_(low - _ARROW_CORNER * (high - low), high)


SOURCE_TSNE_LABELS = ("t-SNE 1", "t-SNE 2")
SAVE_NAME = "cusanovich_results.png"


def _panel_a_title(inputs: CusanovichInputs, selected, n_clusters: int) -> str:
    """The selection, then the map A is drawn on."""
    estimator = str(selected["estimator"]).removesuffix("Clustering")
    return (
        f"CARVE: {estimator}, {inputs.carve.sweep_.param} "
        f"{float(selected['sweep_value']):g}, {n_clusters} clusters\n"
        f"on {inputs.map_name}"
    )


def _mark_source(ax: Axes, inputs: CusanovichInputs, method_id: str) -> None:
    """The source's operating point on C's curves.

    The operating point is placed on the pooled curves at the resolution where
    their mean observed cluster count, the count the secondary axis names,
    comes nearest the source's, and labeled with both counts.
    """
    results = inputs.carve.estimator_results_
    curve = results.loc[results["method_id"] == method_id]
    resolution, observed = inputs.operating_point
    n_source = int(np.unique(inputs.y).size)
    at_point = curve.loc[np.isclose(curve["sweep_value"].astype(float), resolution)]

    for index, measure in enumerate(("stability", "generalizability")):
        ax.plot(
            [resolution],
            [float(at_point[f"ari_{measure}"].iloc[0])],
            marker="D",
            markersize=8.0 * _CARVE_LINES_SCALE,
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=1.5 * _CARVE_LINES_SCALE,
            markeredgecolor=metric_color(f"ari_{measure}_1se"),
            zorder=3,
            label=(
                f"nearest the source's {n_source} clusters ({observed:.0f} observed)"
                if index == 0
                else "_nolegend_"
            ),
        )

    ax.legend(fontsize=PRINT_FONT_SIZES["legend"], frameon=False)


def _observed_cluster_axis(ax: Axes, inputs: CusanovichInputs, method_id: str) -> None:
    """A secondary x axis naming the mean observed cluster count per resolution."""
    results = inputs.carve.estimator_results_
    curve = results.loc[results["method_id"] == method_id].sort_values("sweep_value")
    top = ax.secondary_xaxis("top")
    top.set_xticks(
        curve["sweep_value"].to_numpy(dtype=float),
        labels=[f"{count:.0f}" for count in curve["n_clusters_observed"]],
    )
    top.tick_params(labelsize=PRINT_FONT_SIZES["tick"])
    top.set_xlabel("Mean observed clusters", fontsize=PRINT_FONT_SIZES["axis_label"])


def figure_cusanovich_results(
    inputs: CusanovichInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build the Cusanovich case-study figure from assembled inputs."""
    carve = inputs.carve
    selected, _, n_clusters, _ = carve._select_row(
        measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
    )
    method_id = str(selected["method_id"])
    source_cmap, carve_cmap = aligned_color_maps(
        inputs.y, inputs.carve_labels, matched_first=True
    )
    n_source = int(np.unique(inputs.y).size)

    marker_size = MARKER_SIZE * _PRINT_MARKER_SCALE
    with print_theme_context():
        fig, axes = plt.subplots(2, 2, figsize=(PRINT_WIDTH_IN, _PRINT_HEIGHT_IN))
        ax_a, ax_b, ax_c, ax_d = axes.flat
        ax_d.sharex(ax_c)

        scatter_clusters(
            ax_a,
            inputs.map_coordinates,
            inputs.carve_labels,
            color_map=carve_cmap,
            s=marker_size,
            linewidth=_PRINT_OUTLINE_WIDTH,
            title=_panel_a_title(inputs, selected, n_clusters),
        )
        _clear_corner(ax_a)
        axis_arrows(ax_a, inputs.map_axis_labels)
        scatter_clusters(
            ax_b,
            inputs.map_coordinates,
            inputs.y,
            color_map=source_cmap,
            s=marker_size,
            linewidth=_PRINT_OUTLINE_WIDTH,
            title=f"Cusanovich et al.: {n_source} clusters\non {inputs.map_name}",
        )
        _clear_corner(ax_b)
        axis_arrows(ax_b, inputs.map_axis_labels)

        carve_lines(
            ax_c,
            carve,
            rule=inputs.rule,
            not_two=inputs.not_two,
            title=(
                f"Stability & Generalizability over {carve.sweep_.param},\n"
                "pooled over pipelines"
            ),
        )
        # Before _mark_source, whose legend copies the curves' handles.
        thin_lines(ax_c, _CARVE_LINES_SCALE)
        # The resolution grid is log-spaced, so C draws it on a log axis ticked
        # at the grid values; D shares C's x axis and follows. The scale is set
        # before the secondary axis exists, because a later scale change resets
        # that axis's ticks when the figure is drawn.
        grid = np.unique(carve.estimator_results_["sweep_value"].to_numpy(dtype=float))
        ax_c.set_xscale("log")
        ax_c.set_xticks(grid, labels=resolution_tick_labels(grid))
        ax_c.minorticks_off()
        _mark_source(ax_c, inputs, method_id)
        _observed_cluster_axis(ax_c, inputs, method_id)

        pipeline_lines(
            ax_d,
            carve,
            method_id=method_id,
            rule=inputs.rule,
            not_two=inputs.not_two,
            title="Stability & Generalizability\nacross preprocessing pipelines",
        )
        # The legend's proxy lines take the print theme's widths already.
        thin_lines(ax_d, _PIPELINE_LINES_SCALE)
        for ax in (ax_c, ax_d):
            _shorten_legend(ax)
            _room_for_legend(ax)

        for ax in fig.get_axes():
            print_axes_text(ax)
            print_free_text(ax)
        for letter, ax in zip("ABCD", axes.flat):
            panel_letter(ax, letter, fontsize=PRINT_FONT_SIZES["panel_letter"])
        fig.tight_layout()

        if save:
            save_figure(
                fig, figure_path(SAVE_NAME, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
            save_plos_tiff(fig, plos_figure_path(PLOS_FIGURE_NUMBER, out_dir=out_dir))
    return fig
