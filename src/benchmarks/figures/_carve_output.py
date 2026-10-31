"""Fig 3 and its SI counterparts: CARVE's own output on a case study.

Six panels in three rows of two, and every one of them is a plot CARVE
already ships as a method on the fitted object: (A) stability ARI over k,
(B) generalizability ARI over k (over resolution, on a log axis, for a
resolution run), (C) the consensus matrix for the selected
configuration, (D) per-cluster stability as a violin plot, (E) the embedding
colored by consensus labels with dubious samples emphasized, and (F) the same
embedding with marker shape per cluster and score-encoded color. This module
only arranges the six axes into one gridspec, calls each method with ``ax=``
set, and saves the result once; it does not resolve config_id, recompute a
selection, or reimplement any encoding CARVE's own plotting functions already
provide.

All four case studies draw through _draw_six_panels: Klein (Fig 3), Levine
(S10 Fig), Cusanovich and hECA. hECA's figure shows a caller-chosen
resolution rather than the rule's selection, and its two scatters draw a
subsample, which CARVE's public scatter methods do not take; for those it
resolves the configuration through CARVE's _select_row, as the methods do,
and calls the plotting functions behind them.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import ListedColormap
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from carve._plotting import _get_annotation
from carve._plotting import plot_cluster_scatter as _plot_cluster_scatter
from carve._plotting import plot_diagnostic_scatter as _plot_diagnostic_scatter

from .._cusanovich_compare import CusanovichInputs
from .._panels import PANEL_LETTER_X, aligned_color_maps, compact_setting, panel_letter
from .._theme import (
    ESTIMATOR_CMAP_NAME,
    FOREGROUND_COLOR,
    PRINT_FONT_SIZES,
    PRINT_WIDTH_IN,
    REFERENCE_LINEWIDTH,
    cluster_cmap,
    print_axes_text,
    print_theme_context,
    save_figure,
    save_plos_tiff,
    thin_lines,
)
from ._case_study import (
    CompositeInputs,
    _swept_resolutions,
    carve_cluster_numbers,
    carve_labels_aligned,
    carve_order_cmap,
    carve_output_cmap,
    short_estimator_names,
)
from ._cusanovich_results import _PRINT_MARKER_SCALE as FIG6_MARKER_SCALE
from ._cusanovich_results import _PRINT_OUTLINE_WIDTH as FIG6_OUTLINE_WIDTH
from ._cusanovich_results import MARKER_SIZE as FIG6_MARKER_SIZE
from ._heca_carve import AXIS_LABELS, resolution_axis
from ._paths import CASE_STUDY_DIR, figure_path, plos_figure_path

# Panel A is always the stability overview and panel B the
# generalizability overview; both use the caller's rule and not_two. Panels
# C to F show the selection under CompositeInputs.measure: generalizability
# for Klein (the manuscript's headline k=4), stability for Levine.
_MEASURE_STABILITY = "stability"
_MEASURE_GENERALIZABILITY = "generalizability"


# Every figure here is laid out at PRINT_WIDTH_IN, where a row of two panels
# is PRINT_ROW_HEIGHT_IN tall. At that width titles and axis labels take a
# larger share of each row than on screen, so the scatter row is given
# about 1.55 rows (Fig 3; 1.6 for the SI figures) to keep the embeddings
# from flattening, and the figure stays under PRINT_MAX_HEIGHT_IN.
_SCREEN_WIDTH_IN = 13.0
PRINT_ROW_HEIGHT_IN = 2.35

# Scatter marker sizes are given for the screen-width figure. Marker area is
# in points squared, so at the print width it scales with the width ratio
# squared, which keeps each marker's size relative to its panel.
_PRINT_MARKER_SCALE = (PRINT_WIDTH_IN / _SCREEN_WIDTH_IN) ** 2

# CARVE draws its metric curves with 2-point lines, 6-point markers and
# 4-point caps, which suit the screen width; at the print width they are
# drawn at this fraction.
_PRINT_CURVE_SCALE = 0.6

# At the print width, D's cluster numbers run together past this many
# clusters if set level.
_MAX_LEVEL_CLUSTER_NUMBERS = 15

# CARVE's public scatter methods resolve these names for the gini source;
# a subsample's scatters call the functions behind them and pass the same.
_GINI_SCORES_NAME = "Gini Stability"

# CARVE's legend entry for the rule's line, such as "Selected resolution
# (1-SE rule): 0.01". When C to F show a value chosen from the curves, the
# rule's line is renamed after the rule alone, "1-SE rule: 0.01".
_RULE_LINE_LABEL = re.compile(r"Selected \S+ \((?P<rule>[^)]+ rule)\): (?P<value>.+)")


def _print_curve_legend(
    ax: Axes, *, estimators: bool | None = None, title: str | None = "Estimators"
) -> Legend:
    """The estimator legend CARVE draws, rebuilt from the thinned curves.

    A legend copies its handles when it is made, so it is drawn after the
    curves are thinned, in CARVE's place, with short names. At the
    print width the legend takes a larger share of the panel, so the y axis
    is extended downward by a fifth of its range to keep it under the
    curves; _clear_legend extends it further, once the layout is final,
    where the curves above the legend dip lower than that leaves room for.

    ``estimators`` keeps only the estimator curves' entries (True) or only
    the reference lines' (False); None keeps every entry.
    """
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom - 0.2 * (top - bottom), top)
    handles, labels = ax.get_legend_handles_labels()
    if estimators is not None:
        entries = [
            (handle, label)
            for handle, label in zip(handles, labels, strict=True)
            if isinstance(handle, ErrorbarContainer) == estimators
        ]
        handles = [handle for handle, _ in entries]
        labels = [label for _, label in entries]
    # The print theme's frameless legend, as in the other figures' curves.
    return ax.legend(
        handles,
        [short_estimator_names(label) for label in labels],
        loc="lower left",
        title=title,
    )


# Space kept between a curve legend's top and the lowest curve point above
# it, as a fraction of the axes height.
_LEGEND_CLEARANCE = 0.02


def _clear_legend(ax: Axes, legend: Legend) -> None:
    """Lower the y axis until no curve point or error bar falls inside the legend.

    The legend is anchored to the axes' lower-left corner, so its box in
    axes coordinates does not move with the y limits, only with the axes'
    size, which is why this runs after the last tight_layout. The lowest
    point drawn in data coordinates within its horizontal span sets how far
    the axis must extend for that point to sit above the box.
    """
    box = legend.get_window_extent().transformed(ax.transAxes.inverted())
    points = [
        np.column_stack([line.get_xdata(), line.get_ydata()])
        for line in ax.get_lines()
        if line.get_transform() == ax.transData
    ]
    points += [
        np.asarray(segment)
        for bars in ax.collections
        if isinstance(bars, LineCollection)
        for segment in bars.get_segments()
    ]
    if not points:
        return
    xy = np.vstack(points).astype(float)
    x_axes = ax.transAxes.inverted().transform(ax.transData.transform(xy))[:, 0]
    beneath = (x_axes >= box.x0) & (x_axes <= box.x1)
    if not beneath.any():
        return
    lowest = float(np.nanmin(xy[beneath, 1]))
    bottom, top = ax.get_ylim()
    needed = box.y1 + _LEGEND_CLEARANCE
    if (lowest - bottom) / (top - bottom) < needed:
        ax.set_ylim((lowest - needed * top) / (1.0 - needed), top)


# Space between neighboring panels' text, across rows and between columns,
# in units of the print theme's 8-point font: 0.2 inches.
_PANEL_PAD = 1.8

# C's colorbar, in the matrix's axes coordinates: beside the matrix, a
# twentieth of its width away and as wide, as tall as the matrix.
_CONSENSUS_COLORBAR_BOUNDS = (1.05, 0.0, 0.05, 1.0)

# CARVE appends its cluster band above the matrix at this fraction of the
# matrix's height.
_CONSENSUS_BAND_FRACTION = 0.04


def _consensus_colorbar(ax: Axes) -> Colorbar:
    """C's colorbar, drawn as an inset of the matrix axes.

    An inset follows the matrix wherever its square aspect places it, and
    the layout counts the colorbar's labels as part of C, so they stay in
    C's column.
    """
    cax = ax.inset_axes(_CONSENSUS_COLORBAR_BOUNDS)
    return ax.figure.colorbar(ax.images[0], cax=cax, label="Consensus")


def _title_above_band(ax: Axes) -> None:
    """Set C's title the theme's title pad above the cluster band.

    CARVE pads the title a fixed 25 points to clear the band, which at the
    print width lifts it a quarter inch above the band, close to A's axis
    label, and the shared row spacing then leaves every row that much
    further apart. The band's height is read from the matrix's current
    place in the layout.
    """
    fig = ax.figure
    box = ax.get_axes_locator()(ax, fig.canvas.get_renderer())
    band = _CONSENSUS_BAND_FRACTION * box.height * fig.get_figheight() * 72
    ax.set_title(
        ax.get_title(),
        pad=band + plt.rcParams["axes.titlepad"],
        fontsize=ax.title.get_fontsize(),
    )


# The scatter row's titles sit this far under the row above, in inches, a
# twentieth of an inch closer than the other rows. The other rows' titles
# sit on a frame; the scatters have none, and their points start a margin
# below the title, so at the same spacing the last gap reads as the widest.
_SCATTER_ROW_GAP_IN = 0.15


def _close_up_scatter_row(
    fig: Figure, above: Sequence[Axes], scatters: Sequence[Axes]
) -> None:
    """Raise the scatters' tops until their titles sit _SCATTER_ROW_GAP_IN under C and D.

    tight_layout spaces every row alike, by the boundary that needs the most
    room. Over a resolution sweep that is the one under A and B, whose tick
    labels are set vertically, and the room it needs is left over between
    C-D and E-F. The scatters grow upward, so their bottoms, F's colorbar and
    the figure size stay as laid out.
    """
    fig.draw_without_rendering()
    to_inches = fig.dpi_scale_trans.inverted()
    bottom = min(ax.get_tightbbox().transformed(to_inches).y0 for ax in above)
    top = max(ax.get_tightbbox().transformed(to_inches).y1 for ax in scatters)
    rise = (bottom - _SCATTER_ROW_GAP_IN - top) / fig.get_figheight()
    for ax in scatters:
        box = ax.get_position(original=True)
        ax.set_position((box.x0, box.y0, box.width, box.height + rise))
        # set_position takes an axes out of the layout; it stays in.
        ax.set_in_layout(True)


def _six_panel_grid(scatter_row_height: float) -> tuple[Figure, tuple[Axes, ...]]:
    """The figure and its six axes, A to F, in three rows of two."""
    # No hspace/wspace: setting them makes the tight_layout() call at the
    # end of each CARVE plotting method warn.
    fig = plt.figure(
        figsize=(PRINT_WIDTH_IN, PRINT_ROW_HEIGHT_IN * (2 + scatter_row_height))
    )
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, scatter_row_height))
    axes = tuple(fig.add_subplot(gs[row, col]) for row in range(3) for col in range(2))
    return fig, axes


def _subsample_scatters(
    carve: Any,
    ax_e: Axes,
    ax_f: Axes,
    *,
    X: np.ndarray,
    embedding: np.ndarray,
    rows: np.ndarray,
    selection: dict[str, Any],
    palette: ListedColormap,
    size_range: tuple[float, float],
    marker_size: float,
    marker_linewidth: float,
    axis_labels: Sequence[str],
) -> None:
    """Panels E and F over ``rows``, as CARVE's public scatter methods draw them.

    The methods take coordinates for every cell, so this resolves what they
    would (the configuration, its labels and per-sample scores, and the
    configuration note) and passes the subsample's rows to the plotting
    functions behind them.
    """
    row, config_id, selected_k, pinned = carve._select_row(**selection)
    labels = np.asarray(carve.get_labels(**selection))
    scores = carve._sample_scores("gini", config_id)
    note = _get_annotation(
        measure=selection["measure"],
        rule=selection["rule"],
        estimator_results=carve.estimator_results_,
        row=row,
        selected_k=selected_k,
        pinned=pinned,
        tight_layout=True,
    )
    scatter_args = (np.asarray(X)[rows], labels[rows], scores[rows])
    _plot_cluster_scatter(
        *scatter_args,
        embedding=embedding,
        ax=ax_e,
        palette=palette,
        size_range=size_range,
        legend=False,
        annotation=note,
        title="Consensus labels",
        scores_name=_GINI_SCORES_NAME,
        xlabel=axis_labels[0],
        ylabel=axis_labels[1],
        show=False,
    )
    _plot_diagnostic_scatter(
        *scatter_args,
        embedding=embedding,
        ax=ax_f,
        marker_size=marker_size,
        marker_linewidth=marker_linewidth,
        legend=False,
        annotation=note,
        title="Consensus assignment (diagnostic)",
        scores_name=_GINI_SCORES_NAME,
        xlabel=axis_labels[0],
        ylabel=axis_labels[1],
        show=False,
    )


def _draw_six_panels(
    carve: Any,
    *,
    X: np.ndarray,
    embedding: np.ndarray,
    palette: ListedColormap,
    measure: str,
    rule: str,
    not_two: bool,
    marker_size: float,
    axis_labels: Sequence[str],
    save_name: str,
    marker_linewidth: float = 0.2,
    cluster_outline: float | None = None,
    scatter_row_height: float = 1.0,
    violin_dots: bool = True,
    violin_note: bool = True,
    sweep_value: float | None = None,
    rows: np.ndarray | None = None,
    plos_number: int | None = None,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Draw the six-panel CARVE diagnostic figure.

    ``palette`` colors the clusters of C, D and E, entry j for the cluster
    CARVE numbers j + 1. Panels A and B color one line per estimator, not
    per cluster, so they take the estimator map instead.

    ``marker_size`` scales panel F's fixed marker size directly (that
    parameter's name on ``plot_diagnostic_scatter`` matches this one
    exactly) and panel E's ``size_range`` proportionally, so the case
    studies' scatters differ only in dot size, not in the score encoding
    itself. ``marker_linewidth`` is panel F's outline width, under the same
    name. ``cluster_outline`` is panel E's; None keeps the 0.2 points
    CARVE's plot_cluster_scatter draws.

    Panels E and F carry no legend: D numbers the clusters in E's colors,
    and F keeps its colorbar.

    ``scatter_row_height`` is the height of the E-F row relative to each of
    the other two rows. The figure grows with it, so panels A to D keep
    their size. ``violin_dots`` draws CARVE's per-cell dots over D's
    violins; over tens of thousands of cells they merge into a solid band.
    ``violin_note`` draws CARVE's configuration note in D; past a dozen
    clusters it covers the lower violins, and C's title and the caption
    carry the same information. Past _MAX_LEVEL_CLUSTER_NUMBERS clusters,
    D's cluster numbers are set vertically so they do not run together.

    ``sweep_value`` makes C to F show the partition at that value of the
    swept parameter instead of the rule's selection, and marks it in A and
    B with a dotted line, "Selected after inspection", beside the rule's
    dashed one, which is then named after the rule alone. ``rows`` draws E and
    F over a subsample, one ``embedding`` row per entry; it must include
    every cluster, since CARVE's scatters color and shape clusters by their
    position among the labels drawn.

    The figure is laid out at PRINT_WIDTH_IN under print_theme_context, and
    the text CARVE's methods size for the screen is reset to the print
    sizes. ``plos_number`` names the PLOS figure file saved beside the PNG;
    an SI figure, which PLOS takes in any format, leaves it None.
    """
    selection = {"measure": measure, "rule": rule, "not_two": not_two}
    if sweep_value is not None:
        selection["sweep_value"] = sweep_value
    if rows is not None:
        labels = np.asarray(carve.get_labels(**selection))
        if not np.array_equal(np.unique(labels[rows]), np.unique(labels)):
            raise ValueError(
                "rows must include every cluster of the partition shown; CARVE's "
                "scatters color clusters by position among the labels drawn."
            )
    _, _, selected_k, _ = carve._select_row(**selection)
    marker_size = marker_size * _PRINT_MARKER_SCALE
    scatter_size_range = (marker_size * 0.75, marker_size * 3.0)
    # A resolution run's curves go on a log axis, with one tick per fitted
    # resolution.
    resolutions = _swept_resolutions(carve)
    over = "$k$" if resolutions is None else "resolution"
    consensus_title = f"Consensus matrix ($k={selected_k}$)"
    if getattr(carve, "consensus_anchors_", None) is not None:
        consensus_title = (
            f"Consensus matrix ($k={selected_k}$, "
            f"{carve.consensus_anchors_.size:,} anchor cells)"
        )

    with print_theme_context():
        fig, (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f) = _six_panel_grid(scatter_row_height)

        for ax, curve_measure, title in (
            (ax_a, _MEASURE_STABILITY, f"Stability ARI over {over}"),
            (ax_b, _MEASURE_GENERALIZABILITY, f"Generalizability ARI over {over}"),
        ):
            carve.plot_metric_over_n_clusters(
                measure=curve_measure,
                rule=rule,
                not_two=not_two,
                ax=ax,
                palette=ESTIMATOR_CMAP_NAME,
                title=title,
                legend=False,
                show=False,
            )
            if sweep_value is not None:
                for line in ax.get_lines():
                    match = _RULE_LINE_LABEL.fullmatch(str(line.get_label()))
                    if match:
                        line.set_label(f"{match['rule']}: {match['value']}")
                ax.axvline(
                    sweep_value,
                    color=FOREGROUND_COLOR,
                    linestyle=":",
                    linewidth=REFERENCE_LINEWIDTH,
                    label=f"Selected after inspection: {sweep_value:g}",
                    zorder=0,
                )
        if resolutions is not None:
            for ax in (ax_a, ax_b):
                ax.set_xlabel("Resolution")
                resolution_axis(ax, resolutions)
        for ax in (ax_a, ax_b):
            thin_lines(ax, _PRINT_CURVE_SCALE)
        # A and B draw the same estimators in the same colors, so one
        # legend on A names them for both. A's dashed line joins it when it
        # is the selection C to F show. Otherwise B names its own reference
        # lines: the selection on B's curve, or the rule's line and the
        # value selected after inspection, which under the estimators would
        # make one legend too tall for the panel.
        if sweep_value is None and measure == _MEASURE_STABILITY:
            legends = [(ax_a, _print_curve_legend(ax_a))]
        else:
            legends = [
                (ax_a, _print_curve_legend(ax_a, estimators=True)),
                (ax_b, _print_curve_legend(ax_b, estimators=False, title=None)),
            ]
        carve.plot_consensus_matrix(
            **selection,
            ax=ax_c,
            palette=palette,
            colorbar=False,
            title=consensus_title,
            show=False,
        )
        consensus_colorbar = _consensus_colorbar(ax_c)
        carve.plot_cluster_violin(
            source="gini",
            **selection,
            ax=ax_d,
            palette=palette,
            stripplot=violin_dots,
            title="Per-cluster stability",
            annotation=violin_note,
            size=8.0 * _PRINT_MARKER_SCALE,
            show=False,
        )
        if selected_k > _MAX_LEVEL_CLUSTER_NUMBERS:
            ax_d.tick_params(axis="x", labelrotation=90)
        if rows is None:
            carve.plot_cluster_scatter(
                source="gini",
                **selection,
                X=X,
                embedding=embedding,
                ax=ax_e,
                palette=palette,
                size_range=scatter_size_range,
                legend=False,
                title="Consensus labels",
                xlabel=axis_labels[0],
                ylabel=axis_labels[1],
                show=False,
            )
            carve.plot_diagnostic_scatter(
                source="gini",
                **selection,
                X=X,
                embedding=embedding,
                ax=ax_f,
                marker_size=marker_size,
                marker_linewidth=marker_linewidth,
                legend=False,
                title="Consensus assignment (diagnostic)",
                xlabel=axis_labels[0],
                ylabel=axis_labels[1],
                show=False,
            )
        else:
            _subsample_scatters(
                carve,
                ax_e,
                ax_f,
                X=X,
                embedding=embedding,
                rows=rows,
                selection=selection,
                palette=palette,
                size_range=scatter_size_range,
                marker_size=marker_size,
                marker_linewidth=marker_linewidth,
                axis_labels=axis_labels,
            )
        # plot_cluster_scatter fixes E's outline, so it is set on the drawn
        # markers.
        if cluster_outline is not None:
            for points in ax_e.collections:
                points.set_linewidths(cluster_outline)

        # Colorbars are axes of their own, so every axes is reset, not
        # only the six panels.
        for ax in (*fig.get_axes(), consensus_colorbar.ax):
            print_axes_text(ax)
        for letter, ax in zip("ABD", (ax_a, ax_b, ax_d)):
            panel_letter(ax, letter, fontsize=PRINT_FONT_SIZES["panel_letter"])

        # CARVE writes a graph clusterer's full configuration into the curve
        # legend and D's note, where it runs past the figure. The note also
        # breaks before its resolution, which keeps its box narrow inside D.
        for text in fig.findobj(mtext.Text):
            short = compact_setting(text.get_text())
            if short != text.get_text():
                text.set_text(short.replace(" (resolution =", "\n(resolution ="))

        _title_above_band(ax_c)
        fig.tight_layout(h_pad=_PANEL_PAD, w_pad=_PANEL_PAD)
        # The layout resized the matrix, and with it the band.
        _title_above_band(ax_c)
        _close_up_scatter_row(fig, (ax_c, ax_d), (ax_e, ax_f))
        for ax, legend in legends:
            _clear_legend(ax, legend)
        # panel_letter places a letter relative to the axes it labels, and
        # raises it by a fraction of their height. C's square matrix sits
        # centered in its wider cell and lower than D's top, so once the
        # draw has sized it, C's letter goes as far left as A's and level
        # with D's. The taller scatter row lifted E's and F's letters above
        # their titles, so they go as far above their axes as D's goes.
        fig.draw_without_rendering()
        (d_letter,) = [text for text in ax_d.texts if text.get_text() == "D"]
        rise = (d_letter.get_position()[1] - 1.0) * ax_d.get_position().height
        above, matrix = ax_a.get_position(), ax_c.get_position()
        letter_x = above.x0 + PANEL_LETTER_X * above.width
        panel_letter(
            ax_c,
            "C",
            x=(letter_x - matrix.x0) / matrix.width,
            y=(ax_d.get_position().y1 + rise - matrix.y0) / matrix.height,
            fontsize=PRINT_FONT_SIZES["panel_letter"],
        )
        for letter, ax in (("E", ax_e), ("F", ax_f)):
            panel_letter(
                ax,
                letter,
                y=1.0 + rise / ax.get_position().height,
                fontsize=PRINT_FONT_SIZES["panel_letter"],
            )

        if save:
            save_figure(
                fig,
                figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir),
            )
            if plos_number is not None:
                save_plos_tiff(fig, plos_figure_path(plos_number, out_dir=out_dir))
    return fig


def carve_output_figure(
    inputs: CompositeInputs,
    *,
    marker_size: float,
    axis_labels: Sequence[str],
    save_name: str,
    marker_linewidth: float = 0.2,
    scatter_row_height: float = 1.0,
    plos_number: int | None = None,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Draw the six panels for a case study with a composite (Fig 3 and S10 Fig).

    C to E take the composite's colors (carve_output_cmap), and CARVE's own
    methods draw while its labels are aligned to the reported ones, so a
    cluster keeps its color and number across the composite, the
    composition figure and this one. The remaining arguments are
    _draw_six_panels'.
    """
    palette = carve_output_cmap(inputs)
    with carve_labels_aligned(inputs.carve, inputs.y):
        return _draw_six_panels(
            inputs.carve,
            X=inputs.X,
            embedding=inputs.Z,
            palette=palette,
            measure=inputs.measure,
            rule=inputs.rule,
            not_two=inputs.not_two,
            marker_size=marker_size,
            axis_labels=axis_labels,
            save_name=save_name,
            marker_linewidth=marker_linewidth,
            scatter_row_height=scatter_row_height,
            plos_number=plos_number,
            save=save,
            out_dir=out_dir,
        )


def figure_carve_output_klein(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build Fig 3: CARVE output on the Klein droplet scRNA-seq case study.

    Panels E and F carry no legend; the manuscript caption names their
    encodings. Their row is a fifth taller than the others, since panel F
    shares its cell with the colorbar.
    """
    return carve_output_figure(
        inputs,
        marker_size=20.0,
        axis_labels=("PC1", "PC2"),
        save_name="CARVE_output_klein.png",
        scatter_row_height=1.55,
        plos_number=3,
        save=save,
        out_dir=out_dir,
    )


def figure_carve_output_levine(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build S10 Fig: CARVE output on the Levine mass cytometry case study.

    Panel F's outlines are a quarter of CARVE's default width: over 5,000
    cells the default black outlines covered the color of the unstable
    cells. The E-F row is 1.6 times the height of the others.
    """
    return carve_output_figure(
        inputs,
        marker_size=8.0,
        axis_labels=("t-SNE 1", "t-SNE 2"),
        save_name="CARVE_output_levine.png",
        marker_linewidth=0.05,
        scatter_row_height=1.6,
        save=save,
        out_dir=out_dir,
    )


# Cusanovich's scatters draw all 81,173 cells on the map of Fig 6 (A, B) and
# hECA's 50,000, at Fig 6's printed marker area, and E at its outline width;
# F's outlines are Levine's, which keep the unstable cells' color visible.
# Thicker outlines on markers this small turn every cluster dark.
_ATLAS_MARKER_SIZE = FIG6_MARKER_SIZE * FIG6_MARKER_SCALE / _PRINT_MARKER_SCALE
_ATLAS_DIAGNOSTIC_OUTLINE = 0.05


def figure_carve_output_cusanovich(
    inputs: CusanovichInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build the Cusanovich counterpart of Fig 3: CARVE output on the atlas.

    The fit sweeps one Louvain setting with randomized preprocessing, so A
    and B draw one curve each, pooled over the four pipelines (Fig 6 (D)
    draws them per pipeline). C to F show the selection inputs names, C on
    the fit's anchor block. C to E color each cluster as Fig 6 (A) does,
    in the color of the source cluster it matches. E and F draw every cell
    on Fig 6's map; D leaves out CARVE's per-cell dots, which over 81,173
    cells merge into a solid band, and CARVE's configuration note, which
    covers the lower violins of 21 clusters.
    """
    _, matched = aligned_color_maps(inputs.y, inputs.carve_labels, matched_first=True)
    palette = carve_order_cmap(matched, carve_cluster_numbers(inputs))
    with carve_labels_aligned(inputs.carve, inputs.y):
        return _draw_six_panels(
            inputs.carve,
            X=inputs.X,
            embedding=inputs.map_coordinates,
            palette=palette,
            measure=inputs.measure,
            rule=inputs.rule,
            not_two=inputs.not_two,
            marker_size=_ATLAS_MARKER_SIZE,
            axis_labels=inputs.map_axis_labels,
            save_name="CARVE_output_cusanovich.png",
            marker_linewidth=_ATLAS_DIAGNOSTIC_OUTLINE,
            cluster_outline=FIG6_OUTLINE_WIDTH,
            scatter_row_height=1.6,
            violin_dots=False,
            violin_note=False,
            save=save,
            out_dir=out_dir,
        )


def figure_carve_output_heca(
    carve: Any,
    *,
    sweep_value: float,
    rows: np.ndarray,
    embedding: np.ndarray,
    X: np.ndarray | None = None,
    measure: str = "stability",
    rule: str = "1se",
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Build the hECA counterpart of Fig 3: CARVE's output at one resolution.

    Panels C to F show the partition at ``sweep_value``, a resolution the
    caller selects by inspecting the curves, rather than the rule's
    selection; A and B mark both, and B's legend names the first "Selected
    after inspection" and the second after the rule alone.
    ``measure`` and ``rule`` pick between the settings fitted at
    ``sweep_value``. C is the anchor block, the consensus an anchored fit
    stores. D's violins take every cell's score, without CARVE's per-cell
    dots, which at this size merge into a solid band, or its configuration
    note, which covers the lower violins. E and F draw
    ``rows``, a subsample, at ``embedding``, one coordinate pair per
    subsample row, with the markers of the Cusanovich figure. ``X``
    defaults to ``carve.X_``.

    Raises ValueError when ``rows`` misses a cluster: CARVE's scatters color
    and shape clusters by their position among the labels drawn, so a
    missing cluster would shift E and F away from C and D.
    """
    labels = carve.get_labels(measure=measure, rule=rule, sweep_value=sweep_value)
    return _draw_six_panels(
        carve,
        X=carve.X_ if X is None else X,
        embedding=embedding,
        palette=cluster_cmap(np.unique(labels).size),
        measure=measure,
        rule=rule,
        not_two=False,
        marker_size=_ATLAS_MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="CARVE_output_heca.png",
        marker_linewidth=_ATLAS_DIAGNOSTIC_OUTLINE,
        cluster_outline=FIG6_OUTLINE_WIDTH,
        scatter_row_height=1.6,
        violin_dots=False,
        violin_note=False,
        sweep_value=sweep_value,
        rows=np.asarray(rows),
        save=save,
        out_dir=out_dir,
    )
