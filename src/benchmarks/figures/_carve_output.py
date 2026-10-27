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
(S8 Fig), Cusanovich and hECA. hECA's figure shows a caller-chosen
resolution rather than the rule's selection, and its two scatters draw a
subsample, which CARVE's public scatter methods do not take; for those it
resolves the configuration through CARVE's _select_row, as the methods do,
and calls the plotting functions behind them.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
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

# Panel A is always the stability overview and panel B always the
# generalizability overview -- their titles say so, and swapping either
# would misname the curve it draws -- but both honor the caller's rule and
# not_two rather than a hardcoded "1se"/False. Panels C, D, E and F show
# whatever CompositeInputs.measure names as the caller's primary selection:
# for Klein that is generalizability (prepare_composite is called with
# measure="generalizability", matching the manuscript's headline k=4
# result); for Levine it is stability, matching that study's own
# prepare_composite call. A caller that leaves measure at its default
# ("stability") gets panel A and panels C to F showing the same selection.
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


def _six_panel_grid(scatter_row_height: float) -> tuple[Figure, tuple[Axes, ...]]:
    """The figure and its six axes, A to F, in three rows of two."""
    # No explicit hspace/wspace here: passing them marks the gridspec as
    # manually customized, which makes every one of CARVE's own plotting
    # calls -- each ends with its own fig.tight_layout() -- warn that the
    # figure is "not compatible with tight_layout" on every subsequent
    # call. Leaving spacing to the default lets those internal calls lay
    # the panels out cleanly instead.
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
    would -- the configuration, its labels and per-sample scores, and the
    configuration note -- and passes the subsample's rows to the plotting
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
    B with a dotted line beside the rule's dashed one. ``rows`` draws E and
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

        # A and B draw the same estimators in the same colors, so one
        # legend names them for both. It goes on the curve of the measure
        # panels C to F show, whose selection its dashed-line entry names.
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
                ax.axvline(
                    sweep_value,
                    color=FOREGROUND_COLOR,
                    linestyle=":",
                    linewidth=REFERENCE_LINEWIDTH,
                    label=f"Shown in C to F: {sweep_value:g}",
                    zorder=0,
                )
        if resolutions is not None:
            for ax in (ax_a, ax_b):
                ax.set_xlabel("Resolution")
                resolution_axis(ax, resolutions)
        for ax in (ax_a, ax_b):
            thin_lines(ax, _PRINT_CURVE_SCALE)
        shown, other = (ax_a, ax_b) if measure == _MEASURE_STABILITY else (ax_b, ax_a)
        if sweep_value is None:
            legends = [(shown, _print_curve_legend(shown))]
        else:
            # Two reference lines under the estimators make one legend too
            # tall for the panel, so the other panel names its own lines.
            legends = [
                (shown, _print_curve_legend(shown, estimators=True)),
                (other, _print_curve_legend(other, estimators=False, title=None)),
            ]
        carve.plot_consensus_matrix(
            **selection,
            ax=ax_c,
            palette=palette,
            title=consensus_title,
            show=False,
        )
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
        for ax in fig.get_axes():
            print_axes_text(ax)
        for letter, ax in zip("ABDEF", (ax_a, ax_b, ax_d, ax_e, ax_f)):
            panel_letter(ax, letter, fontsize=PRINT_FONT_SIZES["panel_letter"])

        # CARVE writes a graph clusterer's full configuration into the curve
        # legend and D's note, where it runs past the figure. The note also
        # breaks before its resolution, which keeps its box narrow inside D.
        for text in fig.findobj(mtext.Text):
            short = compact_setting(text.get_text())
            if short != text.get_text():
                text.set_text(short.replace(" (resolution =", "\n(resolution ="))

        fig.tight_layout()
        for ax, legend in legends:
            _clear_legend(ax, legend)
        # C's square matrix sits centered in its wider cell, and panel_letter
        # places a letter relative to the axes it labels. Once the draw has
        # sized the matrix, C's letter goes where A's is.
        fig.draw_without_rendering()
        above, matrix = ax_a.get_position(), ax_c.get_position()
        letter_x = above.x0 + PANEL_LETTER_X * above.width
        panel_letter(
            ax_c,
            "C",
            x=(letter_x - matrix.x0) / matrix.width,
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
    """Draw the six panels for a case study with a composite (Fig 3 / S8 Fig).

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
    """Build S8 Fig: CARVE output on the Levine mass cytometry case study.

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
    caller chooses, rather than the rule's selection; A and B mark both.
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
