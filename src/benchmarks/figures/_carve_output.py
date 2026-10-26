"""Fig 3 and S4 Fig: CARVE's own output on a case study.

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

figure_carve_output_heca draws the same six panels for the hECA notebook at
a caller-chosen resolution. Its two scatters draw a subsample, which CARVE's
public scatter methods do not take, so for those it resolves the
configuration through CARVE's _select_row, as the methods do, and calls the
plotting functions behind them.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from carve._plotting import plot_cluster_scatter as _plot_cluster_scatter
from carve._plotting import plot_diagnostic_scatter as _plot_diagnostic_scatter

from .._panels import PANEL_LETTER_X, compact_setting, panel_letter
from .._theme import (
    ESTIMATOR_CMAP_NAME,
    FONT_SIZES,
    FOREGROUND_COLOR,
    PRINT_FONT_SIZES,
    PRINT_WIDTH_IN,
    REFERENCE_LINEWIDTH,
    cluster_cmap,
    print_axes_text,
    print_theme_context,
    save_figure,
    save_plos_tiff,
    theme_context,
)
from ._case_study import (
    CompositeInputs,
    _swept_resolutions,
    carve_labels_aligned,
    carve_output_cmap,
)
from ._heca_carve import AXIS_LABELS, MARKER_LINEWIDTH, MARKER_SIZE, resolution_axis
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


# The hECA figure is drawn at this width and scaled down on the page. Fig 3
# and S8 Fig are laid out at PRINT_WIDTH_IN, where a row of two panels is
# PRINT_ROW_HEIGHT_IN tall; Fig 3's taller scatter row then keeps the figure
# under PRINT_MAX_HEIGHT_IN.
_SCREEN_WIDTH_IN = 13.0
_SCREEN_ROW_HEIGHT_IN = 14.0 / 3
PRINT_ROW_HEIGHT_IN = 2.5

# Scatter marker sizes are given for the screen-width figure. Marker area is
# in points squared, so at the print width it scales with the width ratio
# squared, which keeps each marker's size relative to its panel.
_PRINT_MARKER_SCALE = (PRINT_WIDTH_IN / _SCREEN_WIDTH_IN) ** 2

# CARVE draws its metric curves with 2-point lines, 6-point markers and
# 4-point caps, which suit the screen width; at the print width they are
# drawn at this fraction.
_PRINT_CURVE_SCALE = 0.6

# The estimator reprs CARVE writes into the legend, as the print legend
# names them. At the print width the full reprs run across the curves.
_ESTIMATOR_LABELS = {
    "AgglomerativeClustering, linkage=ward": "Ward agglomerative",
    "SpectralClustering, affinity=self_tuning": "Spectral, self-tuning",
}


def _thin_curves(ax: Axes, scale: float) -> None:
    """Scale the line widths and marker sizes CARVE drew a curve panel with.

    Error bar caps are markers on lines of their own, and the bars are line
    collections, so both follow.
    """
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth() * scale)
        line.set_markersize(line.get_markersize() * scale)
    for collection in ax.collections:
        collection.set_linewidths(np.asarray(collection.get_linewidths()) * scale)


def _print_curve_legend(ax: Axes) -> None:
    """The estimator legend CARVE draws, rebuilt from the thinned curves.

    A legend copies its handles when it is made, so it is drawn after the
    curves are thinned, in CARVE's place and style, with short names. At the
    print width the legend takes a larger share of the panel, so the y axis
    is extended downward by a fifth of its range to keep it under the curves.
    """
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom - 0.2 * (top - bottom), top)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [_ESTIMATOR_LABELS.get(label, label) for label in labels],
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        title="Estimators",
    )


def _six_panel_grid(
    scatter_row_height: float,
    *,
    width_in: float = _SCREEN_WIDTH_IN,
    row_height_in: float = _SCREEN_ROW_HEIGHT_IN,
) -> tuple[Figure, tuple[Axes, ...]]:
    """The figure and its six axes, A to F, in three rows of two."""
    # No explicit hspace/wspace here: passing them marks the gridspec as
    # manually customized, which makes every one of CARVE's own plotting
    # calls -- each ends with its own fig.tight_layout() -- warn that the
    # figure is "not compatible with tight_layout" on every subsequent
    # call. Leaving spacing to the default lets those internal calls lay
    # the panels out cleanly instead.
    fig = plt.figure(figsize=(width_in, row_height_in * (2 + scatter_row_height)))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, scatter_row_height))
    axes = tuple(fig.add_subplot(gs[row, col]) for row in range(3) for col in range(2))
    return fig, axes


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
    """Draw the six-panel CARVE diagnostic figure (Fig 3 / S8 Fig).

    ``marker_size`` scales panel F's fixed marker size directly (that
    parameter's name on ``plot_diagnostic_scatter`` matches this one
    exactly) and panel E's ``size_range`` proportionally, so the two
    datasets' scatters differ only in dot size, not in the score encoding
    itself. ``marker_linewidth`` is panel F's outline width, under the same
    name.

    Panels E and F carry no legend: D numbers the clusters in E's colors,
    and F keeps its colorbar.

    ``scatter_row_height`` is the height of the E-F row relative to each of
    the other two rows. The figure grows with it, so panels A to D keep
    their size.

    The figure is laid out at PRINT_WIDTH_IN under print_theme_context, and
    the text CARVE's methods size for the screen is reset to the print
    sizes. ``plos_number`` names the PLOS figure file saved beside the PNG;
    an SI figure, which PLOS takes in any format, leaves it None.
    """
    carve = inputs.carve
    measure = inputs.measure
    rule = inputs.rule
    not_two = inputs.not_two
    selected_k = int(carve.get_k(measure=measure, rule=rule, not_two=not_two))
    # Exactly selected_k colors, so CARVE's direct-index panels (D, E) and
    # its imshow-normalized one (C's cluster band) resolve the same cluster
    # to the same color, the one the composite and the composition figure
    # give it. Panels A and B color one line per estimator, not per cluster,
    # so they take the estimator map instead.
    cluster_panel_cmap = carve_output_cmap(inputs)
    marker_size = marker_size * _PRINT_MARKER_SCALE
    scatter_size_range = (marker_size * 0.75, marker_size * 3.0)
    # A resolution run's curves go on the log axis hECA's figure uses, with
    # one tick per fitted resolution.
    resolutions = _swept_resolutions(carve)
    over = "$k$" if resolutions is None else "resolution"

    with print_theme_context(), carve_labels_aligned(carve, inputs.y):
        fig, (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f) = _six_panel_grid(
            scatter_row_height,
            width_in=PRINT_WIDTH_IN,
            row_height_in=PRINT_ROW_HEIGHT_IN,
        )

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
        if resolutions is not None:
            for ax in (ax_a, ax_b):
                ax.set_xlabel("Resolution")
                resolution_axis(ax, resolutions)
        for ax in (ax_a, ax_b):
            _thin_curves(ax, _PRINT_CURVE_SCALE)
        _print_curve_legend(ax_a if measure == _MEASURE_STABILITY else ax_b)
        carve.plot_consensus_matrix(
            measure=measure,
            rule=rule,
            not_two=not_two,
            ax=ax_c,
            palette=cluster_panel_cmap,
            title=f"Consensus matrix ($k={selected_k}$)",
            show=False,
        )
        carve.plot_cluster_violin(
            source="gini",
            measure=measure,
            rule=rule,
            not_two=not_two,
            ax=ax_d,
            palette=cluster_panel_cmap,
            title="Per-cluster stability",
            size=8.0 * _PRINT_MARKER_SCALE,
            show=False,
        )
        carve.plot_cluster_scatter(
            source="gini",
            measure=measure,
            rule=rule,
            not_two=not_two,
            X=inputs.X,
            embedding=inputs.Z,
            ax=ax_e,
            palette=cluster_panel_cmap,
            size_range=scatter_size_range,
            legend=False,
            title="Consensus labels",
            xlabel=axis_labels[0],
            ylabel=axis_labels[1],
            show=False,
        )
        carve.plot_diagnostic_scatter(
            source="gini",
            measure=measure,
            rule=rule,
            not_two=not_two,
            X=inputs.X,
            embedding=inputs.Z,
            ax=ax_f,
            marker_size=marker_size,
            marker_linewidth=marker_linewidth,
            legend=False,
            title="Consensus assignment (diagnostic)",
            xlabel=axis_labels[0],
            ylabel=axis_labels[1],
            show=False,
        )

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
        scatter_row_height=1.2,
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
    cells. The E-F row is 1.3 times the height of the others.
    """
    return carve_output_figure(
        inputs,
        marker_size=8.0,
        axis_labels=("t-SNE 1", "t-SNE 2"),
        save_name="CARVE_output_levine.png",
        marker_linewidth=0.05,
        scatter_row_height=1.3,
        save=save,
        out_dir=out_dir,
    )


# CARVE's public scatter methods resolve these names for the gini source;
# the hECA figure calls the functions behind them and passes the same.
_GINI_SCORES_NAME = "Gini Stability"

# The UMAP's islands spread over its full height, and F shares its cell with
# the colorbar, so the E-F row is 1.4 of the other rows' height; the figure
# grows with it and panels A to D keep their size.
_HECA_SCATTER_ROW_HEIGHT = 1.4


def figure_carve_output_heca(
    carve: Any,
    *,
    sweep_value: float,
    resolutions: Sequence[float],
    rows: np.ndarray,
    embedding: np.ndarray,
    X: np.ndarray | None = None,
    measure: str = "stability",
    rule: str = "1se",
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Build the hECA counterpart of Fig 3: CARVE's output at one resolution.

    The six panels of carve_output_figure, adapted to a resolution sweep on
    695,304 cells. Panels C to F show the partition at ``sweep_value``, a
    resolution the caller chooses, rather than the rule's selection; A and
    B mark both on a log resolution axis, with one legend in B that names
    each setting by its neighbor count. ``measure`` and ``rule`` pick
    between the settings fitted at ``sweep_value``. CARVE's configuration
    note is left out of D, E and F: at this panel width it overflows the
    figure, and C's title and the caption carry the same information.

    C is the anchor block, the consensus an anchored fit stores. D's violins
    take every cell's score, without CARVE's per-cell dots, which at this
    size merge into a solid band. E and F draw ``rows``, a subsample, at
    ``embedding``, one coordinate pair per subsample row. CARVE's public
    scatter methods take coordinates for every cell, so E and F call the
    plotting functions those methods wrap, with the labels and per-sample
    scores the methods would resolve. Neither carries a cluster legend; C's
    band and D's axis number the clusters in the same colors, and F keeps
    its score colorbar. Their markers take hECA's size and outline width;
    CARVE's plot_cluster_scatter fixes E's outline at 0.2 points, so it is
    thinned on the drawn markers afterwards. ``X`` defaults to ``carve.X_``.

    Raises ValueError when ``rows`` misses a cluster: CARVE's scatters color
    and shape clusters by their position among the labels drawn, so a
    missing cluster would shift E and F away from C and D.
    """
    rows = np.asarray(rows)
    data = np.asarray(carve.X_ if X is None else X)
    _, config_id, selected_k, _ = carve._select_row(
        measure=measure, rule=rule, sweep_value=sweep_value
    )
    labels = np.asarray(
        carve.get_labels(
            measure=measure,
            rule=rule,
            sweep_value=sweep_value,
            consensus_k=selected_k,
        )
    )
    clusters = np.unique(labels)
    if not np.array_equal(np.unique(labels[rows]), clusters):
        raise ValueError(
            "rows must include every cluster at this resolution; CARVE's "
            "scatters color clusters by position among the labels drawn."
        )
    scores = carve._sample_scores("gini", config_id)
    cluster_panel_cmap = cluster_cmap(clusters.size)

    with theme_context():
        fig, (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f) = _six_panel_grid(
            _HECA_SCATTER_ROW_HEIGHT
        )

        for ax, curve_measure, title in (
            (ax_a, _MEASURE_STABILITY, "Stability ARI over resolution"),
            (ax_b, _MEASURE_GENERALIZABILITY, "Generalizability ARI over resolution"),
        ):
            carve.plot_metric_over_n_clusters(
                measure=curve_measure,
                rule=rule,
                ax=ax,
                palette=ESTIMATOR_CMAP_NAME,
                title=title,
                xlabel="Resolution",
                legend=False,
                show=False,
            )
            ax.axvline(
                sweep_value,
                color=FOREGROUND_COLOR,
                linestyle=":",
                linewidth=REFERENCE_LINEWIDTH,
                label=f"Shown in C to F: {sweep_value:g}",
                zorder=0,
            )
            resolution_axis(ax, resolutions)
        handles, texts = ax_b.get_legend_handles_labels()
        ax_b.legend(
            handles,
            [compact_setting(text) for text in texts],
            loc="upper right",
            frameon=False,
            fontsize=FONT_SIZES["legend"],
        )

        carve.plot_consensus_matrix(
            measure=measure,
            rule=rule,
            sweep_value=sweep_value,
            ax=ax_c,
            palette=cluster_panel_cmap,
            title=(
                f"Consensus matrix ($k={clusters.size}$, "
                f"{carve.consensus_anchors_.size:,} anchor cells)"
            ),
            show=False,
        )
        carve.plot_cluster_violin(
            source="gini",
            measure=measure,
            rule=rule,
            sweep_value=sweep_value,
            ax=ax_d,
            palette=cluster_panel_cmap,
            stripplot=False,
            title="Per-cluster stability",
            annotation=False,
            show=False,
        )
        scatter_args = (data[rows], labels[rows], scores[rows])
        _plot_cluster_scatter(
            *scatter_args,
            embedding=embedding,
            ax=ax_e,
            palette=cluster_panel_cmap,
            size_range=(MARKER_SIZE * 0.75, MARKER_SIZE * 3.0),
            legend=False,
            title="Consensus labels",
            scores_name=_GINI_SCORES_NAME,
            xlabel=AXIS_LABELS[0],
            ylabel=AXIS_LABELS[1],
            show=False,
        )
        for points in ax_e.collections:
            points.set_linewidths(MARKER_LINEWIDTH)
        _plot_diagnostic_scatter(
            *scatter_args,
            embedding=embedding,
            ax=ax_f,
            marker_size=MARKER_SIZE,
            marker_linewidth=MARKER_LINEWIDTH,
            legend=False,
            title="Consensus assignment (diagnostic)",
            scores_name=_GINI_SCORES_NAME,
            xlabel=AXIS_LABELS[0],
            ylabel=AXIS_LABELS[1],
            show=False,
        )

        for letter, ax in zip("ABCDEF", (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f)):
            panel_letter(ax, letter)

        fig.tight_layout()

        if save:
            save_figure(
                fig,
                figure_path(
                    "CARVE_output_heca.png", subdir=CASE_STUDY_DIR, out_dir=out_dir
                ),
            )
    return fig
