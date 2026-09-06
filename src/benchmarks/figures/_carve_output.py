"""Fig 3 and S4 Fig: CARVE's own diagnostic output on a case study.

Six panels, and every one of them is a plot CARVE already ships as a method
on the fitted object: (A) stability ARI over k, (B) the consensus matrix for
the selected configuration, (C) generalizability ARI over k, (D) per-cluster
stability as a violin plot, (E) the embedding colored by consensus labels
with dubious samples emphasized, and (F) the same embedding with marker
shape per cluster and score-encoded color. This module only arranges the six
axes into one gridspec, calls each method with ``ax=`` set, and saves the
result once; it does not resolve config_id, recompute a selection, or
reimplement any encoding CARVE's own plotting functions already provide.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .._panels import panel_letter
from .._theme import CLUSTER_CMAP_NAME, save_figure, theme_context
from ._case_study import CompositeInputs
from ._paths import CASE_STUDY_DIR, figure_path

# Panel A is always the stability overview and panel C always the
# generalizability overview -- their titles say so, and swapping either
# would misname the curve it draws -- but both honor the caller's rule and
# not_two rather than a hardcoded "1se"/False. Panels B, D, E and F show
# whatever CompositeInputs.measure names as the caller's primary selection:
# for Klein that is generalizability (prepare_composite is called with
# measure="generalizability", matching the manuscript's headline k=4
# result); for Levine it is stability, matching that study's own
# prepare_composite call. A caller that leaves measure at its default
# ("stability") gets the same panels A and B/D/E/F showing the same
# selection, as the previous, unparameterized version of this figure always
# did.
_MEASURE_STABILITY = "stability"
_MEASURE_GENERALIZABILITY = "generalizability"


def carve_output_figure(
    inputs: CompositeInputs,
    *,
    marker_size: float,
    axis_labels: Sequence[str],
    save_name: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Draw the six-panel CARVE diagnostic figure (Fig 3 / S4 Fig).

    ``marker_size`` scales panel F's fixed marker size directly (that
    parameter's name on ``plot_diagnostic_scatter`` matches this one
    exactly) and panel E's ``size_range`` proportionally, so the two
    datasets' scatters differ only in dot size, not in the score encoding
    itself.
    """
    carve = inputs.carve
    measure = inputs.measure
    rule = inputs.rule
    not_two = inputs.not_two
    selected_k = int(carve.get_k(measure=measure, rule=rule, not_two=not_two))
    scatter_size_range = (marker_size * 0.75, marker_size * 3.0)

    with theme_context():
        # No explicit hspace/wspace here: passing them marks the gridspec as
        # manually customized, which makes every one of CARVE's own plotting
        # calls -- each ends with its own fig.tight_layout() -- warn that the
        # figure is "not compatible with tight_layout" on every subsequent
        # call. Leaving spacing to the default lets those internal calls lay
        # the panels out cleanly instead.
        fig = plt.figure(figsize=(18.0, 11.0))
        gs = fig.add_gridspec(2, 3)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, 0])
        ax_e = fig.add_subplot(gs[1, 1])
        ax_f = fig.add_subplot(gs[1, 2])

        carve.plot_metric_over_n_clusters(
            measure=_MEASURE_STABILITY,
            rule=rule,
            not_two=not_two,
            ax=ax_a,
            palette=CLUSTER_CMAP_NAME,
            title="Stability ARI over $k$",
            show=False,
        )
        carve.plot_consensus_matrix(
            measure=measure,
            rule=rule,
            not_two=not_two,
            ax=ax_b,
            palette=CLUSTER_CMAP_NAME,
            title=f"Consensus matrix ($k={selected_k}$)",
            show=False,
        )
        carve.plot_metric_over_n_clusters(
            measure=_MEASURE_GENERALIZABILITY,
            rule=rule,
            not_two=not_two,
            ax=ax_c,
            palette=CLUSTER_CMAP_NAME,
            title="Generalizability ARI over $k$",
            show=False,
        )
        carve.plot_cluster_violin(
            source="gini",
            measure=measure,
            rule=rule,
            not_two=not_two,
            ax=ax_d,
            palette=CLUSTER_CMAP_NAME,
            title="Per-cluster stability",
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
            palette=CLUSTER_CMAP_NAME,
            size_range=scatter_size_range,
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
            title="Consensus assignment (diagnostic)",
            xlabel=axis_labels[0],
            ylabel=axis_labels[1],
            show=False,
        )

        for letter, ax in zip("ABCDEF", (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f)):
            panel_letter(ax, letter)

        fig.tight_layout()

        if save:
            save_figure(
                fig,
                figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir),
            )
    return fig


def figure_carve_output_klein(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build Fig 3: CARVE output on the Klein droplet scRNA-seq case study."""
    return carve_output_figure(
        inputs,
        marker_size=20.0,
        axis_labels=("PC1", "PC2"),
        save_name="CARVE_output_klein.png",
        save=save,
        out_dir=out_dir,
    )


def figure_carve_output_levine(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build S4 Fig: CARVE output on the Levine mass cytometry case study."""
    return carve_output_figure(
        inputs,
        marker_size=8.0,
        axis_labels=("t-SNE 1", "t-SNE 2"),
        save_name="CARVE_output_levine.png",
        save=save,
        out_dir=out_dir,
    )
