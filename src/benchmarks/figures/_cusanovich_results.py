"""The Cusanovich mouse sci-ATAC case-study figure.

Four panels over CusanovichInputs. (A) CARVE's consensus labels on the
embedding of the pipeline CARVE rates best at its selected configuration.
(B) The source's clusters on the source's own t-SNE, which cell_metadata.txt
ships and the loader carries as meta["source_tsne"]. (C) CARVE's stability
and generalizability over resolution, pooled over pipelines, with the selected
resolution, the source's operating point and the published partition's own
generalizability marked, and the mean observed cluster count on a secondary
axis. (D) The same two criteria per pipeline at the selected configuration,
on C's x axis. A and B share one color map, so a CARVE cluster takes the
color of the source cluster it best matches; with 30 source clusters the
palette cycles, as _theme.cluster_colors documents.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._cusanovich_compare import CusanovichInputs, axis_prefix
from .._panels import (
    aligned_color_maps,
    axis_arrows,
    carve_lines,
    panel_letter,
    pipeline_lines,
    scatter_clusters,
)
from .._theme import FONT_SIZES, metric_color, save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path

MARKER_SIZE = 8.0
SOURCE_TSNE_LABELS = ("t-SNE 1", "t-SNE 2")
SAVE_NAME = "cusanovich_results.png"


def _panel_a_title(inputs: CusanovichInputs, selected, n_clusters: int) -> str:
    carve = inputs.carve
    estimator = str(selected["estimator"]).removesuffix("Clustering")
    pooled = sorted(
        {
            axis_prefix(spec.dim_reduction)
            for spec in carve.preprocessing_pipelines_.values()
        },
        key=str.lower,
    )
    shown = inputs.embedding_A_labels[0].removesuffix(" 1")
    return (
        f"CARVE: {estimator}, {carve.sweep_.param} "
        f"{float(selected['sweep_value']):g}, {n_clusters} clusters\n"
        f"consensus over {', '.join(pooled)}; shown on {shown} 1/2"
    )


def _mark_source(ax: Axes, inputs: CusanovichInputs, method_id: str) -> None:
    """The source's operating point on C's curves, and its partition's score.

    The operating point is placed on the pooled curves at the resolution where
    the t-SNE pipeline's clusterings come nearest the source's cluster count,
    and labeled with that count. The horizontal line is the published
    partition's own generalizability under the same classifier probe.
    """
    results = inputs.carve.estimator_results_
    curve = results.loc[results["method_id"] == method_id]
    resolution, observed = inputs.operating_point
    at_point = curve.loc[np.isclose(curve["sweep_value"].astype(float), resolution)]
    n_source = int(np.unique(inputs.y).size)

    for index, measure in enumerate(("stability", "generalizability")):
        ax.plot(
            [resolution],
            [float(at_point[f"ari_{measure}"].iloc[0])],
            marker="D",
            markersize=8.0,
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=1.5,
            markeredgecolor=metric_color(f"ari_{measure}_1se"),
            zorder=3,
            label=(
                f"source operating point, {observed:.0f} clusters on t-SNE"
                if index == 0
                else "_nolegend_"
            ),
        )

    mean, _ = inputs.published_generalizability
    ax.axhline(
        mean,
        color=metric_color("ari_generalizability_1se"),
        linestyle="--",
        linewidth=1.2,
        label=f"published {n_source} clusters, RF probe",
    )
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)


def _observed_cluster_axis(ax: Axes, inputs: CusanovichInputs, method_id: str) -> None:
    """A secondary x axis naming the mean observed cluster count per resolution."""
    results = inputs.carve.estimator_results_
    curve = results.loc[results["method_id"] == method_id].sort_values("sweep_value")
    top = ax.secondary_xaxis("top")
    top.set_xticks(
        curve["sweep_value"].to_numpy(dtype=float),
        labels=[f"{count:.0f}" for count in curve["n_clusters_observed"]],
    )
    top.tick_params(labelsize=FONT_SIZES["tick"])
    top.set_xlabel("Mean observed clusters", fontsize=FONT_SIZES["axis_label"])


def figure_cusanovich_results(
    inputs: CusanovichInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build the Cusanovich case-study figure from assembled inputs."""
    carve = inputs.carve
    selected, _, n_clusters, _ = carve._select_row(
        measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
    )
    method_id = str(selected["method_id"])
    source_cmap, carve_cmap = aligned_color_maps(inputs.y, inputs.carve_labels)
    n_source = int(np.unique(inputs.y).size)

    with theme_context():
        fig, axes = plt.subplots(2, 2, figsize=(15.0, 13.0))
        ax_a, ax_b, ax_c, ax_d = axes.flat
        ax_d.sharex(ax_c)

        scatter_clusters(
            ax_a,
            inputs.embedding_A,
            inputs.carve_labels,
            color_map=carve_cmap,
            s=MARKER_SIZE,
            title=_panel_a_title(inputs, selected, n_clusters),
        )
        axis_arrows(ax_a, inputs.embedding_A_labels)
        scatter_clusters(
            ax_b,
            inputs.source_tsne,
            inputs.y,
            color_map=source_cmap,
            s=MARKER_SIZE,
            title=f"Cusanovich et al.: Louvain on t-SNE, {n_source} clusters",
        )
        axis_arrows(ax_b, SOURCE_TSNE_LABELS)

        carve_lines(
            ax_c,
            carve,
            rule=inputs.rule,
            not_two=inputs.not_two,
            title=f"CARVE over {carve.sweep_.param}, pooled over pipelines",
        )
        _mark_source(ax_c, inputs, method_id)
        _observed_cluster_axis(ax_c, inputs, method_id)

        pipeline_lines(
            ax_d,
            carve,
            method_id=method_id,
            rule=inputs.rule,
            not_two=inputs.not_two,
            title="Per pipeline, at the selected configuration",
        )

        for letter, ax in zip("ABCD", axes.flat):
            panel_letter(ax, letter)
        fig.tight_layout()

        if save:
            save_figure(
                fig, figure_path(SAVE_NAME, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
