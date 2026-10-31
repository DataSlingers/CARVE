"""CARVE's own view of the hECA fit: both criteria and the cluster count.

Stability and generalizability over resolution, one line per setting, and
the observed cluster count over resolution, all drawn by CARVE's plotting
methods on a log resolution axis.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._theme import ESTIMATOR_CMAP_NAME, FONT_SIZES, save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path

#: The notebook's scatters: a declared fixed subsample of cells, because half
#: a million points do not rasterize legibly, drawn on a UMAP of those cells.
#: At 50,000 cells the markers are small and thinly outlined, so the outline
#: does not darken the dense regions; the reference scatter's legend markers
#: take their own, larger size, since at 2 points the organ colors cannot be
#: told apart in the legend.
MARKER_SIZE = 2.0
MARKER_LINEWIDTH = 0.1
LEGEND_MARKER_SIZE = 40.0
AXIS_LABELS = ("UMAP 1", "UMAP 2")
DEFAULT_SCATTER_SUBSAMPLE = 50_000


def resolution_axis(ax: Axes, resolutions: Sequence[float]) -> None:
    """Log-scale x axis with one labeled tick per fitted resolution."""
    ax.set_xscale("log")
    ax.set_xticks(list(resolutions), labels=[f"{r:g}" for r in resolutions])
    ax.tick_params(axis="x", labelrotation=90)
    ax.minorticks_off()


def figure_heca_carve(
    carve: Any,
    *,
    resolutions: Sequence[float],
    rule: str = "1se",
    save: bool = True,
    out_dir: Path | None = None,
    save_name: str = "heca_carve.png",
) -> Figure:
    """Draw stability, generalizability and cluster count over resolution.

    Each panel's own legend is suppressed (``legend=False``), since CARVE's
    full estimator reprs overflow the panel width. One legend below the
    figure, de-duplicated by label from the handles CARVE attached to each
    axes, names the two settings once and keeps each panel's
    selected-resolution line.
    """
    with theme_context():
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
        carve.plot_metric_over_n_clusters(
            measure="stability",
            rule=rule,
            ax=axes[0],
            palette=ESTIMATOR_CMAP_NAME,
            title="Stability",
            legend=False,
        )
        carve.plot_metric_over_n_clusters(
            measure="generalizability",
            rule=rule,
            ax=axes[1],
            palette=ESTIMATOR_CMAP_NAME,
            title="Generalizability",
            legend=False,
        )
        carve.plot_n_clusters_over_sweep(
            measure="stability",
            rule=rule,
            ax=axes[2],
            palette=ESTIMATOR_CMAP_NAME,
            title="Observed clusters",
            legend=False,
        )
        for ax in axes:
            resolution_axis(ax, resolutions)

        handles: list[Any] = []
        labels: list[str] = []
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=3,
            frameon=False,
            fontsize=FONT_SIZES["legend"],
        )
        fig.tight_layout()
        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
