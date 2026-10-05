"""CARVE's own view of the hECA fit: both criteria and the cluster count.

Stability and generalizability over resolution, one line per setting, and
the observed cluster count over resolution, all drawn by CARVE's plotting
methods on a log resolution axis.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .._theme import ESTIMATOR_CMAP_NAME, FONT_SIZES, save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path

#: The notebook's reference scatter: a declared fixed subsample of cells,
#: because half a million points do not rasterize legibly, drawn on a UMAP
#: of those cells.
MARKER_SIZE = 3.0
AXIS_LABELS = ("UMAP 1", "UMAP 2")
DEFAULT_SCATTER_SUBSAMPLE = 50_000


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

    Each panel's own legend is suppressed (``legend=False``): CARVE's
    legend text is long enough -- the full estimator repr, e.g.
    "LeidenClustering, n_neighbors=15, objective_function=modularity" --
    that three of them side by side at this panel width overflow their own
    axes into the next panel. The three panels' colored lines name the same
    two settings, so one legend below the figure, de-duplicated by label
    text and built from the handles CARVE already attached to each axes,
    says it once instead of three times, while keeping each panel's own
    selected-resolution line (its label differs per panel) in the legend.
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
            ax.set_xscale("log")
            ax.set_xticks(list(resolutions), labels=[f"{r:g}" for r in resolutions])
            ax.tick_params(axis="x", labelrotation=90)
            ax.minorticks_off()

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
