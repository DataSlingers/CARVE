"""The hECA calibration scan: cluster count and agreement over resolution.

Shows what the committed grid was chosen from. The left panel draws each
setting's cluster count on the training split, with the organ count and
twice the cell-type count as the rule's targets; the right panel draws
agreement with organ (solid) and cell type (dashed). The committed grid's
range is shaded on both.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._theme import (
    ESTIMATOR_COLORS,
    FALLBACK_COLOR,
    FOREGROUND_COLOR,
    REFERENCE_LINEWIDTH,
    save_figure,
    theme_context,
)
from ._paths import CASE_STUDY_DIR, figure_path


def figure_heca_calibration(
    scan: pd.DataFrame,
    *,
    grid: Sequence[float],
    lower_target: int,
    upper_target: int,
    save: bool = True,
    out_dir: Path | None = None,
    save_name: str = "heca_calibration.png",
) -> Figure:
    """Draw the calibration scan with the rule's targets and the grid's range."""
    if scan.empty:
        raise ValueError("figure_heca_calibration needs a non-empty scan.")
    settings = list(dict.fromkeys(scan["setting"]))
    if len(settings) > len(ESTIMATOR_COLORS):
        raise ValueError(
            f"{len(settings)} settings but {len(ESTIMATOR_COLORS)} estimator "
            "colors in the theme."
        )

    with theme_context():
        fig, (count_ax, ari_ax) = plt.subplots(1, 2, figsize=(10.0, 3.6))
        for color, setting in zip(ESTIMATOR_COLORS, settings, strict=False):
            rows = scan[scan["setting"] == setting].sort_values("resolution")
            count_ax.plot(
                rows["resolution"],
                rows["n_clusters"],
                marker="o",
                color=color,
                label=setting,
            )
            ari_ax.plot(
                rows["resolution"],
                rows["ari_organ"],
                color=color,
                linestyle="-",
                label=f"{setting}, organ",
            )
            ari_ax.plot(
                rows["resolution"],
                rows["ari_cell_type"],
                color=color,
                linestyle="--",
                label=f"{setting}, cell type",
            )
        for target in (lower_target, upper_target):
            count_ax.axhline(
                target,
                color=FOREGROUND_COLOR,
                linewidth=REFERENCE_LINEWIDTH,
                linestyle=":",
            )
        for ax in (count_ax, ari_ax):
            ax.set_xscale("log")
            ax.axvspan(min(grid), max(grid), color=FALLBACK_COLOR, alpha=0.15, lw=0)
            ax.set_xlabel("Resolution")
        count_ax.set_yscale("log")
        count_ax.set_ylabel("Clusters on the training split")
        count_ax.legend()
        ari_ax.set_ylabel("ARI against the reference")
        ari_ax.legend(fontsize="small")
        fig.tight_layout()
        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
