"""Fig 1: one data set per row, clustered with a different k in each panel.

The Motivation notebook computes the embeddings and k-means partitions and
hands them over as DatasetRows; this module only draws them. It is laid out
at the PLOS page width under print_theme_context and saved both as the PNG
the manuscript includes and as Fig1.tif.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .._panels import scatter_clusters
from .._theme import (
    PRINT_FONT_SIZES,
    PRINT_WIDTH_IN,
    print_axes_text,
    print_theme_context,
    save_figure,
    save_plos_tiff,
)
from ._paths import VIS_ROOT, figure_path, plos_figure_path

# The manuscript's figure number, which names the PLOS file.
PLOS_FIGURE_NUMBER = 1
SAVE_NAME = "clustering_problem.png"
MOTIVATION_DIR = VIS_ROOT / "motivational_plots"

# Marker areas are given for the 16-inch layout the figure was first drawn
# at, and scale with the area at the print width.
_SCREEN_WIDTH_IN = 16.0
_PRINT_MARKER_SCALE = (PRINT_WIDTH_IN / _SCREEN_WIDTH_IN) ** 2
_ROW_HEIGHT_IN = 2.1


class DatasetRow(NamedTuple):
    """One row of Fig 1: a data set's embedding and its partition per k."""

    name: str
    embedding: np.ndarray
    partitions: Mapping[int, np.ndarray]
    marker_size: float


def figure_clustering_problem(
    rows: Sequence[DatasetRow], *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build Fig 1: each row's embedding colored by each of its partitions.

    Rows are lettered A, B, ... at their left edge, and each row's first
    panel names the data set. Every row needs the same number of panels.
    """
    n_panels = {len(row.partitions) for row in rows}
    if len(n_panels) != 1:
        raise ValueError("Every row needs the same number of partitions.")
    n_cols = n_panels.pop()

    with print_theme_context():
        fig, axes = plt.subplots(
            len(rows),
            n_cols,
            figsize=(PRINT_WIDTH_IN, _ROW_HEIGHT_IN * len(rows)),
            squeeze=False,
            layout="constrained",
        )
        for row, panels in zip(rows, axes):
            for ax, (k, labels) in zip(panels, row.partitions.items()):
                scatter_clusters(
                    ax,
                    row.embedding,
                    labels,
                    title=f"k={k}",
                    s=row.marker_size * _PRINT_MARKER_SCALE,
                )
                print_axes_text(ax)
            # scatter_clusters hides spines and ticks but leaves the ylabel,
            # which names the row's data set.
            panels[0].set_ylabel(
                row.name, fontsize=PRINT_FONT_SIZES["title"], fontweight="bold"
            )

        # Each row's letter sits at its top-left corner. The layout engine
        # has placed the axes by the time it is drawn, so the letters are
        # positioned against the drawn axes.
        fig.canvas.draw()
        for letter, panels in zip("ABCDEFGH", axes):
            top = panels[0].get_position().y1
            fig.text(
                0.0,
                top,
                letter,
                fontsize=PRINT_FONT_SIZES["panel_letter"],
                fontweight="bold",
                va="bottom",
                ha="left",
            )

        if save:
            save_figure(
                fig, figure_path(SAVE_NAME, subdir=MOTIVATION_DIR, out_dir=out_dir)
            )
            save_plos_tiff(fig, plos_figure_path(PLOS_FIGURE_NUMBER, out_dir=out_dir))
    return fig
