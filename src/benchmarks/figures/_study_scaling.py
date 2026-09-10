"""Cost of a case study as a function of sample count.

Runtime and peak resident memory answer the reviewer request directly;
selected k alongside them shows whether the selection itself is stable as n
grows, which a cost-only figure would leave open.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._theme import save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path

PANELS: tuple[tuple[str, str], ...] = (
    ("wall_clock_s", "Runtime (seconds)"),
    ("peak_rss_gb", "Peak memory (GB)"),
    ("selected_k", "Selected k"),
)


def figure_study_scaling(
    sweep_df: pd.DataFrame,
    *,
    save: bool = True,
    out_dir: Path | None = None,
    save_name: str = "heca_scaling.png",
) -> Figure:
    """Draw runtime, peak memory and selected k against sample count."""
    if sweep_df.empty:
        raise ValueError("figure_study_scaling needs at least one row to draw.")

    frame = sweep_df.sort_values("n").copy()
    # Bytes on the axis would render as 9e9 and be unreadable.
    frame["peak_rss_gb"] = frame["peak_rss_bytes"] / 1e9

    with theme_context():
        fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 3.0))
        for ax, (column, label) in zip(axes, PANELS):
            ax.plot(frame["n"], frame[column], marker="o")
            ax.set_xlabel("Number of cells (n)")
            ax.set_ylabel(label)
            if column != "selected_k":
                ax.set_xscale("log")
                ax.set_yscale("log")

        fig.tight_layout()
        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
