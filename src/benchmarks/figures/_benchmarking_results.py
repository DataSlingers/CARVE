"""Fig 4: ARI against difficulty, one panel per simulated scenario.

Takes the artifact frames it draws. The function this replaces re-simulated
data inside the plotting call, which meant a figure could disagree with the
table beside it, and it took 19 parameters across 408 lines.
"""

import string
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._panels import grouped_legend, metric_lines, panel_letter
from .._theme import FONT_SIZES, save_figure, theme_context
from ._benchmarking_examples import SCENARIO_TITLES
from ._paths import BENCHMARKING_DIR, figure_path

DEFAULT_METRICS: tuple[str, ...] = (
    "ari_stability_1se",
    "ari_generalizability_1se",
    "silhouette",
    "gap",
    "davies_bouldin",
    "calinski_harabasz",
)

_DIFFICULTY_TICKS = ("easy", "medium", "hard")


def figure_benchmarking_results(
    results_by_scenario: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    k_star: int = 5,
    ncols: int = 3,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Build Fig 4.

    Parameters
    ----------
    results_by_scenario : mapping of scenario name to its artifact frame
        Frames follow the unified schema, so no axis-column sniffing is needed.
    metrics : sequence of str
        Metric names to draw, in legend order.
    k_star : int
        True cluster count, used only for the panel subtitle.
    ncols : int
        Panels per row.
    save, out_dir
        As in every figure function.
    """
    if not results_by_scenario:
        raise ValueError("Need at least one scenario to draw.")

    names = list(results_by_scenario)
    n_rows = -(-len(names) // ncols)

    with theme_context():
        fig, axes = plt.subplots(
            n_rows,
            ncols,
            figsize=(4.0 * ncols, 3.2 * n_rows),
            squeeze=False,
            sharey=True,
        )

        for index, name in enumerate(names):
            ax = axes[index // ncols, index % ncols]
            frame = results_by_scenario[name]
            metric_lines(
                ax,
                frame,
                metrics=metrics,
                x_col="axis_value",
                x_label="Difficulty",
                show_legend=False,
            )
            ax.set_xticks(sorted(frame["axis_value"].unique()))
            ax.set_xticklabels(_DIFFICULTY_TICKS[: frame["axis_value"].nunique()])
            ax.set_title(SCENARIO_TITLES.get(name, name), fontsize=FONT_SIZES["title"])
            if index % ncols != 0:
                ax.set_ylabel("")
            panel_letter(ax, string.ascii_uppercase[index])

        for index in range(len(names), n_rows * ncols):
            axes[index // ncols, index % ncols].set_visible(False)

        fig.suptitle(f"$k^\\star = {k_star}$", fontsize=FONT_SIZES["title"], y=1.0)
        fig.tight_layout()
        grouped_legend(fig, axes, y_offset=0.06)

        if save:
            save_figure(
                fig,
                figure_path(
                    "benchmarking_results.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig
