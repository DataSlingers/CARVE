"""Fig 4: ARI against difficulty, one panel per simulated scenario.

Takes the artifact frames it draws. The function this replaces re-simulated
data inside the plotting call, which meant a figure could disagree with the
table beside it, and it took 19 parameters across 408 lines.

The figure is laid out at the PLOS page width under print_theme_context, so
its text prints at the sizes it is set in, and it is saved both as the PNG
the manuscript includes and as Fig4.tif for the PLOS upload.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._panels import metric_legend, metric_lines
from .._theme import (
    CARVE_LINEWIDTH,
    PRINT_FONT_SIZES,
    PRINT_LINEWIDTH,
    PRINT_WIDTH_IN,
    print_axes_text,
    print_theme_context,
    save_figure,
    save_plos_tiff,
)
from ._benchmarking_examples import DEFAULT_SCENARIOS, SCENARIO_TITLES
from ._paths import BENCHMARKING_DIR, figure_path, plos_figure_path

# Legend order, which metric_legend reads directly: the oracle first in its
# own column, then CARVE's two measures in one column, then the four
# classical indices down two more. The classical four are ordered so those
# two columns read Silhouette / Davies-Bouldin and Calinski-Harabasz / Gap,
# as the published figure does.
DEFAULT_METRICS: tuple[str, ...] = (
    "baseline_oracle",
    "ari_stability_1se",
    "ari_generalizability_1se",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "gap",
)

# The difficulty axis is a signal-to-noise sweep; the paper names it that.
X_LABEL = "SNR"
Y_LABEL = r"ARI (selected $\hat{k}$ vs. true labels)"

# The manuscript's figure number, which names the PLOS file.
PLOS_FIGURE_NUMBER = 4

# Panel height at print size, and the room under the grid for the legend.
ROW_HEIGHT_IN = 2.05
LEGEND_HEIGHT_IN = 0.75

# Capitalized, as the published figure prints them and as the per-family
# section overviews draw them.
_DIFFICULTY_TICKS = ("Easy", "Medium", "Hard")


def figure_benchmarking_results(
    results_by_scenario: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
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
    ncols : int
        Panels per row.
    save, out_dir
        As in every figure function.
    """
    if not results_by_scenario:
        raise ValueError("Need at least one scenario to draw.")

    # The x ticks below are relabeled easy/medium/hard unconditionally, so a
    # scaling frame handed to this function would not merely look odd -- its
    # axis would read as difficulty while showing n or p. Rejecting it is the
    # difference between a wrong figure and a traceback.
    off_axis = {
        name: str(frame["axis_name"].iloc[0])
        for name, frame in results_by_scenario.items()
        if not frame.empty and frame["axis_name"].iloc[0] != "difficulty_level"
    }
    if off_axis:
        raise ValueError(
            "Fig 4 labels its x axis easy/medium/hard, so every frame must "
            f"sweep difficulty_level. Got {off_axis}. Use figure_scaling_ari "
            "for scaling sweeps, or figure_scenario_dashboard for one family."
        )

    # Panel order comes from the registry-side reading order, not from how
    # the caller built its mapping -- the notebook builds it by iterating
    # SCENARIOS, whose order exists for the runner rather than for this
    # figure. Anything not in that list keeps caller order, after the rest.
    ranked = {name: index for index, name in enumerate(DEFAULT_SCENARIOS)}
    names = sorted(results_by_scenario, key=lambda name: ranked.get(name, len(ranked)))
    n_rows = -(-len(names) // ncols)

    with print_theme_context():
        fig, axes = plt.subplots(
            n_rows,
            ncols,
            figsize=(PRINT_WIDTH_IN, ROW_HEIGHT_IN * n_rows + LEGEND_HEIGHT_IN),
            squeeze=False,
            sharey=True,
            layout="constrained",
        )

        for index, name in enumerate(names):
            ax = axes[index // ncols, index % ncols]
            frame = results_by_scenario[name]
            metric_lines(
                ax,
                frame,
                metrics=metrics,
                x_col="axis_value",
                x_label=X_LABEL,
                # metric_lines draws CARVE at CARVE_LINEWIDTH and the
                # references lighter; scaled so CARVE prints at the print
                # theme's line width.
                element_scale=PRINT_LINEWIDTH / CARVE_LINEWIDTH,
                show_legend=False,
            )
            print_axes_text(ax)
            ax.set_xticks(sorted(frame["axis_value"].unique()))
            ax.set_xticklabels(_DIFFICULTY_TICKS[: frame["axis_value"].nunique()])
            ax.set_title(
                SCENARIO_TITLES.get(name, name), fontsize=PRINT_FONT_SIZES["title"]
            )
            # One y label for the whole grid, set below as a figure label, so
            # every panel clears its own.
            ax.set_ylabel("")

        for index in range(len(names), n_rows * ncols):
            axes[index // ncols, index % ncols].set_visible(False)

        fig.supylabel(Y_LABEL, fontsize=PRINT_FONT_SIZES["axis_label"])
        metric_legend(
            fig, axes, metrics, fontsize=PRINT_FONT_SIZES["legend"], outside=True
        )

        if save:
            save_figure(
                fig,
                figure_path(
                    "benchmarking_results.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
            save_plos_tiff(fig, plos_figure_path(PLOS_FIGURE_NUMBER, out_dir=out_dir))
    return fig
