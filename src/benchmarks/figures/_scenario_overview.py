"""One family's section overview: example datasets above ARI over its axis.

Not a manuscript figure. This is the per-scenario view each family section of
the benchmarking notebook carries -- the pair the pre-rebuild notebook drew
with plot_examples and plot_ari_over_difficulty -- assembled from the same
primitives Fig 4 and S1 Fig use, so a section figure and the combined figure
at the end of the notebook cannot disagree about what a family looks like.

The x axis is resolved from the frame rather than assumed. Fig 4 can label
its ticks easy/medium/hard because it only ever draws difficulty sweeps; this
function draws the scaling families too, where those labels would be a
mislabeling rather than a cosmetic difference.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .._panels import metric_legend, metric_lines
from .._registry import PUBLISHED_RANDOM_STATE, SCENARIOS
from .._theme import FONT_SIZES, save_figure, theme_context
from ._benchmarking_examples import SCENARIO_TITLES, draw_example_row
from ._benchmarking_results import DEFAULT_METRICS
from ._paths import BENCHMARKING_DIR, figure_path
from ._scaling import SCALING_PANELS

DIFFICULTY_AXIS_NAME = "difficulty_level"
# The difficulty axis is a signal-to-noise sweep; the paper names it that.
DIFFICULTY_X_LABEL = "SNR"


def _axis_points(frame: pd.DataFrame) -> tuple[list, list[str]]:
    """The frame's (axis_value, axis_label) pairs, ordered by axis value.

    Never by label text: rows are read back by a lexical glob over the
    per-cell checkpoints (_artifacts.read_run), so ordering on the label
    would put "hard" between "easy" and "medium". Same reasoning, and same
    numeric key, as _tables._axis_label_order.
    """
    pairs = (
        frame[["axis_value", "axis_label"]].drop_duplicates().sort_values("axis_value")
    )
    return pairs["axis_value"].tolist(), [str(v) for v in pairs["axis_label"]]


def _x_label(name: str, axis_name: str) -> str:
    """The x-axis label for one scenario, reusing the scaling figure's wording."""
    if axis_name == DIFFICULTY_AXIS_NAME:
        return DIFFICULTY_X_LABEL
    return dict(SCALING_PANELS).get(name, axis_name)


def figure_scenario_overview(
    name: str,
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    seed: int = 0,
    random_state: int = PUBLISHED_RANDOM_STATE,
    save: bool = False,
    out_dir: Path | None = None,
) -> Figure:
    """Build one family's section overview.

    Parameters
    ----------
    name : str
        Registry key naming the scenario.
    results : DataFrame
        That scenario's artifact frame, in the unified schema.
    metrics : sequence of str
        Metric names to draw in the lower panel, in legend order. Defaults to
        Fig 4's metrics, so a section panel and its Fig 4 counterpart show the
        same series.
    seed : int
        Seed-loop index for the example row.
    random_state : int
        The run's random_state term in the example row's seed derivation.
    save : bool
        Write the file. Defaults to False: this is a notebook view, not a
        manuscript figure, so it stays off disk unless asked for.
    out_dir : Path or None
        Override the destination directory.
    """
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}. Known: {sorted(SCENARIOS)}.")
    if results.empty:
        raise ValueError(f"No rows for scenario {name!r}; nothing to draw.")

    scenario = SCENARIOS[name]
    axis_name = str(results["axis_name"].iloc[0])
    values, labels = _axis_points(results)

    # The upper panels are simulated from the registry; the lower panel is
    # read from an artifact. A run directory is content-addressed on its
    # config, so an older run whose axis has since been redefined still sits
    # on disk under its own hash and can still be read. The two halves of
    # this figure would then describe different experiments, with the x ticks
    # taken from the artifact hiding the disagreement. Compare the axes.
    registered = list(scenario.axis.values)
    if values != registered:
        raise ValueError(
            f"Scenario {name!r} is registered on axis {scenario.axis.name}="
            f"{registered}, but the frame sweeps {values}. The example panels "
            "and the curve would describe different experiments. Re-run this "
            f"scenario: python -m benchmarks.run --scenario {name}"
        )

    with theme_context():
        n_cols = len(scenario.axis)
        fig = plt.figure(figsize=(3.7 * n_cols, 6.4))
        grid = fig.add_gridspec(2, n_cols, height_ratios=(1.0, 1.3))
        example_axes = [fig.add_subplot(grid[0, index]) for index in range(n_cols)]
        ari_ax = fig.add_subplot(grid[1, :])

        if axis_name == DIFFICULTY_AXIS_NAME:
            titles = [label.capitalize() for _, _, label in scenario.axis]
        else:
            titles = [
                f"{scenario.axis.name} = {value}" for _, value, _ in scenario.axis
            ]
        draw_example_row(
            example_axes,
            scenario,
            seed=seed,
            random_state=random_state,
            titles=titles,
        )

        metric_lines(
            ari_ax,
            results,
            metrics=metrics,
            x_col="axis_value",
            x_label=_x_label(name, axis_name),
            show_legend=False,
        )
        # Ticks sit on the swept anchors, not on matplotlib's automatic
        # scale. A scaling family holds data at exactly three values, and
        # auto ticks put marks at 2000 and 8000 where nothing was measured.
        ari_ax.set_xticks(values)
        if axis_name == DIFFICULTY_AXIS_NAME:
            ari_ax.set_xticklabels([label.capitalize() for label in labels])

        fig.suptitle(
            SCENARIO_TITLES.get(name, name), fontsize=FONT_SIZES["title"], y=1.0
        )
        fig.tight_layout()
        metric_legend(fig, np.array([ari_ax]), metrics, y_offset=0.06)

        if save:
            save_figure(
                fig,
                figure_path(
                    f"scenario_overview_{name}.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig
