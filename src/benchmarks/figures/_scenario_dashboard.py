"""One scenario's whole story in one figure.

Not a manuscript figure. This is what each family section of the
benchmarking notebook carries, in place of the overview-plus-table pair it
used to. The four rows answer four questions in reading order: what the data
looks like, what each method scored on it, why each method chose the k it
chose, and how often it chose each k.

Every panel is drawn through the same primitives as the manuscript figures,
and the ARI row draws Fig 4's series from the same artifact, so a section and
the combined figure at the end of the notebook cannot disagree about what a
family looks like.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .._panels import (
    CRITERION_REFERENCE_LABEL,
    criterion_curves,
    k_hat_heatmap,
    metric_legend,
    metric_lines,
)
from .._registry import PUBLISHED_RANDOM_STATE, SCENARIOS
from .._theme import FONT_SIZES, save_figure, theme_context
from ._benchmarking_examples import SCENARIO_TITLES, draw_example_row
from ._benchmarking_results import DEFAULT_METRICS
from ._paths import BENCHMARKING_DIR, figure_path
from ._scenario_overview import DIFFICULTY_AXIS_NAME, _axis_points, _x_label

# Seven methods' per-seed clouds at one x merge into a band; this spreads
# them far enough to read and not so far that a method's cloud sits over the
# next axis point's.
POINT_DODGE_FRACTION: float = 0.055


def figure_scenario_dashboard(
    name: str,
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    seed: int = 0,
    random_state: int = PUBLISHED_RANDOM_STATE,
    save: bool = False,
    out_dir: Path | None = None,
) -> Figure:
    """Build one scenario's dashboard.

    Parameters
    ----------
    name : str
        Registry key naming the scenario.
    results : DataFrame
        That scenario's artifact frame, in the unified schema.
    metrics : sequence of str
        Metric names, in legend order. Defaults to Fig 4's, so a section
        panel and its Fig 4 counterpart show the same series.
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

    # The example and criterion panels are simulated or read per registry
    # anchor; the curves come from the artifact. A run directory is
    # content-addressed on its config, so an older run whose axis has since
    # been redefined still sits on disk and still reads. The two halves would
    # then describe different experiments, with the x ticks taken from the
    # artifact hiding the disagreement.
    registered = list(scenario.axis.values)
    if values != registered:
        raise ValueError(
            f"Scenario {name!r} is registered on axis {scenario.axis.name}="
            f"{registered}, but the frame sweeps {values}. The example panels "
            "and the curves would describe different experiments. Re-run this "
            f"scenario: python -m benchmarks.run --scenario {name}"
        )

    with theme_context():
        n_cols = len(scenario.axis)
        fig = plt.figure(figsize=(3.9 * n_cols, 13.2))
        grid = fig.add_gridspec(
            4,
            n_cols,
            height_ratios=(1.0, 1.25, 1.15, 1.0),
            hspace=0.42,
            wspace=0.28,
        )

        example_axes = [fig.add_subplot(grid[0, i]) for i in range(n_cols)]
        ari_ax = fig.add_subplot(grid[1, :])
        criterion_axes = [fig.add_subplot(grid[2, i]) for i in range(n_cols)]
        heatmap_axes = [fig.add_subplot(grid[3, i]) for i in range(n_cols)]

        if axis_name == DIFFICULTY_AXIS_NAME:
            titles = [label.capitalize() for _, _, label in scenario.axis]
        else:
            titles = [
                f"{scenario.axis.name} = {value}" for _, value, _ in scenario.axis
            ]

        # Row 1 -- what the data looks like.
        draw_example_row(
            example_axes,
            scenario,
            seed=seed,
            random_state=random_state,
            titles=titles,
        )

        # Row 2 -- what each method scored.
        span = max(values) - min(values) if len(values) > 1 else 1.0
        metric_lines(
            ari_ax,
            results,
            metrics=metrics,
            x_col="axis_value",
            x_label=_x_label(name, axis_name),
            show_legend=False,
            show_points=True,
            dodge=POINT_DODGE_FRACTION * float(span),
        )
        # Ticks sit on the swept anchors, not on matplotlib's automatic
        # scale. A scaling family holds data at exactly three values, and
        # auto ticks put marks at 2000 and 8000 where nothing was measured.
        ari_ax.set_xticks(values)
        if axis_name == DIFFICULTY_AXIS_NAME:
            ari_ax.set_xticklabels([label.capitalize() for label in labels])

        # Rows 3 and 4 -- why that k, and how often.
        for index, axis_label in enumerate(labels):
            criterion_curves(
                criterion_axes[index],
                results,
                metrics=metrics,
                axis_label=axis_label,
                candidate_k=scenario.candidate_k,
                k_star=scenario.k_star,
                title=titles[index],
            )
            k_hat_heatmap(
                heatmap_axes[index],
                results,
                metrics=metrics,
                axis_label=axis_label,
                candidate_k=scenario.candidate_k,
                k_star=scenario.k_star,
            )
            if index:
                criterion_axes[index].set_ylabel("")
                heatmap_axes[index].set_yticklabels([])

        # The grey reference band has no entry in the shared foot legend --
        # metric_legend's grouping only classifies metric names, so a
        # free-form label like this one cannot surface through it, and a
        # second figure-level legend would break the "one shared legend"
        # contract. Naming it on the leftmost criterion panel alone, from
        # that panel's own handle, is what keeps the label appearing once
        # rather than once per axis point. criterion_curves omits the line
        # entirely when a cell has no classical-index rows, so this is a
        # no-op then.
        handles, texts = criterion_axes[0].get_legend_handles_labels()
        reference = [
            (handle, text)
            for handle, text in zip(handles, texts)
            if text == CRITERION_REFERENCE_LABEL
        ]
        if reference:
            ref_handle, ref_label = reference[0]
            criterion_axes[0].legend(
                [ref_handle],
                [ref_label],
                loc="lower center",
                frameon=False,
                fontsize=FONT_SIZES["legend"],
            )

        fig.suptitle(
            SCENARIO_TITLES.get(name, name), fontsize=FONT_SIZES["title"], y=0.995
        )
        metric_legend(fig, np.array([ari_ax]), metrics, y_offset=0.03)

        if save:
            save_figure(
                fig,
                figure_path(
                    f"scenario_dashboard_{name}.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig
