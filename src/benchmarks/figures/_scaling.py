"""The two scaling figures: ARI and runtime against the swept axis.

The notebook cell this replaces carried its own geometry constants and an
ELEMENT_SCALE fudge factor whose only purpose was to make line and marker
widths match Fig 4 by eye. With a single theme they match by construction, so
the fudge is gone.

The moons panels commented out in that cell are not ported; the moons scaling
experiment is dropped.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._panels import grouped_legend, metric_lines, runtime_lines
from .._theme import save_figure, theme_context
from ._paths import BENCHMARKING_DIR, figure_path

SCALING_PANELS: tuple[tuple[str, str], ...] = (
    ("gaussians_samples", "Number of samples (n)"),
    ("gaussians_dimensionality", "Number of features (p)"),
)

DEFAULT_METRICS: tuple[str, ...] = (
    "baseline_oracle",
    "ari_stability_1se",
    "ari_generalizability_1se",
)


def _panel_grid(n_panels: int) -> tuple[Figure, list]:
    """One row of panels sized from the theme, not from hand-tuned margins."""
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.0), squeeze=False)
    return fig, axes


def _ordered_panels(available: Mapping[str, pd.DataFrame]) -> list[tuple[str, str]]:
    panels = [(name, label) for name, label in SCALING_PANELS if name in available]
    if not panels:
        raise ValueError(
            "Need at least one scenario to draw. Expected keys from "
            f"{[name for name, _ in SCALING_PANELS]}, got {sorted(available)}."
        )
    return panels


def figure_scaling_ari(
    results_by_scenario: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    k_star: int = 5,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """ARI at the selected k against each scaling axis."""
    panels = _ordered_panels(results_by_scenario)

    with theme_context():
        fig, axes = _panel_grid(len(panels))
        for index, (name, x_label) in enumerate(panels):
            ax = axes[0, index]
            metric_lines(
                ax,
                results_by_scenario[name],
                metrics=metrics,
                x_col="axis_value",
                x_label=x_label,
                show_legend=False,
            )
            if index != 0:
                ax.set_ylabel("")

        fig.tight_layout()
        grouped_legend(fig, axes, y_offset=0.10)

        if save:
            save_figure(
                fig,
                figure_path(
                    f"paper_fig_scaling_ari_k{k_star}.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig


def figure_scaling_runtime(
    runtimes_by_scenario: Mapping[str, pd.DataFrame],
    *,
    k_star: int = 5,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """CARVE wall-clock against each scaling axis, on a log y axis.

    Two curves per panel, one per timed CARVE mode, matching the published
    figure. Those timings come from the extra mode-specific fits the runner
    performs for the scenarios listed in TIMED_SCENARIOS; a scenario without
    them records nan and its series is skipped rather than drawn at zero.
    """
    panels = _ordered_panels(runtimes_by_scenario)

    with theme_context():
        fig, axes = _panel_grid(len(panels))
        for index, (name, x_label) in enumerate(panels):
            ax = axes[0, index]
            runtime_lines(
                ax,
                runtimes_by_scenario[name],
                x_col="axis_value",
                x_label=x_label,
                yscale="log",
                show_legend=False,
            )
            if index != 0:
                ax.set_ylabel("")

        fig.tight_layout()
        grouped_legend(fig, axes, y_offset=0.10)

        if save:
            save_figure(
                fig,
                figure_path(
                    f"paper_fig_scaling_runtime_k{k_star}.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig
