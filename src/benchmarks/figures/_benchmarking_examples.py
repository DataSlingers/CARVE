"""S1 Fig: example scatters for each scenario at each difficulty.

One row per scenario, one column per difficulty anchor, each panel a PCA
projection of one simulated dataset colored by the true labels.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from .._panels import cluster_color_map, scatter_clusters
from .._registry import PUBLISHED_RANDOM_STATE, SCENARIOS
from .._simulate import simulate
from .._theme import FONT_SIZES, save_figure, theme_context
from .._types import Scenario
from ._paths import BENCHMARKING_DIR, figure_path

# The canonical reading order for the six difficulty families, and the order
# the published S1 Fig and Fig 4 both present them in. Fig 4 orders its
# panels by this too, so the two figures cannot disagree about where a family
# sits, and neither depends on the order a caller happened to build its
# mapping in.
DEFAULT_SCENARIOS = (
    "gaussians",
    "t_dist",
    "t_dist_noise",
    "swiss_rolls",
    "circles",
    "moons",
)

SCENARIO_TITLES = {
    "gaussians": "Gaussian Mixtures",
    "t_dist": "t-Distributed",
    "t_dist_noise": "t-Distributed + Nuisance Dims",
    "circles": "RFF Circles",
    "moons": "RFF Moons",
    "swiss_rolls": "RFF Swiss Rolls",
    "gaussians_dimensionality": "Gaussian Mixtures (Dimensionality)",
    "gaussians_samples": "Gaussian Mixtures (Sample Size)",
}


def draw_example_row(
    axes: Sequence[Axes],
    scenario: Scenario,
    *,
    seed: int = 0,
    random_state: int = PUBLISHED_RANDOM_STATE,
    titles: Sequence[str] | None = None,
    s: float = 6.0,
    alpha: float = 0.7,
) -> None:
    """Scatter one simulated dataset per axis point onto a row of axes.

    The seed each panel is drawn at is derived exactly as _run.py's run_cell
    derives it -- seed + axis_idx * 10000 + random_state -- so a panel shows
    the data the benchmark actually scored. Both the S1 grid and the
    per-scenario overview draw their examples through here rather than
    repeating that arithmetic, which is the kind of duplication that lets one
    of two figures drift onto different data than the other.

    Parameters
    ----------
    axes : sequence of Axes
        One axes per axis point, indexed by the axis index.
    scenario : Scenario
        The experiment definition to simulate from.
    seed : int
        Seed-loop index, combined with axis_idx and random_state as above.
    random_state : int
        The run's random_state term in that derivation.
    titles : sequence of str or None
        Per-panel titles, aligned with the axis. None draws no titles.
    s, alpha
        Scatter marker size and opacity.
    """
    for axis_idx, axis_value, axis_label in scenario.axis:
        benchmark_seed = seed + (axis_idx * 10000) + random_state
        X, y = simulate(
            scenario,
            axis_value=axis_value,
            axis_label=axis_label,
            seed=benchmark_seed,
        )
        Z = PCA(n_components=2, random_state=0).fit_transform(np.asarray(X))
        scatter_clusters(
            axes[axis_idx],
            Z,
            np.asarray(y),
            color_map=cluster_color_map(np.asarray(y)),
            s=s,
            alpha=alpha,
            title=None if titles is None else titles[axis_idx],
        )


def figure_benchmarking_examples(
    *,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
    seed: int = 0,
    random_state: int = PUBLISHED_RANDOM_STATE,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Build S1 Fig.

    Parameters
    ----------
    scenarios : tuple of str
        Registry keys, one row each.
    seed : int
        Seed-loop index. Combined with axis_idx and random_state exactly as
        _run.py's run_cell does (seed + axis_idx * 10000 + random_state), so
        the panels show the same data the benchmark scored.
    random_state : int
        The run's random_state term in that derivation. Defaults to
        PUBLISHED_RANDOM_STATE, the value the published benchmark used.
    save : bool
        Write the file. False returns the figure without touching disk.
    out_dir : Path or None
        Override the destination directory.
    """
    with theme_context():
        n_rows = len(scenarios)
        fig, axes = plt.subplots(n_rows, 3, figsize=(11.0, 3.1 * n_rows), squeeze=False)

        for row, scenario_name in enumerate(scenarios):
            scenario = SCENARIOS[scenario_name]
            draw_example_row(
                axes[row],
                scenario,
                seed=seed,
                random_state=random_state,
                titles=(
                    [label.capitalize() for _, _, label in scenario.axis]
                    if row == 0
                    else None
                ),
            )
            axes[row, 0].set_ylabel(
                SCENARIO_TITLES.get(scenario_name, scenario_name),
                fontsize=FONT_SIZES["axis_label"],
            )

        fig.tight_layout()

        if save:
            save_figure(
                fig,
                figure_path(
                    "benchmarking_examples.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig
