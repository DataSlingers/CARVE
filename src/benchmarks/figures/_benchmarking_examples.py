"""S1 Fig: example scatters for each scenario at each difficulty.

One row per scenario, one column per difficulty anchor, each panel a PCA
projection of one simulated dataset colored by the true labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from .._panels import cluster_color_map, scatter_clusters
from .._registry import SCENARIOS
from .._simulate import simulate
from .._theme import FONT_SIZES, save_figure, theme_context
from ._paths import BENCHMARKING_DIR, figure_path

DEFAULT_SCENARIOS = (
    "gaussians",
    "t_dist",
    "t_dist_noise",
    "circles",
    "moons",
    "swiss_rolls",
)

SCENARIO_TITLES = {
    "gaussians": "Gaussian Mixtures",
    "t_dist": "t-Distributed",
    "t_dist_noise": "t-Distributed + Nuisance Dims",
    "circles": "RFF Circles",
    "moons": "RFF Moons",
    "swiss_rolls": "RFF Swiss Rolls",
}


def figure_benchmarking_examples(
    *,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
    seed: int = 0,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Build S1 Fig.

    Parameters
    ----------
    scenarios : tuple of str
        Registry keys, one row each.
    seed : int
        Seed index passed through the standard derivation, so the panels show
        the same data the benchmark scored.
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
            for axis_idx, axis_value, axis_label in scenario.axis:
                ax = axes[row, axis_idx]
                benchmark_seed = seed + (axis_idx * 10000)
                X, y = simulate(
                    scenario,
                    axis_value=axis_value,
                    axis_label=axis_label,
                    seed=benchmark_seed,
                )
                Z = PCA(n_components=2, random_state=0).fit_transform(np.asarray(X))
                scatter_clusters(
                    ax,
                    Z,
                    np.asarray(y),
                    color_map=cluster_color_map(np.asarray(y)),
                    s=6.0,
                    alpha=0.7,
                    title=axis_label.capitalize() if row == 0 else None,
                )
                if axis_idx == 0:
                    ax.set_ylabel(
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
