"""Fig 3 and S4 Fig: CARVE's own diagnostic output on a case study.

Panel A is the validation curves over k, B the consensus matrix at the
selected k, C the per-sample stability distribution. Both figures were
previously right-click-saved out of Jupyter with no savefig anywhere.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .._panels import carve_lines, cluster_color_map, panel_letter, scatter_clusters
from .._theme import FONT_SIZES, save_figure, theme_context
from ._case_study import CompositeInputs
from ._paths import CASE_STUDY_DIR, figure_path


def _config_id_at_k(carve: object, k: int) -> int:
    """Look up the config_id for one k.

    config_id is a join key linking estimator_results_ rows to the consensus
    matrices, never a positional index. Selecting a row and using its pandas
    label to index the matrices positionally is the mistake this guards
    against.
    """
    results = carve.estimator_results_
    match = results.loc[results["n_clusters"] == k, "config_id"]
    if match.empty:
        raise ValueError(
            f"No configuration at k={k}; available: "
            f"{sorted(results['n_clusters'].unique())}."
        )
    return int(match.iloc[0])


def carve_output_figure(
    inputs: CompositeInputs,
    *,
    marker_size: float,
    axis_labels: Sequence[str],
    save_name: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Draw the three-panel CARVE diagnostic figure."""
    carve = inputs.carve
    selected_k = int(carve.get_k(measure="stability", rule="1se"))
    config_id = _config_id_at_k(carve, selected_k)

    with theme_context():
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
        ax_a, ax_b, ax_c = axes

        carve_lines(ax_a, carve, title="Validation over $k$")

        matrix = np.asarray(carve.consensus_matrices_[config_id], dtype=float)
        image = ax_b.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
        ax_b.set_title(
            f"Consensus matrix ($k={selected_k}$)", fontsize=FONT_SIZES["title"]
        )
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        fig.colorbar(image, ax=ax_b, fraction=0.046, pad=0.04)

        gini = np.asarray(carve.stability_gini_scores_[config_id], dtype=float)
        scatter_clusters(
            ax_c,
            inputs.Z,
            inputs.carve_labels,
            color_map=cluster_color_map(inputs.carve_labels),
            s=marker_size,
            title="Sample-level stability",
            axis_labels=axis_labels,
        )
        ax_c.set_xlabel(
            f"median Gini = {np.median(gini):.3f}", fontsize=FONT_SIZES["axis_label"]
        )

        for letter, ax in zip("ABC", (ax_a, ax_b, ax_c)):
            panel_letter(ax, letter)

        fig.tight_layout()

        if save:
            save_figure(
                fig,
                figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir),
            )
    return fig


def figure_carve_output_klein(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build Fig 3."""
    return carve_output_figure(
        inputs,
        marker_size=20.0,
        axis_labels=("PC1", "PC2"),
        save_name="CARVE_output_klein.png",
        save=save,
        out_dir=out_dir,
    )


def figure_carve_output_levine(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build S4 Fig."""
    return carve_output_figure(
        inputs,
        marker_size=8.0,
        axis_labels=("t-SNE 1", "t-SNE 2"),
        save_name="CARVE_output_levine.png",
        save=save,
        out_dir=out_dir,
    )
