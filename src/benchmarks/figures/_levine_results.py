"""Fig 6: the Levine 32-dimension case study.

Panel F is an ARI lollipop against the reported labels. Panels A to C use the
precomputed t-SNE embedding carried on CompositeInputs.Z and 8-point markers,
because the dataset has an order of magnitude more cells than Klein.
"""

from pathlib import Path

from matplotlib.figure import Figure

from ._case_study import CompositeInputs, ari_panel, composite_figure

MARKER_SIZE = 8.0
AXIS_LABELS = ("t-SNE 1", "t-SNE 2")


def figure_levine_results(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build Fig 6."""
    return composite_figure(
        inputs,
        bottom_panel=ari_panel,
        marker_size=MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="levine_results.png",
        save=save,
        out_dir=out_dir,
    )
