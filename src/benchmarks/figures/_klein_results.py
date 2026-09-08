"""Fig 5: the Klein case study.

Panel F is an alluvial linking the CARVE clustering, the reported labels, and
the CVI clustering. Panels A to C use a PCA embedding and 20-point markers.
"""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._panels import alluvial
from ._case_study import CompositeInputs, composite_color_maps, composite_figure

MARKER_SIZE = 20.0
AXIS_LABELS = ("PC1", "PC2")


def _alluvial_panel(ax: Axes, inputs: CompositeInputs) -> Axes:
    # The same three maps the scatter panels above use, so a cluster keeps
    # one color down the whole figure.
    true_cmap, carve_cmap, comparison_cmap = composite_color_maps(inputs)
    return alluvial(
        ax,
        inputs.y,
        inputs.carve_labels,
        inputs.comparison_labels,
        left_cmap=carve_cmap,
        right_cmap=comparison_cmap,
        true_cmap=true_cmap,
        left_title="CARVE",
        right_title=f"CVI ({inputs.comparison_name})",
        true_title="Reported Label",
    )


def figure_klein_results(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build Fig 5."""
    return composite_figure(
        inputs,
        bottom_panel=_alluvial_panel,
        marker_size=MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="klein_results.png",
        save=save,
        out_dir=out_dir,
    )
