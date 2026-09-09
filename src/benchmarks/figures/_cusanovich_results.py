"""The Cusanovich mouse sci-ATAC case-study composite.

Panel F is an alluvial linking the CARVE clustering, the reported labels and
the CVI clustering, the same shape the Klein figure uses. Panels A to C use
the first two LSI components, which is the embedding the source publication's
own pipeline produces, and 8-point markers because the study is Levine sized
rather than Klein sized.
"""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._panels import alluvial
from ._case_study import CompositeInputs, composite_color_maps, composite_figure

MARKER_SIZE = 8.0
AXIS_LABELS = ("LSI 1", "LSI 2")


def _alluvial_panel(ax: Axes, inputs: CompositeInputs) -> Axes:
    # The same three maps the scatter panels use, so a cluster keeps one
    # color down the whole figure.
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
        true_title="Reported Tissue",
    )


def figure_cusanovich_results(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build the Cusanovich case-study figure."""
    return composite_figure(
        inputs,
        bottom_panel=_alluvial_panel,
        marker_size=MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="cusanovich_results.png",
        save=save,
        out_dir=out_dir,
    )
