"""The Cusanovich mouse sci-ATAC case-study composite.

Panel F is the ARI comparison the Levine and hECA figures use: the number
this case study exists to report is CARVE's agreement with the reported
tissue labels against each CVI's, and an alluvial does not state it. Panels
A to C draw the source publication's own t-SNE coordinates, which
cell_metadata.txt ships and the loader carries through as
meta["source_tsne"], with 8-point markers because the study is Levine sized
rather than Klein sized.
"""

from pathlib import Path

from matplotlib.figure import Figure

from ._case_study import CompositeInputs, ari_panel, composite_figure

MARKER_SIZE = 8.0
AXIS_LABELS = ("t-SNE 1", "t-SNE 2")


def figure_cusanovich_results(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build the Cusanovich case-study figure."""
    return composite_figure(
        inputs,
        bottom_panel=ari_panel,
        marker_size=MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="cusanovich_results.png",
        save=save,
        out_dir=out_dir,
    )
