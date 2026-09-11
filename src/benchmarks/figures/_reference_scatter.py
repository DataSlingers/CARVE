"""The case-study overview: a two-dimensional embedding colored by reference.

Not a manuscript figure. Each case-study notebook shows this right after its
data loads, before anything is fit, so the reader sees what CARVE will be
scored against. It is one panel drawn the way the composites draw their panel
A, and the notebook then hands the same embedding to prepare_composite, so the
data is shown one way throughout.

The embedding is the caller's. Which one is the right one follows the data
source (the Cusanovich atlas ships its own t-SNE; hECA ships nothing, so the
notebook computes a UMAP), and that choice belongs in the notebook, next to
the load, not here.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .._panels import aligned_color_maps, axis_arrows, scatter_clusters
from .._theme import FONT_SIZES, save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path

# Sized to match one panel of the 16-by-16 inch composites (each of panels A
# to C spans a third of that width), so a marker size chosen for a composite
# reads at the same density here.
FIGSIZE = (6.0, 6.0)
DEFAULT_MARKER_SIZE = 8.0


def _subsample(n: int, max_points: int | None, random_state: int) -> np.ndarray:
    """One sorted index shared by Z and y.

    The same pattern as _heca_results.subsample_inputs: a single index so a
    cell keeps its position and its label together, sorted so the drawing
    order is the input order.
    """
    if max_points is None or max_points >= n:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=int(max_points), replace=False))


def figure_reference_scatter(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    axis_labels: Sequence[str],
    color_map: Mapping[Any, str] | None = None,
    title: str | None = None,
    max_points: int | None = None,
    random_state: int = 42,
    marker_size: float = DEFAULT_MARKER_SIZE,
    save: bool = True,
    out_dir: Path | None = None,
    save_name: str = "reference_scatter.png",
) -> Figure:
    """Scatter a two-dimensional embedding colored by the reference label.

    Parameters
    ----------
    Z : ndarray of shape (n_cells, 2)
        The embedding to draw. Row i is the same cell as row i of ``y``.
    y : array-like of shape (n_cells,)
        Reference label per cell: the study's tissue or organ.
    axis_labels : sequence of two str
        Names for the corner axis arrows, e.g. ("t-SNE 1", "t-SNE 2").
    color_map : mapping or None
        Label to color. Defaults to the reported-label map aligned_color_maps
        builds from ``y`` alone, which is the map the composite's panel A
        uses, so a label keeps its color from this figure to that one.
    title : str or None
        Panel title.
    max_points : int or None
        Draw at most this many cells, chosen without replacement with
        ``random_state``. None draws every cell.
    random_state : int
        Seed for the ``max_points`` subsample only.
    marker_size : float
        Scatter marker size, in points squared. Pass the composite's own
        marker size so the two figures agree.
    save : bool
        Write the figure under ``save_name``.
    out_dir : Path or None
        Directory to write into; defaults to the case-study figure directory.
    save_name : str
        Filename to write. Both case-study notebooks write into the same
        directory at publication scale, so each passes its own name.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    if Z.ndim != 2 or Z.shape[1] != 2 or Z.shape[0] != y.shape[0]:
        raise ValueError(
            f"Z must be an (n_cells, 2) embedding aligned with y; got Z of shape "
            f"{Z.shape} and y of length {y.shape[0]}."
        )

    if color_map is None:
        (color_map,) = aligned_color_maps(y)

    idx = _subsample(Z.shape[0], max_points, random_state)

    with theme_context():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        scatter_clusters(
            ax, Z[idx], y[idx], color_map=color_map, s=marker_size, title=title
        )
        axis_arrows(ax, axis_labels)

        # Legend entries in sorted order of the raw label values, which is
        # the order aligned_color_maps assigns the palette in, so the legend
        # walks the palette in sequence. scatter_clusters labels each
        # collection str(label), so sorting those strings instead would put
        # integer labels out of sequence ("1", "10", "2").
        handle_for = dict(zip(*reversed(ax.get_legend_handles_labels())))
        categories = sorted(set(y[idx].tolist()))
        handles = [handle_for[str(category)] for category in categories]
        labels = [str(category) for category in categories]
        # Hung from the figure's bottom edge, so however many rows the
        # labels need (13 tissues take four) the legend grows downward into
        # the margin save_figure's tight bounding box picks up, never upward
        # into the axes and the corner arrows. At most four columns, filled
        # evenly: five organs read as 2+2+1 across three columns rather than
        # 4+1 across four.
        n_rows = -(-len(labels) // 4)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=-(-len(labels) // n_rows),
            frameon=False,
            fontsize=FONT_SIZES["legend"],
        )
        fig.tight_layout()

        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
