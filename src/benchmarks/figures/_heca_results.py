"""The hECA large-scale case-study composite.

Two deviations from the other case-study figures, both forced by the data.
The scatter panels are drawn on a declared fixed subsample, because half a
million points do not rasterize legibly and an undeclared crop would be worse
than a declared sample. Panel F is the ARI comparison the Levine figure uses
rather than an alluvial, which is unreadable at roughly 20 cell types.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._case_study import CompositeInputs, ari_panel, composite_figure

MARKER_SIZE = 3.0
AXIS_LABELS = ("PC1", "PC2")
DEFAULT_SCATTER_SUBSAMPLE = 50_000


def subsample_inputs(
    inputs: CompositeInputs, *, size: int, random_state: int = 42
) -> CompositeInputs:
    """Reduce every per-cell array with one shared index.

    One index for all of them, so a cell keeps its label, its cluster and its
    embedding position together. Subsampling each array separately would
    decouple them and no shape check would notice.
    """
    n = np.asarray(inputs.y).shape[0]
    if size >= n:
        return inputs

    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(n, size=int(size), replace=False))

    return replace(
        inputs,
        X=np.asarray(inputs.X)[idx],
        y=np.asarray(inputs.y)[idx],
        Z=np.asarray(inputs.Z)[idx],
        carve_labels=np.asarray(inputs.carve_labels)[idx],
        comparison_labels=np.asarray(inputs.comparison_labels)[idx],
    )


def figure_heca_results(
    inputs: CompositeInputs,
    *,
    scatter_subsample: int = DEFAULT_SCATTER_SUBSAMPLE,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Build the hECA case-study figure.

    Only the declared scatter subsample goes to composite_figure, and only
    panels A to C read Z/y/carve_labels/comparison_labels from it. Panel F's
    ARI table is built from the full, unsubsampled ``inputs`` and closed
    over here -- best_df's CVI rows are already computed at full scale (the
    hECA publication scale loads every cell), so scoring CARVE against a
    scatter-legibility subsample would silently compare the two methods on
    different populations.
    """

    def _bottom_panel(ax: Axes, _subsampled: CompositeInputs) -> Axes:
        return ari_panel(ax, inputs)

    return composite_figure(
        subsample_inputs(inputs, size=scatter_subsample),
        bottom_panel=_bottom_panel,
        marker_size=MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="heca_results.png",
        save=save,
        out_dir=out_dir,
    )
