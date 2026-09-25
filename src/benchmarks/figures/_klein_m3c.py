"""The Klein M3C comparison, a supplementary figure.

Fig 5 is not modified. It is a published six-panel composite whose panels the
main text and S6 Text cite by letter, so a new panel there would mean
rewriting prose. This is a separate figure that reuses Fig 5's embedding and
color maps, so the two read together.

Three panels in every case: M3C's RCSI curve, its partition at its own
selected k, and the ARI panel. M3C's partition at the k CARVE selected is
reported in the SI table (m3c_ari_rows' default) and left out of the figure.

M3C runs on a subsample when the case study has more cells than its
vignette recommends (M3C_MAX_SAMPLES). Its scatter then draws those cells
only, and its ARI is taken over them; every other method's ARI is over every
cell, so M3C's rows name the subsample size.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import adjusted_rand_score

from .._m3c import M3CResult
from .._panels import (
    aligned_color_maps,
    ari_lollipop,
    m3c_lines,
    panel_letter,
    scatter_clusters,
)
from .._studies import STUDIES
from .._theme import save_figure, theme_context
from ._case_study import CompositeInputs, _align_to_reference, ari_table
from ._paths import CASE_STUDY_DIR, figure_path

SAVE_NAME = "klein_m3c.png"
MARKER_SIZE = 20.0
AXIS_LABELS = ("PC1", "PC2")

#: The k CARVE selects on Klein, read from STUDIES rather than restated here.
#: The SI table reports M3C's partition at this k so the two tools can be
#: compared at one granularity as well as at each one's own choice.
CARVE_K: int = int(STUDIES["klein"].reported_k)


def _m3c_cells(
    inputs: CompositeInputs, m3c: M3CResult
) -> tuple[np.ndarray, np.ndarray]:
    """The reported labels and embedding rows that M3C's labels line up with."""
    y = np.asarray(inputs.y)
    Z = np.asarray(inputs.Z)
    if m3c.rows is not None:
        y, Z = y[m3c.rows], Z[m3c.rows]
    n_labels = len(m3c.labels[m3c.selected_k])
    if n_labels != len(y):
        raise ValueError(
            f"M3C returned {n_labels} labels but its recorded rows cover "
            f"{len(y)} cells, so the labels cannot be matched to cells."
        )
    return y, Z


def _cells_note(m3c: M3CResult) -> str:
    return "" if m3c.rows is None else f"{len(m3c.rows):,} cells"


def m3c_ari_rows(
    inputs: CompositeInputs, m3c: M3CResult, *, at_carve_k: bool = True
) -> list[dict[str, Any]]:
    """M3C's ARI rows: at its own selection and, unless at_carve_k=False, at CARVE's k.

    Returns one row when M3C also selects CARVE's k, since two rows would
    then be the same partition under two names. A run on a subsample is
    scored against the reported labels of its own cells.
    """
    y, _ = _m3c_cells(inputs, m3c)
    ks = [m3c.selected_k]
    if at_carve_k and m3c.selected_k != CARVE_K:
        ks.append(CARVE_K)
    cells = _cells_note(m3c)
    rows = []
    for k in ks:
        labels = _align_to_reference(np.asarray(m3c.labels[k]), y)
        details = [] if k == m3c.selected_k else [f"at $k={k}$"]
        details += [cells] if cells else []
        rows.append(
            {
                "method": f"M3C ({', '.join(details)})" if details else "M3C",
                "metric": "m3c_rcsi",
                "ari": float(adjusted_rand_score(y, labels)),
                "k": int(k),
            }
        )
    return rows


def figure_klein_m3c(
    inputs: CompositeInputs,
    m3c: M3CResult,
    *,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Draw the three-panel Klein M3C comparison."""
    with theme_context():
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

        m3c_lines(
            axes[0],
            m3c.scores,
            selected_k=m3c.selected_k,
            title="M3C: Relative Cluster Stability Index",
        )
        panel_letter(axes[0], "A")

        y, Z = _m3c_cells(inputs, m3c)
        labels = _align_to_reference(np.asarray(m3c.labels[m3c.selected_k]), y)
        # Colored from M3C's own aligned labels (aligned_color_maps), not
        # from composite_color_maps' comparison_cmap -- that map is keyed on
        # the CVI comparison partition alone, so any M3C cluster id absent
        # from it would fall through scatter_clusters' color_map.get() and
        # render grey.
        _, m3c_cmap = aligned_color_maps(y, labels)
        cells = _cells_note(m3c)
        scatter_clusters(
            axes[1],
            Z,
            labels,
            color_map=m3c_cmap,
            s=MARKER_SIZE,
            axis_labels=AXIS_LABELS,
            title=f"M3C clustering ($k={m3c.selected_k}$"
            + (f", {cells})" if cells else ")"),
        )
        panel_letter(axes[1], "B")

        ari_lollipop(
            axes[2],
            ari_table(inputs, extra_rows=m3c_ari_rows(inputs, m3c, at_carve_k=False)),
            title="Agreement with Reported Labels (ARI)",
        )
        panel_letter(axes[2], "C")

        fig.tight_layout()

    if save:
        save_figure(fig, figure_path(SAVE_NAME, subdir=CASE_STUDY_DIR, out_dir=out_dir))
    return fig
