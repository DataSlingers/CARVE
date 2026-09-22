"""The Klein M3C comparison, a supplementary figure.

Fig 5 is not modified. It is a published six-panel composite whose panels the
main text and S6 Text cite by letter, so a new panel there would mean
rewriting prose. This is a separate figure that reuses Fig 5's embedding and
color maps, so the two read together.

Three panels in every case. M3C's partition at k=4, which answers what M3C
says at the k CARVE selected, is a row in the ARI panel rather than a fourth
scatter: drawing it conditionally would give the figure a panel count that
depends on the result, and drawing it unconditionally would duplicate panel B
whenever M3C also selects 4.
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
#: M3C's partition at this k is reported so the two tools can be compared at
#: one granularity as well as at each one's own choice.
CARVE_K: int = int(STUDIES["klein"].reported_k)


def m3c_ari_rows(inputs: CompositeInputs, m3c: M3CResult) -> list[dict[str, Any]]:
    """M3C's rows for the ARI panel: at its own selection, and at CARVE's k.

    Returns one row when M3C also selects CARVE's k, since two rows would
    then be the same partition under two names.
    """
    ks = [m3c.selected_k] if m3c.selected_k == CARVE_K else [m3c.selected_k, CARVE_K]
    rows = []
    for k in ks:
        labels = _align_to_reference(np.asarray(m3c.labels[k]), inputs.y)
        rows.append(
            {
                "method": "M3C" if k == m3c.selected_k else f"M3C (at $k={k}$)",
                "metric": "m3c_rcsi",
                "ari": float(adjusted_rand_score(inputs.y, labels)),
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

        labels = _align_to_reference(np.asarray(m3c.labels[m3c.selected_k]), inputs.y)
        # Colored from M3C's own aligned labels (aligned_color_maps), not
        # from composite_color_maps' comparison_cmap -- that map is keyed on
        # the CVI comparison partition alone, so any M3C cluster id absent
        # from it would fall through scatter_clusters' color_map.get() and
        # render grey.
        _, m3c_cmap = aligned_color_maps(inputs.y, labels)
        scatter_clusters(
            axes[1],
            inputs.Z,
            labels,
            color_map=m3c_cmap,
            s=MARKER_SIZE,
            axis_labels=AXIS_LABELS,
            title=f"M3C clustering ($k={m3c.selected_k}$)",
        )
        panel_letter(axes[1], "B")

        ari_lollipop(
            axes[2],
            ari_table(inputs, extra_rows=m3c_ari_rows(inputs, m3c)),
            title="Agreement with Reported Labels (ARI)",
        )
        panel_letter(axes[2], "C")

        fig.tight_layout()

    if save:
        save_figure(fig, figure_path(SAVE_NAME, subdir=CASE_STUDY_DIR, out_dir=out_dir))
    return fig
