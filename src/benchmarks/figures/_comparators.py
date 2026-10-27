"""The case-study comparison with M3C and SC3, a supplementary figure.

One figure per case study, each external tool at the k it selects itself:
M3C by its RCSI maximum, SC3 by its Tracy-Widom estimate. No row or panel
shows a tool at a k chosen by anything else.

A and B are M3C's RCSI curve and its partition, C and D SC3's spectrum and
its partition, and E the ARI of every method's selection against the
reported labels. M3C runs on a subsample when the case study has more cells
than its vignette recommends (M3C_MAX_SAMPLES); its scatter then draws those
cells, its ARI is over them, and its row names the subsample size. SC3
clusters every cell.
"""

from collections.abc import Sequence
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
    axis_arrows,
    m3c_lines,
    panel_letter,
    sc3_spectrum,
    scatter_clusters,
)
from .._sc3 import SC3Result
from .._theme import (
    PRINT_FONT_SIZES,
    PRINT_WIDTH_IN,
    print_axes_text,
    print_free_text,
    print_theme_context,
    save_figure,
    thin_lines,
)
from ._case_study import (
    _PRINT_CURVE_SCALE,
    _PRINT_MARKER_SCALE,
    CompositeInputs,
    _align_to_reference,
    ari_table,
)
from ._paths import CASE_STUDY_DIR, figure_path

#: Three panel rows at the print width; under PRINT_MAX_HEIGHT_IN.
HEIGHT_IN = 8.0

#: Left edge of panel E, in figure fractions, wide enough for its longest label.
_E_LEFT = 0.24


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


def m3c_ari_rows(inputs: CompositeInputs, m3c: M3CResult) -> list[dict[str, Any]]:
    """M3C's ARI row at its own selection, scored over the cells it clustered."""
    y, _ = _m3c_cells(inputs, m3c)
    method = "M3C" if m3c.rows is None else f"M3C ({len(m3c.rows):,} cells)"
    return [
        {
            "method": method,
            "metric": "m3c_rcsi",
            "ari": float(adjusted_rand_score(y, m3c.labels[m3c.selected_k])),
            "k": int(m3c.selected_k),
        }
    ]


def sc3_ari_rows(inputs: CompositeInputs, sc3: SC3Result) -> list[dict[str, Any]]:
    """SC3's ARI row at its own Tracy-Widom estimate, over every cell."""
    y = np.asarray(inputs.y)
    if len(sc3.labels) != len(y):
        raise ValueError(
            f"SC3 returned {len(sc3.labels)} labels for {len(y)} cells; it "
            "clusters every cell, so the two must match."
        )
    return [
        {
            "method": "SC3",
            "metric": "sc3",
            "ari": float(adjusted_rand_score(y, sc3.labels)),
            "k": int(sc3.k),
        }
    ]


def figure_comparators(
    inputs: CompositeInputs,
    m3c: M3CResult,
    sc3: SC3Result,
    *,
    axis_labels: Sequence[str],
    marker_size: float,
    save_name: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Draw the five-panel comparison of CARVE, M3C and SC3 for one case study.

    axis_labels and marker_size are the ones the notebook passes to
    figure_reference_scatter for the same embedding. marker_size is a
    screen-scale area and is scaled to the print width here, as
    composite_figure scales its own.
    """
    y = np.asarray(inputs.y)
    Z = np.asarray(inputs.Z)
    s = marker_size * _PRINT_MARKER_SCALE

    with print_theme_context():
        fig = plt.figure(figsize=(PRINT_WIDTH_IN, HEIGHT_IN))
        grid = fig.add_gridspec(
            3,
            2,
            height_ratios=(1.0, 1.0, 0.9),
            hspace=0.5,
            wspace=0.28,
            left=0.08,
            right=0.98,
            top=0.96,
            bottom=0.06,
        )
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        ax_c = fig.add_subplot(grid[1, 0])
        ax_d = fig.add_subplot(grid[1, 1])
        ax_e = fig.add_subplot(grid[2, :])
        # E's y tick labels ("Calinski Harabasz", "M3C (1,000 cells)") are
        # wider than the shared left margin; without room for them the tight
        # bbox grows past PRINT_WIDTH_IN.
        box = ax_e.get_position()
        ax_e.set_position([_E_LEFT, box.y0, box.x1 - _E_LEFT, box.height])

        m3c_lines(
            ax_a,
            m3c.scores,
            selected_k=m3c.selected_k,
            title="M3C: Relative Cluster Stability Index",
        )

        y_m3c, Z_m3c = _m3c_cells(inputs, m3c)
        m3c_labels = _align_to_reference(np.asarray(m3c.labels[m3c.selected_k]), y_m3c)
        # Colored from M3C's own aligned labels; composite_color_maps is keyed
        # on the CVI partition and would leave unmatched ids grey.
        _, m3c_cmap = aligned_color_maps(y_m3c, m3c_labels)
        scatter_clusters(
            ax_b,
            Z_m3c,
            m3c_labels,
            color_map=m3c_cmap,
            s=s,
            title=f"M3C clustering (k={m3c.selected_k}, {len(y_m3c):,} cells)",
        )
        axis_arrows(ax_b, axis_labels)

        sc3_spectrum(
            ax_c,
            sc3.spectrum,
            bound=sc3.bound,
            k=sc3.k,
            title="SC3: Tracy-Widom Estimate of k",
        )

        sc3_labels = _align_to_reference(np.asarray(sc3.labels), y)
        _, sc3_cmap = aligned_color_maps(y, sc3_labels)
        scatter_clusters(
            ax_d,
            Z,
            sc3_labels,
            color_map=sc3_cmap,
            s=s,
            title=f"SC3 clustering (k={sc3.k}, {len(y):,} cells)",
        )
        axis_arrows(ax_d, axis_labels)

        table = ari_table(
            inputs, extra_rows=m3c_ari_rows(inputs, m3c) + sc3_ari_rows(inputs, sc3)
        )
        ari_lollipop(ax_e, table, title="Agreement with Reported Labels (ARI)")

        thin_lines(ax_a, _PRINT_CURVE_SCALE)
        for ax in fig.get_axes():
            print_axes_text(ax)
            print_free_text(ax)
        for letter, ax in zip("ABCDE", (ax_a, ax_b, ax_c, ax_d, ax_e)):
            panel_letter(ax, letter, fontsize=PRINT_FONT_SIZES["panel_letter"])

        # Saved inside the theme so savefig.pad_inches is the print theme's;
        # outside it the default padding widens the file past PRINT_WIDTH_IN.
        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )

    return fig
