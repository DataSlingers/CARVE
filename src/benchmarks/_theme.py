"""The single source of figure styling.

Everything that decides how a figure looks lives here: palette, rcParams,
spine treatment, font sizes, and saving. The code this replaces had four
uncoordinated palettes, so CARVE stability was drawn in two different greens
in the same paper, and a set_paper_style() that nothing ever called, so every
published figure used stock matplotlib defaults and pdf.fonttype 42 was never
applied.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.figure import Figure

# Okabe-Ito, which is colorblind-safe and already what two of the three
# previous definitions used for CARVE stability.
METRIC_COLORS: dict[str, str] = {
    "ari_stability_1se": "#009E73",
    "ari_stability": "#009E73",
    "ari_stability_quant": "#009E73",
    "ari_generalizability_1se": "#56B4E9",
    "ari_generalizability": "#56B4E9",
    "ari_generalizability_quant": "#56B4E9",
    "ari_average_1se": "#E69F00",
    "ari_average": "#E69F00",
    "ari_average_quant": "#E69F00",
    "consensus_pac_stability": "#8C8C8C",
    "consensus_gini_stability": "#6E6E6E",
    "consensus_ce_stability": "#4F4F4F",
    "accuracy_generalizability": "#7BC8F0",
    "silhouette": "#E8588C",
    "gap": "#0072B2",
    "davies_bouldin": "#D55E00",
    "calinski_harabasz": "#CC79A7",
    "baseline_oracle": "#000000",
}

FALLBACK_COLOR = "#7F7F7F"

CLUSTER_PALETTE: tuple[str, ...] = (
    "#009ADE",
    "#00CD6C",
    "#FF1F5B",
    "#AF58BA",
    "#F28522",
    "#A6761D",
    "#A0B1BA",
    "#1B9E77",
    "#7570B3",
    "#66A61E",
)

# CARVE's own plotting functions take a Matplotlib colormap *name* --
# ``palette: str = "Accent"``, consumed as ``plt.get_cmap(palette, n)`` -- so
# CLUSTER_PALETTE (a tuple of hex strings) cannot be passed to them as-is.
#
# ``plt.get_cmap`` accepts an already-built Colormap instance and returns it
# unchanged, but then ignores the requested ``n`` entirely (verified against
# the installed Matplotlib, not assumed from the ``str`` type hint). That
# breaks cross-panel color agreement: call sites that index the colormap by
# plain integer (plot_cluster_violin, plot_cluster_scatter) walk the palette
# in order, but plot_consensus_matrix's top color band goes through imshow,
# which normalizes the cluster index to a float in [0, 1] first -- and
# against an unresampled 10-color instance that lands on different, scattered
# palette entries than the direct-index panels for any cluster count under
# 10, so the same cluster gets two different colors across the figure.
#
# Registering the palette under a name and passing that name instead avoids
# this: ``plt.get_cmap(name, n)`` resamples to exactly n colors, so both
# indexing styles agree. Guarded so importing this module twice (e.g. a test
# re-import) does not raise on re-registration.
CLUSTER_CMAP_NAME: str = "carve_cluster"
_cluster_cmap = ListedColormap(list(CLUSTER_PALETTE), name=CLUSTER_CMAP_NAME)
if CLUSTER_CMAP_NAME not in mpl.colormaps:
    mpl.colormaps.register(_cluster_cmap)

FONT_SIZES: dict[str, float] = {
    "tick": 9.0,
    "legend": 9.0,
    "axis_label": 10.0,
    "title": 11.0,
    "panel_letter": 28.0,
}

RC_PARAMS: dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": FONT_SIZES["title"],
    "axes.labelsize": FONT_SIZES["axis_label"],
    "legend.fontsize": FONT_SIZES["legend"],
    "xtick.labelsize": FONT_SIZES["tick"],
    "ytick.labelsize": FONT_SIZES["tick"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    # Type 42 keeps text editable and selectable in the submitted PDF, which
    # PLOS requires. The previous code set this in a function nothing called.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_theme() -> None:
    """Apply the manuscript style to the global rcParams."""
    plt.rcParams.update(RC_PARAMS)


@contextmanager
def theme_context() -> Iterator[None]:
    """Apply the theme for the duration of a block, then restore."""
    with mpl.rc_context(RC_PARAMS):
        yield


def metric_color(metric: str) -> str:
    """Return the color for a metric, falling back to gray for unknown names."""
    return METRIC_COLORS.get(metric, FALLBACK_COLOR)


def cluster_colors(n: int) -> list[str]:
    """Return n cluster colors, agreeing with CARVE's own panels up to ten.

    For n <= len(CLUSTER_PALETTE), this matches ``plt.get_cmap(
    CLUSTER_CMAP_NAME, n)`` exactly -- the same colormap CARVE's own
    plotting methods resample when a caller passes them CLUSTER_CMAP_NAME as
    ``palette``. Two figures coloring the same (<=10) clusters must agree,
    and slicing CLUSTER_PALETTE directly (an earlier implementation) did
    not: matplotlib spreads n samples evenly across the full 10-color
    palette rather than taking its first n entries in order (n=3 lands on
    palette indices 0, 5 and 9, not 0, 1 and 2). Verified against the
    installed Matplotlib, not assumed.

    For n > len(CLUSTER_PALETTE), this cycles CLUSTER_PALETTE from the
    start (``CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]``) instead of
    continuing to resample the registered colormap. Resampling a 10-color
    map past 10 entries duplicates colors by construction -- there is no n
    above 10 where cluster identity can be fully distinguished by color --
    but a first attempt at this function that resampled unconditionally
    made those duplicates land on *adjacent* cluster indices (verified: at
    n=17, 7 of 16 adjacent pairs shared a color), which is worse than
    duplicates ten indices apart. Cycling instead means every repeat is
    exactly len(CLUSTER_PALETTE) apart, so two adjacent clusters are never
    the same color even though the palette is exhausted. This is a
    deliberate trade: a CARVE-drawn panel (which resamples internally via
    its own ``palette=CLUSTER_CMAP_NAME`` and cannot follow this cycling)
    and a benchmarks-drawn panel can disagree on colors above n=10, but
    both are already degenerate there, and legibility within one panel
    matters more than cross-panel agreement neither panel can fully honor
    anyway.
    """
    if n <= len(CLUSTER_PALETTE):
        cmap = plt.get_cmap(CLUSTER_CMAP_NAME, max(n, 1))
        return [to_hex(cmap(i)) for i in range(n)]
    return [to_hex(CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]) for i in range(n)]


def style_axes(ax: Axes, *, grid: bool = True) -> Axes:
    """Apply the spine and grid treatment to one axes.

    Replaces the spine block that was retyped in three places with three
    different alpha values.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_alpha(0.6)
    if grid:
        ax.grid(
            True, alpha=RC_PARAMS["grid.alpha"], linewidth=RC_PARAMS["grid.linewidth"]
        )
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    return ax


def save_figure(fig: Figure, path: Path, *, dpi: int = 300) -> Path:
    """Save a figure under its manuscript filename, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
