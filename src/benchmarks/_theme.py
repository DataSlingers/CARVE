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
    # The four classical indices, in the published Fig 4's hues: pink,
    # purple, red, orange. These are the literal values the figure was drawn
    # with (lines_pallette_contrastive_other in the pre-rebuild plotting
    # module), not an approximation of them.
    #
    # That module assigned the four positionally, in whatever order a caller
    # listed its metrics, so the published figures disagree with each other
    # about which index is which color: Fig 4 draws Gap orange and
    # Davies-Bouldin purple, while the Klein and Levine panels draw Gap
    # purple and Davies-Bouldin red. One index cannot have two colors across
    # one paper. Fig 4's assignment is the one adopted here, so the case
    # study panels change rather than Fig 4.
    "silhouette": "#E0457B",
    "davies_bouldin": "#A8389E",
    "calinski_harabasz": "#D6292E",
    "gap": "#F28522",
    # Grey, not black: the published figures (Fig 4, S2 Fig) draw the oracle
    # reference as a grey dashed line, distinct from the black axis text and
    # spines it would otherwise be confused with.
    "baseline_oracle": "#595959",
}

FALLBACK_COLOR = "#7F7F7F"

# Matplotlib's tab10, which is what the published case-study composites draw
# their clusters in: the Klein figure's four timepoints are tab10's first
# four hues. Spelled out rather than read off plt.get_cmap("tab10") so the
# values are greppable and cannot shift under a Matplotlib release.
#
# tab10's eighth entry is the same grey as FALLBACK_COLOR. That only matters
# for a figure with at least eight clusters *and* a color map missing an
# entry, which no current figure has; if one appears, the fallback is what
# should move, not the palette.
CLUSTER_PALETTE: tuple[str, ...] = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
)

# The thin outline on scatter markers and the corner axis arrows both read as
# frame rather than as data, so they take the axis text's black instead of any
# palette hue.
FOREGROUND_COLOR: str = "#000000"

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
# Handing those call sites ``cluster_cmap(n)`` -- an instance carrying
# exactly the n colors they need -- avoids this: with n colors for n
# clusters, index i and the imshow-normalized i land on the same entry, so
# both indexing styles agree and both equal ``cluster_colors(n)``.
#
# Passing CLUSTER_CMAP_NAME and letting Matplotlib resample the ten-color map
# to n (what this replaces) also made the two styles agree, but on the wrong
# colors: resampling spreads n samples across the whole palette rather than
# taking its first n, so four clusters came out palette entries 0, 3, 6 and 9
# instead of 0 to 3. The name stays registered for the call sites that color
# by *estimator* rather than by cluster (plot_metric_over_n_clusters), where
# a stable ten-color map is what is wanted and the count of clusters is
# irrelevant. Guarded so importing this module twice (e.g. a test re-import)
# does not raise on re-registration.
CLUSTER_CMAP_NAME: str = "carve_cluster"
_cluster_cmap = ListedColormap(list(CLUSTER_PALETTE), name=CLUSTER_CMAP_NAME)
if CLUSTER_CMAP_NAME not in mpl.colormaps:
    mpl.colormaps.register(_cluster_cmap)

# Preprocessing pipelines in the Cusanovich figure's per-pipeline panel.
# CARVE's plot_metric_by_pipeline takes a colormap name and samples it at
# evenly spaced points, one per pipeline in sorted label order, so a
# ListedColormap of exactly these colors gives each of up to four pipelines
# its own entry. Okabe-Ito hues and tab10's brown, none of them a
# METRIC_COLORS value, so a pipeline line is never read as a criterion.
PIPELINE_COLORS: tuple[str, ...] = ("#0072B2", "#D55E00", "#CC79A7", "#8C564B")
PIPELINE_CMAP_NAME: str = "carve_pipeline"
if PIPELINE_CMAP_NAME not in mpl.colormaps:
    mpl.colormaps.register(
        ListedColormap(list(PIPELINE_COLORS), name=PIPELINE_CMAP_NAME)
    )

# The per-pipeline panel draws both criteria for every pipeline on one axes;
# color carries the pipeline, so line style carries the criterion.
MEASURE_LINESTYLES: dict[str, str] = {"stability": "-", "generalizability": "--"}

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


# Solid is reserved for CARVE's own selection curves. Every reference series
# a figure draws beside them -- the oracle baseline and the four classical
# indices -- is dashed, so the two families separate at a glance, and still
# separate in grayscale or for a reader who cannot distinguish the hues. The
# rule is a prefix rather than an enumerated list so a new ari_* measure
# inherits it instead of silently defaulting to dashed.
CARVE_ARI_PREFIX: str = "ari_"


def metric_linestyle(metric: str) -> str:
    """Return the line style for a metric: solid for CARVE, dashed for the rest."""
    return "-" if metric.startswith(CARVE_ARI_PREFIX) else "--"


# The same separation as the line style, carried by weight. The reference
# series are the busier half of a panel -- five of the seven curves -- so
# drawing them lighter lets CARVE's two read as the subject rather than as
# two more lines among seven.
CARVE_LINEWIDTH: float = 1.8
REFERENCE_LINEWIDTH: float = 1.2


def metric_linewidth(metric: str) -> float:
    """Return the line width for a metric: reference series are drawn lighter."""
    return (
        CARVE_LINEWIDTH if metric.startswith(CARVE_ARI_PREFIX) else REFERENCE_LINEWIDTH
    )


def cluster_colors(n: int) -> list[str]:
    """Return n cluster colors: CLUSTER_PALETTE in order, cycling if exhausted.

    Cluster i takes palette entry i. The published case-study composites draw
    four clusters in tab10's first four hues, and a figure's own legend reads
    C1, C2, C3, C4 down the palette in order, so "in order" is the only
    assignment that matches either.

    An earlier implementation returned ``plt.get_cmap(CLUSTER_CMAP_NAME,
    n)``'s entries instead, which is not the same list: Matplotlib spreads n
    samples evenly across the whole ten-color map rather than taking its
    first n, so four clusters landed on palette indices 0, 3, 6 and 9. That
    was done to agree with CARVE's own panels, which resample whatever
    ``palette`` they are handed; cluster_cmap(n) now gives those panels a map
    that is already exactly n colors long, so they agree with this function
    without it having to adopt their resampling.

    Past len(CLUSTER_PALETTE) the palette cycles from the start, so every
    repeat is exactly len(CLUSTER_PALETTE) apart and two *adjacent* cluster
    indices are never the same color. Resampling instead (a first attempt at
    this function) put duplicates on adjacent indices -- verified: at n=17, 7
    of 16 adjacent pairs shared a color. Color cannot distinguish more than
    ten clusters either way; adjacency is what is left to protect.
    """
    return [to_hex(CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]) for i in range(n)]


def cluster_cmap(n: int) -> ListedColormap:
    """cluster_colors(n) as a Colormap, for call sites that take one.

    CARVE's own plotting methods take a ``palette`` and resolve it with
    ``plt.get_cmap(palette, n)``, which returns an already-built Colormap
    unchanged. Handing them a map that is already the right length is what
    makes their direct-index panels and their imshow-normalized panels agree
    with each other and with cluster_colors -- see the comment on
    CLUSTER_CMAP_NAME.
    """
    return ListedColormap(cluster_colors(max(n, 1)))


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
