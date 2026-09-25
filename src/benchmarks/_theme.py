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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex
from matplotlib.figure import Figure

# Okabe-Ito, which is colorblind-safe and already what two of the three
# previous definitions used for CARVE stability.
METRIC_COLORS: dict[str, str] = {
    "ari_stability_1se": "#009E73",
    "ari_stability": "#009E73",
    "ari_stability_quant": "#009E73",
    "ari_generalizability_1se": "#0072B2",
    "ari_generalizability": "#0072B2",
    "ari_generalizability_quant": "#0072B2",
    "ari_average_1se": "#E69F00",
    "ari_average": "#E69F00",
    "ari_average_quant": "#E69F00",
    "consensus_pac_stability": "#8C8C8C",
    "consensus_gini_stability": "#6E6E6E",
    "consensus_ce_stability": "#4F4F4F",
    "accuracy_generalizability": "#7BC8F0",
    # The four classical indices, in the published Fig 4's families: pink,
    # purple, red, orange. Three of the four moved from the literal
    # published values after the dataviz validator flagged two failures
    # against a light chart surface: Silhouette #E0457B and Davies-Bouldin
    # #A8389E were dE 13.6 apart in normal vision, below the floor of 15, so
    # full-color readers could not separate them either -- visible in the
    # published t-distributed panel; and Gap #F28522 sat at 2.51:1 contrast,
    # below the 3:1 floor. Each replacement is a deeper step of the same
    # hue, so the figure stays recognizable and the family assignment is
    # unchanged. Silhouette keeps its published value.
    #
    # The re-stepped set passes every check: lightness band, chroma floor,
    # CVD separation at worst adjacent dE 8.7 protan and 8.6 tritan,
    # normal-vision floor at worst adjacent dE 16.2, and contrast at or above
    # 3:1 for all seven series.
    #
    # That module assigned the four positionally, in whatever order a caller
    # listed its metrics, so the published figures disagree with each other
    # about which index is which color. Fig 4's assignment is the one adopted
    # here, so the case study panels change rather than Fig 4.
    "silhouette": "#E0457B",
    "davies_bouldin": "#6A2C91",
    "calinski_harabasz": "#B01B20",
    "gap": "#C96A05",
    # M3C's RCSI. Paul Tol muted indigo, used because the Okabe-Ito palette
    # is exhausted -- its blue is already PIPELINE_COLORS[0] -- and the metric
    # and pipeline palettes must stay disjoint. RCSI is neither a CARVE
    # criterion nor a geometric CVI and must not be mistaken for either.
    "m3c_rcsi": "#332288",
    # Grey, not black: the published figures (Fig 4, S2 Fig) draw the oracle
    # reference as a grey dashed line, distinct from the black axis text and
    # spines it would otherwise be confused with.
    "baseline_oracle": "#595959",
}

FALLBACK_COLOR = "#7F7F7F"

# The first 64 entries of colorcet's glasbey_category10 (colorcet 3.2.1,
# CC BY 4.0). Its first ten entries are Matplotlib's tab10, which is what the
# published case-study composites draw their clusters in: the Klein figure's
# four timepoints are tab10's first four hues, so figures with at most ten
# labels are unchanged. The remaining entries follow Glasbey et al. (2007):
# each is chosen as far as possible from every color before it. tab10 alone
# cycled, so the Cusanovich source partition's 30 clusters shared ten colors
# three to a color. Spelled out rather than read from colorcet, which is not a
# dependency, so the values are greppable and cannot shift under a release.
#
# Measured in OKLab (times 100), the closest pair is 9.8 apart among the first
# 30 entries, 7.0 among the first 64 and 4.0 among the first 128, so the
# palette stops at 64. No palette of 30 colors keeps every pair apart under
# color-vision deficiency; tab10's green and orange already do not.
#
# tab10's eighth entry is the same grey as FALLBACK_COLOR. That only matters
# for a figure with at least eight clusters and a color map missing an entry,
# which no current figure has; if one appears, the fallback is what should
# move, not the palette.
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
    "#3A0183",
    "#004301",
    "#0FFFA9",
    "#5E0040",
    "#BCBCFF",
    "#D8AFA2",
    "#B80080",
    "#004E53",
    "#6B6500",
    "#7D0200",
    "#6126FF",
    "#FFFF9A",
    "#574964",
    "#8CB894",
    "#94FCFF",
    "#028268",
    "#91FF00",
    "#8300A0",
    "#AD8944",
    "#5B3400",
    "#FFC0F3",
    "#FF6F76",
    "#798CFF",
    "#DD00FF",
    "#515646",
    "#00458A",
    "#FFBF60",
    "#FF018D",
    "#BEC9CF",
    "#AF98B5",
    "#B75700",
    "#027000",
    "#CD88FF",
    "#1DD646",
    "#C0ECC4",
    "#7A98B5",
    "#A56089",
    "#6F8957",
    "#BD7D76",
    "#8B2945",
    "#00ADFF",
    "#8FD4FF",
    "#4B6D77",
    "#00D4B1",
    "#9300F3",
    "#8B9500",
    "#5D5C9F",
    "#FEDFBB",
    "#00939F",
    "#FFDC00",
    "#00AB79",
    "#520068",
    "#000092",
    "#0B5D3E",
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
# Passing a registered ten-color map by name and letting Matplotlib resample
# it to n also makes the two styles agree, but on the wrong colors:
# resampling spreads n samples across the whole palette rather than taking
# its first n, so four clusters come out palette entries 0, 3, 6 and 9
# instead of 0 to 3.

# One line per estimator in the CARVE-output figures' metric-over-k panels.
# plot_metric_over_n_clusters samples the map it is given at
# np.linspace(0, 1, n), so a two-entry map puts the two estimators each case
# study sweeps (Ward or KMeans, then spectral) on its two entries exactly; a
# third estimator would repeat the second color, so extend the tuple before
# adding one. tab10 blue and red pass the palette checks: contrast at or
# above 3:1, CVD separation dE 21.1 protan and 33.8 tritan. Guarded so
# importing this module twice (e.g. a test re-import) does not raise on
# re-registration.
ESTIMATOR_COLORS: tuple[str, ...] = (CLUSTER_PALETTE[0], CLUSTER_PALETTE[3])
ESTIMATOR_CMAP_NAME: str = "carve_estimator"
if ESTIMATOR_CMAP_NAME not in mpl.colormaps:
    mpl.colormaps.register(
        ListedColormap(list(ESTIMATOR_COLORS), name=ESTIMATOR_CMAP_NAME)
    )

# Preprocessing pipelines in the Cusanovich figure's per-pipeline panel.
# CARVE's plot_metric_by_pipeline takes a colormap name and samples it at
# evenly spaced points, one per pipeline in sorted label order, so a
# ListedColormap of exactly these colors gives each of up to five pipelines
# its own entry. Okabe-Ito hues, tab10's brown and Paul Tol's high-contrast
# yellow. The yellow was the candidate that kept every pair apart under
# simulated color-vision deficiency; the brown is low in chroma and closest to
# the vermillion, which predates it.
#
# The first entry, Okabe-Ito blue, is also CARVE generalizability's color
# (METRIC_COLORS), which moved onto it from #56B4E9 to clear the 3:1
# contrast floor -- the one deliberate exception to "never a METRIC_COLORS
# value" here. The collision is visible, not merely theoretical:
# figures/_cusanovich_results.py draws carve_lines (metric colors) in panel
# C and pipeline_lines (this palette) in panel D of the same Figure, so
# blue reads as CARVE Generalizability in one panel and the first pipeline
# in the panel beside it. Accepted rather than fixed: generalizability's
# color is fixed by the benchmark figures' contrast requirement, and this
# palette belongs to the Cusanovich case-study figure alone; see
# TestPipelinePalette.test_no_pipeline_color_is_a_criterion_color for the
# one overlap the test still allows. If the ambiguity ever matters, the fix
# is re-stepping this entry and re-running the five pipeline colors through
# a CVD check.
PIPELINE_COLORS: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#8C564B",
    "#DDAA33",
)
PIPELINE_CMAP_NAME: str = "carve_pipeline"
if PIPELINE_CMAP_NAME not in mpl.colormaps:
    mpl.colormaps.register(
        ListedColormap(list(PIPELINE_COLORS), name=PIPELINE_CMAP_NAME)
    )

# One hue, light to dark, for magnitude. P(k-hat = k) is a proportion, so it
# takes a sequential ramp; a diverging or rainbow map would encode its
# ordering with hue, which reads as category. The hue is CARVE stability's
# green, so the heatmap belongs to the same figure as the curves above it.
SEQUENTIAL_COLORS: tuple[str, ...] = (
    "#F4FBF8",
    "#D6F0E5",
    "#ABE0CB",
    "#74CBAC",
    "#3FB48D",
    "#149A71",
    "#007A57",
    "#00573E",
)
SEQUENTIAL_CMAP_NAME: str = "carve_sequential"
if SEQUENTIAL_CMAP_NAME not in mpl.colormaps:
    mpl.colormaps.register(
        LinearSegmentedColormap.from_list(SEQUENTIAL_CMAP_NAME, list(SEQUENTIAL_COLORS))
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

# Hue alone cannot separate four comparator series under color-vision
# deficiency -- four warm hues never clear the threshold pairwise, whatever
# steps they take -- so each carries a distinct dash pattern as well. The
# patterns are in points, alternating on and off, and are ordered so no two
# read alike at the 1.2pt reference line width.
DEFAULT_DASHES: tuple[float, ...] = (4.0, 1.6)
COMPARATOR_DASHES: dict[str, tuple[float, ...]] = {
    "silhouette": (4.0, 1.6),
    "davies_bouldin": (1.4, 1.4),
    "calinski_harabasz": (5.5, 1.6, 1.2, 1.6),
    "gap": (2.6, 1.4, 1.4, 1.4),
    # The oracle keeps the long dash the published figures draw it with.
    "baseline_oracle": (5.0, 2.0),
}


def metric_dashes(metric: str) -> tuple[float, ...] | None:
    """The dash pattern for a metric, or None for CARVE's solid curves."""
    if metric.startswith(CARVE_ARI_PREFIX):
        return None
    return COMPARATOR_DASHES.get(metric, DEFAULT_DASHES)


def metric_linestyle(metric: str) -> str | tuple[int, tuple[float, ...]]:
    """Return the line style for a metric: solid for CARVE, dashed for the rest.

    A comparator's style is its own (offset, dashes) pair rather than the
    shared "--", so the four classical indices separate from each other in
    grayscale and under color-vision deficiency, not only from CARVE.
    """
    dashes = metric_dashes(metric)
    return "-" if dashes is None else (0, dashes)


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

    Resampling a registered ten-color map to n gives a different list:
    Matplotlib spreads n samples evenly across the whole map rather than
    taking its first n, so four clusters land on palette indices 0, 3, 6 and
    9. CARVE's own panels resample whatever ``palette`` they are handed;
    cluster_cmap(n) gives them a map that is already exactly n colors long,
    so they agree with this function without it adopting their resampling.

    Past len(CLUSTER_PALETTE) the palette cycles from the start, so every
    repeat is exactly len(CLUSTER_PALETTE) apart and two adjacent cluster
    indices are never the same color. Resampling instead (a first attempt at
    this function, against the ten-color palette) put duplicates on adjacent
    indices -- verified: at n=17, 7 of 16 adjacent pairs shared a color.
    """
    return [to_hex(CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]) for i in range(n)]


def cluster_cmap(n: int) -> ListedColormap:
    """cluster_colors(n) as a Colormap, for call sites that take one.

    CARVE's own plotting methods take a ``palette`` and resolve it with
    ``plt.get_cmap(palette, n)``, which returns an already-built Colormap
    unchanged. Handing them a map that is already the right length is what
    makes their direct-index panels and their imshow-normalized panels agree
    with each other and with cluster_colors -- see the module-level comment
    on CARVE's ``palette`` argument.
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
