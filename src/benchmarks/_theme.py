"""The single source of figure styling.

Everything that decides how a figure looks lives here: palette, rcParams,
spine treatment, font sizes, and saving.
"""

import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex
from matplotlib.figure import Figure

# Okabe-Ito hues for CARVE's criteria.
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
    # The four classical indices in Fig 4's color families (pink, purple,
    # red, orange), darkened where needed to keep them apart and legible on
    # white.
    "silhouette": "#E0457B",
    "davies_bouldin": "#6A2C91",
    "calinski_harabasz": "#B01B20",
    "gap": "#C96A05",
    # M3C's RCSI, in Paul Tol's muted indigo: Okabe-Ito has no hue left that
    # is not already a metric or pipeline color.
    "m3c_rcsi": "#332288",
    # SC3's Tracy-Widom selection, in tab10 brown. PIPELINE_COLORS[3] is the
    # same hex, but it appears only in the Cusanovich figure and SC3 only in
    # the Klein and Levine figures.
    "sc3": "#8C564B",
    # The oracle reference is a grey dashed line, set apart from the black
    # axis text and spines.
    "baseline_oracle": "#595959",
}

FALLBACK_COLOR = "#7F7F7F"

# The first 64 entries of colorcet's glasbey_category10 (colorcet 3.2.1,
# CC BY 4.0), listed here because colorcet is not a dependency. The first ten
# are Matplotlib's tab10; each later entry is as far as possible from every
# color before it (Glasbey et al. 2007). Past 64 entries neighboring colors
# get too close to tell apart. tab10's eighth entry is the same grey as
# FALLBACK_COLOR.
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

# Scatter marker outlines and the corner axis arrows, in the axis text's
# black.
FOREGROUND_COLOR: str = "#000000"

# Neutral greys for the schematic guidance figures (figures/_guidance.py):
# header bars and badges, card backgrounds, dartboard rings, the highlight
# behind the cluster in question and the grey bands over untested or
# excluded k.
GUIDE_COLORS: dict[str, str] = {
    "header": "#E4E6E8",
    "badge": "#3B4651",
    "card": "#F3F4F5",
    "muted": "#595959",
    "axis": "#8C8C8C",
    "candidate": "#7F7F7F",
    "ring_dark": "#E3E3E3",
    "ring_light": "#FFFFFF",
    "ring_edge": "#B5B5B5",
    "bull": "#C4C4C4",
    "focus": "#E2E4E7",
    "untested": "#E9E9E9",
}

# CARVE's plotting functions take a colormap and resolve it with
# ``plt.get_cmap(palette, n)``, which returns a Colormap instance unchanged
# and ignores n. plot_cluster_violin and plot_cluster_scatter index the map by
# integer, while plot_consensus_matrix's color band goes through imshow, which
# first normalizes the cluster index to [0, 1]. On a ten-color map the two
# give the same cluster different colors. cluster_cmap(n) holds exactly n
# colors, so both agree with cluster_colors(n). Resampling a ten-color map
# by name would spread the n colors over the whole palette (entries 0, 3, 6
# and 9 for four clusters).

# One line per estimator in the CARVE-output figures' metric-over-k panels.
# plot_metric_over_n_clusters samples the map it is given at
# np.linspace(0, 1, n), so a two-entry map puts the two estimators each case
# study sweeps (Ward or KMeans, then spectral) on its two entries; a third
# estimator needs a third entry. Registration is guarded so a second import
# does not raise.
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
# its own entry: Okabe-Ito hues, tab10's brown and Paul Tol's high-contrast
# yellow. The first entry is also CARVE generalizability's color in
# METRIC_COLORS, and the Cusanovich figure draws both, in panels C and D.
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

# One hue, light to dark, for the proportion P(k-hat = k), in CARVE
# stability's green.
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
    # Type 42 embeds TrueType fonts, so text in a PDF stays editable.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Figures laid out at the size they print at. PLOS Computational Biology
# takes figures 2.63 to 7.5 inches wide and at most 8.75 inches tall, with
# text in Arial, Times or Symbol at 8 to 12 point. The sizes above suit
# figures drawn larger and scaled down on the page, which takes their text
# below that range; a figure built under print_theme_context at
# PRINT_WIDTH_IN is not scaled, so these sizes are the printed sizes.
# Liberation Sans is metric-compatible with Arial and stands in where Arial
# is not installed; DejaVu Sans, which ships with matplotlib, is the last
# resort, so the figures render anywhere. Math text uses the same sans, so
# rho and k-hat match the surrounding labels.
PRINT_WIDTH_IN: float = 7.5
PRINT_MAX_HEIGHT_IN: float = 8.75
PRINT_FONT_SIZES: dict[str, float] = {
    "tick": 8.0,
    "legend": 8.0,
    "axis_label": 9.0,
    "title": 9.0,
    "panel_letter": 12.0,
}
PRINT_LINEWIDTH: float = 1.3
PRINT_REFERENCE_LINEWIDTH: float = 0.8
PRINT_MARKERSIZE: float = 3.5

PRINT_RC_PARAMS: dict[str, Any] = {
    **RC_PARAMS,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    # The custom set resolves every math slot on first use. At its "cursive"
    # default the cal slot finds Apple Chancery on macOS, and matplotlib logs
    # a font-weight warning for it.
    "mathtext.cal": "sans:italic",
    "font.size": PRINT_FONT_SIZES["tick"],
    "axes.titlesize": PRINT_FONT_SIZES["title"],
    "axes.labelsize": PRINT_FONT_SIZES["axis_label"],
    "legend.fontsize": PRINT_FONT_SIZES["legend"],
    "xtick.labelsize": PRINT_FONT_SIZES["tick"],
    "ytick.labelsize": PRINT_FONT_SIZES["tick"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": PRINT_LINEWIDTH,
    "lines.markersize": PRINT_MARKERSIZE,
    # One legend style for every print figure: frameless, and spaced tighter
    # than Matplotlib's defaults (0.5, 2.0, 0.8, 0.5), since 8 points is the
    # smallest text PLOS takes.
    "legend.frameon": False,
    "legend.title_fontsize": PRINT_FONT_SIZES["legend"],
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3,
    # bbox_inches="tight" pads the saved file by this much on every side; at
    # the default 0.1 inch a figure laid out at PRINT_WIDTH_IN saves wider
    # than the page.
    "savefig.pad_inches": 0.02,
}


def apply_theme() -> None:
    """Apply the manuscript style to the global rcParams."""
    plt.rcParams.update(RC_PARAMS)


@contextmanager
def theme_context() -> Iterator[None]:
    """Apply the theme for the duration of a block, then restore."""
    with mpl.rc_context(RC_PARAMS):
        yield


@contextmanager
def print_theme_context() -> Iterator[None]:
    """The theme at printed size (PRINT_RC_PARAMS) for a block, then restore."""
    with mpl.rc_context(PRINT_RC_PARAMS):
        yield


def metric_color(metric: str) -> str:
    """Return the color for a metric, falling back to gray for unknown names."""
    return METRIC_COLORS.get(metric, FALLBACK_COLOR)


# Solid lines for CARVE's selection curves (any ari_* measure), dashed for
# the reference series beside them (the oracle and the four classical
# indices), so the two groups also separate in grayscale.
CARVE_ARI_PREFIX: str = "ari_"

# Four warm hues cannot all be told apart under color-vision deficiency, so
# each comparator also gets its own dash pattern (on/off lengths in points).
DEFAULT_DASHES: tuple[float, ...] = (4.0, 1.6)
COMPARATOR_DASHES: dict[str, tuple[float, ...]] = {
    "silhouette": (4.0, 1.6),
    "davies_bouldin": (1.4, 1.4),
    "calinski_harabasz": (5.5, 1.6, 1.2, 1.6),
    "gap": (2.6, 1.4, 1.4, 1.4),
    "baseline_oracle": (5.0, 2.0),
}


def metric_dashes(metric: str) -> tuple[float, ...] | None:
    """The dash pattern for a metric, or None for CARVE's solid curves."""
    if metric.startswith(CARVE_ARI_PREFIX):
        return None
    return COMPARATOR_DASHES.get(metric, DEFAULT_DASHES)


def metric_linestyle(metric: str) -> str | tuple[int, tuple[float, ...]]:
    """Return the line style for a metric: solid for CARVE, dashed for the rest.

    Each comparator has its own (offset, dashes) pair, so the four classical
    indices also separate from each other.
    """
    dashes = metric_dashes(metric)
    return "-" if dashes is None else (0, dashes)


# Reference series are drawn thinner than CARVE's curves.
CARVE_LINEWIDTH: float = 1.8
REFERENCE_LINEWIDTH: float = 1.2


def metric_linewidth(metric: str) -> float:
    """Return the line width for a metric: reference series are drawn lighter."""
    return (
        CARVE_LINEWIDTH if metric.startswith(CARVE_ARI_PREFIX) else REFERENCE_LINEWIDTH
    )


def cluster_colors(n: int) -> list[str]:
    """Return n cluster colors: CLUSTER_PALETTE in order, cycling if exhausted.

    Cluster i takes palette entry i, so legends read C1, C2, ... down the
    palette. Past len(CLUSTER_PALETTE) the palette starts over, and adjacent
    clusters never share a color.
    """
    return [to_hex(CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]) for i in range(n)]


def cluster_cmap(n: int) -> ListedColormap:
    """cluster_colors(n) as a Colormap, for call sites that take one.

    CARVE's plotting methods use it unchanged, so their integer-indexed and
    imshow panels agree with cluster_colors (see the comment above
    ESTIMATOR_COLORS).
    """
    return ListedColormap(cluster_colors(max(n, 1)))


def style_axes(ax: Axes, *, grid: bool = True) -> Axes:
    """Apply the spine and grid treatment to one axes."""
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


def print_axes_text(ax: Axes) -> Axes:
    """Set an axes' title, axis labels, tick labels and legend to the print sizes.

    CARVE's own plotting methods size their text for a figure viewed on
    screen, at up to 13 points, which a figure laid out at PRINT_WIDTH_IN
    prints as is. Free text placed with ax.text, such as panel letters, is
    left alone.
    """
    ax.title.set_fontsize(PRINT_FONT_SIZES["title"])
    ax.xaxis.label.set_fontsize(PRINT_FONT_SIZES["axis_label"])
    ax.yaxis.label.set_fontsize(PRINT_FONT_SIZES["axis_label"])
    ax.tick_params(labelsize=PRINT_FONT_SIZES["tick"])
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(PRINT_FONT_SIZES["legend"])
        legend.get_title().set_fontsize(PRINT_FONT_SIZES["legend"])
    return ax


def print_free_text(ax: Axes) -> Axes:
    """Set an axes' free text from its screen size to its print size.

    The panel helpers place text with ax.text and ax.annotate at FONT_SIZES's
    sizes: alluvial labels, axis arrows, panel letters. Each takes the
    PRINT_FONT_SIZES size of the role it was set at; text at any other size
    is left alone.
    """
    by_screen = {FONT_SIZES[role]: PRINT_FONT_SIZES[role] for role in FONT_SIZES}
    for text in ax.texts:
        size = by_screen.get(text.get_fontsize())
        if size is not None:
            text.set_fontsize(size)
    return ax


def thin_lines(ax: Axes, scale: float) -> Axes:
    """Scale an axes' line widths and marker sizes by ``scale``.

    For curves drawn at a width that suits a figure laid out larger than it
    prints. Error bar caps are markers on lines of their own, and the bars
    are line collections, so both follow.
    """
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth() * scale)
        line.set_markersize(line.get_markersize() * scale)
    for collection in ax.collections:
        collection.set_linewidths(np.asarray(collection.get_linewidths()) * scale)
    return ax


def save_figure(fig: Figure, path: Path, *, dpi: int = 300) -> Path:
    """Save a figure under its manuscript filename, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# PLOS Computational Biology takes each main figure as its own file: a TIFF
# with LZW compression, RGB at 8 bits per channel with no alpha channel or
# layers, 300 to 600 dpi, 2.63 to 7.5 inches wide, at most 8.75 inches tall
# and 10 MB or less (journals.plos.org/ploscompbiol/s/figures). The width
# and height limits are PRINT_WIDTH_IN and PRINT_MAX_HEIGHT_IN above.
PLOS_TIFF_DPI: int = 300
PLOS_MIN_WIDTH_IN: float = 2.63
PLOS_MAX_FILE_BYTES: int = 10_000_000


def save_plos_tiff(fig: Figure, path: Path, *, dpi: int = PLOS_TIFF_DPI) -> Path:
    """Save a figure as a PLOS figure file, creating parents as needed.

    The figure is rendered as save_figure renders it and flattened onto
    white, since Matplotlib's raster output carries an alpha channel that
    PLOS does not accept. A file outside the size limits raises ValueError
    before anything is written, so a figure that would be rescaled in
    production, and its text with it, never reaches the submission folder.
    """
    from PIL import Image

    path = Path(path)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        rgba = rendered.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, rgba).convert("RGB")

    width_in, height_in = rgb.width / dpi, rgb.height / dpi
    if rgb.width > round(PRINT_WIDTH_IN * dpi):
        raise ValueError(
            f"{path.name} is {width_in:.2f} inches wide; PLOS allows at most "
            f"{PRINT_WIDTH_IN} inches."
        )
    if width_in < PLOS_MIN_WIDTH_IN:
        raise ValueError(
            f"{path.name} is {width_in:.2f} inches wide; PLOS requires at least "
            f"{PLOS_MIN_WIDTH_IN} inches."
        )
    if rgb.height > round(PRINT_MAX_HEIGHT_IN * dpi):
        raise ValueError(
            f"{path.name} is {height_in:.2f} inches tall; PLOS allows at most "
            f"{PRINT_MAX_HEIGHT_IN} inches."
        )

    encoded = io.BytesIO()
    rgb.save(encoded, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    size = encoded.getbuffer().nbytes
    if size > PLOS_MAX_FILE_BYTES:
        raise ValueError(
            f"{path.name} is {size / 1e6:.1f} MB; PLOS allows at most "
            f"{PLOS_MAX_FILE_BYTES / 1e6:g} MB."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.getvalue())
    return path
