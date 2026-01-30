from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import textwrap
import warnings
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from cmap import Colormap


from ._selection import get_best_row, MEASURE_MAP, Measure, Rule


PlotMode = Literal["screen", "plos"]
ClusterMetric = Literal["gini", "ce", "misclassification"]


# ---------------------------------------------------------------------
# PLOS Comp Bio figure constraints 
# ---------------------------------------------------------------------
# According to PLOS Comp Bio Figures page
# - File format: TIFF or EPS
# - Resolution: 300–600 dpi
# - Dimensions (at 300 dpi): width 789–2250 px (2.63–7.5 in); height max 2625 px (8.75 in)
# - Text: Arial, Times, or Symbol only, 8–12 pt
# - Color: RGB (8bit/channel) or grayscale
# - Size: <10 MB
# - Recommended: 2-pt white border, crop excess whitespace

PLOS_MIN_W_IN = 2.63
PLOS_MAX_W_IN = 7.50
PLOS_MAX_H_IN = 8.75
PLOS_DPI_MIN = 300
PLOS_DPI_MAX = 600
PLOS_BORDER_PT = 2.0

# Default: align with text column width suggestion (<= 5.2 in)
PLOS_COLUMN_W_IN = 5.2

# Okabe-Ito colorblind-friendly palette
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", #"#000000",
]


def _pt_to_in(pt: float) -> float:
    return float(pt) / 72.0


@dataclass(frozen=True)
class PlotSpec:
    """Central plot configuration shared by all plotting functions."""
    mode: PlotMode = "screen"

    # Figure geometry (in inches)
    width_in: float = 7.0
    height_in: float = 4.5

    # Typography
    font_family: str = "DejaVu Sans"
    base_fontsize: float = 11.0

    # Lines/ticks
    axes_linewidth: float = 1.0
    tick_width: float = 0.8
    tick_length: float = 3.0

    # Export
    dpi: int = 120
    file_format: Literal["png", "tiff", "eps", "pdf"] = "png"

    # Layout
    constrained_layout: bool = True


def plos_spec(
    *,
    width_in: float = PLOS_COLUMN_W_IN,
    height_in: float | None = None,
    dpi: int = 600,
    font_family: Literal["Arial", "Times New Roman", "Symbol"] | str = "Arial",
    base_fontsize: float = 9.0,
) -> PlotSpec:
    """
    Convenience spec aimed at PLOS Comp Bio.
    We cap dpi into [300, 600] as recommended by PLOS.
    """
    dpi = int(np.clip(dpi, PLOS_DPI_MIN, PLOS_DPI_MAX))

    width_in = float(width_in)
    if width_in < PLOS_MIN_W_IN or width_in > PLOS_MAX_W_IN:
        warnings.warn(
            f"width_in={width_in} is outside PLOS recommended [{PLOS_MIN_W_IN}, {PLOS_MAX_W_IN}] inches.",
            RuntimeWarning,
            stacklevel=2,
        )

    if height_in is None:
        # A sane default aspect for “figure panels” (tweak per-plot later)
        height_in = min(0.62 * width_in, PLOS_MAX_H_IN)

    height_in = float(height_in)
    if height_in > PLOS_MAX_H_IN:
        warnings.warn(
            f"height_in={height_in} exceeds PLOS max height {PLOS_MAX_H_IN} inches.",
            RuntimeWarning,
            stacklevel=2,
        )

    return PlotSpec(
        mode="plos",
        width_in=width_in,
        height_in=height_in,
        font_family=str(font_family),
        base_fontsize=float(base_fontsize),
        axes_linewidth=0.8,
        tick_width=0.8,
        tick_length=3.0,
        dpi=dpi,
        file_format="tiff", 
        constrained_layout=True,
    )


def screen_spec(
    *,
    width_in: float = 8.0,
    height_in: float = 5.0,
    dpi: int = 120,
    base_fontsize: float = 11.0,
    font_family: str = "DejaVu Sans",
) -> PlotSpec:
    return PlotSpec(
        mode="screen",
        width_in=float(width_in),
        height_in=float(height_in),
        dpi=int(dpi),
        base_fontsize=float(base_fontsize),
        font_family=str(font_family),
        file_format="png",
        constrained_layout=True,
    )


def _rcparams_for(spec: PlotSpec) -> dict[str, Any]:
    """
    Matplotlib rcParams tuned for either screen use or PLOS export.

    For PLOS we try to:
    - use Arial/Times/Symbol (matplotlib falls back if missing)
    - keep text 8–12 pt
    - keep line widths readable after downscaling
    - embed TrueType fonts in PDF/PS outputs
    """
    font_list = [spec.font_family]
    if spec.mode == "plos":
        # Prefer the allowed fonts first; if not available, matplotlib falls back.
        font_list = [spec.font_family, "Arial", "Times New Roman", "Symbol", "DejaVu Sans"]

    cm = Colormap('okabeito:okabeito')

    return {
        "figure.dpi": spec.dpi,
        "savefig.dpi": spec.dpi,
        "font.family": "sans-serif",
        "font.sans-serif": font_list,
        "font.size": spec.base_fontsize,
        "axes.titlesize": spec.base_fontsize,
        "axes.labelsize": spec.base_fontsize,
        "legend.fontsize": max(spec.base_fontsize - 1, 8),
        "xtick.labelsize": max(spec.base_fontsize - 1, 8),
        "ytick.labelsize": max(spec.base_fontsize - 1, 8),
        "axes.linewidth": spec.axes_linewidth,
        "xtick.major.width": spec.tick_width,
        "ytick.major.width": spec.tick_width,
        "xtick.major.size": spec.tick_length,
        "ytick.major.size": spec.tick_length,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        # We encourage proper font embedding for vector outputs
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


class plotting_context:
    """
    Context manager that applies CARVE plotting defaults.

    Example:
        with plotting_context(plos_spec()):
            fig, ax = new_figure()
            ...
            save_figure(fig, "Fig1.tiff", spec=plos_spec())
    """
    def __init__(self, spec: PlotSpec):
        self.spec = spec
        self._ctx = mpl.rc_context(_rcparams_for(spec))

    def __enter__(self):
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


def new_figure(
    *,
    spec: PlotSpec,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool | str = False,
    sharey: bool | str = False,
) -> tuple[mpl.figure.Figure, Any]:
    """
    Create a figure sized according to spec.
    Returns (fig, ax) where ax is an ndarray for grids, or Axes for 1×1.
    """
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(spec.width_in, spec.height_in),
        sharex=sharex,
        sharey=sharey,
        constrained_layout=spec.constrained_layout,
    )
    return fig, ax


def save_figure(
    fig: mpl.figure.Figure,
    path: str,
    *,
    spec: PlotSpec,
    transparent: bool = False,
) -> None:
    """
    Save figure with mode-aware defaults.

    Notes for PLOS:
    - recommended resolution 300–600 dpi
    - TIFF/EPS accepted
    - recommended 2-pt border
    """
    kwargs: dict[str, Any] = {}

    if spec.mode == "plos":
        dpi = int(np.clip(spec.dpi, PLOS_DPI_MIN, PLOS_DPI_MAX))
        kwargs["dpi"] = dpi
        kwargs["bbox_inches"] = "tight"
        kwargs["pad_inches"] = _pt_to_in(PLOS_BORDER_PT)
    else:
        kwargs["dpi"] = spec.dpi
        kwargs["bbox_inches"] = "tight"
        kwargs["pad_inches"] = 0.01

    fig.savefig(path, transparent=transparent, **kwargs)


# ---------------------------------------------------------------------
# DataFrame helpers: grouping, method IDs, display labels, basic checks
# ---------------------------------------------------------------------

_DEFAULT_METRIC_BASE_COLS = (
    "ari_stability",
    "ari_generalizability",
    "ari_average",
    "consensus_pac_stability",
    "consensus_gini_stability",
    "consensus_ce_stability",
    "misclassification_generalizability",
)

_DEFAULT_METRIC_SUFFIXES = ("_se", "_lower", "_upper")

_Y_LABELS: Mapping[str, str] = {
    "ari_stability": "ARI (stability)",
    "ari_generalizability": "ARI (generalizability)",
    "ari_average": "ARI (average)",
    "consensus_pac_stability": "Consensus PAC",
    "consensus_gini_stability": "Consensus Gini",
    "consensus_ce_stability": "Consensus CE",
    "misclassification_generalizability": "Global misclassification",
}

_BOX_YLABELS: Mapping[str, str] = {
    "gini": "Cluster Stability (Gini)",
    "ce": "Cluster Stability (Cross-Entropy)",
    "misclassification": "Cluster Generalizability",
}


def infer_metric_cols(df: pd.DataFrame) -> set[str]:
    """
    Infer “metric-like” columns to exclude from hyperparameter grouping.
    We exclude:
      - known metric bases
      - any col that matches base+suffix for common suffixes
    """
    cols = set(df.columns)
    out: set[str] = set()
    for base in _DEFAULT_METRIC_BASE_COLS:
        if base in cols:
            out.add(base)
        for suf in _DEFAULT_METRIC_SUFFIXES:
            c = f"{base}{suf}"
            if c in cols:
                out.add(c)
    return out


def infer_param_cols(df: pd.DataFrame) -> list[str]:
    """
    Columns that define a method configuration, excluding n_clusters and metric outputs.
    Unique identifiers: estimator + hyperparams (except n_clusters).
    """
    metric_cols = infer_metric_cols(df)
    exclude = {"n_clusters"} | metric_cols
    # keep estimator first if present, then others stable-sorted
    cols = [c for c in df.columns if c not in exclude]
    if "estimator" in cols:
        cols = ["estimator"] + [c for c in cols if c != "estimator"]
    return cols


def _fmt_param(v: Any, decimals: int = 4) -> str:
    if pd.isna(v):
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{decimals}f}"
    return str(v)


def build_method_id(row: pd.Series, *, param_cols: Sequence[str], decimals: int = 4) -> str:
    """
    Stable ID: 'Estimator|p1=v1|p2=v2|...'
    Excludes n_clusters by construction – param_cols should already exclude it.
    """
    parts: list[str] = []
    for c in param_cols:
        parts.append(f"{c}={_fmt_param(row.get(c), decimals)}")
    return "|".join(parts)


def build_method_label(row: pd.Series, *, param_cols: Sequence[str], decimals: int = 3) -> str:
    """
    Human label for legends. Tries to stay compact.
    Format: 'Estimator (p1=v1, p2=v2)'
    """
    est = _fmt_param(row.get("estimator"), decimals)
    params = []
    for c in param_cols:
        if c == "estimator":
            continue
        v = row.get(c)
        if pd.isna(v):
            continue
        params.append(f"{c}={_fmt_param(v, decimals)}")
    if not params:
        return est
    return f"{est} ({', '.join(params)})"


def add_method_columns(
    df: pd.DataFrame,
    *,
    param_cols: Sequence[str] | None = None,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Returns a copy with:
      - _method_id
      - _method_label
    """
    if param_cols is None:
        param_cols = infer_param_cols(df)

    out = df.copy()
    out["_method_id"] = out.apply(lambda r: build_method_id(r, param_cols=param_cols, decimals=decimals), axis=1)
    out["_method_label"] = out.apply(lambda r: build_method_label(r, param_cols=param_cols, decimals=max(decimals - 1, 2)), axis=1)
    return out


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


# ---------------------------------------------------------------------
# Common plot helpers (legend placement, bands, palettes, DR hook)
# ---------------------------------------------------------------------

def legend_outside_right(ax: mpl.axes.Axes, *, title: str | None = None) -> None:
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        title=title,
    )


def compute_1se_band(y: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    se = np.asarray(se, float)
    return y - se, y + se


# -------------- dimensionality reduction hook --------------

DRFunc = Callable[[np.ndarray], np.ndarray]


def default_dr_pca(X: np.ndarray) -> np.ndarray:
    # local import so sklearn is only required if DR is used
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(X)


def project_2d(X: np.ndarray, *, dr: DRFunc | None = None) -> np.ndarray:
    """
    Returns an (n,2) projection. If dr returns >2 dims, we take the first two.
    Accepts either a callable or a transformer object with .fit_transform.
    """
    if dr is None:
        Z = default_dr_pca(X)
    elif callable(dr):
        Z = dr(X)
    elif hasattr(dr, "fit_transform"):
        Z = dr.fit_transform(X)
    else:
        raise TypeError("dr must be a callable or have a .fit_transform method.")
    Z = np.asarray(Z, float)
    if Z.ndim != 2 or Z.shape[0] != X.shape[0]:
        raise ValueError("dr(X) must return a 2D array with the same number of rows as X.")
    if Z.shape[1] < 2:
        raise ValueError("dr(X) must return at least 2 components.")
    return Z[:, :2]


# -------------- cluster palette + ordering helpers --------------

def cluster_colors(labels: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
    """
    Map integer cluster labels -> RGBA using:
      - OKABE_ITO palette for <=7 clusters,
      - tab20 for 8–20 clusters,
      - a continuous colormap (hsv) for >20 clusters.
    Keeps a stable order by sorted(unique(labels)).
    """
    labs = np.asarray(labels)
    uniq = np.unique(labs)
    uniq = uniq[uniq >= 0]  # ignore -1 if used for outliers
    k = len(uniq)
    if k <= 7:
        palette = OKABE_ITO * ((k + len(OKABE_ITO) - 1) // len(OKABE_ITO))
        cols = [mpl.colors.to_rgba(palette[i]) for i in range(k)]
    elif 7 < k <= 20:
        tab20 = plt.get_cmap("tab20")
        cols = [tab20(i) for i in range(k)]
    else:
        cmap = plt.get_cmap("hsv")
        cols = [cmap(i / max(k - 1, 1)) for i in range(k)]
    return {int(c): cols[i] for i, c in enumerate(sorted(map(int, uniq)))}


def cluster_boundaries(labels_ordered: np.ndarray) -> np.ndarray:
    """
    Given labels in the display order (e.g., consensus ordering), return boundary indices
    where cluster changes (useful for drawing black separators).
    """
    lab = np.asarray(labels_ordered)
    if lab.ndim != 1:
        raise ValueError("labels_ordered must be 1D")
    changes = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    return changes.astype(int)


def _values_by_cluster(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    cluster_ids: Sequence[int],
) -> list[np.ndarray]:
    """Split a per-sample vector into a list of arrays (one array per cluster)."""
    v = np.asarray(values, float)
    y = np.asarray(labels)

    if v.ndim != 1 or y.ndim != 1:
        raise ValueError("Values and labels must be 1D arrays.")
    if len(v) != len(y):
        raise ValueError(f"Values and labels must have same length; got {len(v)} vs {len(y)}.")

    out: list[np.ndarray] = []
    for cid in cluster_ids:
        arr = v[y == cid]
        arr = arr[np.isfinite(arr)]
        out.append(arr)
    return out


def _cluster_id_order(labels: np.ndarray) -> list[int]:
    """Stable cluster ordering for display."""
    labs = np.asarray(labels)
    uniq = np.unique(labs)
    # Remove outlier label if you use one (e.g., -1 or 'outlier')
    # Here, we remove -1 and 'outlier' as examples; adjust as needed
    uniq = [x for x in uniq if x != -1 and x != 'outlier']
    return sorted(uniq, key=str)


# ---------------------------------------------------------------------
#  Plotting functions
# ---------------------------------------------------------------------

def plot_global_metric_over_k(
    model_df: pd.DataFrame,
    *,
    measure: Measure = "average",
    rule: Rule = "1se",
    spec: PlotSpec | None = None,
    ax: mpl.axes.Axes | None = None,
    show_1se: bool = True,
    show_quant: bool = False,
    # legend/key strategy
    show_legend_panel: bool | Literal["auto"] = "auto",
    max_full_annotation: int = 7,
    key_label_wrap: int = 20,
    # aesthetics
    marker: str = "o",
    linewidth: float = 1.6,
    alpha_band: float = 0.18,
    highlight_linewidth: float = 2.8,
    grid_alpha: float = 0.22,
    title: str | None = None,
    y_label: str | None = None,
    x_label: str = "Number of Clusters (k)",
    annotate_selection: bool = True,
) -> dict[str, Any]:
    """
    Plot a global metric vs k for each method configuration (estimator + hyperparams).
    """
    spec = spec or screen_spec()

    # resolve the actual column name in model_df
    if measure not in MEASURE_MAP:
        raise ValueError(f"Unknown measure {measure!r}. Valid options: {list(MEASURE_MAP)}")
    y_col = MEASURE_MAP[measure]

    require_columns(model_df, ["n_clusters", y_col])

    # build method ids/labels (stable grouping)
    df = model_df
    if "_method_id" not in df.columns or "_method_label" not in df.columns:
        df = add_method_columns(df)

    # drop rows without required x/y
    df = df[pd.notna(df["n_clusters"]) & pd.notna(df[y_col])].copy()
    if df.empty:
        raise ValueError(f"No rows to plot for measure={measure!r} (column={y_col!r})")
    df["n_clusters"] = df["n_clusters"].astype(int)

    # decide key panel mode (only possible if we control the figure layout)
    n_methods = df["_method_id"].nunique()
    if ax is None:
        show_legend = (n_methods > max_full_annotation) if show_legend_panel == "auto" else bool(show_legend_panel)
    else:
        show_legend = False

    # --- selection ---
    # note: selection expects full model_df shape; extra cols are fine
    best_row = get_best_row(df, measure=measure, rule=rule, return_idx=False)
    best_k = int(best_row["n_clusters"])
    best_method_id = str(best_row["_method_id"])

    # order methods by their peak performance 
    peak = df.groupby("_method_id", dropna=False)[y_col].max()
    method_order = peak.sort_values(ascending=False).index.tolist()

    # assign short codes for clean plots
    method_codes = {mid: f"M{i+1}" for i, mid in enumerate(method_order)}
    df["_method_code"] = df["_method_id"].map(method_codes)

    method_map = (
        df[["_method_id", "_method_label", "_method_code"]]
        .drop_duplicates()
        .sort_values("_method_code")
        .rename(columns={"_method_code": "method_code"})
        .reset_index(drop=True)
    )

    # --- plotting ---
    with plotting_context(spec):
        # figure / axes creation
        if ax is None:
            if not show_legend:
                fig = plt.figure(figsize=(spec.width_in, spec.height_in), constrained_layout=spec.constrained_layout)
                gs = fig.add_gridspec(1, 2, width_ratios=[0.73, 0.27])
                ax_main = fig.add_subplot(gs[0, 0])
            else:
                fig, ax_main = new_figure(spec=spec)
        else:
            ax_main = ax
            fig = ax.figure

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or [f"C{i}" for i in range(10)]

        def _band_cols(col: str) -> tuple[str | None, str | None, str | None]:
            lower = f"{col}_lower" if f"{col}_lower" in df.columns else None
            upper = f"{col}_upper" if f"{col}_upper" in df.columns else None
            se = f"{col}_se" if f"{col}_se" in df.columns else None
            return lower, upper, se

        lower_col, upper_col, se_col = _band_cols(y_col)

        k_min = int(df["n_clusters"].min())
        k_max = int(df["n_clusters"].max())
        ax_main.set_xlim(k_min - 0.1, k_max + 0.15)
        ax_main.set_ylim(0, 1 + 0.1)

        endpoints = []  # (code, x_end, y_end, color)

        for j, mid in enumerate(method_order):
            g = df[df["_method_id"] == mid].sort_values("n_clusters")
            x = g["n_clusters"].to_numpy()
            y = g[y_col].astype(float).to_numpy()
            c = color_cycle[j % len(color_cycle)]
            code = method_codes[mid]
            
            label = method_map.loc[method_map["method_code"] == code, "_method_label"].values[0]
            wrapped_label = textwrap.fill(label, width=key_label_wrap, break_long_words=False, break_on_hyphens=False)

            lw = highlight_linewidth if mid == best_method_id else linewidth
            z = 4 if mid == best_method_id else 2

            ax_main.plot(x, y, marker=marker, linewidth=lw, label=wrapped_label, color=c, zorder=z)

            if show_1se:
                if se_col is not None and se_col in g.columns:
                    se = g[se_col].astype(float).to_numpy()
                    lo, hi = compute_1se_band(y, se)
                    ax_main.fill_between(x, lo, hi, color=c, alpha=alpha_band, linewidth=0, zorder=1)
                if show_1se and show_quant:
                    warnings.warn(
                        "Both show_1se and show_quant are True; defaulting to only showing 1SE bands. "
                        "To see quantile bands, set show_1se=False.",
                        UserWarning,
                        stacklevel=2,
                    )
            elif show_quant:
                if lower_col is not None and upper_col is not None:
                    lo = g[lower_col].astype(float).to_numpy()
                    hi = g[upper_col].astype(float).to_numpy()
                    ax_main.fill_between(x, lo, hi, color=c, alpha=alpha_band, linewidth=0, zorder=1)

            endpoints.append((code, float(x[-1]), float(y[-1]), c))

        # selection markers
        ax_main.axvline(best_k, linestyle="--", linewidth=1.2, c='#777777', alpha=0.7, zorder=0)
        sel_y = float(best_row[y_col])
        ax_main.scatter([best_k], [sel_y], s=52, zorder=5, edgecolor="none", color="black")

        # labels / title
        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel(y_label or _Y_LABELS.get(y_col, y_col))
        ax_main.set_title(title)

        ax_main.grid(axis="y", alpha=grid_alpha)
        ax_main.xaxis.set_major_locator(MaxNLocator(integer=True))

        # explicit selection annotation
        if annotate_selection:
            best_label = str(best_row["_method_label"])
            best_label_short = textwrap.shorten(best_label, width=20, placeholder="…")
            msg = f"Selected: k*={best_k} \n{best_label_short}"
            ax_main.text(
                0.02, 0.98, msg,
                transform=ax_main.transAxes,
                ha="left", va="top",
                fontsize=max(spec.base_fontsize - 1, 8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.9),
                zorder=10,
            )

        # declutter strategy: legend or key panel + endpoint labels
        if show_legend:
            legend_outside_right(ax_main, title="model")
        else:
            # endpoint code labels with simple collision-avoidance
            ymin, ymax = ax_main.get_ylim()
            yr = max(ymax - ymin, 1e-9)
            min_sep = 0.035 * yr

            endpoints_sorted = sorted(endpoints, key=lambda t: t[2])
            ys = [t[2] for t in endpoints_sorted]
            ys_adj = ys.copy()
            for i in range(1, len(ys_adj)):
                ys_adj[i] = max(ys_adj[i], ys_adj[i - 1] + min_sep)
            overflow = ys_adj[-1] - ymax
            if overflow > 0:
                ys_adj = [y - overflow for y in ys_adj]

            x_pad = 0.14
            for (code, x_end, y_end, col), y_txt in zip(endpoints_sorted, ys_adj):
                label = method_map.loc[method_map["method_code"] == code, "_method_label"].values[0]
                wrapped_label = textwrap.fill(label, width=key_label_wrap, break_long_words=False, break_on_hyphens=False)
                ax_main.annotate(
                    wrapped_label,
                    xy=(x_end, y_end),
                    xytext=(x_end + x_pad, y_txt),
                    textcoords="data",
                    ha="left",
                    va="center",
                    fontsize=max(spec.base_fontsize - 2, 8),
                    color=col,
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=col, alpha=0.9),
                    clip_on=False,
                    zorder=6,
                )

        return {
            "fig": fig,
            "ax": ax_main,
            "best_row": best_row,
            "best_k": best_k,
            "best_method_id": best_method_id,
            "method_map": method_map,
        }


def plot_consensus_matrix(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("phase 3")


def plot_consensus_clustering(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("phase 4")


def plot_clustering(
    carve: Any | None = None,
    *,
    measure: Measure = "stability",
    rule: Rule = "1se",
    k: int | None = None,
    cluster_metric: ClusterMetric = "gini",
    dr: DRFunc | None = None,
    spec: PlotSpec | None = None,
    title: str | None = None,
    point_size: float = 40.0,
    min_point_size: float = 10.0,
    point_alpha: float = 0.85,
    min_point_alpha: float = 0.35,
    grid_alpha: float = 0.15,
    show_scatter_grid: bool = False,
    show_scatter_axes: bool = False,
    show_boxplot_axes: bool = False,
    annotate_selection: bool = True,
) -> dict[str, Any]:
    """
    Scatter (left): 2D projection of X using user-provided DR (default PCA).
    Boxplot (right): cluster-wise distribution of `values`, colored by cluster.

    Selection pathway:
      - If `carve` is provided and labels/values are None, we:
        1) select the best row via get_best_row(model_df_, measure, rule)
        2) get labels via carve.get_optimal_labels(measure, rule, k=...)
        3) pick per-sample vectors from carve.{stab_gini_arrs_, stab_ce_arrs_, misclassification_arrs_}

    Boxplot stats:
      - whiskers: 5th/95th percentiles
      - box: 25th/75th percentiles
      - median line + mean line
      - no fliers (clean)
    """
    spec = spec or screen_spec()

    # --- resolve (X, labels, values) either from explicit inputs or via carve ---
    selected_row = None
    selected_idx = None
    # selected_method_label = None
    selected_k = None

    # if labels is None or values is None:
    if carve is None:
        raise ValueError("Provide `carve` to select (labels, values).")

    # pull X from carve
    X = getattr(carve, "X_", None)
    if X is None:
        raise ValueError("X is required. Did you forget to provide to run fit()?")

    model_df = getattr(carve, "model_df_", None)
    if model_df is None or getattr(model_df, "empty", True):
        raise ValueError("carve.model_df_ is empty. Did you forget to provide to run fit()?")

    df_sel = model_df.copy()
    if k is not None:
        df_sel = df_sel[df_sel["n_clusters"] == int(k)]
        if df_sel.empty:
            raise ValueError(f"No rows found in model_df_ with n_clusters={k}")

    # choose best row under the unified selection rules
    selected_idx = get_best_row(df_sel, measure=measure, rule=rule, return_idx=True) 
    selected_row = df_sel.loc[selected_idx]
    selected_k = int(selected_row["n_clusters"])

    # labels from CARVE (aligned / consistent with your pipeline)
    labels = carve.get_optimal_labels(measure=str(measure), rule=str(rule), k=k)

    # find "pos" (row position in original model_df_) for per-sample arrays
    pos = model_df.index.get_loc(selected_idx)

    # pick per-sample values for boxplots
    if cluster_metric == "gini":
        arrs = getattr(carve, "stab_gini_arrs_", None)
        if arrs is None:
            raise ValueError("carve.stab_gini_arrs_ not found; cannot plot gini stability.")
        values = np.asarray(arrs)[pos]
        # selected_method_label = "stability (gini)"
    elif cluster_metric == "ce":
        arrs = getattr(carve, "stab_ce_arrs_", None)
        if arrs is None:
            raise ValueError("carve.stab_ce_arrs_ not found; cannot plot ce stability.")
        values = np.asarray(arrs)[pos]
        # selected_method_label = "stability (ce)"
    elif cluster_metric == "misclassification":
        arrs = getattr(carve, "misclassification_arrs_", None)
        if arrs is None:
            raise ValueError("carve.misclassification_arrs_ not found; cannot plot generalizability.")
        values = np.asarray(arrs)[pos]
        # selected_method_label = "misclassification"
    else:
        raise ValueError(f"Unknown cluster_metric={cluster_metric!r}")

    # make sure method label exists for annotation, if added it earlier
    # if "_method_label" in model_df.columns:
    #     try:
    #         selected_method_label = str(model_df.loc[selected_idx, "_method_label"])
    #     except Exception:
    #         pass

    # now X, labels, values must exist
    if X is None:
        raise ValueError("X is required.")
    labels = np.asarray(labels)
    # values = np.asarray(values)

    if labels.ndim != 1:
        raise ValueError("Labels must be 1D.")
    if values.ndim != 1:
        raise ValueError("Values must be 1D.")
    if X.shape[0] != labels.shape[0] or X.shape[0] != values.shape[0]:
        raise ValueError("X, labels, and values must have the same number of rows/elements.")

    # --- cluster ordering, names, colors ---
    cluster_ids = _cluster_id_order(labels)
    if len(cluster_ids) == 0:
        raise ValueError("No non-negative cluster labels found to plot.")

    # map cluster id -> C1, C2, ...
    name_map = {cid: f"C{i+1}" for i, cid in enumerate(cluster_ids)}
    tick_names = [name_map[cid] for cid in cluster_ids]

    color_map = cluster_colors(labels)  # cid -> rgba

    # --- prepare boxplot data ---
    by_cluster = _values_by_cluster(values, labels, cluster_ids=cluster_ids)

    # --- project X to 2D ---
    Z = project_2d(np.asarray(X), dr=dr)
    
    # --- scale point sizes by value ---
    vmin, vmax = np.min(values), np.max(values)
    if np.ptp(values) < 1e-6:
        point_sizes = np.full_like(values, point_size, dtype=float)
    else:
        # Linear scaling: min value -> min_size, max value -> max_size
        point_sizes = min_point_size + (values - vmin) / (vmax - vmin) * ((point_size + 10) - min_point_size)
        
    # --- cluster-wise opacity ---
    # Compute mean value per cluster
    cluster_means = {cid: np.mean(values[labels == cid]) for cid in cluster_ids}
    mean_vals = np.array(list(cluster_means.values()))

    # Map mean values to alpha range
    if np.ptp(mean_vals) < 1e-6:
        cluster_alphas = {cid: (point_alpha + 0.1) for cid in cluster_ids}
    else:
        cluster_alphas = {
            cid: min_point_alpha + (cluster_means[cid] - mean_vals.min()) / (mean_vals.max() - mean_vals.min()) * ((point_alpha + 0.1) - min_point_alpha)
            for cid in cluster_ids
        }
    # Assign each sample the alpha of its cluster
    point_alphas = np.array([cluster_alphas[int(l)] for l in labels], dtype=float)

    # --- draw ---
    with plotting_context(spec):
        fig, axs = plt.subplots(
            1, 2,
            figsize=(spec.width_in, spec.height_in),
            constrained_layout=spec.constrained_layout,
            gridspec_kw={"width_ratios": [1, 1]},
        )
        ax_scatter, ax_box = axs

        # scatter: color by cluster, no legend
        cols = np.array([color_map.get(int(l), (0.5, 0.5, 0.5, 1.0)) for l in labels], dtype=float)
        ax_scatter.scatter(
            Z[:, 0], Z[:, 1],
            s=point_sizes,
            alpha=point_alphas,
            c=cols,
            edgecolors="#777777",
            linewidths=0.3,
        )
        
        if show_scatter_axes:
            # Attempt infer DR method name
            if dr is None:
                dr_name = "PC"
            elif hasattr(dr, "__class__") and hasattr(dr, "fit_transform"):
                dr_name = dr.__class__.__name__
            elif callable(dr):
                dr_name = getattr(dr, "__name__", None)
                if dr_name is None and hasattr(dr, "__class__"):
                    dr_name = dr.__class__.__name__
            else:
                dr_name = "Component"
            ax_scatter.set_xlabel(f"{dr_name} 1")
            ax_scatter.set_ylabel(f"{dr_name} 2")
            ax_scatter.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            for spine in ["left", "bottom"]:
                ax_scatter.spines[spine].set_visible(True)
            for spine in ["top", "right"]:
                ax_scatter.spines[spine].set_visible(False)
        else:
            ax_scatter.set_xticks([])
            ax_scatter.set_yticks([])
            for sp in ax_scatter.spines.values():
                sp.set_visible(False)

        # boxplot: 5/25/50/75/95 + mean line, colored per cluster
        bp = ax_box.boxplot(
            by_cluster,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            whis=(5, 95),
            showfliers=True,
            widths=0.65,
        )

        # style the boxes to match cluster colors
        for i, cid in enumerate(cluster_ids):
            c = color_map.get(int(cid), (0.5, 0.5, 0.5, 1.0))
            bp["boxes"][i].set(facecolor=c, alpha=0.75, edgecolor="black", linewidth=0.8)

        for w in bp["whiskers"]:
            w.set(color="black", linewidth=0.8)
        for cap in bp["caps"]:
            cap.set(color="black", linewidth=0.8)

        # median + mean lines (discernible)
        for med in bp["medians"]:
            med.set(color="black", linewidth=1.4)
        for mean in bp["means"]:
            mean.set(color="black", linewidth=1.4, alpha=0.75)

        ax_box.set_xticks(np.arange(1, len(cluster_ids) + 1))
        ax_box.set_xticklabels(tick_names)
        # xticklabels = ax_box.get_xticklabels()
        # for lbl, cid in zip(xticklabels, cluster_ids):
        #     c = color_map.get(int(cid), (0.5, 0.5, 0.5, 1.0))
        #     lbl.set_color(c)
            
        ax_box.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax_box.yaxis.tick_right()
        ax_box.yaxis.set_label_position("right")
        ax_box.set_ylabel(_BOX_YLABELS.get(cluster_metric, cluster_metric))
        
        if show_boxplot_axes:
            ax_box.spines["right"].set_visible(True)
            ax_box.spines["left"].set_visible(False)
        else:
            for spine in ["top", "bottom", "left", "right"]:
                ax_box.spines[spine].set_visible(False)
            ax_box.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
            ax_box.tick_params(axis="y", which="both", left=False, right=True, labelleft=False, labelright=True)

        if show_scatter_grid:
            ax_box.grid(True, axis="y", alpha=grid_alpha)
        else:
            ax_box.grid(False)

        # titles
        # if title is None:
        #     title = "clustering + cluster-wise scores"
        fig.suptitle(title, y=0.98)

        # selection annotation
        if annotate_selection and (selected_k is not None):
            model_df = getattr(carve, "model_df_", None)
            if model_df is None or getattr(model_df, "empty", True):
                raise ValueError("carve.model_df_ is empty. Did you forget to provide to run fit()?")

            if "_method_label" not in model_df.columns or "_method_id" not in model_df.columns:
                model_df = add_method_columns(model_df)

            selected_row_dup = model_df.loc[selected_idx]
            best_label = str(selected_row_dup["_method_label"])
            wrapped_label = textwrap.fill(best_label, width=20, break_long_words=False, break_on_hyphens=False)
            msg = f"Selected: k*={selected_k}\n{wrapped_label}"
            ax_scatter.text(
                0.02, 0.98, msg,
                transform=ax_scatter.transAxes,
                ha="left", va="top",
                fontsize=max(spec.base_fontsize - 1, 8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.90),
            )

        return {
            "fig": fig,
            "ax_scatter": ax_scatter,
            "ax_box": ax_box,
            "cluster_ids": cluster_ids,
            "cluster_names": tick_names,
            "selected_idx": selected_idx,
            "selected_row": selected_row,
            "selected_k": selected_k,
        }
