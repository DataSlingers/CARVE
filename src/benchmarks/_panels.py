"""Drawing primitives. Every one takes an ax and returns it.

Nothing here creates a figure, calls plt.show, or saves anything. Composition
is the job of the figures package; these only draw.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from carve._sweep import SWEEP_REGISTRY

from ._registry import (
    BASELINE_METRIC,
    CARVE_METRICS_ALL,
    CVI_METRICS,
    METRIC_DISPLAY_NAMES,
    METRIC_LEGEND_NAMES,
)
from ._theme import (
    FALLBACK_COLOR,
    FONT_SIZES,
    FOREGROUND_COLOR,
    MEASURE_LINESTYLES,
    PIPELINE_CMAP_NAME,
    SEQUENTIAL_CMAP_NAME,
    cluster_colors,
    metric_color,
    metric_linestyle,
    metric_linewidth,
    style_axes,
)


def _display(metric: str) -> str:
    """The name a figure legend gives a metric.

    Prefers METRIC_LEGEND_NAMES, which carries the handful of names that read
    differently in a legend than in a table row.
    """
    return METRIC_LEGEND_NAMES.get(metric, METRIC_DISPLAY_NAMES.get(metric, metric))


def cluster_color_map(labels: np.ndarray) -> dict[Any, str]:
    """Map each distinct label to a stable color."""
    unique = list(dict.fromkeys(np.asarray(labels).tolist()))
    return dict(zip(unique, cluster_colors(len(unique))))


#: OKLab (Ottosson 2020): linear sRGB to LMS, then cube-rooted LMS to Lab.
_LMS_FROM_LINEAR_RGB = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_OKLAB_FROM_LMS = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)

#: CLUSTER_PALETTE's first ten entries are tab10.
_TAB10_HEAD = 10

#: The OKLab lightness band a matched label's color is drawn from after tab10.
#: Below it, a color under the scatter's black marker outlines reads as near
#: black; above it, it barely shows on the white background.
_MATCHED_LIGHTNESS = (0.5, 0.9)


def _oklab(color: str) -> np.ndarray:
    """A color's OKLab (L, a, b), L running from 0 (black) to 1 (white)."""
    rgb = np.asarray(to_rgba(color)[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return _OKLAB_FROM_LMS @ np.cbrt(_LMS_FROM_LINEAR_RGB @ linear)


def _distinct_first(palette: Sequence[str]) -> list[int]:
    """Palette indices in the order matched labels take them.

    tab10's entries first, in order; then the entries inside
    _MATCHED_LIGHTNESS, each the one farthest in OKLab from every entry
    already taken; then the rest in palette order.
    """
    lab = [_oklab(color) for color in palette]
    order = list(range(min(_TAB10_HEAD, len(palette))))
    low, high = _MATCHED_LIGHTNESS
    pool = [i for i in range(len(order), len(palette)) if low <= lab[i][0] <= high]
    while pool:
        farthest = max(
            pool, key=lambda i: min(np.linalg.norm(lab[i] - lab[j]) for j in order)
        )
        order.append(farthest)
        pool.remove(farthest)
    return order + [i for i in range(len(palette)) if i not in order]


def aligned_color_maps(
    y_true: np.ndarray, *clusterings: np.ndarray, matched_first: bool = False
) -> tuple[dict[Any, str], ...]:
    """One palette shared by the reported labels and every clustering.

    Returns ``(true_cmap, *clustering_cmaps)``. The reported labels take the
    palette in sorted category order; each clustering's integer id indexes
    the same palette directly, so a cluster that ``align_cluster_labels``
    matched to reported label *i* is drawn in reported label *i*'s color.

    The palette is sized to cover both the reported labels and the highest
    cluster id present, so a clustering with more clusters than there are
    reported labels still gets a color for every one of them.

    With matched_first, the reported labels a clustering was matched to take
    their colors first, in _distinct_first's order, and the rest take what is
    left in palette order. The reported labels still share one set of colors
    and every matched cluster keeps its label's color; only which label gets
    which color changes. Without it, a clustering coarser than the reported
    labels can match labels whose palette entries are dark.
    """
    categories = sorted(set(np.asarray(y_true).tolist()))
    highest_id = 0
    for labels in clusterings:
        values = np.asarray(labels)
        if values.size:
            highest_id = max(highest_id, int(np.max(values)) + 1)

    palette = cluster_colors(max(len(categories), highest_id))
    if matched_first:
        n = len(categories)
        matched = sorted(
            {int(c) for labels in clusterings for c in np.unique(labels) if c < n}
        )
        front = _distinct_first(palette[:n])[: len(matched)]
        unmatched = [code for code in range(n) if code not in matched]
        rest = [i for i in range(n) if i not in front]
        slot = dict(zip(matched, front)) | dict(zip(unmatched, rest))
        palette = [palette[slot[code]] for code in range(n)] + palette[n:]
    true_cmap = {category: palette[i] for i, category in enumerate(categories)}
    cluster_cmaps = [
        {int(c): palette[int(c)] for c in np.unique(np.asarray(labels))}
        for labels in clusterings
    ]
    return (true_cmap, *cluster_cmaps)


def scatter_clusters(
    ax: Axes,
    Z: np.ndarray,
    labels: np.ndarray,
    *,
    color_map: Mapping[Any, str] | None = None,
    s: float = 20.0,
    alpha: float = 0.85,
    linewidth: float = 0.3,
    edgecolor: str = FOREGROUND_COLOR,
    title: str | None = None,
    hide_axes: bool = True,
    axis_labels: Sequence[str] | None = None,
) -> Axes:
    """Scatter a two-dimensional embedding, colored by label.

    Markers carry a thin ``edgecolor`` outline, which is what separates
    overlapping points where two clusters meet; the case-study scatters
    are dense enough that unoutlined markers merge into a single mass at
    every cluster boundary. Pass ``edgecolor="none"`` for a scatter dense
    enough that the outlines themselves would dominate.
    """
    Z = np.asarray(Z)
    labels = np.asarray(labels)
    color_map = color_map or cluster_color_map(labels)

    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=s,
            alpha=alpha,
            linewidth=linewidth,
            edgecolor=edgecolor,
            color=color_map.get(label, FALLBACK_COLOR),
            label=str(label),
        )

    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0], fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel(axis_labels[1], fontsize=FONT_SIZES["axis_label"])
    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    else:
        style_axes(ax)
    return ax


def axis_arrows(
    ax: Axes,
    labels: Sequence[str] = ("PC1", "PC2"),
    *,
    origin: tuple[float, float] = (0.03, 0.06),
    length: tuple[float, float] = (0.10, 0.14),
    pad: float = 0.02,
) -> Axes:
    """Draw a short arrow pair in the lower-left corner naming the axes.

    A hidden-spine scatter still has to say what its two directions are. An
    xlabel/ylabel pair does that, but it also reserves a full margin on two
    sides of every panel it is applied to; a corner marker says the same
    thing inside the data area.

    ``origin``, ``length`` and ``pad`` are axes fractions, so the marker
    keeps its size and position whatever the panel's data limits are.
    """
    x0, y0 = origin
    dx, dy = length

    for end in ((x0 + dx, y0), (x0, y0 + dy)):
        ax.annotate(
            "",
            xy=end,
            xytext=(x0, y0),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": FOREGROUND_COLOR, "lw": 1.2},
        )

    ax.annotate(
        labels[0],
        xy=(x0 + dx + pad / 2, y0),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=FONT_SIZES["axis_label"],
    )
    ax.annotate(
        labels[1],
        xy=(x0, y0 + dy + pad),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZES["axis_label"],
    )
    return ax


def _dodge_offsets(metrics: Sequence[str], dodge: float) -> dict[str, float]:
    """Per-metric x offsets, centered on zero.

    Seven methods' per-seed clouds drawn at one x merge into a vertical band
    that carries no per-method information. Spreading them symmetrically
    keeps each method's spread readable while leaving the mean lines on the
    real axis positions.
    """
    if dodge <= 0 or len(metrics) < 2:
        return dict.fromkeys(metrics, 0.0)
    span = dodge * (len(metrics) - 1)
    return {metric: -span / 2 + index * dodge for index, metric in enumerate(metrics)}


def metric_lines(
    ax: Axes,
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    x_col: str = "axis_value",
    x_label: str | None = None,
    y_label: str | None = None,
    element_scale: float = 1.0,
    show_legend: bool = True,
    show_points: bool = False,
    point_alpha: float = 0.28,
    dodge: float = 0.0,
) -> Axes:
    """Plot mean ARI at the selected k against the axis, one line per metric.

    Reads the unified artifact schema.

    ``"baseline_oracle"`` may be listed in ``metrics``. It is drawn from the
    deduplicated ``oracle_ari`` column, one value per axis point and seed,
    instead of the ``is_selected`` rows the other metrics use.

    show_points draws each dataset's own value behind the mean, dodged along
    x by dodge so seven methods' clouds do not merge into one band. It is
    off by default; Fig 4 shows means and error bars only.
    """
    selected = df.loc[df["is_selected"]]
    offsets = _dodge_offsets(list(metrics), dodge)

    for metric in metrics:
        if metric == "baseline_oracle":
            oracle = df[[x_col, "seed", "oracle_ari"]].drop_duplicates(
                subset=[x_col, "seed"]
            )
            if oracle.empty:
                continue
            grouped = oracle.groupby(x_col)["oracle_ari"]
            centers = grouped.mean()
            errors = grouped.sem()
            ax.errorbar(
                centers.index,
                centers.to_numpy(),
                yerr=errors.to_numpy(),
                marker="none",
                linewidth=metric_linewidth(metric) * element_scale,
                linestyle=metric_linestyle(metric),
                capsize=2.5 * element_scale,
                color=metric_color(metric),
                label=_display(metric),
            )
            if show_points:
                ax.scatter(
                    oracle[x_col].to_numpy(dtype=float) + offsets[metric],
                    oracle["oracle_ari"].to_numpy(dtype=float),
                    s=9.0 * element_scale,
                    color=metric_color(metric),
                    alpha=point_alpha,
                    linewidth=0.0,
                    zorder=1,
                    label="_points",
                )
            continue

        sub = selected.loc[selected["metric_name"] == metric]
        if sub.empty:
            continue
        grouped = sub.groupby(x_col)["ari_at_k"]
        centers = grouped.mean()
        errors = grouped.sem()
        ax.errorbar(
            centers.index,
            centers.to_numpy(),
            yerr=errors.to_numpy(),
            marker="o",
            markersize=5.0 * element_scale,
            linewidth=metric_linewidth(metric) * element_scale,
            linestyle=metric_linestyle(metric),
            capsize=2.5 * element_scale,
            color=metric_color(metric),
            label=_display(metric),
        )
        if show_points:
            ax.scatter(
                sub[x_col].to_numpy(dtype=float) + offsets[metric],
                sub["ari_at_k"].to_numpy(dtype=float),
                s=9.0 * element_scale,
                color=metric_color(metric),
                alpha=point_alpha,
                linewidth=0.0,
                zorder=1,
                label="_points",
            )

    ax.set_xlabel(x_label or x_col, fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(
        y_label or r"ARI (selected $\hat{k}$ vs. true labels)",
        fontsize=FONT_SIZES["axis_label"],
    )
    if show_legend:
        ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)


def k_hat_selection_matrix(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    axis_label: str,
    candidate_k: Sequence[int],
) -> np.ndarray:
    """P(k-hat = k) as a metrics-by-k array, NaN where a metric has no rows.

    Row order follows metrics and column order follows candidate_k, both as
    given, so the caller's ordering is the figure's ordering. A metric with
    no rows in this cell is an all-NaN row, so rows stay aligned with
    metrics.
    """
    cell = df.loc[df["axis_label"] == axis_label]
    matrix = np.full((len(metrics), len(candidate_k)), np.nan)
    for row, metric in enumerate(metrics):
        selected = cell.loc[
            (cell["metric_name"] == metric) & cell["is_selected"].astype(bool)
        ]
        if selected.empty:
            continue
        counts = selected["k"].value_counts()
        total = float(counts.sum())
        for column, k in enumerate(candidate_k):
            matrix[row, column] = float(counts.get(k, 0)) / total
    return matrix


def k_hat_heatmap(
    ax: Axes,
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    axis_label: str,
    candidate_k: Sequence[int],
    k_star: int,
    annotate: bool = True,
    title: str | None = None,
) -> Axes:
    """How often each method chose each k, as a metrics-by-k heatmap.

    This is the k-recovery column of the supplementary tables drawn rather
    than tabulated. A mean ARI hides the shape of a method's selections: a
    method that picks k* half the time and k*+2 the other half scores the
    same as one that always picks something in between, and only this panel
    separates them.

    The oracle has no selected k (it is fit at k_star), so it is not drawn.
    """
    metrics = [m for m in metrics if m != BASELINE_METRIC]
    candidate_k = list(candidate_k)
    matrix = k_hat_selection_matrix(
        df, metrics=metrics, axis_label=axis_label, candidate_k=candidate_k
    )

    ax.imshow(
        matrix,
        cmap=SEQUENTIAL_CMAP_NAME,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(range(len(candidate_k)))
    ax.set_xticklabels([str(k) for k in candidate_k], fontsize=FONT_SIZES["tick"])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([_display(m) for m in metrics], fontsize=FONT_SIZES["tick"])
    ax.set_xlabel("$k$", fontsize=FONT_SIZES["axis_label"])

    if k_star in candidate_k:
        ax.axvline(
            candidate_k.index(k_star),
            color=FOREGROUND_COLOR,
            linewidth=1.2,
            alpha=0.55,
        )

    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isnan(value) or value < 0.005:
                    continue
                # Light text on the dark end of the ramp, dark on the light
                # end.
                ax.text(
                    column,
                    row,
                    f"{value:.2f}".lstrip("0") or "0",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZES["tick"] * 0.85,
                    color="#FFFFFF" if value > 0.55 else FOREGROUND_COLOR,
                )

    if title is not None:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.grid(False)
    return ax


CRITERION_REFERENCE_LABEL: str = r"ARI at $k$ (base estimator)"


def normalized_criterion(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    axis_label: str,
) -> pd.DataFrame:
    """Each criterion min-max normalized over k within a dataset, then averaged.

    Returns long-form rows of (metric_name, k, value).

    The criteria live on different scales (silhouette in [-1, 1],
    Calinski-Harabasz in the thousands, the gap statistic a log ratio), and
    the panel asks which k each one prefers, which normalization preserves.
    Normalizing within each dataset first keeps a dataset with large raw
    values from dominating the mean. A criterion that is flat across k
    normalizes to 0.5.
    """
    cell = df.loc[df["axis_label"] == axis_label]
    out: list[dict[str, Any]] = []
    for metric in metrics:
        sub = cell.loc[cell["metric_name"] == metric]
        if sub.empty:
            continue
        per_seed = []
        for _, group in sub.groupby("seed"):
            values = group.set_index("k")["metric_value"].astype(float).sort_index()
            span = values.max() - values.min()
            if not np.isfinite(span) or span <= 0:
                per_seed.append(pd.Series(0.5, index=values.index))
            else:
                per_seed.append((values - values.min()) / span)
        averaged = pd.concat(per_seed, axis=1).mean(axis=1)
        out.extend(
            {"metric_name": metric, "k": int(k), "value": float(v)}
            for k, v in averaged.items()
        )
    return pd.DataFrame(out, columns=["metric_name", "k", "value"])


def criterion_curves(
    ax: Axes,
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    axis_label: str,
    candidate_k: Sequence[int],
    k_star: int,
    show_ari_reference: bool = True,
    title: str | None = None,
) -> Axes:
    """Each method's normalized criterion over k, for one point on the axis.

    Every criterion is normalized to [0, 1] over the candidate k. The base
    estimator's own ARI at each k is drawn on the same axis as a reference,
    styled like the oracle baseline in metric_lines, which it equals at
    k = k_star.

    The reference is the mean, over seeds, of the classical indices'
    ari_at_k rows (CVI_METRICS), which carry the base estimator's full-data
    fit at each k (see _run.run_cell). Nothing is drawn when the cell has no
    classical-index rows.

    Each metric's selected k is marked on its own curve. The gap statistic's
    mark need not sit on its maximum: Tibshirani's rule takes the smallest k
    with Gap(k) >= Gap(k+1) - s(k+1).
    """
    metrics = [m for m in metrics if m != BASELINE_METRIC]
    candidate_k = list(candidate_k)
    cell = df.loc[df["axis_label"] == axis_label]

    if show_ari_reference:
        base = cell.loc[cell["metric_name"].isin(CVI_METRICS)].drop_duplicates(
            subset=["seed", "k"]
        )
        if not base.empty:
            grouped = base.groupby("k")["ari_at_k"]
            ax.errorbar(
                candidate_k,
                grouped.mean().reindex(candidate_k).to_numpy(),
                yerr=grouped.sem().reindex(candidate_k).to_numpy(),
                marker="none",
                linewidth=metric_linewidth(BASELINE_METRIC),
                linestyle=metric_linestyle(BASELINE_METRIC),
                capsize=2.5,
                color=metric_color(BASELINE_METRIC),
                zorder=1,
                label=CRITERION_REFERENCE_LABEL,
            )

    curves = normalized_criterion(cell, metrics=metrics, axis_label=axis_label)
    selected = cell.loc[cell["is_selected"].astype(bool)]

    for metric in metrics:
        sub = curves.loc[curves["metric_name"] == metric]
        if sub.empty:
            continue
        values = sub.set_index("k")["value"].reindex(candidate_k)
        ax.plot(
            candidate_k,
            values.to_numpy(),
            color=metric_color(metric),
            linewidth=metric_linewidth(metric),
            linestyle=metric_linestyle(metric),
            zorder=2,
            label=_display(metric),
        )
        picks = selected.loc[selected["metric_name"] == metric, "k"]
        if picks.empty:
            continue
        # The modal selection across datasets, which is the k this panel's
        # averaged curve is being read as having chosen.
        modal = int(picks.mode().iloc[0])
        if modal in candidate_k:
            ax.scatter(
                [modal],
                [values.loc[modal]],
                s=46.0,
                marker="D",
                facecolor=metric_color(metric),
                edgecolor="#FFFFFF",
                linewidth=0.9,
                zorder=3,
                label="_selected",
            )

    if k_star in candidate_k:
        ax.axvline(k_star, color=FOREGROUND_COLOR, linewidth=1.2, alpha=0.45, zorder=0)

    ax.set_xticks(candidate_k)
    ax.set_xlabel("$k$", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Normalized criterion", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(-0.05, 1.05)
    if title is not None:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    return style_axes(ax)


RUNTIME_COLS: tuple[str, str] = ("t_stability_s", "t_generalizability_s")
RUNTIME_LABELS: tuple[str, str] = ("CARVE Stability", "CARVE Generalizability")
_RUNTIME_METRIC_FOR_COLOR = {
    "t_stability_s": "ari_stability_1se",
    "t_generalizability_s": "ari_generalizability_1se",
    "t_default_s": "ari_average_1se",
}


def runtime_lines(
    ax: Axes,
    runtimes_df: pd.DataFrame,
    *,
    runtime_cols: Sequence[str] = RUNTIME_COLS,
    runtime_labels: Sequence[str] = RUNTIME_LABELS,
    x_col: str = "axis_value",
    x_label: str | None = None,
    y_label: str = "Runtime (seconds)",
    yscale: str = "log",
    dodge: float = 0.01,
    element_scale: float = 1.0,
    show_legend: bool = True,
) -> Axes:
    """Plot CARVE wall-clock against the swept axis, one line per timed mode.

    Two series by default, matching the published figure: the stability-mode
    and generalizability-mode fits are timed separately. They are dodged
    horizontally by a fraction of the x-range so the error bars do not
    overlap at each anchor.
    """
    if len(runtime_cols) != len(runtime_labels):
        raise ValueError("runtime_cols and runtime_labels must be the same length.")

    x_values = np.sort(runtimes_df[x_col].unique().astype(float))
    span = float(x_values[-1] - x_values[0]) if len(x_values) > 1 else 1.0
    offsets = np.linspace(
        -dodge * span * (len(runtime_cols) - 1) / 2,
        dodge * span * (len(runtime_cols) - 1) / 2,
        len(runtime_cols),
    )

    for column, label, offset in zip(runtime_cols, runtime_labels, offsets):
        if column not in runtimes_df.columns:
            continue
        grouped = runtimes_df.groupby(x_col)[column]
        centers = grouped.mean().dropna()
        if centers.empty:
            continue
        errors = grouped.sem().reindex(centers.index)

        ax.errorbar(
            centers.index.to_numpy(dtype=float) + offset,
            centers.to_numpy(),
            yerr=errors.to_numpy(),
            marker="o",
            markersize=5.0 * element_scale,
            linewidth=1.8 * element_scale,
            capsize=2.5 * element_scale,
            color=metric_color(_RUNTIME_METRIC_FOR_COLOR.get(column, column)),
            label=label,
        )

    ax.set_yscale(yscale)
    ax.set_xlabel(x_label or x_col, fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(y_label, fontsize=FONT_SIZES["axis_label"])
    if show_legend:
        ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)


# A Leiden or Louvain configuration as CARVE renders it ("LeidenClustering,
# n_neighbors=30, ..."), as the CVI sweep does ("LeidenClustering
# (n_neighbors=30, ...)") or one parameter per line, as in CARVE's notes.
_GRAPH_SETTING = re.compile(
    r"(Leiden|Louvain)Clustering(?:[,\s]+|\s*\()"
    r"(\w+=[^,()\s]+(?:[,\s]+\w+=[^,()\s]+)*)\)?"
)


def compact_setting(text: str) -> str:
    """Shorten every graph-clusterer configuration in text for a legend or title.

    "LeidenClustering, n_neighbors=30, objective_function=modularity,
    weighting=jaccard" reads "Leiden, 30 neighbors, Jaccard". Modularity, the
    registered objective, is left unsaid; another objective is named, since it
    changes the resolution scale. Text without such a configuration is
    returned unchanged.
    """

    def shorten(match: re.Match) -> str:
        params = dict(re.findall(r"(\w+)=([^,()\s]+)", match.group(2)))
        parts = [match.group(1)]
        if "n_neighbors" in params:
            parts.append(f"{params['n_neighbors']} neighbors")
        if params.get("weighting") == "jaccard":
            parts.append("Jaccard")
        if params.get("objective_function", "modularity") != "modularity":
            parts.append(params["objective_function"])
        return ", ".join(parts)

    return _GRAPH_SETTING.sub(shorten, text)


def _sweep_axis_label(param: str) -> str:
    """The x-axis label for a sweep parameter, in CARVE's own wording."""
    if param == "n_clusters":
        return "Number of clusters $k$"
    known = SWEEP_REGISTRY.get(param)
    return known[2] if known else param.replace("_", " ").title()


def carve_lines(
    ax: Axes,
    carve_obj: Any,
    *,
    measures: Sequence[str] = ("stability", "generalizability"),
    rule: str = "1se",
    not_two: bool = False,
    title: str | None = None,
    annotate: bool = False,
    show_selected_k: bool = True,
) -> Axes:
    """Plot CARVE validation curves over the sweep axis, one line per measure.

    ``estimator_results_`` has one row per (configuration, k), so a case
    study that sweeps several estimators (Klein: Ward and spectral) carries
    all of them. Each line is restricted to the configuration CARVE's 1-SE
    selection returns for that measure, by its ``method_id``, and the legend
    names that estimator.

    The x axis is the run's sweep parameter, read from carve_obj.sweep_: the
    number of clusters for a k-based run (and for a fit cached before sweep_
    existed, every one of which is k-based), the swept value otherwise, such
    as Leiden resolution. The selected value is marked from get_k or
    get_sweep_value to match. rule and not_two are forwarded to every
    selection call.
    """
    results = carve_obj.estimator_results_
    sweep = getattr(carve_obj, "sweep_", None)
    param = "n_clusters" if sweep is None else sweep.param
    by_k = param == "n_clusters"
    x_col = "n_clusters" if by_k else "sweep_value"

    for measure in measures:
        selected_row, _, _, _ = carve_obj._select_row(
            measure=measure, rule=rule, not_two=not_two
        )
        method_id = selected_row["method_id"]
        method_label = selected_row["method_label"]
        curve = results.loc[results["method_id"] == method_id].sort_values(x_col)

        color = metric_color(f"ari_{measure}_1se")
        xs = curve[x_col].to_numpy()
        values = curve[f"ari_{measure}"].to_numpy()
        label = f"{_display(f'ari_{measure}_1se')} — {method_label}"
        ax.plot(
            xs,
            values,
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=color,
            label=label,
        )
        if f"ari_{measure}_se" in curve.columns:
            se = curve[f"ari_{measure}_se"].to_numpy()
            ax.fill_between(xs, values - se, values + se, color=color, alpha=0.15)

        if show_selected_k:
            if by_k:
                selected = int(
                    carve_obj.get_k(measure=measure, rule=rule, not_two=not_two)
                )
                note = f"$\\hat{{k}}={selected}$"
            else:
                selected = float(
                    carve_obj.get_sweep_value(
                        measure=measure, rule=rule, not_two=not_two
                    )
                )
                note = f"{selected:g}"
            ax.axvline(selected, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
            if annotate:
                ax.annotate(
                    note,
                    xy=(selected, float(np.nanmax(values))),
                    fontsize=FONT_SIZES["legend"],
                    color=color,
                )

    ax.set_xlabel(_sweep_axis_label(param), fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("ARI", fontsize=FONT_SIZES["axis_label"])
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)


def pipeline_lines(
    ax: Axes,
    carve_obj: Any,
    *,
    method_id: str,
    measures: Sequence[str] = ("stability", "generalizability"),
    rule: str = "1se",
    not_two: bool = False,
    title: str | None = None,
) -> Axes:
    """Plot per-pipeline validation curves for one estimator configuration.

    The drawing is carve._plotting.plot_metric_by_pipeline, called once per
    measure on the same axes, so the per-pipeline rows, the error bars at one
    standard error and the line at CARVE's selected sweep value, read from
    estimator_results_, are CARVE's own; this only themes them. Pipelines
    take PIPELINE_CMAP_NAME's colors, each measure its MEASURE_LINESTYLES
    style, and one legend names both.
    """
    from carve._plotting import plot_metric_by_pipeline

    table = carve_obj.preprocessing_results_
    for measure in measures:
        plot_metric_by_pipeline(
            table,
            estimator_df=carve_obj.estimator_results_,
            method_id=method_id,
            measure=measure,
            rule=rule,
            not_two=not_two,
            ax=ax,
            legend=False,
            palette=PIPELINE_CMAP_NAME,
            linestyle=MEASURE_LINESTYLES[measure],
        )

    handles: dict[str, Line2D] = {}
    for container in ax.containers:
        label = container.get_label()
        if label not in handles:
            handles[label] = Line2D(
                [], [], color=container.lines[0].get_color(), marker="o", label=label
            )
    for measure in measures:
        handles[measure] = Line2D(
            [],
            [],
            color=FOREGROUND_COLOR,
            linestyle=MEASURE_LINESTYLES[measure],
            label=measure.capitalize(),
        )
    ax.legend(
        handles=list(handles.values()), fontsize=FONT_SIZES["legend"], frameon=False
    )

    ax.set_xlabel(
        _sweep_axis_label(str(table["sweep_param"].iloc[0])),
        fontsize=FONT_SIZES["axis_label"],
    )
    ax.set_ylabel("ARI", fontsize=FONT_SIZES["axis_label"])
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    return style_axes(ax)


def cvi_lines(
    ax: Axes,
    curves_df: pd.DataFrame,
    best_df: pd.DataFrame,
    *,
    title: str | None = None,
    normalize: bool = True,
) -> Axes:
    """Plot classical index curves over k, marking each index's selected k.

    ``curves_df`` has one row per (metric, model, k), so a case study that
    sweeps several estimators carries every model's scores for every index.
    Each line is restricted to ``best_df``'s winning model for that metric,
    named in the legend, and the selected-k marker is read from that row's
    own ``k``.

    Indices live on incompatible scales, so they are min-max normalized to a
    common axis by default; the selected k is unaffected by that rescaling.

    Curves from cvi_resolution_sweep carry a resolution column. Their k is an
    outcome that two resolutions can share, so the lines and the selection
    are placed by resolution instead.
    """
    x_col = "resolution" if "resolution" in curves_df.columns else "k"
    for _, best in best_df.iterrows():
        metric = best["metric"]
        model = best["model"]
        sub = curves_df.loc[
            (curves_df["metric"] == metric) & (curves_df["model"] == model)
        ].sort_values(x_col)
        if sub.empty:
            continue

        values = sub["score"].to_numpy(dtype=float)
        if normalize:
            span = np.nanmax(values) - np.nanmin(values)
            values = (values - np.nanmin(values)) / span if span > 0 else values * 0.0

        color = metric_color(str(metric))
        ax.plot(
            sub[x_col].to_numpy(),
            values,
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            color=color,
            label=f"{_display(str(metric))} — {model}",
        )

        # A vertical rule at the selected k, as carve_lines marks its
        # selections.
        ax.axvline(
            float(best[x_col]), color=color, linestyle="--", linewidth=1.0, alpha=0.6
        )

    ax.set_xlabel(
        _sweep_axis_label("resolution" if x_col == "resolution" else "n_clusters"),
        fontsize=FONT_SIZES["axis_label"],
    )
    ax.set_ylabel(
        "Score (normalized)" if normalize else "Score",
        fontsize=FONT_SIZES["axis_label"],
    )
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)


def m3c_lines(
    ax: Axes,
    scores: pd.DataFrame,
    *,
    selected_k: int,
    title: str | None = None,
) -> Axes:
    """Plot M3C's RCSI over K with its own error bars, marking the selection.

    RCSI is M3C's selection statistic: the mean difference, on the log scale,
    between the reference stability scores and the real one at each K. It is
    drawn unnormalized, unlike cvi_lines' indices, because it has a
    meaningful zero: RCSI at or below zero means the real data is no more
    stable at that K than M3C's Monte Carlo reference.

    The error bars are plus or minus 1.96 RCSI_SE, which is M3C's own plot
    idiom, and the dashed vertical marks the selected K the way cvi_lines and
    carve_lines mark theirs. The Monte Carlo p-value at selected_k is
    annotated in the corner, read from the scores frame.
    """
    ordered = scores.sort_values("K")
    color = metric_color("m3c_rcsi")
    p_value = float(scores.loc[scores["K"] == selected_k, "MONTECARLO_P"].iloc[0])

    ax.errorbar(
        ordered["K"].to_numpy(),
        ordered["RCSI"].to_numpy(dtype=float),
        yerr=1.96 * ordered["RCSI_SE"].to_numpy(dtype=float),
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        color=color,
        ecolor=color,
        elinewidth=1.0,
        capsize=3.0,
        label="M3C RCSI",
    )
    ax.axvline(int(selected_k), color=color, linestyle="--", linewidth=1.0, alpha=0.6)

    ax.annotate(
        f"Monte Carlo $p$ = {p_value:.3f}",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=FONT_SIZES["legend"],
        color=FOREGROUND_COLOR,
    )

    ax.set_xlabel("Number of clusters $k$", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("RCSI", fontsize=FONT_SIZES["axis_label"])
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)


#: Eigenvalues below this fraction of the largest are numerically zero: the
#: cell-by-cell matrix SC3 thresholds has rank at most its feature count, so
#: Levine's 32 markers leave every eigenvalue past the 31st at round-off.
_SPECTRUM_FLOOR = 1e-8


def sc3_spectrum(
    ax: Axes,
    spectrum: pd.DataFrame,
    *,
    bound: float,
    k: int,
    title: str | None = None,
) -> Axes:
    """Plot the eigenvalues SC3's Tracy-Widom estimate counts, and the bound.

    sc3_estimate_k counts the eigenvalues that lie above the bound; that count
    is SC3's k. They are drawn on a log scale, filled above the bound and
    hollow below it, with the bound dashed, so the reader sees the k points
    SC3 counted. Numerically zero eigenvalues are left out, and of the rest
    the leading max(3k, 30) are drawn.
    """
    ordered = spectrum.sort_values("index")
    index = ordered["index"].to_numpy()
    values = ordered["eigenvalue"].to_numpy(dtype=float)
    keep = values > values.max() * _SPECTRUM_FLOOR
    n_shown = max(3 * int(k), 30)
    index, values = index[keep][:n_shown], values[keep][:n_shown]
    above = values > bound
    color = metric_color("sc3")

    ax.scatter(
        index[above],
        values[above],
        s=18,
        color=color,
        zorder=3,
        label="Above the bound",
    )
    ax.scatter(
        index[~above],
        values[~above],
        s=18,
        facecolors="none",
        edgecolors=FALLBACK_COLOR,
        linewidths=0.8,
        zorder=3,
        label="Below the bound",
    )
    ax.axhline(
        bound,
        color=color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Tracy-Widom bound",
    )
    ax.set_yscale("log")
    ax.annotate(
        f"k = {int(k)} eigenvalues above the bound",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=FONT_SIZES["legend"],
        color=FOREGROUND_COLOR,
    )
    ax.set_xlabel("Eigenvalue index", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Eigenvalue", fontsize=FONT_SIZES["axis_label"])
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.88),
    )
    return style_axes(ax)


def _stack_segments(
    sizes: Sequence[int], gap_frac: float = 0.015
) -> list[tuple[float, float]]:
    """Return (bottom, top) spans for a stacked bar with proportional gaps.

    Stacks downward from y=1, so the first entry sits at the top of the
    column and reading order matches the order the caller passed.
    """
    total = float(sum(sizes))
    if total <= 0:
        return []
    usable = 1.0 - gap_frac * max(len(sizes) - 1, 0)
    spans = []
    cursor = 1.0
    for size in sizes:
        height = usable * (size / total)
        spans.append((cursor - height, cursor))
        cursor -= height + gap_frac
    return spans


def _luminance(color: str) -> float:
    """Relative luminance, for choosing readable text over a filled bar."""
    red, green, blue = to_rgba(color)[:3]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _order_by_reference(
    labels: np.ndarray, reference: np.ndarray, order: Sequence[Any]
):
    """Order a clustering's ids by where its mass sits in the reference column.

    A cluster is placed at the weighted mean position of the reference
    categories its samples carry, so a cluster made mostly of the reference
    column's third category is drawn third.
    """
    keys = sorted(np.unique(labels).tolist())
    counts = pd.crosstab(pd.Series(labels), pd.Series(reference)).reindex(
        index=keys, columns=list(order), fill_value=0
    )
    totals = np.maximum(counts.sum(axis=1).to_numpy(), 1)
    positions = (counts.to_numpy() * np.arange(len(order))).sum(axis=1) / totals
    return [key for _, key in sorted(zip(positions, keys))]


def _purity(labels: np.ndarray, reference: np.ndarray, order: Sequence[Any]):
    """Fraction of each cluster that falls in its single largest reference class."""
    counts = pd.crosstab(pd.Series(labels), pd.Series(reference)).reindex(
        index=list(order), fill_value=0
    )
    totals = np.maximum(counts.sum(axis=1).to_numpy(), 1)
    return counts.max(axis=1).to_numpy() / totals


def _cluster_name(key: Any) -> str:
    """Name a cluster ``C1``, ``C2``, ... from its zero-based id."""
    try:
        return f"C{int(key) + 1}"
    except (TypeError, ValueError):
        return str(key)


def _flow_ribbon(
    ax: Axes,
    x0: float,
    y0: tuple[float, float],
    x1: float,
    y1: tuple[float, float],
    color: str,
    alpha: float,
) -> None:
    """Add one Bezier ribbon spanning two stacked columns.

    Both edges are cubic Beziers with their control points on the vertical
    midline between the columns, which gives the band a flat departure and
    arrival and an S-curve in between.
    """
    midpoint = (x0 + x1) / 2
    top0, bottom0 = y0
    top1, bottom1 = y1

    vertices = [
        (x0, top0),
        (midpoint, top0),
        (midpoint, top1),
        (x1, top1),
        (x1, bottom1),
        (midpoint, bottom1),
        (midpoint, bottom0),
        (x0, bottom0),
        (x0, top0),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            Path(vertices, codes),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            linewidth=0,
            zorder=1,
        )
    )


def _column_flows(
    ax: Axes,
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_order: Sequence[Any],
    target_order: Sequence[Any],
    source_spans: Sequence[tuple[float, float]],
    target_spans: Sequence[tuple[float, float]],
    x_source: float,
    x_target: float,
    cmap: Mapping[Any, str],
    color_by: str,
    alpha: float,
) -> None:
    """Draw every non-empty ribbon between two adjacent columns.

    Each side's bands are packed top-down in the other column's order, so a
    node's outgoing bands leave in the same vertical order they arrive in
    and the ribbons nest instead of braiding.
    """
    counts = pd.crosstab(pd.Series(source), pd.Series(target)).reindex(
        index=list(source_order), columns=list(target_order), fill_value=0
    )
    source_totals = {key: max(int(counts.loc[key].sum()), 1) for key in source_order}
    target_totals = {key: max(int(counts[key].sum()), 1) for key in target_order}
    source_used = dict.fromkeys(source_order, 0)
    target_used = dict.fromkeys(target_order, 0)

    for i, source_key in enumerate(source_order):
        s_bottom, s_top = source_spans[i]
        s_height = s_top - s_bottom
        for j, target_key in enumerate(target_order):
            overlap = int(counts.loc[source_key, target_key])
            if overlap == 0:
                continue
            t_bottom, t_top = target_spans[j]
            t_height = t_top - t_bottom

            s_upper = (
                s_top - s_height * source_used[source_key] / source_totals[source_key]
            )
            s_lower = s_upper - s_height * overlap / source_totals[source_key]
            source_used[source_key] += overlap

            t_upper = (
                t_top - t_height * target_used[target_key] / target_totals[target_key]
            )
            t_lower = t_upper - t_height * overlap / target_totals[target_key]
            target_used[target_key] += overlap

            key = source_key if color_by == "source" else target_key
            _flow_ribbon(
                ax,
                x_source,
                (s_upper, s_lower),
                x_target,
                (t_upper, t_lower),
                cmap.get(key, FALLBACK_COLOR),
                alpha,
            )


def alluvial(
    ax: Axes,
    y_true: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    *,
    left_cmap: Mapping[Any, str],
    right_cmap: Mapping[Any, str],
    true_cmap: Mapping[Any, str],
    left_title: str,
    right_title: str,
    true_title: str,
    link_alpha: float = 0.4,
    bar_width: float = 0.045,
    gap_frac: float = 0.015,
    true_gap_frac: float = 0.045,
) -> Axes:
    """Draw a three-column alluvial: left clustering, truth, right clustering.

    The reported labels anchor the diagram. They stack in first-occurrence
    order, and both clusterings are ordered against that column by
    _order_by_reference.

    Every ribbon takes its reported label's color on both sides; the cluster
    bars keep their own column's color. The middle column has wider gaps
    (``true_gap_frac``) than the two cluster columns.

    Cluster bars are named ``C1``, ``C2``, ... with their purity (the
    fraction of the cluster falling in its single largest reported class)
    beside them; the reported-label bars carry their own name inside.
    """
    y_true = np.asarray(y_true)
    left_labels = np.asarray(left_labels)
    right_labels = np.asarray(right_labels)

    true_order = list(dict.fromkeys(y_true.tolist()))
    left_order = _order_by_reference(left_labels, y_true, true_order)
    right_order = _order_by_reference(right_labels, y_true, true_order)

    left_spans = _stack_segments(
        [int((left_labels == key).sum()) for key in left_order], gap_frac
    )
    true_spans = _stack_segments(
        [int((y_true == label).sum()) for label in true_order], true_gap_frac
    )
    right_spans = _stack_segments(
        [int((right_labels == key).sum()) for key in right_order], gap_frac
    )

    x_left, x_true, x_right = 0.0, 0.5, 1.0
    half = bar_width / 2

    def _draw_bars(order, spans, x, cmap):
        for key, (bottom, top) in zip(order, spans):
            ax.add_patch(
                plt_rectangle(
                    x - half,
                    bottom,
                    bar_width,
                    top - bottom,
                    cmap.get(key, FALLBACK_COLOR),
                )
            )

    _draw_bars(left_order, left_spans, x_left, left_cmap)
    _draw_bars(true_order, true_spans, x_true, true_cmap)
    _draw_bars(right_order, right_spans, x_right, right_cmap)

    for order, spans, purities, x, side in (
        (
            left_order,
            left_spans,
            _purity(left_labels, y_true, left_order),
            x_left,
            "right",
        ),
        (
            right_order,
            right_spans,
            _purity(right_labels, y_true, right_order),
            x_right,
            "left",
        ),
    ):
        offset = -half - 0.012 if side == "right" else half + 0.012
        for key, (bottom, top), purity in zip(order, spans, purities):
            ax.text(
                x + offset,
                (bottom + top) / 2,
                f"{_cluster_name(key)}  {purity * 100:.0f}%",
                ha=side,
                va="center",
                fontsize=FONT_SIZES["legend"],
            )

    for label, (bottom, top) in zip(true_order, true_spans):
        color = true_cmap.get(label, FALLBACK_COLOR)
        ax.text(
            x_true,
            (bottom + top) / 2,
            str(label),
            ha="center",
            va="center",
            fontsize=FONT_SIZES["legend"],
            color="white" if _luminance(color) < 0.45 else FOREGROUND_COLOR,
            zorder=3,
        )

    _column_flows(
        ax,
        left_labels,
        y_true,
        source_order=left_order,
        target_order=true_order,
        source_spans=left_spans,
        target_spans=true_spans,
        x_source=x_left + half,
        x_target=x_true - half,
        cmap=true_cmap,
        color_by="target",
        alpha=link_alpha,
    )
    _column_flows(
        ax,
        y_true,
        right_labels,
        source_order=true_order,
        target_order=right_order,
        source_spans=true_spans,
        target_spans=right_spans,
        x_source=x_true + half,
        x_target=x_right - half,
        cmap=true_cmap,
        color_by="source",
        alpha=link_alpha,
    )

    for x, title in (
        (x_left, left_title),
        (x_true, true_title),
        (x_right, right_title),
    ):
        ax.text(
            x,
            1.06,
            title,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["title"],
        )

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.02, 1.12)
    ax.set_axis_off()
    return ax


def plt_rectangle(x: float, y: float, width: float, height: float, color: str):
    """Build a filled bar patch. Split out so alluvial stays readable."""
    return Rectangle(
        (x, y),
        width,
        height,
        facecolor=color,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )


def ari_lollipop(
    ax: Axes,
    ari_df: pd.DataFrame,
    *,
    title: str | None = None,
    annotate_k: bool = True,
) -> Axes:
    """Horizontal lollipop of ARI against reported labels, one row per method."""
    ordered = ari_df.sort_values("ari", ascending=True).reset_index(drop=True)
    positions = np.arange(len(ordered))
    color_key = ordered["metric"] if "metric" in ordered.columns else ordered["method"]
    colors = [metric_color(str(m)) for m in color_key]

    ax.hlines(positions, 0, ordered["ari"].to_numpy(), color=colors, linewidth=2.0)
    markers = ax.scatter(
        ordered["ari"].to_numpy(), positions, color=colors, s=60, zorder=3
    )

    if annotate_k and "k" in ordered.columns:
        # Offset in points from the marker's center: its radius, half its
        # edge line and a 3 pt gap, so the label keeps its distance at any
        # x range.
        edge_pt = float(markers.get_linewidth()[0])
        offset_pt = np.sqrt(markers.get_sizes()[0]) / 2 + edge_pt / 2 + 3.0
        for position, (ari, k) in enumerate(zip(ordered["ari"], ordered["k"])):
            ax.annotate(
                f"k={int(k)}",
                xy=(ari, position),
                xytext=(offset_pt, 0),
                textcoords="offset points",
                va="center",
                fontsize=FONT_SIZES["legend"],
            )

    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["method"], fontsize=FONT_SIZES["tick"])
    ax.set_xlabel("ARI vs. reported labels", fontsize=FONT_SIZES["axis_label"])
    ax.set_xlim(0, max(1.0, float(ordered["ari"].max()) * 1.15))
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    return style_axes(ax)


def grouped_legend(
    fig: Figure,
    axes: np.ndarray,
    *,
    fontsize: float | None = None,
    y_offset: float = 0.10,
    ncol: int | None = None,
) -> Legend:
    """One deduplicated legend below a grid of axes."""
    handles: list[Line2D] = []
    labels: list[str] = []
    for ax in np.asarray(axes).flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    return fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -y_offset),
        ncol=ncol or min(len(labels), 4),
        frameon=False,
        fontsize=fontsize or FONT_SIZES["legend"],
    )


def _legend_groups(metrics: Sequence[str]) -> list[list[list[str]]]:
    """Split metrics into families, and each family into its own columns.

    Returns one entry per family, each holding that family's columns. Left to
    right: the oracle baseline alone, then CARVE's own measures in one
    column, then the classical indices across two. Families are decided from
    the metric names.

    Order within a family is the caller's order. The classical family takes
    a second column only when it holds more than two indices.
    """
    baseline = [m for m in metrics if m == BASELINE_METRIC]
    carve = [m for m in metrics if m in CARVE_METRICS_ALL]
    classical = [m for m in metrics if m in CVI_METRICS]
    leftover = [
        m
        for m in metrics
        if m not in baseline and m not in carve and m not in classical
    ]

    groups: list[list[list[str]]] = []
    if baseline:
        groups.append([baseline])
    if carve:
        groups.append([carve])
    if classical:
        per_column = len(classical) if len(classical) <= 2 else -(-len(classical) // 2)
        groups.append(
            [
                classical[start : start + per_column]
                for start in range(0, len(classical), per_column)
            ]
        )
    if leftover:
        groups.append([leftover])
    return groups


def metric_legend(
    fig: Figure,
    axes: np.ndarray,
    metrics: Sequence[str],
    *,
    fontsize: float | None = None,
    y_offset: float = 0.06,
    columnspacing: float = 0.6,
    outside: bool = False,
) -> Legend:
    """One legend below a grid, with the metric families in their own columns.

    outside places it with loc="outside lower center", for a figure with
    constrained layout, which then reserves the room for it; y_offset is
    ignored. Otherwise it hangs y_offset below the figure's lower edge.

    Matplotlib fills a multi-column legend top to bottom, so a column shorter
    than the tallest one is padded with blank entries; the next family then
    starts at the top of its own column.

    A Legend has one gutter width, so an empty column separates the families
    and the gutter inside the classical pair stays narrow.
    """
    available: dict[str, Line2D] = {}
    for ax in np.asarray(axes).flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            available.setdefault(label, handle)

    groups = [
        [
            drawn
            for drawn in (
                [m for m in column if _display(m) in available] for column in group
            )
            if drawn
        ]
        for group in _legend_groups(metrics)
    ]
    groups = [group for group in groups if group]
    if not groups:
        raise ValueError("None of the requested metrics were drawn on these axes.")

    rows = max(len(column) for group in groups for column in group)

    handles: list[Line2D] = []
    labels: list[str] = []

    def pad(count: int) -> None:
        for _ in range(count):
            handles.append(Line2D([], [], linestyle="none", marker="none"))
            labels.append("")

    n_columns = 0
    for index, group in enumerate(groups):
        if index:
            pad(rows)
            n_columns += 1
        for column in group:
            for metric in column:
                handles.append(available[_display(metric)])
                labels.append(_display(metric))
            pad(rows - len(column))
            n_columns += 1

    placement = (
        {"loc": "outside lower center"}
        if outside
        else {"loc": "lower center", "bbox_to_anchor": (0.5, -y_offset)}
    )
    return fig.legend(
        handles,
        labels,
        **placement,
        ncol=n_columns,
        frameon=False,
        fontsize=fontsize or FONT_SIZES["legend"],
        columnspacing=columnspacing,
    )


# Fraction of the axes width left of the axes where a panel letter ends.
PANEL_LETTER_X = -0.08


def panel_letter(
    ax: Axes,
    letter: str,
    *,
    x: float = PANEL_LETTER_X,
    y: float = 1.1,
    fontsize: float = FONT_SIZES["panel_letter"],
) -> None:
    """Place a bold panel letter in axes coordinates.

    A figure laid out at the print width passes PRINT_FONT_SIZES's letter
    size; the default suits a figure drawn larger and scaled down.
    """
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="right",
    )
