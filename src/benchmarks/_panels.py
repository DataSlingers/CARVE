"""Drawing primitives. Every one takes an ax and returns it.

Nothing here creates a figure, calls plt.show, or saves anything. Composition
is the job of the figures package; these only draw.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from ._registry import METRIC_DISPLAY_NAMES
from ._theme import FONT_SIZES, cluster_colors, metric_color, style_axes


def _display(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def cluster_color_map(labels: np.ndarray) -> dict[Any, str]:
    """Map each distinct label to a stable color."""
    unique = list(dict.fromkeys(np.asarray(labels).tolist()))
    return dict(zip(unique, cluster_colors(len(unique))))


def scatter_clusters(
    ax: Axes,
    Z: np.ndarray,
    labels: np.ndarray,
    *,
    color_map: Mapping[Any, str] | None = None,
    s: float = 20.0,
    alpha: float = 0.85,
    linewidth: float = 0.3,
    title: str | None = None,
    hide_axes: bool = True,
    axis_labels: Sequence[str] | None = None,
) -> Axes:
    """Scatter a two-dimensional embedding, colored by label."""
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
            edgecolor="none",
            color=color_map.get(label, "#7F7F7F"),
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
) -> Axes:
    """Plot mean ARI at the selected k against the axis, one line per metric.

    Consumes the unified artifact schema directly, so there is no axis-column
    sniffing. The function this replaces inferred its x column by inspecting
    which of two possible schemas it had been handed.
    """
    selected = df.loc[df["is_selected"]]

    for metric in metrics:
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
            linewidth=1.8 * element_scale,
            capsize=2.5 * element_scale,
            color=metric_color(metric),
            label=_display(metric),
        )

    ax.set_xlabel(x_label or x_col, fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(
        y_label or r"ARI (selected $\hat{k}$ vs. true labels)",
        fontsize=FONT_SIZES["axis_label"],
    )
    if show_legend:
        ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
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


def carve_lines(
    ax: Axes,
    carve_obj: Any,
    *,
    measures: Sequence[str] = ("stability", "generalizability"),
    not_two: bool = False,
    title: str | None = None,
    annotate: bool = False,
    show_selected_k: bool = True,
) -> Axes:
    """Plot CARVE validation curves over k, one line per measure."""
    results = carve_obj.estimator_results_
    ks = results["n_clusters"].to_numpy()

    for measure in measures:
        color = metric_color(f"ari_{measure}_1se")
        values = results[f"ari_{measure}"].to_numpy()
        ax.plot(
            ks,
            values,
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=color,
            label=_display(f"ari_{measure}_1se"),
        )
        if f"ari_{measure}_se" in results.columns:
            se = results[f"ari_{measure}_se"].to_numpy()
            ax.fill_between(ks, values - se, values + se, color=color, alpha=0.15)

        if show_selected_k:
            selected = int(
                carve_obj.get_k(measure=measure, rule="1se", not_two=not_two)
            )
            ax.axvline(selected, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
            if annotate:
                ax.annotate(
                    f"$\\hat{{k}}={selected}$",
                    xy=(selected, float(np.nanmax(values))),
                    fontsize=FONT_SIZES["legend"],
                    color=color,
                )

    ax.set_xlabel("Number of clusters $k$", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Validation score", fontsize=FONT_SIZES["axis_label"])
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
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

    Indices live on incompatible scales, so they are min-max normalized to a
    common axis by default; the selected k is unaffected by that rescaling.
    """
    for metric, sub in curves_df.groupby("metric", sort=False):
        sub = sub.sort_values("k")
        values = sub["score"].to_numpy(dtype=float)
        if normalize:
            span = np.nanmax(values) - np.nanmin(values)
            values = (values - np.nanmin(values)) / span if span > 0 else values * 0.0

        color = metric_color(str(metric))
        ax.plot(
            sub["k"].to_numpy(),
            values,
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            color=color,
            label=_display(str(metric)),
        )

        best = best_df.loc[best_df["metric"] == metric]
        if not best.empty:
            best_k = int(best["k"].iloc[0])
            match = np.where(sub["k"].to_numpy() == best_k)[0]
            if match.size:
                ax.scatter(
                    [best_k],
                    [values[match[0]]],
                    s=90,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.8,
                    zorder=5,
                )

    ax.set_xlabel("Number of clusters $k$", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(
        "Normalized index" if normalize else "Index value",
        fontsize=FONT_SIZES["axis_label"],
    )
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)


def _stack_segments(
    sizes: Sequence[int], gap_frac: float = 0.015
) -> list[tuple[float, float]]:
    """Return (bottom, top) spans for a stacked bar with proportional gaps."""
    total = float(sum(sizes))
    if total <= 0:
        return []
    gap = gap_frac
    usable = 1.0 - gap * max(len(sizes) - 1, 0)
    spans = []
    cursor = 0.0
    for size in sizes:
        height = usable * (size / total)
        spans.append((cursor, cursor + height))
        cursor += height + gap
    return spans


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
    link_alpha: float = 0.35,
    bar_width: float = 0.08,
) -> Axes:
    """Draw a three-column alluvial: left clustering, truth, right clustering."""
    y_true = np.asarray(y_true)
    left_labels = np.asarray(left_labels)
    right_labels = np.asarray(right_labels)

    columns = [
        (0.0, left_labels, left_cmap, left_title),
        (0.5, y_true, true_cmap, true_title),
        (1.0, right_labels, right_cmap, right_title),
    ]

    spans_by_column = []
    for x, labels, cmap, title in columns:
        categories = list(dict.fromkeys(labels.tolist()))
        sizes = [int((labels == c).sum()) for c in categories]
        spans = _stack_segments(sizes)
        for category, (bottom, top) in zip(categories, spans):
            ax.add_patch(
                plt_rectangle(
                    x - bar_width / 2,
                    bottom,
                    bar_width,
                    top - bottom,
                    cmap.get(category, "#7F7F7F"),
                )
            )
        ax.text(
            x, 1.04, title, ha="center", va="bottom", fontsize=FONT_SIZES["axis_label"]
        )
        spans_by_column.append((x, categories, dict(zip(categories, spans))))

    for (x0, cats0, spans0), (x1, cats1, spans1), left_arr, right_arr, flow_cmap in (
        (*spans_by_column[0:2], left_labels, y_true, left_cmap),
        (*spans_by_column[1:3], y_true, right_labels, true_cmap),
    ):
        cursor0 = {c: spans0[c][0] for c in cats0}
        cursor1 = {c: spans1[c][0] for c in cats1}
        for c0 in cats0:
            total0 = max(int((left_arr == c0).sum()), 1)
            height0 = spans0[c0][1] - spans0[c0][0]
            for c1 in cats1:
                overlap = int(((left_arr == c0) & (right_arr == c1)).sum())
                if overlap == 0:
                    continue
                h0 = height0 * overlap / total0
                total1 = max(int((right_arr == c1).sum()), 1)
                h1 = (spans1[c1][1] - spans1[c1][0]) * overlap / total1
                ax.fill_between(
                    np.linspace(x0 + bar_width / 2, x1 - bar_width / 2, 32),
                    np.linspace(cursor0[c0], cursor1[c1], 32),
                    np.linspace(cursor0[c0] + h0, cursor1[c1] + h1, 32),
                    color=flow_cmap.get(c0, "#7F7F7F"),
                    alpha=link_alpha,
                    linewidth=0,
                )
                cursor0[c0] += h0
                cursor1[c1] += h1

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.02, 1.12)
    ax.set_axis_off()
    return ax


def plt_rectangle(x: float, y: float, width: float, height: float, color: str):
    """Build a filled rectangle patch. Split out so alluvial stays readable."""
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), width, height, facecolor=color, edgecolor="none")


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
    colors = [metric_color(str(m)) for m in ordered.get("metric", ordered["method"])]

    ax.hlines(positions, 0, ordered["ari"].to_numpy(), color=colors, linewidth=2.0)
    ax.scatter(ordered["ari"].to_numpy(), positions, color=colors, s=60, zorder=3)

    if annotate_k and "k" in ordered.columns:
        for position, (ari, k) in enumerate(zip(ordered["ari"], ordered["k"])):
            ax.text(
                ari + 0.01,
                position,
                f"k={int(k)}",
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
    """One deduplicated legend below a grid of axes.

    The old code extracted this helper out of a 408-line figure function and
    then never removed the original, so the same legend was built twice.
    """
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


def panel_letter(ax: Axes, letter: str, *, x: float = -0.08, y: float = 1.1) -> None:
    """Place a bold panel letter in axes coordinates."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=FONT_SIZES["panel_letter"],
        fontweight="bold",
        va="top",
        ha="right",
    )
