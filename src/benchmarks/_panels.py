"""Drawing primitives. Every one takes an ax and returns it.

Nothing here creates a figure, calls plt.show, or saves anything. Composition
is the job of the figures package; these only draw.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

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
