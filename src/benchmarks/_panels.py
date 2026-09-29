"""Drawing primitives. Every one takes an ax and returns it.

Nothing here creates a figure, calls plt.show, or saves anything. Composition
is the job of the figures package; these only draw.
"""

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


def aligned_color_maps(
    y_true: np.ndarray, *clusterings: np.ndarray
) -> tuple[dict[Any, str], ...]:
    """One palette shared by the reported labels and every clustering.

    Returns ``(true_cmap, *clustering_cmaps)``. The reported labels take the
    palette in sorted category order; each clustering's integer id indexes
    the same palette directly, so a cluster that ``align_cluster_labels``
    matched to reported label *i* is drawn in reported label *i*'s color.

    This is what makes the composite's scatter panels comparable. Calling
    cluster_color_map once per panel -- which is what this replaces -- keys
    each panel's map on that panel's own first-occurrence order, so the same
    cluster came out a different color in each of the three panels and the
    reader could not read one panel against the next.

    The palette is sized to cover both the reported labels and the highest
    cluster id present, so a clustering with more clusters than there are
    reported labels still gets a color for every one of them.
    """
    categories = sorted(set(np.asarray(y_true).tolist()))
    highest_id = 0
    for labels in clusterings:
        values = np.asarray(labels)
        if values.size:
            highest_id = max(highest_id, int(np.max(values)) + 1)

    palette = cluster_colors(max(len(categories), highest_id))
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
    overlapping points where two clusters meet -- the case-study scatters
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
    thing inside the data area, which is why the published composites carry
    it on their first panel only and leave the rest unlabeled.

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

    Consumes the unified artifact schema directly, so there is no axis-column
    sniffing. The function this replaces inferred its x column by inspecting
    which of two possible schemas it had been handed.

    ``"baseline_oracle"`` is accepted in ``metrics`` alongside the ordinary
    metric names. It is not a row in ``metric_name`` -- the oracle ARI is a
    per-cell constant carried on every row's ``oracle_ari`` column, one value
    per (axis point, seed) rather than one per (metric, k) -- so it is drawn
    from the deduplicated ``oracle_ari`` column instead of the
    ``is_selected``-filtered rows the other metrics use.

    show_points draws each dataset's own value behind the mean, dodged along
    x by dodge so seven methods' clouds do not merge into one band. It is
    off by default: Fig 4 draws means and error bars, and this function is
    what draws Fig 4.
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
    no rows in this cell is an all-NaN row rather than a dropped one: a
    dropped row would slide every metric below it up by one and silently
    mislabel the axis.
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

    The oracle is not drawn. It has no selected k -- it is fit once per cell
    at k_star -- so a row for it would be either empty or a tautological
    column of ones.
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
                # end. A single ink color is unreadable at one end or the
                # other, and the threshold sits where the ramp crosses.
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

    Normalization is what makes seven criteria comparable on one axis.
    Silhouette lives in [-1, 1], inverted Davies-Bouldin in (0, 1],
    Calinski-Harabasz runs to the thousands and the gap statistic is a log
    ratio; drawn raw, six of the seven collapse into a line at the bottom of
    whatever axis the largest one needs. The panel's question is which k each
    criterion prefers, and that is a property of the curve's shape, which
    normalization preserves exactly.

    Within a dataset, then averaged -- not the other way round. Averaging
    raw values first lets one dataset whose criterion happens to be an order
    of magnitude larger dominate the mean and drag the averaged curve onto
    its own shape.

    A criterion that is flat across k normalizes to 0.5 rather than dividing
    by zero. Flat means the criterion expressed no preference, and 0.5 draws
    that as the flat line it is.
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
    """Why each method chose the k it chose, for one point on the axis.

    Every criterion is normalized to [0, 1] over the candidate k, so the
    shapes are comparable; the base estimator's own ARI at each k is drawn
    on the same axis as a reference, which is what turns "this criterion
    peaked at the wrong k" into "and here is what that cost". It is drawn
    the way metric_lines draws the oracle baseline -- the same color, dash,
    width and standard-error bars -- because at k = k_star it is that
    baseline: both come from the same estimator fit at the same k with the
    same seed.

    The reference is the mean, over seeds, of the classical indices'
    ari_at_k rows (CVI_METRICS), which carry the base estimator's full-data
    fit at each k (see _run.run_cell). Reading it off those rows alone keeps
    any other rows the frame happens to carry out of the average. Nothing is
    drawn when the cell carries no classical-index rows.

    Both series are proportions in [0, 1], so no second y axis is needed --
    and a dual-axis panel would invite reading a crossing that is an
    artifact of two independent scales.

    Each metric's selected k is marked on its own curve. The gap statistic's
    mark will not sit on its maximum: Tibshirani's rule takes the smallest k
    with Gap(k) >= Gap(k+1) - s(k+1), not an argmax. That is the panel
    showing the rule at work.
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

    ``estimator_results_`` has one row per (configuration, k): a case study
    that sweeps more than one estimator (Klein sweeps Ward agglomerative
    clustering and spectral clustering, for instance) carries every swept
    estimator's rows in that one table. Plotting every row for a measure
    would draw one polyline per estimator end to end, jumping back to the
    first estimator's k range after the last -- so each line here is
    restricted to the single configuration CARVE's own 1-SE selection
    returns for that measure, identified by its ``method_id`` (the join key
    ``estimator_results_`` uses for "one line" -- see
    ``carve._sweep.MethodIds``), and the winning estimator's identity is
    named in the legend rather than left implicit.

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
    # Both measures this draws are adjusted Rand indices against the
    # resampled reference, so "ARI" names the quantity; "Validation score",
    # which this replaces, named the role instead and matched neither the
    # published axis nor the panel title.
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

    ``curves_df`` has one row per (metric, model, k): a case study that
    sweeps more than one estimator carries every swept model's scores for
    every index. Grouping only by metric would draw one line per index that
    zig-zags between models at every k, so each line here is restricted to
    ``best_df``'s winning model for that metric, and the model is named in
    the legend. The selected-k marker is read directly from that row's own
    ``k`` rather than re-located by matching scores, so it always marks the
    winning model's selection and not an arbitrary other model's row that
    happens to share a k.

    Indices live on incompatible scales, so they are min-max normalized to a
    common axis by default; the selected k is unaffected by that rescaling.
    """
    for _, best in best_df.iterrows():
        metric = best["metric"]
        model = best["model"]
        sub = curves_df.loc[
            (curves_df["metric"] == metric) & (curves_df["model"] == model)
        ].sort_values("k")
        if sub.empty:
            continue

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
            label=f"{_display(str(metric))} — {model}",
        )

        # A vertical rule at the selected k, matching how carve_lines marks
        # its own selections in the panel beside this one. The ring this
        # replaces sat on the curve, so it read as a data point rather than
        # as a selection and was lost wherever two indices crossed.
        ax.axvline(
            int(best["k"]), color=color, linestyle="--", linewidth=1.0, alpha=0.6
        )

    ax.set_xlabel("Number of clusters $k$", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(
        "Score (normalized)" if normalize else "Score",
        fontsize=FONT_SIZES["axis_label"],
    )
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)


def _monte_carlo_p_floor(values: np.ndarray) -> int | None:
    """The iteration count implied by a MONTECARLO_P column, if self-consistent.

    M3C computes MONTECARLO_P as (count + 1) / (iters + 1) for a
    non-negative integer count, so every value the same run produces is an
    integer multiple of 1 / (iters + 1). Hypothesizing iters + 1 from the
    column's smallest value only recovers the true denominator when some
    K's count is 0 -- the theoretical floor -- so the hypothesis is
    confirmed against every other value in the column before it is trusted.
    That confirmation is what rules out a coincidental reduced fraction
    (Klein's K=7 carries 2/26 == 1/13) being mistaken for the floor: it
    would require every other row to also be an exact multiple of 1/13,
    which a genuine run at iters=25 does not produce.

    Returns None when the column does not round-trip this way, including
    when it is empty or degenerate.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    denom = round(1.0 / float(values.min()))
    if denom < 2:
        return None
    multiples = values * denom
    if np.any(np.abs(multiples - np.round(multiples)) > 1e-6):
        return None
    return denom - 1


def _monte_carlo_p_label(scores: pd.DataFrame, selected_k: int) -> str:
    """The Monte Carlo p-value annotation text for m3c_lines' Panel A.

    Read from the scores frame at selected_k rather than a separate
    parameter, the way RCSI and RCSI_SE already are. M3C's p is
    (count + 1) / (iters + 1), so at the vignette's default iters=25 the
    smallest attainable value is 1/26 = 0.038462 -- identical for every K
    whose real stability beat all 25 Monte Carlo references. A bare
    "p = 0.038" there would read as a precise estimate rather than a floor,
    so _monte_carlo_p_floor recovers the iteration count from the column
    itself and the note is qualified only when selected_k's own value sits
    exactly at that floor.
    """
    p_value = float(scores.loc[scores["K"] == selected_k, "MONTECARLO_P"].iloc[0])
    iters = _monte_carlo_p_floor(scores["MONTECARLO_P"].to_numpy(dtype=float))
    if iters is not None and np.isclose(p_value, 1.0 / (iters + 1), rtol=1e-6):
        return f"Monte Carlo $p$ = {p_value:.3f} (smallest attainable at {iters} iterations)"
    return f"Monte Carlo $p$ = {p_value:.3f}"


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
    meaningful zero -- RCSI at or below zero means the real data is no more
    stable at that K than M3C's Monte Carlo reference.

    The error bars are plus or minus 1.96 RCSI_SE, which is M3C's own plot
    idiom, and the dashed vertical marks the selected K the way cvi_lines and
    carve_lines mark theirs. The Monte Carlo p-value at selected_k is
    annotated in the corner, per the spec's Reporting section; see
    _monte_carlo_p_label for how a floor value is worded honestly.
    """
    ordered = scores.sort_values("K")
    color = metric_color("m3c_rcsi")

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
    ax.axhline(0.0, color=FOREGROUND_COLOR, linestyle=":", linewidth=1.0, alpha=0.5)
    ax.axvline(int(selected_k), color=color, linestyle="--", linewidth=1.0, alpha=0.6)

    ax.annotate(
        _monte_carlo_p_label(scores, selected_k),
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
    column's third category is drawn third. Stacking each column by its own
    id order instead -- which is what this replaces -- left the ribbons to
    cross the full height of the panel to reach their anchor.
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
    midline between the columns, which is what gives the band its flat
    departure and arrival and an S-curve in between. Straight-line bands --
    what this replaces -- read as a bar chart of connections rather than as
    flow, and cross each other at hard angles.
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
    _order_by_reference, so the panel reads as "how does each clustering
    partition the reported labels" rather than as two unrelated stacks.

    Every ribbon takes its reported label's color on both sides, which is
    what lets one reported class be followed across the whole panel; the
    cluster bars keep their own column's color. The middle column carries
    more generous gaps (``true_gap_frac``) than the two cluster columns, so
    the anchor reads as a separate register rather than as a third stack of
    the same kind.

    Cluster bars are named ``C1``, ``C2``, ... with their purity -- the
    fraction of the cluster falling in its single largest reported class --
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
    # ordered["method"] must not be evaluated unless "metric" is absent --
    # ordered.get("metric", ordered["method"]) evaluates the fallback
    # argument eagerly, which raises KeyError on a frame that carries
    # "metric" but not "method", defeating the point of .get().
    color_key = ordered["metric"] if "metric" in ordered.columns else ordered["method"]
    colors = [metric_color(str(m)) for m in color_key]

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


def _legend_groups(metrics: Sequence[str]) -> list[list[list[str]]]:
    """Split metrics into families, and each family into its own columns.

    Returns one entry per family, each holding that family's columns. Left to
    right: the oracle baseline alone, then CARVE's own measures in one
    column, then the classical indices across two. Grouping is decided from
    the metric names, not from the label strings a previous implementation of
    this pattern matched on ("carve" in the label), which silently
    reclassified any metric whose display name did not happen to contain the
    word.

    Order within a family is the caller's order, so the two classical columns
    are set by how ``metrics`` is written rather than by a rule here that a
    caller cannot see.

    The classical family takes a second column only when it holds more than
    two indices. Splitting two of them in half would produce two columns one
    entry tall standing beside a two-entry CARVE column, which reads as a
    layout accident rather than as a grouping.
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
    than the tallest one is padded with a blank entry rather than letting the
    next family start halfway up a column. That padding is what lets the
    single-entry baseline column sit beside two-entry columns, which is the
    arrangement the published figure uses; it is one Legend, not three
    overlaid ones positioned by hand.

    A Legend has a single gutter width, so separating the families by
    widening ``columnspacing`` would widen the gap inside the classical pair
    too. Instead the gutter is narrow and an empty column stands between
    families: the classical pair sits at the narrow gutter, and each family
    boundary gets that gutter twice over plus the empty column's width.
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
