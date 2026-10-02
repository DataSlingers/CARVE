"""The two SI figures of the rho/B ablation.

figure_ablation answers the sensitivity question (R4.4): the ARI to the
true labels of the labels each headline selector returns, against the
subsampling proportion (the rho arm, at the default B) and against the
resample count (the B arm, at the default rho). A dashed line marks the
oracle at k*. An ablation with a case study adds a row with the study's
selected labels scored against its reference labels.

figure_ablation_resampling answers how resampling affects the clustering
(R1.3), against rho: the bias of the selected k, recall of the smallest
true cluster, and the agreement of a subsample's clustering with the
full-data clustering. All three need a true k, so it draws the simulations
only.

Both are laid out at the PLOS page width under print_theme_context, so
their text prints at the sizes set in _theme. Color carries the selector,
stability 1SE and generalizability 1SE in their metric colors; panels draw
the pooled simulations, which with one scenario are that scenario. Shaded
bands are the mean plus or minus Z_95 standard errors over datasets, and a
dotted line marks the package default. Every number comes from
_ablation_summary.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

from .._ablation_cells import REFERENCE_RATIO, arm_view
from .._ablation_summary import (
    DATASET_KEY,
    HEADLINE_METRICS,
    POOLED,
    ari_summary,
    bias_summary,
    rare_recall_summary,
    similarity_at_k_star,
    study_ari_summary,
)
from .._registry import METRIC_DISPLAY_NAMES
from .._theme import (
    FALLBACK_COLOR,
    FOREGROUND_COLOR,
    PRINT_FONT_SIZES,
    PRINT_REFERENCE_LINEWIDTH,
    PRINT_WIDTH_IN,
    metric_color,
    print_theme_context,
    save_figure,
    style_axes,
)
from ._benchmarking_examples import SCENARIO_TITLES
from ._paths import BENCHMARKING_DIR, figure_path

STUDY_TITLES: dict[str, str] = {**SCENARIO_TITLES, "klein": "Klein"}
RHO, B = "subsample_ratio", "n_resamples"
RHO_LABEL = r"Subsampling proportion $\rho$"
B_LABEL = r"Resamples $B$"

# A 95% interval from a mean and its standard error, by the normal
# approximation.
Z_95 = 1.96
# ARI on its full scale in every panel, so a difference of a few hundredths
# reads as what it is. Autoscaling zoomed the panels to a tenth of the scale
# and made flat lines look like trends.
ARI_LIMITS: tuple[float, float] = (0.0, 1.0)
ARI_TICKS = np.linspace(0.0, 1.0, 6)
# Rare-cluster recall is a proportion, padded so markers at 0 and 1 are not
# clipped, and extended above 1 to hold the panel's line-style key off the
# data; the ticks stay 0 to 1.
PROPORTION_LIMITS: tuple[float, float] = (-0.04, 1.04)
KEY_HEADROOM_TOP = 1.32
ROW_HEIGHT_IN = 2.6
RESAMPLING_HEIGHT_IN = 2.2
LEGEND_HEIGHT_IN = 0.35
BAND_ALPHA = 0.18
RHO_TICKS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
# The panel letter sits outside the axes, left of the spine by this many
# points, on the title's baseline; the title starts at the spine, so the two
# cannot overlap however long the title is.
LETTER_OFFSET_PT = -30.0


def _label(ax: Axes, letter: str, title: str) -> None:
    """The panel letter, in bold outside the axes' top-left corner, and the
    title, left-aligned at the spine. The axes' own label is set to the
    letter, which is how the figure (and its tests) find a panel."""
    ax.set_label(letter)
    ax.set_title(title, loc="left")
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(LETTER_OFFSET_PT, plt.rcParams["axes.titlepad"]),
        textcoords="offset points",
        ha="left",
        va="baseline",
        fontweight="bold",
        fontsize=PRINT_FONT_SIZES["panel_letter"],
    )


def _reference(ax: Axes, x_default: float) -> None:
    ax.axvline(
        x_default,
        color=FALLBACK_COLOR,
        linestyle=":",
        linewidth=PRINT_REFERENCE_LINEWIDTH,
        zorder=1,
    )


def _rho_axis(ax: Axes) -> None:
    """rho ticked every 0.2 up to what the panel draws (the similarity
    panel reaches the refit at 1), written as plain numbers."""
    top = ax.get_xlim()[1]
    ticks = [t for t in RHO_TICKS if t <= top]
    ax.set_xticks(ticks, [f"{t:.1f}" for t in ticks])


def _key(ax: Axes, **kwargs) -> None:
    """An in-panel legend, left out of the layout so it cannot shrink the
    other panels."""
    ax.legend(**kwargs).set_in_layout(False)


def _band(ax: Axes, x, mean, sem, color: str) -> None:
    ax.fill_between(
        x,
        mean - Z_95 * sem,
        mean + Z_95 * sem,
        color=color,
        alpha=BAND_ALPHA,
        linewidth=0,
    )


def _line(ax: Axes, x, y, color: str, *, linestyle: str = "-") -> None:
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker="o",
        markeredgecolor="white",
        markeredgewidth=0.5,
    )


def _b_axis(ax: Axes, grid: Sequence[int]) -> None:
    """Log B axis ticked at the grid values, written as plain numbers."""
    ax.set_xscale("log")
    ax.set_xticks(list(grid), [f"{b:g}" for b in grid])
    ax.xaxis.set_minor_locator(NullLocator())


def _selectors(
    ax: Axes,
    summary: pd.DataFrame,
    *,
    x: str,
    y: str = "ari_mean",
    sem: str = "ari_sem",
    study: str | None = POOLED,
    linestyle: str = "-",
) -> None:
    """One line per headline selector in its metric color, with the mean
    plus or minus Z_95 standard errors as its band. study=None takes the
    frame as already restricted to one study."""
    for metric in HEADLINE_METRICS:
        rows = summary[summary["metric_name"] == metric]
        if study is not None:
            rows = rows[rows["study"] == study]
        rows = rows.sort_values(x)
        if rows.empty:
            continue
        color = metric_color(metric)
        _line(ax, rows[x], rows[y], color, linestyle=linestyle)
        _band(ax, rows[x], rows[y], rows[sem], color)


def _oracle(ax: Axes, value: float) -> None:
    """The oracle's ARI as a dashed line, labeled at its left end, which
    keeps the label off the default's reference line: both grids place the
    default right of center."""
    color = metric_color("baseline_oracle")
    ax.axhline(
        value,
        color=color,
        linestyle="--",
        linewidth=PRINT_REFERENCE_LINEWIDTH,
    )
    ax.annotate(
        "Oracle $k^*$",
        xy=(0.0, value),
        xycoords=("axes fraction", "data"),
        xytext=(2, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=color,
    )


def _simulations(ax: Axes, view: Mapping[str, pd.DataFrame], *, x: str) -> None:
    """The pooled selectors and, under them, the oracle averaged over the
    datasets this arm ran (datasets passes through arm_view whole)."""
    _selectors(ax, ari_summary(view["selection"], view["at_k"], x=x), x=x)
    ran = view["cells"][list(DATASET_KEY)].drop_duplicates()
    oracle = view["datasets"].merge(ran, on=list(DATASET_KEY))["oracle_ari"]
    _oracle(ax, float(oracle.mean()))
    ax.set_ylabel("ARI to true labels")


def _study(ax: Axes, view: Mapping[str, pd.DataFrame], *, x: str, study: str) -> None:
    """Mean and 95% interval over replicates of the selected labels' ARI
    against the study's reference labels."""
    stats = study_ari_summary(
        view["selection"], x=x, study=study, metrics=HEADLINE_METRICS
    )
    _selectors(ax, stats, x=x, study=None)
    ax.set_ylabel("ARI to reference labels")


def _legend(fig: Figure, default_label: str) -> None:
    """The figure's one shared legend, above the panels."""
    handles = [
        Line2D(
            [],
            [],
            color=metric_color(metric),
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=METRIC_DISPLAY_NAMES.get(metric, metric),
        )
        for metric in HEADLINE_METRICS
    ]
    handles.append(Patch(facecolor=FALLBACK_COLOR, alpha=0.35, label="95% CI"))
    handles.append(
        Line2D(
            [],
            [],
            color=FALLBACK_COLOR,
            linestyle=":",
            linewidth=PRINT_REFERENCE_LINEWIDTH,
            label=default_label,
        )
    )
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))


def figure_ablation(
    frames: Mapping[str, pd.DataFrame],
    *,
    ablation,
    scale: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """ARI of the selected labels against rho (left) and B (right)."""
    rho_view = arm_view(frames, ablation=ablation, scale=scale, arm="rho")
    b_view = arm_view(frames, ablation=ablation, scale=scale, arm="b")
    rho_default, b_default = ablation.rho_default, ablation.b_default
    study = ablation.study
    n_rows = 1 if study is None else 2

    with print_theme_context():
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(PRINT_WIDTH_IN, n_rows * ROW_HEIGHT_IN + LEGEND_HEIGHT_IN),
            layout="constrained",
            squeeze=False,
        )

        _simulations(axes[0, 0], rho_view, x=RHO)
        _label(axes[0, 0], "A", rf"Varying $\rho$, $B$ = {b_default:g}")
        _simulations(axes[0, 1], b_view, x=B)
        _label(axes[0, 1], "B", rf"Varying $B$, $\rho$ = {rho_default:g}")

        if study is not None:
            title = STUDY_TITLES.get(study, study)
            _study(axes[1, 0], rho_view, x=RHO, study=study)
            _label(axes[1, 0], "C", rf"{title}: varying $\rho$")
            _study(axes[1, 1], b_view, x=B, study=study)
            _label(axes[1, 1], "D", rf"{title}: varying $B$")

        for ax in axes[:, 0]:
            _reference(ax, rho_default)
            _rho_axis(ax)
            ax.set_xlabel(RHO_LABEL)
        for ax in axes[:, 1]:
            _reference(ax, b_default)
            _b_axis(ax, list(ablation.b_grid))
            ax.set_xlabel(B_LABEL)
        for ax in axes.flat:
            ax.set_ylim(ARI_LIMITS)
            ax.set_yticks(ARI_TICKS)
            style_axes(ax)
        _legend(fig, rf"Default ($\rho$ = {rho_default:g}, $B$ = {b_default:g})")
        if save:
            save_figure(
                fig,
                figure_path(
                    "si_fig_ablation.png", subdir=BENCHMARKING_DIR, out_dir=out_dir
                ),
            )
    return fig


def _bias(ax: Axes, view: Mapping[str, pd.DataFrame]) -> None:
    _selectors(
        ax, bias_summary(view["selection"], x=RHO), x=RHO, y="bias_mean", sem="bias_sem"
    )
    ax.axhline(0.0, color=FALLBACK_COLOR, linewidth=PRINT_REFERENCE_LINEWIDTH)
    ax.set_ylabel(r"Mean $\hat{k} - k^*$")


def _rare_recall(
    ax: Axes, view: Mapping[str, pd.DataFrame], *, difficulty: str
) -> None:
    """Recall of the smallest true cluster in the labels at the selected k
    (solid, one line per rule) and at k* (dashed). Every rule returns the
    stability-consensus labels, so recall at k* is the same for both and is
    drawn once, in the neutral color."""
    summary = rare_recall_summary(
        view["at_k"], view["selection"], x=RHO, difficulty=difficulty
    )
    _selectors(ax, summary, x=RHO, y="recall_selected", sem="recall_selected_sem")
    at_star = summary[
        (summary["study"] == POOLED) & (summary["metric_name"] == HEADLINE_METRICS[0])
    ].sort_values(RHO)
    _line(ax, at_star[RHO], at_star["recall_k_star"], FOREGROUND_COLOR, linestyle="--")
    _band(
        ax,
        at_star[RHO],
        at_star["recall_k_star"],
        at_star["recall_k_star_sem"],
        FOREGROUND_COLOR,
    )
    ax.set_ylim(PROPORTION_LIMITS[0], KEY_HEADROOM_TOP)
    ax.set_yticks(ARI_TICKS)
    _key(
        ax,
        handles=[
            Line2D([], [], color=FOREGROUND_COLOR, label=r"at $\hat{k}$"),
            Line2D([], [], color=FOREGROUND_COLOR, linestyle="--", label="at $k^*$"),
        ],
        loc="upper center",
        ncol=2,
    )
    ax.set_ylabel("Recall of the smallest cluster")


def _similarity(ax: Axes, frames: Mapping[str, pd.DataFrame]) -> None:
    """Subsample versus full data at k*, over the swept rho, and the refit
    reference at REFERENCE_RATIO as its own open marker: the full data
    clustered again with another seed, not a subsample."""
    summary = similarity_at_k_star(frames["similarity"], frames["datasets"])
    pooled = summary[summary["study"] == POOLED].sort_values(RHO)
    swept = pooled[pooled[RHO] < REFERENCE_RATIO]
    refit = pooled[pooled[RHO] == REFERENCE_RATIO]
    _line(ax, swept[RHO], swept["ari_mean"], FOREGROUND_COLOR)
    _band(ax, swept[RHO], swept["ari_mean"], swept["ari_sem"], FOREGROUND_COLOR)
    marker = {
        "linestyle": "none",
        "marker": "o",
        "markersize": 5,
        "markerfacecolor": "white",
        "markeredgecolor": FOREGROUND_COLOR,
    }
    ax.plot(refit[RHO], refit["ari_mean"], **marker)
    _key(
        ax,
        handles=[Line2D([], [], label="Full-data refit", **marker)],
        loc="lower right",
    )
    ax.set_ylim(ARI_LIMITS)
    ax.set_yticks(ARI_TICKS)
    ax.set_ylabel("ARI to full-data clustering")


def figure_ablation_resampling(
    frames: Mapping[str, pd.DataFrame],
    *,
    ablation,
    scale: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Bias of the selected k, rare-cluster recall and subsample-versus-full
    agreement against rho."""
    view = arm_view(frames, ablation=ablation, scale=scale, arm="rho")
    difficulty = ablation.scales[scale].rho_arm.difficulties[-1]

    with print_theme_context():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(PRINT_WIDTH_IN, RESAMPLING_HEIGHT_IN + LEGEND_HEIGHT_IN),
            layout="constrained",
        )
        _bias(axes[0], view)
        _label(axes[0], "A", "Bias of the selected $k$")
        _rare_recall(axes[1], view, difficulty=difficulty)
        _label(axes[1], "B", f"Rare-cluster recall, {difficulty}")
        _similarity(axes[2], frames)
        _label(axes[2], "C", "Subsample vs. full at $k^*$")
        for ax in axes:
            _reference(ax, ablation.rho_default)
            _rho_axis(ax)
            ax.set_xlabel(RHO_LABEL)
            style_axes(ax)
        _legend(fig, rf"Default $\rho$ = {ablation.rho_default:g}")
        if save:
            save_figure(
                fig,
                figure_path(
                    "si_fig_ablation_resampling.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig
