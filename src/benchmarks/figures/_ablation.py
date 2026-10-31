"""The two SI figures of the rho/B ablation.

figure_ablation draws the ARI to the true labels of the labels each
headline selector returns, against the subsampling proportion (the rho arm,
at the default B) and against the resample count (the B arm, at the default
rho). A dashed line marks the
oracle at k*. An ablation with a case study adds a row with the study's
selected labels scored against its reference labels.

figure_ablation_resampling draws how resampling affects the clustering,
against rho: the bias of the selected k, recall of the smallest
true cluster, and the agreement of a subsample's clustering with the
full-data clustering. All three need a true k, so it draws the simulations
only.

Both are laid out at the PLOS page width under print_theme_context, so
their text prints at the sizes set in _theme. Color carries the selector,
stability 1SE and generalizability 1SE in their metric colors; a line that
belongs to no selector is grey, and the legend names it, since a label
inside a panel lands on the error bars. Each simulated scenario gets its own
row of panels, titled by scenario only when there is more than one. Error
bars are the mean plus or minus Z_95 standard errors over datasets, the
selectors nudged apart so their bars do not overlap, and a dotted line
marks the package default. The y axis of an ARI or proportion panel is
fitted to what it draws (fitted_limits). Every number comes from
_ablation_summary.

draw_selectors and arm_axis are public so the notebook's working plots
draw in the same style.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, NullLocator
from matplotlib.transforms import ScaledTranslation

from .._ablation_cells import REFERENCE_RATIO, arm_view
from .._ablation_summary import (
    HEADLINE_METRICS,
    POOLED,
    ari_summary,
    bias_summary,
    oracle_summary,
    rare_recall_summary,
    similarity_at_k_star,
    study_ari_summary,
)
from .._registry import METRIC_DISPLAY_NAMES
from .._theme import (
    FALLBACK_COLOR,
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
# The y axis of an ARI or proportion panel is fitted to what the panel
# draws, error bars included, rounded out to whole Y_STEPs and never
# narrower than MIN_Y_SPAN. Y_PAD keeps markers at the limits clear of the
# spines. Ticks fall every Y_STEP, or every two on a span wider than
# WIDE_Y_SPAN.
Y_STEP = 0.05
MIN_Y_SPAN = 0.2
WIDE_Y_SPAN = 0.3
Y_PAD = 0.02
# The selectors sit this many points apart, nudged evenly to either side of
# their x, so their error bars do not draw over each other. The nudge is in
# display space, so it is the same on the linear rho axis and the log B
# axis.
DODGE_PT = 2.5
ERROR_LINEWIDTH = 0.8
ERROR_CAPSIZE = 1.5
# A line that belongs to no selector (recall at k*, subsample versus full
# data) is drawn in the oracle's grey, the theme's neutral data color.
NEUTRAL_COLOR = metric_color("baseline_oracle")
ROW_HEIGHT_IN = 2.6
RESAMPLING_HEIGHT_IN = 2.2
LEGEND_HEIGHT_IN = 0.35
RHO_TICKS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
# The panel letter sits outside the axes, left of the spine by this many
# points, on the title's baseline; the title starts at the spine, so the two
# cannot overlap however long the title is.
LETTER_OFFSET_PT = -30.0


def fitted_limits(
    values: Iterable[float],
    *,
    step: float = Y_STEP,
    min_span: float = MIN_Y_SPAN,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """The narrowest range of whole steps that holds every finite value and
    spans at least min_span.

    A range short of min_span is widened evenly and then shifted back inside
    bounds, so a proportion near 1 widens downward rather than past 1. The
    values themselves are never cut: an error bar that crosses a bound
    extends the range past it.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return bounds
    eps = 1e-9
    data_lo = np.floor(v.min() / step + eps) * step
    data_hi = np.ceil(v.max() / step - eps) * step
    lower, upper = min(bounds[0], data_lo), max(bounds[1], data_hi)
    lo, hi = data_lo, data_hi
    missing = int(np.ceil((min_span - (hi - lo)) / step - eps))
    if missing > 0:
        lo -= (missing // 2) * step
        hi += (missing - missing // 2) * step
    if hi > upper:
        lo, hi = lo - (hi - upper), upper
    if lo < lower:
        lo, hi = lower, min(hi + (lower - lo), upper)
    return round(float(lo), 10), round(float(hi), 10)


def _set_fitted_ylim(axes: Iterable[Axes], values: Iterable[float]) -> None:
    """One fitted y range for every axes in axes, padded by Y_PAD and ticked
    at whole steps."""
    lo, hi = fitted_limits(values)
    pad = Y_PAD * (hi - lo)
    wide = hi - lo > WIDE_Y_SPAN + 1e-9
    for ax in axes:
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MultipleLocator(2 * Y_STEP if wide else Y_STEP))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f" if wide else "%.2f"))


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


def _title(scenario: str | None, text: str) -> str:
    """text, led by the scenario's title when there is one to tell apart."""
    if scenario is None:
        return text[0].upper() + text[1:]
    return f"{STUDY_TITLES.get(scenario, scenario)}: {text}"


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


def _b_axis(ax: Axes, grid: Sequence[int]) -> None:
    """Log B axis ticked at the grid values, written as plain numbers."""
    ax.set_xscale("log")
    ax.set_xticks(list(grid), [f"{b:g}" for b in grid])
    ax.xaxis.set_minor_locator(NullLocator())


def arm_axis(ax: Axes, *, x: str, ablation) -> None:
    """The x axis of one arm, after the panel is drawn: rho linear and
    ticked every 0.2, or B on a log scale ticked at the grid; its label; and
    a dotted line at the package default."""
    if x == RHO:
        _reference(ax, ablation.rho_default)
        _rho_axis(ax)
        ax.set_xlabel(RHO_LABEL)
    else:
        _reference(ax, ablation.b_default)
        _b_axis(ax, list(ablation.b_grid))
        ax.set_xlabel(B_LABEL)


def _errors(
    ax: Axes,
    x,
    mean,
    err,
    color: str,
    *,
    offset_pt: float = 0.0,
    **kwargs,
) -> None:
    """The mean as a line with markers and err (None for none) as error
    bars, shifted offset_pt points sideways. kwargs override the line's
    style.

    Autoscaling skips artists drawn through a shifted transform, so the
    unshifted data limits are registered by hand.
    """
    x, mean = np.asarray(x, dtype=float), np.asarray(mean, dtype=float)
    ys = [mean] if err is None else [mean - err, mean + err]
    ax.update_datalim(np.column_stack([np.tile(x, len(ys)), np.concatenate(ys)]))
    ax.autoscale_view()
    transform = ax.transData
    if offset_pt:
        transform = transform + ScaledTranslation(
            offset_pt / 72, 0, ax.figure.dpi_scale_trans
        )
    style = {
        "color": color,
        "marker": "o",
        "markeredgecolor": "white",
        "markeredgewidth": 0.5,
        "elinewidth": ERROR_LINEWIDTH,
        "capsize": ERROR_CAPSIZE,
        "capthick": ERROR_LINEWIDTH,
        "transform": transform,
    }
    ax.errorbar(x, mean, yerr=err, **{**style, **kwargs})


def draw_selectors(
    ax: Axes,
    summary: pd.DataFrame,
    *,
    x: str,
    y: str = "ari_mean",
    sem: str | None = "ari_sem",
    study: str | None = POOLED,
) -> np.ndarray:
    """One line per headline selector in its metric color, labeled with its
    display name, with the mean plus or minus Z_95 standard errors as error
    bars and the selectors nudged DODGE_PT apart. sem=None draws the lines
    alone. study=None takes the frame as already restricted to one study.

    Returns every value drawn, error-bar ends included, to fit the y axis
    to.
    """
    drawn: list[pd.Series] = []
    center = (len(HEADLINE_METRICS) - 1) / 2
    for index, metric in enumerate(HEADLINE_METRICS):
        rows = summary[summary["metric_name"] == metric]
        if study is not None:
            rows = rows[rows["study"] == study]
        rows = rows.sort_values(x)
        if rows.empty:
            continue
        err = None if sem is None else Z_95 * rows[sem]
        _errors(
            ax,
            rows[x],
            rows[y],
            err,
            metric_color(metric),
            offset_pt=DODGE_PT * (index - center),
            label=METRIC_DISPLAY_NAMES.get(metric, metric),
        )
        drawn += [rows[y]] if err is None else [rows[y] - err, rows[y] + err]
    return np.concatenate(drawn) if drawn else np.empty(0)


# The grey references: their style on the axes and their legend key.
REFERENCE_STYLE: dict[str, object] = {
    "color": NEUTRAL_COLOR,
    "linestyle": "--",
    "linewidth": PRINT_REFERENCE_LINEWIDTH,
}
REFIT_STYLE: dict[str, object] = {
    "linestyle": "none",
    "marker": "o",
    "markersize": 5,
    "markerfacecolor": "white",
    "markeredgecolor": NEUTRAL_COLOR,
    "markeredgewidth": 1.0,
}


def _oracle(ax: Axes, value: float) -> None:
    """The oracle's ARI as a dashed grey line."""
    ax.axhline(value, **REFERENCE_STYLE)


def _simulations(
    ax: Axes, view: Mapping[str, pd.DataFrame], *, x: str, study: str
) -> np.ndarray:
    """One scenario: its selectors and, under them, its oracle averaged over
    the datasets this arm ran, left out while the scenario has no cell yet.
    Returns the values drawn."""
    drawn = draw_selectors(
        ax, ari_summary(view["selection"], view["at_k"], x=x), x=x, study=study
    )
    oracle = oracle_summary(view)
    value = oracle.loc[oracle["study"] == study, "oracle_ari"]
    if not value.empty:
        _oracle(ax, float(value.iloc[0]))
        drawn = np.append(drawn, float(value.iloc[0]))
    ax.set_ylabel("ARI to true labels")
    return drawn


def _study(
    ax: Axes, view: Mapping[str, pd.DataFrame], *, x: str, study: str
) -> np.ndarray:
    """Mean and 95% interval over replicates of the selected labels' ARI
    against the study's reference labels."""
    stats = study_ari_summary(
        view["selection"], x=x, study=study, metrics=HEADLINE_METRICS
    )
    drawn = draw_selectors(ax, stats, x=x, study=None)
    ax.set_ylabel("ARI to reference labels")
    return drawn


def _interval_handle() -> ErrorbarContainer:
    """A grey error bar with caps, the legend's key for every panel's 95%
    interval. It is drawn by the legend only, never on an axes."""
    bar = LineCollection([], colors=FALLBACK_COLOR, linewidths=ERROR_LINEWIDTH)
    cap = Line2D(
        [],
        [],
        color=FALLBACK_COLOR,
        marker="_",
        markersize=2 * ERROR_CAPSIZE,
        markeredgewidth=ERROR_LINEWIDTH,
        linestyle="none",
    )
    return ErrorbarContainer((None, (cap, cap), (bar,)), has_yerr=True, label="95% CI")


def _legend(fig: Figure, default_label: str, references: Sequence[Line2D] = ()) -> None:
    """The figure's one shared legend, below the panels as in the other
    benchmarking figures: the selectors, the grey references, the interval
    and the default. More than five entries wrap onto two rows of three."""
    handles: list[Line2D | ErrorbarContainer] = [
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
    handles += list(references)
    handles.append(_interval_handle())
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
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles) if len(handles) <= 5 else 3,
    )


def figure_ablation(
    frames: Mapping[str, pd.DataFrame],
    *,
    ablation,
    scale: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """ARI of the selected labels against rho (left) and B (right), one row
    per simulated scenario, and a row for the case study when the ablation
    has one. Every panel shares one fitted y axis."""
    rho_view = arm_view(frames, ablation=ablation, scale=scale, arm="rho")
    b_view = arm_view(frames, ablation=ablation, scale=scale, arm="b")
    rho_default, b_default = ablation.rho_default, ablation.b_default
    study = ablation.study
    several = len(ablation.scenarios) > 1
    n_rows = len(ablation.scenarios) + (study is not None)

    with print_theme_context():
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(PRINT_WIDTH_IN, n_rows * ROW_HEIGHT_IN + LEGEND_HEIGHT_IN),
            layout="constrained",
            squeeze=False,
        )
        letters = iter("ABCDEFGH")
        drawn = []
        for row, name in enumerate(ablation.scenarios):
            scenario = name if several else None
            drawn.append(_simulations(axes[row, 0], rho_view, x=RHO, study=name))
            _label(
                axes[row, 0],
                next(letters),
                _title(scenario, rf"varying $\rho$, $B$ = {b_default:g}"),
            )
            drawn.append(_simulations(axes[row, 1], b_view, x=B, study=name))
            _label(
                axes[row, 1],
                next(letters),
                _title(scenario, rf"varying $B$, $\rho$ = {rho_default:g}"),
            )

        if study is not None:
            drawn.append(_study(axes[-1, 0], rho_view, x=RHO, study=study))
            _label(axes[-1, 0], next(letters), _title(study, r"varying $\rho$"))
            drawn.append(_study(axes[-1, 1], b_view, x=B, study=study))
            _label(axes[-1, 1], next(letters), _title(study, "varying $B$"))

        for ax in axes[:, 0]:
            arm_axis(ax, x=RHO, ablation=ablation)
        for ax in axes[:, 1]:
            arm_axis(ax, x=B, ablation=ablation)
        _set_fitted_ylim(axes.flat, np.concatenate(drawn))
        for ax in axes.flat:
            style_axes(ax)
        _legend(
            fig,
            rf"Default ($\rho$ = {rho_default:g}, $B$ = {b_default:g})",
            [Line2D([], [], label="Oracle $k^*$", **REFERENCE_STYLE)],
        )
        if save:
            save_figure(
                fig,
                figure_path(
                    "si_fig_ablation.png", subdir=BENCHMARKING_DIR, out_dir=out_dir
                ),
            )
    return fig


def _bias(ax: Axes, view: Mapping[str, pd.DataFrame], *, study: str) -> None:
    draw_selectors(
        ax,
        bias_summary(view["selection"], x=RHO),
        x=RHO,
        y="bias_mean",
        sem="bias_sem",
        study=study,
    )
    ax.axhline(0.0, color=FALLBACK_COLOR, linewidth=PRINT_REFERENCE_LINEWIDTH)
    ax.set_ylabel(r"Mean $\hat{k} - k^*$")


def _rare_recall(
    ax: Axes, view: Mapping[str, pd.DataFrame], *, difficulty: str, study: str
) -> np.ndarray:
    """Recall of the smallest true cluster in the labels at the selected k,
    one line per rule, and at k* as a dashed grey reference without error
    bars. Every rule returns the stability-consensus labels, so recall at k*
    is the same for both and is drawn once. Returns the values drawn."""
    summary = rare_recall_summary(
        view["at_k"], view["selection"], x=RHO, difficulty=difficulty
    )
    drawn = draw_selectors(
        ax,
        summary,
        x=RHO,
        y="recall_selected",
        sem="recall_selected_sem",
        study=study,
    )
    at_star = summary[
        (summary["study"] == study) & (summary["metric_name"] == HEADLINE_METRICS[0])
    ].sort_values(RHO)
    ax.plot(at_star[RHO], at_star["recall_k_star"], **REFERENCE_STYLE)
    drawn = np.append(drawn, at_star["recall_k_star"])
    ax.set_ylabel(r"Recall at $\hat{k}$")
    return drawn


def _similarity(
    ax: Axes, frames: Mapping[str, pd.DataFrame], *, study: str
) -> np.ndarray:
    """Subsample versus full data at k*, over the swept rho, and the refit
    reference at REFERENCE_RATIO as its own open marker: the full data
    clustered again with another seed, not a subsample. Returns the values
    drawn."""
    summary = similarity_at_k_star(frames["similarity"], frames["datasets"])
    rows = summary[summary["study"] == study].sort_values(RHO)
    swept = rows[rows[RHO] < REFERENCE_RATIO]
    refit = rows[rows[RHO] == REFERENCE_RATIO]
    _errors(ax, swept[RHO], swept["ari_mean"], Z_95 * swept["ari_sem"], NEUTRAL_COLOR)
    _errors(
        ax,
        refit[RHO],
        refit["ari_mean"],
        Z_95 * refit["ari_sem"],
        NEUTRAL_COLOR,
        **REFIT_STYLE,
    )
    ax.set_ylabel("ARI to full-data clustering")
    return np.concatenate(
        [
            rows["ari_mean"] - Z_95 * rows["ari_sem"],
            rows["ari_mean"] + Z_95 * rows["ari_sem"],
        ]
    )


def figure_ablation_resampling(
    frames: Mapping[str, pd.DataFrame],
    *,
    ablation,
    scale: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Bias of the selected k, rare-cluster recall and subsample-versus-full
    agreement against rho, one row per simulated scenario. The two
    proportion panels each get a fitted y axis; bias is autoscaled."""
    references = [
        Line2D([], [], label="Recall at $k^*$", **REFERENCE_STYLE),
        Line2D([], [], label="Full-data refit", **REFIT_STYLE),
    ]
    view = arm_view(frames, ablation=ablation, scale=scale, arm="rho")
    difficulty = ablation.scales[scale].rho_arm.difficulties[-1]
    several = len(ablation.scenarios) > 1
    n_rows = len(ablation.scenarios)

    with print_theme_context():
        fig, axes = plt.subplots(
            n_rows,
            3,
            figsize=(
                PRINT_WIDTH_IN,
                n_rows * RESAMPLING_HEIGHT_IN + 2 * LEGEND_HEIGHT_IN,
            ),
            layout="constrained",
            squeeze=False,
        )
        letters = iter("ABCDEFGHIJKL")
        for row, name in enumerate(ablation.scenarios):
            scenario = name if several else None
            _bias(axes[row, 0], view, study=name)
            _label(axes[row, 0], next(letters), _title(scenario, r"bias of $\hat{k}$"))
            recall = _rare_recall(axes[row, 1], view, difficulty=difficulty, study=name)
            _set_fitted_ylim([axes[row, 1]], recall)
            _label(axes[row, 1], next(letters), "Recall of the smallest cluster")
            similarity = _similarity(axes[row, 2], frames, study=name)
            _set_fitted_ylim([axes[row, 2]], similarity)
            _label(axes[row, 2], next(letters), "Subsample vs. full data at $k^*$")
        for ax in axes.flat:
            arm_axis(ax, x=RHO, ablation=ablation)
            style_axes(ax)
        _legend(fig, rf"Default $\rho$ = {ablation.rho_default:g}", references)
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
