"""The two SI figures of the rho/B ablation.

figure_ablation_rho reads the rho arm, figure_ablation_b the B arm. Both are
laid out at the PLOS page width under print_theme_context, so their text
prints at the sizes set in _theme. Color carries the selector in every
panel, stability 1SE and generalizability 1SE in their metric colors;
simulation panels draw the pooled simulations, which with one scenario are
that scenario. Shaded bands are 95% intervals: Wilson for recovery, the
mean plus or minus Z_95 standard errors otherwise. The B figure's
repeatability panels (agreement, spread, reported standard error) measure
variability themselves and carry no band. A dotted line marks the package
default. Every number comes from _ablation_summary, except the pooled
subsample-versus-full similarity at k*, which _similarity_at_k_star
aggregates here.
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
    HEADLINE_METRICS,
    POOLED,
    agreement_summary,
    curve_at_k_star,
    rare_recall_summary,
    selection_summary,
    similarity_summary,
    spread_summary,
    study_ari_summary,
    study_selection_shares,
)
from .._registry import METRIC_DISPLAY_NAMES
from .._studies import STUDIES
from .._theme import (
    FALLBACK_COLOR,
    FOREGROUND_COLOR,
    PRINT_FONT_SIZES,
    PRINT_REFERENCE_LINEWIDTH,
    PRINT_WIDTH_IN,
    cluster_colors,
    metric_color,
    print_theme_context,
    save_figure,
    style_axes,
)
from ._benchmarking_examples import SCENARIO_TITLES
from ._paths import BENCHMARKING_DIR, figure_path

STUDY_TITLES: dict[str, str] = {**SCENARIO_TITLES, "klein": "Klein"}
RHO_LABEL = r"Subsampling proportion $\rho$"
B_LABEL = r"Resamples $B$"
STAB, GEN = HEADLINE_METRICS

# A 95% interval from a mean and its standard error, by the normal
# approximation; wilson_ci uses the same z for recovery.
Z_95 = 1.96
# Recovery, recall and agreement are proportions and share one fixed axis,
# padded so markers at 0 and 1 are not clipped. Autoscaling them made a
# difference of one dataset in twenty look like a trend.
PROPORTION_LIMITS: tuple[float, float] = (-0.04, 1.04)
PROPORTION_TICKS = np.linspace(0.0, 1.0, 6)
# A proportion panel that carries its own key (rare-cluster recall, the
# study's share bars) extends above 1 to hold it off the data; the ticks
# stay 0 to 1.
KEY_HEADROOM_TOP = 1.32
ROW_HEIGHT_IN = 2.2
LEGEND_HEIGHT_IN = 0.35
BAND_ALPHA = 0.18
# The estimators of a study's similarity panel, told apart by dash since
# color is reserved for the selectors.
ESTIMATOR_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")
# A share bar on the log B axis spans its B divided and multiplied by this
# factor, so it is centered on its B; the grid's closest neighbors (25 and
# 50) are a factor of 2 apart, so bars do not touch.
B_BAR_FACTOR = 1.25
RHO_BAR_WIDTH = 0.05
RHO_TICKS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
# The panel letter sits outside the axes, left of the spine by this many
# points, on the title's baseline; the title starts at the spine, so the two
# cannot overlap however long the title is.
LETTER_OFFSET_PT = -30.0


def _figure(n_rows: int) -> tuple[Figure, np.ndarray]:
    height = n_rows * ROW_HEIGHT_IN + LEGEND_HEIGHT_IN
    return plt.subplots(
        n_rows,
        3,
        figsize=(PRINT_WIDTH_IN, height),
        layout="constrained",
        squeeze=False,
    )


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


def _key(ax: Axes, **kwargs) -> None:
    """An in-panel legend. It is left out of the layout, so a long key
    (the study's share bars can list many choices) overflows its panel
    rather than shrinking every panel in the figure."""
    ax.legend(**kwargs).set_in_layout(False)


def _short(estimator: str) -> str:
    return estimator.removesuffix("Clustering")


def _reference(ax: Axes, x_default: float) -> None:
    ax.axvline(
        x_default,
        color=FALLBACK_COLOR,
        linestyle=":",
        linewidth=PRINT_REFERENCE_LINEWIDTH,
        zorder=1,
    )


def _proportion(ax: Axes, *, headroom: bool = False) -> None:
    top = KEY_HEADROOM_TOP if headroom else PROPORTION_LIMITS[1]
    ax.set_ylim(PROPORTION_LIMITS[0], top)
    ax.set_yticks(PROPORTION_TICKS)


def _zero(ax: Axes) -> None:
    ax.axhline(0.0, color=FALLBACK_COLOR, linewidth=PRINT_REFERENCE_LINEWIDTH)


def _rho_axis(ax: Axes) -> None:
    """rho ticked every 0.2 up to what the panel draws (the similarity
    panels reach the refit at 1), written as plain numbers."""
    top = ax.get_xlim()[1]
    ticks = [t for t in RHO_TICKS if t <= top]
    ax.set_xticks(ticks, [f"{t:.1f}" for t in ticks])


def _b_axis(ax: Axes, grid: Sequence[int]) -> None:
    """Log B axis ticked at the grid values, written as plain numbers."""
    ax.set_xscale("log")
    ax.set_xticks(list(grid), [f"{b:g}" for b in grid])
    ax.xaxis.set_minor_locator(NullLocator())


def _band(ax: Axes, x, lower, upper, color: str) -> None:
    ax.fill_between(x, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0)


def _line(ax: Axes, x, y, color: str, *, linestyle="-", label=None) -> None:
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker="o",
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=label,
    )


def _selectors(
    ax: Axes,
    summary: pd.DataFrame,
    *,
    x: str,
    y: str,
    study: str | None = POOLED,
    lo: str | None = None,
    hi: str | None = None,
    sem: str | None = None,
    linestyle: str = "-",
) -> None:
    """One line per headline selector in its metric color, with its band.

    lo and hi name the interval columns directly (Wilson bounds); sem draws
    the mean plus or minus Z_95 standard errors. study=None takes the frame
    as already restricted to one study.
    """
    for metric in HEADLINE_METRICS:
        rows = summary[summary["metric_name"] == metric]
        if study is not None:
            rows = rows[rows["study"] == study]
        rows = rows.sort_values(x)
        if rows.empty:
            continue
        color = metric_color(metric)
        _line(
            ax,
            rows[x],
            rows[y],
            color,
            linestyle=linestyle,
            label=METRIC_DISPLAY_NAMES.get(metric, metric),
        )
        if lo is not None and hi is not None:
            _band(ax, rows[x], rows[lo], rows[hi], color)
        elif sem is not None:
            _band(
                ax,
                rows[x],
                rows[y] - Z_95 * rows[sem],
                rows[y] + Z_95 * rows[sem],
                color,
            )


def _oracle(ax: Axes, value: float) -> None:
    """The oracle's ARI as a dashed line, labeled at its left end.

    The axis is extended above the line if needed, so the label sits in
    clear space rather than over the selectors' bands; the left end keeps
    it off the default's reference line, which both grids place right of
    center.
    """
    color = metric_color("baseline_oracle")
    ax.axhline(
        value,
        color=color,
        linestyle="--",
        linewidth=PRINT_REFERENCE_LINEWIDTH,
        label="Oracle $k^*$",
    )
    low, high = ax.get_ylim()
    ax.set_ylim(low, max(high, value + 0.12 * (value - low)))
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


def _study_ari(ax: Axes, selection: pd.DataFrame, *, x: str, study: str) -> None:
    """Mean and 95% interval over replicates of the selected labels' ARI
    against the study's reference labels."""
    stats = study_ari_summary(selection, x=x, study=study, metrics=HEADLINE_METRICS)
    _selectors(ax, stats, x=x, y="ari_mean", sem="ari_sem", study=None)
    ax.set_ylabel("ARI to reference labels")


def _study_shares(
    ax: Axes,
    selection: pd.DataFrame,
    *,
    x: str,
    grid: Sequence,
    study: str,
    metric: str,
    log: bool,
) -> None:
    """Stacked bars at the swept values: share of replicates selecting each
    (estimator, k)."""
    shares = study_selection_shares(selection, x=x, metric=metric, study=study)
    # study_selection_shares returns selected_k as an int, even when an
    # undefined selection elsewhere in the run has made the selection
    # frame's column float64; the ":d" format relies on that and would
    # raise on a float rather than label a bar "k=4.0".
    shares["choice"] = [
        f"{_short(estimator)}, k={k:d}"
        for estimator, k in zip(shares["selected_estimator"], shares["selected_k"])
    ]
    choices = sorted(
        shares["choice"].unique(),
        key=lambda c: (c.split(", k=")[0], int(c.split("k=")[1])),
    )
    colors = dict(zip(choices, cluster_colors(len(choices))))
    values = np.asarray(grid, dtype=float)
    if log:
        left = values / B_BAR_FACTOR
        widths = values * B_BAR_FACTOR - left
    else:
        left = values - RHO_BAR_WIDTH / 2
        widths = np.full(len(values), RHO_BAR_WIDTH)
    bottom = np.zeros(len(values))
    for choice in choices:
        heights = np.array(
            [
                float(
                    shares[(shares[x] == value) & (shares["choice"] == choice)][
                        "share"
                    ].sum()
                )
                for value in grid
            ]
        )
        ax.bar(
            left,
            heights,
            width=widths,
            bottom=bottom,
            align="edge",
            color=colors[choice],
            edgecolor="white",
            linewidth=0.5,
            label=choice,
        )
        bottom += heights
    _proportion(ax, headroom=True)
    ax.set_ylabel("Share of replicates")
    _key(ax, loc="upper center", ncol=2, handlelength=1.0, columnspacing=0.8)


def _similarity_at_k_star(
    similarity: pd.DataFrame, datasets: pd.DataFrame
) -> pd.DataFrame:
    """Pooled subsample-versus-full ARI at each dataset's k*, per rho.

    Draws (and estimators) are averaged within a dataset first, so the
    standard error is over datasets, the independent unit. The study has no
    k* and drops out at the merge.
    """
    merged = similarity.merge(
        datasets[["study", "difficulty", "dataset", "k_star"]],
        on=["study", "difficulty", "dataset"],
    )
    rows = merged[merged["k"] == merged["k_star"]]
    per_dataset = rows.groupby(
        ["subsample_ratio", "study", "difficulty", "dataset"], as_index=False
    )["ari"].mean()
    return per_dataset.groupby("subsample_ratio", as_index=False).agg(
        ari_mean=("ari", "mean"), ari_sem=("ari", "sem")
    )


def _swept_and_refit(
    ax: Axes, frame: pd.DataFrame, *, linestyle: str = "-", label=None
) -> None:
    """A similarity line over the swept rho values, and the refit reference
    at REFERENCE_RATIO as its own open marker: the full data clustered again
    with another seed, not a subsample."""
    frame = frame.sort_values("subsample_ratio")
    swept = frame[frame["subsample_ratio"] < REFERENCE_RATIO]
    refit = frame[frame["subsample_ratio"] == REFERENCE_RATIO]
    _line(
        ax,
        swept["subsample_ratio"],
        swept["ari_mean"],
        FOREGROUND_COLOR,
        linestyle=linestyle,
        label=label,
    )
    _band(
        ax,
        swept["subsample_ratio"],
        swept["ari_mean"] - Z_95 * swept["ari_sem"],
        swept["ari_mean"] + Z_95 * swept["ari_sem"],
        FOREGROUND_COLOR,
    )
    if not refit.empty:
        ax.plot(
            refit["subsample_ratio"],
            refit["ari_mean"],
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=FOREGROUND_COLOR,
            label="_refit",
        )


def _refit_legend_handle() -> Line2D:
    return Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor=FOREGROUND_COLOR,
        label="Full-data refit",
    )


def figure_ablation_rho(
    frames: Mapping[str, pd.DataFrame],
    *,
    ablation,
    scale: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Selections, scores and recovery against the subsampling proportion."""
    x = "subsample_ratio"
    view = arm_view(frames, ablation=ablation, scale=scale, arm="rho")
    study = ablation.study
    selection = view["selection"]
    sel = selection_summary(selection, x=x)
    rare_difficulty = ablation.scales[scale].rho_arm.difficulties[-1]
    recall = rare_recall_summary(
        view["at_k"], selection, x=x, difficulty=rare_difficulty
    )
    at_star = curve_at_k_star(view["curves"], view["datasets"], x=x)
    similarity = _similarity_at_k_star(view["similarity"], view["datasets"])
    oracle = float(view["datasets"]["oracle_ari"].mean())

    with print_theme_context():
        fig, axes = _figure(3 if study is not None else 2)

        ax = axes[0, 0]
        _selectors(ax, sel, x=x, y="recovery", lo="recovery_lo", hi="recovery_hi")
        _proportion(ax)
        _label(ax, "A", "$k^*$ recovery")
        ax.set_ylabel("Recovery rate")

        ax = axes[0, 1]
        _selectors(ax, sel, x=x, y="bias_mean", sem="bias_sem")
        _zero(ax)
        _label(ax, "B", "Bias of the selected $k$")
        ax.set_ylabel(r"Mean $\hat{k} - k^*$")

        ax = axes[0, 2]
        _selectors(ax, sel, x=x, y="ari_mean", sem="ari_sem")
        _oracle(ax, oracle)
        _label(ax, "C", "ARI at the selected $k$")
        ax.set_ylabel("ARI to true labels")

        ax = axes[1, 0]
        _selectors(ax, recall, x=x, y="recall_selected", sem="recall_selected_sem")
        _selectors(
            ax,
            recall,
            x=x,
            y="recall_k_star",
            sem="recall_k_star_sem",
            linestyle="--",
        )
        _proportion(ax, headroom=True)
        _key(
            ax,
            handles=[
                Line2D([], [], color=FOREGROUND_COLOR, label=r"at $\hat{k}$"),
                Line2D(
                    [], [], color=FOREGROUND_COLOR, linestyle="--", label="at $k^*$"
                ),
            ],
            loc="upper center",
            ncol=2,
        )
        _label(ax, "D", f"Rare-cluster recall, {rare_difficulty}")
        ax.set_ylabel("Recall of the smallest cluster")

        ax = axes[1, 1]
        _selectors(ax, at_star, x=x, y="value_mean", sem="value_sem")
        _label(ax, "E", "Score at $k^*$")
        ax.set_ylabel("Criterion score (ARI)")

        ax = axes[1, 2]
        _swept_and_refit(ax, similarity)
        _key(ax, handles=[_refit_legend_handle()], loc="lower right")
        _label(ax, "F", "Subsample vs. full at $k^*$")
        ax.set_ylabel("ARI to full-data clustering")

        if study is not None:
            title = STUDY_TITLES.get(study, study)
            ax = axes[2, 0]
            _study_ari(ax, selection, x=x, study=study)
            _label(ax, "G", f"{title}: selected labels")

            ax = axes[2, 1]
            # The manuscript reports one k for this study; read it from the
            # registry rather than restating it, and leave the panel empty
            # if the study reports none.
            reported_k = STUDIES[study].reported_k
            if reported_k is not None:
                rows = similarity_summary(
                    view["similarity"][view["similarity"]["study"] == study]
                )
                rows = rows[rows["k"] == reported_k]
                for (estimator, part), linestyle in zip(
                    rows.groupby("estimator"), ESTIMATOR_LINESTYLES
                ):
                    _swept_and_refit(
                        ax,
                        part,
                        linestyle=linestyle,
                        label=f"{_short(estimator)}, k={reported_k}",
                    )
                handles = [
                    line for line in ax.lines if not line.get_label().startswith("_")
                ]
                _key(ax, handles=[*handles, _refit_legend_handle()], loc="lower right")
            _label(ax, "H", f"{title}: subsample vs. full")
            ax.set_ylabel("ARI to full-data clustering")

            ax = axes[2, 2]
            _study_shares(
                ax,
                selection,
                x=x,
                grid=list(ablation.rho_grid),
                study=study,
                metric=GEN,
                log=False,
            )
            _label(ax, "I", f"{title}: selections")

        for ax in axes.flat:
            if ax.get_label() != "I":
                _reference(ax, ablation.rho_default)
            _rho_axis(ax)
            ax.set_xlabel(RHO_LABEL)
            style_axes(ax)
        _legend(fig, rf"Default $\rho$ = {ablation.rho_default:g}")
        if save:
            save_figure(
                fig,
                figure_path(
                    "si_fig_ablation_rho.png", subdir=BENCHMARKING_DIR, out_dir=out_dir
                ),
            )
    return fig


def figure_ablation_b(
    frames: Mapping[str, pd.DataFrame],
    *,
    ablation,
    scale: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Repeatability, spread and selections against the resample count."""
    x = "n_resamples"
    view = arm_view(frames, ablation=ablation, scale=scale, arm="b")
    study = ablation.study
    selection = view["selection"]
    agreement = agreement_summary(selection, x=x)
    spread = spread_summary(view["curves"], x=x)
    at_star = curve_at_k_star(view["curves"], view["datasets"], x=x)
    sel = selection_summary(selection, x=x)
    oracle = float(view["datasets"]["oracle_ari"].mean())
    b_default = ablation.b_default
    grid = list(ablation.b_grid)

    def _guide(ax, summary, y):
        # One over the square root of B, anchored at the default.
        rows = summary[(summary["study"] == POOLED) & (summary["metric_name"] == STAB)]
        anchor = rows[rows[x] == b_default][y]
        if anchor.empty or np.isnan(anchor.iloc[0]):
            return
        values = np.array(grid, dtype=float)
        ax.plot(
            values,
            float(anchor.iloc[0]) * np.sqrt(b_default / values),
            linestyle="--",
            color=FALLBACK_COLOR,
            linewidth=PRINT_REFERENCE_LINEWIDTH,
            label=r"$\propto B^{-1/2}$",
        )
        _key(ax, handles=[ax.lines[-1]], loc="upper right")

    with print_theme_context():
        fig, axes = _figure(3 if study is not None else 2)

        ax = axes[0, 0]
        _selectors(ax, agreement, x=x, y="agreement")
        _proportion(ax)
        _label(ax, "A", "Replicate agreement")
        ax.set_ylabel("Agreeing replicate pairs")

        ax = axes[0, 1]
        _selectors(ax, spread, x=x, y="spread")
        _guide(ax, spread, "spread")
        _label(ax, "B", "Spread over replicates")
        ax.set_ylabel("SD of the score")

        ax = axes[0, 2]
        _selectors(ax, at_star, x=x, y="se_mean")
        _guide(ax, at_star, "se_mean")
        _label(ax, "C", "Reported SE at $k^*$")
        ax.set_ylabel("Mean standard error")

        ax = axes[1, 0]
        _selectors(ax, sel, x=x, y="recovery", lo="recovery_lo", hi="recovery_hi")
        _proportion(ax)
        _label(ax, "D", "$k^*$ recovery")
        ax.set_ylabel("Recovery rate")

        ax = axes[1, 1]
        _selectors(ax, sel, x=x, y="bias_mean", sem="bias_sem")
        _zero(ax)
        _label(ax, "E", "Bias of the selected $k$")
        ax.set_ylabel(r"Mean $\hat{k} - k^*$")

        ax = axes[1, 2]
        _selectors(ax, sel, x=x, y="ari_mean", sem="ari_sem")
        _oracle(ax, oracle)
        _label(ax, "F", "ARI at the selected $k$")
        ax.set_ylabel("ARI to true labels")

        if study is not None:
            title = STUDY_TITLES.get(study, study)
            ax = axes[2, 0]
            _selectors(ax, agreement, x=x, y="agreement", study=study)
            _proportion(ax)
            _label(ax, "G", f"{title}: replicate agreement")
            ax.set_ylabel("Agreeing replicate pairs")

            ax = axes[2, 1]
            _study_ari(ax, selection, x=x, study=study)
            _label(ax, "H", f"{title}: selected labels")

            ax = axes[2, 2]
            _study_shares(
                ax,
                selection,
                x=x,
                grid=grid,
                study=study,
                metric=GEN,
                log=True,
            )
            _label(ax, "I", f"{title}: selections")

        for ax in axes.flat:
            if ax.get_label() != "I":
                _reference(ax, b_default)
            _b_axis(ax, grid)
            ax.set_xlabel(B_LABEL)
            style_axes(ax)
        _legend(fig, f"Default $B$ = {b_default:g}")
        if save:
            save_figure(
                fig,
                figure_path(
                    "si_fig_ablation_b.png", subdir=BENCHMARKING_DIR, out_dir=out_dir
                ),
            )
    return fig
