"""The two SI figures of the rho/B ablation.

figure_ablation_rho reads the rho arm, figure_ablation_b the B arm; both
draw pooled simulation lines in the foreground color, one line per study
in the cluster palette, and mark the package default with a dotted
reference line. Every number comes from _ablation_summary.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .._ablation_cells import arm_view
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
from .._panels import grouped_legend
from .._registry import METRIC_DISPLAY_NAMES
from .._studies import STUDIES
from .._theme import (
    CARVE_LINEWIDTH,
    FALLBACK_COLOR,
    FONT_SIZES,
    FOREGROUND_COLOR,
    REFERENCE_LINEWIDTH,
    cluster_colors,
    metric_color,
    save_figure,
    style_axes,
    theme_context,
)
from ._benchmarking_examples import SCENARIO_TITLES
from ._paths import BENCHMARKING_DIR, figure_path

STUDY_TITLES: dict[str, str] = {**SCENARIO_TITLES, "klein": "Klein"}
RHO_LABEL = "Subsampling proportion $\\rho$"
B_LABEL = "Resamples $B$"
STAB, GEN = HEADLINE_METRICS


def _title(ax, letter: str, text: str) -> None:
    ax.set_title(f"{letter}  {text}", loc="left", fontsize=FONT_SIZES["title"])


def _reference(ax, x_default: float) -> None:
    ax.axvline(
        x_default, color=FALLBACK_COLOR, linestyle=":", linewidth=REFERENCE_LINEWIDTH
    )


def _study_colors(studies: Sequence[str]) -> dict[str, str]:
    return dict(zip(studies, cluster_colors(len(studies))))


def _lines_by_study(
    ax,
    summary: pd.DataFrame,
    *,
    x: str,
    y: str,
    metric: str,
    studies: Sequence[str],
    lo: str | None = None,
    hi: str | None = None,
    pooled_label: str = "Pooled (simulations)",
) -> None:
    """One faint line per study plus the pooled line, for one metric."""
    rows = summary[summary["metric_name"] == metric]
    colors = _study_colors(studies)
    for study in studies:
        part = rows[rows["study"] == study].sort_values(x)
        if part.empty:
            continue
        ax.plot(
            part[x],
            part[y],
            marker="o",
            markersize=3,
            linewidth=REFERENCE_LINEWIDTH,
            color=colors[study],
            alpha=0.75,
            label=STUDY_TITLES.get(study, study),
        )
    pooled = rows[rows["study"] == POOLED].sort_values(x)
    if not pooled.empty:
        ax.plot(
            pooled[x],
            pooled[y],
            marker="o",
            markersize=4,
            linewidth=CARVE_LINEWIDTH,
            color=FOREGROUND_COLOR,
            label=pooled_label,
        )
        if lo is not None and hi is not None:
            ax.fill_between(
                pooled[x],
                pooled[lo],
                pooled[hi],
                color=FOREGROUND_COLOR,
                alpha=0.12,
                linewidth=0,
            )


def _pooled_by_metric(
    ax,
    summary: pd.DataFrame,
    *,
    x: str,
    y: str,
    metrics: Sequence[str],
    sem: str | None = None,
    linestyle: str = "-",
    suffix: str = "",
) -> None:
    """The pooled line of each metric, in the metric's color."""
    for metric in metrics:
        rows = summary[
            (summary["metric_name"] == metric) & (summary["study"] == POOLED)
        ]
        rows = rows.sort_values(x)
        if rows.empty:
            continue
        label = METRIC_DISPLAY_NAMES.get(metric, metric) + suffix
        ax.plot(
            rows[x],
            rows[y],
            marker="o",
            markersize=4,
            linewidth=CARVE_LINEWIDTH,
            linestyle=linestyle,
            color=metric_color(metric),
            label=label,
        )
        if sem is not None:
            ax.fill_between(
                rows[x],
                rows[y] - rows[sem],
                rows[y] + rows[sem],
                color=metric_color(metric),
                alpha=0.15,
                linewidth=0,
            )


def _study_ari(
    ax, selection: pd.DataFrame, *, x: str, study: str, metrics: Sequence[str]
) -> None:
    """Mean and standard error over replicates of the selected labels' ARI
    against the study's reference labels. Only plots; the aggregation lives
    in _ablation_summary.study_ari_summary so the notebook can share it."""
    stats = study_ari_summary(selection, x=x, study=study, metrics=metrics)
    for metric in metrics:
        part = stats[stats["metric_name"] == metric].sort_values(x)
        if part.empty:
            continue
        ax.errorbar(
            part[x],
            part["ari_mean"],
            yerr=part["ari_sem"].fillna(0.0),
            marker="o",
            markersize=4,
            linewidth=CARVE_LINEWIDTH,
            color=metric_color(metric),
            capsize=2,
            label=f"{METRIC_DISPLAY_NAMES.get(metric, metric)}, ARI to reference",
        )


def _study_shares(
    ax,
    selection: pd.DataFrame,
    *,
    x: str,
    grid: Sequence,
    study: str,
    metric: str,
    x_label: str,
) -> None:
    """Stacked bars: share of replicates selecting each (estimator, k)."""
    shares = study_selection_shares(selection, x=x, metric=metric, study=study)
    shares["choice"] = (
        shares["selected_estimator"] + ", k=" + shares["selected_k"].astype(str)
    )
    choices = sorted(
        shares["choice"].unique(),
        key=lambda c: (c.split(", k=")[0], int(c.split("k=")[1])),
    )
    colors = dict(zip(choices, cluster_colors(len(choices))))
    positions = np.arange(len(grid))
    bottom = np.zeros(len(grid))
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
            positions,
            heights,
            bottom=bottom,
            color=colors[choice],
            label=choice,
            width=0.8,
        )
        bottom += heights
    ax.set_xticks(positions, [str(v) for v in grid])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Share of replicates")
    ax.set_ylim(0, 1)


def _similarity_at_k_star(
    similarity: pd.DataFrame, datasets: pd.DataFrame
) -> pd.DataFrame:
    merged = similarity.merge(
        datasets[["study", "difficulty", "dataset", "k_star"]],
        on=["study", "difficulty", "dataset"],
    )
    rows = merged[merged["k"] == merged["k_star"]]
    per = similarity_summary(rows)
    pooled = rows.groupby("subsample_ratio", as_index=False).agg(
        ari_mean=("ari", "mean")
    )
    pooled["study"] = POOLED
    per["metric_name"] = "similarity"
    pooled["metric_name"] = "similarity"
    return pd.concat([per, pooled], ignore_index=True)


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
    studies = list(ablation.scenarios)
    study = ablation.study
    selection = view["selection"]
    sel = selection_summary(selection, x=x)
    recall = rare_recall_summary(
        view["at_k"],
        selection,
        x=x,
        difficulty=ablation.scales[scale].rho_arm.difficulties[-1],
    )
    at_star = curve_at_k_star(view["curves"], view["datasets"], x=x)
    similarity = _similarity_at_k_star(view["similarity"], view["datasets"])
    oracle = float(view["datasets"]["oracle_ari"].mean())

    with theme_context():
        fig, axes = plt.subplots(3, 3, figsize=(12.6, 11.4))
        ax = axes[0, 0]
        _lines_by_study(
            ax,
            sel,
            x=x,
            y="recovery",
            metric=STAB,
            studies=studies,
            lo="recovery_lo",
            hi="recovery_hi",
        )
        _title(ax, "A", "k* recovery, stability 1SE")
        ax.set_ylabel("Recovery rate")
        ax = axes[0, 1]
        _lines_by_study(
            ax,
            sel,
            x=x,
            y="recovery",
            metric=GEN,
            studies=studies,
            lo="recovery_lo",
            hi="recovery_hi",
        )
        _title(ax, "B", "k* recovery, generalizability 1SE")
        ax = axes[0, 2]
        _pooled_by_metric(ax, sel, x=x, y="bias_mean", metrics=HEADLINE_METRICS)
        ax.axhline(0.0, color=FALLBACK_COLOR, linewidth=REFERENCE_LINEWIDTH)
        _title(ax, "C", "Bias of the selected k")
        ax.set_ylabel("Mean of k-hat minus k*")
        ax = axes[1, 0]
        _pooled_by_metric(
            ax, sel, x=x, y="ari_mean", metrics=HEADLINE_METRICS, sem="ari_sem"
        )
        ax.axhline(
            oracle,
            color=metric_color("baseline_oracle"),
            linestyle="--",
            linewidth=REFERENCE_LINEWIDTH,
            label="Baseline (Oracle k*)",
        )
        _title(ax, "D", "ARI at the selected k")
        ax.set_ylabel("ARI to truth")
        ax = axes[1, 1]
        _pooled_by_metric(
            ax,
            recall,
            x=x,
            y="recall_selected",
            metrics=HEADLINE_METRICS,
            suffix=", at k-hat",
        )
        _pooled_by_metric(
            ax,
            recall,
            x=x,
            y="recall_k_star",
            metrics=HEADLINE_METRICS,
            linestyle=":",
            suffix=", at k*",
        )
        _title(ax, "E", "Rare-cluster recall, hard setting")
        ax.set_ylabel("Recall of the smallest cluster")
        ax = axes[1, 2]
        _pooled_by_metric(
            ax, at_star, x=x, y="value_mean", metrics=HEADLINE_METRICS, sem="value_sem"
        )
        _title(ax, "F", "Score at k*")
        ax.set_ylabel("ARI")
        ax = axes[2, 0]
        _lines_by_study(
            ax, similarity, x=x, y="ari_mean", metric="similarity", studies=studies
        )
        _title(ax, "G", "Subsample versus full-data clustering at k*")
        ax.set_ylabel("ARI to full-data fit")
        ax = axes[2, 1]
        _study_shares(
            ax,
            selection,
            x=x,
            grid=list(ablation.rho_grid),
            study=study,
            metric=GEN,
            x_label=RHO_LABEL,
        )
        _title(
            ax,
            "H",
            f"{STUDY_TITLES.get(study, study)}: selection, generalizability 1SE",
        )
        ax = axes[2, 2]
        _study_ari(ax, selection, x=x, study=study, metrics=HEADLINE_METRICS)
        # The manuscript reports one k for this study; read it from the
        # registry rather than restating it, and skip the similarity lines
        # if the study reports none.
        reported_k = STUDIES[study].reported_k
        if reported_k is not None:
            klein_sim = similarity_summary(
                view["similarity"][view["similarity"]["study"] == study]
            )
            klein_sim = klein_sim[klein_sim["k"] == reported_k]
            for estimator, part in klein_sim.groupby("estimator"):
                part = part.sort_values(x)
                ax.plot(
                    part[x],
                    part["ari_mean"],
                    marker="s",
                    markersize=3,
                    linestyle=":",
                    linewidth=REFERENCE_LINEWIDTH,
                    color=FOREGROUND_COLOR,
                    alpha=0.9 if estimator.startswith("Agglomerative") else 0.5,
                    label=f"{estimator}, subsample vs full at k={reported_k}",
                )
        _title(
            ax, "I", f"{STUDY_TITLES.get(study, study)}: selected labels and similarity"
        )
        ax.set_ylabel("ARI")

        for ax in axes.flat:
            if ax is not axes[2, 1]:
                _reference(ax, ablation.rho_default)
                ax.set_xlabel(RHO_LABEL)
            style_axes(ax)
        # The merged legend below the grid can run to many rows (every study,
        # every headline metric variant, the per-choice bars of panel H, and
        # the two Klein similarity lines), so tight_layout reserves a fixed
        # bottom strip for it rather than letting a tall legend grow upward
        # from just below the canvas into row 3's axes.
        fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))
        grouped_legend(fig, axes, y_offset=0.01, ncol=4)
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
    studies = [*ablation.scenarios, ablation.study]
    study = ablation.study
    selection = view["selection"]
    agreement = agreement_summary(selection, x=x)
    spread = spread_summary(view["curves"], x=x)
    at_star = curve_at_k_star(view["curves"], view["datasets"], x=x)
    sel = selection_summary(selection, x=x)
    b_default = ablation.b_default

    def _guide(ax, summary, y):
        # One over the square root of B, anchored at the default.
        rows = summary[(summary["study"] == POOLED) & (summary["metric_name"] == STAB)]
        anchor = rows[rows[x] == b_default][y]
        if anchor.empty or np.isnan(anchor.iloc[0]):
            return
        grid = np.array(ablation.b_grid, dtype=float)
        ax.plot(
            grid,
            float(anchor.iloc[0]) * np.sqrt(b_default / grid),
            linestyle="--",
            color=FALLBACK_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            label="1 / sqrt(B) guide",
        )

    with theme_context():
        fig, axes = plt.subplots(4, 2, figsize=(8.4, 15.6))
        ax = axes[0, 0]
        _lines_by_study(ax, agreement, x=x, y="agreement", metric=STAB, studies=studies)
        _title(ax, "A", "Replicate agreement, stability 1SE")
        ax.set_ylabel("Fraction of agreeing pairs")
        ax = axes[0, 1]
        _lines_by_study(ax, agreement, x=x, y="agreement", metric=GEN, studies=studies)
        _title(ax, "B", "Replicate agreement, generalizability 1SE")
        ax = axes[1, 0]
        _pooled_by_metric(ax, spread, x=x, y="spread", metrics=HEADLINE_METRICS)
        _guide(ax, spread, "spread")
        _title(ax, "C", "Curve spread across replicates")
        ax.set_ylabel("SD of the score over replicates")
        ax = axes[1, 1]
        _pooled_by_metric(ax, at_star, x=x, y="se_mean", metrics=HEADLINE_METRICS)
        _guide(ax, at_star, "se_mean")
        _title(ax, "D", "Reported standard error at k*")
        ax.set_ylabel("Mean standard error")
        ax = axes[2, 0]
        _pooled_by_metric(ax, sel, x=x, y="recovery", metrics=HEADLINE_METRICS)
        _title(ax, "E", "k* recovery")
        ax.set_ylabel("Recovery rate")
        ax = axes[2, 1]
        _pooled_by_metric(ax, sel, x=x, y="bias_mean", metrics=HEADLINE_METRICS)
        ax.axhline(0.0, color=FALLBACK_COLOR, linewidth=REFERENCE_LINEWIDTH)
        _title(ax, "F", "Bias of the selected k")
        ax.set_ylabel("Mean of k-hat minus k*")
        ax = axes[3, 0]
        _study_ari(ax, selection, x=x, study=study, metrics=HEADLINE_METRICS)
        _title(ax, "G", f"{STUDY_TITLES.get(study, study)}: selected labels")
        ax.set_ylabel("ARI to reference labels")
        ax = axes[3, 1]
        _study_shares(
            ax,
            selection,
            x=x,
            grid=list(ablation.b_grid),
            study=study,
            metric=GEN,
            x_label=B_LABEL,
        )
        _title(
            ax,
            "H",
            f"{STUDY_TITLES.get(study, study)}: selection, generalizability 1SE",
        )

        for ax in axes.flat:
            if ax is not axes[3, 1]:
                ax.set_xscale("log")
                _reference(ax, b_default)
                ax.set_xlabel(B_LABEL)
            style_axes(ax)
        # Same reasoning as figure_ablation_rho: reserve a fixed bottom strip
        # so a tall merged legend (studies, headline metrics, the guide line
        # and panel H's per-choice bars) cannot grow upward into row 4.
        fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
        grouped_legend(fig, axes, y_offset=0.01, ncol=4)
        if save:
            save_figure(
                fig,
                figure_path(
                    "si_fig_ablation_b.png", subdir=BENCHMARKING_DIR, out_dir=out_dir
                ),
            )
    return fig
