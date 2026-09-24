"""Summarize a run into the manuscript's supplementary tables.

One summarizer covers every experiment. The two it replaces shared 105 of
122 code lines and differed only in whether they grouped by "difficulty" or
"stage"; with axis_label unified there is nothing left to differ about.

Tables are written to disk as .tex fragments under their manuscript names
rather than printed for copy-paste, so the supplementary tables stop drifting
from what the code produces.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._registry import BASELINE_METRIC, METRIC_DISPLAY_NAMES, TABLE_ROW_GROUPS

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denominator
    half = (z / denominator) * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def summary_stats(values: pd.Series) -> dict[str, float]:
    """Mean, sd, median, and the standard quantiles of a numeric series."""
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    keys = ["mean", "sd", "median", "q05", "q25", "q75", "q95"]
    if values.empty:
        return dict.fromkeys(keys, float("nan"))
    quantiles = values.quantile(list(_QUANTILES))
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(quantiles.loc[0.50]),
        "q05": float(quantiles.loc[0.05]),
        "q25": float(quantiles.loc[0.25]),
        "q75": float(quantiles.loc[0.75]),
        "q95": float(quantiles.loc[0.95]),
    }


def _axis_label_order(df: pd.DataFrame) -> list[str]:
    """Axis labels in the order their axis was declared, not lexical order.

    Rows are checkpointed one file per (axis_label, seed) and read back by a
    lexical glob (``_artifacts.read_run``), so a table built straight off
    row order groups columns alphabetically -- "easy, hard, medium" instead
    of "easy, medium, hard". Each row already carries the numeric
    ``axis_value`` a Scenario's Axis assigns that label, and every Axis in
    the registry declares its values in the intended reading order, so
    sorting on that numeric column (never on the label string) reproduces
    the intended column order.
    """
    return (
        df[["axis_value", "axis_label"]]
        .drop_duplicates()
        .sort_values("axis_value")["axis_label"]
        .tolist()
    )


def _baseline_row(df: pd.DataFrame, axis_label: str) -> dict[str, Any]:
    """The Baseline (Oracle) row for one axis label.

    oracle_ari is a per-(axis_label, seed) constant -- the oracle estimator
    is fit once per cell, not once per metric -- so it is deduplicated by
    seed here rather than read off the is_selected-filtered rows the other
    metrics use. The oracle has no notion of a selected k, so k-recovery and
    the k-bias columns are left NaN rather than a misleading 0 or 1.
    """
    oracle = df.loc[df["axis_label"] == axis_label, ["seed", "oracle_ari"]]
    oracle = oracle.drop_duplicates(subset=["seed"])["oracle_ari"]
    stats = summary_stats(oracle)
    return {
        "axis_label": axis_label,
        "metric": BASELINE_METRIC,
        "display_name": METRIC_DISPLAY_NAMES.get(BASELINE_METRIC, BASELINE_METRIC),
        "n_datasets": int(oracle.shape[0]),
        "ari_mean": stats["mean"],
        "ari_sd": stats["sd"],
        "ari_median": stats["median"],
        "ari_q25": stats["q25"],
        "ari_q75": stats["q75"],
        "delta_mean": float("nan"),
        "delta_median": float("nan"),
        "k_recovery": float("nan"),
        "k_rec_lo95": float("nan"),
        "k_rec_hi95": float("nan"),
        "k_bias_median": float("nan"),
        "p_under": float("nan"),
        "p_over": float("nan"),
    }


def summarize(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    decimals: int = 3,
) -> pd.DataFrame:
    """One row per (axis_label, metric), computed from the selected k only.

    ``"baseline_oracle"`` in ``metrics`` is handled separately from every
    other name: it names a schema column (``oracle_ari``), not a value of
    ``metric_name``, so it is never a member of ``present`` and is emitted
    first for each axis label regardless of where it sorts among the
    others -- matching the published tables, whose first data row is always
    Baseline (Oracle).
    """
    selected = df.loc[df["is_selected"]].copy()

    if metrics is None:
        metrics = tuple(sorted(selected["metric_name"].unique()))
    include_baseline = BASELINE_METRIC in metrics
    present = [
        m
        for m in metrics
        if m != BASELINE_METRIC and (selected["metric_name"] == m).any()
    ]
    if not present and not include_baseline:
        raise ValueError(
            "none of the requested metrics were found after filtering to selected rows"
        )

    rows = []
    for axis_label in _axis_label_order(selected):
        by_label = selected.loc[selected["axis_label"] == axis_label]

        if include_baseline:
            rows.append(_baseline_row(df, axis_label))

        for metric in present:
            sub = by_label.loc[by_label["metric_name"] == metric]
            n = int(len(sub))
            if n == 0:
                continue

            ari = summary_stats(sub["ari_at_k"])
            delta = summary_stats(
                pd.to_numeric(sub["oracle_ari"], errors="coerce")
                - pd.to_numeric(sub["ari_at_k"], errors="coerce")
            )
            successes = int(sub["selects_true_k"].astype(bool).sum())
            low, high = wilson_ci(successes, n)
            k_bias = pd.to_numeric(sub["k"], errors="coerce") - pd.to_numeric(
                sub["k_star"], errors="coerce"
            )

            rows.append(
                {
                    "axis_label": axis_label,
                    "metric": metric,
                    "display_name": METRIC_DISPLAY_NAMES.get(metric, metric),
                    "n_datasets": n,
                    "ari_mean": ari["mean"],
                    "ari_sd": ari["sd"],
                    "ari_median": ari["median"],
                    "ari_q25": ari["q25"],
                    "ari_q75": ari["q75"],
                    "delta_mean": delta["mean"],
                    "delta_median": delta["median"],
                    "k_recovery": successes / n,
                    "k_rec_lo95": low,
                    "k_rec_hi95": high,
                    "k_bias_median": float(k_bias.median()),
                    "p_under": float((k_bias < 0).mean()),
                    "p_over": float((k_bias > 0).mean()),
                }
            )

    out = pd.DataFrame(rows)
    numeric = out.select_dtypes(include=[float]).columns
    out[numeric] = out[numeric].round(decimals)
    return out


def _tex_escape(text: str) -> str:
    """Escape the LaTeX special characters that appear in captions.

    Maps each input character through the table in a single pass over the
    original string, rather than a sequence of ``str.replace`` calls. A
    sequential-replace implementation would rescan its own output: escaping
    a backslash inserts the literal braces in ``\\textbackslash{}``, and a
    later rule for ``{``/``}`` would then re-escape those, corrupting the
    result. Mapping character-by-character means nothing already emitted is
    ever looked at again.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def render_grouped_tex(
    summary: pd.DataFrame,
    *,
    caption: str,
    label: str,
    decimals: int = 3,
) -> str:
    """Render the summary as a table grouped by axis label.

    Column groups are the axis labels; rows are metrics. This reproduces the
    shape of the tables already in the manuscript.
    """
    axis_labels = list(dict.fromkeys(summary["axis_label"]))
    metrics = list(dict.fromkeys(summary["metric"]))

    column_spec = "l" + "cc" * len(axis_labels)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
    ]

    header = ["Method"]
    for axis_label in axis_labels:
        header.append(f"\\multicolumn{{2}}{{c}}{{{_tex_escape(str(axis_label))}}}")
    lines.append(" & ".join(header) + r" \\")
    lines.append("Method" + " & ARI & $k$-rec" * len(axis_labels) + r" \\")
    lines.append(r"\hline")

    for metric in metrics:
        cells = [_tex_escape(METRIC_DISPLAY_NAMES.get(metric, metric))]
        for axis_label in axis_labels:
            match = summary[
                (summary["metric"] == metric) & (summary["axis_label"] == axis_label)
            ]
            if match.empty:
                cells.extend(["", ""])
                continue
            row = match.iloc[0]
            cells.append(
                f"{row['ari_mean']:.{decimals}f} ({row['ari_sd']:.{decimals}f})"
            )
            # Baseline (Oracle) has no selected k, so k_recovery is NaN --
            # rendered as a blank cell rather than the literal text "nan",
            # matching the published tables' blank Baseline k-recovery cell.
            cells.append(
                "" if pd.isna(row["k_recovery"]) else f"{row['k_recovery']:.2f}"
            )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _ranked(values: Sequence[float]) -> tuple[float | None, float | None]:
    """The best and second-best finite values in a column.

    Second-best is None when the best is tied, matching the committed S2
    table: three cells at 0.932 are all bold and nothing in that column is
    underlined. Marking a runner-up below a tied first place would say the
    column has a clear ordering when it does not.
    """
    finite = sorted({float(v) for v in values if pd.notna(v)}, reverse=True)
    if not finite:
        return None, None
    best = finite[0]
    if sum(1 for v in values if pd.notna(v) and float(v) == best) > 1:
        return best, None
    return best, (finite[1] if len(finite) > 1 else None)


def _marked(
    value: float, best: float | None, second: float | None, decimals: int
) -> str:
    """One cell, bold if best in its column and underlined if second."""
    if pd.isna(value):
        return ""
    text = f"{value:.{decimals}f}"
    if best is not None and float(value) == best:
        return rf"\textbf{{{text}}}"
    if second is not None and float(value) == second:
        return rf"\underline{{{text}}}"
    return text


# The committed S2 table and the manuscript prose write these two CVI names
# with an en dash; METRIC_DISPLAY_NAMES keeps the plain hyphen because
# figure legends read it too. Table-only, so it lives here rather than
# there.
_TABLE_EN_DASH_NAMES: dict[str, str] = {
    "davies_bouldin": "Davies--Bouldin",
    "calinski_harabasz": "Calinski--Harabasz",
}


def _table_display_name(metric: str) -> str:
    """The row label render_paired_tex writes for a metric.

    _tex_escape has no rule for "-", so the en dash survives escaping
    unchanged.
    """
    return _TABLE_EN_DASH_NAMES.get(metric, METRIC_DISPLAY_NAMES.get(metric, metric))


def _sub_table(
    summary: pd.DataFrame,
    groups: Sequence[Sequence[str]],
    axis_labels: Sequence[str],
    column: str,
    decimals: int,
    *,
    headers: Sequence[str] | None = None,
    trailing_percent: bool = False,
) -> list[str]:
    """One of the pair: rows by metric group, columns by axis label.

    headers, one per axis label, replace the labels in the header row; None
    writes the labels themselves.

    trailing_percent closes the tabular with ``\\end{tabular}%`` instead of
    a bare ``\\end{tabular}``, matching the committed table: the ``%``
    suppresses the space LaTeX would otherwise put before the following
    ``\\quad``. Only the first (left) sub-table of a pair uses it.
    """
    headers = list(axis_labels) if headers is None else list(headers)
    lines = [
        f"\\begin{{tabular}}{{l{'c' * len(axis_labels)}}}",
        r"\toprule",
        "Metric & " + " & ".join(_tex_escape(str(h)) for h in headers) + r" \\",
        r"\midrule",
    ]

    # Ranking is per column, over the rows this call actually renders --
    # the metrics in `groups`, i.e. the declared groups intersected with
    # what is present in `summary` -- and excludes the oracle, which is the
    # reference the rest are measured against rather than a competitor.
    # summary can carry metrics outside the groups (metrics= wasn't
    # restricted to the published set), and ranking over those too let an
    # unrendered value outscore every rendered one, which then rendered as
    # unmarked instead of bold.
    rendered_metrics = {metric for group in groups for metric in group}
    ranks: dict[str, tuple[float | None, float | None]] = {}
    for axis_label in axis_labels:
        candidates = summary[
            (summary["axis_label"] == axis_label)
            & (summary["metric"] != BASELINE_METRIC)
            & (summary["metric"].isin(rendered_metrics))
        ][column]
        ranks[axis_label] = _ranked(candidates.tolist())

    for index, group in enumerate(groups):
        if index:
            lines.append(r"\midrule")
        for metric in group:
            match = summary[summary["metric"] == metric]
            if match.empty:
                continue
            cells = [_tex_escape(_table_display_name(metric))]
            for axis_label in axis_labels:
                row = match[match["axis_label"] == axis_label]
                if row.empty:
                    cells.append("")
                    continue
                value = row.iloc[0][column]
                if metric == BASELINE_METRIC:
                    cells.append("" if pd.isna(value) else f"{value:.{decimals}f}")
                else:
                    best, second = ranks[axis_label]
                    cells.append(_marked(value, best, second, decimals))
            lines.append(" & ".join(cells) + r" \\")

    end_tabular = r"\end{tabular}%" if trailing_percent else r"\end{tabular}"
    lines.extend([r"\bottomrule", end_tabular])
    return lines


def _comment_safe(text: str) -> str:
    """Text safe to place after a single "%" on one line of a .tex file.

    A literal newline would end the LaTeX comment right there, leaving
    whatever follows as real (un-commented) file content rather than part
    of the comment, so newlines become spaces. A percent sign is stripped
    too, defensively -- TABLE_CAPTIONS carries none today, but nothing here
    should depend on that staying true.
    """
    single_line = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return single_line.replace("%", "")


def render_paired_tex(
    summary: pd.DataFrame,
    *,
    caption: str,
    label: str,
    decimals: int = 3,
    groups: Sequence[Sequence[str]] = TABLE_ROW_GROUPS,
    column_headers: Mapping[str, str] | None = None,
) -> str:
    """The manuscript's shape: mean ARI and k-recovery side by side.

    Two tabulars inside one resizebox, separated by \\quad, the left giving
    mean ARI at the selected k and the right the proportion of datasets
    where that k was k*. Rows follow groups, TABLE_ROW_GROUPS unless a
    table family declares its own, with a rule between groups; bold marks
    the best value in a column and underline the second-best.

    column_headers maps an axis label to the text its column is headed
    with. The scaling tables head theirs with the swept values (1000, 5500,
    10000) rather than start/middle/end. A label it does not map keeps its
    own text.

    This replaces render_grouped_tex's single flat table, which merged both
    quantities into one unranked grid. The numbers were right; the layout
    was not, and every fragment had to be reshaped by hand before it could
    go into the manuscript.

    The fragment is the float only -- no \\caption, no \\label. The
    manuscript never puts either inside a supplementary table: each is an
    author-written \\paragraph*{Sn Table.}, \\label{Sn_Table}, and caption
    prose above a caption-free \\begin{table}[H]. caption and label are kept
    as a single leading LaTeX comment instead, so a generated file can still
    be identified without adding a second label or an in-float caption a
    dropped-in fragment would otherwise carry.
    """
    axis_labels = list(dict.fromkeys(summary["axis_label"]))
    present = set(summary["metric"])
    groups = [[m for m in group if m in present] for group in groups]
    groups = [group for group in groups if group]
    headers = [(column_headers or {}).get(a, a) for a in axis_labels]

    lines = [
        f"% {label}: {_comment_safe(caption)}",
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
    ]
    lines += _sub_table(
        summary,
        groups,
        axis_labels,
        "ari_mean",
        decimals,
        headers=headers,
        trailing_percent=True,
    )
    lines.append(r"\quad")
    lines += _sub_table(
        summary, groups, axis_labels, "k_recovery", decimals, headers=headers
    )
    lines.extend([r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_tables(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    *,
    caption: str,
    label: str,
    metrics: Sequence[str] | None = None,
    groups: Sequence[Sequence[str]] = TABLE_ROW_GROUPS,
    column_headers: Mapping[str, str] | None = None,
) -> Path:
    """Summarize a run and write the .tex fragment under its manuscript name.

    groups and column_headers pass through to render_paired_tex.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tex"
    path.write_text(
        render_paired_tex(
            summarize(df, metrics=metrics),
            caption=caption,
            label=label,
            groups=groups,
            column_headers=column_headers,
        )
    )
    return path


# --- The rho/B ablation table -----------------------------------------------
ESTIMATOR_SHORT_NAMES: dict[str, str] = {
    "AgglomerativeClustering": "Ward",
    "SpectralClustering": "Spectral",
    "KMeans": "KMeans",
}

_ARM_HEADINGS: dict[str, str] = {"rho": r"$\rho$", "b": "$B$"}


def _fmt(value: float, decimals: int) -> str:
    return "" if pd.isna(value) else f"{value:.{decimals}f}"


def _modal(text: object, share: float) -> str:
    if pd.isna(share) or not isinstance(text, str):
        return ""
    estimator, k = text.split(", ")
    return f"{ESTIMATOR_SHORT_NAMES.get(estimator, estimator)}, ${k}$ ({share:.2f})"


def render_ablation_tex(
    rows_by_arm: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str] | None = None,
    caption: str,
    label: str,
    study_title: str,
    decimals: int = 3,
) -> str:
    """Two sub-tables, one row per rho value and one per B value.

    Per headline selector: pooled k* recovery with its Wilson interval, the
    pooled mean ARI of the selected labels, in the B sub-table the pooled
    replicate agreement, and the study's modal selection with its share. Metrics
    default to the headline selectors.
    """
    if metrics is None:
        from ._ablation_summary import HEADLINE_METRICS

        metrics = HEADLINE_METRICS
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\label{{{label}}}",
    ]
    for arm in ("rho", "b"):
        rows = rows_by_arm[arm]
        with_agreement = arm == "b"
        per_metric = 4 if with_agreement else 3
        lines += [
            f"\\begin{{tabular}}{{l{'c' * per_metric * len(metrics)}}}",
            r"\hline",
        ]
        head = [""] + [
            f"\\multicolumn{{{per_metric}}}{{c}}{{{_tex_escape(METRIC_DISPLAY_NAMES.get(m, m))}}}"
            for m in metrics
        ]
        lines.append(" & ".join(head) + r" \\")
        sub = [_ARM_HEADINGS[arm]]
        for _ in metrics:
            sub += ["$k$-rec [95\\% CI]", "ARI"]
            if with_agreement:
                sub.append("Agreement")
            sub.append(f"{_tex_escape(study_title)} modal (share)")
        lines += [" & ".join(sub) + r" \\", r"\hline"]
        for setting in sorted(rows["setting"].unique()):
            cells = [f"{setting:g}"]
            for metric in metrics:
                match = rows[
                    (rows["setting"] == setting) & (rows["metric_name"] == metric)
                ]
                if match.empty:
                    cells += [""] * per_metric
                    continue
                row = match.iloc[0]
                cells.append(
                    f"{_fmt(row['recovery'], 2)} [{_fmt(row['recovery_lo'], 2)}, "
                    f"{_fmt(row['recovery_hi'], 2)}]"
                )
                cells.append(_fmt(row["ari_mean"], decimals))
                if with_agreement:
                    cells.append(_fmt(row["agreement"], 2))
                cells.append(_modal(row["study_modal"], row["study_share"]))
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}"]
        if arm == "rho":
            lines.append(r"\vspace{1em}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_ablation_table(
    frames: Mapping[str, pd.DataFrame],
    *,
    ablation,
    scale: str,
    out_dir: Path,
    name: str = "si_table_ablation",
    caption: str = "Sensitivity of CARVE's selections to the subsampling proportion and the resample count.",
    label: str = "tab:ablation_rho_b",
) -> Path:
    """Summarize a run's two arms and write the .tex fragment."""
    # Imported here: _ablation_summary imports wilson_ci from this module.
    from ._ablation_cells import arm_view
    from ._ablation_summary import table_rows

    rows = {
        "rho": table_rows(
            arm_view(frames, ablation=ablation, scale=scale, arm="rho"),
            x="subsample_ratio",
            study=ablation.study,
        ),
        "b": table_rows(
            arm_view(frames, ablation=ablation, scale=scale, arm="b"),
            x="n_resamples",
            study=ablation.study,
        ),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tex"
    path.write_text(
        render_ablation_tex(
            rows, caption=caption, label=label, study_title=ablation.study.capitalize()
        )
    )
    return path


# --- The Klein M3C comparison table -----------------------------------------------
#: The default caption. Stated here rather than at the call site so the
#: notebook and any later caller cannot caption the same table differently.
M3C_CAPTION: str = (
    "Selections and their agreement with the reported labels on the Klein "
    "case study, for CARVE, the four classical validation indices, and M3C "
    "run at its published defaults."
)


def render_m3c_tex(
    ari_df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    decimals: int = 3,
) -> str:
    """One row per method: its selected k and its ARI against the reported labels.

    Runtime is deliberately not a column. fit_or_load_carve caches state
    rather than timings, so a runtime column would be measured under
    different conditions for CARVE than for M3C; the two runtimes are
    reported in prose instead.
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & $k$ & ARI \\",
        r"\midrule",
    ]
    for _, row in ari_df.iterrows():
        method = str(row["method"])
        # Method names may already carry math, as "M3C (at $k=4$)" does, so
        # only names with no math are escaped.
        rendered = method if "$" in method else _tex_escape(method)
        lines.append(
            f"{rendered} & ${int(row['k'])}$ & {_fmt(row['ari'], decimals)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_m3c_table(
    ari_df: pd.DataFrame,
    *,
    out_dir: Path,
    name: str = "si_table_m3c",
    caption: str = M3C_CAPTION,
    label: str = "tab:klein_m3c",
    decimals: int = 3,
) -> Path:
    """Write the comparison table as .tex, with the underlying frame as .csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tex"
    path.write_text(
        render_m3c_tex(ari_df, caption=caption, label=label, decimals=decimals)
    )
    ari_df.to_csv(out_dir / f"{name}.csv", index=False)
    return path
