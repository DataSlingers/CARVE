"""Summarize a run into the manuscript's supplementary tables.

One summarizer covers every experiment. The two it replaces shared 105 of
122 code lines and differed only in whether they grouped by "difficulty" or
"stage"; with axis_label unified there is nothing left to differ about.

Tables are written to disk as .tex fragments under their manuscript names
rather than printed for copy-paste, so the supplementary tables stop drifting
from what the code produces.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._registry import METRIC_DISPLAY_NAMES

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


BASELINE_METRIC: str = "baseline_oracle"


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


def write_tables(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    *,
    caption: str,
    label: str,
    metrics: Sequence[str] | None = None,
) -> Path:
    """Summarize a run and write the .tex fragment under its manuscript name."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tex"
    path.write_text(
        render_grouped_tex(summarize(df, metrics=metrics), caption=caption, label=label)
    )
    return path
