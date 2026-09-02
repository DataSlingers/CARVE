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


def summarize(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    decimals: int = 3,
) -> pd.DataFrame:
    """One row per (axis_label, metric), computed from the selected k only."""
    selected = df.loc[df["is_selected"]].copy()

    if metrics is None:
        metrics = tuple(sorted(selected["metric_name"].unique()))
    present = [m for m in metrics if (selected["metric_name"] == m).any()]
    if not present:
        raise ValueError(
            "none of the requested metrics were found after filtering to selected rows"
        )

    rows = []
    for axis_label, by_label in selected.groupby("axis_label", sort=False):
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
            cells.append(f"{row['k_recovery']:.2f}")
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
