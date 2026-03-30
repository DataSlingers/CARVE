"""Summary / IO helpers and paper-style plotting setup.

Metric constants are imported from ``benchmarking_config`` — edit them there.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from benchmarking_config import PLOT_METRICS
from benchmarking_utils import _wilson_ci as wilson_ci


def set_paper_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# io – read all benchmark csvs
def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    # strings / objects
    x = series.astype(str).str.strip().str.lower()
    true_vals = {"true", "t", "1", "yes", "y"}
    false_vals = {"false", "f", "0", "no", "n"}
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[x.isin(true_vals)] = 1.0
    out[x.isin(false_vals)] = 0.0
    return out.fillna(0.0).astype(int).astype(bool)


def load_benchmark_csvs(
    results_dir: str | Path, pattern: str = "results_*.csv"
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"no files matching {pattern} in {results_dir.resolve()}"
        )

    dfs = []
    for p in paths:
        df = pd.read_csv(p)

        stem = p.stem
        benchmark = stem[len("results_") :] if stem.startswith("results_") else stem
        df["benchmark"] = benchmark
        df["source_file"] = p.name
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    for col in ["difficulty", "true_k", "k", "dataset_id"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    for col in ["is_optimal", "is_correct"]:
        if col in out.columns:
            out[col] = _coerce_bool(out[col])

    return out


def instance_key_cols(results_df: pd.DataFrame) -> list[str]:
    base = ["benchmark", "metric_name", "dataset_id", "true_k"]
    condition_cols = []
    for c in ["difficulty", "p", "n_samples"]:
        if c in results_df.columns and results_df[c].notna().any():
            condition_cols.append(c)
    return base + condition_cols


# overall k-recovery summary
def summarize_true_k_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    sel = results.loc[results["is_optimal"]].copy()

    key_cols = instance_key_cols(sel)
    if sel.duplicated(key_cols).any():
        sel = sel.sort_values(key_cols + ["k"]).drop_duplicates(key_cols, keep="first")

    sel["k_error"] = (sel["k"] - sel["true_k"]).astype(float)
    sel["abs_k_error"] = sel["k_error"].abs()
    sel["over"] = sel["k_error"] > 0
    sel["under"] = sel["k_error"] < 0

    g = sel.groupby("metric_name", as_index=False).agg(
        n=("is_correct", "size"),
        correct=("is_correct", "sum"),
        accuracy=("is_correct", "mean"),
        mae_k=("abs_k_error", "mean"),
        bias_k=("k_error", "mean"),
        over_rate=("over", "mean"),
        under_rate=("under", "mean"),
    )

    ci = [wilson_ci(int(r.correct), int(r.n)) for r in g.itertuples(index=False)]
    g["acc_ci_low"] = [c[0] for c in ci]
    g["acc_ci_high"] = [c[1] for c in ci]
    g["accuracy_ci"] = g.apply(
        lambda r: f"{r.accuracy:.3f} [{r.acc_ci_low:.3f}, {r.acc_ci_high:.3f}]", axis=1
    )

    return g.sort_values(["accuracy", "mae_k"], ascending=[False, True]).reset_index(
        drop=True
    )


def best_metric_per_benchmark(results: pd.DataFrame) -> pd.DataFrame:
    sel = results.loc[results["is_optimal"]].copy()

    key_cols = instance_key_cols(sel)
    if sel.duplicated(key_cols).any():
        sel = sel.sort_values(key_cols + ["k"]).drop_duplicates(key_cols, keep="first")

    sel["abs_k_error"] = (sel["k"] - sel["true_k"]).abs().astype(float)

    per = sel.groupby(["benchmark", "metric_name"], as_index=False).agg(
        n=("is_correct", "size"),
        accuracy=("is_correct", "mean"),
        mae_k=("abs_k_error", "mean"),
    )

    winners = (
        per.sort_values(
            ["benchmark", "accuracy", "mae_k", "metric_name"],
            ascending=[True, False, True, True],
        )
        .groupby("benchmark", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return winners


def filter_plot_metrics(results: pd.DataFrame) -> pd.DataFrame:
    return results.loc[results["metric_name"].isin(PLOT_METRICS)].copy()
