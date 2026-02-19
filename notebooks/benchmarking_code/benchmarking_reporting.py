import numpy as np
import pandas as pd

from benchmarking_utils import _wilson_ci, _summary_stats


# ---------------------------------------------------------------------------
# Private helper: summarize a single (already-filtered) slice of results
# ---------------------------------------------------------------------------

def _summarize_single_group(
    df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] | None = None,
    decimals: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Compute summary tables for a *single* difficulty group.

    Returns dict with keys:
      - overall_numeric
      - overall_formatted
      - by_true_k_numeric
    """

    # identify dataset id cols (for deduping baseline / counts)
    id_cols = []
    for c in ["difficulty_level", "true_k", "dataset_iteration", "dataset_id"]:
        if c in df.columns:
            id_cols.append(c)

    # keep only selected-k rows per (dataset, metric)
    df_sel = df.loc[df["is_optimal"] == True].copy()

    # restrict metrics if requested
    if metrics is None:
        metrics = tuple(sorted(df_sel["metric_name"].unique()))
    present = [m for m in metrics if (df_sel["metric_name"] == m).any()]
    if not present:
        raise ValueError(
            "none of the requested metrics were found after filtering to is_optimal rows"
        )

    # baseline (oracle) distribution: dedupe so it is one per dataset instance
    if id_cols:
        base = df.drop_duplicates(id_cols)[["baseline_ari"]].copy()
    else:
        base = df[["baseline_ari"]].drop_duplicates().copy()

    # ----- overall per-metric summary -----
    rows = []
    for m in present:
        sub = df_sel.loc[df_sel["metric_name"] == m].copy()
        n = int(len(sub))

        # ARI(selected)
        ari_stats = _summary_stats(sub["metric_ari"])

        # delta to oracle
        delta = (
            pd.to_numeric(sub["baseline_ari"], errors="coerce")
            - pd.to_numeric(sub["metric_ari"], errors="coerce")
        )
        delta_stats = _summary_stats(delta)

        # k recovery (as stored: is_correct)
        k_success = int(
            pd.to_numeric(sub["is_correct"], errors="coerce").fillna(0).astype(int).sum()
        )
        k_lo, k_hi = _wilson_ci(k_success, n)

        # k bias (k_hat - k_true)
        k_hat = pd.to_numeric(sub["k"], errors="coerce")
        k_true = pd.to_numeric(sub["true_k"], errors="coerce")
        kbias = k_hat - k_true
        kbias_stats = _summary_stats(kbias)

        under = float((kbias < 0).mean()) if n else np.nan
        over = float((kbias > 0).mean()) if n else np.nan

        rows.append({
            "metric": m,
            "n": n,
            "ari_mean": ari_stats["mean"],
            "ari_sd": ari_stats["sd"],
            "ari_median": ari_stats["median"],
            "ari_q25": ari_stats["q25"],
            "ari_q75": ari_stats["q75"],
            "ari_q05": ari_stats["q05"],
            "ari_q95": ari_stats["q95"],
            "delta_mean": delta_stats["mean"],
            "delta_sd": delta_stats["sd"],
            "delta_median": delta_stats["median"],
            "delta_q25": delta_stats["q25"],
            "delta_q75": delta_stats["q75"],
            "delta_q05": delta_stats["q05"],
            "delta_q95": delta_stats["q95"],
            "k_recovery": (k_success / n) if n else np.nan,
            "k_rec_lo95": k_lo,
            "k_rec_hi95": k_hi,
            "k_rec_num": float(k_success),
            "k_rec_den": float(n),
            "kbias_mean": kbias_stats["mean"],
            "kbias_median": kbias_stats["median"],
            "p_under": under,
            "p_over": over,
        })

    overall = pd.DataFrame(rows)

    # baseline summary row (context)
    base_stats = _summary_stats(base["baseline_ari"])
    baseline_row = pd.DataFrame([{
        "metric": "baseline_oracle",
        "n": int(len(base)),
        "ari_mean": base_stats["mean"],
        "ari_sd": base_stats["sd"],
        "ari_median": base_stats["median"],
        "ari_q25": base_stats["q25"],
        "ari_q75": base_stats["q75"],
        "ari_q05": base_stats["q05"],
        "ari_q95": base_stats["q95"],
        "delta_mean": np.nan,
        "delta_sd": np.nan,
        "delta_median": np.nan,
        "delta_q25": np.nan,
        "delta_q75": np.nan,
        "delta_q05": np.nan,
        "delta_q95": np.nan,
        "k_recovery": np.nan,
        "k_rec_lo95": np.nan,
        "k_rec_hi95": np.nan,
        "k_rec_num": np.nan,
        "k_rec_den": np.nan,
        "kbias_mean": np.nan,
        "kbias_median": np.nan,
        "p_under": np.nan,
        "p_over": np.nan,
    }])

    overall = pd.concat([baseline_row, overall], ignore_index=True)

    # ----- ordering: keep baseline first, then sort methods by ARI mean -----
    if "ari_mean" in overall.columns:
        base_mask = overall["metric"].eq("baseline_oracle")
        overall_base = overall.loc[base_mask]
        overall_methods = overall.loc[~base_mask].sort_values("ari_mean", ascending=False)
        overall = pd.concat([overall_base, overall_methods], ignore_index=True)

    # ----- by true_k (SI-friendly) -----
    def _group_summary_by_k(df_in: pd.DataFrame, by: str) -> pd.DataFrame:
        if by not in df_in.columns:
            return pd.DataFrame()

        out_rows = []
        for (m, gval), sub in df_in.groupby(["metric_name", by], dropna=False):
            n = int(len(sub))

            ari_stats = _summary_stats(sub["metric_ari"])
            delta = (
                pd.to_numeric(sub["baseline_ari"], errors="coerce")
                - pd.to_numeric(sub["metric_ari"], errors="coerce")
            )
            delta_stats = _summary_stats(delta)

            k_success = int(
                pd.to_numeric(sub["is_correct"], errors="coerce").fillna(0).astype(int).sum()
            )
            k_lo, k_hi = _wilson_ci(k_success, n)

            out_rows.append({
                "metric": m,
                by: gval,
                "n": n,
                "ari_mean": ari_stats["mean"],
                "ari_median": ari_stats["median"],
                "ari_q25": ari_stats["q25"],
                "ari_q75": ari_stats["q75"],
                "ari_q05": ari_stats["q05"],
                "ari_q95": ari_stats["q95"],
                "delta_mean": delta_stats["mean"],
                "delta_median": delta_stats["median"],
                "delta_q25": delta_stats["q25"],
                "delta_q75": delta_stats["q75"],
                "k_recovery": (k_success / n) if n else np.nan,
                "k_rec_lo95": k_lo,
                "k_rec_hi95": k_hi,
                "k_rec_num": float(k_success),
                "k_rec_den": float(n),
            })

        return pd.DataFrame(out_rows).sort_values(["metric", by])

    by_true_k = _group_summary_by_k(df_sel, "true_k")

    # ----- formatted view (manuscript-ready) -----
    def _fmt_pm(mean: float, sd: float) -> str:
        if not np.isfinite(mean) or not np.isfinite(sd):
            return ""
        return f"{mean:.{decimals}f} ({sd:.{decimals}f})"

    def _fmt_med_iqr(med: float, q25: float, q75: float) -> str:
        if not all(np.isfinite([med, q25, q75])):
            return ""
        return f"{med:.{decimals}f} [{q25:.{decimals}f}, {q75:.{decimals}f}]"

    def _fmt_prop(p: float, lo: float, hi: float, num: float, den: float) -> str:
        if not all(np.isfinite([p, lo, hi, num, den])) or den <= 0:
            return ""
        return f"{p:.{decimals}f} ({lo:.{decimals}f}, {hi:.{decimals}f}); {int(num)}/{int(den)}"

    formatted = overall.copy()
    formatted["ARI mean (sd)"] = [
        _fmt_pm(m, s) for m, s in zip(formatted["ari_mean"], formatted["ari_sd"])
    ]
    formatted["ARI median [q25, q75]"] = [
        _fmt_med_iqr(med, q25, q75)
        for med, q25, q75 in zip(
            formatted["ari_median"], formatted["ari_q25"], formatted["ari_q75"]
        )
    ]
    formatted["\u0394 mean (sd)"] = [
        _fmt_pm(m, s) for m, s in zip(formatted["delta_mean"], formatted["delta_sd"])
    ]
    formatted["k recovery (95% CI); num/den"] = [
        _fmt_prop(p, lo, hi, num, den)
        for p, lo, hi, num, den in zip(
            formatted["k_recovery"],
            formatted["k_rec_lo95"],
            formatted["k_rec_hi95"],
            formatted["k_rec_num"],
            formatted["k_rec_den"],
        )
    ]

    formatted = formatted[
        [
            "metric",
            "n",
            "ARI mean (sd)",
            "ARI median [q25, q75]",
            "\u0394 mean (sd)",
            "k recovery (95% CI); num/den",
            "kbias_mean",
            "kbias_median",
            "p_under",
            "p_over",
        ]
    ]

    for c in ["kbias_mean", "kbias_median", "p_under", "p_over"]:
        formatted[c] = formatted[c].apply(
            lambda v: "" if not np.isfinite(v) else f"{float(v):.{decimals}f}"
        )

    return {
        "overall_numeric": overall,
        "overall_formatted": formatted,
        "by_true_k_numeric": by_true_k,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_difficulty_col(df: pd.DataFrame) -> str:
    """Return the name of the difficulty-level column, handling both schemas."""
    if "difficulty_level" in df.columns:
        return "difficulty_level"
    if "axis_value" in df.columns and "axis_name" in df.columns:
        if (df["axis_name"] == "difficulty_level").any():
            return "axis_value"
    raise ValueError(
        "Cannot find a difficulty column. Expected 'difficulty_level' or "
        "'axis_value' (with axis_name='difficulty_level') in the DataFrame."
    )


def summarize_benchmark_regime(
    results_df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] | None = None,
    decimals: int = 3,
    difficulty_anchors: dict[int, str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Return paper-ready summary tables split by S/N difficulty level.

    Each difficulty anchor (easy / medium / hard) is summarized independently;
    results are *never* aggregated across difficulty levels.

    Parameters
    ----------
    results_df : DataFrame
        Must contain: metric_name, is_optimal, metric_ari, baseline_ari,
        is_correct, k, true_k, and a difficulty column (``difficulty_level``
        or ``axis_value`` with ``axis_name == 'difficulty_level'``).
    metrics : tuple of str, optional
        Restrict to these metric names.  ``None`` -> use all.
    decimals : int
        Rounding precision for formatted columns.
    difficulty_anchors : dict mapping int -> str, optional
        Which integer difficulty levels to keep and their human-readable
        labels.  Defaults to ``{0: "easy", 5: "medium", 9: "hard"}``.

    Returns
    -------
    dict[str, dict]
        Keyed by difficulty label (e.g. ``"easy"``, ``"medium"``, ``"hard"``).
        Each value is a dict with:
          - ``"overall_numeric"``   - raw numeric DataFrame
          - ``"overall_formatted"`` - manuscript-ready string table
          - ``"by_true_k_numeric"`` - stratified by ``true_k``

        Compatible with ``render_tex_table(out["easy"])`` etc.
    """
    needed = {
        "metric_name", "is_optimal", "metric_ari",
        "baseline_ari", "is_correct", "k", "true_k",
    }
    missing = sorted(c for c in needed if c not in results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {missing}")

    if difficulty_anchors is None:
        difficulty_anchors = {0: "easy", 5: "medium", 9: "hard"}

    diff_col = _resolve_difficulty_col(results_df)
    df = results_df.copy()

    result: dict[str, dict[str, pd.DataFrame]] = {}
    for level_int, label in sorted(difficulty_anchors.items()):
        group = df.loc[df[diff_col] == level_int]
        if group.empty:
            continue
        result[label] = _summarize_single_group(
            group,
            metrics=metrics,
            decimals=decimals,
        )

    if not result:
        available = sorted(df[diff_col].dropna().unique())
        raise ValueError(
            f"No rows matched the requested difficulty anchors "
            f"{list(difficulty_anchors.keys())}. "
            f"Available values in '{diff_col}': {available}"
        )

    return result


def render_tex_table(out: dict) -> str:
    """Return a LaTeX table string from a summarize_benchmark_regime output dict."""
    numeric = out["overall_numeric"].copy()
    formatted = out["overall_formatted"].copy()

    # Columns to rank: mapping from formatted col -> (numeric col, "high"/"low"/"abs")
    rank_spec = {
        "ARI mean (sd)": ("ari_mean", "high"),
        "ARI median [q25, q75]": ("ari_median", "high"),
        "\u0394 mean (sd)": ("delta_mean", "low"),
        "k recovery (95% CI); num/den": ("k_recovery", "high"),
        "kbias_mean": ("kbias_mean", "abs"),
        "kbias_median": ("kbias_median", "abs"),
        "p_under": ("p_under", "low"),
        "p_over": ("p_over", "low"),
    }

    # Mask: only rank non-baseline rows
    is_metric = numeric["metric"] != "baseline_oracle"

    for fmt_col, (num_col, direction) in rank_spec.items():
        if fmt_col not in formatted.columns or num_col not in numeric.columns:
            continue

        vals = numeric.loc[is_metric, num_col].copy()

        # Skip columns where fewer than 2 finite values exist
        finite_mask = vals.apply(lambda v: np.isfinite(v) if np.isscalar(v) else False)
        if finite_mask.sum() < 2:
            continue

        if direction == "abs":
            rank_vals = vals.abs()
            ascending = True
        elif direction == "low":
            rank_vals = vals
            ascending = True
        else:  # "high"
            rank_vals = vals
            ascending = False

        ranked = rank_vals[finite_mask].rank(ascending=ascending, method="min")

        best_idx = ranked[ranked == 1].index
        second_idx = ranked[ranked == 2].index

        for idx in best_idx:
            cell = str(formatted.at[idx, fmt_col])
            if cell:
                formatted.at[idx, fmt_col] = r"\textbf{" + cell + "}"
        for idx in second_idx:
            cell = str(formatted.at[idx, fmt_col])
            if cell:
                formatted.at[idx, fmt_col] = r"\underline{" + cell + "}"

    # --- TeX escaping ---
    df_tex = formatted.copy()
    df_tex["metric"] = df_tex["metric"].astype(str).str.replace("_", r"\_", regex=False)
    df_tex = df_tex.rename(columns={"\u0394 mean (sd)": r"$\Delta$ mean (sd)"})

    def escape_header(s: str) -> str:
        if ("$" in s) or ("\\" in s):
            return s
        return (
            s.replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#")
            .replace("{", r"\{")
            .replace("}", r"\}")
        )

    df_tex.columns = [escape_header(str(c)) for c in df_tex.columns]

    ncols = df_tex.shape[1]
    colfmt = "l" + "r" + "l" * (ncols - 2)

    return df_tex.to_latex(index=False, escape=False, column_format=colfmt)
