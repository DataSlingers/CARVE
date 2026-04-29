import numpy as np
import pandas as pd

from benchmarking_config import (
    SELECTED_CARVE,
    CLASSICAL,
    METRIC_DISPLAY_NAMES,
    EXCLUDE_FROM_TABLES as EXCLUDE,
)
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
        delta = pd.to_numeric(sub["baseline_ari"], errors="coerce") - pd.to_numeric(
            sub["metric_ari"], errors="coerce"
        )
        delta_stats = _summary_stats(delta)

        # k recovery (as stored: is_correct)
        k_success = int(
            pd.to_numeric(sub["is_correct"], errors="coerce")
            .fillna(0)
            .astype(int)
            .sum()
        )
        k_lo, k_hi = _wilson_ci(k_success, n)

        # k bias (k_hat - k_true)
        k_hat = pd.to_numeric(sub["k"], errors="coerce")
        k_true = pd.to_numeric(sub["true_k"], errors="coerce")
        kbias = k_hat - k_true
        kbias_stats = _summary_stats(kbias)

        under = float((kbias < 0).mean()) if n else np.nan
        over = float((kbias > 0).mean()) if n else np.nan

        rows.append(
            {
                "metric": m,
                "B_datasets": n,
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
            }
        )

    overall = pd.DataFrame(rows)

    # baseline summary row (context)
    base_stats = _summary_stats(base["baseline_ari"])
    baseline_row = pd.DataFrame(
        [
            {
                "metric": "baseline_oracle",
                "B_datasets": int(len(base)),
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
            }
        ]
    )

    overall = pd.concat([baseline_row, overall], ignore_index=True)

    # ----- ordering: keep baseline first, then sort methods by ARI mean -----
    if "ari_mean" in overall.columns:
        base_mask = overall["metric"].eq("baseline_oracle")
        overall_base = overall.loc[base_mask]
        overall_methods = overall.loc[~base_mask].sort_values(
            "ari_mean", ascending=False
        )
        overall = pd.concat([overall_base, overall_methods], ignore_index=True)

    # ----- by true_k (SI-friendly) -----
    def _group_summary_by_k(df_in: pd.DataFrame, by: str) -> pd.DataFrame:
        if by not in df_in.columns:
            return pd.DataFrame()

        out_rows = []
        for (m, gval), sub in df_in.groupby(["metric_name", by], dropna=False):
            n = int(len(sub))

            ari_stats = _summary_stats(sub["metric_ari"])
            delta = pd.to_numeric(sub["baseline_ari"], errors="coerce") - pd.to_numeric(
                sub["metric_ari"], errors="coerce"
            )
            delta_stats = _summary_stats(delta)

            k_success = int(
                pd.to_numeric(sub["is_correct"], errors="coerce")
                .fillna(0)
                .astype(int)
                .sum()
            )
            k_lo, k_hi = _wilson_ci(k_success, n)

            out_rows.append(
                {
                    "metric": m,
                    by: gval,
                    "B_datasets": n,
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
                }
            )

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
            "B_datasets",
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
        "metric_name",
        "is_optimal",
        "metric_ari",
        "baseline_ari",
        "is_correct",
        "k",
        "true_k",
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
    df_tex = df_tex.rename(
        columns={
            "\u0394 mean (sd)": r"$\Delta$ mean (sd)",
            "B_datasets": r"$B_{\mathrm{datasets}}$",
        }
    )

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


# ---------------------------------------------------------------------------
# New: per-(k*, difficulty) regime summary tables
# ---------------------------------------------------------------------------


def _display_name(metric: str) -> str:
    """Map internal metric name to a human-readable row label."""
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def _order_metrics_in_group(metrics: list[str], numeric_df: pd.DataFrame) -> list[str]:
    """Sort *metrics* by their overall (across columns) mean value, descending."""
    if not metrics:
        return []
    means = {}
    for m in metrics:
        row = numeric_df.loc[numeric_df["_metric_raw"] == m]
        if row.empty:
            means[m] = -np.inf
        else:
            vals = row.select_dtypes(include="number").values.ravel()
            vals = vals[np.isfinite(vals)]
            means[m] = float(vals.mean()) if vals.size else -np.inf
    return sorted(metrics, key=lambda m: means[m], reverse=True)


def summarize_regime_tables(
    results_df: pd.DataFrame,
    *,
    difficulty_anchors: dict[int, str] | None = None,
    decimals: int = 3,
    true_ks: tuple[int, ...] = (3, 4, 5, 6),
) -> dict[str, object]:
    """
    Build two summary tables (mean ARI and k-recovery) for a single regime.

    Rows   = metrics, grouped as:
             Baseline >> selected CARVE >> classical >> others
    Columns = 12 (k*, difficulty) pairs, ordered k-first then difficulty.

    Parameters
    ----------
    results_df : DataFrame
        One regime's benchmarking output (from ``benchmark_cluster_metrics``).
    difficulty_anchors : dict mapping int -> str, optional
        ``{0: "easy", 5: "medium", 9: "hard"}`` by default.
    decimals : int
        Rounding precision.
    true_ks : tuple of int
        Ground-truth cluster counts to include as columns.

    Returns
    -------
    dict with keys:
      - ``"ari_df"``         – pd.DataFrame (MultiIndex columns)
      - ``"k_recovery_df"``  – pd.DataFrame (MultiIndex columns)
      - ``"ari_tex"``        – LaTeX string
      - ``"k_recovery_tex"`` – LaTeX string
    """
    if difficulty_anchors is None:
        difficulty_anchors = {0: "easy", 5: "medium", 9: "hard"}

    diff_col = _resolve_difficulty_col(results_df)
    df = results_df.copy()

    # Column tuples in desired order: k-first, difficulty-second
    sorted_anchors = sorted(difficulty_anchors.items())  # by int
    col_tuples = []
    for k in sorted(true_ks):
        for lvl_int, lvl_label in sorted_anchors:
            col_tuples.append((k, lvl_label, lvl_int))

    # ---- compute per-cell values ----------------------------------------
    df_sel = df.loc[df["is_optimal"] == True].copy()

    # Discover all metric names present
    all_metrics = sorted(df_sel["metric_name"].unique())

    # --- ARI table ---
    ari_rows: list[dict] = []
    krec_rows: list[dict] = []

    # Baseline oracle row
    bl_ari_row: dict[str, object] = {
        "_metric_raw": "baseline_oracle",
        "Metric": _display_name("baseline_oracle"),
    }
    bl_krec_row: dict[str, object] = {
        "_metric_raw": "baseline_oracle",
        "Metric": _display_name("baseline_oracle"),
    }
    id_cols = [
        c
        for c in ["difficulty_level", "true_k", "dataset_iteration", "dataset_id"]
        if c in df.columns
    ]
    id_cols_diff = [diff_col if c == "difficulty_level" else c for c in id_cols]
    base = df.drop_duplicates(
        subset=[diff_col, "true_k", "dataset_iteration"]
        if "dataset_iteration" in df.columns
        else [diff_col, "true_k"]
    )
    for k, lbl, lvl_int in col_tuples:
        cell = base.loc[
            (base["true_k"] == k) & (base[diff_col] == lvl_int), "baseline_ari"
        ]
        bl_ari_row[(k, lbl)] = (
            round(float(cell.mean()), decimals) if len(cell) else np.nan
        )
        bl_krec_row[(k, lbl)] = np.nan  # baseline has no k-selection
    ari_rows.append(bl_ari_row)
    krec_rows.append(bl_krec_row)

    # Metric rows
    for m in all_metrics:
        sub = df_sel.loc[df_sel["metric_name"] == m]
        a_row: dict[str, object] = {"_metric_raw": m, "Metric": _display_name(m)}
        k_row: dict[str, object] = {"_metric_raw": m, "Metric": _display_name(m)}
        for k, lbl, lvl_int in col_tuples:
            cell = sub.loc[(sub["true_k"] == k) & (sub[diff_col] == lvl_int)]
            a_row[(k, lbl)] = (
                round(float(cell["metric_ari"].mean()), decimals)
                if len(cell)
                else np.nan
            )
            k_row[(k, lbl)] = (
                round(float(cell["is_correct"].astype(float).mean()), decimals)
                if len(cell)
                else np.nan
            )
        ari_rows.append(a_row)
        krec_rows.append(k_row)

    ari_flat = pd.DataFrame(ari_rows)
    krec_flat = pd.DataFrame(krec_rows)

    # ---- row ordering ---------------------------------------------------
    def _build_ordered(flat_df: pd.DataFrame) -> pd.DataFrame:
        present = set(flat_df["_metric_raw"])
        selected = [m for m in SELECTED_CARVE if m in present]
        classical = [m for m in CLASSICAL if m in present]
        exclude = [m for m in EXCLUDE if m in present]
        others = [
            m
            for m in present
            if m not in {"baseline_oracle"}
            and m not in selected
            and m not in classical
            and m not in exclude
        ]

        # Sort each group by overall mean (descending) using the ARI table values
        selected = _order_metrics_in_group(selected, ari_flat)
        classical = _order_metrics_in_group(classical, ari_flat)
        others = _order_metrics_in_group(others, ari_flat)

        sep = {"_metric_raw": "---", "Metric": "---"}
        for kt in col_tuples:
            sep[(kt[0], kt[1])] = np.nan

        ordered_rows = []
        # Baseline
        ordered_rows.append(
            flat_df.loc[flat_df["_metric_raw"] == "baseline_oracle"].iloc[0]
        )
        # Separator
        ordered_rows.append(pd.Series(sep))
        # Selected CARVE
        for m in selected:
            ordered_rows.append(flat_df.loc[flat_df["_metric_raw"] == m].iloc[0])
        # Separator
        ordered_rows.append(pd.Series(sep))
        # Classical
        for m in classical:
            ordered_rows.append(flat_df.loc[flat_df["_metric_raw"] == m].iloc[0])
        # Separator
        ordered_rows.append(pd.Series(sep))
        # Others
        for m in others:
            ordered_rows.append(flat_df.loc[flat_df["_metric_raw"] == m].iloc[0])

        out = pd.DataFrame(ordered_rows).reset_index(drop=True)
        return out

    ari_ordered = _build_ordered(ari_flat)
    krec_ordered = _build_ordered(krec_flat)

    # ---- reshape into MultiIndex columns --------------------------------
    value_cols = [(k, lbl) for k, lbl, _ in col_tuples]
    mi = pd.MultiIndex.from_tuples(value_cols, names=["k*", "Difficulty"])

    def _to_multiindex(ordered_df: pd.DataFrame) -> pd.DataFrame:
        metric_col = ordered_df["Metric"].values
        raw_col = ordered_df["_metric_raw"].values
        data = ordered_df[value_cols].values
        result = pd.DataFrame(data, columns=mi)
        result.insert(0, "Metric", metric_col)
        result.insert(0, "_metric_raw", raw_col)
        return result

    ari_df = _to_multiindex(ari_ordered)
    krec_df = _to_multiindex(krec_ordered)

    # ---- LaTeX rendering ------------------------------------------------
    # Pass value_cols explicitly so the renderer doesn't rely on column name matching
    ari_tex = _render_grouped_tex(
        ari_df, value_cols=value_cols, decimals=decimals, caption="Mean ARI"
    )
    krec_tex = _render_grouped_tex(
        krec_df, value_cols=value_cols, decimals=decimals, caption="$k$-Recovery Rate"
    )

    # Drop internal helper column from the returned DataFrames
    # Use .pop() to avoid MultiIndex drop issues
    ari_df.pop("_metric_raw")
    krec_df.pop("_metric_raw")

    return {
        "ari_df": ari_df,
        "k_recovery_df": krec_df,
        "ari_tex": ari_tex,
        "k_recovery_tex": krec_tex,
    }


def _render_grouped_tex(
    df: pd.DataFrame,
    *,
    value_cols: list[tuple[int, str]],
    decimals: int = 3,
    caption: str = "",
) -> str:
    """
    Render a grouped summary DataFrame as a LaTeX table.

    Separator rows (Metric == "---") are replaced with ``\\midrule``.
    Best value per column is **bold**, second-best is \\underline.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``_metric_raw``, ``Metric``, and all keys in *value_cols*.
    value_cols : list of (k, difficulty_label) tuples
        The data column keys in display order.
    """
    raw_metrics = df["_metric_raw"].values.copy()
    metric_col_vals = df["Metric"].values.copy()

    # Extract just the numeric data into a fresh DataFrame with integer column indices.
    # This avoids all MultiIndex/tuple ambiguity when accessing cells.
    n_rows = len(df)
    n_val = len(value_cols)
    raw_data = np.empty((n_rows, n_val), dtype=object)
    for j, vc in enumerate(value_cols):
        raw_data[:, j] = df.loc[:, vc].values
    val_data = pd.DataFrame(raw_data, columns=list(range(n_val)))

    # ---- Rank and annotate cells ----------------------------------------
    is_sep = raw_metrics == "---"
    is_baseline = raw_metrics == "baseline_oracle"
    rankable = ~is_sep & ~is_baseline

    for col in val_data.columns:
        vals = pd.to_numeric(val_data.loc[rankable, col], errors="coerce")
        finite = vals.dropna()
        if len(finite) < 2:
            continue
        ranked = finite.rank(ascending=False, method="min")
        best_idx = ranked[ranked == 1].index
        second_idx = ranked[ranked == 2].index
        for idx in best_idx:
            v = val_data.at[idx, col]
            if pd.notna(v) and not isinstance(v, str):
                val_data.at[idx, col] = f"\\textbf{{{float(v):.{decimals}f}}}"
        for idx in second_idx:
            v = val_data.at[idx, col]
            if pd.notna(v) and not isinstance(v, str):
                val_data.at[idx, col] = f"\\underline{{{float(v):.{decimals}f}}}"

    # Format remaining numeric cells
    for col in val_data.columns:
        val_data[col] = val_data[col].apply(
            lambda v: (
                f"{v:.{decimals}f}"
                if isinstance(v, (int, float)) and np.isfinite(v)
                else (v if isinstance(v, str) else "")
            )
        )

    # ---- Build the LaTeX string manually --------------------------------
    # Escape metric names
    metric_col_vals = [str(m).replace("_", r"\_") for m in metric_col_vals]

    # Detect k* groups from value_cols (which are (k, difficulty) tuples)
    k_groups: list[tuple[str, int]] = []  # (k_label, count)
    prev_k = None
    for k_star, diff_label in value_cols:
        if k_star != prev_k:
            k_groups.append((str(k_star), 1))
            prev_k = k_star
        else:
            k_groups[-1] = (k_groups[-1][0], k_groups[-1][1] + 1)

    n_val = len(value_cols)
    col_fmt = "l" + "c" * n_val

    lines = []
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    # Top header row: k* groups
    top_cells = [""]
    for k_label, span in k_groups:
        top_cells.append(f"\\multicolumn{{{span}}}{{c}}{{$k^* = {k_label}$}}")
    lines.append(" & ".join(top_cells) + r" \\")

    # Cmidrules under each k* group
    col_idx = 2  # first value column is 2 (1-indexed; col 1 = Metric)
    cmi_parts = []
    for _, span in k_groups:
        cmi_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + span - 1}}}")
        col_idx += span
    lines.append(" ".join(cmi_parts))

    # Sub-header row: difficulty labels
    sub_cells = ["Metric"]
    for _, diff_label in value_cols:
        sub_cells.append(diff_label)
    lines.append(" & ".join(sub_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for i in range(len(metric_col_vals)):
        if is_sep[i]:
            lines.append(r"\midrule")
            continue
        cells = [metric_col_vals[i]]
        for j in range(n_val):
            cells.append(str(val_data.at[i, j]))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    """Minimal TeX escaping for header strings."""
    if "$" in s or "\\" in s:
        return s
    return (
        s.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


# ---------------------------------------------------------------------------
# Scaling-experiment summary tables
# ---------------------------------------------------------------------------


def _pick_three_axis_anchors(axis_values: pd.Series) -> list[float]:
    """Pick (low, mid, high) anchors from a series of distinct axis values."""
    unique = sorted(pd.to_numeric(axis_values, errors="coerce").dropna().unique())
    if len(unique) < 3:
        raise ValueError(
            f"Need at least 3 distinct axis values to pick anchors; got {len(unique)}"
        )
    return [unique[0], unique[len(unique) // 2], unique[-1]]


def _format_axis_anchor(v: float) -> str:
    """Format an axis anchor value as a column label (integer if whole, else compact)."""
    fv = float(v)
    return f"{int(fv)}" if fv.is_integer() else f"{fv:g}"


def summarize_scaling_tables(
    results_df: pd.DataFrame,
    *,
    axis_anchors: list[float] | None = None,
    decimals: int = 3,
    true_ks: tuple[int, ...] = (3, 4),
) -> dict[str, object]:
    """
    Build mean-ARI and k-recovery summary tables for one scaling experiment.

    Mirrors :func:`summarize_regime_tables` but groups by ``axis_value`` rather
    than ``difficulty_level``. Three column anchors (low / mid / high) are
    derived automatically from the distinct axis values present in the data
    when ``axis_anchors`` is not supplied.

    Parameters
    ----------
    results_df : DataFrame
        One scaling experiment's benchmarking output (from
        ``benchmark_scaling``). Must contain columns: ``axis_value``,
        ``metric_name``, ``is_optimal``, ``metric_ari``, ``baseline_ari``,
        ``is_correct``, ``k``, ``true_k``.
    axis_anchors : list of numeric, optional
        Three axis values to use as columns. Defaults to the min, median, and
        max of the distinct axis values.
    decimals : int
        Rounding precision.
    true_ks : tuple of int
        Ground-truth cluster counts to include as columns (default ``(3, 4)``).

    Returns
    -------
    dict with keys:
      - ``"ari_df"``         – pd.DataFrame (MultiIndex columns)
      - ``"k_recovery_df"``  – pd.DataFrame (MultiIndex columns)
      - ``"ari_tex"``        – LaTeX string
      - ``"k_recovery_tex"`` – LaTeX string
    """
    if "axis_value" not in results_df.columns:
        raise ValueError("results_df must have an 'axis_value' column")

    df = results_df.copy()

    if axis_anchors is None:
        axis_anchors = _pick_three_axis_anchors(df["axis_value"])

    # Column tuples in desired order: k-first, anchor-second
    col_tuples: list[tuple[int, str, float]] = []
    for k in sorted(true_ks):
        for v in axis_anchors:
            col_tuples.append((k, _format_axis_anchor(v), float(v)))

    # ---- compute per-cell values ----------------------------------------
    df_sel = df.loc[df["is_optimal"] == True].copy()
    all_metrics = sorted(df_sel["metric_name"].unique())

    ari_rows: list[dict] = []
    krec_rows: list[dict] = []

    # Baseline oracle row (no metric selection; just baseline ARI per dataset)
    bl_ari_row: dict[str, object] = {
        "_metric_raw": "baseline_oracle",
        "Metric": _display_name("baseline_oracle"),
    }
    bl_krec_row: dict[str, object] = {
        "_metric_raw": "baseline_oracle",
        "Metric": _display_name("baseline_oracle"),
    }
    base = df.drop_duplicates(
        subset=["axis_value", "true_k", "dataset_iteration"]
        if "dataset_iteration" in df.columns
        else ["axis_value", "true_k"]
    )
    for k, lbl, v in col_tuples:
        cell = base.loc[
            (base["true_k"] == k) & (np.isclose(base["axis_value"].astype(float), v)),
            "baseline_ari",
        ]
        bl_ari_row[(k, lbl)] = (
            round(float(cell.mean()), decimals) if len(cell) else np.nan
        )
        bl_krec_row[(k, lbl)] = np.nan
    ari_rows.append(bl_ari_row)
    krec_rows.append(bl_krec_row)

    # Metric rows
    for m in all_metrics:
        sub = df_sel.loc[df_sel["metric_name"] == m]
        a_row: dict[str, object] = {"_metric_raw": m, "Metric": _display_name(m)}
        k_row: dict[str, object] = {"_metric_raw": m, "Metric": _display_name(m)}
        for k, lbl, v in col_tuples:
            cell = sub.loc[
                (sub["true_k"] == k)
                & (np.isclose(sub["axis_value"].astype(float), v))
            ]
            a_row[(k, lbl)] = (
                round(float(cell["metric_ari"].mean()), decimals)
                if len(cell)
                else np.nan
            )
            k_row[(k, lbl)] = (
                round(float(cell["is_correct"].astype(float).mean()), decimals)
                if len(cell)
                else np.nan
            )
        ari_rows.append(a_row)
        krec_rows.append(k_row)

    ari_flat = pd.DataFrame(ari_rows)
    krec_flat = pd.DataFrame(krec_rows)

    # ---- row ordering ---------------------------------------------------
    def _build_ordered(flat_df: pd.DataFrame) -> pd.DataFrame:
        present = set(flat_df["_metric_raw"])
        selected = [m for m in SELECTED_CARVE if m in present]
        classical = [m for m in CLASSICAL if m in present]
        exclude = [m for m in EXCLUDE if m in present]
        others = [
            m
            for m in present
            if m not in {"baseline_oracle"}
            and m not in selected
            and m not in classical
            and m not in exclude
        ]

        selected = _order_metrics_in_group(selected, ari_flat)
        classical = _order_metrics_in_group(classical, ari_flat)
        others = _order_metrics_in_group(others, ari_flat)

        sep = {"_metric_raw": "---", "Metric": "---"}
        for kt in col_tuples:
            sep[(kt[0], kt[1])] = np.nan

        ordered_rows = []
        ordered_rows.append(
            flat_df.loc[flat_df["_metric_raw"] == "baseline_oracle"].iloc[0]
        )
        ordered_rows.append(pd.Series(sep))
        for m in selected:
            ordered_rows.append(flat_df.loc[flat_df["_metric_raw"] == m].iloc[0])
        ordered_rows.append(pd.Series(sep))
        for m in classical:
            ordered_rows.append(flat_df.loc[flat_df["_metric_raw"] == m].iloc[0])
        ordered_rows.append(pd.Series(sep))
        for m in others:
            ordered_rows.append(flat_df.loc[flat_df["_metric_raw"] == m].iloc[0])

        return pd.DataFrame(ordered_rows).reset_index(drop=True)

    ari_ordered = _build_ordered(ari_flat)
    krec_ordered = _build_ordered(krec_flat)

    # ---- reshape into MultiIndex columns --------------------------------
    value_cols = [(k, lbl) for k, lbl, _ in col_tuples]
    mi = pd.MultiIndex.from_tuples(value_cols, names=["k*", "Axis value"])

    def _to_multiindex(ordered_df: pd.DataFrame) -> pd.DataFrame:
        metric_col = ordered_df["Metric"].values
        raw_col = ordered_df["_metric_raw"].values
        data = ordered_df[value_cols].values
        result = pd.DataFrame(data, columns=mi)
        result.insert(0, "Metric", metric_col)
        result.insert(0, "_metric_raw", raw_col)
        return result

    ari_df = _to_multiindex(ari_ordered)
    krec_df = _to_multiindex(krec_ordered)

    # ---- LaTeX rendering ------------------------------------------------
    ari_tex = _render_grouped_tex(
        ari_df, value_cols=value_cols, decimals=decimals, caption="Mean ARI"
    )
    krec_tex = _render_grouped_tex(
        krec_df, value_cols=value_cols, decimals=decimals, caption="$k$-Recovery Rate"
    )

    ari_df.pop("_metric_raw")
    krec_df.pop("_metric_raw")

    return {
        "ari_df": ari_df,
        "k_recovery_df": krec_df,
        "ari_tex": ari_tex,
        "k_recovery_tex": krec_tex,
    }


def summarize_scaling_runtime_table(
    runtimes_by_experiment: dict[str, pd.DataFrame],
    *,
    axis_anchors_by_experiment: dict[str, list[float]] | None = None,
    decimals: int = 2,
) -> dict[str, object]:
    """
    Build a combined wall-clock runtime table across the four scaling experiments.

    Each experiment contributes two rows (CARVE Stability, CARVE Generalizability).
    Columns are the three axis anchors (low / mid / high) per experiment;
    each cell is rendered as ``"mean (sd)"`` seconds, aggregated across
    ``true_k`` and ``dataset_iteration`` at that anchor.

    Parameters
    ----------
    runtimes_by_experiment : dict[str, DataFrame]
        Mapping ``experiment_label -> runtimes_df`` (from ``benchmark_scaling``).
        Each DataFrame must have ``axis_value``, ``t_carve_sec_s``, ``t_carve_sec_g``.
        The label is used as the row prefix and is also displayed alongside
        the actual axis values in the row label.
    axis_anchors_by_experiment : dict, optional
        Override the auto-derived anchors per experiment. Default: min/mid/max
        of distinct axis values.
    decimals : int
        Rounding precision for the formatted cells.

    Returns
    -------
    dict with keys:
      - ``"runtime_df"``  – pd.DataFrame (rows = experiment×mode; cols = low/mid/high)
      - ``"runtime_tex"`` – LaTeX string (``\\begin{tabular}{lccc}`` ... ``\\end{tabular}``)
    """
    if axis_anchors_by_experiment is None:
        axis_anchors_by_experiment = {}

    rows: list[dict[str, object]] = []
    col_labels = ["low", "mid", "high"]

    for exp_label, rt_df in runtimes_by_experiment.items():
        if "axis_value" not in rt_df.columns:
            raise ValueError(
                f"Runtime DataFrame for '{exp_label}' must have an 'axis_value' column"
            )
        anchors = axis_anchors_by_experiment.get(exp_label)
        if anchors is None:
            anchors = _pick_three_axis_anchors(rt_df["axis_value"])

        anchor_strs = [_format_axis_anchor(v) for v in anchors]
        axis_name = (
            str(rt_df["axis_name"].iloc[0])
            if "axis_name" in rt_df.columns and len(rt_df)
            else "axis"
        )
        setting_label = f"{exp_label} ({axis_name} = {' / '.join(anchor_strs)})"

        for mode_label, col_t in (
            ("CARVE Stability", "t_carve_sec_s"),
            ("CARVE Generalizability", "t_carve_sec_g"),
        ):
            row: dict[str, object] = {"Setting": setting_label, "Mode": mode_label}
            for col_label, anchor in zip(col_labels, anchors):
                cell = rt_df.loc[
                    np.isclose(rt_df["axis_value"].astype(float), float(anchor)),
                    col_t,
                ].astype(float)
                if len(cell):
                    m = float(cell.mean())
                    s = float(cell.std(ddof=1)) if len(cell) > 1 else 0.0
                    row[col_label] = f"{m:.{decimals}f} ({s:.{decimals}f})"
                else:
                    row[col_label] = ""
            rows.append(row)

    runtime_df = pd.DataFrame(rows, columns=["Setting", "Mode", *col_labels])

    # ---- LaTeX rendering ------------------------------------------------
    lines: list[str] = []
    lines.append(r"\begin{tabular}{ll" + "c" * len(col_labels) + "}")
    lines.append(r"\toprule")
    header = ["Setting", "Mode"] + col_labels
    lines.append(" & ".join(_tex_escape(str(h)) for h in header) + r" \\")
    lines.append(r"\midrule")
    last_setting: str | None = None
    for r in rows:
        setting_cell = (
            _tex_escape(str(r["Setting"])) if r["Setting"] != last_setting else ""
        )
        last_setting = r["Setting"]
        cells = [setting_cell, _tex_escape(str(r["Mode"]))]
        for cl in col_labels:
            cells.append(str(r.get(cl, "")))
        lines.append(" & ".join(cells) + r" \\")
        # Light separator between settings (after the second mode row of each)
        if r["Mode"] == "CARVE Generalizability" and r is not rows[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    runtime_tex = "\n".join(lines)

    return {"runtime_df": runtime_df, "runtime_tex": runtime_tex}
