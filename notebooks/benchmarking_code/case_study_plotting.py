"""Case-study plotting helpers for CARVE benchmarking notebooks.

Sections
--------
1. Constants
2. Private utilities — colors, formatting, projections
3. Baseline metric computation
4. Dimensionality reduction and scatter plots
5. Line plots — CARVE and baseline metrics over k
6. Alluvial diagrams (Plotly and Matplotlib)
7. Composite figure assembly
8. ARI comparison — data extraction and plots
"""

from __future__ import annotations
from typing import Any, Iterable, List, Literal

from itertools import product

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

import glasbey

from umap import UMAP

from benchmarking_utils import (
    align_labels,
    _build_estimator,
)
from benchmarking_metrics import calculate_metric


# ============================================================================
# 1. Constants
# ============================================================================
# Default matplotlib palette for cluster colors.  Change this string to
# switch every scatter plot to a different palette (e.g. "tab20", "Set3").
CLUSTER_PALETTE_NAME: str = "tab20"

# Colors used by the composite multi-panel figure and CARVE line plots.
CARVE_GREEN = "#009E73"
CARVE_BLUE = "#0072B2"

CARVE_LINE_COLORS = {"generalizability": CARVE_BLUE, "stability": CARVE_GREEN}

BASELINE_WARM = [
    "#FF36B5",  # pink
    "#A8389E",  # reddish-purple
    "#FF252D",  # red
    "#F28522",  # orange
]

# ARI comparison plot colors.
_CARVE_COLOR = "#009ADE"
_BASELINE_COLOR = "#FF1F5B"


# ============================================================================
# 2. Private utilities — colors, formatting, projections
# ============================================================================
def _get_color_mapping(k: int, palette_name: str | None = None) -> List[Any]:
    """Return *k* colors from a discrete categorical palette.

    The palette name is treated as a discrete colormap such as ``tab20``,
    ``Set1``, or ``Accent``. If the requested palette does not provide enough
    colors for *k*, the function falls back to Glasbey.
    """
    if k <= 0:
        return []

    cmap = plt.get_cmap(palette_name or CLUSTER_PALETTE_NAME)
    colors = getattr(cmap, "colors", None)
    if colors is None:
        raise ValueError(
            f"Palette {palette_name or CLUSTER_PALETTE_NAME!r} must be a discrete colormap"
        )

    colors = list(colors)
    if len(colors) >= k:
        return colors[:k]

    palette = glasbey.create_palette(palette_size=k)
    return [tuple(c / 255 for c in v) if not isinstance(v, str) else v for v in palette]


def _cluster_color_map(labels: np.ndarray) -> dict[int, Any]:
    """Map each unique cluster id in *labels* to a color."""
    labels = np.asarray(labels)
    uniq = sorted(int(x) for x in np.unique(labels) if x != -1)
    palette = _get_color_mapping(len(uniq))
    return {cid: palette[i] for i, cid in enumerate(uniq)}


def _metric_color_map(metric_names: Iterable[str]) -> dict[str, Any]:
    """Build a stable mapping of ``metric_name -> color``."""
    names = list(metric_names)
    cols = _get_color_mapping(len(names))
    return {name: cols[i] for i, name in enumerate(names)}


def _numeric_label_color_map(
    labels: np.ndarray, palette_name: str | None = None
) -> dict[int, Any]:
    """Map integer-like labels to colors by label value.

    This keeps the same numeric label on the same color even when some labels
    are absent from a particular plot.
    """
    int_labels = _coerce_integer_labels(labels)
    if int_labels is None:
        return {}

    uniq = sorted(int(x) for x in np.unique(int_labels))
    non_negative = [x for x in uniq if x >= 0]
    palette = _get_color_mapping(
        max(non_negative) + 1 if non_negative else 0,
        palette_name=palette_name,
    )

    color_map = {cid: palette[cid] for cid in non_negative}
    if -1 in uniq:
        color_map[-1] = (0.5, 0.5, 0.5, 1.0)
    return color_map


def _coerce_integer_labels(labels: np.ndarray) -> np.ndarray | None:
    """Return integer labels when are integer-like, else ``None``."""
    series = pd.Series(labels)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return None

    values = numeric.to_numpy()
    if not np.all(np.isclose(values, np.round(values))):
        return None

    return np.round(values).astype(int)


def _pretty_metric_name(metric: str) -> str:
    """Human-readable names used in paper-facing plots."""
    pretty = {
        "ari_stability_1se": "CARVE Stability (1se)",
        "ari_generalizability_1se": "CARVE Generalizability (1se)",
        "silhouette": "Silhouette",
        "gap": "Gap Statistic",
        "davies_bouldin": "Davies–Bouldin",
        "calinski_harabasz": "Calinski–Harabasz",
        "accuracy_generalizability": "Accuracy (Global)",
    }
    return pretty.get(metric, metric)


def _pretty_model_label(estimator_cls, params: dict[str, Any]) -> str:
    """One-line label for an estimator + its non-k params."""
    name = estimator_cls.__name__.replace("Clustering", "")
    bits = []
    for k, v in (params or {}).items():
        if k == "n_clusters":
            continue
        if isinstance(v, float):
            bits.append(f"{k}={v:.3g}")
        else:
            bits.append(f"{k}={v}")
    return name if not bits else f"{name} ({', '.join(bits)})"


def _unique_in_order(x):
    """Deduplicate while preserving first-occurrence order."""
    seen = set()
    out = []
    for v in x:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _hex_to_rgba(hex_color, a=0.35):
    """Convert ``#RRGGBB`` to a Plotly-compatible ``rgba(...)`` string."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _pca_project(X: np.ndarray) -> tuple[np.ndarray, PCA]:
    """Return (Z, fitted_pca) for 2-D PCA projection."""
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(np.asarray(X, dtype=float))
    return Z, pca


def _add_method_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``_method_id`` and ``_method_label`` columns to a CARVE results DF.

    Every unique combination of non-metric, non-``n_clusters`` columns defines
    a "method".
    """
    _GROUPBY_NA = "_NA_"
    metric_cols = {
        col
        for col in df.columns
        if any(
            x in col
            for x in ["ari_", "consensus_", "accuracy_", "_se", "_upper", "_lower"]
        )
    }
    exclude = metric_cols | {
        "estimator",
        "n_clusters",
        "index",
        "_method_id",
        "_method_label",
    }
    group_cols = [c for c in df.columns if c not in exclude and c != "n_clusters"]

    df = df.copy()
    df[group_cols] = df[group_cols].fillna(_GROUPBY_NA)

    # Step 1: Build a unique id and label per parameter group.
    seen: dict[tuple, tuple[str, str]] = {}
    ids, labels = [], []
    for _, row in df.iterrows():
        key = tuple(row[c] for c in group_cols)
        if key not in seen:
            parts = [str(row["estimator"])]
            for c in group_cols:
                val = row[c]
                if val != _GROUPBY_NA:
                    parts.append(f"{c}={val}")
            label = ", ".join(parts)
            mid = f"m{len(seen)}"
            seen[key] = (mid, label)
        ids.append(seen[key][0])
        labels.append(seen[key][1])

    # Step 2: Assign columns.
    df["_method_id"] = ids
    df["_method_label"] = labels
    return df


# ============================================================================
# 3. Baseline metric computation
# ============================================================================
def _baseline_metric_iter(
    *,
    X: np.ndarray,
    y_arr: np.ndarray | None,
    estimator_cls,
    fixed_params: dict[str, Any],
    model_label: str,
    k: int,
    metrics: list[str],
    random_state: int,
) -> list[dict]:
    """Fit one (model, k) and score all *metrics*. Used inside joblib."""
    est = _build_estimator(estimator_cls, k, fixed_params, random_state)
    labels = est.fit_predict(X)
    ari = float(adjusted_rand_score(y_arr, labels)) if y_arr is not None else np.nan
    rows = []
    for metric in metrics:
        score = calculate_metric(
            X,
            labels,
            metric,
            estimator_cls=estimator_cls,
            estimator_params=fixed_params,
            random_state=random_state,
        )
        rows.append(
            {
                "metric": metric,
                "model": model_label,
                "k": k,
                "score": float(score) if np.isfinite(score) else np.nan,
                "ari": ari,
            }
        )
    return rows


def baseline_metrics_over_k(
    X: np.ndarray,
    *,
    y: np.ndarray | None,
    model_grids: list[tuple[Any, dict[str, Any]]],
    metrics: list[str] = ["silhouette", "gap", "DB", "CH"],
    random_state: int = 0,
    n_jobs: int = 1,
    ncols: int = 2,
    figsize: tuple[float, float] = (12, 8),
    legend_below: bool = True,
    decimals: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline sweep: metric(k) for each (model variant, k).

    Returns
    -------
    curves_df : long-form (metric, model, k, score, ari)
    best_df   : best (model, k) per metric
    """
    X = np.asarray(X)
    y_arr = None if y is None else np.asarray(y)

    # Step 1: Build joblib task list for all (model, param-combo, k) triples.
    jobs = []
    for estimator_cls, grid in model_grids:
        ks = list(np.asarray(grid["n_clusters"]).astype(int))
        other_keys = [k for k in grid.keys() if k != "n_clusters"]
        other_vals = [
            grid[k] if isinstance(grid[k], (list, tuple, np.ndarray)) else [grid[k]]
            for k in other_keys
        ]
        combos = list(product(*other_vals)) if other_keys else [()]

        for combo in combos:
            fixed_params = (
                {k: v for k, v in zip(other_keys, combo)} if other_keys else {}
            )
            model_label = _pretty_model_label(estimator_cls, fixed_params)

            for k in ks:
                jobs.append(
                    delayed(_baseline_metric_iter)(
                        X=X,
                        y_arr=y_arr,
                        estimator_cls=estimator_cls,
                        fixed_params=fixed_params,
                        model_label=model_label,
                        k=int(k),
                        metrics=metrics,
                        random_state=int(random_state),
                    )
                )

    # Step 2: Execute in parallel and collect results.
    results = Parallel(n_jobs=n_jobs)(jobs) if jobs else []
    rows = [row for sub in results for row in sub]
    curves_df = pd.DataFrame(rows)

    # Step 3: Pick best per metric (max score; tie-break = smaller k).
    best_rows = []
    for metric, g in curves_df.groupby("metric", sort=False):
        gg = g[np.isfinite(g["score"])].copy()
        if gg.empty:
            best_rows.append(
                {
                    "metric": metric,
                    "best_model": None,
                    "best_k": np.nan,
                    "best_score": np.nan,
                    "best_ari": np.nan,
                }
            )
            continue
        gg = gg.sort_values(["score", "k"], ascending=[False, True], kind="mergesort")
        top = gg.iloc[0]
        best_rows.append(
            {
                "metric": metric,
                "best_model": top["model"],
                "best_k": int(top["k"]),
                "best_score": float(top["score"]),
                "best_ari": float(top["ari"]) if np.isfinite(top["ari"]) else np.nan,
            }
        )
    best_df = pd.DataFrame(best_rows)

    # Step 4: Print console summary.
    if not best_df.empty:
        print("\nbaseline selections (max metric):")
        for _, r in best_df.iterrows():
            m = _pretty_metric_name(r["metric"])
            bm = r["best_model"]
            bk = r["best_k"]
            bs = r["best_score"]
            ba = r["best_ari"]
            if bm is None:
                print(f"  - {m}: no valid scores")
            else:
                if np.isfinite(ba):
                    print(
                        f"  - {m}: k={bk}, score={bs:.{decimals}f}, ari={ba:.{decimals}f}   [{bm}]"
                    )
                else:
                    print(f"  - {m}: k={bk}, score={bs:.{decimals}f}   [{bm}]")

    # Step 5: Plot one subplot per metric (each line = a model variant).
    ms = list(metrics)
    n = len(ms)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    all_models = list(dict.fromkeys(curves_df["model"].tolist()))
    model_colors = _metric_color_map(all_models)

    for i, metric in enumerate(ms):
        ax = axes[i]
        dfm = curves_df[curves_df["metric"] == metric]
        for model, gm in dfm.groupby("model", sort=False):
            gm = gm.sort_values("k")
            ax.plot(
                gm["k"].to_numpy(),
                gm["score"].to_numpy(),
                marker="o",
                linewidth=1.8,
                label=model,
                color=model_colors.get(model, None),
            )
        ax.set_title(_pretty_metric_name(metric))
        ax.set_xlabel("k")
        ax.set_ylabel("score")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    if legend_below and all_models:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(3, len(labels)),
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()

    plt.show()
    return curves_df, best_df


# ============================================================================
# 4. Dimensionality reduction and scatter plots
# ============================================================================
def plot_dim_red(
    X: np.ndarray,
    *,
    y: np.ndarray,
    method: Literal["pca", "tsne", "umap"] = "pca",
    label_col: str = "label",
    s: int = 50,
    alpha: float = 0.8,
    linewidth: float = 0.1,
    palette: str | None = None,
    hide_axes: bool = True,
    figsize: tuple[float, float] = (12, 10),
    title: str = "PCA of raw cell counts (colored by stage label)",
    show_legend: bool = True,
    legend_title: str = "stage",
    fontsize: int = 12,
    title_fontsize: int | None = None,
    show_axis_labels: bool = True,
    axis_labels: tuple[str, str] | None = None,
    axis_arrows: bool = False,
    category_draw_order: Literal["label", "size_desc"] = "label",
    ax: plt.Axes | None = None,
    show: bool = True,
    Z: np.ndarray | None = None,
) -> None:
    """Scatter plot of a 2-D embedding (PCA / t-SNE / UMAP) colored by *y*.

    If *Z* (an ``(n, 2)`` array) is provided it is used as the embedding directly;
    *method* then only controls axis-label text. Otherwise the embedding is
    computed from *X* using the selected *method*.
    """
    # Step 1: Coerce inputs.
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    else:
        X = np.asarray(X)
    if isinstance(y, (pd.Series, pd.Index)):
        y = y.to_numpy()
    else:
        y = np.asarray(y)

    # Step 2: Compute embedding (or use the precomputed one).
    pca = None
    if Z is not None:
        pcs = np.asarray(Z)
    elif method == "pca":
        pca = PCA(n_components=2, random_state=0)
        pcs = pca.fit_transform(X)
    elif method == "tsne":
        tsne = TSNE(n_components=2, random_state=0)
        pcs = tsne.fit_transform(X)
    elif method == "umap":
        umap = UMAP(n_components=2, random_state=0)
        pcs = umap.fit_transform(X)
    else:
        raise ValueError(f"unsupported method: {method}")

    # Step 3: Build plotting dataframe.
    pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pc_df[label_col] = y
    int_labels = _coerce_integer_labels(y)
    if int_labels is not None:
        pc_df[label_col] = int_labels
        categories = sorted(int(x) for x in np.unique(int_labels))
        color_map = _numeric_label_color_map(int_labels, palette_name=palette)
    else:
        cat = pd.Categorical(pc_df[label_col])
        if hasattr(cat, "remove_unused_categories"):
            cat = cat.remove_unused_categories()

        categories = list(cat.categories)
        n_labels = len(categories)
        colors = _get_color_mapping(n_labels, palette_name=palette)

        color_map = {lab: colors[i] for i, lab in enumerate(categories)}

    # Iterate categories largest-first so the smallest cluster lands on top of
    # the z-stack and is not occluded.
    if category_draw_order == "size_desc":
        sizes = pc_df[label_col].value_counts()
        draw_categories = sorted(
            categories, key=lambda lab: sizes.get(lab, 0), reverse=True
        )
    else:
        draw_categories = categories

    # Step 4: Draw scatter.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    for lab in draw_categories:
        sel = pc_df[pc_df[label_col] == lab]
        ax.scatter(
            sel["PC1"],
            sel["PC2"],
            label=lab,
            color=color_map[lab],
            s=s,
            alpha=alpha,
            edgecolors=[(0.0, 0.0, 0.0, 0.6)],
            linewidths=linewidth,
        )

    # Step 5: Decorate axes.
    ax.set_title(
        title, fontsize=title_fontsize
    ) if title_fontsize is not None else ax.set_title(title)
    if show_axis_labels:
        if method == "pca" and pca is not None:
            ax.set_xlabel(
                f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% var)",
                fontsize=fontsize,
            )
            ax.set_ylabel(
                f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% var)",
                fontsize=fontsize,
            )
        elif method == "pca":
            ax.set_xlabel("PC1", fontsize=fontsize)
            ax.set_ylabel("PC2", fontsize=fontsize)
        elif method == "tsne":
            ax.set_xlabel("t-SNE 1", fontsize=fontsize)
            ax.set_ylabel("t-SNE 2", fontsize=fontsize)
        elif method == "umap":
            ax.set_xlabel("UMAP 1", fontsize=fontsize)
            ax.set_ylabel("UMAP 2", fontsize=fontsize)
        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0], fontsize=fontsize)
            ax.set_ylabel(axis_labels[1], fontsize=fontsize)

    if axis_arrows:
        if method == "pca":
            xlab, ylab = "PC1", "PC2"
        elif method == "tsne":
            xlab, ylab = "t-SNE 1", "t-SNE 2"
        elif method == "umap":
            xlab, ylab = "UMAP 1", "UMAP 2"
        if axis_labels is not None:
            xlab, ylab = axis_labels
        else:
            xlab, ylab = "x", "y"
        x0, y0 = 0.04, 0.04
        arrow_len = 0.16
        arrow_kw = dict(arrowstyle="->", lw=1.4, color="black")
        ax.annotate(
            "",
            xy=(x0 + arrow_len, y0),
            xytext=(x0, y0),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=arrow_kw,
        )
        ax.annotate(
            "",
            xy=(x0, y0 + arrow_len),
            xytext=(x0, y0),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=arrow_kw,
        )
        ax.text(
            x0 + arrow_len + 0.01,
            y0,
            xlab,
            transform=ax.transAxes,
            fontsize=fontsize,
            va="center",
            ha="left",
        )
        ax.text(
            x0,
            y0 + arrow_len + 0.01,
            ylab,
            transform=ax.transAxes,
            fontsize=fontsize,
            va="bottom",
            ha="center",
        )

    if show_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")

    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if owns_fig:
        fig.tight_layout()
    if show:
        plt.show()


def plot_cluster_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    ax: plt.Axes,
    color_map: dict[int, Any] | None = None,
    s: float = 40.0,
    alpha: float = 0.85,
    edgecolor: Any = (0.0, 0.0, 0.0, 0.6),
    linewidth: float = 0.3,
    hide_axes: bool = True,
    title: str = "",
    axis_labels: tuple[str, str] = ("PC1", "PC2"),
    pca_obj: PCA | None = None,
    Z: np.ndarray | None = None,
) -> plt.Axes:
    """Draw a 2-D scatter colored by cluster labels onto *ax*.

    If *Z* is provided it is used directly; otherwise PCA is computed from *X*.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    # Step 1: Obtain 2-D coordinates.
    if Z is None:
        Z, pca_obj = _pca_project(X)
    else:
        Z = np.asarray(Z)

    if color_map is None:
        color_map = _cluster_color_map(labels)

    # Step 2: Plot each cluster.
    uniq = sorted(int(c) for c in np.unique(labels) if c != -1)
    for cid in uniq:
        m = labels == cid
        c = color_map.get(cid, (0.5, 0.5, 0.5, 1.0))
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            s=s,
            alpha=alpha,
            c=[c],
            edgecolors=edgecolor,
            linewidths=linewidth,
            label=f"C{cid}",
        )

    # Step 3: Decorate.
    ax.set_title(title, fontsize=13)
    if pca_obj is not None and not hide_axes:
        ax.set_xlabel(f"{axis_labels[0]} ({pca_obj.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"{axis_labels[1]} ({pca_obj.explained_variance_ratio_[1] * 100:.1f}%)")

    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)

    return ax


# ============================================================================
# 5. Line plots — CARVE and baseline metrics over k
# ============================================================================
def plot_baseline_best_lines(
    curves_df: pd.DataFrame,
    best_df: pd.DataFrame,
    *,
    ax: plt.Axes,
    colors: list[str] | None = None,
    marker: str = "o",
    linewidth: float = 1.8,
    title: str = "Classical metrics",
    legend_fontsize: int = 10,
    annotate: bool = True,
    grid_alpha: float = 0.22,
    normalize: bool = True,
) -> plt.Axes:
    """Plot one line per baseline metric (only the winning model).

    Parameters
    ----------
    curves_df, best_df : from ``baseline_metrics_over_k``
    normalize : If True, min-max scale each metric to [0, 1].
    """
    if colors is None:
        n = len(best_df)
        colors = (BASELINE_WARM * ((n // len(BASELINE_WARM)) + 1))[:n]

    # Step 1: Draw one line per metric's winning model.
    for i, (_, row) in enumerate(best_df.iterrows()):
        metric = row["metric"]
        model = row["best_model"]
        best_k = row["best_k"]
        if model is None or pd.isna(best_k):
            continue

        sub = curves_df[
            (curves_df["metric"] == metric) & (curves_df["model"] == model)
        ].sort_values("k")
        if sub.empty:
            continue

        x = sub["k"].to_numpy()
        y = sub["score"].to_numpy().astype(float).copy()

        if normalize:
            lo, hi = np.nanmin(y), np.nanmax(y)
            rng = hi - lo
            if rng > 0:
                y = (y - lo) / rng
            else:
                y = np.zeros_like(y)

        label = f"{_pretty_metric_name(metric)} — {model}"
        c = colors[i % len(colors)]
        ax.plot(
            x, y, marker=marker, linewidth=linewidth, color=c, label=label, zorder=2
        )

        # Step 2: Mark selected k with a vertical line and dot.
        if annotate and np.isfinite(best_k):
            idx_best = np.argmin(np.abs(sub["k"].to_numpy() - int(best_k)))
            best_score_plot = y[idx_best]
            ax.axvline(
                int(best_k),
                linestyle="--",
                linewidth=1.0,
                color=c,
                alpha=0.35,
                zorder=0,
            )
            ax.scatter(
                [int(best_k)],
                [best_score_plot],
                s=60,
                color=c,
                edgecolor="black",
                linewidths=0.6,
                zorder=5,
            )

    # Step 3: Axes formatting.
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Score (normalized)" if normalize else "Score", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=grid_alpha)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.legend(
        fontsize=legend_fontsize,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
    )
    return ax


def plot_carve_best_lines(
    carve_obj: Any,
    *,
    ax: plt.Axes,
    measures: list[tuple[str, str]] | None = None,
    colors: dict[str, str] | None = None,
    marker: str = "o",
    linewidth: float = 2.0,
    alpha_band: float = 0.18,
    title: str = "CARVE",
    legend_fontsize: int = 10,
    annotate: bool = True,
    show_selected_k: bool = True,
    grid_alpha: float = 0.22,
    show_1se: bool = True,
    not_two: bool = False,
) -> plt.Axes:
    """Plot multiple CARVE measure lines onto a single *ax*.

    Parameters
    ----------
    annotate : bool
        Show a text-box annotation listing each selected k*.
    show_selected_k : bool
        Draw a vertical dashed line and dot at each selected k
        (independent of *annotate*).
    not_two : bool
        If True, exclude k=2 configurations from selection.
    """
    from carve._selection import MEASURE_MAP, select_best_row_by_rule

    if measures is None:
        measures = [("generalizability", "1se"), ("stability", "1se")]
    if colors is None:
        colors = CARVE_LINE_COLORS

    df = carve_obj.estimator_results_.copy()
    if "_method_id" not in df.columns or "_method_label" not in df.columns:
        df = _add_method_columns(df)

    annotations_text = []

    for measure, rule in measures:
        color = colors.get(measure, CARVE_GREEN)

        # Step 1: Select the winning method for this measure.
        best_row = select_best_row_by_rule(
            df, measure=measure, rule=rule, not_two=not_two, return_idx=False
        )
        best_mid = str(best_row["_method_id"])
        best_k = int(best_row["n_clusters"])
        df_best = df[df["_method_id"] == best_mid].sort_values("n_clusters").copy()

        y_col = MEASURE_MAP[measure]
        se_col = f"{y_col}_se"
        has_se = se_col in df_best.columns

        x = df_best["n_clusters"].astype(int).to_numpy()
        y_vals = df_best[y_col].astype(float).to_numpy()
        method_label = str(df_best["_method_label"].iloc[0])
        pretty_measure = measure.replace("_", " ").title()
        line_label = f"{pretty_measure} ({rule}) — {method_label}"

        # Step 2: Draw the line and optional 1-SE band.
        ax.plot(
            x,
            y_vals,
            marker=marker,
            linewidth=linewidth,
            color=color,
            label=line_label,
            zorder=3,
        )

        if show_1se and has_se:
            se = df_best[se_col].astype(float).to_numpy()
            lo, hi = y_vals - se, y_vals + se
            ax.fill_between(
                x, lo, hi, color=color, alpha=alpha_band, linewidth=0, zorder=1
            )

        # Step 3: Mark selected k (vertical line + dot).
        if show_selected_k:
            sel_y = float(best_row[y_col])
            ax.axvline(
                best_k, linestyle="--", linewidth=1.0, color=color, alpha=0.4, zorder=0
            )
            ax.scatter(
                [best_k],
                [sel_y],
                s=60,
                color=color,
                edgecolor="black",
                linewidths=0.6,
                zorder=5,
            )

        # Step 4: Collect text for optional annotation box.
        if annotate:
            annotations_text.append(f"{pretty_measure}: k*={best_k}")

    # Step 5: Draw annotation text box.
    if annotate and annotations_text:
        ax.text(
            0.02,
            0.98,
            "\n".join(annotations_text),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.9
            ),
            zorder=10,
        )

    # Step 6: Axes formatting.
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("ARI", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=grid_alpha)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.legend(
        fontsize=legend_fontsize,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
    )
    return ax


# ============================================================================
# 6. Alluvial diagrams
# ============================================================================

# ---------------------------------------------------------------------------
# 6a. Plotly alluvial
# ---------------------------------------------------------------------------
def _luminance(color) -> float:
    """Relative luminance of a color (RGBA tuple or hex string)."""
    rgba = mpl.colors.to_rgba(color)
    return 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]


def _stack_segments(sizes, gap_frac=0.015):
    """Stack segments proportionally in [0, 1]. Returns list of (y_bottom, y_top)."""
    total = sum(sizes)
    n = len(sizes)
    total_gap = gap_frac * max(n - 1, 0)
    available = 1.0 - total_gap
    positions = []
    y = 1.0
    for s in sizes:
        h = (s / total) * available if total > 0 else 0
        y_top = y
        y_bot = y - h
        positions.append((y_bot, y_top))
        y = y_bot - gap_frac
    return positions


def _draw_flow(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot, color, alpha=0.35):
    """Draw a Sankey-style ribbon band from [x0, y0] to [x1, y1] using cubic Béziers."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    mx = (x0 + x1) / 2

    verts = [
        (x0, y0_top),
        (mx, y0_top),
        (mx, y1_top),
        (x1, y1_top),
        (x1, y1_bot),
        (mx, y1_bot),
        (mx, y0_bot),
        (x0, y0_bot),
        (x0, y0_top),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = PathPatch(
        Path(verts, codes), fc=color, ec="none", alpha=alpha, lw=0, zorder=1
    )
    ax.add_patch(patch)


def plot_alluvial(
    ax,
    y_true,
    left_labels,
    right_labels,
    left_cmap: dict[int, Any],
    right_cmap: dict[int, Any],
    true_cmap: dict[str, Any],
    left_title: str = "CARVE",
    right_title: str = "Classical",
    true_title: str = "Reported Label",
    link_alpha: float = 0.4,
    bar_width: float = 0.045,
    gap_frac: float = 0.015,
    true_gap_frac: float = 0.045,
    font_size: int = 10,
):
    """Draw a 3-column alluvial diagram on a matplotlib axes."""
    # Step 1: Coerce and compute ordering / sizes.
    y_true_str = np.asarray(pd.Series(y_true).astype(str))
    left_int = np.asarray(pd.Series(left_labels).astype(int))
    right_int = np.asarray(pd.Series(right_labels).astype(int))

    true_order = _unique_in_order(y_true_str)

    # Order left/right clusters by their weighted-mean position in true_order so
    # bands align with the anchored middle column and flow crossings are reduced.
    def _order_by_true(pred_int):
        keys = sorted(np.unique(pred_int))
        ct = pd.crosstab(pd.Series(pred_int), pd.Series(y_true_str)).reindex(
            index=keys, columns=true_order, fill_value=0
        )
        idx = np.arange(len(true_order))
        totals = ct.sum(axis=1).to_numpy()
        scores = (ct.to_numpy() * idx).sum(axis=1) / np.maximum(totals, 1)
        return [k for _, k in sorted(zip(scores, keys))]

    left_order = _order_by_true(left_int)
    right_order = _order_by_true(right_int)

    left_sizes = [int(np.sum(left_int == k)) for k in left_order]
    true_sizes = [int(np.sum(y_true_str == lab)) for lab in true_order]
    right_sizes = [int(np.sum(right_int == k)) for k in right_order]

    x_L, x_T, x_R = 0.0, 0.5, 1.0
    hw = bar_width / 2

    left_pos = _stack_segments(left_sizes, gap_frac)
    true_pos = _stack_segments(true_sizes, true_gap_frac)
    right_pos = _stack_segments(right_sizes, gap_frac)

    # Step 2: Compute purity for cluster nodes.
    def _purity(pred, order):
        df_tmp = pd.DataFrame({"pred": pred, "true": y_true_str})
        ct = pd.crosstab(df_tmp["pred"], df_tmp["true"]).reindex(
            index=order,
            columns=true_order,
            fill_value=0,
        )
        sizes_tmp = ct.sum(axis=1).to_numpy()
        return ct.max(axis=1).to_numpy() / np.maximum(sizes_tmp, 1)

    pur_left = _purity(left_int, left_order)
    pur_right = _purity(right_int, right_order)

    # Step 3: Draw bars and labels.
    def _draw_bars(order, positions, x_c, cmap, purities, side):
        for i, key in enumerate(order):
            yb, yt = positions[i]
            color = cmap.get(key, (0.5, 0.5, 0.5, 1.0))
            ax.fill_betweenx(
                [yb, yt],
                x_c - hw,
                x_c + hw,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
            pur_text = f"  {purities[i] * 100:.0f}%" if purities is not None else ""
            if side == "left":
                ax.text(
                    x_c - hw - 0.012,
                    (yb + yt) / 2,
                    f"C{int(key) + 1}{pur_text}",
                    ha="right",
                    va="center",
                    fontsize=font_size,
                )
            elif side == "right":
                ax.text(
                    x_c + hw + 0.012,
                    (yb + yt) / 2,
                    f"C{int(key) + 1}{pur_text}",
                    ha="left",
                    va="center",
                    fontsize=font_size,
                )

    def _draw_true_bars(order, positions, x_c, cmap):
        for i, lab in enumerate(order):
            yb, yt = positions[i]
            color = cmap.get(lab, (0.5, 0.5, 0.5, 1.0))
            ax.fill_betweenx(
                [yb, yt],
                x_c - hw,
                x_c + hw,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
            txt_color = "white" if _luminance(color) < 0.45 else "black"
            ax.text(
                x_c,
                (yb + yt) / 2,
                str(lab),
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="normal",
                color=txt_color,
            )

    _draw_bars(left_order, left_pos, x_L, left_cmap, pur_left, "left")
    _draw_true_bars(true_order, true_pos, x_T, true_cmap)
    _draw_bars(right_order, right_pos, x_R, right_cmap, pur_right, "right")

    # Step 4: Compute cross-tabs and draw flows.
    ct_lt = pd.crosstab(pd.Series(left_int), pd.Series(y_true_str))
    ct_lt = ct_lt.reindex(index=left_order, columns=true_order, fill_value=0)

    ct_tr = pd.crosstab(pd.Series(y_true_str), pd.Series(right_int))
    ct_tr = ct_tr.reindex(index=true_order, columns=right_order, fill_value=0)

    left_used = {k: 0 for k in left_order}
    true_used_l = {lab: 0 for lab in true_order}
    true_used_r = {lab: 0 for lab in true_order}
    right_used = {k: 0 for k in right_order}

    # Left -> True flows.
    for i_l, k in enumerate(left_order):
        yb_l, yt_l = left_pos[i_l]
        h_l = yt_l - yb_l
        for i_t, lab in enumerate(true_order):
            v = int(ct_lt.loc[k, lab])
            if v == 0:
                continue
            yb_t, yt_t = true_pos[i_t]
            h_t = yt_t - yb_t

            frac_s = v / left_sizes[i_l] if left_sizes[i_l] > 0 else 0
            s_top = (
                yt_l - (left_used[k] / left_sizes[i_l]) * h_l
                if left_sizes[i_l] > 0
                else yt_l
            )
            s_bot = s_top - frac_s * h_l
            left_used[k] += v

            frac_t = v / true_sizes[i_t] if true_sizes[i_t] > 0 else 0
            t_top = (
                yt_t - (true_used_l[lab] / true_sizes[i_t]) * h_t
                if true_sizes[i_t] > 0
                else yt_t
            )
            t_bot = t_top - frac_t * h_t
            true_used_l[lab] += v

            _draw_flow(
                ax,
                x_L + hw,
                s_top,
                s_bot,
                x_T - hw,
                t_top,
                t_bot,
                true_cmap.get(lab, (0.5, 0.5, 0.5, 1.0)),
                link_alpha,
            )

    # True -> Right flows.
    for i_t, lab in enumerate(true_order):
        yb_t, yt_t = true_pos[i_t]
        h_t = yt_t - yb_t
        for i_r, k in enumerate(right_order):
            v = int(ct_tr.loc[lab, k])
            if v == 0:
                continue
            yb_r, yt_r = right_pos[i_r]
            h_r = yt_r - yb_r

            frac_s = v / true_sizes[i_t] if true_sizes[i_t] > 0 else 0
            s_top = (
                yt_t - (true_used_r[lab] / true_sizes[i_t]) * h_t
                if true_sizes[i_t] > 0
                else yt_t
            )
            s_bot = s_top - frac_s * h_t
            true_used_r[lab] += v

            frac_t = v / right_sizes[i_r] if right_sizes[i_r] > 0 else 0
            t_top = (
                yt_r - (right_used[k] / right_sizes[i_r]) * h_r
                if right_sizes[i_r] > 0
                else yt_r
            )
            t_bot = t_top - frac_t * h_r
            right_used[k] += v

            _draw_flow(
                ax,
                x_T + hw,
                s_top,
                s_bot,
                x_R - hw,
                t_top,
                t_bot,
                true_cmap.get(lab, (0.5, 0.5, 0.5, 1.0)),
                link_alpha,
            )

    # Step 5: Column titles and axis cleanup.
    ax.text(
        x_L,
        1.06,
        left_title,
        ha="center",
        va="bottom",
        fontsize=font_size + 4,
        fontweight="normal",
    )
    ax.text(
        x_T,
        1.06,
        true_title,
        ha="center",
        va="bottom",
        fontsize=font_size + 4,
        fontweight="normal",
    )
    ax.text(
        x_R,
        1.06,
        right_title,
        ha="center",
        va="bottom",
        fontsize=font_size + 4,
        fontweight="normal",
    )

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.02, 1.12)
    ax.axis("off")



# ============================================================================
# 7. Composite figure assembly
# ============================================================================
def prepare_composite_data(
    X, y, carve_obj, comparison_labels,
    *,
    measure, rule, not_two, consensus_type, embedding,
):
    """Coerce inputs, align labels, build 2-D projection and shared color maps.

    Used by the case-study notebooks to set up the data behind the multi-panel
    paper figure: the same shared palette is re-used across the reported-label,
    CARVE-clustering, and baseline-clustering scatter panels so colors are
    comparable, and label alignment is done once up-front.

    Returns a dict with keys: ``X``, ``y_arr``, ``y_cat``, ``Z``, ``pca_obj``,
    ``carve_labels``, ``comparison_labels``, ``true_cmap``, ``carve_cmap``,
    ``sil_cmap``.
    """
    X = np.asarray(X)
    y_arr = np.asarray(y) if not isinstance(y, np.ndarray) else y
    comparison_labels = np.asarray(comparison_labels)

    carve_labels = carve_obj.get_labels(
        measure=measure, rule=rule, not_two=not_two, mode=consensus_type
    )
    carve_labels = np.asarray(carve_labels)

    y_codes = pd.Categorical(y_arr).codes
    carve_labels = align_labels(y_codes, carve_labels)
    comparison_labels = align_labels(y_codes, comparison_labels)

    if embedding is not None:
        Z = np.asarray(embedding)
        pca_obj = None
    else:
        Z, pca_obj = _pca_project(X)

    y_cat = pd.Categorical(y_arr)
    if hasattr(y_cat, "remove_unused_categories"):
        y_cat = y_cat.remove_unused_categories()
    n_true = len(y_cat.categories)

    n_extra = max(
        int(np.max(carve_labels)) + 1 - n_true if carve_labels.size else 0,
        int(np.max(comparison_labels)) + 1 - n_true if comparison_labels.size else 0,
        0,
    )
    shared_palette = _get_color_mapping(n_true + n_extra)

    true_cmap = {str(lab): shared_palette[i] for i, lab in enumerate(y_cat.categories)}
    carve_cmap = {int(c): shared_palette[int(c)] for c in np.unique(carve_labels)}
    sil_cmap = {int(c): shared_palette[int(c)] for c in np.unique(comparison_labels)}

    return {
        "X": X,
        "y_arr": y_arr,
        "y_cat": y_cat,
        "Z": Z,
        "pca_obj": pca_obj,
        "carve_labels": carve_labels,
        "comparison_labels": comparison_labels,
        "true_cmap": true_cmap,
        "carve_cmap": carve_cmap,
        "sil_cmap": sil_cmap,
    }


# ============================================================================
# 8. ARI comparison — data extraction and plots
# ============================================================================
def build_baseline_best_labels(
    X: np.ndarray,
    best_df: pd.DataFrame,
    model_grids: list[tuple[Any, dict[str, Any]]],
    metric: str = "silhouette",
    random_state: int = 42,
) -> tuple[np.ndarray, str, int]:
    """Reconstruct labels for a baseline metric's best (model, k).

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    model_name : str
    k : int
    """
    # Step 1: Look up best model and k for this metric.
    match_rows = best_df[best_df["metric"] == metric]
    if match_rows.empty:
        raise KeyError(
            f"Metric {metric!r} not found in best_df. "
            f"Available: {best_df['metric'].tolist()}"
        )
    row = match_rows.iloc[0]
    target_model = row["best_model"]
    target_k = int(row["best_k"])

    # Step 2: Find matching estimator in model_grids and fit.
    for est_cls, grid in model_grids:
        other_keys = [kk for kk in grid.keys() if kk != "n_clusters"]
        other_vals = [
            grid[kk] if isinstance(grid[kk], (list, tuple, np.ndarray)) else [grid[kk]]
            for kk in other_keys
        ]
        combos = list(product(*other_vals)) if other_keys else [()]
        for combo in combos:
            fixed = {kk: v for kk, v in zip(other_keys, combo)} if other_keys else {}
            if _pretty_model_label(est_cls, fixed) == target_model:
                est = _build_estimator(est_cls, target_k, fixed, random_state)
                labels = est.fit_predict(np.asarray(X))
                return np.asarray(labels), target_model, target_k

    raise KeyError(f"Could not find model {target_model!r} in model_grids")


def extract_ari_comparison(
    y_true: np.ndarray,
    best_df: pd.DataFrame,
    carve_obj: Any,
    X: np.ndarray,
    carve_measures: list[tuple[str, str]] | None = None,
    not_two: bool = False,
) -> pd.DataFrame:
    """Build a comparison table of ARI-vs-ground-truth for each method.

    Returns
    -------
    DataFrame with columns: method, model, k, ari, source
    """
    if carve_measures is None:
        carve_measures = [("stability", "1se"), ("generalizability", "1se")]

    y_arr = np.asarray(y_true)
    result_rows: list[dict[str, Any]] = []

    # Step 1: Baselines — read ARI directly from best_df.
    for _, r in best_df.iterrows():
        ari_val = r.get("best_ari", np.nan)
        result_rows.append(
            {
                "method": _pretty_metric_name(str(r["metric"])),
                "model": r.get("best_model", ""),
                "k": int(r["best_k"]) if np.isfinite(r["best_k"]) else 0,
                "ari": float(ari_val) if np.isfinite(ari_val) else np.nan,
                "source": "baseline",
            }
        )

    # Step 2: CARVE — compute ARI from consensus labels.
    for measure, rule in carve_measures:
        carve_k = carve_obj.get_k(measure=measure, rule=rule, not_two=not_two)
        carve_labels = carve_obj.get_labels(measure=measure, rule=rule, not_two=not_two)
        ari_val = float(adjusted_rand_score(y_arr, np.asarray(carve_labels)))
        method_key = f"ari_{measure}_{rule}"
        result_rows.append(
            {
                "method": _pretty_metric_name(method_key),
                "model": "CARVE consensus",
                "k": int(carve_k),
                "ari": ari_val,
                "source": "carve",
            }
        )

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# ARI comparison plots
# ---------------------------------------------------------------------------
def _ari_colors(df: pd.DataFrame) -> list[Any]:
    """Return a color list matching df rows: CARVE vs baseline."""
    return [_CARVE_COLOR if s == "carve" else _BASELINE_COLOR for s in df["source"]]


def plot_ari_comparison_bar(
    ari_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 4),
    title: str = "Agreement with Reported Labels (ARI)",
    annotate_k: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Horizontal bar chart of ARI by method (ranked, left-to-right bars)."""
    df = (
        ari_df.dropna(subset=["ari"])
        .sort_values("ari", ascending=True)
        .reset_index(drop=True)
    )
    colors = _ari_colors(df)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_pos = np.arange(len(df))

    # Step 1: Bars (left -> right).
    ax.barh(
        y_pos,
        df["ari"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.65,
        zorder=2,
    )

    # Step 2: Decorate.
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["method"], fontsize=10)

    ari_min, ari_max = df["ari"].min(), df["ari"].max()
    margin = max(0.02, (ari_max - ari_min) * 0.08) if len(df) else 0.02
    ax.set_xlim(max(0, ari_min - margin), ari_max + margin)

    ax.set_xlabel("ARI", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    if not df.empty:
        ax.axvline(
            df["ari"].max(), color="grey", linestyle="--", linewidth=0.8, alpha=0.5
        )

    if annotate_k:
        for i, row in df.iterrows():
            ax.annotate(
                f"k={row['k']}",
                (row["ari"], i),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=10,
                va="center",
                color="black",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


