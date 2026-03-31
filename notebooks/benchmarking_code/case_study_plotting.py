from __future__ import annotations
from typing import Any, Iterable, List, Literal

from itertools import product

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from carve.cluster import SpectralClusteringCARVE

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

import glasbey

from umap import UMAP

from leiden_clustering import LeidenClustering
from benchmarking_utils import (
    align_labels,
    _build_estimator,
)
from benchmarking_metrics import calculate_metric


# --- Setup and basic handlers ---
cluster_pallette = [
    "#009ADE",
    "#00CD6C",
    "#FF1F5B",
    "#AF58BA",
    "#F28522",
    "#A6761D",
    "#A0B1BA",
]


# --- Utils ---
def _infer_axis_cols(df: pd.DataFrame) -> tuple[str, str]:
    """
    returns (axis_value_col, axis_name_col)
    preference:
      1) long-form: ('axis_value','axis_name')
      2) legacy difficulty: ('difficulty_level', '')
      3) legacy scaling raw column: (first of n_total/p/embed_dim present, '')
    """
    if "axis_value" in df.columns:
        return "axis_value", ("axis_name" if "axis_name" in df.columns else "")
    if "difficulty_level" in df.columns:
        return "difficulty_level", ""
    for c in ("n_total", "p", "embed_dim"):
        if c in df.columns:
            return c, ""
    raise ValueError(
        "could not infer benchmark axis; expected axis_value or difficulty_level or one of {n_total,p,embed_dim}"
    )


def _get_color_mapping(k: int) -> List[Any]:
    """okabe-ito for k<=7, tab20 for 8..20, hsv fallback."""
    if k <= 7:
        cols = [mpl.colors.to_rgba(cluster_pallette[i]) for i in range(k)]
    elif k <= 20:
        tab20 = plt.get_cmap("tab20")
        cols = [tab20(i) for i in range(k)]
    else:
        hex_cols = glasbey.create_palette(palette_size=k)
        cols = [mpl.colors.to_rgba(h) for h in hex_cols]

    return cols


def _cluster_color_map(labels: np.ndarray) -> dict[int, Any]:
    """
    okabe-ito for k<=7, tab20 for 8..20, hsv fallback.
    returns mapping of cluster id -> color
    """
    labs = np.asarray(labels)
    uniq = sorted(int(x) for x in np.unique(labs) if x != -1)
    k = len(uniq)

    cols = _get_color_mapping(k)

    return {cid: cols[i] for i, cid in enumerate(uniq)}


def _metric_color_map(metric_names: Iterable[str]) -> dict[str, Any]:
    """
    okabe-ito for m<=7, tab20 for 8..20, hsv fallback.
    returns mapping of metric name -> color
    """
    names = list(metric_names)
    m = len(names)

    cols = _get_color_mapping(m)

    return {name: cols[i] for i, name in enumerate(names)}


def _scatter_clusters(
    ax, Z: np.ndarray, labels: np.ndarray, title: str, subtitle: str = ""
):
    labels = np.asarray(labels)
    cmap = _cluster_color_map(labels)

    # stable ordering for legend (optional)
    uniq = sorted(int(x) for x in np.unique(labels) if x != -1)

    for cid in uniq:
        m = labels == cid
        ax.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.90, c=[cmap[cid]], linewidths=0)

    ax.set_title(title, fontsize=10)
    if subtitle:
        ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=9)

    # equal aspect
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.2)


def _pretty_metric_name(metric: str) -> str:
    # tune freely
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
    seen = set()
    out = []
    for v in x:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _hex_to_rgba(hex_color, a=0.35):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _baseline_metric_iter(
    *,
    X: np.ndarray,
    y_arr: np.ndarray | None,
    estimator_cls: Any,
    fixed_params: dict[str, Any],
    model_label: str,
    k: int,
    metrics: list[str],
    random_state: int,
) -> list[dict[str, Any]]:
    try:
        est = _build_estimator(estimator_cls, int(k), fixed_params, int(random_state))
        labels = est.fit_predict(X)
        labels = np.asarray(labels)
    except Exception:
        labels = None

    ari = np.nan
    if labels is not None and y_arr is not None and len(y_arr) == len(labels):
        ari = float(adjusted_rand_score(y_arr, labels))

    rows = []
    for metric in metrics:
        score = np.nan
        if labels is not None:
            try:
                score = float(
                    calculate_metric(
                        X,
                        labels,
                        metric=metric,
                        estimator_cls=estimator_cls,
                        estimator_params=fixed_params,
                        random_state=int(random_state),
                    )
                )
            except Exception:
                score = np.nan

        rows.append(
            {
                "metric": metric,
                "model": model_label,
                "k": int(k),
                "score": score,
                "ari": ari,
            }
        )

    return rows


# --- Main plotting functions for case studies ---
def plot_dim_red(
    X: np.ndarray,
    *,
    y: np.ndarray,
    method: Literal["pca", "tsne", "umap"] = "pca",
    label_col: str = "label",
    s: int = 50,
    alpha: float = 0.8,
    linewidth: float = 0.1,
    hide_axes: bool = True,
    figsize: tuple[float, float] = (12, 10),
    title: str = "PCA of raw cell counts (colored by stage label)",
    show_legend: bool = True,
    legend_title: str = "stage",
    show_axis_labels: bool = True,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> None:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    else:
        X = np.asarray(X)

    if isinstance(y, (pd.Series, pd.Index)):
        y = y.to_numpy()
    else:
        y = np.asarray(y)

    if method == "pca":
        # Get PCs
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

    # Construct data frame
    pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pc_df[label_col] = y

    # Get colors
    cat = pd.Categorical(pc_df[label_col])
    if hasattr(cat, "remove_unused_categories"):
        cat = cat.remove_unused_categories()
    codes = cat.codes  # -1 for NaN
    code_color_map = _cluster_color_map(codes)
    color_map = {
        lab: code_color_map[code]
        for lab, code in zip(cat.categories, range(len(cat.categories)))
    }

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for lab in cat.categories:
        sel = pc_df[pc_df[label_col] == lab]
        ax.scatter(
            sel["PC1"],
            sel["PC2"],
            label=lab,
            color=color_map[lab],
            s=s,
            alpha=alpha,
            edgecolor="k",
            linewidths=linewidth,
        )

    # Set title, axis labels, legend
    ax.set_title(title)
    if show_axis_labels:
        if method == "pca":
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% var)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% var)")
        elif method == "tsne":
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
        elif method == "umap":
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")

    if show_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")

    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def calculate_baseline_aris_and_plot(
    X: np.ndarray,
    *,
    y: np.ndarray,
    label_col: str = "label",
    s: int = 50,
    alpha: float = 0.8,
    linewidth: float = 0.1,
    hide_axes: bool = True,
    show_legend: bool = False,
    figsize: tuple[float, float] = (12, 10),
    random_state: int = 0,
) -> tuple[float, np.ndarray]:
    # --- coerce inputs ---
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy()
    else:
        X_arr = np.asarray(X)

    if isinstance(y, (pd.Series, pd.Index)):
        y_arr = y.to_numpy()
    else:
        y_arr = np.asarray(y)

    n_clusters = len(np.unique(y_arr))

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(X_arr)
    ari_kmeans = adjusted_rand_score(y_arr, kmeans_labels)

    # Agglomerative (Ward linkage)
    agg_w = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    agg_w_labels = agg_w.fit_predict(X_arr)
    ari_agg_w = adjusted_rand_score(y_arr, agg_w_labels)

    # Agglomerative (single linkage)
    agg_s = AgglomerativeClustering(n_clusters=n_clusters, linkage="single")
    agg_s_labels = agg_s.fit_predict(X_arr)
    ari_agg_s = adjusted_rand_score(y_arr, agg_s_labels)

    # Leiden
    leiden = LeidenClustering(n_clusters=n_clusters, random_state=random_state)
    leiden_labels = leiden.fit_predict(X_arr)
    ari_leiden = adjusted_rand_score(y_arr, leiden_labels)

    # Spectral Clustering (median heuristic for gamma)
    spectral = SpectralClusteringCARVE(                                                                                        
        n_clusters=n_clusters, affinity="self_tuning", random_state=random_state
    )
    spectral_labels = spectral.fit_predict(X_arr)
    ari_spectral = adjusted_rand_score(y_arr, spectral_labels)

    # Get PCs
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X_arr)

    # Construct data frame
    pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pc_df[label_col] = y_arr

    # Plot PCA with labels
    _, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)

    y_cat = pd.Categorical(y_arr)
    if hasattr(y_cat, "remove_unused_categories"):
        y_cat = y_cat.remove_unused_categories()
    y_codes = y_cat.codes
    y_code_colors = _cluster_color_map(y_codes)
    y_color_map = {
        lab: y_code_colors[code]
        for lab, code in zip(y_cat.categories, range(len(y_cat.categories)))
    }

    # Plot true labels
    ax_true = axes[0, 0]
    for lab in y_cat.categories:
        sel = pc_df[label_col] == lab
        ax_true.scatter(
            pc_df.loc[sel, "PC1"],
            pc_df.loc[sel, "PC2"],
            s=s,
            alpha=alpha,
            edgecolor="k",
            linewidth=linewidth,
            color=y_color_map[lab],
            label=lab,
        )
    ax_true.set_title("True Cell Type Labels")

    if show_legend:
        ax_true.legend(
            markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10
        )

    titles = [
        f"KMeans (ARI={ari_kmeans:.3f})",
        f"Agglomerative (Ward) (ARI={ari_agg_w:.3f})",
        f"Agglomerative (Single Linkage) (ARI={ari_agg_s:.3f})",
        f"Leiden (ARI={ari_leiden:.3f})",
        f"Spectral (ARI={ari_spectral:.3f})",
    ]
    clusterings = [
        kmeans_labels,
        agg_w_labels,
        agg_s_labels,
        leiden_labels,
        spectral_labels,
    ]
    aligned_clusterings = [
        align_labels(pd.Categorical(y).codes, labels) for labels in clusterings
    ]

    for ax, labels, title in zip(axes.flat[1:], aligned_clusterings, titles):
        cluster_color_map = _cluster_color_map(labels)
        for cluster in np.unique(labels):
            sel = labels == cluster
            ax.scatter(
                pc_df.loc[sel, "PC1"],
                pc_df.loc[sel, "PC2"],
                s=s,
                alpha=alpha,
                edgecolor="k",
                linewidth=linewidth,
                color=cluster_color_map[int(cluster)],
                label=f"Cluster {cluster}",
            )
        ax.set_title(title)

        if show_legend:
            ax.legend(
                markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10
            )

    if hide_axes:
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.suptitle("Clustering results on PCA-reduced data", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


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
    """
    simple baseline sweep:
      - computes metric(k) for each (model variant, k)
      - selects best (model, k) per metric (max score)
      - reports ARI(best) if y is given
      - plots a small grid of metric-vs-k lineplots
      - n_jobs controls joblib workers

    returns:
      curves_df (long): metric, model, k, score, ari
      best_df: metric, best_model, best_k, best_score, best_ari
    """
    X = np.asarray(X)
    y_arr = None if y is None else np.asarray(y)

    jobs = []
    for estimator_cls, grid in model_grids:
        ks = list(np.asarray(grid["n_clusters"]).astype(int))

        # expand non-k params (but keep it minimal)
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

    results = Parallel(n_jobs=n_jobs)(jobs) if jobs else []
    rows = [row for sub in results for row in sub]

    curves_df = pd.DataFrame(rows)

    # pick best per metric (max score; tie-break = smaller k)
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

    # print a clean summary (optional but handy)
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

    # ---- plotting: one subplot per metric, lines = models
    ms = list(metrics)
    n = len(ms)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    all_models = list(dict.fromkeys(curves_df["model"].tolist()))
    model_colors = _metric_color_map(all_models)  # reusing your helper for model lines

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
        fig.tight_layout(rect=[0, 0.10, 1, 1])
    else:
        fig.tight_layout()

    plt.show()

    return curves_df, best_df


# Split quantification
def plot_label_split_counts(
    true_labels,
    labels_a,
    labels_b,
    title_a="Agglomerative (single linkage)",
    title_b="KMeans",
    color_a=None,
    color_b=None,
    figsize=(14, 5),
):
    """
    Plot how many clusters each true label is split into, for two labelings.
    """
    true_labels = np.asarray(true_labels)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    # default to Okabe–Ito colors
    if color_a is None or color_b is None:
        okabe = _get_color_mapping(2)
        color_a = okabe[0] if color_a is None else color_a
        color_b = okabe[1] if color_b is None else color_b

    def count_splits(true_labels, cluster_labels):
        splits = []
        for lab in np.unique(true_labels):
            members = cluster_labels[true_labels == lab]
            splits.append(len(np.unique(members)))
        return np.asarray(splits, dtype=int)

    def plot_counts(ax, splits, title, color):
        values, counts = np.unique(splits, return_counts=True)
        ax.bar(values, counts, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Number of clusters per true label")
        ax.set_xticks(values)

    splits_a = count_splits(true_labels, labels_a)
    splits_b = count_splits(true_labels, labels_b)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    plot_counts(axes[0], splits_a, title_a, color_a)
    axes[0].set_ylabel("Count")

    plot_counts(axes[1], splits_b, title_b, color_b)
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()


def fragmentation_report(
    y_true,
    labels_a,
    labels_b,
    name_a="method A",
    name_b="method B",
    top_n=10,
    print_compare=True,
):
    """
    Compute fragmentation stats for two labelings and optionally print a comparison.
    Returns:
        (summ_a, splits_a, dist_a, top_a), (summ_b, splits_b, dist_b, top_b)
    """

    def fragmentation_by_label(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        labels = pd.unique(y_true)
        split_counts = {}
        for lab in labels:
            members = y_pred[y_true == lab]
            split_counts[lab] = pd.unique(members).size

        splits = pd.Series(split_counts, name="n_clusters").sort_values(ascending=False)
        return splits

    def summarize_fragmentation(splits: pd.Series, name="method"):
        arr = splits.to_numpy(dtype=float)
        L = splits.size

        out = {
            "name": name,
            "L_total_labels": int(L),
            "n_intact": int((arr == 1).sum()),
            "pct_intact": float((arr == 1).mean() * 100),
            "n_split_ge2": int((arr >= 2).sum()),
            "pct_split_ge2": float((arr >= 2).mean() * 100),
            "mean_splits": float(arr.mean()),
            "median_splits": float(np.median(arr)),
            "q25_splits": float(np.quantile(arr, 0.25)),
            "q75_splits": float(np.quantile(arr, 0.75)),
            "q90_splits": float(np.quantile(arr, 0.90)),
            "q95_splits": float(np.quantile(arr, 0.95)),
            "max_splits": int(arr.max()),
            "n_at_max": int((arr == arr.max()).sum()),
            "n_split_ge3": int((arr >= 3).sum()),
            "n_split_ge5": int((arr >= 5).sum()),
            "n_split_ge8": int((arr >= 8).sum()),
        }
        return out

    def split_distribution_table(splits: pd.Series):
        dist = splits.value_counts().sort_index()
        dist.index.name = "clusters_per_label"
        dist.name = "n_labels"
        return dist.reset_index()

    def print_report(y_true, y_pred, name):
        splits = fragmentation_by_label(y_true, y_pred)
        summ = summarize_fragmentation(splits, name=name)
        dist = split_distribution_table(splits)
        top = splits.head(top_n)

        label_map = {
            "name": "Method",
            "L_total_labels": "Total true labels",
            "n_intact": "Intact labels (split into 1 cluster)",
            "pct_intact": "Intact labels (%)",
            "n_split_ge2": "Labels split into 2+ clusters",
            "pct_split_ge2": "Labels split into 2+ clusters (%)",
            "mean_splits": "Average # clusters per true label",
            "median_splits": "Median # clusters per true label",
            "q25_splits": "25th percentile of splits",
            "q75_splits": "75th percentile of splits",
            "q90_splits": "90th percentile of splits",
            "q95_splits": "95th percentile of splits",
            "max_splits": "Max # clusters for any true label",
            "n_at_max": "# labels at max split count",
            "n_split_ge3": "Labels split into 3+ clusters",
            "n_split_ge5": "Labels split into 5+ clusters",
            "n_split_ge8": "Labels split into 8+ clusters",
        }

        summary_order = [
            "name",
            "L_total_labels",
            "n_intact",
            "pct_intact",
            "n_split_ge2",
            "pct_split_ge2",
            "mean_splits",
            "median_splits",
            "q25_splits",
            "q75_splits",
            "q90_splits",
            "q95_splits",
            "max_splits",
            "n_at_max",
            "n_split_ge3",
            "n_split_ge5",
            "n_split_ge8",
        ]

        print("\n" + "=" * 80)
        print(f"Label fragmentation summary: {name}")
        print("=" * 80)
        for k in summary_order:
            v = summ.get(k)
            if isinstance(v, float):
                print(f"{label_map.get(k, k):>40}: {v:.3f}")
            else:
                print(f"{label_map.get(k, k):>40}: {v}")

        print("\nHow many true labels were split into N clusters:")
        print(dist.to_string(index=False))

        print(f"\nMost fragmented true labels (top {top_n}):")
        print(top.to_string())

        return summ, splits, dist, top

    y_true = np.asarray(y_true)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    res_a = print_report(y_true, labels_a, name_a)
    res_b = print_report(y_true, labels_b, name_b)

    if print_compare:
        summ_a, *_ = res_a
        summ_b, *_ = res_b
        print("\n" + "-" * 80)
        print("Comparison (A minus B):")
        print(
            f"Change in intact labels (split into 1 cluster): {summ_a['n_intact'] - summ_b['n_intact']}"
        )
        print(
            f"Change in intact label percentage: {summ_a['pct_intact'] - summ_b['pct_intact']:.2f} percentage points"
        )
        print(
            f"Change in average # of clusters per true label: {summ_a['mean_splits'] - summ_b['mean_splits']:.3f}"
        )
        print(
            f"Change in max # of clusters for any true label: {summ_a['max_splits'] - summ_b['max_splits']}"
        )

    return res_a, res_b


def plot_label_merge_counts(
    true_labels,
    labels_a,
    labels_b,
    title_a="method A",
    title_b="method B",
    color_a=None,
    color_b=None,
    figsize=(14, 5),
):
    """
    Plot how many true labels each predicted cluster contains, for two labelings.

    This is the underclustering counterpart of `plot_label_split_counts`:
      - split counts  → overclustering  (true label → N predicted clusters)
      - merge counts  → underclustering  (predicted cluster → N true labels)
    """
    true_labels = np.asarray(true_labels)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    if color_a is None or color_b is None:
        okabe = _get_color_mapping(2)
        color_a = okabe[0] if color_a is None else color_a
        color_b = okabe[1] if color_b is None else color_b

    def count_merges(true_labels, cluster_labels):
        merges = []
        for cl in np.unique(cluster_labels):
            members = true_labels[cluster_labels == cl]
            merges.append(len(np.unique(members)))
        return np.asarray(merges, dtype=int)

    def plot_counts(ax, merges, title, color):
        values, counts = np.unique(merges, return_counts=True)
        ax.bar(values, counts, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Number of true labels per predicted cluster")
        ax.set_xticks(values)

    merges_a = count_merges(true_labels, labels_a)
    merges_b = count_merges(true_labels, labels_b)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    plot_counts(axes[0], merges_a, title_a, color_a)
    axes[0].set_ylabel("Count")

    plot_counts(axes[1], merges_b, title_b, color_b)
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()


def merging_report(
    y_true,
    labels_a,
    labels_b,
    name_a="method A",
    name_b="method B",
    top_n=10,
    print_compare=True,
):
    """
    Compute merging / underclustering stats for two labelings and optionally
    print a comparison.

    This is the directional complement of `fragmentation_report`:
      - fragmentation  → overclustering  (true label → N predicted clusters)
      - merging         → underclustering  (predicted cluster → N true labels)

    For each predicted cluster we compute:
      - **merge count**: how many distinct true labels its members belong to.
      - **purity**: fraction of members from the dominant (most frequent) true label.

    Returns:
        (summ_a, merges_a, dist_a, top_a), (summ_b, merges_b, dist_b, top_b)
    """

    def merges_by_cluster(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        clusters = pd.unique(y_pred)
        merge_counts = {}
        for cl in clusters:
            members = y_true[y_pred == cl]
            merge_counts[cl] = pd.unique(members).size

        merges = pd.Series(merge_counts, name="n_true_labels").sort_values(
            ascending=False
        )
        return merges

    def purity_by_cluster(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        clusters = pd.unique(y_pred)
        purities = {}
        for cl in clusters:
            members = y_true[y_pred == cl]
            counts = pd.Series(members).value_counts()
            purities[cl] = float(counts.iloc[0] / len(members))

        return pd.Series(purities, name="purity").sort_values(ascending=True)

    def summarize_merging(merges: pd.Series, purities: pd.Series, name="method"):
        arr = merges.to_numpy(dtype=float)
        pur = purities.to_numpy(dtype=float)
        K = merges.size

        out = {
            "name": name,
            "K_total_clusters": int(K),
            "n_pure": int((arr == 1).sum()),
            "pct_pure": float((arr == 1).mean() * 100),
            "n_merged_ge2": int((arr >= 2).sum()),
            "pct_merged_ge2": float((arr >= 2).mean() * 100),
            "mean_merges": float(arr.mean()),
            "median_merges": float(np.median(arr)),
            "q25_merges": float(np.quantile(arr, 0.25)),
            "q75_merges": float(np.quantile(arr, 0.75)),
            "q90_merges": float(np.quantile(arr, 0.90)),
            "q95_merges": float(np.quantile(arr, 0.95)),
            "max_merges": int(arr.max()),
            "n_at_max": int((arr == arr.max()).sum()),
            "n_merged_ge3": int((arr >= 3).sum()),
            "n_merged_ge5": int((arr >= 5).sum()),
            "n_merged_ge8": int((arr >= 8).sum()),
            # purity statistics
            "mean_purity": float(pur.mean()),
            "median_purity": float(np.median(pur)),
            "min_purity": float(pur.min()),
            "q10_purity": float(np.quantile(pur, 0.10)),
            "q25_purity": float(np.quantile(pur, 0.25)),
        }
        return out

    def merge_distribution_table(merges: pd.Series):
        dist = merges.value_counts().sort_index()
        dist.index.name = "true_labels_per_cluster"
        dist.name = "n_clusters"
        return dist.reset_index()

    def print_report(y_true, y_pred, name):
        merges = merges_by_cluster(y_true, y_pred)
        purities = purity_by_cluster(y_true, y_pred)
        summ = summarize_merging(merges, purities, name=name)
        dist = merge_distribution_table(merges)
        top = merges.head(top_n)

        label_map = {
            "name": "Method",
            "K_total_clusters": "Total predicted clusters",
            "n_pure": "Pure clusters (contain 1 true label)",
            "pct_pure": "Pure clusters (%)",
            "n_merged_ge2": "Clusters merging 2+ true labels",
            "pct_merged_ge2": "Clusters merging 2+ true labels (%)",
            "mean_merges": "Average # true labels per cluster",
            "median_merges": "Median # true labels per cluster",
            "q25_merges": "25th percentile of merge counts",
            "q75_merges": "75th percentile of merge counts",
            "q90_merges": "90th percentile of merge counts",
            "q95_merges": "95th percentile of merge counts",
            "max_merges": "Max # true labels in any cluster",
            "n_at_max": "# clusters at max merge count",
            "n_merged_ge3": "Clusters merging 3+ true labels",
            "n_merged_ge5": "Clusters merging 5+ true labels",
            "n_merged_ge8": "Clusters merging 8+ true labels",
            "mean_purity": "Mean cluster purity",
            "median_purity": "Median cluster purity",
            "min_purity": "Minimum cluster purity",
            "q10_purity": "10th percentile of purity",
            "q25_purity": "25th percentile of purity",
        }

        summary_order = [
            "name",
            "K_total_clusters",
            "n_pure",
            "pct_pure",
            "n_merged_ge2",
            "pct_merged_ge2",
            "mean_merges",
            "median_merges",
            "q25_merges",
            "q75_merges",
            "q90_merges",
            "q95_merges",
            "max_merges",
            "n_at_max",
            "n_merged_ge3",
            "n_merged_ge5",
            "n_merged_ge8",
            "mean_purity",
            "median_purity",
            "min_purity",
            "q10_purity",
            "q25_purity",
        ]

        print("\n" + "=" * 80)
        print(f"Cluster merging summary: {name}")
        print("=" * 80)
        for k in summary_order:
            v = summ.get(k)
            if isinstance(v, float):
                print(f"{label_map.get(k, k):>45}: {v:.3f}")
            else:
                print(f"{label_map.get(k, k):>45}: {v}")

        print("\nHow many predicted clusters contain N true labels:")
        print(dist.to_string(index=False))

        print(f"\nMost merged predicted clusters (top {top_n}):")
        print(top.to_string())

        return summ, merges, dist, top

    y_true = np.asarray(y_true)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    res_a = print_report(y_true, labels_a, name_a)
    res_b = print_report(y_true, labels_b, name_b)

    if print_compare:
        summ_a, *_ = res_a
        summ_b, *_ = res_b
        print("\n" + "-" * 80)
        print("Comparison (A minus B):")
        print(
            f"Change in pure clusters (contain 1 true label): {summ_a['n_pure'] - summ_b['n_pure']}"
        )
        print(
            f"Change in pure cluster percentage: {summ_a['pct_pure'] - summ_b['pct_pure']:.2f} percentage points"
        )
        print(
            f"Change in average # true labels per cluster: {summ_a['mean_merges'] - summ_b['mean_merges']:.3f}"
        )
        print(
            f"Change in max # true labels in any cluster: {summ_a['max_merges'] - summ_b['max_merges']}"
        )
        print(
            f"Change in mean cluster purity: {summ_a['mean_purity'] - summ_b['mean_purity']:.3f}"
        )

    return res_a, res_b


# Alluvial Plot:
def alluvial_compare(
    y_true,
    left_labels,
    right_labels,
    left_title="CARVE (spectral)",
    right_title="silhouette (agg/ward)",
    true_title="true label",
    palette_name="tab10",
    link_alpha=0.35,
    node_pad=20,
    node_thickness=18,
    font_size=14,
    height=600,
    width=1200,
    title_y=1.08,
    vertical_margin=100,
):
    # --- coerce to strings for clean labeling ---
    y_true = pd.Series(y_true).astype(str).to_numpy()
    left_labels = pd.Series(left_labels).astype(int).to_numpy()
    right_labels = pd.Series(right_labels).astype(int).to_numpy()

    n = len(y_true)
    assert len(left_labels) == n and len(right_labels) == n, "length mismatch"

    # --- order true labels by first appearance (closest to how people expect them) ---
    true_order = _unique_in_order(y_true)

    # --- order clusters numerically, show as 1..K ---
    left_order = sorted(np.unique(left_labels))
    right_order = sorted(np.unique(right_labels))

    # --- colors: assign a stable color per TRUE label ---
    cmap = plt.get_cmap(palette_name)
    true_colors = {
        lab: "#{:02x}{:02x}{:02x}".format(
            int(255 * cmap(i % cmap.N)[0]),
            int(255 * cmap(i % cmap.N)[1]),
            int(255 * cmap(i % cmap.N)[2]),
        )
        for i, lab in enumerate(true_order)
    }

    # --- helper: compute purity + share and dominant true label per predicted cluster ---
    def cluster_stats(pred):
        df = pd.DataFrame({"pred": pred, "true": y_true})
        ct = pd.crosstab(df["pred"], df["true"]).reindex(
            index=sorted(df["pred"].unique()), columns=true_order, fill_value=0
        )
        sizes = ct.sum(axis=1).to_numpy()
        shares = sizes / n
        dom_true = ct.idxmax(axis=1).to_numpy()
        purities = ct.max(axis=1).to_numpy() / np.maximum(sizes, 1)
        return ct, sizes, shares, dom_true, purities

    ct_left, _, share_left, dom_left, pur_left = cluster_stats(left_labels)
    ct_right, _, share_right, dom_right, pur_right = cluster_stats(right_labels)

    # --- node labels (match screenshot style: big purity %, then share %) ---
    left_node_labels = []
    left_node_colors = []
    for i, k in enumerate(left_order):
        purity = pur_left[i]
        share = share_left[i]
        left_node_labels.append(f"{k + 1}<br>{purity * 100:.0f}%<br>{share * 100:.0f}%")
        left_node_colors.append(true_colors[dom_left[i]])

    true_node_labels = [lab for lab in true_order]
    true_node_colors = [true_colors[lab] for lab in true_order]

    right_node_labels = []
    right_node_colors = []
    for i, k in enumerate(right_order):
        purity = pur_right[i]
        share = share_right[i]
        right_node_labels.append(
            f"{k + 1}<br>{purity * 100:.0f}%<br>{share * 100:.0f}%"
        )
        right_node_colors.append(true_colors[dom_right[i]])

    # --- node indexing ---
    nL = len(left_order)
    nT = len(true_order)
    nR = len(right_order)

    idx_left = {k: i for i, k in enumerate(left_order)}
    idx_true = {lab: nL + i for i, lab in enumerate(true_order)}
    idx_right = {k: nL + nT + i for i, k in enumerate(right_order)}

    # --- links: left -> true and true -> right, colored by TRUE label ---
    sources = []
    targets = []
    values = []
    colors = []

    # left -> true
    for k in left_order:
        for lab in true_order:
            v = (
                int(ct_left.loc[k, lab])
                if (k in ct_left.index and lab in ct_left.columns)
                else 0
            )
            if v > 0:
                sources.append(idx_left[k])
                targets.append(idx_true[lab])
                values.append(v)
                colors.append(_hex_to_rgba(true_colors[lab], link_alpha))

    # true -> right
    ct_tr = pd.crosstab(
        pd.Series(y_true, name="true"), pd.Series(right_labels, name="pred")
    ).reindex(index=true_order, columns=right_order, fill_value=0)
    for lab in true_order:
        for k in right_order:
            v = int(ct_tr.loc[lab, k])
            if v > 0:
                sources.append(idx_true[lab])
                targets.append(idx_right[k])
                values.append(v)
                colors.append(_hex_to_rgba(true_colors[lab], link_alpha))

    # --- fixed 3-column layout (x positions), evenly spaced y positions per column ---
    def col_positions(m, top=0.06, bottom=0.94):
        if m == 1:
            return [0.5]
        return list(np.linspace(top, bottom, m))

    x = ([0.0] * nL) + ([0.5] * nT) + ([1.0] * nR)
    y = (col_positions(nL)) + (col_positions(nT)) + (col_positions(nR))

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=node_pad,
                thickness=node_thickness,
                line=dict(color="rgba(0,0,0,0.25)", width=0.5),
                label=left_node_labels + true_node_labels + right_node_labels,
                color=left_node_colors + true_node_colors + right_node_colors,
                x=x,
                y=y,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors,
            ),
        )
    )

    fig.update_layout(
        font=dict(size=font_size, family="Arial"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=width,
        height=height,
        margin=dict(
            l=40, r=40, t=max(int(vertical_margin), 80), b=int(vertical_margin)
        ),
        annotations=[
            dict(
                x=0.0,
                y=title_y,
                xref="paper",
                yref="paper",
                text=left_title,
                showarrow=False,
                font=dict(size=font_size + 2),
            ),
            dict(
                x=0.5,
                y=title_y,
                xref="paper",
                yref="paper",
                text=true_title,
                showarrow=False,
                font=dict(size=font_size + 2),
            ),
            dict(
                x=1.0,
                y=title_y,
                xref="paper",
                yref="paper",
                text=right_title,
                showarrow=False,
                font=dict(size=font_size + 2),
            ),
        ],
    )
    return fig


# ===================================================================
# Composite paper figure
# ===================================================================

# --- Color constants for composite figure ---
CARVE_GREEN = "#009E73"
CARVE_BLUE = "#0072B2"
CARVE_GREEN_LIGHT = "#66C2A5"

CARVE_LINE_COLORS = {"generalizability": CARVE_GREEN, "stability": CARVE_BLUE}

BASELINE_WARM = [
    "#E0457B",
    "#A8389E",
    "#D6292E",
    "#F28522",
]  # pink, reddish-purple, red, orange


def _pca_project(X: np.ndarray) -> tuple[np.ndarray, PCA]:
    """Return (Z, fitted_pca) for 2-D PCA projection."""
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(np.asarray(X, dtype=float))
    return Z, pca


def plot_cluster_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    ax: plt.Axes,
    color_map: dict[int, Any] | None = None,
    s: float = 40.0,
    alpha: float = 0.85,
    edgecolor: str = "#777777",
    linewidth: float = 0.3,
    hide_axes: bool = True,
    title: str = "",
    pca_obj: PCA | None = None,
    Z: np.ndarray | None = None,
) -> plt.Axes:
    """Draw a PCA scatter colored by cluster labels onto *ax*.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Feature matrix (used for PCA if *Z* is None).
    labels : array-like, shape (n,)
        Cluster assignment per sample.
    ax : matplotlib Axes
        Target axes.
    color_map : dict or None
        Cluster-id -> color.  Auto-generated when None.
    s, alpha, edgecolor, linewidth : scatter aesthetics.
    hide_axes : bool
        Strip ticks / spines.
    title : str
        Subplot title.
    pca_obj : PCA or None
        Pre-fitted PCA (for axis labels).
    Z : array-like or None
        Pre-computed 2-D embedding.  Computed from *X* when None.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    if Z is None:
        Z, pca_obj = _pca_project(X)
    else:
        Z = np.asarray(Z)

    if color_map is None:
        color_map = _cluster_color_map(labels)

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

    ax.set_title(title, fontsize=13)
    if pca_obj is not None and not hide_axes:
        ax.set_xlabel(f"PC1 ({pca_obj.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_obj.explained_variance_ratio_[1] * 100:.1f}%)")

    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)

    return ax


def plot_baseline_best_lines(
    curves_df: pd.DataFrame,
    best_df: pd.DataFrame,
    *,
    ax: plt.Axes,
    colors: list[str] | None = None,
    marker: str = "o",
    linewidth: float = 1.8,
    title: str = "Baseline metrics",
    annotate: bool = True,
    grid_alpha: float = 0.22,
    normalize: bool = True,
) -> plt.Axes:
    """Plot one line per baseline metric (only the winning model).

    Parameters
    ----------
    curves_df : DataFrame
        Long-form output of ``baseline_metrics_over_k``
        (columns: metric, model, k, score, ari).
    best_df : DataFrame
        Best-row-per-metric output of ``baseline_metrics_over_k``
        (columns: metric, best_model, best_k, best_score).
    ax : matplotlib Axes
        Target axes.
    colors : list of hex strings or None
        One color per metric row in *best_df*.  Defaults to pink/purple/red/orange.
    marker, linewidth : line aesthetics.
    title : str
        Subplot title.
    annotate : bool
        Add a dot at the selected k.
    grid_alpha : float
        Y-axis grid opacity.
    normalize : bool
        If True, min-max scale each metric's scores to [0, 1].
    """
    if colors is None:
        n = len(best_df)
        colors = (BASELINE_WARM * ((n // len(BASELINE_WARM)) + 1))[:n]

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

        # --- min-max normalize to [0, 1] ---
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

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Score (normalized)" if normalize else "Score", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=grid_alpha)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.legend(
        fontsize=11,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    return ax


def _add_method_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``_method_id`` and ``_method_label`` columns to a CARVE results DataFrame.

    Replicates the grouping logic from ``carve._plotting``: every unique
    combination of non-metric, non-``n_clusters`` columns defines a "method".
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
    exclude = metric_cols | {"estimator", "n_clusters", "index", "_method_id", "_method_label"}
    group_cols = [c for c in df.columns if c not in exclude and c != "n_clusters"]

    df = df.copy()
    df[group_cols] = df[group_cols].fillna(_GROUPBY_NA)

    # build id and label per unique group
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

    df["_method_id"] = ids
    df["_method_label"] = labels
    return df


def plot_carve_best_line(
    carve_obj: Any,
    *,
    ax: plt.Axes,
    measure: str = "generalizability",
    rule: str = "1se",
    color: str = CARVE_GREEN,
    marker: str = "o",
    linewidth: float = 2.0,
    alpha_band: float = 0.18,
    title: str = "CARVE",
    annotate: bool = True,
    grid_alpha: float = 0.22,
    show_1se: bool = True,
) -> plt.Axes:
    """Plot only the winning CARVE line onto *ax*.

    Reads ``carve_obj.estimator_results_`` and filters to the best
    ``_method_id`` before drawing.

    Parameters
    ----------
    carve_obj : fitted CARVE instance
    ax : matplotlib Axes
    measure, rule : selection parameters
    color : hex color for the line
    marker, linewidth : aesthetics
    alpha_band : opacity for 1-SE band
    title : subplot title
    annotate : overlay a dot + annotation at the selected k
    grid_alpha : Y-grid opacity
    show_1se : draw 1-SE shaded band
    """
    from carve._selection import MEASURE_MAP, select_best_row_by_rule

    df = carve_obj.estimator_results_.copy()
    if "_method_id" not in df.columns or "_method_label" not in df.columns:
        df = _add_method_columns(df)

    # --- identify the winning method ---
    best_row = select_best_row_by_rule(df, measure=measure, rule=rule, return_idx=False)
    best_mid = str(best_row["_method_id"])
    best_k = int(best_row["n_clusters"])

    # --- filter to winning method only ---
    df_best = df[df["_method_id"] == best_mid].sort_values("n_clusters").copy()

    y_col = MEASURE_MAP[measure]
    se_col = f"{y_col}_se"
    has_se = se_col in df_best.columns

    x = df_best["n_clusters"].astype(int).to_numpy()
    y = df_best[y_col].astype(float).to_numpy()
    label = str(df_best["_method_label"].iloc[0])

    ax.plot(
        x, y, marker=marker, linewidth=linewidth, color=color, label=label, zorder=3
    )

    if show_1se and has_se:
        se = df_best[se_col].astype(float).to_numpy()
        lo, hi = y - se, y + se
        ax.fill_between(x, lo, hi, color=color, alpha=alpha_band, linewidth=0, zorder=1)

    # mark selected k
    if annotate:
        sel_y = float(best_row[y_col])
        ax.axvline(
            best_k, linestyle="--", linewidth=1.0, color=color, alpha=0.35, zorder=0
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
        ax.text(
            0.02,
            0.98,
            f"Selected: k*={best_k}\n{label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.9
            ),
            zorder=10,
        )

    ax.set_xlabel("k")
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="y", alpha=grid_alpha)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="best", frameon=False)
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
    annotate: bool = True,
    grid_alpha: float = 0.22,
    show_1se: bool = True,
) -> plt.Axes:
    """Plot multiple CARVE measure lines onto a single *ax*.

    Parameters
    ----------
    carve_obj : fitted CARVE instance
    ax : matplotlib Axes
    measures : list of (measure, rule) tuples.
        Default: ``[("generalizability", "1se"), ("stability", "quantile")]``.
    colors : dict mapping measure name -> hex color.
        Default: green for generalizability, blue for stability.
    marker, linewidth : aesthetics
    alpha_band : opacity for 1-SE band
    title : subplot title
    annotate : overlay a dot + annotation at each selected k
    grid_alpha : Y-grid opacity
    show_1se : draw 1-SE shaded band
    """
    from carve._selection import MEASURE_MAP, select_best_row_by_rule

    if measures is None:
        measures = [("generalizability", "1se"), ("stability", "quantile")]

    if colors is None:
        colors = CARVE_LINE_COLORS

    df = carve_obj.estimator_results_.copy()
    if "_method_id" not in df.columns or "_method_label" not in df.columns:
        df = _add_method_columns(df)

    annotations_text = []

    for measure, rule in measures:
        color = colors.get(measure, CARVE_GREEN)

        best_row = select_best_row_by_rule(
            df, measure=measure, rule=rule, return_idx=False
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

        if annotate:
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
            annotations_text.append(f"{pretty_measure}: k*={best_k}")

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

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("ARI", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=grid_alpha)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.legend(
        fontsize=11,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
    )
    return ax


def _rgba_to_plotly_hex(rgba_tuple) -> str:
    """Convert an RGBA tuple (0..1 floats) to a '#RRGGBB' hex string."""
    r, g, b = (
        int(rgba_tuple[0] * 255),
        int(rgba_tuple[1] * 255),
        int(rgba_tuple[2] * 255),
    )
    return f"#{r:02x}{g:02x}{b:02x}"


# ---- Matplotlib alluvial helpers ----


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
    """Draw a smooth S-curve band from [x0, y0] to [x1, y1].

    Uses a densely-sampled sigmoid to produce a visually smooth flow.
    """
    from matplotlib.patches import Polygon

    n_pts = 80
    xs = np.linspace(0, 1, n_pts)
    # smooth sigmoid interpolation
    t = 0.5 * (1 + np.tanh(6 * (xs - 0.5)))

    top_y = (1 - t) * y0_top + t * y1_top
    bot_y = (1 - t) * y0_bot + t * y1_bot
    x_vals = x0 + (x1 - x0) * xs

    # Build polygon: top curve forward, bottom curve backward
    verts = list(zip(x_vals, top_y)) + list(zip(x_vals[::-1], bot_y[::-1]))
    poly = Polygon(verts, closed=True, fc=color, ec="none", alpha=alpha, lw=0, zorder=1)
    ax.add_patch(poly)


def _draw_alluvial_mpl(
    ax,
    y_true,
    left_labels,
    right_labels,
    left_cmap: dict[int, Any],
    right_cmap: dict[int, Any],
    true_cmap: dict[str, Any],
    left_title: str = "CARVE",
    right_title: str = "Baseline",
    true_title: str = "True Label",
    link_alpha: float = 0.4,
    bar_width: float = 0.06,
    gap_frac: float = 0.015,
    font_size: int = 8,
):
    """Draw a 3-column alluvial diagram on a matplotlib axes.

    Colors are taken from the provided color maps so they align with existing
    scatter plots.
    """
    y_true_str = np.asarray(pd.Series(y_true).astype(str))
    left_int = np.asarray(pd.Series(left_labels).astype(int))
    right_int = np.asarray(pd.Series(right_labels).astype(int))
    n = len(y_true_str)

    true_order = _unique_in_order(y_true_str)
    left_order = sorted(np.unique(left_int))
    right_order = sorted(np.unique(right_int))

    left_sizes = [int(np.sum(left_int == k)) for k in left_order]
    true_sizes = [int(np.sum(y_true_str == lab)) for lab in true_order]
    right_sizes = [int(np.sum(right_int == k)) for k in right_order]

    x_L, x_T, x_R = 0.0, 0.5, 1.0
    hw = bar_width / 2

    left_pos = _stack_segments(left_sizes, gap_frac)
    true_pos = _stack_segments(true_sizes, gap_frac)
    right_pos = _stack_segments(right_sizes, gap_frac)

    # --- Compute purity for cluster nodes ---
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

    # --- Draw bars and labels ---
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

    # --- Cross-tabs ---
    ct_lt = pd.crosstab(pd.Series(left_int), pd.Series(y_true_str))
    ct_lt = ct_lt.reindex(index=left_order, columns=true_order, fill_value=0)

    ct_tr = pd.crosstab(pd.Series(y_true_str), pd.Series(right_int))
    ct_tr = ct_tr.reindex(index=true_order, columns=right_order, fill_value=0)

    left_used = {k: 0 for k in left_order}
    true_used_l = {lab: 0 for lab in true_order}
    true_used_r = {lab: 0 for lab in true_order}
    right_used = {k: 0 for k in right_order}

    # --- Left → True flows ---
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

    # --- True → Right flows ---
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

    # --- Column titles ---
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


def plot_composite_figure(
    X: np.ndarray,
    y: np.ndarray,
    carve_obj: Any,
    curves_df: pd.DataFrame,
    best_df: pd.DataFrame,
    silhouette_labels: np.ndarray,
    *,
    measure: str = "generalizability",
    rule: str = "1se",
    consensus_type: str = "stability",
    carve_measures: list[tuple[str, str]] | None = None,
    carve_line_colors: dict[str, str] | None = None,
    baseline_colors: list[str] | None = None,
    normalize_baseline: bool = True,
    figsize: tuple[float, float] = (18, 14),
    scatter_s: float = 30.0,
    scatter_alpha: float = 0.85,
    true_label_legend_title: str = "Stage",
    carve_title: str = "CARVE clustering",
    baseline_title: str = "Silhouette clustering",
    carve_line_title: str = "CARVE ARI over k",
    baseline_line_title: str = "Classic metrics over k",
    alluvial_left_title: str = "CARVE",
    alluvial_right_title: str = "Baseline",
    alluvial_true_title: str = "True Label",
    show_alluvial: bool = True,
    alluvial_link_alpha: float = 0.4,
    show_1se: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Build the composite paper figure.

    **Top row** (3 scatter plots):
        (A) PCA colored by true labels
        (B) PCA colored by CARVE consensus labels
        (C) PCA colored by baseline-selected labels

    **Middle row** (2 line plots):
        (D) CARVE metric-over-k (multiple measures)
        (E) Baseline metrics-over-k (normalized to [0,1])

    **Bottom row** (centered, optional):
        (F) Alluvial: CARVE → True → Baseline

    All cluster colors are aligned across scatter plots and alluvial.

    Parameters
    ----------
    X : array, shape (n, p)
    y : array-like, shape (n,)
    carve_obj : fitted CARVE instance
    curves_df, best_df : from ``baseline_metrics_over_k()``
    silhouette_labels : array-like, shape (n,)
    measure, rule, consensus_type : primary CARVE selection params
    carve_measures : list of (measure, rule) tuples for the CARVE line plot.
        Default: ``[("generalizability", "1se"), ("stability", "quantile")]``.
    carve_line_colors : dict mapping measure -> hex color.
    baseline_colors : list of hex strings for baseline metric lines.
    normalize_baseline : bool
    figsize : tuple
    scatter_s, scatter_alpha : scatter aesthetics
    true_label_legend_title : str
    carve_title, baseline_title : scatter subplot titles
    carve_line_title, baseline_line_title : line subplot titles
    alluvial_left_title, alluvial_right_title, alluvial_true_title : str
    show_alluvial : bool
    alluvial_link_alpha : float
    show_1se : bool
    save_path, dpi : save options

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    X = np.asarray(X)
    y_arr = np.asarray(y) if not isinstance(y, np.ndarray) else y
    silhouette_labels = np.asarray(silhouette_labels)

    # Get CARVE consensus labels
    carve_labels = carve_obj.get_labels(measure=measure, rule=rule, mode=consensus_type)
    carve_labels = np.asarray(carve_labels)

    # --- Align cluster labels to true labels via Hungarian ---
    from benchmarking_utils import align_labels as _align_labels

    y_codes = pd.Categorical(y_arr).codes
    carve_labels = _align_labels(y_codes, carve_labels)
    silhouette_labels = _align_labels(y_codes, silhouette_labels)

    # Shared PCA projection
    Z, pca_obj = _pca_project(X)

    # =============================================================
    # Build color maps
    # =============================================================
    # Independent color maps for each labeling (own palette per k)
    carve_cmap = _cluster_color_map(carve_labels)
    sil_cmap = _cluster_color_map(silhouette_labels)

    # True-label color map (matches plot_dim_red logic exactly)
    y_cat = pd.Categorical(y_arr)
    if hasattr(y_cat, "remove_unused_categories"):
        y_cat = y_cat.remove_unused_categories()
    n_true = len(y_cat.categories)
    true_palette = _get_color_mapping(n_true)
    true_cmap = {str(lab): true_palette[i] for i, lab in enumerate(y_cat.categories)}

    # =============================================================
    # Figure layout
    # =============================================================
    n_rows = 3 if show_alluvial else 2
    height_ratios = [1, 1.0, 0.7] if show_alluvial else [1, 1.0]

    fig_w, fig_h = figsize
    fig = plt.figure(figsize=(fig_w, fig_h * 1.15), constrained_layout=False)

    gs = fig.add_gridspec(
        n_rows,
        6,
        height_ratios=height_ratios,
        hspace=0.5,
        wspace=0.4,
    )

    # Manually tighten row 0→1 gap and widen row 1→2 gap
    # by adjusting subplot positions after creation
    ax_a = fig.add_subplot(gs[0, 0:2])  # true labels
    ax_b = fig.add_subplot(gs[0, 2:4])  # CARVE clusters
    ax_c = fig.add_subplot(gs[0, 4:6])  # baseline clusters
    ax_d = fig.add_subplot(gs[1, 0:3])  # CARVE lines
    ax_e = fig.add_subplot(gs[1, 3:6])  # baseline lines

    # Nudge scatter row down and lineplot row up to reduce their gap
    for ax in (ax_a, ax_b, ax_c):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - 0.02, box.width, box.height])
    for ax in (ax_d, ax_e):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + 0.01, box.width, box.height])

    # =============================================================
    # (A) PCA with true labels
    # =============================================================
    plot_dim_red(
        X,
        y=y_arr,
        ax=ax_a,
        show=False,
        title="True Labels",
        legend_title=true_label_legend_title,
        s=scatter_s,
        alpha=scatter_alpha,
        show_legend=False,
        hide_axes=True,
    )
    ax_a.set_title("True Labels", fontsize=13)
    # Strip PCA axis labels
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    # Build a custom horizontal legend below the plot
    handles = ax_a.collections  # scatter handles from plot_dim_red
    # Reconstruct labels from the true-label categories
    y_cat_leg = pd.Categorical(y_arr)
    if hasattr(y_cat_leg, "remove_unused_categories"):
        y_cat_leg = y_cat_leg.remove_unused_categories()
    leg_labels = [str(c) for c in y_cat_leg.categories]
    if len(handles) >= len(leg_labels):
        ax_a.legend(
            handles[: len(leg_labels)],
            leg_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=len(leg_labels),
            frameon=False,
            fontsize=11,
            columnspacing=1.0,
            handletextpad=0.4,
        )

    # =============================================================
    # (B) CARVE consensus clustering scatter
    # =============================================================
    plot_cluster_scatter(
        X,
        carve_labels,
        ax=ax_b,
        color_map=carve_cmap,
        s=scatter_s,
        alpha=scatter_alpha,
        title=carve_title,
        Z=Z,
        pca_obj=pca_obj,
    )

    # =============================================================
    # (C) Baseline clustering scatter
    # =============================================================
    plot_cluster_scatter(
        X,
        silhouette_labels,
        ax=ax_c,
        color_map=sil_cmap,
        s=scatter_s,
        alpha=scatter_alpha,
        title=baseline_title,
        Z=Z,
        pca_obj=pca_obj,
    )

    # =============================================================
    # (D) CARVE metric-over-k (multiple measures)
    # =============================================================
    plot_carve_best_lines(
        carve_obj,
        ax=ax_d,
        measures=carve_measures,
        colors=carve_line_colors,
        title=carve_line_title,
        show_1se=show_1se,
        annotate=True,
    )

    # =============================================================
    # (E) Baseline metrics (normalized)
    # =============================================================
    plot_baseline_best_lines(
        curves_df,
        best_df,
        ax=ax_e,
        colors=baseline_colors,
        title=baseline_line_title,
        normalize=normalize_baseline,
    )

    # --- Panel letter labels ---
    panel_axes = [(ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D"), (ax_e, "E")]

    # =============================================================
    # (F) Alluvial (matplotlib, centered in bottom row)
    # =============================================================
    if show_alluvial:
        ax_f = fig.add_subplot(gs[2, 1:5])
        _draw_alluvial_mpl(
            ax_f,
            y_true=y_arr,
            left_labels=carve_labels,
            right_labels=silhouette_labels,
            left_cmap=carve_cmap,
            right_cmap=sil_cmap,
            true_cmap=true_cmap,
            left_title=alluvial_left_title,
            right_title=alluvial_right_title,
            true_title=alluvial_true_title,
            link_alpha=alluvial_link_alpha,
        )
        panel_axes.append((ax_f, "F"))

    for ax, letter in panel_axes:
        # Panel F: place letter closer to content (alluvial has internal padding)
        x_off = 0.05 if letter == "F" else -0.05
        ax.text(
            x_off,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="right",
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_composite_figure_with_splits(
    X: np.ndarray,
    y: np.ndarray,
    carve_obj: Any,
    curves_df: pd.DataFrame,
    best_df: pd.DataFrame,
    silhouette_labels: np.ndarray,
    *,
    measure: str = "generalizability",
    rule: str = "1se",
    consensus_type: str = "stability",
    carve_measures: list[tuple[str, str]] | None = None,
    carve_line_colors: dict[str, str] | None = None,
    baseline_colors: list[str] | None = None,
    normalize_baseline: bool = True,
    figsize: tuple[float, float] = (18, 16),
    scatter_s: float = 30.0,
    scatter_alpha: float = 0.85,
    true_label_legend_title: str = "Cell Type",
    carve_title: str = "CARVE clustering",
    baseline_title: str = "Silhouette clustering",
    carve_line_title: str = "CARVE ARI over k",
    baseline_line_title: str = "Classic metrics (normalized)",
    split_title_a: str = "CARVE",
    split_title_b: str = "Silhouette",
    split_color_a: str | None = None,
    split_color_b: str | None = None,
    show_true_legend: bool = False,
    annotate_carve: bool = True,
    show_1se: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Composite paper figure with split-count bar charts in the bottom row.

    Identical to ``plot_composite_figure`` for the top two rows (scatter +
    line plots), but replaces the alluvial diagram with two side-by-side bar
    charts showing how many true labels are kept intact (1 cluster), split
    into 2 clusters, 3 clusters, etc.

    Parameters
    ----------
    split_title_a, split_title_b : str
        Titles for the CARVE and baseline bar-chart panels.
    split_color_a, split_color_b : str | None
        Bar colors. Defaults to CARVE green / warm pink.
    show_true_legend : bool
        Show a legend beneath the true-label scatter.  Default False
        (useful when the number of true classes is large).
    annotate_carve : bool
        Show vertical lines, dots, and a text annotation at the
        selected k in the CARVE line plot (panel D).  Default True.
    All other parameters are identical to ``plot_composite_figure``.
    """

    X = np.asarray(X)
    y_arr = np.asarray(y) if not isinstance(y, np.ndarray) else y
    silhouette_labels = np.asarray(silhouette_labels)

    # Get CARVE consensus labels
    carve_labels = carve_obj.get_labels(measure=measure, rule=rule, mode=consensus_type)
    carve_labels = np.asarray(carve_labels)

    # Align labels via Hungarian for scatter plots
    from benchmarking_utils import align_labels as _align_labels

    y_codes = pd.Categorical(y_arr).codes
    carve_labels = _align_labels(y_codes, carve_labels)
    silhouette_labels = _align_labels(y_codes, silhouette_labels)

    # Shared PCA projection
    Z, pca_obj = _pca_project(X)

    # Color maps
    carve_cmap = _cluster_color_map(carve_labels)
    sil_cmap = _cluster_color_map(silhouette_labels)

    y_cat = pd.Categorical(y_arr)
    if hasattr(y_cat, "remove_unused_categories"):
        y_cat = y_cat.remove_unused_categories()

    # =========  layout  =========
    height_ratios = [1, 1.0, 0.7]
    fig_w, fig_h = figsize
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=height_ratios,
        hspace=0.55,
        wspace=0.4,
    )

    ax_a = fig.add_subplot(gs[0, 0:2])  # true labels
    ax_b = fig.add_subplot(gs[0, 2:4])  # CARVE clusters
    ax_c = fig.add_subplot(gs[0, 4:6])  # baseline clusters
    ax_d = fig.add_subplot(gs[1, 0:3])  # CARVE lines
    ax_e = fig.add_subplot(gs[1, 3:6])  # baseline lines
    ax_f = fig.add_subplot(gs[2, 0:3])  # split bars CARVE
    ax_g = fig.add_subplot(gs[2, 3:6])  # split bars baseline

    # Nudge scatter row down and lineplot row up
    for ax in (ax_a, ax_b, ax_c):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - 0.02, box.width, box.height])
    for ax in (ax_d, ax_e):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + 0.01, box.width, box.height])

    # =========  Row 0: Scatter plots  =========
    # (A) True labels
    plot_dim_red(
        X,
        y=y_arr,
        ax=ax_a,
        show=False,
        title="True Labels",
        legend_title=true_label_legend_title,
        s=scatter_s,
        alpha=scatter_alpha,
        show_legend=False,
        hide_axes=True,
    )
    ax_a.set_title("True Labels", fontsize=13)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    if show_true_legend:
        handles = ax_a.collections
        y_cat_leg = pd.Categorical(y_arr)
        if hasattr(y_cat_leg, "remove_unused_categories"):
            y_cat_leg = y_cat_leg.remove_unused_categories()
        leg_labels = [str(c) for c in y_cat_leg.categories]
        if len(handles) >= len(leg_labels):
            ax_a.legend(
                handles[: len(leg_labels)],
                leg_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.06),
                ncol=min(len(leg_labels), 8),
                frameon=False,
                fontsize=8,
                columnspacing=0.8,
                handletextpad=0.3,
            )

    # (B) CARVE
    plot_cluster_scatter(
        X,
        carve_labels,
        ax=ax_b,
        color_map=carve_cmap,
        s=scatter_s,
        alpha=scatter_alpha,
        title=carve_title,
        Z=Z,
        pca_obj=pca_obj,
    )

    # (C) Baseline
    plot_cluster_scatter(
        X,
        silhouette_labels,
        ax=ax_c,
        color_map=sil_cmap,
        s=scatter_s,
        alpha=scatter_alpha,
        title=baseline_title,
        Z=Z,
        pca_obj=pca_obj,
    )

    # =========  Row 1: Line plots  =========
    # (D) CARVE
    plot_carve_best_lines(
        carve_obj,
        ax=ax_d,
        measures=carve_measures,
        colors=carve_line_colors,
        title=carve_line_title,
        show_1se=show_1se,
        annotate=annotate_carve,
    )

    # (E) Baseline
    plot_baseline_best_lines(
        curves_df,
        best_df,
        ax=ax_e,
        colors=baseline_colors,
        title=baseline_line_title,
        normalize=normalize_baseline,
    )

    # =========  Row 2: Split-count bar charts  =========
    def _count_splits(true_labels, cluster_labels):
        true_labels = np.asarray(true_labels)
        cluster_labels = np.asarray(cluster_labels)
        splits = []
        for lab in np.unique(true_labels):
            members = cluster_labels[true_labels == lab]
            splits.append(len(np.unique(members)))
        return np.asarray(splits, dtype=int)

    if split_color_a is None:
        split_color_a = CARVE_GREEN
    if split_color_b is None:
        split_color_b = BASELINE_WARM[0]

    splits_carve = _count_splits(y_arr, carve_labels)
    splits_sil = _count_splits(y_arr, silhouette_labels)

    # Shared x range for visual comparability
    all_split_vals = np.union1d(np.unique(splits_carve), np.unique(splits_sil))
    x_lo, x_hi = int(all_split_vals.min()), int(all_split_vals.max())
    all_x = np.arange(x_lo, x_hi + 1)

    def _plot_split_bars(ax, splits, title, color, all_x):
        values, counts = np.unique(splits, return_counts=True)
        count_map = dict(zip(values, counts))
        bar_counts = [count_map.get(v, 0) for v in all_x]

        ax.bar(all_x, bar_counts, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Number of clusters per true label", fontsize=11)
        ax.set_xticks(all_x)
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.22)
        # annotate counts on bars
        for xv, ct in zip(all_x, bar_counts):
            if ct > 0:
                ax.text(
                    xv,
                    ct + 0.3,
                    str(ct),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    _plot_split_bars(ax_f, splits_carve, split_title_a, split_color_a, all_x)
    ax_f.set_ylabel("Number of true labels", fontsize=11)

    _plot_split_bars(ax_g, splits_sil, split_title_b, split_color_b, all_x)
    ax_g.set_ylabel("")

    # Shared y limit
    y_max = max(ax_f.get_ylim()[1], ax_g.get_ylim()[1])
    ax_f.set_ylim(0, y_max * 1.12)
    ax_g.set_ylim(0, y_max * 1.12)

    # --- Panel letter labels ---
    panel_axes = [
        (ax_a, "A"),
        (ax_b, "B"),
        (ax_c, "C"),
        (ax_d, "D"),
        (ax_e, "E"),
        (ax_f, "F"),
        (ax_g, "G"),
    ]
    for ax, letter in panel_axes:
        ax.text(
            -0.05,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="right",
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def _build_alluvial_from_labels(
    y_true,
    left_labels,
    right_labels,
    left_color_map: dict[int, str],
    right_color_map: dict[int, str],
    left_title: str = "CARVE",
    right_title: str = "Baseline",
    true_title: str = "True Label",
    link_alpha: float = 0.35,
    node_pad: int = 20,
    node_thickness: int = 18,
    font_size: int = 14,
    height: int = 600,
    width: int = 1200,
    title_y: float = 1.08,
    vertical_margin: int = 100,
) -> go.Figure:
    """Build an alluvial (Sankey) diagram with cluster colors matching scatter plots.

    Standalone Plotly version — use ``plot_composite_figure`` with
    ``show_alluvial=True`` for the embedded matplotlib version.
    """
    y_true = pd.Series(y_true).astype(str).to_numpy()
    left_labels = pd.Series(left_labels).astype(int).to_numpy()
    right_labels = pd.Series(right_labels).astype(int).to_numpy()

    n = len(y_true)

    true_order = _unique_in_order(y_true)
    left_order = sorted(np.unique(left_labels))
    right_order = sorted(np.unique(right_labels))

    # True-label colors from tab10
    cmap_true = plt.get_cmap("tab10")
    true_colors = {
        lab: "#{:02x}{:02x}{:02x}".format(
            int(255 * cmap_true(i % cmap_true.N)[0]),
            int(255 * cmap_true(i % cmap_true.N)[1]),
            int(255 * cmap_true(i % cmap_true.N)[2]),
        )
        for i, lab in enumerate(true_order)
    }

    # Cluster stats helper
    def cluster_stats(pred):
        df = pd.DataFrame({"pred": pred, "true": y_true})
        ct = pd.crosstab(df["pred"], df["true"]).reindex(
            index=sorted(df["pred"].unique()), columns=true_order, fill_value=0
        )
        sizes = ct.sum(axis=1).to_numpy()
        shares = sizes / n
        dom_true = ct.idxmax(axis=1).to_numpy()
        purities = ct.max(axis=1).to_numpy() / np.maximum(sizes, 1)
        return ct, sizes, shares, dom_true, purities

    ct_left, _, share_left, dom_left, pur_left = cluster_stats(left_labels)
    ct_right, _, share_right, dom_right, pur_right = cluster_stats(right_labels)

    # Node labels & colors
    left_node_labels, left_node_colors = [], []
    for i, k in enumerate(left_order):
        left_node_labels.append(
            f"{k + 1}<br>{pur_left[i] * 100:.0f}%<br>{share_left[i] * 100:.0f}%"
        )
        left_node_colors.append(left_color_map.get(k, "#999999"))

    true_node_labels = list(true_order)
    true_node_colors = [true_colors[lab] for lab in true_order]

    right_node_labels, right_node_colors = [], []
    for i, k in enumerate(right_order):
        right_node_labels.append(
            f"{k + 1}<br>{pur_right[i] * 100:.0f}%<br>{share_right[i] * 100:.0f}%"
        )
        right_node_colors.append(right_color_map.get(k, "#999999"))

    nL, nT, nR = len(left_order), len(true_order), len(right_order)
    idx_left = {k: i for i, k in enumerate(left_order)}
    idx_true = {lab: nL + i for i, lab in enumerate(true_order)}
    idx_right = {k: nL + nT + i for i, k in enumerate(right_order)}

    sources, targets, values, colors = [], [], [], []

    # left → true
    for k in left_order:
        for lab in true_order:
            v = (
                int(ct_left.loc[k, lab])
                if (k in ct_left.index and lab in ct_left.columns)
                else 0
            )
            if v > 0:
                sources.append(idx_left[k])
                targets.append(idx_true[lab])
                values.append(v)
                colors.append(_hex_to_rgba(true_colors[lab], link_alpha))

    # true → right
    ct_tr = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(right_labels, name="pred"),
    ).reindex(index=true_order, columns=right_order, fill_value=0)
    for lab in true_order:
        for k in right_order:
            v = int(ct_tr.loc[lab, k])
            if v > 0:
                sources.append(idx_true[lab])
                targets.append(idx_right[k])
                values.append(v)
                colors.append(_hex_to_rgba(true_colors[lab], link_alpha))

    def col_positions(m, top=0.06, bottom=0.94):
        if m == 1:
            return [0.5]
        return list(np.linspace(top, bottom, m))

    x = [0.0] * nL + [0.5] * nT + [1.0] * nR
    y_pos = col_positions(nL) + col_positions(nT) + col_positions(nR)

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=node_pad,
                thickness=node_thickness,
                line=dict(color="rgba(0,0,0,0.25)", width=0.5),
                label=left_node_labels + true_node_labels + right_node_labels,
                color=left_node_colors + true_node_colors + right_node_colors,
                x=x,
                y=y_pos,
            ),
            link=dict(source=sources, target=targets, value=values, color=colors),
        )
    )

    fig.update_layout(
        font=dict(size=font_size, family="Arial"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=width,
        height=height,
        margin=dict(
            l=40, r=40, t=max(int(vertical_margin), 80), b=int(vertical_margin)
        ),
        annotations=[
            dict(
                x=0.0,
                y=title_y,
                xref="paper",
                yref="paper",
                text=left_title,
                showarrow=False,
                font=dict(size=font_size + 2),
            ),
            dict(
                x=0.5,
                y=title_y,
                xref="paper",
                yref="paper",
                text=true_title,
                showarrow=False,
                font=dict(size=font_size + 2),
            ),
            dict(
                x=1.0,
                y=title_y,
                xref="paper",
                yref="paper",
                text=right_title,
                showarrow=False,
                font=dict(size=font_size + 2),
            ),
        ],
    )
    return fig  # end _build_alluvial_from_labels


# ---------------------------------------------------------------------------
# ARI comparison helpers
# ---------------------------------------------------------------------------


def build_baseline_best_labels(
    X: np.ndarray,
    best_df: pd.DataFrame,
    model_grids: list[tuple[Any, dict[str, Any]]],
    metric: str = "silhouette",
    random_state: int = 42,
) -> tuple[np.ndarray, str, int]:
    """Reconstruct labels for a baseline metric's best (model, k).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    best_df : DataFrame returned by ``baseline_metrics_over_k``
    model_grids : estimator grids used during the baseline sweep
    metric : which baseline metric row to use (e.g. "silhouette", "gap")
    random_state : seed passed to the estimator

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    model_name : str  (pretty-printed model description)
    k : int           (selected number of clusters)

    Raises
    ------
    KeyError
        If *metric* is not found in *best_df*.
    """
    match_rows = best_df[best_df["metric"] == metric]
    if match_rows.empty:
        raise KeyError(
            f"Metric {metric!r} not found in best_df. "
            f"Available: {best_df['metric'].tolist()}"
        )
    row = match_rows.iloc[0]
    target_model = row["best_model"]
    target_k = int(row["best_k"])

    for est_cls, grid in model_grids:
        other_keys = [kk for kk in grid.keys() if kk != "n_clusters"]
        other_vals = [
            grid[kk]
            if isinstance(grid[kk], (list, tuple, np.ndarray))
            else [grid[kk]]
            for kk in other_keys
        ]
        combos = list(product(*other_vals)) if other_keys else [()]
        for combo in combos:
            fixed = (
                {kk: v for kk, v in zip(other_keys, combo)}
                if other_keys
                else {}
            )
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
) -> pd.DataFrame:
    """Build a comparison table of ARI-vs-ground-truth for each method.

    For baselines, the ARI at each metric's best (model, k) is read directly
    from *best_df* (already computed by ``baseline_metrics_over_k``).

    For CARVE, consensus labels are obtained via ``carve_obj.get_labels()``
    and ARI is computed against *y_true*.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    best_df : DataFrame
        Returned by ``baseline_metrics_over_k``.
    carve_obj : CARVE
        A fitted CARVE instance.
    X : array-like of shape (n_samples, n_features)
        Data matrix (needed by CARVE get_labels in some modes).
    carve_measures : list of (measure, rule) tuples, optional
        CARVE measure/rule combos to evaluate.
        Defaults to ``[("stability", "1se"), ("generalizability", "1se")]``.

    Returns
    -------
    DataFrame with columns: method, model, k, ari, source
    """
    if carve_measures is None:
        carve_measures = [("stability", "1se"), ("generalizability", "1se")]

    y_arr = np.asarray(y_true)
    result_rows: list[dict[str, Any]] = []

    # --- Baselines ---
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

    # --- CARVE ---
    for measure, rule in carve_measures:
        carve_k = carve_obj.get_k(measure=measure, rule=rule)
        carve_labels = carve_obj.get_labels(measure=measure, rule=rule)
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
# ARI comparison plots (three variants)
# ---------------------------------------------------------------------------

_CARVE_COLOR = "#009ADE"
_BASELINE_COLOR = "#FF1F5B"


def _ari_colors(df: pd.DataFrame) -> list[Any]:
    """Return a color list matching df rows: CARVE vs baseline."""
    return [
        _CARVE_COLOR if s == "carve" else _BASELINE_COLOR for s in df["source"]
    ]


def plot_ari_comparison_lollipop(
    ari_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 4),
    title: str = "Agreement with Ground Truth (ARI)",
    annotate_k: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Horizontal lollipop chart of ARI by method.

    Methods are ranked top-to-bottom by descending ARI.
    CARVE entries are colored blue, baselines pink.
    """
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

    # stems
    ax.hlines(y_pos, 0, df["ari"], colors=colors, linewidth=2.2, zorder=2)
    # dots
    ax.scatter(
        df["ari"],
        y_pos,
        c=colors,
        s=80,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["method"], fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("ARI", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    # best-ARI reference line
    if not df.empty:
        ax.axvline(
            df["ari"].max(),
            color="grey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    if annotate_k:
        for i, row in df.iterrows():
            ax.annotate(
                f"k={row['k']}",
                (row["ari"], i),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=8,
                va="center",
                color="0.35",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_ari_comparison_bar(
    ari_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 4),
    title: str = "Agreement with Ground Truth (ARI)",
    annotate_k: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Grouped vertical bar chart of ARI by method.

    CARVE entries are colored blue, baselines pink.
    """
    df = ari_df.dropna(subset=["ari"]).reset_index(drop=True)
    colors = _ari_colors(df)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_pos = np.arange(len(df))
    ax.bar(
        x_pos,
        df["ari"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.65,
        zorder=2,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df["method"], fontsize=9, rotation=35, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("ARI", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    # best-ARI reference line
    if not df.empty:
        ax.axhline(
            df["ari"].max(),
            color="grey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    if annotate_k:
        for i, row in df.iterrows():
            ax.annotate(
                f"k={row['k']}",
                (i, row["ari"]),
                textcoords="offset points",
                xytext=(0, 6),
                fontsize=8,
                ha="center",
                color="0.35",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_ari_comparison_dotplot(
    ari_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 4),
    title: str = "Agreement with Ground Truth (ARI)",
    annotate_k: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Forest-plot-style dot plot of ARI by method.

    Methods ranked top-to-bottom by descending ARI.
    CARVE entries shown as circles, baselines as diamonds.
    ARI value printed to the right of each dot.
    """
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

    for i, row in df.iterrows():
        marker = "o" if row["source"] == "carve" else "D"
        ax.scatter(
            row["ari"],
            i,
            c=[colors[i]],
            s=110,
            marker=marker,
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
        )
        # ARI value annotation
        label = f"{row['ari']:.3f}"
        if annotate_k:
            label += f"  (k={row['k']})"
        ax.annotate(
            label,
            (row["ari"], i),
            textcoords="offset points",
            xytext=(12, 0),
            fontsize=9,
            va="center",
            color="0.25",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["method"], fontsize=10)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("ARI", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    # best-ARI reference line
    if not df.empty:
        ax.axvline(
            df["ari"].max(),
            color="grey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
