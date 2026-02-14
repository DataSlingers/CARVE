from __future__ import annotations
from typing import Any, Iterable, List, Literal

from itertools import product

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

import glasbey

from umap import UMAP

from benchmarking_utils import gamma_quantile_approx, LeidenClustering, align_labels, _build_estimator
from benchmarking_metrics import calculate_metric


# --- Setup and basic handlers ---
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", 
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
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
        cols = [mpl.colors.to_rgba(OKABE_ITO[i]) for i in range(k)]
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


def _scatter_clusters(ax, Z: np.ndarray, labels: np.ndarray, title: str, subtitle: str = ""):
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
        "misclassification_generalizability": "Misclassification (Global)",
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
    method: Literal['pca', 'tsne', 'umap'] = 'pca',
    label_col: str = 'label', 
    s: int = 50,
    alpha: float = 0.8,
    linewidth: float = 0.1,
    hide_axes: bool = True,
    figsize: tuple[float, float] = (12, 10),
    title: str = 'PCA of raw cell counts (colored by stage label)', 
    show_legend: bool = True,
    legend_title: str = 'stage',
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
        
    if method == 'pca':
         # Get PCs
        pca = PCA(n_components=2, random_state=0)
        pcs = pca.fit_transform(X)
    elif method == 'tsne':
        tsne = TSNE(n_components=2, random_state=0)
        pcs = tsne.fit_transform(X)
    elif method == 'umap':
        umap = UMAP(n_components=2, random_state=0)
        pcs = umap.fit_transform(X)
    else:
        raise ValueError(f"unsupported method: {method}")

    # Construct data frame
    pc_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
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
            sel['PC1'], sel['PC2'],
            label=lab, color=color_map[lab],
            s=s, alpha=alpha, edgecolor='k', linewidths=linewidth
        )

    # Set title, axis labels, legend
    ax.set_title(title)
    if method == 'pca':
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
    elif method == 'tsne':
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    elif method == 'umap':
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    
    if show_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc='upper left')
    
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
    label_col: str = 'label', 
    spectral_quant: float = 0.5,
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
    agg_w = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agg_w_labels = agg_w.fit_predict(X_arr)
    ari_agg_w = adjusted_rand_score(y_arr, agg_w_labels)

    # Agglomerative (single linkage)
    agg_s = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    agg_s_labels = agg_s.fit_predict(X_arr)
    ari_agg_s = adjusted_rand_score(y_arr, agg_s_labels)

    # Leiden
    leiden = LeidenClustering(n_clusters=n_clusters, random_state=random_state)
    leiden_labels = leiden.fit_predict(X_arr)
    ari_leiden = adjusted_rand_score(y_arr, leiden_labels)

    # Spectral Clustering (median heuristic for gamma)
    gamma = gamma_quantile_approx(X_arr, q=spectral_quant, random_state=random_state)
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=gamma, random_state=random_state)
    spectral_labels = spectral.fit_predict(X_arr)
    ari_spectral = adjusted_rand_score(y_arr, spectral_labels)
    
    # Get PCs
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X_arr)

    # Construct data frame
    pc_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    pc_df[label_col] = y_arr
    
    # Plot PCA with labels 
    _, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)

    y_cat = pd.Categorical(y_arr)
    if hasattr(y_cat, "remove_unused_categories"):
        y_cat = y_cat.remove_unused_categories()
    y_codes = y_cat.codes
    y_code_colors = _cluster_color_map(y_codes)
    y_color_map = {lab: y_code_colors[code] for lab, code in zip(y_cat.categories, range(len(y_cat.categories)))}

    # Plot true labels
    ax_true = axes[0, 0]
    for lab in y_cat.categories:
        sel = pc_df[label_col] == lab
        ax_true.scatter(
            pc_df.loc[sel, 'PC1'],
            pc_df.loc[sel, 'PC2'],
            s=s, alpha=alpha, edgecolor='k', linewidth=linewidth,
            color=y_color_map[lab],
            label=lab
        )
    ax_true.set_title("True Cell Type Labels")
    
    if show_legend:
        ax_true.legend(markerscale=1.5, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    titles = [
        f"KMeans (ARI={ari_kmeans:.3f})",
        f"Agglomerative (Ward) (ARI={ari_agg_w:.3f})",
        f"Agglomerative (Single Linkage) (ARI={ari_agg_s:.3f})",
        f"Leiden (ARI={ari_leiden:.3f})",
        f"Spectral (ARI={ari_spectral:.3f})"
    ]
    clusterings = [kmeans_labels, agg_w_labels, agg_s_labels, leiden_labels, spectral_labels]
    aligned_clusterings = [align_labels(pd.Categorical(y).codes, labels) for labels in clusterings]

    for ax, labels, title in zip(axes.flat[1:], aligned_clusterings, titles):
        cluster_color_map = _cluster_color_map(labels)
        for cluster in np.unique(labels):
            sel = labels == cluster
            ax.scatter(
                pc_df.loc[sel, 'PC1'],
                pc_df.loc[sel, 'PC2'],
                s=s, alpha=alpha, edgecolor='k', linewidth=linewidth,
                color=cluster_color_map[int(cluster)],
                label=f"Cluster {cluster}"
            )
        ax.set_title(title)
        
        if show_legend:
            ax.legend(markerscale=1.5, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
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
        other_vals = [grid[k] if isinstance(grid[k], (list, tuple, np.ndarray)) else [grid[k]] for k in other_keys]
        combos = list(product(*other_vals)) if other_keys else [()]

        for combo in combos:
            fixed_params = {k: v for k, v in zip(other_keys, combo)} if other_keys else {}
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
                {"metric": metric, "best_model": None, "best_k": np.nan, "best_score": np.nan, "best_ari": np.nan}
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
                    print(f"  - {m}: k={bk}, score={bs:.{decimals}f}, ari={ba:.{decimals}f}   [{bm}]")
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
        print(f"Change in intact labels (split into 1 cluster): {summ_a['n_intact'] - summ_b['n_intact']}")
        print(f"Change in intact label percentage: {summ_a['pct_intact'] - summ_b['pct_intact']:.2f} percentage points")
        print(f"Change in average # of clusters per true label: {summ_a['mean_splits'] - summ_b['mean_splits']:.3f}")
        print(f"Change in max # of clusters for any true label: {summ_a['max_splits'] - summ_b['max_splits']}")

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
    true_colors = {lab: "#{:02x}{:02x}{:02x}".format(
        int(255*cmap(i % cmap.N)[0]),
        int(255*cmap(i % cmap.N)[1]),
        int(255*cmap(i % cmap.N)[2]),
    ) for i, lab in enumerate(true_order)}

    # --- helper: compute purity + share and dominant true label per predicted cluster ---
    def cluster_stats(pred):
        df = pd.DataFrame({"pred": pred, "true": y_true})
        ct = pd.crosstab(df["pred"], df["true"]).reindex(index=sorted(df["pred"].unique()), columns=true_order, fill_value=0)
        sizes = ct.sum(axis=1).to_numpy()
        shares = sizes / n
        dom_true = ct.idxmax(axis=1).to_numpy()
        purities = (ct.max(axis=1).to_numpy() / np.maximum(sizes, 1))
        return ct, sizes, shares, dom_true, purities

    ct_left, _, share_left, dom_left, pur_left = cluster_stats(left_labels)
    ct_right, _, share_right, dom_right, pur_right = cluster_stats(right_labels)

    # --- node labels (match screenshot style: big purity %, then share %) ---
    left_node_labels = []
    left_node_colors = []
    for i, k in enumerate(left_order):
        purity = pur_left[i]
        share = share_left[i]
        left_node_labels.append(f"{k+1}<br>{purity*100:.0f}%<br>{share*100:.0f}%")
        left_node_colors.append(true_colors[dom_left[i]])

    true_node_labels = [lab for lab in true_order]
    true_node_colors = [true_colors[lab] for lab in true_order]

    right_node_labels = []
    right_node_colors = []
    for i, k in enumerate(right_order):
        purity = pur_right[i]
        share = share_right[i]
        right_node_labels.append(f"{k+1}<br>{purity*100:.0f}%<br>{share*100:.0f}%")
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
            v = int(ct_left.loc[k, lab]) if (k in ct_left.index and lab in ct_left.columns) else 0
            if v > 0:
                sources.append(idx_left[k])
                targets.append(idx_true[lab])
                values.append(v)
                colors.append(_hex_to_rgba(true_colors[lab], link_alpha))

    # true -> right
    ct_tr = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(right_labels, name="pred")).reindex(index=true_order, columns=right_order, fill_value=0)
    for lab in true_order:
        for k in right_order:
            v = int(ct_tr.loc[lab, k])
            if v > 0:
                sources.append(idx_true[lab])
                targets.append(idx_right[k])
                values.append(v)
                colors.append(_hex_to_rgba(true_colors[lab], link_alpha))

    # --- fixed 3-column layout (x positions), evenly spaced y positions per column ---
    def col_positions(m, top=0.02, bottom=0.98):
        if m == 1:
            return [0.5]
        return list(np.linspace(top, bottom, m))

    x = ([0.0]*nL) + ([0.5]*nT) + ([1.0]*nR)
    y = (col_positions(nL)) + (col_positions(nT)) + (col_positions(nR))

    fig = go.Figure(go.Sankey(
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
        )
    ))

    fig.update_layout(
        font=dict(size=font_size, family="Arial"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=width,
        height=height,
        margin=dict(l=40, r=40, t=60, b=20),
        annotations=[
            dict(x=0.0, y=1.08, xref="paper", yref="paper", text=left_title, showarrow=False, font=dict(size=font_size+2)),
            dict(x=0.5, y=1.08, xref="paper", yref="paper", text=true_title, showarrow=False, font=dict(size=font_size+2)),
            dict(x=1.0, y=1.08, xref="paper", yref="paper", text=right_title, showarrow=False, font=dict(size=font_size+2)),
        ],
    )
    return fig
