from __future__ import annotations
from typing import Any, Dict, Iterable, List, Literal, Literal, Mapping, Sequence

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from carve.sim import simulate_clusters

from .benchmarking_utils import gamma_quantile_approx, _wilson_ci
import warnings


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
        hsv = plt.get_cmap("hsv")
        cols = [hsv(i / max(k - 1, 1)) for i in range(k)]
        
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
        "ari_stability_quant": "CARVE Stability (95% Quantile)",
        "ari_generalizability_1se": "CARVE Generalizability (1se)",
        "silhouette": "Silhouette",
        "gap": "Gap Statistic",
        "davies_bouldin": "Davies–Bouldin",
        "calinski_harabasz": "Calinski–Harabasz",
        "misclassification_generalizability": "Misclassification (Global)",
    }
    return pretty.get(metric, metric)

def _plotting_iter(
    settings_by_k: Dict,
    other_settings: Dict,
    j: int,
    level: Any,
    level_label: str,
    true_k: int,
    true_cluster_counts: np.ndarray,
    estimator_type: str,
    spectral_quant: float,
    sampler: Literal['default', 'scaling'],
    seed: int,
    random_state: int,
):
    level_seed = {1: 4, 2: 9}.get(j, 0)
    benchmark_seed = seed + ((true_k - min(true_cluster_counts)) * 100) + (level_seed * 10000) + random_state
    
    if sampler == 'default':
        X, y = simulate_clusters(
            k=true_k,
            plotting=False,
            random_state=benchmark_seed,
            **settings_by_k[true_k][level],
            **other_settings
        )
    elif sampler == 'scaling':
        if level_label not in {"n_total", "p", "embed_dim"}:
            raise ValueError("level_label must be 'n_total', 'p', or 'embed_dim'")

        n_total = int(other_settings.get("n_total", 500))
        p = int(other_settings.get("p", 50))
        embed_dim = int(other_settings.get("embed_dim", 64))

        if level_label == "n_total":
            n_total = int(level)
        elif level_label == "p":
            p = int(level)
        elif level_label == "embed_dim":
            embed_dim = int(level)
        else:
            raise ValueError("level_label must be 'n_total', 'p', or 'embed_dim'")

        dict_key = {1: 'middle', 2: 'end'}.get(j, 'start')
        
        X, y = simulate_clusters(
            n_total=n_total,
            p=p,
            embed_dim=embed_dim,
            k=true_k,
            plotting=False,
            random_state=random_state,
            **settings_by_k[true_k][dict_key],
            **other_settings
        )
    
    if estimator_type == 'agglomerative':
        estimator = AgglomerativeClustering(
            n_clusters=true_k
        )
    elif estimator_type == 'spectral':
        gamma = gamma_quantile_approx(X, q=spectral_quant, random_state=benchmark_seed)
        estimator = SpectralClustering(
            n_clusters=true_k,
            affinity='rbf',
            gamma=gamma,
            random_state=benchmark_seed
        )
    else:
        estimator = KMeans(
            n_clusters=true_k,
            n_init=10,
            random_state=benchmark_seed
        )
    
    y_hat = estimator.fit_predict(X)
    ari = adjusted_rand_score(y, y_hat)

    return ari, X, y
    

# --- Main plotting functions ---
def plot_examples(
    settings_by_k: Dict,
    other_settings: Dict,
    true_cluster_counts: np.ndarray = np.array([3, 4, 5, 6]),
    level_label: str = "difficulty",
    levels: List[Any] = ['easy', 'medium', 'hard'],
    n_seeds_per_dataset: int = 20,
    estimator_type: str = 'kmeans',
    spectral_quant: float = 0.5,
    example_title: str = 'Gaussian Mixtures', 
    sampler: Literal['default', 'scaling'] = 'default',
    n_jobs: int = 1,
    random_state: int = 0
) -> None:
    """
    Visualizes example clustering datasets for different cluster counts and difficulty levels,
    and computes baseline clustering performance (ARI) for each configuration.

    Args:
        - settings_by_k (Dict): Simulation parameters for each k and difficulty anchor.
        - other_settings (Dict): Shared/global simulation parameters.
        - true_cluster_counts (np.ndarray): Array of cluster counts to visualize.
        - difficulty_levels (List[str]): List of difficulty levels.
        - n_seeds_per_dataset (int): Number of replicates for baseline ARI estimation.
        - estimator_type (str): Clustering algorithm type ('kmeans', 'agglomerative' or 'spectral').
        - spectral_quant (float): Quantile used for spectral gamma estimation (default: 0.5).
        - example_title (str): Title for the figure.
        - random_state (int): Seed for reproducibility.

    Returns:
        None. Displays the matplotlib figure.
    """
    _, axes = plt.subplots(len(true_cluster_counts), len(levels), figsize=(12, 14))
    
    if level_label == 'difficulty':
        levels = ['easy', 'medium', 'hard']
    elif level_label == "n_total":
        levels = [int(x) for x in np.logspace(np.log10(100), np.log10(10000), num=3)]
    elif level_label == "p":
        levels = [int(x) for x in np.logspace(np.log10(10), np.log10(2500), num=3)]
    elif level_label == "embed_dim":
        levels = [int(x) for x in np.logspace(np.log10(10), np.log10(2500), num=3)]
    else:
        raise ValueError("level_label must be 'difficulty', 'n_total', 'p', or 'embed_dim'.")
    
    for j, level in enumerate(levels):
        for i, true_k in enumerate(true_cluster_counts): 
            
            worker = delayed(_plotting_iter)
            results = Parallel(n_jobs=n_jobs)(
                worker(
                    settings_by_k=settings_by_k,
                    other_settings=other_settings,
                    j=j,
                    level=level,
                    level_label=level_label,
                    true_k=true_k,
                    true_cluster_counts=true_cluster_counts,
                    estimator_type=estimator_type,
                    spectral_quant=spectral_quant,
                    sampler=sampler,
                    seed=seed,
                    random_state=random_state,
                )
                for seed in range(n_seeds_per_dataset)
            )
            ari_arr = [res[0] for res in results]
            _, X, y = results[0]
                
            ari_mean = np.mean(ari_arr)

            # plot PCA of dataset
            X_pca = PCA(n_components=2, random_state=0).fit_transform(X)
            
            # get labels and colors
            labels = np.asarray(y)
            cmap = _cluster_color_map(labels)
            colors = [cmap[int(lbl)] for lbl in labels]
            
            # plot scatter plot
            ax = axes[i, j]
            ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=colors, s=10, alpha=0.8
            )
            
            # titles, legends, &c.
            ax.set_title(f"k={true_k}, {level_label}={level} | Baseline ARI={ari_mean:.3f}")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_alpha(0.1)
            
    plt.suptitle(f"Example datasets: {example_title}", fontsize=16)
    plt.tight_layout(); plt.show()


def plot_benchmark_snapshot(
    *,
    X: np.ndarray,
    results_df: pd.DataFrame,
    plotting_dict: Mapping[str, Mapping[str, Any]],
    true_labels: np.ndarray,
    baseline_labels: np.ndarray,
    baseline_ari: float,
    panel_metrics: Sequence[str] = (
        "ari_generalizability_1se", 
        "ari_stability_quant", 
        "silhouette",
        "davies_bouldin",
    ),
    figsize_pca: tuple[float, float] = (10.5, 6.8),
    figsize_summary: tuple[float, float] = (10.5, 3.6),
) -> tuple[mpl.figure.Figure, mpl.figure.Figure]:
    """
    returns (fig_pca, fig_summary)

    expected plotting_dict entries:
      plotting_dict[key] = {"k": int, "labels": array, "ari": float, ...}
    """
    X = np.asarray(X)
    y = np.asarray(true_labels)
    base_lab = np.asarray(baseline_labels)
    metric_keys = list(panel_metrics)

    # PCA projection
    Z = PCA(n_components=2, random_state=0).fit_transform(X)

    # Build 2x3 PCA panel figure
    fig_pca, axs = plt.subplots(2, 3, figsize=figsize_pca, constrained_layout=True)
    axs = np.asarray(axs).reshape(2, 3)

    k_true = len(np.unique(y))
    _scatter_clusters(
        axs[0, 0], Z, y,
        title="true labels",
        subtitle=f"k={k_true}"
    )

    _scatter_clusters(
        axs[0, 1], Z, base_lab,
        title="baseline",
        subtitle=f"ari={baseline_ari:.3f}  |  k={len(np.unique(base_lab))}"
    )

    targets = [
        (axs[0, 2], metric_keys[0]),
        (axs[1, 0], metric_keys[1]),
        (axs[1, 1], metric_keys[2]),
        (axs[1, 2], metric_keys[3]),
    ]

    for ax, key in targets:
        info = plotting_dict.get(key, None)
        if info is None:
            ax.axis("off")
            ax.set_title(f"missing: {key}", fontsize=10)
            continue

        labs = np.asarray(info["labels"])
        k_sel = int(info.get("k", len(np.unique(labs))))
        ari_sel = float(info.get("ari", np.nan))

        _scatter_clusters(
            ax, Z, labs,
            title=_pretty_metric_name(key),
            subtitle=f"k={k_sel}  |  ari={ari_sel:.3f}"
        )

    # --- Summary figure: accuracy+wilson + ari boxplots
    needed = {"metric_name", "is_optimal", "is_correct", "metric_ari", "baseline_ari"}
    missing = sorted(c for c in needed if c not in results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {missing}")

    df_opt = results_df.loc[results_df["is_optimal"]].copy()

    # Order: CARVE metrics > externals
    accuracy_metrics = [
        "ari_stability_1se", "ari_generalizability_1se", 
        "silhouette", "gap", "davies_bouldin", "calinski_harabasz",
    ]

    # Accuracy + wilson
    rows = []
    for m in accuracy_metrics:
        sub = df_opt.loc[df_opt["metric_name"] == m]
        n = int(len(sub))
        k_succ = int(sub["is_correct"].sum())
        acc = k_succ / n if n > 0 else np.nan
        lo, hi = _wilson_ci(k_succ, n) if n > 0 else (np.nan, np.nan)
        rows.append((m, acc, lo, hi, n))

    acc_df = pd.DataFrame(rows, columns=["metric", "acc", "lo", "hi", "n"])

    # Baseline ARI distribution: one per dataset instance (dedupe)
    dedupe_cols = [c for c in ["difficulty_level", "dataset_iteration", "true_k"] if c in results_df.columns]
    if dedupe_cols:
        base_ari = results_df.drop_duplicates(dedupe_cols)["baseline_ari"].astype(float).values
    else:
        base_ari = results_df["baseline_ari"].dropna().astype(float).unique()

    # ARI-by-metric distribution (optimal only)
    ari_data = [df_opt.loc[df_opt["metric_name"] == m, "metric_ari"].astype(float).values for m in accuracy_metrics]
    ari_labels = [_pretty_metric_name(m) for m in accuracy_metrics]

    # Include baseline as first box
    ari_data = [base_ari] + ari_data
    ari_labels = ["baseline"] + ari_labels

    fig_sum, ax = plt.subplots(1, 2, figsize=figsize_summary, constrained_layout=True)

    # Left: accuracy points + wilson intervals
    x = np.arange(len(acc_df))
    ax0 = ax[0]
    ax0.vlines(x, acc_df["lo"], acc_df["hi"], linewidth=2)
    ax0.scatter(x, acc_df["acc"], s=35)
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_xticks(x)
    ax0.set_xticklabels([_pretty_metric_name(m) for m in acc_df["metric"]], rotation=35, ha="right")
    ax0.set_ylabel("p(k^hat = k*)")
    ax0.set_title("k* Recovery (Wilson 95% CI)")

    # Right: boxplot of ari for selected solutions
    ax1 = ax[1]
    ax1.boxplot(ari_data, showfliers=False)
    ax1.set_xticks(np.arange(1, len(ari_labels) + 1))
    ax1.set_xticklabels(ari_labels, rotation=35, ha="right")
    ax1.set_ylabel("ARI(y, y^hat)")
    ax1.set_title("ARI : true labels (for k^hat)")

    return fig_pca, fig_sum


def plot_ari_over_difficulty(
    results_df: pd.DataFrame,
    *,
    metrics=("ari_generalizability_1se", "ari_stability_quant", "silhouette", "gap", "davies_bouldin", "calinski_harabasz"),
    center: str = "mean",
    band: tuple = (0.05, 0.95),
    show_band_for=("baseline", "ari_stability_1se", "ari_generalizability_1se"),
    x_col: str | None = None,
    x_label: str = "Signal/Noise Ratio",
    xscale: str = "linear",
    title: str | None = None,
    figsize: tuple = (12, 10),
    ax=None,
    ylim: tuple | str = "auto",
):
    """
    Plots ARI vs. difficulty level for selected metrics.

    Args:
        - results_df (pd.DataFrame): DataFrame with benchmarking results.
        - metrics (tuple): Metrics to plot.
        - center (str): Central tendency measure ('mean' or 'median').
        - band (tuple): Quantiles for uncertainty band.
        - show_band_for (tuple): Metrics for which to show uncertainty band.
        - title (str | None): Plot title.
        - ax: Matplotlib axis.
        - ylim (tuple | str): Y-axis limits.
    """
    axis_col, axis_name_col = _infer_axis_cols(results_df) if x_col is None else (x_col, "")
    needed = {axis_col, "metric_name", "is_optimal", "metric_ari", "baseline_ari"}
    missing = sorted(c for c in needed if c not in results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {missing}")
    
    present = [m for m in metrics if (results_df["metric_name"] == m).any()]
    if len(present) == 0:
        raise ValueError("none of the requested metrics were found in results_df")

    # --- 1) Filter to one row per (dataset instance, metric): selected "optimal" k --- 
    df_opt = results_df.loc[results_df["is_optimal"] == True].copy()
    
    # --- 2) Baseline: dedupe per dataset instance (we only want one per (dataset instance, true_k, difficulty_level)) --- 
    dedupe_cols = [c for c in [axis_col, "dataset_iteration", "true_k"] if c in results_df.columns]
    base = results_df.drop_duplicates(dedupe_cols)[[axis_col, "baseline_ari"]].copy()

    # Helper to get quantiles and average/median
    def _summarize(y: pd.Series):
        y = y.astype(float).dropna().values
        if y.size == 0:
            return np.nan, np.nan, np.nan
        lo = np.quantile(y, band[0])
        hi = np.quantile(y, band[1])
        mid = np.mean(y) if center == "mean" else np.median(y)
        return mid, lo, hi
    
    # --- 3) Build summary tables ---
    base_sum = (
        base.groupby(axis_col)["baseline_ari"].apply(lambda s: pd.Series(_summarize(s), index=["mid", "lo", "hi"]))
        .unstack()
        .reset_index()
        .sort_values(axis_col)
    )  # cols: difficulty_level, mid, lo, hi
    
    sums = []
    for m in metrics:
        sub = df_opt.loc[df_opt["metric_name"] == m]
        s = (
            sub.groupby(axis_col)["metric_ari"]
            .apply(lambda s: pd.Series(_summarize(s), index=["mid", "lo", "hi"]))
            .unstack()
            .reset_index()
            .sort_values(axis_col)
        )
        s["metric_name"] = m
        sums.append(s)
        
    sum_df = pd.concat(sums, ignore_index=True) if sums else pd.DataFrame()  # cols: difficulty_level, mid, lo, hi, metric_name
    
    # --- 4) Plot ---
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
        
    x = base_sum[axis_col].astype(int).values  # if throw error, uncomment next line
    # x = pd.to_numeric(base_sum[axis_col], errors="coerce").values
    
    # Plot baseline line and (optional) CI band
    baseline_color = (0.35, 0.35, 0.35, 1.0)
    ax.plot(x, base_sum['mid'].values, color=baseline_color, linestyle='--', linewidth=2.6, label='Baseline (Oracle k*)')
    if "baseline" in show_band_for:
        ax.fill_between(x, base_sum["lo"].values, base_sum["hi"].values, color=baseline_color, alpha=0.15)
        
    # Ensure metrics are actually present 
    present_metrics = []
    for m in metrics:
        if (sum_df["metric_name"] == m).any():
            present_metrics.append(m)

    color_map = _metric_color_map(present_metrics)
        
    # Plot metrics
    for m in metrics:
        s = sum_df.loc[sum_df["metric_name"] == m]  # Subset df to metric
        if s.empty:
            continue
            
        # Get values
        xx = s[axis_col].astype(int).values  # if throw error, uncomment next line
        # xx = pd.to_numeric(s[axis_col], errors="coerce").values
        yy = s['mid'].values
        
        # Plot line and (optional) band
        color = color_map[m]
        ax.plot(xx, yy, color=color, linewidth=2.2 if 'ari_' in m else 1.6, label=_pretty_metric_name(m))
        if m in show_band_for:
            ax.fill_between(xx, s["lo"].values, s["hi"].values, color=color, alpha=0.12)
            
    ax.set_xlabel(x_label)
    ax.set_ylabel("ARI (selected k^hat vs. true labels)")
    ax.set_xlim(x.min(), x.max())
    if ylim != "auto":
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(auto=True)
    ax.set_xticks(np.unique(x))
    
    if xscale in {"log", "linear"}:
        ax.set_xscale(xscale)
    
    if title is not None:
        ax.set_title(title)

    ax.legend(frameon=False, fontsize=9, ncol=2)

    return fig


def plot_baseline_vs_metric_ari_grid(
    results_df: pd.DataFrame,
    *,
    metrics=("ari_generalizability_1se", "ari_stability_quant", "silhouette", "gap", "davies_bouldin", "calinski_harabasz"),
    ncols: int = 3,
    figsize=(12, 10),
    center: str = "mean",
    point_alpha: float = 0.33,
    point_size: float = 12.0,
    rasterize_points: bool = True,
    color_by_difficulty: bool = True,
    difficulty_cmap: str = "cividis",
    show_difficulty_colorbar: bool = True,
    invert_x: bool = True,
    xlim=(0.0, 1.02),
    ylim=(0.0, 1.02),
):
    """
    Plots baseline ARI vs. metric ARI scatter plots in a grid layout.

    Args:
        - results_df (pd.DataFrame): DataFrame with benchmarking results.
        - metrics (tuple): Metrics to plot.
        - ncols (int): Number of columns in the grid.
        - figsize (tuple): Figure size.
        - center (str): Central tendency measure ('mean' or 'median').
        - point_alpha (float): Transparency of scatter points.
        - point_size (float): Size of scatter points.
        - rasterize_points (bool): Whether to rasterize scatter points.
        - xlim (tuple): X-axis limits.
        - ylim (tuple): Y-axis limits.
    """
    needed = {"metric_name", "is_optimal", "metric_ari", "baseline_ari"}
    missing = sorted(c for c in needed if c not in results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {missing}")
    
    present = [m for m in metrics if (results_df["metric_name"] == m).any()]
    if len(present) == 0:
        raise ValueError("none of the requested metrics were found in results_df")

    color_map = _metric_color_map(present)  # Get colors
    
    # --- 1) Filter to one row per (dataset instance, metric): selected "optimal" k --- 
    df_opt = results_df.loc[results_df["is_optimal"] == True].copy()
    
    axis_col, axis_name_col = _infer_axis_cols(df_opt)
    group_col = "difficulty_level" if "difficulty_level" in df_opt.columns else axis_col
    
    # Difficulty color mapping
    levels = None
    diff_color = None
    diff_cmap = None
    diff_norm = None
    if color_by_difficulty and group_col in df_opt.columns:
        levels = sorted(df_opt[group_col].dropna().unique())
        diff_cmap = plt.get_cmap(difficulty_cmap, len(levels))
        diff_color = {lvl: diff_cmap(i) for i, lvl in enumerate(levels)}
        diff_norm = mpl.colors.BoundaryNorm(
            boundaries=np.arange(-0.5, len(levels) + 0.5, 1.0),
            ncolors=len(levels),
        )
    
    # --- 2) Setup grid ---
    n = len(present)  # Compute grid dims
    nrows = int(np.ceil(n / ncols))
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True
    )
    axes = np.asarray(axes).flatten()
    
    baseline_color = (0.35, 0.35, 0.35, 1.0)
    diag_x = np.array([xlim[0], xlim[1]])  # Diagonal reference line
    
    # --- 3) Plot ---
    for i, m in enumerate(present):
        ax = axes[i]
        sub = df_opt.loc[df_opt["metric_name"] == m]  # Subset df to metric
        
        # 1) Scatter actual ARIs (Baseline ARI : Metric ARI)
        if color_by_difficulty and diff_color is not None:
            for lvl in levels:
                pts = sub.loc[sub[group_col] == lvl]
                if pts.empty:
                    continue
                ax.scatter(
                    pts["baseline_ari"].astype(float).values,
                    pts["metric_ari"].astype(float).values,
                    s=point_size,
                    alpha=point_alpha,
                    rasterized=rasterize_points,
                    color=diff_color[lvl], 
                    linewidths=0,
                )
        else:
            ax.scatter(
                sub["baseline_ari"].astype(float).values,
                sub["metric_ari"].astype(float).values,
                s=point_size,
                alpha=point_alpha,
                rasterized=rasterize_points,
                color=baseline_color, 
                linewidths=0,
            )

        # 2) Diagonal reference line
        ax.plot(diag_x, diag_x, linestyle="--", linewidth=1.0, color=baseline_color)
    
        # 3) Difficulty path overlay (median/mean per difficulty, connected)
        if group_col in sub.columns:
            g = sub.groupby(group_col)[["baseline_ari", "metric_ari"]]
            if center == "mean":
                path = g.mean()
            elif center == "median":
                path = g.median()
            else:
                raise ValueError("center must be 'mean' or 'median'")

            # Enforce consistent ordering of difficulty steps
            path = path.reindex(levels).dropna()

            xy = path[["baseline_ari", "metric_ari"]].values
            lvls = list(path.index)

            if color_by_difficulty and diff_color is not None and len(xy) >= 2:  # Colored segments (for difficulty)
                segments = np.stack([xy[:-1], xy[1:]], axis=1)  # (n-1, 2, 2)
                seg_colors = [diff_color[lvl] for lvl in lvls[:-1]]
                lc = LineCollection(segments, colors=seg_colors, linewidths=2.2, zorder=3)
                ax.add_collection(lc)
                
                ax.scatter(  # Colored markers at each difficulty point
                    xy[:, 0], xy[:, 1],
                    s=40,
                    color=[diff_color[lvl] for lvl in lvls],
                    edgecolor="none",
                    zorder=4,
                )
            else:
                # Fallback: single-color overlay
                ax.plot(
                    xy[:, 0], xy[:, 1],
                    marker="o",
                    linewidth=2.2,
                    color=color_map[m],
                    zorder=3,
                )
            
        # 4) Compact panel annotation: median loss to oracle
        if center == "mean":
            loss = (sub["baseline_ari"].astype(float) - sub["metric_ari"].astype(float)).mean()
        else:
            loss = (sub["baseline_ari"].astype(float) - sub["metric_ari"].astype(float)).median()
        ax.set_title(f"{_pretty_metric_name(m)}  ({center} Δ={loss:.3f})", fontsize=10)

        # Handle axes
        if invert_x:
            ax.set_xlim(xlim[1], xlim[0])
        else:
            ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")

    # Remove unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Labels
    for ax in axes[:n]:
        ax.label_outer()
        
    # Difficulty colorbar
    if color_by_difficulty and show_difficulty_colorbar and diff_cmap is not None:
        sm = mpl.cm.ScalarMappable(cmap=diff_cmap, norm=diff_norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axes[:n],
            fraction=0.03,
            pad=0.02,
        )
        if group_col == 'difficulty ladder':
            cbar.set_label("difficulty ladder", rotation=90)
        else:
            cbar.set_label(group_col, rotation=90)
        tick_pos = np.arange(len(levels))
        cbar.set_ticks(tick_pos)
        cbar.set_ticklabels([str(lvl) for lvl in levels])

    fig.supxlabel("Baseline ARI (Oracle k*)")
    fig.supylabel("ARI (selected k^hat)")

    return fig