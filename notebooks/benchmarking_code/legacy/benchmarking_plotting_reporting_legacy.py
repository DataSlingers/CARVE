from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from carve.sim import simulate_clusters

from .benchmarking_simulation_helpers import simulate_scaling
from .benchmarking_utils import gamma_quantile_approx, _wilson_ci


OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", 
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
]


def _cluster_color_map(labels: np.ndarray) -> dict[int, Any]:
    """okabe-ito for k<=7, tab20 for 8..20, hsv fallback."""
    labs = np.asarray(labels)
    uniq = sorted(int(x) for x in np.unique(labs) if x != -1)
    k = len(uniq)

    if k <= 7:
        cols = [mpl.colors.to_rgba(OKABE_ITO[i]) for i in range(k)]
    elif k <= 20:
        tab20 = plt.get_cmap("tab20")
        cols = [tab20(i) for i in range(k)]
    else:
        hsv = plt.get_cmap("hsv")
        cols = [hsv(i / max(k - 1, 1)) for i in range(k)]

    return {cid: cols[i] for i, cid in enumerate(uniq)}


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


def plot_examples(
    settings_by_k: Dict,
    other_settings: Dict,
    true_cluster_counts: np.ndarray = np.array([3, 4, 5, 6]),
    difficulty_levels: List[str] = ['easy', 'medium', 'hard'],
    n_seeds_per_dataset: int = 20,
    estimator_type: str = 'kmeans',
    spectral_quant: float = 0.5,
    example_title: str = 'Gaussian Mixtures', 
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
    _, axes = plt.subplots(len(true_cluster_counts), len(difficulty_levels), figsize=(12, 14))
    
    for j, difficulty_level in enumerate(difficulty_levels):
        for i, true_k in enumerate(true_cluster_counts): 
            
            # compute baseline ARI over B replicates
            ari_arr = []
            for seed in range(n_seeds_per_dataset):
                difficulty_level_seed = {"medium": 4, "hard": 9}.get(difficulty_level, 0)
                benchmark_seed = seed + ((true_k - min(true_cluster_counts)) * 100) + (difficulty_level_seed * 10000) + random_state
                
                X_val, y_val = simulate_clusters(
                    k=true_k,
                    plotting=False,
                    random_state=benchmark_seed,
                    **settings_by_k[true_k][difficulty_level],
                    **other_settings
                )
                
                if estimator_type == 'agglomerative':
                    estimator = AgglomerativeClustering(
                        n_clusters=true_k
                    )
                elif estimator_type == 'spectral':
                    gamma = gamma_quantile_approx(X_val, q=spectral_quant, random_state=benchmark_seed)
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
                
                y_hat_val = estimator.fit_predict(X_val)
                ari = adjusted_rand_score(y_val, y_hat_val)
                ari_arr.append(ari)
                
            ari_mean = np.mean(ari_arr)

            # plot PCA of dataset
            X_pca = PCA(n_components=2, random_state=0).fit_transform(X_val)
            ax = axes[i, j]
            ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=y_val, cmap="tab10", s=20, alpha=0.8, edgecolors="k"
            )
            ax.set_title(f"k={true_k}, difficulty={difficulty_level} | Baseline ARI={ari_mean:.3f}")
            ax.set_xticks([]); ax.set_yticks([])
            
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
        "ari_stability_1se",
        "ari_generalizability_1se",
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


# def plot_benchmark_snapshot(
#     *,
#     results_df: pd.DataFrame,
#     X: np.ndarray,
#     y: np.ndarray,
#     true_k: int,
#     difficulty_level: int,
#     dataset_iteration: int,
#     benchmark_seed: int,
#     labels_true: np.ndarray,
#     labels_baseline: np.ndarray,
#     labels_carve_stab_1se: np.ndarray,
#     labels_carve_gen_1se: np.ndarray,
#     labels_silhouette: np.ndarray,
#     labels_db: np.ndarray,
#     k_baseline: int,
#     k_carve_stab_1se: int,
#     k_carve_gen_1se: int,
#     k_silhouette: int,
#     k_db: int,
#     max_points: int = 4000,
# ) -> None:
#     """
#     Minimal live snapshot:
#       - 2x3 PCA scatter panels with ARI annotations
#       - 2 boxplots (so-far): correctness (0/1) and ARI of selected solutions
#     """

#     # --- PCA for display (deterministic; no RNG needed unless subsampling) ---
#     n = X.shape[0]
#     if (max_points is not None) and (n > max_points):
#         rng = np.random.default_rng(int(benchmark_seed))
#         idx = rng.choice(n, size=int(max_points), replace=False)
#     else:
#         idx = np.arange(n)

#     Xs = X[idx]
#     ys = y[idx]

#     Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(Xs)
#     Z = PCA(n_components=2, svd_solver="full").fit_transform(Xs)

#     def _safe_labels(labels, fallback):
#         try:
#             arr = np.asarray(labels)
#             if arr.ndim == 0 or arr.shape[0] != n:
#                 return np.asarray(fallback)
#             return arr
#         except Exception:
#             return np.asarray(fallback)

#     def _panel(ax, labels, title, k):
#         labels = _safe_labels(labels, labels_true)
#         ls = labels[idx]
#         ari = adjusted_rand_score(ys, ls)
#         ax.scatter(Z[:, 0], Z[:, 1], c=ls, s=8, linewidths=0, alpha=0.9, cmap="tab20")
#         ax.set_title(f"{title}\n(k={k}, ari={ari:.3f})", fontsize=10)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         for sp in ax.spines.values():
#             sp.set_alpha(0.2)

#     # --- layout ---
#     fig = plt.figure(figsize=(14, 9), dpi=120)
#     gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.85])

#     axs = [
#         fig.add_subplot(gs[0, 0:2]),
#         fig.add_subplot(gs[0, 2:4]),
#         fig.add_subplot(gs[0, 4:6]),
#         fig.add_subplot(gs[1, 0:2]),
#         fig.add_subplot(gs[1, 2:4]),
#         fig.add_subplot(gs[1, 4:6]),
#     ]

#     _panel(axs[0], labels_true, "true labels", true_k)
#     _panel(axs[1], labels_baseline, "oracle baseline", k_baseline)
#     _panel(axs[2], labels_carve_stab_1se, "carve 1se stability", k_carve_stab_1se)
#     _panel(axs[3], labels_carve_gen_1se, "carve 1se generalizability", k_carve_gen_1se)
#     _panel(axs[4], labels_silhouette, "silhouette", k_silhouette)
#     _panel(axs[5], labels_db, "davies-bouldin", k_db)

#     fig.suptitle(
#         f"snapshot · difficulty={difficulty_level} · iter={dataset_iteration} · true_k={true_k} · seed={benchmark_seed}",
#         fontsize=12,
#         y=0.98,
#     )

#     # --- so-far boxplots from results_df ---
#     ax_acc = fig.add_subplot(gs[2, 0:3])
#     ax_ari = fig.add_subplot(gs[2, 3:6])

#     if results_df is None or len(results_df) == 0:
#         ax_acc.axis("off")
#         ax_ari.axis("off")
#         fig.tight_layout(rect=[0, 0, 1, 0.96])
#         plt.show()
#         return

#     # select one row per (dataset, metric) where is_optimal=True
#     # (assumes your results rows include difficulty_level, dataset_iteration, true_k, metric_name)
#     sel = results_df.loc[results_df["is_optimal"] == True].copy()

#     # keep the specific metrics you care about in the boxplots
#     wanted = [
#         "ari_stability_1se",
#         "ari_generalizability_1se",
#         "silhouette",
#         "davies_bouldin",
#         "calinski_harabasz",
#         "gap",
#     ]
#     sel = sel[sel["metric_name"].isin(wanted)]

#     if len(sel) == 0:
#         ax_acc.axis("off")
#         ax_ari.axis("off")
#     else:
#         # group ordering
#         order = [m for m in wanted if m in set(sel["metric_name"])]

#         acc_data = [(sel.loc[sel["metric_name"] == m, "is_correct"].astype(float)).values for m in order]
#         ari_data = [(sel.loc[sel["metric_name"] == m, "measure_ari"].astype(float)).values for m in order]

#         ax_acc.boxplot(acc_data, labels=order, showmeans=True)
#         ax_acc.set_title("so-far: correctness (0/1)")
#         ax_acc.set_ylabel("is_correct")

#         ax_ari.boxplot(ari_data, labels=order, showmeans=True)
#         ax_ari.set_title("so-far: ari of selected solution")
#         ax_ari.set_ylabel("ari")

#         ax_acc.tick_params(axis="x", rotation=30)
#         ax_ari.tick_params(axis="x", rotation=30)

#     fig.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()


# def plot_scaling_examples(
#     regime: Dict,
#     axis_name: str,
#     cluster_counts: np.ndarray = np.array([3, 4, 5, 6]),
#     n_replicates: int = 10,
#     random_state: int = 0,
#     estimator_type: str = 'hierarchical',
#     spectral_quant: float = 0.5,
#     example_title: str = 'Scaling Benchmark'
# ) -> None:
#     """
#     Visualizes scaling examples across cluster counts and a chosen scaling axis.

#     Args:
#         - regime (Dict): Simulation regime settings keyed by k.
#         - axis_name (str): Scaling axis name ('n_total', 'p', or 'embed_dim').
#         - cluster_counts (np.ndarray): Array of cluster counts to visualize.
#         - n_replicates (int): Number of replicates for baseline ARI estimation.
#         - random_state (int): Seed for reproducible simulation and clustering.
#         - estimator_type (str): Clustering algorithm type ('hierarchical', 'kmeans', or 'spectral').
#         - spectral_quant (float): Quantile used for spectral gamma estimation (default: 0.5).
#         - example_title (str): Title for the figure.

#     Returns:
#         None. Displays the matplotlib figure.
#     """
#     if axis_name == "n_total":
#         x_values = [int(x) for x in np.logspace(np.log10(100), np.log10(10000), num=3)]
#     elif axis_name == "p":
#         x_values = [int(x) for x in np.logspace(np.log10(10), np.log10(2500), num=3)]
#     elif axis_name == "embed_dim":
#         x_values = [int(x) for x in np.logspace(np.log10(128), np.log10(2048), num=3)]
#     else:
#         raise ValueError("axis_name must be 'n_total', 'p', or 'embed_dim'.")

#     _, axes = plt.subplots(len(cluster_counts), len(x_values), figsize=(3 * len(x_values), 3 * len(cluster_counts)))
#     for i, k in enumerate(cluster_counts):
#         for j, x_value in enumerate(x_values):
#             # Simulate dataset to be displayed
#             X, y = simulate_scaling(
#                 regime=regime[k],
#                 true_cluster_count=k,
#                 axis_name=axis_name,
#                 axis_value=x_value,
#                 seed_offset=1 * 10 + j,
#                 base_random_state=random_state
#             )

#             # Compute baseline ARI over B replicates
#             ari_arr = []
#             for b in range(n_replicates):
#                 X_val, y_val = simulate_scaling(
#                     regime=regime[k],
#                     true_cluster_count=k,
#                     axis_name=axis_name,
#                     axis_value=x_value,
#                     seed_offset=i * 100 + j * 10 + b,
#                     base_random_state=random_state
#                 )

#                 if estimator_type == 'hierarchical':
#                     estimator = AgglomerativeClustering(n_clusters=k)
#                 elif estimator_type == 'kmeans':
#                     estimator = KMeans(n_clusters=k, random_state=random_state)
#                 elif estimator_type == 'spectral':
#                     gamma = gamma_quantile_approx(X_val, q=spectral_quant)
#                     estimator = SpectralClustering(n_clusters=k, affinity='rbf', gamma=gamma, random_state=random_state)
#                 else:
#                     raise ValueError("estimator_type must be 'hierarchical' or 'spectral'")

#                 y_hat_val = estimator.fit_predict(X_val)
#                 ari = adjusted_rand_score(y_val, y_hat_val)
#                 ari_arr.append(ari)

#             ari_mean = np.mean(ari_arr)

#             # Plot PCA of dataset
#             X_pca = PCA(n_components=2, random_state=0).fit_transform(X)
#             ax = axes[i, j]
#             ax.scatter(
#                 X_pca[:, 0], X_pca[:, 1],
#                 c=y, cmap="tab10", s=20, alpha=0.8, edgecolors="k"
#             )
#             ax.set_title(f"k={k}, {axis_name}={x_value} | ARI={ari_mean:.3f}")
#             ax.set_xticks([]); ax.set_yticks([])

#     plt.suptitle(f"Scaling Benchmark: {example_title}", fontsize=16)
#     plt.tight_layout(); plt.show()
    
    
# def plot_accuracy_vs_difficulty(
#     results_df: pd.DataFrame, 
#     show_se: bool = False
# ) -> None:
#     """
#     Plots the accuracy of clustering metrics as a function of problem difficulty.

#     Args:
#         - results_df (pd.DataFrame): DataFrame containing benchmarking results with columns
#           ['difficulty', 'metric_name', 'is_optimal', 'is_correct', 'dataset_id'].
#         - show_se (bool): If True, displays standard error bands around the mean accuracy.

#     Returns:
#         None. Displays a line plot of accuracy vs. difficulty for each metric.
#     """
#     accuracy_df = (
#         results_df[results_df['is_optimal']]
#             .groupby(['difficulty', 'metric_name', 'dataset_id'])
#             .agg({'is_correct': 'mean'})
#             .reset_index()
#             .groupby(['difficulty', 'metric_name'])
#             .agg({'is_correct': ['mean', 'std']})
#             .reset_index()
#     )
    
#     # Plot
#     plt.figure(figsize=(12, 8))
#     for metric in accuracy_df['metric_name'].unique():
#         subset = accuracy_df[accuracy_df['metric_name'] == metric]
#         plt.plot(subset['difficulty'], subset[('is_correct', 'mean')], marker='o', label=metric)
#         if show_se:
#             plt.fill_between(
#                 subset['difficulty'],
#                 subset[('is_correct', 'mean')] - subset[('is_correct', 'std')],
#                 subset[('is_correct', 'mean')] + subset[('is_correct', 'std')],
#                 alpha=0.2
#         )
    
#     plt.xlabel('Difficulty (difficulty)')
#     plt.ylabel('Accuracy')
#     plt.title('Metric Accuracy vs. Problem Difficulty')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
    
    
# def plot_metric_lineplots_grid(
#     results_df: pd.DataFrame,
#     difficulties: List[float],
#     metrics_to_display: Optional[List[str]] = None,
#     show_majority_votes: bool = True,
#     figsize: Optional[Tuple[float, float]] = None
# ) -> None:
#     """
#     Plots a grid of line plots for clustering metrics across different difficulties and true_k values.

#     Args:
#         - results_df (pd.DataFrame): DataFrame containing clustering results with columns:
#           ['true_k', 'k', 'difficulty', 'metric_name', 'metric_value', ...].
#         - difficulties (List[float]): List of difficulty values to anchor the grid columns.
#         - metrics_to_display (Optional[List[str]]): List of metric names to plot. If None, all available metrics are used.
#         - show_majority_votes (bool): If True, displays majority vote markers (diamond, star, cross) for optimal k selection.
#         - figsize (Optional[Tuple[float, float]]): Figure size for the plot. If None, size is determined automatically.

#     Returns:
#         None. Displays the matplotlib figure.
#     """
#     PAIRED = {
#         "ari_stability_1se": "ari_stability",
#         "ari_generalizability_1se": "ari_generalizability",
#     }
    
#     BASE_WITH_PAIR = set(PAIRED.values())  # metrics that have a 1se partner

#     # numeric safety
#     results_df = results_df.copy()
#     if 'is_optimal' in results_df.columns:
#         results_df['is_optimal'] = results_df['is_optimal'].astype(int)

#     anchors = [difficulties[0], difficulties[len(difficulties)//2], difficulties[-1]]

#     all_metrics = sorted(results_df['metric_name'].unique())
#     if metrics_to_display is not None:
#         metrics = [m for m in metrics_to_display if (m in all_metrics) and (m not in PAIRED.keys())]
#         if not metrics:
#             raise ValueError(f"None of the provided metrics {metrics_to_display} found in data")
#     else:
#         metrics = [m for m in all_metrics if m not in PAIRED.keys()]

#     true_ks = sorted(results_df['true_k'].unique())

#     if figsize is None:
#         figsize = (3.5*len(anchors), 2.5*len(true_ks))

#     fig, axes = plt.subplots(len(true_ks), len(anchors), figsize=figsize)

#     if len(true_ks) == 1:
#         axes = np.array([axes])
#     if len(anchors) == 1:
#         axes = np.array([axes]).T

#     def difficulty_mask(series, anchor_value):
#         return np.isclose(series.to_numpy(dtype=float), float(anchor_value), rtol=1e-6, atol=1e-8)

#     for i, true_k in enumerate(true_ks):
#         for j, difficulty in enumerate(anchors):
#             ax = axes[i, j]
#             line_y = {}

#             # plot each (base) metric line
#             for metric in metrics:
#                 subset_mask = (
#                     (results_df['true_k'] == true_k) &
#                     (results_df['metric_name'] == metric) &
#                     difficulty_mask(results_df['difficulty'], difficulty)
#                 )
#                 subset = results_df.loc[subset_mask]
#                 if subset.empty:
#                     continue

#                 avg_df = (
#                     subset.groupby('k')['metric_value']
#                         .agg(mean='mean', std='std')
#                         .reset_index()
#                 )
#                 mean_values = avg_df['mean'].to_numpy()
                
#                 min_val = np.min(mean_values)
#                 max_val = np.max(mean_values)
#                 mean_values = (mean_values - min_val) / (max_val - min_val)

#                 ax.plot(avg_df['k'], mean_values, marker='o', label=metric)
#                 line_y[metric] = dict(zip(avg_df['k'].to_numpy(), mean_values))

#                 if show_majority_votes and 'is_optimal' in subset.columns and metric not in BASE_WITH_PAIR:
#                     votes = subset.groupby('k')['is_optimal'].sum().sort_values(ascending=False)
#                     if not votes.empty:
#                         k_base_vote = int(votes[votes == votes.iloc[0]].index.max())  # if tie, then largest k
#                         if k_base_vote in line_y[metric]:
#                             y_on_curve = float(line_y[metric][k_base_vote])
#                             ax.scatter(
#                                 [k_base_vote], [y_on_curve], marker='D', s=70,
#                                 edgecolors='black', facecolors='white',
#                                 zorder=5, label=None
#                             )

#             # handle paired markers with agreement logic
#             if show_majority_votes:
#                 for excl_metric, base_metric in PAIRED.items():
#                     excl_mask = (
#                         (results_df['true_k'] == true_k) &
#                         (results_df['metric_name'] == excl_metric) &
#                         difficulty_mask(results_df['difficulty'], difficulty)
#                     )
#                     base_mask = (
#                         (results_df['true_k'] == true_k) &
#                         (results_df['metric_name'] == base_metric) &
#                         difficulty_mask(results_df['difficulty'], difficulty)
#                     )
#                     excl_subset = results_df.loc[excl_mask]
#                     base_subset = results_df.loc[base_mask]
#                     if excl_subset.empty or base_subset.empty:
#                         continue
#                     if 'is_optimal' not in excl_subset.columns or 'is_optimal' not in base_subset.columns:
#                         continue

#                     # votes (consistent tie rule: largest k)
#                     excl_votes = excl_subset.groupby('k')['is_optimal'].sum().sort_values(ascending=False)
#                     base_votes = base_subset.groupby('k')['is_optimal'].sum().sort_values(ascending=False)
#                     if excl_votes.empty or base_votes.empty:
#                         continue

#                     k_excl_vote = int(excl_votes[excl_votes == excl_votes.iloc[0]].index.max())
#                     k_base_vote = int(base_votes[base_votes == base_votes.iloc[0]].index.max())

#                     # y anchors from the base metric curve
#                     y_map = line_y.get(base_metric, {})
#                     if not y_map:
#                         continue
#                     # require the y for at least the locations we will plot
#                     if (k_base_vote not in y_map) and (k_excl_vote not in y_map):
#                         continue

#                     if k_base_vote == k_excl_vote:
#                         y_star = float(y_map.get(k_base_vote, list(y_map.values())[0]))
#                         y_star += 0.02
#                         ax.scatter([k_base_vote], [y_star], marker='*', s=220,
#                                    facecolors='none', edgecolors='black',
#                                    linewidths=2.0, zorder=8, label=None)
#                     else:
#                         # draw ◇ at base vote
#                         if k_base_vote in y_map:
#                             y_d = float(y_map[k_base_vote])
#                             ax.scatter(
#                                 [k_base_vote], [y_d], marker='D', s=70,
#                                 edgecolors='black', facecolors='white',
#                                 zorder=6, label=None
#                             )

#                         if k_excl_vote in y_map:
#                             y_x = float(y_map[k_excl_vote]) + 0.025
#                             ax.scatter(
#                                 [k_excl_vote], [y_x], marker='x', s=90,
#                                 color='black', linewidths=2, zorder=7, label=None
#                             )

#             # guides / labels
#             ax.axvline(true_k, color='gray', linestyle='--', alpha=0.5)
#             if i == 0:
#                 ax.set_title(f'α = {difficulty:.3f}')
#             if j == 0:
#                 ax.set_ylabel(f'true_k = {true_k}')
#             ax.set_xlabel('k')
#             ax.grid(True, alpha=0.3)

#             k_values = results_df['k'].unique()
#             ax.set_xlim(min(k_values) - 0.5, max(k_values) + 0.5)

#     # legend
#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     if show_majority_votes:
#         handles += [
#             Line2D(
#                 [0],[0], marker='D', linestyle='None',
#                 markerfacecolor='white', markeredgecolor='black', label='max rule (vote)'
#             ),
#             Line2D(
#                 [0],[0], marker='x', linestyle='None',
#                 color='black', label='1se rule (vote)'
#             ),
#             Line2D(
#                 [0],[0], marker='*', linestyle='None',
#                 markerfacecolor='none', markeredgecolor='black', label='agree (vote)'
#             ),
#         ]
#         labels += ['max rule (vote)', '1se rule (vote)', 'agree (vote)']

#     fig.legend(
#         handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
#         fancybox=True, shadow=True, ncol=min(5, max(1, len(metrics)))
#     )

#     plt.tight_layout(rect=[0, 0.05, 1, 0.97])
#     plt.subplots_adjust(wspace=0.3, hspace=0.3)
#     plt.show()


# def print_summary_stats(
#     results_df: pd.DataFrame,
#     group_col: str = "difficulty"
# ) -> None:
#     """
#     Prints a summary table of clustering metric accuracy statistics,
#     reporting mean, standard deviation (SD), and standard error (SE) per metric and group_col.

#     Args:
#         - results_df (pd.DataFrame): DataFrame with columns [group_col, 'metric_name', 'is_optimal', 'is_correct', ...].
#         - group_col (str): Column to group by (e.g., 'difficulty', 'p', 'n_total', 'embed_dim').

#     Returns:
#         None. Displays formatted accuracy statistics per metric and group_col, and overall ranking.
#     """
#     if group_col not in results_df.columns:
#         raise ValueError(f"Column '{group_col}' not found in results_df.")

#     df = results_df.loc[results_df['is_optimal']].copy()

#     # Per-(group_col, metric) stats
#     g = (
#         df.groupby([group_col, 'metric_name'])['is_correct']
#             .agg(mean='mean', std='std', n='count')
#             .reset_index()
#     )
#     g['std'] = g['std'].fillna(0.0)
#     g['se'] = g['std'] / g['n'].clip(lower=1).pow(0.5)

#     # Mark best per group_col
#     stars = (
#         g.loc[g.groupby(group_col)['mean'].idxmax(), [group_col, 'metric_name']]
#         .set_index([group_col, 'metric_name'])
#         .assign(star='*')
#     )

#     # Format: mean (SD, SE)[*]
#     def _fmt(row):
#         key = (row[group_col], row['metric_name'])
#         star = stars.loc[key, 'star'] if key in stars.index else ''
#         return f"{row['mean']:.3f} ({row['std']:.3f}, {row['se']:.3f}){star}"

#     g['cell'] = g.apply(_fmt, axis=1)
#     groups = sorted(g[group_col].unique())

#     # Compact grid
#     grid = (
#         g.pivot(index='metric_name', columns=group_col, values='cell')
#          .reindex(columns=groups)
#     )

#     # Overall stats
#     overall = (
#         df.groupby('metric_name')['is_correct']
#             .agg(mean='mean', std='std', n='count')
#     )
#     overall['std'] = overall['std'].fillna(0.0)
#     overall['se'] = overall['std'] / overall['n'].clip(lower=1).pow(0.5)

#     # Print
#     print(f"\n=== Accuracy (mean (SD, SE)) by {group_col} ===")
#     with pd.option_context('display.max_columns', None, 'display.width', 120):
#         print(grid.fillna('').sort_index())

#     print(f"\n======= Overall ranking (aggregated over all {group_col} & k) ========")
#     ranked = overall.sort_values('mean', ascending=False)
#     for i, (metric, r) in enumerate(ranked.iterrows(), start=1):
#         print(f"{i:>2}. {metric:<35} {r['mean']:.3f} ({r['std']:.3f}, {r['se']:.3f})  (n={int(r['n'])})")