from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from carve.sim import simulate_clusters

from benchmarking_simulation_helpers import simulate_scaling
from benchmarking_utils import gamma_quantile_approx


def plot_scaling_examples(
    regime: Dict,
    x_name: str,
    ks: np.ndarray = np.array([3, 4, 5, 6]),
    B: int = 10,
    random_state: int = 0,
    estimator_type: str = 'hierarchical',
    spectral_quant: float = 0.5,
    name_of_ex: str = 'Scaling Benchmark'
) -> None:
    """
    Visualizes scaling examples across cluster counts and a chosen scaling axis.

    Args:
        - regime (Dict): Simulation regime settings keyed by k.
        - x_name (str): Scaling axis name ('n_total', 'p', or 'embed_dim').
        - ks (np.ndarray): Array of cluster counts to visualize.
        - B (int): Number of replicates for baseline ARI estimation.
        - random_state (int): Seed for reproducible simulation and clustering.
        - estimator_type (str): Clustering algorithm type ('hierarchical', 'kmeans', or 'spectral').
        - spectral_quant (float): Quantile used for spectral gamma estimation (default: 0.5).
        - name_of_ex (str): Title for the figure.

    Returns:
        None. Displays the matplotlib figure.
    """
    if x_name == "n_total":
        x_values = [int(x) for x in np.logspace(np.log10(100), np.log10(10000), num=3)]
    elif x_name == "p":
        x_values = [int(x) for x in np.logspace(np.log10(10), np.log10(2500), num=3)]
    elif x_name == "embed_dim":
        x_values = [int(x) for x in np.logspace(np.log10(128), np.log10(2048), num=3)]
    else:
        raise ValueError("x_name must be 'n_total', 'p', or 'embed_dim'.")

    _, axes = plt.subplots(len(ks), len(x_values), figsize=(3 * len(x_values), 3 * len(ks)))
    for i, k in enumerate(ks):
        for j, x_value in enumerate(x_values):
            # Simulate dataset to be displayed
            X, y = simulate_scaling(
                regime=regime[k],
                true_k=k,
                x_name=x_name,
                x_value=x_value,
                seed=1 * 10 + j,
                base_random_state=random_state
            )

            # Compute baseline ARI over B replicates
            ari_arr = []
            for b in range(B):
                X_val, y_val = simulate_scaling(
                    regime=regime[k],
                    true_k=k,
                    x_name=x_name,
                    x_value=x_value,
                    seed=i * 100 + j * 10 + b,
                    base_random_state=random_state
                )

                if estimator_type == 'hierarchical':
                    estimator = AgglomerativeClustering(n_clusters=k)
                elif estimator_type == 'kmeans':
                    estimator = KMeans(n_clusters=k, random_state=random_state)
                elif estimator_type == 'spectral':
                    gamma = gamma_quantile_approx(X_val, q=spectral_quant)
                    estimator = SpectralClustering(n_clusters=k, affinity='rbf', gamma=gamma, random_state=random_state)
                else:
                    raise ValueError("estimator_type must be 'hierarchical' or 'spectral'")

                y_hat_val = estimator.fit_predict(X_val)
                ari = adjusted_rand_score(y_val, y_hat_val)
                ari_arr.append(ari)

            ari_mean = np.mean(ari_arr)

            # Plot PCA of dataset
            X_pca = PCA(n_components=2, random_state=0).fit_transform(X)
            ax = axes[i, j]
            ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=y, cmap="tab10", s=20, alpha=0.8, edgecolors="k"
            )
            ax.set_title(f"k={k}, {x_name}={x_value} | ARI={ari_mean:.3f}")
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"Scaling Benchmark: {name_of_ex}", fontsize=16)
    plt.tight_layout(); plt.show()


def plot_examples(
    settings_by_k: Dict, 
    other_settings: Dict, 
    ks: np.ndarray = np.array([3, 4, 5, 6]), 
    anchors: List[str] = ['easy', 'medium', 'difficult'], 
    B: int = 10, 
    estimator_type: str = 'kmeans', 
    spectral_quant: float = 0.5,
    name_of_ex: str = 'Uncorrelated Gaussians'
) -> None:
    """
    Visualizes example clustering datasets for different cluster counts and difficulty levels,
    and computes baseline clustering performance (ARI) for each configuration.

    Args:
        - settings_by_k (Dict): Simulation parameters for each k and difficulty anchor.
        - other_settings (Dict): Shared/global simulation parameters.
        - ks (np.ndarray): Array of cluster counts to visualize.
        - anchors (List[str]): List of difficulty levels.
        - B (int): Number of replicates for baseline ARI estimation.
        - estimator_type (str): Clustering algorithm type ('kmeans', 'agglomerative' or 'spectral').
        - spectral_quant (float): Quantile used for spectral gamma estimation (default: 0.5).
        - name_of_ex (str): Title for the figure.

    Returns:
        None. Displays the matplotlib figure.
    """
    
    _, axes = plt.subplots(len(ks), len(anchors), figsize=(12, 14))
    for i, k in enumerate(ks):
        for j, a in enumerate(anchors):
            
            # simulate dataset to be displayed
            X, y, _ = simulate_clusters(
                k=k,
                plotting=False,
                random_state=1*10 + j,
                **settings_by_k[k][a],
                **other_settings
            )
            
            # compute baseline ARI over B replicates
            ari_arr = []
            for b in range(B):
                X_val, y_val, _ = simulate_clusters(
                    k=k,
                    plotting=False,
                    random_state=1*100 + j*10 + b,
                    **settings_by_k[k][a],
                    **other_settings
                )
                
                if estimator_type == 'agglomerative':
                    estimator = AgglomerativeClustering(
                        n_clusters=k
                    )
                elif estimator_type == 'spectral':
                    gamma = gamma_quantile_approx(X_val, q=spectral_quant)
                    estimator = SpectralClustering(
                        n_clusters=k,
                        affinity='rbf',
                        gamma=gamma,
                        random_state=0
                    )
                else:
                    estimator = KMeans(
                        n_clusters=k,
                        random_state=0
                    )
                
                y_hat_val = estimator.fit_predict(X_val)
                ari = adjusted_rand_score(y_val, y_hat_val)
                ari_arr.append(ari)
                
            ari_mean = np.mean(ari_arr)

            # plot PCA of dataset
            X_pca = PCA(n_components=2, random_state=0).fit_transform(X)
            ax = axes[i, j]
            ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=y, cmap="tab10", s=20, alpha=0.8, edgecolors="k"
            )
            ax.set_title(f"k={k}, difficulty={a} | Baseline ARI={ari_mean:.3f}")
            ax.set_xticks([]); ax.set_yticks([])
            
    plt.suptitle(f"Example datasets: {name_of_ex}", fontsize=16)
    plt.tight_layout(); plt.show()
    
    
def plot_accuracy_vs_difficulty(
    results_df: pd.DataFrame, 
    show_se: bool = False
) -> None:
    """
    Plots the accuracy of clustering metrics as a function of problem difficulty.

    Args:
        - results_df (pd.DataFrame): DataFrame containing benchmarking results with columns
          ['difficulty', 'metric_name', 'is_optimal', 'is_correct', 'dataset_id'].
        - show_se (bool): If True, displays standard error bands around the mean accuracy.

    Returns:
        None. Displays a line plot of accuracy vs. difficulty for each metric.
    """
    accuracy_df = (
        results_df[results_df['is_optimal']]
            .groupby(['difficulty', 'metric_name', 'dataset_id'])
            .agg({'is_correct': 'mean'})
            .reset_index()
            .groupby(['difficulty', 'metric_name'])
            .agg({'is_correct': ['mean', 'std']})
            .reset_index()
    )
    
    # Plot
    plt.figure(figsize=(12, 8))
    for metric in accuracy_df['metric_name'].unique():
        subset = accuracy_df[accuracy_df['metric_name'] == metric]
        plt.plot(subset['difficulty'], subset[('is_correct', 'mean')], marker='o', label=metric)
        if show_se:
            plt.fill_between(
                subset['difficulty'],
                subset[('is_correct', 'mean')] - subset[('is_correct', 'std')],
                subset[('is_correct', 'mean')] + subset[('is_correct', 'std')],
                alpha=0.2
        )
    
    plt.xlabel('Difficulty (difficulty)')
    plt.ylabel('Accuracy')
    plt.title('Metric Accuracy vs. Problem Difficulty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    
def plot_metric_lineplots_grid(
    results_df: pd.DataFrame,
    difficulties: List[float],
    metrics_to_display: Optional[List[str]] = None,
    show_majority_votes: bool = True,
    figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plots a grid of line plots for clustering metrics across different difficulties and true_k values.

    Args:
        - results_df (pd.DataFrame): DataFrame containing clustering results with columns:
          ['true_k', 'k', 'difficulty', 'metric_name', 'metric_value', ...].
        - difficulties (List[float]): List of difficulty values to anchor the grid columns.
        - metrics_to_display (Optional[List[str]]): List of metric names to plot. If None, all available metrics are used.
        - show_majority_votes (bool): If True, displays majority vote markers (diamond, star, cross) for optimal k selection.
        - figsize (Optional[Tuple[float, float]]): Figure size for the plot. If None, size is determined automatically.

    Returns:
        None. Displays the matplotlib figure.
    """
    PAIRED = {
        "ari_stability_1se": "ari_stability",
        "ari_generalizability_1se": "ari_generalizability",
    }
    
    BASE_WITH_PAIR = set(PAIRED.values())  # metrics that have a 1se partner

    # numeric safety
    results_df = results_df.copy()
    if 'is_optimal' in results_df.columns:
        results_df['is_optimal'] = results_df['is_optimal'].astype(int)

    anchors = [difficulties[0], difficulties[len(difficulties)//2], difficulties[-1]]

    all_metrics = sorted(results_df['metric_name'].unique())
    if metrics_to_display is not None:
        metrics = [m for m in metrics_to_display if (m in all_metrics) and (m not in PAIRED.keys())]
        if not metrics:
            raise ValueError(f"None of the provided metrics {metrics_to_display} found in data")
    else:
        metrics = [m for m in all_metrics if m not in PAIRED.keys()]

    true_ks = sorted(results_df['true_k'].unique())

    if figsize is None:
        figsize = (3.5*len(anchors), 2.5*len(true_ks))

    fig, axes = plt.subplots(len(true_ks), len(anchors), figsize=figsize)

    if len(true_ks) == 1:
        axes = np.array([axes])
    if len(anchors) == 1:
        axes = np.array([axes]).T

    def difficulty_mask(series, a):
        return np.isclose(series.to_numpy(dtype=float), float(a), rtol=1e-6, atol=1e-8)

    for i, true_k in enumerate(true_ks):
        for j, difficulty in enumerate(anchors):
            ax = axes[i, j]
            line_y = {}

            # plot each (base) metric line
            for metric in metrics:
                subset_mask = (
                    (results_df['true_k'] == true_k) &
                    (results_df['metric_name'] == metric) &
                    difficulty_mask(results_df['difficulty'], difficulty)
                )
                subset = results_df.loc[subset_mask]
                if subset.empty:
                    continue

                avg_df = (
                    subset.groupby('k')['metric_value']
                        .agg(mean='mean', std='std')
                        .reset_index()
                )
                mean_values = avg_df['mean'].to_numpy()
                
                min_val = np.min(mean_values)
                max_val = np.max(mean_values)
                mean_values = (mean_values - min_val) / (max_val - min_val)

                ax.plot(avg_df['k'], mean_values, marker='o', label=metric)
                line_y[metric] = dict(zip(avg_df['k'].to_numpy(), mean_values))

                if show_majority_votes and 'is_optimal' in subset.columns and metric not in BASE_WITH_PAIR:
                    votes = subset.groupby('k')['is_optimal'].sum().sort_values(ascending=False)
                    if not votes.empty:
                        k_base_vote = int(votes[votes == votes.iloc[0]].index.max())  # if tie, then largest k
                        if k_base_vote in line_y[metric]:
                            y_on_curve = float(line_y[metric][k_base_vote])
                            ax.scatter(
                                [k_base_vote], [y_on_curve], marker='D', s=70,
                                edgecolors='black', facecolors='white',
                                zorder=5, label=None
                            )

            # handle paired markers with agreement logic
            if show_majority_votes:
                for excl_metric, base_metric in PAIRED.items():
                    excl_mask = (
                        (results_df['true_k'] == true_k) &
                        (results_df['metric_name'] == excl_metric) &
                        difficulty_mask(results_df['difficulty'], difficulty)
                    )
                    base_mask = (
                        (results_df['true_k'] == true_k) &
                        (results_df['metric_name'] == base_metric) &
                        difficulty_mask(results_df['difficulty'], difficulty)
                    )
                    excl_subset = results_df.loc[excl_mask]
                    base_subset = results_df.loc[base_mask]
                    if excl_subset.empty or base_subset.empty:
                        continue
                    if 'is_optimal' not in excl_subset.columns or 'is_optimal' not in base_subset.columns:
                        continue

                    # votes (consistent tie rule: largest k)
                    excl_votes = excl_subset.groupby('k')['is_optimal'].sum().sort_values(ascending=False)
                    base_votes = base_subset.groupby('k')['is_optimal'].sum().sort_values(ascending=False)
                    if excl_votes.empty or base_votes.empty:
                        continue

                    k_excl_vote = int(excl_votes[excl_votes == excl_votes.iloc[0]].index.max())
                    k_base_vote = int(base_votes[base_votes == base_votes.iloc[0]].index.max())

                    # y anchors from the base metric curve
                    y_map = line_y.get(base_metric, {})
                    if not y_map:
                        continue
                    # require the y for at least the locations we will plot
                    if (k_base_vote not in y_map) and (k_excl_vote not in y_map):
                        continue

                    if k_base_vote == k_excl_vote:
                        y_star = float(y_map.get(k_base_vote, list(y_map.values())[0]))
                        y_star += 0.02
                        ax.scatter([k_base_vote], [y_star], marker='*', s=220,
                                   facecolors='none', edgecolors='black',
                                   linewidths=2.0, zorder=8, label=None)
                    else:
                        # draw ◇ at base vote
                        if k_base_vote in y_map:
                            y_d = float(y_map[k_base_vote])
                            ax.scatter(
                                [k_base_vote], [y_d], marker='D', s=70,
                                edgecolors='black', facecolors='white',
                                zorder=6, label=None
                            )

                        if k_excl_vote in y_map:
                            y_x = float(y_map[k_excl_vote]) + 0.025
                            ax.scatter(
                                [k_excl_vote], [y_x], marker='x', s=90,
                                color='black', linewidths=2, zorder=7, label=None
                            )

            # guides / labels
            ax.axvline(true_k, color='gray', linestyle='--', alpha=0.5)
            if i == 0:
                ax.set_title(f'α = {difficulty:.3f}')
            if j == 0:
                ax.set_ylabel(f'true_k = {true_k}')
            ax.set_xlabel('k')
            ax.grid(True, alpha=0.3)

            k_values = results_df['k'].unique()
            ax.set_xlim(min(k_values) - 0.5, max(k_values) + 0.5)

    # legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if show_majority_votes:
        handles += [
            Line2D(
                [0],[0], marker='D', linestyle='None',
                markerfacecolor='white', markeredgecolor='black', label='max rule (vote)'
            ),
            Line2D(
                [0],[0], marker='x', linestyle='None',
                color='black', label='1se rule (vote)'
            ),
            Line2D(
                [0],[0], marker='*', linestyle='None',
                markerfacecolor='none', markeredgecolor='black', label='agree (vote)'
            ),
        ]
        labels += ['max rule (vote)', '1se rule (vote)', 'agree (vote)']

    fig.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
        fancybox=True, shadow=True, ncol=min(5, max(1, len(metrics)))
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def print_summary_stats(
    results_df: pd.DataFrame,
    group_col: str = "difficulty"
) -> None:
    """
    Prints a summary table of clustering metric accuracy statistics,
    reporting mean, standard deviation (SD), and standard error (SE) per metric and group_col.

    Args:
        - results_df (pd.DataFrame): DataFrame with columns [group_col, 'metric_name', 'is_optimal', 'is_correct', ...].
        - group_col (str): Column to group by (e.g., 'difficulty', 'p', 'n_total', 'embed_dim').

    Returns:
        None. Displays formatted accuracy statistics per metric and group_col, and overall ranking.
    """
    if group_col not in results_df.columns:
        raise ValueError(f"Column '{group_col}' not found in results_df.")

    df = results_df.loc[results_df['is_optimal']].copy()

    # Per-(group_col, metric) stats
    g = (
        df.groupby([group_col, 'metric_name'])['is_correct']
            .agg(mean='mean', std='std', n='count')
            .reset_index()
    )
    g['std'] = g['std'].fillna(0.0)
    g['se'] = g['std'] / g['n'].clip(lower=1).pow(0.5)

    # Mark best per group_col
    stars = (
        g.loc[g.groupby(group_col)['mean'].idxmax(), [group_col, 'metric_name']]
        .set_index([group_col, 'metric_name'])
        .assign(star='*')
    )

    # Format: mean (SD, SE)[*]
    def _fmt(row):
        key = (row[group_col], row['metric_name'])
        star = stars.loc[key, 'star'] if key in stars.index else ''
        return f"{row['mean']:.3f} ({row['std']:.3f}, {row['se']:.3f}){star}"

    g['cell'] = g.apply(_fmt, axis=1)
    groups = sorted(g[group_col].unique())

    # Compact grid
    grid = (
        g.pivot(index='metric_name', columns=group_col, values='cell')
         .reindex(columns=groups)
    )

    # Overall stats
    overall = (
        df.groupby('metric_name')['is_correct']
            .agg(mean='mean', std='std', n='count')
    )
    overall['std'] = overall['std'].fillna(0.0)
    overall['se'] = overall['std'] / overall['n'].clip(lower=1).pow(0.5)

    # Print
    print(f"\n=== Accuracy (mean (SD, SE)) by {group_col} ===")
    with pd.option_context('display.max_columns', None, 'display.width', 120):
        print(grid.fillna('').sort_index())

    print(f"\n======= Overall ranking (aggregated over all {group_col} & k) ========")
    ranked = overall.sort_values('mean', ascending=False)
    for i, (metric, r) in enumerate(ranked.iterrows(), start=1):
        print(f"{i:>2}. {metric:<35} {r['mean']:.3f} ({r['std']:.3f}, {r['se']:.3f})  (n={int(r['n'])})")