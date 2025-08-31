from __future__ import annotations
import warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

from ._selection import MEASURE_MAP, select_best_row, select_best_row_1se
from ._consensus import order_consensus_matrix

non_param_cols = {
    "n_clusters", 
    "ari_stability", "ari_stability_se", 
    "ari_generalizability", "ari_generalizability_se", 
    "consensus_pac_stability"
}

def plot_measure_vs_k(
    model_df: pd.DataFrame,
    *,
    measure: str = "stability",
    rule: str = "max",
    figsize: Tuple[int, int] = (10, 8)
) -> None:    
    y_col = MEASURE_MAP[measure]
    se_col = f"{y_col}_se"
    has_se = se_col in model_df.columns
        
    ylabel = "ARI (stability)" if measure == "stability" else "ARI (generalizability)"
    title = f"{measure} per estimator vs. k"

    _, ax = plt.subplots(figsize=figsize)

    group_cols = [c for c in model_df.columns if c not in non_param_cols]

    # pick best row according to rule
    best_row = select_best_row(model_df, measure=measure, return_idx=False) if rule == "max" else select_best_row_1se(model_df, measure=measure, return_idx=False)
    best_k = best_row["n_clusters"]

    # normalize keys for robust comparison (since NaN != NaN)
    def norm_key(values):
        return tuple(None if pd.isna(v) else v for v in values)

    best_key = norm_key([best_row[col] for col in group_cols])

    for keys, group in model_df.groupby(group_cols, dropna=False):
        group = group.sort_values("n_clusters")
        if not isinstance(keys, tuple):
            keys = (keys,)

        is_best_group = norm_key(keys) == best_key

        # build clean label: estimator shown plain; other params as k=v; skip NaN/None; round nums
        label_parts = []
        for col, val in zip(group_cols, keys):
            if pd.isna(val):
                continue
            if isinstance(val, (int, float)):
                val = f"{val:.4f}"
            if col == "estimator":
                label_parts.append(f"{val}")
            else:
                label_parts.append(f"{col}={val}")

        label = ", ".join(label_parts) if label_parts else "default"
        if is_best_group:
            label = f"{label} ★"   # star at the end

        ax.plot(group["n_clusters"], group[y_col], marker="o", label=label)

        if has_se:
            ax.fill_between(
                group["n_clusters"],
                group[y_col] - group[se_col],
                group[y_col] + group[se_col],
                alpha=0.2,
            )
    
    # vertical dashed line at best k
    ax.axvline(x=best_k, linestyle="--", alpha=0.7)

    ax.set_xlabel("n_clusters")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_consensus_matrix(
    model_df: pd.DataFrame,
    consensus_mats_raw: List[np.ndarray],
    *,
    measure: str = 'stability', 
    rule: str = 'max',
    k: int = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    if k is not None:  # subset dataframe if k specified
        model_df = model_df[model_df['n_clusters'] == k]

    # pick best method
    best_idx = select_best_row(model_df, measure=measure, return_idx=True) if rule == "max" else select_best_row_1se(model_df, measure=measure, return_idx=True)
    best_row = model_df.loc[best_idx]

    # get consensus matrix
    C = consensus_mats_raw[best_idx]
    C_ordered = order_consensus_matrix(C)
    
    optimal_k = best_row['n_clusters']

    # build title
    params = {
        key: best_row[key]
        for key in best_row.index
        if key not in (non_param_cols | {"estimator"}) and pd.notnull(best_row[key])
    }
    formatted_params = {}
    for key, value in params.items():
        if isinstance(value, (int, float)) and key != 'n_clusters':
            formatted_params[key] = f"{value:.4f}"
        else:
            formatted_params[key] = value
    param_str = ', '.join(f"{key} = {value}" for key, value in formatted_params.items())
    title = f"{best_row['estimator']} | k = {optimal_k}" + (f", {param_str}" if param_str else "")

    # plot
    plt.figure(figsize=figsize)
    plt.imshow(C_ordered, aspect='auto', interpolation='none')
    plt.colorbar(label='consensus')
    plt.title(title)
    plt.xlabel('samples (ordered)')
    plt.ylabel('samples (ordered)')
    plt.show()
    
def plot_clustering(
    X: np.ndarray,
    row: pd.Series,
    labels: np.ndarray,
    sample_level_measures: np.ndarray,
    *,
    measure: str = 'stability', 
    figsize: Tuple[int, int] = (20, 8),
    min_size: float = 20.0,
    max_size: float = 180.0, 
    min_alpha: float = 0.3, 
    max_alpha: float = 1.0
) -> None:
    if np.isclose(sample_level_measures.max(), sample_level_measures.min()):  # catch if all sample_level_measures are same
        # set uniform dot sizes and opacities
        sizes = np.full_like(sample_level_measures, np.mean([min_size, max_size]))
        alpha_map = {c: np.mean([min_alpha, max_alpha]) for c in np.unique(labels)}
    else:
        # normalize dot sizes
        sizes = min_size + (sample_level_measures - sample_level_measures.min()) / (sample_level_measures.max() - sample_level_measures.min()) * max_size
        
        # get cluster opacities
        clu_var = {c: sample_level_measures[labels == c].mean() for c in np.unique(labels)}
        v = np.array(list(clu_var.values()))
        mn, mx = v.min(), v.max()
        norm_var = {c: (clu_var[c] - mn) / (mx - mn + 1e-8) for c in clu_var}  # norm to [0, 1] 
        alpha_map = {
            c: min_alpha + norm_var[c] * (max_alpha - min_alpha)
            for c in norm_var
        }  # map to alpha in [min_alpha, max_alpha]
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for c in np.unique(labels):
        mask = labels == c
        ax1.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            s=sizes[mask],
            alpha=alpha_map[c],
            label=f"cluster {c}"
        )

    optimal_k = row['n_clusters']
    params = {}
    for key, value in row.items():
        if key in (non_param_cols | {"estimator"}) or pd.isna(value):
            continue
        if isinstance(value, (int, float)):
            value = f"{value:.4f}"
        params[key] = value

    param_str = ', '.join(f"{k} = {v}" for k, v in params.items())

    ax1.set_title(
        f"{row['estimator']} | k = {optimal_k}" +
        (f", {param_str}" if param_str else "")
    )
    ax1.set_xlabel("pc1")
    ax1.set_ylabel("pc2")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    data = [sample_level_measures[labels == c] for c in np.unique(labels)]
    ax2.boxplot(data, labels=[str(c) for c in np.unique(labels)])
    ax2.set_title("per-sample variance by cluster") if MEASURE_MAP[measure] == "ari_stability" else ax2.set_title("per-sample generalizability by cluster")
    ax2.set_xlabel("cluster")
    ax2.set_ylabel("sample variance") if MEASURE_MAP[measure] == "ari_stability" else ax2.set_ylabel("sample generalizability")

    plt.tight_layout()
    plt.show()    
    
def plot_cluster_stability(
    X: np.ndarray,
    results: List[Tuple],
    max_plots: int = 4,
    figsize: Tuple[int, int] = (10, 20)
) -> None:
    n_runs = min(len(results), max_plots)
    rng = np.random.RandomState()
    
    if len(results) > max_plots:
        selected_indices = rng.choice(len(results), size=max_plots, replace=False)
        selected_indices = sorted(selected_indices)
    else:
        selected_indices = range(len(results))
        
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_runs, 2, figure=fig)
    
    for i, b in enumerate(selected_indices):
        # Extract information from results
        ari_stab = results[b][0]
        labels_1 = results[b][2]
        labels_2 = results[b][5]
        P_1_idx = results[b][6]
        P_2_idx = results[b][8]
        
        # Find intersection between P_1_idx and P_2_idx
        _, i_1, i_2 = np.intersect1d(P_1_idx, P_2_idx, return_indices=True)
        
        # Extract data points for common samples
        X_common_1 = X[P_1_idx[i_1]]
        X_common_2 = X[P_2_idx[i_2]]
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_common_1_pca = pca.fit_transform(X_common_1)
        X_common_2_pca = pca.transform(X_common_2)
        
        # Get labels for common samples
        labels_1_common = labels_1[i_1]
        labels_2_common = labels_2[i_2]
        
        # Create subplots
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        
        # Plot first dataset
        ax1.scatter(
            X_common_1_pca[:, 0], X_common_1_pca[:, 1], 
            c=labels_1_common, cmap='tab10', alpha=0.7
        )
        ax1.set_title(f'Run {b+1}: X₁ Clustering')
        ax1.set_xlabel('PCA 1')
        ax1.set_ylabel('PCA 2')
        
        # Plot second dataset
        ax2.scatter(
            X_common_2_pca[:, 0], X_common_2_pca[:, 1], 
            c=labels_2_common, cmap='tab10', alpha=0.7
        )
        ax2.set_title(f'Run {b+1}: X₂ Clustering')
        ax2.set_xlabel('PCA 1')
        ax2.set_ylabel('PCA 2')
        
        # Add ARI text
        ax1.text(
            0.05, 0.95, f'ARI: {ari_stab:.3f}', transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.show()