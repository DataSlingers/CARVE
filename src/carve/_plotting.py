from __future__ import annotations
import warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._selection import MEASURE_MAP, select_best_row, select_best_row_1se
from ._consensus import order_consensus_matrix

non_param_cols = {
    "n_clusters", "ari_stability", "ari_stability_se", 
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
        if key not in non_param_cols and pd.notnull(best_row[key])
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
    