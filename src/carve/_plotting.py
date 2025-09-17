from __future__ import annotations
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
from plotly.graph_objs import FigureWidget
from plotly.subplots import make_subplots

import plotly.io as pio

# pick a notebook-friendly renderer if none set
if pio.renderers.default == "":
    for cand in ("jupyterlab", "notebook", "vscode"):
        if cand in pio.renderers:
            pio.renderers.default = cand
            break


from ._selection import MEASURE_MAP, select_best_row, select_best_row_1se
from ._consensus import order_consensus_matrix

non_param_cols = {
    "n_clusters", 
    "ari_stability", "ari_stability_se", 
    "ari_generalizability", "ari_generalizability_se", 
    "consensus_pac_stability", "consensus_gini_stability", "consensus_ce_stability", 
    # "flip_instability", "contin_entropy"
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
    
    
    # --- debugging plots --- #
    
    
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
    
def plot_ari_hist(
    k: int,
    results: list,
    figsize: tuple = (8, 5),
    show_rug: bool = True
) -> None:
    if not results:
        raise ValueError("`results` is empty — nothing to plot.")

    # extract ARIs; guard against bad/missing values
    idx = np.arange(len(results))
    aris = []
    for i in idx:
        try:
            val = float(results[i][0])
            if np.isfinite(val):
                aris.append(val)
        except Exception:
            # skip malformed entries
            continue
    aris = np.asarray(aris, dtype=float)
    if aris.size == 0:
        raise ValueError("No finite ARI values found in the selected runs.")

    mean = float(np.mean(aris))
    median = float(np.median(aris))
    std = float(np.std(aris, ddof=1)) if aris.size > 1 else 0.0

    _, ax = plt.subplots(figsize=figsize)
    ax.hist(aris, bins="auto", alpha=0.85, edgecolor="black")

    # fix axis
    ax.set_xlim(0, 1)
    ax.set_xlabel("Adjusted Rand Index (ARI)")
    ax.set_ylabel("Count")
    ax.set_title(f"ARI across runs (n={aris.size}, k={k})")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.axvline(mean, linestyle="--", linewidth=1.5, label=f"mean={mean:.3f}")
    ax.axvline(median, linestyle="-.", linewidth=1.5, label=f"median={median:.3f}")

    txt = (
        f"mean = {mean:.3f}\n"
        f"median = {median:.3f}\n"
        f"std = {std:.3f}\n"
        f"min/max = {aris.min():.3f} / {aris.max():.3f}"
    )
    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    if show_rug:
        y0 = ax.get_ylim()[0]
        jitter = (ax.get_ylim()[1] - y0) * 0.005
        ax.vlines(aris, y0, y0 + jitter, linewidth=1)

    ax.legend()
    plt.tight_layout()
    plt.show()
    
    
    # --- interactive plots --- #


_display_name_map = {
    "ARI (stability)": "ari_stability",
    "ARI (generalizability)": "ari_generalizability",
    "Consensus PAC": "consensus_pac_stability",
    "Consensus Gini": "consensus_gini_stability",
    "Consensus CE": "consensus_ce_stability",
}

_inv_display_name_map = {v: k for k, v in _display_name_map.items()}

def plot_measure_vs_k_interactive(
    model_df: pd.DataFrame,
    *,
    measure: str = "stability",   # << keep parity with your static API
    rule: str = "max",
    width: int = 900,
    height: int = 600
) -> go.Figure:
    if model_df is None or model_df.empty:
        raise ValueError("model_df is empty; run validate() first.")
    if measure not in MEASURE_MAP:
        raise ValueError(f"Invalid measure {measure!r}. Options: {list(MEASURE_MAP)}")
    if rule not in {"max", "1se"}:
        raise ValueError("rule must be 'max' or '1se'")

    # available metric columns present in the DF
    available = {disp: col for disp, col in _display_name_map.items() if col in model_df.columns}
    if not available:
        raise ValueError("No global metrics found to plot.")

    # group columns as in your static function
    group_cols = [c for c in model_df.columns if c not in non_param_cols]

    df = model_df.copy()

    # assign legend/label per group (same formatting as your static plot)
    def _build_group_label(keys, group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        parts = []
        for col, val in zip(group_cols, keys):
            if pd.isna(val):
                continue
            if isinstance(val, (int, float)):
                val = f"{val:.4f}"
            if col == "estimator":
                parts.append(f"{val}")
            else:
                parts.append(f"{col}={val}")
        return ", ".join(parts) if parts else "default"

    df["_group_label"] = ""
    for keys, g in df.groupby(group_cols, dropna=False):
        df.loc[g.index, "_group_label"] = _build_group_label(keys, group_cols)

    # build hover text: label + hyperparams + ALL global metrics
    def _hovertext(row: pd.Series) -> str:
        # metrics block
        metrics_lines = []
        for disp, col in available.items():
            val = row.get(col, np.nan)
            if pd.notna(val):
                metrics_lines.append(f"{disp}: {float(val):.4f}")
        metrics_block = "<br>".join(metrics_lines)

        # hyper-params (exclude non-params & helper cols; estimator already in label)
        parts = []
        for k, v in row.items():
            if k in (non_param_cols | {"_group_label"}) or pd.isna(v):
                continue
            if k == "estimator":
                continue
            if isinstance(v, (int, float)):
                v = f"{v:.4f}"
            parts.append(f"{k}={v}")
        param_str = ", ".join(parts)

        base = row["_group_label"]
        if param_str:
            base = f"{base}<br><span style='font-size:0.9em; color:#666;'>({param_str})</span>"
        return f"{base}<br><br>{metrics_block}"

    df["_hovertext"] = df.apply(_hovertext, axis=1)

    # --- traces grouped by METRIC so dropdown can toggle visibility
    fig = go.Figure()
    metric_names = list(available.keys())
    traces_per_metric = []

    for disp in metric_names:
        col = available[disp]
        block = []
        for _, g in df.groupby(group_cols, dropna=False):
            g = g.sort_values("n_clusters")
            block.append(
                go.Scatter(
                    x=g["n_clusters"],
                    y=g[col],
                    mode="lines+markers",
                    name=g["_group_label"].iloc[0],   # same label as static
                    text=g["_hovertext"],
                    hovertemplate="%{text}<extra></extra>",
                    visible=False,
                    showlegend=True   # << keep legend entries
                )
            )
        traces_per_metric.append(block)
        fig.add_traces(block)

    # which metric should be shown by default? -> the one tied to the given `measure`
    default_y_col = MEASURE_MAP[measure]
    default_disp = _inv_display_name_map.get(default_y_col, metric_names[0])
    default_idx = metric_names.index(default_disp) if default_disp in metric_names else 0

    for tr in traces_per_metric[default_idx]:
        tr.visible = True

    # vertical dashed line at best k based on YOUR selection helpers (no re-calculation)
    best_row = (
        select_best_row(df, measure=measure, return_idx=False)
        if rule == "max"
        else select_best_row_1se(df, measure=measure, return_idx=False)
    )
    best_k = int(best_row["n_clusters"])

    fig.update_layout(
        width=width, height=height,
        title="Global metrics vs. k (interactive)",
        xaxis_title="n_clusters",
        yaxis_title=metric_names[default_idx],
        legend_title="Model (estimator + params)",
        shapes=[{
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": best_k,
            "x1": best_k,
            "y0": 0,
            "y1": 1,
            "line": {"dash": "dash"}
        }],
        updatemenus=[dict(
            type="dropdown",
            x=1.0, xanchor="right",
            y=1.2, yanchor="top",
            buttons=[
                dict(
                    label=disp,
                    method="update",
                    args=[
                        # toggle trace visibility block-wise
                        {
                            "visible": [
                                (cum <= i < cum + len(traces_per_metric[j]))
                                for j in range(len(traces_per_metric))
                                for i, cum in enumerate(
                                    np.repeat([0], sum(len(b) for b in traces_per_metric))
                                )
                            ]  # we'll override properly below
                        },
                        {"yaxis": {"title": disp}}
                    ]
                ) for disp in metric_names
            ]
        )]
    )

    # The quick way to build the visibility masks (one block on at a time):
    counts = [len(b) for b in traces_per_metric]
    cum = np.cumsum([0] + counts)
    n_total = cum[-1]

    buttons = []
    for i, disp in enumerate(metric_names):
        vis = [False] * n_total
        vis[cum[i]:cum[i+1]] = [True] * counts[i]
        buttons.append(dict(
            label=disp,
            method="update",
            args=[{"visible": vis}, {"yaxis": {"title": disp}}]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            x=1.0, xanchor="right",
            y=1.2, yanchor="top",
            buttons=buttons
        )]
    )
    
    fig.update_layout(
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.6)"
        )
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig

def plot_clustering_interactive(
    X: np.ndarray,
    row: pd.Series,
    labels: np.ndarray,
    *,
    stab_gini_vec: np.ndarray,
    stab_ce_vec: np.ndarray,
    gen_vec: np.ndarray,
    measure: str = "stability",
    width: int = 1100,
    height: int = 520,
    min_size: float = 20.0,     # matplotlib-style areas (pt^2)
    max_size: float = 180.0,    # matplotlib-style areas (pt^2)
    min_alpha: float = 0.30,
    max_alpha: float = 1.00,
) -> go.Figure:

    scatter_measure_map = {
        "stability": "Stability (Gini)",
        "generalizability": "Generalizability",
        "stability_ce": "Stability (CE)",
    }

    n = X.shape[0]
    assert labels.shape[0] == n

    hover_tmpl_scatter = (
        "sample %{customdata[0]}<br>"
        "cluster %{customdata[1]}<br>"
        "stability (gini): %{customdata[2]:.4f}<br>"
        "stability (ce): %{customdata[3]:.4f}<br>"
        "generalizability: %{customdata[4]:.4f}<extra></extra>"
    )
    hover_tmpl_strip = (
        "cluster %{x}<br>"
        "sample %{customdata[0]}<br>"
        "value %{y:.4f}<br><br>"
        "<b>all metrics</b><br>"
        "stability (gini): %{customdata[2]:.4f}<br>"
        "stability (ce): %{customdata[3]:.4f}<br>"
        "generalizability: %{customdata[4]:.4f}<extra></extra>"
    )

    # --- PCA (fixed)
    X_pca = PCA(n_components=2).fit_transform(X)
    x_min, x_max = float(X_pca[:, 0].min()), float(X_pca[:, 0].max())
    y_min, y_max = float(X_pca[:, 1].min()), float(X_pca[:, 1].max())
    dx = (x_max - x_min) or 1.0
    dy = (y_max - y_min) or 1.0
    x_rng = [x_min - 0.05 * dx, x_max + 0.05 * dx]
    y_rng = [y_min - 0.05 * dy, y_max + 0.05 * dy]

    # --- customdata: [sample_idx, cluster, gini, ce, gen]
    sample_idx = np.arange(n, dtype=int)
    customdata_all = np.column_stack([sample_idx, labels, stab_gini_vec, stab_ce_vec, gen_vec])

    # --- metrics
    metric_names = ["Stability (Gini)", "Stability (CE)", "Generalizability"]
    metric_arrays = {
        "Stability (Gini)": np.asarray(stab_gini_vec, float),
        "Stability (CE)":   np.asarray(stab_ce_vec,   float),
        "Generalizability": np.asarray(gen_vec,       float),
    }
    default_metric = scatter_measure_map.get(measure, "Stability (Gini)")

    # optional: lock the right y-axis range across metrics for visual comparability
    all_vals = np.concatenate([metric_arrays[m] for m in metric_names])
    y2_pad = 0.04 * (all_vals.max() - all_vals.min() or 1.0)
    y2_range = [float(all_vals.min() - y2_pad), float(all_vals.max() + y2_pad)]

    # --- helpers
    def _sizes_from_metric(values, pt2_min, pt2_max):
        v = np.asarray(values, float)
        vmin, vmax = float(v.min()), float(v.max())
        if np.isclose(vmin, vmax):
            pt2 = np.full_like(v, np.mean([pt2_min, pt2_max]))
        else:
            pt2 = pt2_min + (v - vmin) / (vmax - vmin) * (pt2_max - pt2_min)
        # plotly marker.size is a diameter-ish pixel value; map from area via sqrt
        px = np.sqrt(pt2)
        return np.clip(px, 4.0, 36.0)

    def _cluster_opacity_from_metric(values, labels, a_min, a_max):
        clus = np.unique(labels)
        means = {c: float(np.mean(values[labels == c])) for c in clus}
        vv = np.array([means[c] for c in clus], float)
        vmin, vmax = float(vv.min()), float(vv.max())
        if np.isclose(vmin, vmax):
            return {c: np.mean([a_min, a_max]) for c in clus}
        return {c: a_min + (means[c] - vmin) / (vmax - vmin) * (a_max - a_min) for c in clus}

    size_by_metric  = {m: _sizes_from_metric(metric_arrays[m], min_size, max_size) for m in metric_names}
    alpha_by_metric = {m: _cluster_opacity_from_metric(metric_arrays[m], labels, min_alpha, max_alpha)
                       for m in metric_names}

    # --- figure & traces
    base = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]], horizontal_spacing=0.08)
    fig = FigureWidget(base)

    clus = np.unique(labels)
    left_trace_ids = []
    orig_sizes = []  # keep initial sizes to reuse for non-updated traces in buttons

    # LEFT panel: one scatter per cluster
    for c in clus:
        mask = (labels == c)
        tr = go.Scatter(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            mode="markers",
            name=f"cluster {c}",
            legendgroup=f"cluster-{c}",
            showlegend=True,
            customdata=customdata_all[mask],
            hovertemplate=hover_tmpl_scatter,
            marker=dict(
                size=size_by_metric[default_metric][mask],
                opacity=alpha_by_metric[default_metric][c],
                line=dict(width=0),
            ),
        )
        left_trace_ids.append(len(fig.data))
        orig_sizes.append(tr.marker.size)
        fig.add_trace(tr, row=1, col=1)

    # RIGHT panel — prebuild PER METRIC, then toggle visibility
    right_ids_by_metric = {m: {"box": [], "strip": []} for m in metric_names}
    cat_labels = [str(int(c)) for c in clus]

    for m in metric_names:
        vals_m = metric_arrays[m]
        for c in clus:
            mask = (labels == c)
            cat = str(int(c))

            tr_box = go.Box(
                x=[cat] * int(mask.sum()),
                y=vals_m[mask],
                name=f"cluster {c}",
                legendgroup=f"cluster-{c}-{m}",
                showlegend=False,
                boxpoints=False,
                hoverinfo="skip",
                offsetgroup=cat,
                visible=(m == default_metric),
                width=0.6,
            )
            right_ids_by_metric[m]["box"].append(len(fig.data))
            fig.add_trace(tr_box, row=1, col=2)

            tr_pts = go.Scatter(
                x=[cat] * int(mask.sum()),
                y=vals_m[mask],
                mode="markers",
                name=f"points {c}",
                legendgroup=f"cluster-{c}-{m}",
                showlegend=False,
                customdata=customdata_all[mask],
                hovertemplate=hover_tmpl_strip,
                marker=dict(size=9, opacity=0.9, line=dict(width=0)),
                offsetgroup=cat,
                visible=(m == default_metric),
            )
            right_ids_by_metric[m]["strip"].append(len(fig.data))
            fig.add_trace(tr_pts, row=1, col=2)

    # make the right x-axis categorical with a fixed order
    fig.update_xaxes(
        title_text="cluster",
        type="category",
        categoryorder="array",
        categoryarray=cat_labels,
        row=1, col=2
    )

    # overlay highlight trace on LEFT
    highlight_trace_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode="markers",
            showlegend=False,
            marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color="white", width=3)),
        ),
        row=1, col=1
    )

    # layout
    fig.update_layout(
        uirevision="carve-clustering-v1",
        width=width, height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.6)")
    )
    fig.update_xaxes(title_text="pc1", range=x_rng, row=1, col=1)
    fig.update_yaxes(title_text="pc2", range=y_rng, row=1, col=1)
    fig.update_xaxes(title_text="cluster", row=1, col=2)
    fig.update_yaxes(
        title_text=("per-sample generalizability" if default_metric == "Generalizability" else "per-sample value"),
        range=y2_range,
        row=1, col=2
    )

    all_trace_count = len(fig.data)

    # --- button helpers
    def _visible_mask_for_metric(metric_label: str):
        vis = [True] * all_trace_count
        for m in metric_names:
            turn_on = (m == metric_label)
            for tid in (right_ids_by_metric[m]["box"] + right_ids_by_metric[m]["strip"]):
                vis[tid] = turn_on
        vis[highlight_trace_index] = True
        return vis

    # build per-trace style lists WITHOUT touching x/y for any trace
    def _style_for_metric(metric_label: str):
        sizes  = size_by_metric[metric_label]
        alphas = alpha_by_metric[metric_label]

        marker_size_list   = []
        marker_opacity_list = []

        # iterate over all traces; only left traces get new per-point sizes/alpha
        for tidx in range(all_trace_count):
            if tidx in left_trace_ids:
                c = clus[left_trace_ids.index(tidx)]
                mask = (labels == c)
                marker_size_list.append(sizes[mask])
                marker_opacity_list.append(alphas[c])
            else:
                # keep whatever they currently have (avoid sending None)
                tr = fig.data[tidx]
                marker_size_list.append(getattr(tr.marker, "size", None))
                marker_opacity_list.append(getattr(tr.marker, "opacity", None))

        return {
            "marker.size": marker_size_list,
            "marker.opacity": marker_opacity_list,
            # also set visibility here for all traces
            "visible": _visible_mask_for_metric(metric_label),
        }

    # hover callbacks for strip traces (all metrics)
    per_cluster_abs_ids = {int(c): sample_idx[labels == c] for c in clus}

    def _mk_hover_cb(trace_index):
        def _on_hover(trace, points, state):
            if points.point_inds:
                local_idx = points.point_inds[0]
                c = int(fig.data[trace_index].customdata[local_idx, 1])
                abs_id = int(per_cluster_abs_ids[c][local_idx])
                xy = X_pca[abs_id]
                with fig.batch_update():
                    fig.data[highlight_trace_index].x = [xy[0]]
                    fig.data[highlight_trace_index].y = [xy[1]]
        return _on_hover

    def _mk_unhover_cb(_trace_index):
        def _on_unhover(trace, points, state):
            with fig.batch_update():
                fig.data[highlight_trace_index].x = []
                fig.data[highlight_trace_index].y = []
        return _on_unhover

    # attach to all strip traces so it keeps working after metric switches
    for m in metric_names:
        for tidx in right_ids_by_metric[m]["strip"]:
            fig.data[tidx].on_hover(_mk_hover_cb(tidx))
            fig.data[tidx].on_unhover(_mk_unhover_cb(tidx))

    # buttons
    def _yaxis2_title(metric_label: str):
        return "per-sample generalizability" if metric_label == "Generalizability" else "per-sample value"

    buttons = []
    for m in metric_names:
        buttons.append(dict(
            label=m,
            method="update",
            args=[
                _style_for_metric(m),     # restyle across all traces (no x/y!)
                {"yaxis2": {"title": _yaxis2_title(m), "range": y2_range}},
            ],
        ))

    fig.update_layout(updatemenus=[dict(
        type="dropdown", x=0.99, xanchor="right", y=1.18, yanchor="top", buttons=buttons
    )])

    # build title
    def _is_scalar_like(v):
        import numpy as _np
        from numpy import floating, integer
        return isinstance(v, (bool, int, float, str, _np.generic, floating, integer))

    def _to_python(v):
        try:
            # numpy scalars -> python
            return v.item() if hasattr(v, "item") else v
        except Exception:
            return v

    # build param dict
    params = {
        key: _to_python(row[key])
        for key in row.index
        if key not in non_param_cols and pd.notna(row[key]) and _is_scalar_like(row[key])
    }

    # format values
    formatted_params = {}
    for key, value in params.items():
        if isinstance(value, float):
            formatted_params[key] = f"{value:.4f}"
        else:
            formatted_params[key] = str(value)

    param_str = ", ".join(f"{k} = {v}" for k, v in formatted_params.items())

    # base title
    est = str(_to_python(row.get("estimator", "")))
    k_val = _to_python(row.get("n_clusters", ""))
    title = f"{est} | k = {k_val}" + (f", {param_str}" if param_str else "")

    # cap to avoid massive titles wrecking layout
    MAX_TITLE_CHARS = 220
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS - 1] + "…"

    # attach to layout (simple form)
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font=dict(size=14)
    )

    return fig