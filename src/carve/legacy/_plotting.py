"""Legacy plotting utilities for CARVE (static and interactive)."""

from __future__ import annotations
import warnings

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple
from IPython.display import display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
from sklearn.decomposition import PCA

try:
    import plotly.graph_objects as go
    from plotly.colors import qualitative as qual
    from plotly.colors import hex_to_rgb
    from plotly.graph_objs import FigureWidget
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except Exception: 
    _HAS_PLOTLY = False

from ._selection import (
    MEASURE_MAP,
    select_best_row_max,
    select_best_row_1se,
    select_best_row_quantile,
    select_best_row_by_rule
)

from ._consensus import reorder_consensus_matrix

Measure = Literal[
    "stability", "generalizability", "consensus_pac", "consensus_gini", "consensus_ce"
]
Rule = Literal["max", "1se", "quantile"]

# Columns that are not hyper-parameters.
NON_PARAM_COLS: frozenset[str] = frozenset(
    {
        "n_clusters",
        "ari_stability", "ari_stability_se", "ari_stability_upper", "ari_stability_lower",
        "ari_generalizability", "ari_generalizability_se", "ari_generalizability_upper", "ari_generalizability_lower",
        "ari_average", "ari_average_se", "ari_average_upper", "ari_average_lower",
        "consensus_pac_stability", "consensus_gini_stability", "consensus_ce_stability",
        "misclassification_generalizability"
    }
)

Y_LABELS: Mapping[str, str] = {
    "ari_stability": "ARI (stability)",
    "ari_generalizability": "ARI (generalizability)",
    "ari_average": "ARI (average)",
    "consensus_pac_stability": "Consensus PAC",
    "consensus_gini_stability": "Consensus Gini",
    "consensus_ce_stability": "Consensus CE",
    "misclassification_generalizability": "Global Misclassification",
}

DISPLAY_NAME: Mapping[str, str] = {
    "ARI (stability)": "ari_stability",
    "ARI (generalizability)": "ari_generalizability",
    "ARI (average)": "ari_average",
    "Consensus PAC": "consensus_pac_stability",
    "Consensus Gini": "consensus_gini_stability",
    "Consensus CE": "consensus_ce_stability",
    "Global Misclassification": "misclassification_generalizability",
}
INV_DISPLAY_NAME: Mapping[str, str] = {v: k for k, v in DISPLAY_NAME.items()}

ARI_COLS = {"ari_stability", "ari_generalizability", "ari_average"}

RULES_FOR_COL = {
    "ari_stability": ["max", "1se", "quantile"],
    "ari_generalizability": ["max", "1se", "quantile"],
    "ari_average": ["max", "1se", "quantile"],
    
    "consensus_pac": ["max"],
    "consensus_gini": ["max"],
    "consensus_ce": ["max"],
    "misclassification_generalizability": ["max"],
}


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for legacy plotting functions.

    Attributes
    ----------
    figsize : tuple of int
        Figure size in inches.
    decimals : int
        Decimal precision for labels.
    show_grid : bool
        Whether to show grid lines.
    legend_outside : bool
        Whether to place legend outside the axes.
    """
    figsize: Tuple[int, int] = (10, 8)
    decimals: int = 4
    show_grid: bool = True
    legend_outside: bool = True
    
def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Raise if required columns are missing.

    Parameters
    ----------
    df : pandas.DataFrame
        Results table.
    cols : iterable of str
        Required column names.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    
def _group_columns(df: pd.DataFrame) -> List[str]:
    """Return parameter columns used to group configurations.

    Parameters
    ----------
    df : pandas.DataFrame
        Results table.

    Returns
    -------
    cols : list of str
        Columns excluding ``NON_PARAM_COLS``.
    """
    return [c for c in df.columns if c not in NON_PARAM_COLS]

def _fmt_float(x: float, decimals: int = 4) -> str:
    """Format a float with fixed precision.

    Parameters
    ----------
    x : float
        Value to format.
    decimals : int, default=4
        Decimal precision.

    Returns
    -------
    text : str
        Formatted value.
    """
    return f"{float(x):.{decimals}f}"

def _build_group_label(keys: Tuple, group_cols: Sequence[str], decimals: int) -> str:
    """Build a group label from grouped column keys.

    Parameters
    ----------
    keys : tuple
        Group key values.
    group_cols : sequence of str
        Group column names.
    decimals : int
        Decimal precision for floats.

    Returns
    -------
    label : str
        Human-readable label.
    """
    if not isinstance(keys, tuple):
        keys = (keys,)
    
    parts: List[str] = []

    for col, val in zip(group_cols, keys):
        if pd.isna(val):
            continue
        if isinstance(val, (int, float)):
            sval = _fmt_float(val, decimals)
        else:
            sval = str(val)

        if col == "estimator":
            parts.append(f"{sval}")
        else:
            parts.append(f"{col}={sval}")

    return ", ".join(parts) if parts else "default"

# ––––– Regular Plotting Functions ––––– 
def plot_measure_vs_k(
    model_df: pd.DataFrame,
    measure: Measure = "stability",
    rule: Rule = "max",
    config: PlotConfig = PlotConfig()
) -> None:
    """Plot a global measure versus k for each configuration.

    Parameters
    ----------
    model_df : pandas.DataFrame
        Results table.
    measure : Measure, default="stability"
        Metric key to plot.
    rule : Rule, default="max"
        Selection rule for highlighting the best k.
    config : PlotConfig, default=PlotConfig()
        Plot configuration.
    """
    cfg = config or PlotConfig()
    y_col = MEASURE_MAP[measure]
    _require_columns(model_df, ["n_clusters", y_col])
    
    group_cols = _group_columns(model_df)
    best_row = select_best_row_by_rule(model_df, measure, rule)
    best_k = int(best_row["n_clusters"]) if pd.notna(best_row.get("n_clusters")) else None
    
    fig, ax = plt.subplots(figsize=cfg.figsize)
    
    for keys, group in model_df.groupby(group_cols, dropna=False):
        g = group.sort_values("n_clusters")
        label = _build_group_label(keys, group_cols, cfg.decimals)
        ax.plot(g["n_clusters"], g[y_col], marker="o", label=label)

        se_col = f"{y_col}_se"
        if se_col in g.columns:
            ax.fill_between(
                g["n_clusters"],
                g[y_col] - g[se_col],
                g[y_col] + g[se_col],
                alpha=0.2
            )
        
    if best_k is not None:
        ax.axvline(best_k, linestyle="--", alpha=0.7)

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel(Y_LABELS.get(y_col, y_col))
    ax.set_title(f"{measure} per estimator vs. k")
    
    if cfg.legend_outside:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend()
    ax.grid(cfg.show_grid, alpha=0.3)
    
    fig.tight_layout()
    plt.show()

def plot_pipeline_vs_k(
    pipeline_df: pd.DataFrame,
    *,
    measure: Measure = "stability",
    config: PlotConfig | None = None
) -> None:
    """Plot a measure over k for preprocessing pipelines.

    Parameters
    ----------
    pipeline_df : pandas.DataFrame
        Preprocessing results table.
    measure : Measure, default="stability"
        Metric key to plot.
    config : PlotConfig or None, default=None
        Plot configuration.
    """
    config = config or PlotConfig()
    y_col = MEASURE_MAP[measure]
    _require_columns(pipeline_df, ["norm__func", "dr__method", "n_clusters", y_col])
    
    df = pipeline_df.assign(pipeline=lambda d: d["norm__func"] + " + " + d["dr__method"])
    
    fig, ax = plt.subplots(figsize=config.figsize)
    for name, g in df.groupby("pipeline"):
        g = g.sort_values("n_clusters")
        ax.plot(g["n_clusters"], g[y_col], marker="o", label=str(name))
        
    ax.set_title(f"Clustering {measure} by Preprocessing Pipeline")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(Y_LABELS.get(y_col, y_col))
    
    if config.legend_outside:
        ax.legend(title="Pipeline", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend(title="Pipeline")
    ax.grid(config.show_grid, alpha=0.3)
    
    fig.tight_layout()
    plt.show()

def plot_consensus_matrix(
    model_df: pd.DataFrame, 
    consensus_mats_raw: Sequence[np.ndarray], 
    *, 
    measure: Measure = "stability", 
    rule: Rule = "max", 
    k: Optional[int] = None, 
    config: PlotConfig | None = None
) -> None:
    """Plot the consensus matrix for the selected configuration.

    Parameters
    ----------
    model_df : pandas.DataFrame
        Results table.
    consensus_mats_raw : sequence of ndarray
        Consensus matrices aligned with ``model_df`` rows.
    measure : Measure, default="stability"
        Metric key used for selection.
    rule : Rule, default="max"
        Selection rule.
    k : int or None, default=None
        Optional fixed number of clusters.
    config : PlotConfig or None, default=None
        Plot configuration.
    """
    cfg = config or PlotConfig()

    df = model_df if k is None else model_df[model_df["n_clusters"] == k]
    if df.empty:
        raise ValueError("No rows available for the given k filter.")
    
    best_idx = select_best_row_by_rule(df, measure, rule, return_idx=True)
    
    best_row = df.loc[int(best_idx)]
    
    C = consensus_mats_raw[int(best_idx)]
    C_ordered, _ = reorder_consensus_matrix(C)
    
    params: Dict[str, str] = {}
    for key, value in best_row.items():
        if key in NON_PARAM_COLS or pd.isna(value) or key == "estimator":
            continue
        
        if isinstance(value, float):
            params[key] = _fmt_float(value, cfg.decimals)
        else:
            params[key] = str(value)
    
    est = str(best_row.get("estimator", ""))
    k_val = best_row.get("n_clusters", "")
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    title = f"{est} | k={k_val}" + (f", {param_str}" if param_str else "")

    fig, ax = plt.subplots(figsize=cfg.figsize)
    im = ax.imshow(C_ordered, aspect="auto", interpolation="none")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("consensus")
    ax.set_title(title)
    ax.set_xlabel("samples (ordered)")
    ax.set_ylabel("samples (ordered)")
    
    fig.tight_layout()
    plt.show()
    
def plot_clustering(
    X: np.ndarray,
    row: pd.Series, 
    labels: np.ndarray,
    sample_level_measures: np.ndarray,
    *,
    measure: Measure = "stability",
    config: PlotConfig | None = None,
    min_size: float = 20.0,
    max_size: float = 180.0,
    min_alpha: float = 0.30,
    max_alpha: float = 1.00,
) -> None:
    """Plot clustering scatter with sample-level encodings.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    row : pandas.Series
        Selected results row.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    sample_level_measures : ndarray of shape (n_samples,)
        Per-sample metric values.
    measure : Measure, default="stability"
        Metric key used for labels/annotation.
    config : PlotConfig or None, default=None
        Plot configuration.
    min_size : float, default=20.0
        Minimum point size.
    max_size : float, default=180.0
        Maximum point size.
    min_alpha : float, default=0.30
        Minimum point alpha.
    max_alpha : float, default=1.00
        Maximum point alpha.
    """
    cfg = config or PlotConfig()
    
    n = X.shape[0]
    if labels.shape[0] != n or sample_level_measures.shape[0] != n:
        raise ValueError("Shapes must align: len(labels) == len(sample_level_measures) == X.shape[0]")

    vmin, vmax = float(np.min(sample_level_measures)), float(np.max(sample_level_measures))
    if np.isclose(vmin, vmax):  # if consensus matrix is very clean, all samples will have similar values
        sizes = np.full(n, 0.5 * (min_size + max_size))
        alpha_map = {c: 0.5 * (min_alpha + max_alpha) for c in np.unique(labels)}
    
    else:  # usual case
        norm = (sample_level_measures - vmin)
        sizes = min_size + norm * (max_size - min_size)
        
        means = {c: float(np.mean(sample_level_measures[labels == c])) for c in np.unique(labels)}
        mvals = np.array(list(means.values()))
        mmin, mmax = float(mvals.min()), float(mvals.max())
        alpha_map = {
            c: (0.5 * (min_alpha + max_alpha) if np.isclose(mmin, mmax)  # if cluster means are close
                else min_alpha + (means[c] - mmin) / (mmax - mmin) * (max_alpha - min_alpha))  # usual case
            for c in means
        }

    X_pca = PCA(n_components=2).fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.figsize)
    
    # scatterplot by cluster
    unique_clusters = np.unique(labels)
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for i, c in enumerate(unique_clusters):
        mask = (labels == c)
        ax1.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            s=sizes[mask],
            alpha=alpha_map[c],
            color=cluster_colors[i],
            label=f"cluster {i+1}"
        )
        
    # title/params
    params: Dict[str, str] = {}
    for key, value in row.items():
        if key in NON_PARAM_COLS or pd.isna(value) or key == "estimator":
            continue
        params[key] = _fmt_float(value, cfg.decimals) if isinstance(value, float) else str(value)
    
    handles, labels_ = ax1.get_legend_handles_labels()
        
    est = str(row.get("estimator", ""))
    k_val = row.get("n_clusters", "")
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    
    ax1.set_title(f"{est} | k={k_val}" + (f", {param_str}" if param_str else ""))
    ax1.set_xlabel("pc1")
    ax1.set_ylabel("pc2")
        
    # right panel boxplot
    data = [sample_level_measures[labels == c] for c in unique_clusters]
    
    bp = ax2.boxplot(data, patch_artist=True, labels=[str(int(i+1)) for i in range(len(unique_clusters))])
    for patch, color in zip(bp['boxes'], cluster_colors):
        patch.set_facecolor(color)

    ax2.set_title(
        "per-sample generalizability by cluster"
        if MEASURE_MAP[measure] == "ari_generalizability"
        else "per-sample stability by cluster"
    )
    ax2.set_xlabel("cluster")
    ax2.set_ylabel("per-sample value")
    
    if cfg.legend_outside:
        ax2.legend(
            handles, labels_,
            bbox_to_anchor=(1.02, 0.5),
            loc="upper left",
            borderaxespad=0.0,
            frameon=False,
            title="clusters"
        )
    else:
        ax1.legend()

    fig.tight_layout()
    plt.show()
    
def plot_consensus_clustering(
    X: np.ndarray,
    row: pd.Series, 
    labels: np.ndarray,
    sample_level_measures: np.ndarray,
    consensus_mat_raw: np.ndarray, 
    *,
    measure: Measure = "stability",
    config: PlotConfig | None = None,
    min_size: float = 20.0,
    max_size: float = 180.0,
    min_alpha: float = 0.30,
    max_alpha: float = 1.00,
) -> None:
    """Plot clustering scatter, boxplot, and consensus matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    row : pandas.Series
        Selected results row.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    sample_level_measures : ndarray of shape (n_samples,)
        Per-sample metric values.
    consensus_mat_raw : ndarray of shape (n_samples, n_samples)
        Raw consensus matrix.
    measure : Measure, default="stability"
        Metric key used for labels/annotation.
    config : PlotConfig or None, default=None
        Plot configuration.
    min_size : float, default=20.0
        Minimum point size.
    max_size : float, default=180.0
        Maximum point size.
    min_alpha : float, default=0.30
        Minimum point alpha.
    max_alpha : float, default=1.00
        Maximum point alpha.
    """
    cfg = config or PlotConfig()
    
    n = X.shape[0]
    if labels.shape[0] != n or sample_level_measures.shape[0] != n:
        raise ValueError("Shapes must align: len(labels) == len(sample_level_measures) == X.shape[0]")

    vmin, vmax = float(np.min(sample_level_measures)), float(np.max(sample_level_measures))
    if np.isclose(vmin, vmax):  # if consensus matrix is very clean, all samples will have similar values
        sizes = np.full(n, 0.5 * (min_size + max_size))
        alpha_map = {c: 0.5 * (min_alpha + max_alpha) for c in np.unique(labels)}
    
    else:  # usual case
        norm = (sample_level_measures - vmin)
        sizes = min_size + norm * (max_size - min_size)
        
        means = {c: float(np.mean(sample_level_measures[labels == c])) for c in np.unique(labels)}
        mvals = np.array(list(means.values()))
        mmin, mmax = float(mvals.min()), float(mvals.max())
        alpha_map = {
            c: (0.5 * (min_alpha + max_alpha) if np.isclose(mmin, mmax)  # if cluster means are close
                else min_alpha + (means[c] - mmin) / (mmax - mmin) * (max_alpha - min_alpha))  # usual case
            for c in means
        }

    X_pca = PCA(n_components=2).fit_transform(X)
    
    # layout
    fig = plt.figure(figsize=cfg.figsize)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.0, 1.1],
        height_ratios=[0.05, 0.95],
        wspace=0.1, hspace=0.02
    )

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    axbar = fig.add_subplot(gs[0, 1])
    
    # left panel: scatter
    codes, uniques = pd.factorize(labels)
    k = len(uniques)
    cmap = plt.cm.tab20 if k > 10 else plt.cm.tab10
    palette = cmap(np.linspace(0, 1, k))
    
    for code in range(k):
        mask = (codes == code)
        ax1.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            s=sizes[mask],
            alpha=alpha_map.get(uniques[code], 0.9),
            color=palette[code],
            label=f"cluster {code+1}",
            edgecolors="black",
            linewidths=1.0
        )
        
    # title/params
    params: Dict[str, str] = {}
    for key, value in row.items():
        if key in NON_PARAM_COLS or pd.isna(value) or key == "estimator":
            continue
        params[key] = _fmt_float(value, cfg.decimals) if isinstance(value, float) else str(value)
    
    handles, labels_ = ax1.get_legend_handles_labels()
        
    est = str(row.get("estimator", ""))
    k_val = row.get("n_clusters", "")
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    
    ax1.set_title(f"{est} | k={k_val}" + (f", {param_str}" if param_str else ""))
    # ax1.set_xlabel("pc1")
    # ax1.set_ylabel("pc2")
        
    # right panel: consensus matrix
    C_ordered, order = reorder_consensus_matrix(consensus_mat_raw)
    ax2.imshow(C_ordered, aspect="auto", interpolation="none")
    
    # top color strip
    sample_colors = palette[codes]
    bar_rgba = sample_colors[order][None, :, :]
    axbar.set_xlim(0, consensus_mat_raw.shape[0])
    
    fig.canvas.draw()
    bar_pos = ax2.get_position()
    bar_height = 0.04

    axbar.set_position([
        bar_pos.x0 + 0.0005,
        bar_pos.y1,
        bar_pos.width,
        bar_height
    ])

    axbar.imshow(bar_rgba, aspect="auto", interpolation="none")
    axbar.set_axis_off()
    axbar.set_title("Consensus Matrix by Cluster", pad=8, fontsize=12)
    
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_xlabel("")

    # legend + rest of figure
    if cfg.legend_outside:
        ax2.legend(
            handles, labels_,
            bbox_to_anchor=(1.02, 0.5),
            loc="upper left",
            borderaxespad=0.0,
            frameon=False,
            title="clusters"
        )
    else:
        ax1.legend()

    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

# ––––– Interactive Plotting Functions ––––– 
def plot_measure_vs_k_interactive(
    model_df: pd.DataFrame,
    *,
    measure: Measure = "stability",
    rule: Rule = "max",
    width: int = 1000,
    height: int = 800,
) -> None:
    """Plot global metrics vs k using Plotly (interactive).

    Parameters
    ----------
    model_df : pandas.DataFrame
        Results table.
    measure : Measure, default="stability"
        Metric key to plot by default.
    rule : Rule, default="max"
        Selection rule for highlighting the best k.
    width : int, default=1000
        Figure width in pixels.
    height : int, default=800
        Figure height in pixels.
    """
    if not _HAS_PLOTLY:
        raise RuntimeError("plotly is not available. Install plotly to use interactive plots.")
    
    available = {pretty: col for pretty, col in DISPLAY_NAME.items() if col in model_df.columns}
    metric_names = list(available.keys())

    if not available:
        raise ValueError("No recognizable global metrics present in model_df.")  # should really never happen
    
    # 1) build group labels
    group_cols = _group_columns(model_df)
    df = model_df.copy()
    df['_group_label'] = ''
    for keys, g in df.groupby(group_cols, dropna=False):
        df.loc[g.index, '_group_label'] = _build_group_label(keys, group_cols, decimals=4)

    # 2) hover text builder
    def _hovertext(row: pd.Series) -> str:
        metrics_lines = []
        for pretty, col in available.items():  # iterate all available metrics and, if present/not NaN, add a formatted line
            val = row.get(col, np.nan)
            
            if pd.notna(val):
                metrics_lines.append(f"{pretty}: {_fmt_float(float(val), 4)}")
                
        metrics_block = "<br>".join(metrics_lines)
        
        parts = []
        for k, v in row.items():  # collect model hyperparameters from the row for display
            if k in (NON_PARAM_COLS | {'_group_label'}) or pd.isna(v) or k == "estimator":
                continue
            parts.append(f"{k}={_fmt_float(v) if isinstance(v, (int,float)) else v}")
            
        param_str = ", ".join(parts)
        
        base = row["_group_label"]
        if param_str:  # start with group label; if there is parameter string, add it on smaller gray line
            base = f"{base}<br><span style='font-size:0.9em; color:#666;'>({param_str})</span>"
            
        return f"{base}<br><br>{metrics_block}"

    df["_hovertext"] = df.apply(_hovertext, axis=1)
        
    # 3) set default visible metric and display
    default_y_col = MEASURE_MAP[measure]
    default_pretty = INV_DISPLAY_NAME.get(default_y_col, metric_names[0])
    default_idx = metric_names.index(default_pretty) if default_pretty in metric_names else 0
    
    # 4.1) create traces lines per metric & per group
    fig = go.Figure()
    trace_blocks = []
    
    for m_idx, pretty in enumerate(metric_names):
        col = available[pretty]
        show_now = (m_idx == default_idx)
        
        start_idx = len(fig.data)
        for _, g in df.groupby(group_cols, dropna=False):
            g = g.sort_values("n_clusters")
            fig.add_trace(go.Scatter(
                x=g["n_clusters"], y=g[col],
                mode="lines+markers",
                name=g["_group_label"].iloc[0],
                text=g["_hovertext"],
                hovertemplate="%{text}<extra></extra>",
                visible=show_now,
                showlegend=True,
            ))
            
            if col in ("ari_stability", "ari_generalizability"):
                se_col = f"{col}_se"
                if se_col in g.columns:
                    lower = g[col] - g[se_col]
                    upper = g[col] + g[se_col]
                    
                    fig.add_trace(go.Scatter(
                        x=g["n_clusters"], y=lower,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                        visible=show_now,
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=g["n_clusters"], y=upper,
                        mode="lines",
                        fill="tonexty", fillcolor="rgba(0,0,0,0.08)",
                        line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                        visible=show_now,
                    ))
                    
        end_idx = len(fig.data) 
        trace_blocks.append((pretty, start_idx, end_idx))
    
    # 4.2) build flat list of all metric trace indices, and per-metric slice into flat list
    metric_blocks = trace_blocks[:]
    all_metric_idxs: list[int] = []
    metric_slices: dict[str, tuple[int, int]] = {}

    for pretty, start_idx, end_idx in metric_blocks:
        block = list(range(start_idx, end_idx))
        s = len(all_metric_idxs)
        all_metric_idxs.extend(block)
        e = len(all_metric_idxs)
        metric_slices[pretty] = (s, e)
    
    # 5) get and mark best k
    MEASURE_INV_MAP = {v: k for k, v in MEASURE_MAP.items()}
    best_k_map: dict[tuple[str, str], int | None] = {}
    
    for pretty in metric_names:
        col = available[pretty]
        measure_key = MEASURE_INV_MAP.get(col)
        if measure_key is None:
            continue
        for r in RULES_FOR_COL.get(col, ["max"]):
            row = select_best_row_by_rule(model_df, measure_key, r)
            k = int(row["n_clusters"]) if pd.notna(row.get("n_clusters")) else None
            best_k_map[(pretty, r)] = k

    # 6) build base layout
    k0 = best_k_map.get((default_pretty, rule), None)
    
    fig.update_layout(
        width=width,
        height=height,
        title="Global metrics vs. k",
        xaxis_title="n_clusters",
        yaxis_title=metric_names[default_idx],
        legend_title="Model (estimator + params)",
        shapes=[{
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": k0, "x1": k0,
            "y0": 0, "y1": 1,
            "line": {"dash": "dash"},
            "layer": "above",
        }],
    )
    
    # 7a) metric dropdown (controls traces)
    n_metric_traces = len(all_metric_idxs)
    buttons = []
    for pretty in metric_names:
        s, e = metric_slices[pretty]
        vis = [False] * n_metric_traces
        for i in range(s, e):
            vis[i] = True
        buttons.append(dict(
            label=pretty,
            method="update",
            args=[
                {"visible": vis}, 
                {"yaxis": {"title": pretty, "autorange": True}},
                all_metric_idxs,
            ],
        ))
    
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            x=1.0, xanchor="right",
            y=1.2, yanchor="top",
            buttons=buttons,
            showactive=True,
            active=default_idx,
        )]
    )
        
    # 7b) rule dropdown (controls best_k vertical line)
    rule_buttons = []
    active_rule_index = 0
    
    def _pretty_rule_label(pretty_metric: str, rule_key: str) -> str:
        col = available[pretty_metric]
        return f"{pretty_metric} — {rule_key}" if col in ARI_COLS else f"{pretty_metric}"

    for (pretty_i, rules_i) in [(pm, RULES_FOR_COL.get(available[pm], ["max"])) for pm in metric_names]:
        for r in rules_i:
            k = best_k_map.get((pretty_i, r))
            label = _pretty_rule_label(pretty_i, r)
            if pretty_i == default_pretty and r == rule:
                active_rule_index = len(rule_buttons)

            rule_buttons.append(dict(
                label=label,
                method="relayout",
                args=[{
                    "shapes[0].x0": k,
                    "shapes[0].x1": k,
                }],
            ))
        
    # 8) attach dropdown, style legend & grids, return
    menus = fig.layout.updatemenus or []
    menus = list(menus)
    menus.append(dict(
        type="dropdown",
        x=0.0, xanchor="left",
        y=1.2, yanchor="top",
        direction="down",
        buttons=rule_buttons,
        showactive=True,
        active=active_rule_index,
    ))
    fig.update_layout(updatemenus=menus)
    fig.update_layout(uirevision="static")
    fig.update_layout(legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.6)"))
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    fig.show()

def plot_clustering_interactive(
    X: np.ndarray,
    row: pd.Series,
    labels: np.ndarray,
    *,
    stab_gini_vec: np.ndarray,
    stab_ce_vec: np.ndarray,
    gen_vec: np.ndarray,
    measure: Measure = "stability",
    width: int = 1000,
    height: int = 800,
    min_size: float = 20.0,
    max_size: float = 180.0,
    min_alpha: float = 0.30,
    max_alpha: float = 1.00,
    auto_display: bool = True
) -> "go.Figure":
    """Plot interactive clustering scatter and boxplots with Plotly.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    row : pandas.Series
        Selected results row.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    stab_gini_vec : ndarray of shape (n_samples,)
        Per-sample Gini stability scores.
    stab_ce_vec : ndarray of shape (n_samples,)
        Per-sample cross-entropy stability scores.
    gen_vec : ndarray of shape (n_samples,)
        Per-sample generalizability scores.
    measure : Measure, default="stability"
        Metric key used to select the default display.
    width : int, default=1000
        Figure width in pixels.
    height : int, default=800
        Figure height in pixels.
    min_size : float, default=20.0
        Minimum marker size.
    max_size : float, default=180.0
        Maximum marker size.
    min_alpha : float, default=0.30
        Minimum marker alpha.
    max_alpha : float, default=1.00
        Maximum marker alpha.
    auto_display : bool, default=True
        If True, display the figure immediately.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure instance.
    """
    if not _HAS_PLOTLY:
        raise RuntimeError("plotly is not available. Install plotly to use interactive plots.")
    
    n = X.shape[0]
    if any(arr.shape[0] != n for arr in (labels, stab_gini_vec, stab_ce_vec, gen_vec)):
        raise ValueError("All vectors must have length X.shape[0]")

    X_pca = PCA(n_components=2).fit_transform(X)  # PCA for left panel
    x_min, x_max = float(np.nanmin(X_pca[:, 0])), float(np.nanmax(X_pca[:, 0]))
    y_min, y_max = float(np.nanmin(X_pca[:, 1])), float(np.nanmax(X_pca[:, 1]))
    dx = (x_max - x_min) or 1.0
    dy = (y_max - y_min) or 1.0
    x_rng = [x_min - 0.05 * dx, x_max + 0.05 * dx]
    y_rng = [y_min - 0.05 * dy, y_max + 0.05 * dy]

    sample_idx = np.arange(n, dtype=int)  # per sample meta data for hover
    if not np.issubdtype(labels.dtype, np.integer):
        labels = pd.Categorical(labels).codes
    display_labels = (labels.astype(int) + 1).astype(int)  # display labels are 1-based
    customdata_all = np.column_stack([sample_idx, labels, display_labels, stab_gini_vec, stab_ce_vec, gen_vec])

    metric_names = ["Stability (Gini)", "Stability (CE)", "Generalizability"]
    metric_arrays: Dict[str, np.ndarray] = {
        "Stability (Gini)": np.asarray(stab_gini_vec, float),
        "Stability (CE)": np.asarray(stab_ce_vec, float),
        "Generalizability": np.asarray(gen_vec, float),
    }
    default_metric = {
        "stability": "Stability (Gini)",
        "generalizability": "Generalizability",
        "consensus_pac": "Stability (Gini)",
        "consensus_gini": "Stability (Gini)",
        "consensus_ce": "Stability (CE)",
    }.get(measure, "Stability (Gini)")

    # helper functions & compute sizes and alphas
    def _sizes_from_metric(values: np.ndarray, pt2_min: float, pt2_max: float) -> np.ndarray:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or np.isclose(vmin, vmax):
            pt2 = np.full_like(values, 0.5 * (pt2_min + pt2_max), dtype=float)
        else:
            pt2 = pt2_min + (values - vmin) / (vmax - vmin) * (pt2_max - pt2_min)
        
        return np.clip(np.sqrt(pt2), 4.0, 36.0)

    def _cluster_opacity_from_metric(values: np.ndarray, lbls: np.ndarray, a_min: float, a_max: float) -> Dict[int, float]:
        clus = np.unique(lbls)
        means = {int(c): float(np.nanmean(values[lbls == c])) for c in clus}
        vv = np.array(list(means.values()), dtype=float)
        vmin, vmax = float(np.nanmin(vv)), float(np.nanmax(vv))
        
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or np.isclose(vmin, vmax):
            return {int(c): 0.5 * (a_min + a_max) for c in clus}
        
        return {int(c): a_min + (means[int(c)] - vmin) / (vmax - vmin) * (a_max - a_min) for c in clus}

    size_by_metric = {m: _sizes_from_metric(metric_arrays[m], min_size, max_size) for m in metric_names}
    alpha_by_metric = {m: _cluster_opacity_from_metric(metric_arrays[m], labels, min_alpha, max_alpha) for m in metric_names}
    
    # 1) figure scaffolding
    base = make_subplots(
        rows=1, cols=2, 
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.08
    )
    fig = FigureWidget(base)  # callbacks only work in Jupyter; guarding use
    is_widget = isinstance(fig, FigureWidget)
    
    clus = np.unique(labels)
    _palette = qual.Plotly  # handle colors
    colors_for_clusters = {int(c): _palette[i % len(_palette)] for i, c in enumerate(clus)}

    def _rgba(hex_color: str, a: float) -> str:
        r, g, b = hex_to_rgb(hex_color)
        return f"rgba({r},{g},{b},{a})"

    # 2.1) left panel, scatter by cluster
    left_trace_ids: List[int] = []
    left_tid_to_cluster: Dict[int, int] = {}
    
    for c in clus:
        mask = (labels == c)
        
        tr = go.Scatter(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            mode="markers",
            name=f"cluster {int(c)+1}",
            legendgroup=f"cluster-{int(c)}",
            showlegend=True,
            customdata=customdata_all[mask],
            hovertemplate=(
                "sample %{customdata[0]}<br>"
                "cluster %{customdata[2]}<br>"
                "Gini: %{customdata[3]:.4f}<br>"
                "CE: %{customdata[4]:.4f}<br>"
                "Generalizability: %{customdata[5]:.4f}"
                "<extra></extra>"
            ),
            marker=dict(
                color=colors_for_clusters[int(c)],
                size=size_by_metric[default_metric][mask],
                opacity=alpha_by_metric[default_metric][int(c)],
                line=dict(width=1.0, color="black"), 
            )
        )
        
        fig.add_trace(tr, row=1, col=1)
        idx = len(fig.data) - 1
        left_trace_ids.append(idx)
        left_tid_to_cluster[idx] = int(c)

    # 2.2) right panel, per-cluster boxplots
    right_ids_by_metric: Dict[str, Dict[str, List[int]]] = {m: {"box": [], "strip": []} for m in metric_names}
    cat_labels = [str(int(c)+1) for c in clus]
    
    for m in metric_names:
        vals_m = metric_arrays[m]
        
        for c in clus:
            mask = (labels == c)
            cat = str(int(c)+1)

            tr_box = go.Box(
                x=[cat] * int(mask.sum()), y=vals_m[mask],
                name=f"cluster {int(c)}",
                legendgroup=f"cluster-{int(c)+1}-{m}",
                showlegend=False,
                boxpoints=False,
                hoverinfo="skip",
                offsetgroup=cat,
                visible=(m == default_metric),
                width=0.6,
                line=dict(color=colors_for_clusters[int(c)], width=1.5),
                fillcolor=_rgba(colors_for_clusters[int(c)], 0.25)
            )
            fig.add_trace(tr_box, row=1, col=2)
            right_ids_by_metric[m]["box"].append(len(fig.data) - 1)
            
            tr_pts = go.Scatter(
                x=[cat] * int(mask.sum()), y=vals_m[mask],
                mode="markers",
                name=f"points {int(c)+1}",
                legendgroup=f"cluster-{int(c)+1}-{m}",
                showlegend=False,
                customdata=customdata_all[mask],
                hovertemplate=(
                    "cluster %{x}<br>"
                    "sample %{customdata[0]}<br>"
                    "Gini: %{customdata[3]:.4f}<br>"
                    "CE: %{customdata[4]:.4f}<br>"
                    "Generalizability: %{customdata[5]:.4f}"
                    "<extra></extra>"
                ),
                marker=dict(
                    color=colors_for_clusters[int(c)],
                    size=9,
                    opacity=0.9,
                    line=dict(width=0.8, color="black")
                ),
                offsetgroup=cat,
                visible=(m == default_metric),
            )
            fig.add_trace(tr_pts, row=1, col=2)
            right_ids_by_metric[m]["strip"].append(len(fig.data) - 1)
    
    # 2.3) left highlight for hover from right panel
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
    
    # 3) layout, axes, titles
    fig.update_layout(
        uirevision="carve-clustering-v1",
        width=width, height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.6)"),
    )
    fig.update_xaxes(title_text="PC1", range=x_rng, row=1, col=1)
    fig.update_yaxes(title_text="PC2", range=y_rng, row=1, col=1)
    fig.update_xaxes(
        title_text="cluster", type="category",
        categoryorder="array", categoryarray=cat_labels,
        row=1, col=2
    )
    fig.update_yaxes(
        title_text=("per-sample generalizability" if default_metric == "Generalizability" else "per-sample stability"),
        autorange=True,
        row=1, col=2
    )
    fig.update_yaxes(side="left",  row=1, col=1)
    fig.update_yaxes(side="left", row=1, col=2)
    
    # 4) hover callbacks if in widget context
    if is_widget:
        per_cluster_abs_ids = {int(c): sample_idx[labels == c] for c in clus}

        def _mk_hover_cb(trace_index: int):
            def _on_hover(trace, points, state):
                if points.point_inds:
                    local_idx = points.point_inds[0]
                    
                    c = int(fig.data[trace_index].customdata[local_idx, 1])  # Read cluster from the hovered right-trace's customdata
                    abs_id = int(per_cluster_abs_ids[c][local_idx])
                    xy = X_pca[abs_id]
                    
                    with fig.batch_update():
                        fig.data[highlight_trace_index].x = [xy[0]]
                        fig.data[highlight_trace_index].y = [xy[1]]
                        
            return _on_hover

        def _mk_unhover_cb(_trace_index: int):
            def _on_unhover(trace, points, state):
                with fig.batch_update():
                    fig.data[highlight_trace_index].x = []
                    fig.data[highlight_trace_index].y = []
                    
            return _on_unhover

        for m in metric_names:
            for tidx in right_ids_by_metric[m]["strip"]:
                fig.data[tidx].on_hover(_mk_hover_cb(tidx))
                fig.data[tidx].on_unhover(_mk_unhover_cb(tidx))
                
    # 5) helpers for dropdown switching
    right_ids_all = []
    for m in metric_names:
        right_ids_all.extend(right_ids_by_metric[m]["box"])
        right_ids_all.extend(right_ids_by_metric[m]["strip"])
    
    right_ids_all = set(right_ids_all)

    def _visible_mask_for_metric(metric_label: str) -> List[bool]:
        total = len(fig.data)
        vis = [True] * total
        
        for tid in right_ids_all:  # hide all right-panel traces, then enable only chosen metric's right traces
            vis[tid] = False
        for tid in (right_ids_by_metric[metric_label]["box"] + right_ids_by_metric[metric_label]["strip"]):
            vis[tid] = True
        
        vis[highlight_trace_index] = True  # always keep the left highlight trace visible
        return vis

    def _style_for_metric(metric_label: str) -> Dict[str, Any]:
        sizes = _sizes_from_metric(metric_arrays[metric_label], min_size, max_size)
        alphas = _cluster_opacity_from_metric(metric_arrays[metric_label], labels, min_alpha, max_alpha)
        updates = {"marker.size": [], "marker.opacity": [], "visible": _visible_mask_for_metric(metric_label)}
        
        for tidx, tr in enumerate(fig.data):
            if tidx in left_tid_to_cluster:
                c = left_tid_to_cluster[tidx]
                mask = (labels == c)
                updates["marker.size"].append(sizes[mask])  # per-point vector for that cluster
                updates["marker.opacity"].append(alphas[c])  # scalar per cluster
            
            else:
                ms = getattr(getattr(tr, "marker", None), "size", None)  # keep existing for non-left traces
                mo = getattr(getattr(tr, "marker", None), "opacity", None)
                updates["marker.size"].append(ms)
                updates["marker.opacity"].append(mo)
                
        return updates

    def _y2_title(metric_label: str) -> str:
        return "per-sample generalizability" if metric_label == "Generalizability" else "per-sample value"

    # 6) dropdown: metric switcher (updates styles + right y-axis)
    buttons = [
        dict(
            label=m,
            method="update",
            args=[
                _style_for_metric(m),  # trace updates (sizes/alphas/visibility)
                {"yaxis2": {"title": _y2_title(m), "autorange": True}},  # layout updates for right axis
            ],
        )
        for m in metric_names
    ]
    fig.update_layout(updatemenus=[dict(
        type="dropdown",
        x=0.99, xanchor="right",
        y=1.18, yanchor="top",
        buttons=buttons
    )])
    
    # 7) dynamic title from row params
    params: Dict[str, str] = {}
    for key, value in row.items():
        if key in NON_PARAM_COLS or pd.isna(value):
            continue
        
        params[key] = _fmt_float(value) if isinstance(value, float) else str(value)
        
    est = str(row.get("estimator", ""))
    k_val = row.get("n_clusters", "")
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    title = f"{est} | k={k_val}" + (f", {param_str}" if param_str else "")
    
    if len(title) > 220:
        title = title[:219] + "…"
        
    fig.update_layout(title_text=title, title_x=0.5, title_font=dict(size=14))

    if auto_display:
        try:
            display(fig)
            return
        except Exception:
            pass
        
    return fig

# ––––– Debugging Plotting Functions ––––– 
def plot_ari_hist():
    """Placeholder for ARI histogram plotting."""
    ...
    pass

def plot_cluster_stability():
    """Placeholder for cluster stability plotting."""
    ...
    pass