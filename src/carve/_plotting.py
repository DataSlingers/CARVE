"""Plotting functions for CARVE validation results.

This module provides publication-ready visualizations for CARVE results,
following the conventions of scanpy and other scientific Python packages.
"""

from pathlib import Path
from typing import Literal, Optional, Sequence, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA

from ._selection import MEASURE_MAP, select_best_k


def _build_estimator_label(row: pd.Series, exclude_cols: set) -> str:
    """Build a human-readable estimator label from a results row.
    
    Parameters
    ----------
    row : pd.Series
        A row from estimator_results_ DataFrame.
    exclude_cols : set
        Column names to exclude from the label (e.g., n_clusters, metrics).
    
    Returns
    -------
    label : str
        Human-readable estimator label.
    """
    parts = [row["estimator"]]
    
    for col in row.index:
        if col not in exclude_cols and not pd.isna(row[col]):
            val = row[col]
            # Format numeric values nicely
            if isinstance(val, (int, np.integer)):
                parts.append(f"{col}={val}")
            elif isinstance(val, (float, np.floating)):
                if val == int(val):
                    parts.append(f"{col}={int(val)}")
                else:
                    parts.append(f"{col}={val:.3g}")
            else:
                parts.append(f"{col}={val}")
    
    return ", ".join(parts)


def plot_metric_over_n_clusters(
    results_df: pd.DataFrame,
    *,
    measure: str = "generalizability",
    rule: str = "1se",
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    legend_loc: str = "best",
    palette: Optional[str] = None,
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    **kwargs,
) -> Axes:
    """Plot stability or generalizability metric across n_clusters.
    
    Creates a line plot with one line per unique estimator configuration
    (estimator name + hyperparameters, excluding n_clusters). Error bars
    are drawn at ±1 standard error. A vertical dashed line indicates the
    selected k according to the specified rule.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from CARVE.fit(), containing columns:
        "estimator", "n_clusters", metric columns, and hyperparameter columns.
    measure : str, default="stability"
        Metric to plot. Options: "stability", "ari_stability", "generalizability",
        "ari_generalizability", "average", "ari_average", "pac",
        "consensus_pac_stability", "gini", "consensus_gini_stability",
        "ce", "consensus_ce_stability", "misclassification", etc.
    rule : str, default="1se"
        Selection rule for choosing best k. Options: "max", "1se", "quantile".
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (8, 5).
    title : str, optional
        Figure title. 
    xlabel : str, optional
        X-axis label. Default is "Number of Clusters (k)".
    ylabel : str, optional
        Y-axis label. If None, auto-generated from measure name.
    legend : bool, default=True
        Whether to display a legend.
    legend_loc : str, default="best"
        Legend location (passed to ax.legend).
    palette : str, optional
        Matplotlib colormap name for line colors. Default is "tab10".
    show : bool, default=False
        Whether to call plt.show() before returning.
    save : str or Path, optional
        Path to save the figure (e.g., "plot.pdf"). If provided, figure is
        saved and None is returned instead of the Axes object.
    dpi : int, default=300
        Dots per inch for saved figures.
    **kwargs
        Additional keyword arguments passed to ax.errorbar() for line styling.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object on which the plot was drawn, or None if save is used.
    
    Raises
    ------
    RuntimeError
        If results_df is empty.
    ValueError
        If measure is not found in the results.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from carve.api import CARVE
    >>> carve = CARVE().fit(X)
    >>> ax = carve.plot_metric_over_n_clusters(measure="stability", rule="1se")
    >>> plt.show()
    
    >>> # Save to file
    >>> carve.plot_metric_over_n_clusters(measure="generalizability", save="results.pdf")
    """
    if results_df.empty:
        raise RuntimeError("Results DataFrame is empty.")
    
    # Normalize measure name
    if measure not in MEASURE_MAP:
        raise ValueError(
            f"Measure {measure!r} not found. Valid options: {list(MEASURE_MAP.keys())}"
        )
    measure_col = MEASURE_MAP[measure]
    se_col = f"{measure_col}_se"
    
    if measure_col not in results_df.columns:
        raise ValueError(
            f"Metric column {measure_col!r} not found in results_df."
        )
    if se_col not in results_df.columns:
        raise ValueError(
            f"Standard error column {se_col!r} not found in results_df."
        )
    
    # Create figure if needed
    if ax is None:
        if figsize is None:
            figsize = (9, 5.5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Define columns to exclude from estimator label
    metric_cols = {
        col for col in results_df.columns
        if any(x in col for x in ["ari_", "consensus_", "misclassification_", "_se", "_upper", "_lower"])
    }
    exclude_cols = metric_cols | {"estimator", "n_clusters", "index"}
    
    # Group by estimator name + params (excluding n_clusters)
    # Create a grouping key
    group_cols = [c for c in results_df.columns if c not in exclude_cols and c != "n_clusters"]
    grouped = results_df.groupby(group_cols, dropna=False)
    
    # Set up color palette
    if palette is None:
        palette = "tab10"
    colors = plt.cm.get_cmap(palette)(np.linspace(0, 1, len(grouped)))
    
    # Plot each estimator
    for color_idx, (group_key, group_df) in enumerate(grouped):
        # Get label
        if isinstance(group_key, tuple):
            label_row = pd.Series(dict(zip(group_cols, group_key)))
            label_row["estimator"] = group_df.iloc[0]["estimator"]
        else:
            label_row = group_df.iloc[0]
        
        label = _build_estimator_label(label_row, exclude_cols)
        
        # Sort by n_clusters for plotting
        group_df_sorted = group_df.sort_values("n_clusters")
        
        # Extract data
        x = group_df_sorted["n_clusters"].values
        y = group_df_sorted[measure_col].values
        yerr = group_df_sorted[se_col].values
        
        # Get color
        color = colors[color_idx % len(colors)]
        
        # Plot with error bars
        ax.errorbar(
            x, y,
            yerr=yerr,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=4,
            capthick=1.5,
            alpha=0.8,
            color=color,
            label=label,
            **kwargs,
        )
    
    # Add vertical line at selected k
    try:
        best_k = select_best_k(results_df, measure=measure, rule=rule)
        rule_str = '1-SE' if rule == "1se" else rule.title()
        ax.axvline(
            best_k,
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label=f"Selected k ({rule_str} rule): {best_k}",
            zorder=0,
        )
    except Exception:
        # Skip if selection fails (e.g., due to missing columns)
        pass
    
    # Labels and formatting
    if xlabel is None:
        xlabel = "Number of Clusters (k)"
    ax.set_xlabel(xlabel, fontsize=12)
    
    if ylabel is None:
        # Generate smart label from measure name
        ylabel = measure_col.replace("_", " ").title()
        ylabel = ylabel.replace("Ari", "ARI")
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title is not None:
        ax.set_title(title, fontsize=13, pad=15)
    
    # Set x-axis to integer ticks
    n_clusters_unique = sorted(results_df["n_clusters"].unique())
    if len(n_clusters_unique) <= 20:
        ax.set_xticks(n_clusters_unique)
    
    # Legend
    if legend:
        ax.legend(
            loc=legend_loc,
            frameon=True,
            framealpha=0.95,
            fontsize=10,
            title="Estimators" if rule else None,
        )
    
    # Clean up
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    
    # Save or show
    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None
    
    if show:
        plt.show()
    
    return ax


def plot_consensus_matrix(
    consensus_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
    cmap: str = "viridis",
    cluster_palette: str = "tab20",
    colorbar: bool = True,
    colorbar_label: str = "Consensus",
    title: Optional[str] = None,
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Axes:
    """Plot a consensus matrix with a flush top cluster-color band.

    Samples are ordered by cluster labels before plotting.

    Parameters
    ----------
    consensus_matrix : ndarray of shape (n_samples, n_samples)
        Consensus similarity matrix in [0, 1].
    labels : ndarray of shape (n_samples,)
        Cluster labels for the selected matrix.
    ax : matplotlib.axes.Axes, optional
        Main axis for the heatmap. If None, a new figure is created.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (6.5, 6.5).
    cmap : str, default="viridis"
        Colormap for the consensus heatmap.
    cluster_palette : str, default="tab20"
        Discrete colormap for the top cluster band.
    colorbar : bool, default=True
        Whether to draw a colorbar for the heatmap.
    colorbar_label : str, default="Consensus"
        Label shown on the colorbar.
    title : str, optional
        Figure title.
    show : bool, default=False
        Whether to call plt.show() before returning.
    save : str or Path, optional
        Path to save the figure. If provided, figure is saved and None is returned.
    dpi : int, default=300
        Dots per inch for saved figures.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The heatmap axis, or None if ``save`` is provided.
    """
    M = np.asarray(consensus_matrix, dtype=float)
    labels = np.asarray(labels)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("consensus_matrix must be a square 2D array.")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if M.shape[0] != labels.shape[0]:
        raise ValueError(
            "consensus_matrix and labels must have matching first dimension."
        )

    # Ensure symmetric, bounded matrix for display.
    M = 0.5 * (M + M.T)
    M = np.clip(np.nan_to_num(M, nan=0.5), 0.0, 1.0)
    np.fill_diagonal(M, 1.0)

    # Stable ordering by cluster id for contiguous cluster blocks.
    order = np.argsort(labels, kind="stable")
    M_ord = M[np.ix_(order, order)]
    labels_ord = labels[order]

    unique_labels = np.unique(labels_ord)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    band_values = np.array([label_to_idx[lab] for lab in labels_ord], dtype=int)[None, :]

    if ax is None:
        if figsize is None:
            figsize = (6.5, 6.5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        M_ord,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )

    divider = make_axes_locatable(ax)
    band_ax = divider.append_axes("top", size="4%", pad=0.0)
    band_cmap = plt.get_cmap(cluster_palette, max(len(unique_labels), 1))
    band_ax.imshow(
        band_values,
        cmap=band_cmap,
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=max(len(unique_labels) - 1, 1),
    )

    boundaries = np.where(np.diff(labels_ord) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="white", lw=0.6, alpha=0.8)
        ax.axvline(b, color="white", lw=0.6, alpha=0.8)
        band_ax.axvline(b, color="white", lw=0.8, alpha=0.9)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Samples (ordered by cluster)")
    # ax.set_ylabel("Samples (ordered by cluster)")

    band_ax.set_xticks([])
    band_ax.set_yticks([])
    for spine in band_ax.spines.values():
        spine.set_visible(False)

    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(colorbar_label)
        
    if title is not None:
        ax.set_title(title, pad=25)

    fig.tight_layout()

    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    if show:
        plt.show()

    return ax


def _cluster_distribution_inputs(
    uncertainty_scores: np.ndarray,
    labels: np.ndarray,
    *,
    order: Optional[Sequence[Union[int, str]]] = None,
) -> tuple[list[np.ndarray], list[Union[int, str]]]:
    """Prepare per-cluster score arrays and plotting order."""
    scores = np.asarray(uncertainty_scores, dtype=float)
    labels = np.asarray(labels)

    if scores.ndim != 1:
        raise ValueError("uncertainty_scores must be a 1D array.")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("uncertainty_scores and labels must have matching length.")

    mask = np.isfinite(scores)
    scores = scores[mask]
    labels = labels[mask]
    if scores.size == 0:
        raise ValueError("No finite uncertainty scores available for plotting.")

    if order is None:
        unique_labels = np.unique(labels)
        try:
            sorted_labels = sorted(unique_labels, key=lambda x: float(x))
        except Exception:
            sorted_labels = sorted(unique_labels, key=lambda x: str(x))
        order_vals = list(sorted_labels)
    else:
        order_vals = list(order)

    groups: list[np.ndarray] = []
    kept_order: list[Union[int, str]] = []
    for lab in order_vals:
        vals = scores[labels == lab]
        if vals.size > 0:
            groups.append(vals)
            kept_order.append(lab)

    if not groups:
        raise ValueError("No cluster values found for the provided order.")

    return groups, kept_order


def plot_cluster_uncertainty_boxplot(
    uncertainty_scores: np.ndarray,
    labels: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
    order: Optional[Sequence[Union[int, str]]] = None,
    palette: str = "tab20",
    showfliers: bool = False,
    width: float = 0.75,
    title: Optional[str] = None,
    xlabel: str = "Cluster",
    ylabel: str = "Uncertainty",
    rotation: Optional[float] = None,
    ylim: tuple[float, float] = (0.0, 1.0),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Axes:
    """Plot per-cluster uncertainty distributions as a boxplot."""
    groups, kept_order = _cluster_distribution_inputs(
        uncertainty_scores, labels, order=order
    )

    if ax is None:
        if figsize is None:
            figsize = (6.5, 4.0)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap = plt.get_cmap(palette, max(len(groups), 1))
    bp = ax.boxplot(
        groups,
        patch_artist=True,
        widths=width,
        showfliers=showfliers,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.2},
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i))
        patch.set_alpha(0.8)

    ax.set_xticks(np.arange(1, len(kept_order) + 1))
    ax.set_xticklabels([str(x) for x in kept_order], rotation=rotation)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if ylim is not None:
        ax.set_ylim(*ylim)
        
    if title is not None:
        ax.set_title(title)

    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    if show:
        plt.show()

    return ax


def plot_cluster_uncertainty_violin(
    uncertainty_scores: np.ndarray,
    labels: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
    order: Optional[Sequence[Union[int, str]]] = None,
    palette: str = "tab20",
    density_norm: Literal["width", "area", "count"] = "width",
    stripplot: bool = True,
    jitter: Union[bool, float] = True,
    size: float = 8.0,
    alpha: float = 0.22,
    inner: Literal["box", "quartile", "none"] = "box",
    title: Optional[str] = None,
    xlabel: str = "Cluster",
    ylabel: str = "Uncertainty",
    rotation: Optional[float] = None,
    ylim: tuple[float, float] = (0.0, 1.0),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Axes:
    """Plot per-cluster uncertainty distributions as a violin plot.

    The interface mirrors common Scanpy options (`stripplot`, `jitter`,
    `size`, `density_norm`, `show`, `ax`, `save`) while using matplotlib
    primitives to avoid an additional plotting dependency.
    """
    groups, kept_order = _cluster_distribution_inputs(
        uncertainty_scores, labels, order=order
    )

    if ax is None:
        if figsize is None:
            figsize = (6.5, 4.0)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    widths = 0.8
    if density_norm == "count":
        max_n = max(len(g) for g in groups)
        widths = [0.2 + 0.6 * (len(g) / max_n) for g in groups]

    vp = ax.violinplot(
        groups,
        positions=np.arange(1, len(groups) + 1),
        widths=widths,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    cmap = plt.get_cmap(palette, max(len(groups), 1))
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(cmap(i))
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(0.8)

    for pos, vals in enumerate(groups, start=1):
        if inner == "box":
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([pos - 0.13, pos + 0.13], [med, med], color="black", lw=1.5)
            ax.plot([pos, pos], [q1, q3], color="black", lw=1.2)
            ax.plot([pos - 0.08, pos + 0.08], [q1, q1], color="black", lw=1.0)
            ax.plot([pos - 0.08, pos + 0.08], [q3, q3], color="black", lw=1.0)
        elif inner == "quartile":
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([pos - 0.13, pos + 0.13], [q1, q1], color="black", lw=1.0)
            ax.plot([pos - 0.13, pos + 0.13], [med, med], color="black", lw=1.5)
            ax.plot([pos - 0.13, pos + 0.13], [q3, q3], color="black", lw=1.0)

        if stripplot:
            if jitter is True:
                jitter_width = 0.11
            elif jitter is False:
                jitter_width = 0.0
            else:
                jitter_width = float(jitter)
            x = pos + np.random.uniform(-jitter_width, jitter_width, size=len(vals))
            ax.scatter(x, vals, s=size, alpha=alpha, color="black", linewidths=0)

    if density_norm not in {"width", "area", "count"}:
        warnings.warn(
            f"Unknown density_norm={density_norm!r}; using 'width'.",
            RuntimeWarning,
            stacklevel=2,
        )

    ax.set_xticks(np.arange(1, len(kept_order) + 1))
    ax.set_xticklabels([str(x) for x in kept_order], rotation=rotation)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if ylim is not None:
        ax.set_ylim(*ylim)
        
    if title is not None:
        ax.set_title(title)

    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    if show:
        plt.show()

    return ax


def plot_cluster_score_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    sample_scores: np.ndarray,
    *,
    embedding: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
    palette: str = "tab20",
    alpha_range: Optional[tuple[float, float]] = None,
    size_range: tuple[float, float] = (20.0, 100.0),
    sort_order: bool = True,
    legend: bool = True,
    legend_loc: str = "right margin",
    title: Optional[str] = None,
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    frameon: bool = False,
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Axes:
    """Scatter plot with cluster opacity and sample-size score encoding.

    Cluster-level mean scores are mapped to alpha, while sample-level scores
    are mapped to marker size.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    scores = np.asarray(sample_scores, dtype=float)

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if scores.ndim != 1:
        raise ValueError("sample_scores must be a 1D array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] != labels.shape[0] or X.shape[0] != scores.shape[0]:
        raise ValueError("X, labels, and sample_scores must have matching n_samples.")

    valid = np.isfinite(scores)
    if not np.any(valid):
        raise ValueError("No finite sample_scores available for plotting.")

    if embedding is None:
        if X.shape[1] >= 2:
            coords = X[:, :2]
        elif X.shape[1] == 1:
            coords = np.column_stack([X[:, 0], np.zeros(X.shape[0], dtype=float)])
        else:
            raise ValueError("X must have at least 1 feature for scatter plotting.")
        if X.shape[1] > 2:
            # Scanpy commonly visualizes reduced 2D embeddings; use PCA fallback.
            coords = PCA(n_components=2, random_state=0).fit_transform(X)
    else:
        coords = np.asarray(embedding, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("embedding must be a 2D array with at least 2 columns.")
        if coords.shape[0] != X.shape[0]:
            raise ValueError("embedding must have the same number of rows as X.")
        coords = coords[:, :2]

    lo_score = np.nanmin(scores[valid])
    hi_score = np.nanmax(scores[valid])
    if np.isclose(lo_score, hi_score):
        size_vals = np.full(scores.shape[0], np.mean(size_range), dtype=float)
    else:
        size_vals = size_range[0] + (scores - lo_score) * (
            (size_range[1] - size_range[0]) / (hi_score - lo_score)
        )

    uniq = np.unique(labels)
    try:
        uniq = np.array(sorted(uniq, key=lambda x: float(x)))
    except Exception:
        uniq = np.array(sorted(uniq, key=lambda x: str(x)))

    cluster_mean = {}
    for lab in uniq:
        lab_scores = scores[(labels == lab) & valid]
        cluster_mean[lab] = float(np.nanmean(lab_scores)) if lab_scores.size else np.nan

    if alpha_range is not None:
        alpha_lo, alpha_hi = alpha_range
        means = np.array([cluster_mean[lab] for lab in uniq], dtype=float)
        finite_means = means[np.isfinite(means)]
        if finite_means.size == 0 or np.isclose(finite_means.min(), finite_means.max()):
            alpha_map = {lab: float(np.mean(alpha_range)) for lab in uniq}
        else:
            m0, m1 = finite_means.min(), finite_means.max()
            alpha_map = {
                lab: float(alpha_lo + (cluster_mean[lab] - m0) * ((alpha_hi - alpha_lo) / (m1 - m0)))
                if np.isfinite(cluster_mean[lab])
                else float(alpha_lo)
                for lab in uniq
            }
    else:
        alpha_map = {
            lab: float(cluster_mean[lab])
            if np.isfinite(cluster_mean[lab])
            else float(1.0)
            for lab in uniq
        }
        
    label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    cmap = plt.get_cmap(palette, max(len(uniq), 1))
    rgba = np.zeros((labels.shape[0], 4), dtype=float)
    for i, lab in enumerate(labels):
        base = cmap(label_to_idx[lab])
        rgba[i, :3] = base[:3]
        rgba[i, 3] = np.clip(alpha_map[lab], 0.0, 1.0)

    order = np.arange(labels.shape[0])
    if sort_order:
        order = np.argsort(rgba[:, 3], kind="stable")

    if ax is None:
        if figsize is None:
            figsize = (6.5, 5.0)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        s=size_vals[order],
        c=rgba[order],
        linewidths=0.2,
        edgecolor="black",
        rasterized=(coords.shape[0] > 5000),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title is not None:
        ax.set_title(title)

    if not frameon:
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax.grid(False)

    if legend:
        from matplotlib.lines import Line2D

        handles = []
        for lab in uniq:
            col = cmap(label_to_idx[lab])
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    label=f"{lab} (mean={cluster_mean[lab]:.2f})",
                    markerfacecolor=(col[0], col[1], col[2], np.clip(alpha_map[lab], 0.0, 1.0)),
                    markeredgecolor="none",
                    markersize=7,
                )
            )

        if legend_loc == "right margin":
            ax.legend(
                handles=handles,
                title="Cluster",
                frameon=False,
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
                borderaxespad=0.0,
            )
        else:
            ax.legend(handles=handles, title="Cluster", frameon=False, loc=legend_loc)

    fig.tight_layout()

    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    if show:
        plt.show()

    return ax
