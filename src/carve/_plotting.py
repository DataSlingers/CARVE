"""Plotting functions for CARVE validation results.

This module provides publication-ready visualizations for CARVE results,
following the conventions of scanpy and other scientific Python packages.
"""

from pathlib import Path
from typing import Literal
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA

from ._selection import MEASURE_MAP, select_best_k

_GROUPBY_NA_SENTINEL = "_NA_"

_DEFAULT_DIAGNOSTIC_MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "*",
    "X",
    "p",
    "h",
    "<",
    ">",
    "d",
    "8",
]


def _build_estimator_label(
    row: pd.Series, exclude_cols: set, *, tight_layout: bool = False
) -> str:
    """Build a human-readable estimator label from a results row.

    Parameters
    ----------
    row : pd.Series
        A row from estimator_results_ DataFrame.
    exclude_cols : set
        Column names to exclude from the label (e.g., n_clusters, metrics).
    tight_layout : bool, default=False
        Whether to separate with line feeds.

    Returns
    -------
    label : str
        Human-readable estimator label.
    """
    parts = [row["estimator"]]

    for col in row.index:
        if (
            col not in exclude_cols
            and not pd.isna(row[col])
            and row[col] != _GROUPBY_NA_SENTINEL
        ):
            val = row[col]
            if isinstance(val, (int, np.integer)):
                parts.append(f"{col}={val}")
            elif isinstance(val, (float, np.floating)):
                if val == int(val):
                    parts.append(f"{col}={int(val)}")
                else:
                    parts.append(f"{col}={val:.3g}")
            else:
                parts.append(f"{col}={val}")

    return ", ".join(parts) if not tight_layout else "\n".join(parts)


def plot_metric_over_n_clusters(
    results_df: pd.DataFrame,
    *,
    measure: str = "generalizability",
    rule: str = "1se",
    not_two: bool = False,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    palette: str = "Accent",
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
    **kwargs,
) -> Axes:
    """Plot stability or generalizability metric across n_clusters.

    Creates a line plot with one line per unique estimator configuration
    (estimator name + hyperparameters, excluding n_clusters). Error bars
    are drawn at +/-1 standard error. A vertical dashed line indicates the
    selected k according to the specified rule.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from CARVE.fit(), containing columns:
        "estimator", "n_clusters", metric columns, and hyperparameter columns.
    measure : str, default="generalizability"
        Metric to plot. Options: "stability", "ari_stability", "generalizability",
        "ari_generalizability", "average", "ari_average", "pac",
        "consensus_pac_stability", "gini", "consensus_gini_stability",
        "ce", "consensus_ce_stability", "accuracy", etc.
    rule : str, default="1se"
        Selection rule for choosing best k. Options: "max", "1se", "quantile".
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (9, 5.5).
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
    palette : str, default="Accent"
        Matplotlib colormap name for line colors. Default is "Accent".
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

    if measure not in MEASURE_MAP:
        raise ValueError(
            f"Measure {measure!r} not found. Valid options: {list(MEASURE_MAP.keys())}"
        )
    measure_col = MEASURE_MAP[measure]
    se_col = f"{measure_col}_se"

    if measure_col not in results_df.columns:
        raise ValueError(f"Metric column {measure_col!r} not found in results_df.")
    has_se = se_col in results_df.columns

    # --- Figure setup ---
    if ax is None:
        if figsize is None:
            figsize = (9, 5.5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- Identify metric and grouping columns ---
    metric_cols = {
        col
        for col in results_df.columns
        if any(
            x in col
            for x in [
                "ari_",
                "consensus_",
                "accuracy_",
                "_se",
                "_upper",
                "_lower",
            ]
        )
    }
    exclude_cols = metric_cols | {"estimator", "n_clusters", "index"}

    group_cols = [
        c for c in results_df.columns if c not in exclude_cols and c != "n_clusters"
    ]
    results_df = results_df.copy()
    results_df[group_cols] = results_df[group_cols].fillna(_GROUPBY_NA_SENTINEL)
    grouped = results_df.groupby(group_cols)

    colors = plt.cm.get_cmap(palette)(np.linspace(0, 1, len(grouped)))

    # --- Plot each estimator configuration ---
    for color_idx, (group_key, group_df) in enumerate(grouped):
        if isinstance(group_key, tuple):
            label_row = pd.Series(dict(zip(group_cols, group_key)))
            label_row["estimator"] = group_df.iloc[0]["estimator"]
        else:
            label_row = group_df.iloc[0]

        label = _build_estimator_label(label_row, exclude_cols)
        group_df_sorted = group_df.sort_values("n_clusters")

        x = group_df_sorted["n_clusters"].values
        y = group_df_sorted[measure_col].values
        yerr = group_df_sorted[se_col].values if has_se else None
        color = colors[color_idx % len(colors)]

        ax.errorbar(
            x,
            y,
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

    # --- Vertical line at selected k ---
    try:
        best_k = select_best_k(results_df, measure=measure, rule=rule, not_two=not_two)
        rule_str = "1-SE" if rule == "1se" else rule.title()
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
        pass

    # --- Labels and formatting ---
    if xlabel is None:
        xlabel = "Number of Clusters (k)"
    ax.set_xlabel(xlabel, fontsize=12)

    if ylabel is None:
        ylabel = measure_col.replace("_", " ").title()
        ylabel = ylabel.replace("Ari", "ARI")
    ax.set_ylabel(ylabel, fontsize=12)

    if title is not None:
        ax.set_title(title, fontsize=13, pad=15)

    n_clusters_unique = sorted(results_df["n_clusters"].unique())
    if len(n_clusters_unique) <= 20:
        ax.set_xticks(n_clusters_unique)

    if legend:
        ax.legend(
            loc=legend_loc,
            frameon=True,
            framealpha=0.95,
            fontsize=10,
            title="Estimators" if rule else None,
        )

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    # --- Save or show ---
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
    ax: Axes | None = None,
    figsize: tuple | None = None,
    cmap: str = "viridis",
    palette: str = "Accent",
    colorbar: bool = True,
    colorbar_label: str = "Consensus",
    title: str | None = None,
    show: bool = False,
    save: str | Path | None = None,
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
    palette : str, default="Accent"
        Discrete colormap for the top cluster band. Default is "Accent".
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

    # Ensure symmetric, bounded matrix for display
    M = 0.5 * (M + M.T)
    M = np.clip(np.nan_to_num(M, nan=0.5), 0.0, 1.0)
    np.fill_diagonal(M, 1.0)

    # Stable ordering by cluster id for contiguous blocks
    order = np.argsort(labels, kind="stable")
    M_ord = M[np.ix_(order, order)]
    labels_ord = labels[order]

    unique_labels = np.unique(labels_ord)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    band_values = np.array([label_to_idx[lab] for lab in labels_ord], dtype=int)[
        None, :
    ]

    # --- Figure setup ---
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

    # --- Cluster color band ---
    divider = make_axes_locatable(ax)
    band_ax = divider.append_axes("top", size="4%", pad=0.0)
    band_cmap = plt.get_cmap(palette, max(len(unique_labels), 1))
    band_ax.imshow(
        band_values,
        cmap=band_cmap,
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=max(len(unique_labels) - 1, 1),
    )

    # --- Cluster boundary lines ---
    boundaries = np.where(np.diff(labels_ord) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="white", lw=0.6, alpha=0.8)
        ax.axvline(b, color="white", lw=0.6, alpha=0.8)
        band_ax.axvline(b, color="white", lw=0.8, alpha=0.9)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Samples (ordered by cluster)")

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


def _prepare_cluster_score_groups(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    order: list[int | str] | None = None,
) -> tuple[list[np.ndarray], list[int | str]]:
    """Prepare per-cluster score arrays and plotting order.

    Parameters
    ----------
    scores : ndarray of shape (n_samples,)
        Per-sample scores.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    order : list of int or str, optional
        Explicit cluster ordering for the x-axis.

    Returns
    -------
    groups : list of ndarray
        Score arrays, one per cluster.
    kept_order : list of int or str
        Cluster labels in plotting order.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)

    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array.")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have matching length.")

    mask = np.isfinite(scores)
    scores = scores[mask]
    labels = labels[mask]
    if scores.size == 0:
        raise ValueError("No finite scores available for plotting.")

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
    kept_order: list[int | str] = []
    for lab in order_vals:
        vals = scores[labels == lab]
        if vals.size > 0:
            groups.append(vals)
            kept_order.append(lab)

    if not groups:
        raise ValueError("No cluster values found for the provided order.")

    # Enumerate clusters starting at 1
    kept_order = [int(lab) + 1 for lab in kept_order]

    return groups, kept_order


def _place_adaptive_annotation(
    ax: Axes,
    text: str,
    *,
    replace_legend: bool = True,
) -> Legend:
    """Place a text annotation at the least-obstructed axes location.

    Uses matplotlib's ``loc='best'`` legend placement with an invisible
    handle so the annotation box avoids plotted data automatically,
    mirroring the adaptive legend in :func:`plot_metric_over_n_clusters`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to place the annotation.
    text : str
        Annotation text. May contain newlines for multi-line display.
    replace_legend : bool, default=True
        Whether to create the annotation via ``ax.legend`` (replacing the
        current Axes legend). If False, creates a standalone ``Legend`` artist
        and adds it to the axes without replacing any existing legend.

    Returns
    -------
    matplotlib.legend.Legend
        The created annotation legend artist.
    """
    phantom = Line2D([], [], linestyle="none", marker="none")
    legend_kwargs = dict(
        handles=[phantom],
        labels=[text],
        loc="best",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        fontsize=9,
        handlelength=0,
        handletextpad=0,
        borderpad=0.6,
    )

    if replace_legend:
        leg = ax.legend(**legend_kwargs)
    else:
        leg = Legend(
            ax,
            legend_kwargs["handles"],
            legend_kwargs["labels"],
            **{
                k: v for k, v in legend_kwargs.items() if k not in {"handles", "labels"}
            },
        )
        ax.add_artist(leg)

    leg.set_zorder(5)
    leg.set_in_layout(False)
    return leg


def _get_annotation(
    measure: str,
    rule: str,
    k: int | None,
    estimator_results: pd.DataFrame,
    row: pd.Series,
    selected_k: int | None,
    tight_layout: bool = False,
) -> str:
    """Build an annotation string describing the selected model and criteria.

    Combines the human-readable estimator label (via
    :func:`_build_estimator_label`) with the selection measure, rule, and
    chosen *k* into a two-line summary suitable for plot annotations.

    Parameters
    ----------
    measure : str
        Metric key used for model selection (e.g., ``"generalizability"``).
    rule : str
        Selection rule (``"max"``, ``"1se"``, or ``"quantile"``).
    k : int or None
        User-supplied number of clusters, or None if *k* was selected
        automatically. When not None, the annotation marks *k* as fixed.
    estimator_results : pd.DataFrame
        Full results DataFrame, used to identify metric columns that
        should be excluded from the estimator label.
    row : pd.Series
        The selected row from ``estimator_results``.
    selected_k : int or None
        The resolved number of clusters shown in the annotation.
    tight_layout : bool, default=False
        Whether to separate label components with line feeds.

    Returns
    -------
    annotation : str
        Two-line annotation string of the form
        ``"<estimator> (k = <n>[, fixed])\n<Measure>, <Rule> rule"``.
    """
    metric_cols = {
        col
        for col in estimator_results.columns
        if any(
            x in col
            for x in [
                "ari_",
                "consensus_",
                "accuracy_",
                "_se",
                "_upper",
                "_lower",
            ]
        )
    }

    exclude_cols = metric_cols | {"estimator", "n_clusters", "index"}
    model_label = _build_estimator_label(row, exclude_cols, tight_layout=tight_layout)

    rule_str = "1-SE" if rule == "1se" else rule.title()
    measure_str = measure.replace("_", " ").title()

    if k is not None:
        annotation_text = (
            f"{model_label} (k = {selected_k}, fixed)\n{measure_str}, {rule_str} rule"
        )

    else:
        annotation_text = (
            f"{model_label} (k = {selected_k})\n{measure_str}, {rule_str} rule"
        )

    if tight_layout:
        annotation_text += "\n"

    return annotation_text


def plot_cluster_boxplot(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    order: list[int | str] | None = None,
    palette: str = "Accent",
    showfliers: bool = False,
    width: float = 0.75,
    title: str | None = None,
    xlabel: str = "Cluster",
    ylabel: str = "Uncertainty",
    annotation: str | None = None,
    rotation: float | None = None,
    ylim: tuple[float, float] = (-0.02, 1.02),
    fit_ylim: bool = True,
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes:
    """Plot per-cluster score distributions as a boxplot.

    Parameters
    ----------
    scores : ndarray of shape (n_samples,)
        Per-sample scores.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (6.5, 4.0).
    order : list of int or str, optional
        Explicit cluster ordering for the x-axis.
    palette : str, default="Accent"
        Discrete colormap for the boxes.
    showfliers : bool, default=False
        Whether to show outlier points.
    width : float, default=0.75
        Box width.
    title : str, optional
        Figure title.
    xlabel : str, default="Cluster"
        X-axis label.
    ylabel : str, default="Uncertainty"
        Y-axis label.
    annotation : str, optional
        Text for adaptive annotation.
    rotation : float, optional
        Tick label rotation angle.
    ylim : tuple of float, default=(-0.02, 1.02)
        Y-axis limits. A small margin beyond [0, 1] prevents plot elements
        at the boundary from being hidden behind axis spines.
    fit_ylim : bool, default=True
        Whether to automatically fit y-limits to the data range.
    show : bool, default=False
        Whether to call plt.show() before returning.
    save : str or Path, optional
        Path to save the figure.
    dpi : int, default=300
        Dots per inch for saved figures.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object, or None if ``save`` is provided.
    """
    groups, kept_order = _prepare_cluster_score_groups(scores, labels, order=order)

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

    if fit_ylim:
        all_vals = np.concatenate(groups)
        data_min, data_max = all_vals.min(), all_vals.max()
        margin = max(0.02, 0.05 * (data_max - data_min))
        effective_ylim = (data_min - margin, data_max + margin)
    else:
        effective_ylim = ylim

    if effective_ylim is not None:
        ax.set_ylim(*effective_ylim)

    if title is not None:
        ax.set_title(title)

    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)

    if annotation is not None:
        _place_adaptive_annotation(ax, annotation)

    fig.tight_layout()

    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    if show:
        plt.show()

    return ax


def plot_cluster_violin(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    order: list[int | str] | None = None,
    palette: str = "Accent",
    density_norm: Literal["width", "area", "count"] = "width",
    stripplot: bool = True,
    jitter: bool | float = True,
    size: float = 8.0,
    alpha: float = 0.22,
    inner: Literal["box", "quartile", "none"] = "box",
    title: str | None = None,
    xlabel: str = "Cluster",
    ylabel: str = "Uncertainty",
    annotation: str | None = None,
    rotation: float | None = None,
    ylim: tuple[float, float] = (-0.02, 1.02),
    fit_ylim: bool = True,
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes:
    """Plot per-cluster score distributions as a violin plot.

    The interface mirrors common scanpy options (``stripplot``, ``jitter``,
    ``size``, ``density_norm``, ``show``, ``ax``, ``save``) while using
    matplotlib primitives to avoid an additional plotting dependency.

    Parameters
    ----------
    scores : ndarray of shape (n_samples,)
        Per-sample scores.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (6.5, 4.0).
    order : list of int or str, optional
        Explicit cluster ordering for the x-axis.
    palette : str, default="Accent"
        Discrete colormap for violin colors.
    density_norm : {"width", "area", "count"}, default="width"
        How to normalize violin widths.
    stripplot : bool, default=True
        Whether to overlay individual data points.
    jitter : bool or float, default=True
        Jitter width for the strip plot. True uses 0.11.
    size : float, default=8.0
        Marker size for strip plot points.
    alpha : float, default=0.22
        Marker alpha for strip plot points.
    inner : {"box", "quartile", "none"}, default="box"
        Inner annotation style.
    title : str, optional
        Figure title.
    xlabel : str, default="Cluster"
        X-axis label.
    ylabel : str, default="Uncertainty"
        Y-axis label.
    annotation : str, optional
        Text for adaptive annotation.
    rotation : float, optional
        Tick label rotation angle.
    ylim : tuple of float, default=(-0.02, 1.02)
        Y-axis limits. A small margin beyond [0, 1] prevents plot elements
        at the boundary from being hidden behind axis spines.
    fit_ylim : bool, default=True
        Whether to automatically fit y-limits to the data range.
    show : bool, default=False
        Whether to call plt.show() before returning.
    save : str or Path, optional
        Path to save the figure.
    dpi : int, default=300
        Dots per inch for saved figures.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object, or None if ``save`` is provided.
    """
    groups, kept_order = _prepare_cluster_score_groups(scores, labels, order=order)

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

    if fit_ylim:
        all_vals = np.concatenate(groups)
        data_min, data_max = all_vals.min(), all_vals.max()
        margin = max(0.02, 0.05 * (data_max - data_min))
        effective_ylim = (data_min - margin, data_max + margin)
    else:
        effective_ylim = ylim

    # Clip violin polygon vertices to ylim to prevent KDE tails from
    # visually extending beyond the axis boundaries.
    if effective_ylim is not None:
        ymin, ymax = effective_ylim
        for body in vp["bodies"]:
            for path in body.get_paths():
                path.vertices[:, 1] = np.clip(path.vertices[:, 1], ymin, ymax)

    for pos, vals in enumerate(groups, start=1):
        # --- Inner annotation ---
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

        # --- Strip plot overlay ---
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

    if effective_ylim is not None:
        ax.set_ylim(*effective_ylim)

    if title is not None:
        ax.set_title(title)

    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)

    if annotation is not None:
        _place_adaptive_annotation(ax, annotation)

    fig.tight_layout()

    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    if show:
        plt.show()

    return ax


def plot_cluster_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    embedding: np.ndarray | None = None,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    palette: str = "Accent",
    alpha_range: tuple[float, float] = (0.45, 0.9),
    size_range: tuple[float, float] = (15.0, 60.0),
    sort_order: bool = True,
    legend: bool = True,
    legend_loc: str = "right margin",
    annotation: str | None = None,
    annotation_style: Literal["legend", "box"] = "legend",
    title: str | None = None,
    scores_name: str = "Score",
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    frameon: bool = False,
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes:
    """Scatter plot with per-sample opacity and size score encoding.

    Per-sample scores are mapped to alpha (low scores are more opaque,
    highlighting unstable samples) and to marker size.  Cluster mean
    scores are shown in the legend.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    scores : ndarray of shape (n_samples,)
        Per-sample scores.
    embedding : ndarray of shape (n_samples, 2), optional
        Pre-computed 2D embedding. If None, uses PCA when p > 2.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (6.5, 5.0).
    palette : str, default="Accent"
        Colormap for cluster colors.
    alpha_range : tuple of float, default=(0.45, 0.9)
        ``(alpha_high_score, alpha_low_score)``.  High-score (stable)
        samples get the first value (faint); low-score (unstable)
        samples get the second value (opaque).
    size_range : tuple of float, default=(15.0, 60.0)
        ``(size_high_score, size_low_score)``.  Stable samples get the
        first value (small); unstable samples get the second (large).
    sort_order : bool, default=True
        Whether to sort points by alpha so transparent points are drawn first.
    legend : bool, default=True
        Whether to display a legend.
    legend_loc : str, default="right margin"
        Legend location. "right margin" places the legend outside the axes.
    annotation : str, optional
        Text for an informational annotation box. When provided, the
        annotation is displayed according to ``annotation_style``.
    annotation_style : {"legend", "box"}, default="legend"
        How to display the annotation.  ``"legend"`` appends the text
        to the cluster legend (overlaid on the legend area).
        ``"box"`` places a free-floating annotation at the
        least-obstructed location, identical to the violin/boxplot
        annotation boxes.
    title : str, optional
        Figure title.
    scores_name : str, default="Score"
        Name for the score variable in the plot.
    xlabel : str, default="Component 1"
        X-axis label.
    ylabel : str, default="Component 2"
        Y-axis label.
    frameon : bool, default=False
        Whether to draw axis spines.
    show : bool, default=False
        Whether to call plt.show() before returning.
    save : str or Path, optional
        Path to save the figure.
    dpi : int, default=300
        Dots per inch for saved figures.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object, or None if ``save`` is provided.
    """
    # --- Input validation and preparation ---
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] != labels.shape[0] or X.shape[0] != scores.shape[0]:
        raise ValueError("X, labels, and scores must have matching n_samples.")

    valid = np.isfinite(scores)
    if not np.any(valid):
        raise ValueError("No finite scores available for plotting.")

    # --- Compute 2D coordinates ---
    if embedding is None:
        if X.shape[1] >= 2:
            coords = X[:, :2]
        elif X.shape[1] == 1:
            coords = np.column_stack([X[:, 0], np.zeros(X.shape[0], dtype=float)])
        else:
            raise ValueError("X must have at least 1 feature for scatter plotting.")
        if X.shape[1] > 2:
            coords = PCA(n_components=2, random_state=0).fit_transform(X)
    else:
        coords = np.asarray(embedding, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("embedding must be a 2D array with at least 2 columns.")
        if coords.shape[0] != X.shape[0]:
            raise ValueError("embedding must have the same number of rows as X.")
        coords = coords[:, :2]

    # --- Map scores to marker size (low score = large, high score = small) ---
    lo_score = np.nanmin(scores[valid])
    hi_score = np.nanmax(scores[valid])
    size_hi, size_lo = size_range  # (15.0, 60.0) default
    if np.isclose(lo_score, hi_score):
        size_vals = np.full(scores.shape[0], np.mean(size_range), dtype=float)
    else:
        norm = (scores - lo_score) / (hi_score - lo_score)
        # norm~0 (unstable) → size_lo (large), norm~1 (stable) → size_hi (small)
        size_vals = size_lo + norm * (size_hi - size_lo)

    # --- Compute per-cluster mean scores ---
    uniq = np.unique(labels)
    try:
        uniq = np.array(sorted(uniq, key=lambda x: float(x)))
    except Exception:
        uniq = np.array(sorted(uniq, key=lambda x: str(x)))

    cluster_mean = {}
    for lab in uniq:
        lab_scores = scores[(labels == lab) & valid]
        cluster_mean[lab] = float(np.nanmean(lab_scores)) if lab_scores.size else np.nan

    # --- Map per-sample scores to alpha ---
    alpha_hi, alpha_lo = alpha_range  # (0.45, 0.9) default
    if np.isclose(lo_score, hi_score):
        sample_alpha = np.full(scores.shape[0], np.mean(alpha_range), dtype=float)
    else:
        norm = (scores - lo_score) / (hi_score - lo_score)
        # Low score (norm~0) → alpha_lo (opaque), high score (norm~1) → alpha_hi (faint)
        sample_alpha = alpha_lo + norm * (alpha_hi - alpha_lo)

    # --- Build per-sample RGBA colors ---
    label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    cmap = plt.get_cmap(palette, max(len(uniq), 1))
    rgba = np.zeros((labels.shape[0], 4), dtype=float)
    for i, lab in enumerate(labels):
        base = cmap(label_to_idx[lab])
        rgba[i, :3] = base[:3]
        rgba[i, 3] = np.clip(sample_alpha[i], 0.0, 1.0)

    order = np.arange(labels.shape[0])
    if sort_order:
        order = np.argsort(rgba[:, 3], kind="stable")

    # --- Figure setup ---
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

    # --- Legend ---
    if legend:
        handles = []

        # Add cluster entries with mean score in the label
        for lab in uniq:
            col = cmap(label_to_idx[lab])
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    label=f"{int(lab) + 1} (Mean = {cluster_mean[lab]:.2f})",
                    markerfacecolor=(col[0], col[1], col[2], 0.8),
                    markeredgecolor="none",
                    markersize=7,
                )
            )

        # Place legend according to specified location
        base_title = f"Cluster, {scores_name}" if scores_name else "Cluster"

        # Prepend annotation as legend title text (above base title)
        if annotation is not None and annotation_style == "legend":
            title = f"{annotation}\n{base_title}"
        else:
            title = base_title

        legend_kwargs = dict(handles=handles, title=title, frameon=False)
        if annotation is not None and annotation_style == "legend":
            legend_kwargs["title_fontproperties"] = {"size": 9}

        if legend_loc == "right margin":
            ax.legend(
                **legend_kwargs,
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
                borderaxespad=0.0,
            )
        else:
            ax.legend(**legend_kwargs, loc=legend_loc)

    # Annotation as floating box (when not embedded in legend)
    if annotation is not None and annotation_style == "box":
        _place_adaptive_annotation(ax, annotation, replace_legend=not legend)

    fig.tight_layout()

    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    if show:
        plt.show()

    return ax


def plot_diagnostic_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    embedding: np.ndarray | None = None,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    cmap: str = "Greens_r",
    alpha_encoding: bool = True,
    alpha_range: tuple[float, float] = (0.3, 1.0),
    marker_size: float = 30.0,
    markers: list[str] | None = None,
    sort_order: bool = True,
    legend: bool = True,
    legend_loc: str = "right margin",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    annotation: str | None = None,
    annotation_style: Literal["legend", "box"] = "legend",
    title: str | None = None,
    scores_name: str = "Score",
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    frameon: bool = False,
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes:
    """Diagnostic scatter plot with shape-per-cluster and color-per-score encoding.

    Cluster membership is encoded via marker shapes, while per-sample scores
    are mapped to a sequential colormap.  Unstable / ungeneralizable samples
    (low scores) appear in saturated, opaque colors; stable samples (high
    scores) fade toward a neutral tone.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    scores : ndarray of shape (n_samples,)
        Per-sample scores (e.g., stability or generalizability).
    embedding : ndarray of shape (n_samples, 2), optional
        Pre-computed 2D embedding. If None, uses PCA when p > 2.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (6.5, 5.0).
    cmap : str, default="Greens_r"
        Sequential matplotlib colormap. Low scores map to the dark end,
        high scores to the light end.
    alpha_encoding : bool, default=True
        Whether to also vary point transparency with score (low score =
        opaque, high score = transparent), reinforcing the color encoding.
    alpha_range : tuple of float, default=(0.3, 1.0)
        ``(alpha_high_score, alpha_low_score)``.  Only used when
        ``alpha_encoding=True``.
    marker_size : float, default=30.0
        Fixed marker size for all points.
    markers : list of str, optional
        Marker codes for each cluster. Defaults to 14 distinct filled
        markers. Cycles with a warning if clusters exceed the list length.
    sort_order : bool, default=True
        Draw stable (high-score) points first so unstable points render
        on top.
    legend : bool, default=True
        Whether to display a shape legend for clusters.
    legend_loc : str, default="right margin"
        Legend location. "right margin" places the legend outside the axes.
    colorbar : bool, default=True
        Whether to draw a colorbar for the score encoding.
    colorbar_label : str, optional
        Label for the colorbar. Defaults to ``scores_name``.
    annotation : str, optional
        Text for an informational annotation.
    annotation_style : {"legend", "box"}, default="legend"
        How to display the annotation.
    title : str, optional
        Figure title.
    scores_name : str, default="Score"
        Name for the score variable (used as default colorbar label).
    xlabel : str, default="Component 1"
        X-axis label.
    ylabel : str, default="Component 2"
        Y-axis label.
    frameon : bool, default=False
        Whether to draw axis spines.
    show : bool, default=False
        Whether to call plt.show() before returning.
    save : str or Path, optional
        Path to save the figure.
    dpi : int, default=300
        Dots per inch for saved figures.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes object, or None if ``save`` is provided.
    """
    # --- Input validation and preparation ---
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] != labels.shape[0] or X.shape[0] != scores.shape[0]:
        raise ValueError("X, labels, and scores must have matching n_samples.")

    valid = np.isfinite(scores)
    if not np.any(valid):
        raise ValueError("No finite scores available for plotting.")

    # --- Compute 2D coordinates ---
    if embedding is None:
        if X.shape[1] >= 2:
            coords = X[:, :2]
        elif X.shape[1] == 1:
            coords = np.column_stack([X[:, 0], np.zeros(X.shape[0], dtype=float)])
        else:
            raise ValueError("X must have at least 1 feature for scatter plotting.")
        if X.shape[1] > 2:
            coords = PCA(n_components=2, random_state=0).fit_transform(X)
    else:
        coords = np.asarray(embedding, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("embedding must be a 2D array with at least 2 columns.")
        if coords.shape[0] != X.shape[0]:
            raise ValueError("embedding must have the same number of rows as X.")
        coords = coords[:, :2]

    # --- Resolve marker list ---
    if markers is None:
        markers = _DEFAULT_DIAGNOSTIC_MARKERS

    uniq = np.unique(labels)
    try:
        uniq = np.array(sorted(uniq, key=lambda x: float(x)))
    except Exception:
        uniq = np.array(sorted(uniq, key=lambda x: str(x)))

    if len(uniq) > len(
        markers
    ):  # Warn if more clusters than markers, but still proceed with cycling
        warnings.warn(
            f"Number of clusters ({len(uniq)}) exceeds available markers "
            f"({len(markers)}). Markers will cycle.",
            stacklevel=2,
        )

    label_to_marker = {lab: markers[i % len(markers)] for i, lab in enumerate(uniq)}

    # --- Normalize scores and map to colormap ---
    cmap_obj = plt.get_cmap(cmap)

    lo_score = np.nanmin(scores[valid])
    hi_score = np.nanmax(scores[valid])

    norm_scores = np.full(scores.shape[0], 0.5, dtype=float)

    if not np.isclose(lo_score, hi_score):
        norm_scores[valid] = (scores[valid] - lo_score) / (hi_score - lo_score)

    rgba = cmap_obj(norm_scores)

    # --- Optional alpha reinforcement ---
    if alpha_encoding:
        alpha_hi, alpha_lo = alpha_range
        if np.isclose(alpha_hi, alpha_lo):
            rgba[:, 3] = np.mean(alpha_range)
        else:
            rgba[valid, 3] = alpha_lo + norm_scores[valid] * (alpha_hi - alpha_lo)

    # Non-finite scores: neutral gray, low alpha
    nan_mask = ~valid
    if np.any(nan_mask):
        rgba[nan_mask] = (0.5, 0.5, 0.5, 0.2)

    # --- Figure setup ---
    if ax is None:
        if figsize is None:
            figsize = (6.5, 5.0)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- Draw points (per-cluster for distinct markers) ---
    # Order clusters: draw stable (high mean) first, unstable (low mean) last (on top)
    cluster_means = {
        lab: float(np.nanmean(scores[(labels == lab) & valid]))
        if np.any((labels == lab) & valid)
        else 0.0
        for lab in uniq
    }

    cluster_draw_order = sorted(uniq, key=lambda lab: -cluster_means[lab])

    for lab in cluster_draw_order:
        mask = labels == lab
        idx = np.where(mask)[0]

        # Within cluster: draw high-score first, low-score on top
        if sort_order:
            within_order = np.argsort(-norm_scores[idx], kind="stable")
            idx = idx[within_order]

        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=marker_size,
            c=rgba[idx],
            marker=label_to_marker[lab],
            linewidths=0.3,
            edgecolor="black",
            rasterized=(coords.shape[0] > 5000),
        )

    # --- Colorbar ---
    if colorbar:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sm = ScalarMappable(cmap=cmap_obj, norm=Normalize(vmin=lo_score, vmax=hi_score))
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            fraction=0.046,
            pad=0.09,
        )
        cbar.set_label(colorbar_label if colorbar_label is not None else scores_name)

    # --- Shape legend ---
    if legend:
        handles = []
        for lab in uniq:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=label_to_marker[lab],
                    linestyle="",
                    label=str(int(lab) + 1),
                    markerfacecolor="gray",
                    markeredgecolor="black",
                    markersize=7,
                    markeredgewidth=0.3,
                )
            )

        base_title = "Cluster"

        if annotation is not None and annotation_style == "legend":
            legend_title = f"{annotation}\n{base_title}"
        else:
            legend_title = base_title

        legend_kwargs = dict(handles=handles, title=legend_title, frameon=False)
        if annotation is not None and annotation_style == "legend":
            legend_kwargs["title_fontproperties"] = {"size": 9}

        if legend_loc == "right margin":
            ax.legend(
                **legend_kwargs,
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
                borderaxespad=0.0,
            )
        else:
            ax.legend(**legend_kwargs, loc=legend_loc)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if not frameon:
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax.grid(False)

    # Annotation as floating box (when not embedded in legend)
    if annotation is not None and annotation_style == "box":
        _place_adaptive_annotation(ax, annotation, replace_legend=not legend)

    fig.tight_layout()

    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    if show:
        plt.show()

    return ax
