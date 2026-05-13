"""Plotting helpers for the benchmarking notebooks.

Public entry points (in order of appearance):
- ``plot_examples``                — example PCA scatter row across anchors
- ``plot_benchmark_snapshot``      — live PCA + summary panels (per-iteration)
- ``display_benchmark_snapshot``   — tqdm-safe in-place display wrapper
- ``plot_ari_over_difficulty``     — ARI vs. difficulty / scaling axis line plot
- ``plot_ari_overview_grid``       — 2×3 grid of ARI lineplots (one per regime)
- ``plot_paper_figure``            — combined Section 2 paper figure (Fig 3)
- ``plot_runtime_over_axis``       — CARVE runtime vs. scaling axis
- ``filter_true_k``                — slice a results frame to a single ``true_k``
- ``draw_grouped_legend``          — Baseline / CARVE / Classical legend helper

Scaling-experiment figures are composed at the call site (in
``Benchmarking.ipynb``) by looping over the per-panel helpers above so the
grid shape can adapt to which dataset families are included.
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import cm

from sklearn.cluster import AgglomerativeClustering, KMeans
from carve.cluster import SpectralClusteringCARVE
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from carve.sim import simulate_clusters

from benchmarking_config import SCALING_RANGES, METRIC_DISPLAY_NAMES, SCALING_CONSTANTS
from benchmarking_utils import _wilson_ci


# =============================================================================
# Palettes
# =============================================================================
cmap = cm.get_cmap("Dark2")
cmap_length = cmap.N

cluster_pallette = [
    color for i, color in enumerate(cmap.colors) if i != 99
]  # Exclude yellow as nth color if needed

# cluster_pallette = [
#     "#009ADE",
#     "#00CD6C",
#     "#FF1F5B",
#     "#AF58BA",
#     "#F28522",
#     "#A6761D",
#     "#A0B1BA",
# ]

lines_pallette_contrastive_carve = [
    "#00CD6C",
    "#009ADE",
]

lines_pallette_contrastive_other = [
    "#E0457B",
    "#A8389E",
    "#D6292E",
    "#F28522",
]


# =============================================================================
# Color mapping helpers
# =============================================================================
def _get_color_mapping(k: int) -> List[Any]:
    """okabe-ito for k<=7, tab20 for 8..20, hsv fallback."""
    if k <= cmap_length:
        cols = [mpl.colors.to_rgba(cluster_pallette[i]) for i in range(k)]
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


def _constrastive_color_map(metric_names: Iterable[str]) -> dict[str, Any]:
    """Build a CARVE-vs-classical contrastive palette keyed by metric name."""
    names = list(metric_names)

    internal_metrics = {"silhouette", "gap", "davies_bouldin", "calinski_harabasz"}

    carve_metrics = [n for n in names if n not in internal_metrics]
    other_metrics = [n for n in names if n in internal_metrics]

    cols_dict = {}

    if carve_metrics:
        cols = [
            mpl.colors.to_rgba(lines_pallette_contrastive_carve[i])
            for i in range(len(carve_metrics))
        ]
        for i, name in enumerate(carve_metrics):
            cols_dict[name] = cols[i]

    if other_metrics:
        cols = [
            mpl.colors.to_rgba(lines_pallette_contrastive_other[i])
            for i in range(len(other_metrics))
        ]
        for i, name in enumerate(other_metrics):
            cols_dict[name] = cols[i]

    return cols_dict


# =============================================================================
# Axis, label and DataFrame helpers
# =============================================================================
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


def _pretty_metric_name(metric: str) -> str:
    """Human-readable label for a metric (falls back to raw name)."""
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def filter_true_k(df: pd.DataFrame, k_star: int) -> pd.DataFrame:
    """Return rows of ``df`` with ``true_k == k_star`` (or all rows if absent)."""
    if "true_k" not in df.columns:
        return df
    return df.loc[df["true_k"] == k_star].copy()


# =============================================================================
# Drawing primitives — scatter and legend
# =============================================================================
def _scatter_clusters(
    ax, Z: np.ndarray, labels: np.ndarray, title: str, subtitle: str = ""
):
    """Render a single 2-D scatter coloured by ``labels`` onto ``ax``."""
    labels = np.asarray(labels)
    cmap = _cluster_color_map(labels)

    # stable ordering for legend (optional)
    uniq = sorted(int(x) for x in np.unique(labels) if x != -1)

    for cid in uniq:
        m = labels == cid
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            s=12,
            alpha=0.90,
            c=[cmap[cid]],
            edgecolor="k",
            linewidth=0.2,
        )

    ax.set_title(title, fontsize=10)
    if subtitle:
        ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=9)

    # equal aspect
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.2)


def _split_legend_groups(handles, labels) -> list[tuple[list, list, int]]:
    """Split legend handles into (Baseline, CARVE, Classical) groups.

    Mirrors the grouping rule used inside ``plot_paper_figure`` so the
    section-3 grids advertise their methods identically.
    """
    baseline_h, baseline_l = [], []
    carve_h, carve_l = [], []
    classical_h, classical_l = [], []
    for h, lab in zip(handles, labels):
        low = lab.lower()
        if "baseline" in low or "oracle" in low:
            baseline_h.append(h)
            baseline_l.append(lab)
        elif "carve" in low:
            carve_h.append(h)
            carve_l.append(lab)
        else:
            classical_h.append(h)
            classical_l.append(lab)
    groups: list[tuple[list, list, int]] = []
    if baseline_h:
        groups.append((baseline_h, baseline_l, 1))
    if carve_h:
        groups.append((carve_h, carve_l, 1))
    if classical_h:
        groups.append((classical_h, classical_l, 2))
    return groups


def _draw_lineplot_series(
    ax,
    x,
    mid,
    se,
    *,
    color,
    label: str,
    linewidth: float = 2.6,
):
    """Antenna-style errorbar series shared by ARI and runtime line plots.

    Caps + marker + line widths match the section-2/3 visual identity so the
    runtime panels stay legible alongside the ARI panels.
    """
    ax.errorbar(
        x,
        mid,
        yerr=se,
        fmt="-o",
        color=color,
        linewidth=linewidth,
        markersize=5,
        capsize=3,
        capthick=1.2,
        elinewidth=1.2,
        label=label,
    )


def draw_grouped_legend(
    fig,
    axes_grid,
    *,
    fontsize: float = 12,
    legend_y_offset: float = 0.04,
):
    """Add a grouped Baseline / CARVE / Classical legend below ``axes_grid``."""
    handles, labels = axes_grid[0, 0].get_legend_handles_labels()
    groups = _split_legend_groups(handles, labels)
    if not groups:
        return

    # Compute the combined extent of the axes grid so the legend stays
    # centred under the panels regardless of figsize.
    x0 = min(ax.get_position().x0 for ax in axes_grid.ravel())
    x1 = max(ax.get_position().x1 for ax in axes_grid.ravel())
    y_min = min(ax.get_position().y0 for ax in axes_grid.ravel())
    grid_w = x1 - x0
    legend_y = y_min - legend_y_offset

    legend_kw = dict(
        frameon=False,
        fontsize=fontsize,
        handlelength=2.2,
        borderpad=0.3,
        labelspacing=0.30,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    n_groups = len(groups)
    for gi, (gh, gl, gncol) in enumerate(groups):
        frac = (gi + 0.5) / n_groups
        gx = x0 + frac * grid_w
        fig.legend(
            gh,
            gl,
            loc="upper center",
            bbox_to_anchor=(gx, legend_y),
            ncol=gncol,
            **legend_kw,
        )


# =============================================================================
# Seed selection helper (paper figures)
# =============================================================================
def _select_visually_full_seed(
    results_list: list[tuple[float, np.ndarray, np.ndarray]],
) -> int:
    """Pick the seed whose 2-D PCA spread is most *square* (least whitespace).

    For each seed we project X to 2-D PCA and compute the ratio
    ``min(var) / max(var)`` of the two explained-variance components.
    A ratio close to 1 means data fills both axes evenly → less whitespace.
    """
    best_idx, best_ratio = 0, -1.0
    for idx, (_, X, _) in enumerate(results_list):
        pca = PCA(n_components=2, random_state=0)
        pca.fit(X)
        ev = pca.explained_variance_ratio_[:2]
        ratio = ev.min() / max(ev.max(), 1e-12)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx
    return best_idx


# =============================================================================
# Per-iteration simulation helper used by example/paper plots
# =============================================================================
def _plotting_iter(
    anchor_settings: Dict,
    other_settings: Dict,
    j: int,
    level: Any,
    level_label: str,
    true_k: int,
    estimator_type: str,
    sampler: Literal["default", "scaling"],
    seed: int,
    random_state: int,
):
    """Simulate one (anchor, seed) dataset, fit the requested estimator, return (ari, X, y)."""
    benchmark_seed = seed + (j * 10000) + random_state

    if sampler == "default":
        X, y = simulate_clusters(
            k=true_k,
            plotting=False,
            random_state=benchmark_seed,
            **anchor_settings[level],
            **other_settings,
        )
    elif sampler == "scaling":
        if level_label not in {"n_total", "p", "embed_dim"}:
            raise ValueError("level_label must be 'n_total', 'p', or 'embed_dim'")

        n_total = SCALING_CONSTANTS["n_total"]
        p = SCALING_CONSTANTS["p"]
        embed_dim = SCALING_CONSTANTS["embed_dim"]

        if level_label == "n_total":
            n_total = int(level)
        elif level_label == "p":
            p = int(level)
        elif level_label == "embed_dim":
            embed_dim = int(level)
        else:
            raise ValueError("level_label must be 'n_total', 'p', or 'embed_dim'")

        dict_key = {0: "start", 1: "middle", 2: "end"}[j]

        filtered_other = {
            k: v
            for k, v in other_settings.items()
            if k not in ("n_total", "p", "embed_dim")
        }

        X, y = simulate_clusters(
            n_total=n_total,
            p=p,
            embed_dim=embed_dim,
            k=true_k,
            plotting=False,
            random_state=benchmark_seed,
            **anchor_settings[dict_key],
            **filtered_other,
        )

    if estimator_type == "agglomerative":
        estimator = AgglomerativeClustering(n_clusters=true_k)
    elif estimator_type == "spectral":
        estimator = SpectralClusteringCARVE(
            n_clusters=true_k, affinity="self_tuning", random_state=benchmark_seed
        )
    else:
        estimator = KMeans(n_clusters=true_k, n_init=10, random_state=benchmark_seed)

    y_hat = estimator.fit_predict(X)
    ari = adjusted_rand_score(y, y_hat)

    return ari, X, y


# =============================================================================
# Public — example data plots
# =============================================================================
def plot_examples(
    anchor_settings: Dict,
    other_settings: Dict,
    true_cluster_count: int = 5,
    level_label: str = "difficulty",
    levels: List[Any] = ["easy", "medium", "hard"],
    n_seeds_per_dataset: int = 20,
    estimator_type: str = "kmeans",
    example_title: str = "Gaussian Mixtures",
    sampler: Literal["default", "scaling"] = "default",
    figsize: tuple = (12, 4),
    n_jobs: int = 1,
    random_state: int = 0,
) -> None:
    """
    Visualizes example clustering datasets for the three anchors at the given
    true cluster count, and computes baseline clustering performance (ARI).

    Args:
        - anchor_settings (Dict): Simulation parameters keyed by anchor name
          (``easy``/``medium``/``hard`` for difficulty, ``start``/``middle``/``end``
          for scaling).
        - other_settings (Dict): Shared/global simulation parameters.
        - true_cluster_count (int): True cluster count to visualize.
        - n_seeds_per_dataset (int): Number of replicates for baseline ARI estimation.
        - estimator_type (str): Clustering algorithm type ('kmeans', 'agglomerative' or 'spectral').
        - example_title (str): Title for the figure.
        - random_state (int): Seed for reproducibility.
        - figsize (tuple): Figure size for the matplotlib figure.

    Returns:
        None. Displays the matplotlib figure.
    """
    if level_label == "difficulty":
        levels = ["easy", "medium", "hard"]
    elif level_label in SCALING_RANGES:
        r = SCALING_RANGES[level_label]
        levels = [int(x) for x in np.linspace(r["min"], r["max"], num=3)]
    else:
        raise ValueError(
            f"level_label must be 'difficulty' or one of {sorted(SCALING_RANGES)}, "
            f"got {level_label!r}"
        )

    _, axes = plt.subplots(1, len(levels), figsize=figsize)

    for j, level in enumerate(levels):
        worker = delayed(_plotting_iter)
        results = Parallel(n_jobs=n_jobs)(
            worker(
                anchor_settings=anchor_settings,
                other_settings=other_settings,
                j=j,
                level=level,
                level_label=level_label,
                true_k=true_cluster_count,
                estimator_type=estimator_type,
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

        ax = axes[j]
        ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=colors,
            s=20,
            alpha=0.8,
            edgecolor="k",
            linewidth=0.2,
        )

        # titles, legends, &c.
        if level_label == "difficulty":
            ax.set_title(
                f"k={true_cluster_count}, 'S/N Ratio': {level} | Baseline ARI={ari_mean:.3f}"
            )
        else:
            ax.set_title(
                f"k={true_cluster_count}, {level_label}={level} | Baseline ARI={ari_mean:.3f}"
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.1)

    plt.suptitle(f"Example datasets: {example_title}", fontsize=16)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Public — live benchmark snapshot
# =============================================================================
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
        "ari_stability_1se",
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
    _scatter_clusters(axs[0, 0], Z, y, title="True Labels", subtitle=f"k^*={k_true}")

    _scatter_clusters(
        axs[0, 1], Z, base_lab, title="Baseline", subtitle=f"ARI={baseline_ari:.3f}"
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
            ax,
            Z,
            labs,
            title=_pretty_metric_name(key),
            subtitle=f"k^hat={k_sel}  |  ARI={ari_sel:.3f}",
        )

    # --- Summary figure: accuracy+wilson + ari boxplots
    needed = {"metric_name", "is_optimal", "is_correct", "metric_ari", "baseline_ari"}
    missing = sorted(c for c in needed if c not in results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {missing}")

    df_opt = results_df.loc[results_df["is_optimal"]].copy()

    # Order: CARVE metrics > externals
    accuracy_metrics = [
        "ari_stability_1se",
        "ari_generalizability_1se",
        "silhouette",
        "gap",
        "davies_bouldin",
        "calinski_harabasz",
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
    dedupe_cols = [
        c
        for c in ["difficulty_level", "dataset_iteration", "true_k"]
        if c in results_df.columns
    ]
    if dedupe_cols:
        base_ari = (
            results_df.drop_duplicates(dedupe_cols)["baseline_ari"].astype(float).values
        )
    else:
        base_ari = results_df["baseline_ari"].dropna().astype(float).unique()

    # ARI-by-metric distribution (optimal only)
    ari_data = [
        df_opt.loc[df_opt["metric_name"] == m, "metric_ari"].astype(float).values
        for m in accuracy_metrics
    ]
    ari_labels = [_pretty_metric_name(m) for m in accuracy_metrics]

    # Include baseline as first box
    ari_data = [base_ari] + ari_data
    ari_labels = ["Baseline"] + ari_labels

    fig_sum, ax = plt.subplots(1, 2, figsize=figsize_summary, constrained_layout=True)

    # Left: accuracy points + wilson intervals
    x = np.arange(len(acc_df))
    ax0 = ax[0]
    ax0.vlines(x, acc_df["lo"], acc_df["hi"], linewidth=2)
    ax0.scatter(x, acc_df["acc"], s=35)
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_xticks(x)
    ax0.set_xticklabels(
        [_pretty_metric_name(m) for m in acc_df["metric"]], rotation=35, ha="right"
    )
    ax0.set_ylabel("p(k^hat = k*)")
    ax0.set_title("k* Recovery (Wilson 95% CI)")

    # Right: boxplot of ari for selected solutions
    ax1 = ax[1]
    ax1.boxplot(ari_data, showfliers=False)
    ax1.set_xticks(np.arange(1, len(ari_labels) + 1))
    ax1.set_xticklabels(ari_labels, rotation=35, ha="right")
    ax1.set_ylabel("ARI(y, y^hat)")
    ax1.set_title("ARI : True Labels (for k^hat)")

    return fig_pca, fig_sum


def display_benchmark_snapshot(
    snapshot_handles: Dict[str, Any],
    *,
    X: np.ndarray,
    results_df: pd.DataFrame,
    plotting_dict: Mapping[str, Mapping[str, Any]],
    true_labels: np.ndarray,
    baseline_labels: np.ndarray,
    baseline_ari: float,
) -> None:
    """Render the live benchmark snapshot in-place (tqdm-safe).

    Builds the PCA + summary figures via ``plot_benchmark_snapshot`` and
    either initializes IPython display handles (first call) or updates
    them in place (subsequent calls). Avoiding ``clear_output()`` keeps
    the tqdm progress bar visible and stationary throughout the loop.

    Parameters
    ----------
    snapshot_handles : dict
        Mutable dict with keys ``"pca"`` and ``"summary"``. Initialise both
        to ``None`` before the loop; this function fills them on the first
        call and reuses them thereafter.
    """
    from IPython.display import display

    fig_pca, fig_sum = plot_benchmark_snapshot(
        X=X,
        results_df=results_df,
        plotting_dict=plotting_dict,
        true_labels=true_labels,
        baseline_labels=baseline_labels,
        baseline_ari=baseline_ari,
    )
    if snapshot_handles.get("pca") is None:
        snapshot_handles["pca"] = display(fig_pca, display_id=True)
        snapshot_handles["summary"] = display(fig_sum, display_id=True)
    else:
        snapshot_handles["pca"].update(fig_pca)
        snapshot_handles["summary"].update(fig_sum)
    plt.close(fig_pca)
    plt.close(fig_sum)


# =============================================================================
# Public — ARI-over-difficulty line plots
# =============================================================================
# Fixed left-to-right slot order at each x-anchor. The ``None`` slot is a
# blank gap that visually separates CARVE selection rules from heuristic
# baselines. Anything in ``metrics`` that is not in this tuple is appended
# to the right of the cluster.
LINEPLOT_DISPLAY_ORDER: tuple[str | None, ...] = (
    "ari_stability_1se",
    "ari_generalizability_1se",
    None,
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "gap",
)


def plot_ari_over_difficulty(
    results_df: pd.DataFrame,
    *,
    metrics=(
        "ari_stability_1se",
        "ari_generalizability_1se",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "gap",
    ),
    center: str = "mean",
    dodge: float = 0.005,
    x_col: str | None = None,
    x_label: str = "Signal/Noise Ratio",
    x_label_font_size: int = 12,
    xscale: str = "linear",
    title: str | None = None,
    figsize: tuple = (12, 10),
    ax=None,
    ylim: tuple | str = "auto",
    show_legend: bool = True,
    legend_ncol: int = 1,
    band: tuple | None = None,
    show_band_for=None,
):
    """
    Plot ARI vs. difficulty (or scaling axis) with mean ± 1 SE error bars.

    For difficulty-axis data, x-ticks are categorical ("Easy", "Medium", "Hard")
    and metric markers are dodged horizontally in the fixed order defined by
    ``LINEPLOT_DISPLAY_ORDER`` so antennas don't overlap.

    Args:
        - results_df (pd.DataFrame): DataFrame with benchmarking results.
        - metrics (tuple): Metrics to plot.
        - center (str): Central tendency measure ('mean' or 'median').
        - dodge (float): Per-slot horizontal offset (in x-axis units).
        - title (str | None): Plot title.
        - ax: Matplotlib axis.
        - ylim (tuple | str): Y-axis limits.
        - legend_ncol (int): Number of columns in the legend.
        - band, show_band_for: deprecated, ignored. Retained so existing
          callers in notebooks/grids don't break.
    """
    if x_col is None:
        axis_col, axis_name_col = _infer_axis_cols(results_df)
    else:
        axis_col, axis_name_col = x_col, ""
    needed = {axis_col, "metric_name", "is_optimal", "metric_ari", "baseline_ari"}
    missing = sorted(c for c in needed if c not in results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {missing}")

    # Detect difficulty-axis mode so we can label x-ticks with anchor names.
    is_difficulty = (
        axis_name_col != ""
        and axis_name_col in results_df.columns
        and (results_df[axis_name_col] == "difficulty_level").any()
    ) or axis_col == "difficulty_level"

    present = [m for m in metrics if (results_df["metric_name"] == m).any()]
    if len(present) == 0:
        raise ValueError("none of the requested metrics were found in results_df")

    # --- 1) Filter to one row per (dataset instance, metric): selected "optimal" k ---
    df_opt = results_df.loc[results_df["is_optimal"] == True].copy()

    # --- 2) Baseline: dedupe per dataset instance ---
    dedupe_cols = [
        c for c in [axis_col, "dataset_iteration", "true_k"] if c in results_df.columns
    ]
    base = results_df.drop_duplicates(dedupe_cols)[[axis_col, "baseline_ari"]].copy()

    # Mean ± 1 SE across replicates at each axis value.
    def _summarize(y: pd.Series):
        y = y.astype(float).dropna().values
        if y.size == 0:
            return np.nan, np.nan
        mid = np.mean(y) if center == "mean" else np.median(y)
        se = np.std(y, ddof=1) / np.sqrt(y.size) if y.size > 1 else 0.0
        return mid, se

    # --- 3) Build summary tables ---
    base_sum = (
        base.groupby(axis_col)["baseline_ari"]
        .apply(lambda s: pd.Series(_summarize(s), index=["mid", "se"]))
        .unstack()
        .reset_index()
        .sort_values(axis_col)
    )

    sums = []
    for m in metrics:
        sub = df_opt.loc[df_opt["metric_name"] == m]
        s = (
            sub.groupby(axis_col)["metric_ari"]
            .apply(lambda s: pd.Series(_summarize(s), index=["mid", "se"]))
            .unstack()
            .reset_index()
            .sort_values(axis_col)
        )
        s["metric_name"] = m
        sums.append(s)

    sum_df = pd.concat(sums, ignore_index=True) if sums else pd.DataFrame()

    # --- 4) Plot ---
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    # Anchor positions: contiguous integers for difficulty mode, raw values otherwise.
    raw_levels = np.sort(base_sum[axis_col].dropna().unique())
    if is_difficulty:
        anchor_x = np.arange(len(raw_levels), dtype=float)
        level_to_x = {lvl: float(i) for i, lvl in enumerate(raw_levels)}
    else:
        anchor_x = raw_levels.astype(float)
        level_to_x = {lvl: float(lvl) for lvl in raw_levels}

    # Per-slot x-offset following LINEPLOT_DISPLAY_ORDER (gap slot included).
    n_slots = len(LINEPLOT_DISPLAY_ORDER)
    center_offset = (n_slots - 1) / 2
    slot_dx: dict[str, float] = {
        m: (i - center_offset) * dodge
        for i, m in enumerate(LINEPLOT_DISPLAY_ORDER)
        if m is not None
    }
    extras = [m for m in metrics if m not in slot_dx]
    for j, m in enumerate(extras):
        slot_dx[m] = (n_slots + j - center_offset) * dodge

    # Apply dodge only when the categorical layout is in effect; for numeric
    # scaling axes the absolute spacing is data-driven and a fixed unit-dodge
    # would distort the picture.
    effective_dx = slot_dx if is_difficulty else {k: 0.0 for k in slot_dx}

    # Oracle baseline: dashed gray reference, no whiskers.
    baseline_color = (0.35, 0.35, 0.35, 1.0)
    base_x_vals = np.array(
        [level_to_x[v] for v in base_sum[axis_col].values], dtype=float
    )
    ax.plot(
        base_x_vals,
        base_sum["mid"].values,
        color=baseline_color,
        linestyle="--",
        linewidth=2.6,
        label="Baseline (Oracle k*)",
    )

    # Iterate metrics in display order so legend matches left-to-right layout.
    ordered_metrics = [
        m
        for m in LINEPLOT_DISPLAY_ORDER
        if m is not None and m in metrics and (sum_df["metric_name"] == m).any()
    ]
    ordered_metrics += [m for m in extras if (sum_df["metric_name"] == m).any()]

    color_map = _constrastive_color_map(ordered_metrics)

    for m in ordered_metrics:
        s = sum_df.loc[sum_df["metric_name"] == m]
        if s.empty:
            continue
        xx = (
            np.array([level_to_x[v] for v in s[axis_col].values], dtype=float)
            + effective_dx[m]
        )
        _draw_lineplot_series(
            ax,
            xx,
            s["mid"].values,
            s["se"].values,
            color=color_map[m],
            label=_pretty_metric_name(m),
            linewidth=2.6 if "ari_" in m else 2.0,
        )

    ax.set_xlabel(x_label, fontsize=x_label_font_size, labelpad=8)
    ax.set_ylabel(r"ARI (selected $\hat{k}$ vs. true labels)")
    ax.set_xticks(anchor_x)

    if is_difficulty:
        anchor_labels = {0: "Easy", 1: "Medium", 2: "Hard"}
        ax.set_xticklabels([anchor_labels.get(int(v), str(v)) for v in raw_levels])
        # Hug the leftmost and rightmost dodged markers with a small margin.
        max_abs_dx = max((abs(v) for v in effective_dx.values()), default=0.0)
        margin = max_abs_dx + 0.04
        ax.set_xlim(anchor_x.min() - margin, anchor_x.max() + margin)
    else:
        if xscale in {"log", "linear"}:
            ax.set_xscale(xscale)
        ax.set_xlim(anchor_x.min(), anchor_x.max())

    if ylim != "auto":
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(auto=True)

    if title is not None:
        ax.set_title(title)

    if show_legend:
        ax.legend(frameon=False, fontsize=9, ncol=legend_ncol)

    return fig


def plot_ari_overview_grid(
    regimes: list[tuple[str, pd.DataFrame]],
    *,
    metrics=(
        "ari_generalizability_1se",
        "ari_stability_1se",
        "silhouette",
        "gap",
        "davies_bouldin",
        "calinski_harabasz",
    ),
    center: str = "mean",
    band: tuple = (0.05, 0.95),
    show_band_for=(),
    x_label: str = "Signal/Noise Ratio",
    figsize: tuple = (16, 18),
    ylim: tuple | str = "auto",
    legend_fontsize: float = 13,
    title_fontsize: float = 14,
    suptitle: str | None = None,
):
    """
    2×3 overview grid of ARI-over-difficulty plots, one panel per benchmarking
    regime, with a single shared legend at the bottom.

    Args:
        regimes: list of (panel_title, results_df) pairs (length ≤ 6).
        metrics, center, band, show_band_for, x_label, ylim: forwarded to
            ``plot_ari_over_difficulty``.
        figsize: figure size.
        legend_fontsize: font size for the shared legend.
        title_fontsize: font size for each panel title.
        suptitle: optional overall figure title.
    """
    n = len(regimes)
    nrows, ncols = 2, 3
    if n > nrows * ncols:
        raise ValueError(f"At most {nrows * ncols} regimes are supported, got {n}")

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        constrained_layout=False,
    )
    axes_flat = axes.ravel()

    for idx, (panel_title, df) in enumerate(regimes):
        ax = axes_flat[idx]
        plot_ari_over_difficulty(
            df,
            metrics=metrics,
            center=center,
            band=band,
            show_band_for=show_band_for,
            x_label=x_label,
            title=panel_title,
            ax=ax,
            ylim=ylim,
            show_legend=False,
        )
        ax.set_title(panel_title, fontsize=title_fontsize, fontweight="bold")

        # Only keep y-axis label on the leftmost column of each row
        row, col = divmod(idx, ncols)
        if col == 0:
            ax.set_ylabel(r"ARI (selected $\hat{k}$ vs. true labels)", fontsize=11)
        else:
            ax.set_ylabel("")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes_flat[idx].set_visible(False)

    # Shared legend from first panel (all panels have the same lines)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(len(handles), 4),
        frameon=False,
        fontsize=legend_fontsize,
    )

    fig.subplots_adjust(
        hspace=0.18,
        wspace=0.15,
        bottom=0.08,
        top=0.95 if suptitle is None else 0.92,
        left=0.07,
        right=0.97,
    )

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=title_fontsize + 2, fontweight="bold")

    return fig


# =============================================================================
# Public — Paper figure 3 (combined per-k* overview)
# =============================================================================
def plot_paper_figure(
    regimes: list[tuple],
    k_star: int = 5,
    *,
    difficulty: str = "medium",
    metrics: tuple[str, ...] = (
        "ari_generalizability_1se",
        "ari_stability_1se",
        "silhouette",
        "gap",
        "davies_bouldin",
        "calinski_harabasz",
    ),
    n_seeds_per_dataset: int = 20,
    center: str = "median",
    band: tuple[float, float] = (0.05, 0.95),
    show_band_for: tuple[str, ...] = (),
    n_jobs: int = 1,
    random_state: int = 0,
    figsize: tuple[float, float] | None = None,
    scale: float = 1.0,
    lineplot_height_ratio: float = 0.75,
    scatter_size: float = 14,
    title_fontsize: float = 15,
    legend_fontsize: float = 15,
    tick_fontsize: float = 10,
    select_full_seed: bool = True,
    panels: str = "both",
) -> mpl.figure.Figure:
    """Create a publication figure for a given k*.

    Section **A** (top, 2×3): example scatter plots per regime (PCA projection).
    Section **B** (bottom, 2×3): ARI-over-difficulty line plots, filtered to
    ``true_k == k_star``.  Lineplots are the primary visual element and
    drive the figure width; scatter plots are compact secondary panels.

    Parameters
    ----------
    regimes : list of tuples
        Each entry is a 5- or 6-tuple::

            (name, results_df, settings_by_k, other_settings, estimator_type)
            (name, results_df, settings_by_k, other_settings, estimator_type, spectral_quant)

    k_star : int
        True cluster count to visualise (3, 4, 5, or 6).
    difficulty : str
        Difficulty anchor for example scatter plots (``'easy'``, ``'medium'``,
        or ``'hard'``).
    select_full_seed : bool
        If *True*, pick the seed whose PCA projection fills the axes most
        evenly (reduces whitespace, especially for non-linear manifolds).
    panels : {"both", "A", "B"}
        Which sections to render. ``"both"`` (default) draws the combined
        scatter + lineplot figure used in the original layout. ``"A"`` draws
        only the PCA scatter row (no lineplots, no shared y-label, no legend).
        ``"B"`` draws only the ARI-over-difficulty lineplots row (no scatter
        row, no section labels). The figure size is reduced accordingly.
    """
    if panels not in ("both", "A", "B"):
        raise ValueError(f"panels must be 'both', 'A', or 'B'; got {panels!r}")
    draw_top = panels in ("both", "A")
    draw_bot = panels in ("both", "B")
    show_section_labels = panels == "both"
    n_regimes = len(regimes)

    # ---- unpack regime tuples ----
    parsed = []
    for tup in regimes:
        if len(tup) == 6:
            name, df, sbk, osett, etype, sq = tup
        elif len(tup) == 5:
            name, df, sbk, osett, etype = tup
            sq = 0.5
        else:
            raise ValueError(
                "Each regime must be a 5- or 6-tuple: "
                "(name, results_df, settings_by_k, other_settings, "
                "estimator_type[, spectral_quant])"
            )
        parsed.append((name, df, sbk, osett, etype, sq))

    # ---- figure sizing (content-driven) ----
    ncols = 3
    nrows_top = 2 if n_regimes > 3 else 1
    nrows_bot = 2 if n_regimes > 3 else 1

    if figsize is None:
        # --- derive figure size: identical cells for scatter & lineplot ---
        cell_w_base = 3.6  # base cell width (inches)
        cell_w = cell_w_base * scale
        cell_h = cell_w * lineplot_height_ratio  # landscape: wider than tall

        margin_l = (
            0.65 * scale if draw_bot else 0.30 * scale
        )  # shared y-label only with B
        margin_r = 0.20 * scale  # right margin
        margin_t = 0.45 * scale if show_section_labels else 0.20 * scale
        margin_b = 1.60 * scale if draw_bot else 0.30 * scale  # legend only with B
        section_gap = 0.30 * scale if (draw_top and draw_bot) else 0.0

        gap_h = 0.45 * scale  # horizontal gap between cols (shared)
        gap_v = 0.50 * scale  # vertical gap between rows  (shared)

        # Width: driven by uniform cell grid
        fig_width = margin_l + ncols * cell_w + (ncols - 1) * gap_h + margin_r

        # Heights — both sections use the same cell dimensions
        top_h = nrows_top * cell_h + (nrows_top - 1) * gap_v if draw_top else 0.0
        bot_h = nrows_bot * cell_h + (nrows_bot - 1) * gap_v if draw_bot else 0.0

        fig_height = margin_t + top_h + section_gap + bot_h + margin_b
        figsize = (fig_width, fig_height)

        # Convert absolute margins → fractional for gridspec
        frac_top = 1.0 - margin_t / fig_height
        frac_bottom = margin_b / fig_height
        denom = top_h + bot_h if (top_h + bot_h) > 0 else 1.0
        frac_hspace = section_gap / denom

        # Height ratios (equal — same cell sizes)
        hr_top = top_h
        hr_bot = bot_h

        # Both grids share the same left / right margins
        frac_left = margin_l / fig_width
        frac_right = 1.0 - margin_r / fig_width

        lp_frac_left = frac_left
        lp_frac_right = frac_right
        scat_frac_left = frac_left
        scat_frac_right = frac_right

        # Inner spacing as fractions of cell dimensions (identical)
        wspace = gap_h / cell_w
        hspace = gap_v / cell_h
        scat_wspace = wspace
        scat_hspace = hspace
        lp_wspace = wspace
        lp_hspace = hspace + 0.08
    else:
        # If explicit figsize is given, use the old fractional defaults
        lp_frac_left, lp_frac_right = 0.06, 0.98
        scat_frac_left, scat_frac_right = 0.10, 0.94
        frac_top, frac_bottom = 0.95, 0.06
        frac_hspace = 0.08
        hr_top, hr_bot = 1, 1.8
        scat_wspace, scat_hspace = 0.04, 0.10
        lp_wspace, lp_hspace = 0.18, 0.28

    fig = plt.figure(figsize=figsize, constrained_layout=False)

    # Overall two-row layout (scatter row, lineplot row)
    # Use separate gridspecs so scatter and lineplot grids can have
    # independent horizontal extents (scatter is centred & narrower).
    content_top = frac_top
    content_bot = frac_bottom
    content_h = content_top - content_bot
    total_content = hr_top + hr_bot
    gap_frac = frac_hspace * content_h / (1.0 + frac_hspace)  # approx

    top_frac_h = (hr_top / total_content) * (content_h - gap_frac)
    bot_frac_h = (hr_bot / total_content) * (content_h - gap_frac)

    top_top = content_top
    top_bot = content_top - top_frac_h
    bot_top = top_bot - gap_frac
    bot_bot = bot_top - bot_frac_h

    # map regime index → (grid_row, grid_col)
    def _rc(idx):
        return idx // ncols, idx % ncols

    gs_top = None
    gs_bot = None
    axes_top: list = []
    axes_bot: list = []

    if draw_top:
        # Inner grid for Section A (PCA example scatter plots) — compact & centred
        gs_top = fig.add_gridspec(
            nrows=nrows_top,
            ncols=ncols,
            wspace=scat_wspace,
            hspace=scat_hspace,
            left=scat_frac_left,
            right=scat_frac_right,
            top=top_top,
            bottom=top_bot,
        )
        axes_top = [fig.add_subplot(gs_top[_rc(i)]) for i in range(n_regimes)]
        for i in range(n_regimes, nrows_top * ncols):
            fig.add_subplot(gs_top[_rc(i)]).set_visible(False)

    if draw_bot:
        # Inner grid for Section B (ARI lineplots) — full width, landscape
        gs_bot = fig.add_gridspec(
            nrows=nrows_bot,
            ncols=ncols,
            wspace=lp_wspace,
            hspace=lp_hspace,
            left=lp_frac_left,
            right=lp_frac_right,
            top=bot_top,
            bottom=bot_bot,
        )
        axes_bot = [fig.add_subplot(gs_bot[_rc(i)]) for i in range(n_regimes)]
        for i in range(n_regimes, nrows_bot * ncols):
            fig.add_subplot(gs_bot[_rc(i)]).set_visible(False)

    # ---- section labels A / B ----
    if show_section_labels:
        fig.text(
            0.008,
            0.95,
            "A",
            fontsize=20,
            fontweight="bold",
            va="top",
            ha="left",
            fontfamily="sans-serif",
        )
        # B label: y-position at the top of the lineplot rows
        b_y = gs_bot[0, 0].get_position(fig).y1 + 0.025
        fig.text(
            0.008,
            b_y,
            "B",
            fontsize=20,
            fontweight="bold",
            va="top",
            ha="left",
            fontfamily="sans-serif",
        )

    # ---- Row A — example scatter plots ----
    difficulty_j = {"easy": 0, "medium": 1, "hard": 2}.get(difficulty, 1)

    for col, (name, _df, sbk, osett, etype, sq) in enumerate(parsed):
        if not draw_top:
            break
        ax = axes_top[col]

        # generate seeds in parallel
        worker = delayed(_plotting_iter)
        results_list = Parallel(n_jobs=n_jobs)(
            worker(
                anchor_settings=sbk,
                other_settings=osett,
                j=difficulty_j,
                level=difficulty,
                level_label="difficulty",
                true_k=k_star,
                estimator_type=etype,
                sampler="default",
                seed=seed,
                random_state=random_state,
            )
            for seed in range(n_seeds_per_dataset)
        )

        # pick seed for plotting
        if select_full_seed and len(results_list) > 1:
            pick = _select_visually_full_seed(results_list)
        else:
            pick = 0
        _, X, y = results_list[pick]

        # PCA projection
        X_pca = PCA(n_components=2, random_state=0).fit_transform(X)
        labels = np.asarray(y)
        cmap = _cluster_color_map(labels)
        colors = [cmap[int(lbl)] for lbl in labels]

        ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=colors,
            s=scatter_size,
            alpha=0.90,
            edgecolor="k",
            linewidth=0.2,
            rasterized=True,
        )

        # non-linear regimes → auto aspect (less whitespace)
        is_nonlinear = osett.get("nonlinear", False)
        if is_nonlinear:
            ax.set_aspect("auto")
            ax.margins(0.04)
        else:
            ax.set_aspect("equal", adjustable="datalim")
            ax.margins(0.06)

        ax.set_title(name, fontsize=title_fontsize, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.15)

    # ---- Section B — ARI-over-difficulty lineplots (2×3) ----
    if draw_bot:
        for idx, (name, df, _sbk, _osett, _etype, _sq) in enumerate(parsed):
            ax = axes_bot[idx]

            # filter to k_star
            df_k = df.loc[df["true_k"] == k_star].copy()
            if df_k.empty:
                ax.set_visible(False)
                continue

            plot_ari_over_difficulty(
                df_k,
                metrics=metrics,
                center=center,
                band=band,
                show_band_for=show_band_for,
                x_label="SNR",
                x_label_font_size=tick_fontsize,
                ax=ax,
                ylim="auto",
                show_legend=False,
            )

            ax.set_title(name, fontsize=title_fontsize, fontweight="bold")
            ax.tick_params(labelsize=tick_fontsize)
            ax.set_ylabel("")  # remove per-axis y-labels; shared label below

    if not draw_bot:
        return fig

    # ---- shared y-axis label for all lineplots ----
    # Position at the left edge of the lineplot grid, vertically centred
    bot_pos = gs_bot[:, :].get_position(fig)
    fig.text(
        bot_pos.x0 - 0.035,
        (bot_pos.y0 + bot_pos.y1) / 2,
        r"ARI (selected $\hat{k}$ vs. true labels)",
        va="center",
        ha="center",
        rotation=90,
        fontsize=title_fontsize - 1,
    )

    # ---- shared legend (grouped: Baseline | CARVE | Classical) ----
    handles, labels_leg = axes_bot[0].get_legend_handles_labels()

    # split into groups
    baseline_h, baseline_l = [], []
    carve_h, carve_l = [], []
    classical_h, classical_l = [], []
    for h, lab in zip(handles, labels_leg):
        if "baseline" in lab.lower() or "oracle" in lab.lower():
            baseline_h.append(h)
            baseline_l.append(lab)
        elif "carve" in lab.lower():
            carve_h.append(h)
            carve_l.append(lab)
        else:
            classical_h.append(h)
            classical_l.append(lab)

    # ---- grouped legend (Baseline | CARVE | Classical) ----
    # Place legend centred below the lineplot grid, inside the bottom margin.
    # Positions are derived from the actual grid bounds so they adapt to any
    # figure size / scale.
    bot_pos = gs_bot[:, :].get_position(fig)
    grid_x0 = bot_pos.x0
    grid_x1 = bot_pos.x1
    grid_w = grid_x1 - grid_x0
    legend_y = bot_pos.y0 - 0.05  # just below the lowest lineplot row

    legend_kw = dict(
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.2,
        borderpad=0.3,
        labelspacing=0.30,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    # Build list of non-empty groups with their ncol preference
    groups = []
    if baseline_h:
        groups.append((baseline_h, baseline_l, 1))  # single column
    if carve_h:
        groups.append((carve_h, carve_l, 1))  # single column
    if classical_h:
        groups.append((classical_h, classical_l, 2))  # 2-col grid

    n_groups = len(groups)
    if n_groups > 0:
        # Spread groups evenly across the grid width
        for gi, (gh, gl, gncol) in enumerate(groups):
            # Position each group at equal fractions of the grid width
            frac = (gi + 0.5) / n_groups
            gx = grid_x0 + frac * grid_w
            fig.legend(
                gh,
                gl,
                loc="upper center",
                bbox_to_anchor=(gx, legend_y),
                ncol=gncol,
                **legend_kw,
            )

    return fig


# =============================================================================
# Public — Section 3 scaling plots (runtime + ARI grids per k★)
# =============================================================================
def plot_runtime_over_axis(
    runtimes_df: pd.DataFrame,
    *,
    runtime_cols: tuple[str, str] = ("t_carve_sec_s", "t_carve_sec_g"),
    runtime_labels: tuple[str, str] = (
        "CARVE Stability",
        "CARVE Generalizability",
    ),
    center: str = "mean",
    dodge: float = 0.01,
    x_col: str | None = None,
    x_label: str = "Number of samples (n)",
    x_label_font_size: int = 12,
    xscale: str = "linear",
    yscale: str = "log",
    y_label: str = "Runtime (seconds)",
    title: str | None = None,
    figsize: tuple = (6, 4.5),
    ax=None,
    show_legend: bool = True,
):
    """Plot CARVE runtime vs. a scaling axis.

    Mirrors :func:`plot_ari_over_difficulty` visually: each series is rendered
    by :func:`_draw_lineplot_series` with mean ± 1 SE antennas (capped error
    bars), and the two CARVE-mode curves are dodged horizontally by a small
    fraction of the x-range so the antennas don't overlap at each anchor.

    Args:
        runtimes_df: Long-form runtimes dataframe with at least
            ``axis_value`` (or one of ``n_total``/``p``/``embed_dim``),
            ``dataset_iteration`` and the columns listed in ``runtime_cols``.
        runtime_cols: Column names to plot as separate lines.
        runtime_labels: Display labels for ``runtime_cols`` (same length).
        center: ``'mean'`` or ``'median'``.
        dodge: Per-series horizontal offset, expressed as a fraction of the
            x-range. Set to ``0`` to disable.
        yscale: ``'log'`` (default) or ``'linear'``.
    """
    if len(runtime_cols) != len(runtime_labels):
        raise ValueError("runtime_cols and runtime_labels must be the same length")

    axis_col, _ = _infer_axis_cols(runtimes_df) if x_col is None else (x_col, "")
    needed = {axis_col, *runtime_cols}
    missing = sorted(c for c in needed if c not in runtimes_df.columns)
    if missing:
        raise ValueError(f"runtimes_df missing columns: {missing}")

    def _summarize(y: pd.Series):
        y = y.astype(float).dropna().values
        if y.size == 0:
            return np.nan, np.nan
        mid = np.mean(y) if center == "mean" else np.median(y)
        se = np.std(y, ddof=1) / np.sqrt(y.size) if y.size > 1 else 0.0
        return mid, se

    sums = []
    for col in runtime_cols:
        s = (
            runtimes_df.groupby(axis_col)[col]
            .apply(lambda v: pd.Series(_summarize(v), index=["mid", "se"]))
            .unstack()
            .reset_index()
            .sort_values(axis_col)
        )
        s["runtime_col"] = col
        sums.append(s)
    sum_df = pd.concat(sums, ignore_index=True)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    # Re-use the CARVE contrastive palette so colours match the ARI plot:
    # stability → green, generalizability → blue.
    palette = [mpl.colors.to_rgba(c) for c in lines_pallette_contrastive_carve]

    # Per-series horizontal offset so the two curves' antennas don't collide
    # at each anchor. Sized as a fraction of the x-range (or the log-x range
    # so it works on both scales).
    x_all = sum_df[axis_col].astype(float).values
    if dodge and x_all.size > 1:
        if xscale == "log" and (x_all > 0).all():
            log_span = np.log10(x_all.max()) - np.log10(x_all.min())
            dx_unit = dodge * log_span  # additive in log-space
        else:
            dx_unit = dodge * (x_all.max() - x_all.min())
    else:
        dx_unit = 0.0
    n = len(runtime_cols)
    center_off = (n - 1) / 2

    for i, (col, lab) in enumerate(zip(runtime_cols, runtime_labels)):
        s = sum_df.loc[sum_df["runtime_col"] == col]
        if s.empty:
            continue
        x_base = s[axis_col].astype(float).values
        if dx_unit and xscale == "log":
            xx = 10.0 ** (np.log10(x_base) + (i - center_off) * dx_unit)
        else:
            xx = x_base + (i - center_off) * dx_unit
        _draw_lineplot_series(
            ax,
            xx,
            s["mid"].values,
            s["se"].values,
            color=palette[i % len(palette)],
            label=lab,
        )

    ax.set_xlabel(x_label, fontsize=x_label_font_size, labelpad=8)
    ax.set_ylabel(y_label)

    if x_all.size > 0:
        ax.set_xlim(x_all.min(), x_all.max())
        ax.set_xticks(np.unique(x_all))

    if xscale in {"log", "linear"}:
        ax.set_xscale(xscale)
    if yscale in {"log", "linear"}:
        ax.set_yscale(yscale)

    if title is not None:
        ax.set_title(title)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, ncol=1)

    return fig
