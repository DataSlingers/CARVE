"""Centralized configuration for benchmarking experiments.

Edit constants here to change scaling ranges, metric lists, or display settings.
All other benchmarking modules import from this file.
"""

import numpy as np


# ── Scaling ranges (must match Calibrate_Settings.ipynb) ──────────────────────
# All axes use LINEAR spacing (np.linspace), never logarithmic.
# To change ranges: edit here and re-run the scaling benchmarks.
SCALING_RANGES: dict[str, dict[str, int]] = {
    "n_total": {"min": 100, "max": 1500},
    "p": {"min": 10, "max": 500},
    "embed_dim": {"min": 10, "max": 500},
}

GRANULARITY = 10  # number of points per scaling axis


def make_scaling_x_values(axis_name: str, granularity: int = GRANULARITY) -> list[int]:
    """Generate linearly-spaced integer values for a scaling axis.

    Uses SCALING_RANGES as the single source of truth for min/max bounds.

    Args:
        axis_name: One of the keys in SCALING_RANGES ('n_total', 'p', 'embed_dim').
        granularity: Number of evenly-spaced points to generate.

    Returns:
        List of integer values from min to max (inclusive) with linear spacing.
    """
    if axis_name not in SCALING_RANGES:
        raise ValueError(
            f"axis_name must be one of {sorted(SCALING_RANGES)}, got {axis_name!r}"
        )
    r = SCALING_RANGES[axis_name]
    return [int(x) for x in np.linspace(r["min"], r["max"], num=granularity)]


# ── CARVE metrics ─────────────────────────────────────────────────────────────

CARVE_METRICS_STABILITY = [
    "ari_stability",
    "ari_stability_1se",
    "ari_stability_quant",
    "consensus_pac_stability",
    "consensus_gini_stability",
    "consensus_ce_stability",
]

CARVE_METRICS_GENERALIZABILITY = [
    "ari_generalizability",
    "ari_generalizability_1se",
    "ari_generalizability_quant",
    "accuracy_generalizability",
]

CARVE_METRICS_COMBINED = [
    "ari_average",
    "ari_average_1se",
    "ari_average_quant",
]

CARVE_METRICS_ALL = sorted(
    set(
        CARVE_METRICS_STABILITY
        + CARVE_METRICS_GENERALIZABILITY
        + CARVE_METRICS_COMBINED
    )
)


# ── External (classical) metrics ──────────────────────────────────────────────

EXTERNAL_METRICS = ("silhouette", "gap", "davies_bouldin", "calinski_harabasz")


# ── Metric display names (used by reporting and plotting) ─────────────────────

METRIC_DISPLAY_NAMES = {
    "baseline_oracle": "Baseline (Oracle)",
    # CARVE – stability
    "ari_stability": "ARI (stab, max)",
    "ari_stability_1se": "CARVE Stability (1SE)",
    "ari_stability_quant": "ARI (stab, quantile)",
    # CARVE – generalizability
    "ari_generalizability": "ARI (gen, max)",
    "ari_generalizability_1se": "CARVE Generalizability (1SE)",
    "ari_generalizability_quant": "ARI (gen, quantile)",
    # CARVE – combined
    "ari_average": "ARI (avg, max)",
    "ari_average_1se": "ARI (avg, 1SE)",
    "ari_average_quant": "ARI (avg, quantile)",
    # Consensus
    "consensus_pac_stability": "PAC (stab)",
    "consensus_gini_stability": "Gini (stab)",
    "consensus_ce_stability": "CE (stab)",
    # Misclassification
    "accuracy_generalizability": "Accuracy (gen)",
    # Classical
    "silhouette": "Silhouette",
    "gap": "Gap Statistic",
    "davies_bouldin": "Davies\u2013Bouldin",
    "calinski_harabasz": "Calinski\u2013Harabasz",
}


# ── Metric groupings for plotting / reporting ─────────────────────────────────

SELECTED_CARVE = ["ari_generalizability_1se", "ari_stability_1se"]

CLASSICAL = list(EXTERNAL_METRICS)

PLOT_METRICS = [
    "ari_average_1se",
    "ari_generalizability_1se",
    "ari_stability_1se",
    "silhouette",
    "gap",
    "davies_bouldin",
    "calinski_harabasz",
]

PLOT_METRICS_CARVE = [
    "ari_average_1se",
    "ari_generalizability_1se",
    "ari_stability_1se",
]

PLOT_METRICS_EXT = [
    "ari_stability_1se",
    "silhouette",
    "gap",
    "davies_bouldin",
    "calinski_harabasz",
]

EXCLUDE_FROM_TABLES = [
    "ari_average",
    "ari_average_1se",
    "ari_average_quant",
    "consensus_pac_stability",
    "consensus_ce_stability",
]


# ── Metric colors (Okabe-Ito colorblind-safe palette) ─────────────────────────

METRIC_COLOR = {
    "ari_average_1se": "#E69F00",  # orange
    "ari_generalizability_1se": "#56B4E9",  # sky blue
    "ari_stability_1se": "#009E73",  # bluish green
    "silhouette": "#E8588C",  # pink
    "gap": "#0072B2",  # blue
    "davies_bouldin": "#D55E00",  # vermillion
    "calinski_harabasz": "#CC79A7",  # reddish purple
}

METRIC_LABEL = {
    "ari_average_1se": "ARI (avg, 1se)",
    "ari_generalizability_1se": "ARI (gen, 1se)",
    "ari_stability_1se": "ARI (stab, 1se)",
    "silhouette": "Silhouette",
    "gap": "Gap",
    "davies_bouldin": "DB",
    "calinski_harabasz": "CH",
}
