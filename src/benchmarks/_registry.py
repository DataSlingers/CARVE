"""The study design, in code.

Every benchmark scenario, its axis, its calibrated anchors, and the metric
vocabulary live here. Adding an experiment means adding a Scenario, not
writing a runner.
"""

from typing import Any

from ._types import Axis, EstimatorSpec, Scenario

# =============================================================================
# Metric vocabulary
# =============================================================================
CARVE_METRICS_STABILITY: tuple[str, ...] = (
    "ari_stability",
    "ari_stability_1se",
    "ari_stability_quant",
    "consensus_pac_stability",
    "consensus_gini_stability",
    "consensus_ce_stability",
)

CARVE_METRICS_GENERALIZABILITY: tuple[str, ...] = (
    "ari_generalizability",
    "ari_generalizability_1se",
    "ari_generalizability_quant",
    "accuracy_generalizability",
)

CARVE_METRICS_COMBINED: tuple[str, ...] = (
    "ari_average",
    "ari_average_1se",
    "ari_average_quant",
)

CARVE_METRICS_ALL: tuple[str, ...] = tuple(
    sorted(
        set(
            CARVE_METRICS_STABILITY
            + CARVE_METRICS_GENERALIZABILITY
            + CARVE_METRICS_COMBINED
        )
    )
)

CVI_METRICS: tuple[str, ...] = (
    "silhouette",
    "gap",
    "davies_bouldin",
    "calinski_harabasz",
)

# Which CARVE consensus matrix a metric's labels must come from. The old
# difficulty runner never passed mode= to get_labels, so "default" resolved to
# run_stability=True and every metric's ari_at_k came from the stability
# matrix, including the generalizability metrics.
GENERALIZABILITY_METRICS: frozenset[str] = frozenset(
    m for m in CARVE_METRICS_ALL if "generalizability" in m
)

METRIC_DISPLAY_NAMES: dict[str, str] = {
    "baseline_oracle": "Baseline (Oracle)",
    "ari_stability": "ARI (stab, max)",
    "ari_stability_1se": "CARVE Stability (1SE)",
    "ari_stability_quant": "ARI (stab, quantile)",
    "ari_generalizability": "ARI (gen, max)",
    "ari_generalizability_1se": "CARVE Generalizability (1SE)",
    "ari_generalizability_quant": "ARI (gen, quantile)",
    "ari_average": "ARI (avg, max)",
    "ari_average_1se": "ARI (avg, 1SE)",
    "ari_average_quant": "ARI (avg, quantile)",
    "consensus_pac_stability": "PAC (stab)",
    "consensus_gini_stability": "Gini (stab)",
    "consensus_ce_stability": "CE (stab)",
    "accuracy_generalizability": "Accuracy (gen)",
    "silhouette": "Silhouette",
    "gap": "Gap Statistic",
    "davies_bouldin": "Davies-Bouldin",
    "calinski_harabasz": "Calinski-Harabasz",
}

N_REFERENCE_DATASETS: int = 10


def metric_rule(name: str) -> str:
    """Return the CARVE selection rule encoded in a metric name's suffix."""
    if name.endswith("_quant"):
        return "quantile"
    if name.endswith("_1se"):
        return "1se"
    return "max"


def metric_measure(name: str) -> str:
    """Return the CARVE measure name with any selection-rule suffix removed."""
    if name.endswith("_1se"):
        return name[:-4]
    if name.endswith("_quant"):
        return name[:-6]
    return name


# =============================================================================
# Axes
# =============================================================================
DIFFICULTY_AXIS = Axis(
    name="difficulty_level",
    values=(0, 1, 2),
    labels=("easy", "medium", "hard"),
)

# Ranges and the three-point granularity are ports of SCALING_RANGES and
# GRANULARITY in benchmarking_config.py, evaluated to literals so the axis is
# readable without running numpy.linspace in your head. All three axes are
# kept available even though only "p" and "n_total" are swept by a registered
# scenario below (see gaussians_dimensionality / gaussians_samples).
SCALING_AXES: dict[str, Axis] = {
    "n_total": Axis(
        name="n_total", values=(1000, 5500, 10000), labels=("start", "middle", "end")
    ),
    "p": Axis(name="p", values=(50, 525, 1000), labels=("start", "middle", "end")),
    "embed_dim": Axis(
        name="embed_dim", values=(10, 255, 500), labels=("start", "middle", "end")
    ),
}

# =============================================================================
# Published anchors — ported verbatim from notebooks/Benchmarking.ipynb
# =============================================================================
# Cell numbers refer to notebooks/Benchmarking.ipynb as of this port. Extract
# with:
#   .venv/bin/python - <<'PY'
#   import json
#   nb = json.load(open("notebooks/Benchmarking.ipynb"))
#   for i in (9, 15, 21, 27, 33, 39, 46, 51):
#       print(f"# ----- cell {i} -----")
#       print("".join(nb["cells"][i]["source"]))
#   PY
PUBLISHED_ANCHORS: dict[str, dict[str, dict[str, Any]]] = {
    # Cell 9
    "gaussians": {
        "easy": {"cluster_scale": [4.0] * 5, "cluster_size_dirichlet_alpha": 0.9},
        "medium": {"cluster_scale": [4.5] * 5, "cluster_size_dirichlet_alpha": 0.5},
        "hard": {"cluster_scale": [4.6] * 5, "cluster_size_dirichlet_alpha": 0.1},
    },
    # Cell 15
    "t_dist": {
        "easy": {
            "cluster_scale": [3.5, 1.0, 1.0, 1.0, 1.0],
            "cluster_size_dirichlet_alpha": 0.9,
            "t_df": 5,
        },
        "medium": {
            "cluster_scale": [3.0, 1.5, 1.0, 1.0, 1.0],
            "cluster_size_dirichlet_alpha": 0.3,
            "t_df": 3,
        },
        "hard": {
            "cluster_scale": [4.0, 3.0, 2.0, 1.0, 1.0],
            "cluster_size_dirichlet_alpha": 0.1,
            "t_df": 3,
        },
    },
    # Cell 21
    "t_dist_noise": {
        "easy": {
            "cluster_scale": [1.0] * 5,
            "cluster_size_dirichlet_alpha": 0.9,
            "t_df": 5,
            "corr_strength": 0.1,
            "noise_dims": 512,
        },
        "medium": {
            "cluster_scale": [1.1, 1.0, 1.0, 1.0, 1.0],
            "cluster_size_dirichlet_alpha": 0.3,
            "t_df": 4,
            "corr_strength": 0.3,
            "noise_dims": 1280,
        },
        "hard": {
            "cluster_scale": [1.1, 1.0, 1.0, 1.0, 1.0],
            "cluster_size_dirichlet_alpha": 0.1,
            "t_df": 3,
            "corr_strength": 0.5,
            "noise_dims": 1536,
        },
    },
    # Cell 27
    "circles": {
        "easy": {
            "cluster_scale": [4.08, 4.08, 3.0, 3.0, 3.0],
            "cluster_size_dirichlet_alpha": 0.9,
            "corr_strength": 0.1,
            "embed_param": 12.0,
        },
        "medium": {
            "cluster_scale": [4.08, 4.08, 3.0, 3.0, 3.0],
            "cluster_size_dirichlet_alpha": 0.61,
            "corr_strength": 0.23,
            "embed_param": 7.3,
        },
        "hard": {
            "cluster_scale": [4.38, 4.08, 4.08, 4.08, 4.08],
            "cluster_size_dirichlet_alpha": 0.35,
            "corr_strength": 0.20,
            "embed_param": 6.0,
        },
    },
    # Cell 33
    "moons": {
        "easy": {
            "cluster_scale": [5.5, 3.97, 3.97, 3.97, 3.97],
            "cluster_size_dirichlet_alpha": 0.67,
            "corr_strength": 0.39,
            "embed_param": 10.7,
        },
        "medium": {
            "cluster_scale": [4.8, 4.06, 4.06, 4.06, 4.06],
            "cluster_size_dirichlet_alpha": 0.57,
            "corr_strength": 0.30,
            "embed_param": 5.7,
        },
        "hard": {
            "cluster_scale": [4.06, 2.65, 2.65, 2.65, 2.65],
            "cluster_size_dirichlet_alpha": 0.10,
            "corr_strength": 0.16,
            "embed_param": 14.5,
        },
    },
    # Cell 39
    "swiss_rolls": {
        "easy": {
            "cluster_scale": [1.0, 1.0, 1.0, 1.0, 1.0],
            "cluster_size_dirichlet_alpha": 0.9,
            "corr_strength": 0.1,
            "embed_param": 8.0,
        },
        "medium": {
            "cluster_scale": [2.0, 2.0, 1.0, 1.0, 1.0],
            "cluster_size_dirichlet_alpha": 0.7,
            "corr_strength": 0.1,
            "embed_param": 5.0,
        },
        "hard": {
            "cluster_scale": [2.0, 2.0, 1.0, 1.0, 1.0],
            "cluster_size_dirichlet_alpha": 0.5,
            "corr_strength": 0.3,
            "embed_param": 5.0,
        },
    },
    # Cell 46
    "gaussians_dimensionality": {
        "start": {"cluster_scale": [4.3] * 5},
        "middle": {"cluster_scale": [10.5] * 5},
        "end": {"cluster_scale": [12.2] * 5},
    },
    # Cell 51
    "gaussians_samples": {
        "start": {"cluster_scale": [4.2] * 5},
        "middle": {"cluster_scale": [4.4] * 5},
        "end": {"cluster_scale": [4.5] * 5},
    },
}

# other_settings_* from the same notebook cells. For the two scaling
# scenarios the notebook's other_settings carries none of n_total, p, or
# embed_dim — parse_range_and_simulate (benchmarking_simulation_helpers.py)
# always supplies all three, taking the swept one from the axis value and the
# other two from SCALING_CONSTANTS = {"n_total": 1500, "p": 50,
# "embed_dim": 64}. The swept parameter itself (p for
# gaussians_dimensionality, n_total for gaussians_samples) is set by
# Scenario.sim_kwargs from the axis value and must not also appear here.
_SHARED: dict[str, dict[str, Any]] = {
    "gaussians": {"n_total": 1500, "p": 50},
    "t_dist": {"n_total": 1500, "p": 50, "distribution": "t"},
    "t_dist_noise": {
        "n_total": 1500,
        "p": 50,
        "corr_type": "ar1",
        "distribution": "t",
    },
    "circles": {
        "distribution": "circles",
        "nonlinear": True,
        "n_total": 1500,
        "p": 50,
        "corr_type": "ar1",
        "embed_dim": 64,
    },
    "moons": {
        "distribution": "moons",
        "nonlinear": True,
        "n_total": 1500,
        "p": 50,
        "corr_type": "ar1",
        "embed_dim": 64,
    },
    "swiss_rolls": {
        "distribution": "swiss_roll",
        "nonlinear": True,
        "n_total": 1500,
        "p": 50,
        "corr_type": "ar1",
        "embed_dim": 64,
        "center_box": 3.0,
    },
    # Axis is "p" — n_total and embed_dim are the SCALING_CONSTANTS values.
    "gaussians_dimensionality": {
        "corr_type": "ar1",
        "corr_strength": 0.5,
        "cluster_size_dirichlet_alpha": 0.5,
        "n_total": 1500,
        "embed_dim": 64,
    },
    # Axis is "n_total" — p and embed_dim are the SCALING_CONSTANTS values.
    "gaussians_samples": {
        "corr_type": "ar1",
        "corr_strength": 0.5,
        "cluster_size_dirichlet_alpha": 0.5,
        "p": 50,
        "embed_dim": 64,
    },
}

# The calibration notebook that produced PUBLISHED_ANCHORS was lost; see
# _calibrate.py, which defines a documented replacement search. Both sets are
# kept so a regenerated calibration can be compared against what the
# manuscript reported, and reverted to. ACTIVE_ANCHOR_SET_NAME is recorded in
# every run manifest so no artifact is ambiguous about which produced it.
ACTIVE_ANCHORS: dict[str, dict[str, dict[str, Any]]] = PUBLISHED_ANCHORS
ACTIVE_ANCHOR_SET_NAME: str = "PUBLISHED_ANCHORS"

_ESTIMATORS: dict[str, str] = {
    "gaussians": "kmeans",
    "t_dist": "agglomerative",
    "t_dist_noise": "agglomerative",
    "circles": "spectral",
    "moons": "spectral",
    # The published S1 Fig panel was drawn with "spectral", but the benchmark
    # that produced Table S7 ran agglomerative, and S3 Text says Ward. The
    # results were correct; only the illustration disagreed. Agglomerative is
    # what the numbers came from.
    "swiss_rolls": "agglomerative",
    "gaussians_dimensionality": "kmeans",
    "gaussians_samples": "kmeans",
}

# n_trees sizes the random forest behind CARVE's generalizability computation.
# The published benchmark did not use one value everywhere: notebook cells
# 29, 35, and 41 pass n_trees=500 for circles, moons, and swiss_rolls; cells
# 11, 17, 23, 48, and 53 omit it and get Scenario's default of 100.
_N_TREES: dict[str, int] = {
    "gaussians": 100,
    "t_dist": 100,
    "t_dist_noise": 100,
    "circles": 500,
    "moons": 500,
    "swiss_rolls": 500,
    "gaussians_dimensionality": 100,
    "gaussians_samples": 100,
}

_AXES: dict[str, Axis] = {
    "gaussians": DIFFICULTY_AXIS,
    "t_dist": DIFFICULTY_AXIS,
    "t_dist_noise": DIFFICULTY_AXIS,
    "circles": DIFFICULTY_AXIS,
    "moons": DIFFICULTY_AXIS,
    "swiss_rolls": DIFFICULTY_AXIS,
    # Notebook cell 48 calls benchmark_scaling(..., axis_name="p"); the
    # committed results_gaussian_dimensionality.csv records axis_name="p"
    # with axis_value in {50, 525, 1000}. "Dimensionality" here means the
    # ambient feature dimension, not the embedding dimension.
    "gaussians_dimensionality": SCALING_AXES["p"],
    "gaussians_samples": SCALING_AXES["n_total"],
}

SCENARIOS: dict[str, Scenario] = {
    name: Scenario(
        name=name,
        axis=_AXES[name],
        anchors=ACTIVE_ANCHORS[name],
        shared=_SHARED[name],
        estimator=EstimatorSpec(name=_ESTIMATORS[name]),
        n_trees=_N_TREES[name],
    )
    for name in _ESTIMATORS
}

# The seed the published benchmarks ran with. Notebook cell 3 sets
# RANDOM_SEED = 42 and every scenario call passes it. Seeds derive as
# benchmark_seed = seed + axis_idx * 10000 + random_state, so this value
# is what makes a run reproduce the committed results.
PUBLISHED_RANDOM_STATE: int = 42
