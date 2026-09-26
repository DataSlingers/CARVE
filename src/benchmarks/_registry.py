"""The study design, in code.

Every benchmark scenario, its axis, its calibrated anchors, and the metric
vocabulary live here. Adding an experiment means adding a Scenario, not
writing a runner.
"""

from typing import Any

from ._types import Ablation, AblationScale, ArmScale, Axis, EstimatorSpec, Scenario

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

# The oracle reference. Not a value of metric_name -- it names the schema's
# oracle_ari column, one value per (axis point, seed) -- but it is requested
# alongside real metric names wherever a figure or table draws it, so it
# lives in the vocabulary with them.
BASELINE_METRIC: str = "baseline_oracle"

# Figure legends name the oracle with the k it is an oracle for, matching the
# published Fig 4. The supplementary tables' row label stays "Baseline
# (Oracle)", which is what the manuscript prints (S2-S9). One string cannot
# serve both, so the legend override lives here and _tables keeps reading
# METRIC_DISPLAY_NAMES directly.
#
# A literal asterisk, not mathtext. "$k^\star$" renders the superscript as a
# filled five-pointed star at roughly six points in a nine-point legend,
# which is a smudge rather than a symbol. The published figure used this
# plain string for the same reason.
METRIC_LEGEND_NAMES: dict[str, str] = {
    "baseline_oracle": "Baseline (Oracle k*)",
}

# The supplementary tables' row order and grouping, as the committed S2
# table prints them. Four groups, separated by a rule: the oracle, the two
# headline CARVE selectors, the four classical indices, then the remaining
# CARVE selectors.
#
# Declared rather than derived. table_metrics previously built its order
# from sorted(CARVE_METRICS_ALL) plus CVI_METRICS, so the generated
# fragment's rows were in a different order from the manuscript's and adding
# a metric would have silently reordered all eight tables. The three
# ari_average variants, PAC and CE are absent because tables.EXCLUDED_METRICS
# drops them, which is what the published tables do.
TABLE_ROW_GROUPS: tuple[tuple[str, ...], ...] = (
    (BASELINE_METRIC,),
    ("ari_stability_1se", "ari_generalizability_1se"),
    ("davies_bouldin", "silhouette", "gap", "calinski_harabasz"),
    (
        "ari_stability_quant",
        "consensus_gini_stability",
        "ari_stability",
        "ari_generalizability_quant",
        "ari_generalizability",
        "accuracy_generalizability",
    ),
)

# The scaling tables' order. The committed S8 and S9 order the classical
# indices Silhouette, Davies-Bouldin, Calinski-Harabasz, Gap, where S2 orders
# them Davies-Bouldin, Silhouette, Gap, Calinski-Harabasz; the other three
# groups match S2's. Declared for the same reason as TABLE_ROW_GROUPS, so a
# regenerated S8 or S9 drops in without reordering.
SCALING_TABLE_ROW_GROUPS: tuple[tuple[str, ...], ...] = (
    (BASELINE_METRIC,),
    ("ari_stability_1se", "ari_generalizability_1se"),
    ("silhouette", "davies_bouldin", "calinski_harabasz", "gap"),
    (
        "ari_stability_quant",
        "consensus_gini_stability",
        "ari_stability",
        "ari_generalizability_quant",
        "ari_generalizability",
        "accuracy_generalizability",
    ),
)

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

# Regenerated 2026-09-23 by _calibrate.calibrate_scenario, against the
# documented bands in TARGET_ARI_BANDS, with the shift-invert eigensolver
# and each scenario's final estimator in place. Run with the calibration
# module's defaults: n_seeds=20, random_state=PUBLISHED_RANDOM_STATE (42),
# max_iter=25. The achieved mean oracle ARI and its spread over the 20
# calibration datasets are recorded per anchor; the full comparison against
# PUBLISHED_ANCHORS is in
# docs/superpowers/notes/2026-09-22-calibration-comparison.md.
#
# Two knobs, because no single parameter controls difficulty on every
# scenario -- see _calibrate.CALIBRATION_KNOBS. The four scale-driven
# scenarios carry their published cluster_scale vector times a multiplier,
# so the unequal per-cluster scales survive; circles and moons carry an
# absolute embed_param. All eighteen anchors landed in their target band on
# the first calibration run -- no band needed widening.
CALIBRATED_ANCHORS: dict[str, dict[str, dict[str, Any]]] = {
    # Knob: cluster_scale multiplier. The published vector was [4.0] * 5 at
    # easy, [4.5] * 5 at medium and [4.6] * 5 at hard.
    "gaussians": {
        # multiplier 0.8250000000000001, ARI 0.984 +/- 0.004
        "easy": {
            "cluster_scale": [3.3000000000000003] * 5,
            "cluster_size_dirichlet_alpha": 0.9,
        },
        # multiplier 1.00625, ARI 0.862 +/- 0.022
        "medium": {
            "cluster_scale": [4.528125] * 5,
            "cluster_size_dirichlet_alpha": 0.5,
        },
        # multiplier 1.00625, ARI 0.740 +/- 0.163
        "hard": {
            "cluster_scale": [4.62875] * 5,
            "cluster_size_dirichlet_alpha": 0.1,
        },
    },
    # Knob: cluster_scale multiplier. The published vectors were
    # [3.5, 1.0, 1.0, 1.0, 1.0] at easy, [3.0, 1.5, 1.0, 1.0, 1.0] at
    # medium and [4.0, 3.0, 2.0, 1.0, 1.0] at hard.
    "t_dist": {
        # multiplier 1.55, ARI 0.935 +/- 0.034
        "easy": {
            "cluster_scale": [5.425, 1.55, 1.55, 1.55, 1.55],
            "cluster_size_dirichlet_alpha": 0.9,
            "t_df": 5,
        },
        # multiplier 1.1875, ARI 0.811 +/- 0.114
        "medium": {
            "cluster_scale": [3.5625, 1.78125, 1.1875, 1.1875, 1.1875],
            "cluster_size_dirichlet_alpha": 0.3,
            "t_df": 3,
        },
        # multiplier 1.00625, ARI 0.726 +/- 0.137
        "hard": {
            "cluster_scale": [4.025, 3.0187500000000003, 2.0125, 1.00625, 1.00625],
            "cluster_size_dirichlet_alpha": 0.1,
            "t_df": 3,
        },
    },
    # Knob: cluster_scale multiplier. The published vector was [1.0] * 5 at
    # easy, [1.1, 1.0, 1.0, 1.0, 1.0] at medium and hard.
    "t_dist_noise": {
        # multiplier 0.8250000000000001, ARI 0.995 +/- 0.003
        "easy": {
            "cluster_scale": [0.8250000000000001] * 5,
            "cluster_size_dirichlet_alpha": 0.9,
            "t_df": 5,
            "corr_strength": 0.1,
            "noise_dims": 512,
        },
        # multiplier 1.00625, ARI 0.846 +/- 0.037
        "medium": {
            "cluster_scale": [
                1.1068750000000003,
                1.00625,
                1.00625,
                1.00625,
                1.00625,
            ],
            "cluster_size_dirichlet_alpha": 0.3,
            "t_df": 4,
            "corr_strength": 0.3,
            "noise_dims": 1280,
        },
        # multiplier 1.00625, ARI 0.711 +/- 0.090
        "hard": {
            "cluster_scale": [
                1.1068750000000003,
                1.00625,
                1.00625,
                1.00625,
                1.00625,
            ],
            "cluster_size_dirichlet_alpha": 0.1,
            "t_df": 3,
            "corr_strength": 0.5,
            "noise_dims": 1536,
        },
    },
    # Knob: embed_param, absolute. The published values were 12.0, 7.3 and
    # 6.0, all past the 4.0 saturation point the shift-invert solver exposed.
    "circles": {
        # embed_param 3.5, ARI 0.981 +/- 0.084
        "easy": {
            "cluster_scale": [4.08, 4.08, 3.0, 3.0, 3.0],
            "cluster_size_dirichlet_alpha": 0.9,
            "corr_strength": 0.1,
            "embed_param": 3.5,
        },
        # embed_param 3.0, ARI 0.800 +/- 0.186
        "medium": {
            "cluster_scale": [4.08, 4.08, 3.0, 3.0, 3.0],
            "cluster_size_dirichlet_alpha": 0.61,
            "corr_strength": 0.23,
            "embed_param": 3.0,
        },
        # embed_param 3.25, ARI 0.724 +/- 0.282
        "hard": {
            "cluster_scale": [4.38, 4.08, 4.08, 4.08, 4.08],
            "cluster_size_dirichlet_alpha": 0.35,
            "corr_strength": 0.20,
            "embed_param": 3.25,
        },
    },
    # Knob: embed_param, absolute. The published values were 10.7, 5.7 and
    # 14.5, all past or straddling the saturation point.
    "moons": {
        # embed_param 3.5, ARI 0.967 +/- 0.102
        "easy": {
            "cluster_scale": [5.5, 3.97, 3.97, 3.97, 3.97],
            "cluster_size_dirichlet_alpha": 0.67,
            "corr_strength": 0.39,
            "embed_param": 3.5,
        },
        # embed_param 3.0, ARI 0.842 +/- 0.168
        "medium": {
            "cluster_scale": [4.8, 4.06, 4.06, 4.06, 4.06],
            "cluster_size_dirichlet_alpha": 0.57,
            "corr_strength": 0.30,
            "embed_param": 3.0,
        },
        # embed_param 3.5, ARI 0.766 +/- 0.204
        "hard": {
            "cluster_scale": [4.06, 2.65, 2.65, 2.65, 2.65],
            "cluster_size_dirichlet_alpha": 0.10,
            "corr_strength": 0.16,
            "embed_param": 3.5,
        },
    },
    # Knob: cluster_scale multiplier. The published vector was [1.0] * 5 at
    # easy, [2.0, 2.0, 1.0, 1.0, 1.0] at medium and hard.
    "swiss_rolls": {
        # multiplier 1.55, ARI 1.000 +/- 0.000
        "easy": {
            "cluster_scale": [1.55] * 5,
            "cluster_size_dirichlet_alpha": 0.9,
            "corr_strength": 0.1,
            "embed_param": 8.0,
        },
        # multiplier 1.1875, ARI 0.889 +/- 0.144
        "medium": {
            "cluster_scale": [2.375, 2.375, 1.1875, 1.1875, 1.1875],
            "cluster_size_dirichlet_alpha": 0.7,
            "corr_strength": 0.1,
            "embed_param": 5.0,
        },
        # multiplier 1.55, ARI 0.739 +/- 0.264
        "hard": {
            "cluster_scale": [3.1, 3.1, 1.55, 1.55, 1.55],
            "cluster_size_dirichlet_alpha": 0.5,
            "corr_strength": 0.3,
            "embed_param": 5.0,
        },
    },
    # The two scaling scenarios are carried over: TARGET_ARI_BANDS is keyed
    # on easy/medium/hard and they sweep n and p.
    "gaussians_dimensionality": PUBLISHED_ANCHORS["gaussians_dimensionality"],
    "gaussians_samples": PUBLISHED_ANCHORS["gaussians_samples"],
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
# kept so the regenerated calibration can be compared against what the
# manuscript reported, and reverted to. ACTIVE_ANCHOR_SET_NAME is recorded in
# every run manifest so no artifact is ambiguous about which produced it.
ACTIVE_ANCHORS: dict[str, dict[str, dict[str, Any]]] = CALIBRATED_ANCHORS
ACTIVE_ANCHOR_SET_NAME: str = "CALIBRATED_ANCHORS"

_ESTIMATORS: dict[str, str] = {
    "gaussians": "kmeans",
    "t_dist": "agglomerative",
    "t_dist_noise": "agglomerative",
    "circles": "spectral",
    "moons": "spectral",
    # The published S1 Fig panel was drawn with "spectral" while the
    # benchmark that produced Table S7 ran agglomerative, and S3 Text said
    # Ward. Ward was chosen because spectral scored worse here -- which was
    # carve.cluster's which="SM" eigensolver failing to converge at n=1500,
    # not the data. With the shift-invert solver spectral wins at every
    # difficulty and runs 30x faster, so the illustration was right and the
    # numbers were not.
    "swiss_rolls": "spectral",
    "gaussians_dimensionality": "kmeans",
    "gaussians_samples": "kmeans",
}

# One forest size for every scenario. The published benchmark did not use
# one value everywhere -- notebook cells 29, 35 and 41 passed n_trees=500
# for circles, moons and swiss rolls while cells 11, 17, 23, 48 and 53
# omitted it and got Scenario's default of 100 -- but that split tracked
# which cell an argument was typed into, not anything about the data. 500 is
# the better generalizability estimate; the cost is linear in a term that is
# not the bottleneck.
_N_TREES: dict[str, int] = dict.fromkeys(_ESTIMATORS, 500)

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

# Scenarios that additionally run two mode-specific CARVE fits purely to time
# them. This reproduces the published two-curve runtime figure, which was
# produced by a runner that fit each mode separately. The extra fits more than
# double a cell's cost, so only the scaling scenarios carry them.
TIMED_SCENARIOS: frozenset[str] = frozenset(
    {"gaussians_samples", "gaussians_dimensionality"}
)

# The seed the published benchmarks ran with. Notebook cell 3 sets
# RANDOM_SEED = 42 and every scenario call passes it. Seeds derive as
# benchmark_seed = seed + axis_idx * 10000 + random_state, so this value
# is what makes a run reproduce the committed results.
PUBLISHED_RANDOM_STATE: int = 42

# =============================================================================
# Ablations
# =============================================================================
# One CARVE fit draws its subsamples from seeds in [random_state,
# random_state + 3 * n_resamples): P_1 at random_state + b, P_2 at
# random_state + b + B and the P_test pipeline at random_state + b + 2B.
# Replicate fits of one dataset are spaced by this much so no two share a
# subsample; the spacing also clears the largest benchmark seed offset
# (two axis steps of 10,000 plus the dataset index), so the seed windows
# of different replicate indices never intersect, whichever datasets they
# belong to. Within one replicate index, the windows of different datasets
# may overlap, as in the published benchmark (spec 4.2): they draw
# subsamples of different data.
REPLICATE_SEED_SPACING: int = 1_000_000

# Subsample-versus-full similarity draws seed from base + this + draw; it
# sits above replicate 0's window and below replicate 1's.
SIMILARITY_SEED_OFFSET: int = 500_000


def package_defaults() -> tuple[float, int]:
    """CARVE's own (subsample_ratio, n_resamples) defaults, read off the class.

    Imported lazily so this module's top level stays free of carve, as
    _types does; Scenario already imports carve.sim at construction.
    """
    from dataclasses import fields

    from carve import CARVE

    defaults = {field.name: field.default for field in fields(CARVE)}
    return float(defaults["subsample_ratio"]), int(defaults["n_resamples"])


_RHO_DEFAULT, _B_DEFAULT = package_defaults()

# The medium gaussians setting alone, with no case study (the author's
# decision on 2026-09-26; the ablation previously ran six scenarios at every
# difficulty plus Klein). The rho arm's twenty datasets are the main
# benchmark's twenty medium gaussians datasets, so its default column
# (rho_default, b_default, replicate 0) reproduces those fits.
ABLATION_SCALES: dict[str, AblationScale] = {
    "publication": AblationScale(
        rho_arm=ArmScale(
            difficulties=("medium",),
            datasets=tuple(range(20)),
            replicates=1,
            study_replicates=0,
        ),
        b_arm=ArmScale(
            difficulties=("medium",),
            datasets=tuple(range(10)),
            replicates=3,
            study_replicates=0,
        ),
        similarity_draws=20,
    ),
}

ABLATIONS: dict[str, Ablation] = {
    "rho_b": Ablation(
        name="rho_b",
        scenarios=("gaussians",),
        study=None,
        # 0.5 and 0.8 are proportions used elsewhere in the resampling
        # literature, so the default can be placed against them.
        rho_grid=(0.2, 0.3, 0.4, 0.5, _RHO_DEFAULT, 0.7, 0.8, 0.9),
        b_grid=(10, 25, 50, _B_DEFAULT, 200),
        rho_default=_RHO_DEFAULT,
        b_default=_B_DEFAULT,
        scales=ABLATION_SCALES,
        default_scale="publication",
    ),
}


def max_benchmark_seed_offset(ablation: Ablation) -> int:
    """Largest benchmark seed minus PUBLISHED_RANDOM_STATE over the ablation's cells."""
    offset = 0
    for scale in ablation.scales.values():
        for arm in (scale.rho_arm, scale.b_arm):
            for label in arm.difficulties:
                if label in DIFFICULTY_AXIS.labels:
                    idx = DIFFICULTY_AXIS.labels.index(label)
                    offset = max(offset, idx * 10000 + max(arm.datasets))
    return offset


def validate_ablation(ablation: Ablation) -> None:
    """Check an ablation against the registry it refers to.

    Ablation.__post_init__ checks what the dataclass can see on its own.
    This checks the rest: scenario names, the difficulty axis, and the seed
    constants against the widest seed span any cell can use.
    """
    for name in ablation.scenarios:
        if name not in SCENARIOS:
            raise ValueError(
                f"Ablation {ablation.name!r}: scenario {name!r} is not registered."
            )
        if SCENARIOS[name].axis.name != DIFFICULTY_AXIS.name:
            raise ValueError(
                f"Ablation {ablation.name!r}: scenario {name!r} sweeps "
                f"{SCENARIOS[name].axis.name!r}, not the difficulty axis."
            )
    for scale_name, scale in ablation.scales.items():
        for arm in (scale.rho_arm, scale.b_arm):
            unknown = sorted(set(arm.difficulties) - set(DIFFICULTY_AXIS.labels))
            if unknown:
                raise ValueError(
                    f"Ablation {ablation.name!r}, scale {scale_name!r}: unknown "
                    f"difficulty label(s) {unknown}."
                )
    span = 3 * max(ablation.b_grid)
    offset = max_benchmark_seed_offset(ablation)
    draws = max(scale.similarity_draws for scale in ablation.scales.values())
    if REPLICATE_SEED_SPACING <= span + offset:
        raise ValueError(
            f"REPLICATE_SEED_SPACING={REPLICATE_SEED_SPACING} does not clear one "
            f"fit's seed span ({span}) plus the seed offset ({offset})."
        )
    if SIMILARITY_SEED_OFFSET < span + offset:
        raise ValueError(
            f"SIMILARITY_SEED_OFFSET={SIMILARITY_SEED_OFFSET} lies inside replicate 0's "
            f"seed window (span {span}, offset {offset})."
        )
    if SIMILARITY_SEED_OFFSET + draws + offset > REPLICATE_SEED_SPACING - span:
        raise ValueError(
            f"SIMILARITY_SEED_OFFSET={SIMILARITY_SEED_OFFSET} plus {draws} draws reaches "
            f"replicate 1's seed window."
        )


for _ablation in ABLATIONS.values():
    validate_ablation(_ablation)
