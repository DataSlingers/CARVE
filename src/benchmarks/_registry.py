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

# The oracle reference. It names the schema's oracle_ari column (one value
# per axis point and seed) rather than a metric_name value, but figures and
# tables request it together with the metrics.
BASELINE_METRIC: str = "baseline_oracle"

# Figure legends name the oracle with its k. The supplementary tables keep
# "Baseline (Oracle)" from METRIC_DISPLAY_NAMES. A plain asterisk, because
# mathtext's \star is illegible at legend size.
METRIC_LEGEND_NAMES: dict[str, str] = {
    "baseline_oracle": "Baseline (Oracle k*)",
}

# Row order of the supplementary benchmark tables, in four groups separated
# by a rule: the oracle, the two headline CARVE selectors, the four classical
# indices, then the other CARVE selectors. The ari_average variants, PAC and
# CE are not listed because tables.EXCLUDED_METRICS drops them.
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

# The scaling tables order the classical indices Silhouette,
# Davies-Bouldin, Calinski-Harabasz, Gap. The other groups match
# TABLE_ROW_GROUPS.
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

# Only "p" and "n_total" are swept by a registered scenario
# (gaussians_dimensionality and gaussians_samples).
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
# Published anchors
# =============================================================================
# The simulation settings of the benchmark in the first manuscript version.
PUBLISHED_ANCHORS: dict[str, dict[str, dict[str, Any]]] = {
    "gaussians": {
        "easy": {"cluster_scale": [4.0] * 5, "cluster_size_dirichlet_alpha": 0.9},
        "medium": {"cluster_scale": [4.5] * 5, "cluster_size_dirichlet_alpha": 0.5},
        "hard": {"cluster_scale": [4.6] * 5, "cluster_size_dirichlet_alpha": 0.1},
    },
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
    # One roll per cluster, each around its own center. The registered
    # scenario draws interleaved spirals (see _SHARED), which these
    # parameters do not apply to.
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
    "gaussians_dimensionality": {
        "start": {"cluster_scale": [4.3] * 5},
        "middle": {"cluster_scale": [10.5] * 5},
        "end": {"cluster_scale": [12.2] * 5},
    },
    "gaussians_samples": {
        "start": {"cluster_scale": [4.2] * 5},
        "middle": {"cluster_scale": [4.4] * 5},
        "end": {"cluster_scale": [4.5] * 5},
    },
}

# Calibrated at n_total = BENCHMARK_N_TOTAL against TARGET_ARI_BANDS by
# _calibrate.calibrate_anchor with calibrate_scenario's defaults (20 seeds,
# random_state = PUBLISHED_RANDOM_STATE, max_iter = 25). Each anchor notes
# its knob value and the mean and standard deviation of the oracle ARI over
# the 20 calibration datasets. The knob depends on the scenario
# (_calibrate.CALIBRATION_KNOBS): a multiplier on the published cluster_scale
# vector for gaussians, t_dist and t_dist_noise, an absolute embed_param for
# circles and moons, and an absolute spiral_band for swiss_rolls.
CALIBRATED_ANCHORS: dict[str, dict[str, dict[str, Any]]] = {
    "gaussians": {
        # multiplier 0.825, ARI 0.984 +/- 0.006
        "easy": {
            "cluster_scale": [3.3000000000000003] * 5,
            "cluster_size_dirichlet_alpha": 0.9,
        },
        # multiplier 1.00625, ARI 0.867 +/- 0.022
        "medium": {
            "cluster_scale": [4.528125] * 5,
            "cluster_size_dirichlet_alpha": 0.5,
        },
        # multiplier 1.00625, ARI 0.784 +/- 0.139
        "hard": {
            "cluster_scale": [4.62875] * 5,
            "cluster_size_dirichlet_alpha": 0.1,
        },
    },
    "t_dist": {
        # multiplier 1.55, ARI 0.938 +/- 0.045
        "easy": {
            "cluster_scale": [5.425, 1.55, 1.55, 1.55, 1.55],
            "cluster_size_dirichlet_alpha": 0.9,
            "t_df": 5,
        },
        # multiplier 1.1875, ARI 0.831 +/- 0.114
        "medium": {
            "cluster_scale": [3.5625, 1.78125, 1.1875, 1.1875, 1.1875],
            "cluster_size_dirichlet_alpha": 0.3,
            "t_df": 3,
        },
        # multiplier 1.00625, ARI 0.766 +/- 0.114
        "hard": {
            "cluster_scale": [4.025, 3.0187500000000003, 2.0125, 1.00625, 1.00625],
            "cluster_size_dirichlet_alpha": 0.1,
            "t_df": 3,
        },
    },
    "t_dist_noise": {
        # multiplier 0.825, ARI 0.995 +/- 0.002
        "easy": {
            "cluster_scale": [0.8250000000000001] * 5,
            "cluster_size_dirichlet_alpha": 0.9,
            "t_df": 5,
            "corr_strength": 0.1,
            "noise_dims": 512,
        },
        # multiplier 1.00625, ARI 0.861 +/- 0.034
        "medium": {
            "cluster_scale": [1.1068750000000003, 1.00625, 1.00625, 1.00625, 1.00625],
            "cluster_size_dirichlet_alpha": 0.3,
            "t_df": 4,
            "corr_strength": 0.3,
            "noise_dims": 1280,
        },
        # multiplier 1.00625, ARI 0.747 +/- 0.073
        "hard": {
            "cluster_scale": [1.1068750000000003, 1.00625, 1.00625, 1.00625, 1.00625],
            "cluster_size_dirichlet_alpha": 0.1,
            "t_df": 3,
            "corr_strength": 0.5,
            "noise_dims": 1536,
        },
    },
    "circles": {
        # embed_param 3.0, ARI 0.965 +/- 0.107
        "easy": {
            "cluster_scale": [4.08, 4.08, 3.0, 3.0, 3.0],
            "cluster_size_dirichlet_alpha": 0.9,
            "corr_strength": 0.1,
            "embed_param": 3.0,
        },
        # embed_param 2.5, ARI 0.853 +/- 0.209
        "medium": {
            "cluster_scale": [4.08, 4.08, 3.0, 3.0, 3.0],
            "cluster_size_dirichlet_alpha": 0.61,
            "corr_strength": 0.23,
            "embed_param": 2.5,
        },
        # embed_param 2.5, ARI 0.775 +/- 0.221
        "hard": {
            "cluster_scale": [4.38, 4.08, 4.08, 4.08, 4.08],
            "cluster_size_dirichlet_alpha": 0.35,
            "corr_strength": 0.2,
            "embed_param": 2.5,
        },
    },
    "moons": {
        # embed_param 3.0, ARI 1.000 +/- 0.000
        "easy": {
            "cluster_scale": [5.5, 3.97, 3.97, 3.97, 3.97],
            "cluster_size_dirichlet_alpha": 0.67,
            "corr_strength": 0.39,
            "embed_param": 3.0,
        },
        # embed_param 2.3125, ARI 0.880 +/- 0.153
        "medium": {
            "cluster_scale": [4.8, 4.06, 4.06, 4.06, 4.06],
            "cluster_size_dirichlet_alpha": 0.57,
            "corr_strength": 0.3,
            "embed_param": 2.3125,
        },
        # embed_param 2.5625, ARI 0.721 +/- 0.210
        "hard": {
            "cluster_scale": [4.06, 2.65, 2.65, 2.65, 2.65],
            "cluster_size_dirichlet_alpha": 0.1,
            "corr_strength": 0.16,
            "embed_param": 2.5625,
        },
    },
    # Arm length and twist are fixed in _SHARED. Dirichlet alpha and AR(1)
    # strength keep the published per-level values; embed_param is 5.0 at
    # every level.
    "swiss_rolls": {
        # spiral_band 0.09, ARI 1.000 +/- 0.000
        "easy": {
            "spiral_band": 0.09,
            "cluster_size_dirichlet_alpha": 0.9,
            "corr_strength": 0.1,
            "embed_param": 5.0,
        },
        # spiral_band 0.11, ARI 0.826 +/- 0.154
        "medium": {
            "spiral_band": 0.11,
            "cluster_size_dirichlet_alpha": 0.7,
            "corr_strength": 0.1,
            "embed_param": 5.0,
        },
        # spiral_band 0.115, ARI 0.769 +/- 0.148
        "hard": {
            "spiral_band": 0.115,
            "cluster_size_dirichlet_alpha": 0.5,
            "corr_strength": 0.3,
            "embed_param": 5.0,
        },
    },
    # Not calibrated: TARGET_ARI_BANDS is keyed on difficulty, and these
    # sweep n and p.
    "gaussians_dimensionality": PUBLISHED_ANCHORS["gaussians_dimensionality"],
    "gaussians_samples": PUBLISHED_ANCHORS["gaussians_samples"],
}

# Sample size of every scenario that does not sweep n. Generalizability
# clusters the held-out (1 - rho) * n rows on their own, and at n = 1500
# spectral clustering of circles, moons and swiss rolls lost enough agreement
# with the full-data fit on those rows to move generalizability's selection.
BENCHMARK_N_TOTAL: int = 3000

# Simulation settings shared by every level of a scenario. A scaling
# scenario's swept parameter comes from the axis value (Scenario.sim_kwargs)
# and must not appear here.
_SHARED: dict[str, dict[str, Any]] = {
    "gaussians": {"n_total": BENCHMARK_N_TOTAL, "p": 50},
    "t_dist": {"n_total": BENCHMARK_N_TOTAL, "p": 50, "distribution": "t"},
    "t_dist_noise": {
        "n_total": BENCHMARK_N_TOTAL,
        "p": 50,
        "corr_type": "ar1",
        "distribution": "t",
    },
    "circles": {
        "distribution": "circles",
        "nonlinear": True,
        "n_total": BENCHMARK_N_TOTAL,
        "p": 50,
        "corr_type": "ar1",
        "embed_dim": 64,
    },
    "moons": {
        "distribution": "moons",
        "nonlinear": True,
        "n_total": BENCHMARK_N_TOTAL,
        "p": 50,
        "corr_type": "ar1",
        "embed_dim": 64,
    },
    # Five interleaved arms of one spiral in a shared plane, so the clusters
    # are non-convex. The arms are half a turn long with no twist; the band
    # sets the difficulty (see _calibrate._SPIRAL_KNOB).
    "swiss_rolls": {
        "distribution": "interleaved_spirals",
        "nonlinear": True,
        "n_total": BENCHMARK_N_TOTAL,
        "p": 50,
        "corr_type": "ar1",
        "embed_dim": 64,
        "spiral_turns": 0.5,
        "spiral_twist": 0.0,
    },
    # Axis is "p".
    "gaussians_dimensionality": {
        "corr_type": "ar1",
        "corr_strength": 0.5,
        "cluster_size_dirichlet_alpha": 0.5,
        "n_total": BENCHMARK_N_TOTAL,
        "embed_dim": 64,
    },
    # Axis is "n_total".
    "gaussians_samples": {
        "corr_type": "ar1",
        "corr_strength": 0.5,
        "cluster_size_dirichlet_alpha": 0.5,
        "p": 50,
        "embed_dim": 64,
    },
}

# Every run manifest records ACTIVE_ANCHOR_SET_NAME.
ACTIVE_ANCHORS: dict[str, dict[str, dict[str, Any]]] = CALIBRATED_ANCHORS
ACTIVE_ANCHOR_SET_NAME: str = "CALIBRATED_ANCHORS"

_ESTIMATORS: dict[str, str] = {
    "gaussians": "kmeans",
    "t_dist": "agglomerative",
    "t_dist_noise": "agglomerative",
    "circles": "spectral",
    "moons": "spectral",
    "swiss_rolls": "spectral",
    "gaussians_dimensionality": "kmeans",
    "gaussians_samples": "kmeans",
}

# One forest size for every scenario.
_N_TREES: dict[str, int] = dict.fromkeys(_ESTIMATORS, 500)

# The generalizability classifier has to be able to draw the boundaries the
# clustering draws. k-means partitions are linear, so shrinkage LDA scores
# the three k-means scenarios and the forest scores the rest; n_trees is
# unused where LDA is used. On the Gaussian scenarios the forest reproduced
# held-out k-means partitions much worse than LDA, most of all at p = 1000.
_CLASSIFIERS: dict[str, str] = {
    **dict.fromkeys(_ESTIMATORS, "random_forest"),
    "gaussians": "lda",
    "gaussians_dimensionality": "lda",
    "gaussians_samples": "lda",
}

_AXES: dict[str, Axis] = {
    "gaussians": DIFFICULTY_AXIS,
    "t_dist": DIFFICULTY_AXIS,
    "t_dist_noise": DIFFICULTY_AXIS,
    "circles": DIFFICULTY_AXIS,
    "moons": DIFFICULTY_AXIS,
    "swiss_rolls": DIFFICULTY_AXIS,
    # "Dimensionality" is the ambient feature dimension p, not embed_dim.
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
        classifier=_CLASSIFIERS[name],
    )
    for name in _ESTIMATORS
}

# Scenarios that also run one CARVE fit per mode, to time the modes for the
# runtime figure. The extra fits more than double a cell's cost.
TIMED_SCENARIOS: frozenset[str] = frozenset(
    {"gaussians_samples", "gaussians_dimensionality"}
)

# Seeds derive as benchmark_seed = seed + axis_idx * 10000 + random_state.
PUBLISHED_RANDOM_STATE: int = 42

# =============================================================================
# Ablations
# =============================================================================
# One CARVE fit draws its subsamples from seeds in [random_state,
# random_state + 3 * n_resamples): P_1 at random_state + b, P_2 at
# random_state + b + B and the P_test pipeline at random_state + b + 2B.
# Replicate fits of one dataset are spaced by this much so no two share a
# subsample. The spacing is larger than the largest benchmark seed offset
# (two axis steps of 10,000 plus the dataset index), so the seed windows of
# different replicate indices never meet. Within one replicate index, the
# windows of different datasets may overlap; they draw from different data.
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

# Medium gaussians only. Both arms run ten seeds per dataset, which puts the
# seed noise in a twenty-dataset mean near 0.002 ARI. The rho arm's twenty
# datasets are the main benchmark's medium gaussians datasets, so its default
# column (rho_default, b_default, replicate 0) reproduces those fits.
ABLATION_SCALES: dict[str, AblationScale] = {
    "publication": AblationScale(
        rho_arm=ArmScale(
            difficulties=("medium",),
            datasets=tuple(range(20)),
            replicates=10,
            study_replicates=0,
        ),
        b_arm=ArmScale(
            difficulties=("medium",),
            datasets=tuple(range(10)),
            replicates=10,
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
