"""Regenerate the SNR anchors from the documented target ARI bands.

This is a replacement, not a reproduction. The notebook that produced the
published anchors was lost, and the anchors it produced move four parameters
at once and non-monotonically across difficulty levels, which is not the
signature of a single ordered sweep. The original search space is therefore
not recoverable.

What this module does instead is documented and reproducible: hold every
anchor parameter fixed except one scale parameter, and bisect it until a base
estimator informed with the true k_star reaches a mean ARI inside the target
band over the calibration seeds. The manuscript's S3 Text is updated to
describe this procedure.

Run offline, not as part of a benchmark run. Compare its output against
_registry.PUBLISHED_ANCHORS before adopting anything.
"""

from dataclasses import replace
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ._estimators import build_estimator
from ._registry import PUBLISHED_RANDOM_STATE
from ._simulate import simulate
from ._types import Scenario

TARGET_ARI_BANDS: dict[str, tuple[float, float]] = {
    "easy": (0.9, 1.0),
    "medium": (0.8, 0.9),
    "hard": (0.7, 0.8),
}

# The single calibrated parameter and the interval it is searched over.
# Larger cluster_scale means more diffuse clusters and a lower ARI, so the
# objective is monotonically decreasing in it, which is what makes bisection
# valid.
CALIBRATION_SPACE: dict[str, tuple[float, float]] = {
    "cluster_scale": (0.1, 12.0),
}


def mean_oracle_ari(
    scenario: Scenario,
    axis_label: str,
    overrides: dict[str, Any],
    *,
    n_seeds: int = 20,
    random_state: int = PUBLISHED_RANDOM_STATE,
) -> float:
    """Mean ARI of the oracle estimator at k_star over the calibration seeds.

    Calibration uses the same seed derivation as the benchmark, so the
    calibration and benchmark datasets coincide, as the manuscript states.
    That claim holds only at random_state=PUBLISHED_RANDOM_STATE (42), which
    is why that is the default here rather than 0.
    """
    axis_idx = list(scenario.axis.labels).index(axis_label)
    axis_value = scenario.axis.values[axis_idx]

    anchors = {k: dict(v) for k, v in scenario.anchors.items()}
    anchors[axis_label].update(overrides)
    probe = replace(scenario, anchors=anchors)

    aris = []
    for seed in range(n_seeds):
        benchmark_seed = seed + (axis_idx * 10000) + random_state
        X, y = simulate(
            probe, axis_value=axis_value, axis_label=axis_label, seed=benchmark_seed
        )
        estimator = build_estimator(
            probe.estimator, n_clusters=probe.k_star, random_state=benchmark_seed
        )
        aris.append(adjusted_rand_score(y, estimator.fit_predict(X)))
    return float(np.mean(aris))


def calibrate_anchor(
    scenario: Scenario,
    axis_label: str,
    *,
    target: tuple[float, float],
    n_seeds: int = 20,
    random_state: int = PUBLISHED_RANDOM_STATE,
    max_iter: int = 25,
) -> dict[str, Any]:
    """Bisect the scale parameter until mean oracle ARI lands in the band."""
    low_target, high_target = target
    if not low_target < high_target:
        raise ValueError(f"Invalid target band {target!r}: low must be below high.")

    midpoint_target = 0.5 * (low_target + high_target)
    low, high = CALIBRATION_SPACE["cluster_scale"]

    best_scale = 0.5 * (low + high)
    best_ari = float("nan")

    for _ in range(max_iter):
        scale = 0.5 * (low + high)
        ari = mean_oracle_ari(
            scenario,
            axis_label,
            {"cluster_scale": scale},
            n_seeds=n_seeds,
            random_state=random_state,
        )
        best_scale, best_ari = scale, ari

        if low_target <= ari <= high_target:
            break
        # ARI decreases as cluster_scale grows.
        if ari > midpoint_target:
            low = scale
        else:
            high = scale

    return {
        "anchor": {"cluster_scale": best_scale},
        "achieved_ari": best_ari,
        "target": target,
        "in_band": bool(low_target <= best_ari <= high_target),
    }


def calibrate_scenario(
    scenario: Scenario,
    *,
    n_seeds: int = 20,
    random_state: int = PUBLISHED_RANDOM_STATE,
    max_iter: int = 25,
) -> dict[str, dict[str, Any]]:
    """Calibrate every anchor of a scenario against its target band."""
    return {
        label: calibrate_anchor(
            scenario,
            label,
            target=TARGET_ARI_BANDS[label],
            n_seeds=n_seeds,
            random_state=random_state,
            max_iter=max_iter,
        )
        for label in scenario.axis.labels
        if label in TARGET_ARI_BANDS
    }
