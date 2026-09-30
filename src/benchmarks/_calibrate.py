"""Regenerate the SNR anchors from the documented target ARI bands.

This is a replacement, not a reproduction. The notebook that produced the
published anchors was lost, and the anchors it produced move four parameters
at once and non-monotonically across difficulty levels, which is not the
signature of a single ordered sweep. The original search space is therefore
not recoverable.

What this module does instead is documented and reproducible: each scenario
declares its own knob in CALIBRATION_KNOBS, with a direction, and calibration
holds every other anchor parameter fixed while it bisects that knob until a
base estimator informed with the true k_star reaches a mean ARI inside the
target band over the calibration seeds. The manuscript's S3 Text is updated
to describe this procedure.

Run offline, not as part of a benchmark run. Compare its output against
_registry.PUBLISHED_ANCHORS before adopting anything.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class CalibrationKnob:
    """The anchor parameter a scenario's difficulty is bisected on.

    parameter names a simulate_clusters keyword, or a tuple of them that the
    one bisected value moves together. multiplier=True scales the
    anchor's published value rather than replacing it, which is what keeps
    the unequal per-cluster cluster_scale vectors the nonlinear anchors
    carry -- [4.38, 4.08, 4.08, 4.08, 4.08] on circles/hard -- from being
    flattened to a scalar by the search.

    increasing says which way ARI moves as the parameter grows. It is
    declared rather than assumed because the two knobs run in opposite
    directions: diffuser clusters score lower, a wider RFF bandwidth scores
    higher.
    """

    parameter: str | tuple[str, ...]
    low: float
    high: float
    increasing: bool
    multiplier: bool

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(
                f"CalibrationKnob({self.parameter!r}): low must be below high, "
                f"got {self.low} and {self.high}."
            )

    @property
    def parameters(self) -> tuple[str, ...]:
        """The knob's simulate_clusters keywords, as a tuple."""
        if isinstance(self.parameter, str):
            return (self.parameter,)
        return tuple(self.parameter)


_SCALE_KNOB = CalibrationKnob(
    parameter="cluster_scale",
    low=0.1,
    high=3.0,
    increasing=False,
    multiplier=True,
)

# circles and moons do not respond to cluster_scale at all: measured over 20
# datasets with the fixed eigensolver, multiplying it by 3 leaves oracle ARI
# at 1.000 +/- 0.000 at every anchor. corr_strength does nothing either
# (1.000 at 0.23 through 0.9). embed_param, the random Fourier feature
# bandwidth, is the knob, and it behaves as a threshold: ARI is near zero
# below 2, exactly 1.000 above 4, and all three target bands fall between
# about 2.7 and 3.5. The interval is bounded to that transition so the
# bisection never wanders into either flat region, where it has no gradient
# to follow.
_BANDWIDTH_KNOB = CalibrationKnob(
    parameter="embed_param",
    low=2.0,
    high=4.0,
    increasing=True,
    multiplier=False,
)

# swiss_rolls draws interleaved spiral arms in one shared plane, so there is
# no per-cluster scale to multiply. Arm length, arm thickness and the towel
# twist each lower the oracle ARI, and the knob multiplies all three together
# from the reference in _registry.SPIRAL_REFERENCE (0.5 turns, band 0.05,
# twist 0.2). Measured with spectral clustering over 20 datasets at band 0.05
# and 0.5 turns: twist 0.1 scores 0.92-0.99, 0.2 scores 0.79-0.90, 0.25
# scores 0.68-0.84, and a half-turn twist about 0.5. The interval spans that
# range from arms too short to interleave to arms spectral cannot follow.
_SPIRAL_KNOB = CalibrationKnob(
    parameter=("spiral_turns", "spiral_band", "spiral_twist"),
    low=0.5,
    high=1.5,
    increasing=False,
    multiplier=True,
)

CALIBRATION_KNOBS: dict[str, CalibrationKnob] = {
    "gaussians": _SCALE_KNOB,
    "t_dist": _SCALE_KNOB,
    "t_dist_noise": _SCALE_KNOB,
    "swiss_rolls": _SPIRAL_KNOB,
    "circles": _BANDWIDTH_KNOB,
    "moons": _BANDWIDTH_KNOB,
}


def knob_for(scenario_name: str) -> CalibrationKnob:
    """Return the calibration knob declared for a scenario.

    The two scaling scenarios have no knob: TARGET_ARI_BANDS is keyed on
    easy/medium/hard and they sweep n and p instead.
    """
    if scenario_name not in CALIBRATION_KNOBS:
        raise KeyError(
            f"No calibration knob for {scenario_name!r}. Declared scenarios "
            f"are {sorted(CALIBRATION_KNOBS)}."
        )
    return CALIBRATION_KNOBS[scenario_name]


def apply_knob(
    anchor: Mapping[str, Any], knob: CalibrationKnob, value: float
) -> dict[str, Any]:
    """Return the anchor with the knob set to value, leaving the rest alone."""
    out = dict(anchor)
    for name in knob.parameters:
        if not knob.multiplier:
            out[name] = float(value)
            continue
        if name not in anchor:
            raise KeyError(
                f"Anchor has no {name!r} to scale; a multiplier knob needs "
                f"a published value to multiply. Anchor keys: {sorted(anchor)}."
            )
        published = anchor[name]
        if np.isscalar(published):
            out[name] = float(published) * float(value)
        else:
            out[name] = [float(v) * float(value) for v in published]
    return out


def oracle_ari_stats(
    scenario: Scenario,
    axis_label: str,
    overrides: dict[str, Any],
    *,
    n_seeds: int = 20,
    random_state: int = PUBLISHED_RANDOM_STATE,
) -> tuple[float, float]:
    """Mean and standard deviation of the oracle's ARI over the calibration seeds.

    Calibration uses the same seed derivation as the benchmark, so the
    calibration and benchmark datasets coincide, as the manuscript states.
    That claim holds only at random_state=PUBLISHED_RANDOM_STATE (42), which
    is why that is the default here rather than 0.

    The spread is returned as well as the mean because the RFF bandwidth's
    transition band is wide in spread even where the mean is on target --
    sd 0.10 to 0.23, individual datasets from 0.39 to 1.000 at one setting --
    so an anchor that reports only its mean hides how much it varies.
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
    values = np.asarray(aris, dtype=float)
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean()), sd


def mean_oracle_ari(
    scenario: Scenario,
    axis_label: str,
    overrides: dict[str, Any],
    *,
    n_seeds: int = 20,
    random_state: int = PUBLISHED_RANDOM_STATE,
) -> float:
    """Mean ARI of the oracle estimator at k_star over the calibration seeds."""
    mean, _ = oracle_ari_stats(
        scenario, axis_label, overrides, n_seeds=n_seeds, random_state=random_state
    )
    return mean


def calibrate_anchor(
    scenario: Scenario,
    axis_label: str,
    *,
    target: tuple[float, float],
    knob: CalibrationKnob | None = None,
    n_seeds: int = 20,
    random_state: int = PUBLISHED_RANDOM_STATE,
    max_iter: int = 25,
) -> dict[str, Any]:
    """Bisect the scenario's knob until mean oracle ARI lands in the band.

    knob defaults to the one CALIBRATION_KNOBS declares for this scenario.
    Passing one explicitly is for tests and for probing an alternative.

    The bisection reads its direction off the knob. Assuming ARI falls as
    the parameter grows, which the first implementation did, walks away from
    the band on an increasing knob rather than toward it.
    """
    low_target, high_target = target
    if not low_target < high_target:
        raise ValueError(f"Invalid target band {target!r}: low must be below high.")

    knob = knob_for(scenario.name) if knob is None else knob
    midpoint_target = 0.5 * (low_target + high_target)
    low, high = knob.low, knob.high

    published = scenario.anchors[axis_label]
    best_value = 0.5 * (low + high)
    best_anchor = apply_knob(published, knob, best_value)
    best_ari = float("nan")
    best_sd = float("nan")

    for _ in range(max_iter):
        value = 0.5 * (low + high)
        anchor = apply_knob(published, knob, value)
        ari, sd = oracle_ari_stats(
            scenario,
            axis_label,
            anchor,
            n_seeds=n_seeds,
            random_state=random_state,
        )
        best_value, best_anchor, best_ari, best_sd = value, anchor, ari, sd

        if low_target <= ari <= high_target:
            break
        # Move toward the band. On a decreasing knob a too-high ARI means
        # the parameter must grow; on an increasing one it means the
        # opposite.
        too_high = ari > midpoint_target
        if too_high != knob.increasing:
            low = value
        else:
            high = value

    return {
        "anchor": best_anchor,
        "knob_value": float(best_value),
        "achieved_ari": best_ari,
        "achieved_sd": best_sd,
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
    """Calibrate every anchor of a scenario against its target band.

    Uses the knob CALIBRATION_KNOBS declares for the scenario. Run this with
    the scenario's final estimator and the fixed eigensolver in place;
    calibrating against a solver that will not be used calibrates the wrong
    thing.
    """
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
