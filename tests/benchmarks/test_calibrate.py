"""Tests for the SNR calibration replacement."""

import pytest

from benchmarks._calibrate import (
    CALIBRATION_KNOBS,
    TARGET_ARI_BANDS,
    CalibrationKnob,
    apply_knob,
    calibrate_anchor,
    knob_for,
    mean_oracle_ari,
    oracle_ari_stats,
)
from benchmarks._registry import PUBLISHED_RANDOM_STATE
from benchmarks._types import Axis, EstimatorSpec, Scenario

# tiny_scenario's name ("tiny") is not registered in CALIBRATION_KNOBS, and
# knob_for raises KeyError for unregistered names by design (see
# TestKnobs.test_knob_for_rejects_an_unregistered_scenario). Every test that
# calls calibrate_anchor on tiny_scenario therefore passes this knob
# explicitly rather than relying on the registry default.
_TINY_KNOB = CalibrationKnob(
    parameter="cluster_scale",
    low=0.1,
    high=3.0,
    increasing=False,
    multiplier=True,
)


@pytest.fixture
def tiny_scenario():
    return Scenario(
        name="tiny",
        axis=Axis(name="difficulty_level", values=(0,), labels=("easy",)),
        anchors={"easy": {"cluster_scale": 1.0}},
        shared={"n_total": 120, "p": 4, "distribution": "gaussian"},
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=(3, 4, 5),
        n_seeds=3,
    )


@pytest.fixture
def rising_scenario():
    """A nonlinear scenario whose ARI rises with embed_param.

    The same shape as circles: below the bandwidth threshold the embedding
    destroys the structure, above it the problem is trivial.
    """
    return Scenario(
        name="rising",
        axis=Axis(name="difficulty_level", values=(0,), labels=("easy",)),
        anchors={"easy": {"embed_param": 1.0}},
        shared={
            "n_total": 300,
            "p": 8,
            "distribution": "circles",
            "nonlinear": True,
            "embed_dim": 32,
            "corr_type": "ar1",
        },
        estimator=EstimatorSpec(name="spectral"),
        candidate_k=(3, 4, 5),
        n_seeds=3,
    )


class TestTargets:
    def test_bands_match_the_manuscript(self):
        assert TARGET_ARI_BANDS == {
            "easy": (0.9, 1.0),
            "medium": (0.8, 0.9),
            "hard": (0.7, 0.8),
        }


class TestKnobs:
    def test_every_difficulty_scenario_declares_a_knob(self):
        from benchmarks._registry import DIFFICULTY_AXIS, SCENARIOS

        difficulty = [
            name
            for name, s in SCENARIOS.items()
            if s.axis.name == DIFFICULTY_AXIS.name
        ]
        assert set(CALIBRATION_KNOBS) == set(difficulty)

    def test_the_rff_scenarios_calibrate_on_the_bandwidth(self):
        """cluster_scale does not move circles or moons at all.

        Measured over 20 datasets with the fixed solver: multiplying
        cluster_scale by 3 leaves oracle ARI at 1.000 +/- 0.000. embed_param
        is the knob, and all three target bands live between about 2.7 and
        3.5.
        """
        for name in ("circles", "moons"):
            knob = knob_for(name)
            assert knob.parameter == "embed_param"
            assert knob.increasing is True
            assert knob.multiplier is False
            assert (knob.low, knob.high) == (2.0, 4.0)

    def test_the_scale_driven_scenarios_calibrate_on_a_multiplier(self):
        for name in ("gaussians", "t_dist", "t_dist_noise", "swiss_rolls"):
            knob = knob_for(name)
            assert knob.parameter == "cluster_scale"
            assert knob.increasing is False
            assert knob.multiplier is True

    def test_knob_for_rejects_an_unregistered_scenario(self):
        with pytest.raises(KeyError, match="gaussians_samples"):
            knob_for("gaussians_samples")


class TestApplyKnob:
    def test_a_multiplier_scales_the_published_vector(self):
        """The nonlinear anchors carry unequal per-cluster scales.

        Replacing the vector with a scalar, which is what the first
        implementation did, flattens that structure. Scaling preserves it.
        """
        knob = CalibrationKnob(
            parameter="cluster_scale",
            low=0.1,
            high=3.0,
            increasing=False,
            multiplier=True,
        )
        anchor = {"cluster_scale": [4.0, 2.0, 2.0], "corr_strength": 0.3}
        out = apply_knob(anchor, knob, 1.5)
        assert out["cluster_scale"] == [6.0, 3.0, 3.0]
        assert out["corr_strength"] == 0.3

    def test_a_multiplier_handles_a_scalar_published_value(self):
        knob = CalibrationKnob(
            parameter="cluster_scale",
            low=0.1,
            high=3.0,
            increasing=False,
            multiplier=True,
        )
        assert apply_knob({"cluster_scale": 2.0}, knob, 0.5)["cluster_scale"] == 1.0

    def test_an_absolute_knob_replaces_the_value(self):
        knob = CalibrationKnob(
            parameter="embed_param",
            low=2.0,
            high=4.0,
            increasing=True,
            multiplier=False,
        )
        out = apply_knob({"embed_param": 7.3, "corr_strength": 0.2}, knob, 3.1)
        assert out["embed_param"] == 3.1
        assert out["corr_strength"] == 0.2

    def test_a_multiplier_needs_the_parameter_to_be_present(self):
        knob = CalibrationKnob(
            parameter="cluster_scale",
            low=0.1,
            high=3.0,
            increasing=False,
            multiplier=True,
        )
        with pytest.raises(KeyError, match="cluster_scale"):
            apply_knob({"corr_strength": 0.3}, knob, 1.5)


class TestMeanOracleAri:
    def test_returns_a_value_in_the_ari_range(self, tiny_scenario):
        ari = mean_oracle_ari(
            tiny_scenario, "easy", {"cluster_scale": 0.5}, n_seeds=2, random_state=0
        )
        # ARI is bounded below by -1, not 0; the oracle fit can land anywhere in it.
        assert -1.0 <= ari <= 1.0

    def test_tighter_clusters_score_higher(self, tiny_scenario):
        tight = mean_oracle_ari(
            tiny_scenario, "easy", {"cluster_scale": 0.4}, n_seeds=3, random_state=0
        )
        loose = mean_oracle_ari(
            tiny_scenario, "easy", {"cluster_scale": 4.0}, n_seeds=3, random_state=0
        )
        assert tight > loose

    def test_is_deterministic(self, tiny_scenario):
        kwargs = dict(n_seeds=2, random_state=0)
        first = mean_oracle_ari(tiny_scenario, "easy", {"cluster_scale": 1.0}, **kwargs)
        second = mean_oracle_ari(tiny_scenario, "easy", {"cluster_scale": 1.0}, **kwargs)
        assert first == second


class TestOracleAriStats:
    def test_mean_matches_mean_oracle_ari(self, tiny_scenario):
        kwargs = dict(n_seeds=3, random_state=0)
        mean, _ = oracle_ari_stats(
            tiny_scenario, "easy", {"cluster_scale": 1.0}, **kwargs
        )
        assert mean == mean_oracle_ari(
            tiny_scenario, "easy", {"cluster_scale": 1.0}, **kwargs
        )

    def test_sd_is_nonnegative(self, tiny_scenario):
        _, sd = oracle_ari_stats(
            tiny_scenario, "easy", {"cluster_scale": 1.0}, n_seeds=3, random_state=0
        )
        assert sd >= 0.0


class TestCalibrateAnchor:
    def test_returns_an_anchor_and_the_ari_it_achieved(self, tiny_scenario):
        result = calibrate_anchor(
            tiny_scenario,
            "easy",
            target=(0.9, 1.0),
            knob=_TINY_KNOB,
            n_seeds=2,
            max_iter=8,
        )
        assert "anchor" in result
        assert "achieved_ari" in result
        assert "cluster_scale" in result["anchor"]

    def test_reports_whether_the_band_was_reached(self, tiny_scenario):
        result = calibrate_anchor(
            tiny_scenario,
            "easy",
            target=(0.9, 1.0),
            knob=_TINY_KNOB,
            n_seeds=2,
            max_iter=8,
        )
        assert isinstance(result["in_band"], bool)

    def test_rejects_an_inverted_band(self, tiny_scenario):
        with pytest.raises(ValueError, match="band"):
            calibrate_anchor(
                tiny_scenario,
                "easy",
                target=(1.0, 0.9),
                knob=_TINY_KNOB,
                n_seeds=2,
            )


class TestDirection:
    def test_an_increasing_knob_bisects_the_other_way(self, rising_scenario):
        """A bisection that assumes ARI falls with the knob walks away from
        the band when it rises. calibrate_anchor must read the direction off
        the knob rather than assuming it.

        The interval is narrower than embed_param's full transition
        (0.5-6.0) so the first bisection midpoint (1.75) lands below the
        target band: measured over this scenario's shape, ARI at 1.75 is
        roughly 0.8, and the band is (0.9, 1.0). A midpoint that already
        landed in band would pass even under the wrong direction, on the
        first iteration, and prove nothing.
        """
        knob = CalibrationKnob(
            parameter="embed_param",
            low=0.5,
            high=3.0,
            increasing=True,
            multiplier=False,
        )
        result = calibrate_anchor(
            rising_scenario,
            "easy",
            target=(0.9, 1.0),
            knob=knob,
            n_seeds=3,
            random_state=0,
            max_iter=12,
        )
        assert result["in_band"]


class TestCalibrateAnchorResult:
    def test_reports_the_spread_alongside_the_mean(self, tiny_scenario):
        """The RFF transition band has sd 0.10-0.23 where the mean is on
        target, so an anchor's spread is part of what a reader needs."""
        result = calibrate_anchor(
            tiny_scenario,
            "easy",
            target=(0.9, 1.0),
            knob=_TINY_KNOB,
            n_seeds=3,
            random_state=0,
        )
        assert set(result) == {
            "anchor",
            "knob_value",
            "achieved_ari",
            "achieved_sd",
            "target",
            "in_band",
        }
        assert result["achieved_sd"] >= 0.0
        assert isinstance(result["knob_value"], float)


class TestPublishedRandomStateDefault:
    """The module docstring claims calibration reuses the benchmark's seed
    derivation, which is only true when the default random_state matches the
    published run (PUBLISHED_RANDOM_STATE = 42, not 0)."""

    def test_mean_oracle_ari_defaults_to_the_published_random_state(self):
        import inspect

        default = inspect.signature(mean_oracle_ari).parameters["random_state"].default
        assert default == PUBLISHED_RANDOM_STATE

    def test_calibrate_anchor_defaults_to_the_published_random_state(self):
        import inspect

        default = inspect.signature(calibrate_anchor).parameters["random_state"].default
        assert default == PUBLISHED_RANDOM_STATE

    def test_calibrate_scenario_defaults_to_the_published_random_state(self):
        import inspect

        from benchmarks._calibrate import calibrate_scenario

        default = inspect.signature(calibrate_scenario).parameters[
            "random_state"
        ].default
        assert default == PUBLISHED_RANDOM_STATE
