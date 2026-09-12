"""Tests for the SNR calibration replacement."""

import pytest

from benchmarks._calibrate import (
    CALIBRATION_SPACE,
    TARGET_ARI_BANDS,
    calibrate_anchor,
    mean_oracle_ari,
)
from benchmarks._registry import PUBLISHED_RANDOM_STATE
from benchmarks._types import Axis, EstimatorSpec, Scenario


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


class TestTargets:
    def test_bands_match_the_manuscript(self):
        assert TARGET_ARI_BANDS == {
            "easy": (0.9, 1.0),
            "medium": (0.8, 0.9),
            "hard": (0.7, 0.8),
        }

    def test_the_search_parameter_is_documented(self):
        assert "cluster_scale" in CALIBRATION_SPACE
        low, high = CALIBRATION_SPACE["cluster_scale"]
        assert low < high


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


class TestCalibrateAnchor:
    def test_returns_an_anchor_and_the_ari_it_achieved(self, tiny_scenario):
        result = calibrate_anchor(
            tiny_scenario, "easy", target=(0.9, 1.0), n_seeds=2, max_iter=8
        )
        assert "anchor" in result
        assert "achieved_ari" in result
        assert "cluster_scale" in result["anchor"]

    def test_reports_whether_the_band_was_reached(self, tiny_scenario):
        result = calibrate_anchor(
            tiny_scenario, "easy", target=(0.9, 1.0), n_seeds=2, max_iter=8
        )
        assert isinstance(result["in_band"], bool)

    def test_rejects_an_inverted_band(self, tiny_scenario):
        with pytest.raises(ValueError, match="band"):
            calibrate_anchor(tiny_scenario, "easy", target=(1.0, 0.9), n_seeds=2)


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
