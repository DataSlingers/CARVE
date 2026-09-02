"""Tests for the experiment registry."""

import pytest

from benchmarks._registry import (
    ACTIVE_ANCHOR_SET_NAME,
    ACTIVE_ANCHORS,
    CARVE_METRICS_ALL,
    CVI_METRICS,
    DIFFICULTY_AXIS,
    GENERALIZABILITY_METRICS,
    PUBLISHED_ANCHORS,
    PUBLISHED_RANDOM_STATE,
    SCALING_AXES,
    SCENARIOS,
    metric_measure,
    metric_rule,
)

DIFFICULTY_SCENARIOS = (
    "gaussians",
    "t_dist",
    "t_dist_noise",
    "circles",
    "moons",
    "swiss_rolls",
)
SCALING_SCENARIOS = ("gaussians_dimensionality", "gaussians_samples")


class TestMetricNames:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("ari_stability", "max"),
            ("ari_stability_1se", "1se"),
            ("ari_stability_quant", "quantile"),
        ],
    )
    def test_rule_is_parsed_from_the_suffix(self, name, expected):
        assert metric_rule(name) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("ari_stability", "ari_stability"),
            ("ari_stability_1se", "ari_stability"),
            ("ari_stability_quant", "ari_stability"),
        ],
    )
    def test_measure_strips_the_suffix(self, name, expected):
        assert metric_measure(name) == expected

    def test_carve_metrics_are_unique_and_sorted(self):
        assert list(CARVE_METRICS_ALL) == sorted(set(CARVE_METRICS_ALL))

    def test_cvi_metrics_are_the_four_classical_indices(self):
        assert set(CVI_METRICS) == {
            "silhouette",
            "gap",
            "davies_bouldin",
            "calinski_harabasz",
        }

    def test_generalizability_metrics_are_a_subset_of_all_carve_metrics(self):
        assert GENERALIZABILITY_METRICS <= set(CARVE_METRICS_ALL)

    def test_generalizability_metrics_are_exactly_those_named_generalizability(self):
        assert GENERALIZABILITY_METRICS == {
            m for m in CARVE_METRICS_ALL if "generalizability" in m
        }


class TestAxes:
    def test_difficulty_axis_matches_the_published_labels(self):
        assert DIFFICULTY_AXIS.labels == ("easy", "medium", "hard")
        assert DIFFICULTY_AXIS.values == (0, 1, 2)
        assert DIFFICULTY_AXIS.name == "difficulty_level"

    def test_scaling_axes_use_the_published_stage_labels(self):
        for axis in SCALING_AXES.values():
            assert axis.labels == ("start", "middle", "end")

    def test_scaling_ranges_match_the_published_config(self):
        assert SCALING_AXES["n_total"].values == (1000, 5500, 10000)
        assert SCALING_AXES["embed_dim"].values == (10, 255, 500)

    def test_scaling_axes_include_the_p_axis_even_though_it_is_the_only_one_swept(self):
        # SCALING_RANGES defines three axes; the registry keeps all three
        # available even though only p and n_total are actually swept by a
        # registered scenario.
        assert SCALING_AXES["p"].values == (50, 525, 1000)

    def test_moons_has_no_scaling_scenario(self):
        assert not any("moons" in name for name in SCALING_SCENARIOS)
        assert not any(
            name.startswith("moons_") for name in SCENARIOS if name not in DIFFICULTY_SCENARIOS
        )


class TestScalingScenarioAxes:
    """Controller correction 1: dimensionality sweeps p, not embed_dim."""

    def test_gaussians_dimensionality_sweeps_p(self):
        axis = SCENARIOS["gaussians_dimensionality"].axis
        assert axis.name == "p"
        assert axis.values == (50, 525, 1000)

    def test_gaussians_samples_sweeps_n_total(self):
        axis = SCENARIOS["gaussians_samples"].axis
        assert axis.name == "n_total"
        assert axis.values == (1000, 5500, 10000)

    @pytest.mark.parametrize(
        ("name", "swept_key", "fixed"),
        [
            ("gaussians_dimensionality", "p", {"n_total": 1500, "embed_dim": 64}),
            ("gaussians_samples", "n_total", {"p": 50, "embed_dim": 64}),
        ],
    )
    def test_sim_kwargs_carries_all_three_scaling_constants(self, name, swept_key, fixed):
        scenario = SCENARIOS[name]
        for _, value, label in scenario.axis:
            kwargs = scenario.sim_kwargs(value, label)
            assert kwargs["n_total"] is not None
            assert kwargs["p"] is not None
            assert kwargs["embed_dim"] is not None
            assert kwargs[swept_key] == value
            for key, expected in fixed.items():
                assert kwargs[key] == expected


class TestAnchors:
    def test_published_anchors_are_retained_for_every_scenario(self):
        assert set(PUBLISHED_ANCHORS) == set(SCENARIOS)

    def test_active_anchors_are_named_so_provenance_is_unambiguous(self):
        assert ACTIVE_ANCHOR_SET_NAME in {"PUBLISHED_ANCHORS", "CALIBRATED_ANCHORS"}

    def test_active_anchors_default_to_the_published_set(self):
        assert ACTIVE_ANCHORS is PUBLISHED_ANCHORS
        assert ACTIVE_ANCHOR_SET_NAME == "PUBLISHED_ANCHORS"


class TestScenarios:
    def test_every_expected_scenario_is_registered(self):
        assert set(SCENARIOS) == set(DIFFICULTY_SCENARIOS) | set(SCALING_SCENARIOS)

    @pytest.mark.parametrize("name", DIFFICULTY_SCENARIOS + SCALING_SCENARIOS)
    def test_scenario_name_matches_its_key(self, name):
        assert SCENARIOS[name].name == name

    @pytest.mark.parametrize("name", DIFFICULTY_SCENARIOS + SCALING_SCENARIOS)
    def test_every_scenario_constructs_and_validates(self, name):
        scenario = SCENARIOS[name]
        assert scenario.k_star == 5
        assert scenario.candidate_k == (3, 4, 5, 6, 7)
        assert scenario.n_seeds == 20

    def test_swiss_rolls_uses_the_estimator_the_benchmark_actually_ran(self):
        assert SCENARIOS["swiss_rolls"].estimator.name == "agglomerative"

    def test_gaussians_uses_kmeans(self):
        assert SCENARIOS["gaussians"].estimator.name == "kmeans"

    def test_t_dist_uses_agglomerative(self):
        assert SCENARIOS["t_dist"].estimator.name == "agglomerative"

    @pytest.mark.parametrize("name", DIFFICULTY_SCENARIOS)
    def test_difficulty_scenarios_sweep_the_difficulty_axis(self, name):
        assert SCENARIOS[name].axis.name == "difficulty_level"

    def test_sim_kwargs_resolve_for_every_scenario_and_label(self):
        for scenario in SCENARIOS.values():
            for _, value, label in scenario.axis:
                kwargs = scenario.sim_kwargs(value, label)
                assert "k" not in kwargs

    @pytest.mark.parametrize("name", ("circles", "moons", "swiss_rolls"))
    def test_n_trees_is_500_for_the_scenarios_the_notebook_ran_with_500_trees(self, name):
        assert SCENARIOS[name].n_trees == 500

    @pytest.mark.parametrize(
        "name",
        ("gaussians", "t_dist", "t_dist_noise", "gaussians_dimensionality", "gaussians_samples"),
    )
    def test_n_trees_defaults_to_100_for_the_remaining_scenarios(self, name):
        assert SCENARIOS[name].n_trees == 100


class TestPublishedRandomState:
    def test_matches_the_notebooks_random_seed(self):
        assert PUBLISHED_RANDOM_STATE == 42
