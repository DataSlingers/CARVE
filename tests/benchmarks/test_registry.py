"""Tests for the experiment registry."""

import dataclasses

import pytest

from benchmarks._registry import (
    ABLATIONS,
    ACTIVE_ANCHOR_SET_NAME,
    ACTIVE_ANCHORS,
    BASELINE_METRIC,
    CALIBRATED_ANCHORS,
    CARVE_METRICS_ALL,
    CVI_METRICS,
    DIFFICULTY_AXIS,
    GENERALIZABILITY_METRICS,
    PUBLISHED_ANCHORS,
    PUBLISHED_RANDOM_STATE,
    REPLICATE_SEED_SPACING,
    SCALING_AXES,
    SCALING_TABLE_ROW_GROUPS,
    SCENARIOS,
    SIMILARITY_SEED_OFFSET,
    TABLE_ROW_GROUPS,
    metric_measure,
    metric_rule,
    package_defaults,
    validate_ablation,
)
from benchmarks._types import AblationScale, ArmScale

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

    def test_active_anchors_are_the_calibrated_set(self):
        """Every difficulty anchor is regenerated through _calibrate.py.

        The published anchors missed their own documented bands on circles
        and moons, and the eigensolver fix moved the three RFF scenarios
        further. A half-published, half-calibrated set would leave "easy"
        meaning a different ARI on different scenarios, which is the
        inconsistency this removes.
        """
        assert ACTIVE_ANCHORS is CALIBRATED_ANCHORS
        assert ACTIVE_ANCHOR_SET_NAME == "CALIBRATED_ANCHORS"

    def test_published_anchors_are_kept_for_comparison(self):
        """PUBLISHED_ANCHORS stays in the file so the calibration can be
        compared against what the manuscript reported, and reverted to."""
        assert set(PUBLISHED_ANCHORS) == set(SCENARIOS)
        assert PUBLISHED_ANCHORS is not CALIBRATED_ANCHORS

    @pytest.mark.parametrize("name", DIFFICULTY_SCENARIOS)
    def test_calibrated_anchors_cover_every_difficulty_label(self, name):
        assert set(CALIBRATED_ANCHORS[name]) == set(DIFFICULTY_AXIS.labels)

    @pytest.mark.parametrize("name", SCALING_SCENARIOS)
    def test_the_scaling_scenarios_keep_their_published_anchors(self, name):
        """TARGET_ARI_BANDS is keyed on easy/medium/hard, so the two scaling
        scenarios have no band to calibrate against and are carried over."""
        assert CALIBRATED_ANCHORS[name] == PUBLISHED_ANCHORS[name]


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

    @pytest.mark.parametrize("name", ("circles", "moons", "swiss_rolls"))
    def test_the_rff_family_shares_one_estimator(self, name):
        """All three RFF-embedded scenarios run self-tuning spectral.

        swiss_rolls ran Ward because spectral lost to it by 0.22 ARI at the
        easy anchor -- which was the which='SM' eigensolver failing to
        converge, not a property of the data. With the shift-invert solver,
        oracle ARI at k*=5 over 20 datasets is 1.000/0.900/0.834 for spectral
        against 0.982/0.860/0.752 for Ward, and spectral is 30x faster.
        """
        assert SCENARIOS[name].estimator.name == "spectral"

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

    @pytest.mark.parametrize("name", DIFFICULTY_SCENARIOS + SCALING_SCENARIOS)
    def test_n_trees_is_500_everywhere(self, name):
        """One forest size for every scenario.

        The published benchmark ran 500 trees on circles, moons and swiss
        rolls and 100 elsewhere, decided by which notebook cell happened to
        pass the argument rather than by anything about the data. 500 is the
        better generalizability estimate and the cost is linear in a term
        that is not the bottleneck.
        """
        assert SCENARIOS[name].n_trees == 500

    def test_swiss_rolls_does_not_restate_the_simulator_default(self):
        """center_box=3.0 is simulate_clusters's own default.

        Restating it made the key read as a deliberate choice for this
        scenario when it is not one, and it was the only scenario that set
        it.
        """
        assert "center_box" not in SCENARIOS["swiss_rolls"].shared


class TestPublishedRandomState:
    def test_matches_the_notebooks_random_seed(self):
        assert PUBLISHED_RANDOM_STATE == 42


class TestAblationRegistry:
    def test_rho_b_is_registered(self):
        assert "rho_b" in ABLATIONS
        assert ABLATIONS["rho_b"].name == "rho_b"

    def test_grids_are_the_spec_values(self):
        ablation = ABLATIONS["rho_b"]
        assert ablation.rho_grid == (0.2, 0.3, 0.4, 0.5, 0.618, 0.7, 0.8, 0.9)
        assert ablation.b_grid == (10, 25, 50, 100, 200)

    def test_defaults_are_read_from_carve(self):
        from dataclasses import fields

        from carve import CARVE

        defaults = {f.name: f.default for f in fields(CARVE)}
        assert package_defaults() == (
            defaults["subsample_ratio"],
            defaults["n_resamples"],
        )
        assert ABLATIONS["rho_b"].rho_default == defaults["subsample_ratio"]
        assert ABLATIONS["rho_b"].b_default == defaults["n_resamples"]

    def test_runs_the_six_difficulty_scenarios_and_klein(self):
        ablation = ABLATIONS["rho_b"]
        assert set(ablation.scenarios) == {
            "gaussians", "t_dist", "t_dist_noise", "circles", "moons", "swiss_rolls",
        }
        assert ablation.study == "klein"

    def test_publication_scale_matches_the_spec(self):
        scale = ABLATIONS["rho_b"].scales["publication"]
        assert scale.rho_arm == ArmScale(("easy", "medium", "hard"), tuple(range(10)), 1, 10)
        assert scale.b_arm == ArmScale(("medium", "hard"), tuple(range(5)), 3, 10)
        assert scale.similarity_draws == 20
        assert scale.n_total is None
        assert scale.study_scale == "publication"

    def test_dev_scale_is_small_and_overrides_n_total(self):
        scale = ABLATIONS["rho_b"].scales["dev"]
        assert scale.rho_arm == ArmScale(("medium",), (0, 1), 1, 2)
        assert scale.b_arm == ArmScale(("medium",), (0, 1), 2, 2)
        assert scale.similarity_draws == 5
        assert scale.n_total == 500
        assert scale.study_scale == "dev"

    def test_default_scale_is_publication(self):
        assert ABLATIONS["rho_b"].default_scale == "publication"

    def test_rejects_a_scenario_that_is_not_registered(self):
        bad = dataclasses.replace(ABLATIONS["rho_b"], scenarios=("nope",))
        with pytest.raises(ValueError, match="nope"):
            validate_ablation(bad)

    def test_rejects_a_scaling_scenario(self):
        bad = dataclasses.replace(ABLATIONS["rho_b"], scenarios=("gaussians_samples",))
        with pytest.raises(ValueError, match="difficulty"):
            validate_ablation(bad)

    def test_rejects_an_unknown_difficulty(self):
        scale = ABLATIONS["rho_b"].scales["dev"]
        bad_scale = dataclasses.replace(
            scale, rho_arm=dataclasses.replace(scale.rho_arm, difficulties=("brutal",))
        )
        bad = dataclasses.replace(ABLATIONS["rho_b"], scales={"dev": bad_scale}, default_scale="dev")
        with pytest.raises(ValueError, match="brutal"):
            validate_ablation(bad)

    def test_seed_constants_keep_replicates_and_similarity_apart(self):
        # One fit's seeds span 3 * B; bases differ by up to two axis steps
        # plus the largest dataset index. The spacing must clear both.
        ablation = ABLATIONS["rho_b"]
        span = 3 * max(ablation.b_grid)
        assert REPLICATE_SEED_SPACING == 1_000_000
        assert SIMILARITY_SEED_OFFSET == 500_000
        assert SIMILARITY_SEED_OFFSET > span
        assert SIMILARITY_SEED_OFFSET + 20 < REPLICATE_SEED_SPACING - span

    def test_validation_rejects_a_spacing_inside_one_fits_seed_span(self, monkeypatch):
        import benchmarks._registry as registry

        monkeypatch.setattr(registry, "REPLICATE_SEED_SPACING", 100)
        with pytest.raises(ValueError, match="REPLICATE_SEED_SPACING"):
            validate_ablation(ABLATIONS["rho_b"])

    def test_validation_rejects_a_similarity_offset_inside_a_replicate_window(
        self, monkeypatch
    ):
        import benchmarks._registry as registry

        monkeypatch.setattr(registry, "SIMILARITY_SEED_OFFSET", 10)
        with pytest.raises(ValueError, match="SIMILARITY_SEED_OFFSET"):
            validate_ablation(ABLATIONS["rho_b"])


class TestTableRowGroups:
    def test_the_groups_reproduce_the_committed_s2_table(self):
        """Row order and grouping are part of the manuscript's layout.

        Deriving them from a sorted set, which is what table_metrics did,
        means a new metric silently reorders every supplementary table.
        """
        assert TABLE_ROW_GROUPS == (
            ("baseline_oracle",),
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

    def test_no_metric_appears_twice(self):
        flat = [m for group in TABLE_ROW_GROUPS for m in group]
        assert len(flat) == len(set(flat))

    def test_every_group_member_is_a_known_metric(self):
        known = set(CARVE_METRICS_ALL) | set(CVI_METRICS) | {BASELINE_METRIC}
        for group in TABLE_ROW_GROUPS:
            assert set(group) <= known

    def test_the_excluded_metrics_are_absent(self):
        """EXCLUDED_METRICS drops the three ari_average variants, PAC and
        CE, which is why the published tables do not carry them."""
        from benchmarks.tables import EXCLUDED_METRICS

        flat = {m for group in TABLE_ROW_GROUPS for m in group}
        assert flat & EXCLUDED_METRICS == set()


class TestScalingTableRowGroups:
    def test_the_groups_reproduce_the_committed_s8_and_s9_tables(self):
        """S8 and S9 order the classical indices Silhouette, Davies-Bouldin,
        Calinski-Harabasz, Gap; S2 orders them Davies-Bouldin, Silhouette,
        Gap, Calinski-Harabasz."""
        assert SCALING_TABLE_ROW_GROUPS == (
            ("baseline_oracle",),
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

    def test_the_scaling_tables_carry_the_same_rows_as_s2(self):
        """Only the order differs. A metric added to one declaration and not
        the other would silently drop a row from one family of tables."""
        assert [set(g) for g in SCALING_TABLE_ROW_GROUPS] == [
            set(g) for g in TABLE_ROW_GROUPS
        ]
