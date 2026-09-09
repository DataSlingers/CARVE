"""Tests for the leaf dataclasses in benchmarks._types.

Named test_benchmarks_types.py rather than test_types.py to keep the basename
unique against tests/test_types.py: pytest's default prepend import mode
aborts the whole suite with an "import file mismatch" error when two test
modules share a basename and neither tests/ nor tests/benchmarks/ is a
package.
"""

import pytest

from benchmarks._types import KNOWN_ESTIMATORS, Axis, EstimatorSpec


class TestAxis:
    def test_iterates_index_value_label_triples(self):
        axis = Axis(name="difficulty", values=(0, 1, 2), labels=("easy", "medium", "hard"))
        assert list(axis) == [(0, 0, "easy"), (1, 1, "medium"), (2, 2, "hard")]

    def test_length_is_the_number_of_points(self):
        axis = Axis(name="n_total", values=(1000, 5500, 10000), labels=("start", "middle", "end"))
        assert len(axis) == 3

    def test_rejects_mismatched_values_and_labels(self):
        with pytest.raises(ValueError, match="same length"):
            Axis(name="p", values=(1, 2, 3), labels=("a", "b"))

    def test_rejects_an_empty_axis(self):
        with pytest.raises(ValueError, match="at least one point"):
            Axis(name="p", values=(), labels=())

    def test_rejects_duplicate_labels(self):
        with pytest.raises(ValueError, match="duplicate"):
            Axis(name="p", values=(1, 2), labels=("a", "a"))

    def test_is_frozen(self):
        axis = Axis(name="p", values=(1,), labels=("a",))
        with pytest.raises(AttributeError):
            axis.name = "q"


class TestEstimatorSpec:
    @pytest.mark.parametrize("name", sorted(KNOWN_ESTIMATORS))
    def test_accepts_every_known_estimator(self, name):
        assert EstimatorSpec(name=name).name == name

    def test_raises_on_an_unknown_name(self):
        with pytest.raises(ValueError, match="aglomerative"):
            EstimatorSpec(name="aglomerative")

    def test_error_lists_the_valid_names(self):
        with pytest.raises(ValueError, match="kmeans"):
            EstimatorSpec(name="nope")


from benchmarks._types import Scenario


def _axis():
    return Axis(name="difficulty_level", values=(0, 1, 2), labels=("easy", "medium", "hard"))


def _scenario(**overrides):
    kwargs = dict(
        name="demo",
        axis=_axis(),
        anchors={
            "easy": {"cluster_scale": 1.0},
            "medium": {"cluster_scale": 2.0},
            "hard": {"cluster_scale": 3.0},
        },
        shared={"n_total": 120, "p": 4, "distribution": "gaussian"},
        estimator=EstimatorSpec(name="kmeans"),
    )
    kwargs.update(overrides)
    return Scenario(**kwargs)


class TestScenario:
    def test_accepts_a_valid_scenario(self):
        assert _scenario().name == "demo"

    def test_defaults_match_the_published_benchmark(self):
        s = _scenario()
        assert s.k_star == 5
        assert s.candidate_k == (3, 4, 5, 6, 7)
        assert s.n_seeds == 20

    def test_rejects_an_anchor_key_the_simulator_does_not_accept(self):
        with pytest.raises(ValueError, match="cluster_scal"):
            _scenario(anchors={
                "easy": {"cluster_scal": 1.0},
                "medium": {"cluster_scale": 2.0},
                "hard": {"cluster_scale": 3.0},
            })

    def test_rejects_a_shared_key_the_simulator_does_not_accept(self):
        with pytest.raises(ValueError, match="n_totl"):
            _scenario(shared={"n_totl": 120})

    def test_rejects_a_key_present_in_both_anchors_and_shared(self):
        with pytest.raises(ValueError, match="both"):
            _scenario(shared={"n_total": 120, "cluster_scale": 9.0})

    def test_requires_an_anchor_for_every_axis_label(self):
        with pytest.raises(ValueError, match="hard"):
            _scenario(anchors={"easy": {"cluster_scale": 1.0}, "medium": {"cluster_scale": 2.0}})

    def test_rejects_a_k_star_not_in_candidate_k(self):
        with pytest.raises(ValueError, match="k_star"):
            _scenario(k_star=99)

    def test_rejects_a_shared_key_matching_a_non_simulator_axis_name(self):
        # difficulty_level is the axis name but not a simulate_clusters
        # parameter; putting it in shared must not be silently accepted, or
        # it would reach simulate_clusters as an unknown keyword at run time.
        with pytest.raises(ValueError, match="difficulty_level"):
            _scenario(shared={"n_total": 120, "p": 4, "difficulty_level": 1})

    def test_n_trees_defaults_to_100_and_can_be_overridden(self):
        assert _scenario().n_trees == 100
        assert _scenario(n_trees=500).n_trees == 500

    def test_sim_kwargs_merges_shared_and_the_selected_anchor(self):
        kwargs = _scenario().sim_kwargs(axis_value=1, axis_label="medium")
        assert kwargs["cluster_scale"] == 2.0
        assert kwargs["n_total"] == 120

    def test_sim_kwargs_sets_the_axis_parameter_when_the_axis_is_a_sim_kwarg(self):
        scenario = _scenario(
            axis=Axis(name="n_total", values=(1000, 5500), labels=("start", "end")),
            anchors={"start": {"cluster_scale": 1.0}, "end": {"cluster_scale": 2.0}},
            shared={"p": 4},
        )
        assert scenario.sim_kwargs(axis_value=5500, axis_label="end")["n_total"] == 5500

    def test_sim_kwargs_ignores_the_axis_value_for_a_non_sim_axis(self):
        scenario = _scenario(
            axis=_axis(),
            shared={"n_total": 120, "p": 4},
        )
        assert "difficulty_level" not in scenario.sim_kwargs(axis_value=0, axis_label="easy")


import numpy as np

from benchmarks._types import Manifest, Study


class TestStudy:
    def test_holds_a_loader_and_an_estimator(self):
        def loader():
            return np.zeros((4, 2)), np.array([0, 0, 1, 1]), {"source": "test"}

        study = Study(
            name="demo",
            loader=loader,
            estimator=EstimatorSpec(name="kmeans"),
            candidate_k=(2, 3),
        )
        X, y, meta = study.loader()
        assert X.shape == (4, 2)
        assert meta["source"] == "test"
        assert study.k_star is None

    def test_rejects_empty_candidate_k(self):
        with pytest.raises(ValueError, match="candidate_k"):
            Study(
                name="demo",
                loader=lambda: (np.zeros((2, 2)), np.zeros(2), {}),
                estimator=EstimatorSpec(name="kmeans"),
                candidate_k=(),
            )


class TestManifest:
    def test_to_dict_is_json_serializable(self):
        import json

        manifest = Manifest(
            run_id="abc123",
            scenario="demo",
            config_hash="deadbeef0000",
            anchor_set="ACTIVE_ANCHORS",
            n_seeds=2,
            n_resamples=3,
            n_jobs=1,
            random_state=0,
            wall_clock_s=1.5,
            peak_rss_bytes=1024,
            peak_rss_unit="bytes",
            git_sha="0" * 40,
            package_versions={"numpy": "2.0.0"},
            platform={"system": "Darwin", "cpu": "arm64", "cores": 10},
            config={"k_star": 5},
        )
        payload = json.dumps(manifest.to_dict())
        assert "deadbeef0000" in payload

    def test_records_the_rss_unit_explicitly(self):
        manifest = Manifest(
            run_id="a",
            scenario="s",
            config_hash="h",
            anchor_set="ACTIVE_ANCHORS",
            n_seeds=1,
            n_resamples=1,
            n_jobs=1,
            random_state=0,
            wall_clock_s=0.0,
            peak_rss_bytes=1,
            peak_rss_unit="bytes",
            git_sha="",
            package_versions={},
            platform={},
            config={},
        )
        assert manifest.to_dict()["peak_rss_unit"] == "bytes"


def test_graph_and_minibatch_estimators_are_known():
    assert "leiden" in KNOWN_ESTIMATORS
    assert "minibatch_kmeans" in KNOWN_ESTIMATORS
    assert EstimatorSpec(name="leiden").name == "leiden"
    assert EstimatorSpec(name="minibatch_kmeans").name == "minibatch_kmeans"


def test_unknown_estimator_still_raises():
    with pytest.raises(ValueError, match="Unknown estimator"):
        EstimatorSpec(name="kmenas")
