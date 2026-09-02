"""Tests for the simulation wrapper."""

import numpy as np

from benchmarks._simulate import simulate
from benchmarks._types import Axis, EstimatorSpec, Scenario


def _scenario(axis=None, anchors=None, shared=None):
    return Scenario(
        name="demo",
        axis=axis or Axis(name="difficulty_level", values=(0, 1), labels=("easy", "hard")),
        anchors=anchors or {"easy": {"cluster_scale": 1.0}, "hard": {"cluster_scale": 4.0}},
        shared=shared or {"n_total": 150, "p": 4, "distribution": "gaussian"},
        estimator=EstimatorSpec(name="kmeans"),
    )


class TestSimulate:
    def test_returns_matching_x_and_y_shapes(self):
        X, y = simulate(_scenario(), axis_value=0, axis_label="easy", seed=0)
        assert X.shape == (150, 4)
        assert y.shape == (150,)

    def test_simulates_k_star_clusters(self):
        _, y = simulate(_scenario(), axis_value=0, axis_label="easy", seed=0)
        assert len(np.unique(y)) == 5

    def test_is_deterministic_for_a_fixed_seed(self):
        a, _ = simulate(_scenario(), axis_value=0, axis_label="easy", seed=7)
        b, _ = simulate(_scenario(), axis_value=0, axis_label="easy", seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_data(self):
        a, _ = simulate(_scenario(), axis_value=0, axis_label="easy", seed=1)
        b, _ = simulate(_scenario(), axis_value=0, axis_label="easy", seed=2)
        assert not np.array_equal(a, b)

    def test_a_scaling_axis_changes_the_simulated_size(self):
        scenario = _scenario(
            axis=Axis(name="n_total", values=(120, 300), labels=("start", "end")),
            anchors={"start": {"cluster_scale": 1.0}, "end": {"cluster_scale": 1.0}},
            shared={"p": 4, "distribution": "gaussian"},
        )
        X_small, _ = simulate(scenario, axis_value=120, axis_label="start", seed=0)
        X_large, _ = simulate(scenario, axis_value=300, axis_label="end", seed=0)
        assert X_small.shape[0] == 120
        assert X_large.shape[0] == 300
