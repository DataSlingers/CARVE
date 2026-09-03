"""Tests for the S1 example-scatter figure."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from benchmarks._registry import SCENARIOS as REGISTRY
from benchmarks.figures import figure_benchmarking_examples

SCENARIOS = ("gaussians", "moons")


class TestFigureBenchmarkingExamples:
    def test_returns_a_figure_without_saving(self, tmp_path):
        fig = figure_benchmarking_examples(scenarios=SCENARIOS, save=False)
        assert fig.get_axes()
        plt.close(fig)

    def test_one_axes_per_scenario_and_difficulty(self):
        fig = figure_benchmarking_examples(scenarios=SCENARIOS, save=False)
        assert len(fig.get_axes()) == len(SCENARIOS) * 3
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, tmp_path):
        fig = figure_benchmarking_examples(
            scenarios=SCENARIOS, save=True, out_dir=tmp_path
        )
        assert (tmp_path / "benchmarking_examples.png").exists()
        plt.close(fig)

    def test_uses_the_swiss_roll_estimator_the_benchmark_actually_ran(self):
        """S1 was drawn with spectral while the results came from agglomerative."""
        assert REGISTRY["swiss_rolls"].estimator.name == "agglomerative"

    def test_is_deterministic_for_a_fixed_seed(self):
        first = figure_benchmarking_examples(scenarios=("gaussians",), seed=3, save=False)
        second = figure_benchmarking_examples(scenarios=("gaussians",), seed=3, save=False)
        a = first.get_axes()[0].collections[0].get_offsets()
        b = second.get_axes()[0].collections[0].get_offsets()
        assert (a == b).all()
        plt.close(first)
        plt.close(second)
