"""Tests for the S1 example-scatter figure."""

import inspect

import matplotlib.pyplot as plt

from benchmarks._registry import PUBLISHED_RANDOM_STATE
from benchmarks._registry import SCENARIOS as REGISTRY
from benchmarks.figures import _benchmarking_examples, figure_benchmarking_examples

SCENARIOS = ("gaussians", "moons")


class TestFigureBenchmarkingExamples:
    def test_returns_a_figure_without_saving(self, tmp_path):
        fig = figure_benchmarking_examples(
            scenarios=SCENARIOS, save=False, out_dir=tmp_path
        )
        assert fig.get_axes()
        assert not (tmp_path / "benchmarking_examples.png").exists()
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

    def test_default_random_state_matches_the_published_benchmark(self):
        """The docstring's "same data the benchmark scored" claim needs this."""
        default = inspect.signature(figure_benchmarking_examples).parameters[
            "random_state"
        ].default
        assert default == PUBLISHED_RANDOM_STATE

    def test_seed_derivation_matches_the_runners_formula(self, monkeypatch):
        """benchmark_seed must equal _run.py's run_cell formula exactly:
        seed + axis_idx * 10000 + random_state.

        A hardcoded expected list (e.g. 42, 10042, 20042) would still pass if
        the random_state term were silently dropped and random_state's
        default were coincidentally 0 elsewhere. Deriving the expectation
        from the same arithmetic, applied to arbitrary seed/random_state
        arguments, catches that regardless of what any default happens to be.
        """
        captured_seeds = []
        original_simulate = _benchmarking_examples.simulate

        def spy_simulate(scenario, *, axis_value, axis_label, seed):
            captured_seeds.append(seed)
            return original_simulate(
                scenario, axis_value=axis_value, axis_label=axis_label, seed=seed
            )

        monkeypatch.setattr(_benchmarking_examples, "simulate", spy_simulate)

        seed_arg = 5
        random_state_arg = 11
        fig = figure_benchmarking_examples(
            scenarios=("gaussians",),
            seed=seed_arg,
            random_state=random_state_arg,
            save=False,
        )
        plt.close(fig)

        expected = [
            seed_arg + axis_idx * 10000 + random_state_arg for axis_idx in range(3)
        ]
        assert captured_seeds == expected
