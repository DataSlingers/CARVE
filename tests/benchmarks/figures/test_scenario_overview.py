"""Tests for the per-family section overview.

Not a manuscript figure, so it is not in the figure contract's EXPECTED
list. What it must get right is the pairing the contract cannot check: the
example row and the ARI panel have to describe the same family on the same
axis, and the axis has to be labeled for the sweep it actually is.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarks._artifacts import SCHEMA
from benchmarks._registry import PUBLISHED_RANDOM_STATE, SCENARIOS
from benchmarks.figures import figure_scenario_overview
from benchmarks.figures import _benchmarking_examples

METRICS = ("baseline_oracle", "ari_stability_1se", "silhouette")


def _frame(scenario: str) -> pd.DataFrame:
    """An artifact frame for a scenario, built off its registered axis."""
    axis = SCENARIOS[scenario].axis
    rows = []
    for axis_idx, axis_value, axis_label in axis:
        for seed in range(4):
            for metric in METRICS[1:]:
                for k in (4, 5, 6):
                    rows.append(
                        {
                            "run_id": "r1",
                            "scenario": scenario,
                            "axis_name": axis.name,
                            "axis_value": axis_value,
                            "axis_label": axis_label,
                            "seed": seed,
                            "k_star": 5,
                            "estimator": SCENARIOS[scenario].estimator.name,
                            "metric_name": metric,
                            "k": k,
                            "metric_value": 0.1 * k,
                            "is_selected": k == 5,
                            "selects_true_k": k == 5,
                            "ari_at_k": 0.9 - 0.1 * axis_idx,
                            "oracle_ari": 0.95,
                        }
                    )
    return pd.DataFrame(rows)[list(SCHEMA)]


@pytest.fixture
def gaussians():
    return _frame("gaussians")


class TestFigureScenarioOverview:
    def test_returns_a_figure(self, gaussians):
        fig = figure_scenario_overview("gaussians", gaussians, metrics=METRICS)
        assert fig.get_axes()
        plt.close(fig)

    def test_one_example_panel_per_axis_point_plus_the_ari_panel(self, gaussians):
        fig = figure_scenario_overview("gaussians", gaussians, metrics=METRICS)
        assert len(fig.get_axes()) == len(SCENARIOS["gaussians"].axis) + 1
        plt.close(fig)

    def test_every_example_panel_is_drawn(self, gaussians):
        fig = figure_scenario_overview("gaussians", gaussians, metrics=METRICS)
        example_axes = fig.get_axes()[:-1]
        assert all(ax.collections for ax in example_axes)
        plt.close(fig)

    def test_defaults_to_writing_nothing(self, gaussians, tmp_path):
        """A section view is not a manuscript figure; disk is opt-in."""
        fig = figure_scenario_overview(
            "gaussians", gaussians, metrics=METRICS, out_dir=tmp_path
        )
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

    def test_saves_under_a_name_that_cannot_clash_with_a_manuscript_figure(
        self, gaussians, tmp_path
    ):
        fig = figure_scenario_overview(
            "gaussians", gaussians, metrics=METRICS, save=True, out_dir=tmp_path
        )
        written = {p.name for p in tmp_path.iterdir()}
        assert written == {"scenario_overview_gaussians.png"}
        assert "benchmarking_examples.png" not in written
        assert "benchmarking_results.png" not in written
        plt.close(fig)

    def test_carries_one_deduplicated_figure_legend(self, gaussians):
        """Blank entries pad the short columns, so count the labelled ones."""
        fig = figure_scenario_overview("gaussians", gaussians, metrics=METRICS)
        assert len(fig.legends) == 1
        labelled = [t.get_text() for t in fig.legends[0].get_texts() if t.get_text()]
        assert len(labelled) == len(METRICS)
        plt.close(fig)

    def test_legend_matches_the_paper_figures_grouping(self, gaussians):
        """A section figure and Fig 4 must advertise their series identically."""
        fig = figure_scenario_overview("gaussians", gaussians, metrics=METRICS)
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        assert labels[0] == "Baseline (Oracle k*)"
        plt.close(fig)

    def test_raises_on_an_empty_frame(self):
        empty = pd.DataFrame(columns=list(SCHEMA))
        with pytest.raises(ValueError, match="nothing to draw"):
            figure_scenario_overview("gaussians", empty)

    def test_raises_on_an_unknown_scenario(self, gaussians):
        with pytest.raises(KeyError, match="Unknown scenario"):
            figure_scenario_overview("not_a_scenario", gaussians)

    def test_rejects_a_frame_whose_axis_no_longer_matches_the_registry(self):
        """A stale run and the registry would describe different experiments.

        The example row is simulated from the registry while the curve is
        read from the artifact, and runs are content-addressed, so a run
        made before an axis was redefined stays readable under its own hash.
        Its x ticks come from the frame, so nothing on the figure would show
        that the two halves disagree.
        """
        stale = _frame("gaussians_samples")
        moved = dict(
            zip(SCENARIOS["gaussians_samples"].axis.values, (2000, 6000, 20000))
        )
        stale["axis_value"] = stale["axis_value"].map(moved)
        with pytest.raises(ValueError, match="different experiments"):
            figure_scenario_overview("gaussians_samples", stale, metrics=METRICS)


class TestDifficultyAxis:
    def test_ticks_read_easy_medium_hard_in_axis_order(self, gaussians):
        fig = figure_scenario_overview("gaussians", gaussians, metrics=METRICS)
        ari_ax = fig.get_axes()[-1]
        assert [t.get_text() for t in ari_ax.get_xticklabels()] == [
            "Easy",
            "Medium",
            "Hard",
        ]
        assert ari_ax.get_xlabel() == "SNR"
        plt.close(fig)

    def test_tick_order_follows_axis_value_not_label_text(self):
        """Lexical order would read easy, hard, medium -- mislabeling two points.

        The frame is built with its rows shuffled into lexical label order,
        which is the order read_run's glob returns checkpoints in.
        """
        frame = _frame("gaussians").sort_values("axis_label").reset_index(drop=True)
        fig = figure_scenario_overview("gaussians", frame, metrics=METRICS)
        ari_ax = fig.get_axes()[-1]
        assert [t.get_text() for t in ari_ax.get_xticklabels()] == [
            "Easy",
            "Medium",
            "Hard",
        ]
        plt.close(fig)

    def test_example_panels_are_titled_with_the_difficulty_labels(self, gaussians):
        fig = figure_scenario_overview("gaussians", gaussians, metrics=METRICS)
        titles = [ax.get_title() for ax in fig.get_axes()[:-1]]
        assert titles == ["Easy", "Medium", "Hard"]
        plt.close(fig)


class TestScalingAxis:
    def test_x_axis_names_the_swept_quantity_not_difficulty(self):
        """The bug this guards: a scaling sweep labeled easy/medium/hard."""
        fig = figure_scenario_overview(
            "gaussians_samples", _frame("gaussians_samples"), metrics=METRICS
        )
        ari_ax = fig.get_axes()[-1]
        assert ari_ax.get_xlabel() == "Number of samples (n)"
        labels = [t.get_text() for t in ari_ax.get_xticklabels()]
        assert "Easy" not in labels
        plt.close(fig)

    def test_ticks_sit_on_the_swept_values(self):
        fig = figure_scenario_overview(
            "gaussians_samples", _frame("gaussians_samples"), metrics=METRICS
        )
        ari_ax = fig.get_axes()[-1]
        expected = list(SCENARIOS["gaussians_samples"].axis.values)
        assert list(ari_ax.get_xticks()) == expected
        plt.close(fig)

    def test_example_panels_are_titled_with_the_axis_value(self):
        fig = figure_scenario_overview(
            "gaussians_dimensionality",
            _frame("gaussians_dimensionality"),
            metrics=METRICS,
        )
        titles = [ax.get_title() for ax in fig.get_axes()[:-1]]
        assert titles == [
            f"p = {value}"
            for value in SCENARIOS["gaussians_dimensionality"].axis.values
        ]
        plt.close(fig)


class TestSharedWithS1:
    def test_example_seeds_use_the_runners_formula(self, gaussians, monkeypatch):
        """Both this figure and S1 must show the data the benchmark scored.

        They share draw_example_row precisely so the derivation cannot drift
        between them, so the expectation is rederived from the arithmetic
        rather than hardcoded -- a hardcoded list would still pass if the
        random_state term were dropped and the default happened to be 0.
        """
        captured = []
        original = _benchmarking_examples.simulate

        def spy(scenario, *, axis_value, axis_label, seed):
            captured.append(seed)
            return original(
                scenario, axis_value=axis_value, axis_label=axis_label, seed=seed
            )

        monkeypatch.setattr(_benchmarking_examples, "simulate", spy)

        fig = figure_scenario_overview(
            "gaussians", gaussians, metrics=METRICS, seed=5, random_state=11
        )
        plt.close(fig)
        assert captured == [5 + axis_idx * 10000 + 11 for axis_idx in range(3)]

    def test_default_random_state_matches_the_published_benchmark(self):
        import inspect

        default = (
            inspect.signature(figure_scenario_overview)
            .parameters["random_state"]
            .default
        )
        assert default == PUBLISHED_RANDOM_STATE
