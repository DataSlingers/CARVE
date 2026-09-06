"""Tests for Fig 4."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarks._artifacts import SCHEMA
from benchmarks.figures import figure_benchmarking_results

METRICS = ("ari_stability_1se", "ari_generalizability_1se", "silhouette")


def _frame(scenario: str) -> pd.DataFrame:
    rows = []
    for axis_value, axis_label in enumerate(("easy", "medium", "hard")):
        for seed in range(4):
            for metric in METRICS:
                for k in (4, 5, 6):
                    rows.append(
                        {
                            "run_id": "r1",
                            "scenario": scenario,
                            "axis_name": "difficulty_level",
                            "axis_value": axis_value,
                            "axis_label": axis_label,
                            "seed": seed,
                            "k_star": 5,
                            "estimator": "kmeans",
                            "metric_name": metric,
                            "k": k,
                            "metric_value": 0.1 * k,
                            "is_selected": k == 5,
                            "selects_true_k": k == 5,
                            "ari_at_k": 0.9 - 0.1 * axis_value,
                            "oracle_ari": 0.95,
                        }
                    )
    return pd.DataFrame(rows)[list(SCHEMA)]


@pytest.fixture
def results():
    return {name: _frame(name) for name in ("gaussians", "t_dist", "moons")}


class TestFigureBenchmarkingResults:
    def test_returns_a_figure(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        assert fig.get_axes()
        plt.close(fig)

    def test_one_panel_per_scenario(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        drawn = [ax for ax in fig.get_axes() if ax.lines or ax.collections]
        assert len(drawn) == len(results)
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, results, tmp_path):
        fig = figure_benchmarking_results(
            results, metrics=METRICS, save=True, out_dir=tmp_path
        )
        assert (tmp_path / "benchmarking_results.png").exists()
        plt.close(fig)

    def test_save_false_writes_nothing(self, results, tmp_path):
        fig = figure_benchmarking_results(
            results, metrics=METRICS, save=False, out_dir=tmp_path
        )
        assert not (tmp_path / "benchmarking_results.png").exists()
        plt.close(fig)

    def test_carries_one_deduplicated_figure_legend(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        assert len(fig.legends) == 1
        assert len(fig.legends[0].get_texts()) == len(METRICS)
        plt.close(fig)

    def test_panels_are_lettered(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        letters = {t.get_text() for ax in fig.get_axes() for t in ax.texts}
        assert {"A", "B", "C"} <= letters
        plt.close(fig)

    def test_carve_stability_is_drawn_in_the_theme_green(self, results):
        from benchmarks._theme import METRIC_COLORS

        fig = figure_benchmarking_results(
            results, metrics=("ari_stability_1se",), save=False
        )
        colors = {ln.get_color() for ax in fig.get_axes() for ln in ax.lines}
        assert METRIC_COLORS["ari_stability_1se"] in colors
        plt.close(fig)

    def test_raises_on_an_empty_mapping(self):
        with pytest.raises(ValueError, match="at least one scenario"):
            figure_benchmarking_results({}, save=False)

    def test_default_metrics_draw_the_oracle_baseline(self, results):
        # Fig 4's caption: "The grey line shows the oracle ARI(k*=5)". Every
        # frame here carries oracle_ari, so the default metrics (used when
        # no explicit metrics= is passed) must include the baseline series.
        from benchmarks._theme import METRIC_COLORS

        fig = figure_benchmarking_results(results, save=False)
        colors = {ln.get_color() for ax in fig.get_axes() for ln in ax.lines}
        assert METRIC_COLORS["baseline_oracle"] in colors
        plt.close(fig)
