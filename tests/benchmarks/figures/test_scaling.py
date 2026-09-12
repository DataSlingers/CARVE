"""Tests for the two scaling figures."""

import inspect

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarks._artifacts import RUNTIME_SCHEMA, SCHEMA
from benchmarks._theme import METRIC_COLORS
from benchmarks.figures import _scaling, figure_scaling_ari, figure_scaling_runtime

METRICS = ("ari_stability_1se", "ari_generalizability_1se")


def _results(scenario, axis_name, values):
    rows = []
    for axis_value, axis_label in zip(values, ("start", "middle", "end")):
        for seed in range(3):
            for metric in METRICS:
                for k in (4, 5):
                    rows.append({
                        "run_id": "r1", "scenario": scenario, "axis_name": axis_name,
                        "axis_value": axis_value, "axis_label": axis_label, "seed": seed,
                        "k_star": 5, "estimator": "kmeans", "metric_name": metric, "k": k,
                        "metric_value": 0.1 * k, "is_selected": k == 5,
                        "selects_true_k": k == 5, "ari_at_k": 0.8, "oracle_ari": 0.9,
                    })
    return pd.DataFrame(rows)[list(SCHEMA)]


def _runtimes(scenario, axis_name, values):
    rows = []
    for axis_value, axis_label in zip(values, ("start", "middle", "end")):
        for seed in range(3):
            rows.append({
                "run_id": "r1", "scenario": scenario, "axis_name": axis_name,
                "axis_value": axis_value, "axis_label": axis_label, "seed": seed,
                "n_samples": int(axis_value), "n_features": 50, "n_resamples": 100,
                "n_jobs": 1, "estimator": "kmeans",
                "t_default_s": 1.5 + axis_value / 800.0,
                "t_stability_s": 1.0 + axis_value / 1000.0,
                "t_generalizability_s": 1.2 + axis_value / 900.0,
                "t_per_k_stability_s": (1.0 + axis_value / 1000.0) / 5,
                "t_per_k_generalizability_s": (1.2 + axis_value / 900.0) / 5,
            })
    return pd.DataFrame(rows)[list(RUNTIME_SCHEMA)]


@pytest.fixture
def results():
    return {
        "gaussians_samples": _results("gaussians_samples", "n_total", (1000, 5500, 10000)),
        "gaussians_dimensionality": _results(
            "gaussians_dimensionality", "embed_dim", (10, 255, 500)
        ),
    }


@pytest.fixture
def runtimes():
    return {
        "gaussians_samples": _runtimes("gaussians_samples", "n_total", (1000, 5500, 10000)),
        "gaussians_dimensionality": _runtimes(
            "gaussians_dimensionality", "embed_dim", (10, 255, 500)
        ),
    }


class TestFigureScalingAri:
    def test_returns_a_figure_with_one_panel_per_scenario(self, results):
        fig = figure_scaling_ari(results, metrics=METRICS, save=False)
        drawn = [ax for ax in fig.get_axes() if ax.lines]
        assert len(drawn) == 2
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, results, tmp_path):
        fig = figure_scaling_ari(results, metrics=METRICS, save=True, out_dir=tmp_path)
        assert (tmp_path / "paper_fig_scaling_ari_k5.png").exists()
        plt.close(fig)

    def test_save_false_writes_nothing(self, results, tmp_path):
        fig = figure_scaling_ari(results, metrics=METRICS, save=False, out_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

    def test_filename_carries_k_star(self, results, tmp_path):
        fig = figure_scaling_ari(
            results, metrics=METRICS, k_star=7, save=True, out_dir=tmp_path
        )
        assert (tmp_path / "paper_fig_scaling_ari_k7.png").exists()
        plt.close(fig)

    def test_x_axis_labels_name_the_swept_quantity(self, results):
        fig = figure_scaling_ari(results, metrics=METRICS, save=False)
        labels = {ax.get_xlabel() for ax in fig.get_axes()}
        assert "Number of samples (n)" in labels
        plt.close(fig)

    def test_has_one_deduplicated_legend(self, results):
        fig = figure_scaling_ari(results, metrics=METRICS, save=False)
        assert len(fig.legends) == 1
        assert len(fig.legends[0].get_texts()) == len(METRICS)
        plt.close(fig)

    def test_no_element_scale_fudge_is_applied_by_default(self, results):
        """One theme means Fig 4 and this figure match without a fudge factor."""
        source = inspect.getsource(_scaling)
        assert "0.47" not in source

    def test_default_metrics_draw_the_oracle_baseline(self, results):
        # S2 Fig's caption: curves are shown "against the oracle-k* baseline".
        fig = figure_scaling_ari(results, save=False)
        colors = {ln.get_color() for ax in fig.get_axes() for ln in ax.lines}
        assert METRIC_COLORS["baseline_oracle"] in colors
        plt.close(fig)


class TestFigureScalingRuntime:
    def test_returns_a_figure_with_a_log_y_axis(self, runtimes):
        fig = figure_scaling_runtime(runtimes, save=False)
        drawn = [ax for ax in fig.get_axes() if ax.lines]
        assert drawn
        assert all(ax.get_yscale() == "log" for ax in drawn)
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, runtimes, tmp_path):
        fig = figure_scaling_runtime(runtimes, save=True, out_dir=tmp_path)
        assert (tmp_path / "paper_fig_scaling_runtime_k5.png").exists()
        plt.close(fig)

    def test_save_false_writes_nothing(self, runtimes, tmp_path):
        fig = figure_scaling_runtime(runtimes, save=False, out_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

    def test_y_axis_is_labeled_in_seconds(self, runtimes):
        fig = figure_scaling_runtime(runtimes, save=False)
        assert any("second" in ax.get_ylabel().lower() for ax in fig.get_axes())
        plt.close(fig)

    def test_draws_two_curves_per_panel_like_the_published_figure(self, runtimes):
        # ax.errorbar(..., label=...) attaches the label to the returned
        # ErrorbarContainer, not to the constituent Line2D objects it adds to
        # ax.lines -- those keep matplotlib's default "_nolegend_" label
        # regardless of what label was passed. Checking ax.containers is the
        # only way to see the two labelled curves this figure actually draws.
        fig = figure_scaling_runtime(runtimes, save=False)
        drawn = [ax for ax in fig.get_axes() if ax.lines]
        for ax in drawn:
            labelled = [c for c in ax.containers if not c.get_label().startswith("_")]
            assert len(labelled) == 2
        plt.close(fig)

    def test_legend_names_both_carve_modes(self, runtimes):
        fig = figure_scaling_runtime(runtimes, save=False)
        labels = {t.get_text() for t in fig.legends[0].get_texts()}
        assert labels == {"CARVE Stability", "CARVE Generalizability"}
        plt.close(fig)

    def test_raises_on_an_empty_mapping(self):
        with pytest.raises(ValueError, match="at least one scenario"):
            figure_scaling_runtime({}, save=False)
