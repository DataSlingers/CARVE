"""Tests for the panel drawing primitives.

The uniform ax-in / ax-out contract is the reason this module exists, so it
is asserted directly rather than left to convention.
"""

import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pytest

from benchmarks import _panels
from benchmarks._panels import (
    cluster_color_map,
    metric_lines,
    runtime_lines,
    scatter_clusters,
)
from benchmarks._artifacts import SCHEMA
from benchmarks._theme import metric_color

AX_FIRST_FUNCTIONS = (
    "scatter_clusters",
    "metric_lines",
    "runtime_lines",
)


class TestContract:
    @pytest.mark.parametrize("name", AX_FIRST_FUNCTIONS)
    def test_first_parameter_is_ax(self, name):
        signature = inspect.signature(getattr(_panels, name))
        assert next(iter(signature.parameters)) == "ax"

    def test_no_primitive_calls_plt_show(self):
        source = inspect.getsource(_panels)
        assert "plt.show(" not in source

    def test_no_primitive_creates_its_own_figure(self):
        source = inspect.getsource(_panels)
        assert "plt.subplots(" not in source
        assert "plt.figure(" not in source


def _results_frame():
    rows = []
    for axis_value in (0, 1, 2):
        for seed in range(3):
            for metric in ("ari_stability_1se", "silhouette"):
                for k in (4, 5):
                    rows.append(
                        {
                            "run_id": "r1",
                            "scenario": "demo",
                            "axis_name": "difficulty_level",
                            "axis_value": axis_value,
                            "axis_label": ["easy", "medium", "hard"][axis_value],
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


class TestScatterClusters:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        Z = np.random.default_rng(0).normal(size=(30, 2))
        labels = np.repeat([0, 1, 2], 10)
        assert scatter_clusters(ax, Z, labels) is ax
        plt.close(fig)

    def test_draws_one_collection_per_label(self):
        fig, ax = plt.subplots()
        Z = np.random.default_rng(0).normal(size=(30, 2))
        labels = np.repeat([0, 1, 2], 10)
        scatter_clusters(ax, Z, labels)
        assert len(ax.collections) == 3
        plt.close(fig)

    def test_hides_axes_when_asked(self):
        fig, ax = plt.subplots()
        Z = np.random.default_rng(0).normal(size=(10, 2))
        scatter_clusters(ax, Z, np.zeros(10, dtype=int), hide_axes=True)
        assert list(ax.get_xticks()) == []
        plt.close(fig)

    def test_uses_cluster_color_map(self):
        fig, ax = plt.subplots()
        Z = np.random.default_rng(0).normal(size=(30, 2))
        labels = np.repeat([0, 1, 2], 10)
        expected_cmap = cluster_color_map(labels)
        scatter_clusters(ax, Z, labels, color_map=expected_cmap)
        # Verify each collection has the expected color (RGB only, alpha may vary)
        unique_labels = list(dict.fromkeys(labels.tolist()))
        for i, (label, collection) in enumerate(zip(unique_labels, ax.collections)):
            expected_color = expected_cmap[label]
            expected_rgb = mcolors.to_rgba(expected_color)[:3]
            face_colors = collection.get_facecolors()
            assert len(face_colors) > 0
            actual_rgb = face_colors[0][:3]
            assert np.allclose(actual_rgb, expected_rgb), \
                f"Label {label}: expected {expected_rgb}, got {actual_rgb}"
        plt.close(fig)

    def test_uses_fallback_for_missing_color(self):
        fig, ax = plt.subplots()
        Z = np.random.default_rng(0).normal(size=(30, 2))
        labels = np.repeat([0, 1, 2], 10)
        # Provide a color map that does not include label 2
        partial_cmap = {0: "#FF0000", 1: "#00FF00"}
        scatter_clusters(ax, Z, labels, color_map=partial_cmap)
        # Label 2 should use the fallback color #7F7F7F
        expected_rgb = mcolors.to_rgba("#7F7F7F")[:3]
        # Find the collection for label 2 (should be the third one based on order)
        # The order is determined by dict.fromkeys(labels.tolist())
        third_collection = ax.collections[2]
        face_colors = third_collection.get_facecolors()
        actual_rgb = face_colors[0][:3]
        assert np.allclose(actual_rgb, expected_rgb), \
            f"Expected fallback {expected_rgb}, got {actual_rgb}"
        plt.close(fig)


class TestClusterColorMap:
    def test_assigns_one_color_per_distinct_label(self):
        mapping = cluster_color_map(np.array([2, 2, 0, 1, 1]))
        assert set(mapping) == {0, 1, 2}
        assert len(set(mapping.values())) == 3


class TestMetricLines:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        assert metric_lines(ax, _results_frame(), metrics=("silhouette",)) is ax
        plt.close(fig)

    def test_draws_one_series_per_requested_metric(self):
        fig, ax = plt.subplots()
        metric_lines(ax, _results_frame(), metrics=("ari_stability_1se", "silhouette"))
        assert len(ax.get_legend().get_texts()) == 2
        plt.close(fig)

    def test_uses_only_selected_rows(self):
        fig, ax = plt.subplots()
        metric_lines(ax, _results_frame(), metrics=("silhouette",), show_legend=False)
        line = ax.lines[0]
        # three axis values, one point each, drawn from the k=5 rows only
        assert len(line.get_xdata()) == 3
        plt.close(fig)

    def test_skips_a_metric_with_no_rows(self):
        fig, ax = plt.subplots()
        metric_lines(ax, _results_frame(), metrics=("gap",), show_legend=False)
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_lines_use_metric_colors(self):
        fig, ax = plt.subplots()
        metrics = ("ari_stability_1se", "silhouette")
        metric_lines(ax, _results_frame(), metrics=metrics, show_legend=False)
        # Get data lines (those with marker 'o')
        data_lines = [line for line in ax.lines if line.get_marker() == 'o']
        assert len(data_lines) == len(metrics)
        for line, expected_metric in zip(data_lines, metrics):
            expected_color = metric_color(expected_metric)
            actual_color = line.get_color()
            assert actual_color == expected_color, \
                f"Metric {expected_metric}: expected {expected_color}, got {actual_color}"
        plt.close(fig)


def _runtime_frame():
    return pd.DataFrame(
        {
            "axis_value": [1000, 1000, 5500, 5500],
            "t_default_s": [1.5, 1.7, 6.0, 6.6],
            "t_stability_s": [1.0, 1.2, 4.0, 4.4],
            "t_generalizability_s": [1.3, 1.4, 5.1, 5.5],
        }
    )


class TestRuntimeLines:
    def test_returns_the_same_axes_and_uses_a_log_scale(self):
        fig, ax = plt.subplots()
        assert runtime_lines(ax, _runtime_frame()) is ax
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_draws_both_mode_curves_by_default(self):
        fig, ax = plt.subplots()
        runtime_lines(ax, _runtime_frame())
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["CARVE Stability", "CARVE Generalizability"]
        plt.close(fig)

    def test_the_two_curves_are_dodged_apart(self):
        fig, ax = plt.subplots()
        runtime_lines(ax, _runtime_frame())
        # errorbar creates multiple lines per call; find the main data lines (with markers)
        data_lines = [line for line in ax.lines if line.get_marker() == 'o']
        assert len(data_lines) >= 2
        first, second = data_lines[0].get_xdata(), data_lines[1].get_xdata()
        assert not np.allclose(first, second)
        plt.close(fig)

    def test_skips_a_column_that_is_all_nan(self):
        """Untimed cells record nan, and an untimed series must not be drawn."""
        fig, ax = plt.subplots()
        df = _runtime_frame()
        df["t_generalizability_s"] = np.nan
        runtime_lines(ax, df)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["CARVE Stability"]
        plt.close(fig)

    def test_skips_a_column_that_is_absent(self):
        fig, ax = plt.subplots()
        runtime_lines(ax, _runtime_frame()[["axis_value", "t_stability_s"]])
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_rejects_mismatched_columns_and_labels(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="same length"):
            runtime_lines(ax, _runtime_frame(), runtime_cols=("t_stability_s",))
        plt.close(fig)

    def test_lines_use_correct_mode_colors(self):
        fig, ax = plt.subplots()
        runtime_lines(ax, _runtime_frame())
        # Get data lines (those with marker 'o')
        data_lines = [line for line in ax.lines if line.get_marker() == 'o']
        assert len(data_lines) == 2
        # The two modes should have colors from their mapped metrics
        expected_metrics = ("ari_stability_1se", "ari_generalizability_1se")
        for line, expected_metric in zip(data_lines, expected_metrics):
            expected_color = metric_color(expected_metric)
            actual_color = line.get_color()
            assert actual_color == expected_color, \
                f"Mode metric {expected_metric}: expected {expected_color}, got {actual_color}"
        plt.close(fig)
