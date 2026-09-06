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
from matplotlib.legend import Legend

from benchmarks import _panels
from benchmarks._panels import (
    alluvial,
    ari_lollipop,
    carve_lines,
    cluster_color_map,
    cvi_lines,
    grouped_legend,
    metric_lines,
    panel_letter,
    runtime_lines,
    scatter_clusters,
)
from benchmarks._artifacts import SCHEMA
from benchmarks._theme import metric_color

AX_FIRST_FUNCTIONS = (
    "scatter_clusters",
    "metric_lines",
    "carve_lines",
    "cvi_lines",
    "alluvial",
    "ari_lollipop",
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
            assert np.allclose(actual_rgb, expected_rgb), (
                f"Label {label}: expected {expected_rgb}, got {actual_rgb}"
            )
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
        assert np.allclose(actual_rgb, expected_rgb), (
            f"Expected fallback {expected_rgb}, got {actual_rgb}"
        )
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
        data_lines = [line for line in ax.lines if line.get_marker() == "o"]
        assert len(data_lines) == len(metrics)
        for line, expected_metric in zip(data_lines, metrics):
            expected_color = metric_color(expected_metric)
            actual_color = line.get_color()
            assert actual_color == expected_color, (
                f"Metric {expected_metric}: expected {expected_color}, got {actual_color}"
            )
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
        data_lines = [line for line in ax.lines if line.get_marker() == "o"]
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
        data_lines = [line for line in ax.lines if line.get_marker() == "o"]
        assert len(data_lines) == 2
        # The two modes should have colors from their mapped metrics
        expected_metrics = ("ari_stability_1se", "ari_generalizability_1se")
        for line, expected_metric in zip(data_lines, expected_metrics):
            expected_color = metric_color(expected_metric)
            actual_color = line.get_color()
            assert actual_color == expected_color, (
                f"Mode metric {expected_metric}: expected {expected_color}, got {actual_color}"
            )
        plt.close(fig)


def _curves_and_best():
    # Two estimator configurations (Agglomerative and KMeans), each swept
    # over the same three k values -- the shape a case study that sweeps
    # more than one estimator actually produces, and the shape that exposed
    # cvi_lines' original interleaving bug: grouping by metric alone drew
    # both models' rows as one zig-zagging line per metric.
    curves = pd.DataFrame(
        {
            "metric": ["silhouette"] * 6 + ["gap"] * 6,
            "model": (["Agglomerative"] * 3 + ["KMeans"] * 3) * 2,
            "k": [3, 4, 5] * 4,
            "score": [
                0.55,
                0.65,
                0.60,  # silhouette, Agglomerative
                0.40,
                0.60,
                0.50,  # silhouette, KMeans
                0.20,
                0.30,
                0.25,  # gap, Agglomerative
                0.28,
                0.32,
                0.35,  # gap, KMeans
            ],
        }
    )
    # The two metrics deliberately pick different winning models, so a line
    # restricted to the wrong model would draw the wrong curve entirely, not
    # just the right curve with a wrong point count.
    best = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["Agglomerative", "KMeans"],
            "k": [4, 5],
            "score": [0.65, 0.35],
        }
    )
    return curves, best


class _StubCarve:
    """Minimal stand-in for a fitted CARVE object.

    carve_lines calls _select_row() (to resolve which estimator
    configuration a measure selects) and get_k() (to place the selected-k
    marker), so the stub implements exactly those two members instead of
    fitting a real model. estimator_results_ carries two method_ids, each
    swept over the same four k values -- mirroring a real case study that
    sweeps two estimators, the shape that exposed the original interleaving
    bug (see _curves_and_best's docstring for the cvi_lines analogue).

    Column names are the canonical estimator_results_ names a real fitted
    CARVE object uses (see carve._selection.MEASURE_MAP and carve._output,
    which reads record["ari_stability"] / record["ari_stability_se"]) --
    "stability" and "generalizability" are only measure aliases, never
    column names. A stub that named its columns after the aliases would let
    carve_lines index the alias directly and still pass, which is exactly
    the bug this is guarding against.
    """

    def __init__(self, results: pd.DataFrame, selection: dict):
        self.estimator_results_ = results
        self._selection = selection  # measure -> {"method_id", "n_clusters"}

    def _select_row(self, *, measure, rule="1se", not_two=False):
        choice = self._selection[measure]
        results = self.estimator_results_
        row = results.loc[
            (results["method_id"] == choice["method_id"])
            & (results["n_clusters"] == choice["n_clusters"])
        ].iloc[0]
        return row, 0, choice["n_clusters"], False

    def get_k(self, *, measure, rule="1se", not_two=False):
        return self._selection[measure]["n_clusters"]


def _carve_obj():
    results = pd.DataFrame(
        {
            "n_clusters": [3, 4, 5, 6, 3, 4, 5, 6],
            "method_id": ["m0"] * 4 + ["m1"] * 4,
            "method_label": ["KMeans"] * 4
            + ["AgglomerativeClustering, linkage=ward"] * 4,
            "ari_stability": [0.40, 0.70, 0.65, 0.55, 0.35, 0.50, 0.80, 0.60],
            "ari_stability_se": [0.05, 0.04, 0.03, 0.04, 0.05, 0.04, 0.03, 0.04],
            "ari_generalizability": [0.30, 0.45, 0.60, 0.50, 0.25, 0.55, 0.62, 0.58],
            "ari_generalizability_se": [0.06, 0.05, 0.05, 0.04, 0.06, 0.05, 0.05, 0.04],
        }
    )
    return _StubCarve(
        results,
        {
            "stability": {"method_id": "m0", "n_clusters": 4},
            "generalizability": {"method_id": "m1", "n_clusters": 5},
        },
    )


class TestCarveLines:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        assert carve_lines(ax, _carve_obj()) is ax
        plt.close(fig)

    def test_draws_one_line_per_requested_measure(self):
        # A non-default, single-element tuple: if the measures argument were
        # ignored in favor of the ("stability", "generalizability") default,
        # this would draw two lines instead of one.
        fig, ax = plt.subplots()
        carve_lines(ax, _carve_obj(), measures=("generalizability",))
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 1
        plt.close(fig)

    def test_lines_use_measure_specific_colors(self):
        # Measures are requested in the opposite order from the default
        # tuple. A regression that fell back to the default order rather
        # than the caller's order would draw the colors in the wrong slots.
        fig, ax = plt.subplots()
        carve_lines(ax, _carve_obj(), measures=("generalizability", "stability"))
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        expected_colors = [
            metric_color("ari_generalizability_1se"),
            metric_color("ari_stability_1se"),
        ]
        for line, expected in zip(data_lines, expected_colors):
            assert line.get_color() == expected
        plt.close(fig)

    def test_shows_the_selected_k_by_default(self):
        fig, ax = plt.subplots()
        carve_lines(ax, _carve_obj(), measures=("stability",))
        marker_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        other_lines = [ln for ln in ax.lines if ln.get_marker() != "o"]
        assert len(marker_lines) == 1
        assert len(other_lines) == 1
        assert other_lines[0].get_xdata()[0] == 4
        plt.close(fig)

    def test_omits_the_selected_k_marker_when_asked(self):
        fig, ax = plt.subplots()
        carve_lines(ax, _carve_obj(), measures=("stability",), show_selected_k=False)
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_line_length_matches_one_configurations_k_values_not_every_row(self):
        # _carve_obj's estimator_results_ carries eight rows -- two method
        # ids swept over four k values each. A line that read every row
        # instead of filtering to the selected method_id would be eight
        # points long, not four.
        fig, ax = plt.subplots()
        carve_lines(ax, _carve_obj(), measures=("stability",), show_selected_k=False)
        line = ax.lines[0]
        assert len(line.get_xdata()) == 4
        plt.close(fig)

    def test_line_x_values_are_monotonic(self):
        # Concatenating both method_ids' k ranges end to end (the
        # interleaving bug) draws k = 3, 4, 5, 6, 3, 4, 5, 6 -- a line that
        # runs up and then jumps back down. A single configuration's own
        # sweep is strictly increasing.
        fig, ax = plt.subplots()
        carve_lines(
            ax,
            _carve_obj(),
            measures=("stability", "generalizability"),
            show_selected_k=False,
        )
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        for line in data_lines:
            x = line.get_xdata()
            assert np.all(np.diff(x) > 0)
        plt.close(fig)


class TestCviLines:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        assert cvi_lines(ax, curves, best) is ax
        plt.close(fig)

    def test_draws_one_line_per_metric(self):
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        assert len([ln for ln in ax.lines if ln.get_label() != "_nolegend_"]) >= 2
        plt.close(fig)

    def test_marks_the_selected_k_for_each_metric(self):
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        assert len(ax.collections) >= 2
        plt.close(fig)

    def test_lines_use_metric_specific_colors(self):
        # silhouette and gap map to different theme colors, so a bug that
        # mixed up which curve gets which color would be caught here.
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        expected_colors = [metric_color("silhouette"), metric_color("gap")]
        for line, expected in zip(data_lines, expected_colors):
            assert line.get_color() == expected
        plt.close(fig)

    def test_line_names_the_winning_model_in_its_label(self):
        # silhouette's winner is Agglomerative, gap's is KMeans (see
        # _curves_and_best) -- the legend must say which, not just the
        # metric name.
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        labels = [ln.get_label() for ln in ax.lines if ln.get_marker() == "o"]
        assert any("Agglomerative" in label for label in labels)
        assert any("KMeans" in label for label in labels)
        plt.close(fig)

    def test_line_length_matches_one_models_k_values_not_every_row(self):
        # Each metric has six rows in curves_df (two models x three k's). A
        # line drawn from every row for that metric, instead of only the
        # winning model's three, would be six points long.
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        for line in data_lines:
            assert len(line.get_xdata()) == 3
        plt.close(fig)

    def test_line_x_values_are_monotonic(self):
        # Grouping by metric alone and sorting by k (the interleaving bug)
        # produces k = 3, 3, 4, 4, 5, 5 -- ties, not a strictly increasing
        # sweep. One model's own k values are strictly increasing.
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        for line in data_lines:
            x = line.get_xdata()
            assert np.all(np.diff(x) > 0)
        plt.close(fig)

    def test_selected_k_marker_uses_the_winning_models_own_score(self):
        # best_df's gap winner is KMeans at k=5, the top of KMeans' own
        # curve (normalized value 1.0). curves_df lists Agglomerative's rows
        # before KMeans', so sorting the metric's *combined* rows by k puts
        # Agglomerative's k=5 row ahead of KMeans': a best-k lookup that
        # matched the first row at k=5 regardless of model -- the defect
        # this guards against -- would mark Agglomerative's score (a
        # combined-normalization value of about 0.33) instead of KMeans'.
        fig, ax = plt.subplots()
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        gap_marker = next(
            c
            for c in ax.collections
            if c.get_offsets().shape[0] == 1 and float(c.get_offsets()[0, 0]) == 5.0
        )
        marked_y = float(gap_marker.get_offsets()[0, 1])
        assert marked_y == pytest.approx(1.0)
        plt.close(fig)


class TestAlluvial:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        y_true = np.repeat([0, 1], 20)
        left = np.repeat([0, 1], 20)
        right = np.repeat([1, 0], 20)
        result = alluvial(
            ax,
            y_true,
            left,
            right,
            left_cmap=cluster_color_map(left),
            right_cmap=cluster_color_map(right),
            true_cmap=cluster_color_map(y_true),
            left_title="CARVE",
            right_title="CVI",
            true_title="Reported",
        )
        assert result is ax
        plt.close(fig)

    def test_draws_the_three_column_titles(self):
        fig, ax = plt.subplots()
        y_true = np.repeat([0, 1], 20)
        alluvial(
            ax,
            y_true,
            y_true,
            y_true,
            left_cmap=cluster_color_map(y_true),
            right_cmap=cluster_color_map(y_true),
            true_cmap=cluster_color_map(y_true),
            left_title="CARVE",
            right_title="CVI",
            true_title="Reported",
        )
        texts = [t.get_text() for t in ax.texts]
        assert "CARVE" in texts and "CVI" in texts and "Reported" in texts
        plt.close(fig)

    def test_column_bars_use_the_supplied_colormaps(self):
        # left and right deliberately assign different colors to the same
        # category (0/1), so a bug that reused one column's cmap for
        # another's bars would be caught here.
        fig, ax = plt.subplots()
        y_true = np.repeat([0, 1], 20)
        left = np.repeat([0, 1], 20)
        right = np.repeat([1, 0], 20)
        left_cmap = cluster_color_map(left)
        right_cmap = cluster_color_map(right)
        true_cmap = cluster_color_map(y_true)
        alluvial(
            ax,
            y_true,
            left,
            right,
            left_cmap=left_cmap,
            right_cmap=right_cmap,
            true_cmap=true_cmap,
            left_title="CARVE",
            right_title="CVI",
            true_title="Reported",
        )
        rectangles = list(ax.patches)
        assert len(rectangles) == 6
        # Column order is left, true, right; within a column, categories
        # appear in first-occurrence order (left/true: [0, 1], right: [1, 0]).
        expected = [
            left_cmap[0],
            left_cmap[1],
            true_cmap[0],
            true_cmap[1],
            right_cmap[1],
            right_cmap[0],
        ]
        for rect, expected_color in zip(rectangles, expected):
            actual_rgb = mcolors.to_rgba(rect.get_facecolor())[:3]
            expected_rgb = mcolors.to_rgba(expected_color)[:3]
            assert np.allclose(actual_rgb, expected_rgb)
        plt.close(fig)

    def test_flow_count_and_color_match_the_transition_structure(self):
        # left equals y_true (no left/true mixing: 2 non-zero transitions),
        # right is the flip of y_true (2 non-zero transitions the other
        # way). A bug that drew every category pair regardless of overlap
        # would produce 8 ribbons instead of 4; one that used the wrong
        # source colormap for a boundary would fail the color check.
        fig, ax = plt.subplots()
        y_true = np.array([0, 0, 1, 1])
        left = np.array([0, 0, 1, 1])
        right = np.array([1, 1, 0, 0])
        left_cmap = {0: "#123456", 1: "#654321"}
        true_cmap = {0: "#ABCDEF", 1: "#FEDCBA"}
        right_cmap = {0: "#111111", 1: "#222222"}
        alluvial(
            ax,
            y_true,
            left,
            right,
            left_cmap=left_cmap,
            right_cmap=right_cmap,
            true_cmap=true_cmap,
            left_title="CARVE",
            right_title="CVI",
            true_title="Reported",
        )
        assert len(ax.collections) == 4
        actual = [
            mcolors.to_rgba(coll.get_facecolor()[0])[:3] for coll in ax.collections
        ]
        expected = [
            mcolors.to_rgba(left_cmap[0])[:3],
            mcolors.to_rgba(left_cmap[1])[:3],
            mcolors.to_rgba(true_cmap[0])[:3],
            mcolors.to_rgba(true_cmap[1])[:3],
        ]
        for actual_rgb, expected_rgb in zip(actual, expected):
            assert np.allclose(actual_rgb, expected_rgb)
        plt.close(fig)


class TestAriLollipop:
    def test_returns_the_same_axes(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame(
            {
                "method": ["CARVE (stab)", "Silhouette"],
                "ari": [0.78, 0.63],
                "k": [10, 7],
            }
        )
        assert ari_lollipop(ax, df) is ax
        plt.close(fig)

    def test_draws_one_marker_per_method(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame(
            {"method": ["a", "b", "c"], "ari": [0.1, 0.2, 0.3], "k": [3, 4, 5]}
        )
        ari_lollipop(ax, df, annotate_k=False)
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_annotates_k_when_asked(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame({"method": ["a"], "ari": [0.5], "k": [9]})
        ari_lollipop(ax, df, annotate_k=True)
        assert any("9" in t.get_text() for t in ax.texts)
        plt.close(fig)

    def test_hlines_and_markers_use_metric_colors_in_sorted_order(self):
        # method and metric disagree on which name maps to a recognized
        # theme color here, so matching colors correctly proves they come
        # from "metric", not "method", when both columns are present.
        fig, ax = plt.subplots()
        df = pd.DataFrame(
            {
                "method": ["CARVE (stab)", "Silhouette", "Gap Stat"],
                "metric": ["ari_stability_1se", "silhouette", "gap"],
                "ari": [0.8, 0.3, 0.5],
                "k": [10, 4, 6],
            }
        )
        ari_lollipop(ax, df, annotate_k=False)
        line_collection, marker_collection = ax.collections[:2]
        hline_colors = line_collection.get_color()
        marker_colors = marker_collection.get_facecolor()
        # sorted ascending by ari: silhouette (0.3), gap (0.5), then
        # ari_stability_1se (0.8).
        expected = [
            metric_color("silhouette"),
            metric_color("gap"),
            metric_color("ari_stability_1se"),
        ]
        for i, expected_color in enumerate(expected):
            expected_rgb = mcolors.to_rgba(expected_color)[:3]
            assert np.allclose(hline_colors[i][:3], expected_rgb)
            assert np.allclose(marker_colors[i][:3], expected_rgb)
        plt.close(fig)

    def test_hlines_and_markers_fall_back_to_method_colors_when_metric_absent(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame(
            {"method": ["gap", "silhouette"], "ari": [0.5, 0.3], "k": [6, 4]}
        )
        ari_lollipop(ax, df, annotate_k=False)
        line_collection, marker_collection = ax.collections[:2]
        hline_colors = line_collection.get_color()
        marker_colors = marker_collection.get_facecolor()
        # sorted ascending by ari: silhouette (0.3) then gap (0.5).
        expected = [metric_color("silhouette"), metric_color("gap")]
        for i, expected_color in enumerate(expected):
            expected_rgb = mcolors.to_rgba(expected_color)[:3]
            assert np.allclose(hline_colors[i][:3], expected_rgb)
            assert np.allclose(marker_colors[i][:3], expected_rgb)
        plt.close(fig)

    def test_a_metric_only_frame_fails_after_drawing_not_before(self):
        # "method" is still required later for the y-tick labels, so this
        # frame can never fully succeed -- but the KeyError must come from
        # that line, not from eagerly evaluating ordered["method"] as the
        # unused default for .get("metric", ...) before hlines/scatter
        # ever run. Under the old eager-default bug this raised before
        # either collection existed; both existing here proves the fix.
        fig, ax = plt.subplots()
        df = pd.DataFrame({"metric": ["silhouette", "gap"], "ari": [0.3, 0.5]})
        with pytest.raises(KeyError, match="method"):
            ari_lollipop(ax, df, annotate_k=False)
        assert len(ax.collections) == 2
        plt.close(fig)


class TestGroupedLegend:
    def test_returns_a_legend_attached_to_the_figure(self):
        fig, axes = plt.subplots(1, 2, squeeze=False)
        for ax in axes.flat:
            ax.plot([0, 1], [0, 1], label="CARVE Stability (1SE)")
        legend = grouped_legend(fig, axes)
        assert isinstance(legend, Legend)
        plt.close(fig)

    def test_deduplicates_repeated_labels(self):
        fig, axes = plt.subplots(1, 3, squeeze=False)
        for ax in axes.flat:
            ax.plot([0, 1], [0, 1], label="Silhouette")
        legend = grouped_legend(fig, axes)
        assert len(legend.get_texts()) == 1
        plt.close(fig)


class TestPanelLetter:
    def test_adds_one_text_artist(self):
        fig, ax = plt.subplots()
        panel_letter(ax, "A")
        assert [t.get_text() for t in ax.texts] == ["A"]
        plt.close(fig)

    def test_uses_the_theme_font_size(self):
        from benchmarks._theme import FONT_SIZES

        fig, ax = plt.subplots()
        panel_letter(ax, "B")
        assert ax.texts[0].get_fontsize() == FONT_SIZES["panel_letter"]
        plt.close(fig)
