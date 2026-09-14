"""Tests for the panel drawing primitives.

The uniform ax-in / ax-out contract is the reason this module exists, so it
is asserted directly rather than left to convention.
"""

import inspect
from types import SimpleNamespace

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.legend import Legend
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from benchmarks import _panels
from benchmarks._artifacts import SCHEMA
from benchmarks._panels import (
    _display,
    _legend_groups,
    _stack_segments,
    aligned_color_maps,
    alluvial,
    ari_lollipop,
    axis_arrows,
    carve_lines,
    cluster_color_map,
    cvi_lines,
    grouped_legend,
    metric_legend,
    metric_lines,
    panel_letter,
    pipeline_lines,
    runtime_lines,
    scatter_clusters,
)
from benchmarks._theme import (
    FONT_SIZES,
    FOREGROUND_COLOR,
    PIPELINE_COLORS,
    cluster_colors,
    metric_color,
)
from tests.benchmarks._helpers import StubCarve, pipeline_results, resolution_results


@pytest.fixture
def ax():
    """A fresh Axes; the conftest fixture closes every figure afterwards."""
    _, ax = plt.subplots()
    return ax


AX_FIRST_FUNCTIONS = (
    "scatter_clusters",
    "axis_arrows",
    "metric_lines",
    "carve_lines",
    "pipeline_lines",
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
                            # Varies by seed (not by axis_value or metric) so
                            # a baseline computation that fails to
                            # deduplicate by seed before averaging can be
                            # told apart from one that does: the mean is the
                            # same either way, but the standard error is not.
                            "oracle_ari": 0.95 + 0.01 * seed,
                        }
                    )
    return pd.DataFrame(rows)[list(SCHEMA)]


class TestScatterClusters:
    def test_returns_the_same_axes(self, ax):
        Z = np.random.default_rng(0).normal(size=(30, 2))
        labels = np.repeat([0, 1, 2], 10)
        assert scatter_clusters(ax, Z, labels) is ax

    def test_draws_one_collection_per_label(self, ax):
        Z = np.random.default_rng(0).normal(size=(30, 2))
        labels = np.repeat([0, 1, 2], 10)
        scatter_clusters(ax, Z, labels)
        assert len(ax.collections) == 3

    def test_hides_axes_when_asked(self, ax):
        Z = np.random.default_rng(0).normal(size=(10, 2))
        scatter_clusters(ax, Z, np.zeros(10, dtype=int), hide_axes=True)
        assert list(ax.get_xticks()) == []

    def test_uses_cluster_color_map(self, ax):
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

    def test_markers_carry_a_thin_outline(self, ax):
        # The outline is what keeps overlapping points readable where two
        # clusters meet; without it a dense scatter merges into one mass.
        Z = np.random.default_rng(0).normal(size=(10, 2))
        scatter_clusters(ax, Z, np.zeros(10, dtype=int))
        edges = ax.collections[0].get_edgecolor()
        assert len(edges) == 1
        assert np.allclose(edges[0][:3], mcolors.to_rgba(FOREGROUND_COLOR)[:3])
        assert ax.collections[0].get_linewidth()[0] > 0

    def test_outline_can_be_turned_off(self, ax):
        Z = np.random.default_rng(0).normal(size=(10, 2))
        scatter_clusters(ax, Z, np.zeros(10, dtype=int), edgecolor="none")
        assert len(ax.collections[0].get_edgecolor()) == 0

    def test_uses_fallback_for_missing_color(self, ax):
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


class TestClusterColorMap:
    def test_assigns_one_color_per_distinct_label(self):
        mapping = cluster_color_map(np.array([2, 2, 0, 1, 1]))
        assert set(mapping) == {0, 1, 2}
        assert len(set(mapping.values())) == 3


class TestMetricLines:
    def test_returns_the_same_axes(self, ax):
        assert metric_lines(ax, _results_frame(), metrics=("silhouette",)) is ax

    def test_draws_one_series_per_requested_metric(self, ax):
        metric_lines(ax, _results_frame(), metrics=("ari_stability_1se", "silhouette"))
        assert len(ax.get_legend().get_texts()) == 2

    def test_uses_only_selected_rows(self, ax):
        metric_lines(ax, _results_frame(), metrics=("silhouette",), show_legend=False)
        line = ax.lines[0]
        # three axis values, one point each, drawn from the k=5 rows only
        assert len(line.get_xdata()) == 3

    def test_skips_a_metric_with_no_rows(self, ax):
        metric_lines(ax, _results_frame(), metrics=("gap",), show_legend=False)
        assert len(ax.lines) == 0

    def test_lines_use_metric_colors(self, ax):
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

    def test_draws_the_oracle_baseline_from_oracle_ari_not_ari_at_k(self, ax):
        metric_lines(
            ax, _results_frame(), metrics=("baseline_oracle",), show_legend=False
        )
        baseline_lines = [ln for ln in ax.lines if ln.get_marker() == "none"]
        assert len(baseline_lines) == 1
        # _results_frame's oracle_ari averages to 0.96 across seeds
        # (0.95, 0.96, 0.97) at every axis value, unlike ari_at_k (which
        # varies with axis_value) -- if the baseline branch mistakenly read
        # ari_at_k this would come out as [0.9, 0.8, 0.7] instead.
        np.testing.assert_allclose(baseline_lines[0].get_ydata(), 0.96)

    def test_oracle_baseline_uses_the_theme_color_and_is_dashed(self, ax):
        metric_lines(
            ax, _results_frame(), metrics=("baseline_oracle",), show_legend=False
        )
        line = ax.lines[0]
        assert line.get_color() == metric_color("baseline_oracle")
        assert line.get_linestyle() == "--"

    def test_oracle_baseline_error_bars_reflect_seed_level_spread(self, ax):
        # oracle_ari repeats each seed's value across every (metric, k) row
        # within a cell (four rows per seed here). A baseline branch that
        # grouped those raw rows instead of deduplicating by (axis_value,
        # seed) first would compute its standard error over an artificially
        # inflated sample -- the mean would still come out right, but the
        # error bar would be too tight.
        df = _results_frame()
        metric_lines(ax, df, metrics=("baseline_oracle",), show_legend=False)

        container = ax.containers[0]
        y_error_segments = container.lines[2][0].get_segments()
        drawn_yerr = float(y_error_segments[0][1][1] - y_error_segments[0][0][1]) / 2.0

        seed_level_sem = (
            df[["axis_value", "seed", "oracle_ari"]]
            .drop_duplicates()
            .groupby("axis_value")["oracle_ari"]
            .sem()
        )
        assert drawn_yerr == pytest.approx(float(seed_level_sem.iloc[0]), rel=1e-6)


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
    def test_returns_the_same_axes_and_uses_a_log_scale(self, ax):
        assert runtime_lines(ax, _runtime_frame()) is ax
        assert ax.get_yscale() == "log"

    def test_draws_both_mode_curves_by_default(self, ax):
        runtime_lines(ax, _runtime_frame())
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["CARVE Stability", "CARVE Generalizability"]

    def test_the_two_curves_are_dodged_apart(self, ax):
        runtime_lines(ax, _runtime_frame())
        # errorbar creates multiple lines per call; find the main data lines (with markers)
        data_lines = [line for line in ax.lines if line.get_marker() == "o"]
        assert len(data_lines) >= 2
        first, second = data_lines[0].get_xdata(), data_lines[1].get_xdata()
        assert not np.allclose(first, second)

    def test_skips_a_column_that_is_all_nan(self, ax):
        """Untimed cells record nan, and an untimed series must not be drawn."""
        df = _runtime_frame()
        df["t_generalizability_s"] = np.nan
        runtime_lines(ax, df)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["CARVE Stability"]

    def test_skips_a_column_that_is_absent(self, ax):
        runtime_lines(ax, _runtime_frame()[["axis_value", "t_stability_s"]])
        assert len(ax.lines) >= 1

    def test_rejects_mismatched_columns_and_labels(self, ax):
        with pytest.raises(ValueError, match="same length"):
            runtime_lines(ax, _runtime_frame(), runtime_cols=("t_stability_s",))

    def test_lines_use_correct_mode_colors(self, ax):
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
    selection = {
        "stability": {"method_id": "m0", "n_clusters": 4},
        "generalizability": {"method_id": "m1", "n_clusters": 5},
    }
    return StubCarve(
        results,
        select=lambda measure, not_two: (
            selection[measure]["method_id"],
            selection[measure]["n_clusters"],
        ),
    )


class TestCarveLines:
    def test_returns_the_same_axes(self, ax):
        assert carve_lines(ax, _carve_obj()) is ax

    def test_draws_one_line_per_requested_measure(self, ax):
        # A non-default, single-element tuple: if the measures argument were
        # ignored in favor of the ("stability", "generalizability") default,
        # this would draw two lines instead of one.
        carve_lines(ax, _carve_obj(), measures=("generalizability",))
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 1

    def test_lines_use_measure_specific_colors(self, ax):
        # Measures are requested in the opposite order from the default
        # tuple. A regression that fell back to the default order rather
        # than the caller's order would draw the colors in the wrong slots.
        carve_lines(ax, _carve_obj(), measures=("generalizability", "stability"))
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        expected_colors = [
            metric_color("ari_generalizability_1se"),
            metric_color("ari_stability_1se"),
        ]
        for line, expected in zip(data_lines, expected_colors):
            assert line.get_color() == expected

    def test_shows_the_selected_k_by_default(self, ax):
        carve_lines(ax, _carve_obj(), measures=("stability",))
        marker_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        other_lines = [ln for ln in ax.lines if ln.get_marker() != "o"]
        assert len(marker_lines) == 1
        assert len(other_lines) == 1
        assert other_lines[0].get_xdata()[0] == 4

    def test_omits_the_selected_k_marker_when_asked(self, ax):
        carve_lines(ax, _carve_obj(), measures=("stability",), show_selected_k=False)
        assert len(ax.lines) == 1

    def test_line_length_matches_one_configurations_k_values_not_every_row(self, ax):
        # _carve_obj's estimator_results_ carries eight rows -- two method
        # ids swept over four k values each. A line that read every row
        # instead of filtering to the selected method_id would be eight
        # points long, not four.
        carve_lines(ax, _carve_obj(), measures=("stability",), show_selected_k=False)
        line = ax.lines[0]
        assert len(line.get_xdata()) == 4

    def test_line_x_values_are_monotonic(self, ax):
        # Concatenating both method_ids' k ranges end to end (the
        # interleaving bug) draws k = 3, 4, 5, 6, 3, 4, 5, 6 -- a line that
        # runs up and then jumps back down. A single configuration's own
        # sweep is strictly increasing.
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

    def test_a_k_based_run_keeps_the_number_of_clusters_axis(self, ax):
        carve_lines(ax, _carve_obj(), measures=("stability",))
        assert ax.get_xlabel() == "Number of clusters $k$"

    def test_a_resolution_run_draws_over_resolution(self, ax):
        # A decoy configuration at other resolutions must not join the line,
        # and the selected resolution, 0.6, comes from get_sweep_value: get_k
        # would round it to a cluster count.
        results = pd.concat(
            [
                resolution_results([0.2, 0.4, 0.6, 0.8], "Leiden"),
                resolution_results([0.3, 0.5], "Leiden decoy", method_id="m1"),
            ],
            ignore_index=True,
        )
        carve = StubCarve(
            results, select=lambda measure, not_two: ("m0", 0.6), sweep_param="resolution"
        )
        carve_lines(ax, carve, measures=("stability",))
        data_line = next(ln for ln in ax.lines if ln.get_marker() == "o")
        selection = next(ln for ln in ax.lines if ln.get_marker() != "o")
        np.testing.assert_allclose(data_line.get_xdata(), [0.2, 0.4, 0.6, 0.8])
        assert selection.get_xdata()[0] == pytest.approx(0.6)
        assert ax.get_xlabel() == "Resolution"

    def test_forwards_rule_and_not_two_to_every_selection(self, ax):
        carve = _carve_obj()
        carve_lines(ax, carve, rule="max", not_two=True)
        assert len(carve.selection_calls) == 4
        assert all(
            call["rule"] == "max" and call["not_two"] for call in carve.selection_calls
        )


_PIPELINES = (
    "identity | identity",
    "identity | TSNE(perplexity=30)",
    "identity | UMAP(n_neighbors=15)",
)


class TestPipelineLines:
    @staticmethod
    def _draw(ax, **kwargs):
        table = pipeline_results(_PIPELINES, (0.2, 0.4, 0.6), method_ids=("m0", "m1"))
        carve = SimpleNamespace(preprocessing_results_=table)
        return pipeline_lines(ax, carve, method_id="m0", **kwargs)

    def test_returns_the_same_axes(self, ax):
        assert self._draw(ax) is ax

    def test_one_line_pair_per_pipeline_sharing_a_pipeline_color(self, ax):
        self._draw(ax)
        drawn: dict[str, list] = {}
        for container in ax.containers:
            line = container.lines[0]
            drawn.setdefault(container.get_label(), []).append(
                (mcolors.to_hex(line.get_color()), line.get_linestyle())
            )
        assert set(drawn) == set(_PIPELINES)
        for pair in drawn.values():
            assert [style for _, style in pair] == ["-", "--"]
            assert len({color for color, _ in pair}) == 1
        colors = {pair[0][0] for pair in drawn.values()}
        assert len(colors) == len(_PIPELINES)
        assert colors <= {color.lower() for color in PIPELINE_COLORS}

    def test_draws_only_the_named_configuration(self, ax):
        # m1 scores 0.3 higher everywhere; pipeline 0 of m0 scores
        # 0.5, 0.45, 0.4 on stability.
        self._draw(ax)
        stability = next(
            container
            for container in ax.containers
            if container.get_label() == "identity | identity"
        )
        np.testing.assert_allclose(stability.lines[0].get_ydata(), [0.5, 0.45, 0.4])

    def test_one_legend_names_the_pipelines_and_the_two_criteria(self, ax):
        self._draw(ax)
        texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert texts == [*sorted(_PIPELINES), "Stability", "Generalizability"]

    def test_labels_the_sweep_axis_and_the_criterion(self, ax):
        self._draw(ax, title="By pipeline")
        assert ax.get_xlabel() == "Resolution"
        assert ax.get_ylabel() == "ARI"
        assert ax.get_title() == "By pipeline"

    def test_forwards_the_selection_arguments(self, ax, monkeypatch):
        import carve._plotting as carve_plotting

        calls = []
        real = carve_plotting.plot_metric_by_pipeline

        def spy(*args, **kwargs):
            calls.append(
                (kwargs["measure"], kwargs["rule"], kwargs["not_two"], kwargs["method_id"])
            )
            return real(*args, **kwargs)

        monkeypatch.setattr(carve_plotting, "plot_metric_by_pipeline", spy)
        self._draw(ax, rule="max", not_two=True)
        assert calls == [
            ("stability", "max", True, "m0"),
            ("generalizability", "max", True, "m0"),
        ]


class TestCviLines:
    def test_returns_the_same_axes(self, ax):
        curves, best = _curves_and_best()
        assert cvi_lines(ax, curves, best) is ax

    def test_draws_one_line_per_metric(self, ax):
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        assert len([ln for ln in ax.lines if ln.get_label() != "_nolegend_"]) >= 2

    def test_marks_the_selected_k_for_each_metric(self, ax):
        """One dashed rule per metric, standing at its own selected k.

        The ring these replace sat on the curve at (k, score), so it read as
        another data point and disappeared wherever two indices crossed.
        """
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        rules = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
        assert len(rules) == len(best)
        assert {float(ln.get_xdata()[0]) for ln in rules} == set(
            best["k"].astype(float)
        )

    def test_selection_rules_take_their_own_metric_color(self, ax):
        # silhouette selects k=4 and gap k=5 (see _curves_and_best), so a
        # rule drawn in the other metric's color would be caught here.
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        rules = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
        color_at_k = {float(ln.get_xdata()[0]): ln.get_color() for ln in rules}
        for _, row in best.iterrows():
            assert color_at_k[float(row["k"])] == metric_color(str(row["metric"]))

    def test_no_ring_markers_are_drawn_on_the_curves(self, ax):
        # The rings were scatter collections; nothing should draw them now.
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        assert list(ax.collections) == []

    def test_lines_use_metric_specific_colors(self, ax):
        # silhouette and gap map to different theme colors, so a bug that
        # mixed up which curve gets which color would be caught here.
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        expected_colors = [metric_color("silhouette"), metric_color("gap")]
        for line, expected in zip(data_lines, expected_colors):
            assert line.get_color() == expected

    def test_line_names_the_winning_model_in_its_label(self, ax):
        # silhouette's winner is Agglomerative, gap's is KMeans (see
        # _curves_and_best) -- the legend must say which, not just the
        # metric name.
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        labels = [ln.get_label() for ln in ax.lines if ln.get_marker() == "o"]
        assert any("Agglomerative" in label for label in labels)
        assert any("KMeans" in label for label in labels)

    def test_line_length_matches_one_models_k_values_not_every_row(self, ax):
        # Each metric has six rows in curves_df (two models x three k's). A
        # line drawn from every row for that metric, instead of only the
        # winning model's three, would be six points long.
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        for line in data_lines:
            assert len(line.get_xdata()) == 3

    def test_line_x_values_are_monotonic(self, ax):
        # Grouping by metric alone and sorting by k (the interleaving bug)
        # produces k = 3, 3, 4, 4, 5, 5 -- ties, not a strictly increasing
        # sweep. One model's own k values are strictly increasing.
        curves, best = _curves_and_best()
        cvi_lines(ax, curves, best)
        data_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2
        for line in data_lines:
            x = line.get_xdata()
            assert np.all(np.diff(x) > 0)

class TestAlluvial:
    def test_returns_the_same_axes(self, ax):
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

    def test_draws_the_three_column_titles(self, ax):
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

    def test_column_bars_use_the_supplied_colormaps(self, ax):
        # left and right deliberately assign different colors to the same
        # category (0/1), so a bug that reused one column's cmap for
        # another's bars would be caught here.
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
        rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(rectangles) == 6
        # Column order is left, true, right. The truth column stacks in
        # first-occurrence order ([0, 1]); each cluster column is then
        # ordered against it by _order_by_reference, which puts left at
        # [0, 1] and right -- the flip of y_true -- at [1, 0].
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

    def test_flow_count_and_color_match_the_transition_structure(self, ax):
        # left equals y_true (no left/true mixing: 2 non-zero transitions),
        # right is the flip of y_true (2 non-zero transitions the other
        # way). A bug that drew every category pair regardless of overlap
        # would produce 8 ribbons instead of 4.
        #
        # Every ribbon takes its *reported label's* color on both sides of
        # the truth column -- that is what lets one reported class be
        # followed across the whole panel -- so neither left_cmap nor
        # right_cmap may appear among the ribbons, only true_cmap.
        y_true = np.array([0, 0, 1, 1])
        # left is the *flip* of y_true, not a copy: with a copy, colouring a
        # ribbon by its source cluster and by its target label give the same
        # answer, and this test cannot tell the two apart.
        left = np.array([1, 1, 0, 0])
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
        ribbons = [p for p in ax.patches if isinstance(p, PathPatch)]
        assert len(ribbons) == 4
        drawn = [
            tuple(np.round(mcolors.to_rgba(r.get_facecolor())[:3], 3)) for r in ribbons
        ]
        expected = [
            tuple(np.round(mcolors.to_rgba(true_cmap[c])[:3], 3)) for c in (0, 1, 0, 1)
        ]
        assert drawn == expected

    def test_ribbons_are_curved_not_straight(self, ax):
        """Panel F reads as flow, which is the Bezier control points.

        A straight-edged band -- what this replaces -- is a four-vertex
        polygon; each ribbon here is a nine-vertex path whose edges are
        cubic curves. Asserting the curve codes rather than the vertex
        count alone means a path that merely gained vertices would not pass.
        """
        y_true = np.repeat([0, 1], 10)
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
        ribbons = [p for p in ax.patches if isinstance(p, PathPatch)]
        assert ribbons
        for ribbon in ribbons:
            assert (ribbon.get_path().codes == Path.CURVE4).sum() == 6

    def test_cluster_bars_are_named_and_carry_purity(self, ax):
        # left cluster 0 is pure (all y_true 0); cluster 1 splits 3/1, so
        # its purity is 75%. Names are one-based, so id 0 reads "C1".
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0])
        left = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        alluvial(
            ax,
            y_true,
            left,
            left,
            left_cmap=cluster_color_map(left),
            right_cmap=cluster_color_map(left),
            true_cmap=cluster_color_map(y_true),
            left_title="CARVE",
            right_title="CVI",
            true_title="Reported",
        )
        texts = [t.get_text() for t in ax.texts]
        assert "C1  100%" in texts
        assert "C2  75%" in texts

    def test_reported_labels_are_named_inside_their_own_bars(self, ax):
        y_true = np.array(["d0"] * 10 + ["d7"] * 10)
        left = np.repeat([0, 1], 10)
        alluvial(
            ax,
            y_true,
            left,
            left,
            left_cmap=cluster_color_map(left),
            right_cmap=cluster_color_map(left),
            true_cmap=cluster_color_map(y_true),
            left_title="CARVE",
            right_title="CVI",
            true_title="Reported",
        )
        centered = [t.get_text() for t in ax.texts if t.get_ha() == "center"]
        assert "d0" in centered and "d7" in centered

    def test_truth_column_is_stacked_more_loosely_than_the_cluster_columns(self, ax):
        """The middle column is the anchor and is set apart by its gaps.

        Equal sizes everywhere, so any difference in bar height comes from
        the gap fraction alone: the truth column gives up more of its
        height to gaps, so each of its bars is shorter.
        """
        y_true = np.repeat([0, 1, 2], 10)
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
        bars = [p for p in ax.patches if isinstance(p, Rectangle)]
        left_heights = [b.get_height() for b in bars[:3]]
        true_heights = [b.get_height() for b in bars[3:6]]
        assert max(true_heights) < min(left_heights)

    def test_clusters_are_ordered_against_the_reported_column(self, ax):
        """A cluster is stacked where its mass sits in the truth column.

        Cluster ids here run opposite to the truth column's order, so
        stacking by id -- what this replaces -- would send every ribbon
        across the full height of the panel. The bars must come out in
        reverse id order instead.
        """
        y_true = np.repeat([0, 1], 10)
        left = np.repeat([1, 0], 10)
        left_cmap = {0: "#111111", 1: "#222222"}
        alluvial(
            ax,
            y_true,
            left,
            left,
            left_cmap=left_cmap,
            right_cmap=left_cmap,
            true_cmap=cluster_color_map(y_true),
            left_title="CARVE",
            right_title="CVI",
            true_title="Reported",
        )
        bars = [p for p in ax.patches if isinstance(p, Rectangle)]
        top_left_bar = max(bars[:2], key=lambda b: b.get_y())
        assert np.allclose(
            mcolors.to_rgba(top_left_bar.get_facecolor())[:3],
            mcolors.to_rgba(left_cmap[1])[:3],
        )


class TestStackSegments:
    def test_stacks_downward_from_the_top(self):
        # First entry at the top is what makes a column's reading order
        # match the order the caller passed it.
        spans = _stack_segments([1, 1, 1])
        assert spans[0][1] == pytest.approx(1.0)
        assert spans[0][0] > spans[1][0] > spans[2][0]

    def test_gaps_come_out_of_the_available_height(self):
        gapless = _stack_segments([1, 1], gap_frac=0.0)
        gapped = _stack_segments([1, 1], gap_frac=0.2)
        assert sum(t - b for b, t in gapless) == pytest.approx(1.0)
        assert sum(t - b for b, t in gapped) == pytest.approx(0.8)

    def test_empty_input_returns_no_spans(self):
        assert _stack_segments([]) == []
        assert _stack_segments([0, 0]) == []


class TestAlignedColorMaps:
    def test_reported_labels_take_the_palette_in_sorted_order(self):
        (true_cmap,) = aligned_color_maps(np.array(["d7", "d0", "d2"]))
        assert list(true_cmap) == ["d0", "d2", "d7"]
        assert list(true_cmap.values()) == cluster_colors(3)

    def test_a_cluster_shares_its_reported_label_color(self):
        # The invariant the composite's three scatter panels rest on: an
        # aligned cluster id and the reported label it was matched to are
        # the same color, in every panel.
        y_true = np.array(["a", "a", "b", "b", "c", "c"])
        labels = np.array([0, 0, 1, 1, 2, 2])
        true_cmap, cluster_cmap_ = aligned_color_maps(y_true, labels)
        assert cluster_cmap_[0] == true_cmap["a"]
        assert cluster_cmap_[1] == true_cmap["b"]
        assert cluster_cmap_[2] == true_cmap["c"]

    def test_palette_covers_more_clusters_than_reported_labels(self):
        # A clustering may overshoot the reported label count; every id it
        # uses still has to resolve to a color.
        y_true = np.array(["a", "a", "b", "b"])
        labels = np.array([0, 1, 2, 3])
        _, cluster_cmap_ = aligned_color_maps(y_true, labels)
        assert set(cluster_cmap_) == {0, 1, 2, 3}
        assert len(set(cluster_cmap_.values())) == 4

    def test_returns_one_map_per_clustering_plus_the_reported_one(self):
        y_true = np.array([0, 0, 1, 1])
        maps = aligned_color_maps(y_true, y_true, y_true, y_true)
        assert len(maps) == 4


class TestAxisArrows:
    def test_returns_the_same_axes(self, ax):
        assert axis_arrows(ax) is ax

    def test_labels_the_two_directions(self, ax):
        axis_arrows(ax, ("t-SNE 1", "t-SNE 2"))
        texts = [t.get_text() for t in ax.texts]
        assert "t-SNE 1" in texts and "t-SNE 2" in texts

    def test_draws_two_arrows(self, ax):
        axis_arrows(ax)
        arrows = [t for t in ax.texts if t.arrow_patch is not None]
        assert len(arrows) == 2

    def test_position_is_in_axes_fractions_not_data_units(self, ax):
        # The marker has to stay put whatever the data limits are, which is
        # what xycoords="axes fraction" buys.
        axis_arrows(ax)
        assert all(
            t.xycoords == "axes fraction" for t in ax.texts if t.arrow_patch is not None
        )


class TestAriLollipop:
    def test_returns_the_same_axes(self, ax):
        df = pd.DataFrame(
            {
                "method": ["CARVE (stab)", "Silhouette"],
                "ari": [0.78, 0.63],
                "k": [10, 7],
            }
        )
        assert ari_lollipop(ax, df) is ax

    def test_draws_one_marker_per_method(self, ax):
        df = pd.DataFrame(
            {"method": ["a", "b", "c"], "ari": [0.1, 0.2, 0.3], "k": [3, 4, 5]}
        )
        ari_lollipop(ax, df, annotate_k=False)
        assert len(ax.collections) >= 1

    def test_annotates_k_when_asked(self, ax):
        df = pd.DataFrame({"method": ["a"], "ari": [0.5], "k": [9]})
        ari_lollipop(ax, df, annotate_k=True)
        assert any("9" in t.get_text() for t in ax.texts)

    def test_hlines_and_markers_use_metric_colors_in_sorted_order(self, ax):
        # method and metric disagree on which name maps to a recognized
        # theme color here, so matching colors correctly proves they come
        # from "metric", not "method", when both columns are present.
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

    def test_hlines_and_markers_fall_back_to_method_colors_when_metric_absent(self, ax):
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

    def test_a_metric_only_frame_fails_after_drawing_not_before(self, ax):
        # "method" is still required later for the y-tick labels, so this
        # frame can never fully succeed -- but the KeyError must come from
        # that line, not from eagerly evaluating ordered["method"] as the
        # unused default for .get("metric", ...) before hlines/scatter
        # ever run. Under the old eager-default bug this raised before
        # either collection existed; both existing here proves the fix.
        df = pd.DataFrame({"metric": ["silhouette", "gap"], "ari": [0.3, 0.5]})
        with pytest.raises(KeyError, match="method"):
            ari_lollipop(ax, df, annotate_k=False)
        assert len(ax.collections) == 2


class TestGroupedLegend:
    def test_returns_a_legend_attached_to_the_figure(self):
        fig, axes = plt.subplots(1, 2, squeeze=False)
        for ax in axes.flat:
            ax.plot([0, 1], [0, 1], label="CARVE Stability (1SE)")
        legend = grouped_legend(fig, axes)
        assert isinstance(legend, Legend)

    def test_deduplicates_repeated_labels(self):
        fig, axes = plt.subplots(1, 3, squeeze=False)
        for ax in axes.flat:
            ax.plot([0, 1], [0, 1], label="Silhouette")
        legend = grouped_legend(fig, axes)
        assert len(legend.get_texts()) == 1


class TestLegendGroups:
    """The grouping rule behind the paper figure's legend."""

    def test_splits_into_oracle_carve_and_a_two_column_index_family(self):
        groups = _legend_groups(
            (
                "baseline_oracle",
                "ari_stability_1se",
                "ari_generalizability_1se",
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "gap",
            )
        )
        assert groups == [
            [["baseline_oracle"]],
            [["ari_stability_1se", "ari_generalizability_1se"]],
            [
                ["silhouette", "davies_bouldin"],
                ["calinski_harabasz", "gap"],
            ],
        ]

    def test_classical_column_order_follows_the_callers_order(self):
        """The two index columns are set by how metrics is written.

        Passing the same four indices in a different order must move them
        between columns, or the caller has no way to control the layout and
        the docstring's promise is false.
        """
        groups = _legend_groups(
            ("silhouette", "gap", "davies_bouldin", "calinski_harabasz")
        )
        assert groups == [
            [["silhouette", "gap"], ["davies_bouldin", "calinski_harabasz"]]
        ]

    def test_groups_are_decided_by_metric_name_not_label_text(self):
        """ari_stability's display name is "ARI (stab, max)" -- no "CARVE".

        A label-matching rule would file it with the classical indices.
        """
        groups = _legend_groups(("ari_stability", "silhouette"))
        assert groups == [[["ari_stability"]], [["silhouette"]]]

    def test_absent_families_do_not_leave_empty_groups(self):
        assert _legend_groups(("ari_stability_1se",)) == [[["ari_stability_1se"]]]

    def test_two_indices_stay_in_one_column(self):
        """Splitting two in half gives one-entry columns beside a two-entry one."""
        groups = _legend_groups(
            ("ari_stability_1se", "ari_generalizability_1se", "silhouette", "gap")
        )
        assert groups == [
            [["ari_stability_1se", "ari_generalizability_1se"]],
            [["silhouette", "gap"]],
        ]

    def test_an_odd_number_of_indices_fills_the_first_column_first(self):
        groups = _legend_groups(("silhouette", "gap", "davies_bouldin"))
        assert groups == [[["silhouette", "gap"], ["davies_bouldin"]]]


class TestLegendLabels:
    def test_the_oracle_label_is_not_mathtext(self):
        """A "$k^\\star$" superscript is a smudge at legend size.

        The figure legend names the oracle with a literal asterisk, as the
        published figure does, while the table row label keeps the plain
        "Baseline (Oracle)" the manuscript prints.
        """
        label = _display("baseline_oracle")
        assert "$" not in label
        assert label == "Baseline (Oracle k*)"

    def test_the_table_display_name_is_left_alone(self):
        from benchmarks._registry import METRIC_DISPLAY_NAMES

        assert METRIC_DISPLAY_NAMES["baseline_oracle"] == "Baseline (Oracle)"

    def test_a_metric_without_an_override_falls_back_to_its_display_name(self):
        assert _display("silhouette") == "Silhouette"


class TestMetricLegend:
    def _figure(self, metrics):
        fig, axes = plt.subplots(1, 2, squeeze=False)
        for metric in metrics:
            axes[0, 0].plot([0, 1], [0, 1], label=_display(metric))
        return fig, axes

    def test_an_empty_column_stands_between_families_but_not_inside_one(self):
        """The gap inside the classical pair must stay the narrow gutter.

        A Legend has one gutter width, so the family separation is an empty
        column rather than a wider columnspacing -- which would push the two
        classical columns apart too. Reading the labels column by column,
        the blank columns must fall only on family boundaries.
        """
        metrics = (
            "baseline_oracle",
            "ari_stability_1se",
            "ari_generalizability_1se",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "gap",
        )
        fig, axes = self._figure(metrics)
        legend = metric_legend(fig, axes, metrics)
        labels = [t.get_text() for t in legend.get_texts()]
        # Legend._ncols is private; it was _ncol before matplotlib 3.6 and the
        # matplotlib>=3.9.4 floor makes the current name safe.
        rows = len(labels) // legend._ncols
        columns = [labels[i * rows : (i + 1) * rows] for i in range(legend._ncols)]
        blank = [index for index, c in enumerate(columns) if not any(c)]
        assert legend._ncols == 6
        assert blank == [1, 3]
        assert columns[4] == ["Silhouette", "Davies-Bouldin"]
        assert columns[5] == ["Calinski-Harabasz", "Gap Statistic"]

    def test_the_gutter_stays_narrow_so_the_empty_column_does_the_separating(self):
        metrics = ("ari_stability_1se", "silhouette", "gap")
        fig, axes = self._figure(metrics)
        legend = metric_legend(fig, axes, metrics)
        assert legend.columnspacing < 2.0

    def test_pads_short_columns_so_each_family_starts_its_own(self):
        metrics = (
            "baseline_oracle",
            "ari_stability_1se",
            "ari_generalizability_1se",
            "silhouette",
            "davies_bouldin",
        )
        fig, axes = self._figure(metrics)
        legend = metric_legend(fig, axes, metrics)
        labels = [t.get_text() for t in legend.get_texts()]
        # three families, so two separating columns
        assert legend._ncols == 5
        assert labels[0] == _display("baseline_oracle")
        assert labels[1] == ""

    def test_skips_metrics_that_were_never_drawn(self):
        """A frame missing a metric must not put a dead entry in the legend."""
        fig, axes = self._figure(("ari_stability_1se",))
        legend = metric_legend(fig, axes, ("ari_stability_1se", "silhouette"))
        labelled = [t.get_text() for t in legend.get_texts() if t.get_text()]
        assert labelled == [_display("ari_stability_1se")]

    def test_a_family_with_nothing_drawn_leaves_no_stray_empty_column(self):
        fig, axes = self._figure(("ari_stability_1se",))
        legend = metric_legend(fig, axes, ("ari_stability_1se", "silhouette"))
        assert legend._ncols == 1

    def test_raises_when_nothing_requested_was_drawn(self):
        fig, axes = plt.subplots(1, 1, squeeze=False)
        with pytest.raises(ValueError, match="None of the requested metrics"):
            metric_legend(fig, axes, ("silhouette",))


class TestPanelLetter:
    def test_adds_one_text_artist(self, ax):
        panel_letter(ax, "A")
        assert [t.get_text() for t in ax.texts] == ["A"]

    def test_uses_the_theme_font_size(self, ax):
        panel_letter(ax, "B")
        assert ax.texts[0].get_fontsize() == FONT_SIZES["panel_letter"]
