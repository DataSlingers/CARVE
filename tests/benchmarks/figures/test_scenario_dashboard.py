"""Tests for the per-scenario dashboard."""

import matplotlib

matplotlib.use("Agg")

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from benchmarks._panels import (
    CRITERION_REFERENCE_LABEL,
    k_hat_selection_matrix,
    normalized_criterion,
)
from benchmarks._registry import BASELINE_METRIC, METRIC_DISPLAY_NAMES, SCENARIOS
from benchmarks.figures import figure_scenario_dashboard
from benchmarks.figures._benchmarking_results import DEFAULT_METRICS
from benchmarks.figures._scaling import SCALING_PANELS

# The metrics rows 3 and 4 draw: the oracle has no selected k and no
# criterion, so neither row carries it.
DRAWN_METRICS = [m for m in DEFAULT_METRICS if m != BASELINE_METRIC]


@pytest.fixture
def gaussians_frame():
    """A complete three-anchor frame for gaussians, two datasets each."""
    from tests.benchmarks._helpers import synthetic_run_frame

    return synthetic_run_frame("gaussians", n_seeds=2)


class TestLayout:
    def test_four_rows_of_panels(self, gaussians_frame):
        """Examples, outcome, selection anatomy, k-recovery. The example row
        and the two per-anchor rows carry one panel per axis point; the ARI
        row spans the width."""
        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        n_points = len(SCENARIOS["gaussians"].axis)
        assert len(fig.axes) == 3 * n_points + 1
        plt.close(fig)

    def test_the_ari_row_spans_the_figure(self, gaussians_frame):
        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        widths = [ax.get_position().width for ax in fig.axes]
        assert max(widths) > 3 * min(widths)
        plt.close(fig)

    def test_the_example_and_ari_rows_have_content(self, gaussians_frame):
        """An empty panel in a dashboard is indistinguishable from a panel
        whose data happened to be flat. Rows 3 and 4 always draw a k* rule
        or an image, even for a cell with no rows, so they are checked
        against their own column's data in TestPanelsReadTheirOwnColumn."""
        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        n_points = len(SCENARIOS["gaussians"].axis)
        for ax in fig.axes[: n_points + 1]:
            assert ax.get_lines() or ax.collections or ax.get_images()
        plt.close(fig)

    def test_one_shared_legend(self, gaussians_frame):
        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        assert len(fig.legends) == 1
        plt.close(fig)

    def test_titles_come_from_the_shared_map(self, gaussians_frame):
        from benchmarks.figures._benchmarking_examples import SCENARIO_TITLES

        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        assert fig._suptitle.get_text() == SCENARIO_TITLES["gaussians"]
        plt.close(fig)


class TestPanelsReadTheirOwnColumn:
    """Each criterion panel and heatmap draws the cell of the axis point
    its column names, and nothing else.

    The synthetic frame draws its values at random per axis point, so the
    columns differ; each test checks that first, since a panel drawn from
    the wrong column would otherwise still match.
    """

    def test_each_heatmap_draws_its_own_columns_selections(self, gaussians_frame):
        scenario = SCENARIOS["gaussians"]
        n_points = len(scenario.axis)
        expected = [
            k_hat_selection_matrix(
                gaussians_frame,
                metrics=DRAWN_METRICS,
                axis_label=label,
                candidate_k=scenario.candidate_k,
            )
            for _, _, label in scenario.axis
        ]
        for a, b in itertools.combinations(expected, 2):
            assert not np.array_equal(a, b, equal_nan=True)

        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        heatmaps = fig.axes[2 * n_points + 1 :]
        assert len(heatmaps) == n_points
        for ax, want in zip(heatmaps, expected):
            [image] = ax.get_images()
            drawn = np.ma.filled(np.ma.asarray(image.get_array(), dtype=float), np.nan)
            assert np.isfinite(want).all()
            np.testing.assert_array_equal(drawn, want)
        plt.close(fig)

    def test_each_criterion_panel_draws_one_curve_per_metric_from_its_own_column(
        self, gaussians_frame
    ):
        scenario = SCENARIOS["gaussians"]
        n_points = len(scenario.axis)
        candidate_k = list(scenario.candidate_k)

        def expected_curves(label):
            curves = normalized_criterion(
                gaussians_frame, metrics=DRAWN_METRICS, axis_label=label
            )
            return [
                curves.loc[curves["metric_name"] == metric]
                .set_index("k")["value"]
                .reindex(candidate_k)
                .to_numpy()
                for metric in DRAWN_METRICS
            ]

        expected = [expected_curves(label) for _, _, label in scenario.axis]
        for a, b in itertools.combinations(expected, 2):
            assert not np.allclose(np.array(a), np.array(b))

        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        panels = fig.axes[n_points + 1 : 2 * n_points + 1]
        for ax, want in zip(panels, expected):
            curves = [
                line
                for line in ax.get_lines()
                if not line.get_label().startswith("_")
                and line.get_label() != CRITERION_REFERENCE_LABEL
            ]
            assert [line.get_label() for line in curves] == [
                METRIC_DISPLAY_NAMES[m] for m in DRAWN_METRICS
            ]
            for line, values in zip(curves, want):
                np.testing.assert_allclose(line.get_ydata(), values)
        plt.close(fig)


class TestCriterionReferenceLegend:
    """The grey reference band in criterion_curves has no name on its own.

    metric_legend's _legend_groups only classifies handles by metric name
    (see _legend_groups in .._panels), so a free-form label like
    CRITERION_REFERENCE_LABEL can never surface through the shared foot
    legend. Naming it needs its own legend, on the leftmost criterion panel
    only, so the label appears exactly once rather than once per axis
    point.
    """

    def test_leftmost_criterion_panel_names_the_reference_line(
        self, gaussians_frame
    ):
        from benchmarks._panels import CRITERION_REFERENCE_LABEL

        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        n_points = len(SCENARIOS["gaussians"].axis)
        criterion_axes = fig.axes[n_points + 1 : 2 * n_points + 1]

        leftmost, *rest = criterion_axes
        legend = leftmost.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == [
            CRITERION_REFERENCE_LABEL
        ]
        for ax in rest:
            assert ax.get_legend() is None
        plt.close(fig)

    def test_the_reference_legend_sits_outside_the_axes_data_rectangle(
        self, gaussians_frame
    ):
        """A legend placed inside the panel can be crossed by a curve.

        candidate_k is centered on k_star for every registered scenario, so
        an in-panel "lower center" placement sits exactly on the one k
        every criterion's curve is normalized toward: a criterion that
        fails to prefer k* troughs there, and min-max normalization puts
        every such curve at 0 simultaneously, right where the legend text
        would be. Anchoring the legend below the axes' own data rectangle
        rules that out regardless of what the curves do.
        """
        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        fig.canvas.draw()
        n_points = len(SCENARIOS["gaussians"].axis)
        leftmost = fig.axes[n_points + 1]
        legend = leftmost.get_legend()
        assert legend is not None

        axes_bbox = leftmost.get_window_extent()
        legend_bbox = legend.get_window_extent()
        # Display y increases upward, so "outside, below the axes" means
        # the legend's own top sits at or below the axes' bottom edge.
        assert legend_bbox.y1 <= axes_bbox.y0
        plt.close(fig)


class TestScalingScenarios:
    def test_a_scaling_scenario_labels_its_own_axis(self):
        """easy/medium/hard would be a mislabeling on an n or p sweep."""
        from tests.benchmarks._helpers import synthetic_run_frame

        frame = synthetic_run_frame("gaussians_samples", n_seeds=2)
        fig = figure_scenario_dashboard("gaussians_samples", frame)
        ari_ax = fig.axes[len(SCENARIOS["gaussians_samples"].axis)]
        assert ari_ax.get_xlabel() == dict(SCALING_PANELS)["gaussians_samples"]
        labels = [ax.get_xlabel() for ax in fig.axes]
        assert not any("SNR" in label for label in labels)
        plt.close(fig)


class TestGuards:
    def test_rejects_an_unknown_scenario(self, gaussians_frame):
        with pytest.raises(KeyError, match="not_a_scenario"):
            figure_scenario_dashboard("not_a_scenario", gaussians_frame)

    def test_rejects_an_empty_frame(self):
        with pytest.raises(ValueError, match="No rows"):
            figure_scenario_dashboard("gaussians", pd.DataFrame())

    def test_rejects_a_frame_from_a_different_axis(self, gaussians_frame):
        """A run directory is content-addressed on its config, so an older
        run whose axis has since been redefined is still readable. The
        example panels come from the registry and the curves from the
        artifact, so a mismatch would draw two different experiments in one
        figure with the x ticks hiding it."""
        frame = gaussians_frame.copy()
        frame["axis_value"] = frame["axis_value"] + 100
        with pytest.raises(ValueError, match="Re-run this scenario"):
            figure_scenario_dashboard("gaussians", frame)


class TestSaving:
    def test_does_not_write_by_default(self, gaussians_frame, tmp_path):
        """A notebook view, not a manuscript figure."""
        fig = figure_scenario_dashboard("gaussians", gaussians_frame, out_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

    def test_writes_under_its_scenario_name_when_asked(
        self, gaussians_frame, tmp_path
    ):
        fig = figure_scenario_dashboard(
            "gaussians", gaussians_frame, save=True, out_dir=tmp_path
        )
        assert (tmp_path / "scenario_dashboard_gaussians.png").exists()
        plt.close(fig)
