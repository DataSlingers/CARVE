"""Tests for the per-scenario dashboard."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarks._registry import SCENARIOS
from benchmarks.figures import figure_scenario_dashboard
from benchmarks.figures._benchmarking_results import DEFAULT_METRICS


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

    def test_every_panel_has_content(self, gaussians_frame):
        """An empty panel in a dashboard is indistinguishable from a panel
        whose data happened to be flat."""
        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        for ax in fig.axes:
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


class TestScalingScenarios:
    def test_a_scaling_scenario_labels_its_own_axis(self):
        """easy/medium/hard would be a mislabeling on an n or p sweep."""
        from tests.benchmarks._helpers import synthetic_run_frame

        frame = synthetic_run_frame("gaussians_samples", n_seeds=2)
        fig = figure_scenario_dashboard("gaussians_samples", frame)
        labels = [ax.get_xlabel() for ax in fig.axes]
        assert any("n" in label for label in labels)
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
