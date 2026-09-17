"""Tests for the two SI ablation figures, on synthetic frames."""

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.figure import Figure

from benchmarks._registry import ABLATIONS
from benchmarks.figures import figure_ablation_b, figure_ablation_rho
from tests.benchmarks._helpers import synthetic_ablation_frames

RHO_B = ABLATIONS["rho_b"]


@pytest.fixture(scope="module")
def frames():
    return synthetic_ablation_frames(RHO_B, "dev", seed=1)


class TestFigureAblationRho:
    def test_returns_a_nine_panel_figure(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="dev", save=False)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 9

    def test_saves_under_its_si_name(self, frames, tmp_path):
        figure_ablation_rho(frames, ablation=RHO_B, scale="dev", out_dir=tmp_path)
        assert (tmp_path / "si_fig_ablation_rho.png").exists()

    def test_save_false_writes_nothing(self, frames, tmp_path):
        figure_ablation_rho(frames, ablation=RHO_B, scale="dev", save=False, out_dir=tmp_path)
        assert not list(tmp_path.iterdir())

    def test_marks_the_default_rho_on_every_sweep_panel(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="dev", save=False)
        # The stacked-bar panel (Klein shares) has no reference line.
        with_line = [
            ax for ax in fig.axes
            if any(abs(line.get_xdata()[0] - RHO_B.rho_default) < 1e-9 for line in ax.lines
                   if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1])
        ]
        assert len(with_line) == 8

    def test_similarity_panel_reaches_the_reference_ratio(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="dev", save=False)
        # Restricted to lines with more than two points: axvline and axhline
        # reference lines carry exactly two points (their xdata is [0, 1] or
        # [default, default] in blended axes coordinates), so an unrestricted
        # max over every Line2D passes at 1.0 without the similarity panel
        # ever being drawn. The swept curves are the only remaining source of
        # an x value at REFERENCE_RATIO.
        xmax = max(
            max(line.get_xdata())
            for ax in fig.axes
            for line in ax.lines
            if len(line.get_xdata()) > 2
        )
        assert xmax == pytest.approx(1.0)


class TestFigureAblationB:
    def test_returns_an_eight_panel_figure(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="dev", save=False)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 8

    def test_saves_under_its_si_name(self, frames, tmp_path):
        figure_ablation_b(frames, ablation=RHO_B, scale="dev", out_dir=tmp_path)
        assert (tmp_path / "si_fig_ablation_b.png").exists()

    def test_b_axes_are_logarithmic(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="dev", save=False)
        # Every panel but the stacked-bar one sweeps B on a log axis.
        assert sum(ax.get_xscale() == "log" for ax in fig.axes) == 7
