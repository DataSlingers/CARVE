"""Tests for the two SI ablation figures, on synthetic frames."""

import dataclasses

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from benchmarks._registry import ABLATIONS
from benchmarks._studies import STUDIES
from benchmarks.figures import figure_ablation_b, figure_ablation_rho
from tests.benchmarks._helpers import small_ablation, synthetic_ablation_frames

RHO_B = small_ablation(ABLATIONS["rho_b"], study="klein")


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    # Written as checkpoint files and read back, so the figures meet the
    # dtypes a real run directory produces (see synthetic_ablation_frames).
    return synthetic_ablation_frames(
        RHO_B, "test", seed=1, tmp_path=tmp_path_factory.mktemp("ablation_frames")
    )


class TestSyntheticFrames:
    def test_carry_what_a_run_directory_produces(self, frames):
        # The premise of every figure test below: the frames went through
        # parquet and read_frames, so one undefined selection has made
        # selected_k float64 for the whole run (the production shape that
        # crashed both figures) while at_k stays typed past the study
        # cells' empty files. A helper that handed the figures in-memory
        # frames again would fail here.
        selection = frames["selection"]
        assert selection["selected_k"].dtype.kind == "f"
        assert selection["selected_k"].isna().sum() == 1
        assert selection.loc[selection["selected_k"].isna(), "study"].tolist() == [RHO_B.study]
        assert frames["at_k"]["k"].dtype.kind == "i"
        assert frames["at_k"]["ari_at_k"].dtype.kind == "f"


class TestFigureAblationRho:
    def test_returns_a_nine_panel_figure(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 9
        plt.close(fig)

    def test_saves_under_its_si_name(self, frames, tmp_path):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", out_dir=tmp_path)
        assert (tmp_path / "si_fig_ablation_rho.png").exists()
        plt.close(fig)

    def test_save_false_writes_nothing(self, frames, tmp_path):
        fig = figure_ablation_rho(
            frames, ablation=RHO_B, scale="test", save=False, out_dir=tmp_path
        )
        assert not list(tmp_path.iterdir())
        plt.close(fig)

    def test_marks_the_default_rho_on_every_sweep_panel(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        # The stacked-bar panel (Klein shares) has no reference line.
        with_line = [
            ax for ax in fig.axes
            if any(abs(line.get_xdata()[0] - RHO_B.rho_default) < 1e-9 for line in ax.lines
                   if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1])
        ]
        assert len(with_line) == 8
        plt.close(fig)

    def test_similarity_panel_reaches_the_reference_ratio(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
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
        plt.close(fig)

    def test_panel_i_similarity_lines_read_k_from_the_registry(
        self, frames, monkeypatch
    ):
        # If the figure restated Klein's reported k as a literal instead of
        # reading Study.reported_k, this would still pass at the registry's
        # actual value (4); patching it to something else is what would
        # catch a reintroduced literal.
        patched = dataclasses.replace(STUDIES["klein"], reported_k=7)
        monkeypatch.setitem(STUDIES, "klein", patched)
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        labels = [line.get_label() for ax in fig.axes for line in ax.lines]
        assert any(label.endswith("subsample vs full at k=7") for label in labels)
        assert not any(label.endswith("subsample vs full at k=4") for label in labels)
        plt.close(fig)

    def test_panel_i_skips_similarity_lines_when_the_study_reports_no_k(
        self, frames, monkeypatch
    ):
        patched = dataclasses.replace(STUDIES["klein"], reported_k=None)
        monkeypatch.setitem(STUDIES, "klein", patched)
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        labels = [line.get_label() for ax in fig.axes for line in ax.lines]
        assert not any("subsample vs full at k=" in label for label in labels)
        plt.close(fig)


class TestFigureAblationB:
    def test_returns_an_eight_panel_figure(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 8
        plt.close(fig)

    def test_saves_under_its_si_name(self, frames, tmp_path):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", out_dir=tmp_path)
        assert (tmp_path / "si_fig_ablation_b.png").exists()
        plt.close(fig)

    def test_b_axes_are_logarithmic(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        # Every panel but the stacked-bar one sweeps B on a log axis.
        assert sum(ax.get_xscale() == "log" for ax in fig.axes) == 7
        plt.close(fig)


# The registry's ablation runs no case study (gaussians at medium only).
NO_STUDY = small_ablation(ABLATIONS["rho_b"])


@pytest.fixture(scope="module")
def frames_without_study(tmp_path_factory):
    return synthetic_ablation_frames(
        NO_STUDY, "test", seed=3, tmp_path=tmp_path_factory.mktemp("no_study_frames")
    )


def _left_titles(fig):
    return [ax.get_title(loc="left") for ax in fig.axes]


class TestWithoutCaseStudy:
    def test_the_registry_shape_has_no_study(self):
        assert NO_STUDY.study is None

    def test_rho_figure_drops_the_study_panels(self, frames_without_study):
        fig = figure_ablation_rho(
            frames_without_study, ablation=NO_STUDY, scale="test", save=False
        )
        assert len(fig.axes) == 7
        assert [title.split()[0] for title in _left_titles(fig)] == list("ABCDEFG")
        plt.close(fig)

    def test_rho_figure_marks_the_default_rho_on_every_panel(self, frames_without_study):
        fig = figure_ablation_rho(
            frames_without_study, ablation=NO_STUDY, scale="test", save=False
        )
        with_line = [
            ax for ax in fig.axes
            if any(abs(line.get_xdata()[0] - NO_STUDY.rho_default) < 1e-9 for line in ax.lines
                   if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1])
        ]
        assert len(with_line) == 7
        plt.close(fig)

    def test_rare_recall_panel_names_the_setting_it_reads(self, frames_without_study):
        fig = figure_ablation_rho(
            frames_without_study, ablation=NO_STUDY, scale="test", save=False
        )
        assert "E  Rare-cluster recall, medium setting" in _left_titles(fig)
        plt.close(fig)

    def test_b_figure_drops_its_study_row(self, frames_without_study):
        fig = figure_ablation_b(
            frames_without_study, ablation=NO_STUDY, scale="test", save=False
        )
        assert [title.split()[0] for title in _left_titles(fig)] == list("ABCDEF")
        assert sum(ax.get_xscale() == "log" for ax in fig.axes) == 6
        assert fig.get_size_inches()[1] == pytest.approx(3 * 3.9)
        plt.close(fig)

    def test_one_scenario_is_drawn_once_under_its_own_name(self, frames_without_study):
        # With one scenario the pooled line is that scenario; drawing both
        # hid one under the other and listed it twice in the legend.
        for figure in (figure_ablation_rho, figure_ablation_b):
            fig = figure(frames_without_study, ablation=NO_STUDY, scale="test", save=False)
            labels = [
                label for ax in fig.axes for label in ax.get_legend_handles_labels()[1]
            ]
            assert "Pooled (simulations)" not in labels
            assert "Gaussian Mixtures" in labels
            for ax in fig.axes:
                own = ax.get_legend_handles_labels()[1]
                assert own.count("Gaussian Mixtures") <= 1
            plt.close(fig)
