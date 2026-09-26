"""Tests for the two SI ablation figures, on synthetic frames."""

import dataclasses

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text

from benchmarks._ablation_cells import REFERENCE_RATIO, arm_view
from benchmarks._ablation_summary import HEADLINE_METRICS, POOLED, selection_summary
from benchmarks._registry import ABLATIONS, METRIC_DISPLAY_NAMES
from benchmarks._studies import STUDIES
from benchmarks._theme import (
    FALLBACK_COLOR,
    PRINT_MAX_HEIGHT_IN,
    PRINT_WIDTH_IN,
    metric_color,
)
from benchmarks.figures import figure_ablation_b, figure_ablation_rho
from benchmarks.figures._ablation import KEY_HEADROOM_TOP, PROPORTION_LIMITS, Z_95
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


STAB, GEN = HEADLINE_METRICS
SELECTOR_COLORS = {to_hex(metric_color(STAB)), to_hex(metric_color(GEN))}


def _panels(fig):
    """Panel letter to axes, read from each axes' label."""
    return {ax.get_label(): ax for ax in fig.axes}


def _has_default_line(ax, default):
    return any(
        len(line.get_xdata()) == 2
        and line.get_xdata()[0] == line.get_xdata()[1]
        and abs(line.get_xdata()[0] - default) < 1e-9
        for line in ax.lines
    )


def _data_line_colors(ax):
    """Colors of the swept lines, leaving out two-point reference lines and
    the grey 1/sqrt(B) guide."""
    colors = {to_hex(line.get_color()) for line in ax.lines if len(line.get_xdata()) > 2}
    return colors - {to_hex(FALLBACK_COLOR)}


def _band(ax, color):
    (band,) = [
        c for c in ax.collections
        if isinstance(c, PolyCollection) and to_hex(c.get_facecolor()[0]) == to_hex(color)
    ]
    return band.get_paths()[0].vertices


class TestPrintSize:
    @pytest.mark.parametrize("figure", [figure_ablation_rho, figure_ablation_b])
    @pytest.mark.parametrize("which", ["with_study", "without_study"])
    def test_drawn_at_the_plos_page_width(self, figure, which, request):
        ablation, frames = request.getfixturevalue(which)
        fig = figure(frames, ablation=ablation, scale="test", save=False)
        width, height = fig.get_size_inches()
        assert width == pytest.approx(PRINT_WIDTH_IN)
        assert height <= PRINT_MAX_HEIGHT_IN
        plt.close(fig)

    @pytest.mark.parametrize("figure", [figure_ablation_rho, figure_ablation_b])
    def test_every_text_is_within_plos_sizes(self, figure, with_study):
        ablation, frames = with_study
        fig = figure(frames, ablation=ablation, scale="test", save=False)
        fig.canvas.draw()
        sizes = [
            text.get_fontsize() for text in fig.findobj(Text)
            if text.get_visible() and text.get_text().strip()
        ]
        assert sizes
        assert all(8.0 <= size <= 12.0 for size in sizes)
        plt.close(fig)

    @pytest.mark.parametrize(
        ("figure", "name"),
        [(figure_ablation_rho, "si_fig_ablation_rho.png"), (figure_ablation_b, "si_fig_ablation_b.png")],
    )
    def test_the_saved_file_fits_the_page_at_300_dpi(self, figure, name, with_study, tmp_path):
        ablation, frames = with_study
        fig = figure(frames, ablation=ablation, scale="test", out_dir=tmp_path)
        height, width = mpimg.imread(tmp_path / name).shape[:2]
        assert width <= 2250
        assert height <= 2625
        plt.close(fig)


class TestPanelLabels:
    @pytest.mark.parametrize("figure", [figure_ablation_rho, figure_ablation_b])
    def test_no_letter_overlaps_its_title(self, figure, with_study):
        ablation, frames = with_study
        fig = figure(frames, ablation=ablation, scale="test", save=False)
        renderer = fig.canvas.get_renderer()
        fig.canvas.draw()
        for letter, ax in _panels(fig).items():
            (letter_text,) = [t for t in ax.texts if t.get_text() == letter]
            title = ax.get_title(loc="left")
            (title_text,) = [
                t for t in ax.get_children()
                if isinstance(t, Text) and t.get_text() == title
            ]
            letter_box = letter_text.get_window_extent(renderer)
            title_box = title_text.get_window_extent(renderer)
            assert letter_box.x1 < title_box.x0, letter
            assert letter_text.get_fontweight() == "bold"
        plt.close(fig)


class TestFigureAblationRho:
    def test_returns_a_nine_panel_figure(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        assert isinstance(fig, Figure)
        assert list(_panels(fig)) == list("ABCDEFGHI")
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
        # The stacked-bar panel (Klein shares, I) has no reference line.
        panels = _panels(fig)
        assert [p for p, ax in panels.items() if _has_default_line(ax, RHO_B.rho_default)] == list("ABCDEFGH")
        plt.close(fig)

    def test_one_legend_names_the_selectors_the_interval_and_the_default(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        (legend,) = fig.legends
        assert [t.get_text() for t in legend.get_texts()] == [
            METRIC_DISPLAY_NAMES[STAB],
            METRIC_DISPLAY_NAMES[GEN],
            "95% CI",
            f"Default $\\rho$ = {RHO_B.rho_default:g}",
        ]
        plt.close(fig)

    def test_selector_panels_draw_both_selectors_in_their_own_colors(self, frames):
        # One encoding everywhere: color is the selector. A panel that drew
        # a selector in black or a study color would fail here.
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        panels = _panels(fig)
        for letter in "ABCDEG":
            assert _data_line_colors(panels[letter]) == SELECTOR_COLORS, letter
        plt.close(fig)

    def test_bias_band_is_the_95_percent_interval(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        sel = selection_summary(
            arm_view(frames, ablation=RHO_B, scale="test", arm="rho")["selection"],
            x="subsample_ratio",
        )
        pooled = sel[(sel["study"] == POOLED) & (sel["metric_name"] == STAB)]
        ys = _band(_panels(fig)["B"], metric_color(STAB))[:, 1]
        assert ys.max() == pytest.approx((pooled["bias_mean"] + Z_95 * pooled["bias_sem"]).max())
        assert ys.min() == pytest.approx((pooled["bias_mean"] - Z_95 * pooled["bias_sem"]).min())
        plt.close(fig)

    def test_proportions_share_one_fixed_axis(self, frames):
        assert PROPORTION_LIMITS[0] < 0.0 and PROPORTION_LIMITS[1] > 1.0
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        panels = _panels(fig)
        assert panels["A"].get_ylim() == pytest.approx(PROPORTION_LIMITS)
        # D carries its line-style key above 1; its ticks still stop at 1.
        assert panels["D"].get_ylim() == pytest.approx((PROPORTION_LIMITS[0], KEY_HEADROOM_TOP))
        assert list(panels["D"].get_yticks()) == list(panels["A"].get_yticks())
        plt.close(fig)

    def test_rare_recall_draws_k_hat_solid_and_k_star_dashed(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        lines = [line for line in _panels(fig)["D"].lines if len(line.get_xdata()) > 2]
        assert sorted(line.get_linestyle() for line in lines) == ["-", "-", "--", "--"]
        plt.close(fig)

    def test_similarity_panel_marks_the_refit_reference(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        ax = _panels(fig)["F"]
        # The swept line stops below the reference ratio; the refit is its
        # own marker at it.
        swept = [line for line in ax.lines if len(line.get_xdata()) > 2]
        assert max(max(line.get_xdata()) for line in swept) < REFERENCE_RATIO
        refit = [line for line in ax.lines if list(line.get_xdata()) == [REFERENCE_RATIO]]
        assert len(refit) == 1
        assert np.isfinite(refit[0].get_ydata()[0])
        plt.close(fig)

    def test_study_similarity_lines_read_k_from_the_registry(
        self, frames, monkeypatch
    ):
        # If the figure restated Klein's reported k as a literal instead of
        # reading Study.reported_k, this would still pass at the registry's
        # actual value (4); patching it to something else is what would
        # catch a reintroduced literal.
        patched = dataclasses.replace(STUDIES["klein"], reported_k=7)
        monkeypatch.setitem(STUDIES, "klein", patched)
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        labels = [line.get_label() for line in _panels(fig)["H"].lines]
        assert any(label.endswith("k=7") for label in labels)
        assert not any(label.endswith("k=4") for label in labels)
        plt.close(fig)

    def test_study_similarity_panel_is_empty_when_the_study_reports_no_k(
        self, frames, monkeypatch
    ):
        patched = dataclasses.replace(STUDIES["klein"], reported_k=None)
        monkeypatch.setitem(STUDIES, "klein", patched)
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="test", save=False)
        assert not [line for line in _panels(fig)["H"].lines if len(line.get_xdata()) > 2]
        plt.close(fig)


class TestFigureAblationB:
    def test_returns_a_nine_panel_figure(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        assert isinstance(fig, Figure)
        assert list(_panels(fig)) == list("ABCDEFGHI")
        plt.close(fig)

    def test_saves_under_its_si_name(self, frames, tmp_path):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", out_dir=tmp_path)
        assert (tmp_path / "si_fig_ablation_b.png").exists()
        plt.close(fig)

    def test_b_axes_are_logarithmic(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        # The stacked bars sit at the B values too, so every panel shares
        # the log axis.
        assert all(ax.get_xscale() == "log" for ax in fig.axes)
        plt.close(fig)

    def test_marks_the_default_b_on_every_sweep_panel(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        panels = _panels(fig)
        assert [p for p, ax in panels.items() if _has_default_line(ax, RHO_B.b_default)] == list("ABCDEFGH")
        plt.close(fig)

    def test_share_bars_sit_at_the_b_values(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        ax = _panels(fig)["I"]
        # Each bar spans the same factor either side of its B, so its
        # geometric center is the B value on the log axis.
        centers = {
            round(float(np.sqrt(bar.get_x() * (bar.get_x() + bar.get_width()))), 6)
            for bar in ax.patches
        }
        assert centers == {float(b) for b in RHO_B.b_grid}
        plt.close(fig)

    def test_b_ticks_are_the_grid_values_in_plain_numbers(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        ax = _panels(fig)["A"]
        assert list(ax.get_xticks()) == list(RHO_B.b_grid)
        assert [t.get_text() for t in ax.get_xticklabels()] == [f"{b:g}" for b in RHO_B.b_grid]
        assert not ax.xaxis.get_minorticklocs().size
        plt.close(fig)

    def test_one_legend_names_the_default_b(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        (legend,) = fig.legends
        assert legend.get_texts()[-1].get_text() == f"Default $B$ = {RHO_B.b_default:g}"
        plt.close(fig)

    def test_selector_panels_draw_both_selectors_in_their_own_colors(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        panels = _panels(fig)
        for letter in "ABCDEFGH":
            assert _data_line_colors(panels[letter]) == SELECTOR_COLORS, letter
        plt.close(fig)

    def test_recovery_and_agreement_share_the_proportion_axis(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        panels = _panels(fig)
        for letter in "ADG":
            assert panels[letter].get_ylim() == pytest.approx(PROPORTION_LIMITS)
        plt.close(fig)


# The registry's ablation runs no case study (gaussians at medium only).
NO_STUDY = small_ablation(ABLATIONS["rho_b"])


@pytest.fixture(scope="module")
def frames_without_study(tmp_path_factory):
    return synthetic_ablation_frames(
        NO_STUDY, "test", seed=3, tmp_path=tmp_path_factory.mktemp("no_study_frames")
    )


@pytest.fixture(scope="module")
def with_study(frames):
    return RHO_B, frames


@pytest.fixture(scope="module")
def without_study(frames_without_study):
    return NO_STUDY, frames_without_study


class TestWithoutCaseStudy:
    def test_the_registry_shape_has_no_study(self):
        assert NO_STUDY.study is None

    def test_rho_figure_drops_the_study_row(self, frames_without_study):
        fig = figure_ablation_rho(
            frames_without_study, ablation=NO_STUDY, scale="test", save=False
        )
        assert list(_panels(fig)) == list("ABCDEF")
        plt.close(fig)

    def test_rho_figure_marks_the_default_rho_on_every_panel(self, frames_without_study):
        fig = figure_ablation_rho(
            frames_without_study, ablation=NO_STUDY, scale="test", save=False
        )
        assert all(_has_default_line(ax, NO_STUDY.rho_default) for ax in fig.axes)
        plt.close(fig)

    def test_rare_recall_panel_names_the_setting_it_reads(self, frames_without_study):
        fig = figure_ablation_rho(
            frames_without_study, ablation=NO_STUDY, scale="test", save=False
        )
        assert "medium" in _panels(fig)["D"].get_title(loc="left")
        plt.close(fig)

    def test_b_figure_drops_its_study_row(self, frames_without_study, frames):
        fig = figure_ablation_b(
            frames_without_study, ablation=NO_STUDY, scale="test", save=False
        )
        assert list(_panels(fig)) == list("ABCDEF")
        assert sum(ax.get_xscale() == "log" for ax in fig.axes) == 6
        full = figure_ablation_b(frames, ablation=RHO_B, scale="test", save=False)
        assert fig.get_size_inches()[1] < full.get_size_inches()[1]
        plt.close(fig)
        plt.close(full)

    def test_no_study_name_reaches_a_legend(self, frames_without_study):
        # A single scenario is the pooled simulations; the figure names the
        # selectors, not the scenario.
        for figure in (figure_ablation_rho, figure_ablation_b):
            fig = figure(frames_without_study, ablation=NO_STUDY, scale="test", save=False)
            labels = [t.get_text() for t in fig.legends[0].get_texts()]
            assert "Gaussian Mixtures" not in labels
            assert "Pooled (simulations)" not in labels
            plt.close(fig)
