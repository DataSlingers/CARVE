"""Tests for the single source of figure styling."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from benchmarks._registry import CARVE_METRICS_ALL, CVI_METRICS, METRIC_DISPLAY_NAMES
from benchmarks._theme import (
    CLUSTER_PALETTE,
    FONT_SIZES,
    METRIC_COLORS,
    RC_PARAMS,
    apply_theme,
    cluster_colors,
    metric_color,
    save_figure,
    style_axes,
    theme_context,
)

PLOTTED_METRICS = (
    "ari_stability_1se",
    "ari_generalizability_1se",
    "ari_average_1se",
    *CVI_METRICS,
)


class TestPalette:
    def test_every_plotted_metric_has_a_color(self):
        for metric in PLOTTED_METRICS:
            assert metric in METRIC_COLORS

    def test_carve_stability_is_the_okabe_ito_green_everywhere(self):
        assert METRIC_COLORS["ari_stability_1se"] == "#009E73"

    def test_every_display_name_has_a_color(self):
        for metric in METRIC_DISPLAY_NAMES:
            assert metric in METRIC_COLORS

    def test_no_two_plotted_metrics_share_a_color(self):
        colors = [METRIC_COLORS[m] for m in PLOTTED_METRICS]
        assert len(set(colors)) == len(colors)

    def test_metric_color_falls_back_without_raising(self):
        assert metric_color("not_a_metric").startswith("#")

    def test_every_carve_metric_resolves_to_a_color(self):
        for metric in CARVE_METRICS_ALL:
            assert metric_color(metric).startswith("#")

    def test_cluster_colors_cycle_beyond_the_palette(self):
        assert len(cluster_colors(3)) == 3
        assert len(cluster_colors(len(CLUSTER_PALETTE) + 5)) == len(CLUSTER_PALETTE) + 5

    def test_cluster_colors_are_stable(self):
        assert cluster_colors(4) == cluster_colors(4)


class TestRcParams:
    def test_pins_pdf_fonttype_42_for_editable_text(self):
        assert RC_PARAMS["pdf.fonttype"] == 42
        assert RC_PARAMS["ps.fonttype"] == 42

    def test_saves_at_publication_resolution(self):
        assert RC_PARAMS["savefig.dpi"] == 300

    def test_hides_the_top_and_right_spines(self):
        assert RC_PARAMS["axes.spines.top"] is False
        assert RC_PARAMS["axes.spines.right"] is False

    def test_apply_theme_actually_mutates_rcparams(self):
        plt.rcParams["pdf.fonttype"] = 3
        apply_theme()
        assert plt.rcParams["pdf.fonttype"] == 42

    def test_theme_context_restores_previous_state(self):
        plt.rcParams["pdf.fonttype"] = 3
        with theme_context():
            assert plt.rcParams["pdf.fonttype"] == 42
        assert plt.rcParams["pdf.fonttype"] == 3


class TestFontSizes:
    def test_named_sizes_replace_inline_literals(self):
        for key in ("panel_letter", "axis_label", "tick", "legend", "title"):
            assert key in FONT_SIZES

    def test_sizes_are_ordered_sensibly(self):
        assert FONT_SIZES["tick"] <= FONT_SIZES["axis_label"] <= FONT_SIZES["title"]
        assert FONT_SIZES["panel_letter"] > FONT_SIZES["title"]


class TestStyleAxes:
    def test_returns_the_same_axes_object(self):
        fig, ax = plt.subplots()
        assert style_axes(ax) is ax
        plt.close(fig)

    def test_removes_the_top_and_right_spines(self):
        fig, ax = plt.subplots()
        style_axes(ax)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close(fig)

    def test_grid_can_be_disabled(self):
        fig, ax = plt.subplots()
        style_axes(ax, grid=False)
        assert not ax.xaxis.get_gridlines()[0].get_visible()
        plt.close(fig)


class TestSaveFigure:
    def test_writes_the_file_and_returns_its_path(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = save_figure(fig, tmp_path / "sub" / "demo.png")
        assert path.exists()
        assert path.name == "demo.png"
        plt.close(fig)

    def test_creates_missing_parent_directories(self, tmp_path):
        fig, _ = plt.subplots()
        save_figure(fig, tmp_path / "a" / "b" / "c.png")
        assert (tmp_path / "a" / "b").is_dir()
        plt.close(fig)
