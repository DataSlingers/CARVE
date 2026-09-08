"""Tests for the single source of figure styling."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from benchmarks._registry import CARVE_METRICS_ALL, CVI_METRICS, METRIC_DISPLAY_NAMES
from benchmarks._theme import (
    CLUSTER_CMAP_NAME,
    CLUSTER_PALETTE,
    FONT_SIZES,
    METRIC_COLORS,
    RC_PARAMS,
    apply_theme,
    cluster_cmap,
    cluster_colors,
    metric_color,
    metric_linestyle,
    metric_linewidth,
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


class TestLineStyle:
    def test_carve_measures_are_solid(self):
        for metric in (
            "ari_stability_1se",
            "ari_generalizability_1se",
            "ari_average_1se",
        ):
            assert metric_linestyle(metric) == "-"

    def test_the_oracle_and_every_index_are_dashed(self):
        for metric in ("baseline_oracle", *CVI_METRICS):
            assert metric_linestyle(metric) == "--"

    def test_an_unknown_metric_is_dashed_rather_than_raising(self):
        assert metric_linestyle("not_a_metric") == "--"

    def test_reference_series_are_drawn_thinner_than_carve(self):
        for metric in ("baseline_oracle", *CVI_METRICS):
            assert metric_linewidth(metric) < metric_linewidth("ari_stability_1se")

    def test_the_two_weights_are_the_chosen_values(self):
        """Pinned so a later edit cannot drift the emphasis silently."""
        assert metric_linewidth("ari_stability_1se") == 1.8
        assert metric_linewidth("silhouette") == 1.2

    def test_an_unknown_metric_gets_the_reference_weight(self):
        assert metric_linewidth("not_a_metric") == metric_linewidth("silhouette")

    def test_no_carve_measure_is_dashed(self):
        """The prefix rule has to cover the whole CARVE vocabulary.

        A consensus_* measure is a CARVE quantity but not an ARI curve, so
        it is deliberately not solid; only the ari_* selection curves are.
        """
        for metric in CARVE_METRICS_ALL:
            expected = "-" if metric.startswith("ari_") else "--"
            assert metric_linestyle(metric) == expected


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

    def test_the_four_indices_carry_the_published_figure_hues(self):
        """Fig 4's own values: Silhouette pink, DB purple, CH red, Gap orange.

        Pinned because the pre-rebuild code assigned these four positionally
        from the caller's metric order, which is why the published Fig 4 and
        the published Klein panel disagree about which index is which color.
        """
        assert METRIC_COLORS["silhouette"] == "#E0457B"
        assert METRIC_COLORS["davies_bouldin"] == "#A8389E"
        assert METRIC_COLORS["calinski_harabasz"] == "#D6292E"
        assert METRIC_COLORS["gap"] == "#F28522"

    def test_metric_color_falls_back_without_raising(self):
        assert metric_color("not_a_metric").startswith("#")

    def test_every_carve_metric_resolves_to_a_color(self):
        for metric in CARVE_METRICS_ALL:
            assert metric_color(metric).startswith("#")

    def test_cluster_colors_extends_beyond_the_palette_length(self):
        # Cycles CLUSTER_PALETTE from the start past its own length -- see
        # cluster_colors' docstring -- just checking it still returns n
        # colors past the palette's own length here; the cycling pattern
        # itself is pinned by the adjacent-duplicate tests below.
        assert len(cluster_colors(3)) == 3
        assert len(cluster_colors(len(CLUSTER_PALETTE) + 5)) == len(CLUSTER_PALETTE) + 5

    def test_cluster_colors_are_stable(self):
        assert cluster_colors(4) == cluster_colors(4)

    @pytest.mark.parametrize("n", [1, 3, 10])
    def test_cluster_colors_is_the_palette_in_order(self, n):
        """Cluster i is palette entry i, not a resampled spread over the map.

        This is what puts four clusters on tab10's first four hues, which is
        what the published composites draw and what a C1..C4 legend reads as.
        An earlier version returned plt.get_cmap(CLUSTER_CMAP_NAME, n)'s
        entries, which spread n samples across the whole ten-color palette --
        four clusters landed on entries 0, 3, 6 and 9. Pinned so it cannot
        drift back.
        """
        assert cluster_colors(n) == [to_hex(c) for c in CLUSTER_PALETTE[:n]]

    @pytest.mark.parametrize("n", [1, 3, 4, 10])
    def test_cluster_cmap_agrees_under_both_indexing_styles(self, n):
        """The cross-panel agreement cluster_cmap exists to buy.

        CARVE's own panels resolve a cluster's color two different ways:
        plot_cluster_violin and plot_cluster_scatter index the colormap by
        plain integer, while plot_consensus_matrix's cluster band goes
        through imshow, which normalizes the index to [0, 1] first. Handing
        them a map of exactly n colors makes both land on the same entry --
        and on the same entry cluster_colors gives every other figure.

        Against the unsliced ten-color map the two styles disagree for any
        n < 10, which is the defect that put one cluster in two colors
        inside a single figure.
        """
        cmap = cluster_cmap(n)
        by_index = [to_hex(cmap(i)) for i in range(n)]
        by_imshow = [to_hex(cmap(i / (n - 1) if n > 1 else 0.0)) for i in range(n)]
        assert by_index == cluster_colors(n)
        assert by_imshow == cluster_colors(n)

    @pytest.mark.parametrize("n", [11, 14, 17])
    def test_cluster_colors_no_adjacent_duplicates_above_the_palette_length(self, n):
        """Above ten clusters, agreement with the registered colormap is
        not achievable (it duplicates by construction), so cluster_colors
        instead guarantees legibility within one panel: cycling
        CLUSTER_PALETTE from the start means every repeat falls exactly
        len(CLUSTER_PALETTE) indices apart, so two *adjacent* cluster
        indices are never the same color. An earlier version of this
        function resampled the registered colormap unconditionally, which
        put duplicates on adjacent indices instead (measured: 7 of 16
        adjacent pairs identical at n=17) -- exactly the failure mode this
        pins against.

        n=14 and n=17 are not arbitrary: Levine's reported labels run to
        roughly 14 populations and its swept k reaches 17, and
        cluster_color_map colors the alluvial and ARI-lollipop panels'
        true-label column with this function, so both counts are real,
        not hypothetical.
        """
        colors = cluster_colors(n)
        assert all(a != b for a, b in zip(colors, colors[1:]))


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
