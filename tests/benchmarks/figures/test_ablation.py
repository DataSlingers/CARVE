"""Tests for the SI ablation figure, on synthetic frames."""

import dataclasses

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text

from benchmarks._ablation_cells import arm_view
from benchmarks._ablation_summary import HEADLINE_METRICS, POOLED, ari_summary
from benchmarks._registry import ABLATIONS, METRIC_DISPLAY_NAMES
from benchmarks._theme import (
    FALLBACK_COLOR,
    PRINT_MAX_HEIGHT_IN,
    PRINT_WIDTH_IN,
    metric_color,
)
from benchmarks.figures import figure_ablation
from benchmarks.figures._ablation import ARI_LIMITS, Z_95
from tests.benchmarks._helpers import small_ablation, synthetic_ablation_frames

RHO_B = small_ablation(ABLATIONS["rho_b"], study="klein")
# The registry's ablation runs no case study (gaussians at medium only).
NO_STUDY = small_ablation(ABLATIONS["rho_b"])
STAB, GEN = HEADLINE_METRICS
SELECTOR_COLORS = {to_hex(metric_color(STAB)), to_hex(metric_color(GEN))}
RHO, B = "subsample_ratio", "n_resamples"


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    # Written as checkpoint files and read back, so the figure meets the
    # dtypes a real run directory produces (see synthetic_ablation_frames).
    return synthetic_ablation_frames(
        RHO_B, "test", seed=1, tmp_path=tmp_path_factory.mktemp("ablation_frames")
    )


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


@pytest.fixture(scope="module")
def figure(without_study):
    ablation, frames = without_study
    fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
    yield fig
    plt.close(fig)


class TestSyntheticFrames:
    def test_carry_what_a_run_directory_produces(self, frames):
        # The premise of every figure test below: the frames went through
        # parquet and read_frames, so one undefined selection has made
        # selected_k float64 for the whole run while at_k stays typed past
        # the study cells' empty files. A helper that handed the figure
        # in-memory frames again would fail here.
        selection = frames["selection"]
        assert selection["selected_k"].dtype.kind == "f"
        assert selection["selected_k"].isna().sum() == 1
        assert selection.loc[selection["selected_k"].isna(), "study"].tolist() == [RHO_B.study]
        assert frames["at_k"]["k"].dtype.kind == "i"
        assert frames["at_k"]["ari_at_k"].dtype.kind == "f"


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


def _selector_line(ax, metric):
    (line,) = [
        line for line in ax.lines
        if len(line.get_xdata()) > 2 and to_hex(line.get_color()) == to_hex(metric_color(metric))
    ]
    return line


def _band(ax, color):
    (band,) = [
        c for c in ax.collections
        if isinstance(c, PolyCollection) and to_hex(c.get_facecolor()[0]) == to_hex(color)
    ]
    return band.get_paths()[0].vertices


def _pooled(frames, ablation, arm, x, metric):
    view = arm_view(frames, ablation=ablation, scale="test", arm=arm)
    summary = ari_summary(view["selection"], view["at_k"], x=x)
    rows = summary[(summary["study"] == POOLED) & (summary["metric_name"] == metric)]
    return rows.sort_values(x)


class TestPrintSize:
    @pytest.mark.parametrize("which", ["with_study", "without_study"])
    def test_drawn_at_the_plos_page_width(self, which, request):
        ablation, frames = request.getfixturevalue(which)
        fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
        width, height = fig.get_size_inches()
        assert width == pytest.approx(PRINT_WIDTH_IN)
        assert height <= PRINT_MAX_HEIGHT_IN
        plt.close(fig)

    def test_every_text_is_within_plos_sizes(self, with_study):
        ablation, frames = with_study
        fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
        fig.canvas.draw()
        sizes = [
            text.get_fontsize() for text in fig.findobj(Text)
            if text.get_visible() and text.get_text().strip()
        ]
        assert sizes
        assert all(8.0 <= size <= 12.0 for size in sizes)
        plt.close(fig)

    def test_the_saved_file_fits_the_page_at_300_dpi(self, with_study, tmp_path):
        ablation, frames = with_study
        fig = figure_ablation(frames, ablation=ablation, scale="test", out_dir=tmp_path)
        height, width = mpimg.imread(tmp_path / "si_fig_ablation.png").shape[:2]
        assert width <= 2250
        assert height <= 2625
        plt.close(fig)


class TestPanelLabels:
    def test_no_letter_overlaps_its_title(self, with_study):
        ablation, frames = with_study
        fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
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


class TestFigureAblation:
    def test_returns_two_panels_without_a_study(self, figure):
        assert isinstance(figure, Figure)
        assert list(_panels(figure)) == list("AB")

    def test_a_study_adds_a_row(self, with_study, figure):
        ablation, frames = with_study
        fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
        assert list(_panels(fig)) == list("ABCD")
        assert fig.get_size_inches()[1] > figure.get_size_inches()[1]
        plt.close(fig)

    def test_saves_under_its_si_name(self, without_study, tmp_path):
        ablation, frames = without_study
        fig = figure_ablation(frames, ablation=ablation, scale="test", out_dir=tmp_path)
        assert [p.name for p in tmp_path.iterdir()] == ["si_fig_ablation.png"]
        plt.close(fig)

    def test_save_false_writes_nothing(self, without_study, tmp_path):
        ablation, frames = without_study
        fig = figure_ablation(
            frames, ablation=ablation, scale="test", save=False, out_dir=tmp_path
        )
        assert not list(tmp_path.iterdir())
        plt.close(fig)

    def test_the_left_column_sweeps_rho_and_the_right_column_b(self, with_study):
        ablation, frames = with_study
        fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
        panels = _panels(fig)
        for letter in "AC":
            ax = panels[letter]
            assert ax.get_xscale() == "linear"
            assert _has_default_line(ax, ablation.rho_default)
        for letter in "BD":
            ax = panels[letter]
            assert ax.get_xscale() == "log"
            assert list(ax.get_xticks()) == list(ablation.b_grid)
            assert [t.get_text() for t in ax.get_xticklabels()] == [f"{b:g}" for b in ablation.b_grid]
            assert not ax.xaxis.get_minorticklocs().size
            assert _has_default_line(ax, ablation.b_default)
        plt.close(fig)

    def test_every_panel_draws_both_selectors_in_their_own_colors(self, with_study):
        ablation, frames = with_study
        fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
        for letter, ax in _panels(fig).items():
            colors = {to_hex(line.get_color()) for line in ax.lines if len(line.get_xdata()) > 2}
            assert colors == SELECTOR_COLORS, letter
        plt.close(fig)

    @pytest.mark.parametrize(("letter", "arm", "x"), [("A", "rho", RHO), ("B", "b", B)])
    def test_lines_are_the_stability_consensus_ari_at_the_selected_k(
        self, without_study, figure, letter, arm, x
    ):
        # The synthetic frames draw ari_selected and at_k independently, so
        # a figure that plotted the stored ari_selected would miss these.
        ablation, frames = without_study
        ax = _panels(figure)[letter]
        for metric in HEADLINE_METRICS:
            pooled = _pooled(frames, ablation, arm, x, metric)
            line = _selector_line(ax, metric)
            assert list(line.get_xdata()) == pytest.approx(list(pooled[x]))
            assert list(line.get_ydata()) == pytest.approx(list(pooled["ari_mean"]))
        stored = arm_view(frames, ablation=ablation, scale="test", arm=arm)["selection"]
        stored = stored[stored["metric_name"] == STAB].groupby(x)["ari_selected"].mean()
        assert list(_selector_line(ax, STAB).get_ydata()) != pytest.approx(list(stored))

    def test_band_is_the_95_percent_interval_over_datasets(self, without_study, figure):
        ablation, frames = without_study
        pooled = _pooled(frames, ablation, "b", B, STAB)
        ys = _band(_panels(figure)["B"], metric_color(STAB))[:, 1]
        assert ys.max() == pytest.approx((pooled["ari_mean"] + Z_95 * pooled["ari_sem"]).max())
        assert ys.min() == pytest.approx((pooled["ari_mean"] - Z_95 * pooled["ari_sem"]).min())

    def test_ari_is_drawn_on_the_full_scale(self, with_study):
        # A fixed 0 to 1 axis, so a difference of a few hundredths reads as
        # what it is; autoscaling zoomed the old panels to 0.60-0.85.
        assert ARI_LIMITS == (0.0, 1.0)
        ablation, frames = with_study
        fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
        for letter, ax in _panels(fig).items():
            assert ax.get_ylim() == pytest.approx(ARI_LIMITS), letter
        plt.close(fig)

    def test_oracle_reads_only_the_datasets_the_arm_ran(self, tmp_path):
        # As in the registry, the B arm runs a subset of the rho arm's
        # datasets. Dataset 1, which only the rho arm runs, gets an oracle
        # of 0: a figure that averaged over every dataset would draw B's
        # oracle at 0.45 rather than 0.9, and A's must include it.
        scale = NO_STUDY.scales["test"]
        ablation = dataclasses.replace(
            NO_STUDY,
            scales={"test": dataclasses.replace(
                scale, b_arm=dataclasses.replace(scale.b_arm, datasets=(0,))
            )},
        )
        frames = synthetic_ablation_frames(ablation, "test", seed=5, tmp_path=tmp_path)
        datasets = frames["datasets"].copy()
        assert set(datasets["dataset"]) == {0, 1}
        datasets.loc[datasets["dataset"] == 1, "oracle_ari"] = 0.0
        fig = figure_ablation(
            {**frames, "datasets": datasets}, ablation=ablation, scale="test", save=False
        )
        for letter, value in {"A": 0.45, "B": 0.9}.items():
            (oracle,) = [
                line for line in _panels(fig)[letter].lines
                if to_hex(line.get_color()) == to_hex(metric_color("baseline_oracle"))
            ]
            assert oracle.get_ydata()[0] == pytest.approx(value), letter
        plt.close(fig)

    def test_one_legend_names_the_selectors_the_interval_and_the_defaults(self, figure):
        (legend,) = figure.legends
        assert [t.get_text() for t in legend.get_texts()] == [
            METRIC_DISPLAY_NAMES[STAB],
            METRIC_DISPLAY_NAMES[GEN],
            "95% CI",
            f"Default ($\\rho$ = {NO_STUDY.rho_default:g}, $B$ = {NO_STUDY.b_default:g})",
        ]

    def test_titles_name_the_parameter_held_fixed(self, figure):
        panels = _panels(figure)
        assert f"{NO_STUDY.b_default:g}" in panels["A"].get_title(loc="left")
        assert f"{NO_STUDY.rho_default:g}" in panels["B"].get_title(loc="left")

    def test_no_scenario_name_reaches_the_legend(self, figure):
        # A single scenario is the pooled simulations; the figure names the
        # selectors, not the scenario.
        labels = [t.get_text() for t in figure.legends[0].get_texts()]
        assert "Gaussian Mixtures" not in labels
        assert "Pooled (simulations)" not in labels
