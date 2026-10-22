"""Tests for the two SI ablation figures, on synthetic frames."""

import dataclasses

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text

from benchmarks._ablation_cells import REFERENCE_RATIO, arm_view
from benchmarks._ablation_summary import (
    HEADLINE_METRICS,
    ari_summary,
    bias_summary,
    rare_recall_summary,
    similarity_at_k_star,
)
from benchmarks._registry import ABLATIONS, METRIC_DISPLAY_NAMES
from benchmarks._theme import (
    FALLBACK_COLOR,
    PRINT_MAX_HEIGHT_IN,
    PRINT_WIDTH_IN,
    metric_color,
)
from benchmarks.figures import figure_ablation, figure_ablation_resampling
from benchmarks.figures._ablation import (
    DODGE_PT,
    MIN_Y_SPAN,
    NEUTRAL_COLOR,
    STUDY_TITLES,
    Y_PAD,
    Z_95,
    fitted_limits,
)
from tests.benchmarks._helpers import small_ablation, synthetic_ablation_frames

# The registry's ablation runs gaussians at medium and no case study. The
# tests run circles beside it, so the one-row-per-scenario layout and the
# separation of the scenarios stay covered.
TWO_SCENARIOS = ("gaussians", "circles")
RHO_B = small_ablation(ABLATIONS["rho_b"], study="klein", scenarios=TWO_SCENARIOS)
STAB, GEN = HEADLINE_METRICS
SELECTOR_COLORS = {to_hex(metric_color(STAB)), to_hex(metric_color(GEN))}
RHO, B = "subsample_ratio", "n_resamples"
NO_STUDY = small_ablation(ABLATIONS["rho_b"], scenarios=TWO_SCENARIOS)
GAUSSIANS, CIRCLES = NO_STUDY.scenarios
# figure_ablation's panels: letter to (scenario, arm, swept column).
ARI_PANELS = {
    "A": (GAUSSIANS, "rho", RHO),
    "B": (GAUSSIANS, "b", B),
    "C": (CIRCLES, "rho", RHO),
    "D": (CIRCLES, "b", B),
}
FIGURES = [
    (figure_ablation, "si_fig_ablation.png"),
    (figure_ablation_resampling, "si_fig_ablation_resampling.png"),
]


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
def registry_scenarios(tmp_path_factory):
    # The registry's own scenarios, at the test scale.
    ablation = small_ablation(ABLATIONS["rho_b"])
    return ablation, synthetic_ablation_frames(
        ablation, "test", seed=5, tmp_path=tmp_path_factory.mktemp("registry_frames")
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


def _selector_bars(ax, metric):
    """The selector's error-bar container: lines[0] is its mean line and
    lines[2] its bars."""
    (container,) = [
        c for c in ax.containers
        if to_hex(c.lines[0].get_color()) == to_hex(metric_color(metric))
    ]
    return container


def _selector_line(ax, metric):
    return _selector_bars(ax, metric).lines[0]


def _bar_ends(container):
    """The lower and upper end of each error bar, in x order."""
    (bars,) = container.lines[2]
    segments = bars.get_segments()
    return np.array([s[0][1] for s in segments]), np.array([s[1][1] for s in segments])


def _drawn_lines(ax):
    """Lines with more than two points that draw a line, not caps alone."""
    return [
        line for line in ax.lines
        if len(line.get_xdata()) > 2 and line.get_linestyle() != "None"
    ]


def _per_study(summary, study, metric, x):
    rows = summary[(summary["study"] == study) & (summary["metric_name"] == metric)]
    return rows.sort_values(x)


def _ari(frames, ablation, arm, x, study, metric):
    view = arm_view(frames, ablation=ablation, scale="test", arm=arm)
    return _per_study(ari_summary(view["selection"], view["at_k"], x=x), study, metric, x)


class TestPrintSize:
    @pytest.mark.parametrize(("figure_fn", "name"), FIGURES)
    @pytest.mark.parametrize("which", ["with_study", "without_study"])
    def test_drawn_at_the_plos_page_width(self, figure_fn, name, which, request):
        ablation, frames = request.getfixturevalue(which)
        fig = figure_fn(frames, ablation=ablation, scale="test", save=False)
        width, height = fig.get_size_inches()
        assert width == pytest.approx(PRINT_WIDTH_IN)
        assert height <= PRINT_MAX_HEIGHT_IN
        plt.close(fig)

    @pytest.mark.parametrize(("figure_fn", "name"), FIGURES)
    def test_every_text_is_within_plos_sizes(self, figure_fn, name, with_study):
        ablation, frames = with_study
        fig = figure_fn(frames, ablation=ablation, scale="test", save=False)
        fig.canvas.draw()
        sizes = [
            text.get_fontsize() for text in fig.findobj(Text)
            if text.get_visible() and text.get_text().strip()
        ]
        assert sizes
        assert all(8.0 <= size <= 12.0 for size in sizes)
        plt.close(fig)

    @pytest.mark.parametrize(("figure_fn", "name"), FIGURES)
    def test_the_saved_file_fits_the_page_at_300_dpi(self, figure_fn, name, with_study, tmp_path):
        ablation, frames = with_study
        fig = figure_fn(frames, ablation=ablation, scale="test", out_dir=tmp_path)
        height, width = mpimg.imread(tmp_path / name).shape[:2]
        assert width <= 2250
        assert height <= 2625
        plt.close(fig)


class TestPanelLabels:
    @pytest.mark.parametrize(("figure_fn", "name"), FIGURES)
    def test_no_letter_overlaps_its_title(self, figure_fn, name, with_study):
        ablation, frames = with_study
        fig = figure_fn(frames, ablation=ablation, scale="test", save=False)
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
    def test_one_row_per_scenario_without_a_study(self, figure):
        assert isinstance(figure, Figure)
        assert list(_panels(figure)) == list("ABCD")

    def test_the_registrys_gaussians_alone_is_one_row(self, registry_scenarios, figure):
        ablation, frames = registry_scenarios
        assert ablation.scenarios == ("gaussians",)
        fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
        panels = _panels(fig)
        assert list(panels) == list("AB")
        assert fig.get_size_inches()[1] < figure.get_size_inches()[1]
        # One scenario needs no name in the titles; the caption gives it.
        for ax in panels.values():
            assert STUDY_TITLES["gaussians"] not in ax.get_title(loc="left")
        plt.close(fig)

    def test_a_study_adds_a_row(self, with_study, figure):
        ablation, frames = with_study
        fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
        assert list(_panels(fig)) == list("ABCDEF")
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
        for letter in "ACE":
            ax = panels[letter]
            assert ax.get_xscale() == "linear"
            assert _has_default_line(ax, ablation.rho_default)
        for letter in "BDF":
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

    @pytest.mark.parametrize("letter", sorted(ARI_PANELS))
    def test_lines_are_each_scenarios_stability_consensus_ari_at_the_selected_k(
        self, without_study, figure, letter
    ):
        # The synthetic frames draw the two scenarios' values independently,
        # so a figure that pooled them or swapped rows would miss these. They
        # also draw ari_selected and at_k independently, so a figure that
        # plotted the stored ari_selected would miss them too.
        ablation, frames = without_study
        study, arm, x = ARI_PANELS[letter]
        ax = _panels(figure)[letter]
        for metric in HEADLINE_METRICS:
            expected = _ari(frames, ablation, arm, x, study, metric)
            line = _selector_line(ax, metric)
            assert list(line.get_xdata()) == pytest.approx(list(expected[x]))
            assert list(line.get_ydata()) == pytest.approx(list(expected["ari_mean"]))
        stored = arm_view(frames, ablation=ablation, scale="test", arm=arm)["selection"]
        stored = stored[(stored["study"] == study) & (stored["metric_name"] == STAB)]
        stored = stored.groupby(x)["ari_selected"].mean()
        assert list(_selector_line(ax, STAB).get_ydata()) != pytest.approx(list(stored))

    def test_error_bars_are_the_95_percent_interval_over_datasets(self, without_study, figure):
        ablation, frames = without_study
        expected = _ari(frames, ablation, "b", B, GAUSSIANS, STAB)
        lower, upper = _bar_ends(_selector_bars(_panels(figure)["B"], STAB))
        mean, sem = expected["ari_mean"], expected["ari_sem"]
        assert list(lower) == pytest.approx(list(mean - Z_95 * sem))
        assert list(upper) == pytest.approx(list(mean + Z_95 * sem))

    def test_one_line_per_selector_and_no_seed_lines(self, figure):
        # The B arm of the test scale runs two seeds; a line per seed would
        # add two more lines per selector.
        for letter, ax in _panels(figure).items():
            colors = [to_hex(line.get_color()) for line in _drawn_lines(ax)]
            assert sorted(colors) == sorted(SELECTOR_COLORS), letter

    def test_the_selectors_are_nudged_apart_by_the_same_points_on_both_axes(self, figure):
        # Stability sits DODGE_PT points left of generalizability, on the
        # linear rho axis and the log B axis alike.
        figure.canvas.draw()
        expected = DODGE_PT * figure.dpi / 72
        for letter in "AB":
            ax = _panels(figure)[letter]
            stab, gen = (_selector_line(ax, metric) for metric in (STAB, GEN))
            assert list(stab.get_xdata()) == list(gen.get_xdata())
            point = (stab.get_xdata()[0], stab.get_ydata()[0])
            shift = gen.get_transform().transform(point)[0] - stab.get_transform().transform(point)[0]
            assert shift == pytest.approx(expected), letter

    def test_every_ari_panel_shares_one_fitted_axis(self, with_study):
        ablation, frames = with_study
        fig = figure_ablation(frames, ablation=ablation, scale="test", save=False)
        panels = _panels(fig)
        limits = {ax.get_ylim() for ax in panels.values()}
        assert len(limits) == 1
        (lo, hi) = limits.pop()
        assert hi - lo >= MIN_Y_SPAN
        assert (lo, hi) != pytest.approx((0.0, 1.0))
        for letter, ax in panels.items():
            for metric in HEADLINE_METRICS:
                lower, upper = _bar_ends(_selector_bars(ax, metric))
                assert lo <= lower.min() and upper.max() <= hi, (letter, metric)
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
        for letter, value in {"A": 0.45, "B": 0.9, "C": 0.45, "D": 0.9}.items():
            (oracle,) = [
                line for line in _panels(fig)[letter].lines
                if to_hex(line.get_color()) == to_hex(metric_color("baseline_oracle"))
            ]
            assert oracle.get_ydata()[0] == pytest.approx(value), letter
        plt.close(fig)

    def test_a_scenario_with_no_cells_yet_draws_no_oracle(self, without_study):
        # The notebook can be opened during a run, before any circles cell
        # has finished. Every table that carries a study column loses its
        # circles rows, as a run directory without circles units would.
        ablation, frames = without_study
        partial = {
            name: frame[frame["study"] != CIRCLES] if "study" in frame else frame
            for name, frame in frames.items()
        }
        assert all(CIRCLES not in set(f["study"]) for f in partial.values() if "study" in f)
        fig = figure_ablation(partial, ablation=ablation, scale="test", save=False)
        panels = _panels(fig)
        assert list(panels) == list("ABCD")
        for letter, n_oracle in {"A": 1, "B": 1, "C": 0, "D": 0}.items():
            lines = [
                line for line in panels[letter].lines
                if to_hex(line.get_color()) == to_hex(metric_color("baseline_oracle"))
            ]
            assert len(lines) == n_oracle, letter
        plt.close(fig)

    def test_each_row_reads_its_own_scenarios_oracle(self, without_study):
        ablation, frames = without_study
        datasets = frames["datasets"].copy()
        datasets["oracle_ari"] = datasets["study"].map({GAUSSIANS: 0.9, CIRCLES: 0.3})
        fig = figure_ablation(
            {**frames, "datasets": datasets}, ablation=ablation, scale="test", save=False
        )
        for letter, value in {"A": 0.9, "B": 0.9, "C": 0.3, "D": 0.3}.items():
            (oracle,) = [
                line for line in _panels(fig)[letter].lines
                if to_hex(line.get_color()) == to_hex(metric_color("baseline_oracle"))
            ]
            assert oracle.get_ydata()[0] == pytest.approx(value), letter
        plt.close(fig)

    def test_one_legend_names_the_selectors_the_oracle_the_interval_and_the_defaults(self, figure):
        (legend,) = figure.legends
        assert [t.get_text() for t in legend.get_texts()] == [
            METRIC_DISPLAY_NAMES[STAB],
            METRIC_DISPLAY_NAMES[GEN],
            "Oracle $k^*$",
            "95% CI",
            f"Default ($\\rho$ = {NO_STUDY.rho_default:g}, $B$ = {NO_STUDY.b_default:g})",
        ]

    def test_titles_name_the_scenario_and_the_parameter_held_fixed(self, figure):
        panels = _panels(figure)
        for letter, (study, arm, _) in ARI_PANELS.items():
            title = panels[letter].get_title(loc="left")
            assert STUDY_TITLES[study] in title, letter
            held = NO_STUDY.b_default if arm == "rho" else NO_STUDY.rho_default
            assert f"{held:g}" in title, letter

    def test_no_scenario_name_reaches_the_legend(self, figure):
        # The titles name the scenarios; the legend names the selectors.
        labels = [t.get_text() for t in figure.legends[0].get_texts()]
        for study in (GAUSSIANS, CIRCLES):
            assert STUDY_TITLES[study] not in labels
        assert "Pooled (simulations)" not in labels


@pytest.fixture(scope="module")
def resampling(without_study):
    ablation, frames = without_study
    fig = figure_ablation_resampling(frames, ablation=ablation, scale="test", save=False)
    yield fig
    plt.close(fig)


def _rho_view(frames, ablation):
    return arm_view(frames, ablation=ablation, scale="test", arm="rho")


# figure_ablation_resampling's rows: scenario to its (bias, recall,
# similarity) panel letters.
RESAMPLING_PANELS = {GAUSSIANS: "ABC", CIRCLES: "DEF"}


class TestFigureAblationResampling:
    def test_one_row_of_three_panels_per_scenario(self, resampling):
        assert isinstance(resampling, Figure)
        assert list(_panels(resampling)) == list("ABCDEF")

    def test_the_registrys_gaussians_alone_is_one_row(self, registry_scenarios):
        ablation, frames = registry_scenarios
        fig = figure_ablation_resampling(frames, ablation=ablation, scale="test", save=False)
        assert list(_panels(fig)) == list("ABC")
        plt.close(fig)

    def test_a_study_adds_no_panel(self, with_study):
        # Bias and rare-cluster recall need a true k, and the subsample
        # comparison is read at k*, so the study has nothing to draw here.
        ablation, frames = with_study
        fig = figure_ablation_resampling(frames, ablation=ablation, scale="test", save=False)
        assert list(_panels(fig)) == list("ABCDEF")
        plt.close(fig)

    def test_saves_under_its_si_name(self, without_study, tmp_path):
        ablation, frames = without_study
        fig = figure_ablation_resampling(frames, ablation=ablation, scale="test", out_dir=tmp_path)
        assert [p.name for p in tmp_path.iterdir()] == ["si_fig_ablation_resampling.png"]
        plt.close(fig)

    def test_save_false_writes_nothing(self, without_study, tmp_path):
        ablation, frames = without_study
        fig = figure_ablation_resampling(
            frames, ablation=ablation, scale="test", save=False, out_dir=tmp_path
        )
        assert not list(tmp_path.iterdir())
        plt.close(fig)

    def test_every_panel_sweeps_rho_and_marks_the_default(self, resampling):
        for letter, ax in _panels(resampling).items():
            assert ax.get_xscale() == "linear", letter
            assert _has_default_line(ax, NO_STUDY.rho_default), letter

    def test_the_first_panel_of_a_row_names_its_scenario(self, resampling):
        panels = _panels(resampling)
        for study, letters in RESAMPLING_PANELS.items():
            assert STUDY_TITLES[study] in panels[letters[0]].get_title(loc="left"), study

    @pytest.mark.parametrize("study", [GAUSSIANS, CIRCLES])
    def test_bias_lines_and_bands_over_datasets(self, without_study, resampling, study):
        ablation, frames = without_study
        summary = bias_summary(_rho_view(frames, ablation)["selection"], x=RHO)
        ax = _panels(resampling)[RESAMPLING_PANELS[study][0]]
        for metric in HEADLINE_METRICS:
            rows = _per_study(summary, study, metric, RHO)
            assert list(_selector_line(ax, metric).get_ydata()) == pytest.approx(list(rows["bias_mean"]))
            lower, upper = _bar_ends(_selector_bars(ax, metric))
            assert list(lower) == pytest.approx(list(rows["bias_mean"] - Z_95 * rows["bias_sem"]))
            assert list(upper) == pytest.approx(list(rows["bias_mean"] + Z_95 * rows["bias_sem"]))
        zero = [line for line in ax.lines if list(line.get_ydata()) == [0.0, 0.0]]
        assert len(zero) == 1

    @pytest.mark.parametrize("study", [GAUSSIANS, CIRCLES])
    def test_rare_recall_draws_each_rule_at_k_hat_and_one_line_at_k_star(
        self, without_study, resampling, study
    ):
        ablation, frames = without_study
        view = _rho_view(frames, ablation)
        difficulty = ablation.scales["test"].rho_arm.difficulties[-1]
        summary = rare_recall_summary(view["at_k"], view["selection"], x=RHO, difficulty=difficulty)
        ax = _panels(resampling)[RESAMPLING_PANELS[study][1]]
        lines = [line for line in ax.lines if len(line.get_xdata()) > 2]
        solid = [line for line in lines if line.get_linestyle() == "-"]
        dashed = [line for line in lines if line.get_linestyle() == "--"]
        assert len(solid) == 2 and len(dashed) == 1
        for metric in HEADLINE_METRICS:
            rows = _per_study(summary, study, metric, RHO)
            assert list(_selector_line(ax, metric).get_ydata()) == pytest.approx(list(rows["recall_selected"]))
        # Recall at k* reads the same labels for both rules, so the two
        # rules' values agree and the one dashed line in the neutral color
        # carries them.
        stab = _per_study(summary, study, STAB, RHO)["recall_k_star"]
        gen = _per_study(summary, study, GEN, RHO)["recall_k_star"]
        assert list(stab) == pytest.approx(list(gen))
        assert list(dashed[0].get_ydata()) == pytest.approx(list(stab))
        assert to_hex(dashed[0].get_color()) == to_hex(NEUTRAL_COLOR)

    def test_proportion_panels_are_fitted_to_what_they_draw(self, resampling):
        # Recall and subsample-versus-full ARI, each on its own axis fitted
        # to its error bars and reference line; the old recall panel ran to
        # 1.32 to hold an in-panel key.
        for letters in RESAMPLING_PANELS.values():
            for letter in letters[1:]:
                ax = _panels(resampling)[letter]
                ends = [np.concatenate(_bar_ends(c)) for c in ax.containers if c.has_yerr]
                lines = [line.get_ydata() for line in _drawn_lines(ax)]
                lo, hi = fitted_limits(np.concatenate([*ends, *lines]))
                pad = Y_PAD * (hi - lo)
                assert ax.get_ylim() == pytest.approx((lo - pad, hi + pad)), letter
                assert hi - lo >= MIN_Y_SPAN, letter

    @pytest.mark.parametrize("study", [GAUSSIANS, CIRCLES])
    def test_similarity_line_stops_before_the_refit_reference(
        self, without_study, resampling, study
    ):
        ablation, frames = without_study
        summary = similarity_at_k_star(frames["similarity"], frames["datasets"])
        rows = summary[summary["study"] == study].sort_values(RHO)
        swept = rows[rows[RHO] < REFERENCE_RATIO]
        ax = _panels(resampling)[RESAMPLING_PANELS[study][2]]
        (line,) = _drawn_lines(ax)
        assert list(line.get_xdata()) == pytest.approx(list(swept[RHO]))
        assert list(line.get_ydata()) == pytest.approx(list(swept["ari_mean"]))
        assert to_hex(line.get_color()) == to_hex(NEUTRAL_COLOR)
        (refit,) = [
            line for line in ax.lines
            if list(line.get_xdata()) == [REFERENCE_RATIO] and line.get_marker() == "o"
        ]
        assert refit.get_ydata()[0] == pytest.approx(
            float(rows[rows[RHO] == REFERENCE_RATIO]["ari_mean"].iloc[0])
        )
        assert [t.get_text() for t in ax.get_xticklabels()][-1] == "1.0"

    def test_one_legend_names_the_selectors_the_references_the_interval_and_the_default(
        self, resampling
    ):
        (legend,) = resampling.legends
        assert [t.get_text() for t in legend.get_texts()] == [
            METRIC_DISPLAY_NAMES[STAB],
            METRIC_DISPLAY_NAMES[GEN],
            "Recall at $k^*$",
            "Full-data refit",
            "95% CI",
            f"Default $\\rho$ = {NO_STUDY.rho_default:g}",
        ]


class TestFittedLimits:
    def test_rounds_out_to_whole_steps_and_widens_to_the_minimum_span(self):
        assert fitted_limits([0.83, 0.86]) == pytest.approx((0.75, 0.95))

    def test_a_range_wider_than_the_minimum_is_only_rounded_out(self):
        assert fitted_limits([0.12, 0.88]) == pytest.approx((0.1, 0.9))

    def test_widening_stays_inside_the_bounds(self):
        assert fitted_limits([0.95, 0.99]) == pytest.approx((0.8, 1.0))
        assert fitted_limits([0.01, 0.04]) == pytest.approx((0.0, 0.2))

    def test_values_outside_the_bounds_are_kept(self):
        # An error bar that crosses 0 or 1 is drawn whole.
        assert fitted_limits([0.52, 1.03]) == pytest.approx((0.5, 1.05))
        assert fitted_limits([-0.37, 0.4]) == pytest.approx((-0.4, 0.4))
        assert fitted_limits([-0.02, 0.06]) == pytest.approx((-0.05, 0.15))

    def test_no_finite_value_gives_the_bounds(self):
        assert fitted_limits([np.nan]) == (0.0, 1.0)
