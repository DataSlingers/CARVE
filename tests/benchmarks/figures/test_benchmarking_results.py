"""Tests for Fig 4."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarks._artifacts import SCHEMA
from benchmarks._theme import METRIC_COLORS
from benchmarks.figures import figure_benchmarking_results
from benchmarks.figures._benchmarking_results import DEFAULT_METRICS

METRICS = ("ari_stability_1se", "ari_generalizability_1se", "silhouette")

# Every series the default legend expects. "baseline_oracle" is not one of
# them: it names the oracle_ari column rather than a value of metric_name,
# and every row already carries it.
ALL_METRICS = tuple(m for m in DEFAULT_METRICS if m != "baseline_oracle")


def _frame(scenario: str, metrics: tuple[str, ...] = METRICS) -> pd.DataFrame:
    rows = []
    for axis_value, axis_label in enumerate(("easy", "medium", "hard")):
        for seed in range(4):
            for metric in metrics:
                for k in (4, 5, 6):
                    rows.append(
                        {
                            "run_id": "r1",
                            "scenario": scenario,
                            "axis_name": "difficulty_level",
                            "axis_value": axis_value,
                            "axis_label": axis_label,
                            "seed": seed,
                            "k_star": 5,
                            "estimator": "kmeans",
                            "metric_name": metric,
                            "k": k,
                            "metric_value": 0.1 * k,
                            "is_selected": k == 5,
                            "selects_true_k": k == 5,
                            "ari_at_k": 0.9 - 0.1 * axis_value,
                            "oracle_ari": 0.95,
                        }
                    )
    return pd.DataFrame(rows)[list(SCHEMA)]


@pytest.fixture
def results():
    return {name: _frame(name) for name in ("gaussians", "t_dist", "moons")}


@pytest.fixture
def full_results():
    """Frames carrying every metric the default legend draws."""
    return {
        name: _frame(name, ALL_METRICS) for name in ("gaussians", "t_dist", "moons")
    }


class TestFigureBenchmarkingResults:
    def test_returns_a_figure(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        assert fig.get_axes()
        plt.close(fig)

    def test_one_panel_per_scenario(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        drawn = [ax for ax in fig.get_axes() if ax.lines or ax.collections]
        assert len(drawn) == len(results)
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, results, tmp_path):
        fig = figure_benchmarking_results(
            results, metrics=METRICS, save=True, out_dir=tmp_path
        )
        assert (tmp_path / "benchmarking_results.png").exists()
        plt.close(fig)

    def test_save_false_writes_nothing(self, results, tmp_path):
        fig = figure_benchmarking_results(
            results, metrics=METRICS, save=False, out_dir=tmp_path
        )
        assert not (tmp_path / "benchmarking_results.png").exists()
        plt.close(fig)

    def test_carries_one_deduplicated_figure_legend(self, results):
        """One Legend, not one per metric family positioned by hand.

        Blank entries pad the short columns, so the count is of the labelled
        ones rather than of every text the legend holds.
        """
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        assert len(fig.legends) == 1
        labelled = [t.get_text() for t in fig.legends[0].get_texts() if t.get_text()]
        assert len(labelled) == len(METRICS)
        plt.close(fig)

    def test_panels_carry_no_letters(self, results):
        fig = figure_benchmarking_results(results, metrics=METRICS, save=False)
        assert [t.get_text() for ax in fig.get_axes() for t in ax.texts] == []
        plt.close(fig)

    def test_x_axis_is_named_for_the_sweep(self, full_results):
        fig = figure_benchmarking_results(full_results, save=False)
        assert {ax.get_xlabel() for ax in fig.get_axes()} == {"SNR"}
        plt.close(fig)

    def test_carries_no_k_star_title(self, full_results):
        fig = figure_benchmarking_results(full_results, save=False)
        assert fig._suptitle is None
        plt.close(fig)

    def test_one_figure_wide_y_label_and_none_on_the_panels(self, full_results):
        """Two column-leading y labels became a single centered one."""
        fig = figure_benchmarking_results(full_results, save=False)
        assert fig.supylabel is not None
        assert {ax.get_ylabel() for ax in fig.get_axes()} == {""}
        texts = [t.get_text() for t in fig.texts]
        assert sum(r"vs. true labels" in t for t in texts) == 1
        plt.close(fig)

    def test_legend_columns_follow_the_published_grouping(self, full_results):
        """Oracle alone, CARVE stacked, the four indices across two columns.

        Matplotlib fills a legend column by column, so reading the labels in
        order and chunking by the row count recovers the columns as drawn.
        The blank columns are the family separators.
        """
        fig = figure_benchmarking_results(full_results, save=False)
        legend = fig.legends[0]
        labels = [t.get_text() for t in legend.get_texts()]
        # Legend._ncols is private; it was _ncol before matplotlib 3.6 and the
        # matplotlib>=3.9.4 floor makes the current name safe.
        rows = len(labels) // legend._ncols
        columns = [labels[i * rows : (i + 1) * rows] for i in range(legend._ncols)]
        drawn = [c for c in columns if any(c)]
        assert drawn[0] == ["Baseline (Oracle k*)", ""]
        assert drawn[1] == ["CARVE Stability (1SE)", "CARVE Generalizability (1SE)"]
        assert drawn[2] == ["Silhouette", "Davies-Bouldin"]
        assert drawn[3] == ["Calinski-Harabasz", "Gap Statistic"]
        plt.close(fig)

    def test_the_classical_pair_is_not_split_by_a_separating_column(self, full_results):
        """The two index columns must sit adjacent, at the narrow gutter."""
        fig = figure_benchmarking_results(full_results, save=False)
        legend = fig.legends[0]
        labels = [t.get_text() for t in legend.get_texts()]
        rows = len(labels) // legend._ncols
        columns = [labels[i * rows : (i + 1) * rows] for i in range(legend._ncols)]
        blank = [index for index, c in enumerate(columns) if not any(c)]
        assert blank == [1, 3]
        plt.close(fig)

    def test_only_the_carve_series_are_solid(self, full_results):
        """CARVE solid, oracle and every classical index dashed."""
        fig = figure_benchmarking_results(full_results, save=False)
        # An errorbar draws cap and bar Line2Ds alongside the data line, and
        # those carry linestyle "None". container[0] is the data line.
        styles = {
            container[0].get_color(): container[0].get_linestyle()
            for ax in fig.get_axes()
            for container in ax.containers
        }
        for metric in ("ari_stability_1se", "ari_generalizability_1se"):
            assert styles[METRIC_COLORS[metric]] == "-"
        for metric in (
            "baseline_oracle",
            "silhouette",
            "gap",
            "davies_bouldin",
            "calinski_harabasz",
        ):
            assert styles[METRIC_COLORS[metric]] == "--"
        plt.close(fig)

    def test_carve_stability_is_drawn_in_the_theme_green(self, results):
        fig = figure_benchmarking_results(
            results, metrics=("ari_stability_1se",), save=False
        )
        colors = {ln.get_color() for ax in fig.get_axes() for ln in ax.lines}
        assert METRIC_COLORS["ari_stability_1se"] in colors
        plt.close(fig)

    def test_panels_are_ordered_by_the_canonical_reading_order(self, full_results):
        """Not by the caller's mapping order.

        The notebook builds its mapping by iterating SCENARIOS, whose order
        exists for the runner. Passing the families in a deliberately wrong
        order here proves the figure imposes its own.
        """
        from benchmarks.figures._benchmarking_examples import SCENARIO_TITLES

        scrambled = {
            name: _frame(name, ALL_METRICS)
            for name in ("moons", "gaussians", "swiss_rolls", "t_dist")
        }
        fig = figure_benchmarking_results(scrambled, save=False)
        titles = [ax.get_title() for ax in fig.get_axes() if ax.get_title()]
        assert titles == [
            SCENARIO_TITLES[name]
            for name in ("gaussians", "t_dist", "swiss_rolls", "moons")
        ]
        plt.close(fig)

    def test_raises_on_an_empty_mapping(self):
        with pytest.raises(ValueError, match="at least one scenario"):
            figure_benchmarking_results({}, save=False)

    def test_rejects_a_scaling_frame_instead_of_mislabeling_its_axis(self, results):
        """The x ticks are relabeled easy/medium/hard unconditionally.

        Handed a scaling sweep, this function would draw n_total on an axis
        reading "Difficulty" with ticks reading easy/medium/hard -- a wrong
        figure that looks right. figure_scaling_ari and
        figure_scenario_overview are the paths that handle those sweeps.
        """
        scaling = _frame("gaussians_samples").assign(
            axis_name="n_total", axis_value=1000
        )
        with pytest.raises(ValueError, match="difficulty_level"):
            figure_benchmarking_results(
                {**results, "gaussians_samples": scaling},
                metrics=METRICS,
                save=False,
            )

    def test_default_metrics_draw_the_oracle_baseline(self, full_results):
        # Fig 4's caption: "The grey line shows the oracle ARI(k*=5)". Every
        # frame here carries oracle_ari, so the default metrics (used when
        # no explicit metrics= is passed) must include the baseline series.
        fig = figure_benchmarking_results(full_results, save=False)
        colors = {ln.get_color() for ax in fig.get_axes() for ln in ax.lines}
        assert METRIC_COLORS["baseline_oracle"] in colors
        plt.close(fig)
