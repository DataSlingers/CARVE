"""Tests for the unified summarizer."""

import numpy as np
import pandas as pd
import pytest

from benchmarks._artifacts import SCHEMA
from benchmarks._tables import (
    _tex_escape,
    render_grouped_tex,
    summarize,
    summary_stats,
    wilson_ci,
    write_tables,
)


def _frame():
    """Two axis labels x two metrics x two seeds x three k values."""
    rows = []
    for axis_label in ("easy", "hard"):
        for seed in range(2):
            for metric in ("ari_stability_1se", "silhouette"):
                selected = 5 if metric == "ari_stability_1se" else 4
                for k in (3, 4, 5):
                    rows.append(
                        {
                            "run_id": "r1",
                            "scenario": "demo",
                            "axis_name": "difficulty_level",
                            "axis_value": 0 if axis_label == "easy" else 2,
                            "axis_label": axis_label,
                            "seed": seed,
                            "k_star": 5,
                            "estimator": "kmeans",
                            "metric_name": metric,
                            "k": k,
                            "metric_value": 0.1 * k,
                            "is_selected": k == selected,
                            "selects_true_k": k == 5,
                            "ari_at_k": 0.5 + 0.1 * k,
                            "oracle_ari": 0.9,
                        }
                    )
    return pd.DataFrame(rows)[list(SCHEMA)]


def _scaling_frame():
    """Three axis points whose labels sort differently than their values.

    "end" < "middle" < "start" alphabetically, but the intended reading
    order -- the order SCALING_AXES declares, matching increasing
    axis_value -- is start (1000), middle (5500), end (10000). Rows are
    built in the same order read_run would hand back from a lexically
    sorted glob of cell__<label>__<seed>.parquet checkpoints (end, middle,
    start), so a summarizer that orders columns by first-appearance-in-that-
    row-order reproduces the bug: end, middle, start.
    """
    rows = []
    for axis_value, axis_label in ((10000, "end"), (5500, "middle"), (1000, "start")):
        for seed in range(2):
            for k in (4, 5):
                rows.append(
                    {
                        "run_id": "r1",
                        "scenario": "demo",
                        "axis_name": "n_total",
                        "axis_value": axis_value,
                        "axis_label": axis_label,
                        "seed": seed,
                        "k_star": 5,
                        "estimator": "kmeans",
                        "metric_name": "ari_stability_1se",
                        "k": k,
                        "metric_value": 0.1 * k,
                        "is_selected": k == 5,
                        "selects_true_k": k == 5,
                        "ari_at_k": 0.8,
                        "oracle_ari": 0.9,
                    }
                )
    return pd.DataFrame(rows)[list(SCHEMA)]


class TestWilsonCi:
    def test_brackets_the_point_estimate(self):
        low, high = wilson_ci(7, 10)
        assert low < 0.7 < high

    def test_is_bounded_by_zero_and_one(self):
        low, high = wilson_ci(0, 10)
        assert low >= 0.0
        high_low, high_high = wilson_ci(10, 10)
        assert high_high <= 1.0

    def test_returns_nan_for_an_empty_sample(self):
        low, high = wilson_ci(0, 0)
        assert np.isnan(low) and np.isnan(high)


class TestSummaryStats:
    def test_reports_mean_sd_median_and_quartiles(self):
        stats = summary_stats(pd.Series([0.0, 1.0, 2.0, 3.0, 4.0]))
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["median"] == pytest.approx(2.0)
        assert stats["q25"] == pytest.approx(1.0)
        assert stats["q75"] == pytest.approx(3.0)

    def test_sd_is_zero_for_a_single_value(self):
        assert summary_stats(pd.Series([1.0]))["sd"] == 0.0

    def test_empty_input_gives_nan(self):
        assert np.isnan(summary_stats(pd.Series([], dtype=float))["mean"])


class TestSummarize:
    def test_one_row_per_axis_label_and_metric(self):
        out = summarize(_frame())
        assert len(out) == 2 * 2

    def test_uses_only_selected_rows(self):
        out = summarize(_frame())
        row = out[
            (out["axis_label"] == "easy") & (out["metric"] == "ari_stability_1se")
        ].iloc[0]
        # selected k is 5, so ari_at_k is 0.5 + 0.5 = 1.0
        assert row["ari_mean"] == pytest.approx(1.0)

    def test_k_recovery_reflects_selects_true_k(self):
        out = summarize(_frame())
        stability = out[out["metric"] == "ari_stability_1se"]
        silhouette = out[out["metric"] == "silhouette"]
        assert (stability["k_recovery"] == 1.0).all()
        assert (silhouette["k_recovery"] == 0.0).all()

    def test_delta_to_oracle_is_oracle_minus_selected_ari(self):
        out = summarize(_frame())
        row = out[(out["axis_label"] == "easy") & (out["metric"] == "silhouette")].iloc[
            0
        ]
        # selected k is 4, ari_at_k 0.9, oracle 0.9
        assert row["delta_mean"] == pytest.approx(0.0)

    def test_k_bias_is_selected_k_minus_k_star(self):
        out = summarize(_frame())
        row = out[out["metric"] == "silhouette"].iloc[0]
        assert row["k_bias_median"] == pytest.approx(-1.0)

    def test_over_and_under_rates_sum_with_recovery_to_one(self):
        out = summarize(_frame())
        for _, row in out.iterrows():
            assert row["p_under"] + row["p_over"] + row["k_recovery"] == pytest.approx(
                1.0
            )

    def test_restricting_metrics_filters_the_output(self):
        out = summarize(_frame(), metrics=("silhouette",))
        assert set(out["metric"]) == {"silhouette"}

    def test_raises_when_no_requested_metric_is_present(self):
        with pytest.raises(ValueError, match="none of the requested metrics"):
            summarize(_frame(), metrics=("nonexistent",))

    def test_axis_labels_are_ordered_by_axis_value_not_row_order(self):
        out = summarize(_scaling_frame())
        label_order = list(dict.fromkeys(out["axis_label"]))
        assert label_order == ["start", "middle", "end"]


class TestSummarizeBaseline:
    """The Baseline (Oracle) row every published S2-S9 table starts with."""

    def test_included_only_when_requested(self):
        out = summarize(_frame(), metrics=("silhouette",))
        assert "baseline_oracle" not in set(out["metric"])

    def test_baseline_oracle_row_is_included_when_requested(self):
        out = summarize(_frame(), metrics=("baseline_oracle", "silhouette"))
        assert "baseline_oracle" in set(out["metric"])

    def test_is_the_first_metric_for_every_axis_label(self):
        out = summarize(
            _frame(), metrics=("silhouette", "ari_stability_1se", "baseline_oracle")
        )
        first_per_label = out.groupby("axis_label", sort=False)["metric"].first()
        assert (first_per_label == "baseline_oracle").all()

    def test_ari_comes_from_oracle_ari_not_ari_at_k(self):
        # _frame's oracle_ari is a constant 0.9; ari_at_k is not, so a
        # baseline row that accidentally summarized ari_at_k would not
        # come out at exactly 0.9.
        out = summarize(_frame(), metrics=("baseline_oracle",))
        assert all(v == pytest.approx(0.9) for v in out["ari_mean"])

    def test_k_recovery_is_undefined_not_zero_or_one(self):
        out = summarize(_frame(), metrics=("baseline_oracle",))
        assert out["k_recovery"].isna().all()

    def test_deduplicated_by_seed_not_counted_once_per_metric_and_k(self):
        # _frame carries two metrics x three k's per (axis_label, seed) --
        # six rows -- over two distinct seeds. A baseline computed without
        # deduplicating by seed would count all six as separate datasets
        # instead of two.
        out = summarize(_frame(), metrics=("baseline_oracle",))
        assert (out["n_datasets"] == 2).all()


class TestTexEscape:
    def test_a_lone_backslash_is_escaped_without_reescaping_its_own_braces(self):
        # A sequential str.replace implementation would emit the braces in
        # \textbackslash{} and then re-escape them via the later { and }
        # rules, corrupting the output to \textbackslash\{\}.
        assert _tex_escape("\\") == r"\textbackslash{}"

    def test_a_backslash_adjacent_to_other_specials_is_escaped_once_each(self):
        assert _tex_escape(r"\alpha_1 100%") == r"\textbackslash{}alpha\_1 100\%"

    def test_percent_in_a_caption_is_still_escaped(self):
        assert _tex_escape("100% of runs") == r"100\% of runs"


class TestRenderGroupedTex:
    def test_produces_a_tabular_environment(self):
        tex = render_grouped_tex(summarize(_frame()), caption="Demo", label="tab:demo")
        assert "\\begin{table}" in tex
        assert "\\end{table}" in tex
        assert "tab:demo" in tex

    def test_includes_every_axis_label_as_a_column_group(self):
        tex = render_grouped_tex(summarize(_frame()), caption="Demo", label="tab:demo")
        assert "easy" in tex
        assert "hard" in tex

    def test_escapes_latex_special_characters_in_the_caption(self):
        tex = render_grouped_tex(
            summarize(_frame()), caption="100% of runs", label="tab:demo"
        )
        assert "100\\%" in tex

    def test_axis_columns_follow_axis_value_order_not_row_order(self):
        tex = render_grouped_tex(
            summarize(_scaling_frame()), caption="Demo", label="tab:demo"
        )
        assert tex.index("start") < tex.index("middle") < tex.index("end")

    def test_baseline_row_renders_a_blank_k_recovery_cell_not_nan(self):
        tex = render_grouped_tex(
            summarize(_frame(), metrics=("baseline_oracle",)),
            caption="Demo",
            label="tab:demo",
        )
        assert "nan" not in tex.lower()
        baseline_line = next(
            line for line in tex.splitlines() if line.startswith("Baseline")
        )
        assert baseline_line.rstrip().endswith(r"&  \\")


class TestWriteTables:
    def test_writes_a_tex_fragment_under_the_manuscript_name(self, tmp_path):
        path = write_tables(
            _frame(), tmp_path, "S2_table", caption="Demo", label="tab:s2"
        )
        assert path.name == "S2_table.tex"
        assert "\\begin{table}" in path.read_text()
