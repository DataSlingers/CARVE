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
        row = out[
            (out["axis_label"] == "easy") & (out["metric"] == "silhouette")
        ].iloc[0]
        # selected k is 4, ari_at_k 0.9, oracle 0.9
        assert row["delta_mean"] == pytest.approx(0.0)

    def test_k_bias_is_selected_k_minus_k_star(self):
        out = summarize(_frame())
        row = out[out["metric"] == "silhouette"].iloc[0]
        assert row["k_bias_median"] == pytest.approx(-1.0)

    def test_over_and_under_rates_sum_with_recovery_to_one(self):
        out = summarize(_frame())
        for _, row in out.iterrows():
            assert row["p_under"] + row["p_over"] + row["k_recovery"] == pytest.approx(1.0)

    def test_restricting_metrics_filters_the_output(self):
        out = summarize(_frame(), metrics=("silhouette",))
        assert set(out["metric"]) == {"silhouette"}

    def test_raises_when_no_requested_metric_is_present(self):
        with pytest.raises(ValueError, match="none of the requested metrics"):
            summarize(_frame(), metrics=("nonexistent",))


class TestTexEscape:
    def test_a_lone_backslash_is_escaped_without_reescaping_its_own_braces(self):
        # A sequential str.replace implementation would emit the braces in
        # \textbackslash{} and then re-escape them via the later { and }
        # rules, corrupting the output to \textbackslash\{\}.
        assert _tex_escape("\\") == r"\textbackslash{}"

    def test_a_backslash_adjacent_to_other_specials_is_escaped_once_each(self):
        assert (
            _tex_escape(r"\alpha_1 100%") == r"\textbackslash{}alpha\_1 100\%"
        )

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


class TestWriteTables:
    def test_writes_a_tex_fragment_under_the_manuscript_name(self, tmp_path):
        path = write_tables(
            _frame(), tmp_path, "S2_table", caption="Demo", label="tab:s2"
        )
        assert path.name == "S2_table.tex"
        assert "\\begin{table}" in path.read_text()
