"""Tests for the unified summarizer."""

import numpy as np
import pandas as pd
import pytest

from benchmarks._artifacts import SCHEMA
from benchmarks._registry import METRIC_DISPLAY_NAMES, TABLE_ROW_GROUPS
from benchmarks._tables import (
    _table_display_name,
    _tex_escape,
    render_ablation_tex,
    render_grouped_tex,
    render_paired_tex,
    summarize,
    summary_stats,
    wilson_ci,
    write_ablation_table,
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


@pytest.fixture
def paired_summary():
    """A summary with a known best and second-best per column.

    Carries one metric per TABLE_ROW_GROUPS group, including group 4
    (ari_stability_quant), so all four groups render and the three internal
    rules are exercised. ari_stability_quant's values (0.700, 0.30) sit
    below both the best and the second-best in each column, so the other
    assertions in this class are unaffected by its presence.
    """
    rows = []
    values = {
        "baseline_oracle": (0.914, float("nan")),
        "ari_stability_1se": (0.932, 1.000),
        "ari_generalizability_1se": (0.868, 0.550),
        "davies_bouldin": (0.928, 0.950),
        "silhouette": (0.900, 0.700),
        "gap": (0.851, 0.650),
        "calinski_harabasz": (0.640, 0.000),
        "ari_stability_quant": (0.700, 0.30),
    }
    for metric, (ari, recovery) in values.items():
        rows.append(
            {
                "axis_label": "easy",
                "metric": metric,
                "display_name": METRIC_DISPLAY_NAMES[metric],
                "n_datasets": 20,
                "ari_mean": ari,
                "ari_sd": 0.05,
                "k_recovery": recovery,
            }
        )
    return pd.DataFrame(rows)


class TestPairedTex:
    def test_two_tabulars_inside_one_resizebox(self, paired_summary):
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert out.count(r"\begin{tabular}") == 2
        assert r"\resizebox{\textwidth}{!}{" in out
        assert r"\quad" in out

    def test_booktabs_rules(self, paired_summary):
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert r"\toprule" in out
        assert r"\bottomrule" in out
        assert r"\hline" not in out

    def test_one_midrule_between_each_pair_of_row_groups(self, paired_summary):
        """Four groups means three internal rules, plus the header's, per
        sub-table."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert out.count(r"\midrule") == 2 * (1 + 3)

    def test_the_best_in_a_column_is_bold(self, paired_summary):
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert r"\textbf{0.932}" in out

    def test_the_second_best_is_underlined(self, paired_summary):
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert r"\underline{0.928}" in out

    def test_the_oracle_is_never_ranked(self, paired_summary):
        """It is the reference the others are measured against, not a
        competitor. The committed tables leave it unmarked."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert r"\textbf{0.914}" not in out
        assert r"\underline{0.914}" not in out

    def test_ties_for_best_are_all_bold_with_no_underline(self):
        """Matching the committed S2 table: three cells at 0.932 are all
        bold and nothing is underlined in that column."""
        rows = [
            {
                "axis_label": "easy",
                "metric": m,
                "display_name": METRIC_DISPLAY_NAMES[m],
                "n_datasets": 20,
                "ari_mean": v,
                "ari_sd": 0.05,
                "k_recovery": 0.5,
            }
            for m, v in (
                ("ari_stability_1se", 0.932),
                ("ari_stability_quant", 0.932),
                ("ari_stability", 0.932),
                ("silhouette", 0.800),
            )
        ]
        out = render_paired_tex(
            pd.DataFrame(rows), caption="c", label="tab:x"
        )
        assert out.count(r"\textbf{0.932}") == 3
        assert r"\underline{" not in out

    def test_the_oracles_recovery_cells_are_blank(self, paired_summary):
        """The oracle has no selected k, so k-recovery is NaN. The published
        tables print an empty cell, not the literal nan.

        Every cell after the metric name in the right-hand table's oracle
        row must be empty -- not merely free of the literal string "nan",
        which a row with only some cells blank would also satisfy.
        """
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        right = out.split(r"\quad")[1]
        oracle_row = next(
            line for line in right.splitlines() if "Baseline (Oracle)" in line
        )
        body = oracle_row[: -len(r" \\")]
        cells = body.split(" & ")
        assert cells[0] == "Baseline (Oracle)"
        assert all(cell == "" for cell in cells[1:])

    def test_rows_follow_the_declared_groups(self, paired_summary):
        """Positions must be looked up by the name the renderer actually
        writes: davies_bouldin and calinski_harabasz render with an en dash,
        not the plain hyphen METRIC_DISPLAY_NAMES carries, so looking those
        two up by METRIC_DISPLAY_NAMES would silently drop them from the
        check instead of covering all eight rows."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        left = out.split(r"\quad")[0]
        positions = [
            left.index(_table_display_name(m))
            for group in TABLE_ROW_GROUPS
            for m in group
            if _table_display_name(m) in left
        ]
        assert len(positions) == len(paired_summary)
        assert positions == sorted(positions)

    def test_groups_set_the_row_order_and_the_rules(self, paired_summary):
        """A table family whose committed layout orders the rows differently
        passes its own groups; the rules fall between those groups."""
        groups = (
            ("baseline_oracle",),
            ("ari_generalizability_1se", "ari_stability_1se"),
            ("gap", "calinski_harabasz", "silhouette", "davies_bouldin"),
            ("ari_stability_quant",),
        )
        out = render_paired_tex(
            paired_summary, caption="c", label="tab:x", groups=groups
        )
        left = out.split(r"\quad")[0]
        body = left.split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
        tokens = [
            line if line == r"\midrule" else line.split(" & ")[0]
            for line in body.strip().splitlines()
        ]
        expected = []
        for index, group in enumerate(groups):
            if index:
                expected.append(r"\midrule")
            expected.extend(_table_display_name(m) for m in group)
        assert tokens == expected

    def test_column_headers_replace_the_axis_labels(self):
        """A scaling table heads its columns with the swept values, as the
        committed S8 and S9 do, not with start/middle/end."""
        rows = [
            {
                "axis_label": label,
                "metric": "ari_stability_1se",
                "display_name": METRIC_DISPLAY_NAMES["ari_stability_1se"],
                "n_datasets": 20,
                "ari_mean": 0.9,
                "ari_sd": 0.05,
                "k_recovery": 0.5,
            }
            for label in ("start", "middle", "end")
        ]
        out = render_paired_tex(
            pd.DataFrame(rows),
            caption="c",
            label="tab:x",
            column_headers={"start": "1000", "middle": "5500", "end": "10000"},
        )
        headers = [line for line in out.splitlines() if line.startswith("Metric & ")]
        assert headers == [r"Metric & 1000 & 5500 & 10000 \\"] * 2

    def test_the_cvi_row_labels_use_an_en_dash(self, paired_summary):
        """Matches the committed S2 table and the manuscript prose, which
        both write Davies-Bouldin and Calinski-Harabasz with an en dash.
        METRIC_DISPLAY_NAMES keeps the plain hyphen, since figure legends
        read it too; the en dash is table-only."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert "Davies--Bouldin" in out
        assert "Davies-Bouldin" not in out
        assert "Calinski--Harabasz" in out
        assert "Calinski-Harabasz" not in out

    def test_k_recovery_uses_the_same_decimals_as_ari_mean(self, paired_summary):
        """The committed table prints k-recovery to three decimals (1.000,
        0.550, 0.700, ...), not two. ari_stability_1se's k-recovery is the
        unique best (1.0) in its column, so it is bold in the right-hand
        table."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        right = out.split(r"\quad")[1]
        assert r"\textbf{1.000}" in right

    def test_a_metric_missing_from_the_summary_is_skipped(self, paired_summary):
        """A run that produced no rows for one metric must not leave a row
        of the literal nan."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert "nan" not in out

    def test_the_caption_and_label_survive(self, paired_summary):
        """The manuscript never puts \\caption or \\label inside a
        supplementary table's float: each is an author-written
        \\paragraph*{Sn Table.}, \\label{Sn_Table}, and caption prose above a
        caption-free \\begin{table}[H]. The fragment is the float only --
        caption and label are carried as one leading LaTeX comment instead,
        so a generated file can still be identified without adding a second
        label or an in-float caption a dropped-in fragment would otherwise
        carry."""
        out = render_paired_tex(
            paired_summary, caption="Gaussian mixtures.", label="tab:s2"
        )
        assert r"\caption" not in out
        assert r"\label" not in out
        lines = out.splitlines()
        assert lines[0] == "% tab:s2: Gaussian mixtures."

    def test_the_first_non_comment_line_opens_the_table_float(self, paired_summary):
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        first_non_comment = next(
            line for line in out.splitlines() if not line.startswith("%")
        )
        assert first_non_comment == r"\begin{table}[H]"

    def test_a_newline_or_percent_sign_in_the_caption_cannot_escape_the_comment(
        self, paired_summary
    ):
        """A literal newline in the caption must not end the LaTeX comment
        early and leave un-commented text as the file's second line; a
        percent sign must not survive into the comment either, defensively,
        though TABLE_CAPTIONS carries none today."""
        out = render_paired_tex(
            paired_summary,
            caption="Two lines.\nSecond line % with a percent.",
            label="tab:s2",
        )
        lines = out.splitlines()
        assert lines[0].startswith("%")
        assert "%" not in lines[0][1:]
        assert lines[1] == r"\begin{table}[H]"

    def test_the_first_subtable_closes_with_a_percent_the_second_does_not(
        self, paired_summary
    ):
        """Matches the committed table: the trailing "%" after the first
        \\end{tabular} suppresses the space LaTeX would otherwise insert
        before \\quad; the second \\end{tabular}, at the end of the pair, is
        bare."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert "\\end{tabular}%\n\\quad" in out
        assert out.count(r"\end{tabular}%") == 1
        assert "\\end{tabular}\n}\n\\end{table}" in out

    def test_ranking_excludes_metrics_outside_the_declared_groups(self):
        """A metric present in the summary but outside TABLE_ROW_GROUPS
        (ari_average is excluded from the published tables entirely) must
        not enter the ranking for the rows that do render. Ranked over
        everything, ari_average's 0.99 outranked every rendered value, so
        ari_stability_1se's 0.930 -- the best among what actually renders --
        came out underlined instead of bold."""
        rows = [
            {
                "axis_label": "easy",
                "metric": m,
                "display_name": METRIC_DISPLAY_NAMES.get(m, m),
                "n_datasets": 20,
                "ari_mean": v,
                "ari_sd": 0.05,
                "k_recovery": 0.5,
            }
            for m, v in (
                ("baseline_oracle", 0.90),
                ("ari_average", 0.99),
                ("ari_stability_1se", 0.93),
                ("ari_generalizability_1se", 0.85),
            )
        ]
        out = render_paired_tex(pd.DataFrame(rows), caption="c", label="tab:x")
        left = out.split(r"\quad")[0]
        assert r"\textbf{0.930}" in left
        assert r"\underline{0.850}" in left
        assert "0.990" not in left
        assert r"\underline{0.930}" not in left


class TestAblationTable:
    @pytest.fixture(scope="class")
    @classmethod
    def frames(cls, tmp_path_factory):
        from benchmarks._registry import ABLATIONS
        from tests.benchmarks._helpers import small_ablation, synthetic_ablation_frames

        ablation = small_ablation(ABLATIONS["rho_b"], study="klein")
        return ablation, synthetic_ablation_frames(
            ablation, "test", seed=2,
            tmp_path=tmp_path_factory.mktemp("ablation_frames"),
        )

    def test_renders_two_sub_tables_with_every_setting(self, frames):
        ablation, data = frames
        from benchmarks._ablation_cells import arm_view
        from benchmarks._ablation_summary import table_rows

        rows = {
            "rho": table_rows(arm_view(data, ablation=ablation, scale="test", arm="rho"),
                              x="subsample_ratio", study="klein"),
            "b": table_rows(arm_view(data, ablation=ablation, scale="test", arm="b"),
                            x="n_resamples", study="klein"),
        }
        tex = render_ablation_tex(rows, caption="Sensitivity", label="tab:ablation",
                                  study_title="Klein")
        assert tex.count(r"\begin{tabular}") == 2
        for rho in ablation.rho_grid:
            assert f"\n{rho:g} &" in tex
        for b in ablation.b_grid:
            assert f"\n{b} &" in tex
        assert "Ward" in tex or "Spectral" in tex
        assert r"\caption{Sensitivity}" in tex
        assert "nan" not in tex

    def test_write_ablation_table_writes_the_fragment(self, frames, tmp_path):
        ablation, data = frames
        path = write_ablation_table(data, ablation=ablation, scale="test", out_dir=tmp_path)
        assert path == tmp_path / "si_table_ablation.tex"
        assert r"\begin{table}" in path.read_text()

    def test_without_a_case_study_the_modal_column_is_dropped(self, tmp_path):
        from benchmarks._ablation_summary import HEADLINE_METRICS
        from benchmarks._registry import ABLATIONS
        from tests.benchmarks._helpers import small_ablation, synthetic_ablation_frames

        ablation = small_ablation(ABLATIONS["rho_b"])
        data = synthetic_ablation_frames(ablation, "test", seed=4)
        tex = write_ablation_table(
            data, ablation=ablation, scale="test", out_dir=tmp_path
        ).read_text()
        assert "modal" not in tex
        # Per headline selector: recovery and ARI, plus agreement in the B arm.
        n_metrics = len(HEADLINE_METRICS)
        assert f"\\begin{{tabular}}{{l{'c' * 2 * n_metrics}}}" in tex
        assert f"\\begin{{tabular}}{{l{'c' * 3 * n_metrics}}}" in tex
        for rho in ablation.rho_grid:
            assert f"\n{rho:g} &" in tex
        assert "nan" not in tex

    def test_renders_nan_and_missing_cells_empty(self):
        """Verify NaN and missing-row rendering paths are covered.

        Builds rows_by_arm by hand with: one row with NaN recovery/recovery_lo/
        recovery_hi/ari_mean; one row with NaN study_share; one (setting, metric)
        pair absent entirely (match.empty fires); and defined rows with
        study_modal "AgglomerativeClustering, k=4" and "SpectralClustering, k=3".
        """
        # Build rows_by_arm with both headline metrics for each arm
        rho_rows = pd.DataFrame(
            [
                {
                    "setting": 0.5,
                    "metric_name": "ari_stability_1se",
                    "recovery": float("nan"),
                    "recovery_lo": float("nan"),
                    "recovery_hi": float("nan"),
                    "ari_mean": float("nan"),
                    "agreement": 0.75,
                    "study_modal": "AgglomerativeClustering, k=4",
                    "study_share": 0.50,
                },
                {
                    "setting": 0.5,
                    "metric_name": "ari_generalizability_1se",
                    "recovery": 0.33,
                    "recovery_lo": 0.14,
                    "recovery_hi": 0.61,
                    "ari_mean": 0.45,
                    "agreement": 0.60,
                    "study_modal": "SpectralClustering, k=3",
                    "study_share": float("nan"),
                },
                {
                    "setting": 0.7,
                    "metric_name": "ari_stability_1se",
                    "recovery": 0.25,
                    "recovery_lo": 0.09,
                    "recovery_hi": 0.53,
                    "ari_mean": 0.50,
                    "agreement": 0.42,
                    "study_modal": "Ward, k=2",
                    "study_share": 0.75,
                },
                {
                    "setting": 0.9,
                    "metric_name": "ari_stability_1se",
                    "recovery": 0.33,
                    "recovery_lo": 0.14,
                    "recovery_hi": 0.61,
                    "ari_mean": 0.52,
                    "agreement": 0.50,
                    "study_modal": "SpectralClustering, k=3",
                    "study_share": 0.80,
                },
                # 0.7 + ari_generalizability_1se intentionally omitted (match.empty)
            ]
        )

        b_rows = pd.DataFrame(
            [
                {
                    "setting": 50,
                    "metric_name": "ari_stability_1se",
                    "recovery": float("nan"),
                    "recovery_lo": float("nan"),
                    "recovery_hi": float("nan"),
                    "ari_mean": float("nan"),
                    "agreement": 0.33,
                    "study_modal": "AgglomerativeClustering, k=4",
                    "study_share": 0.50,
                },
                {
                    "setting": 50,
                    "metric_name": "ari_generalizability_1se",
                    "recovery": 0.42,
                    "recovery_lo": 0.19,
                    "recovery_hi": 0.68,
                    "ari_mean": 0.55,
                    "agreement": 0.67,
                    "study_modal": "SpectralClustering, k=3",
                    "study_share": float("nan"),
                },
                {
                    "setting": 100,
                    "metric_name": "ari_stability_1se",
                    "recovery": 0.17,
                    "recovery_lo": 0.05,
                    "recovery_hi": 0.45,
                    "ari_mean": 0.48,
                    "agreement": 0.25,
                    "study_modal": "Ward, k=5",
                    "study_share": 0.60,
                },
                {
                    "setting": 150,
                    "metric_name": "ari_stability_1se",
                    "recovery": 0.42,
                    "recovery_lo": 0.19,
                    "recovery_hi": 0.68,
                    "ari_mean": 0.56,
                    "agreement": 0.58,
                    "study_modal": "SpectralClustering, k=3",
                    "study_share": 0.75,
                },
                # 100 + ari_generalizability_1se intentionally omitted (match.empty)
            ]
        )

        rows_by_arm = {"rho": rho_rows, "b": b_rows}
        tex = render_ablation_tex(
            rows_by_arm,
            metrics=("ari_stability_1se", "ari_generalizability_1se"),
            caption="Test",
            label="tab:test",
            study_title="Test",
        )

        # Assert no "nan" literal appears
        assert "nan" not in tex

        # Verify short names render correctly (these have defined study_share)
        assert "Ward, $k=2$" in tex
        assert "Ward, $k=5$" in tex
        assert "Ward, $k=4$" in tex
        assert "Spectral, $k=3$" in tex

        # Extract rho data lines: 0.5, 0.7, 0.9
        rho_section = tex.split(r"\begin{tabular}{lcccccc}")[1].split(r"\end{tabular}")[0]
        rho_data_lines = [
            line
            for line in rho_section.split("\n")
            if line.strip() and "0." in line and not line.startswith("\\")
        ]
        assert len(rho_data_lines) == 3, f"Expected 3 rho data rows, got {len(rho_data_lines)}"

        # Extract B data lines: 50, 100, 150
        b_section = tex.split(r"\begin{tabular}{lcccccccc}")[1].split(r"\end{tabular}")[0]
        b_data_lines = [
            line
            for line in b_section.split("\n")
            if line.strip()
            and (line.startswith("50 ") or line.startswith("100 ") or line.startswith("150 "))
        ]
        assert len(b_data_lines) == 3, f"Expected 3 b data rows, got {len(b_data_lines)}"

        # Verify cell counts: rho sub-table has 7 cells per row (1 setting + 3 * 2 metrics)
        for line in rho_data_lines:
            assert "nan" not in line.lower(), f"Line should not contain nan: {line}"
            # Strip the LaTeX row-end marker (space-backslash-backslash)
            line_content = line[:-3] if line.endswith(r" \\") else line
            cells = line_content.split(" & ")
            assert (
                len(cells) == 7
            ), f"Rho row should have 7 cells, got {len(cells)}: {line}"

        # Verify cell counts: B sub-table has 9 cells per row (1 setting + 4 * 2 metrics)
        for line in b_data_lines:
            assert "nan" not in line.lower(), f"Line should not contain nan: {line}"
            # Strip the LaTeX row-end marker (space-backslash-backslash)
            line_content = line[:-3] if line.endswith(r" \\") else line
            cells = line_content.split(" & ")
            assert (
                len(cells) == 9
            ), f"B row should have 9 cells, got {len(cells)}: {line}"


class TestRenderM3CTex:
    @pytest.fixture
    def ari_df(self):
        return pd.DataFrame(
            [
                {"method": "CARVE", "metric": "ari_generalizability_1se", "ari": 0.81, "k": 4},
                {"method": "Silhouette", "metric": "silhouette", "ari": 0.42, "k": 2},
                {"method": "M3C", "metric": "m3c_rcsi", "ari": 0.41, "k": 2},
            ]
        )

    def test_renders_one_row_per_method(self, ari_df):
        from benchmarks._tables import render_m3c_tex

        tex = render_m3c_tex(ari_df, caption="Klein comparison.", label="tab:klein_m3c")
        for method in ("CARVE", "Silhouette", "M3C"):
            assert method in tex

    def test_carries_the_caption_and_label(self, ari_df):
        from benchmarks._tables import render_m3c_tex

        tex = render_m3c_tex(ari_df, caption="Klein comparison.", label="tab:klein_m3c")
        assert r"\caption{Klein comparison.}" in tex
        assert r"\label{tab:klein_m3c}" in tex

    def test_escapes_latex_specials_in_the_caption(self, ari_df):
        from benchmarks._tables import render_m3c_tex

        tex = render_m3c_tex(ari_df, caption="50% of cells", label="tab:x")
        assert r"50\% of cells" in tex

    def test_rounds_ari_to_the_requested_precision(self, ari_df):
        from benchmarks._tables import render_m3c_tex

        tex = render_m3c_tex(
            ari_df, caption="c", label="l", decimals=2
        )
        assert "0.81" in tex
        assert "0.810" not in tex

    def test_writes_a_tex_file(self, ari_df, tmp_path):
        from benchmarks._tables import write_m3c_table

        path = write_m3c_table(ari_df, out_dir=tmp_path)
        assert path == tmp_path / "si_table_m3c.tex"
        assert r"\begin{table}" in path.read_text()

    def test_writes_a_csv_beside_the_tex(self, ari_df, tmp_path):
        from benchmarks._tables import write_m3c_table

        write_m3c_table(ari_df, out_dir=tmp_path)
        written = pd.read_csv(tmp_path / "si_table_m3c.csv")
        assert written["method"].tolist() == ["CARVE", "Silhouette", "M3C"]
        assert written["ari"].tolist() == [0.81, 0.42, 0.41]
        assert written["k"].tolist() == [4, 2, 2]
        assert written["metric"].tolist() == [
            "ari_generalizability_1se",
            "silhouette",
            "m3c_rcsi",
        ]

    def test_preserves_math_notation_in_method_names(self):
        from benchmarks._tables import render_m3c_tex

        ari_df = pd.DataFrame(
            [
                {
                    "method": "M3C (at $k=4$)",
                    "metric": "m3c_rcsi",
                    "ari": 0.41,
                    "k": 4,
                }
            ]
        )
        tex = render_m3c_tex(ari_df, caption="Test", label="tab:test")
        # The method name with math should appear verbatim, unescaped
        assert "M3C (at $k=4$)" in tex
        # Verify no backslash escaping of the parentheses or dollar signs
        assert r"M3C \(at" not in tex
        assert r"\$k" not in tex
