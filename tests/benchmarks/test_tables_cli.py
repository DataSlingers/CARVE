"""Tests for manuscript table generation."""

import pandas as pd

from benchmarks._artifacts import SCHEMA
from benchmarks.tables import (
    EXCLUDED_METRICS,
    TABLE_CAPTIONS,
    TABLE_NAMES,
    write_all_tables,
)

METRICS = ("ari_stability_1se", "ari_average_1se", "silhouette")


def _frame(scenario):
    rows = []
    for axis_value, axis_label in enumerate(("easy", "medium", "hard")):
        for seed in range(3):
            for metric in METRICS:
                for k in (4, 5):
                    rows.append({
                        "run_id": "r1", "scenario": scenario,
                        "axis_name": "difficulty_level", "axis_value": axis_value,
                        "axis_label": axis_label, "seed": seed, "k_star": 5,
                        "estimator": "kmeans", "metric_name": metric, "k": k,
                        "metric_value": 0.1 * k, "is_selected": k == 5,
                        "selects_true_k": k == 5, "ari_at_k": 0.9, "oracle_ari": 0.95,
                    })
    return pd.DataFrame(rows)[list(SCHEMA)]


class TestTableNames:
    def test_every_simulated_scenario_has_a_manuscript_table_name(self):
        from benchmarks._registry import SCENARIOS

        assert set(TABLE_NAMES) == set(SCENARIOS)

    def test_every_table_has_a_caption(self):
        assert set(TABLE_CAPTIONS) == set(TABLE_NAMES)

    def test_names_are_unique(self):
        assert len(set(TABLE_NAMES.values())) == len(TABLE_NAMES)


class TestWriteAllTables:
    def test_writes_one_fragment_per_scenario(self, tmp_path):
        frames = {"gaussians": _frame("gaussians"), "moons": _frame("moons")}
        paths = write_all_tables(frames, tmp_path)
        assert len(paths) == 2
        assert all(p.suffix == ".tex" and p.exists() for p in paths)

    def test_uses_the_manuscript_table_name(self, tmp_path):
        paths = write_all_tables({"gaussians": _frame("gaussians")}, tmp_path)
        assert paths[0].stem == TABLE_NAMES["gaussians"]

    def test_fragments_contain_a_tabular_environment(self, tmp_path):
        paths = write_all_tables({"gaussians": _frame("gaussians")}, tmp_path)
        assert "\\begin{tabular}" in paths[0].read_text()

    def test_excluded_metrics_do_not_appear(self, tmp_path):
        paths = write_all_tables({"gaussians": _frame("gaussians")}, tmp_path)
        text = paths[0].read_text()
        assert "ari_average_1se" not in text
        assert "ARI (avg, 1SE)" not in text

    def test_the_k_star_header_is_generated_not_stripped(self, tmp_path):
        paths = write_all_tables({"gaussians": _frame("gaussians")}, tmp_path)
        assert "k^\\star" in paths[0].read_text()

    def test_skips_a_scenario_with_no_rows(self, tmp_path):
        empty = pd.DataFrame(columns=list(SCHEMA))
        paths = write_all_tables({"gaussians": empty}, tmp_path)
        assert paths == []


class TestExcludedMetrics:
    def test_matches_the_published_exclusion_list(self):
        assert EXCLUDED_METRICS == frozenset({
            "ari_average",
            "ari_average_1se",
            "ari_average_quant",
            "consensus_pac_stability",
            "consensus_ce_stability",
        })
