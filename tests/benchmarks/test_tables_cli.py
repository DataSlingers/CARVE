"""Tests for manuscript table generation."""

import pandas as pd
import pytest

from benchmarks._artifacts import SCHEMA
from benchmarks.run import main
from benchmarks.tables import (
    EXCLUDED_METRICS,
    TABLE_CAPTIONS,
    TABLE_NAMES,
    table_metrics,
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


def _write_run_dir(root, scenario, cfg_hash, frame):
    """Write the one artifact read_run needs: a cell__*.parquet checkpoint.

    read_run globs cell__*.parquet and concatenates them -- it does not
    require a manifest.json (only promote does) -- so this is enough to
    drive the --tables CLI path end to end without running a real scenario.
    """
    rd = root / scenario / cfg_hash
    rd.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(rd / "cell__easy__0000.parquet", index=False)
    return rd


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

    def test_baseline_oracle_appears_in_the_generated_fragment(self, tmp_path):
        paths = write_all_tables({"gaussians": _frame("gaussians")}, tmp_path)
        assert "Baseline (Oracle)" in paths[0].read_text()

    def test_baseline_oracle_is_the_first_data_row(self, tmp_path):
        paths = write_all_tables({"gaussians": _frame("gaussians")}, tmp_path)
        text = paths[0].read_text()
        assert text.index("Baseline (Oracle)") < text.index("CARVE Stability (1SE)")


class TestTableMetrics:
    def test_baseline_oracle_leads_the_tuple(self):
        assert table_metrics()[0] == "baseline_oracle"


class TestExcludedMetrics:
    def test_matches_the_published_exclusion_list(self):
        assert EXCLUDED_METRICS == frozenset({
            "ari_average",
            "ari_average_1se",
            "ari_average_quant",
            "consensus_pac_stability",
            "consensus_ce_stability",
        })


class TestTablesCli:
    """--tables is the only path a user takes to these tables, so it needs
    its own coverage rather than relying on write_all_tables being called
    correctly by main -- that wiring, and the run-directory discovery it
    does via read_run, is untested by TestWriteAllTables above.
    """

    def test_writes_fragments_from_a_run_directory(self, tmp_path):
        root = tmp_path / "runs"
        out = tmp_path / "tables"
        _write_run_dir(root, "gaussians", "aaaaaaaa", _frame("gaussians"))

        code = main(["--tables", str(out), "--root", str(root)])

        assert code == 0
        fragment = out / f"{TABLE_NAMES['gaussians']}.tex"
        assert fragment.exists()
        assert "\\begin{tabular}" in fragment.read_text()

    def test_warns_when_a_scenario_has_multiple_run_directories(self, tmp_path):
        root = tmp_path / "runs"
        _write_run_dir(root, "gaussians", "aaaaaaaa", _frame("gaussians"))
        _write_run_dir(root, "gaussians", "bbbbbbbb", _frame("gaussians"))

        with pytest.warns(UserWarning, match="gaussians.*2 run directories"):
            main(["--tables", str(tmp_path / "tables"), "--root", str(root)])

    def test_does_not_warn_with_a_single_run_directory(self, tmp_path, recwarn):
        root = tmp_path / "runs"
        _write_run_dir(root, "gaussians", "aaaaaaaa", _frame("gaussians"))

        main(["--tables", str(tmp_path / "tables"), "--root", str(root)])

        messages = [str(w.message) for w in recwarn.list]
        assert not any("run directories found" in m for m in messages)

    def test_uses_the_lexicographically_last_run_directory(self, tmp_path):
        # The chosen frame's k_star is written into the caption, so two runs
        # with different k_star reveal which one was read.
        root = tmp_path / "runs"
        _write_run_dir(root, "gaussians", "aaaaaaaa", _frame("gaussians").assign(k_star=5))
        _write_run_dir(root, "gaussians", "bbbbbbbb", _frame("gaussians").assign(k_star=7))
        out = tmp_path / "tables"

        with pytest.warns(UserWarning, match="using .*bbbbbbbb"):
            main(["--tables", str(out), "--root", str(root)])

        fragment = (out / f"{TABLE_NAMES['gaussians']}.tex").read_text()
        assert "k^\\star = 7" in fragment
        assert "k^\\star = 5" not in fragment
