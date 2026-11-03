"""Tests for manuscript table generation."""

import json

import pandas as pd
import pytest

from benchmarks._artifacts import SCHEMA
from benchmarks._registry import SCALING_TABLE_ROW_GROUPS
from benchmarks._tables import _table_display_name
from benchmarks.run import main
from benchmarks.tables import (
    EXCLUDED_METRICS,
    TABLE_CAPTIONS,
    TABLE_NAMES,
    table_metrics,
    write_all_tables,
)

from ._helpers import _fake_run, synthetic_run_frame

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
                        "selects_true_k": k == 5, "ari_at_k": 0.9,
                        "consensus_ari_at_k": float("nan"), "oracle_ari": 0.95,
                    })
    return pd.DataFrame(rows)[list(SCHEMA)]


def _write_run_dir(root, scenario, cfg_hash, frame):
    """Write a cell__*.parquet checkpoint plus a manifest.json.

    read_run only needs the checkpoint, but the --tables CLI locates a run
    directory through widest_run, which reads the manifest and requires
    status "complete". n_seeds and n_resamples are fixed placeholders;
    widest_run's width selection is tested in test_artifacts.py.
    """
    rd = root / scenario / cfg_hash
    rd.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(rd / "cell__easy__0000.parquet", index=False)
    (rd / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": cfg_hash,
                "status": "complete",
                "scenario": scenario,
                "n_seeds": 3,
                "n_resamples": 20,
            }
        )
    )
    return rd


class TestTableNames:
    def test_every_simulated_scenario_has_a_manuscript_table_name(self):
        from benchmarks._registry import SCENARIOS

        assert set(TABLE_NAMES) == set(SCENARIOS)

    def test_every_table_has_a_caption(self):
        assert set(TABLE_CAPTIONS) == set(TABLE_NAMES)

    def test_names_are_unique(self):
        assert len(set(TABLE_NAMES.values())) == len(TABLE_NAMES)


def test_dimensionality_caption_names_the_feature_dimension():
    """The axis is p, the ambient feature dimension, not embed_dim, which
    this scenario holds at 64.
    """
    from benchmarks.tables import TABLE_CAPTIONS

    caption = TABLE_CAPTIONS["gaussians_dimensionality"]
    assert "feature dimension" in caption
    assert "embedding dimension" not in caption


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


def _left_body(text):
    """The left sub-table's rows and rules, between its header and its
    closing rule: a row as its label, a rule as the \\midrule token."""
    left = text.split(r"\quad")[0]
    body = left.split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
    return [
        line if line == r"\midrule" else line.split(" & ")[0]
        for line in body.strip().splitlines()
    ]


class TestScalingTables:
    """The committed S11 and S12 head their columns with the swept values and
    order the classical indices differently from S2. A fragment that
    differs in either needs hand-editing before it can replace them."""

    @pytest.fixture(
        params=[
            ("gaussians_samples", r"Metric & 1000 & 5500 & 10000 \\"),
            ("gaussians_dimensionality", r"Metric & 50 & 525 & 1000 \\"),
        ],
        ids=["S11", "S12"],
    )
    def fragment(self, request, tmp_path):
        name, header = request.param
        frame = synthetic_run_frame(name, n_seeds=2, metrics=table_metrics())
        [path] = write_all_tables({name: frame}, tmp_path)
        return path.read_text(), header

    def test_the_header_carries_the_axis_values(self, fragment):
        text, header = fragment
        headers = [line for line in text.splitlines() if line.startswith("Metric & ")]
        assert headers == [header] * 2

    def test_rows_follow_the_declared_scaling_groups(self, fragment):
        text, _ = fragment
        expected = []
        for index, group in enumerate(SCALING_TABLE_ROW_GROUPS):
            if index:
                expected.append(r"\midrule")
            expected.extend(_table_display_name(m) for m in group)
        assert _left_body(text) == expected

    def test_a_difficulty_table_keeps_its_labels(self, tmp_path):
        frame = synthetic_run_frame("gaussians", n_seeds=2, metrics=table_metrics())
        [path] = write_all_tables({"gaussians": frame}, tmp_path)
        headers = [
            line for line in path.read_text().splitlines() if line.startswith("Metric & ")
        ]
        assert headers == [r"Metric & easy & medium & hard \\"] * 2


class TestTableMetrics:
    def test_baseline_oracle_leads_the_tuple(self):
        assert table_metrics()[0] == "baseline_oracle"


def test_table_metrics_follows_the_declared_groups():
    from benchmarks._registry import TABLE_ROW_GROUPS
    from benchmarks.tables import table_metrics

    assert table_metrics() == tuple(m for group in TABLE_ROW_GROUPS for m in group)


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
    """--tables, the path a user takes to these tables, with its
    run-directory discovery.
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

    def test_tables_reads_the_widest_run_not_the_lexically_last(self, tmp_path, monkeypatch):
        """The CLI and the notebook must agree about which run they read."""
        root = tmp_path / "runs"
        _fake_run(root, "gaussians", "zzz", n_seeds=5, n_resamples=20)
        _fake_run(root, "gaussians", "aaa", n_seeds=20, n_resamples=100)

        seen = {}

        def _fake_read_run(rd):
            seen["path"] = rd
            return pd.DataFrame(columns=list(SCHEMA))

        monkeypatch.setattr("benchmarks._artifacts.read_run", _fake_read_run)
        main(["--tables", str(tmp_path / "out"), "--root", str(root)])
        assert seen["path"].name == "aaa"


class TestManuscriptTables:
    def test_one_table_per_scenario_in_the_order_of_their_numbers(self):
        from benchmarks.tables import manuscript_tables

        tables = manuscript_tables({"moons": _frame("moons"), "gaussians": _frame("gaussians")})
        assert [t.caption.split(".")[0] for t in tables] == ["S2 Table", "S6 Table"]

    def test_the_caption_states_k_star(self):
        from benchmarks.tables import manuscript_tables

        (table,) = manuscript_tables({"gaussians": _frame("gaussians")})
        assert table.caption == f"S2 Table. {TABLE_CAPTIONS['gaussians']} k* = 5."

    def test_the_cells_are_those_of_the_written_fragment(self, tmp_path):
        from benchmarks.tables import manuscript_tables

        from ._helpers import tex_lines

        frames = {"gaussians": _frame("gaussians")}
        (path,) = write_all_tables(frames, tmp_path)
        (table,) = manuscript_tables(frames)
        tex = path.read_text().splitlines()
        for block in ("Mean ARI", "k-recovery"):
            columns = [c for c in table.cells.columns if c[0] == block]
            assert all(line in tex for line in tex_lines(table, columns)), block

    def test_excluded_metrics_have_no_row(self):
        from benchmarks.tables import manuscript_tables

        (table,) = manuscript_tables({"gaussians": _frame("gaussians")})
        assert "ARI (avg, 1SE)" not in table.cells.index
        assert "Baseline (Oracle)" in table.cells.index

    def test_skips_a_scenario_with_no_rows(self):
        from benchmarks.tables import manuscript_tables

        assert manuscript_tables({"gaussians": pd.DataFrame(columns=list(SCHEMA))}) == []

    def test_a_scaling_table_is_headed_with_its_swept_values(self):
        from benchmarks.tables import manuscript_tables

        frame = synthetic_run_frame("gaussians_samples")
        (table,) = manuscript_tables({"gaussians_samples": frame})
        values = sorted(frame["axis_value"].unique())
        assert [c[1] for c in table.cells.columns][: len(values)] == [str(int(v)) for v in values]


def _runtimes(scenario, axis_name, values, stability, generalizability):
    """Three seeds per axis value: base + seed for stability, base + 2 * seed
    for generalizability, so the sample SDs are 1 and 2."""
    rows = []
    for value, stab, gen in zip(values, stability, generalizability):
        for seed in range(3):
            rows.append(
                {
                    "scenario": scenario,
                    "axis_name": axis_name,
                    "axis_value": value,
                    "seed": seed,
                    "t_stability_s": stab + seed,
                    "t_generalizability_s": gen + 2 * seed,
                }
            )
    return pd.DataFrame(rows)


class TestRuntimeSecondsTable:
    @pytest.fixture
    def runtimes(self):
        return {
            "gaussians_dimensionality": _runtimes(
                "gaussians_dimensionality", "p", (50, 525, 1000), (10, 20, 30), (5, 15, 25)
            ),
            "gaussians_samples": _runtimes(
                "gaussians_samples", "n_total", (1000, 5500, 10000), (1, 2, 3), (4, 5, 6)
            ),
        }

    def test_cells_are_the_mean_and_sample_sd_over_data_sets(self, runtimes):
        from benchmarks.tables import runtime_seconds_table

        table = runtime_seconds_table(runtimes)
        samples = table.cells.xs("CARVE stability", level=1).iloc[0]
        assert list(samples) == ["2.00 (1.00)", "3.00 (1.00)", "4.00 (1.00)"]
        generalizability = table.cells.xs("CARVE generalizability", level=1).iloc[1]
        assert list(generalizability) == ["7.00 (2.00)", "17.00 (2.00)", "27.00 (2.00)"]

    def test_settings_follow_their_table_numbers(self, runtimes):
        from benchmarks.tables import runtime_seconds_table

        settings = list(dict.fromkeys(runtime_seconds_table(runtimes).cells.index.get_level_values(0)))
        assert settings == [
            "Gaussian mixtures over sample size (n = 1000 / 5500 / 10000)",
            "Gaussian mixtures over feature dimension (p = 50 / 525 / 1000)",
        ]

    def test_columns_run_from_the_lowest_axis_value_to_the_highest(self, runtimes):
        from benchmarks.tables import runtime_seconds_table

        assert list(runtime_seconds_table(runtimes).cells.columns) == ["low", "mid", "high"]

    def test_the_caption_is_s13(self, runtimes):
        from benchmarks.tables import runtime_seconds_table

        assert runtime_seconds_table(runtimes).caption.startswith("S13 Table. ")
