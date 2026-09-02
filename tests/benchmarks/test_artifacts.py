"""Tests for artifact writing, content addressing, and promotion."""

import json

import pandas as pd
import pytest

from benchmarks._artifacts import (
    SCHEMA,
    build_manifest,
    completed_cells,
    config_hash,
    peak_rss_bytes,
    promote,
    read_run,
    run_dir,
    write_checkpoint,
    write_manifest,
)
from benchmarks._registry import SCENARIOS


def _row(**overrides):
    row = {
        "run_id": "r1",
        "scenario": "demo",
        "axis_name": "difficulty_level",
        "axis_value": 0,
        "axis_label": "easy",
        "seed": 0,
        "k_star": 5,
        "estimator": "kmeans",
        "metric_name": "silhouette",
        "k": 3,
        "metric_value": 0.5,
        "is_selected": False,
        "selects_true_k": False,
        "ari_at_k": 0.4,
        "oracle_ari": 0.9,
    }
    row.update(overrides)
    return row


class TestSchema:
    def test_column_order_is_the_documented_contract(self):
        assert SCHEMA == (
            "run_id",
            "scenario",
            "axis_name",
            "axis_value",
            "axis_label",
            "seed",
            "k_star",
            "estimator",
            "metric_name",
            "k",
            "metric_value",
            "is_selected",
            "selects_true_k",
            "ari_at_k",
            "oracle_ari",
        )


class TestConfigHash:
    def test_is_stable_across_calls(self):
        s = SCENARIOS["gaussians"]
        assert config_hash(s, n_seeds=20, n_resamples=100, random_state=42) == config_hash(
            s, n_seeds=20, n_resamples=100, random_state=42
        )

    def test_changes_when_an_anchor_changes(self):
        s = SCENARIOS["gaussians"]
        anchors = {k: dict(v) for k, v in s.anchors.items()}
        anchors["easy"] = {**anchors["easy"], "corr_strength": 0.999}
        changed = type(s)(
            name=s.name,
            axis=s.axis,
            anchors=anchors,
            shared=s.shared,
            estimator=s.estimator,
        )
        assert config_hash(s, n_seeds=20, n_resamples=100, random_state=42) != config_hash(
            changed, n_seeds=20, n_resamples=100, random_state=42
        )

    def test_changes_when_n_resamples_changes(self):
        s = SCENARIOS["gaussians"]
        assert config_hash(s, n_seeds=20, n_resamples=100, random_state=42) != config_hash(
            s, n_seeds=20, n_resamples=50, random_state=42
        )

    def test_changes_when_n_trees_changes(self):
        s = SCENARIOS["gaussians"]
        changed = type(s)(
            name=s.name,
            axis=s.axis,
            anchors=s.anchors,
            shared=s.shared,
            estimator=s.estimator,
            k_star=s.k_star,
            candidate_k=s.candidate_k,
            n_seeds=s.n_seeds,
            n_trees=s.n_trees + 1,
        )
        assert config_hash(s, n_seeds=20, n_resamples=100, random_state=42) != config_hash(
            changed, n_seeds=20, n_resamples=100, random_state=42
        )

    def test_changes_when_random_state_changes(self):
        s = SCENARIOS["gaussians"]
        assert config_hash(s, n_seeds=20, n_resamples=100, random_state=42) != config_hash(
            s, n_seeds=20, n_resamples=100, random_state=0
        )

    def test_is_twelve_hex_characters(self):
        h = config_hash(SCENARIOS["gaussians"], n_seeds=20, n_resamples=100, random_state=42)
        assert len(h) == 12
        assert set(h) <= set("0123456789abcdef")


class TestCheckpoints:
    def test_round_trips_rows_through_parquet(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row(), _row(k=4)])
        df = read_run(rd)
        assert len(df) == 2
        assert list(df.columns) == list(SCHEMA)

    def test_reports_completed_cells_for_resume(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row()])
        write_checkpoint(rd, "hard", 3, [_row(axis_label="hard", seed=3)])
        assert completed_cells(rd) == {("easy", 0), ("hard", 3)}

    def test_an_empty_run_has_no_completed_cells(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        assert completed_cells(rd) == set()

    def test_read_run_concatenates_every_checkpoint(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        for seed in range(3):
            write_checkpoint(rd, "easy", seed, [_row(seed=seed)])
        assert sorted(read_run(rd)["seed"].tolist()) == [0, 1, 2]

    def test_rejects_rows_that_do_not_match_the_schema(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        bad = _row()
        del bad["oracle_ari"]
        with pytest.raises(ValueError, match="oracle_ari"):
            write_checkpoint(rd, "easy", 0, [bad])

    def test_an_axis_label_with_a_double_underscore_round_trips(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        label = "study__pbmc3k"
        write_checkpoint(rd, label, 7, [_row(axis_label=label, seed=7)])
        assert completed_cells(rd) == {(label, 7)}


class TestManifest:
    def test_records_provenance_and_the_active_anchor_set(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        manifest = build_manifest(
            SCENARIOS["gaussians"],
            run_id="r1",
            config_hash="abc123def456",
            n_seeds=2,
            n_resamples=3,
            n_jobs=1,
            random_state=0,
            wall_clock_s=1.0,
        )
        path = write_manifest(rd, manifest)
        payload = json.loads(path.read_text())
        assert payload["anchor_set"] == "PUBLISHED_ANCHORS"
        assert payload["peak_rss_unit"] == "bytes"
        assert payload["config"]["k_star"] == 5
        assert "numpy" in payload["package_versions"]

    def test_peak_rss_is_a_positive_byte_count(self):
        assert peak_rss_bytes() > 1_000_000

    def test_write_leaves_no_temporary_file_and_the_manifest_parses(self, tmp_path):
        """write_manifest writes through a temp file and os.replace. On
        success that temp file must not survive -- a leftover .tmp file
        would mean the atomic-rename path isn't actually being taken -- and
        the file left at manifest.json must be the real, complete payload.
        """
        rd = run_dir(tmp_path, "demo", "abc123def456")
        manifest = build_manifest(
            SCENARIOS["gaussians"],
            run_id="r1",
            config_hash="abc123def456",
            n_seeds=2,
            n_resamples=3,
            n_jobs=1,
            random_state=0,
            wall_clock_s=1.0,
        )
        path = write_manifest(rd, manifest)

        leftovers = [p for p in rd.iterdir() if p.name != "manifest.json"]
        assert leftovers == []

        payload = json.loads(path.read_text())
        assert payload["run_id"] == "r1"


class TestPromote:
    def test_writes_csv_and_manifest_to_the_published_root(self, tmp_path):
        rd = run_dir(tmp_path / "runs", "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row()])
        write_manifest(
            rd,
            build_manifest(
                SCENARIOS["gaussians"],
                run_id="r1",
                config_hash="abc123def456",
                n_seeds=1,
                n_resamples=1,
                n_jobs=1,
                random_state=0,
                wall_clock_s=0.1,
            ),
        )
        out = promote(rd, tmp_path / "published")
        assert (out / "results.csv").exists()
        assert (out / "manifest.json").exists()

    def test_promoted_csv_preserves_the_schema(self, tmp_path):
        rd = run_dir(tmp_path / "runs", "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row(), _row(k=4, is_selected=True)])
        write_manifest(
            rd,
            build_manifest(
                SCENARIOS["gaussians"],
                run_id="r1",
                config_hash="abc123def456",
                n_seeds=1,
                n_resamples=1,
                n_jobs=1,
                random_state=0,
                wall_clock_s=0.1,
            ),
        )
        out = promote(rd, tmp_path / "published")
        df = pd.read_csv(out / "results.csv")
        assert list(df.columns) == list(SCHEMA)
        assert df["is_selected"].sum() == 1

    def test_refuses_to_promote_a_run_with_no_manifest(self, tmp_path):
        rd = run_dir(tmp_path / "runs", "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row()])
        with pytest.raises(FileNotFoundError, match="manifest"):
            promote(rd, tmp_path / "published")


from benchmarks._artifacts import (
    RUNTIME_SCHEMA,
    read_runtimes,
    write_runtime_checkpoint,
)


def _runtime_row(**overrides):
    row = {
        "run_id": "r1",
        "scenario": "demo",
        "axis_name": "difficulty_level",
        "axis_value": 0,
        "axis_label": "easy",
        "seed": 0,
        "n_samples": 120,
        "n_features": 4,
        "n_resamples": 3,
        "n_jobs": 1,
        "estimator": "kmeans",
        "t_default_s": 1.25,
        "t_stability_s": 0.80,
        "t_generalizability_s": 0.95,
        "t_per_k_stability_s": 0.16,
        "t_per_k_generalizability_s": 0.19,
    }
    row.update(overrides)
    return row


class TestRuntimeSidecar:
    def test_schema_is_the_documented_contract(self):
        assert RUNTIME_SCHEMA == (
            "run_id",
            "scenario",
            "axis_name",
            "axis_value",
            "axis_label",
            "seed",
            "n_samples",
            "n_features",
            "n_resamples",
            "n_jobs",
            "estimator",
            "t_default_s",
            "t_stability_s",
            "t_generalizability_s",
            "t_per_k_stability_s",
            "t_per_k_generalizability_s",
        )

    def test_round_trips_through_parquet(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        write_runtime_checkpoint(rd, "easy", 0, [_runtime_row()])
        df = read_runtimes(rd)
        assert len(df) == 1
        assert list(df.columns) == list(RUNTIME_SCHEMA)

    def test_runtime_files_do_not_pollute_the_metric_run(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row()])
        write_runtime_checkpoint(rd, "easy", 0, [_runtime_row()])
        assert len(read_run(rd)) == 1
        assert len(read_runtimes(rd)) == 1

    def test_an_empty_run_returns_an_empty_runtime_frame(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        assert read_runtimes(rd).empty

    def test_rejects_rows_outside_the_runtime_schema(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        bad = _runtime_row()
        del bad["t_stability_s"]
        with pytest.raises(ValueError, match="t_stability_s"):
            write_runtime_checkpoint(rd, "easy", 0, [bad])
