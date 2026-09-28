"""Tests for artifact writing, content addressing, and promotion."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks._artifacts import (
    ABLATION_SCHEMAS,
    RUNTIME_SCHEMA,
    SCHEMA,
    CELL_KEY,
    ablation_dir,
    build_manifest,
    check_fingerprint,
    completed_cells,
    config_hash,
    fingerprint,
    fingerprint_path,
    peak_rss_bytes,
    promote,
    provenance,
    read_frames,
    read_run,
    read_runtimes,
    run_dir,
    scenario_identity,
    write_checkpoint,
    write_frame,
    widest_run,
    write_manifest,
    write_runtime_checkpoint,
)
from benchmarks._registry import ACTIVE_ANCHOR_SET_NAME, SCENARIOS

from ._helpers import _fake_run


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
        "consensus_ari_at_k": float("nan"),
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
            "consensus_ari_at_k",
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
        changed = dataclasses.replace(s, anchors=anchors)
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

    def test_read_run_rescores_a_checkpoint_from_before_the_consensus_column(
        self, tmp_path
    ):
        """An old checkpoint stored consensus ARIs in ari_at_k, and cut the
        generalizability metrics from the generalizability consensus.

        Every source is distinct per metric and per k, so reading any of
        them from the wrong rows fails.
        """
        rd = run_dir(tmp_path, "demo", "abc123def456")
        rows = []
        for k in (3, 4, 5):
            for metric, ari in (
                ("silhouette", 0.10 * k),
                ("gap", 0.10 * k),
                ("ari_stability", 0.50 + 0.01 * k),
                ("ari_stability_1se", 0.50 + 0.01 * k),
                ("ari_generalizability_1se", 0.20 + 0.01 * k),
            ):
                rows.append(_row(metric_name=metric, k=k, ari_at_k=ari))
        frame = pd.DataFrame(rows).drop(columns="consensus_ari_at_k")
        frame.to_parquet(rd / "cell__easy__0000.parquet", index=False)

        out = read_run(rd)
        assert list(out.columns) == list(SCHEMA)
        for _, row in out.iterrows():
            assert row["ari_at_k"] == pytest.approx(0.10 * row["k"])
            if row["metric_name"] in ("silhouette", "gap"):
                assert np.isnan(row["consensus_ari_at_k"])
            else:
                assert row["consensus_ari_at_k"] == pytest.approx(0.50 + 0.01 * row["k"])

    def test_read_run_leaves_a_current_checkpoint_alone(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        write_checkpoint(
            rd,
            "easy",
            0,
            [_row(metric_name="ari_stability_1se", ari_at_k=0.7, consensus_ari_at_k=0.6)],
        )
        row = read_run(rd).iloc[0]
        assert (row["ari_at_k"], row["consensus_ari_at_k"]) == (0.7, 0.6)

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
        # build_manifest reads whichever anchor set is active, not a fixed
        # name -- this test's intent is that provenance tracks that switch,
        # not that the published set stays active forever.
        assert payload["anchor_set"] == ACTIVE_ANCHOR_SET_NAME
        assert payload["peak_rss_unit"] == "bytes"
        assert payload["config"]["k_star"] == 5
        assert "numpy" in payload["package_versions"]

    def test_peak_rss_is_a_positive_byte_count(self):
        assert peak_rss_bytes() > 1_000_000

    def test_build_manifest_records_a_status(self):
        manifest = build_manifest(
            SCENARIOS["gaussians"],
            run_id="abc123",
            config_hash="def456",
            n_seeds=1,
            n_resamples=2,
            n_jobs=1,
            random_state=42,
            wall_clock_s=1.0,
            status="running",
        )
        assert manifest.status == "running"

    def test_build_manifest_rejects_an_unknown_status(self):
        with pytest.raises(ValueError, match="status"):
            build_manifest(
                SCENARIOS["gaussians"],
                run_id="abc123",
                config_hash="def456",
                n_seeds=1,
                n_resamples=2,
                n_jobs=1,
                random_state=42,
                wall_clock_s=1.0,
                status="halfway",
            )

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


class TestWidestRun:
    def test_picks_the_widest_sweep_not_the_lexically_last(self, tmp_path):
        """The CLI took the lexically last directory and the notebook took
        the widest sweep, so S4 regenerated from the CLI was the 5-dataset
        run while the notebook showed the 20-dataset one."""
        _fake_run(tmp_path, "t_dist_noise", "zzz", n_seeds=5, n_resamples=20)
        wide = _fake_run(tmp_path, "t_dist_noise", "aaa", n_seeds=20, n_resamples=100)
        assert widest_run(tmp_path, "t_dist_noise") == wide

    def test_breaks_a_seed_tie_on_the_resample_count(self, tmp_path):
        _fake_run(tmp_path, "gaussians", "aaa", n_seeds=20, n_resamples=20)
        wide = _fake_run(tmp_path, "gaussians", "bbb", n_seeds=20, n_resamples=100)
        assert widest_run(tmp_path, "gaussians") == wide

    def test_raises_on_a_tie(self, tmp_path):
        _fake_run(tmp_path, "gaussians", "aaa", n_seeds=20, n_resamples=100)
        _fake_run(tmp_path, "gaussians", "bbb", n_seeds=20, n_resamples=100)
        with pytest.raises(RuntimeError, match="equally wide"):
            widest_run(tmp_path, "gaussians")

    def test_skips_an_incomplete_run(self, tmp_path):
        done = _fake_run(tmp_path, "gaussians", "aaa", n_seeds=5, n_resamples=20)
        _fake_run(
            tmp_path, "gaussians", "bbb", n_seeds=20, n_resamples=100, status="running"
        )
        assert widest_run(tmp_path, "gaussians") == done

    def test_can_be_asked_for_an_incomplete_run(self, tmp_path):
        _fake_run(tmp_path, "gaussians", "aaa", n_seeds=5, n_resamples=20)
        running = _fake_run(
            tmp_path, "gaussians", "bbb", n_seeds=20, n_resamples=100, status="running"
        )
        assert widest_run(tmp_path, "gaussians", require_complete=False) == running

    def test_names_the_command_when_there_is_no_run(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="benchmarks.run --scenario moons"):
            widest_run(tmp_path, "moons")

    def test_ignores_a_directory_with_no_manifest(self, tmp_path):
        done = _fake_run(tmp_path, "gaussians", "aaa", n_seeds=5, n_resamples=20)
        (tmp_path / "gaussians" / "orphan").mkdir()
        assert widest_run(tmp_path, "gaussians") == done


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

    def test_refuses_to_promote_an_interrupted_run(self, tmp_path):
        """run_scenario writes its manifest with status "running" before the
        pool starts, so an interrupted run has a manifest beside a partial
        set of checkpoints. Publishing it would commit a partial
        results.csv."""
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
                status="running",
            ),
        )
        with pytest.raises(ValueError, match="status 'running'") as excinfo:
            promote(rd, tmp_path / "published")
        assert str(rd) in str(excinfo.value)
        assert not (tmp_path / "published").exists()

    def test_refuses_a_manifest_with_no_status(self, tmp_path):
        """A manifest written before status was recorded cannot be told apart
        from an interrupted run, so it counts as incomplete, as in
        widest_run."""
        rd = run_dir(tmp_path / "runs", "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row()])
        manifest = build_manifest(
            SCENARIOS["gaussians"],
            run_id="r1",
            config_hash="abc123def456",
            n_seeds=1,
            n_resamples=1,
            n_jobs=1,
            random_state=0,
            wall_clock_s=0.1,
        ).to_dict()
        del manifest["status"]
        write_manifest(rd, manifest)
        with pytest.raises(ValueError, match="no status") as excinfo:
            promote(rd, tmp_path / "published")
        assert str(rd) in str(excinfo.value)
        assert not (tmp_path / "published").exists()


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


class TestAblationSchemas:
    def test_every_per_cell_schema_starts_with_the_cell_key(self):
        for name in ("curves", "selection", "at_k", "cells"):
            assert ABLATION_SCHEMAS[name][: len(CELL_KEY)] == CELL_KEY

    def test_schemas_are_the_documented_contract(self):
        assert ABLATION_SCHEMAS["curves"] == CELL_KEY + (
            "metric_name", "estimator", "k", "metric_value", "metric_se",
        )
        assert ABLATION_SCHEMAS["selection"] == CELL_KEY + (
            "metric_name", "selected_estimator", "selected_k", "k_star", "ari_selected",
        )
        assert ABLATION_SCHEMAS["at_k"] == CELL_KEY + ("mode", "k", "ari_at_k", "rare_recall_at_k")
        assert ABLATION_SCHEMAS["cells"] == CELL_KEY + (
            "carve_random_state", "n_samples", "fit_seconds",
            "consensus_nan_fraction", "consensus_generalizability_nan_fraction",
            "n_cluster_count_warnings",
        )
        assert ABLATION_SCHEMAS["datasets"] == (
            "study", "difficulty", "dataset", "n_samples", "k_star",
            "oracle_ari", "rare_label", "rare_fraction",
        )
        assert ABLATION_SCHEMAS["similarity"] == (
            "study", "difficulty", "dataset", "subsample_ratio", "estimator", "k", "draw", "ari",
        )


class TestAblationFrames:
    def _row(self, **overrides):
        row = {
            "study": "gaussians", "difficulty": "medium", "dataset": 0,
            "subsample_ratio": 0.618, "n_resamples": 100, "replicate": 0,
            "carve_random_state": 10042, "n_samples": 1500, "fit_seconds": 1.0,
            "consensus_nan_fraction": 0.0, "consensus_generalizability_nan_fraction": 0.0,
            "n_cluster_count_warnings": 0,
        }
        row.update(overrides)
        return row

    def test_write_frame_validates_and_orders_columns(self, tmp_path):
        path = write_frame(tmp_path / "cells__x.parquet", [self._row()], ABLATION_SCHEMAS["cells"])
        frame = pd.read_parquet(path)
        assert list(frame.columns) == list(ABLATION_SCHEMAS["cells"])

    def test_write_frame_is_atomic_on_a_failed_write(self, tmp_path, monkeypatch):
        # A worker killed mid-write (or, here, a write that fails after
        # putting bytes on disk) must not leave a file at the final path:
        # unit_done is existence-only, so a resumed run would otherwise
        # treat a half-written parquet file as a completed unit. The mock
        # writes junk to whatever path it is handed before raising, so a
        # write_frame that wrote straight to the final path would leave
        # that junk behind and fail the existence assertion; a mock that
        # only raised could not tell the two apart.
        def boom(self, target, *args, **kwargs):
            Path(target).write_bytes(b"not parquet")
            raise OSError("disk full")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
        path = tmp_path / "cells__x.parquet"
        with pytest.raises(OSError, match="disk full"):
            write_frame(path, [self._row()], ABLATION_SCHEMAS["cells"])
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_write_frame_accepts_no_rows(self, tmp_path):
        path = write_frame(tmp_path / "at_k__x.parquet", [], ABLATION_SCHEMAS["at_k"])
        frame = pd.read_parquet(path)
        assert frame.empty
        assert list(frame.columns) == list(ABLATION_SCHEMAS["at_k"])

    def test_write_frame_rejects_a_row_outside_the_schema(self, tmp_path):
        with pytest.raises(ValueError, match="outside the schema"):
            write_frame(tmp_path / "cells__x.parquet", [self._row(extra=1)], ABLATION_SCHEMAS["cells"])

    def test_read_frames_returns_every_schema_even_when_empty(self, tmp_path):
        frames = read_frames(tmp_path)
        assert set(frames) == set(ABLATION_SCHEMAS)
        for name, frame in frames.items():
            assert list(frame.columns) == list(ABLATION_SCHEMAS[name])
            assert frame.empty

    def test_read_frames_concatenates_by_prefix(self, tmp_path):
        write_frame(tmp_path / "cells__a.parquet", [self._row(dataset=0)], ABLATION_SCHEMAS["cells"])
        write_frame(tmp_path / "cells__b.parquet", [self._row(dataset=1)], ABLATION_SCHEMAS["cells"])
        frames = read_frames(tmp_path)
        assert sorted(frames["cells"]["dataset"]) == [0, 1]
        assert frames["curves"].empty

    def _at_k_row(self):
        return {
            "study": "gaussians", "difficulty": "medium", "dataset": 0,
            "subsample_ratio": 0.618, "n_resamples": 100, "replicate": 0,
            "mode": "default", "k": 5, "ari_at_k": 0.9, "rare_recall_at_k": 1.0,
        }

    def test_read_frames_keeps_dtypes_past_an_empty_part(self, tmp_path):
        # Every study cell writes an empty at_k file, whose columns come
        # back from parquet as nulls. pandas 3 no longer leaves empty
        # entries out when it determines a concat result dtype, so one such
        # file would turn every at_k column object for the whole run (the
        # dev run had all ten object-typed); read_frames must skip empty
        # parts before concatenating.
        at_k = ABLATION_SCHEMAS["at_k"]
        write_frame(tmp_path / "at_k__study.parquet", [], at_k)
        write_frame(tmp_path / "at_k__sim.parquet", [self._at_k_row()], at_k)
        frame = read_frames(tmp_path)["at_k"]
        assert len(frame) == 1
        assert frame["k"].dtype.kind == "i"
        assert frame["ari_at_k"].dtype.kind == "f"
        assert frame["dataset"].dtype.kind == "i"

    def test_read_frames_scores_every_rule_on_the_stability_labels(self, tmp_path):
        """A stored generalizability-rule ari_selected came from the
        generalizability consensus; it is replaced by the stability-mode
        at_k value at the selected k. A study cell, with no at_k, keeps its
        own, and so does an undefined selection.
        """
        key = {
            "study": "gaussians", "difficulty": "medium", "dataset": 0,
            "subsample_ratio": 0.618, "n_resamples": 100, "replicate": 0,
        }
        study_key = {**key, "study": "klein", "difficulty": ""}
        selection = [
            {**key, "metric_name": "ari_generalizability_1se",
             "selected_estimator": "KMeans", "selected_k": 4, "k_star": 5.0,
             "ari_selected": 0.11},
            {**key, "metric_name": "ari_stability_1se",
             "selected_estimator": "KMeans", "selected_k": 5, "k_star": 5.0,
             "ari_selected": 0.95},
            {**key, "metric_name": "ari_average_1se",
             "selected_estimator": None, "selected_k": np.nan, "k_star": 5.0,
             "ari_selected": np.nan},
            {**study_key, "metric_name": "ari_generalizability_1se",
             "selected_estimator": "Ward", "selected_k": 4, "k_star": np.nan,
             "ari_selected": 0.33},
        ]
        at_k = [
            {**key, "mode": mode, "k": k, "ari_at_k": ari, "rare_recall_at_k": 1.0}
            for mode, k, ari in (
                ("default", 4, 0.80), ("default", 5, 0.95),
                ("generalizability", 4, 0.11), ("generalizability", 5, 0.40),
            )
        ]
        write_frame(tmp_path / "selection__a.parquet", selection, ABLATION_SCHEMAS["selection"])
        write_frame(tmp_path / "at_k__a.parquet", at_k, ABLATION_SCHEMAS["at_k"])
        out = read_frames(tmp_path)["selection"]
        assert out["ari_selected"].iloc[:2].tolist() == [0.80, 0.95]
        assert np.isnan(out["ari_selected"].iloc[2])
        assert out["ari_selected"].iloc[3] == 0.33

    def test_read_frames_all_empty_parts_give_the_schema_frame(self, tmp_path):
        at_k = ABLATION_SCHEMAS["at_k"]
        write_frame(tmp_path / "at_k__study_a.parquet", [], at_k)
        write_frame(tmp_path / "at_k__study_b.parquet", [], at_k)
        frame = read_frames(tmp_path)["at_k"]
        assert frame.empty
        assert list(frame.columns) == list(at_k)

    def test_ablation_dir_nests_name_scale_and_hash(self, tmp_path):
        rd = ablation_dir(tmp_path, "rho_b", "dev", "abc123")
        assert rd == tmp_path / "ablation_rho_b" / "dev" / "abc123"
        assert rd.is_dir()


class TestProvenance:
    def test_carries_the_manifest_provenance_fields(self):
        record = provenance()
        assert set(record) == {
            "git_sha", "package_versions", "platform", "peak_rss_bytes", "peak_rss_unit",
        }
        assert record["peak_rss_unit"] == "bytes"

    def test_scenario_identity_is_the_config_hash_input_minus_run_parameters(self):
        scenario = SCENARIOS["gaussians"]
        identity = scenario_identity(scenario)
        assert set(identity) == {
            "name", "axis_name", "axis_values", "axis_labels", "anchors",
            "shared", "estimator", "k_star", "candidate_k", "n_trees",
        }

    def test_write_manifest_accepts_a_mapping(self, tmp_path):
        path = write_manifest(tmp_path, {"run_id": "r1", "ablation": "rho_b"})
        assert json.loads(path.read_text()) == {"run_id": "r1", "ablation": "rho_b"}


class TestFingerprint:
    def test_same_values_give_the_same_digest(self):
        X = np.arange(12, dtype=float).reshape(4, 3)
        assert fingerprint(X) == fingerprint(X.copy())

    def test_a_different_x_of_the_same_shape_gives_a_different_digest(self):
        # Same shape deliberately: a shape check alone would pass this, which
        # is exactly what the fingerprint exists to catch.
        X = np.arange(12, dtype=float).reshape(4, 3)
        Y = X.copy()
        Y[0, 0] += 1.0
        assert fingerprint(X) != fingerprint(Y)

    def test_sidecar_sits_beside_the_cache_file(self, tmp_path):
        cache = tmp_path / "klein.carve"
        assert fingerprint_path(cache) == tmp_path / "klein.carve.x-sha1"

    def test_check_raises_when_the_recorded_digest_differs(self, tmp_path):
        cache = tmp_path / "klein.carve"
        X = np.arange(12, dtype=float).reshape(4, 3)
        Y = X.copy()
        Y[0, 0] += 1.0
        fingerprint_path(cache).write_text(fingerprint(X))
        with pytest.raises(ValueError, match="fit on a different X"):
            check_fingerprint(cache, Y)

    def test_check_warns_when_no_digest_was_recorded(self, tmp_path):
        cache = tmp_path / "klein.carve"
        X = np.arange(12, dtype=float).reshape(4, 3)
        with pytest.warns(UserWarning, match="carries no fingerprint"):
            check_fingerprint(cache, X)

    def test_check_is_silent_when_the_digest_matches(self, tmp_path, recwarn):
        cache = tmp_path / "klein.carve"
        X = np.arange(12, dtype=float).reshape(4, 3)
        fingerprint_path(cache).write_text(fingerprint(X))
        check_fingerprint(cache, X)
        assert len(recwarn) == 0
