"""Tests for reading a copied-back hECA run."""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from benchmarks._heca_calibration import run_calibrate
from benchmarks._heca_report import (
    component_table,
    load_heca_run,
    parse_sacct,
    projection_vs_actual,
    runtime_table,
    selection_table,
    slurm_duration_seconds,
    slurm_memory_bytes,
    study_composition,
)
from benchmarks._heca_stages import run_embed, run_fit
from benchmarks._studies import load_study
from tests.benchmarks._helpers import small_heca_study

SACCT = (
    "JobID|Elapsed|TotalCPU|MaxRSS|AveRSS|NCPUS\n"
    "123.0|1-02:00:00|20-00:00:00|200G|180G|48\n"
)


@pytest.mark.parametrize(
    "text, seconds",
    [("1-02:00:00", 93_600.0), ("02:03:04", 7_384.0), ("05:06.500", 306.5)],
)
def test_slurm_durations(text, seconds):
    assert slurm_duration_seconds(text) == pytest.approx(seconds)


@pytest.mark.parametrize(
    "text, size",
    [("200G", 200 * 1024**3), ("512K", 512 * 1024), ("1234", 1234.0)],
)
def test_slurm_memory(text, size):
    assert slurm_memory_bytes(text) == pytest.approx(size)


def test_blank_slurm_fields_are_nan():
    assert math.isnan(slurm_duration_seconds(""))
    assert math.isnan(slurm_memory_bytes(""))


def test_parse_sacct_converts_the_fields():
    frame = parse_sacct(SACCT)
    row = frame.iloc[0]
    assert row["JobID"] == "123.0"
    assert row["TotalCPU_s"] == pytest.approx(20 * 86_400)
    assert row["MaxRSS_bytes"] == pytest.approx(200 * 1024**3)


def test_study_composition_by_hand():
    obs = pd.DataFrame({"study_id": ["a", "a", "b", "b", "b"]})
    table = study_composition(np.array([0, 0, 0, 1, 1]), obs)
    assert table.loc[0, "a"] == pytest.approx(2 / 3)
    assert table.loc[1, "b"] == pytest.approx(1.0)
    assert table.loc[0, "n_cells"] == 3
    assert table.loc[0, "dominant_share"] == pytest.approx(2 / 3)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A small run through every stage, with a sacct.txt as SLURM writes it."""
    root = tmp_path_factory.mktemp("heca")
    study = small_heca_study(root / "data")
    run = root / "run"
    run_embed(run, study=study)
    run_calibrate(run, study=study, scan=(0.5, 1.0, 2.0))
    run_fit(run, study=study, n_jobs=1, sample_interval=0.05)
    (run / "fit" / "sacct.txt").write_text(SACCT)
    X, _, meta = load_study(study)
    return study, run, X, meta


def test_load_heca_run_reads_every_artifact(finished):
    study, run_dir, X, _ = finished
    run = load_heca_run(run_dir, X=X, study=study)
    assert not run.carve.estimator_results_.empty
    assert set(run.leiden["setting"]) == {"leiden_15", "leiden_50"}
    assert run.sacct is not None
    assert len(run.scan) == 6


def test_the_fit_is_found_after_the_studys_grid_moves_on(finished):
    study, run_dir, X, _ = finished
    moved = replace(study, resolutions=(0.5, 3.0))
    run = load_heca_run(run_dir, X=X, study=moved)
    assert run.study.resolutions == (1.0, 2.0)


def test_runtime_table_prefers_slurms_memory(finished):
    study, run_dir, X, _ = finished
    table = runtime_table(load_heca_run(run_dir, X=X, study=study))
    assert list(table["stage"]) == ["embedding", "CARVE fit"]
    fit = table.set_index("stage").loc["CARVE fit"]
    assert fit["memory_source"] == "slurm"
    assert fit["peak_memory_gb"] == pytest.approx(200 * 1024**3 / 1e9)
    assert fit["cpu_h"] == pytest.approx(20 * 24)
    assert fit["sampled_peak_memory_gb"] > 0


def test_component_shares_sum_to_one_with_the_overhead(finished):
    study, run_dir, X, _ = finished
    table = component_table(load_heca_run(run_dir, X=X, study=study))
    assert set(table["component"]) == {
        "neighbor search",
        "graph assembly",
        "Leiden",
        "forest fit",
        "forest predict",
        "CARVE overhead",
    }
    assert table["share"].sum() == pytest.approx(1.0)


def test_selection_table_scores_both_criteria(finished):
    study, run_dir, X, meta = finished
    table = selection_table(load_heca_run(run_dir, X=X, study=study), meta["obs"])
    assert list(table["measure"]) == ["stability", "generalizability"]
    assert set(table["setting"]) <= {"leiden_15", "leiden_50"}
    assert table["ari_organ"].between(-1, 1).all()


def test_projection_vs_actual_reports_both(finished):
    study, run_dir, X, _ = finished
    out = projection_vs_actual(load_heca_run(run_dir, X=X, study=study))
    assert out["n_jobs"] == 1
    assert out["projected_wall_clock_h"] > 0
    assert out["actual_wall_clock_h"] > 0
