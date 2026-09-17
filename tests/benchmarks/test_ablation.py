"""Tests for the ablation runner: unit executors, hashing, resume."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

import benchmarks._ablation as ablation_module
import benchmarks._ablation_cells as cells_module
from benchmarks._ablation import (
    ablation_hash,
    cell_rows,
    dataset_rows,
    run_ablation,
    similarity_rows,
)
from benchmarks._ablation_cells import (
    REFERENCE_RATIO,
    STUDY_DATASET,
    STUDY_DIFFICULTY,
    Cell,
    Unit,
    enumerate_cells,
    enumerate_units,
    unit_paths,
)
from benchmarks._artifacts import ABLATION_SCHEMAS, CELL_KEY, read_frames
from benchmarks._registry import CARVE_METRICS_ALL
from benchmarks._types import (
    Ablation,
    AblationScale,
    ArmScale,
    Axis,
    EstimatorSpec,
    Scenario,
    Study,
)

TINY_SCENARIO = Scenario(
    name="tiny",
    axis=Axis(name="difficulty_level", values=(0, 1), labels=("easy", "hard")),
    anchors={"easy": {"cluster_scale": 0.6}, "hard": {"cluster_scale": 2.5}},
    shared={"n_total": 120, "p": 4, "distribution": "gaussian"},
    estimator=EstimatorSpec(name="kmeans"),
    candidate_k=(3, 4, 5),
    n_seeds=2,
)


def _blobs(subsample):
    X, y = make_blobs(n_samples=90, centers=3, n_features=4, random_state=0)
    return X, y, {}


PROBE_STUDY = Study(
    name="probe",
    loader=_blobs,
    estimator=EstimatorSpec(name="agglomerative"),
    candidate_k=(2, 3, 4),
    scales={"dev": None},
    default_scale="dev",
    partners=(),
    not_two=True,
)

TINY_SCALE = AblationScale(
    rho_arm=ArmScale(("easy",), (0, 1), 1, 2),
    b_arm=ArmScale(("easy",), (0, 1), 2, 2),
    similarity_draws=2,
    study_scale="dev",
)

TINY_ABLATION = Ablation(
    name="tiny",
    scenarios=("tiny",),
    study="probe",
    rho_grid=(0.5, 0.618),
    b_grid=(8, 12),
    rho_default=0.618,
    b_default=12,
    scales={"dev": TINY_SCALE, "big": dataclasses.replace(TINY_SCALE, similarity_draws=3)},
    default_scale="dev",
)


@pytest.fixture(scope="module", autouse=True)
def tiny_registry():
    """Point both ablation modules at the tiny scenario and study."""
    mp = pytest.MonkeyPatch()
    mp.setattr(cells_module, "SCENARIOS", {"tiny": TINY_SCENARIO})
    mp.setattr(ablation_module, "SCENARIOS", {"tiny": TINY_SCENARIO})
    mp.setattr(ablation_module, "STUDIES", {"probe": PROBE_STUDY})
    yield
    mp.undo()


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    """A finished dev-scale run of the tiny ablation, at n_jobs=1 so unit
    functions run in this process."""
    root = tmp_path_factory.mktemp("ablation")
    rd = run_ablation(TINY_ABLATION, scale="dev", root=root, n_jobs=1)
    return root, rd


class TestDatasetRows:
    def test_simulated_dataset_records_truth_and_oracle(self):
        unit = Unit("dataset", Cell("tiny", "easy", 0, 0.0, 0, 0))
        rows = dataset_rows(unit, ablation=TINY_ABLATION, scale="dev", data=None)
        assert len(rows) == 1
        row = rows[0]
        assert set(row) == set(ABLATION_SCHEMAS["datasets"])
        assert row["k_star"] == 5
        assert row["n_samples"] == 120
        assert 0.0 < row["rare_fraction"] <= 0.2
        assert -1.0 <= row["oracle_ari"] <= 1.0

    def test_study_dataset_has_nan_truth(self):
        X, y, _ = _blobs(None)
        unit = Unit("dataset", Cell("probe", STUDY_DIFFICULTY, STUDY_DATASET, 0.0, 0, 0))
        rows = dataset_rows(unit, ablation=TINY_ABLATION, scale="dev", data=(X, y))
        assert rows[0]["n_samples"] == 90
        assert np.isnan(rows[0]["k_star"]) and np.isnan(rows[0]["rare_label"])


class TestSimilarityRows:
    def test_wards_refit_reference_is_one(self):
        X, y, _ = _blobs(None)
        unit = Unit("similarity", Cell("probe", STUDY_DIFFICULTY, STUDY_DATASET, REFERENCE_RATIO, 0, 0))
        rows = similarity_rows(unit, ablation=TINY_ABLATION, scale="dev", data=(X, y))
        assert rows and all(r["ari"] == 1.0 for r in rows)
        assert {r["estimator"] for r in rows} == {"AgglomerativeClustering"}
        assert {r["k"] for r in rows} == {2, 3, 4}
        assert {r["draw"] for r in rows} == {0, 1}

    def test_subsample_rows_score_the_draw_against_the_full_fit(self):
        unit = Unit("similarity", Cell("tiny", "easy", 0, 0.5, 0, 0))
        rows = similarity_rows(unit, ablation=TINY_ABLATION, scale="dev", data=None)
        assert len(rows) == 3 * 2  # candidate k x draws
        assert all(-1.0 <= r["ari"] <= 1.0 for r in rows)
        assert {r["subsample_ratio"] for r in rows} == {0.5}

    def test_a_draw_has_the_subsample_size_carve_uses(self):
        from carve._utils import split_subsample_indices

        idx, rest = split_subsample_indices(90, subsample_ratio=0.5, random_state=1)
        assert idx.size == int(0.5 * 90)
        assert idx.size + rest.size == 90


class TestCellRows:
    @classmethod
    @pytest.fixture(scope="class")
    def sim_cell(cls):
        unit = Unit("cell", Cell("tiny", "easy", 0, 0.5, 12, 1))
        return cell_rows(unit, ablation=TINY_ABLATION, scale="dev", data=None, thread_cap=None)

    def test_every_frame_carries_its_schema(self, sim_cell):
        for name in ("curves", "selection", "at_k", "cells"):
            assert sim_cell[name], name
            assert set(sim_cell[name][0]) == set(ABLATION_SCHEMAS[name]), name

    def test_curves_cover_every_metric_and_k(self, sim_cell):
        curves = pd.DataFrame(sim_cell["curves"])
        assert set(curves["metric_name"]) == set(CARVE_METRICS_ALL)
        assert set(curves["k"]) == {3, 4, 5}
        assert curves["metric_se"].notna().sum() > 0
        assert curves.loc[curves["metric_name"] == "consensus_pac_stability", "metric_se"].isna().all()

    def test_one_selection_per_metric_with_a_true_k(self, sim_cell):
        selection = pd.DataFrame(sim_cell["selection"])
        assert len(selection) == len(CARVE_METRICS_ALL)
        assert set(selection["k_star"]) == {5.0}
        assert set(selection["selected_k"]) <= {3, 4, 5}

    def test_at_k_rows_cover_both_modes(self, sim_cell):
        at_k = pd.DataFrame(sim_cell["at_k"])
        assert set(at_k["mode"]) == {"default", "generalizability"}
        assert len(at_k) == 2 * 3
        assert at_k["rare_recall_at_k"].between(0, 1).all()

    def test_cell_row_records_the_replicate_seed(self, sim_cell):
        row = sim_cell["cells"][0]
        assert row["replicate"] == 1
        assert row["carve_random_state"] == cells_module.carve_seed(
            Cell("tiny", "easy", 0, 0.5, 12, 1), ablation=TINY_ABLATION
        )
        assert row["n_samples"] == 120
        assert 0.0 <= row["consensus_nan_fraction"] <= 1.0

    def test_study_cell_has_no_at_k_rows_and_honors_not_two(self):
        X, y, _ = _blobs(None)
        unit = Unit("cell", Cell("probe", STUDY_DIFFICULTY, STUDY_DATASET, 0.618, 8, 0))
        rows = cell_rows(unit, ablation=TINY_ABLATION, scale="dev", data=(X, y), thread_cap=None)
        assert rows["at_k"] == []
        selection = pd.DataFrame(rows["selection"])
        assert selection["k_star"].isna().all()
        assert (selection["selected_k"] != 2).all()
        assert selection["ari_selected"].between(-1, 1).all()


class TestAblationHash:
    def test_is_stable(self):
        assert ablation_hash(TINY_ABLATION, "dev") == ablation_hash(TINY_ABLATION, "dev")

    def test_differs_between_scales(self):
        assert ablation_hash(TINY_ABLATION, "dev") != ablation_hash(TINY_ABLATION, "big")

    def test_changes_with_the_grid(self):
        other = dataclasses.replace(TINY_ABLATION, rho_grid=(0.4, 0.618))
        assert ablation_hash(other, "dev") != ablation_hash(TINY_ABLATION, "dev")

    def test_changes_with_the_studys_selection_setting(self):
        mp = pytest.MonkeyPatch()
        mp.setattr(
            ablation_module,
            "STUDIES",
            {"probe": dataclasses.replace(PROBE_STUDY, not_two=False)},
        )
        try:
            changed = ablation_hash(TINY_ABLATION, "dev")
        finally:
            mp.undo()
        assert changed != ablation_hash(TINY_ABLATION, "dev")

    def test_is_twelve_hex_characters(self):
        assert len(ablation_hash(TINY_ABLATION, "dev")) == 12


class TestRunAblation:
    def test_writes_every_unit_and_a_manifest(self, completed):
        _, rd = completed
        units = enumerate_units(TINY_ABLATION, "dev")
        assert all(all(p.exists() for p in unit_paths(rd, u).values()) for u in units)
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["ablation"] == "tiny"
        assert manifest["scale"] == "dev"
        assert manifest["workers"] == 1
        assert manifest["n_units"] == len(units)
        assert "git_sha" in manifest and "thread_cap" in manifest

    def test_run_directory_nests_scale_and_hash(self, completed):
        root, rd = completed
        assert rd == root / "ablation_tiny" / "dev" / ablation_hash(TINY_ABLATION, "dev")

    def test_frames_read_back_with_one_row_per_cell(self, completed):
        _, rd = completed
        frames = read_frames(rd)
        cells = enumerate_cells(TINY_ABLATION, "dev")
        assert len(frames["cells"]) == len(cells) == 16
        assert frames["cells"][list(CELL_KEY)].drop_duplicates().shape[0] == 16
        assert len(frames["selection"]) == 16 * len(CARVE_METRICS_ALL)
        assert len(frames["datasets"]) == 3
        # Simulated cells only: 10 cells x 2 modes x 3 k
        assert len(frames["at_k"]) == 10 * 2 * 3
        # 3 datasets x (2 rho + reference) x k values x 2 draws
        assert len(frames["similarity"]) == 2 * 3 * 3 * 2 + 1 * 3 * 3 * 2

    def test_the_shared_default_cell_is_fitted_once(self, completed):
        _, rd = completed
        cells = read_frames(rd)["cells"]
        shared = cells[
            (cells["study"] == "tiny")
            & (cells["dataset"] == 0)
            & (cells["replicate"] == 0)
            & (cells["subsample_ratio"] == 0.618)
            & (cells["n_resamples"] == 12)
        ]
        assert len(shared) == 1

    def test_resume_recomputes_only_the_missing_unit(self, completed, tmp_path, monkeypatch):
        import shutil

        root, rd = completed
        new_root = tmp_path / "runs"
        shutil.copytree(root, new_root)
        new_rd = new_root / rd.relative_to(root)
        unit = next(u for u in enumerate_units(TINY_ABLATION, "dev") if u.kind == "cell")
        for path in unit_paths(new_rd, unit).values():
            path.unlink()

        calls = []
        original = ablation_module.run_unit

        def counting(unit, **kwargs):
            calls.append(unit)
            return original(unit, **kwargs)

        monkeypatch.setattr(ablation_module, "run_unit", counting)
        run_ablation(TINY_ABLATION, scale="dev", root=new_root, n_jobs=1)
        assert calls == [unit]

    def test_resume_keeps_the_run_id_and_accumulates_wall_clock(self, completed, tmp_path):
        import shutil

        root, rd = completed
        new_root = tmp_path / "runs"
        shutil.copytree(root, new_root)
        new_rd = new_root / rd.relative_to(root)
        before = json.loads((new_rd / "manifest.json").read_text())
        run_ablation(TINY_ABLATION, scale="dev", root=new_root, n_jobs=1)
        after = json.loads((new_rd / "manifest.json").read_text())
        assert after["run_id"] == before["run_id"]
        assert after["wall_clock_s"] >= before["wall_clock_s"]

    def test_no_resume_mints_a_new_run_id(self, completed, tmp_path):
        import shutil

        root, rd = completed
        new_root = tmp_path / "runs"
        shutil.copytree(root, new_root)
        new_rd = new_root / rd.relative_to(root)
        before = json.loads((new_rd / "manifest.json").read_text())
        run_ablation(TINY_ABLATION, scale="dev", root=new_root, n_jobs=1, resume=False)
        after = json.loads((new_rd / "manifest.json").read_text())
        assert after["run_id"] != before["run_id"]

    def test_explicit_units_run_only_those(self, tmp_path):
        units = [u for u in enumerate_units(TINY_ABLATION, "dev") if u.kind == "dataset"][:1]
        rd = run_ablation(TINY_ABLATION, scale="dev", root=tmp_path, n_jobs=1, units=units)
        frames = read_frames(rd)
        assert len(frames["datasets"]) == 1
        assert frames["cells"].empty

    def test_rejects_an_unknown_scale(self, tmp_path):
        with pytest.raises(ValueError, match="scale"):
            run_ablation(TINY_ABLATION, scale="huge", root=tmp_path, n_jobs=1)

    def test_replicates_differ_and_default_cells_agree_across_arms(self, completed):
        # Two replicates of one setting differ (different seeds); the cell
        # shared by both arms has one row, checked above, so nothing to
        # compare there. Curves of replicate 0 and 1 at (0.618, 12) differ.
        _, rd = completed
        curves = read_frames(rd)["curves"]
        rows = curves[
            (curves["study"] == "tiny")
            & (curves["dataset"] == 0)
            & (curves["subsample_ratio"] == 0.618)
            & (curves["n_resamples"] == 12)
            & (curves["metric_name"] == "ari_stability")
        ]
        by_rep = rows.pivot(index="k", columns="replicate", values="metric_value")
        assert not np.allclose(by_rep[0], by_rep[1])
