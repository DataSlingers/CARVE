"""Tests for cell enumeration, unit naming, seeds and arm views."""

import dataclasses

import pandas as pd
import pytest

from benchmarks._ablation_cells import (
    REFERENCE_RATIO,
    STUDY_DATASET,
    STUDY_DIFFICULTY,
    Cell,
    Unit,
    arm_view,
    carve_seed,
    cells_frame,
    dataset_seed,
    enumerate_cells,
    enumerate_units,
    scenario_at_scale,
    seed_span,
    similarity_seed,
    timing_units,
    unit_done,
    unit_paths,
    unit_stem,
)
from benchmarks._artifacts import ABLATION_SCHEMAS, CELL_KEY, write_frame
from benchmarks._registry import (
    ABLATIONS,
    PUBLISHED_RANDOM_STATE,
    REPLICATE_SEED_SPACING,
    SCENARIOS,
    SIMILARITY_SEED_OFFSET,
)
from benchmarks._run import benchmark_seed
from benchmarks._types import AblationScale, ArmScale

RHO_B = ABLATIONS["rho_b"]


def _tiny_ablation():
    scale = AblationScale(
        rho_arm=ArmScale(("medium",), (0, 1), 1, 2),
        b_arm=ArmScale(("medium",), (0, 1), 2, 2),
        similarity_draws=2,
        study_scale="publication",
    )
    # The registry's ablation has no case study; this one keeps Klein so the
    # study cells, which the runner still supports, stay covered.
    return dataclasses.replace(
        RHO_B,
        scenarios=("gaussians",),
        study="klein",
        rho_grid=(0.5, RHO_B.rho_default),
        b_grid=(10, RHO_B.b_default),
        scales={"tiny": scale},
        default_scale="tiny",
    )


class TestEnumerateCells:
    def test_counts_match_the_arithmetic(self):
        cells = enumerate_cells(_tiny_ablation(), "tiny")
        # Simulations: rho arm 2 datasets x 2 rho = 4; B arm 2 datasets x 2
        # replicates x 2 B = 8; the two default cells are shared -> 10.
        # Study: rho arm 2 x 2 = 4; B arm 2 x 2 = 4; shared 2 -> 6.
        assert len(cells) == 16
        assert len(set(cells)) == 16

    def test_a_cell_in_both_arms_appears_once(self):
        ablation = _tiny_ablation()
        cells = enumerate_cells(ablation, "tiny")
        shared = [
            c
            for c in cells
            if c.subsample_ratio == ablation.rho_default
            and c.n_resamples == ablation.b_default
            and c.replicate == 0
            and c.study == "gaussians"
            and c.dataset == 0
        ]
        assert len(shared) == 1

    def test_study_cells_come_first_then_simulations_grouped_by_dataset(self):
        ablation = _tiny_ablation()
        cells = enumerate_cells(ablation, "tiny")
        studies = [c.study for c in cells]
        first_sim = studies.index("gaussians")
        assert set(studies[:first_sim]) == {"klein"}
        datasets = [c.dataset for c in cells[first_sim:]]
        assert datasets == sorted(datasets)

    def test_study_cells_use_the_study_key(self):
        cells = enumerate_cells(_tiny_ablation(), "tiny", arm="rho")
        study = [c for c in cells if c.study == "klein"]
        assert {(c.difficulty, c.dataset) for c in study} == {(STUDY_DIFFICULTY, STUDY_DATASET)}

    def test_arm_views_are_subsets_of_the_union(self):
        ablation = _tiny_ablation()
        union = set(enumerate_cells(ablation, "tiny"))
        rho = set(enumerate_cells(ablation, "tiny", arm="rho"))
        b = set(enumerate_cells(ablation, "tiny", arm="b"))
        assert rho | b == union
        assert all(c.n_resamples == ablation.b_default for c in rho)
        assert all(c.subsample_ratio == ablation.rho_default for c in b)

    def test_publication_counts_match_the_spec(self):
        # rho arm: 20 datasets x 8 rho; B arm: 10 datasets x 3 replicates x
        # 5 B; the 10 default cells of replicate 0 are shared. No study cells.
        cells = enumerate_cells(RHO_B, "publication")
        assert {c.study for c in cells} == {"gaussians"}
        assert {c.difficulty for c in cells} == {"medium"}
        assert len(cells) == 20 * 8 + 10 * 3 * 5 - 10

    def test_rejects_an_unknown_arm(self):
        with pytest.raises(ValueError, match="arm"):
            enumerate_cells(_tiny_ablation(), "tiny", arm="gamma")


class TestEnumerateUnits:
    def test_each_dataset_gets_a_dataset_unit_similarity_units_and_its_cells(self):
        ablation = _tiny_ablation()
        units = enumerate_units(ablation, "tiny")
        kinds = [u.kind for u in units]
        assert kinds.count("dataset") == 3  # klein, gaussians 0, gaussians 1
        # rho grid plus the refit reference, per dataset
        assert kinds.count("similarity") == 3 * (len(ablation.rho_grid) + 1)
        assert kinds.count("cell") == 16

    def test_similarity_units_include_the_reference_ratio(self):
        units = enumerate_units(_tiny_ablation(), "tiny")
        ratios = {u.cell.subsample_ratio for u in units if u.kind == "similarity"}
        assert REFERENCE_RATIO in ratios

    def test_units_are_unique(self):
        units = enumerate_units(_tiny_ablation(), "tiny")
        assert len(units) == len(set(units))


class TestTimingUnits:
    def test_is_three_gaussians_cells_without_a_case_study(self):
        # The default, smallest and largest rho at medium, dataset 0.
        units = timing_units(RHO_B, "publication")
        assert {u.kind for u in units} == {"cell"}
        assert [(u.cell.study, u.cell.difficulty, u.cell.subsample_ratio) for u in units] == [
            ("gaussians", "medium", RHO_B.rho_default),
            ("gaussians", "medium", RHO_B.rho_grid[0]),
            ("gaussians", "medium", RHO_B.rho_grid[-1]),
        ]
        assert all(u.cell.n_resamples == RHO_B.b_default for u in units)

    def test_adds_four_study_replicates_with_a_case_study(self):
        units = timing_units(_tiny_ablation(), "tiny")
        assert sum(u.cell.study == "klein" for u in units) == 4


class TestUnitFiles:
    def test_stem_is_deterministic_and_readable(self):
        unit = Unit("cell", Cell("gaussians", "medium", 3, 0.618, 100, 2))
        assert unit_stem(unit) == "gaussians__medium__0003__rho0.618__b0100__rep02"

    def test_study_stem_marks_the_empty_difficulty(self):
        unit = Unit("cell", Cell("klein", STUDY_DIFFICULTY, STUDY_DATASET, 0.2, 100, 0))
        assert unit_stem(unit).startswith("klein__na__0000__")

    def test_cell_units_write_four_frames(self, tmp_path):
        unit = Unit("cell", Cell("gaussians", "medium", 0, 0.618, 100, 0))
        paths = unit_paths(tmp_path, unit)
        assert set(paths) == {"curves", "selection", "at_k", "cells"}
        assert all(p.parent == tmp_path for p in paths.values())

    def test_done_only_when_every_frame_exists(self, tmp_path):
        unit = Unit("cell", Cell("gaussians", "medium", 0, 0.618, 100, 0))
        assert not unit_done(tmp_path, unit)
        for name, path in unit_paths(tmp_path, unit).items():
            write_frame(path, [], ABLATION_SCHEMAS[name])
        assert unit_done(tmp_path, unit)

    def test_dataset_and_similarity_units_write_one_frame(self, tmp_path):
        d = Unit("dataset", Cell("gaussians", "medium", 0, 0.0, 0, 0))
        s = Unit("similarity", Cell("gaussians", "medium", 0, 0.5, 0, 0))
        assert set(unit_paths(tmp_path, d)) == {"datasets"}
        assert set(unit_paths(tmp_path, s)) == {"similarity"}

    def test_stems_are_unique_over_every_unit(self, tmp_path):
        units = enumerate_units(RHO_B, "publication")
        paths = []
        for unit in units:
            paths.extend(unit_paths(tmp_path, unit).values())
        assert len(paths) == len(set(paths)), "Checkpoint paths must be unique to avoid collision"


class TestSeeds:
    def test_simulation_base_is_the_benchmark_seed(self):
        cell = Cell("gaussians", "hard", 7, 0.618, 100, 0)
        assert dataset_seed(cell, ablation=RHO_B) == benchmark_seed(7, 2, PUBLISHED_RANDOM_STATE)

    def test_study_base_is_the_published_random_state(self):
        cell = Cell("klein", STUDY_DIFFICULTY, STUDY_DATASET, 0.618, 100, 0)
        assert dataset_seed(cell, ablation=_tiny_ablation()) == PUBLISHED_RANDOM_STATE

    def test_replicate_zero_reproduces_the_published_fit_seed(self):
        cell = Cell("gaussians", "medium", 0, 0.618, 100, 0)
        assert carve_seed(cell, ablation=RHO_B) == benchmark_seed(0, 1, PUBLISHED_RANDOM_STATE)

    def test_replicates_are_spaced(self):
        c0 = Cell("gaussians", "medium", 0, 0.618, 100, 0)
        c1 = Cell("gaussians", "medium", 0, 0.618, 100, 1)
        assert carve_seed(c1, ablation=RHO_B) - carve_seed(c0, ablation=RHO_B) == REPLICATE_SEED_SPACING

    def test_similarity_seed_offsets_the_base(self):
        assert similarity_seed(42, 3) == 42 + SIMILARITY_SEED_OFFSET + 3

    def test_seed_span_is_three_times_b(self):
        assert seed_span(200) == 600

    def test_publication_seed_windows_of_different_replicates_never_meet(self):
        # A sweep over every publication cell: sort the windows and check
        # that any two that overlap belong to the same replicate index.
        # Setting REPLICATE_SEED_SPACING to 100 makes this fail.
        windows = []
        for cell in enumerate_cells(RHO_B, "publication"):
            start = carve_seed(cell, ablation=RHO_B)
            windows.append((start, start + seed_span(cell.n_resamples), cell.replicate))
        windows.sort()
        reach = -1
        reach_rep = None
        for start, end, rep in windows:
            if start < reach and rep != reach_rep:
                pytest.fail(f"windows of replicates {reach_rep} and {rep} overlap at seed {start}")
            if end > reach:
                reach, reach_rep = end, rep

    def test_publication_similarity_seeds_avoid_every_cell_window(self):
        cells = enumerate_cells(RHO_B, "publication")
        windows = sorted(
            (carve_seed(c, ablation=RHO_B), carve_seed(c, ablation=RHO_B) + seed_span(c.n_resamples))
            for c in cells
        )
        draws = RHO_B.scales["publication"].similarity_draws
        bases = {dataset_seed(c, ablation=RHO_B) for c in cells}
        for base in bases:
            for draw in range(draws):
                seed = similarity_seed(base, draw)
                assert not any(start <= seed < end for start, end in windows), seed

    def test_spacing_of_one_hundred_would_fail(self, monkeypatch):
        import benchmarks._ablation_cells as cells_module

        monkeypatch.setattr(cells_module, "REPLICATE_SEED_SPACING", 100)
        c0 = Cell("gaussians", "medium", 0, 0.618, 200, 0)
        c1 = Cell("gaussians", "medium", 0, 0.618, 200, 1)
        assert carve_seed(c1, ablation=RHO_B) < carve_seed(c0, ablation=RHO_B) + seed_span(200)


class TestScenarioAtScale:
    def test_overrides_n_total_when_set(self):
        scale = dataclasses.replace(RHO_B.scales["publication"], n_total=500)
        assert scenario_at_scale(SCENARIOS["gaussians"], scale).shared["n_total"] == 500

    def test_keeps_the_scenario_when_n_total_is_none(self):
        scale = RHO_B.scales["publication"]
        assert scenario_at_scale(SCENARIOS["gaussians"], scale) is SCENARIOS["gaussians"]


class TestArmView:
    def test_keeps_only_the_arms_cells(self):
        ablation = _tiny_ablation()
        cells = enumerate_cells(ablation, "tiny")
        frame = cells_frame(cells)
        frame["fit_seconds"] = 1.0
        frames = {name: pd.DataFrame(columns=list(schema)) for name, schema in ABLATION_SCHEMAS.items()}
        frames["cells"] = frame
        rho = arm_view(frames, ablation=ablation, scale="tiny", arm="rho")["cells"]
        assert set(rho["n_resamples"]) == {ablation.b_default}
        assert len(rho) == len(enumerate_cells(ablation, "tiny", arm="rho"))

    def test_passes_datasets_and_similarity_through(self):
        ablation = _tiny_ablation()
        frames = {name: pd.DataFrame(columns=list(schema)) for name, schema in ABLATION_SCHEMAS.items()}
        frames["datasets"] = pd.DataFrame([{"study": "x"}])
        view = arm_view(frames, ablation=ablation, scale="tiny", arm="b")
        assert view["datasets"] is frames["datasets"]

    def test_cells_frame_has_the_cell_key_columns(self):
        frame = cells_frame(enumerate_cells(_tiny_ablation(), "tiny"))
        assert list(frame.columns) == list(CELL_KEY)
