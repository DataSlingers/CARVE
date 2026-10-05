"""Tests for the hECA run's stages."""

import json

import pytest

from benchmarks._heca_stages import (
    Setting,
    StageOutputExists,
    run_embed,
    scaled_neighbors,
    split_sizes,
    sweep_settings,
)
from benchmarks._studies import STUDIES
from carve._utils import split_subsample_indices
from tests.benchmarks._helpers import small_heca_study


@pytest.fixture
def study(tmp_path):
    return small_heca_study(tmp_path / "data")


class TestSweep:
    def test_settings_follow_the_studys_grids(self):
        assert sweep_settings(STUDIES["heca"]) == [
            Setting("leiden_15", 15),
            Setting("leiden_50", 50),
        ]

    @pytest.mark.parametrize("n", [10, 113, 1000, 695_304])
    def test_split_sizes_match_carves_split(self, n):
        train, test = split_subsample_indices(n, subsample_ratio=0.618, random_state=0)
        assert split_sizes(n) == (len(train), len(test))

    @pytest.mark.parametrize(
        "setting, n_fit, expected",
        [
            (Setting("leiden_15", 15), 429_698, 9),
            (Setting("leiden_15", 15), 265_606, 6),
            (Setting("leiden_50", 50), 429_698, 31),
            (Setting("leiden_50", 50), 265_606, 19),
        ],
    )
    def test_neighbors_are_scaled_as_carve_scales_them(self, setting, n_fit, expected):
        assert scaled_neighbors(setting, n_fit=n_fit, n_full=695_304) == expected


class TestEmbed:
    def test_records_the_pooled_counts(self, study, tmp_path):
        run = tmp_path / "run"
        record = run_embed(run, study=study)
        assert json.loads((run / "embed.json").read_text()) == record
        assert record["n_organs"] == 2
        assert record["n_cell_types"] == 2
        assert record["n_studies"] == 2
        assert record["n_features"] == 5
        assert record["cached"] is False
        env = json.loads((run / "env.json").read_text())
        assert [stage["stage"] for stage in env["stages"]] == ["embed"]
        assert "git_sha" in env["provenance"]
        assert env["stages"][0]["physical_cores"] >= 1

    def test_refuses_to_overwrite_and_force_records_a_second_run(self, study, tmp_path):
        run = tmp_path / "run"
        run_embed(run, study=study)
        with pytest.raises(StageOutputExists):
            run_embed(run, study=study)
        record = run_embed(run, study=study, force=True)
        assert record["cached"] is True
        env = json.loads((run / "env.json").read_text())
        assert len(env["stages"]) == 2
