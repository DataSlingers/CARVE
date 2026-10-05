"""Tests for the hECA run's stages."""

import json
import os
import time

import pandas as pd
import pytest

from benchmarks._heca_stages import (
    MemorySampler,
    Setting,
    StageOutputExists,
    choose_n_jobs,
    run_embed,
    run_fit,
    scaled_neighbors,
    split_sizes,
    sweep_settings,
)
from benchmarks._studies import STUDIES, carve_cache_path, study_resolution_grids
from benchmarks._timing import read_timings
from carve import CARVE
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


class TestChooseNJobs:
    def test_one_worker_per_core_when_memory_allows(self):
        assert (
            choose_n_jobs(cores=24, memory_bytes=256 * 10**9, worker_peak_bytes=4 * 10**9)
            == 24
        )

    def test_memory_caps_the_workers(self):
        # 0.9 x 256 GB / 20 GB = 11.52.
        assert (
            choose_n_jobs(
                cores=24, memory_bytes=256 * 10**9, worker_peak_bytes=20 * 10**9
            )
            == 11
        )

    def test_never_fewer_than_one(self):
        assert choose_n_jobs(cores=24, memory_bytes=10**9, worker_peak_bytes=10**10) == 1


class TestMemorySampler:
    def test_writes_a_row_on_entry_and_on_exit(self, tmp_path):
        path = tmp_path / "memory.csv"
        with MemorySampler(path, interval=3600.0):
            pass
        frame = pd.read_csv(path)
        assert list(frame.columns) == [
            "elapsed_s",
            "own_rss_bytes",
            "children_rss_bytes",
            "n_children",
        ]
        assert len(frame) == 2
        assert (frame["own_rss_bytes"] > 0).all()

    def test_samples_on_its_interval(self, tmp_path):
        path = tmp_path / "memory.csv"
        with MemorySampler(path, interval=0.01):
            time.sleep(0.2)
        assert len(pd.read_csv(path)) > 2


class TestRunFit:
    def test_fits_every_configuration_and_times_it(self, study, tmp_path, monkeypatch):
        monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
        run = tmp_path / "run"
        record = run_fit(run, study=study, n_jobs=1, sample_interval=0.05)

        fit_dir = run / "fit"
        cache = carve_cache_path(
            study,
            root=fit_dir,
            model_grids=study_resolution_grids(study),
            n_resamples=study.n_resamples,
        )
        assert cache.is_file()
        assert record["cache_file"] == cache.name
        tables = read_timings(fit_dir / "timings")
        # 2 settings x 2 resolutions x 2 resamples x 3 subsets.
        assert len(tables["leiden"]) == 24
        assert int((tables["forest"]["call"] == "fit").sum()) == 8

        results = CARVE.load(str(cache)).estimator_results_
        assert set(results["estimator"]) == {"LeidenClustering"}
        assert sorted(set(results["n_neighbors"])) == [15, 50]

        started = json.loads((fit_dir / "started.json").read_text())
        assert started["n_jobs"] == 1
        assert started["n_resamples"] == 2
        assert started["resolutions"] == [1.0, 2.0]
        assert len(pd.read_csv(fit_dir / "memory.csv")) >= 2
        assert record["wall_clock_s"] > 0
        assert "LOKY_MAX_CPU_COUNT" not in os.environ

        runtime = json.loads((fit_dir / "runtime.json").read_text())
        assert "git_sha" in runtime["provenance"]

    def test_without_n_jobs_it_needs_calibration(self, study, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_fit(tmp_path / "run", study=study)

    def test_refuses_to_overwrite_and_force_starts_the_timings_afresh(
        self, study, tmp_path
    ):
        run = tmp_path / "run"
        run_fit(run, study=study, n_jobs=1)
        with pytest.raises(StageOutputExists):
            run_fit(run, study=study, n_jobs=1)
        run_fit(run, study=study, n_jobs=1, force=True)
        assert len(read_timings(run / "fit" / "timings")["leiden"]) == 24
