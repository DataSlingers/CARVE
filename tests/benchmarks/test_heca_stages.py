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
    fit_status,
    format_status,
    node_memory_bytes,
    physical_cores,
    run_embed,
    run_fit,
    scaled_neighbors,
    split_sizes,
    sweep_settings,
    write_json,
)
from benchmarks._studies import STUDIES, carve_cache_path, study_resolution_grids
from benchmarks._timing import LEIDEN_COLUMNS, read_timings
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


def write_topology(root, siblings):
    for cpu, listed in siblings.items():
        directory = root / f"cpu{cpu}" / "topology"
        directory.mkdir(parents=True)
        (directory / "thread_siblings_list").write_text(listed + "\n")


class TestPhysicalCores:
    def test_counts_the_cores_behind_the_jobs_cpus(self, tmp_path, monkeypatch):
        # Eight hardware threads on four cores (cpu n and n + 4 are siblings);
        # the job holds threads 0, 1, 4 and 5, so two whole cores.
        write_topology(tmp_path, {n: f"{n % 4},{n % 4 + 4}" for n in range(8)})
        monkeypatch.setattr("benchmarks._heca_stages.CPU_TOPOLOGY", tmp_path)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 4, 5}, raising=False)
        assert physical_cores() == 2

    def test_one_thread_per_core_counts_each_core(self, tmp_path, monkeypatch):
        write_topology(tmp_path, {n: f"{n % 4},{n % 4 + 4}" for n in range(8)})
        monkeypatch.setattr("benchmarks._heca_stages.CPU_TOPOLOGY", tmp_path)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
        assert physical_cores() == 3

    def test_without_the_topology_the_machines_cores(self, tmp_path, monkeypatch):
        import psutil

        monkeypatch.setattr("benchmarks._heca_stages.CPU_TOPOLOGY", tmp_path / "absent")
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
        assert physical_cores() == psutil.cpu_count(logical=False)


class TestNodeMemoryBytes:
    def test_slurms_allocation_in_megabytes(self, monkeypatch):
        monkeypatch.setenv("SLURM_MEM_PER_NODE", "716800")
        assert node_memory_bytes() == 700 * 1024**3

    @pytest.mark.parametrize("value", [None, "", "0"])
    def test_otherwise_the_machines_memory(self, monkeypatch, value):
        import psutil

        if value is None:
            monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
        else:
            monkeypatch.setenv("SLURM_MEM_PER_NODE", value)
        assert node_memory_bytes() == psutil.virtual_memory().total


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

    def test_a_failed_sample_warns_once_and_sampling_goes_on(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "memory.csv"
        sample = MemorySampler._sample
        calls = []

        def flaky(self):
            calls.append(None)
            # The first call is __enter__'s; the second, the thread's first.
            if len(calls) == 2:
                raise OSError("disk quota exceeded")
            sample(self)

        monkeypatch.setattr(MemorySampler, "_sample", flaky)
        with pytest.warns(RuntimeWarning, match="disk quota exceeded") as caught:
            with MemorySampler(path, interval=0.01) as sampler:
                deadline = time.monotonic() + 10.0
                while len(calls) < 6 and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert sampler._thread.is_alive()
        assert len(caught) == 1
        # Entry, at least three samples after the failure, and exit.
        assert len(pd.read_csv(path)) >= 5


class TestRunFit:
    def test_fits_every_configuration_and_times_it(
        self, study, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
        run = tmp_path / "run"
        record = run_fit(run, study=study, n_jobs=1, sample_interval=0.05)

        # CARVE reports each configuration's scores as it completes, so the
        # job log shows the fit's progress: one line per configuration.
        progress = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("[CARVE] [")
        ]
        assert len(progress) == 4
        assert all("ARI_stab=" in line and "ARI_gen=" in line for line in progress)

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
        # The two settings are distinct methods, not one method's sweep.
        method_ids = results.groupby("n_neighbors")["method_id"].unique()
        assert all(len(ids) == 1 for ids in method_ids)
        assert method_ids[15][0] != method_ids[50][0]

        started = json.loads((fit_dir / "started.json").read_text())
        assert started["n_jobs"] == 1
        assert started["n_resamples"] == 2
        assert started["resolutions"] == [1.0, 2.0]
        assert len(pd.read_csv(fit_dir / "memory.csv")) >= 2
        assert record["wall_clock_s"] > 0
        assert "LOKY_MAX_CPU_COUNT" not in os.environ

        runtime = json.loads((fit_dir / "runtime.json").read_text())
        assert "git_sha" in runtime["provenance"]
        assert runtime["cpu_s"] > 0
        assert runtime["thread_env"] == {
            "OMP_NUM_THREADS": "3",
            "OPENBLAS_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
        }

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
        # The previous run's accounting, as fit.sbatch leaves it.
        stale = run / "fit" / "sacct.txt"
        stale.write_text("JobID|TotalCPU\n1.0|00:01:00\n")
        run_fit(run, study=study, n_jobs=1, force=True)
        assert len(read_timings(run / "fit" / "timings")["leiden"]) == 24
        assert not stale.exists()

    def test_a_forced_rerun_that_fails_leaves_no_previous_results(
        self, study, tmp_path, monkeypatch
    ):
        run = tmp_path / "run"
        run_fit(run, study=study, n_jobs=1)
        (run / "fit" / "sacct.txt").write_text("JobID|TotalCPU\n1.0|00:01:00\n")

        def killed(*args, **kwargs):
            raise RuntimeError("killed partway")

        monkeypatch.setattr("benchmarks._heca_stages.fit_or_load_carve", killed)
        with pytest.raises(RuntimeError, match="killed partway"):
            run_fit(run, study=study, n_jobs=1, force=True)
        # The new started.json must not sit beside the previous run's results.
        assert (run / "fit" / "started.json").exists()
        assert not (run / "fit" / "runtime.json").exists()
        assert not (run / "fit" / "sacct.txt").exists()


def _leiden_row(n_neighbors, n_samples, resolution, n_clusters):
    return {
        "pid": 1,
        "host": "node",
        "unix_time": 0.0,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "knn_s": 1.0,
        "graph_s": 1.0,
        "leiden_s": 1.0,
        "fit_s": 3.0,
    }


class TestStatus:
    N_CELLS = 1000

    def _write(self, run, rows, *, started_at=1000.0):
        fit_dir = run / "fit"
        write_json(
            fit_dir / "started.json",
            {
                "started_at": started_at,
                "n_cells": self.N_CELLS,
                "n_jobs": 4,
                "resolutions": [1.0, 2.0],
                "n_resamples": 2,
                "settings": ["leiden_15", "leiden_50"],
            },
        )
        (fit_dir / "timings").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=list(LEIDEN_COLUMNS)).to_csv(
            fit_dir / "timings" / "leiden-node-1.csv", index=False
        )

    def _rows(self):
        # leiden_15 at 1.0 is complete (2 resamples x 3 subsets); at 2.0 it
        # has half its rows.
        n_train, n_test = split_sizes(self.N_CELLS)
        setting = Setting("leiden_15", 15)
        k_train = scaled_neighbors(setting, n_fit=n_train, n_full=self.N_CELLS)
        k_test = scaled_neighbors(setting, n_fit=n_test, n_full=self.N_CELLS)
        return (
            [_leiden_row(k_train, n_train, 1.0, 4)] * 4
            + [_leiden_row(k_test, n_test, 1.0, 3)] * 2
            + [_leiden_row(k_train, n_train, 2.0, 7)] * 3
        )

    def test_counts_complete_configurations_and_projects_the_finish(self, tmp_path):
        run = tmp_path / "run"
        self._write(run, self._rows())
        status = fit_status(run, now=1000.0 + 3600.0)
        assert status["completed"] == 1
        assert status["total"] == 4
        assert status["seconds_per_configuration"] == pytest.approx(3600.0)
        assert status["projected_total_s"] == pytest.approx(4 * 3600.0)
        assert status["projected_finish"] == pytest.approx(1000.0 + 4 * 3600.0)
        assert status["abort"] is False
        assert status["clusters"].loc[1.0, "leiden_15"] == 4
        assert status["clusters"].loc[2.0, "leiden_15"] == 7

    def test_the_abort_rule_fires_past_ten_days(self, tmp_path):
        run = tmp_path / "run"
        self._write(run, self._rows())
        # One configuration in three days projects twelve for four.
        status = fit_status(run, now=1000.0 + 3 * 86_400)
        assert status["abort"] is True

    def test_no_projection_before_a_configuration_completes(self, tmp_path):
        run = tmp_path / "run"
        self._write(run, self._rows()[-3:])
        status = fit_status(run, now=5000.0)
        assert status["completed"] == 0
        assert status["seconds_per_configuration"] is None
        assert status["projected_total_s"] is None
        assert status["abort"] is False

    def test_format_names_the_count_and_the_rule(self, tmp_path):
        run = tmp_path / "run"
        self._write(run, self._rows())
        text = format_status(fit_status(run, now=1000.0 + 3600.0))
        assert "Configurations complete: 1 of 4" in text
        assert "Abort rule" in text
        assert "not triggered" in text
