"""Tests for the runtime and memory measurements of a study's CARVE fit."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

import carve._runner as runner
from benchmarks._runtime import (
    COLUMNS,
    Measurement,
    RuntimeSpec,
    component_timers,
    measurement_plan,
    projected_hours,
    read_runtime,
    run_measurements,
    runtime_ladder_latex,
    runtime_ladder_table,
    runtime_parallel_table,
    runtime_provenance,
    runtime_slopes,
    subset_indices,
)
from benchmarks._studies import STUDIES, fit_carve, study_resolution_grids


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0, 0.0, 0.0], [6.0, 0.0, 1.0, 0.0], [3.0, 5.0, 0.0, 1.0]])
    labels = rng.integers(0, 3, 300)
    return centers[labels] + rng.normal(0, 0.6, (300, 4)), labels


@pytest.fixture
def toy_spec():
    return RuntimeSpec(
        estimator="kmeans",
        model_grids=[(KMeans, {"n_clusters": [2, 3], "n_init": [2]})],
        normalization_options=[(FunctionTransformer, {})],
        dim_reduction_options=[
            (PCA, {"n_components": [2]}),
            (PCA, {"n_components": [3]}),
        ],
    )


# --- plan and subsets ------------------------------------------------------


def test_plan_ends_on_every_cell_and_skips_rungs_above_it():
    plan = measurement_plan(30_000)
    ladder = [m for m in plan if m.label == "ladder"]
    for setting in ("fixed", "randomized"):
        cells = [m.n_cells for m in ladder if m.setting == setting]
        assert cells == [5_000, 10_000, 25_000, 30_000]
    assert {(m.n_resamples, m.n_jobs) for m in ladder} == {(4, 1)}


def test_plan_runs_the_parallel_check_on_the_randomized_setting():
    parallel = [m for m in measurement_plan(30_000) if m.label == "parallel"]
    assert [m.n_jobs for m in parallel] == [1, 4, 8]
    assert {(m.setting, m.n_cells, m.n_resamples) for m in parallel} == {
        ("randomized", 10_000, 8)
    }


def test_plan_clips_the_parallel_check_to_a_smaller_data_set():
    parallel = [m for m in measurement_plan(6_000) if m.label == "parallel"]
    assert {m.n_cells for m in parallel} == {6_000}


def test_plan_without_randomized_preprocessing_measures_the_fixed_setting_only():
    plan = measurement_plan(30_000, settings=("fixed",))
    assert {m.setting for m in plan} == {"fixed"}


def test_subset_indices_are_a_sorted_seeded_draw():
    idx = subset_indices(1_000, 200, random_state=42)
    assert idx.size == 200
    assert np.unique(idx).size == 200
    assert np.all(np.diff(idx) > 0)
    np.testing.assert_array_equal(idx, subset_indices(1_000, 200, random_state=42))
    assert not np.array_equal(idx, subset_indices(1_000, 200, random_state=7))


def test_subset_indices_keep_every_cell_at_full_size():
    np.testing.assert_array_equal(subset_indices(500, 500, random_state=42), np.arange(500))
    np.testing.assert_array_equal(subset_indices(500, 900, random_state=42), np.arange(500))


# --- timers -----------------------------------------------------------------


def test_component_timers_split_a_fit_and_restore_what_they_wrap(blobs, toy_spec):
    X, y = blobs
    original_fit = KMeans.fit
    original_embed = runner.embed_resample
    with component_timers(toy_spec.model_grids) as times:
        fit_carve(
            X,
            y,
            model_grids=toy_spec.model_grids,
            n_resamples=4,
            n_jobs=1,
            random_state=0,
            randomize_preprocessing=True,
            normalization_options=toy_spec.normalization_options,
            dim_reduction_options=toy_spec.dim_reduction_options,
        )
    assert set(times) == {"embedding", "clustering", "forest_fit", "forest_predict"}
    assert all(seconds > 0 for seconds in times.values())
    assert KMeans.fit is original_fit
    assert runner.embed_resample is original_embed
    # Both forest methods are inherited; restoring must not leave a copy on
    # the class itself.
    assert "fit" not in vars(RandomForestClassifier)
    assert "predict" not in vars(RandomForestClassifier)


def test_component_timers_count_nested_calls_once(monkeypatch):
    # A clock that advances one second per reading: an outer call read twice
    # lasts one second, and timing the nested call as well would read it four
    # times and add the inner second to the outer call's three.
    ticks = iter(range(100))
    monkeypatch.setattr("benchmarks._runtime._clock", lambda: float(next(ticks)))

    class Nested(KMeans):
        def fit(self, X, y=None, sample_weight=None):
            if getattr(self, "_inner", False):
                return super().fit(X)
            self._inner = True
            return Nested.fit(self, X)

    X = np.random.default_rng(0).normal(size=(50, 2))
    with component_timers([(Nested, {})]) as times:
        Nested(n_clusters=2, n_init=1).fit(X)
    assert times["clustering"] == 1.0


# --- measuring, storing and reading back ------------------------------------


def _plan():
    return [
        Measurement("ladder", "fixed", 200, 2, 1),
        Measurement("ladder", "randomized", 300, 4, 1),
        Measurement("parallel", "randomized", 300, 4, 2),
    ]


def test_run_measurements_stores_one_row_per_fit(tmp_path, blobs, toy_spec):
    X, y = blobs
    path = tmp_path / "runtime.csv"
    frame = run_measurements(toy_spec, X, y, path=path, plan=_plan(), random_state=0)

    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == 3
    pd.testing.assert_frame_equal(frame, read_runtime(path))

    fixed, randomized, parallel = (frame.iloc[i] for i in range(3))
    assert (fixed.n_cells, randomized.n_cells) == (200, 300)
    assert fixed.n_configs == 2
    # Every fit records where its time went with one worker, and the fixed
    # setting draws no pipeline, so it embeds nothing.
    assert fixed.embedding_s == 0
    assert fixed.pipelines == ""
    assert randomized.embedding_s > 0
    for column in ("clustering_s", "forest_fit_s", "forest_predict_s"):
        assert fixed[column] > 0
        assert randomized[column] > 0
    # With two workers the calls run in child processes, so nothing is split.
    for column in ("embedding_s", "clustering_s", "forest_fit_s", "forest_predict_s"):
        assert np.isnan(parallel[column])
    assert json.loads(randomized.pipelines) == {
        "identity | PCA(n_components=2)": 2,
        "identity | PCA(n_components=3)": 2,
    }
    assert (frame.peak_memory_gb > 0).all()
    assert (frame.wall_s > 0).all()


def test_run_measurements_reads_stored_rows_back_instead_of_fitting_again(
    tmp_path, blobs, toy_spec, monkeypatch
):
    X, y = blobs
    path = tmp_path / "runtime.csv"
    plan = _plan()[:1]
    first = run_measurements(toy_spec, X, y, path=path, plan=plan, random_state=0)

    def refuse(*args, **kwargs):
        raise AssertionError("a stored measurement was run again")

    monkeypatch.setattr(subprocess, "run", refuse)
    again = run_measurements(toy_spec, X, y, path=path, plan=plan, random_state=0)
    pd.testing.assert_frame_equal(first, again)


def test_run_measurements_runs_only_the_fits_the_file_lacks(
    tmp_path, blobs, toy_spec, monkeypatch
):
    X, y = blobs
    path = tmp_path / "runtime.csv"
    plan = _plan()[:2]
    run_measurements(toy_spec, X, y, path=path, plan=plan[:1], random_state=0)

    calls = []
    real_run = subprocess.run

    def counting(*args, **kwargs):
        calls.append(args)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", counting)
    frame = run_measurements(toy_spec, X, y, path=path, plan=plan, random_state=0)
    assert len(calls) == 1
    assert list(frame.setting) == ["fixed", "randomized"]


def test_reading_rows_measured_with_other_pipelines_warns(tmp_path, blobs, toy_spec):
    X, y = blobs
    path = tmp_path / "runtime.csv"
    plan = _plan()[1:2]
    frame = run_measurements(toy_spec, X, y, path=path, plan=plan, random_state=0)
    frame.loc[0, "pipelines"] = json.dumps({"identity | TSNE(perplexity=300)": 4})
    frame.to_csv(path, index=False)

    with pytest.warns(UserWarning, match="other pipelines"):
        run_measurements(toy_spec, X, y, path=path, plan=plan, random_state=0)


def test_reading_rows_measured_with_another_grid_warns(tmp_path, blobs, toy_spec):
    X, y = blobs
    path = tmp_path / "runtime.csv"
    plan = _plan()[:1]
    frame = run_measurements(toy_spec, X, y, path=path, plan=plan, random_state=0)
    frame.loc[0, "n_configs"] = 15
    frame.to_csv(path, index=False)

    with pytest.warns(UserWarning, match="configurations"):
        run_measurements(toy_spec, X, y, path=path, plan=plan, random_state=0)


def test_read_runtime_rejects_a_file_with_missing_columns(tmp_path):
    path = tmp_path / "runtime.csv"
    pd.DataFrame({"label": ["ladder"], "wall_s": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        read_runtime(path)


def test_read_runtime_of_a_missing_file_is_empty(tmp_path):
    frame = read_runtime(tmp_path / "absent.csv")
    assert frame.empty
    assert list(frame.columns) == list(COLUMNS)


def test_spec_from_cusanovich_matches_the_notebooks_fit():
    study = STUDIES["cusanovich"]
    spec = RuntimeSpec.from_study(study)
    assert spec.estimator == "louvain"
    assert spec.consensus_anchors == study.consensus_anchors
    assert repr(spec.model_grids) == repr(study_resolution_grids(study))
    assert len(spec.dim_reduction_options) == len(study.preprocessing.dim_reduction)


# --- tables -------------------------------------------------------------------


def _row(**values):
    row = dict.fromkeys(COLUMNS, np.nan)
    row.update(
        estimator="louvain",
        n_configs=15,
        pipelines="",
        random_state=42,
        started="2026-09-15T19:09:25",
        carve_commit="3975da5",
        python="3.12.11",
        sklearn="1.7.2",
        machine="arm64",
        cpu_count=11,
        ram_gb=19.3,
        n_anchors=2000,
        consensus_retained_gb=0.48,
    )
    row.update(values)
    return row


def _ladder_frame():
    rows = []
    for setting, scale in (("fixed", 1.0), ("randomized", 10.0)):
        for cells in (5_000, 10_000, 20_000, 40_000):
            wall = scale * 60.0 * (cells / 5_000) ** 1.5
            rows.append(
                _row(
                    label="ladder",
                    setting=setting,
                    n_cells=cells,
                    n_resamples=4,
                    n_jobs=1,
                    wall_s=wall,
                    embedding_s=0.0 if setting == "fixed" else 0.5 * wall,
                    clustering_s=0.2 * wall,
                    forest_fit_s=0.1 * wall,
                    forest_predict_s=0.05 * wall,
                    peak_memory_gb=1.0 + cells / 10_000,
                )
            )
    return pd.DataFrame(rows, columns=list(COLUMNS))


def test_ladder_table_reports_per_resample_projection_shares_and_memory():
    table = runtime_ladder_table(_ladder_frame())
    assert list(table.setting) == ["fixed"] * 4 + ["randomized"] * 4
    first = table.iloc[0]
    assert first.cells == 5_000
    assert first.per_resample_s == pytest.approx(15.0)
    assert first.projected_h == pytest.approx(15.0 * 100 / 3600)
    assert first.embedding_pct == pytest.approx(0.0)
    assert first.clustering_pct == pytest.approx(20.0)
    assert first.forest_pct == pytest.approx(15.0)
    assert first.other_pct == pytest.approx(65.0)
    assert first.peak_memory_gb == pytest.approx(1.5)
    assert table.iloc[4].embedding_pct == pytest.approx(50.0)
    assert table.iloc[4].other_pct == pytest.approx(15.0)


def test_ladder_table_projects_to_the_requested_resample_count():
    table = runtime_ladder_table(_ladder_frame(), n_resamples=150)
    assert table.iloc[0].projected_h == pytest.approx(15.0 * 150 / 3600)


def test_ladder_latex_marks_the_fixed_setting_as_embedding_nothing():
    latex = runtime_ladder_latex(_ladder_frame())
    assert r"\toprule" in latex and r"\bottomrule" in latex
    assert "Louvain" in latex
    assert "$B = 100$" in latex
    fixed_line = next(line for line in latex.splitlines() if line.startswith("5{,}000 & 15.0"))
    assert " -- " in fixed_line
    randomized_line = next(
        line for line in latex.splitlines() if line.startswith("5{,}000 & 150.0")
    )
    assert " -- " not in randomized_line


def test_slopes_recover_the_power_of_the_cell_count():
    slopes = runtime_slopes(_ladder_frame(), min_cells=10_000)
    assert slopes.loc["total", "fixed"] == pytest.approx(1.5)
    assert slopes.loc["total", "randomized"] == pytest.approx(1.5)
    assert slopes.loc["embedding", "randomized"] == pytest.approx(1.5)
    assert np.isnan(slopes.loc["embedding", "fixed"])


def test_projected_hours_scale_the_largest_rung_per_resample():
    hours = projected_hours(_ladder_frame(), n_resamples=100).set_index("setting")
    per_resample_s = 60.0 * 8**1.5 / 4
    assert hours.loc["fixed", "cells"] == 40_000
    assert hours.loc["fixed", "hours"] == pytest.approx(per_resample_s * 100 / 3600)
    assert hours.loc["randomized", "hours"] == pytest.approx(10 * per_resample_s * 100 / 3600)


def test_parallel_table_reports_speedup_against_one_worker():
    rows = [
        _row(
            label="parallel",
            setting="randomized",
            n_cells=10_000,
            n_resamples=8,
            n_jobs=jobs,
            wall_s=wall,
            peak_memory_gb=memory,
        )
        for jobs, wall, memory in ((1, 1000.0, 2.0), (4, 500.0, 3.0), (8, 800.0, 4.0))
    ]
    table = runtime_parallel_table(pd.DataFrame(rows, columns=list(COLUMNS)))
    assert list(table.n_jobs) == [1, 4, 8]
    assert list(table.speedup) == pytest.approx([1.0, 2.0, 1.25])
    assert list(table.peak_memory_gb) == pytest.approx([2.0, 3.0, 4.0])


def test_provenance_groups_fits_by_when_where_and_what_they_measured():
    frame = _ladder_frame()
    frame.loc[frame.setting == "randomized", "pipelines"] = json.dumps(
        {"identity | TSNE(perplexity=30)": 2, "identity | identity": 2}
    )
    provenance = runtime_provenance(frame)
    assert list(provenance.fits) == [4, 4]
    assert set(provenance.carve_commit) == {"3975da5"}
    assert set(provenance.measured) == {"2026-09-15"}
    assert "identity | TSNE(perplexity=30)" in " ".join(provenance.pipelines)
