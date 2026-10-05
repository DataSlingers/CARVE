"""Tests for the timing instrumentation."""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

import carve.cluster as cluster
from benchmarks._estimators import resolution_grids
from benchmarks._timing import (
    TIMING_DIR_ENV,
    LeidenClustering,
    TimedRandomForestClassifier,
    graph_timers,
    instrumented_grids,
    read_timings,
    timed_forest,
    timing_directory,
)
from benchmarks._types import EstimatorSpec
from carve import CARVE
from carve._utils import default_generalizability_classifier


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    labels = rng.integers(0, 3, 240)
    return centers[labels] + rng.normal(0, 0.6, (240, 2))


def _grids():
    spec = EstimatorSpec(name="leiden", params=(("n_neighbors", 10),))
    return resolution_grids(spec, (0.5, 1.0))


def _fit(X, grids, classifier, *, n_jobs):
    return CARVE(
        estimator_param_grids=grids,
        n_resamples=4,
        n_jobs=n_jobs,
        random_state=0,
        classifier=classifier,
    ).fit(X)


def test_the_subclass_keeps_its_parents_name():
    # CARVE labels configurations by the class's __name__.
    assert LeidenClustering.__name__ == cluster.LeidenClustering.__name__
    assert issubclass(LeidenClustering, cluster.LeidenClustering)


def test_without_the_variable_the_subclass_is_its_parent(blobs, monkeypatch):
    monkeypatch.delenv(TIMING_DIR_ENV, raising=False)
    timed = LeidenClustering(n_neighbors=10, resolution=1.0, random_state=0)
    plain = cluster.LeidenClustering(n_neighbors=10, resolution=1.0, random_state=0)
    assert np.array_equal(timed.fit(blobs).labels_, plain.fit(blobs).labels_)


def test_a_leiden_fit_writes_one_row_whose_parts_sum_to_the_fit(tmp_path, blobs):
    with timing_directory(tmp_path):
        estimator = LeidenClustering(n_neighbors=10, resolution=1.0, random_state=0)
        estimator.fit(blobs)
    leiden = read_timings(tmp_path)["leiden"]
    assert len(leiden) == 1
    row = leiden.iloc[0]
    assert row["n_neighbors"] == 10
    assert row["n_samples"] == 240
    assert row["n_clusters"] == estimator.n_clusters_
    assert row["knn_s"] > 0
    assert row["graph_s"] > 0
    assert row["leiden_s"] > 0
    assert row["knn_s"] + row["graph_s"] + row["leiden_s"] == pytest.approx(
        row["fit_s"]
    )


def test_graph_timers_time_both_parts_and_restore_carve_cluster(blobs):
    search, build = cluster.kneighbors_graph, cluster.build_knn_graph
    with graph_timers() as times:
        cluster.build_knn_graph(blobs, n_neighbors=5)
    assert cluster.kneighbors_graph is search
    assert cluster.build_knn_graph is build
    assert 0 < times.knn_s < times.build_s


def test_timing_directory_sets_and_restores_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(TIMING_DIR_ENV, "before")
    with timing_directory(tmp_path / "timings") as path:
        assert os.environ[TIMING_DIR_ENV] == str(path)
        assert path.is_dir()
    assert os.environ[TIMING_DIR_ENV] == "before"


def test_timed_forest_carries_the_default_forests_parameters():
    plain = default_generalizability_classifier(
        classifier=None, n_features=50, n_trees=100, random_state=None, n_jobs=1
    )
    timed = timed_forest(n_features=50, n_trees=100)
    assert isinstance(timed, TimedRandomForestClassifier)
    assert timed.get_params() == plain.get_params()


def test_instrumented_grids_swap_only_the_leiden_class():
    grids = _grids() + [(KMeans, {"n_clusters": [2]})]
    swapped = instrumented_grids(grids)
    assert swapped[0][0] is LeidenClustering
    assert swapped[1][0] is KMeans
    assert swapped[0][1] == grids[0][1]


def test_an_instrumented_fit_matches_a_plain_one_and_times_inside_workers(
    tmp_path, blobs
):
    # The plain fit runs first and leaves joblib's worker pool alive without
    # the variable; timing_directory must replace that pool for any worker
    # row to appear.
    plain = _fit(blobs, _grids(), None, n_jobs=2)
    with timing_directory(tmp_path):
        timed = _fit(
            blobs,
            instrumented_grids(_grids()),
            timed_forest(n_features=blobs.shape[1], n_trees=100),
            n_jobs=2,
        )
    pd.testing.assert_frame_equal(plain.estimator_results_, timed.estimator_results_)

    tables = read_timings(tmp_path)
    leiden, forest = tables["leiden"], tables["forest"]
    # Two resolutions x four resamples x three subsets (both training
    # splits and the complement).
    assert len(leiden) == 2 * 4 * 3
    assert int((forest["call"] == "fit").sum()) == 2 * 4
    assert set(leiden["pid"]) - {os.getpid()}, "no row came from a worker"


def test_read_timings_of_an_empty_directory_gives_empty_frames(tmp_path):
    tables = read_timings(tmp_path)
    assert tables["leiden"].empty
    assert tables["forest"].empty
