"""Tests for the hECA calibration stage."""

import json

import numpy as np
import pandas as pd
import pytest

import carve.cluster as cluster
from benchmarks._heca_calibration import (
    GridRuleError,
    calibration_costs,
    fine_end_row,
    project_runtime,
    propose_grid,
    run_calibrate,
    scan_resolutions,
)
from benchmarks._heca_stages import StageOutputExists, scaled_neighbors, sweep_settings
from tests.benchmarks._helpers import small_heca_study


def _scan(rows):
    return pd.DataFrame(rows, columns=["setting", "resolution", "n_clusters"])


class TestProposeGrid:
    def test_spans_the_earliest_start_and_the_latest_end(self):
        # a reaches 5 clusters at 0.1, b only at 1.0: the grid starts at 0.1.
        # a passes 108 above 1.0, b stays within it to 10: it ends at 10.
        scan = _scan(
            [
                ("a", 0.01, 3), ("a", 0.1, 5), ("a", 1.0, 80), ("a", 10.0, 300),
                ("b", 0.01, 2), ("b", 0.1, 4), ("b", 1.0, 30), ("b", 10.0, 100),
            ]
        )
        assert propose_grid(scan, lower_target=5, upper_target=108) == (
            0.1, 0.14, 0.19, 0.27, 0.37, 0.52, 0.72, 1.0,
            1.4, 1.9, 2.7, 3.7, 5.2, 7.2, 10.0,
        )

    def test_a_count_that_dips_does_not_move_the_endpoints(self):
        # Leiden's count is not monotone in resolution. The rule reads the
        # smallest value reaching the lower target and the largest within the
        # upper one, wherever they fall.
        scan = _scan(
            [
                ("a", 0.01, 1), ("a", 0.1, 6), ("a", 0.3, 4),
                ("a", 1.0, 50), ("a", 3.0, 120), ("a", 10.0, 90),
            ]
        )
        grid = propose_grid(scan, lower_target=5, upper_target=108)
        assert grid[0] == pytest.approx(0.1)
        assert grid[-1] == pytest.approx(10.0)

    def test_a_setting_that_never_reaches_the_lower_target_raises(self):
        with pytest.raises(GridRuleError, match="'a' never reaches 5"):
            propose_grid(
                _scan([("a", 0.1, 2), ("a", 10.0, 4)]),
                lower_target=5,
                upper_target=108,
            )

    def test_a_setting_already_past_the_upper_target_raises(self):
        with pytest.raises(GridRuleError, match="'a' already exceeds 108"):
            propose_grid(
                _scan([("a", 0.001, 200), ("a", 1.0, 400)]),
                lower_target=5,
                upper_target=108,
            )

    def test_crossed_endpoints_raise(self):
        with pytest.raises(GridRuleError, match="cross"):
            propose_grid(
                _scan([("a", 0.1, 2), ("a", 1.0, 200)]),
                lower_target=5,
                upper_target=108,
            )

    def test_endpoints_too_close_for_distinct_values_raise(self):
        with pytest.raises(GridRuleError, match="duplicate"):
            propose_grid(
                _scan([("a", 1.0, 5), ("a", 1.05, 100)]),
                lower_target=5,
                upper_target=108,
            )


class TestProjection:
    COSTS = {
        "graph_train_s": 10.0,
        "graph_test_s": 4.0,
        "leiden_s": 2.0,
        "forest_fit_s": 30.0,
        "forest_predict_s": 1.0,
    }

    def test_one_setting_matches_the_hand_computation(self):
        out = project_runtime(
            {"a": self.COSTS},
            n_resolutions=15,
            n_resamples=100,
            n_train=600,
            n_test=400,
            n_jobs_options=(24, 128),
        )
        # Two training graphs, one complement graph, three Leiden runs (the
        # complement's scaled by its size), one forest fit and predict.
        per = 2 * 10.0 + 4.0 + (2 + 400 / 600) * 2.0 + 30.0 + 1.0
        assert out["per_resample_seconds"]["a"] == pytest.approx(per)
        assert out["core_hours"] == pytest.approx(15 * 100 * per / 3600)
        # 100 resamples in rounds of 24 workers take 5 rounds; of 128, one.
        assert out["wall_clock_hours"]["24"] == pytest.approx(15 * 5 * per / 3600)
        assert out["wall_clock_hours"]["128"] == pytest.approx(15 * 1 * per / 3600)

    def test_settings_add(self):
        kwargs = dict(n_resolutions=15, n_resamples=100, n_train=600, n_test=400)
        one = project_runtime({"a": self.COSTS}, **kwargs)
        two = project_runtime({"a": self.COSTS, "b": self.COSTS}, **kwargs)
        assert two["core_hours"] == pytest.approx(2 * one["core_hours"])


def test_calibration_costs_average_leiden_inside_the_grid():
    calibration = {
        "graph_costs": {
            "a": {
                "knn_train_s": 8.0,
                "graph_train_s": 10.0,
                "knn_test_s": 3.0,
                "graph_test_s": 4.0,
            }
        },
        "forest_fit_s": 30.0,
        "forest_predict_s": 1.0,
    }
    scan = pd.DataFrame(
        {
            "setting": ["a", "a", "a"],
            "resolution": [0.01, 0.1, 1.0],
            "seconds": [100.0, 2.0, 4.0],
        }
    )
    costs = calibration_costs(calibration, scan, resolutions=(0.1, 1.0))
    assert costs["a"] == {
        "graph_train_s": 10.0,
        "graph_test_s": 4.0,
        "leiden_s": pytest.approx(3.0),
        "forest_fit_s": 30.0,
        "forest_predict_s": 1.0,
    }


class TestForestCosts:
    CALIBRATION = {
        "graph_costs": {"a": {"graph_train_s": 10.0, "graph_test_s": 4.0}},
        "forest_fit_s": 30.0,
        "forest_predict_s": 1.0,
    }
    SCAN = pd.DataFrame({"setting": ["a"], "resolution": [1.0], "seconds": [2.0]})

    def test_the_middle_and_fine_forests_are_averaged(self):
        calibration = {
            **self.CALIBRATION,
            "forest_fit_fine_s": 90.0,
            "forest_predict_fine_s": 3.0,
        }
        costs = calibration_costs(calibration, self.SCAN, resolutions=(1.0,))
        assert costs["a"]["forest_fit_s"] == pytest.approx(60.0)
        assert costs["a"]["forest_predict_s"] == pytest.approx(2.0)

    def test_a_calibration_without_the_fine_forest_uses_the_middle_one(self):
        # A calibration.json written before the fine-end forest was timed.
        costs = calibration_costs(self.CALIBRATION, self.SCAN, resolutions=(1.0,))
        assert costs["a"]["forest_fit_s"] == pytest.approx(30.0)
        assert costs["a"]["forest_predict_s"] == pytest.approx(1.0)


class TestFineEndRow:
    def test_the_most_clusters_within_the_upper_target_across_settings(self):
        # b at 3.0 sits exactly on the target and beats a's 80; the counts
        # past it, a's 300 and b's 109, are out.
        scan = _scan(
            [
                ("a", 0.1, 5), ("a", 1.0, 80), ("a", 10.0, 300),
                ("b", 0.1, 4), ("b", 1.0, 100), ("b", 3.0, 108), ("b", 10.0, 109),
            ]
        )
        assert fine_end_row(scan, upper_target=108) == ("b", 3.0)

    def test_without_a_row_within_the_target_the_fewest_clusters(self):
        scan = _scan(
            [("a", 0.1, 200), ("a", 1.0, 400), ("b", 0.1, 150), ("b", 1.0, 300)]
        )
        assert fine_end_row(scan, upper_target=108) == ("b", 0.1)


def test_scan_matches_leiden_clustering():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.5, (60, 2)), rng.normal(5, 0.5, (60, 2))])
    graph = cluster.build_knn_graph(X, n_neighbors=10)
    ((resolution, labels, seconds),) = scan_resolutions(graph, (1.0,), seed=0)
    expected = cluster.LeidenClustering(n_neighbors=10, resolution=1.0, random_state=0)
    assert resolution == 1.0
    assert seconds > 0
    assert np.array_equal(labels, expected.fit(X).labels_)


class TestRunCalibrate:
    def test_writes_the_scan_the_costs_and_a_projection(self, tmp_path):
        study = small_heca_study(tmp_path / "data")
        run = tmp_path / "run"
        record = run_calibrate(run, study=study, scan=(0.5, 1.0, 2.0))

        scan = pd.read_csv(run / "calibration.csv")
        assert len(scan) == 2 * 3
        assert set(scan["setting"]) == {"leiden_15", "leiden_50"}
        assert set(scan["n_neighbors_scaled"]) == {
            scaled_neighbors(s, n_fit=record["n_train"], n_full=record["n_cells"])
            for s in sweep_settings(study)
        }
        assert scan["ari_organ"].between(-1, 1).all()

        assert record["n_train"] + record["n_test"] == record["n_cells"]
        assert record["lower_target"] == 2
        assert record["upper_target"] == 4
        assert (record["proposed_grid"] is None) != (record["grid_error"] is None)
        assert set(record["graph_costs"]) == {"leiden_15", "leiden_50"}
        assert record["forest_fit_s"] > 0
        assert record["forest_predict_s"] > 0
        # The second forest is fit on the scan's fine end, the most clusters
        # a worker's forest trains on.
        assert record["forest_fit_fine_s"] > 0
        assert record["forest_predict_fine_s"] > 0
        fine = record["forest_fine"]
        assert set(fine) == {"setting", "resolution", "n_clusters"}
        assert fine["n_clusters"] <= record["upper_target"]
        assert (fine["setting"], fine["resolution"]) == fine_end_row(
            scan, upper_target=record["upper_target"]
        )
        (scanned,) = scan.loc[
            (scan["setting"] == fine["setting"])
            & (scan["resolution"] == fine["resolution"]),
            "n_clusters",
        ]
        assert fine["n_clusters"] == scanned
        assert record["projection"]["core_hours"] > 0
        assert record["worker_peak_bytes"] >= record["baseline_rss_bytes"]
        on_disk = json.loads((run / "calibration.json").read_text())
        assert on_disk["n_cells"] == record["n_cells"]
        assert on_disk["forest_fine"] == fine

    def test_refuses_to_overwrite(self, tmp_path):
        study = small_heca_study(tmp_path / "data")
        run = tmp_path / "run"
        run_calibrate(run, study=study, scan=(1.0,))
        with pytest.raises(StageOutputExists):
            run_calibrate(run, study=study, scan=(1.0,))
