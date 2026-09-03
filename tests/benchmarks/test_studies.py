"""Tests for case-study compute."""

import numpy as np
import pandas as pd
import pytest

from benchmarks._estimators import param_grids
from benchmarks._studies import (
    CVI_SWEEP_METRICS,
    STUDIES,
    cvi_sweep,
    fit_or_load_carve,
)
from benchmarks._types import EstimatorSpec


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=0.0, scale=0.4, size=(40, 3)),
        rng.normal(loc=6.0, scale=0.4, size=(40, 3)),
        rng.normal(loc=(0.0, 6.0, 0.0), scale=0.4, size=(40, 3)),
    ])
    y = np.repeat(["a", "b", "c"], 40)
    return X, y


class TestCviSweep:
    def test_returns_curves_and_best(self, blobs):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3, 4))
        curves, best = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3, 4))
        assert set(curves.columns) == {"metric", "model", "k", "score", "ari"}
        assert set(best.columns) == {"metric", "model", "k", "score", "ari"}

    def test_one_curve_row_per_metric_model_and_k(self, blobs):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3, 4))
        curves, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3, 4))
        assert len(curves) == len(CVI_SWEEP_METRICS) * 3

    def test_one_best_row_per_metric(self, blobs):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3, 4))
        _, best = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3, 4))
        assert len(best) == len(CVI_SWEEP_METRICS)
        assert set(best["metric"]) == set(CVI_SWEEP_METRICS)

    def test_gap_uses_tibshirani_not_argmax(self, blobs):
        """The selected k must come from select_k, not from the score column."""
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3, 4))
        curves, best = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3, 4))
        gap_curve = curves[curves["metric"] == "gap"].sort_values("k")
        argmax_k = int(gap_curve.loc[gap_curve["score"].idxmax(), "k"])
        selected_k = int(best.loc[best["metric"] == "gap", "k"].iloc[0])
        assert selected_k <= argmax_k

    def test_ari_is_recorded_against_the_true_labels(self, blobs):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3, 4))
        curves, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3, 4))
        at_three = curves[(curves["k"] == 3) & (curves["metric"] == "silhouette")]
        assert at_three["ari"].iloc[0] > 0.9

    def test_works_without_labels(self, blobs):
        X, _ = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        curves, _ = cvi_sweep(X, None, model_grids=grids, candidate_k=(2, 3))
        assert curves["ari"].isna().all()

    def test_is_deterministic(self, blobs):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        first, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3), random_state=1)
        second, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3), random_state=1)
        pd.testing.assert_frame_equal(first, second)


class TestFitOrLoadCarve:
    def test_fits_and_writes_the_cache_on_first_call(self, blobs, tmp_path):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        carve = fit_or_load_carve(
            X, y, cache_path=cache, model_grids=grids, n_resamples=3
        )
        assert cache.exists()
        assert carve.estimator_results_ is not None

    def test_second_call_loads_the_cache(self, blobs, tmp_path):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        mtime = cache.stat().st_mtime_ns
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        assert cache.stat().st_mtime_ns == mtime

    def test_loaded_object_has_the_data_matrix_restored(self, blobs, tmp_path):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        carve = fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        np.testing.assert_array_equal(carve.X_, X)

    def test_force_refits_even_with_a_cache(self, blobs, tmp_path):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        mtime = cache.stat().st_mtime_ns
        fit_or_load_carve(
            X, y, cache_path=cache, model_grids=grids, n_resamples=3, force=True
        )
        assert cache.stat().st_mtime_ns != mtime


class TestStudies:
    def test_both_case_studies_are_registered(self):
        assert set(STUDIES) == {"klein", "levine32"}

    def test_each_study_has_a_loader_and_candidate_k(self):
        for study in STUDIES.values():
            assert callable(study.loader)
            assert len(study.candidate_k) > 0

    def test_klein_sweeps_k_two_through_ten(self):
        assert STUDIES["klein"].candidate_k == tuple(range(2, 11))

    def test_levine_sweeps_k_seven_through_seventeen(self):
        assert STUDIES["levine32"].candidate_k == tuple(range(7, 18))
