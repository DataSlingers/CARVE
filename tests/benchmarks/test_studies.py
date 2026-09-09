"""Tests for case-study compute."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans

from carve.cluster import SpectralClustering

from benchmarks._estimators import param_grids
from benchmarks._studies import (
    CVI_SWEEP_METRICS,
    STUDIES,
    _klein_loader,
    _levine_loader,
    carve_cache_path,
    cvi_sweep,
    fit_or_load_carve,
    load_study,
    resolve_scale,
    study_model_grids,
)
from benchmarks._types import EstimatorSpec, Study


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

    def test_gap_uses_tibshirani_not_argmax(self, blobs, monkeypatch):
        """The selected k must come from select_k's Tibshirani rule, not a
        plain argmax of the score column.

        select_k's own unit tests (test_cvi.py::TestSelectK) already
        fabricate a gap curve where Tibshirani and argmax disagree; this
        test is about cvi_sweep's integration with select_k rather than the
        rule itself, so the real gap-statistic computation is monkeypatched
        here with exactly that same fabricated curve rather than hoping the
        geometry of the ``blobs`` fixture happens to produce a divergence --
        an assertion like ``selected_k <= argmax_k`` is satisfied by a
        plain-argmax implementation too (equality is not "less than"), so it
        can never fail even when cvi_sweep silently drops the Tibshirani
        rule.
        """
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (3, 4, 5, 6))

        # Gap rises to k=5 but k=4 is already within s_5 of Gap(5), so
        # Tibshirani's rule stops at k=4 where a plain argmax would say 5 --
        # the same fixture test_cvi.py::TestSelectK uses to pin select_k
        # itself.
        gap_by_k = {3: (0.10, 0.02), 4: (0.50, 0.02), 5: (0.55, 0.10), 6: (0.30, 0.02)}

        from benchmarks import _studies

        real_calculate_cvi = _studies.calculate_cvi

        def fake_calculate_cvi(X, labels, metric, *, spec, random_state):
            if metric == "gap":
                return gap_by_k[int(np.unique(labels).size)]
            return real_calculate_cvi(X, labels, metric, spec=spec, random_state=random_state)

        monkeypatch.setattr(_studies, "calculate_cvi", fake_calculate_cvi)

        curves, best = cvi_sweep(X, y, model_grids=grids, candidate_k=(3, 4, 5, 6))
        gap_curve = curves[curves["metric"] == "gap"].sort_values("k")
        argmax_k = int(gap_curve.loc[gap_curve["score"].idxmax(), "k"])
        selected_k = int(best.loc[best["metric"] == "gap", "k"].iloc[0])

        assert argmax_k == 5
        assert selected_k == 4
        assert selected_k != argmax_k

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


class TestStudyModelGrids:
    """Pins the case-study estimators to the manuscript.

    Manuscript line 624/1501: Klein sweeps Ward agglomerative clustering and
    spectral clustering with self-tuning affinity. Manuscript line 1523:
    Levine sweeps KMeans and spectral.
    """

    def test_klein_grid_is_ward_agglomerative_and_spectral_not_kmeans(self):
        grid = study_model_grids(STUDIES["klein"])
        classes = {estimator_cls for estimator_cls, _ in grid}
        assert classes == {AgglomerativeClustering, SpectralClustering}
        assert KMeans not in classes

        agglo_params = dict(
            next(params for cls, params in grid if cls is AgglomerativeClustering)
        )
        assert agglo_params["linkage"] == ["ward"]

    def test_levine_grid_is_kmeans_and_spectral_not_agglomerative(self):
        grid = study_model_grids(STUDIES["levine32"])
        classes = {estimator_cls for estimator_cls, _ in grid}
        assert classes == {KMeans, SpectralClustering}
        assert AgglomerativeClustering not in classes

    def test_grid_varies_with_study_estimator_not_study_name(self):
        """Two Study objects differing only in estimator must produce
        different grids. This is stronger than checking the two live STUDIES
        entries, which would still pass if someone re-hardcoded a branch on
        study.name instead of reading study.estimator.
        """

        def _unused_loader(subsample):
            raise AssertionError("the loader must not run for this test")

        ward_study = Study(
            name="probe",
            loader=_unused_loader,
            estimator=EstimatorSpec(name="agglomerative"),
            candidate_k=(2, 3),
            scales={"dev": 100},
            default_scale="dev",
        )
        kmeans_study = Study(
            name="probe",
            loader=_unused_loader,
            estimator=EstimatorSpec(name="kmeans"),
            candidate_k=(2, 3),
            scales={"dev": 100},
            default_scale="dev",
        )

        ward_classes = {cls for cls, _ in study_model_grids(ward_study)}
        kmeans_classes = {cls for cls, _ in study_model_grids(kmeans_study)}

        assert ward_classes == {AgglomerativeClustering, SpectralClustering}
        assert kmeans_classes == {KMeans, SpectralClustering}
        assert ward_classes != kmeans_classes


class TestLoaderSubsampling:
    """Pins the loader calls to the manuscript's reported sample sizes.

    Klein (manuscript line 606): 1,358 cells, a 0.5 subsample of the
    2,717-cell preprocessed set. Levine (manuscript line 627): a stratified
    subsample of 5,000 cells. Real data is never loaded here; the dataset
    loader is patched out and the call arguments are inspected instead.
    """

    def test_klein_loader_requests_a_half_subsample(self, monkeypatch):
        calls = {}

        def fake_load_klein(**kwargs):
            calls.update(kwargs)
            return np.zeros((1, 1)), pd.Series(["a"]), {}

        monkeypatch.setattr("benchmarks.datasets.load_klein", fake_load_klein)
        _klein_loader(0.5)
        assert calls["subsample"] == 0.5

    def test_levine_loader_requests_five_thousand_cells(self, monkeypatch):
        calls = {}

        def fake_load_levine32(**kwargs):
            calls.update(kwargs)
            return np.zeros((1, 1)), pd.Series(["a"]), {}

        monkeypatch.setattr("benchmarks.datasets.load_levine32", fake_load_levine32)
        _levine_loader(5000)
        assert calls["subsample"] == 5000


def _study(**kw):
    base = dict(
        name="demo",
        loader=lambda subsample: (subsample, None, {"subsample": subsample}),
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=(2, 3),
        scales={"dev": 100, "publication": None},
        default_scale="dev",
    )
    base.update(kw)
    return Study(**base)


class TestScaleResolution:
    def test_default_scale_is_used_when_none_is_given(self):
        assert resolve_scale(_study(), None) == 100

    def test_named_scale_wins(self):
        assert resolve_scale(_study(), "publication") is None

    def test_unknown_scale_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="dev"):
            resolve_scale(_study(), "enormous")

    def test_load_study_passes_the_resolved_size_to_the_loader(self):
        # The loader echoes its argument, so this pins that configuration
        # travels from STUDIES rather than being re-derived at the call site.
        X, _, meta = load_study(_study(), scale="dev")
        assert X == 100
        assert meta["subsample"] == 100


class TestCarveCachePath:
    def test_scale_is_part_of_the_filename(self, tmp_path):
        dev = carve_cache_path(_study(), scale="dev", root=tmp_path)
        pub = carve_cache_path(_study(), scale="publication", root=tmp_path)
        assert dev != pub
        assert "dev" in dev.name
        assert "publication" in pub.name
        assert dev.suffix == ".carve"

    def test_study_name_is_part_of_the_filename(self, tmp_path):
        a = carve_cache_path(_study(name="alpha"), scale="dev", root=tmp_path)
        b = carve_cache_path(_study(name="beta"), scale="dev", root=tmp_path)
        assert a != b


class TestRegisteredStudiesCarryScales:
    def test_every_study_declares_its_default_scale(self):
        for study in STUDIES.values():
            assert study.default_scale in study.scales

    def test_klein_publication_scale_is_the_published_half_subsample(self):
        assert STUDIES["klein"].scales["publication"] == 0.5

    def test_levine_publication_scale_is_five_thousand(self):
        assert STUDIES["levine32"].scales["publication"] == 5000
