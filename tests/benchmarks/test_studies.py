"""Tests for case-study compute."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans

from benchmarks._estimators import param_grids, resolution_grids
from benchmarks._preprocessing import PREPROCESSOR_DEFAULTS, resolve_preprocessing
from benchmarks._studies import (
    CVI_SWEEP_METRICS,
    STUDIES,
    _cusanovich_loader,
    _heca_loader,
    _klein_loader,
    _levine_loader,
    carve_cache_path,
    cvi_sweep,
    fit_or_load_carve,
    load_study,
    resolve_scale,
    study_model_grids,
    study_resolution_grids,
    study_scaling_sweep,
)
from benchmarks._types import EstimatorSpec, PreprocessingSpec, Study
from carve._pipeline import allocate_pipelines
from carve.cluster import LeidenClustering, LouvainClustering, SpectralClustering
from tests.benchmarks._helpers import make_carve_spy


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
    def test_gap_reference_fits_use_the_cells_own_linkage(self, blobs, monkeypatch):
        """A grid sweeping linkage over ward and single produces two model
        labels, and the gap statistic must refit its reference datasets with
        the same linkage the cell used. Mapping the class alone back to a
        spec sent every AgglomerativeClustering cell to the ward defaults.
        """
        import benchmarks._cvi as cvi_module

        X, y = blobs
        seen = set()
        original = cvi_module.build_estimator

        def spy(spec, **kwargs):
            seen.add(spec.name)
            return original(spec, **kwargs)

        monkeypatch.setattr(cvi_module, "build_estimator", spy)
        grids = [
            (AgglomerativeClustering, {"n_clusters": [2, 3], "linkage": ["ward", "single"]})
        ]
        curves, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3), n_jobs=1)

        assert set(curves["model"]) == {
            "AgglomerativeClustering (linkage=ward)",
            "AgglomerativeClustering (linkage=single)",
        }
        assert seen == {"agglomerative", "agglomerative_single"}

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


class TestFitOrLoadCarveFingerprint:
    """A cached fit is only valid for the X it was fit on.

    fit_or_load_carve restores X_ onto whatever it loads, so a fit taken on
    one embedding served against another would misreport silently. The hECA
    development embedding changes under the same scale name (subsample-first
    now, pooled once that cache exists), which is exactly that case.
    """

    def test_loading_against_a_different_x_raises(self, blobs, tmp_path):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        with pytest.raises(ValueError, match="force=True") as excinfo:
            fit_or_load_carve(
                X + 1.0, y, cache_path=cache, model_grids=grids, n_resamples=3
            )
        assert str(cache) in str(excinfo.value)

    def test_force_refits_and_records_the_new_x(self, blobs, tmp_path):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        fit_or_load_carve(
            X + 1.0, y, cache_path=cache, model_grids=grids, n_resamples=3, force=True
        )
        mtime = cache.stat().st_mtime_ns
        fit_or_load_carve(X + 1.0, y, cache_path=cache, model_grids=grids, n_resamples=3)
        assert cache.stat().st_mtime_ns == mtime

    # CARVE.load here also hits joblib's shape-deprecation warning (the one
    # pyproject.toml ignores for CARVE.save/load), but pytest.warns always
    # re-emits an unmatched warning under a fixed module of "warnings", not
    # the issuing frame, so that module-scoped ignore can never match here.
    @pytest.mark.filterwarnings(
        "ignore:Setting the shape on a NumPy array:DeprecationWarning"
    )
    def test_a_cache_without_a_fingerprint_warns_and_loads(self, blobs, tmp_path):
        # Fits cached before the fingerprint existed have no record to
        # check against; refusing them would throw away hours of compute,
        # and silently trusting them is the failure this guards against.
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        cache = tmp_path / "demo.carve"
        fit_or_load_carve(X, y, cache_path=cache, model_grids=grids, n_resamples=3)
        for sidecar in tmp_path.glob("demo.carve.*"):
            sidecar.unlink()
        with pytest.warns(UserWarning, match="fingerprint"):
            carve = fit_or_load_carve(
                X, y, cache_path=cache, model_grids=grids, n_resamples=3
            )
        assert carve.estimator_results_ is not None


class TestConsensusAnchorsForwarding:
    def test_fit_or_load_carve_forwards_consensus_anchors(
        self, blobs, tmp_path, monkeypatch
    ):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)

        fit_or_load_carve(
            X,
            y,
            cache_path=tmp_path / "demo.carve",
            model_grids=grids,
            n_resamples=3,
            consensus_anchors=123,
        )

        assert spy.captured_kwargs["consensus_anchors"] == 123

    def test_fit_or_load_carve_omits_consensus_anchors_when_none(
        self, blobs, tmp_path, monkeypatch
    ):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)

        fit_or_load_carve(
            X,
            y,
            cache_path=tmp_path / "demo.carve",
            model_grids=grids,
            n_resamples=3,
        )

        assert "consensus_anchors" not in spy.captured_kwargs

    def test_study_scaling_sweep_forwards_consensus_anchors(self, blobs, monkeypatch):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)

        frame = study_scaling_sweep(
            X, y, sizes=[len(X)], model_grids=grids, n_resamples=3, consensus_anchors=77
        )

        assert spy.captured_kwargs["consensus_anchors"] == 77
        assert list(frame["n"]) == [len(X)]


_TSNE_SPEC = PreprocessingSpec(
    normalization=(("identity", {}),),
    dim_reduction=(("identity", {}), ("tsne", {"perplexity": [5]})),
)


class TestPreprocessingForwarding:
    def test_forwards_the_preprocessing_arguments(self, blobs, tmp_path, monkeypatch):
        X, y = blobs
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)
        normalization = [("normalization sentinel",)]
        dim_reduction = [("dim_reduction sentinel",)]

        fit_or_load_carve(
            X,
            y,
            cache_path=tmp_path / "demo.carve",
            model_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
            n_resamples=3,
            randomize_preprocessing=True,
            normalization_options=normalization,
            dim_reduction_options=dim_reduction,
        )

        assert spy.captured_kwargs["normalization_options"] is normalization
        assert spy.captured_kwargs["dim_reduction_options"] is dim_reduction
        assert spy.captured_fit_kwargs["randomize_preprocessing"] is True

    def test_omits_the_preprocessing_arguments_when_unset(
        self, blobs, tmp_path, monkeypatch
    ):
        X, y = blobs
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)

        fit_or_load_carve(
            X,
            y,
            cache_path=tmp_path / "demo.carve",
            model_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
            n_resamples=3,
        )

        assert "normalization_options" not in spy.captured_kwargs
        assert "dim_reduction_options" not in spy.captured_kwargs
        assert "randomize_preprocessing" not in spy.captured_fit_kwargs

    def test_options_without_randomize_preprocessing_raise(self, blobs, tmp_path):
        # CARVE ignores the option lists on a non-randomized fit, so passing
        # them there is a misconfiguration, not a no-op.
        X, y = blobs
        with pytest.raises(ValueError, match="randomize_preprocessing=True"):
            fit_or_load_carve(
                X,
                y,
                cache_path=tmp_path / "demo.carve",
                model_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
                n_resamples=3,
                dim_reduction_options=[],
            )
        assert not (tmp_path / "demo.carve").exists()

    def test_a_randomized_fit_survives_the_cache(self, blobs, tmp_path):
        # resolve_preprocessing binds t-SNE's defaults with functools.partial;
        # the cached fit must pickle and reload it with the per-pipeline
        # table and the pipeline registry intact.
        X, y = blobs
        kwargs = dict(
            cache_path=tmp_path / "demo.carve",
            model_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
            n_resamples=4,
            randomize_preprocessing=True,
            **resolve_preprocessing(_TSNE_SPEC),
        )
        fitted = fit_or_load_carve(X, y, **kwargs)
        mtime = kwargs["cache_path"].stat().st_mtime_ns
        loaded = fit_or_load_carve(X, y, **kwargs)

        assert kwargs["cache_path"].stat().st_mtime_ns == mtime
        pd.testing.assert_frame_equal(
            loaded.preprocessing_results_, fitted.preprocessing_results_
        )
        assert set(loaded.preprocessing_pipelines_) == {
            "identity | identity",
            "identity | TSNE(perplexity=5)",
        }


class TestDenseEstimatorGuard:
    """cvi_sweep, fit_or_load_carve, and study_scaling_sweep all bring X and
    model_grids together before handing them to a real estimator, which is
    why the O(n^2)-estimator guard lives in each of them rather than in
    study_model_grids -- which never sees n, and which a caller assembling
    model_grids by hand would not even go through.
    """

    def test_cvi_sweep_rejects_spectral_at_large_n(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6000, 3))
        y = np.repeat(["a", "b"], 3000)
        grids = param_grids(EstimatorSpec(name="spectral"), (2, 3))
        with pytest.raises(ValueError, match="SpectralClustering"):
            cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3))

    def test_fit_or_load_carve_rejects_agglomerative_at_large_n(self, tmp_path):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6000, 3))
        y = np.repeat(["a", "b"], 3000)
        grids = param_grids(EstimatorSpec(name="agglomerative"), (2, 3))
        with pytest.raises(ValueError, match="AgglomerativeClustering"):
            fit_or_load_carve(
                X,
                y,
                cache_path=tmp_path / "demo.carve",
                model_grids=grids,
                n_resamples=3,
            )

    def test_study_scaling_sweep_rejects_spectral_at_a_large_rung(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6000, 2))
        grids = param_grids(EstimatorSpec(name="spectral"), (2, 3))
        with pytest.raises(ValueError, match="SpectralClustering"):
            study_scaling_sweep(X, None, sizes=[6000], model_grids=grids, n_resamples=3)

    def test_error_names_the_resolved_n(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6000, 3))
        grids = param_grids(EstimatorSpec(name="spectral"), (2, 3))
        with pytest.raises(ValueError, match="6000"):
            cvi_sweep(X, None, model_grids=grids, candidate_k=(2, 3))

    def test_guard_does_not_block_a_safe_n(self):
        # At or below the safe boundary this must not become a blanket ban
        # on spectral or agglomerative.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = np.repeat(["a", "b"], 100)
        grids = param_grids(EstimatorSpec(name="spectral"), (2, 3))
        curves, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3))
        assert not curves.empty

    def test_guard_ignores_estimators_that_are_not_dense(self):
        # KMeans has no quadratic-memory affinity/distance matrix, so a large
        # n paired only with KMeans must not raise.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6000, 3))
        y = np.repeat(["a", "b"], 3000)
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        curves, _ = cvi_sweep(X, y, model_grids=grids, candidate_k=(2, 3))
        assert not curves.empty


class TestStudies:
    def test_every_case_study_is_registered(self):
        assert set(STUDIES) == {"klein", "levine32", "cusanovich", "heca"}

    def test_each_study_has_a_loader_and_a_sweep(self):
        for study in STUDIES.values():
            assert callable(study.loader)
            assert study.candidate_k or study.resolutions

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

    def test_grid_reads_partners_from_the_study(self):
        # The partner estimators are declared on the Study, not chosen by a
        # branch on study.name inside study_model_grids.
        def _unused_loader(subsample):
            raise AssertionError("the loader must not run for this test")

        study = Study(
            name="probe",
            loader=_unused_loader,
            estimator=EstimatorSpec(name="agglomerative"),
            candidate_k=(2, 3),
            scales={"dev": 100},
            default_scale="dev",
            partners=(
                EstimatorSpec(name="kmeans"),
                EstimatorSpec(name="agglomerative_single"),
            ),
        )
        grid = study_model_grids(study)
        assert [(cls, params.get("linkage")) for cls, params in grid] == [
            (AgglomerativeClustering, ["ward"]),
            (KMeans, None),
            (AgglomerativeClustering, ["single"]),
        ]

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

    def test_heca_loader_draws_the_subsample_before_embedding(self, monkeypatch):
        # The pooled hECA embedding holds tens of GB; a development-scale
        # load only fits on a laptop if the rows are drawn from the organ
        # files first. The loader passes the flag at every scale; load_heca
        # ignores it when subsample is None (publication) and whenever the
        # pooled cache is present.
        calls = {}

        def fake_load_heca(**kwargs):
            calls.update(kwargs)
            return np.zeros((1, 1)), pd.Series(["a"]), {}

        monkeypatch.setattr("benchmarks.datasets.load_heca", fake_load_heca)
        _heca_loader(25_000)
        assert calls["subsample"] == 25_000
        assert calls["subsample_before_embedding"] is True

    def test_cusanovich_loader_references_the_source_clusters(self, monkeypatch):
        # The reference is the source's 30 clusters, which assign every cell,
        # so the Unknown-labeled cells stay: drop_unknown is left at False.
        calls = {}

        def fake_load_cusanovich(**kwargs):
            calls.update(kwargs)
            return np.zeros((1, 1)), pd.Series(["a"]), {}

        monkeypatch.setattr(
            "benchmarks.datasets.load_cusanovich", fake_load_cusanovich
        )
        _cusanovich_loader(1500)
        assert calls["subsample"] == 1500
        assert calls["label_column"] == "cluster"
        assert calls.get("drop_unknown", False) is False

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

    def test_editing_a_scales_resolved_size_changes_the_path(self, tmp_path):
        # Same study name, same scale name ("dev"), different resolved size
        # -- as if STUDIES[...].scales["dev"] were edited from 25,000 to
        # 50,000. Keying only on the scale name would leave this path
        # unchanged and fit_or_load_carve would silently serve the fit taken
        # at the old size.
        small = _study(scales={"dev": 100, "publication": None})
        big = _study(scales={"dev": 50_000, "publication": None})
        a = carve_cache_path(small, scale="dev", root=tmp_path)
        b = carve_cache_path(big, scale="dev", root=tmp_path)
        assert a != b

    def test_the_full_data_none_scale_gets_a_stable_path(self, tmp_path):
        # None (full data) has no natural filename spelling; the hash must
        # still be deterministic across calls.
        a = carve_cache_path(_study(), scale="publication", root=tmp_path)
        b = carve_cache_path(_study(), scale="publication", root=tmp_path)
        assert a == b

    def test_default_runs_keep_their_existing_filenames(self, tmp_path):
        # The Klein and Levine publication caches are hours of compute each
        # and were written before run keys existed, so their default runs
        # must resolve to the same names as before, byte for byte.
        klein = carve_cache_path(STUDIES["klein"], root=tmp_path)
        levine = carve_cache_path(STUDIES["levine32"], root=tmp_path)
        heca = carve_cache_path(STUDIES["heca"], root=tmp_path)
        assert klein.name == "carve_klein_publication_1b390cd5.carve"
        assert levine.name == "carve_levine32_publication_f8237d89.carve"
        assert heca.name == "carve_heca_dev_8314e95d.carve"

    def test_passing_the_default_run_explicitly_changes_nothing(self, tmp_path):
        study = _study()
        explicit = carve_cache_path(
            study,
            scale="dev",
            root=tmp_path,
            model_grids=study_model_grids(study),
            n_resamples=100,
            preprocessing=None,
        )
        assert explicit == carve_cache_path(study, scale="dev", root=tmp_path)

    @pytest.mark.parametrize(
        "change",
        [
            {"model_grids": resolution_grids(EstimatorSpec(name="leiden"), (0.5, 1.0))},
            {"preprocessing": _TSNE_SPEC},
            {"n_resamples": 150},
        ],
        ids=["resolution-grid", "preprocessing", "n_resamples"],
    )
    def test_any_other_run_gets_its_own_filename(self, tmp_path, change):
        default = carve_cache_path(_study(), scale="dev", root=tmp_path)
        path = carve_cache_path(_study(), scale="dev", root=tmp_path, **change)
        assert path != default
        assert path.name.startswith(default.stem + "_")
        assert path.suffix == ".carve"

    def test_different_preprocessing_gets_a_different_filename(self, tmp_path):
        other = PreprocessingSpec(
            normalization=(("identity", {}),),
            dim_reduction=(("identity", {}), ("tsne", {"perplexity": [15]})),
        )
        a = carve_cache_path(_study(), scale="dev", root=tmp_path, preprocessing=_TSNE_SPEC)
        b = carve_cache_path(_study(), scale="dev", root=tmp_path, preprocessing=other)
        assert a != b

    def test_editing_the_bound_defaults_changes_the_filename(self, tmp_path, monkeypatch):
        before = carve_cache_path(
            _study(), scale="dev", root=tmp_path, preprocessing=_TSNE_SPEC
        )
        monkeypatch.setitem(PREPROCESSOR_DEFAULTS, "tsne", {"n_components": 3})
        after = carve_cache_path(
            _study(), scale="dev", root=tmp_path, preprocessing=_TSNE_SPEC
        )
        assert before != after

    def test_a_study_without_a_k_grid_must_name_its_grid(self, tmp_path):
        study = _study(
            estimator=EstimatorSpec(name="leiden"), candidate_k=(), resolutions=(0.5, 1.0)
        )
        with pytest.raises(ValueError, match="model_grids"):
            carve_cache_path(study, scale="dev", root=tmp_path)


class TestRegisteredStudiesCarryScales:
    def test_every_study_declares_its_default_scale(self):
        for study in STUDIES.values():
            assert study.default_scale in study.scales

    def test_klein_publication_scale_is_the_published_half_subsample(self):
        assert STUDIES["klein"].scales["publication"] == 0.5

    def test_levine_publication_scale_is_five_thousand(self):
        assert STUDIES["levine32"].scales["publication"] == 5000


class TestNewStudies:
    def test_cusanovich_sweeps_louvain_resolution_only(self):
        # The source clustered its t-SNE with Seurat's Louvain, so the study
        # sweeps Louvain resolution and nothing k-based.
        study = STUDIES["cusanovich"]
        assert study.estimator == EstimatorSpec(name="louvain")
        assert study.candidate_k == ()
        assert study.partners == ()
        with pytest.raises(ValueError, match="study_resolution_grids"):
            study_model_grids(study)
        ((cls, grid),) = study_resolution_grids(study)
        assert cls is LouvainClustering
        assert grid["resolution"] == pytest.approx(
            [round(0.2 * i, 1) for i in range(1, 16)]
        )
        assert grid["n_neighbors"] == [15]

    def test_cusanovich_randomizes_over_the_lsi_and_three_tsne_perplexities(self):
        study = STUDIES["cusanovich"]
        assert study.n_resamples == 50
        assert study.preprocessing == PreprocessingSpec(
            normalization=(("identity", {}),),
            dim_reduction=(
                ("identity", {}),
                ("tsne", {"perplexity": [30]}),
                ("tsne", {"perplexity": [100]}),
                ("tsne", {"perplexity": [300]}),
            ),
        )

    def test_cusanovich_offers_the_source_perplexity(self):
        # The source's operating point is read from the t-SNE pipeline at the
        # source's own perplexity, so the study must offer that value.
        from benchmarks._cusanovich_compare import SOURCE_TSNE_PERPLEXITY

        offered = [
            grid["perplexity"]
            for key, grid in STUDIES["cusanovich"].preprocessing.dim_reduction
            if key == "tsne"
        ]
        assert [SOURCE_TSNE_PERPLEXITY] in offered

    def test_cusanovich_balances_resamples_across_its_four_pipelines(self):
        # Stratified allocation is over options, and each perplexity is its
        # own option, so 50 resamples split 12 or 13 to each of the LSI and
        # the three t-SNE pipelines instead of t-SNE's share being drawn at
        # random between perplexities.
        study = STUDIES["cusanovich"]
        options = resolve_preprocessing(study.preprocessing)
        pipelines = allocate_pipelines(
            options["normalization_options"],
            options["dim_reduction_options"],
            n_resamples=study.n_resamples,
            random_state=42,
        )
        counts = Counter(pipeline.label for pipeline in pipelines)
        assert set(counts) == {
            "identity | identity",
            "identity | TSNE(perplexity=30)",
            "identity | TSNE(perplexity=100)",
            "identity | TSNE(perplexity=300)",
        }
        assert sorted(counts.values()) == [12, 12, 13, 13]

    def test_cusanovich_pipelines_fit_the_pipeline_palette(self):
        # The per-pipeline panel samples PIPELINE_COLORS once per pipeline, so
        # a study with more pipelines than colors draws two of them alike.
        from benchmarks._theme import PIPELINE_COLORS

        study = STUDIES["cusanovich"]
        options = resolve_preprocessing(study.preprocessing)
        pipelines = allocate_pipelines(
            options["normalization_options"],
            options["dim_reduction_options"],
            n_resamples=study.n_resamples,
            random_state=42,
        )
        assert len({pipeline.label for pipeline in pipelines}) <= len(PIPELINE_COLORS)

    def test_cusanovich_cache_never_resolves_to_the_invalid_pre_fix_cache(
        self, tmp_path
    ):
        # carve_cusanovich_dev_7841fb1f.carve was fit before the cell-order
        # fix (1dea71d) and must never be served again.
        study = STUDIES["cusanovich"]
        path = carve_cache_path(
            study,
            root=tmp_path,
            model_grids=study_resolution_grids(study),
            n_resamples=study.n_resamples,
            preprocessing=study.preprocessing,
        )
        assert path.name.startswith("carve_cusanovich_dev_7841fb1f_")
        assert path.name != "carve_cusanovich_dev_7841fb1f.carve"

    def test_heca_sweeps_four_through_fifteen(self):
        # Five pooled organs; the sweep starts just below that count.
        assert STUDIES["heca"].candidate_k == tuple(range(4, 16))

    def test_heca_uses_estimators_that_can_run_at_scale(self):
        # Spectral builds a dense n-by-n affinity and Ward is quadratic in
        # memory, so neither may appear in the large-scale study.
        classes = {cls for cls, _ in study_model_grids(STUDIES["heca"])}
        assert SpectralClustering not in classes
        assert AgglomerativeClustering not in classes

    def test_heca_declares_a_resolution_sweep(self):
        grids = study_resolution_grids(STUDIES["heca"])
        assert len(grids) == 1
        cls, grid = grids[0]
        assert cls is LeidenClustering
        assert grid["resolution"][0] == pytest.approx(0.1)
        assert grid["resolution"][-1] == pytest.approx(2.0)
        assert len(grid["resolution"]) == 20

    def test_a_study_without_resolutions_raises(self):
        with pytest.raises(ValueError, match="declares no resolutions"):
            study_resolution_grids(STUDIES["klein"])

    def test_heca_pins_its_anchor_count(self):
        # The 20-config Leiden resolution sweep retains both
        # consensus_matrices_ and consensus_generalizability_matrices_ per
        # config: 20 * 2 * 2000**2 * 8 B = 1.28 GB at m=2000, against 8.0 GB
        # at the package default (anchor_threshold=5000), which applies at
        # hECA's own scale.
        assert STUDIES["heca"].consensus_anchors == 2000

    def test_cusanovich_pins_the_same_anchor_count_as_heca(self):
        # The 15-configuration resolution sweep retains 0.96 GB of consensus
        # blocks at 2000 anchors, against 6.0 GB at the package default once
        # n exceeds anchor_threshold. Pinned to hECA's count for that reason.
        assert STUDIES["cusanovich"].consensus_anchors == 2000

    def test_new_studies_declare_dev_and_publication_scales(self):
        for name in ("cusanovich", "heca"):
            assert {"dev", "publication"} <= set(STUDIES[name].scales)

    def test_cusanovich_publication_scale_matches_levine(self):
        assert STUDIES["cusanovich"].scales["publication"] == 5000
        assert STUDIES["cusanovich"].scales["atlas"] is None


@pytest.fixture
def ladder_data():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [4.0, 7.0]])
    labels = rng.integers(0, 3, 400)
    X = centers[labels] + rng.normal(0, 1.0, (400, 2))
    return X, pd.Series(labels.astype(str), name="truth")


class TestStudyScalingSweep:
    def _grids(self):
        return [(KMeans, {"n_clusters": [2, 3, 4], "n_init": [10]})]

    def test_one_row_per_size(self, ladder_data):
        X, y = ladder_data
        out = study_scaling_sweep(
            X, y, sizes=[100, 200], model_grids=self._grids(), n_resamples=5
        )
        assert list(out["n"]) == [100, 200]
        assert len(out) == 2

    def test_columns_are_the_documented_contract(self, ladder_data):
        X, y = ladder_data
        out = study_scaling_sweep(
            X, y, sizes=[100], model_grids=self._grids(), n_resamples=5
        )
        assert list(out.columns) == [
            "n",
            "n_configs",
            "wall_clock_s",
            "peak_rss_bytes",
            "selected_k",
            "ari",
        ]

    def test_measurements_are_populated(self, ladder_data):
        X, y = ladder_data
        out = study_scaling_sweep(
            X, y, sizes=[100, 200], model_grids=self._grids(), n_resamples=5
        )
        assert (out["wall_clock_s"] > 0).all()
        assert (out["peak_rss_bytes"] > 0).all()
        assert out["n_configs"].nunique() == 1
        assert out["selected_k"].between(2, 4).all()

    def test_ari_reflects_the_planted_structure(self, ladder_data):
        # A sweep that scored against shuffled labels would still produce a
        # populated frame, so pin that the ARI is meaningful.
        X, y = ladder_data
        out = study_scaling_sweep(
            X, y, sizes=[200], model_grids=self._grids(), n_resamples=8
        )
        assert out["ari"].iloc[0] > 0.5

    def test_a_size_larger_than_n_is_skipped_with_a_warning(self, ladder_data):
        # Failing on an oversized top rung after already running every
        # smaller one would waste the prior rungs' compute for no reason
        # better than a hardcoded target the data may not actually reach
        # (see hECA's publication ladder). A skip-and-warn must not silently
        # swallow every rung, though -- the rows for sizes that do fit must
        # still come back.
        X, y = ladder_data
        with pytest.warns(UserWarning, match="10000"):
            out = study_scaling_sweep(
                X, y, sizes=[100, 10_000], model_grids=self._grids(), n_resamples=5
            )
        assert list(out["n"]) == [100]

    def test_every_size_larger_than_n_yields_an_empty_frame(self, ladder_data):
        X, y = ladder_data
        with pytest.warns(UserWarning, match="10000"):
            out = study_scaling_sweep(
                X, y, sizes=[10_000], model_grids=self._grids(), n_resamples=5
            )
        assert list(out.columns) == [
            "n",
            "n_configs",
            "wall_clock_s",
            "peak_rss_bytes",
            "selected_k",
            "ari",
        ]
        assert len(out) == 0

    def test_is_deterministic(self, ladder_data):
        X, y = ladder_data
        a = study_scaling_sweep(
            X, y, sizes=[150], model_grids=self._grids(), n_resamples=5
        )
        b = study_scaling_sweep(
            X, y, sizes=[150], model_grids=self._grids(), n_resamples=5
        )
        assert a["selected_k"].equals(b["selected_k"])
        assert np.allclose(a["ari"], b["ari"])

    def test_peak_rss_is_read_after_fit_not_before(self, ladder_data, monkeypatch):
        # wall_clock_s > 0 and peak_rss_bytes > 0 (test_measurements_are_
        # populated) cannot fail on a measurement taken at the wrong moment:
        # any running process has nonzero RSS and any real fit takes
        # nonzero time.
        #
        # A numeric "peak_rss_bytes is non-decreasing across an ascending
        # size ladder" check was tried instead and does not work either:
        # ru_maxrss is a per-process high-water mark that never falls, so
        # sampling it anywhere in a sequential, ascending-size loop -- even
        # right before fit() runs, which still lands after the previous
        # size's fit finished -- yields a non-decreasing sequence regardless
        # of where the sample is taken. Verified directly: moving the
        # peak_rss_bytes() call in study_scaling_sweep to before carve.fit()
        # left such a check green.
        #
        # This test instead pins the call order itself, independent of what
        # any particular OS reports for RSS: peak_rss_bytes must be read
        # after fit() has already run for that size, not before.
        import benchmarks._artifacts as artifacts
        from carve import CARVE

        X, y = ladder_data
        events: list[str] = []
        real_fit = CARVE.fit

        def spy_fit(self, X, *args, **kwargs):
            result = real_fit(self, X, *args, **kwargs)
            events.append("fit")
            return result

        def spy_peak_rss_bytes():
            events.append("peak")
            return 1

        monkeypatch.setattr(CARVE, "fit", spy_fit)
        monkeypatch.setattr(artifacts, "peak_rss_bytes", spy_peak_rss_bytes)

        study_scaling_sweep(
            X, y, sizes=[50, 100], model_grids=self._grids(), n_resamples=5
        )

        # One (fit, peak) pair per size, fit strictly before its peak.
        assert events == ["fit", "peak", "fit", "peak"]
