"""Tests for carve._runner module."""

import warnings
from collections import Counter

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans

import carve._runner as carve_runner
import carve._utils as carve_utils
from carve._consensus import compute_consensus_metrics
from carve._runner import (
    ResampleResult,
    _compute_generalizability_ari,
    _compute_stability_ari,
    run_validation,
    validation_iter,
)
from carve._sweep import resolve_sweep
from carve._types import ConsensusSummary, ModePolicy


class _LabelStub(BaseEstimator, ClusterMixin):
    """Estimator returning a fixed label pattern, ignoring its input.

    Lets the runner's sanity checks be exercised deterministically without
    depending on what a real clusterer happens to produce on a subsample.
    """

    _pattern: np.ndarray

    def __init__(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

    def fit(self, X, y=None):
        n = np.asarray(X).shape[0]
        self.labels_ = np.resize(self._pattern, n)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_


class _TwoClusterStub(_LabelStub):
    """Always produces exactly 2 clusters."""

    _pattern = np.array([0, 1])


class _OneClusterStub(_LabelStub):
    """Always produces a single cluster (a degenerate partition)."""

    _pattern = np.array([0])


class _NoisyStub(_LabelStub):
    """Produces 2 clusters plus noise points, like HDBSCAN."""

    _pattern = np.array([0, 1, -1, 0, 1])


class _NJobsSpy(BaseEstimator, ClassifierMixin):
    """Records the ``n_jobs`` CARVE injects at fit time, then predicts a constant."""

    seen: list = []

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def fit(self, X, y):
        type(self).seen.append(self.n_jobs)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.classes_[0])


class _ParallelSpy:
    """Stands in for joblib.Parallel: records n_jobs, runs the tasks inline."""

    seen: list = []

    def __init__(self, n_jobs=None, **kwargs):
        type(self).seen.append(n_jobs)

    def __call__(self, tasks):
        return [func(*args, **kwargs) for func, args, kwargs in tasks]


@pytest.fixture()
def default_policy():
    return ModePolicy(
        mode="default",
        run_stability=True,
        run_generalizability=True,
        compute_average_ari=True,
    )


@pytest.fixture()
def stability_policy():
    return ModePolicy(
        mode="stability",
        run_stability=True,
        run_generalizability=False,
        compute_average_ari=False,
    )


@pytest.fixture()
def generalizability_policy():
    return ModePolicy(
        mode="generalizability",
        run_stability=False,
        run_generalizability=True,
        compute_average_ari=False,
    )


# -----------------------------------------------------------------------
# ResampleResult
# -----------------------------------------------------------------------


class TestResampleResult:
    def test_is_namedtuple(self):
        r = ResampleResult(
            ari_stability=0.8,
            ari_generalizability=0.7,
            labels_train=np.array([0, 1]),
            labels_test=np.array([0, 1]),
            labels_predicted=np.array([0, 1]),
            labels_stability=np.array([0, 1]),
            train_indices=np.array([0, 1]),
            test_indices=np.array([2, 3]),
            stability_indices=np.array([4, 5]),
            normalization_params={},
            dim_reduction_params={},
            normalization_name="Identity",
            dim_reduction_name="Identity",
            n_clusters_train=2,
            n_clusters_test=2,
            n_clusters_stability=2,
            noise_fraction=0.0,
        )
        assert r.ari_stability == 0.8
        assert r.ari_generalizability == 0.7
        assert r.normalization_name == "Identity"
        assert r.n_clusters_train == 2
        assert r.noise_fraction == 0.0

    def test_field_count(self):
        assert len(ResampleResult._fields) == 17


# -----------------------------------------------------------------------
# _compute_stability_ari
# -----------------------------------------------------------------------


class TestComputeStabilityAri:
    def test_basic(self, default_policy):
        # Overlapping indices [1, 2] with identical labels
        ari = _compute_stability_ari(
            policy=default_policy,
            P_1_idx=np.array([0, 1, 2, 3]),
            P_2_idx=np.array([1, 2, 4, 5]),
            labels_1=np.array([0, 0, 1, 1]),
            labels_2=np.array([0, 1, 0, 1]),
        )
        assert isinstance(ari, float)
        assert -0.5 <= ari <= 1.0

    def test_perfect_overlap(self, default_policy):
        idx = np.array([0, 1, 2, 3])
        labels = np.array([0, 0, 1, 1])
        ari = _compute_stability_ari(
            default_policy,
            idx,
            idx,
            labels,
            labels,
        )
        assert ari == pytest.approx(1.0)

    def test_skipped(self, generalizability_policy):
        ari = _compute_stability_ari(
            generalizability_policy,
            np.array([0, 1]),
            None,
            np.array([0, 1]),
            None,
        )
        assert np.isnan(ari)


# -----------------------------------------------------------------------
# _compute_generalizability_ari
# -----------------------------------------------------------------------


class TestComputeGeneralizabilityAri:
    def test_basic(self, default_policy):
        rng = np.random.RandomState(0)
        X_train = np.vstack([rng.randn(20, 3) + [3, 0, 0], rng.randn(20, 3)])
        X_test = np.vstack([rng.randn(5, 3) + [3, 0, 0], rng.randn(5, 3)])
        labels_train = np.array([0] * 20 + [1] * 20)
        labels_test = np.array([0] * 5 + [1] * 5)

        labels_pred, ari = _compute_generalizability_ari(
            default_policy,
            X_train,
            X_test,
            labels_train,
            labels_test,
            classifier=None,
            n_trees=100,
            seed=0,
        )
        assert labels_pred is not None
        assert labels_pred.shape == (10,)
        assert isinstance(ari, float)

    def test_skipped(self, stability_policy):
        labels_pred, ari = _compute_generalizability_ari(
            stability_policy,
            None,
            None,
            None,
            None,
            classifier=None,
            n_trees=100,
            seed=0,
        )
        assert labels_pred is None
        assert np.isnan(ari)

    def test_classifier_receives_the_thread_budget(self, default_policy):
        rng = np.random.RandomState(0)
        X_train = rng.randn(20, 3)
        X_test = rng.randn(5, 3)
        labels_train = np.array([0] * 10 + [1] * 10)
        labels_test = np.array([0] * 3 + [1] * 2)

        _NJobsSpy.seen.clear()
        try:
            _compute_generalizability_ari(
                default_policy,
                X_train,
                X_test,
                labels_train,
                labels_test,
                classifier=_NJobsSpy(),
                n_trees=100,
                seed=0,
                classifier_n_jobs=3,
            )
            recorded = list(_NJobsSpy.seen)
        finally:
            _NJobsSpy.seen.clear()
        assert recorded == [3]


# -----------------------------------------------------------------------
# validation_iter
# -----------------------------------------------------------------------


class TestValidationIter:
    def test_basic(self, X_two_clusters):
        result = validation_iter(
            X=X_two_clusters,
            est_class=KMeans,
            params={"n_clusters": 2},
            subsample_ratio=0.8,
            n_resamples=3,
            seed=0,
            normalization_options=[],
            dim_reduction_options=[],
            randomize_preprocessing=False,
            mode="default",
            random_state=0,
        )
        assert isinstance(result, ResampleResult)
        assert result.labels_train.shape[0] > 0
        assert result.train_indices.shape[0] > 0
        assert not np.isnan(result.ari_stability)
        assert not np.isnan(result.ari_generalizability)

    def test_stability_mode(self, X_two_clusters):
        result = validation_iter(
            X=X_two_clusters,
            est_class=KMeans,
            params={"n_clusters": 2},
            subsample_ratio=0.8,
            n_resamples=3,
            seed=0,
            normalization_options=[],
            dim_reduction_options=[],
            randomize_preprocessing=False,
            mode="stability",
            random_state=0,
        )
        assert not np.isnan(result.ari_stability)
        assert np.isnan(result.ari_generalizability)
        assert result.labels_predicted is None

    def test_generalizability_mode(self, X_two_clusters):
        result = validation_iter(
            X=X_two_clusters,
            est_class=KMeans,
            params={"n_clusters": 2},
            subsample_ratio=0.8,
            n_resamples=3,
            seed=0,
            normalization_options=[],
            dim_reduction_options=[],
            randomize_preprocessing=False,
            mode="generalizability",
            random_state=0,
        )
        assert np.isnan(result.ari_stability)
        assert not np.isnan(result.ari_generalizability)
        assert result.labels_stability is None

    def test_classifier_n_jobs_reaches_the_classifier(self, X_two_clusters):
        _NJobsSpy.seen.clear()
        try:
            validation_iter(
                X=X_two_clusters,
                est_class=KMeans,
                params={"n_clusters": 2},
                subsample_ratio=0.8,
                n_resamples=3,
                seed=0,
                normalization_options=[],
                dim_reduction_options=[],
                classifier=_NJobsSpy(),
                randomize_preprocessing=False,
                mode="default",
                random_state=0,
                classifier_n_jobs=5,
            )
            recorded = list(_NJobsSpy.seen)
        finally:
            _NJobsSpy.seen.clear()
        assert recorded == [5]


# -----------------------------------------------------------------------
# run_validation
# -----------------------------------------------------------------------


class TestRunValidation:
    def test_basic(self, X_two_clusters):
        records, pipeline_records, cons, cons_gen, gen_scores, _ = run_validation(
            X=X_two_clusters,
            estimator_grids=[(KMeans, {"n_clusters": [2]})],
            n_resamples=3,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=0,
            verbose=0,
        )
        assert len(records) == 1
        assert "ari_stability" in records[0]
        assert "ari_generalizability" in records[0]
        assert len(cons) == 1
        assert cons[0].shape == (60, 60)
        assert len(cons_gen) == 1
        assert len(gen_scores) == 1

    def test_multiple_k(self, X_two_clusters):
        records, *_ = run_validation(
            X=X_two_clusters,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_resamples=3,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=0,
            verbose=0,
        )
        assert len(records) == 2
        assert records[0]["n_clusters"] == 2
        assert records[1]["n_clusters"] == 3

    def test_stability_mode(self, X_two_clusters):
        records, _, cons, cons_gen, gen_scores, _ = run_validation(
            X=X_two_clusters,
            estimator_grids=[(KMeans, {"n_clusters": [2]})],
            n_resamples=3,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=0,
            mode="stability",
            verbose=0,
        )
        assert cons[0] is not None
        assert cons_gen[0] is None  # No generalizability consensus
        assert gen_scores[0] is None

    def test_generalizability_mode(self, X_two_clusters):
        records, _, cons, cons_gen, gen_scores, _ = run_validation(
            X=X_two_clusters,
            estimator_grids=[(KMeans, {"n_clusters": [2]})],
            n_resamples=3,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=0,
            mode="generalizability",
            verbose=0,
        )
        assert cons[0] is None  # No stability consensus
        assert cons_gen[0] is not None
        assert gen_scores[0] is not None

    def test_reproducibility(self, X_two_clusters):
        r1, *_ = run_validation(
            X=X_two_clusters,
            estimator_grids=[(KMeans, {"n_clusters": [2]})],
            n_resamples=3,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=42,
            verbose=0,
        )
        r2, *_ = run_validation(
            X=X_two_clusters,
            estimator_grids=[(KMeans, {"n_clusters": [2]})],
            n_resamples=3,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=42,
            verbose=0,
        )
        assert r1[0]["ari_stability"] == r2[0]["ari_stability"]

    def test_ari_in_range(self, X_two_clusters):
        records, *_ = run_validation(
            X=X_two_clusters,
            estimator_grids=[(KMeans, {"n_clusters": [2]})],
            n_resamples=5,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=0,
            verbose=0,
        )
        rec = records[0]
        assert -0.5 <= rec["ari_stability"] <= 1.0
        assert -0.5 <= rec["ari_generalizability"] <= 1.0
        assert rec["ari_stability_se"] >= 0


class TestRunValidationCoreBudget:
    """n_jobs is split once per run: workers over resamples, threads inside.

    Parallel receives the worker count and every resample's classifier
    receives the per-worker thread count. Parallel is replaced by an inline
    spy so the classifier spy's record survives the call, which it would not
    across loky processes.
    """

    @pytest.fixture(autouse=True)
    def eleven_cores(self, monkeypatch):
        monkeypatch.setattr(carve_utils, "cpu_count", lambda: 11)
        monkeypatch.setattr(carve_runner, "Parallel", _ParallelSpy)
        _ParallelSpy.seen.clear()
        _NJobsSpy.seen.clear()
        yield
        _ParallelSpy.seen.clear()
        _NJobsSpy.seen.clear()

    def _run(self, X, n_jobs, n_resamples=6, n_clusters=(2,)):
        run_validation(
            X=X,
            estimator_grids=[(KMeans, {"n_clusters": list(n_clusters)})],
            n_resamples=n_resamples,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            classifier=_NJobsSpy(),
            n_jobs=n_jobs,
            random_state=0,
            verbose=0,
        )

    def test_one_worker_gives_the_classifier_every_core(self, X_two_clusters):
        self._run(X_two_clusters, n_jobs=1)
        assert _ParallelSpy.seen == [1]
        assert _NJobsSpy.seen == [11] * 6

    def test_all_cores_gives_one_thread_per_resample(self, X_two_clusters):
        self._run(X_two_clusters, n_jobs=-1, n_resamples=20)
        assert _ParallelSpy.seen == [11]
        assert _NJobsSpy.seen == [1] * 20

    def test_partial_budget_splits_the_remainder(self, X_two_clusters):
        self._run(X_two_clusters, n_jobs=4)
        assert _ParallelSpy.seen == [4]
        assert _NJobsSpy.seen == [2] * 6

    def test_budget_is_resolved_once_per_run(self, X_two_clusters):
        # Two configurations, one Parallel call each, the same split for both.
        self._run(X_two_clusters, n_jobs=4, n_clusters=(2, 3))
        assert _ParallelSpy.seen == [4, 4]
        assert _NJobsSpy.seen == [2] * 12


# -----------------------------------------------------------------------
# Sweep bookkeeping and noise handling
# -----------------------------------------------------------------------


class TestValidationIterSweepMode:
    def _run(self, X, **kwargs):
        params = kwargs.pop("params", {"n_clusters": 2})
        est_class = kwargs.pop("est_class", KMeans)
        return validation_iter(
            X=X,
            est_class=est_class,
            params=params,
            subsample_ratio=0.8,
            n_resamples=3,
            seed=0,
            normalization_options=[],
            dim_reduction_options=[],
            randomize_preprocessing=False,
            mode="default",
            random_state=0,
            **kwargs,
        )

    def test_records_cluster_counts(self, X_two_clusters):
        result = self._run(X_two_clusters)
        assert result.n_clusters_train == 2
        assert result.n_clusters_test == 2
        assert result.n_clusters_stability == 2

    def test_noise_fraction_zero_without_noise(self, X_two_clusters):
        assert self._run(X_two_clusters).noise_fraction == 0.0

    def test_k_mode_warns_on_wrong_cluster_count(self, X_two_clusters):
        """The expected-k check only makes sense when k is pinned."""
        with pytest.warns(UserWarning, match="expected 3"):
            self._run(
                X_two_clusters,
                est_class=_TwoClusterStub,
                params={"n_clusters": 3},
                sweep_param="n_clusters",
            )

    def test_sweep_mode_does_not_warn_about_expected_k(self, X_two_clusters):
        """Resolution mode has no expected k, so that check must be skipped."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            self._run(
                X_two_clusters,
                est_class=_TwoClusterStub,
                params={"resolution": 0.5},
                sweep_param="resolution",
            )

    def test_sweep_mode_warns_on_degenerate_partition(self, X_two_clusters):
        with pytest.warns(UserWarning, match="degenerate at this point"):
            self._run(
                X_two_clusters,
                est_class=_OneClusterStub,
                params={"resolution": 0.01},
                sweep_param="resolution",
            )

    def test_noise_policy_drop_shrinks_indices(self, X_two_clusters):
        result = self._run(
            X_two_clusters,
            est_class=_NoisyStub,
            params={"resolution": 1.0},
            sweep_param="resolution",
            noise_policy="drop",
        )
        assert result.noise_fraction > 0
        assert result.labels_train.shape == result.train_indices.shape
        assert (result.labels_train >= 0).all()

    def test_noise_policy_as_cluster_keeps_noise(self, X_two_clusters):
        result = self._run(
            X_two_clusters,
            est_class=_NoisyStub,
            params={"resolution": 1.0},
            sweep_param="resolution",
            noise_policy="as_cluster",
        )
        assert result.noise_fraction > 0
        assert (result.labels_train < 0).any()

    def test_noise_fraction_agrees_across_policies(self, X_two_clusters):
        """noise_fraction is measured before the policy is applied."""
        fractions = {
            policy: self._run(
                X_two_clusters,
                est_class=_NoisyStub,
                params={"resolution": 1.0},
                sweep_param="resolution",
                noise_policy=policy,
            ).noise_fraction
            for policy in ("drop", "as_cluster", "singleton")
        }
        assert len(set(fractions.values())) == 1

    def test_unknown_noise_policy_raises(self, X_two_clusters):
        with pytest.raises(ValueError, match="Unknown noise_policy"):
            self._run(X_two_clusters, noise_policy="bogus")


class TestRunValidationRecords:
    def _records(self, X, **kwargs):
        records, *_ = run_validation(
            X=X,
            estimator_grids=kwargs.pop(
                "estimator_grids", [(KMeans, {"n_clusters": [2, 3]})]
            ),
            n_resamples=3,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=0,
            **kwargs,
        )
        return records

    def test_identity_columns_present(self, X_two_clusters):
        records = self._records(X_two_clusters)
        for record in records:
            assert set(record) >= {"config_id", "method_id", "method_label"}

    def test_config_id_is_contiguous(self, X_two_clusters):
        records = self._records(X_two_clusters)
        assert [r["config_id"] for r in records] == list(range(len(records)))

    def test_sweep_columns_present(self, X_two_clusters):
        for record in self._records(X_two_clusters):
            assert set(record) >= {
                "sweep_param",
                "sweep_value",
                "sweep_rank",
                "n_clusters_observed",
                "n_clusters_observed_se",
                "noise_fraction",
            }

    def test_sweep_values_and_ranks(self, X_two_clusters):
        records = self._records(X_two_clusters)
        assert [r["sweep_param"] for r in records] == ["n_clusters"] * 2
        assert [r["sweep_value"] for r in records] == [2, 3]
        assert [r["sweep_rank"] for r in records] == [0, 1]

    def test_observed_k_matches_requested_k(self, X_two_clusters):
        records = self._records(X_two_clusters)
        assert [r["n_clusters_observed"] for r in records] == [2.0, 3.0]

    def test_one_method_id_per_curve(self, X_two_clusters):
        records = self._records(
            X_two_clusters,
            estimator_grids=[
                (KMeans, {"n_clusters": [2, 3]}),
                (
                    AgglomerativeClustering,
                    {"n_clusters": [2, 3], "linkage": ["ward", "average"]},
                ),
            ],
        )
        counts = Counter(r["method_id"] for r in records)
        assert len(counts) == 3
        assert set(counts.values()) == {2}

    def test_method_label_omits_the_swept_param(self, X_two_clusters):
        records = self._records(
            X_two_clusters,
            estimator_grids=[
                (
                    AgglomerativeClustering,
                    {"n_clusters": [2, 3], "linkage": ["ward"]},
                )
            ],
        )
        labels = {r["method_label"] for r in records}
        assert labels == {"AgglomerativeClustering, linkage=ward"}

    @pytest.mark.filterwarnings(
        "ignore:All points in a subsample were labelled as noise:UserWarning"
    )
    def test_sweep_rank_inverted_for_min_cluster_size(self, X_two_clusters):
        records = self._records(
            X_two_clusters,
            estimator_grids=[(HDBSCAN, {"min_cluster_size": [3, 5, 8]})],
            sweep=resolve_sweep(sweep="min_cluster_size", sweep_values=[3, 5, 8]),
        )
        assert [r["sweep_value"] for r in records] == [3, 5, 8]
        assert [r["sweep_rank"] for r in records] == [2, 1, 0]


class TestRunValidationAnchors:
    def _grids(self):
        return [(KMeans, {"n_clusters": [2, 3], "n_init": [10]})]

    def test_summaries_are_full_length_under_anchoring(self):
        rng = np.random.default_rng(0)
        n = 60
        # A mild gap, not the well-separated blobs used elsewhere: with a
        # clean 6-sigma split, k=2 recovers the identical partition on every
        # resample and the per-sample scores are trivially constant at 1,
        # which would defeat the "not constant" check below. This overlap
        # keeps some samples near the boundary so scores genuinely vary.
        X = np.vstack([rng.normal(0, 1, (n // 2, 4)), rng.normal(2, 1, (n // 2, 4))])
        anchors = np.sort(rng.choice(n, 20, replace=False))

        out = run_validation(
            X=X,
            estimator_grids=self._grids(),
            n_resamples=6,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            random_state=0,
            anchors=anchors,
        )
        summaries = out[5]
        matrices = out[2]

        assert summaries is not None
        assert len(summaries) == len(matrices)
        for summary, matrix in zip(summaries, matrices):
            assert isinstance(summary, ConsensusSummary)
            # Scores cover every sample even though the block is anchor sized.
            assert summary.gini.shape == (n,)
            assert summary.ce.shape == (n,)
            assert matrix.shape == (anchors.size, anchors.size)

            # Scores are real stability values, not placeholders: bounded,
            # finite, and not degenerately constant across samples.
            assert np.isfinite(summary.pac)
            assert 0.0 <= summary.pac <= 1.0
            for scores in (summary.gini, summary.ce):
                assert np.all(np.isfinite(scores))
                assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
                assert np.std(scores) > 0.0

    def test_exact_path_is_unchanged_when_anchors_is_none(self):
        rng = np.random.default_rng(1)
        n = 60
        X = np.vstack([rng.normal(0, 1, (n // 2, 4)), rng.normal(6, 1, (n // 2, 4))])

        out = run_validation(
            X=X,
            estimator_grids=self._grids(),
            n_resamples=6,
            subsample_ratio=0.8,
            normalization_options=[],
            dim_reduction_options=[],
            random_state=0,
            anchors=None,
        )
        matrices = out[2]
        summaries = out[5]

        for matrix in matrices:
            assert matrix.shape == (n, n)
        for summary in summaries:
            assert summary.gini.shape == (n,)

        # The claim under test: the runner's exact-path summaries equal what
        # the old api.fit code path computed post-hoc from the same matrices.
        gini_list, ce_list, pac_list = compute_consensus_metrics(matrices)
        for summary, gini, ce, pac in zip(summaries, gini_list, ce_list, pac_list):
            assert np.allclose(summary.gini, gini)
            assert np.allclose(summary.ce, ce)
            assert np.isclose(summary.pac, pac)
