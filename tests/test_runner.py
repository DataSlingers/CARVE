"""Tests for carve._runner module."""

import numpy as np
import pytest
from sklearn.cluster import KMeans

from carve._runner import (
    ResampleResult,
    _compute_generalizability_ari,
    _compute_stability_ari,
    run_validation,
    validation_iter,
)
from carve._types import ModePolicy


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
        )
        assert r.ari_stability == 0.8
        assert r.ari_generalizability == 0.7
        assert r.normalization_name == "Identity"

    def test_field_count(self):
        assert len(ResampleResult._fields) == 13


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
            seed=0,
        )
        assert labels_pred is None
        assert np.isnan(ari)


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


# -----------------------------------------------------------------------
# run_validation
# -----------------------------------------------------------------------


class TestRunValidation:
    def test_basic(self, X_two_clusters):
        records, pipeline_records, cons, cons_gen, gen_scores = run_validation(
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
        records, _, cons, cons_gen, gen_scores = run_validation(
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
        records, _, cons, cons_gen, gen_scores = run_validation(
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
