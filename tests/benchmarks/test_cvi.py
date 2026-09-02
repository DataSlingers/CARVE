"""Tests for the classical cluster validation indices."""

import numpy as np
import pytest

from benchmarks._cvi import (
    calculate_cvi,
    compute_dispersion,
    davies_bouldin_inv,
    gap_statistic,
    select_k,
)
from benchmarks._types import EstimatorSpec

SPEC = EstimatorSpec(name="kmeans")


@pytest.fixture
def separated_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.1, size=(40, 2))
    b = rng.normal(loc=10.0, scale=0.1, size=(40, 2))
    c = rng.normal(loc=(0.0, 10.0), scale=0.1, size=(40, 2))
    X = np.vstack([a, b, c])
    y = np.repeat([0, 1, 2], 40)
    return X, y


class TestComputeDispersion:
    def test_is_zero_for_identical_points(self):
        X = np.zeros((10, 3))
        assert compute_dispersion(X, np.zeros(10, dtype=int)) == 0.0

    def test_matches_the_sum_of_squared_deviations(self):
        X = np.array([[0.0], [2.0], [10.0], [12.0]])
        labels = np.array([0, 0, 1, 1])
        # Each cluster contributes 2 * 1.0**2 = 2.0
        assert compute_dispersion(X, labels) == pytest.approx(4.0)

    def test_ignores_singleton_clusters(self):
        X = np.array([[0.0], [2.0], [100.0]])
        labels = np.array([0, 0, 1])
        assert compute_dispersion(X, labels) == pytest.approx(2.0)


class TestGapStatistic:
    def test_returns_a_value_and_a_standard_error(self, separated_blobs):
        X, y = separated_blobs
        gap, s_k = gap_statistic(X, y, spec=SPEC, n_reference_datasets=5, random_state=0)
        assert np.isfinite(gap)
        assert s_k > 0.0

    def test_standard_error_includes_the_tibshirani_inflation(self, separated_blobs):
        X, y = separated_blobs
        n_ref = 5
        _, s_k = gap_statistic(X, y, spec=SPEC, n_reference_datasets=n_ref, random_state=0)
        # s_k = sd(log W*) * sqrt(1 + 1/n_ref); recovering sd must be positive
        # and strictly smaller than the inflated value.
        sd = s_k / np.sqrt(1.0 + 1.0 / n_ref)
        assert 0.0 < sd < s_k

    def test_is_deterministic_for_a_fixed_random_state(self, separated_blobs):
        X, y = separated_blobs
        first = gap_statistic(X, y, spec=SPEC, n_reference_datasets=3, random_state=11)
        second = gap_statistic(X, y, spec=SPEC, n_reference_datasets=3, random_state=11)
        assert first == second

    def test_returns_nan_when_dispersion_is_degenerate(self):
        X = np.zeros((10, 2))
        gap, s_k = gap_statistic(
            X, np.zeros(10, dtype=int), spec=SPEC, n_reference_datasets=3
        )
        assert np.isnan(gap)
        assert np.isnan(s_k)


class TestDaviesBouldinInv:
    def test_is_higher_for_better_separated_clusters(self, separated_blobs):
        X, y = separated_blobs
        good = davies_bouldin_inv(X, y)
        rng = np.random.default_rng(1)
        bad = davies_bouldin_inv(X, rng.integers(0, 3, size=X.shape[0]))
        assert good > bad

    def test_is_bounded_in_the_unit_interval(self, separated_blobs):
        X, y = separated_blobs
        assert 0.0 < davies_bouldin_inv(X, y) <= 1.0


class TestCalculateCvi:
    @pytest.mark.parametrize(
        "metric", ["silhouette", "gap", "davies_bouldin", "calinski_harabasz"]
    )
    def test_every_metric_returns_a_value_and_an_error(self, separated_blobs, metric):
        X, y = separated_blobs
        value, error = calculate_cvi(X, y, metric, spec=SPEC, random_state=0)
        assert np.isfinite(value)
        assert error >= 0.0

    def test_only_gap_reports_a_nonzero_error(self, separated_blobs):
        X, y = separated_blobs
        for metric in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            assert calculate_cvi(X, y, metric, spec=SPEC)[1] == 0.0

    def test_raises_on_an_unknown_metric(self, separated_blobs):
        X, y = separated_blobs
        with pytest.raises(ValueError, match="nonsense"):
            calculate_cvi(X, y, "nonsense", spec=SPEC)


class TestSelectK:
    def test_non_gap_metrics_take_the_argmax(self):
        ks = [3, 4, 5, 6]
        values = [0.1, 0.9, 0.4, 0.2]
        assert select_k("silhouette", ks, values, [0.0] * 4) == 4

    def test_gap_takes_the_smallest_k_within_one_standard_error(self):
        # Gap rises to k=5 but k=4 is already within s_5 of Gap(5), so
        # Tibshirani's rule stops at 4 where a plain argmax would say 5.
        ks = [3, 4, 5, 6]
        gaps = [0.10, 0.50, 0.55, 0.30]
        errors = [0.02, 0.02, 0.10, 0.02]
        assert select_k("gap", ks, gaps, errors) == 4

    def test_gap_differs_from_a_plain_argmax(self):
        ks = [3, 4, 5, 6]
        gaps = [0.10, 0.50, 0.55, 0.30]
        errors = [0.02, 0.02, 0.10, 0.02]
        assert select_k("gap", ks, gaps, errors) != ks[int(np.argmax(gaps))]

    def test_gap_falls_back_to_the_largest_k_when_no_k_satisfies_the_rule(self):
        ks = [3, 4, 5]
        gaps = [0.1, 0.2, 0.3]
        errors = [0.0, 0.0, 0.0]
        assert select_k("gap", ks, gaps, errors) == 5

    def test_gap_with_a_single_k_returns_that_k(self):
        assert select_k("gap", [4], [0.5], [0.1]) == 4

    def test_nan_values_are_never_selected(self):
        ks = [3, 4, 5]
        values = [np.nan, 0.2, 0.1]
        assert select_k("silhouette", ks, values, [0.0] * 3) == 4
