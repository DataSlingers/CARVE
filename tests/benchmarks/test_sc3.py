"""Tests for the SC3 bridge.

The tests here never touch R, except the live tests behind the
requires_sc3 marker.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from benchmarks._sc3 import (
    SC3_DEFAULTS,
    SC3Result,
    align_labels,
    check_input,
    check_spectrum_selection,
    run_sc3,
    tracy_widom_bound,
)


def stub_result(**overrides) -> SC3Result:
    """A result shaped as run_sc3 returns one, for tests that never reach R."""
    base = dict(
        labels=np.array([0, 0, 1, 1, 2, 2]),
        k=3,
        spectrum=pd.DataFrame(
            {"index": [1, 2, 3, 4], "eigenvalue": [900.0, 400.0, 150.0, 20.0]}
        ),
        bound=100.0,
        n_features_used=3,
        n_dim=(1, 2),
        kmeans_nstart=1000,
        n_cores=2,
        runtime_s=12.5,
        config={**SC3_DEFAULTS, "gene_filter": False, "seed": 1},
        sc3_version="1.36.0",
        r_version="R version 4.5.1 (2025-06-13)",
    )
    base.update(overrides)
    return SC3Result(**base)


class TestDefaults:
    def test_pins_sc3_s_defaults(self):
        assert dict(SC3_DEFAULTS) == {
            "pct_dropout_min": 10,
            "pct_dropout_max": 90,
            "d_region_min": 0.04,
            "d_region_max": 0.07,
            "svm_max": 5000,
            "kmeans_iter_max": 1e9,
            "biology": False,
        }

    def test_defaults_are_not_mutable(self):
        with pytest.raises(TypeError):
            SC3_DEFAULTS["svm_max"] = 1


class TestResult:
    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            stub_result().k = 4


class TestTracyWidomBound:
    # Bounds SC3 1.36.0 itself reported on the case-study matrices.
    def test_matches_sc3_on_levine(self):
        assert tracy_widom_bound(32, 5000) == pytest.approx(5962.8648, abs=1e-3)

    def test_matches_sc3_on_klein_after_the_gene_filter(self):
        assert tracy_widom_bound(1817, 2717) == pytest.approx(9083.8935, abs=1e-3)

    def test_matches_sc3_on_klein_without_the_gene_filter(self):
        assert tracy_widom_bound(2000, 2717) == pytest.approx(9486.7982, abs=1e-3)


class TestCheckSpectrumSelection:
    def test_silent_when_the_count_matches(self):
        check_spectrum_selection(np.array([900.0, 400.0, 150.0, 20.0]), 100.0, k_estimation=3)

    def test_raises_when_the_count_differs(self):
        with pytest.raises(RuntimeError, match="estkTW"):
            check_spectrum_selection(np.array([900.0, 400.0, 150.0, 20.0]), 100.0, k_estimation=2)

    def test_an_eigenvalue_at_the_bound_is_not_counted(self):
        # estkTW counts evals[i] > bd, strictly.
        check_spectrum_selection(np.array([900.0, 100.0]), 100.0, k_estimation=1)


class TestAlignLabels:
    names = ["c0", "c1", "c2", "c3"]

    def test_reorders_by_name(self):
        labels = align_labels([2, 1, 2, 1], ["c3", "c2", "c1", "c0"], self.names, k=2)
        np.testing.assert_array_equal(labels, [1, 2, 1, 2])

    def test_raises_when_the_count_does_not_match(self):
        with pytest.raises(RuntimeError, match="orientation"):
            align_labels([1, 2, 1], ["c0", "c1", "c2"], self.names, k=2)

    def test_raises_when_the_names_do_not_match(self):
        with pytest.raises(RuntimeError, match="names"):
            align_labels([1, 2, 1, 2], ["x0", "x1", "x2", "x3"], self.names, k=2)

    def test_raises_unless_there_are_exactly_k_clusters(self):
        with pytest.raises(RuntimeError, match="distinct"):
            align_labels([1, 1, 1, 2], self.names, self.names, k=3)

    def test_returns_integer_labels(self):
        labels = align_labels(np.array([1.0, 2.0, 1.0, 2.0]), self.names, self.names, k=2)
        assert labels.dtype.kind == "i"


class TestCheckInput:
    def test_refuses_negative_values_under_the_gene_filter(self):
        X = np.array([[0.0, 1.0], [-0.1, 2.0]])
        with pytest.raises(ValueError, match="gene_filter=False"):
            check_input(X, gene_filter=True)

    def test_accepts_negative_values_without_the_gene_filter(self):
        X = np.array([[0.0, 1.0], [-0.1, 2.0]])
        assert check_input(X, gene_filter=False).dtype == np.float64

    def test_refuses_more_cells_than_svm_max(self):
        X = np.zeros((SC3_DEFAULTS["svm_max"] + 1, 2))
        with pytest.raises(ValueError, match="svm_max"):
            check_input(X, gene_filter=False)

    def test_accepts_exactly_svm_max_cells(self):
        # Levine's 5,000 cells: SC3 enters SVM mode only when n > svm_max.
        X = np.zeros((SC3_DEFAULTS["svm_max"], 2))
        assert check_input(X, gene_filter=False).shape == X.shape

    def test_expm1_keeps_the_zero_pattern_the_gene_filter_reads(self):
        # run_sc3 derives counts as expm1(logcounts); the filter reads only
        # counts == 0, which must coincide with X == 0.
        X = np.array([[0.0, 1e-12, 0.5], [7.3, 0.0, 2.0]])
        np.testing.assert_array_equal(np.expm1(X) == 0, X == 0)


class TestRunGuards:
    def test_refuses_to_install_unless_asked(self, monkeypatch):
        # Simulate SC3 being absent so the install branch is reached on a
        # machine that has it.
        monkeypatch.setattr("benchmarks._sc3._sc3_is_installed", lambda: False)
        with pytest.raises(RuntimeError, match=r'BiocManager::install\("SC3"\)'):
            run_sc3(np.ones((10, 4)), gene_filter=False, allow_install=False)

    def test_checks_the_input_before_touching_r(self, monkeypatch):
        def fail():
            raise AssertionError("R was reached")

        monkeypatch.setattr("benchmarks._sc3._sc3_is_installed", fail)
        with pytest.raises(ValueError, match="svm_max"):
            run_sc3(np.zeros((5001, 2)), gene_filter=False)


def _blobs(*, low_sd: float = 0.5, seed: int = 0) -> np.ndarray:
    """Three groups of 50 cells, each high on its own block of 5 of 30 features.

    The 15 features no group is high on matter. SC3 scales each cell across
    its features before counting eigenvalues, and block profiles covering
    every feature sum to a constant and leave one direction too few: the same
    blocks over 20 features gave k = 2 with SC3 1.36.0, this design k = 3.
    """
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, 30))
    centers[0, 0:5] = 4.0
    centers[1, 5:10] = 4.0
    centers[2, 10:15] = 4.0
    return np.vstack(
        [
            np.abs(rng.normal(c, np.where(c > 0, 0.5, low_sd), size=(50, 30)))
            for c in centers
        ]
    )


TRUTH = np.repeat([0, 1, 2], 50)


@pytest.mark.requires_sc3
class TestRunLive:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        # A class-scoped fixture must be a classmethod under pytest 9 (see
        # test_m3c.TestRunLive).
        return run_sc3(_blobs(), gene_filter=False, seed=1, n_cores=1)

    def test_estimates_three_clusters(self, result):
        assert result.k == 3

    def test_recovers_the_blobs(self, result):
        assert adjusted_rand_score(TRUTH, result.labels) == 1.0

    def test_counts_k_eigenvalues_above_the_bound(self, result):
        assert int((result.spectrum["eigenvalue"] > result.bound).sum()) == 3

    def test_spectrum_is_largest_first_with_a_one_based_index(self, result):
        values = result.spectrum["eigenvalue"].to_numpy()
        assert np.all(np.diff(values) <= 0)
        assert result.spectrum["index"].tolist()[:3] == [1, 2, 3]

    def test_bound_matches_the_restated_formula(self, result):
        assert result.bound == pytest.approx(tracy_widom_bound(30, 150), rel=1e-9)

    def test_records_what_was_run(self, result):
        assert result.sc3_version == "1.36.0"
        assert result.r_version.startswith("R version")
        assert result.config == {**SC3_DEFAULTS, "gene_filter": False, "seed": 1}
        assert result.n_features_used == 30
        assert result.kmeans_nstart == 1000
        # floor(0.04 * 150) to ceiling(0.07 * 150): six values, none sampled.
        assert result.n_dim == (6, 7, 8, 9, 10, 11)
        assert result.n_cores == 1

    def test_the_same_seed_gives_identical_labels(self, result):
        again = run_sc3(_blobs(), gene_filter=False, seed=1, n_cores=1)
        np.testing.assert_array_equal(again.labels, result.labels)

    def test_the_core_count_does_not_change_the_labels(self, result):
        # doRNG assigns one stream per task, not per worker. This is what
        # lets n_cores stay out of the cache key (run_or_load_sc3).
        two = run_sc3(_blobs(), gene_filter=False, seed=1, n_cores=2)
        np.testing.assert_array_equal(two.labels, result.labels)

    def test_the_gene_filter_drops_features_with_no_detections(self):
        X = _blobs(low_sd=1.0)
        X[X < 1.0] = 0.0
        X = np.hstack([X, np.zeros((150, 2))])
        result = run_sc3(X, gene_filter=True, seed=1, n_cores=1)
        assert result.n_features_used == 30
        assert result.k == 3
