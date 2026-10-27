"""Tests for the SC3 bridge.

The tests here never touch R, except the live tests behind the
requires_sc3 marker.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from benchmarks._sc3 import (
    SC3_DEFAULTS,
    SC3Result,
    align_labels,
    check_input,
    check_spectrum_selection,
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
