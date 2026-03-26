"""Tests for carve._selection module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans, AgglomerativeClustering

from carve._selection import (
    MEASURE_MAP,
    build_estimator_from_row,
    get_estimator_param_names,
    row_to_estimator_params,
    select_best_estimator,
    select_best_k,
    select_best_row_1se,
    select_best_row_by_rule,
    select_best_row_max,
    select_best_row_quantile,
)


# -----------------------------------------------------------------------
# select_best_row_max
# -----------------------------------------------------------------------


class TestSelectBestRowMax:
    def test_basic(self, results_df):
        row = select_best_row_max(results_df, measure="stability")
        assert row["n_clusters"] == 2  # highest ari_stability
        assert row["ari_stability"] == 0.9

    def test_return_idx(self, results_df):
        idx = select_best_row_max(results_df, measure="stability", return_idx=True)
        assert idx == 0

    def test_different_measure(self, results_df):
        row = select_best_row_max(results_df, measure="generalizability")
        assert row["n_clusters"] == 2  # highest ari_generalizability

    def test_aliases(self, results_df):
        # Short aliases should work
        row_s = select_best_row_max(results_df, measure="s")
        row_stab = select_best_row_max(results_df, measure="stab")
        row_full = select_best_row_max(results_df, measure="ari_stability")
        assert row_s["n_clusters"] == row_stab["n_clusters"] == row_full["n_clusters"]


# -----------------------------------------------------------------------
# select_best_row_1se
# -----------------------------------------------------------------------


class TestSelectBestRow1se:
    def test_basic(self, results_df):
        row = select_best_row_1se(results_df, measure="stability")
        # Best score is 0.9 at k=2, SE=0.02, threshold=0.88
        # k=3 has 0.85 < 0.88, so only k=2 qualifies => largest k within 1se is 2
        assert row["n_clusters"] == 2

    def test_wider_se(self):
        df = pd.DataFrame(
            {
                "estimator": ["KMeans"] * 4,
                "n_clusters": [2, 3, 4, 5],
                "ari_stability": [0.90, 0.89, 0.88, 0.50],
                "ari_stability_se": [0.05, 0.04, 0.03, 0.02],
            }
        )
        row = select_best_row_1se(df, measure="stability")
        # Threshold: 0.90 - 0.05 = 0.85; k=2,3,4 all >= 0.85
        # Largest k within 1se is 4
        assert row["n_clusters"] == 4

    def test_return_idx(self, results_df):
        idx = select_best_row_1se(results_df, measure="stability", return_idx=True)
        assert isinstance(idx, (int, np.integer))

    def test_invalid_measure(self, results_df):
        with pytest.raises(ValueError, match="Invalid measure"):
            select_best_row_1se(results_df, measure="nonexistent")


# -----------------------------------------------------------------------
# select_best_row_quantile
# -----------------------------------------------------------------------


class TestSelectBestRowQuantile:
    def test_basic(self, results_df):
        row = select_best_row_quantile(results_df, measure="stability")
        # Best score at k=2: upper=0.92, lower=0.88
        # k=2 (0.9) is within [0.88, 0.92], k=3 (0.85) < 0.88
        assert row["n_clusters"] == 2

    def test_wider_bounds(self):
        df = pd.DataFrame(
            {
                "estimator": ["KMeans"] * 3,
                "n_clusters": [2, 3, 4],
                "ari_stability": [0.85, 0.83, 0.80],
                "ari_stability_se": [0.03, 0.03, 0.03],
                "ari_stability_upper": [0.90, 0.88, 0.85],
                "ari_stability_lower": [0.80, 0.78, 0.75],
            }
        )
        row = select_best_row_quantile(df, measure="stability")
        # Best at k=2: bounds [0.80, 0.90]; k=3 (0.83) and k=4 (0.80)
        # are within bounds => largest k is 4
        assert row["n_clusters"] == 4

    def test_return_idx(self, results_df):
        idx = select_best_row_quantile(
            results_df,
            measure="stability",
            return_idx=True,
        )
        assert isinstance(idx, (int, np.integer))

    def test_invalid_measure(self, results_df):
        with pytest.raises(ValueError, match="Invalid measure"):
            select_best_row_quantile(results_df, measure="nonexistent")

    def test_empty_fallback(self):
        """When no rows fall within quantile bounds, fall back to max."""
        df = pd.DataFrame(
            {
                "estimator": ["KMeans"] * 2,
                "n_clusters": [2, 3],
                "ari_stability": [0.5, 0.3],
                "ari_stability_se": [0.01, 0.01],
                # Inverted bounds so that no row satisfies >= lower AND <= upper
                "ari_stability_upper": [0.40, 0.20],
                "ari_stability_lower": [0.60, 0.40],
            }
        )
        with pytest.warns(RuntimeWarning, match="falling back to max"):
            row = select_best_row_quantile(df, measure="stability")
        assert row["n_clusters"] == 2


# -----------------------------------------------------------------------
# select_best_row_by_rule
# -----------------------------------------------------------------------


class TestSelectBestRowByRule:
    def test_max_rule(self, results_df):
        row = select_best_row_by_rule(results_df, measure="stability", rule="max")
        assert row["n_clusters"] == 2

    def test_1se_rule(self, results_df):
        row = select_best_row_by_rule(results_df, measure="stability", rule="1se")
        assert isinstance(row, pd.Series)

    def test_quantile_rule(self, results_df):
        row = select_best_row_by_rule(results_df, measure="stability", rule="quantile")
        assert isinstance(row, pd.Series)

    def test_invalid_rule(self, results_df):
        with pytest.raises(ValueError, match="Unknown rule"):
            select_best_row_by_rule(results_df, measure="stability", rule="invalid")

    def test_missing_se_fallback(self):
        """Missing SE column should warn and fall back to max."""
        df = pd.DataFrame(
            {
                "estimator": ["KMeans"] * 2,
                "n_clusters": [2, 3],
                "ari_stability": [0.9, 0.8],
            }
        )
        with pytest.warns(RuntimeWarning, match="not found.*falling back"):
            row = select_best_row_by_rule(df, measure="stability", rule="1se")
        assert row["n_clusters"] == 2

    def test_return_idx(self, results_df):
        idx = select_best_row_by_rule(
            results_df,
            measure="stability",
            rule="max",
            return_idx=True,
        )
        assert isinstance(idx, (int, np.integer))


# -----------------------------------------------------------------------
# select_best_k
# -----------------------------------------------------------------------


class TestSelectBestK:
    def test_basic(self, results_df):
        k = select_best_k(results_df, measure="stability", rule="max")
        assert k == 2

    def test_different_measures(self, results_df):
        for measure in ["stability", "generalizability", "pac", "gini", "ce"]:
            k = select_best_k(results_df, measure=measure, rule="max")
            assert k == 2  # all peak at k=2 in our fixture


# -----------------------------------------------------------------------
# select_best_estimator
# -----------------------------------------------------------------------


class TestSelectBestEstimator:
    def test_basic(self, results_df):
        grids = [(KMeans, {"n_clusters": [2, 3, 4, 5]})]
        est = select_best_estimator(results_df, grids, measure="stability")
        assert isinstance(est, KMeans)
        assert est.n_clusters == 2

    def test_with_k_filter(self, results_df):
        grids = [(KMeans, {"n_clusters": [2, 3, 4, 5]})]
        est = select_best_estimator(results_df, grids, measure="stability", k=3)
        assert est.n_clusters == 3


# -----------------------------------------------------------------------
# build_estimator_from_row
# -----------------------------------------------------------------------


class TestBuildEstimatorFromRow:
    def test_basic(self):
        grids = [(KMeans, {"n_clusters": [2]})]
        row = pd.Series({"estimator": "KMeans", "n_clusters": 3})
        est = build_estimator_from_row(grids, row)
        assert isinstance(est, KMeans)
        assert est.n_clusters == 3

    def test_with_extra_columns(self):
        grids = [(KMeans, {"n_clusters": [2]})]
        row = pd.Series(
            {
                "estimator": "KMeans",
                "n_clusters": 4,
                "ari_stability": 0.9,  # should be ignored
            }
        )
        est = build_estimator_from_row(grids, row)
        assert est.n_clusters == 4


# -----------------------------------------------------------------------
# get_estimator_param_names
# -----------------------------------------------------------------------


class TestGetEstimatorParamNames:
    def test_kmeans(self):
        names = get_estimator_param_names(KMeans)
        assert "n_clusters" in names
        assert "random_state" in names

    def test_agglomerative(self):
        names = get_estimator_param_names(AgglomerativeClustering)
        assert "n_clusters" in names
        assert "linkage" in names


# -----------------------------------------------------------------------
# row_to_estimator_params
# -----------------------------------------------------------------------


class TestRowToEstimatorParams:
    def test_basic(self):
        row = pd.Series({"n_clusters": 3, "random_state": 42})
        params = row_to_estimator_params(row, {"n_clusters", "random_state"})
        assert params == {"n_clusters": 3, "random_state": 42}

    def test_filters_nan(self):
        row = pd.Series({"n_clusters": 3, "linkage": np.nan})
        params = row_to_estimator_params(row, {"n_clusters", "linkage"})
        assert "linkage" not in params
        assert params["n_clusters"] == 3

    def test_filters_none(self):
        row = pd.Series({"n_clusters": 3, "linkage": None})
        params = row_to_estimator_params(row, {"n_clusters", "linkage"})
        assert "linkage" not in params

    def test_ignores_unknown_keys(self):
        row = pd.Series({"n_clusters": 3, "not_a_param": "foo"})
        params = row_to_estimator_params(row, {"n_clusters"})
        assert "not_a_param" not in params


# -----------------------------------------------------------------------
# MEASURE_MAP
# -----------------------------------------------------------------------


class TestMeasureMap:
    def test_all_aliases_resolve(self):
        expected_targets = {
            "ari_stability",
            "ari_generalizability",
            "ari_average",
            "consensus_pac_stability",
            "consensus_gini_stability",
            "consensus_ce_stability",
            "accuracy_generalizability",
        }
        assert set(MEASURE_MAP.values()) == expected_targets

    def test_short_aliases(self):
        assert MEASURE_MAP["s"] == "ari_stability"
        assert MEASURE_MAP["g"] == "ari_generalizability"
        assert MEASURE_MAP["avg"] == "ari_average"
        assert MEASURE_MAP["pac"] == "consensus_pac_stability"
        assert MEASURE_MAP["gini"] == "consensus_gini_stability"
        assert MEASURE_MAP["ce"] == "consensus_ce_stability"
        assert MEASURE_MAP["acc"] == "accuracy_generalizability"
