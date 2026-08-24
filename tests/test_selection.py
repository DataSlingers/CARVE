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

from conftest import with_sweep_cols


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
        df = with_sweep_cols(
            pd.DataFrame(
                {
                    "estimator": ["KMeans"] * 4,
                    "n_clusters": [2, 3, 4, 5],
                    "ari_stability": [0.90, 0.89, 0.88, 0.50],
                    "ari_stability_se": [0.05, 0.04, 0.03, 0.02],
                }
            ),
            param="n_clusters",
            method_label="KMeans",
            observed=[2.0, 3.0, 4.0, 5.0],
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
        df = with_sweep_cols(
            pd.DataFrame(
                {
                    "estimator": ["KMeans"] * 3,
                    "n_clusters": [2, 3, 4],
                    "ari_stability": [0.85, 0.83, 0.80],
                    "ari_stability_se": [0.03, 0.03, 0.03],
                    "ari_stability_upper": [0.90, 0.88, 0.85],
                    "ari_stability_lower": [0.80, 0.78, 0.75],
                }
            ),
            param="n_clusters",
            method_label="KMeans",
            observed=[2.0, 3.0, 4.0],
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


# -----------------------------------------------------------------------
# Sweep-axis selection (rank, not k)
# -----------------------------------------------------------------------


class TestSelectionOnResolutionAxis:
    def test_1se_picks_finest_within_one_se(self, resolution_results_df):
        # Threshold: 0.90 - 0.02 = 0.88; only resolution 0.25 (0.90) and
        # 0.5 (0.85 -> below) qualify, so the coarsest survives.
        row = select_best_row_1se(resolution_results_df, measure="stability")
        assert row["sweep_value"] == 0.25
        assert row["sweep_rank"] == 0

    def test_1se_agrees_with_k_mode_on_the_same_numbers(
        self, resolution_results_df, results_df
    ):
        """The rule is axis-agnostic: identical metrics select the same rank."""
        k_row = select_best_row_1se(results_df, measure="stability")
        r_row = select_best_row_1se(resolution_results_df, measure="stability")
        assert k_row["sweep_rank"] == r_row["sweep_rank"]

    def test_1se_picks_finer_rank_when_ses_are_wide(self):
        df = with_sweep_cols(
            pd.DataFrame(
                {
                    "estimator": ["LeidenClustering"] * 4,
                    "resolution": [0.25, 0.5, 1.0, 2.0],
                    "ari_stability": [0.90, 0.89, 0.88, 0.50],
                    "ari_stability_se": [0.05, 0.04, 0.03, 0.02],
                }
            ),
            param="resolution",
            method_label="LeidenClustering",
            observed=[2.0, 3.0, 5.0, 9.0],
        )
        # Threshold 0.85; resolutions 0.25/0.5/1.0 qualify, finest is 1.0.
        row = select_best_row_1se(df, measure="stability")
        assert row["sweep_value"] == 1.0
        assert row["sweep_rank"] == 2

    def test_select_best_k_uses_observed_clusters(self, resolution_results_df):
        """There is no n_clusters column; k comes from n_clusters_observed."""
        assert "n_clusters" not in resolution_results_df.columns
        assert select_best_k(resolution_results_df, measure="stability") == 2

    def test_select_best_k_rounds(self):
        df = with_sweep_cols(
            pd.DataFrame(
                {
                    "estimator": ["LeidenClustering"] * 2,
                    "resolution": [0.5, 1.0],
                    "ari_stability": [0.9, 0.5],
                    "ari_stability_se": [0.01, 0.01],
                }
            ),
            param="resolution",
            method_label="LeidenClustering",
            observed=[3.4, 8.6],
        )
        assert select_best_k(df, measure="stability", rule="max") == 3

    def test_not_two_filters_on_observed_k(self, resolution_results_df):
        row = select_best_row_by_rule(
            resolution_results_df, measure="stability", rule="1se", not_two=True
        )
        assert row["n_clusters_observed"] != 2.0
        assert row["sweep_value"] == 0.5

    def test_not_two_exhausted_raises(self):
        df = with_sweep_cols(
            pd.DataFrame(
                {
                    "estimator": ["LeidenClustering"] * 2,
                    "resolution": [0.5, 1.0],
                    "ari_stability": [0.9, 0.8],
                    "ari_stability_se": [0.01, 0.01],
                }
            ),
            param="resolution",
            method_label="LeidenClustering",
            observed=[2.0, 2.4],
        )
        with pytest.raises(ValueError, match="excluding k=2"):
            select_best_row_by_rule(df, measure="stability", rule="max", not_two=True)


class TestSelectionOnInvertedAxis:
    """min_cluster_size is the one axis where larger means coarser."""

    def test_ranks_run_backwards(self, min_cluster_size_results_df):
        np.testing.assert_array_equal(
            min_cluster_size_results_df["sweep_rank"], [2, 1, 0]
        )

    def test_1se_picks_smallest_size_within_one_se(self, min_cluster_size_results_df):
        # Best is size 25 (0.88); threshold 0.88 - 0.02 = 0.86. Size 10
        # (0.85) is below it, so only size 25 qualifies.
        row = select_best_row_1se(min_cluster_size_results_df, measure="stability")
        assert row["min_cluster_size"] == 25

    def test_1se_prefers_finer_rank_not_larger_value(self):
        """With a wide SE the *smallest* size wins, i.e. the finest rank."""
        df = with_sweep_cols(
            pd.DataFrame(
                {
                    "estimator": ["HDBSCAN"] * 3,
                    "min_cluster_size": [5, 10, 25],
                    "ari_stability": [0.86, 0.87, 0.88],
                    "ari_stability_se": [0.02, 0.02, 0.05],
                }
            ),
            param="min_cluster_size",
            method_label="HDBSCAN",
            observed=[9.0, 5.0, 3.0],
        )
        # Threshold 0.83; all three qualify. Finest rank is size 5.
        row = select_best_row_1se(df, measure="stability")
        assert row["min_cluster_size"] == 5
        assert row["sweep_rank"] == 2

    def test_select_best_k_on_inverted_axis(self, min_cluster_size_results_df):
        assert select_best_k(min_cluster_size_results_df, measure="stability") == 3
