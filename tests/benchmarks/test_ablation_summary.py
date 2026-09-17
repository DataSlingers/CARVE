"""Tests for the ablation summaries, on hand-built frames."""

import numpy as np
import pandas as pd
import pytest

from benchmarks._ablation_summary import (
    HEADLINE_METRICS,
    POOLED,
    agreement_summary,
    curve_at_k_star,
    diagnostics_summary,
    metric_mode,
    rare_recall_summary,
    selection_summary,
    similarity_summary,
    spread_summary,
    study_selection_shares,
    table_rows,
)
from benchmarks._registry import CARVE_METRICS_ALL
from benchmarks._run import _labels_mode
from benchmarks._tables import wilson_ci

X = "subsample_ratio"
STAB, GEN = HEADLINE_METRICS


def _key(study="gaussians", difficulty="medium", dataset=0, rho=0.618, b=100, rep=0):
    return {
        "study": study, "difficulty": difficulty, "dataset": dataset,
        "subsample_ratio": rho, "n_resamples": b, "replicate": rep,
    }


def _selection():
    rows = []
    # Two datasets, two replicates, two rho values; stability recovers k*
    # at 0.618 and never at 0.2 (replicate 0 picks 3, replicate 1 picks 4);
    # generalizability always does.
    for dataset in (0, 1):
        for rep in (0, 1):
            for rho in (0.2, 0.618):
                for metric in (STAB, GEN):
                    hit = metric == GEN or rho == 0.618
                    rows.append({
                        **_key(dataset=dataset, rho=rho, rep=rep),
                        "metric_name": metric, "selected_estimator": "KMeans",
                        "selected_k": 5 if hit else 3 + rep, "k_star": 5.0,
                        "ari_selected": 0.9 if hit else 0.5,
                    })
    # A study with two replicates that disagree at 0.2 and agree at 0.618.
    for rep in (0, 1):
        for rho in (0.2, 0.618):
            for metric in (STAB, GEN):
                rows.append({
                    **_key(study="klein", difficulty="", dataset=0, rho=rho, rep=rep),
                    "metric_name": metric, "selected_estimator": "AgglomerativeClustering",
                    "selected_k": 4 if (rho == 0.618 or rep == 0) else 3, "k_star": np.nan,
                    "ari_selected": 0.7,
                })
    return pd.DataFrame(rows)


def _undefined_gini_rows():
    """Task 6's exact row shape: a non-headline metric whose measure column
    was NaN for every configuration of a cell records selected_estimator,
    selected_k and ari_selected as NaN. One setting (0.2) is undefined for
    every replicate; the other (0.618) has one defined replicate and one
    undefined."""
    metric = "consensus_gini_stability"
    rows = [
        {**_key(rho=0.2, rep=0), "metric_name": metric, "selected_estimator": np.nan,
         "selected_k": np.nan, "k_star": 5.0, "ari_selected": np.nan},
        {**_key(rho=0.2, rep=1), "metric_name": metric, "selected_estimator": np.nan,
         "selected_k": np.nan, "k_star": 5.0, "ari_selected": np.nan},
        {**_key(rho=0.618, rep=0), "metric_name": metric, "selected_estimator": "KMeans",
         "selected_k": 5, "k_star": 5.0, "ari_selected": 0.9},
        {**_key(rho=0.618, rep=1), "metric_name": metric, "selected_estimator": np.nan,
         "selected_k": np.nan, "k_star": 5.0, "ari_selected": np.nan},
    ]
    return pd.DataFrame(rows)


class TestMetricMode:
    @pytest.mark.parametrize("metric", CARVE_METRICS_ALL)
    def test_agrees_with_the_runner(self, metric):
        assert metric_mode(metric) == _labels_mode(metric)


class TestSelectionSummary:
    def test_recovery_bias_and_ari_per_study_and_pooled(self):
        out = selection_summary(_selection(), x=X)
        stab_02 = out[(out["study"] == "gaussians") & (out["metric_name"] == STAB) & (out[X] == 0.2)].iloc[0]
        assert stab_02["n"] == 4
        assert stab_02["recovery"] == 0.0
        assert stab_02["bias_mean"] == pytest.approx(-1.5)  # k-hat 3 or 4 minus 5
        assert stab_02["ari_mean"] == pytest.approx(0.5)
        lo, hi = wilson_ci(0, 4)
        assert (stab_02["recovery_lo"], stab_02["recovery_hi"]) == (lo, hi)
        pooled = out[(out["study"] == POOLED) & (out["metric_name"] == GEN) & (out[X] == 0.618)].iloc[0]
        assert pooled["recovery"] == 1.0

    def test_pooled_rows_exclude_the_study(self):
        out = selection_summary(_selection(), x=X)
        assert "klein" not in set(out["study"])
        assert set(out["study"]) == {"gaussians", POOLED}

    def test_columns(self):
        out = selection_summary(_selection(), x=X)
        assert list(out.columns) == [X, "study", "metric_name", "n", "recovery", "recovery_lo",
                                     "recovery_hi", "bias_mean", "ari_mean", "ari_sem"]


class TestAgreementSummary:
    def test_pairs_that_pick_the_same_k_agree(self):
        out = agreement_summary(_selection(), x=X)
        klein = out[out["study"] == "klein"].set_index([X, "metric_name"])["agreement"]
        assert klein.loc[(0.2, STAB)] == 0.0
        assert klein.loc[(0.618, STAB)] == 1.0
        sims = out[out["study"] == "gaussians"].set_index([X, "metric_name"])["agreement"]
        assert sims.loc[(0.2, STAB)] == 0.0  # replicates picked 3 and 4
        assert sims.loc[(0.618, GEN)] == 1.0

    def test_pooled_excludes_the_study(self):
        out = agreement_summary(_selection(), x=X)
        pooled = out[out["study"] == POOLED].set_index([X, "metric_name"])
        assert pooled.loc[(0.2, STAB), "agreement"] == 0.0
        assert pooled.loc[(0.2, STAB), "n_datasets"] == 2

    def test_a_single_replicate_has_no_agreement(self):
        frame = _selection()
        frame = frame[frame["replicate"] == 0]
        out = agreement_summary(frame, x=X)
        assert out["agreement"].isna().all()

    def test_three_replicates_use_every_pair(self):
        rows = [
            {**_key(rep=rep), "metric_name": STAB, "selected_estimator": "KMeans",
             "selected_k": k, "k_star": 5.0, "ari_selected": 0.9}
            for rep, k in enumerate((5, 5, 4))
        ]
        out = agreement_summary(pd.DataFrame(rows), x=X)
        assert out[out["study"] == "gaussians"]["agreement"].iloc[0] == pytest.approx(1 / 3)


class TestUndefinedSelections:
    """An undefined selection (Task 6's all-NaN-measure case) is missing
    data, not a miss: it must not count toward n, must not read as
    "disagreed", and a setting left with no defined selection at all must
    not appear in the output."""

    def test_selection_summary_emits_no_row_for_an_entirely_undefined_setting(self):
        out = selection_summary(
            _undefined_gini_rows(), x=X, metrics=("consensus_gini_stability",)
        )
        assert 0.2 not in set(out[X])

    def test_selection_summary_counts_only_defined_rows_at_a_mixed_setting(self):
        out = selection_summary(
            _undefined_gini_rows(), x=X, metrics=("consensus_gini_stability",)
        )
        mixed = out[(out["study"] == "gaussians") & (out[X] == 0.618)].iloc[0]
        assert mixed["n"] == 1
        assert mixed["recovery"] == 1.0

    def test_selection_summary_all_undefined_frame_does_not_raise(self):
        frame = _undefined_gini_rows()
        frame = frame[frame[X] == 0.2]
        out = selection_summary(frame, x=X, metrics=("consensus_gini_stability",))
        assert out.empty
        assert list(out.columns) == [X, "study", "metric_name", "n", "recovery", "recovery_lo",
                                     "recovery_hi", "bias_mean", "ari_mean", "ari_sem"]

    def test_agreement_summary_is_nan_not_zero_with_fewer_than_two_defined(self):
        out = agreement_summary(
            _undefined_gini_rows(), x=X, metrics=("consensus_gini_stability",)
        )
        mixed = out[(out["study"] == "gaussians") & (out[X] == 0.618)].iloc[0]
        assert pd.isna(mixed["agreement"])

    def test_agreement_summary_all_undefined_frame_does_not_raise(self):
        frame = _undefined_gini_rows()
        frame = frame[frame[X] == 0.2]
        out = agreement_summary(frame, x=X, metrics=("consensus_gini_stability",))
        assert out.empty
        assert list(out.columns) == [X, "study", "metric_name", "agreement", "n_datasets"]

    def test_headline_metrics_are_unaffected(self):
        # The fixture's headline-metric rows never have undefined selections;
        # this class's filter must not change their existing behavior.
        out = selection_summary(_selection(), x=X)
        stab_02 = out[(out["study"] == "gaussians") & (out["metric_name"] == STAB)
                      & (out[X] == 0.2)].iloc[0]
        assert stab_02["n"] == 4


class TestSpreadSummary:
    def test_mean_sd_across_replicates(self):
        # k=3 and k=4 carry different replicate spreads, so the mean over k
        # is only right if both per-k standard deviations were computed
        # (equal per-k spreads by construction could pass "averaged over k"
        # without ever exercising the average).
        rows = [
            {**_key(rep=0), "metric_name": STAB, "estimator": "KMeans", "k": 3,
             "metric_value": 0.5, "metric_se": 0.02},
            {**_key(rep=1), "metric_name": STAB, "estimator": "KMeans", "k": 3,
             "metric_value": 0.7, "metric_se": 0.02},
            {**_key(rep=0), "metric_name": STAB, "estimator": "KMeans", "k": 4,
             "metric_value": 0.4, "metric_se": 0.02},
            {**_key(rep=1), "metric_name": STAB, "estimator": "KMeans", "k": 4,
             "metric_value": 1.0, "metric_se": 0.02},
        ]
        out = spread_summary(pd.DataFrame(rows), x=X)
        sd_k3 = np.std([0.5, 0.7], ddof=1)
        sd_k4 = np.std([0.4, 1.0], ddof=1)
        expected = np.mean([sd_k3, sd_k4])
        assert sd_k3 != pytest.approx(sd_k4)  # guards against a degenerate fixture
        assert out[out["study"] == "gaussians"]["spread"].iloc[0] == pytest.approx(expected)
        assert out[out["study"] == POOLED]["spread"].iloc[0] == pytest.approx(expected)


class TestCurveAtKStar:
    def test_reads_the_value_and_se_at_the_true_k(self):
        curves = pd.DataFrame([
            {**_key(), "metric_name": STAB, "estimator": "KMeans", "k": k,
             "metric_value": 0.1 * k, "metric_se": 0.01 * k}
            for k in (3, 4, 5)
        ])
        datasets = pd.DataFrame([{"study": "gaussians", "difficulty": "medium", "dataset": 0,
                                  "n_samples": 100, "k_star": 5.0, "oracle_ari": 1.0,
                                  "rare_label": 0.0, "rare_fraction": 0.1}])
        out = curve_at_k_star(curves, datasets, x=X)
        row = out[out["study"] == "gaussians"].iloc[0]
        assert row["value_mean"] == pytest.approx(0.5)
        assert row["se_mean"] == pytest.approx(0.05)


class TestRareRecallSummary:
    def test_recall_at_the_selected_k_and_at_k_star(self):
        selection = pd.DataFrame([
            {**_key(), "metric_name": STAB, "selected_estimator": "KMeans", "selected_k": 4,
             "k_star": 5.0, "ari_selected": 0.8},
            {**_key(), "metric_name": GEN, "selected_estimator": "KMeans", "selected_k": 5,
             "k_star": 5.0, "ari_selected": 0.9},
        ])
        at_k = pd.DataFrame([
            {**_key(), "mode": mode, "k": k, "ari_at_k": 0.5,
             "rare_recall_at_k": {4: 0.0, 5: 1.0}[k] if mode == "default" else {4: 0.2, 5: 0.9}[k]}
            for mode in ("default", "generalizability") for k in (4, 5)
        ])
        out = rare_recall_summary(at_k, selection, x=X)
        stab = out[(out["study"] == "gaussians") & (out["metric_name"] == STAB)].iloc[0]
        gen = out[(out["study"] == "gaussians") & (out["metric_name"] == GEN)].iloc[0]
        assert (stab["recall_selected"], stab["recall_k_star"]) == (0.0, 1.0)
        assert (gen["recall_selected"], gen["recall_k_star"]) == (0.9, 0.9)

    def test_difficulty_filter(self):
        selection = pd.DataFrame([
            {**_key(difficulty=d), "metric_name": STAB, "selected_estimator": "KMeans",
             "selected_k": 5, "k_star": 5.0, "ari_selected": 0.8}
            for d in ("medium", "hard")
        ])
        at_k = pd.DataFrame([
            {**_key(difficulty=d), "mode": "default", "k": 5, "ari_at_k": 0.5,
             "rare_recall_at_k": {"medium": 1.0, "hard": 0.0}[d]}
            for d in ("medium", "hard")
        ])
        out = rare_recall_summary(at_k, selection, x=X, metrics=(STAB,), difficulty="hard")
        assert out[out["study"] == "gaussians"]["recall_selected"].iloc[0] == 0.0


class TestSimilaritySummary:
    def test_means_over_datasets_and_draws(self):
        rows = [
            {"study": "gaussians", "difficulty": "medium", "dataset": d, "subsample_ratio": 0.5,
             "estimator": "KMeans", "k": 5, "draw": m, "ari": 0.6 + 0.1 * m}
            for d in (0, 1) for m in (0, 1)
        ]
        out = similarity_summary(pd.DataFrame(rows))
        assert len(out) == 1
        assert out["ari_mean"].iloc[0] == pytest.approx(0.65)
        assert out["n"].iloc[0] == 4

    def test_groups_by_rho_estimator_and_k_separately(self):
        # A single (subsample_ratio, estimator, k) combination cannot tell
        # the documented four-column group key apart from a narrower one
        # that happens to collapse to the same single group.
        rows = []
        for d in (0, 1):
            for m in (0, 1):
                rows.append({"study": "gaussians", "difficulty": "medium", "dataset": d,
                             "subsample_ratio": 0.5, "estimator": "KMeans", "k": 5,
                             "draw": m, "ari": 0.6 + 0.1 * m})
                rows.append({"study": "gaussians", "difficulty": "medium", "dataset": d,
                             "subsample_ratio": 0.8, "estimator": "AgglomerativeClustering",
                             "k": 6, "draw": m, "ari": 0.2 + 0.1 * m})
        out = similarity_summary(pd.DataFrame(rows))
        assert len(out) == 2
        low = out[(out["subsample_ratio"] == 0.5) & (out["estimator"] == "KMeans")
                  & (out["k"] == 5)].iloc[0]
        high = out[(out["subsample_ratio"] == 0.8)
                   & (out["estimator"] == "AgglomerativeClustering") & (out["k"] == 6)].iloc[0]
        assert low["ari_mean"] == pytest.approx(0.65)
        assert high["ari_mean"] == pytest.approx(0.25)
        assert low["n"] == 4 and high["n"] == 4


class TestStudySelectionShares:
    def test_shares_per_setting(self):
        out = study_selection_shares(_selection(), x=X, metric=STAB, study="klein")
        at_02 = out[out[X] == 0.2].set_index("selected_k")["share"]
        assert at_02.loc[4] == 0.5 and at_02.loc[3] == 0.5
        at_default = out[out[X] == 0.618]
        assert len(at_default) == 1 and at_default["share"].iloc[0] == 1.0
        assert set(out["n"]) == {2}


class TestDiagnosticsSummary:
    def test_means_per_setting_and_study(self):
        # fit_seconds and consensus_nan_fraction differ between the two
        # replicates being averaged, so "take first" and "mean" disagree
        # and the assertions can tell them apart.
        cells = pd.DataFrame([
            {**_key(rho=0.2, rep=0), "carve_random_state": 1, "n_samples": 100,
             "fit_seconds": 1.0, "consensus_nan_fraction": 0.08, "n_cluster_count_warnings": 0},
            {**_key(rho=0.2, rep=1), "carve_random_state": 1, "n_samples": 100,
             "fit_seconds": 3.0, "consensus_nan_fraction": 0.12, "n_cluster_count_warnings": 1},
            {**_key(rho=0.618, rep=0), "carve_random_state": 1, "n_samples": 100,
             "fit_seconds": 6.0, "consensus_nan_fraction": 0.0, "n_cluster_count_warnings": 0},
            {**_key(rho=0.618, rep=1), "carve_random_state": 1, "n_samples": 100,
             "fit_seconds": 6.2, "consensus_nan_fraction": 0.02, "n_cluster_count_warnings": 1},
        ])
        out = diagnostics_summary(cells, x=X)
        row = out[out[X] == 0.2].iloc[0]
        assert row["fit_seconds"] == pytest.approx(2.0)  # mean of 1.0 and 3.0, not 1.0
        assert row["consensus_nan_fraction"] == pytest.approx(0.10)  # mean of 0.08 and 0.12
        assert row["n_cluster_count_warnings"] == pytest.approx(0.5)


class TestTableRows:
    def test_one_row_per_setting_and_metric_with_study_mode(self):
        view = {"selection": _selection()}
        rows = table_rows(view, x=X, study="klein")
        assert set(rows["setting"]) == {0.2, 0.618}
        assert set(rows["metric_name"]) == set(HEADLINE_METRICS)
        row = rows[(rows["setting"] == 0.618) & (rows["metric_name"] == STAB)].iloc[0]
        assert row["recovery"] == 1.0
        assert row["study_modal"] == "AgglomerativeClustering, k=4"
        assert row["study_share"] == 1.0
        assert row["agreement"] == 1.0
