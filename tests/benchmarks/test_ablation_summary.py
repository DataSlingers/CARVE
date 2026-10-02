"""Tests for the ablation summaries, on hand-built frames."""

import numpy as np
import pandas as pd
import pytest

from benchmarks._ablation_summary import (
    HEADLINE_METRICS,
    POOLED,
    agreement_summary,
    ari_summary,
    bias_summary,
    curve_at_k_star,
    diagnostics_summary,
    rare_recall_summary,
    similarity_at_k_star,
    similarity_summary,
    spread_summary,
    study_ari_summary,
    study_selection_shares,
    table_rows,
)
from benchmarks._artifacts import ABLATION_AT_K_SCHEMA, ABLATION_SELECTION_SCHEMA

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
    """The row shape _ablation.cell_rows writes when a non-headline metric's
    measure column was NaN for every configuration of a cell:
    selected_estimator, selected_k and ari_selected are NaN. One setting
    (0.2) is undefined for every replicate; the other (0.618) has one
    defined replicate and one undefined."""
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


def _ari_at(k, dataset):
    """The stability-consensus ARI _at_k writes at k: distinct per k and per
    dataset, and never equal to a selection's ari_selected."""
    return 0.1 * k + 0.02 * dataset


def _at_k(selection, ks=(3, 4, 5, 6, 7)):
    """at_k rows for every simulated cell of a selection frame. The
    generalizability mode carries a constant 0.99, so a summary that read
    the wrong mode is caught."""
    cells = selection[selection["difficulty"] != ""][list(_key())].drop_duplicates()
    rows = [
        {**cell, "mode": mode, "k": k,
         "ari_at_k": _ari_at(k, cell["dataset"]) if mode == "default" else 0.99,
         "rare_recall_at_k": 1.0}
        for cell in cells.to_dict("records")
        for mode in ("default", "generalizability")
        for k in ks
    ]
    return pd.DataFrame(rows)


ARI_COLUMNS = [X, "study", "metric_name", "ari_mean", "ari_sem", "n_datasets"]


class TestAriSummary:
    def test_reads_the_stability_consensus_ari_at_the_selected_k(self):
        # Generalizability selects k=5 on both datasets at 0.618. Its labels
        # are the stability consensus, so the ARI is the default-mode value
        # there (0.50 and 0.52), not the stored ari_selected (0.9) and not
        # the generalizability-mode value (0.99).
        selection = _selection()
        out = ari_summary(selection, _at_k(selection), x=X)
        gen = out[(out["study"] == "gaussians") & (out["metric_name"] == GEN) & (out[X] == 0.618)].iloc[0]
        assert gen["ari_mean"] == pytest.approx(0.51)
        assert gen["n_datasets"] == 2

    def test_replicates_are_averaged_within_a_dataset_first(self):
        # Stability at 0.2 selects k=3 (replicate 0) and k=4 (replicate 1):
        # 0.30 and 0.40 on dataset 0, 0.32 and 0.42 on dataset 1. The
        # dataset means are 0.35 and 0.37, so the standard error is over
        # those two, not over the four fits.
        selection = _selection()
        out = ari_summary(selection, _at_k(selection), x=X)
        stab = out[(out["study"] == "gaussians") & (out["metric_name"] == STAB) & (out[X] == 0.2)].iloc[0]
        assert stab["ari_mean"] == pytest.approx(0.36)
        assert stab["ari_sem"] == pytest.approx(np.std([0.35, 0.37], ddof=1) / np.sqrt(2))
        assert stab["ari_sem"] != pytest.approx(np.std([0.30, 0.40, 0.32, 0.42], ddof=1) / 2)

    def test_pooled_rows_exclude_the_study(self):
        selection = _selection()
        out = ari_summary(selection, _at_k(selection), x=X)
        assert set(out["study"]) == {"gaussians", POOLED}
        pooled = out[(out["study"] == POOLED) & (out["metric_name"] == GEN) & (out[X] == 0.618)].iloc[0]
        assert pooled["ari_mean"] == pytest.approx(0.51)

    def test_columns(self):
        selection = _selection()
        assert list(ari_summary(selection, _at_k(selection), x=X).columns) == ARI_COLUMNS


BIAS_COLUMNS = [X, "study", "metric_name", "bias_mean", "bias_sem", "n_datasets"]


class TestBiasSummary:
    def test_replicates_are_averaged_within_a_dataset_first(self):
        # Stability at 0.2 selects k=3 (replicate 0) and k=4 (replicate 1)
        # against k* = 5 on both datasets: -2 and -1, so each dataset's
        # mean is -1.5 and the standard error over the two datasets is 0.
        # Over the four fits it would be 0.29.
        out = bias_summary(_selection(), x=X)
        stab = out[(out["study"] == "gaussians") & (out["metric_name"] == STAB) & (out[X] == 0.2)].iloc[0]
        assert stab["bias_mean"] == pytest.approx(-1.5)
        assert stab["bias_sem"] == pytest.approx(0.0)
        assert stab["n_datasets"] == 2
        gen = out[(out["study"] == "gaussians") & (out["metric_name"] == GEN)]
        assert (gen["bias_mean"] == 0.0).all()

    def test_sem_is_over_datasets(self):
        rows = pd.DataFrame([
            {**_key(dataset=d), "metric_name": STAB, "selected_estimator": "KMeans",
             "selected_k": k, "k_star": 5.0, "ari_selected": 0.9}
            for d, k in ((0, 3), (1, 5), (2, 6))
        ])
        out = bias_summary(rows, x=X, metrics=(STAB,))
        pooled = out[out["study"] == POOLED].iloc[0]
        assert pooled["bias_mean"] == pytest.approx(-1 / 3)
        assert pooled["bias_sem"] == pytest.approx(np.std([-2, 0, 1], ddof=1) / np.sqrt(3))

    def test_pooled_rows_exclude_the_study(self):
        out = bias_summary(_selection(), x=X)
        assert set(out["study"]) == {"gaussians", POOLED}

    def test_undefined_selections_are_dropped(self):
        out = bias_summary(_undefined_gini_rows(), x=X, metrics=("consensus_gini_stability",))
        assert 0.2 not in set(out[X])
        mixed = out[(out["study"] == "gaussians") & (out[X] == 0.618)].iloc[0]
        assert mixed["n_datasets"] == 1
        assert mixed["bias_mean"] == 0.0

    def test_all_undefined_frame_does_not_raise(self):
        frame = _undefined_gini_rows()
        frame = frame[frame[X] == 0.2]
        out = bias_summary(frame, x=X, metrics=("consensus_gini_stability",))
        assert out.empty
        assert list(out.columns) == BIAS_COLUMNS

    def test_columns(self):
        assert list(bias_summary(_selection(), x=X).columns) == BIAS_COLUMNS


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
    """An undefined selection (cell_rows' all-NaN-measure case) is missing
    data, not a miss: it must not count toward n, must not read as
    "disagreed", and a setting left with no defined selection at all must
    not appear in the output."""

    def test_ari_summary_emits_no_row_for_an_entirely_undefined_setting(self):
        rows = _undefined_gini_rows()
        out = ari_summary(rows, _at_k(rows), x=X, metrics=("consensus_gini_stability",))
        assert 0.2 not in set(out[X])

    def test_ari_summary_reads_only_defined_rows_at_a_mixed_setting(self):
        # At 0.618 replicate 0 selected k=5 and replicate 1 is undefined:
        # the dataset's ARI is replicate 0's alone, not averaged with a zero.
        rows = _undefined_gini_rows()
        out = ari_summary(rows, _at_k(rows), x=X, metrics=("consensus_gini_stability",))
        mixed = out[(out["study"] == "gaussians") & (out[X] == 0.618)].iloc[0]
        assert mixed["n_datasets"] == 1
        assert mixed["ari_mean"] == pytest.approx(_ari_at(5, 0))

    def test_ari_summary_all_undefined_frame_does_not_raise(self):
        frame = _undefined_gini_rows()
        frame = frame[frame[X] == 0.2]
        out = ari_summary(frame, _at_k(frame), x=X, metrics=("consensus_gini_stability",))
        assert out.empty
        assert list(out.columns) == ARI_COLUMNS

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
        selection = _selection()
        out = ari_summary(selection, _at_k(selection), x=X)
        stab_02 = out[(out["study"] == "gaussians") & (out["metric_name"] == STAB)
                      & (out[X] == 0.2)].iloc[0]
        assert stab_02["n_datasets"] == 2

    def test_study_selection_shares_count_only_defined_rows(self):
        # At 0.618 one replicate is defined and one undefined: n is 1 and
        # the one share is 1.0, not n = 2 and a share of 0.5 that would
        # leave the setting's shares summing below one. The all-undefined
        # setting emits no row. selected_k comes back as an int although
        # the input column is float64 (the NaN rows force that dtype, as
        # read_frames does on a run with one undefined cell): the figure
        # labels bars from it and must not read "k=5.0".
        rows = _undefined_gini_rows()
        assert rows["selected_k"].dtype.kind == "f"
        out = study_selection_shares(
            rows, x=X, metric="consensus_gini_stability", study="gaussians"
        )
        assert set(out[X]) == {0.618}
        assert len(out) == 1
        assert out["n"].iloc[0] == 1
        assert out["count"].iloc[0] == 1
        assert out["share"].iloc[0] == 1.0
        assert out["selected_k"].dtype.kind == "i"
        assert out["selected_k"].iloc[0] == 5


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
    def test_recall_at_the_selected_k_and_at_k_star_on_stability_labels(self):
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
        # The generalizability rule picked k=5; its labels are still the
        # stability consensus, whose recall there is 1.0, not 0.9.
        assert (gen["recall_selected"], gen["recall_k_star"]) == (1.0, 1.0)

    def test_sems_are_over_the_datasets(self):
        # Two datasets: recall at the selected k is 0 and 1, at k* 1 and 1.
        selection = pd.DataFrame([
            {**_key(dataset=d), "metric_name": STAB, "selected_estimator": "KMeans",
             "selected_k": 4 if d == 0 else 5, "k_star": 5.0, "ari_selected": 0.8}
            for d in (0, 1)
        ])
        at_k = pd.DataFrame([
            {**_key(dataset=d), "mode": "default", "k": k, "ari_at_k": 0.5,
             "rare_recall_at_k": {4: 0.0, 5: 1.0}[k]}
            for d in (0, 1) for k in (4, 5)
        ])
        out = rare_recall_summary(at_k, selection, x=X, metrics=(STAB,))
        pooled = out[out["study"] == POOLED].iloc[0]
        assert pooled["recall_selected_sem"] == pytest.approx(np.std([0.0, 1.0], ddof=1) / np.sqrt(2))
        assert pooled["recall_k_star_sem"] == 0.0

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


class TestSimilarityAtKStar:
    @staticmethod
    def _frames():
        """Two datasets with k* = 5 and a study without one. Draws differ,
        and k=4 rows carry a value the summary must not read."""
        rows = []
        for d in (0, 1):
            for rho in (0.5, 1.0):
                for k in (4, 5):
                    for m in (0, 1):
                        ari = 0.0 if k == 4 else 0.6 + 0.2 * d + 0.1 * m + (0.1 if rho == 1.0 else 0.0)
                        rows.append({"study": "gaussians", "difficulty": "medium", "dataset": d,
                                     "subsample_ratio": rho, "estimator": "KMeans", "k": k,
                                     "draw": m, "ari": ari})
        for k in (4, 5):
            rows.append({"study": "klein", "difficulty": "", "dataset": 0, "subsample_ratio": 0.5,
                         "estimator": "AgglomerativeClustering", "k": k, "draw": 0, "ari": 0.1})
        datasets = pd.DataFrame([
            {"study": "gaussians", "difficulty": "medium", "dataset": d, "n_samples": 100,
             "k_star": 5.0, "oracle_ari": 0.9, "rare_label": 0.0, "rare_fraction": 0.1}
            for d in (0, 1)
        ] + [{"study": "klein", "difficulty": "", "dataset": 0, "n_samples": 100,
              "k_star": np.nan, "oracle_ari": np.nan, "rare_label": np.nan,
              "rare_fraction": np.nan}])
        return pd.DataFrame(rows), datasets

    def test_reads_k_star_and_averages_draws_within_a_dataset_first(self):
        # Dataset means at 0.5 are 0.65 and 0.85: mean 0.75, and the
        # standard error is over the two datasets, not the four draws.
        similarity, datasets = self._frames()
        out = similarity_at_k_star(similarity, datasets)
        row = out[(out["study"] == POOLED) & (out["subsample_ratio"] == 0.5)].iloc[0]
        assert row["ari_mean"] == pytest.approx(0.75)
        assert row["ari_sem"] == pytest.approx(np.std([0.65, 0.85], ddof=1) / np.sqrt(2))
        assert row["n_datasets"] == 2

    def test_keeps_the_refit_reference(self):
        similarity, datasets = self._frames()
        out = similarity_at_k_star(similarity, datasets)
        refit = out[(out["study"] == POOLED) & (out["subsample_ratio"] == 1.0)].iloc[0]
        assert refit["ari_mean"] == pytest.approx(0.85)

    def test_the_study_has_no_k_star_and_drops_out(self):
        similarity, datasets = self._frames()
        out = similarity_at_k_star(similarity, datasets)
        assert set(out["study"]) == {"gaussians", POOLED}
        assert list(out.columns) == ["subsample_ratio", "study", "ari_mean", "ari_sem", "n_datasets"]


class TestStudySelectionShares:
    def test_shares_per_setting(self):
        out = study_selection_shares(_selection(), x=X, metric=STAB, study="klein")
        at_02 = out[out[X] == 0.2].set_index("selected_k")["share"]
        assert at_02.loc[4] == 0.5 and at_02.loc[3] == 0.5
        at_default = out[out[X] == 0.618]
        assert len(at_default) == 1 and at_default["share"].iloc[0] == 1.0
        assert set(out["n"]) == {2}


class TestStudyAriSummary:
    def test_mean_and_sem_over_replicates(self):
        # The fixture's Klein ari_selected is a constant 0.7; mutate one
        # replicate's value so the mean and standard error can actually
        # move (and so a broken implementation that ignored replicate 1
        # could fail this test).
        selection = _selection().copy()
        mask = (
            (selection["study"] == "klein")
            & (selection[X] == 0.618)
            & (selection["replicate"] == 1)
            & (selection["metric_name"] == STAB)
        )
        assert mask.sum() == 1
        selection.loc[mask, "ari_selected"] = 0.9

        out = study_ari_summary(selection, x=X, study="klein", metrics=HEADLINE_METRICS)
        assert list(out.columns) == [X, "metric_name", "ari_mean", "ari_sem", "n"]
        assert len(out) == 4  # two rho values x two headline metrics

        moved = out[(out[X] == 0.618) & (out["metric_name"] == STAB)].iloc[0]
        assert moved["ari_mean"] == pytest.approx(0.8)
        assert moved["ari_sem"] == pytest.approx(0.1)
        assert moved["n"] == 2

        unchanged = out[(out[X] == 0.618) & (out["metric_name"] == GEN)].iloc[0]
        assert unchanged["ari_mean"] == pytest.approx(0.7)
        assert unchanged["ari_sem"] == pytest.approx(0.0)
        assert unchanged["n"] == 2

    def test_excludes_other_studies(self):
        # The gaussians rows share the x set {0.2, 0.618} with klein, so
        # the settings alone cannot tell a filtered summary from an
        # unfiltered one. n and the mean can: klein has two replicates at
        # 0.7 per (x, metric); without the study filter each row would also
        # count the four gaussians rows at 0.5 or 0.9.
        out = study_ari_summary(_selection(), x=X, study="klein", metrics=HEADLINE_METRICS)
        assert set(out[X]) == {0.2, 0.618}
        assert len(out) == 4
        assert (out["n"] == 2).all()
        assert out["ari_mean"].tolist() == pytest.approx([0.7] * 4)

    def test_drops_undefined_selections_rather_than_counting_them(self):
        selection = _selection().copy()
        extra = selection[
            (selection["study"] == "klein")
            & (selection[X] == 0.618)
            & (selection["replicate"] == 0)
            & (selection["metric_name"] == STAB)
        ].copy()
        extra["replicate"] = 2
        extra["ari_selected"] = np.nan
        selection = pd.concat([selection, extra], ignore_index=True)

        out = study_ari_summary(selection, x=X, study="klein", metrics=HEADLINE_METRICS)
        row = out[(out[X] == 0.618) & (out["metric_name"] == STAB)].iloc[0]
        assert row["n"] == 2
        assert row["ari_mean"] == pytest.approx(0.7)


class TestDiagnosticsSummary:
    def test_means_per_setting_and_study(self):
        # fit_seconds and both NaN fractions differ between the two
        # replicates being averaged, so "take first" and "mean" disagree
        # and the assertions can tell them apart. The generalizability
        # fraction is set well above the stability one, as it is on a real
        # fit, so a summary that read the wrong column would also fail.
        cells = pd.DataFrame([
            {**_key(rho=0.2, rep=0), "carve_random_state": 1, "n_samples": 100,
             "fit_seconds": 1.0, "consensus_nan_fraction": 0.08,
             "consensus_generalizability_nan_fraction": 0.30, "n_cluster_count_warnings": 0},
            {**_key(rho=0.2, rep=1), "carve_random_state": 1, "n_samples": 100,
             "fit_seconds": 3.0, "consensus_nan_fraction": 0.12,
             "consensus_generalizability_nan_fraction": 0.40, "n_cluster_count_warnings": 1},
            {**_key(rho=0.618, rep=0), "carve_random_state": 1, "n_samples": 100,
             "fit_seconds": 6.0, "consensus_nan_fraction": 0.0,
             "consensus_generalizability_nan_fraction": 0.0, "n_cluster_count_warnings": 0},
            {**_key(rho=0.618, rep=1), "carve_random_state": 1, "n_samples": 100,
             "fit_seconds": 6.2, "consensus_nan_fraction": 0.02,
             "consensus_generalizability_nan_fraction": 0.04, "n_cluster_count_warnings": 1},
        ])
        out = diagnostics_summary(cells, x=X)
        assert list(out.columns) == [
            X, "study", "fit_seconds", "consensus_nan_fraction",
            "consensus_generalizability_nan_fraction", "n_cluster_count_warnings",
        ]
        row = out[out[X] == 0.2].iloc[0]
        assert row["fit_seconds"] == pytest.approx(2.0)  # mean of 1.0 and 3.0, not 1.0
        assert row["consensus_nan_fraction"] == pytest.approx(0.10)  # mean of 0.08 and 0.12
        assert row["consensus_generalizability_nan_fraction"] == pytest.approx(0.35)
        assert row["n_cluster_count_warnings"] == pytest.approx(0.5)


TABLE_COLUMNS = ["setting", "metric_name", "ari_mean", "agreement", "study_modal", "study_share"]


class TestTableRows:
    def test_one_row_per_setting_and_metric_with_study_mode(self):
        selection = _selection()
        rows = table_rows({"selection": selection, "at_k": _at_k(selection)}, x=X, study="klein")
        assert list(rows.columns) == TABLE_COLUMNS
        assert set(rows["setting"]) == {0.2, 0.618}
        assert set(rows["metric_name"]) == set(HEADLINE_METRICS)
        row = rows[(rows["setting"] == 0.618) & (rows["metric_name"] == STAB)].iloc[0]
        # Stability selects k=5 on both datasets: 0.50 and 0.52.
        assert row["ari_mean"] == pytest.approx(0.51)
        assert row["study_modal"] == "AgglomerativeClustering, k=4"
        assert row["study_share"] == 1.0
        assert row["agreement"] == 1.0

    def test_empty_selection_frame_returns_an_empty_frame_with_documented_columns(self):
        # A schema-shaped, zero-row selection frame is what read_frames
        # returns before any checkpoint for that arm exists (a fresh or
        # interrupted run). study_selection_shares then returns empty for
        # every metric, leaving modal == [] for the merge; table_rows must
        # not raise KeyError building that merge's frame.
        view = {
            "selection": pd.DataFrame(columns=list(ABLATION_SELECTION_SCHEMA)),
            "at_k": pd.DataFrame(columns=list(ABLATION_AT_K_SCHEMA)),
        }
        out = table_rows(view, x=X, study="klein")
        assert list(out.columns) == TABLE_COLUMNS
        assert out.empty

    def test_simulated_only_frame_has_no_row_for_the_requested_study(self):
        # Every row belongs to a simulated study ("gaussians"); none belong
        # to the requested study ("klein"). study_selection_shares(study=
        # "klein") is then empty for every metric (same root cause as the
        # fully-empty case), but ari_summary/agreement_summary still have
        # real pooled data from "gaussians" to report: the correct result
        # keeps those (setting, metric) rows with study_modal and
        # study_share as NaN, rather than being empty outright.
        selection = pd.DataFrame([
            {**_key(rho=rho, rep=rep), "metric_name": metric, "selected_estimator": "KMeans",
             "selected_k": 5, "k_star": 5.0, "ari_selected": 0.9}
            for rho in (0.2, 0.618) for rep in (0, 1) for metric in (STAB, GEN)
        ])
        out = table_rows({"selection": selection, "at_k": _at_k(selection)}, x=X, study="klein")
        assert list(out.columns) == TABLE_COLUMNS
        assert not out.empty
        assert set(out["setting"]) == {0.2, 0.618}
        assert out["study_modal"].isna().all()
        assert out["study_share"].isna().all()
        # The pooled headline numbers are unaffected by the missing study.
        row = out[(out["setting"] == 0.618) & (out["metric_name"] == STAB)].iloc[0]
        assert row["ari_mean"] == pytest.approx(_ari_at(5, 0))
