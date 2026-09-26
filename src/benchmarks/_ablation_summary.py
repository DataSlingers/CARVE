"""Per-setting summaries of the ablation frames.

Pure pandas over the frames read_frames returns, so the figures, the table
and the notebook compute nothing themselves. x names the swept column,
"subsample_ratio" for the rho arm and "n_resamples" for the B arm. Rows
with study == POOLED pool the simulated studies; the case study has
difficulty "" and is never pooled with them.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._artifacts import CELL_KEY
from ._registry import GENERALIZABILITY_METRICS
from ._tables import wilson_ci

HEADLINE_METRICS: tuple[str, ...] = ("ari_stability_1se", "ari_generalizability_1se")
POOLED: str = "pooled"
DATASET_KEY: tuple[str, ...] = ("study", "difficulty", "dataset")


def metric_mode(metric_name: str) -> str:
    """Which consensus matrix a metric's labels are cut from; mirrors _run._labels_mode."""
    return "generalizability" if metric_name in GENERALIZABILITY_METRICS else "default"


def _is_simulation(frame: pd.DataFrame) -> pd.Series:
    return frame["difficulty"] != ""


def _with_pooled(
    frame: pd.DataFrame, group: list[str], agg: dict, *, x: str
) -> pd.DataFrame:
    """Aggregate per (x, study, ...) and append pooled rows over the simulations."""
    per = frame.groupby(group, as_index=False).agg(**agg)
    sims = frame[_is_simulation(frame)]
    pooled_group = [g for g in group if g != "study"]
    pooled = sims.groupby(pooled_group, as_index=False).agg(**agg)
    pooled.insert(group.index("study"), "study", POOLED)
    return pd.concat([per, pooled], ignore_index=True)


def selection_summary(
    selection: pd.DataFrame, *, x: str, metrics: Sequence[str] = HEADLINE_METRICS
) -> pd.DataFrame:
    """k* recovery with Wilson bounds, and mean and standard error of the
    bias and of the ARI of the selected labels.

    Simulations only: the study has no true k. An undefined selection
    (_ablation.cell_rows records selected_k as NaN when a metric's measure
    column is NaN for every configuration of a cell) is missing data, not a
    miss: it is dropped before hit/bias are computed, so it does not count
    toward n or drag recovery down, and a group left with no defined
    selections at all emits no row.
    """
    rows = selection[
        selection["k_star"].notna()
        & selection["selected_k"].notna()
        & selection["metric_name"].isin(metrics)
    ].copy()
    rows["hit"] = (rows["selected_k"] == rows["k_star"]).astype(float)
    rows["bias"] = rows["selected_k"] - rows["k_star"]
    out = _with_pooled(
        rows,
        [x, "study", "metric_name"],
        {
            "n": ("hit", "size"),
            "hits": ("hit", "sum"),
            "bias_mean": ("bias", "mean"),
            "bias_sem": ("bias", "sem"),
            "ari_mean": ("ari_selected", "mean"),
            "ari_sem": ("ari_selected", "sem"),
        },
        x=x,
    )
    out["recovery"] = out["hits"] / out["n"]
    bounds = [wilson_ci(int(h), int(n)) for h, n in zip(out["hits"], out["n"])]
    out["recovery_lo"] = [lo for lo, _ in bounds]
    out["recovery_hi"] = [hi for _, hi in bounds]
    return out[
        [
            x,
            "study",
            "metric_name",
            "n",
            "recovery",
            "recovery_lo",
            "recovery_hi",
            "bias_mean",
            "bias_sem",
            "ari_mean",
            "ari_sem",
        ]
    ]


def _pair_agreement(choices: pd.Series) -> float:
    n = len(choices)
    if n < 2:
        return np.nan
    counts = choices.value_counts().to_numpy(dtype=float)
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def agreement_summary(
    selection: pd.DataFrame, *, x: str, metrics: Sequence[str] = HEADLINE_METRICS
) -> pd.DataFrame:
    """Fraction of replicate pairs that select the same (estimator, k), per
    dataset, then averaged over datasets. NaN with a single replicate.

    A row whose selection is undefined (selected_estimator or selected_k is
    NaN, per cell_rows' all-NaN-measure case) is dropped before choices are
    built: it never selected anything, so it neither agrees nor disagrees.
    Casting both sides to str before concatenating (rather than relying on
    selected_estimator already being an object column of strings) keeps this
    from raising when a metric's selected_estimator is NaN for every row
    passed in, which leaves the column float64 even after filtering to zero
    rows.
    """
    rows = selection[selection["metric_name"].isin(metrics)].copy()
    rows = rows[rows["selected_estimator"].notna() & rows["selected_k"].notna()]
    rows["choice"] = (
        rows["selected_estimator"].astype(str) + "@" + rows["selected_k"].astype(str)
    )
    per_dataset = (
        rows.groupby([x, *DATASET_KEY, "metric_name"], as_index=False)["choice"]
        .agg(_pair_agreement)
        .rename(columns={"choice": "agreement"})
    )
    # On a fully empty input (every selection in a group undefined), the
    # groupby-apply above has nothing to call _pair_agreement on and keeps
    # "agreement"'s dtype from "choice" (a string column) instead of the
    # float _pair_agreement actually returns; force it back to float so the
    # downstream mean in _with_pooled does not raise on an all-undefined
    # group.
    per_dataset["agreement"] = per_dataset["agreement"].astype(float)
    return _with_pooled(
        per_dataset,
        [x, "study", "metric_name"],
        {"agreement": ("agreement", "mean"), "n_datasets": ("agreement", "size")},
        x=x,
    )


def spread_summary(
    curves: pd.DataFrame, *, x: str, metrics: Sequence[str] = HEADLINE_METRICS
) -> pd.DataFrame:
    """Standard deviation of a metric across replicates at each (dataset,
    estimator, k), averaged over those."""
    rows = curves[curves["metric_name"].isin(metrics)]
    sd = (
        rows.groupby(
            [x, *DATASET_KEY, "metric_name", "estimator", "k"], as_index=False
        )["metric_value"]
        .std(ddof=1)
        .rename(columns={"metric_value": "sd"})
    )
    return _with_pooled(
        sd, [x, "study", "metric_name"], {"spread": ("sd", "mean")}, x=x
    )


def curve_at_k_star(
    curves: pd.DataFrame,
    datasets: pd.DataFrame,
    *,
    x: str,
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """Mean metric value and mean reported standard error at the true k."""
    merged = curves.merge(datasets[[*DATASET_KEY, "k_star"]], on=list(DATASET_KEY))
    rows = merged[
        (merged["k"] == merged["k_star"]) & merged["metric_name"].isin(metrics)
    ]
    return _with_pooled(
        rows,
        [x, "study", "metric_name"],
        {
            "value_mean": ("metric_value", "mean"),
            "value_sem": ("metric_value", "sem"),
            "se_mean": ("metric_se", "mean"),
        },
        x=x,
    )


def rare_recall_summary(
    at_k: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    x: str,
    metrics: Sequence[str] = HEADLINE_METRICS,
    difficulty: str | None = None,
) -> pd.DataFrame:
    """Recall of the smallest true cluster at the selected k and at k*, with
    the standard error of each mean.

    Each metric reads the at_k rows of its own consensus mode. difficulty
    restricts to one axis label (the hard setting is where rare clusters
    are smallest).
    """
    key = list(CELL_KEY)
    parts = []
    for metric in metrics:
        sel = selection[selection["metric_name"] == metric]
        sel = sel[sel["k_star"].notna()][key + ["selected_k", "k_star"]].copy()
        sel["k_star"] = sel["k_star"].astype(int)
        at = at_k[at_k["mode"] == metric_mode(metric)][key + ["k", "rare_recall_at_k"]]
        at_selected = sel.merge(at, left_on=key + ["selected_k"], right_on=key + ["k"])
        at_star = sel.merge(at, left_on=key + ["k_star"], right_on=key + ["k"])
        frame = (
            at_selected[key + ["rare_recall_at_k"]]
            .rename(columns={"rare_recall_at_k": "recall_selected"})
            .merge(
                at_star[key + ["rare_recall_at_k"]].rename(
                    columns={"rare_recall_at_k": "recall_k_star"}
                ),
                on=key,
            )
        )
        frame["metric_name"] = metric
        parts.append(frame)
    rows = pd.concat(parts, ignore_index=True)
    if difficulty is not None:
        rows = rows[rows["difficulty"] == difficulty]
    return _with_pooled(
        rows,
        [x, "study", "metric_name"],
        {
            "recall_selected": ("recall_selected", "mean"),
            "recall_selected_sem": ("recall_selected", "sem"),
            "recall_k_star": ("recall_k_star", "mean"),
            "recall_k_star_sem": ("recall_k_star", "sem"),
        },
        x=x,
    )


def similarity_summary(similarity: pd.DataFrame) -> pd.DataFrame:
    """Mean subsample-versus-full ARI per (rho, study, estimator, k)."""
    return similarity.groupby(
        ["subsample_ratio", "study", "estimator", "k"], as_index=False
    ).agg(ari_mean=("ari", "mean"), ari_sem=("ari", "sem"), n=("ari", "size"))


def study_selection_shares(
    selection: pd.DataFrame, *, x: str, metric: str, study: str
) -> pd.DataFrame:
    """Share of replicates selecting each (estimator, k) at each setting.

    Undefined selections (cell_rows' all-NaN-measure rows) are missing data,
    as in the other summaries: they are dropped before counting, so n is
    the number of defined replicates and the shares at a setting sum to
    one. selected_k comes back as an int. One undefined row anywhere in a
    run makes the frame's selected_k column float64 (read_frames
    concatenates that cell's file with the int64 ones), and the figure's
    "k=4" labels and the table's modal choice must not read "k=4.0".
    """
    rows = selection[
        (selection["study"] == study) & (selection["metric_name"] == metric)
    ]
    rows = rows[rows["selected_estimator"].notna() & rows["selected_k"].notna()]
    counts = rows.groupby(
        [x, "selected_estimator", "selected_k"], as_index=False
    ).size()
    counts = counts.rename(columns={"size": "count"})
    counts["selected_k"] = counts["selected_k"].astype(int)
    totals = rows.groupby(x, as_index=False).size().rename(columns={"size": "n"})
    out = counts.merge(totals, on=x)
    out["share"] = out["count"] / out["n"]
    return out[[x, "selected_estimator", "selected_k", "count", "n", "share"]]


def study_ari_summary(
    selection: pd.DataFrame,
    *,
    x: str,
    study: str,
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """Mean, standard error and count of the selected labels' ARI to a
    study's reference labels, one row per (x, metric).

    Restricted to one study rather than pooled: unlike the simulations,
    a case study's replicates all score against the same reference
    labels, so there is nothing to pool over. An undefined selection
    (ari_selected NaN, cell_rows' all-NaN-measure case) is dropped before
    aggregating rather than counted as a zero.
    """
    rows = selection[
        (selection["study"] == study) & selection["metric_name"].isin(metrics)
    ]
    rows = rows[rows["ari_selected"].notna()]
    return rows.groupby([x, "metric_name"], as_index=False).agg(
        ari_mean=("ari_selected", "mean"),
        ari_sem=("ari_selected", "sem"),
        n=("ari_selected", "size"),
    )


def diagnostics_summary(cells: pd.DataFrame, *, x: str) -> pd.DataFrame:
    """Mean fit time, NaN fraction of each consensus matrix and cluster-count
    warnings per setting and study.

    The two NaN fractions are reported side by side because they diverge
    where the study looks: the generalizability consensus, cut for the
    generalizability-mode labels, is a third undefined at rho 0.9 and B
    100 while the stability one is nearly full.
    """
    return cells.groupby([x, "study"], as_index=False).agg(
        fit_seconds=("fit_seconds", "mean"),
        consensus_nan_fraction=("consensus_nan_fraction", "mean"),
        consensus_generalizability_nan_fraction=(
            "consensus_generalizability_nan_fraction",
            "mean",
        ),
        n_cluster_count_warnings=("n_cluster_count_warnings", "mean"),
    )


def table_rows(
    view: dict[str, pd.DataFrame],
    *,
    x: str,
    study: str | None,
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """One row per (setting, metric) for the SI table: pooled recovery with
    its Wilson interval, pooled ARI, pooled replicate agreement, and the
    study's modal selection with its share of replicates. With study=None
    (an ablation without a case study) the modal columns are left empty.

    A partial run (an empty selection frame, or one with no rows for the
    named study) leaves modal empty; declaring the merge frame's columns
    up front keeps the [x, "metric_name"] merge keys present so this
    returns a zero-row frame with the documented columns instead of
    raising KeyError.
    """
    selection = view["selection"]
    pooled = selection_summary(selection, x=x, metrics=metrics)
    pooled = pooled[pooled["study"] == POOLED]
    agreement = agreement_summary(selection, x=x, metrics=metrics)
    agreement = agreement[agreement["study"] == POOLED][[x, "metric_name", "agreement"]]
    rows = pooled.merge(agreement, on=[x, "metric_name"], how="left")
    modal = []
    for metric in metrics if study is not None else ():
        shares = study_selection_shares(selection, x=x, metric=metric, study=study)
        for setting, group in shares.groupby(x):
            top = group.sort_values(
                ["share", "selected_k"], ascending=[False, True]
            ).iloc[0]
            modal.append(
                {
                    x: setting,
                    "metric_name": metric,
                    "study_modal": f"{top['selected_estimator']}, k={int(top['selected_k'])}",
                    "study_share": float(top["share"]),
                }
            )
    modal_frame = pd.DataFrame(
        modal, columns=[x, "metric_name", "study_modal", "study_share"]
    )
    rows = rows.merge(modal_frame, on=[x, "metric_name"], how="left")
    rows = rows.rename(columns={x: "setting"})
    return rows[
        [
            "setting",
            "metric_name",
            "recovery",
            "recovery_lo",
            "recovery_hi",
            "ari_mean",
            "agreement",
            "study_modal",
            "study_share",
        ]
    ].sort_values(["setting", "metric_name"], ignore_index=True)
