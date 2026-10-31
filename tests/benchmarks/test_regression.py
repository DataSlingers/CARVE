"""Regression: the pipeline must reproduce the committed results.

The comparison is skipped unless CARVE_RUN_REGRESSION=1, because each
scenario is a full 3 x 20 benchmark; the check that the oracle pins the
published configuration always runs. Run one comparison with:

    CARVE_RUN_REGRESSION=1 .venv/bin/pytest \
        tests/benchmarks/test_regression.py -k gaussians -v
"""

import dataclasses
import os
from pathlib import Path

import pandas as pd
import pytest

from benchmarks._artifacts import read_run
from benchmarks._registry import (
    CVI_METRICS,
    PUBLISHED_ANCHORS,
    PUBLISHED_RANDOM_STATE,
    SCENARIOS,
)
from benchmarks._run import run_scenario
from benchmarks._types import Scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

# Column names in the committed CSVs and their current names.
OLD_TO_NEW = {
    "difficulty": "axis_label",
    "stage": "axis_label",
    "is_optimal": "is_selected",
    "is_correct": "selects_true_k",
    "metric_ari": "ari_at_k",
    "baseline_ari": "oracle_ari",
    "dataset_iteration": "seed",
    "true_k": "k_star",
}

# Scenario name in the registry -> committed CSV stem.
#
# Only the k-means and agglomerative scenarios are compared. The committed
# circles and moons numbers came from a spectral eigensolver that drew its
# starting vector from numpy's global RNG, so they cannot be reproduced, and
# swiss_rolls' committed CSV came from Ward. The three scenarios below still
# cover the runner, the artifact layer, the seed derivation, the classical
# indices and the CARVE metric extraction.
UNAFFECTED = {
    "gaussians": "results_gaussian",
    "t_dist": "results_t_dist",
    "t_dist_noise": "results_t_dist_noise",
}

# The committed CSVs above were produced under PUBLISHED_ANCHORS with this
# tree count, pinned here rather than read from SCENARIOS.
_PUBLISHED_N_TREES: int = 100

# The same CSVs were produced at this sample size and with the forest; the
# registry uses BENCHMARK_N_TOTAL and shrinkage LDA on the Gaussian mixtures.
_PUBLISHED_N_TOTAL: int = 1500
_PUBLISHED_CLASSIFIER: str = "random_forest"


def _published_scenario(scenario_name: str) -> Scenario:
    """The registered scenario with every setting the committed CSVs were
    produced under put back: anchors, tree count, sample size, classifier."""
    live = SCENARIOS[scenario_name]
    return dataclasses.replace(
        live,
        anchors=PUBLISHED_ANCHORS[scenario_name],
        n_trees=_PUBLISHED_N_TREES,
        shared={**live.shared, "n_total": _PUBLISHED_N_TOTAL},
        classifier=_PUBLISHED_CLASSIFIER,
    )


requires_regression = pytest.mark.skipif(
    os.environ.get("CARVE_RUN_REGRESSION") != "1",
    reason="set CARVE_RUN_REGRESSION=1 to run the full-scenario regression",
)


def _load_old(stem: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{stem}.csv")
    return df.rename(columns={k: v for k, v in OLD_TO_NEW.items() if k in df.columns})


def _comparable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows the bug fixes changed, then sort."""
    df = df[df["metric_name"] != "gap"].copy()
    keys = ["axis_label", "seed", "metric_name", "k"]
    return df.sort_values(keys).reset_index(drop=True)


@pytest.fixture(scope="module")
def scenario_run(tmp_path_factory):
    """Return a callable that runs a scenario once and hands back its run dir.

    The cache holds only the root passed to run_scenario, so every call
    goes through run_scenario and returns its run directory.

    random_state is PUBLISHED_RANDOM_STATE: seeds derive as
    seed + axis_idx * 10000 + random_state, so another base seed simulates
    different data. Each run goes through _published_scenario, which pins
    anchors, n_trees, n_total and the classifier to the values that
    produced the committed CSVs.
    """
    roots: dict[str, Path] = {}

    def _run(scenario_name: str) -> Path:
        if scenario_name not in roots:
            roots[scenario_name] = tmp_path_factory.mktemp(scenario_name)
        scenario = _published_scenario(scenario_name)
        return run_scenario(
            scenario,
            root=roots[scenario_name],
            n_jobs=-1,
            random_state=PUBLISHED_RANDOM_STATE,
        )

    return _run


@requires_regression
@pytest.mark.parametrize(("scenario_name", "stem"), sorted(UNAFFECTED.items()))
def test_rebuilt_pipeline_reproduces_committed_results(scenario_name, stem, scenario_run):
    rd = scenario_run(scenario_name)

    new = _comparable(read_run(rd))
    old = _comparable(_load_old(stem))

    assert len(new) == len(old), (
        f"{scenario_name}: row count changed, {len(old)} -> {len(new)}"
    )

    merged = old.merge(
        new,
        on=["axis_label", "seed", "metric_name", "k"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    assert len(merged) == len(old)

    pd.testing.assert_series_equal(
        merged["metric_value_old"],
        merged["metric_value_new"],
        check_names=False,
        rtol=1e-6,
        atol=1e-8,
    )
    pd.testing.assert_series_equal(
        merged["oracle_ari_old"],
        merged["oracle_ari_new"],
        check_names=False,
        rtol=1e-6,
        atol=1e-8,
    )

    # is_selected records which k CARVE picked, which the k_recovery tables
    # are computed from. The 1se and quantile rules select on sweep_rank, so
    # it is compared on its own. Both sides are boolean.
    #
    # selects_true_k is not compared: it is k == k_star, a function of the
    # merge key.
    mismatched = merged[merged["is_selected_old"] != merged["is_selected_new"]]
    sample_columns = [
        "axis_label",
        "seed",
        "metric_name",
        "k",
        "is_selected_old",
        "is_selected_new",
    ]
    assert mismatched.empty, (
        f"{scenario_name}: is_selected disagrees on {len(mismatched)} of "
        f"{len(merged)} rows. Sample:\n{mismatched[sample_columns].head(10)}"
    )

    # The committed files scored every CARVE metric on its stability
    # consensus and every classical index on the full-data fit. The rebuild
    # keeps the first in consensus_ari_at_k and puts the full-data fit in
    # ari_at_k for every row.
    classical = merged["metric_name"].isin(CVI_METRICS)
    pd.testing.assert_series_equal(
        merged.loc[~classical, "ari_at_k_old"],
        merged.loc[~classical, "consensus_ari_at_k"],
        check_names=False,
        rtol=1e-6,
        atol=1e-8,
    )
    pd.testing.assert_series_equal(
        merged.loc[classical, "ari_at_k_old"],
        merged.loc[classical, "ari_at_k_new"],
        check_names=False,
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize("scenario_name", sorted(UNAFFECTED))
def test_the_oracle_simulates_the_submitted_configuration(scenario_name):
    """The committed CSVs came from n=1500 and the forest, so the oracle
    pins both."""
    scenario = _published_scenario(scenario_name)
    assert scenario.shared["n_total"] == 1500
    assert scenario.classifier == "random_forest"
    assert scenario.n_trees == 100
    assert scenario.anchors == PUBLISHED_ANCHORS[scenario_name]
    live = SCENARIOS[scenario_name]
    assert {k: v for k, v in scenario.shared.items() if k != "n_total"} == {
        k: v for k, v in live.shared.items() if k != "n_total"
    }
