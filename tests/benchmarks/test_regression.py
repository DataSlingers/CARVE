"""Regression: the rebuilt pipeline must reproduce the committed results.

The comparison is skipped unless CARVE_RUN_REGRESSION=1, because each
scenario is a full 3 x 20 benchmark; the check that the oracle pins the
submitted configuration always runs. Run one comparison with:

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

# The rebuild renames six columns. This mapping exists only to compare against
# the old files and is deleted with them once the port is verified.
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
# The full anchor set has six scenarios, but only three are checked for exact
# reproduction here. circles and moons both ran spectral clustering at
# n=1500, and carve.cluster.SpectralClustering accepted a random_state but
# never threaded it into the sparse ARPACK eigensolver it used at that size --
# ARPACK drew its starting vector from the global numpy RNG instead. That
# describes the which="SM" solver that 09b810c replaced with shift-invert; the
# committed circles/moons numbers came from the replaced solver. Verified
# directly: with identical data and an identical random_state, repeated
# oracle fits on moons/easy gave ARIs of 0.814, 0.814, 1.0, 1.0, while seeding
# np.random before each fit made them identical. The committed circles/moons
# numbers are therefore a single draw from a process that depended on ambient
# global RNG state during sequential notebook execution, which cannot be
# reconstructed -- not a flaky test to work around. This has been reported to
# the author separately.
#
# The remaining three scenarios use k-means and agglomerative clustering,
# which are deterministic here, and they still exercise every shared code
# path: the runner, the artifact layer, the seed derivation, the classical
# indices, and the CARVE metric extraction.
#
# swiss_rolls was in this set until it moved to spectral clustering. Its
# committed CSV was produced by Ward, so it is no longer a comparable
# oracle for this scenario.
UNAFFECTED = {
    "gaussians": "results_gaussian",
    "t_dist": "results_t_dist",
    "t_dist_noise": "results_t_dist_noise",
}

# The committed CSVs above were produced under PUBLISHED_ANCHORS and this
# tree count -- the notebook cells for these three scenarios omitted
# n_trees and got Scenario's old default of 100 (see the _N_TREES comment
# in _registry.py). Pinned here as a named constant, not read from
# SCENARIOS, so this oracle stays reproducible after later branch changes
# to n_trees or ACTIVE_ANCHORS.
_PUBLISHED_N_TREES: int = 100

# The same CSVs were produced at this sample size and with the forest, before
# the registry moved to BENCHMARK_N_TOTAL = 3000 and to shrinkage LDA on the
# Gaussian mixtures (2026-10-16). Pinned for the same reason as the tree
# count.
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
    """Drop the rows the bug fixes deliberately changed, then sort."""
    df = df[df["metric_name"] != "gap"].copy()
    keys = ["axis_label", "seed", "metric_name", "k"]
    return df.sort_values(keys).reset_index(drop=True)


@pytest.fixture(scope="module")
def scenario_run(tmp_path_factory):
    """Return a callable that runs a scenario once and hands back its run dir.

    A dict cache here, keyed by scenario name, holds only the root passed
    to run_scenario -- not a precomputed run directory -- so every call
    still goes through run_scenario itself and returns exactly what
    run_scenario returns, never a hand-built guess at its content-addressed
    path.

    random_state is pinned to PUBLISHED_RANDOM_STATE (42), not a literal.
    The published benchmarks ran at 42 (notebook cell 3, RANDOM_SEED = 42,
    passed to every scenario call). Seeds derive as
    benchmark_seed = seed + axis_idx * 10000 + random_state, so any other
    base seed simulates entirely different data and cannot match the
    committed CSVs.

    Each run goes through _published_scenario, which pins anchors, n_trees,
    n_total and the classifier to the values that produced the committed
    CSVs rather than reading them off SCENARIOS[scenario_name], so this
    oracle keeps comparing against that configuration as the live registry
    moves.
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

    # is_selected records which k CARVE actually picked -- the benchmark's
    # headline output, and the one the manuscript's k_recovery tables are
    # computed from. The 1se and quantile rules select on sweep_rank, not on
    # the raw values just compared above, so a selection-rule regression
    # that left every score untouched could still flip which k is flagged
    # and pass unnoticed if this were not checked separately. Both sides are
    # boolean, so compare directly rather than with a float tolerance.
    #
    # selects_true_k is not compared: it is k == k_star, a deterministic
    # function of the merge key (k_star is a constant 5 across every
    # scenario here), so it carries no information beyond what the merge
    # already pins and checking it would only re-assert the merge succeeded.
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
    """The committed CSVs came from n=1500 and the forest. The live registry
    has since moved to BENCHMARK_N_TOTAL=3000 and to LDA on the Gaussian
    mixtures, so the oracle has to pin both, or it compares a different
    experiment against them and fails only when someone sets
    CARVE_RUN_REGRESSION=1."""
    scenario = _published_scenario(scenario_name)
    assert scenario.shared["n_total"] == 1500
    assert scenario.classifier == "random_forest"
    assert scenario.n_trees == 100
    assert scenario.anchors == PUBLISHED_ANCHORS[scenario_name]
    live = SCENARIOS[scenario_name]
    assert {k: v for k, v in scenario.shared.items() if k != "n_total"} == {
        k: v for k, v in live.shared.items() if k != "n_total"
    }
