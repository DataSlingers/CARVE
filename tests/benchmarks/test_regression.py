"""Regression: the rebuilt pipeline must reproduce the committed results.

Marked slow and skipped unless CARVE_RUN_REGRESSION=1, because each scenario
is a full 3 x 20 benchmark. Run one with:

    CARVE_RUN_REGRESSION=1 .venv/bin/pytest \
        tests/benchmarks/test_regression.py -k gaussians -v
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from benchmarks._artifacts import read_run
from benchmarks._registry import (
    GENERALIZABILITY_METRICS,
    PUBLISHED_RANDOM_STATE,
    SCENARIOS,
)
from benchmarks._run import run_scenario

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
# The full anchor set has six scenarios, but only four are checked for exact
# reproduction here. circles and moons both run spectral clustering at
# n=1500, and carve.cluster.SpectralClustering accepts a random_state but
# never threads it into the sparse ARPACK eigensolver it uses at that size --
# ARPACK draws its starting vector from the global numpy RNG instead. Verified
# directly: with identical data and an identical random_state, repeated
# oracle fits on moons/easy gave ARIs of 0.814, 0.814, 1.0, 1.0, while seeding
# np.random before each fit made them identical. The committed circles/moons
# numbers are therefore a single draw from a process that depended on ambient
# global RNG state during sequential notebook execution, which cannot be
# reconstructed -- not a flaky test to work around. This has been reported to
# the author separately.
#
# The remaining four scenarios use k-means and agglomerative clustering,
# which are deterministic here, and they still exercise every shared code
# path: the runner, the artifact layer, the seed derivation, the classical
# indices, and the CARVE metric extraction.
UNAFFECTED = {
    "gaussians": "results_gaussian",
    "t_dist": "results_t_dist",
    "t_dist_noise": "results_t_dist_noise",
    "swiss_rolls": "results_swiss_rolls",
}

pytestmark = pytest.mark.skipif(
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


@pytest.mark.slow
@pytest.mark.parametrize(("scenario_name", "stem"), sorted(UNAFFECTED.items()))
def test_rebuilt_pipeline_reproduces_committed_results(scenario_name, stem, tmp_path):
    scenario = SCENARIOS[scenario_name]
    # The published benchmarks ran at random_state=42 (notebook cell 3,
    # RANDOM_SEED = 42, passed to every scenario call). Seeds derive as
    # benchmark_seed = seed + axis_idx * 10000 + random_state, so any other
    # base seed simulates entirely different data and cannot match the
    # committed CSVs. run_scenario defaults to PUBLISHED_RANDOM_STATE; it is
    # passed explicitly here so the dependency is visible in the test.
    rd = run_scenario(
        scenario, root=tmp_path, n_jobs=-1, random_state=PUBLISHED_RANDOM_STATE
    )

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

    # ari_at_k is expected to move only for the generalizability metrics.
    stability_only = merged[~merged["metric_name"].isin(GENERALIZABILITY_METRICS)]
    pd.testing.assert_series_equal(
        stability_only["ari_at_k_old"],
        stability_only["ari_at_k_new"],
        check_names=False,
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.slow
@pytest.mark.parametrize(("scenario_name", "stem"), sorted(UNAFFECTED.items()))
def test_generalizability_ari_changed_as_the_fix_intended(scenario_name, stem, tmp_path):
    """The ari_at_k fix must actually change something, or it did nothing."""
    scenario = SCENARIOS[scenario_name]
    rd = run_scenario(
        scenario, root=tmp_path, n_jobs=-1, random_state=PUBLISHED_RANDOM_STATE
    )

    new = _comparable(read_run(rd))
    old = _comparable(_load_old(stem))
    merged = old.merge(
        new,
        on=["axis_label", "seed", "metric_name", "k"],
        suffixes=("_old", "_new"),
    )
    gen = merged[merged["metric_name"].isin(GENERALIZABILITY_METRICS)]
    assert not gen["ari_at_k_old"].equals(gen["ari_at_k_new"])
