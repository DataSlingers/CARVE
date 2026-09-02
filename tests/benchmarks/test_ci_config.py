"""CI configuration invariants."""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Below roughly 7 resamples, consensus_gini_stability and
# consensus_ce_stability come back all-NaN, and CARVE.get_k then raises
# ValueError: Encountered all NA values from idxmax. 7 is the measured
# boundary on a small fixture; the workflow is asserted to stay above it
# with margin rather than pinned to a specific literal, so the value can be
# tuned upward later without a spurious test failure.
MIN_SAFE_N_RESAMPLES = 7


def test_ruff_covers_the_whole_src_tree():
    ci = (WORKFLOWS / "ci.yml").read_text()
    assert "ruff check src/" in ci
    assert "ruff format --check src/" in ci


def test_nightly_runs_a_real_benchmark_not_only_plotting():
    nightly = yaml.safe_load((WORKFLOWS / "nightly-notebooks.yml").read_text())
    steps = nightly["jobs"]["run-notebooks"]["steps"]

    matches = [s for s in steps if s.get("name") == "Benchmark smoke run"]
    assert matches, "no 'Benchmark smoke run' step found in nightly-notebooks.yml"
    step = matches[0]

    # A step gated behind `if: false` (or any other disabling condition)
    # would satisfy every check below while running nothing -- the same
    # silently-useless nightly job this task exists to remove.
    assert "if" not in step, (
        "the smoke step carries an 'if:' key and can be disabled without "
        "failing any other assertion here"
    )

    run = step["run"]
    assert "benchmarks.run" in run
    assert "--n-seeds" in run

    match = re.search(r"--n-resamples\s+(\d+)", run)
    assert match, "no --n-resamples value found in the smoke step's run block"
    n_resamples = int(match.group(1))
    assert n_resamples >= MIN_SAFE_N_RESAMPLES, (
        f"--n-resamples is {n_resamples}, below the measured all-NaN boundary "
        f"of {MIN_SAFE_N_RESAMPLES}: consensus_gini_stability and "
        "consensus_ce_stability come back all-NaN below that point, and "
        "CARVE.get_k raises ValueError: Encountered all NA values"
    )
