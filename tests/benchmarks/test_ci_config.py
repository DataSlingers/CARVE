"""CI configuration invariants."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def test_ruff_covers_the_whole_src_tree():
    ci = (WORKFLOWS / "ci.yml").read_text()
    assert "ruff check src/" in ci
    assert "ruff format --check src/" in ci


def test_nightly_runs_a_real_benchmark_not_only_plotting():
    nightly = (WORKFLOWS / "nightly-notebooks.yml").read_text()
    assert "benchmarks.run" in nightly
    assert "--n-seeds" in nightly
    assert "--n-resamples" in nightly
