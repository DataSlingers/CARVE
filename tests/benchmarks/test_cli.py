"""Tests for the benchmarks command-line entry point."""

import pytest

from benchmarks._registry import PUBLISHED_RANDOM_STATE
from benchmarks.run import _parser, main


@pytest.fixture(scope="module")
def gaussians_run(tmp_path_factory):
    """One reduced gaussians run through the CLI: (exit code, root)."""
    root = tmp_path_factory.mktemp("cli")
    code = main(
        [
            "--scenario",
            "gaussians",
            "--root",
            str(root),
            "--n-seeds",
            "1",
            "--n-resamples",
            "20",
        ]
    )
    return code, root


class TestCli:
    def test_listing_scenarios_succeeds(self, capsys):
        assert main(["--list"]) == 0
        assert "gaussians" in capsys.readouterr().out

    def test_unknown_scenario_is_rejected(self, capsys):
        assert main(["--scenario", "nope"]) == 2
        assert "nope" in capsys.readouterr().err

    def test_requires_one_of_scenario_all_list_or_promote(self, capsys):
        assert main([]) == 2

    def test_random_state_default_is_published_random_state(self):
        # The published benchmarks ran with 42, not 0. A default of 0
        # silently simulates different data and cannot reproduce committed
        # results, so this pins the default against a silent revert.
        args = _parser().parse_args(["--scenario", "gaussians"])
        assert args.random_state == PUBLISHED_RANDOM_STATE == 42

    def test_runs_a_scenario_end_to_end(self, gaussians_run):
        code, root = gaussians_run
        assert code == 0
        assert list(root.glob("gaussians/*/manifest.json"))

    def test_promote_publishes_a_finished_run(self, gaussians_run, tmp_path):
        _, root = gaussians_run
        rd = next((root / "gaussians").iterdir())
        code = main(["--promote", str(rd), "--published-root", str(tmp_path / "pub")])
        assert code == 0
        assert (tmp_path / "pub" / "gaussians" / "results.csv").exists()

    def test_promote_rejects_a_directory_with_no_manifest(self, tmp_path, capsys):
        (tmp_path / "empty").mkdir()
        assert main(["--promote", str(tmp_path / "empty")]) == 1
        assert "manifest" in capsys.readouterr().err
