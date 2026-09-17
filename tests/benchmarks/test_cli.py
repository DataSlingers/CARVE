"""Tests for the benchmarks command-line entry point."""

import pytest

import benchmarks.run as run_module
from benchmarks._registry import ABLATIONS, PUBLISHED_RANDOM_STATE
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


class TestAblationCli:
    def test_listing_includes_ablations(self, capsys):
        assert main(["--list"]) == 0
        assert "ablation:rho_b" in capsys.readouterr().out

    def test_unknown_ablation_is_rejected(self, capsys):
        assert main(["--ablation", "nope"]) == 2
        assert "nope" in capsys.readouterr().err

    def test_ablation_is_exclusive_with_scenario_and_all(self, capsys):
        assert main(["--ablation", "rho_b", "--scenario", "gaussians"]) == 2
        assert "exclusive" in capsys.readouterr().err
        assert main(["--ablation", "rho_b", "--all"]) == 2
        assert "exclusive" in capsys.readouterr().err

    def test_n_jobs_defaults_differ_by_mode(self):
        args = _parser().parse_args(["--scenario", "gaussians"])
        assert args.n_jobs is None
        assert run_module._resolve_n_jobs(args) == 1
        args = _parser().parse_args(["--ablation", "rho_b"])
        assert run_module._resolve_n_jobs(args) == -1
        args = _parser().parse_args(["--ablation", "rho_b", "--n-jobs", "6"])
        assert run_module._resolve_n_jobs(args) == 6

    def test_scale_defaults_to_the_ablations_default(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_ablation(ablation, **kwargs):
            calls.append(kwargs)
            return tmp_path / "rd"

        monkeypatch.setattr(run_module, "run_ablation", fake_run_ablation)
        code = main(["--ablation", "rho_b", "--root", str(tmp_path)])
        assert code == 0
        kwargs = calls[0]
        assert kwargs["scale"] == ABLATIONS["rho_b"].default_scale

    def test_runs_an_ablation_through_the_runner(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_ablation(ablation, **kwargs):
            calls.append((ablation.name, kwargs))
            return tmp_path / "rd"

        monkeypatch.setattr(run_module, "run_ablation", fake_run_ablation)
        code = main(["--ablation", "rho_b", "--scale", "dev", "--root", str(tmp_path)])
        assert code == 0
        name, kwargs = calls[0]
        assert name == "rho_b"
        assert kwargs["scale"] == "dev"
        assert kwargs["n_jobs"] == -1
        assert kwargs["resume"] is True
        assert kwargs["units"] is None

    def test_timing_batch_passes_the_timing_units_without_resume(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_ablation(ablation, **kwargs):
            calls.append(kwargs)
            return tmp_path / "rd"

        monkeypatch.setattr(run_module, "run_ablation", fake_run_ablation)
        code = main(
            ["--ablation", "rho_b", "--scale", "publication", "--timing-batch",
             "--n-jobs", "5", "--root", str(tmp_path)]
        )
        assert code == 0
        kwargs = calls[0]
        assert kwargs["resume"] is False
        assert kwargs["n_jobs"] == 5
        assert len(kwargs["units"]) == 22
