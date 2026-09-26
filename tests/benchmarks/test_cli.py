"""Tests for the benchmarks command-line entry point."""

import json
import shutil
from pathlib import Path

import pytest

import benchmarks.run as run_module
from benchmarks._registry import ABLATIONS, PUBLISHED_RANDOM_STATE
from benchmarks.run import _parser, main


@pytest.fixture(scope="module")
def gaussians_run(tmp_path_factory):
    """One reduced gaussians run through the CLI: (exit code, root).

    The CLI runs full scale only, so the reduction is injected into the
    runner it calls: one dataset per axis value at B=20.

    --n-jobs 1 pinned explicitly: "gaussians" is a difficulty scenario, so it
    otherwise resolves to -1, and this three-cell run would pay a real loky
    pool's startup cost and run its cells where a warning cannot reach
    filterwarnings = error. This fixture only needs a completed run, not the
    n_jobs resolution, which test_scenarios_default_to_per_scenario_resolution
    covers on a mock.
    """
    root = tmp_path_factory.mktemp("cli")
    real_run_scenario = run_module.run_scenario

    def reduced(scenario, **kwargs):
        return real_run_scenario(scenario, **kwargs, n_seeds=1, n_resamples=20)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_module, "run_scenario", reduced)
        code = main(
            ["--scenario", "gaussians", "--root", str(root), "--n-jobs", "1"]
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

    def test_promote_rejects_an_interrupted_run(self, gaussians_run, tmp_path, capsys):
        """A copy of the finished run with its manifest set back to
        "running", which is what an interrupted run leaves on disk."""
        _, root = gaussians_run
        rd = tmp_path / "interrupted"
        shutil.copytree(next((root / "gaussians").iterdir()), rd)
        manifest = json.loads((rd / "manifest.json").read_text())
        manifest["status"] = "running"
        (rd / "manifest.json").write_text(json.dumps(manifest))

        code = main(["--promote", str(rd), "--published-root", str(tmp_path / "pub")])
        assert code == 1
        assert "status 'running'" in capsys.readouterr().err
        assert not (tmp_path / "pub").exists()


def test_allow_code_change_reaches_run_scenario(monkeypatch):
    captured = {}

    def _fake(scenario, **kwargs):
        captured.update(kwargs)
        return Path("results/runs/gaussians/abc")

    monkeypatch.setattr("benchmarks.run.run_scenario", _fake)
    main(["--scenario", "gaussians", "--allow-code-change"])
    assert captured["allow_code_change"] is True


def test_the_code_version_is_checked_by_default(monkeypatch):
    captured = {}

    def _fake(scenario, **kwargs):
        captured.update(kwargs)
        return Path("results/runs/gaussians/abc")

    monkeypatch.setattr("benchmarks.run.run_scenario", _fake)
    main(["--scenario", "gaussians"])
    assert captured["allow_code_change"] is False


def test_scenarios_default_to_per_scenario_resolution(monkeypatch):
    """The CLI hands None down and the runner decides, because one flag
    cannot be right for both the difficulty scenarios and the timed pair."""
    captured = {}
    monkeypatch.setattr(
        "benchmarks.run.run_scenario",
        lambda scenario, **kwargs: captured.update(kwargs)
        or Path("results/runs/x/y"),
    )
    main(["--scenario", "gaussians"])
    assert captured["n_jobs"] is None


def test_scenarios_run_at_full_scale(monkeypatch):
    """The CLI passes no dataset or resample count, so every scenario runs
    its registered n_seeds at the runner's B=100."""
    captured = {}
    monkeypatch.setattr(
        "benchmarks.run.run_scenario",
        lambda scenario, **kwargs: captured.update(kwargs)
        or Path("results/runs/x/y"),
    )
    main(["--scenario", "gaussians"])
    assert "n_seeds" not in captured
    assert "n_resamples" not in captured


@pytest.mark.parametrize(
    "flag, value",
    [("--n-seeds", "3"), ("--n-resamples", "10"), ("--scale", "dev")],
)
def test_reduced_scale_flags_are_rejected(flag, value, capsys):
    with pytest.raises(SystemExit) as excinfo:
        _parser().parse_args(["--scenario", "gaussians", flag, value])
    assert excinfo.value.code == 2
    assert flag in capsys.readouterr().err


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
        # A scenario's default is None, not a number: the runner decides
        # per scenario class (see test_scenarios_default_to_per_scenario_
        # resolution below), because one flag cannot be right for both the
        # difficulty scenarios and the timed pair.
        args = _parser().parse_args(["--scenario", "gaussians"])
        assert args.n_jobs is None
        assert run_module._resolve_n_jobs(args) is None
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
        code = main(["--ablation", "rho_b", "--root", str(tmp_path)])
        assert code == 0
        name, kwargs = calls[0]
        assert name == "rho_b"
        assert kwargs["scale"] == "publication"
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
            ["--ablation", "rho_b", "--timing-batch",
             "--n-jobs", "5", "--root", str(tmp_path)]
        )
        assert code == 0
        kwargs = calls[0]
        assert kwargs["resume"] is False
        assert kwargs["n_jobs"] == 5
        assert len(kwargs["units"]) == 3
