"""Tests for the hECA command-line entry point."""

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks._studies import STUDIES
from benchmarks.heca import _parser, main
from tests.benchmarks._helpers import small_heca_study


def test_an_unknown_stage_is_rejected():
    with pytest.raises(SystemExit):
        _parser().parse_args(["ladder", "--run-dir", "x"])


# The default scan runs resolution up to 10, which on this toy fixture's
# ~69-cell training split fragments every cell into its own cluster. sklearn
# then warns that the cluster labels look more like a regression target than
# a classification one when scoring ARI. Real hECA scale never approaches
# one cluster per cell, so this is a toy-scale artifact, not a defect.
@pytest.mark.filterwarnings(
    "ignore:The number of unique classes is greater than 50% "
    "of the number of samples:UserWarning"
)
def test_the_stages_run_in_order_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(STUDIES, "heca", small_heca_study(tmp_path / "data"))
    run = str(tmp_path / "run")

    assert main(["embed", "--run-dir", run]) == 0
    # The default 40-value scan on a toy embedding may or may not satisfy the
    # rule; both outcomes write their outputs.
    assert main(["calibrate", "--run-dir", run]) in (0, 1)
    calibration = json.loads((tmp_path / "run" / "calibration.json").read_text())
    assert len(calibration["scan"]) == 40
    assert main(["fit", "--run-dir", run, "--n-jobs", "1"]) == 0
    capsys.readouterr()
    assert main(["status", "--run-dir", run]) == 0
    assert "Configurations complete: 4 of 4" in capsys.readouterr().out


def test_status_before_the_fit_starts_says_so(tmp_path, capsys):
    run = tmp_path / "run"
    assert main(["status", "--run-dir", str(run)]) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == f"No fit has started in {run}."
    assert captured.out == ""


def test_a_stage_that_would_overwrite_exits_two(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(STUDIES, "heca", small_heca_study(tmp_path / "data"))
    run = str(tmp_path / "run")
    assert main(["embed", "--run-dir", run]) == 0
    assert main(["embed", "--run-dir", run]) == 2
    assert "--force" in capsys.readouterr().err


SLURM_DIR = Path(__file__).resolve().parents[2] / "slurm" / "heca"


@pytest.mark.parametrize("stage", ["embed", "calibrate", "fit"])
def test_each_sbatch_script_runs_its_stage_and_parses(stage):
    script = SLURM_DIR / f"{stage}.sbatch"
    assert f"python -m benchmarks.heca {stage} --run-dir" in script.read_text()
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


@pytest.mark.parametrize("stage", ["embed", "calibrate", "fit"])
def test_each_sbatch_script_passes_its_arguments_to_the_stage(stage):
    # sbatch <script> --force reaches the stage as "$@", so a rerun the
    # runbook asks for does not exit 2 on the overwrite guard.
    (line,) = [
        line
        for line in (SLURM_DIR / f"{stage}.sbatch").read_text().splitlines()
        if f"python -m benchmarks.heca {stage}" in line
    ]
    command = line.split("||")[0].strip()
    assert command.endswith('"$@"')


GENERAL_MAX_MEM_PER_NODE_MB = 232448


def test_the_fit_script_takes_a_whole_node_and_saves_its_accounting():
    text = (SLURM_DIR / "fit.sbatch").read_text()
    directives = dict(
        line.removeprefix("#SBATCH --").partition("=")[::2]
        for line in text.splitlines()
        if line.startswith("#SBATCH --")
    )
    # Longleaf's submit filter rejects --exclusive and a named partition.
    # The job asks for every CPU of a general_big AMD node instead, and for
    # more memory than the general partition allows, so Slurm routes it to
    # general_big.
    assert "exclusive" not in directives
    assert "partition" not in directives
    assert directives["constraint"] == "amd"
    assert int(directives["cpus-per-task"]) == 2 * 96 * 2
    assert directives["mem"].endswith("G")
    assert int(directives["mem"][:-1]) * 1024 > GENERAL_MAX_MEM_PER_NODE_MB
    assert '--cpus-per-task="${SLURM_CPUS_ON_NODE}"' in text
    assert "sacct -j" in text
    assert "fit/sacct.txt" in text
    # --export=ALL carries the submitting shell's thread counts into the
    # job, and joblib passes a parent's value on to every worker.
    lines = text.splitlines()
    unset = lines.index("unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS")
    (srun,) = [i for i, line in enumerate(lines) if line.startswith("srun ")]
    assert unset < srun
    # CARVE's per-configuration lines reach the log as they are printed.
    assert lines.index("export PYTHONUNBUFFERED=1") < srun


def test_calibration_runs_single_threaded():
    text = (SLURM_DIR / "calibrate.sbatch").read_text()
    assert "OMP_NUM_THREADS=1" in text


def test_the_runbook_names_every_script():
    text = (SLURM_DIR / "README.md").read_text()
    for stage in ("embed", "calibrate", "fit"):
        assert f"slurm/heca/{stage}.sbatch" in text
