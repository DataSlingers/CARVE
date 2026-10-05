"""Tests for the hECA command-line entry point."""

import json

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


def test_a_stage_that_would_overwrite_exits_two(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(STUDIES, "heca", small_heca_study(tmp_path / "data"))
    run = str(tmp_path / "run")
    assert main(["embed", "--run-dir", run]) == 0
    assert main(["embed", "--run-dir", run]) == 2
    assert "--force" in capsys.readouterr().err
