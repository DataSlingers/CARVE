"""Tests for figure output path resolution."""

from pathlib import Path

from benchmarks.figures import _paths

# Derived independently of anything under test, from this test file's own
# location (tests/benchmarks/figures/test_paths.py), so these tests do not
# rely on the module attribute they are checking for.
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_repo_root_is_the_directory_containing_pyproject_toml():
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_vis_root_is_the_repo_root_vis_directory():
    assert _paths.VIS_ROOT == REPO_ROOT / "vis"


def test_vis_root_survives_a_working_directory_change(monkeypatch, tmp_path):
    # nbconvert runs a notebook in its own directory, so VIS_ROOT must not
    # depend on the working directory.
    monkeypatch.chdir(tmp_path)
    assert _paths.VIS_ROOT.resolve() == REPO_ROOT / "vis"


def test_plos_figure_files_are_named_by_their_figure_number():
    assert _paths.plos_figure_path(3) == REPO_ROOT / "vis" / "plos" / "Fig3.tif"


def test_plos_figure_path_honors_an_explicit_out_dir(tmp_path):
    assert _paths.plos_figure_path(4, out_dir=tmp_path) == tmp_path / "Fig4.tif"
