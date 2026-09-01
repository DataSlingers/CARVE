"""Packaging invariants for the benchmarks package.

benchmarks must be importable from an editable install (it lives under src/,
which the install's .pth file puts on sys.path) while staying out of the built
wheel (it is listed in the packages.find exclude list).
"""

import tomllib
from pathlib import Path

import benchmarks

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_benchmarks_is_importable_and_versioned():
    assert isinstance(benchmarks.__version__, str)
    assert benchmarks.__version__


def test_benchmarks_lives_under_src():
    assert Path(benchmarks.__file__).resolve().parent.parent.name == "src"


def test_benchmarks_is_excluded_from_the_wheel():
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    exclude = cfg["tool"]["setuptools"]["packages"]["find"]["exclude"]
    assert "benchmarks" in exclude
    assert "benchmarks.*" in exclude


def test_pyarrow_is_declared_in_the_benchmarks_extra():
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extra = cfg["project"]["optional-dependencies"]["benchmarks"]
    assert any(dep.startswith("pyarrow") for dep in extra)


def test_working_run_tree_is_gitignored():
    ignored = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "results/runs/" in [line.strip() for line in ignored]
