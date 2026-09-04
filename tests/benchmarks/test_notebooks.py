"""Structural checks on the notebooks.

These do not execute the notebooks; they assert that the code the rebuild
moved into the package is no longer duplicated in notebook cells.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = {
    "benchmarking": REPO_ROOT / "notebooks" / "Benchmarking.ipynb",
    "klein": REPO_ROOT / "notebooks" / "case_studies" / "Klein.ipynb",
    "levine": REPO_ROOT / "notebooks" / "case_studies" / "Levine_32dim.ipynb",
    "motivation": REPO_ROOT / "notebooks" / "case_studies" / "Motivation.ipynb",
}


def _code(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_no_sys_path_manipulation(name):
    source = _code(NOTEBOOKS[name])
    assert "sys.path.insert" not in source
    assert "sys.path.append" not in source


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_imports_come_from_the_installed_package(name):
    source = _code(NOTEBOOKS[name])
    assert "import benchmarking_code" not in source
    assert "import case_study_plotting" not in source
    assert "import benchmarking_plotting" not in source


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_no_gridspec_layout_in_notebooks(name):
    assert "add_gridspec" not in _code(NOTEBOOKS[name])


def test_anchor_dicts_live_in_the_registry_not_the_notebook():
    assert "anchor_settings_gaussians" not in _code(NOTEBOOKS["benchmarking"])


@pytest.mark.parametrize("name", ["klein", "levine", "motivation"])
def test_loader_code_is_not_duplicated_in_case_study_notebooks(name):
    source = _code(NOTEBOOKS[name])
    assert "BiocManager::install" not in source
    assert "read_block" not in source
    assert "highly_variable_genes" not in source


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_no_get_n_jobs_boilerplate(name):
    assert "def get_n_jobs" not in _code(NOTEBOOKS[name])
