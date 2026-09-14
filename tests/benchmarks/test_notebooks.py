"""Structural checks on the notebooks.

These do not execute the notebooks; they assert that the code the rebuild
moved into the package is no longer duplicated in notebook cells.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Modules the pre-rebuild notebooks pulled code from, either as
# `import benchmarking_runners` or `from benchmarking_runners import ...`.
# Checking for one fixed spelling of the import missed whichever spelling a
# given notebook actually used, so this checks for the module name itself,
# under either import form.
FORBIDDEN_MODULES = (
    "benchmarking_code",
    "case_study_plotting",
    "benchmarking_plotting",
    "benchmarking_runners",
)
NOTEBOOKS = {
    "benchmarking": REPO_ROOT / "notebooks" / "Benchmarking.ipynb",
    "klein": REPO_ROOT / "notebooks" / "case_studies" / "Klein.ipynb",
    "levine": REPO_ROOT / "notebooks" / "case_studies" / "Levine_32dim.ipynb",
    "motivation": REPO_ROOT / "notebooks" / "case_studies" / "Motivation.ipynb",
    "cusanovich": REPO_ROOT / "notebooks" / "case_studies" / "Cusanovich.ipynb",
    "heca": REPO_ROOT / "notebooks" / "case_studies" / "hECA.ipynb",
    "tutorial": REPO_ROOT / "notebooks" / "Tutorial.ipynb",
    "resolution_tutorial": REPO_ROOT / "notebooks" / "Resolution_Tutorial.ipynb",
}


def _code(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )


def _imports_module(source: str, module: str) -> bool:
    """True if `source` imports `module`, as `import X` or `from X import ...`."""
    pattern = rf"(?m)^\s*(?:import\s+{module}\b|from\s+{module}\s+import\b)"
    return re.search(pattern, source) is not None


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_no_sys_path_manipulation(name):
    source = _code(NOTEBOOKS[name])
    assert "sys.path.insert" not in source
    assert "sys.path.append" not in source


@pytest.mark.parametrize("module", FORBIDDEN_MODULES)
@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_imports_come_from_the_installed_package(name, module):
    source = _code(NOTEBOOKS[name])
    assert not _imports_module(source, module)


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


# Rulings 41 and 53 had to fix these call-site values by hand -- and the
# Levine/Motivation pair drifted from the manuscript again (Fix 3) after
# that -- so they get their own pinned assertions rather than relying on a
# human to notice a re-hardcoded default the next time a case-study
# notebook is edited. Every value here traces to a manuscript line or a
# sibling committed configuration (STUDIES["klein"] in _studies.py):
#   - Klein subsample=0.5: manuscript line 606 (1,358 of 2,717 cells).
#   - Levine/Motivation subsample=5000: manuscript line 627 (a stratified
#     subsample of 5,000 cells).
#   - Klein's prepare_composite selection (generalizability, 1se,
#     not_two=True): the manuscript's headline Ward-agglomerative-at-k=4
#     result (line 624), threaded through CompositeInputs by Fix 4.
def test_klein_loader_uses_the_manuscript_subsample():
    assert "subsample=0.5" in _code(NOTEBOOKS["klein"])


def test_levine_loader_uses_the_manuscript_subsample():
    assert "subsample=5000" in _code(NOTEBOOKS["levine"])


def test_motivation_levine_loader_uses_the_manuscript_subsample():
    assert "subsample=5000" in _code(NOTEBOOKS["motivation"])


def test_klein_prepare_composite_uses_the_generalizability_selection():
    source = _code(NOTEBOOKS["klein"])
    assert 'measure="generalizability"' in source
    assert 'rule="1se"' in source
    assert "not_two=True" in source


def test_cusanovich_notebook_reads_its_config_from_studies():
    source = _code(NOTEBOOKS["cusanovich"])
    # Configuration must be read from STUDIES, not restated. Four manuscript
    # mismatches on this project came from re-derivation at the call site.
    assert 'STUDIES["cusanovich"]' in source
    assert "study_resolution_grids(study)" in source
    assert "n_resamples=study.n_resamples" in source
    assert "preprocessing=study.preprocessing" in source
    assert "resolve_preprocessing(study.preprocessing)" in source
    assert "consensus_anchors=study.consensus_anchors" in source
    assert "randomize_preprocessing=True" in source
    for restated in ("perplexity", "n_neighbors", "n_resamples=150", "np.arange"):
        assert restated not in source


@pytest.mark.parametrize(
    "retired",
    [
        "cvi_sweep",
        "study_model_grids",
        "figure_carve_output_cusanovich",
        "prepare_composite",
    ],
)
def test_cusanovich_notebook_no_longer_runs_the_k_based_comparison(retired):
    # The CVI sweep, the k-based fit and the CompositeInputs figures left the
    # study with the move to randomized preprocessing (case-study spec,
    # section 6).
    assert retired not in _code(NOTEBOOKS["cusanovich"])


def test_cusanovich_notebook_does_not_offer_the_atlas_scale():
    # Randomized preprocessing at every cell means hundreds of t-SNE fits on
    # tens of thousands of cells, so the notebook runs dev and publication
    # only; the atlas scale stays declared in STUDIES for the loader (spec
    # section 6). Cell 4 still names the atlas when it reports the whole
    # release's Unknown count, so the check is on the quoted scale name.
    assert '"atlas"' not in _code(NOTEBOOKS["cusanovich"])


def test_heca_notebook_reads_its_config_from_studies():
    source = _code(NOTEBOOKS["heca"])
    assert 'STUDIES["heca"]' in source
    assert "study_resolution_grids(study)" in source
    assert "consensus_anchors=study.consensus_anchors" in source


# Each ATAC case-study notebook opens with a reference-label scatter of the
# embedding its source provides. The Cusanovich atlas ships its own t-SNE,
# which the loader carries through as meta["source_tsne"] and the figure's
# panel B draws again. hECA ships nothing, so its notebook computes a UMAP the
# way Levine_32dim.ipynb computes its t-SNE and hands that same embedding to
# prepare_composite.
def test_cusanovich_notebook_draws_the_source_tsne_and_writes_its_tables():
    source = _code(NOTEBOOKS["cusanovich"])
    assert 'meta["source_tsne"]' in source
    assert "figure_reference_scatter(" in source
    assert "figure_cusanovich_results(inputs" in source
    assert "save_tables(inputs" in source
    assert "plt.subplots" not in source


def test_heca_notebook_draws_one_umap_throughout():
    source = _code(NOTEBOOKS["heca"])
    assert "UMAP(random_state=RANDOM_SEED)" in source
    assert "figure_reference_scatter(" in source
    assert "embedding=" in source
    assert "plt.subplots" not in source
