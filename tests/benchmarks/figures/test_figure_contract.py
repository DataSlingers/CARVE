"""Contract every manuscript figure function must satisfy."""

import inspect

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.figure import Figure

from benchmarks import figures

EXPECTED = (
    "figure_benchmarking_examples",
    "figure_benchmarking_results",
    "figure_scaling_ari",
    "figure_scaling_runtime",
    "figure_klein_results",
    "figure_levine_results",
    "figure_carve_output_klein",
    "figure_carve_output_levine",
    "figure_cusanovich_results",
    "figure_heca_results",
    "figure_study_scaling",
    "figure_reference_scatter",
)

# Figures not yet implemented. Tasks 9 through 12 each removed their own
# figures from this set as they landed; it is now empty, since all eleven
# manuscript figures exist, as does the case-study reference-label overview
# (figure_reference_scatter), a notebook figure held to the same contract.
# It is a one-way ratchet from here: re-adding a name would silence that
# figure's contract checks below, so doing so must be a deliberate, visible
# act, never an incidental append to a list.
NOT_YET_IMPLEMENTED = frozenset()


def test_not_yet_implemented_is_empty():
    assert NOT_YET_IMPLEMENTED == frozenset()


def test_all_twelve_figures_are_exported():
    assert set(figures.__all__) >= set(EXPECTED) - NOT_YET_IMPLEMENTED


@pytest.mark.parametrize("name", EXPECTED)
def test_every_figure_function_exists(name):
    if name in NOT_YET_IMPLEMENTED:
        pytest.skip(f"{name} is not implemented yet")
    assert callable(getattr(figures, name))


@pytest.mark.parametrize("name", EXPECTED)
def test_every_figure_function_returns_a_figure(name):
    if name in NOT_YET_IMPLEMENTED:
        pytest.skip(f"{name} is not implemented yet")
    signature = inspect.signature(getattr(figures, name))
    assert signature.return_annotation in (Figure, "Figure")


@pytest.mark.parametrize("name", EXPECTED)
def test_every_figure_function_accepts_save_and_out_dir(name):
    if name in NOT_YET_IMPLEMENTED:
        pytest.skip(f"{name} is not implemented yet")
    parameters = inspect.signature(getattr(figures, name)).parameters
    assert "save" in parameters
    assert "out_dir" in parameters


@pytest.mark.parametrize("name", EXPECTED)
def test_no_figure_module_calls_plt_show(name):
    if name in NOT_YET_IMPLEMENTED:
        pytest.skip(f"{name} is not implemented yet")
    source = inspect.getsource(inspect.getmodule(getattr(figures, name)))
    assert "plt.show(" not in source
