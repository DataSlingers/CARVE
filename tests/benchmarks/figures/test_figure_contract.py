"""Contract every manuscript figure function must satisfy."""

import inspect

import pytest
from matplotlib.figure import Figure

from benchmarks import figures

EXPECTED = (
    "figure_clustering_problem",
    "figure_benchmarking_examples",
    "figure_benchmarking_results",
    "figure_scaling_ari",
    "figure_scaling_runtime",
    "figure_klein_results",
    "figure_levine_results",
    "figure_levine_composition",
    "figure_carve_output_klein",
    "figure_carve_output_levine",
    "figure_carve_output_cusanovich",
    "figure_carve_output_heca",
    "figure_comparators",
    "figure_cusanovich_results",
    "figure_heca_calibration",
    "figure_heca_carve",
    "figure_heca_runtime",
    "figure_reference_scatter",
    "figure_ablation",
    "figure_ablation_resampling",
    "figure_guidance_global",
    "figure_guidance_clusters",
)

# Figure functions exempt from the contract checks below. Empty: every
# manuscript figure, and figure_reference_scatter, is implemented.
NOT_YET_IMPLEMENTED = frozenset()


def test_not_yet_implemented_is_empty():
    assert NOT_YET_IMPLEMENTED == frozenset()


def test_all_expected_figures_are_exported():
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
