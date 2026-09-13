"""One module per manuscript figure.

Every function returns a Figure and, unless save=False, writes it under its
exact manuscript filename at 300 dpi. Four of these figures previously had no
savefig at all and were right-click-saved out of Jupyter output.

_scenario_overview and _reference_scatter are the two modules here that are
not manuscript figures. The first draws the per-family view the benchmarking
notebook's sections carry, built from the same primitives the manuscript
figures use, and so defaults to save=False. The second is the case-study
notebooks' opening view of their reference labels, and saves alongside the
composites it precedes.
"""

from ._benchmarking_examples import figure_benchmarking_examples
from ._benchmarking_results import figure_benchmarking_results
from ._carve_output import (
    figure_carve_output_cusanovich,
    figure_carve_output_klein,
    figure_carve_output_levine,
)
from ._case_study import CompositeInputs, composite_figure, prepare_composite
from ._cusanovich_results import figure_cusanovich_results
from ._heca_results import figure_heca_results
from ._klein_results import figure_klein_results
from ._levine_results import figure_levine_results
from ._paths import BENCHMARKING_DIR, CASE_STUDY_DIR, VIS_ROOT, figure_path
from ._reference_scatter import figure_reference_scatter
from ._scaling import figure_scaling_ari, figure_scaling_runtime
from ._scenario_overview import figure_scenario_overview
from ._study_scaling import figure_study_scaling

__all__ = [
    "BENCHMARKING_DIR",
    "CASE_STUDY_DIR",
    "CompositeInputs",
    "VIS_ROOT",
    "composite_figure",
    "figure_benchmarking_examples",
    "figure_benchmarking_results",
    "figure_carve_output_cusanovich",
    "figure_carve_output_klein",
    "figure_carve_output_levine",
    "figure_cusanovich_results",
    "figure_heca_results",
    "figure_klein_results",
    "figure_levine_results",
    "figure_path",
    "figure_reference_scatter",
    "figure_scaling_ari",
    "figure_scaling_runtime",
    "figure_scenario_overview",
    "figure_study_scaling",
    "prepare_composite",
]
