"""One module per manuscript figure.

Every function returns a Figure and, unless save=False, writes it under its
exact manuscript filename at 300 dpi. Four of these figures previously had no
savefig at all and were right-click-saved out of Jupyter output.

_scenario_overview is the one module here that is not a manuscript figure. It
draws the per-family view the notebook's sections carry, built from the same
primitives the manuscript figures use, and so defaults to save=False.
"""

from ._benchmarking_examples import figure_benchmarking_examples
from ._benchmarking_results import figure_benchmarking_results
from ._carve_output import figure_carve_output_klein, figure_carve_output_levine
from ._case_study import CompositeInputs, composite_figure, prepare_composite
from ._klein_results import figure_klein_results
from ._levine_results import figure_levine_results
from ._paths import BENCHMARKING_DIR, CASE_STUDY_DIR, VIS_ROOT, figure_path
from ._scaling import figure_scaling_ari, figure_scaling_runtime
from ._scenario_overview import figure_scenario_overview

__all__ = [
    "BENCHMARKING_DIR",
    "CASE_STUDY_DIR",
    "CompositeInputs",
    "VIS_ROOT",
    "composite_figure",
    "figure_benchmarking_examples",
    "figure_benchmarking_results",
    "figure_carve_output_klein",
    "figure_carve_output_levine",
    "figure_klein_results",
    "figure_levine_results",
    "figure_path",
    "figure_scaling_ari",
    "figure_scaling_runtime",
    "figure_scenario_overview",
    "prepare_composite",
]
