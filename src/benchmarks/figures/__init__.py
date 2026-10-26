"""One module per manuscript figure.

Every function returns a Figure and, unless save=False, writes it under its
exact manuscript filename at 300 dpi. Four of these figures previously had no
savefig at all and were right-click-saved out of Jupyter output.

_scenario_overview, _scenario_dashboard and _reference_scatter are the three
modules here that are not manuscript figures. The first two draw the
per-family views the benchmarking notebook's sections carry, built from the
same primitives the manuscript figures use, and so default to save=False.
The third is the case-study notebooks' opening view of their reference
labels, and saves alongside the composites it precedes.

_heca_calibration, _heca_carve and _heca_runtime draw the hECA notebook's
views of a run on Longleaf; _carve_output also draws its CARVE overview.

_levine_composition draws what the Levine clusters are made of: composition by
gated population and median marker expression. composition_tables and
marker_table return the numbers it draws.

_guidance draws the two schematic figures of the practical-guidance S Text,
from hand-specified shapes rather than a fit.

The six main figures are laid out at the PLOS page width and also saved as
Fig1.tif to Fig6.tif under vis/plos/ (PLOS_DIR) for the submission; Fig 2,
the CARVE overview, is drawn outside this package.
"""

from ._ablation import figure_ablation, figure_ablation_resampling
from ._benchmarking_examples import figure_benchmarking_examples
from ._benchmarking_results import figure_benchmarking_results
from ._carve_output import (
    figure_carve_output_heca,
    figure_carve_output_klein,
    figure_carve_output_levine,
)
from ._case_study import CompositeInputs, composite_figure, prepare_composite
from ._clustering_problem import DatasetRow, figure_clustering_problem
from ._cusanovich_results import figure_cusanovich_results
from ._guidance import figure_guidance_clusters, figure_guidance_global
from ._heca_calibration import figure_heca_calibration
from ._heca_carve import figure_heca_carve
from ._heca_runtime import figure_heca_runtime
from ._klein_m3c import figure_klein_m3c
from ._klein_results import figure_klein_results
from ._levine_composition import (
    composition_tables,
    figure_levine_composition,
    marker_table,
)
from ._levine_results import figure_levine_results
from ._paths import (
    BENCHMARKING_DIR,
    CASE_STUDY_DIR,
    GUIDANCE_DIR,
    PLOS_DIR,
    VIS_ROOT,
    figure_path,
    plos_figure_path,
)
from ._reference_scatter import figure_reference_scatter
from ._scaling import figure_scaling_ari, figure_scaling_runtime
from ._scenario_dashboard import figure_scenario_dashboard
from ._scenario_overview import figure_scenario_overview

__all__ = [
    "BENCHMARKING_DIR",
    "CASE_STUDY_DIR",
    "CompositeInputs",
    "DatasetRow",
    "GUIDANCE_DIR",
    "PLOS_DIR",
    "VIS_ROOT",
    "composite_figure",
    "composition_tables",
    "figure_ablation",
    "figure_ablation_resampling",
    "figure_benchmarking_examples",
    "figure_benchmarking_results",
    "figure_carve_output_heca",
    "figure_carve_output_klein",
    "figure_carve_output_levine",
    "figure_clustering_problem",
    "figure_cusanovich_results",
    "figure_guidance_clusters",
    "figure_guidance_global",
    "figure_heca_calibration",
    "figure_heca_carve",
    "figure_heca_runtime",
    "figure_klein_m3c",
    "figure_klein_results",
    "figure_levine_composition",
    "figure_levine_results",
    "figure_path",
    "figure_reference_scatter",
    "figure_scaling_ari",
    "figure_scaling_runtime",
    "figure_scenario_dashboard",
    "figure_scenario_overview",
    "marker_table",
    "plos_figure_path",
    "prepare_composite",
]
