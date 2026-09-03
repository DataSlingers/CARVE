"""One module per manuscript figure.

Every function returns a Figure and, unless save=False, writes it under its
exact manuscript filename at 300 dpi. Four of these figures previously had no
savefig at all and were right-click-saved out of Jupyter output.
"""

from ._benchmarking_examples import figure_benchmarking_examples
from ._paths import BENCHMARKING_DIR, CASE_STUDY_DIR, VIS_ROOT, figure_path

__all__ = [
    "BENCHMARKING_DIR",
    "CASE_STUDY_DIR",
    "VIS_ROOT",
    "figure_benchmarking_examples",
    "figure_path",
]
