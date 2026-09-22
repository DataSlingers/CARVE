"""Run M3C on a case study through rpy2.

M3C (John et al., 2020) is Monte Carlo reference-based consensus clustering,
an R/Bioconductor package with no Python port. This module is the whole
bridge to it: it takes the same X a case study hands CARVE and the CVIs,
runs M3C at its published defaults, and returns the scores and labels as
plain pandas and numpy.

This module is a leaf. It imports the standard library, numpy and pandas,
and nothing else from this package. rpy2 is imported inside run_m3c rather
than at module scope: rpy2 lives in the [notebooks] extra, the benchmarks CI
job installs [dev,graph,benchmarks], and a module-level import would break
collection of every test here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

#: M3C 1.30.0's published defaults, restated here rather than left to R. A
#: Bioconductor release that changes one of these then shows up as a diff in
#: this file instead of silently changing a manuscript number.
#:
#: removeplots and silent are deliberately absent: they are presentation
#: flags, not method flags, and run_m3c sets them itself.
M3C_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "method": 1,
        "clusteralg": "pam",
        "objective": "entropy",
        "iters": 25,
        "repsref": 100,
        "repsreal": 100,
        "pItem": 0.8,
        "seed": 123,
    }
)

#: The columns M3C's scores frame carries under method=1 and the default
#: objective="entropy".
#:
#: M3C builds the frame with columns named PAC_REAL and PAC_REF and then,
#: on the last line of the entropy branch, renames columns 2 and 3 in place:
#: colnames(real)[2:3] <- c("ENTROPY_REAL", "ENTROPY_REF"). The PAC names
#: survive only under objective="PAC", which also substitutes BETA_P for
#: NORM_P. Reading PAC_REAL off a default run raises KeyError.
SCORES_COLUMNS: tuple[str, ...] = (
    "K",
    "ENTROPY_REAL",
    "ENTROPY_REF",
    "RCSI",
    "RCSI_SE",
    "MONTECARLO_P",
    "NORM_P",
    "P_SCORE",
)

#: The PAC-objective spellings of columns 2 and 3, recognized only so that a
#: run under the wrong objective fails with a message that says so.
_PAC_COLUMNS: frozenset[str] = frozenset({"PAC_REAL", "PAC_REF"})


@dataclass(frozen=True)
class M3CResult:
    """One M3C run, as plain pandas and numpy.

    labels carries every K M3C clustered, not only the selected one, so a
    caller can ask what M3C says at a K chosen by something else. Labels come
    from res$realdataresults[[k]]$assignments, which is the named vector in
    the original column order; ordered_annotation is permuted by the
    dendrogram order and would scramble the mapping back onto X's rows.
    """

    scores: pd.DataFrame
    labels: dict[int, np.ndarray]
    selected_k: int
    p_value: float
    runtime_s: float
    config: Mapping[str, Any] = field(default_factory=dict)
    m3c_version: str = ""
    r_version: str = ""


def validate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Check a scores frame against the entropy-objective schema.

    Returns the frame with its columns in SCORES_COLUMNS order, so callers
    downstream of this do not depend on R's column order.
    """
    columns = set(scores.columns)
    if columns & _PAC_COLUMNS:
        raise ValueError(
            "This scores frame carries "
            f"{sorted(columns & _PAC_COLUMNS)}, which M3C produces only under "
            'objective="PAC". This module pins objective="entropy", so the '
            "run that produced this frame is not the one it describes."
        )
    missing = [name for name in SCORES_COLUMNS if name not in columns]
    if missing:
        raise ValueError(
            f"M3C's scores frame is missing {missing}. Expected the "
            f"entropy-objective schema {list(SCORES_COLUMNS)}, got "
            f"{sorted(columns)}."
        )
    return scores.loc[:, list(SCORES_COLUMNS)]


def select_k_m3c(scores: pd.DataFrame) -> int:
    """The K M3C selects: the one at the RCSI maximum.

    M3C's own expression is which.max(real$RCSI) + 1, where the plus one
    converts a 1-based row index into a K value because K starts at 2.
    Transcribing that arithmetic into Python would be wrong twice over, since
    numpy's argmax is 0-based. Read the K column instead of recomputing the
    offset -- the same reason config_id is a join key in carve and never a
    positional index.
    """
    if "RCSI" not in scores.columns:
        raise KeyError(
            "This frame has no RCSI column, so M3C's selection cannot be "
            f"recovered from it. Columns: {sorted(scores.columns)}."
        )
    return int(scores.loc[scores["RCSI"].idxmax(), "K"])
