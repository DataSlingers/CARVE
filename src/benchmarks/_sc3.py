"""Run SC3 on a case study through rpy2.

SC3 (Kiselev et al., 2017) is single-cell consensus clustering, an
R/Bioconductor package with no Python port. This module is the whole bridge
to it: it takes the same X a case study hands CARVE, M3C and the CVIs, runs
SC3 at its defaults with its own Tracy-Widom estimate of k, and returns the
labels and the spectrum behind the estimate as plain numpy and pandas.

SC3's vignette names two routes to k: visual exploration, which is analyst
judgment, and sc3_estimate_k, which counts the data's eigenvalues above a
Tracy-Widom bound. Only the second is a rule, so only the second is run.

This module is a leaf, like _m3c.py: the standard library, numpy, pandas,
and the cache-fingerprint helpers from ._artifacts. rpy2 is imported inside
functions, since it lives in the [notebooks] extra and the benchmarks CI job
does not install it.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

#: SC3 1.36.0's defaults for the settings this run depends on, restated so a
#: Bioconductor release that changes one shows up as a diff here. kmeans_nstart
#: is left to SC3's own rule (50 above 2,000 cells, else 1,000) and recorded.
SC3_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "pct_dropout_min": 10,
        "pct_dropout_max": 90,
        "d_region_min": 0.04,
        "d_region_max": 0.07,
        "svm_max": 5000,
        "kmeans_iter_max": 1e9,
        "biology": False,
    }
)

#: SC3's rand_seed default. SC3 1.36.0 stores rand_seed but its clustering
#: steps never read it: they run foreach with doRNG, which seeds each task
#: from R's global random state. run_sc3 therefore calls set.seed(seed)
#: itself, immediately before sc3_prepare.
_DEFAULT_SEED = 1


@dataclass(frozen=True)
class SC3Result:
    """One SC3 run at its own k, as plain numpy and pandas.

    labels are in the row order of the X that was passed. spectrum holds
    every eigenvalue sc3_estimate_k thresholds, largest first, with a 1-based
    index; k of them lie above bound.
    """

    labels: np.ndarray
    k: int
    spectrum: pd.DataFrame
    bound: float
    n_features_used: int
    n_dim: tuple[int, ...]
    kmeans_nstart: int
    n_cores: int
    runtime_s: float
    config: Mapping[str, Any] = field(default_factory=dict)
    sc3_version: str = ""
    r_version: str = ""


def _config(*, gene_filter: bool, seed: int) -> dict[str, Any]:
    """The settings a run records and its cache is checked against."""
    return dict(SC3_DEFAULTS) | {"gene_filter": bool(gene_filter), "seed": int(seed)}


def _cell_names(n: int) -> list[str]:
    return [f"c{i}" for i in range(n)]


def _feature_names(p: int) -> list[str]:
    return [f"g{j + 1}" for j in range(p)]


def tracy_widom_bound(n_features: int, n_cells: int) -> float:
    """The bound sc3_estimate_k counts eigenvalues against.

    SC3:::estkTW in SC3 1.36.0, restated. With n features and p cells,
    muTW = (sqrt(n - 1) + sqrt(p))^2 and
    sigmaTW = (sqrt(n - 1) + sqrt(p)) * (1 / sqrt(n - 1) + 1 / sqrt(p))^(1/3),
    and the bound is 3.273 * sigmaTW + muTW. run_sc3 checks the bound R
    reports against this one, so a change in SC3's formula fails loudly.
    """
    root_n = math.sqrt(n_features - 1)
    root_p = math.sqrt(n_cells)
    mu = (root_n + root_p) ** 2
    sigma = (root_n + root_p) * (1.0 / root_n + 1.0 / root_p) ** (1.0 / 3.0)
    return 3.273 * sigma + mu


def check_spectrum_selection(
    eigenvalues: np.ndarray, bound: float, *, k_estimation: int
) -> None:
    """Check that the recomputed spectrum gives the k SC3 estimated.

    The counterpart of _m3c.check_own_selection. The spectrum is recomputed
    in R with two unexported SC3 functions; if their behavior changes, the
    count above the bound stops matching SC3's own k_estimation.
    """
    above = int(np.sum(np.asarray(eigenvalues, dtype=float) > bound))
    if above != int(k_estimation):
        raise RuntimeError(
            f"{above} eigenvalues lie above the Tracy-Widom bound {bound:.3f}, but "
            f"SC3 estimated k={k_estimation}. The spectrum recomputed here no "
            "longer matches SC3:::estkTW, so SC3's internals have changed."
        )


def align_labels(
    values: Any,
    names_from_r: Sequence[str],
    names_expected: Sequence[str],
    *,
    k: int,
) -> np.ndarray:
    """Put SC3's labels back into the row order of X and check there are k.

    A wrong count means the matrix reached SC3 in the wrong orientation;
    the right count under unrecognized names means the names were changed on
    the way. SC3 cuts a hierarchical clustering at k, so anything other than
    exactly k distinct labels means the run is not the one it describes.
    """
    values = np.asarray(values)
    if values.size != len(names_expected):
        raise RuntimeError(
            f"SC3 returned {values.size} labels for an X with "
            f"{len(names_expected)} rows. SC3 takes features in rows and cells in "
            "columns, so a result this size means the matrix reached it in the "
            "wrong orientation."
        )
    by_name = pd.Series(values, index=[str(name) for name in names_from_r]).reindex(
        list(names_expected)
    )
    if by_name.isna().any():
        raise RuntimeError(
            "SC3's labels do not cover every cell after reindexing, so the names "
            "it returned are not the ones it was given. Missing: "
            f"{by_name[by_name.isna()].index.tolist()[:5]}."
        )
    labels = by_name.to_numpy(dtype=int)
    n_distinct = len(np.unique(labels))
    if n_distinct != int(k):
        raise RuntimeError(
            f"SC3's partition at its estimate k={k} has {n_distinct} distinct "
            "labels; its hierarchical cut gives exactly k."
        )
    return labels


def check_input(X: np.ndarray, *, gene_filter: bool) -> np.ndarray:
    """Validate X before it reaches R, and return it as float64.

    Above svm_max cells SC3 clusters a training subset and transfers labels
    by SVM, a different method from the one this bridge runs, so that is
    refused. The gene filter reads counts, which run_sc3 derives as
    expm1(X); negative values mean X is not log1p of counts.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, cells by features; got {X.ndim}.")
    svm_max = int(SC3_DEFAULTS["svm_max"])
    if X.shape[0] > svm_max:
        raise ValueError(
            f"X has {X.shape[0]:,} cells, above SC3's svm_max of {svm_max:,}. SC3 "
            "would cluster a training subset and transfer labels by SVM, a "
            "different method from the one this bridge runs."
        )
    if gene_filter and (X < 0).any():
        raise ValueError(
            "SC3's gene filter reads counts, which run_sc3 derives as expm1(X), "
            "and X has negative values, so it is not log1p of counts. Pass "
            "gene_filter=False."
        )
    return X
