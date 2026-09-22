"""Run M3C on a case study through rpy2.

M3C (John et al., 2020) is Monte Carlo reference-based consensus clustering,
an R/Bioconductor package with no Python port. This module is the whole
bridge to it: it takes the same X a case study hands CARVE and the CVIs,
runs M3C at its published defaults, and returns the scores and labels as
plain pandas and numpy.

This module is a leaf. It imports the standard library, numpy and pandas,
and the cache-fingerprint helpers from ._artifacts -- the one import it
takes from this package, since _artifacts imports only ._registry and
._types, so no cycle appears. rpy2 is imported inside functions (run_m3c
and its two small rpy2-touching helpers) rather than at module scope: rpy2
lives in the [notebooks] extra, the benchmarks CI job installs
[dev,graph,benchmarks], and a module-level import would break collection of
every test here.
"""

import hashlib
import json
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ._artifacts import check_fingerprint, fingerprint, fingerprint_path

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


#: The column names given to X's rows before handing them to R. M3C converts
#: a matrix to a data frame and then runs gsub("X", "", colnames(...)), which
#: mangles any name containing an X -- R's own default V1, V2 names survive,
#: but so would a corrupted name from any other scheme. Plain c0, c1, ... are
#: unaffected by that substitution.
def _sample_names(n: int) -> list[str]:
    return [f"c{i}" for i in range(n)]


def _m3c_is_installed() -> bool:
    """True when the R package M3C is available. Split out so tests can stub it.

    Returns False rather than raising when rpy2 itself cannot be imported.
    run_m3c calls this before its own rpy2 import, so a machine without rpy2
    (the benchmarks CI job, or a developer who never installed the
    [notebooks] extra) reaches the RuntimeError below instead of a bare
    ModuleNotFoundError.
    """
    try:
        import rpy2.robjects as ro
    except ImportError:
        return False
    return bool(ro.r('isTRUE(requireNamespace("M3C", quietly=TRUE))')[0])


def _install_m3c() -> None:
    """Install M3C via BiocManager. Only reached when allow_install=True."""
    import rpy2.robjects as ro

    ro.r(
        'if (!requireNamespace("BiocManager", quietly=TRUE)) '
        'install.packages("BiocManager", repos="https://cran.rstudio.com/"); '
        'BiocManager::install("M3C", ask=FALSE, update=FALSE)'
    )


def align_assignment(
    values: np.ndarray,
    names_from_r: Sequence[str],
    names_expected: Sequence[str],
    *,
    k: int,
) -> np.ndarray:
    """Put one M3C assignment back into the row order of X.

    Pure, so the two ways this can go wrong are testable without R. They fail
    differently and are worth telling apart: the wrong count means the matrix
    reached M3C in the wrong orientation, while the right count under
    unrecognized names means M3C's gsub("X", "", colnames(...)) mangled them.
    """
    values = np.asarray(values)
    if values.size != len(names_expected):
        raise RuntimeError(
            f"M3C returned {values.size} labels at K={k} for an X with "
            f"{len(names_expected)} rows. M3C takes features in rows and "
            "samples in columns, so a result this size means the matrix "
            "reached it in the wrong orientation."
        )
    by_name = pd.Series(values, index=[str(name) for name in names_from_r]).reindex(
        list(names_expected)
    )
    if by_name.isna().any():
        raise RuntimeError(
            f"M3C's labels at K={k} do not cover every sample after "
            "reindexing, which means the names it returned are not the ones "
            f"it was given. Missing: {by_name[by_name.isna()].index.tolist()[:5]}."
        )
    return by_name.to_numpy(dtype=int)


def run_m3c(
    X: np.ndarray,
    *,
    max_k: int = 10,
    allow_install: bool = False,
    cores: int = 1,
    verbose: bool = False,
    **overrides: Any,
) -> M3CResult:
    """Run M3C on X at its published defaults.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The same matrix the case study hands CARVE and the CVIs. M3C wants
        features in rows, so this is transposed here rather than by the
        caller, and the result is checked against X's own row count.
    max_k : int, default=10
        M3C's maxK. It always sweeps K from 2 to maxK; there is no lower
        bound to set. The default is M3C's own, which is also Klein's CARVE
        ceiling.
    allow_install : bool, default=False
        Installing an R package is a side effect on the user's machine, so it
        is opt-in, matching datasets/_levine.py.
    cores : int, default=1
        M3C's own parallelism over the Monte Carlo iterations. Left at 1: the
        Klein run takes about eight minutes serially, and a single worker is
        one fewer variable between runs.
    **overrides
        Individual M3C_DEFAULTS entries to override. Used by the tests to
        drop iters; a case-study run passes none.

    Returns
    -------
    M3CResult
    """
    if max_k < 2:
        raise ValueError(
            f"max_k must be at least 2, since M3C sweeps K from 2 upward; got {max_k}."
        )

    unknown = sorted(set(overrides) - set(M3C_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown M3C parameter(s) {unknown}. "
            f"Overridable names are {sorted(M3C_DEFAULTS)}."
        )

    if not _m3c_is_installed():
        if not allow_install:
            raise RuntimeError(
                "The R package M3C is not installed, and this run needs it. "
                'Run `make m3c-setup` (which calls BiocManager::install("M3C")), '
                "or re-run with allow_install=True to permit the install here."
            )
        _install_m3c()

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri

    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]
    names = _sample_names(n_samples)

    config = (
        dict(M3C_DEFAULTS)
        | dict(overrides)
        | {
            "maxK": int(max_k),
            "cores": int(cores),
        }
    )

    ro.r("suppressPackageStartupMessages(library(M3C))")
    with (numpy2ri.converter + pandas2ri.converter).context():
        # Features in rows, samples in columns: M3C's required orientation.
        ro.globalenv["mydata"] = ro.r["as.data.frame"](X.T)
        ro.globalenv["sample_names"] = ro.StrVector(names)
        ro.r("colnames(mydata) <- sample_names")

        # M3C() is invoked as an R expression rather than as a Python call
        # whose return value rpy2 would auto-convert. M3C's return list
        # carries ggplot objects even under removeplots=TRUE, and at least
        # one of their components is an S4 object the active
        # numpy2ri/pandas2ri converter has no rpy2py registration for;
        # letting rpy2 convert the whole list raises
        # KeyError(rpy2.rinterface.SexpS4) before a single score is read.
        # Assigning res in R and pulling out only res$scores and the per-K
        # assignments -- both plain data frames and named vectors -- avoids
        # the plot objects entirely.
        ro.globalenv["m3c_maxK"] = config["maxK"]
        ro.globalenv["m3c_cores"] = config["cores"]
        ro.globalenv["m3c_iters"] = config["iters"]
        ro.globalenv["m3c_repsref"] = config["repsref"]
        ro.globalenv["m3c_repsreal"] = config["repsreal"]
        ro.globalenv["m3c_pItem"] = config["pItem"]
        ro.globalenv["m3c_clusteralg"] = config["clusteralg"]
        ro.globalenv["m3c_objective"] = config["objective"]
        ro.globalenv["m3c_method"] = config["method"]
        ro.globalenv["m3c_seed"] = config["seed"]
        ro.globalenv["m3c_silent"] = not verbose

        started = time.perf_counter()
        ro.r(
            "res <- M3C(mydata, maxK=m3c_maxK, cores=m3c_cores, iters=m3c_iters, "
            "repsref=m3c_repsref, repsreal=m3c_repsreal, pItem=m3c_pItem, "
            "clusteralg=m3c_clusteralg, objective=m3c_objective, "
            "method=m3c_method, seed=m3c_seed, removeplots=TRUE, "
            "silent=m3c_silent)"
        )
        runtime_s = time.perf_counter() - started

        scores = validate_scores(ro.r("res$scores"))
        labels = {}
        for k in range(2, int(max_k) + 1):
            # realdataresults is indexed by K itself, so element 1 is unset
            # and element k holds the K=k result. assignments is the named
            # vector in the original column order; ordered_annotation is
            # permuted by the dendrogram order and would scramble the mapping.
            assignment = ro.r(f"res$realdataresults[[{k}]]$assignments")
            names_from_r = ro.r(f"names(res$realdataresults[[{k}]]$assignments)")
            labels[k] = align_assignment(
                np.asarray(assignment), list(names_from_r), names, k=k
            )

        m3c_version = str(ro.r('as.character(packageVersion("M3C"))')[0])
        r_version = str(ro.r("R.version.string")[0])

    selected_k = select_k_m3c(scores)
    reported_k = int(len(np.unique(labels[selected_k])))
    if reported_k != selected_k:
        warnings.warn(
            f"M3C's partition at its selected K={selected_k} has {reported_k} "
            "distinct labels. This happens when a consensus cluster comes back "
            "empty; the selection stands, but the partition is coarser than K.",
            stacklevel=2,
        )

    p_value = float(scores.loc[scores["K"] == selected_k, "MONTECARLO_P"].iloc[0])
    if verbose:
        print(
            f"M3C selected K={selected_k} (Monte Carlo p={p_value:.4f}) "
            f"in {runtime_s:.1f} s."
        )
    return M3CResult(
        scores=scores.reset_index(drop=True),
        labels=labels,
        selected_k=selected_k,
        p_value=p_value,
        runtime_s=runtime_s,
        config=config,
        m3c_version=m3c_version,
        r_version=r_version,
    )


def m3c_cache_path(
    *, study_name: str, scale: str, root: Path, config: Mapping[str, Any]
) -> Path:
    """Where a study's M3C result is cached, per scale and per configuration.

    The config hash is part of the filename for the same reason
    carve_cache_path carries a run key: run_or_load_m3c loads whatever file
    sits at the path it is handed, so two runs that differ in maxK or in the
    inner algorithm must not share a name.
    """
    key = hashlib.sha1(
        json.dumps(dict(config), sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return Path(root) / f"m3c_{study_name}_{scale}_{key}.parquet"


def _write_cache(path: Path, result: M3CResult, X: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result.scores.to_parquet(path, index=False)
    pd.DataFrame(result.labels).to_parquet(_labels_path(path), index=False)
    _sidecar_path(path).write_text(
        json.dumps(
            {
                "selected_k": result.selected_k,
                "p_value": result.p_value,
                "runtime_s": result.runtime_s,
                "config": dict(result.config),
                "m3c_version": result.m3c_version,
                "r_version": result.r_version,
            },
            indent=2,
            default=str,
        )
    )
    fingerprint_path(path).write_text(fingerprint(X))


def _labels_path(path: Path) -> Path:
    return path.with_name(path.stem + ".labels.parquet")


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.stem + ".json")


def _read_cache(path: Path) -> M3CResult:
    scores = validate_scores(pd.read_parquet(path))
    labels_frame = pd.read_parquet(_labels_path(path))
    meta = json.loads(_sidecar_path(path).read_text())
    return M3CResult(
        scores=scores.reset_index(drop=True),
        labels={
            int(column): labels_frame[column].to_numpy(dtype=int)
            for column in labels_frame.columns
        },
        selected_k=int(meta["selected_k"]),
        p_value=float(meta["p_value"]),
        runtime_s=float(meta["runtime_s"]),
        config=meta["config"],
        m3c_version=str(meta["m3c_version"]),
        r_version=str(meta["r_version"]),
    )


def run_or_load_m3c(
    X: np.ndarray, *, cache_path: Path, force: bool = False, **kwargs: Any
) -> M3CResult:
    """Run M3C on X, caching the result, and serve the cache on later calls.

    A Klein run is about eight minutes, so the cache is what makes
    regenerating the figure practical. The fingerprint guard is the same one
    fit_or_load_carve uses: a cached result is never served against a matrix
    it was not computed on.
    """
    cache_path = Path(cache_path)
    if cache_path.is_file() and not force:
        check_fingerprint(cache_path, X)
        return _read_cache(cache_path)

    result = run_m3c(X, **kwargs)
    _write_cache(cache_path, result, X)
    return result
