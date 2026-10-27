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

import contextlib
import math
import os
import sys
import time
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


#: The R side of run_sc3: SC3's own sequence, step by step. sc3() with
#: k_estimator = TRUE runs sc3_prepare, sc3_estimate_k, sc3_calc_dists,
#: sc3_calc_transfs, sc3_kmeans and sc3_calc_consens in that order;
#: sc3_estimate_k draws no random numbers, so reading the spectrum between
#: the steps and stopping before clustering when k < 2 changes nothing about
#: the random numbers the clustering sees. Wrapping the steps in a function
#: keeps the distance, transformation and consensus matrices out of R's
#: global environment once it returns.
_SC3_R_FUNCTION = r"""
function(logcounts, gene_filter, seed, n_cores, pct_dropout_min,
         pct_dropout_max, d_region_min, d_region_max, svm_max,
         kmeans_iter_max, quiet) {
  suppressPackageStartupMessages({
    library(SC3)
    library(SingleCellExperiment)
  })
  step <- function(f, ...) {
    if (!quiet) return(f(...))
    out <- NULL
    utils::capture.output(out <- suppressMessages(f(...)), type = "output")
    out
  }
  assays <- list(logcounts = logcounts)
  if (gene_filter) assays$counts <- expm1(logcounts)
  sce <- SingleCellExperiment(assays = assays)
  rowData(sce)$feature_symbol <- rownames(logcounts)
  set.seed(seed)
  sce <- step(sc3_prepare, sce, gene_filter = gene_filter,
              pct_dropout_min = pct_dropout_min,
              pct_dropout_max = pct_dropout_max,
              d_region_min = d_region_min, d_region_max = d_region_max,
              svm_max = svm_max, n_cores = n_cores,
              kmeans_iter_max = kmeans_iter_max, rand_seed = seed)
  sce <- step(sc3_estimate_k, sce)
  k <- as.integer(metadata(sce)$sc3$k_estimation)
  dataset <- SC3:::get_processed_dataset(sce)
  n <- nrow(dataset)
  p <- ncol(dataset)
  mu_tw <- (sqrt(n - 1) + sqrt(p))^2
  sigma_tw <- (sqrt(n - 1) + sqrt(p)) * (1 / sqrt(n - 1) + 1 / sqrt(p))^(1 / 3)
  bound <- 3.273 * sigma_tw + mu_tw
  eigenvalues <- eigen(SC3:::tmult(scale(dataset)), symmetric = TRUE,
                       only.values = TRUE)$values
  labels <- integer(0)
  if (k >= 2) {
    sce <- step(sc3_calc_dists, sce)
    sce <- step(sc3_calc_transfs, sce)
    sce <- step(sc3_kmeans, sce, ks = k)
    sce <- step(sc3_calc_consens, sce)
    labels <- as.integer(as.character(
      colData(sce)[[paste0("sc3_", k, "_clusters")]]))
  }
  list(
    k = k, bound = bound, eigenvalues = eigenvalues, labels = labels,
    cells = colnames(sce), n_features_used = as.integer(n),
    n_dim = as.integer(metadata(sce)$sc3$n_dim),
    kmeans_nstart = as.integer(metadata(sce)$sc3$kmeans_nstart),
    n_cores = as.integer(metadata(sce)$sc3$n_cores),
    sc3_version = as.character(packageVersion("SC3")),
    r_version = R.version.string
  )
}
"""


def _sc3_is_installed() -> bool:
    """True when the R package SC3 is available. Split out so tests can stub it.

    Returns False rather than raising when rpy2 itself cannot be imported, so
    a machine without rpy2 reaches run_sc3's RuntimeError instead of a bare
    ModuleNotFoundError.
    """
    try:
        import rpy2.robjects as ro
    except ImportError:
        return False
    return bool(ro.r('isTRUE(requireNamespace("SC3", quietly=TRUE))')[0])


def _install_sc3() -> None:
    """Install SC3 via BiocManager. Only reached when allow_install=True."""
    import rpy2.robjects as ro

    ro.r(
        'if (!requireNamespace("BiocManager", quietly=TRUE)) '
        'install.packages("BiocManager", repos="https://cran.rstudio.com/"); '
        'BiocManager::install("SC3", ask=FALSE, update=FALSE)'
    )


@contextlib.contextmanager
def _silenced_fds(active: bool):
    """Point file descriptors 1 and 2 at os.devnull for the duration, if active.

    SC3 starts its worker processes with parallel::makeCluster(outfile = ""),
    so each worker writes its startup lines, package warnings and the k-means
    progress bar straight to the descriptors it inherits. capture.output and
    suppressMessages only reach the master R process. Redirecting the
    descriptors before the workers are spawned silences them, and restoring
    them afterward leaves later output untouched.
    """
    if not active:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def run_sc3(
    X: np.ndarray,
    *,
    gene_filter: bool,
    seed: int = _DEFAULT_SEED,
    n_cores: int | None = None,
    allow_install: bool = False,
    verbose: bool = False,
) -> SC3Result:
    """Run SC3 on X at its defaults and cluster at its own Tracy-Widom k.

    Parameters
    ----------
    X : ndarray, shape (n_cells, n_features)
        The same matrix the case study hands CARVE. SC3 wants features in
        rows, so it is transposed here. It becomes SC3's logcounts.
    gene_filter : bool
        SC3's dropout filter (keep genes with 10 to 90 percent zeros). With
        it on, counts are derived as expm1(X), which is zero exactly where X
        is; the filter reads nothing else from counts.
    seed : int, default=1
        Passed to set.seed immediately before sc3_prepare, and as rand_seed.
    n_cores : int or None, default=None
        SC3's worker count; None is SC3's own default (all cores but one).
        It stays out of the cache key because SC3 draws its random numbers
        through doRNG, which assigns one stream per task rather than per
        worker. TestRunLive checks that 2 and 3 cores agree on one fixture;
        it does not prove the invariance in general.
    allow_install : bool, default=False
        Installing an R package is a side effect on the user's machine, so it
        is opt-in, as in _m3c.run_m3c.
    verbose : bool, default=False
        Show SC3's own messages and progress bars. When False, the R
        process's messages are suppressed and file descriptors 1 and 2 are
        redirected to os.devnull while SC3 runs, which also silences its
        worker processes.
    """
    X = check_input(X, gene_filter=gene_filter)
    if not _sc3_is_installed():
        if not allow_install:
            raise RuntimeError(
                "The R package SC3 is not installed, and this run needs it. Run "
                '`make sc3-setup` (which calls BiocManager::install("SC3")), or '
                "re-run with allow_install=True to permit the install here."
            )
        _install_sc3()

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri

    n_cells, n_features = X.shape
    cells = _cell_names(n_cells)
    config = _config(gene_filter=gene_filter, seed=seed)

    with (numpy2ri.converter + pandas2ri.converter).context():
        try:
            ro.globalenv["carve_sc3_logcounts"] = np.ascontiguousarray(X.T)
            ro.globalenv["carve_sc3_features"] = ro.StrVector(
                _feature_names(n_features)
            )
            ro.globalenv["carve_sc3_cells"] = ro.StrVector(cells)
            ro.r(
                "dimnames(carve_sc3_logcounts) <- "
                "list(carve_sc3_features, carve_sc3_cells)"
            )
            ro.r(f"carve_sc3_fn <- {_SC3_R_FUNCTION}")
            ro.globalenv["carve_sc3_gene_filter"] = bool(gene_filter)
            ro.globalenv["carve_sc3_seed"] = int(seed)
            if n_cores is None:
                ro.r("carve_sc3_n_cores <- NULL")
            else:
                ro.globalenv["carve_sc3_n_cores"] = int(n_cores)
            for name in (
                "pct_dropout_min",
                "pct_dropout_max",
                "d_region_min",
                "d_region_max",
                "svm_max",
                "kmeans_iter_max",
            ):
                ro.globalenv[f"carve_sc3_{name}"] = SC3_DEFAULTS[name]
            ro.globalenv["carve_sc3_quiet"] = not verbose

            started = time.perf_counter()
            with _silenced_fds(not verbose):
                ro.r(
                    "carve_sc3_result <- carve_sc3_fn(carve_sc3_logcounts, "
                    "carve_sc3_gene_filter, carve_sc3_seed, carve_sc3_n_cores, "
                    "carve_sc3_pct_dropout_min, carve_sc3_pct_dropout_max, "
                    "carve_sc3_d_region_min, carve_sc3_d_region_max, "
                    "carve_sc3_svm_max, carve_sc3_kmeans_iter_max, carve_sc3_quiet)"
                )
            runtime_s = time.perf_counter() - started

            def scalar(name: str) -> Any:
                return np.asarray(ro.r(f"carve_sc3_result${name}"))[0]

            k = int(scalar("k"))
            bound = float(scalar("bound"))
            eigenvalues = np.asarray(ro.r("carve_sc3_result$eigenvalues"), dtype=float)
            raw_labels = np.asarray(ro.r("carve_sc3_result$labels"))
            names_from_r = [str(name) for name in ro.r("carve_sc3_result$cells")]
            n_features_used = int(scalar("n_features_used"))
            n_dim = tuple(int(d) for d in np.asarray(ro.r("carve_sc3_result$n_dim")))
            kmeans_nstart = int(scalar("kmeans_nstart"))
            used_cores = int(scalar("n_cores"))
            sc3_version = str(scalar("sc3_version"))
            r_version = str(scalar("r_version"))
        finally:
            ro.r("rm(list = ls(pattern = '^carve_sc3_')); invisible(gc())")

    if k < 2:
        raise RuntimeError(
            f"SC3 estimated k={k} from its Tracy-Widom test, so it found no "
            "clustering to compare against the case study's other methods."
        )
    check_spectrum_selection(eigenvalues, bound, k_estimation=k)
    expected_bound = tracy_widom_bound(n_features_used, n_cells)
    if not math.isclose(bound, expected_bound, rel_tol=1e-9):
        raise RuntimeError(
            f"SC3's Tracy-Widom bound is {bound:.6f}, but the formula restated in "
            f"tracy_widom_bound gives {expected_bound:.6f} for {n_features_used} "
            f"features and {n_cells} cells. SC3:::estkTW has changed."
        )
    labels = align_labels(raw_labels, names_from_r, cells, k=k)
    spectrum = pd.DataFrame(
        {"index": np.arange(1, len(eigenvalues) + 1), "eigenvalue": eigenvalues}
    )
    if verbose:
        print(f"SC3 estimated k={k} in {runtime_s:.1f} s on {used_cores} cores.")
    return SC3Result(
        labels=labels,
        k=k,
        spectrum=spectrum,
        bound=bound,
        n_features_used=n_features_used,
        n_dim=n_dim,
        kmeans_nstart=kmeans_nstart,
        n_cores=used_cores,
        runtime_s=runtime_s,
        config=config,
        sc3_version=sc3_version,
        r_version=r_version,
    )
