"""Levine et al. 32-dimension CyTOF bone marrow data, via HDCytoData.

Every load reads the data through rpy2 from the Bioconductor package
HDCytoData, which downloads them from ExperimentHub on first use and keeps
them in R's own cache. No copy is kept in this repository or under data/, so
the analysis always starts from the public source, and the preprocessing below
is the only step between the two.

HDCytoData's Levine_32dim_SE holds 265,627 cells from two healthy donors on
39 channels. Manual gating assigned 104,184 of them to 14 populations; the
other 161,443 carry the population "unassigned". The loader keeps the 32 type
markers (marker_class "type"), arcsinh transforms with cofactor 5, robust
scales, and then drops the unassigned cells.

The ordering of the last two steps is deliberate and is preserved from the
code that produced the published results: RobustScaler is fit on all 265,627
cells, the unassigned ones included, and the unassigned cells are dropped from
the already-scaled matrix afterwards. The scaler's quantiles therefore reflect
cells that are not in the returned data. Do not "correct" this without
re-running the case study; it changes the numbers.

Labels come from R as the factor's integer codes, are subsampled as codes, and
are returned as population names. The unassigned level is code 15, which is
what older notes and the manuscript call "population 15". Stratifying on the
codes rather than the names keeps the subsample the one this loader has always
drawn: StratifiedShuffleSplit walks the classes in sorted order, and the codes
("1", "10", "11", ...) sort differently from the names.
"""

import numpy as np
import pandas as pd

_COFACTOR = 5.0
_QUANTILE_RANGE = (10, 90)

#: Levels of HDCytoData's population_id factor in level order, so a cell's
#: integer code is its level's position here plus one.
_POPULATION_LEVELS = (
    "Basophils",
    "CD16-_NK_cells",
    "CD16+_NK_cells",
    "CD34+CD38+CD123-_HSPCs",
    "CD34+CD38+CD123+_HSPCs",
    "CD34+CD38lo_HSCs",
    "CD4_T_cells",
    "CD8_T_cells",
    "Mature_B_cells",
    "Monocytes",
    "pDCs",
    "Plasma_B_cells",
    "Pre_B_cells",
    "Pro_B_cells",
    "unassigned",
)
_UNASSIGNED_CODE = str(_POPULATION_LEVELS.index("unassigned") + 1)
_POPULATION_NAMES = {
    str(code): name
    for code, name in enumerate(_POPULATION_LEVELS, start=1)
    if name != "unassigned"
}


def _hdcytodata_is_installed() -> bool:
    """True when the R package HDCytoData is available. Split out so tests can stub it.

    Returns False rather than raising when rpy2 itself cannot be imported, so a
    machine without rpy2 reaches load_levine32's RuntimeError instead of a bare
    ModuleNotFoundError. The same arrangement as _m3c._m3c_is_installed.
    """
    try:
        import rpy2.robjects as ro
    except ImportError:
        return False
    return bool(ro.r('isTRUE(requireNamespace("HDCytoData", quietly=TRUE))')[0])


def _install_hdcytodata() -> None:
    """Install HDCytoData via BiocManager. Only reached when allow_install=True."""
    import rpy2.robjects as ro

    ro.r(
        'if (!requireNamespace("BiocManager", quietly=TRUE)) '
        'install.packages("BiocManager", repos="https://cran.rstudio.com/"); '
        'BiocManager::install("HDCytoData", ask=FALSE, update=FALSE)'
    )


def _check_levels(levels: list[str]) -> None:
    """Raise unless the factor's levels are the ones the codes are read with."""
    if tuple(levels) != _POPULATION_LEVELS:
        raise RuntimeError(
            "The population_id levels of HDCytoData's Levine_32dim_SE are not the "
            "ones this loader maps codes with, so its codes would name the wrong "
            f"populations. Got {list(levels)}."
        )


def _population_names(codes: pd.Series) -> pd.Series:
    """Map integer codes to population names, refusing unknown codes."""
    unknown = sorted(set(codes) - set(_POPULATION_NAMES))
    if unknown:
        raise ValueError(
            f"Levine population codes {unknown} name no gated population; the "
            "unassigned cells (15) should have been dropped in R."
        )
    return codes.map(_POPULATION_NAMES).rename("population")


def _load_from_r() -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str]]:
    """Fetch and preprocess the dataset through rpy2.

    Returns the scaled matrix, the population codes, the marker names and the
    HDCytoData and R versions the data were read with.
    """
    import rpy2.robjects as ro
    from sklearn.preprocessing import RobustScaler

    ro.r(
        "suppressPackageStartupMessages({"
        "library(HDCytoData); library(SummarizedExperiment)})"
    )
    ro.r("sce <- suppressMessages(Levine_32dim_SE())")

    X_all = np.array(ro.r("assay(sce)"), dtype=float)

    # Codes and levels are read explicitly. Converting rowData through
    # pandas2ri also yields the factor's integer codes, never its labels,
    # which once left a filter on "unassigned" matching no cell at all.
    _check_levels([str(level) for level in ro.r("levels(rowData(sce)$population_id)")])
    codes = np.asarray(ro.r("as.integer(rowData(sce)$population_id)")).astype(str)

    type_markers = np.asarray(
        ro.r('as.character(colData(sce)$marker_class) == "type"')
    ).astype(bool)
    channels = [str(name) for name in ro.r("rownames(colData(sce))")]
    markers = [name for name, keep in zip(channels, type_markers) if keep]

    # The scaler is fit on every cell, unassigned included. See module docstring.
    X_transformed = np.arcsinh(X_all[:, type_markers].astype(np.float64) / _COFACTOR)
    X_scaled = RobustScaler(quantile_range=_QUANTILE_RANGE).fit_transform(X_transformed)

    versions = {
        "hdcytodata_version": str(
            ro.r('as.character(packageVersion("HDCytoData"))')[0]
        ),
        "r_version": str(ro.r("R.version.string")[0]),
    }

    keep = codes != _UNASSIGNED_CODE
    return X_scaled[keep], codes[keep], markers, versions


def load_levine32(
    *,
    allow_install: bool = False,
    subsample: int | float | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess the Levine 32-dimension dataset through R.

    Needs rpy2 (the notebooks extra) and the R package HDCytoData. Installing
    an R package is a side effect on the user's machine, so a missing
    HDCytoData is installed only with allow_install=True; `make levine-setup`
    installs it ahead of time.

    Returns
    -------
    (X, y, meta)
        y holds the population names, one of 14 per cell. meta records the
        HDCytoData and R versions the data were read with.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    if not _hdcytodata_is_installed():
        if not allow_install:
            raise RuntimeError(
                "The R package HDCytoData is not installed, and loading the Levine "
                "data needs it. Run `make levine-setup` (which calls "
                'BiocManager::install("HDCytoData")), or re-run with '
                "allow_install=True to permit the install here."
            )
        _install_hdcytodata()

    X, codes_arr, markers, versions = _load_from_r()
    codes = pd.Series(np.asarray(codes_arr).astype(str))

    n_full = X.shape[0]
    if subsample is not None:
        size = (
            int(subsample * n_full) if isinstance(subsample, float) else int(subsample)
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=size / n_full, random_state=random_state
        )
        # Stratified on the codes, not the names. See module docstring.
        _, idx = next(splitter.split(X, codes))
        X = X[idx]
        codes = codes.iloc[idx].reset_index(drop=True)

    y = _population_names(codes)

    meta = {
        "source": "Levine_32dim",
        "package": "HDCytoData",
        "citation": "Levine et al. 2015, Cell 162(1):184-197",
        **versions,
        "n_cells_full": int(n_full),
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "markers": markers,
        "label_name": "population",
        "preprocessing": [
            "keep the 32 type markers (marker_class == 'type')",
            f"arcsinh(x / {_COFACTOR})",
            f"RobustScaler(quantile_range={_QUANTILE_RANGE}) fit on all cells, "
            "the unassigned ones (population 15) included, deliberately",
            "drop the unassigned cells (population 15)",
        ],
        "subsample": subsample,
        "random_state": random_state,
    }
    return X, y, meta
