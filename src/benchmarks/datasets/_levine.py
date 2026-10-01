"""Levine et al. 32-dimension CyTOF bone marrow data, via HDCytoData.

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

Labels are cached and subsampled as the factor's integer codes and returned as
population names. The unassigned level is code 15, which is what older notes
and the manuscript call "population 15". Stratifying on the codes rather than
the names keeps the subsample the one this loader has always drawn:
StratifiedShuffleSplit walks the classes in sorted order, and the codes ("1",
"10", "11", ...) sort differently from the names.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ._klein import DATA_ROOT

_CACHE_NAME = "levine32_preprocessed.npz"
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


def _cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / _CACHE_NAME


def _write_cache(
    cache_dir: Path, X: np.ndarray, y: np.ndarray, markers: list[str]
) -> Path:
    path = _cache_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=np.asarray(y).astype(str),
        markers=np.asarray(markers, dtype=object),
    )
    return path


def _read_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    path = _cache_path(cache_dir)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as payload:
        return payload["X"], payload["y"], [str(m) for m in payload["markers"]]


def _check_levels(levels: list[str]) -> None:
    """Raise unless the factor's levels are the ones the codes are read with."""
    if tuple(levels) != _POPULATION_LEVELS:
        raise RuntimeError(
            "The population_id levels of HDCytoData's Levine_32dim_SE are not the "
            "ones this loader maps codes with, so its codes would name the wrong "
            f"populations. Got {list(levels)}."
        )


def _population_names(codes: pd.Series) -> pd.Series:
    """Map cached integer codes to population names, refusing unknown codes."""
    unknown = sorted(set(codes) - set(_POPULATION_NAMES))
    if unknown:
        raise ValueError(
            f"Levine population codes {unknown} name no gated population; the "
            "cache holds labels this loader did not write."
        )
    return codes.map(_POPULATION_NAMES).rename("population")


def _load_from_r(allow_install: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch and preprocess the dataset through rpy2.

    Installing an R package is a side effect on the user's machine, so it is
    opt-in. The previous loader ran BiocManager::install unconditionally as a
    side effect of executing a notebook cell.
    """
    if not allow_install:
        raise RuntimeError(
            "The Levine dataset is not cached and fetching it requires the R package "
            "HDCytoData, which may need installing. Re-run with allow_install=True to "
            "permit BiocManager::install, or place a prepared cache at the cache_dir."
        )

    import rpy2.robjects as ro
    from sklearn.preprocessing import RobustScaler

    ro.r(r"""
    if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
    if (!requireNamespace("HDCytoData", quietly=TRUE)) BiocManager::install("HDCytoData")
    library(HDCytoData)
    library(SummarizedExperiment)
    """)
    ro.r("sce <- Levine_32dim_SE()")

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

    keep = codes != _UNASSIGNED_CODE
    return X_scaled[keep], codes[keep], markers


def load_levine32(
    *,
    cache_dir: Path | None = None,
    allow_install: bool = False,
    subsample: int | float | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess the Levine 32-dimension dataset.

    Reads a local cache when one exists, and otherwise goes through rpy2,
    which requires allow_install=True.

    Returns
    -------
    (X, y, meta)
        y holds the population names, one of 14 per cell.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_ROOT / "refs"

    cached = _read_cache(cache_dir)
    from_cache = cached is not None
    if cached is None:
        X, codes_arr, markers = _load_from_r(allow_install)
        _write_cache(cache_dir, X, codes_arr, markers)
    else:
        X, codes_arr, markers = cached

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
        "n_cells_full": int(n_full),
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "markers": markers,
        "label_name": "population",
        "from_cache": bool(from_cache),
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
