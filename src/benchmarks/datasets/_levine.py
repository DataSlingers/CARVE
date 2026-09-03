"""Levine et al. 32-dimension CyTOF bone marrow data, via HDCytoData.

Cells with an assigned population are kept, restricted to the type markers
(marker_class == 2), arcsinh transformed with cofactor 5, robust scaled, and
population 15 is then removed.

The ordering of the last two steps is deliberate and is preserved from the
code that produced the published results: RobustScaler is fit on the labeled
cells while population 15 is still present, and population 15 is dropped from
the already-scaled matrix afterwards. The scaler's quantiles therefore
reflect a population that is not in the returned data. Do not "correct" this
without re-running the case study; it changes the numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ._klein import DATA_ROOT

_CACHE_NAME = "levine32_preprocessed.npz"
_COFACTOR = 5.0
_QUANTILE_RANGE = (10, 90)
_DROPPED_POPULATION = "15"


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
    from rpy2.robjects import pandas2ri
    from sklearn.preprocessing import RobustScaler

    ro.r(r"""
    if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
    if (!requireNamespace("HDCytoData", quietly=TRUE)) BiocManager::install("HDCytoData")
    library(HDCytoData)
    library(SummarizedExperiment)
    """)
    ro.r("sce <- Levine_32dim_SE()")

    X_all = np.array(ro.r("assay(sce)"), dtype=float)
    row_df = pandas2ri.rpy2py(ro.r("as.data.frame(rowData(sce))"))
    col_df = pandas2ri.rpy2py(ro.r("as.data.frame(colData(sce))"))

    labeled = (row_df["population_id"] != "unassigned").to_numpy()
    X_labeled = X_all[labeled]
    y_labeled = row_df["population_id"].to_numpy()[labeled].astype(str)

    type_markers = (col_df["marker_class"].astype(int) == 2).to_numpy()
    X_labeled = X_labeled[:, type_markers]
    markers = [str(m) for m in col_df.index[type_markers].tolist()]

    # Scaler is fit while population 15 is still present. See module docstring.
    X_transformed = np.arcsinh(X_labeled.astype(np.float64) / _COFACTOR)
    X_scaled = RobustScaler(quantile_range=_QUANTILE_RANGE).fit_transform(X_transformed)

    keep = y_labeled != _DROPPED_POPULATION
    return X_scaled[keep], y_labeled[keep], markers


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
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_ROOT / "refs"

    cached = _read_cache(cache_dir)
    from_cache = cached is not None
    if cached is None:
        X, y_arr, markers = _load_from_r(allow_install)
        _write_cache(cache_dir, X, y_arr, markers)
    else:
        X, y_arr, markers = cached

    y = pd.Series(np.asarray(y_arr).astype(str), name="population_id")

    n_full = X.shape[0]
    if subsample is not None:
        size = (
            int(subsample * n_full) if isinstance(subsample, float) else int(subsample)
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=size / n_full, random_state=random_state
        )
        _, idx = next(splitter.split(X, y))
        X = X[idx]
        y = y.iloc[idx].reset_index(drop=True)

    meta = {
        "source": "Levine_32dim",
        "package": "HDCytoData",
        "citation": "Levine et al. 2015, Cell 162(1):184-197",
        "n_cells_full": int(n_full),
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "markers": markers,
        "label_name": "population_id",
        "from_cache": bool(from_cache),
        "preprocessing": [
            "keep cells with an assigned population",
            "keep type markers (marker_class == 2)",
            f"arcsinh(x / {_COFACTOR})",
            f"RobustScaler(quantile_range={_QUANTILE_RANGE}) fit before "
            "population 15 is dropped, deliberately",
            "drop population 15",
        ],
        "subsample": subsample,
        "random_state": random_state,
    }
    return X, y, meta
