"""Klein et al. mouse embryonic stem cell scRNA-seq (GSE65525).

Four collection days (d0, d2, d4, d7) are concatenated and the day is used as
the reported label. Preprocessing is scanpy's standard chain: total-count
normalization to 1e4, log1p, then the top 2000 highly variable genes selected
per batch with the seurat flavor.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("data")

_FILES = {
    "d0": "GSM1599494_ES_d0_main.csv.bz2",
    "d2": "GSM1599497_ES_d2_LIFminus.csv.bz2",
    "d4": "GSM1599498_ES_d4_LIFminus.csv.bz2",
    "d7": "GSM1599499_ES_d7_LIFminus.csv.bz2",
}


def resolve_data_dir(name: str, *, root: Path | None = None) -> Path:
    """Find a dataset directory, ignoring case.

    The notebooks referred to data/klein while the directory on disk is
    data/Klein. That works on a case-insensitive macOS filesystem and fails
    on Linux, which is what CI runs.
    """
    root = Path(root) if root is not None else DATA_ROOT

    if root.is_dir():
        entries = [c for c in sorted(root.iterdir()) if c.is_dir()]
        for candidate in entries:
            if candidate.name == name:
                return candidate
        lowered = name.lower()
        for candidate in entries:
            if candidate.name.lower() == lowered:
                return candidate
        seen = sorted(c.name for c in entries)
    else:
        seen = []

    raise FileNotFoundError(
        f"No dataset directory matching {name!r} under {root}. Found: {seen}."
    )


def _read_block(path: Path) -> pd.DataFrame:
    """Read one genes-by-cells block, coercing stray strings to zero."""
    df = pd.read_csv(path, header=None, index_col=0, compression="bz2")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df[~df.index.duplicated(keep="first")]


def load_klein(
    *,
    root: Path | None = None,
    subsample: int | float | None = None,
    random_state: int = 42,
    n_top_genes: int = 2000,
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess the Klein dataset.

    Parameters
    ----------
    root : Path or None
        Directory holding the dataset folders. Defaults to data/.
    subsample : int, float, or None
        Stratified subsample size, as a count or a fraction. None keeps all.
    random_state : int
        Seed for the stratified subsample.
    n_top_genes : int
        Highly variable genes to keep.

    Returns
    -------
    (X, y, meta)
    """
    import anndata as ad
    import scanpy as sc
    from sklearn.model_selection import StratifiedShuffleSplit

    data_dir = resolve_data_dir("Klein", root=root)

    blocks = {key: _read_block(data_dir / filename) for key, filename in _FILES.items()}

    genes = blocks["d0"].index
    for key in ("d2", "d4", "d7"):
        genes = genes.union(blocks[key].index)
    blocks = {key: block.reindex(genes).fillna(0.0) for key, block in blocks.items()}

    for key, block in blocks.items():
        block.columns = [f"{key}_{i + 1:04d}" for i in range(block.shape[1])]

    expression = pd.concat([blocks[k] for k in ("d0", "d2", "d4", "d7")], axis=1)
    labels = [
        key for key in ("d0", "d2", "d4", "d7") for _ in range(blocks[key].shape[1])
    ]
    cells = expression.columns
    condition = pd.Series(labels, index=cells, name="condition").astype("category")

    adata = ad.AnnData(
        X=expression.T.values.astype(np.float32),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=expression.index.astype(str)),
    )
    adata.obs["condition"] = condition
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes, batch_key="condition", flavor="seurat"
        )
    adata = adata[:, adata.var["highly_variable"]].copy()

    X = np.asarray(adata.X)
    y = pd.Series(adata.obs["condition"].to_numpy(), name="condition")

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
        "source": "Klein",
        "accession": "GSE65525",
        "citation": "Klein et al. 2015, Cell 161(5):1187-1201",
        "n_cells_full": int(n_full),
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_top_genes": int(n_top_genes),
        "label_name": "condition",
        "preprocessing": [
            "concatenate d0/d2/d4/d7 on the gene union, missing filled with 0",
            "normalize_total(target_sum=1e4)",
            "log1p",
            f"highly_variable_genes(n_top_genes={n_top_genes}, "
            "batch_key='condition', flavor='seurat')",
        ],
        "subsample": subsample,
        "random_state": random_state,
    }
    return X, y, meta
