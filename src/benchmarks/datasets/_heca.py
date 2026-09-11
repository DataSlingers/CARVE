"""hECA v2.0 scATAC-seq, pooled across organs.

1,450,511 cells over 25 organs against a 1,657,194-entry cPeak reference,
with cell types harmonized by uHAF. This is the scalability case study, so
organs are pooled to exceed 500,000 cells and the reference label is the
organ, which is unambiguous. The harmonized cell type is available as a
secondary reference but is coarse, averaging roughly four types per organ.

Preprocessing follows the hECA publication rather than the Cusanovich chain:
cPeaks open in at least 0.5% of cells, highly variable peak selection, then
log-normalization and PCA. The peak matrix is never fully resident; two
chunked passes over the backed CSR do the work.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from ._klein import resolve_data_dir

ZENODO_RECORD = "15627886"
DOWNLOAD_ROOT = f"https://zenodo.org/records/{ZENODO_RECORD}"

#: Five organs clear 500,000 cells while keeping the organ label at five
#: levels. Lung and Brain supply most of the cells; the rest give the
#: reference label enough classes to be usable.
DEFAULT_ORGANS: tuple[str, ...] = ("Lung", "Brain", "Kidney", "Heart", "Thymus")

#: uHAF value marking cells left unannotated, dropped like Levine's
#: uncharacterized population 15.
UNCLASSIFIED_LABEL = "Unclassified"

#: Column checked for UNCLASSIFIED_LABEL to decide which cells to drop. This
#: is fixed regardless of label_column, mirroring the Cusanovich loader:
#: "organ" is always known and never "Unclassified" -- only the
#: uHAF-harmonized cell_type can be. Checking label_column instead would
#: silently keep every Unclassified cell whenever label_column="organ",
#: which is the default.
_ANNOTATION_COLUMN = "cell_type"

#: Rows read per chunk in both passes over a backed h5ad. Not exposed on
#: load_heca (its own parameter list is fixed by the interface); the two
#: standalone chunked functions still take it, since that is where the
#: memory/speed trade-off actually lives.
DEFAULT_BLOCK = 20_000

_CACHE_PREFIX = ".heca_cache_"


def _organ_path(data_dir: Path, organ: str) -> Path:
    path = data_dir / f"ATAC-{organ}.h5ad"
    if not path.is_file():
        # Show the resolved absolute path so the message names one exact
        # location rather than a relative spelling.
        raise FileNotFoundError(
            f"Missing {path.resolve()}. Download ATAC-{organ}.h5ad.zip from "
            f"Zenodo record {ZENODO_RECORD} ({DOWNLOAD_ROOT}) and extract "
            f"it into {data_dir.resolve()}/. All organs must be present "
            "together for the pooled pass, so delete each zip as soon as it "
            "is extracted; once this loader has written its .npz cache the "
            "h5ad files can be deleted too."
        )
    return path


def _csr_block(handle: h5py.File, lo: int, hi: int, n_peaks: int) -> sparse.csr_matrix:
    """Read rows [lo, hi) of an h5ad CSR X group without loading the rest."""
    indptr = handle["X"]["indptr"]
    start = int(indptr[lo])
    stop = int(indptr[hi])
    data = handle["X"]["data"][start:stop]
    indices = handle["X"]["indices"][start:stop]
    # The block's own indptr must be rebased to 0 at its first row: indptr
    # values read straight off the file are offsets into the *whole* data
    # array, not into the data/indices slice just pulled out above.
    local_indptr = indptr[lo : hi + 1][:] - start
    return sparse.csr_matrix((data, indices, local_indptr), shape=(hi - lo, n_peaks))


def _shape(handle: h5py.File) -> tuple[int, int]:
    shape = handle["X"].attrs["shape"]
    return int(shape[0]), int(shape[1])


def peak_open_counts(
    path: Path, *, block: int = DEFAULT_BLOCK
) -> tuple[np.ndarray, int]:
    """Number of cells each peak is open in, accumulated over row blocks.

    Returns (counts of length n_peaks, n_cells).
    """
    with h5py.File(path, "r") as handle:
        n_cells, n_peaks = _shape(handle)
        counts = np.zeros(n_peaks, dtype=np.int64)
        for lo in range(0, n_cells, block):
            hi = min(lo + block, n_cells)
            chunk = _csr_block(handle, lo, hi, n_peaks)
            chunk.data = np.ones_like(chunk.data)
            counts += np.asarray(chunk.sum(axis=0)).ravel().astype(np.int64)
    return counts, n_cells


def reduce_to_peaks(
    path: Path, peaks: np.ndarray, *, block: int = DEFAULT_BLOCK
) -> sparse.csr_matrix:
    """Build the cells-by-selected-peaks matrix, one row block at a time."""
    peaks = np.asarray(peaks)
    with h5py.File(path, "r") as handle:
        n_cells, n_peaks = _shape(handle)
        blocks = []
        for lo in range(0, n_cells, block):
            hi = min(lo + block, n_cells)
            blocks.append(_csr_block(handle, lo, hi, n_peaks)[:, peaks])
    return sparse.vstack(blocks, format="csr")


def _read_obs(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read only the requested obs columns, never touching X."""
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    frame = adata.obs[list(columns)].copy()
    adata.file.close()
    return frame.reset_index(drop=True)


def _peak_ids(path: Path) -> pd.Index:
    """Read only the var index (cPeak ids), never touching X."""
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    names = adata.var_names.copy()
    adata.file.close()
    return names


def _check_peaks_aligned(organs: Sequence[str], paths: Sequence[Path]) -> None:
    """Fail loudly if the pooled organs do not share one cPeak reference.

    total_counts + counts below only requires matching length, which raises
    on a mismatched peak count but says nothing when two organs carry the
    same number of cPeaks in a different order. In that case the pooled
    feature selection and reduce_to_peaks would silently mix different
    peaks across organs, and every downstream number would be wrong with no
    symptom. All five real hECA files come from one cPeak reference, so this
    is expected to pass; it exists for when that assumption stops holding.
    """
    reference_organ, reference_peaks = organs[0], _peak_ids(paths[0])
    for organ, path in zip(organs[1:], paths[1:]):
        peaks = _peak_ids(path)
        if not peaks.equals(reference_peaks):
            raise ValueError(
                f"ATAC-{organ}.h5ad's cPeaks do not match "
                f"ATAC-{reference_organ}.h5ad's -- pooling would silently "
                "mix different peaks across organs. Expected the same "
                "cPeak reference (same peaks, same order) in every organ "
                "file being pooled."
            )


def _cache_key(
    organs: Sequence[str],
    open_fraction: float,
    n_top_peaks: int,
    n_components: int,
    label_column: str,
) -> str:
    """A stable key for the pre-subsample result of one preprocessing config.

    subsample and random_state are deliberately excluded: they are applied
    fresh on top of the cached, already-preprocessed embedding, matching how
    the uncached path handles them.
    """
    raw = "|".join(
        [
            ",".join(organs),
            repr(float(open_fraction)),
            str(int(n_top_peaks)),
            str(int(n_components)),
            label_column,
        ]
    )
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cache_path(data_dir: Path, key: str) -> Path:
    return data_dir / f"{_CACHE_PREFIX}{key}.npz"


def _read_cache(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as payload:
        return {
            "X": payload["X"],
            "y": payload["y"],
            "n_cells_full": int(payload["n_cells_full"]),
            "n_peaks_full": int(payload["n_peaks_full"]),
            "n_peaks_open": int(payload["n_peaks_open"]),
            "n_peaks_selected": int(payload["n_peaks_selected"]),
        }


def _write_cache(path: Path, X: np.ndarray, y: np.ndarray, **scalars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=np.asarray(y).astype(str), **scalars)


def load_heca(
    *,
    root: Path | None = None,
    organs: Sequence[str] = DEFAULT_ORGANS,
    subsample: int | float | None = None,
    random_state: int = 42,
    open_fraction: float = 0.005,
    n_top_peaks: int = 50_000,
    n_components: int = 50,
    label_column: str = "organ",
    cache: bool = True,
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess pooled hECA v2.0 ATAC organs.

    Parameters
    ----------
    root : Path or None
        Directory holding the dataset folders. Defaults to data/.
    organs : sequence of str
        Organ names to pool. Files are read as data/hECA/ATAC-<organ>.h5ad.
    subsample : int, float, or None
        Stratified subsample size, as a count or a fraction. None keeps all.
    random_state : int
        Seed for the subsample and the PCA.
    open_fraction : float
        Keep cPeaks open in at least this fraction of pooled cells. 0.005 is
        the source publication's value.
    n_top_peaks : int
        Highly variable cPeaks retained. The source uses 500,000; see the
        recorded deviation.
    n_components : int
        PCA components retained.
    label_column : str
        Reference label column. "organ" is unambiguous; "cell_type" is the
        uHAF-harmonized annotation and is coarse.
    cache : bool
        Cache the preprocessed embedding for every annotated cell, before
        subsampling, alongside the organ h5ad files, keyed by organs and
        preprocessing parameters. On a hit, the two chunked passes and the
        scanpy preprocessing are skipped entirely; only the stratified
        subsample is redone for the current call. True by default: at real
        hECA scale the chunked passes are the expensive part, and the source
        h5ad files may already be deleted by the time a second call is made.

    Returns
    -------
    (X, y, meta)
    """
    import scanpy as sc
    from anndata import AnnData
    from sklearn.model_selection import StratifiedShuffleSplit

    data_dir = resolve_data_dir("hECA", root=root)

    cache_path = _cache_path(
        data_dir,
        _cache_key(organs, open_fraction, n_top_peaks, n_components, label_column),
    )
    cached = _read_cache(cache_path) if cache else None

    if cached is not None:
        X_full = np.asarray(cached["X"], dtype=np.float64)
        y_full = pd.Series(cached["y"], name=label_column).astype(str)
        n_full = cached["n_cells_full"]
        n_peaks_full = cached["n_peaks_full"]
        n_peaks_open = cached["n_peaks_open"]
        n_top = cached["n_peaks_selected"]
    else:
        paths = [_organ_path(data_dir, organ) for organ in organs]
        _check_peaks_aligned(list(organs), paths)

        # Pass 1: per-peak open counts pooled across every organ, so the
        # feature set is shared. Selecting features per organ would make the
        # pooled embedding meaningless.
        total_counts: np.ndarray | None = None
        total_cells = 0
        for path in paths:
            counts, n_cells = peak_open_counts(path, block=DEFAULT_BLOCK)
            total_counts = counts if total_counts is None else total_counts + counts
            total_cells += n_cells

        keep = np.flatnonzero(total_counts >= open_fraction * total_cells)
        if keep.size == 0:
            raise ValueError(
                f"open_fraction={open_fraction} removed every cPeak. Lower it."
            )

        # Pass 2: build the reduced matrix, one organ and one row block at a
        # time.
        matrices = [reduce_to_peaks(path, keep, block=DEFAULT_BLOCK) for path in paths]
        counts_matrix = sparse.vstack(matrices, format="csr")
        del matrices

        obs = pd.concat(
            [
                _read_obs(path, ["cell_type", "organ", "donor_id", "study_id"])
                for path in paths
            ],
            ignore_index=True,
        )
        # AnnData requires a string obs index; the integer RangeIndex from
        # ignore_index=True would otherwise trigger an implicit-conversion
        # warning on every call.
        obs.index = obs.index.astype(str)

        if label_column not in obs.columns:
            raise ValueError(
                f"obs has no column {label_column!r}. Available: {sorted(obs.columns)}."
            )

        adata = AnnData(X=counts_matrix, obs=obs)
        n_full = int(adata.n_obs)

        annotated = adata.obs[_ANNOTATION_COLUMN].notna().to_numpy() & (
            adata.obs[_ANNOTATION_COLUMN].to_numpy() != UNCLASSIFIED_LABEL
        )
        adata = adata[annotated].copy()

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_top = min(int(n_top_peaks), int(adata.n_vars))
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"]].copy()
        sc.tl.pca(
            adata,
            n_comps=min(int(n_components), adata.n_vars - 1),
            random_state=random_state,
        )

        X_full = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        y_full = pd.Series(
            adata.obs[label_column].to_numpy(), name=label_column
        ).astype(str)
        n_peaks_full = int(total_counts.size)
        n_peaks_open = int(keep.size)

        if cache:
            _write_cache(
                cache_path,
                X_full,
                y_full.to_numpy(),
                n_cells_full=n_full,
                n_peaks_full=n_peaks_full,
                n_peaks_open=n_peaks_open,
                n_peaks_selected=n_top,
            )

    X = X_full
    y = y_full
    n_annotated = int(X.shape[0])
    if subsample is not None:
        size = (
            int(subsample * n_annotated)
            if isinstance(subsample, float)
            else int(subsample)
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=size / n_annotated, random_state=random_state
        )
        _, idx = next(splitter.split(X, y))
        X = X[idx]
        y = y.iloc[idx].reset_index(drop=True)

    meta = {
        "source": "hECA v2.0 ATAC",
        "citation": "Chen et al. 2025, Scientific Data",
        "download": DOWNLOAD_ROOT,
        "organs": list(organs),
        "n_cells_full": n_full,
        "n_cells_annotated": n_annotated,
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_peaks_full": n_peaks_full,
        "n_peaks_open": n_peaks_open,
        "n_peaks_selected": int(n_top),
        "feature_selection_scope": "pooled across organs",
        "label_name": label_column,
        "preprocessing": [
            f"pool organs {list(organs)}",
            f"drop cells labeled {UNCLASSIFIED_LABEL!r} in {_ANNOTATION_COLUMN} "
            "(checked regardless of label_column)",
            f"keep cPeaks open in at least {open_fraction:.1%} of pooled cells "
            "(0.5%, the source value)",
            "normalize_total(target_sum=1e4) then log1p",
            f"highly_variable_genes(n_top_genes={n_top}, flavor='seurat')",
            f"PCA to {X.shape[1]} components",
        ],
        "deviations": [
            "deviation from source: the publication selects 500,000 highly "
            f"variable cPeaks; {n_top} are selected here, for memory",
        ],
        "batch_note": (
            "cells pool multiple study_id values, so a pooled clustering may "
            "partly recover study of origin rather than biology"
        ),
        "cached": cached is not None,
        "subsample": subsample,
        "random_state": random_state,
    }
    return X, y, meta
