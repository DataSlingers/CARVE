"""hECA v2.0 scATAC-seq, pooled across organs.

1,450,511 cells over 25 organs against a 1,657,194-entry cPeak reference,
with cell types harmonized by uHAF. This is the scalability case study, so
organs are pooled to exceed 500,000 cells and the reference label is the
organ, which is unambiguous. The harmonized cell type is available as a
secondary reference but is coarse, averaging roughly four types per organ.

Preprocessing follows the hECA publication rather than the Cusanovich chain:
cPeaks open in at least 0.5% of cells, highly variable peak selection, then
log-normalization and PCA.

Two paths produce the embedding. The pooled path runs the chain on every
annotated cell: two chunked passes over the backed CSR files, then the full
cells-by-kept-peaks matrix in memory -- on the five default organs that is
about 5 billion nonzeros and tens of GB, so it belongs on a large machine,
after which its cache serves every later call. The subsample-first path
(subsample_before_embedding=True) draws the stratified subsample from the
organ files and runs the chain on those rows only, which fits on a laptop;
its embedding is computed on the subsample rather than the population, and
meta says so.
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

_OBS_COLUMNS = ("cell_type", "organ", "donor_id", "study_id")


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


def _open_backed(path: Path):
    """Open an organ file backed, naming the file and the fix if it is unreadable.

    A download that stopped short leaves a file whose HDF5 signature is
    intact but whose objects are not. h5py reports that as an OSError at
    open ("truncated file") for a small file, or as a RuntimeError or
    KeyError while resolving the root group ("free block size is zero?",
    "bad symbol table node signature") for a large one, from a dozen frames
    deep and naming neither the file nor what to do. Both mean the same
    thing here.
    """
    import anndata as ad

    try:
        return ad.read_h5ad(path, backed="r")
    except (OSError, RuntimeError, KeyError) as exc:
        raise OSError(
            f"{path.resolve()} is truncated or corrupt and cannot be read "
            f"({exc}). Re-download {path.name}.zip from Zenodo record "
            f"{ZENODO_RECORD} ({DOWNLOAD_ROOT}), check the zip's md5 against "
            "the record before extracting, and extract it in place."
        ) from exc


def read_rows(path: Path, rows: np.ndarray) -> sparse.csr_matrix:
    """Read the given rows of an h5ad CSR X, and nothing else.

    One slice of data/indices per row, so memory is bounded by the rows
    asked for rather than by the file. The hECA files are uncompressed and
    chunked, so a row read costs a fraction of a millisecond; 25,000 rows
    out of 765,000 read in seconds and pull about 1.4 GB.
    """
    rows = np.asarray(rows, dtype=np.int64)
    with h5py.File(path, "r") as handle:
        _, n_peaks = _shape(handle)
        indptr = handle["X"]["indptr"][:]
        data_set = handle["X"]["data"]
        index_set = handle["X"]["indices"]
        data, indices, local_indptr = [], [], [0]
        for row in rows:
            start, stop = int(indptr[row]), int(indptr[row + 1])
            data.append(data_set[start:stop])
            indices.append(index_set[start:stop])
            local_indptr.append(local_indptr[-1] + (stop - start))
    empty = np.zeros(0, dtype=np.int32)
    return sparse.csr_matrix(
        (
            np.concatenate(data) if data else empty,
            np.concatenate(indices) if indices else empty,
            np.asarray(local_indptr),
        ),
        shape=(rows.size, n_peaks),
    )


def _read_obs(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read only the requested obs columns, never touching X."""
    adata = _open_backed(path)
    frame = adata.obs[list(columns)].copy()
    adata.file.close()
    return frame.reset_index(drop=True)


def _peak_ids(path: Path) -> pd.Index:
    """Read only the var index (cPeak ids), never touching X."""
    adata = _open_backed(path)
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
    subset: tuple[int | float, int] | None = None,
) -> str:
    """A stable key for the cached result of one preprocessing config.

    For the pooled embedding, subsample and random_state are deliberately
    excluded: they are applied fresh on top of the cached, already-
    preprocessed embedding, matching how the uncached path handles them.
    For a subsample-first embedding they are the population, so ``subset``
    carries (subsample, random_state) into the key.
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
    if subset is not None:
        raw += f"|subset:{subset[0]!r}:{int(subset[1])}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cache_path(data_dir: Path, key: str) -> Path:
    return data_dir / f"{_CACHE_PREFIX}{key}.npz"


_CACHE_SCALARS = (
    "n_cells_full",
    "n_cells_annotated",
    "n_peaks_full",
    "n_peaks_open",
    "n_peaks_selected",
)


def _read_cache(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as payload:
        return {
            "X": payload["X"],
            "y": payload["y"],
            **{key: int(payload[key]) for key in _CACHE_SCALARS},
        }


def _write_cache(path: Path, X: np.ndarray, y: np.ndarray, **scalars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=np.asarray(y).astype(str), **scalars)


def _annotated(obs: pd.DataFrame) -> np.ndarray:
    return obs[_ANNOTATION_COLUMN].notna().to_numpy() & (
        obs[_ANNOTATION_COLUMN].to_numpy() != UNCLASSIFIED_LABEL
    )


def _embed(
    counts_matrix: sparse.csr_matrix,
    obs: pd.DataFrame,
    *,
    n_top_peaks: int,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    """The source's chain on already-annotated cells: normalize, log1p, HVG, PCA.

    Returns (X, obs, n_top). The caller has dropped Unclassified cells and
    selected the open peaks; this is the part the pooled and the
    subsample-first paths share.
    """
    import scanpy as sc
    from anndata import AnnData

    # AnnData requires a string obs index; the integer RangeIndex from
    # ignore_index=True would otherwise trigger an implicit-conversion
    # warning on every call.
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    adata = AnnData(X=counts_matrix, obs=obs)
    del counts_matrix

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_top = min(int(n_top_peaks), int(adata.n_vars))
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.tl.pca(
        adata,
        n_comps=min(int(n_components), adata.n_vars - 1, adata.n_obs - 1),
        random_state=random_state,
    )
    return np.asarray(adata.obsm["X_pca"], dtype=np.float64), adata.obs, n_top


def _pooled_embedding(
    paths: Sequence[Path],
    *,
    open_fraction: float,
    n_top_peaks: int,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    """Embed every annotated cell of every organ.

    Two chunked passes select the open peaks and build the reduced matrix
    one row block at a time, but the reduced matrix itself is fully
    resident from there on. On the five default organs that is tens of GB.
    """
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
        [_read_obs(path, _OBS_COLUMNS) for path in paths], ignore_index=True
    )
    annotated = _annotated(obs)
    counts_matrix = counts_matrix[annotated]
    obs = obs[annotated].reset_index(drop=True)

    X, obs, n_top = _embed(
        counts_matrix,
        obs,
        n_top_peaks=n_top_peaks,
        n_components=n_components,
        random_state=random_state,
    )
    return (
        X,
        obs,
        {
            "n_cells_full": int(total_cells),
            "n_cells_annotated": int(annotated.sum()),
            "n_peaks_full": int(total_counts.size),
            "n_peaks_open": int(keep.size),
            "n_peaks_selected": int(n_top),
        },
    )


def _allocate(size: int, counts: Sequence[int]) -> list[int]:
    """Split ``size`` across strata in proportion to ``counts``, summing exactly.

    Largest-remainder rounding, so the total is ``size`` and no stratum is
    off by more than one from its proportional share.
    """
    total = sum(counts)
    shares = [size * count / total for count in counts]
    allocation = [int(share) for share in shares]
    for index in sorted(
        range(len(counts)), key=lambda i: shares[i] - allocation[i], reverse=True
    )[: size - sum(allocation)]:
        allocation[index] += 1
    return allocation


def _subsample_embedding(
    paths: Sequence[Path],
    *,
    subsample: int | float,
    open_fraction: float,
    n_top_peaks: int,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    """Draw the rows first, then embed only those.

    The subsample is stratified by organ over annotated cells, in proportion
    to each organ's annotated count, and drawn from the files row by row.
    Open-peak selection, HVG and PCA then run on the drawn rows, so the
    embedding describes the subsample rather than the population; a
    publication run must use the pooled path.
    """
    obs_per_organ = [_read_obs(path, _OBS_COLUMNS) for path in paths]
    annotated_per_organ = [_annotated(obs) for obs in obs_per_organ]
    n_full = sum(len(obs) for obs in obs_per_organ)
    n_annotated = sum(int(mask.sum()) for mask in annotated_per_organ)

    size = (
        int(subsample * n_annotated) if isinstance(subsample, float) else int(subsample)
    )
    if not 0 < size <= n_annotated:
        raise ValueError(
            f"subsample={subsample!r} resolves to {size} cells, but the organs "
            f"hold {n_annotated} annotated cells."
        )

    rng = np.random.default_rng(random_state)
    rows_per_organ = []
    for mask, n_rows in zip(
        annotated_per_organ,
        _allocate(size, [int(mask.sum()) for mask in annotated_per_organ]),
    ):
        candidates = np.flatnonzero(mask)
        rows_per_organ.append(
            np.sort(rng.choice(candidates, size=n_rows, replace=False))
        )

    counts_matrix = sparse.vstack(
        [read_rows(path, rows) for path, rows in zip(paths, rows_per_organ)],
        format="csr",
    )
    obs = pd.concat(
        [
            obs.iloc[rows].reset_index(drop=True)
            for obs, rows in zip(obs_per_organ, rows_per_organ)
        ],
        ignore_index=True,
    )

    open_counts = np.asarray((counts_matrix > 0).sum(axis=0)).ravel()
    keep = np.flatnonzero(open_counts >= open_fraction * size)
    if keep.size == 0:
        raise ValueError(
            f"open_fraction={open_fraction} removed every cPeak. Lower it."
        )
    n_peaks_full = int(counts_matrix.shape[1])
    counts_matrix = counts_matrix[:, keep]

    X, obs, n_top = _embed(
        counts_matrix,
        obs,
        n_top_peaks=n_top_peaks,
        n_components=n_components,
        random_state=random_state,
    )
    return (
        X,
        obs,
        {
            "n_cells_full": int(n_full),
            "n_cells_annotated": int(n_annotated),
            "n_peaks_full": n_peaks_full,
            "n_peaks_open": int(keep.size),
            "n_peaks_selected": int(n_top),
        },
    )


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
    subsample_before_embedding: bool = False,
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess pooled hECA v2.0 ATAC organs.

    Parameters
    ----------
    root : Path or None
        Directory holding the dataset folders. Defaults to data/.
    organs : sequence of str
        Organ files to pool, ATAC-{organ}.h5ad each.
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
        Cache the preprocessed embedding alongside the organ h5ad files,
        keyed by organs and preprocessing parameters (and, for a
        subsample-first embedding, by subsample and random_state). On a
        hit, the chunked passes and the scanpy preprocessing are skipped
        entirely. True by default: at real hECA scale the passes are the
        expensive part, and the source h5ad files may already be deleted by
        the time a second call is made.
    subsample_before_embedding : bool
        When True and ``subsample`` is given, draw the subsample from the
        organ files first and run the preprocessing chain on those rows
        only, instead of embedding every annotated cell and subsampling the
        result. The pooled path holds the whole cells-by-kept-peaks matrix
        in memory, tens of GB on the five default organs, so this is what
        makes a development-scale load possible on a laptop. The embedding
        is then computed on the subsample, not the population, and
        ``meta["embedding_population"]`` records which. A cached pooled
        embedding, when present, is used regardless: it is the better
        embedding and it is free.

    Returns
    -------
    (X, y, meta)
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    data_dir = resolve_data_dir("hECA", root=root)
    config = (organs, open_fraction, n_top_peaks, n_components, label_column)

    pooled_path = _cache_path(data_dir, _cache_key(*config))
    cached = _read_cache(pooled_path) if cache else None
    population = "pooled"
    cache_path = pooled_path

    if cached is None and subsample is not None and subsample_before_embedding:
        population = "subsample"
        cache_path = _cache_path(
            data_dir, _cache_key(*config, subset=(subsample, random_state))
        )
        cached = _read_cache(cache_path) if cache else None

    if cached is not None:
        X_full = np.asarray(cached["X"], dtype=np.float64)
        y_full = pd.Series(cached["y"], name=label_column).astype(str)
        scalars = {key: cached[key] for key in _CACHE_SCALARS}
    else:
        paths = [_organ_path(data_dir, organ) for organ in organs]
        _check_peaks_aligned(list(organs), paths)
        if population == "subsample":
            X_full, obs, scalars = _subsample_embedding(
                paths,
                subsample=subsample,
                open_fraction=open_fraction,
                n_top_peaks=n_top_peaks,
                n_components=n_components,
                random_state=random_state,
            )
        else:
            X_full, obs, scalars = _pooled_embedding(
                paths,
                open_fraction=open_fraction,
                n_top_peaks=n_top_peaks,
                n_components=n_components,
                random_state=random_state,
            )
        if label_column not in obs.columns:
            raise ValueError(
                f"obs has no column {label_column!r}. Available: {sorted(obs.columns)}."
            )
        y_full = pd.Series(obs[label_column].to_numpy(), name=label_column).astype(str)
        if cache:
            _write_cache(cache_path, X_full, y_full.to_numpy(), **scalars)

    X = X_full
    y = y_full
    n_annotated = int(scalars["n_cells_annotated"])
    # A subsample-first embedding already is the subsample; the pooled one
    # is subsampled here, on top of the cached or freshly computed result.
    if subsample is not None and population == "pooled":
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

    n_top = int(scalars["n_peaks_selected"])
    if population == "pooled":
        feature_scope = "pooled across organs"
        open_line = (
            f"keep cPeaks open in at least {open_fraction:.1%} of pooled cells "
            "(0.5%, the source value)"
        )
    else:
        feature_scope = "pooled across organs, on the row subsample"
        open_line = (
            f"keep cPeaks open in at least {open_fraction:.1%} of the "
            f"{X.shape[0]} subsampled cells (0.5%, the source value); the "
            "subsample was drawn from the organ files before any "
            "preprocessing, so peak selection, HVG and PCA describe the "
            "subsample, not the pooled population"
        )

    meta = {
        "source": "hECA v2.0 ATAC",
        "citation": "Chen et al. 2025, Scientific Data",
        "download": DOWNLOAD_ROOT,
        "organs": list(organs),
        "n_cells_full": int(scalars["n_cells_full"]),
        "n_cells_annotated": n_annotated,
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_peaks_full": int(scalars["n_peaks_full"]),
        "n_peaks_open": int(scalars["n_peaks_open"]),
        "n_peaks_selected": n_top,
        "feature_selection_scope": feature_scope,
        "embedding_population": population,
        "label_name": label_column,
        "preprocessing": [
            f"pool organs {list(organs)}",
            f"drop cells labeled {UNCLASSIFIED_LABEL!r} in {_ANNOTATION_COLUMN} "
            "(checked regardless of label_column)",
            open_line,
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
