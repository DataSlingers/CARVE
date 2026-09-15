"""Cusanovich et al. mouse sci-ATAC-seq atlas.

436,206 peaks by 81,173 cells of binarized chromatin accessibility across 13
adult mouse tissues. The reference label used here is the source
publication's own clustering: 30 clusters from graph community detection on a
two-dimensional t-SNE of their LSI. It is a published partition, not ground
truth, so agreement with it is reported as agreement and never as accuracy.
The 30 clusters assign every cell, so every cell is kept by default. Tissue of
dissection and the 40 marker-based cell labels ride along in
meta["source_labels"].

Preprocessing follows the source publication's own dim_reduction.R: a 3%
site-frequency threshold, TF-IDF, and a 50-component SVD whose cell
coordinates are the right singular vectors scaled by the singular values.
All 50 components are kept, because the source keeps them; dropping LSI
component 1 is a later Signac and ArchR convention.
"""

import gzip
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ._klein import DATA_ROOT, resolve_data_dir

DOWNLOAD_ROOT = (
    "http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release/"
)

_MATRIX = "matrices/atac_matrix.binary.qc_filtered.mtx.gz"
_CELLS = "matrices/atac_matrix.binary.qc_filtered.cells.txt"
_METADATA = "metadata/cell_metadata.txt"

#: Columns of cell_metadata.txt holding the source publication's own t-SNE
#: coordinates (their Figure 1). Carried through as meta["source_tsne"] so a
#: notebook can show the data the way the source does, at zero compute.
_SOURCE_TSNE_COLUMNS = ("tsne_1", "tsne_2")

#: Columns of cell_metadata.txt holding the source's own labels, carried
#: through as meta["source_labels"] for coloring and side checks.
_SOURCE_LABEL_COLUMNS = ("tissue", "cluster", "subset_cluster", "cell_label")

#: Label value marking cells the source left unannotated. Dropped only under
#: drop_unknown=True.
UNKNOWN_LABEL = "Unknown"

#: Column checked for UNKNOWN_LABEL to decide which cells to drop. This is
#: fixed to the source's own marker-based annotation regardless of
#: label_column, because "tissue" (the dissection of origin) is always
#: known and never "Unknown" -- only the marker-based cell_label can be.
_ANNOTATION_COLUMN = "cell_label"


def _require(path: Path) -> Path:
    if not path.is_file():
        # Show the resolved absolute path so the message names one exact
        # location rather than a relative spelling.
        raise FileNotFoundError(
            f"Missing {path.resolve()}. Download the Cusanovich atlas from "
            f"{DOWNLOAD_ROOT} into {path.resolve().parent.parent}/, keeping "
            "the matrices/ and metadata/ subdirectories."
        )
    return path


def _resolve_cusanovich_dir(root: Path | None) -> Path:
    """Find the Cusanovich directory, naming the download when it is absent.

    resolve_data_dir's own error only reports what it found under root, with
    no knowledge of where this particular dataset comes from. Wrap it so a
    missing data/Cusanovich/ still points at the download location.
    """
    try:
        return resolve_data_dir("Cusanovich", root=root)
    except FileNotFoundError as exc:
        searched = (root if root is not None else DATA_ROOT).resolve()
        raise FileNotFoundError(
            f"No Cusanovich atlas found under {searched}. Download the atlas from "
            f"{DOWNLOAD_ROOT} into {searched / 'Cusanovich'}/, keeping the "
            "matrices/ and metadata/ subdirectories."
        ) from exc


def _read_matrix(path: Path) -> sparse.csr_matrix:
    """Read the peaks-by-cells Matrix Market file."""
    from scipy.io import mmread

    # scipy 1.18 warns when spmatrix is left to its default, which changes
    # in 1.20; the csr_matrix wrap returns the same matrix either way.
    with gzip.open(path, "rb") as handle:
        return sparse.csr_matrix(mmread(handle, spmatrix=False))


def _tfidf(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """TF-IDF exactly as the source computes it.

    Term frequency divides each cell column by its column sum; inverse
    document frequency is log(1 + n_cells / cells_per_peak). The input is
    peaks by cells and binary, so a peak's row sum is the number of cells it
    is accessible in.
    """
    counts = counts.tocsr().astype(np.float64)
    n_cells = counts.shape[1]

    cell_sums = np.asarray(counts.sum(axis=0)).ravel()
    cell_sums[cell_sums == 0] = 1.0
    peak_sums = np.asarray(counts.sum(axis=1)).ravel()
    peak_sums[peak_sums == 0] = 1.0

    term_frequency = counts @ sparse.diags(1.0 / cell_sums)
    inverse_document = np.log1p(n_cells / peak_sums)
    return sparse.diags(inverse_document) @ term_frequency


def load_cusanovich(
    *,
    root: Path | None = None,
    subsample: int | float | None = None,
    random_state: int = 42,
    site_frequency_threshold: float = 0.03,
    n_components: int = 50,
    label_column: str = "cluster",
    drop_unknown: bool = False,
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess the Cusanovich mouse sci-ATAC atlas.

    Parameters
    ----------
    root : Path or None
        Directory holding the dataset folders. Defaults to data/.
    subsample : int, float, or None
        Subsample size, as a count or a fraction, stratified by the reference
        label. None keeps all.
    random_state : int
        Seed for the stratified subsample and the SVD.
    site_frequency_threshold : float
        Keep peaks accessible in at least this fraction of cells. 0.03 is the
        source publication's own value.
    n_components : int
        SVD components retained. All of them are kept; none is dropped.
    label_column : str
        Metadata column used as the reference label, returned as strings.
        "cluster" is the source's 30-cluster partition of every cell;
        "tissue" is the dissection of origin; "cell_label" is the source's
        marker-based annotation.
    drop_unknown : bool
        Drop cells whose cell_label is UNKNOWN_LABEL, 10,029 of the release's
        81,173. Off by default: the source's clusters assign every cell, so
        dropping these would compare against a partition of a different set.

    Returns
    -------
    (X, y, meta)
        Row i of X, y, meta["source_tsne"] and meta["source_labels"] is the
        same cell. meta["source_tsne"] is an (n_cells, 2) float array of the
        source's own t-SNE coordinates; meta["source_labels"] is a DataFrame
        of the source's tissue, cluster, subset_cluster and cell_label.
        meta["n_cells_annotated"] counts the atlas cells whose cell_label is
        not UNKNOWN_LABEL, whether or not they were dropped.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import StratifiedShuffleSplit

    data_dir = _resolve_cusanovich_dir(root)

    matrix = _read_matrix(_require(data_dir / _MATRIX))
    cells = _require(data_dir / _CELLS).read_text().split()
    metadata = pd.read_csv(_require(data_dir / _METADATA), sep="\t")

    if label_column not in metadata.columns:
        raise ValueError(
            f"cell_metadata.txt has no column {label_column!r}. "
            f"Available: {sorted(metadata.columns)}."
        )

    # The matrix columns are in cell_metadata.txt row order. The release's
    # .cells.txt holds the same barcodes in a different order and is not the
    # column order: reindexing the metadata by it kept tissue blocks roughly
    # intact but gave every cell another cell's label, cluster and t-SNE
    # position within its tissue. Verified on the released atlas (2026-09):
    # in the loader's own LSI, 15-nearest-neighbor purity by the source's
    # cell_label is 0.83 under metadata order and 0.29 under .cells.txt
    # order, and within-tissue cell-type classification is 0.89-1.00 versus
    # the permuted-label baseline. The cells file is still checked, as the
    # one independent record of which barcodes the matrix holds.
    n_cells_full = int(matrix.shape[1])
    if (
        len(cells) != n_cells_full
        or len(metadata) != n_cells_full
        or set(cells) != set(metadata["cell"])
    ):
        raise ValueError(
            "atac_matrix.binary.qc_filtered.cells.txt and cell_metadata.txt do "
            f"not describe the same {n_cells_full} matrix columns "
            f"({len(cells)} cell ids, {len(metadata)} metadata rows, "
            f"{len(set(cells) & set(metadata['cell']))} in common)."
        )
    metadata = metadata.set_index("cell")

    # UNKNOWN_LABEL covers 10,029 of 81,173 cells in the release, about 12
    # percent. The source's 30 clusters assign every one of them, so they
    # stay unless drop_unknown asks otherwise; n_cells_annotated reports the
    # annotated count either way.
    annotated = metadata[_ANNOTATION_COLUMN].notna().to_numpy() & (
        metadata[_ANNOTATION_COLUMN].to_numpy() != UNKNOWN_LABEL
    )
    n_annotated = int(annotated.sum())
    keep_cells = annotated if drop_unknown else np.ones(n_cells_full, dtype=bool)

    peak_cell_counts = np.asarray((matrix > 0).sum(axis=1)).ravel()
    keep_peaks = peak_cell_counts >= site_frequency_threshold * n_cells_full

    n_peaks_kept = int(keep_peaks.sum())
    n_kept_cells = int(keep_cells.sum())

    preprocessing = [
        (
            f"drop cells labeled {UNKNOWN_LABEL!r} in {_ANNOTATION_COLUMN}"
            if drop_unknown
            else f"keep every cell, including the {n_cells_full - n_annotated} "
            f"labeled {UNKNOWN_LABEL!r} in {_ANNOTATION_COLUMN}"
        ),
        f"keep peaks accessible in at least {site_frequency_threshold:.0%} "
        "of cells (site_frequency_threshold, source default 3%)",
    ]

    if n_peaks_kept == 0:
        # A threshold this strict leaves no signal to embed. TruncatedSVD
        # requires at least one feature, so report a zero embedding rather
        # than raising a confusing sklearn error deep in the SVD call. Warn
        # and record the skip in meta rather than silently claiming the
        # TF-IDF/SVD chain ran, which it did not.
        warnings.warn(
            f"site_frequency_threshold={site_frequency_threshold} removed "
            "every peak in the Cusanovich atlas; returning a zero embedding "
            "instead of running TF-IDF/SVD.",
            stacklevel=2,
        )
        embedding = np.zeros((n_kept_cells, n_components), dtype=np.float64)
        preprocessing.append(
            "TF-IDF and TruncatedSVD were skipped: "
            f"site_frequency_threshold={site_frequency_threshold} left no "
            "peaks, so X is a zero embedding, not a real projection"
        )
    else:
        counts = matrix[keep_peaks][:, keep_cells]
        embedding = TruncatedSVD(
            n_components=n_components, random_state=random_state
        ).fit_transform(_tfidf(counts).T.tocsr())
        preprocessing.extend(
            [
                "TF-IDF: per-cell term frequency, "
                "IDF = log(1 + n_cells / cells_per_peak)",
                f"TruncatedSVD(n_components={n_components}); all components "
                "kept, matching the source, which does not drop component 1",
            ]
        )

    X = np.asarray(embedding, dtype=np.float64)
    kept = metadata.loc[keep_cells]
    y = pd.Series(kept[label_column].to_numpy(), name=label_column).astype(str)
    source_tsne = np.asarray(
        kept[list(_SOURCE_TSNE_COLUMNS)].to_numpy(), dtype=np.float64
    )
    source_labels = kept[list(_SOURCE_LABEL_COLUMNS)].reset_index(drop=True)

    n_kept = int(X.shape[0])
    if subsample is not None:
        size = (
            int(subsample * n_kept) if isinstance(subsample, float) else int(subsample)
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=size / n_kept, random_state=random_state
        )
        _, idx = next(splitter.split(X, y))
        X = X[idx]
        y = y.iloc[idx].reset_index(drop=True)
        source_tsne = source_tsne[idx]
        source_labels = source_labels.iloc[idx].reset_index(drop=True)

    meta = {
        "source": "Cusanovich",
        "citation": "Cusanovich et al. 2018, Cell 174(5):1309-1324",
        "download": DOWNLOAD_ROOT,
        "n_cells_full": n_cells_full,
        "n_cells_annotated": n_annotated,
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_peaks_full": int(matrix.shape[0]),
        "n_peaks_kept": n_peaks_kept,
        "label_name": label_column,
        "drop_unknown": drop_unknown,
        "drops_first_component": False,
        "preprocessing": preprocessing,
        "subsample": subsample,
        "random_state": random_state,
        "source_tsne": source_tsne,
        "source_labels": source_labels,
    }
    return X, y, meta
