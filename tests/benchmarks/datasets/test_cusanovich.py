"""Cusanovich loader tests.

The real matrix is 1.1 GB, so every test builds a small fixture in the same
on-disk layout and exercises the preprocessing chain against it.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

from benchmarks.datasets import load_cusanovich

N_CELLS = 120
N_PEAKS = 400
TISSUE_CODE = {"Lung": 0.0, "Liver": 1.0, "Spleen": 2.0}


def _planted_cluster(tsne_1: np.ndarray, tsne_2: np.ndarray) -> np.ndarray:
    """The fixture's source cluster, recomputed from its t-SNE columns.

    Two clusters per tissue, split by the parity of the cell's matrix column,
    so a row of source_labels can be checked against the same row of
    source_tsne without reconstructing the loader's filter or subsample.
    """
    return (2 * tsne_2 + tsne_1 % 2 + 1).astype(int)


@pytest.fixture
def atlas(tmp_path):
    rng = np.random.default_rng(0)
    tissues = np.repeat(["Lung", "Liver", "Spleen"], N_CELLS // 3)

    # Three tissue-specific peak blocks plus a shared background, so the
    # embedding has structure a clustering can actually recover.
    M = np.zeros((N_PEAKS, N_CELLS), dtype=np.int8)
    for block, tissue in enumerate(["Lung", "Liver", "Spleen"]):
        rows = slice(block * 100, (block + 1) * 100)
        cols = tissues == tissue
        M[rows, cols] = rng.random((100, cols.sum())) < 0.6
    M[300:, :] = rng.random((100, N_CELLS)) < 0.3

    root = tmp_path / "data"
    d = root / "Cusanovich"
    (d / "matrices").mkdir(parents=True)
    (d / "metadata").mkdir(parents=True)

    path = d / "matrices" / "atac_matrix.binary.qc_filtered.mtx"
    mmwrite(str(path), sparse.csr_matrix(M), field="integer")
    with open(path, "rb") as src, gzip.open(str(path) + ".gz", "wb") as dst:
        dst.write(src.read())
    path.unlink()

    cells = [f"cell{i:04d}" for i in range(N_CELLS)]
    # The release's .cells.txt holds the same barcodes as cell_metadata.txt
    # but in a different order, and it is the metadata's row order, not the
    # cells file, that matches the matrix columns. Write the cells file
    # shuffled so a loader that reindexes by it scrambles the labels here
    # the way it does on the real atlas.
    (d / "matrices" / "atac_matrix.binary.qc_filtered.cells.txt").write_text(
        "\n".join(cells[i] for i in rng.permutation(N_CELLS)) + "\n"
    )
    (d / "matrices" / "atac_matrix.binary.qc_filtered.peaks.txt").write_text(
        "\n".join(f"chr1_{i}_{i + 100}" for i in range(N_PEAKS)) + "\n"
    )
    # The source's own t-SNE coordinates. tsne_1 encodes the cell's column
    # in the matrix and tsne_2 its tissue, and the source cluster is a
    # function of both, so a test can check which cell a row belongs to
    # after the loader has filtered and subsampled. Rows are in matrix
    # column order, as in the real release.
    tsne_1 = np.arange(N_CELLS, dtype=float)
    tsne_2 = np.array([TISSUE_CODE[t] for t in tissues], dtype=float)
    pd.DataFrame(
        {
            "cell": cells,
            "tissue": tissues,
            "cluster": _planted_cluster(tsne_1, tsne_2),
            "subset_cluster": np.arange(N_CELLS) % 3 + 1,
            "tsne_1": tsne_1,
            "tsne_2": tsne_2,
            "cell_label": np.where(rng.random(N_CELLS) < 0.15, "Unknown", tissues),
        }
    ).to_csv(d / "metadata" / "cell_metadata.txt", sep="\t", index=False)
    return root


def _metadata(atlas: Path) -> pd.DataFrame:
    return pd.read_csv(atlas / "Cusanovich" / "metadata" / "cell_metadata.txt", sep="\t")


class TestLoadCusanovich:
    def test_returns_cells_by_components(self, atlas):
        X, y, meta = load_cusanovich(root=atlas, n_components=10)
        assert X.ndim == 2
        assert X.shape[1] == 10
        assert X.shape[0] == y.shape[0]
        assert np.isfinite(X).all()

    def test_every_cell_is_kept_by_default(self, atlas):
        # The source's clusters assign every cell, Unknown-labeled ones
        # included, so the default keeps them all and still reports how many
        # are annotated.
        X, _, meta = load_cusanovich(root=atlas, n_components=10)
        labels = meta["source_labels"]["cell_label"]
        assert X.shape[0] == N_CELLS
        assert meta["n_cells"] == N_CELLS
        assert "Unknown" in set(labels)
        assert meta["n_cells_annotated"] == int((labels != "Unknown").sum())
        assert meta["n_cells_annotated"] < N_CELLS

    def test_drop_unknown_keeps_only_the_annotated_cells(self, atlas):
        X, _, meta = load_cusanovich(root=atlas, n_components=10, drop_unknown=True)
        annotated = int((_metadata(atlas)["cell_label"] != "Unknown").sum())
        assert X.shape[0] == annotated
        assert meta["n_cells_annotated"] == annotated
        assert meta["n_cells_full"] == N_CELLS
        assert "Unknown" not in set(meta["source_labels"]["cell_label"])

    def test_label_column_defaults_to_the_source_clusters(self, atlas):
        _, cluster, _ = load_cusanovich(root=atlas, n_components=10)
        _, tissue, _ = load_cusanovich(
            root=atlas, n_components=10, label_column="tissue"
        )
        assert cluster.name == "cluster"
        assert set(cluster) == {str(c) for c in range(1, 7)}
        assert tissue.name == "tissue"

    def test_site_frequency_threshold_filters_peaks(self, atlas):
        _, _, loose = load_cusanovich(
            root=atlas, n_components=10, site_frequency_threshold=0.0
        )
        # threshold=0.5 leaves zero peaks for this fixture (see
        # test_zero_surviving_peaks_warns_and_does_not_claim_svd_ran, below),
        # which load_cusanovich warns about; that warning is exercised on
        # its own there; it is incidental here and must not leak into the
        # suite's output.
        with pytest.warns(UserWarning, match="site_frequency_threshold=0.5"):
            _, _, strict = load_cusanovich(
                root=atlas, n_components=10, site_frequency_threshold=0.5
            )
        assert strict["n_peaks_kept"] < loose["n_peaks_kept"]

    def test_zero_surviving_peaks_warns_and_does_not_claim_svd_ran(self, atlas):
        # threshold=0.5 leaves zero peaks for this fixture (see the test
        # above), so X degenerates to a zero embedding. That must be loud,
        # not silent: a warning naming the threshold, and meta that does not
        # claim TF-IDF/SVD ran when they were skipped.
        with pytest.warns(UserWarning, match="site_frequency_threshold=0.5"):
            _, _, meta = load_cusanovich(
                root=atlas, n_components=10, site_frequency_threshold=0.5
            )
        assert meta["n_peaks_kept"] == 0
        chain = " ".join(meta["preprocessing"])
        assert "skipped" in chain
        assert "TruncatedSVD(n_components=10);" not in chain
        assert "TF-IDF: per-cell term frequency" not in chain

    def test_labels_follow_the_metadata_row_order_not_the_cells_file(self, atlas):
        # On the released atlas the matrix columns are in cell_metadata.txt
        # row order; .cells.txt lists the same barcodes in another order.
        # Reindexing by the cells file kept tissue blocks roughly intact but
        # gave every cell another cell's label, cluster and t-SNE position
        # within its tissue. Row i of y must be metadata row i.
        _, y, meta = load_cusanovich(root=atlas, n_components=10)
        metadata = _metadata(atlas)
        np.testing.assert_array_equal(
            y.to_numpy(), metadata["cluster"].astype(str).to_numpy()
        )
        np.testing.assert_array_equal(
            meta["source_tsne"][:, 0], metadata["tsne_1"].to_numpy()
        )

    def test_cells_file_disagreeing_with_the_metadata_raises(self, atlas):
        # The cells file is still read: it is the only independent record of
        # which barcodes the matrix holds, so a mismatch with the metadata
        # means the two files are not from the same release.
        path = atlas / "Cusanovich" / "matrices" / "atac_matrix.binary.qc_filtered.cells.txt"
        lines = path.read_text().split()
        lines[0] = "not_a_cell"
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="cells.txt"):
            load_cusanovich(root=atlas, n_components=10)

    def test_embedding_separates_the_planted_tissues(self, atlas):
        # A loader that returned noise would satisfy every shape assertion
        # above, so pin that the preprocessing preserves real structure.
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        X, y, _ = load_cusanovich(root=atlas, n_components=10, label_column="tissue")
        labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_subsample_is_stratified_by_the_source_clusters(self, atlas):
        # Six planted clusters of 20 cells each; a draw of 30 stratified by
        # cluster takes exactly five from each.
        X, y, meta = load_cusanovich(root=atlas, n_components=10, subsample=30)
        assert X.shape[0] == 30
        assert meta["subsample"] == 30
        assert y.value_counts().to_dict() == {str(c): 5 for c in range(1, 7)}

    def test_source_tsne_rows_follow_x_through_filter_and_subsample(self, atlas):
        # meta["source_tsne"] is the source's own t-SNE, carried through as
        # an (n, 2) array whose row i is the same cell as row i of X and y.
        # The fixture encodes each cell's matrix column in tsne_1 and its
        # tissue in tsne_2, so both the Unknown filter and the stratified
        # subsample can be checked without reconstructing the split.
        X, y, meta = load_cusanovich(
            root=atlas,
            n_components=10,
            subsample=30,
            drop_unknown=True,
            label_column="tissue",
        )
        tsne = meta["source_tsne"]
        assert isinstance(tsne, np.ndarray)
        assert tsne.dtype == np.float64
        assert tsne.shape == (X.shape[0], 2)

        # Row i's tissue code matches row i of y.
        expected = np.array([TISSUE_CODE[t] for t in y], dtype=float)
        np.testing.assert_array_equal(tsne[:, 1], expected)

        # Every row is a distinct annotated cell: no Unknown cell survives
        # and no cell appears twice.
        metadata = _metadata(atlas).set_index("cell")
        columns = tsne[:, 0].astype(int)
        assert len(set(columns)) == len(columns)
        labels = metadata["cell_label"].loc[[f"cell{i:04d}" for i in columns]]
        assert "Unknown" not in set(labels)

    def test_source_labels_rows_follow_x_through_filter_and_subsample(self, atlas):
        # meta["source_labels"] is filtered and subsampled with X, so row i
        # names the same cell as row i of y and of source_tsne. The fixture's
        # cluster is a function of that cell's t-SNE columns, so a table left
        # unsubsampled, or subsampled with a different index, disagrees.
        X, y, meta = load_cusanovich(
            root=atlas, n_components=10, subsample=30, drop_unknown=True
        )
        labels = meta["source_labels"]
        tsne = meta["source_tsne"]
        assert list(labels.columns) == [
            "tissue",
            "cluster",
            "subset_cluster",
            "cell_label",
        ]
        assert len(labels) == X.shape[0]
        np.testing.assert_array_equal(
            labels["cluster"].astype(str).to_numpy(), y.to_numpy()
        )
        np.testing.assert_array_equal(
            labels["cluster"].to_numpy(), _planted_cluster(tsne[:, 0], tsne[:, 1])
        )
        np.testing.assert_array_equal(
            labels["tissue"].map(TISSUE_CODE).to_numpy(), tsne[:, 1]
        )
        assert "Unknown" not in set(labels["cell_label"])

    def test_source_tables_are_full_length_without_subsample(self, atlas):
        X, _, meta = load_cusanovich(root=atlas, n_components=10)
        assert meta["source_tsne"].shape == (X.shape[0], 2)
        assert meta["source_labels"].shape == (X.shape[0], 4)

    def test_meta_records_the_preprocessing_chain(self, atlas):
        _, _, meta = load_cusanovich(root=atlas, n_components=10)
        chain = " ".join(meta["preprocessing"])
        assert "TF-IDF" in chain
        assert "3%" in chain or "site_frequency_threshold" in chain
        assert "keep every cell" in chain
        assert meta["drop_unknown"] is False
        assert meta["drops_first_component"] is False
        assert meta["source"] == "Cusanovich"

    def test_missing_data_directory_names_the_download(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="mouse_atlas_data_release"):
            load_cusanovich(root=tmp_path / "nothing")

    def test_missing_data_directory_names_a_path_that_resolves_correctly(
        self, tmp_path, monkeypatch
    ):
        # load_cusanovich takes no root in the notebooks, so a relative
        # "data/" resolves against the process's cwd at call time -- the
        # notebook's own directory when run via nbconvert, not the
        # repository root. A message that just repeats "data/Cusanovich/"
        # reads correctly only from the repo root; the resolved absolute
        # path is correct regardless of where the process actually started.
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError) as excinfo:
            load_cusanovich(root=Path("nothing"))
        expected = Path("nothing").resolve()
        assert str(expected) in str(excinfo.value)
