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
    (d / "matrices" / "atac_matrix.binary.qc_filtered.cells.txt").write_text(
        "\n".join(cells) + "\n"
    )
    (d / "matrices" / "atac_matrix.binary.qc_filtered.peaks.txt").write_text(
        "\n".join(f"chr1_{i}_{i + 100}" for i in range(N_PEAKS)) + "\n"
    )
    pd.DataFrame(
        {
            "cell": cells,
            "tissue": tissues,
            "cluster": rng.integers(1, 5, N_CELLS),
            "cell_label": np.where(rng.random(N_CELLS) < 0.15, "Unknown", tissues),
        }
    ).to_csv(d / "metadata" / "cell_metadata.txt", sep="\t", index=False)
    return root


class TestLoadCusanovich:
    def test_returns_cells_by_components(self, atlas):
        X, y, meta = load_cusanovich(root=atlas, n_components=10)
        assert X.ndim == 2
        assert X.shape[1] == 10
        assert X.shape[0] == y.shape[0]
        assert np.isfinite(X).all()

    def test_unknown_cells_are_dropped(self, atlas):
        X, y, meta = load_cusanovich(root=atlas, n_components=10)
        assert "Unknown" not in set(y)
        assert meta["n_cells"] < N_CELLS
        assert meta["n_cells_full"] == N_CELLS

    def test_label_column_selects_the_reference(self, atlas):
        _, tissue, _ = load_cusanovich(root=atlas, n_components=10)
        _, label, _ = load_cusanovich(
            root=atlas, n_components=10, label_column="cell_label"
        )
        assert tissue.name == "tissue"
        assert label.name == "cell_label"

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

    def test_embedding_separates_the_planted_tissues(self, atlas):
        # A loader that returned noise would satisfy every shape assertion
        # above, so pin that the preprocessing preserves real structure.
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        X, y, _ = load_cusanovich(root=atlas, n_components=10)
        labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_subsample_is_stratified_and_sized(self, atlas):
        X, y, meta = load_cusanovich(root=atlas, n_components=10, subsample=30)
        assert X.shape[0] == 30
        assert meta["subsample"] == 30
        assert y.nunique() == 3

    def test_meta_records_the_preprocessing_chain(self, atlas):
        _, _, meta = load_cusanovich(root=atlas, n_components=10)
        chain = " ".join(meta["preprocessing"])
        assert "TF-IDF" in chain
        assert "3%" in chain or "site_frequency_threshold" in chain
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
