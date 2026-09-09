"""hECA loader tests.

The real organs are gigabytes each, so the fixtures are small h5ad files in
the same layout: CSR raw counts, cells by cPeaks, annotations in obs, and an
empty obsm.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from benchmarks.datasets import load_heca
from benchmarks.datasets._heca import peak_open_counts, reduce_to_peaks

N_PEAKS = 300


def _organ(path, name, n_cells, seed):
    rng = np.random.default_rng(seed)
    types = rng.choice(["T cell", "Epithelial cell", "Unclassified"], n_cells)
    M = np.zeros((n_cells, N_PEAKS))
    # An organ-specific block plus a shared background gives the embedding
    # structure that clustering can recover.
    offset = {"Lung": 0, "Brain": 100}.get(name, 200)
    M[:, offset : offset + 100] = rng.random((n_cells, 100)) < 0.5
    M[:, 250:] = rng.random((n_cells, 50)) < 0.4

    adata = ad.AnnData(
        X=sparse.csr_matrix(M.astype(np.int32)),
        obs=pd.DataFrame(
            {
                "cell_type": types,
                "organ": name,
                "donor_id": rng.choice(["d1", "d2"], n_cells),
                "study_id": "10.1000/x",
            },
            index=[f"{name}_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"peak{i}" for i in range(N_PEAKS)]),
    )
    adata.write_h5ad(path / f"ATAC-{name}.h5ad")


@pytest.fixture
def heca(tmp_path):
    root = tmp_path / "data"
    d = root / "hECA"
    d.mkdir(parents=True)
    _organ(d, "Lung", 90, 0)
    _organ(d, "Brain", 80, 1)
    return root


class TestChunkedPasses:
    def test_open_counts_match_a_dense_computation(self, heca):
        path = heca / "hECA" / "ATAC-Lung.h5ad"
        counts, n_cells = peak_open_counts(path, block=17)
        dense = ad.read_h5ad(path).X.toarray()
        assert n_cells == dense.shape[0]
        assert np.array_equal(counts, (dense > 0).sum(axis=0))

    def test_open_counts_are_block_size_invariant(self, heca):
        path = heca / "hECA" / "ATAC-Brain.h5ad"
        a, _ = peak_open_counts(path, block=7)
        b, _ = peak_open_counts(path, block=100_000)
        assert np.array_equal(a, b)

    def test_reduction_selects_the_requested_columns(self, heca):
        path = heca / "hECA" / "ATAC-Lung.h5ad"
        peaks = np.array([3, 10, 250])
        reduced = reduce_to_peaks(path, peaks, block=13)
        dense = ad.read_h5ad(path).X.toarray()
        assert reduced.shape == (dense.shape[0], peaks.size)
        assert np.array_equal(reduced.toarray(), dense[:, peaks])


class TestLoadHeca:
    def test_pools_the_requested_organs(self, heca):
        X, y, meta = load_heca(
            root=heca, organs=["Lung", "Brain"], n_top_peaks=50, n_components=5
        )
        assert X.shape[1] == 5
        assert set(y) == {"Lung", "Brain"}
        assert meta["organs"] == ["Lung", "Brain"]
        assert meta["n_cells"] == X.shape[0]

    def test_unclassified_cells_are_dropped(self, heca):
        _, _, meta = load_heca(
            root=heca,
            organs=["Lung"],
            n_top_peaks=50,
            n_components=5,
            label_column="cell_type",
        )
        assert meta["n_cells"] < meta["n_cells_full"]

    def test_unclassified_cells_are_dropped_even_with_the_organ_label(self, heca):
        # "organ" is never "Unclassified", so a filter that checks
        # label_column instead of a fixed annotation column would silently
        # keep every cell_type-Unclassified cell whenever label_column is
        # left at its default. Drop must happen regardless of label_column.
        _, _, meta = load_heca(
            root=heca, organs=["Lung"], n_top_peaks=50, n_components=5
        )
        assert meta["n_cells"] < meta["n_cells_full"]

    def test_cell_type_is_available_as_the_reference(self, heca):
        _, y, _ = load_heca(
            root=heca,
            organs=["Lung"],
            n_top_peaks=50,
            n_components=5,
            label_column="cell_type",
        )
        assert "Unclassified" not in set(y)
        assert y.name == "cell_type"

    def test_embedding_separates_the_planted_organs(self, heca):
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        X, y, _ = load_heca(
            root=heca, organs=["Lung", "Brain"], n_top_peaks=100, n_components=5
        )
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_features_are_shared_across_organs(self, heca):
        # Selecting features per organ would make the pooled embedding
        # meaningless, so the selection must be computed across all of them.
        _, _, meta = load_heca(
            root=heca, organs=["Lung", "Brain"], n_top_peaks=50, n_components=5
        )
        assert meta["n_peaks_selected"] == 50
        assert meta["feature_selection_scope"] == "pooled across organs"

    def test_subsample_is_stratified_and_sized(self, heca):
        X, y, meta = load_heca(
            root=heca,
            organs=["Lung", "Brain"],
            n_top_peaks=50,
            n_components=5,
            subsample=40,
        )
        assert X.shape[0] == 40
        assert y.nunique() == 2

    def test_meta_records_the_hvg_deviation(self, heca):
        _, _, meta = load_heca(
            root=heca, organs=["Lung"], n_top_peaks=50, n_components=5
        )
        chain = " ".join(meta["preprocessing"])
        assert "0.5%" in chain
        assert "log1p" in chain
        assert "PCA" in chain
        assert "deviation" in " ".join(meta["deviations"]).lower()

    def test_missing_organ_names_the_download(self, heca):
        with pytest.raises(FileNotFoundError, match="15627886"):
            load_heca(root=heca, organs=["Pancreas"], n_top_peaks=50)

    def test_open_fraction_that_empties_every_peak_raises(self, heca):
        # A threshold above 1.0 can never be met, so the pass-1 accumulator
        # must reject it rather than silently handing PCA a zero-feature
        # matrix and reporting a preprocessing chain that never ran.
        with pytest.raises(ValueError, match="open_fraction"):
            load_heca(
                root=heca,
                organs=["Lung"],
                open_fraction=2.0,
                n_top_peaks=50,
                n_components=5,
            )


class TestCache:
    # The interface's signature carries cache: bool = True, but the plan's
    # own implementation sketch never used it, so this fills the gap and
    # verifies it actually skips recomputation rather than being a dead
    # parameter.

    def test_cache_survives_source_deletion(self, heca):
        first_X, first_y, first_meta = load_heca(
            root=heca, organs=["Lung"], n_top_peaks=50, n_components=5
        )
        assert first_meta["cached"] is False

        (heca / "hECA" / "ATAC-Lung.h5ad").unlink()

        second_X, second_y, second_meta = load_heca(
            root=heca, organs=["Lung"], n_top_peaks=50, n_components=5
        )
        assert second_meta["cached"] is True
        np.testing.assert_array_equal(second_X, first_X)
        assert list(second_y) == list(first_y)

    def test_cache_false_does_not_survive_source_deletion(self, heca):
        load_heca(
            root=heca,
            organs=["Lung"],
            n_top_peaks=50,
            n_components=5,
            cache=False,
        )
        (heca / "hECA" / "ATAC-Lung.h5ad").unlink()

        with pytest.raises(FileNotFoundError):
            load_heca(
                root=heca,
                organs=["Lung"],
                n_top_peaks=50,
                n_components=5,
                cache=False,
            )
