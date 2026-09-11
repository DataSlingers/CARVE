"""hECA loader tests.

The real organs are gigabytes each, so the fixtures are small h5ad files in
the same layout: CSR raw counts, cells by cPeaks, annotations in obs, and an
empty obsm.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from benchmarks.datasets import load_heca
from benchmarks.datasets._heca import peak_open_counts, read_rows, reduce_to_peaks

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


    def test_read_rows_matches_the_dense_rows(self, heca):
        path = heca / "hECA" / "ATAC-Lung.h5ad"
        dense = ad.read_h5ad(path).X.toarray()
        rows = np.array([3, 4, 17, 42, 88])
        out = read_rows(path, rows)
        assert out.shape == (5, N_PEAKS)
        np.testing.assert_array_equal(out.toarray(), dense[rows])


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

    def test_random_state_controls_the_pca_embedding(self, heca):
        # cache=False on every call: a cache hit would serve the first
        # call's stored embedding regardless of random_state and mask a
        # real difference (or a real absence of one).
        kwargs = dict(
            root=heca,
            organs=["Lung", "Brain"],
            n_top_peaks=50,
            n_components=5,
            cache=False,
        )
        X_seed0, _, _ = load_heca(random_state=0, **kwargs)
        X_seed1, _, _ = load_heca(random_state=1, **kwargs)
        X_seed0_again, _, _ = load_heca(random_state=0, **kwargs)

        assert not np.array_equal(X_seed0, X_seed1)
        np.testing.assert_array_equal(X_seed0, X_seed0_again)

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

    def test_default_path_embeds_the_pooled_population(self, heca):
        _, _, meta = load_heca(root=heca, organs=["Lung", "Brain"], n_top_peaks=50)
        assert meta["embedding_population"] == "pooled"

    def test_meta_records_the_hvg_deviation(self, heca):
        _, _, meta = load_heca(
            root=heca, organs=["Lung"], n_top_peaks=50, n_components=5
        )
        chain = " ".join(meta["preprocessing"])
        assert "0.5%" in chain
        assert "log1p" in chain
        assert "PCA" in chain
        assert "deviation" in " ".join(meta["deviations"]).lower()

    def test_mismatched_peak_reference_across_organs_is_rejected(self, heca):
        # Lung and Brain (the `heca` fixture) share one var index. A third
        # organ with the same peak count but a different order must be
        # caught before pooling silently mixes different peaks across
        # organs -- total_counts + counts only checks length, so it would
        # not notice on its own.
        adata = ad.AnnData(
            X=sparse.csr_matrix(np.zeros((10, N_PEAKS), dtype=np.int32)),
            obs=pd.DataFrame(
                {
                    "cell_type": ["T cell"] * 10,
                    "organ": "Kidney",
                    "donor_id": "d1",
                    "study_id": "10.1000/x",
                },
                index=[f"Kidney_{i}" for i in range(10)],
            ),
            var=pd.DataFrame(index=[f"peak{i}" for i in range(N_PEAKS)][::-1]),
        )
        adata.write_h5ad(heca / "hECA" / "ATAC-Kidney.h5ad")

        with pytest.raises(ValueError, match="Kidney"):
            load_heca(
                root=heca,
                organs=["Lung", "Kidney"],
                n_top_peaks=50,
                n_components=5,
            )

    def test_missing_organ_names_the_download(self, heca):
        with pytest.raises(FileNotFoundError, match="15627886"):
            load_heca(root=heca, organs=["Pancreas"], n_top_peaks=50)

    def test_missing_organ_names_a_path_that_resolves_correctly(self, heca, monkeypatch):
        # load_heca takes no root in the notebooks, so a relative "data/"
        # resolves against the process's cwd at call time -- the notebook's
        # own directory when run via nbconvert, not the repository root. A
        # message that just repeats "data/hECA/" reads correctly only from
        # the repo root; the resolved absolute path is correct regardless
        # of where the process actually started.
        monkeypatch.chdir(heca.parent)
        with pytest.raises(FileNotFoundError) as excinfo:
            load_heca(root=Path("data"), organs=["Pancreas"], n_top_peaks=50)
        expected = (Path("data") / "hECA" / "ATAC-Pancreas.h5ad").resolve()
        assert str(expected) in str(excinfo.value)

    def test_truncated_organ_file_names_the_file_and_the_fix(self, heca):
        # A download that stopped short leaves a file HDF5 can open but not
        # read: h5py's own error ("free block size is zero?", "bad symbol
        # table node signature") surfaces from ten frames deep and names
        # neither the file nor what to do. The real ATAC-Brain.h5ad failed
        # exactly this way after an interrupted download.
        path = heca / "hECA" / "ATAC-Brain.h5ad"
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])
        with pytest.raises(OSError, match="ATAC-Brain.h5ad") as excinfo:
            load_heca(root=heca, organs=["Lung", "Brain"], n_top_peaks=50)
        message = str(excinfo.value)
        assert str(path.resolve()) in message
        assert "truncated or corrupt" in message
        assert "md5" in message

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

    def test_changing_n_top_peaks_invalidates_the_cache(self, heca):
        # n_top_peaks is part of _cache_key alongside organs, open_fraction,
        # n_components, and label_column. A change to it must miss the
        # cache and recompute -- not silently serve the previous call's
        # cached embedding, selected under a different peak count.
        first_X, _, first_meta = load_heca(
            root=heca, organs=["Lung"], n_top_peaks=50, n_components=5
        )
        assert first_meta["cached"] is False
        assert first_meta["n_peaks_selected"] == 50

        second_X, _, second_meta = load_heca(
            root=heca, organs=["Lung"], n_top_peaks=100, n_components=5
        )
        assert second_meta["cached"] is False
        assert second_meta["n_peaks_selected"] == 100
        assert first_X.shape[1] == second_X.shape[1] == 5

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



def _annotated_per_organ(root, organs):
    counts = {}
    for organ in organs:
        obs = ad.read_h5ad(root / "hECA" / f"ATAC-{organ}.h5ad", backed="r").obs
        counts[organ] = int((obs["cell_type"] != "Unclassified").sum())
    return counts


class TestSubsampleBeforeEmbedding:
    """The dev-scale path: draw the rows first, embed only those.

    The pooled pass holds the whole cells-by-kept-peaks matrix in memory
    (tens of GB on the real five organs), and the subsample was drawn from
    the pooled embedding afterwards, so even SCALE="dev" could not load on
    a laptop without a cache. With subsample_before_embedding=True the rows
    are drawn from the organ files and the preprocessing chain runs on those
    rows only. The pooled embedding still wins whenever it is cached.
    """

    def test_embeds_only_the_drawn_rows_and_never_runs_the_pooled_pass(self, heca):
        X, y, meta = load_heca(
            root=heca,
            organs=["Lung", "Brain"],
            subsample=40,
            n_top_peaks=50,
            subsample_before_embedding=True,
        )
        assert X.shape[0] == 40
        assert y.shape[0] == 40
        assert meta["embedding_population"] == "subsample"
        assert meta["n_cells"] == 40
        assert meta["cached"] is False

        # The pooled embedding was never computed: a pooled call afterwards
        # finds no cache to read.
        _, _, pooled = load_heca(root=heca, organs=["Lung", "Brain"], n_top_peaks=50)
        assert pooled["cached"] is False

    def test_rows_are_stratified_by_organ_over_annotated_cells(self, heca):
        # A third, much smaller organ, so a proportional allocation and an
        # equal split disagree by more than the rounding tolerance below.
        _organ(heca / "hECA", "Kidney", 20, 2)
        organs = ["Lung", "Brain", "Kidney"]
        annotated = _annotated_per_organ(heca, organs)
        _, y, meta = load_heca(
            root=heca,
            organs=organs,
            subsample=40,
            n_top_peaks=50,
            subsample_before_embedding=True,
        )
        total = sum(annotated.values())
        assert meta["n_cells_full"] == 190
        assert meta["n_cells_annotated"] == total
        for organ, n_annotated in annotated.items():
            expected = 40 * n_annotated / total
            assert abs(int((y == organ).sum()) - expected) <= 1

    def test_unclassified_cells_are_never_drawn(self, heca):
        _, label, _ = load_heca(
            root=heca,
            organs=["Lung", "Brain"],
            subsample=40,
            n_top_peaks=50,
            label_column="cell_type",
            subsample_before_embedding=True,
        )
        assert "Unclassified" not in set(label)

    def test_embedding_separates_the_planted_organs(self, heca):
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        X, y, _ = load_heca(
            root=heca,
            organs=["Lung", "Brain"],
            subsample=60,
            n_top_peaks=50,
            n_components=5,
            subsample_before_embedding=True,
        )
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_is_cached_by_subsample_and_seed(self, heca):
        kwargs = dict(
            root=heca,
            organs=["Lung", "Brain"],
            subsample=40,
            n_top_peaks=50,
            subsample_before_embedding=True,
        )
        X1, y1, meta1 = load_heca(**kwargs)
        X2, y2, meta2 = load_heca(**kwargs)
        assert meta1["cached"] is False
        assert meta2["cached"] is True
        assert meta2["embedding_population"] == "subsample"
        np.testing.assert_array_equal(X1, X2)
        assert y1.equals(y2)

        X3, _, meta3 = load_heca(**kwargs, random_state=7)
        assert meta3["cached"] is False
        assert not np.array_equal(X1, X3)

    def test_pooled_cache_wins_when_present(self, heca):
        load_heca(root=heca, organs=["Lung", "Brain"], n_top_peaks=50)
        X, y, meta = load_heca(
            root=heca,
            organs=["Lung", "Brain"],
            subsample=40,
            n_top_peaks=50,
            subsample_before_embedding=True,
        )
        assert meta["embedding_population"] == "pooled"
        assert meta["cached"] is True
        assert X.shape[0] == 40

    def test_without_a_subsample_the_flag_changes_nothing(self, heca):
        _, _, meta = load_heca(
            root=heca,
            organs=["Lung", "Brain"],
            n_top_peaks=50,
            subsample_before_embedding=True,
        )
        assert meta["embedding_population"] == "pooled"
        assert meta["n_cells"] == meta["n_cells_annotated"]

    def test_open_fraction_is_applied_to_the_drawn_rows(self, heca):
        kwargs = dict(
            root=heca,
            organs=["Lung", "Brain"],
            subsample=40,
            n_top_peaks=50,
            cache=False,
            subsample_before_embedding=True,
        )
        _, _, loose = load_heca(**kwargs, open_fraction=0.0)
        _, _, strict = load_heca(**kwargs, open_fraction=0.45)
        assert strict["n_peaks_open"] < loose["n_peaks_open"]
        with pytest.raises(ValueError, match="open_fraction"):
            load_heca(**kwargs, open_fraction=0.99)

    def test_preprocessing_says_features_were_selected_on_the_subsample(self, heca):
        _, _, meta = load_heca(
            root=heca,
            organs=["Lung", "Brain"],
            subsample=40,
            n_top_peaks=50,
            subsample_before_embedding=True,
        )
        chain = " ".join(meta["preprocessing"])
        assert "subsample" in chain
        assert "subsample" in meta["feature_selection_scope"]

