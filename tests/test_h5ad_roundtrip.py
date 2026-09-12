"""Persistence tests: CARVE results must survive an h5ad round trip.

The load-bearing case is a *heterogeneous* estimator grid. KMeans has no
``linkage`` and agglomerative clustering has no ``affinity``, so
``DataFrame.from_records`` fills the gaps with NaN. On pandas 2.3 -- the
dependency floor, since anndata 0.13 requires pandas >= 2.3 -- those columns
land in object dtype mixing ``str`` with ``float('nan')``, and anndata's
variable-length string writer rejects them with ``TypeError``. Under pandas 3
the new ``str`` dtype hides the problem entirely, so this file is only a real
regression test on the floor. CI installs the newest releases only, so a
regression here would surface in a local environment on pandas 2.3, not in CI.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans

import carve
from carve import CARVE, SpectralClustering
from carve._anndata import results_from_uns

PLOTS = [
    "metric_over_n_clusters",
    "consensus_matrix",
    "cluster_boxplot",
    "cluster_violin",
    "cluster_scatter",
    "diagnostic_scatter",
]


@pytest.fixture(scope="module")
def heterogeneous_grid():
    """A grid whose estimators do not share a parameter vocabulary."""
    ks = list(range(2, 5))
    return [
        (KMeans, {"n_clusters": ks, "n_init": [3]}),
        (AgglomerativeClustering, {"n_clusters": ks, "linkage": ["ward", "average"]}),
        (SpectralClustering, {"n_clusters": ks, "affinity": ["self_tuning"]}),
    ]


@pytest.fixture(scope="module")
def written(heterogeneous_grid):
    """An AnnData carrying results from a heterogeneous grid."""
    rng = np.random.RandomState(0)
    X = np.vstack(
        [rng.randn(30, 6) + 9, rng.randn(30, 6) - 9, rng.randn(30, 6) * 0.4]
    ).astype(np.float32)
    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = X[:, :4]
    adata.obsm["X_umap"] = X[:, :2]
    carve.tl.carve(
        adata,
        use_rep="X_pca",
        estimator_param_grids=heterogeneous_grid,
        n_resamples=4,
        random_state=0,
    )
    return adata


@pytest.fixture(scope="module")
def reloaded(written, tmp_path_factory):
    path = tmp_path_factory.mktemp("h5ad") / "carve.h5ad"
    written.write_h5ad(path)
    return ad.read_h5ad(path)


class TestSanitisation:
    def test_results_have_no_float_nan_in_string_columns(self, written):
        df = written.uns["carve"]["results"]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(
                df[col]
            ):
                continue
            assert not any(isinstance(v, float) for v in df[col]), col

    def test_missing_parameters_become_empty_strings(self, written):
        """KMeans rows carry no linkage; that gap must be "" not NaN."""
        df = written.uns["carve"]["results"]
        assert "linkage" in df.columns
        kmeans_rows = df[df["estimator"] == "KMeans"]
        assert len(kmeans_rows) > 0
        assert set(kmeans_rows["linkage"]) == {""}

    def test_sanitiser_is_load_bearing(self, heterogeneous_grid, tmp_path):
        """The raw frame is what breaks; ours is what fixes it.

        On pandas 2.3 writing the unsanitised frame raises
        ``TypeError: Can't implicitly convert non-string objects to strings``.
        On pandas 3 the new ``str`` dtype makes it succeed, so the assertion is
        conditioned on the pandas major version rather than pretending the
        failure is universal. Either way the sanitised frame must write.
        """
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(20, 5) + 9, rng.randn(20, 5) - 9]).astype(np.float32)
        model = CARVE(
            estimator_param_grids=heterogeneous_grid,
            n_resamples=3,
            random_state=0,
        ).fit(X)

        raw = ad.AnnData(X)
        raw.uns["naive"] = {"results": model.estimator_results_}
        pandas_major = int(pd.__version__.split(".")[0])
        if pandas_major < 3:
            with pytest.raises(TypeError, match="non-string objects"):
                raw.write_h5ad(tmp_path / "naive.h5ad")
        else:
            raw.write_h5ad(tmp_path / "naive.h5ad")

        clean = ad.AnnData(X)
        carve.tl.attach_results(clean, model)
        clean.write_h5ad(tmp_path / "clean.h5ad")  # must work on both


class TestRoundTrip:
    def test_obs_columns_survive(self, written, reloaded):
        for col in written.obs.columns:
            if col == "carve":
                assert (
                    written.obs[col].astype(str) == reloaded.obs[col].astype(str)
                ).all()
            else:
                np.testing.assert_allclose(
                    written.obs[col].to_numpy(),
                    reloaded.obs[col].to_numpy(),
                )

    def test_labels_stay_categorical(self, reloaded):
        assert isinstance(reloaded.obs["carve"].dtype, pd.CategoricalDtype)

    def test_consensus_matrix_survives(self, written, reloaded):
        np.testing.assert_allclose(
            np.nan_to_num(np.asarray(written.obsp["carve_consensus"])),
            np.nan_to_num(np.asarray(reloaded.obsp["carve_consensus"])),
        )

    def test_numeric_results_survive(self, written, reloaded):
        before = written.uns["carve"]["results"]
        after = results_from_uns(reloaded.uns["carve"]["results"])
        numeric = [
            c for c in before.columns if pd.api.types.is_numeric_dtype(before[c])
        ]
        assert len(numeric) > 5
        for col in numeric:
            np.testing.assert_allclose(
                before[col].to_numpy(dtype=float),
                after[col].to_numpy(dtype=float),
                equal_nan=True,
                err_msg=col,
            )

    def test_params_survive(self, written, reloaded):
        before = written.uns["carve"]["params"]
        after = reloaded.uns["carve"]["params"]
        assert set(before) == set(after)
        for key, value in before.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_allclose(value, after[key])
            else:
                assert str(value) == str(after[key]), key

    def test_results_index_is_reset(self, reloaded):
        df = results_from_uns(reloaded.uns["carve"]["results"])
        assert list(df.index) == list(range(len(df)))

    def test_config_id_is_integral(self, reloaded):
        df = results_from_uns(reloaded.uns["carve"]["results"])
        assert df["config_id"].dtype.kind == "i"


class TestPlottingFromDisk:
    @pytest.mark.parametrize("name", PLOTS)
    def test_plot_from_reloaded(self, reloaded, name):
        """Every plot must work off a file, with no fitted model in memory."""
        from matplotlib.axes import Axes

        assert isinstance(getattr(carve.pl, name)(reloaded), Axes)


class TestZarr:
    def test_zarr_round_trip(self, written, tmp_path):
        pytest.importorskip("zarr")
        path = tmp_path / "carve.zarr"
        written.write_zarr(path)
        back = ad.read_zarr(path)
        assert (written.obs["carve"].astype(str) == back.obs["carve"].astype(str)).all()
