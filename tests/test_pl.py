"""Tests for carve.pl -- plots that read tl.carve's output out of an AnnData.

One CARVE fit is shared by the module. tl.carve is that fit followed by
tl.attach_results, so attach_results into a fresh AnnData reproduces every
write-side option (key_added, store_results, store_consensus, use_rep,
layer, n_pcs, a pinned k) without refitting.
"""

import anndata as ad
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.legend import Legend
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import carve
import carve.pl._plots as pl_plots
from carve import CARVE


def _X():
    rng = np.random.RandomState(0)
    return np.vstack(
        [rng.randn(30, 6) + 8, rng.randn(30, 6) - 8, rng.randn(30, 6) * 0.4]
    ).astype(np.float32)


def _bare_adata():
    """An AnnData with the representations the fit used and no results."""
    X = _X()
    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = X[:, :4]
    adata.obsm["X_umap"] = X[:, :2]
    adata.layers["counts"] = X * 2.0
    return adata


@pytest.fixture(scope="module")
def model():
    """CARVE fitted on X_pca of _bare_adata()."""
    m = CARVE(
        n_clusters=np.arange(2, 5),
        estimator_param_grids=[(KMeans, {"n_clusters": [2, 3, 4], "n_init": [3]})],
        n_resamples=5,
        random_state=0,
    )
    m.fit(_bare_adata(), use_rep="X_pca")
    return m


@pytest.fixture(scope="module")
def written(model):
    """What tl.carve(adata, use_rep="X_pca") leaves behind."""
    adata = _bare_adata()
    carve.tl.attach_results(adata, model, use_rep="X_pca")
    return adata


def _attach(model, **kwargs):
    """A fresh AnnData with attach_results applied under kwargs."""
    adata = _bare_adata()
    carve.tl.attach_results(adata, model, **kwargs)
    return adata


@pytest.fixture(scope="module")
def randomized_model():
    """CARVE fitted with randomized preprocessing, two estimators, on X_pca."""
    m = CARVE(
        n_clusters=np.arange(2, 5),
        estimator_param_grids=[
            (KMeans, {"n_clusters": [2, 3, 4], "n_init": [3]}),
            (AgglomerativeClustering, {"n_clusters": [2, 3, 4], "linkage": ["ward"]}),
        ],
        normalization_options=[(FunctionTransformer, {}), (StandardScaler, {})],
        dim_reduction_options=[(FunctionTransformer, {})],
        n_resamples=6,
        random_state=0,
    )
    m.fit(_bare_adata(), use_rep="X_pca", randomize_preprocessing=True)
    return m


@pytest.fixture(scope="module")
def randomized_written(randomized_model, tmp_path_factory):
    """A randomized result written into an AnnData and read back from h5ad."""
    adata = _bare_adata()
    carve.tl.attach_results(adata, randomized_model, use_rep="X_pca")
    path = tmp_path_factory.mktemp("h5ad") / "randomized.h5ad"
    adata.write_h5ad(path)
    return ad.read_h5ad(path)


PL_FUNCTIONS = [
    carve.pl.metric_over_n_clusters,
    carve.pl.consensus_matrix,
    carve.pl.cluster_boxplot,
    carve.pl.cluster_violin,
    carve.pl.cluster_scatter,
    carve.pl.diagnostic_scatter,
]


# -----------------------------------------------------------------------
# Each pl function draws what the corresponding CARVE method draws
# -----------------------------------------------------------------------


class TestMatchesTheModelMethods:
    """pl.* reads the stored artifacts; CARVE.plot_* re-selects from the
    fitted model. With the recorded measure and rule they must agree on the
    drawn data, not merely both return an Axes.
    """

    def test_metric_lines(self, written, model):
        a = carve.pl.metric_over_n_clusters(written)
        b = model.plot_metric_over_n_clusters(measure="stability", rule="1se")
        assert len(a.lines) == len(b.lines) > 0
        for la, lb in zip(a.lines, b.lines):
            np.testing.assert_array_equal(la.get_ydata(), lb.get_ydata())

    def test_consensus_image(self, written, model):
        a = carve.pl.consensus_matrix(written)
        b = model.plot_consensus_matrix()
        np.testing.assert_array_equal(
            np.asarray(a.images[0].get_array()), np.asarray(b.images[0].get_array())
        )

    def test_boxplot_medians(self, written, model):
        def medians(ax):
            return sorted(
                line.get_ydata()[0]
                for line in ax.lines
                if line.get_ydata().size == 2
                and line.get_ydata()[0] == line.get_ydata()[1]
            )

        a = carve.pl.cluster_boxplot(written)
        b = model.plot_cluster_boxplot()
        assert medians(a) == medians(b)
        assert len(medians(a)) >= 2

    def test_violin_collections(self, written, model):
        a = carve.pl.cluster_violin(written)
        b = model.plot_cluster_violin()
        assert len(a.collections) == len(b.collections) > 0
        pairs_compared = 0
        for ca, cb in zip(a.collections, b.collections):
            if isinstance(ca, PolyCollection) and isinstance(cb, PolyCollection):
                np.testing.assert_allclose(
                    ca.get_paths()[0].vertices, cb.get_paths()[0].vertices
                )
                pairs_compared += 1
        assert pairs_compared >= 2

    def test_scatter_offsets(self, written, model):
        a = carve.pl.cluster_scatter(written)
        b = model.plot_cluster_scatter(embedding=written.obsm["X_umap"])
        np.testing.assert_array_equal(
            a.collections[0].get_offsets(), b.collections[0].get_offsets()
        )

    def test_diagnostic_offsets_per_cluster(self, written, model):
        a = carve.pl.diagnostic_scatter(written)
        b = model.plot_diagnostic_scatter(embedding=written.obsm["X_umap"])
        assert len(a.collections) == len(b.collections)
        for ca, cb in zip(a.collections, b.collections):
            np.testing.assert_array_equal(ca.get_offsets(), cb.get_offsets())

    @pytest.mark.parametrize("fn", PL_FUNCTIONS, ids=lambda f: f.__name__)
    def test_save_writes_the_file_and_returns_none(self, written, fn, tmp_path):
        path = tmp_path / f"{fn.__name__}.png"
        assert fn(written, save=path) is None
        assert path.exists()


# -----------------------------------------------------------------------
# key= and the recorded selection
# -----------------------------------------------------------------------


class TestKeys:
    def test_non_default_key(self, model):
        adata = _attach(model, use_rep="X_pca", key_added="other")
        assert "other_stability" in adata.obs
        assert "other_consensus" in adata.obsp
        ax = carve.pl.cluster_boxplot(adata, key="other")
        assert len(ax.lines) > 0
        # Under the default key nothing was written: the score column is
        # the first thing cluster_boxplot reads, so that is what it reports.
        with pytest.raises(KeyError, match=r"adata.obs\['carve_stability'\] not found"):
            carve.pl.cluster_boxplot(adata)
        with pytest.raises(KeyError, match=r"adata.uns\['carve'\] not found"):
            carve.pl.metric_over_n_clusters(adata)

    def test_metric_defaults_to_the_recorded_measure(self, written):
        assert written.uns["carve"]["params"]["measure"] == "stability"
        assert "Stability" in carve.pl.metric_over_n_clusters(written).get_ylabel()
        assert (
            "Generalizability"
            in carve.pl.metric_over_n_clusters(
                written, measure="generalizability"
            ).get_ylabel()
        )

    def test_kwargs_reach_the_errorbar_line(self, written):
        # The data line is the ErrorbarContainer's first artist; the
        # selected-k marker is dashed on its own, so it cannot be the probe.
        dashed = carve.pl.metric_over_n_clusters(written, linestyle="--")
        assert dashed.containers[0][0].get_linestyle() == "--"
        plain = carve.pl.metric_over_n_clusters(written)
        assert plain.containers[0][0].get_linestyle() == "-"


# -----------------------------------------------------------------------
# Error paths in _entry, _labels, _scores, _results
# -----------------------------------------------------------------------


class TestErrors:
    def test_missing_uns_entry(self, written):
        with pytest.raises(KeyError, match=r"adata.uns\['nope'\] not found"):
            carve.pl.metric_over_n_clusters(written, key="nope")

    def test_malformed_uns_entry(self, written):
        bad = written.copy()
        bad.uns["carve"] = {"results": 1}
        with pytest.raises(KeyError, match="does not look like a CARVE result"):
            carve.pl.metric_over_n_clusters(bad)

    def test_missing_labels_column(self, written):
        bad = written.copy()
        del bad.obs["carve"]
        with pytest.raises(KeyError, match=r"adata.obs\['carve'\] not found"):
            carve.pl.consensus_matrix(bad)

    def test_bad_source(self, written):
        with pytest.raises(ValueError, match="source must be one of"):
            carve.pl.cluster_boxplot(written, source="nope")

    def test_missing_score_column_names_the_producing_mode(self, written):
        bad = written.copy()
        del bad.obs["carve_generalizability"]
        with pytest.raises(KeyError, match="source='accuracy' is unavailable"):
            carve.pl.cluster_boxplot(bad, source="accuracy")

    def test_store_results_false(self, model):
        adata = _attach(model, use_rep="X_pca", store_results=False)
        with pytest.raises(KeyError, match="store_results=False"):
            carve.pl.metric_over_n_clusters(adata)
        # The annotation needs the results table too, but a missing table
        # drops the annotation instead of failing the plot.
        ax = carve.pl.cluster_boxplot(adata)
        assert ax.get_legend() is None

    def test_store_consensus_false(self, model):
        adata = _attach(model, use_rep="X_pca", store_consensus=False)
        with pytest.raises(KeyError, match="store_consensus=False"):
            carve.pl.consensus_matrix(adata)

    def test_missing_basis(self, written):
        with pytest.raises(ValueError, match="basis='tsne' not found"):
            carve.pl.cluster_scatter(written, basis="tsne")


# -----------------------------------------------------------------------
# _annotation_text
# -----------------------------------------------------------------------


class TestAnnotation:
    def _legend_texts(self, ax):
        legend = ax.get_legend()
        return [] if legend is None else [t.get_text() for t in legend.get_texts()]

    def test_true_describes_the_recorded_selection(self, written):
        texts = self._legend_texts(carve.pl.cluster_boxplot(written, annotation=True))
        assert len(texts) == 1
        assert "KMeans, n_init=3" in texts[0]
        assert "Stability, 1-SE rule" in texts[0]

    def test_string_is_used_verbatim(self, written):
        assert self._legend_texts(
            carve.pl.cluster_boxplot(written, annotation="hello")
        ) == ["hello"]

    def test_false_draws_no_annotation(self, written):
        assert carve.pl.cluster_boxplot(written, annotation=False).get_legend() is None

    def test_pinned_k_is_reported_as_fixed(self, model):
        adata = _attach(model, use_rep="X_pca", k=3)
        assert adata.uns["carve"]["params"]["pinned"] is True
        texts = self._legend_texts(carve.pl.cluster_boxplot(adata))
        assert "(k = 3, fixed)" in texts[0]

    def test_unmatched_config_id_yields_no_annotation(self, model):
        adata = _attach(model, use_rep="X_pca")
        adata.uns["carve"]["params"]["selected_config_id"] = 99
        assert carve.pl.cluster_boxplot(adata).get_legend() is None

    def test_box_style_on_diagnostic_scatter(self, written):
        ax = carve.pl.diagnostic_scatter(
            written, annotation="note", annotation_style="box"
        )
        legends = [c for c in ax.get_children() if isinstance(c, Legend)]
        assert any(
            [t.get_text() for t in legend.get_texts()] == ["note"] for legend in legends
        )
        assert ax.get_legend().get_title().get_text() == "Cluster"


# -----------------------------------------------------------------------
# _representation: use_rep, layer and n_pcs are reconstructed from params
# -----------------------------------------------------------------------


class TestRepresentation:
    """With no embedding in obsm the scatter falls back to the representation
    the params record, so the drawn coordinates reveal which one was used.
    """

    def _no_obsm(self, model, **kwargs):
        adata = _attach(model, **kwargs)
        del adata.obsm["X_pca"]
        del adata.obsm["X_umap"]
        return adata

    def test_use_rep_x_with_two_pcs_draws_the_columns(self, model):
        adata = self._no_obsm(model, use_rep="X", n_pcs=2)
        ax = carve.pl.cluster_scatter(adata, sort_order=False)
        np.testing.assert_allclose(ax.collections[0].get_offsets(), adata.X[:, :2])

    def test_use_rep_x_with_three_pcs_is_reduced_by_pca(self, model):
        adata = self._no_obsm(model, use_rep="X", n_pcs=3)
        ax = carve.pl.cluster_scatter(adata, sort_order=False)
        expected = PCA(n_components=2, random_state=0).fit_transform(
            np.asarray(adata.X[:, :3], dtype=float)
        )
        np.testing.assert_allclose(ax.collections[0].get_offsets(), expected, rtol=1e-5)

    def test_layer_is_read_instead_of_x(self, model):
        adata = self._no_obsm(model, layer="counts")
        ax = carve.pl.cluster_scatter(adata, sort_order=False)
        expected = PCA(n_components=2, random_state=0).fit_transform(
            np.asarray(adata.layers["counts"], dtype=float)
        )
        np.testing.assert_allclose(ax.collections[0].get_offsets(), expected, rtol=1e-5)


# -----------------------------------------------------------------------
# metric_by_pipeline
# -----------------------------------------------------------------------


class TestMetricByPipeline:
    def test_matches_the_model_method_after_an_h5ad_round_trip(
        self, randomized_written, randomized_model
    ):
        a = carve.pl.metric_by_pipeline(randomized_written)
        b = randomized_model.plot_metric_by_pipeline(measure="stability", rule="1se")
        assert len(a.containers) == len(b.containers) == 2
        assert [c.get_label() for c in a.containers] == [
            c.get_label() for c in b.containers
        ]
        for ca, cb in zip(a.containers, b.containers):
            np.testing.assert_allclose(ca[0].get_xdata(), cb[0].get_xdata())
            np.testing.assert_allclose(ca[0].get_ydata(), cb[0].get_ydata())
        selected = randomized_model._select_row(measure="stability", rule="1se")[0]
        marked = [
            [line.get_xdata()[0] for line in ax.get_lines() if line.get_linestyle() == "--"]
            for ax in (a, b)
        ]
        assert marked[0] == marked[1] == [float(selected["sweep_value"])]

    def test_defaults_come_from_the_recorded_selection(
        self, randomized_written, monkeypatch
    ):
        # randomized_written is module-scoped and shared with other tests in
        # this class, so every mutation below goes through a fresh .copy().
        # The recorded params are set to non-default values here
        # (measure="generalizability", rule="max", not_two=True) so that a
        # pl.metric_by_pipeline that hard-coded "stability"/"1se"/False
        # instead of reading params could not pass this test by accident.
        seen = {}

        def spy(table, **kwargs):
            seen.update(kwargs, n_rows=len(table))

        monkeypatch.setattr(pl_plots, "_plot_metric_by_pipeline", spy)
        for recorded in ("m0", "m1"):
            adata = randomized_written.copy()
            adata.uns["carve"]["params"]["selected_method_id"] = recorded
            adata.uns["carve"]["params"]["measure"] = "generalizability"
            adata.uns["carve"]["params"]["rule"] = "max"
            adata.uns["carve"]["params"]["not_two"] = True
            carve.pl.metric_by_pipeline(adata)
            assert seen["method_id"] == recorded
            assert (seen["measure"], seen["rule"], seen["not_two"]) == (
                "generalizability",
                "max",
                True,
            )
        carve.pl.metric_by_pipeline(
            randomized_written, method_id="m1", measure="generalizability", rule="max"
        )
        assert (seen["method_id"], seen["measure"], seen["rule"]) == (
            "m1",
            "generalizability",
            "max",
        )
        assert seen["n_rows"] == len(randomized_written.uns["carve"]["preprocessing_results"])

    def test_recorded_measure_not_carried_falls_back_to_stability_with_a_warning(
        self, randomized_written, monkeypatch
    ):
        seen = {}

        def spy(table, **kwargs):
            seen.update(kwargs, n_rows=len(table))

        monkeypatch.setattr(pl_plots, "_plot_metric_by_pipeline", spy)
        adata = randomized_written.copy()
        adata.uns["carve"]["params"]["measure"] = "average"
        with pytest.warns(
            UserWarning,
            match=(
                r"tl\.carve recorded measure 'average', which the "
                r"per-pipeline table does not carry; plotting 'stability'\."
            ),
        ):
            carve.pl.metric_by_pipeline(adata)
        assert seen["measure"] == "stability"

    @pytest.mark.parametrize("case", ["not_randomized", "store_results_false"])
    def test_absent_table_names_both_reasons(self, model, randomized_model, case):
        if case == "not_randomized":
            adata = _attach(model, use_rep="X_pca")
        else:
            adata = _attach(randomized_model, use_rep="X_pca", store_results=False)
        with pytest.raises(
            KeyError,
            match=r"preprocessing_results'\] not found.*randomize_preprocessing=True.*store_results=False",
        ):
            carve.pl.metric_by_pipeline(adata)

    def test_save_writes_the_file_and_returns_none(self, randomized_written, tmp_path):
        path = tmp_path / "metric_by_pipeline.png"
        assert carve.pl.metric_by_pipeline(randomized_written, save=path) is None
        assert path.exists()
