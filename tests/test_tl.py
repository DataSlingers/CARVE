"""Tests for carve.tl -- the AnnData-native entry point."""

import inspect

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import carve
from carve import CARVE

KEYS = ("carve", "carve_stability", "carve_stability_ce", "carve_generalizability")


def _run(adata, **kwargs):
    kwargs.setdefault("use_rep", "X_pca")
    kwargs.setdefault("n_clusters", range(2, 5))
    kwargs.setdefault("n_resamples", 4)
    kwargs.setdefault("random_state", 0)
    return carve.tl.carve(adata, **kwargs)


class TestWrittenKeys:
    def test_all_obs_columns_written(self, adata_three_clusters):
        _run(adata_three_clusters)
        for key in KEYS:
            assert key in adata_three_clusters.obs, key

    def test_labels_are_categorical_strings(self, adata_three_clusters):
        _run(adata_three_clusters)
        col = adata_three_clusters.obs["carve"]
        assert isinstance(col.dtype, pd.CategoricalDtype)
        assert all(isinstance(c, str) for c in col.cat.categories)

    def test_categories_sorted_numerically(self, adata_three_clusters):
        """Ten or more clusters must not order as "1", "10", "2"."""
        _run(adata_three_clusters, n_clusters=range(2, 13), k=12)
        cats = list(adata_three_clusters.obs["carve"].cat.categories)
        assert cats == sorted(cats, key=int)
        if "10" in cats and "9" in cats:
            assert cats.index("10") > cats.index("9")

    def test_scores_are_float_and_full_length(self, adata_three_clusters):
        _run(adata_three_clusters)
        n = adata_three_clusters.n_obs
        for key in KEYS[1:]:
            col = adata_three_clusters.obs[key]
            assert len(col) == n
            assert np.issubdtype(col.to_numpy().dtype, np.floating)

    def test_consensus_matrix_shape(self, adata_three_clusters):
        _run(adata_three_clusters)
        n = adata_three_clusters.n_obs
        assert adata_three_clusters.obsp["carve_consensus"].shape == (n, n)

    def test_uns_structure(self, adata_three_clusters):
        _run(adata_three_clusters)
        entry = adata_three_clusters.uns["carve"]
        assert set(entry) == {"params", "results"}
        assert len(entry["results"]) > 0

    def test_params_have_no_none(self, adata_three_clusters):
        _run(adata_three_clusters)
        params = adata_three_clusters.uns["carve"]["params"]
        assert all(v is not None for v in params.values())

    def test_params_record_selection(self, adata_three_clusters):
        _run(adata_three_clusters, measure="generalizability", rule="max")
        params = adata_three_clusters.uns["carve"]["params"]
        assert params["measure"] == "generalizability"
        assert params["rule"] == "max"
        assert params["use_rep"] == "X_pca"
        assert params["n_obs"] == adata_three_clusters.n_obs

    def test_selected_k_matches_labels(self, adata_three_clusters):
        _run(adata_three_clusters)
        n_cats = len(adata_three_clusters.obs["carve"].cat.categories)
        assert adata_three_clusters.uns["carve"]["params"]["selected_k"] == n_cats


class TestCopySemantics:
    def test_copy_true_returns_new_object(self, adata_three_clusters):
        out = _run(adata_three_clusters, copy=True)
        assert isinstance(out, ad.AnnData)
        assert out is not adata_three_clusters

    def test_copy_true_leaves_original_untouched(self, adata_three_clusters):
        _run(adata_three_clusters, copy=True)
        assert "carve" not in adata_three_clusters.obs
        assert "carve" not in adata_three_clusters.uns

    def test_copy_false_returns_none_and_mutates(self, adata_three_clusters):
        assert _run(adata_three_clusters, copy=False) is None
        assert "carve" in adata_three_clusters.obs


class TestKeyAdded:
    def test_renames_every_key(self, adata_three_clusters):
        _run(adata_three_clusters, key_added="foo")
        assert "foo" in adata_three_clusters.obs
        assert "foo_stability" in adata_three_clusters.obs
        assert "foo_consensus" in adata_three_clusters.obsp
        assert "foo" in adata_three_clusters.uns
        assert "carve" not in adata_three_clusters.obs

    def test_two_runs_coexist(self, adata_three_clusters):
        _run(adata_three_clusters, key_added="a")
        _run(adata_three_clusters, key_added="b", measure="generalizability")
        assert "a" in adata_three_clusters.obs
        assert "b" in adata_three_clusters.obs
        assert adata_three_clusters.uns["a"]["params"]["measure"] == "stability"
        assert adata_three_clusters.uns["b"]["params"]["measure"] == "generalizability"


class TestOptionalOutputs:
    def test_store_consensus_false(self, adata_three_clusters):
        _run(adata_three_clusters, store_consensus=False)
        assert "carve_consensus" not in adata_three_clusters.obsp

    def test_store_results_false(self, adata_three_clusters):
        _run(adata_three_clusters, store_results=False)
        assert "results" not in adata_three_clusters.uns["carve"]
        assert "params" in adata_three_clusters.uns["carve"]

    @pytest.mark.parametrize(
        ("mode", "measure", "absent"),
        [
            ("stability", "stability", "carve_generalizability"),
            ("generalizability", "generalizability", "carve_stability"),
        ],
    )
    def test_partial_modes_omit_columns_without_raising(
        self, adata_three_clusters, mode, measure, absent
    ):
        """A skipped analysis drops its column rather than writing garbage.

        ``measure`` has to match ``mode``: a generalizability-only run has no
        stability metrics to select on, so pairing them is a user error rather
        than something tl.carve should paper over.
        """
        _run(adata_three_clusters, mode=mode, measure=measure)
        assert "carve" in adata_three_clusters.obs
        assert absent not in adata_three_clusters.obs

    def test_measure_without_matching_mode_raises(self, adata_three_clusters):
        """Selecting on a metric the run never computed must fail loudly."""
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            _run(
                adata_three_clusters,
                mode="generalizability",
                measure="stability",
            )


class TestReferenceKey:
    def test_accepts_obs_column(self, adata_three_clusters):
        _run(adata_three_clusters, reference_key="cell_type")
        assert "carve" in adata_three_clusters.obs

    def test_missing_column_raises(self, adata_three_clusters):
        with pytest.raises(ValueError, match="reference_key"):
            _run(adata_three_clusters, reference_key="nope")


class TestSweepModes:
    def test_pinned_k_is_honoured(self, adata_three_clusters):
        _run(adata_three_clusters, k=4)
        assert adata_three_clusters.uns["carve"]["params"]["selected_k"] == 4
        assert len(adata_three_clusters.obs["carve"].cat.categories) == 4

    def test_n_clusters_none_allows_resolution(self, adata_three_clusters):
        pytest.importorskip("igraph")
        pytest.importorskip("leidenalg")
        carve.tl.carve(
            adata_three_clusters,
            use_rep="X_pca",
            resolution=[0.5, 1.0],
            n_resamples=3,
            random_state=0,
        )
        params = adata_three_clusters.uns["carve"]["params"]
        assert params["sweep_param"] == "resolution"


class TestSignature:
    def test_covers_carve_constructor(self):
        """tl.carve must expose every CARVE knob, or users lose access to it."""
        model = set(inspect.signature(CARVE.__init__).parameters)
        tool = set(inspect.signature(carve.tl.carve).parameters)
        missing = model - tool - {"self", "reference_labels"}
        assert not missing, f"tl.carve is missing: {sorted(missing)}"

    def test_no_var_keyword(self):
        """**kwargs would defeat 'document every parameter'."""
        params = inspect.signature(carve.tl.carve).parameters.values()
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)

    def test_all_params_documented(self):
        doc = carve.tl.carve.__doc__
        for name in inspect.signature(carve.tl.carve).parameters:
            if name == "adata":
                continue
            assert f"{name} :" in doc, f"{name} is undocumented"


class TestAttachResults:
    def test_matches_tl_carve(self, adata_three_clusters):
        model = CARVE(n_clusters=np.arange(2, 5), n_resamples=4, random_state=0).fit(
            adata_three_clusters, use_rep="X_pca"
        )
        carve.tl.attach_results(
            adata_three_clusters, model, use_rep="X_pca", key_added="manual"
        )
        _run(adata_three_clusters, key_added="auto")
        assert (
            adata_three_clusters.obs["manual"].astype(str)
            == adata_three_clusters.obs["auto"].astype(str)
        ).all()

    def test_unfitted_model_raises(self, adata_three_clusters):
        with pytest.raises(RuntimeError, match="not fitted"):
            carve.tl.attach_results(adata_three_clusters, CARVE())

    def test_size_mismatch_raises(self, adata_three_clusters):
        model = CARVE(n_clusters=np.arange(2, 5), n_resamples=4, random_state=0).fit(
            adata_three_clusters, use_rep="X_pca"
        )
        smaller = ad.AnnData(
            np.zeros((5, 3), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="observations"):
            carve.tl.attach_results(smaller, model)


class TestAnchoringProvenance:
    """n_consensus_anchors is how a reader of a written h5ad learns that the
    consensus quantities in it are anchored rather than exact.

    The anchored runs here pass store_consensus=False. An anchored model's
    consensus matrix is m-by-m, and writing it into obsp, which is indexed by
    obs, is not currently supported.
    """

    def test_resolved_anchor_count_is_recorded_under_anchoring(
        self, adata_three_clusters
    ):
        with pytest.warns(RuntimeWarning):
            _run(adata_three_clusters, anchor_threshold=40, store_consensus=False)
        params = adata_three_clusters.uns["carve"]["params"]
        assert params["n_consensus_anchors"] == 40

    def test_explicit_anchor_count_is_recorded(self, adata_three_clusters):
        with pytest.warns(RuntimeWarning):
            _run(adata_three_clusters, consensus_anchors=25, store_consensus=False)
        params = adata_three_clusters.uns["carve"]["params"]
        assert params["n_consensus_anchors"] == 25

    def test_key_is_absent_on_the_exact_path(self, adata_three_clusters):
        # params_to_uns drops None, so an exact run must leave no key at all
        # rather than writing one that reads as an anchor count of nothing.
        _run(adata_three_clusters)
        params = adata_three_clusters.uns["carve"]["params"]
        assert "n_consensus_anchors" not in params

