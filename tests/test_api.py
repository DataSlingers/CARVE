"""Tests for CARVE public API (fit, get_labels, get_k, get_estimator, plotting, persistence)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.dummy import DummyClassifier
from sklearn.metrics import adjusted_rand_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
import warnings as _w

import carve._runner as carve_runner
import carve._utils as carve_utils
import carve.api as carve_api
from carve import CARVE, LeidenClustering, LouvainClustering
from carve._utils import resolve_anchors

from conftest import requires_graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_carve():
    """Return a small fitted CARVE instance (module-scoped for speed)."""
    rng = np.random.RandomState(42)
    X = np.vstack(
        [
            rng.randn(30, 5) + [3, 0, 0, 0, 0],
            rng.randn(30, 5) + [0, 3, 0, 0, 0],
        ]
    )
    carve = CARVE(
        n_clusters=2,
        n_resamples=5,
        subsample_ratio=0.8,
        estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
        normalization_options=[],
        dim_reduction_options=[],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    carve.fit(X)
    return carve


@pytest.fixture(scope="module")
def fitted_carve_multi_k():
    """Fitted CARVE with multiple k values and estimators.

    Uses AgglomerativeClustering with a linkage parameter so the results
    DataFrame has a non-metric column for the metric-over-k groupby.
    """
    rng = np.random.RandomState(42)
    X = np.vstack(
        [
            rng.randn(30, 5) + [4, 0, 0, 0, 0],
            rng.randn(30, 5) + [0, 4, 0, 0, 0],
            rng.randn(30, 5) + [0, 0, 4, 0, 0],
        ]
    )
    carve = CARVE(
        n_clusters=np.array([2, 3]),
        n_resamples=3,
        subsample_ratio=0.8,
        estimator_param_grids=[
            (AgglomerativeClustering, {"n_clusters": [2, 3], "linkage": ["ward"]}),
        ],
        normalization_options=[],
        dim_reduction_options=[],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    carve.fit(X)
    return carve


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestFit:
    def test_returns_self(self, X_two_clusters):
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            subsample_ratio=0.8,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            normalization_options=[],
            dim_reduction_options=[],
            verbose=0,
        )
        result = carve.fit(X_two_clusters)
        assert result is carve

    def test_estimator_results_populated(self, fitted_carve):
        assert fitted_carve.estimator_results_ is not None
        assert isinstance(fitted_carve.estimator_results_, pd.DataFrame)
        assert len(fitted_carve.estimator_results_) > 0

    def test_consensus_matrices_populated(self, fitted_carve):
        assert fitted_carve.consensus_matrices_ is not None
        assert len(fitted_carve.consensus_matrices_) > 0

    def test_stores_X(self, fitted_carve):
        assert fitted_carve.X_ is not None
        assert fitted_carve.X_.shape == (60, 5)

    def test_stability_scores_populated(self, fitted_carve):
        assert fitted_carve.stability_gini_scores_ is not None
        assert fitted_carve.stability_ce_scores_ is not None

    def test_generalizability_scores_populated(self, fitted_carve):
        assert fitted_carve.generalizability_scores_ is not None
        assert fitted_carve.generalizability_scores_ is not None

    def test_results_df_columns(self, fitted_carve):
        df = fitted_carve.estimator_results_
        expected = [
            "estimator",
            "n_clusters",
            "ari_stability",
            "ari_stability_se",
            "ari_generalizability",
            "ari_generalizability_se",
            "consensus_pac_stability",
            "consensus_gini_stability",
            "consensus_ce_stability",
            "accuracy_generalizability",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_multi_k(self, fitted_carve_multi_k):
        df = fitted_carve_multi_k.estimator_results_
        assert len(df) == 2
        assert set(df["n_clusters"]) == {2, 3}

    def test_stability_mode(self, X_two_clusters):
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            subsample_ratio=0.8,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            normalization_options=[],
            dim_reduction_options=[],
            verbose=0,
        )
        with pytest.warns(RuntimeWarning, match="experimental"):
            carve.fit(X_two_clusters, mode="stability")
        assert all(s is None for s in carve.generalizability_scores_)

    def test_generalizability_mode(self, X_two_clusters):
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            subsample_ratio=0.8,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            normalization_options=[],
            dim_reduction_options=[],
            verbose=0,
        )
        with pytest.warns(RuntimeWarning, match="experimental"):
            carve.fit(X_two_clusters, mode="generalizability")
        assert carve.stability_gini_scores_ is None
        assert carve.stability_ce_scores_ is None

    def test_reproducibility(self, X_two_clusters):
        def make_carve():
            return CARVE(
                n_clusters=2,
                n_resamples=3,
                subsample_ratio=0.8,
                estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
                normalization_options=[],
                dim_reduction_options=[],
                verbose=0,
                random_state=42,
            )

        c1 = make_carve().fit(X_two_clusters)
        c2 = make_carve().fit(X_two_clusters)
        assert (
            c1.estimator_results_["ari_stability"].iloc[0]
            == (c2.estimator_results_["ari_stability"].iloc[0])
        )

    def test_per_call_random_state(self, X_two_clusters):
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            subsample_ratio=0.8,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            normalization_options=[],
            dim_reduction_options=[],
            verbose=0,
        )
        carve.fit(X_two_clusters, random_state=99)
        assert carve.estimator_results_ is not None

    def test_dataframe_input(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame(rng.randn(60, 3), columns=["a", "b", "c"])
        df.iloc[:30] += [4, 0, 0]
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            normalization_options=[],
            dim_reduction_options=[],
            verbose=0,
        )
        carve.fit(df)
        assert carve.estimator_results_ is not None

    def test_inconsistent_n_clusters_raises(self, X_two_clusters):
        carve = CARVE(
            estimator_param_grids=[
                (KMeans, {"n_clusters": [2, 3]}),
                (KMeans, {"n_clusters": [4, 5]}),
            ],
            normalization_options=[],
            dim_reduction_options=[],
            verbose=0,
        )
        with pytest.raises(ValueError, match="same n_clusters"):
            carve.fit(X_two_clusters)

    def test_reference_labels_passed_at_fit(self, X_two_clusters):
        ref = np.array([0] * 30 + [1] * 30)
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            normalization_options=[],
            dim_reduction_options=[],
            verbose=0,
        )
        carve.fit(X_two_clusters, reference_labels=ref)
        np.testing.assert_array_equal(carve.reference_labels, ref)

    def test_string_reference_labels_factorized(self, X_two_clusters):
        ref = np.array(["A"] * 30 + ["B"] * 30)
        carve = CARVE(
            n_clusters=2,
            n_resamples=3,
            estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
            normalization_options=[],
            dim_reduction_options=[],
            verbose=0,
        )
        carve.fit(X_two_clusters, reference_labels=ref)
        assert np.issubdtype(carve.reference_labels.dtype, np.integer)


# ---------------------------------------------------------------------------
# get_labels()
# ---------------------------------------------------------------------------


class TestGetLabels:
    def test_basic(self, fitted_carve):
        labels = fitted_carve.get_labels()
        assert labels.shape == (60,)
        assert labels.dtype == np.int32

    def test_n_clusters_correct(self, fitted_carve):
        labels = fitted_carve.get_labels()
        assert len(np.unique(labels)) == 2

    def test_with_k(self, fitted_carve_multi_k):
        labels = fitted_carve_multi_k.get_labels(k=2)
        assert len(np.unique(labels)) == 2

    def test_with_k_3(self, fitted_carve_multi_k):
        labels = fitted_carve_multi_k.get_labels(k=3)
        assert len(np.unique(labels)) == 3

    def test_invalid_k_raises(self, fitted_carve):
        with pytest.raises(ValueError, match="No configurations"):
            fitted_carve.get_labels(k=99)

    def test_unfitted_raises(self):
        carve = CARVE(verbose=0)
        with pytest.raises(RuntimeError, match="fit"):
            carve.get_labels()

    def test_different_measures(self, fitted_carve):
        for measure in ["stability", "generalizability", "pac", "gini", "ce"]:
            labels = fitted_carve.get_labels(measure=measure)
            assert labels.shape == (60,)

    def test_different_rules(self, fitted_carve):
        for rule in ["max", "1se", "quantile"]:
            labels = fitted_carve.get_labels(rule=rule)
            assert labels.shape == (60,)

    def test_custom_estimator(self, fitted_carve):
        est = AgglomerativeClustering(
            n_clusters=2,
            linkage="complete",
            metric="precomputed",
        )
        labels = fitted_carve.get_labels(estimator=est)
        assert labels.shape == (60,)

    def test_reference_labels_alignment(self, fitted_carve):
        """Successive calls should produce consistent labels."""
        l1 = fitted_carve.get_labels()
        l2 = fitted_carve.get_labels()
        np.testing.assert_array_equal(l1, l2)


# ---------------------------------------------------------------------------
# get_k()
# ---------------------------------------------------------------------------


class TestGetK:
    def test_basic(self, fitted_carve):
        k = fitted_carve.get_k()
        assert isinstance(k, (int, np.integer))
        assert k == 2

    def test_unfitted_raises(self):
        carve = CARVE(verbose=0)
        with pytest.raises(RuntimeError, match="fit"):
            carve.get_k()

    def test_multi_k(self, fitted_carve_multi_k):
        k = fitted_carve_multi_k.get_k(measure="stability", rule="max")
        assert k in [2, 3]

    def test_different_measures(self, fitted_carve_multi_k):
        for measure in ["stability", "generalizability"]:
            k = fitted_carve_multi_k.get_k(measure=measure, rule="max")
            assert k in [2, 3]


# ---------------------------------------------------------------------------
# get_estimator()
# ---------------------------------------------------------------------------


class TestGetEstimator:
    def test_basic(self, fitted_carve):
        est = fitted_carve.get_estimator()
        assert isinstance(est, KMeans)
        assert est.n_clusters == 2

    def test_unfitted_raises(self):
        carve = CARVE(verbose=0)
        with pytest.raises(RuntimeError, match="fit"):
            carve.get_estimator()

    def test_multi_k(self, fitted_carve_multi_k):
        est = fitted_carve_multi_k.get_estimator()
        assert isinstance(est, AgglomerativeClustering)
        assert est.n_clusters in [2, 3]


# ---------------------------------------------------------------------------
# Plotting (smoke tests)
# ---------------------------------------------------------------------------


class TestPlotting:
    def test_plot_metric_over_n_clusters(self, fitted_carve_multi_k):
        ax = fitted_carve_multi_k.plot_metric_over_n_clusters(measure="stability")
        assert ax is not None

    def test_plot_metric_save(self, fitted_carve_multi_k, tmp_path):
        path = tmp_path / "metric.png"
        result = fitted_carve_multi_k.plot_metric_over_n_clusters(
            measure="stability",
            save=str(path),
        )
        assert result is None
        assert path.exists()

    def test_plot_metric_unfitted(self):
        carve = CARVE(verbose=0)
        with pytest.raises(RuntimeError, match="fit"):
            carve.plot_metric_over_n_clusters()

    def test_plot_consensus_matrix(self, fitted_carve):
        ax = fitted_carve.plot_consensus_matrix()
        assert ax is not None

    def test_plot_consensus_save(self, fitted_carve, tmp_path):
        path = tmp_path / "consensus.png"
        result = fitted_carve.plot_consensus_matrix(save=str(path))
        assert result is None
        assert path.exists()

    def test_plot_consensus_unfitted(self):
        carve = CARVE(verbose=0)
        with pytest.raises(RuntimeError, match="fit"):
            carve.plot_consensus_matrix()

    def test_plot_cluster_boxplot(self, fitted_carve):
        for source in ["gini", "ce", "accuracy"]:
            ax = fitted_carve.plot_cluster_boxplot(source=source)
            assert ax is not None

    def test_plot_cluster_boxplot_save(self, fitted_carve, tmp_path):
        path = tmp_path / "boxplot.png"
        result = fitted_carve.plot_cluster_boxplot(save=str(path))
        assert result is None
        assert path.exists()

    def test_plot_cluster_violin(self, fitted_carve):
        for source in ["gini", "ce", "accuracy"]:
            ax = fitted_carve.plot_cluster_violin(source=source)
            assert ax is not None

    def test_plot_cluster_violin_save(self, fitted_carve, tmp_path):
        path = tmp_path / "violin.png"
        result = fitted_carve.plot_cluster_violin(save=str(path))
        assert result is None
        assert path.exists()

    def test_plot_cluster_scatter(self, fitted_carve):
        ax = fitted_carve.plot_cluster_scatter()
        assert ax is not None

    def test_plot_cluster_scatter_save(self, fitted_carve, tmp_path):
        path = tmp_path / "scatter.png"
        result = fitted_carve.plot_cluster_scatter(save=str(path))
        assert result is None
        assert path.exists()

    def test_plot_scatter_with_embedding(self, fitted_carve):
        emb = np.random.RandomState(0).randn(60, 2)
        ax = fitted_carve.plot_cluster_scatter(embedding=emb)
        assert ax is not None


# ---------------------------------------------------------------------------
# save() guards
# ---------------------------------------------------------------------------


class TestSaveGuards:
    def test_save_unfitted_raises(self, tmp_path):
        carve = CARVE(verbose=0)
        with pytest.raises(RuntimeError, match="not been fitted"):
            carve.save(tmp_path / "should_not_exist.carve")

        assert not (tmp_path / "should_not_exist.carve").exists()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_roundtrip_without_data(self, fitted_carve, tmp_path):
        """save(include_data=False) -> load() preserves fitted results but not X_."""
        path = tmp_path / "no_data.carve"
        fitted_carve.save(path, include_data=False)

        loaded = CARVE.load(path)

        assert loaded.estimator_results_ is not None
        assert loaded.estimator_results_.equals(fitted_carve.estimator_results_)

        assert len(loaded.consensus_matrices_) == len(fitted_carve.consensus_matrices_)
        for orig, restored in zip(
            fitted_carve.consensus_matrices_, loaded.consensus_matrices_
        ):
            if orig is None:
                assert restored is None
            else:
                np.testing.assert_array_equal(orig, restored)

        assert loaded.X_ is None
        assert fitted_carve.X_ is not None

    def test_roundtrip_with_data(self, fitted_carve, tmp_path):
        path = tmp_path / "with_data.carve"
        fitted_carve.save(path, include_data=True)

        loaded = CARVE.load(path)
        np.testing.assert_array_equal(loaded.X_, fitted_carve.X_)

    def test_labels_match_after_load(self, fitted_carve, tmp_path):
        path = tmp_path / "labels.carve"
        fitted_carve.save(path)

        loaded = CARVE.load(path)
        original_labels = fitted_carve.get_labels()
        loaded_labels = loaded.get_labels()

        np.testing.assert_array_equal(original_labels, loaded_labels)

    def test_config_preserved(self, fitted_carve, tmp_path):
        path = tmp_path / "config.carve"
        fitted_carve.save(path)

        loaded = CARVE.load(path)
        assert loaded.n_clusters == fitted_carve.n_clusters
        assert loaded.n_resamples == fitted_carve.n_resamples
        assert loaded.subsample_ratio == fitted_carve.subsample_ratio
        assert loaded.random_state == fitted_carve.random_state

    def test_sample_level_scores_preserved(self, fitted_carve, tmp_path):
        path = tmp_path / "scores.carve"
        fitted_carve.save(path)
        loaded = CARVE.load(path)

        if fitted_carve.stability_gini_scores_ is not None:
            np.testing.assert_array_equal(
                loaded.stability_gini_scores_, fitted_carve.stability_gini_scores_
            )
        if fitted_carve.stability_ce_scores_ is not None:
            np.testing.assert_array_equal(
                loaded.stability_ce_scores_, fitted_carve.stability_ce_scores_
            )
        if fitted_carve.generalizability_scores_ is not None:
            for orig, loaded_arr in zip(
                fitted_carve.generalizability_scores_,
                loaded.generalizability_scores_,
            ):
                np.testing.assert_array_equal(loaded_arr, orig)


# ---------------------------------------------------------------------------
# load() guards
# ---------------------------------------------------------------------------


class TestLoadGuards:
    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CARVE.load(tmp_path / "no_such_file.carve")

    def test_load_non_carve_object_raises(self, tmp_path):
        import joblib

        path = tmp_path / "not_carve.carve"
        joblib.dump({"hello": "world"}, path)

        with pytest.raises(TypeError, match="Expected a CARVE instance"):
            CARVE.load(path)


# ---------------------------------------------------------------------------
# Compression & file-size
# ---------------------------------------------------------------------------


class TestCompression:
    def test_compressed_smaller_than_uncompressed(self, fitted_carve, tmp_path):
        compressed = tmp_path / "compressed.carve"
        uncompressed = tmp_path / "uncompressed.carve"

        fitted_carve.save(compressed, compress=3)
        fitted_carve.save(uncompressed, compress=0)

        assert compressed.stat().st_size < uncompressed.stat().st_size

    def test_no_data_smaller_than_with_data(self, fitted_carve, tmp_path):
        no_data = tmp_path / "no_data.carve"
        with_data = tmp_path / "with_data.carve"

        fitted_carve.save(no_data, include_data=False)
        fitted_carve.save(with_data, include_data=True)

        assert no_data.stat().st_size < with_data.stat().st_size


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_save_creates_parent_dirs(self, fitted_carve, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "result.carve"
        fitted_carve.save(nested)

        assert nested.exists()
        loaded = CARVE.load(nested)
        assert loaded.estimator_results_ is not None


# ---------------------------------------------------------------------------
# Resolution mode
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def X_res_blobs():
    """Three well-separated blobs, large enough for a 15-neighbor graph."""
    rng = np.random.RandomState(0)
    return np.vstack([rng.randn(40, 6) + c for c in np.eye(6)[:3] * 6])


@pytest.fixture(scope="module")
def fitted_resolution(X_res_blobs):
    """CARVE fitted in resolution mode (Leiden + Louvain defaults)."""
    carve = CARVE(
        resolution=np.array([0.5, 1.0, 2.0]),
        n_resamples=3,
        normalization_options=[],
        dim_reduction_options=[],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    carve.fit(X_res_blobs)
    return carve


@requires_graph
class TestResolutionMode:
    def test_sweep_spec_recorded(self, fitted_resolution):
        sweep = fitted_resolution.sweep_
        assert sweep.param == "resolution"
        assert not sweep.is_k_mode
        assert not sweep.fixes_k
        assert sweep.finer_is_larger
        assert sweep.label == "Resolution"
        np.testing.assert_allclose(sweep.values, [0.5, 1.0, 2.0])

    def test_results_carry_sweep_columns(self, fitted_resolution):
        df = fitted_resolution.estimator_results_
        assert set(df.columns) >= {
            "config_id",
            "method_id",
            "method_label",
            "sweep_param",
            "sweep_value",
            "sweep_rank",
            "n_clusters_observed",
            "n_clusters_observed_se",
            "noise_fraction",
            "resolution",
        }

    def test_no_n_clusters_column(self, fitted_resolution):
        assert "n_clusters" not in fitted_resolution.estimator_results_.columns

    def test_sweep_values_and_ranks(self, fitted_resolution):
        df = fitted_resolution.estimator_results_
        assert set(df["sweep_param"]) == {"resolution"}
        np.testing.assert_allclose(sorted(df["sweep_value"].unique()), [0.5, 1.0, 2.0])
        # Two estimators x three resolutions.
        assert df.shape[0] == 6
        assert sorted(df["sweep_rank"].unique()) == [0, 1, 2]

    def test_estimators_are_graph_methods(self, fitted_resolution):
        assert set(fitted_resolution.estimator_results_["estimator"]) == {
            "LeidenClustering",
            "LouvainClustering",
        }

    def test_noise_fraction_is_zero(self, fitted_resolution):
        """Graph community methods assign every point."""
        assert (fitted_resolution.estimator_results_["noise_fraction"] == 0).all()

    def test_get_sweep_value(self, fitted_resolution):
        value = fitted_resolution.get_sweep_value()
        assert value in [0.5, 1.0, 2.0]

    def test_get_k_uses_observed_clusters(self, fitted_resolution):
        assert fitted_resolution.get_k() >= 2

    def test_get_labels_shape(self, fitted_resolution, X_res_blobs):
        labels = fitted_resolution.get_labels()
        assert labels.shape == (X_res_blobs.shape[0],)
        assert np.unique(labels).size == fitted_resolution.get_k()

    def test_get_labels_rejects_k(self, fitted_resolution):
        with pytest.raises(ValueError, match="sweep_value"):
            fitted_resolution.get_labels(k=3)

    def test_get_labels_accepts_sweep_value(self, fitted_resolution, X_res_blobs):
        labels = fitted_resolution.get_labels(sweep_value=1.0)
        assert labels.shape == (X_res_blobs.shape[0],)

    def test_get_labels_unknown_sweep_value_raises(self, fitted_resolution):
        with pytest.raises(ValueError, match="No configurations found"):
            fitted_resolution.get_labels(sweep_value=99.0)

    def test_get_labels_rejects_both_pins(self, fitted_resolution):
        with pytest.raises(ValueError, match="at most one of"):
            fitted_resolution.get_labels(k=3, sweep_value=1.0)

    def test_consensus_k_overrides_the_cut(self, fitted_resolution):
        labels = fitted_resolution.get_labels(consensus_k=4)
        assert np.unique(labels).size == 4

    def test_get_estimator_is_a_graph_method(self, fitted_resolution):
        est = fitted_resolution.get_estimator()
        assert isinstance(est, (LeidenClustering, LouvainClustering))
        assert est.resolution == fitted_resolution.get_sweep_value()

    def test_plot_xlabel_is_resolution(self, fitted_resolution):
        ax = fitted_resolution.plot_metric_over_n_clusters()
        assert ax.get_xlabel() == "Resolution"

    def test_plot_draws_one_curve_per_method(self, fitted_resolution):
        ax = fitted_resolution.plot_metric_over_n_clusters()
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        method_labels = set(fitted_resolution.estimator_results_["method_label"])
        assert method_labels <= set(texts)
        assert len(method_labels) == 2

    def test_plot_consensus_matrix_accepts_sweep_value(self, fitted_resolution):
        ax = fitted_resolution.plot_consensus_matrix(sweep_value=1.0)
        assert ax is not None

    def test_save_load_round_trips_sweep(self, fitted_resolution, tmp_path):
        path = tmp_path / "res.carve"
        fitted_resolution.save(path)
        loaded = CARVE.load(path)
        # SweepSpec holds an ndarray, so compare field-by-field rather than
        # relying on dataclass equality.
        original = fitted_resolution.sweep_
        assert loaded.sweep_.param == original.param
        assert loaded.sweep_.label == original.label
        assert loaded.sweep_.fixes_k == original.fixes_k
        assert loaded.sweep_.finer_is_larger == original.finer_is_larger
        np.testing.assert_allclose(loaded.sweep_.values, original.values)
        assert loaded.get_sweep_value() == fitted_resolution.get_sweep_value()

    def test_inferred_from_custom_grids(self, X_res_blobs):
        """Supplying resolution grids alone switches the run to that axis."""
        carve = CARVE(
            estimator_param_grids=[
                (LeidenClustering, {"resolution": [0.5, 1.0], "n_neighbors": [15]})
            ],
            n_resamples=2,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=0,
            verbose=0,
        ).fit(X_res_blobs)
        assert carve.sweep_.param == "resolution"

    def test_mixing_sweep_axes_raises(self, X_res_blobs):
        carve = CARVE(
            estimator_param_grids=[
                (KMeans, {"n_clusters": [2, 3]}),
                (LeidenClustering, {"resolution": [0.5, 1.0]}),
            ],
            n_resamples=2,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            verbose=0,
        )
        with pytest.raises(ValueError, match="one sweep axis"):
            carve.fit(X_res_blobs)


# ---------------------------------------------------------------------------
# min_cluster_size / HDBSCAN mode
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_hdbscan(X_res_blobs):
    """CARVE fitted over HDBSCAN's min_cluster_size axis."""
    carve = CARVE(
        sweep="min_cluster_size",
        sweep_values=np.array([3, 5, 8]),
        n_resamples=3,
        normalization_options=[],
        dim_reduction_options=[],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    carve.fit(X_res_blobs)
    return carve


class TestMinClusterSizeMode:
    def test_sweep_spec_is_inverted(self, fitted_hdbscan):
        sweep = fitted_hdbscan.sweep_
        assert sweep.param == "min_cluster_size"
        assert not sweep.finer_is_larger
        assert not sweep.fixes_k
        assert sweep.label == "Minimum Cluster Size"

    def test_uses_hdbscan(self, fitted_hdbscan):
        assert set(fitted_hdbscan.estimator_results_["estimator"]) == {"HDBSCAN"}

    def test_ranks_run_backwards(self, fitted_hdbscan):
        df = fitted_hdbscan.estimator_results_.sort_values("sweep_value")
        assert df["sweep_rank"].tolist() == [2, 1, 0]

    def test_records_noise(self, fitted_hdbscan):
        """HDBSCAN leaves points unassigned; the runner must record that."""
        assert (fitted_hdbscan.estimator_results_["noise_fraction"] > 0).any()
        assert (fitted_hdbscan.estimator_results_["noise_fraction"] <= 1).all()

    def test_observed_k_is_recorded(self, fitted_hdbscan):
        assert (fitted_hdbscan.estimator_results_["n_clusters_observed"] > 0).all()

    def test_get_sweep_value_and_k(self, fitted_hdbscan):
        assert fitted_hdbscan.get_sweep_value() in [3, 5, 8]
        assert fitted_hdbscan.get_k() >= 2

    def test_get_labels(self, fitted_hdbscan, X_res_blobs):
        labels = fitted_hdbscan.get_labels()
        assert labels.shape == (X_res_blobs.shape[0],)

    def test_plot_xlabel(self, fitted_hdbscan):
        ax = fitted_hdbscan.plot_metric_over_n_clusters()
        assert ax.get_xlabel() == "Minimum Cluster Size"

    def _fit(self, X, **kwargs):
        return CARVE(
            sweep="min_cluster_size",
            sweep_values=np.array([5, 10, 20]),
            n_resamples=3,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=0,
            verbose=0,
            **kwargs,
        ).fit(X)

    def test_all_noise_resample_does_not_crash(self, X_res_blobs):
        """Regression: a held-out split labelled entirely as noise.

        Under noise_policy="drop" such a resample yields no predicted
        labels at all. Those resamples must be skipped when aggregating,
        not passed on to the consensus and accuracy machinery.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            carve = self._fit(X_res_blobs)
        assert carve.estimator_results_.shape[0] == 3
        assert carve.consensus_matrices_[0] is not None

    def test_noise_fraction_agrees_across_policies(self, X_res_blobs):
        """noise_fraction is measured before the policy is applied."""
        fractions = {}
        for policy in ("drop", "as_cluster", "singleton"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                carve = self._fit(X_res_blobs, noise_policy=policy)
            fractions[policy] = carve.estimator_results_["noise_fraction"].tolist()

        assert fractions["drop"] == pytest.approx(fractions["as_cluster"])
        assert fractions["drop"] == pytest.approx(fractions["singleton"])

    def test_non_integer_sweep_values_rejected(self, X_res_blobs):
        carve = CARVE(
            sweep="min_cluster_size",
            sweep_values=np.array([5.5, 10.0]),
            n_resamples=2,
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            verbose=0,
        )
        with pytest.raises(TypeError, match="must be integers"):
            carve.fit(X_res_blobs)


# ---------------------------------------------------------------------------
# Row identity: artifacts are joined on config_id, never on row position
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_identity():
    """A multi-row k-mode run with more than one method."""
    rng = np.random.RandomState(42)
    X = np.vstack(
        [
            rng.randn(30, 5) + [6, 0, 0, 0, 0],
            rng.randn(30, 5) + [0, 6, 0, 0, 0],
            rng.randn(30, 5) + [0, 0, 6, 0, 0],
        ]
    )
    carve = CARVE(
        n_clusters=np.array([2, 3, 4]),
        n_resamples=4,
        estimator_param_grids=[
            (KMeans, {"n_clusters": [2, 3, 4]}),
            (
                AgglomerativeClustering,
                {"n_clusters": [2, 3, 4], "linkage": ["ward", "average"]},
            ),
        ],
        normalization_options=[],
        dim_reduction_options=[],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    carve.fit(X)
    return carve


@pytest.fixture(scope="module")
def fitted_identity_single():
    """A single-method run, so rule-based selection is unambiguous.

    ``select_best_row_1se`` breaks ties with ``idxmax``, which follows the
    frame's current row order. With several methods the top metric can tie
    across them, and a shuffle would then legitimately select a different
    row -- masking, rather than testing, the config_id join. One method
    over three k values gives exactly one qualifying row.
    """
    rng = np.random.RandomState(42)
    X = np.vstack(
        [
            rng.randn(30, 5) + [6, 0, 0, 0, 0],
            rng.randn(30, 5) + [0, 6, 0, 0, 0],
            rng.randn(30, 5) + [0, 0, 6, 0, 0],
        ]
    )
    carve = CARVE(
        n_clusters=np.array([2, 3, 4]),
        n_resamples=4,
        estimator_param_grids=[(KMeans, {"n_clusters": [2, 3, 4]})],
        normalization_options=[],
        dim_reduction_options=[],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    carve.fit(X)
    return carve


class TestRowIdentity:
    def test_selection_is_unambiguous(self, fitted_identity_single):
        """Guards the premise of the shuffle tests below."""
        df = fitted_identity_single.estimator_results_
        best = df["ari_stability"].max()
        threshold = best - df.loc[df["ari_stability"].idxmax(), "ari_stability_se"]
        assert (df["ari_stability"] >= threshold).sum() == 1
        # ...and the selected row is not at position 0, so a label/position
        # mix-up would actually show up.
        assert fitted_identity_single.get_k() == 3

    def test_config_id_indexes_the_artifact_containers(self, fitted_identity):
        config_ids = fitted_identity.estimator_results_["config_id"].tolist()
        assert config_ids == list(range(len(fitted_identity.consensus_matrices_)))
        assert config_ids == list(range(len(fitted_identity.generalizability_scores_)))
        assert config_ids == list(range(len(fitted_identity.stability_gini_scores_)))

    @staticmethod
    def _reindexed(df):
        """Reorder rows AND relabel the index.

        ``sample(frac=1)`` alone preserves index labels, so the old
        ``int(row.name)`` lookup would still land on the right artifact and
        the test would prove nothing. Dropping the index is what actually
        divorces label from position.
        """
        return df.sort_values("ari_stability").reset_index(drop=True)

    def test_reindexing_actually_divorces_label_from_position(
        self, fitted_identity_single
    ):
        """Non-vacuity guard for the three tests below.

        If the selected row's index label still equalled its config_id,
        those tests would pass even with the bug reinstated.
        """
        reindexed = self._reindexed(fitted_identity_single.estimator_results_)
        row, config_id, _, _ = fitted_identity_single._select_row(
            measure="stability", rule="1se"
        )
        selected = reindexed[reindexed["config_id"] == config_id]
        assert len(selected) == 1
        assert int(selected.index[0]) != config_id

    def test_select_row_joins_on_config_id_not_index_label(
        self, fitted_identity_single
    ):
        """The regression this whole change exists for.

        Before config_id, artifact lookups used the row's index *label* as
        a list position, so any reordering silently returned the wrong
        consensus matrix -- with no error, and plausible-looking output.
        """
        _, before_id, _, _ = fitted_identity_single._select_row(
            measure="stability", rule="1se"
        )
        original = fitted_identity_single.estimator_results_
        try:
            fitted_identity_single.estimator_results_ = self._reindexed(original)
            row, after_id, _, _ = fitted_identity_single._select_row(
                measure="stability", rule="1se"
            )
            assert after_id == before_id
            # The label moved; the join key did not.
            assert int(row.name) != after_id
        finally:
            fitted_identity_single.estimator_results_ = original

    def test_consensus_matrix_survives_reindexing(self, fitted_identity_single):
        before = fitted_identity_single.plot_consensus_matrix()
        before_data = before.images[0].get_array().copy()
        plt.close("all")

        original = fitted_identity_single.estimator_results_
        try:
            fitted_identity_single.estimator_results_ = self._reindexed(original)
            after = fitted_identity_single.plot_consensus_matrix()
            np.testing.assert_array_equal(after.images[0].get_array(), before_data)
        finally:
            fitted_identity_single.estimator_results_ = original
            plt.close("all")

    def test_labels_survive_reindexing(self, fitted_identity_single):
        before = fitted_identity_single.get_labels()
        original = fitted_identity_single.estimator_results_
        try:
            fitted_identity_single.estimator_results_ = self._reindexed(original)
            np.testing.assert_array_equal(fitted_identity_single.get_labels(), before)
        finally:
            fitted_identity_single.estimator_results_ = original

    def test_labels_survive_row_reordering(self, fitted_identity_single):
        """Reordering without relabelling must also be safe."""
        before = fitted_identity_single.get_labels()
        original = fitted_identity_single.estimator_results_
        try:
            fitted_identity_single.estimator_results_ = original.sample(
                frac=1, random_state=0
            )
            np.testing.assert_array_equal(fitted_identity_single.get_labels(), before)
        finally:
            fitted_identity_single.estimator_results_ = original

    def test_get_estimator_ignores_identity_columns(self, fitted_identity_single):
        """row_to_estimator_params must filter the bookkeeping columns out."""
        est = fitted_identity_single.get_estimator()
        params = est.get_params()
        for column in (
            "config_id",
            "method_id",
            "method_label",
            "sweep_param",
            "sweep_value",
            "sweep_rank",
            "n_clusters_observed",
            "noise_fraction",
        ):
            assert column not in params

    def test_get_estimator_survives_a_shuffle(self, fitted_identity_single):
        before = fitted_identity_single.get_estimator().get_params()
        original = fitted_identity_single.estimator_results_
        try:
            fitted_identity_single.estimator_results_ = original.sample(
                frac=1, random_state=1
            )
            assert fitted_identity_single.get_estimator().get_params() == before
        finally:
            fitted_identity_single.estimator_results_ = original

    def test_one_method_id_per_curve(self, fitted_identity):
        sizes = fitted_identity.estimator_results_.groupby("method_id")[
            "sweep_value"
        ].size()
        # KMeans + ward + average = 3 curves, 3 swept values each.
        assert len(sizes) == 3
        assert set(sizes) == {3}

    def test_method_label_is_unique_per_method_id(self, fitted_identity):
        pairs = fitted_identity.estimator_results_[
            ["method_id", "method_label"]
        ].drop_duplicates()
        assert len(pairs) == pairs["method_id"].nunique()

    def test_misaligned_config_id_raises(self, X_two_clusters):
        """The fit()-time guard is the last line of defence."""
        carve = CARVE(
            n_clusters=np.array([2, 3]),
            n_resamples=2,
            estimator_param_grids=[(KMeans, {"n_clusters": [2, 3]})],
            normalization_options=[],
            dim_reduction_options=[],
            n_jobs=1,
            random_state=0,
            verbose=0,
        )
        original = carve_runner.run_validation

        def _corrupt(*args, **kwargs):
            records, *rest = original(*args, **kwargs)
            for record in records:
                record["config_id"] = 0
            return (records, *rest)

        carve_runner.run_validation = _corrupt
        carve_api.run_validation = _corrupt
        try:
            with pytest.raises(RuntimeError, match="config_id is misaligned"):
                carve.fit(X_two_clusters)
        finally:
            carve_runner.run_validation = original
            carve_api.run_validation = original


def _blobs(n, seed=0, p=4):
    rng = np.random.default_rng(seed)
    half = n // 2
    return np.vstack([rng.normal(0, 1, (half, p)), rng.normal(6, 1, (n - half, p))])


def _grids():
    return [(KMeans, {"n_clusters": [2, 3], "n_init": [10]})]


def _overlapping_blobs(n, seed=0, p=4, sep=1.2):
    """Two Gaussians close enough that borderline points are genuinely ambiguous.

    The well separated ``_blobs`` fixture is classified identically by any
    random forest, seeded or not, so it cannot detect an unseeded extension.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    return np.vstack([rng.normal(0, 1, (half, p)), rng.normal(sep, 1, (n - half, p))])


class _SeedSpy(BaseEstimator, ClassifierMixin):
    """Records the ``random_state`` CARVE injects, then predicts a constant."""

    seen: list = []

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        type(self).seen.append(self.random_state)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.classes_[0])


class _NJobsSpy(BaseEstimator, ClassifierMixin):
    """Records the ``n_jobs`` CARVE injects, then predicts a constant."""

    seen: list = []

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def fit(self, X, y):
        type(self).seen.append(self.n_jobs)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.classes_[0])


class TestAnchoredConsensus:
    def test_below_threshold_stores_no_anchors_and_full_matrices(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=100,
        ).fit(X)
        assert c.consensus_anchors_ is None
        assert c.consensus_matrices_[0].shape == (60, 60)

    def test_above_threshold_stores_anchors_and_block_matrices(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning, match="anchored consensus"):
            c.fit(X)
        assert c.consensus_anchors_ is not None
        assert c.consensus_anchors_.size == 30
        assert c.consensus_matrices_[0].shape == (30, 30)

    def test_per_sample_scores_stay_full_length_under_anchoring(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)
        assert c.stability_gini_scores_.shape[1] == 60
        assert c.stability_ce_scores_.shape[1] == 60
        assert c.generalizability_scores_[0].shape == (60,)

    def test_explicit_anchor_count_opts_in_below_threshold(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=1000,
            consensus_anchors=25,
        )
        # n=60 is far below anchor_threshold=1000, so a message blaming the
        # threshold would be false. The opt-in is what engaged anchoring.
        with pytest.warns(
            RuntimeWarning,
            match=r"consensus_anchors=25 opts this run in regardless of "
            r"anchor_threshold, so CARVE is using anchored consensus over 25 "
            r"anchors",
        ):
            c.fit(X)
        assert c.consensus_anchors_.size == 25

    def test_threshold_triggered_anchoring_names_the_threshold(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=30,
        )
        with pytest.warns(
            RuntimeWarning,
            match=r"n=60 exceeds anchor_threshold=30, so CARVE is using "
            r"anchored consensus over 30 anchors",
        ):
            c.fit(X)

    def test_config_id_alignment_holds_under_anchoring(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)
        n_rows = c.estimator_results_.shape[0]
        assert np.array_equal(
            c.estimator_results_["config_id"].to_numpy(), np.arange(n_rows)
        )
        assert len(c.consensus_matrices_) == n_rows
        assert c.stability_gini_scores_.shape[0] == n_rows

    def test_every_configuration_indexes_the_same_anchors(self):
        # Configurations must be comparable across k, so one draw is reused.
        # If each config drew its own anchors, blocks at different k would
        # index different samples and cross-k comparison would be meaningless.
        # Two configs are fitted here, so identical off-diagonal NaN masks are
        # evidence they were built over the same co-sampled pairs.
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)
        assert c.consensus_anchors_.ndim == 1
        assert np.all(np.diff(c.consensus_anchors_) > 0)
        assert len(c.consensus_matrices_) >= 2
        first, second = c.consensus_matrices_[0], c.consensus_matrices_[1]
        assert first.shape == second.shape == (30, 30)
        assert np.array_equal(np.isnan(first), np.isnan(second))


class TestAnchoredLabels:
    def _fitted(self, n=60, threshold=30):
        X = _blobs(n)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=threshold,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)
        return X, c

    def test_labels_cover_every_sample(self):
        X, c = self._fitted()
        labels = c.get_labels(k=2)
        assert labels.shape == (X.shape[0],)
        assert labels.dtype == np.int32

    def test_labels_recover_the_planted_structure(self):
        # _blobs plants two well separated groups, so an extension that
        # assigned arbitrary labels would fail this even with a valid shape.
        X, c = self._fitted()
        truth = np.repeat([0, 1], X.shape[0] // 2)
        assert adjusted_rand_score(truth, c.get_labels(k=2)) > 0.9

    def test_anchor_positions_keep_their_cut_labels(self):
        X, c = self._fitted()
        anchors = c.consensus_anchors_
        planted = np.arange(anchors.size) % 2

        # A RandomForest fitted on the anchors and asked to predict those
        # same anchors tends to reproduce their labels regardless of whether
        # the implementation wrongly overwrote them, so whether it memorizes
        # is a probabilistic margin rather than a guarantee. That would make
        # the assertion below an unreliable check for the one defect this
        # test exists to catch. A constant classifier cannot reproduce an
        # alternating cut, so an overwrite is visible, and the mutation kill
        # is exact and deterministic.
        c.classifier = DummyClassifier(strategy="constant", constant=0)

        extended = c._extend_anchor_labels(planted)
        assert np.array_equal(extended[anchors], planted)
        assert extended.shape == (X.shape[0],)

    def test_number_of_clusters_matches_the_cut(self):
        X, c = self._fitted()
        assert np.unique(c.get_labels(k=3)).size == 3

    def test_generalizability_mode_also_covers_every_sample(self):
        # get_labels serves both mode="default" (stability) and
        # mode="generalizability"; both matrix lists are anchor blocks
        # under anchoring, so the single guarded call must cover both.
        X, c = self._fitted()
        labels = c.get_labels(k=2, mode="generalizability")
        assert labels.shape == (X.shape[0],)

    def test_default_random_state_still_gives_deterministic_labels(self):
        # Below the threshold get_labels is an agglomerative cut on a fixed
        # matrix, so it is exactly deterministic. Anchoring must not weaken
        # that: with random_state left at its default of None the extension
        # classifier still has to be seeded.
        X = _overlapping_blobs(120)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            anchor_threshold=40,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)
        assert c.consensus_anchors_ is not None

        first = c.get_labels(k=2)
        second = c.get_labels(k=2)
        assert np.array_equal(first, second)

    def test_fit_time_random_state_is_honored_by_the_extension(self):
        # fit(X, random_state=7) is honored by the anchor draw and by
        # run_validation; the extension must read the same resolved seed
        # rather than self.random_state, which fit() never writes.
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X, random_state=7)

        c.classifier = _SeedSpy()
        _SeedSpy.seen.clear()
        try:
            c._extend_anchor_labels(np.arange(c.consensus_anchors_.size) % 2)
        finally:
            recorded = list(_SeedSpy.seen)
            _SeedSpy.seen.clear()

        assert recorded == [7]

    def test_extension_fit_receives_the_whole_core_budget(self, monkeypatch):
        # fit() spreads n_jobs=4 over 4 workers x 2 threads on 11 cores. The
        # extension is one fit outside that loop, so it gets the product, not
        # the per-worker share.
        monkeypatch.setattr(carve_utils, "cpu_count", lambda: 11)
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            anchor_threshold=30,
            n_jobs=4,
            random_state=0,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)

        c.classifier = _NJobsSpy()
        _NJobsSpy.seen.clear()
        try:
            c._extend_anchor_labels(np.arange(c.consensus_anchors_.size) % 2)
        finally:
            recorded = list(_NJobsSpy.seen)
            _NJobsSpy.seen.clear()

        assert recorded == [8]

    def test_default_save_leaves_an_anchored_model_unable_to_label(self, tmp_path):
        # save() drops X_ by default, and the extension needs it. The exact
        # path does not, so this interaction only exists under anchoring and
        # has to surface as a clear error rather than a crash.
        X, c = self._fitted()
        path = tmp_path / "anchored.carve"
        c.save(path)

        loaded = CARVE.load(path)
        assert loaded.consensus_anchors_ is not None
        assert loaded.X_ is None

        with pytest.raises(RuntimeError, match="restore X_ after load"):
            loaded.get_labels(k=2)

        # Restoring X_ makes it work again, as the docstring promises.
        loaded.X_ = X
        assert loaded.get_labels(k=2).shape == (X.shape[0],)

    def test_saving_with_data_keeps_an_anchored_model_able_to_label(self, tmp_path):
        X, c = self._fitted()
        path = tmp_path / "anchored_with_data.carve"
        c.save(path, include_data=True)

        loaded = CARVE.load(path)
        assert loaded.get_labels(k=2).shape == (X.shape[0],)

    def test_exact_path_labels_are_unaffected(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=1000,
        ).fit(X)
        assert c.consensus_anchors_ is None
        assert c.get_labels(k=2).shape == (60,)


class TestExactPathUnchanged:
    def _fit(self, n, **kwargs):
        X = _blobs(n, seed=3)
        return X, CARVE(
            estimator_param_grids=_grids(),
            n_resamples=8,
            random_state=11,
            **kwargs,
        ).fit(X)

    def test_default_threshold_keeps_five_thousand_exact(self):
        # Not a fit at n=5000, which is slow; assert the resolution rule that
        # governs it, which is what the promise actually rests on.
        assert resolve_anchors(
            5000, consensus_anchors=None, anchor_threshold=5000, random_state=42
        ) is None

    def test_defaults_do_not_engage_anchoring_at_small_n(self):
        _, c = self._fit(80)
        assert c.consensus_anchors_ is None
        assert c.consensus_matrices_[0].shape == (80, 80)

    def test_results_are_identical_with_and_without_the_feature_present(self):
        # Two fits differing only in anchor_threshold, neither of which
        # anchors. The first sets anchor_threshold=80, exactly n, so it
        # exercises the inclusive n == anchor_threshold boundary through
        # fit() itself rather than only through a resolve_anchors unit call;
        # the second sets a threshold far above n as a plain control. Both
        # are expected to take the exact path and agree on every value below.
        # This does not detect an exclusive-comparison mutation of
        # resolve_anchors: at m == n the anchored computation reproduces the
        # exact one exactly, so no threshold pair at this n can distinguish
        # them by value.
        X, a = self._fit(80, anchor_threshold=80)
        _, b = self._fit(80, anchor_threshold=10_000)

        assert a.consensus_anchors_ is None and b.consensus_anchors_ is None
        assert np.array_equal(a.get_labels(k=2), b.get_labels(k=2))
        assert np.allclose(a.consensus_matrices_[0], b.consensus_matrices_[0],
                           equal_nan=True)
        assert np.allclose(a.stability_gini_scores_, b.stability_gini_scores_)
        assert np.allclose(
            a.estimator_results_["ari_stability"].to_numpy(),
            b.estimator_results_["ari_stability"].to_numpy(),
        )

    def test_no_warning_on_the_exact_path(self):
        X = _blobs(80, seed=3)
        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)
            CARVE(
                estimator_param_grids=_grids(),
                n_resamples=8,
                random_state=11,
            ).fit(X)
