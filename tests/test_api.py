"""Tests for CARVE public API (fit, get_labels, get_k, get_estimator, plotting, persistence)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans, AgglomerativeClustering

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from carve import CARVE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_carve():
    """Return a small fitted CARVE instance (module-scoped for speed)."""
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(30, 5) + [3, 0, 0, 0, 0],
        rng.randn(30, 5) + [0, 3, 0, 0, 0],
    ])
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
    X = np.vstack([
        rng.randn(30, 5) + [4, 0, 0, 0, 0],
        rng.randn(30, 5) + [0, 4, 0, 0, 0],
        rng.randn(30, 5) + [0, 0, 4, 0, 0],
    ])
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
            "estimator", "n_clusters",
            "ari_stability", "ari_stability_se",
            "ari_generalizability", "ari_generalizability_se",
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
        assert c1.estimator_results_["ari_stability"].iloc[0] == (
            c2.estimator_results_["ari_stability"].iloc[0]
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
        ref = np.array([0]*30 + [1]*30)
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
        ref = np.array(["A"]*30 + ["B"]*30)
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
            n_clusters=2, linkage="complete", metric="precomputed",
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
            measure="stability", save=str(path),
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
