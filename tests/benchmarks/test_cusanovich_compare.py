"""Tests for the Cusanovich comparison compute."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from benchmarks._cusanovich_compare import (
    axis_prefix,
    best_pipeline,
    pipeline_embedding,
    published_partition_generalizability,
    source_operating_point,
)
from carve._pipeline import PipelineSpec, PipelineStep

IDENTITY = PipelineStep(cls=FunctionTransformer, params={}, name="identity")


def _spec(dim_reduction: PipelineStep) -> PipelineSpec:
    return PipelineSpec(normalization=IDENTITY, dim_reduction=dim_reduction)


def _row(method_id, pipeline, resolution, stability, generalizability, observed):
    return {
        "method_id": method_id,
        "pipeline": pipeline,
        "resolution": resolution,
        "sweep_value": resolution,
        "ari_stability": stability,
        "ari_generalizability": generalizability,
        "n_clusters_observed": observed,
    }


class _FakeCarve:
    """The members of a fitted CARVE these functions read."""

    def __init__(self, table, pipelines=None, selected=None):
        self.preprocessing_results_ = table
        self.preprocessing_pipelines_ = pipelines
        self._selected = selected
        self.select_calls = []

    def _select_row(self, *, measure, rule, not_two=False):
        self.select_calls.append({"measure": measure, "rule": rule, "not_two": not_two})
        return pd.Series(self._selected), 0, 0, False


class TestBestPipeline:
    @pytest.fixture
    def carve(self):
        # Two pipelines at two resolutions of one configuration. At the
        # selected resolution, 0.5, B has the higher stability and A the
        # higher generalizability. Resolution 1.0 holds a decoy that beats
        # both on both criteria, which a function ignoring the selected
        # configuration would pick.
        table = pd.DataFrame(
            [
                _row("m0", "A", 0.5, 0.60, 0.80, 8.0),
                _row("m0", "B", 0.5, 0.70, 0.50, 9.0),
                _row("m0", "A", 1.0, 0.95, 0.95, 20.0),
                _row("m0", "B", 1.0, 0.20, 0.20, 21.0),
            ]
        )
        pipelines = {"A": object(), "B": object()}
        return _FakeCarve(
            table, pipelines, selected={"method_id": "m0", "sweep_value": 0.5}
        )

    def test_picks_the_best_row_at_the_selected_configuration_not_overall(self, carve):
        row, spec = best_pipeline(carve, measure="stability", rule="1se")
        assert row["pipeline"] == "B"
        assert row["sweep_value"] == 0.5
        assert spec is carve.preprocessing_pipelines_["B"]

    def test_the_measure_decides_the_winner(self, carve):
        row, spec = best_pipeline(carve, measure="generalizability", rule="1se")
        assert row["pipeline"] == "A"
        assert spec is carve.preprocessing_pipelines_["A"]

    def test_forwards_the_selection_arguments(self, carve):
        best_pipeline(carve, measure="generalizability", rule="max", not_two=True)
        assert carve.select_calls == [
            {"measure": "generalizability", "rule": "max", "not_two": True}
        ]

    def test_another_configuration_at_the_same_resolution_does_not_compete(
        self, carve
    ):
        decoy = pd.DataFrame([_row("m1", "A", 0.5, 0.99, 0.99, 8.0)])
        carve.preprocessing_results_ = pd.concat(
            [carve.preprocessing_results_, decoy], ignore_index=True
        )
        row, _ = best_pipeline(carve, measure="stability", rule="1se")
        assert (row["method_id"], row["pipeline"]) == ("m0", "B")

    def test_a_non_ari_measure_raises(self, carve):
        with pytest.raises(ValueError, match="stability or generalizability"):
            best_pipeline(carve, measure="pac", rule="1se")

    def test_a_fit_without_randomized_preprocessing_raises(self):
        with pytest.raises(ValueError, match="randomize_preprocessing=True"):
            best_pipeline(_FakeCarve(None), measure="stability", rule="1se")


class TestPipelineEmbedding:
    @pytest.fixture
    def X(self):
        return np.random.default_rng(0).normal(size=(60, 50))

    def test_the_identity_pipeline_gives_the_first_two_lsi_components(self, X):
        Z, labels = pipeline_embedding(X, _spec(IDENTITY), random_state=0)
        np.testing.assert_array_equal(Z, X[:, :2])
        assert labels == ("LSI 1", "LSI 2")

    def test_a_two_dimensional_pipeline_gives_its_output_fit_on_all_of_x(self, X):
        step = PipelineStep(cls=PCA, params={"n_components": 2}, name="PCA")
        Z, labels = pipeline_embedding(X, _spec(step), random_state=0)
        np.testing.assert_allclose(
            Z, PCA(n_components=2, random_state=0).fit_transform(X)
        )
        assert labels == ("PC 1", "PC 2")

    def test_tsne_axes_are_named_for_tsne_and_seeded(self, X):
        step = PipelineStep(cls=TSNE, params={"perplexity": 5}, name="TSNE")
        first, labels = pipeline_embedding(X, _spec(step), random_state=3)
        second, _ = pipeline_embedding(X, _spec(step), random_state=3)
        assert first.shape == (60, 2)
        np.testing.assert_array_equal(first, second)
        assert labels == ("t-SNE 1", "t-SNE 2")

    def test_axis_prefixes(self):
        # A display name decides the prefix; an unnamed step falls back to
        # its class name, mapped when the registry knows it and kept when not.
        assert axis_prefix(PipelineStep(cls=object, params={}, name="UMAP")) == "UMAP"
        assert axis_prefix(PipelineStep(cls=PCA, params={}, name=None)) == "PC"
        scaler = PipelineStep(cls=StandardScaler, params={}, name=None)
        assert axis_prefix(scaler) == "StandardScaler"


class TestSourceOperatingPoint:
    @pytest.fixture
    def carve(self):
        # The t-SNE pipeline's counts rise with resolution, listed out of
        # order. A decoy pipeline hits 30 exactly, which a function ignoring
        # the pipeline would return.
        table = pd.DataFrame(
            [
                _row("m0", "tsne", 0.6, 0.5, 0.5, 33.0),
                _row("m0", "tsne", 0.2, 0.5, 0.5, 12.0),
                _row("m0", "tsne", 0.4, 0.5, 0.5, 26.0),
                _row("m0", "umap", 0.8, 0.5, 0.5, 30.0),
            ]
        )
        return _FakeCarve(table)

    def test_picks_the_named_pipelines_nearest_count(self, carve):
        assert source_operating_point(carve, pipeline="tsne", target_k=30) == (
            0.6,
            33.0,
        )

    def test_ties_go_to_the_lower_resolution(self, carve):
        # 26 and 34 are both 4 from 30; the table lists 0.6 first.
        carve.preprocessing_results_.loc[0, "n_clusters_observed"] = 34.0
        assert source_operating_point(carve, pipeline="tsne", target_k=30) == (
            0.4,
            26.0,
        )

    def test_stays_silent_within_a_quarter_of_the_target(self, carve):
        # 33 is 7 from 40, inside 25 percent (10); any warning fails the test
        # under filterwarnings = error.
        assert source_operating_point(carve, pipeline="tsne", target_k=40) == (
            0.6,
            33.0,
        )

    def test_warns_past_a_quarter_of_the_target(self, carve):
        # 33 is 17 from 50, past 25 percent (12.5).
        with pytest.warns(UserWarning, match="more than 25%"):
            point = source_operating_point(carve, pipeline="tsne", target_k=50)
        assert point == (0.6, 33.0)

    def test_an_unknown_pipeline_raises_naming_the_available_ones(self, carve):
        with pytest.raises(ValueError, match="umap"):
            source_operating_point(carve, pipeline="pca", target_k=30)

    def test_more_than_one_configuration_raises(self, carve):
        carve.preprocessing_results_.loc[1, "method_id"] = "m1"
        with pytest.raises(ValueError, match="more than one estimator configuration"):
            source_operating_point(carve, pipeline="tsne", target_k=30)


class TestPublishedPartitionGeneralizability:
    @pytest.fixture
    def blobs(self):
        rng = np.random.default_rng(0)
        centers = rng.normal(scale=10.0, size=(3, 5))
        members = np.repeat(np.arange(3), 60)
        X = centers[members] + rng.normal(scale=0.3, size=(180, 5))
        return X, np.array(["a", "b", "c"])[members]

    @staticmethod
    def _score(X, labels, **overrides):
        settings = {
            "n_splits": 10,
            "subsample_ratio": 0.618,
            "n_trees": 20,
            "random_state": 0,
        }
        settings.update(overrides)
        return published_partition_generalizability(X, labels, **settings)

    def test_a_partition_that_is_a_function_of_x_generalizes_perfectly(self, blobs):
        mean, se = self._score(*blobs)
        assert mean == pytest.approx(1.0)
        assert se == pytest.approx(0.0)

    def test_a_random_relabeling_does_not_generalize(self, blobs):
        # A forest memorizes any labeling of its training rows, so scoring on
        # the training rows, or against them, would come out near 1.
        X, labels = blobs
        shuffled = np.random.default_rng(1).permutation(labels)
        mean, _ = self._score(X, shuffled, n_splits=30)
        assert abs(mean) < 0.05

    def test_is_seeded(self, blobs):
        X, labels = blobs
        shuffled = np.random.default_rng(1).permutation(labels)
        assert self._score(X, shuffled) == self._score(X, shuffled)

    def test_a_single_split_has_no_standard_error(self, blobs):
        mean, se = self._score(*blobs, n_splits=1)
        assert mean == pytest.approx(1.0)
        assert np.isnan(se)
