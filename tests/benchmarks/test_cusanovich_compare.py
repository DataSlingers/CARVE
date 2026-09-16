"""Tests for the Cusanovich comparison compute."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sklearn.metrics import adjusted_rand_score

from benchmarks._cusanovich_compare import (
    TABLE_FILENAMES,
    axis_prefix,
    best_pipeline,
    pipeline_embedding,
    prepare_cusanovich_inputs,
    save_tables,
    selection_summary,
    source_operating_point,
)
from benchmarks._estimators import resolution_grids
from benchmarks._preprocessing import resolve_preprocessing
from benchmarks._types import EstimatorSpec, PreprocessingSpec
from benchmarks.figures._case_study import _align_to_reference
from carve import CARVE
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

    def __init__(self, table, pipelines=None, selected=None, results=None):
        self.preprocessing_results_ = table
        self.estimator_results_ = results
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
        # Configuration m0's pooled counts rise with resolution, listed out of
        # order. Configuration m1 hits 30 exactly, and so does one pipeline of
        # m0 at 0.2, where the pooled count is 12; a function reading another
        # configuration, or a single pipeline, would return one of those.
        per_pipeline = pd.DataFrame([_row("m0", "tsne", 0.2, 0.5, 0.5, 30.0)])
        pooled = pd.DataFrame(
            {
                "method_id": ["m0", "m0", "m0", "m1"],
                "sweep_value": [0.6, 0.2, 0.4, 0.8],
                "n_clusters_observed": [33.0, 12.0, 26.0, 30.0],
            }
        )
        return _FakeCarve(per_pipeline, results=pooled)

    def test_picks_the_configurations_nearest_pooled_count(self, carve):
        assert source_operating_point(carve, method_id="m0", target_k=30) == (
            0.6,
            33.0,
        )

    def test_ties_go_to_the_lower_resolution(self, carve):
        # 26 and 34 are both 4 from 30; the table lists 0.6 first.
        carve.estimator_results_.loc[0, "n_clusters_observed"] = 34.0
        assert source_operating_point(carve, method_id="m0", target_k=30) == (
            0.4,
            26.0,
        )

    def test_stays_silent_within_a_quarter_of_the_target(self, carve):
        # 33 is 7 from 40, inside 25 percent (10); any warning fails the test
        # under filterwarnings = error.
        assert source_operating_point(carve, method_id="m0", target_k=40) == (
            0.6,
            33.0,
        )

    def test_warns_past_a_quarter_of_the_target(self, carve):
        # 33 is 17 from 50, past 25 percent (12.5).
        with pytest.warns(UserWarning, match="more than 25%"):
            point = source_operating_point(carve, method_id="m0", target_k=50)
        assert point == (0.6, 33.0)

    def test_an_unknown_configuration_raises_naming_the_available_ones(self, carve):
        with pytest.raises(ValueError, match=r"\['m0', 'm1'\]"):
            source_operating_point(carve, method_id="m2", target_k=30)


# Four separated blobs, labeled so the reference's sorted codes (a, b, c, d)
# differ from its first-appearance order (d, b, a, c), which is the order
# CARVE factorizes reference labels in.
_BLOB_NAMES = np.array(["d", "b", "a", "c"])


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=8.0, size=(4, 6))
    members = np.repeat(np.arange(4), 40)
    X = centers[members] + rng.normal(scale=0.5, size=(160, 6))
    y = _BLOB_NAMES[members]
    spec = PreprocessingSpec(
        normalization=(("identity", {}),),
        dim_reduction=(("identity", {}), ("tsne", {"perplexity": [30]})),
    )
    carve = CARVE(
        estimator_param_grids=resolution_grids(EstimatorSpec(name="leiden"), (0.1, 0.5)),
        n_resamples=6,
        n_trees=20,
        random_state=0,
        **resolve_preprocessing(spec),
    )
    carve.fit(X, reference_labels=y, randomize_preprocessing=True)
    return X, y, carve


@pytest.fixture(scope="module")
def inputs(fitted):
    X, y, carve = fitted
    return prepare_cusanovich_inputs(
        X, y, carve, source_tsne=X[:, :2], random_state=0
    )


@pytest.mark.requires_graph
class TestPrepareCusanovichInputs:
    def test_embeds_all_of_x_with_the_best_pipeline(self, fitted, inputs):
        X, _, carve = fitted
        row, spec = best_pipeline(carve, measure="stability", rule="1se")
        assert inputs.best_pipeline_row.equals(row)
        expected, labels = pipeline_embedding(X, spec, random_state=0)
        np.testing.assert_allclose(inputs.embedding_A, expected)
        assert inputs.embedding_A_labels == labels
        assert (inputs.measure, inputs.rule, inputs.not_two) == ("stability", "1se", False)

    def test_carve_labels_are_relabeled_onto_the_reference_codes(self, fitted, inputs):
        _, y, carve = fitted
        raw = carve.get_labels(measure="stability", rule="1se")
        np.testing.assert_array_equal(inputs.carve_labels, _align_to_reference(raw, y))

    def test_the_operating_point_targets_the_reference_cluster_count(
        self, fitted, inputs
    ):
        # y has four clusters, so the operating point is the selected
        # configuration's resolution nearest four; a fixed 30 would be far off
        # and warn.
        _, _, carve = fitted
        selected, _, _, _ = carve._select_row(measure="stability", rule="1se")
        assert inputs.operating_point == source_operating_point(
            carve, method_id=selected["method_id"], target_k=4
        )

    def test_embeds_with_the_given_seed(self, fitted, monkeypatch):
        # The best pipeline may be the seedless identity, so the embedding
        # cannot show which seed was passed. Record the call instead.
        import benchmarks._cusanovich_compare as compare

        X, y, carve = fitted
        embeddings = []
        real_embedding = compare.pipeline_embedding

        def embedding(X, spec, *, random_state):
            embeddings.append((spec, random_state))
            return real_embedding(X, spec, random_state=random_state)

        monkeypatch.setattr(compare, "pipeline_embedding", embedding)
        prepare_cusanovich_inputs(X, y, carve, source_tsne=X[:, :2], random_state=7)

        _, spec = best_pipeline(carve, measure="stability", rule="1se")
        assert embeddings == [(spec, 7)]

    def test_the_summary_reports_the_selection_and_the_comparison(
        self, fitted, inputs
    ):
        _, y, carve = fitted
        summary = selection_summary(inputs).set_index("quantity")["value"]
        selected, _, n_clusters, _ = carve._select_row(measure="stability", rule="1se")
        assert summary["selected_resolution"] == float(selected["sweep_value"])
        assert summary["selected_n_clusters"] == n_clusters
        assert summary["best_pipeline"] == inputs.best_pipeline_row["pipeline"]
        assert summary["operating_point_resolution"] == inputs.operating_point[0]
        assert summary["operating_point_n_clusters"] == inputs.operating_point[1]
        assert not any(
            quantity.startswith(("published", "source_pipeline"))
            for quantity in summary.index
        )
        raw = carve.get_labels(measure="stability", rule="1se")
        assert summary["consensus_ari_vs_source_clusters"] == pytest.approx(
            adjusted_rand_score(y, raw)
        )

    def test_save_tables_writes_both_csvs(self, fitted, inputs, tmp_path):
        _, _, carve = fitted
        paths = save_tables(inputs, tmp_path / "out")
        assert [path.name for path in paths] == list(TABLE_FILENAMES)
        assert len(pd.read_csv(paths[0])) == len(carve.preprocessing_results_)
        written = pd.read_csv(paths[1])
        assert list(written["quantity"]) == list(selection_summary(inputs)["quantity"])
