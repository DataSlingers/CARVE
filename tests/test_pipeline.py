"""Tests for carve._pipeline module."""

import itertools
from collections import Counter

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from carve._pipeline import (
    PipelineSpec,
    PipelineStep,
    _parse_option,
    accepts_random_state,
    allocate_pipelines,
    option_label,
    pipeline_from_spec,
)
from tests._helpers import make_noise_embedding

IDENTITY = PipelineStep(FunctionTransformer, {}, None)

# -----------------------------------------------------------------------
# PipelineStep and PipelineSpec labels
# -----------------------------------------------------------------------


class TestLabels:
    def test_identity_renders_as_identity(self):
        assert IDENTITY.label == "identity"

    def test_function_transformer_renders_its_func(self):
        step = PipelineStep(FunctionTransformer, {"func": np.log1p}, None)
        assert step.label == "log1p"

    def test_class_name_with_sampled_hyperparameters_in_grid_order(self):
        step = PipelineStep(TSNE, {"n_components": 2, "perplexity": 30}, None)
        assert step.label == "TSNE(n_components=2, perplexity=30)"

    def test_class_without_hyperparameters(self):
        assert PipelineStep(StandardScaler, {}, None).label == "StandardScaler"

    def test_user_name_replaces_the_class_name(self):
        step = PipelineStep(TSNE, {"perplexity": 30}, "tsne")
        assert step.label == "tsne(perplexity=30)"

    def test_floats_render_compactly(self):
        step = PipelineStep(PCA, {"n_components": 0.95}, None)
        assert step.label == "PCA(n_components=0.95)"

    def test_spec_label_joins_the_steps(self):
        spec = PipelineSpec(IDENTITY, PipelineStep(PCA, {"n_components": 2}, None))
        assert spec.label == "identity | PCA(n_components=2)"

    def test_specs_with_different_hyperparameters_are_distinct(self):
        a = PipelineSpec(IDENTITY, PipelineStep(PCA, {"n_components": 2}, None))
        b = PipelineSpec(IDENTITY, PipelineStep(PCA, {"n_components": 3}, None))
        assert a != b and a.label != b.label


# -----------------------------------------------------------------------
# allocate_pipelines
# -----------------------------------------------------------------------


NORM = [(FunctionTransformer, {}), (StandardScaler, {})]
DR = [
    (FunctionTransformer, {}),
    (PCA, {"n_components": [2, 3]}),
    (TSNE, {"n_components": [2], "perplexity": [15, 30]}),
]


def _combination(spec):
    """The (normalization option, reduction option) a spec was drawn from."""
    return (spec.normalization.cls, spec.dim_reduction.cls)


class TestAllocatePipelines:
    def test_one_spec_per_resample(self):
        specs = allocate_pipelines(NORM, DR, 20, 0)
        assert len(specs) == 20
        assert all(isinstance(s, PipelineSpec) for s in specs)

    def test_counts_per_combination_differ_by_at_most_one(self):
        # 2 x 3 = 6 combinations over 100 resamples: 16 or 17 each.
        counts = Counter(_combination(s) for s in allocate_pipelines(NORM, DR, 100, 0))
        assert len(counts) == 6
        assert set(counts.values()) <= {16, 17}
        assert sum(counts.values()) == 100

    def test_every_combination_used_when_resamples_exceed_combinations(self):
        counts = Counter(_combination(s) for s in allocate_pipelines(NORM, DR, 7, 0))
        assert len(counts) == 6
        assert set(counts.values()) == {1, 2}

    def test_order_is_shuffled_not_cyclic(self):
        order = [_combination(s) for s in allocate_pipelines(NORM, DR, 60, 0)]
        cyclic = [(n[0], d[0]) for n, d in itertools.product(NORM, DR)] * 10
        assert sorted(order, key=repr) == sorted(cyclic, key=repr)
        assert order != cyclic

    def test_hyperparameters_are_drawn_from_the_candidate_lists(self):
        specs = allocate_pipelines(NORM, DR, 60, 0)
        pca = [s.dim_reduction.params for s in specs if s.dim_reduction.cls is PCA]
        tsne = [s.dim_reduction.params for s in specs if s.dim_reduction.cls is TSNE]
        assert {p["n_components"] for p in pca} == {2, 3}
        assert {p["perplexity"] for p in tsne} == {15, 30}
        assert all(p["n_components"] == 2 for p in tsne)

    def test_identical_for_the_same_seed_and_different_otherwise(self):
        a = allocate_pipelines(NORM, DR, 30, 5)
        b = allocate_pipelines(NORM, DR, 30, 5)
        c = allocate_pipelines(NORM, DR, 30, 6)
        assert a == b
        assert a != c

    def test_none_seed_means_zero(self):
        assert allocate_pipelines(NORM, DR, 10, None) == allocate_pipelines(NORM, DR, 10, 0)

    @pytest.mark.parametrize("norm, dr", [([], DR), (NORM, [])])
    def test_empty_option_list_raises(self, norm, dr):
        with pytest.raises(ValueError, match="at least one normalization option"):
            allocate_pipelines(norm, dr, 5, 0)

    def test_two_pipelines_with_one_label_raise(self):
        # Both normalization options are named "x", so two different
        # pipelines would share a key in preprocessing_pipelines_.
        with pytest.raises(ValueError, match=r"render as the same label 'x \| identity'"):
            allocate_pipelines(
                [(StandardScaler, "x", {}), (FunctionTransformer, "x", {})],
                [(FunctionTransformer, {})],
                4,
                0,
            )

    def test_empty_candidate_list_raises(self):
        with pytest.raises(ValueError, match="candidate list for 'n_components'"):
            allocate_pipelines(NORM, [(PCA, {"n_components": []})], 5, 0)

    def test_accepts_named_and_dict_options(self):
        specs = allocate_pipelines(
            [(StandardScaler, "scaled", {})],
            [{"cls": PCA, "params": {"n_components": [2]}, "name": "pca"}],
            3,
            0,
        )
        assert {s.label for s in specs} == {"scaled | pca(n_components=2)"}


# -----------------------------------------------------------------------
# pipeline_from_spec and accepts_random_state
# -----------------------------------------------------------------------


class TestPipelineFromSpec:
    def test_builds_a_two_step_pipeline(self):
        spec = PipelineSpec(
            PipelineStep(StandardScaler, {}, None),
            PipelineStep(PCA, {"n_components": 2}, None),
        )
        pipeline = pipeline_from_spec(spec, random_state=0)
        assert isinstance(pipeline, Pipeline)
        assert [name for name, _ in pipeline.steps] == ["norm", "dr"]
        assert isinstance(pipeline.named_steps["norm"], StandardScaler)
        assert pipeline.named_steps["dr"].n_components == 2

    def test_seeds_transformers_that_accept_random_state(self):
        spec = PipelineSpec(IDENTITY, PipelineStep(PCA, {"n_components": 2}, None))
        assert pipeline_from_spec(spec, random_state=7).named_steps["dr"].random_state == 7

    def test_does_not_seed_transformers_without_random_state(self):
        spec = PipelineSpec(PipelineStep(StandardScaler, {}, None), IDENTITY)
        params = pipeline_from_spec(spec, random_state=7).named_steps["norm"].get_params()
        assert "random_state" not in params

    def test_overrides_a_random_state_in_the_sampled_params(self):
        spec = PipelineSpec(
            IDENTITY, PipelineStep(PCA, {"n_components": 2, "random_state": 99}, None)
        )
        assert pipeline_from_spec(spec, random_state=7).named_steps["dr"].random_state == 7

    def test_identity_pipeline_passes_data_through(self):
        X = np.arange(6.0).reshape(3, 2)
        np.testing.assert_array_equal(
            pipeline_from_spec(PipelineSpec(IDENTITY, IDENTITY), 0).fit_transform(X), X
        )

    def test_seed_reaches_a_stochastic_transformer(self):
        noise = make_noise_embedding()
        spec = PipelineSpec(IDENTITY, PipelineStep(noise, {}, None))
        X = np.zeros((5, 3))
        a = pipeline_from_spec(spec, random_state=1).fit_transform(X)
        b = pipeline_from_spec(spec, random_state=1).fit_transform(X)
        c = pipeline_from_spec(spec, random_state=2).fit_transform(X)
        np.testing.assert_array_equal(a, b)
        assert not np.allclose(a, c)

    def test_accepts_random_state_probe(self):
        assert accepts_random_state(PCA)
        assert accepts_random_state(TSNE)
        assert not accepts_random_state(StandardScaler)
        assert not accepts_random_state(FunctionTransformer)


# -----------------------------------------------------------------------
# option_label and _parse_option
# -----------------------------------------------------------------------


class TestOptionLabel:
    def test_labels(self):
        assert option_label((FunctionTransformer, {})) == "identity"
        assert option_label((FunctionTransformer, {"func": [np.log1p]})) == "log1p"
        assert option_label((StandardScaler, {})) == "StandardScaler"
        assert option_label((PCA, {"n_components": [2, 5]})) == "PCA"
        assert option_label((PCA, "pca", {"n_components": [2]})) == "pca"
        assert option_label({"cls": TSNE, "params": {}}) == "TSNE"


class TestParseOption:
    def test_tuple2(self):
        assert _parse_option((StandardScaler, {})) == (StandardScaler, None, {})

    def test_tuple3(self):
        assert _parse_option((StandardScaler, "s", {"a": [1]})) == (
            StandardScaler,
            "s",
            {"a": [1]},
        )

    def test_dict_forms(self):
        assert _parse_option({"cls": PCA, "params": {"n_components": [2]}}) == (
            PCA,
            None,
            {"n_components": [2]},
        )
        assert _parse_option({"estimator": PCA, "grid": {"n_components": [2]}, "name": "p"}) == (
            PCA,
            "p",
            {"n_components": [2]},
        )

    def test_invalid_tuple_length(self):
        with pytest.raises(ValueError, match="Option tuples must be"):
            _parse_option((StandardScaler,))

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="Unsupported option type"):
            _parse_option("not_a_valid_option")

    def test_dict_missing_cls(self):
        with pytest.raises(ValueError, match="Dict option must contain"):
            _parse_option({"params": {}})
