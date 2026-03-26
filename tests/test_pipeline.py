"""Tests for carve._pipeline module."""

import random

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from carve._pipeline import (
    build_preprocessing_pipeline,
    sample_preprocessing_pipeline,
    _choose_preprocessor,
)


# -----------------------------------------------------------------------
# build_preprocessing_pipeline
# -----------------------------------------------------------------------


class TestBuildPreprocessingPipeline:
    def test_identity_when_not_randomized(self):
        pipeline, norm_p, dr_p, norm_name, dr_name = build_preprocessing_pipeline(
            randomize_preprocessing=False,
            normalization_options=[],
            dim_reduction_options=[],
            seed=0,
        )
        assert isinstance(pipeline, Pipeline)
        assert norm_p == {}
        assert dr_p == {}
        assert norm_name == "Identity"
        assert dr_name == "Identity"

    def test_identity_passthrough(self):
        """Identity pipeline should not transform data."""
        pipeline, *_ = build_preprocessing_pipeline(
            randomize_preprocessing=False,
            normalization_options=[],
            dim_reduction_options=[],
            seed=0,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = pipeline.fit_transform(X)
        np.testing.assert_array_equal(result, X)

    def test_randomized(self):
        norm_opts = [
            (FunctionTransformer, {}),
            (StandardScaler, {}),
        ]
        dr_opts = [
            (FunctionTransformer, {}),
        ]
        pipeline, norm_p, dr_p, norm_name, dr_name = build_preprocessing_pipeline(
            randomize_preprocessing=True,
            normalization_options=norm_opts,
            dim_reduction_options=dr_opts,
            seed=42,
        )
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2

    def test_reproducibility(self):
        norm_opts = [
            (FunctionTransformer, {}),
            (StandardScaler, {}),
        ]
        dr_opts = [(FunctionTransformer, {})]

        _, _, _, name1, _ = build_preprocessing_pipeline(
            True,
            norm_opts,
            dr_opts,
            seed=42,
        )
        _, _, _, name2, _ = build_preprocessing_pipeline(
            True,
            norm_opts,
            dr_opts,
            seed=42,
        )
        assert name1 == name2


# -----------------------------------------------------------------------
# sample_preprocessing_pipeline
# -----------------------------------------------------------------------


class TestSamplePreprocessingPipeline:
    def test_basic(self):
        norm_opts = [(StandardScaler, {})]
        dr_opts = [(FunctionTransformer, {})]
        pipeline, norm_p, dr_p, norm_name, dr_name = sample_preprocessing_pipeline(
            norm_opts,
            dr_opts,
            seed=0,
        )
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert norm_name == "StandardScaler"
        assert dr_name == "FunctionTransformer"

    def test_transforms_data(self):
        norm_opts = [(StandardScaler, {})]
        dr_opts = [(FunctionTransformer, {})]
        pipeline, *_ = sample_preprocessing_pipeline(norm_opts, dr_opts, seed=0)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = pipeline.fit_transform(X)
        # StandardScaler should center the data
        assert abs(result.mean()) < 1e-10


# -----------------------------------------------------------------------
# _choose_preprocessor
# -----------------------------------------------------------------------


class TestChoosePreprocessor:
    def test_tuple2(self):
        rnd = random.Random(0)
        options = [(StandardScaler, {})]
        inst, params, name = _choose_preprocessor(rnd, options)
        assert isinstance(inst, StandardScaler)
        assert params == {}
        assert name == "StandardScaler"

    def test_tuple3_with_name(self):
        rnd = random.Random(0)
        options = [(StandardScaler, "MyScaler", {})]
        inst, params, name = _choose_preprocessor(rnd, options)
        assert isinstance(inst, StandardScaler)
        assert name == "MyScaler"

    def test_tuple2_with_params(self):
        rnd = random.Random(0)
        options = [(FunctionTransformer, {"func": [np.log1p, np.sqrt]})]
        inst, params, name = _choose_preprocessor(rnd, options)
        assert isinstance(inst, FunctionTransformer)
        assert "func" in params
        assert params["func"] in [np.log1p, np.sqrt]

    def test_dict_option(self):
        rnd = random.Random(0)
        options = [{"cls": StandardScaler, "params": {}, "name": "DictScaler"}]
        inst, params, name = _choose_preprocessor(rnd, options)
        assert isinstance(inst, StandardScaler)
        assert name == "DictScaler"

    def test_dict_without_name(self):
        rnd = random.Random(0)
        options = [{"cls": StandardScaler, "params": {}}]
        inst, params, name = _choose_preprocessor(rnd, options)
        assert name == "StandardScaler"

    def test_invalid_tuple_length(self):
        rnd = random.Random(0)
        options = [(StandardScaler,)]
        with pytest.raises(ValueError, match="Option tuples must be"):
            _choose_preprocessor(rnd, options)

    def test_invalid_type(self):
        rnd = random.Random(0)
        options = ["not_a_valid_option"]
        with pytest.raises(TypeError, match="Unsupported option type"):
            _choose_preprocessor(rnd, options)

    def test_dict_missing_cls(self):
        rnd = random.Random(0)
        options = [{"params": {}}]
        with pytest.raises(ValueError, match="Dict option must contain"):
            _choose_preprocessor(rnd, options)
