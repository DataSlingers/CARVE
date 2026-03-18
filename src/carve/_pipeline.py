"""Preprocessing pipeline construction and sampling utilities."""

import random
from typing import Any

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ._types import PreprocOption


def build_preprocessing_pipeline(
    randomize_preprocessing: bool,
    normalization_options: list[PreprocOption],
    dim_reduction_options: list[PreprocOption],
    seed: int,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any], str, str]:
    """Build a preprocessing pipeline.

    Parameters
    ----------
    randomize_preprocessing : bool
        If True, randomly sample normalization and DR options.
    normalization_options : list
        Normalization options compatible with ``_choose_preprocessor``.
    dim_reduction_options : list
        Dimensionality reduction options compatible with ``_choose_preprocessor``.
    seed : int
        Random seed for sampling preprocessors.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Assembled preprocessing pipeline.
    normalization_params : dict
        Sampled normalization parameters.
    dim_reduction_params : dict
        Sampled dimensionality reduction parameters.
    normalization_name : str
        Resolved name for the normalization step.
    dim_reduction_name : str
        Resolved name for the DR step.
    """
    if randomize_preprocessing:
        (
            pipeline,
            normalization_params,
            dim_reduction_params,
            normalization_name,
            dim_reduction_name,
        ) = sample_preprocessing_pipeline(
            normalization_options, dim_reduction_options, seed
        )
    else:
        pipeline = Pipeline([("id", FunctionTransformer(lambda x: x))])
        normalization_params = dim_reduction_params = {}
        normalization_name = dim_reduction_name = "Identity"

    return (
        pipeline,
        normalization_params,
        dim_reduction_params,
        normalization_name,
        dim_reduction_name,
    )


def sample_preprocessing_pipeline(
    normalization_options: list[PreprocOption],
    dim_reduction_options: list[PreprocOption],
    seed: int,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any], str, str]:
    """Randomly sample normalization and DR steps into a pipeline.

    Parameters
    ----------
    normalization_options : list
        Normalization options compatible with ``_choose_preprocessor``.
    dim_reduction_options : list
        DR options compatible with ``_choose_preprocessor``.
    seed : int
        Random seed for sampling.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Assembled pipeline with two steps: normalization and DR.
    normalization_params : dict
        Sampled normalization parameters.
    dim_reduction_params : dict
        Sampled dimensionality reduction parameters.
    normalization_name : str
        Resolved name for the normalization step.
    dim_reduction_name : str
        Resolved name for the DR step.
    """
    rnd = random.Random(seed)

    normalization_step, normalization_params, normalization_name = _choose_preprocessor(
        rnd, normalization_options
    )
    dim_reduction_step, dim_reduction_params, dim_reduction_name = _choose_preprocessor(
        rnd, dim_reduction_options
    )

    pipeline = Pipeline([("norm", normalization_step), ("dr", dim_reduction_step)])
    return (
        pipeline,
        normalization_params,
        dim_reduction_params,
        normalization_name,
        dim_reduction_name,
    )


def _choose_preprocessor(
    rnd: random.Random,
    options: list[PreprocOption],
) -> tuple[TransformerMixin, dict[str, Any], str]:
    """Randomly select and instantiate a preprocessor from a set of options.

    Parameters
    ----------
    rnd : random.Random
        Random generator for sampling.
    options : list
        Options supporting:
        - (cls, params)
        - (cls, name, params)
        - {'cls': cls, 'params': {...}, 'name': optional}

    Returns
    -------
    transformer : TransformerMixin
        Instantiated transformer with sampled parameters.
    chosen_params : dict
        Sampled parameter values.
    name : str
        Resolved display name for the transformer.

    Raises
    ------
    ValueError
        If an option tuple has an unsupported arity.
    TypeError
        If an option has an unsupported type.
    """
    option = rnd.choice(options)
    name = None
    params: dict[str, list[Any]] = {}

    if isinstance(option, tuple):
        if len(option) == 3:
            cls, name, params = option
        elif len(option) == 2:
            cls, params = option
        else:
            raise ValueError(
                "Option tuples must be (cls, params) or (cls, name, params)."
            )

    elif isinstance(option, dict):
        cls = option.get("cls") or option.get("estimator") or option.get("transformer")

        if cls is None:
            raise ValueError(
                "Dict option must contain key 'cls' (or 'estimator'/'transformer')."
            )

        params = option.get("params") or option.get("grid") or {}
        name = option.get("name")

    else:
        raise TypeError(
            "Unsupported option type. Use (cls, params), (cls, name, params), or {'cls','params','name'}."
        )

    chosen_params = {k: rnd.choice(v) for k, v in (params or {}).items()}
    inst = cls(**chosen_params)
    resolved_name = name or inst.__class__.__name__
    return inst, chosen_params, resolved_name
