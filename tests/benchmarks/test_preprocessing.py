"""Tests for preprocessing option construction."""

import sys
import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from benchmarks._preprocessing import (
    PREPROCESSOR_CLASSES,
    PREPROCESSOR_DEFAULTS,
    PREPROCESSOR_NAMES,
    preprocessor_class,
    resolve_preprocessing,
)
from benchmarks._types import KNOWN_PREPROCESSORS, PreprocessingSpec
from carve._pipeline import allocate_pipelines, pipeline_from_spec


def _spec(dim_reduction, normalization=(("identity", {}),)):
    return PreprocessingSpec(normalization=normalization, dim_reduction=dim_reduction)


def _pipelines(spec, n_resamples):
    options = resolve_preprocessing(spec)
    return allocate_pipelines(
        options["normalization_options"],
        options["dim_reduction_options"],
        n_resamples=n_resamples,
        random_state=0,
    )


def test_every_known_preprocessor_is_registered():
    assert set(PREPROCESSOR_CLASSES) == set(KNOWN_PREPROCESSORS)
    assert set(PREPROCESSOR_DEFAULTS) == set(KNOWN_PREPROCESSORS)
    assert set(PREPROCESSOR_NAMES) == set(KNOWN_PREPROCESSORS)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("identity", FunctionTransformer),
        ("standard_scaler", StandardScaler),
        ("pca", PCA),
        ("tsne", TSNE),
    ],
)
def test_registry_keys_name_their_classes(key, expected):
    assert preprocessor_class(key) is expected


def test_the_umap_key_names_umap():
    umap = pytest.importorskip("umap")
    assert preprocessor_class("umap") is umap.UMAP


def test_an_unknown_key_raises():
    with pytest.raises(ValueError, match="Valid names"):
        preprocessor_class("nmf")


def test_a_missing_umap_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(ImportError, match=r"carve-validate\[umap\]"):
        preprocessor_class("umap")


def test_the_pinned_defaults():
    assert PREPROCESSOR_DEFAULTS["tsne"] == {"n_components": 2}
    assert PREPROCESSOR_DEFAULTS["umap"] == {"n_components": 2, "min_dist": 0.1}


# (role, key, grid, output width on six features)
_FIT_CASES = [
    ("normalization", "identity", {}, 6),
    ("normalization", "standard_scaler", {}, 6),
    ("dim_reduction", "identity", {}, 6),
    ("dim_reduction", "pca", {"n_components": [3]}, 3),
    ("dim_reduction", "tsne", {"perplexity": [5]}, 2),
    ("dim_reduction", "umap", {"n_neighbors": [5]}, 2),
]


@pytest.mark.parametrize(
    ("role", "key", "grid", "width"),
    _FIT_CASES,
    ids=[f"{role}-{key}" for role, key, _, _ in _FIT_CASES],
)
def test_every_registry_key_resolves_to_an_option_carve_fits(role, key, grid, width):
    if key == "umap":
        pytest.importorskip("umap")
    option = ((key, grid),)
    spec = (
        _spec(dim_reduction=option)
        if role == "dim_reduction"
        else _spec(normalization=option, dim_reduction=(("identity", {}),))
    )
    (pipeline,) = _pipelines(spec, n_resamples=1)
    X = np.random.default_rng(0).normal(size=(40, 6))
    with warnings.catch_warnings():
        # umap-learn warns that a seed disables its parallelism; carve seeds
        # every transformer on purpose (see _runner.embed_resample).
        warnings.filterwarnings(
            "ignore",
            message=r"n_jobs value .* overridden to 1 by setting random_state",
            category=UserWarning,
        )
        Z = pipeline_from_spec(pipeline, random_state=0).fit_transform(X)
    assert Z.shape == (40, width)


def test_bound_defaults_come_from_the_registry(monkeypatch):
    # The fixed settings are bound from PREPROCESSOR_DEFAULTS rather than
    # left to the library, so editing the table changes the transformer.
    monkeypatch.setitem(PREPROCESSOR_DEFAULTS, "tsne", {"n_components": 3})
    (pipeline,) = _pipelines(
        _spec(dim_reduction=(("tsne", {"perplexity": [5]}),)), n_resamples=1
    )
    transformer = pipeline_from_spec(pipeline, random_state=0).named_steps["dr"]
    assert transformer.n_components == 3


def test_labels_name_only_the_drawn_hyperparameters():
    # TSNE's bound n_components=2 stays out of the label; the display name
    # and the drawn values are all it shows, in the spec's option order.
    spec = _spec(
        dim_reduction=(
            ("identity", {}),
            ("tsne", {"perplexity": [30]}),
            ("pca", {"n_components": [5, 10]}),
        )
    )
    options = resolve_preprocessing(spec)
    assert [name for _, name, _ in options["dim_reduction_options"]] == [
        "identity",
        "TSNE",
        "PCA",
    ]
    assert {pipeline.label for pipeline in _pipelines(spec, n_resamples=30)} == {
        "identity | identity",
        "identity | TSNE(perplexity=30)",
        "identity | PCA(n_components=5)",
        "identity | PCA(n_components=10)",
    }
