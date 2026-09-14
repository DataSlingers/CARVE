"""Construction of the preprocessing options case studies randomize over.

The preprocessing counterpart of _estimators. A PreprocessingSpec names each
option by registry key; resolve_preprocessing turns the keys into the option
lists CARVE takes, with PREPROCESSOR_DEFAULTS bound into each transformer
and a display name attached. Bound settings stay out of the pipeline label,
so it reads "identity | TSNE(perplexity=30)" and names only what is drawn
per resample. Transformer classes are imported when a spec is resolved, not
when this module is, so umap-learn is needed only by a spec that names UMAP.
"""

import importlib
from functools import partial
from typing import Any

from ._types import PreprocessingSpec

#: Registry key to transformer class, as an import path.
PREPROCESSOR_CLASSES: dict[str, str] = {
    "identity": "sklearn.preprocessing.FunctionTransformer",
    "standard_scaler": "sklearn.preprocessing.StandardScaler",
    "pca": "sklearn.decomposition.PCA",
    "tsne": "sklearn.manifold.TSNE",
    "umap": "umap.UMAP",
}

#: Settings bound into every transformer of a kind, pinned so results do not
#: move when a library changes its default. t-SNE mirrors the source's
#: Rtsne(pca=F, perplexity=30, max_iter=5000): random initialization and a
#: learning rate of 200, as Rtsne uses, and no early stop, so every fit runs
#: all 5,000 iterations. Perplexity is drawn from the spec's grid.
PREPROCESSOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "identity": {},
    "standard_scaler": {},
    "pca": {},
    "tsne": {
        "n_components": 2,
        "max_iter": 5000,
        "init": "random",
        "learning_rate": 200.0,
        "n_iter_without_progress": 5000,
        "min_grad_norm": 0.0,
    },
    "umap": {"n_components": 2, "min_dist": 0.1},
}

#: Display names, which pipeline labels use in place of the class name.
PREPROCESSOR_NAMES: dict[str, str] = {
    "identity": "identity",
    "standard_scaler": "StandardScaler",
    "pca": "PCA",
    "tsne": "TSNE",
    "umap": "UMAP",
}

#: One resolved option: transformer (or a partial binding its defaults),
#: display name, and the grid drawn from per resample.
ResolvedOption = tuple[Any, str, dict[str, list[Any]]]


def preprocessor_class(key: str) -> type:
    """Import and return the transformer class registered under key."""
    if key not in PREPROCESSOR_CLASSES:
        raise ValueError(
            f"Unknown preprocessor {key!r}. "
            f"Valid names are {sorted(PREPROCESSOR_CLASSES)}."
        )
    module_name, _, class_name = PREPROCESSOR_CLASSES[key].rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"The {key!r} preprocessor needs {module_name}, which is not "
            'installed. UMAP ships with the umap extra: pip install "carve-validate[umap]".'
        ) from exc
    return getattr(module, class_name)


def _resolve_option(key: str, grid: Any) -> ResolvedOption:
    cls = preprocessor_class(key)
    fixed = PREPROCESSOR_DEFAULTS[key]
    transformer = partial(cls, **fixed) if fixed else cls
    return (
        transformer,
        PREPROCESSOR_NAMES[key],
        {name: list(values) for name, values in grid.items()},
    )


def preprocessing_fingerprint(spec: PreprocessingSpec | None) -> str:
    """A stable text form of a spec and the defaults it binds, for cache keys.

    Two specs that resolve to different transformers differ here, including
    when only PREPROCESSOR_DEFAULTS changed, which the spec alone would not
    show. Nothing is imported, so computing a cache path never needs
    umap-learn.
    """
    if spec is None:
        return repr(None)
    keys = [key for key, _ in (*spec.normalization, *spec.dim_reduction)]
    return repr((spec, {key: PREPROCESSOR_DEFAULTS[key] for key in keys}))


def resolve_preprocessing(spec: PreprocessingSpec) -> dict[str, list[ResolvedOption]]:
    """The option lists CARVE takes, built from a spec.

    Returns a dict with the keys normalization_options and
    dim_reduction_options, each a list of (transformer, display name, grid)
    triples in the spec's order, so it can be passed straight to
    fit_or_load_carve or CARVE as keyword arguments. A transformer with bound
    defaults is a functools.partial of its class; carve instantiates it with
    the drawn hyperparameters and a derived seed.
    """
    return {
        "normalization_options": [
            _resolve_option(key, grid) for key, grid in spec.normalization
        ],
        "dim_reduction_options": [
            _resolve_option(key, grid) for key, grid in spec.dim_reduction
        ],
    }
