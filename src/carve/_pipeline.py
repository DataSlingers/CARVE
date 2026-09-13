"""Preprocessing pipeline specs: allocation, instantiation and labels.

A randomized fit draws one pipeline spec per resample. The spec records the
transformer classes and the sampled hyperparameter values, so the same
pipeline can be rebuilt on any subset of the data with its own seed. The
runner does that three times per resample (``_runner.embed_resample``) and
the case-study code does it on the full data to draw the winning pipeline.
"""

import inspect
import itertools
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ._sweep import _format_param_value
from ._types import PreprocOption

# A parsed option: transformer class, user-supplied display name (or None),
# and the hyperparameter grid to draw from.
ParsedOption = tuple[Any, str | None, dict[str, list[Any]]]


def _is_function_transformer(cls: Any) -> bool:
    return isinstance(cls, type) and issubclass(cls, FunctionTransformer)


class PipelineStep(NamedTuple):
    """One step of a sampled pipeline.

    Attributes
    ----------
    cls : type
        Transformer class (or factory) to instantiate.
    params : dict
        Sampled hyperparameter values, one per grid key, in grid order.
    name : str or None
        Display name from a ``(cls, name, params)`` or dict option, else
        None.
    """

    cls: Any
    params: dict[str, Any]
    name: str | None

    @property
    def label(self) -> str:
        """Render the step: ``identity``, a function's name, or
        ``Class(key=value, ...)``.

        A ``FunctionTransformer`` without ``func`` is ``identity``; with one
        it renders the function's ``__name__``. A user-supplied name replaces
        the class name. Remaining hyperparameters follow in parentheses.
        """
        params = dict(self.params)
        func = params.pop("func", None) if _is_function_transformer(self.cls) else None
        if self.name is not None:
            base = self.name
        elif _is_function_transformer(self.cls):
            base = "identity" if func is None else getattr(func, "__name__", repr(func))
        else:
            base = getattr(self.cls, "__name__", repr(self.cls))
        if not params:
            return base
        inner = ", ".join(f"{k}={_format_param_value(v)}" for k, v in params.items())
        return f"{base}({inner})"


@dataclass(frozen=True)
class PipelineSpec:
    """A sampled (normalization, dimensionality reduction) pipeline.

    Frozen so a spec can be shared between the precompute pass, the results
    table and ``preprocessing_pipelines_`` without any of them mutating it.
    ``label`` is the key into ``preprocessing_pipelines_`` and the
    ``pipeline`` column of ``preprocessing_results_``.
    """

    normalization: PipelineStep
    dim_reduction: PipelineStep

    @property
    def label(self) -> str:
        return f"{self.normalization.label} | {self.dim_reduction.label}"


def _parse_option(option: PreprocOption) -> ParsedOption:
    """Normalize one option into ``(cls, name, grid)``.

    Accepts ``(cls, params)``, ``(cls, name, params)`` and
    ``{'cls': cls, 'params': {...}, 'name': optional}`` (``'estimator'`` or
    ``'transformer'`` for ``'cls'``, ``'grid'`` for ``'params'``).

    Raises
    ------
    ValueError
        If a tuple has an unsupported arity or a dict lacks the class.
    TypeError
        If the option is neither a tuple nor a dict.
    """
    name = None
    if isinstance(option, tuple):
        if len(option) == 3:
            cls, name, grid = option
        elif len(option) == 2:
            cls, grid = option
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
        grid = option.get("params") or option.get("grid") or {}
        name = option.get("name")
    else:
        raise TypeError(
            "Unsupported option type. Use (cls, params), (cls, name, params), or {'cls','params','name'}."
        )
    return cls, name, dict(grid or {})


def option_label(option: PreprocOption) -> str:
    """Short name of an option for the run header: ``identity``, ``log1p``,
    a user-supplied name, or the class name."""
    cls, name, grid = _parse_option(option)
    if name is not None:
        return name
    if _is_function_transformer(cls):
        funcs = list(grid.get("func") or [])
        if not funcs:
            return "identity"
        return "/".join(getattr(f, "__name__", repr(f)) for f in funcs)
    return getattr(cls, "__name__", repr(cls))


def _draw_step(rng: np.random.Generator, parsed: ParsedOption) -> PipelineStep:
    """Draw one value per grid key, uniformly, from ``rng``."""
    cls, name, grid = parsed
    params: dict[str, Any] = {}
    for key, candidates in grid.items():
        candidates = list(candidates)
        if not candidates:
            raise ValueError(
                f"The candidate list for {key!r} of "
                f"{getattr(cls, '__name__', cls)} is empty."
            )
        params[key] = candidates[int(rng.integers(len(candidates)))]
    return PipelineStep(cls=cls, params=params, name=name)


def allocate_pipelines(
    normalization_options: list[PreprocOption],
    dim_reduction_options: list[PreprocOption],
    n_resamples: int,
    random_state: int | None,
) -> list[PipelineSpec]:
    """Assign one pipeline spec to each resample.

    Stratified: the Cartesian product of normalization and reduction options
    (options, not hyperparameter values) is cycled so every combination
    receives floor(n_resamples / n_combinations) or ceil(...) resamples,
    then the order is shuffled. Hyperparameters are drawn per resample from
    each option's candidate lists. Everything comes from one generator
    seeded with ``random_state``, so the allocation depends only on the
    option lists, ``n_resamples`` and the seed: identical across
    configurations, and across runs that share those three.

    Parameters
    ----------
    normalization_options, dim_reduction_options : list
        Options in any form ``_parse_option`` accepts.
    n_resamples : int
        Length of the returned list; entry ``b`` is resample ``b``'s spec.
    random_state : int or None
        Seed for the allocation generator; None means 0.

    Returns
    -------
    specs : list of PipelineSpec

    Raises
    ------
    ValueError
        If either option list is empty, a candidate list is empty, or two
        different pipelines render as the same label.
    """
    if not normalization_options or not dim_reduction_options:
        raise ValueError(
            "randomize_preprocessing=True needs at least one normalization "
            "option and one dimensionality reduction option."
        )
    combinations = list(
        itertools.product(
            [_parse_option(o) for o in normalization_options],
            [_parse_option(o) for o in dim_reduction_options],
        )
    )
    rng = np.random.default_rng(0 if random_state is None else int(random_state))
    assigned = rng.permutation(np.arange(int(n_resamples)) % len(combinations))
    specs: list[PipelineSpec] = []
    for combo_idx in assigned:
        norm, dr = combinations[int(combo_idx)]
        specs.append(
            PipelineSpec(
                normalization=_draw_step(rng, norm),
                dim_reduction=_draw_step(rng, dr),
            )
        )

    # The label keys preprocessing_pipelines_ and the results table, so two
    # different pipelines sharing one would silently pool into a single row.
    seen: dict[str, PipelineSpec] = {}
    for spec in specs:
        if seen.setdefault(spec.label, spec) != spec:
            raise ValueError(
                f"Two different pipelines render as the same label {spec.label!r}. "
                "Give the options distinct names with (cls, name, params)."
            )
    return specs


def accepts_random_state(cls: Any) -> bool:
    """Whether ``cls`` takes a ``random_state`` keyword, probed from its
    signature so a transformer without one is never handed it."""
    target = cls.__init__ if isinstance(cls, type) else cls
    try:
        return "random_state" in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False


def _instantiate(step: PipelineStep, random_state: int) -> TransformerMixin:
    params = dict(step.params)
    if accepts_random_state(step.cls):
        # CARVE's derived seed wins over one sampled from a user grid, so the
        # per-subset seeding of spec 4.3 holds for every transformer.
        params["random_state"] = int(random_state)
    return step.cls(**params)


def pipeline_from_spec(spec: PipelineSpec, random_state: int) -> Pipeline:
    """Build the sklearn pipeline a spec describes, seeded.

    Each step receives ``random_state`` when its signature accepts it. Steps
    are named ``"norm"`` and ``"dr"``.
    """
    return Pipeline(
        [
            ("norm", _instantiate(spec.normalization, random_state)),
            ("dr", _instantiate(spec.dim_reduction, random_state)),
        ]
    )
