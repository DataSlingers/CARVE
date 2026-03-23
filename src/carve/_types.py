"""Canonical type aliases and mode policy for CARVE.

Defines shared type aliases (``GridSpec``, ``PreprocSpec``, ``Measure``,
``Rule``, etc.) and the ``ModePolicy`` dataclass used to control which
pipeline stages are executed.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal

from sklearn.base import ClusterMixin, TransformerMixin

# (EstimatorClass, param_grid) pair for grid search
GridSpec = tuple[type[ClusterMixin], dict[str, list[Any]]]

# (TransformerClass, param_grid) pair for preprocessing
PreprocSpec = tuple[Callable[..., TransformerMixin], dict[str, list[Any]]]

# (TransformerClass, display_name, param_grid) triple
PreprocSpecWithName = tuple[Callable[..., TransformerMixin], str, dict[str, list[Any]]]

# Either a 2-tuple or 3-tuple preprocessing spec
PreprocOption = PreprocSpec | PreprocSpecWithName

# Per-configuration result dictionary
EstimatorRecord = dict[str, Any]

# Per-resample preprocessing record
PipelineRecord = dict[str, Any]

# Accepted metric name aliases
Measure = Literal[
    "s",
    "stab",
    "stability",
    "ari_stability",
    "g",
    "gen",
    "generalizability",
    "ari_generalizability",
    "avg",
    "average",
    "ari_average",
    "pac",
    "consensus_pac_stability",
    "gini",
    "consensus_gini_stability",
    "ce",
    "consensus_ce_stability",
    "acc",
    "accuracy",
    "accuracy_generalizability",
]

# Accepted selection rule names
Rule = Literal["max", "1se", "quantile"]

# Handling RunMode
RunMode = Literal["default", "stability", "generalizability"]


@dataclass(frozen=True)
class ModePolicy:
    """Immutable policy controlling which pipeline stages are executed.

    Attributes
    ----------
    mode : RunMode
        The mode string that produced this policy.
    run_stability : bool
        Whether to run the stability analysis branch.
    run_generalizability : bool
        Whether to run the generalizability analysis branch.
    compute_average_ari : bool
        Whether to compute the average of stability and generalizability
        ARI scores.
    """

    mode: RunMode
    run_stability: bool
    run_generalizability: bool
    compute_average_ari: bool


def resolve_mode(mode: RunMode) -> ModePolicy:
    """Resolve a run-mode string into an execution policy.

    Parameters
    ----------
    mode : RunMode
        One of ``"default"``, ``"stability"``, or ``"generalizability"``.

    Returns
    -------
    policy : ModePolicy
        Frozen dataclass describing which pipeline stages to run.

    Raises
    ------
    ValueError
        If *mode* is not a recognized run-mode string.
    """
    if mode == "default":
        return ModePolicy(
            mode=mode,
            run_stability=True,
            run_generalizability=True,
            compute_average_ari=True,
        )

    if mode == "stability":
        return ModePolicy(
            mode=mode,
            run_stability=True,
            run_generalizability=False,
            compute_average_ari=False,
        )

    if mode == "generalizability":
        return ModePolicy(
            mode=mode,
            run_stability=False,
            run_generalizability=True,
            compute_average_ari=False,
        )

    raise ValueError(
        f"Unknown mode: {mode!r}. Expected one of "
        "'default', 'stability', 'generalizability'."
    )
