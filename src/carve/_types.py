"""Canonical type aliases for CARVE."""

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
    "misclass",
    "misclassification",
    "misclassification_generalizability",
]

# Accepted selection rule names
Rule = Literal["max", "1se", "quantile"]
