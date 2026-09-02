"""Frozen dataclasses describing a benchmark experiment.

This module is a leaf: it imports only the standard library, so every other
module in the package may depend on it without creating a cycle.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

KNOWN_ESTIMATORS: frozenset[str] = frozenset({"kmeans", "agglomerative", "spectral"})


@dataclass(frozen=True)
class Axis:
    """One swept experimental dimension.

    An axis unifies what used to be two separate experiments. The difficulty
    benchmark is Axis("difficulty_level", (0, 1, 2), ("easy", "medium",
    "hard")); the scaling benchmark is Axis("n_total", (1000, 5500, 10000),
    ("start", "middle", "end")). Iterating yields (index, value, label), and
    the index feeds the seed derivation.
    """

    name: str
    values: tuple[Any, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.labels):
            raise ValueError(
                f"Axis {self.name!r}: values and labels must have the same length, "
                f"got {len(self.values)} and {len(self.labels)}."
            )
        if not self.values:
            raise ValueError(f"Axis {self.name!r} must have at least one point.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"Axis {self.name!r} has duplicate labels: {self.labels}.")

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[tuple[int, Any, str]]:
        for index, (value, label) in enumerate(zip(self.values, self.labels)):
            yield index, value, label


@dataclass(frozen=True)
class EstimatorSpec:
    """The base clustering estimator a scenario is run with.

    Validation is strict on purpose. The routine this replaces had no else
    clause, so a misspelled name silently produced KMeans while the
    provenance column recorded the string that was passed.
    """

    name: str

    def __post_init__(self) -> None:
        if self.name not in KNOWN_ESTIMATORS:
            raise ValueError(
                f"Unknown estimator {self.name!r}. "
                f"Valid names are {sorted(KNOWN_ESTIMATORS)}."
            )
