"""Frozen dataclasses describing a benchmark experiment.

This module is a leaf: it imports only the standard library, so every other
module in the package may depend on it without creating a cycle.
"""

from collections.abc import Callable, Iterator, Mapping
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


def _simulator_parameters() -> frozenset[str]:
    """Return the keyword names simulate_clusters accepts.

    Imported lazily so this module has no package-level import of carve.
    """
    import inspect

    from carve.sim import simulate_clusters

    return frozenset(inspect.signature(simulate_clusters).parameters)


@dataclass(frozen=True)
class Scenario:
    """One simulated experiment: an axis, its anchors, and an estimator.

    anchors maps each axis label to the simulate_clusters keyword arguments
    that define that point on the axis. shared holds the keyword arguments
    common to every point. A key may appear in one or the other, never both.
    """

    name: str
    axis: Axis
    anchors: Mapping[str, Mapping[str, Any]]
    shared: Mapping[str, Any]
    estimator: EstimatorSpec
    k_star: int = 5
    candidate_k: tuple[int, ...] = (3, 4, 5, 6, 7)
    n_seeds: int = 20

    def __post_init__(self) -> None:
        missing = [label for label in self.axis.labels if label not in self.anchors]
        if missing:
            raise ValueError(
                f"Scenario {self.name!r}: no anchor for axis label(s) {missing}."
            )

        valid = _simulator_parameters()

        for label, anchor in self.anchors.items():
            unknown = sorted(set(anchor) - valid)
            if unknown:
                raise ValueError(
                    f"Scenario {self.name!r}, anchor {label!r}: "
                    f"simulate_clusters does not accept {unknown}."
                )

        unknown_shared = sorted(set(self.shared) - valid - {self.axis.name})
        if unknown_shared:
            raise ValueError(
                f"Scenario {self.name!r}, shared settings: "
                f"simulate_clusters does not accept {unknown_shared}."
            )

        for label, anchor in self.anchors.items():
            overlap = sorted(set(anchor) & set(self.shared))
            if overlap:
                raise ValueError(
                    f"Scenario {self.name!r}: {overlap} appear in both the "
                    f"{label!r} anchor and the shared settings. Put each key in "
                    "exactly one of them."
                )

        if self.k_star not in self.candidate_k:
            raise ValueError(
                f"Scenario {self.name!r}: k_star={self.k_star} is not in "
                f"candidate_k={self.candidate_k}."
            )

    def sim_kwargs(self, axis_value: Any, axis_label: str) -> dict[str, Any]:
        """Build the simulate_clusters keyword arguments for one axis point.

        When the axis name is itself a simulator keyword (n_total, p,
        embed_dim), the axis value overrides whatever shared provides. When it
        is not (difficulty_level), the axis value only selects the anchor.
        """
        kwargs: dict[str, Any] = dict(self.shared)
        kwargs.update(self.anchors[axis_label])
        if self.axis.name in _simulator_parameters():
            kwargs[self.axis.name] = axis_value
        return kwargs


@dataclass(frozen=True)
class Study:
    """One case study: a real dataset run through the same runner.

    A Study has no axis. The runner treats it as a single cell, so the
    artifact schema is identical with axis_name, axis_value, and axis_label
    set to the study name, 0, and the study name respectively.
    """

    name: str
    loader: Callable[[], tuple[Any, Any, Mapping[str, Any]]]
    estimator: EstimatorSpec
    candidate_k: tuple[int, ...]
    k_star: int | None = None

    def __post_init__(self) -> None:
        if not self.candidate_k:
            raise ValueError(f"Study {self.name!r}: candidate_k must not be empty.")


@dataclass(frozen=True)
class Manifest:
    """Provenance for one run, written beside the artifact as manifest.json.

    peak_rss_unit is recorded explicitly because resource.getrusage reports
    ru_maxrss in bytes on macOS and kilobytes on Linux. A memory benchmark
    cannot carry a silent factor-of-1024 ambiguity between the machine an
    author ran it on and the machine CI ran it on.
    """

    run_id: str
    scenario: str
    config_hash: str
    anchor_set: str
    n_seeds: int
    n_resamples: int
    n_jobs: int
    random_state: int
    wall_clock_s: float
    peak_rss_bytes: int
    peak_rss_unit: str
    git_sha: str
    package_versions: Mapping[str, str]
    platform: Mapping[str, Any]
    config: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
