"""Frozen dataclasses describing a benchmark experiment.

This module is a leaf: it imports only the standard library, so every other
module in the package may depend on it without creating a cycle.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

KNOWN_ESTIMATORS: frozenset[str] = frozenset(
    {
        "kmeans",
        "kmeans_n_init_100",
        "minibatch_kmeans",
        "agglomerative",
        "agglomerative_single",
        "spectral",
        "leiden",
        "louvain",
    }
)

#: Registry keys a PreprocessingSpec may name. _preprocessing maps each one to
#: its transformer; the names live here so this module stays a leaf.
KNOWN_PREPROCESSORS: frozenset[str] = frozenset(
    {"identity", "standard_scaler", "pca", "lsi", "tsne", "umap"}
)

#: Classifiers a Scenario may score generalizability with. _estimators maps
#: each one to its construction; the names live here so this module stays a
#: leaf. "random_forest" is CARVE's own default forest of n_trees trees.
KNOWN_CLASSIFIERS: frozenset[str] = frozenset({"random_forest", "lda"})


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

    params holds fixed hyperparameters as (name, value) pairs that override
    the registered defaults for this spec, so two specs can run one
    estimator at two settings (hECA's Leiden on 15 and on 50 neighbors). A
    tuple of pairs rather than a dict keeps the spec frozen and hashable.
    Whether each name is a parameter of the estimator is checked in
    _estimators, where the class is known; this module stays a leaf.
    """

    name: str
    params: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.name not in KNOWN_ESTIMATORS:
            raise ValueError(
                f"Unknown estimator {self.name!r}. "
                f"Valid names are {sorted(KNOWN_ESTIMATORS)}."
            )
        names = [key for key, _ in self.params]
        if len(set(names)) != len(names):
            raise ValueError(
                f"EstimatorSpec {self.name!r} sets a parameter more than once: {names}."
            )


@dataclass(frozen=True)
class PreprocessingSpec:
    """The preprocessing options a randomized case-study fit draws from.

    Each role holds (key, grid) pairs: key names a registered transformer and
    grid maps each of its hyperparameters to candidate values, one of which is
    drawn per resample. Options are named by key rather than by class so this
    module stays a leaf; _preprocessing.resolve_preprocessing turns them into
    the option lists CARVE takes. Validation is strict for the same reason
    EstimatorSpec's is.
    """

    normalization: tuple[tuple[str, Mapping[str, Sequence[Any]]], ...]
    dim_reduction: tuple[tuple[str, Mapping[str, Sequence[Any]]], ...]

    def __post_init__(self) -> None:
        for role in ("normalization", "dim_reduction"):
            options = getattr(self, role)
            if not options:
                raise ValueError(
                    f"PreprocessingSpec: {role} needs at least one option."
                )
            unknown = sorted({key for key, _ in options} - KNOWN_PREPROCESSORS)
            if unknown:
                raise ValueError(
                    f"PreprocessingSpec: unknown {role} option(s) {unknown}. "
                    f"Valid names are {sorted(KNOWN_PREPROCESSORS)}."
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

    classifier names the generalizability classifier. n_trees sizes the
    forest and is ignored under any other classifier.
    """

    name: str
    axis: Axis
    anchors: Mapping[str, Mapping[str, Any]]
    shared: Mapping[str, Any]
    estimator: EstimatorSpec
    k_star: int = 5
    candidate_k: tuple[int, ...] = (3, 4, 5, 6, 7)
    n_seeds: int = 20
    n_trees: int = 100
    classifier: str = "random_forest"

    def __post_init__(self) -> None:
        if self.classifier not in KNOWN_CLASSIFIERS:
            raise ValueError(
                f"Scenario {self.name!r}: unknown classifier {self.classifier!r}. "
                f"Valid names are {sorted(KNOWN_CLASSIFIERS)}."
            )

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

        unknown_shared = sorted(set(self.shared) - valid)
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
    """One case study: a real dataset run through the same pipeline.

    A Study has no axis. Its loader is parameterized by a subsample size, and
    each named scale's size (None for the whole dataset) is declared here
    rather than chosen at a call site.

    partners are the estimators swept alongside the study's own, in the
    order study_model_grids emits them. Declared here so the set of
    estimators a case study compares is part of its configuration rather
    than a branch on its name.

    A study declares candidate_k, resolutions or both, depending on which
    sweeps it runs. n_resamples is the resample count its CARVE fit runs,
    and preprocessing, when set, is the option set a randomized fit draws
    its pipelines from; both live here so a notebook reads them rather than
    restating them.

    not_two is the study's selection setting: True excludes k=2 from every
    selection made on its fits, as the Klein notebook does for the
    manuscript's headline result.

    reported_k is the k the manuscript reports for this study, read by
    figures that show a single k; None when the study reports none.

    map_option, when set, is one of preprocessing's dim_reduction options:
    the case-study figure fits it on every cell and draws its scatter panels
    on the result, so the map does not change with whichever pipeline a fit
    rates best.
    """

    name: str
    loader: Callable[[int | float | None], tuple[Any, Any, Mapping[str, Any]]]
    estimator: EstimatorSpec
    candidate_k: tuple[int, ...]
    scales: Mapping[str, int | float | None]
    default_scale: str
    partners: tuple[EstimatorSpec, ...] = (EstimatorSpec(name="spectral"),)
    resolutions: tuple[float, ...] = ()
    consensus_anchors: int | None = None
    k_star: int | None = None
    n_resamples: int = 100
    preprocessing: PreprocessingSpec | None = None
    not_two: bool = False
    reported_k: int | None = None
    map_option: tuple[str, dict[str, list[Any]]] | None = None

    def __post_init__(self) -> None:
        if not self.candidate_k and not self.resolutions:
            raise ValueError(
                f"Study {self.name!r}: declare candidate_k, resolutions or "
                "both; both are empty."
            )
        if not self.scales:
            raise ValueError(f"Study {self.name!r}: declare at least one scale.")
        if self.default_scale not in self.scales:
            raise ValueError(
                f"Study {self.name!r}: default_scale {self.default_scale!r} is "
                f"not among the declared scales {sorted(self.scales)}."
            )
        if self.map_option is not None and (
            self.preprocessing is None
            or self.map_option not in self.preprocessing.dim_reduction
        ):
            raise ValueError(
                f"Study {self.name!r}: map_option {self.map_option!r} is not "
                "one of the study's dim_reduction options."
            )


@dataclass(frozen=True)
class ArmScale:
    """How much of the grid one ablation arm runs at one scale.

    difficulties and datasets name the simulated cells; replicates is the
    number of CARVE fits with distinct seeds per simulated cell, and
    study_replicates the same for the case study, which has one dataset; it
    is 0 when the ablation has no case study.
    """

    difficulties: tuple[str, ...]
    datasets: tuple[int, ...]
    replicates: int
    study_replicates: int

    def __post_init__(self) -> None:
        if not self.difficulties:
            raise ValueError("ArmScale: declare at least one difficulty.")
        if not self.datasets:
            raise ValueError("ArmScale: declare at least one dataset.")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError(f"ArmScale: datasets repeat: {self.datasets}.")
        if self.replicates < 1:
            raise ValueError("ArmScale: replicates must be at least 1.")
        if self.study_replicates < 0:
            raise ValueError("ArmScale: study_replicates must be at least 0.")


@dataclass(frozen=True)
class AblationScale:
    """One scale of an ablation: how much runs, not what is measured.

    n_total overrides the simulated sample count; None keeps each
    scenario's own. study_scale names the case study's scale (a key of
    Study.scales), or is None when the ablation has no case study.
    """

    rho_arm: ArmScale
    b_arm: ArmScale
    similarity_draws: int
    study_scale: str | None = None
    n_total: int | None = None

    def __post_init__(self) -> None:
        if self.similarity_draws < 1:
            raise ValueError("AblationScale: similarity_draws must be at least 1.")
        if self.n_total is not None and self.n_total < 1:
            raise ValueError("AblationScale: n_total must be at least 1 or None.")


@dataclass(frozen=True)
class Ablation:
    """A sensitivity study over CARVE's subsampling proportion and resample count.

    The rho arm sweeps rho_grid at b_default; the B arm sweeps b_grid at
    rho_default. The defaults are fields so the configuration is complete
    on its own; the registry fills them from CARVE's own defaults. Scenario
    and study names are checked against the registry there, not here, so
    this module stays a leaf. study is None for an ablation over simulations
    only; its scales then run no study replicates and name no study scale.
    """

    name: str
    scenarios: tuple[str, ...]
    study: str | None
    rho_grid: tuple[float, ...]
    b_grid: tuple[int, ...]
    rho_default: float
    b_default: int
    scales: Mapping[str, AblationScale]
    default_scale: str

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError(f"Ablation {self.name!r}: declare at least one scenario.")
        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError(
                f"Ablation {self.name!r}: scenarios repeat: {self.scenarios}."
            )
        if list(self.rho_grid) != sorted(set(self.rho_grid)):
            raise ValueError(
                f"Ablation {self.name!r}: rho_grid must be strictly increasing."
            )
        if any(not 0.0 < rho < 1.0 for rho in self.rho_grid):
            raise ValueError(f"Ablation {self.name!r}: every rho must lie in (0, 1).")
        if list(self.b_grid) != sorted(set(self.b_grid)):
            raise ValueError(
                f"Ablation {self.name!r}: b_grid must be strictly increasing."
            )
        if any(b < 2 for b in self.b_grid):
            raise ValueError(f"Ablation {self.name!r}: every B must be at least 2.")
        if self.rho_default not in self.rho_grid:
            raise ValueError(
                f"Ablation {self.name!r}: rho_default {self.rho_default} is not in "
                f"rho_grid {self.rho_grid}."
            )
        if self.b_default not in self.b_grid:
            raise ValueError(
                f"Ablation {self.name!r}: b_default {self.b_default} is not in "
                f"b_grid {self.b_grid}."
            )
        if not self.scales:
            raise ValueError(f"Ablation {self.name!r}: declare at least one scale.")
        if self.default_scale not in self.scales:
            raise ValueError(
                f"Ablation {self.name!r}: default_scale {self.default_scale!r} is not "
                f"among the declared scales {sorted(self.scales)}."
            )
        for scale_name, scale in self.scales.items():
            where = f"Ablation {self.name!r}, scale {scale_name!r}"
            replicates = (scale.rho_arm.study_replicates, scale.b_arm.study_replicates)
            if self.study is None:
                if any(replicates):
                    raise ValueError(
                        f"{where}: study_replicates must be 0 without a case study."
                    )
                if scale.study_scale is not None:
                    raise ValueError(
                        f"{where}: study_scale must be None without a case study."
                    )
            else:
                if min(replicates) < 1:
                    raise ValueError(
                        f"{where}: study_replicates must be at least 1 with a case "
                        "study."
                    )
                if scale.study_scale is None:
                    raise ValueError(f"{where}: study_scale names no case study scale.")


@dataclass(frozen=True)
class Manifest:
    """Provenance for one run, written beside the artifact as manifest.json.

    peak_rss_unit is recorded explicitly because resource.getrusage reports
    ru_maxrss in bytes on macOS and kilobytes on Linux. A memory benchmark
    cannot carry a silent factor-of-1024 ambiguity between the machine an
    author ran it on and the machine CI ran it on.

    status is "running" while the pool is in flight and "complete" once every
    cell is done. It is written before the pool starts so an interrupted run
    still has a manifest: run directories are located by it, and a directory
    with checkpoints and no manifest cannot be opened at all.
    """

    run_id: str
    scenario: str
    config_hash: str
    status: str
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
