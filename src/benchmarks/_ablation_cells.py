"""Cells, units, seeds and file names of the rho/B ablation.

A cell is one CARVE fit keyed by (study, difficulty, dataset, rho, B,
replicate). The two arms are views over cells: the rho arm sweeps rho at
the default B, the B arm sweeps B at the default rho, and the cell at both
defaults belongs to both and is fitted once. Units of work are cells,
per-dataset records and per-(dataset, rho) similarity draws; each has its
own checkpoint file, named from its key, so resume is a file-existence
check and needs no filename parsing.

This module imports no carve code, so the notebook and the table can read
a run through it without paying for the runner's imports.
"""

import dataclasses
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from ._artifacts import CELL_KEY
from ._registry import (
    PUBLISHED_RANDOM_STATE,
    REPLICATE_SEED_SPACING,
    SCENARIOS,
    SIMILARITY_SEED_OFFSET,
)
from ._types import Ablation, AblationScale, Scenario

#: The study (Klein) has one dataset and no difficulty.
STUDY_DIFFICULTY: str = ""
STUDY_DATASET: int = 0

#: A similarity unit at this ratio refits the full data with the draw's
#: seed instead of subsampling; it separates the estimator's own
#: randomness from the effect of subsampling.
REFERENCE_RATIO: float = 1.0

ARMS: tuple[str, ...] = ("rho", "b")


class Cell(NamedTuple):
    """One CARVE fit."""

    study: str
    difficulty: str
    dataset: int
    subsample_ratio: float
    n_resamples: int
    replicate: int

    def key(self) -> dict[str, Any]:
        """The cell as a row fragment in CELL_KEY order."""
        return {
            "study": self.study,
            "difficulty": self.difficulty,
            "dataset": int(self.dataset),
            "subsample_ratio": float(self.subsample_ratio),
            "n_resamples": int(self.n_resamples),
            "replicate": int(self.replicate),
        }


class Unit(NamedTuple):
    """One unit of work. Dataset units zero rho, B and replicate; similarity
    units zero B and replicate."""

    kind: str
    cell: Cell


_UNIT_FRAMES: dict[str, tuple[str, ...]] = {
    "dataset": ("datasets",),
    "similarity": ("similarity",),
    "cell": ("curves", "selection", "at_k", "cells"),
}


def _benchmark_seed(dataset: int, axis_idx: int, random_state: int) -> int:
    # The published derivation, as _run.benchmark_seed; repeated here so
    # this module does not import the runner. A test pins the two equal.
    return int(dataset + (axis_idx * 10000) + random_state)


def _study_cell(ablation: Ablation, rho: float, b: int, replicate: int) -> Cell:
    return Cell(
        ablation.study, STUDY_DIFFICULTY, STUDY_DATASET, float(rho), int(b), replicate
    )


def enumerate_cells(
    ablation: Ablation, scale: str, arm: str | None = None
) -> list[Cell]:
    """Unique cells at a scale, in run order.

    Study cells first, then simulated cells grouped by dataset index, so a
    stopped run holds a balanced design over fewer datasets. arm=None gives
    the union of both arms; "rho" or "b" gives one arm's cells.
    """
    if arm is not None and arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS} or None.")
    sc = ablation.scales[scale]
    study: dict[Cell, None] = {}
    sims: dict[Cell, None] = {}

    def _extend(arm_scale, settings: Iterable[tuple[float, int]]) -> None:
        settings = list(settings)
        for replicate in range(arm_scale.study_replicates):
            for rho, b in settings:
                study.setdefault(_study_cell(ablation, rho, b, replicate))
        for name in ablation.scenarios:
            for difficulty in arm_scale.difficulties:
                for dataset in arm_scale.datasets:
                    for replicate in range(arm_scale.replicates):
                        for rho, b in settings:
                            sims.setdefault(
                                Cell(
                                    name,
                                    difficulty,
                                    int(dataset),
                                    float(rho),
                                    int(b),
                                    replicate,
                                )
                            )

    if arm in (None, "rho"):
        _extend(sc.rho_arm, ((rho, ablation.b_default) for rho in ablation.rho_grid))
    if arm in (None, "b"):
        _extend(sc.b_arm, ((ablation.rho_default, b) for b in ablation.b_grid))

    def _order(cell: Cell) -> tuple:
        scenario = SCENARIOS[cell.study]
        return (
            cell.dataset,
            ablation.scenarios.index(cell.study),
            scenario.axis.labels.index(cell.difficulty),
            cell.replicate,
            cell.subsample_ratio,
            cell.n_resamples,
        )

    return list(study) + sorted(sims, key=_order)


def enumerate_units(ablation: Ablation, scale: str) -> list[Unit]:
    """Every unit at a scale: per dataset, its record, its similarity draws
    at every rho of the rho grid plus the reference, then its cells."""
    cells = enumerate_cells(ablation, scale)
    rho_datasets = {
        (c.study, c.difficulty, c.dataset)
        for c in enumerate_cells(ablation, scale, arm="rho")
    }
    datasets = list(dict.fromkeys((c.study, c.difficulty, c.dataset) for c in cells))
    units: list[Unit] = []
    for study, difficulty, dataset in datasets:
        units.append(Unit("dataset", Cell(study, difficulty, dataset, 0.0, 0, 0)))
        if (study, difficulty, dataset) in rho_datasets:
            for rho in (*ablation.rho_grid, REFERENCE_RATIO):
                units.append(
                    Unit(
                        "similarity", Cell(study, difficulty, dataset, float(rho), 0, 0)
                    )
                )
        units.extend(
            Unit("cell", c)
            for c in cells
            if (c.study, c.difficulty, c.dataset) == (study, difficulty, dataset)
        )
    return units


def timing_units(ablation: Ablation, scale: str) -> list[Unit]:
    """A fixed batch of cells for choosing n_jobs: four study replicates at
    the defaults, and per scenario the default, smallest and largest rho at
    the last difficulty of the rho arm, dataset 0, replicate 0."""
    sc = ablation.scales[scale]
    difficulty = sc.rho_arm.difficulties[-1]
    dataset = sc.rho_arm.datasets[0]
    units: dict[Unit, None] = {}
    for replicate in range(4):
        units.setdefault(
            Unit(
                "cell",
                _study_cell(
                    ablation, ablation.rho_default, ablation.b_default, replicate
                ),
            )
        )
    for name in ablation.scenarios:
        for rho in (ablation.rho_default, ablation.rho_grid[0], ablation.rho_grid[-1]):
            units.setdefault(
                Unit(
                    "cell",
                    Cell(
                        name,
                        difficulty,
                        int(dataset),
                        float(rho),
                        ablation.b_default,
                        0,
                    ),
                )
            )
    return list(units)


def unit_stem(unit: Unit) -> str:
    c = unit.cell
    difficulty = c.difficulty or "na"
    return (
        f"{c.study}__{difficulty}__{c.dataset:04d}__rho{c.subsample_ratio:.3f}"
        f"__b{c.n_resamples:04d}__rep{c.replicate:02d}"
    )


def unit_paths(rd: Path, unit: Unit) -> dict[str, Path]:
    """The checkpoint file of each frame this unit writes."""
    stem = unit_stem(unit)
    return {
        name: Path(rd) / f"{name}__{stem}.parquet" for name in _UNIT_FRAMES[unit.kind]
    }


def unit_done(rd: Path, unit: Unit) -> bool:
    return all(path.exists() for path in unit_paths(rd, unit).values())


def dataset_seed(cell: Cell, *, ablation: Ablation) -> int:
    """The base seed: the benchmark seed of a simulated dataset, or the
    published random state for the study."""
    if cell.study == ablation.study:
        return PUBLISHED_RANDOM_STATE
    scenario = SCENARIOS[cell.study]
    axis_idx = scenario.axis.labels.index(cell.difficulty)
    return _benchmark_seed(cell.dataset, axis_idx, PUBLISHED_RANDOM_STATE)


def carve_seed(cell: Cell, *, ablation: Ablation) -> int:
    """CARVE's random_state for a cell. Replicate 0 is the published fit's seed."""
    return (
        dataset_seed(cell, ablation=ablation) + cell.replicate * REPLICATE_SEED_SPACING
    )


def similarity_seed(base: int, draw: int) -> int:
    return int(base + SIMILARITY_SEED_OFFSET + draw)


def seed_span(n_resamples: int) -> int:
    """How many consecutive seeds one fit uses: P_1, P_2 and the P_test pipeline."""
    return 3 * int(n_resamples)


def scenario_at_scale(scenario: Scenario, scale: AblationScale) -> Scenario:
    """The scenario with the scale's n_total override applied, if any."""
    if scale.n_total is None:
        return scenario
    return dataclasses.replace(
        scenario, shared={**scenario.shared, "n_total": int(scale.n_total)}
    )


def cells_frame(cells: Iterable[Cell]) -> pd.DataFrame:
    return pd.DataFrame([c.key() for c in cells], columns=list(CELL_KEY))


def arm_view(
    frames: Mapping[str, pd.DataFrame], *, ablation: Ablation, scale: str, arm: str
) -> dict[str, pd.DataFrame]:
    """Restrict the per-cell frames to one arm's cells.

    datasets and similarity pass through unchanged: they are keyed by
    dataset, not by cell.
    """
    keys = cells_frame(enumerate_cells(ablation, scale, arm=arm))
    view = dict(frames)
    for name in ("curves", "selection", "at_k", "cells"):
        frame = frames[name]
        # An empty frame (a schema with no checkpoint yet) has object
        # columns, and pandas refuses to merge object keys with numeric
        # ones. Nothing to restrict, so pass it through.
        view[name] = (
            frame if frame.empty else frame.merge(keys, on=list(CELL_KEY), how="inner")
        )
    return view
