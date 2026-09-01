# Benchmarks Compute Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/benchmarks/` as an importable, CI-linted, tested package that runs every simulated benchmark scenario through one unified runner and writes a single versioned artifact schema, reproducing the committed results within tolerance.

**Architecture:** A frozen-dataclass registry describes each experiment as one `Scenario` sweeping one `Axis`. A single runner iterates (axis value x seed), simulates, fits an oracle estimator and CARVE, evaluates CARVE metrics and classical CVIs at every candidate k, and emits long-format rows. Artifacts are content-addressed parquet plus a `manifest.json`, checkpointed per cell so runs resume; a deliberate `promote()` step publishes a chosen run as committed CSV. Compute imports no matplotlib.

**Tech Stack:** Python 3.12+, numpy, pandas, scikit-learn, joblib, pyarrow, pytest, ruff 0.16.4, `carve` (editable install from this repo).

**Spec:** `docs/superpowers/specs/2026-09-01-benchmarks-rebuild-design.md`

**Deviation from the spec's layout:** the spec lists `SimParams` among the `_types.py` dataclasses.
This plan does not implement it. `Scenario.anchors` plus `Scenario.shared`, resolved by
`Scenario.sim_kwargs`, already carry every simulator argument, and a separate `SimParams` would be
a third place for the same keys to live and disagree. If a later plan needs a typed simulator
parameter object, it can be added without changing any signature defined here.

**Scope note:** This is plan 1 of 2. It covers the compute half of the spec's split-run-and-report decision: `_types`, `_registry`, `_estimators`, `_simulate`, `_cvi`, `_artifacts`, `_run`, `run.py`, `_calibrate`, `_tables`, the regression harness, and CI. Plan 2 covers the reporting half: `_theme`, `_panels`, `figures/`, `datasets/`, the notebooks, and the figure and table verification steps. Deleting `notebooks/benchmarking_code/` belongs to plan 2, after both halves are verified.

## Global Constraints

- Python `>=3.12`. Do not use syntax newer than 3.12.
- Ruff is pinned to `==0.16.4` from the `dev` extra. Use `.venv/bin/ruff`, never a system ruff.
- Ruff lint set is exactly `["E4", "E7", "E9", "F", "I", "UP"]`. Do not add rules.
- All commands run from `/Users/kaiwycik/GitHub/CARVE/code` using `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`.
- No `logging` anywhere. Diagnostics use `warnings.warn(..., stacklevel=2)` and `print()` gated on a `verbose` argument. Progress uses `tqdm.auto`, never `tqdm.notebook`.
- Nothing under `src/benchmarks/_*.py` may import `benchmarks.run`. `_types.py` imports nothing from the package.
- Compute modules (`_types`, `_registry`, `_estimators`, `_simulate`, `_cvi`, `_artifacts`, `_run`, `_calibrate`, `_tables`) must never import `matplotlib`.
- American spelling in prose. No bold or italics in authored files, including docstrings and Markdown.
- Artifact schema, exactly these columns in this order: `run_id, scenario, axis_name, axis_value, axis_label, seed, k_star, estimator, metric_name, k, metric_value, is_selected, selects_true_k, ari_at_k, oracle_ari`.
- Seed derivation, unchanged from the current code: `benchmark_seed = seed + axis_idx * 10000 + random_state`.
- The resample count is named `n_resamples` everywhere. Never `B`.
- CARVE is always constructed with `n_jobs=1`. Parallelism lives in the runner's outer loop only.
- Coverage stays scoped to `carve`; do not add `benchmarks` to `[tool.coverage.run] source`.
- Defaults carried over from the current benchmarks: `k_star=5`, `candidate_k=(3, 4, 5, 6, 7)`, `n_seeds=20`, `n_reference_datasets=10`, `n_init=10` for KMeans, `linkage="ward"` for agglomerative, `affinity="self_tuning"` for spectral.

---

### Task 1: Package scaffolding and packaging configuration

**Files:**
- Create: `src/benchmarks/__init__.py`
- Create: `src/benchmarks/README.md`
- Create: `tests/benchmarks/test_packaging.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `benchmarks` package exposing `benchmarks.__version__: str`. A `benchmarks` optional-dependency group providing `pyarrow`. A registered pytest marker `slow`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_packaging.py`:

```python
"""Packaging invariants for the benchmarks package.

benchmarks must be importable from an editable install (it lives under src/,
which the install's .pth file puts on sys.path) while staying out of the built
wheel (it is listed in the packages.find exclude list).
"""

import tomllib
from pathlib import Path

import benchmarks

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_benchmarks_is_importable_and_versioned():
    assert isinstance(benchmarks.__version__, str)
    assert benchmarks.__version__


def test_benchmarks_lives_under_src():
    assert Path(benchmarks.__file__).resolve().parent.parent.name == "src"


def test_benchmarks_is_excluded_from_the_wheel():
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    exclude = cfg["tool"]["setuptools"]["packages"]["find"]["exclude"]
    assert "benchmarks" in exclude
    assert "benchmarks.*" in exclude


def test_pyarrow_is_declared_in_the_benchmarks_extra():
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extra = cfg["project"]["optional-dependencies"]["benchmarks"]
    assert any(dep.startswith("pyarrow") for dep in extra)


def test_working_run_tree_is_gitignored():
    ignored = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "results/runs/" in [line.strip() for line in ignored]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_packaging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks'`

- [ ] **Step 3: Create the package**

Create `src/benchmarks/__init__.py`:

```python
"""Benchmarking and case-study code for the CARVE manuscript.

This package is not shipped in the carve-validate wheel. It is importable
only from a repo checkout with an editable install, which is also required
because carve.sim is excluded from the wheel.
"""

__version__ = "1.0.0"
```

Create `src/benchmarks/README.md`:

```markdown
# CARVE benchmarks

Reproduces the simulated benchmarks, tables, and figures for the CARVE
manuscript.

## Install

Benchmarks require a repo checkout with an editable install. A wheel install
is not enough: `carve.sim`, which generates the simulated data, is excluded
from the wheel, and this package is excluded too.

```bash
pip install -e ".[dev,benchmarks]"
```

The editable install writes a `.pth` file pointing at `code/src`, which is
what makes `import benchmarks` work from any working directory, including
`notebooks/`. If `import benchmarks` fails immediately after an editable
install, setuptools has switched from its static-path strategy to the strict
import finder; the fix is to add `benchmarks` back to
`[tool.setuptools.packages.find]` and exclude it from the wheel another way.

## Run

```bash
python -m benchmarks.run --scenario gaussians
python -m benchmarks.run --all --n-jobs 8
```

Working runs land in `results/runs/<scenario>/<config-hash>/` and are
gitignored. Publishing a run is a separate, deliberate step:

```bash
python -m benchmarks.run --promote results/runs/gaussians/<config-hash>
```
```

- [ ] **Step 4: Update pyproject.toml**

In `[tool.setuptools.packages.find]`, change the `exclude` line to:

```toml
exclude = ["carve.sim", "carve.sim.*", "benchmarks", "benchmarks.*"]
```

Add a new group at the end of `[project.optional-dependencies]`:

```toml
benchmarks = [
  "pyarrow>=17",
]
```

Extend `[tool.pytest.ini_options]` so the regression marker is registered:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "slow: regression runs that re-execute a full benchmark scenario",
]
```

- [ ] **Step 5: Update .gitignore**

Add below the existing `results/legacy/` line:

```
results/runs/
```

- [ ] **Step 6: Install and run the test to verify it passes**

Run: `.venv/bin/pip install -e ".[dev,benchmarks]" && .venv/bin/pytest tests/benchmarks/test_packaging.py -v`
Expected: 5 passed

- [ ] **Step 7: Verify the wheel really excludes benchmarks**

Run: `.venv/bin/python -m build --wheel -n -o /tmp/wheeltest . && .venv/bin/python -c "import zipfile,glob; print([n for n in zipfile.ZipFile(glob.glob('/tmp/wheeltest/*.whl')[0]).namelist() if n.startswith('benchmarks')])"`
Expected: `[]` — an empty list. If `build` is missing, install it first with `.venv/bin/pip install build`; it is deliberately not a declared dev dependency.

- [ ] **Step 8: Commit**

```bash
git add src/benchmarks/__init__.py src/benchmarks/README.md tests/benchmarks/test_packaging.py pyproject.toml .gitignore
git commit -m "feat(benchmarks): scaffold package under src/ excluded from wheel"
```

---

### Task 2: Core leaf types — Axis and EstimatorSpec

**Files:**
- Create: `src/benchmarks/_types.py`
- Create: `tests/benchmarks/test_types.py`

**Interfaces:**
- Consumes: nothing. `_types.py` is a leaf and imports only the standard library.
- Produces:
  - `KNOWN_ESTIMATORS: frozenset[str]` = `{"kmeans", "agglomerative", "spectral"}`
  - `Axis(name: str, values: tuple[Any, ...], labels: tuple[str, ...])`, frozen; `len(axis) -> int`; iterating yields `(index: int, value: Any, label: str)` triples
  - `EstimatorSpec(name: str)`, frozen, raises `ValueError` on an unknown name

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_types.py`:

```python
"""Tests for the leaf dataclasses in benchmarks._types."""

import pytest

from benchmarks._types import KNOWN_ESTIMATORS, Axis, EstimatorSpec


class TestAxis:
    def test_iterates_index_value_label_triples(self):
        axis = Axis(name="difficulty", values=(0, 1, 2), labels=("easy", "medium", "hard"))
        assert list(axis) == [(0, 0, "easy"), (1, 1, "medium"), (2, 2, "hard")]

    def test_length_is_the_number_of_points(self):
        axis = Axis(name="n_total", values=(1000, 5500, 10000), labels=("start", "middle", "end"))
        assert len(axis) == 3

    def test_rejects_mismatched_values_and_labels(self):
        with pytest.raises(ValueError, match="same length"):
            Axis(name="p", values=(1, 2, 3), labels=("a", "b"))

    def test_rejects_an_empty_axis(self):
        with pytest.raises(ValueError, match="at least one point"):
            Axis(name="p", values=(), labels=())

    def test_rejects_duplicate_labels(self):
        with pytest.raises(ValueError, match="duplicate"):
            Axis(name="p", values=(1, 2), labels=("a", "a"))

    def test_is_frozen(self):
        axis = Axis(name="p", values=(1,), labels=("a",))
        with pytest.raises(AttributeError):
            axis.name = "q"


class TestEstimatorSpec:
    @pytest.mark.parametrize("name", sorted(KNOWN_ESTIMATORS))
    def test_accepts_every_known_estimator(self, name):
        assert EstimatorSpec(name=name).name == name

    def test_raises_on_an_unknown_name(self):
        with pytest.raises(ValueError, match="aglomerative"):
            EstimatorSpec(name="aglomerative")

    def test_error_lists_the_valid_names(self):
        with pytest.raises(ValueError, match="kmeans"):
            EstimatorSpec(name="nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._types'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_types.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_types.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_types.py tests/benchmarks/test_types.py
git commit -m "feat(benchmarks): add Axis and EstimatorSpec leaf types"
```

---

### Task 3: Scenario with signature-validated anchors

**Files:**
- Modify: `src/benchmarks/_types.py`
- Modify: `tests/benchmarks/test_types.py`

**Interfaces:**
- Consumes: `Axis`, `EstimatorSpec` from Task 2.
- Produces: `Scenario(name, axis, anchors, shared, estimator, k_star=5, candidate_k=(3,4,5,6,7), n_seeds=20)`, frozen, validating anchor and shared keys against `inspect.signature(simulate_clusters)` and rejecting keys present in both. Exposes `Scenario.sim_kwargs(axis_value, axis_label) -> dict[str, Any]`.

Note on the dependency rule: `_types` stays a leaf for package imports, but this validation needs the simulator's signature. Import `carve.sim.simulate_clusters` lazily inside `__post_init__`, not at module scope, so `_types` still imports with only the standard library at module level.

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmarks/test_types.py`:

```python
from benchmarks._types import Scenario


def _axis():
    return Axis(name="difficulty_level", values=(0, 1, 2), labels=("easy", "medium", "hard"))


def _scenario(**overrides):
    kwargs = dict(
        name="demo",
        axis=_axis(),
        anchors={
            "easy": {"cluster_scale": 1.0},
            "medium": {"cluster_scale": 2.0},
            "hard": {"cluster_scale": 3.0},
        },
        shared={"n_total": 120, "p": 4, "distribution": "gaussian"},
        estimator=EstimatorSpec(name="kmeans"),
    )
    kwargs.update(overrides)
    return Scenario(**kwargs)


class TestScenario:
    def test_accepts_a_valid_scenario(self):
        assert _scenario().name == "demo"

    def test_defaults_match_the_published_benchmark(self):
        s = _scenario()
        assert s.k_star == 5
        assert s.candidate_k == (3, 4, 5, 6, 7)
        assert s.n_seeds == 20

    def test_rejects_an_anchor_key_the_simulator_does_not_accept(self):
        with pytest.raises(ValueError, match="cluster_scal"):
            _scenario(anchors={
                "easy": {"cluster_scal": 1.0},
                "medium": {"cluster_scale": 2.0},
                "hard": {"cluster_scale": 3.0},
            })

    def test_rejects_a_shared_key_the_simulator_does_not_accept(self):
        with pytest.raises(ValueError, match="n_totl"):
            _scenario(shared={"n_totl": 120})

    def test_rejects_a_key_present_in_both_anchors_and_shared(self):
        with pytest.raises(ValueError, match="both"):
            _scenario(shared={"n_total": 120, "cluster_scale": 9.0})

    def test_requires_an_anchor_for_every_axis_label(self):
        with pytest.raises(ValueError, match="hard"):
            _scenario(anchors={"easy": {"cluster_scale": 1.0}, "medium": {"cluster_scale": 2.0}})

    def test_sim_kwargs_merges_shared_and_the_selected_anchor(self):
        kwargs = _scenario().sim_kwargs(axis_value=1, axis_label="medium")
        assert kwargs["cluster_scale"] == 2.0
        assert kwargs["n_total"] == 120

    def test_sim_kwargs_sets_the_axis_parameter_when_the_axis_is_a_sim_kwarg(self):
        scenario = _scenario(
            axis=Axis(name="n_total", values=(1000, 5500), labels=("start", "end")),
            anchors={"start": {"cluster_scale": 1.0}, "end": {"cluster_scale": 2.0}},
            shared={"p": 4},
        )
        assert scenario.sim_kwargs(axis_value=5500, axis_label="end")["n_total"] == 5500

    def test_sim_kwargs_ignores_the_axis_value_for_a_non_sim_axis(self):
        scenario = _scenario(
            axis=_axis(),
            shared={"n_total": 120, "p": 4},
        )
        assert "difficulty_level" not in scenario.sim_kwargs(axis_value=0, axis_label="easy")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_types.py -k Scenario -v`
Expected: FAIL with `ImportError: cannot import name 'Scenario'`

- [ ] **Step 3: Write the implementation**

Add to `src/benchmarks/_types.py`, after `EstimatorSpec`:

```python
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
```

Add `Mapping` to the imports at the top of the file:

```python
from collections.abc import Iterator, Mapping
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_types.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_types.py tests/benchmarks/test_types.py
git commit -m "feat(benchmarks): add Scenario with signature-validated anchors"
```

---

### Task 4: Study and Manifest

**Files:**
- Modify: `src/benchmarks/_types.py`
- Modify: `tests/benchmarks/test_types.py`

**Interfaces:**
- Consumes: `EstimatorSpec` from Task 2.
- Produces:
  - `Study(name, loader, estimator, candidate_k, k_star=None)`, frozen. `loader` is a zero-argument callable returning `(X, y, meta)`. Runs through the same runner with `axis=None`.
  - `Manifest`, frozen, with `to_dict() -> dict[str, Any]` producing JSON-serializable output.

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmarks/test_types.py`:

```python
import numpy as np

from benchmarks._types import Manifest, Study


class TestStudy:
    def test_holds_a_loader_and_an_estimator(self):
        def loader():
            return np.zeros((4, 2)), np.array([0, 0, 1, 1]), {"source": "test"}

        study = Study(
            name="demo",
            loader=loader,
            estimator=EstimatorSpec(name="kmeans"),
            candidate_k=(2, 3),
        )
        X, y, meta = study.loader()
        assert X.shape == (4, 2)
        assert meta["source"] == "test"
        assert study.k_star is None

    def test_rejects_empty_candidate_k(self):
        with pytest.raises(ValueError, match="candidate_k"):
            Study(
                name="demo",
                loader=lambda: (np.zeros((2, 2)), np.zeros(2), {}),
                estimator=EstimatorSpec(name="kmeans"),
                candidate_k=(),
            )


class TestManifest:
    def test_to_dict_is_json_serializable(self):
        import json

        manifest = Manifest(
            run_id="abc123",
            scenario="demo",
            config_hash="deadbeef0000",
            anchor_set="ACTIVE_ANCHORS",
            n_seeds=2,
            n_resamples=3,
            n_jobs=1,
            random_state=0,
            wall_clock_s=1.5,
            peak_rss_bytes=1024,
            peak_rss_unit="bytes",
            git_sha="0" * 40,
            package_versions={"numpy": "2.0.0"},
            platform={"system": "Darwin", "cpu": "arm64", "cores": 10},
            config={"k_star": 5},
        )
        payload = json.dumps(manifest.to_dict())
        assert "deadbeef0000" in payload

    def test_records_the_rss_unit_explicitly(self):
        manifest = Manifest(
            run_id="a",
            scenario="s",
            config_hash="h",
            anchor_set="ACTIVE_ANCHORS",
            n_seeds=1,
            n_resamples=1,
            n_jobs=1,
            random_state=0,
            wall_clock_s=0.0,
            peak_rss_bytes=1,
            peak_rss_unit="bytes",
            git_sha="",
            package_versions={},
            platform={},
            config={},
        )
        assert manifest.to_dict()["peak_rss_unit"] == "bytes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_types.py -k "Study or Manifest" -v`
Expected: FAIL with `ImportError: cannot import name 'Manifest'`

- [ ] **Step 3: Write the implementation**

Add to `src/benchmarks/_types.py`:

```python
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
```

Extend the imports at the top of the file:

```python
from collections.abc import Callable, Iterator, Mapping
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_types.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_types.py tests/benchmarks/test_types.py
git commit -m "feat(benchmarks): add Study and Manifest types"
```

---

### Task 5: Strict estimator construction

**Files:**
- Create: `src/benchmarks/_estimators.py`
- Create: `tests/benchmarks/test_estimators.py`

**Interfaces:**
- Consumes: `EstimatorSpec`, `KNOWN_ESTIMATORS` from Task 2.
- Produces:
  - `ESTIMATOR_CLASSES: dict[str, type[ClusterMixin]]`
  - `ESTIMATOR_DEFAULTS: dict[str, dict[str, Any]]`
  - `build_estimator(spec: EstimatorSpec, n_clusters: int, random_state: int) -> ClusterMixin`
  - `param_grids(spec: EstimatorSpec, candidate_k: Sequence[int]) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]` — the structure CARVE takes as `estimator_param_grids`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_estimators.py`:

```python
"""Tests for estimator construction."""

import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans

from benchmarks._estimators import (
    ESTIMATOR_CLASSES,
    build_estimator,
    param_grids,
)
from benchmarks._types import KNOWN_ESTIMATORS, EstimatorSpec


def test_every_known_estimator_has_a_class():
    assert set(ESTIMATOR_CLASSES) == set(KNOWN_ESTIMATORS)


class TestBuildEstimator:
    def test_kmeans_gets_the_pinned_n_init(self):
        est = build_estimator(EstimatorSpec(name="kmeans"), n_clusters=5, random_state=7)
        assert isinstance(est, KMeans)
        assert est.n_clusters == 5
        assert est.n_init == 10

    def test_random_state_is_set_when_the_estimator_supports_it(self):
        est = build_estimator(EstimatorSpec(name="kmeans"), n_clusters=3, random_state=42)
        assert est.random_state == 42

    def test_random_state_is_omitted_when_unsupported(self):
        est = build_estimator(
            EstimatorSpec(name="agglomerative"), n_clusters=3, random_state=42
        )
        assert isinstance(est, AgglomerativeClustering)
        assert not hasattr(est, "random_state")

    def test_agglomerative_uses_ward_linkage(self):
        est = build_estimator(EstimatorSpec(name="agglomerative"), n_clusters=3, random_state=0)
        assert est.linkage == "ward"

    def test_spectral_uses_self_tuning_affinity(self):
        est = build_estimator(EstimatorSpec(name="spectral"), n_clusters=3, random_state=0)
        assert est.affinity == "self_tuning"


class TestParamGrids:
    def test_shape_matches_what_carve_expects(self):
        grids = param_grids(EstimatorSpec(name="kmeans"), candidate_k=(3, 4, 5))
        assert len(grids) == 1
        cls, grid = grids[0]
        assert cls is KMeans
        assert grid["n_clusters"] == [3, 4, 5]
        assert grid["n_init"] == [10]

    def test_agglomerative_grid_carries_linkage(self):
        _, grid = param_grids(EstimatorSpec(name="agglomerative"), candidate_k=(2, 3))[0]
        assert grid["linkage"] == ["ward"]

    def test_rejects_empty_candidate_k(self):
        with pytest.raises(ValueError, match="candidate_k"):
            param_grids(EstimatorSpec(name="kmeans"), candidate_k=())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_estimators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._estimators'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_estimators.py`:

```python
"""Construction of the base clustering estimators used by the benchmarks.

The literals that used to be scattered through the old estimator factory
(n_init=10, ward linkage, self-tuning affinity) live here as one table.
"""

import inspect
from collections.abc import Sequence
from typing import Any

from carve.cluster import SpectralClustering
from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans

from ._types import EstimatorSpec

ESTIMATOR_CLASSES: dict[str, type[ClusterMixin]] = {
    "kmeans": KMeans,
    "agglomerative": AgglomerativeClustering,
    "spectral": SpectralClustering,
}

ESTIMATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    # n_init is pinned so results do not move when scikit-learn changes its
    # default, which it has done before.
    "kmeans": {"n_init": 10},
    "agglomerative": {"linkage": "ward"},
    "spectral": {"affinity": "self_tuning"},
}


def build_estimator(
    spec: EstimatorSpec, n_clusters: int, random_state: int
) -> ClusterMixin:
    """Instantiate the estimator for one scenario at one k.

    random_state is passed only when the estimator's signature accepts it,
    probed rather than hardcoded so a swapped estimator class does not raise.
    """
    estimator_cls = ESTIMATOR_CLASSES[spec.name]
    params: dict[str, Any] = dict(ESTIMATOR_DEFAULTS[spec.name])
    params["n_clusters"] = int(n_clusters)

    signature = inspect.signature(estimator_cls.__init__)
    if "random_state" in signature.parameters:
        params["random_state"] = int(random_state)

    return estimator_cls(**params)


def param_grids(
    spec: EstimatorSpec, candidate_k: Sequence[int]
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """Build the estimator_param_grids structure CARVE takes.

    One entry, sweeping n_clusters over the candidate values, with the
    estimator's fixed defaults alongside.
    """
    if not candidate_k:
        raise ValueError("candidate_k must not be empty.")

    grid: dict[str, list[Any]] = {"n_clusters": [int(k) for k in candidate_k]}
    for key, value in ESTIMATOR_DEFAULTS[spec.name].items():
        grid[key] = [value]

    return [(ESTIMATOR_CLASSES[spec.name], grid)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_estimators.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_estimators.py tests/benchmarks/test_estimators.py
git commit -m "feat(benchmarks): add strict estimator construction"
```

---

### Task 6: The registry — metric constants, anchors, and scenarios

**Files:**
- Create: `src/benchmarks/_registry.py`
- Create: `tests/benchmarks/test_registry.py`
- Read for data: `notebooks/Benchmarking.ipynb` cells 9, 15, 21, 27, 33, 39, 46, 51

**Interfaces:**
- Consumes: `Axis`, `EstimatorSpec`, `Scenario` from Tasks 2 and 3.
- Produces:
  - `CARVE_METRICS_STABILITY`, `CARVE_METRICS_GENERALIZABILITY`, `CARVE_METRICS_COMBINED`, `CARVE_METRICS_ALL`, `CVI_METRICS` — all `tuple[str, ...]`
  - `GENERALIZABILITY_METRICS: frozenset[str]`
  - `METRIC_DISPLAY_NAMES: dict[str, str]`
  - `metric_rule(name: str) -> str`, `metric_measure(name: str) -> str`
  - `DIFFICULTY_AXIS: Axis`, `SCALING_AXES: dict[str, Axis]`
  - `PUBLISHED_ANCHORS: dict[str, dict[str, dict[str, Any]]]`, `ACTIVE_ANCHORS`, `ACTIVE_ANCHOR_SET_NAME: str`
  - `SCENARIOS: dict[str, Scenario]`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_registry.py`:

```python
"""Tests for the experiment registry."""

import pytest

from benchmarks._registry import (
    ACTIVE_ANCHOR_SET_NAME,
    ACTIVE_ANCHORS,
    CARVE_METRICS_ALL,
    CVI_METRICS,
    DIFFICULTY_AXIS,
    GENERALIZABILITY_METRICS,
    PUBLISHED_ANCHORS,
    SCALING_AXES,
    SCENARIOS,
    metric_measure,
    metric_rule,
)

DIFFICULTY_SCENARIOS = (
    "gaussians",
    "t_dist",
    "t_dist_noise",
    "circles",
    "moons",
    "swiss_rolls",
)
SCALING_SCENARIOS = ("gaussians_dimensionality", "gaussians_samples")


class TestMetricNames:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("ari_stability", "max"),
            ("ari_stability_1se", "1se"),
            ("ari_stability_quant", "quantile"),
        ],
    )
    def test_rule_is_parsed_from_the_suffix(self, name, expected):
        assert metric_rule(name) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("ari_stability", "ari_stability"),
            ("ari_stability_1se", "ari_stability"),
            ("ari_stability_quant", "ari_stability"),
        ],
    )
    def test_measure_strips_the_suffix(self, name, expected):
        assert metric_measure(name) == expected

    def test_carve_metrics_are_unique_and_sorted(self):
        assert list(CARVE_METRICS_ALL) == sorted(set(CARVE_METRICS_ALL))

    def test_cvi_metrics_are_the_four_classical_indices(self):
        assert set(CVI_METRICS) == {
            "silhouette",
            "gap",
            "davies_bouldin",
            "calinski_harabasz",
        }

    def test_generalizability_metrics_are_a_subset_of_all_carve_metrics(self):
        assert GENERALIZABILITY_METRICS <= set(CARVE_METRICS_ALL)

    def test_generalizability_metrics_are_exactly_those_named_generalizability(self):
        assert GENERALIZABILITY_METRICS == {
            m for m in CARVE_METRICS_ALL if "generalizability" in m
        }


class TestAxes:
    def test_difficulty_axis_matches_the_published_labels(self):
        assert DIFFICULTY_AXIS.labels == ("easy", "medium", "hard")
        assert DIFFICULTY_AXIS.values == (0, 1, 2)
        assert DIFFICULTY_AXIS.name == "difficulty_level"

    def test_scaling_axes_use_the_published_stage_labels(self):
        for axis in SCALING_AXES.values():
            assert axis.labels == ("start", "middle", "end")

    def test_scaling_ranges_match_the_published_config(self):
        assert SCALING_AXES["n_total"].values == (1000, 5500, 10000)
        assert SCALING_AXES["embed_dim"].values == (10, 255, 500)

    def test_moons_has_no_scaling_scenario(self):
        assert not any("moons" in name for name in SCALING_SCENARIOS)
        assert not any(
            name.startswith("moons_") for name in SCENARIOS if name not in DIFFICULTY_SCENARIOS
        )


class TestAnchors:
    def test_published_anchors_are_retained_for_every_scenario(self):
        assert set(PUBLISHED_ANCHORS) == set(SCENARIOS)

    def test_active_anchors_are_named_so_provenance_is_unambiguous(self):
        assert ACTIVE_ANCHOR_SET_NAME in {"PUBLISHED_ANCHORS", "CALIBRATED_ANCHORS"}

    def test_active_anchors_default_to_the_published_set(self):
        assert ACTIVE_ANCHORS is PUBLISHED_ANCHORS
        assert ACTIVE_ANCHOR_SET_NAME == "PUBLISHED_ANCHORS"


class TestScenarios:
    def test_every_expected_scenario_is_registered(self):
        assert set(SCENARIOS) == set(DIFFICULTY_SCENARIOS) | set(SCALING_SCENARIOS)

    @pytest.mark.parametrize("name", DIFFICULTY_SCENARIOS + SCALING_SCENARIOS)
    def test_scenario_name_matches_its_key(self, name):
        assert SCENARIOS[name].name == name

    @pytest.mark.parametrize("name", DIFFICULTY_SCENARIOS + SCALING_SCENARIOS)
    def test_every_scenario_constructs_and_validates(self, name):
        scenario = SCENARIOS[name]
        assert scenario.k_star == 5
        assert scenario.candidate_k == (3, 4, 5, 6, 7)
        assert scenario.n_seeds == 20

    def test_swiss_rolls_uses_the_estimator_the_benchmark_actually_ran(self):
        assert SCENARIOS["swiss_rolls"].estimator.name == "agglomerative"

    def test_gaussians_uses_kmeans(self):
        assert SCENARIOS["gaussians"].estimator.name == "kmeans"

    def test_t_dist_uses_agglomerative(self):
        assert SCENARIOS["t_dist"].estimator.name == "agglomerative"

    @pytest.mark.parametrize("name", DIFFICULTY_SCENARIOS)
    def test_difficulty_scenarios_sweep_the_difficulty_axis(self, name):
        assert SCENARIOS[name].axis.name == "difficulty_level"

    def test_sim_kwargs_resolve_for_every_scenario_and_label(self):
        for scenario in SCENARIOS.values():
            for _, value, label in scenario.axis:
                kwargs = scenario.sim_kwargs(value, label)
                assert "k" not in kwargs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._registry'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_registry.py`. The metric constants and helpers are ports of `benchmarking_config.py` and `benchmarking_utils.py:77,97`:

```python
"""The study design, in code.

Every benchmark scenario, its axis, its calibrated anchors, and the metric
vocabulary live here. Adding an experiment means adding a Scenario, not
writing a runner.
"""

from typing import Any

from ._types import Axis, EstimatorSpec, Scenario

# =============================================================================
# Metric vocabulary
# =============================================================================
CARVE_METRICS_STABILITY: tuple[str, ...] = (
    "ari_stability",
    "ari_stability_1se",
    "ari_stability_quant",
    "consensus_pac_stability",
    "consensus_gini_stability",
    "consensus_ce_stability",
)

CARVE_METRICS_GENERALIZABILITY: tuple[str, ...] = (
    "ari_generalizability",
    "ari_generalizability_1se",
    "ari_generalizability_quant",
    "accuracy_generalizability",
)

CARVE_METRICS_COMBINED: tuple[str, ...] = (
    "ari_average",
    "ari_average_1se",
    "ari_average_quant",
)

CARVE_METRICS_ALL: tuple[str, ...] = tuple(
    sorted(
        set(
            CARVE_METRICS_STABILITY
            + CARVE_METRICS_GENERALIZABILITY
            + CARVE_METRICS_COMBINED
        )
    )
)

CVI_METRICS: tuple[str, ...] = (
    "silhouette",
    "gap",
    "davies_bouldin",
    "calinski_harabasz",
)

# Which CARVE consensus matrix a metric's labels must come from. The old
# difficulty runner never passed mode= to get_labels, so "default" resolved to
# run_stability=True and every metric's ari_at_k came from the stability
# matrix, including the generalizability metrics.
GENERALIZABILITY_METRICS: frozenset[str] = frozenset(
    m for m in CARVE_METRICS_ALL if "generalizability" in m
)

METRIC_DISPLAY_NAMES: dict[str, str] = {
    "baseline_oracle": "Baseline (Oracle)",
    "ari_stability": "ARI (stab, max)",
    "ari_stability_1se": "CARVE Stability (1SE)",
    "ari_stability_quant": "ARI (stab, quantile)",
    "ari_generalizability": "ARI (gen, max)",
    "ari_generalizability_1se": "CARVE Generalizability (1SE)",
    "ari_generalizability_quant": "ARI (gen, quantile)",
    "ari_average": "ARI (avg, max)",
    "ari_average_1se": "ARI (avg, 1SE)",
    "ari_average_quant": "ARI (avg, quantile)",
    "consensus_pac_stability": "PAC (stab)",
    "consensus_gini_stability": "Gini (stab)",
    "consensus_ce_stability": "CE (stab)",
    "accuracy_generalizability": "Accuracy (gen)",
    "silhouette": "Silhouette",
    "gap": "Gap Statistic",
    "davies_bouldin": "Davies-Bouldin",
    "calinski_harabasz": "Calinski-Harabasz",
}

N_REFERENCE_DATASETS: int = 10


def metric_rule(name: str) -> str:
    """Return the CARVE selection rule encoded in a metric name's suffix."""
    if name.endswith("_quant"):
        return "quantile"
    if name.endswith("_1se"):
        return "1se"
    return "max"


def metric_measure(name: str) -> str:
    """Return the CARVE measure name with any selection-rule suffix removed."""
    if name.endswith("_1se"):
        return name[:-4]
    if name.endswith("_quant"):
        return name[:-6]
    return name


# =============================================================================
# Axes
# =============================================================================
DIFFICULTY_AXIS = Axis(
    name="difficulty_level",
    values=(0, 1, 2),
    labels=("easy", "medium", "hard"),
)

# Ranges and the three-point granularity are ports of SCALING_RANGES and
# GRANULARITY in benchmarking_config.py, evaluated to literals so the axis is
# readable without running numpy.linspace in your head.
SCALING_AXES: dict[str, Axis] = {
    "n_total": Axis(
        name="n_total", values=(1000, 5500, 10000), labels=("start", "middle", "end")
    ),
    "p": Axis(name="p", values=(50, 525, 1000), labels=("start", "middle", "end")),
    "embed_dim": Axis(
        name="embed_dim", values=(10, 255, 500), labels=("start", "middle", "end")
    ),
}
```

- [ ] **Step 4: Port the anchor dictionaries**

Copy the eight dicts verbatim from `notebooks/Benchmarking.ipynb` into `_registry.py`. Cell numbers and variable names:

| Notebook cell | Variables | Registry key |
|---|---|---|
| 9 | `anchor_settings_gaussians`, `other_settings_gaussians` | `gaussians` |
| 15 | `anchor_settings_t_dist`, `other_settings_t_dist` | `t_dist` |
| 21 | `anchor_settings_t_dist_noise`, `other_settings_t_dist_noise` | `t_dist_noise` |
| 27 | `anchor_settings_circles`, `other_settings_circles` | `circles` |
| 33 | `anchor_settings_moons`, `other_settings_moons` | `moons` |
| 39 | `anchor_settings_swiss_rolls`, `other_settings_swiss_rolls` | `swiss_rolls` |
| 46 | `anchor_settings_gaussians_dimensionality`, `other_settings_gaussians_dimensionality` | `gaussians_dimensionality` |
| 51 | `anchor_settings_gaussians_samples`, `other_settings_gaussians_samples` | `gaussians_samples` |

Do not port `results_moons_dimensionality`; the moons scaling experiment is dropped.

Extract them with this command rather than retyping, to rule out transcription error:

```bash
.venv/bin/python - <<'PY'
import json
nb = json.load(open("notebooks/Benchmarking.ipynb"))
for i in (9, 15, 21, 27, 33, 39, 46, 51):
    print(f"# ----- cell {i} -----")
    print("".join(nb["cells"][i]["source"]))
PY
```

Structure them into two module-level constants. `PUBLISHED_ANCHORS` maps scenario name to the anchor dict; a parallel `_SHARED` maps scenario name to the `other_settings` dict. The moons entry, as an example of the exact shape:

```python
PUBLISHED_ANCHORS: dict[str, dict[str, dict[str, Any]]] = {
    # ... other scenarios ...
    "moons": {
        "easy": {
            "cluster_scale": [5.5, 3.97, 3.97, 3.97, 3.97],
            "cluster_size_dirichlet_alpha": 0.67,
            "corr_strength": 0.39,
            "embed_param": 10.7,
        },
        "medium": {
            "cluster_scale": [4.8, 4.06, 4.06, 4.06, 4.06],
            "cluster_size_dirichlet_alpha": 0.57,
            "corr_strength": 0.30,
            "embed_param": 5.7,
        },
        "hard": {
            "cluster_scale": [4.06, 2.65, 2.65, 2.65, 2.65],
            "cluster_size_dirichlet_alpha": 0.10,
            "corr_strength": 0.16,
            "embed_param": 14.5,
        },
    },
}

_SHARED: dict[str, dict[str, Any]] = {
    # ... other scenarios ...
    "moons": {
        "distribution": "moons",
        "nonlinear": True,
        "n_total": 1500,
        "p": 50,
        "corr_type": "ar1",
        "embed_dim": 64,
    },
}
```

For the two scaling scenarios, drop `n_total`, `p`, and `embed_dim` from `_SHARED` when that key is the scenario's axis name — `Scenario.sim_kwargs` sets it from the axis value, and leaving it in `shared` as well is harmless but misleading. Keep the other two constants, whose published values are `n_total=1500`, `p=50`, `embed_dim=64` (`SCALING_CONSTANTS` in `benchmarking_config.py`).

- [ ] **Step 5: Add the anchor-set selector and the scenario table**

Append to `_registry.py`:

```python
# The calibration notebook that produced PUBLISHED_ANCHORS was lost; see
# _calibrate.py, which defines a documented replacement search. Both sets are
# kept so a regenerated calibration can be compared against what the
# manuscript reported, and reverted to. ACTIVE_ANCHOR_SET_NAME is recorded in
# every run manifest so no artifact is ambiguous about which produced it.
ACTIVE_ANCHORS: dict[str, dict[str, dict[str, Any]]] = PUBLISHED_ANCHORS
ACTIVE_ANCHOR_SET_NAME: str = "PUBLISHED_ANCHORS"

_ESTIMATORS: dict[str, str] = {
    "gaussians": "kmeans",
    "t_dist": "agglomerative",
    "t_dist_noise": "agglomerative",
    "circles": "spectral",
    "moons": "spectral",
    # The published S1 Fig panel was drawn with "spectral", but the benchmark
    # that produced Table S7 ran agglomerative, and S3 Text says Ward. The
    # results were correct; only the illustration disagreed. Agglomerative is
    # what the numbers came from.
    "swiss_rolls": "agglomerative",
    "gaussians_dimensionality": "kmeans",
    "gaussians_samples": "kmeans",
}

_AXES: dict[str, Axis] = {
    "gaussians": DIFFICULTY_AXIS,
    "t_dist": DIFFICULTY_AXIS,
    "t_dist_noise": DIFFICULTY_AXIS,
    "circles": DIFFICULTY_AXIS,
    "moons": DIFFICULTY_AXIS,
    "swiss_rolls": DIFFICULTY_AXIS,
    "gaussians_dimensionality": SCALING_AXES["embed_dim"],
    "gaussians_samples": SCALING_AXES["n_total"],
}

SCENARIOS: dict[str, Scenario] = {
    name: Scenario(
        name=name,
        axis=_AXES[name],
        anchors=ACTIVE_ANCHORS[name],
        shared=_SHARED[name],
        estimator=EstimatorSpec(name=_ESTIMATORS[name]),
    )
    for name in _ESTIMATORS
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_registry.py -v`
Expected: all passed. A failure in `test_every_scenario_constructs_and_validates` means an anchor key was transcribed wrong; the error message names the scenario, the anchor, and the offending key.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_registry.py tests/benchmarks/test_registry.py
git commit -m "feat(benchmarks): add scenario registry with published anchors"
```

---

### Task 7: Simulation

**Files:**
- Create: `src/benchmarks/_simulate.py`
- Create: `tests/benchmarks/test_simulate.py`

**Interfaces:**
- Consumes: `Scenario` from Task 3.
- Produces: `simulate(scenario: Scenario, axis_value: Any, axis_label: str, seed: int) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_simulate.py`:

```python
"""Tests for the simulation wrapper."""

import numpy as np

from benchmarks._simulate import simulate
from benchmarks._types import Axis, EstimatorSpec, Scenario


def _scenario(axis=None, anchors=None, shared=None):
    return Scenario(
        name="demo",
        axis=axis or Axis(name="difficulty_level", values=(0, 1), labels=("easy", "hard")),
        anchors=anchors or {"easy": {"cluster_scale": 1.0}, "hard": {"cluster_scale": 4.0}},
        shared=shared or {"n_total": 150, "p": 4, "distribution": "gaussian"},
        estimator=EstimatorSpec(name="kmeans"),
    )


class TestSimulate:
    def test_returns_matching_x_and_y_shapes(self):
        X, y = simulate(_scenario(), axis_value=0, axis_label="easy", seed=0)
        assert X.shape == (150, 4)
        assert y.shape == (150,)

    def test_simulates_k_star_clusters(self):
        _, y = simulate(_scenario(), axis_value=0, axis_label="easy", seed=0)
        assert len(np.unique(y)) == 5

    def test_is_deterministic_for_a_fixed_seed(self):
        a, _ = simulate(_scenario(), axis_value=0, axis_label="easy", seed=7)
        b, _ = simulate(_scenario(), axis_value=0, axis_label="easy", seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_data(self):
        a, _ = simulate(_scenario(), axis_value=0, axis_label="easy", seed=1)
        b, _ = simulate(_scenario(), axis_value=0, axis_label="easy", seed=2)
        assert not np.array_equal(a, b)

    def test_a_scaling_axis_changes_the_simulated_size(self):
        scenario = _scenario(
            axis=Axis(name="n_total", values=(120, 300), labels=("start", "end")),
            anchors={"start": {"cluster_scale": 1.0}, "end": {"cluster_scale": 1.0}},
            shared={"p": 4, "distribution": "gaussian"},
        )
        X_small, _ = simulate(scenario, axis_value=120, axis_label="start", seed=0)
        X_large, _ = simulate(scenario, axis_value=300, axis_label="end", seed=0)
        assert X_small.shape[0] == 120
        assert X_large.shape[0] == 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._simulate'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_simulate.py`:

```python
"""Turn a Scenario and one axis point into simulated data.

This replaces parse_difficulty_and_simulate and parse_range_and_simulate,
which differed only in how they resolved the swept parameter. Scenario.
sim_kwargs now does that, so one function covers both.
"""

from typing import Any

import numpy as np

from ._types import Scenario


def simulate(
    scenario: Scenario, axis_value: Any, axis_label: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one dataset at one point on a scenario's axis.

    Parameters
    ----------
    scenario : Scenario
        The experiment definition.
    axis_value : Any
        The axis value for this cell. Used as a simulator argument when the
        axis name is a simulator keyword, and only to pick the anchor
        otherwise.
    axis_label : str
        The axis label for this cell, selecting the anchor.
    seed : int
        The fully derived benchmark seed, not the raw loop index.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    """
    from carve.sim import simulate_clusters

    kwargs = scenario.sim_kwargs(axis_value=axis_value, axis_label=axis_label)
    X, y = simulate_clusters(
        k=scenario.k_star,
        plotting=False,
        random_state=int(seed),
        **kwargs,
    )
    return np.asarray(X), np.asarray(y)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_simulate.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_simulate.py tests/benchmarks/test_simulate.py
git commit -m "feat(benchmarks): add unified simulation wrapper"
```

---

### Task 8: Cluster validation indices, with Tibshirani's gap rule

**Files:**
- Create: `src/benchmarks/_cvi.py`
- Create: `tests/benchmarks/test_cvi.py`

**Interfaces:**
- Consumes: `build_estimator` from Task 5, `N_REFERENCE_DATASETS` from Task 6, `EstimatorSpec` from Task 2.
- Produces:
  - `compute_dispersion(X: np.ndarray, labels: np.ndarray) -> float`
  - `gap_statistic(X, labels, *, spec: EstimatorSpec, n_reference_datasets: int = 10, random_state: int = 0) -> tuple[float, float]` returning `(gap, s_k)`
  - `davies_bouldin_inv(X, labels) -> float`
  - `calculate_cvi(X, labels, metric: str, *, spec: EstimatorSpec, random_state: int = 0) -> tuple[float, float]` returning `(value, standard_error)`; the standard error is `0.0` for every metric but gap
  - `select_k(metric: str, ks: Sequence[int], values: Sequence[float], errors: Sequence[float]) -> int`

This is where bug 2 is fixed. The old `gap_statistic` returned only `Gap(k)` and the runner took a plain `argmax`, which is not Tibshirani's rule. The fix returns `s_k` alongside and selects the smallest k satisfying `Gap(k) >= Gap(k+1) - s_{k+1}`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_cvi.py`:

```python
"""Tests for the classical cluster validation indices."""

import numpy as np
import pytest

from benchmarks._cvi import (
    calculate_cvi,
    compute_dispersion,
    davies_bouldin_inv,
    gap_statistic,
    select_k,
)
from benchmarks._types import EstimatorSpec

SPEC = EstimatorSpec(name="kmeans")


@pytest.fixture
def separated_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.1, size=(40, 2))
    b = rng.normal(loc=10.0, scale=0.1, size=(40, 2))
    c = rng.normal(loc=(0.0, 10.0), scale=0.1, size=(40, 2))
    X = np.vstack([a, b, c])
    y = np.repeat([0, 1, 2], 40)
    return X, y


class TestComputeDispersion:
    def test_is_zero_for_identical_points(self):
        X = np.zeros((10, 3))
        assert compute_dispersion(X, np.zeros(10, dtype=int)) == 0.0

    def test_matches_the_sum_of_squared_deviations(self):
        X = np.array([[0.0], [2.0], [10.0], [12.0]])
        labels = np.array([0, 0, 1, 1])
        # Each cluster contributes 2 * 1.0**2 = 2.0
        assert compute_dispersion(X, labels) == pytest.approx(4.0)

    def test_ignores_singleton_clusters(self):
        X = np.array([[0.0], [2.0], [100.0]])
        labels = np.array([0, 0, 1])
        assert compute_dispersion(X, labels) == pytest.approx(2.0)


class TestGapStatistic:
    def test_returns_a_value_and_a_standard_error(self, separated_blobs):
        X, y = separated_blobs
        gap, s_k = gap_statistic(X, y, spec=SPEC, n_reference_datasets=5, random_state=0)
        assert np.isfinite(gap)
        assert s_k > 0.0

    def test_standard_error_includes_the_tibshirani_inflation(self, separated_blobs):
        X, y = separated_blobs
        n_ref = 5
        _, s_k = gap_statistic(X, y, spec=SPEC, n_reference_datasets=n_ref, random_state=0)
        # s_k = sd(log W*) * sqrt(1 + 1/n_ref); recovering sd must be positive
        # and strictly smaller than the inflated value.
        sd = s_k / np.sqrt(1.0 + 1.0 / n_ref)
        assert 0.0 < sd < s_k

    def test_is_deterministic_for_a_fixed_random_state(self, separated_blobs):
        X, y = separated_blobs
        first = gap_statistic(X, y, spec=SPEC, n_reference_datasets=3, random_state=11)
        second = gap_statistic(X, y, spec=SPEC, n_reference_datasets=3, random_state=11)
        assert first == second

    def test_returns_nan_when_dispersion_is_degenerate(self):
        X = np.zeros((10, 2))
        gap, s_k = gap_statistic(
            X, np.zeros(10, dtype=int), spec=SPEC, n_reference_datasets=3
        )
        assert np.isnan(gap)
        assert np.isnan(s_k)


class TestDaviesBouldinInv:
    def test_is_higher_for_better_separated_clusters(self, separated_blobs):
        X, y = separated_blobs
        good = davies_bouldin_inv(X, y)
        rng = np.random.default_rng(1)
        bad = davies_bouldin_inv(X, rng.integers(0, 3, size=X.shape[0]))
        assert good > bad

    def test_is_bounded_in_the_unit_interval(self, separated_blobs):
        X, y = separated_blobs
        assert 0.0 < davies_bouldin_inv(X, y) <= 1.0


class TestCalculateCvi:
    @pytest.mark.parametrize(
        "metric", ["silhouette", "gap", "davies_bouldin", "calinski_harabasz"]
    )
    def test_every_metric_returns_a_value_and_an_error(self, separated_blobs, metric):
        X, y = separated_blobs
        value, error = calculate_cvi(X, y, metric, spec=SPEC, random_state=0)
        assert np.isfinite(value)
        assert error >= 0.0

    def test_only_gap_reports_a_nonzero_error(self, separated_blobs):
        X, y = separated_blobs
        for metric in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            assert calculate_cvi(X, y, metric, spec=SPEC)[1] == 0.0

    def test_raises_on_an_unknown_metric(self, separated_blobs):
        X, y = separated_blobs
        with pytest.raises(ValueError, match="nonsense"):
            calculate_cvi(X, y, "nonsense", spec=SPEC)


class TestSelectK:
    def test_non_gap_metrics_take_the_argmax(self):
        ks = [3, 4, 5, 6]
        values = [0.1, 0.9, 0.4, 0.2]
        assert select_k("silhouette", ks, values, [0.0] * 4) == 4

    def test_gap_takes_the_smallest_k_within_one_standard_error(self):
        # Gap rises to k=5 but k=4 is already within s_5 of Gap(5), so
        # Tibshirani's rule stops at 4 where a plain argmax would say 5.
        ks = [3, 4, 5, 6]
        gaps = [0.10, 0.50, 0.55, 0.30]
        errors = [0.02, 0.02, 0.10, 0.02]
        assert select_k("gap", ks, gaps, errors) == 4

    def test_gap_differs_from_a_plain_argmax(self):
        ks = [3, 4, 5, 6]
        gaps = [0.10, 0.50, 0.55, 0.30]
        errors = [0.02, 0.02, 0.10, 0.02]
        assert select_k("gap", ks, gaps, errors) != ks[int(np.argmax(gaps))]

    def test_gap_falls_back_to_the_largest_k_when_no_k_satisfies_the_rule(self):
        ks = [3, 4, 5]
        gaps = [0.1, 0.2, 0.3]
        errors = [0.0, 0.0, 0.0]
        assert select_k("gap", ks, gaps, errors) == 5

    def test_gap_with_a_single_k_returns_that_k(self):
        assert select_k("gap", [4], [0.5], [0.1]) == 4

    def test_nan_values_are_never_selected(self):
        ks = [3, 4, 5]
        values = [np.nan, 0.2, 0.1]
        assert select_k("silhouette", ks, values, [0.0] * 3) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_cvi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._cvi'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_cvi.py`:

```python
"""Classical cluster validation indices.

The gap statistic returns its standard error alongside its value so that
selection can apply Tibshirani's rule. The routine this replaces returned
only Gap(k) and the caller took a plain argmax, which gave CARVE a 1SE rule
and its main competitor none.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from ._estimators import build_estimator
from ._registry import N_REFERENCE_DATASETS
from ._types import EstimatorSpec


def compute_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster dispersion W_k for squared Euclidean distance.

    Uses W_k = sum_c sum_{i in C_c} ||x_i - mu_c||^2, which is equivalent to
    the pairwise-distance form and runs in O(n*d) rather than O(n^2*d).
    """
    total = 0.0
    for label in np.unique(labels):
        points = X[labels == label]
        if len(points) <= 1:
            continue
        centroid = points.mean(axis=0)
        total += float(np.sum((points - centroid) ** 2))
    return total


def _uniform_reference(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from the axis-aligned bounding box of X."""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return rng.random(size=X.shape) * (maxs - mins) + mins


def gap_statistic(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    spec: EstimatorSpec,
    n_reference_datasets: int = N_REFERENCE_DATASETS,
    random_state: int = 0,
) -> tuple[float, float]:
    """Gap(k) and its standard error s_k.

    Gap(k) = mean_b log W*_kb - log W_k, and
    s_k = sd(log W*_kb) * sqrt(1 + 1 / n_reference_datasets),
    following Tibshirani, Walther and Hastie (2001). The standard deviation
    is the population form (ddof=0), as in the original.

    Returns
    -------
    (gap, s_k), both nan when W_k is non-positive or non-finite.
    """
    rng = np.random.default_rng(int(random_state))

    W_k = compute_dispersion(X, labels)
    if not np.isfinite(W_k) or W_k <= 0:
        return float("nan"), float("nan")

    k = int(np.unique(labels).size)

    log_dispersions = np.empty(n_reference_datasets, dtype=float)
    for b in range(n_reference_datasets):
        X_ref = _uniform_reference(X, rng)
        seed_b = int(rng.integers(0, 2**32 - 1))
        estimator = build_estimator(spec, n_clusters=k, random_state=seed_b)
        ref_labels = estimator.fit_predict(X_ref)
        log_dispersions[b] = np.log(compute_dispersion(X_ref, ref_labels))

    gap = float(np.mean(log_dispersions) - np.log(W_k))
    s_k = float(np.std(log_dispersions, ddof=0) * np.sqrt(1.0 + 1.0 / n_reference_datasets))
    return gap, s_k


def davies_bouldin_inv(X: np.ndarray, labels: np.ndarray) -> float:
    """Inverse Davies-Bouldin index, 1 / (1 + DB), so higher is better."""
    db = davies_bouldin_score(X, labels)
    return 1.0 / (1.0 + db) if np.isfinite(db) else float("nan")


def calculate_cvi(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str,
    *,
    spec: EstimatorSpec,
    random_state: int = 0,
) -> tuple[float, float]:
    """Evaluate one classical index, returning (value, standard_error).

    Only the gap statistic has a meaningful standard error; the others
    report 0.0 so callers can treat every index uniformly.
    """
    if metric == "gap":
        return gap_statistic(X, labels, spec=spec, random_state=random_state)
    if metric == "silhouette":
        return float(silhouette_score(X, labels, random_state=random_state)), 0.0
    if metric in ("davies_bouldin", "DB"):
        return davies_bouldin_inv(X, labels), 0.0
    if metric in ("calinski_harabasz", "CH"):
        return float(calinski_harabasz_score(X, labels)), 0.0
    raise ValueError(f"Unknown metric: {metric!r}")


def select_k(
    metric: str,
    ks: Sequence[int],
    values: Sequence[float],
    errors: Sequence[float],
) -> int:
    """Choose k for one classical index.

    Every index but the gap statistic selects the argmax. The gap statistic
    uses Tibshirani's rule: the smallest k with
    Gap(k) >= Gap(k+1) - s_{k+1}. When no k satisfies it, the largest
    candidate is returned, which is the conventional fallback.
    """
    ks = list(ks)
    values = np.asarray(values, dtype=float)

    if metric != "gap":
        if np.all(np.isnan(values)):
            return ks[-1]
        return ks[int(np.nanargmax(values))]

    errors = np.asarray(errors, dtype=float)
    for i in range(len(ks) - 1):
        if not np.isfinite(values[i]) or not np.isfinite(values[i + 1]):
            continue
        if values[i] >= values[i + 1] - errors[i + 1]:
            return ks[i]
    return ks[-1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_cvi.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_cvi.py tests/benchmarks/test_cvi.py
git commit -m "fix(benchmarks): apply Tibshirani's rule to the gap statistic"
```

---

### Task 9: Artifacts — schema, content addressing, manifest, promotion

**Files:**
- Create: `src/benchmarks/_artifacts.py`
- Create: `tests/benchmarks/test_artifacts.py`

**Interfaces:**
- Consumes: `Scenario`, `Manifest` from Tasks 3 and 4; `ACTIVE_ANCHOR_SET_NAME` from Task 6.
- Produces:
  - `SCHEMA: tuple[str, ...]` — the fifteen columns, in order
  - `config_hash(scenario: Scenario, *, n_seeds: int, n_resamples: int) -> str` — 12 hex characters
  - `run_dir(root: Path, scenario_name: str, cfg_hash: str) -> Path`
  - `write_checkpoint(rd: Path, axis_label: str, seed: int, rows: list[dict]) -> Path`
  - `completed_cells(rd: Path) -> set[tuple[str, int]]`
  - `read_run(rd: Path) -> pd.DataFrame`
  - `peak_rss_bytes() -> int`
  - `build_manifest(scenario, *, run_id, config_hash, n_seeds, n_resamples, n_jobs, random_state, wall_clock_s) -> Manifest`
  - `write_manifest(rd: Path, manifest: Manifest) -> Path`
  - `promote(rd: Path, published_root: Path) -> Path`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_artifacts.py`:

```python
"""Tests for artifact writing, content addressing, and promotion."""

import json

import pandas as pd
import pytest

from benchmarks._artifacts import (
    SCHEMA,
    build_manifest,
    completed_cells,
    config_hash,
    peak_rss_bytes,
    promote,
    read_run,
    run_dir,
    write_checkpoint,
    write_manifest,
)
from benchmarks._registry import SCENARIOS


def _row(**overrides):
    row = {
        "run_id": "r1",
        "scenario": "demo",
        "axis_name": "difficulty_level",
        "axis_value": 0,
        "axis_label": "easy",
        "seed": 0,
        "k_star": 5,
        "estimator": "kmeans",
        "metric_name": "silhouette",
        "k": 3,
        "metric_value": 0.5,
        "is_selected": False,
        "selects_true_k": False,
        "ari_at_k": 0.4,
        "oracle_ari": 0.9,
    }
    row.update(overrides)
    return row


class TestSchema:
    def test_column_order_is_the_documented_contract(self):
        assert SCHEMA == (
            "run_id",
            "scenario",
            "axis_name",
            "axis_value",
            "axis_label",
            "seed",
            "k_star",
            "estimator",
            "metric_name",
            "k",
            "metric_value",
            "is_selected",
            "selects_true_k",
            "ari_at_k",
            "oracle_ari",
        )


class TestConfigHash:
    def test_is_stable_across_calls(self):
        s = SCENARIOS["gaussians"]
        assert config_hash(s, n_seeds=20, n_resamples=100) == config_hash(
            s, n_seeds=20, n_resamples=100
        )

    def test_changes_when_an_anchor_changes(self):
        s = SCENARIOS["gaussians"]
        anchors = {k: dict(v) for k, v in s.anchors.items()}
        anchors["easy"] = {**anchors["easy"], "corr_strength": 0.999}
        changed = type(s)(
            name=s.name,
            axis=s.axis,
            anchors=anchors,
            shared=s.shared,
            estimator=s.estimator,
        )
        assert config_hash(s, n_seeds=20, n_resamples=100) != config_hash(
            changed, n_seeds=20, n_resamples=100
        )

    def test_changes_when_n_resamples_changes(self):
        s = SCENARIOS["gaussians"]
        assert config_hash(s, n_seeds=20, n_resamples=100) != config_hash(
            s, n_seeds=20, n_resamples=50
        )

    def test_is_twelve_hex_characters(self):
        h = config_hash(SCENARIOS["gaussians"], n_seeds=20, n_resamples=100)
        assert len(h) == 12
        assert set(h) <= set("0123456789abcdef")


class TestCheckpoints:
    def test_round_trips_rows_through_parquet(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row(), _row(k=4)])
        df = read_run(rd)
        assert len(df) == 2
        assert list(df.columns) == list(SCHEMA)

    def test_reports_completed_cells_for_resume(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row()])
        write_checkpoint(rd, "hard", 3, [_row(axis_label="hard", seed=3)])
        assert completed_cells(rd) == {("easy", 0), ("hard", 3)}

    def test_an_empty_run_has_no_completed_cells(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        assert completed_cells(rd) == set()

    def test_read_run_concatenates_every_checkpoint(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        for seed in range(3):
            write_checkpoint(rd, "easy", seed, [_row(seed=seed)])
        assert sorted(read_run(rd)["seed"].tolist()) == [0, 1, 2]

    def test_rejects_rows_that_do_not_match_the_schema(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        bad = _row()
        del bad["oracle_ari"]
        with pytest.raises(ValueError, match="oracle_ari"):
            write_checkpoint(rd, "easy", 0, [bad])


class TestManifest:
    def test_records_provenance_and_the_active_anchor_set(self, tmp_path):
        rd = run_dir(tmp_path, "demo", "abc123def456")
        manifest = build_manifest(
            SCENARIOS["gaussians"],
            run_id="r1",
            config_hash="abc123def456",
            n_seeds=2,
            n_resamples=3,
            n_jobs=1,
            random_state=0,
            wall_clock_s=1.0,
        )
        path = write_manifest(rd, manifest)
        payload = json.loads(path.read_text())
        assert payload["anchor_set"] == "PUBLISHED_ANCHORS"
        assert payload["peak_rss_unit"] == "bytes"
        assert payload["config"]["k_star"] == 5
        assert "numpy" in payload["package_versions"]

    def test_peak_rss_is_a_positive_byte_count(self):
        assert peak_rss_bytes() > 1_000_000


class TestPromote:
    def test_writes_csv_and_manifest_to_the_published_root(self, tmp_path):
        rd = run_dir(tmp_path / "runs", "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row()])
        write_manifest(
            rd,
            build_manifest(
                SCENARIOS["gaussians"],
                run_id="r1",
                config_hash="abc123def456",
                n_seeds=1,
                n_resamples=1,
                n_jobs=1,
                random_state=0,
                wall_clock_s=0.1,
            ),
        )
        out = promote(rd, tmp_path / "published")
        assert (out / "results.csv").exists()
        assert (out / "manifest.json").exists()

    def test_promoted_csv_preserves_the_schema(self, tmp_path):
        rd = run_dir(tmp_path / "runs", "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row(), _row(k=4, is_selected=True)])
        write_manifest(
            rd,
            build_manifest(
                SCENARIOS["gaussians"],
                run_id="r1",
                config_hash="abc123def456",
                n_seeds=1,
                n_resamples=1,
                n_jobs=1,
                random_state=0,
                wall_clock_s=0.1,
            ),
        )
        out = promote(rd, tmp_path / "published")
        df = pd.read_csv(out / "results.csv")
        assert list(df.columns) == list(SCHEMA)
        assert df["is_selected"].sum() == 1

    def test_refuses_to_promote_a_run_with_no_manifest(self, tmp_path):
        rd = run_dir(tmp_path / "runs", "demo", "abc123def456")
        write_checkpoint(rd, "easy", 0, [_row()])
        with pytest.raises(FileNotFoundError, match="manifest"):
            promote(rd, tmp_path / "published")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._artifacts'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_artifacts.py`:

```python
"""Reading and writing benchmark artifacts.

Runs are content-addressed: results/runs/<scenario>/<config-hash>/. Changing
an anchor produces a new directory rather than a silent stale read. Each cell
is checkpointed as its own parquet file so a crash at seed 55 of 60 costs one
cell instead of the whole run.
"""

import hashlib
import json
import platform
import resource
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from ._registry import ACTIVE_ANCHOR_SET_NAME
from ._types import Manifest, Scenario

SCHEMA: tuple[str, ...] = (
    "run_id",
    "scenario",
    "axis_name",
    "axis_value",
    "axis_label",
    "seed",
    "k_star",
    "estimator",
    "metric_name",
    "k",
    "metric_value",
    "is_selected",
    "selects_true_k",
    "ari_at_k",
    "oracle_ari",
)

_TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "carve-validate")


def _canonical_config(
    scenario: Scenario, *, n_seeds: int, n_resamples: int
) -> dict[str, Any]:
    """The subset of a scenario that changing must invalidate a run."""
    return {
        "name": scenario.name,
        "axis_name": scenario.axis.name,
        "axis_values": list(scenario.axis.values),
        "axis_labels": list(scenario.axis.labels),
        "anchors": {k: dict(v) for k, v in sorted(scenario.anchors.items())},
        "shared": dict(sorted(scenario.shared.items())),
        "estimator": scenario.estimator.name,
        "k_star": scenario.k_star,
        "candidate_k": list(scenario.candidate_k),
        "n_seeds": n_seeds,
        "n_resamples": n_resamples,
    }


def config_hash(scenario: Scenario, *, n_seeds: int, n_resamples: int) -> str:
    """Twelve-hex-character digest of everything that defines a run."""
    payload = json.dumps(
        _canonical_config(scenario, n_seeds=n_seeds, n_resamples=n_resamples),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_dir(root: Path, scenario_name: str, cfg_hash: str) -> Path:
    """Create and return the directory for one run."""
    path = Path(root) / scenario_name / cfg_hash
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(rd: Path, axis_label: str, seed: int) -> Path:
    return Path(rd) / f"cell__{axis_label}__{seed:04d}.parquet"


def write_checkpoint(
    rd: Path, axis_label: str, seed: int, rows: list[dict[str, Any]]
) -> Path:
    """Write one cell's rows, validating them against the schema first."""
    frame = pd.DataFrame(rows)
    missing = [column for column in SCHEMA if column not in frame.columns]
    if missing:
        raise ValueError(f"Rows are missing schema columns: {missing}.")
    extra = [column for column in frame.columns if column not in SCHEMA]
    if extra:
        raise ValueError(f"Rows carry columns outside the schema: {extra}.")

    path = _checkpoint_path(rd, axis_label, seed)
    frame[list(SCHEMA)].to_parquet(path, index=False)
    return path


def completed_cells(rd: Path) -> set[tuple[str, int]]:
    """Return the (axis_label, seed) pairs already on disk."""
    cells: set[tuple[str, int]] = set()
    for path in Path(rd).glob("cell__*.parquet"):
        _, axis_label, seed = path.stem.split("__")
        cells.add((axis_label, int(seed)))
    return cells


def read_run(rd: Path) -> pd.DataFrame:
    """Concatenate every checkpoint in a run directory."""
    paths = sorted(Path(rd).glob("cell__*.parquet"))
    if not paths:
        return pd.DataFrame(columns=list(SCHEMA))
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return frame[list(SCHEMA)]


def peak_rss_bytes() -> int:
    """Peak resident set size of this process and its children, in bytes.

    ru_maxrss is reported in bytes on macOS and in kilobytes on Linux. The
    unit is normalized here and recorded in the manifest, because a benchmark
    whose headline claim is memory cannot carry a silent factor-of-1024
    difference between the machine an author ran it on and CI.
    """
    scale = 1 if sys.platform == "darwin" else 1024
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(max(own, children) * scale)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def build_manifest(
    scenario: Scenario,
    *,
    run_id: str,
    config_hash: str,
    n_seeds: int,
    n_resamples: int,
    n_jobs: int,
    random_state: int,
    wall_clock_s: float,
) -> Manifest:
    """Assemble the provenance record for one run."""
    import os

    return Manifest(
        run_id=run_id,
        scenario=scenario.name,
        config_hash=config_hash,
        anchor_set=ACTIVE_ANCHOR_SET_NAME,
        n_seeds=n_seeds,
        n_resamples=n_resamples,
        n_jobs=n_jobs,
        random_state=random_state,
        wall_clock_s=float(wall_clock_s),
        peak_rss_bytes=peak_rss_bytes(),
        peak_rss_unit="bytes",
        git_sha=_git_sha(),
        package_versions=_package_versions(),
        platform={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cores": os.cpu_count(),
        },
        config=_canonical_config(scenario, n_seeds=n_seeds, n_resamples=n_resamples),
    )


def write_manifest(rd: Path, manifest: Manifest) -> Path:
    """Write manifest.json beside the checkpoints."""
    path = Path(rd) / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str))
    return path


def promote(rd: Path, published_root: Path) -> Path:
    """Publish a run as committed CSV plus its manifest.

    Deliberate and explicit. A run is never promoted as a side effect of
    executing one.
    """
    rd = Path(rd)
    manifest_path = rd / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. A run without provenance is not "
            "publishable."
        )

    manifest = json.loads(manifest_path.read_text())
    out = Path(published_root) / manifest["scenario"]
    out.mkdir(parents=True, exist_ok=True)

    read_run(rd).to_csv(out / "results.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_artifacts.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_artifacts.py tests/benchmarks/test_artifacts.py
git commit -m "feat(benchmarks): add content-addressed artifacts with manifest and promotion"
```

---

### Task 10: The single runner

**Files:**
- Create: `src/benchmarks/_run.py`
- Create: `tests/benchmarks/test_run.py`

**Interfaces:**
- Consumes: everything from Tasks 2 through 9.
- Produces:
  - `run_cell(scenario, *, axis_idx, axis_value, axis_label, seed, run_id, random_state, n_resamples) -> list[dict]`
  - `run_scenario(scenario, *, root, n_jobs=1, random_state=0, n_seeds=None, n_resamples=100, resume=True, verbose=0) -> Path`

This is where bug 3 is fixed. `get_labels` resolves its `mode` argument through `resolve_mode`, and `"default"` yields `run_stability=True`, so the stability consensus matrix is chosen. The old difficulty runner never passed `mode`, so `ari_at_k` came from stability-mode labels for every metric, generalizability metrics included. The fix is to pass `mode="generalizability"` for the metrics in `GENERALIZABILITY_METRICS` and `mode="default"` otherwise, against a single default-mode fit that has both matrices.

A related divergence is deliberately not changed. The old scaling runner passed `reference_labels=y` to `fit` and the difficulty runner did not. `reference_labels` only permutes cluster ids so they stay consistent across `get_labels` calls (`api.py:728-736`); it never enters the stability or generalizability computation, and ARI is invariant to label permutation. The divergence is cosmetic, so the rebuild simply omits it. Record that in the module docstring so it is not rediscovered as a bug.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_run.py`:

```python
"""Tests for the unified runner."""

import numpy as np
import pytest

from benchmarks._artifacts import SCHEMA, read_run
from benchmarks._registry import CARVE_METRICS_ALL, CVI_METRICS
from benchmarks._run import run_cell, run_scenario
from benchmarks._types import Axis, EstimatorSpec, Scenario

N_METRICS = len(CARVE_METRICS_ALL) + len(CVI_METRICS)


@pytest.fixture
def tiny_scenario():
    """A scenario small enough to fit and score in a couple of seconds."""
    return Scenario(
        name="tiny",
        axis=Axis(name="difficulty_level", values=(0, 1), labels=("easy", "hard")),
        anchors={"easy": {"cluster_scale": 0.6}, "hard": {"cluster_scale": 2.5}},
        shared={"n_total": 120, "p": 4, "distribution": "gaussian"},
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=(3, 4, 5),
        n_seeds=2,
    )


class TestRunCell:
    def test_emits_one_row_per_metric_and_k(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=3,
        )
        assert len(rows) == N_METRICS * len(tiny_scenario.candidate_k)

    def test_rows_carry_exactly_the_schema(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=3,
        )
        assert set(rows[0]) == set(SCHEMA)

    def test_exactly_one_k_is_selected_per_metric(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=3,
        )
        for metric in CARVE_METRICS_ALL + CVI_METRICS:
            selected = [r for r in rows if r["metric_name"] == metric and r["is_selected"]]
            assert len(selected) == 1, metric

    def test_selects_true_k_marks_k_star(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=3,
        )
        for row in rows:
            assert row["selects_true_k"] == (row["k"] == tiny_scenario.k_star)

    def test_oracle_ari_is_constant_within_a_cell(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=3,
        )
        assert len({row["oracle_ari"] for row in rows}) == 1

    def test_generalizability_metrics_use_the_generalizability_matrix(
        self, tiny_scenario, monkeypatch
    ):
        """The old runner never passed mode=, so every metric got stability.

        Asserted on the calls rather than on the values, because the two
        consensus matrices can legitimately agree at a given k on easy data.
        What must hold is that generalizability metrics are cut from the
        generalizability matrix at all.
        """
        import benchmarks._run as run_module

        seen_modes = []
        original = run_module.CARVE

        class RecordingCarve(original):
            def get_labels(self, **kwargs):
                seen_modes.append(kwargs.get("mode", "default"))
                return super().get_labels(**kwargs)

        monkeypatch.setattr(run_module, "CARVE", RecordingCarve)
        run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=5,
        )
        assert "generalizability" in seen_modes
        assert "default" in seen_modes

    def test_labels_mode_routes_each_metric_family(self):
        from benchmarks._registry import GENERALIZABILITY_METRICS
        from benchmarks._run import _labels_mode

        assert _labels_mode("ari_stability_1se") == "default"
        assert _labels_mode("consensus_gini_stability") == "default"
        for metric in GENERALIZABILITY_METRICS:
            assert _labels_mode(metric) == "generalizability"

    def test_is_deterministic_for_a_fixed_seed(self, tiny_scenario):
        kwargs = dict(
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=3,
        )
        first = run_cell(tiny_scenario, **kwargs)
        second = run_cell(tiny_scenario, **kwargs)
        assert [r["metric_value"] for r in first] == [r["metric_value"] for r in second]

    def test_provenance_columns_record_the_actual_estimator(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=3,
        )
        assert {row["estimator"] for row in rows} == {"kmeans"}
        assert {row["axis_label"] for row in rows} == {"easy"}


class TestRunScenario:
    def test_writes_a_complete_run(self, tiny_scenario, tmp_path):
        rd = run_scenario(
            tiny_scenario, root=tmp_path, n_jobs=1, random_state=0, n_resamples=3
        )
        df = read_run(rd)
        expected = (
            len(tiny_scenario.axis)
            * tiny_scenario.n_seeds
            * N_METRICS
            * len(tiny_scenario.candidate_k)
        )
        assert len(df) == expected

    def test_writes_a_manifest(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=3)
        assert (rd / "manifest.json").exists()

    def test_resumes_without_recomputing_completed_cells(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=3)
        before = {p: p.stat().st_mtime_ns for p in rd.glob("cell__*.parquet")}
        run_scenario(tiny_scenario, root=tmp_path, n_resamples=3, resume=True)
        after = {p: p.stat().st_mtime_ns for p in rd.glob("cell__*.parquet")}
        assert before == after

    def test_the_same_config_reuses_one_directory(self, tiny_scenario, tmp_path):
        first = run_scenario(tiny_scenario, root=tmp_path, n_resamples=3)
        second = run_scenario(tiny_scenario, root=tmp_path, n_resamples=3)
        assert first == second

    def test_a_changed_config_gets_a_new_directory(self, tiny_scenario, tmp_path):
        first = run_scenario(tiny_scenario, root=tmp_path, n_resamples=3)
        second = run_scenario(tiny_scenario, root=tmp_path, n_resamples=4)
        assert first != second

    def test_n_seeds_override_shortens_the_run(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_seeds=1, n_resamples=3)
        assert len(read_run(rd)["seed"].unique()) == 1


def test_the_runner_does_not_import_matplotlib():
    import sys

    for module in list(sys.modules):
        if module.startswith("matplotlib"):
            del sys.modules[module]
    import benchmarks._run  # noqa: F401

    assert not any(m.startswith("matplotlib") for m in sys.modules)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._run'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_run.py`:

```python
"""The single benchmark runner.

One entry point covers every simulated experiment, because a difficulty
sweep and a scaling sweep are the same experiment over different axes.

Two behaviors differ deliberately from the code this replaces.

Parallelism is inverted. The old runners parallelized five-element inner
loops with processes while the sixty-iteration outer loop ran serially, and
passed the same n_jobs to both the outer loop and CARVE, so the two nested.
Here joblib parallelizes the outer (axis x seed) loop and CARVE always gets
n_jobs=1.

reference_labels is not passed to fit. The old scaling runner passed the true
labels and the difficulty runner did not. reference_labels only permutes
cluster ids for consistency across get_labels calls (carve/api.py:728-736);
it never enters the stability or generalizability computation, and ARI is
invariant to label permutation. The divergence was cosmetic.
"""

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from carve import CARVE
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

from ._artifacts import (
    build_manifest,
    completed_cells,
    config_hash,
    run_dir,
    write_checkpoint,
    write_manifest,
)
from ._cvi import calculate_cvi, select_k
from ._estimators import build_estimator, param_grids
from ._registry import (
    CARVE_METRICS_ALL,
    CVI_METRICS,
    GENERALIZABILITY_METRICS,
    metric_measure,
    metric_rule,
)
from ._simulate import simulate
from ._types import Scenario


def _labels_mode(metric_name: str) -> str:
    """Which consensus matrix a metric's labels must be cut from.

    get_labels resolves this argument through resolve_mode, and "default"
    yields run_stability=True, which selects the stability matrix. The old
    difficulty runner never passed it, so every metric's ari_at_k came from
    stability-mode labels.
    """
    return "generalizability" if metric_name in GENERALIZABILITY_METRICS else "default"


def run_cell(
    scenario: Scenario,
    *,
    axis_idx: int,
    axis_value: Any,
    axis_label: str,
    seed: int,
    run_id: str,
    random_state: int,
    n_resamples: int,
) -> list[dict[str, Any]]:
    """Run one (axis point, seed) cell and return its rows.

    Simulates, fits an oracle estimator at k_star, fits CARVE, then scores
    every CARVE metric and every classical index at every candidate k.
    """
    benchmark_seed = seed + (axis_idx * 10000) + random_state
    candidate_k = list(scenario.candidate_k)

    X, y = simulate(scenario, axis_value=axis_value, axis_label=axis_label, seed=benchmark_seed)

    oracle = build_estimator(
        scenario.estimator, n_clusters=scenario.k_star, random_state=benchmark_seed
    )
    oracle_ari = float(adjusted_rand_score(y, oracle.fit_predict(X)))

    grids = param_grids(scenario.estimator, candidate_k)
    carve = CARVE(
        estimator_param_grids=grids,
        n_resamples=n_resamples,
        n_jobs=1,
        random_state=benchmark_seed,
    )
    carve.fit(X)

    context = {
        "run_id": run_id,
        "scenario": scenario.name,
        "axis_name": scenario.axis.name,
        "axis_value": axis_value,
        "axis_label": axis_label,
        "seed": seed,
        "k_star": scenario.k_star,
        "estimator": scenario.estimator.name,
        "oracle_ari": oracle_ari,
    }

    rows: list[dict[str, Any]] = []

    # --- CARVE metrics -----------------------------------------------------
    # ari_at_k depends only on the consensus matrix a metric is cut from, so
    # labels are computed once per mode rather than once per metric.
    ari_by_mode: dict[str, dict[int, float]] = {}
    for mode in ("default", "generalizability"):
        ari_by_mode[mode] = {
            k: float(adjusted_rand_score(y, carve.get_labels(k=k, mode=mode)))
            for k in candidate_k
        }

    for metric_name in CARVE_METRICS_ALL:
        measure = metric_measure(metric_name)
        rule = metric_rule(metric_name)
        selected_k = int(carve.get_k(measure=measure, rule=rule))
        aris = ari_by_mode[_labels_mode(metric_name)]

        results = carve.estimator_results_
        for k in candidate_k:
            value = float(results[measure].loc[results["n_clusters"] == k].values[0])
            rows.append(
                {
                    **context,
                    "metric_name": metric_name,
                    "k": k,
                    "metric_value": value,
                    "is_selected": k == selected_k,
                    "selects_true_k": k == scenario.k_star,
                    "ari_at_k": aris[k],
                }
            )

    # --- Classical indices -------------------------------------------------
    labels_by_k = {
        k: np.asarray(
            build_estimator(
                scenario.estimator, n_clusters=k, random_state=benchmark_seed
            ).fit_predict(X),
            dtype=np.int32,
        )
        for k in candidate_k
    }
    cvi_ari = {k: float(adjusted_rand_score(y, labels_by_k[k])) for k in candidate_k}

    for metric_name in CVI_METRICS:
        values: list[float] = []
        errors: list[float] = []
        for k in candidate_k:
            value, error = calculate_cvi(
                X,
                labels_by_k[k],
                metric_name,
                spec=scenario.estimator,
                random_state=benchmark_seed,
            )
            values.append(value)
            errors.append(error)

        selected_k = select_k(metric_name, candidate_k, values, errors)
        for k, value in zip(candidate_k, values):
            rows.append(
                {
                    **context,
                    "metric_name": metric_name,
                    "k": k,
                    "metric_value": value,
                    "is_selected": k == selected_k,
                    "selects_true_k": k == scenario.k_star,
                    "ari_at_k": cvi_ari[k],
                }
            )

    return rows


def run_scenario(
    scenario: Scenario,
    *,
    root: Path,
    n_jobs: int = 1,
    random_state: int = 0,
    n_seeds: int | None = None,
    n_resamples: int = 100,
    resume: bool = True,
    verbose: int = 0,
) -> Path:
    """Run every cell of a scenario, checkpointing as it goes.

    Returns the run directory, which is content-addressed on the scenario
    configuration so a changed anchor cannot read a stale result.
    """
    n_seeds = scenario.n_seeds if n_seeds is None else int(n_seeds)
    cfg_hash = config_hash(scenario, n_seeds=n_seeds, n_resamples=n_resamples)
    rd = run_dir(Path(root), scenario.name, cfg_hash)

    done = completed_cells(rd) if resume else set()
    pending = [
        (axis_idx, axis_value, axis_label, seed)
        for axis_idx, axis_value, axis_label in scenario.axis
        for seed in range(n_seeds)
        if (axis_label, seed) not in done
    ]

    if verbose:
        print(f"{scenario.name}: {len(pending)} cells to run, {len(done)} already done.")

    run_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()

    def _one(axis_idx, axis_value, axis_label, seed):
        rows = run_cell(
            scenario,
            axis_idx=axis_idx,
            axis_value=axis_value,
            axis_label=axis_label,
            seed=seed,
            run_id=run_id,
            random_state=random_state,
            n_resamples=n_resamples,
        )
        write_checkpoint(rd, axis_label, seed, rows)

    if pending:
        Parallel(n_jobs=n_jobs)(
            delayed(_one)(*cell)
            for cell in tqdm(pending, desc=scenario.name, leave=False)
        )

    write_manifest(
        rd,
        build_manifest(
            scenario,
            run_id=run_id,
            config_hash=cfg_hash,
            n_seeds=n_seeds,
            n_resamples=n_resamples,
            n_jobs=n_jobs,
            random_state=random_state,
            wall_clock_s=time.perf_counter() - started,
        ),
    )
    return rd
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_run.py -v`
Expected: all passed. These fit real CARVE models, so expect roughly one to three minutes.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_run.py tests/benchmarks/test_run.py
git commit -m "feat(benchmarks): add the unified runner and fix ari_at_k fit mode"
```

---

### Task 11: Command-line entry point

**Files:**
- Create: `src/benchmarks/run.py`
- Create: `tests/benchmarks/test_cli.py`

**Interfaces:**
- Consumes: `SCENARIOS` from Task 6, `run_scenario` from Task 10, `promote` from Task 9.
- Produces: `main(argv: list[str] | None = None) -> int`, invoked as `python -m benchmarks.run`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_cli.py`:

```python
"""Tests for the benchmarks command-line entry point."""

import pytest

from benchmarks.run import main


class TestCli:
    def test_listing_scenarios_succeeds(self, capsys):
        assert main(["--list"]) == 0
        assert "gaussians" in capsys.readouterr().out

    def test_unknown_scenario_is_rejected(self, capsys):
        assert main(["--scenario", "nope"]) == 2
        assert "nope" in capsys.readouterr().err

    def test_requires_one_of_scenario_all_list_or_promote(self, capsys):
        assert main([]) == 2

    def test_runs_a_scenario_end_to_end(self, tmp_path):
        code = main(
            [
                "--scenario",
                "gaussians",
                "--root",
                str(tmp_path),
                "--n-seeds",
                "1",
                "--n-resamples",
                "3",
            ]
        )
        assert code == 0
        assert list(tmp_path.glob("gaussians/*/manifest.json"))

    def test_promote_publishes_a_finished_run(self, tmp_path):
        main(
            [
                "--scenario",
                "gaussians",
                "--root",
                str(tmp_path / "runs"),
                "--n-seeds",
                "1",
                "--n-resamples",
                "3",
            ]
        )
        rd = next((tmp_path / "runs" / "gaussians").iterdir())
        code = main(["--promote", str(rd), "--published-root", str(tmp_path / "pub")])
        assert code == 0
        assert (tmp_path / "pub" / "gaussians" / "results.csv").exists()

    def test_promote_rejects_a_directory_with_no_manifest(self, tmp_path, capsys):
        (tmp_path / "empty").mkdir()
        assert main(["--promote", str(tmp_path / "empty")]) == 1
        assert "manifest" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.run'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/run.py`:

```python
"""Command-line entry point: python -m benchmarks.run.

Compute only. Nothing here imports matplotlib or renders anything; figures
and tables are produced separately from the artifacts this writes.
"""

import argparse
import sys
from pathlib import Path

from ._artifacts import promote
from ._registry import SCENARIOS
from ._run import run_scenario

DEFAULT_ROOT = Path("results/runs")
DEFAULT_PUBLISHED_ROOT = Path("results/published")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Run CARVE benchmark scenarios and write versioned artifacts.",
    )
    parser.add_argument("--scenario", help="Name of a single scenario to run.")
    parser.add_argument("--all", action="store_true", help="Run every scenario.")
    parser.add_argument("--list", action="store_true", help="List scenario names.")
    parser.add_argument("--promote", help="Publish the run directory at this path.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Working run root.")
    parser.add_argument(
        "--published-root",
        default=str(DEFAULT_PUBLISHED_ROOT),
        help="Destination root for promoted runs.",
    )
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-resamples", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="count", default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.list:
        for name in sorted(SCENARIOS):
            print(name)
        return 0

    if args.promote:
        try:
            out = promote(Path(args.promote), Path(args.published_root))
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Promoted to {out}")
        return 0

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(
                f"Unknown scenario {args.scenario!r}. "
                f"Valid names: {sorted(SCENARIOS)}.",
                file=sys.stderr,
            )
            return 2
        names = [args.scenario]
    elif args.all:
        names = sorted(SCENARIOS)
    else:
        print(
            "Nothing to do. Pass --scenario, --all, --list, or --promote.",
            file=sys.stderr,
        )
        return 2

    for name in names:
        rd = run_scenario(
            SCENARIOS[name],
            root=Path(args.root),
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            n_seeds=args.n_seeds,
            n_resamples=args.n_resamples,
            resume=not args.no_resume,
            verbose=args.verbose,
        )
        print(f"{name}: {rd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_cli.py -v`
Expected: all passed

- [ ] **Step 5: Verify the module runs as a script**

Run: `.venv/bin/python -m benchmarks.run --list`
Expected: eight scenario names, one per line

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/run.py tests/benchmarks/test_cli.py
git commit -m "feat(benchmarks): add python -m benchmarks.run entry point"
```

---

### Task 12: SNR calibration

**Files:**
- Create: `src/benchmarks/_calibrate.py`
- Create: `tests/benchmarks/test_calibrate.py`

**Interfaces:**
- Consumes: `Scenario` from Task 3, `simulate` from Task 7, `build_estimator` from Task 5.
- Produces:
  - `TARGET_ARI_BANDS: dict[str, tuple[float, float]]`
  - `CALIBRATION_SPACE: dict[str, tuple[float, float]]`
  - `mean_oracle_ari(scenario, axis_label, overrides, *, n_seeds, random_state) -> float`
  - `calibrate_anchor(scenario, axis_label, *, target, n_seeds=20, random_state=0, max_iter=25) -> dict[str, Any]`
  - `calibrate_scenario(scenario, *, n_seeds=20, random_state=0, max_iter=25) -> dict[str, dict[str, Any]]`

The original search space is not recoverable: the published anchors move four parameters at once and non-monotonically, which is not the signature of an ordered sweep. This module therefore defines its own documented search over a single scale parameter, and S3 Text is updated to describe what the code does. That is a deliberate substitution, not a reproduction, and the module docstring must say so.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_calibrate.py`:

```python
"""Tests for the SNR calibration replacement."""

import pytest

from benchmarks._calibrate import (
    CALIBRATION_SPACE,
    TARGET_ARI_BANDS,
    calibrate_anchor,
    mean_oracle_ari,
)
from benchmarks._types import Axis, EstimatorSpec, Scenario


@pytest.fixture
def tiny_scenario():
    return Scenario(
        name="tiny",
        axis=Axis(name="difficulty_level", values=(0,), labels=("easy",)),
        anchors={"easy": {"cluster_scale": 1.0}},
        shared={"n_total": 120, "p": 4, "distribution": "gaussian"},
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=(3, 4, 5),
        n_seeds=3,
    )


class TestTargets:
    def test_bands_match_the_manuscript(self):
        assert TARGET_ARI_BANDS == {
            "easy": (0.9, 1.0),
            "medium": (0.8, 0.9),
            "hard": (0.7, 0.8),
        }

    def test_the_search_parameter_is_documented(self):
        assert "cluster_scale" in CALIBRATION_SPACE
        low, high = CALIBRATION_SPACE["cluster_scale"]
        assert low < high


class TestMeanOracleAri:
    def test_returns_a_value_in_the_unit_interval(self, tiny_scenario):
        ari = mean_oracle_ari(
            tiny_scenario, "easy", {"cluster_scale": 0.5}, n_seeds=2, random_state=0
        )
        assert -1.0 <= ari <= 1.0

    def test_tighter_clusters_score_higher(self, tiny_scenario):
        tight = mean_oracle_ari(
            tiny_scenario, "easy", {"cluster_scale": 0.4}, n_seeds=3, random_state=0
        )
        loose = mean_oracle_ari(
            tiny_scenario, "easy", {"cluster_scale": 4.0}, n_seeds=3, random_state=0
        )
        assert tight > loose

    def test_is_deterministic(self, tiny_scenario):
        kwargs = dict(n_seeds=2, random_state=0)
        first = mean_oracle_ari(tiny_scenario, "easy", {"cluster_scale": 1.0}, **kwargs)
        second = mean_oracle_ari(tiny_scenario, "easy", {"cluster_scale": 1.0}, **kwargs)
        assert first == second


class TestCalibrateAnchor:
    def test_returns_an_anchor_and_the_ari_it_achieved(self, tiny_scenario):
        result = calibrate_anchor(
            tiny_scenario, "easy", target=(0.9, 1.0), n_seeds=2, max_iter=8
        )
        assert "anchor" in result
        assert "achieved_ari" in result
        assert "cluster_scale" in result["anchor"]

    def test_reports_whether_the_band_was_reached(self, tiny_scenario):
        result = calibrate_anchor(
            tiny_scenario, "easy", target=(0.9, 1.0), n_seeds=2, max_iter=8
        )
        assert isinstance(result["in_band"], bool)

    def test_rejects_an_inverted_band(self, tiny_scenario):
        with pytest.raises(ValueError, match="band"):
            calibrate_anchor(tiny_scenario, "easy", target=(1.0, 0.9), n_seeds=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._calibrate'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_calibrate.py`:

```python
"""Regenerate the SNR anchors from the documented target ARI bands.

This is a replacement, not a reproduction. The notebook that produced the
published anchors was lost, and the anchors it produced move four parameters
at once and non-monotonically across difficulty levels, which is not the
signature of a single ordered sweep. The original search space is therefore
not recoverable.

What this module does instead is documented and reproducible: hold every
anchor parameter fixed except one scale parameter, and bisect it until a base
estimator informed with the true k_star reaches a mean ARI inside the target
band over the calibration seeds. The manuscript's S3 Text is updated to
describe this procedure.

Run offline, not as part of a benchmark run. Compare its output against
_registry.PUBLISHED_ANCHORS before adopting anything.
"""

from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ._estimators import build_estimator
from ._simulate import simulate
from ._types import Scenario

TARGET_ARI_BANDS: dict[str, tuple[float, float]] = {
    "easy": (0.9, 1.0),
    "medium": (0.8, 0.9),
    "hard": (0.7, 0.8),
}

# The single calibrated parameter and the interval it is searched over.
# Larger cluster_scale means more diffuse clusters and a lower ARI, so the
# objective is monotonically decreasing in it, which is what makes bisection
# valid.
CALIBRATION_SPACE: dict[str, tuple[float, float]] = {
    "cluster_scale": (0.1, 12.0),
}


def mean_oracle_ari(
    scenario: Scenario,
    axis_label: str,
    overrides: dict[str, Any],
    *,
    n_seeds: int = 20,
    random_state: int = 0,
) -> float:
    """Mean ARI of the oracle estimator at k_star over the calibration seeds.

    Calibration uses the same seed derivation as the benchmark, so the
    calibration and benchmark datasets coincide, as the manuscript states.
    """
    axis_idx = list(scenario.axis.labels).index(axis_label)
    axis_value = scenario.axis.values[axis_idx]

    anchors = {k: dict(v) for k, v in scenario.anchors.items()}
    anchors[axis_label].update(overrides)
    probe = Scenario(
        name=scenario.name,
        axis=scenario.axis,
        anchors=anchors,
        shared=scenario.shared,
        estimator=scenario.estimator,
        k_star=scenario.k_star,
        candidate_k=scenario.candidate_k,
        n_seeds=scenario.n_seeds,
    )

    aris = []
    for seed in range(n_seeds):
        benchmark_seed = seed + (axis_idx * 10000) + random_state
        X, y = simulate(
            probe, axis_value=axis_value, axis_label=axis_label, seed=benchmark_seed
        )
        estimator = build_estimator(
            probe.estimator, n_clusters=probe.k_star, random_state=benchmark_seed
        )
        aris.append(adjusted_rand_score(y, estimator.fit_predict(X)))
    return float(np.mean(aris))


def calibrate_anchor(
    scenario: Scenario,
    axis_label: str,
    *,
    target: tuple[float, float],
    n_seeds: int = 20,
    random_state: int = 0,
    max_iter: int = 25,
) -> dict[str, Any]:
    """Bisect the scale parameter until mean oracle ARI lands in the band."""
    low_target, high_target = target
    if not low_target < high_target:
        raise ValueError(f"Invalid target band {target!r}: low must be below high.")

    midpoint_target = 0.5 * (low_target + high_target)
    low, high = CALIBRATION_SPACE["cluster_scale"]

    best_scale = 0.5 * (low + high)
    best_ari = float("nan")

    for _ in range(max_iter):
        scale = 0.5 * (low + high)
        ari = mean_oracle_ari(
            scenario,
            axis_label,
            {"cluster_scale": scale},
            n_seeds=n_seeds,
            random_state=random_state,
        )
        best_scale, best_ari = scale, ari

        if low_target <= ari <= high_target:
            break
        # ARI decreases as cluster_scale grows.
        if ari > midpoint_target:
            low = scale
        else:
            high = scale

    return {
        "anchor": {"cluster_scale": best_scale},
        "achieved_ari": best_ari,
        "target": target,
        "in_band": bool(low_target <= best_ari <= high_target),
    }


def calibrate_scenario(
    scenario: Scenario,
    *,
    n_seeds: int = 20,
    random_state: int = 0,
    max_iter: int = 25,
) -> dict[str, dict[str, Any]]:
    """Calibrate every anchor of a scenario against its target band."""
    return {
        label: calibrate_anchor(
            scenario,
            label,
            target=TARGET_ARI_BANDS[label],
            n_seeds=n_seeds,
            random_state=random_state,
            max_iter=max_iter,
        )
        for label in scenario.axis.labels
        if label in TARGET_ARI_BANDS
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_calibrate.py -v`
Expected: all passed

- [ ] **Step 5: Regenerate the anchors and compare against the published set**

This is verification item 4 in the spec. Report the difference; do not silently adopt either set.

Run:

```bash
.venv/bin/python - <<'PY'
import json
from benchmarks._calibrate import calibrate_scenario
from benchmarks._registry import PUBLISHED_ANCHORS, SCENARIOS

report = {}
for name in ("gaussians", "t_dist", "t_dist_noise", "circles", "moons", "swiss_rolls"):
    result = calibrate_scenario(SCENARIOS[name], n_seeds=20, random_state=0)
    report[name] = {
        label: {
            "calibrated_cluster_scale": r["anchor"]["cluster_scale"],
            "published_cluster_scale": PUBLISHED_ANCHORS[name][label].get("cluster_scale"),
            "achieved_ari": r["achieved_ari"],
            "target": list(r["target"]),
            "in_band": r["in_band"],
        }
        for label, r in result.items()
    }
print(json.dumps(report, indent=2))
PY
```

Save the output to `docs/superpowers/notes/2026-09-01-calibration-comparison.md` with a short
paragraph stating which anchors landed in band, how far the regenerated scale parameters sit from
the published ones, and the recommendation on whether to switch `ACTIVE_ANCHORS`.

Leave `ACTIVE_ANCHORS = PUBLISHED_ANCHORS`. Switching it changes every published number and is a
decision for the author after reading this comparison, not a step in this task.

Note that several published anchors set `cluster_scale` to a per-cluster list rather than a
scalar, so the comparison prints the published value as-is and the two are not always directly
commensurable. Say so in the note rather than forcing a number.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_calibrate.py tests/benchmarks/test_calibrate.py docs/superpowers/notes/2026-09-01-calibration-comparison.md
git commit -m "feat(benchmarks): add documented SNR calibration replacement"
```

---

### Task 13: The single summarizer and LaTeX renderer

**Files:**
- Create: `src/benchmarks/_tables.py`
- Create: `tests/benchmarks/test_tables.py`
- Read for reference: `notebooks/benchmarking_code/benchmarking_reporting.py:94-480`

**Interfaces:**
- Consumes: `SCHEMA` from Task 9, `METRIC_DISPLAY_NAMES` from Task 6.
- Produces:
  - `wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]`
  - `summary_stats(values: pd.Series) -> dict[str, float]`
  - `summarize(df: pd.DataFrame, *, metrics=None, decimals=3) -> pd.DataFrame` — one row per (axis_label, metric)
  - `render_grouped_tex(summary: pd.DataFrame, *, caption: str, label: str, decimals: int = 3) -> str`
  - `write_tables(df, out_dir: Path, name: str, *, caption: str, label: str) -> Path`

The two old summarizers shared 105 of 122 code lines and differed only in a column header, because one grouped by `difficulty` and the other by `stage`. With `axis_label` unified they collapse into one function. Keep the statistics from `_summarize_single_group` (mean, sd, median, IQR of ARI; delta to oracle; k-recovery with a Wilson interval; k-bias; over- and under-selection rates) and the output shape of `_render_grouped_tex`. Do not port `render_tex_table`, which routes through `df.to_latex` and has no callers.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_tables.py`:

```python
"""Tests for the unified summarizer."""

import numpy as np
import pandas as pd
import pytest

from benchmarks._artifacts import SCHEMA
from benchmarks._tables import (
    render_grouped_tex,
    summarize,
    summary_stats,
    wilson_ci,
    write_tables,
)


def _frame():
    """Two axis labels x two metrics x two seeds x three k values."""
    rows = []
    for axis_label in ("easy", "hard"):
        for seed in range(2):
            for metric in ("ari_stability_1se", "silhouette"):
                selected = 5 if metric == "ari_stability_1se" else 4
                for k in (3, 4, 5):
                    rows.append(
                        {
                            "run_id": "r1",
                            "scenario": "demo",
                            "axis_name": "difficulty_level",
                            "axis_value": 0 if axis_label == "easy" else 2,
                            "axis_label": axis_label,
                            "seed": seed,
                            "k_star": 5,
                            "estimator": "kmeans",
                            "metric_name": metric,
                            "k": k,
                            "metric_value": 0.1 * k,
                            "is_selected": k == selected,
                            "selects_true_k": k == 5,
                            "ari_at_k": 0.5 + 0.1 * k,
                            "oracle_ari": 0.9,
                        }
                    )
    return pd.DataFrame(rows)[list(SCHEMA)]


class TestWilsonCi:
    def test_brackets_the_point_estimate(self):
        low, high = wilson_ci(7, 10)
        assert low < 0.7 < high

    def test_is_bounded_by_zero_and_one(self):
        low, high = wilson_ci(0, 10)
        assert low >= 0.0
        high_low, high_high = wilson_ci(10, 10)
        assert high_high <= 1.0

    def test_returns_nan_for_an_empty_sample(self):
        low, high = wilson_ci(0, 0)
        assert np.isnan(low) and np.isnan(high)


class TestSummaryStats:
    def test_reports_mean_sd_median_and_quartiles(self):
        stats = summary_stats(pd.Series([0.0, 1.0, 2.0, 3.0, 4.0]))
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["median"] == pytest.approx(2.0)
        assert stats["q25"] == pytest.approx(1.0)
        assert stats["q75"] == pytest.approx(3.0)

    def test_sd_is_zero_for_a_single_value(self):
        assert summary_stats(pd.Series([1.0]))["sd"] == 0.0

    def test_empty_input_gives_nan(self):
        assert np.isnan(summary_stats(pd.Series([], dtype=float))["mean"])


class TestSummarize:
    def test_one_row_per_axis_label_and_metric(self):
        out = summarize(_frame())
        assert len(out) == 2 * 2

    def test_uses_only_selected_rows(self):
        out = summarize(_frame())
        row = out[
            (out["axis_label"] == "easy") & (out["metric"] == "ari_stability_1se")
        ].iloc[0]
        # selected k is 5, so ari_at_k is 0.5 + 0.5 = 1.0
        assert row["ari_mean"] == pytest.approx(1.0)

    def test_k_recovery_reflects_selects_true_k(self):
        out = summarize(_frame())
        stability = out[out["metric"] == "ari_stability_1se"]
        silhouette = out[out["metric"] == "silhouette"]
        assert (stability["k_recovery"] == 1.0).all()
        assert (silhouette["k_recovery"] == 0.0).all()

    def test_delta_to_oracle_is_oracle_minus_selected_ari(self):
        out = summarize(_frame())
        row = out[
            (out["axis_label"] == "easy") & (out["metric"] == "silhouette")
        ].iloc[0]
        # selected k is 4, ari_at_k 0.9, oracle 0.9
        assert row["delta_mean"] == pytest.approx(0.0)

    def test_k_bias_is_selected_k_minus_k_star(self):
        out = summarize(_frame())
        row = out[out["metric"] == "silhouette"].iloc[0]
        assert row["k_bias_median"] == pytest.approx(-1.0)

    def test_over_and_under_rates_sum_with_recovery_to_one(self):
        out = summarize(_frame())
        for _, row in out.iterrows():
            assert row["p_under"] + row["p_over"] + row["k_recovery"] == pytest.approx(1.0)

    def test_restricting_metrics_filters_the_output(self):
        out = summarize(_frame(), metrics=("silhouette",))
        assert set(out["metric"]) == {"silhouette"}

    def test_raises_when_no_requested_metric_is_present(self):
        with pytest.raises(ValueError, match="none of the requested metrics"):
            summarize(_frame(), metrics=("nonexistent",))


class TestRenderGroupedTex:
    def test_produces_a_tabular_environment(self):
        tex = render_grouped_tex(summarize(_frame()), caption="Demo", label="tab:demo")
        assert "\\begin{table}" in tex
        assert "\\end{table}" in tex
        assert "tab:demo" in tex

    def test_includes_every_axis_label_as_a_column_group(self):
        tex = render_grouped_tex(summarize(_frame()), caption="Demo", label="tab:demo")
        assert "easy" in tex
        assert "hard" in tex

    def test_escapes_latex_special_characters_in_the_caption(self):
        tex = render_grouped_tex(
            summarize(_frame()), caption="100% of runs", label="tab:demo"
        )
        assert "100\\%" in tex


class TestWriteTables:
    def test_writes_a_tex_fragment_under_the_manuscript_name(self, tmp_path):
        path = write_tables(
            _frame(), tmp_path, "S2_table", caption="Demo", label="tab:s2"
        )
        assert path.name == "S2_table.tex"
        assert "\\begin{table}" in path.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_tables.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._tables'`

- [ ] **Step 3: Write the implementation**

Create `src/benchmarks/_tables.py`:

```python
"""Summarize a run into the manuscript's supplementary tables.

One summarizer covers every experiment. The two it replaces shared 105 of
122 code lines and differed only in whether they grouped by "difficulty" or
"stage"; with axis_label unified there is nothing left to differ about.

Tables are written to disk as .tex fragments under their manuscript names
rather than printed for copy-paste, so the supplementary tables stop drifting
from what the code produces.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ._registry import METRIC_DISPLAY_NAMES

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denominator
    half = (z / denominator) * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def summary_stats(values: pd.Series) -> dict[str, float]:
    """Mean, sd, median, and the standard quantiles of a numeric series."""
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    keys = ["mean", "sd", "median", "q05", "q25", "q75", "q95"]
    if values.empty:
        return dict.fromkeys(keys, float("nan"))
    quantiles = values.quantile(list(_QUANTILES))
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(quantiles.loc[0.50]),
        "q05": float(quantiles.loc[0.05]),
        "q25": float(quantiles.loc[0.25]),
        "q75": float(quantiles.loc[0.75]),
        "q95": float(quantiles.loc[0.95]),
    }


def summarize(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    decimals: int = 3,
) -> pd.DataFrame:
    """One row per (axis_label, metric), computed from the selected k only."""
    selected = df.loc[df["is_selected"]].copy()

    if metrics is None:
        metrics = tuple(sorted(selected["metric_name"].unique()))
    present = [m for m in metrics if (selected["metric_name"] == m).any()]
    if not present:
        raise ValueError(
            "none of the requested metrics were found after filtering to "
            "selected rows"
        )

    rows = []
    for axis_label, by_label in selected.groupby("axis_label", sort=False):
        for metric in present:
            sub = by_label.loc[by_label["metric_name"] == metric]
            n = int(len(sub))
            if n == 0:
                continue

            ari = summary_stats(sub["ari_at_k"])
            delta = summary_stats(
                pd.to_numeric(sub["oracle_ari"], errors="coerce")
                - pd.to_numeric(sub["ari_at_k"], errors="coerce")
            )
            successes = int(sub["selects_true_k"].astype(bool).sum())
            low, high = wilson_ci(successes, n)
            k_bias = pd.to_numeric(sub["k"], errors="coerce") - pd.to_numeric(
                sub["k_star"], errors="coerce"
            )

            rows.append(
                {
                    "axis_label": axis_label,
                    "metric": metric,
                    "display_name": METRIC_DISPLAY_NAMES.get(metric, metric),
                    "n_datasets": n,
                    "ari_mean": ari["mean"],
                    "ari_sd": ari["sd"],
                    "ari_median": ari["median"],
                    "ari_q25": ari["q25"],
                    "ari_q75": ari["q75"],
                    "delta_mean": delta["mean"],
                    "delta_median": delta["median"],
                    "k_recovery": successes / n,
                    "k_rec_lo95": low,
                    "k_rec_hi95": high,
                    "k_bias_median": float(k_bias.median()),
                    "p_under": float((k_bias < 0).mean()),
                    "p_over": float((k_bias > 0).mean()),
                }
            )

    out = pd.DataFrame(rows)
    numeric = out.select_dtypes(include=[float]).columns
    out[numeric] = out[numeric].round(decimals)
    return out


def _tex_escape(text: str) -> str:
    """Escape the LaTeX special characters that appear in captions."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def render_grouped_tex(
    summary: pd.DataFrame,
    *,
    caption: str,
    label: str,
    decimals: int = 3,
) -> str:
    """Render the summary as a table grouped by axis label.

    Column groups are the axis labels; rows are metrics. This reproduces the
    shape of the tables already in the manuscript.
    """
    axis_labels = list(dict.fromkeys(summary["axis_label"]))
    metrics = list(dict.fromkeys(summary["metric"]))

    column_spec = "l" + "cc" * len(axis_labels)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
    ]

    header = ["Method"]
    for axis_label in axis_labels:
        header.append(f"\\multicolumn{{2}}{{c}}{{{_tex_escape(str(axis_label))}}}")
    lines.append(" & ".join(header) + r" \\")
    lines.append("Method" + " & ARI & $k$-rec" * len(axis_labels) + r" \\")
    lines.append(r"\hline")

    for metric in metrics:
        cells = [_tex_escape(METRIC_DISPLAY_NAMES.get(metric, metric))]
        for axis_label in axis_labels:
            match = summary[
                (summary["metric"] == metric) & (summary["axis_label"] == axis_label)
            ]
            if match.empty:
                cells.extend(["", ""])
                continue
            row = match.iloc[0]
            cells.append(
                f"{row['ari_mean']:.{decimals}f} ({row['ari_sd']:.{decimals}f})"
            )
            cells.append(f"{row['k_recovery']:.2f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_tables(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    *,
    caption: str,
    label: str,
    metrics: Sequence[str] | None = None,
) -> Path:
    """Summarize a run and write the .tex fragment under its manuscript name."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tex"
    path.write_text(
        render_grouped_tex(
            summarize(df, metrics=metrics), caption=caption, label=label
        )
    )
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_tables.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_tables.py tests/benchmarks/test_tables.py
git commit -m "feat(benchmarks): add the unified summarizer and LaTeX renderer"
```

---

### Task 14: Regression harness against the committed results

**Files:**
- Create: `tests/benchmarks/test_regression.py`
- Read for data: `results/results_gaussian.csv`, `results/results_t_dist.csv`, `results/results_t_dist_noise.csv`, `results/results_circles.csv`, `results/results_moons.csv`, `results/results_swiss_rolls.csv`

**Interfaces:**
- Consumes: `run_scenario`, `read_run`, `SCENARIOS`.
- Produces: `OLD_TO_NEW: dict[str, str]` and a `slow`-marked test per unaffected scenario.

This is the primary evidence that the rewrite preserved the logic. Three columns are expected to differ and are excluded from the comparison, because Tasks 8 and 10 deliberately changed them:

- every row where `metric_name == "gap"`, because the selection rule changed from `argmax` to Tibshirani's
- `ari_at_k` for the metrics in `GENERALIZABILITY_METRICS`, because the old value came from stability-mode labels
- `swiss_rolls` illustrations, which used the wrong estimator; the numbers themselves were produced with agglomerative and are compared normally

The old CSVs stay in place until the port is verified. They are the oracle, so they cannot be deleted in the step that checks against them. The mapping below lives here, in the test, not in the library.

- [ ] **Step 1: Write the regression test**

Create `tests/benchmarks/test_regression.py`:

```python
"""Regression: the rebuilt pipeline must reproduce the committed results.

Marked slow and skipped unless CARVE_RUN_REGRESSION=1, because each scenario
is a full 3 x 20 benchmark. Run one with:

    CARVE_RUN_REGRESSION=1 .venv/bin/pytest \
        tests/benchmarks/test_regression.py -k gaussians -v
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from benchmarks._artifacts import read_run
from benchmarks._registry import GENERALIZABILITY_METRICS, SCENARIOS
from benchmarks._run import run_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

# The rebuild renames six columns. This mapping exists only to compare against
# the old files and is deleted with them once the port is verified.
OLD_TO_NEW = {
    "difficulty": "axis_label",
    "stage": "axis_label",
    "is_optimal": "is_selected",
    "is_correct": "selects_true_k",
    "metric_ari": "ari_at_k",
    "baseline_ari": "oracle_ari",
    "dataset_iteration": "seed",
    "true_k": "k_star",
}

# Scenario name in the registry -> committed CSV stem.
UNAFFECTED = {
    "gaussians": "results_gaussian",
    "t_dist": "results_t_dist",
    "t_dist_noise": "results_t_dist_noise",
    "circles": "results_circles",
    "moons": "results_moons",
    "swiss_rolls": "results_swiss_rolls",
}

pytestmark = pytest.mark.skipif(
    os.environ.get("CARVE_RUN_REGRESSION") != "1",
    reason="set CARVE_RUN_REGRESSION=1 to run the full-scenario regression",
)


def _load_old(stem: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{stem}.csv")
    return df.rename(columns={k: v for k, v in OLD_TO_NEW.items() if k in df.columns})


def _comparable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows the bug fixes deliberately changed, then sort."""
    df = df[df["metric_name"] != "gap"].copy()
    keys = ["axis_label", "seed", "metric_name", "k"]
    return df.sort_values(keys).reset_index(drop=True)


@pytest.mark.slow
@pytest.mark.parametrize(("scenario_name", "stem"), sorted(UNAFFECTED.items()))
def test_rebuilt_pipeline_reproduces_committed_results(scenario_name, stem, tmp_path):
    scenario = SCENARIOS[scenario_name]
    rd = run_scenario(scenario, root=tmp_path, n_jobs=-1, random_state=0)

    new = _comparable(read_run(rd))
    old = _comparable(_load_old(stem))

    assert len(new) == len(old), (
        f"{scenario_name}: row count changed, {len(old)} -> {len(new)}"
    )

    merged = old.merge(
        new,
        on=["axis_label", "seed", "metric_name", "k"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    assert len(merged) == len(old)

    pd.testing.assert_series_equal(
        merged["metric_value_old"],
        merged["metric_value_new"],
        check_names=False,
        rtol=1e-6,
        atol=1e-8,
    )
    pd.testing.assert_series_equal(
        merged["oracle_ari_old"],
        merged["oracle_ari_new"],
        check_names=False,
        rtol=1e-6,
        atol=1e-8,
    )

    # ari_at_k is expected to move only for the generalizability metrics.
    stability_only = merged[~merged["metric_name"].isin(GENERALIZABILITY_METRICS)]
    pd.testing.assert_series_equal(
        stability_only["ari_at_k_old"],
        stability_only["ari_at_k_new"],
        check_names=False,
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.slow
@pytest.mark.parametrize(("scenario_name", "stem"), sorted(UNAFFECTED.items()))
def test_generalizability_ari_changed_as_the_fix_intended(scenario_name, stem, tmp_path):
    """The ari_at_k fix must actually change something, or it did nothing."""
    scenario = SCENARIOS[scenario_name]
    rd = run_scenario(scenario, root=tmp_path, n_jobs=-1, random_state=0)

    new = _comparable(read_run(rd))
    old = _comparable(_load_old(stem))
    merged = old.merge(
        new,
        on=["axis_label", "seed", "metric_name", "k"],
        suffixes=("_old", "_new"),
    )
    gen = merged[merged["metric_name"].isin(GENERALIZABILITY_METRICS)]
    assert not gen["ari_at_k_old"].equals(gen["ari_at_k_new"])
```

- [ ] **Step 2: Verify the test is collected and skipped by default**

Run: `.venv/bin/pytest tests/benchmarks/test_regression.py -v`
Expected: all skipped, with the reason `set CARVE_RUN_REGRESSION=1 ...`

- [ ] **Step 3: Run one scenario for real**

Run: `CARVE_RUN_REGRESSION=1 .venv/bin/pytest tests/benchmarks/test_regression.py -k gaussians -v`
Expected: PASS. This takes tens of minutes.

If `metric_value` differs, the port is wrong; the failure message names the scenario, the axis label, the seed, the metric, and k. Do not widen the tolerance to make it pass. If it differs only for one metric family, look first at the anchor transcription in Task 6 and the estimator mapping in `_ESTIMATORS`.

- [ ] **Step 4: Run the remaining five scenarios and record the outcome**

Run: `CARVE_RUN_REGRESSION=1 .venv/bin/pytest tests/benchmarks/test_regression.py -v`
Expected: all pass.

Write the outcome to `docs/superpowers/notes/2026-09-01-regression-outcome.md`: which scenarios reproduced exactly, which moved, and by how much. This is the raw material for the response letter's account of the corrected numbers.

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/test_regression.py docs/superpowers/notes/2026-09-01-regression-outcome.md
git commit -m "test(benchmarks): add regression harness against committed results"
```

---

### Task 15: CI wiring and the nightly smoke run

**Files:**
- Modify: `.github/workflows/nightly-notebooks.yml`
- Modify: `/Users/kaiwycik/GitHub/CARVE/_claude_playground/CLAUDE.md`
- Create: `tests/benchmarks/test_ci_config.py`

**Interfaces:**
- Consumes: the CLI from Task 11.
- Produces: a nightly job step that runs a real reduced benchmark rather than always hitting a cache.

`ci.yml` needs no change to its ruff steps. `ruff check src/` and `ruff format --check src/` already cover `src/benchmarks/` now that the package lives under `src/`. Confirm that rather than assume it.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_ci_config.py`:

```python
"""CI configuration invariants."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def test_ruff_covers_the_whole_src_tree():
    ci = (WORKFLOWS / "ci.yml").read_text()
    assert "ruff check src/" in ci
    assert "ruff format --check src/" in ci


def test_nightly_runs_a_real_benchmark_not_only_plotting():
    nightly = (WORKFLOWS / "nightly-notebooks.yml").read_text()
    assert "benchmarks.run" in nightly
    assert "--n-seeds" in nightly
    assert "--n-resamples" in nightly
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_ci_config.py -v`
Expected: `test_nightly_runs_a_real_benchmark_not_only_plotting` FAILS; the ruff test passes already, which is the point.

- [ ] **Step 3: Confirm ruff covers the new package**

Run: `.venv/bin/ruff check src/ --statistics && .venv/bin/ruff format --check src/`
Expected: clean, and the file list includes `src/benchmarks/`. Confirm with `.venv/bin/ruff check src/ --show-files | grep benchmarks`.

- [ ] **Step 4: Add the nightly smoke step**

In `.github/workflows/nightly-notebooks.yml`, add a step before the notebook execution step, keeping the existing `OMP_NUM_THREADS: 1` env block:

```yaml
      - name: Benchmark smoke run
        env:
          OMP_NUM_THREADS: 1
        run: |
          python -m benchmarks.run \
            --scenario gaussians \
            --root "${RUNNER_TEMP}/benchmark-runs" \
            --n-seeds 2 \
            --n-resamples 5 \
            --n-jobs 2
          test -f "${RUNNER_TEMP}"/benchmark-runs/gaussians/*/manifest.json
```

Writing into `RUNNER_TEMP` rather than the checkout is what makes this a genuine compute check. The previous arrangement tested `if not os.path.exists(csv)` against a committed CSV, so the cache branch always won and no benchmark ever ran.

Make sure the install step in this workflow includes the new extra: `pip install -e ".[dev,benchmarks,notebooks]"`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/benchmarks/test_ci_config.py -v`
Expected: all passed

- [ ] **Step 6: Update CLAUDE.md**

In `/Users/kaiwycik/GitHub/CARVE/_claude_playground/CLAUDE.md`:

- Under "Commands", change the two ruff lines so the comment reads `# lint  — src/ (carve and benchmarks)` and `# format — src/ (carve and benchmarks)`, and update the bullet that says `tests/` and `notebooks/` are not linted to note that `src/benchmarks/` is.
- Under "Key structural rules", add: `benchmarks/ lives at src/benchmarks/ and is excluded from the wheel. It imports carve; carve never imports it.`
- Under "Do not", change "Do not run ruff on `tests/` or `notebooks/`" to keep its meaning, since `src/` now covers two packages.
- Add a line noting that `notebooks/benchmarking_code/` is being replaced by `src/benchmarks/` and is scheduled for deletion in plan 2.

- [ ] **Step 7: Run the whole suite and lint**

Run: `.venv/bin/pytest -v --tb=short && .venv/bin/ruff check src/ && .venv/bin/ruff format --check src/`
Expected: all pass, ruff clean.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/nightly-notebooks.yml tests/benchmarks/test_ci_config.py
git commit -m "ci(benchmarks): add a real nightly smoke run and confirm ruff coverage"
```

---

## Completion criteria

Plan 1 is done when all of these hold:

1. `.venv/bin/pytest -v --tb=short` passes, including every `tests/benchmarks/` module.
2. `.venv/bin/ruff check src/` and `.venv/bin/ruff format --check src/` are clean.
3. `.venv/bin/python -m benchmarks.run --list` prints the eight scenarios.
4. `CARVE_RUN_REGRESSION=1 .venv/bin/pytest tests/benchmarks/test_regression.py -v` passes, and the outcome is recorded in `docs/superpowers/notes/2026-09-01-regression-outcome.md`.
5. A built wheel contains no `benchmarks/` entries.
6. `manifest.json` from a real run contains the git SHA, package versions, CPU and core count, wall-clock, peak RSS with its unit, and the active anchor set name.

Plan 2 then covers `_theme`, `_panels`, `figures/`, `datasets/`, the notebooks, deleting `notebooks/benchmarking_code/` and the moons scaling CSVs, and the figure and table verification steps from the spec.
