# Rho and B Sensitivity Ablation Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: Measure how CARVE's selections, scores and cluster recovery respond to the subsampling proportion rho and the resample count B, on the six simulation scenarios and the Klein case study, and produce the SI figures, table, notebook and manuscript drafts that answer reviewer comments R4.4 and R1.3.

Architecture: A dedicated ablation runner in `src/benchmarks/_ablation.py` enumerates cells keyed by (study, difficulty, dataset, rho, B, replicate), fits CARVE once per cell through fit-and-score code shared with `run_cell`, and checkpoints six typed parquet frames per run directory. A summary module turns those frames into per-setting statistics that two figure functions, one table renderer and a read-only notebook consume. Configuration lives in `_registry.py` as a frozen `Ablation`; nothing under `src/carve/` changes and no published scenario output moves.

Tech Stack: Python 3.13, numpy, pandas, pyarrow parquet, joblib (loky), scikit-learn, matplotlib, nbformat, pytest, ruff 0.16.4.

Spec: `docs/superpowers/specs/2026-09-16-rho-b-ablation-design.md`

## Global Constraints

- Work in `~/GitHub/CARVE/code` on branch `rho-b-ablation`. Run everything with `.venv/bin/python`, `.venv/bin/pytest` and `.venv/bin/ruff` (ruff is pinned to 0.16.4; never a system ruff). The venv has no pip; installs go through `VIRTUAL_ENV=$PWD/.venv uv pip install`.
- No change to `src/carve/`. No change to any published scenario's output or `config_hash`. No edits under `../overleaf/`.
- `_types.py` is a leaf: it imports only the standard library. `benchmarks` imports `carve`; `carve` never imports `benchmarks`.
- Tests mirror modules one to one (`_ablation.py` <-> `tests/benchmarks/test_ablation.py`). `filterwarnings = error` is on: a new warning fails the suite and needs a `match=` assertion or a targeted mark. Never run ruff on `tests/` or `notebooks/`.
- Configuration is read from the registry (`ABLATIONS`, `SCENARIOS`, `STUDIES`), never restated at a call site. Package defaults for rho and B are read from `dataclasses.fields(CARVE)`.
- Diagnostics use `warnings.warn(..., stacklevel=2)` or `verbose`-gated `print`; no `logging`.
- Figures take every color, font size and line style from `benchmarks._theme`; no module-level color literal.
- Prose in Markdown, notebook cells, docstrings and LaTeX drafts uses American spelling and no bold or italics.
- Parallelism: `run_ablation` defaults to `n_jobs=-1`; each CARVE fit gets `n_jobs=1` and a thread cap of `max(1, cpu_count() // workers)` set through `LOKY_MAX_CPU_COUNT` for the duration of the fit. `--scenario` and `--all` keep `--n-jobs` default 1.
- Seeds: CARVE `random_state = base + replicate * REPLICATE_SEED_SPACING` (1,000,000); similarity draw m uses `base + SIMILARITY_SEED_OFFSET + m` (500,000); `base` is the benchmark seed for simulations and `PUBLISHED_RANDOM_STATE` (42) for Klein.
- Commit after every task with the message shown; end each commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Long runs (the dev run, the timing check) run in the foreground with a long timeout; do not background them and end the turn.

---

## File structure

Created:

- `src/benchmarks/_ablation_cells.py` — cells, units, seeds, unit file names, `arm_view`, `timing_units`. Imports only `_registry`, `_artifacts` and `_types`, so the table and the notebook can read a run without importing carve.
- `src/benchmarks/_ablation.py` — configuration hash, unit executors (dataset, similarity, cell) and `run_ablation`. Imports carve through `_run` and `_studies`.
- `src/benchmarks/_ablation_summary.py` — pure pandas summaries of the ablation frames (recovery, bias, agreement, spread, curves at k*, rare recall, similarity, study selection shares, table rows).
- `src/benchmarks/figures/_ablation.py` — `figure_ablation_rho`, `figure_ablation_b`.
- `notebooks/Resampling_Ablation.ipynb` — reads a run directory; computes nothing.
- `tests/benchmarks/test_ablation_cells.py`, `tests/benchmarks/test_ablation.py`, `tests/benchmarks/test_ablation_summary.py`, `tests/benchmarks/figures/test_ablation.py`.
- `_claude_playground/rho_b_ablation/manuscript_drafts.tex` (outside the repo).

Modified:

- `src/benchmarks/_types.py` — `ArmScale`, `AblationScale`, `Ablation`; `Study.not_two`.
- `src/benchmarks/_registry.py` — seed constants, `package_defaults`, `ABLATION_SCALES`, `ABLATIONS`, `validate_ablation`.
- `src/benchmarks/_studies.py` — `STUDIES["klein"].not_two = True`.
- `src/benchmarks/_run.py` — `benchmark_seed`, `cpu_cap`, `fit_carve`, `labels_by_mode`, `smallest_cluster`, `rare_cluster_recall`; `run_cell` and `run_scenario` use them.
- `src/benchmarks/_artifacts.py` — ablation schemas, `scenario_identity`, `provenance`, `write_frame`, `read_frames`, `write_manifest` accepting a mapping.
- `src/benchmarks/_tables.py` — `render_ablation_tex`, `write_ablation_table`.
- `src/benchmarks/figures/__init__.py`, `src/benchmarks/run.py`.
- `tests/benchmarks/test_benchmarks_types.py`, `test_registry.py`, `test_run.py`, `test_artifacts.py`, `test_studies.py`, `test_tables.py`, `test_cli.py`, `test_notebooks.py`, `figures/test_figure_contract.py`.

Cell key columns, used by every frame that has one cell per fit:

```python
CELL_KEY = ("study", "difficulty", "dataset", "subsample_ratio", "n_resamples", "replicate")
```

The study (Klein) has `difficulty == ""` and `dataset == 0`; it has one dataset.

---

### Task 1: Ablation configuration types and `Study.not_two`

Files:
- Modify: `src/benchmarks/_types.py` (after `Study`, before `Manifest`; `Study` gains a field)
- Modify: `src/benchmarks/_studies.py:501-509` (`STUDIES["klein"]`)
- Test: `tests/benchmarks/test_benchmarks_types.py`, `tests/benchmarks/test_studies.py`, `tests/benchmarks/test_notebooks.py`

Interfaces:
- Produces: `ArmScale(difficulties, datasets, replicates, study_replicates)`, `AblationScale(rho_arm, b_arm, similarity_draws, study_scale, n_total=None)`, `Ablation(name, scenarios, study, rho_grid, b_grid, rho_default, b_default, scales, default_scale)`; `Study.not_two: bool = False`.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_benchmarks_types.py` (extend the import to include `Ablation, AblationScale, ArmScale`):

```python
def _arm(**kw):
    base = dict(difficulties=("easy",), datasets=(0, 1), replicates=1, study_replicates=1)
    base.update(kw)
    return ArmScale(**base)


def _scale(**kw):
    base = dict(rho_arm=_arm(), b_arm=_arm(), similarity_draws=2, study_scale="dev")
    base.update(kw)
    return AblationScale(**base)


def _ablation(**kw):
    base = dict(
        name="probe",
        scenarios=("gaussians",),
        study="klein",
        rho_grid=(0.5, 0.618),
        b_grid=(10, 100),
        rho_default=0.618,
        b_default=100,
        scales={"dev": _scale()},
        default_scale="dev",
    )
    base.update(kw)
    return Ablation(**base)


class TestArmScale:
    def test_rejects_an_empty_dataset_list(self):
        with pytest.raises(ValueError, match="dataset"):
            _arm(datasets=())

    def test_rejects_repeated_datasets(self):
        with pytest.raises(ValueError, match="repeat"):
            _arm(datasets=(0, 0))

    def test_rejects_zero_replicates(self):
        with pytest.raises(ValueError, match="replicates"):
            _arm(replicates=0)


class TestAblationScale:
    def test_rejects_zero_similarity_draws(self):
        with pytest.raises(ValueError, match="similarity_draws"):
            _scale(similarity_draws=0)

    def test_n_total_defaults_to_none(self):
        assert _scale().n_total is None


class TestAblation:
    def test_constructs(self):
        assert _ablation().name == "probe"

    def test_rejects_a_default_rho_missing_from_its_grid(self):
        with pytest.raises(ValueError, match="rho_default"):
            _ablation(rho_default=0.7)

    def test_rejects_a_default_b_missing_from_its_grid(self):
        with pytest.raises(ValueError, match="b_default"):
            _ablation(b_default=50)

    def test_rejects_an_unsorted_rho_grid(self):
        with pytest.raises(ValueError, match="increasing"):
            _ablation(rho_grid=(0.618, 0.5))

    def test_rejects_rho_outside_the_open_unit_interval(self):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            _ablation(rho_grid=(0.618, 1.0))

    def test_rejects_b_below_two(self):
        with pytest.raises(ValueError, match="at least 2"):
            _ablation(b_grid=(1, 100))

    def test_rejects_an_undeclared_default_scale(self):
        with pytest.raises(ValueError, match="default_scale"):
            _ablation(default_scale="publication")

    def test_rejects_repeated_scenarios(self):
        with pytest.raises(ValueError, match="repeat"):
            _ablation(scenarios=("gaussians", "gaussians"))


class TestStudyNotTwo:
    def test_defaults_to_false(self):
        study = Study(
            name="demo",
            loader=lambda subsample: (None, None, {}),
            estimator=EstimatorSpec(name="kmeans"),
            candidate_k=(2, 3),
            scales={"dev": 100},
            default_scale="dev",
        )
        assert study.not_two is False
```

Append to `tests/benchmarks/test_studies.py` inside the class that holds `test_klein_sweeps_k_two_through_ten` (it imports `STUDIES` already):

```python
    def test_klein_selection_excludes_k_two(self):
        # The manuscript's Klein result (Ward at k=4) is selected with
        # not_two=True; the notebook and the ablation both read it from here.
        assert STUDIES["klein"].not_two is True
```

Append to `tests/benchmarks/test_notebooks.py` after `test_klein_prepare_composite_uses_the_generalizability_selection`:

```python
def test_klein_notebook_not_two_agrees_with_the_study():
    from benchmarks._studies import STUDIES

    source = _code(NOTEBOOKS["klein"])
    assert ("not_two=True" in source) == STUDIES["klein"].not_two
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/test_benchmarks_types.py tests/benchmarks/test_studies.py -k "ArmScale or AblationScale or TestAblation or NotTwo or excludes_k_two" -q`
Expected: ImportError on `ArmScale` (collection error) — the test file cannot import the new names.

- [ ] Step 3: Add the dataclasses and the Study field

In `src/benchmarks/_types.py`, add `not_two: bool = False` as the last field of `Study`, directly after `preprocessing: PreprocessingSpec | None = None`, and extend the class docstring's last paragraph with:

```
    not_two is the study's selection setting: True excludes k=2 from every
    selection made on its fits, as the Klein notebook does for the
    manuscript's headline result.
```

Then insert after the `Study` class and before `Manifest`:

```python
@dataclass(frozen=True)
class ArmScale:
    """How much of the grid one ablation arm runs at one scale.

    difficulties and datasets name the simulated cells; replicates is the
    number of CARVE fits with distinct seeds per simulated cell, and
    study_replicates the same for the case study, which has one dataset.
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
        if self.replicates < 1 or self.study_replicates < 1:
            raise ValueError("ArmScale: replicates and study_replicates must be at least 1.")


@dataclass(frozen=True)
class AblationScale:
    """One scale of an ablation: how much runs, not what is measured.

    n_total overrides the simulated sample count at a development scale;
    None keeps each scenario's own. study_scale names the case study's scale
    (a key of Study.scales).
    """

    rho_arm: ArmScale
    b_arm: ArmScale
    similarity_draws: int
    study_scale: str
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
    this module stays a leaf.
    """

    name: str
    scenarios: tuple[str, ...]
    study: str
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
            raise ValueError(f"Ablation {self.name!r}: scenarios repeat: {self.scenarios}.")
        if list(self.rho_grid) != sorted(set(self.rho_grid)):
            raise ValueError(
                f"Ablation {self.name!r}: rho_grid must be strictly increasing."
            )
        if any(not 0.0 < rho < 1.0 for rho in self.rho_grid):
            raise ValueError(f"Ablation {self.name!r}: every rho must lie in (0, 1).")
        if list(self.b_grid) != sorted(set(self.b_grid)):
            raise ValueError(f"Ablation {self.name!r}: b_grid must be strictly increasing.")
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
```

In `src/benchmarks/_studies.py`, add `not_two=True,` as the last argument of `STUDIES["klein"]`, with the comment:

```python
        # The manuscript's headline selection (Ward at k=4, generalizability,
        # 1SE) excludes k=2; the Klein notebook and the ablation read it here.
        not_two=True,
```

- [ ] Step 4: Run the tests

Run: `.venv/bin/pytest tests/benchmarks/test_benchmarks_types.py tests/benchmarks/test_studies.py tests/benchmarks/test_notebooks.py -q`
Expected: all pass.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_types.py src/benchmarks/_studies.py tests/benchmarks/test_benchmarks_types.py tests/benchmarks/test_studies.py tests/benchmarks/test_notebooks.py
git commit -m "feat(benchmarks): ablation configuration types and Study.not_two

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Register the rho_b ablation

Files:
- Modify: `src/benchmarks/_registry.py` (imports; new section after `PUBLISHED_RANDOM_STATE`)
- Test: `tests/benchmarks/test_registry.py`

Interfaces:
- Consumes: `Ablation`, `AblationScale`, `ArmScale` from Task 1.
- Produces: `REPLICATE_SEED_SPACING: int`, `SIMILARITY_SEED_OFFSET: int`, `package_defaults() -> tuple[float, int]`, `ABLATION_SCALES: dict[str, AblationScale]`, `ABLATIONS: dict[str, Ablation]`, `validate_ablation(ablation) -> None`, `max_benchmark_seed_offset(ablation) -> int`.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_registry.py` (extend the `from benchmarks._registry import (...)` block with `ABLATIONS, REPLICATE_SEED_SPACING, SIMILARITY_SEED_OFFSET, package_defaults, validate_ablation`; add `import dataclasses` and `from benchmarks._types import AblationScale, ArmScale`):

```python
class TestAblationRegistry:
    def test_rho_b_is_registered(self):
        assert "rho_b" in ABLATIONS
        assert ABLATIONS["rho_b"].name == "rho_b"

    def test_grids_are_the_spec_values(self):
        ablation = ABLATIONS["rho_b"]
        assert ablation.rho_grid == (0.2, 0.3, 0.4, 0.5, 0.618, 0.7, 0.8, 0.9)
        assert ablation.b_grid == (10, 25, 50, 100, 200)

    def test_defaults_are_read_from_carve(self):
        from dataclasses import fields

        from carve import CARVE

        defaults = {f.name: f.default for f in fields(CARVE)}
        assert package_defaults() == (
            defaults["subsample_ratio"],
            defaults["n_resamples"],
        )
        assert ABLATIONS["rho_b"].rho_default == defaults["subsample_ratio"]
        assert ABLATIONS["rho_b"].b_default == defaults["n_resamples"]

    def test_runs_the_six_difficulty_scenarios_and_klein(self):
        ablation = ABLATIONS["rho_b"]
        assert set(ablation.scenarios) == {
            "gaussians", "t_dist", "t_dist_noise", "circles", "moons", "swiss_rolls",
        }
        assert ablation.study == "klein"

    def test_publication_scale_matches_the_spec(self):
        scale = ABLATIONS["rho_b"].scales["publication"]
        assert scale.rho_arm == ArmScale(("easy", "medium", "hard"), tuple(range(10)), 1, 10)
        assert scale.b_arm == ArmScale(("medium", "hard"), tuple(range(5)), 3, 10)
        assert scale.similarity_draws == 20
        assert scale.n_total is None
        assert scale.study_scale == "publication"

    def test_dev_scale_is_small_and_overrides_n_total(self):
        scale = ABLATIONS["rho_b"].scales["dev"]
        assert scale.rho_arm == ArmScale(("medium",), (0, 1), 1, 2)
        assert scale.b_arm == ArmScale(("medium",), (0, 1), 2, 2)
        assert scale.similarity_draws == 5
        assert scale.n_total == 500
        assert scale.study_scale == "dev"

    def test_default_scale_is_publication(self):
        assert ABLATIONS["rho_b"].default_scale == "publication"

    def test_rejects_a_scenario_that_is_not_registered(self):
        bad = dataclasses.replace(ABLATIONS["rho_b"], scenarios=("nope",))
        with pytest.raises(ValueError, match="nope"):
            validate_ablation(bad)

    def test_rejects_a_scaling_scenario(self):
        bad = dataclasses.replace(ABLATIONS["rho_b"], scenarios=("gaussians_samples",))
        with pytest.raises(ValueError, match="difficulty"):
            validate_ablation(bad)

    def test_rejects_an_unknown_difficulty(self):
        scale = ABLATIONS["rho_b"].scales["dev"]
        bad_scale = dataclasses.replace(
            scale, rho_arm=dataclasses.replace(scale.rho_arm, difficulties=("brutal",))
        )
        bad = dataclasses.replace(ABLATIONS["rho_b"], scales={"dev": bad_scale}, default_scale="dev")
        with pytest.raises(ValueError, match="brutal"):
            validate_ablation(bad)

    def test_seed_constants_keep_replicates_and_similarity_apart(self):
        # One fit's seeds span 3 * B; bases differ by up to two axis steps
        # plus the largest dataset index. The spacing must clear both.
        ablation = ABLATIONS["rho_b"]
        span = 3 * max(ablation.b_grid)
        assert REPLICATE_SEED_SPACING == 1_000_000
        assert SIMILARITY_SEED_OFFSET == 500_000
        assert SIMILARITY_SEED_OFFSET > span
        assert SIMILARITY_SEED_OFFSET + 20 < REPLICATE_SEED_SPACING - span

    def test_validation_rejects_a_spacing_inside_one_fits_seed_span(self, monkeypatch):
        import benchmarks._registry as registry

        monkeypatch.setattr(registry, "REPLICATE_SEED_SPACING", 100)
        with pytest.raises(ValueError, match="REPLICATE_SEED_SPACING"):
            validate_ablation(ABLATIONS["rho_b"])

    def test_validation_rejects_a_similarity_offset_inside_a_replicate_window(
        self, monkeypatch
    ):
        import benchmarks._registry as registry

        monkeypatch.setattr(registry, "SIMILARITY_SEED_OFFSET", 10)
        with pytest.raises(ValueError, match="SIMILARITY_SEED_OFFSET"):
            validate_ablation(ABLATIONS["rho_b"])
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/test_registry.py -q`
Expected: ImportError on `ABLATIONS`.

- [ ] Step 3: Add the registry section

In `src/benchmarks/_registry.py`, change the import to `from ._types import Ablation, AblationScale, ArmScale, Axis, EstimatorSpec, Scenario` and append after `PUBLISHED_RANDOM_STATE`:

```python
# =============================================================================
# Ablations
# =============================================================================
# One CARVE fit draws its subsamples from seeds in [random_state,
# random_state + 3 * n_resamples): P_1 at random_state + b, P_2 at
# random_state + b + B and the P_test pipeline at random_state + b + 2B.
# Replicate fits of one dataset are spaced by this much so no two share a
# subsample; the spacing also clears the largest benchmark seed offset
# (two axis steps of 10,000 plus the dataset index), so cells of different
# datasets never collide either.
REPLICATE_SEED_SPACING: int = 1_000_000

# Subsample-versus-full similarity draws seed from base + this + draw; it
# sits above replicate 0's window and below replicate 1's.
SIMILARITY_SEED_OFFSET: int = 500_000


def package_defaults() -> tuple[float, int]:
    """CARVE's own (subsample_ratio, n_resamples) defaults, read off the class.

    Imported lazily so this module's top level stays free of carve, as
    _types does; Scenario already imports carve.sim at construction.
    """
    from dataclasses import fields

    from carve import CARVE

    defaults = {field.name: field.default for field in fields(CARVE)}
    return float(defaults["subsample_ratio"]), int(defaults["n_resamples"])


_RHO_DEFAULT, _B_DEFAULT = package_defaults()

ABLATION_SCALES: dict[str, AblationScale] = {
    "publication": AblationScale(
        rho_arm=ArmScale(
            difficulties=DIFFICULTY_AXIS.labels,
            datasets=tuple(range(10)),
            replicates=1,
            study_replicates=10,
        ),
        b_arm=ArmScale(
            difficulties=("medium", "hard"),
            datasets=tuple(range(5)),
            replicates=3,
            study_replicates=10,
        ),
        similarity_draws=20,
        study_scale="publication",
    ),
    # Exercises every code path and figure; its numbers are not reported.
    "dev": AblationScale(
        rho_arm=ArmScale(
            difficulties=("medium",), datasets=(0, 1), replicates=1, study_replicates=2
        ),
        b_arm=ArmScale(
            difficulties=("medium",), datasets=(0, 1), replicates=2, study_replicates=2
        ),
        similarity_draws=5,
        study_scale="dev",
        n_total=500,
    ),
}

ABLATIONS: dict[str, Ablation] = {
    "rho_b": Ablation(
        name="rho_b",
        scenarios=("gaussians", "t_dist", "t_dist_noise", "circles", "moons", "swiss_rolls"),
        study="klein",
        # 0.5 and 0.8 are proportions used elsewhere in the resampling
        # literature, so the default can be placed against them.
        rho_grid=(0.2, 0.3, 0.4, 0.5, _RHO_DEFAULT, 0.7, 0.8, 0.9),
        b_grid=(10, 25, 50, _B_DEFAULT, 200),
        rho_default=_RHO_DEFAULT,
        b_default=_B_DEFAULT,
        scales=ABLATION_SCALES,
        default_scale="publication",
    ),
}


def max_benchmark_seed_offset(ablation: Ablation) -> int:
    """Largest benchmark seed minus PUBLISHED_RANDOM_STATE over the ablation's cells."""
    offset = 0
    for scale in ablation.scales.values():
        for arm in (scale.rho_arm, scale.b_arm):
            for label in arm.difficulties:
                if label in DIFFICULTY_AXIS.labels:
                    idx = DIFFICULTY_AXIS.labels.index(label)
                    offset = max(offset, idx * 10000 + max(arm.datasets))
    return offset


def validate_ablation(ablation: Ablation) -> None:
    """Check an ablation against the registry it refers to.

    Ablation.__post_init__ checks what the dataclass can see on its own.
    This checks the rest: scenario names, the difficulty axis, and the seed
    constants against the widest seed span any cell can use.
    """
    for name in ablation.scenarios:
        if name not in SCENARIOS:
            raise ValueError(
                f"Ablation {ablation.name!r}: scenario {name!r} is not registered."
            )
        if SCENARIOS[name].axis.name != DIFFICULTY_AXIS.name:
            raise ValueError(
                f"Ablation {ablation.name!r}: scenario {name!r} sweeps "
                f"{SCENARIOS[name].axis.name!r}, not the difficulty axis."
            )
    for scale_name, scale in ablation.scales.items():
        for arm in (scale.rho_arm, scale.b_arm):
            unknown = sorted(set(arm.difficulties) - set(DIFFICULTY_AXIS.labels))
            if unknown:
                raise ValueError(
                    f"Ablation {ablation.name!r}, scale {scale_name!r}: unknown "
                    f"difficulty label(s) {unknown}."
                )
    span = 3 * max(ablation.b_grid)
    offset = max_benchmark_seed_offset(ablation)
    draws = max(scale.similarity_draws for scale in ablation.scales.values())
    if REPLICATE_SEED_SPACING <= span + offset:
        raise ValueError(
            f"REPLICATE_SEED_SPACING={REPLICATE_SEED_SPACING} does not clear one "
            f"fit's seed span ({span}) plus the seed offset ({offset})."
        )
    if SIMILARITY_SEED_OFFSET < span + offset:
        raise ValueError(
            f"SIMILARITY_SEED_OFFSET={SIMILARITY_SEED_OFFSET} lies inside replicate 0's "
            f"seed window (span {span}, offset {offset})."
        )
    if SIMILARITY_SEED_OFFSET + draws + offset > REPLICATE_SEED_SPACING - span:
        raise ValueError(
            f"SIMILARITY_SEED_OFFSET={SIMILARITY_SEED_OFFSET} plus {draws} draws reaches "
            f"replicate 1's seed window."
        )


for _ablation in ABLATIONS.values():
    validate_ablation(_ablation)
```

Note `DIFFICULTY_AXIS.labels` is already a tuple `("easy", "medium", "hard")`, so `ArmScale.difficulties` receives it as is.

- [ ] Step 4: Run the tests

Run: `.venv/bin/pytest tests/benchmarks/test_registry.py tests/benchmarks/test_artifacts.py -q`
Expected: all pass (the artifacts tests confirm `config_hash` is untouched).

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_registry.py tests/benchmarks/test_registry.py
git commit -m "feat(benchmarks): register the rho_b ablation and its seed constants

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Shared fit-and-score code in `_run.py`, with the thread cap

Files:
- Modify: `src/benchmarks/_run.py` (imports; new helpers before `run_cell`; `run_cell` lines 92-146 and 190-220; `run_scenario` lines 349-368)
- Test: `tests/benchmarks/test_run.py`

Interfaces:
- Produces:
  - `benchmark_seed(seed: int, axis_idx: int, random_state: int) -> int`
  - `cpu_cap(cap: int | None)` context manager
  - `thread_cap_for(n_jobs: int) -> tuple[int, int]` returning `(workers, cap)`
  - `CLUSTER_COUNT_WARNING: str` regex
  - `CarveFit(carve, fit_seconds, n_cluster_count_warnings)` frozen dataclass
  - `fit_carve(X, *, grids, n_resamples, n_trees, random_state, subsample_ratio=None, thread_cap=None) -> CarveFit`
  - `labels_by_mode(carve, y, *, candidate_k, rare_label=None) -> dict[str, dict[int, dict[str, float]]]` with keys `"ari"` and, when `rare_label` is given, `"rare_recall"`
  - `smallest_cluster(y) -> tuple[Any, float]` (label, size fraction)
  - `rare_cluster_recall(y, labels, rare_label) -> float`
  - `run_cell(..., thread_cap: int | None = None)`

- [ ] Step 1: Record the current `run_cell` output for the before/after comparison

Run (scratch only, nothing committed):

```bash
.venv/bin/python - <<'PY'
import json
from benchmarks._registry import SCENARIOS
from benchmarks._run import run_cell
import dataclasses
scenario = dataclasses.replace(SCENARIOS["gaussians"], shared={**SCENARIOS["gaussians"].shared, "n_total": 300})
rows, runtime = run_cell(scenario, axis_idx=1, axis_value=1, axis_label="medium", seed=0,
                         run_id="probe", random_state=42, n_resamples=20)
json.dump([(r["metric_name"], r["k"], r["metric_value"], r["is_selected"], r["ari_at_k"]) for r in rows],
          open("/tmp/run_cell_before.json", "w"))
print(len(rows), "rows saved")
PY
```

Expected: `85 rows saved` (13 CARVE metrics plus 4 classical indices, times 5 candidate k).

- [ ] Step 2: Write the failing tests

Append to `tests/benchmarks/test_run.py` (extend the `from benchmarks._run import ...` block with `CLUSTER_COUNT_WARNING, benchmark_seed, cpu_cap, fit_carve, labels_by_mode, rare_cluster_recall, smallest_cluster, thread_cap_for`; add `import os`, `from joblib import Parallel, cpu_count, delayed`, `from benchmarks._estimators import param_grids`):

```python
class TestBenchmarkSeed:
    def test_matches_the_published_derivation(self):
        assert benchmark_seed(3, 2, 42) == 3 + 20000 + 42


class TestSmallestCluster:
    def test_returns_the_smallest_label_and_its_fraction(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        assert smallest_cluster(y) == (2, 1 / 6)

    def test_ties_go_to_the_lowest_label(self):
        y = np.array([0, 1, 1, 2, 3, 3])
        assert smallest_cluster(y)[0] == 0


class TestRareClusterRecall:
    def test_perfect_recovery_is_one(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        labels = np.array([5, 5, 5, 7, 7, 9])
        assert rare_cluster_recall(y, labels, rare_label=2) == 1.0

    def test_a_rare_cluster_merged_into_another_is_zero(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        labels = np.array([0, 0, 0, 1, 1, 1])
        assert rare_cluster_recall(y, labels, rare_label=2) == 0.0

    def test_fewer_clusters_than_truth_leaves_the_rare_one_unmatched(self):
        y = np.array([0, 0, 0, 1, 1, 2, 2])
        labels = np.array([0, 0, 0, 1, 1, 1, 1])
        assert rare_cluster_recall(y, labels, rare_label=2) == 0.0

    def test_partial_recovery_is_the_matched_fraction(self):
        y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 1, 1])
        assert rare_cluster_recall(y, labels, rare_label=2) == pytest.approx(0.5)

    def test_rejects_a_label_absent_from_the_truth(self):
        with pytest.raises(ValueError, match="rare_label"):
            rare_cluster_recall(np.array([0, 1]), np.array([0, 1]), rare_label=7)


def _budget_under_cap(cap):
    from carve._utils import resolve_core_budget

    with cpu_cap(cap):
        inside = resolve_core_budget(1, n_resamples=100)
    after = resolve_core_budget(1, n_resamples=100)
    return inside, after, os.environ.get("LOKY_MAX_CPU_COUNT")


class TestCpuCap:
    def test_reaches_carves_core_budget_inside_loky_workers(self):
        cores = cpu_count()
        cap = max(1, cores // 2)
        results = Parallel(n_jobs=2)(delayed(_budget_under_cap)(cap) for _ in range(4))
        for inside, after, env in results:
            assert inside == (1, cap)
            assert after == (1, cores)
            assert env is None

    def test_none_leaves_the_environment_alone(self, monkeypatch):
        monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
        with cpu_cap(None):
            assert "LOKY_MAX_CPU_COUNT" not in os.environ

    def test_restores_a_previous_value(self, monkeypatch):
        monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "3")
        with cpu_cap(1):
            assert os.environ["LOKY_MAX_CPU_COUNT"] == "1"
        assert os.environ["LOKY_MAX_CPU_COUNT"] == "3"

    def test_thread_cap_for_splits_the_machine(self):
        workers, cap = thread_cap_for(1)
        assert (workers, cap) == (1, cpu_count())
        workers, cap = thread_cap_for(-1)
        assert workers == cpu_count()
        assert cap == 1


class TestFitCarve:
    def test_regex_matches_carves_own_message(self):
        # The literal format string lives in carve/_runner.py:
        # f"labels_1 has {k_1} clusters, expected {expected}".
        assert re.match(CLUSTER_COUNT_WARNING, "labels_1 has 3 clusters, expected 4")
        assert re.match(CLUSTER_COUNT_WARNING, "labels_test has 2 clusters, expected 10")
        assert not re.match(CLUSTER_COUNT_WARNING, "Non-default mode is experimental")

    def test_counts_cluster_count_warnings_without_raising(self, monkeypatch):
        # filterwarnings=error is on, so if fit_carve did not intercept these
        # they would raise here instead of being counted.
        import benchmarks._run as run_module

        class ShortCarve(run_module.CARVE):
            def fit(self, X, **kwargs):
                warnings.warn("labels_1 has 3 clusters, expected 4", stacklevel=2)
                warnings.warn("labels_test has 2 clusters, expected 4", stacklevel=2)
                return super().fit(X, **kwargs)

        monkeypatch.setattr(run_module, "CARVE", ShortCarve)
        X = np.random.default_rng(0).normal(size=(60, 3))
        fit = fit_carve(
            X,
            grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
            n_resamples=4,
            n_trees=10,
            random_state=0,
            subsample_ratio=0.5,
        )
        assert fit.n_cluster_count_warnings == 2
        assert fit.fit_seconds > 0
        assert fit.carve.estimator_results_ is not None

    def test_re_emits_other_warnings(self, tiny_scenario, monkeypatch):
        import benchmarks._run as run_module

        class NoisyCarve(run_module.CARVE):
            def fit(self, X, **kwargs):
                warnings.warn("something else entirely", stacklevel=2)
                return super().fit(X, **kwargs)

        monkeypatch.setattr(run_module, "CARVE", NoisyCarve)
        X = np.random.default_rng(0).normal(size=(60, 3))
        with pytest.warns(UserWarning, match="something else entirely"):
            fit_carve(
                X,
                grids=param_grids(tiny_scenario.estimator, (2, 3)),
                n_resamples=4,
                n_trees=10,
                random_state=0,
            )

    def test_subsample_ratio_none_keeps_the_constructor_kwargs(self, recording_carve):
        X = np.random.default_rng(0).normal(size=(60, 3))
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        fit_carve(X, grids=grids, n_resamples=4, n_trees=10, random_state=0)
        assert "subsample_ratio" not in recording_carve.init_kwargs[0]
        fit_carve(X, grids=grids, n_resamples=4, n_trees=10, random_state=0, subsample_ratio=0.5)
        assert recording_carve.init_kwargs[1]["subsample_ratio"] == 0.5


class TestLabelsByMode:
    def test_scores_every_mode_and_k(self, tiny_scenario):
        X, y = simulate(tiny_scenario, axis_value=0, axis_label="easy", seed=0)
        fit = fit_carve(
            X,
            grids=param_grids(tiny_scenario.estimator, tiny_scenario.candidate_k),
            n_resamples=8,
            n_trees=10,
            random_state=0,
        )
        rare_label, _ = smallest_cluster(y)
        scores = labels_by_mode(
            fit.carve, y, candidate_k=tiny_scenario.candidate_k, rare_label=rare_label
        )
        assert set(scores) == {"default", "generalizability"}
        for mode in scores:
            assert set(scores[mode]) == set(tiny_scenario.candidate_k)
            for k in scores[mode]:
                assert -1.0 <= scores[mode][k]["ari"] <= 1.0
                assert 0.0 <= scores[mode][k]["rare_recall"] <= 1.0

    def test_omits_rare_recall_without_a_rare_label(self, tiny_scenario):
        X, y = simulate(tiny_scenario, axis_value=0, axis_label="easy", seed=0)
        fit = fit_carve(
            X,
            grids=param_grids(tiny_scenario.estimator, (3,)),
            n_resamples=8,
            n_trees=10,
            random_state=0,
        )
        scores = labels_by_mode(fit.carve, y, candidate_k=(3,))
        assert set(scores["default"][3]) == {"ari"}
```

`simulate` and `EstimatorSpec` may already be imported in this file; add `from benchmarks._simulate import simulate` and `from benchmarks._types import EstimatorSpec` if not. Add `import re` and `import warnings`.

Note on `test_re_emits_other_warnings`: `pytest.warns` installs an `always` filter, so the noisy warning is recorded inside `fit_carve` rather than raised, and the assertion sees the re-emitted copy. Outside `pytest.warns`, the suite's `error` filter would raise it inside the fit, which is the intended behavior for an unexpected warning.

One behavior change for scenario runs to be aware of: `run_cell` now goes through `fit_carve`, so cluster-count warnings during a scenario fit are counted and swallowed rather than printed. The metric rows are unchanged (Step 6 checks this).

- [ ] Step 3: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/test_run.py -q -k "BenchmarkSeed or SmallestCluster or RareCluster or CpuCap or FitCarve or LabelsByMode"`
Expected: ImportError on `benchmark_seed`.

- [ ] Step 4: Add the helpers and rewire `run_cell` and `run_scenario`

In `src/benchmarks/_run.py`, extend the imports:

```python
import json
import os
import re
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, cpu_count, delayed, effective_n_jobs
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

from carve import CARVE
from carve._utils import align_cluster_labels
```

Insert before `def _labels_mode`:

```python
# CARVE warns once per subsample whose clustering did not reach the
# requested k. At a large rho the hold-out set is small and this fires
# often; the ablation counts it per cell instead of letting it escalate.
CLUSTER_COUNT_WARNING: str = r"^labels_\w+ has \d+ clusters, expected \d+$"


def benchmark_seed(seed: int, axis_idx: int, random_state: int) -> int:
    """The seed a cell simulates and fits with: the published derivation."""
    return int(seed + (axis_idx * 10000) + random_state)


@contextmanager
def cpu_cap(cap: int | None):
    """Cap the CPU count joblib reports for the duration of the block.

    Inside a loky worker, joblib caps OpenMP and BLAS at one thread but
    joblib.cpu_count() still reports the whole machine, so a CARVE fit with
    n_jobs=1 hands its random forest every core and a pool of such fits
    oversubscribes the machine. loky's cpu_count takes the minimum of the
    system count and LOKY_MAX_CPU_COUNT, which is what CARVE's core budget
    reads, so setting the variable here is enough. None caps nothing.
    """
    if cap is None:
        yield
        return
    previous = os.environ.get("LOKY_MAX_CPU_COUNT")
    os.environ["LOKY_MAX_CPU_COUNT"] = str(int(cap))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LOKY_MAX_CPU_COUNT", None)
        else:
            os.environ["LOKY_MAX_CPU_COUNT"] = previous


def thread_cap_for(n_jobs: int) -> tuple[int, int]:
    """Resolve n_jobs to (workers, threads per fit) so the product fits the machine."""
    workers = max(1, int(effective_n_jobs(n_jobs)))
    return workers, max(1, cpu_count() // workers)


@dataclass(frozen=True)
class CarveFit:
    """One CARVE fit with what the benchmarks record about it."""

    carve: CARVE
    fit_seconds: float
    n_cluster_count_warnings: int


def fit_carve(
    X: np.ndarray,
    *,
    grids: list[tuple[type, dict[str, list[Any]]]],
    n_resamples: int,
    n_trees: int,
    random_state: int,
    subsample_ratio: float | None = None,
    thread_cap: int | None = None,
) -> CarveFit:
    """Fit CARVE the way every benchmark fits it: one worker, timed.

    subsample_ratio=None leaves CARVE's default in place and passes nothing,
    so a scenario cell constructs CARVE with exactly the arguments it did
    before this function existed. Cluster-count warnings are counted and
    swallowed; every other warning is re-emitted after the fit.
    """
    optional = {} if subsample_ratio is None else {"subsample_ratio": float(subsample_ratio)}
    carve = CARVE(
        estimator_param_grids=grids,
        n_resamples=n_resamples,
        n_trees=n_trees,
        n_jobs=1,
        random_state=random_state,
        **optional,
    )
    with warnings.catch_warnings(record=True) as caught, cpu_cap(thread_cap):
        warnings.filterwarnings("always", message=CLUSTER_COUNT_WARNING)
        t0 = time.perf_counter()
        carve.fit(X)
        elapsed = time.perf_counter() - t0
    n_count = 0
    for record in caught:
        if re.match(CLUSTER_COUNT_WARNING, str(record.message)):
            n_count += 1
        else:
            warnings.warn(record.message, stacklevel=2)
    return CarveFit(carve=carve, fit_seconds=float(elapsed), n_cluster_count_warnings=n_count)


def smallest_cluster(y: np.ndarray) -> tuple[Any, float]:
    """The label of the smallest true cluster and its size fraction.

    Ties go to the lowest label, because np.unique sorts and argmin returns
    the first minimum.
    """
    labels, counts = np.unique(np.asarray(y), return_counts=True)
    index = int(np.argmin(counts))
    return labels[index].item(), float(counts[index] / counts.sum())


def rare_cluster_recall(y: np.ndarray, labels: np.ndarray, rare_label: Any) -> float:
    """Fraction of the rare true cluster's samples the aligned labels recover.

    Labels are aligned to the truth by Hungarian matching. A true cluster the
    matching leaves unassigned, which happens whenever there are fewer
    predicted clusters than true ones, scores 0.
    """
    y = np.asarray(y)
    mask = y == rare_label
    if not mask.any():
        raise ValueError(f"rare_label {rare_label!r} does not occur in y.")
    aligned = align_cluster_labels(y, np.asarray(labels))
    return float(np.mean(aligned[mask] == rare_label))


def labels_by_mode(
    carve: CARVE,
    y: np.ndarray,
    *,
    candidate_k: tuple[int, ...] | list[int],
    rare_label: Any | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """ARI (and rare-cluster recall) of the consensus labels at every k and mode.

    Labels depend only on the consensus matrix a metric is cut from, so
    they are computed once per mode rather than once per metric.
    """
    scores: dict[str, dict[int, dict[str, float]]] = {}
    for mode in ("default", "generalizability"):
        per_k: dict[int, dict[str, float]] = {}
        for k in candidate_k:
            labels = carve.get_labels(k=int(k), mode=mode)
            entry = {"ari": float(adjusted_rand_score(y, labels))}
            if rare_label is not None:
                entry["rare_recall"] = rare_cluster_recall(y, labels, rare_label)
            per_k[int(k)] = entry
        scores[mode] = per_k
    return scores
```

Rewrite the top of `run_cell` so it reads:

```python
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
    timing_fits: bool = False,
    thread_cap: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
```

with the docstring gaining the sentence `thread_cap bounds the threads the fit's classifier may use; None leaves every core available, as a single-worker run had before.` Then replace the body from `benchmark_seed = seed + ...` through `t_default_s = time.perf_counter() - t0` with:

```python
    cell_seed = benchmark_seed(seed, axis_idx, random_state)
    candidate_k = list(scenario.candidate_k)

    X, y = simulate(scenario, axis_value=axis_value, axis_label=axis_label, seed=cell_seed)

    oracle = build_estimator(
        scenario.estimator, n_clusters=scenario.k_star, random_state=cell_seed
    )
    oracle_ari = float(adjusted_rand_score(y, oracle.fit_predict(X)))

    grids = param_grids(scenario.estimator, candidate_k)
    fit = fit_carve(
        X,
        grids=grids,
        n_resamples=n_resamples,
        n_trees=scenario.n_trees,
        random_state=cell_seed,
        thread_cap=thread_cap,
    )
    carve = fit.carve
    t_default_s = fit.fit_seconds
```

Replace every later `benchmark_seed` reference in `run_cell` (the CVI `labels_by_k` and `calculate_cvi` calls, and the timing fits' `random_state`) with `cell_seed`. Replace the `ari_by_mode` block with:

```python
    scores_by_mode = labels_by_mode(carve, y, candidate_k=candidate_k)
```

and the lookup `aris = ari_by_mode[_labels_mode(metric_name)]` / `"ari_at_k": aris[k]` with `scores = scores_by_mode[_labels_mode(metric_name)]` / `"ari_at_k": scores[k]["ari"]`. Wrap the two timing fits in the cap: change `timer.fit(X, mode=mode)` to

```python
                with cpu_cap(thread_cap):
                    t0 = time.perf_counter()
                    timer.fit(X, mode=mode)
                    elapsed = time.perf_counter() - t0
```

(moving the existing `t0 = ...` and `elapsed = ...` lines inside the block).

In `run_scenario`, after `started = time.perf_counter()` add

```python
    _, thread_cap = thread_cap_for(n_jobs)
```

and pass `thread_cap=thread_cap` to `run_cell` inside `_one`. Update the module docstring's second paragraph to end with: `Each fit is also capped at its share of the machine's threads (see cpu_cap), which changes nothing at n_jobs=1.`

- [ ] Step 5: Run the tests

Run: `.venv/bin/pytest tests/benchmarks/test_run.py tests/benchmarks/test_cli.py -q`
Expected: all pass, including the existing `TestRunCell` and `TestRunScenario` classes.

- [ ] Step 6: Compare `run_cell` output against the recording from Step 1

```bash
.venv/bin/python - <<'PY'
import json, dataclasses
from benchmarks._registry import SCENARIOS
from benchmarks._run import run_cell
scenario = dataclasses.replace(SCENARIOS["gaussians"], shared={**SCENARIOS["gaussians"].shared, "n_total": 300})
rows, _ = run_cell(scenario, axis_idx=1, axis_value=1, axis_label="medium", seed=0,
                   run_id="probe", random_state=42, n_resamples=20)
after = [(r["metric_name"], r["k"], r["metric_value"], r["is_selected"], r["ari_at_k"]) for r in rows]
before = [tuple(x) for x in json.load(open("/tmp/run_cell_before.json"))]
assert after == before, "run_cell output moved"
print("identical")
PY
```

Expected: `identical`. If it prints anything else, the refactor changed a computation; find the difference before continuing.

- [ ] Step 7: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_run.py tests/benchmarks/test_run.py
git commit -m "refactor(benchmarks): share the CARVE fit and label scoring; cap forest threads per worker

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Ablation artifacts

Files:
- Modify: `src/benchmarks/_artifacts.py` (schemas after `RUNTIME_SCHEMA`; `_canonical_config`; new functions after `read_runtimes`; `build_manifest`; `write_manifest`)
- Test: `tests/benchmarks/test_artifacts.py`

Interfaces:
- Produces:
  - `CELL_KEY: tuple[str, ...]`
  - `ABLATION_CURVE_SCHEMA`, `ABLATION_SELECTION_SCHEMA`, `ABLATION_AT_K_SCHEMA`, `ABLATION_CELL_SCHEMA`, `ABLATION_DATASET_SCHEMA`, `ABLATION_SIMILARITY_SCHEMA`, and `ABLATION_SCHEMAS: dict[str, tuple[str, ...]]` keyed `curves, selection, at_k, cells, datasets, similarity`
  - `scenario_identity(scenario) -> dict`
  - `provenance() -> dict` (git_sha, package_versions, platform, peak_rss_bytes, peak_rss_unit)
  - `ablation_dir(root, name, scale, cfg_hash) -> Path`
  - `write_frame(path, rows, schema) -> Path`
  - `read_frames(rd) -> dict[str, DataFrame]` (one frame per schema name; empty frames carry the schema columns)
  - `write_manifest(rd, manifest)` now accepts a `Manifest` or a mapping

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_artifacts.py` (extend the import with `ABLATION_SCHEMAS, CELL_KEY, ablation_dir, provenance, read_frames, scenario_identity, write_frame`):

```python
class TestAblationSchemas:
    def test_every_per_cell_schema_starts_with_the_cell_key(self):
        for name in ("curves", "selection", "at_k", "cells"):
            assert ABLATION_SCHEMAS[name][: len(CELL_KEY)] == CELL_KEY

    def test_schemas_are_the_documented_contract(self):
        assert ABLATION_SCHEMAS["curves"] == CELL_KEY + (
            "metric_name", "estimator", "k", "metric_value", "metric_se",
        )
        assert ABLATION_SCHEMAS["selection"] == CELL_KEY + (
            "metric_name", "selected_estimator", "selected_k", "k_star", "ari_selected",
        )
        assert ABLATION_SCHEMAS["at_k"] == CELL_KEY + ("mode", "k", "ari_at_k", "rare_recall_at_k")
        assert ABLATION_SCHEMAS["cells"] == CELL_KEY + (
            "carve_random_state", "n_samples", "fit_seconds",
            "consensus_nan_fraction", "n_cluster_count_warnings",
        )
        assert ABLATION_SCHEMAS["datasets"] == (
            "study", "difficulty", "dataset", "n_samples", "k_star",
            "oracle_ari", "rare_label", "rare_fraction",
        )
        assert ABLATION_SCHEMAS["similarity"] == (
            "study", "difficulty", "dataset", "subsample_ratio", "estimator", "k", "draw", "ari",
        )


class TestAblationFrames:
    def _row(self, **overrides):
        row = {
            "study": "gaussians", "difficulty": "medium", "dataset": 0,
            "subsample_ratio": 0.618, "n_resamples": 100, "replicate": 0,
            "carve_random_state": 10042, "n_samples": 1500, "fit_seconds": 1.0,
            "consensus_nan_fraction": 0.0, "n_cluster_count_warnings": 0,
        }
        row.update(overrides)
        return row

    def test_write_frame_validates_and_orders_columns(self, tmp_path):
        path = write_frame(tmp_path / "cells__x.parquet", [self._row()], ABLATION_SCHEMAS["cells"])
        frame = pd.read_parquet(path)
        assert list(frame.columns) == list(ABLATION_SCHEMAS["cells"])

    def test_write_frame_accepts_no_rows(self, tmp_path):
        path = write_frame(tmp_path / "at_k__x.parquet", [], ABLATION_SCHEMAS["at_k"])
        frame = pd.read_parquet(path)
        assert frame.empty
        assert list(frame.columns) == list(ABLATION_SCHEMAS["at_k"])

    def test_write_frame_rejects_a_row_outside_the_schema(self, tmp_path):
        with pytest.raises(ValueError, match="outside the schema"):
            write_frame(tmp_path / "cells__x.parquet", [self._row(extra=1)], ABLATION_SCHEMAS["cells"])

    def test_read_frames_returns_every_schema_even_when_empty(self, tmp_path):
        frames = read_frames(tmp_path)
        assert set(frames) == set(ABLATION_SCHEMAS)
        for name, frame in frames.items():
            assert list(frame.columns) == list(ABLATION_SCHEMAS[name])
            assert frame.empty

    def test_read_frames_concatenates_by_prefix(self, tmp_path):
        write_frame(tmp_path / "cells__a.parquet", [self._row(dataset=0)], ABLATION_SCHEMAS["cells"])
        write_frame(tmp_path / "cells__b.parquet", [self._row(dataset=1)], ABLATION_SCHEMAS["cells"])
        frames = read_frames(tmp_path)
        assert sorted(frames["cells"]["dataset"]) == [0, 1]
        assert frames["curves"].empty

    def test_ablation_dir_nests_name_scale_and_hash(self, tmp_path):
        rd = ablation_dir(tmp_path, "rho_b", "dev", "abc123")
        assert rd == tmp_path / "ablation_rho_b" / "dev" / "abc123"
        assert rd.is_dir()


class TestProvenance:
    def test_carries_the_manifest_provenance_fields(self):
        record = provenance()
        assert set(record) == {
            "git_sha", "package_versions", "platform", "peak_rss_bytes", "peak_rss_unit",
        }
        assert record["peak_rss_unit"] == "bytes"

    def test_scenario_identity_is_the_config_hash_input_minus_run_parameters(self):
        scenario = SCENARIOS["gaussians"]
        identity = scenario_identity(scenario)
        assert set(identity) == {
            "name", "axis_name", "axis_values", "axis_labels", "anchors",
            "shared", "estimator", "k_star", "candidate_k", "n_trees",
        }

    def test_write_manifest_accepts_a_mapping(self, tmp_path):
        path = write_manifest(tmp_path, {"run_id": "r1", "ablation": "rho_b"})
        assert json.loads(path.read_text()) == {"run_id": "r1", "ablation": "rho_b"}
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/test_artifacts.py -q`
Expected: ImportError on `ABLATION_SCHEMAS`.

- [ ] Step 3: Implement

In `src/benchmarks/_artifacts.py`, add `from collections.abc import Mapping` to the imports, and after `RUNTIME_SCHEMA`:

```python
# --- Ablation frames ---------------------------------------------------------
# A cell is one CARVE fit. Every per-cell frame starts with this key so the
# frames join on it; the study (Klein) has difficulty "" and dataset 0.
CELL_KEY: tuple[str, ...] = (
    "study",
    "difficulty",
    "dataset",
    "subsample_ratio",
    "n_resamples",
    "replicate",
)

ABLATION_CURVE_SCHEMA: tuple[str, ...] = CELL_KEY + (
    "metric_name",
    "estimator",
    "k",
    "metric_value",
    # The <measure>_se column where CARVE provides one, NaN otherwise.
    "metric_se",
)

ABLATION_SELECTION_SCHEMA: tuple[str, ...] = CELL_KEY + (
    "metric_name",
    "selected_estimator",
    "selected_k",
    # NaN for the study, which has no true k.
    "k_star",
    "ari_selected",
)

# Simulations only: labels at every candidate k, per consensus mode.
ABLATION_AT_K_SCHEMA: tuple[str, ...] = CELL_KEY + (
    "mode",
    "k",
    "ari_at_k",
    "rare_recall_at_k",
)

ABLATION_CELL_SCHEMA: tuple[str, ...] = CELL_KEY + (
    "carve_random_state",
    "n_samples",
    "fit_seconds",
    "consensus_nan_fraction",
    "n_cluster_count_warnings",
)

# One row per simulated dataset; NaN in every numeric column for the study.
ABLATION_DATASET_SCHEMA: tuple[str, ...] = (
    "study",
    "difficulty",
    "dataset",
    "n_samples",
    "k_star",
    "oracle_ari",
    "rare_label",
    "rare_fraction",
)

# subsample_ratio 1.0 marks the refit reference: the full data clustered
# again with the draw's seed and scored against the base full-data fit.
ABLATION_SIMILARITY_SCHEMA: tuple[str, ...] = (
    "study",
    "difficulty",
    "dataset",
    "subsample_ratio",
    "estimator",
    "k",
    "draw",
    "ari",
)

ABLATION_SCHEMAS: dict[str, tuple[str, ...]] = {
    "curves": ABLATION_CURVE_SCHEMA,
    "selection": ABLATION_SELECTION_SCHEMA,
    "at_k": ABLATION_AT_K_SCHEMA,
    "cells": ABLATION_CELL_SCHEMA,
    "datasets": ABLATION_DATASET_SCHEMA,
    "similarity": ABLATION_SIMILARITY_SCHEMA,
}
```

Split `_canonical_config`:

```python
def scenario_identity(scenario: Scenario) -> dict[str, Any]:
    """What defines a scenario, independent of how many cells a run draws."""
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
        "n_trees": scenario.n_trees,
    }


def _canonical_config(
    scenario: Scenario, *, n_seeds: int, n_resamples: int, random_state: int
) -> dict[str, Any]:
    """The subset of a scenario that changing must invalidate a run."""
    return {
        **scenario_identity(scenario),
        "n_seeds": n_seeds,
        "n_resamples": n_resamples,
        "random_state": random_state,
    }
```

(The keys and values are the same as before, so `config_hash` is unchanged; `TestConfigHash` confirms it.)

After `read_runtimes`:

```python
def ablation_dir(root: Path, name: str, scale: str, cfg_hash: str) -> Path:
    """Create and return results/runs/ablation_<name>/<scale>/<hash>/.

    The scale is in the path as well as in the hash, so a publication read
    cannot pick up a development-scale run by accident.
    """
    path = Path(root) / f"ablation_{name}" / scale / cfg_hash
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_frame(path: Path, rows: list[dict[str, Any]], schema: tuple[str, ...]) -> Path:
    """Write rows as one parquet file after validating them against a schema.

    An empty row list writes an empty frame with the schema's columns; the
    study's at_k checkpoint is empty by design.
    """
    path = Path(path)
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(schema))
    _validate_against(frame, schema).to_parquet(path, index=False)
    return path


def read_frames(rd: Path) -> dict[str, pd.DataFrame]:
    """Concatenate every ablation checkpoint in a run directory, per schema.

    Files are named <schema>__<unit>.parquet. A schema with no file yet
    comes back as an empty frame with its columns, so a partial run reads.
    """
    frames: dict[str, pd.DataFrame] = {}
    for name, schema in ABLATION_SCHEMAS.items():
        paths = sorted(Path(rd).glob(f"{name}__*.parquet"))
        if not paths:
            frames[name] = pd.DataFrame(columns=list(schema))
            continue
        frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        frames[name] = frame[list(schema)]
    return frames


def provenance() -> dict[str, Any]:
    """The machine-and-code record every manifest carries."""
    return {
        "git_sha": _git_sha(),
        "package_versions": _package_versions(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cores": os.cpu_count(),
        },
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_rss_unit": "bytes",
    }
```

In `build_manifest`, replace the `peak_rss_bytes=...` through `platform={...}` arguments with `**provenance(),` so the manifest's content is unchanged. In `write_manifest`, change the signature to `def write_manifest(rd: Path, manifest: Manifest | Mapping[str, Any]) -> Path:` and the dump line to

```python
            payload = manifest.to_dict() if isinstance(manifest, Manifest) else dict(manifest)
            f.write(json.dumps(payload, indent=2, default=str))
```

with the docstring gaining `Accepts a Manifest or any mapping, so the ablation writes the same atomic file.`

- [ ] Step 4: Run the tests

Run: `.venv/bin/pytest tests/benchmarks/test_artifacts.py tests/benchmarks/test_run.py -q`
Expected: all pass. `TestConfigHash` passing is the check that the split did not move the hash.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_artifacts.py tests/benchmarks/test_artifacts.py
git commit -m "feat(benchmarks): ablation frame schemas, run directory and shared provenance

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Cells, units, seeds and arm views (`_ablation_cells.py`)

Files:
- Create: `src/benchmarks/_ablation_cells.py`
- Test: `tests/benchmarks/test_ablation_cells.py`

Interfaces:
- Consumes: `Ablation`, `AblationScale` (Task 1); `ABLATIONS`, `SCENARIOS`, `PUBLISHED_RANDOM_STATE`, `REPLICATE_SEED_SPACING`, `SIMILARITY_SEED_OFFSET` (Task 2); `benchmark_seed` is re-implemented here as `_benchmark_seed` with the same arithmetic so this module does not import `_run` (a test pins the two equal); `CELL_KEY` (Task 4).
- Produces:
  - `STUDY_DIFFICULTY = ""`, `STUDY_DATASET = 0`, `REFERENCE_RATIO = 1.0`
  - `Cell(study, difficulty, dataset, subsample_ratio, n_resamples, replicate)` NamedTuple with `.key() -> dict`
  - `Unit(kind, cell)` NamedTuple; `kind in {"dataset", "similarity", "cell"}`
  - `enumerate_cells(ablation, scale, arm=None) -> list[Cell]` (`arm` is `None`, `"rho"` or `"b"`)
  - `enumerate_units(ablation, scale) -> list[Unit]`
  - `timing_units(ablation, scale) -> list[Unit]`
  - `unit_stem(unit) -> str`, `unit_paths(rd, unit) -> dict[str, Path]`, `unit_done(rd, unit) -> bool`
  - `dataset_seed(cell, *, ablation) -> int`, `carve_seed(cell, *, ablation) -> int`, `similarity_seed(base, draw) -> int`, `seed_span(n_resamples) -> int`
  - `scenario_at_scale(scenario, scale: AblationScale) -> Scenario`
  - `cells_frame(cells) -> DataFrame`, `arm_view(frames, *, ablation, scale, arm) -> dict[str, DataFrame]`

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/test_ablation_cells.py`:

```python
"""Tests for cell enumeration, unit naming, seeds and arm views."""

import dataclasses

import pandas as pd
import pytest

from benchmarks._ablation_cells import (
    REFERENCE_RATIO,
    STUDY_DATASET,
    STUDY_DIFFICULTY,
    Cell,
    Unit,
    arm_view,
    carve_seed,
    cells_frame,
    dataset_seed,
    enumerate_cells,
    enumerate_units,
    scenario_at_scale,
    seed_span,
    similarity_seed,
    timing_units,
    unit_done,
    unit_paths,
    unit_stem,
)
from benchmarks._artifacts import ABLATION_SCHEMAS, CELL_KEY, write_frame
from benchmarks._registry import (
    ABLATIONS,
    PUBLISHED_RANDOM_STATE,
    REPLICATE_SEED_SPACING,
    SCENARIOS,
    SIMILARITY_SEED_OFFSET,
)
from benchmarks._run import benchmark_seed
from benchmarks._types import AblationScale, ArmScale

RHO_B = ABLATIONS["rho_b"]


def _tiny_ablation():
    scale = AblationScale(
        rho_arm=ArmScale(("medium",), (0, 1), 1, 2),
        b_arm=ArmScale(("medium",), (0, 1), 2, 2),
        similarity_draws=2,
        study_scale="dev",
    )
    return dataclasses.replace(
        RHO_B,
        scenarios=("gaussians",),
        rho_grid=(0.5, RHO_B.rho_default),
        b_grid=(10, RHO_B.b_default),
        scales={"dev": scale},
        default_scale="dev",
    )


class TestEnumerateCells:
    def test_counts_match_the_arithmetic(self):
        cells = enumerate_cells(_tiny_ablation(), "dev")
        # Simulations: rho arm 2 datasets x 2 rho = 4; B arm 2 datasets x 2
        # replicates x 2 B = 8; the two default cells are shared -> 10.
        # Study: rho arm 2 x 2 = 4; B arm 2 x 2 = 4; shared 2 -> 6.
        assert len(cells) == 16
        assert len(set(cells)) == 16

    def test_a_cell_in_both_arms_appears_once(self):
        ablation = _tiny_ablation()
        cells = enumerate_cells(ablation, "dev")
        shared = [
            c
            for c in cells
            if c.subsample_ratio == ablation.rho_default
            and c.n_resamples == ablation.b_default
            and c.replicate == 0
            and c.study == "gaussians"
            and c.dataset == 0
        ]
        assert len(shared) == 1

    def test_study_cells_come_first_then_simulations_grouped_by_dataset(self):
        ablation = _tiny_ablation()
        cells = enumerate_cells(ablation, "dev")
        studies = [c.study for c in cells]
        first_sim = studies.index("gaussians")
        assert set(studies[:first_sim]) == {"klein"}
        datasets = [c.dataset for c in cells[first_sim:]]
        assert datasets == sorted(datasets)

    def test_study_cells_use_the_study_key(self):
        cells = enumerate_cells(_tiny_ablation(), "dev", arm="rho")
        study = [c for c in cells if c.study == "klein"]
        assert {(c.difficulty, c.dataset) for c in study} == {(STUDY_DIFFICULTY, STUDY_DATASET)}

    def test_arm_views_are_subsets_of_the_union(self):
        ablation = _tiny_ablation()
        union = set(enumerate_cells(ablation, "dev"))
        rho = set(enumerate_cells(ablation, "dev", arm="rho"))
        b = set(enumerate_cells(ablation, "dev", arm="b"))
        assert rho | b == union
        assert all(c.n_resamples == ablation.b_default for c in rho)
        assert all(c.subsample_ratio == ablation.rho_default for c in b)

    def test_publication_counts_match_the_spec(self):
        cells = enumerate_cells(RHO_B, "publication")
        sims = [c for c in cells if c.study != "klein"]
        study = [c for c in cells if c.study == "klein"]
        assert len(sims) == 1440 + 900 - 60
        assert len(study) == 80 + 50 - 10

    def test_rejects_an_unknown_arm(self):
        with pytest.raises(ValueError, match="arm"):
            enumerate_cells(_tiny_ablation(), "dev", arm="gamma")


class TestEnumerateUnits:
    def test_each_dataset_gets_a_dataset_unit_similarity_units_and_its_cells(self):
        ablation = _tiny_ablation()
        units = enumerate_units(ablation, "dev")
        kinds = [u.kind for u in units]
        assert kinds.count("dataset") == 3  # klein, gaussians 0, gaussians 1
        # rho grid plus the refit reference, per dataset
        assert kinds.count("similarity") == 3 * (len(ablation.rho_grid) + 1)
        assert kinds.count("cell") == 16

    def test_similarity_units_include_the_reference_ratio(self):
        units = enumerate_units(_tiny_ablation(), "dev")
        ratios = {u.cell.subsample_ratio for u in units if u.kind == "similarity"}
        assert REFERENCE_RATIO in ratios

    def test_units_are_unique(self):
        units = enumerate_units(_tiny_ablation(), "dev")
        assert len(units) == len(set(units))


class TestTimingUnits:
    def test_is_twenty_two_cells_including_klein(self):
        units = timing_units(RHO_B, "publication")
        assert len(units) == 22
        assert {u.kind for u in units} == {"cell"}
        assert sum(u.cell.study == "klein" for u in units) == 4
        assert all(u.cell.n_resamples == RHO_B.b_default for u in units)


class TestUnitFiles:
    def test_stem_is_deterministic_and_readable(self):
        unit = Unit("cell", Cell("gaussians", "medium", 3, 0.618, 100, 2))
        assert unit_stem(unit) == "gaussians__medium__0003__rho0.618__b0100__rep02"

    def test_study_stem_marks_the_empty_difficulty(self):
        unit = Unit("cell", Cell("klein", STUDY_DIFFICULTY, STUDY_DATASET, 0.2, 100, 0))
        assert unit_stem(unit).startswith("klein__na__0000__")

    def test_cell_units_write_four_frames(self, tmp_path):
        unit = Unit("cell", Cell("gaussians", "medium", 0, 0.618, 100, 0))
        paths = unit_paths(tmp_path, unit)
        assert set(paths) == {"curves", "selection", "at_k", "cells"}
        assert all(p.parent == tmp_path for p in paths.values())

    def test_done_only_when_every_frame_exists(self, tmp_path):
        unit = Unit("cell", Cell("gaussians", "medium", 0, 0.618, 100, 0))
        assert not unit_done(tmp_path, unit)
        for name, path in unit_paths(tmp_path, unit).items():
            write_frame(path, [], ABLATION_SCHEMAS[name])
        assert unit_done(tmp_path, unit)

    def test_dataset_and_similarity_units_write_one_frame(self, tmp_path):
        d = Unit("dataset", Cell("gaussians", "medium", 0, 0.0, 0, 0))
        s = Unit("similarity", Cell("gaussians", "medium", 0, 0.5, 0, 0))
        assert set(unit_paths(tmp_path, d)) == {"datasets"}
        assert set(unit_paths(tmp_path, s)) == {"similarity"}


class TestSeeds:
    def test_simulation_base_is_the_benchmark_seed(self):
        cell = Cell("gaussians", "hard", 7, 0.618, 100, 0)
        assert dataset_seed(cell, ablation=RHO_B) == benchmark_seed(7, 2, PUBLISHED_RANDOM_STATE)

    def test_study_base_is_the_published_random_state(self):
        cell = Cell("klein", STUDY_DIFFICULTY, STUDY_DATASET, 0.618, 100, 0)
        assert dataset_seed(cell, ablation=RHO_B) == PUBLISHED_RANDOM_STATE

    def test_replicate_zero_reproduces_the_published_fit_seed(self):
        cell = Cell("gaussians", "medium", 0, 0.618, 100, 0)
        assert carve_seed(cell, ablation=RHO_B) == benchmark_seed(0, 1, PUBLISHED_RANDOM_STATE)

    def test_replicates_are_spaced(self):
        c0 = Cell("gaussians", "medium", 0, 0.618, 100, 0)
        c1 = Cell("gaussians", "medium", 0, 0.618, 100, 1)
        assert carve_seed(c1, ablation=RHO_B) - carve_seed(c0, ablation=RHO_B) == REPLICATE_SEED_SPACING

    def test_similarity_seed_offsets_the_base(self):
        assert similarity_seed(42, 3) == 42 + SIMILARITY_SEED_OFFSET + 3

    def test_seed_span_is_three_times_b(self):
        assert seed_span(200) == 600

    def test_publication_seed_windows_of_different_replicates_never_meet(self):
        # A sweep over every publication cell: sort the windows and check
        # that any two that overlap belong to the same replicate index.
        # Setting REPLICATE_SEED_SPACING to 100 makes this fail.
        windows = []
        for cell in enumerate_cells(RHO_B, "publication"):
            start = carve_seed(cell, ablation=RHO_B)
            windows.append((start, start + seed_span(cell.n_resamples), cell.replicate))
        windows.sort()
        reach = -1
        reach_rep = None
        for start, end, rep in windows:
            if start < reach and rep != reach_rep:
                pytest.fail(f"windows of replicates {reach_rep} and {rep} overlap at seed {start}")
            if end > reach:
                reach, reach_rep = end, rep

    def test_publication_similarity_seeds_avoid_every_cell_window(self):
        cells = enumerate_cells(RHO_B, "publication")
        windows = sorted(
            (carve_seed(c, ablation=RHO_B), carve_seed(c, ablation=RHO_B) + seed_span(c.n_resamples))
            for c in cells
        )
        draws = RHO_B.scales["publication"].similarity_draws
        bases = {dataset_seed(c, ablation=RHO_B) for c in cells}
        for base in bases:
            for draw in range(draws):
                seed = similarity_seed(base, draw)
                assert not any(start <= seed < end for start, end in windows), seed

    def test_spacing_of_one_hundred_would_fail(self, monkeypatch):
        import benchmarks._ablation_cells as cells_module

        monkeypatch.setattr(cells_module, "REPLICATE_SEED_SPACING", 100)
        c0 = Cell("gaussians", "medium", 0, 0.618, 200, 0)
        c1 = Cell("gaussians", "medium", 0, 0.618, 200, 1)
        assert carve_seed(c1, ablation=RHO_B) < carve_seed(c0, ablation=RHO_B) + seed_span(200)


class TestScenarioAtScale:
    def test_overrides_n_total_when_set(self):
        scale = RHO_B.scales["dev"]
        assert scenario_at_scale(SCENARIOS["gaussians"], scale).shared["n_total"] == 500

    def test_keeps_the_scenario_when_n_total_is_none(self):
        scale = RHO_B.scales["publication"]
        assert scenario_at_scale(SCENARIOS["gaussians"], scale) is SCENARIOS["gaussians"]


class TestArmView:
    def test_keeps_only_the_arms_cells(self):
        ablation = _tiny_ablation()
        cells = enumerate_cells(ablation, "dev")
        frame = cells_frame(cells)
        frame["fit_seconds"] = 1.0
        frames = {name: pd.DataFrame(columns=list(schema)) for name, schema in ABLATION_SCHEMAS.items()}
        frames["cells"] = frame
        rho = arm_view(frames, ablation=ablation, scale="dev", arm="rho")["cells"]
        assert set(rho["n_resamples"]) == {ablation.b_default}
        assert len(rho) == len(enumerate_cells(ablation, "dev", arm="rho"))

    def test_passes_datasets_and_similarity_through(self):
        ablation = _tiny_ablation()
        frames = {name: pd.DataFrame(columns=list(schema)) for name, schema in ABLATION_SCHEMAS.items()}
        frames["datasets"] = pd.DataFrame([{"study": "x"}])
        view = arm_view(frames, ablation=ablation, scale="dev", arm="b")
        assert view["datasets"] is frames["datasets"]

    def test_cells_frame_has_the_cell_key_columns(self):
        frame = cells_frame(enumerate_cells(_tiny_ablation(), "dev"))
        assert list(frame.columns) == list(CELL_KEY)
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/test_ablation_cells.py -q`
Expected: ModuleNotFoundError for `benchmarks._ablation_cells`.

- [ ] Step 3: Create the module

Create `src/benchmarks/_ablation_cells.py`:

```python
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
    return Cell(ablation.study, STUDY_DIFFICULTY, STUDY_DATASET, float(rho), int(b), replicate)


def enumerate_cells(ablation: Ablation, scale: str, arm: str | None = None) -> list[Cell]:
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
                                Cell(name, difficulty, int(dataset), float(rho), int(b), replicate)
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
        (c.study, c.difficulty, c.dataset) for c in enumerate_cells(ablation, scale, arm="rho")
    }
    datasets = list(dict.fromkeys((c.study, c.difficulty, c.dataset) for c in cells))
    units: list[Unit] = []
    for study, difficulty, dataset in datasets:
        units.append(Unit("dataset", Cell(study, difficulty, dataset, 0.0, 0, 0)))
        if (study, difficulty, dataset) in rho_datasets:
            for rho in (*ablation.rho_grid, REFERENCE_RATIO):
                units.append(Unit("similarity", Cell(study, difficulty, dataset, float(rho), 0, 0)))
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
            Unit("cell", _study_cell(ablation, ablation.rho_default, ablation.b_default, replicate))
        )
    for name in ablation.scenarios:
        for rho in (ablation.rho_default, ablation.rho_grid[0], ablation.rho_grid[-1]):
            units.setdefault(
                Unit("cell", Cell(name, difficulty, int(dataset), float(rho), ablation.b_default, 0))
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
    return {name: Path(rd) / f"{name}__{stem}.parquet" for name in _UNIT_FRAMES[unit.kind]}


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
    return dataset_seed(cell, ablation=ablation) + cell.replicate * REPLICATE_SEED_SPACING


def similarity_seed(base: int, draw: int) -> int:
    return int(base + SIMILARITY_SEED_OFFSET + draw)


def seed_span(n_resamples: int) -> int:
    """How many consecutive seeds one fit uses: P_1, P_2 and the P_test pipeline."""
    return 3 * int(n_resamples)


def scenario_at_scale(scenario: Scenario, scale: AblationScale) -> Scenario:
    """The scenario with the scale's n_total override applied, if any."""
    if scale.n_total is None:
        return scenario
    return dataclasses.replace(scenario, shared={**scenario.shared, "n_total": int(scale.n_total)})


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
        view[name] = frame if frame.empty else frame.merge(keys, on=list(CELL_KEY), how="inner")
    return view
```

Note `enumerate_cells`'s `_order` looks up `SCENARIOS[cell.study]`; a test that uses a scenario not in the registry must monkeypatch `benchmarks._ablation_cells.SCENARIOS` (Task 6 does).

- [ ] Step 4: Run the tests

Run: `.venv/bin/pytest tests/benchmarks/test_ablation_cells.py -q`
Expected: all pass.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_ablation_cells.py tests/benchmarks/test_ablation_cells.py
git commit -m "feat(benchmarks): ablation cells, units, seeds and arm views

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Unit executors and the runner (`_ablation.py`)

Files:
- Create: `src/benchmarks/_ablation.py`
- Test: `tests/benchmarks/test_ablation.py`

Interfaces:
- Consumes: Task 5's names; `fit_carve`, `labels_by_mode`, `smallest_cluster`, `thread_cap_for`, `_labels_mode` (Task 3); `write_frame`, `read_frames`, `ablation_dir`, `provenance`, `write_manifest`, `scenario_identity`, `ABLATION_SCHEMAS` (Task 4); `select_best_row_by_rule` from `carve._selection`; `split_subsample_indices` from `carve._utils`.
- Produces:
  - `ablation_config(ablation, scale) -> dict`, `ablation_hash(ablation, scale) -> str`
  - `dataset_rows(unit, *, ablation, scale, data) -> list[dict]`
  - `similarity_rows(unit, *, ablation, scale, data) -> list[dict]`
  - `cell_rows(unit, *, ablation, scale, data, thread_cap) -> dict[str, list[dict]]`
  - `run_unit(unit, *, ablation, scale, data, thread_cap) -> dict[str, list[dict]]`
  - `run_ablation(ablation, *, scale=None, root, n_jobs=-1, resume=True, verbose=0, units=None) -> Path`

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/test_ablation.py`:

```python
"""Tests for the ablation runner: unit executors, hashing, resume."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

import benchmarks._ablation as ablation_module
import benchmarks._ablation_cells as cells_module
from benchmarks._ablation import (
    ablation_hash,
    cell_rows,
    dataset_rows,
    run_ablation,
    similarity_rows,
)
from benchmarks._ablation_cells import (
    REFERENCE_RATIO,
    STUDY_DATASET,
    STUDY_DIFFICULTY,
    Cell,
    Unit,
    enumerate_cells,
    enumerate_units,
    unit_paths,
)
from benchmarks._artifacts import ABLATION_SCHEMAS, CELL_KEY, read_frames
from benchmarks._registry import CARVE_METRICS_ALL
from benchmarks._types import (
    Ablation,
    AblationScale,
    ArmScale,
    Axis,
    EstimatorSpec,
    Scenario,
    Study,
)

TINY_SCENARIO = Scenario(
    name="tiny",
    axis=Axis(name="difficulty_level", values=(0, 1), labels=("easy", "hard")),
    anchors={"easy": {"cluster_scale": 0.6}, "hard": {"cluster_scale": 2.5}},
    shared={"n_total": 120, "p": 4, "distribution": "gaussian"},
    estimator=EstimatorSpec(name="kmeans"),
    candidate_k=(3, 4, 5),
    n_seeds=2,
)


def _blobs(subsample):
    X, y = make_blobs(n_samples=90, centers=3, n_features=4, random_state=0)
    return X, y, {}


PROBE_STUDY = Study(
    name="probe",
    loader=_blobs,
    estimator=EstimatorSpec(name="agglomerative"),
    candidate_k=(2, 3, 4),
    scales={"dev": None},
    default_scale="dev",
    partners=(),
    not_two=True,
)

TINY_SCALE = AblationScale(
    rho_arm=ArmScale(("easy",), (0, 1), 1, 2),
    b_arm=ArmScale(("easy",), (0, 1), 2, 2),
    similarity_draws=2,
    study_scale="dev",
)

TINY_ABLATION = Ablation(
    name="tiny",
    scenarios=("tiny",),
    study="probe",
    rho_grid=(0.5, 0.618),
    b_grid=(8, 12),
    rho_default=0.618,
    b_default=12,
    scales={"dev": TINY_SCALE, "big": dataclasses.replace(TINY_SCALE, similarity_draws=3)},
    default_scale="dev",
)


@pytest.fixture(scope="module", autouse=True)
def tiny_registry():
    """Point both ablation modules at the tiny scenario and study."""
    mp = pytest.MonkeyPatch()
    mp.setattr(cells_module, "SCENARIOS", {"tiny": TINY_SCENARIO})
    mp.setattr(ablation_module, "SCENARIOS", {"tiny": TINY_SCENARIO})
    mp.setattr(ablation_module, "STUDIES", {"probe": PROBE_STUDY})
    yield
    mp.undo()


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    """A finished dev-scale run of the tiny ablation, at n_jobs=1 so unit
    functions run in this process."""
    root = tmp_path_factory.mktemp("ablation")
    rd = run_ablation(TINY_ABLATION, scale="dev", root=root, n_jobs=1)
    return root, rd


class TestDatasetRows:
    def test_simulated_dataset_records_truth_and_oracle(self):
        unit = Unit("dataset", Cell("tiny", "easy", 0, 0.0, 0, 0))
        rows = dataset_rows(unit, ablation=TINY_ABLATION, scale="dev", data=None)
        assert len(rows) == 1
        row = rows[0]
        assert set(row) == set(ABLATION_SCHEMAS["datasets"])
        assert row["k_star"] == 5
        assert row["n_samples"] == 120
        assert 0.0 < row["rare_fraction"] <= 0.2
        assert -1.0 <= row["oracle_ari"] <= 1.0

    def test_study_dataset_has_nan_truth(self):
        X, y, _ = _blobs(None)
        unit = Unit("dataset", Cell("probe", STUDY_DIFFICULTY, STUDY_DATASET, 0.0, 0, 0))
        rows = dataset_rows(unit, ablation=TINY_ABLATION, scale="dev", data=(X, y))
        assert rows[0]["n_samples"] == 90
        assert np.isnan(rows[0]["k_star"]) and np.isnan(rows[0]["rare_label"])


class TestSimilarityRows:
    def test_wards_refit_reference_is_one(self):
        X, y, _ = _blobs(None)
        unit = Unit("similarity", Cell("probe", STUDY_DIFFICULTY, STUDY_DATASET, REFERENCE_RATIO, 0, 0))
        rows = similarity_rows(unit, ablation=TINY_ABLATION, scale="dev", data=(X, y))
        assert rows and all(r["ari"] == 1.0 for r in rows)
        assert {r["estimator"] for r in rows} == {"AgglomerativeClustering"}
        assert {r["k"] for r in rows} == {2, 3, 4}
        assert {r["draw"] for r in rows} == {0, 1}

    def test_subsample_rows_score_the_draw_against_the_full_fit(self):
        unit = Unit("similarity", Cell("tiny", "easy", 0, 0.5, 0, 0))
        rows = similarity_rows(unit, ablation=TINY_ABLATION, scale="dev", data=None)
        assert len(rows) == 3 * 2  # candidate k x draws
        assert all(-1.0 <= r["ari"] <= 1.0 for r in rows)
        assert {r["subsample_ratio"] for r in rows} == {0.5}

    def test_a_draw_has_the_subsample_size_carve_uses(self):
        from carve._utils import split_subsample_indices

        idx, rest = split_subsample_indices(90, subsample_ratio=0.5, random_state=1)
        assert idx.size == int(0.5 * 90)
        assert idx.size + rest.size == 90


class TestCellRows:
    @pytest.fixture(scope="class")
    def sim_cell(self):
        unit = Unit("cell", Cell("tiny", "easy", 0, 0.5, 8, 1))
        return cell_rows(unit, ablation=TINY_ABLATION, scale="dev", data=None, thread_cap=None)

    def test_every_frame_carries_its_schema(self, sim_cell):
        for name in ("curves", "selection", "at_k", "cells"):
            assert sim_cell[name], name
            assert set(sim_cell[name][0]) == set(ABLATION_SCHEMAS[name]), name

    def test_curves_cover_every_metric_and_k(self, sim_cell):
        curves = pd.DataFrame(sim_cell["curves"])
        assert set(curves["metric_name"]) == set(CARVE_METRICS_ALL)
        assert set(curves["k"]) == {3, 4, 5}
        assert curves["metric_se"].notna().sum() > 0
        assert curves.loc[curves["metric_name"] == "consensus_pac_stability", "metric_se"].isna().all()

    def test_one_selection_per_metric_with_a_true_k(self, sim_cell):
        selection = pd.DataFrame(sim_cell["selection"])
        assert len(selection) == len(CARVE_METRICS_ALL)
        assert set(selection["k_star"]) == {5.0}
        assert set(selection["selected_k"]) <= {3, 4, 5}

    def test_at_k_rows_cover_both_modes(self, sim_cell):
        at_k = pd.DataFrame(sim_cell["at_k"])
        assert set(at_k["mode"]) == {"default", "generalizability"}
        assert len(at_k) == 2 * 3
        assert at_k["rare_recall_at_k"].between(0, 1).all()

    def test_cell_row_records_the_replicate_seed(self, sim_cell):
        row = sim_cell["cells"][0]
        assert row["replicate"] == 1
        assert row["carve_random_state"] == cells_module.carve_seed(
            Cell("tiny", "easy", 0, 0.5, 8, 1), ablation=TINY_ABLATION
        )
        assert row["n_samples"] == 120
        assert 0.0 <= row["consensus_nan_fraction"] <= 1.0

    def test_study_cell_has_no_at_k_rows_and_honors_not_two(self):
        X, y, _ = _blobs(None)
        unit = Unit("cell", Cell("probe", STUDY_DIFFICULTY, STUDY_DATASET, 0.618, 8, 0))
        rows = cell_rows(unit, ablation=TINY_ABLATION, scale="dev", data=(X, y), thread_cap=None)
        assert rows["at_k"] == []
        selection = pd.DataFrame(rows["selection"])
        assert selection["k_star"].isna().all()
        assert (selection["selected_k"] != 2).all()
        assert selection["ari_selected"].between(-1, 1).all()


class TestAblationHash:
    def test_is_stable(self):
        assert ablation_hash(TINY_ABLATION, "dev") == ablation_hash(TINY_ABLATION, "dev")

    def test_differs_between_scales(self):
        assert ablation_hash(TINY_ABLATION, "dev") != ablation_hash(TINY_ABLATION, "big")

    def test_changes_with_the_grid(self):
        other = dataclasses.replace(TINY_ABLATION, rho_grid=(0.4, 0.618))
        assert ablation_hash(other, "dev") != ablation_hash(TINY_ABLATION, "dev")

    def test_changes_with_the_studys_selection_setting(self):
        mp = pytest.MonkeyPatch()
        mp.setattr(
            ablation_module,
            "STUDIES",
            {"probe": dataclasses.replace(PROBE_STUDY, not_two=False)},
        )
        try:
            changed = ablation_hash(TINY_ABLATION, "dev")
        finally:
            mp.undo()
        assert changed != ablation_hash(TINY_ABLATION, "dev")

    def test_is_twelve_hex_characters(self):
        assert len(ablation_hash(TINY_ABLATION, "dev")) == 12


class TestRunAblation:
    def test_writes_every_unit_and_a_manifest(self, completed):
        _, rd = completed
        units = enumerate_units(TINY_ABLATION, "dev")
        assert all(all(p.exists() for p in unit_paths(rd, u).values()) for u in units)
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["ablation"] == "tiny"
        assert manifest["scale"] == "dev"
        assert manifest["workers"] == 1
        assert manifest["n_units"] == len(units)
        assert "git_sha" in manifest and "thread_cap" in manifest

    def test_run_directory_nests_scale_and_hash(self, completed):
        root, rd = completed
        assert rd == root / "ablation_tiny" / "dev" / ablation_hash(TINY_ABLATION, "dev")

    def test_frames_read_back_with_one_row_per_cell(self, completed):
        _, rd = completed
        frames = read_frames(rd)
        cells = enumerate_cells(TINY_ABLATION, "dev")
        assert len(frames["cells"]) == len(cells) == 16
        assert frames["cells"][list(CELL_KEY)].drop_duplicates().shape[0] == 16
        assert len(frames["selection"]) == 16 * len(CARVE_METRICS_ALL)
        assert len(frames["datasets"]) == 3
        # Simulated cells only: 10 cells x 2 modes x 3 k
        assert len(frames["at_k"]) == 10 * 2 * 3
        # 3 datasets x (2 rho + reference) x k values x 2 draws
        assert len(frames["similarity"]) == 2 * 3 * 3 * 2 + 1 * 3 * 3 * 2

    def test_the_shared_default_cell_is_fitted_once(self, completed):
        _, rd = completed
        cells = read_frames(rd)["cells"]
        shared = cells[
            (cells["study"] == "tiny")
            & (cells["dataset"] == 0)
            & (cells["replicate"] == 0)
            & (cells["subsample_ratio"] == 0.618)
            & (cells["n_resamples"] == 12)
        ]
        assert len(shared) == 1

    def test_resume_recomputes_only_the_missing_unit(self, completed, tmp_path, monkeypatch):
        import shutil

        root, rd = completed
        new_root = tmp_path / "runs"
        shutil.copytree(root, new_root)
        new_rd = new_root / rd.relative_to(root)
        unit = next(u for u in enumerate_units(TINY_ABLATION, "dev") if u.kind == "cell")
        for path in unit_paths(new_rd, unit).values():
            path.unlink()

        calls = []
        original = ablation_module.run_unit

        def counting(unit, **kwargs):
            calls.append(unit)
            return original(unit, **kwargs)

        monkeypatch.setattr(ablation_module, "run_unit", counting)
        run_ablation(TINY_ABLATION, scale="dev", root=new_root, n_jobs=1)
        assert calls == [unit]

    def test_resume_keeps_the_run_id_and_accumulates_wall_clock(self, completed, tmp_path):
        import shutil

        root, rd = completed
        new_root = tmp_path / "runs"
        shutil.copytree(root, new_root)
        new_rd = new_root / rd.relative_to(root)
        before = json.loads((new_rd / "manifest.json").read_text())
        run_ablation(TINY_ABLATION, scale="dev", root=new_root, n_jobs=1)
        after = json.loads((new_rd / "manifest.json").read_text())
        assert after["run_id"] == before["run_id"]
        assert after["wall_clock_s"] >= before["wall_clock_s"]

    def test_no_resume_mints_a_new_run_id(self, completed, tmp_path):
        import shutil

        root, rd = completed
        new_root = tmp_path / "runs"
        shutil.copytree(root, new_root)
        new_rd = new_root / rd.relative_to(root)
        before = json.loads((new_rd / "manifest.json").read_text())
        run_ablation(TINY_ABLATION, scale="dev", root=new_root, n_jobs=1, resume=False)
        after = json.loads((new_rd / "manifest.json").read_text())
        assert after["run_id"] != before["run_id"]

    def test_explicit_units_run_only_those(self, tmp_path):
        units = [u for u in enumerate_units(TINY_ABLATION, "dev") if u.kind == "dataset"][:1]
        rd = run_ablation(TINY_ABLATION, scale="dev", root=tmp_path, n_jobs=1, units=units)
        frames = read_frames(rd)
        assert len(frames["datasets"]) == 1
        assert frames["cells"].empty

    def test_rejects_an_unknown_scale(self, tmp_path):
        with pytest.raises(ValueError, match="scale"):
            run_ablation(TINY_ABLATION, scale="huge", root=tmp_path, n_jobs=1)

    def test_replicates_differ_and_default_cells_agree_across_arms(self, completed):
        # Two replicates of one setting differ (different seeds); the cell
        # shared by both arms has one row, checked above, so nothing to
        # compare there. Curves of replicate 0 and 1 at (0.618, 12) differ.
        _, rd = completed
        curves = read_frames(rd)["curves"]
        rows = curves[
            (curves["study"] == "tiny")
            & (curves["dataset"] == 0)
            & (curves["subsample_ratio"] == 0.618)
            & (curves["n_resamples"] == 12)
            & (curves["metric_name"] == "ari_stability")
        ]
        by_rep = rows.pivot(index="k", columns="replicate", values="metric_value")
        assert not np.allclose(by_rep[0], by_rep[1])
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/test_ablation.py -q`
Expected: ModuleNotFoundError for `benchmarks._ablation`.

- [ ] Step 3: Create the module

Create `src/benchmarks/_ablation.py`:

```python
"""The rho/B ablation runner: configuration hash, unit executors, run loop.

Each unit runs in a loky worker and returns rows for the frames it owns;
the parent writes them. Every CARVE fit runs on one worker with its forest
capped at the worker's share of the cores (see _run.cpu_cap). Nothing here
saves a fitted CARVE: 120 Klein fits of n-by-n consensus matrices would
cost far more disk than the rows the analysis reads, and the fit cache's
filename does not distinguish rho or the seed.
"""

import dataclasses
import hashlib
import json
import time
import uuid
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

from carve import CARVE
from carve._selection import select_best_row_by_rule
from carve._utils import split_subsample_indices

from ._ablation_cells import (
    REFERENCE_RATIO,
    Cell,
    Unit,
    carve_seed,
    dataset_seed,
    enumerate_units,
    scenario_at_scale,
    similarity_seed,
    unit_done,
    unit_paths,
)
from ._artifacts import (
    ABLATION_SCHEMAS,
    ablation_dir,
    provenance,
    scenario_identity,
    write_frame,
    write_manifest,
)
from ._estimators import ESTIMATOR_CLASSES, build_estimator, param_grids
from ._registry import (
    ACTIVE_ANCHOR_SET_NAME,
    CARVE_METRICS_ALL,
    PUBLISHED_RANDOM_STATE,
    REPLICATE_SEED_SPACING,
    SCENARIOS,
    SIMILARITY_SEED_OFFSET,
    metric_measure,
    metric_rule,
)
from ._run import (
    _labels_mode,
    fit_carve,
    labels_by_mode,
    smallest_cluster,
    thread_cap_for,
)
from ._simulate import simulate
from ._studies import STUDIES, load_study, resolve_scale, study_model_grids
from ._types import Ablation, Scenario

#: The study fits with CARVE's own tree count, as fit_or_load_carve does.
CARVE_N_TREES: int = next(field.default for field in fields(CARVE) if field.name == "n_trees")


# --- Configuration ------------------------------------------------------------
def ablation_config(ablation: Ablation, scale: str) -> dict[str, Any]:
    """Everything that defines a run at one scale.

    Only the chosen scale is included, so editing the dev scale cannot
    invalidate a publication run.
    """
    sc = ablation.scales[scale]
    study = STUDIES[ablation.study]
    return {
        "ablation": {
            k: v for k, v in dataclasses.asdict(ablation).items() if k != "scales"
        },
        "scale": {scale: dataclasses.asdict(sc)},
        "scenarios": {
            name: scenario_identity(scenario_at_scale(SCENARIOS[name], sc))
            for name in ablation.scenarios
        },
        "study": {
            "name": study.name,
            "grids": repr(study_model_grids(study)),
            "scale": sc.study_scale,
            "resolved": resolve_scale(study, sc.study_scale),
            "not_two": study.not_two,
        },
        "random_state": PUBLISHED_RANDOM_STATE,
        "replicate_seed_spacing": REPLICATE_SEED_SPACING,
        "similarity_seed_offset": SIMILARITY_SEED_OFFSET,
        "anchor_set": ACTIVE_ANCHOR_SET_NAME,
    }


def ablation_hash(ablation: Ablation, scale: str) -> str:
    payload = json.dumps(ablation_config(ablation, scale), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# --- Data ---------------------------------------------------------------------
def _simulated(cell: Cell, ablation: Ablation, scale: str):
    """(X, y, scenario, base seed) of a simulated cell's dataset."""
    scenario = scenario_at_scale(SCENARIOS[cell.study], ablation.scales[scale])
    axis_idx = scenario.axis.labels.index(cell.difficulty)
    base = dataset_seed(cell, ablation=ablation)
    X, y = simulate(
        scenario,
        axis_value=scenario.axis.values[axis_idx],
        axis_label=cell.difficulty,
        seed=base,
    )
    return X, y, scenario, base


def _dataset_fragment(cell: Cell) -> dict[str, Any]:
    return {"study": cell.study, "difficulty": cell.difficulty, "dataset": int(cell.dataset)}


# --- Unit executors -----------------------------------------------------------
def dataset_rows(
    unit: Unit, *, ablation: Ablation, scale: str, data: tuple | None
) -> list[dict[str, Any]]:
    """One row: sample count, and for a simulation its truth and oracle."""
    cell = unit.cell
    if cell.study == ablation.study:
        X, _ = data
        return [
            {
                **_dataset_fragment(cell),
                "n_samples": int(X.shape[0]),
                "k_star": np.nan,
                "oracle_ari": np.nan,
                "rare_label": np.nan,
                "rare_fraction": np.nan,
            }
        ]
    X, y, scenario, base = _simulated(cell, ablation, scale)
    oracle = build_estimator(scenario.estimator, n_clusters=scenario.k_star, random_state=base)
    rare_label, rare_fraction = smallest_cluster(y)
    return [
        {
            **_dataset_fragment(cell),
            "n_samples": int(X.shape[0]),
            "k_star": float(scenario.k_star),
            "oracle_ari": float(adjusted_rand_score(y, oracle.fit_predict(X))),
            "rare_label": float(rare_label),
            "rare_fraction": float(rare_fraction),
        }
    ]


def similarity_rows(
    unit: Unit, *, ablation: Ablation, scale: str, data: tuple | None
) -> list[dict[str, Any]]:
    """Subsample-versus-full ARI at every candidate k for one dataset and rho.

    F is the full-data fit with the base seed (for a simulation, exactly
    run_cell's labels_by_k). Draw m subsamples with similarity_seed(base, m)
    and clusters it with the same estimator, k and seed; its ARI is taken
    against F restricted to the draw. At REFERENCE_RATIO the full data is
    refit with the draw's seed instead and scored against F.
    """
    cell = unit.cell
    sc = ablation.scales[scale]
    if cell.study == ablation.study:
        X, _ = data
        study = STUDIES[ablation.study]
        specs = (study.estimator, *study.partners)
        candidate_k = study.candidate_k
        base = dataset_seed(cell, ablation=ablation)
    else:
        X, _, scenario, base = _simulated(cell, ablation, scale)
        specs = (scenario.estimator,)
        candidate_k = scenario.candidate_k
    X = np.asarray(X)
    n = X.shape[0]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        estimator_name = ESTIMATOR_CLASSES[spec.name].__name__
        for k in candidate_k:
            full = build_estimator(spec, n_clusters=k, random_state=base).fit_predict(X)
            for draw in range(sc.similarity_draws):
                seed = similarity_seed(base, draw)
                estimator = build_estimator(spec, n_clusters=k, random_state=seed)
                if cell.subsample_ratio == REFERENCE_RATIO:
                    ari = adjusted_rand_score(full, estimator.fit_predict(X))
                else:
                    idx, _ = split_subsample_indices(
                        n, subsample_ratio=cell.subsample_ratio, random_state=seed
                    )
                    ari = adjusted_rand_score(full[idx], estimator.fit_predict(X[idx]))
                rows.append(
                    {
                        **_dataset_fragment(cell),
                        "subsample_ratio": float(cell.subsample_ratio),
                        "estimator": estimator_name,
                        "k": int(k),
                        "draw": int(draw),
                        "ari": float(ari),
                    }
                )
    return rows


def cell_rows(
    unit: Unit,
    *,
    ablation: Ablation,
    scale: str,
    data: tuple | None,
    thread_cap: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """Fit CARVE once and return the curves, selection, at_k and cells rows."""
    cell = unit.cell
    if cell.study == ablation.study:
        X, y = data
        study = STUDIES[ablation.study]
        grids = study_model_grids(study)
        candidate_k = tuple(study.candidate_k)
        not_two = study.not_two
        n_trees = CARVE_N_TREES
        k_star = None
        rare_label = None
    else:
        X, y, scenario, _ = _simulated(cell, ablation, scale)
        grids = param_grids(scenario.estimator, list(scenario.candidate_k))
        candidate_k = tuple(scenario.candidate_k)
        not_two = False
        n_trees = scenario.n_trees
        k_star = scenario.k_star
        rare_label, _ = smallest_cluster(y)
    X = np.asarray(X)
    y = np.asarray(y)
    seed = carve_seed(cell, ablation=ablation)

    fit = fit_carve(
        X,
        grids=grids,
        n_resamples=cell.n_resamples,
        n_trees=n_trees,
        random_state=seed,
        subsample_ratio=cell.subsample_ratio,
        thread_cap=thread_cap,
    )
    carve = fit.carve
    results = carve.estimator_results_
    key = cell.key()

    curves: list[dict[str, Any]] = []
    for metric_name in CARVE_METRICS_ALL:
        measure = metric_measure(metric_name)
        se_col = f"{measure}_se"
        for _, row in results.iterrows():
            curves.append(
                {
                    **key,
                    "metric_name": metric_name,
                    "estimator": str(row["estimator"]),
                    "k": int(row["n_clusters"]),
                    "metric_value": float(row[measure]),
                    "metric_se": float(row[se_col]) if se_col in results.columns else np.nan,
                }
            )

    selection: list[dict[str, Any]] = []
    for metric_name in CARVE_METRICS_ALL:
        measure = metric_measure(metric_name)
        rule = metric_rule(metric_name)
        row = select_best_row_by_rule(results, measure=measure, rule=rule, not_two=not_two)
        labels = carve.get_labels(
            measure=measure, rule=rule, not_two=not_two, mode=_labels_mode(metric_name)
        )
        selection.append(
            {
                **key,
                "metric_name": metric_name,
                "selected_estimator": str(row["estimator"]),
                "selected_k": int(row["n_clusters"]),
                "k_star": np.nan if k_star is None else float(k_star),
                "ari_selected": float(adjusted_rand_score(y, labels)),
            }
        )

    at_k: list[dict[str, Any]] = []
    if k_star is not None:
        scores = labels_by_mode(carve, y, candidate_k=candidate_k, rare_label=rare_label)
        for mode, per_k in scores.items():
            for k, entry in per_k.items():
                at_k.append(
                    {
                        **key,
                        "mode": mode,
                        "k": int(k),
                        "ari_at_k": entry["ari"],
                        "rare_recall_at_k": entry["rare_recall"],
                    }
                )

    nan_fraction = max(
        float(np.isnan(np.asarray(M, dtype=float)).mean()) for M in carve.consensus_matrices_
    )
    cells = [
        {
            **key,
            "carve_random_state": int(seed),
            "n_samples": int(X.shape[0]),
            "fit_seconds": fit.fit_seconds,
            "consensus_nan_fraction": nan_fraction,
            "n_cluster_count_warnings": int(fit.n_cluster_count_warnings),
        }
    ]
    return {"curves": curves, "selection": selection, "at_k": at_k, "cells": cells}


def run_unit(
    unit: Unit,
    *,
    ablation: Ablation,
    scale: str,
    data: tuple | None,
    thread_cap: int | None,
) -> dict[str, list[dict[str, Any]]]:
    if unit.kind == "dataset":
        return {"datasets": dataset_rows(unit, ablation=ablation, scale=scale, data=data)}
    if unit.kind == "similarity":
        return {"similarity": similarity_rows(unit, ablation=ablation, scale=scale, data=data)}
    if unit.kind == "cell":
        return cell_rows(unit, ablation=ablation, scale=scale, data=data, thread_cap=thread_cap)
    raise ValueError(f"Unknown unit kind {unit.kind!r}.")


def _run_and_write(unit, rd, ablation, scale, data, thread_cap) -> None:
    # Module level, not a closure: joblib memmaps large array arguments
    # (the study's X) once per call, while a closure would pickle them
    # into every task.
    frames = run_unit(unit, ablation=ablation, scale=scale, data=data, thread_cap=thread_cap)
    for name, path in unit_paths(rd, unit).items():
        write_frame(path, frames[name], ABLATION_SCHEMAS[name])


# --- Runner -------------------------------------------------------------------
def run_ablation(
    ablation: Ablation,
    *,
    scale: str | None = None,
    root: Path,
    n_jobs: int = -1,
    resume: bool = True,
    verbose: int = 0,
    units: list[Unit] | None = None,
) -> Path:
    """Run every unit of an ablation at a scale, checkpointing as it goes.

    Returns the run directory, results/runs/ablation_<name>/<scale>/<hash>/.
    Provenance across resumes follows run_scenario: a resumed run keeps the
    manifest's run_id and adds its wall clock; resume=False mints a new id.
    units overrides what runs (the timing batch); resume still applies.
    """
    scale = ablation.default_scale if scale is None else scale
    if scale not in ablation.scales:
        raise ValueError(
            f"Ablation {ablation.name!r} has no scale {scale!r}; declared scales "
            f"are {sorted(ablation.scales)}."
        )
    sc = ablation.scales[scale]
    cfg_hash = ablation_hash(ablation, scale)
    rd = ablation_dir(Path(root), ablation.name, scale, cfg_hash)
    workers, thread_cap = thread_cap_for(n_jobs)

    manifest_path = rd / "manifest.json"
    previous = None
    if resume and manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(
                f"Could not read manifest at {manifest_path} ({exc}); "
                "continuing this run with fresh provenance.",
                stacklevel=2,
            )

    all_units = enumerate_units(ablation, scale) if units is None else list(units)
    pending = [u for u in all_units if not (resume and unit_done(rd, u))]

    study_data = None
    if any(u.cell.study == ablation.study for u in pending):
        X, y, _ = load_study(STUDIES[ablation.study], scale=sc.study_scale)
        study_data = (np.asarray(X), np.asarray(y))

    if verbose:
        print(
            f"ablation {ablation.name} ({scale}): {len(pending)} units to run, "
            f"{len(all_units) - len(pending)} already done; {workers} workers, "
            f"{thread_cap} threads per fit."
        )

    run_id = previous["run_id"] if previous else uuid.uuid4().hex[:12]
    started = time.perf_counter()
    if pending:
        Parallel(n_jobs=n_jobs)(
            delayed(_run_and_write)(
                unit,
                rd,
                ablation,
                scale,
                study_data if unit.cell.study == ablation.study else None,
                thread_cap,
            )
            for unit in tqdm(pending, desc=f"ablation {ablation.name}", leave=False)
        )
    elapsed = time.perf_counter() - started
    if previous is not None:
        elapsed += float(previous["wall_clock_s"])

    write_manifest(
        rd,
        {
            "run_id": run_id,
            "ablation": ablation.name,
            "scale": scale,
            "config_hash": cfg_hash,
            "anchor_set": ACTIVE_ANCHOR_SET_NAME,
            "n_jobs": int(n_jobs),
            "workers": int(workers),
            "thread_cap": int(thread_cap),
            "cores": int(cpu_count()),
            "n_units": len(all_units),
            "wall_clock_s": float(elapsed),
            "config": ablation_config(ablation, scale),
            **provenance(),
        },
    )
    return rd
```

- [ ] Step 4: Run the tests

Run: `.venv/bin/pytest tests/benchmarks/test_ablation.py -q` (allow a few minutes; the module fixture fits sixteen small CARVE runs).
Expected: all pass. If `test_curves_cover_every_metric_and_k` fails because `metric_se` is NaN for every metric, check that `estimator_results_` carries `ari_stability_se` (it does, `_runner.py` "ari_stability_se") and that `metric_measure` strips the suffix. If any test raises a `UserWarning` about cluster counts, `fit_carve`'s interception is not covering the fit; do not add a `filterwarnings` mark.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_ablation.py tests/benchmarks/test_ablation.py
git commit -m "feat(benchmarks): ablation unit executors and runner

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: CLI entry

Files:
- Modify: `src/benchmarks/run.py`
- Test: `tests/benchmarks/test_cli.py`

Interfaces:
- Consumes: `ABLATIONS` (Task 2), `run_ablation` (Task 6), `timing_units` (Task 5).
- Produces: `python -m benchmarks.run --ablation NAME [--scale S] [--timing-batch] [--n-jobs N] [--no-resume] [--root R]`; `--list` prints `ablation:<name>` lines after the scenarios; `--n-jobs` defaults to `None` and resolves to 1 for scenarios and -1 for an ablation.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_cli.py` (add `import benchmarks.run as run_module` and `from benchmarks._registry import ABLATIONS`):

```python
class TestAblationCli:
    def test_listing_includes_ablations(self, capsys):
        assert main(["--list"]) == 0
        assert "ablation:rho_b" in capsys.readouterr().out

    def test_unknown_ablation_is_rejected(self, capsys):
        assert main(["--ablation", "nope"]) == 2
        assert "nope" in capsys.readouterr().err

    def test_ablation_is_exclusive_with_scenario_and_all(self, capsys):
        assert main(["--ablation", "rho_b", "--scenario", "gaussians"]) == 2
        assert main(["--ablation", "rho_b", "--all"]) == 2

    def test_n_jobs_defaults_differ_by_mode(self):
        args = _parser().parse_args(["--scenario", "gaussians"])
        assert args.n_jobs is None
        assert run_module._resolve_n_jobs(args) == 1
        args = _parser().parse_args(["--ablation", "rho_b"])
        assert run_module._resolve_n_jobs(args) == -1
        args = _parser().parse_args(["--ablation", "rho_b", "--n-jobs", "6"])
        assert run_module._resolve_n_jobs(args) == 6

    def test_scale_defaults_to_the_ablations_default(self):
        args = _parser().parse_args(["--ablation", "rho_b"])
        assert args.scale is None

    def test_runs_an_ablation_through_the_runner(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_ablation(ablation, **kwargs):
            calls.append((ablation.name, kwargs))
            return tmp_path / "rd"

        monkeypatch.setattr(run_module, "run_ablation", fake_run_ablation)
        code = main(["--ablation", "rho_b", "--scale", "dev", "--root", str(tmp_path)])
        assert code == 0
        name, kwargs = calls[0]
        assert name == "rho_b"
        assert kwargs["scale"] == "dev"
        assert kwargs["n_jobs"] == -1
        assert kwargs["resume"] is True
        assert kwargs["units"] is None

    def test_timing_batch_passes_the_timing_units_without_resume(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_ablation(ablation, **kwargs):
            calls.append(kwargs)
            return tmp_path / "rd"

        monkeypatch.setattr(run_module, "run_ablation", fake_run_ablation)
        code = main(
            ["--ablation", "rho_b", "--scale", "publication", "--timing-batch",
             "--n-jobs", "5", "--root", str(tmp_path)]
        )
        assert code == 0
        kwargs = calls[0]
        assert kwargs["resume"] is False
        assert kwargs["n_jobs"] == 5
        assert len(kwargs["units"]) == 22
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/test_cli.py -q`
Expected: the new tests fail (`--ablation` unrecognized, `_resolve_n_jobs` missing).

- [ ] Step 3: Extend the CLI

In `src/benchmarks/run.py`, extend the imports:

```python
from ._ablation import run_ablation
from ._ablation_cells import timing_units
from ._artifacts import promote
from ._registry import ABLATIONS, PUBLISHED_RANDOM_STATE, SCENARIOS
from ._run import run_scenario
```

Add to `_parser()` after `--scenario`:

```python
    parser.add_argument(
        "--ablation",
        help="Name of an ablation to run (see --list). Exclusive with --scenario and --all.",
    )
    parser.add_argument(
        "--scale",
        default=None,
        help="Ablation scale to run; defaults to the ablation's own default scale.",
    )
    parser.add_argument(
        "--timing-batch",
        action="store_true",
        help="With --ablation: run only its fixed timing batch, without resume, "
        "to choose --n-jobs.",
    )
```

Change `--n-jobs` to `parser.add_argument("--n-jobs", type=int, default=None, help="Defaults to 1 for scenarios and -1 for an ablation.")`, and add the module-level helper:

```python
def _resolve_n_jobs(args: argparse.Namespace) -> int:
    """Scenarios keep one worker by default: the scaling scenarios time their
    fits, and concurrent workers change those timings. An ablation runs on
    every core."""
    if args.n_jobs is not None:
        return int(args.n_jobs)
    return -1 if args.ablation else 1
```

In `main`, after the `--list` block prints scenarios, add `for name in sorted(ABLATIONS): print(f"ablation:{name}")`. Before the `if args.scenario:` block, add:

```python
    if args.ablation:
        if args.scenario or args.all:
            print("--ablation is exclusive with --scenario and --all.", file=sys.stderr)
            return 2
        if args.ablation not in ABLATIONS:
            print(
                f"Unknown ablation {args.ablation!r}. Valid names: {sorted(ABLATIONS)}.",
                file=sys.stderr,
            )
            return 2
        ablation = ABLATIONS[args.ablation]
        scale = args.scale if args.scale is not None else ablation.default_scale
        units = timing_units(ablation, scale) if args.timing_batch else None
        rd = run_ablation(
            ablation,
            scale=scale,
            root=Path(args.root),
            n_jobs=_resolve_n_jobs(args),
            resume=(not args.no_resume) and not args.timing_batch,
            verbose=args.verbose,
            units=units,
        )
        print(f"{args.ablation} ({scale}): {rd}")
        return 0
```

Replace `n_jobs=args.n_jobs` in the scenario loop with `n_jobs=_resolve_n_jobs(args)`. Update the "Nothing to do" message to name `--ablation` as well.

- [ ] Step 4: Run the tests

Run: `.venv/bin/pytest tests/benchmarks/test_cli.py -q`
Expected: all pass.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/run.py tests/benchmarks/test_cli.py
git commit -m "feat(benchmarks): --ablation, --scale and --timing-batch on the CLI

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Summaries (`_ablation_summary.py`)

Files:
- Create: `src/benchmarks/_ablation_summary.py`
- Test: `tests/benchmarks/test_ablation_summary.py`

Interfaces:
- Consumes: `CELL_KEY` (Task 4); `GENERALIZABILITY_METRICS`, `CARVE_METRICS_ALL` (registry); `wilson_ci` from `_tables`.
- Produces (every function returns a new DataFrame; `x` is `"subsample_ratio"` or `"n_resamples"`; rows with `study == POOLED` pool the simulated studies only):
  - `HEADLINE_METRICS = ("ari_stability_1se", "ari_generalizability_1se")`, `POOLED = "pooled"`
  - `metric_mode(metric_name) -> str`
  - `selection_summary(selection, *, x, metrics=HEADLINE_METRICS)` -> `[x, study, metric_name, n, recovery, recovery_lo, recovery_hi, bias_mean, ari_mean, ari_sem]`
  - `agreement_summary(selection, *, x, metrics=HEADLINE_METRICS)` -> `[x, study, metric_name, agreement, n_datasets]`
  - `spread_summary(curves, *, x, metrics=HEADLINE_METRICS)` -> `[x, study, metric_name, spread]`
  - `curve_at_k_star(curves, datasets, *, x, metrics=HEADLINE_METRICS)` -> `[x, study, metric_name, value_mean, value_sem, se_mean]`
  - `rare_recall_summary(at_k, selection, *, x, metrics=HEADLINE_METRICS, difficulty=None)` -> `[x, study, metric_name, recall_selected, recall_k_star]`
  - `similarity_summary(similarity)` -> `[subsample_ratio, study, estimator, k, ari_mean, ari_sem, n]`
  - `study_selection_shares(selection, *, x, metric, study)` -> `[x, selected_estimator, selected_k, count, n, share]`
  - `diagnostics_summary(cells, *, x)` -> `[x, study, fit_seconds, consensus_nan_fraction, n_cluster_count_warnings]`
  - `table_rows(view, *, x, study, metrics=HEADLINE_METRICS)` -> `[setting, metric_name, recovery, recovery_lo, recovery_hi, ari_mean, agreement, study_modal, study_share]`

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/test_ablation_summary.py`:

```python
"""Tests for the ablation summaries, on hand-built frames."""

import numpy as np
import pandas as pd
import pytest

from benchmarks._ablation_summary import (
    HEADLINE_METRICS,
    POOLED,
    agreement_summary,
    curve_at_k_star,
    diagnostics_summary,
    metric_mode,
    rare_recall_summary,
    selection_summary,
    similarity_summary,
    spread_summary,
    study_selection_shares,
    table_rows,
)
from benchmarks._registry import CARVE_METRICS_ALL
from benchmarks._run import _labels_mode
from benchmarks._tables import wilson_ci

X = "subsample_ratio"
STAB, GEN = HEADLINE_METRICS


def _key(study="gaussians", difficulty="medium", dataset=0, rho=0.618, b=100, rep=0):
    return {
        "study": study, "difficulty": difficulty, "dataset": dataset,
        "subsample_ratio": rho, "n_resamples": b, "replicate": rep,
    }


def _selection():
    rows = []
    # Two datasets, two replicates, two rho values; stability recovers k*
    # at 0.618 and never at 0.2 (replicate 0 picks 3, replicate 1 picks 4);
    # generalizability always does.
    for dataset in (0, 1):
        for rep in (0, 1):
            for rho in (0.2, 0.618):
                for metric in (STAB, GEN):
                    hit = metric == GEN or rho == 0.618
                    rows.append({
                        **_key(dataset=dataset, rho=rho, rep=rep),
                        "metric_name": metric, "selected_estimator": "KMeans",
                        "selected_k": 5 if hit else 3 + rep, "k_star": 5.0,
                        "ari_selected": 0.9 if hit else 0.5,
                    })
    # A study with two replicates that disagree at 0.2 and agree at 0.618.
    for rep in (0, 1):
        for rho in (0.2, 0.618):
            for metric in (STAB, GEN):
                rows.append({
                    **_key(study="klein", difficulty="", dataset=0, rho=rho, rep=rep),
                    "metric_name": metric, "selected_estimator": "AgglomerativeClustering",
                    "selected_k": 4 if (rho == 0.618 or rep == 0) else 3, "k_star": np.nan,
                    "ari_selected": 0.7,
                })
    return pd.DataFrame(rows)


class TestMetricMode:
    @pytest.mark.parametrize("metric", CARVE_METRICS_ALL)
    def test_agrees_with_the_runner(self, metric):
        assert metric_mode(metric) == _labels_mode(metric)


class TestSelectionSummary:
    def test_recovery_bias_and_ari_per_study_and_pooled(self):
        out = selection_summary(_selection(), x=X)
        stab_02 = out[(out["study"] == "gaussians") & (out["metric_name"] == STAB) & (out[X] == 0.2)].iloc[0]
        assert stab_02["n"] == 4
        assert stab_02["recovery"] == 0.0
        assert stab_02["bias_mean"] == pytest.approx(-1.5)  # k-hat 3 or 4 minus 5
        assert stab_02["ari_mean"] == pytest.approx(0.5)
        lo, hi = wilson_ci(0, 4)
        assert (stab_02["recovery_lo"], stab_02["recovery_hi"]) == (lo, hi)
        pooled = out[(out["study"] == POOLED) & (out["metric_name"] == GEN) & (out[X] == 0.618)].iloc[0]
        assert pooled["recovery"] == 1.0

    def test_pooled_rows_exclude_the_study(self):
        out = selection_summary(_selection(), x=X)
        assert "klein" not in set(out["study"])
        assert set(out["study"]) == {"gaussians", POOLED}

    def test_columns(self):
        out = selection_summary(_selection(), x=X)
        assert list(out.columns) == [X, "study", "metric_name", "n", "recovery", "recovery_lo",
                                     "recovery_hi", "bias_mean", "ari_mean", "ari_sem"]


class TestAgreementSummary:
    def test_pairs_that_pick_the_same_k_agree(self):
        out = agreement_summary(_selection(), x=X)
        klein = out[out["study"] == "klein"].set_index([X, "metric_name"])["agreement"]
        assert klein.loc[(0.2, STAB)] == 0.0
        assert klein.loc[(0.618, STAB)] == 1.0
        sims = out[out["study"] == "gaussians"].set_index([X, "metric_name"])["agreement"]
        assert sims.loc[(0.2, STAB)] == 0.0  # replicates picked 3 and 4
        assert sims.loc[(0.618, GEN)] == 1.0

    def test_pooled_excludes_the_study(self):
        out = agreement_summary(_selection(), x=X)
        pooled = out[out["study"] == POOLED].set_index([X, "metric_name"])
        assert pooled.loc[(0.2, STAB), "agreement"] == 0.0
        assert pooled.loc[(0.2, STAB), "n_datasets"] == 2

    def test_a_single_replicate_has_no_agreement(self):
        frame = _selection()
        frame = frame[frame["replicate"] == 0]
        out = agreement_summary(frame, x=X)
        assert out["agreement"].isna().all()

    def test_three_replicates_use_every_pair(self):
        rows = [
            {**_key(rep=rep), "metric_name": STAB, "selected_estimator": "KMeans",
             "selected_k": k, "k_star": 5.0, "ari_selected": 0.9}
            for rep, k in enumerate((5, 5, 4))
        ]
        out = agreement_summary(pd.DataFrame(rows), x=X)
        assert out[out["study"] == "gaussians"]["agreement"].iloc[0] == pytest.approx(1 / 3)


class TestSpreadSummary:
    def test_mean_sd_across_replicates(self):
        rows = []
        for rep, value in ((0, 0.5), (1, 0.7)):
            for k in (3, 4):
                rows.append({**_key(rep=rep), "metric_name": STAB, "estimator": "KMeans",
                             "k": k, "metric_value": value + 0.01 * k, "metric_se": 0.02})
        out = spread_summary(pd.DataFrame(rows), x=X)
        expected = np.std([0.5, 0.7], ddof=1)
        assert out[out["study"] == "gaussians"]["spread"].iloc[0] == pytest.approx(expected)
        assert out[out["study"] == POOLED]["spread"].iloc[0] == pytest.approx(expected)


class TestCurveAtKStar:
    def test_reads_the_value_and_se_at_the_true_k(self):
        curves = pd.DataFrame([
            {**_key(), "metric_name": STAB, "estimator": "KMeans", "k": k,
             "metric_value": 0.1 * k, "metric_se": 0.01 * k}
            for k in (3, 4, 5)
        ])
        datasets = pd.DataFrame([{"study": "gaussians", "difficulty": "medium", "dataset": 0,
                                  "n_samples": 100, "k_star": 5.0, "oracle_ari": 1.0,
                                  "rare_label": 0.0, "rare_fraction": 0.1}])
        out = curve_at_k_star(curves, datasets, x=X)
        row = out[out["study"] == "gaussians"].iloc[0]
        assert row["value_mean"] == pytest.approx(0.5)
        assert row["se_mean"] == pytest.approx(0.05)


class TestRareRecallSummary:
    def test_recall_at_the_selected_k_and_at_k_star(self):
        selection = pd.DataFrame([
            {**_key(), "metric_name": STAB, "selected_estimator": "KMeans", "selected_k": 4,
             "k_star": 5.0, "ari_selected": 0.8},
            {**_key(), "metric_name": GEN, "selected_estimator": "KMeans", "selected_k": 5,
             "k_star": 5.0, "ari_selected": 0.9},
        ])
        at_k = pd.DataFrame([
            {**_key(), "mode": mode, "k": k, "ari_at_k": 0.5,
             "rare_recall_at_k": {4: 0.0, 5: 1.0}[k] if mode == "default" else {4: 0.2, 5: 0.9}[k]}
            for mode in ("default", "generalizability") for k in (4, 5)
        ])
        out = rare_recall_summary(at_k, selection, x=X)
        stab = out[(out["study"] == "gaussians") & (out["metric_name"] == STAB)].iloc[0]
        gen = out[(out["study"] == "gaussians") & (out["metric_name"] == GEN)].iloc[0]
        assert (stab["recall_selected"], stab["recall_k_star"]) == (0.0, 1.0)
        assert (gen["recall_selected"], gen["recall_k_star"]) == (0.9, 0.9)

    def test_difficulty_filter(self):
        selection = pd.DataFrame([
            {**_key(difficulty=d), "metric_name": STAB, "selected_estimator": "KMeans",
             "selected_k": 5, "k_star": 5.0, "ari_selected": 0.8}
            for d in ("medium", "hard")
        ])
        at_k = pd.DataFrame([
            {**_key(difficulty=d), "mode": "default", "k": 5, "ari_at_k": 0.5,
             "rare_recall_at_k": {"medium": 1.0, "hard": 0.0}[d]}
            for d in ("medium", "hard")
        ])
        out = rare_recall_summary(at_k, selection, x=X, metrics=(STAB,), difficulty="hard")
        assert out[out["study"] == "gaussians"]["recall_selected"].iloc[0] == 0.0


class TestSimilaritySummary:
    def test_means_over_datasets_and_draws(self):
        rows = [
            {"study": "gaussians", "difficulty": "medium", "dataset": d, "subsample_ratio": 0.5,
             "estimator": "KMeans", "k": 5, "draw": m, "ari": 0.6 + 0.1 * m}
            for d in (0, 1) for m in (0, 1)
        ]
        out = similarity_summary(pd.DataFrame(rows))
        assert len(out) == 1
        assert out["ari_mean"].iloc[0] == pytest.approx(0.65)
        assert out["n"].iloc[0] == 4


class TestStudySelectionShares:
    def test_shares_per_setting(self):
        out = study_selection_shares(_selection(), x=X, metric=STAB, study="klein")
        at_02 = out[out[X] == 0.2].set_index("selected_k")["share"]
        assert at_02.loc[4] == 0.5 and at_02.loc[3] == 0.5
        at_default = out[out[X] == 0.618]
        assert len(at_default) == 1 and at_default["share"].iloc[0] == 1.0
        assert set(out["n"]) == {2}


class TestDiagnosticsSummary:
    def test_means_per_setting_and_study(self):
        cells = pd.DataFrame([
            {**_key(rho=rho, rep=rep), "carve_random_state": 1, "n_samples": 100,
             "fit_seconds": 10.0 * rho, "consensus_nan_fraction": 0.0 if rho > 0.5 else 0.1,
             "n_cluster_count_warnings": rep}
            for rho in (0.2, 0.618) for rep in (0, 1)
        ])
        out = diagnostics_summary(cells, x=X)
        row = out[out[X] == 0.2].iloc[0]
        assert row["fit_seconds"] == pytest.approx(2.0)
        assert row["consensus_nan_fraction"] == pytest.approx(0.1)
        assert row["n_cluster_count_warnings"] == pytest.approx(0.5)


class TestTableRows:
    def test_one_row_per_setting_and_metric_with_study_mode(self):
        view = {"selection": _selection()}
        rows = table_rows(view, x=X, study="klein")
        assert set(rows["setting"]) == {0.2, 0.618}
        assert set(rows["metric_name"]) == set(HEADLINE_METRICS)
        row = rows[(rows["setting"] == 0.618) & (rows["metric_name"] == STAB)].iloc[0]
        assert row["recovery"] == 1.0
        assert row["study_modal"] == "AgglomerativeClustering, k=4"
        assert row["study_share"] == 1.0
        assert row["agreement"] == 1.0
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/test_ablation_summary.py -q`
Expected: ModuleNotFoundError.

- [ ] Step 3: Create the module

Create `src/benchmarks/_ablation_summary.py`:

```python
"""Per-setting summaries of the ablation frames.

Pure pandas over the frames read_frames returns, so the figures, the table
and the notebook compute nothing themselves. x names the swept column,
"subsample_ratio" for the rho arm and "n_resamples" for the B arm. Rows
with study == POOLED pool the simulated studies; the case study has
difficulty "" and is never pooled with them.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._artifacts import CELL_KEY
from ._registry import GENERALIZABILITY_METRICS
from ._tables import wilson_ci

HEADLINE_METRICS: tuple[str, ...] = ("ari_stability_1se", "ari_generalizability_1se")
POOLED: str = "pooled"
DATASET_KEY: tuple[str, ...] = ("study", "difficulty", "dataset")


def metric_mode(metric_name: str) -> str:
    """Which consensus matrix a metric's labels are cut from; mirrors _run._labels_mode."""
    return "generalizability" if metric_name in GENERALIZABILITY_METRICS else "default"


def _is_simulation(frame: pd.DataFrame) -> pd.Series:
    return frame["difficulty"] != ""


def _with_pooled(frame: pd.DataFrame, group: list[str], agg: dict, *, x: str) -> pd.DataFrame:
    """Aggregate per (x, study, ...) and append pooled rows over the simulations."""
    per = frame.groupby(group, as_index=False).agg(**agg)
    sims = frame[_is_simulation(frame)]
    pooled_group = [g for g in group if g != "study"]
    pooled = sims.groupby(pooled_group, as_index=False).agg(**agg)
    pooled.insert(group.index("study"), "study", POOLED)
    return pd.concat([per, pooled], ignore_index=True)


def selection_summary(
    selection: pd.DataFrame, *, x: str, metrics: Sequence[str] = HEADLINE_METRICS
) -> pd.DataFrame:
    """k* recovery with Wilson bounds, mean bias and mean ARI of the selected labels.

    Simulations only: the study has no true k.
    """
    rows = selection[selection["k_star"].notna() & selection["metric_name"].isin(metrics)].copy()
    rows["hit"] = (rows["selected_k"] == rows["k_star"]).astype(float)
    rows["bias"] = rows["selected_k"] - rows["k_star"]
    out = _with_pooled(
        rows,
        [x, "study", "metric_name"],
        {
            "n": ("hit", "size"),
            "hits": ("hit", "sum"),
            "bias_mean": ("bias", "mean"),
            "ari_mean": ("ari_selected", "mean"),
            "ari_sem": ("ari_selected", "sem"),
        },
        x=x,
    )
    out["recovery"] = out["hits"] / out["n"]
    bounds = [wilson_ci(int(h), int(n)) for h, n in zip(out["hits"], out["n"])]
    out["recovery_lo"] = [lo for lo, _ in bounds]
    out["recovery_hi"] = [hi for _, hi in bounds]
    return out[
        [x, "study", "metric_name", "n", "recovery", "recovery_lo", "recovery_hi",
         "bias_mean", "ari_mean", "ari_sem"]
    ]


def _pair_agreement(choices: pd.Series) -> float:
    n = len(choices)
    if n < 2:
        return np.nan
    counts = choices.value_counts().to_numpy(dtype=float)
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def agreement_summary(
    selection: pd.DataFrame, *, x: str, metrics: Sequence[str] = HEADLINE_METRICS
) -> pd.DataFrame:
    """Fraction of replicate pairs that select the same (estimator, k), per
    dataset, then averaged over datasets. NaN with a single replicate."""
    rows = selection[selection["metric_name"].isin(metrics)].copy()
    rows["choice"] = rows["selected_estimator"] + "@" + rows["selected_k"].astype(str)
    per_dataset = (
        rows.groupby([x, *DATASET_KEY, "metric_name"], as_index=False)["choice"]
        .agg(_pair_agreement)
        .rename(columns={"choice": "agreement"})
    )
    return _with_pooled(
        per_dataset,
        [x, "study", "metric_name"],
        {"agreement": ("agreement", "mean"), "n_datasets": ("agreement", "size")},
        x=x,
    )


def spread_summary(
    curves: pd.DataFrame, *, x: str, metrics: Sequence[str] = HEADLINE_METRICS
) -> pd.DataFrame:
    """Standard deviation of a metric across replicates at each (dataset,
    estimator, k), averaged over those."""
    rows = curves[curves["metric_name"].isin(metrics)]
    sd = (
        rows.groupby([x, *DATASET_KEY, "metric_name", "estimator", "k"], as_index=False)[
            "metric_value"
        ]
        .std(ddof=1)
        .rename(columns={"metric_value": "sd"})
    )
    return _with_pooled(sd, [x, "study", "metric_name"], {"spread": ("sd", "mean")}, x=x)


def curve_at_k_star(
    curves: pd.DataFrame,
    datasets: pd.DataFrame,
    *,
    x: str,
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """Mean metric value and mean reported standard error at the true k."""
    merged = curves.merge(datasets[[*DATASET_KEY, "k_star"]], on=list(DATASET_KEY))
    rows = merged[(merged["k"] == merged["k_star"]) & merged["metric_name"].isin(metrics)]
    return _with_pooled(
        rows,
        [x, "study", "metric_name"],
        {
            "value_mean": ("metric_value", "mean"),
            "value_sem": ("metric_value", "sem"),
            "se_mean": ("metric_se", "mean"),
        },
        x=x,
    )


def rare_recall_summary(
    at_k: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    x: str,
    metrics: Sequence[str] = HEADLINE_METRICS,
    difficulty: str | None = None,
) -> pd.DataFrame:
    """Recall of the smallest true cluster at the selected k and at k*.

    Each metric reads the at_k rows of its own consensus mode. difficulty
    restricts to one axis label (the hard setting is where rare clusters
    are smallest).
    """
    key = list(CELL_KEY)
    parts = []
    for metric in metrics:
        sel = selection[selection["metric_name"] == metric]
        sel = sel[sel["k_star"].notna()][key + ["selected_k", "k_star"]].copy()
        sel["k_star"] = sel["k_star"].astype(int)
        at = at_k[at_k["mode"] == metric_mode(metric)][key + ["k", "rare_recall_at_k"]]
        at_selected = sel.merge(at, left_on=key + ["selected_k"], right_on=key + ["k"])
        at_star = sel.merge(at, left_on=key + ["k_star"], right_on=key + ["k"])
        frame = at_selected[key + ["rare_recall_at_k"]].rename(
            columns={"rare_recall_at_k": "recall_selected"}
        ).merge(
            at_star[key + ["rare_recall_at_k"]].rename(
                columns={"rare_recall_at_k": "recall_k_star"}
            ),
            on=key,
        )
        frame["metric_name"] = metric
        parts.append(frame)
    rows = pd.concat(parts, ignore_index=True)
    if difficulty is not None:
        rows = rows[rows["difficulty"] == difficulty]
    return _with_pooled(
        rows,
        [x, "study", "metric_name"],
        {"recall_selected": ("recall_selected", "mean"), "recall_k_star": ("recall_k_star", "mean")},
        x=x,
    )


def similarity_summary(similarity: pd.DataFrame) -> pd.DataFrame:
    """Mean subsample-versus-full ARI per (rho, study, estimator, k)."""
    return similarity.groupby(
        ["subsample_ratio", "study", "estimator", "k"], as_index=False
    ).agg(ari_mean=("ari", "mean"), ari_sem=("ari", "sem"), n=("ari", "size"))


def study_selection_shares(
    selection: pd.DataFrame, *, x: str, metric: str, study: str
) -> pd.DataFrame:
    """Share of replicates selecting each (estimator, k) at each setting."""
    rows = selection[(selection["study"] == study) & (selection["metric_name"] == metric)]
    counts = rows.groupby([x, "selected_estimator", "selected_k"], as_index=False).size()
    counts = counts.rename(columns={"size": "count"})
    totals = rows.groupby(x, as_index=False).size().rename(columns={"size": "n"})
    out = counts.merge(totals, on=x)
    out["share"] = out["count"] / out["n"]
    return out[[x, "selected_estimator", "selected_k", "count", "n", "share"]]


def diagnostics_summary(cells: pd.DataFrame, *, x: str) -> pd.DataFrame:
    """Mean fit time, NaN fraction and cluster-count warnings per setting and study."""
    return cells.groupby([x, "study"], as_index=False).agg(
        fit_seconds=("fit_seconds", "mean"),
        consensus_nan_fraction=("consensus_nan_fraction", "mean"),
        n_cluster_count_warnings=("n_cluster_count_warnings", "mean"),
    )


def table_rows(
    view: dict[str, pd.DataFrame],
    *,
    x: str,
    study: str,
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """One row per (setting, metric) for the SI table: pooled recovery with
    its Wilson interval, pooled ARI, pooled replicate agreement, and the
    study's modal selection with its share of replicates."""
    selection = view["selection"]
    pooled = selection_summary(selection, x=x, metrics=metrics)
    pooled = pooled[pooled["study"] == POOLED]
    agreement = agreement_summary(selection, x=x, metrics=metrics)
    agreement = agreement[agreement["study"] == POOLED][[x, "metric_name", "agreement"]]
    rows = pooled.merge(agreement, on=[x, "metric_name"], how="left")
    modal = []
    for metric in metrics:
        shares = study_selection_shares(selection, x=x, metric=metric, study=study)
        for setting, group in shares.groupby(x):
            top = group.sort_values(["share", "selected_k"], ascending=[False, True]).iloc[0]
            modal.append(
                {
                    x: setting,
                    "metric_name": metric,
                    "study_modal": f"{top['selected_estimator']}, k={int(top['selected_k'])}",
                    "study_share": float(top["share"]),
                }
            )
    rows = rows.merge(pd.DataFrame(modal), on=[x, "metric_name"], how="left")
    rows = rows.rename(columns={x: "setting"})
    return rows[
        ["setting", "metric_name", "recovery", "recovery_lo", "recovery_hi", "ari_mean",
         "agreement", "study_modal", "study_share"]
    ].sort_values(["setting", "metric_name"], ignore_index=True)
```

- [ ] Step 4: Run the tests

Run: `.venv/bin/pytest tests/benchmarks/test_ablation_summary.py -q`
Expected: all pass. A `FutureWarning` or `DeprecationWarning` from pandas here (for instance about `groupby(...).agg` on an empty group or `observed=`) fails the suite; fix the call, do not add a filter.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_ablation_summary.py tests/benchmarks/test_ablation_summary.py
git commit -m "feat(benchmarks): per-setting summaries of the ablation frames

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: The two SI figures

Files:
- Create: `src/benchmarks/figures/_ablation.py`
- Modify: `src/benchmarks/figures/__init__.py`, `tests/benchmarks/figures/test_figure_contract.py` (`EXPECTED`)
- Modify: `tests/benchmarks/_helpers.py` (synthetic frames builder)
- Test: `tests/benchmarks/figures/test_ablation.py`

Interfaces:
- Consumes: `arm_view`, `REFERENCE_RATIO` (Task 5); every summary (Task 8); `SCENARIO_TITLES` from `figures/_benchmarking_examples.py`; theme names `CARVE_LINEWIDTH, REFERENCE_LINEWIDTH, FALLBACK_COLOR, FOREGROUND_COLOR, FONT_SIZES, cluster_colors, metric_color, save_figure, style_axes, theme_context`; `grouped_legend` from `_panels`.
- Produces: `figure_ablation_rho(frames, *, ablation, scale, save=True, out_dir=None) -> Figure` (nine panels, file `si_fig_ablation_rho.png`), `figure_ablation_b(frames, *, ablation, scale, save=True, out_dir=None) -> Figure` (eight panels, file `si_fig_ablation_b.png`), both under `BENCHMARKING_DIR`; `STUDY_TITLES`; test helper `synthetic_ablation_frames(ablation, scale, *, seed=0) -> dict[str, DataFrame]`.

- [ ] Step 1: Add the synthetic frames builder to the test helpers

Append to `tests/benchmarks/_helpers.py`:

```python
def synthetic_ablation_frames(ablation, scale: str, *, seed: int = 0) -> dict:
    """Every ablation frame for every cell at a scale, with random values.

    Shapes follow the runner exactly (one row per cell, metric, estimator
    and k; at_k for simulations only; similarity per rho-arm dataset,
    including the refit reference), so figures and tables can be exercised
    without a fit.
    """
    from benchmarks._ablation_cells import (
        REFERENCE_RATIO,
        enumerate_cells,
        carve_seed,
    )
    from benchmarks._artifacts import ABLATION_SCHEMAS
    from benchmarks._registry import CARVE_METRICS_ALL

    rng = np.random.default_rng(seed)
    cells = enumerate_cells(ablation, scale)
    rho_cells = enumerate_cells(ablation, scale, arm="rho")
    draws = ablation.scales[scale].similarity_draws

    def _estimators(cell):
        if cell.study == ablation.study:
            return ("AgglomerativeClustering", "SpectralClustering"), tuple(range(2, 11))
        return ("KMeans",), (3, 4, 5, 6, 7)

    datasets, cells_rows, curves, selection, at_k, similarity = [], [], [], [], [], []
    seen = set()
    for cell in cells:
        key = cell.key()
        is_study = cell.study == ablation.study
        estimators, ks = _estimators(cell)
        dataset_id = (cell.study, cell.difficulty, cell.dataset)
        if dataset_id not in seen:
            seen.add(dataset_id)
            datasets.append(
                {
                    "study": cell.study, "difficulty": cell.difficulty, "dataset": cell.dataset,
                    "n_samples": 500,
                    "k_star": np.nan if is_study else 5.0,
                    "oracle_ari": np.nan if is_study else 0.9,
                    "rare_label": np.nan if is_study else 0.0,
                    "rare_fraction": np.nan if is_study else 0.1,
                }
            )
        cells_rows.append(
            {
                **key, "carve_random_state": carve_seed(cell, ablation=ablation),
                "n_samples": 500, "fit_seconds": float(rng.uniform(1, 5)),
                "consensus_nan_fraction": float(rng.uniform(0, 0.05)),
                "n_cluster_count_warnings": int(rng.integers(0, 3)),
            }
        )
        for metric in CARVE_METRICS_ALL:
            for estimator in estimators:
                for k in ks:
                    curves.append(
                        {
                            **key, "metric_name": metric, "estimator": estimator, "k": k,
                            "metric_value": float(rng.uniform(0, 1)),
                            "metric_se": float(rng.uniform(0.01, 0.05)) if metric.startswith("ari_") else np.nan,
                        }
                    )
            selection.append(
                {
                    **key, "metric_name": metric,
                    "selected_estimator": estimators[int(rng.integers(len(estimators)))],
                    "selected_k": int(rng.choice(ks)),
                    "k_star": np.nan if is_study else 5.0,
                    "ari_selected": float(rng.uniform(0, 1)),
                }
            )
        if not is_study:
            for mode in ("default", "generalizability"):
                for k in ks:
                    at_k.append(
                        {
                            **key, "mode": mode, "k": k,
                            "ari_at_k": float(rng.uniform(0, 1)),
                            "rare_recall_at_k": float(rng.uniform(0, 1)),
                        }
                    )
    for dataset_id in dict.fromkeys((c.study, c.difficulty, c.dataset) for c in rho_cells):
        study, difficulty, dataset = dataset_id
        estimators, ks = _estimators(next(c for c in rho_cells if (c.study, c.difficulty, c.dataset) == dataset_id))
        for rho in (*ablation.rho_grid, REFERENCE_RATIO):
            for estimator in estimators:
                for k in ks:
                    for draw in range(draws):
                        similarity.append(
                            {
                                "study": study, "difficulty": difficulty, "dataset": dataset,
                                "subsample_ratio": float(rho), "estimator": estimator, "k": k,
                                "draw": draw, "ari": float(rng.uniform(0.5, 1.0)),
                            }
                        )
    frames = {
        "curves": pd.DataFrame(curves), "selection": pd.DataFrame(selection),
        "at_k": pd.DataFrame(at_k), "cells": pd.DataFrame(cells_rows),
        "datasets": pd.DataFrame(datasets), "similarity": pd.DataFrame(similarity),
    }
    return {name: frame[list(ABLATION_SCHEMAS[name])] for name, frame in frames.items()}
```

- [ ] Step 2: Write the failing tests

Create `tests/benchmarks/figures/test_ablation.py`:

```python
"""Tests for the two SI ablation figures, on synthetic frames."""

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.figure import Figure

from benchmarks._registry import ABLATIONS
from benchmarks.figures import figure_ablation_b, figure_ablation_rho
from tests.benchmarks._helpers import synthetic_ablation_frames

RHO_B = ABLATIONS["rho_b"]


@pytest.fixture(scope="module")
def frames():
    return synthetic_ablation_frames(RHO_B, "dev", seed=1)


class TestFigureAblationRho:
    def test_returns_a_nine_panel_figure(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="dev", save=False)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 9

    def test_saves_under_its_si_name(self, frames, tmp_path):
        figure_ablation_rho(frames, ablation=RHO_B, scale="dev", out_dir=tmp_path)
        assert (tmp_path / "si_fig_ablation_rho.png").exists()

    def test_save_false_writes_nothing(self, frames, tmp_path):
        figure_ablation_rho(frames, ablation=RHO_B, scale="dev", save=False, out_dir=tmp_path)
        assert not list(tmp_path.iterdir())

    def test_marks_the_default_rho_on_every_sweep_panel(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="dev", save=False)
        # The stacked-bar panel (Klein shares) has no reference line.
        with_line = [
            ax for ax in fig.axes
            if any(abs(line.get_xdata()[0] - RHO_B.rho_default) < 1e-9 for line in ax.lines
                   if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1])
        ]
        assert len(with_line) == 8

    def test_similarity_panel_reaches_the_reference_ratio(self, frames):
        fig = figure_ablation_rho(frames, ablation=RHO_B, scale="dev", save=False)
        xmax = max(max(line.get_xdata()) for ax in fig.axes for line in ax.lines if len(line.get_xdata()))
        assert xmax == pytest.approx(1.0)


class TestFigureAblationB:
    def test_returns_an_eight_panel_figure(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="dev", save=False)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 8

    def test_saves_under_its_si_name(self, frames, tmp_path):
        figure_ablation_b(frames, ablation=RHO_B, scale="dev", out_dir=tmp_path)
        assert (tmp_path / "si_fig_ablation_b.png").exists()

    def test_b_axes_are_logarithmic(self, frames):
        fig = figure_ablation_b(frames, ablation=RHO_B, scale="dev", save=False)
        # Every panel but the stacked-bar one sweeps B on a log axis.
        assert sum(ax.get_xscale() == "log" for ax in fig.axes) == 7
```

Add `"figure_ablation_rho"` and `"figure_ablation_b"` to `EXPECTED` in `tests/benchmarks/figures/test_figure_contract.py` and change its docstring count sentence if it names a number.

- [ ] Step 3: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/figures/test_ablation.py tests/benchmarks/figures/test_figure_contract.py -q`
Expected: ImportError on `figure_ablation_b`; the contract tests fail for the two new names.

- [ ] Step 4: Create the figure module and export it

Create `src/benchmarks/figures/_ablation.py`:

```python
"""The two SI figures of the rho/B ablation.

figure_ablation_rho reads the rho arm, figure_ablation_b the B arm; both
draw pooled simulation lines in the foreground color, one line per study
in the cluster palette, and mark the package default with a dotted
reference line. Every number comes from _ablation_summary.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .._ablation_cells import REFERENCE_RATIO, arm_view
from .._ablation_summary import (
    HEADLINE_METRICS,
    POOLED,
    agreement_summary,
    curve_at_k_star,
    rare_recall_summary,
    selection_summary,
    similarity_summary,
    spread_summary,
    study_selection_shares,
)
from .._panels import grouped_legend
from .._registry import METRIC_DISPLAY_NAMES
from .._theme import (
    CARVE_LINEWIDTH,
    FALLBACK_COLOR,
    FONT_SIZES,
    FOREGROUND_COLOR,
    REFERENCE_LINEWIDTH,
    cluster_colors,
    metric_color,
    save_figure,
    style_axes,
    theme_context,
)
from ._benchmarking_examples import SCENARIO_TITLES
from ._paths import BENCHMARKING_DIR, figure_path

STUDY_TITLES: dict[str, str] = {**SCENARIO_TITLES, "klein": "Klein"}
RHO_LABEL = "Subsampling proportion $\\rho$"
B_LABEL = "Resamples $B$"
STAB, GEN = HEADLINE_METRICS


def _title(ax, letter: str, text: str) -> None:
    ax.set_title(f"{letter}  {text}", loc="left", fontsize=FONT_SIZES["title"])


def _reference(ax, x_default: float) -> None:
    ax.axvline(x_default, color=FALLBACK_COLOR, linestyle=":", linewidth=REFERENCE_LINEWIDTH)


def _study_colors(studies: Sequence[str]) -> dict[str, str]:
    return dict(zip(studies, cluster_colors(len(studies))))


def _lines_by_study(
    ax,
    summary: pd.DataFrame,
    *,
    x: str,
    y: str,
    metric: str,
    studies: Sequence[str],
    lo: str | None = None,
    hi: str | None = None,
    pooled_label: str = "Pooled (simulations)",
) -> None:
    """One faint line per study plus the pooled line, for one metric."""
    rows = summary[summary["metric_name"] == metric]
    colors = _study_colors(studies)
    for study in studies:
        part = rows[rows["study"] == study].sort_values(x)
        if part.empty:
            continue
        ax.plot(
            part[x], part[y], marker="o", markersize=3, linewidth=REFERENCE_LINEWIDTH,
            color=colors[study], alpha=0.75, label=STUDY_TITLES.get(study, study),
        )
    pooled = rows[rows["study"] == POOLED].sort_values(x)
    if not pooled.empty:
        ax.plot(
            pooled[x], pooled[y], marker="o", markersize=4, linewidth=CARVE_LINEWIDTH,
            color=FOREGROUND_COLOR, label=pooled_label,
        )
        if lo is not None and hi is not None:
            ax.fill_between(
                pooled[x], pooled[lo], pooled[hi], color=FOREGROUND_COLOR, alpha=0.12, linewidth=0
            )


def _pooled_by_metric(
    ax, summary: pd.DataFrame, *, x: str, y: str, metrics: Sequence[str], sem: str | None = None,
    linestyle: str = "-", suffix: str = "",
) -> None:
    """The pooled line of each metric, in the metric's color."""
    for metric in metrics:
        rows = summary[(summary["metric_name"] == metric) & (summary["study"] == POOLED)]
        rows = rows.sort_values(x)
        if rows.empty:
            continue
        label = METRIC_DISPLAY_NAMES.get(metric, metric) + suffix
        ax.plot(
            rows[x], rows[y], marker="o", markersize=4, linewidth=CARVE_LINEWIDTH,
            linestyle=linestyle, color=metric_color(metric), label=label,
        )
        if sem is not None:
            ax.fill_between(
                rows[x], rows[y] - rows[sem], rows[y] + rows[sem],
                color=metric_color(metric), alpha=0.15, linewidth=0,
            )


def _study_ari(ax, selection: pd.DataFrame, *, x: str, study: str, metrics: Sequence[str]) -> None:
    """Mean and standard error over replicates of the selected labels' ARI
    against the study's reference labels."""
    rows = selection[(selection["study"] == study) & selection["metric_name"].isin(metrics)]
    stats = rows.groupby([x, "metric_name"], as_index=False).agg(
        mean=("ari_selected", "mean"), sem=("ari_selected", "sem")
    )
    for metric in metrics:
        part = stats[stats["metric_name"] == metric].sort_values(x)
        if part.empty:
            continue
        ax.errorbar(
            part[x], part["mean"], yerr=part["sem"].fillna(0.0), marker="o", markersize=4,
            linewidth=CARVE_LINEWIDTH, color=metric_color(metric), capsize=2,
            label=f"{METRIC_DISPLAY_NAMES.get(metric, metric)}, ARI to reference",
        )


def _study_shares(ax, selection: pd.DataFrame, *, x: str, grid: Sequence, study: str, metric: str,
                  x_label: str) -> None:
    """Stacked bars: share of replicates selecting each (estimator, k)."""
    shares = study_selection_shares(selection, x=x, metric=metric, study=study)
    shares["choice"] = shares["selected_estimator"] + ", k=" + shares["selected_k"].astype(str)
    choices = sorted(shares["choice"].unique(), key=lambda c: (c.split(", k=")[0], int(c.split("k=")[1])))
    colors = dict(zip(choices, cluster_colors(len(choices))))
    positions = np.arange(len(grid))
    bottom = np.zeros(len(grid))
    for choice in choices:
        heights = np.array([
            float(shares[(shares[x] == value) & (shares["choice"] == choice)]["share"].sum())
            for value in grid
        ])
        ax.bar(positions, heights, bottom=bottom, color=colors[choice], label=choice, width=0.8)
        bottom += heights
    ax.set_xticks(positions, [str(v) for v in grid])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Share of replicates")
    ax.set_ylim(0, 1)


def _similarity_at_k_star(similarity: pd.DataFrame, datasets: pd.DataFrame) -> pd.DataFrame:
    merged = similarity.merge(datasets[["study", "difficulty", "dataset", "k_star"]],
                              on=["study", "difficulty", "dataset"])
    rows = merged[merged["k"] == merged["k_star"]]
    per = similarity_summary(rows)
    pooled = rows.groupby("subsample_ratio", as_index=False).agg(ari_mean=("ari", "mean"))
    pooled["study"] = POOLED
    per["metric_name"] = "similarity"
    pooled["metric_name"] = "similarity"
    return pd.concat([per, pooled], ignore_index=True)


def figure_ablation_rho(
    frames: Mapping[str, pd.DataFrame],
    *,
    ablation,
    scale: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Selections, scores and recovery against the subsampling proportion."""
    x = "subsample_ratio"
    view = arm_view(frames, ablation=ablation, scale=scale, arm="rho")
    studies = list(ablation.scenarios)
    study = ablation.study
    selection = view["selection"]
    sel = selection_summary(selection, x=x)
    recall = rare_recall_summary(view["at_k"], selection, x=x,
                                 difficulty=ablation.scales[scale].rho_arm.difficulties[-1])
    at_star = curve_at_k_star(view["curves"], view["datasets"], x=x)
    similarity = _similarity_at_k_star(view["similarity"], view["datasets"])
    oracle = float(view["datasets"]["oracle_ari"].mean())

    with theme_context():
        fig, axes = plt.subplots(3, 3, figsize=(12.6, 9.6))
        ax = axes[0, 0]
        _lines_by_study(ax, sel, x=x, y="recovery", metric=STAB, studies=studies,
                        lo="recovery_lo", hi="recovery_hi")
        _title(ax, "A", "k* recovery, stability 1SE")
        ax.set_ylabel("Recovery rate")
        ax = axes[0, 1]
        _lines_by_study(ax, sel, x=x, y="recovery", metric=GEN, studies=studies,
                        lo="recovery_lo", hi="recovery_hi")
        _title(ax, "B", "k* recovery, generalizability 1SE")
        ax = axes[0, 2]
        _pooled_by_metric(ax, sel, x=x, y="bias_mean", metrics=HEADLINE_METRICS)
        ax.axhline(0.0, color=FALLBACK_COLOR, linewidth=REFERENCE_LINEWIDTH)
        _title(ax, "C", "Bias of the selected k")
        ax.set_ylabel("Mean of k-hat minus k*")
        ax = axes[1, 0]
        _pooled_by_metric(ax, sel, x=x, y="ari_mean", metrics=HEADLINE_METRICS, sem="ari_sem")
        ax.axhline(oracle, color=metric_color("baseline_oracle"), linestyle="--",
                   linewidth=REFERENCE_LINEWIDTH, label="Baseline (Oracle k*)")
        _title(ax, "D", "ARI at the selected k")
        ax.set_ylabel("ARI to truth")
        ax = axes[1, 1]
        _pooled_by_metric(ax, recall, x=x, y="recall_selected", metrics=HEADLINE_METRICS,
                          suffix=", at k-hat")
        _pooled_by_metric(ax, recall, x=x, y="recall_k_star", metrics=HEADLINE_METRICS,
                          linestyle=":", suffix=", at k*")
        _title(ax, "E", "Rare-cluster recall, hard setting")
        ax.set_ylabel("Recall of the smallest cluster")
        ax = axes[1, 2]
        _pooled_by_metric(ax, at_star, x=x, y="value_mean", metrics=HEADLINE_METRICS, sem="value_sem")
        _title(ax, "F", "Score at k*")
        ax.set_ylabel("ARI")
        ax = axes[2, 0]
        _lines_by_study(ax, similarity, x=x, y="ari_mean", metric="similarity", studies=studies)
        _title(ax, "G", "Subsample versus full-data clustering at k*")
        ax.set_ylabel("ARI to full-data fit")
        ax = axes[2, 1]
        _study_shares(ax, selection, x=x, grid=list(ablation.rho_grid), study=study, metric=GEN,
                      x_label=RHO_LABEL)
        _title(ax, "H", f"{STUDY_TITLES.get(study, study)}: selection, generalizability 1SE")
        ax = axes[2, 2]
        _study_ari(ax, selection, x=x, study=study, metrics=HEADLINE_METRICS)
        klein_sim = similarity_summary(view["similarity"][view["similarity"]["study"] == study])
        klein_sim = klein_sim[klein_sim["k"] == 4]
        for estimator, part in klein_sim.groupby("estimator"):
            part = part.sort_values(x)
            ax.plot(part[x], part["ari_mean"], marker="s", markersize=3, linestyle=":",
                    linewidth=REFERENCE_LINEWIDTH, color=FOREGROUND_COLOR,
                    alpha=0.9 if estimator.startswith("Agglomerative") else 0.5,
                    label=f"{estimator}, subsample vs full at k=4")
        _title(ax, "I", f"{STUDY_TITLES.get(study, study)}: selected labels and similarity")
        ax.set_ylabel("ARI")

        for ax in axes.flat:
            if ax is not axes[2, 1]:
                _reference(ax, ablation.rho_default)
                ax.set_xlabel(RHO_LABEL)
            style_axes(ax)
        fig.tight_layout()
        grouped_legend(fig, axes, y_offset=0.06, ncol=4)
        if save:
            save_figure(fig, figure_path("si_fig_ablation_rho.png", subdir=BENCHMARKING_DIR,
                                         out_dir=out_dir))
    return fig


def figure_ablation_b(
    frames: Mapping[str, pd.DataFrame],
    *,
    ablation,
    scale: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Repeatability, spread and selections against the resample count."""
    x = "n_resamples"
    view = arm_view(frames, ablation=ablation, scale=scale, arm="b")
    studies = [*ablation.scenarios, ablation.study]
    study = ablation.study
    selection = view["selection"]
    agreement = agreement_summary(selection, x=x)
    spread = spread_summary(view["curves"], x=x)
    at_star = curve_at_k_star(view["curves"], view["datasets"], x=x)
    sel = selection_summary(selection, x=x)
    b_default = ablation.b_default

    def _guide(ax, summary, y):
        # One over the square root of B, anchored at the default.
        rows = summary[(summary["study"] == POOLED) & (summary["metric_name"] == STAB)]
        anchor = rows[rows[x] == b_default][y]
        if anchor.empty or np.isnan(anchor.iloc[0]):
            return
        grid = np.array(ablation.b_grid, dtype=float)
        ax.plot(grid, float(anchor.iloc[0]) * np.sqrt(b_default / grid), linestyle="--",
                color=FALLBACK_COLOR, linewidth=REFERENCE_LINEWIDTH, label="1 / sqrt(B) guide")

    with theme_context():
        fig, axes = plt.subplots(4, 2, figsize=(8.4, 12.8))
        ax = axes[0, 0]
        _lines_by_study(ax, agreement, x=x, y="agreement", metric=STAB, studies=studies)
        _title(ax, "A", "Replicate agreement, stability 1SE")
        ax.set_ylabel("Fraction of agreeing pairs")
        ax = axes[0, 1]
        _lines_by_study(ax, agreement, x=x, y="agreement", metric=GEN, studies=studies)
        _title(ax, "B", "Replicate agreement, generalizability 1SE")
        ax = axes[1, 0]
        _pooled_by_metric(ax, spread, x=x, y="spread", metrics=HEADLINE_METRICS)
        _guide(ax, spread, "spread")
        _title(ax, "C", "Curve spread across replicates")
        ax.set_ylabel("SD of the score over replicates")
        ax = axes[1, 1]
        _pooled_by_metric(ax, at_star, x=x, y="se_mean", metrics=HEADLINE_METRICS)
        _guide(ax, at_star, "se_mean")
        _title(ax, "D", "Reported standard error at k*")
        ax.set_ylabel("Mean standard error")
        ax = axes[2, 0]
        _pooled_by_metric(ax, sel, x=x, y="recovery", metrics=HEADLINE_METRICS)
        _title(ax, "E", "k* recovery")
        ax.set_ylabel("Recovery rate")
        ax = axes[2, 1]
        _pooled_by_metric(ax, sel, x=x, y="bias_mean", metrics=HEADLINE_METRICS)
        ax.axhline(0.0, color=FALLBACK_COLOR, linewidth=REFERENCE_LINEWIDTH)
        _title(ax, "F", "Bias of the selected k")
        ax.set_ylabel("Mean of k-hat minus k*")
        ax = axes[3, 0]
        _study_ari(ax, selection, x=x, study=study, metrics=HEADLINE_METRICS)
        _title(ax, "G", f"{STUDY_TITLES.get(study, study)}: selected labels")
        ax.set_ylabel("ARI to reference labels")
        ax = axes[3, 1]
        _study_shares(ax, selection, x=x, grid=list(ablation.b_grid), study=study, metric=GEN,
                      x_label=B_LABEL)
        _title(ax, "H", f"{STUDY_TITLES.get(study, study)}: selection, generalizability 1SE")

        for ax in axes.flat:
            if ax is not axes[3, 1]:
                ax.set_xscale("log")
                _reference(ax, b_default)
                ax.set_xlabel(B_LABEL)
            style_axes(ax)
        fig.tight_layout()
        grouped_legend(fig, axes, y_offset=0.04, ncol=4)
        if save:
            save_figure(fig, figure_path("si_fig_ablation_b.png", subdir=BENCHMARKING_DIR,
                                         out_dir=out_dir))
    return fig
```

In `src/benchmarks/figures/__init__.py`, add `from ._ablation import figure_ablation_b, figure_ablation_rho` and both names to `__all__` (alphabetical order).

- [ ] Step 5: Run the tests

Run: `.venv/bin/pytest tests/benchmarks/figures/test_ablation.py tests/benchmarks/figures/test_figure_contract.py tests/benchmarks/test_theme.py -q`
Expected: all pass. If `test_marks_the_default_rho_on_every_sweep_panel` counts wrong, check that `axvline` produces a `Line2D` with two equal x values (it does) and that the Klein shares panel is the only one skipped. If matplotlib warns about `tight_layout` with the figure legend, place the legend after `tight_layout` (as written) and do not filter the warning.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/figures/_ablation.py src/benchmarks/figures/__init__.py tests/benchmarks/_helpers.py tests/benchmarks/figures/test_ablation.py tests/benchmarks/figures/test_figure_contract.py
git commit -m "feat(figures): the two SI ablation figures

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: The SI table

Files:
- Modify: `src/benchmarks/_tables.py` (append)
- Modify: `.gitignore` (add `results/tables/`)
- Test: `tests/benchmarks/test_tables.py`

Interfaces:
- Consumes: `table_rows`, `HEADLINE_METRICS` (Task 8, imported lazily inside the writer because `_ablation_summary` imports `wilson_ci` from this module); `arm_view` (Task 5); `synthetic_ablation_frames` (Task 9 helper).
- Produces: `ESTIMATOR_SHORT_NAMES`, `render_ablation_tex(rows_by_arm, *, metrics, caption, label, study_title) -> str`, `write_ablation_table(frames, *, ablation, scale, out_dir, name="si_table_ablation") -> Path`.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_tables.py` (extend the `_tables` import with `render_ablation_tex, write_ablation_table`):

```python
class TestAblationTable:
    @pytest.fixture(scope="class")
    def frames(self):
        from benchmarks._registry import ABLATIONS
        from tests.benchmarks._helpers import synthetic_ablation_frames

        return ABLATIONS["rho_b"], synthetic_ablation_frames(ABLATIONS["rho_b"], "dev", seed=2)

    def test_renders_two_sub_tables_with_every_setting(self, frames):
        ablation, data = frames
        from benchmarks._ablation_cells import arm_view
        from benchmarks._ablation_summary import table_rows

        rows = {
            "rho": table_rows(arm_view(data, ablation=ablation, scale="dev", arm="rho"),
                              x="subsample_ratio", study="klein"),
            "b": table_rows(arm_view(data, ablation=ablation, scale="dev", arm="b"),
                            x="n_resamples", study="klein"),
        }
        tex = render_ablation_tex(rows, caption="Sensitivity", label="tab:ablation",
                                  study_title="Klein")
        assert tex.count(r"\begin{tabular}") == 2
        for rho in ablation.rho_grid:
            assert f"\n{rho:g} &" in tex
        for b in ablation.b_grid:
            assert f"\n{b} &" in tex
        assert "Ward" in tex or "Spectral" in tex
        assert r"\caption{Sensitivity}" in tex
        assert "nan" not in tex

    def test_write_ablation_table_writes_the_fragment(self, frames, tmp_path):
        ablation, data = frames
        path = write_ablation_table(data, ablation=ablation, scale="dev", out_dir=tmp_path)
        assert path == tmp_path / "si_table_ablation.tex"
        assert r"\begin{table}" in path.read_text()
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/test_tables.py -q -k Ablation`
Expected: ImportError on `render_ablation_tex`.

- [ ] Step 3: Implement

Append to `src/benchmarks/_tables.py`:

```python
# --- The rho/B ablation table -----------------------------------------------
ESTIMATOR_SHORT_NAMES: dict[str, str] = {
    "AgglomerativeClustering": "Ward",
    "SpectralClustering": "Spectral",
    "KMeans": "KMeans",
}

_ARM_HEADINGS: dict[str, str] = {"rho": r"$\rho$", "b": "$B$"}


def _fmt(value: float, decimals: int) -> str:
    return "" if pd.isna(value) else f"{value:.{decimals}f}"


def _modal(text: object, share: float) -> str:
    if pd.isna(share) or not isinstance(text, str):
        return ""
    estimator, k = text.split(", ")
    return f"{ESTIMATOR_SHORT_NAMES.get(estimator, estimator)}, ${k}$ ({share:.2f})"


def render_ablation_tex(
    rows_by_arm: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str] = ("ari_stability_1se", "ari_generalizability_1se"),
    caption: str,
    label: str,
    study_title: str,
    decimals: int = 3,
) -> str:
    """Two sub-tables, one row per rho value and one per B value.

    Per headline selector: pooled k* recovery with its Wilson interval, the
    pooled mean ARI of the selected labels, in the B sub-table the pooled
    replicate agreement, and the study's modal selection with its share.
    """
    lines = [r"\begin{table}[ht]", r"\centering", f"\\caption{{{_tex_escape(caption)}}}",
             f"\\label{{{label}}}"]
    for arm in ("rho", "b"):
        rows = rows_by_arm[arm]
        with_agreement = arm == "b"
        per_metric = 4 if with_agreement else 3
        lines += [f"\\begin{{tabular}}{{l{'c' * per_metric * len(metrics)}}}", r"\hline"]
        head = [""] + [
            f"\\multicolumn{{{per_metric}}}{{c}}{{{_tex_escape(METRIC_DISPLAY_NAMES.get(m, m))}}}"
            for m in metrics
        ]
        lines.append(" & ".join(head) + r" \\")
        sub = [_ARM_HEADINGS[arm]]
        for _ in metrics:
            sub += ["$k$-rec [95\\% CI]", "ARI"]
            if with_agreement:
                sub.append("Agreement")
            sub.append(f"{_tex_escape(study_title)} modal (share)")
        lines += [" & ".join(sub) + r" \\", r"\hline"]
        for setting in sorted(rows["setting"].unique()):
            cells = [f"{setting:g}"]
            for metric in metrics:
                match = rows[(rows["setting"] == setting) & (rows["metric_name"] == metric)]
                if match.empty:
                    cells += [""] * per_metric
                    continue
                row = match.iloc[0]
                cells.append(
                    f"{_fmt(row['recovery'], 2)} [{_fmt(row['recovery_lo'], 2)}, "
                    f"{_fmt(row['recovery_hi'], 2)}]"
                )
                cells.append(_fmt(row["ari_mean"], decimals))
                if with_agreement:
                    cells.append(_fmt(row["agreement"], 2))
                cells.append(_modal(row["study_modal"], row["study_share"]))
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}"]
        if arm == "rho":
            lines.append(r"\vspace{1em}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_ablation_table(
    frames: Mapping[str, pd.DataFrame],
    *,
    ablation,
    scale: str,
    out_dir: Path,
    name: str = "si_table_ablation",
    caption: str = "Sensitivity of CARVE's selections to the subsampling proportion and the resample count.",
    label: str = "tab:ablation_rho_b",
) -> Path:
    """Summarize a run's two arms and write the .tex fragment."""
    # Imported here: _ablation_summary imports wilson_ci from this module.
    from ._ablation_cells import arm_view
    from ._ablation_summary import table_rows

    rows = {
        "rho": table_rows(arm_view(frames, ablation=ablation, scale=scale, arm="rho"),
                          x="subsample_ratio", study=ablation.study),
        "b": table_rows(arm_view(frames, ablation=ablation, scale=scale, arm="b"),
                        x="n_resamples", study=ablation.study),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tex"
    path.write_text(
        render_ablation_tex(rows, caption=caption, label=label,
                            study_title=ablation.study.capitalize())
    )
    return path
```

`Mapping` is already imported in `_tables.py` through `collections.abc`? It imports `Sequence`; extend that import to `from collections.abc import Mapping, Sequence`.

Add `results/tables/` to `.gitignore` after `results/runs/`.

- [ ] Step 4: Run the tests

Run: `.venv/bin/pytest tests/benchmarks/test_tables.py tests/benchmarks/test_tables_cli.py -q`
Expected: all pass.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_tables.py tests/benchmarks/test_tables.py .gitignore
git commit -m "feat(benchmarks): the SI ablation table

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: The notebook

Files:
- Create: `notebooks/Resampling_Ablation.ipynb` (built by the script below; the script itself is not committed)
- Modify: `tests/benchmarks/test_notebooks.py` (`NOTEBOOKS`, pins)

Interfaces:
- Consumes: everything above; `REPO_ROOT` from `benchmarks.figures._paths`.
- Produces: a notebook that reads one run directory and shows every summary, the two figures and the table.

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/test_notebooks.py`, add `"ablation": REPO_ROOT / "notebooks" / "Resampling_Ablation.ipynb",` to `NOTEBOOKS`, and append:

```python
def test_ablation_notebook_reads_its_config_from_the_registry():
    source = _code(NOTEBOOKS["ablation"])
    assert 'ABLATIONS["rho_b"]' in source
    assert "read_frames(" in source
    assert "arm_view(" in source
    assert "figure_ablation_rho(" in source
    assert "figure_ablation_b(" in source
    assert "write_ablation_table(" in source
    # No grid literal: the rho and B values come from the Ablation.
    for restated in ("0.618", "(10, 25", "[10, 25", "0.2, 0.3"):
        assert restated not in source


def test_ablation_notebook_computes_nothing():
    source = _code(NOTEBOOKS["ablation"])
    for forbidden in ("run_ablation(", "CARVE(", "fit_carve(", ".fit("):
        assert forbidden not in source
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/benchmarks/test_notebooks.py -q`
Expected: the parametrized notebook tests error on the missing file; the two new tests fail.

- [ ] Step 3: Build the notebook

Run this script once from `code/` (it is scratch; do not commit it):

```bash
.venv/bin/python - <<'PY'
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

md("""# Resampling ablation: sensitivity to rho and B

This notebook reads one run of the `rho_b` ablation and shows every summary the
SI figures and table are built from. It computes nothing: produce a run with

```
python -m benchmarks.run --ablation rho_b --scale dev
python -m benchmarks.run --ablation rho_b --scale publication
```

from `code/`, then set `SCALE` below. The rho arm sweeps the subsampling
proportion at the default resample count; the B arm sweeps the resample count
at the default proportion. The Klein case study runs through the same cells
with replicate fits; the simulations carry the ground truth.""")

md("## 1) Setup")
code("""from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmarks._ablation_cells import REFERENCE_RATIO, arm_view, enumerate_cells
from benchmarks._ablation_summary import (
    HEADLINE_METRICS,
    POOLED,
    agreement_summary,
    curve_at_k_star,
    diagnostics_summary,
    rare_recall_summary,
    selection_summary,
    similarity_summary,
    spread_summary,
    study_selection_shares,
    table_rows,
)
from benchmarks._artifacts import read_frames
from benchmarks._registry import ABLATIONS, METRIC_DISPLAY_NAMES
from benchmarks._tables import write_ablation_table
from benchmarks._theme import metric_color, theme_context
from benchmarks.figures import figure_ablation_b, figure_ablation_rho
from benchmarks.figures._paths import REPO_ROOT

ABLATION = ABLATIONS["rho_b"]
SCALE = "dev"  # "publication" for the reported run
STUDY = ABLATION.study
RHO, B = "subsample_ratio", "n_resamples"

runs = sorted((REPO_ROOT / "results" / "runs" / f"ablation_{ABLATION.name}" / SCALE).glob("*/manifest.json"))
if not runs:
    raise FileNotFoundError(f"No {SCALE} run of ablation {ABLATION.name}; run python -m benchmarks.run --ablation {ABLATION.name} --scale {SCALE}")
RUN_DIR = runs[-1].parent
frames = read_frames(RUN_DIR)
rho_view = arm_view(frames, ablation=ABLATION, scale=SCALE, arm="rho")
b_view = arm_view(frames, ablation=ABLATION, scale=SCALE, arm="b")
print(RUN_DIR)
print({name: len(frame) for name, frame in frames.items()})
print("cells expected:", len(enumerate_cells(ABLATION, SCALE)), "recorded:", len(frames["cells"]))""")

code("""import json

manifest = json.loads((RUN_DIR / "manifest.json").read_text())
pd.Series({k: manifest[k] for k in ("run_id", "scale", "config_hash", "workers", "thread_cap", "cores", "n_units", "wall_clock_s", "git_sha")})""")

md("""## 2) Rho arm, simulations

Each plot below draws one line per scenario and the pooled line over all
simulated cells, against the subsampling proportion, for the two headline
selectors. Every difficulty of the rho arm is included; the hard setting is
shown on its own where rare clusters matter.""")

code("""def lines(ax, summary, *, x, y, metric, lo=None, hi=None, title=None, y_label=None):
    rows = summary[summary["metric_name"] == metric]
    for study, part in rows.groupby("study"):
        part = part.sort_values(x)
        if study == POOLED:
            ax.plot(part[x], part[y], color="black", linewidth=2, marker="o", label="pooled")
            if lo is not None:
                ax.fill_between(part[x], part[lo], part[hi], color="black", alpha=0.1)
        else:
            ax.plot(part[x], part[y], marker="o", markersize=3, alpha=0.7, label=study)
    ax.set_xlabel(x)
    ax.set_ylabel(y_label or y)
    ax.set_title(title or METRIC_DISPLAY_NAMES.get(metric, metric))


def pair(summary, *, x, y, lo=None, hi=None, y_label=None, logx=False):
    with theme_context():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
        for ax, metric in zip(axes, HEADLINE_METRICS):
            lines(ax, summary, x=x, y=y, metric=metric, lo=lo, hi=hi, y_label=y_label)
            if logx:
                ax.set_xscale("log")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
    return fig""")

code("""rho_sel = selection_summary(rho_view["selection"], x=RHO)
pair(rho_sel, x=RHO, y="recovery", lo="recovery_lo", hi="recovery_hi", y_label="k* recovery")
rho_sel[rho_sel["study"] == POOLED]""")

code("""pair(rho_sel, x=RHO, y="bias_mean", y_label="mean of k-hat minus k*")
pair(rho_sel, x=RHO, y="ari_mean", y_label="ARI at the selected k");""")

code("""rho_star = curve_at_k_star(rho_view["curves"], rho_view["datasets"], x=RHO)
pair(rho_star, x=RHO, y="value_mean", y_label="score at k*")
pair(rho_star, x=RHO, y="se_mean", y_label="reported standard error at k*");""")

code("""hard = ABLATION.scales[SCALE].rho_arm.difficulties[-1]
rho_recall = rare_recall_summary(rho_view["at_k"], rho_view["selection"], x=RHO, difficulty=hard)
pair(rho_recall, x=RHO, y="recall_selected", y_label=f"rare-cluster recall at k-hat ({hard})")
pair(rho_recall, x=RHO, y="recall_k_star", y_label=f"rare-cluster recall at k* ({hard})");""")

code("""# Score curves over k at every rho, per scenario, for the stability selector.
curves = rho_view["curves"]
sims = curves[curves["difficulty"] != ""]
with theme_context():
    scenarios = list(ABLATION.scenarios)
    fig, axes = plt.subplots(2, len(scenarios), figsize=(3.2 * len(scenarios), 6), sharey="row")
    for row, metric in enumerate(("ari_stability", "ari_generalizability")):
        for col, name in enumerate(scenarios):
            ax = axes[row, col]
            part = sims[(sims["study"] == name) & (sims["metric_name"] == metric)]
            for rho, group in part.groupby(RHO):
                mean = group.groupby("k")["metric_value"].mean()
                ax.plot(mean.index, mean.values, marker="o", markersize=3, label=f"rho={rho:g}")
            ax.set_title(f"{name}: {METRIC_DISPLAY_NAMES.get(metric, metric)}", fontsize=8)
            ax.set_xlabel("k")
    axes[0, 0].set_ylabel("stability ARI")
    axes[1, 0].set_ylabel("generalizability ARI")
    axes[0, -1].legend(fontsize=6)
    fig.tight_layout()""")

md("## 3) Rho arm, Klein")
code("""klein_sel = rho_view["selection"][rho_view["selection"]["study"] == STUDY]
for metric in HEADLINE_METRICS:
    print(METRIC_DISPLAY_NAMES.get(metric, metric))
    display(study_selection_shares(klein_sel, x=RHO, metric=metric, study=STUDY))""")

code("""klein_ari = klein_sel[klein_sel["metric_name"].isin(HEADLINE_METRICS)].groupby([RHO, "metric_name"], as_index=False).agg(
    ari_mean=("ari_selected", "mean"), ari_sem=("ari_selected", "sem"), n=("ari_selected", "size")
)
with theme_context():
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for metric, part in klein_ari.groupby("metric_name"):
        ax.errorbar(part[RHO], part["ari_mean"], yerr=part["ari_sem"], marker="o", capsize=2, color=metric_color(metric), label=METRIC_DISPLAY_NAMES.get(metric, metric))
    ax.set_xlabel(RHO); ax.set_ylabel("ARI of selected labels to time points"); ax.legend(fontsize=7)
klein_ari""")

md("## 4) B arm, simulations and Klein")
code("""b_agree = agreement_summary(b_view["selection"], x=B)
pair(b_agree, x=B, y="agreement", y_label="replicate agreement on the selection", logx=True)
b_agree[b_agree["study"].isin([POOLED, STUDY])]""")

code("""b_spread = spread_summary(b_view["curves"], x=B)
pair(b_spread, x=B, y="spread", y_label="SD of the score over replicates", logx=True)
b_star = curve_at_k_star(b_view["curves"], b_view["datasets"], x=B)
pair(b_star, x=B, y="se_mean", y_label="reported standard error at k*", logx=True);""")

code("""b_sel = selection_summary(b_view["selection"], x=B)
pair(b_sel, x=B, y="recovery", lo="recovery_lo", hi="recovery_hi", y_label="k* recovery", logx=True)
pair(b_sel, x=B, y="bias_mean", y_label="mean of k-hat minus k*", logx=True)
b_sel[b_sel["study"] == POOLED]""")

code("""klein_b = b_view["selection"][b_view["selection"]["study"] == STUDY]
for metric in HEADLINE_METRICS:
    print(METRIC_DISPLAY_NAMES.get(metric, metric))
    display(study_selection_shares(klein_b, x=B, metric=metric, study=STUDY))""")

md("""## 5) Subsample versus full-data similarity

ARI between a clustering of a subsample and the full-data clustering
restricted to that subsample, per scenario and candidate k. The point at
ratio 1 is the refit reference: the full data clustered again with another
seed, which separates the estimator's own randomness from subsampling.""")
code("""sim = similarity_summary(rho_view["similarity"])
with theme_context():
    studies = [*ABLATION.scenarios, STUDY]
    fig, axes = plt.subplots(1, len(studies), figsize=(3.0 * len(studies), 3.2), sharey=True)
    for ax, name in zip(axes, studies):
        part = sim[sim["study"] == name]
        for (estimator, k), group in part.groupby(["estimator", "k"]):
            group = group.sort_values(RHO)
            ax.plot(group[RHO], group["ari_mean"], marker="o", markersize=3, label=f"{estimator[:5]} k={k}")
        ax.axvline(REFERENCE_RATIO, color="grey", linestyle=":")
        ax.set_title(name, fontsize=8); ax.set_xlabel(RHO)
    axes[0].set_ylabel("ARI to full-data fit"); axes[-1].legend(fontsize=5)
    fig.tight_layout()
sim[sim["study"] == STUDY]""")

md("## 6) Diagnostics")
code("""diag_rho = diagnostics_summary(rho_view["cells"], x=RHO)
diag_b = diagnostics_summary(b_view["cells"], x=B)
with theme_context():
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for row, (diag, x) in enumerate(((diag_rho, RHO), (diag_b, B))):
        for col, y in enumerate(("fit_seconds", "consensus_nan_fraction", "n_cluster_count_warnings")):
            ax = axes[row, col]
            for study, part in diag.groupby("study"):
                part = part.sort_values(x)
                ax.plot(part[x], part[y], marker="o", markersize=3, label=study)
            ax.set_xlabel(x); ax.set_ylabel(y)
            if x == B:
                ax.set_xscale("log")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
pd.concat([diag_rho.assign(arm="rho"), diag_b.assign(arm="b")])""")

md("""## 7) SI figures and table

Written under `vis/benchmarking/` and `results/tables/`; copying into the
manuscript stays a manual step.""")
code("""figure_ablation_rho(frames, ablation=ABLATION, scale=SCALE)
figure_ablation_b(frames, ablation=ABLATION, scale=SCALE)
table_path = write_ablation_table(frames, ablation=ABLATION, scale=SCALE, out_dir=REPO_ROOT / "results" / "tables")
print(table_path.read_text())""")

code("""table_rows(rho_view, x=RHO, study=STUDY)""")
code("""table_rows(b_view, x=B, study=STUDY)""")

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}})
nbformat.write(nb, "notebooks/Resampling_Ablation.ipynb")
print("written")
PY
```

- [ ] Step 4: Run the notebook tests

Run: `.venv/bin/pytest tests/benchmarks/test_notebooks.py -q`
Expected: all pass (including `test_no_sys_path_manipulation`, `test_imports_come_from_the_installed_package` and `test_no_gridspec_layout_in_notebooks` for the new notebook).

The notebook is executed in Task 13, after the dev run exists.

- [ ] Step 5: Commit

```bash
git add notebooks/Resampling_Ablation.ipynb tests/benchmarks/test_notebooks.py
git commit -m "docs(notebooks): Resampling_Ablation reads a run and shows every summary

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Manuscript drafts (outside the repo)

Files:
- Create: `../_claude_playground/rho_b_ablation/manuscript_drafts.tex`

No tests; a LaTeX file of draft paragraphs the author moves into the manuscript by hand. Nothing is written under `../overleaf/`.

- [ ] Step 1: Write the drafts

Create `/Users/kaiwycik/GitHub/CARVE/_claude_playground/rho_b_ablation/manuscript_drafts.tex`:

```latex
% Drafts for the rho/B sensitivity analysis (reviewer comments R4.4 and R1.3).
% Two paragraphs need no results and can be placed now; the SI Text is
% written after the publication run from the notebook's outputs.

% ---------------------------------------------------------------------------
% 1. Methods, where rho and B are introduced (CARVE_manuscript.tex, the
%    paragraph at line 499 that reads "Additional run-level parameters are
%    the number of resampling iterations B and the subsampling proportion
%    rho"). Append after that sentence.
% ---------------------------------------------------------------------------
By default $B = 100$ and $\rho = 0.618$. The proportion is the positive root of $\rho^2 = 1 - \rho$, which balances the two evaluation sets: the overlap of the two subsamples on which stability is scored has expected size $\rho^2 n$, and the hold-out set on which generalizability is scored has size $(1 - \rho) n$, so both receive about $0.38\,n$ samples. Both parameters are analyzed in \nameref{S9_Text}, which sweeps $\rho$ from 0.2 to 0.9 and $B$ from 10 to 200 on the simulated benchmarks and the Klein data.

% ---------------------------------------------------------------------------
% 2. S1 Text, "Detailed methodology": a paragraph after the one that defines
%    P_1, P_2 and P_test (line 678).
% ---------------------------------------------------------------------------
\textbf{Choice of the subsampling proportion.}
Each iteration clusters three sets: the two subsamples $P_1^{(b)}$ and $P_2^{(b)}$ of size $\lfloor \rho n \rfloor$, and the hold-out set $P_{\mathrm{test}}^{(b)}$ of size $n - \lfloor \rho n \rfloor$. Stability is scored on the overlap $P_1^{(b)} \cap P_2^{(b)}$, which has expected size $\rho^2 n$ because the two subsamples are drawn independently, and generalizability is scored on $P_{\mathrm{test}}^{(b)}$, of size $(1 - \rho) n$. Equating the two, $\rho^2 = 1 - \rho$, gives $\rho = (\sqrt{5} - 1)/2 \approx 0.618$, the default. A pair of samples is co-sampled in $P_1^{(b)}$ in about $B \rho^2$ iterations, so the consensus matrix at the defaults rests on about 38 co-samples per pair; at $\rho = 0.2$ and $B = 100$ it rests on about 4, and a small fraction of pairs is never co-sampled. The sensitivity of the selections and scores to $\rho$ and $B$ is reported in \nameref{S9_Text}.

% ---------------------------------------------------------------------------
% 3. S9 Text (new). Skeleton; the results paragraphs are written after the
%    publication run from notebooks/Resampling_Ablation.ipynb. Keep the
%    statement that rho cannot reach 1 in CARVE.
% ---------------------------------------------------------------------------
\paragraph*{S9 Text.}
\label{S9_Text}
\textbf{Sensitivity to the subsampling proportion and the resample count.}

\textbf{Design.}
We varied one parameter at a time. The subsampling proportion $\rho$ took the values $0.2, 0.3, 0.4, 0.5, 0.618, 0.7, 0.8$ and $0.9$ at $B = 100$, and the resample count $B$ took the values $10, 25, 50, 100$ and $200$ at $\rho = 0.618$. The $\rho$ sweep ran on ten simulated data sets at each of the three difficulty settings of the six cluster shapes of \nameref{S3_Text}, with the same seeds as \nameref{S2_Table}--\nameref{S7_Table}, and on the Klein subsample of \nameref{S6_Text} with ten CARVE fits per setting under distinct seeds. The $B$ sweep ran on five data sets at the medium and hard settings with three CARVE fits per data set, and on Klein with ten fits per setting. Selections use the stability and generalizability 1-SE rules of the main text. For the simulations we report the rate at which the selected $k$ equals $k^\star$ with Wilson intervals, the mean of $\widehat{k} - k^\star$, the ARI of the consensus labels to the truth, and the recall of the smallest true cluster after Hungarian alignment, at $\widehat{k}$ and at $k^\star$. For $B$ we report, across the fits of one data set, the fraction of pairs of fits that select the same $k$ and the standard deviation of the score curves. To address whether the clustering of a resample resembles that of the full data, we clustered subsamples of size $\lfloor \rho n \rfloor$ with the same estimator and $k$ and computed the ARI against the full-data clustering restricted to the subsample, with twenty draws per data set and $k$; the reference at $\rho = 1$ refits the full data under a new seed. Note that $\rho = 1$ is not attainable within CARVE, whose hold-out set would be empty, so the $\rho = 0.9$ end of the sweep and this reference are the closest evidence about the full data.

\textbf{Results.}
% Written after the publication run. Cover, in this order: the shape of the
% stability and generalizability curves over rho (S9 Fig A-F); whether the
% 1-SE selections move, and in which direction, at the ends of the sweep;
% rare-cluster recall at the hard setting (R1.3); subsample-versus-full
% similarity (R1.3); replicate agreement and curve spread over B (S10 Fig);
% Klein's selection at every setting; and the consensus-matrix metrics at
% small rho (co-sampling). End with the recommendation the numbers support.

% ---------------------------------------------------------------------------
% 4. Supporting information legends (after the references).
% ---------------------------------------------------------------------------
\paragraph*{S9 Fig.}
\label{S9_Fig}
\textbf{Sensitivity to the subsampling proportion.} Generated by \texttt{figure\_ablation\_rho} from \texttt{notebooks/Resampling\_Ablation.ipynb}. (A, B) Rate at which the 1-SE rules select $k^\star$, per cluster shape (thin lines) and pooled (thick line, Wilson 95\% band). (C) Mean of $\widehat{k} - k^\star$. (D) ARI of the selected consensus labels to the truth, with the oracle at $k^\star$. (E) Recall of the smallest true cluster at the hard setting, at $\widehat{k}$ (solid) and at $k^\star$ (dotted). (F) Stability and generalizability at $k^\star$. (G) ARI between the clustering of a subsample and the full-data clustering restricted to it, at $k^\star$; the point at 1 refits the full data under a new seed. (H) Klein: share of ten fits selecting each (estimator, $k$) under the generalizability 1-SE rule. (I) Klein: ARI of the selected labels to the time points, and subsample-versus-full ARI at $k = 4$. The dotted vertical line marks the default $\rho = 0.618$.

\paragraph*{S10 Fig.}
\label{S10_Fig}
\textbf{Sensitivity to the resample count.} Generated by \texttt{figure\_ablation\_b}. (A, B) Fraction of pairs of fits of one data set that select the same $k$, per cluster shape and Klein, and pooled over the simulations. (C) Standard deviation of the score curves across fits, with a $1/\sqrt{B}$ guide anchored at the default. (D) Mean reported standard error at $k^\star$, with the same guide. (E) Rate of selecting $k^\star$. (F) Mean of $\widehat{k} - k^\star$. (G) Klein: ARI of the selected labels to the time points. (H) Klein: share of fits selecting each (estimator, $k$). $B$ is on a logarithmic axis; the dotted line marks the default $B = 100$.

\paragraph*{S11 Table.}
\label{S11_Table}
\textbf{Sensitivity of the selections.} For each value of $\rho$ (top) and $B$ (bottom) and each 1-SE rule: the pooled rate of selecting $k^\star$ with its Wilson 95\% interval, the pooled mean ARI of the selected labels, for $B$ the pooled agreement between fits, and the Klein selection chosen most often with its share of the ten fits. Generated by \texttt{write\_ablation\_table}.

% Numbering: S8 Text, S5 Fig and S11-S15 Table are reserved by the
% Cusanovich material (progress document, 2026-09-16). Renumber this
% material to follow whatever lands first.
```

- [ ] Step 2: Check the arithmetic in the drafts

Run:

```bash
.venv/bin/python -c "
r=(5**0.5-1)/2; print(round(r,3), round(r*r,3), round(1-r,3), round(100*r*r,1), round(100*0.2*0.2,1))"
```

Expected: `0.618 0.382 0.382 38.2 4.0`, matching the numbers in the drafts (0.38 n, about 38 co-samples, about 4 at rho = 0.2).

- [ ] Step 3: Record the file

This file is outside the git repository; nothing to commit. Note its path in the final summary (Task 14).

---

### Task 13: Dev run, notebook execution, published-result checks and the timing check

Files: none committed except the executed notebook's cleaned source (unchanged) and, if a check fails, the fix it needs.

These steps run on the author's machine from `code/`. They are the spec's section 10 verification list. The dev run takes about an hour; do not claim any of the checks passed without the command output in front of you.

- [ ] Step 1: Run the dev scale

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
caffeinate -i .venv/bin/python -m benchmarks.run --ablation rho_b --scale dev --verbose \
  > /tmp/ablation_dev.log 2>&1
```

This exceeds one Bash call's timeout, so start it with `run_in_background`, then wait for the process to exit (the harness re-invokes you when it does) before reading `/tmp/ablation_dev.log`. Do not end the turn with the run still going and report it finished.

Expected at the end of the log: `rho_b (dev): results/runs/ablation_rho_b/dev/<hash>`. Then:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from benchmarks._artifacts import read_frames
from benchmarks._registry import ABLATIONS
from benchmarks._ablation_cells import enumerate_cells, enumerate_units
rd = sorted(Path("results/runs/ablation_rho_b/dev").glob("*/manifest.json"))[-1].parent
frames = read_frames(rd)
print({k: len(v) for k, v in frames.items()})
print("cells", len(enumerate_cells(ABLATIONS["rho_b"], "dev")), "units", len(enumerate_units(ABLATIONS["rho_b"], "dev")))
print(frames["cells"].groupby("study")[["fit_seconds", "consensus_nan_fraction", "n_cluster_count_warnings"]].mean())
print(frames["cells"]["consensus_nan_fraction"].max())
print(frames["curves"]["metric_value"].isna().sum(), "NaN metric values")
PY
```

Expected: `cells 228` (204 simulated plus 24 Klein) and every frame non-empty; the NaN fraction is largest at rho = 0.2 and B = 10; zero NaN metric values (the consensus metrics come back finite, spec section 12).

- [ ] Step 2: Execute the notebook into scratch

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
  --output-dir=/tmp/ablation_nb notebooks/Resampling_Ablation.ipynb
ls vis/benchmarking/si_fig_ablation_rho.png vis/benchmarking/si_fig_ablation_b.png results/tables/si_table_ablation.tex
```

Expected: the executed copy lands in `/tmp/ablation_nb/`, both figures and the table exist. Open the two PNGs (Read tool) and check every panel has data and a legend entry; fix the figure module if a panel is blank. Do not execute the notebook in place: the committed notebook stays output-free until the publication run (spec section 8).

- [ ] Step 3: The published Klein selection at the defaults

```bash
.venv/bin/python - <<'PY'
from benchmarks._ablation import cell_rows
from benchmarks._ablation_cells import Cell, Unit, STUDY_DIFFICULTY, STUDY_DATASET
from benchmarks._registry import ABLATIONS
from benchmarks._studies import STUDIES, load_study
import numpy as np, pandas as pd
ab = ABLATIONS["rho_b"]
X, y, _ = load_study(STUDIES["klein"], scale="publication")
print(X.shape)  # (1358, 2000)
unit = Unit("cell", Cell("klein", STUDY_DIFFICULTY, STUDY_DATASET, ab.rho_default, ab.b_default, 0))
rows = cell_rows(unit, ablation=ab, scale="publication", data=(np.asarray(X), np.asarray(y)), thread_cap=None)
sel = pd.DataFrame(rows["selection"]).set_index("metric_name")
print(sel.loc["ari_generalizability_1se", ["selected_estimator", "selected_k", "ari_selected"]])
print(rows["cells"][0]["fit_seconds"], "s")
PY
```

Expected: `AgglomerativeClustering` at `selected_k == 4` for `ari_generalizability_1se` (the manuscript's Klein result, `_studies.py` comment and memory `carve-config-drift-pattern`), in about six minutes. If it is not, the Klein path differs from the notebook's (`fit_or_load_carve` with `random_state=42`, `n_jobs=-1`); compare the grids and seed before touching anything else.

- [ ] Step 4: Simulation cells at the defaults match the scenario runs

Only where a full-scale scenario run from the current code exists under `results/runs/<scenario>/` (the manifest's `n_seeds` is 20 and `n_resamples` 100; as of 2026-09-16 that is gaussians, t_dist, t_dist_noise and swiss_rolls). For one such scenario:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np, pandas as pd
from benchmarks._ablation import cell_rows
from benchmarks._ablation_cells import Cell, Unit
from benchmarks._artifacts import read_run
from benchmarks._registry import ABLATIONS, CARVE_METRICS_ALL, metric_measure
ab = ABLATIONS["rho_b"]
name, difficulty, dataset = "gaussians", "medium", 0
rds = [p.parent for p in Path(f"results/runs/{name}").glob("*/manifest.json")
       if json.loads(p.read_text())["n_seeds"] == 20 and json.loads(p.read_text())["n_resamples"] == 100]
assert rds, "no full-scale run"
published = read_run(rds[-1])
published = published[(published["axis_label"] == difficulty) & (published["seed"] == dataset)]
unit = Unit("cell", Cell(name, difficulty, dataset, ab.rho_default, ab.b_default, 0))
rows = cell_rows(unit, ablation=ab, scale="publication", data=None, thread_cap=None)
curves = pd.DataFrame(rows["curves"])
mismatches = 0
for metric in CARVE_METRICS_ALL:
    mine = curves[curves["metric_name"] == metric].set_index("k")["metric_value"]
    theirs = published[published["metric_name"] == metric].set_index("k")["metric_value"]
    if not np.allclose(mine.sort_index(), theirs.sort_index(), atol=1e-9):
        mismatches += 1; print("differs:", metric)
print("mismatching metrics:", mismatches)
PY
```

Expected: `mismatching metrics: 0`. A mismatch means the ablation's default cell is not the published fit: check `carve_seed` for replicate 0 against `benchmark_seed(dataset, axis_idx, 42)` and that `scenario_at_scale` at publication returns the scenario unchanged.

- [ ] Step 5: The timing check

Runs the fixed 22-cell batch at each candidate worker count into a scratch root. Each pass takes on the order of 15 minutes; run the five passes one after another with `run_in_background` and wait for each to exit.

```bash
for N in 5 6 7 8 -1; do
  caffeinate -i .venv/bin/python -m benchmarks.run --ablation rho_b --scale publication \
    --timing-batch --n-jobs $N --root /tmp/ablation_timing/nj$N --verbose \
    > /tmp/ablation_timing_nj$N.log 2>&1
done
.venv/bin/python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path("/tmp/ablation_timing").glob("nj*/ablation_rho_b/publication/*/manifest.json")):
    m = json.loads(path.read_text())
    print(f"n_jobs={m['n_jobs']:>3} workers={m['workers']:>2} cap={m['thread_cap']} wall={m['wall_clock_s']/60:.1f} min")
PY
```

Expected: one line per value; the smallest wall clock names the publication `--n-jobs`. Record the five numbers in the final summary. Re-derive the publication cost: from the dev manifests' `fit_seconds`, the per-fit time at n = 500 scales to n = 1,500 roughly linearly for KMeans and Ward and faster for spectral; the spec's estimate is 90 h of wall clock on 11 workers. State the new estimate.

- [ ] Step 6: Commit nothing from this task unless a check required a fix

If a check in Steps 1 to 4 required a code change, it was made with a test in the task that owns the file, and is committed there with its own message.

---

### Task 14: Full suite, lint, spec walk-through and handoff

- [ ] Step 1: Run both test legs in the foreground

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
.venv/bin/ruff check src/ && .venv/bin/ruff format --check src/
.venv/bin/pytest tests -q -x --ignore=tests/benchmarks
.venv/bin/pytest tests/benchmarks -q
```

Expected: ruff clean; every test passes; no new warnings (the suite runs with `filterwarnings = error`). Budget 20 to 25 minutes for the benchmarks leg.

- [ ] Step 2: Walk the spec

Open `docs/superpowers/specs/2026-09-16-rho-b-ablation-design.md` and confirm each section has landed: 4.1 grids and scales (Task 2), 4.2 seeds (Tasks 2, 5), 4.3 configuration (Task 6), 5 outcomes (Tasks 6, 8), 7.1 to 7.5 (Tasks 1 to 7), 8 figures, table and notebook (Tasks 9 to 11), 9 drafts (Task 12), 10 verification (Task 13), 13 out of scope (nothing under `src/carve/` changed: `git diff main --stat -- src/carve` is empty).

- [ ] Step 3: Update the open note

In `docs/superpowers/notes/2026-09-17-forest-thread-oversubscription.md`, change the `Status:` line to `Status: fixed in the rho-b-ablation branch (cpu_cap in benchmarks/_run.py, Task 3 of docs/superpowers/plans/2026-09-17-rho-b-ablation.md)`, and commit:

```bash
git add docs/superpowers/notes/2026-09-17-forest-thread-oversubscription.md
git commit -m "docs(notes): forest-thread oversubscription is fixed on this branch

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] Step 4: Hand off

Report to the author, in outcome terms:

- The branch, its commits, and that no published number moved (Task 3 Step 6 and Task 13 Step 4 outputs).
- The dev run's location and the Klein check's result.
- The timing check's five wall-clock numbers and the chosen `--n-jobs`, with the re-derived publication cost.
- The command for the publication run: `caffeinate -i .venv/bin/python -m benchmarks.run --ablation rho_b --scale publication --n-jobs <chosen> --verbose`, that it resumes if interrupted, and that after it finishes the notebook is executed with `SCALE = "publication"`, its outputs committed, and the S9 Text results written from them.
- The location of the manuscript drafts and the SI numbering note inside them.
- The two open decisions from the spec: run order against the Cusanovich 150-resample run, and whether to commit the publication notebook with outputs.

Do not start the publication run; the author does.

---

## Self-review notes

Spec coverage, checked against `2026-09-16-rho-b-ablation-design.md`:

- 4.1 cells, arms, scales: Tasks 2 and 5. 4.2 seeds: Tasks 2 and 5, with the sweep test in Task 5. 4.3 CARVE configuration per cell: Task 6 (`cell_rows`), thread cap Tasks 3 and 6.
- 5 outcomes: curves, selection, at_k, cells and datasets rows in Task 6; every derived quantity in Task 8; similarity with the rho = 1 reference in Tasks 5 and 6.
- 7.1 configuration and `Study.not_two`: Tasks 1 and 2. 7.2 shared fit code and cap: Task 3. 7.3 runner, order, checkpoints, hash, manifest, no saved fits: Task 6. 7.4 schemas and readers: Task 4. 7.5 CLI: Task 7.
- 8 figures, table, notebook and their tests: Tasks 9 to 11. 9 drafts: Task 12. 10 tests and verification: every bullet has a test in Tasks 1 to 11 or a step in Task 13. 11 cost re-derivation and 12 risks (NaN fraction, warnings counted): Task 13 Steps 1 and 5.

Names used across tasks were checked for consistency: `fit_carve`, `labels_by_mode`, `smallest_cluster`, `rare_cluster_recall`, `thread_cap_for`, `cpu_cap` (Task 3, used in Task 6); `CELL_KEY`, `ABLATION_SCHEMAS`, `write_frame`, `read_frames`, `ablation_dir`, `provenance`, `scenario_identity` (Task 4, used in Tasks 5, 6, 8); `Cell`, `Unit`, `enumerate_cells`, `enumerate_units`, `timing_units`, `unit_paths`, `unit_done`, `carve_seed`, `dataset_seed`, `similarity_seed`, `scenario_at_scale`, `arm_view`, `REFERENCE_RATIO` (Task 5, used in Tasks 6, 7, 9, 10); the summary functions (Task 8, used in Tasks 9, 10, 11).
