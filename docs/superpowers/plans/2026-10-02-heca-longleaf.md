# hECA on Longleaf Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: run one CARVE fit on all 695,304 annotated hECA cells on UNC's Longleaf cluster, as embed, calibrate and fit SLURM jobs, with the fit timing its own components, and read the copied-back run in `hECA.ipynb`.

Architecture: a staged batch entry point, `python -m benchmarks.heca {embed,calibrate,fit,status}`, writes everything into one run directory. Timing subclasses of `LeidenClustering` and `RandomForestClassifier` record per-call times from inside CARVE's loky workers into that directory. A report module and three figure functions read the directory on the laptop for the notebook.

Tech stack: Python 3.13, carve, scikit-learn, joblib/loky, igraph and leidenalg, psutil, scanpy (loader), pandas, matplotlib, SLURM.

Spec: `docs/superpowers/specs/2026-10-01-heca-longleaf-runtime-design.md` (commit fed189e). Executors read both.

## Global Constraints

- Branch: `heca-longleaf` in the `code/` repository. Commit after every task.
- Run every command from the repository root. Commands are written for the main checkout (`.venv/bin/python`). In a worktree, use the main checkout's interpreter and keep the `PYTHONPATH=src` prefix, so the worktree's sources shadow the editable install, which points at the main checkout's `src/`.
- Run tests in the foreground with a long timeout. `tests/benchmarks` alone takes about seven minutes.
- `filterwarnings = error` is on. A new warning fails the suite. A subsample clustered into one cluster warns (`_runner.py:815`), which is why every test grid below uses resolutions that split the synthetic data into at least two clusters.
- Lint and format `src/` only, with the pinned ruff: `.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/`. Never run ruff on `tests/` or `notebooks/`.
- No `logging`. Diagnostics go through `warnings.warn(..., stacklevel=2)` or `print`.
- Tests mirror modules one to one (`_heca_stages.py` and `tests/benchmarks/test_heca_stages.py`).
- One theme: colors come from `benchmarks._theme`; no module-level color literals.
- Documents and comments: American spelling, plain prose, no bold or italics.
- Do not write to `../overleaf/`. Do not touch `carve-r/` or `src/carve/`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- The notebook is committed without outputs.

## Refinements to the spec

Found while writing this plan; each is reflected in the tasks.

1. Module layout. The stages are split between `_heca_stages.py` (run directory, embed, fit, status) and `_heca_calibration.py`. The notebook-side reading lives in `_heca_report.py`, following `_cusanovich_compare.py`, not in `benchmarks.tables`, which writes .tex fragments.
2. Neighbor scaling. CARVE scales a neighbor count to each subset's size (`carve._utils.scale_neighbor_count`). In a resample, Leiden 15 runs with 9 neighbors on the training split and 6 on its complement; Leiden 50 runs with 31 and 19. Calibration builds its graphs with the same scaled counts, since the cluster count at a resolution depends on them.
3. Hyperthreading. The fit sets `LOKY_MAX_CPU_COUNT` to the physical core count. CARVE's core budget reads joblib's `cpu_count`, which honors it at call time (checked 2026-10-02). With one worker per physical core, each worker's forest and BLAS then run single-threaded.
4. `fit.sbatch` runs the fit with `srun --cpus-per-task="$SLURM_CPUS_ON_NODE"`, because srun does not inherit `--cpus-per-task` from sbatch. The step gives `sacct` a finished step to report.
5. Forest timing rows carry no setting; the classifier cannot see which configuration it serves. Forest time is reported for the whole fit, not per setting.
6. Calibration writes its scan and costs even when the grid rule fails, recording `grid_error`; the CLI then exits 1. A failed rule does not discard hours of scanning.
7. The notebook's reference UMAP embeds only the 50,000 cells it draws. The composite that reused a whole-data UMAP is removed.
8. `timing_directory` restarts joblib's reusable worker pool on entry and exit. Workers inherit the environment when they are spawned, and a pool started earlier would never see `CARVE_TIMING_DIR`.
9. The fit records its grid and resample count in `fit/started.json`, and the report rebuilds the study from it. The notebook finds the fit even after `STUDIES` changes.

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/benchmarks/_types.py` | Modify | `EstimatorSpec.params` |
| `src/benchmarks/_estimators.py` | Modify | `estimator_params`; grid builders apply spec params |
| `src/benchmarks/_studies.py` | Modify | hECA as a Leiden 15/50 resolution study; partners in resolution grids; `fit_or_load_carve(classifier=)`; `study_scaling_sweep` removed |
| `src/benchmarks/datasets/_heca.py` | Modify | `meta["obs"]` (cell_type, organ, study_id) cached and returned; deviation text |
| `src/benchmarks/_timing.py` | Create | Timing subclasses, graph timers, timing directory, reading rows |
| `src/benchmarks/_heca_stages.py` | Create | Run directory helpers, settings, environment record, embed, fit, status |
| `src/benchmarks/_heca_calibration.py` | Create | Resolution scan, grid rule, runtime projection, calibrate stage |
| `src/benchmarks/heca.py` | Create | CLI |
| `src/benchmarks/_heca_report.py` | Create | Load a run; runtime, component, selection and composition tables; sacct parsing |
| `src/benchmarks/figures/_heca_calibration.py` | Create | `figure_heca_calibration` |
| `src/benchmarks/figures/_heca_carve.py` | Create | `figure_heca_carve`; the reference scatter's constants |
| `src/benchmarks/figures/_heca_runtime.py` | Create | `figure_heca_runtime` |
| `src/benchmarks/figures/_heca_results.py`, `_study_scaling.py` | Delete | Dropped composite and ladder figures |
| `src/benchmarks/figures/__init__.py` | Modify | Exports |
| `notebooks/case_studies/hECA.ipynb` | Rewrite | Reads a run directory |
| `slurm/heca/{README.md,embed.sbatch,calibrate.sbatch,fit.sbatch}` | Create | Runbook and jobs |
| `src/benchmarks/README.md` | Modify | Points at the runbook |
| `pyproject.toml` | Modify | `psutil` in the `benchmarks` extra |
| `tests/benchmarks/_helpers.py` | Modify | `write_heca_organ`, `small_heca_study` |
| Tests | Create or modify | One file per module, listed per task |

---

### Task 1: EstimatorSpec fixed parameters

Files:
- Modify: `src/benchmarks/_types.py` (class `EstimatorSpec`)
- Modify: `src/benchmarks/_estimators.py` (`build_estimator`, `param_grids`, `resolution_grids`; new `estimator_params`)
- Test: `tests/benchmarks/test_benchmarks_types.py`, `tests/benchmarks/test_estimators.py`

Interfaces:
- Produces: `EstimatorSpec(name: str, params: tuple[tuple[str, Any], ...] = ())`. `estimator_params(spec: EstimatorSpec) -> dict[str, Any]`: the registered defaults overridden by `spec.params`. `param_grids`, `resolution_grids` and `build_estimator` apply it. A spec without params produces exactly the grids it did before, so no cache key or run hash moves; scenario hashes use `estimator.name` only (`_artifacts.scenario_identity`).

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_benchmarks_types.py` (`EstimatorSpec` is already imported there; add it to the import if not):

```python
class TestEstimatorSpecParams:
    def test_params_default_to_none_fixed(self):
        assert EstimatorSpec(name="leiden").params == ()

    def test_a_spec_with_params_is_hashable_and_compares_by_value(self):
        a = EstimatorSpec(name="leiden", params=(("n_neighbors", 50),))
        b = EstimatorSpec(name="leiden", params=(("n_neighbors", 50),))
        assert a == b
        assert hash(a) == hash(b)
        assert a != EstimatorSpec(name="leiden")

    def test_setting_a_parameter_twice_raises(self):
        with pytest.raises(ValueError, match="more than once"):
            EstimatorSpec(
                name="leiden", params=(("n_neighbors", 15), ("n_neighbors", 50))
            )
```

Append to `tests/benchmarks/test_estimators.py`. Add `build_estimator`, `param_grids` and `resolution_grids` to its existing `from benchmarks._estimators import (...)` block if any is missing, and add `from carve.cluster import LeidenClustering`.

```python
class TestFixedParams:
    def test_params_override_the_registered_default(self):
        spec = EstimatorSpec(name="leiden", params=(("n_neighbors", 50),))
        ((cls, grid),) = resolution_grids(spec, (0.5, 1.0))
        assert cls is LeidenClustering
        assert grid == {
            "resolution": [0.5, 1.0],
            "n_neighbors": [50],
            "objective_function": ["modularity"],
        }

    def test_a_spec_without_params_builds_the_grid_it_always_did(self):
        ((_, grid),) = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        assert grid == {"n_clusters": [2, 3], "n_init": [10]}

    def test_params_reach_a_k_based_grid_and_build_estimator(self):
        spec = EstimatorSpec(name="kmeans", params=(("n_init", 3),))
        ((_, grid),) = param_grids(spec, (2,))
        assert grid["n_init"] == [3]
        assert build_estimator(spec, 4, random_state=0).n_init == 3

    def test_an_unknown_parameter_raises(self):
        spec = EstimatorSpec(name="leiden", params=(("n_neighbours", 50),))
        with pytest.raises(ValueError, match="no parameter 'n_neighbours'"):
            resolution_grids(spec, (1.0,))

    def test_the_swept_parameter_cannot_be_fixed(self):
        spec = EstimatorSpec(name="leiden", params=(("resolution", 1.0),))
        with pytest.raises(ValueError, match="which the sweep sets"):
            resolution_grids(spec, (1.0,))
```

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_benchmarks_types.py tests/benchmarks/test_estimators.py -q`
Expected: the new tests FAIL with `TypeError: EstimatorSpec.__init__() got an unexpected keyword argument 'params'`.

- [ ] Step 3: Implement

In `src/benchmarks/_types.py`, replace the class `EstimatorSpec` with:

```python
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
                f"EstimatorSpec {self.name!r} sets a parameter more than once: "
                f"{names}."
            )
```

In `src/benchmarks/_estimators.py`, add after `apply_random_state`:

```python
#: The parameters a grid builder sweeps. A spec may not fix them: the sweep
#: would overwrite the value without a word.
_SWEPT_PARAMETERS: frozenset[str] = frozenset({"n_clusters", "resolution"})


def estimator_params(spec: EstimatorSpec) -> dict[str, Any]:
    """The registered defaults for spec's estimator, overridden by spec.params.

    Each name in spec.params must be a constructor parameter of the
    estimator class, and none may be a swept parameter, which the grid
    builders set themselves.
    """
    estimator_cls = ESTIMATOR_CLASSES[spec.name]
    accepted = inspect.signature(estimator_cls.__init__).parameters
    params: dict[str, Any] = dict(ESTIMATOR_DEFAULTS[spec.name])
    for key, value in spec.params:
        if key in _SWEPT_PARAMETERS:
            raise ValueError(
                f"EstimatorSpec {spec.name!r} fixes {key!r}, which the sweep sets."
            )
        if key not in accepted:
            raise ValueError(
                f"{estimator_cls.__name__} has no parameter {key!r} "
                f"(EstimatorSpec {spec.name!r})."
            )
        params[key] = value
    return params
```

In `build_estimator`, replace `params: dict[str, Any] = dict(ESTIMATOR_DEFAULTS[spec.name])` with `params: dict[str, Any] = estimator_params(spec)`. In both `param_grids` and `resolution_grids`, replace `for key, value in ESTIMATOR_DEFAULTS[spec.name].items():` with `for key, value in estimator_params(spec).items():`.

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_benchmarks_types.py tests/benchmarks/test_estimators.py tests/benchmarks/test_ablation.py tests/benchmarks/test_artifacts.py -q`
Expected: all PASS. The ablation and artifact tests pin hashes that must not move.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add src/benchmarks/_types.py src/benchmarks/_estimators.py tests/benchmarks/test_benchmarks_types.py tests/benchmarks/test_estimators.py
git commit -m "feat(benchmarks): let an EstimatorSpec fix its estimator's parameters

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: hECA as a Leiden resolution study

Files:
- Modify: `src/benchmarks/_studies.py` (`STUDIES["heca"]`, `study_resolution_grids`, `fit_or_load_carve`, `_check_dense_fit` docstring; delete `study_scaling_sweep`)
- Modify: `tests/benchmarks/test_studies.py`
- Modify: `tests/benchmarks/_helpers.py` (`make_carve_spy` docstring)

Interfaces:
- Consumes: `EstimatorSpec(name, params)` from Task 1.
- Produces: `STUDIES["heca"]`: estimator Leiden with `n_neighbors=15`, one partner Leiden with `n_neighbors=50`, `candidate_k=()`, 15 provisional resolutions from 0.005 to 3.0, `n_resamples=100`, `consensus_anchors=2000`. `study_resolution_grids(study)` returns the estimator's grid then each partner's, and raises for a study whose estimator does not take a resolution. `fit_or_load_carve(..., classifier: ClassifierMixin | None = None)` forwards a non-None classifier to `CARVE(...)`.

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/test_studies.py`:

Add `from sklearn.ensemble import RandomForestClassifier` to the imports.

In `class TestNewStudies`, delete `test_heca_sweeps_four_through_fifteen`, `test_heca_uses_estimators_that_can_run_at_scale` and `test_heca_declares_a_resolution_sweep`, and add:

```python
    def test_heca_sweeps_leiden_resolution_at_two_neighbor_counts(self):
        study = STUDIES["heca"]
        assert study.candidate_k == ()
        grids = study_resolution_grids(study)
        assert [cls for cls, _ in grids] == [LeidenClustering, LeidenClustering]
        assert [grid["n_neighbors"] for _, grid in grids] == [[15], [50]]
        assert all(grid["objective_function"] == ["modularity"] for _, grid in grids)

    def test_heca_provisional_grid_spans_five_thousandths_to_three(self):
        resolutions = STUDIES["heca"].resolutions
        assert len(resolutions) == 15
        assert resolutions[0] == pytest.approx(0.005)
        assert resolutions[-1] == pytest.approx(3.0)
        assert list(resolutions) == sorted(resolutions)

    def test_heca_runs_one_hundred_resamples(self):
        assert STUDIES["heca"].n_resamples == 100

    def test_heca_has_no_k_based_grid(self):
        with pytest.raises(ValueError, match="study_resolution_grids"):
            study_model_grids(STUDIES["heca"])

    def test_resolution_partners_join_the_sweep(self):
        study = _study(
            estimator=EstimatorSpec(name="leiden", params=(("n_neighbors", 15),)),
            partners=(EstimatorSpec(name="louvain"),),
            candidate_k=(),
            resolutions=(0.5, 1.0),
        )
        grids = study_resolution_grids(study)
        assert [cls for cls, _ in grids] == [LeidenClustering, LouvainClustering]
        assert all(grid["resolution"] == [0.5, 1.0] for _, grid in grids)

    def test_a_k_based_study_has_no_resolution_sweep(self):
        with pytest.raises(ValueError, match="does not take a resolution"):
            study_resolution_grids(_study(resolutions=(0.5, 1.0)))
```

In the test that contains `heca = carve_cache_path(STUDIES["heca"], root=tmp_path)`, replace this block:

```python
        # The Levine cache is hours of compute and was written before run
        # keys existed, so its default run must resolve to the same name as
        # before, byte for byte. Klein's and hECA's publication scales are
        # every cell, so their names carry that size's hash: neither Klein's
        # half-subsample cache (1b390cd5) nor hECA's 25,000-cell development
        # cache (dev_8314e95d) is ever served for them.
        klein = carve_cache_path(STUDIES["klein"], root=tmp_path)
        levine = carve_cache_path(STUDIES["levine32"], root=tmp_path)
        heca = carve_cache_path(STUDIES["heca"], root=tmp_path)
        assert klein.name == "carve_klein_publication_6eef6648.carve"
        assert levine.name == "carve_levine32_publication_f8237d89.carve"
        assert heca.name == "carve_heca_publication_6eef6648.carve"
```

with:

```python
        # The Levine cache is hours of compute and was written before run
        # keys existed, so its default run must resolve to the same name as
        # before, byte for byte. Klein's publication scale is every cell, so
        # its name carries that size's hash: Klein's half-subsample cache
        # (1b390cd5) is never served for it.
        klein = carve_cache_path(STUDIES["klein"], root=tmp_path)
        levine = carve_cache_path(STUDIES["levine32"], root=tmp_path)
        assert klein.name == "carve_klein_publication_6eef6648.carve"
        assert levine.name == "carve_levine32_publication_f8237d89.carve"

    def test_heca_cache_name_carries_a_resolution_run_key(self, tmp_path):
        # hECA has no k-based grid, so its fit names its grids, and the file
        # name carries a run key after the every-cell size hash.
        study = STUDIES["heca"]
        path = carve_cache_path(
            study,
            root=tmp_path,
            model_grids=study_resolution_grids(study),
            n_resamples=study.n_resamples,
        )
        assert path.name.startswith("carve_heca_publication_6eef6648_")
        assert path.suffix == ".carve"
```

Next to `test_fit_or_load_carve_forwards_consensus_anchors`, in the same class, add:

```python
    def test_fit_or_load_carve_forwards_a_classifier(
        self, blobs, tmp_path, monkeypatch
    ):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)
        classifier = RandomForestClassifier(n_estimators=5)

        fit_or_load_carve(
            X,
            y,
            cache_path=tmp_path / "demo.carve",
            model_grids=grids,
            n_resamples=3,
            classifier=classifier,
        )

        assert spy.captured_kwargs["classifier"] is classifier

    def test_fit_or_load_carve_omits_the_classifier_when_none(
        self, blobs, tmp_path, monkeypatch
    ):
        X, y = blobs
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        spy = make_carve_spy()
        monkeypatch.setattr("benchmarks._studies.CARVE", spy)

        fit_or_load_carve(
            X, y, cache_path=tmp_path / "demo.carve", model_grids=grids, n_resamples=3
        )

        assert "classifier" not in spy.captured_kwargs
```

Replace `test_heca_pins_its_anchor_count` with:

```python
    def test_heca_pins_its_anchor_count(self):
        # 30 configurations each retain a stability and a generalizability
        # consensus block of 2000 x 2000 anchors, about 1 GB in all.
        assert STUDIES["heca"].consensus_anchors == 2000
```

Remove everything that exercised the ladder: `study_scaling_sweep` from the `from benchmarks._studies import (...)` block, `test_study_scaling_sweep_forwards_consensus_anchors`, `test_study_scaling_sweep_rejects_spectral_at_a_large_rung`, the `ladder_data` fixture and the whole `class TestStudyScalingSweep`. In the docstring that reads `cvi_sweep, fit_or_load_carve, and study_scaling_sweep all bring X and`, change that phrase to `cvi_sweep and fit_or_load_carve both bring X and`.

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_studies.py -q`
Expected: FAIL. The hECA tests fail on the old configuration, `test_fit_or_load_carve_forwards_a_classifier` fails with `TypeError: ... unexpected keyword argument 'classifier'`, and `test_a_k_based_study_has_no_resolution_sweep` fails because the old code falls back to Leiden.

- [ ] Step 3: Implement

In `src/benchmarks/_studies.py`:

Add `ClassifierMixin` to the sklearn import: `from sklearn.base import ClassifierMixin, ClusterMixin`.

Replace the `"heca": Study(...)` entry with:

```python
    "heca": Study(
        name="heca",
        loader=_heca_loader,
        # Leiden on two graphs: 15 neighbors, the scanpy convention hECA's
        # own EpiScanpy annotation inherits, and 50, the graph CATlas (Zhang
        # 2021) and Hocker 2021 clustered; those two supply 221,029 of the
        # pooled cells. Both optimize modularity. See the 2026-10-01 spec.
        estimator=EstimatorSpec(name="leiden", params=(("n_neighbors", 15),)),
        partners=(EstimatorSpec(name="leiden", params=(("n_neighbors", 50),)),),
        candidate_k=(),
        scales={"publication": None},
        default_scale="publication",
        # Provisional: 15 log-spaced values from 0.005 to 3.0. The single
        # studies the cells come from reported 1.0 and 1.5 on 8,500 to 91,500
        # cells; each resample here clusters about 430,000, where modularity
        # splits further at the same value, so organ level needs far smaller
        # ones. python -m benchmarks.heca calibrate proposes the grid that
        # replaces this one before the fit runs.
        resolutions=tuple(float(f"{0.005 * 600 ** (i / 14):.2g}") for i in range(15)),
        # 30 configurations each retain a stability and a generalizability
        # consensus block of 2000 x 2000 anchors, about 1 GB in all.
        consensus_anchors=2000,
        n_resamples=100,
    ),
```

Replace `study_resolution_grids` with:

```python
def study_resolution_grids(
    study: Study,
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """The study's resolution sweep: its own estimator, then every partner.

    Cusanovich sweeps Louvain alone; hECA sweeps Leiden at two neighbor
    counts, declared as its estimator and one partner. A separate CARVE run
    from study_model_grids: SweepSpec is frozen, so a k-based and a
    resolution-based sweep cannot share one run.
    """
    if not study.resolutions:
        raise ValueError(f"Study {study.name!r} declares no resolutions.")
    if study.estimator.name not in RESOLUTION_ESTIMATORS:
        raise ValueError(
            f"Study {study.name!r} sweeps {study.estimator.name!r}, which does "
            "not take a resolution; use study_model_grids."
        )
    grids = resolution_grids(study.estimator, study.resolutions)
    for partner in study.partners:
        grids += resolution_grids(partner, study.resolutions)
    return grids
```

In `fit_or_load_carve`, add the parameter `classifier: ClassifierMixin | None = None,` after `dim_reduction_options`, document it in the docstring:

```
    classifier : sklearn classifier or None, default=None
        Forwarded to CARVE only when not None. hECA's fit passes a forest
        with the default's settings that times itself (see _timing).
```

and add `("classifier", classifier),` to the tuple the `optional` dict is built from.

Delete the function `study_scaling_sweep`. In `_check_dense_fit`'s docstring, change `cvi_sweep, fit_or_load_carve, and study_scaling_sweep are the one place` to `cvi_sweep and fit_or_load_carve are the one place`. In `tests/benchmarks/_helpers.py`, change `make_carve_spy`'s docstring phrase `for fit_or_load_carve and study_scaling_sweep to run to completion` to `for fit_or_load_carve to run to completion`.

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_studies.py tests/benchmarks/test_notebooks.py -q`
Expected: all PASS. The hECA notebook still holds the strings `test_notebooks` pins until Task 12.

Run: `grep -rn "study_scaling_sweep" src tests`
Expected: no output.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add src/benchmarks/_studies.py tests/benchmarks/test_studies.py tests/benchmarks/_helpers.py
git commit -m "feat(benchmarks): sweep hECA as Leiden resolution on 15 and 50 neighbors

The scaling ladder and the k-based hECA fit are dropped; one resolution
fit on every cell answers R1.1. fit_or_load_carve forwards a classifier.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The loader returns organ, cell type and study per row

Files:
- Modify: `src/benchmarks/datasets/_heca.py` (`META_OBS_COLUMNS`, `_read_cache`, `_write_cache`, `load_heca`, deviation text)
- Modify: `tests/benchmarks/_helpers.py` (add `HECA_N_PEAKS`, `write_heca_organ`)
- Modify: `tests/benchmarks/datasets/test_heca.py`

Interfaces:
- Produces: `load_heca(...)` returns `meta["obs"]`, a DataFrame with string columns `cell_type`, `organ`, `study_id`, aligned to the rows of X (subsampled with X). Caches carry these columns as arrays `obs_cell_type`, `obs_organ`, `obs_study_id`; a cache without them is a miss. `tests.benchmarks._helpers.write_heca_organ(path, name, n_cells, seed, *, study_id="10.1000/x")` and `HECA_N_PEAKS = 300`.

- [ ] Step 1: Move the synthetic organ writer into the shared helpers

In `tests/benchmarks/_helpers.py`, append:

```python
HECA_N_PEAKS = 300


def write_heca_organ(path, name, n_cells, seed, *, study_id="10.1000/x"):
    """Write one synthetic hECA organ file in the real layout.

    CSR raw counts, cells by cPeaks, annotations in obs, an empty obsm. An
    organ-specific block plus a shared background gives the embedding
    structure that clustering can recover. The random draws are made in the
    order the loader tests were written against, so their numbers do not
    move.
    """
    import anndata as ad
    from scipy import sparse

    rng = np.random.default_rng(seed)
    types = rng.choice(["T cell", "Epithelial cell", "Unclassified"], n_cells)
    M = np.zeros((n_cells, HECA_N_PEAKS))
    offset = {"Lung": 0, "Brain": 100}.get(name, 200)
    M[:, offset : offset + 100] = rng.random((n_cells, 100)) < 0.5
    M[:, 250:] = rng.random((n_cells, 50)) < 0.4

    adata = ad.AnnData(
        X=sparse.csr_matrix(M.astype(np.int32)),
        obs=pd.DataFrame(
            {
                "cell_type": types,
                "organ": name,
                "donor_id": rng.choice(["d1", "d2"], n_cells),
                "study_id": study_id,
            },
            index=[f"{name}_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"peak{i}" for i in range(HECA_N_PEAKS)]),
    )
    adata.write_h5ad(path / f"ATAC-{name}.h5ad")
```

In `tests/benchmarks/datasets/test_heca.py`, delete `N_PEAKS = 300` and the function `_organ`, and add to the imports:

```python
from tests.benchmarks._helpers import HECA_N_PEAKS as N_PEAKS
from tests.benchmarks._helpers import write_heca_organ as _organ
```

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/datasets/test_heca.py -q`
Expected: all PASS (pure move).

- [ ] Step 2: Write the failing tests

Append to `tests/benchmarks/datasets/test_heca.py`:

```python
class TestMetaObs:
    def test_obs_is_aligned_with_the_rows(self, heca):
        X, y, meta = load_heca(
            root=heca, organs=["Lung", "Brain"], n_top_peaks=50, n_components=5
        )
        obs = meta["obs"]
        assert list(obs.columns) == ["cell_type", "organ", "study_id"]
        assert len(obs) == X.shape[0]
        assert (obs["organ"].to_numpy() == y.to_numpy()).all()
        assert "Unclassified" not in set(obs["cell_type"])

    def test_obs_survives_the_cache(self, heca):
        _, _, first = load_heca(
            root=heca, organs=["Lung", "Brain"], n_top_peaks=50, n_components=5
        )
        for path in (heca / "hECA").glob("ATAC-*.h5ad"):
            path.unlink()
        _, _, second = load_heca(
            root=heca, organs=["Lung", "Brain"], n_top_peaks=50, n_components=5
        )
        assert second["cached"]
        pd.testing.assert_frame_equal(first["obs"], second["obs"])

    def test_obs_follows_the_subsample(self, heca):
        _, y, meta = load_heca(
            root=heca,
            organs=["Lung", "Brain"],
            subsample=40,
            n_top_peaks=50,
            n_components=5,
        )
        assert len(meta["obs"]) == 40
        assert (meta["obs"]["organ"].to_numpy() == y.to_numpy()).all()

    def test_a_cache_without_obs_is_rebuilt(self, heca):
        load_heca(root=heca, organs=["Lung", "Brain"], n_top_peaks=50, n_components=5)
        (cache,) = (heca / "hECA").glob(".heca_cache_*.npz")
        with np.load(cache, allow_pickle=True) as payload:
            old = {k: payload[k] for k in payload.files if not k.startswith("obs_")}
        np.savez_compressed(cache, **old)

        _, _, meta = load_heca(
            root=heca, organs=["Lung", "Brain"], n_top_peaks=50, n_components=5
        )
        assert not meta["cached"]
        assert list(meta["obs"].columns) == ["cell_type", "organ", "study_id"]

    def test_deviation_gives_the_sources_reason_not_memory(self, heca):
        _, _, meta = load_heca(
            root=heca, organs=["Lung"], n_top_peaks=50, n_components=5
        )
        text = " ".join(meta["deviations"])
        assert "PeakVI" in text
        assert "for memory" not in text
```

- [ ] Step 3: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/datasets/test_heca.py::TestMetaObs -q`
Expected: FAIL with `KeyError: 'obs'`.

- [ ] Step 4: Implement

In `src/benchmarks/datasets/_heca.py`, after `_OBS_COLUMNS`, add:

```python
#: obs columns load_heca returns in meta["obs"], aligned to the rows of X, and
#: stores in its cache: both reference labels and the study of origin, which
#: a pooled clustering may partly recover.
META_OBS_COLUMNS: tuple[str, ...] = ("cell_type", "organ", "study_id")
```

Replace `_read_cache` and `_write_cache` with:

```python
def _read_cache(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as payload:
        # A cache written before meta["obs"] existed lacks these arrays. It
        # is treated as a miss and rebuilt, not served without them.
        if any(f"obs_{column}" not in payload.files for column in META_OBS_COLUMNS):
            return None
        return {
            "X": payload["X"],
            "y": payload["y"],
            "obs": pd.DataFrame(
                {
                    column: payload[f"obs_{column}"].astype(str)
                    for column in META_OBS_COLUMNS
                }
            ),
            **{key: int(payload[key]) for key in _CACHE_SCALARS},
        }


def _write_cache(
    path: Path, X: np.ndarray, y: np.ndarray, obs: pd.DataFrame, **scalars: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=np.asarray(y).astype(str),
        **{
            f"obs_{column}": obs[column].to_numpy().astype(str)
            for column in META_OBS_COLUMNS
        },
        **scalars,
    )
```

In `load_heca`:

- In the `if cached is not None:` branch, add `obs_full = cached["obs"]`.
- In the `else:` branch, after `y_full = pd.Series(...)`, add `obs_full = obs[list(META_OBS_COLUMNS)].astype(str).reset_index(drop=True)`, and change the cache write to `_write_cache(cache_path, X_full, y_full.to_numpy(), obs_full, **scalars)`.
- After `y = y_full`, add `obs_rows = obs_full`. Inside the pooled subsample block, after `y = y.iloc[idx].reset_index(drop=True)`, add `obs_rows = obs_full.iloc[idx].reset_index(drop=True)`.
- Add `"obs": obs_rows,` to `meta`.
- Replace the `"deviations"` entry with:

```python
        "deviations": [
            "deviation from source: the publication selects 500,000 highly "
            "variable cPeaks for its PeakVI visualization, and its annotation "
            "selected peaks per dataset with no fixed count; "
            f"{n_top} are selected here, which keeps the embedding cheap",
        ],
```

In the `n_top_peaks` docstring entry, change `The source uses 500,000; see the recorded deviation.` to `The source's PeakVI visualization uses 500,000; see the recorded deviation.`

- [ ] Step 5: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/datasets/test_heca.py tests/benchmarks/test_studies.py -q`
Expected: all PASS.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add src/benchmarks/datasets/_heca.py tests/benchmarks/_helpers.py tests/benchmarks/datasets/test_heca.py
git commit -m "feat(datasets): return hECA's organ, cell type and study per row

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Timing instrumentation

Files:
- Create: `src/benchmarks/_timing.py`
- Test: `tests/benchmarks/test_timing.py`

Interfaces:
- Produces:
  - `TIMING_DIR_ENV = "CARVE_TIMING_DIR"`.
  - `LEIDEN_COLUMNS`, `FOREST_COLUMNS` (tuples of column names).
  - `GraphTimes(knn_s: float, build_s: float)`.
  - `graph_timers() -> ContextManager[GraphTimes]`.
  - `LeidenClustering` (subclass of `carve.cluster.LeidenClustering`, same `__name__`).
  - `TimedRandomForestClassifier`.
  - `timed_forest(*, n_features: int, n_trees: int) -> TimedRandomForestClassifier`.
  - `instrumented_grids(grids) -> grids`.
  - `timing_directory(path) -> ContextManager[Path]`.
  - `read_timings(directory) -> dict[str, pd.DataFrame]` with keys `"leiden"` and `"forest"`.
  - Leiden rows: pid, host, unix_time, n_neighbors (the scaled count CARVE fitted with), resolution, n_samples, n_clusters, knn_s, graph_s, leiden_s, fit_s. Forest rows: pid, host, unix_time, call (`"fit"` or `"predict"`), n_samples, seconds.

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/test_timing.py`:

```python
"""Tests for the timing instrumentation."""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

import carve.cluster as cluster
from benchmarks._estimators import resolution_grids
from benchmarks._timing import (
    TIMING_DIR_ENV,
    LeidenClustering,
    TimedRandomForestClassifier,
    graph_timers,
    instrumented_grids,
    read_timings,
    timed_forest,
    timing_directory,
)
from benchmarks._types import EstimatorSpec
from carve import CARVE
from carve._utils import default_generalizability_classifier


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    labels = rng.integers(0, 3, 240)
    return centers[labels] + rng.normal(0, 0.6, (240, 2))


def _grids():
    spec = EstimatorSpec(name="leiden", params=(("n_neighbors", 10),))
    return resolution_grids(spec, (0.5, 1.0))


def _fit(X, grids, classifier, *, n_jobs):
    return CARVE(
        estimator_param_grids=grids,
        n_resamples=4,
        n_jobs=n_jobs,
        random_state=0,
        classifier=classifier,
    ).fit(X)


def test_the_subclass_keeps_its_parents_name():
    # CARVE labels configurations by the class's __name__.
    assert LeidenClustering.__name__ == cluster.LeidenClustering.__name__
    assert issubclass(LeidenClustering, cluster.LeidenClustering)


def test_without_the_variable_the_subclass_is_its_parent(blobs, monkeypatch):
    monkeypatch.delenv(TIMING_DIR_ENV, raising=False)
    timed = LeidenClustering(n_neighbors=10, resolution=1.0, random_state=0)
    plain = cluster.LeidenClustering(n_neighbors=10, resolution=1.0, random_state=0)
    assert np.array_equal(timed.fit(blobs).labels_, plain.fit(blobs).labels_)


def test_a_leiden_fit_writes_one_row_whose_parts_sum_to_the_fit(tmp_path, blobs):
    with timing_directory(tmp_path):
        estimator = LeidenClustering(n_neighbors=10, resolution=1.0, random_state=0)
        estimator.fit(blobs)
    leiden = read_timings(tmp_path)["leiden"]
    assert len(leiden) == 1
    row = leiden.iloc[0]
    assert row["n_neighbors"] == 10
    assert row["n_samples"] == 240
    assert row["n_clusters"] == estimator.n_clusters_
    assert row["knn_s"] > 0
    assert row["graph_s"] > 0
    assert row["leiden_s"] > 0
    assert row["knn_s"] + row["graph_s"] + row["leiden_s"] == pytest.approx(
        row["fit_s"]
    )


def test_graph_timers_time_both_parts_and_restore_carve_cluster(blobs):
    search, build = cluster.kneighbors_graph, cluster.build_knn_graph
    with graph_timers() as times:
        cluster.build_knn_graph(blobs, n_neighbors=5)
    assert cluster.kneighbors_graph is search
    assert cluster.build_knn_graph is build
    assert 0 < times.knn_s < times.build_s


def test_timing_directory_sets_and_restores_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(TIMING_DIR_ENV, "before")
    with timing_directory(tmp_path / "timings") as path:
        assert os.environ[TIMING_DIR_ENV] == str(path)
        assert path.is_dir()
    assert os.environ[TIMING_DIR_ENV] == "before"


def test_timed_forest_carries_the_default_forests_parameters():
    plain = default_generalizability_classifier(
        classifier=None, n_features=50, n_trees=100, random_state=None, n_jobs=1
    )
    timed = timed_forest(n_features=50, n_trees=100)
    assert isinstance(timed, TimedRandomForestClassifier)
    assert timed.get_params() == plain.get_params()


def test_instrumented_grids_swap_only_the_leiden_class():
    grids = _grids() + [(KMeans, {"n_clusters": [2]})]
    swapped = instrumented_grids(grids)
    assert swapped[0][0] is LeidenClustering
    assert swapped[1][0] is KMeans
    assert swapped[0][1] == grids[0][1]


def test_an_instrumented_fit_matches_a_plain_one_and_times_inside_workers(
    tmp_path, blobs
):
    # The plain fit runs first and leaves joblib's worker pool alive without
    # the variable; timing_directory must replace that pool for any worker
    # row to appear.
    plain = _fit(blobs, _grids(), None, n_jobs=2)
    with timing_directory(tmp_path):
        timed = _fit(
            blobs,
            instrumented_grids(_grids()),
            timed_forest(n_features=blobs.shape[1], n_trees=100),
            n_jobs=2,
        )
    pd.testing.assert_frame_equal(plain.estimator_results_, timed.estimator_results_)

    tables = read_timings(tmp_path)
    leiden, forest = tables["leiden"], tables["forest"]
    # Two resolutions x four resamples x three subsets (both training
    # splits and the complement).
    assert len(leiden) == 2 * 4 * 3
    assert int((forest["call"] == "fit").sum()) == 2 * 4
    assert set(leiden["pid"]) - {os.getpid()}, "no row came from a worker"


def test_read_timings_of_an_empty_directory_gives_empty_frames(tmp_path):
    tables = read_timings(tmp_path)
    assert tables["leiden"].empty
    assert tables["forest"].empty
```

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_timing.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._timing'`.

- [ ] Step 3: Implement

Create `src/benchmarks/_timing.py`:

```python
"""Timing instrumentation for a CARVE fit's components.

CARVE runs each resample in a loky worker process, where a profiler in the
parent cannot see. The classes here time themselves inside the worker and
append one CSV row per call to the directory named by CARVE_TIMING_DIR, which
loky workers inherit when they are spawned. With the variable unset they
behave exactly as their parents and write nothing.

LeidenClustering keeps its parent's class name, because CARVE labels each
configuration in estimator_results_ by the estimator class's __name__; an
instrumented run's tables then read exactly as an uninstrumented run's.
"""

import csv
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import ClusterMixin
from sklearn.ensemble import RandomForestClassifier

import carve.cluster as _cluster

TIMING_DIR_ENV = "CARVE_TIMING_DIR"

LEIDEN_COLUMNS: tuple[str, ...] = (
    "pid",
    "host",
    "unix_time",
    "n_neighbors",
    "resolution",
    "n_samples",
    "n_clusters",
    "knn_s",
    "graph_s",
    "leiden_s",
    "fit_s",
)
FOREST_COLUMNS: tuple[str, ...] = (
    "pid",
    "host",
    "unix_time",
    "call",
    "n_samples",
    "seconds",
)


@dataclass
class GraphTimes:
    """Seconds spent building neighbor graphs, accumulated across calls.

    knn_s is the neighbor search alone; build_s is all of build_knn_graph,
    the search included, so graph assembly is build_s - knn_s.
    """

    knn_s: float = 0.0
    build_s: float = 0.0


@contextmanager
def graph_timers() -> Iterator[GraphTimes]:
    """Time carve.cluster's neighbor search and graph construction.

    Replaces the two module attributes LeidenClustering.fit reaches through
    for the duration of the block, so the fit itself is not reimplemented.
    The replacement is process-local, and a loky worker runs one task at a
    time.
    """
    times = GraphTimes()
    search = _cluster.kneighbors_graph
    build = _cluster.build_knn_graph

    def timed_search(*args: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return search(*args, **kwargs)
        finally:
            times.knn_s += time.perf_counter() - started

    def timed_build(*args: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return build(*args, **kwargs)
        finally:
            times.build_s += time.perf_counter() - started

    _cluster.kneighbors_graph = timed_search
    _cluster.build_knn_graph = timed_build
    try:
        yield times
    finally:
        _cluster.kneighbors_graph = search
        _cluster.build_knn_graph = build


def _append_row(kind: str, row: dict[str, Any], columns: tuple[str, ...]) -> None:
    """Append one row to this process's file of the given kind."""
    directory = Path(os.environ[TIMING_DIR_ENV])
    host, pid = socket.gethostname(), os.getpid()
    path = directory / f"{kind}-{host}-{pid}.csv"
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow({"pid": pid, "host": host, "unix_time": time.time(), **row})


class LeidenClustering(_cluster.LeidenClustering):
    """carve.cluster.LeidenClustering that times its graph build and Leiden."""

    def fit(self, X, y=None):
        if not os.environ.get(TIMING_DIR_ENV):
            return super().fit(X, y)
        started = time.perf_counter()
        with graph_timers() as times:
            super().fit(X, y)
        total = time.perf_counter() - started
        _append_row(
            "leiden",
            {
                "n_neighbors": int(self.n_neighbors),
                "resolution": float(self.resolution),
                "n_samples": int(len(X)),
                "n_clusters": int(self.n_clusters_),
                "knn_s": times.knn_s,
                "graph_s": times.build_s - times.knn_s,
                "leiden_s": total - times.build_s,
                "fit_s": total,
            },
            LEIDEN_COLUMNS,
        )
        return self


class TimedRandomForestClassifier(RandomForestClassifier):
    """RandomForestClassifier that times fit and predict.

    Defines no __init__, so get_params and clone see the parent's
    parameters, and CARVE clones it and sets random_state and n_jobs as it
    does for any user classifier.
    """

    def fit(self, X, y, sample_weight=None):
        if not os.environ.get(TIMING_DIR_ENV):
            return super().fit(X, y, sample_weight=sample_weight)
        started = time.perf_counter()
        super().fit(X, y, sample_weight=sample_weight)
        seconds = time.perf_counter() - started
        _append_row(
            "forest",
            {"call": "fit", "n_samples": int(len(X)), "seconds": seconds},
            FOREST_COLUMNS,
        )
        return self

    def predict(self, X):
        if not os.environ.get(TIMING_DIR_ENV):
            return super().predict(X)
        started = time.perf_counter()
        labels = super().predict(X)
        seconds = time.perf_counter() - started
        _append_row(
            "forest",
            {"call": "predict", "n_samples": int(len(X)), "seconds": seconds},
            FOREST_COLUMNS,
        )
        return labels


def timed_forest(*, n_features: int, n_trees: int) -> TimedRandomForestClassifier:
    """CARVE's default generalizability forest, timed.

    Built from the parameters of CARVE's own factory rather than restated,
    so an instrumented fit trains exactly the classifier an uninstrumented
    one does. CARVE sets random_state and n_jobs on its clone per resample.
    """
    from carve._utils import default_generalizability_classifier

    plain = default_generalizability_classifier(
        classifier=None,
        n_features=n_features,
        n_trees=n_trees,
        random_state=None,
        n_jobs=1,
    )
    return TimedRandomForestClassifier(**plain.get_params())


def instrumented_grids(
    grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """grids with carve.cluster.LeidenClustering swapped for the timed class."""
    return [
        (LeidenClustering if cls is _cluster.LeidenClustering else cls, dict(grid))
        for cls, grid in grids
    ]


def _shutdown_worker_pool() -> None:
    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True)


@contextmanager
def timing_directory(path: Path) -> Iterator[Path]:
    """Point the timing classes at path for the duration of the block.

    loky workers copy the environment when they are spawned, and joblib
    keeps its worker pool alive between calls, so a pool started before the
    variable was set would never see it. The pool is shut down on entry, so
    the block's first parallel call spawns workers that inherit the
    variable, and again on exit, so no later call reuses workers that still
    carry it.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get(TIMING_DIR_ENV)
    _shutdown_worker_pool()
    os.environ[TIMING_DIR_ENV] = str(path)
    try:
        yield path
    finally:
        if previous is None:
            os.environ.pop(TIMING_DIR_ENV, None)
        else:
            os.environ[TIMING_DIR_ENV] = previous
        _shutdown_worker_pool()


def read_timings(directory: Path) -> dict[str, pd.DataFrame]:
    """Every row the timing classes wrote under directory, by kind.

    Safe to call while a fit is still writing: a row cut off mid-write is
    dropped rather than read as data.
    """
    directory = Path(directory)
    tables: dict[str, pd.DataFrame] = {}
    for kind, columns in (("leiden", LEIDEN_COLUMNS), ("forest", FOREST_COLUMNS)):
        frames = [
            pd.read_csv(path, on_bad_lines="skip").dropna()
            for path in sorted(directory.glob(f"{kind}-*.csv"))
        ]
        frames = [frame for frame in frames if not frame.empty]
        tables[kind] = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=list(columns))
        )
    return tables
```

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_timing.py -q`
Expected: all PASS.

If `test_an_instrumented_fit_matches_a_plain_one_and_times_inside_workers` finds no worker rows, the pool restart did not take effect. Inspect `joblib.externals.loky.reusable_executor` before changing the mechanism, and report what you find rather than weakening the assertion.

- [ ] Step 5: Mutation check

Verify each guard can fail. Change one line at a time, run the test file, confirm a failure, and revert:
1. In `timing_directory`, delete the `_shutdown_worker_pool()` before `os.environ[...] = str(path)`. Expected: the worker test fails on its last assertion.
2. In `timed_forest`, return `TimedRandomForestClassifier()` (default parameters). Expected: `assert_frame_equal` fails, and so does `test_timed_forest_carries_the_default_forests_parameters`.
3. In `LeidenClustering.fit`, write `"leiden_s": total`. Expected: the sum test fails.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add src/benchmarks/_timing.py tests/benchmarks/test_timing.py
git commit -m "feat(benchmarks): time Leiden, graph and forest calls inside CARVE's workers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Run directory, settings and the embed stage

Files:
- Create: `src/benchmarks/_heca_stages.py`
- Modify: `pyproject.toml` (`psutil` in the `benchmarks` extra)
- Modify: `tests/benchmarks/_helpers.py` (add `small_heca_study`)
- Test: `tests/benchmarks/test_heca_stages.py`

Interfaces:
- Consumes: `STUDIES`, `load_study`, `study_resolution_grids` (Task 2); `meta["obs"]` (Task 3).
- Produces (in `benchmarks._heca_stages`):
  - `SEED = 42`, `SUBSAMPLE_RATIO` (CARVE's default, 0.618), `CARVE_N_TREES` (CARVE's default, 100).
  - `StageOutputExists(FileExistsError)`.
  - `Setting(label: str, n_neighbors: int)` (frozen dataclass).
  - `resolve_study(study: Study | None) -> Study`.
  - `sweep_settings(study) -> list[Setting]` (labels `"leiden_15"`, `"leiden_50"`).
  - `split_sizes(n_cells: int) -> tuple[int, int]`.
  - `scaled_neighbors(setting, *, n_fit, n_full) -> int`.
  - `write_json(path, payload) -> Path`, `read_json(path) -> dict`, `guard_output(path, *, force) -> None`.
  - `physical_cores() -> int`, `node_memory_bytes() -> int`.
  - `record_stage(run_dir, stage) -> dict`.
  - `run_embed(run_dir, *, study=None, force=False) -> dict`.
  - In `tests.benchmarks._helpers`: `small_heca_study(root) -> Study`.

- [ ] Step 1: Add psutil and the test helper

In `pyproject.toml`, in the `benchmarks` extra, add after `"scanpy>=1.11.4",`:

```toml
  # The hECA stages read physical cores and node memory, and sample the fit's
  # memory, through psutil.
  "psutil>=5.9",
```

Append to `tests/benchmarks/_helpers.py`:

```python
def small_heca_study(root):
    """STUDIES["heca"] on two synthetic organs, sized for a test.

    Lung and Brain from different studies, the real 15- and 50-neighbor
    settings, a two-value grid that splits the organs into at least two
    clusters on every subsample (a single cluster warns, and warnings are
    errors here), two resamples, and no anchors: the fit is far below
    anchor_threshold.
    """
    from dataclasses import replace
    from pathlib import Path

    from benchmarks._studies import STUDIES
    from benchmarks.datasets import load_heca

    root = Path(root)
    directory = root / "hECA"
    directory.mkdir(parents=True, exist_ok=True)
    write_heca_organ(directory, "Lung", 90, 0, study_id="10.1000/lung")
    write_heca_organ(directory, "Brain", 80, 1, study_id="10.1000/brain")

    def loader(subsample):
        return load_heca(
            root=root,
            organs=("Lung", "Brain"),
            subsample=subsample,
            random_state=42,
            label_column="organ",
            n_top_peaks=50,
            n_components=5,
        )

    return replace(
        STUDIES["heca"],
        loader=loader,
        resolutions=(1.0, 2.0),
        n_resamples=2,
        consensus_anchors=None,
    )
```

- [ ] Step 2: Write the failing tests

Create `tests/benchmarks/test_heca_stages.py`:

```python
"""Tests for the hECA run's stages."""

import json

import pytest

from benchmarks._heca_stages import (
    Setting,
    StageOutputExists,
    run_embed,
    scaled_neighbors,
    split_sizes,
    sweep_settings,
)
from benchmarks._studies import STUDIES
from carve._utils import split_subsample_indices
from tests.benchmarks._helpers import small_heca_study


@pytest.fixture
def study(tmp_path):
    return small_heca_study(tmp_path / "data")


class TestSweep:
    def test_settings_follow_the_studys_grids(self):
        assert sweep_settings(STUDIES["heca"]) == [
            Setting("leiden_15", 15),
            Setting("leiden_50", 50),
        ]

    @pytest.mark.parametrize("n", [10, 113, 1000, 695_304])
    def test_split_sizes_match_carves_split(self, n):
        train, test = split_subsample_indices(n, subsample_ratio=0.618, random_state=0)
        assert split_sizes(n) == (len(train), len(test))

    @pytest.mark.parametrize(
        "setting, n_fit, expected",
        [
            (Setting("leiden_15", 15), 429_698, 9),
            (Setting("leiden_15", 15), 265_606, 6),
            (Setting("leiden_50", 50), 429_698, 31),
            (Setting("leiden_50", 50), 265_606, 19),
        ],
    )
    def test_neighbors_are_scaled_as_carve_scales_them(self, setting, n_fit, expected):
        assert scaled_neighbors(setting, n_fit=n_fit, n_full=695_304) == expected


class TestEmbed:
    def test_records_the_pooled_counts(self, study, tmp_path):
        run = tmp_path / "run"
        record = run_embed(run, study=study)
        assert json.loads((run / "embed.json").read_text()) == record
        assert record["n_organs"] == 2
        assert record["n_cell_types"] == 2
        assert record["n_studies"] == 2
        assert record["n_features"] == 5
        assert record["cached"] is False
        env = json.loads((run / "env.json").read_text())
        assert [stage["stage"] for stage in env["stages"]] == ["embed"]
        assert "git_sha" in env["provenance"]
        assert env["stages"][0]["physical_cores"] >= 1

    def test_refuses_to_overwrite_and_force_records_a_second_run(self, study, tmp_path):
        run = tmp_path / "run"
        run_embed(run, study=study)
        with pytest.raises(StageOutputExists):
            run_embed(run, study=study)
        record = run_embed(run, study=study, force=True)
        assert record["cached"] is True
        env = json.loads((run / "env.json").read_text())
        assert len(env["stages"]) == 2
```

- [ ] Step 3: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca_stages.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._heca_stages'`.

- [ ] Step 4: Implement

Create `src/benchmarks/_heca_stages.py`:

```python
"""Stages of the full-scale hECA run on Longleaf: embed, fit and status.

Each stage reads and writes one run directory, so the stages run as separate
SLURM jobs and their outputs copy back as a unit. The calibration stage is
in _heca_calibration, the notebook-side reading in _heca_report. Layout:

    <run-dir>/env.json          provenance, then one record per stage run
    <run-dir>/embed.json
    <run-dir>/calibration.csv   one row per setting and scanned resolution
    <run-dir>/calibration.json
    <run-dir>/fit/              the CARVE cache and its fingerprint,
                                timings/, memory.csv, started.json,
                                runtime.json, sacct.txt

Design: docs/superpowers/specs/2026-10-01-heca-longleaf-runtime-design.md.
"""

import json
import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from carve import CARVE
from carve._utils import scale_neighbor_count
from carve.cluster import LeidenClustering

from ._artifacts import peak_rss_bytes, provenance
from ._studies import STUDIES, load_study, study_resolution_grids
from ._types import Study

#: The seed every stage draws with, as for the other case studies.
SEED = 42

#: CARVE's own defaults, read rather than restated, so the calibration split
#: and forest are the ones the fit uses.
SUBSAMPLE_RATIO: float = next(
    field.default for field in fields(CARVE) if field.name == "subsample_ratio"
)
CARVE_N_TREES: int = next(
    field.default for field in fields(CARVE) if field.name == "n_trees"
)

#: Versions recorded beside provenance()'s, for the packages this run adds.
EXTRA_PACKAGES: tuple[str, ...] = (
    "igraph",
    "leidenalg",
    "psutil",
    "scanpy",
    "anndata",
    "threadpoolctl",
)


class StageOutputExists(FileExistsError):
    """A stage's output is already in the run directory."""


@dataclass(frozen=True)
class Setting:
    """One graph setting of the sweep: a label and its configured neighbors."""

    label: str
    n_neighbors: int


def resolve_study(study: Study | None) -> Study:
    """The study a stage runs: STUDIES["heca"] unless a test passes another."""
    return STUDIES["heca"] if study is None else study


def sweep_settings(study: Study) -> list[Setting]:
    """The study's settings in grid order, one per resolution grid."""
    settings = []
    for _, grid in study_resolution_grids(study):
        (n_neighbors,) = grid["n_neighbors"]
        settings.append(
            Setting(label=f"leiden_{n_neighbors}", n_neighbors=int(n_neighbors))
        )
    return settings


def split_sizes(n_cells: int) -> tuple[int, int]:
    """Rows in one resample's training split and in its complement.

    The arithmetic of carve._utils.split_subsample_indices, which a test
    holds this to.
    """
    n_train = int(np.float64(SUBSAMPLE_RATIO * n_cells))
    return n_train, n_cells - n_train


def scaled_neighbors(setting: Setting, *, n_fit: int, n_full: int) -> int:
    """The neighbor count CARVE fits setting with on n_fit of n_full rows."""
    params = scale_neighbor_count(
        LeidenClustering,
        {"n_neighbors": setting.n_neighbors},
        n_fit=n_fit,
        n_full=n_full,
    )
    return int(params["n_neighbors"])


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def guard_output(path: Path, *, force: bool) -> None:
    """Refuse to overwrite a stage's output unless forced."""
    if Path(path).exists() and not force:
        raise StageOutputExists(f"{path} exists. Pass --force to overwrite it.")


def physical_cores() -> int:
    """Physical cores, or logical ones where psutil cannot tell."""
    import psutil

    return int(psutil.cpu_count(logical=False) or os.cpu_count() or 1)


def node_memory_bytes() -> int:
    import psutil

    return int(psutil.virtual_memory().total)


def _extra_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in EXTRA_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def record_stage(run_dir: Path, stage: str) -> dict[str, Any]:
    """Append this stage's machine record to env.json, creating the file first.

    Provenance (commit, package versions, platform) is written once, by the
    first stage to run. Each stage then adds its host, SLURM job id, core
    counts and memory, since the stages run on different nodes.
    """
    import psutil

    path = Path(run_dir) / "env.json"
    record = (
        read_json(path)
        if path.exists()
        else {"provenance": provenance(), "packages": _extra_versions(), "stages": []}
    )
    record["stages"].append(
        {
            "stage": stage,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "physical_cores": physical_cores(),
            "logical_cores": psutil.cpu_count(logical=True),
            "memory_bytes": node_memory_bytes(),
        }
    )
    write_json(path, record)
    return record


def run_embed(
    run_dir: Path, *, study: Study | None = None, force: bool = False
) -> dict[str, Any]:
    """Compute or load the pooled embedding and record what it cost.

    The loader caches the embedding under data/hECA/, so the later stages
    load it in seconds. wall_clock_s is the preprocessing cost only when
    cached is False; a run whose cache already existed records a load.
    """
    study = resolve_study(study)
    run_dir = Path(run_dir)
    out = run_dir / "embed.json"
    guard_output(out, force=force)
    record_stage(run_dir, "embed")

    started = time.perf_counter()
    X, _, meta = load_study(study)
    elapsed = time.perf_counter() - started

    obs = meta["obs"]
    record = {
        "cached": bool(meta["cached"]),
        "wall_clock_s": elapsed,
        "peak_rss_bytes": int(peak_rss_bytes()),
        "organs": list(meta["organs"]),
        "n_cells_full": int(meta["n_cells_full"]),
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_peaks_full": int(meta["n_peaks_full"]),
        "n_peaks_open": int(meta["n_peaks_open"]),
        "n_peaks_selected": int(meta["n_peaks_selected"]),
        "n_organs": int(obs["organ"].nunique()),
        "n_cell_types": int(obs["cell_type"].nunique()),
        "n_studies": int(obs["study_id"].nunique()),
    }
    write_json(out, record)
    return record
```

- [ ] Step 5: Run the tests to verify they pass

Run: `VIRTUAL_ENV=$PWD/.venv ~/.local/bin/uv pip install -e ".[dev,graph,benchmarks]"` (records the new dependency; psutil 7.2.2 is already installed), then `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca_stages.py -q`
Expected: all PASS.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add pyproject.toml src/benchmarks/_heca_stages.py tests/benchmarks/_helpers.py tests/benchmarks/test_heca_stages.py
git commit -m "feat(benchmarks): add the hECA run directory and its embed stage

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The calibration stage

Files:
- Create: `src/benchmarks/_heca_calibration.py`
- Test: `tests/benchmarks/test_heca_calibration.py`

Interfaces:
- Consumes: `graph_timers`, `GraphTimes` (Task 4); `CARVE_N_TREES`, `SEED`, `SUBSAMPLE_RATIO`, `guard_output`, `record_stage`, `resolve_study`, `scaled_neighbors`, `sweep_settings`, `write_json` (Task 5).
- Produces (in `benchmarks._heca_calibration`):
  - `SCAN_RESOLUTIONS` (40 log-spaced values, 0.001 to 10), `GRID_SIZE = 15`, `PROJECTION_N_JOBS = (16, 24, 32, 48, 64, 128)`.
  - `GridRuleError(ValueError)`.
  - `propose_grid(scan: pd.DataFrame, *, lower_target: int, upper_target: int, size: int = GRID_SIZE) -> tuple[float, ...]`. scan has columns setting, resolution, n_clusters.
  - `project_runtime(costs: Mapping[str, Mapping[str, float]], *, n_resolutions, n_resamples, n_train, n_test, n_jobs_options=PROJECTION_N_JOBS) -> dict`, returning keys `per_resample_seconds` (by setting), `core_hours`, and `wall_clock_hours` (keyed by `str(n_jobs)`). Cost keys: `graph_train_s`, `graph_test_s`, `leiden_s`, `forest_fit_s`, `forest_predict_s`.
  - `calibration_costs(calibration: Mapping, scan: pd.DataFrame, *, resolutions: Sequence[float]) -> dict[str, dict[str, float]]`.
  - `timed_graph(X, n_neighbors) -> tuple[igraph.Graph, GraphTimes]`.
  - `scan_resolutions(graph, resolutions, *, seed) -> list[tuple[float, np.ndarray, float]]`.
  - `run_calibrate(run_dir, *, study=None, scan=SCAN_RESOLUTIONS, force=False) -> dict`. `calibration.json` keys: `n_cells`, `n_train`, `n_test`, `scan`, `lower_target`, `upper_target`, `proposed_grid` (list or None), `grid_error` (str or None), `graph_costs` (setting to `knn_train_s`, `graph_train_s`, `knn_test_s`, `graph_test_s`), `forest_fit_s`, `forest_predict_s`, `baseline_rss_bytes`, `worker_peak_bytes`, `projection`. `calibration.csv` columns: setting, n_neighbors, n_neighbors_scaled, resolution, n_clusters, seconds, ari_organ, ari_cell_type.

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/test_heca_calibration.py`:

```python
"""Tests for the hECA calibration stage."""

import json

import numpy as np
import pandas as pd
import pytest

import carve.cluster as cluster
from benchmarks._heca_calibration import (
    GridRuleError,
    calibration_costs,
    project_runtime,
    propose_grid,
    run_calibrate,
    scan_resolutions,
)
from benchmarks._heca_stages import StageOutputExists, scaled_neighbors, sweep_settings
from tests.benchmarks._helpers import small_heca_study


def _scan(rows):
    return pd.DataFrame(rows, columns=["setting", "resolution", "n_clusters"])


class TestProposeGrid:
    def test_spans_the_earliest_start_and_the_latest_end(self):
        # a reaches 5 clusters at 0.1, b only at 1.0: the grid starts at 0.1.
        # a passes 108 above 1.0, b stays within it to 10: it ends at 10.
        scan = _scan(
            [
                ("a", 0.01, 3), ("a", 0.1, 5), ("a", 1.0, 80), ("a", 10.0, 300),
                ("b", 0.01, 2), ("b", 0.1, 4), ("b", 1.0, 30), ("b", 10.0, 100),
            ]
        )
        assert propose_grid(scan, lower_target=5, upper_target=108) == (
            0.1, 0.14, 0.19, 0.27, 0.37, 0.52, 0.72, 1.0,
            1.4, 1.9, 2.7, 3.7, 5.2, 7.2, 10.0,
        )

    def test_a_count_that_dips_does_not_move_the_endpoints(self):
        # Leiden's count is not monotone in resolution. The rule reads the
        # smallest value reaching the lower target and the largest within the
        # upper one, wherever they fall.
        scan = _scan(
            [
                ("a", 0.01, 1), ("a", 0.1, 6), ("a", 0.3, 4),
                ("a", 1.0, 50), ("a", 3.0, 120), ("a", 10.0, 90),
            ]
        )
        grid = propose_grid(scan, lower_target=5, upper_target=108)
        assert grid[0] == pytest.approx(0.1)
        assert grid[-1] == pytest.approx(10.0)

    def test_a_setting_that_never_reaches_the_lower_target_raises(self):
        with pytest.raises(GridRuleError, match="'a' never reaches 5"):
            propose_grid(
                _scan([("a", 0.1, 2), ("a", 10.0, 4)]),
                lower_target=5,
                upper_target=108,
            )

    def test_a_setting_already_past_the_upper_target_raises(self):
        with pytest.raises(GridRuleError, match="'a' already exceeds 108"):
            propose_grid(
                _scan([("a", 0.001, 200), ("a", 1.0, 400)]),
                lower_target=5,
                upper_target=108,
            )

    def test_crossed_endpoints_raise(self):
        with pytest.raises(GridRuleError, match="cross"):
            propose_grid(
                _scan([("a", 0.1, 2), ("a", 1.0, 200)]),
                lower_target=5,
                upper_target=108,
            )

    def test_endpoints_too_close_for_distinct_values_raise(self):
        with pytest.raises(GridRuleError, match="duplicate"):
            propose_grid(
                _scan([("a", 1.0, 5), ("a", 1.05, 100)]),
                lower_target=5,
                upper_target=108,
            )


class TestProjection:
    COSTS = {
        "graph_train_s": 10.0,
        "graph_test_s": 4.0,
        "leiden_s": 2.0,
        "forest_fit_s": 30.0,
        "forest_predict_s": 1.0,
    }

    def test_one_setting_matches_the_hand_computation(self):
        out = project_runtime(
            {"a": self.COSTS},
            n_resolutions=15,
            n_resamples=100,
            n_train=600,
            n_test=400,
            n_jobs_options=(24, 128),
        )
        # Two training graphs, one complement graph, three Leiden runs (the
        # complement's scaled by its size), one forest fit and predict.
        per = 2 * 10.0 + 4.0 + (2 + 400 / 600) * 2.0 + 30.0 + 1.0
        assert out["per_resample_seconds"]["a"] == pytest.approx(per)
        assert out["core_hours"] == pytest.approx(15 * 100 * per / 3600)
        # 100 resamples in rounds of 24 workers take 5 rounds; of 128, one.
        assert out["wall_clock_hours"]["24"] == pytest.approx(15 * 5 * per / 3600)
        assert out["wall_clock_hours"]["128"] == pytest.approx(15 * 1 * per / 3600)

    def test_settings_add(self):
        kwargs = dict(n_resolutions=15, n_resamples=100, n_train=600, n_test=400)
        one = project_runtime({"a": self.COSTS}, **kwargs)
        two = project_runtime({"a": self.COSTS, "b": self.COSTS}, **kwargs)
        assert two["core_hours"] == pytest.approx(2 * one["core_hours"])


def test_calibration_costs_average_leiden_inside_the_grid():
    calibration = {
        "graph_costs": {
            "a": {
                "knn_train_s": 8.0,
                "graph_train_s": 10.0,
                "knn_test_s": 3.0,
                "graph_test_s": 4.0,
            }
        },
        "forest_fit_s": 30.0,
        "forest_predict_s": 1.0,
    }
    scan = pd.DataFrame(
        {
            "setting": ["a", "a", "a"],
            "resolution": [0.01, 0.1, 1.0],
            "seconds": [100.0, 2.0, 4.0],
        }
    )
    costs = calibration_costs(calibration, scan, resolutions=(0.1, 1.0))
    assert costs["a"] == {
        "graph_train_s": 10.0,
        "graph_test_s": 4.0,
        "leiden_s": pytest.approx(3.0),
        "forest_fit_s": 30.0,
        "forest_predict_s": 1.0,
    }


def test_scan_matches_leiden_clustering():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.5, (60, 2)), rng.normal(5, 0.5, (60, 2))])
    graph = cluster.build_knn_graph(X, n_neighbors=10)
    ((resolution, labels, seconds),) = scan_resolutions(graph, (1.0,), seed=0)
    expected = cluster.LeidenClustering(n_neighbors=10, resolution=1.0, random_state=0)
    assert resolution == 1.0
    assert seconds > 0
    assert np.array_equal(labels, expected.fit(X).labels_)


class TestRunCalibrate:
    def test_writes_the_scan_the_costs_and_a_projection(self, tmp_path):
        study = small_heca_study(tmp_path / "data")
        run = tmp_path / "run"
        record = run_calibrate(run, study=study, scan=(0.5, 1.0, 2.0))

        scan = pd.read_csv(run / "calibration.csv")
        assert len(scan) == 2 * 3
        assert set(scan["setting"]) == {"leiden_15", "leiden_50"}
        assert set(scan["n_neighbors_scaled"]) == {
            scaled_neighbors(s, n_fit=record["n_train"], n_full=record["n_cells"])
            for s in sweep_settings(study)
        }
        assert scan["ari_organ"].between(-1, 1).all()

        assert record["n_train"] + record["n_test"] == record["n_cells"]
        assert record["lower_target"] == 2
        assert record["upper_target"] == 4
        assert (record["proposed_grid"] is None) != (record["grid_error"] is None)
        assert set(record["graph_costs"]) == {"leiden_15", "leiden_50"}
        assert record["forest_fit_s"] > 0
        assert record["projection"]["core_hours"] > 0
        assert record["worker_peak_bytes"] >= record["baseline_rss_bytes"]
        on_disk = json.loads((run / "calibration.json").read_text())
        assert on_disk["n_cells"] == record["n_cells"]

    def test_refuses_to_overwrite(self, tmp_path):
        study = small_heca_study(tmp_path / "data")
        run = tmp_path / "run"
        run_calibrate(run, study=study, scan=(1.0,))
        with pytest.raises(StageOutputExists):
            run_calibrate(run, study=study, scan=(1.0,))
```

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca_calibration.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._heca_calibration'`.

- [ ] Step 3: Implement

Create `src/benchmarks/_heca_calibration.py`:

```python
"""The calibration stage: a measured resolution grid and a runtime projection.

Runs once on the cluster before the fit, on the training split and complement
that the fit's first resample draws. For each setting it builds both graphs
with the neighbor counts CARVE scales to each subset, scans Leiden
resolution on the training graph, and times every component the fit
repeats, single-threaded as each fit worker runs. It proposes the grid by
the spec's rule and projects the fit's runtime; the author commits the grid
to STUDIES before submitting the fit. The scan and the costs are written
even when the rule fails, so a failed rule never discards hours of scanning.
"""

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import carve.cluster as _cluster
from carve._utils import default_generalizability_classifier, split_subsample_indices

from ._artifacts import peak_rss_bytes
from ._heca_stages import (
    CARVE_N_TREES,
    SEED,
    SUBSAMPLE_RATIO,
    guard_output,
    record_stage,
    resolve_study,
    scaled_neighbors,
    sweep_settings,
    write_json,
)
from ._studies import load_study
from ._timing import GraphTimes, graph_timers
from ._types import Study

#: Forty log-spaced values from 0.001 to 10: wide enough for either setting
#: to pass organ level and twice the cell-type count.
SCAN_RESOLUTIONS: tuple[float, ...] = tuple(
    float(value) for value in np.geomspace(0.001, 10.0, 40)
)

#: Values in the proposed grid, as many as the provisional grid holds.
GRID_SIZE = 15

#: Worker counts the projection is tabulated for. The fit node's count is
#: not known when calibration runs.
PROJECTION_N_JOBS: tuple[int, ...] = (16, 24, 32, 48, 64, 128)


class GridRuleError(ValueError):
    """The scan does not support a grid under the calibration rule."""


def propose_grid(
    scan: pd.DataFrame,
    *,
    lower_target: int,
    upper_target: int,
    size: int = GRID_SIZE,
) -> tuple[float, ...]:
    """The grid the calibration rule proposes from a scan.

    For each setting, lo is the smallest scanned resolution giving at least
    lower_target clusters and hi the largest giving at most upper_target.
    The grid runs from the smallest lo to the largest hi, so every setting
    covers organ level at the coarse end and the upper target at the fine
    end, in size log-spaced values rounded to two significant figures.
    """
    lows, highs = [], []
    for setting, rows in scan.groupby("setting", sort=False):
        reaching = rows.loc[rows["n_clusters"] >= lower_target, "resolution"]
        if reaching.empty:
            raise GridRuleError(
                f"Setting {setting!r} never reaches {lower_target} clusters "
                f"(largest scanned resolution {rows['resolution'].max():g}); "
                "widen the scan upward."
            )
        within = rows.loc[rows["n_clusters"] <= upper_target, "resolution"]
        if within.empty:
            raise GridRuleError(
                f"Setting {setting!r} already exceeds {upper_target} clusters at "
                f"{rows['resolution'].min():g}; widen the scan downward."
            )
        lows.append(float(reaching.min()))
        highs.append(float(within.max()))

    lo, hi = min(lows), max(highs)
    if lo >= hi:
        raise GridRuleError(
            f"The endpoints cross: the grid would start at {lo:g} and end at "
            f"{hi:g}. Inspect the scan; the targets may not suit this data."
        )
    grid = tuple(
        float(f"{lo * (hi / lo) ** (i / (size - 1)):.2g}") for i in range(size)
    )
    if len(set(grid)) != size:
        raise GridRuleError(
            f"From {lo:g} to {hi:g}, rounding to two significant figures gives "
            f"duplicate values: {grid}."
        )
    return grid


def project_runtime(
    costs: Mapping[str, Mapping[str, float]],
    *,
    n_resolutions: int,
    n_resamples: int,
    n_train: int,
    n_test: int,
    n_jobs_options: Sequence[int] = PROJECTION_N_JOBS,
) -> dict[str, Any]:
    """Project the fit's runtime from single-threaded component costs.

    One configuration and one resample cost two training graphs, one
    complement graph, three Leiden runs (the complement's scaled by its
    size) and one forest fit and predict. Resamples run in rounds of n_jobs
    workers, so wall-clock time counts whole rounds. A lower bound: it
    leaves out CARVE's own overhead and contention between workers.
    """
    per_resample = {
        setting: (
            2 * c["graph_train_s"]
            + c["graph_test_s"]
            + (2 + n_test / n_train) * c["leiden_s"]
            + c["forest_fit_s"]
            + c["forest_predict_s"]
        )
        for setting, c in costs.items()
    }
    core_seconds = n_resolutions * n_resamples * sum(per_resample.values())
    wall_clock_hours = {
        str(n_jobs): n_resolutions
        * sum(math.ceil(n_resamples / n_jobs) * c for c in per_resample.values())
        / 3600
        for n_jobs in n_jobs_options
    }
    return {
        "per_resample_seconds": per_resample,
        "core_hours": core_seconds / 3600,
        "wall_clock_hours": wall_clock_hours,
    }


def calibration_costs(
    calibration: Mapping[str, Any],
    scan: pd.DataFrame,
    *,
    resolutions: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Per-setting costs for project_runtime, for a grid's range.

    Leiden's cost is the mean over the scanned resolutions inside the grid's
    range, or over the whole scan when none falls inside.
    """
    lo, hi = min(resolutions), max(resolutions)
    costs: dict[str, dict[str, float]] = {}
    for label, graph in calibration["graph_costs"].items():
        rows = scan[scan["setting"] == label]
        inside = rows[(rows["resolution"] >= lo) & (rows["resolution"] <= hi)]
        leiden_s = float((inside if not inside.empty else rows)["seconds"].mean())
        costs[label] = {
            "graph_train_s": float(graph["graph_train_s"]),
            "graph_test_s": float(graph["graph_test_s"]),
            "leiden_s": leiden_s,
            "forest_fit_s": float(calibration["forest_fit_s"]),
            "forest_predict_s": float(calibration["forest_predict_s"]),
        }
    return costs


def timed_graph(X: np.ndarray, n_neighbors: int) -> tuple[Any, GraphTimes]:
    """Build the graph LeidenClustering.fit builds on X, timing its two parts."""
    with graph_timers() as times:
        graph = _cluster.build_knn_graph(X, n_neighbors=n_neighbors)
    return graph, times


def scan_resolutions(
    graph: Any, resolutions: Sequence[float], *, seed: int
) -> list[tuple[float, np.ndarray, float]]:
    """Leiden on one prebuilt graph at each resolution.

    The partition call and its arguments are LeidenClustering.fit's, so the
    labels match a fit on the same data and seed.
    """
    import leidenalg as la

    results = []
    for resolution in resolutions:
        started = time.perf_counter()
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            resolution_parameter=float(resolution),
            weights="weight",
            n_iterations=-1,
            seed=seed,
        )
        seconds = time.perf_counter() - started
        labels = np.asarray(partition.membership, dtype=np.int32)
        results.append((float(resolution), labels, seconds))
    return results


def run_calibrate(
    run_dir: Path,
    *,
    study: Study | None = None,
    scan: Sequence[float] = SCAN_RESOLUTIONS,
    force: bool = False,
) -> dict[str, Any]:
    """Scan, time and project; write calibration.csv and calibration.json."""
    study = resolve_study(study)
    run_dir = Path(run_dir)
    csv_path = run_dir / "calibration.csv"
    json_path = run_dir / "calibration.json"
    guard_output(csv_path, force=force)
    guard_output(json_path, force=force)
    record_stage(run_dir, "calibrate")

    X, _, meta = load_study(study)
    baseline_rss = peak_rss_bytes()
    obs = meta["obs"]
    n_cells = int(X.shape[0])
    # The split the fit's first resample draws (seed 42 + resample 0).
    train_idx, test_idx = split_subsample_indices(
        n_cells, subsample_ratio=SUBSAMPLE_RATIO, random_state=SEED
    )
    X_train, X_test = X[train_idx], X[test_idx]
    organ = obs["organ"].to_numpy()[train_idx]
    cell_type = obs["cell_type"].to_numpy()[train_idx]
    middle = math.sqrt(min(study.resolutions) * max(study.resolutions))

    rows: list[dict[str, Any]] = []
    graph_costs: dict[str, dict[str, float]] = {}
    forest_labels: np.ndarray | None = None
    for setting in sweep_settings(study):
        k_train = scaled_neighbors(setting, n_fit=len(train_idx), n_full=n_cells)
        k_test = scaled_neighbors(setting, n_fit=len(test_idx), n_full=n_cells)
        graph, train_times = timed_graph(X_train, k_train)
        _, test_times = timed_graph(X_test, k_test)
        graph_costs[setting.label] = {
            "knn_train_s": train_times.knn_s,
            "graph_train_s": train_times.build_s,
            "knn_test_s": test_times.knn_s,
            "graph_test_s": test_times.build_s,
        }
        scanned = scan_resolutions(graph, scan, seed=SEED)
        del graph
        for resolution, labels, seconds in scanned:
            rows.append(
                {
                    "setting": setting.label,
                    "n_neighbors": setting.n_neighbors,
                    "n_neighbors_scaled": k_train,
                    "resolution": resolution,
                    "n_clusters": int(np.unique(labels).size),
                    "seconds": seconds,
                    "ari_organ": float(adjusted_rand_score(organ, labels)),
                    "ari_cell_type": float(adjusted_rand_score(cell_type, labels)),
                }
            )
        if forest_labels is None:
            # Any realistic partition times the forest: take the first
            # setting's at the scan value nearest the provisional grid's
            # geometric middle.
            _, forest_labels, _ = min(
                scanned, key=lambda item: abs(math.log(item[0] / middle))
            )

    forest = default_generalizability_classifier(
        classifier=None,
        n_features=int(X.shape[1]),
        n_trees=CARVE_N_TREES,
        random_state=SEED,
        n_jobs=1,
    )
    started = time.perf_counter()
    forest.fit(X_train, forest_labels)
    forest_fit_s = time.perf_counter() - started
    started = time.perf_counter()
    forest.predict(X_test)
    forest_predict_s = time.perf_counter() - started

    scan_frame = pd.DataFrame(rows)
    scan_frame.to_csv(csv_path, index=False)

    lower_target = int(obs["organ"].nunique())
    upper_target = 2 * int(obs["cell_type"].nunique())
    grid: tuple[float, ...] | None
    try:
        grid = propose_grid(
            scan_frame, lower_target=lower_target, upper_target=upper_target
        )
        grid_error = None
    except GridRuleError as error:
        grid, grid_error = None, str(error)

    record: dict[str, Any] = {
        "n_cells": n_cells,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "scan": [float(value) for value in scan],
        "lower_target": lower_target,
        "upper_target": upper_target,
        "proposed_grid": None if grid is None else list(grid),
        "grid_error": grid_error,
        "graph_costs": graph_costs,
        "forest_fit_s": forest_fit_s,
        "forest_predict_s": forest_predict_s,
        "baseline_rss_bytes": int(baseline_rss),
        "worker_peak_bytes": int(peak_rss_bytes()),
    }
    costs = calibration_costs(
        record, scan_frame, resolutions=grid if grid is not None else tuple(scan)
    )
    record["projection"] = project_runtime(
        costs,
        n_resolutions=len(grid) if grid is not None else len(study.resolutions),
        n_resamples=study.n_resamples,
        n_train=record["n_train"],
        n_test=record["n_test"],
    )
    write_json(json_path, record)
    return record
```

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca_calibration.py -q`
Expected: all PASS.

- [ ] Step 5: Mutation check

One change at a time, confirm a failure, revert:
1. In `propose_grid`, use `lo, hi = max(lows), min(highs)`. Expected: `test_spans_the_earliest_start_and_the_latest_end` fails.
2. In `project_runtime`, drop `math.ceil` (use `n_resamples / n_jobs`). Expected: `test_one_setting_matches_the_hand_computation` fails.
3. In `scan_resolutions`, pass `n_iterations=2`. Expected: `test_scan_matches_leiden_clustering` should fail. If it still passes on this small graph, record that in the task report; do not weaken the test.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add src/benchmarks/_heca_calibration.py tests/benchmarks/test_heca_calibration.py
git commit -m "feat(benchmarks): calibrate hECA's resolution grid and project the fit's runtime

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The fit stage

Files:
- Modify: `src/benchmarks/_heca_stages.py` (add `MEMORY_FRACTION`, `choose_n_jobs`, `MemorySampler`, `run_fit`)
- Modify: `tests/benchmarks/test_heca_stages.py`

Interfaces:
- Consumes: `fit_or_load_carve(classifier=)`, `carve_cache_path` (Task 2); `instrumented_grids`, `timed_forest`, `timing_directory`, `read_timings` (Task 4); `calibration.json`'s `worker_peak_bytes` (Task 6).
- Produces:
  - `choose_n_jobs(*, cores, memory_bytes, worker_peak_bytes, memory_fraction=0.9) -> int`.
  - `MemorySampler(path, *, interval=60.0)`, a context manager whose CSV columns are elapsed_s, own_rss_bytes, children_rss_bytes, n_children.
  - `run_fit(run_dir, *, study=None, n_jobs=None, force=False, sample_interval=60.0) -> dict`. It writes `fit/started.json` (keys started_at, n_cells, n_jobs, resolutions, n_resamples, settings), `fit/timings/`, `fit/memory.csv`, `fit/runtime.json`, and the CARVE cache at `carve_cache_path(study, root=run_dir / "fit", model_grids=study_resolution_grids(study), n_resamples=study.n_resamples)`.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_heca_stages.py`. Add to its imports: `import os`, `import time`, `import pandas as pd`, `from carve import CARVE`, `from benchmarks._studies import carve_cache_path, study_resolution_grids`, `from benchmarks._timing import read_timings`, and `MemorySampler`, `choose_n_jobs`, `run_fit` from `benchmarks._heca_stages`.

```python
class TestChooseNJobs:
    def test_one_worker_per_core_when_memory_allows(self):
        assert (
            choose_n_jobs(cores=24, memory_bytes=256 * 10**9, worker_peak_bytes=4 * 10**9)
            == 24
        )

    def test_memory_caps_the_workers(self):
        # 0.9 x 256 GB / 20 GB = 11.52.
        assert (
            choose_n_jobs(
                cores=24, memory_bytes=256 * 10**9, worker_peak_bytes=20 * 10**9
            )
            == 11
        )

    def test_never_fewer_than_one(self):
        assert choose_n_jobs(cores=24, memory_bytes=10**9, worker_peak_bytes=10**10) == 1


class TestMemorySampler:
    def test_writes_a_row_on_entry_and_on_exit(self, tmp_path):
        path = tmp_path / "memory.csv"
        with MemorySampler(path, interval=3600.0):
            pass
        frame = pd.read_csv(path)
        assert list(frame.columns) == [
            "elapsed_s",
            "own_rss_bytes",
            "children_rss_bytes",
            "n_children",
        ]
        assert len(frame) == 2
        assert (frame["own_rss_bytes"] > 0).all()

    def test_samples_on_its_interval(self, tmp_path):
        path = tmp_path / "memory.csv"
        with MemorySampler(path, interval=0.01):
            time.sleep(0.2)
        assert len(pd.read_csv(path)) > 2


class TestRunFit:
    def test_fits_every_configuration_and_times_it(self, study, tmp_path, monkeypatch):
        monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
        run = tmp_path / "run"
        record = run_fit(run, study=study, n_jobs=1, sample_interval=0.05)

        fit_dir = run / "fit"
        cache = carve_cache_path(
            study,
            root=fit_dir,
            model_grids=study_resolution_grids(study),
            n_resamples=study.n_resamples,
        )
        assert cache.is_file()
        assert record["cache_file"] == cache.name
        tables = read_timings(fit_dir / "timings")
        # 2 settings x 2 resolutions x 2 resamples x 3 subsets.
        assert len(tables["leiden"]) == 24
        assert int((tables["forest"]["call"] == "fit").sum()) == 8

        results = CARVE.load(str(cache)).estimator_results_
        assert set(results["estimator"]) == {"LeidenClustering"}
        assert sorted(set(results["n_neighbors"])) == [15, 50]

        started = json.loads((fit_dir / "started.json").read_text())
        assert started["n_jobs"] == 1
        assert started["n_resamples"] == 2
        assert started["resolutions"] == [1.0, 2.0]
        assert len(pd.read_csv(fit_dir / "memory.csv")) >= 2
        assert record["wall_clock_s"] > 0
        assert "LOKY_MAX_CPU_COUNT" not in os.environ

    def test_without_n_jobs_it_needs_calibration(self, study, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_fit(tmp_path / "run", study=study)

    def test_refuses_to_overwrite_and_force_starts_the_timings_afresh(
        self, study, tmp_path
    ):
        run = tmp_path / "run"
        run_fit(run, study=study, n_jobs=1)
        with pytest.raises(StageOutputExists):
            run_fit(run, study=study, n_jobs=1)
        run_fit(run, study=study, n_jobs=1, force=True)
        assert len(read_timings(run / "fit" / "timings")["leiden"]) == 24
```

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca_stages.py -q`
Expected: FAIL with `ImportError: cannot import name 'MemorySampler'`.

- [ ] Step 3: Implement

In `src/benchmarks/_heca_stages.py`, add `import csv`, `import shutil` and `import threading` to the standard-library imports, and extend the package imports:

```python
from ._studies import (
    STUDIES,
    carve_cache_path,
    fit_or_load_carve,
    load_study,
    study_resolution_grids,
)
from ._timing import instrumented_grids, timed_forest, timing_directory
```

Append:

```python
#: Share of node memory the fit's workers may plan to fill.
MEMORY_FRACTION = 0.9


def choose_n_jobs(
    *,
    cores: int,
    memory_bytes: int,
    worker_peak_bytes: int,
    memory_fraction: float = MEMORY_FRACTION,
) -> int:
    """Workers for the fit: one per physical core, unless memory allows fewer."""
    by_memory = int(memory_fraction * memory_bytes // max(int(worker_peak_bytes), 1))
    return max(1, min(int(cores), by_memory))


class MemorySampler:
    """Sample the resident memory of this process and of its children.

    One row per interval: seconds since entry, this process's RSS, and the
    summed RSS of every descendant (the loky workers). A row is also written
    on entry and on exit, so even a short block leaves two. This is the
    fit's memory curve, and a cross-check on SLURM's accounting.
    """

    COLUMNS = ("elapsed_s", "own_rss_bytes", "children_rss_bytes", "n_children")

    def __init__(self, path: Path, *, interval: float = 60.0) -> None:
        self.path = Path(path)
        self.interval = float(interval)

    def __enter__(self) -> "MemorySampler":
        import psutil

        self._process = psutil.Process()
        self._started = time.perf_counter()
        self._stop = threading.Event()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as handle:
            csv.writer(handle).writerow(self.COLUMNS)
        self._sample()
        self._thread = threading.Thread(
            target=self._run, name="memory-sampler", daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        import psutil

        own = self._process.memory_info().rss
        children_rss, n_children = 0, 0
        for child in self._process.children(recursive=True):
            try:
                children_rss += child.memory_info().rss
                n_children += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        elapsed = time.perf_counter() - self._started
        with self.path.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [f"{elapsed:.3f}", own, children_rss, n_children]
            )

    def __exit__(self, *exc_info: object) -> bool:
        self._stop.set()
        self._thread.join()
        self._sample()
        return False


def run_fit(
    run_dir: Path,
    *,
    study: Study | None = None,
    n_jobs: int | None = None,
    force: bool = False,
    sample_interval: float = 60.0,
) -> dict[str, Any]:
    """Fit CARVE on every cell with the timing classes in place.

    n_jobs defaults to one worker per physical core, capped by node memory
    over the per-worker peak calibration measured. LOKY_MAX_CPU_COUNT is set
    to the physical core count for the fit, so CARVE's core budget sees
    physical cores rather than hyperthreads: with one worker per core, each
    worker's forest and BLAS run single-threaded, and the spare cores of a
    memory-capped run go to forest threads.

    The cache path comes from the study's plain grids, not the instrumented
    ones: the cache key hashes the grids' repr, which names the estimator
    class's module, and the notebook looks the fit up by the plain grids.
    """
    study = resolve_study(study)
    run_dir = Path(run_dir)
    fit_dir = run_dir / "fit"
    grids = study_resolution_grids(study)
    cache_path = carve_cache_path(
        study, root=fit_dir, model_grids=grids, n_resamples=study.n_resamples
    )
    guard_output(cache_path, force=force)
    guard_output(fit_dir / "runtime.json", force=force)
    record_stage(run_dir, "fit")

    X, y, _ = load_study(study)
    cores = physical_cores()
    if n_jobs is None:
        calibration = read_json(run_dir / "calibration.json")
        n_jobs = choose_n_jobs(
            cores=cores,
            memory_bytes=node_memory_bytes(),
            worker_peak_bytes=int(calibration["worker_peak_bytes"]),
        )

    # A forced rerun starts its timing rows afresh; memory.csv is rewritten.
    shutil.rmtree(fit_dir / "timings", ignore_errors=True)
    started_at = time.time()
    write_json(
        fit_dir / "started.json",
        {
            "started_at": started_at,
            "n_cells": int(X.shape[0]),
            "n_jobs": int(n_jobs),
            "resolutions": [float(r) for r in study.resolutions],
            "n_resamples": int(study.n_resamples),
            "settings": [setting.label for setting in sweep_settings(study)],
        },
    )

    previous_cap = os.environ.get("LOKY_MAX_CPU_COUNT")
    os.environ["LOKY_MAX_CPU_COUNT"] = str(cores)
    try:
        with (
            timing_directory(fit_dir / "timings"),
            MemorySampler(fit_dir / "memory.csv", interval=sample_interval),
        ):
            started = time.perf_counter()
            fit_or_load_carve(
                X,
                y,
                cache_path=cache_path,
                model_grids=instrumented_grids(grids),
                n_resamples=study.n_resamples,
                n_jobs=int(n_jobs),
                random_state=SEED,
                force=force,
                consensus_anchors=study.consensus_anchors,
                classifier=timed_forest(
                    n_features=int(X.shape[1]), n_trees=CARVE_N_TREES
                ),
            )
            wall_clock_s = time.perf_counter() - started
    finally:
        if previous_cap is None:
            os.environ.pop("LOKY_MAX_CPU_COUNT", None)
        else:
            os.environ["LOKY_MAX_CPU_COUNT"] = previous_cap

    record = {
        "started_at": started_at,
        "finished_at": time.time(),
        "wall_clock_s": wall_clock_s,
        "n_jobs": int(n_jobs),
        "physical_cores": cores,
        "logical_cores": os.cpu_count(),
        "memory_bytes": node_memory_bytes(),
        "loky_max_cpu_count": cores,
        "resolutions": [float(r) for r in study.resolutions],
        "n_resamples": int(study.n_resamples),
        "seed": SEED,
        "n_cells": int(X.shape[0]),
        "cache_file": cache_path.name,
    }
    write_json(fit_dir / "runtime.json", record)
    return record
```

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca_stages.py -q`
Expected: all PASS.

- [ ] Step 5: Mutation check

One change at a time, confirm a failure, revert:
1. In `run_fit`, compute the cache path from `instrumented_grids(grids)`. Expected: `test_fits_every_configuration_and_times_it` fails on `cache.is_file()`.
2. Delete the `shutil.rmtree(...)` line. Expected: the force test counts 48 Leiden rows and fails.
3. Delete the `finally:` restoration of `LOKY_MAX_CPU_COUNT`. Expected: the first fit test fails on its last assertion.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add src/benchmarks/_heca_stages.py tests/benchmarks/test_heca_stages.py
git commit -m "feat(benchmarks): add the instrumented hECA fit stage

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Status of a running fit

Files:
- Modify: `src/benchmarks/_heca_stages.py` (add `ABORT_AFTER_S`, `label_settings`, `fit_status`, `format_status`)
- Modify: `tests/benchmarks/test_heca_stages.py`

Interfaces:
- Consumes: `read_timings`, `LEIDEN_COLUMNS` (Task 4); `fit/started.json` (Task 7).
- Produces:
  - `ABORT_AFTER_S = 10 * 86_400`.
  - `label_settings(leiden: pd.DataFrame, study, *, n_cells: int) -> pd.DataFrame`, which adds a `setting` column.
  - `fit_status(run_dir, *, study=None, now=None) -> dict` with keys completed, total, elapsed_s, seconds_per_configuration (float or None), projected_total_s (float or None), projected_finish (unix time or None), abort (bool), and clusters (a DataFrame indexed by resolution, one column per setting, median cluster count on the training split).
  - `format_status(status) -> str`.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_heca_stages.py`. Add `fit_status`, `format_status` and `write_json` to the `benchmarks._heca_stages` import, and `from benchmarks._timing import LEIDEN_COLUMNS`.

```python
def _leiden_row(n_neighbors, n_samples, resolution, n_clusters):
    return {
        "pid": 1,
        "host": "node",
        "unix_time": 0.0,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "knn_s": 1.0,
        "graph_s": 1.0,
        "leiden_s": 1.0,
        "fit_s": 3.0,
    }


class TestStatus:
    N_CELLS = 1000

    def _write(self, run, rows, *, started_at=1000.0):
        fit_dir = run / "fit"
        write_json(
            fit_dir / "started.json",
            {
                "started_at": started_at,
                "n_cells": self.N_CELLS,
                "n_jobs": 4,
                "resolutions": [1.0, 2.0],
                "n_resamples": 2,
                "settings": ["leiden_15", "leiden_50"],
            },
        )
        (fit_dir / "timings").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=list(LEIDEN_COLUMNS)).to_csv(
            fit_dir / "timings" / "leiden-node-1.csv", index=False
        )

    def _rows(self):
        # leiden_15 at 1.0 is complete (2 resamples x 3 subsets); at 2.0 it
        # has half its rows.
        n_train, n_test = split_sizes(self.N_CELLS)
        setting = Setting("leiden_15", 15)
        k_train = scaled_neighbors(setting, n_fit=n_train, n_full=self.N_CELLS)
        k_test = scaled_neighbors(setting, n_fit=n_test, n_full=self.N_CELLS)
        return (
            [_leiden_row(k_train, n_train, 1.0, 4)] * 4
            + [_leiden_row(k_test, n_test, 1.0, 3)] * 2
            + [_leiden_row(k_train, n_train, 2.0, 7)] * 3
        )

    def test_counts_complete_configurations_and_projects_the_finish(self, tmp_path):
        run = tmp_path / "run"
        self._write(run, self._rows())
        status = fit_status(run, now=1000.0 + 3600.0)
        assert status["completed"] == 1
        assert status["total"] == 4
        assert status["seconds_per_configuration"] == pytest.approx(3600.0)
        assert status["projected_total_s"] == pytest.approx(4 * 3600.0)
        assert status["projected_finish"] == pytest.approx(1000.0 + 4 * 3600.0)
        assert status["abort"] is False
        assert status["clusters"].loc[1.0, "leiden_15"] == 4
        assert status["clusters"].loc[2.0, "leiden_15"] == 7

    def test_the_abort_rule_fires_past_ten_days(self, tmp_path):
        run = tmp_path / "run"
        self._write(run, self._rows())
        # One configuration in three days projects twelve for four.
        status = fit_status(run, now=1000.0 + 3 * 86_400)
        assert status["abort"] is True

    def test_no_projection_before_a_configuration_completes(self, tmp_path):
        run = tmp_path / "run"
        self._write(run, self._rows()[-3:])
        status = fit_status(run, now=5000.0)
        assert status["completed"] == 0
        assert status["seconds_per_configuration"] is None
        assert status["projected_total_s"] is None
        assert status["abort"] is False

    def test_format_names_the_count_and_the_rule(self, tmp_path):
        run = tmp_path / "run"
        self._write(run, self._rows())
        text = format_status(fit_status(run, now=1000.0 + 3600.0))
        assert "Configurations complete: 1 of 4" in text
        assert "Abort rule" in text
        assert "not triggered" in text
```

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca_stages.py::TestStatus -q`
Expected: FAIL with `ImportError: cannot import name 'fit_status'`.

- [ ] Step 3: Implement

In `src/benchmarks/_heca_stages.py`, add `import pandas as pd` to the third-party imports and change the `_timing` import to `from ._timing import instrumented_grids, read_timings, timed_forest, timing_directory`. Append:

```python
#: The early-abort rule: cancel and revisit if the projected total passes this.
ABORT_AFTER_S = 10 * 86_400


def label_settings(
    leiden: pd.DataFrame, study: Study, *, n_cells: int
) -> pd.DataFrame:
    """Leiden timing rows with the setting each came from.

    A row records the neighbor count CARVE actually fitted with, which it
    scales to each subset's size; this maps (scaled count, subset size)
    back to the setting.
    """
    n_train, n_test = split_sizes(n_cells)
    lookup = {}
    for setting in sweep_settings(study):
        for n_fit in (n_train, n_test):
            key = (scaled_neighbors(setting, n_fit=n_fit, n_full=n_cells), n_fit)
            lookup[key] = setting.label
    settings = [
        lookup.get((int(k), int(n)))
        for k, n in zip(leiden["n_neighbors"], leiden["n_samples"], strict=True)
    ]
    return leiden.assign(setting=settings)


def fit_status(
    run_dir: Path, *, study: Study | None = None, now: float | None = None
) -> dict[str, Any]:
    """Progress of a running or finished fit, from its live timing rows.

    Configurations run one after another, each across every resample, so
    the mean time of the completed ones projects the total. A configuration
    is complete once it has 3 x n_resamples Leiden rows: both training
    splits and the complement, per resample.
    """
    study = resolve_study(study)
    fit_dir = Path(run_dir) / "fit"
    started = read_json(fit_dir / "started.json")
    n_cells = int(started["n_cells"])
    n_resamples = int(started["n_resamples"])
    n_train, _ = split_sizes(n_cells)
    leiden = label_settings(
        read_timings(fit_dir / "timings")["leiden"], study, n_cells=n_cells
    )

    total = len(sweep_settings(study)) * len(started["resolutions"])
    counts = leiden.groupby(["setting", "resolution"]).size()
    completed = int((counts >= 3 * n_resamples).sum())

    now = time.time() if now is None else float(now)
    elapsed = now - float(started["started_at"])
    per_configuration = elapsed / completed if completed else None
    projected_total = None if per_configuration is None else per_configuration * total
    training = leiden[leiden["n_samples"] == n_train]
    clusters = (
        pd.DataFrame()
        if training.empty
        else training.groupby(["resolution", "setting"])["n_clusters"]
        .median()
        .unstack("setting")
    )
    return {
        "completed": completed,
        "total": total,
        "elapsed_s": elapsed,
        "seconds_per_configuration": per_configuration,
        "projected_total_s": projected_total,
        "projected_finish": None
        if projected_total is None
        else float(started["started_at"]) + projected_total,
        "abort": projected_total is not None and projected_total > ABORT_AFTER_S,
        "clusters": clusters,
    }


def format_status(status: Mapping[str, Any]) -> str:
    """The lines python -m benchmarks.heca status prints."""
    lines = [
        f"Configurations complete: {status['completed']} of {status['total']}",
        f"Elapsed: {status['elapsed_s'] / 3600:.1f} h",
    ]
    if status["seconds_per_configuration"] is None:
        lines.append("No configuration has completed yet, so there is no projection.")
    else:
        finish = datetime.fromtimestamp(status["projected_finish"], UTC)
        lines.append(
            f"Per configuration: {status['seconds_per_configuration'] / 3600:.2f} h"
        )
        lines.append(
            f"Projected total: {status['projected_total_s'] / 3600:.1f} h, "
            f"finishing {finish:%Y-%m-%d %H:%M} UTC"
        )
    verdict = (
        "triggered: cancel the fit and revisit the grid or the resample count"
        if status["abort"]
        else "not triggered"
    )
    lines.append(
        f"Abort rule (projected total over {ABORT_AFTER_S / 86_400:.0f} days): "
        f"{verdict}"
    )
    clusters = status["clusters"]
    if not clusters.empty:
        lines.append("Median clusters on the training split, by resolution:")
        lines.append(clusters.to_string())
    return "\n".join(lines)
```

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca_stages.py -q`
Expected: all PASS.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add src/benchmarks/_heca_stages.py tests/benchmarks/test_heca_stages.py
git commit -m "feat(benchmarks): report a running hECA fit's progress and projected finish

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: The CLI

Files:
- Create: `src/benchmarks/heca.py`
- Test: `tests/benchmarks/test_heca.py`

Interfaces:
- Consumes: `run_embed`, `run_fit`, `fit_status`, `format_status`, `StageOutputExists` (Tasks 5, 7, 8); `run_calibrate` (Task 6).
- Produces: `python -m benchmarks.heca {embed,calibrate,fit,status} --run-dir PATH [--force] [--n-jobs N]`. `main(argv: list[str] | None = None) -> int` returns 0 on success, 1 when calibration proposes no grid, and 2 when a stage refuses to overwrite.

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/test_heca.py`:

```python
"""Tests for the hECA command-line entry point."""

import json

import pytest

from benchmarks._studies import STUDIES
from benchmarks.heca import _parser, main
from tests.benchmarks._helpers import small_heca_study


def test_an_unknown_stage_is_rejected():
    with pytest.raises(SystemExit):
        _parser().parse_args(["ladder", "--run-dir", "x"])


def test_the_stages_run_in_order_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(STUDIES, "heca", small_heca_study(tmp_path / "data"))
    run = str(tmp_path / "run")

    assert main(["embed", "--run-dir", run]) == 0
    # The default 40-value scan on a toy embedding may or may not satisfy the
    # rule; both outcomes write their outputs.
    assert main(["calibrate", "--run-dir", run]) in (0, 1)
    calibration = json.loads((tmp_path / "run" / "calibration.json").read_text())
    assert len(calibration["scan"]) == 40
    assert main(["fit", "--run-dir", run, "--n-jobs", "1"]) == 0
    capsys.readouterr()
    assert main(["status", "--run-dir", run]) == 0
    assert "Configurations complete: 4 of 4" in capsys.readouterr().out


def test_a_stage_that_would_overwrite_exits_two(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(STUDIES, "heca", small_heca_study(tmp_path / "data"))
    run = str(tmp_path / "run")
    assert main(["embed", "--run-dir", run]) == 0
    assert main(["embed", "--run-dir", run]) == 2
    assert "--force" in capsys.readouterr().err
```

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.heca'`.

- [ ] Step 3: Implement

Create `src/benchmarks/heca.py`:

```python
"""Command-line entry point: python -m benchmarks.heca.

The full-scale hECA run in stages, one SLURM job each: embed, calibrate and
fit, plus status for a fit in progress. Every stage reads and writes one run
directory. Runbook: slurm/heca/README.md. Design:
docs/superpowers/specs/2026-10-01-heca-longleaf-runtime-design.md.
"""

import argparse
import sys
from pathlib import Path

from ._heca_calibration import run_calibrate
from ._heca_stages import (
    StageOutputExists,
    fit_status,
    format_status,
    run_embed,
    run_fit,
)

STAGES: tuple[str, ...] = ("embed", "calibrate", "fit", "status")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.heca",
        description="Run the full-scale hECA case study in stages.",
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite this stage's outputs."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="fit only: worker count. Default: one per physical core, capped "
        "by node memory over calibration's per-worker peak.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.stage == "embed":
            record = run_embed(args.run_dir, force=args.force)
            print(
                f"Embedded {record['n_cells']} cells "
                f"({record['wall_clock_s'] / 3600:.2f} h, cached={record['cached']})."
            )
        elif args.stage == "calibrate":
            record = run_calibrate(args.run_dir, force=args.force)
            if record["grid_error"] is not None:
                print(f"No grid proposed: {record['grid_error']}", file=sys.stderr)
                return 1
            grid = ", ".join(f"{value:g}" for value in record["proposed_grid"])
            print(f"Proposed grid: {grid}")
            print(f"Projected core-hours: {record['projection']['core_hours']:.0f}")
            for n_jobs, hours in record["projection"]["wall_clock_hours"].items():
                print(f"  {n_jobs:>4} workers: {hours:.1f} h")
        elif args.stage == "fit":
            record = run_fit(args.run_dir, n_jobs=args.n_jobs, force=args.force)
            print(
                f"Fit {record['n_cells']} cells on {record['n_jobs']} workers in "
                f"{record['wall_clock_s'] / 3600:.2f} h."
            )
        else:
            print(format_status(fit_status(args.run_dir)))
    except StageOutputExists as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca.py -q`
Expected: all PASS.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add src/benchmarks/heca.py tests/benchmarks/test_heca.py
git commit -m "feat(benchmarks): add python -m benchmarks.heca

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Reading a run on the laptop

Files:
- Create: `src/benchmarks/_heca_report.py`
- Test: `tests/benchmarks/test_heca_report.py`

Interfaces:
- Consumes: `label_settings`, `read_json`, `resolve_study` (Tasks 5, 8); `calibration_costs`, `project_runtime` (Task 6); `read_timings` (Task 4); `carve_cache_path`, `study_resolution_grids` (Task 2).
- Produces (in `benchmarks._heca_report`):
  - `HecaRun` (dataclass, fields as below).
  - `load_heca_run(run_dir, *, X, study=None) -> HecaRun`.
  - `slurm_duration_seconds(text) -> float`, `slurm_memory_bytes(text) -> float`, `parse_sacct(text) -> pd.DataFrame`, which adds `<column>_s` for Elapsed and TotalCPU and `<column>_bytes` for MaxRSS and AveRSS.
  - `fit_cpu_hours(run) -> float`.
  - `runtime_table(run) -> pd.DataFrame` with columns stage, wall_clock_h, cpu_h, workers, peak_memory_gb, memory_source, sampled_peak_memory_gb.
  - `component_table(run) -> pd.DataFrame` with columns component, setting, core_hours, share.
  - `selection_table(run, obs, *, rule="1se") -> pd.DataFrame` with columns measure, setting, resolution, n_clusters_observed, n_clusters_consensus, ari_organ, ari_cell_type.
  - `study_composition(labels, obs) -> pd.DataFrame`.
  - `projection_vs_actual(run) -> dict`.

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/test_heca_report.py`:

```python
"""Tests for reading a copied-back hECA run."""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from benchmarks._heca_calibration import run_calibrate
from benchmarks._heca_report import (
    component_table,
    load_heca_run,
    parse_sacct,
    projection_vs_actual,
    runtime_table,
    selection_table,
    slurm_duration_seconds,
    slurm_memory_bytes,
    study_composition,
)
from benchmarks._heca_stages import run_embed, run_fit
from benchmarks._studies import load_study
from tests.benchmarks._helpers import small_heca_study

SACCT = (
    "JobID|Elapsed|TotalCPU|MaxRSS|AveRSS|NCPUS\n"
    "123.0|1-02:00:00|20-00:00:00|200G|180G|48\n"
)


@pytest.mark.parametrize(
    "text, seconds",
    [("1-02:00:00", 93_600.0), ("02:03:04", 7_384.0), ("05:06.500", 306.5)],
)
def test_slurm_durations(text, seconds):
    assert slurm_duration_seconds(text) == pytest.approx(seconds)


@pytest.mark.parametrize(
    "text, size",
    [("200G", 200 * 1024**3), ("512K", 512 * 1024), ("1234", 1234.0)],
)
def test_slurm_memory(text, size):
    assert slurm_memory_bytes(text) == pytest.approx(size)


def test_blank_slurm_fields_are_nan():
    assert math.isnan(slurm_duration_seconds(""))
    assert math.isnan(slurm_memory_bytes(""))


def test_parse_sacct_converts_the_fields():
    frame = parse_sacct(SACCT)
    row = frame.iloc[0]
    assert row["JobID"] == "123.0"
    assert row["TotalCPU_s"] == pytest.approx(20 * 86_400)
    assert row["MaxRSS_bytes"] == pytest.approx(200 * 1024**3)


def test_study_composition_by_hand():
    obs = pd.DataFrame({"study_id": ["a", "a", "b", "b", "b"]})
    table = study_composition(np.array([0, 0, 0, 1, 1]), obs)
    assert table.loc[0, "a"] == pytest.approx(2 / 3)
    assert table.loc[1, "b"] == pytest.approx(1.0)
    assert table.loc[0, "n_cells"] == 3
    assert table.loc[0, "dominant_share"] == pytest.approx(2 / 3)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A small run through every stage, with a sacct.txt as SLURM writes it."""
    root = tmp_path_factory.mktemp("heca")
    study = small_heca_study(root / "data")
    run = root / "run"
    run_embed(run, study=study)
    run_calibrate(run, study=study, scan=(0.5, 1.0, 2.0))
    run_fit(run, study=study, n_jobs=1, sample_interval=0.05)
    (run / "fit" / "sacct.txt").write_text(SACCT)
    X, _, meta = load_study(study)
    return study, run, X, meta


def test_load_heca_run_reads_every_artifact(finished):
    study, run_dir, X, _ = finished
    run = load_heca_run(run_dir, X=X, study=study)
    assert not run.carve.estimator_results_.empty
    assert set(run.leiden["setting"]) == {"leiden_15", "leiden_50"}
    assert run.sacct is not None
    assert len(run.scan) == 6


def test_the_fit_is_found_after_the_studys_grid_moves_on(finished):
    study, run_dir, X, _ = finished
    moved = replace(study, resolutions=(0.5, 3.0))
    run = load_heca_run(run_dir, X=X, study=moved)
    assert run.study.resolutions == (1.0, 2.0)


def test_runtime_table_prefers_slurms_memory(finished):
    study, run_dir, X, _ = finished
    table = runtime_table(load_heca_run(run_dir, X=X, study=study))
    assert list(table["stage"]) == ["embedding", "CARVE fit"]
    fit = table.set_index("stage").loc["CARVE fit"]
    assert fit["memory_source"] == "slurm"
    assert fit["peak_memory_gb"] == pytest.approx(200 * 1024**3 / 1e9)
    assert fit["cpu_h"] == pytest.approx(20 * 24)
    assert fit["sampled_peak_memory_gb"] > 0


def test_component_shares_sum_to_one_with_the_overhead(finished):
    study, run_dir, X, _ = finished
    table = component_table(load_heca_run(run_dir, X=X, study=study))
    assert set(table["component"]) == {
        "neighbor search",
        "graph assembly",
        "Leiden",
        "forest fit",
        "forest predict",
        "CARVE overhead",
    }
    assert table["share"].sum() == pytest.approx(1.0)


def test_selection_table_scores_both_criteria(finished):
    study, run_dir, X, meta = finished
    table = selection_table(load_heca_run(run_dir, X=X, study=study), meta["obs"])
    assert list(table["measure"]) == ["stability", "generalizability"]
    assert set(table["setting"]) <= {"leiden_15", "leiden_50"}
    assert table["ari_organ"].between(-1, 1).all()


def test_projection_vs_actual_reports_both(finished):
    study, run_dir, X, _ = finished
    out = projection_vs_actual(load_heca_run(run_dir, X=X, study=study))
    assert out["n_jobs"] == 1
    assert out["projected_wall_clock_h"] > 0
    assert out["actual_wall_clock_h"] > 0
```

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca_report.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks._heca_report'`.

- [ ] Step 3: Implement

Create `src/benchmarks/_heca_report.py`:

```python
"""Read a copied-back hECA run and summarize it for the notebook.

Nothing here fits: the run directory holds what the cluster computed, and
figures._heca_* draw it. This module plays the part _cusanovich_compare
plays for its study.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from carve import CARVE

from ._artifacts import check_fingerprint
from ._heca_calibration import calibration_costs, project_runtime
from ._heca_stages import label_settings, read_json, resolve_study
from ._studies import carve_cache_path, study_resolution_grids
from ._timing import read_timings
from ._types import Study

_MEMORY_UNITS: dict[str, int] = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


@dataclass
class HecaRun:
    """Everything one run directory holds, with the fitted CARVE loaded."""

    run_dir: Path
    study: Study
    env: dict[str, Any]
    embed: dict[str, Any]
    calibration: dict[str, Any]
    scan: pd.DataFrame
    started: dict[str, Any]
    runtime: dict[str, Any]
    leiden: pd.DataFrame
    forest: pd.DataFrame
    memory: pd.DataFrame
    sacct: pd.DataFrame | None
    carve: CARVE


def slurm_duration_seconds(text: str) -> float:
    """A SLURM duration, [DD-][HH:]MM:SS[.mmm], in seconds."""
    text = str(text).strip()
    if not text:
        return float("nan")
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    parts = [float(part) for part in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts
    return days * 86_400 + hours * 3_600 + minutes * 60 + seconds


def slurm_memory_bytes(text: str) -> float:
    """A SLURM memory field such as 200G or 512K, in bytes."""
    text = str(text).strip()
    if not text:
        return float("nan")
    unit = text[-1]
    if unit in _MEMORY_UNITS:
        return float(text[:-1]) * _MEMORY_UNITS[unit]
    return float(text)


def parse_sacct(text: str) -> pd.DataFrame:
    """sacct -P output (pipe-separated, header first) as a frame."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    header = lines[0].split("|")
    frame = pd.DataFrame(
        [dict(zip(header, line.split("|"), strict=False)) for line in lines[1:]],
        columns=header,
    )
    for column in ("Elapsed", "TotalCPU"):
        if column in frame:
            frame[f"{column}_s"] = frame[column].map(slurm_duration_seconds)
    for column in ("MaxRSS", "AveRSS"):
        if column in frame:
            frame[f"{column}_bytes"] = frame[column].map(slurm_memory_bytes)
    return frame


def load_heca_run(run_dir: Path, *, X: np.ndarray, study: Study | None = None) -> HecaRun:
    """Read a run directory and load its fit.

    The study is rebuilt with the grid and resample count the fit recorded
    in started.json, so the fit is found even after STUDIES moves on. X is
    the embedding the fit ran on, checked against the cache's fingerprint.
    """
    run_dir = Path(run_dir)
    fit_dir = run_dir / "fit"
    started = read_json(fit_dir / "started.json")
    study = replace(
        resolve_study(study),
        resolutions=tuple(float(r) for r in started["resolutions"]),
        n_resamples=int(started["n_resamples"]),
    )
    cache_path = carve_cache_path(
        study,
        root=fit_dir,
        model_grids=study_resolution_grids(study),
        n_resamples=study.n_resamples,
    )
    check_fingerprint(cache_path, X)
    carve = CARVE.load(str(cache_path))
    carve.X_ = np.asarray(X)

    timings = read_timings(fit_dir / "timings")
    sacct_path = fit_dir / "sacct.txt"
    return HecaRun(
        run_dir=run_dir,
        study=study,
        env=read_json(run_dir / "env.json"),
        embed=read_json(run_dir / "embed.json"),
        calibration=read_json(run_dir / "calibration.json"),
        scan=pd.read_csv(run_dir / "calibration.csv"),
        started=started,
        runtime=read_json(fit_dir / "runtime.json"),
        leiden=label_settings(timings["leiden"], study, n_cells=int(started["n_cells"])),
        forest=timings["forest"],
        memory=pd.read_csv(fit_dir / "memory.csv"),
        sacct=parse_sacct(sacct_path.read_text()) if sacct_path.exists() else None,
        carve=carve,
    )


def fit_cpu_hours(run: HecaRun) -> float:
    """The fit step's total CPU time from SLURM, or NaN without sacct.txt."""
    if run.sacct is None or run.sacct.empty:
        return float("nan")
    return float(run.sacct["TotalCPU_s"].iloc[0]) / 3600


def runtime_table(run: HecaRun) -> pd.DataFrame:
    """Wall-clock time, CPU time, workers and peak memory: embedding and fit.

    The fit's peak memory is SLURM's MaxRSS when sacct.txt was copied back,
    else the largest sampled total of the fit process and its workers; the
    sampled value is reported beside it either way.
    """
    sampled = float(
        (run.memory["own_rss_bytes"] + run.memory["children_rss_bytes"]).max()
    )
    if run.sacct is not None and not run.sacct.empty:
        fit_memory, fit_source = float(run.sacct["MaxRSS_bytes"].iloc[0]), "slurm"
    else:
        fit_memory, fit_source = sampled, "sampled"
    return pd.DataFrame(
        [
            {
                "stage": "embedding",
                "wall_clock_h": run.embed["wall_clock_s"] / 3600,
                "cpu_h": float("nan"),
                "workers": 1,
                "peak_memory_gb": run.embed["peak_rss_bytes"] / 1e9,
                "memory_source": "process",
                "sampled_peak_memory_gb": float("nan"),
            },
            {
                "stage": "CARVE fit",
                "wall_clock_h": run.runtime["wall_clock_s"] / 3600,
                "cpu_h": fit_cpu_hours(run),
                "workers": int(run.runtime["n_jobs"]),
                "peak_memory_gb": fit_memory / 1e9,
                "memory_source": fit_source,
                "sampled_peak_memory_gb": sampled / 1e9,
            },
        ]
    )


def component_table(run: HecaRun) -> pd.DataFrame:
    """Core-hours per component, and each one's share of the fit's CPU time.

    The graph and Leiden components are per setting; the forest is for the
    whole fit, since the classifier cannot tell which configuration it
    serves. With sacct.txt, CARVE's own overhead (ARI scoring, consensus
    accumulation, scheduling) is the rest of the step's CPU time; without
    it, shares are of the timed components alone. Components are wall-clock
    seconds in single-threaded workers, which equals their CPU time when the
    fit ran one worker per physical core.
    """
    rows = []
    for setting, part in run.leiden.groupby("setting", sort=False):
        for column, name in (
            ("knn_s", "neighbor search"),
            ("graph_s", "graph assembly"),
            ("leiden_s", "Leiden"),
        ):
            rows.append(
                {
                    "component": name,
                    "setting": setting,
                    "core_hours": float(part[column].sum()) / 3600,
                }
            )
    for call, name in (("fit", "forest fit"), ("predict", "forest predict")):
        seconds = run.forest.loc[run.forest["call"] == call, "seconds"].sum()
        rows.append(
            {"component": name, "setting": "all", "core_hours": float(seconds) / 3600}
        )
    table = pd.DataFrame(rows)
    cpu_hours = fit_cpu_hours(run)
    if np.isnan(cpu_hours):
        table["share"] = table["core_hours"] / table["core_hours"].sum()
        return table
    overhead = {
        "component": "CARVE overhead",
        "setting": "all",
        "core_hours": cpu_hours - float(table["core_hours"].sum()),
    }
    table = pd.concat([table, pd.DataFrame([overhead])], ignore_index=True)
    table["share"] = table["core_hours"] / cpu_hours
    return table


def selection_table(
    run: HecaRun, obs: pd.DataFrame, *, rule: str = "1se"
) -> pd.DataFrame:
    """The configuration each criterion selects, and its agreement with the
    two references."""
    results = run.carve.estimator_results_
    rows = []
    for measure in ("stability", "generalizability"):
        estimator = run.carve.get_estimator(measure=measure, rule=rule)
        resolution = float(run.carve.get_sweep_value(measure=measure, rule=rule))
        labels = run.carve.get_labels(measure=measure, rule=rule)
        match = results[
            (results["n_neighbors"] == estimator.n_neighbors)
            & np.isclose(results["resolution"].astype(float), resolution)
        ]
        rows.append(
            {
                "measure": measure,
                "setting": f"leiden_{estimator.n_neighbors}",
                "resolution": resolution,
                "n_clusters_observed": float(match["n_clusters_observed"].iloc[0]),
                "n_clusters_consensus": int(np.unique(labels).size),
                "ari_organ": float(adjusted_rand_score(obs["organ"], labels)),
                "ari_cell_type": float(adjusted_rand_score(obs["cell_type"], labels)),
            }
        )
    return pd.DataFrame(rows)


def study_composition(labels: Sequence[int] | np.ndarray, obs: pd.DataFrame) -> pd.DataFrame:
    """Each cluster's cells by study of origin, as fractions of the cluster.

    A cluster drawn mostly from one study may be recovering its study rather
    than biology: the pooled embedding is not batch-corrected.
    """
    clusters = pd.Series(np.asarray(labels), name="cluster")
    studies = pd.Series(obs["study_id"].to_numpy(), name="study_id")
    table = pd.crosstab(clusters, studies, normalize="index")
    table["n_cells"] = clusters.value_counts().reindex(table.index)
    table["dominant_share"] = table.drop(columns="n_cells").max(axis=1)
    return table.sort_values("n_cells", ascending=False)


def projection_vs_actual(run: HecaRun) -> dict[str, Any]:
    """Calibration's projection at the fit's worker count, beside the fit."""
    costs = calibration_costs(
        run.calibration, run.scan, resolutions=run.study.resolutions
    )
    n_jobs = int(run.runtime["n_jobs"])
    projection = project_runtime(
        costs,
        n_resolutions=len(run.study.resolutions),
        n_resamples=run.study.n_resamples,
        n_train=int(run.calibration["n_train"]),
        n_test=int(run.calibration["n_test"]),
        n_jobs_options=(n_jobs,),
    )
    return {
        "n_jobs": n_jobs,
        "projected_wall_clock_h": projection["wall_clock_hours"][str(n_jobs)],
        "actual_wall_clock_h": run.runtime["wall_clock_s"] / 3600,
    }
```

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca_report.py -q`
Expected: all PASS.

If `selection_table` fails because `get_estimator` returns an estimator without the configured `n_neighbors`, read `CARVE.get_estimator` (`src/carve/api.py:1036`). Select the row from `estimator_results_` by the method `get_sweep_value` resolves; do not change carve.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add src/benchmarks/_heca_report.py tests/benchmarks/test_heca_report.py
git commit -m "feat(benchmarks): read a copied-back hECA run into runtime and result tables

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Figures

Files:
- Create: `src/benchmarks/figures/_heca_calibration.py`, `src/benchmarks/figures/_heca_carve.py`, `src/benchmarks/figures/_heca_runtime.py`
- Delete: `src/benchmarks/figures/_heca_results.py`, `src/benchmarks/figures/_study_scaling.py`, `tests/benchmarks/figures/test_heca_results.py`, `tests/benchmarks/figures/test_study_scaling.py`
- Modify: `src/benchmarks/figures/__init__.py`, `tests/benchmarks/figures/test_figure_contract.py`
- Test: `tests/benchmarks/figures/test_heca_calibration.py`, `tests/benchmarks/figures/test_heca_carve.py`, `tests/benchmarks/figures/test_heca_runtime.py`

Interfaces:
- Consumes: a calibration scan frame (Task 6 columns); a fitted CARVE; `component_table` output and `memory.csv` (Tasks 7, 10).
- Produces:
  - `figure_heca_calibration(scan, *, grid, lower_target, upper_target, save=True, out_dir=None, save_name="heca_calibration.png") -> Figure`.
  - `figure_heca_carve(carve, *, resolutions, rule="1se", save=True, out_dir=None, save_name="heca_carve.png") -> Figure`.
  - `figure_heca_runtime(components, memory, *, save=True, out_dir=None, save_name="heca_runtime.png") -> Figure`.
  - In `figures._heca_carve`: the constants `MARKER_SIZE = 3.0`, `AXIS_LABELS = ("UMAP 1", "UMAP 2")` and `DEFAULT_SCATTER_SUBSAMPLE = 50_000`.

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/figures/test_heca_calibration.py`:

```python
"""Tests for the hECA calibration figure."""

import pandas as pd
import pytest
from matplotlib.figure import Figure

from benchmarks.figures import figure_heca_calibration


def _scan(settings=("leiden_15", "leiden_50")):
    rows = []
    for offset, setting in enumerate(settings):
        for resolution, n in ((0.01, 3), (0.1, 8), (1.0, 60), (10.0, 300)):
            rows.append(
                {
                    "setting": setting,
                    "resolution": resolution,
                    "n_clusters": n + offset,
                    "ari_organ": 0.5,
                    "ari_cell_type": 0.3,
                }
            )
    return pd.DataFrame(rows)


def test_draws_counts_and_agreement_and_saves(tmp_path):
    fig = figure_heca_calibration(
        _scan(), grid=(0.1, 1.0), lower_target=5, upper_target=108, out_dir=tmp_path
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert all(ax.get_xscale() == "log" for ax in fig.axes)
    assert (tmp_path / "heca_calibration.png").is_file()


def test_an_empty_scan_raises():
    with pytest.raises(ValueError, match="empty"):
        figure_heca_calibration(
            _scan().iloc[:0], grid=(0.1, 1.0), lower_target=5, upper_target=108,
            save=False,
        )


def test_more_settings_than_colors_raises():
    with pytest.raises(ValueError, match="settings"):
        figure_heca_calibration(
            _scan(("a", "b", "c")), grid=(0.1, 1.0), lower_target=5,
            upper_target=108, save=False,
        )
```

Create `tests/benchmarks/figures/test_heca_carve.py`:

```python
"""Tests for the hECA CARVE figure."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from benchmarks._estimators import resolution_grids
from benchmarks._types import EstimatorSpec
from benchmarks.figures import figure_heca_carve
from carve import CARVE


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X = centers[rng.integers(0, 3, 240)] + rng.normal(0, 0.6, (240, 2))
    grids = []
    for k in (10, 20):
        spec = EstimatorSpec(name="leiden", params=(("n_neighbors", k),))
        grids += resolution_grids(spec, (0.5, 1.0))
    return CARVE(estimator_param_grids=grids, n_resamples=3, random_state=0).fit(X)


def test_draws_three_panels_on_a_log_resolution_axis(fitted, tmp_path):
    fig = figure_heca_carve(fitted, resolutions=(0.5, 1.0), out_dir=tmp_path)
    assert isinstance(fig, Figure)
    panels = fig.axes[:3]
    assert len(panels) == 3
    assert all(ax.get_xscale() == "log" for ax in panels)
    assert (tmp_path / "heca_carve.png").is_file()
```

Create `tests/benchmarks/figures/test_heca_runtime.py`:

```python
"""Tests for the hECA runtime figure."""

import pandas as pd
import pytest
from matplotlib.figure import Figure

from benchmarks.figures import figure_heca_runtime


def _components():
    rows = []
    for setting in ("leiden_15", "leiden_50"):
        for name, hours in (
            ("neighbor search", 30.0),
            ("graph assembly", 5.0),
            ("Leiden", 8.0),
        ):
            rows.append({"component": name, "setting": setting, "core_hours": hours})
    rows += [
        {"component": "forest fit", "setting": "all", "core_hours": 20.0},
        {"component": "forest predict", "setting": "all", "core_hours": 1.0},
        {"component": "CARVE overhead", "setting": "all", "core_hours": 4.0},
    ]
    table = pd.DataFrame(rows)
    table["share"] = table["core_hours"] / table["core_hours"].sum()
    return table


def _memory():
    return pd.DataFrame(
        {
            "elapsed_s": [0.0, 3600.0, 7200.0],
            "own_rss_bytes": [2e9, 3e9, 3e9],
            "children_rss_bytes": [0.0, 50e9, 60e9],
            "n_children": [0, 24, 24],
        }
    )


def test_draws_three_panels_and_saves(tmp_path):
    fig = figure_heca_runtime(_components(), _memory(), out_dir=tmp_path)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    assert (tmp_path / "heca_runtime.png").is_file()


def test_empty_components_raise():
    with pytest.raises(ValueError, match="empty"):
        figure_heca_runtime(_components().iloc[:0], _memory(), save=False)
```

In `tests/benchmarks/figures/test_figure_contract.py`, in `EXPECTED`, remove `"figure_heca_results"` and `"figure_study_scaling"`, and add `"figure_heca_calibration"`, `"figure_heca_carve"` and `"figure_heca_runtime"`.

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/figures/test_heca_calibration.py tests/benchmarks/figures/test_heca_carve.py tests/benchmarks/figures/test_heca_runtime.py tests/benchmarks/figures/test_figure_contract.py -q`
Expected: FAIL with `ImportError: cannot import name 'figure_heca_calibration'`.

- [ ] Step 3: Implement

Create `src/benchmarks/figures/_heca_calibration.py`:

```python
"""The hECA calibration scan: cluster count and agreement over resolution.

Shows what the committed grid was chosen from. The left panel draws each
setting's cluster count on the training split, with the organ count and
twice the cell-type count as the rule's targets; the right panel draws
agreement with organ (solid) and cell type (dashed). The committed grid's
range is shaded on both.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._theme import (
    ESTIMATOR_COLORS,
    FALLBACK_COLOR,
    FOREGROUND_COLOR,
    REFERENCE_LINEWIDTH,
    save_figure,
    theme_context,
)
from ._paths import CASE_STUDY_DIR, figure_path


def figure_heca_calibration(
    scan: pd.DataFrame,
    *,
    grid: Sequence[float],
    lower_target: int,
    upper_target: int,
    save: bool = True,
    out_dir: Path | None = None,
    save_name: str = "heca_calibration.png",
) -> Figure:
    """Draw the calibration scan with the rule's targets and the grid's range."""
    if scan.empty:
        raise ValueError("figure_heca_calibration needs a non-empty scan.")
    settings = list(dict.fromkeys(scan["setting"]))
    if len(settings) > len(ESTIMATOR_COLORS):
        raise ValueError(
            f"{len(settings)} settings but {len(ESTIMATOR_COLORS)} estimator "
            "colors in the theme."
        )

    with theme_context():
        fig, (count_ax, ari_ax) = plt.subplots(1, 2, figsize=(10.0, 3.6))
        for color, setting in zip(ESTIMATOR_COLORS, settings, strict=False):
            rows = scan[scan["setting"] == setting].sort_values("resolution")
            count_ax.plot(
                rows["resolution"], rows["n_clusters"], marker="o", color=color,
                label=setting,
            )
            ari_ax.plot(
                rows["resolution"], rows["ari_organ"], color=color, linestyle="-",
                label=f"{setting}, organ",
            )
            ari_ax.plot(
                rows["resolution"], rows["ari_cell_type"], color=color,
                linestyle="--", label=f"{setting}, cell type",
            )
        for target in (lower_target, upper_target):
            count_ax.axhline(
                target, color=FOREGROUND_COLOR, linewidth=REFERENCE_LINEWIDTH,
                linestyle=":",
            )
        for ax in (count_ax, ari_ax):
            ax.set_xscale("log")
            ax.axvspan(min(grid), max(grid), color=FALLBACK_COLOR, alpha=0.15, lw=0)
            ax.set_xlabel("Resolution")
        count_ax.set_yscale("log")
        count_ax.set_ylabel("Clusters on the training split")
        count_ax.legend()
        ari_ax.set_ylabel("ARI against the reference")
        ari_ax.legend(fontsize="small")
        fig.tight_layout()
        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
```

Create `src/benchmarks/figures/_heca_carve.py`:

```python
"""CARVE's own view of the hECA fit: both criteria and the cluster count.

Stability and generalizability over resolution, one line per setting, and
the observed cluster count over resolution, all drawn by CARVE's plotting
methods on a log resolution axis.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .._theme import save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path

#: The notebook's reference scatter: a declared fixed subsample of cells,
#: because half a million points do not rasterize legibly, drawn on a UMAP
#: of those cells.
MARKER_SIZE = 3.0
AXIS_LABELS = ("UMAP 1", "UMAP 2")
DEFAULT_SCATTER_SUBSAMPLE = 50_000


def figure_heca_carve(
    carve: Any,
    *,
    resolutions: Sequence[float],
    rule: str = "1se",
    save: bool = True,
    out_dir: Path | None = None,
    save_name: str = "heca_carve.png",
) -> Figure:
    """Draw stability, generalizability and cluster count over resolution."""
    with theme_context():
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
        carve.plot_metric_over_n_clusters(
            measure="stability", rule=rule, ax=axes[0], title="Stability"
        )
        carve.plot_metric_over_n_clusters(
            measure="generalizability", rule=rule, ax=axes[1],
            title="Generalizability",
        )
        carve.plot_n_clusters_over_sweep(
            measure="stability", rule=rule, ax=axes[2], title="Observed clusters"
        )
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xticks(list(resolutions), labels=[f"{r:g}" for r in resolutions])
            ax.tick_params(axis="x", labelrotation=90)
            ax.minorticks_off()
        fig.tight_layout()
        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
```

Create `src/benchmarks/figures/_heca_runtime.py`:

```python
"""Where the hECA fit's CPU time went, and its memory over time.

Left: the whole fit's core-hours by component, as one stacked bar. Middle:
the graph and Leiden components per setting. Right: the resident memory of
the fit process and its workers, as sampled during the run.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._theme import FOREGROUND_COLOR, cluster_colors, save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path


def figure_heca_runtime(
    components: pd.DataFrame,
    memory: pd.DataFrame,
    *,
    save: bool = True,
    out_dir: Path | None = None,
    save_name: str = "heca_runtime.png",
) -> Figure:
    """Draw the component breakdown and the memory curve.

    components is _heca_report.component_table's frame; memory is the fit's
    memory.csv.
    """
    if components.empty:
        raise ValueError("figure_heca_runtime needs a non-empty component table.")

    totals = components.groupby("component", sort=False)["core_hours"].sum()
    palette = dict(zip(totals.index, cluster_colors(len(totals)), strict=True))

    with theme_context():
        fig, (total_ax, setting_ax, memory_ax) = plt.subplots(
            1, 3, figsize=(15.0, 3.8)
        )

        left = 0.0
        for name, hours in totals.items():
            total_ax.barh(0, hours, left=left, color=palette[name], label=name)
            left += hours
        total_ax.set_yticks([])
        total_ax.set_xlabel("Core-hours, whole fit")
        total_ax.legend(
            fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3
        )

        per_setting = components[components["setting"] != "all"]
        settings = list(dict.fromkeys(per_setting["setting"]))
        bottoms = [0.0] * len(settings)
        for name in dict.fromkeys(per_setting["component"]):
            rows = per_setting[per_setting["component"] == name].set_index("setting")
            heights = [float(rows["core_hours"].get(s, 0.0)) for s in settings]
            setting_ax.bar(
                settings, heights, bottom=bottoms, color=palette[name], label=name
            )
            bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
        setting_ax.set_ylabel("Core-hours")
        setting_ax.set_title("Graph and Leiden, by setting")

        hours = memory["elapsed_s"] / 3600
        total_gb = (memory["own_rss_bytes"] + memory["children_rss_bytes"]) / 1e9
        memory_ax.plot(hours, total_gb, color=FOREGROUND_COLOR)
        memory_ax.set_xlabel("Hours since the fit started")
        memory_ax.set_ylabel("Resident memory (GB)")

        fig.tight_layout()
        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
```

Delete the replaced modules and their tests:

```bash
git rm src/benchmarks/figures/_heca_results.py src/benchmarks/figures/_study_scaling.py tests/benchmarks/figures/test_heca_results.py tests/benchmarks/figures/test_study_scaling.py
```

In `src/benchmarks/figures/__init__.py`:
- Remove `from ._heca_results import figure_heca_results` and `from ._study_scaling import figure_study_scaling`.
- Add `from ._heca_calibration import figure_heca_calibration`, `from ._heca_carve import figure_heca_carve` and `from ._heca_runtime import figure_heca_runtime`.
- In `__all__`, replace `"figure_heca_results"` and `"figure_study_scaling"` with `"figure_heca_calibration"`, `"figure_heca_carve"` and `"figure_heca_runtime"`, keeping the list sorted.
- In the module docstring, after the sentence about `_reference_scatter`, add: `_heca_calibration, _heca_carve and _heca_runtime draw the hECA notebook's views of a run on Longleaf.`

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/figures -q`
Expected: all PASS.

Run: `grep -rn "_heca_results\|figure_heca_results\|_study_scaling\|figure_study_scaling\|subsample_inputs" src tests`
Expected: no output.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check --fix src/ && .venv/bin/ruff format src/
git add -A src/benchmarks/figures tests/benchmarks/figures
git commit -m "feat(figures): draw hECA's calibration, CARVE result and runtime

The composite and ladder figures go with the runs they drew.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: The notebook reads a run

Files:
- Rewrite: `notebooks/case_studies/hECA.ipynb`
- Modify: `tests/benchmarks/test_notebooks.py`

Interfaces:
- Consumes: `load_heca_run`, `runtime_table`, `component_table`, `selection_table`, `study_composition`, `projection_vs_actual` (Task 10); the three figure functions and `AXIS_LABELS`, `DEFAULT_SCATTER_SUBSAMPLE`, `MARKER_SIZE` (Task 11).

- [ ] Step 1: Write the failing tests

In `tests/benchmarks/test_notebooks.py`, replace `test_heca_notebook_reads_its_config_from_studies` and `test_heca_notebook_draws_one_umap_throughout` with:

```python
def test_heca_notebook_reads_its_config_from_studies():
    source = _code(NOTEBOOKS["heca"])
    assert 'STUDIES["heca"]' in source
    assert "load_heca_run(" in source


def test_heca_notebook_computes_nothing():
    # The fit runs on Longleaf; the notebook only reads its run directory.
    source = _code(NOTEBOOKS["heca"])
    for forbidden in (
        "fit_or_load_carve(",
        "CARVE(",
        "run_fit(",
        "run_calibrate(",
        "cvi_sweep(",
    ):
        assert forbidden not in source


def test_heca_notebook_draws_one_umap_throughout():
    source = _code(NOTEBOOKS["heca"])
    assert "UMAP(random_state=RANDOM_SEED)" in source
    assert "figure_reference_scatter(" in source
    assert "plt.subplots" not in source
    for figure in (
        "figure_heca_calibration(",
        "figure_heca_carve(",
        "figure_heca_runtime(",
    ):
        assert figure in source
```

In the comment block above `test_cusanovich_notebook_draws_the_source_tsne_and_writes_its_tables`, replace the sentence `hECA ships nothing, so its notebook computes a UMAP the way Levine_32dim.ipynb computes its t-SNE and hands that same embedding to prepare_composite.` with `hECA ships nothing, so its notebook computes a UMAP of the cells it draws.`

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_notebooks.py -q -k heca`
Expected: FAIL. The old notebook has no `load_heca_run(` and still calls `fit_or_load_carve(`.

- [ ] Step 3: Rebuild the notebook

Run this builder from the repository root. It overwrites `notebooks/case_studies/hECA.ipynb`, without outputs:

```bash
PYTHONPATH=src .venv/bin/python - <<'EOF'
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = [
    new_markdown_cell(
        "# hECA v2.0 -- Pooled scATAC-seq Across Five Organs\n\n"
        "Source: Chen et al. 2025, Scientific Data. Zenodo record 15627886 "
        "(https://zenodo.org/records/15627886).\n\n"
        "Five organs from the hECA v2.0 ATAC collection are pooled: Lung, Brain, "
        "Kidney, Heart, and Thymus, 695,304 annotated cells from 8 source studies "
        "(cells the source left Unclassified are dropped). CARVE sweeps Leiden "
        "resolution on a 15-neighbor and a 50-neighbor graph. Results are scored "
        "against organ and against the uHAF-harmonized cell type.\n\n"
        "The pooled embedding is not batch-corrected, so a cluster may partly "
        "recover its study of origin; Section 2 reports study composition per "
        "cluster."
    ),
    new_markdown_cell(
        "---\n\n## 0. Configuration\n\n"
        "This notebook fits nothing. The fit ran on UNC's Longleaf cluster through "
        "python -m benchmarks.heca (runbook: slurm/heca/README.md), and its run "
        "directory was copied back under results/runs/heca/. RUN_DIR picks the "
        "newest copied-back run; set it explicitly to pin one. The embedding is "
        "the loader's cache, copied back into data/hECA/."
    ),
    new_code_cell(
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import umap\n\n"
        "from benchmarks._heca_report import (\n"
        "    component_table,\n"
        "    load_heca_run,\n"
        "    projection_vs_actual,\n"
        "    runtime_table,\n"
        "    selection_table,\n"
        "    study_composition,\n"
        ")\n"
        "from benchmarks._studies import STUDIES, load_study\n"
        "from benchmarks.figures import (\n"
        "    figure_heca_calibration,\n"
        "    figure_heca_carve,\n"
        "    figure_heca_runtime,\n"
        "    figure_reference_scatter,\n"
        ")\n"
        "from benchmarks.figures._heca_carve import (\n"
        "    AXIS_LABELS,\n"
        "    DEFAULT_SCATTER_SUBSAMPLE,\n"
        "    MARKER_SIZE,\n"
        ")\n"
        "from benchmarks.figures._paths import CASE_STUDY_DIR\n\n"
        "RANDOM_SEED = 42\n"
        "RUN_DIR = sorted(Path(\"../../results/runs/heca\").iterdir())[-1]\n\n"
        "OUT_DIR = CASE_STUDY_DIR\n"
        "OUT_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "study = STUDIES[\"heca\"]\n"
        "X, y, meta = load_study(study)\n"
        "run = load_heca_run(RUN_DIR, X=X, study=study)\n"
        "print(RUN_DIR.name, \"|\", meta[\"n_cells\"], \"cells across\", meta[\"organs\"])\n\n\n"
        "def show(fig):\n"
        "    display(fig)\n"
        "    plt.close(fig)"
    ),
    new_markdown_cell(
        "### 0.1 Reference Labels\n\n"
        "UMAP of a fixed subsample of the principal components the loader returns, "
        "colored by organ. The source ships no coordinates, so the embedding is "
        "computed here, on the cells the scatter draws."
    ),
    new_code_cell(
        "rng = np.random.default_rng(RANDOM_SEED)\n"
        "shown = np.sort(\n"
        "    rng.choice(X.shape[0], size=min(DEFAULT_SCATTER_SUBSAMPLE, X.shape[0]), replace=False)\n"
        ")\n"
        "reference_embedding = umap.UMAP(random_state=RANDOM_SEED).fit_transform(X[shown])\n"
        "show(\n"
        "    figure_reference_scatter(\n"
        "        reference_embedding, y.to_numpy()[shown],\n"
        "        axis_labels=AXIS_LABELS, title=\"Reported Organ\",\n"
        "        marker_size=MARKER_SIZE, out_dir=OUT_DIR,\n"
        "        save_name=\"heca_reference_scatter.png\",\n"
        "    )\n"
        ")"
    ),
    new_markdown_cell(
        "---\n\n## 1. Calibration\n\n"
        "Leiden at 40 resolutions on the training split the fit's first resample "
        "draws, for both settings. The shaded range is the grid the fit ran; the "
        "dotted lines are the rule's targets, the organ count and twice the "
        "cell-type count."
    ),
    new_code_cell(
        "print(\"Proposed grid:\", run.calibration[\"proposed_grid\"])\n"
        "print(\"Fitted grid:  \", list(run.study.resolutions))\n"
        "show(\n"
        "    figure_heca_calibration(\n"
        "        run.scan, grid=run.study.resolutions,\n"
        "        lower_target=run.calibration[\"lower_target\"],\n"
        "        upper_target=run.calibration[\"upper_target\"],\n"
        "        out_dir=OUT_DIR,\n"
        "    )\n"
        ")"
    ),
    new_markdown_cell(
        "---\n\n## 2. CARVE Results\n\n"
        "Both criteria and the observed cluster count over resolution, then the "
        "configuration each criterion selects and its agreement with organ and "
        "cell type."
    ),
    new_code_cell(
        "show(figure_heca_carve(run.carve, resolutions=run.study.resolutions, out_dir=OUT_DIR))\n"
        "selection_table(run, meta[\"obs\"])"
    ),
    new_code_cell(
        "labels = run.carve.get_labels(measure=\"stability\", rule=\"1se\")\n"
        "study_composition(labels, meta[\"obs\"])"
    ),
    new_markdown_cell(
        "---\n\n## 3. Runtime\n\n"
        "Wall-clock time, CPU time and peak memory for the embedding and the fit; "
        "where the fit's CPU time went, by component; calibration's projection "
        "against the actual time; and the fit's memory over time."
    ),
    new_code_cell(
        "components = component_table(run)\n"
        "display(runtime_table(run))\n"
        "display(components)\n"
        "print(projection_vs_actual(run))\n"
        "show(figure_heca_runtime(components, run.memory, out_dir=OUT_DIR))"
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
nbformat.validate(nb)
nbformat.write(nb, "notebooks/case_studies/hECA.ipynb")
EOF
```

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_notebooks.py -q`
Expected: all PASS.

- [ ] Step 5: Commit

```bash
git add notebooks/case_studies/hECA.ipynb tests/benchmarks/test_notebooks.py
git commit -m "docs(notebooks): rebuild hECA to read a Longleaf run directory

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: SLURM scripts and the runbook

Files:
- Create: `slurm/heca/README.md`, `slurm/heca/embed.sbatch`, `slurm/heca/calibrate.sbatch`, `slurm/heca/fit.sbatch`
- Modify: `src/benchmarks/README.md`
- Modify: `tests/benchmarks/test_heca.py`

Interfaces:
- Consumes: the CLI (Task 9). Each script requires `RUN_DIR` in its environment and is submitted from the repository root.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_heca.py` (add `import subprocess` and `from pathlib import Path` to its imports):

```python
SLURM_DIR = Path(__file__).resolve().parents[2] / "slurm" / "heca"


@pytest.mark.parametrize("stage", ["embed", "calibrate", "fit"])
def test_each_sbatch_script_runs_its_stage_and_parses(stage):
    script = SLURM_DIR / f"{stage}.sbatch"
    assert f"python -m benchmarks.heca {stage} --run-dir" in script.read_text()
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_the_fit_script_takes_a_whole_node_and_saves_its_accounting():
    text = (SLURM_DIR / "fit.sbatch").read_text()
    assert "#SBATCH --exclusive" in text
    assert '--cpus-per-task="${SLURM_CPUS_ON_NODE}"' in text
    assert "sacct -j" in text
    assert "fit/sacct.txt" in text


def test_calibration_runs_single_threaded():
    text = (SLURM_DIR / "calibrate.sbatch").read_text()
    assert "OMP_NUM_THREADS=1" in text


def test_the_runbook_names_every_script():
    text = (SLURM_DIR / "README.md").read_text()
    for stage in ("embed", "calibrate", "fit"):
        assert f"slurm/heca/{stage}.sbatch" in text
```

- [ ] Step 2: Run them to verify they fail

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca.py -q`
Expected: FAIL with `FileNotFoundError` for `slurm/heca/embed.sbatch`.

- [ ] Step 3: Write the scripts

Create `slurm/heca/embed.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=heca-embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=heca-embed-%j.out
#
# The pooled embedding of every annotated hECA cell: two chunked passes over
# the organ files, then normalization, peak selection and PCA in memory. Its
# cache in data/hECA/ serves every later stage. Submit from the repository
# root with RUN_DIR set; see slurm/heca/README.md.

set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR to the run directory}"
source .venv/bin/activate
python -m benchmarks.heca embed --run-dir "$RUN_DIR"
```

Create `slurm/heca/calibrate.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=heca-calibrate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=heca-calibrate-%j.out
#
# Scans Leiden resolution for both settings, times every component the fit
# repeats, proposes the grid and projects the fit's runtime. Every timed
# component runs single-threaded, as each worker of the fit does. Exits 1
# when no grid satisfies the rule; calibration.json then says why.

set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR to the run directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source .venv/bin/activate
python -m benchmarks.heca calibrate --run-dir "$RUN_DIR"
```

Create `slurm/heca/fit.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=heca-fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=11-00:00:00
#SBATCH --output=heca-fit-%j.out
#
# The CARVE fit on every cell, alone on one node so no other job shares its
# memory bandwidth. The stage picks its worker count from the node's
# physical cores and calibration.json. srun makes the fit a job step, so
# SLURM's accounting for it is final when the step ends and is saved beside
# the run. srun does not inherit --cpus-per-task from sbatch, so the step
# asks for every CPU on the node explicitly. Lower --time on the command
# line to the projection plus a margin to schedule sooner.

set -uo pipefail
: "${RUN_DIR:?Set RUN_DIR to the run directory}"
source .venv/bin/activate

status=0
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_ON_NODE}" \
    python -m benchmarks.heca fit --run-dir "$RUN_DIR" || status=$?
mkdir -p "$RUN_DIR/fit"
sacct -j "${SLURM_JOB_ID}.0" \
    --format=JobID,Elapsed,TotalCPU,MaxRSS,AveRSS,NCPUS -P > "$RUN_DIR/fit/sacct.txt"
exit "$status"
```

Create `slurm/heca/README.md` with exactly this content. It is a document, so no bold or italics.

````markdown
# hECA on Longleaf

The full-scale hECA run, 695,304 cells, as three SLURM jobs on UNC's Longleaf
cluster: embed, calibrate and fit. Design:
docs/superpowers/specs/2026-10-01-heca-longleaf-runtime-design.md. Commands
run from the repository root on Longleaf unless marked "on the laptop".

Longleaf facts used here, from https://help.rc.unc.edu/getting-started-on-longleaf/
(read 2026-10-01): SLURM; the general partition, which is the default, allows
jobs up to 11 days; home is capped at 50 GB; /work/users holds 8 TB per user;
the datamover partition is for transfers.

## 0. Access

- Request a Longleaf account from UNC Research Computing (see the page above).
- On the laptop, push the branch the run uses: `git push origin heca-longleaf`.

## 1. Code and environment, once

```bash
ssh <onyen>@longleaf.unc.edu
cd /work/users/<o>/<n>/<onyen>
git clone git@github.com:DataSlingers/CARVE.git carve   # needs a GitHub SSH key on Longleaf
cd carve
git checkout heca-longleaf
curl -LsSf https://astral.sh/uv/install.sh | sh          # installs uv into ~/.local/bin
export UV_CACHE_DIR=/work/users/<o>/<n>/<onyen>/.uv-cache
uv venv --python 3.13 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[graph,benchmarks]"
```

## 2. Data, once

About 12 GB to download and 42 GB extracted, into data/hECA/ where the loader
looks.

```bash
mkdir -p data/hECA
sbatch -p datamover -t 12:00:00 -J heca-download --wrap '
  cd data/hECA &&
  for organ in Lung Brain Kidney Heart Thymus; do
    curl -fL --retry 5 -o "ATAC-$organ.h5ad.zip" \
      "https://zenodo.org/records/15627886/files/ATAC-$organ.h5ad.zip?download=1" &&
    unzip -o "ATAC-$organ.h5ad.zip" && rm "ATAC-$organ.h5ad.zip"
  done'
```

When it finishes, check that all five files open. A truncated extraction has
happened before; it fails here rather than hours into the embed job. If unzip
created a subdirectory, move the h5ad files up into data/hECA/ first.

```bash
.venv/bin/python -c "
import glob, h5py
paths = sorted(glob.glob('data/hECA/ATAC-*.h5ad'))
for path in paths:
    h5py.File(path, 'r').close()
print(len(paths), 'files open')"
```

## 3. Embed and calibrate

```bash
RUN_DIR=/work/users/<o>/<n>/<onyen>/carve-runs/heca/$(date +%Y%m%d)-$(git rev-parse --short HEAD)
mkdir -p "$RUN_DIR"
embed=$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/embed.sbatch)
sbatch --dependency=afterok:"$embed" --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/calibrate.sbatch
squeue -u "$USER"
```

The embed job requests 8 cores, 200 GB and one day; if it runs out of memory,
resubmit with a larger `--mem` on the command line. Calibration requests
4 cores, 64 GB and one day.

## 4. The gate

Read `$RUN_DIR/calibration.json`:

- `proposed_grid`, or `grid_error` when the rule failed. On a failure, widen
  `SCAN_RESOLUTIONS` in src/benchmarks/_heca_calibration.py as the message
  says, push, pull here, and rerun calibration with `--force`.
- `projection.wall_clock_hours` at the fit node's physical core count, and
  `worker_peak_bytes`. If the projection exceeds 8 days (192 h), decide before
  fitting: fewer resamples (100 to 50) first, then fewer resolutions.

On the laptop, set `STUDIES["heca"].resolutions` to the proposed values as a
literal tuple, with a comment naming the calibration run, run
`tests/benchmarks/test_studies.py`, commit and push. Here, `git pull`.

## 5. Fit

```bash
sbatch --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/fit.sbatch
.venv/bin/python -m benchmarks.heca status --run-dir "$RUN_DIR"
```

`status` reads the fit's timing rows and can run at any time; it is light
enough for a login node. The first configuration completes within hours. If
the abort rule triggers (projected total over 10 days), cancel with
`scancel <jobid>` and revisit the grid or the resample count. The fit saves
only when it finishes; a killed fit reruns from the start with `--force`.

## 6. Copy back, on the laptop

```bash
cd ~/GitHub/CARVE/code
rsync -av <onyen>@longleaf.unc.edu:/work/users/<o>/<n>/<onyen>/carve-runs/heca/<run> results/runs/heca/
rsync -av "<onyen>@longleaf.unc.edu:/work/users/<o>/<n>/<onyen>/carve/data/hECA/.heca_cache_*" data/hECA/
```

The second command brings the embedding cache, about 280 MB, so the notebook
loads the embedding the fit ran on. Then run
notebooks/case_studies/hECA.ipynb.
````

In `src/benchmarks/README.md`, after the `## Run` section's last code block, add:

```markdown
The full-scale hECA case study runs on UNC's Longleaf cluster in stages,
through `python -m benchmarks.heca`; see `slurm/heca/README.md`.
```

- [ ] Step 4: Run the tests to verify they pass

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks/test_heca.py -q`
Expected: all PASS.

- [ ] Step 5: Commit

```bash
git add slurm/heca src/benchmarks/README.md tests/benchmarks/test_heca.py
git commit -m "docs(slurm): add the hECA jobs and the Longleaf runbook

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Final verification

Files: none new.

- [ ] Step 1: Lint

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format --check src/`
Expected: `All checks passed!` and no files to reformat.

- [ ] Step 2: The benchmarks leg

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/benchmarks -q`
Expected: all PASS (about seven minutes or more). Record the pass and skip counts in the task report.

- [ ] Step 3: The carve leg is untouched

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests --ignore=tests/benchmarks -q`
Expected: all PASS. `git diff main --stat -- src/carve carve-r` must print nothing.

- [ ] Step 4: Nothing references the removed code

Run: `grep -rn "study_scaling_sweep\|figure_study_scaling\|figure_heca_results\|_heca_results" src tests notebooks slurm`
Expected: no output.

- [ ] Step 5: A real-data smoke check on the laptop, without the pooled embedding

The pooled embedding needs the cluster, so this check uses the subsample-first path. It only confirms that the real files flow through `meta["obs"]` and the settings; it is not a result.

```bash
PYTHONPATH=src .venv/bin/python - <<'EOF'
from benchmarks.datasets import load_heca
from benchmarks._heca_stages import sweep_settings
from benchmarks._studies import STUDIES

X, y, meta = load_heca(subsample=2000, subsample_before_embedding=True, random_state=42)
print(X.shape, meta["obs"]["organ"].value_counts().to_dict())
print(meta["obs"]["study_id"].nunique(), "studies;", sweep_settings(STUDIES["heca"]))
EOF
```

Expected: `(2000, 50)`, five organs, a study count of at most 8, and both settings. This reads about 2,000 rows from the 41 GB of local h5ad files. If the files are absent, skip this step and say so in the report.

- [ ] Step 6: Report

Summarize for the author:
- the commits on `heca-longleaf`;
- the suite counts from Steps 2 and 3;
- every refinement from this plan's list that changed during execution.

The branch is not merged or pushed; the author decides both.
````
