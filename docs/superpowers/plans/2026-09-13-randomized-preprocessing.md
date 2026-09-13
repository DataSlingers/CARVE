# Randomized Preprocessing Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: make `fit(randomize_preprocessing=True)` do what the design says: one pipeline spec per resample, allocated evenly over the option product, fit separately on each subsample with arithmetically derived seeds, computed once per resample rather than once per configuration, with the generalizability classifier trained on raw features, a per-pipeline results table with counts and standard errors, a registry of rebuildable pipeline specs, and a per-pipeline plot at the module, `CARVE` and `pl` levels.

Architecture: `_pipeline` gains a frozen `PipelineSpec` (two `PipelineStep` triples) plus `allocate_pipelines` and `pipeline_from_spec`. `_runner` gains a precompute pass (`embed_resample` over resamples under joblib) that returns per-resample `ResampleEmbeddings`; `validation_iter` takes those embeddings instead of building pipelines and hands raw `X[P_1]`, `X[P_test]` to the classifier. The non-randomized path passes raw slices straight to the clusterer and is pinned by a regression fixture captured before any source change. `_utils.summarize_preprocessing_records` produces the per-(configuration, pipeline, sweep value) table; `_plotting` draws both metric plots through one private helper parameterized on the grouping column.

Tech stack: Python (`requires-python >= 3.13`; the main venv is 3.12.11 and runs the suite fine), NumPy, pandas, scikit-learn 1.7, joblib 1.5 (`return_as="generator"` is used for the precompute progress bar), matplotlib, anndata, pytest. No new dependencies.

Spec: `docs/superpowers/specs/2026-09-13-randomized-preprocessing-design.md`. Sections are cited as "spec 4.2" etc. The companion case-study spec, `docs/superpowers/specs/2026-09-13-cusanovich-randomized-preprocessing-case-study-design.md`, is the consumer of `preprocessing_results_`, `preprocessing_pipelines_`, `pipeline_from_spec` and `plot_metric_by_pipeline`; it is planned after this plan lands and is not touched here.

## Scope

This plan changes the Python package only: `src/carve/` and `tests/`, plus the plan and spec documents under `docs/superpowers/`.

The R port is out of scope. No task edits anything under `carve-r/`, and no task edits `.github/workflows/r-ci.yml`. CLAUDE.md says a semantic change in `src/carve/_*.py` generally needs an R counterpart in `carve-r/R/*.R`; this plan deliberately breaks that one-to-one mirror, as the anchored-consensus work did, and spec 2 and spec 8 record the decision. For the people carrying out and reviewing the plan, that means:

- Do not add, port or stub R code, and do not change R tests, even where a Python change has an obvious R counterpart (`pipeline.R`, `runner.R`, `grids.R`, `output.R`, `api.R`, `plotting.R`).
- A reviewer does not flag a missing R counterpart as a gap in any task.
- After this branch, `carve-r/R/pipeline.R` and `runner.R` still implement the old semantics (pipeline fit on the full X, classifier trained on the embedding). That divergence is expected. The port is a separate follow-up, listed in "After this plan".
- Task 9 checks that `carve-r/` is unchanged.

The manuscript in `../overleaf/` is also out of scope and read-only.

## Global Constraints

- All commands run from `/Users/kaiwycik/GitHub/CARVE/code`. The venv is uv-managed and has no pip; use `.venv/bin/pytest`, `.venv/bin/ruff` and `.venv/bin/python` directly. If a worktree is used it needs its own venv: `uv venv --python 3.13`, `uv pip install -r <freeze of the main venv>`, `uv pip install --no-deps -e .`.
- Dependency direction: `_types`, `_utils`, `_anndata` and `cluster` are leaves. Nothing under `_*` imports `api`. `_utils` must not import `_pipeline`; the summary function is duck-typed on `.pipeline.label` for that reason.
- Tests mirror modules one to one: `_pipeline.py` to `tests/test_pipeline.py`, `_runner.py` to `tests/test_runner.py`, `_utils.py` to `tests/test_utils.py`, `_grids.py` to `tests/test_grids.py`, `_output.py` to `tests/test_output.py`, `_plotting.py` to `tests/test_plotting.py`, `api.py` to `tests/test_api.py`, `_anndata.py` to `tests/test_anndata.py`, `tl/_carve.py` to `tests/test_tl.py`, `pl/_plots.py` to `tests/test_pl.py`. Shared test doubles go in `tests/_helpers.py`.
- `filterwarnings = error` is on. Every new warning needs a `pytest.warns(..., match=)` where it fires and must not fire anywhere else. `-ra` is the only implicit addopt; `--cov-fail-under=75` fails a single-file run, so omit it locally.
- Lint and format with the pinned ruff only: `.venv/bin/ruff check src/` and `.venv/bin/ruff format src/`. Never on `tests/` or `notebooks/`.
- Run tests in the foreground with a long timeout (`timeout` of 600000 ms on the Bash tool). When this plan was verified, the carve leg with coverage took 338 s and the benchmarks leg 185 s (Python 3.12.11, the main venv). A slow run is not a hang. Never background a test run.
- `config_id` is a join key, never a positional index. The precompute pass produces nothing per configuration, so it must not touch the artifact containers `fit` asserts alignment on.
- Seeds are derived arithmetically, never shared (spec 4.3). Per resample `b` with base `s0`: subsamples `s0 + b` and `s0 + b + n_resamples`; transformer fits `s0 + b`, `s0 + b + n_resamples`, `s0 + b + 2 * n_resamples`. Allocation uses `np.random.default_rng(s0)`. The stdlib `random` module leaves `_pipeline`.
- Selection rules operate on `sweep_rank`. The per-pipeline table carries the sweep column for reporting; selection reads `estimator_results_`.
- Noise policy unchanged: `"drop"` shrinks the index arrays; the classifier reads raw slices by the shrunken global indices.
- Parallelism stays joblib over resamples inside a serial loop over configurations. The precompute pass is a second joblib-over-resamples stage before that loop under the same `outer_n_jobs` from `resolve_core_budget`.
- No `logging`. Diagnostics go through `warnings.warn(..., stacklevel=2)` and `verbose`-gated `print`; progress via `tqdm.auto`.
- Public surface stays small: the new public names are `CARVE.preprocessing_pipelines_`, `CARVE.plot_metric_by_pipeline` and `pl.metric_by_pipeline`. `_pipeline.PipelineSpec`, `_pipeline.pipeline_from_spec` and `_plotting.plot_metric_by_pipeline` are private but importable by `benchmarks`.
- The R port is out of scope; see Scope above. Do not edit anything under `carve-r/` or `.github/workflows/r-ci.yml`, and do not treat a missing R counterpart as a defect.
- `../overleaf/` is read-only. The manuscript corrections in spec 8 are for the author.
- American spelling. No bold or italics in code comments, docstrings or authored Markdown.
- Commit after every task with the attribution lines from the session's system reminder.

## Refinements the plan makes to the spec

Stated here so an executor does not read them as deviations.

1. `preprocessing_results_` carries `sweep_param`, `sweep_value` and `sweep_rank` in addition to the named sweep column (`n_clusters` or `resolution`) listed in spec 5.1. The shared drawing helper of spec 5.5 and `select_best_row_by_rule` read the sweep axis from those columns; without them the per-pipeline plot needs a second code path, which spec 5.5 forbids. Every column spec 5.1 lists is present with the listed meaning; these three are bookkeeping, as they are on `estimator_results_`.
2. The dashed line in `plot_metric_by_pipeline` marks the sweep value `select_best_row_by_rule` picks among the per-pipeline rows of the plotted configuration. That is the direct mirror of `plot_metric_over_n_clusters`, which applies the same function to the table it draws. It can differ from the pooled selection in `estimator_results_`; the docstring says so.
3. `tl.attach_results` records `selected_method_id` in `uns[key]["params"]`, so `pl.metric_by_pipeline` can default to the selected configuration without re-running selection, which the `pl` module rules out.
4. A step label renders every sampled hyperparameter, including keys with a single candidate (`TSNE(n_components=2, perplexity=30)`, not `TSNE(perplexity=30)`). Omitting single-candidate keys would merge two options of the same class with different fixed values, which is the pooling defect spec 1 point 4 describes. A caller who wants the shorter label passes a display name or leaves the fixed value out of the grid. `allocate_pipelines` raises when two different specs render the same label.
5. `run_validation` keeps its `normalization_options`, `dim_reduction_options` and `randomize_preprocessing` parameters and calls `allocate_pipelines` itself; `validation_iter` loses those three parameters and gains `embeddings`. Spec 4.2 places allocation "before any configuration runs" without naming the caller.
6. `default_dim_reduction_options` limits the grids by the smaller of a resample's training subsample and held-out set. Spec 5.3 says "the P_test size"; the two coincide at any ratio above 0.5, and below it the training subsample is what t-SNE and UMAP are fitted on.
7. Subsample indices are derived in one helper, `_runner._resample_indices`, which both the precompute pass and `validation_iter` call. That helper is what makes spec 4.2's "same P1, P2 and P_test by construction" literal. `validation_iter` also checks the precomputed embeddings' row counts against its subsamples and raises `RuntimeError` on a mismatch.
8. `embed_resample` ignores umap-learn's `UserWarning` that setting `random_state` disables its parallelism. Spec 4.3 requires the seed, CARVE parallelizes over resamples instead, and the warning fires three times per resample; under `filterwarnings = error` it aborts the fit. Found by an end-to-end fit with the default grids during verification; no unit test fitted UMAP before Task 3 added one.
9. The raw-feature classifier test embeds with two columns of seeded noise rather than the constant vector spec 7 describes. Both make the embedding useless for prediction, but a constant embedding makes KMeans emit `ConvergenceWarning` and CARVE its cluster-count warnings, each of which `filterwarnings = error` would turn into a failure unrelated to the assertion.
10. `TestBuildPreprocessingPipeline`, `TestSamplePreprocessingPipeline` and `TestChoosePreprocessor` are removed in Task 3 with the functions they test. Their assertions are carried by Task 2: the option-syntax errors move to `TestParseOption` with the same messages, and identity passthrough, the two-step pipeline and reproducibility move to `TestPipelineFromSpec` and `TestAllocatePipelines`. Spec 7 asks for tests pinning old return shapes to be updated rather than deleted; these pin functions that no longer exist.

## How this plan was verified

The code in Tasks 1 through 8 was executed on 2026-09-13 in a scratch copy of `src/` and `tests/` taken from commit 1dea71d, task by task, before the plan was finalized. Tasks 3 to 8 are generated from that run: each edit below is a replacement whose old text occurs exactly once in the file at that point, and applying the Markdown in order to the Task 2 state reproduces the verified tree file for file. Each task's Step 2 and Step 4 output was captured from the same run. Every new test was checked by mutation: a named perturbation of the source (fitting on all of X, sharing one seed, pooling the standard error over a configuration, drawing the pipeline plot over the whole table, and 40 others) turned it red, with an unmutated control run green in the same harness. Line numbers are not used anywhere, so the edits stay valid as long as the task order is kept.

## File structure

Created:
- `tests/fixtures/__init__.py` (empty) and `tests/fixtures/nonrandomized_gate.py`: the regression gate dataset, model, array extraction, and the script that writes `tests/fixtures/nonrandomized_gate.npz`.
- `tests/fixtures/nonrandomized_gate.npz`: captured numbers from the start commit.

Modified, in task order:
- `src/carve/_pipeline.py`: `PipelineStep`, `PipelineSpec`, `_parse_option`, `option_label`, `allocate_pipelines`, `pipeline_from_spec`, `accepts_random_state`. The old `build_preprocessing_pipeline`, `sample_preprocessing_pipeline` and `_choose_preprocessor` are removed in Task 3.
- `src/carve/_runner.py`: `ResampleEmbeddings`, `_resample_indices`, `embed_resample`, `precompute_embeddings`, `_check_embeddings`; `ResampleResult.pipeline`; `validation_iter(embeddings=...)`; allocation and precompute in `run_validation`; per-configuration pipeline records carrying `method_id`, `method_label`, sweep bookkeeping and results.
- `src/carve/_utils.py`: `summarize_preprocessing_records` returns `(table, pipelines)` in the spec 5.1 schema.
- `src/carve/_types.py`: the `PipelineRecord` comment.
- `src/carve/api.py`: `preprocessing_pipelines_`; `fit` wiring, default-option gating, header names; docstrings; `plot_metric_by_pipeline`.
- `src/carve/_grids.py`: `default_normalization_options(X)`, discrete `default_dim_reduction_options`.
- `src/carve/_output.py`: header lists option names when randomized.
- `src/carve/_plotting.py`: `_draw_metric_lines`, `plot_metric_by_pipeline`.
- `src/carve/tl/_carve.py`: stores `preprocessing_results` in `uns`, records `selected_method_id`.
- `src/carve/pl/_plots.py`, `src/carve/pl/__init__.py`: `metric_by_pipeline`.
- `src/carve/_anndata.py`: docstrings generalized to both tables; `results_from_uns` re-casts `n_resamples`.
- `tests/_helpers.py`: `make_noise_embedding` (Task 2), `make_fit_recorder` and `make_fit_input_spy` (Task 3).
- Tests as listed per task.

---

### Task 0: Branch and commit the specs

- [ ] Step 1: Create the branch from `main` and commit the two untracked spec files

```bash
git checkout -b randomized-preprocessing
git add docs/superpowers/specs/2026-09-13-randomized-preprocessing-design.md docs/superpowers/specs/2026-09-13-cusanovich-randomized-preprocessing-case-study-design.md docs/superpowers/plans/2026-09-13-randomized-preprocessing.md
git commit -m "docs: randomized preprocessing design, its Cusanovich case study, and the implementation plan"
```

---

### Task 1: Regression gate for the non-randomized path

Captured before any source change, from the start commit, so every later task can prove the non-randomized numbers did not move (spec 4.1, spec 7 "Regression gate").

Files:
- Create: `tests/fixtures/__init__.py`, `tests/fixtures/nonrandomized_gate.py`, `tests/fixtures/nonrandomized_gate.npz`
- Test: `tests/test_api.py` (append)

Interfaces:
- Consumes: `carve.CARVE` as it exists at the start commit.
- Produces: `tests.fixtures.nonrandomized_gate.{GATE_PATH, gate_dataset, gate_model, gate_arrays, fit_gate}`; `fit_gate(**fit_kwargs) -> dict[str, np.ndarray]` with keys `columns`, `method_labels`, `results`, `consensus`, `consensus_generalizability`, `gini`, `ce`, `generalizability`.

- [ ] Step 1: Create the fixture module

`tests/fixtures/__init__.py` is empty. `tests/fixtures/nonrandomized_gate.py`:

```python
"""Regression gate for the non-randomized path.

Captures what CARVE computes on a fixed dataset under a fixed seed with
randomize_preprocessing=False, so a change to the randomized path can be
checked against it. tests/test_api.py::TestNonRandomizedRegressionGate
refits the same model and compares.

The fixture was written from commit 1dea71d, before the 2026-09-13
randomized-preprocessing redesign. Regenerate it only on purpose, from a
commit whose non-randomized numbers are known good:

    .venv/bin/python -m tests.fixtures.nonrandomized_gate
"""

from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from carve import CARVE

GATE_PATH = Path(__file__).with_name("nonrandomized_gate.npz")

GATE_KEYS = (
    "results",
    "consensus",
    "consensus_generalizability",
    "gini",
    "ce",
    "generalizability",
)


def gate_dataset() -> np.ndarray:
    """Three Gaussian clusters with boundary points, 60 x 5.

    Offsets of 3 rather than 6 leave points near the boundaries, so a change
    in the feature space the clusterers see moves at least some labels. A
    fully separated dataset gives the same partition under almost any
    preprocessing and could not detect the mutation the gate exists for.
    """
    rng = np.random.RandomState(7)
    return np.vstack(
        [
            rng.randn(20, 5) + [3, 0, 0, 0, 0],
            rng.randn(20, 5) + [0, 3, 0, 0, 0],
            rng.randn(20, 5) + [0, 0, 3, 0, 0],
        ]
    )


def gate_model(**kwargs) -> CARVE:
    """The gate's model. kwargs override constructor arguments."""
    options = dict(
        n_clusters=np.array([2, 3, 4]),
        n_resamples=6,
        subsample_ratio=0.618,
        estimator_param_grids=[
            (KMeans, {"n_clusters": [2, 3, 4], "n_init": [3]}),
            (AgglomerativeClustering, {"n_clusters": [2, 3, 4], "linkage": ["ward"]}),
        ],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    options.update(kwargs)
    return CARVE(**options)


def gate_arrays(model: CARVE) -> dict[str, np.ndarray]:
    """Every number the gate compares, keyed for np.savez."""
    df = model.estimator_results_
    numeric = df.select_dtypes(include="number")
    return {
        "columns": np.asarray(list(numeric.columns)),
        "method_labels": np.asarray(df["method_label"].astype(str).tolist()),
        "results": numeric.to_numpy(dtype=float),
        "consensus": np.stack(model.consensus_matrices_),
        "consensus_generalizability": np.stack(
            model.consensus_generalizability_matrices_
        ),
        "gini": np.asarray(model.stability_gini_scores_, dtype=float),
        "ce": np.asarray(model.stability_ce_scores_, dtype=float),
        "generalizability": np.vstack(model.generalizability_scores_),
    }


def fit_gate(**fit_kwargs) -> dict[str, np.ndarray]:
    """Fit the gate model on the gate dataset and extract its arrays."""
    return gate_arrays(gate_model().fit(gate_dataset(), **fit_kwargs))


if __name__ == "__main__":
    np.savez_compressed(GATE_PATH, **fit_gate())
    print(f"wrote {GATE_PATH}")
```

- [ ] Step 2: Write the fixture from the start commit

Run: `git stash list` must be empty of source changes and `git status --short src/` must print nothing. Then:

```bash
.venv/bin/python -m tests.fixtures.nonrandomized_gate
ls -la tests/fixtures/nonrandomized_gate.npz
```

Expected: `wrote .../tests/fixtures/nonrandomized_gate.npz`; the file is well under 1 MB (about 100 KB).

- [ ] Step 3: Write the gate tests

Append to `tests/test_api.py`, after the existing imports add:

```python
from tests.fixtures.nonrandomized_gate import (
    GATE_KEYS,
    GATE_PATH,
    fit_gate,
    gate_arrays,
    gate_dataset,
    gate_model,
)
```

and at the end of the file:

```python
# ---------------------------------------------------------------------------
# Regression gate: the non-randomized path is pinned to a captured fixture
# ---------------------------------------------------------------------------


class TestNonRandomizedRegressionGate:
    """The non-randomized path must compute exactly what it did before the
    randomized path was redesigned. The fixture was captured from the start
    commit of that work; see tests/fixtures/nonrandomized_gate.py.
    """

    def test_matches_the_captured_fixture(self):
        before = np.load(GATE_PATH)
        after = fit_gate()
        assert list(before["columns"]) == list(after["columns"])
        assert list(before["method_labels"]) == list(after["method_labels"])
        for key in GATE_KEYS:
            np.testing.assert_allclose(
                after[key], before[key], rtol=1e-6, atol=1e-8, err_msg=key
            )

    def test_gate_can_detect_the_randomized_path(self):
        """A randomized fit that standardizes and projects each subsample
        must not reproduce the fixture, or the gate above proves nothing.
        """
        model = gate_model(
            normalization_options=[(StandardScaler, {})],
            dim_reduction_options=[(PCA, {"n_components": [2]})],
        )
        randomized = gate_arrays(model.fit(gate_dataset(), randomize_preprocessing=True))
        before = np.load(GATE_PATH)
        assert not np.allclose(randomized["results"], before["results"], equal_nan=True)
```

- [ ] Step 4: Run the gate tests

Run: `.venv/bin/pytest tests/test_api.py::TestNonRandomizedRegressionGate -v`
Expected: 2 passed.

- [ ] Step 5: Commit

```bash
git add tests/fixtures/__init__.py tests/fixtures/nonrandomized_gate.py tests/fixtures/nonrandomized_gate.npz tests/test_api.py
git commit -m "test(api): pin the non-randomized path to a captured regression fixture"
```

---

### Task 2: Pipeline specs, allocation and seeded instantiation

Spec 4.2 "Allocation", 4.3 "Seeding", 4.4 "The pipeline spec". Additive: the old functions stay until Task 3 rewires the runner.

Files:
- Modify: `src/carve/_pipeline.py`
- Test: `tests/test_pipeline.py`

Interfaces:
- Consumes: `carve._sweep._format_param_value`, `carve._types.PreprocOption`.
- Produces:
  - `class PipelineStep(NamedTuple)`: `cls: Any`, `params: dict[str, Any]`, `name: str | None`; property `label -> str`.
  - `@dataclass(frozen=True) class PipelineSpec`: `normalization: PipelineStep`, `dim_reduction: PipelineStep`; property `label -> str` rendering `"<norm label> | <dr label>"`.
  - `allocate_pipelines(normalization_options, dim_reduction_options, n_resamples: int, random_state: int | None) -> list[PipelineSpec]`.
  - `pipeline_from_spec(spec: PipelineSpec, random_state: int) -> sklearn.pipeline.Pipeline` with steps named `"norm"` and `"dr"`.
  - `accepts_random_state(cls) -> bool`.
  - `option_label(option: PreprocOption) -> str` for the run header.
  - `_parse_option(option) -> tuple[cls, name | None, grid]` with the existing `ValueError`/`TypeError` messages.

- [ ] Step 1: Write the failing tests

Replace the contents of `tests/test_pipeline.py` from the imports down to (not including) the `# build_preprocessing_pipeline` section header with:

```python
"""Tests for carve._pipeline module."""

import itertools
import random
from collections import Counter

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from carve._pipeline import (
    PipelineSpec,
    PipelineStep,
    _choose_preprocessor,
    _parse_option,
    accepts_random_state,
    allocate_pipelines,
    build_preprocessing_pipeline,
    option_label,
    pipeline_from_spec,
    sample_preprocessing_pipeline,
)
from tests._helpers import make_noise_embedding

IDENTITY = PipelineStep(FunctionTransformer, {}, None)

# -----------------------------------------------------------------------
# PipelineStep and PipelineSpec labels
# -----------------------------------------------------------------------


class TestLabels:
    def test_identity_renders_as_identity(self):
        assert IDENTITY.label == "identity"

    def test_function_transformer_renders_its_func(self):
        step = PipelineStep(FunctionTransformer, {"func": np.log1p}, None)
        assert step.label == "log1p"

    def test_class_name_with_sampled_hyperparameters_in_grid_order(self):
        step = PipelineStep(TSNE, {"n_components": 2, "perplexity": 30}, None)
        assert step.label == "TSNE(n_components=2, perplexity=30)"

    def test_class_without_hyperparameters(self):
        assert PipelineStep(StandardScaler, {}, None).label == "StandardScaler"

    def test_user_name_replaces_the_class_name(self):
        step = PipelineStep(TSNE, {"perplexity": 30}, "tsne")
        assert step.label == "tsne(perplexity=30)"

    def test_floats_render_compactly(self):
        step = PipelineStep(PCA, {"n_components": 0.95}, None)
        assert step.label == "PCA(n_components=0.95)"

    def test_spec_label_joins_the_steps(self):
        spec = PipelineSpec(IDENTITY, PipelineStep(PCA, {"n_components": 2}, None))
        assert spec.label == "identity | PCA(n_components=2)"

    def test_specs_with_different_hyperparameters_are_distinct(self):
        a = PipelineSpec(IDENTITY, PipelineStep(PCA, {"n_components": 2}, None))
        b = PipelineSpec(IDENTITY, PipelineStep(PCA, {"n_components": 3}, None))
        assert a != b and a.label != b.label


# -----------------------------------------------------------------------
# allocate_pipelines
# -----------------------------------------------------------------------


NORM = [(FunctionTransformer, {}), (StandardScaler, {})]
DR = [
    (FunctionTransformer, {}),
    (PCA, {"n_components": [2, 3]}),
    (TSNE, {"n_components": [2], "perplexity": [15, 30]}),
]


def _combination(spec):
    """The (normalization option, reduction option) a spec was drawn from."""
    return (spec.normalization.cls, spec.dim_reduction.cls)


class TestAllocatePipelines:
    def test_one_spec_per_resample(self):
        specs = allocate_pipelines(NORM, DR, 20, 0)
        assert len(specs) == 20
        assert all(isinstance(s, PipelineSpec) for s in specs)

    def test_counts_per_combination_differ_by_at_most_one(self):
        # 2 x 3 = 6 combinations over 100 resamples: 16 or 17 each.
        counts = Counter(_combination(s) for s in allocate_pipelines(NORM, DR, 100, 0))
        assert len(counts) == 6
        assert set(counts.values()) <= {16, 17}
        assert sum(counts.values()) == 100

    def test_every_combination_used_when_resamples_exceed_combinations(self):
        counts = Counter(_combination(s) for s in allocate_pipelines(NORM, DR, 7, 0))
        assert len(counts) == 6
        assert set(counts.values()) == {1, 2}

    def test_order_is_shuffled_not_cyclic(self):
        order = [_combination(s) for s in allocate_pipelines(NORM, DR, 60, 0)]
        cyclic = [(n[0], d[0]) for n, d in itertools.product(NORM, DR)] * 10
        assert sorted(order, key=repr) == sorted(cyclic, key=repr)
        assert order != cyclic

    def test_hyperparameters_are_drawn_from_the_candidate_lists(self):
        specs = allocate_pipelines(NORM, DR, 60, 0)
        pca = [s.dim_reduction.params for s in specs if s.dim_reduction.cls is PCA]
        tsne = [s.dim_reduction.params for s in specs if s.dim_reduction.cls is TSNE]
        assert {p["n_components"] for p in pca} == {2, 3}
        assert {p["perplexity"] for p in tsne} == {15, 30}
        assert all(p["n_components"] == 2 for p in tsne)

    def test_identical_for_the_same_seed_and_different_otherwise(self):
        a = allocate_pipelines(NORM, DR, 30, 5)
        b = allocate_pipelines(NORM, DR, 30, 5)
        c = allocate_pipelines(NORM, DR, 30, 6)
        assert a == b
        assert a != c

    def test_none_seed_means_zero(self):
        assert allocate_pipelines(NORM, DR, 10, None) == allocate_pipelines(NORM, DR, 10, 0)

    @pytest.mark.parametrize("norm, dr", [([], DR), (NORM, [])])
    def test_empty_option_list_raises(self, norm, dr):
        with pytest.raises(ValueError, match="at least one normalization option"):
            allocate_pipelines(norm, dr, 5, 0)

    def test_two_pipelines_with_one_label_raise(self):
        # Both normalization options are named "x", so two different
        # pipelines would share a key in preprocessing_pipelines_.
        with pytest.raises(ValueError, match="render as the same label 'x | identity'"):
            allocate_pipelines(
                [(StandardScaler, "x", {}), (FunctionTransformer, "x", {})],
                [(FunctionTransformer, {})],
                4,
                0,
            )

    def test_empty_candidate_list_raises(self):
        with pytest.raises(ValueError, match="candidate list for 'n_components'"):
            allocate_pipelines(NORM, [(PCA, {"n_components": []})], 5, 0)

    def test_accepts_named_and_dict_options(self):
        specs = allocate_pipelines(
            [(StandardScaler, "scaled", {})],
            [{"cls": PCA, "params": {"n_components": [2]}, "name": "pca"}],
            3,
            0,
        )
        assert {s.label for s in specs} == {"scaled | pca(n_components=2)"}


# -----------------------------------------------------------------------
# pipeline_from_spec and accepts_random_state
# -----------------------------------------------------------------------


class TestPipelineFromSpec:
    def test_builds_a_two_step_pipeline(self):
        spec = PipelineSpec(
            PipelineStep(StandardScaler, {}, None),
            PipelineStep(PCA, {"n_components": 2}, None),
        )
        pipeline = pipeline_from_spec(spec, random_state=0)
        assert isinstance(pipeline, Pipeline)
        assert [name for name, _ in pipeline.steps] == ["norm", "dr"]
        assert isinstance(pipeline.named_steps["norm"], StandardScaler)
        assert pipeline.named_steps["dr"].n_components == 2

    def test_seeds_transformers_that_accept_random_state(self):
        spec = PipelineSpec(IDENTITY, PipelineStep(PCA, {"n_components": 2}, None))
        assert pipeline_from_spec(spec, random_state=7).named_steps["dr"].random_state == 7

    def test_does_not_seed_transformers_without_random_state(self):
        spec = PipelineSpec(PipelineStep(StandardScaler, {}, None), IDENTITY)
        params = pipeline_from_spec(spec, random_state=7).named_steps["norm"].get_params()
        assert "random_state" not in params

    def test_overrides_a_random_state_in_the_sampled_params(self):
        spec = PipelineSpec(
            IDENTITY, PipelineStep(PCA, {"n_components": 2, "random_state": 99}, None)
        )
        assert pipeline_from_spec(spec, random_state=7).named_steps["dr"].random_state == 7

    def test_identity_pipeline_passes_data_through(self):
        X = np.arange(6.0).reshape(3, 2)
        np.testing.assert_array_equal(
            pipeline_from_spec(PipelineSpec(IDENTITY, IDENTITY), 0).fit_transform(X), X
        )

    def test_seed_reaches_a_stochastic_transformer(self):
        noise = make_noise_embedding()
        spec = PipelineSpec(IDENTITY, PipelineStep(noise, {}, None))
        X = np.zeros((5, 3))
        a = pipeline_from_spec(spec, random_state=1).fit_transform(X)
        b = pipeline_from_spec(spec, random_state=1).fit_transform(X)
        c = pipeline_from_spec(spec, random_state=2).fit_transform(X)
        np.testing.assert_array_equal(a, b)
        assert not np.allclose(a, c)

    def test_accepts_random_state_probe(self):
        assert accepts_random_state(PCA)
        assert accepts_random_state(TSNE)
        assert not accepts_random_state(StandardScaler)
        assert not accepts_random_state(FunctionTransformer)


# -----------------------------------------------------------------------
# option_label and _parse_option
# -----------------------------------------------------------------------


class TestOptionLabel:
    def test_labels(self):
        assert option_label((FunctionTransformer, {})) == "identity"
        assert option_label((FunctionTransformer, {"func": [np.log1p]})) == "log1p"
        assert option_label((StandardScaler, {})) == "StandardScaler"
        assert option_label((PCA, {"n_components": [2, 5]})) == "PCA"
        assert option_label((PCA, "pca", {"n_components": [2]})) == "pca"
        assert option_label({"cls": TSNE, "params": {}}) == "TSNE"


class TestParseOption:
    def test_tuple2(self):
        assert _parse_option((StandardScaler, {})) == (StandardScaler, None, {})

    def test_tuple3(self):
        assert _parse_option((StandardScaler, "s", {"a": [1]})) == (
            StandardScaler,
            "s",
            {"a": [1]},
        )

    def test_dict_forms(self):
        assert _parse_option({"cls": PCA, "params": {"n_components": [2]}}) == (
            PCA,
            None,
            {"n_components": [2]},
        )
        assert _parse_option({"estimator": PCA, "grid": {"n_components": [2]}, "name": "p"}) == (
            PCA,
            "p",
            {"n_components": [2]},
        )

    def test_invalid_tuple_length(self):
        with pytest.raises(ValueError, match="Option tuples must be"):
            _parse_option((StandardScaler,))

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="Unsupported option type"):
            _parse_option("not_a_valid_option")

    def test_dict_missing_cls(self):
        with pytest.raises(ValueError, match="Dict option must contain"):
            _parse_option({"params": {}})
```

Keep the existing `TestBuildPreprocessingPipeline`, `TestSamplePreprocessingPipeline` and `TestChoosePreprocessor` classes below this for now; Task 3 removes them.

Add the noise-embedding double to `tests/_helpers.py` (append):

```python
def make_noise_embedding() -> type:
    """A transformer whose output is noise drawn from its random_state.

    The embedding carries no information about X, so labels clustered on it
    cannot be predicted from X, which is what the raw-feature classifier
    tests rely on. It is only reproducible when CARVE seeds it, which is
    what the seeding tests rely on; an unseeded fit draws fresh entropy.
    fit() records the random_state it ran with on the class, so a test can
    read the seeds CARVE derived.
    """

    class NoiseEmbedding(BaseEstimator, TransformerMixin):
        seen: list = []

        def __init__(self, random_state=None):
            self.random_state = random_state

        def fit(self, X, y=None):
            type(self).seen.append(self.random_state)
            return self

        def transform(self, X):
            rng = np.random.default_rng(self.random_state)
            return rng.standard_normal((np.asarray(X).shape[0], 2))

    return NoiseEmbedding
```

and extend the helpers' import line to `from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin`.

- [ ] Step 2: Run the new tests to verify they fail

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: collection error, `ImportError: cannot import name 'PipelineSpec' from 'carve._pipeline'`.

- [ ] Step 3: Implement the new API in `src/carve/_pipeline.py`

Replace the module docstring and imports, and add the new code above `build_preprocessing_pipeline` (which stays for now):

```python
"""Preprocessing pipeline specs: allocation, instantiation and labels.

A randomized fit draws one pipeline spec per resample. The spec records the
transformer classes and the sampled hyperparameter values, so the same
pipeline can be rebuilt on any subset of the data with its own seed. The
runner does that three times per resample (``_runner.embed_resample``) and
the case-study code does it on the full data to draw the winning pipeline.
"""

import inspect
import itertools
import random
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
```

`import random` stays in the import block for the three old functions, which remain below unchanged until Task 3.

- [ ] Step 4: Run the pipeline tests

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: all pass (the new classes plus the three retained old classes).

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/carve/_pipeline.py tests/test_pipeline.py tests/_helpers.py
git commit -m "feat(pipeline): PipelineSpec, stratified allocation and seeded pipeline_from_spec"
```

### Task 3: Per-subsample fit, precompute pass, raw-feature classifier and the per-pipeline table

Spec 4.1, 4.2, 4.3, 5.1, 5.2 and the verification items "Per-subsample fit", "Once per resample", "Classifier on raw features", "Seeding" and "Summary schema". The runner, the summary and the `fit` wiring change together: the per-resample result loses its four preprocessing fields, which the summary reads and `fit` unpacks, so no subset of them leaves the suite green. The old `build_preprocessing_pipeline`, `sample_preprocessing_pipeline` and `_choose_preprocessor` are removed with their tests; Task 2's `TestParseOption` and `TestOptionLabel` cover the option syntax they validated.

Files:
- Modify: `src/carve/_pipeline.py`
- Modify: `src/carve/_runner.py`
- Modify: `src/carve/_types.py`
- Modify: `src/carve/_utils.py`
- Modify: `src/carve/api.py`
- Test: `tests/_helpers.py`
- Test: `tests/test_api.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_runner.py`
- Test: `tests/test_utils.py`

Interfaces:
- Consumes: `allocate_pipelines`, `pipeline_from_spec` and `PipelineSpec` from Task 2; `tests._helpers.make_noise_embedding` from Task 2; the regression gate from Task 1, which must stay green.
- Produces:
  - `_runner._resample_indices(n_samples, *, seed, n_resamples, subsample_ratio, random_state, run_stability) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]` returning `(P_1, P_test, P_2)`; the only place subsample seeds are derived.
  - `_runner.ResampleEmbeddings(NamedTuple)`: `spec: PipelineSpec`, `X_1: np.ndarray`, `X_2: np.ndarray | None`, `X_test: np.ndarray | None`.
  - `_runner.embed_resample(X, spec, *, seed, n_resamples, subsample_ratio, random_state, mode="default") -> ResampleEmbeddings`.
  - `_runner.precompute_embeddings(X, pipelines, *, subsample_ratio, random_state, mode, n_jobs, show_progress=False) -> list[ResampleEmbeddings]`, with a tqdm bar labeled `Preprocessing`.
  - `_runner.validation_iter(..., embeddings: ResampleEmbeddings | None = None)`. Its `normalization_options`, `dim_reduction_options` and `randomize_preprocessing` parameters are removed. It raises `RuntimeError("Precomputed embedding X_test has 11 rows but this resample's subsample has 12. ...")` on a mismatch.
  - `ResampleResult.pipeline: PipelineSpec | None` replaces `normalization_params`, `dim_reduction_params`, `normalization_name` and `dim_reduction_name`.
  - `run_validation` returns one pipeline record per configuration when randomized: `{"method_id", "method_label", "sweep_value", "sweep_rank", "results"}`.
  - `_utils.summarize_preprocessing_records(pipeline_records, sweep_param="n_clusters") -> tuple[pd.DataFrame, dict[str, PipelineSpec]]`. Columns, in order: `method_id`, `method_label`, `pipeline`, `normalization`, `dim_reduction`, the sweep column, `n_resamples`, `ari_stability`, `ari_stability_se`, `ari_generalizability`, `ari_generalizability_se`, `n_clusters_observed`, `sweep_param`, `sweep_value`, `sweep_rank`.
  - `CARVE.preprocessing_pipelines_: dict[str, PipelineSpec] | None`.
  - Test helpers `make_fit_recorder()` and `make_fit_input_spy()` in `tests/_helpers.py`; the module fixture `randomized_fit` and the static factory `TestRandomizedPreprocessing.make(**kwargs)` in `tests/test_api.py`, which Task 7 reuses.

- [ ] Step 1: Write the failing tests

The recorder helpers read a row's own index from column 0 (`_indexed` in `tests/test_runner.py` writes it there), so the tests assert exactly which samples every fit saw rather than only how many. In `tests/test_runner.py` the fifteen `normalization_options=[]`, `dim_reduction_options=[]` and `randomize_preprocessing=False` lines inside `validation_iter(...)` calls go, because those parameters no longer exist; the `run_validation(...)` calls keep them.

In `tests/_helpers.py`, replace:

```python

    return NoiseEmbedding
```

with:

```python

    return NoiseEmbedding


def make_fit_recorder() -> type:
    """An identity transformer class that records which rows each fit sees.

    Rows are read from column 0, so a test that stores each row's own index
    there gets back exactly the samples a fit received, as a tuple.
    """

    class FitRecorder(BaseEstimator, TransformerMixin):
        seen: list = []

        def fit(self, X, y=None):
            type(self).seen.append(tuple(np.asarray(X)[:, 0].astype(int)))
            return self

        def transform(self, X):
            return np.asarray(X)

    return FitRecorder


def make_fit_input_spy() -> type:
    """A classifier class that records every matrix it fits and predicts on."""

    class FitInputSpy(BaseEstimator, ClassifierMixin):
        fitted: list = []
        predicted: list = []

        def fit(self, X, y):
            type(self).fitted.append(np.array(X))
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            type(self).predicted.append(np.array(X))
            return np.full(X.shape[0], self.classes_[0])

    return FitInputSpy
```

In `tests/test_api.py`, replace:

```python
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler
```

with:

```python
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler
```

In `tests/test_api.py`, replace:

```python
import carve.api as carve_api
from carve import CARVE, LeidenClustering, LouvainClustering
from carve._utils import resolve_anchors
from tests._helpers import make_njobs_spy, make_seed_spy
from tests.fixtures.nonrandomized_gate import (
    GATE_KEYS,
```

with:

```python
import carve.api as carve_api
from carve import CARVE, LeidenClustering, LouvainClustering
from carve._pipeline import PipelineSpec, pipeline_from_spec
from carve._utils import resolve_anchors
from tests._helpers import make_njobs_spy, make_noise_embedding, make_seed_spy
from tests.fixtures.nonrandomized_gate import (
    GATE_KEYS,
```

In `tests/test_api.py`, replace:

```python
        assert "min_cluster_size" in carve.estimator_results_.columns
        assert carve.preprocessing_results_ is not None

        carve.sweep = None
```

with:

```python
        assert "min_cluster_size" in carve.estimator_results_.columns
        assert carve.preprocessing_results_ is not None
        assert carve.preprocessing_pipelines_ is not None

        carve.sweep = None
```

In `tests/test_api.py`, replace:

```python
        assert "min_cluster_size" not in carve.estimator_results_.columns
        assert carve.preprocessing_results_ is None
        assert carve.consensus_matrices_[0].shape == (120, 120)
        np.testing.assert_array_equal(carve.reference_labels, np.repeat([1, 0, 2], 40))
```

with:

```python
        assert "min_cluster_size" not in carve.estimator_results_.columns
        assert carve.preprocessing_results_ is None
        assert carve.preprocessing_pipelines_ is None
        assert carve.consensus_matrices_[0].shape == (120, 120)
        np.testing.assert_array_equal(carve.reference_labels, np.repeat([1, 0, 2], 40))
```

In `tests/test_api.py`, replace:

```python


class TestRandomizedPreprocessing:
    """fit(randomize_preprocessing=True) samples a normalization and a
    reduction per resample and summarizes them in preprocessing_results_.
    Explicit option lists keep the defaults' t-SNE and UMAP out of the test.
    """

    def _make(self):
        return CARVE(
            n_clusters=np.array([2, 3]),
            n_resamples=4,
            estimator_param_grids=[(KMeans, {"n_clusters": [2, 3]})],
            normalization_options=[(FunctionTransformer, {}), (StandardScaler, {})],
            dim_reduction_options=[(FunctionTransformer, {}), (PCA, {"n_components": [2, 3]})],
            random_state=0,
            verbose=0,
        )

    def test_summary_groups_on_the_sweep_parameter(self, X_two_clusters):
        carve = self._make().fit(X_two_clusters, randomize_preprocessing=True)
        df = carve.preprocessing_results_
        assert list(df.columns) == [
            "norm__func",
            "dr__method",
            "n_clusters",
            "ari_stability",
            "ari_generalizability",
        ]
        assert set(df["norm__func"]) <= {"identity", "StandardScaler"}
        assert set(df["dr__method"]) <= {"identity", "PCA"}
        assert set(df["n_clusters"]) == {2, 3}
        # More than one pipeline was actually drawn.
        assert len(set(zip(df["norm__func"], df["dr__method"]))) > 1

    def test_summary_is_reproducible_under_a_seed(self, X_two_clusters):
        a = self._make().fit(X_two_clusters, randomize_preprocessing=True)
        b = self._make().fit(X_two_clusters, randomize_preprocessing=True)
        pd.testing.assert_frame_equal(a.preprocessing_results_, b.preprocessing_results_)

    def test_summary_is_none_without_randomization(self, X_two_clusters):
        assert self._make().fit(X_two_clusters).preprocessing_results_ is None


```

with:

```python


@pytest.fixture(scope="module")
def randomized_fit():
    """A randomized fit over two estimators, two k and four pipelines."""
    rng = np.random.RandomState(1)
    X = np.vstack([rng.randn(30, 5) + [4, 0, 0, 0, 0], rng.randn(30, 5) + [0, 4, 0, 0, 0]])
    return X, TestRandomizedPreprocessing.make().fit(X, randomize_preprocessing=True)


class TestRandomizedPreprocessing:
    """fit(randomize_preprocessing=True) allocates one pipeline per resample,
    fits it on each subsample, and reports metrics per pipeline. Explicit
    option lists keep the defaults' t-SNE and UMAP out of most tests.
    """

    @staticmethod
    def make(**kwargs):
        options = dict(
            n_clusters=np.array([2, 3]),
            n_resamples=8,
            estimator_param_grids=[
                (KMeans, {"n_clusters": [2, 3], "n_init": [3]}),
                (AgglomerativeClustering, {"n_clusters": [2, 3], "linkage": ["ward"]}),
            ],
            normalization_options=[(FunctionTransformer, {}), (StandardScaler, {})],
            dim_reduction_options=[
                (FunctionTransformer, {}),
                (PCA, {"n_components": [2, 3]}),
            ],
            random_state=0,
            verbose=0,
        )
        options.update(kwargs)
        return CARVE(**options)

    def test_results_schema(self, randomized_fit):
        _, carve = randomized_fit
        df = carve.preprocessing_results_
        assert list(df.columns) == [
            "method_id",
            "method_label",
            "pipeline",
            "normalization",
            "dim_reduction",
            "n_clusters",
            "n_resamples",
            "ari_stability",
            "ari_stability_se",
            "ari_generalizability",
            "ari_generalizability_se",
            "n_clusters_observed",
            "sweep_param",
            "sweep_value",
            "sweep_rank",
        ]
        # Every configuration is covered, and within one the per-pipeline
        # counts add up to the run's resamples.
        totals = df.groupby(["method_id", "n_clusters"])["n_resamples"].sum()
        assert len(totals) == len(carve.estimator_results_)
        assert (totals == 8).all()
        assert df.groupby(["method_id", "n_clusters"])["pipeline"].nunique().min() > 1
        # Joins to estimator_results_ on method_id and the sweep column.
        keys = carve.estimator_results_[["method_id", "n_clusters", "config_id"]]
        joined = df.merge(
            keys, on=["method_id", "n_clusters"], how="left", validate="many_to_one"
        )
        assert joined["config_id"].notna().all()

    def test_rows_split_on_hyperparameters(self, randomized_fit):
        _, carve = randomized_fit
        reductions = set(carve.preprocessing_results_["dim_reduction"])
        assert {"PCA(n_components=2)", "PCA(n_components=3)"} <= reductions

    def test_pipelines_registry_matches_the_table(self, randomized_fit):
        X, carve = randomized_fit
        registry = carve.preprocessing_pipelines_
        assert set(registry) == set(carve.preprocessing_results_["pipeline"])
        for label, spec in registry.items():
            assert isinstance(spec, PipelineSpec)
            assert spec.label == label
            embedded = pipeline_from_spec(spec, random_state=0).fit_transform(X)
            assert embedded.shape[0] == X.shape[0]

    def test_same_seed_reproduces_and_another_seed_does_not(self, X_two_clusters):
        noise = make_noise_embedding()

        def fit(seed):
            model = self.make(
                normalization_options=[(FunctionTransformer, {})],
                dim_reduction_options=[(noise, {})],
                random_state=seed,
            )
            model.fit(X_two_clusters, randomize_preprocessing=True)
            return model.preprocessing_results_

        a, b, c = fit(0), fit(0), fit(1)
        pd.testing.assert_frame_equal(a, b)
        assert not np.allclose(a["ari_stability"], c["ari_stability"])

    def test_tsne_runs_end_to_end(self):
        """t-SNE has no transform method; the per-subsample fit and the
        raw-feature classifier are what let it take part at all."""
        assert not hasattr(TSNE, "transform")
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(40, 5) + [5, 0, 0, 0, 0], rng.randn(40, 5)])
        carve = CARVE(
            n_clusters=np.array([2, 3]),
            n_resamples=4,
            estimator_param_grids=[(KMeans, {"n_clusters": [2, 3], "n_init": [3]})],
            normalization_options=[(FunctionTransformer, {})],
            dim_reduction_options=[
                (FunctionTransformer, {}),
                (TSNE, {"n_components": [2], "perplexity": [15]}),
            ],
            random_state=0,
        ).fit(X, randomize_preprocessing=True)
        tsne = "identity | TSNE(n_components=2, perplexity=15)"
        assert set(carve.preprocessing_results_["pipeline"]) == {
            "identity | identity",
            tsne,
        }
        assert tsne in carve.preprocessing_pipelines_
        assert len(carve.estimator_results_) == 2
        assert carve.get_labels().shape == (80,)

    def test_none_without_randomization(self, X_two_clusters):
        carve = self.make().fit(X_two_clusters)
        assert carve.preprocessing_results_ is None
        assert carve.preprocessing_pipelines_ is None

    def test_registry_survives_save_and_load(self, randomized_fit, tmp_path):
        _, carve = randomized_fit
        path = tmp_path / "randomized.carve"
        carve.save(path)
        loaded = CARVE.load(path)
        assert loaded.preprocessing_pipelines_ == carve.preprocessing_pipelines_
        pd.testing.assert_frame_equal(
            loaded.preprocessing_results_, carve.preprocessing_results_
        )


```

In `tests/test_pipeline.py`, replace:

```python

import itertools
import random
from collections import Counter

```

with:

```python

import itertools
from collections import Counter

```

In `tests/test_pipeline.py`, replace:

```python
    PipelineSpec,
    PipelineStep,
    _choose_preprocessor,
    _parse_option,
    accepts_random_state,
    allocate_pipelines,
    build_preprocessing_pipeline,
    option_label,
    pipeline_from_spec,
    sample_preprocessing_pipeline,
)
from tests._helpers import make_noise_embedding
```

with:

```python
    PipelineSpec,
    PipelineStep,
    _parse_option,
    accepts_random_state,
    allocate_pipelines,
    option_label,
    pipeline_from_spec,
)
from tests._helpers import make_noise_embedding
```

In `tests/test_pipeline.py`, replace:

```python
        with pytest.raises(ValueError, match="Dict option must contain"):
            _parse_option({"params": {}})


# -----------------------------------------------------------------------
# build_preprocessing_pipeline
# -----------------------------------------------------------------------


class TestBuildPreprocessingPipeline:
    def test_identity_when_not_randomized(self):
        pipeline, norm_p, dr_p, norm_name, dr_name = build_preprocessing_pipeline(
            randomize_preprocessing=False,
            normalization_options=[],
            dim_reduction_options=[],
            seed=0,
        )
        assert isinstance(pipeline, Pipeline)
        assert norm_p == {}
        assert dr_p == {}
        assert norm_name == "Identity"
        assert dr_name == "Identity"

    def test_identity_passthrough(self):
        """Identity pipeline should not transform data."""
        pipeline, *_ = build_preprocessing_pipeline(
            randomize_preprocessing=False,
            normalization_options=[],
            dim_reduction_options=[],
            seed=0,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = pipeline.fit_transform(X)
        np.testing.assert_array_equal(result, X)

    def test_randomized(self):
        norm_opts = [
            (FunctionTransformer, {}),
            (StandardScaler, {}),
        ]
        dr_opts = [
            (FunctionTransformer, {}),
        ]
        pipeline, norm_p, dr_p, norm_name, dr_name = build_preprocessing_pipeline(
            randomize_preprocessing=True,
            normalization_options=norm_opts,
            dim_reduction_options=dr_opts,
            seed=42,
        )
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2

    def test_reproducibility(self):
        norm_opts = [
            (FunctionTransformer, {}),
            (StandardScaler, {}),
        ]
        dr_opts = [(FunctionTransformer, {})]

        _, _, _, name1, _ = build_preprocessing_pipeline(
            True,
            norm_opts,
            dr_opts,
            seed=42,
        )
        _, _, _, name2, _ = build_preprocessing_pipeline(
            True,
            norm_opts,
            dr_opts,
            seed=42,
        )
        assert name1 == name2


# -----------------------------------------------------------------------
# sample_preprocessing_pipeline
# -----------------------------------------------------------------------


class TestSamplePreprocessingPipeline:
    def test_basic(self):
        norm_opts = [(StandardScaler, {})]
        dr_opts = [(FunctionTransformer, {})]
        pipeline, norm_p, dr_p, norm_name, dr_name = sample_preprocessing_pipeline(
            norm_opts,
            dr_opts,
            seed=0,
        )
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert norm_name == "StandardScaler"
        assert dr_name == "FunctionTransformer"

    def test_transforms_data(self):
        norm_opts = [(StandardScaler, {})]
        dr_opts = [(FunctionTransformer, {})]
        pipeline, *_ = sample_preprocessing_pipeline(norm_opts, dr_opts, seed=0)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = pipeline.fit_transform(X)
        # StandardScaler should center the data
        assert abs(result.mean()) < 1e-10


# -----------------------------------------------------------------------
# _choose_preprocessor
# -----------------------------------------------------------------------


class TestChoosePreprocessor:
    def test_tuple2(self):
        rnd = random.Random(0)
        options = [(StandardScaler, {})]
        inst, params, name = _choose_preprocessor(rnd, options)
        assert isinstance(inst, StandardScaler)
        assert params == {}
        assert name == "StandardScaler"

    def test_tuple3_with_name(self):
        rnd = random.Random(0)
        options = [(StandardScaler, "MyScaler", {})]
        inst, params, name = _choose_preprocessor(rnd, options)
        assert isinstance(inst, StandardScaler)
        assert name == "MyScaler"

    def test_tuple2_with_params(self):
        rnd = random.Random(0)
        options = [(FunctionTransformer, {"func": [np.log1p, np.sqrt]})]
        inst, params, name = _choose_preprocessor(rnd, options)
        assert isinstance(inst, FunctionTransformer)
        assert "func" in params
        assert params["func"] in [np.log1p, np.sqrt]

    def test_dict_option(self):
        rnd = random.Random(0)
        options = [{"cls": StandardScaler, "params": {}, "name": "DictScaler"}]
        inst, params, name = _choose_preprocessor(rnd, options)
        assert isinstance(inst, StandardScaler)
        assert name == "DictScaler"

    def test_dict_without_name(self):
        rnd = random.Random(0)
        options = [{"cls": StandardScaler, "params": {}}]
        inst, params, name = _choose_preprocessor(rnd, options)
        assert name == "StandardScaler"

    def test_invalid_tuple_length(self):
        rnd = random.Random(0)
        options = [(StandardScaler,)]
        with pytest.raises(ValueError, match="Option tuples must be"):
            _choose_preprocessor(rnd, options)

    def test_invalid_type(self):
        rnd = random.Random(0)
        options = ["not_a_valid_option"]
        with pytest.raises(TypeError, match="Unsupported option type"):
            _choose_preprocessor(rnd, options)

    def test_dict_missing_cls(self):
        rnd = random.Random(0)
        options = [{"params": {}}]
        with pytest.raises(ValueError, match="Dict option must contain"):
            _choose_preprocessor(rnd, options)
```

with:

```python
        with pytest.raises(ValueError, match="Dict option must contain"):
            _parse_option({"params": {}})
```

In `tests/test_runner.py`, replace:

```python
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans

import carve._runner as carve_runner
import carve._utils as carve_utils
from carve._consensus import compute_consensus_metrics
from carve._runner import (
    ResampleResult,
    _compute_generalizability_ari,
    _compute_stability_ari,
    run_validation,
    validation_iter,
)
from carve._sweep import resolve_sweep
from carve._types import ConsensusSummary, ModePolicy
from tests._helpers import make_njobs_spy, make_parallel_spy


```

with:

```python
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer

import carve._runner as carve_runner
import carve._utils as carve_utils
from carve._consensus import compute_consensus_metrics
from carve._pipeline import allocate_pipelines
from carve._runner import (
    ResampleResult,
    _compute_generalizability_ari,
    _compute_stability_ari,
    _resample_indices,
    embed_resample,
    run_validation,
    validation_iter,
)
from carve._sweep import resolve_sweep
from carve._types import ConsensusSummary, ModePolicy, resolve_mode
from tests._helpers import (
    make_fit_input_spy,
    make_fit_recorder,
    make_noise_embedding,
    make_njobs_spy,
    make_parallel_spy,
)


```

In `tests/test_runner.py`, replace:

```python
            test_indices=np.array([2, 3]),
            stability_indices=np.array([4, 5]),
            normalization_params={},
            dim_reduction_params={},
            normalization_name="Identity",
            dim_reduction_name="Identity",
            n_clusters_train=2,
            n_clusters_test=2,
```

with:

```python
            test_indices=np.array([2, 3]),
            stability_indices=np.array([4, 5]),
            pipeline=None,
            n_clusters_train=2,
            n_clusters_test=2,
```

In `tests/test_runner.py`, replace:

```python
        assert r.ari_stability == 0.8
        assert r.ari_generalizability == 0.7
        assert r.normalization_name == "Identity"
        assert r.n_clusters_train == 2
        assert r.noise_fraction == 0.0
```

with:

```python
        assert r.ari_stability == 0.8
        assert r.ari_generalizability == 0.7
        assert r.pipeline is None
        assert r.n_clusters_train == 2
        assert r.noise_fraction == 0.0
```

In `tests/test_runner.py`, replace:

```python
            n_resamples=3,
            seed=0,
            normalization_options=[],
            dim_reduction_options=[],
            randomize_preprocessing=False,
            mode="default",
            random_state=0,
        )
        assert isinstance(result, ResampleResult)
        assert result.labels_train.shape[0] > 0
        assert result.train_indices.shape[0] > 0
```

with:

```python
            n_resamples=3,
            seed=0,
            mode="default",
            random_state=0,
        )
        assert isinstance(result, ResampleResult)
        assert result.pipeline is None
        assert result.labels_train.shape[0] > 0
        assert result.train_indices.shape[0] > 0
```

In `tests/test_runner.py`, replace:

```python
            n_resamples=3,
            seed=0,
            normalization_options=[],
            dim_reduction_options=[],
            randomize_preprocessing=False,
            mode="stability",
            random_state=0,
```

with:

```python
            n_resamples=3,
            seed=0,
            mode="stability",
            random_state=0,
```

In `tests/test_runner.py`, replace:

```python
            n_resamples=3,
            seed=0,
            normalization_options=[],
            dim_reduction_options=[],
            randomize_preprocessing=False,
            mode="generalizability",
            random_state=0,
```

with:

```python
            n_resamples=3,
            seed=0,
            mode="generalizability",
            random_state=0,
```

In `tests/test_runner.py`, replace:

```python
            n_resamples=3,
            seed=0,
            normalization_options=[],
            dim_reduction_options=[],
            classifier=spy(),
            randomize_preprocessing=False,
            mode="default",
            random_state=0,
```

with:

```python
            n_resamples=3,
            seed=0,
            classifier=spy(),
            mode="default",
            random_state=0,
```

In `tests/test_runner.py`, replace:

```python
        assert self.njobs_spy.seen == [2] * 6

    def test_budget_is_resolved_once_per_run(self, X_two_clusters):
        # Two configurations, one Parallel call each, the same split for both.
```

with:

```python
        assert self.njobs_spy.seen == [2] * 6

    def test_precompute_pass_uses_the_same_worker_count(self, X_two_clusters):
        run_validation(
            X=X_two_clusters,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_resamples=6,
            subsample_ratio=0.8,
            normalization_options=[(FunctionTransformer, {})],
            dim_reduction_options=[(FunctionTransformer, {})],
            classifier=self.njobs_spy(),
            randomize_preprocessing=True,
            n_jobs=4,
            random_state=0,
            verbose=0,
        )
        # One Parallel for the precompute pass, then one per configuration.
        assert self.parallel_spy.seen == [4, 4, 4]
        assert self.njobs_spy.seen == [2] * 12

    def test_budget_is_resolved_once_per_run(self, X_two_clusters):
        # Two configurations, one Parallel call each, the same split for both.
```

In `tests/test_runner.py`, replace:

```python
            n_resamples=3,
            seed=0,
            normalization_options=[],
            dim_reduction_options=[],
            randomize_preprocessing=False,
            mode="default",
            random_state=0,
```

with:

```python
            n_resamples=3,
            seed=0,
            mode="default",
            random_state=0,
```

In `tests/test_runner.py`, replace:

```python
            assert np.allclose(summary.ce, ce)
            assert np.isclose(summary.pac, pac)
```

with:

```python
            assert np.allclose(summary.ce, ce)
            assert np.isclose(summary.pac, pac)


# -----------------------------------------------------------------------
# Randomized preprocessing: fit scope, precompute, classifier features
# -----------------------------------------------------------------------


def _indexed(X):
    """X with each row's own index in column 0, so a recorder reads back
    exactly which samples a matrix holds."""
    X = np.array(X, dtype=float)
    X[:, 0] = np.arange(X.shape[0])
    return X


class TestRandomizedFitScope:
    """One pipeline per resample, fit separately on each subsample, once per
    resample rather than once per configuration, with the classifier on raw
    features.
    """

    N_RESAMPLES = 4
    RATIO = 0.7

    def _run(self, X, dr_class, *, n_clusters=(2,), mode="default", **kwargs):
        kwargs.setdefault("random_state", 0)
        return run_validation(
            X=X,
            estimator_grids=[
                (KMeans, {"n_clusters": list(n_clusters), "n_init": [3]})
            ],
            n_resamples=self.N_RESAMPLES,
            subsample_ratio=self.RATIO,
            normalization_options=[(FunctionTransformer, {})],
            dim_reduction_options=[(dr_class, {})],
            randomize_preprocessing=True,
            n_jobs=1,
            mode=mode,
            verbose=0,
            **kwargs,
        )

    def _subsets(self, n, mode="default"):
        """Every subset a run draws, in (P_1, P_2, P_test) order per resample,
        skipping the ones the mode does not use."""
        policy = resolve_mode(mode)
        subsets = []
        for b in range(self.N_RESAMPLES):
            P_1, P_test, P_2 = _resample_indices(
                n,
                seed=b,
                n_resamples=self.N_RESAMPLES,
                subsample_ratio=self.RATIO,
                random_state=0,
                run_stability=policy.run_stability,
            )
            subsets.append(tuple(P_1))
            if policy.run_stability:
                subsets.append(tuple(P_2))
            if policy.run_generalizability:
                subsets.append(tuple(P_test))
        return subsets

    @pytest.mark.parametrize("mode", ["default", "stability", "generalizability"])
    def test_each_subsample_is_fit_separately(self, X_two_clusters, mode):
        recorder = make_fit_recorder()
        X = _indexed(X_two_clusters)
        self._run(X, recorder, mode=mode)
        assert Counter(recorder.seen) == Counter(self._subsets(X.shape[0], mode))
        assert X.shape[0] not in {len(rows) for rows in recorder.seen}

    def test_once_per_resample_not_per_configuration(self, X_two_clusters):
        recorder = make_fit_recorder()
        self._run(_indexed(X_two_clusters), recorder, n_clusters=(2, 3))
        assert len(recorder.seen) == self.N_RESAMPLES * 3

    def test_classifier_fits_and_predicts_on_raw_features(self, X_two_clusters):
        spy = make_fit_input_spy()
        X = _indexed(X_two_clusters)
        records, *_ = self._run(X, make_noise_embedding(), classifier=spy())

        # The embedding is two columns of noise; the classifier must see the
        # raw rows, all p columns, of P_1 when fitting and P_test when predicting.
        assert [m.shape[1] for m in spy.fitted] == [X.shape[1]] * self.N_RESAMPLES
        for m in spy.fitted + spy.predicted:
            np.testing.assert_array_equal(m, X[m[:, 0].astype(int)])
        subsets = self._subsets(X.shape[0], "generalizability")
        assert Counter(tuple(m[:, 0].astype(int)) for m in spy.fitted) == Counter(
            subsets[0::2]
        )
        assert Counter(tuple(m[:, 0].astype(int)) for m in spy.predicted) == Counter(
            subsets[1::2]
        )
        assert np.isfinite(records[0]["ari_generalizability"])

    def test_each_subset_gets_its_own_derived_seed(self, X_two_clusters):
        noise = make_noise_embedding()
        self._run(X_two_clusters, noise, random_state=10)
        B = self.N_RESAMPLES
        expected = [10 + b + offset for b in range(B) for offset in (0, B, 2 * B)]
        assert sorted(noise.seen) == sorted(expected)

    def test_records_carry_join_keys_and_the_allocated_pipelines(
        self, X_two_clusters
    ):
        norm = [(FunctionTransformer, {})]
        dr = [(FunctionTransformer, {}), (PCA, {"n_components": [2, 3]})]
        records, pipeline_records, *_ = run_validation(
            X=X_two_clusters,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3], "n_init": [3]})],
            n_resamples=6,
            subsample_ratio=0.7,
            normalization_options=norm,
            dim_reduction_options=dr,
            randomize_preprocessing=True,
            n_jobs=1,
            random_state=3,
            verbose=0,
        )
        allocated = allocate_pipelines(norm, dr, 6, 3)
        assert len(pipeline_records) == len(records) == 2
        for record, pipeline_record in zip(records, pipeline_records):
            for key in ("method_id", "method_label", "sweep_value", "sweep_rank"):
                assert pipeline_record[key] == record[key]
            assert [r.pipeline for r in pipeline_record["results"]] == allocated

    def test_no_pipeline_records_without_randomization(self, X_two_clusters):
        out = run_validation(
            X=X_two_clusters,
            estimator_grids=[(KMeans, {"n_clusters": [2]})],
            n_resamples=3,
            subsample_ratio=0.8,
            normalization_options=[(FunctionTransformer, {})],
            dim_reduction_options=[(FunctionTransformer, {})],
            n_jobs=1,
            random_state=0,
            verbose=0,
        )
        assert out[1] == []


class TestEmbedResample:
    def _spec(self):
        return allocate_pipelines(
            [(FunctionTransformer, {})], [(FunctionTransformer, {})], 3, 0
        )[0]

    def test_identity_embeddings_reproduce_the_raw_path(self, X_two_clusters):
        kwargs = dict(
            X=X_two_clusters,
            est_class=KMeans,
            params={"n_clusters": 2},
            subsample_ratio=0.8,
            n_resamples=3,
            seed=1,
            random_state=0,
        )
        spec = self._spec()
        embeddings = embed_resample(
            X_two_clusters,
            spec,
            seed=1,
            n_resamples=3,
            subsample_ratio=0.8,
            random_state=0,
        )
        raw = validation_iter(**kwargs)
        embedded = validation_iter(**kwargs, embeddings=embeddings)
        assert embedded.ari_stability == raw.ari_stability
        assert embedded.ari_generalizability == raw.ari_generalizability
        np.testing.assert_array_equal(embedded.labels_train, raw.labels_train)
        np.testing.assert_array_equal(embedded.labels_predicted, raw.labels_predicted)
        assert embedded.pipeline == spec
        assert raw.pipeline is None

    def test_mode_skips_the_unused_subsets(self, X_two_clusters):
        common = dict(seed=0, n_resamples=3, subsample_ratio=0.8, random_state=0)
        stab = embed_resample(X_two_clusters, self._spec(), mode="stability", **common)
        gen = embed_resample(
            X_two_clusters, self._spec(), mode="generalizability", **common
        )
        assert stab.X_test is None and stab.X_2.shape == (48, 5)
        assert gen.X_2 is None and gen.X_test.shape == (12, 5)

    def test_seeded_umap_fits_without_its_parallelism_warning(self):
        # filterwarnings = error turns umap-learn's warning that the seed
        # disables its parallelism into a failure unless embed_resample
        # filters it.
        umap = pytest.importorskip("umap")
        X = np.random.RandomState(0).randn(100, 5)
        spec = allocate_pipelines(
            [(FunctionTransformer, {})],
            [(umap.UMAP, {"n_components": [2], "n_neighbors": [15]})],
            1,
            0,
        )[0]
        embeddings = embed_resample(
            X, spec, seed=0, n_resamples=1, subsample_ratio=0.7, random_state=0
        )
        assert embeddings.X_1.shape == (70, 2)
        assert embeddings.X_2.shape == (70, 2)
        assert embeddings.X_test.shape == (30, 2)

    def test_mismatched_embeddings_raise(self, X_two_clusters):
        good = embed_resample(
            X_two_clusters,
            self._spec(),
            seed=0,
            n_resamples=3,
            subsample_ratio=0.8,
            random_state=0,
        )
        with pytest.raises(RuntimeError, match="X_test has 11 rows but .* has 12"):
            validation_iter(
                X=X_two_clusters,
                est_class=KMeans,
                params={"n_clusters": 2},
                subsample_ratio=0.8,
                n_resamples=3,
                seed=0,
                random_state=0,
                embeddings=good._replace(X_test=good.X_test[:-1]),
            )


class TestRandomizedProgress:
    def _run(self, X, randomize):
        run_validation(
            X=X,
            estimator_grids=[(KMeans, {"n_clusters": [2]})],
            n_resamples=3,
            subsample_ratio=0.8,
            normalization_options=[(FunctionTransformer, {})],
            dim_reduction_options=[(FunctionTransformer, {})],
            randomize_preprocessing=randomize,
            n_jobs=1,
            random_state=0,
            show_progress=True,
            verbose=0,
        )

    def test_preprocessing_bar_precedes_the_configuration_bar(
        self, X_two_clusters, capsys
    ):
        self._run(X_two_clusters, randomize=True)
        err = capsys.readouterr().err
        assert "Preprocessing" in err
        assert err.index("Preprocessing") < err.index("Grid configs")

    def test_no_preprocessing_bar_without_randomization(self, X_two_clusters, capsys):
        self._run(X_two_clusters, randomize=False)
        err = capsys.readouterr().err
        assert "Grid configs" in err
        assert "Preprocessing" not in err
```

In `tests/test_utils.py`, replace:

```python
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import carve._utils as carve_utils
from carve._runner import ResampleResult
from carve._utils import (
```

with:

```python
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer

import carve._utils as carve_utils
from carve._pipeline import PipelineSpec, PipelineStep
from carve._runner import ResampleResult
from carve._utils import (
```

In `tests/test_utils.py`, replace:

```python


def _pipeline_record(sweep_param, sweep_value):
    """One randomized-preprocessing record with two identity-pipeline runs."""
    blank = dict.fromkeys(
        (
```

with:

```python


def _spec(dr_step):
    return PipelineSpec(PipelineStep(FunctionTransformer, {}, None), dr_step)


PCA2 = _spec(PipelineStep(PCA, {"n_components": 2}, None))
PCA3 = _spec(PipelineStep(PCA, {"n_components": 3}, None))


def _result(stability, generalizability, spec, k=3):
    """One ResampleResult carrying only what the summary reads."""
    blank = dict.fromkeys(
        (
```

In `tests/test_utils.py`, replace:

```python
        None,
    )
    results = [
        ResampleResult(
            ari_stability=s,
            ari_generalizability=g,
            normalization_params={},
            dim_reduction_params={},
            normalization_name="StandardScaler",
            dim_reduction_name="PCA",
            n_clusters_train=3,
            n_clusters_test=3,
            n_clusters_stability=3,
            noise_fraction=0.0,
            **blank,
        )
        for s, g in [(0.8, 0.7), (0.6, 0.5)]
    ]
    return {
        "estimator": "KMeans",
        "params": {sweep_param: sweep_value},
        "results": results,
    }


class TestSummarizePreprocessingRecords:
    def test_groups_on_n_clusters_by_default(self):
        records = [_pipeline_record("n_clusters", 2), _pipeline_record("n_clusters", 3)]
        out = summarize_preprocessing_records(records)
        assert "n_clusters" in out.columns
        assert set(out["n_clusters"]) == {2, 3}
        assert set(out["norm__func"]) == {"StandardScaler"}
        assert set(out["dr__method"]) == {"PCA"}

    def test_averages_within_a_group(self):
        out = summarize_preprocessing_records([_pipeline_record("n_clusters", 2)])
        assert out["ari_stability"].iloc[0] == pytest.approx(0.7)
        assert out["ari_generalizability"].iloc[0] == pytest.approx(0.6)

    def test_groups_on_sweep_param(self):
        records = [
            _pipeline_record("resolution", 0.5),
            _pipeline_record("resolution", 1.0),
        ]
        out = summarize_preprocessing_records(records, sweep_param="resolution")
        assert "resolution" in out.columns
        assert "n_clusters" not in out.columns
        assert set(out["resolution"]) == {0.5, 1.0}


```

with:

```python
        None,
    )
    return ResampleResult(
        ari_stability=stability,
        ari_generalizability=generalizability,
        pipeline=spec,
        n_clusters_train=k,
        n_clusters_test=k,
        n_clusters_stability=k,
        noise_fraction=0.0,
        **blank,
    )


def _record(method_id, sweep_value, results, sweep_rank=0):
    """One configuration's record, shaped as run_validation returns it."""
    return {
        "method_id": method_id,
        "method_label": f"KMeans {method_id}",
        "sweep_value": sweep_value,
        "sweep_rank": sweep_rank,
        "results": results,
    }


class TestSummarizePreprocessingRecords:
    def test_columns(self):
        out, _ = summarize_preprocessing_records(
            [_record("m0", 2, [_result(0.8, 0.7, PCA2)])]
        )
        assert list(out.columns) == [
            "method_id",
            "method_label",
            "pipeline",
            "normalization",
            "dim_reduction",
            "n_clusters",
            "n_resamples",
            "ari_stability",
            "ari_stability_se",
            "ari_generalizability",
            "ari_generalizability_se",
            "n_clusters_observed",
            "sweep_param",
            "sweep_value",
            "sweep_rank",
        ]

    def test_hyperparameters_split_rows(self):
        out, pipelines = summarize_preprocessing_records(
            [_record("m0", 2, [_result(0.8, 0.7, PCA2), _result(0.4, 0.3, PCA3)])]
        )
        assert list(out["pipeline"]) == [
            "identity | PCA(n_components=2)",
            "identity | PCA(n_components=3)",
        ]
        assert list(out["normalization"]) == ["identity", "identity"]
        assert list(out["dim_reduction"]) == [
            "PCA(n_components=2)",
            "PCA(n_components=3)",
        ]
        assert pipelines == {PCA2.label: PCA2, PCA3.label: PCA3}

    def test_estimators_split_rows_and_sort_numerically(self):
        records = [
            _record("m10", 2, [_result(0.5, 0.5, PCA2)]),
            _record("m2", 2, [_result(0.9, 0.9, PCA2)]),
        ]
        out, _ = summarize_preprocessing_records(records)
        assert list(out["method_id"]) == ["m2", "m10"]
        assert list(out["ari_stability"]) == [0.9, 0.5]

    def test_counts_sum_to_the_run_within_each_configuration(self):
        records = [
            _record(
                "m0",
                2,
                [_result(0.8, 0.7, PCA2), _result(0.6, 0.5, PCA2), _result(0.4, 0.3, PCA3)],
            ),
            _record(
                "m0",
                3,
                [_result(0.8, 0.7, PCA3), _result(0.6, 0.5, PCA2), _result(0.4, 0.3, PCA3)],
                sweep_rank=1,
            ),
        ]
        out, _ = summarize_preprocessing_records(records)
        assert out.groupby("n_clusters")["n_resamples"].sum().to_dict() == {2: 3, 3: 3}
        # Sorted by pipeline, then sweep value.
        assert list(out["n_resamples"]) == [2, 1, 1, 2]

    def test_standard_error_uses_only_the_rows_resamples(self):
        out, _ = summarize_preprocessing_records(
            [
                _record(
                    "m0",
                    2,
                    [
                        _result(0.8, 0.7, PCA2),
                        _result(0.6, 0.5, PCA2),
                        _result(0.1, 0.0, PCA3),
                        _result(0.3, 0.2, PCA3),
                    ],
                )
            ]
        )
        pca2, pca3 = out.iloc[0], out.iloc[1]
        assert pca2["ari_stability"] == pytest.approx(0.7)
        assert pca2["ari_stability_se"] == pytest.approx(0.1)
        assert pca2["ari_generalizability_se"] == pytest.approx(0.1)
        assert pca3["ari_stability"] == pytest.approx(0.2)
        assert pca3["ari_stability_se"] == pytest.approx(0.1)

    def test_single_resample_has_no_standard_error(self):
        out, _ = summarize_preprocessing_records(
            [_record("m0", 2, [_result(0.8, 0.7, PCA2)])]
        )
        assert np.isnan(out.loc[0, "ari_stability_se"])

    def test_skipped_criterion_is_nan(self):
        out, _ = summarize_preprocessing_records(
            [_record("m0", 2, [_result(0.8, np.nan, PCA2), _result(0.6, np.nan, PCA2)])]
        )
        assert out.loc[0, "ari_stability"] == pytest.approx(0.7)
        assert np.isnan(out.loc[0, "ari_generalizability"])
        assert np.isnan(out.loc[0, "ari_generalizability_se"])

    def test_observed_cluster_count_is_the_rows_mean(self):
        out, _ = summarize_preprocessing_records(
            [
                _record(
                    "m0",
                    2,
                    [
                        _result(0.8, 0.7, PCA2, k=2),
                        _result(0.6, 0.5, PCA2, k=3),
                        _result(0.4, 0.3, PCA3, k=5),
                    ],
                )
            ]
        )
        assert list(out["n_clusters_observed"]) == [2.5, 5.0]

    def test_sweep_column_follows_the_sweep_param(self):
        out, _ = summarize_preprocessing_records(
            [_record("m0", 0.5, [_result(0.8, 0.7, PCA2)])], sweep_param="resolution"
        )
        assert "resolution" in out.columns
        assert "n_clusters" not in out.columns
        assert out.loc[0, "resolution"] == 0.5
        assert out.loc[0, "sweep_param"] == "resolution"

    def test_empty(self):
        out, pipelines = summarize_preprocessing_records([])
        assert out.empty
        assert "pipeline" in out.columns
        assert pipelines == {}


```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_runner.py tests/test_utils.py::TestSummarizePreprocessingRecords tests/test_api.py::TestRandomizedPreprocessing tests/test_api.py::TestRefit -q`

Expected: `1 error`. The first distinct errors:

- `ImportError: cannot import name '_resample_indices' from 'carve._runner'`

- [ ] Step 3: Implement

Edit `src/carve/_runner.py` first, then `_pipeline.py`, `_types.py`, `_utils.py` and `api.py`. The edits below are in file order.

In `src/carve/_pipeline.py`, replace:

```python
import inspect
import itertools
import random
from dataclasses import dataclass
from typing import Any, NamedTuple
```

with:

```python
import inspect
import itertools
from dataclasses import dataclass
from typing import Any, NamedTuple
```

In `src/carve/_pipeline.py`, replace:

```python
        ]
    )


def build_preprocessing_pipeline(
    randomize_preprocessing: bool,
    normalization_options: list[PreprocOption],
    dim_reduction_options: list[PreprocOption],
    seed: int,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any], str, str]:
    """Build a preprocessing pipeline.

    Parameters
    ----------
    randomize_preprocessing : bool
        If True, randomly sample normalization and DR options.
    normalization_options : list
        Normalization options compatible with ``_choose_preprocessor``.
    dim_reduction_options : list
        Dimensionality reduction options compatible with ``_choose_preprocessor``.
    seed : int
        Random seed for sampling preprocessors.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Assembled preprocessing pipeline.
    normalization_params : dict
        Sampled normalization parameters.
    dim_reduction_params : dict
        Sampled dimensionality reduction parameters.
    normalization_name : str
        Resolved name for the normalization step.
    dim_reduction_name : str
        Resolved name for the DR step.
    """
    if randomize_preprocessing:
        (
            pipeline,
            normalization_params,
            dim_reduction_params,
            normalization_name,
            dim_reduction_name,
        ) = sample_preprocessing_pipeline(
            normalization_options, dim_reduction_options, seed
        )
    else:
        pipeline = Pipeline([("id", FunctionTransformer(lambda x: x))])
        normalization_params = dim_reduction_params = {}
        normalization_name = dim_reduction_name = "Identity"

    return (
        pipeline,
        normalization_params,
        dim_reduction_params,
        normalization_name,
        dim_reduction_name,
    )


def sample_preprocessing_pipeline(
    normalization_options: list[PreprocOption],
    dim_reduction_options: list[PreprocOption],
    seed: int,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any], str, str]:
    """Randomly sample normalization and DR steps into a pipeline.

    Parameters
    ----------
    normalization_options : list
        Normalization options compatible with ``_choose_preprocessor``.
    dim_reduction_options : list
        DR options compatible with ``_choose_preprocessor``.
    seed : int
        Random seed for sampling.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Assembled pipeline with two steps: normalization and DR.
    normalization_params : dict
        Sampled normalization parameters.
    dim_reduction_params : dict
        Sampled dimensionality reduction parameters.
    normalization_name : str
        Resolved name for the normalization step.
    dim_reduction_name : str
        Resolved name for the DR step.
    """
    rnd = random.Random(seed)

    normalization_step, normalization_params, normalization_name = _choose_preprocessor(
        rnd, normalization_options
    )
    dim_reduction_step, dim_reduction_params, dim_reduction_name = _choose_preprocessor(
        rnd, dim_reduction_options
    )

    pipeline = Pipeline([("norm", normalization_step), ("dr", dim_reduction_step)])
    return (
        pipeline,
        normalization_params,
        dim_reduction_params,
        normalization_name,
        dim_reduction_name,
    )


def _choose_preprocessor(
    rnd: random.Random,
    options: list[PreprocOption],
) -> tuple[TransformerMixin, dict[str, Any], str]:
    """Randomly select and instantiate a preprocessor from a set of options.

    Parameters
    ----------
    rnd : random.Random
        Random generator for sampling.
    options : list
        Options supporting:
        - (cls, params)
        - (cls, name, params)
        - {'cls': cls, 'params': {...}, 'name': optional}

    Returns
    -------
    transformer : TransformerMixin
        Instantiated transformer with sampled parameters.
    chosen_params : dict
        Sampled parameter values.
    name : str
        Resolved display name for the transformer.

    Raises
    ------
    ValueError
        If an option tuple has an unsupported arity.
    TypeError
        If an option has an unsupported type.
    """
    option = rnd.choice(options)
    name = None
    params: dict[str, list[Any]] = {}

    if isinstance(option, tuple):
        if len(option) == 3:
            cls, name, params = option
        elif len(option) == 2:
            cls, params = option
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

        params = option.get("params") or option.get("grid") or {}
        name = option.get("name")

    else:
        raise TypeError(
            "Unsupported option type. Use (cls, params), (cls, name, params), or {'cls','params','name'}."
        )

    chosen_params = {k: rnd.choice(v) for k, v in (params or {}).items()}
    inst = cls(**chosen_params)
    resolved_name = name or inst.__class__.__name__
    return inst, chosen_params, resolved_name
```

with:

```python
        ]
    )
```

In `src/carve/_runner.py`, replace:

```python
Coordinates the resampling loop: for each estimator configuration it
draws subsamples, runs clustering, collects ARI scores, builds consensus
matrices, and computes generalizability scores.
"""

```

with:

```python
Coordinates the resampling loop: for each estimator configuration it
draws subsamples, runs clustering, collects ARI scores, builds consensus
matrices, and computes generalizability scores. Under randomized
preprocessing a precompute pass first fits each resample's pipeline on each
of its subsamples, once per resample, before any configuration runs.
"""

```

In `src/carve/_runner.py`, replace:

```python
)
from ._output import _log_config_progress
from ._pipeline import build_preprocessing_pipeline
from ._sweep import MethodIds, SweepSpec
from ._sweep import resolve_sweep as _resolve_sweep
```

with:

```python
)
from ._output import _log_config_progress
from ._pipeline import PipelineSpec, allocate_pipelines, pipeline_from_spec
from ._sweep import MethodIds, SweepSpec
from ._sweep import resolve_sweep as _resolve_sweep
```

In `src/carve/_runner.py`, replace:

```python
    stability_indices : ndarray or None
        Row indices of the second subsample used for stability.
    normalization_params : dict
        Parameters of the normalization transformer used.
    dim_reduction_params : dict
        Parameters of the dimensionality reduction transformer used.
    normalization_name : str
        Class name of the normalization transformer.
    dim_reduction_name : str
        Class name of the dimensionality reduction transformer.
    n_clusters_train : int
        Number of non-noise clusters found on the training subsample.
```

with:

```python
    stability_indices : ndarray or None
        Row indices of the second subsample used for stability.
    pipeline : PipelineSpec or None
        The preprocessing pipeline this resample used, or None when
        preprocessing was not randomized.
    n_clusters_train : int
        Number of non-noise clusters found on the training subsample.
```

In `src/carve/_runner.py`, replace:

```python
    test_indices: np.ndarray
    stability_indices: np.ndarray
    normalization_params: dict[str, Any]
    dim_reduction_params: dict[str, Any]
    normalization_name: str
    dim_reduction_name: str
    n_clusters_train: int
    n_clusters_test: int
    n_clusters_stability: int
    noise_fraction: float


```

with:

```python
    test_indices: np.ndarray
    stability_indices: np.ndarray
    pipeline: PipelineSpec | None
    n_clusters_train: int
    n_clusters_test: int
    n_clusters_stability: int
    noise_fraction: float


class ResampleEmbeddings(NamedTuple):
    """One resample's subsamples, each embedded by its own pipeline fit.

    The pipeline is fitted separately on every subsample, so the arrays come
    from independent fits and do not share coordinates.

    Attributes
    ----------
    spec : PipelineSpec
        The pipeline allocated to this resample.
    X_1 : ndarray of shape (n_train, d)
        Embedding of the first subsample.
    X_2 : ndarray of shape (n_train, d) or None
        Embedding of the second subsample; None when stability is skipped.
    X_test : ndarray of shape (n_test, d) or None
        Embedding of the held-out set; None when generalizability is skipped.
    """

    spec: PipelineSpec
    X_1: np.ndarray
    X_2: np.ndarray | None
    X_test: np.ndarray | None


def _resample_indices(
    n_samples: int,
    *,
    seed: int,
    n_resamples: int,
    subsample_ratio: float,
    random_state: int | None,
    run_stability: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return one resample's subsample indices: P_1, P_test and P_2.

    The only place these seeds are derived, so the precompute pass and the
    configuration loop draw the same subsamples by construction. P_2 is None
    when stability is skipped.
    """
    random_state0 = random_state if random_state is not None else 0

    P_1_idx, P_test_idx = split_subsample_indices(
        n_samples, subsample_ratio=subsample_ratio, random_state=random_state0 + seed
    )

    P_2_idx = None
    if run_stability:
        P_2_idx, _ = split_subsample_indices(
            n_samples,
            subsample_ratio=subsample_ratio,
            random_state=random_state0 + seed + n_resamples,
        )

    return P_1_idx, P_test_idx, P_2_idx


def embed_resample(
    X: np.ndarray,
    spec: PipelineSpec,
    *,
    seed: int,
    n_resamples: int,
    subsample_ratio: float,
    random_state: int | None,
    mode: RunMode = "default",
) -> ResampleEmbeddings:
    """Fit one resample's pipeline on each subsample the mode needs.

    The pipeline fitted on P_1 is seeded with ``random_state + seed``, on
    P_2 with ``random_state + seed + n_resamples`` and on P_test with
    ``random_state + seed + 2 * n_resamples``, so no two fits share a seed
    and no RNG state crosses a joblib boundary.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    spec : PipelineSpec
        The pipeline allocated to this resample.
    seed : int
        Resample index, the per-resample seed offset.
    n_resamples : int
        Total resamples, used to offset seeds.
    subsample_ratio : float
        Proportion of samples in each subsample.
    random_state : int or None
        Base seed; None means 0.
    mode : {"default", "stability", "generalizability"}, default="default"
        P_2 is embedded only when stability runs, P_test only when
        generalizability runs.

    Returns
    -------
    embeddings : ResampleEmbeddings
    """
    policy = resolve_mode(mode)
    random_state0 = random_state if random_state is not None else 0

    P_1_idx, P_test_idx, P_2_idx = _resample_indices(
        X.shape[0],
        seed=seed,
        n_resamples=n_resamples,
        subsample_ratio=subsample_ratio,
        random_state=random_state0,
        run_stability=policy.run_stability,
    )

    base = random_state0 + seed
    X_2 = X_test = None

    with warnings.catch_warnings():
        # umap-learn warns on every seeded fit that the seed disables its own
        # parallelism. CARVE seeds every transformer so the fit is
        # reproducible and parallelizes over resamples instead, so the warning
        # carries nothing here, and under warnings-as-errors it aborts the fit.
        warnings.filterwarnings(
            "ignore",
            message=r"n_jobs value .* overridden to 1 by setting random_state",
            category=UserWarning,
        )
        X_1 = np.asarray(pipeline_from_spec(spec, base).fit_transform(X[P_1_idx]))

        if policy.run_stability:
            pipeline = pipeline_from_spec(spec, base + n_resamples)
            X_2 = np.asarray(pipeline.fit_transform(X[P_2_idx]))

        if policy.run_generalizability:
            pipeline = pipeline_from_spec(spec, base + 2 * n_resamples)
            X_test = np.asarray(pipeline.fit_transform(X[P_test_idx]))

    return ResampleEmbeddings(spec=spec, X_1=X_1, X_2=X_2, X_test=X_test)


def precompute_embeddings(
    X: np.ndarray,
    pipelines: list[PipelineSpec],
    *,
    subsample_ratio: float,
    random_state: int | None,
    mode: RunMode,
    n_jobs: int,
    show_progress: bool = False,
) -> list[ResampleEmbeddings]:
    """Embed every resample's subsamples once, before any configuration runs.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    pipelines : list of PipelineSpec
        Entry ``b`` is resample ``b``'s pipeline, from
        ``_pipeline.allocate_pipelines``; its length is the resample count.
    subsample_ratio : float
        Proportion of samples in each subsample.
    random_state : int or None
        Base seed; None means 0.
    mode : {"default", "stability", "generalizability"}
        Which subsamples to embed.
    n_jobs : int
        Worker count over resamples, the same one the configuration loop uses.
    show_progress : bool, default=False
        Display a tqdm bar labeled "Preprocessing".

    Returns
    -------
    embeddings : list of ResampleEmbeddings
        Entry ``b`` belongs to resample ``b``.
    """
    n_resamples = len(pipelines)
    worker = delayed(embed_resample)
    tasks = (
        worker(
            X,
            pipelines[b],
            seed=b,
            n_resamples=n_resamples,
            subsample_ratio=subsample_ratio,
            random_state=random_state,
            mode=mode,
        )
        for b in range(n_resamples)
    )
    results = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
    return list(
        tqdm(
            results,
            total=n_resamples,
            desc="Preprocessing",
            disable=not show_progress,
        )
    )


```

In `src/carve/_runner.py`, replace:

```python
        Number of trees in the default random-forest classifier.
    randomize_preprocessing : bool, default=False
        Whether to sample preprocessing randomly per resample.
    n_jobs : int, default=1
        Core budget for the run. Split once by ``resolve_core_budget`` into
        workers over resamples and threads per worker for the classifier.
    random_state : int or None, default=None
        Random seed for reproducibility.
```

with:

```python
        Number of trees in the default random-forest classifier.
    randomize_preprocessing : bool, default=False
        Whether to allocate one pipeline per resample from the option lists
        (``_pipeline.allocate_pipelines``) and cluster each subsample's own
        embedding of it. The embeddings are computed once, before the
        configuration loop, by ``precompute_embeddings``.
    n_jobs : int, default=1
        Core budget for the run. Split once by ``resolve_core_budget`` into
        workers over resamples and threads per worker for the classifier.
        The precompute pass uses the same worker count.
    random_state : int or None, default=None
        Random seed for reproducibility.
```

In `src/carve/_runner.py`, replace:

```python
        One record per estimator configuration with aggregate metrics.
    pipeline_records : list of dict
        Optional per-resample preprocessing records when randomized.
    consensus_matrices : list of ndarray
        Consensus matrices for each configuration.
```

with:

```python
        One record per estimator configuration with aggregate metrics.
    pipeline_records : list of dict
        One record per configuration when randomized, empty otherwise:
        ``method_id``, ``method_label``, ``sweep_value``, ``sweep_rank`` and
        ``results``, the configuration's ``ResampleResult`` list.
    consensus_matrices : list of ndarray
        Consensus matrices for each configuration.
```

In `src/carve/_runner.py`, replace:

```python
    )

    with tqdm(
        total=total_configs, desc="Grid configs", disable=not show_progress
```

with:

```python
    )

    # --- Randomized preprocessing: allocate, then embed once per resample ---
    # Nothing here is per configuration, so the config_id alignment fit()
    # asserts is untouched.
    embeddings: list[ResampleEmbeddings] | None = None
    if randomize_preprocessing:
        pipelines = allocate_pipelines(
            normalization_options, dim_reduction_options, n_resamples, random_state
        )
        embeddings = precompute_embeddings(
            X,
            pipelines,
            subsample_ratio=subsample_ratio,
            random_state=random_state,
            mode=mode,
            n_jobs=outer_n_jobs,
            show_progress=show_progress,
        )

    with tqdm(
        total=total_configs, desc="Grid configs", disable=not show_progress
```

In `src/carve/_runner.py`, replace:

```python
                        n_resamples=n_resamples,
                        seed=b,
                        normalization_options=normalization_options,
                        dim_reduction_options=dim_reduction_options,
                        classifier=classifier,
                        n_trees=n_trees,
                        randomize_preprocessing=randomize_preprocessing,
                        sweep_param=sweep.param,
                        noise_policy=noise_policy,
                        mode=mode,
                        random_state=random_state,
                        classifier_n_jobs=classifier_n_jobs,
                    )
                    for b in range(n_resamples)
```

with:

```python
                        n_resamples=n_resamples,
                        seed=b,
                        classifier=classifier,
                        n_trees=n_trees,
                        sweep_param=sweep.param,
                        noise_policy=noise_policy,
                        mode=mode,
                        random_state=random_state,
                        classifier_n_jobs=classifier_n_jobs,
                        # Each worker receives only its own resample's arrays.
                        embeddings=None if embeddings is None else embeddings[b],
                    )
                    for b in range(n_resamples)
```

In `src/carve/_runner.py`, replace:

```python
                    pipeline_records.append(
                        {
                            "estimator": est_class.__name__,
                            "params": params,
                            "results": results,
                        }
```

with:

```python
                    pipeline_records.append(
                        {
                            "method_id": method_id,
                            "method_label": method_label,
                            "sweep_value": sweep_value,
                            "sweep_rank": record["sweep_rank"],
                            "results": results,
                        }
```

In `src/carve/_runner.py`, replace:

```python
    n_resamples: int,
    seed: int,
    normalization_options: list[PreprocSpec],
    dim_reduction_options: list[PreprocSpec],
    classifier: ClassifierMixin | None = None,
    n_trees: int = 100,
    sweep_param: str = "n_clusters",
    noise_policy: NoisePolicy = "drop",
    randomize_preprocessing: bool = False,
    mode: RunMode = "default",
    random_state: int = None,
    classifier_n_jobs: int = 1,
) -> ResampleResult:
    """Run a single resampling iteration for one estimator configuration.
```

with:

```python
    n_resamples: int,
    seed: int,
    classifier: ClassifierMixin | None = None,
    n_trees: int = 100,
    sweep_param: str = "n_clusters",
    noise_policy: NoisePolicy = "drop",
    mode: RunMode = "default",
    random_state: int = None,
    classifier_n_jobs: int = 1,
    embeddings: ResampleEmbeddings | None = None,
) -> ResampleResult:
    """Run a single resampling iteration for one estimator configuration.
```

In `src/carve/_runner.py`, replace:

```python
    seed : int
        Per-resample seed offset.
    normalization_options : list
        Normalization options.
    dim_reduction_options : list
        Dimensionality reduction options.
    classifier : sklearn classifier instance, optional
        Classifier used to score generalizability.
```

with:

```python
    seed : int
        Per-resample seed offset.
    classifier : sklearn classifier instance, optional
        Classifier used to score generalizability.
```

In `src/carve/_runner.py`, replace:

```python
    noise_policy : {"drop", "as_cluster", "singleton"}, default="drop"
        How to resolve negative labels emitted by density-based methods.
    randomize_preprocessing : bool, default=False
        Whether to randomize preprocessing.
    mode : Literal['default', 'stability', 'generalizability'], default='default'
        Determines whether to run CARVE regularly ('default') or whether
```

with:

```python
    noise_policy : {"drop", "as_cluster", "singleton"}, default="drop"
        How to resolve negative labels emitted by density-based methods.
    mode : Literal['default', 'stability', 'generalizability'], default='default'
        Determines whether to run CARVE regularly ('default') or whether
```

In `src/carve/_runner.py`, replace:

```python
        Threads for this resample's classifier, its share of the run's core
        budget.

    Returns
    -------
    result : ResampleResult
        Named tuple containing ARI metrics, labels, sample indices, and
        preprocessing metadata for this resample.
    """
    policy = resolve_mode(mode)

    n_samples = X.shape[0]
    random_state0 = random_state if random_state is not None else 0

    # --- Subsample indices ---
    P_1_idx, P_test_idx = split_subsample_indices(
        n_samples, subsample_ratio=subsample_ratio, random_state=random_state0 + seed
    )

    P_2_idx = None
    if policy.run_stability:
        P_2_idx, _ = split_subsample_indices(
            n_samples,
            subsample_ratio=subsample_ratio,
            random_state=random_state0 + seed + n_resamples,
        )

    # --- Preprocessing ---
    (
        pipeline,
        normalization_params,
        dim_reduction_params,
        normalization_name,
        dim_reduction_name,
    ) = build_preprocessing_pipeline(
        randomize_preprocessing=randomize_preprocessing,
        normalization_options=normalization_options,
        dim_reduction_options=dim_reduction_options,
        seed=random_state0 + seed,
    )

    X_preprocessed = pipeline.fit_transform(X)

    X_1 = X_preprocessed[P_1_idx]
    X_test = X_preprocessed[P_test_idx] if policy.run_generalizability else None
    X_2 = X_preprocessed[P_2_idx] if policy.run_stability else None

    # --- Clustering ---
```

with:

```python
        Threads for this resample's classifier, its share of the run's core
        budget.
    embeddings : ResampleEmbeddings or None, default=None
        This resample's precomputed embeddings under randomized
        preprocessing. None clusters the raw subsamples. The classifier
        trains on raw features either way.

    Returns
    -------
    result : ResampleResult
        Named tuple containing ARI metrics, labels, sample indices, and
        the pipeline this resample used.

    Raises
    ------
    RuntimeError
        If ``embeddings`` does not match this resample's subsample sizes.
    """
    policy = resolve_mode(mode)

    random_state0 = random_state if random_state is not None else 0

    # --- Subsample indices ---
    P_1_idx, P_test_idx, P_2_idx = _resample_indices(
        X.shape[0],
        seed=seed,
        n_resamples=n_resamples,
        subsample_ratio=subsample_ratio,
        random_state=random_state0,
        run_stability=policy.run_stability,
    )

    # --- Features to cluster ---
    # Raw slices without randomization; with it, the resample's precomputed
    # embeddings, one independent fit per subsample. After clustering the
    # embeddings are not used again, so the noise policy's index shrinkage
    # below needs no re-slicing of them.
    if embeddings is None:
        spec = None
        X_1 = X[P_1_idx]
        X_test = X[P_test_idx] if policy.run_generalizability else None
        X_2 = X[P_2_idx] if policy.run_stability else None
    else:
        _check_embeddings(
            policy,
            embeddings,
            P_1_idx=P_1_idx,
            P_2_idx=P_2_idx,
            P_test_idx=P_test_idx,
        )
        spec = embeddings.spec
        X_1, X_2, X_test = embeddings.X_1, embeddings.X_2, embeddings.X_test

    # --- Clustering ---
```

In `src/carve/_runner.py`, replace:

```python
    # --- Resolve noise labels (density-based methods emit -1) ---
    #
    # Under "drop" policy, index arrays shrink:
    # feature matrices must be re-sliced to stay aligned with labels.
    P_1_idx, labels_1, noise_fraction = apply_noise_policy(
        P_1_idx, labels_1, noise_policy
    )
    X_1 = X_preprocessed[P_1_idx]

    if policy.run_generalizability:
        P_test_idx, labels_test, _ = apply_noise_policy(
            P_test_idx, labels_test, noise_policy
        )
        X_test = X_preprocessed[P_test_idx]

    if policy.run_stability:
        P_2_idx, labels_2, _ = apply_noise_policy(P_2_idx, labels_2, noise_policy)
        X_2 = X_preprocessed[P_2_idx]

    # --- Cluster count bookkeeping ---
```

with:

```python
    # --- Resolve noise labels (density-based methods emit -1) ---
    #
    # Under "drop" policy, index arrays shrink. The classifier below reads
    # raw features by these global indices, so nothing needs re-slicing.
    P_1_idx, labels_1, noise_fraction = apply_noise_policy(
        P_1_idx, labels_1, noise_policy
    )

    if policy.run_generalizability:
        P_test_idx, labels_test, _ = apply_noise_policy(
            P_test_idx, labels_test, noise_policy
        )

    if policy.run_stability:
        P_2_idx, labels_2, _ = apply_noise_policy(P_2_idx, labels_2, noise_policy)

    # --- Cluster count bookkeeping ---
```

In `src/carve/_runner.py`, replace:

```python

    # --- Generalizability ARI (RF prediction on held-out set) ---
    labels_pred, ari_pred = _compute_generalizability_ari(
        policy=policy,
        X_1=X_1,
        X_test=X_test,
        labels_1=labels_1,
        labels_test=labels_test,
```

with:

```python

    # --- Generalizability ARI (RF prediction on held-out set) ---
    # The classifier trains on raw features, not on the embedding the labels
    # came from: a cluster generalizes only if it can be learned from the
    # data, and a transformer without transform (t-SNE) could not embed
    # unseen rows anyway. Under identity preprocessing these equal X_1, X_test.
    labels_pred, ari_pred = _compute_generalizability_ari(
        policy=policy,
        X_1=X[P_1_idx],
        X_test=X[P_test_idx] if policy.run_generalizability else None,
        labels_1=labels_1,
        labels_test=labels_test,
```

In `src/carve/_runner.py`, replace:

```python
        test_indices=P_test_idx,
        stability_indices=P_2_idx,
        normalization_params=normalization_params,
        dim_reduction_params=dim_reduction_params,
        normalization_name=normalization_name,
        dim_reduction_name=dim_reduction_name,
        n_clusters_train=k_1,
        n_clusters_test=k_test,
        n_clusters_stability=k_2,
        noise_fraction=noise_fraction,
    )


```

with:

```python
        test_indices=P_test_idx,
        stability_indices=P_2_idx,
        pipeline=spec,
        n_clusters_train=k_1,
        n_clusters_test=k_test,
        n_clusters_stability=k_2,
        noise_fraction=noise_fraction,
    )


def _check_embeddings(
    policy,
    embeddings: ResampleEmbeddings,
    *,
    P_1_idx: np.ndarray,
    P_2_idx: np.ndarray | None,
    P_test_idx: np.ndarray,
) -> None:
    """Raise if precomputed embeddings do not match this resample's subsamples.

    The precompute pass and ``validation_iter`` derive the same indices from
    ``_resample_indices``; a row-count mismatch means that invariant broke.
    """
    expected = [("X_1", embeddings.X_1, P_1_idx)]
    if policy.run_stability:
        expected.append(("X_2", embeddings.X_2, P_2_idx))
    if policy.run_generalizability:
        expected.append(("X_test", embeddings.X_test, P_test_idx))

    for name, embedded, idx in expected:
        n_rows = None if embedded is None else embedded.shape[0]
        if n_rows != idx.size:
            raise RuntimeError(
                f"Precomputed embedding {name} has {n_rows} rows but this "
                f"resample's subsample has {idx.size}. This is an internal "
                "CARVE error."
            )


```

In `src/carve/_runner.py`, replace:

```python
        ``policy.run_generalizability`` is False.
    X_1 : ndarray
        Training features.
    X_test : ndarray or None
        Held-out test features.
    labels_1 : ndarray
        Cluster labels for the training set.
```

with:

```python
        ``policy.run_generalizability`` is False.
    X_1 : ndarray
        Raw training features, ``X[P_1]`` after the noise policy.
    X_test : ndarray or None
        Raw held-out features, ``X[P_test]`` after the noise policy.
    labels_1 : ndarray
        Cluster labels for the training set.
```

In `src/carve/_types.py`, replace:

```python
EstimatorRecord = dict[str, Any]

# Per-resample preprocessing record
PipelineRecord = dict[str, Any]

```

with:

```python
EstimatorRecord = dict[str, Any]

# Per-configuration record of a randomized-preprocessing run
PipelineRecord = dict[str, Any]

```

In `src/carve/_utils.py`, replace:

```python
    pipeline_records: list[dict[str, Any]],
    sweep_param: str = "n_clusters",
) -> pd.DataFrame:
    """Summarize randomized preprocessing records.

    Parameters
    ----------
    pipeline_records : list of dict
        Records from randomized preprocessing runs.
    sweep_param : str, default="n_clusters"
        Name of the swept hyperparameter used as the grouping key.

    Returns
    -------
    summary : pandas.DataFrame
        Mean ARI metrics grouped by normalization, DR, and the sweep value.
    """
    rows = []
    for record in pipeline_records:
        params = record["params"]
        sweep_value = params[sweep_param]

        for r in record["results"]:
            ari_s = r.ari_stability
            ari_g = r.ari_generalizability
            norm_p = r.normalization_params
            dr_p = r.dim_reduction_params
            norm_name = r.normalization_name
            dr_name = r.dim_reduction_name

            if norm_name != "FunctionTransformer":
                norm_label = norm_name
            else:
                func = norm_p.get("func", None)
                norm_label = func.__name__ if func is not None else "identity"

            if dr_name != "FunctionTransformer":
                dr_label = dr_name
            else:
                func = dr_p.get("func", None)
                dr_label = func.__name__ if func is not None else "identity"

            rows.append(
                {
                    sweep_param: sweep_value,
                    "norm__func": norm_label,
                    "dr__method": dr_label,
                    "ari_stability": ari_s,
                    "ari_generalizability": ari_g,
                }
            )

    dfp = pd.DataFrame(rows)

    return dfp.groupby(["norm__func", "dr__method", sweep_param], as_index=False).mean()


```

with:

```python
    pipeline_records: list[dict[str, Any]],
    sweep_param: str = "n_clusters",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize a randomized run per configuration, pipeline and sweep value.

    Parameters
    ----------
    pipeline_records : list of dict
        One record per configuration, as ``_runner.run_validation`` returns
        them: ``method_id``, ``method_label``, ``sweep_value``,
        ``sweep_rank`` and ``results``, the configuration's per-resample
        results, each carrying the ``pipeline`` it used.
    sweep_param : str, default="n_clusters"
        Name of the swept hyperparameter, used to name the sweep column.

    Returns
    -------
    summary : pandas.DataFrame
        One row per (configuration, pipeline, sweep value), sorted by
        ``method_id``, ``pipeline`` and sweep value. Means and standard
        errors are over the resamples that used the row's pipeline at the
        row's configuration only, and ``n_resamples`` counts them.
    pipelines : dict
        The pipeline specs, keyed by the ``pipeline`` column.
    """
    columns = [
        "method_id",
        "method_label",
        "pipeline",
        "normalization",
        "dim_reduction",
        sweep_param,
        "n_resamples",
        "ari_stability",
        "ari_stability_se",
        "ari_generalizability",
        "ari_generalizability_se",
        "n_clusters_observed",
        "sweep_param",
        "sweep_value",
        "sweep_rank",
    ]
    rows: list[dict[str, Any]] = []
    pipelines: dict[str, Any] = {}

    for record in pipeline_records:
        # Group the configuration's resamples by pipeline. The spec is
        # duck-typed on .label: _utils is a leaf and cannot import _pipeline.
        by_pipeline: dict[str, list[Any]] = {}
        for result in record["results"]:
            label = result.pipeline.label
            pipelines.setdefault(label, result.pipeline)
            by_pipeline.setdefault(label, []).append(result)

        for label, runs in by_pipeline.items():
            spec = pipelines[label]
            stab_mean, stab_se, _, _ = _summarize_ari_scores(
                [r.ari_stability for r in runs], len(runs)
            )
            gen_mean, gen_se, _, _ = _summarize_ari_scores(
                [r.ari_generalizability for r in runs], len(runs)
            )
            rows.append(
                {
                    "method_id": record["method_id"],
                    "method_label": record["method_label"],
                    "pipeline": label,
                    "normalization": spec.normalization.label,
                    "dim_reduction": spec.dim_reduction.label,
                    sweep_param: record["sweep_value"],
                    "n_resamples": len(runs),
                    "ari_stability": stab_mean,
                    "ari_stability_se": stab_se,
                    "ari_generalizability": gen_mean,
                    "ari_generalizability_se": gen_se,
                    "n_clusters_observed": float(
                        np.mean([r.n_clusters_train for r in runs])
                    ),
                    "sweep_param": sweep_param,
                    "sweep_value": record["sweep_value"],
                    "sweep_rank": record["sweep_rank"],
                }
            )

    summary = pd.DataFrame(rows, columns=columns)
    if summary.empty:
        return summary, pipelines

    # method_id is "m<n>"; sort on n so that m10 follows m9.
    summary = summary.sort_values(
        ["method_id", "pipeline", "sweep_value"],
        key=lambda col: col.str[1:].astype(int) if col.name == "method_id" else col,
        kind="stable",
    ).reset_index(drop=True)
    return summary, pipelines


```

In `src/carve/api.py`, replace:

```python
)
from ._output import _print_run_footer, _print_run_header
from ._plotting import (
    _get_annotation,
```

with:

```python
)
from ._output import _print_run_footer, _print_run_header
from ._pipeline import PipelineSpec
from ._plotting import (
    _get_annotation,
```

In `src/carve/api.py`, replace:

```python
        Resolved estimator grids used during fitting.
    preprocessing_results_ : pandas.DataFrame or None
        Preprocessing summary when ``randomize_preprocessing=True``.
    sweep_ : SweepSpec or None
        The resolved sweep axis used during fitting.
```

with:

```python
        Resolved estimator grids used during fitting.
    preprocessing_results_ : pandas.DataFrame or None
        Per-pipeline metrics when ``randomize_preprocessing=True``, else None.
        One row per (estimator configuration, pipeline, sweep value); joins to
        ``estimator_results_`` on ``method_id`` and the sweep column. Columns:
        ``method_id`` and ``method_label``; ``pipeline``, the key into
        ``preprocessing_pipelines_``; ``normalization`` and ``dim_reduction``,
        the step labels with their hyperparameters; the sweep column
        (``n_clusters``, ``resolution``, ...); ``n_resamples``, the resamples
        this pipeline received at this configuration; ``ari_stability``,
        ``ari_generalizability`` and their ``_se`` columns, over those
        resamples only; ``n_clusters_observed``; and the ``sweep_param``,
        ``sweep_value`` and ``sweep_rank`` bookkeeping columns. The criterion
        a non-default ``mode`` skips is NaN.
    preprocessing_pipelines_ : dict of str to PipelineSpec, or None
        The pipelines behind ``preprocessing_results_``, keyed by its
        ``pipeline`` column, else None. ``carve._pipeline.pipeline_from_spec
        (spec, random_state)`` rebuilds one as a scikit-learn ``Pipeline``,
        for example to embed the full data with the best-rated pipeline.
    sweep_ : SweepSpec or None
        The resolved sweep axis used during fitting.
```

In `src/carve/api.py`, replace:

```python
    estimator_param_grids_: list[GridSpec] | None = field(init=False, default=None)
    preprocessing_results_: pd.DataFrame | None = field(init=False, default=None)
    sweep_: SweepSpec | None = field(init=False, default=None)

```

with:

```python
    estimator_param_grids_: list[GridSpec] | None = field(init=False, default=None)
    preprocessing_results_: pd.DataFrame | None = field(init=False, default=None)
    preprocessing_pipelines_: dict[str, PipelineSpec] | None = field(
        init=False, default=None
    )
    sweep_: SweepSpec | None = field(init=False, default=None)

```

In `src/carve/api.py`, replace:

```python
        # --- Resolve preprocessing options ---
        # The default option lists are only consumed when a random pipeline is
        # sampled per resample (see _pipeline.build_preprocessing_pipeline), so
        # resolving them otherwise would import UMAP for nothing.
        norm_options = self.normalization_options or default_normalization_options()
```

with:

```python
        # --- Resolve preprocessing options ---
        # The default option lists are only consumed when a random pipeline is
        # allocated per resample (see _runner.run_validation), so
        # resolving them otherwise would import UMAP for nothing.
        norm_options = self.normalization_options or default_normalization_options()
```

In `src/carve/api.py`, replace:

```python
        self.estimator_results_ = pd.DataFrame.from_records(estimator_records)

        self.preprocessing_results_ = (
            None
            if not randomize_preprocessing
            else summarize_preprocessing_records(
                pipeline_records, sweep_param=sweep_spec.param
            )
        )

        n_rows = int(self.estimator_results_.shape[0])
```

with:

```python
        self.estimator_results_ = pd.DataFrame.from_records(estimator_records)

        if randomize_preprocessing:
            self.preprocessing_results_, self.preprocessing_pipelines_ = (
                summarize_preprocessing_records(
                    pipeline_records, sweep_param=sweep_spec.param
                )
            )
        else:
            self.preprocessing_results_ = None
            self.preprocessing_pipelines_ = None

        n_rows = int(self.estimator_results_.shape[0])
```

- [ ] Step 4: Run the tests to verify they pass

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
.venv/bin/pytest tests/test_pipeline.py tests/test_runner.py tests/test_utils.py tests/test_api.py -q
```

Expected: ruff reports `All checks passed!`; pytest reports `360 passed`. The regression gate in `tests/test_api.py` is part of this run and must pass: the non-randomized path now clusters raw slices directly, and the gate proves the numbers did not move.

- [ ] Step 5: Commit

```bash
git add src/carve/_pipeline.py src/carve/_runner.py src/carve/_types.py src/carve/_utils.py src/carve/api.py tests/_helpers.py tests/test_api.py tests/test_pipeline.py tests/test_runner.py tests/test_utils.py
git commit -m "feat(runner): fit the sampled pipeline per subsample, once per resample; classifier on raw features; per-pipeline results table"
```

---

### Task 4: Discrete default grids, the log1p guard, and defaults resolved only when randomized

Spec 5.3 and the verification item "Defaults". `default_normalization_options` now takes X; `fit` resolves either default only for a randomized fit, since resolving the normalization default would otherwise warn on every fit whose input has negative values, and `filterwarnings = error` would turn that into failures across the suite.

Files:
- Modify: `src/carve/_grids.py`
- Modify: `src/carve/api.py`
- Test: `tests/test_api.py`
- Test: `tests/test_grids.py`

Interfaces:
- Consumes: the option resolution block in `CARVE.fit` as Task 3 left it.
- Produces:
  - `default_normalization_options(X: np.ndarray) -> list[PreprocSpec]`, warning `UserWarning("X has negative values (minimum -2.55), so log1p is omitted from the default normalization options.")` when `X.min() < 0`.
  - `default_dim_reduction_options(X, subsample_ratio=0.6) -> list[PreprocSpec]` with discrete grids from `_PCA_COMPONENTS = (2, 5, 10, 20, 50)`, `_TSNE_PERPLEXITIES = (15, 30, 50)`, `_UMAP_NEIGHBORS = (15, 30)`, filtered below `n_min = min(n_train, n_samples - n_train)` (PCA below `min(n_min, p)`), each empty grid dropped with `UserWarning("TSNE is omitted from the default dimensionality reduction options: no candidate perplexity is below 12, the limit the smallest subsample sets.")`.
  - In `CARVE.fit`, `norm_options` and `dr_options` hold the resolved lists (empty lists when not randomized); Task 5 passes them to the run header.

- [ ] Step 1: Write the failing tests

The `no_umap` fixture makes the grids independent of whether the `[umap]` extra is installed; tests about UMAP itself use `pytest.importorskip("umap")`.

In `tests/test_api.py`, replace:

```python


class TestShowProgress:
    def test_progress_bar_and_per_config_lines(self, X_two_clusters, capsys):
```

with:

```python


class TestDefaultPreprocessingOptions:
    """fit resolves the default option lists only when it will use them."""

    def _model(self, **kwargs):
        return CARVE(
            n_clusters=np.array([2]),
            n_resamples=4,
            subsample_ratio=0.7,
            estimator_param_grids=[(KMeans, {"n_clusters": [2], "n_init": [3]})],
            random_state=0,
            **kwargs,
        )

    def test_not_resolved_without_randomization(self, X_two_clusters, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("defaults resolved for a non-randomized fit")

        monkeypatch.setattr(carve_api, "default_normalization_options", fail)
        monkeypatch.setattr(carve_api, "default_dim_reduction_options", fail)
        self._model().fit(X_two_clusters)

    def test_randomized_fit_resolves_both_from_X(self, X_two_clusters, monkeypatch):
        seen = {}

        def norm(X):
            seen["norm"] = X.shape
            return [(FunctionTransformer, {})]

        def dr(X, subsample_ratio):
            seen["dr"] = (X.shape, subsample_ratio)
            return [(FunctionTransformer, {})]

        monkeypatch.setattr(carve_api, "default_normalization_options", norm)
        monkeypatch.setattr(carve_api, "default_dim_reduction_options", dr)
        model = self._model().fit(X_two_clusters, randomize_preprocessing=True)
        assert seen == {"norm": (60, 5), "dr": ((60, 5), 0.7)}
        assert set(model.preprocessing_pipelines_) == {"identity | identity"}

    def test_negative_input_drops_log1p(self, X_two_clusters):
        model = self._model(dim_reduction_options=[(FunctionTransformer, {})])
        with pytest.warns(UserWarning, match="so log1p is omitted"):
            model.fit(X_two_clusters, randomize_preprocessing=True)
        assert set(model.preprocessing_pipelines_) == {
            "identity | identity",
            "StandardScaler | identity",
        }


class TestShowProgress:
    def test_progress_bar_and_per_config_lines(self, X_two_clusters, capsys):
```

In `tests/test_grids.py`, replace:

```python
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import FunctionTransformer, StandardScaler
```

with:

```python
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import FunctionTransformer, StandardScaler
```

In `tests/test_grids.py`, replace:

```python

class TestDefaultNormalizationOptions:
    def test_structure(self):
        options = default_normalization_options()
        assert isinstance(options, list)
        assert len(options) == 3  # identity, StandardScaler, log1p

    def test_contains_identity(self):
        options = default_normalization_options()
        has_identity = any(
            cls is FunctionTransformer and params == {} for cls, params in options
        )
        assert has_identity

    def test_contains_standard_scaler(self):
        options = default_normalization_options()
        has_scaler = any(cls is StandardScaler for cls, params in options)
        assert has_scaler

    def test_contains_log1p(self):
        options = default_normalization_options()
        has_log = any(
            cls is FunctionTransformer and params.get("func") == [np.log1p]
            for cls, params in options
        )
        assert has_log


# -----------------------------------------------------------------------
# default_dim_reduction_options
# -----------------------------------------------------------------------


```

with:

```python

class TestDefaultNormalizationOptions:
    def test_nonnegative_input_offers_log1p(self):
        X = np.abs(np.random.RandomState(0).randn(20, 3))
        assert default_normalization_options(X) == [
            (FunctionTransformer, {}),
            (StandardScaler, {}),
            (FunctionTransformer, {"func": [np.log1p]}),
        ]

    def test_zero_minimum_still_offers_log1p(self):
        X = np.zeros((5, 2))
        X[0, 0] = 3.0
        options = default_normalization_options(X)
        assert (FunctionTransformer, {"func": [np.log1p]}) in options

    def test_negative_input_omits_log1p_with_a_warning(self):
        X = np.random.RandomState(0).randn(20, 3)
        with pytest.warns(
            UserWarning, match=r"negative values \(minimum -[0-9.]+\), so log1p is omitted"
        ):
            options = default_normalization_options(X)
        assert options == [(FunctionTransformer, {}), (StandardScaler, {})]


# -----------------------------------------------------------------------
# default_dim_reduction_options
# -----------------------------------------------------------------------


@pytest.fixture()
def no_umap(monkeypatch):
    """Make umap-learn look absent, so the grids do not depend on the extra."""
    monkeypatch.setattr("carve._grids.importlib.util.find_spec", lambda name: None)


```

In `tests/test_grids.py`, replace:

```python
        assert any(cls is UMAP for cls, _ in options)

    def test_warns_when_umap_missing(self, monkeypatch):
        monkeypatch.setattr(
            "carve._grids.importlib.util.find_spec", lambda name: None
        )
        X = np.random.RandomState(0).randn(100, 10)
        with pytest.warns(UserWarning, match="umap-learn is not installed"):
            options = default_dim_reduction_options(X)
        assert len(options) == 3

    def test_contains_identity(self):
```

with:

```python
        assert any(cls is UMAP for cls, _ in options)

    def test_warns_when_umap_missing(self, no_umap):
        X = np.random.RandomState(0).randn(100, 10)
        with pytest.warns(UserWarning, match="umap-learn is not installed"):
            options = default_dim_reduction_options(X)
        assert len(options) == 3

    def test_grids_are_discrete_and_filtered(self, no_umap):
        # n=60 at ratio 0.618: 37 training rows and 23 held out, so n_min=23
        # and the PCA limit is min(23, p=10) = 10.
        X = np.random.RandomState(0).randn(60, 10)
        with pytest.warns(UserWarning, match="umap-learn is not installed"):
            options = default_dim_reduction_options(X, subsample_ratio=0.618)
        assert options == [
            (FunctionTransformer, {}),
            (PCA, {"n_components": [2, 5]}),
            (TSNE, {"n_components": [2], "perplexity": [15]}),
        ]

    def test_wide_margin_keeps_every_candidate(self, no_umap):
        # n=400 at ratio 0.618: n_min=153, p=60.
        X = np.random.RandomState(0).randn(400, 60)
        with pytest.warns(UserWarning, match="umap-learn is not installed"):
            grids = dict(default_dim_reduction_options(X, subsample_ratio=0.618))
        assert grids[PCA] == {"n_components": [2, 5, 10, 20, 50]}
        assert grids[TSNE] == {"n_components": [2], "perplexity": [15, 30, 50]}

    def test_limit_is_the_smaller_subsample_at_low_ratios(self, no_umap):
        # Ratio 0.3 on n=100: 30 training rows and 70 held out, so the limit
        # is 30, not the held-out 70.
        X = np.random.RandomState(0).randn(100, 60)
        with pytest.warns(UserWarning, match="umap-learn is not installed"):
            grids = dict(default_dim_reduction_options(X, subsample_ratio=0.3))
        assert grids[PCA] == {"n_components": [2, 5, 10, 20]}
        assert grids[TSNE]["perplexity"] == [15]

    def test_umap_grid_is_filtered(self):
        pytest.importorskip("umap")
        from umap import UMAP

        X = np.random.RandomState(0).randn(60, 10)
        grids = dict(default_dim_reduction_options(X, subsample_ratio=0.618))
        assert grids[UMAP] == {
            "n_components": [2],
            "n_neighbors": [15],
            "min_dist": [0.1],
        }

    def test_tsne_with_an_empty_grid_is_omitted_with_a_warning(self, no_umap):
        # n=30 at ratio 0.618: 18 training rows and 12 held out; no candidate
        # perplexity is below 12.
        X = np.random.RandomState(0).randn(30, 10)
        with pytest.warns(UserWarning) as record:
            options = default_dim_reduction_options(X, subsample_ratio=0.618)
        messages = [str(w.message) for w in record]
        assert any(
            "TSNE is omitted" in m and "no candidate perplexity is below 12" in m
            for m in messages
        )
        assert [cls for cls, _ in options] == [FunctionTransformer, PCA]

    def test_pca_with_an_empty_grid_is_omitted_with_a_warning(self, no_umap):
        X = np.random.RandomState(0).randn(100, 2)
        with pytest.warns(UserWarning) as record:
            options = default_dim_reduction_options(X, subsample_ratio=0.618)
        assert any("PCA is omitted" in str(w.message) for w in record)
        assert [cls for cls, _ in options] == [FunctionTransformer, TSNE]

    def test_umap_with_an_empty_grid_is_omitted_with_a_warning(self):
        pytest.importorskip("umap")
        X = np.random.RandomState(0).randn(30, 10)
        with pytest.warns(UserWarning) as record:
            options = default_dim_reduction_options(X, subsample_ratio=0.618)
        assert any("UMAP is omitted" in str(w.message) for w in record)
        assert [cls for cls, _ in options] == [FunctionTransformer, PCA]

    def test_contains_identity(self):
```

In `tests/test_grids.py`, replace:

```python
        assert has_identity

    def test_contains_pca(self):
        X = np.random.RandomState(0).randn(100, 10)
        options = default_dim_reduction_options(X)
        has_pca = any(cls is PCA for cls, params in options)
        assert has_pca

    def test_pca_components_respect_data(self):
        X = np.random.RandomState(0).randn(50, 5)
        options = default_dim_reduction_options(X, subsample_ratio=0.6)
        pca_option = next((cls, params) for cls, params in options if cls is PCA)
        n_components = pca_option[1]["n_components"]
        # Should respect min(min_n, p) where min_n = round(50 * 0.4) - 1
        assert all(c >= 2 for c in n_components)
        assert max(n_components) < 5  # p = 5
```

with:

```python
        assert has_identity

    def test_pca_components_respect_data(self):
        X = np.random.RandomState(0).randn(50, 5)
        options = default_dim_reduction_options(X, subsample_ratio=0.6)
        pca_option = next((cls, params) for cls, params in options if cls is PCA)
        n_components = pca_option[1]["n_components"]
        assert all(c >= 2 for c in n_components)
        assert max(n_components) < 5  # p = 5
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/test_grids.py tests/test_api.py::TestDefaultPreprocessingOptions -q`

Expected: `13 failed, 29 passed`. The first distinct errors:

- `TypeError: default_normalization_options() takes 0 positional arguments but 1 was given`
- `Failed: DID NOT WARN. No warnings of type (<class 'UserWarning'>,) were emitted.`
- `AssertionError: assert [(<class 'skl...9, 10, ...]})] == [(<class 'skl...xity': [15]})]`

- [ ] Step 3: Implement

In `src/carve/_grids.py`, replace:

```python


def default_normalization_options() -> list[PreprocSpec]:
    """Return default normalization preprocessing options.

    Returns
```

with:

```python


#: Candidate values for the default dimensionality reduction grids, before
#: filtering to what the smallest subsample supports.
_PCA_COMPONENTS = (2, 5, 10, 20, 50)
_TSNE_PERPLEXITIES = (15, 30, 50)
_UMAP_NEIGHBORS = (15, 30)


def default_normalization_options(X: np.ndarray) -> list[PreprocSpec]:
    """Return default normalization preprocessing options for X.

    Identity and standardization are always offered. ``log1p`` is offered
    only when X has no negative values: it is NaN below -1, and an input
    with negatives (an LSI, a z-scored matrix) is not the count-like data it
    is meant for. The omission is announced with a warning.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.

    Returns
```

In `src/carve/_grids.py`, replace:

```python
        List of (TransformerClass, param_grid) pairs.
    """
    return [
        (FunctionTransformer, {}),
        (StandardScaler, {}),
        (FunctionTransformer, {"func": [np.log1p]}),
    ]


```

with:

```python
        List of (TransformerClass, param_grid) pairs.
    """
    options: list[PreprocSpec] = [
        (FunctionTransformer, {}),
        (StandardScaler, {}),
    ]

    x_min = float(np.min(X))
    if x_min >= 0:
        options.append((FunctionTransformer, {"func": [np.log1p]}))
    else:
        warnings.warn(
            f"X has negative values (minimum {x_min:.3g}), so log1p is omitted "
            "from the default normalization options.",
            UserWarning,
            stacklevel=2,
        )

    return options


def _warn_omitted(name: str, param: str, limit: int) -> None:
    """Warn that a default option was dropped because its grid filtered empty."""
    warnings.warn(
        f"{name} is omitted from the default dimensionality reduction options: "
        f"no candidate {param} is below {limit}, the limit the smallest "
        "subsample sets.",
        UserWarning,
        stacklevel=3,
    )


```

In `src/carve/_grids.py`, replace:

```python
) -> list[PreprocSpec]:
    """Return default dimensionality reduction options.

    Parameters
```

with:

```python
) -> list[PreprocSpec]:
    """Return default dimensionality reduction options.

    Discrete grids, each filtered to what the smallest subsample a pipeline
    is fitted on can support:

    - identity
    - PCA, ``n_components`` in {2, 5, 10, 20, 50}, below ``min(n_min, p)``
    - t-SNE, two components, ``perplexity`` in {15, 30, 50}, below ``n_min``
    - UMAP when umap-learn is installed, two components, ``n_neighbors`` in
      {15, 30} below ``n_min``, ``min_dist`` 0.1

    ``n_min`` is the size of the smaller of a resample's training subsample
    and held-out set, which is the held-out set at any ratio above 0.5. An
    option whose filtered grid is empty is omitted with a warning.

    Parameters
```

In `src/carve/_grids.py`, replace:

```python
    """
    n_samples, p = X.shape
    min_n = int(round(n_samples * (1 - subsample_ratio))) - 1

    options: list[PreprocSpec] = [
        (FunctionTransformer, {}),
        (PCA, {"n_components": list(range(2, min(min_n, p)))}),
        (TSNE, {"n_components": [2], "perplexity": list(range(5, min(min_n, 51)))}),
    ]

    if importlib.util.find_spec("umap") is not None:
        from umap import UMAP

        options.append(
            (
                UMAP,
                {
                    "n_components": list(range(2, min(min_n, p))),
                    "n_neighbors": list(range(5, 51)),
                    "min_dist": [0.1],
                },
            )
        )
    else:
        warnings.warn(
            "umap-learn is not installed; UMAP is omitted from the default "
```

with:

```python
    """
    n_samples, p = X.shape
    # Same arithmetic split_subsample_indices uses for the training size.
    n_train = int(np.float64(subsample_ratio * n_samples))
    n_min = min(n_train, n_samples - n_train)

    options: list[PreprocSpec] = [(FunctionTransformer, {})]

    pca_limit = min(n_min, p)
    components = [c for c in _PCA_COMPONENTS if c < pca_limit]
    if components:
        options.append((PCA, {"n_components": components}))
    else:
        _warn_omitted("PCA", "n_components", pca_limit)

    perplexities = [x for x in _TSNE_PERPLEXITIES if x < n_min]
    if perplexities:
        options.append((TSNE, {"n_components": [2], "perplexity": perplexities}))
    else:
        _warn_omitted("TSNE", "perplexity", n_min)

    if importlib.util.find_spec("umap") is None:
        warnings.warn(
            "umap-learn is not installed; UMAP is omitted from the default "
```

In `src/carve/_grids.py`, replace:

```python
            '`pip install "carve-validate[umap]"`.',
            UserWarning,
            stacklevel=2,
        )

    return options
```

with:

```python
            '`pip install "carve-validate[umap]"`.',
            UserWarning,
            stacklevel=2,
        )
        return options

    neighbors = [x for x in _UMAP_NEIGHBORS if x < n_min]
    if not neighbors:
        _warn_omitted("UMAP", "n_neighbors", n_min)
        return options

    from umap import UMAP

    options.append(
        (
            UMAP,
            {"n_components": [2], "n_neighbors": neighbors, "min_dist": [0.1]},
        )
    )
    return options
```

In `src/carve/api.py`, replace:

```python
        of (EstimatorClass, param_grid) tuples may also be passed.
    normalization_options : list of preprocessing specs, optional
        Normalization preprocessing options. If None, defaults include
        identity, StandardScaler, and log1p.
    dim_reduction_options : list of dimensionality reduction specs, optional
        Dimensionality reduction preprocessing options. If None, defaults
        include identity, PCA, t-SNE, and UMAP.
    classifier : sklearn classifier instance, optional
        Classifier used to score generalizability. If None (default), a
```

with:

```python
        of (EstimatorClass, param_grid) tuples may also be passed.
    normalization_options : list of preprocessing specs, optional
        Normalization options for ``fit(randomize_preprocessing=True)``. If
        None, a randomized fit uses identity, StandardScaler, and log1p when
        X has no negative values (``_grids.default_normalization_options``).
    dim_reduction_options : list of dimensionality reduction specs, optional
        Dimensionality reduction options for a randomized fit. If None, it
        uses identity, PCA, t-SNE, and UMAP when installed, over discrete
        grids filtered to the subsample sizes
        (``_grids.default_dim_reduction_options``).
    classifier : sklearn classifier instance, optional
        Classifier used to score generalizability. If None (default), a
```

In `src/carve/api.py`, replace:

```python

        # --- Resolve preprocessing options ---
        # The default option lists are only consumed when a random pipeline is
        # allocated per resample (see _runner.run_validation), so
        # resolving them otherwise would import UMAP for nothing.
        norm_options = self.normalization_options or default_normalization_options()
        if self.dim_reduction_options is not None:
            dr_options = self.dim_reduction_options
        elif randomize_preprocessing:
            dr_options = default_dim_reduction_options(X, self.subsample_ratio)
        else:
            dr_options = []

        # --- Print run header ---
```

with:

```python

        # --- Resolve preprocessing options ---
        # Only a randomized fit consumes them. Resolving the defaults otherwise
        # would import UMAP for nothing and warn about log1p on any input with
        # negative values.
        if randomize_preprocessing:
            norm_options = self.normalization_options or default_normalization_options(
                X
            )
            if self.dim_reduction_options is not None:
                dr_options = self.dim_reduction_options
            else:
                dr_options = default_dim_reduction_options(X, self.subsample_ratio)
        else:
            norm_options = self.normalization_options or []
            dr_options = self.dim_reduction_options or []

        # --- Print run header ---
```

- [ ] Step 4: Run the tests to verify they pass

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
.venv/bin/pytest tests/test_grids.py tests/test_api.py::TestDefaultPreprocessingOptions tests/test_api.py::TestRandomizedPreprocessing -q
```

Expected: ruff reports `All checks passed!`; pytest reports `49 passed`.

- [ ] Step 5: Commit

```bash
git add src/carve/_grids.py src/carve/api.py tests/test_api.py tests/test_grids.py
git commit -m "feat(grids): discrete default preprocessing grids filtered to the smallest subsample; log1p only for nonnegative input"
```

---

### Task 5: Run header names the pipeline options; fit and class documentation

Spec 5.4. The header lists option names under `verbose=2` when randomized. The `randomize_preprocessing` entry of `CARVE.fit` states the allocation, the per-subsample fit, the raw-feature classifier, what that means for a user classifier, and the memory formula; the class Notes record that the submitted S1 Text describes the old fit scope and the old table name.

Files:
- Modify: `src/carve/_output.py`
- Modify: `src/carve/api.py`
- Test: `tests/test_api.py`
- Test: `tests/test_output.py`

Interfaces:
- Consumes: `_pipeline.option_label` (Task 2); `norm_options` and `dr_options` in `CARVE.fit` (Task 4).
- Produces: `_print_run_header(..., verbose, normalization_options: list[PreprocOption] | None = None, dim_reduction_options: list[PreprocOption] | None = None)`, printing `[CARVE] normalization      : identity, log1p` and `[CARVE] dim_reduction      : PCA, tsne` when randomized.

- [ ] Step 1: Write the failing tests

`test_randomized_header_names_the_options` hands both option lists straight to `_print_run_header` and checks the exact lines; `test_verbose_header_names_the_resolved_options` checks that `fit` passes the lists it resolved.

In `tests/test_api.py`, replace:

```python
        assert carve.get_labels().shape == (80,)

    def test_none_without_randomization(self, X_two_clusters):
        carve = self.make().fit(X_two_clusters)
```

with:

```python
        assert carve.get_labels().shape == (80,)

    def test_verbose_header_names_the_resolved_options(self, X_two_clusters, capsys):
        self.make(n_resamples=2, verbose=2).fit(
            X_two_clusters, randomize_preprocessing=True
        )
        out = capsys.readouterr().out
        assert "[CARVE] normalization      : identity, StandardScaler\n" in out
        assert "[CARVE] dim_reduction      : identity, PCA\n" in out

    def test_none_without_randomization(self, X_two_clusters):
        carve = self.make().fit(X_two_clusters)
```

In `tests/test_output.py`, replace:

```python
import pandas as pd
from sklearn.cluster import KMeans

from carve._output import _log_config_progress, _print_run_footer, _print_run_header
```

with:

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from carve._output import _log_config_progress, _print_run_footer, _print_run_header
```

In `tests/test_output.py`, replace:

```python
        assert "resolution" in out
        assert "n_clusters" not in out


```

with:

```python
        assert "resolution" in out
        assert "n_clusters" not in out

    def test_randomized_header_names_the_options(self, capsys):
        _print_run_header(
            X=np.zeros((10, 3)),
            sweep=resolve_sweep(n_clusters=np.array([2, 3])),
            n_resamples=5,
            subsample_ratio=0.8,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_jobs=1,
            randomize_preprocessing=True,
            random_state=0,
            verbose=2,
            normalization_options=[
                (FunctionTransformer, {}),
                (FunctionTransformer, {"func": [np.log1p]}),
            ],
            dim_reduction_options=[
                (PCA, {"n_components": [2]}),
                (TSNE, "tsne", {"perplexity": [30]}),
            ],
        )
        out = capsys.readouterr().out
        assert "[CARVE] normalization      : identity, log1p\n" in out
        assert "[CARVE] dim_reduction      : PCA, tsne\n" in out

    def test_header_lists_no_options_without_randomization(self, capsys):
        _print_run_header(
            X=np.zeros((10, 3)),
            sweep=resolve_sweep(n_clusters=np.array([2, 3])),
            n_resamples=5,
            subsample_ratio=0.8,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_jobs=1,
            randomize_preprocessing=False,
            random_state=0,
            verbose=2,
            normalization_options=[(StandardScaler, {})],
            dim_reduction_options=[(PCA, {"n_components": [2]})],
        )
        out = capsys.readouterr().out
        assert "normalization" not in out
        assert "dim_reduction" not in out


```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/test_output.py tests/test_api.py::TestRandomizedPreprocessing::test_verbose_header_names_the_resolved_options -q`

Expected: `3 failed, 10 passed`. The first distinct errors:

- `TypeError: _print_run_header() got an unexpected keyword argument 'normalization_options'`
- `AssertionError: assert '[CARVE] normalization      : identity, StandardScaler\n' in '[CARVE] ======================================================...`

- [ ] Step 3: Implement

The docstring text is part of the deliverable; keep it verbatim.

In `src/carve/_output.py`, replace:

```python
from tqdm.auto import tqdm

from ._sweep import SweepSpec
from ._types import EstimatorRecord, GridSpec


```

with:

```python
from tqdm.auto import tqdm

from ._pipeline import option_label
from ._sweep import SweepSpec
from ._types import EstimatorRecord, GridSpec, PreprocOption


```

In `src/carve/_output.py`, replace:

```python
    random_state: int | None,
    verbose: int,
) -> None:
    """Print a standard header describing the validation configuration.
```

with:

```python
    random_state: int | None,
    verbose: int,
    normalization_options: list[PreprocOption] | None = None,
    dim_reduction_options: list[PreprocOption] | None = None,
) -> None:
    """Print a standard header describing the validation configuration.
```

In `src/carve/_output.py`, replace:

```python
        Random seed.
    verbose : int
        Verbosity level; prints only if >= 2.
    """
    if verbose < 2:
        return
```

with:

```python
        Random seed.
    verbose : int
        Verbosity level; prints only if >= 2.
    normalization_options : list, optional
        Normalization options of a randomized fit, listed by name.
    dim_reduction_options : list, optional
        Dimensionality reduction options of a randomized fit, listed by name.
    """
    if verbose < 2:
        return
```

In `src/carve/_output.py`, replace:

```python
    print(f"[CARVE] total configs      : {total_configs}")
    print(f"[CARVE] randomize_preproc  : {randomize_preprocessing}")
    print(f"[CARVE] random_state       : {random_state}")
    print(f"[CARVE] {line}")
```

with:

```python
    print(f"[CARVE] total configs      : {total_configs}")
    print(f"[CARVE] randomize_preproc  : {randomize_preprocessing}")
    if randomize_preprocessing:
        norm_names = ", ".join(option_label(o) for o in normalization_options or [])
        dr_names = ", ".join(option_label(o) for o in dim_reduction_options or [])
        print(f"[CARVE] normalization      : {norm_names}")
        print(f"[CARVE] dim_reduction      : {dr_names}")
    print(f"[CARVE] random_state       : {random_state}")
    print(f"[CARVE] {line}")
```

In `src/carve/api.py`, replace:

```python
    and generalizability (held-out prediction via a random forest).

    See Also
    --------
```

with:

```python
    and generalizability (held-out prediction via a random forest).

    Under ``fit(randomize_preprocessing=True)`` each resample's pipeline is
    fit separately on each subsample, and the generalizability classifier
    trains on raw features. The S1 Text submitted with the manuscript
    describes an earlier version that fit the pipeline once on all of X and
    trained the classifier on the embedding, and names the per-pipeline
    table ``pipeline_df_``; the table is ``preprocessing_results_``.

    See Also
    --------
```

In `src/carve/api.py`, replace:

```python
            Overrides the ``reference_labels`` passed at __init__ if given.
        randomize_preprocessing : bool, default=False
            Whether to randomize preprocessing pipelines. When True, a
            random normalization and dimensionality reduction combination
            is sampled independently for each resample iteration.
        show_progress : bool, default=False
            Display a tqdm progress bar over grid configurations.
```

with:

```python
            Overrides the ``reference_labels`` passed at __init__ if given.
        randomize_preprocessing : bool, default=False
            Draw a preprocessing pipeline per resample, so that stability and
            generalizability reflect preprocessing choice as well as
            sampling. Pipelines are allocated evenly: each (normalization,
            dimensionality reduction) option pair is used in floor or ceil
            of ``n_resamples / n_pairs`` resamples, in a seeded random order,
            with hyperparameters drawn per resample. A resample's pipeline is
            fit separately on each of its subsamples, the two clustered
            subsamples and the held-out set, so an embedding that does not
            reproduce across independent fits lowers both criteria. Each
            pipeline is fit once per resample, not once per configuration.

            The generalizability classifier trains on the raw features of
            the first subsample, with the labels clustered from its
            embedding, and predicts the raw held-out features. A cluster
            that exists only in an embedding therefore does not generalize,
            and transformers without a ``transform`` method, such as t-SNE,
            can take part. A user-supplied ``classifier`` sees unnormalized
            features, which matters if it is not invariant to feature
            scaling; the default random forest is.

            Every embedding is held in memory for the duration of the fit,
            about ``8 * n_resamples * (n_1 + n_2 + n_test) * d`` bytes, where
            the n are the subsample sizes and d is the embedded dimension:
            roughly 320 MB at n=5,000 with ``n_resamples=100``,
            ``subsample_ratio=0.618`` and a 50-dimensional identity pipeline,
            negligible for 2-D embeddings. Results are in
            ``preprocessing_results_`` and ``preprocessing_pipelines_``.
        show_progress : bool, default=False
            Display a tqdm progress bar over grid configurations.
```

In `src/carve/api.py`, replace:

```python
            random_state=self._random_state_,
            verbose=self.verbose,
        )

```

with:

```python
            random_state=self._random_state_,
            verbose=self.verbose,
            normalization_options=norm_options,
            dim_reduction_options=dr_options,
        )

```

- [ ] Step 4: Run the tests to verify they pass

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
.venv/bin/pytest tests/test_output.py tests/test_api.py::TestRandomizedPreprocessing -q
```

Expected: ruff reports `All checks passed!`; pytest reports `20 passed`.

- [ ] Step 5: Commit

```bash
git add src/carve/_output.py src/carve/api.py tests/test_api.py tests/test_output.py
git commit -m "feat(output): name the preprocessing options in the run header; document randomized preprocessing"
```

---

### Task 6: Shared metric-line drawing and plot_metric_by_pipeline

Spec 5.5, module level. The body of `plot_metric_over_n_clusters` moves into `_draw_metric_lines`, parameterized on the grouping column, the label function and the legend title, and both plots call it. The moved code is otherwise unchanged; the existing `TestPlotMetricOverNClusters` and the `tests/test_pl.py` comparisons pin that.

Files:
- Modify: `src/carve/_plotting.py`
- Test: `tests/test_plotting.py`

Interfaces:
- Consumes: the `preprocessing_results_` schema from Task 3.
- Produces:
  - `_plotting._draw_metric_lines(results_df, *, group_col, label_of, legend_title, measure, rule, not_two, ax, figsize, title, xlabel, ylabel, legend, legend_loc, palette, show, save, dpi, **kwargs) -> Axes | None`.
  - `_plotting.plot_metric_by_pipeline(preprocessing_df, *, method_id, measure="stability", rule="1se", not_two=False, ax=None, figsize=None, title=None, xlabel=None, ylabel=None, legend=True, legend_loc="best", palette="Accent", show=False, save=None, dpi=300, **kwargs) -> Axes | None`. Raises `RuntimeError("There is no preprocessing table to plot: the fit was not randomized. ...")` for None, `RuntimeError("Preprocessing DataFrame is empty.")`, and `ValueError("method_id 'm9' not found in the preprocessing table. Available: ['m0', 'm1'].")`.

- [ ] Step 1: Write the failing tests

The synthetic table gives the configuration under test (`m0`) values that peak at k=3 and a decoy configuration (`m1`) that peaks higher at k=4, so drawing or selecting over the whole table instead of `m0`'s rows is visibly wrong.

In `tests/test_plotting.py`, replace:

```python

from carve import CARVE
from carve._plotting import (
    _build_estimator_label,
```

with:

```python

from carve import CARVE
import carve._plotting as carve_plotting
from carve._plotting import (
    _build_estimator_label,
```

In `tests/test_plotting.py`, replace:

```python
    plot_consensus_matrix,
    plot_diagnostic_scatter,
    plot_metric_over_n_clusters,
)
```

with:

```python
    plot_consensus_matrix,
    plot_diagnostic_scatter,
    plot_metric_by_pipeline,
    plot_metric_over_n_clusters,
)
```

In `tests/test_plotting.py`, replace:

```python
        assert ax.get_xlabel() == "k"
        assert ax.get_ylabel() == "Score"


```

with:

```python
        assert ax.get_xlabel() == "k"
        assert ax.get_ylabel() == "Score"


# -----------------------------------------------------------------------
# plot_metric_by_pipeline
# -----------------------------------------------------------------------


def _pipeline_results_df():
    """A preprocessing_results_ table over two configurations.

    m0 has two pipelines whose values peak at k=3. m1 has a third pipeline,
    a decoy peaking at k=4 above anything in m0, so a plot or a selection
    that reads the whole table instead of m0's rows is visibly wrong.
    """
    curves = {
        ("m0", "identity | identity"): [0.70, 0.90, 0.60],
        ("m0", "identity | PCA(n_components=2)"): [0.65, 0.80, 0.50],
        ("m1", "StandardScaler | identity"): [0.20, 0.30, 0.99],
    }
    rows = []
    for (method_id, pipeline), values in curves.items():
        normalization, dim_reduction = pipeline.split(" | ")
        for rank, (k, value) in enumerate(zip((2, 3, 4), values)):
            rows.append(
                {
                    "method_id": method_id,
                    "method_label": f"KMeans {method_id}",
                    "pipeline": pipeline,
                    "normalization": normalization,
                    "dim_reduction": dim_reduction,
                    "n_clusters": k,
                    "n_resamples": 5,
                    "ari_stability": value,
                    "ari_stability_se": 0.01,
                    "ari_generalizability": value - 0.1,
                    "ari_generalizability_se": 0.02,
                    "n_clusters_observed": float(k),
                    "sweep_param": "n_clusters",
                    "sweep_value": k,
                    "sweep_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def _curves(ax):
    """Each errorbar's data line, keyed by its legend label."""
    return {container.get_label(): container[0] for container in ax.containers}


class TestPlotMetricByPipeline:
    def test_one_line_per_pipeline_of_the_configuration(self):
        ax = plot_metric_by_pipeline(_pipeline_results_df(), method_id="m0")
        assert set(_curves(ax)) == {
            "identity | identity",
            "identity | PCA(n_components=2)",
        }

    def test_lines_carry_the_rows_values(self):
        ax = plot_metric_by_pipeline(
            _pipeline_results_df(), method_id="m0", measure="generalizability"
        )
        curves = _curves(ax)
        np.testing.assert_allclose(
            curves["identity | identity"].get_ydata(), [0.60, 0.80, 0.50]
        )
        np.testing.assert_allclose(
            curves["identity | PCA(n_components=2)"].get_ydata(), [0.55, 0.70, 0.40]
        )
        np.testing.assert_allclose(curves["identity | identity"].get_xdata(), [2, 3, 4])

    def test_selected_value_comes_from_the_plotted_configuration(self):
        # Over the whole table the best stability is m1 at k=4; within m0, k=3.
        ax = plot_metric_by_pipeline(_pipeline_results_df(), method_id="m0", rule="max")
        dashed = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
        assert len(dashed) == 1
        assert list(dashed[0].get_xdata()) == [3.0, 3.0]

    def test_legend_and_axis_labels(self):
        ax = plot_metric_by_pipeline(_pipeline_results_df(), method_id="m0")
        assert ax.get_legend().get_title().get_text() == "Pipelines"
        assert ax.get_xlabel() == "Number of Clusters (k)"
        assert ax.get_ylabel() == "ARI Stability"

    def test_resolution_axis(self):
        df = _pipeline_results_df().rename(columns={"n_clusters": "resolution"})
        df["sweep_param"] = "resolution"
        df["sweep_value"] = df["sweep_value"] / 4
        ax = plot_metric_by_pipeline(df, method_id="m0")
        assert ax.get_xlabel() == "Resolution"
        np.testing.assert_allclose(ax.get_xticks(), [0.5, 0.75, 1.0])

    def test_colors_come_from_the_palette(self):
        ax = plot_metric_by_pipeline(
            _pipeline_results_df(), method_id="m0", palette="viridis"
        )
        drawn = [tuple(c[0].get_color()) for c in ax.containers]
        expected = [tuple(rgba) for rgba in plt.get_cmap("viridis")(np.linspace(0, 1, 2))]
        assert drawn == expected

    def test_kwargs_reach_the_errorbar(self):
        ax = plot_metric_by_pipeline(
            _pipeline_results_df(), method_id="m0", linestyle=":"
        )
        assert [c[0].get_linestyle() for c in ax.containers] == [":", ":"]

    def test_save_writes_the_file_and_returns_none(self, tmp_path):
        path = tmp_path / "pipelines.png"
        result = plot_metric_by_pipeline(
            _pipeline_results_df(), method_id="m0", save=path
        )
        assert result is None
        assert path.exists()

    def test_none_table_names_the_cause(self):
        with pytest.raises(RuntimeError, match="the fit was not randomized"):
            plot_metric_by_pipeline(None, method_id="m0")

    def test_empty_table(self):
        with pytest.raises(RuntimeError, match="empty"):
            plot_metric_by_pipeline(_pipeline_results_df().iloc[0:0], method_id="m0")

    def test_unknown_method_id(self):
        with pytest.raises(ValueError, match=r"method_id 'm9' not found.*\['m0', 'm1'\]"):
            plot_metric_by_pipeline(_pipeline_results_df(), method_id="m9")

    def test_measure_the_table_does_not_carry(self):
        with pytest.raises(ValueError, match="consensus_pac_stability"):
            plot_metric_by_pipeline(_pipeline_results_df(), method_id="m0", measure="pac")


class TestSharedMetricDrawing:
    def test_both_metric_plots_draw_through_one_helper(
        self, monkeypatch, metric_results_df
    ):
        calls = []

        def spy(df, **kwargs):
            calls.append((kwargs["group_col"], kwargs["legend_title"], len(df)))
            return "drawn"

        monkeypatch.setattr(carve_plotting, "_draw_metric_lines", spy)
        assert plot_metric_over_n_clusters(metric_results_df) == "drawn"
        assert plot_metric_by_pipeline(_pipeline_results_df(), method_id="m0") == "drawn"
        assert calls == [("method_id", "Estimators", 3), ("pipeline", "Pipelines", 6)]


```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/test_plotting.py -q`

Expected: `1 error`. The first distinct errors:

- `ImportError: cannot import name 'plot_metric_by_pipeline' from 'carve._plotting'`

- [ ] Step 3: Implement

In `src/carve/_plotting.py`, replace:

```python

import warnings
from pathlib import Path
from typing import Literal
```

with:

```python

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal
```

In `src/carve/_plotting.py`, replace:

```python
        raise RuntimeError("Results DataFrame is empty.")

    if measure not in MEASURE_MAP:
        raise ValueError(
```

with:

```python
        raise RuntimeError("Results DataFrame is empty.")

    return _draw_metric_lines(
        results_df,
        group_col="method_id",
        label_of=_build_estimator_label,
        legend_title="Estimators",
        measure=measure,
        rule=rule,
        not_two=not_two,
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
        legend_loc=legend_loc,
        palette=palette,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def plot_metric_by_pipeline(
    preprocessing_df: pd.DataFrame | None,
    *,
    method_id: str,
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    palette: str = "Accent",
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
    **kwargs,
) -> Axes | None:
    """Plot a metric across the sweep axis, one line per preprocessing pipeline.

    The per-pipeline companion to :func:`plot_metric_over_n_clusters`, drawn
    by the same code. It reads ``preprocessing_results_`` instead of
    ``estimator_results_`` and draws one line per ``pipeline`` within one
    estimator configuration, ``method_id``, instead of one line per
    configuration. The dashed line marks the sweep value ``rule`` selects
    among the plotted rows; it can differ from the selection over
    ``estimator_results_``, which pools every pipeline. The table has no
    quantile columns, so ``rule="quantile"`` falls back to ``"max"`` with a
    warning.

    Parameters
    ----------
    preprocessing_df : pandas.DataFrame or None
        ``CARVE.preprocessing_results_``. None means the fit was not
        randomized.
    method_id : str
        The estimator configuration whose pipelines are drawn, a value of
        the ``method_id`` column.
    measure : str, default="stability"
        ``"stability"`` or ``"generalizability"``, or an alias of either. The
        table carries no consensus or accuracy columns.
    rule : str, default="1se"
        Selection rule for the marked sweep value: "max", "1se", "quantile".
    not_two : bool, default=False
        Exclude two-cluster rows when selecting the marked sweep value.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (9, 5.5).
    title : str, optional
        Figure title.
    xlabel : str, optional
        X-axis label. Default is derived from the sweep parameter.
    ylabel : str, optional
        Y-axis label. If None, auto-generated from measure name.
    legend : bool, default=True
        Whether to display a legend naming the pipelines.
    legend_loc : str, default="best"
        Legend location (passed to ax.legend).
    palette : str, default="Accent"
        Matplotlib colormap name for line colors.
    show : bool, default=False
        Whether to call plt.show() before returning.
    save : str or Path, optional
        Path to save the figure. If provided, the figure is saved and None
        is returned instead of the Axes object.
    dpi : int, default=300
        Dots per inch for saved figures.
    **kwargs
        Additional keyword arguments passed to ax.errorbar().

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None if save is used.

    Raises
    ------
    RuntimeError
        If ``preprocessing_df`` is None (the fit was not randomized) or
        empty.
    ValueError
        If ``method_id`` is not in the table, or ``measure`` names a column
        the table does not carry.
    """
    if preprocessing_df is None:
        raise RuntimeError(
            "There is no preprocessing table to plot: the fit was not "
            "randomized. Fit with randomize_preprocessing=True."
        )
    if preprocessing_df.empty:
        raise RuntimeError("Preprocessing DataFrame is empty.")

    method_ids = preprocessing_df["method_id"].astype(str)
    rows = preprocessing_df[method_ids == str(method_id)]
    if rows.empty:
        raise ValueError(
            f"method_id {method_id!r} not found in the preprocessing table. "
            f"Available: {sorted(method_ids.unique())}."
        )

    return _draw_metric_lines(
        rows,
        group_col="pipeline",
        label_of=lambda row: str(row["pipeline"]),
        legend_title="Pipelines",
        measure=measure,
        rule=rule,
        not_two=not_two,
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
        legend_loc=legend_loc,
        palette=palette,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def _draw_metric_lines(
    results_df: pd.DataFrame,
    *,
    group_col: str,
    label_of: Callable[[pd.Series], str],
    legend_title: str,
    measure: str,
    rule: str,
    not_two: bool,
    ax: Axes | None,
    figsize: tuple | None,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    legend: bool,
    legend_loc: str,
    palette: str,
    show: bool,
    save: str | Path | None,
    dpi: int,
    **kwargs,
) -> Axes | None:
    """Draw one metric line per group across the sweep axis.

    The drawing behind both :func:`plot_metric_over_n_clusters` (grouped on
    ``method_id``) and :func:`plot_metric_by_pipeline` (grouped on
    ``pipeline``), so the two plots cannot drift apart. ``results_df`` is
    non-empty and carries ``sweep_param``, ``sweep_value``, ``sweep_rank``,
    ``n_clusters_observed`` and the measure's column.
    """
    if measure not in MEASURE_MAP:
        raise ValueError(
```

In `src/carve/_plotting.py`, replace:

```python
    param = sweep_param_name(results_df)

    group_cols = ["method_id"]
    results_df = results_df.copy()
    grouped = results_df.groupby(group_cols)

    colors = plt.get_cmap(palette)(np.linspace(0, 1, len(grouped)))

    # --- Plot each estimator configuration ---
    for color_idx, (group_key, group_df) in enumerate(grouped):
        label_row = group_df.iloc[0]

        label = _build_estimator_label(label_row)
        group_df_sorted = group_df.sort_values(x_col)

```

with:

```python
    param = sweep_param_name(results_df)

    results_df = results_df.copy()
    grouped = results_df.groupby([group_col])

    colors = plt.get_cmap(palette)(np.linspace(0, 1, len(grouped)))

    # --- Plot each group ---
    for color_idx, (group_key, group_df) in enumerate(grouped):
        label = label_of(group_df.iloc[0])
        group_df_sorted = group_df.sort_values(x_col)

```

In `src/carve/_plotting.py`, replace:

```python
            framealpha=0.95,
            fontsize=10,
            title="Estimators" if rule else None,
        )

```

with:

```python
            framealpha=0.95,
            fontsize=10,
            title=legend_title if rule else None,
        )

```

- [ ] Step 4: Run the tests to verify they pass

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
.venv/bin/pytest tests/test_plotting.py tests/test_pl.py -q
```

Expected: ruff reports `All checks passed!`; pytest reports `114 passed`.

- [ ] Step 5: Commit

```bash
git add src/carve/_plotting.py tests/test_plotting.py
git commit -m "feat(plotting): plot_metric_by_pipeline, drawn by the helper plot_metric_over_n_clusters now shares"
```

---

### Task 7: CARVE.plot_metric_by_pipeline

Spec 5.5, `CARVE` level. With no `method_id`, the method resolves the configuration through `_select_row(measure, rule, not_two)`, so the default plot shows the pipelines behind the configuration CARVE selected.

Files:
- Modify: `src/carve/api.py`
- Test: `tests/test_api.py`

Interfaces:
- Consumes: `_plotting.plot_metric_by_pipeline` (Task 6); the `randomized_fit` fixture (Task 3).
- Produces: `CARVE.plot_metric_by_pipeline(*, method_id=None, measure="stability", rule="1se", not_two=False, ax=None, figsize=None, title=None, xlabel=None, ylabel=None, legend=True, legend_loc="best", palette="Accent", show=False, save=None, dpi=300, **kwargs) -> Axes | None`.

- [ ] Step 1: Write the failing tests

`_decoyed` deep-copies the module fixture, forces `rule="max"` to select `m1`, and renames `m0`'s pipelines with a `decoy | ` prefix, so a default that ignored the selection would draw decoys.

In `tests/test_api.py`, replace:

```python
"""Tests for CARVE public API (fit, get_labels, get_k, get_estimator, plotting, persistence)."""

import warnings

```

with:

```python
"""Tests for CARVE public API (fit, get_labels, get_k, get_estimator, plotting, persistence)."""

import copy
import warnings

```

In `tests/test_api.py`, replace:

```python
from carve import CARVE, LeidenClustering, LouvainClustering
from carve._pipeline import PipelineSpec, pipeline_from_spec
from carve._utils import resolve_anchors
from tests._helpers import make_njobs_spy, make_noise_embedding, make_seed_spy
```

with:

```python
from carve import CARVE, LeidenClustering, LouvainClustering
from carve._pipeline import PipelineSpec, pipeline_from_spec
from carve._plotting import plot_metric_by_pipeline
from carve._utils import resolve_anchors
from tests._helpers import make_njobs_spy, make_noise_embedding, make_seed_spy
```

In `tests/test_api.py`, replace:

```python


class TestShowProgress:
    def test_progress_bar_and_per_config_lines(self, X_two_clusters, capsys):
```

with:

```python


class TestPlotMetricByPipeline:
    """CARVE.plot_metric_by_pipeline draws the pipelines behind the selected
    configuration unless told otherwise."""

    @staticmethod
    def _decoyed(randomized_fit):
        """A copy of the randomized fit in which rule="max" can only select
        m1, and m0's pipelines are renamed as decoys."""
        _, fitted = randomized_fit
        carve = copy.deepcopy(fitted)
        results = carve.estimator_results_
        is_m1 = results["method_id"] == "m1"
        results.loc[is_m1, "ari_stability"] = 1.0
        results.loc[~is_m1, "ari_stability"] = 0.0
        table = carve.preprocessing_results_
        is_m0 = table["method_id"] == "m0"
        table.loc[is_m0, "pipeline"] = "decoy | " + table.loc[is_m0, "pipeline"]
        return carve

    def test_defaults_to_the_selected_configuration(self, randomized_fit):
        carve = self._decoyed(randomized_fit)
        assert carve._select_row(measure="stability", rule="max")[0]["method_id"] == "m1"
        ax = carve.plot_metric_by_pipeline(measure="stability", rule="max")
        drawn = {c.get_label() for c in ax.containers}
        m1 = carve.preprocessing_results_[carve.preprocessing_results_["method_id"] == "m1"]
        assert drawn == set(m1["pipeline"])
        assert not any(label.startswith("decoy") for label in drawn)

    def test_explicit_method_id_is_drawn(self, randomized_fit):
        carve = self._decoyed(randomized_fit)
        ax = carve.plot_metric_by_pipeline(method_id="m0", rule="max")
        assert len(ax.containers) > 1
        assert all(c.get_label().startswith("decoy | ") for c in ax.containers)

    def test_matches_the_module_function(self, randomized_fit):
        _, carve = randomized_fit
        row = carve._select_row(measure="generalizability", rule="1se")[0]
        a = carve.plot_metric_by_pipeline(measure="generalizability")
        b = plot_metric_by_pipeline(
            carve.preprocessing_results_,
            method_id=row["method_id"],
            measure="generalizability",
        )
        assert a.get_ylabel() == b.get_ylabel() == "ARI Generalizability"
        assert [c.get_label() for c in a.containers] == [
            c.get_label() for c in b.containers
        ]
        for ca, cb in zip(a.containers, b.containers):
            np.testing.assert_array_equal(ca[0].get_ydata(), cb[0].get_ydata())

    def test_save(self, randomized_fit, tmp_path):
        _, carve = randomized_fit
        path = tmp_path / "pipelines.png"
        assert carve.plot_metric_by_pipeline(save=path) is None
        assert path.exists()

    def test_not_randomized(self, fitted_carve):
        with pytest.raises(RuntimeError, match="the fit was not randomized"):
            fitted_carve.plot_metric_by_pipeline()

    def test_unfitted(self):
        with pytest.raises(RuntimeError, match="Call fit"):
            CARVE(verbose=0).plot_metric_by_pipeline()


class TestShowProgress:
    def test_progress_bar_and_per_config_lines(self, X_two_clusters, capsys):
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/test_api.py::TestPlotMetricByPipeline -q`

Expected: `6 failed`. The first distinct errors:

- `AttributeError: 'CARVE' object has no attribute 'plot_metric_by_pipeline'`

- [ ] Step 3: Implement

In `src/carve/api.py`, replace:

```python
from ._plotting import (
    plot_diagnostic_scatter as _plot_diagnostic_scatter,
)
from ._plotting import (
```

with:

```python
from ._plotting import (
    plot_diagnostic_scatter as _plot_diagnostic_scatter,
)
from ._plotting import (
    plot_metric_by_pipeline as _plot_metric_by_pipeline,
)
from ._plotting import (
```

In `src/carve/api.py`, replace:

```python
        )

    def plot_consensus_matrix(
        self,
```

with:

```python
        )

    def plot_metric_by_pipeline(
        self,
        *,
        method_id: str | None = None,
        measure: str = "stability",
        rule: str = "1se",
        not_two: bool = False,
        ax=None,
        figsize: tuple | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        palette: str = "Accent",
        show: bool = False,
        save: str | Path | None = None,
        dpi: int = 300,
        **kwargs,
    ) -> Axes | None:
        """Plot a metric across the sweep axis, one line per preprocessing pipeline.

        The per-pipeline companion to ``plot_metric_over_n_clusters`` for a fit
        with ``randomize_preprocessing=True``. Draws the rows of
        ``preprocessing_results_`` for one estimator configuration, one line
        per pipeline, with error bars at +/-1 standard error over the
        resamples that pipeline received.

        Parameters
        ----------
        method_id : str, optional
            The configuration to draw, a value of the ``method_id`` column.
            Defaults to the configuration CARVE selects under ``measure``,
            ``rule`` and ``not_two``, so the plot shows the pipelines behind
            the selected configuration.
        measure : str, default="stability"
            ``"stability"`` or ``"generalizability"``, or an alias of either.
        rule : str, default="1se"
            Selection rule: "max", "1se", "quantile". Used for the default
            ``method_id`` and for the sweep value marked among the plotted
            rows.
        not_two : bool, default=False
            Whether to exclude two-cluster configurations from both
            selections.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (9, 5.5).
        title : str, optional
            Figure title.
        xlabel : str, optional
            X-axis label. Default is derived from the sweep parameter.
        ylabel : str, optional
            Y-axis label. If None, auto-generated from metric name.
        legend : bool, default=True
            Whether to display a legend naming the pipelines.
        legend_loc : str, default="best"
            Legend location (passed to matplotlib's ax.legend).
        palette : str, default="Accent"
            Matplotlib colormap name for line colors.
        show : bool, default=False
            Whether to call plt.show() before returning.
        save : str or Path, optional
            Path to save the figure. If provided, the figure is saved and
            None is returned instead of an Axes object.
        dpi : int, default=300
            Dots per inch for saved figures.
        **kwargs
            Additional keyword arguments passed to matplotlib's errorbar
            function.

        Returns
        -------
        ax : matplotlib.axes.Axes or None
            The Axes object, or None if save was used.

        Raises
        ------
        RuntimeError
            If the instance has not been fitted, or was fitted without
            ``randomize_preprocessing=True``.
        ValueError
            If ``method_id`` is not in ``preprocessing_results_``.

        Examples
        --------
        >>> carve = CARVE().fit(X, randomize_preprocessing=True)
        >>> ax = carve.plot_metric_by_pipeline(measure="stability", rule="1se")
        """
        if self.estimator_results_ is None:
            raise RuntimeError("Call fit() first.")

        if method_id is None:
            row, _, _, _ = self._select_row(measure=measure, rule=rule, not_two=not_two)
            method_id = str(row["method_id"])

        return _plot_metric_by_pipeline(
            self.preprocessing_results_,
            method_id=method_id,
            measure=measure,
            rule=rule,
            not_two=not_two,
            ax=ax,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
            legend_loc=legend_loc,
            palette=palette,
            show=show,
            save=save,
            dpi=dpi,
            **kwargs,
        )

    def plot_consensus_matrix(
        self,
```

- [ ] Step 4: Run the tests to verify they pass

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
.venv/bin/pytest tests/test_api.py::TestPlotMetricByPipeline tests/test_api.py::TestPlotting -q
```

Expected: ruff reports `All checks passed!`; pytest reports `19 passed`.

- [ ] Step 5: Commit

```bash
git add src/carve/api.py tests/test_api.py
git commit -m "feat(api): CARVE.plot_metric_by_pipeline defaults to the selected configuration"
```

---

### Task 8: Per-pipeline table in AnnData and pl.metric_by_pipeline

Spec 5.5, `pl` level, and the plotting verification items for `tests/test_pl.py`, `tests/test_tl.py` and `tests/test_anndata.py`. `tl.attach_results` writes `uns[key]["preprocessing_results"]` through `results_to_uns` when the model was randomized and `store_results` is True, and records `selected_method_id` in `params` (refinement 3), which `pl.metric_by_pipeline` uses as its default.

Files:
- Modify: `src/carve/_anndata.py`
- Modify: `src/carve/pl/__init__.py`
- Modify: `src/carve/pl/_plots.py`
- Modify: `src/carve/tl/_carve.py`
- Test: `tests/test_anndata.py`
- Test: `tests/test_pl.py`
- Test: `tests/test_tl.py`

Interfaces:
- Consumes: `CARVE.plot_metric_by_pipeline` (Task 7) for the round-trip comparison; `_plotting.plot_metric_by_pipeline` (Task 6).
- Produces:
  - `adata.uns[key]["preprocessing_results"]` and `adata.uns[key]["params"]["selected_method_id"]`.
  - `pl.metric_by_pipeline(adata, *, key="carve", method_id=None, measure=None, rule=None, not_two=None, ax=None, figsize=None, title=None, xlabel=None, ylabel=None, legend=True, legend_loc="best", palette="Accent", show=False, save=None, dpi=300, **kwargs) -> Axes | None`, exported from `carve.pl`. Raises `KeyError("adata.uns['carve']['preprocessing_results'] not found. It is written only for a randomized fit, so either tl.carve ran without randomize_preprocessing=True or it ran with store_results=False.")`.
  - `_anndata.results_from_uns` re-casts `n_resamples` to int alongside `config_id` and `sweep_rank`.

- [ ] Step 1: Write the failing tests

The `randomized_written` fixture writes the AnnData to h5ad and reads it back, so every `pl` test runs on a round-tripped table.

In `tests/test_anndata.py`, replace:

```python
import anndata as ad
import numpy as np
import pytest
from scipy import sparse
```

with:

```python
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
```

In `tests/test_anndata.py`, replace:

```python
    labels_to_categorical,
    resolve_basis,
)
from carve._utils import ensure_2d_array
```

with:

```python
    labels_to_categorical,
    resolve_basis,
    results_from_uns,
    results_to_uns,
)
from carve._utils import ensure_2d_array
```

In `tests/test_anndata.py`, replace:

```python
        with pytest.raises(ValueError, match="not valid when X is an array"):
            self._model().fit(X_three_clusters, **{kwarg: value})
```

with:

```python
        with pytest.raises(ValueError, match="not valid when X is an array"):
            self._model().fit(X_three_clusters, **{kwarg: value})


# -----------------------------------------------------------------------
# results_to_uns and results_from_uns on the per-pipeline table
# -----------------------------------------------------------------------


class TestPreprocessingTableRoundTrip:
    def _table(self):
        return pd.DataFrame(
            {
                "method_id": ["m0", "m0"],
                "pipeline": ["identity | identity", "identity | PCA(n_components=2)"],
                "n_clusters": [2, 2],
                "n_resamples": [3, 5],
                "ari_stability": [0.5, 0.25],
                "ari_generalizability_se": [np.nan, 0.1],
                "sweep_rank": [0, 0],
            }
        )

    def test_strings_stay_strings_and_numbers_stay_numbers(self):
        clean = results_to_uns(self._table())
        assert list(clean["pipeline"]) == [
            "identity | identity",
            "identity | PCA(n_components=2)",
        ]
        assert clean["n_resamples"].dtype.kind == "i"
        assert np.isnan(clean.loc[0, "ari_generalizability_se"])

    def test_widened_counts_come_back_integral(self):
        # An h5ad round trip can return integer columns widened to float and
        # the index as strings; results_from_uns undoes both.
        stored = results_to_uns(self._table())
        stored["n_resamples"] = stored["n_resamples"].astype(float)
        stored["sweep_rank"] = stored["sweep_rank"].astype(float)
        stored.index = stored.index.astype(str)
        back = results_from_uns(stored)
        assert back["n_resamples"].dtype.kind == "i"
        assert list(back["n_resamples"]) == [3, 5]
        assert back.index.equals(pd.RangeIndex(2))
```

In `tests/test_pl.py`, replace:

```python
from matplotlib.collections import PolyCollection
from matplotlib.legend import Legend
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import carve
from carve import CARVE

```

with:

```python
from matplotlib.collections import PolyCollection
from matplotlib.legend import Legend
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import carve
import carve.pl._plots as pl_plots
from carve import CARVE

```

In `tests/test_pl.py`, replace:

```python
    carve.tl.attach_results(adata, model, **kwargs)
    return adata


```

with:

```python
    carve.tl.attach_results(adata, model, **kwargs)
    return adata


@pytest.fixture(scope="module")
def randomized_model():
    """CARVE fitted with randomized preprocessing, two estimators, on X_pca."""
    m = CARVE(
        n_clusters=np.arange(2, 5),
        estimator_param_grids=[
            (KMeans, {"n_clusters": [2, 3, 4], "n_init": [3]}),
            (AgglomerativeClustering, {"n_clusters": [2, 3, 4], "linkage": ["ward"]}),
        ],
        normalization_options=[(FunctionTransformer, {}), (StandardScaler, {})],
        dim_reduction_options=[(FunctionTransformer, {})],
        n_resamples=6,
        random_state=0,
    )
    m.fit(_bare_adata(), use_rep="X_pca", randomize_preprocessing=True)
    return m


@pytest.fixture(scope="module")
def randomized_written(randomized_model, tmp_path_factory):
    """A randomized result written into an AnnData and read back from h5ad."""
    adata = _bare_adata()
    carve.tl.attach_results(adata, randomized_model, use_rep="X_pca")
    path = tmp_path_factory.mktemp("h5ad") / "randomized.h5ad"
    adata.write_h5ad(path)
    return ad.read_h5ad(path)


```

In `tests/test_pl.py`, replace:

```python
            np.asarray(adata.layers["counts"], dtype=float)
        )
        np.testing.assert_allclose(ax.collections[0].get_offsets(), expected, rtol=1e-5)
```

with:

```python
            np.asarray(adata.layers["counts"], dtype=float)
        )
        np.testing.assert_allclose(ax.collections[0].get_offsets(), expected, rtol=1e-5)


# -----------------------------------------------------------------------
# metric_by_pipeline
# -----------------------------------------------------------------------


class TestMetricByPipeline:
    def test_matches_the_model_method_after_an_h5ad_round_trip(
        self, randomized_written, randomized_model
    ):
        a = carve.pl.metric_by_pipeline(randomized_written)
        b = randomized_model.plot_metric_by_pipeline(measure="stability", rule="1se")
        assert len(a.containers) == len(b.containers) == 2
        assert [c.get_label() for c in a.containers] == [
            c.get_label() for c in b.containers
        ]
        for ca, cb in zip(a.containers, b.containers):
            np.testing.assert_allclose(ca[0].get_xdata(), cb[0].get_xdata())
            np.testing.assert_allclose(ca[0].get_ydata(), cb[0].get_ydata())

    def test_defaults_come_from_the_recorded_selection(
        self, randomized_written, monkeypatch
    ):
        seen = {}

        def spy(table, **kwargs):
            seen.update(kwargs, n_rows=len(table))

        monkeypatch.setattr(pl_plots, "_plot_metric_by_pipeline", spy)
        for recorded in ("m0", "m1"):
            adata = randomized_written.copy()
            adata.uns["carve"]["params"]["selected_method_id"] = recorded
            carve.pl.metric_by_pipeline(adata)
            assert seen["method_id"] == recorded
            assert (seen["measure"], seen["rule"], seen["not_two"]) == (
                "stability",
                "1se",
                False,
            )
        carve.pl.metric_by_pipeline(
            randomized_written, method_id="m1", measure="generalizability", rule="max"
        )
        assert (seen["method_id"], seen["measure"], seen["rule"]) == (
            "m1",
            "generalizability",
            "max",
        )
        assert seen["n_rows"] == len(randomized_written.uns["carve"]["preprocessing_results"])

    @pytest.mark.parametrize("case", ["not_randomized", "store_results_false"])
    def test_absent_table_names_both_reasons(self, model, randomized_model, case):
        if case == "not_randomized":
            adata = _attach(model, use_rep="X_pca")
        else:
            adata = _attach(randomized_model, use_rep="X_pca", store_results=False)
        with pytest.raises(
            KeyError,
            match=r"preprocessing_results'\] not found.*randomize_preprocessing=True.*store_results=False",
        ):
            carve.pl.metric_by_pipeline(adata)

    def test_save_writes_the_file_and_returns_none(self, randomized_written, tmp_path):
        path = tmp_path / "metric_by_pipeline.png"
        assert carve.pl.metric_by_pipeline(randomized_written, save=path) is None
        assert path.exists()
```

In `tests/test_tl.py`, replace:

```python
import pandas as pd
import pytest

import carve
```

with:

```python
import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import carve
```

In `tests/test_tl.py`, replace:

```python
        assert adata_three_clusters.uns["carve"]["params"]["n_consensus_anchors"] == 40

```

with:

```python
        assert adata_three_clusters.uns["carve"]["params"]["n_consensus_anchors"] == 40


class TestPreprocessingResults:
    def _run_randomized(self, adata, **kwargs):
        return _run(
            adata,
            randomize_preprocessing=True,
            normalization_options=[(FunctionTransformer, {}), (StandardScaler, {})],
            dim_reduction_options=[(FunctionTransformer, {})],
            **kwargs,
        )

    def test_written_for_a_randomized_run(self, adata_three_clusters):
        self._run_randomized(adata_three_clusters)
        entry = adata_three_clusters.uns["carve"]
        assert set(entry) == {"params", "results", "preprocessing_results"}
        table = entry["preprocessing_results"]
        assert set(table["pipeline"]) == {
            "identity | identity",
            "StandardScaler | identity",
        }
        assert entry["params"]["selected_method_id"] in set(table["method_id"])

    def test_absent_without_randomization(self, adata_three_clusters):
        _run(adata_three_clusters)
        assert "preprocessing_results" not in adata_three_clusters.uns["carve"]

    def test_absent_with_store_results_false(self, adata_three_clusters):
        self._run_randomized(adata_three_clusters, store_results=False)
        assert set(adata_three_clusters.uns["carve"]) == {"params"}

    def test_selected_method_id_is_the_selected_row(self, adata_three_clusters):
        model = CARVE(n_clusters=np.arange(2, 5), n_resamples=4, random_state=0)
        model.fit(adata_three_clusters, use_rep="X_pca")
        for measure in ("stability", "generalizability"):
            carve.tl.attach_results(
                adata_three_clusters, model, use_rep="X_pca", measure=measure, rule="max"
            )
            row = model._select_row(measure=measure, rule="max")[0]
            params = adata_three_clusters.uns["carve"]["params"]
            assert params["selected_method_id"] == row["method_id"]
            assert params["selected_method_label"] == row["method_label"]
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/test_tl.py::TestPreprocessingResults tests/test_pl.py tests/test_anndata.py::TestPreprocessingTableRoundTrip -q`

Expected: `8 failed, 35 passed`. The first distinct errors:

- `AssertionError: assert {'params', 'results'} == {'params', 'p...s', 'results'}`
- `KeyError: 'selected_method_id'`
- `AttributeError: module 'carve.pl' has no attribute 'metric_by_pipeline'`

- [ ] Step 3: Implement

In `src/carve/_anndata.py`, replace:

```python

def results_to_uns(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise ``estimator_results_`` so it survives an h5ad round trip.

    Numeric and boolean columns pass through untouched. Every other column is
```

with:

```python

def results_to_uns(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise a results table so it survives an h5ad round trip.

    Used for both ``estimator_results_`` and ``preprocessing_results_``.

    Numeric and boolean columns pass through untouched. Every other column is
```

In `src/carve/_anndata.py`, replace:

```python
    ----------
    df : pandas.DataFrame
        The fitted ``estimator_results_`` table.

    Returns
```

with:

```python
    ----------
    df : pandas.DataFrame
        A fitted results table, ``estimator_results_`` or
        ``preprocessing_results_``.

    Returns
```

In `src/carve/_anndata.py`, replace:

```python

    An h5ad round trip turns the original ``RangeIndex`` into string labels and
    may widen integer columns, so the index is reset and the identity columns
    are re-cast.

    Parameters
```

with:

```python

    An h5ad round trip turns the original ``RangeIndex`` into string labels and
    may widen integer columns, so the index is reset and the identity and
    count columns are re-cast.

    Parameters
```

In `src/carve/_anndata.py`, replace:

```python
    """
    df = pd.DataFrame(stored).reset_index(drop=True)
    for col in ("config_id", "sweep_rank"):
        if col in df.columns:
            df[col] = df[col].astype(int)
```

with:

```python
    """
    df = pd.DataFrame(stored).reset_index(drop=True)
    for col in ("config_id", "sweep_rank", "n_resamples"):
        if col in df.columns:
            df[col] = df[col].astype(int)
```

In `src/carve/pl/__init__.py`, replace:

```python
    consensus_matrix,
    diagnostic_scatter,
    metric_over_n_clusters,
)
```

with:

```python
    consensus_matrix,
    diagnostic_scatter,
    metric_by_pipeline,
    metric_over_n_clusters,
)
```

In `src/carve/pl/__init__.py`, replace:

```python
    "consensus_matrix",
    "diagnostic_scatter",
    "metric_over_n_clusters",
]
```

with:

```python
    "consensus_matrix",
    "diagnostic_scatter",
    "metric_by_pipeline",
    "metric_over_n_clusters",
]
```

In `src/carve/pl/_plots.py`, replace:

```python
from .._plotting import plot_diagnostic_scatter as _plot_diagnostic_scatter
from .._plotting import (
    plot_metric_over_n_clusters as _plot_metric_over_n_clusters,
)
```

with:

```python
from .._plotting import plot_diagnostic_scatter as _plot_diagnostic_scatter
from .._plotting import (
    plot_metric_by_pipeline as _plot_metric_by_pipeline,
)
from .._plotting import (
    plot_metric_over_n_clusters as _plot_metric_over_n_clusters,
)
```

In `src/carve/pl/_plots.py`, replace:

```python
    "consensus_matrix",
    "diagnostic_scatter",
    "metric_over_n_clusters",
]
```

with:

```python
    "consensus_matrix",
    "diagnostic_scatter",
    "metric_by_pipeline",
    "metric_over_n_clusters",
]
```

In `src/carve/pl/_plots.py`, replace:

```python
        )
    return _anndata.results_from_uns(entry["results"])


```

with:

```python
        )
    return _anndata.results_from_uns(entry["results"])


def _preprocessing_results(adata: AnnData, key: str) -> pd.DataFrame:
    """Return the per-pipeline metrics table, or explain why it is absent."""
    entry = _entry(adata, key)
    if "preprocessing_results" not in entry:
        raise KeyError(
            f"adata.uns[{key!r}]['preprocessing_results'] not found. It is "
            "written only for a randomized fit, so either tl.carve ran without "
            "randomize_preprocessing=True or it ran with store_results=False."
        )
    return _anndata.results_from_uns(entry["preprocessing_results"])


```

In `src/carve/pl/_plots.py`, replace:

```python
    return _plot_metric_over_n_clusters(
        _results(adata, key),
        measure=(
            measure if measure is not None else str(params.get("measure", "stability"))
```

with:

```python
    return _plot_metric_over_n_clusters(
        _results(adata, key),
        measure=(
            measure if measure is not None else str(params.get("measure", "stability"))
        ),
        rule=rule if rule is not None else str(params.get("rule", "1se")),
        not_two=(
            not_two if not_two is not None else bool(params.get("not_two", False))
        ),
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
        legend_loc=legend_loc,
        palette=palette,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def metric_by_pipeline(
    adata: AnnData,
    *,
    key: str = "carve",
    method_id: str | None = None,
    measure: str | None = None,
    rule: str | None = None,
    not_two: bool | None = None,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    palette: str = "Accent",
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
    **kwargs,
) -> Axes | None:
    """Plot a validation metric across the sweep axis, one line per pipeline.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix carrying results from a
        :func:`carve.tl.carve` run with ``randomize_preprocessing=True``.
    key : str, default="carve"
        The ``key_added`` used when the results were written.
    method_id : str, optional
        Configuration whose pipelines are drawn. Defaults to the
        configuration selected when the results were written, so the lines
        belong to the labels in ``adata.obs``.
    measure : str, optional
        ``"stability"`` or ``"generalizability"``. Defaults to the measure
        recorded at fit time.
    rule : str, optional
        Selection rule for the marked sweep value. Defaults to the recorded
        rule.
    not_two : bool, optional
        Whether two-cluster solutions are excluded when marking the sweep
        value. Defaults to the recorded value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when None.
    figsize : tuple, optional
        Figure size, used only when ``ax`` is None.
    title, xlabel, ylabel : str, optional
        Axis text overrides.
    legend : bool, default=True
        Draw the legend.
    legend_loc : str, default="best"
        Legend location.
    palette : str, default="Accent"
        Matplotlib colormap name used for the per-pipeline colors.
    show : bool, default=False
        Call ``plt.show()`` before returning.
    save : str or pathlib.Path, optional
        Write the figure to this path and return None.
    dpi : int, default=300
        Resolution used when saving.
    **kwargs
        Forwarded to the underlying line plot.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None when ``save`` is given.

    Raises
    ------
    KeyError
        If the per-pipeline table is absent: the fit was not randomized, or
        the results were written with ``store_results=False``.
    """
    params = dict(_entry(adata, key)["params"])
    table = _preprocessing_results(adata, key)
    return _plot_metric_by_pipeline(
        table,
        method_id=(
            method_id if method_id is not None else str(params["selected_method_id"])
        ),
        measure=(
            measure if measure is not None else str(params.get("measure", "stability"))
```

In `src/carve/tl/_carve.py`, replace:

```python
        ``randomize_preprocessing=True``.
    randomize_preprocessing : bool, default=False
        Sample a random preprocessing pipeline per resample.
    classifier : sklearn classifier, optional
        Classifier used to score generalizability. Defaults to a random
```

with:

```python
        ``randomize_preprocessing=True``.
    randomize_preprocessing : bool, default=False
        Draw a preprocessing pipeline per resample and fit it on each
        subsample; see :meth:`carve.CARVE.fit`. The per-pipeline metrics are
        written to ``uns`` alongside the results.
    classifier : sklearn classifier, optional
        Classifier used to score generalizability. Defaults to a random
```

In `src/carve/tl/_carve.py`, replace:

```python
        nothing is written and a ``UserWarning`` is issued instead.
    store_results : bool, default=True
        Write the per-configuration metrics table into ``adata.uns``.
    mode : {"default", "stability", "generalizability"}, default="default"
        Which analyses to run.
```

with:

```python
        nothing is written and a ``UserWarning`` is issued instead.
    store_results : bool, default=True
        Write the per-configuration metrics table into ``adata.uns``, and the
        per-pipeline table when the fit was randomized.
    mode : {"default", "stability", "generalizability"}, default="default"
        Which analyses to run.
```

In `src/carve/tl/_carve.py`, replace:

```python
    ``adata.uns["carve"]["results"]``
        Per-configuration metrics, one row per configuration.

    Everything written survives :meth:`~anndata.AnnData.write_h5ad`. The
```

with:

```python
    ``adata.uns["carve"]["results"]``
        Per-configuration metrics, one row per configuration.
    ``adata.uns["carve"]["preprocessing_results"]``
        Per-pipeline metrics, one row per configuration, pipeline and sweep
        value. Written only when ``randomize_preprocessing=True``; read by
        :func:`carve.pl.metric_by_pipeline`.

    Everything written survives :meth:`~anndata.AnnData.write_h5ad`. The
```

In `src/carve/tl/_carve.py`, replace:

```python
        nothing is written and a ``UserWarning`` is issued instead.
    store_results : bool, default=True
        Write the per-configuration metrics table into ``adata.uns``.
    use_rep : str, optional
        Representation the model was fitted on, recorded in ``params`` so
```

with:

```python
        nothing is written and a ``UserWarning`` is issued instead.
    store_results : bool, default=True
        Write the per-configuration metrics table into ``adata.uns``, and the
        per-pipeline table when the fit was randomized.
    use_rep : str, optional
        Representation the model was fitted on, recorded in ``params`` so
```

In `src/carve/tl/_carve.py`, replace:

```python
    if store_results:
        entry["results"] = _anndata.results_to_uns(model.estimator_results_)
    adata.uns[key_added] = entry

```

with:

```python
    if store_results:
        entry["results"] = _anndata.results_to_uns(model.estimator_results_)
        if model.preprocessing_results_ is not None:
            entry["preprocessing_results"] = _anndata.results_to_uns(
                model.preprocessing_results_
            )
    adata.uns[key_added] = entry

```

In `src/carve/tl/_carve.py`, replace:

```python
    if "method_label" in row.index:
        params["selected_method_label"] = row["method_label"]

    return params
```

with:

```python
    if "method_label" in row.index:
        params["selected_method_label"] = row["method_label"]
    if "method_id" in row.index:
        # pl.metric_by_pipeline defaults to this configuration's pipelines.
        params["selected_method_id"] = row["method_id"]

    return params
```

- [ ] Step 4: Run the tests to verify they pass

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
.venv/bin/pytest tests/test_tl.py tests/test_pl.py tests/test_anndata.py tests/test_h5ad_roundtrip.py -q
```

Expected: ruff reports `All checks passed!`; pytest reports `141 passed`.

- [ ] Step 5: Commit

```bash
git add src/carve/_anndata.py src/carve/pl/__init__.py src/carve/pl/_plots.py src/carve/tl/_carve.py tests/test_anndata.py tests/test_pl.py tests/test_tl.py
git commit -m "feat(pl): store the per-pipeline table in uns and add pl.metric_by_pipeline"
```

---

### Task 9: Full verification

No code changes. This task checks the branch the way CI will.

- [ ] Step 1: Lint and format check, as CI runs them

```bash
.venv/bin/ruff check src/
.venv/bin/ruff format --check src/
```

Expected: `All checks passed!` and `64 files already formatted`.

- [ ] Step 2: The carve leg, with CI's flags

```bash
.venv/bin/pytest tests --ignore=tests/benchmarks --tb=short --cov=carve --cov-report=term-missing --cov-fail-under=75 -q
```

Expected: `873 passed`, no warnings summary, `Required test coverage of 75% reached. Total coverage: 94.43%`, and `src/carve/_runner.py` at 100%. Measured in 344 s on Python 3.12.11; CI runs 3.13, where an environmental skip can move the count. pytest-xdist is not installed in the main venv, so do not pass `-n`.

- [ ] Step 3: The benchmarks leg

```bash
.venv/bin/pytest tests/benchmarks --tb=short -q
```

Expected: `786 passed, 10 skipped` on a machine with `data/Klein` present, without rpy2 and without `CARVE_RUN_REGRESSION`; those three conditions are the skips. This plan changes no benchmark code, but `src/benchmarks/figures/_carve_output.py` calls `plot_metric_over_n_clusters`, which Task 6 moved onto the shared helper, so this leg is part of the check. When the plan was verified this leg ran against the patched source imported from a scratch copy: 778 passed, and the 8 tests that resolve paths next to the imported package (six Klein loader tests, two `VIS_ROOT` tests) failed only because the copy had no `data/` or `vis/`. The same 8 pass against the repository.

- [ ] Step 4: Fit the default grids end to end

No unit test runs t-SNE and UMAP through CARVE's default option lists, and the UMAP seeding warning in refinement 8 was found this way.

```bash
MPLBACKEND=Agg .venv/bin/python - <<'EOF'
import warnings
import numpy as np
from sklearn.cluster import KMeans
from carve import CARVE

rng = np.random.RandomState(0)
X = np.vstack([rng.randn(50, 8) + c for c in np.eye(8)[:3] * 5])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = CARVE(
        n_clusters=np.array([2, 3, 4]),
        n_resamples=16,
        estimator_param_grids=[(KMeans, {"n_clusters": [2, 3, 4], "n_init": [3]})],
        random_state=0,
        verbose=2,
    ).fit(X, randomize_preprocessing=True)
print(sorted({w.category.__name__ for w in caught}), [str(w.message)[:24] for w in caught if w.category is UserWarning])
df = model.preprocessing_results_
print(df["pipeline"].nunique(), len(df), sorted(set(df.groupby("sweep_value")["n_resamples"].sum())))
print(model.get_labels().shape, len(model.plot_metric_by_pipeline().containers))
EOF
```

Expected, in about 12 s: the header contains `[CARVE] normalization      : identity, StandardScaler` (log1p is dropped because X has negatives) and `[CARVE] dim_reduction      : identity, PCA, TSNE, UMAP`, and the last three lines are

```
['ImportWarning', 'UserWarning'] ['X has negative values (m']
13 39 [16]
(150,) 13
```

The `ImportWarning` is umap-learn's note that TensorFlow is absent. A second `UserWarning` entry reading `n_jobs value 1 overridd` would mean the Task 3 filter is missing.

- [ ] Step 5: Confirm the regression fixture was written once

```bash
git log --oneline main..HEAD -- tests/fixtures/nonrandomized_gate.npz
```

Expected: one line, Task 1's commit.

- [ ] Step 6: Confirm the R port was not touched

```bash
git diff --stat main..HEAD -- carve-r/ .github/workflows/r-ci.yml
git status --short -- carve-r/ .github/workflows/r-ci.yml
```

Expected: both commands print nothing. Any output means an R file changed, which is outside this plan's scope; revert it rather than keeping it.

- [ ] Step 7: Finish the branch

Use superpowers:finishing-a-development-branch.

---

## After this plan

Tracked in spec 8, not part of this branch:

- R port, out of scope for this plan and deferred on purpose. `carve-r/R/pipeline.R` and `runner.R` still fit the pipeline on the full X and train the classifier on the embedding, and `plotting.R` has no per-pipeline plot. `r-ci.yml` checks the port separately; same status as the deferred anchored-consensus port.
- Manuscript, for the author in Overleaf: the S1 Text fit scope and classifier features, `pipeline_df_` renamed `preprocessing_results_`, and the default grids if S1 Table lists them. Suggested text is in the case-study spec, section 9.
- Independent draws per subsample, as in Chang et al., as an option.
- The Cusanovich case-study plan (`docs/superpowers/specs/2026-09-13-cusanovich-randomized-preprocessing-case-study-design.md`), which consumes `preprocessing_results_`, `preprocessing_pipelines_`, `pipeline_from_spec` and `plot_metric_by_pipeline`.
