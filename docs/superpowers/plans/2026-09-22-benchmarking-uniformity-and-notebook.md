# Benchmarking Uniformity and Notebook Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every simulated benchmark scenario run the same experiment, stop the runner from producing silently wrong artifacts, and replace the notebook's inline tables with one visual dashboard per scenario.

**Architecture:** The benchmarks package is already decomposed into a registry (the study design), a runner, an artifact layer, drawing primitives in `_panels`, and one module per figure. This plan changes values and adds primitives inside that structure rather than restructuring it. One defect is fixed in `carve` itself (`cluster.py`'s spectral eigensolver); everything else lives under `src/benchmarks/`.

**Tech Stack:** Python 3.13, numpy, pandas, scikit-learn, scipy, joblib, matplotlib, pytest, ruff 0.16.4. Run everything from `code/` using `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-09-21-benchmarking-uniformity-and-notebook-design.md`

## Global Constraints

- Work from `/Users/kaiwycik/GitHub/CARVE/code`. It is the only git repo in the tree.
- Use the pinned toolchain in `code/.venv/`. Never a system `ruff` or `pytest`.
- `ruff check src/` and `ruff format src/` only. Never run ruff on `tests/` or `notebooks/`.
- `filterwarnings = error` is on. A new warning fails the suite and needs either a `match=` assertion or a targeted `filterwarnings` mark.
- `addopts` is `-ra` only. Do not pass `--cov-fail-under=75` locally; it fails a single-file run.
- Tests mirror modules 1:1. `_calibrate.py` ↔ `tests/benchmarks/test_calibrate.py`. A new test belongs in the matching file.
- Dependency direction: `_types`, `_utils`, `_anndata`, `cluster` are leaves. Nothing under `_*` imports `api`. `benchmarks` imports `carve`; `carve` never imports `benchmarks`.
- `benchmarks._theme` is the single source for palette, rcParams and geometry. Never add a module-level color literal anywhere else.
- No `logging`. Diagnostics go through `warnings.warn(..., stacklevel=2)` and `print()` gated on `verbose`.
- American spelling in prose. No bold or italics in code comments, docstrings or notebook markdown.
- The R port under `carve-r/` is out of scope. Do not touch it.
- `../overleaf/` is read-only. Never write there.
- Every commit message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

## File Structure

**Modified in `src/carve/`:**
- `cluster.py` — `SpectralClustering._spectral_embedding`, the sparse branch only.

**Modified in `src/benchmarks/`:**
- `_registry.py` — `_N_TREES`, `_ESTIMATORS`, `_SHARED`, `ACTIVE_ANCHORS`; gains `CALIBRATED_ANCHORS` and `TABLE_ROW_GROUPS`.
- `_calibrate.py` — `CALIBRATION_SPACE` becomes per-scenario knobs with a direction.
- `_artifacts.py` — gains `widest_run`; `build_manifest` gains `status`.
- `_run.py` — `run_scenario` writes the manifest twice, guards resume on `git_sha`, resolves `n_jobs` per scenario class.
- `run.py` — `--tables` uses the shared resolver; `--n-jobs` default becomes per-scenario; gains `--allow-code-change`.
- `_theme.py` — re-stepped `METRIC_COLORS`, new `COMPARATOR_DASHES` / `metric_dashes`, new sequential colormap.
- `_panels.py` — `metric_lines` gains a per-seed strip; gains `normalized_criterion`, `criterion_curves`, `k_hat_heatmap`.
- `_tables.py` — gains `render_paired_tex`; `write_tables` calls it.
- `tables.py` — `TABLE_CAPTIONS["gaussians_dimensionality"]` correction.

**Created:**
- `src/benchmarks/figures/_scenario_dashboard.py` — `figure_scenario_dashboard`.
- `docs/superpowers/notes/2026-09-22-calibration-comparison.md` — the calibrated-vs-published comparison.
- `docs/superpowers/notes/2026-09-22-rerun-handoff.md` — the re-run handoff.

**Rewritten:**
- `notebooks/Benchmarking.ipynb` — helper cell and the eight scenario sections.

---

## Task 1: Branch setup and the spectral eigensolver fix

**Files:**
- Modify: `src/carve/cluster.py:259-272`
- Test: `tests/test_cluster.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a deterministic, converged `SpectralClustering` at n >= 1000. Tasks 3, 5 and 19 depend on it.

- [ ] **Step 1: Fast-forward main and branch**

`rho-b-ablation` is 27 commits ahead of `main` and 0 behind, so this is a clean fast-forward. Task 9 needs that branch's `cpu_cap` and `thread_cap_for`.

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
git checkout main
git merge --ff-only rho-b-ablation
git push
git checkout -b benchmark-uniformity
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_cluster.py`, inside the same class as `test_sparse_path_is_reproducible_for_a_fixed_seed`. The existing test at line 256 uses n=1000, k=2 on easy moons, where the flip does not occur; this one uses the case that actually breaks.

```python
    def test_sparse_path_is_deterministic_on_a_hard_five_cluster_problem(self):
        """n=1500, k=5 is where which='SM' without a start vector flips.

        The existing n=1000, k=2 test passes on the broken solver. Seven of
        eleven repeated fits of this problem returned a different partition
        before the fix, 412 of 1500 points reassigned.
        """
        from sklearn.datasets import make_blobs

        X, _ = make_blobs(
            n_samples=1500, centers=5, n_features=2, cluster_std=2.4, random_state=7
        )
        first = SpectralClustering(n_clusters=5, random_state=42).fit_predict(X)
        for _ in range(4):
            repeat = SpectralClustering(n_clusters=5, random_state=42).fit_predict(X)
            np.testing.assert_array_equal(first, repeat)

    def test_sparse_path_converges_on_concentric_circles(self):
        """The broken solver returns an unconverged embedding, not a wrong one.

        Shift-invert recovers this exactly; which='SM' scored around 0.79.
        """
        from sklearn.datasets import make_circles

        X, y_true = make_circles(
            n_samples=1500, noise=0.04, factor=0.45, random_state=0
        )
        labels = SpectralClustering(n_clusters=2, random_state=42).fit_predict(X)
        assert adjusted_rand_score(y_true, labels) > 0.99
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cluster.py -k "hard_five_cluster or concentric_circles" -v`

Expected: at least one FAIL. The determinism test fails with an array mismatch; the circles test fails with an ARI well below 0.99. If the determinism test passes on this machine, do not loosen it — ARPACK's start vector draw is ambient, so a pass here is luck. Proceed to Step 4 regardless, because the circles test is the load-bearing one.

- [ ] **Step 4: Fix the solver**

In `src/carve/cluster.py`, the sparse branch of `_spectral_embedding`. Replace this line:

```python
                vals, vecs = eigsh(Lsym, k=k, which="SM", tol=1e-4, maxiter=5000)
```

with:

```python
                # Shift-invert at sigma=0, mirroring carve-r/R/cluster.R:270.
                # which="SM" with no v0 draws its start vector from the ambient
                # global RNG, and on a hard problem it returns before
                # converging: same data and same random_state gave two
                # different partitions, and the ARI it reached was 0.79 where
                # this reaches 1.00. It is also 13-20x slower, because ARPACK
                # converges to the smallest eigenvalues slowly and to the
                # largest quickly. sigma=0 maps the smallest eigenvalues of
                # Lsym to the largest of (Lsym - sigma*I)^-1, so "LM" finds
                # what "SM" was asking for.
                vals, vecs = eigsh(
                    Lsym, k=k, sigma=0.0, which="LM", tol=1e-4, maxiter=5000
                )
```

Leave the `except ArpackNoConvergence` block and the dense fallback exactly as they are.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cluster.py -v`

Expected: PASS, all of them. `test_sparse_path_recovers_moons` and `test_dense_path_just_below_the_sparse_threshold` must still pass.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff format src/carve/cluster.py
.venv/bin/ruff check src/
git add src/carve/cluster.py tests/test_cluster.py
git commit -m "fix(cluster): shift-invert the sparse spectral eigensolver

which='SM' with no start vector was nondeterministic, unconverged and
13-20x slow above n=1000. Mirrors the R port, which is already correct.

The existing sparse-path reproducibility test passes on the broken code
because n=1000, k=2 on easy moons does not flip; the new tests use n=1500,
k=5 and concentric circles, which do.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Uniform n_trees, and two registry corrections

**Files:**
- Modify: `src/benchmarks/_registry.py` (`_N_TREES`, `_SHARED["swiss_rolls"]`)
- Modify: `src/benchmarks/tables.py` (`TABLE_CAPTIONS["gaussians_dimensionality"]`)
- Test: `tests/benchmarks/test_registry.py`, `tests/benchmarks/test_tables_cli.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SCENARIOS[name].n_trees == 500` for all eight names.

- [ ] **Step 1: Replace the split n_trees tests**

In `tests/benchmarks/test_registry.py`, delete both `test_n_trees_is_500_for_the_scenarios_the_notebook_ran_with_500_trees` and `test_n_trees_defaults_to_100_for_the_remaining_scenarios` (lines 187-196), and put this in their place:

```python
    @pytest.mark.parametrize("name", DIFFICULTY_SCENARIOS + SCALING_SCENARIOS)
    def test_n_trees_is_500_everywhere(self, name):
        """One forest size for every scenario.

        The published benchmark ran 500 trees on circles, moons and swiss
        rolls and 100 elsewhere, decided by which notebook cell happened to
        pass the argument rather than by anything about the data. 500 is the
        better generalizability estimate and the cost is linear in a term
        that is not the bottleneck.
        """
        assert SCENARIOS[name].n_trees == 500
```

- [ ] **Step 2: Add a test that the redundant key is gone**

In the same class:

```python
    def test_swiss_rolls_does_not_restate_the_simulator_default(self):
        """center_box=3.0 is simulate_clusters's own default.

        Restating it made the key read as a deliberate choice for this
        scenario when it is not one, and it was the only scenario that set
        it.
        """
        assert "center_box" not in SCENARIOS["swiss_rolls"].shared
```

- [ ] **Step 3: Add the caption test**

In `tests/benchmarks/test_tables_cli.py`:

```python
def test_dimensionality_caption_names_the_feature_dimension():
    """The axis is p, the ambient feature dimension, not embed_dim.

    _registry's comment on _AXES says so and the manuscript's S3 Fig says
    "feature dimension p"; the caption said "embedding dimension", which is
    a different quantity that this scenario holds fixed at 64.
    """
    from benchmarks.tables import TABLE_CAPTIONS

    caption = TABLE_CAPTIONS["gaussians_dimensionality"]
    assert "feature dimension" in caption
    assert "embedding dimension" not in caption
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_registry.py tests/benchmarks/test_tables_cli.py -k "n_trees or simulator_default or dimensionality_caption" -v`

Expected: FAIL. Five n_trees parametrizations fail (the three 100-tree difficulty scenarios plus the two scaling ones), `center_box` is present, and the caption says "embedding dimension".

- [ ] **Step 5: Make the three changes**

In `src/benchmarks/_registry.py`, replace the `_N_TREES` block and its comment:

```python
# One forest size for every scenario. The published benchmark did not use
# one value everywhere -- notebook cells 29, 35 and 41 passed n_trees=500
# for circles, moons and swiss rolls while cells 11, 17, 23, 48 and 53
# omitted it and got Scenario's default of 100 -- but that split tracked
# which cell an argument was typed into, not anything about the data. 500 is
# the better generalizability estimate; the cost is linear in a term that is
# not the bottleneck.
_N_TREES: dict[str, int] = dict.fromkeys(_ESTIMATORS, 500)
```

Move that block below `_ESTIMATORS`, since it now reads from it.

In `_SHARED["swiss_rolls"]`, delete the `"center_box": 3.0,` line. It is `simulate_clusters`'s own default.

In `src/benchmarks/tables.py`, replace the `gaussians_dimensionality` caption:

```python
    "gaussians_dimensionality": (
        "Gaussian mixtures over feature dimension: ARI at the selected k and k-recovery."
    ),
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_registry.py tests/benchmarks/test_tables_cli.py -v`

Expected: PASS.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_registry.py src/benchmarks/tables.py
.venv/bin/ruff check src/
git add src/benchmarks/_registry.py src/benchmarks/tables.py tests/benchmarks/
git commit -m "refactor(benchmarks): one forest size, and two registry corrections

n_trees unifies at 500. The 500/100 split tracked which published notebook
cell the argument was typed into, not the data.

Drops swiss_rolls's center_box=3.0, which restates the simulator default,
and corrects the gaussians_dimensionality caption: the axis is p, the
ambient feature dimension, not embed_dim.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: swiss_rolls moves to spectral

**Files:**
- Modify: `src/benchmarks/_registry.py` (`_ESTIMATORS["swiss_rolls"]`)
- Test: `tests/benchmarks/test_registry.py`

**Interfaces:**
- Consumes: Task 1's fixed solver. This change is only correct on top of it.
- Produces: `SCENARIOS["swiss_rolls"].estimator.name == "spectral"`.

- [ ] **Step 1: Replace the estimator test**

In `tests/benchmarks/test_registry.py`, replace `test_swiss_rolls_uses_the_estimator_the_benchmark_actually_ran` (line 168-169) with:

```python
    @pytest.mark.parametrize("name", ("circles", "moons", "swiss_rolls"))
    def test_the_rff_family_shares_one_estimator(self, name):
        """All three RFF-embedded scenarios run self-tuning spectral.

        swiss_rolls ran Ward because spectral lost to it by 0.22 ARI at the
        easy anchor -- which was the which='SM' eigensolver failing to
        converge, not a property of the data. With the shift-invert solver,
        oracle ARI at k*=5 over 20 datasets is 1.000/0.900/0.834 for spectral
        against 0.982/0.860/0.752 for Ward, and spectral is 30x faster.
        """
        assert SCENARIOS[name].estimator.name == "spectral"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_registry.py -k rff_family -v`

Expected: FAIL on the `swiss_rolls` parametrization with `assert 'agglomerative' == 'spectral'`.

- [ ] **Step 3: Change the estimator**

In `src/benchmarks/_registry.py`, in `_ESTIMATORS`, replace the `swiss_rolls` entry and its comment:

```python
    # The published S1 Fig panel was drawn with "spectral" while the
    # benchmark that produced Table S7 ran agglomerative, and S3 Text said
    # Ward. Ward was chosen because spectral scored worse here -- which was
    # carve.cluster's which="SM" eigensolver failing to converge at n=1500,
    # not the data. With the shift-invert solver spectral wins at every
    # difficulty and runs 30x faster, so the illustration was right and the
    # numbers were not.
    "swiss_rolls": "spectral",
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_registry.py -v`

Expected: PASS.

- [ ] **Step 5: Drop swiss_rolls from the regression oracle**

`tests/benchmarks/test_regression.py:63` maps `swiss_rolls` to `results_swiss_rolls`. Those committed numbers came from Ward, so the scenario is no longer comparable. Remove that line from `UNAFFECTED` and extend the comment above it:

```python
# The remaining three scenarios use k-means and agglomerative clustering,
# which are deterministic here, and they still exercise every shared code
# path: the runner, the artifact layer, the seed derivation, the classical
# indices, and the CARVE metric extraction.
#
# swiss_rolls was in this set until it moved to spectral clustering. Its
# committed CSV was produced by Ward, so it is no longer a comparable
# oracle for this scenario.
UNAFFECTED = {
    "gaussians": "results_gaussian",
    "t_dist": "results_t_dist",
    "t_dist_noise": "results_t_dist_noise",
}
```

- [ ] **Step 6: Run the benchmarks leg and commit**

Run: `.venv/bin/python -m pytest tests/benchmarks -q`

Expected: PASS. The regression tests stay skipped.

```bash
.venv/bin/ruff format src/benchmarks/_registry.py
.venv/bin/ruff check src/
git add src/benchmarks/_registry.py tests/benchmarks/
git commit -m "feat(benchmarks): swiss_rolls runs spectral, like its RFF siblings

Consequent on the eigensolver fix, not an independent choice. Measured over
20 datasets at k*=5: spectral 1.000/0.900/0.834 against Ward
0.982/0.860/0.752, and 30x faster per fit.

Drops swiss_rolls from the regression oracle: its committed CSV came from
Ward.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: Per-scenario calibration knobs

**Files:**
- Modify: `src/benchmarks/_calibrate.py`
- Test: `tests/benchmarks/test_calibrate.py`

**Interfaces:**
- Consumes: Task 1's solver, Task 3's estimator.
- Produces:
  - `CalibrationKnob(parameter: str, low: float, high: float, increasing: bool, multiplier: bool)`
  - `CALIBRATION_KNOBS: dict[str, CalibrationKnob]` keyed by scenario name
  - `knob_for(scenario_name: str) -> CalibrationKnob`
  - `apply_knob(anchor: Mapping[str, Any], knob: CalibrationKnob, value: float) -> dict[str, Any]`
  - `oracle_ari_stats(scenario, axis_label, overrides, *, n_seeds, random_state) -> tuple[float, float]`
  - `calibrate_anchor(...) -> dict` with keys `anchor`, `knob_value`, `achieved_ari`, `achieved_sd`, `target`, `in_band`
  - Task 5 consumes all of these.

- [ ] **Step 1: Write the failing tests**

Replace `TestTargets.test_the_search_parameter_is_documented` in `tests/benchmarks/test_calibrate.py` and add a new class:

```python
class TestKnobs:
    def test_every_difficulty_scenario_declares_a_knob(self):
        from benchmarks._registry import DIFFICULTY_AXIS, SCENARIOS

        difficulty = [
            name
            for name, s in SCENARIOS.items()
            if s.axis.name == DIFFICULTY_AXIS.name
        ]
        assert set(CALIBRATION_KNOBS) == set(difficulty)

    def test_the_rff_scenarios_calibrate_on_the_bandwidth(self):
        """cluster_scale does not move circles or moons at all.

        Measured over 20 datasets with the fixed solver: multiplying
        cluster_scale by 3 leaves oracle ARI at 1.000 +/- 0.000. embed_param
        is the knob, and all three target bands live between about 2.7 and
        3.5.
        """
        for name in ("circles", "moons"):
            knob = knob_for(name)
            assert knob.parameter == "embed_param"
            assert knob.increasing is True
            assert knob.multiplier is False
            assert (knob.low, knob.high) == (2.0, 4.0)

    def test_the_scale_driven_scenarios_calibrate_on_a_multiplier(self):
        for name in ("gaussians", "t_dist", "t_dist_noise", "swiss_rolls"):
            knob = knob_for(name)
            assert knob.parameter == "cluster_scale"
            assert knob.increasing is False
            assert knob.multiplier is True

    def test_knob_for_rejects_an_unregistered_scenario(self):
        with pytest.raises(KeyError, match="gaussians_samples"):
            knob_for("gaussians_samples")


class TestApplyKnob:
    def test_a_multiplier_scales_the_published_vector(self):
        """The nonlinear anchors carry unequal per-cluster scales.

        Replacing the vector with a scalar, which is what the first
        implementation did, flattens that structure. Scaling preserves it.
        """
        knob = CalibrationKnob(
            parameter="cluster_scale",
            low=0.1,
            high=3.0,
            increasing=False,
            multiplier=True,
        )
        anchor = {"cluster_scale": [4.0, 2.0, 2.0], "corr_strength": 0.3}
        out = apply_knob(anchor, knob, 1.5)
        assert out["cluster_scale"] == [6.0, 3.0, 3.0]
        assert out["corr_strength"] == 0.3

    def test_a_multiplier_handles_a_scalar_published_value(self):
        knob = CalibrationKnob(
            parameter="cluster_scale",
            low=0.1,
            high=3.0,
            increasing=False,
            multiplier=True,
        )
        assert apply_knob({"cluster_scale": 2.0}, knob, 0.5)["cluster_scale"] == 1.0

    def test_an_absolute_knob_replaces_the_value(self):
        knob = CalibrationKnob(
            parameter="embed_param",
            low=2.0,
            high=4.0,
            increasing=True,
            multiplier=False,
        )
        out = apply_knob({"embed_param": 7.3, "corr_strength": 0.2}, knob, 3.1)
        assert out["embed_param"] == 3.1
        assert out["corr_strength"] == 0.2

    def test_a_multiplier_needs_the_parameter_to_be_present(self):
        knob = CalibrationKnob(
            parameter="cluster_scale",
            low=0.1,
            high=3.0,
            increasing=False,
            multiplier=True,
        )
        with pytest.raises(KeyError, match="cluster_scale"):
            apply_knob({"corr_strength": 0.3}, knob, 1.5)


class TestDirection:
    def test_an_increasing_knob_bisects_the_other_way(self, rising_scenario):
        """A bisection that assumes ARI falls with the knob walks away from
        the band when it rises. calibrate_anchor must read the direction off
        the knob rather than assuming it."""
        knob = CalibrationKnob(
            parameter="embed_param",
            low=0.5,
            high=6.0,
            increasing=True,
            multiplier=False,
        )
        result = calibrate_anchor(
            rising_scenario,
            "easy",
            target=(0.9, 1.0),
            knob=knob,
            n_seeds=3,
            random_state=0,
            max_iter=12,
        )
        assert result["in_band"]


class TestCalibrateAnchorResult:
    def test_reports_the_spread_alongside_the_mean(self, tiny_scenario):
        """The RFF transition band has sd 0.10-0.23 where the mean is on
        target, so an anchor's spread is part of what a reader needs."""
        result = calibrate_anchor(
            tiny_scenario, "easy", target=(0.9, 1.0), n_seeds=3, random_state=0
        )
        assert set(result) == {
            "anchor",
            "knob_value",
            "achieved_ari",
            "achieved_sd",
            "target",
            "in_band",
        }
        assert result["achieved_sd"] >= 0.0
        assert isinstance(result["knob_value"], float)
```

Add the `rising_scenario` fixture beside `tiny_scenario`:

```python
@pytest.fixture
def rising_scenario():
    """A nonlinear scenario whose ARI rises with embed_param.

    The same shape as circles: below the bandwidth threshold the embedding
    destroys the structure, above it the problem is trivial.
    """
    return Scenario(
        name="rising",
        axis=Axis(name="difficulty_level", values=(0,), labels=("easy",)),
        anchors={"easy": {"embed_param": 1.0}},
        shared={
            "n_total": 300,
            "p": 8,
            "distribution": "circles",
            "nonlinear": True,
            "embed_dim": 32,
            "corr_type": "ar1",
        },
        estimator=EstimatorSpec(name="spectral"),
        candidate_k=(3, 4, 5),
        n_seeds=3,
    )
```

Update the imports at the top of the file:

```python
from benchmarks._calibrate import (
    CALIBRATION_KNOBS,
    TARGET_ARI_BANDS,
    CalibrationKnob,
    apply_knob,
    calibrate_anchor,
    knob_for,
    mean_oracle_ari,
    oracle_ari_stats,
)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_calibrate.py -v`

Expected: collection error — `cannot import name 'CALIBRATION_KNOBS'`.

- [ ] **Step 3: Rewrite the calibration knobs**

In `src/benchmarks/_calibrate.py`, replace `CALIBRATION_SPACE` with:

```python
@dataclass(frozen=True)
class CalibrationKnob:
    """The one anchor parameter a scenario's difficulty is bisected on.

    parameter names a simulate_clusters keyword. multiplier=True scales the
    anchor's published value rather than replacing it, which is what keeps
    the unequal per-cluster cluster_scale vectors the nonlinear anchors
    carry -- [4.38, 4.08, 4.08, 4.08, 4.08] on circles/hard -- from being
    flattened to a scalar by the search.

    increasing says which way ARI moves as the parameter grows. It is
    declared rather than assumed because the two knobs run in opposite
    directions: diffuser clusters score lower, a wider RFF bandwidth scores
    higher.
    """

    parameter: str
    low: float
    high: float
    increasing: bool
    multiplier: bool

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(
                f"CalibrationKnob({self.parameter!r}): low must be below high, "
                f"got {self.low} and {self.high}."
            )


_SCALE_KNOB = CalibrationKnob(
    parameter="cluster_scale",
    low=0.1,
    high=3.0,
    increasing=False,
    multiplier=True,
)

# circles and moons do not respond to cluster_scale at all: measured over 20
# datasets with the fixed eigensolver, multiplying it by 3 leaves oracle ARI
# at 1.000 +/- 0.000 at every anchor. corr_strength does nothing either
# (1.000 at 0.23 through 0.9). embed_param, the random Fourier feature
# bandwidth, is the knob, and it behaves as a threshold: ARI is near zero
# below 2, exactly 1.000 above 4, and all three target bands fall between
# about 2.7 and 3.5. The interval is bounded to that transition so the
# bisection never wanders into either flat region, where it has no gradient
# to follow.
_BANDWIDTH_KNOB = CalibrationKnob(
    parameter="embed_param",
    low=2.0,
    high=4.0,
    increasing=True,
    multiplier=False,
)

CALIBRATION_KNOBS: dict[str, CalibrationKnob] = {
    "gaussians": _SCALE_KNOB,
    "t_dist": _SCALE_KNOB,
    "t_dist_noise": _SCALE_KNOB,
    "swiss_rolls": _SCALE_KNOB,
    "circles": _BANDWIDTH_KNOB,
    "moons": _BANDWIDTH_KNOB,
}


def knob_for(scenario_name: str) -> CalibrationKnob:
    """Return the calibration knob declared for a scenario.

    The two scaling scenarios have no knob: TARGET_ARI_BANDS is keyed on
    easy/medium/hard and they sweep n and p instead.
    """
    if scenario_name not in CALIBRATION_KNOBS:
        raise KeyError(
            f"No calibration knob for {scenario_name!r}. Declared scenarios "
            f"are {sorted(CALIBRATION_KNOBS)}."
        )
    return CALIBRATION_KNOBS[scenario_name]


def apply_knob(
    anchor: Mapping[str, Any], knob: CalibrationKnob, value: float
) -> dict[str, Any]:
    """Return the anchor with the knob set to value, leaving the rest alone."""
    out = dict(anchor)
    if not knob.multiplier:
        out[knob.parameter] = float(value)
        return out
    if knob.parameter not in anchor:
        raise KeyError(
            f"Anchor has no {knob.parameter!r} to scale; a multiplier knob needs "
            f"a published value to multiply. Anchor keys: {sorted(anchor)}."
        )
    published = anchor[knob.parameter]
    if np.isscalar(published):
        out[knob.parameter] = float(published) * float(value)
    else:
        out[knob.parameter] = [float(v) * float(value) for v in published]
    return out
```

Add `from collections.abc import Mapping` and `from dataclasses import dataclass, replace` to the imports.

- [ ] **Step 4: Report the spread alongside the mean**

Replace `mean_oracle_ari` with a stats function that it delegates to, so the existing tests that call `mean_oracle_ari` keep working:

```python
def oracle_ari_stats(
    scenario: Scenario,
    axis_label: str,
    overrides: dict[str, Any],
    *,
    n_seeds: int = 20,
    random_state: int = PUBLISHED_RANDOM_STATE,
) -> tuple[float, float]:
    """Mean and standard deviation of the oracle's ARI over the calibration seeds.

    Calibration uses the same seed derivation as the benchmark, so the
    calibration and benchmark datasets coincide, as the manuscript states.
    That claim holds only at random_state=PUBLISHED_RANDOM_STATE (42), which
    is why that is the default here rather than 0.

    The spread is returned as well as the mean because the RFF bandwidth's
    transition band is wide in spread even where the mean is on target --
    sd 0.10 to 0.23, individual datasets from 0.39 to 1.000 at one setting --
    so an anchor that reports only its mean hides how much it varies.
    """
    axis_idx = list(scenario.axis.labels).index(axis_label)
    axis_value = scenario.axis.values[axis_idx]

    anchors = {k: dict(v) for k, v in scenario.anchors.items()}
    anchors[axis_label].update(overrides)
    probe = replace(scenario, anchors=anchors)

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
    values = np.asarray(aris, dtype=float)
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean()), sd


def mean_oracle_ari(
    scenario: Scenario,
    axis_label: str,
    overrides: dict[str, Any],
    *,
    n_seeds: int = 20,
    random_state: int = PUBLISHED_RANDOM_STATE,
) -> float:
    """Mean ARI of the oracle estimator at k_star over the calibration seeds."""
    mean, _ = oracle_ari_stats(
        scenario, axis_label, overrides, n_seeds=n_seeds, random_state=random_state
    )
    return mean
```

- [ ] **Step 5: Make the bisection direction-aware**

Replace `calibrate_anchor` entirely:

```python
def calibrate_anchor(
    scenario: Scenario,
    axis_label: str,
    *,
    target: tuple[float, float],
    knob: CalibrationKnob | None = None,
    n_seeds: int = 20,
    random_state: int = PUBLISHED_RANDOM_STATE,
    max_iter: int = 25,
) -> dict[str, Any]:
    """Bisect the scenario's knob until mean oracle ARI lands in the band.

    knob defaults to the one CALIBRATION_KNOBS declares for this scenario.
    Passing one explicitly is for tests and for probing an alternative.

    The bisection reads its direction off the knob. Assuming ARI falls as
    the parameter grows, which the first implementation did, walks away from
    the band on an increasing knob rather than toward it.
    """
    low_target, high_target = target
    if not low_target < high_target:
        raise ValueError(f"Invalid target band {target!r}: low must be below high.")

    knob = knob_for(scenario.name) if knob is None else knob
    midpoint_target = 0.5 * (low_target + high_target)
    low, high = knob.low, knob.high

    published = scenario.anchors[axis_label]
    best_value = 0.5 * (low + high)
    best_anchor = apply_knob(published, knob, best_value)
    best_ari = float("nan")
    best_sd = float("nan")

    for _ in range(max_iter):
        value = 0.5 * (low + high)
        anchor = apply_knob(published, knob, value)
        ari, sd = oracle_ari_stats(
            scenario,
            axis_label,
            anchor,
            n_seeds=n_seeds,
            random_state=random_state,
        )
        best_value, best_anchor, best_ari, best_sd = value, anchor, ari, sd

        if low_target <= ari <= high_target:
            break
        # Move toward the band. On a decreasing knob a too-high ARI means
        # the parameter must grow; on an increasing one it means the
        # opposite.
        too_high = ari > midpoint_target
        if too_high != knob.increasing:
            low = value
        else:
            high = value

    return {
        "anchor": best_anchor,
        "knob_value": float(best_value),
        "achieved_ari": best_ari,
        "achieved_sd": best_sd,
        "target": target,
        "in_band": bool(low_target <= best_ari <= high_target),
    }
```

- [ ] **Step 6: Point calibrate_scenario at the knob**

`calibrate_scenario` already loops over the axis labels and calls `calibrate_anchor`. It needs no change now that `knob` defaults from the registry, but update its docstring:

```python
    """Calibrate every anchor of a scenario against its target band.

    Uses the knob CALIBRATION_KNOBS declares for the scenario. Run this with
    the scenario's final estimator and the fixed eigensolver in place;
    calibrating against a solver that will not be used calibrates the wrong
    thing.
    """
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_calibrate.py -v`

Expected: PASS.

- [ ] **Step 8: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_calibrate.py
.venv/bin/ruff check src/
git add src/benchmarks/_calibrate.py tests/benchmarks/test_calibrate.py
git commit -m "feat(benchmarks): per-scenario calibration knobs with a declared direction

CALIBRATION_SPACE's single global cluster_scale entry cannot calibrate
circles or moons: measured with the fixed solver, multiplying cluster_scale
by 3 leaves their oracle ARI at 1.000 +/- 0.000. embed_param is their knob,
it runs the opposite way, and the bisection hardcoded the decreasing
assumption.

A multiplier knob now scales the published cluster_scale vector rather than
replacing it with a scalar, so the unequal per-cluster scales survive the
search. calibrate_anchor also reports the spread alongside the mean.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: Regenerate and adopt the calibrated anchors

**Files:**
- Modify: `src/benchmarks/_registry.py` (`CALIBRATED_ANCHORS`, `ACTIVE_ANCHORS`, `ACTIVE_ANCHOR_SET_NAME`)
- Create: `docs/superpowers/notes/2026-09-22-calibration-comparison.md`
- Test: `tests/benchmarks/test_registry.py`

**Interfaces:**
- Consumes: Task 4's `calibrate_scenario`, Tasks 1 and 3.
- Produces: `ACTIVE_ANCHORS is CALIBRATED_ANCHORS`, `ACTIVE_ANCHOR_SET_NAME == "CALIBRATED_ANCHORS"`. Every anchor's oracle ARI is inside its band.

- [ ] **Step 1: Run the calibration**

This takes roughly 20-40 minutes: six scenarios times three anchors times up to 25 bisection steps times 20 datasets. The four scale-driven scenarios start near their band so they converge in a few steps; circles and moons take longer.

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
.venv/bin/python -u - <<'PY' | tee /tmp/carve-calibration.txt
import json, warnings
warnings.filterwarnings("ignore")
from benchmarks._calibrate import calibrate_scenario
from benchmarks._registry import DIFFICULTY_AXIS, SCENARIOS

out = {}
for name, sc in SCENARIOS.items():
    if sc.axis.name != DIFFICULTY_AXIS.name:
        continue
    print(f"--- {name} ({sc.estimator.name})", flush=True)
    result = calibrate_scenario(sc)
    out[name] = result
    for label, r in result.items():
        flag = "" if r["in_band"] else "   OUT OF BAND"
        print(
            f"   {label:7s} {r['anchor'].get('cluster_scale', r['anchor'].get('embed_param'))!r:>44}"
            f"  ARI {r['achieved_ari']:.3f} +/- {r['achieved_sd']:.3f}{flag}",
            flush=True,
        )
print(json.dumps(out, indent=2, default=str))
PY
```

- [ ] **Step 2: Check every anchor landed in band**

Read the output. Every one of the eighteen anchors must report `in_band`. If circles or moons do not, the transition band is narrower than the probe suggested — stop and report it rather than widening `TARGET_ARI_BANDS`. The spec's open items list this as a decision, not a plan step.

- [ ] **Step 3: Write the comparison note**

Create `docs/superpowers/notes/2026-09-22-calibration-comparison.md` with the published anchor, the calibrated anchor, the achieved ARI and its spread, for all eighteen cells. `_calibrate.py`'s module docstring requires this comparison before adoption. Follow the shape of `docs/superpowers/notes/2026-09-01-calibration-comparison.md`, which is the previous one.

- [ ] **Step 4: Write the failing tests**

In `tests/benchmarks/test_registry.py`, replace `test_active_anchors_default_to_the_published_set`:

```python
    def test_active_anchors_are_the_calibrated_set(self):
        """Every difficulty anchor is regenerated through _calibrate.py.

        The published anchors missed their own documented bands on circles
        and moons, and the eigensolver fix moved the three RFF scenarios
        further. A half-published, half-calibrated set would leave "easy"
        meaning a different ARI on different scenarios, which is the
        inconsistency this removes.
        """
        assert ACTIVE_ANCHORS is CALIBRATED_ANCHORS
        assert ACTIVE_ANCHOR_SET_NAME == "CALIBRATED_ANCHORS"

    def test_published_anchors_are_kept_for_comparison(self):
        """PUBLISHED_ANCHORS stays in the file so the calibration can be
        compared against what the manuscript reported, and reverted to."""
        assert set(PUBLISHED_ANCHORS) == set(SCENARIOS)
        assert PUBLISHED_ANCHORS is not CALIBRATED_ANCHORS

    @pytest.mark.parametrize("name", DIFFICULTY_SCENARIOS)
    def test_calibrated_anchors_cover_every_difficulty_label(self, name):
        assert set(CALIBRATED_ANCHORS[name]) == set(DIFFICULTY_AXIS.labels)

    @pytest.mark.parametrize("name", SCALING_SCENARIOS)
    def test_the_scaling_scenarios_keep_their_published_anchors(self, name):
        """TARGET_ARI_BANDS is keyed on easy/medium/hard, so the two scaling
        scenarios have no band to calibrate against and are carried over."""
        assert CALIBRATED_ANCHORS[name] == PUBLISHED_ANCHORS[name]
```

Add `CALIBRATED_ANCHORS` to the imports at the top of the file.

- [ ] **Step 5: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_registry.py -k anchors -v`

Expected: collection error — `cannot import name 'CALIBRATED_ANCHORS'`.

- [ ] **Step 6: Write the anchors into the registry**

In `src/benchmarks/_registry.py`, after `PUBLISHED_ANCHORS`, add `CALIBRATED_ANCHORS` with the values from Step 1. Carry `gaussians_dimensionality` and `gaussians_samples` over from `PUBLISHED_ANCHORS` unchanged. Each anchor keeps every published key except the calibrated one. Record the achieved ARI in a comment per scenario, in this shape:

```python
# Regenerated 2026-09-22 by _calibrate.calibrate_scenario, against the
# documented bands in TARGET_ARI_BANDS, with the shift-invert eigensolver
# and each scenario's final estimator in place. The achieved mean oracle ARI
# and its spread over the 20 calibration datasets are recorded per anchor;
# the full comparison against PUBLISHED_ANCHORS is in
# docs/superpowers/notes/2026-09-22-calibration-comparison.md.
#
# Two knobs, because no single parameter controls difficulty on every
# scenario -- see _calibrate.CALIBRATION_KNOBS. The four scale-driven
# scenarios carry their published cluster_scale vector times a multiplier,
# so the unequal per-cluster scales survive; circles and moons carry an
# absolute embed_param.
CALIBRATED_ANCHORS: dict[str, dict[str, dict[str, Any]]] = {
    # Knob: cluster_scale multiplier. The published vector was [4.0] * 5 at
    # easy, [4.5] * 5 at medium and [4.6] * 5 at hard.
    "gaussians": {
        # multiplier 0.984, ARI 0.921 +/- 0.061
        "easy": {
            "cluster_scale": [3.936] * 5,
            "cluster_size_dirichlet_alpha": 0.9,
        },
        # multiplier 1.012, ARI 0.857 +/- 0.088
        "medium": {
            "cluster_scale": [4.554] * 5,
            "cluster_size_dirichlet_alpha": 0.5,
        },
        # multiplier 1.031, ARI 0.742 +/- 0.134
        "hard": {
            "cluster_scale": [4.743] * 5,
            "cluster_size_dirichlet_alpha": 0.1,
        },
    },
    # Knob: embed_param, absolute. The published values were 12.0, 7.3 and
    # 6.0, all past the 4.0 saturation point.
    "circles": {
        # embed_param 3.41, ARI 0.938 +/- 0.131
        "easy": {
            "cluster_scale": [4.08, 4.08, 3.0, 3.0, 3.0],
            "cluster_size_dirichlet_alpha": 0.9,
            "corr_strength": 0.1,
            "embed_param": 3.41,
        },
        # embed_param 3.06, ARI 0.842 +/- 0.171
        "medium": {...},
        # embed_param 2.88, ARI 0.759 +/- 0.196
        "hard": {...},
    },
    # ... the remaining four difficulty scenarios in the same shape ...
    # The two scaling scenarios are carried over: TARGET_ARI_BANDS is keyed
    # on easy/medium/hard and they sweep n and p.
    "gaussians_dimensionality": PUBLISHED_ANCHORS["gaussians_dimensionality"],
    "gaussians_samples": PUBLISHED_ANCHORS["gaussians_samples"],
}
```

The numbers above are the shape, not the answer — Step 1's output supplies the real multipliers, bandwidths and achieved ARIs. Every anchor keeps every published key it had; only the calibrated one changes. Do not leave a literal `{...}` in the file.

Then flip the active set:

```python
ACTIVE_ANCHORS: dict[str, dict[str, dict[str, Any]]] = CALIBRATED_ANCHORS
ACTIVE_ANCHOR_SET_NAME: str = "CALIBRATED_ANCHORS"
```

- [ ] **Step 7: Verify the anchors reproduce their bands**

```bash
.venv/bin/python -u - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from benchmarks._calibrate import TARGET_ARI_BANDS, oracle_ari_stats
from benchmarks._registry import DIFFICULTY_AXIS, SCENARIOS

for name, sc in SCENARIOS.items():
    if sc.axis.name != DIFFICULTY_AXIS.name:
        continue
    for label in sc.axis.labels:
        mean, sd = oracle_ari_stats(sc, label, {})
        lo, hi = TARGET_ARI_BANDS[label]
        ok = "ok " if lo <= mean <= hi else "OUT"
        print(f"{ok} {name:14s} {label:7s} {mean:.3f} +/- {sd:.3f}  band {lo}-{hi}")
PY
```

Expected: `ok` on all eighteen lines.

- [ ] **Step 8: Run the tests and commit**

Run: `.venv/bin/python -m pytest tests/benchmarks -q`

Expected: PASS.

```bash
.venv/bin/ruff format src/benchmarks/_registry.py
.venv/bin/ruff check src/
git add src/benchmarks/_registry.py tests/benchmarks/test_registry.py docs/superpowers/notes/2026-09-22-calibration-comparison.md
git commit -m "feat(benchmarks): adopt the calibrated difficulty anchors

Regenerated through _calibrate.calibrate_scenario against the bands the
manuscript documents, with the fixed eigensolver and each scenario's final
estimator in place. All eighteen anchors land in band; the published set
missed it on circles and moons and saturated at 1.000 on both once the
solver was correct.

PUBLISHED_ANCHORS stays for comparison and revert. Every run manifest
already records ACTIVE_ANCHOR_SET_NAME, so no artifact is ambiguous about
which set produced it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

Tasks 6 through 9 are independent of the calibration and of each other, so they can be taken in any order once Task 1 is in.

## Task 6: Manifest before the pool, with a status

**Files:**
- Modify: `src/benchmarks/_run.py` (`run_scenario`)
- Modify: `src/benchmarks/_artifacts.py` (`build_manifest`)
- Test: `tests/benchmarks/test_run.py`, `tests/benchmarks/test_artifacts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `manifest.json` exists with `status` in `{"running", "complete"}` from before the pool starts. Task 8's resolver reads it.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_run.py`, in the class covering `run_scenario`:

```python
    def test_manifest_exists_before_the_pool_finishes(self, tiny_scenario, tmp_path):
        """An interrupted scenario must still leave a readable manifest.

        Run directories are located by their manifest, so a scenario killed
        mid-pool otherwise leaves checkpoints the notebook cannot open. The
        ablation runner was fixed for this in e720073; the scenario runner
        was not.
        """
        seen = {}

        def _explode(*args, **kwargs):
            rd = tmp_path / tiny_scenario.name
            found = list(rd.glob("*/manifest.json"))
            seen["manifest"] = json.loads(found[0].read_text()) if found else None
            raise KeyboardInterrupt

        with mock.patch("benchmarks._run.Parallel", side_effect=_explode):
            with pytest.raises(KeyboardInterrupt):
                run_scenario(tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=1)

        assert seen["manifest"] is not None
        assert seen["manifest"]["status"] == "running"

    def test_manifest_is_complete_after_the_pool(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=1)
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["status"] == "complete"

    def test_a_resumed_run_keeps_its_run_id_across_the_running_manifest(
        self, tiny_scenario, tmp_path
    ):
        """The running manifest carries the same run_id the complete one
        will, so a resume that reads either gets the same provenance."""
        first = run_scenario(tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=1)
        run_id = json.loads((first / "manifest.json").read_text())["run_id"]
        second = run_scenario(tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=2)
        assert second != first or (
            json.loads((second / "manifest.json").read_text())["run_id"] == run_id
        )
```

Add `import json`, `from unittest import mock` and `import pytest` to the file's imports if they are not already there.

In `tests/benchmarks/test_artifacts.py`:

```python
def test_build_manifest_records_a_status(tiny_scenario):
    manifest = build_manifest(
        tiny_scenario,
        run_id="abc123",
        config_hash="def456",
        n_seeds=1,
        n_resamples=2,
        n_jobs=1,
        random_state=42,
        wall_clock_s=1.0,
        status="running",
    )
    assert manifest.status == "running"


def test_build_manifest_rejects_an_unknown_status(tiny_scenario):
    with pytest.raises(ValueError, match="status"):
        build_manifest(
            tiny_scenario,
            run_id="abc123",
            config_hash="def456",
            n_seeds=1,
            n_resamples=2,
            n_jobs=1,
            random_state=42,
            wall_clock_s=1.0,
            status="halfway",
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_run.py tests/benchmarks/test_artifacts.py -k "manifest" -v`

Expected: FAIL. `build_manifest` has no `status` keyword; `Manifest` has no `status` field; the running manifest does not exist.

- [ ] **Step 3: Add status to the Manifest dataclass**

In `src/benchmarks/_types.py`, add the field to `Manifest`, after `config_hash`:

```python
    status: str
```

and extend its docstring:

```python
    status is "running" while the pool is in flight and "complete" once every
    cell is done. It is written before the pool starts so an interrupted run
    still has a manifest: run directories are located by it, and a directory
    with checkpoints and no manifest cannot be opened at all.
```

- [ ] **Step 4: Thread status through build_manifest**

In `src/benchmarks/_artifacts.py`:

```python
RUN_STATUSES: frozenset[str] = frozenset({"running", "complete"})


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
    status: str = "complete",
    timing_fits: bool = False,
) -> Manifest:
```

Inside, before constructing the `Manifest`:

```python
    if status not in RUN_STATUSES:
        raise ValueError(
            f"Unknown run status {status!r}. Valid values are {sorted(RUN_STATUSES)}."
        )
```

and pass `status=status` into the `Manifest(...)` call, right after `config_hash=config_hash`.

- [ ] **Step 5: Write the manifest twice in run_scenario**

In `src/benchmarks/_run.py`, in `run_scenario`, replace the block from `run_id = previous_manifest[...]` through the final `write_manifest(...)` with:

```python
    run_id = previous_manifest["run_id"] if previous_manifest else uuid.uuid4().hex[:12]
    carried = (
        float(previous_manifest["wall_clock_s"]) if previous_manifest is not None else 0.0
    )
    _, thread_cap = thread_cap_for(n_jobs)

    def _manifest(status: str, wall_clock_s: float):
        return build_manifest(
            scenario,
            run_id=run_id,
            config_hash=cfg_hash,
            n_seeds=n_seeds,
            n_resamples=n_resamples,
            n_jobs=n_jobs,
            random_state=random_state,
            wall_clock_s=wall_clock_s,
            status=status,
            timing_fits=timing_fits,
        )

    # Before the pool, not after. A scenario killed mid-pool otherwise leaves
    # a directory of checkpoints with no manifest, which no reader can open.
    write_manifest(rd, _manifest("running", carried))
    started = time.perf_counter()

    def _one(axis_idx, axis_value, axis_label, seed):
        rows, runtime_row = run_cell(
            scenario,
            axis_idx=axis_idx,
            axis_value=axis_value,
            axis_label=axis_label,
            seed=seed,
            run_id=run_id,
            random_state=random_state,
            n_resamples=n_resamples,
            timing_fits=timing_fits,
            thread_cap=thread_cap,
        )
        write_checkpoint(rd, axis_label, seed, rows)
        write_runtime_checkpoint(rd, axis_label, seed, [runtime_row])

    if pending:
        Parallel(n_jobs=n_jobs)(
            delayed(_one)(*cell)
            for cell in tqdm(pending, desc=scenario.name, leave=False)
        )

    elapsed = carried + (time.perf_counter() - started)
    write_manifest(rd, _manifest("complete", elapsed))
    return rd
```

Update the `run_scenario` docstring's provenance paragraph to say the manifest is written twice, mirroring the wording already in `_ablation.run_ablation`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_run.py tests/benchmarks/test_artifacts.py -v`

Expected: PASS.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_run.py src/benchmarks/_artifacts.py src/benchmarks/_types.py
.venv/bin/ruff check src/
git add src/benchmarks/ tests/benchmarks/
git commit -m "fix(benchmarks): run_scenario writes its manifest before the pool

An interrupted scenario left a directory of checkpoints with no manifest,
which the run-directory resolver cannot open. Mirrors the fix the ablation
runner got in e720073: status 'running' before the pool with the wall clock
carried so far, status 'complete' after.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: Refuse to resume across code versions

**Files:**
- Modify: `src/benchmarks/_run.py` (`run_scenario`)
- Modify: `src/benchmarks/run.py` (`--allow-code-change`)
- Test: `tests/benchmarks/test_run.py`, `tests/benchmarks/test_cli.py`

**Interfaces:**
- Consumes: Task 6's manifest.
- Produces: `run_scenario(..., allow_code_change: bool = False)`; CLI flag `--allow-code-change`.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_run.py`:

```python
    def test_resuming_onto_different_code_raises(self, tiny_scenario, tmp_path):
        """read_run concatenates whatever checkpoints it finds.

        A run directory is content-addressed on configuration only, so
        re-running the same config on newer code resumes into the old
        directory and mixes two code versions' rows in one frame with
        nothing marking it. git_sha was already recorded and never checked;
        this is what left the ablation dev run unreadable.
        """
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=1)
        manifest = json.loads((rd / "manifest.json").read_text())
        manifest["git_sha"] = "0" * 40
        (rd / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(RuntimeError, match="different code version"):
            run_scenario(tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=1)

    def test_the_override_allows_the_resume(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=1)
        manifest = json.loads((rd / "manifest.json").read_text())
        manifest["git_sha"] = "0" * 40
        (rd / "manifest.json").write_text(json.dumps(manifest))

        again = run_scenario(
            tiny_scenario,
            root=tmp_path,
            n_resamples=2,
            n_seeds=1,
            allow_code_change=True,
        )
        assert again == rd

    def test_no_resume_does_not_check_the_code_version(self, tiny_scenario, tmp_path):
        """resume=False recomputes every cell, so there is nothing to mix."""
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=1)
        manifest = json.loads((rd / "manifest.json").read_text())
        manifest["git_sha"] = "0" * 40
        (rd / "manifest.json").write_text(json.dumps(manifest))

        again = run_scenario(
            tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=1, resume=False
        )
        assert again == rd
```

In `tests/benchmarks/test_cli.py`:

```python
def test_allow_code_change_reaches_run_scenario(monkeypatch):
    captured = {}

    def _fake(scenario, **kwargs):
        captured.update(kwargs)
        return Path("results/runs/gaussians/abc")

    monkeypatch.setattr("benchmarks.run.run_scenario", _fake)
    main(["--scenario", "gaussians", "--allow-code-change"])
    assert captured["allow_code_change"] is True


def test_the_code_version_is_checked_by_default(monkeypatch):
    captured = {}

    def _fake(scenario, **kwargs):
        captured.update(kwargs)
        return Path("results/runs/gaussians/abc")

    monkeypatch.setattr("benchmarks.run.run_scenario", _fake)
    main(["--scenario", "gaussians"])
    assert captured["allow_code_change"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_run.py tests/benchmarks/test_cli.py -k "code_change or code_version" -v`

Expected: FAIL. `run_scenario` has no `allow_code_change` parameter; no `RuntimeError` is raised.

- [ ] **Step 3: Add the guard**

In `src/benchmarks/_run.py`, add the parameter to `run_scenario`'s signature after `resume`:

```python
    allow_code_change: bool = False,
```

and, immediately after the `previous_manifest` block that reads the manifest:

```python
    if previous_manifest is not None and not allow_code_change:
        recorded = previous_manifest.get("git_sha")
        current = provenance()["git_sha"]
        if recorded and current and recorded != current:
            raise RuntimeError(
                f"{rd} was produced at git_sha {recorded}, and this process is at "
                f"{current}. Resuming would concatenate two code versions' "
                "checkpoints into one frame, because a run directory is "
                "content-addressed on its configuration and not on the code that "
                "produced it, and read_run reads whatever it finds. Re-run with "
                "--no-resume to recompute the directory, or pass "
                "--allow-code-change if the change provably cannot affect a "
                "recorded value."
            )
```

Import `provenance` from `._artifacts` at the top of `_run.py`.

Document the parameter in the docstring:

```python
    allow_code_change permits a resume into a directory whose manifest
    records a different git_sha. It is off by default: the checkpoints of two
    code versions are indistinguishable once read_run concatenates them.
```

- [ ] **Step 4: Add the CLI flag**

In `src/benchmarks/run.py`, beside `--no-resume`:

```python
    parser.add_argument(
        "--allow-code-change",
        action="store_true",
        help="Resume into a run directory recorded at a different git_sha. Off by "
        "default, because resuming across code versions mixes their checkpoints.",
    )
```

and pass it through in the `run_scenario` call:

```python
            resume=not args.no_resume,
            allow_code_change=args.allow_code_change,
            verbose=args.verbose,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_run.py tests/benchmarks/test_cli.py -v`

Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_run.py src/benchmarks/run.py
.venv/bin/ruff check src/
git add src/benchmarks/ tests/benchmarks/
git commit -m "fix(benchmarks): refuse to resume a run onto different code

git_sha reached every manifest and nothing read it, so re-running a config
on newer code resumed into the old directory and read_run concatenated two
code versions' checkpoints. --allow-code-change overrides; --no-resume
sidesteps it by recomputing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: One run-directory resolver

**Files:**
- Modify: `src/benchmarks/_artifacts.py` (add `widest_run`)
- Modify: `src/benchmarks/run.py` (`--tables` branch)
- Test: `tests/benchmarks/test_artifacts.py`, `tests/benchmarks/test_tables_cli.py`

**Interfaces:**
- Consumes: Task 6's `status`.
- Produces: `widest_run(root: Path, scenario_name: str, *, require_complete: bool = True) -> Path`. Task 16's notebook imports it.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_artifacts.py`:

```python
def _fake_run(root, scenario, cfg, *, n_seeds, n_resamples, status="complete"):
    rd = root / scenario / cfg
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": cfg,
                "status": status,
                "scenario": scenario,
                "n_seeds": n_seeds,
                "n_resamples": n_resamples,
            }
        )
    )
    return rd


class TestWidestRun:
    def test_picks_the_widest_sweep_not_the_lexically_last(self, tmp_path):
        """The CLI took the lexically last directory and the notebook took
        the widest sweep, so S4 regenerated from the CLI was the 5-dataset
        run while the notebook showed the 20-dataset one."""
        _fake_run(tmp_path, "t_dist_noise", "zzz", n_seeds=5, n_resamples=20)
        wide = _fake_run(tmp_path, "t_dist_noise", "aaa", n_seeds=20, n_resamples=100)
        assert widest_run(tmp_path, "t_dist_noise") == wide

    def test_breaks_a_seed_tie_on_the_resample_count(self, tmp_path):
        _fake_run(tmp_path, "gaussians", "aaa", n_seeds=20, n_resamples=20)
        wide = _fake_run(tmp_path, "gaussians", "bbb", n_seeds=20, n_resamples=100)
        assert widest_run(tmp_path, "gaussians") == wide

    def test_raises_on_a_tie(self, tmp_path):
        _fake_run(tmp_path, "gaussians", "aaa", n_seeds=20, n_resamples=100)
        _fake_run(tmp_path, "gaussians", "bbb", n_seeds=20, n_resamples=100)
        with pytest.raises(RuntimeError, match="equally wide"):
            widest_run(tmp_path, "gaussians")

    def test_skips_an_incomplete_run(self, tmp_path):
        done = _fake_run(tmp_path, "gaussians", "aaa", n_seeds=5, n_resamples=20)
        _fake_run(
            tmp_path, "gaussians", "bbb", n_seeds=20, n_resamples=100, status="running"
        )
        assert widest_run(tmp_path, "gaussians") == done

    def test_can_be_asked_for_an_incomplete_run(self, tmp_path):
        _fake_run(tmp_path, "gaussians", "aaa", n_seeds=5, n_resamples=20)
        running = _fake_run(
            tmp_path, "gaussians", "bbb", n_seeds=20, n_resamples=100, status="running"
        )
        assert widest_run(tmp_path, "gaussians", require_complete=False) == running

    def test_names_the_command_when_there_is_no_run(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="benchmarks.run --scenario moons"):
            widest_run(tmp_path, "moons")

    def test_ignores_a_directory_with_no_manifest(self, tmp_path):
        done = _fake_run(tmp_path, "gaussians", "aaa", n_seeds=5, n_resamples=20)
        (tmp_path / "gaussians" / "orphan").mkdir()
        assert widest_run(tmp_path, "gaussians") == done
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_artifacts.py -k WidestRun -v`

Expected: collection error — `cannot import name 'widest_run'`.

- [ ] **Step 3: Write the resolver**

In `src/benchmarks/_artifacts.py`, after `run_dir`:

```python
def widest_run(
    root: Path, scenario_name: str, *, require_complete: bool = True
) -> Path:
    """The run directory holding the widest sweep for a scenario.

    Run directories are named by configuration hash, so sorting them
    lexically picks an arbitrary one when a scenario has been run at more
    than one scale. Selecting on the manifest's sweep size instead means a
    reduced exploratory run never silently shadows a full one.

    This is the one resolver. The CLI's --tables path took the lexically
    last directory with a warning while the notebook took the widest sweep,
    so the two disagreed: S4 regenerated from the CLI was the 5-dataset run
    and the notebook's panel was the 20-dataset one, with nothing marking
    the difference.

    A tie raises rather than choosing. Two equally wide runs of one scenario
    differ in something the sweep size does not capture -- an anchor set, a
    random_state -- and picking either silently is how a mixture gets into a
    figure.

    require_complete skips runs whose manifest still says "running", which
    is what an interrupted run leaves behind. Pass False to inspect one.
    """
    candidates: list[tuple[tuple[int, int], Path]] = []
    for path in sorted((Path(root) / scenario_name).glob("*/")):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if require_complete and manifest.get("status") != "complete":
            continue
        candidates.append(
            ((int(manifest["n_seeds"]), int(manifest["n_resamples"])), path)
        )

    if not candidates:
        raise FileNotFoundError(
            f"No {'complete ' if require_complete else ''}run for "
            f"{scenario_name!r} under {root}. Produce one with: "
            f"python -m benchmarks.run --scenario {scenario_name}"
        )

    widest = max(size for size, _ in candidates)
    matching = [path for size, path in candidates if size == widest]
    if len(matching) > 1:
        raise RuntimeError(
            f"{scenario_name}: {len(matching)} equally wide runs at "
            f"{widest[0]} datasets x B={widest[1]} "
            f"({', '.join(p.name for p in matching)}). They differ in something "
            "the sweep size does not capture; read their manifests and remove "
            "or archive the one you do not want."
        )
    return matching[0]
```

- [ ] **Step 4: Point the CLI at it**

In `src/benchmarks/run.py`, replace the `--tables` frame-gathering block:

```python
    if args.tables:
        from ._artifacts import read_run, widest_run
        from .tables import write_all_tables

        frames = {}
        for name in sorted(SCENARIOS):
            try:
                frames[name] = read_run(widest_run(Path(args.root), name))
            except FileNotFoundError:
                continue
        paths = write_all_tables(frames, Path(args.tables))
        for path in paths:
            print(path)
        return 0
```

Remove the now-unused `import warnings` from `run.py` if nothing else uses it.

- [ ] **Step 5: Update the CLI table test**

`tests/benchmarks/test_tables_cli.py` asserts the old lexical-last warning. Replace whichever test does with:

```python
def test_tables_reads_the_widest_run_not_the_lexically_last(tmp_path, monkeypatch):
    """The CLI and the notebook must agree about which run they read."""
    root = tmp_path / "runs"
    _fake_run(root, "gaussians", "zzz", n_seeds=5, n_resamples=20)
    _fake_run(root, "gaussians", "aaa", n_seeds=20, n_resamples=100)

    seen = {}

    def _fake_read_run(rd):
        seen["path"] = rd
        return pd.DataFrame(columns=list(SCHEMA))

    monkeypatch.setattr("benchmarks._artifacts.read_run", _fake_read_run)
    main(["--tables", str(tmp_path / "out"), "--root", str(root)])
    assert seen["path"].name == "aaa"
```

Reuse the `_fake_run` helper from Step 1 by moving it into `tests/benchmarks/_helpers.py` and importing it in both files.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks -q`

Expected: PASS.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_artifacts.py src/benchmarks/run.py
.venv/bin/ruff check src/
git add src/benchmarks/ tests/benchmarks/
git commit -m "fix(benchmarks): one run-directory resolver for the CLI and the notebook

--tables took the lexically last directory and the notebook took the widest
sweep, so S4 from the CLI was the 5-dataset run while the notebook drew the
20-dataset one. widest_run is now the single answer: widest sweep, skipping
incomplete runs, raising on a tie rather than choosing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: n_jobs resolved per scenario class

**Files:**
- Modify: `src/benchmarks/_run.py` (add `scenario_n_jobs`, call it in `run_scenario`)
- Modify: `src/benchmarks/run.py` (`_resolve_n_jobs`)
- Test: `tests/benchmarks/test_run.py`, `tests/benchmarks/test_cli.py`

**Interfaces:**
- Consumes: `TIMED_SCENARIOS` from `_registry`, `thread_cap_for` from `_run`.
- Produces: `scenario_n_jobs(scenario: Scenario, requested: int | None) -> int`.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_run.py`:

```python
class TestScenarioNJobs:
    def test_a_timed_scenario_is_forced_serial(self):
        """The scaling scenarios time their fits. Concurrent workers change
        those timings, so the runtime curves they feed are only meaningful
        at one worker."""
        for name in ("gaussians_samples", "gaussians_dimensionality"):
            assert scenario_n_jobs(SCENARIOS[name], None) == 1

    def test_a_difficulty_scenario_takes_every_worker(self):
        """--all previously defaulted to 1 for every scenario, which turns
        a seven-hour job into a three-day one."""
        assert scenario_n_jobs(SCENARIOS["gaussians"], None) == -1

    def test_an_explicit_request_overrides_both(self):
        assert scenario_n_jobs(SCENARIOS["gaussians"], 4) == 4
        assert scenario_n_jobs(SCENARIOS["gaussians_samples"], 4) == 4

    def test_run_scenario_resolves_it(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=1)
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["n_jobs"] == -1
```

In `tests/benchmarks/test_cli.py`, replace whichever test asserts the old `--n-jobs` default of 1 for scenarios:

```python
def test_scenarios_default_to_per_scenario_resolution(monkeypatch):
    """The CLI hands None down and the runner decides, because one flag
    cannot be right for both the difficulty scenarios and the timed pair."""
    captured = {}
    monkeypatch.setattr(
        "benchmarks.run.run_scenario",
        lambda scenario, **kwargs: captured.update(kwargs)
        or Path("results/runs/x/y"),
    )
    main(["--scenario", "gaussians"])
    assert captured["n_jobs"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_run.py tests/benchmarks/test_cli.py -k "n_jobs or NJobs" -v`

Expected: collection error — `cannot import name 'scenario_n_jobs'`.

- [ ] **Step 3: Add the resolver**

In `src/benchmarks/_run.py`, after `thread_cap_for`:

```python
def scenario_n_jobs(scenario: Scenario, requested: int | None) -> int:
    """Resolve the worker count for one scenario.

    One flag cannot be right for both halves of the suite. The six
    difficulty scenarios want every core; the two scaling scenarios time
    their fits, and a pool of concurrent workers changes exactly the number
    they exist to measure. Deciding here rather than at the CLI is what
    makes --all correct and fast at the same time: it previously defaulted
    to one worker for everything, which is right for two scenarios and turns
    the other six from seven hours into three days.

    An explicit request always wins, so a caller can still force a serial
    difficulty run or a parallel timing run when they know why.
    """
    from ._registry import TIMED_SCENARIOS

    if requested is not None:
        return int(requested)
    return 1 if scenario.name in TIMED_SCENARIOS else -1
```

In `run_scenario`, change the signature's default to `n_jobs: int | None = None` and resolve it as the first statement of the body:

```python
    n_jobs = scenario_n_jobs(scenario, n_jobs)
```

- [ ] **Step 4: Stop the CLI deciding for scenarios**

In `src/benchmarks/run.py`, replace `_resolve_n_jobs`:

```python
def _resolve_n_jobs(args: argparse.Namespace) -> int | None:
    """An ablation runs on every core; a scenario's worker count is the
    runner's decision, because it depends on whether the scenario times its
    fits. None means "you decide"."""
    if args.n_jobs is not None:
        return int(args.n_jobs)
    return -1 if args.ablation else None
```

The ablation branch calls `_resolve_n_jobs(args)` and gets `-1`, unchanged. The scenario branch now passes `None` through to `run_scenario`.

Update the `--n-jobs` help text:

```python
        help="Workers for the outer loop. Defaults to -1 for an ablation and, "
        "for a scenario, to -1 unless the scenario times its fits, in which "
        "case 1.",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks -q`

Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_run.py src/benchmarks/run.py
.venv/bin/ruff check src/
git add src/benchmarks/ tests/benchmarks/
git commit -m "feat(benchmarks): resolve n_jobs per scenario class

One --n-jobs cannot serve both the six difficulty scenarios, which want
every core, and the two scaling scenarios, whose timings are only
meaningful at one worker. The runner decides; an explicit --n-jobs still
overrides. --all is now correct and fast at once, rather than correct and
three days long.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

Tasks 1 through 9 complete the compute half. Nothing below touches the runner.

## Task 10: Re-step the palette and give each comparator its own dashes

**Files:**
- Modify: `src/benchmarks/_theme.py` (`METRIC_COLORS`, `metric_linestyle`, add `COMPARATOR_DASHES` / `metric_dashes` / `SEQUENTIAL_CMAP_NAME`)
- Test: `tests/benchmarks/test_theme.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - re-stepped `METRIC_COLORS` entries for `ari_generalizability*`, `davies_bouldin`, `calinski_harabasz`, `gap`
  - `COMPARATOR_DASHES: dict[str, tuple[float, ...]]`
  - `metric_dashes(metric: str) -> tuple[float, ...] | None` — `None` means solid
  - `metric_linestyle(metric: str) -> str | tuple[int, tuple[float, ...]]`
  - `SEQUENTIAL_CMAP_NAME: str` and a registered single-hue colormap. Task 11 uses it.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_theme.py`, replace the four CVI color assertions at lines 99-102 and add a class:

```python
    def test_the_comparator_hues_are_the_re_stepped_values(self):
        """Silhouette keeps its published pink. The other three move.

        The published set failed the normal-vision separation floor on the
        Silhouette/Davies-Bouldin pair (dE 13.6 against a floor of 15) and
        the 3:1 contrast floor on Gap. Each replacement stays inside the
        comparator family -- pink, purple, red, orange -- so the figure
        remains recognizable.
        """
        assert METRIC_COLORS["silhouette"] == "#E0457B"
        assert METRIC_COLORS["davies_bouldin"] == "#6A2C91"
        assert METRIC_COLORS["calinski_harabasz"] == "#B01B20"
        assert METRIC_COLORS["gap"] == "#C96A05"

    def test_carve_generalizability_clears_the_contrast_floor(self):
        """#56B4E9 sat at 2.25:1 against the chart surface, which made the
        palest line on the page a headline series. #0072B2 is Okabe-Ito's
        blue, so the series stays blue."""
        for metric in (
            "ari_generalizability",
            "ari_generalizability_1se",
            "ari_generalizability_quant",
        ):
            assert METRIC_COLORS[metric] == "#0072B2"

    def test_carve_stability_and_the_oracle_are_unchanged(self):
        for metric in (
            "ari_stability",
            "ari_stability_1se",
            "ari_stability_quant",
        ):
            assert METRIC_COLORS[metric] == "#009E73"
        assert METRIC_COLORS["baseline_oracle"] == "#595959"


class TestComparatorDashes:
    def test_every_comparator_has_its_own_pattern(self):
        """Four warm hues cannot all clear the CVD separation threshold
        pairwise, so hue does the work it can and dash carries the rest."""
        patterns = [COMPARATOR_DASHES[m] for m in CVI_METRICS]
        assert len(set(patterns)) == len(patterns)

    def test_carve_series_are_solid(self):
        for metric in ("ari_stability_1se", "ari_generalizability_1se"):
            assert metric_dashes(metric) is None
            assert metric_linestyle(metric) == "-"

    def test_a_comparator_linestyle_carries_its_dashes(self):
        style = metric_linestyle("silhouette")
        assert style == (0, COMPARATOR_DASHES["silhouette"])

    def test_an_unknown_metric_falls_back_to_a_dashed_default(self):
        assert metric_linestyle("not_a_metric") == (0, DEFAULT_DASHES)


class TestSequentialRamp:
    def test_the_ramp_is_registered_and_monotone_in_lightness(self):
        """P(k-hat = k) is a magnitude, so it takes one hue light to dark.
        A rainbow encodes ordering with hue, which reads as category."""
        import matplotlib as mpl

        cmap = mpl.colormaps[SEQUENTIAL_CMAP_NAME]
        samples = [cmap(v / 8) for v in range(9)]
        luminance = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b, _ in samples]
        assert luminance == sorted(luminance, reverse=True)
```

Add `COMPARATOR_DASHES`, `DEFAULT_DASHES`, `SEQUENTIAL_CMAP_NAME`, `metric_dashes` to the module's imports at the top of the test file, and `CVI_METRICS` from `benchmarks._registry`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_theme.py -v`

Expected: collection error — `cannot import name 'COMPARATOR_DASHES'`.

- [ ] **Step 3: Re-step the colors**

In `src/benchmarks/_theme.py`, update the three CARVE generalizability entries to `"#0072B2"`, and replace the four comparator entries and their comment:

```python
    # The four classical indices, in the published Fig 4's families: pink,
    # purple, red, orange. Three of the four moved from the literal
    # published values after the dataviz validator flagged two failures
    # against a light chart surface: Silhouette #E0457B and Davies-Bouldin
    # #A8389E were dE 13.6 apart in normal vision, below the floor of 15, so
    # full-color readers could not separate them either -- visible in the
    # published t-distributed panel; and Gap #F28522 sat at 2.51:1 contrast,
    # below the 3:1 floor. Each replacement is a deeper step of the same
    # hue, so the figure stays recognizable and the family assignment is
    # unchanged. Silhouette keeps its published value.
    #
    # The re-stepped set passes every check: lightness band, chroma floor,
    # CVD separation at worst adjacent dE 8.7 protan and 8.6 tritan,
    # normal-vision floor at worst adjacent dE 16.2, and contrast at or above
    # 3:1 for all seven series.
    #
    # That module assigned the four positionally, in whatever order a caller
    # listed its metrics, so the published figures disagree with each other
    # about which index is which color. Fig 4's assignment is the one adopted
    # here, so the case study panels change rather than Fig 4.
    "silhouette": "#E0457B",
    "davies_bouldin": "#6A2C91",
    "calinski_harabasz": "#B01B20",
    "gap": "#C96A05",
```

Leave `ari_average*`, the three consensus greys and `accuracy_generalizability` alone. They reach the tables only, never a figure this plan draws; the spec records them as a known deviation from the blue-and-green family rule.

- [ ] **Step 4: Add the dash patterns**

Below `metric_linewidth`, replacing the body of `metric_linestyle`:

```python
# Hue alone cannot separate four comparator series under color-vision
# deficiency -- four warm hues never clear the threshold pairwise, whatever
# steps they take -- so each carries a distinct dash pattern as well. The
# patterns are in points, alternating on and off, and are ordered so no two
# read alike at the 1.2pt reference line width.
DEFAULT_DASHES: tuple[float, ...] = (4.0, 1.6)
COMPARATOR_DASHES: dict[str, tuple[float, ...]] = {
    "silhouette": (4.0, 1.6),
    "davies_bouldin": (1.4, 1.4),
    "calinski_harabasz": (5.5, 1.6, 1.2, 1.6),
    "gap": (2.6, 1.4, 1.4, 1.4),
    # The oracle keeps the long dash the published figures draw it with.
    "baseline_oracle": (5.0, 2.0),
}


def metric_dashes(metric: str) -> tuple[float, ...] | None:
    """The dash pattern for a metric, or None for CARVE's solid curves."""
    if metric.startswith(CARVE_ARI_PREFIX):
        return None
    return COMPARATOR_DASHES.get(metric, DEFAULT_DASHES)


def metric_linestyle(metric: str) -> str | tuple[int, tuple[float, ...]]:
    """Return the line style for a metric: solid for CARVE, dashed for the rest.

    A comparator's style is its own (offset, dashes) pair rather than the
    shared "--", so the four classical indices separate from each other in
    grayscale and under color-vision deficiency, not only from CARVE.
    """
    dashes = metric_dashes(metric)
    return "-" if dashes is None else (0, dashes)
```

- [ ] **Step 5: Add the sequential ramp**

Below `PIPELINE_CMAP_NAME`'s registration block:

```python
# One hue, light to dark, for magnitude. P(k-hat = k) is a proportion, so it
# takes a sequential ramp; a diverging or rainbow map would encode its
# ordering with hue, which reads as category. The hue is CARVE stability's
# green, so the heatmap belongs to the same figure as the curves above it.
SEQUENTIAL_COLORS: tuple[str, ...] = (
    "#F4FBF8",
    "#D6F0E5",
    "#ABE0CB",
    "#74CBAC",
    "#3FB48D",
    "#149A71",
    "#007A57",
    "#00573E",
)
SEQUENTIAL_CMAP_NAME: str = "carve_sequential"
if SEQUENTIAL_CMAP_NAME not in mpl.colormaps:
    mpl.colormaps.register(
        LinearSegmentedColormap.from_list(
            SEQUENTIAL_CMAP_NAME, list(SEQUENTIAL_COLORS)
        )
    )
```

Add `LinearSegmentedColormap` to the `matplotlib.colors` import.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_theme.py tests/benchmarks/test_panels.py -v`

Expected: PASS. If a panel test asserts `linestyle == "--"` on a drawn line, update it to compare against `metric_linestyle(metric)` rather than the literal.

- [ ] **Step 7: Re-run the validator and record the report**

```bash
cd /private/tmp/claude-501/bundled-skills/*/*/dataviz 2>/dev/null || \
  cd "$(dirname "$(find / -name validate_palette.js -path '*dataviz*' 2>/dev/null | head -1)")/.."
node scripts/validate_palette.js \
  "#595959,#009E73,#0072B2,#E0457B,#6A2C91,#B01B20,#C96A05" --mode light
```

Expected: PASS on lightness band, CVD separation, normal-vision floor and contrast. The grey oracle's chroma-floor flag is expected and deliberate — it is a neutral reference line, not a categorical series. Paste the report into the commit message.

- [ ] **Step 8: Lint and commit**

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
.venv/bin/ruff format src/benchmarks/_theme.py
.venv/bin/ruff check src/
git add src/benchmarks/_theme.py tests/benchmarks/
git commit -m "style(theme): re-step the failing hues and separate the comparators by dash

The published Fig 4 palette failed the normal-vision separation floor on
Silhouette/Davies-Bouldin (dE 13.6, floor 15) and the 3:1 contrast floor on
CARVE generalizability (2.25) and Gap (2.51).

Davies-Bouldin, Calinski-Harabasz and Gap take deeper steps of the same
hues; CARVE generalizability takes Okabe-Ito's blue. Families are unchanged:
blue and green for CARVE, pink/purple/red/orange for the comparators. Each
comparator also gets its own dash pattern, because four warm hues cannot
clear the CVD threshold pairwise whatever steps they take.

Adds a single-hue sequential ramp for the k-hat heatmap.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 11: The k-hat heatmap panel

**Files:**
- Modify: `src/benchmarks/_panels.py`
- Test: `tests/benchmarks/test_panels.py`

**Interfaces:**
- Consumes: Task 10's `SEQUENTIAL_CMAP_NAME`.
- Produces: `k_hat_heatmap(ax, df, *, metrics, axis_label, candidate_k, k_star, annotate=True) -> Axes`. Task 14 composes it.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_panels.py`:

```python
def _selection_frame():
    """Three datasets, two metrics, k in 3..7, with known selections.

    stability picks k=5 twice and k=6 once; silhouette picks k=7 every time.
    """
    rows = []
    picks = {"ari_stability_1se": [5, 5, 6], "silhouette": [7, 7, 7]}
    for metric, chosen in picks.items():
        for seed, pick in enumerate(chosen):
            for k in (3, 4, 5, 6, 7):
                rows.append(
                    {
                        "axis_label": "medium",
                        "axis_value": 1,
                        "seed": seed,
                        "metric_name": metric,
                        "k": k,
                        "metric_value": float(k),
                        "is_selected": k == pick,
                        "selects_true_k": k == 5,
                        "ari_at_k": 0.5,
                        "oracle_ari": 0.8,
                        "k_star": 5,
                    }
                )
    return pd.DataFrame(rows)


class TestKHatHeatmap:
    def test_cell_values_are_selection_proportions(self):
        fig, ax = plt.subplots()
        k_hat_heatmap(
            ax,
            _selection_frame(),
            metrics=["ari_stability_1se", "silhouette"],
            axis_label="medium",
            candidate_k=(3, 4, 5, 6, 7),
            k_star=5,
        )
        image = ax.get_images()[0].get_array()
        # Rows follow the metrics argument; columns follow candidate_k.
        assert image[0].tolist() == pytest.approx([0.0, 0.0, 2 / 3, 1 / 3, 0.0])
        assert image[1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0])
        plt.close(fig)

    def test_each_row_sums_to_one(self):
        """Every dataset selects exactly one k per metric, so a row that does
        not sum to 1 means selections were dropped or double counted."""
        fig, ax = plt.subplots()
        k_hat_heatmap(
            ax,
            _selection_frame(),
            metrics=["ari_stability_1se", "silhouette"],
            axis_label="medium",
            candidate_k=(3, 4, 5, 6, 7),
            k_star=5,
        )
        image = ax.get_images()[0].get_array()
        for row in image:
            assert float(row.sum()) == pytest.approx(1.0)
        plt.close(fig)

    def test_k_star_is_ruled(self):
        fig, ax = plt.subplots()
        k_hat_heatmap(
            ax,
            _selection_frame(),
            metrics=["ari_stability_1se"],
            axis_label="medium",
            candidate_k=(3, 4, 5, 6, 7),
            k_star=5,
        )
        # k*=5 is the third of five candidates, so the rule sits at x=2.
        assert [line.get_xdata()[0] for line in ax.get_lines()] == [2.0]
        plt.close(fig)

    def test_a_metric_with_no_rows_is_an_empty_row_not_a_missing_one(self):
        """Row order must match the metrics argument, so a dashboard's
        heatmap rows line up with the curves above them even when one metric
        produced nothing for this cell."""
        fig, ax = plt.subplots()
        k_hat_heatmap(
            ax,
            _selection_frame(),
            metrics=["ari_stability_1se", "gap", "silhouette"],
            axis_label="medium",
            candidate_k=(3, 4, 5, 6, 7),
            k_star=5,
        )
        image = ax.get_images()[0].get_array()
        assert image.shape == (3, 5)
        assert np.isnan(image[1]).all()
        plt.close(fig)

    def test_uses_the_theme_ramp(self):
        fig, ax = plt.subplots()
        k_hat_heatmap(
            ax,
            _selection_frame(),
            metrics=["ari_stability_1se"],
            axis_label="medium",
            candidate_k=(3, 4, 5, 6, 7),
            k_star=5,
        )
        assert ax.get_images()[0].get_cmap().name == SEQUENTIAL_CMAP_NAME
        plt.close(fig)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_panels.py -k KHatHeatmap -v`

Expected: collection error — `cannot import name 'k_hat_heatmap'`.

- [ ] **Step 3: Write the panel**

In `src/benchmarks/_panels.py`, after `metric_lines`:

```python
def k_hat_selection_matrix(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    axis_label: str,
    candidate_k: Sequence[int],
) -> np.ndarray:
    """P(k-hat = k) as a metrics-by-k array, NaN where a metric has no rows.

    Row order follows metrics and column order follows candidate_k, both as
    given, so the caller's ordering is the figure's ordering. A metric with
    no rows in this cell is an all-NaN row rather than a dropped one: a
    dropped row would slide every metric below it up by one and silently
    mislabel the axis.
    """
    cell = df.loc[df["axis_label"] == axis_label]
    matrix = np.full((len(metrics), len(candidate_k)), np.nan)
    for row, metric in enumerate(metrics):
        selected = cell.loc[
            (cell["metric_name"] == metric) & cell["is_selected"].astype(bool)
        ]
        if selected.empty:
            continue
        counts = selected["k"].value_counts()
        total = float(counts.sum())
        for column, k in enumerate(candidate_k):
            matrix[row, column] = float(counts.get(k, 0)) / total
    return matrix


def k_hat_heatmap(
    ax: Axes,
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    axis_label: str,
    candidate_k: Sequence[int],
    k_star: int,
    annotate: bool = True,
    title: str | None = None,
) -> Axes:
    """How often each method chose each k, as a metrics-by-k heatmap.

    This is the k-recovery column of the supplementary tables drawn rather
    than tabulated. A mean ARI hides the shape of a method's selections: a
    method that picks k* half the time and k*+2 the other half scores the
    same as one that always picks something in between, and only this panel
    separates them.

    The oracle is not drawn. It has no selected k -- it is fit once per cell
    at k_star -- so a row for it would be either empty or a tautological
    column of ones.
    """
    metrics = [m for m in metrics if m != BASELINE_METRIC]
    candidate_k = list(candidate_k)
    matrix = k_hat_selection_matrix(
        df, metrics=metrics, axis_label=axis_label, candidate_k=candidate_k
    )

    image = ax.imshow(
        matrix,
        cmap=SEQUENTIAL_CMAP_NAME,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(range(len(candidate_k)))
    ax.set_xticklabels([str(k) for k in candidate_k], fontsize=FONT_SIZES["tick"])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([_display(m) for m in metrics], fontsize=FONT_SIZES["tick"])
    ax.set_xlabel("$k$", fontsize=FONT_SIZES["axis_label"])

    if k_star in candidate_k:
        ax.axvline(
            candidate_k.index(k_star),
            color=FOREGROUND_COLOR,
            linewidth=1.2,
            alpha=0.55,
        )

    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isnan(value) or value < 0.005:
                    continue
                # Light text on the dark end of the ramp, dark on the light
                # end. A single ink color is unreadable at one end or the
                # other, and the threshold sits where the ramp crosses.
                ax.text(
                    column,
                    row,
                    f"{value:.2f}".lstrip("0") or "0",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZES["tick"] * 0.85,
                    color="#FFFFFF" if value > 0.55 else FOREGROUND_COLOR,
                )

    if title is not None:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.grid(False)
    return ax
```

Add `SEQUENTIAL_CMAP_NAME` to the `._theme` import block at the top of `_panels.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_panels.py -v`

Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_panels.py
.venv/bin/ruff check src/
git add src/benchmarks/_panels.py tests/benchmarks/test_panels.py
git commit -m "feat(panels): P(k-hat = k) as a method-by-k heatmap

The k-recovery column of S2-S9, drawn. A mean ARI cannot distinguish a
method that always picks k*+1 from one that splits between k* and k*+2;
this does.

Rows follow the metrics argument and a metric with no rows stays as an
all-NaN row, so a dashboard's heatmap lines up with the curves above it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 12: The normalized criterion-vs-k panel

**Files:**
- Modify: `src/benchmarks/_panels.py`
- Test: `tests/benchmarks/test_panels.py`

**Interfaces:**
- Consumes: Task 10's `metric_linestyle`.
- Produces:
  - `normalized_criterion(df, *, metrics, axis_label) -> pd.DataFrame` with columns `metric_name`, `k`, `value`
  - `criterion_curves(ax, df, *, metrics, axis_label, candidate_k, k_star, show_ari_reference=True, title=None) -> Axes`
  - Task 14 composes the second.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_panels.py`:

```python
def _criterion_frame():
    """Two metrics on wildly different scales, three datasets, k in 3..7.

    silhouette runs 0.1-0.5; calinski_harabasz runs 900-4100. Raw, the first
    is a flat line at the bottom of any axis the second fits on.
    """
    rows = []
    shapes = {
        "silhouette": {3: 0.10, 4: 0.30, 5: 0.50, 6: 0.28, 7: 0.12},
        "calinski_harabasz": {3: 900.0, 4: 2600.0, 5: 4100.0, 6: 2400.0, 7: 1100.0},
    }
    for metric, by_k in shapes.items():
        for seed in range(3):
            for k, value in by_k.items():
                rows.append(
                    {
                        "axis_label": "medium",
                        "axis_value": 1,
                        "seed": seed,
                        "metric_name": metric,
                        "k": k,
                        "metric_value": value,
                        "is_selected": k == 5,
                        "selects_true_k": k == 5,
                        "ari_at_k": {3: 0.4, 4: 0.7, 5: 0.9, 6: 0.6, 7: 0.3}[k],
                        "oracle_ari": 0.85,
                        "k_star": 5,
                    }
                )
    return pd.DataFrame(rows)


class TestNormalizedCriterion:
    def test_each_metric_spans_zero_to_one(self):
        """Criteria live on incompatible scales -- silhouette in [-1, 1],
        inverted Davies-Bouldin in (0, 1], Calinski-Harabasz in the
        thousands, gap a log ratio. Raw on one axis, six of seven flatten."""
        out = normalized_criterion(
            _criterion_frame(),
            metrics=["silhouette", "calinski_harabasz"],
            axis_label="medium",
        )
        for metric in ("silhouette", "calinski_harabasz"):
            values = out.loc[out["metric_name"] == metric, "value"]
            assert values.min() == pytest.approx(0.0)
            assert values.max() == pytest.approx(1.0)

    def test_normalization_preserves_shape(self):
        """Both metrics peak at k=5 in this frame, and must still after."""
        out = normalized_criterion(
            _criterion_frame(),
            metrics=["silhouette", "calinski_harabasz"],
            axis_label="medium",
        )
        for metric in ("silhouette", "calinski_harabasz"):
            sub = out.loc[out["metric_name"] == metric].set_index("k")["value"]
            assert sub.idxmax() == 5

    def test_normalizes_within_a_dataset_before_averaging(self):
        """A dataset whose criterion is ten times another's would otherwise
        dominate the mean and drag the averaged curve onto its own shape."""
        frame = _criterion_frame()
        loud = frame["seed"] == 0
        frame.loc[loud, "metric_value"] = frame.loc[loud, "metric_value"] * 10.0
        out = normalized_criterion(
            frame, metrics=["silhouette"], axis_label="medium"
        )
        sub = out.loc[out["metric_name"] == "silhouette"].set_index("k")["value"]
        # Every dataset has the same shape, so scaling one changes nothing.
        assert sub.loc[5] == pytest.approx(1.0)
        assert sub.loc[3] == pytest.approx(0.0)

    def test_a_flat_criterion_is_mid_scale_not_a_division_by_zero(self):
        """min == max means the criterion said nothing about k. 0.5 is the
        honest rendering; 0/0 is a RuntimeWarning that filterwarnings=error
        turns into a failure."""
        rows = [
            {
                "axis_label": "medium",
                "axis_value": 1,
                "seed": 0,
                "metric_name": "gap",
                "k": k,
                "metric_value": 2.0,
                "is_selected": k == 5,
                "selects_true_k": k == 5,
                "ari_at_k": 0.5,
                "oracle_ari": 0.8,
                "k_star": 5,
            }
            for k in (3, 4, 5, 6, 7)
        ]
        out = normalized_criterion(
            pd.DataFrame(rows), metrics=["gap"], axis_label="medium"
        )
        assert out["value"].tolist() == pytest.approx([0.5] * 5)


class TestCriterionCurves:
    def test_draws_one_line_per_metric_plus_the_reference(self):
        fig, ax = plt.subplots()
        criterion_curves(
            ax,
            _criterion_frame(),
            metrics=["silhouette", "calinski_harabasz"],
            axis_label="medium",
            candidate_k=(3, 4, 5, 6, 7),
            k_star=5,
        )
        labels = [line.get_label() for line in ax.get_lines()]
        assert "Silhouette" in labels
        assert "Calinski-Harabasz" in labels
        assert any("ARI" in str(label) for label in labels)
        plt.close(fig)

    def test_the_reference_shares_the_axis(self):
        """Normalized criteria and ARI are both in [0, 1], so a second y
        axis is unnecessary -- and a dual-axis chart invites reading a
        crossing that is an artifact of two scales."""
        fig, ax = plt.subplots()
        criterion_curves(
            ax,
            _criterion_frame(),
            metrics=["silhouette"],
            axis_label="medium",
            candidate_k=(3, 4, 5, 6, 7),
            k_star=5,
        )
        assert ax.get_shared_x_axes().get_siblings(ax) == [ax]
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_marks_each_metrics_selected_k(self):
        fig, ax = plt.subplots()
        criterion_curves(
            ax,
            _criterion_frame(),
            metrics=["silhouette"],
            axis_label="medium",
            candidate_k=(3, 4, 5, 6, 7),
            k_star=5,
        )
        marks = [c for c in ax.collections if c.get_label() == "_selected"]
        assert len(marks) == 1
        assert marks[0].get_offsets()[0][0] == pytest.approx(5.0)
        plt.close(fig)

    def test_the_reference_can_be_turned_off(self):
        fig, ax = plt.subplots()
        criterion_curves(
            ax,
            _criterion_frame(),
            metrics=["silhouette"],
            axis_label="medium",
            candidate_k=(3, 4, 5, 6, 7),
            k_star=5,
            show_ari_reference=False,
        )
        assert not any("ARI" in str(line.get_label()) for line in ax.get_lines())
        plt.close(fig)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_panels.py -k "NormalizedCriterion or CriterionCurves" -v`

Expected: collection error — `cannot import name 'normalized_criterion'`.

- [ ] **Step 3: Write the normalizer**

In `src/benchmarks/_panels.py`, after `k_hat_heatmap`:

```python
CRITERION_REFERENCE_LABEL: str = r"ARI at $k$ (reference)"


def normalized_criterion(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    axis_label: str,
) -> pd.DataFrame:
    """Each criterion min-max normalized over k within a dataset, then averaged.

    Returns long-form rows of (metric_name, k, value).

    Normalization is what makes seven criteria comparable on one axis.
    Silhouette lives in [-1, 1], inverted Davies-Bouldin in (0, 1],
    Calinski-Harabasz runs to the thousands and the gap statistic is a log
    ratio; drawn raw, six of the seven collapse into a line at the bottom of
    whatever axis the largest one needs. The panel's question is which k each
    criterion prefers, and that is a property of the curve's shape, which
    normalization preserves exactly.

    Within a dataset, then averaged -- not the other way round. Averaging
    raw values first lets one dataset whose criterion happens to be an order
    of magnitude larger dominate the mean and drag the averaged curve onto
    its own shape.

    A criterion that is flat across k normalizes to 0.5 rather than dividing
    by zero. Flat means the criterion expressed no preference, and 0.5 draws
    that as the flat line it is.
    """
    cell = df.loc[df["axis_label"] == axis_label]
    out: list[dict[str, Any]] = []
    for metric in metrics:
        sub = cell.loc[cell["metric_name"] == metric]
        if sub.empty:
            continue
        per_seed = []
        for _, group in sub.groupby("seed"):
            values = group.set_index("k")["metric_value"].astype(float).sort_index()
            span = values.max() - values.min()
            if not np.isfinite(span) or span <= 0:
                per_seed.append(pd.Series(0.5, index=values.index))
            else:
                per_seed.append((values - values.min()) / span)
        averaged = pd.concat(per_seed, axis=1).mean(axis=1)
        out.extend(
            {"metric_name": metric, "k": int(k), "value": float(v)}
            for k, v in averaged.items()
        )
    return pd.DataFrame(out, columns=["metric_name", "k", "value"])
```

- [ ] **Step 4: Write the panel**

```python
def criterion_curves(
    ax: Axes,
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    axis_label: str,
    candidate_k: Sequence[int],
    k_star: int,
    show_ari_reference: bool = True,
    title: str | None = None,
) -> Axes:
    """Why each method chose the k it chose, for one point on the axis.

    Every criterion is normalized to [0, 1] over the candidate k, so the
    shapes are comparable; the mean ARI of the labels at each k is drawn on
    the same axis as a grey reference, which is what turns "this criterion
    peaked at the wrong k" into "and here is what that cost". Both series
    are proportions in [0, 1], so no second y axis is needed -- and a
    dual-axis panel would invite reading a crossing that is an artifact of
    two independent scales.

    Each metric's selected k is marked on its own curve. The gap statistic's
    mark will not sit on its maximum: Tibshirani's rule takes the smallest k
    with Gap(k) >= Gap(k+1) - s(k+1), not an argmax. That is the panel
    showing the rule at work.
    """
    metrics = [m for m in metrics if m != BASELINE_METRIC]
    candidate_k = list(candidate_k)
    cell = df.loc[df["axis_label"] == axis_label]

    if show_ari_reference:
        ari = (
            cell.drop_duplicates(subset=["seed", "metric_name", "k"])
            .groupby("k")["ari_at_k"]
            .mean()
            .reindex(candidate_k)
        )
        ax.plot(
            candidate_k,
            ari.to_numpy(),
            color=FALLBACK_COLOR,
            linewidth=6.0,
            alpha=0.22,
            solid_capstyle="round",
            zorder=1,
            label=CRITERION_REFERENCE_LABEL,
        )

    curves = normalized_criterion(cell, metrics=metrics, axis_label=axis_label)
    selected = cell.loc[cell["is_selected"].astype(bool)]

    for metric in metrics:
        sub = curves.loc[curves["metric_name"] == metric]
        if sub.empty:
            continue
        values = sub.set_index("k")["value"].reindex(candidate_k)
        ax.plot(
            candidate_k,
            values.to_numpy(),
            color=metric_color(metric),
            linewidth=metric_linewidth(metric),
            linestyle=metric_linestyle(metric),
            zorder=2,
            label=_display(metric),
        )
        picks = selected.loc[selected["metric_name"] == metric, "k"]
        if picks.empty:
            continue
        # The modal selection across datasets, which is the k this panel's
        # averaged curve is being read as having chosen.
        modal = int(picks.mode().iloc[0])
        if modal in candidate_k:
            ax.scatter(
                [modal],
                [values.loc[modal]],
                s=46.0,
                marker="D",
                facecolor=metric_color(metric),
                edgecolor="#FFFFFF",
                linewidth=0.9,
                zorder=3,
                label="_selected",
            )

    if k_star in candidate_k:
        ax.axvline(k_star, color=FOREGROUND_COLOR, linewidth=1.2, alpha=0.45, zorder=0)

    ax.set_xticks(candidate_k)
    ax.set_xlabel("$k$", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Normalized criterion", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(-0.05, 1.05)
    if title is not None:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    return style_axes(ax)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_panels.py -v`

Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_panels.py
.venv/bin/ruff check src/
git add src/benchmarks/_panels.py tests/benchmarks/test_panels.py
git commit -m "feat(panels): normalized criterion over k, against an ARI reference

Seven criteria on incompatible scales -- silhouette in [-1,1], inverted
Davies-Bouldin in (0,1], Calinski-Harabasz in the thousands, gap a log
ratio -- flatten into one line if drawn raw. Each is min-max normalized
over k within a dataset, then averaged, which preserves the shape the panel
exists to show.

Mean ARI at each k is drawn on the same [0,1] axis as a grey reference, so
a criterion peaking at the wrong k shows what the miss cost. No second y
axis: both series are proportions.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 13: Per-seed points on the ARI panel

**Files:**
- Modify: `src/benchmarks/_panels.py` (`metric_lines`)
- Test: `tests/benchmarks/test_panels.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `metric_lines(..., show_points: bool = False, point_alpha: float = 0.28, dodge: float = 0.0)`. Task 14 passes `show_points=True`. Also `synthetic_run_frame(scenario_name, *, n_seeds=2, metrics=None) -> pd.DataFrame` in `tests/benchmarks/_helpers.py`, which Tasks 14 and 17 read.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_panels.py`:

```python
class TestMetricLinePoints:
    def test_points_are_off_by_default(self, results_frame):
        """Fig 4 draws means and error bars only. The dashboard asks for the
        per-seed cloud; the manuscript figure must not change shape."""
        fig, ax = plt.subplots()
        metric_lines(ax, results_frame, metrics=["ari_stability_1se"])
        assert ax.collections == []
        plt.close(fig)

    def test_points_draw_one_marker_per_dataset_and_axis_point(
        self, results_frame
    ):
        fig, ax = plt.subplots()
        metric_lines(
            ax, results_frame, metrics=["ari_stability_1se"], show_points=True
        )
        drawn = sum(len(c.get_offsets()) for c in ax.collections)
        expected = (
            results_frame.loc[
                results_frame["is_selected"]
                & (results_frame["metric_name"] == "ari_stability_1se")
            ]
            .drop_duplicates(subset=["axis_value", "seed"])
            .shape[0]
        )
        assert drawn == expected
        plt.close(fig)

    def test_dodge_separates_two_metrics_clouds(self, results_frame):
        """Overplotted at one x, seven methods' clouds merge into a band
        that says nothing. Dodging spreads them along x instead."""
        fig, ax = plt.subplots()
        metric_lines(
            ax,
            results_frame,
            metrics=["ari_stability_1se", "ari_generalizability_1se"],
            show_points=True,
            dodge=0.12,
        )
        first, second = ax.collections[0], ax.collections[1]
        assert first.get_offsets()[0][0] != second.get_offsets()[0][0]
        plt.close(fig)

    def test_the_oracle_gets_points_too(self, results_frame):
        fig, ax = plt.subplots()
        metric_lines(
            ax, results_frame, metrics=["baseline_oracle"], show_points=True
        )
        assert len(ax.collections) == 1
        plt.close(fig)

    def test_points_do_not_enter_the_legend(self, results_frame):
        """One legend entry per method. A second handle for its cloud
        doubles the legend and makes metric_legend's column arithmetic
        wrong."""
        fig, ax = plt.subplots()
        metric_lines(
            ax, results_frame, metrics=["ari_stability_1se"], show_points=True
        )
        labels = ax.get_legend_handles_labels()[1]
        assert labels.count("CARVE Stability (1SE)") == 1
        plt.close(fig)
```

Add the fixture this class needs beside the others in that file:

```python
@pytest.fixture
def results_frame():
    """A three-anchor, three-dataset frame for the two CARVE selectors."""
    from tests.benchmarks._helpers import synthetic_run_frame

    return synthetic_run_frame(
        "gaussians",
        n_seeds=3,
        metrics=["ari_stability_1se", "ari_generalizability_1se"],
    )
```

`synthetic_run_frame` is a shared helper: Task 14's dashboard tests and Task 17's rendering check both read it.

- [ ] **Step 2: Add the frame helper**

In `tests/benchmarks/_helpers.py`:

```python
def synthetic_run_frame(scenario_name, *, n_seeds=2, metrics=None):
    """A schema-complete artifact frame for one scenario, cheap to build.

    Covers every axis point, every candidate k and every requested metric,
    with a selection per (metric, dataset) so is_selected is never all
    False. Figure tests need shape and completeness, not real numbers, and
    running the real pipeline for a layout assertion costs minutes per test.
    """
    import numpy as np
    import pandas as pd

    from benchmarks._registry import SCENARIOS
    from benchmarks.figures._benchmarking_results import DEFAULT_METRICS

    scenario = SCENARIOS[scenario_name]
    metrics = [
        m for m in (DEFAULT_METRICS if metrics is None else metrics)
        if m != "baseline_oracle"
    ]
    rng = np.random.default_rng(0)
    rows = []
    for axis_idx, axis_value, axis_label in scenario.axis:
        for seed in range(n_seeds):
            oracle = 0.9 - 0.1 * axis_idx
            for metric in metrics:
                chosen = int(rng.choice(scenario.candidate_k))
                for k in scenario.candidate_k:
                    rows.append(
                        {
                            "run_id": "synthetic",
                            "scenario": scenario_name,
                            "axis_name": scenario.axis.name,
                            "axis_value": axis_value,
                            "axis_label": axis_label,
                            "seed": seed,
                            "k_star": scenario.k_star,
                            "estimator": scenario.estimator.name,
                            "oracle_ari": oracle,
                            "metric_name": metric,
                            "k": int(k),
                            "metric_value": float(rng.uniform(0.1, 0.9)),
                            "is_selected": k == chosen,
                            "selects_true_k": k == scenario.k_star,
                            "ari_at_k": float(rng.uniform(0.3, 0.95)),
                        }
                    )
    return pd.DataFrame(rows)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_panels.py -k MetricLinePoints -v`

Expected: FAIL — `metric_lines() got an unexpected keyword argument 'show_points'`.

- [ ] **Step 4: Extend metric_lines**

In `src/benchmarks/_panels.py`, add the three parameters to the signature after `element_scale`:

```python
    show_points: bool = False,
    point_alpha: float = 0.28,
    dodge: float = 0.0,
```

Add a helper just above `metric_lines`:

```python
def _dodge_offsets(metrics: Sequence[str], dodge: float) -> dict[str, float]:
    """Per-metric x offsets, centered on zero.

    Seven methods' per-seed clouds drawn at one x merge into a vertical band
    that carries no per-method information. Spreading them symmetrically
    keeps each method's spread readable while leaving the mean lines on the
    real axis positions.
    """
    if dodge <= 0 or len(metrics) < 2:
        return dict.fromkeys(metrics, 0.0)
    span = dodge * (len(metrics) - 1)
    return {
        metric: -span / 2 + index * dodge for index, metric in enumerate(metrics)
    }
```

Inside `metric_lines`, before the loop:

```python
    offsets = _dodge_offsets(list(metrics), dodge)
```

In the `baseline_oracle` branch, after the `ax.errorbar(...)` call and before `continue`:

```python
            if show_points:
                ax.scatter(
                    oracle[x_col].to_numpy(dtype=float) + offsets[metric],
                    oracle["oracle_ari"].to_numpy(dtype=float),
                    s=9.0 * element_scale,
                    color=metric_color(metric),
                    alpha=point_alpha,
                    linewidth=0.0,
                    zorder=1,
                    label="_points",
                )
```

In the ordinary-metric branch, after its `ax.errorbar(...)` call:

```python
        if show_points:
            ax.scatter(
                sub[x_col].to_numpy(dtype=float) + offsets[metric],
                sub["ari_at_k"].to_numpy(dtype=float),
                s=9.0 * element_scale,
                color=metric_color(metric),
                alpha=point_alpha,
                linewidth=0.0,
                zorder=1,
                label="_points",
            )
```

The `"_points"` label keeps the clouds out of `get_legend_handles_labels`, which `metric_legend` reads to build its columns.

Extend the docstring:

```python
    show_points draws each dataset's own value behind the mean, dodged along
    x by dodge so seven methods' clouds do not merge into one band. It is
    off by default: Fig 4 draws means and error bars, and this function is
    what draws Fig 4.
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_panels.py tests/benchmarks/figures -v`

Expected: PASS. The manuscript figure tests must be unaffected, since `show_points` defaults to `False`.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_panels.py
.venv/bin/ruff check src/
git add src/benchmarks/_panels.py tests/benchmarks/test_panels.py
git commit -m "feat(panels): optional per-seed cloud behind the ARI means

Dodged along x per method, because seven overplotted clouds at one x merge
into a band that carries no per-method information. Labelled '_points' so
one method still contributes one legend entry.

Off by default: Fig 4 draws means and error bars, and this function draws
Fig 4.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 14: Assemble the scenario dashboard

**Files:**
- Create: `src/benchmarks/figures/_scenario_dashboard.py`
- Modify: `src/benchmarks/figures/__init__.py`
- Test: `tests/benchmarks/figures/test_scenario_dashboard.py`

**Interfaces:**
- Consumes: Task 13's `synthetic_run_frame` (tests only); `draw_example_row` and `SCENARIO_TITLES` from `._benchmarking_examples`; `DEFAULT_METRICS` from `._benchmarking_results`; `metric_lines`, `criterion_curves`, `k_hat_heatmap`, `metric_legend` from `.._panels`; `SCALING_PANELS` from `._scaling`.
- Produces: `figure_scenario_dashboard(name: str, results: pd.DataFrame, *, metrics: Sequence[str] = DEFAULT_METRICS, seed: int = 0, random_state: int = PUBLISHED_RANDOM_STATE, save: bool = False, out_dir: Path | None = None) -> Figure`. Task 15's notebook calls it.

- [ ] **Step 1: Write the failing tests**

Create `tests/benchmarks/figures/test_scenario_dashboard.py`:

```python
"""Tests for the per-scenario dashboard."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarks._registry import SCENARIOS
from benchmarks.figures import figure_scenario_dashboard
from benchmarks.figures._benchmarking_results import DEFAULT_METRICS


@pytest.fixture
def gaussians_frame():
    """A complete three-anchor frame for gaussians, two datasets each."""
    from tests.benchmarks._helpers import synthetic_run_frame

    return synthetic_run_frame("gaussians", n_seeds=2)


class TestLayout:
    def test_four_rows_of_panels(self, gaussians_frame):
        """Examples, outcome, selection anatomy, k-recovery. The example row
        and the two per-anchor rows carry one panel per axis point; the ARI
        row spans the width."""
        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        n_points = len(SCENARIOS["gaussians"].axis)
        assert len(fig.axes) == 3 * n_points + 1
        plt.close(fig)

    def test_the_ari_row_spans_the_figure(self, gaussians_frame):
        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        widths = [ax.get_position().width for ax in fig.axes]
        assert max(widths) > 3 * min(widths)
        plt.close(fig)

    def test_every_panel_has_content(self, gaussians_frame):
        """An empty panel in a dashboard is indistinguishable from a panel
        whose data happened to be flat."""
        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        for ax in fig.axes:
            assert ax.get_lines() or ax.collections or ax.get_images()
        plt.close(fig)

    def test_one_shared_legend(self, gaussians_frame):
        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        assert len(fig.legends) == 1
        plt.close(fig)

    def test_titles_come_from_the_shared_map(self, gaussians_frame):
        from benchmarks.figures._benchmarking_examples import SCENARIO_TITLES

        fig = figure_scenario_dashboard("gaussians", gaussians_frame)
        assert fig._suptitle.get_text() == SCENARIO_TITLES["gaussians"]
        plt.close(fig)


class TestScalingScenarios:
    def test_a_scaling_scenario_labels_its_own_axis(self):
        """easy/medium/hard would be a mislabeling on an n or p sweep."""
        from tests.benchmarks._helpers import synthetic_run_frame

        frame = synthetic_run_frame("gaussians_samples", n_seeds=2)
        fig = figure_scenario_dashboard("gaussians_samples", frame)
        labels = [ax.get_xlabel() for ax in fig.axes]
        assert any("n" in label for label in labels)
        assert not any("SNR" in label for label in labels)
        plt.close(fig)


class TestGuards:
    def test_rejects_an_unknown_scenario(self, gaussians_frame):
        with pytest.raises(KeyError, match="not_a_scenario"):
            figure_scenario_dashboard("not_a_scenario", gaussians_frame)

    def test_rejects_an_empty_frame(self):
        with pytest.raises(ValueError, match="No rows"):
            figure_scenario_dashboard("gaussians", pd.DataFrame())

    def test_rejects_a_frame_from_a_different_axis(self, gaussians_frame):
        """A run directory is content-addressed on its config, so an older
        run whose axis has since been redefined is still readable. The
        example panels come from the registry and the curves from the
        artifact, so a mismatch would draw two different experiments in one
        figure with the x ticks hiding it."""
        frame = gaussians_frame.copy()
        frame["axis_value"] = frame["axis_value"] + 100
        with pytest.raises(ValueError, match="re-run this scenario"):
            figure_scenario_dashboard("gaussians", frame)


class TestSaving:
    def test_does_not_write_by_default(self, gaussians_frame, tmp_path):
        """A notebook view, not a manuscript figure."""
        fig = figure_scenario_dashboard("gaussians", gaussians_frame, out_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

    def test_writes_under_its_scenario_name_when_asked(
        self, gaussians_frame, tmp_path
    ):
        fig = figure_scenario_dashboard(
            "gaussians", gaussians_frame, save=True, out_dir=tmp_path
        )
        assert (tmp_path / "scenario_dashboard_gaussians.png").exists()
        plt.close(fig)
```

- [ ] **Step 2: Confirm the frame helper is present**

`synthetic_run_frame` is added to `tests/benchmarks/_helpers.py` in Task 13. If that task has not run yet, add it now from the definition given there; the dashboard tests and the table-rendering check both read it.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/figures/test_scenario_dashboard.py -v`

Expected: collection error — `cannot import name 'figure_scenario_dashboard'`.

- [ ] **Step 4: Write the dashboard**

Create `src/benchmarks/figures/_scenario_dashboard.py`:

```python
"""One scenario's whole story in one figure.

Not a manuscript figure. This is what each family section of the
benchmarking notebook carries, in place of the overview-plus-table pair it
used to. The four rows answer four questions in reading order: what the data
looks like, what each method scored on it, why each method chose the k it
chose, and how often it chose each k.

Every panel is drawn through the same primitives as the manuscript figures,
and the ARI row draws Fig 4's series from the same artifact, so a section and
the combined figure at the end of the notebook cannot disagree about what a
family looks like.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .._panels import criterion_curves, k_hat_heatmap, metric_legend, metric_lines
from .._registry import PUBLISHED_RANDOM_STATE, SCENARIOS
from .._theme import FONT_SIZES, save_figure, theme_context
from ._benchmarking_examples import SCENARIO_TITLES, draw_example_row
from ._benchmarking_results import DEFAULT_METRICS
from ._paths import BENCHMARKING_DIR, figure_path
from ._scaling import SCALING_PANELS
from ._scenario_overview import DIFFICULTY_AXIS_NAME, DIFFICULTY_X_LABEL, _axis_points

# Seven methods' per-seed clouds at one x merge into a band; this spreads
# them far enough to read and not so far that a method's cloud sits over the
# next axis point's.
POINT_DODGE_FRACTION: float = 0.055


def _x_label(name: str, axis_name: str) -> str:
    """The x-axis label for one scenario, reusing the scaling figure's wording."""
    if axis_name == DIFFICULTY_AXIS_NAME:
        return DIFFICULTY_X_LABEL
    return dict(SCALING_PANELS).get(name, axis_name)


def figure_scenario_dashboard(
    name: str,
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    seed: int = 0,
    random_state: int = PUBLISHED_RANDOM_STATE,
    save: bool = False,
    out_dir: Path | None = None,
) -> Figure:
    """Build one scenario's dashboard.

    Parameters
    ----------
    name : str
        Registry key naming the scenario.
    results : DataFrame
        That scenario's artifact frame, in the unified schema.
    metrics : sequence of str
        Metric names, in legend order. Defaults to Fig 4's, so a section
        panel and its Fig 4 counterpart show the same series.
    seed : int
        Seed-loop index for the example row.
    random_state : int
        The run's random_state term in the example row's seed derivation.
    save : bool
        Write the file. Defaults to False: this is a notebook view, not a
        manuscript figure, so it stays off disk unless asked for.
    out_dir : Path or None
        Override the destination directory.
    """
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}. Known: {sorted(SCENARIOS)}.")
    if results.empty:
        raise ValueError(f"No rows for scenario {name!r}; nothing to draw.")

    scenario = SCENARIOS[name]
    axis_name = str(results["axis_name"].iloc[0])
    values, labels = _axis_points(results)

    # The example and criterion panels are simulated or read per registry
    # anchor; the curves come from the artifact. A run directory is
    # content-addressed on its config, so an older run whose axis has since
    # been redefined still sits on disk and still reads. The two halves would
    # then describe different experiments, with the x ticks taken from the
    # artifact hiding the disagreement.
    registered = list(scenario.axis.values)
    if values != registered:
        raise ValueError(
            f"Scenario {name!r} is registered on axis {scenario.axis.name}="
            f"{registered}, but the frame sweeps {values}. The example panels "
            "and the curves would describe different experiments. Re-run this "
            f"scenario: python -m benchmarks.run --scenario {name}"
        )

    with theme_context():
        n_cols = len(scenario.axis)
        fig = plt.figure(figsize=(3.9 * n_cols, 13.2))
        grid = fig.add_gridspec(
            4,
            n_cols,
            height_ratios=(1.0, 1.25, 1.15, 1.0),
            hspace=0.42,
            wspace=0.28,
        )

        example_axes = [fig.add_subplot(grid[0, i]) for i in range(n_cols)]
        ari_ax = fig.add_subplot(grid[1, :])
        criterion_axes = [fig.add_subplot(grid[2, i]) for i in range(n_cols)]
        heatmap_axes = [fig.add_subplot(grid[3, i]) for i in range(n_cols)]

        if axis_name == DIFFICULTY_AXIS_NAME:
            titles = [label.capitalize() for _, _, label in scenario.axis]
        else:
            titles = [
                f"{scenario.axis.name} = {value}" for _, value, _ in scenario.axis
            ]

        # Row 1 -- what the data looks like.
        draw_example_row(
            example_axes,
            scenario,
            seed=seed,
            random_state=random_state,
            titles=titles,
        )

        # Row 2 -- what each method scored.
        span = max(values) - min(values) if len(values) > 1 else 1.0
        metric_lines(
            ari_ax,
            results,
            metrics=metrics,
            x_col="axis_value",
            x_label=_x_label(name, axis_name),
            show_legend=False,
            show_points=True,
            dodge=POINT_DODGE_FRACTION * float(span),
        )
        # Ticks sit on the swept anchors, not on matplotlib's automatic
        # scale. A scaling family holds data at exactly three values, and
        # auto ticks put marks at 2000 and 8000 where nothing was measured.
        ari_ax.set_xticks(values)
        if axis_name == DIFFICULTY_AXIS_NAME:
            ari_ax.set_xticklabels([label.capitalize() for label in labels])

        # Rows 3 and 4 -- why that k, and how often.
        for index, axis_label in enumerate(labels):
            criterion_curves(
                criterion_axes[index],
                results,
                metrics=metrics,
                axis_label=axis_label,
                candidate_k=scenario.candidate_k,
                k_star=scenario.k_star,
                title=titles[index],
            )
            k_hat_heatmap(
                heatmap_axes[index],
                results,
                metrics=metrics,
                axis_label=axis_label,
                candidate_k=scenario.candidate_k,
                k_star=scenario.k_star,
            )
            if index:
                criterion_axes[index].set_ylabel("")
                heatmap_axes[index].set_yticklabels([])

        fig.suptitle(
            SCENARIO_TITLES.get(name, name), fontsize=FONT_SIZES["title"], y=0.995
        )
        metric_legend(fig, np.array([ari_ax]), metrics, y_offset=0.03)

        if save:
            save_figure(
                fig,
                figure_path(
                    f"scenario_dashboard_{name}.png",
                    subdir=BENCHMARKING_DIR,
                    out_dir=out_dir,
                ),
            )
    return fig
```

- [ ] **Step 5: Export it**

In `src/benchmarks/figures/__init__.py`, add the import beside `_scenario_overview`'s and the name to `__all__` in alphabetical position:

```python
from ._scenario_dashboard import figure_scenario_dashboard
```

```python
    "figure_scenario_dashboard",
```

Extend the module docstring's second paragraph:

```
_scenario_overview, _scenario_dashboard and _reference_scatter are the three
modules here that are not manuscript figures. The first two draw the
per-family views the benchmarking notebook's sections carry, built from the
same primitives the manuscript figures use, and so default to save=False.
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/figures -v`

Expected: PASS.

- [ ] **Step 7: Look at the output**

The validator checks color, not layout. Render one and open it.

```bash
.venv/bin/python -u - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
from pathlib import Path
# tests/ and tests/benchmarks/ are both packages, so this imports from the
# repo root. Never sys.path.insert("tests") -- tests/benchmarks/ would then
# shadow the real benchmarks package.
from tests.benchmarks._helpers import synthetic_run_frame
from benchmarks.figures import figure_scenario_dashboard
frame = synthetic_run_frame("gaussians", n_seeds=20)
figure_scenario_dashboard("gaussians", frame, save=True, out_dir=Path("/tmp/dash"))
print("/tmp/dash/scenario_dashboard_gaussians.png")
PY
open /tmp/dash/scenario_dashboard_gaussians.png
```

Check for label collisions, heatmap row labels running off the left edge, the legend overlapping row 3, and the figure being unreadably tall. Adjust `figsize`, `height_ratios`, `hspace` or `y_offset` until it reads. This step is not optional — the tests assert structure, not legibility.

- [ ] **Step 8: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/figures/_scenario_dashboard.py src/benchmarks/figures/__init__.py
.venv/bin/ruff check src/
git add src/benchmarks/figures/ tests/benchmarks/
git commit -m "feat(figures): one dashboard per benchmark scenario

Four rows in reading order: example scatters, ARI at the selected k with
each dataset's own value behind the mean, normalized criterion over k
against an ARI reference, and P(k-hat = k) as a method-by-k heatmap.

Replaces the overview-plus-table pair each notebook section carried. The
ARI row draws Fig 4's series from the same artifact, so a section and the
combined figure cannot disagree.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 15: Rewrite the notebook

**Files:**
- Modify: `notebooks/Benchmarking.ipynb`
- Test: `tests/benchmarks/test_notebooks.py`

**Interfaces:**
- Consumes: Task 8's `widest_run`, Task 14's `figure_scenario_dashboard`.
- Produces: a notebook whose scenario sections are one dashboard call each.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_notebooks.py`:

```python
def test_benchmarking_uses_the_shared_run_resolver():
    """The notebook had its own run_dir() helper that duplicated the
    selection rule, which is how it and the CLI came to disagree."""
    source = _code(NOTEBOOKS["benchmarking"])
    assert "widest_run" in source
    assert "def run_dir" not in source


def test_benchmarking_draws_a_dashboard_per_scenario():
    source = _code(NOTEBOOKS["benchmarking"])
    assert source.count("scenario_dashboard(") >= 8


def test_benchmarking_has_no_inline_summary_table():
    """The sections are visual. summarize() still runs, as the engine
    behind the manuscript tables, but not in a notebook cell."""
    source = _code(NOTEBOOKS["benchmarking"])
    assert "scenario_report" not in source
    assert "SUMMARY_COLUMNS" not in source


def test_benchmarking_still_writes_the_manuscript_tables():
    source = _code(NOTEBOOKS["benchmarking"])
    assert "write_all_tables" in source
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_notebooks.py -v`

Expected: FAIL on all four.

- [ ] **Step 3: Rewrite the helper cell**

Open `notebooks/Benchmarking.ipynb` and replace cell 5's source entirely:

```python
from pathlib import Path

from benchmarks._artifacts import read_run, read_runtimes, widest_run
from benchmarks._registry import SCENARIOS
from benchmarks.figures import (
    figure_benchmarking_examples,
    figure_benchmarking_results,
    figure_scaling_ari,
    figure_scaling_runtime,
    figure_scenario_dashboard,
)
from benchmarks.tables import write_all_tables

RESULTS_ROOT = Path("../results/runs")


def load(name):
    """Read the widest complete run for a scenario.

    widest_run is shared with `python -m benchmarks.run --tables`, so a
    regenerated table and a notebook panel always read the same artifact.
    """
    return read_run(widest_run(RESULTS_ROOT, name))


def show(fig):
    """Display a figure exactly once, then close it.

    A figure left open is rendered by the inline backend's end-of-cell
    flush, and a figure returned from a cell is rendered as that cell's
    result. A helper that does both renders twice in some cells and once in
    others. Displaying explicitly and closing is the one path that draws
    every figure exactly once.
    """
    display(fig)
    plt.close(fig)


def scenario_dashboard(name):
    """Draw one scenario's whole story: data, outcome, selection, recovery."""
    show(figure_scenario_dashboard(name, load(name)))
```

- [ ] **Step 4: Replace each scenario section**

For each of the eight scenarios, the section currently holds two code cells — `scenario_overview("<name>")` and `scenario_report("<name>")`. Replace the pair with one cell:

```python
scenario_dashboard("gaussians")
```

and the same for `t_dist`, `t_dist_noise`, `circles`, `moons`, `swiss_rolls`, `gaussians_dimensionality`, `gaussians_samples`.

- [ ] **Step 5: Update the two markdown cells that describe the sections**

Cell 7:

```
Each scenario below carries one figure with four rows: example datasets at
the three anchors, ARI at the k each method selected with every dataset's own
value behind the mean, each criterion's normalized shape over k against the
ARI it was tracking, and how often each method chose each k.

Compute runs headless. Produce artifacts with `python -m benchmarks.run --all`,
then re-run this notebook, which only reads them.
```

Cell 28:

```
These two sweep an axis other than difficulty, so their dashboards are read
against that axis rather than against easy/medium/hard. Runtime is reported
separately in the scaling figures in section 4.
```

- [ ] **Step 6: Update the combined-figures cell**

Cell 37 references `run_dir`, which is gone. Replace its source:

```python
results = {name: load(name) for name in SCENARIOS}

difficulty = {
    k: v for k, v in results.items() if v["axis_name"].iloc[0] == "difficulty_level"
}
scaling = {
    k: v for k, v in results.items() if v["axis_name"].iloc[0] != "difficulty_level"
}
runtimes = {name: read_runtimes(widest_run(RESULTS_ROOT, name)) for name in scaling}

# Each of these writes itself under its manuscript filename in vis/
# benchmarking/. Copying one into overleaf/vis/ stays a manual step.
show(figure_benchmarking_examples())
show(figure_benchmarking_results(difficulty))
show(figure_scaling_ari(scaling))
show(figure_scaling_runtime(runtimes))

write_all_tables(results, Path("../vis/tables"))
```

- [ ] **Step 7: Clear the outputs**

The notebook currently carries 4 MB of stale outputs from the mixed-scale artifacts. Those images are wrong now.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("notebooks/Benchmarking.ipynb")
nb = json.loads(path.read_text())
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
path.write_text(json.dumps(nb, indent=1) + "\n")
print("cleared")
PY
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_notebooks.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add notebooks/Benchmarking.ipynb tests/benchmarks/test_notebooks.py
git commit -m "feat(notebooks): one dashboard per section, no inline tables

Each scenario section is now a single four-row figure. The notebook's
private run_dir helper is gone in favor of the shared widest_run, which is
what the CLI's --tables path reads, so a regenerated table and a notebook
panel cannot read different artifacts.

Outputs cleared: they were rendered from the mixed-scale run tree.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 16: Declare the table row order in the registry

**Files:**
- Modify: `src/benchmarks/_registry.py` (add `TABLE_ROW_GROUPS`)
- Modify: `src/benchmarks/tables.py` (`table_metrics` reads it)
- Test: `tests/benchmarks/test_registry.py`, `tests/benchmarks/test_tables_cli.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TABLE_ROW_GROUPS: tuple[tuple[str, ...], ...]`. Task 17's renderer draws its `\midrule` positions from it.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_registry.py`:

```python
class TestTableRowGroups:
    def test_the_groups_reproduce_the_committed_s2_table(self):
        """Row order and grouping are part of the manuscript's layout.

        Deriving them from a sorted set, which is what table_metrics did,
        means a new metric silently reorders every supplementary table.
        """
        assert TABLE_ROW_GROUPS == (
            ("baseline_oracle",),
            ("ari_stability_1se", "ari_generalizability_1se"),
            ("davies_bouldin", "silhouette", "gap", "calinski_harabasz"),
            (
                "ari_stability_quant",
                "consensus_gini_stability",
                "ari_stability",
                "ari_generalizability_quant",
                "ari_generalizability",
                "accuracy_generalizability",
            ),
        )

    def test_no_metric_appears_twice(self):
        flat = [m for group in TABLE_ROW_GROUPS for m in group]
        assert len(flat) == len(set(flat))

    def test_every_group_member_is_a_known_metric(self):
        known = set(CARVE_METRICS_ALL) | set(CVI_METRICS) | {BASELINE_METRIC}
        for group in TABLE_ROW_GROUPS:
            assert set(group) <= known

    def test_the_excluded_metrics_are_absent(self):
        """EXCLUDED_METRICS drops the three ari_average variants, PAC and
        CE, which is why the published tables do not carry them."""
        from benchmarks.tables import EXCLUDED_METRICS

        flat = {m for group in TABLE_ROW_GROUPS for m in group}
        assert flat & EXCLUDED_METRICS == set()
```

In `tests/benchmarks/test_tables_cli.py`:

```python
def test_table_metrics_follows_the_declared_groups():
    from benchmarks._registry import TABLE_ROW_GROUPS
    from benchmarks.tables import table_metrics

    assert table_metrics() == tuple(m for group in TABLE_ROW_GROUPS for m in group)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_registry.py -k TableRowGroups -v`

Expected: collection error — `cannot import name 'TABLE_ROW_GROUPS'`.

- [ ] **Step 3: Declare the groups**

In `src/benchmarks/_registry.py`, after `METRIC_LEGEND_NAMES`:

```python
# The supplementary tables' row order and grouping, as the committed S2
# table prints them. Four groups, separated by a rule: the oracle, the two
# headline CARVE selectors, the four classical indices, then the remaining
# CARVE selectors.
#
# Declared rather than derived. table_metrics previously built its order
# from sorted(CARVE_METRICS_ALL) plus CVI_METRICS, so the generated
# fragment's rows were in a different order from the manuscript's and adding
# a metric would have silently reordered all eight tables. The three
# ari_average variants, PAC and CE are absent because tables.EXCLUDED_METRICS
# drops them, which is what the published tables do.
TABLE_ROW_GROUPS: tuple[tuple[str, ...], ...] = (
    (BASELINE_METRIC,),
    ("ari_stability_1se", "ari_generalizability_1se"),
    ("davies_bouldin", "silhouette", "gap", "calinski_harabasz"),
    (
        "ari_stability_quant",
        "consensus_gini_stability",
        "ari_stability",
        "ari_generalizability_quant",
        "ari_generalizability",
        "accuracy_generalizability",
    ),
)
```

- [ ] **Step 4: Point table_metrics at it**

In `src/benchmarks/tables.py`:

```python
def table_metrics() -> tuple[str, ...]:
    """Metrics that appear in the manuscript tables, in the published order.

    Flattens TABLE_ROW_GROUPS, so the order a table renders in and the order
    a caller requests are one declaration rather than two that can drift.
    """
    return tuple(metric for group in TABLE_ROW_GROUPS for metric in group)
```

Replace the `CARVE_METRICS_ALL, CVI_METRICS` import with `TABLE_ROW_GROUPS`, keeping `EXCLUDED_METRICS` in place as documentation of why the five are missing.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks -q`

Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_registry.py src/benchmarks/tables.py
.venv/bin/ruff check src/
git add src/benchmarks/ tests/benchmarks/
git commit -m "refactor(benchmarks): declare the table row order in the registry

table_metrics derived its order from a sorted set, so the generated
fragments' rows were in a different order from the manuscript's and a new
metric would have reordered all eight tables. TABLE_ROW_GROUPS is the
committed S2 table's order and grouping, declared once.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 17: The paired-sub-table LaTeX renderer

**Files:**
- Modify: `src/benchmarks/_tables.py` (add `render_paired_tex`, point `write_tables` at it)
- Test: `tests/benchmarks/test_tables.py`

**Interfaces:**
- Consumes: Task 16's `TABLE_ROW_GROUPS`, `summarize` from this module.
- Produces: `render_paired_tex(summary: pd.DataFrame, *, caption: str, label: str, decimals: int = 3) -> str`.

- [ ] **Step 1: Write the failing tests**

In `tests/benchmarks/test_tables.py`:

```python
@pytest.fixture
def paired_summary():
    """A summary with a known best and second-best per column."""
    rows = []
    values = {
        "baseline_oracle": (0.914, float("nan")),
        "ari_stability_1se": (0.932, 1.000),
        "ari_generalizability_1se": (0.868, 0.550),
        "davies_bouldin": (0.928, 0.950),
        "silhouette": (0.900, 0.700),
        "gap": (0.851, 0.650),
        "calinski_harabasz": (0.640, 0.000),
    }
    for metric, (ari, recovery) in values.items():
        rows.append(
            {
                "axis_label": "easy",
                "metric": metric,
                "display_name": METRIC_DISPLAY_NAMES[metric],
                "n_datasets": 20,
                "ari_mean": ari,
                "ari_sd": 0.05,
                "k_recovery": recovery,
            }
        )
    return pd.DataFrame(rows)


class TestPairedTex:
    def test_two_tabulars_inside_one_resizebox(self, paired_summary):
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert out.count(r"\begin{tabular}") == 2
        assert r"\resizebox{\textwidth}{!}{" in out
        assert r"\quad" in out

    def test_booktabs_rules(self, paired_summary):
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert r"\toprule" in out
        assert r"\bottomrule" in out
        assert r"\hline" not in out

    def test_one_midrule_between_each_pair_of_row_groups(self, paired_summary):
        """Four groups means three internal rules, plus the header's, per
        sub-table."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert out.count(r"\midrule") == 2 * (1 + 3)

    def test_the_best_in_a_column_is_bold(self, paired_summary):
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert r"\textbf{0.932}" in out

    def test_the_second_best_is_underlined(self, paired_summary):
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert r"\underline{0.928}" in out

    def test_the_oracle_is_never_ranked(self, paired_summary):
        """It is the reference the others are measured against, not a
        competitor. The committed tables leave it unmarked."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert r"\textbf{0.914}" not in out
        assert r"\underline{0.914}" not in out

    def test_ties_for_best_are_all_bold_with_no_underline(self):
        """Matching the committed S2 table: three cells at 0.932 are all
        bold and nothing is underlined in that column."""
        rows = [
            {
                "axis_label": "easy",
                "metric": m,
                "display_name": METRIC_DISPLAY_NAMES[m],
                "n_datasets": 20,
                "ari_mean": v,
                "ari_sd": 0.05,
                "k_recovery": 0.5,
            }
            for m, v in (
                ("ari_stability_1se", 0.932),
                ("ari_stability_quant", 0.932),
                ("ari_stability", 0.932),
                ("silhouette", 0.800),
            )
        ]
        out = render_paired_tex(
            pd.DataFrame(rows), caption="c", label="tab:x"
        )
        assert out.count(r"\textbf{0.932}") == 3
        assert r"\underline{" not in out

    def test_the_oracles_recovery_cells_are_blank(self, paired_summary):
        """The oracle has no selected k, so k-recovery is NaN. The published
        tables print an empty cell, not the literal nan."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        right = out.split(r"\quad")[1]
        oracle_row = next(
            line for line in right.splitlines() if "Baseline (Oracle)" in line
        )
        assert oracle_row.replace(" ", "").endswith(r"&&&\\") or "nan" not in oracle_row

    def test_rows_follow_the_declared_groups(self, paired_summary):
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        left = out.split(r"\quad")[0]
        positions = [
            left.index(METRIC_DISPLAY_NAMES[m])
            for group in TABLE_ROW_GROUPS
            for m in group
            if METRIC_DISPLAY_NAMES[m] in left
        ]
        assert positions == sorted(positions)

    def test_a_metric_missing_from_the_summary_is_skipped(self, paired_summary):
        """A run that produced no rows for one metric must not leave a row
        of the literal nan."""
        out = render_paired_tex(paired_summary, caption="c", label="tab:x")
        assert "nan" not in out

    def test_the_caption_and_label_survive(self, paired_summary):
        out = render_paired_tex(
            paired_summary, caption="Gaussian mixtures.", label="tab:s2"
        )
        assert r"\caption{Gaussian mixtures.}" in out
        assert r"\label{tab:s2}" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_tables.py -k PairedTex -v`

Expected: collection error — `cannot import name 'render_paired_tex'`.

- [ ] **Step 3: Write the renderer**

In `src/benchmarks/_tables.py`, after `render_grouped_tex`:

```python
def _ranked(values: Sequence[float]) -> tuple[float | None, float | None]:
    """The best and second-best finite values in a column.

    Second-best is None when the best is tied, matching the committed S2
    table: three cells at 0.932 are all bold and nothing in that column is
    underlined. Marking a runner-up below a tied first place would say the
    column has a clear ordering when it does not.
    """
    finite = sorted({float(v) for v in values if pd.notna(v)}, reverse=True)
    if not finite:
        return None, None
    best = finite[0]
    if sum(1 for v in values if pd.notna(v) and float(v) == best) > 1:
        return best, None
    return best, (finite[1] if len(finite) > 1 else None)


def _marked(value: float, best: float | None, second: float | None, decimals: int) -> str:
    """One cell, bold if best in its column and underlined if second."""
    if pd.isna(value):
        return ""
    text = f"{value:.{decimals}f}"
    if best is not None and float(value) == best:
        return rf"\textbf{{{text}}}"
    if second is not None and float(value) == second:
        return rf"\underline{{{text}}}"
    return text


def _sub_table(
    summary: pd.DataFrame,
    groups: Sequence[Sequence[str]],
    axis_labels: Sequence[str],
    column: str,
    decimals: int,
) -> list[str]:
    """One of the pair: rows by metric group, columns by axis label."""
    lines = [
        f"\\begin{{tabular}}{{l{'c' * len(axis_labels)}}}",
        r"\toprule",
        "Metric & " + " & ".join(_tex_escape(str(a)) for a in axis_labels) + r" \\",
        r"\midrule",
    ]

    # Ranking is per column and excludes the oracle, which is the reference
    # the rest are measured against rather than a competitor.
    ranks: dict[str, tuple[float | None, float | None]] = {}
    for axis_label in axis_labels:
        candidates = summary[
            (summary["axis_label"] == axis_label)
            & (summary["metric"] != BASELINE_METRIC)
        ][column]
        ranks[axis_label] = _ranked(candidates.tolist())

    for index, group in enumerate(groups):
        if index:
            lines.append(r"\midrule")
        for metric in group:
            match = summary[summary["metric"] == metric]
            if match.empty:
                continue
            cells = [_tex_escape(METRIC_DISPLAY_NAMES.get(metric, metric))]
            for axis_label in axis_labels:
                row = match[match["axis_label"] == axis_label]
                if row.empty:
                    cells.append("")
                    continue
                value = row.iloc[0][column]
                if metric == BASELINE_METRIC:
                    cells.append(
                        "" if pd.isna(value) else f"{value:.{decimals}f}"
                    )
                else:
                    best, second = ranks[axis_label]
                    cells.append(_marked(value, best, second, decimals))
            lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def render_paired_tex(
    summary: pd.DataFrame,
    *,
    caption: str,
    label: str,
    decimals: int = 3,
) -> str:
    """The manuscript's shape: mean ARI and k-recovery side by side.

    Two tabulars inside one resizebox, separated by \\quad, the left giving
    mean ARI at the selected k and the right the proportion of datasets
    where that k was k*. Rows follow TABLE_ROW_GROUPS with a rule between
    groups; bold marks the best value in a column and underline the
    second-best.

    This replaces render_grouped_tex's single flat table, which merged both
    quantities into one unranked grid. The numbers were right; the layout
    was not, and every fragment had to be reshaped by hand before it could
    go into the manuscript.

    k-recovery has two decimals rather than three throughout, as the
    published tables do: with 20 datasets a proportion can only take
    multiples of 0.05, so a third decimal is always zero.
    """
    axis_labels = list(dict.fromkeys(summary["axis_label"]))
    groups = [
        [m for m in group if m in set(summary["metric"])] for group in TABLE_ROW_GROUPS
    ]
    groups = [group for group in groups if group]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
    ]
    lines += _sub_table(summary, groups, axis_labels, "ari_mean", decimals)
    lines.append(r"\quad")
    lines += _sub_table(summary, groups, axis_labels, "k_recovery", 2)
    lines.extend([r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"
```

Add `TABLE_ROW_GROUPS` to the `._registry` import at the top of `_tables.py`.

- [ ] **Step 4: Point write_tables at it**

```python
    path.write_text(
        render_paired_tex(summarize(df, metrics=metrics), caption=caption, label=label)
    )
```

Keep `render_grouped_tex` in the module. The ablation table is a different shape and does not use it, but removing a public renderer is not this plan's business, and its tests document the old layout.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_tables.py tests/benchmarks/test_tables_cli.py -v`

Expected: PASS.

- [ ] **Step 6: Check one fragment compiles**

```bash
.venv/bin/python -u - <<'PY'
from pathlib import Path
from tests.benchmarks._helpers import synthetic_run_frame
from benchmarks.tables import write_all_tables
out = Path("/tmp/carve-tables")
paths = write_all_tables({"gaussians": synthetic_run_frame("gaussians", n_seeds=4)}, out)
print((out / "S2_table.tex").read_text())
PY
```

Read the output. It must have two `\begin{tabular}` blocks, `\toprule`/`\midrule`/`\bottomrule`, no `nan`, and a bold cell in each column of the left sub-table. If `booktabs` is not already in the manuscript preamble, check `overleaf/CARVE_manuscript.tex` for `\usepackage{booktabs}` — the committed S2 table uses those rules, so it is there; if it is not, note it as a manual preamble addition rather than falling back to `\hline`.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format src/benchmarks/_tables.py
.venv/bin/ruff check src/
git add src/benchmarks/_tables.py tests/benchmarks/
git commit -m "feat(tables): S2-S9 fragments in the manuscript's paired shape

Two tabulars in one resizebox -- mean ARI left, k-recovery right -- with
rows grouped per TABLE_ROW_GROUPS, bold for the best in a column and
underline for the second-best, ties all bold with nothing underlined.

The old flat single table carried the right numbers in the wrong layout, so
every fragment was reshaped by hand before it reached the manuscript.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 18: Archive the old runs and take the dev-scale acceptance run

**Files:**
- Moves: `results/runs/<scenario>/`, `vis/benchmarking/*.png`, `vis/tables/*.tex` (all gitignored)
- Test: the whole suite, plus a manual read of every artifact

**Interfaces:**
- Consumes: every preceding task.
- Produces: a complete set of dev-scale artifacts, figures and tables, proving the pipeline runs end to end.

- [ ] **Step 1: Archive the existing tree**

Every path here is gitignored. The tracked `results/results_*.csv` and `results/scalability_results/` stay where they are — they are the regression oracle for the three unaffected scenarios.

```bash
cd /Users/kaiwycik/GitHub/CARVE/code
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "archive/$STAMP"
mv results/runs "archive/$STAMP/runs"
mv vis/benchmarking "archive/$STAMP/benchmarking"
mv vis/tables "archive/$STAMP/tables"
mkdir -p results/runs vis/benchmarking vis/tables
# The ablation run is a separate piece of work with its own gates.
mv "archive/$STAMP/runs/ablation_rho_b" results/runs/ 2>/dev/null || true
ls "archive/$STAMP"
```

Confirm `git status --short` shows nothing but the files this branch already changed. If `archive/` is not gitignored, add it to `.gitignore` and commit that line on its own.

- [ ] **Step 2: Run all eight scenarios at dev scale**

Three datasets is enough for a standard deviation and a k-recovery fraction; ten resamples is enough for a consensus matrix. Its numbers are not reported.

```bash
.venv/bin/python -m benchmarks.run --all --n-seeds 3 --n-resamples 10 --no-resume -v
```

Expected: eight lines of `<scenario>: results/runs/<scenario>/<hash>`, no traceback, no warning. This takes a few minutes, not hours.

- [ ] **Step 3: Check every manifest**

```bash
.venv/bin/python -u - <<'PY'
import json
from pathlib import Path

for manifest_path in sorted(Path("results/runs").glob("*/*/manifest.json")):
    if "ablation" in str(manifest_path):
        continue
    m = json.loads(manifest_path.read_text())
    print(
        f"{m['scenario']:26s} {m['status']:9s} {m['n_seeds']}x{m['n_resamples']} "
        f"n_jobs={m['n_jobs']:3d} anchors={m['anchor_set']}"
    )
PY
```

Expected: eight rows, every one `complete`, every one `3x10`, `anchors=CALIBRATED_ANCHORS`, and `n_jobs` of `1` for `gaussians_samples` and `gaussians_dimensionality` and `-1` for the other six. That last column is Task 9's whole point and this is the first place it is visible end to end.

- [ ] **Step 4: Regenerate every figure and table**

```bash
.venv/bin/python -u - <<'PY'
import warnings
warnings.simplefilter("error")
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

from benchmarks._artifacts import read_run, read_runtimes, widest_run
from benchmarks._registry import SCENARIOS
from benchmarks.figures import (
    figure_benchmarking_examples,
    figure_benchmarking_results,
    figure_scaling_ari,
    figure_scaling_runtime,
    figure_scenario_dashboard,
)
from benchmarks.tables import write_all_tables

ROOT = Path("results/runs")
results = {n: read_run(widest_run(ROOT, n)) for n in SCENARIOS}

for name, frame in results.items():
    fig = figure_scenario_dashboard(name, frame, save=True)
    empty = [i for i, ax in enumerate(fig.axes)
             if not (ax.get_lines() or ax.collections or ax.get_images())]
    assert not empty, f"{name}: empty panels at {empty}"
    print(f"{name:26s} {len(fig.axes)} panels, none empty")

difficulty = {k: v for k, v in results.items()
              if v["axis_name"].iloc[0] == "difficulty_level"}
scaling = {k: v for k, v in results.items()
           if v["axis_name"].iloc[0] != "difficulty_level"}
figure_benchmarking_examples()
figure_benchmarking_results(difficulty)
figure_scaling_ari(scaling)
figure_scaling_runtime({n: read_runtimes(widest_run(ROOT, n)) for n in scaling})
for p in write_all_tables(results, Path("vis/tables")):
    text = p.read_text()
    assert "nan" not in text, f"{p}: literal nan"
    assert text.count(r"\begin{tabular}") == 2, f"{p}: not a paired table"
    print(p)
PY
```

Expected: eight dashboard lines with no empty panels, four manuscript figures written, eight `.tex` paths. `warnings.simplefilter("error")` makes a new warning fail here as it would in the suite.

- [ ] **Step 5: Look at the figures**

```bash
open vis/benchmarking/scenario_dashboard_circles.png
open vis/benchmarking/scenario_dashboard_gaussians_samples.png
open vis/benchmarking/benchmarking_results.png
```

Check: the re-stepped hues are distinguishable, the four comparator dash patterns read apart at reference line width, the scaling dashboard's x axis says `n` and not `SNR`, and the heatmap annotations are legible against both ends of the ramp. Fix and re-render before moving on.

- [ ] **Step 6: Run the full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: PASS. Both legs — carve and benchmarks — must be green. Note the counts; the baseline before this branch was 1085 passed and 10 skipped on the benchmarks leg.

- [ ] **Step 7: Commit the acceptance evidence**

The artifacts and figures are gitignored, so this commit carries only whatever fixes Steps 4-6 turned up. If nothing needed fixing, skip it.

```bash
.venv/bin/ruff check src/
git add -A src/ tests/
git commit -m "fix(benchmarks): acceptance-run corrections

Found by running all eight scenarios at 3 datasets x B=10 and regenerating
every figure and table under warnings-as-errors.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 19: Write the re-run handoff

**Files:**
- Create: `docs/superpowers/notes/2026-09-22-rerun-handoff.md`

**Interfaces:**
- Consumes: Task 18's measured dev-scale timings.
- Produces: the note the author runs the publication benchmark from.

- [ ] **Step 1: Re-derive the cost from the dev run**

The spec's figures are projections from the old manifests. Task 18's run gives real per-cell timings on the new code, with swiss rolls on spectral and 500 trees everywhere.

```bash
.venv/bin/python -u - <<'PY'
import json
from pathlib import Path

import pandas as pd

from benchmarks._artifacts import read_runtimes, widest_run
from benchmarks._registry import SCENARIOS

ROOT = Path("results/runs")
print(f"{'scenario':26s} {'s/cell':>8s} {'cells':>6s} {'dev CPU-h':>10s} {'20x100 CPU-h':>13s}")
total = 0.0
for name in SCENARIOS:
    rd = widest_run(ROOT, name)
    manifest = json.loads((rd / "manifest.json").read_text())
    runtimes = read_runtimes(rd)
    per_cell = float(runtimes["t_default_s"].mean())
    cells = len(runtimes)
    dev_h = per_cell * cells / 3600
    # B and the dataset count both scale the cost linearly.
    scale = (20 / manifest["n_seeds"]) * (100 / manifest["n_resamples"])
    full_h = dev_h * scale
    total += full_h
    print(f"{name:26s} {per_cell:8.2f} {cells:6d} {dev_h:10.3f} {full_h:13.1f}")
print(f"{'TOTAL':26s} {'':8s} {'':6s} {'':10s} {total:13.1f}")
PY
```

The linear scaling in B is an approximation — the classifier's cost per resample is constant but the consensus matrix assembly is not — so present these as projections and say so.

- [ ] **Step 2: Write the note**

Create `docs/superpowers/notes/2026-09-22-rerun-handoff.md` covering:

- What changed since the artifacts were last produced: the eigensolver, swiss rolls' estimator, `n_trees`, the calibrated anchors. Each moves numbers, so nothing on disk before this branch is comparable.
- The staged validation the spec calls for: run `gaussians`, `t_dist` and `t_dist_noise` once against `PUBLISHED_ANCHORS` on the new code before switching `ACTIVE_ANCHORS`, so a disagreement with the committed CSVs beyond the `n_trees` change is a runner defect rather than a consequence of the redesign. Give the exact commands, including the one-line registry edit and how to revert it.
- The commands for the publication run, one per line, with the measured cost from Step 1 beside each:
  ```
  python -m benchmarks.run --scenario gaussians --no-resume
  python -m benchmarks.run --scenario t_dist --no-resume
  python -m benchmarks.run --scenario t_dist_noise --no-resume
  python -m benchmarks.run --scenario circles --no-resume
  python -m benchmarks.run --scenario moons --no-resume
  python -m benchmarks.run --scenario swiss_rolls --no-resume
  ```
  `--n-jobs` is deliberately absent: Task 9 resolves it per scenario, and passing it by hand is how the scaling scenarios' timings get corrupted.
- The two scaling scenarios, costed separately, with the note that they run serially and cannot share the machine with the rho/B ablation's publication run.
- What to do when it finishes: re-run the notebook, then copy the four figures and eight fragments into `overleaf/vis/` by hand, which stays a manual step.
- The manuscript edits the re-run forces, from the spec's "Consequences outside this repository" section. The circles and moons findings are the load-bearing one: their published gradient was solver non-convergence, so the sentence claiming CARVE beat the CVIs there has to be rewritten against the new numbers rather than carried over.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/notes/2026-09-22-rerun-handoff.md
git commit -m "docs(notes): the publication re-run handoff

Measured cost from the dev-scale acceptance run on the new code, the staged
validation against PUBLISHED_ANCHORS that separates a runner defect from
the redesign, and the manuscript edits the re-run forces.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Report**

Hand back: the branch name, the tasks completed, the full suite's pass count, the dashboard path for one scenario so the author can look at it, and the handoff note's path as the next thing they act on.

---

## Self-Review

Run after all nineteen tasks, against the spec.

**Spec coverage.** A1 Task 2 · A2 Task 1 · A3 Task 3 · A4 Tasks 4 and 5 · A5 Task 2 · A6 Task 2 · B1 Task 6 · B2 Task 7 · B3 Task 8 · B4 Task 9 · B5 Task 18 · C1 Tasks 11, 12, 13, 14 · C2 Task 14 · C3 Task 15 · C4 Task 15 · C5 Task 10 · D1 Task 17 · D2 Task 16 · E1 Task 18 · E2 Task 1 · E3 Tasks 11, 12, 13, 14, 17 · E4 Task 19.

**Interface consistency.** `widest_run` is defined in Task 8 and consumed in Tasks 15 and 18 under that name. `figure_scenario_dashboard` is defined in Task 14 and called as `scenario_dashboard` — the notebook's own thin wrapper, defined in Task 15's helper cell. `metric_dashes` returns `None` for CARVE and a tuple otherwise, and `metric_linestyle` is the only consumer. `SEQUENTIAL_CMAP_NAME` is defined in Task 10 and consumed in Task 11. `TABLE_ROW_GROUPS` is defined in Task 16 and consumed in Task 17. `synthetic_run_frame` is defined in Task 13 and reused in Tasks 14 and 17; it lives in `tests/benchmarks/_helpers.py` and is imported as `tests.benchmarks._helpers`, never via a `sys.path` insert of `tests/`, which would shadow the real `benchmarks` package.

**Ordering constraints.** Task 1 before 3 and 5 — the estimator comparison and the calibration are both only correct on the fixed solver. Task 4 before 5. Task 6 before 8 — `widest_run` reads `status`. Tasks 10 before 11 and 12. Tasks 11, 12, 13 before 14. Tasks 8 and 14 before 15. Task 16 before 17. Everything before 18.

**Known deviations, recorded not fixed.** `_theme` colors `ari_average*` orange and the three consensus metrics grey, both in the comparator family rather than CARVE's blue and green, and the greys collide with the oracle's. They reach the tables only. The two scaling scenarios keep their published anchors: `TARGET_ARI_BANDS` is keyed on easy/medium/hard and they sweep n and p.
