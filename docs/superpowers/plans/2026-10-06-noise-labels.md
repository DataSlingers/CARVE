# Noise Labels in get_labels Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: let `CARVE.get_labels(noise_labels=True)` return `-1` for the samples whose per-sample Gini, CE or accuracy score marks them as ambiguous.

Architecture: a pure function `noise_mask` in `_utils.py` decides which samples are noise from one score array. A private method `CARVE._sample_scores` looks up a configuration's score array, replacing four identical copies in the score plots. `get_labels` validates three new keyword arguments, scores the selected configuration before the consensus cut, and writes `-1` into a copy of the final labels after the alignment step.

Tech stack: Python 3.13, NumPy, pandas, scikit-learn, pytest.

Spec: `docs/superpowers/specs/2026-10-06-noise-labels-design.md` (commit 94b38ea). Executors read both.

## Global Constraints

- Branch: `noise-labels` in the `code/` repository. Commit after every task.
- The working tree carries the author's uncommitted edits to `notebooks/case_studies/Klein.ipynb`, `src/benchmarks/_studies.py`, `src/benchmarks/figures/_carve_output.py`, `tests/benchmarks/figures/test_carve_output.py`, `tests/benchmarks/test_notebooks.py` and `tests/benchmarks/test_studies.py`. Never stage, stash, reset, format or commit them. Stage files by explicit path only; never `git add -A`, `git add .` or `git commit -a`.
- Run every command from the repository root (`code/`). Commands are written for the main checkout (`.venv/bin/python`). In a worktree, use the main checkout's interpreter and prefix `PYTHONPATH=src`, so the worktree's sources shadow the editable install, which points at the main checkout's `src/`.
- Run tests in the foreground with a long timeout. `tests/benchmarks` alone takes about seven minutes.
- `filterwarnings = error` is on. A new warning fails the suite. Since pytest 8, `pytest.warns` re-emits every warning it did not match, so a test expecting the noise warning must match its exact text, and a test not expecting it must not trigger it.
- Lint and format with the pinned ruff in `.venv`, and only the `src/` files the task changed: `.venv/bin/ruff check --fix <file> && .venv/bin/ruff format <file>`. Never run `ruff format src/` or `ruff check --fix src/`: they would rewrite the author's uncommitted files under `src/benchmarks/`. The whole-tree checks run read-only: `.venv/bin/ruff check src/` and `.venv/bin/ruff format --check src/carve/`. Never run ruff on `tests/` or `notebooks/`.
- No `logging`. Diagnostics go through `warnings.warn(..., stacklevel=2)`.
- `_utils.py` is a leaf: it must not import from `carve.api` or any module that does.
- `config_id` is a join key into the per-configuration containers, never a row position.
- Tests mirror modules one to one: `_utils.py` with `tests/test_utils.py`, `api.py` with `tests/test_api.py`.
- Python only. Do not touch `carve-r/`, `src/carve/tl/`, `src/carve/pl/`, `src/carve/_plotting.py` or `src/benchmarks/`. Do not write to `../overleaf/`.
- Exact values from the spec: margin `NOISE_MARGIN = 0.05`; defaults `noise_labels=False`, `noise_quantile=0.05`, `noise_score="gini"`; error text `"noise_score must be one of: 'gini', 'ce', 'accuracy'."`; warning text `noise_quantile=<q> asks for about <n_target> of <n_scored> samples by <score>; <n_flagged> were flagged. Samples tied at the cutoff (<cutoff:.3f>) or within 0.05 of the median (<median:.3f>) stay labeled.`
- Documents and comments: American spelling, plain prose, no bold or italics.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Refinements to the spec

Found while writing this plan; each is reflected in the tasks.

1. The spec's step order computes the scores after the alignment step. `get_labels` instead looks up the scores and computes the mask right after `_select_row`, before the consensus cut, and applies the mask after the alignment step. The rule and the output are unchanged; a missing score or an all-NaN score row now raises before `reference_labels` changes, so a failed call leaves the model as it was.
2. `get_labels` currently ends with `return np.asarray(labels, dtype=np.int32)`. That is a copy only because the cut labels are int64 (checked on 2026-10-06: `reference_labels.dtype` is int64, the return is int32). Writing `-1` into an array that might share memory with `reference_labels` would corrupt the reference, so the new code makes an explicit copy with `np.array(labels, dtype=np.int32)`.
3. The rule is `score < median - 0.05`, so a flagged sample scores more than 0.05 below the median. The spec's prose says "at least"; the formula is authoritative and the docstrings say "more than".
4. The spec's API tests name the module-scoped `fitted_carve`. It has one configuration, so the "reversed pattern at every other configuration" check is impossible on it. The tests use `fitted_identity_single` instead: three KMeans configurations (k = 2, 3, 4) on 90 samples in three blobs (rows 0-29, 30-59, 60-89). Checked on 2026-10-06: the k=3 configuration has `config_id` 1 and sits at position 2 after `TestRowIdentity._reindexed`.
5. The argument checks are tested on an unfitted `CARVE`, which proves they run before the fit check and before configuration selection, and leaves the shared fixtures untouched.
6. Added a test the spec does not list: `noise_score="accuracy"` on the stability-only fit raises the generalizability `RuntimeError`.

## File structure

- `src/carve/_utils.py`: gains `NOISE_MARGIN`, `NoiseCut` and `noise_mask` (Task 1).
- `src/carve/api.py`: gains `CARVE._sample_scores`; the four score plots call it (Task 2). `get_labels` gains the three arguments (Task 3).
- `tests/test_utils.py`: gains `TestNoiseMask` (Task 1).
- `tests/test_api.py`: gains the `_set_scores` helper and `TestSampleScores` (Task 2), and `TestNoiseLabels` with its helpers (Task 3).

---

### Task 1: noise_mask in _utils

Files:
- Modify: `src/carve/_utils.py` (the `typing` import at line 9; new code after `align_cluster_labels`, which ends just before `def ensure_2d_array`)
- Test: `tests/test_utils.py` (import block at lines 17-31; new class appended at the end of the file)

Interfaces:
- Consumes: nothing.
- Produces: `NOISE_MARGIN: float = 0.05`; `class NoiseCut(NamedTuple)` with fields `mask: np.ndarray` (bool, shape (n_samples,)), `n_target: int`, `cutoff: float`, `median: float`; `noise_mask(scores: np.ndarray, *, quantile: float, margin: float = NOISE_MARGIN) -> NoiseCut`. Raises `ValueError("scores contains no finite value.")` when no score is finite. Emits no warning.

- [ ] Step 1: Write the failing tests

In `tests/test_utils.py`, extend the `from carve._utils import (...)` block so it reads:

```python
from carve._utils import (
    NOISE_MARGIN,
    _coerce_n_clusters,
    _summarize_ari_scores,
    align_cluster_labels,
    apply_noise_policy,
    cluster_labels,
    count_clusters,
    default_generalizability_classifier,
    ensure_2d_array,
    noise_mask,
    resolve_anchors,
    resolve_core_budget,
    scale_neighbor_count,
    split_subsample_indices,
    summarize_preprocessing_records,
)
```

Append to the end of `tests/test_utils.py`:

```python
# -----------------------------------------------------------------------
# noise_mask
# -----------------------------------------------------------------------


class TestNoiseMask:
    def test_wide_spread_flags_the_plain_quantile(self):
        scores = np.linspace(0.0, 1.0, 101)
        cut = noise_mask(scores, quantile=0.05)
        # The 0.05 quantile is exactly scores[5]; the strict comparison keeps
        # it labeled, so a <= rule would flag six samples here.
        np.testing.assert_array_equal(np.flatnonzero(cut.mask), [0, 1, 2, 3, 4])
        assert cut.n_target == 5
        assert cut.cutoff == pytest.approx(0.05)
        assert cut.median == pytest.approx(0.5)

    def test_equal_scores_flag_nothing(self):
        cut = noise_mask(np.full(50, 0.8), quantile=0.05)
        assert not cut.mask.any()
        assert cut.n_target == 3

    def test_margin_spares_a_bulk_just_under_one(self):
        bulk = np.linspace(0.995, 1.0, 95)
        low = np.array([0.4, 0.6, 0.4, 0.6, 0.4])
        cut = noise_mask(np.concatenate([bulk, low]), quantile=0.10)
        # Without the margin the quantile would also take five bulk samples.
        np.testing.assert_array_equal(np.flatnonzero(cut.mask), [95, 96, 97, 98, 99])
        assert cut.n_target == 10

    def test_margin_is_anchored_at_the_median_not_the_maximum(self):
        scores = np.concatenate([np.linspace(0.70, 0.72, 99), [1.0]])
        cut = noise_mask(scores, quantile=0.05)
        assert not cut.mask.any()
        # Non-vacuity: a margin anchored at the maximum flags five here.
        at_max = (scores < cut.cutoff) & (scores < scores.max() - NOISE_MARGIN)
        assert at_max.sum() == 5

    def test_ties_at_the_cutoff_stay_labeled(self):
        scores = np.concatenate([np.ones(90), np.full(6, 0.9), np.full(4, 0.5)])
        cut = noise_mask(scores, quantile=0.05)
        np.testing.assert_array_equal(np.flatnonzero(cut.mask), [96, 97, 98, 99])
        assert cut.cutoff == pytest.approx(0.9)
        assert cut.n_target == 5

    def test_nan_scores_are_flagged_and_left_out_of_the_statistics(self):
        clean = np.linspace(0.0, 1.0, 101)
        scores = np.concatenate([clean[:50], [np.nan], clean[50:], [np.nan]])
        cut = noise_mask(scores, quantile=0.05)
        reference = noise_mask(clean, quantile=0.05)
        np.testing.assert_array_equal(
            np.flatnonzero(cut.mask), [0, 1, 2, 3, 4, 50, 102]
        )
        assert cut.cutoff == reference.cutoff
        assert cut.median == reference.median
        assert cut.n_target == reference.n_target

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="scores contains no finite value"):
            noise_mask(np.full(5, np.nan), quantile=0.05)
```

- [ ] Step 2: Run them to verify they fail

Run: `.venv/bin/python -m pytest tests/test_utils.py -q -k TestNoiseMask`
Expected: collection error, `ImportError: cannot import name 'NOISE_MARGIN' from 'carve._utils'`.

- [ ] Step 3: Implement

In `src/carve/_utils.py`, change `from typing import Any` to:

```python
from typing import Any, NamedTuple
```

Insert after the end of `align_cluster_labels` (after its `return aligned`), before `def ensure_2d_array`:

```python
# How far below the median sample's score, on the [0, 1] score scale, a
# sample must lie before noise_mask may flag it.
NOISE_MARGIN = 0.05


class NoiseCut(NamedTuple):
    """Result of :func:`noise_mask`."""

    mask: np.ndarray
    n_target: int
    cutoff: float
    median: float


def noise_mask(
    scores: np.ndarray, *, quantile: float, margin: float = NOISE_MARGIN
) -> NoiseCut:
    """Flag the samples whose per-sample score marks them as ambiguous.

    A sample is flagged when its score is NaN, or when it lies strictly
    below the ``quantile`` of the finite scores and more than ``margin``
    below their median. The strict comparison keeps samples tied with the
    cutoff, so identical scores flag nothing. The margin keeps samples that
    score practically like the median one, which a plain quantile would
    flag whenever nearly every score sits just under 1.

    Parameters
    ----------
    scores : ndarray of shape (n_samples,)
        Per-sample scores on [0, 1], higher meaning more stable.
    quantile : float
        Fraction of the samples considered for noise, strictly between 0
        and 1. The caller validates it.
    margin : float, default=NOISE_MARGIN
        Distance below the median a flagged score must exceed.

    Returns
    -------
    NoiseCut
        ``mask``, True for flagged samples, NaN ones included. ``n_target``,
        the number of finite samples the quantile would flag if all finite
        scores were distinct. ``cutoff`` and ``median``, over the finite
        scores.

    Raises
    ------
    ValueError
        If no score is finite.
    """
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("scores contains no finite value.")

    values = scores[finite]
    cutoff = float(np.quantile(values, quantile))
    median = float(np.median(values))

    mask = ~finite
    mask[finite] = (values < cutoff) & (values < median - margin)

    # Counted through np.quantile itself, so it matches the cutoff's own
    # floating-point position exactly.
    ranks = np.arange(values.size)
    n_target = int((ranks < np.quantile(ranks, quantile)).sum())

    return NoiseCut(mask=mask, n_target=n_target, cutoff=cutoff, median=median)
```

- [ ] Step 4: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_utils.py -q`
Expected: all pass, `TestNoiseMask` included.

- [ ] Step 5: Mutation checks

Apply each edit to `noise_mask`, run `.venv/bin/python -m pytest tests/test_utils.py -q -k TestNoiseMask`, confirm the named test fails, then revert the edit before the next one. Confirm with `git diff src/carve/_utils.py` that only the Step 3 code remains.

| Edit | Test that must fail |
| --- | --- |
| `values < cutoff` becomes `values <= cutoff` | `test_wide_spread_flags_the_plain_quantile` |
| `& (values < median - margin)` removed | `test_margin_spares_a_bulk_just_under_one` |
| `median - margin` becomes `values.max() - margin` | `test_margin_is_anchored_at_the_median_not_the_maximum` |
| `mask = ~finite` becomes `mask = np.zeros_like(finite)` | `test_nan_scores_are_flagged_and_left_out_of_the_statistics` |

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff check --fix src/carve/_utils.py && .venv/bin/ruff format src/carve/_utils.py
.venv/bin/ruff check src/ && .venv/bin/ruff format --check src/carve/
git status --short
git add src/carve/_utils.py tests/test_utils.py
git commit -m "$(cat <<'EOF'
feat(utils): add noise_mask to flag ambiguous samples by score

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

`git status --short` must show the author's six files still modified and unstaged, and nothing else besides the two task files.

---

### Task 2: one per-sample score lookup

Files:
- Modify: `src/carve/api.py` (new method after `_select_row`, which ends just before `def get_labels`; the four `# --- Resolve score source ---` blocks in `plot_cluster_boxplot`, `plot_cluster_violin`, `plot_cluster_scatter` and `plot_diagnostic_scatter`)
- Test: `tests/test_api.py` (appended at the end of the file)

Interfaces:
- Consumes: nothing from Task 1.
- Produces: `CARVE._sample_scores(self, source: str, config_id: int) -> np.ndarray`, float array of shape (n_samples,). Raises `RuntimeError("Gini stability scores are not available for this run.")`, `RuntimeError("CE stability scores are not available for this run.")`, `RuntimeError("Generalizability scores are not available for this run.")`, and `ValueError("source must be one of: 'accuracy', 'gini', 'ce'.")`. In `tests/test_api.py`: module-level helper `_set_scores(model, source, config_id, row)` that writes one configuration's scores in place; Task 3 uses it.

- [ ] Step 1: Write the failing tests

Append to the end of `tests/test_api.py`:

```python
# ---------------------------------------------------------------------------
# Per-sample scores: _sample_scores and get_labels(noise_labels=True)
# ---------------------------------------------------------------------------


def _set_scores(model, source, config_id, row):
    """Write one configuration's per-sample ``source`` scores in place."""
    if source == "gini":
        model.stability_gini_scores_[config_id] = row
    elif source == "ce":
        model.stability_ce_scores_[config_id] = row
    elif source == "accuracy":
        model.generalizability_scores_[config_id] = row
    else:
        raise AssertionError(f"unknown source {source!r}")


class TestSampleScores:
    @pytest.mark.parametrize("source", ["gini", "ce", "accuracy"])
    def test_returns_the_entry_at_config_id(self, fitted_identity, source):
        # Fitted scores can coincide across configurations (accuracy is often
        # 1.0 everywhere on separated blobs), which would hide a wrong lookup.
        # Each configuration gets its own constant instead.
        model = copy.deepcopy(fitted_identity)
        n_configs = len(model.consensus_matrices_)
        n_samples = model.X_.shape[0]
        for cid in range(n_configs):
            _set_scores(model, source, cid, np.full(n_samples, cid / n_configs))
        for cid in range(n_configs):
            np.testing.assert_array_equal(
                model._sample_scores(source, cid), np.full(n_samples, cid / n_configs)
            )

    def test_returns_floats(self, fitted_identity):
        scores = fitted_identity._sample_scores("accuracy", 0)
        assert scores.dtype == np.float64
        assert scores.shape == (fitted_identity.X_.shape[0],)

    def test_unknown_source_raises(self, fitted_identity):
        with pytest.raises(ValueError, match="source must be one of"):
            fitted_identity._sample_scores("nope", 0)
```

- [ ] Step 2: Run them to verify they fail

Run: `.venv/bin/python -m pytest tests/test_api.py -q -k TestSampleScores`
Expected: FAIL with `AttributeError: 'CARVE' object has no attribute '_sample_scores'`.

- [ ] Step 3: Implement

In `src/carve/api.py`, insert after the end of `_select_row` (after `return row, config_id_of(row), observed_k(row), pin is not None`), before `def get_labels`:

```python
    def _sample_scores(self, source: str, config_id: int) -> np.ndarray:
        """Per-sample scores of one configuration.

        Parameters
        ----------
        source : {"gini", "ce", "accuracy"}
            Gini stability, cross-entropy stability, or out-of-sample
            accuracy.
        config_id : int
            Key into the per-configuration containers, as returned by
            ``_select_row``. A join key, never a row position.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Scores on [0, 1], higher meaning more stable or more accurate.
            Gini and CE are NaN for a sample never co-sampled with any
            partner; accuracy is 0 for a sample never held out.

        Raises
        ------
        RuntimeError
            If this fit did not compute the requested scores.
        ValueError
            If ``source`` is not one of the three.
        """
        if source == "gini":
            if self.stability_gini_scores_ is None:
                raise RuntimeError(
                    "Gini stability scores are not available for this run."
                )
            return np.asarray(self.stability_gini_scores_[config_id], dtype=float)

        if source == "ce":
            if self.stability_ce_scores_ is None:
                raise RuntimeError(
                    "CE stability scores are not available for this run."
                )
            return np.asarray(self.stability_ce_scores_[config_id], dtype=float)

        if source == "accuracy":
            # A stability-only fit leaves a list of None, not None itself.
            if (
                self.generalizability_scores_ is None
                or self.generalizability_scores_[config_id] is None
            ):
                raise RuntimeError(
                    "Generalizability scores are not available for this run."
                )
            return np.asarray(self.generalizability_scores_[config_id], dtype=float)

        raise ValueError("source must be one of: 'accuracy', 'gini', 'ce'.")
```

In `plot_cluster_boxplot` and in `plot_cluster_violin`, replace the whole `# --- Resolve score source ---` block (from that comment through `raise ValueError("source must be one of: 'accuracy', 'gini', 'ce'.")`, the block that assigns `default_ylabel`) with:

```python
        # --- Resolve score source ---
        scores = self._sample_scores(source, config_id)
        default_ylabel = {
            "gini": "Cluster Stability (Gini)",
            "ce": "Cluster Stability (CE)",
            "accuracy": "Cluster Generalizability",
        }[source]
```

In `plot_cluster_scatter` and in `plot_diagnostic_scatter`, replace the same block (the one that assigns `scores_name`) with:

```python
        # --- Resolve score source ---
        scores = self._sample_scores(source, config_id)
        scores_name = {
            "gini": "Gini Stability",
            "ce": "CE Stability",
            "accuracy": "Generalizability",
        }[source]
```

`_sample_scores` raises for an unknown source before the dictionary lookup, so the existing `ValueError` contract holds. Afterwards `grep -n "scores are not available" src/carve/api.py` must list exactly the three lines inside `_sample_scores`.

- [ ] Step 4: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_api.py tests/test_pl.py -q`
Expected: all pass. `TestErrorContracts::test_source_and_mode_are_validated`, `test_stability_scores_are_reported_missing_after_a_generalizability_fit` and `test_generalizability_scores_are_reported_missing_after_a_stability_fit` cover the refactored plots' error contracts; `TestPlotting` covers their drawing.

- [ ] Step 5: Mutation check

Change each `[config_id]` inside `_sample_scores` to `[0]`, run `.venv/bin/python -m pytest tests/test_api.py -q -k TestSampleScores`, confirm `test_returns_the_entry_at_config_id` fails for all three sources, and revert.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff check --fix src/carve/api.py && .venv/bin/ruff format src/carve/api.py
.venv/bin/ruff check src/ && .venv/bin/ruff format --check src/carve/
git status --short
git add src/carve/api.py tests/test_api.py
git commit -m "$(cat <<'EOF'
refactor(api): share one per-sample score lookup across the score plots

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: noise arguments on get_labels

Files:
- Modify: `src/carve/api.py` (the `from ._utils import (...)` block at lines 63-70; `get_labels`)
- Test: `tests/test_api.py` (the import block at the top; new code appended after `TestSampleScores`)

Interfaces:
- Consumes: `NOISE_MARGIN`, `noise_mask(scores, *, quantile) -> NoiseCut` from Task 1; `CARVE._sample_scores(source, config_id)` and the test helper `_set_scores(model, source, config_id, row)` from Task 2; the existing fixtures `fitted_identity_single` and `split_mode_fits`, the existing `TestRowIdentity._reindexed` static method, and the existing module helpers `_blobs(n)` and `_grids()` in `tests/test_api.py`.
- Produces: `CARVE.get_labels(..., noise_labels: bool = False, noise_quantile: float = 0.05, noise_score: Literal["gini", "ce", "accuracy"] = "gini") -> np.ndarray`.

- [ ] Step 1: Write the failing tests

In `tests/test_api.py`, add `import re` to the standard-library imports at the top, so they read:

```python
import copy
import re
import warnings
```

Append to the end of `tests/test_api.py`, after `TestSampleScores`:

```python
# Five samples spread over the three blobs of fitted_identity_single (rows
# 0-29, 30-59, 60-89), so the noise positions are not one contiguous block.
NOISE_LOW = np.array([3, 17, 41, 58, 84])


def _known_scores(n, low):
    """Distinct scores on [0.5, 1.0] whose lowest ``len(low)`` sit at ``low``.

    At quantile 0.05 the rule flags exactly ``low`` as long as
    ``len(low) == ceil(0.05 * (n - 1))``: 5 for n=90, 3 for n=60. The
    median, 0.75, lies far above the margin, so no warning fires.
    """
    values = np.linspace(0.5, 1.0, n)
    scores = np.empty(n)
    rest = np.setdiff1d(np.arange(n), low)
    scores[low] = values[: low.size]
    scores[rest] = values[low.size :]
    return scores


def _with_scores(model, source, config_id, scores):
    """Deep copy of ``model`` with ``scores`` at ``config_id``.

    Every other configuration gets the reversed pattern, whose low samples
    sit elsewhere, so reading the wrong configuration moves the noise.
    """
    model = copy.deepcopy(model)
    model.reference_labels = None
    for cid in range(len(model.consensus_matrices_)):
        row = scores if cid == config_id else scores[::-1].copy()
        _set_scores(model, source, cid, row)
    return model


def _config_id(model, **pin):
    _, config_id, _, _ = model._select_row(measure="stability", rule="1se", **pin)
    return config_id


class TestNoiseLabels:
    def test_off_by_default(self, fitted_identity_single):
        model = copy.deepcopy(fitted_identity_single)
        model.reference_labels = None
        plain = model.get_labels(k=3)
        off = model.get_labels(k=3, noise_labels=False)
        np.testing.assert_array_equal(off, plain)
        assert not (plain == -1).any()

    @pytest.mark.parametrize("source", ["gini", "ce", "accuracy"])
    def test_noise_lands_on_the_known_low_samples(self, fitted_identity_single, source):
        cid = _config_id(fitted_identity_single, k=3)
        model = _with_scores(
            fitted_identity_single, source, cid, _known_scores(90, NOISE_LOW)
        )
        plain = model.get_labels(k=3)
        noisy = model.get_labels(k=3, noise_labels=True, noise_score=source)
        np.testing.assert_array_equal(np.flatnonzero(noisy == -1), NOISE_LOW)
        keep = noisy != -1
        np.testing.assert_array_equal(noisy[keep], plain[keep])
        assert noisy.dtype == np.int32

    def test_scores_are_joined_on_config_id(self, fitted_identity_single):
        cid = _config_id(fitted_identity_single, k=3)
        model = _with_scores(
            fitted_identity_single, "gini", cid, _known_scores(90, NOISE_LOW)
        )
        model.estimator_results_ = TestRowIdentity._reindexed(
            model.estimator_results_
        )
        position = int(
            np.flatnonzero(model.estimator_results_["config_id"].to_numpy() == cid)[0]
        )
        assert position != cid  # non-vacuity: row position and config_id differ
        noisy = model.get_labels(k=3, noise_labels=True)
        np.testing.assert_array_equal(np.flatnonzero(noisy == -1), NOISE_LOW)

    def test_reference_labels_stay_free_of_noise(self, fitted_identity_single):
        cid = _config_id(fitted_identity_single, k=3)
        model = _with_scores(
            fitted_identity_single, "gini", cid, _known_scores(90, NOISE_LOW)
        )
        noisy = model.get_labels(k=3, noise_labels=True)
        assert not (model.reference_labels == -1).any()
        plain = model.get_labels(k=3)
        assert not (plain == -1).any()
        keep = noisy != -1
        np.testing.assert_array_equal(plain[keep], noisy[keep])

    def test_ties_at_the_cutoff_warn_with_the_counts(self, fitted_identity_single):
        cid = _config_id(fitted_identity_single, k=3)
        scores = np.ones(90)
        scores[[3, 41, 84]] = 0.5
        scores[[10, 25, 35, 50, 65, 80]] = 0.9
        model = _with_scores(fitted_identity_single, "gini", cid, scores)
        message = (
            "noise_quantile=0.05 asks for about 5 of 90 samples by gini; 3 were "
            "flagged. Samples tied at the cutoff (0.900) or within 0.05 of the "
            "median (1.000) stay labeled."
        )
        with pytest.warns(UserWarning, match=re.escape(message)):
            noisy = model.get_labels(k=3, noise_labels=True)
        np.testing.assert_array_equal(np.flatnonzero(noisy == -1), [3, 41, 84])

    @pytest.mark.parametrize("noise_labels", [False, True])
    def test_unknown_noise_score_raises_before_the_fit_check(self, noise_labels):
        # An unfitted model would raise "Call fit() first." from any later check.
        with pytest.raises(ValueError, match="noise_score must be one of"):
            CARVE(verbose=0).get_labels(noise_labels=noise_labels, noise_score="nope")

    @pytest.mark.parametrize("noise_labels", [False, True])
    @pytest.mark.parametrize("quantile", [0.0, 1.0, 1.5])
    def test_quantile_outside_the_open_unit_interval_raises(
        self, noise_labels, quantile
    ):
        with pytest.raises(
            ValueError, match="noise_quantile must be strictly between 0 and 1"
        ):
            CARVE(verbose=0).get_labels(
                noise_labels=noise_labels, noise_quantile=quantile
            )

    def test_missing_stability_scores_raise(self, split_mode_fits):
        generalizability_only = copy.deepcopy(split_mode_fits[1])
        generalizability_only.reference_labels = None
        with pytest.raises(
            RuntimeError, match="Gini stability scores are not available"
        ):
            generalizability_only.get_labels(
                measure="generalizability",
                rule="max",
                mode="generalizability",
                noise_labels=True,
                noise_score="gini",
            )
        # Scored before the cut, so the failed call left no reference behind.
        assert generalizability_only.reference_labels is None

    def test_missing_accuracy_scores_raise(self, split_mode_fits):
        stability_only = copy.deepcopy(split_mode_fits[0])
        with pytest.raises(
            RuntimeError, match="Generalizability scores are not available"
        ):
            stability_only.get_labels(noise_labels=True, noise_score="accuracy")

    def test_anchored_run_flags_non_anchor_samples(self):
        X = _blobs(60)
        model = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning, match="anchored consensus"):
            model.fit(X)
        low = np.setdiff1d(np.arange(60), model.consensus_anchors_)[:3]
        cid = _config_id(model, k=2)
        model = _with_scores(model, "gini", cid, _known_scores(60, low))
        noisy = model.get_labels(k=2, noise_labels=True)
        assert noisy.shape == (60,)
        np.testing.assert_array_equal(np.flatnonzero(noisy == -1), low)
```

- [ ] Step 2: Run them to verify they fail

Run: `.venv/bin/python -m pytest tests/test_api.py -q -k TestNoiseLabels`
Expected: every test fails with `TypeError: CARVE.get_labels() got an unexpected keyword argument` naming `noise_labels`, `noise_score` or `noise_quantile`. The validation tests fail the same way: `pytest.raises(ValueError)` does not catch a `TypeError`.

- [ ] Step 3: Implement

In `src/carve/api.py`, extend the `_utils` import block to:

```python
from ._utils import (
    NOISE_MARGIN,
    align_cluster_labels,
    default_generalizability_classifier,
    ensure_2d_array,
    noise_mask,
    resolve_anchors,
    resolve_core_budget,
    summarize_preprocessing_records,
)
```

Replace the `get_labels` signature with:

```python
    def get_labels(
        self,
        *,
        measure: str = "stability",
        rule: str = "1se",
        k: int | None = None,
        sweep_value: float | None = None,
        consensus_k: int | None = None,
        not_two: bool = False,
        mode: Literal["default", "generalizability"] = "default",
        estimator: ClusterMixin | None = None,
        noise_labels: bool = False,
        noise_quantile: float = 0.05,
        noise_score: Literal["gini", "ce", "accuracy"] = "gini",
    ) -> np.ndarray:
```

In the docstring, replace everything from the line `estimator : ClusterMixin or None, default=None` through the closing `"""` with the following. The `estimator` entry is unchanged; the rest is new or extended.

```
        estimator : ClusterMixin or None, default=None
            If provided, uses this estimator to cluster the consensus
            distance matrix; otherwise defaults to average-linkage
            ``AgglomerativeClustering`` with precomputed distances.
        noise_labels : bool, default=False
            If True, label ambiguous samples ``-1``, the label scikit-learn
            and HDBSCAN use for noise. A sample is ambiguous when its
            ``noise_score`` at the selected configuration is NaN, or when the
            score lies strictly below the ``noise_quantile`` of the scores
            and more than 0.05 below their median. Samples tied with the
            quantile stay labeled, so identical scores flag no sample, and
            the 0.05 margin keeps samples that score practically like the
            median one. The scores belong to the selected configuration and
            do not depend on ``consensus_k`` or ``estimator``. Every other
            sample keeps the label a call without noise returns, and a
            cluster can lose every member to noise.
        noise_quantile : float, default=0.05
            Fraction of the samples considered for noise, strictly between 0
            and 1. The default considers the least stable 5 percent. Checked
            on every call, also when ``noise_labels`` is False.
        noise_score : {"gini", "ce", "accuracy"}, default="gini"
            Per-sample score that ranks the samples: Gini stability
            (``stability_gini_scores_``), cross-entropy stability
            (``stability_ce_scores_``) or out-of-sample accuracy
            (``generalizability_scores_``). Gini and CE are NaN for a sample
            never co-sampled with any partner, which then becomes noise.
            Accuracy is 0 for a sample never held out, which can then become
            noise. Checked on every call, also when ``noise_labels`` is
            False.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Clustering labels derived from the selected consensus matrix.
            With ``noise_labels=True``, ambiguous samples are ``-1``.

        Raises
        ------
        RuntimeError
            If the instance has not been fitted yet, if the required
            consensus matrix is not available, or if ``noise_labels`` is
            True and this fit did not compute ``noise_score``.
        ValueError
            If no configurations match the given *k*, if ``noise_score`` or
            ``noise_quantile`` is invalid, or if ``noise_labels`` is True and
            every ``noise_score`` of the selected configuration is NaN.

        Warns
        -----
        UserWarning
            If ``noise_labels`` is True and fewer samples are flagged than
            the quantile asks for, because of ties at the quantile, the 0.05
            margin, or both. The message gives both counts, the quantile
            value and the median.
        """
```

Directly after the docstring, before `policy = resolve_mode(mode)`, insert:

```python
        if noise_score not in ("gini", "ce", "accuracy"):
            raise ValueError("noise_score must be one of: 'gini', 'ce', 'accuracy'.")
        if not 0.0 < noise_quantile < 1.0:
            raise ValueError(
                "noise_quantile must be strictly between 0 and 1, got "
                f"{noise_quantile!r}."
            )

```

Directly after the `row, config_id, selected_k, _ = self._select_row(...)` call, before the `# In resolution mode number of clusters is an outcome;` comment, insert:

```python
        # Scored before the cut, so a missing score or an all-NaN score row
        # raises before reference_labels changes.
        noise_scores = None
        noise_cut = None
        if noise_labels:
            noise_scores = self._sample_scores(noise_score, config_id)
            noise_cut = noise_mask(noise_scores, quantile=noise_quantile)

```

Replace the final line of `get_labels`, `return np.asarray(labels, dtype=np.int32)`, with:

```python
        # np.array, not np.asarray: when no alignment ran, reference_labels is
        # the same array, and the noise below must not reach it.
        labels = np.array(labels, dtype=np.int32)
        if noise_cut is None:
            return labels

        labels[noise_cut.mask] = -1

        scored = np.isfinite(noise_scores)
        n_flagged = int(np.count_nonzero(noise_cut.mask & scored))
        if n_flagged < noise_cut.n_target:
            warnings.warn(
                f"noise_quantile={noise_quantile} asks for about "
                f"{noise_cut.n_target} of {int(scored.sum())} samples by "
                f"{noise_score}; {n_flagged} were flagged. Samples tied at the "
                f"cutoff ({noise_cut.cutoff:.3f}) or within {NOISE_MARGIN} of "
                f"the median ({noise_cut.median:.3f}) stay labeled.",
                UserWarning,
                stacklevel=2,
            )
        return labels
```

`warnings` is already imported at the top of `api.py`.

- [ ] Step 4: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_api.py -q -k "TestNoiseLabels or TestGetLabels or TestReferenceLabels or TestAnchoredLabels"`
Expected: all pass, with no warnings reported beyond the ones the tests assert.

- [ ] Step 5: Mutation checks

Apply each edit to `get_labels`, run `.venv/bin/python -m pytest tests/test_api.py -q -k TestNoiseLabels`, confirm the named test fails, then revert. Confirm with `git diff src/carve/api.py` that only the Step 3 code remains.

| Edit | Test that must fail |
| --- | --- |
| The two argument checks wrapped in `if noise_labels:` | `test_unknown_noise_score_raises_before_the_fit_check[False]`, `test_quantile_outside_the_open_unit_interval_raises[...-False]` |
| The two argument checks moved below the `raise RuntimeError("Call fit() first.")` block | `test_unknown_noise_score_raises_before_the_fit_check` (both cases) |
| `self._sample_scores(noise_score, config_id)` becomes `self._sample_scores(noise_score, 0)` | `test_noise_lands_on_the_known_low_samples`, `test_scores_are_joined_on_config_id` |
| `self._sample_scores(noise_score, config_id)` becomes `self._sample_scores("gini", config_id)` | `test_noise_lands_on_the_known_low_samples[ce]`, `[accuracy]` |
| An extra `if noise_cut is not None: labels[noise_cut.mask] = -1` inserted directly after `labels = estimator.fit_predict(D)`, so the reference block sees the noise | `test_reference_labels_stay_free_of_noise` |
| `n_flagged < noise_cut.n_target` becomes `n_flagged > noise_cut.n_target` | `test_ties_at_the_cutoff_warn_with_the_counts` |
| `n_flagged < noise_cut.n_target` becomes `n_flagged <= noise_cut.n_target` | `test_noise_lands_on_the_known_low_samples` (the unexpected warning is an error) |

- [ ] Step 6: Run the whole suite

Run: `.venv/bin/python -m pytest -q` (foreground, timeout of at least 20 minutes).
Expected: all pass. No existing caller passes the new arguments, so no existing test changes behavior.

- [ ] Step 7: Lint and commit

```bash
.venv/bin/ruff check --fix src/carve/api.py && .venv/bin/ruff format src/carve/api.py
.venv/bin/ruff check src/ && .venv/bin/ruff format --check src/carve/
git status --short
git add src/carve/api.py tests/test_api.py
git commit -m "$(cat <<'EOF'
feat(api): label ambiguous samples -1 in get_labels

get_labels(noise_labels=True) flags the samples whose Gini, CE or accuracy
score at the selected configuration lies strictly below noise_quantile and
more than 0.05 below the median, plus samples with no score. A warning
reports when ties or the margin leave fewer flagged than the quantile asks.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```
