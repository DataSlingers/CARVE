# Anchored Consensus Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: let CARVE run past its current ceiling of roughly 15,000 to 20,000 samples by computing consensus quantities over a fixed anchor subset, with no behavior change at or below a configurable threshold.

Architecture: an anchor index is drawn once per `fit` and shared across every configuration. Consensus matrices become `m x m` anchor blocks; per-sample stability scores stay length `n` by streaming an `n x m` slab in row chunks and reducing it immediately; `get_labels` cuts the anchor block and extends the remaining labels with the classifier CARVE already uses for generalizability. Below the threshold nothing changes and the exact code path runs unaltered.

Tech Stack: Python 3.12, NumPy, SciPy, scikit-learn, joblib, pytest. No new dependencies.

Spec: `docs/superpowers/specs/2026-09-07-atac-and-large-scale-case-studies-design.md` (sections 3 and 4 are the ones this plan implements)

## Global Constraints

- Dependency direction: `_types`, `_utils`, `_anndata` and `cluster` are leaves. Nothing under `_*` may import `api`. Do not introduce upward imports.
- Tests mirror modules one to one: `_consensus.py` maps to `tests/test_consensus.py`, `_utils.py` to `tests/test_utils.py`, `_runner.py` to `tests/test_runner.py`, `api.py` to `tests/test_api.py`, `_types.py` to `tests/test_types.py`.
- No `logging` anywhere. Diagnostics go through `warnings.warn(..., stacklevel=2)` and `print()` gated on `verbose`.
- `config_id` is a join key, never a positional index. `fit` asserts alignment and raises `RuntimeError("config_id is misaligned ...")`. Anchored artifact lists must align with `estimator_results_` exactly as the exact ones do.
- Seeds are derived arithmetically and never shared. No global seeding.
- `SweepSpec` is frozen: one run sweeps exactly one parameter.
- Lint and format with the pinned ruff: `ruff check src/` and `ruff format src/`, version `==0.16.4` from the `dev` extra. Never a system ruff. Do not run ruff on `tests/` or `notebooks/`.
- The venv is uv-managed and has no pip. Install with `VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"`.
- Run tests in the foreground with a long timeout. The full suite takes 13 to 22 minutes; `tests/` alone for this package is faster. A slow run is not a hang.
- American spelling in prose. No bold or italics in authored text, including code comments and docstrings.
- Out of scope for this plan: the R port in `carve-r/` (deliberately deferred, tracked separately), sparse-native input, and changing the default classifier.

All commands are run from `/Users/kaiwycik/GitHub/CARVE/code`.

---

### Task 1: Anchor resolution

Files:
- Modify: `src/carve/_utils.py` (append a new function; it is a leaf module and must stay one)
- Test: `tests/test_utils.py`

Interfaces:
- Consumes: nothing.
- Produces: `resolve_anchors(n_samples: int, *, consensus_anchors: int | float | None, anchor_threshold: int, random_state: int | None) -> np.ndarray | None`. Returns `None` to mean "use the exact path", otherwise a sorted `int64` array of anchor indices.

- [ ] Step 1: Write the failing tests

Append to `tests/test_utils.py`:

```python
import numpy as np
import pytest

from carve._utils import resolve_anchors


class TestResolveAnchors:
    def test_below_threshold_is_exact(self):
        assert resolve_anchors(
            4000, consensus_anchors=None, anchor_threshold=5000, random_state=0
        ) is None

    def test_threshold_is_inclusive(self):
        # Levine is exactly 5000 cells; its published numbers must not move.
        assert resolve_anchors(
            5000, consensus_anchors=None, anchor_threshold=5000, random_state=0
        ) is None

    def test_above_threshold_uses_threshold_many_anchors(self):
        idx = resolve_anchors(
            6000, consensus_anchors=None, anchor_threshold=5000, random_state=0
        )
        assert idx is not None
        assert idx.shape == (5000,)
        assert idx.dtype == np.int64
        assert np.all(np.diff(idx) > 0)
        assert idx.min() >= 0 and idx.max() < 6000

    def test_no_discontinuity_across_the_threshold(self):
        # 4000 -> 4000 effective anchors (exact); 6000 -> 5000. The count must
        # not fall as n grows, which a flat default would cause.
        below = resolve_anchors(
            4000, consensus_anchors=None, anchor_threshold=5000, random_state=0
        )
        above = resolve_anchors(
            6000, consensus_anchors=None, anchor_threshold=5000, random_state=0
        )
        below_count = 4000 if below is None else below.size
        assert above.size >= below_count

    def test_explicit_int_opts_in_below_threshold(self):
        idx = resolve_anchors(
            5000, consensus_anchors=2000, anchor_threshold=5000, random_state=0
        )
        assert idx is not None and idx.size == 2000

    def test_float_is_a_fraction(self):
        idx = resolve_anchors(
            10_000, consensus_anchors=0.1, anchor_threshold=5000, random_state=0
        )
        assert idx.size == 1000

    def test_fraction_of_one_is_exact(self):
        assert resolve_anchors(
            10_000, consensus_anchors=1.0, anchor_threshold=5000, random_state=0
        ) is None

    def test_deterministic_in_random_state(self):
        a = resolve_anchors(
            9000, consensus_anchors=100, anchor_threshold=5000, random_state=7
        )
        b = resolve_anchors(
            9000, consensus_anchors=100, anchor_threshold=5000, random_state=7
        )
        c = resolve_anchors(
            9000, consensus_anchors=100, anchor_threshold=5000, random_state=8
        )
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    @pytest.mark.parametrize("bad", [0, 1, -5, 0.0, -0.2, 1.5])
    def test_rejects_degenerate_counts(self, bad):
        with pytest.raises(ValueError):
            resolve_anchors(
                9000, consensus_anchors=bad, anchor_threshold=5000, random_state=0
            )
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/test_utils.py::TestResolveAnchors -v`
Expected: FAIL, `ImportError: cannot import name 'resolve_anchors'`.

- [ ] Step 3: Implement

Append to `src/carve/_utils.py`:

```python
def resolve_anchors(
    n_samples: int,
    *,
    consensus_anchors: int | float | None,
    anchor_threshold: int,
    random_state: int | None,
) -> np.ndarray | None:
    """Resolve the consensus anchor index for a run.

    Returns None when the exact path applies, meaning the consensus matrix is
    built over every sample as it always has been. Otherwise returns a sorted
    array of anchor indices.

    The default resolves to ``min(n_samples, anchor_threshold)``. A flat
    default would make the effective anchor count fall discontinuously as
    n crosses the threshold -- 5,000 anchors at n=5,000 and 2,000 at
    n=5,001 -- which is an artificial jump in estimator variance at exactly
    the boundary users cross.

    Parameters
    ----------
    n_samples : int
        Number of samples in the run.
    consensus_anchors : int, float, or None
        Anchor count as an integer, a fraction of ``n_samples`` as a float in
        (0, 1], or None for the default.
    anchor_threshold : int
        Runs with ``n_samples <= anchor_threshold`` take the exact path. The
        comparison is inclusive so a dataset sized exactly at the threshold
        keeps the exact path.
    random_state : int or None
        Seed for the anchor draw. Drawn from a local Generator, so no global
        RNG state is touched and joblib workers stay reproducible.

    Returns
    -------
    anchors : ndarray of shape (m,) or None
    """
    n_samples = int(n_samples)

    if consensus_anchors is None:
        m = min(n_samples, int(anchor_threshold))
    elif isinstance(consensus_anchors, float):
        if not 0.0 < consensus_anchors <= 1.0:
            raise ValueError(
                "consensus_anchors given as a fraction must be in (0, 1], got "
                f"{consensus_anchors}."
            )
        m = int(round(consensus_anchors * n_samples))
    else:
        m = int(consensus_anchors)

    if m < 2:
        raise ValueError(
            f"consensus_anchors must resolve to at least 2 anchors, got {m}."
        )

    if m >= n_samples:
        return None

    rng = np.random.default_rng(0 if random_state is None else int(random_state))
    return np.sort(rng.choice(n_samples, size=m, replace=False)).astype(np.int64)
```

- [ ] Step 4: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_utils.py::TestResolveAnchors -v`
Expected: PASS, 11 tests.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/carve/_utils.py tests/test_utils.py
git commit -m "feat(carve): resolve consensus anchor indices"
```

---

### Task 2: Anchor factor construction and the anchor block

Files:
- Modify: `src/carve/_consensus.py`
- Test: `tests/test_consensus.py`

Interfaces:
- Consumes: `resolve_anchors` from Task 1 (only in tests, to build fixtures).
- Produces: `_anchor_factors(n_samples: int, runs: list, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, np.ndarray]], np.ndarray]` returning `(Sa, Ba, columns, pos)`, and `consensus_anchor_block(n_samples: int, runs: list, anchors: np.ndarray) -> np.ndarray` returning an `m x m` float32 matrix. `columns` is a list of `(column_index, sorted_member_indices)` used by Task 3.

The defining property, and the one the tests must enforce: `consensus_anchor_block(n, runs, a)` equals `compute_consensus_matrix(n, runs)[np.ix_(a, a)]`. Anchoring restricts which pairs are computed; it must not change any pair's value.

- [ ] Step 1: Write the failing tests

Append to `tests/test_consensus.py`:

```python
import numpy as np

from carve._consensus import compute_consensus_matrix, consensus_anchor_block


def _make_runs(n=200, n_runs=12, k=4, seed=0):
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_runs):
        idx = np.sort(rng.choice(n, size=int(0.618 * n), replace=False))
        runs.append((idx, rng.integers(0, k, idx.size)))
    return runs


class TestConsensusAnchorBlock:
    def test_block_equals_the_exact_submatrix(self):
        n = 200
        runs = _make_runs(n)
        anchors = np.sort(np.random.default_rng(1).choice(n, 40, replace=False))

        exact = compute_consensus_matrix(n, runs)[np.ix_(anchors, anchors)]
        block = consensus_anchor_block(n, runs, anchors)

        # Both carry NaN for never-co-sampled pairs, so compare with equal_nan.
        assert np.allclose(block, exact, equal_nan=True)

    def test_shape_and_dtype(self):
        n = 200
        runs = _make_runs(n)
        anchors = np.arange(0, n, 5)
        block = consensus_anchor_block(n, runs, anchors)
        assert block.shape == (anchors.size, anchors.size)
        assert block.dtype == np.float32

    def test_all_anchors_reproduces_the_full_matrix(self):
        n = 120
        runs = _make_runs(n, n_runs=8, k=3, seed=2)
        anchors = np.arange(n)
        assert np.allclose(
            consensus_anchor_block(n, runs, anchors),
            compute_consensus_matrix(n, runs),
            equal_nan=True,
        )

    def test_values_are_not_all_identical(self):
        # A block of constant values would satisfy a shape-only assertion,
        # so pin that the fixture actually varies.
        n = 200
        runs = _make_runs(n)
        anchors = np.arange(0, n, 5)
        block = consensus_anchor_block(n, runs, anchors)
        off = block[~np.eye(anchors.size, dtype=bool)]
        assert np.nanstd(off) > 0.01

    def test_never_cosampled_pairs_are_nan(self):
        n = 50
        # Two disjoint halves are never co-sampled, so every cross pair is NaN.
        left = np.arange(0, 25)
        right = np.arange(25, 50)
        runs = [
            (left, np.zeros(left.size, dtype=int)),
            (right, np.zeros(right.size, dtype=int)),
        ]
        anchors = np.array([0, 1, 30, 31])
        block = consensus_anchor_block(n, runs, anchors)
        assert np.isnan(block[0, 2])
        assert not np.isnan(block[0, 1])
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/test_consensus.py::TestConsensusAnchorBlock -v`
Expected: FAIL, `ImportError: cannot import name 'consensus_anchor_block'`.

- [ ] Step 3: Implement

Append to `src/carve/_consensus.py`:

```python
def _anchor_factors(
    n_samples: int,
    runs: list[SampledLabels],
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, np.ndarray]], np.ndarray]:
    """Build the anchor-side factors of the consensus decomposition.

    compute_consensus_matrix forms S of shape (n, n_runs) and B of shape
    (n, n_clusters_total), then takes S @ S.T and B @ B.T. Only the anchor
    rows of those factors are needed to form an anchor block or an anchor
    slab, and the full B is 2.46 GB at atlas scale, so it is never built.

    Returns
    -------
    Sa : ndarray of shape (m, n_runs)
    Ba : ndarray of shape (m, n_clusters_total)
    columns : list of (column_index, sorted member indices)
        One entry per cluster across all runs, reused by the slab pass.
    pos : ndarray of shape (n_samples,)
        Position of each sample in the anchor array, or -1 if not an anchor.
    """
    anchors = np.asarray(anchors, dtype=np.int64)
    m = anchors.size

    pos = np.full(n_samples, -1, dtype=np.int64)
    pos[anchors] = np.arange(m, dtype=np.int64)

    n_runs = len(runs)
    n_cols = sum(len(np.unique(labels)) for _, labels in runs)

    Sa = np.zeros((m, n_runs), dtype=np.float32)
    Ba = np.zeros((m, n_cols), dtype=np.float32)
    columns: list[tuple[int, np.ndarray]] = []

    col = 0
    for r, (sample_idx, labels) in enumerate(runs):
        sample_idx = np.asarray(sample_idx)
        labels = np.asarray(labels)

        p = pos[sample_idx]
        Sa[p[p >= 0], r] = 1.0

        for label in np.unique(labels):
            members = np.sort(sample_idx[labels == label])
            pm = pos[members]
            Ba[pm[pm >= 0], col] = 1.0
            columns.append((col, members))
            col += 1

    return Sa, Ba, columns, pos


def consensus_anchor_block(
    n_samples: int,
    runs: list[SampledLabels],
    anchors: np.ndarray,
) -> np.ndarray:
    """Consensus matrix restricted to anchor-by-anchor pairs.

    Equals ``compute_consensus_matrix(n_samples, runs)[np.ix_(anchors,
    anchors)]`` exactly. Anchoring restricts which pairs are computed; it
    does not change any pair's value.
    """
    Sa, Ba, _, _ = _anchor_factors(n_samples, runs, anchors)

    co_sample_counts = Sa @ Sa.T
    co_cluster_counts = Ba @ Ba.T

    with np.errstate(divide="ignore", invalid="ignore"):
        block = co_cluster_counts / co_sample_counts
    block[co_sample_counts == 0] = np.nan
    return block
```

Add `SampledLabels` to the existing imports at the top of `_consensus.py` if it is not already imported there.

- [ ] Step 4: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_consensus.py::TestConsensusAnchorBlock -v`
Expected: PASS, 5 tests.

- [ ] Step 5: Verify the equivalence test can fail

Temporarily change `Ba[pm[pm >= 0], col] = 1.0` to `= 0.5` and rerun. Expected: `test_block_equals_the_exact_submatrix` FAILS. Revert the change and confirm the tests pass again. A test that passes whether or not the block matches the exact submatrix is worthless here, because that equality is the entire correctness claim.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/carve/_consensus.py tests/test_consensus.py
git commit -m "feat(carve): compute consensus blocks over an anchor subset"
```

---

### Task 3: Full-length per-sample stability scores from an anchor slab

Files:
- Modify: `src/carve/_consensus.py`
- Test: `tests/test_consensus.py`

Interfaces:
- Consumes: `_anchor_factors` from Task 2.
- Produces: `stability_from_runs_anchored(n_samples: int, runs: list, anchors: np.ndarray, *, chunk_size: int = 8192) -> tuple[np.ndarray, np.ndarray]` returning `(stability_gini, stability_ce)`, each of length `n_samples`.

Each sample's score in `stability_from_consensus` is a row mean over that sample's consensus row. Restricting the columns to a random anchor set leaves the estimator unbiased and only raises its variance, so every one of the n samples still gets a score. The `n x m` slab is streamed in row chunks and reduced immediately, so peak memory is chunk size times m rather than n times m.

- [ ] Step 1: Write the failing tests

Append to `tests/test_consensus.py`:

```python
from carve._consensus import stability_from_consensus, stability_from_runs_anchored


class TestStabilityFromRunsAnchored:
    def test_all_anchors_matches_the_exact_scores(self):
        n = 150
        runs = _make_runs(n, n_runs=10, k=3, seed=5)
        exact_gini, exact_ce = stability_from_consensus(
            compute_consensus_matrix(n, runs)
        )
        gini, ce = stability_from_runs_anchored(n, runs, np.arange(n))

        assert np.allclose(gini, exact_gini, atol=1e-6)
        assert np.allclose(ce, exact_ce, atol=1e-6)

    def test_scores_cover_every_sample(self):
        n = 300
        runs = _make_runs(n, seed=6)
        anchors = np.sort(np.random.default_rng(3).choice(n, 50, replace=False))
        gini, ce = stability_from_runs_anchored(n, runs, anchors)

        assert gini.shape == (n,)
        assert ce.shape == (n,)
        assert np.isfinite(gini).all()
        assert np.isfinite(ce).all()

    def test_chunking_does_not_change_the_result(self):
        n = 300
        runs = _make_runs(n, seed=7)
        anchors = np.sort(np.random.default_rng(4).choice(n, 60, replace=False))
        a = stability_from_runs_anchored(n, runs, anchors, chunk_size=17)
        b = stability_from_runs_anchored(n, runs, anchors, chunk_size=100_000)
        assert np.allclose(a[0], b[0])
        assert np.allclose(a[1], b[1])

    def test_scores_vary_across_samples(self):
        # Guards against an implementation that returns a constant vector,
        # which every shape assertion above would still accept.
        n = 300
        runs = _make_runs(n, seed=8)
        anchors = np.arange(0, n, 4)
        gini, ce = stability_from_runs_anchored(n, runs, anchors)
        assert gini.std() > 1e-3
        assert ce.std() > 1e-3

    def test_anchor_subset_approximates_the_exact_scores(self):
        # The estimator is a row mean over a random column subset, so a
        # large anchor set should track the exact answer closely.
        n = 400
        runs = _make_runs(n, n_runs=20, k=4, seed=9)
        exact_gini, _ = stability_from_consensus(compute_consensus_matrix(n, runs))
        gini, _ = stability_from_runs_anchored(
            n, runs, np.sort(np.random.default_rng(5).choice(n, 300, replace=False))
        )
        assert np.corrcoef(gini, exact_gini)[0, 1] > 0.95
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/test_consensus.py::TestStabilityFromRunsAnchored -v`
Expected: FAIL, `ImportError: cannot import name 'stability_from_runs_anchored'`.

- [ ] Step 3: Implement

Append to `src/carve/_consensus.py`:

```python
def stability_from_runs_anchored(
    n_samples: int,
    runs: list[SampledLabels],
    anchors: np.ndarray,
    *,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample Gini and cross-entropy stability against an anchor set.

    Returns full-length score arrays. Each score is the same row-mean
    statistic stability_from_consensus computes, taken over m random anchor
    partners instead of over all n - 1 partners, so it is unbiased with
    variance of order 1/m.

    The n-by-m slab is never materialized: rows are processed in chunks and
    reduced to the two score vectors immediately.
    """
    Sa, Ba, columns, pos = _anchor_factors(n_samples, runs, anchors)
    n_runs = Sa.shape[1]
    n_cols = Ba.shape[1]

    stability_gini = np.empty(n_samples, dtype=float)
    stability_ce = np.empty(n_samples, dtype=float)

    run_indices = [np.asarray(sample_idx) for sample_idx, _ in runs]

    for lo in range(0, n_samples, chunk_size):
        hi = min(lo + chunk_size, n_samples)
        rows = hi - lo

        Sc = np.zeros((rows, n_runs), dtype=np.float32)
        for r, sample_idx in enumerate(run_indices):
            start = np.searchsorted(sample_idx, lo)
            stop = np.searchsorted(sample_idx, hi)
            Sc[sample_idx[start:stop] - lo, r] = 1.0

        Bc = np.zeros((rows, n_cols), dtype=np.float32)
        for col, members in columns:
            start = np.searchsorted(members, lo)
            stop = np.searchsorted(members, hi)
            Bc[members[start:stop] - lo, col] = 1.0

        co_sample = Sc @ Sa.T
        co_cluster = Bc @ Ba.T
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = np.asarray(co_cluster / co_sample, dtype=float)
        probs[co_sample == 0] = np.nan

        # Blank each row's self-pair, mirroring the fill_diagonal in
        # stability_from_consensus. A sample that is itself an anchor must not
        # contribute its own perfect self-similarity to its own score.
        self_pos = pos[lo:hi]
        is_anchor = self_pos >= 0
        probs[np.nonzero(is_anchor)[0], self_pos[is_anchor]] = np.nan

        term = probs * (1.0 - probs)
        clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
        entropy = -(
            clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
        )

        uncertainty_gini = 2.0 * np.nanmean(term, axis=1)
        uncertainty_ce = np.nanmean(entropy, axis=1)

        stability_gini[lo:hi] = 1.0 - np.clip(2.0 * uncertainty_gini, 0.0, 1.0)
        stability_ce[lo:hi] = 1.0 - np.clip(uncertainty_ce / np.log(2.0), 0.0, 1.0)

    return stability_gini, stability_ce
```

Note on the two `np.nanmean` calls: a chunk row whose sample was never co-sampled with any anchor produces an all-NaN slice and a `RuntimeWarning: Mean of empty slice`. This is the same warning `_consensus.py` already emits on small fixtures and is why the suite cannot run under `-W error` globally. Do not suppress it here; leave the existing behavior unchanged.

- [ ] Step 4: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_consensus.py::TestStabilityFromRunsAnchored -v`
Expected: PASS, 5 tests.

- [ ] Step 5: Verify the self-pair blanking can fail

Temporarily delete the two lines that set `probs[..., self_pos[is_anchor]] = np.nan` and rerun. Expected: `test_all_anchors_matches_the_exact_scores` FAILS, because the self-pair contributes a certainty of 1.0 to every anchor's mean. Restore and confirm PASS.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/carve/_consensus.py tests/test_consensus.py
git commit -m "feat(carve): stream per-sample stability scores from an anchor slab"
```

---

### Task 4: Carry consensus summaries out of the runner

Files:
- Create: nothing
- Modify: `src/carve/_types.py`, `src/carve/_runner.py:240-310` (the consensus build block) and `src/carve/_runner.py:386-393` (the return), `src/carve/api.py:419-441` (the unpack)
- Test: `tests/test_types.py`, `tests/test_runner.py:281,317,334`

Interfaces:
- Consumes: `consensus_anchor_block` and `stability_from_runs_anchored` from Tasks 2 and 3.
- Produces: `ConsensusSummary` frozen dataclass with fields `gini: np.ndarray`, `ce: np.ndarray`, `pac: float`. `run_validation` gains a keyword argument `anchors: np.ndarray | None = None` and returns a sixth element `consensus_summaries: list[ConsensusSummary] | None`.

Rationale for the sixth element: with anchoring, `consensus_matrices_` holds `m x m` blocks, so `api.fit`'s current call to `compute_consensus_metrics(self.consensus_matrices_)` would produce length-m score vectors. The runner is the only place that still holds `runs`, so it must produce the full-length scores. Returning one composite field rather than three loose lists keeps the tuple readable; three of the existing test call sites unpack exactly five values and must be updated.

- [ ] Step 1: Write the failing tests

Append to `tests/test_types.py`:

```python
import numpy as np

from carve._types import ConsensusSummary


def test_consensus_summary_is_frozen():
    import dataclasses
    import pytest

    s = ConsensusSummary(gini=np.zeros(3), ce=np.zeros(3), pac=0.5)
    assert s.pac == 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.pac = 0.9
```

Append to `tests/test_runner.py`:

```python
import numpy as np

from carve._runner import run_validation
from carve._types import ConsensusSummary


class TestRunValidationAnchors:
    def _grids(self):
        from sklearn.cluster import KMeans

        return [(KMeans, {"n_clusters": [2, 3], "n_init": [10]})]

    def test_summaries_are_full_length_under_anchoring(self):
        rng = np.random.default_rng(0)
        n = 60
        X = np.vstack([rng.normal(0, 1, (n // 2, 4)), rng.normal(6, 1, (n // 2, 4))])
        anchors = np.sort(rng.choice(n, 20, replace=False))

        out = run_validation(
            X=X,
            estimator_grids=self._grids(),
            n_resamples=6,
            random_state=0,
            anchors=anchors,
        )
        summaries = out[5]
        matrices = out[2]

        assert summaries is not None
        assert len(summaries) == len(matrices)
        for summary, matrix in zip(summaries, matrices):
            assert isinstance(summary, ConsensusSummary)
            # Scores cover every sample even though the block is anchor sized.
            assert summary.gini.shape == (n,)
            assert summary.ce.shape == (n,)
            assert matrix.shape == (anchors.size, anchors.size)

    def test_exact_path_is_unchanged_when_anchors_is_none(self):
        rng = np.random.default_rng(1)
        n = 60
        X = np.vstack([rng.normal(0, 1, (n // 2, 4)), rng.normal(6, 1, (n // 2, 4))])

        out = run_validation(
            X=X,
            estimator_grids=self._grids(),
            n_resamples=6,
            random_state=0,
            anchors=None,
        )
        for matrix in out[2]:
            assert matrix.shape == (n, n)
        for summary in out[5]:
            assert summary.gini.shape == (n,)
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/test_types.py -k consensus_summary tests/test_runner.py::TestRunValidationAnchors -v`
Expected: FAIL, `ImportError: cannot import name 'ConsensusSummary'`.

- [ ] Step 3: Add the dataclass

Append to `src/carve/_types.py`:

```python
@dataclass(frozen=True)
class ConsensusSummary:
    """Per-configuration stability quantities derived from the consensus.

    Carried out of the runner rather than recomputed in api.fit, because
    under anchoring the stored consensus matrix is an m-by-m block while
    these score vectors are full length, and the runner is the only place
    that still holds the per-resample runs they are derived from.
    """

    gini: np.ndarray
    ce: np.ndarray
    pac: float
```

Add `import numpy as np` and `from dataclasses import dataclass` to `_types.py` if they are not already present.

- [ ] Step 4: Wire the runner

In `src/carve/_runner.py`, add `anchors: np.ndarray | None = None` to the `run_validation` signature and document it in the docstring's Parameters section. Beside the existing accumulator lists, add:

```python
    consensus_summaries: list[ConsensusSummary | None] = []
```

Replace the consensus build block at lines 269-302 with:

```python
                # --- Build consensus matrices ---
                if not policy.run_stability:
                    M = None
                elif anchors is None:
                    M = compute_consensus_matrix(
                        n_samples=n,
                        runs=[(r.train_indices, r.labels_train) for r in stab_runs],
                    )
                else:
                    M = consensus_anchor_block(
                        n_samples=n,
                        runs=[(r.train_indices, r.labels_train) for r in stab_runs],
                        anchors=anchors,
                    )

                if not policy.run_generalizability:
                    M_g = None
                elif anchors is None:
                    M_g = compute_consensus_matrix(
                        n_samples=n,
                        runs=[(r.test_indices, r.labels_predicted) for r in gen_runs],
                    )
                else:
                    M_g = consensus_anchor_block(
                        n_samples=n,
                        runs=[(r.test_indices, r.labels_predicted) for r in gen_runs],
                        anchors=anchors,
                    )

                # --- Per-sample stability scores, always full length ---
                if not policy.run_stability:
                    summary = None
                else:
                    stab_pairs = [(r.train_indices, r.labels_train) for r in stab_runs]
                    if anchors is None:
                        gini, ce = stability_from_consensus(M)
                    else:
                        gini, ce = stability_from_runs_anchored(
                            n_samples=n, runs=stab_pairs, anchors=anchors
                        )
                    summary = ConsensusSummary(
                        gini=gini, ce=ce, pac=compute_consensus_pac(M)
                    )

                # --- Compute generalizability scores ---
                E = (
                    compute_generalizability_scores(
                        n_samples=n,
                        runs=[
                            (r.test_indices, r.labels_test, r.labels_predicted)
                            for r in gen_runs
                        ],
                    )
                    if policy.run_generalizability
                    else None
                )

                consensus_matrices.append(M)
                consensus_generalizability_matrices.append(M_g)
                generalizability_scores.append(E)
                consensus_summaries.append(summary)
```

Update the imports at the top of `_runner.py` to bring in `consensus_anchor_block`, `stability_from_runs_anchored`, `stability_from_consensus` and `compute_consensus_pac` from `._consensus`, and `ConsensusSummary` from `._types`.

Update the return at the end of `run_validation`:

```python
    return (
        estimator_records,
        pipeline_records,
        consensus_matrices,
        consensus_generalizability_matrices,
        generalizability_scores,
        consensus_summaries if policy.run_stability else None,
    )
```

Add `consensus_summaries : list of ConsensusSummary or None` to the docstring's Returns section.

- [ ] Step 5: Update the three exact-arity unpack sites in the existing tests

In `tests/test_runner.py`, lines 281, 317 and 334 unpack exactly five values. Change each from the form `records, pipeline_records, cons, cons_gen, gen_scores = run_validation(` to `records, pipeline_records, cons, cons_gen, gen_scores, _ = run_validation(` (and likewise `records, _, cons, cons_gen, gen_scores, _ =` at 317 and 334). Sites using `records, *_ =` need no change.

- [ ] Step 6: Update the api.fit unpack

In `src/carve/api.py` at line 419, change the unpack target to add a sixth name, and replace the stability-derived metrics block at lines 466-489 so it reads the summaries rather than recomputing from the matrices:

```python
        (
            estimator_records,
            pipeline_records,
            self.consensus_matrices_,
            self.consensus_generalizability_matrices_,
            self.generalizability_scores_,
            consensus_summaries,
        ) = run_validation(
```

and

```python
        # --- Stability-derived metrics ---
        if policy.run_stability and consensus_summaries is not None:
            self.stability_gini_scores_ = np.vstack(
                [s.gini for s in consensus_summaries]
            )
            self.stability_ce_scores_ = np.vstack([s.ce for s in consensus_summaries])

            self.estimator_results_["consensus_pac_stability"] = [
                s.pac for s in consensus_summaries
            ]
            self.estimator_results_["consensus_gini_stability"] = (
                self.stability_gini_scores_.mean(axis=1)
            )
            self.estimator_results_["consensus_ce_stability"] = (
                self.stability_ce_scores_.mean(axis=1)
            )
        else:
```

Leave the `else` branch exactly as it is. Remove the now-unused `compute_consensus_metrics` import from `api.py` if nothing else uses it; leave the function itself in `_consensus.py`, which still exports it.

- [ ] Step 7: Run the tests

Run: `.venv/bin/python -m pytest tests/test_types.py tests/test_runner.py tests/test_api.py -v`
Expected: PASS. The `test_api.py` monkeypatch at lines 1108-1123 replaces `run_validation` with a corrupting stub; if it returns a five-tuple it must be widened to six.

- [ ] Step 8: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/carve/_types.py src/carve/_runner.py src/carve/api.py tests/
git commit -m "feat(carve): carry full-length consensus summaries out of the runner"
```

---

### Task 5: CARVE constructor parameters and fit wiring

Files:
- Modify: `src/carve/api.py` (constructor fields near lines 190-235, docstring near lines 100-140, and `fit` near lines 335-345)
- Test: `tests/test_api.py`

Interfaces:
- Consumes: `resolve_anchors` from Task 1, the runner's `anchors` argument from Task 4.
- Produces: `CARVE.anchor_threshold: int = 5000`, `CARVE.consensus_anchors: int | float | None = None`, and the fitted attribute `CARVE.consensus_anchors_: np.ndarray | None`.

- [ ] Step 1: Write the failing tests

Append to `tests/test_api.py`:

```python
import numpy as np
import pytest
from sklearn.cluster import KMeans

from carve import CARVE


def _blobs(n, seed=0, p=4):
    rng = np.random.default_rng(seed)
    half = n // 2
    return np.vstack([rng.normal(0, 1, (half, p)), rng.normal(6, 1, (n - half, p))])


def _grids():
    return [(KMeans, {"n_clusters": [2, 3], "n_init": [10]})]


class TestAnchoredConsensus:
    def test_below_threshold_stores_no_anchors_and_full_matrices(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=100,
        ).fit(X)
        assert c.consensus_anchors_ is None
        assert c.consensus_matrices_[0].shape == (60, 60)

    def test_above_threshold_stores_anchors_and_block_matrices(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning, match="anchored consensus"):
            c.fit(X)
        assert c.consensus_anchors_ is not None
        assert c.consensus_anchors_.size == 30
        assert c.consensus_matrices_[0].shape == (30, 30)

    def test_per_sample_scores_stay_full_length_under_anchoring(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)
        assert c.stability_gini_scores_.shape[1] == 60
        assert c.stability_ce_scores_.shape[1] == 60
        assert c.generalizability_scores_[0].shape == (60,)

    def test_explicit_anchor_count_opts_in_below_threshold(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=1000,
            consensus_anchors=25,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)
        assert c.consensus_anchors_.size == 25

    def test_config_id_alignment_holds_under_anchoring(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)
        n_rows = c.estimator_results_.shape[0]
        assert np.array_equal(
            c.estimator_results_["config_id"].to_numpy(), np.arange(n_rows)
        )
        assert len(c.consensus_matrices_) == n_rows
        assert c.stability_gini_scores_.shape[0] == n_rows

    def test_every_configuration_indexes_the_same_anchors(self):
        # Configurations must be comparable across k, so one draw is reused.
        # If each config drew its own anchors, blocks at different k would
        # index different samples and cross-k comparison would be meaningless.
        # Two configs are fitted here, so identical off-diagonal NaN masks are
        # evidence they were built over the same co-sampled pairs.
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=30,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)
        assert c.consensus_anchors_.ndim == 1
        assert np.all(np.diff(c.consensus_anchors_) > 0)
        assert len(c.consensus_matrices_) >= 2
        first, second = c.consensus_matrices_[0], c.consensus_matrices_[1]
        assert first.shape == second.shape == (30, 30)
        assert np.array_equal(np.isnan(first), np.isnan(second))
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/test_api.py::TestAnchoredConsensus -v`
Expected: FAIL, `TypeError: CARVE.__init__() got an unexpected keyword argument 'anchor_threshold'`.

- [ ] Step 3: Add the constructor fields

In `src/carve/api.py`, beside the existing `n_resamples` and `subsample_ratio` fields, add:

```python
    anchor_threshold: int = 5000
    consensus_anchors: int | float | None = None
```

and beside the other fitted attributes add:

```python
    consensus_anchors_: np.ndarray | None = field(init=False, default=None)
```

Add to the class docstring's Parameters section:

```
    anchor_threshold : int, default=5000
        Runs with ``n_samples <= anchor_threshold`` build full consensus
        matrices exactly as before. Above it, consensus quantities are
        computed over a fixed anchor subset, because a dense n-by-n matrix
        per configuration is not feasible at large n. The comparison is
        inclusive.
    consensus_anchors : int, float, or None, default=None
        Number of anchors, or a fraction of ``n_samples`` as a float in
        (0, 1]. None resolves to ``min(n_samples, anchor_threshold)``, which
        keeps the effective anchor count continuous across the threshold.
        Lower it to reduce the memory the retained blocks occupy: each block
        is 4 * m**2 bytes and there are two per configuration.
```

and to the Attributes section:

```
    consensus_anchors_ : ndarray or None
        Anchor indices used for this fit, or None when the exact path ran.
```

- [ ] Step 4: Wire fit

In `src/carve/api.py`, after `X = ensure_2d_array(X)` and `self.X_ = X` in `fit`, insert:

```python
        seed = self.random_state if random_state is None else random_state
        self.consensus_anchors_ = resolve_anchors(
            X.shape[0],
            consensus_anchors=self.consensus_anchors,
            anchor_threshold=self.anchor_threshold,
            random_state=seed,
        )
        if self.consensus_anchors_ is not None:
            warnings.warn(
                f"n={X.shape[0]} exceeds anchor_threshold="
                f"{self.anchor_threshold}, so CARVE is using anchored "
                f"consensus over {self.consensus_anchors_.size} anchors. "
                "Per-sample scores and labels still cover every sample; "
                "consensus matrices and PAC are computed over the anchors.",
                RuntimeWarning,
                stacklevel=2,
            )
```

Add `anchors=self.consensus_anchors_,` to the `run_validation(...)` call, and add `resolve_anchors` to the existing `from ._utils import ...` line.

- [ ] Step 5: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_api.py::TestAnchoredConsensus -v`
Expected: PASS, 6 tests.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/carve/api.py tests/test_api.py
git commit -m "feat(carve): add anchor_threshold and consensus_anchors to CARVE"
```

---

### Task 6: Extend labels from the anchor cut to every sample

Files:
- Modify: `src/carve/api.py:596-735` (`get_labels`)
- Test: `tests/test_api.py`

Interfaces:
- Consumes: `CARVE.consensus_anchors_` from Task 5.
- Produces: `get_labels` returns an array of length `n_samples` whether or not anchoring is active.

The anchor block cut gives labels for m anchors. The remaining n - m samples are labeled by fitting a clone of `self.classifier` on `(X_[anchors], anchor_labels)` and predicting the rest. This is the same out-of-sample prediction the generalizability path already performs on every resample, not a new mechanism.

- [ ] Step 1: Write the failing tests

Append to `tests/test_api.py`:

```python
class TestAnchoredLabels:
    def _fitted(self, n=60, threshold=30):
        X = _blobs(n)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=threshold,
        )
        with pytest.warns(RuntimeWarning):
            c.fit(X)
        return X, c

    def test_labels_cover_every_sample(self):
        X, c = self._fitted()
        labels = c.get_labels(k=2)
        assert labels.shape == (X.shape[0],)
        assert labels.dtype == np.int32

    def test_labels_recover_the_planted_structure(self):
        # _blobs plants two well separated groups, so an extension that
        # assigned arbitrary labels would fail this even with a valid shape.
        from sklearn.metrics import adjusted_rand_score

        X, c = self._fitted()
        truth = np.repeat([0, 1], X.shape[0] // 2)
        assert adjusted_rand_score(truth, c.get_labels(k=2)) > 0.9

    def test_anchor_positions_keep_their_cut_labels(self):
        # The extension must not overwrite the anchors' own labels.
        X, c = self._fitted()
        labels = c.get_labels(k=2)
        anchors = c.consensus_anchors_
        assert np.unique(labels[anchors]).size == 2

    def test_number_of_clusters_matches_the_cut(self):
        X, c = self._fitted()
        assert np.unique(c.get_labels(k=3)).size == 3

    def test_exact_path_labels_are_unaffected(self):
        X = _blobs(60)
        c = CARVE(
            estimator_param_grids=_grids(),
            n_resamples=6,
            random_state=0,
            anchor_threshold=1000,
        ).fit(X)
        assert c.consensus_anchors_ is None
        assert c.get_labels(k=2).shape == (60,)
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/test_api.py::TestAnchoredLabels -v`
Expected: FAIL. The cut currently returns an array of length m, so `test_labels_cover_every_sample` fails on shape `(30,) != (60,)`.

- [ ] Step 3: Implement

In `get_labels`, immediately after `labels = estimator.fit_predict(D)` and before the reference-label alignment block, insert:

```python
        if self.consensus_anchors_ is not None:
            labels = self._extend_anchor_labels(labels)
```

Add this method to `CARVE`, directly after `get_labels`:

```python
    def _extend_anchor_labels(self, anchor_labels: np.ndarray) -> np.ndarray:
        """Label every sample from a cut taken over the anchor subset.

        The anchor block yields labels for m anchors. The remaining samples
        are assigned by the same classifier CARVE clones per resample to
        score generalizability, fitted on the anchors and their cut labels.
        Anchors keep the labels the cut gave them rather than the
        classifier's prediction for them.
        """
        if self.X_ is None:
            raise RuntimeError(
                "X_ is not available, so anchored labels cannot be extended to "
                "every sample. Call fit() first, or restore X_ after load()."
            )

        anchors = self.consensus_anchors_
        n_samples = self.X_.shape[0]
        anchor_labels = np.asarray(anchor_labels)

        labels = np.empty(n_samples, dtype=np.int64)
        labels[anchors] = anchor_labels

        rest = np.setdiff1d(np.arange(n_samples), anchors, assume_unique=False)
        if rest.size == 0:
            return labels

        if np.unique(anchor_labels).size < 2:
            # A degenerate cut gives the classifier a single class; every
            # remaining sample belongs to it by construction.
            labels[rest] = anchor_labels[0]
            return labels

        classifier = (
            clone(self.classifier)
            if self.classifier is not None
            else RandomForestClassifier(
                n_estimators=self.n_trees,
                n_jobs=-1,
                random_state=self.random_state,
            )
        )
        classifier.fit(self.X_[anchors], anchor_labels)
        labels[rest] = classifier.predict(self.X_[rest])
        return labels
```

Add `from sklearn.base import clone` and `from sklearn.ensemble import RandomForestClassifier` to the imports in `api.py` if they are not already present.

- [ ] Step 4: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_api.py::TestAnchoredLabels -v`
Expected: PASS, 5 tests.

- [ ] Step 5: Strengthen the anchor-preservation test, then settle it by mutation

As written, `test_anchor_positions_keep_their_cut_labels` only counts distinct labels at the anchor positions, which an implementation that overwrote the anchors with classifier predictions would still satisfy. Replace it with a direct check that the method returns the cut it was given:

```python
    def test_anchor_positions_keep_their_cut_labels(self):
        X, c = self._fitted()
        anchors = c.consensus_anchors_
        planted = np.arange(anchors.size) % 2
        extended = c._extend_anchor_labels(planted)
        assert np.array_equal(extended[anchors], planted)
        assert extended.shape == (X.shape[0],)
```

Now mutate: change `rest = np.setdiff1d(...)` to `rest = np.arange(n_samples)`, so the classifier overwrites the anchors too.

Run: `.venv/bin/python -m pytest tests/test_api.py::TestAnchoredLabels -v`
Expected: FAIL on `test_anchor_positions_keep_their_cut_labels`. Revert the mutation and confirm PASS.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/carve/api.py tests/test_api.py
git commit -m "feat(carve): extend anchored cut labels to every sample"
```

---

### Task 7: Consensus-matrix plotting under anchoring

Files:
- Modify: `src/carve/api.py:973-1095` (`plot_consensus_matrix`)
- Test: `tests/test_plotting.py`

Interfaces:
- Consumes: `CARVE.consensus_anchors_` from Task 5, `_extend_anchor_labels` from Task 6.
- Produces: `plot_consensus_matrix` renders the `m x m` block against anchor labels when anchoring is active.

`plot_consensus_matrix` in `_plotting.py` raises when `consensus_matrix` and `labels` disagree on their first dimension. Under anchoring the stored matrix is `m x m` while `get_labels` now returns length n, so the method must subset the labels to the anchors.

- [ ] Step 1: Write the failing test

Append to `tests/test_plotting.py`:

```python
import matplotlib
import numpy as np
import pytest
from sklearn.cluster import KMeans

matplotlib.use("Agg")

from carve import CARVE


def test_plot_consensus_matrix_under_anchoring():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(6, 1, (30, 4))])
    c = CARVE(
        estimator_param_grids=[(KMeans, {"n_clusters": [2, 3], "n_init": [10]})],
        n_resamples=6,
        random_state=0,
        anchor_threshold=30,
    )
    with pytest.warns(RuntimeWarning):
        c.fit(X)

    ax = c.plot_consensus_matrix(k=2)
    assert ax is not None
    # The rendered image is the anchor block, not the full matrix.
    assert c.consensus_matrices_[0].shape == (30, 30)
```

- [ ] Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest tests/test_plotting.py::test_plot_consensus_matrix_under_anchoring -v`
Expected: FAIL with a ValueError about `consensus_matrix and labels must have matching first dimension`.

- [ ] Step 3: Implement

In `plot_consensus_matrix`, after the labels are obtained from `self.get_labels(...)` at line 1071 and before they are passed to the plotting function, insert:

```python
        if self.consensus_anchors_ is not None:
            # The stored matrix is the anchor block, so the labels drawn
            # alongside it must be the anchors' labels.
            labels = labels[self.consensus_anchors_]
```

Add to the method docstring, in the Notes section:

```
    When anchored consensus is active the rendered matrix is the anchor
    block rather than the full sample-by-sample matrix, and the axis
    therefore shows ``consensus_anchors_.size`` rows. A full matrix is not
    renderable at the sample counts anchoring exists to support.
```

- [ ] Step 4: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/test_plotting.py::test_plot_consensus_matrix_under_anchoring -v`
Expected: PASS.

- [ ] Step 5: Confirm the four sample-level plots still work

Run: `.venv/bin/python -m pytest tests/test_plotting.py -v`
Expected: PASS. `plot_cluster_boxplot`, `plot_cluster_violin`, `plot_cluster_scatter` and `plot_diagnostic_scatter` each read a per-sample score array and call `get_labels`; both are full length after Tasks 4 and 6, so none of them needs a change. If any fails, that is a real defect in those tasks and must be fixed there rather than patched here.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/carve/api.py tests/test_plotting.py
git commit -m "feat(carve): draw the anchor block in plot_consensus_matrix"
```

---

### Task 8: Backward-compatibility gate

Files:
- Test: `tests/test_api.py`

Interfaces:
- Consumes: everything from Tasks 1 through 7.
- Produces: no source change. This task exists to prove that at or below the threshold nothing moved.

This is the load-bearing promise of the whole plan: the published Klein and Levine results must not change. Levine is exactly 5,000 cells, which is why the threshold comparison is inclusive.

- [ ] Step 1: Write the gate

Append to `tests/test_api.py`:

```python
class TestExactPathUnchanged:
    def _fit(self, n, **kwargs):
        X = _blobs(n, seed=3)
        return X, CARVE(
            estimator_param_grids=_grids(),
            n_resamples=8,
            random_state=11,
            **kwargs,
        ).fit(X)

    def test_default_threshold_keeps_five_thousand_exact(self):
        # Not a fit at n=5000, which is slow; assert the resolution rule that
        # governs it, which is what the promise actually rests on.
        from carve._utils import resolve_anchors

        assert resolve_anchors(
            5000, consensus_anchors=None, anchor_threshold=5000, random_state=42
        ) is None

    def test_defaults_do_not_engage_anchoring_at_small_n(self):
        _, c = self._fit(80)
        assert c.consensus_anchors_ is None
        assert c.consensus_matrices_[0].shape == (80, 80)

    def test_results_are_identical_with_and_without_the_feature_present(self):
        # Two fits differing only in an anchor_threshold that cannot bind.
        X, a = self._fit(80)
        _, b = self._fit(80, anchor_threshold=10_000)

        assert a.consensus_anchors_ is None and b.consensus_anchors_ is None
        assert np.array_equal(a.get_labels(k=2), b.get_labels(k=2))
        assert np.allclose(a.consensus_matrices_[0], b.consensus_matrices_[0],
                           equal_nan=True)
        assert np.allclose(a.stability_gini_scores_, b.stability_gini_scores_)
        assert np.allclose(
            a.estimator_results_["ari_stability"].to_numpy(),
            b.estimator_results_["ari_stability"].to_numpy(),
        )

    def test_no_warning_on_the_exact_path(self):
        import warnings as _w

        X = _blobs(80, seed=3)
        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)
            CARVE(
                estimator_param_grids=_grids(),
                n_resamples=8,
                random_state=11,
            ).fit(X)
```

- [ ] Step 2: Run the gate

Run: `.venv/bin/python -m pytest tests/test_api.py::TestExactPathUnchanged -v`
Expected: PASS, 4 tests.

Note: `test_no_warning_on_the_exact_path` promotes only RuntimeWarning to an error, not all warnings, because `_consensus.py` emits `RuntimeWarning: Mean of empty slice` on some small fixtures. If that fires here, enlarge the fixture rather than weakening the assertion.

- [ ] Step 3: Settle the gate by mutation

Change the comparison in `resolve_anchors` from `if m >= n_samples:` to `if m > n_samples:`. This makes the threshold exclusive, which is precisely the regression the gate exists to catch.

Run: `.venv/bin/python -m pytest tests/test_api.py::TestExactPathUnchanged tests/test_utils.py::TestResolveAnchors -v`
Expected: FAIL on `test_default_threshold_keeps_five_thousand_exact` and `test_threshold_is_inclusive`.

Revert the mutation. Rerun. Expected: PASS. A gate that passes under this mutation would not protect the published results and must be strengthened before proceeding.

- [ ] Step 4: Run the full carve suite

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: PASS. Budget 13 to 22 minutes; run it in the foreground. Compare the pass and skip counts against the pre-change baseline of 1104 passed, 10 skipped; the counts should rise by the tests added in Tasks 1 through 8 and nothing should newly fail.

- [ ] Step 5: Commit

```bash
git add tests/test_api.py
git commit -m "test(carve): gate the exact consensus path against regression"
```

---

### Task 9: Anchored-against-exact accuracy validation

Files:
- Create: `tests/test_anchored_accuracy.py`
- Test: the same file

Interfaces:
- Consumes: everything from Tasks 1 through 7.
- Produces: no source change. Produces the evidence that the approximation is sound, which section 4.5 of the spec designates as a candidate supplementary table for the manuscript.

At moderate n the exact answer is computable, so the anchored path is measured against it directly rather than argued for. This file is a new module rather than an addition to `tests/test_api.py` because it is a study of the approximation, not a unit test of the API, and it is slow enough to want its own marker.

- [ ] Step 1: Write the validation

Create `tests/test_anchored_accuracy.py`:

```python
"""Anchored consensus measured against the exact answer.

At n small enough for the exact path to run, both are computable, so the
approximation is measured rather than asserted. The numbers this produces
back the supplementary table described in the design document.
"""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from carve import CARVE

N = 5000
GRIDS = [(KMeans, {"n_clusters": [2, 3, 4], "n_init": [10]})]

# The design document names m in {500, 1000, 2000, 5000}. At N = 5000 the
# fourth value is the exact path by definition, since resolve_anchors returns
# None once m reaches n; that identity is already covered by
# tests/test_consensus.py::TestConsensusAnchorBlock. The three values below
# are the ones that actually exercise the approximation.


def _data(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [8, 0], [4, 7]], dtype=float)
    y = rng.integers(0, 3, N)
    X = centers[y] + rng.normal(0, 1.2, (N, 2))
    return X, y


def _fit(X, *, anchors):
    kwargs = {"anchor_threshold": 10_000}
    if anchors is not None:
        kwargs["consensus_anchors"] = anchors
    carve = CARVE(
        estimator_param_grids=GRIDS, n_resamples=25, random_state=0, **kwargs
    )
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        carve.fit(X)
    return carve


@pytest.fixture(scope="module")
def exact():
    X, y = _data()
    return X, y, _fit(X, anchors=None)


@pytest.mark.slow
@pytest.mark.parametrize("m", [500, 1000, 2000])
def test_anchored_tracks_exact(exact, m):
    X, y, exact_carve = exact
    anchored = _fit(X, anchors=m)

    assert anchored.consensus_anchors_.size == m

    # Selected k agrees.
    assert anchored.get_k(measure="stability", rule="1se") == exact_carve.get_k(
        measure="stability", rule="1se"
    )

    # Per-sample stability scores correlate with the exact ones.
    corr = np.corrcoef(
        anchored.stability_gini_scores_[0], exact_carve.stability_gini_scores_[0]
    )[0, 1]
    assert corr > 0.8, f"gini correlation {corr:.3f} at m={m}"

    # Labels agree with each other and both recover the planted structure.
    both = adjusted_rand_score(
        exact_carve.get_labels(k=3), anchored.get_labels(k=3)
    )
    assert both > 0.9, f"label ARI {both:.3f} at m={m}"
    assert adjusted_rand_score(y, anchored.get_labels(k=3)) > 0.9


@pytest.mark.slow
def test_accuracy_improves_with_more_anchors(exact):
    X, _, exact_carve = exact
    reference = exact_carve.stability_gini_scores_[0]

    correlations = []
    for m in (500, 2000):
        anchored = _fit(X, anchors=m)
        correlations.append(
            np.corrcoef(anchored.stability_gini_scores_[0], reference)[0, 1]
        )

    # Variance of the row-mean estimator falls as 1/m, so more anchors must
    # track the exact answer better. A flat result would mean the anchor
    # count is not actually being honored.
    assert correlations[1] > correlations[0]
```

- [ ] Step 2: Run the validation

Run: `.venv/bin/python -m pytest tests/test_anchored_accuracy.py -v -m slow`
Expected: PASS, 4 tests. If `test_anchored_tracks_exact` fails at m=200 on the correlation threshold, record the observed value and consider whether the threshold or the anchor default needs revisiting. Do not lower a threshold to make a failing measurement pass without saying so; that is the finding, not a nuisance.

- [ ] Step 3: Confirm the slow marker excludes it by default

Run: `.venv/bin/python -m pytest tests/test_anchored_accuracy.py -v -m "not slow"`
Expected: 4 deselected, 0 run. The `slow` marker is already declared in `pyproject.toml`.

- [ ] Step 4: Record the measured numbers

Run the validation with `-s` and capture the correlation and ARI values it reports for each m. Append them to the spec document under section 4.5 as a short table, so the manuscript's supplementary table has a source.

- [ ] Step 5: Commit

```bash
git add tests/test_anchored_accuracy.py docs/superpowers/specs/2026-09-07-atac-and-large-scale-case-studies-design.md
git commit -m "test(carve): measure anchored consensus against the exact answer"
```

---

## Completion checklist

- [ ] `.venv/bin/ruff check src/` and `.venv/bin/ruff format --check src/` both clean
- [ ] `.venv/bin/python -m pytest tests/ -v --tb=short` passes, with no new failures against the 1104 passed / 10 skipped baseline
- [ ] `.venv/bin/python -m pytest tests/ -q --cov=carve --cov-report=term-missing --cov-fail-under=75` passes
- [ ] The R counterpart in `carve-r/R/` is still absent and the deviation from the one-to-one mirror rule is recorded in a tracked follow-up issue on `DataSlingers/CARVE`
- [ ] Section 4.5 of the spec carries the measured anchored-against-exact numbers
