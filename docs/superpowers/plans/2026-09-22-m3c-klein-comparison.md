# M3C Klein Case-Study Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add M3C, run at its published defaults, as a selection comparator in the Klein case study, so reviewer comment R2.1 is answered with a figure, a table and measured numbers.

**Architecture:** One new leaf module, `src/benchmarks/_m3c.py`, owns the whole rpy2 bridge to M3C and returns plain pandas and numpy. Its result reaches the manuscript through one new figure module and one new table renderer. Fig 5 is not modified. Nothing in the existing dependency graph moves except three private cache helpers, which are promoted from `_studies.py` into `_artifacts.py` so the new module can share them instead of copying them.

**Tech Stack:** Python 3.13, numpy, pandas, matplotlib, pytest, ruff 0.16.4, rpy2 3.6.7, R 4.5.1, Bioconductor 3.21, M3C (Bioconductor). Run everything from `code/` using `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-09-22-m3c-case-study-comparison-design.md`

## Global Constraints

- Work from `/Users/kaiwycik/GitHub/CARVE/code`. It is the only git repo in the tree.
- Use the pinned toolchain in `code/.venv/`. Never a system `ruff`, `pytest`, `python` or `R`.
- `ruff check src/` and `ruff format src/` only. Never run ruff on `tests/` or `notebooks/`.
- `filterwarnings = error` is on. A new warning fails the suite and needs either a `match=` assertion or a targeted `filterwarnings` mark.
- `addopts` is `-ra` only. Do not pass `--cov-fail-under=75` locally; it fails a single-file run.
- Tests mirror modules 1:1. `_m3c.py` ↔ `tests/benchmarks/test_m3c.py`. A new test belongs in the matching file.
- Dependency direction: `_types`, `_utils`, `_anndata`, `cluster` are leaves. Nothing under `_*` imports `api`. `benchmarks` imports `carve`; `carve` never imports `benchmarks`. `_m3c.py` is a leaf: it may import the standard library, numpy, pandas and `_types`, and nothing else from `benchmarks`.
- `benchmarks._theme` is the single source for palette, rcParams and geometry. Never add a module-level color literal anywhere else.
- No `logging`. Diagnostics go through `warnings.warn(..., stacklevel=2)` and `print()` gated on `verbose`.
- American spelling in prose. No bold or italics in code comments, docstrings or notebook markdown.
- The R port under `carve-r/` is out of scope. `benchmarks/` has no R counterpart, so the 1:1 mirror rule does not apply here.
- `../overleaf/` is read-only. Never write there.
- rpy2 lives in the `[notebooks]` extra. The benchmarks CI job installs `.[dev,graph,benchmarks]` and therefore has no rpy2 and no R. Every rpy2 import must be inside a function, never at module scope.
- Every commit message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

## File Structure

**Created in `src/benchmarks/`:**
- `_m3c.py` — the entire M3C bridge: pinned defaults, the result type, the pure-Python selection rule, the rpy2 call, and the on-disk cache. A leaf.

**Created in `src/benchmarks/figures/`:**
- `_klein_m3c.py` — the three-panel supplementary figure comparing M3C against CARVE and the CVIs on Klein.

**Modified in `src/benchmarks/`:**
- `_artifacts.py` — gains the three cache-fingerprint helpers moved out of `_studies.py`.
- `_studies.py` — loses those three helpers, imports them from `_artifacts` instead. No public behavior change.
- `_theme.py` — gains one `METRIC_COLORS` entry for M3C.
- `_panels.py` — gains `m3c_lines`, the RCSI-over-K primitive.
- `_tables.py` — gains `render_m3c_tex` and `write_m3c_table`.
- `figures/_case_study.py` — `ari_table` gains an optional `extra_rows` parameter, defaulting to `None`.
- `figures/__init__.py` — exports `figure_klein_m3c`.

**Modified at the repo root:**
- `Makefile` — gains an `m3c-setup` target.
- `pyproject.toml` — registers the `requires_r` pytest marker.

**Tests:**
- `tests/benchmarks/test_m3c.py` — new, mirrors `_m3c.py`.
- `tests/benchmarks/figures/test_klein_m3c.py` — new, mirrors `figures/_klein_m3c.py`.
- `tests/conftest.py` — gains the `requires_r` skip hook.
- `tests/benchmarks/test_artifacts.py` — gains the moved helpers' tests.
- `tests/benchmarks/test_panels.py` — gains `m3c_lines` tests.
- `tests/benchmarks/figures/test_case_study.py` — gains `ari_table(extra_rows=...)` tests.
- `tests/benchmarks/figures/test_figure_contract.py` — `EXPECTED` gains `figure_klein_m3c`.
- `tests/benchmarks/test_tables.py` — gains `render_m3c_tex` tests.
- `tests/fixtures/m3c_klein_scores.csv` — created in Task 9 from the real run.

**Notebook:**
- `notebooks/case_studies/Klein.ipynb` — one new section after the existing CARVE fit.

---

### Task 1: Share the cache fingerprint helpers

`_studies.py` owns three private helpers that decide whether a cached fit may be served against the `X` passed now. `_m3c.py` needs exactly the same guard. Copying them would be the config-drift pattern this project has already been bitten by, so they move to `_artifacts.py`, which already owns caching and provenance and imports only `_registry` and `_types`, so no cycle appears.

**Files:**
- Modify: `src/benchmarks/_artifacts.py`
- Modify: `src/benchmarks/_studies.py:226-262`
- Test: `tests/benchmarks/test_artifacts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `benchmarks._artifacts.fingerprint(X: np.ndarray) -> str`, `benchmarks._artifacts.fingerprint_path(cache_path: Path) -> Path`, `benchmarks._artifacts.check_fingerprint(cache_path: Path, X: np.ndarray) -> None`. All three are public names now (no leading underscore), because two modules import them.

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmarks/test_artifacts.py`:

```python
import numpy as np

from benchmarks._artifacts import check_fingerprint, fingerprint, fingerprint_path


class TestFingerprint:
    def test_same_values_give_the_same_digest(self):
        X = np.arange(12, dtype=float).reshape(4, 3)
        assert fingerprint(X) == fingerprint(X.copy())

    def test_a_different_x_of_the_same_shape_gives_a_different_digest(self):
        # Same shape deliberately: a shape check alone would pass this, which
        # is exactly what the fingerprint exists to catch.
        X = np.arange(12, dtype=float).reshape(4, 3)
        Y = X.copy()
        Y[0, 0] += 1.0
        assert fingerprint(X) != fingerprint(Y)

    def test_sidecar_sits_beside_the_cache_file(self, tmp_path):
        cache = tmp_path / "klein.carve"
        assert fingerprint_path(cache) == tmp_path / "klein.carve.x-sha1"

    def test_check_raises_when_the_recorded_digest_differs(self, tmp_path):
        cache = tmp_path / "klein.carve"
        X = np.arange(12, dtype=float).reshape(4, 3)
        Y = X.copy()
        Y[0, 0] += 1.0
        fingerprint_path(cache).write_text(fingerprint(X))
        with pytest.raises(ValueError, match="fit on a different X"):
            check_fingerprint(cache, Y)

    def test_check_warns_when_no_digest_was_recorded(self, tmp_path):
        cache = tmp_path / "klein.carve"
        X = np.arange(12, dtype=float).reshape(4, 3)
        with pytest.warns(UserWarning, match="carries no fingerprint"):
            check_fingerprint(cache, X)

    def test_check_is_silent_when_the_digest_matches(self, tmp_path, recwarn):
        cache = tmp_path / "klein.carve"
        X = np.arange(12, dtype=float).reshape(4, 3)
        fingerprint_path(cache).write_text(fingerprint(X))
        check_fingerprint(cache, X)
        assert len(recwarn) == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_artifacts.py -k Fingerprint -q`
Expected: FAIL at collection with `ImportError: cannot import name 'check_fingerprint' from 'benchmarks._artifacts'`.

- [ ] **Step 3: Move the three helpers into `_artifacts.py`**

Cut `_fingerprint`, `_fingerprint_path` and `_check_fingerprint` from `src/benchmarks/_studies.py` (lines 226-262) and paste them into `src/benchmarks/_artifacts.py` immediately above `def scenario_identity`, renamed without the leading underscore and with `numpy` and `warnings` added to that module's imports:

```python
def fingerprint(X: np.ndarray) -> str:
    """A digest of X's values, for checking a cache against the data it holds."""
    return hashlib.sha1(
        np.ascontiguousarray(np.asarray(X, dtype=np.float64)).tobytes()
    ).hexdigest()


def fingerprint_path(cache_path: Path) -> Path:
    """Where the digest sidecar for a cache file lives."""
    return cache_path.with_name(cache_path.name + ".x-sha1")


def check_fingerprint(cache_path: Path, X: np.ndarray) -> None:
    """Refuse to serve a cached result against a different X.

    A cached result is only valid for the matrix it was computed on. The hECA
    development embedding changes under one scale name (subsample-first until
    the pooled cache exists, pooled after), which is exactly the case a
    scale-keyed filename cannot catch. A cache written before this check
    existed has no record to compare against; it is served with a warning
    rather than discarded, since a fit can be hours of compute.

    Shared by fit_or_load_carve and run_or_load_m3c. It lives here rather
    than in either caller because a second copy of a correctness check is how
    the two drift apart.
    """
    sidecar = fingerprint_path(cache_path)
    if not sidecar.is_file():
        warnings.warn(
            f"{cache_path} carries no fingerprint of the X it was computed on, "
            "so it cannot be checked against the X passed now. Pass force=True "
            "to recompute if the data has changed since it was cached.",
            stacklevel=3,
        )
        return
    if sidecar.read_text().strip() != fingerprint(X):
        raise ValueError(
            f"{cache_path} was fit on a different X than the one passed now "
            "(the fingerprint differs). Serving it would report results for "
            "data it never saw. Pass force=True to refit on this X, or load "
            "the data the cache was fit on."
        )
```

In `src/benchmarks/_studies.py`, add to the existing `from ._artifacts import ...` block, or create it if absent:

```python
from ._artifacts import check_fingerprint, fingerprint, fingerprint_path
```

Then replace the three call sites inside `fit_or_load_carve`: `_check_fingerprint(cache_path, X)` becomes `check_fingerprint(cache_path, X)`, and `_fingerprint_path(cache_path).write_text(_fingerprint(X))` becomes `fingerprint_path(cache_path).write_text(fingerprint(X))`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_artifacts.py tests/benchmarks/test_studies.py -q`
Expected: PASS. `test_studies.py::...::test_a_cache_without_a_fingerprint_warns_and_loads` must still pass — it exercises the moved code through `fit_or_load_carve` and is the regression guard on this move.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_artifacts.py src/benchmarks/_studies.py tests/benchmarks/test_artifacts.py
git commit -m "$(cat <<'EOF'
refactor: share the cache fingerprint helpers from _artifacts

_m3c needs the same guard fit_or_load_carve uses. Moving the three
helpers to _artifacts, which already owns caching and provenance, means
one implementation rather than two that can drift.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: The `_m3c` pure-Python core

Everything about M3C that does not need R: the pinned defaults, the scores schema, the result type, and the selection rule. Pulling this out of the R call is what makes most of the module testable in CI, where there is no R.

**Files:**
- Create: `src/benchmarks/_m3c.py`
- Test: `tests/benchmarks/test_m3c.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `M3C_DEFAULTS: Mapping[str, Any]`, `SCORES_COLUMNS: tuple[str, ...]`, `M3CResult` (frozen dataclass with fields `scores`, `labels`, `selected_k`, `p_value`, `runtime_s`, `config`, `m3c_version`, `r_version`), `select_k_m3c(scores: pd.DataFrame) -> int`, `validate_scores(scores: pd.DataFrame) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_m3c.py`:

```python
"""Tests for the M3C bridge.

The tests here never touch R. The live-R tests live in the same file behind
the requires_r marker, added in Task 3.
"""

import numpy as np
import pandas as pd
import pytest

from benchmarks._m3c import (
    M3C_DEFAULTS,
    SCORES_COLUMNS,
    M3CResult,
    select_k_m3c,
    validate_scores,
)


def scores_frame() -> pd.DataFrame:
    """A scores frame shaped exactly as M3C returns one under objective="entropy".

    K runs 2..6, and the RCSI maximum sits at K=4, which is row 2. Both
    facts matter. A frame whose maximum sat in row 0 would let a selection
    that returns a row index instead of a K value pass, and a frame whose K
    started at 0 would let the same bug pass a second way.
    """
    return pd.DataFrame(
        {
            "K": [2, 3, 4, 5, 6],
            "ENTROPY_REAL": [0.62, 0.48, 0.31, 0.35, 0.41],
            "ENTROPY_REF": [0.65, 0.63, 0.60, 0.58, 0.57],
            "RCSI": [0.05, 0.27, 0.66, 0.51, 0.33],
            "RCSI_SE": [0.01, 0.02, 0.03, 0.03, 0.04],
            "MONTECARLO_P": [0.42, 0.08, 0.01, 0.02, 0.06],
            "NORM_P": [0.40, 0.07, 0.01, 0.02, 0.05],
            "P_SCORE": [0.40, 1.15, 2.00, 1.70, 1.30],
        }
    )


class TestSelectK:
    def test_selects_the_k_at_the_rcsi_maximum(self):
        assert select_k_m3c(scores_frame()) == 4

    def test_returns_a_k_value_not_a_row_index(self):
        # The RCSI maximum is at row 2 and at K=4. A selection that returned
        # the row index, or R's which.max + 1 transcribed to a 0-based
        # argmax, would give 2 or 3 here.
        assert select_k_m3c(scores_frame()) not in (2, 3)

    def test_works_when_the_maximum_is_the_last_row(self):
        scores = scores_frame()
        scores.loc[4, "RCSI"] = 0.99
        assert select_k_m3c(scores) == 6

    def test_raises_when_rcsi_is_absent(self):
        scores = scores_frame().drop(columns=["RCSI"])
        with pytest.raises(KeyError, match="RCSI"):
            select_k_m3c(scores)


class TestValidateScores:
    def test_accepts_the_entropy_schema(self):
        frame = scores_frame()
        assert list(validate_scores(frame).columns) == list(SCORES_COLUMNS)

    def test_rejects_the_pac_schema(self):
        # PAC_REAL and PAC_REF exist only under objective="PAC". Seeing them
        # means the run was not the one this module pins, so the numbers are
        # not the ones the spec describes.
        frame = scores_frame().rename(
            columns={"ENTROPY_REAL": "PAC_REAL", "ENTROPY_REF": "PAC_REF"}
        )
        with pytest.raises(ValueError, match="objective"):
            validate_scores(frame)

    def test_rejects_a_frame_missing_a_required_column(self):
        with pytest.raises(ValueError, match="MONTECARLO_P"):
            validate_scores(scores_frame().drop(columns=["MONTECARLO_P"]))


class TestDefaults:
    def test_pins_m3c_s_published_defaults(self):
        assert dict(M3C_DEFAULTS) == {
            "method": 1,
            "clusteralg": "pam",
            "objective": "entropy",
            "iters": 25,
            "repsref": 100,
            "repsreal": 100,
            "pItem": 0.8,
            "seed": 123,
        }

    def test_defaults_are_not_mutable(self):
        with pytest.raises(TypeError):
            M3C_DEFAULTS["iters"] = 5


class TestResult:
    def test_is_frozen(self):
        result = M3CResult(
            scores=scores_frame(),
            labels={4: np.zeros(10, dtype=int)},
            selected_k=4,
            p_value=0.01,
            runtime_s=1.5,
            config={"maxK": 10},
            m3c_version="1.34.0",
            r_version="R version 4.5.1",
        )
        with pytest.raises(AttributeError):
            result.selected_k = 5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_m3c.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'benchmarks._m3c'`.

- [ ] **Step 3: Write the module**

Create `src/benchmarks/_m3c.py`:

```python
"""Run M3C on a case study through rpy2.

M3C (John et al., 2020) is Monte Carlo reference-based consensus clustering,
an R/Bioconductor package with no Python port. This module is the whole
bridge to it: it takes the same X a case study hands CARVE and the CVIs,
runs M3C at its published defaults, and returns the scores and labels as
plain pandas and numpy.

This module is a leaf. It imports the standard library, numpy and pandas,
and nothing else from this package. rpy2 is imported inside run_m3c rather
than at module scope: rpy2 lives in the [notebooks] extra, the benchmarks CI
job installs [dev,graph,benchmarks], and a module-level import would break
collection of every test here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

#: M3C 1.34.0's published defaults, restated here rather than left to R. A
#: Bioconductor release that changes one of these then shows up as a diff in
#: this file instead of silently changing a manuscript number.
#:
#: removeplots and silent are deliberately absent: they are presentation
#: flags, not method flags, and run_m3c sets them itself.
M3C_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "method": 1,
        "clusteralg": "pam",
        "objective": "entropy",
        "iters": 25,
        "repsref": 100,
        "repsreal": 100,
        "pItem": 0.8,
        "seed": 123,
    }
)

#: The columns M3C's scores frame carries under method=1 and the default
#: objective="entropy".
#:
#: M3C builds the frame with columns named PAC_REAL and PAC_REF and then,
#: on the last line of the entropy branch, renames columns 2 and 3 in place:
#: colnames(real)[2:3] <- c("ENTROPY_REAL", "ENTROPY_REF"). The PAC names
#: survive only under objective="PAC", which also substitutes BETA_P for
#: NORM_P. Reading PAC_REAL off a default run raises KeyError.
SCORES_COLUMNS: tuple[str, ...] = (
    "K",
    "ENTROPY_REAL",
    "ENTROPY_REF",
    "RCSI",
    "RCSI_SE",
    "MONTECARLO_P",
    "NORM_P",
    "P_SCORE",
)

#: The PAC-objective spellings of columns 2 and 3, recognized only so that a
#: run under the wrong objective fails with a message that says so.
_PAC_COLUMNS: frozenset[str] = frozenset({"PAC_REAL", "PAC_REF"})


@dataclass(frozen=True)
class M3CResult:
    """One M3C run, as plain pandas and numpy.

    labels carries every K M3C clustered, not only the selected one, so a
    caller can ask what M3C says at a K chosen by something else. Labels come
    from res$realdataresults[[k]]$assignments, which is the named vector in
    the original column order; ordered_annotation is permuted by the
    dendrogram order and would scramble the mapping back onto X's rows.
    """

    scores: pd.DataFrame
    labels: dict[int, np.ndarray]
    selected_k: int
    p_value: float
    runtime_s: float
    config: Mapping[str, Any] = field(default_factory=dict)
    m3c_version: str = ""
    r_version: str = ""


def validate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Check a scores frame against the entropy-objective schema.

    Returns the frame with its columns in SCORES_COLUMNS order, so callers
    downstream of this do not depend on R's column order.
    """
    columns = set(scores.columns)
    if columns & _PAC_COLUMNS:
        raise ValueError(
            "This scores frame carries "
            f"{sorted(columns & _PAC_COLUMNS)}, which M3C produces only under "
            'objective="PAC". This module pins objective="entropy", so the '
            "run that produced this frame is not the one it describes."
        )
    missing = [name for name in SCORES_COLUMNS if name not in columns]
    if missing:
        raise ValueError(
            f"M3C's scores frame is missing {missing}. Expected the "
            f"entropy-objective schema {list(SCORES_COLUMNS)}, got "
            f"{sorted(columns)}."
        )
    return scores.loc[:, list(SCORES_COLUMNS)]


def select_k_m3c(scores: pd.DataFrame) -> int:
    """The K M3C selects: the one at the RCSI maximum.

    M3C's own expression is which.max(real$RCSI) + 1, where the plus one
    converts a 1-based row index into a K value because K starts at 2.
    Transcribing that arithmetic into Python would be wrong twice over, since
    numpy's argmax is 0-based. Read the K column instead of recomputing the
    offset -- the same reason config_id is a join key in carve and never a
    positional index.
    """
    if "RCSI" not in scores.columns:
        raise KeyError(
            "This frame has no RCSI column, so M3C's selection cannot be "
            f"recovered from it. Columns: {sorted(scores.columns)}."
        )
    return int(scores.loc[scores["RCSI"].idxmax(), "K"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_m3c.py -q`
Expected: PASS, 12 tests.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_m3c.py tests/benchmarks/test_m3c.py
git commit -m "$(cat <<'EOF'
feat: add the M3C pure-Python core

Pinned defaults, the entropy-objective scores schema, the result type and
the selection rule. Everything here is testable without R, which is what
the benchmarks CI job has.

The selection reads the K column rather than reproducing M3C's
which.max + 1, which is 1-based R arithmetic and would be wrong twice in
Python.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: The R bridge

The call itself, plus the guards that make a transposition error fail loudly instead of producing plausible garbage.

**Files:**
- Modify: `src/benchmarks/_m3c.py`
- Modify: `tests/conftest.py`
- Modify: `pyproject.toml` (the `markers` list)
- Modify: `Makefile`
- Test: `tests/benchmarks/test_m3c.py`

**Interfaces:**
- Consumes: `M3C_DEFAULTS`, `SCORES_COLUMNS`, `M3CResult`, `select_k_m3c`, `validate_scores` from Task 2.
- Produces: `run_m3c(X, *, max_k=10, allow_install=False, cores=1, verbose=False, **overrides) -> M3CResult`, `align_assignment(values: np.ndarray, names_from_r: Sequence[str], names_expected: Sequence[str], *, k: int) -> np.ndarray`, and the `requires_r` pytest marker.

- [ ] **Step 1: Register the marker and the skip hook**

In `pyproject.toml`, extend the `markers` list under `[tool.pytest.ini_options]`:

```toml
markers = [
  "requires_graph: needs the [graph] extra (igraph + leidenalg); skipped without it",
  "requires_r: needs rpy2 and the R package M3C; skipped without them",
]
```

In `tests/conftest.py`, beside the existing `_HAS_GRAPH` block, add:

```python
def _has_r_m3c() -> bool:
    """True when rpy2 imports and the R package M3C is installed.

    Both halves are checked because they fail independently: the benchmarks
    CI job has neither, a developer machine may have rpy2 from the
    [notebooks] extra without ever running `make m3c-setup`, and importing
    rpy2 without a working R raises rather than returning False.
    """
    if importlib.util.find_spec("rpy2") is None:
        return False
    try:
        import rpy2.robjects as ro

        return bool(ro.r('requireNamespace("M3C", quietly=TRUE)')[0])
    except Exception:
        return False


_HAS_R_M3C = _has_r_m3c()
```

and extend `pytest_collection_modifyitems` so it handles both markers:

```python
def pytest_collection_modifyitems(config, items):
    skips = {}
    if not _HAS_GRAPH:
        skips["requires_graph"] = pytest.mark.skip(
            reason="requires the [graph] extra (igraph + leidenalg)"
        )
    if not _HAS_R_M3C:
        skips["requires_r"] = pytest.mark.skip(
            reason="requires rpy2 and the R package M3C (make m3c-setup)"
        )
    if not skips:
        return
    for item in items:
        for keyword, mark in skips.items():
            if keyword in item.keywords:
                item.add_marker(mark)
```

In `Makefile`, add beside the existing `r-setup` target:

```make
.PHONY: m3c-setup
m3c-setup:
	Rscript -e "if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager', repos='https://cran.rstudio.com/'); BiocManager::install('M3C', ask=FALSE, update=FALSE)"
```

Note the `.PHONY: r-setup` line already present declares only `r-setup`; add the new `.PHONY` line rather than editing that one.

- [ ] **Step 2: Write the failing tests**

Append to `tests/benchmarks/test_m3c.py`:

```python
from benchmarks._m3c import align_assignment, run_m3c


class TestAlignAssignment:
    """The orientation and naming guards, tested without R.

    These are the two ways M3C's labels can come back wrong, and they fail
    differently: the wrong count means the matrix reached M3C transposed,
    while the right count under the wrong names means its gsub mangled them.
    """

    def test_reorders_by_name(self):
        out = align_assignment(
            np.array([7, 8, 9]), ["c2", "c0", "c1"], ["c0", "c1", "c2"], k=3
        )
        np.testing.assert_array_equal(out, [8, 9, 7])

    def test_raises_when_the_count_does_not_match(self):
        # The transposed case: M3C clustered the features, so it returns one
        # label per feature instead of one per sample.
        with pytest.raises(RuntimeError, match="wrong orientation"):
            align_assignment(np.array([1, 2]), ["c0", "c1"], ["c0", "c1", "c2"], k=3)

    def test_raises_when_the_names_do_not_match(self):
        # Right count, wrong names: M3C's gsub("X", "", ...) mangled them.
        with pytest.raises(RuntimeError, match="names it returned"):
            align_assignment(
                np.array([1, 2, 3]), ["0", "1", "2"], ["c0", "c1", "c2"], k=3
            )

    def test_returns_integer_labels(self):
        out = align_assignment(
            np.array([1.0, 2.0]), ["c0", "c1"], ["c0", "c1"], k=2
        )
        assert out.dtype.kind == "i"


class TestRunGuards:
    def test_refuses_to_install_unless_asked(self, monkeypatch):
        # Simulate M3C being absent so the install branch is reached on a
        # machine that has it. Without this the test would pass for the wrong
        # reason wherever M3C happens to be installed.
        monkeypatch.setattr("benchmarks._m3c._m3c_is_installed", lambda: False)
        X = np.zeros((10, 4))
        with pytest.raises(RuntimeError, match=r'BiocManager::install\("M3C"\)'):
            run_m3c(X, max_k=3, allow_install=False)

    def test_rejects_a_max_k_below_two(self):
        with pytest.raises(ValueError, match="max_k"):
            run_m3c(np.zeros((10, 4)), max_k=1)


@pytest.mark.requires_r
class TestRunLive:
    @pytest.fixture(scope="class")
    def result(self):
        # n=80, p=50 sits inside M3C's own recommended 60-1000 band, and
        # iters=5 is the low end of the vignette's own speed advice, so this
        # runs in seconds rather than minutes.
        rng = np.random.default_rng(0)
        X = np.vstack(
            [
                rng.normal(0.0, 1.0, size=(40, 50)),
                rng.normal(3.0, 1.0, size=(40, 50)),
            ]
        )
        return run_m3c(X, max_k=4, iters=5)

    def test_scores_carry_the_entropy_schema(self, result):
        assert list(result.scores.columns) == list(SCORES_COLUMNS)

    def test_scores_cover_k_two_through_max_k(self, result):
        assert result.scores["K"].tolist() == [2, 3, 4]

    def test_labels_are_returned_for_every_k(self, result):
        assert sorted(result.labels) == [2, 3, 4]

    def test_every_label_vector_has_one_entry_per_row_of_x(self, result):
        # Trivially true once align_assignment has run, which is the point:
        # the guard itself is tested in TestAlignAssignment, where it can
        # actually fail. This asserts the live call went through that guard
        # and produced labels indexed like X, not like X.T.
        assert all(labels.shape == (80,) for labels in result.labels.values())

    def test_labels_cover_exactly_k_groups(self, result):
        # Two well-separated blobs: M3C must not return a degenerate
        # partition at any swept K.
        for k, labels in result.labels.items():
            assert 1 < len(np.unique(labels)) <= k

    def test_selected_k_agrees_with_the_scores_frame(self, result):
        assert result.selected_k == select_k_m3c(result.scores)

    def test_records_the_r_side_versions(self, result):
        assert result.m3c_version
        assert result.r_version.startswith("R version")

    def test_config_records_what_was_run(self, result):
        assert result.config["maxK"] == 4
        assert result.config["iters"] == 5
        assert result.config["clusteralg"] == "pam"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/benchmarks/test_m3c.py -q`
Expected: FAIL at collection with `ImportError: cannot import name 'run_m3c'`. The `requires_r` class will skip on a machine without M3C; that is correct, and Step 6 installs it.

- [ ] **Step 4: Write the implementation**

Append to `src/benchmarks/_m3c.py`:

```python
import time
import warnings
from collections.abc import Sequence

#: The column names given to X's rows before handing them to R. M3C converts
#: a matrix to a data frame and then runs gsub("X", "", colnames(...)), which
#: mangles any name containing an X -- R's own default V1, V2 names survive,
#: but so would a corrupted name from any other scheme. Plain c0, c1, ... are
#: unaffected by that substitution.
def _sample_names(n: int) -> list[str]:
    return [f"c{i}" for i in range(n)]


def _m3c_is_installed() -> bool:
    """True when the R package M3C is available. Split out so tests can stub it."""
    import rpy2.robjects as ro

    return bool(ro.r('requireNamespace("M3C", quietly=TRUE)')[0])


def align_assignment(
    values: np.ndarray,
    names_from_r: Sequence[str],
    names_expected: Sequence[str],
    *,
    k: int,
) -> np.ndarray:
    """Put one M3C assignment back into the row order of X.

    Pure, so the two ways this can go wrong are testable without R. They fail
    differently and are worth telling apart: the wrong count means the matrix
    reached M3C in the wrong orientation, while the right count under
    unrecognized names means M3C's gsub("X", "", colnames(...)) mangled them.
    """
    values = np.asarray(values)
    if values.size != len(names_expected):
        raise RuntimeError(
            f"M3C returned {values.size} labels at K={k} for an X with "
            f"{len(names_expected)} rows. M3C takes features in rows and "
            "samples in columns, so a result this size means the matrix "
            "reached it in the wrong orientation."
        )
    by_name = pd.Series(values, index=[str(name) for name in names_from_r]).reindex(
        list(names_expected)
    )
    if by_name.isna().any():
        raise RuntimeError(
            f"M3C's labels at K={k} do not cover every sample after "
            "reindexing, which means the names it returned are not the ones "
            f"it was given. Missing: {by_name[by_name.isna()].index.tolist()[:5]}."
        )
    return by_name.to_numpy(dtype=int)


def run_m3c(
    X: np.ndarray,
    *,
    max_k: int = 10,
    allow_install: bool = False,
    cores: int = 1,
    verbose: bool = False,
    **overrides: Any,
) -> M3CResult:
    """Run M3C on X at its published defaults.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The same matrix the case study hands CARVE and the CVIs. M3C wants
        features in rows, so this is transposed here rather than by the
        caller, and the result is checked against X's own row count.
    max_k : int, default=10
        M3C's maxK. It always sweeps K from 2 to maxK; there is no lower
        bound to set. The default is M3C's own, which is also Klein's CARVE
        ceiling.
    allow_install : bool, default=False
        Installing an R package is a side effect on the user's machine, so it
        is opt-in, matching datasets/_levine.py.
    cores : int, default=1
        M3C's own parallelism over the Monte Carlo iterations. Left at 1: the
        Klein run takes about eight minutes serially, and a single worker is
        one fewer variable between runs.
    **overrides
        Individual M3C_DEFAULTS entries to override. Used by the tests to
        drop iters; a case-study run passes none.

    Returns
    -------
    M3CResult
    """
    if max_k < 2:
        raise ValueError(
            f"max_k must be at least 2, since M3C sweeps K from 2 upward; got {max_k}."
        )

    unknown = sorted(set(overrides) - set(M3C_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown M3C parameter(s) {unknown}. "
            f"Overridable names are {sorted(M3C_DEFAULTS)}."
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri

    if not _m3c_is_installed():
        if not allow_install:
            raise RuntimeError(
                "The R package M3C is not installed, and this run needs it. "
                'Run `make m3c-setup` (which calls BiocManager::install("M3C")), '
                "or re-run with allow_install=True to permit the install here."
            )
        ro.r(
            'if (!requireNamespace("BiocManager", quietly=TRUE)) '
            'install.packages("BiocManager", repos="https://cran.rstudio.com/"); '
            'BiocManager::install("M3C", ask=FALSE, update=FALSE)'
        )

    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]
    names = _sample_names(n_samples)

    config = dict(M3C_DEFAULTS) | dict(overrides) | {"maxK": int(max_k), "cores": int(cores)}

    ro.r("suppressPackageStartupMessages(library(M3C))")
    with (numpy2ri.converter + pandas2ri.converter).context():
        # Features in rows, samples in columns: M3C's required orientation.
        ro.globalenv["mydata"] = ro.r["as.data.frame"](X.T)
        ro.globalenv["sample_names"] = ro.StrVector(names)
        ro.r("colnames(mydata) <- sample_names")

        started = time.perf_counter()
        ro.globalenv["res"] = ro.r["M3C"](
            ro.globalenv["mydata"],
            maxK=config["maxK"],
            cores=config["cores"],
            iters=config["iters"],
            repsref=config["repsref"],
            repsreal=config["repsreal"],
            pItem=config["pItem"],
            clusteralg=config["clusteralg"],
            objective=config["objective"],
            method=config["method"],
            seed=config["seed"],
            removeplots=True,
            silent=not verbose,
        )
        runtime_s = time.perf_counter() - started

        scores = validate_scores(ro.r("res$scores"))
        labels = {}
        for k in range(2, int(max_k) + 1):
            # realdataresults is indexed by K itself, so element 1 is unset
            # and element k holds the K=k result. assignments is the named
            # vector in the original column order; ordered_annotation is
            # permuted by the dendrogram order and would scramble the mapping.
            assignment = ro.r(f"res$realdataresults[[{k}]]$assignments")
            names_from_r = ro.r(f"names(res$realdataresults[[{k}]]$assignments)")
            labels[k] = align_assignment(
                np.asarray(assignment), list(names_from_r), names, k=k
            )

        m3c_version = str(ro.r('as.character(packageVersion("M3C"))')[0])
        r_version = str(ro.r("R.version.string")[0])

    selected_k = select_k_m3c(scores)
    reported_k = int(len(np.unique(labels[selected_k])))
    if reported_k != selected_k:
        warnings.warn(
            f"M3C's partition at its selected K={selected_k} has {reported_k} "
            "distinct labels. This happens when a consensus cluster comes back "
            "empty; the selection stands, but the partition is coarser than K.",
            stacklevel=2,
        )

    p_value = float(scores.loc[scores["K"] == selected_k, "MONTECARLO_P"].iloc[0])
    if verbose:
        print(
            f"M3C selected K={selected_k} (Monte Carlo p={p_value:.4f}) "
            f"in {runtime_s:.1f} s."
        )
    return M3CResult(
        scores=scores.reset_index(drop=True),
        labels=labels,
        selected_k=selected_k,
        p_value=p_value,
        runtime_s=runtime_s,
        config=config,
        m3c_version=m3c_version,
        r_version=r_version,
    )
```

- [ ] **Step 5: Run the non-R tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_m3c.py -q -rs`
Expected: PASS for the pure-Python tests; the `requires_r` class reports as skipped with reason `requires rpy2 and the R package M3C (make m3c-setup)`. Confirm the skip reason appears — a silently-absent test class is the failure mode this step is checking for.

- [ ] **Step 6: Install M3C and run the live tests**

```bash
make m3c-setup
.venv/bin/pytest tests/benchmarks/test_m3c.py -q
```
Expected: PASS, with the `requires_r` class now running. If any R or rpy2 warning surfaces, `filterwarnings = error` will fail the run; add a targeted `filterwarnings` mark with a `match=` on that specific message to the affected test, never a blanket ignore, and record the message in the commit body.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_m3c.py tests/benchmarks/test_m3c.py tests/conftest.py pyproject.toml Makefile
git commit -m "$(cat <<'EOF'
feat: add the M3C rpy2 bridge

run_m3c transposes X into M3C's required orientation and checks the
result against X's own row count, because a transposed call returns
plausible output rather than raising. Labels are reindexed by explicit
c0..cN names, since M3C runs gsub("X", "", colnames(...)) on matrix
input.

rpy2 is imported inside the function: it lives in the [notebooks] extra
and the benchmarks CI job does not install it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Caching

An M3C run is about eight minutes. The figure must be regenerable without re-running R, and a cached result must never be served against a matrix it was not computed on.

**Files:**
- Modify: `src/benchmarks/_m3c.py`
- Test: `tests/benchmarks/test_m3c.py`

**Interfaces:**
- Consumes: `run_m3c`, `M3CResult` from Task 3; `check_fingerprint`, `fingerprint`, `fingerprint_path` from Task 1.
- Produces: `m3c_cache_path(*, study_name: str, scale: str, root: Path, config: Mapping[str, Any]) -> Path`, `run_or_load_m3c(X, *, cache_path: Path, force: bool = False, **kwargs) -> M3CResult`.

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmarks/test_m3c.py`:

```python
from pathlib import Path

from benchmarks._m3c import m3c_cache_path, run_or_load_m3c


def stub_result() -> M3CResult:
    return M3CResult(
        scores=scores_frame(),
        labels={2: np.zeros(6, dtype=int), 3: np.arange(6) % 3, 4: np.arange(6) % 4},
        selected_k=4,
        p_value=0.01,
        runtime_s=2.0,
        config={"maxK": 4, "clusteralg": "pam"},
        m3c_version="1.34.0",
        r_version="R version 4.5.1 (2025-06-13)",
    )


class TestCachePath:
    def test_encodes_study_and_scale(self):
        path = m3c_cache_path(
            study_name="klein", scale="publication", root=Path("/tmp"), config={"maxK": 10}
        )
        assert path.name.startswith("m3c_klein_publication_")
        assert path.suffix == ".parquet"

    def test_a_different_config_gives_a_different_path(self):
        base = dict(study_name="klein", scale="publication", root=Path("/tmp"))
        a = m3c_cache_path(**base, config={"maxK": 10})
        b = m3c_cache_path(**base, config={"maxK": 17})
        assert a != b


class TestRunOrLoad:
    def test_runs_once_then_serves_the_cache(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(X, **kwargs):
            calls.append(X)
            return stub_result()

        monkeypatch.setattr("benchmarks._m3c.run_m3c", fake_run)
        X = np.arange(18, dtype=float).reshape(6, 3)
        cache = tmp_path / "m3c.parquet"

        first = run_or_load_m3c(X, cache_path=cache, max_k=4)
        second = run_or_load_m3c(X, cache_path=cache, max_k=4)

        assert len(calls) == 1
        assert second.selected_k == first.selected_k == 4
        assert second.p_value == pytest.approx(0.01)
        assert second.m3c_version == "1.34.0"
        pd.testing.assert_frame_equal(second.scores, first.scores)
        np.testing.assert_array_equal(second.labels[3], first.labels[3])

    def test_refuses_a_cache_computed_on_a_different_x(self, tmp_path, monkeypatch):
        monkeypatch.setattr("benchmarks._m3c.run_m3c", lambda X, **kw: stub_result())
        X = np.arange(18, dtype=float).reshape(6, 3)
        # Same shape deliberately, so only the fingerprint can catch it.
        Y = X.copy()
        Y[0, 0] += 1.0
        cache = tmp_path / "m3c.parquet"

        run_or_load_m3c(X, cache_path=cache, max_k=4)
        with pytest.raises(ValueError, match="fit on a different X"):
            run_or_load_m3c(Y, cache_path=cache, max_k=4)

    def test_force_recomputes(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "benchmarks._m3c.run_m3c",
            lambda X, **kw: (calls.append(X), stub_result())[1],
        )
        X = np.arange(18, dtype=float).reshape(6, 3)
        cache = tmp_path / "m3c.parquet"

        run_or_load_m3c(X, cache_path=cache, max_k=4)
        run_or_load_m3c(X, cache_path=cache, max_k=4, force=True)
        assert len(calls) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_m3c.py -k "CachePath or RunOrLoad" -q`
Expected: FAIL at collection with `ImportError: cannot import name 'm3c_cache_path'`.

- [ ] **Step 3: Write the implementation**

Append to `src/benchmarks/_m3c.py`, and add `import hashlib`, `import json` and `from pathlib import Path` to its imports. `Path` belongs here rather than in Task 3, where nothing used it and ruff would have flagged it:

```python
def m3c_cache_path(
    *, study_name: str, scale: str, root: Path, config: Mapping[str, Any]
) -> Path:
    """Where a study's M3C result is cached, per scale and per configuration.

    The config hash is part of the filename for the same reason
    carve_cache_path carries a run key: run_or_load_m3c loads whatever file
    sits at the path it is handed, so two runs that differ in maxK or in the
    inner algorithm must not share a name.
    """
    key = hashlib.sha1(
        json.dumps(dict(config), sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return Path(root) / f"m3c_{study_name}_{scale}_{key}.parquet"


def _write_cache(path: Path, result: M3CResult, X: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.scores.copy()
    for k in sorted(result.labels):
        frame[f"labels_{k}"] = pd.Series(result.labels[k]).reindex(frame.index)
    # Labels are longer than the scores frame whenever n_samples exceeds the
    # number of swept K values, which is always, so they go in their own file
    # rather than being padded into this one.
    frame.to_parquet(path, index=False)
    pd.DataFrame(result.labels).to_parquet(_labels_path(path), index=False)
    _sidecar_path(path).write_text(
        json.dumps(
            {
                "selected_k": result.selected_k,
                "p_value": result.p_value,
                "runtime_s": result.runtime_s,
                "config": dict(result.config),
                "m3c_version": result.m3c_version,
                "r_version": result.r_version,
            },
            indent=2,
            default=str,
        )
    )
    fingerprint_path(path).write_text(fingerprint(X))


def _labels_path(path: Path) -> Path:
    return path.with_name(path.stem + ".labels.parquet")


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.stem + ".json")


def _read_cache(path: Path) -> M3CResult:
    scores = validate_scores(pd.read_parquet(path))
    labels_frame = pd.read_parquet(_labels_path(path))
    meta = json.loads(_sidecar_path(path).read_text())
    return M3CResult(
        scores=scores.reset_index(drop=True),
        labels={
            int(column): labels_frame[column].to_numpy(dtype=int)
            for column in labels_frame.columns
        },
        selected_k=int(meta["selected_k"]),
        p_value=float(meta["p_value"]),
        runtime_s=float(meta["runtime_s"]),
        config=meta["config"],
        m3c_version=str(meta["m3c_version"]),
        r_version=str(meta["r_version"]),
    )


def run_or_load_m3c(
    X: np.ndarray, *, cache_path: Path, force: bool = False, **kwargs: Any
) -> M3CResult:
    """Run M3C on X, caching the result, and serve the cache on later calls.

    A Klein run is about eight minutes, so the cache is what makes
    regenerating the figure practical. The fingerprint guard is the same one
    fit_or_load_carve uses: a cached result is never served against a matrix
    it was not computed on.
    """
    cache_path = Path(cache_path)
    if cache_path.is_file() and not force:
        check_fingerprint(cache_path, X)
        return _read_cache(cache_path)

    result = run_m3c(X, **kwargs)
    _write_cache(cache_path, result, X)
    return result
```

Add to the module's imports:

```python
from ._artifacts import check_fingerprint, fingerprint, fingerprint_path
```

This is the one import `_m3c` takes from the package, and `_artifacts` imports only `_registry` and `_types`, so `_m3c` stays a leaf with respect to everything that could form a cycle.

Delete the dead `frame[f"labels_{k}"]` loop from `_write_cache` before running: the labels go in their own parquet, and the loop above it was left in the draft by mistake. `_write_cache` should read:

```python
def _write_cache(path: Path, result: M3CResult, X: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result.scores.to_parquet(path, index=False)
    pd.DataFrame(result.labels).to_parquet(_labels_path(path), index=False)
    _sidecar_path(path).write_text(
        json.dumps(
            {
                "selected_k": result.selected_k,
                "p_value": result.p_value,
                "runtime_s": result.runtime_s,
                "config": dict(result.config),
                "m3c_version": result.m3c_version,
                "r_version": result.r_version,
            },
            indent=2,
            default=str,
        )
    )
    fingerprint_path(path).write_text(fingerprint(X))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_m3c.py -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_m3c.py tests/benchmarks/test_m3c.py
git commit -m "$(cat <<'EOF'
feat: cache M3C results with the shared fingerprint guard

Scores, labels and metadata go to three files beside each other, keyed on
study, scale and a hash of the pinned config. The fingerprint sidecar is
the same guard fit_or_load_carve uses, so a cached result is never served
against a matrix it was not computed on.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: The RCSI panel

M3C's own selection evidence, drawn the way CARVE's and the CVIs' are drawn in the panels beside it.

**Files:**
- Modify: `src/benchmarks/_theme.py:24-58` (the `METRIC_COLORS` dict)
- Modify: `src/benchmarks/_panels.py`
- Test: `tests/benchmarks/test_panels.py`

**Interfaces:**
- Consumes: `M3CResult` from Task 2.
- Produces: `benchmarks._panels.m3c_lines(ax: Axes, scores: pd.DataFrame, *, selected_k: int, title: str | None = None) -> Axes`.

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmarks/test_panels.py`:

```python
class TestM3CLines:
    @pytest.fixture
    def scores(self):
        return pd.DataFrame(
            {
                "K": [2, 3, 4, 5],
                "RCSI": [0.05, 0.27, 0.66, 0.51],
                "RCSI_SE": [0.01, 0.02, 0.03, 0.03],
                "MONTECARLO_P": [0.42, 0.08, 0.01, 0.02],
            }
        )

    def test_draws_one_line_over_k(self, scores):
        from benchmarks._panels import m3c_lines

        _, ax = plt.subplots()
        m3c_lines(ax, scores, selected_k=4)
        assert len(ax.lines) >= 1
        x, y = ax.lines[0].get_data()
        np.testing.assert_array_equal(x, [2, 3, 4, 5])
        np.testing.assert_allclose(y, [0.05, 0.27, 0.66, 0.51])

    def test_marks_the_selected_k(self, scores):
        from benchmarks._panels import m3c_lines

        _, ax = plt.subplots()
        m3c_lines(ax, scores, selected_k=4)
        verticals = [
            line.get_xdata()[0]
            for line in ax.lines
            if line.get_linestyle() == "--" and len(set(line.get_xdata())) == 1
        ]
        assert 4 in verticals

    def test_draws_error_bars_from_rcsi_se(self, scores):
        from benchmarks._panels import m3c_lines

        _, ax = plt.subplots()
        m3c_lines(ax, scores, selected_k=4)
        assert len(ax.containers) >= 1

    def test_uses_the_theme_color_not_the_fallback(self, scores):
        from benchmarks._panels import m3c_lines
        from benchmarks._theme import FALLBACK_COLOR, metric_color

        assert metric_color("m3c_rcsi") != FALLBACK_COLOR
        _, ax = plt.subplots()
        m3c_lines(ax, scores, selected_k=4)
        assert ax.lines[0].get_color() == metric_color("m3c_rcsi")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_panels.py -k M3CLines -q`
Expected: FAIL with `ImportError: cannot import name 'm3c_lines' from 'benchmarks._panels'`.

- [ ] **Step 3: Add the color and the panel**

In `src/benchmarks/_theme.py`, add one entry to `METRIC_COLORS`, immediately after the `"gap"` entry and before the `baseline_oracle` comment block:

```python
    # M3C's RCSI. Teal, from the same Okabe-Ito family as CARVE's own
    # measures above, but distinct from both of them and from the four
    # classical indices, because it is neither a CARVE criterion nor a
    # geometric CVI and must not be mistaken for either.
    "m3c_rcsi": "#0072B2",
```

In `src/benchmarks/_panels.py`, add beside `cvi_lines`:

```python
def m3c_lines(
    ax: Axes,
    scores: pd.DataFrame,
    *,
    selected_k: int,
    title: str | None = None,
) -> Axes:
    """Plot M3C's RCSI over K with its own error bars, marking the selection.

    RCSI is M3C's selection statistic: the mean difference, on the log scale,
    between the reference stability scores and the real one at each K. It is
    drawn unnormalized, unlike cvi_lines' indices, because it has a
    meaningful zero -- RCSI at or below zero means the real data is no more
    stable at that K than M3C's Monte Carlo reference.

    The error bars are plus or minus 1.96 RCSI_SE, which is M3C's own plot
    idiom, and the dashed vertical marks the selected K the way cvi_lines and
    carve_lines mark theirs.
    """
    ordered = scores.sort_values("K")
    color = metric_color("m3c_rcsi")

    ax.errorbar(
        ordered["K"].to_numpy(),
        ordered["RCSI"].to_numpy(dtype=float),
        yerr=1.96 * ordered["RCSI_SE"].to_numpy(dtype=float),
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        color=color,
        ecolor=color,
        elinewidth=1.0,
        capsize=3.0,
        label="M3C RCSI",
    )
    ax.axhline(0.0, color=FOREGROUND_COLOR, linestyle=":", linewidth=1.0, alpha=0.5)
    ax.axvline(int(selected_k), color=color, linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Number of clusters $k$", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("RCSI", fontsize=FONT_SIZES["axis_label"])
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    return style_axes(ax)
```

`FOREGROUND_COLOR` is already defined in `_theme.py`; add it to `_panels.py`'s existing `from ._theme import ...` block if it is not already imported there.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_panels.py -k M3CLines -q`
Expected: PASS, 4 tests.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_theme.py src/benchmarks/_panels.py tests/benchmarks/test_panels.py
git commit -m "$(cat <<'EOF'
feat: add the M3C RCSI panel primitive

RCSI is drawn unnormalized because it has a meaningful zero: at or below
zero the real data is no more stable at that K than M3C's Monte Carlo
reference. A zero rule marks it.

The color is a new METRIC_COLORS entry rather than a literal in _panels,
distinct from both CARVE's measures and the four classical indices
because RCSI is neither.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Let `ari_table` carry extra rows

The ARI panel is shared with the Levine and hECA figures. It needs to carry two more rows for Klein without changing what those two draw.

**Files:**
- Modify: `src/benchmarks/figures/_case_study.py:164-191`
- Test: `tests/benchmarks/figures/test_case_study.py`

**Interfaces:**
- Consumes: `CompositeInputs` (already exists).
- Produces: `ari_table(inputs: CompositeInputs, *, extra_rows: Sequence[Mapping[str, Any]] | None = None) -> pd.DataFrame`, unchanged for callers that pass nothing.

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmarks/figures/test_case_study.py`:

```python
class TestAriTableExtraRows:
    def test_default_is_unchanged(self, composite_inputs):
        from benchmarks.figures._case_study import ari_table

        assert "M3C" not in ari_table(composite_inputs)["method"].tolist()

    def test_extra_rows_are_appended_after_the_cvi_rows(self, composite_inputs):
        from benchmarks.figures._case_study import ari_table

        extra = [
            {"method": "M3C", "metric": "m3c_rcsi", "ari": 0.41, "k": 2},
            {"method": "M3C (k=4)", "metric": "m3c_rcsi", "ari": 0.55, "k": 4},
        ]
        table = ari_table(composite_inputs, extra_rows=extra)
        assert table["method"].tolist()[-2:] == ["M3C", "M3C (k=4)"]
        assert table["ari"].tolist()[-2:] == [0.41, 0.55]

    def test_extra_rows_keep_the_frame_s_columns(self, composite_inputs):
        from benchmarks.figures._case_study import ari_table

        base = ari_table(composite_inputs)
        table = ari_table(
            composite_inputs,
            extra_rows=[{"method": "M3C", "metric": "m3c_rcsi", "ari": 0.41, "k": 2}],
        )
        assert list(table.columns) == list(base.columns)

    def test_rejects_a_row_missing_a_column(self, composite_inputs):
        from benchmarks.figures._case_study import ari_table

        with pytest.raises(ValueError, match="ari"):
            ari_table(composite_inputs, extra_rows=[{"method": "M3C", "k": 2}])
```

If `test_case_study.py` has no `composite_inputs` fixture, add one built from `tests/benchmarks/_helpers.StubCarve`, following whatever that file's existing tests already construct.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/figures/test_case_study.py -k AriTableExtraRows -q`
Expected: FAIL with `TypeError: ari_table() got an unexpected keyword argument 'extra_rows'`.

- [ ] **Step 3: Extend the function**

Replace `ari_table` in `src/benchmarks/figures/_case_study.py` with:

```python
_ARI_ROW_COLUMNS: tuple[str, ...] = ("method", "metric", "ari", "k")


def ari_table(
    inputs: CompositeInputs,
    *,
    extra_rows: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """ARI of each selection against the reported labels.

    Shared by the Levine and hECA figures, whose panel F is this table
    rendered as a lollipop chart rather than an alluvial -- the ARI table is
    the number both case studies exist to report, so a fix to one must reach
    both rather than living in two copies that can drift apart.

    extra_rows appends comparators that do not come out of best_df, which is
    the CVI sweep's own output and has no room for a tool that is not a CVI.
    The Klein M3C figure passes M3C's two rows here. Levine and hECA pass
    nothing and are unaffected.
    """
    rows = [
        {
            "method": "CARVE",
            "metric": "ari_stability_1se",
            "ari": float(adjusted_rand_score(inputs.y, inputs.carve_labels)),
            "k": int(len(set(np.asarray(inputs.carve_labels).tolist()))),
        }
    ]
    for _, row in inputs.best_df.iterrows():
        rows.append(
            {
                "method": str(row["metric"]).replace("_", " ").title(),
                "metric": str(row["metric"]),
                "ari": float(row["ari"]),
                "k": int(row["k"]),
            }
        )
    for extra in extra_rows or ():
        missing = [name for name in _ARI_ROW_COLUMNS if name not in extra]
        if missing:
            raise ValueError(
                f"An extra ARI row is missing {missing}. Every row needs "
                f"{list(_ARI_ROW_COLUMNS)}, so the lollipop can color it by "
                "metric and annotate it with k."
            )
        rows.append({name: extra[name] for name in _ARI_ROW_COLUMNS})
    return pd.DataFrame(rows)
```

Add `Mapping` to the `collections.abc` import at the top of the file if it is not already there.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/figures/ -q`
Expected: PASS. The Levine and hECA figure tests must still pass unchanged; they are the regression guard on the default path.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/figures/_case_study.py tests/benchmarks/figures/test_case_study.py
git commit -m "$(cat <<'EOF'
feat: let ari_table carry comparators outside best_df

best_df is the CVI sweep's own output and has no room for a tool that is
not a CVI. extra_rows appends them after the CVI rows, defaulting to
None so the Levine and hECA figures are unaffected.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: The Klein M3C figure

**Files:**
- Create: `src/benchmarks/figures/_klein_m3c.py`
- Modify: `src/benchmarks/figures/__init__.py`
- Modify: `tests/benchmarks/figures/test_figure_contract.py:9-24`
- Test: `tests/benchmarks/figures/test_klein_m3c.py`

**Interfaces:**
- Consumes: `m3c_lines` (Task 5), `ari_table(extra_rows=...)` (Task 6), `M3CResult` (Task 2), and `CompositeInputs`, `composite_color_maps`, `_align_to_reference` from `figures/_case_study.py`.
- Produces: `figure_klein_m3c(inputs: CompositeInputs, m3c: M3CResult, *, save: bool = True, out_dir: Path | None = None) -> Figure`, and `m3c_ari_rows(inputs: CompositeInputs, m3c: M3CResult) -> list[dict[str, Any]]`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/figures/test_klein_m3c.py`:

```python
"""Tests for the Klein M3C comparison figure."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from benchmarks._m3c import M3CResult
from benchmarks.figures._klein_m3c import figure_klein_m3c, m3c_ari_rows


@pytest.fixture
def m3c_result():
    rng = np.random.default_rng(0)
    n = 40
    return M3CResult(
        scores=pd.DataFrame(
            {
                "K": [2, 3, 4, 5],
                "ENTROPY_REAL": [0.62, 0.48, 0.31, 0.35],
                "ENTROPY_REF": [0.65, 0.63, 0.60, 0.58],
                "RCSI": [0.66, 0.27, 0.20, 0.11],
                "RCSI_SE": [0.01, 0.02, 0.03, 0.03],
                "MONTECARLO_P": [0.01, 0.08, 0.12, 0.20],
                "NORM_P": [0.01, 0.07, 0.11, 0.19],
                "P_SCORE": [2.0, 1.15, 0.96, 0.72],
            }
        ),
        labels={k: rng.integers(0, k, size=n) for k in (2, 3, 4, 5)},
        selected_k=2,
        p_value=0.01,
        runtime_s=480.0,
        config={"maxK": 10},
        m3c_version="1.34.0",
        r_version="R version 4.5.1 (2025-06-13)",
    )


class TestAriRows:
    def test_returns_one_row_at_the_selected_k_and_one_at_four(
        self, composite_inputs, m3c_result
    ):
        rows = m3c_ari_rows(composite_inputs, m3c_result)
        assert [row["k"] for row in rows] == [2, 4]
        assert rows[0]["method"] == "M3C"
        assert rows[1]["method"] == "M3C (at $k=4$)"

    def test_every_row_carries_the_columns_ari_table_requires(
        self, composite_inputs, m3c_result
    ):
        for row in m3c_ari_rows(composite_inputs, m3c_result):
            assert set(row) == {"method", "metric", "ari", "k"}

    def test_returns_one_row_when_m3c_also_selects_four(
        self, composite_inputs, m3c_result
    ):
        # Panel C must not show the same partition twice under a second name.
        at_four = dataclasses.replace(m3c_result, selected_k=4)
        rows = m3c_ari_rows(composite_inputs, at_four)
        assert [row["k"] for row in rows] == [4]


class TestFigure:
    def test_returns_a_figure_with_three_panels(
        self, composite_inputs, m3c_result, tmp_path
    ):
        fig = figure_klein_m3c(
            composite_inputs, m3c_result, save=False, out_dir=tmp_path
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3

    def test_saves_under_its_manuscript_filename(
        self, composite_inputs, m3c_result, tmp_path
    ):
        figure_klein_m3c(composite_inputs, m3c_result, save=True, out_dir=tmp_path)
        assert (tmp_path / "klein_m3c.png").is_file()

    def test_does_not_save_when_asked_not_to(
        self, composite_inputs, m3c_result, tmp_path
    ):
        figure_klein_m3c(composite_inputs, m3c_result, save=False, out_dir=tmp_path)
        assert not (tmp_path / "klein_m3c.png").exists()
```

Reuse the `composite_inputs` fixture from Task 6; if it lives in `test_case_study.py`, move it into `tests/benchmarks/figures/conftest.py` so both files share one definition rather than two that can drift.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/figures/test_klein_m3c.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'benchmarks.figures._klein_m3c'`.

- [ ] **Step 3: Write the figure module**

Create `src/benchmarks/figures/_klein_m3c.py`:

```python
"""The Klein M3C comparison, a supplementary figure.

Fig 5 is not modified. It is a published six-panel composite whose panels the
main text and S6 Text cite by letter, so a new panel there would mean
rewriting prose. This is a separate figure that reuses Fig 5's embedding and
color maps, so the two read together.

Three panels in every case. M3C's partition at k=4, which answers what M3C
says at the k CARVE selected, is a row in the ARI panel rather than a fourth
scatter: drawing it conditionally would give the figure a panel count that
depends on the result, and drawing it unconditionally would duplicate panel B
whenever M3C also selects 4.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import adjusted_rand_score

from .._m3c import M3CResult
from .._panels import ari_lollipop, m3c_lines, panel_letter, scatter_clusters
from .._theme import save_figure, theme_context
from ._case_study import (
    CompositeInputs,
    _align_to_reference,
    ari_table,
    composite_color_maps,
)
from ._paths import CASE_STUDY_DIR, figure_path

SAVE_NAME = "klein_m3c.png"
MARKER_SIZE = 20.0
AXIS_LABELS = ("PC1", "PC2")

#: The k CARVE selects on Klein, per STUDIES["klein"].reported_k. M3C's
#: partition at this k is reported so the two tools can be compared at one
#: granularity as well as at each one's own choice.
CARVE_K = 4


def m3c_ari_rows(inputs: CompositeInputs, m3c: M3CResult) -> list[dict[str, Any]]:
    """M3C's rows for the ARI panel: at its own selection, and at CARVE's k.

    Returns one row when M3C also selects CARVE's k, since two rows would
    then be the same partition under two names.
    """
    ks = [m3c.selected_k] if m3c.selected_k == CARVE_K else [m3c.selected_k, CARVE_K]
    rows = []
    for k in ks:
        labels = _align_to_reference(np.asarray(m3c.labels[k]), inputs.y)
        rows.append(
            {
                "method": "M3C" if k == m3c.selected_k else f"M3C (at $k={k}$)",
                "metric": "m3c_rcsi",
                "ari": float(adjusted_rand_score(inputs.y, labels)),
                "k": int(k),
            }
        )
    return rows


def figure_klein_m3c(
    inputs: CompositeInputs,
    m3c: M3CResult,
    *,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Draw the three-panel Klein M3C comparison."""
    with theme_context():
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

        m3c_lines(
            axes[0],
            m3c.scores,
            selected_k=m3c.selected_k,
            title="M3C: Relative Cluster Stability Index",
        )
        panel_letter(axes[0], "A")

        _, _, comparison_cmap = composite_color_maps(inputs)
        labels = _align_to_reference(
            np.asarray(m3c.labels[m3c.selected_k]), inputs.y
        )
        scatter_clusters(
            axes[1],
            inputs.Z,
            labels,
            cmap=comparison_cmap,
            marker_size=MARKER_SIZE,
            axis_labels=AXIS_LABELS,
            title=f"M3C clustering ($k={m3c.selected_k}$)",
        )
        panel_letter(axes[1], "B")

        ari_lollipop(
            axes[2],
            ari_table(inputs, extra_rows=m3c_ari_rows(inputs, m3c)),
            title="Agreement with Reported Labels (ARI)",
        )
        panel_letter(axes[2], "C")

        fig.tight_layout()

    if save:
        save_figure(fig, figure_path(SAVE_NAME, subdir=CASE_STUDY_DIR, out_dir=out_dir))
    return fig
```

Check `scatter_clusters`' actual signature in `src/benchmarks/_panels.py:94` before writing this call and match it exactly; the parameter names above follow the pattern the other figure modules use, but the signature in the source is authoritative.

In `src/benchmarks/figures/__init__.py`, add the import in alphabetical position and the name to `__all__`:

```python
from ._klein_m3c import figure_klein_m3c
```

In `tests/benchmarks/figures/test_figure_contract.py`, add `"figure_klein_m3c"` to the `EXPECTED` tuple. Leave `NOT_YET_IMPLEMENTED` empty; it is a one-way ratchet and this figure is being implemented now.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/figures/ -q`
Expected: PASS, including the contract tests, which now also hold `figure_klein_m3c` to returning a `Figure`, accepting `save` and `out_dir`, and never calling `plt.show`.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/figures/_klein_m3c.py src/benchmarks/figures/__init__.py tests/benchmarks/figures/
git commit -m "$(cat <<'EOF'
feat: add the Klein M3C comparison figure

Three panels: M3C's RCSI curve with its own error bars, its partition in
Fig 5's embedding and color map, and the ARI lollipop Klein's Fig 5 does
not carry.

Fig 5 itself is untouched. Its panels are cited by letter in the main
text and S6 Text, so a new panel there would mean rewriting prose.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: The comparison table

**Files:**
- Modify: `src/benchmarks/_tables.py`
- Test: `tests/benchmarks/test_tables.py`

**Interfaces:**
- Consumes: the frame `ari_table(inputs, extra_rows=m3c_ari_rows(...))` returns (Tasks 6 and 7).
- Produces: `render_m3c_tex(ari_df: pd.DataFrame, *, caption: str, label: str, decimals: int = 3) -> str`, `write_m3c_table(ari_df: pd.DataFrame, *, out_dir: Path, name: str = "si_table_m3c", caption: str = ..., label: str = "tab:klein_m3c") -> Path`.

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmarks/test_tables.py`:

```python
class TestRenderM3CTex:
    @pytest.fixture
    def ari_df(self):
        return pd.DataFrame(
            [
                {"method": "CARVE", "metric": "ari_generalizability_1se", "ari": 0.81, "k": 4},
                {"method": "Silhouette", "metric": "silhouette", "ari": 0.42, "k": 2},
                {"method": "M3C", "metric": "m3c_rcsi", "ari": 0.41, "k": 2},
            ]
        )

    def test_renders_one_row_per_method(self, ari_df):
        from benchmarks._tables import render_m3c_tex

        tex = render_m3c_tex(ari_df, caption="Klein comparison.", label="tab:klein_m3c")
        for method in ("CARVE", "Silhouette", "M3C"):
            assert method in tex

    def test_carries_the_caption_and_label(self, ari_df):
        from benchmarks._tables import render_m3c_tex

        tex = render_m3c_tex(ari_df, caption="Klein comparison.", label="tab:klein_m3c")
        assert r"\caption{Klein comparison.}" in tex
        assert r"\label{tab:klein_m3c}" in tex

    def test_escapes_latex_specials_in_the_caption(self, ari_df):
        from benchmarks._tables import render_m3c_tex

        tex = render_m3c_tex(ari_df, caption="50% of cells", label="tab:x")
        assert r"50\% of cells" in tex

    def test_rounds_ari_to_the_requested_precision(self, ari_df):
        from benchmarks._tables import render_m3c_tex

        tex = render_m3c_tex(
            ari_df, caption="c", label="l", decimals=2
        )
        assert "0.81" in tex
        assert "0.810" not in tex

    def test_writes_a_tex_file(self, ari_df, tmp_path):
        from benchmarks._tables import write_m3c_table

        path = write_m3c_table(ari_df, out_dir=tmp_path)
        assert path == tmp_path / "si_table_m3c.tex"
        assert r"\begin{table}" in path.read_text()

    def test_writes_a_csv_beside_the_tex(self, ari_df, tmp_path):
        from benchmarks._tables import write_m3c_table

        write_m3c_table(ari_df, out_dir=tmp_path)
        written = pd.read_csv(tmp_path / "si_table_m3c.csv")
        assert written["method"].tolist() == ["CARVE", "Silhouette", "M3C"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/benchmarks/test_tables.py -k M3CTex -q`
Expected: FAIL with `ImportError: cannot import name 'render_m3c_tex'`.

- [ ] **Step 3: Write the renderer**

Append to `src/benchmarks/_tables.py`:

```python
#: The default caption. Stated here rather than at the call site so the
#: notebook and any later caller cannot caption the same table differently.
M3C_CAPTION: str = (
    "Selections and their agreement with the reported labels on the Klein "
    "case study, for CARVE, the four classical validation indices, and M3C "
    "run at its published defaults."
)


def render_m3c_tex(
    ari_df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    decimals: int = 3,
) -> str:
    """One row per method: its selected k and its ARI against the reported labels.

    Runtime is deliberately not a column. fit_or_load_carve caches state
    rather than timings, so a runtime column would be measured under
    different conditions for CARVE than for M3C; the two runtimes are
    reported in prose instead.
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & $k$ & ARI \\",
        r"\midrule",
    ]
    for _, row in ari_df.iterrows():
        method = str(row["method"])
        # Method names may already carry math, as "M3C (at $k=4$)" does, so
        # only names with no math are escaped.
        rendered = method if "$" in method else _tex_escape(method)
        lines.append(f"{rendered} & ${int(row['k'])}$ & {_fmt(row['ari'], decimals)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_m3c_table(
    ari_df: pd.DataFrame,
    *,
    out_dir: Path,
    name: str = "si_table_m3c",
    caption: str = M3C_CAPTION,
    label: str = "tab:klein_m3c",
    decimals: int = 3,
) -> Path:
    """Write the comparison table as .tex, with the underlying frame as .csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tex"
    path.write_text(
        render_m3c_tex(ari_df, caption=caption, label=label, decimals=decimals)
    )
    ari_df.to_csv(out_dir / f"{name}.csv", index=False)
    return path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/benchmarks/test_tables.py -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add src/benchmarks/_tables.py tests/benchmarks/test_tables.py
git commit -m "$(cat <<'EOF'
feat: render the Klein M3C comparison table

One row per method with its selected k and its ARI against the reported
labels, plus the underlying frame as CSV. Runtime is not a column: CARVE's
fit is cached without timings, so the two runtimes are not measured
comparably and belong in prose.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Run it, record the fixture, and wire up the notebook

The first real run. This produces the manuscript numbers and turns the synthetic test fixture into a recorded one.

**Files:**
- Modify: `notebooks/case_studies/Klein.ipynb`
- Create: `tests/fixtures/m3c_klein_scores.csv`
- Modify: `tests/benchmarks/test_m3c.py`

**Interfaces:**
- Consumes: everything from Tasks 2 through 8.
- Produces: `vis/case_studies/klein_m3c.png`, `vis/case_studies/si_table_m3c.tex`, `vis/case_studies/si_table_m3c.csv`, and the recorded scores fixture.

- [ ] **Step 1: Add the notebook section**

In `notebooks/case_studies/Klein.ipynb`, after the cell that calls `prepare_composite`, add one markdown cell and one code cell.

Markdown cell:

```markdown
## Comparison against M3C

M3C (Monte Carlo reference-based consensus clustering) is run on the same matrix
CARVE and the CVIs see, at its published defaults: PAM with Euclidean distance,
maxK=10, 25 Monte Carlo iterations, 100 inner replications, entropy objective.
Klein's CARVE sweep covers k in 2 to 10, which is M3C's default maxK, so both
tools search the same range with nothing adjusted in either direction.

The run takes about eight minutes and is cached, so re-running this cell serves
the cache. Pass force=True to recompute.
```

Code cell:

```python
from benchmarks._m3c import m3c_cache_path, run_or_load_m3c
from benchmarks._tables import write_m3c_table
from benchmarks.figures import CASE_STUDY_DIR, figure_klein_m3c
from benchmarks.figures._klein_m3c import m3c_ari_rows
from benchmarks.figures._case_study import ari_table

m3c = run_or_load_m3c(
    X,
    cache_path=m3c_cache_path(
        study_name=study.name,
        scale=study.default_scale,
        root=Path("./carve_state_saves"),
        config={"maxK": max(study.candidate_k)},
    ),
    max_k=max(study.candidate_k),
    verbose=True,
)
print(f"M3C selected k={m3c.selected_k}, Monte Carlo p={m3c.p_value:.4f}")
print(f"M3C {m3c.m3c_version} on {m3c.r_version}, {m3c.runtime_s / 60:.1f} min")

show(figure_klein_m3c(inputs, m3c))
write_m3c_table(
    ari_table(inputs, extra_rows=m3c_ari_rows(inputs, m3c)), out_dir=CASE_STUDY_DIR
)
```

- [ ] **Step 2: Install M3C if Task 3 did not, and execute the notebook section**

```bash
make m3c-setup
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/case_studies/Klein.ipynb
```

Expected: the cell prints M3C's selected k, its Monte Carlo p-value, the versions and the runtime, and writes `vis/case_studies/klein_m3c.png` plus the two table files. Record the printed numbers; they are the manuscript numbers.

If the run warns that the partition at the selected K is coarser than K, record that too — it is a reportable result, not a failure.

- [ ] **Step 3: Record the real scores as a test fixture**

This writes the fixture and prints the exact assertion line Step 4 needs, so no number is transcribed by hand:

```bash
.venv/bin/python -c "
from pathlib import Path
from benchmarks._m3c import m3c_cache_path, select_k_m3c, _read_cache
from benchmarks._studies import STUDIES
study = STUDIES['klein']
path = m3c_cache_path(
    study_name='klein', scale=study.default_scale,
    root=Path('notebooks/case_studies/carve_state_saves'),
    config={'maxK': max(study.candidate_k)},
)
result = _read_cache(path)
result.scores.to_csv('tests/fixtures/m3c_klein_scores.csv', index=False)
print(result.scores.to_string(index=False))
print()
print('Paste this line into TestRecordedKleinScores:')
print(f'        assert select_k_m3c(recorded) == {result.selected_k}')
print()
print(f'For the manuscript: k={result.selected_k}, p={result.p_value:.4f}, '
      f'M3C {result.m3c_version}, {result.runtime_s / 60:.1f} min')
"
```

- [ ] **Step 4: Write the regression test against the recorded fixture**

Append to `tests/benchmarks/test_m3c.py`:

```python
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "m3c_klein_scores.csv"


class TestRecordedKleinScores:
    """Pin the published Klein numbers against M3C's real output.

    The synthetic frame above fixes the semantics; this fixes the result.
    Regenerating it means the manuscript numbers moved, which must be a
    deliberate act rather than a silent one.
    """

    @pytest.fixture
    def recorded(self):
        return pd.read_csv(FIXTURE)

    def test_carries_the_entropy_schema(self, recorded):
        assert list(validate_scores(recorded).columns) == list(SCORES_COLUMNS)

    def test_sweeps_k_two_through_ten(self, recorded):
        assert recorded["K"].tolist() == list(range(2, 11))

    def test_selection_is_the_k_the_run_reported(self, recorded):
        # Step 3 prints this line with the real k substituted. Paste it here.
        assert select_k_m3c(recorded) == <k printed by Step 3>
```

Replace the last line with the one Step 3 printed. The file will not import until you do, which is deliberate: the fixture and the number that describes it land together or not at all.

- [ ] **Step 5: Run the full suite**

```bash
.venv/bin/pytest tests/benchmarks -q
.venv/bin/pytest tests --ignore=tests/benchmarks -q
```
Expected: PASS for both legs. The second leg is the carve job and should be unaffected by everything in this plan; run it to confirm that.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/
git add notebooks/case_studies/Klein.ipynb tests/fixtures/m3c_klein_scores.csv tests/benchmarks/test_m3c.py
git commit -m "$(cat <<'EOF'
feat: run M3C on Klein and record its scores

The notebook cell runs M3C at its published defaults on the same matrix
CARVE and the CVIs see, draws the comparison figure and writes the table.
The recorded scores frame pins the published numbers, so a change to them
has to be deliberate.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Deliverables for the manuscript

`../overleaf/` is read-only. After Task 9, hand the author:

- `vis/case_studies/klein_m3c.png`, to be copied into `overleaf/vis/` manually, per the existing convention that regenerating a figure never silently changes the manuscript.
- `vis/case_studies/si_table_m3c.tex`.
- The printed numbers: M3C's selected k, its Monte Carlo p-value, its ARI against the reported labels at its own k and at k=4, its version, and its runtime.
- The vignette quotation already recorded in the spec's "The authors' own scope statement" section, for the S6 Text paragraph and the R2.1 response.

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task: the module and its interface to Tasks 2 through 4; caching, provenance and reproducibility to Tasks 1 and 4; reporting to Tasks 5 through 8; testing to the test steps throughout plus Task 9's recorded fixture; dependencies to Task 3's Makefile and marker changes; running it and the manuscript handoff to Task 9 and the Deliverables section. The spec's three Klein-run facts (ref_method stays reverse-pca, des is None, maxK is not extended at the boundary) are honored by Task 3 passing no `des` and no `ref_method`, leaving M3C's own auto-selection in place, and by Task 9 passing `max_k=max(study.candidate_k)` rather than a larger value.

**Two things the plan adds that the spec did not specify.** `validate_scores` rejecting the PAC schema, which exists because the spec's own schema error made the case for a loud failure; and the warning when M3C's partition at the selected K has fewer than K distinct labels, which the spec's Risks section implies but does not name.

**Placeholder scan.** One value in the plan is not knowable before the run: the k M3C selects on Klein. Task 9 Step 3 prints the exact assertion line rather than asking anyone to transcribe a number, and the test file does not import until it is pasted in, so the fixture and the number describing it land together.

**Defects found and fixed in this review.** Three, all in Task 3. The label reindex cast to `dtype=int` before checking for missing entries, so a mismatch would have raised an opaque pandas error instead of the intended message. The shape check that followed it ran after a reindex onto `names`, so it could never fail — the orientation guard it was meant to be did not exist. Both are now `align_assignment`, a pure function with its own CI tests that distinguish the two failures: wrong count means a transposed matrix, right count under unrecognized names means M3C's `gsub` mangled them. Separately, `from pathlib import Path` was introduced in Task 3, where nothing used it, and moved to Task 4.

**Known soft spot.** Task 7 Step 3 instructs the implementer to check `scatter_clusters`' real signature before writing the call, rather than asserting it. That is deliberate: `_panels.py:94` is the authority and its parameter names were not read out during planning. The same applies to whether `test_case_study.py` already has a `composite_inputs` fixture, which Task 6 Step 1 tells the implementer to check.

**Type consistency.** `m3c_cache_path`'s keyword-only signature is the same in Task 4's tests and Task 9's notebook cell. `run_or_load_m3c` forwards `**kwargs` to `run_m3c`, which is what lets Task 9 pass `max_k` and `verbose` through it. Rows appended through `ari_table(extra_rows=...)` carry `metric="m3c_rcsi"`, which is the `METRIC_COLORS` key Task 5 adds, so `ari_lollipop` colors them from the theme rather than falling back to grey.
