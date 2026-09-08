# scATAC-seq and Large-Scale Case Studies Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: add two case studies to `src/benchmarks/` — the Cusanovich mouse sci-ATAC atlas for accuracy on sparse chromatin data, and pooled hECA v2.0 ATAC organs for scalability above 500,000 cells.

Architecture: both studies follow the existing case-study pipeline unchanged, contributing a loader under `datasets/`, an entry in `STUDIES`, a composite figure, and a notebook. Scale becomes a first-class part of the `Study` configuration so implementation runs against subsamples and publication runs against the whole dataset, with the resolved scale carried into the fit cache filename. Deviations from the existing two studies happen only where the data forces them, and each is recorded with its reason.

Tech Stack: Python 3.12, NumPy, SciPy sparse, scikit-learn, scanpy, anndata, h5py, pandas, matplotlib, pytest.

Spec: `docs/superpowers/specs/2026-09-07-atac-and-large-scale-case-studies-design.md` (section 5 is what this plan implements)

Depends on: `docs/superpowers/plans/2026-09-08-anchored-consensus.md`. Tasks 6 and 8 of this plan need `CARVE(consensus_anchors=...)` to exist. Tasks 1 through 5, 7, 9 and 10 do not, so this plan can begin before that one lands.

## Global Constraints

- Dependency direction: `benchmarks` imports `carve`; `carve` never imports `benchmarks`. Within `carve`, `_types`, `_utils`, `_anndata` and `cluster` are leaves and nothing under `_*` imports `api`.
- Tests mirror modules one to one. `datasets/_cusanovich.py` maps to `tests/benchmarks/datasets/test_cusanovich.py`, `figures/_heca_results.py` to `tests/benchmarks/figures/test_heca_results.py`, and so on.
- `config_id` is a join key, never a positional index.
- Case-study configuration is read from `STUDIES`, never re-derived at a call site. Four separate manuscript mismatches on this project came from re-derivation. If a task needs an estimator, a candidate-k range, a subsample size or an anchor count, it reads it from the `Study`.
- Seeds are derived arithmetically and never shared. No global seeding.
- `SweepSpec` is frozen: one run sweeps exactly one parameter. A k-based and a resolution-based sweep are two separate CARVE runs and cannot be combined in one grid.
- One theme. `benchmarks._theme` is the single source for palette, rcParams and geometry. Never add a module-level color literal.
- No `logging`. Diagnostics go through `warnings.warn(..., stacklevel=2)` and `print()` gated on `verbose`.
- No matplotlib in `_studies.py`. Compute lives there; rendering lives in `figures/`.
- Lint and format with the pinned ruff: `.venv/bin/ruff check src/` and `.venv/bin/ruff format src/`, version `==0.16.4`. Never a system ruff. Do not run ruff on `tests/` or `notebooks/`.
- The venv is uv-managed and has no pip. Install with `VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"`.
- Run tests in the foreground with a long timeout. `tests/benchmarks/` alone is about seven minutes.
- `data/` is gitignored working output. Downloaded archives and derived caches go there and are never committed.
- Figures flow code to manuscript by manual re-export. Writing a figure into `vis/case_studies/` does not update the paper.
- American spelling in prose. No bold or italics in authored text, including code comments, docstrings and notebook markdown.
- Never commit a downloaded dataset, an extracted h5ad, or a `.carve` cache.

All commands are run from `/Users/kaiwycik/GitHub/CARVE/code`.

## File structure

| File | Responsibility |
| --- | --- |
| `src/benchmarks/datasets/_cusanovich.py` | Load and preprocess the mouse sci-ATAC atlas: peak filter, TF-IDF, SVD, subsample |
| `src/benchmarks/datasets/_heca.py` | Load and preprocess pooled hECA organs: chunked feature selection, chunked reduction, log-normalization, PCA |
| `src/benchmarks/datasets/__init__.py` | Re-export the loaders |
| `src/benchmarks/_types.py` | `Study` gains a scale mapping; loaders become scale-parameterized |
| `src/benchmarks/_estimators.py` | Register MiniBatchKMeans and Leiden; add resolution grids |
| `src/benchmarks/_studies.py` | `STUDIES` entries, scale resolution, cache paths, `study_scaling_sweep` |
| `src/benchmarks/figures/_cusanovich_results.py` | The Cusanovich composite, alluvial bottom panel |
| `src/benchmarks/figures/_heca_results.py` | The hECA composite, ARI bottom panel, subsampled scatters |
| `src/benchmarks/figures/_study_scaling.py` | Runtime, peak memory and selected k against n |
| `notebooks/case_studies/Cusanovich.ipynb`, `hECA.ipynb` | The two study narratives |

---

### Task 1: Register MiniBatchKMeans and Leiden as benchmark estimators

Files:
- Modify: `src/benchmarks/_types.py` (the `KNOWN_ESTIMATORS` frozenset near line 12)
- Modify: `src/benchmarks/_estimators.py`
- Test: `tests/benchmarks/test_benchmarks_types.py`, `tests/benchmarks/test_estimators.py`

Interfaces:
- Consumes: nothing.
- Produces: `KNOWN_ESTIMATORS` additionally contains `"minibatch_kmeans"` and `"leiden"`. `resolution_grids(spec: EstimatorSpec, resolutions: Sequence[float]) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]` builds a resolution-swept grid. `RESOLUTION_ESTIMATORS: frozenset[str]` names the specs that sweep resolution rather than `n_clusters`.

Spectral clustering builds a dense n-by-n affinity and Ward agglomerative is quadratic in memory, so neither can run at 81,173 or 500,000 samples. MiniBatchKMeans and Leiden are what remain, and Leiden additionally answers the reviewer request to compare against graph-based clustering.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_benchmarks_types.py`:

```python
import pytest

from benchmarks._types import KNOWN_ESTIMATORS, EstimatorSpec


def test_graph_and_minibatch_estimators_are_known():
    assert "leiden" in KNOWN_ESTIMATORS
    assert "minibatch_kmeans" in KNOWN_ESTIMATORS
    assert EstimatorSpec(name="leiden").name == "leiden"
    assert EstimatorSpec(name="minibatch_kmeans").name == "minibatch_kmeans"


def test_unknown_estimator_still_raises():
    with pytest.raises(ValueError, match="Unknown estimator"):
        EstimatorSpec(name="kmenas")
```

Append to `tests/benchmarks/test_estimators.py`:

```python
import pytest
from sklearn.cluster import MiniBatchKMeans

from benchmarks._estimators import (
    RESOLUTION_ESTIMATORS,
    build_estimator,
    param_grids,
    resolution_grids,
)
from benchmarks._types import EstimatorSpec


class TestResolutionEstimators:
    def test_leiden_sweeps_resolution_not_n_clusters(self):
        assert "leiden" in RESOLUTION_ESTIMATORS
        assert "kmeans" not in RESOLUTION_ESTIMATORS

    def test_resolution_grid_shape(self):
        from carve.cluster import LeidenClustering

        grids = resolution_grids(EstimatorSpec(name="leiden"), [0.1, 0.2, 0.3])
        assert len(grids) == 1
        cls, grid = grids[0]
        assert cls is LeidenClustering
        assert grid["resolution"] == [0.1, 0.2, 0.3]
        assert "n_clusters" not in grid

    def test_resolution_grid_rejects_a_k_based_estimator(self):
        with pytest.raises(ValueError, match="does not sweep resolution"):
            resolution_grids(EstimatorSpec(name="kmeans"), [0.1])

    def test_param_grids_rejects_a_resolution_estimator(self):
        # SweepSpec is frozen: one run sweeps exactly one parameter, so a
        # resolution estimator must never be handed an n_clusters grid.
        with pytest.raises(ValueError, match="sweeps resolution"):
            param_grids(EstimatorSpec(name="leiden"), [2, 3])

    def test_minibatch_kmeans_builds_with_n_clusters(self):
        est = build_estimator(
            EstimatorSpec(name="minibatch_kmeans"), n_clusters=4, random_state=0
        )
        assert isinstance(est, MiniBatchKMeans)
        assert est.n_clusters == 4
        assert est.random_state == 0

    def test_build_estimator_rejects_a_resolution_estimator(self):
        with pytest.raises(ValueError, match="sweeps resolution"):
            build_estimator(EstimatorSpec(name="leiden"), n_clusters=4, random_state=0)
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/benchmarks/test_benchmarks_types.py tests/benchmarks/test_estimators.py -v`
Expected: FAIL, `ValueError: Unknown estimator 'leiden'` and `ImportError: cannot import name 'resolution_grids'`.

- [ ] Step 3: Extend the known-estimator set

In `src/benchmarks/_types.py`, replace the `KNOWN_ESTIMATORS` assignment with:

```python
KNOWN_ESTIMATORS: frozenset[str] = frozenset(
    {"kmeans", "minibatch_kmeans", "agglomerative", "spectral", "leiden"}
)
```

- [ ] Step 4: Extend the estimator registry

In `src/benchmarks/_estimators.py`, add `MiniBatchKMeans` to the scikit-learn import and `LeidenClustering` to the `carve.cluster` import, then extend the two tables and add the new functions:

```python
ESTIMATOR_CLASSES: dict[str, type[ClusterMixin]] = {
    "kmeans": KMeans,
    "minibatch_kmeans": MiniBatchKMeans,
    "agglomerative": AgglomerativeClustering,
    "spectral": SpectralClustering,
    "leiden": LeidenClustering,
}

ESTIMATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    # n_init is pinned so results do not move when scikit-learn changes its
    # default, which it has done before.
    "kmeans": {"n_init": 10},
    "minibatch_kmeans": {"n_init": 10},
    "agglomerative": {"linkage": "ward"},
    "spectral": {"affinity": "self_tuning"},
    # 15 neighbors is the scanpy convention practitioners will recognize;
    # modularity is Leiden's default objective. The two objective functions
    # use different resolution scales and must not share a grid.
    "leiden": {"n_neighbors": 15, "objective_function": "modularity"},
}

# Estimators whose granularity is swept through resolution rather than
# n_clusters. A single CARVE run sweeps exactly one parameter, so these
# cannot appear in the same grid as a k-based estimator.
RESOLUTION_ESTIMATORS: frozenset[str] = frozenset({"leiden"})


def resolution_grids(
    spec: EstimatorSpec, resolutions: Sequence[float]
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """Build the estimator_param_grids structure for a resolution sweep.

    The number of clusters is an outcome here, not an input, so CARVE is run
    in resolution mode and reports the observed cluster count per value.
    """
    if spec.name not in RESOLUTION_ESTIMATORS:
        raise ValueError(
            f"Estimator {spec.name!r} does not sweep resolution. Use "
            "param_grids for k-based estimators."
        )
    if len(resolutions) == 0:
        raise ValueError("resolutions must not be empty.")

    grid: dict[str, list[Any]] = {"resolution": [float(r) for r in resolutions]}
    for key, value in ESTIMATOR_DEFAULTS[spec.name].items():
        grid[key] = [value]

    return [(ESTIMATOR_CLASSES[spec.name], grid)]
```

Add a guard as the first statement of both `build_estimator` and `param_grids`:

```python
    if spec.name in RESOLUTION_ESTIMATORS:
        raise ValueError(
            f"Estimator {spec.name!r} sweeps resolution, not n_clusters. Use "
            "resolution_grids and CARVE's resolution mode."
        )
```

- [ ] Step 5: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/benchmarks/test_benchmarks_types.py tests/benchmarks/test_estimators.py -v`
Expected: PASS.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/benchmarks/_types.py src/benchmarks/_estimators.py tests/benchmarks/
git commit -m "feat(benchmarks): register MiniBatchKMeans and Leiden estimators"
```

---

### Task 2: Scale-aware Study configuration

Files:
- Modify: `src/benchmarks/_types.py` (the `Study` dataclass near lines 152-168)
- Modify: `src/benchmarks/_studies.py` (`_klein_loader`, `_levine_loader`, `STUDIES`, `fit_or_load_carve`)
- Test: `tests/benchmarks/test_benchmarks_types.py`, `tests/benchmarks/test_studies.py`

Interfaces:
- Consumes: nothing.
- Produces: `Study` gains `scales: Mapping[str, int | float | None]` and `default_scale: str`, and `Study.loader` becomes `Callable[[int | float | None], tuple[Any, Any, Mapping[str, Any]]]`. New module functions in `_studies.py`: `resolve_scale(study: Study, scale: str | None) -> int | float | None`, `load_study(study: Study, *, scale: str | None = None) -> tuple[Any, Any, Mapping[str, Any]]`, and `carve_cache_path(study: Study, *, scale: str | None = None, root: Path) -> Path`.

Implementation runs against subsamples and the publication run uses the whole dataset. Without scale in the cache filename, a publication run silently loads a development-scale fit, because `fit_or_load_carve` loads whatever file sits at the path it is given. That is the specific defect this mechanism exists to prevent.

Note: this renames the existing caches. `notebooks/case_studies/carve_state_saves/carve_klein.carve` becomes `carve_klein_publication.carve`, so the Klein cache must be regenerated once, which takes about six minutes at 1,358 cells with `n_jobs=8`. The Levine cache is already unusable (it pickles a class since renamed) so nothing is lost there.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_benchmarks_types.py`:

```python
from benchmarks._types import Study


def test_study_requires_its_default_scale_to_exist():
    import pytest

    with pytest.raises(ValueError, match="default_scale"):
        Study(
            name="s",
            loader=lambda subsample: (None, None, {}),
            estimator=EstimatorSpec(name="kmeans"),
            candidate_k=(2, 3),
            scales={"dev": 100},
            default_scale="publication",
        )


def test_study_requires_a_nonempty_scale_map():
    import pytest

    with pytest.raises(ValueError, match="at least one scale"):
        Study(
            name="s",
            loader=lambda subsample: (None, None, {}),
            estimator=EstimatorSpec(name="kmeans"),
            candidate_k=(2, 3),
            scales={},
            default_scale="dev",
        )
```

Append to `tests/benchmarks/test_studies.py`:

```python
from pathlib import Path

import pytest

from benchmarks._studies import STUDIES, carve_cache_path, load_study, resolve_scale
from benchmarks._types import EstimatorSpec, Study


def _study(**kw):
    base = dict(
        name="demo",
        loader=lambda subsample: (subsample, None, {"subsample": subsample}),
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=(2, 3),
        scales={"dev": 100, "publication": None},
        default_scale="dev",
    )
    base.update(kw)
    return Study(**base)


class TestScaleResolution:
    def test_default_scale_is_used_when_none_is_given(self):
        assert resolve_scale(_study(), None) == 100

    def test_named_scale_wins(self):
        assert resolve_scale(_study(), "publication") is None

    def test_unknown_scale_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="dev"):
            resolve_scale(_study(), "enormous")

    def test_load_study_passes_the_resolved_size_to_the_loader(self):
        # The loader echoes its argument, so this pins that configuration
        # travels from STUDIES rather than being re-derived at the call site.
        X, _, meta = load_study(_study(), scale="dev")
        assert X == 100
        assert meta["subsample"] == 100


class TestCarveCachePath:
    def test_scale_is_part_of_the_filename(self, tmp_path):
        dev = carve_cache_path(_study(), scale="dev", root=tmp_path)
        pub = carve_cache_path(_study(), scale="publication", root=tmp_path)
        assert dev != pub
        assert "dev" in dev.name
        assert "publication" in pub.name
        assert dev.suffix == ".carve"

    def test_study_name_is_part_of_the_filename(self, tmp_path):
        a = carve_cache_path(_study(name="alpha"), scale="dev", root=tmp_path)
        b = carve_cache_path(_study(name="beta"), scale="dev", root=tmp_path)
        assert a != b


class TestRegisteredStudiesCarryScales:
    def test_every_study_declares_its_default_scale(self):
        for study in STUDIES.values():
            assert study.default_scale in study.scales

    def test_klein_publication_scale_is_the_published_half_subsample(self):
        assert STUDIES["klein"].scales["publication"] == 0.5

    def test_levine_publication_scale_is_five_thousand(self):
        assert STUDIES["levine32"].scales["publication"] == 5000
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/benchmarks/test_studies.py tests/benchmarks/test_benchmarks_types.py -v`
Expected: FAIL, `ImportError: cannot import name 'resolve_scale'`.

- [ ] Step 3: Extend the Study dataclass

In `src/benchmarks/_types.py`, replace the `Study` dataclass with:

```python
@dataclass(frozen=True)
class Study:
    """One case study: a real dataset run through the same pipeline.

    A Study has no axis. Its loader is parameterized by a subsample size so
    that development runs cheaply against a subsample and the publication run
    uses the whole dataset, with both sizes declared here rather than chosen
    at a call site.
    """

    name: str
    loader: Callable[[int | float | None], tuple[Any, Any, Mapping[str, Any]]]
    estimator: EstimatorSpec
    candidate_k: tuple[int, ...]
    scales: Mapping[str, int | float | None]
    default_scale: str
    resolutions: tuple[float, ...] = ()
    consensus_anchors: int | None = None
    k_star: int | None = None

    def __post_init__(self) -> None:
        if not self.candidate_k:
            raise ValueError(f"Study {self.name!r}: candidate_k must not be empty.")
        if not self.scales:
            raise ValueError(f"Study {self.name!r}: declare at least one scale.")
        if self.default_scale not in self.scales:
            raise ValueError(
                f"Study {self.name!r}: default_scale {self.default_scale!r} is "
                f"not among the declared scales {sorted(self.scales)}."
            )
```

- [ ] Step 4: Add the scale helpers and update the existing loaders

In `src/benchmarks/_studies.py`, replace `_klein_loader` and `_levine_loader` and add the three helpers:

```python
def _klein_loader(subsample: int | float | None):
    from .datasets import load_klein

    # The manuscript's 1,358 cells is a 0.5 subsample of the 2,717-cell
    # preprocessed set (Klein sample size, manuscript line 606).
    return load_klein(subsample=subsample, random_state=42)


def _levine_loader(subsample: int | float | None):
    from .datasets import load_levine32

    # Manuscript line 627: a stratified subsample of 5,000 cells.
    return load_levine32(subsample=subsample, random_state=42)


def resolve_scale(study: Study, scale: str | None) -> int | float | None:
    """Resolve a scale name to the subsample size the loader receives."""
    name = study.default_scale if scale is None else scale
    if name not in study.scales:
        raise ValueError(
            f"Study {study.name!r} has no scale {name!r}. "
            f"Declared scales are {sorted(study.scales)}."
        )
    return study.scales[name]


def load_study(
    study: Study, *, scale: str | None = None
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a study's data at a named scale, recording the scale in meta."""
    name = study.default_scale if scale is None else scale
    X, y, meta = study.loader(resolve_scale(study, name))
    meta = dict(meta)
    meta["scale"] = name
    meta["study"] = study.name
    return X, y, meta


def carve_cache_path(study: Study, *, scale: str | None = None, root: Path) -> Path:
    """Where a study's fitted CARVE state is cached, per scale.

    The scale is part of the filename deliberately. fit_or_load_carve loads
    whatever file sits at the path it is handed, so a shared path would let a
    publication run silently reuse a development-scale fit.
    """
    name = study.default_scale if scale is None else scale
    if name not in study.scales:
        raise ValueError(
            f"Study {study.name!r} has no scale {name!r}. "
            f"Declared scales are {sorted(study.scales)}."
        )
    return Path(root) / f"carve_{study.name}_{name}.carve"
```

Update the two existing `STUDIES` entries to declare scales:

```python
    "klein": Study(
        name="klein",
        loader=_klein_loader,
        estimator=EstimatorSpec(name="agglomerative"),
        candidate_k=tuple(range(2, 11)),
        scales={"dev": 400, "publication": 0.5},
        default_scale="publication",
    ),
    "levine32": Study(
        name="levine32",
        loader=_levine_loader,
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=tuple(range(7, 18)),
        scales={"dev": 800, "publication": 5000},
        default_scale="publication",
    ),
```

Add `from pathlib import Path` and `from typing import Any` to the imports if they are not already present.

- [ ] Step 5: Update the existing tests that construct a Study

`tests/benchmarks/test_studies.py::TestStudies::test_each_study_has_a_loader_and_candidate_k` and any other test that builds a `Study` directly must now pass `scales` and `default_scale`, and any loader stub must accept one positional argument. Run the file and fix each failure by adding the two fields; do not relax an assertion to accommodate the new signature.

- [ ] Step 6: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/benchmarks/test_studies.py tests/benchmarks/test_benchmarks_types.py -v`
Expected: PASS.

- [ ] Step 7: Verify the cache-path guard can fail

Temporarily change `carve_cache_path` to return `Path(root) / f"carve_{study.name}.carve"`, dropping the scale. Rerun `tests/benchmarks/test_studies.py::TestCarveCachePath`. Expected: `test_scale_is_part_of_the_filename` FAILS. Revert and confirm PASS. Without this test the silent wrong-cache defect returns undetected.

- [ ] Step 8: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/benchmarks/_types.py src/benchmarks/_studies.py tests/benchmarks/
git commit -m "feat(benchmarks): make case-study scale first-class configuration"
```

---

### Task 3: Cusanovich mouse sci-ATAC loader

Files:
- Create: `src/benchmarks/datasets/_cusanovich.py`
- Modify: `src/benchmarks/datasets/__init__.py`
- Test: `tests/benchmarks/datasets/test_cusanovich.py`

Interfaces:
- Consumes: `resolve_data_dir` from `datasets/_klein.py`.
- Produces: `load_cusanovich(*, root: Path | None = None, subsample: int | float | None = None, random_state: int = 42, site_frequency_threshold: float = 0.03, n_components: int = 50, label_column: str = "tissue") -> tuple[np.ndarray, pd.Series, dict]`.

The preprocessing follows the source publication's own `dim_reduction.R`: keep peaks accessible in at least 3% of cells, TF-IDF with per-cell term frequency and inverse document frequency `log(1 + n_cells / cells_per_peak)`, then a 50-component SVD taking cell coordinates as the right singular vectors scaled by the singular values. All 50 components are kept. The source does not drop the first component; dropping LSI component 1 is a later Signac and ArchR convention, and since the reference labels are outputs of the source pipeline, matching it is correct.

Expected files under `data/Cusanovich/`, downloaded manually from `http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release/`:

| File | Size |
| --- | --- |
| `matrices/atac_matrix.binary.qc_filtered.mtx.gz` | 1.1 GB |
| `matrices/atac_matrix.binary.qc_filtered.cells.txt` | 2.9 MB |
| `matrices/atac_matrix.binary.qc_filtered.peaks.txt` | 10.0 MB |
| `metadata/cell_metadata.txt` | 13.3 MB |

The matrix is peaks by cells, 436,206 by 81,173, with 421,971,103 nonzeros.

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/datasets/test_cusanovich.py`:

```python
"""Cusanovich loader tests.

The real matrix is 1.1 GB, so every test builds a small fixture in the same
on-disk layout and exercises the preprocessing chain against it.
"""

import gzip

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

from benchmarks.datasets import load_cusanovich

N_CELLS = 120
N_PEAKS = 400


@pytest.fixture
def atlas(tmp_path):
    rng = np.random.default_rng(0)
    tissues = np.repeat(["Lung", "Liver", "Spleen"], N_CELLS // 3)

    # Three tissue-specific peak blocks plus a shared background, so the
    # embedding has structure a clustering can actually recover.
    M = np.zeros((N_PEAKS, N_CELLS), dtype=np.int8)
    for block, tissue in enumerate(["Lung", "Liver", "Spleen"]):
        rows = slice(block * 100, (block + 1) * 100)
        cols = tissues == tissue
        M[rows, cols] = rng.random((100, cols.sum())) < 0.6
    M[300:, :] = rng.random((100, N_CELLS)) < 0.3

    root = tmp_path / "data"
    d = root / "Cusanovich"
    (d / "matrices").mkdir(parents=True)
    (d / "metadata").mkdir(parents=True)

    path = d / "matrices" / "atac_matrix.binary.qc_filtered.mtx"
    mmwrite(str(path), sparse.csr_matrix(M), field="integer")
    with open(path, "rb") as src, gzip.open(str(path) + ".gz", "wb") as dst:
        dst.write(src.read())
    path.unlink()

    cells = [f"cell{i:04d}" for i in range(N_CELLS)]
    (d / "matrices" / "atac_matrix.binary.qc_filtered.cells.txt").write_text(
        "\n".join(cells) + "\n"
    )
    (d / "matrices" / "atac_matrix.binary.qc_filtered.peaks.txt").write_text(
        "\n".join(f"chr1_{i}_{i + 100}" for i in range(N_PEAKS)) + "\n"
    )
    pd.DataFrame(
        {
            "cell": cells,
            "tissue": tissues,
            "cluster": rng.integers(1, 5, N_CELLS),
            "cell_label": np.where(rng.random(N_CELLS) < 0.15, "Unknown", tissues),
        }
    ).to_csv(d / "metadata" / "cell_metadata.txt", sep="\t", index=False)
    return root


class TestLoadCusanovich:
    def test_returns_cells_by_components(self, atlas):
        X, y, meta = load_cusanovich(root=atlas, n_components=10)
        assert X.ndim == 2
        assert X.shape[1] == 10
        assert X.shape[0] == y.shape[0]
        assert np.isfinite(X).all()

    def test_unknown_cells_are_dropped(self, atlas):
        X, y, meta = load_cusanovich(root=atlas, n_components=10)
        assert "Unknown" not in set(y)
        assert meta["n_cells"] < N_CELLS
        assert meta["n_cells_full"] == N_CELLS

    def test_label_column_selects_the_reference(self, atlas):
        _, tissue, _ = load_cusanovich(root=atlas, n_components=10)
        _, label, _ = load_cusanovich(
            root=atlas, n_components=10, label_column="cell_label"
        )
        assert tissue.name == "tissue"
        assert label.name == "cell_label"

    def test_site_frequency_threshold_filters_peaks(self, atlas):
        _, _, loose = load_cusanovich(
            root=atlas, n_components=10, site_frequency_threshold=0.0
        )
        _, _, strict = load_cusanovich(
            root=atlas, n_components=10, site_frequency_threshold=0.5
        )
        assert strict["n_peaks_kept"] < loose["n_peaks_kept"]

    def test_embedding_separates_the_planted_tissues(self, atlas):
        # A loader that returned noise would satisfy every shape assertion
        # above, so pin that the preprocessing preserves real structure.
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        X, y, _ = load_cusanovich(root=atlas, n_components=10)
        labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_subsample_is_stratified_and_sized(self, atlas):
        X, y, meta = load_cusanovich(root=atlas, n_components=10, subsample=30)
        assert X.shape[0] == 30
        assert meta["subsample"] == 30
        assert y.nunique() == 3

    def test_meta_records_the_preprocessing_chain(self, atlas):
        _, _, meta = load_cusanovich(root=atlas, n_components=10)
        chain = " ".join(meta["preprocessing"])
        assert "TF-IDF" in chain
        assert "3%" in chain or "site_frequency_threshold" in chain
        assert meta["drops_first_component"] is False
        assert meta["source"] == "Cusanovich"

    def test_missing_data_directory_names_the_download(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="mouse_atlas_data_release"):
            load_cusanovich(root=tmp_path / "nothing")
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/benchmarks/datasets/test_cusanovich.py -v`
Expected: FAIL, `ImportError: cannot import name 'load_cusanovich'`.

- [ ] Step 3: Implement the loader

Create `src/benchmarks/datasets/_cusanovich.py`:

```python
"""Cusanovich et al. mouse sci-ATAC-seq atlas.

436,206 peaks by 81,173 cells of binarized chromatin accessibility across 13
adult mouse tissues. The reference label used here is the tissue the nucleus
was dissected from, which is externally determined rather than an output of
anybody's clustering, and is therefore the same kind of independent ground
truth as the Klein study's collection timepoints.

Preprocessing follows the source publication's own dim_reduction.R: a 3%
site-frequency threshold, TF-IDF, and a 50-component SVD whose cell
coordinates are the right singular vectors scaled by the singular values.
All 50 components are kept, because the source keeps them; dropping LSI
component 1 is a later Signac and ArchR convention.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ._klein import resolve_data_dir

DOWNLOAD_ROOT = (
    "http://krishna.gs.washington.edu/content/members/ajh24/"
    "mouse_atlas_data_release/"
)

_MATRIX = "matrices/atac_matrix.binary.qc_filtered.mtx.gz"
_CELLS = "matrices/atac_matrix.binary.qc_filtered.cells.txt"
_METADATA = "metadata/cell_metadata.txt"

#: Label value marking cells the source left unannotated. Dropped, mirroring
#: the Levine study's removal of the uncharacterized population 15.
UNKNOWN_LABEL = "Unknown"


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download the Cusanovich atlas from "
            f"{DOWNLOAD_ROOT} into data/Cusanovich/, keeping the "
            "matrices/ and metadata/ subdirectories."
        )
    return path


def _read_matrix(path: Path) -> sparse.csr_matrix:
    """Read the peaks-by-cells Matrix Market file."""
    from scipy.io import mmread

    with gzip.open(path, "rb") as handle:
        return sparse.csr_matrix(mmread(handle))


def _tfidf(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """TF-IDF exactly as the source computes it.

    Term frequency divides each cell column by its column sum; inverse
    document frequency is log(1 + n_cells / cells_per_peak). The input is
    peaks by cells and binary, so a peak's row sum is the number of cells it
    is accessible in.
    """
    counts = counts.tocsr().astype(np.float64)
    n_cells = counts.shape[1]

    cell_sums = np.asarray(counts.sum(axis=0)).ravel()
    cell_sums[cell_sums == 0] = 1.0
    peak_sums = np.asarray(counts.sum(axis=1)).ravel()
    peak_sums[peak_sums == 0] = 1.0

    term_frequency = counts @ sparse.diags(1.0 / cell_sums)
    inverse_document = np.log1p(n_cells / peak_sums)
    return sparse.diags(inverse_document) @ term_frequency


def load_cusanovich(
    *,
    root: Path | None = None,
    subsample: int | float | None = None,
    random_state: int = 42,
    site_frequency_threshold: float = 0.03,
    n_components: int = 50,
    label_column: str = "tissue",
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess the Cusanovich mouse sci-ATAC atlas.

    Parameters
    ----------
    root : Path or None
        Directory holding the dataset folders. Defaults to data/.
    subsample : int, float, or None
        Stratified subsample size, as a count or a fraction. None keeps all.
    random_state : int
        Seed for the stratified subsample and the SVD.
    site_frequency_threshold : float
        Keep peaks accessible in at least this fraction of cells. 0.03 is the
        source publication's own value.
    n_components : int
        SVD components retained. All of them are kept; none is dropped.
    label_column : str
        Metadata column used as the reference label. "tissue" is externally
        determined; "cell_label" is the source's marker-based annotation.

    Returns
    -------
    (X, y, meta)
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import StratifiedShuffleSplit

    data_dir = resolve_data_dir("Cusanovich", root=root)

    matrix = _read_matrix(_require(data_dir / _MATRIX))
    cells = (
        _require(data_dir / _CELLS).read_text().split()
    )
    metadata = pd.read_csv(_require(data_dir / _METADATA), sep="\t")

    if label_column not in metadata.columns:
        raise ValueError(
            f"cell_metadata.txt has no column {label_column!r}. "
            f"Available: {sorted(metadata.columns)}."
        )

    metadata = metadata.set_index("cell").reindex(cells)
    n_full = matrix.shape[1]

    keep_cells = (
        metadata[label_column].notna().to_numpy()
        & (metadata[label_column].to_numpy() != UNKNOWN_LABEL)
    )

    n_cells = int(matrix.shape[1])
    peak_cell_counts = np.asarray((matrix > 0).sum(axis=1)).ravel()
    keep_peaks = peak_cell_counts >= site_frequency_threshold * n_cells
    if not keep_peaks.any():
        raise ValueError(
            f"site_frequency_threshold={site_frequency_threshold} removed every "
            "peak. Lower it."
        )

    counts = matrix[keep_peaks][:, keep_cells]
    embedding = TruncatedSVD(
        n_components=n_components, random_state=random_state
    ).fit_transform(_tfidf(counts).T.tocsr())

    X = np.asarray(embedding, dtype=np.float64)
    y = pd.Series(
        metadata.loc[keep_cells, label_column].to_numpy(),
        name=label_column,
    ).astype(str)

    n_annotated = int(X.shape[0])
    if subsample is not None:
        size = (
            int(subsample * n_annotated)
            if isinstance(subsample, float)
            else int(subsample)
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=size / n_annotated, random_state=random_state
        )
        _, idx = next(splitter.split(X, y))
        X = X[idx]
        y = y.iloc[idx].reset_index(drop=True)

    meta = {
        "source": "Cusanovich",
        "citation": "Cusanovich et al. 2018, Cell 174(5):1309-1324",
        "download": DOWNLOAD_ROOT,
        "n_cells_full": int(n_full),
        "n_cells_annotated": n_annotated,
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_peaks_full": int(matrix.shape[0]),
        "n_peaks_kept": int(keep_peaks.sum()),
        "label_name": label_column,
        "drops_first_component": False,
        "preprocessing": [
            f"drop cells labeled {UNKNOWN_LABEL!r} in {label_column}",
            f"keep peaks accessible in at least {site_frequency_threshold:.0%} "
            "of cells (site_frequency_threshold, source default 3%)",
            "TF-IDF: per-cell term frequency, "
            "IDF = log(1 + n_cells / cells_per_peak)",
            f"TruncatedSVD(n_components={n_components}); all components kept, "
            "matching the source, which does not drop component 1",
        ],
        "subsample": subsample,
        "random_state": random_state,
    }
    return X, y, meta
```

- [ ] Step 4: Export the loader

In `src/benchmarks/datasets/__init__.py`:

```python
from ._cusanovich import load_cusanovich
from ._klein import DATA_ROOT, load_klein, resolve_data_dir
from ._levine import load_levine32

__all__ = [
    "DATA_ROOT",
    "load_cusanovich",
    "load_klein",
    "load_levine32",
    "resolve_data_dir",
]
```

- [ ] Step 5: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/benchmarks/datasets/test_cusanovich.py -v`
Expected: PASS, 8 tests.

- [ ] Step 6: Verify the structure test can fail

Temporarily replace `embedding` with `np.random.default_rng(0).normal(size=(counts.shape[1], n_components))`. Rerun. Expected: `test_embedding_separates_the_planted_tissues` FAILS while every shape test still passes, which is the point of having it. Revert and confirm PASS.

- [ ] Step 7: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/benchmarks/datasets/ tests/benchmarks/datasets/test_cusanovich.py
git commit -m "feat(benchmarks): add the Cusanovich mouse sci-ATAC loader"
```

---

### Task 4: hECA pooled-organ loader

Files:
- Create: `src/benchmarks/datasets/_heca.py`
- Modify: `src/benchmarks/datasets/__init__.py`
- Test: `tests/benchmarks/datasets/test_heca.py`

Interfaces:
- Consumes: `resolve_data_dir` from `datasets/_klein.py`.
- Produces: `load_heca(*, root: Path | None = None, organs: Sequence[str] = DEFAULT_ORGANS, subsample: int | float | None = None, random_state: int = 42, open_fraction: float = 0.005, n_top_peaks: int = 50_000, n_components: int = 50, label_column: str = "organ", cache: bool = True) -> tuple[np.ndarray, pd.Series, dict]`, plus `peak_open_counts(path: Path, *, block: int) -> tuple[np.ndarray, int]` and `reduce_to_peaks(path: Path, peaks: np.ndarray, *, block: int) -> sparse.csr_matrix`.

Preprocessing follows the hECA v2.0 publication rather than the Cusanovich chain, because the two sources differ: cPeaks open in at least 0.5% of cells, then highly variable peak selection, then log-normalization and PCA. One recorded deviation: the source selects 500,000 highly variable cPeaks and we select 50,000, for memory. At 19 GB of RAM the reduced matrix must fit beside the PCA workspace.

The peak matrix is never fully resident. Two chunked passes over the backed CSR do the work: one accumulates per-peak open counts across all organs, and one builds the reduced cells-by-peaks matrix. Archives are extracted one at a time and the extracted h5ad is deleted after each pass, so peak disk stays near the largest single organ rather than all five at once.

Expected files under `data/hECA/`, downloaded from Zenodo record 15627886: `ATAC-Lung.h5ad`, `ATAC-Brain.h5ad`, `ATAC-Kidney.h5ad`, `ATAC-Heart.h5ad`, `ATAC-Thymus.h5ad`. Each is `X` as a CSR group of raw int32 counts, cells by cPeaks, with `obs` carrying `cell_type`, `organ`, `donor_id`, `study_id` and `seq_tech`, and an empty `obsm`.

- [ ] Step 1: Write the failing tests

Create `tests/benchmarks/datasets/test_heca.py`:

```python
"""hECA loader tests.

The real organs are gigabytes each, so the fixtures are small h5ad files in
the same layout: CSR raw counts, cells by cPeaks, annotations in obs, and an
empty obsm.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from benchmarks.datasets import load_heca
from benchmarks.datasets._heca import peak_open_counts, reduce_to_peaks

N_PEAKS = 300


def _organ(path, name, n_cells, seed):
    rng = np.random.default_rng(seed)
    types = rng.choice(["T cell", "Epithelial cell", "Unclassified"], n_cells)
    M = np.zeros((n_cells, N_PEAKS))
    # An organ-specific block plus a shared background gives the embedding
    # structure that clustering can recover.
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
                "study_id": "10.1000/x",
            },
            index=[f"{name}_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"peak{i}" for i in range(N_PEAKS)]),
    )
    adata.write_h5ad(path / f"ATAC-{name}.h5ad")


@pytest.fixture
def heca(tmp_path):
    root = tmp_path / "data"
    d = root / "hECA"
    d.mkdir(parents=True)
    _organ(d, "Lung", 90, 0)
    _organ(d, "Brain", 80, 1)
    return root


class TestChunkedPasses:
    def test_open_counts_match_a_dense_computation(self, heca):
        path = heca / "hECA" / "ATAC-Lung.h5ad"
        counts, n_cells = peak_open_counts(path, block=17)
        dense = ad.read_h5ad(path).X.toarray()
        assert n_cells == dense.shape[0]
        assert np.array_equal(counts, (dense > 0).sum(axis=0))

    def test_open_counts_are_block_size_invariant(self, heca):
        path = heca / "hECA" / "ATAC-Brain.h5ad"
        a, _ = peak_open_counts(path, block=7)
        b, _ = peak_open_counts(path, block=100_000)
        assert np.array_equal(a, b)

    def test_reduction_selects_the_requested_columns(self, heca):
        path = heca / "hECA" / "ATAC-Lung.h5ad"
        peaks = np.array([3, 10, 250])
        reduced = reduce_to_peaks(path, peaks, block=13)
        dense = ad.read_h5ad(path).X.toarray()
        assert reduced.shape == (dense.shape[0], peaks.size)
        assert np.array_equal(reduced.toarray(), dense[:, peaks])


class TestLoadHeca:
    def test_pools_the_requested_organs(self, heca):
        X, y, meta = load_heca(
            root=heca, organs=["Lung", "Brain"], n_top_peaks=50, n_components=5
        )
        assert X.shape[1] == 5
        assert set(y) == {"Lung", "Brain"}
        assert meta["organs"] == ["Lung", "Brain"]
        assert meta["n_cells"] == X.shape[0]

    def test_unclassified_cells_are_dropped(self, heca):
        _, _, meta = load_heca(
            root=heca,
            organs=["Lung"],
            n_top_peaks=50,
            n_components=5,
            label_column="cell_type",
        )
        assert meta["n_cells"] < meta["n_cells_full"]

    def test_cell_type_is_available_as_the_reference(self, heca):
        _, y, _ = load_heca(
            root=heca,
            organs=["Lung"],
            n_top_peaks=50,
            n_components=5,
            label_column="cell_type",
        )
        assert "Unclassified" not in set(y)
        assert y.name == "cell_type"

    def test_embedding_separates_the_planted_organs(self, heca):
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        X, y, _ = load_heca(
            root=heca, organs=["Lung", "Brain"], n_top_peaks=100, n_components=5
        )
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_features_are_shared_across_organs(self, heca):
        # Selecting features per organ would make the pooled embedding
        # meaningless, so the selection must be computed across all of them.
        _, _, meta = load_heca(
            root=heca, organs=["Lung", "Brain"], n_top_peaks=50, n_components=5
        )
        assert meta["n_peaks_selected"] == 50
        assert meta["feature_selection_scope"] == "pooled across organs"

    def test_subsample_is_stratified_and_sized(self, heca):
        X, y, meta = load_heca(
            root=heca,
            organs=["Lung", "Brain"],
            n_top_peaks=50,
            n_components=5,
            subsample=40,
        )
        assert X.shape[0] == 40
        assert y.nunique() == 2

    def test_meta_records_the_hvg_deviation(self, heca):
        _, _, meta = load_heca(
            root=heca, organs=["Lung"], n_top_peaks=50, n_components=5
        )
        chain = " ".join(meta["preprocessing"])
        assert "0.5%" in chain
        assert "log1p" in chain
        assert "PCA" in chain
        assert "deviation" in " ".join(meta["deviations"]).lower()

    def test_missing_organ_names_the_download(self, heca):
        with pytest.raises(FileNotFoundError, match="15627886"):
            load_heca(root=heca, organs=["Pancreas"], n_top_peaks=50)
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/benchmarks/datasets/test_heca.py -v`
Expected: FAIL, `ImportError: cannot import name 'load_heca'`.

- [ ] Step 3: Implement the loader

Create `src/benchmarks/datasets/_heca.py`:

```python
"""hECA v2.0 scATAC-seq, pooled across organs.

1,450,511 cells over 25 organs against a 1,657,194-entry cPeak reference,
with cell types harmonized by uHAF. This is the scalability case study, so
organs are pooled to exceed 500,000 cells and the reference label is the
organ, which is unambiguous. The harmonized cell type is available as a
secondary reference but is coarse, averaging roughly four types per organ.

Preprocessing follows the hECA publication rather than the Cusanovich chain:
cPeaks open in at least 0.5% of cells, highly variable peak selection, then
log-normalization and PCA. The peak matrix is never fully resident; two
chunked passes over the backed CSR do the work.
"""

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from ._klein import resolve_data_dir

ZENODO_RECORD = "15627886"
DOWNLOAD_ROOT = f"https://zenodo.org/records/{ZENODO_RECORD}"

#: Five organs clear 500,000 cells while keeping the organ label at five
#: levels. Lung and Brain supply most of the cells; the rest give the
#: reference label enough classes to be usable.
DEFAULT_ORGANS: tuple[str, ...] = ("Lung", "Brain", "Kidney", "Heart", "Thymus")

#: uHAF value marking cells left unannotated, dropped like Levine's
#: uncharacterized population 15.
UNCLASSIFIED_LABEL = "Unclassified"

DEFAULT_BLOCK = 20_000


def _organ_path(data_dir: Path, organ: str) -> Path:
    path = data_dir / f"ATAC-{organ}.h5ad"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download ATAC-{organ}.h5ad.zip from Zenodo "
            f"record {ZENODO_RECORD} ({DOWNLOAD_ROOT}) and extract it into "
            "data/hECA/. Extract one organ at a time and delete the h5ad "
            "after use; all five extracted at once will not fit."
        )
    return path


def _csr_block(handle: h5py.File, lo: int, hi: int, n_peaks: int) -> sparse.csr_matrix:
    """Read rows [lo, hi) of an h5ad CSR X group without loading the rest."""
    indptr = handle["X"]["indptr"]
    start = int(indptr[lo])
    stop = int(indptr[hi])
    data = handle["X"]["data"][start:stop]
    indices = handle["X"]["indices"][start:stop]
    local_indptr = indptr[lo : hi + 1][:] - start
    return sparse.csr_matrix(
        (data, indices, local_indptr), shape=(hi - lo, n_peaks)
    )


def _shape(handle: h5py.File) -> tuple[int, int]:
    shape = handle["X"].attrs["shape"]
    return int(shape[0]), int(shape[1])


def peak_open_counts(path: Path, *, block: int = DEFAULT_BLOCK) -> tuple[np.ndarray, int]:
    """Number of cells each peak is open in, accumulated over row blocks.

    Returns (counts of length n_peaks, n_cells).
    """
    with h5py.File(path, "r") as handle:
        n_cells, n_peaks = _shape(handle)
        counts = np.zeros(n_peaks, dtype=np.int64)
        for lo in range(0, n_cells, block):
            hi = min(lo + block, n_cells)
            chunk = _csr_block(handle, lo, hi, n_peaks)
            chunk.data = np.ones_like(chunk.data)
            counts += np.asarray(chunk.sum(axis=0)).ravel().astype(np.int64)
    return counts, n_cells


def reduce_to_peaks(
    path: Path, peaks: np.ndarray, *, block: int = DEFAULT_BLOCK
) -> sparse.csr_matrix:
    """Build the cells-by-selected-peaks matrix, one row block at a time."""
    peaks = np.asarray(peaks)
    with h5py.File(path, "r") as handle:
        n_cells, n_peaks = _shape(handle)
        blocks = []
        for lo in range(0, n_cells, block):
            hi = min(lo + block, n_cells)
            blocks.append(_csr_block(handle, lo, hi, n_peaks)[:, peaks])
    return sparse.vstack(blocks, format="csr")


def _read_obs(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read only the requested obs columns, never touching X."""
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    frame = adata.obs[list(columns)].copy()
    adata.file.close()
    return frame.reset_index(drop=True)


def load_heca(
    *,
    root: Path | None = None,
    organs: Sequence[str] = DEFAULT_ORGANS,
    subsample: int | float | None = None,
    random_state: int = 42,
    open_fraction: float = 0.005,
    n_top_peaks: int = 50_000,
    n_components: int = 50,
    label_column: str = "organ",
    block: int = DEFAULT_BLOCK,
) -> tuple[np.ndarray, pd.Series, dict]:
    """Load and preprocess pooled hECA v2.0 ATAC organs.

    Parameters
    ----------
    root : Path or None
        Directory holding the dataset folders. Defaults to data/.
    organs : sequence of str
        Organ names to pool. Files are read as data/hECA/ATAC-<organ>.h5ad.
    subsample : int, float, or None
        Stratified subsample size, as a count or a fraction. None keeps all.
    random_state : int
        Seed for the subsample and the PCA.
    open_fraction : float
        Keep cPeaks open in at least this fraction of pooled cells. 0.005 is
        the source publication's value.
    n_top_peaks : int
        Highly variable cPeaks retained. The source uses 500,000; see the
        recorded deviation.
    n_components : int
        PCA components retained.
    label_column : str
        Reference label column. "organ" is unambiguous; "cell_type" is the
        uHAF-harmonized annotation and is coarse.
    block : int
        Rows read per chunk in both passes.

    Returns
    -------
    (X, y, meta)
    """
    import scanpy as sc
    from anndata import AnnData
    from sklearn.model_selection import StratifiedShuffleSplit

    data_dir = resolve_data_dir("hECA", root=root)
    paths = [_organ_path(data_dir, organ) for organ in organs]

    # Pass 1: per-peak open counts pooled across every organ, so the feature
    # set is shared. Selecting features per organ would make the pooled
    # embedding meaningless.
    total_counts: np.ndarray | None = None
    total_cells = 0
    for path in paths:
        counts, n_cells = peak_open_counts(path, block=block)
        total_counts = counts if total_counts is None else total_counts + counts
        total_cells += n_cells

    keep = np.flatnonzero(total_counts >= open_fraction * total_cells)
    if keep.size == 0:
        raise ValueError(
            f"open_fraction={open_fraction} removed every cPeak. Lower it."
        )

    # Pass 2: build the reduced matrix, one organ and one row block at a time.
    matrices = [reduce_to_peaks(path, keep, block=block) for path in paths]
    counts_matrix = sparse.vstack(matrices, format="csr")
    del matrices

    obs = pd.concat(
        [_read_obs(path, ["cell_type", "organ", "donor_id", "study_id"])
         for path in paths],
        ignore_index=True,
    )

    if label_column not in obs.columns:
        raise ValueError(
            f"obs has no column {label_column!r}. Available: {sorted(obs.columns)}."
        )

    adata = AnnData(X=counts_matrix, obs=obs)
    n_full = int(adata.n_obs)

    annotated = (
        adata.obs[label_column].notna().to_numpy()
        & (adata.obs[label_column].to_numpy() != UNCLASSIFIED_LABEL)
    )
    adata = adata[annotated].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_top = min(int(n_top_peaks), int(adata.n_vars))
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.tl.pca(adata, n_comps=min(int(n_components), adata.n_vars - 1))

    X = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    y = pd.Series(adata.obs[label_column].to_numpy(), name=label_column).astype(str)

    n_annotated = int(X.shape[0])
    if subsample is not None:
        size = (
            int(subsample * n_annotated)
            if isinstance(subsample, float)
            else int(subsample)
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=size / n_annotated, random_state=random_state
        )
        _, idx = next(splitter.split(X, y))
        X = X[idx]
        y = y.iloc[idx].reset_index(drop=True)

    meta = {
        "source": "hECA v2.0 ATAC",
        "citation": "Chen et al. 2025, Scientific Data",
        "download": DOWNLOAD_ROOT,
        "organs": list(organs),
        "n_cells_full": n_full,
        "n_cells_annotated": n_annotated,
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_peaks_full": int(total_counts.size),
        "n_peaks_open": int(keep.size),
        "n_peaks_selected": int(n_top),
        "feature_selection_scope": "pooled across organs",
        "label_name": label_column,
        "preprocessing": [
            f"pool organs {list(organs)}",
            f"drop cells labeled {UNCLASSIFIED_LABEL!r} in {label_column}",
            f"keep cPeaks open in at least {open_fraction:.1%} of pooled cells "
            "(0.5%, the source value)",
            "normalize_total(target_sum=1e4) then log1p",
            f"highly_variable_genes(n_top_genes={n_top}, flavor='seurat')",
            f"PCA to {X.shape[1]} components",
        ],
        "deviations": [
            "deviation from source: the publication selects 500,000 highly "
            f"variable cPeaks; {n_top} are selected here, for memory",
        ],
        "batch_note": (
            "cells pool multiple study_id values, so a pooled clustering may "
            "partly recover study of origin rather than biology"
        ),
        "subsample": subsample,
        "random_state": random_state,
    }
    return X, y, meta
```

- [ ] Step 4: Export the loader

In `src/benchmarks/datasets/__init__.py`, add `from ._heca import load_heca` and add `"load_heca"` to `__all__`, keeping the list alphabetical.

- [ ] Step 5: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/benchmarks/datasets/test_heca.py -v`
Expected: PASS, 11 tests.

- [ ] Step 6: Verify the chunked pass can fail

Temporarily change `_csr_block` to use `local_indptr = indptr[lo : hi + 1][:]` without subtracting `start`. Rerun. Expected: `test_open_counts_match_a_dense_computation` and `test_reduction_selects_the_requested_columns` FAIL. This is the single easiest mistake to make in a chunked CSR reader and the tests must catch it. Revert and confirm PASS.

- [ ] Step 7: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/benchmarks/datasets/ tests/benchmarks/datasets/test_heca.py
git commit -m "feat(benchmarks): add the hECA pooled-organ ATAC loader"
```

---

### Task 5: Register both studies

Files:
- Modify: `src/benchmarks/_studies.py` (`STUDIES` and `study_model_grids`)
- Test: `tests/benchmarks/test_studies.py`

Interfaces:
- Consumes: `load_cusanovich` and `load_heca` from Tasks 3 and 4, `resolution_grids` and `RESOLUTION_ESTIMATORS` from Task 1, the extended `Study` from Task 2.
- Produces: `STUDIES["cusanovich"]` and `STUDIES["heca"]`, and `study_resolution_grids(study: Study) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]`.

Candidate-k ranges are centered on what the data is expected to contain and stop well short of each dataset's finest reported partition. Cusanovich reports 13 tissues, 30 clusters and 40 cell labels; sweeping toward 40 would presuppose that the finest reported partition is correct, when CARVE's claim is that it is not the most stable. hECA's organ and major-lineage level is the target.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_studies.py`:

```python
from benchmarks._studies import study_model_grids, study_resolution_grids


class TestNewStudies:
    def test_both_new_studies_are_registered(self):
        assert {"cusanovich", "heca"} <= set(STUDIES)

    def test_cusanovich_sweeps_four_through_sixteen(self):
        # Centered near the 13 tissues; stops well short of the 30 clusters
        # and 40 cell labels the source reports.
        assert STUDIES["cusanovich"].candidate_k == tuple(range(4, 17))

    def test_heca_sweeps_three_through_fifteen(self):
        assert STUDIES["heca"].candidate_k == tuple(range(3, 16))

    def test_cusanovich_pairs_kmeans_with_spectral(self):
        grids = study_model_grids(STUDIES["cusanovich"])
        from carve.cluster import SpectralClustering
        from sklearn.cluster import KMeans

        assert {cls for cls, _ in grids} == {KMeans, SpectralClustering}

    def test_heca_uses_estimators_that_can_run_at_scale(self):
        # Spectral builds a dense n-by-n affinity and Ward is quadratic in
        # memory, so neither may appear in the large-scale study.
        from carve.cluster import SpectralClustering
        from sklearn.cluster import AgglomerativeClustering

        classes = {cls for cls, _ in study_model_grids(STUDIES["heca"])}
        assert SpectralClustering not in classes
        assert AgglomerativeClustering not in classes

    def test_heca_declares_a_resolution_sweep(self):
        from carve.cluster import LeidenClustering

        grids = study_resolution_grids(STUDIES["heca"])
        assert len(grids) == 1
        cls, grid = grids[0]
        assert cls is LeidenClustering
        assert grid["resolution"][0] == pytest.approx(0.1)
        assert grid["resolution"][-1] == pytest.approx(2.0)
        assert len(grid["resolution"]) == 20

    def test_a_study_without_resolutions_raises(self):
        with pytest.raises(ValueError, match="declares no resolutions"):
            study_resolution_grids(STUDIES["klein"])

    def test_cusanovich_also_declares_resolutions_for_its_atlas_pass(self):
        # At 81,173 cells neither spectral nor Ward can run, so the
        # full-atlas pass sweeps Leiden resolution instead of k.
        assert len(STUDIES["cusanovich"].resolutions) == 20
        assert study_resolution_grids(STUDIES["cusanovich"])

    def test_heca_pins_its_anchor_count(self):
        # 20 configurations at m=5000 would retain about 4 GB of blocks; at
        # m=2000 it is about 0.83 GB.
        assert STUDIES["heca"].consensus_anchors == 2000

    def test_cusanovich_leaves_anchors_at_the_package_default(self):
        assert STUDIES["cusanovich"].consensus_anchors is None

    def test_new_studies_declare_dev_and_publication_scales(self):
        for name in ("cusanovich", "heca"):
            assert {"dev", "publication"} <= set(STUDIES[name].scales)

    def test_cusanovich_publication_scale_matches_levine(self):
        assert STUDIES["cusanovich"].scales["publication"] == 5000
        assert STUDIES["cusanovich"].scales["atlas"] is None
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/benchmarks/test_studies.py::TestNewStudies -v`
Expected: FAIL, `KeyError: 'cusanovich'`.

- [ ] Step 3: Implement

In `src/benchmarks/_studies.py`, add the two loaders and register the studies:

```python
def _cusanovich_loader(subsample: int | float | None):
    from .datasets import load_cusanovich

    # tissue is the reference: it is determined by dissection, not by any
    # clustering, which is what makes it independent ground truth.
    return load_cusanovich(
        subsample=subsample, random_state=42, label_column="tissue"
    )


def _heca_loader(subsample: int | float | None):
    from .datasets import load_heca

    return load_heca(subsample=subsample, random_state=42, label_column="organ")
```

and add to `STUDIES`:

```python
    "cusanovich": Study(
        name="cusanovich",
        loader=_cusanovich_loader,
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=tuple(range(4, 17)),
        scales={"dev": 1500, "publication": 5000, "atlas": None},
        default_scale="dev",
        # The atlas scale runs all 81,173 cells, where spectral and Ward
        # cannot run, so that pass sweeps Leiden resolution instead. This is
        # what shows the case-study conclusion survives past the subsample.
        resolutions=tuple(round(0.1 * i, 1) for i in range(1, 21)),
    ),
    "heca": Study(
        name="heca",
        loader=_heca_loader,
        estimator=EstimatorSpec(name="minibatch_kmeans"),
        candidate_k=tuple(range(3, 16)),
        scales={"dev": 25_000, "publication": None},
        default_scale="dev",
        resolutions=tuple(round(0.1 * i, 1) for i in range(1, 21)),
        consensus_anchors=2000,
    ),
```

Replace `study_model_grids` so it honors resolution-based studies and pairs the right second estimator:

```python
def study_model_grids(
    study: Study,
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """The study's own estimator plus a second one, over candidate_k.

    Klein sweeps Ward agglomerative and spectral; Levine sweeps KMeans and
    spectral; Cusanovich sweeps KMeans and spectral at case-study scale. The
    hECA study cannot use spectral, which builds a dense n-by-n affinity, so
    it pairs MiniBatchKMeans with KMeans instead.
    """
    if study.estimator.name in RESOLUTION_ESTIMATORS:
        raise ValueError(
            f"Study {study.name!r} has a resolution-based estimator; use "
            "study_resolution_grids."
        )

    partner = (
        EstimatorSpec(name="kmeans")
        if study.name == "heca"
        else EstimatorSpec(name="spectral")
    )
    return param_grids(study.estimator, study.candidate_k) + param_grids(
        partner, study.candidate_k
    )


def study_resolution_grids(
    study: Study,
) -> list[tuple[type[ClusterMixin], dict[str, list[Any]]]]:
    """The study's Leiden resolution sweep.

    A separate CARVE run from study_model_grids: SweepSpec is frozen, so a
    k-based and a resolution-based sweep cannot share one run.
    """
    if not study.resolutions:
        raise ValueError(f"Study {study.name!r} declares no resolutions.")
    return resolution_grids(EstimatorSpec(name="leiden"), study.resolutions)
```

Add `RESOLUTION_ESTIMATORS` and `resolution_grids` to the existing `from ._estimators import ...` line.

- [ ] Step 4: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/benchmarks/test_studies.py -v`
Expected: PASS. `TestStudies::test_both_case_studies_are_registered` asserts `set(STUDIES) == {"klein", "levine32"}` and must be widened to the four names; widen it rather than deleting it.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/benchmarks/_studies.py tests/benchmarks/test_studies.py
git commit -m "feat(benchmarks): register the Cusanovich and hECA case studies"
```

---

### Task 6: The scaling sweep

Files:
- Modify: `src/benchmarks/_studies.py`
- Test: `tests/benchmarks/test_studies.py`

Interfaces:
- Consumes: `peak_rss_bytes` from `benchmarks._artifacts`, `CARVE` with `consensus_anchors` from the anchored-consensus plan.
- Produces: `study_scaling_sweep(X, y, *, sizes: Sequence[int], model_grids, n_resamples: int = 100, n_jobs: int = 1, random_state: int = 42, consensus_anchors: int | None = None, measure: str = "stability", rule: str = "1se") -> pd.DataFrame` with columns `n`, `n_configs`, `wall_clock_s`, `peak_rss_bytes`, `selected_k`, `ari`.

This answers the reviewer request for runtime, memory and complexity on datasets of increasing size. Nested subsampling is controlled: the biology is held constant and n is isolated.

- [ ] Step 1: Write the failing tests

Append to `tests/benchmarks/test_studies.py`:

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from benchmarks._studies import study_scaling_sweep


@pytest.fixture
def ladder_data():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [4.0, 7.0]])
    labels = rng.integers(0, 3, 400)
    X = centers[labels] + rng.normal(0, 1.0, (400, 2))
    return X, pd.Series(labels.astype(str), name="truth")


class TestStudyScalingSweep:
    def _grids(self):
        return [(KMeans, {"n_clusters": [2, 3, 4], "n_init": [10]})]

    def test_one_row_per_size(self, ladder_data):
        X, y = ladder_data
        out = study_scaling_sweep(
            X, y, sizes=[100, 200], model_grids=self._grids(), n_resamples=5
        )
        assert list(out["n"]) == [100, 200]
        assert len(out) == 2

    def test_columns_are_the_documented_contract(self, ladder_data):
        X, y = ladder_data
        out = study_scaling_sweep(
            X, y, sizes=[100], model_grids=self._grids(), n_resamples=5
        )
        assert list(out.columns) == [
            "n",
            "n_configs",
            "wall_clock_s",
            "peak_rss_bytes",
            "selected_k",
            "ari",
        ]

    def test_measurements_are_populated(self, ladder_data):
        X, y = ladder_data
        out = study_scaling_sweep(
            X, y, sizes=[100, 200], model_grids=self._grids(), n_resamples=5
        )
        assert (out["wall_clock_s"] > 0).all()
        assert (out["peak_rss_bytes"] > 0).all()
        assert out["n_configs"].nunique() == 1
        assert out["selected_k"].between(2, 4).all()

    def test_ari_reflects_the_planted_structure(self, ladder_data):
        # A sweep that scored against shuffled labels would still produce a
        # populated frame, so pin that the ARI is meaningful.
        X, y = ladder_data
        out = study_scaling_sweep(
            X, y, sizes=[200], model_grids=self._grids(), n_resamples=8
        )
        assert out["ari"].iloc[0] > 0.5

    def test_sizes_larger_than_n_are_rejected(self, ladder_data):
        X, y = ladder_data
        with pytest.raises(ValueError, match="exceeds"):
            study_scaling_sweep(
                X, y, sizes=[10_000], model_grids=self._grids(), n_resamples=5
            )

    def test_is_deterministic(self, ladder_data):
        X, y = ladder_data
        a = study_scaling_sweep(
            X, y, sizes=[150], model_grids=self._grids(), n_resamples=5
        )
        b = study_scaling_sweep(
            X, y, sizes=[150], model_grids=self._grids(), n_resamples=5
        )
        assert a["selected_k"].equals(b["selected_k"])
        assert np.allclose(a["ari"], b["ari"])
```

- [ ] Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/benchmarks/test_studies.py::TestStudyScalingSweep -v`
Expected: FAIL, `ImportError: cannot import name 'study_scaling_sweep'`.

- [ ] Step 3: Implement

Append to `src/benchmarks/_studies.py`:

```python
def study_scaling_sweep(
    X: np.ndarray,
    y: np.ndarray | pd.Series | None,
    *,
    sizes: Sequence[int],
    model_grids: list[tuple[type[ClusterMixin], dict[str, list[Any]]]],
    n_resamples: int = 100,
    n_jobs: int = 1,
    random_state: int = 42,
    consensus_anchors: int | None = None,
    measure: str = "stability",
    rule: str = "1se",
) -> pd.DataFrame:
    """Fit CARVE at a ladder of subsample sizes and measure the cost.

    Nested subsampling holds the biology constant so that n is the only
    thing varying, which per-dataset comparisons cannot do.

    peak_rss_bytes is the process high-water mark at the end of each fit, not
    a per-fit delta: ru_maxrss only ever rises. Because the ladder grows
    monotonically in n, the reported value is still the peak attributable to
    that size, but it must not be read as the memory a single fit would need
    in a fresh process.

    Returns
    -------
    DataFrame with columns (n, n_configs, wall_clock_s, peak_rss_bytes,
    selected_k, ari). ari is against y, or NaN when y is None.
    """
    import time

    from sklearn.metrics import adjusted_rand_score

    from ._artifacts import peak_rss_bytes

    X = np.asarray(X)
    y_arr = None if y is None else np.asarray(y)
    n_total = X.shape[0]

    rows: list[dict[str, Any]] = []
    for size in sizes:
        size = int(size)
        if size > n_total:
            raise ValueError(
                f"Requested size {size} exceeds the {n_total} available samples."
            )

        rng = np.random.default_rng(random_state + size)
        idx = np.sort(rng.choice(n_total, size=size, replace=False))

        carve = CARVE(
            estimator_param_grids=model_grids,
            n_resamples=n_resamples,
            n_jobs=n_jobs,
            random_state=random_state,
            **({} if consensus_anchors is None else
               {"consensus_anchors": consensus_anchors}),
        )

        started = time.perf_counter()
        carve.fit(X[idx])
        elapsed = time.perf_counter() - started

        labels = carve.get_labels(measure=measure, rule=rule)
        rows.append(
            {
                "n": size,
                "n_configs": int(carve.estimator_results_.shape[0]),
                "wall_clock_s": float(elapsed),
                "peak_rss_bytes": int(peak_rss_bytes()),
                "selected_k": int(carve.get_k(measure=measure, rule=rule)),
                "ari": (
                    float("nan")
                    if y_arr is None
                    else float(adjusted_rand_score(y_arr[idx], labels))
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "n",
            "n_configs",
            "wall_clock_s",
            "peak_rss_bytes",
            "selected_k",
            "ari",
        ],
    )
```

Add `from collections.abc import Sequence` to the imports if it is not already present.

- [ ] Step 4: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/benchmarks/test_studies.py::TestStudyScalingSweep -v`
Expected: PASS, 6 tests.

- [ ] Step 5: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/benchmarks/_studies.py tests/benchmarks/test_studies.py
git commit -m "feat(benchmarks): measure runtime and memory across a size ladder"
```

---

### Task 7: The Cusanovich results figure

Files:
- Create: `src/benchmarks/figures/_cusanovich_results.py`
- Modify: `src/benchmarks/figures/__init__.py`
- Test: `tests/benchmarks/figures/test_cusanovich_results.py`

Interfaces:
- Consumes: `composite_figure`, `composite_color_maps` and `CompositeInputs` from `figures/_case_study.py`, `alluvial` from `_panels`.
- Produces: `figure_cusanovich_results(inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None) -> Figure`.

Same shape as the Klein figure: an alluvial bottom panel linking the CARVE clustering, the reported labels and the CVI clustering. The embedding is the first two LSI components rather than PCA, so the axis labels differ.

- [ ] Step 1: Write the failing test

Create `tests/benchmarks/figures/test_cusanovich_results.py`:

```python
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from benchmarks.figures import figure_cusanovich_results
from benchmarks.figures._cusanovich_results import AXIS_LABELS, MARKER_SIZE


@pytest.fixture
def inputs():
    from benchmarks.figures._case_study import CompositeInputs

    rng = np.random.default_rng(0)
    n = 90
    y = np.repeat(["Lung", "Liver", "Spleen"], n // 3)
    Z = rng.normal(size=(n, 2))
    curves = pd.DataFrame(
        {
            "metric": np.repeat(["silhouette", "gap"], 6),
            "model": "KMeans",
            "k": list(range(4, 10)) * 2,
            "score": rng.random(12),
            "ari": rng.random(12),
        }
    )
    best = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["KMeans", "KMeans"],
            "k": [4, 5],
            "score": [0.4, 0.3],
            "ari": [0.5, 0.4],
        }
    )

    class _Carve:
        def plot_metric_over_n_clusters(self, ax=None, **kw):
            ax.plot([4, 5, 6], [0.1, 0.2, 0.3], label="stability")
            return ax

    return CompositeInputs(
        X=rng.normal(size=(n, 5)),
        y=y,
        Z=Z,
        carve=_Carve(),
        carve_labels=rng.integers(0, 3, n),
        comparison_labels=rng.integers(0, 3, n),
        comparison_name="Silhouette",
        comparison_k=4,
        curves_df=curves,
        best_df=best,
    )


def test_axis_labels_name_the_lsi_components():
    # The embedding is LSI, not PCA; mislabeling it would misreport the
    # preprocessing the manuscript describes.
    assert AXIS_LABELS == ("LSI 1", "LSI 2")


def test_marker_size_matches_the_levine_scale():
    # Thousands of cells, like Levine, not Klein's 1,358.
    assert MARKER_SIZE == 8.0


def test_figure_has_six_panels_and_does_not_write_when_save_is_false(inputs, tmp_path):
    fig = figure_cusanovich_results(inputs, save=False, out_dir=tmp_path)
    drawn = [ax for ax in fig.axes if ax.has_data() or ax.get_title()]
    assert len(drawn) >= 6
    assert list(tmp_path.iterdir()) == []


def test_figure_writes_under_its_manuscript_name(inputs, tmp_path):
    figure_cusanovich_results(inputs, save=True, out_dir=tmp_path)
    assert (tmp_path / "cusanovich_results.png").is_file()
```

- [ ] Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest tests/benchmarks/figures/test_cusanovich_results.py -v`
Expected: FAIL, `ImportError: cannot import name 'figure_cusanovich_results'`.

- [ ] Step 3: Implement

Create `src/benchmarks/figures/_cusanovich_results.py`:

```python
"""The Cusanovich mouse sci-ATAC case-study composite.

Panel F is an alluvial linking the CARVE clustering, the reported labels and
the CVI clustering, the same shape the Klein figure uses. Panels A to C use
the first two LSI components, which is the embedding the source publication's
own pipeline produces, and 8-point markers because the study is Levine sized
rather than Klein sized.
"""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._panels import alluvial
from ._case_study import CompositeInputs, composite_color_maps, composite_figure

MARKER_SIZE = 8.0
AXIS_LABELS = ("LSI 1", "LSI 2")


def _alluvial_panel(ax: Axes, inputs: CompositeInputs) -> Axes:
    # The same three maps the scatter panels use, so a cluster keeps one
    # color down the whole figure.
    true_cmap, carve_cmap, comparison_cmap = composite_color_maps(inputs)
    return alluvial(
        ax,
        inputs.y,
        inputs.carve_labels,
        inputs.comparison_labels,
        left_cmap=carve_cmap,
        right_cmap=comparison_cmap,
        true_cmap=true_cmap,
        left_title="CARVE",
        right_title=f"CVI ({inputs.comparison_name})",
        true_title="Reported Tissue",
    )


def figure_cusanovich_results(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build the Cusanovich case-study figure."""
    return composite_figure(
        inputs,
        bottom_panel=_alluvial_panel,
        marker_size=MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="cusanovich_results.png",
        save=save,
        out_dir=out_dir,
    )
```

- [ ] Step 4: Export it

In `src/benchmarks/figures/__init__.py`, add `from ._cusanovich_results import figure_cusanovich_results` and add the name to `__all__`.

- [ ] Step 5: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/benchmarks/figures/test_cusanovich_results.py -v`
Expected: PASS, 4 tests.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/benchmarks/figures/ tests/benchmarks/figures/test_cusanovich_results.py
git commit -m "feat(benchmarks): add the Cusanovich case-study figure"
```

---

### Task 8: The hECA results figure

Files:
- Create: `src/benchmarks/figures/_heca_results.py`
- Modify: `src/benchmarks/figures/__init__.py`
- Test: `tests/benchmarks/figures/test_heca_results.py`

Interfaces:
- Consumes: `composite_figure` and `CompositeInputs` from `figures/_case_study.py`, `ari_lollipop` from `_panels`.
- Produces: `figure_heca_results(inputs: CompositeInputs, *, scatter_subsample: int = 50_000, save: bool = True, out_dir: Path | None = None) -> Figure`, and `subsample_inputs(inputs: CompositeInputs, *, size: int, random_state: int = 42) -> CompositeInputs`.

Two deviations from the existing composites, both forced by the data. Panels A to C are drawn on a declared fixed subsample, because 500,000 points do not rasterize legibly. Panel F is the ARI comparison used by the Levine figure rather than an alluvial, which is unreadable at roughly 20 cell types.

- [ ] Step 1: Write the failing test

Create `tests/benchmarks/figures/test_heca_results.py`:

```python
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from benchmarks.figures import figure_heca_results
from benchmarks.figures._heca_results import AXIS_LABELS, subsample_inputs


@pytest.fixture
def inputs():
    from benchmarks.figures._case_study import CompositeInputs

    rng = np.random.default_rng(0)
    n = 600
    y = rng.choice(["Lung", "Brain", "Kidney"], n)
    curves = pd.DataFrame(
        {
            "metric": np.repeat(["silhouette", "gap"], 5),
            "model": "MiniBatchKMeans",
            "k": list(range(3, 8)) * 2,
            "score": rng.random(10),
            "ari": rng.random(10),
        }
    )
    best = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["MiniBatchKMeans"] * 2,
            "k": [3, 5],
            "score": [0.4, 0.3],
            "ari": [0.5, 0.4],
        }
    )

    class _Carve:
        def plot_metric_over_n_clusters(self, ax=None, **kw):
            ax.plot([3, 4, 5], [0.1, 0.2, 0.3], label="stability")
            return ax

    return CompositeInputs(
        X=rng.normal(size=(n, 5)),
        y=y,
        Z=rng.normal(size=(n, 2)),
        carve=_Carve(),
        carve_labels=rng.integers(0, 3, n),
        comparison_labels=rng.integers(0, 3, n),
        comparison_name="Silhouette",
        comparison_k=3,
        curves_df=curves,
        best_df=best,
    )


class TestSubsampleInputs:
    def test_reduces_every_per_cell_array_together(self, inputs):
        out = subsample_inputs(inputs, size=100)
        assert out.Z.shape[0] == 100
        assert out.y.shape[0] == 100
        assert out.carve_labels.shape[0] == 100
        assert out.comparison_labels.shape[0] == 100
        assert out.X.shape[0] == 100

    def test_rows_stay_aligned_across_arrays(self, inputs):
        # Independently subsampling each array would silently decouple a
        # cell's label from its position, which no shape assertion catches.
        out = subsample_inputs(inputs, size=50, random_state=1)
        again = subsample_inputs(inputs, size=50, random_state=1)
        assert np.array_equal(out.y, again.y)
        assert np.array_equal(out.carve_labels, again.carve_labels)
        pairs = {
            (str(a), int(b)) for a, b in zip(inputs.y, inputs.carve_labels)
        }
        for a, b in zip(out.y, out.carve_labels):
            assert (str(a), int(b)) in pairs

    def test_a_size_at_or_above_n_is_a_passthrough(self, inputs):
        out = subsample_inputs(inputs, size=10_000)
        assert out.y.shape[0] == inputs.y.shape[0]

    def test_non_per_cell_fields_are_preserved(self, inputs):
        out = subsample_inputs(inputs, size=100)
        assert out.comparison_k == inputs.comparison_k
        assert out.comparison_name == inputs.comparison_name
        assert out.curves_df.equals(inputs.curves_df)


def test_axis_labels_name_the_pca_components():
    # hECA follows the source's log-normalization and PCA chain, not the
    # TF-IDF and LSI chain the Cusanovich study uses.
    assert AXIS_LABELS == ("PC1", "PC2")


def test_figure_writes_under_its_manuscript_name(inputs, tmp_path):
    figure_heca_results(inputs, scatter_subsample=200, save=True, out_dir=tmp_path)
    assert (tmp_path / "heca_results.png").is_file()


def test_figure_does_not_write_when_save_is_false(inputs, tmp_path):
    fig = figure_heca_results(inputs, scatter_subsample=200, save=False,
                              out_dir=tmp_path)
    assert len([ax for ax in fig.axes if ax.has_data() or ax.get_title()]) >= 6
    assert list(tmp_path.iterdir()) == []
```

- [ ] Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest tests/benchmarks/figures/test_heca_results.py -v`
Expected: FAIL, `ImportError: cannot import name 'figure_heca_results'`.

- [ ] Step 3: Implement

Create `src/benchmarks/figures/_heca_results.py`:

```python
"""The hECA large-scale case-study composite.

Two deviations from the other case-study figures, both forced by the data.
The scatter panels are drawn on a declared fixed subsample, because half a
million points do not rasterize legibly and an undeclared crop would be worse
than a declared sample. Panel F is the ARI comparison the Levine figure uses
rather than an alluvial, which is unreadable at roughly 20 cell types.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import adjusted_rand_score

from .._panels import ari_lollipop
from ._case_study import CompositeInputs, composite_figure

MARKER_SIZE = 3.0
AXIS_LABELS = ("PC1", "PC2")
DEFAULT_SCATTER_SUBSAMPLE = 50_000


def subsample_inputs(
    inputs: CompositeInputs, *, size: int, random_state: int = 42
) -> CompositeInputs:
    """Reduce every per-cell array with one shared index.

    One index for all of them, so a cell keeps its label, its cluster and its
    embedding position together. Subsampling each array separately would
    decouple them and no shape check would notice.
    """
    n = np.asarray(inputs.y).shape[0]
    if size >= n:
        return inputs

    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(n, size=int(size), replace=False))

    return replace(
        inputs,
        X=np.asarray(inputs.X)[idx],
        y=np.asarray(inputs.y)[idx],
        Z=np.asarray(inputs.Z)[idx],
        carve_labels=np.asarray(inputs.carve_labels)[idx],
        comparison_labels=np.asarray(inputs.comparison_labels)[idx],
    )


def _ari_table(inputs: CompositeInputs) -> pd.DataFrame:
    """ARI of each selection against the reported labels."""
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
    return pd.DataFrame(rows)


def _ari_panel(ax: Axes, inputs: CompositeInputs) -> Axes:
    return ari_lollipop(
        ax, _ari_table(inputs), title="Agreement with Reported Labels (ARI)"
    )


def figure_heca_results(
    inputs: CompositeInputs,
    *,
    scatter_subsample: int = DEFAULT_SCATTER_SUBSAMPLE,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Build the hECA case-study figure."""
    return composite_figure(
        subsample_inputs(inputs, size=scatter_subsample),
        bottom_panel=_ari_panel,
        marker_size=MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="heca_results.png",
        save=save,
        out_dir=out_dir,
    )
```

- [ ] Step 4: Export it

In `src/benchmarks/figures/__init__.py`, add `from ._heca_results import figure_heca_results` and add the name to `__all__`.

- [ ] Step 5: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/benchmarks/figures/test_heca_results.py -v`
Expected: PASS, 7 tests.

- [ ] Step 6: Verify the alignment test can fail

Temporarily give each array its own index by drawing a fresh `rng.choice` per field inside `subsample_inputs`. Rerun. Expected: `test_rows_stay_aligned_across_arrays` FAILS. Revert and confirm PASS.

- [ ] Step 7: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/benchmarks/figures/ tests/benchmarks/figures/test_heca_results.py
git commit -m "feat(benchmarks): add the hECA case-study figure"
```

---

### Task 9: The study scaling figure

Files:
- Create: `src/benchmarks/figures/_study_scaling.py`
- Modify: `src/benchmarks/figures/__init__.py`
- Test: `tests/benchmarks/figures/test_study_scaling.py`

Interfaces:
- Consumes: the DataFrame produced by `study_scaling_sweep` in Task 6.
- Produces: `figure_study_scaling(sweep_df: pd.DataFrame, *, save: bool = True, out_dir: Path | None = None, save_name: str = "heca_scaling.png") -> Figure`.

Three panels against n: wall-clock, peak resident memory in gigabytes, and selected k. The first two answer the reviewer request for runtime and memory; the third shows whether the selection is stable as n grows.

- [ ] Step 1: Write the failing test

Create `tests/benchmarks/figures/test_study_scaling.py`:

```python
import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from benchmarks.figures import figure_study_scaling


@pytest.fixture
def sweep():
    return pd.DataFrame(
        {
            "n": [1000, 5000, 25_000],
            "n_configs": [26, 26, 26],
            "wall_clock_s": [12.0, 61.0, 320.0],
            "peak_rss_bytes": [int(1e9), int(3e9), int(9e9)],
            "selected_k": [5, 5, 6],
            "ari": [0.7, 0.72, 0.71],
        }
    )


def test_draws_three_panels(sweep, tmp_path):
    fig = figure_study_scaling(sweep, save=False, out_dir=tmp_path)
    drawn = [ax for ax in fig.axes if ax.has_data()]
    assert len(drawn) == 3


def test_panels_are_labeled_for_their_quantities(sweep, tmp_path):
    fig = figure_study_scaling(sweep, save=False, out_dir=tmp_path)
    labels = " ".join(ax.get_ylabel() for ax in fig.axes).lower()
    assert "second" in labels
    assert "gb" in labels or "memory" in labels
    assert "k" in labels


def test_memory_is_plotted_in_gigabytes_not_bytes(sweep, tmp_path):
    # Plotting raw bytes would put 9e9 on the axis and make the panel
    # unreadable, which no shape assertion would catch.
    fig = figure_study_scaling(sweep, save=False, out_dir=tmp_path)
    memory_ax = [ax for ax in fig.axes if "gb" in ax.get_ylabel().lower()][0]
    ydata = memory_ax.lines[0].get_ydata()
    assert max(ydata) == pytest.approx(9.0, rel=0.01)


def test_x_axis_is_the_sample_count(sweep, tmp_path):
    fig = figure_study_scaling(sweep, save=False, out_dir=tmp_path)
    for ax in fig.axes:
        if ax.has_data():
            assert list(ax.lines[0].get_xdata()) == [1000, 5000, 25_000]


def test_writes_under_the_requested_name(sweep, tmp_path):
    figure_study_scaling(sweep, save=True, out_dir=tmp_path)
    assert (tmp_path / "heca_scaling.png").is_file()


def test_does_not_write_when_save_is_false(sweep, tmp_path):
    figure_study_scaling(sweep, save=False, out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_empty_sweep_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one row"):
        figure_study_scaling(pd.DataFrame(columns=["n"]), save=False, out_dir=tmp_path)
```

- [ ] Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest tests/benchmarks/figures/test_study_scaling.py -v`
Expected: FAIL, `ImportError: cannot import name 'figure_study_scaling'`.

- [ ] Step 3: Implement

Create `src/benchmarks/figures/_study_scaling.py`:

```python
"""Cost of a case study as a function of sample count.

Runtime and peak resident memory answer the reviewer request directly;
selected k alongside them shows whether the selection itself is stable as n
grows, which a cost-only figure would leave open.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._theme import save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path

PANELS: tuple[tuple[str, str], ...] = (
    ("wall_clock_s", "Runtime (seconds)"),
    ("peak_rss_gb", "Peak memory (GB)"),
    ("selected_k", "Selected k"),
)


def figure_study_scaling(
    sweep_df: pd.DataFrame,
    *,
    save: bool = True,
    out_dir: Path | None = None,
    save_name: str = "heca_scaling.png",
) -> Figure:
    """Draw runtime, peak memory and selected k against sample count."""
    if sweep_df.empty:
        raise ValueError("figure_study_scaling needs at least one row to draw.")

    frame = sweep_df.sort_values("n").copy()
    # Bytes on the axis would render as 9e9 and be unreadable.
    frame["peak_rss_gb"] = frame["peak_rss_bytes"] / 1e9

    with theme_context():
        fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 3.0))
        for ax, (column, label) in zip(axes, PANELS):
            ax.plot(frame["n"], frame[column], marker="o")
            ax.set_xlabel("Number of cells (n)")
            ax.set_ylabel(label)
            if column != "selected_k":
                ax.set_xscale("log")
                ax.set_yscale("log")

        fig.tight_layout()
        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
```

- [ ] Step 4: Export it

In `src/benchmarks/figures/__init__.py`, add `from ._study_scaling import figure_study_scaling` and add the name to `__all__`.

- [ ] Step 5: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/benchmarks/figures/test_study_scaling.py -v`
Expected: PASS, 7 tests.

- [ ] Step 6: Lint and commit

```bash
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add src/benchmarks/figures/ tests/benchmarks/figures/test_study_scaling.py
git commit -m "feat(benchmarks): add the case-study scaling figure"
```

---

### Task 10: The two notebooks

Files:
- Create: `notebooks/case_studies/Cusanovich.ipynb`, `notebooks/case_studies/hECA.ipynb`
- Modify: `tests/benchmarks/test_notebooks.py`
- Test: `tests/benchmarks/test_notebooks.py`

Interfaces:
- Consumes: everything from Tasks 1 through 9.
- Produces: two executable notebooks following the structure of `notebooks/case_studies/Klein.ipynb`.

Notebooks are executed with `jupyter nbconvert --to notebook --execute --output-dir=<scratch>`, never `--inplace`: in-place execution re-commits megabytes of base64 image blobs and makes the diff unreviewable. nbconvert sets the working directory to the notebook's own directory.

- [ ] Step 1: Extend the notebook configuration test

`tests/benchmarks/test_notebooks.py` already checks that a notebook's committed configuration matches its sibling entry in `STUDIES`. Read the existing check near line 88 and add the equivalent for the two new notebooks:

```python
def test_cusanovich_notebook_reads_its_config_from_studies():
    import json
    from pathlib import Path

    source = "".join(
        "".join(cell["source"])
        for cell in json.loads(
            Path("notebooks/case_studies/Cusanovich.ipynb").read_text()
        )["cells"]
        if cell["cell_type"] == "code"
    )
    # Configuration must be read from STUDIES, not restated. Four manuscript
    # mismatches on this project came from re-derivation at the call site.
    assert 'STUDIES["cusanovich"]' in source
    assert "study_model_grids(study)" in source
    assert "candidate_k=study.candidate_k" in source
    assert "range(4, 17)" not in source


def test_heca_notebook_reads_its_config_from_studies():
    import json
    from pathlib import Path

    source = "".join(
        "".join(cell["source"])
        for cell in json.loads(
            Path("notebooks/case_studies/hECA.ipynb").read_text()
        )["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'STUDIES["heca"]' in source
    assert "study_resolution_grids(study)" in source
    assert "consensus_anchors=study.consensus_anchors" in source
```

- [ ] Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest tests/benchmarks/test_notebooks.py -v`
Expected: FAIL, `FileNotFoundError` for the two notebooks.

- [ ] Step 3: Write the Cusanovich notebook

Create `notebooks/case_studies/Cusanovich.ipynb` with these cells, in order. Markdown cells carry no bold or italics.

Markdown: a title cell naming the dataset, the accession and the four collection facts (81,173 cells, 436,206 peaks, 13 tissues, 40 reported cell labels of which one is Unknown), and a sentence saying the reference label is tissue of dissection and is therefore independent of any clustering.

Code cell 1, setup:

```python
from pathlib import Path

import matplotlib.pyplot as plt

from benchmarks._studies import (
    STUDIES,
    carve_cache_path,
    cvi_sweep,
    fit_or_load_carve,
    load_study,
    study_model_grids,
)
from benchmarks.figures import figure_cusanovich_results, prepare_composite
from benchmarks.figures._paths import CASE_STUDY_DIR

RANDOM_SEED = 42
SCALE = "dev"  # switch to "publication" for the manuscript run

# Figures are written under a scale-qualified directory, so a development
# figure can never be mistaken for one that belongs in the manuscript. Only
# the publication directory is ever copied into overleaf/vis/.
OUT_DIR = CASE_STUDY_DIR if SCALE == "publication" else CASE_STUDY_DIR / SCALE
OUT_DIR.mkdir(parents=True, exist_ok=True)

study = STUDIES["cusanovich"]
model_grids = study_model_grids(study)
X, y, meta = load_study(study, scale=SCALE)
print(meta["n_cells"], "cells,", meta["n_features"], "components,", meta["scale"])


def show(fig):
    """Display a figure exactly once, then close it."""
    display(fig)
    plt.close(fig)
```

Code cell 2, the CVI baseline:

```python
curves_df, best_df = cvi_sweep(
    X, y, model_grids=model_grids, candidate_k=study.candidate_k,
    random_state=RANDOM_SEED, n_jobs=-1,
)
best_df
```

Code cell 3, the CARVE fit:

```python
carve = fit_or_load_carve(
    X, y,
    cache_path=carve_cache_path(
        study, scale=SCALE, root=Path("./carve_state_saves")
    ),
    model_grids=model_grids,
    random_state=RANDOM_SEED,
)
```

Code cell 4, the figures:

```python
inputs = prepare_composite(
    X, y.to_numpy(), carve, curves_df=curves_df, best_df=best_df,
    comparison_metric="silhouette", measure="stability", rule="1se",
    random_state=RANDOM_SEED,
)
show(figure_cusanovich_results(inputs, out_dir=OUT_DIR))
```

Code cell 5, the full-atlas pass. Spectral and Ward cannot run at 81,173 cells, so this pass sweeps Leiden resolution on the anchored path and checks that the subsample's conclusion survives. Skipped unless the atlas scale is selected:

```python
from carve import CARVE

from benchmarks._studies import study_resolution_grids

if SCALE == "publication":
    X_atlas, y_atlas, atlas_meta = load_study(study, scale="atlas")
    print(atlas_meta["n_cells"], "cells in the full atlas")

    atlas = CARVE(
        estimator_param_grids=study_resolution_grids(study),
        n_resamples=100,
        n_jobs=1,
        random_state=RANDOM_SEED,
    ).fit(X_atlas)
    display(
        atlas.estimator_results_[["config_id", "resolution", "ari_stability"]]
    )
```

Markdown: a closing cell stating the selected k, which of the three reported resolutions it corresponds to (13 tissues, 30 clusters, 40 cell labels), and the ARI against the tissue labels. Note plainly that the selected k falling well below 40 is the expected finding rather than a limitation, because the finest reported partition is not expected to be the most stable.

- [ ] Step 4: Write the hECA notebook

Create `notebooks/case_studies/hECA.ipynb` with these cells.

Markdown: a title cell naming the dataset, the Zenodo record, the pooled organs, the pooled cell count, and one sentence recording that the pooled set spans several study_id values so a pooled clustering may partly recover study of origin rather than biology.

Code cell 1, setup:

```python
from pathlib import Path

import matplotlib.pyplot as plt

from benchmarks._studies import (
    STUDIES,
    carve_cache_path,
    cvi_sweep,
    fit_or_load_carve,
    load_study,
    study_model_grids,
    study_resolution_grids,
    study_scaling_sweep,
)
from benchmarks.figures import (
    figure_heca_results,
    figure_study_scaling,
    prepare_composite,
)
from benchmarks.figures._paths import CASE_STUDY_DIR

RANDOM_SEED = 42
SCALE = "dev"  # switch to "publication" for the manuscript run

# Scale-qualified output, so a development figure can never be mistaken for
# one that belongs in the manuscript.
OUT_DIR = CASE_STUDY_DIR if SCALE == "publication" else CASE_STUDY_DIR / SCALE
OUT_DIR.mkdir(parents=True, exist_ok=True)

study = STUDIES["heca"]
model_grids = study_model_grids(study)
X, y, meta = load_study(study, scale=SCALE)
print(meta["n_cells"], "cells across", meta["organs"], "|", meta["scale"])


def show(fig):
    display(fig)
    plt.close(fig)
```

Code cell 2, the k-based CARVE run:

```python
carve = fit_or_load_carve(
    X, y,
    cache_path=carve_cache_path(
        study, scale=SCALE, root=Path("./carve_state_saves")
    ),
    model_grids=model_grids,
    random_state=RANDOM_SEED,
    consensus_anchors=study.consensus_anchors,
)
```

Code cell 3, the Leiden resolution run. A separate CARVE object because SweepSpec is frozen:

```python
from carve import CARVE

leiden = CARVE(
    estimator_param_grids=study_resolution_grids(study),
    n_resamples=100,
    n_jobs=1,
    random_state=RANDOM_SEED,
    consensus_anchors=study.consensus_anchors,
).fit(X)
leiden.estimator_results_[["config_id", "resolution", "ari_stability"]]
```

Code cell 4, the scaling ladder. The publication ladder runs the full set of sizes; the development ladder stops at 25,000:

```python
LADDER = {
    "dev": [5_000, 10_000, 25_000],
    "publication": [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, X.shape[0]],
}[SCALE]

sweep_df = study_scaling_sweep(
    X, y, sizes=LADDER, model_grids=model_grids,
    consensus_anchors=study.consensus_anchors, random_state=RANDOM_SEED,
)
show(figure_study_scaling(sweep_df, out_dir=OUT_DIR))
sweep_df
```

Code cell 5, the composite figure:

```python
curves_df, best_df = cvi_sweep(
    X, y, model_grids=model_grids, candidate_k=study.candidate_k,
    random_state=RANDOM_SEED, n_jobs=-1,
)
inputs = prepare_composite(
    X, y.to_numpy(), carve, curves_df=curves_df, best_df=best_df,
    comparison_metric="silhouette", measure="stability", rule="1se",
    random_state=RANDOM_SEED,
)
show(figure_heca_results(inputs, out_dir=OUT_DIR))
```

Markdown: a closing cell reporting the selected k, the Leiden resolution CARVE prefers and how its labels compare to the k-based selection, and the measured runtime and peak memory at the largest n.

- [ ] Step 5: Add the consensus_anchors passthrough

`fit_or_load_carve` does not currently accept `consensus_anchors`. In `src/benchmarks/_studies.py`, add `consensus_anchors: int | None = None` to its signature, document it, and pass it into the `CARVE(...)` construction only when it is not None, so studies that leave it at the package default are unaffected:

```python
    carve = CARVE(
        estimator_param_grids=model_grids,
        n_resamples=n_resamples,
        n_jobs=n_jobs,
        random_state=random_state,
        **({} if consensus_anchors is None else
           {"consensus_anchors": consensus_anchors}),
    )
```

- [ ] Step 6: Execute both notebooks

```bash
SCRATCH=/private/tmp/claude-501/notebook-run && mkdir -p "$SCRATCH"
.venv/bin/jupyter nbconvert --to notebook --execute \
  --output-dir="$SCRATCH" notebooks/case_studies/Cusanovich.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute \
  --output-dir="$SCRATCH" notebooks/case_studies/hECA.ipynb
```

Expected: both complete without error at `SCALE = "dev"`. Never pass `--inplace`. If the loaders raise `FileNotFoundError`, the datasets have not been downloaded; follow the message's instructions and rerun.

- [ ] Step 7: Run the tests

Run: `.venv/bin/python -m pytest tests/benchmarks/test_notebooks.py -v`
Expected: PASS.

- [ ] Step 8: Commit

Commit the notebooks with cleared outputs, so the diff stays reviewable:

```bash
.venv/bin/jupyter nbconvert --clear-output --inplace \
  notebooks/case_studies/Cusanovich.ipynb notebooks/case_studies/hECA.ipynb
.venv/bin/ruff format src/ && .venv/bin/ruff check src/
git add notebooks/case_studies/Cusanovich.ipynb notebooks/case_studies/hECA.ipynb \
        src/benchmarks/_studies.py tests/benchmarks/test_notebooks.py
git commit -m "feat(benchmarks): add the Cusanovich and hECA case-study notebooks"
```

---

## Completion checklist

- [ ] `.venv/bin/ruff check src/` and `.venv/bin/ruff format --check src/` both clean
- [ ] `.venv/bin/python -m pytest tests/ -v --tb=short` passes with no new failures
- [ ] `.venv/bin/python -m pytest tests/ -q --cov=carve --cov-report=term-missing --cov-fail-under=75` passes
- [ ] Both notebooks execute end to end at `SCALE = "dev"`
- [ ] The Klein cache has been regenerated under its new scale-qualified filename and `Klein.ipynb` still reproduces k=4 with Ward agglomerative under `get_k(measure="generalizability", rule="1se", not_two=True)`
- [ ] No dataset archive, extracted h5ad, or `.carve` cache has been committed
- [ ] The publication runs are still outstanding: both notebooks must be re-executed at `SCALE = "publication"` and the resulting figures copied into `overleaf/vis/` by hand
