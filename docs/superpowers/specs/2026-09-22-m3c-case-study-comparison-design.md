# M3C comparison in the Klein case study

Date: 2026-09-22
Status: design approved, not implemented
Reviewer comment addressed: R2.1

## Motivation

Reviewer 2's first comment is that the manuscript compares CARVE only against classical
geometric CVIs:

> The work compared CARVE only with classical geometric CVIs, which were known to be suboptimal
> for high-dimensional biological data. It does not benchmark against other resampling-based or
> consensus-based tools that are already widely used in the field, such as SC3 or M3C. It remains
> unclear whether CARVE offers a genuine advance or merely a repackaging of existing ideas.

The comment is about whether CARVE advances on existing resampling-based tools. Answering it
requires a comparator that selects a clustering, not merely one that produces consensus matrices.
This design adds M3C to the Klein case study as that comparator.

## Decisions

The following were settled during design and should not be relitigated without new information.

### Only M3C, not SC3

SC3 is named alongside M3C in R2.1 and is already installed locally, but it was excluded
deliberately. SC3 is consensus clustering; it offers no principled default for preferring one
clustering over another, so it cannot serve as a selection comparator. M3C can, because RCSI
together with the Monte Carlo p-value against K=1 is an actual selection rule. This mirrors the
distinction the manuscript's software comparison table already draws.

Spectrum, ConsensusClusterPlus, sharp and diceR are out of scope for the same or weaker reasons.

### Only the Klein case study

Klein is the one case study where M3C runs entirely at its published defaults with nothing to
defend. Its CARVE sweep covers k in 2 to 10, which is exactly M3C's default maxK of 10, so the two
tools search the same range without any parameter being adjusted in either direction.

Levine was considered and dropped. It would require maxK=17 to cover its CARVE window, and at
n=5,000 it sits five times beyond M3C's own recommended sample range. Cusanovich and hECA are
excluded structurally: Cusanovich sweeps Louvain resolution and has no k-based arm at all, and both
datasets are far beyond the scale at which M3C's dense consensus matrices are tractable.

### M3C is run as its authors recommend, not matched to CARVE's algorithm grid

An earlier draft of this design proposed running M3C once per algorithm in each study's CARVE grid,
so that the only difference between the CARVE row and the M3C row would be the selection rule. That
was rejected in favor of the comparison the reviewer is actually asking about: what a user gets
walking up to CARVE versus walking up to M3C.

This is also the more informative comparison, because M3C offers the user no algorithm grid at all.
A CARVE user gets a sweep over algorithms and k with a selection rule; an M3C user gets PAM,
K at most 10, and a p-value against K=1. The manuscript's comparison table already records this
asymmetry (M3C's "Multi" column is No); the case study now demonstrates it.

The matched-grid approach was also impractical. M3C's spectral path calls a full dense eigen
decomposition of the subsampled affinity once per K per resample, without caching a decomposition
that does not depend on K. Measured on this machine, that decomposition takes 1.15 s at n=1,086 and
56.0 s at n=4,000, which put the Levine spectral cell at roughly 27 core-days at M3C's default
resample count.

## What M3C is and how it will be run

M3C is Monte Carlo reference-based consensus clustering (John et al., 2020), a Bioconductor package,
current release 1.34.0. It resamples the data, builds a consensus matrix per K, and scores each K by
an entropy or PAC objective. Its contribution over Monti consensus clustering is a Monte Carlo null:
it simulates reference datasets that preserve the feature correlation structure of the input, scores
them the same way, and compares. The comparison yields the Relative Cluster Stability Index (RCSI)
and an empirical p-value for rejecting K=1.

The recommended workflow, per the package vignette, is a single call:

```r
res <- M3C(mydata)
res$scores          # see the schema note below
res$assignments     # labels at the selected K
```

The scores frame's schema depends on the objective, which is a trap worth stating once. M3C builds
the frame with columns named PAC_REAL and PAC_REF, and then, under the default
objective="entropy", renames columns 2 and 3 in place on its last line:
colnames(real)[2:3] <- c("ENTROPY_REAL", "ENTROPY_REF"). Under the default the frame is therefore

    K, ENTROPY_REAL, ENTROPY_REF, RCSI, RCSI_SE, MONTECARLO_P, NORM_P, P_SCORE

and PAC_REAL and PAC_REF do not exist. They survive only under objective="PAC", which also
substitutes BETA_P for NORM_P. Code that reads PAC_REAL will raise a KeyError against the default
run. RCSI, RCSI_SE and MONTECARLO_P are named the same under both objectives, and those are the
three the selection and the figure use.

with defaults method=1 (Monte Carlo), clusteralg="pam" with Euclidean distance, maxK=10, iters=25,
repsref=100, repsreal=100, pItem=0.8, objective="entropy", seed=123. The user reads res$scores, takes
K at the RCSI maximum, and checks that the Monte Carlo p-value rejects K=1 at alpha=0.05. The
vignette's only tuning advice is that iters may be lowered to 5 to 20 for speed and that repsreal
should stay at 100 or above.

This design runs exactly that, with maxK=10 passed explicitly (Klein's CARVE ceiling, which
coincides with M3C's default) and with removeplots=TRUE and silent=TRUE added. Those two are
presentation flags, not method flags.

### Input requirements, and why Klein already satisfies them

M3C requires a matrix of normalized continuous data with samples in columns and features in rows,
filtered to remove low-signal features and reduced by variance, with outliers removed. Klein's
analysis matrix is 1,358 cells by 2,000 highly variable genes after total-count normalization to
1e4 and log1p. The HVG selection is a variance filter, and log1p-normalized expression is
approximately the distributional shape M3C's reference assumes. M3C can therefore be fed the exact
same X that CARVE and the CVIs see, transposed. No M3C-specific preprocessing is introduced, which
is what makes the comparison a head-to-head.

With p=2,000 greater than n=1,358, ref_method stays at M3C's preferred reverse-pca. M3C only
switches to chol when samples exceed features.

The des argument is passed as None. It only reorders M3C's output annotation and has no effect on
clustering. This is recorded here to stop it being added later in the belief that it does.

### The authors' own scope statement

The M3C vignette states:

> We recommend M3C only be used to cluster datasets with high numbers of samples (e.g. 60-1000).
> M3C is mainly aimed at large patient cohort datasets such as those produced by TCGA and other
> consortia. Because of the high complexity of the algorithm and the type of consensus matrix it
> makes, it is not that well suited for single cell RNA-seq data, better to use SC3 or Spectrum,
> for example.

Klein at n=1,358 sits just above that band. This is reported rather than hidden, and it is a
substantive part of the answer to R2.1: the closest consensus-based tool declines the single-cell
use case its own authors describe, and is O(n^2) in the consensus matrix where CARVE's anchored
consensus is not.

### Measured cost

Timed on this machine (R 4.5.1, Apple silicon). A run performs iters * repsref + repsreal = 2,600
resamples, each clustering at every K from 2 to maxK.

| quantity | measurement |
| --- | --- |
| dist on 1,358 by 2,000 | 1.23 s, computed once |
| pam at n=1,086 (pItem=0.8), per K | about 0.02 s |
| Klein run total, cores=1 | about 8 minutes |

cores stays at 1. The run is fast enough serially, and that removes a reproducibility variable.

## Architecture

### The module

New file: `src/benchmarks/_m3c.py`. It is a leaf. It imports the standard library, numpy, pandas and
`_types`, and imports no other benchmarks module and not carve. Nothing in the existing dependency
graph moves.

```python
M3C_DEFAULTS = {                  # M3C 1.34.0's published defaults, restated here
    "method": 1, "clusteralg": "pam", "objective": "entropy",
    "iters": 25, "repsref": 100, "repsreal": 100, "pItem": 0.8, "seed": 123,
}

@dataclass(frozen=True)
class M3CResult:
    scores: pd.DataFrame          # K, ENTROPY_REAL, ENTROPY_REF, RCSI,
                                  # RCSI_SE, MONTECARLO_P, NORM_P, P_SCORE
    labels: dict[int, np.ndarray] # assignments per K, in the row order of X
    selected_k: int
    p_value: float                # MONTECARLO_P at selected_k
    runtime_s: float
    config: Mapping[str, Any]     # the defaults actually used, including maxK
    m3c_version: str
    r_version: str

def select_k_m3c(scores: pd.DataFrame) -> int: ...
def run_m3c(X, *, max_k, allow_install=False, cores=1, verbose=False) -> M3CResult: ...
def run_or_load_m3c(X, *, cache_path, force=False, **kwargs) -> M3CResult: ...
def m3c_cache_path(study, *, scale=None, root, config) -> Path: ...
```

Four properties of this interface are load-bearing.

Defaults are pinned in our source, not inherited from R. M3C_DEFAULTS restates the published values
rather than relying on M3C()'s own, so a Bioconductor version that changes a default appears as a
diff here instead of silently changing a manuscript number.

Orientation is handled once and guarded. M3C wants features in rows; X is cells in rows. run_m3c
transposes internally. Because a transposition error produces plausible-looking output rather than
a crash (2,000 "samples" of 1,358 "features"), the result is checked: every label vector must have
length X.shape[0], or run_m3c raises. Columns are named c0 through c1357, because M3C() runs
gsub("X", "", colnames(...)) on matrix input and would mangle any name containing an X. Labels are
reindexed by those names.

Selection is reimplemented in Python and cross-checked, not trusted. select_k_m3c returns
int(scores.loc[scores["RCSI"].idxmax(), "K"]): a pure function, testable without R, and visible in
our source rather than buried in R. What M3C itself returned is compared against it, and a
disagreement raises.

Note that M3C's own R expression is which.max(real$RCSI) + 1, where the plus one converts a 1-based
row index into a K value because K starts at 2. Reproducing that arithmetic in Python would be
wrong twice over, since numpy's argmax is 0-based. Read the K column instead of recomputing the
offset. This is the kind of index-to-label arithmetic the project already treats as a hazard
elsewhere, in the rule that config_id is a join key and never a positional index.

Labels are collected for every K, not only the selected one. They are read from
res$realdataresults[[k]]$assignments, which is the named vector in the original column order.
res$realdataresults[[k]]$ordered_annotation is permuted by the dendrogram order and would scramble
the mapping to our rows. Keeping every K lets the figure show M3C's partition at its own selected K
and at CARVE's k=4, which is the sharper comparison.

Error handling and diagnostics follow the existing shape. allow_install defaults to False and
raises a RuntimeError naming BiocManager::install("M3C"), as
`datasets/_levine.py:_load_from_r` already does for HDCytoData. Diagnostics go through
warnings.warn(..., stacklevel=2) and verbose-gated print. No logging is introduced.

### Caching, provenance, reproducibility

run_or_load_m3c mirrors fit_or_load_carve: the result is written as parquet plus a JSON sidecar, and
an .x-sha1 fingerprint sidecar guards it so a cached result is never served against a matrix it was
not computed on. m3c_cache_path keys the filename on the study, scale, resolved size and a hash of
the pinned config, the way carve_cache_path keys on scale, size and run.

That fingerprint guard currently exists as three private functions in `_studies.py`: `_fingerprint`,
`_fingerprint_path` and `_check_fingerprint`. Move them into `_artifacts.py` and have both modules
import them, rather than copying them. `_artifacts` already owns caching and provenance, and it
imports only `_registry` and `_types`, so no cycle is introduced. `_studies`' public behavior is
unchanged. A second copy of a correctness check is the drift pattern this project has already been
bitten by.

Provenance needs one extension. `_TRACKED_PACKAGES` in `_artifacts.py` records Python package
versions only, and M3C's version is an R fact. M3CResult carries m3c_version and r_version, read
from R at run time via packageVersion("M3C") and R.version.string, and both are written into the
cache sidecar so a stored result records which M3C produced it.

On seeding: M3C sets a global R seed by design (seed=123). This is at odds with CARVE's rule that
seeds are derived arithmetically and never shared, but that rule governs CARVE's internals, not an
external tool being run as published. The seed is pinned, recorded in config, and left alone.

## Reporting

Fig 5 is not modified. It is a published six-panel composite whose panels the main text and S6 Text
cite by letter, and adding a panel would ripple into prose. M3C gets its own supplementary figure.

New file: `src/benchmarks/figures/_klein_m3c.py`, exposing figure_klein_m3c, saving klein_m3c.png
into CASE_STUDY_DIR through figure_path, inside theme_context. No module-level color literals;
`_theme` stays the single source for palette, rcParams and geometry.

Three panels:

- Panel A: M3C's own selection evidence. RCSI over K with plus or minus 1.96 times RCSI_SE error
  bars, which is M3C's own idiom, the selected K marked, and the Monte Carlo p-value annotated.
  Drawn through a new `_panels.m3c_lines`, so palette and geometry come from `_theme` exactly as
  carve_lines and cvi_lines already do.
- Panel B: M3C's partition at its selected K, as a PCA scatter in the same embedding and through
  the same composite_color_maps and `_align_to_reference` as Fig 5's panels A to C, so it reads
  directly beside CARVE's k=4 and the CVI selection at k=2.
- Panel C: ARI against the reported labels for CARVE, the four CVIs and M3C, as the ari_lollipop
  panel Klein's Fig 5 does not currently carry.

The figure is three panels in every case. M3C's partition at k=4, which answers the sharper question
of what M3C says at the k CARVE selected, is carried as a row in the table rather than as a fourth
scatter. Drawing it conditionally would mean a figure whose panel count depends on the result, and
drawing it unconditionally would duplicate panel B whenever M3C also selects 4.

ari_table is shared with the Levine and hECA figures. It gains an optional extra_rows parameter
defaulting to None, appended after the CVI rows. The default leaves those two figures unchanged;
the Klein M3C figure passes two rows, M3C at its selected K and M3C at k=4. One shared function, no
second copy.

Numbers for the response letter go out as a table: render_m3c_tex in `_tables.py`, beside
render_ablation_tex, over the same frame panel C draws, with columns for method, selected k and ARI
against the reported labels, and one row per method plus the M3C at k=4 row. Runtime is deliberately
not a table column: fit_or_load_carve caches state rather than timings, so a runtime column would be
measured under different conditions for CARVE than for M3C. Runtimes are reported in prose from a
measured one-off for both tools.

## Testing

Test files mirror their modules one to one: `tests/benchmarks/test_m3c.py` and
`tests/benchmarks/figures/test_klein_m3c.py`. The new `_panels.m3c_lines` is tested in the existing
`tests/benchmarks/test_panels.py`, where its siblings carve_lines and cvi_lines already are.

CI has no R, so the split between what runs there and what does not is part of the design. The R
call is a thin wrapper with everything else pulled out of it, which is what makes most of the module
testable without R.

Runs in CI, no R required:

- select_k_m3c against a recorded fixture.
- The cross-check raising when Python's selection and M3C's reported selection disagree.
- The orientation guard raising on a transposed X.
- The scores frame schema.
- m3c_cache_path keying.
- The fingerprint mismatch raising.
- The allow_install=False error naming BiocManager::install("M3C").

Skipped unless R and M3C are both present: a requires_r guard checking that rpy2 imports and that
M3C is installed, running M3C on a small synthetic matrix (about n=80, p=50, inside M3C's own
recommended band) at iters=5 so it completes in seconds, and asserting the result round-trips
through the cache unchanged.

The fixture is the real res$scores frame from the Klein run, recorded once into `tests/fixtures/`,
so the CI tests exercise M3C's actual output rather than invented numbers.

Every assertion must be able to fail, which constrains the fixtures:

- The select_k_m3c fixture must have its RCSI maximum somewhere other than the first row, and its K
  column must start at 2 rather than 0. Otherwise a selection that returns a row index instead of a
  K value passes, which is the exact failure the K-column lookup exists to prevent.
- The orientation test must pass a genuinely transposed X, not merely a well-shaped one.
- The fingerprint test must use a different X of the same shape. Otherwise a shape check alone would
  catch it and the fingerprint would go unexercised.

filterwarnings = error is on, so any warning raised by R or rpy2 is met with a match-asserted filter
or pytest.warns, never a blanket ignore.

## Dependencies

rpy2 is already declared under the notebooks extra in `pyproject.toml`, and
`datasets/_levine.py` already uses it. No Python dependency changes.

M3C is an R package and is not currently installed on this machine. It belongs in the Makefile, not
in `pyproject.toml`: add an m3c-setup target beside the existing r-setup target, running
BiocManager::install("M3C").

## Running it

One new cell in `notebooks/case_studies/Klein.ipynb`, after the existing CARVE fit, reusing the X
and y already in scope. The case-study notebooks are not executed in CI: the nightly workflow runs
only Benchmarking.ipynb, Tutorial.ipynb and Resolution_Tutorial.ipynb, and
`tests/benchmarks/test_notebooks.py` checks notebooks structurally without executing them. A cell
requiring R is therefore safe.

Outputs:

- The cached M3C result in `notebooks/case_studies/carve_state_saves/`, beside the CARVE state the
  notebook already caches there.
- `vis/case_studies/klein_m3c.png`. Both directories are gitignored, per existing convention.
- The comparison table as .tex and .csv.

Numbers produced: M3C's selected k, its RCSI curve, the Monte Carlo p-value at the selected k, ARI
against the reported labels at M3C's own k and at CARVE's k=4, and the measured runtime.

## Manuscript handoff

`../overleaf/` is read-only, so these are the author's to write. The implementation delivers the
figure, the table and the numbers.

- A sentence in the Klein results paragraph giving M3C's selected k and its ARI against the reported
  labels.
- A short S6 Text paragraph recording M3C's version, defaults, maxK, seed and runtime, alongside the
  vignette's own scope statement.
- A supporting-information figure entry with its legend after the reference list, which the
  outstanding journal-formatting item requires anyway.
- Response-letter text for R2.1, covering the result, why M3C rather than SC3, and M3C's authors'
  own position on single-cell data.

The manuscript's software comparison table already lists M3C with Multi set to No. No change is
needed there; the case study now demonstrates what that column asserts.

## Risks and contingencies

If RCSI peaks at the boundary K=10, report it and do not extend maxK. Extending would hand M3C a
search window CARVE never had, which inverts the fairness this design exists to preserve.

If M3C fails to reject K=1 at alpha=0.05, that is a reportable result, not a failed run. It means
the Monte Carlo null cannot distinguish Klein's structure from its reference at any K, which is
directly relevant to R2.1 and should be stated plainly rather than worked around.

If the installed M3C version is not 1.34.0, record the actual version in the spec's numbers and
check M3C_DEFAULTS against that version's signature before running. The pinned defaults exist to
make exactly this discrepancy visible.

## Out of scope

- The Levine, Cusanovich and hECA case studies.
- SC3, Spectrum, ConsensusClusterPlus, sharp and diceR.
- The simulated benchmarks. R2.1's violet annotation suggested the simulation study; this design
  answers it in the case studies instead, which is where the to-do list places it.
- Any carve-r mirror. `benchmarks/` has no R counterpart, so the one-to-one mirror rule does not
  apply.
- Runtime and memory reporting beyond the single measured number. R1.1 has its own machinery.
