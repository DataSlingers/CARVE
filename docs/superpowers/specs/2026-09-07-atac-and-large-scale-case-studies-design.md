# scATAC-seq and large-scale case studies: design

Date: 2026-09-07
Status: approved for planning
Supersedes: nothing
Related: `docs/superpowers/specs/2026-09-01-benchmarks-rebuild-design.md`

## 1. Motivation

The PLOS Computational Biology review of the CARVE manuscript raised two requests that the
current case studies do not answer:

- R1.1: report runtime, memory usage, computational complexity and parallelization strategies
  on datasets of increasing size, including large-scale single-cell datasets, for example more
  than 500,000 cells.
- R1.7: evaluate CARVE on scATAC-seq data, which is sparser than scRNA-seq, and establish
  whether the same resampling strategy remains reliable. The reviewer notes this may require
  code changes if the data is not tabular.

Two further review items are carried by this work at low marginal cost, because the datasets
and instrumentation needed for R1.1 and R1.7 supply them anyway:

- R1.4: run Leiden or Louvain on case-study data and compare their labels to CARVE's.
- R1.1 (memory and complexity): report peak resident memory and empirical complexity, not
  runtime alone. The published S5 Text reports runtime only.

Explicitly not carried by this work, by decision: R1.8 (practical-guidance appendix) and R2.1
(benchmarking against SC3 and M3C). Both remain open review items to be addressed separately.

## 2. Scope

This design covers two sequential plans.

Plan 1: anchored consensus in `src/carve/`, Python only. This lifts a hard ceiling on the
number of samples CARVE can process. It is a prerequisite for the large-scale case study and
for the full-atlas pass of the ATAC case study.

Plan 2: two new case studies in `src/benchmarks/`, following the existing case-study pipeline
(`STUDIES` -> `cvi_sweep` -> `fit_or_load_carve` -> composite figure -> notebook). The studies
deviate from the existing two only where the data forces a deviation, and every deviation is
recorded with its reason.

Out of scope for both plans:

- The R port of the anchored consensus. CLAUDE.md records that `carve-r/R/*.R` mirrors
  `src/carve/_*.py` one to one and that `r-ci.yml` checks this separately. Plan 1 deliberately
  breaks that mirror. The justification is that no reviewer asked the R package to scale: R3.2
  concerns user-supplied clustering algorithms and R3.7 concerns documentation. This must be
  recorded as a tracked follow-up, not left implicit.
- Sparse-native input to CARVE. `carve._utils.ensure_2d_array` densifies sparse input by
  design, because agglomerative clustering, t-SNE and the RBF affinity all require dense input.
  Both new studies cluster a dense low-dimensional embedding, so this limitation is not on the
  critical path.
- Changing the default generalizability classifier. Section 3 shows the classifier dominates
  runtime at scale, which makes it the obvious lever, and R2.3 asks for exactly that change.
  It is nonetheless a separate concern with its own accuracy consequences.

## 3. Measured findings that constrain the design

Every number in this section was measured on the target machine (Apple M3 Pro, 19 GB RAM,
11 cores, 37 GB free disk) rather than estimated, because the design turns on them.

### 3.1 The consensus matrix is the binding constraint

`carve._consensus.compute_consensus_matrix` allocates a dense `n x n` float32 array.
`carve._runner` builds two per configuration, one for stability and one for generalizability,
and retains all of them in `consensus_matrices_` and
`consensus_generalizability_matrices_`.

Measured with KMeans, three configurations, 20 resamples:

| n | one matrix | all matrices retained | peak RSS |
|---|---|---|---|
| 2,000 | 16 MB | 0.096 GB | 0.64 GB |
| 5,000 | 100 MB | 0.600 GB | 2.49 GB |
| 10,000 | 400 MB | 2.400 GB | 5.29 GB |

The growth is exactly quadratic. A realistic case-study grid has roughly 20 configurations
rather than three, so n = 10,000 already implies about 16 GB retained. At n = 50,000 a single
matrix is 10 GB. At n = 615,998 a single matrix would be approximately 1.5 PB.

`CARVE.get_labels` compounds this: it upcasts the selected matrix to float64 and derives `S`
and `D` from it, holding several `n x n` float64 copies, then cuts the result with
average-linkage `AgglomerativeClustering` on a precomputed distance matrix.

The share of fit time spent on consensus work, measured against the same fits:

| n | fit total | consensus construction and metrics | consensus share |
|---|---|---|---|
| 2,000 | 6.0 s | 0.2 s | 3% |
| 5,000 | 9.1 s | 1.3 s | 15% |
| 10,000 | 20.1 s | 11.0 s | 55% |

The residual, which is clustering plus classifier, grows approximately linearly across the
same range: 5.8 s, 7.7 s, 9.1 s.

Conclusion: the practical ceiling of the current implementation is approximately n = 15,000 to
20,000 for a multi-configuration case study, on any estimator. Choosing a cheaper clustering
algorithm does not move it.

### 3.2 Without the consensus matrix, atlas scale is tractable

Measured at n = 615,998 samples by 30 features, k = 10, subsample ratio 0.618:

| Component | Cost |
|---|---|
| KMeans, `n_init=10`, on 380,686 samples | 1.1 s |
| MiniBatchKMeans, same input | 0.04 s |
| RandomForestClassifier, 100 trees, `n_jobs=-1`, fit | 12.3 s |
| RandomForest predict on 235,312 held-out samples | 0.1 s |
| Anchored slab at m = 2,000 | 2.2 s |

Per resample this is approximately 14.6 s, so a full run at 100 resamples and 20
configurations is 8.1 hours on one core, 1.0 hour on eight, 0.7 hours on eleven.

Two consequences. First, the clustering is not the cost; the classifier is roughly 85% of
per-resample time. Second, the intermediate `B` matrix inside `compute_consensus_matrix` is
`n x (n_resamples * k)`, which is 2.46 GB at this scale, so the anchored implementation must
avoid materializing it rather than merely avoiding the square output.

### 3.3 Spectral and agglomerative clustering have their own quadratic ceiling

`carve.cluster.SpectralClustering._compute_self_tuning_affinity` calls
`pairwise_distances(X, metric="sqeuclidean")`, producing a dense `n x n` affinity that is then
eigendecomposed. Ward agglomerative clustering without a connectivity graph is likewise
quadratic in memory.

Both are therefore capped near n = 10,000 to 15,000 independently of the consensus issue. At
81,173 or 500,000 samples the viable estimators are KMeans, MiniBatchKMeans, Leiden and
Louvain. This is a constraint on study design, not a defect to fix here.

### 3.4 The AnnData and scanpy path works

Verified against `data/pbmc3k_processed.h5ad`:

- `CARVE().fit(adata, use_rep="X_pca", n_pcs=20)` runs and returns usable labels.
- `carve.tl.carve(adata, ...)` writes `obs["carve"]`, `obs["carve_stability"]`,
  `obs["carve_stability_ce"]`, `obs["carve_generalizability"]` and `uns["carve"]`.
- The resulting object survives `write_h5ad` and re-read.
- `carve.pl` exports `cluster_boxplot`, `cluster_scatter`, `cluster_violin`,
  `consensus_matrix`, `diagnostic_scatter` and `metric_over_n_clusters`.

On a modern h5ad, `obsm` is an HDF5 group, so an embedding can be read selectively without
touching `X`. This matters for any future dataset that ships a precomputed embedding, though
neither dataset chosen here does.

### 3.5 Dataset survey

Measured directly from the source files and archives rather than from publication text.

| Dataset | Size | Labels | Download | Verdict |
|---|---|---|---|---|
| Cusanovich mouse sci-ATAC atlas | 436,206 peaks by 81,173 cells, 421,971,103 nonzeros | `tissue` (13), `cluster` (30), `cell_label` (40) | 1.1 GB matrix plus 13 MB metadata | Selected for the ATAC study |
| hECA v2.0 ATAC | 1,450,511 cells, 25 organs, 101 harmonized cell types | `cell_type`, `organ`, `donor_id`, `study_id`, `seq_tech` | Zenodo record 15627886, per organ; Lung 4.52 GB, Brain 4.32 GB, Kidney 1.91 GB, Thymus 0.44 GB | Selected for the large-scale study |
| CATlas, Zhang et al. 2021 | 1,154,611 cCREs by 1,323,041 cells, 4,999,481,876 nonzeros | 111 adult cell types | GEO `GSE184462_RAW.tar` 36 GB, or UCSC `matrix.mtx.gz` 16 GB | Rejected: approximately 19 GB as CSR for the adult subset alone, against 19 GB total RAM |
| De Rop scATAC protocol benchmark | approximately 169,000 PBMCs, 47 samples, 8 protocols | transferred from an annotated scRNA reference | via SCope | Not selected |

hECA v2.0 ATAC incorporates the CATlas data under a harmonized re-annotation, so selecting
hECA recovers the rejected atlas through a tractable route.

Inspection of `ATAC-Thymus.h5ad`, the smallest organ, establishes the file shape for all of
them: 36,361 cells by 1,657,194 cPeaks, CSR int32 with 159,939,357 nonzeros, median 3,020 and
mean 4,399 nonzero peaks per cell. `obsm` is empty, so an embedding must be computed. `obs`
carries `cell_id`, `cell_type`, `donor_age`, `donor_gender`, `donor_id`, `markers`, `organ`,
`original_name`, `ref_genome`, `region`, `sample_status`, `seq_tech`, `study_id`, `subregion`.
Thymus has four `cell_type` levels, one of which is `Unclassified` at 3,295 cells, and spans
two `study_id` values.

## 4. Plan 1: anchored consensus

### 4.1 Principle

Partition CARVE's artifacts by what a restricted consensus computation actually costs.

| Artifact | Depends on the square matrix | Behavior under anchoring |
|---|---|---|
| `ari_stability`, `ari_generalizability`, and the `_1se` and `_quant` variants | No; computed per resample from label vectors | Unchanged. This is the manuscript's default selection path. |
| `generalizability_scores_` | No; `np.add.at` over n, linear | Unchanged. All n samples keep a score. |
| `stability_gini_scores_`, `stability_ce_scores_` | Partly; each score is a row mean | Computed from an `n x m` slab. All n samples keep a score, estimated over m random partners instead of n - 1, with variance of order 1/m. |
| `get_labels`, PAC, `plot_consensus_matrix` | Yes; requires a square block | Derived from the `m x m` anchor block, with labels extended to all n samples. |

### 4.2 Public interface

Two new constructor parameters on `CARVE`:

| Parameter | Default | Meaning |
|---|---|---|
| `anchor_threshold` | `5000` | If `n <= anchor_threshold`, the exact path runs unchanged. The comparison is inclusive so that the published Levine case study, at exactly 5,000 cells, keeps the exact path and its published numbers. |
| `consensus_anchors` | `None` | Anchor count as an integer, or a fraction of n as a float. `None` resolves to `min(n, anchor_threshold)`. |

The `min(n, anchor_threshold)` default is deliberate. A flat default such as 2,000 would make
the effective anchor count fall discontinuously from 5,000 at n = 5,000 to 2,000 at n = 5,001,
introducing an artificial jump in estimator variance exactly at the boundary. Resolving to
`min(n, anchor_threshold)` is continuous: n = 4,000 yields 4,000 anchors and the exact path,
n = 6,000 yields 5,000 anchors and the anchored path.

The cost of the larger default is retention rather than compute. At m = 5,000 an `m x m` block
is 100 MB against 16 MB at m = 2,000, so approximately 4 GB across 20 configurations rather
than 0.64 GB. Compute is negligible either way, at roughly 5.5 s against 2.2 s per
configuration set against a 14.6 s per-resample budget. Studies that are memory constrained set
`consensus_anchors` explicitly in their `STUDIES` entry, which is where study configuration
belongs.

### 4.3 Behavior

Anchor selection. The anchor index is drawn once per `fit` call and shared across every
configuration. Sharing is required, not an optimization: consensus matrices must remain
comparable across k, and `config_id` must continue to index the matrix lists in alignment with
`estimator_results_`. The draw is derived arithmetically from `random_state`, consistent with
the existing rule that seeds are derived and never shared through global state.

Consensus construction. As built, `compute_consensus_matrix` is left byte for byte identical
and two siblings are added beside it: `consensus_anchor_block`, which returns the `m x m`
block, and `stability_from_runs_anchored`, which produces the per-sample scores. Both draw on
a private `_anchor_factors` helper that accumulates only the anchor rows of the `S` and `B`
factors, so the `n x (n_resamples * k)` intermediate is never materialized. Section 3.2 shows
that intermediate is 2.46 GB at atlas scale, so avoiding it is required for the path to run at
all. `_runner` selects between the exact function and the anchored siblings on whether the
resolved anchor index is None.

This is not the interface this section originally described, which had `compute_consensus_matrix`
gain an `anchors` argument that returned the block when anchors were supplied and reproduced
current behavior when they were not. Siblings were chosen instead. The promise that results at
or below the threshold do not move is the entire basis of the published Klein and Levine
numbers, and the strongest way to honor it is to leave the exact function untouched rather than
to add a branch to it and then argue the branch is inert. Under siblings the exact path is
unchanged by inspection rather than by test, and no future edit to the anchored code can reach
it.

Per-sample stability scores. `stability_from_consensus` computes each sample's score as a row
mean, so restricting the columns to the anchor set leaves the estimator unbiased and only
increases its variance. The `n x m` slab is processed in row chunks and reduced to the two
length-n score vectors immediately, so peak memory is of order chunk size times m rather than
n times m. No `n x m` array is retained. The chunk row count defaults to a value derived from
m, so that bound stays roughly constant rather than growing with the anchor count; a caller
may still pass one explicitly, and the result does not depend on it.

Label extension. `get_labels` cuts the `m x m` block with the existing average-linkage
precomputed step, producing m anchor labels, then fits a clone of `self.classifier` on
`(X[anchor], anchor_labels)` and predicts the remaining n - m samples. The returned array has
length n, so `plot_cluster_boxplot`, `plot_cluster_violin`, `plot_cluster_scatter` and
`plot_diagnostic_scatter` continue to work without modification; each of those reads a
per-sample score array and calls `get_labels`, and both remain full length. This reuses the
out-of-sample prediction step that the generalizability path already performs on every
resample rather than introducing a new mechanism.

Retention. `m x m` blocks per configuration, plus the length-n per-sample score vectors.

Plotting. `plot_consensus_matrix` receives the `m x m` block and the corresponding anchor
labels. This requires a public accessor for the anchor index. Rendering the anchor block is
also the more defensible choice on its own terms: a square heatmap at atlas scale is neither
renderable nor readable, and published consensus heatmaps are downsampled in practice, so a
declared random anchor set is preferable to an ad hoc crop.

Diagnostics. A `warnings.warn` fires when the anchored path engages, naming n, m and the
threshold. The project uses `warnings.warn` and verbosity-gated `print`, never `logging`.

### 4.4 Invariants that must be preserved

- `config_id` remains a join key and never a positional index. The anchored matrix lists align
  with `estimator_results_` exactly as the exact ones do, and `fit` continues to assert this.
- Seeds are derived arithmetically. The anchor draw introduces no global seeding and no shared
  RNG state across joblib workers.
- Selection rules continue to operate on `sweep_rank` rather than `n_clusters`.
- The noise policy is unaffected. Under `noise_policy="drop"` an anchor that is noise in a
  given resample simply contributes nothing to that resample, exactly as any sample does now.
- Diagnostics go through `warnings.warn` and verbosity-gated `print`. No `logging`.

### 4.5 Verification

Two gates, both of which must be able to fail.

Backward compatibility. A test fits data at n = 5,000, asserts the anchored code path is not
taken, and asserts results are identical to the pre-change implementation. This test is
settled by mutation rather than by reading: forcing the anchored path on must turn it red. A
test that passes whether or not the threshold is honored is worthless here, and the threshold
is the entire basis of the promise that published results do not move.

Approximation accuracy. At n = 5,000 the exact answer is computable, so the anchored path is
validated against it on real data. For m in {500, 1000, 2000, 5000}, compare selected k, PAC,
correlation of the gini and CE score vectors, and ARI between anchored and exact labels. This
produces the evidence that the approximation is sound and is also a candidate supplementary
table for the manuscript. The fixtures must be chosen so that a wrong answer differs visibly
from a right one: anchor indices that differ from their positions, and score vectors that are
not constant.

Measured in `tests/test_anchored_accuracy.py`. Data: n = 5,000 points drawn from three
Gaussian blobs (centers at (0, 0), (8, 0), (4, 7), sigma 1.2), a KMeans grid over k in
{2, 3, 4}, n_resamples = 25, random_state = 0. The anchored fits use anchor_threshold =
10,000 so the anchored path runs at this n. They are compared against an exact fit on the
same data which takes the default anchor_threshold of 5,000, a value n equals exactly, so
the reference fixture also covers the inclusive boundary the Levine study sits on. The
selected k (measure stability, rule 1se) agreed between the anchored and exact fit at every
m tested (k = 3 in every case).

The table has three rows rather than the four m values named above. At n = 5,000 the fourth
value, m = 5,000, is the exact path by definition: `resolve_anchors` returns None once m
reaches n, so there is no anchored fit at that m to compare against. The identity that value
would have tested, that an anchor set covering every sample reproduces the full result, is
covered directly in `tests/test_consensus.py::TestConsensusAnchorBlock` and
`TestStabilityFromRunsAnchored` instead.

| m | Gini correlation (vs. exact) | CE correlation (vs. exact) | Label ARI (vs. exact) | Label ARI (vs. planted truth) | PAC (exact) | PAC (anchored) | PAC abs diff |
|---|---|---|---|---|---|---|---|
| 500 | 0.943 | 0.927 | 0.992 | 0.991 | 0.354 | 0.358 | 0.004 |
| 1000 | 0.986 | 0.981 | 0.993 | 0.993 | 0.354 | 0.358 | 0.004 |
| 2000 | 0.993 | 0.991 | 0.998 | 0.996 | 0.354 | 0.354 | 0.000 |

The gini correlation rises monotonically with m (0.943, 0.986, 0.993 at m = 500, 1000,
2000), consistent with the row-mean estimator's variance falling as 1/m; the CE
correlation follows the same pattern (0.927, 0.981, 0.991), which is expected since it is
the same estimator family under the same variance argument.

PAC is reported without an acceptance threshold. Under anchoring, PAC is computed over
the m-by-m anchor block rather than over all pairs, so it is a legitimately different
quantity from the exact PAC, and neither this plan nor the spec establishes how close the
two should be. At this n the observed anchored PAC happens to track the exact value
closely (absolute difference 0.004, 0.004, 0.000 at m = 500, 1000, 2000), but that
agreement is a measurement, not a claim this document makes or a bound the test enforces.

## 5. Plan 2: the two case studies

Both studies follow the existing pipeline. Each contributes a loader under
`src/benchmarks/datasets/`, an entry in `STUDIES`, a notebook under
`notebooks/case_studies/`, and figures built from the existing `composite_figure` and
`carve_output_figure`.

### 5.1 Study 3: Cusanovich mouse sci-ATAC atlas

Purpose: demonstrate that CARVE performs well on scATAC-seq, not merely that it runs. This is
the answer to R1.7.

Data. `atac_matrix.binary.qc_filtered.mtx.gz` at 1.1 GB, with its `.cells.txt` and
`.peaks.txt` companions and `cell_metadata.txt`, from the mouse-atac data release. The matrix
is peaks by cells, 436,206 by 81,173, with 421,971,103 nonzeros, approximately 3.4 GB as CSR.

Preprocessing, following the source publication's own `dim_reduction.R`:

1. Retain peaks accessible in at least 3% of cells. This is a site-frequency threshold applied
   to the full matrix, not a top-N selection.
2. TF-IDF, with term frequency as the per-cell column normalization and inverse document
   frequency as `log(1 + n_cells / cells_per_peak)`.
3. Truncated SVD to 50 components, taking the cell coordinates as `V` scaled by the singular
   values.
4. Retain all 50 components. The source does not drop the first component; dropping LSI
   component 1 is a later Signac and ArchR convention. Since the reference labels are outputs
   of the source pipeline, matching it is correct. The correlation between component 1 and
   sequencing depth is to be measured and reported rather than silently assumed away.
5. Stratified subsample to the study scale, `random_state=42`, stratified by the reference
   label. This is our own step, for runtime, with the same justification the Klein and Levine
   studies already record. Publication scale is 5,000 cells, matching Levine; development
   scale is 1,500, matching the order of Klein.

Deviation from source: none material. The subsample is additional, not a substitution.

Labels. Primary reference is `tissue`, 13 levels, because tissue of origin is externally
determined by dissection rather than being an output of anybody's clustering. This is the same
kind of independent ground truth as Klein's collection timepoints and is uncommon in ATAC.
Secondary reference is `cell_label`, 40 levels as published. The 10,029 cells labeled
`Unknown`, 12.4% of the atlas, are dropped, mirroring the Levine study's removal of the
uncharacterized population 15; this leaves 71,144 cells across 39 named labels. ARI is
reported against both references.

Candidate k: 4 to 16 inclusive, 13 values. The range is centered near the 13-tissue level and
deliberately stops well short of the 30 reported clusters and 40 reported cell labels.
Sweeping toward 40 would presuppose that the finest reported partition is the correct one,
whereas CARVE's claim is that the finest partition is not the most stable or generalizable.
Starting at 4 is low enough to observe CVIs collapsing toward coarse partitions, which is the
failure mode the Klein study documents. With two estimators this is 26 configurations, roughly
5.2 GB of consensus matrices at n = 5,000, comparable to the 4.4 GB the existing Levine study
already requires on this machine.

Estimators: KMeans as the study estimator, plus spectral clustering with self-tuning affinity,
via the existing `study_model_grids`, matching the two-estimator shape of Klein and Levine.
KMeans is chosen because clustering an LSI embedding with KMeans or a graph method is the
convention in scATAC-seq workflows, and because Ward agglomerative clustering is already the
represented choice in the Klein study. If the empirical result favors Ward, `STUDIES` is the
single place that changes; the estimator must not be re-derived at any call site.

Full-atlas pass. A second pass at all 81,173 cells with KMeans and Leiden on the anchored
path. This shows the conclusion survives beyond the subsample and pre-empts the objection that
the study subsamples again. Spectral and Ward are excluded from this pass for the reason given
in section 3.3.

### 5.2 Study 4: hECA v2.0 ATAC, large scale

Purpose: answer R1.1 with runtime, peak memory and empirical complexity on real single-cell
data of increasing size, culminating in a run above 500,000 cells.

Data. Per-organ h5ad files from Zenodo record 15627886. Five organs are pooled: Lung
(4.52 GB compressed), Brain (4.32 GB), Kidney (1.91 GB), Heart (1.21 GB) and Thymus
(0.44 GB), totaling 12.4 GB compressed. Lung at approximately 370,000 and Brain at
approximately 350,000 cells clear 500,000 between them; the remaining three are pooled so that
`organ` has five levels rather than two and is therefore usable as a reference label. Cell
counts other than Thymus's measured 36,361 are extrapolated from compressed archive size and
must be confirmed during implementation.

To bound peak disk usage, archives are retained but extracted one at a time and the extracted
h5ad is deleted after each pass over it, so peak disk is approximately 12.4 GB of archives
plus the largest single extracted organ, rather than all five extracted at once.

Preprocessing, following the hECA v2.0 publication rather than the Cusanovich chain, because
the two source pipelines differ:

1. Retain cPeaks open in at least 0.5% of cells.
2. Select the top 50,000 highly variable cPeaks by the Seurat method. See the recorded
   deviation below.
3. Log-normalization.
4. Principal component analysis to 50 components.
5. Drop cells labeled `Unclassified`.

Deviation from source, recorded in `meta` with its reason: the source selects 500,000 highly
variable cPeaks; we select 50,000, for memory. At 19 GB of RAM the reduced matrix must fit
alongside the SVD workspace.

Implementation of the loader is two chunked passes over the backed CSR so that the full peak
matrix is never resident: pass one accumulates per-peak accessibility and selects features
shared across organs; pass two builds the reduced cells-by-features CSR, approximately 5 GB at
500,000 cells. The result is cached as an AnnData carrying the embedding in `obsm` and
`obs[["cell_type", "organ", "study_id", "donor_id"]]`, so the expensive preparation runs once.

Labels. Primary reference is `organ`, which is clean and unambiguous. Secondary is
`cell_type`, harmonized across organs by uHAF. The harmonized annotation is coarse, averaging
roughly four types per organ, which is why this study carries the scalability argument and the
Cusanovich study carries the accuracy argument.

Candidate k: 3 to 15 inclusive, centered near the organ and major-lineage level, with
`consensus_anchors=2000` set explicitly in the `STUDIES` entry. That combination is
approximately 0.83 GB of consensus blocks rather than 5.2 GB.

Two CARVE runs, because `SweepSpec` is frozen and a single run sweeps exactly one parameter:

- KMeans and MiniBatchKMeans over `n_clusters`.
- `LeidenClustering` over `resolution`, swept from 0.1 to 2.0 in steps of 0.1, which is 20
  values. This is the R1.4 answer, and it is especially apt here
  because the Cusanovich atlas's own published clustering is Louvain via Seurat
  `FindClusters`, so the comparison is against the source method rather than a generic
  alternative.

Scaling ladder. Nested subsamples of the pooled set at n in {10k, 25k, 50k, 100k, 250k, 500k}
and the full pooled size. Nested subsampling is controlled: the biology is held constant and n
is isolated. Per-organ natural sizes are available as breadth but confound organ identity with
size, so they do not constitute the curve. Publication scale runs the full ladder; development
scale truncates it after 25,000, which exercises both the exact and the anchored code paths
while remaining cheap enough to iterate on.

Instrumentation. Per-configuration wall-clock and peak resident memory. The existing
`Manifest` dataclass already records `peak_rss_bytes` together with `peak_rss_unit`,
specifically because `resource.getrusage` reports bytes on macOS and kilobytes on Linux. Reuse
it rather than reintroducing that ambiguity.

### 5.3 Development scale and publication scale

Implementation proceeds against subsampled data so that the framework is verified cheaply, and
the final analysis is run at full scale once the code is verified.

Scale is part of the `Study` configuration, not a notebook variable. Configuration in this
package has repeatedly drifted by being re-derived at each call site rather than read from
`STUDIES`, and that pattern has produced manuscript mismatches before. Concretely:

- `Study` carries the scale mapping, and the loader is parameterized by the resolved scale
  rather than hardcoding a subsample.
- The resolved scale is recorded in the returned `meta`.
- The resolved scale is part of the `fit_or_load_carve` cache filename. Without this a
  full-scale run silently loads a development-scale fit, because `fit_or_load_carve` loads any
  cache file it finds at the given path. This is the specific defect this mechanism exists to
  prevent.
- Figures and tables derived from a run record which scale produced them, so a
  development-scale figure cannot be mistaken for a publication one. This mirrors the existing
  practice of recording `ACTIVE_ANCHOR_SET_NAME` in every run manifest.

### 5.4 New infrastructure

One helper, `study_scaling_sweep()` in `_studies.py`, mirroring the shape of `cvi_sweep`: it
loops over subsample sizes, fits CARVE at each, and returns a DataFrame of `(n, wall_clock_s,
peak_rss_bytes, selected_k, ari)`. It contains no matplotlib, consistent with the existing
separation between compute in `_studies.py` and rendering in `figures/`.

`run_scenario` cannot serve this purpose: it calls `simulate()` and is simulation-only. The
`Study` docstring in `_types.py` claims the runner treats a study as a single cell, but no such
code path exists; that claim should be corrected while this work is in the file.

### 5.5 Figures

Both studies use the existing `composite_figure` and `carve_output_figure`.

- Cusanovich: the LSI 1 and 2 coordinates for panels A through C, and an alluvial for panel F,
  the same shape as the Klein figure.
- hECA: panels A through C are drawn on a declared fixed 50,000-cell subsample, because
  500,000 points do not rasterize legibly, and panel F becomes the ARI comparison bar used by
  the Levine figure rather than an alluvial, which is unreadable at roughly 20 cell types. Both
  deviations are forced by the data and are recorded as such.
- One additional scaling figure for the hECA study: n against runtime, n against peak resident
  memory, and n against selected k, following the shape of the existing
  `figures/_scaling.py`.

### 5.6 Tests

Tests mirror modules one to one, as the project requires. New loaders under
`src/benchmarks/datasets/` get matching test modules; `study_scaling_sweep` is tested in the
module that already covers `_studies.py`. Tests that touch network resources or multi-gigabyte
archives are gated behind an environment variable in the manner of the existing regression
gate, so the default suite stays runnable.

Assertions must be able to fail. The specific risks here are a test that asserts a preprocessing
chain's output shape without asserting any value, and a scale-mechanism test whose development
and publication paths happen to produce the same cache path. Both are settled by mutation.

## 6. Risks

Disk. Cusanovich requires 1.1 GB. The hECA archives for Lung and Brain are approximately
8.8 GB compressed and roughly 15 GB extracted, against 37 GB free. This is workable only if
archives are deleted after extraction and the reduced embedding is cached rather than the full
h5ad files. Sequencing Cusanovich first limits the exposure.

Batch structure in hECA. The atlas pools ten studies; Thymus alone spans two `study_id`
values. A multi-organ pooled clustering may partly recover study of origin rather than
biology. Using `organ` as the primary reference makes the analysis defensible, and the
confound must be stated in the manuscript rather than omitted.

Organ cell counts are estimated. Lung and Brain cell counts are extrapolated from compressed
archive size calibrated against the measured Thymus file. If the true counts fall short, more
organs are pooled. This does not change the design, only the organ list.

The R mirror is deliberately broken. See section 2. This needs a tracked follow-up item, or it
will be forgotten after acceptance.

## 7. Manuscript deliverables

Figures flow from code to manuscript by manual re-export; regenerating a figure does not
update the paper. The expected additions are one main figure per case study, two supplementary
text sections mirroring the structure of S6 Text and S7 Text, an extension of S5 Text to cover
memory and empirical complexity alongside runtime, and a supplementary table reporting the
anchored-against-exact validation from section 4.5.

## 8. Decisions and rationale

| Decision | Rationale |
|---|---|
| Anchored consensus rather than a scaling envelope alone | A scalability criticism answered by documenting a ceiling concedes the point. The measured evidence in section 3.2 shows the ceiling is removable and that the resulting run is roughly one hour on eight cores. |
| Full fidelity rather than anchor-only diagnostics | CARVE's multi-level diagnostics are what distinguish it from SC3 and M3C, which R2.1 attacks directly. Losing them at scale would weaken the response to a different reviewer. |
| Threshold inclusive at 5,000 | Levine is exactly 5,000 cells. A strict comparison would move published numbers. |
| `consensus_anchors` defaults to `min(n, anchor_threshold)` | Avoids a discontinuity in effective anchor count at the threshold. |
| Python only, R port deferred | No reviewer asked the R package to scale. |
| Cusanovich for accuracy, hECA for scale | Cusanovich has externally determined tissue labels and a nested hierarchy; hECA's harmonized annotation is coarse but its size is what R1.1 requires. |
| CATlas rejected | 5.0 billion nonzeros, approximately 19 GB as CSR for the adult subset, against 19 GB of RAM. hECA incorporates the same data under harmonized annotation. |
| Candidate k stops well below the reported label count | Sweeping to 40 would presuppose the finest reported partition is correct. That the selected k falls well below it is a finding, not an artifact of the grid. |
| Preprocessing follows each source publication separately | The two sources differ: Cusanovich uses TF-IDF and SVD, hECA uses log-normalization and PCA. Following each keeps our partitions comparable to the reference labels they are scored against. |
