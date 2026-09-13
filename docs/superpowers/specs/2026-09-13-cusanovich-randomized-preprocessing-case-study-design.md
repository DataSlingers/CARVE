# Cusanovich case study on randomized preprocessing: design

Date: 2026-09-13
Status: approved for planning, after
`docs/superpowers/specs/2026-09-13-randomized-preprocessing-design.md` is implemented
Supersedes: section 5.1 of
`docs/superpowers/specs/2026-09-07-atac-and-large-scale-case-studies-design.md`
Related: `docs/superpowers/specs/2026-09-01-benchmarks-rebuild-design.md`

## 1. Motivation

The Cusanovich study as built scores CARVE against tissue of dissection. Tissue is
externally determined, which was the reason for choosing it, but cell types cut across
tissues (endothelial, immune and fibroblast populations appear in every organ), so no
clustering tracks it and an ARI against it says nothing about which clustering is better.

The source publication's own clustering is a better thing to argue with. Cusanovich et al.
(2018, Cell 174:1309) built an LSI (3 percent site filter, TF-IDF, 50-component SVD, all
components kept), ran `Rtsne(perplexity = 30, max_iter = 5000)` on it, and clustered in the
two-dimensional t-SNE space with Seurat's graph community detection to obtain 30 clusters;
they then repeated t-SNE and clustering within each cluster to obtain 85 subclusters and 40
marker-based cell labels. Their documentation states this directly: "In our manuscript, we
performed clustering in t-SNE space using an older version of Seurat"
(https://atlas.gs.washington.edu/mouse-atac/docs/; scripts at
https://github.com/shendurelab/mouse-atac, `dim_reduction/dim_reduction.R`). Clustering a
t-SNE is a known way to manufacture clusters: the embedding is stochastic, its islands
depend on seed and perplexity, and a continuum is routinely split into several islands.

CARVE with randomized preprocessing is built for exactly this question. The study is
rebuilt to ask it: run graph community detection over resolution on the LSI as-is, on a
t-SNE of it, and on a UMAP of it, under CARVE's resampling; select the configuration and
report which pipeline is stable and generalizable; compare the result with the source's 30
clusters on their own terms. The expected finding, stated so it can be falsified, is that
clustering in t-SNE space is less stable and less generalizable than clustering the LSI
directly at the same granularity, and that the source's 30-cluster partition sits past the
point where either criterion has started to fall. Prior measurement (2026-09-13) found KMeans
on the LSI reaching ARI 0.77 against the source's 30 clusters at k of 20 to 30, so their
major clusters are largely real; the honest thesis is over-partitioning plus embedding-induced
instability, not that the clusters are fictions, and the notebook reports whatever the data
show.

## 2. Decisions

Taken in the 2026-09-13 brainstorming session. Not to be relitigated in the plan.

- Reference partition y is the source's 30 clusters (`cluster` in `cell_metadata.txt`).
  Every comparison is against it and is reported as agreement, never as accuracy. Tissue and
  the 40 cell labels ride along in meta for coloring and side checks.
- The sweep is Leiden over resolution only, with randomized preprocessing over the LSI
  as-is, t-SNE and UMAP. Graph community detection is the source's own algorithm family, so
  their operating point has an honest position on the axis. KMeans, spectral, Ward and
  single linkage leave this study; the k-based run, the CVI sweep and the full-atlas pass are
  dropped from the notebook (section 6 says why each).
- One figure with four panels: (A) CARVE's consensus labels on the embedding of the pipeline
  CARVE rates best at the selected configuration; (B) the source's 30 clusters on the source's
  own t-SNE; (C) CARVE's stability and generalizability over resolution, pooled over
  pipelines, with the selected resolution, the source's operating point and the published
  partition's own generalizability marked; (D) the same two curves stratified by pipeline.
- The source's recipe is scored two ways: its operating point on the t-SNE-stratified curve
  (the resolution whose observed cluster count is nearest 30), and the published 30 labels'
  own generalizability under the same random-forest probe on the LSI. Both are reported.
- Every cell is kept. The source's 30 clusters partition all 81,173 cells, so dropping the
  10,029 cells whose `cell_label` is `Unknown` would compare CARVE against a partition of a
  different set. `n_cells_annotated` and the `Unknown` drop become optional loader behavior,
  off by default.

## 3. Dependencies and preconditions

- The randomized-preprocessing spec is implemented and merged first. This study needs
  `preprocessing_results_` with `method_id`, `pipeline`, `n_resamples`, the SE columns and
  `n_clusters_observed`; `preprocessing_pipelines_`; `_pipeline.pipeline_from_spec`; and
  `_plotting.plot_metric_by_pipeline`.
- The uncommitted Cusanovich cell-order fix in the working tree (loader reads
  `cell_metadata.txt` row order; fixture writes the cells file shuffled) is committed before
  either branch is cut. Every Cusanovich number produced before that fix is invalid, and the
  cached development fit `carve_cusanovich_dev_7841fb1f.carve` must be deleted, not reused.
- Data under `code/data/Cusanovich/{matrices,metadata}/` as already present (1.23 GB matrix,
  13 MB metadata).

## 4. Data and loader

`src/benchmarks/datasets/_cusanovich.py`, changes only:

- `label_column` defaults to `"cluster"`. y is the cluster id as a string, 30 levels.
- `drop_unknown: bool = False`. When True, the current behavior (drop cells whose
  `cell_label` is `Unknown`, report `n_cells_annotated`). When False, all cells are kept and
  `n_cells_annotated` is reported as the count of cells with a non-`Unknown` label for
  information only. `_cusanovich_loader` in `_studies.py` passes `label_column="cluster"`
  and leaves `drop_unknown` at its default.
- meta gains `source_labels`, a DataFrame aligned row for row with X carrying `tissue`,
  `cluster`, `subset_cluster`, `cell_label`. The existing `source_tsne` stays. The loader
  test that checks `source_tsne` follows X through filter and subsample is extended to
  `source_labels`.
- Stratification of the subsample is by y, hence by the 30 clusters. The smallest source
  cluster has 393 cells (0.5 percent), so publication scale keeps about 24 of them and
  development scale about 7; every cluster is present at both scales.
- The module docstring's reference-label paragraph is rewritten for the new reference.

The LSI itself is unchanged: the loader's preprocessing already matches the source's
`dim_reduction.R`, verified on 2026-09-13.

## 5. Study configuration

All of it in `STUDIES["cusanovich"]`, none re-derived at a call site.

| Field | Value | Reason |
|---|---|---|
| `estimator` | `EstimatorSpec("leiden")` | The source's algorithm family |
| `partners` | `()` | Nothing else is swept |
| `candidate_k` | `()` | No k-sweep; see the `Study` change below |
| `resolutions` | `0.1, 0.2, ..., 2.0` | The existing 20-point grid; verified at dev scale (section 8) |
| `scales` | `{"dev": 1500, "publication": 5000, "atlas": None}` | Unchanged; the atlas scale stays declared for the loader, the notebook does not run it |
| `default_scale` | `"dev"` | Unchanged |
| `consensus_anchors` | `2000` | Unchanged, same arithmetic as before |
| `n_resamples` | `150` | 50 resamples per pipeline under stratified allocation, against 33 at the package default; new field, default 100 |
| `preprocessing` | see below | New field |

`Study` gains two fields: `n_resamples: int = 100` and
`preprocessing: PreprocessingSpec | None = None`. `Study.__post_init__` relaxes
"candidate_k must not be empty" to "at least one of candidate_k and resolutions is
non-empty", and `study_model_grids` keeps raising for a resolution estimator as it does now.

`PreprocessingSpec` is a frozen dataclass in `_types.py` naming options by registry key, so
`_types` stays a leaf with no sklearn import:

```
PreprocessingSpec(
    normalization=(("identity", {}),),
    dim_reduction=(
        ("identity", {}),
        ("tsne", {"perplexity": [30]}),
        ("umap", {"n_neighbors": [15, 30]}),
    ),
)
```

A new module `src/benchmarks/_preprocessing.py`, the preprocessing counterpart of
`_estimators.py`, holds `PREPROCESSOR_CLASSES` (identity as `FunctionTransformer`, `pca`,
`tsne`, `umap`, `standard_scaler`), `PREPROCESSOR_DEFAULTS` (t-SNE `n_components=2`,
`max_iter` at sklearn's default, UMAP `n_components=2`, `min_dist=0.1`) and
`resolve_preprocessing(spec) -> dict` with keys `normalization_options` and
`dim_reduction_options`, in the option syntax `carve` accepts, with display names attached
so the `pipeline` labels read
`identity | identity`, `identity | TSNE(perplexity=30)`, `identity | UMAP(n_neighbors=15)`.
Unknown keys raise, as `EstimatorSpec` does.

Choices recorded with their reasons:

- Perplexity 30 is the source's value. Only that value is offered, so the t-SNE line in
  panel D is the source's recipe and not an average over perplexities.
- Normalization is identity only. The LSI is already scaled by its singular values, and
  standardizing or log-transforming it is not something the source or the field does.
- PCA truncation of the LSI (keeping 20 or 30 of the 50 components, the Signac and ArchR
  convention) is not offered. It would be a fourth pipeline at 37 resamples each and it is not
  the point of the study. It can be added to the spec later without code changes.
- Leiden `n_neighbors` stays at the package's 15 and modularity objective, the same as hECA.
  Seurat's default at the time was 30, but the claim is "graph community detection in
  t-SNE space", not a byte-for-byte replication of their Seurat call, and one convention
  across case studies is worth more than a guess at theirs.
- UMAP is included as the field's current default embedding, so the reader sees that the
  finding is about clustering in embedding space, not about t-SNE in particular. It needs the
  `umap` extra; the notebook says so.

`carve_cache_path` gains a run component. `fit_or_load_carve` loads whatever file sits at
the path it is handed, and today a k-based fit and a resolution fit at the same scale would
share a filename; adding preprocessing makes a third kind. The filename becomes
`carve_{study}_{scale}_{size_key}_{run_key}.carve` where `run_key` is a short hash of the
sweep parameter, the estimator grid, the resolved preprocessing spec (or none) and
`n_resamples`. For the default k-based, non-randomized run at the package's `n_resamples`
the component is omitted, so the existing Klein and Levine publication caches, hours of
compute each, keep their filenames and stay valid. `fit_or_load_carve` gains
`randomize_preprocessing`, `normalization_options` and `dim_reduction_options`, forwarded
only when set, the way `consensus_anchors` already is.

## 6. What leaves the study and why

- The CVI sweep. Silhouette, gap, Davies-Bouldin and Calinski-Harabasz are computed on one
  fixed representation and cannot be stratified by pipeline; they have nothing to say about
  the question this study now asks. Klein and Levine keep them.
- The k-based run with KMeans, spectral, Ward and single linkage. A second randomized run
  would double the preprocessing compute for a comparison the panels do not draw.
- The full-atlas pass. Randomized preprocessing at 81,173 cells means about 450 t-SNE fits on
  50,000-cell subsamples, which is not tractable, and an LSI-only atlas pass would be a
  different experiment from the one the figure shows. The `atlas` scale stays declared so
  the loader can still produce it. R1.1's scalability question is answered by hECA.
- `figure_carve_output_cusanovich` and `figure_cusanovich_results` as they exist, together
  with `CompositeInputs`-based assembly for this study. `prepare_composite` requires the
  CVI tables and a comparison estimator; neither exists here. The two functions, their
  constants and their tests are removed; `_carve_output.py` keeps Klein and Levine untouched.
  This is a deletion the plan should confirm with the author before executing.

## 7. Compute and figure

### 7.1 Compute, in `src/benchmarks/_studies.py` and a new `_cusanovich_compare.py`

- The CARVE fit: `fit_or_load_carve(X, y, cache_path=..., model_grids=
  study_resolution_grids(study), n_resamples=study.n_resamples,
  randomize_preprocessing=True, **resolve_preprocessing(study.preprocessing),
  consensus_anchors=study.consensus_anchors)`.
- `best_pipeline(carve, *, measure, rule, not_two) -> (row, PipelineSpec)`: the selected
  configuration via `carve._select_row`, then the `preprocessing_results_` row at that
  `method_id` and sweep value with the highest `measure`, and its spec from
  `preprocessing_pipelines_`. The notebook's `measure` and `rule` drive it, the same
  convention `prepare_composite` follows.
- `pipeline_embedding(X, spec, *, random_state) -> ndarray (n, 2)`: `pipeline_from_spec(spec,
  random_state).fit_transform(X)` on the full study X; its output when two-dimensional,
  else its first two columns (the LSI as-is gives LSI 1 and LSI 2). Axis labels come back
  with it so panel A names what it draws.
- `source_operating_point(carve, *, pipeline, target_k=30) -> (resolution, observed_k)`:
  from `preprocessing_results_` rows for the named pipeline, the resolution whose
  `n_clusters_observed` is nearest `target_k`. If the nearest is more than 25 percent off
  the target, the function warns; section 8 says what the plan does about it.
- `published_partition_generalizability(X, labels, *, n_splits, subsample_ratio, n_trees,
  random_state) -> (mean, se)`: repeated stratified splits at the study's subsample ratio, a
  random forest with `n_trees` trees fit on `X[train]` with the published labels, ARI between
  its predictions and the published labels on the held-out rows. This is the published
  partition under exactly the probe CARVE applies to its own clusterings, on the same
  features, and is the one number a reviewer will ask for. Stability of a fixed partition is
  not defined and is not reported.
- Tables: `preprocessing_results_` and the selection summary are written as CSV next to the
  figure so the manuscript's numbers have a file behind them.

### 7.2 Figure, `src/benchmarks/figures/_cusanovich_results.py` rewritten

`figure_cusanovich_results(inputs, *, save, out_dir)` over a new frozen
`CusanovichInputs` (X, y, carve, carve_labels aligned to y, embedding_A, embedding_A_labels,
source_tsne, best_pipeline_row, operating_point, published_generalizability, measure, rule,
not_two). Assembly is compute; drawing is reporting; the split lets the figure be tested
without fitting anything, the same principle as `CompositeInputs`.

Layout, two by two, panel letters via `panel_letter`:

- (A) `scatter_clusters` of `carve_labels` on `embedding_A`, marker size 8, corner axis
  arrows labeled by the pipeline that produced the embedding. Title names the selected
  estimator and resolution, the observed cluster count, and the pipeline set the consensus
  pools over, for example "CARVE: Leiden, resolution 0.6, 14 clusters; consensus over LSI,
  t-SNE, UMAP; shown on LSI 1/2".
- (B) `scatter_clusters` of y on `source_tsne`, same marker size, arrows "t-SNE 1", "t-SNE
  2". Title "Cusanovich et al.: Louvain on t-SNE, 30 clusters". A and B share the color map
  `aligned_color_maps(y, carve_labels)` builds, so a CARVE cluster takes the color of the
  source cluster it best matches; with 30 classes the palette cycles, which
  `cluster_colors` documents and which is unavoidable at this class count.
- (C) `carve_lines` generalized to read the sweep axis from `carve.sweep_.param`, so its x
  is resolution here and k elsewhere, with the selected resolution as a vertical line per
  measure as now. Added for this study: a marker at the source's operating point on the
  stability and generalizability curves, labeled with its observed cluster count, and a
  horizontal dashed line at `published_generalizability` labeled "published 30 clusters,
  RF probe". A secondary x-axis on top shows the mean observed cluster count at each
  resolution from `estimator_results_["n_clusters_observed"]`.
- (D) a thin `pipeline_lines(ax, carve, *, measures)` primitive in `_panels.py` that calls
  `carve._plotting.plot_metric_by_pipeline` once per measure on the same axes, with
  `method_id` resolved to the selected configuration and the palette taken from a
  `PIPELINE_COLORS` entry added to `_theme` (no color literals outside the theme);
  stability solid and generalizability dashed, error bars at one standard error as that
  plot draws them, legend naming the pipelines, then `style_axes`. Same x-axis as C. The drawing lives in
  `carve` (randomized-preprocessing spec, section 5.5); this panel only themes it.

Saved as `cusanovich_results.png` under the scale-qualified output directory, the same rule
the notebook already applies so a development figure cannot be mistaken for the manuscript's.

### 7.3 Notebook, `notebooks/case_studies/Cusanovich.ipynb` rewritten

Sections: 0 configuration (reads everything from `STUDIES["cusanovich"]`); 1 the source's
partition on its t-SNE, with the cluster count and the `Unknown` fraction reported for
information; 2 the CARVE fit; 3 the comparison (best pipeline, operating point, published
partition generalizability, the figure, the CSVs); 4 summary. The runtime note gives the
publication-scale estimate from section 8 once measured. Markdown cells state findings and
move on, no emphasis formatting.

`tests/benchmarks/test_notebooks.py` keeps the structural checks: the notebook reads
`STUDIES["cusanovich"]`, does not import forbidden modules, and no longer references
`cvi_sweep`, `study_model_grids`, `figure_carve_output_cusanovich` or `prepare_composite`.

## 8. Verification

Tests mirror modules one to one; every one is settled by the mutation that turns it red.

- Loader: the planted-cluster fixture gains a `cluster` column; `label_column="cluster"`
  returns it; `drop_unknown=False` returns every cell and `drop_unknown=True` returns the
  current count; `source_labels` follows X through filter and subsample (a shuffled fixture
  turns a positional bug red, as the `source_tsne` test already does).
- `_preprocessing`: every registry key resolves to an option `carve` accepts; an unknown key
  raises; the labels render as section 5 says.
- `_studies`: `Study` accepts empty `candidate_k` with non-empty `resolutions` and rejects
  both empty; `carve_cache_path` for Klein's and Levine's default runs is byte-identical to
  today's filename; the same study with `randomize_preprocessing` or a resolution grid gets
  a different filename; `fit_or_load_carve` forwards the preprocessing arguments (a spy
  `CARVE` records them).
- `_cusanovich_compare`: `best_pipeline` returns the row with the highest measure at the
  selected configuration and not the highest overall (fixture with a decoy row at another
  resolution); `pipeline_embedding` returns the first two columns for a 50-dimensional
  identity pipeline and the full output for a 2-D one; `source_operating_point` picks the
  nearest count and warns past 25 percent; `published_partition_generalizability` returns 1.0
  for a partition that is a deterministic function of X and about 0 for a random relabeling
  of it.
- Figure: the contract test (`test_figure_contract.py`) covers the new function; a synthetic
  `CusanovichInputs` renders four axes with the expected titles, C carries the operating-point
  marker and the horizontal line, D carries one line pair per pipeline; `carve_lines` on a
  resolution-mode fit labels its x-axis "Resolution" and on a k-mode fit still says "Number of
  clusters k".
- Development-scale run, executed once in the plan and recorded in this section: the
  resolution grid is checked against the t-SNE pipeline's observed cluster counts. If no
  resolution in 0.1 to 2.0 yields a count within 25 percent of 30 on the t-SNE pipeline at
  1,500 cells, the grid's upper end is extended in `STUDIES` (the one place it lives) until
  it does, and the reason is recorded here. Wall-clock and peak memory at dev scale are
  recorded here and extrapolated to publication scale for the notebook's runtime note.

Estimated publication-scale cost, to be replaced by measurement: 150 resamples x 3 subsets
= 450 pipeline fits, of which 150 are t-SNE (100 at 3,090 cells, 50 at 1,910), 150 UMAP
and 150 identity, so on the order of 30 to 60 minutes of t-SNE on one core, parallel over the core
budget; then 20 configurations x 150 resamples of Leiden on a 15-NN graph plus a
100-tree forest on 3,090 by 50, on the order of one to two hours on one core. Embeddings held
in memory: about 8 x 50 x 8,090 x 50 bytes, 160 MB, for the identity pipeline's 50 resamples and
negligible for the 2-D ones.

## 9. Manuscript text to carry over

`overleaf/` is read-only here; these are drafts for the author. Plain prose, no emphasis.

S1 Text, replacing the sentence "The pipeline T(b) is fit on X yielding a preprocessed
representation X~(b) = T(b)(X)": "The pipeline T(b) is fit separately on each subsample,
giving X~1(b) = T(b)(X[P1(b)]), X~2(b) = T(b)(X[P2(b)]) and X~test(b) = T(b)(X[Ptest(b)]);
an embedding that does not reproduce across independent fits therefore lowers both
criteria." And, in the generalizability sentence, "training a classifier h(b) on
(X[P1(b)], C1(b))" in place of "(X~(b)[P1(b)], C1(b))", with the note that the classifier
sees the original features so that a cluster present only in an embedding and not learnable
from the data does not generalize. Under identity preprocessing both statements reduce to the
current text.

Output paragraph: `pipeline_df_` becomes `preprocessing_results_`, "metrics stratified by
preprocessing pipeline and configuration".

Case-study section, to be written once the numbers exist: the source's recipe, the pipeline
set, the selected configuration, the per-pipeline finding from panel D, the operating point
and the published partition's generalizability from panel C, and the agreement between
CARVE's consensus and the 30 clusters. Figures flow code to manuscript by manual re-export.
