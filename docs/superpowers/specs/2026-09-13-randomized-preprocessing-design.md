# Randomized preprocessing: design

Date: 2026-09-13
Status: approved for planning
Supersedes: nothing
Related: `docs/superpowers/specs/2026-09-13-cusanovich-randomized-preprocessing-case-study-design.md`
(the case study this feature exists to serve; plan that one after this one)

## 1. Motivation

CARVE's `fit(randomize_preprocessing=True)` samples a normalization and a dimensionality
reduction per resample so that stability and generalizability reflect preprocessing choice
as well as sampling. The idea comes from Chang, Tang, Zikry and Allen (2025), Unsupervised
Machine Learning for Scientific Discovery: Workflow and Best Practices (arXiv 2506.04553),
whose Algorithm 1 draws a preprocessing pipeline per subsample and fits it on that subsample.

The feature shipped in 1.0 without being exercised by any manuscript result, and it does not
do what either the manuscript or the source idea describes. Found on 2026-09-13, reading
`src/carve/_pipeline.py`, `_runner.py`, `_grids.py` and `_utils.py`:

1. The sampled pipeline is fit on the full X and the embedding is sliced for P1, P2 and
   P_test. Within a resample, both subsamples are clustered on the same t-SNE, so t-SNE's
   own run-to-run variation never reaches the stability or generalizability ARI. Only the
   consensus matrix, pooled across resamples, sees it. This is the opposite of what the
   feature is for.
2. The pipeline is rebuilt and refit inside `validation_iter`, once per configuration per
   resample. A 20-configuration, 100-resample run fits t-SNE 2,000 times on the full data.
3. Sampled transformers are instantiated as `cls(**chosen_params)` with no `random_state`.
   t-SNE and UMAP draws are unseeded, so a randomized fit is not reproducible. The existing
   reproducibility test passes only because it uses PCA on data small enough for the exact
   solver.
4. `summarize_preprocessing_records` groups on the transformer's class name only. PCA with
   2 components and PCA with 49 components pool into one row; two t-SNE options with
   different perplexity ranges are indistinguishable. Rows also pool across estimators.
5. The default grids draw `n_components` uniformly from 2 to p-1 and perplexity from 5 to
   50, so every resample sees a different, mostly unreasonable, pipeline. The default
   normalization grid applies `log1p` unconditionally, which yields NaN on any input with
   values below -1 (an LSI, for instance).
6. Each pipeline option is drawn independently per resample, so the number of resamples a
   given pipeline receives is a random variable, and a per-pipeline rating is noisier than it
   needs to be.

## 2. Decisions

Taken in the 2026-09-13 brainstorming session. Not to be relitigated in the plan.

Fit scope. A pipeline is fit independently on each subsample: T1 on X[P1], T2 on X[P2],
T_test on X[P_test]. This is what Chang et al. do and what exposes an embedding's own
irreproducibility.

One spec per resample. One (normalization, dimensionality reduction, hyperparameters) spec
is drawn per resample and refit on each subsample. Chang et al. draw two independent
pipelines per iteration; that was considered and rejected because it makes every score
attributable only to a pair, and the case study needs a rating per pipeline. Cross-pipeline
disagreement still reaches the consensus matrix, PAC, the Gini and CE scores, and
`get_labels`, because those pool over resamples that used different pipelines.

Classifier features. The generalizability classifier trains on the raw features X[P1] with
labels C1 and predicts on raw X[P_test]; its predictions are compared with
C_test = f(T_test(X[P_test])). This is the only definition that works for transformers
without a `transform` method, t-SNE among them, and it has the right meaning: a cluster fails
to generalize if it cannot be learned from the data (an embedding artifact) or if an
independent embedding of new data does not reproduce it. Under identity preprocessing this is
exactly the current algorithm, so no published number moves. `_extend_anchor_labels` already
trains on raw `X_`, so the anchored path is consistent with it. A user-supplied classifier
that is not invariant to feature scaling now sees unnormalized features; this is documented,
not engineered around, because the default random forest is invariant.

Generalizability stays on. Restricting preprocessing assessment to stability, as Chang et al.
do, was considered. It would force `mode="stability"` under randomization and drop the more
telling axis for embedding artifacts. Rejected.

Stratified allocation. Each pipeline option is used in floor(B / n_options) or
ceil(B / n_options) resamples, in a seeded random order, with hyperparameters drawn per
resample (section 4.2). Chang et al. draw independently; equal counts behind every
per-pipeline curve were preferred, and the table reports the count either way.
Confirmed 2026-09-13.

Plotting. CARVE gets a per-pipeline companion to `plot_metric_over_n_clusters` at every
level the existing plots have: module function, `CARVE` method, and `pl` wrapper
(section 5.5). The case study's panel D calls it rather than drawing its own.

R port deferred. `carve-r/R/pipeline.R` and `runner.R` keep the old semantics for now. This
breaks the CLAUDE.md one-to-one mirror in the same way anchored consensus did and is
recorded as a tracked follow-up in section 8, not left implicit.

## 3. Scope

In scope, all in `src/carve/`:

- `_pipeline.py`: pipeline allocation, instantiation with seeding, a rebuildable pipeline
  spec.
- `_runner.py`: a precompute pass over resamples, `validation_iter` taking embeddings instead
  of building pipelines, classifier on raw features.
- `_grids.py`: discrete default grids; `log1p` guarded on the input's range.
- `_utils.py`: the per-pipeline summary with hyperparameters, counts, standard errors and
  the estimator join keys.
- `api.py`: `preprocessing_results_` schema, a new `preprocessing_pipelines_` attribute,
  docstrings, the S1 Text discrepancy noted in Notes.
- `_output.py`: the run header names the pipeline set when randomized.
- `_plotting.py`, `api.py`, `pl/_plots.py`, `tl/_carve.py`, `_anndata.py`: the
  per-pipeline plot and the `uns` storage it reads from (section 5.5).
- Tests mirroring each module, plus the regression gate in section 7.

Out of scope:

- The R port (section 8).
- Chang-style independent draws per subsample. Could be added later as a switch; nothing in
  this design precludes it.
- Feature selection as a third pipeline stage. The option lists stay normalization and
  dimensionality reduction.
- Per-pipeline consensus matrices. The consensus stays pooled over pipelines by design; the
  per-pipeline view is the ARI table.
- Sparse-native preprocessing. Input is densified by `ensure_2d_array` as before.
- Moving `randomize_preprocessing` from a `fit` keyword to the constructor. The call shape
  stays as documented in the manuscript.

## 4. Data flow

### 4.1 Non-randomized path

Unchanged in outcome, cheaper in mechanics. `validation_iter` currently builds an identity
`Pipeline` and calls `fit_transform(X)` on the full matrix per call, which copies X once per
configuration per resample. Under this design the non-randomized path passes raw slices
X[P1], X[P2], X[P_test] straight to the clusterer. No pipeline object is built. The
regression gate in section 7 establishes that every number is unchanged.

### 4.2 Randomized path

One allocation, then one precompute pass, then the existing configuration loop.

Allocation. Before any configuration runs, `allocate_pipelines(normalization_options,
dim_reduction_options, n_resamples, random_state)` returns a list of length `n_resamples`
whose entry b is the `PipelineSpec` resample b will use. Allocation is stratified: the
Cartesian product of normalization options and dimensionality-reduction options (options,
not hyperparameter values) is cycled so that every combination receives either
floor(B / n_combinations) or ceil(B / n_combinations) resamples, then the order is shuffled
with a generator seeded from `random_state`. Hyperparameters within an option are drawn
uniformly from that option's candidate lists, per resample, from the same generator. The
allocation depends only on the option lists, `n_resamples` and `random_state`, so it is
identical across configurations and across runs with the same seed, which is what makes two
runs on the same data (a k-sweep and a resolution sweep, say) paired on identical subsamples
and identical pipelines.

Precompute. A joblib pass over b in `range(n_resamples)`, using the same `outer_n_jobs`
budget `resolve_core_budget` already computes, fits the allocated spec on each subset the
mode needs and returns the embeddings: X[P1] always, X[P2] only when `policy.run_stability`,
X[P_test] only when `policy.run_generalizability`. Subsample indices come from
`split_subsample_indices` with the seeds `validation_iter` already uses, so the precompute
and the configuration loop see the same P1, P2 and P_test by construction, not by passing
them around. A tqdm bar labeled for preprocessing is shown under `show_progress`, before the
configuration bar.

Configuration loop. `validation_iter` gains an `embeddings` argument carrying the three
arrays (or None where the mode skips a subset) and the resolved spec for the record. It
clusters the embeddings, applies the noise policy to labels and index arrays exactly as now,
and hands raw X[P1_idx] and X[P_test_idx] (post-policy) to the classifier. Because the
classifier reads raw slices by global index, the noise policy's index shrinkage needs no
positional re-slicing of the embeddings: after clustering, the embeddings are not used again.
The joblib generator passes each worker only its own resample's arrays, so per-call IPC is one
resample's worth of embedding, not the whole set.

Memory. All embeddings are held in the parent for the duration of the fit. The size is
8 bytes x B x (|P1| + |P2| + |P_test|) x d, where d is the embedded dimension of each
pipeline. At n = 5,000, B = 100, subsample ratio 0.618 and a 50-dimensional identity
pipeline this is about 320 MB; 2-D embeddings are negligible. The `fit` docstring states
the formula so a user at larger n can size B and the option lists. Randomized preprocessing is
a case-study-scale feature; at atlas scale (tens of thousands of cells) 300 t-SNE fits are
infeasible regardless of memory, and the case-study design says so rather than pretending
otherwise.

### 4.3 Seeding

No global seeding, arithmetic derivation only, as everywhere else in the package. Per
resample b with base seed s0 = `random_state` or 0:

- subsample P1 and P_test: `s0 + b` (unchanged)
- subsample P2: `s0 + b + n_resamples` (unchanged)
- transformer fit on P1: `random_state = s0 + b`
- transformer fit on P2: `random_state = s0 + b + n_resamples`
- transformer fit on P_test: `random_state = s0 + b + 2 * n_resamples`

A transformer receives `random_state` only when its `__init__` signature accepts it, probed
the way `benchmarks._estimators.apply_random_state` already does. The allocation generator
is `np.random.default_rng(s0)`; the stdlib `random.Random` currently used is dropped so the
module uses the same RNG family as the rest of the package.

### 4.4 The pipeline spec

`_pipeline.PipelineSpec` is a frozen dataclass: `normalization` and `dim_reduction`, each a
`(cls, params, name)` triple with the sampled hyperparameter values, plus a `label` property
rendering, for example, `identity | TSNE(perplexity=30)`. Identity steps render as
`identity`; a `FunctionTransformer` with a `func` renders as the function's `__name__`, as
the current summary does. `pipeline_from_spec(spec, random_state)` builds the sklearn `Pipeline`
with seeding applied; it is what the precompute pass calls and what the case study calls to
draw the winning pipeline on the full data. Both are private to `carve` but importable by
`benchmarks`, which already reaches into `_select_row`.

The existing option syntax is kept in full: `(cls, params)`, `(cls, name, params)` and the
dict form, with the same errors for wrong arity or missing class.

## 5. Public surface

### 5.1 `preprocessing_results_`

One row per (estimator configuration, pipeline, sweep value), so the table mirrors the row
identity of `estimator_results_` and joins to it on `method_id` and the sweep column. Rows
are sorted by `method_id`, `pipeline`, sweep value. Columns:

| Column | Meaning |
|---|---|
| `method_id`, `method_label` | Same join keys `estimator_results_` carries |
| `pipeline` | `PipelineSpec.label`; the key into `preprocessing_pipelines_` |
| `normalization`, `dim_reduction` | Step names, hyperparameters rendered, for filtering |
| sweep column (`n_clusters` or `resolution`) | The swept value |
| `n_resamples` | Resamples this pipeline received at this configuration |
| `ari_stability`, `ari_stability_se` | Mean and standard error over those resamples |
| `ari_generalizability`, `ari_generalizability_se` | Same |
| `n_clusters_observed` | Mean observed cluster count on P1 over those resamples; equals the sweep value under a k-sweep except where a resample warned about a degenerate count |

Under `mode="stability"` the generalizability columns are NaN, and the reverse, matching
`estimator_results_`. This replaces the current `norm__func`, `dr__method` columns, a
breaking change to a documented attribute; it is accepted because the attribute was not
usable as shipped (points 3 and 4 in section 1). The manuscript calls this table
`pipeline_df_`, which has never been its name; section 8 lists that correction.

### 5.2 `preprocessing_pipelines_`

`dict[str, PipelineSpec]`, keyed by `pipeline` label, populated alongside
`preprocessing_results_` and None otherwise. This is what lets a caller rebuild and refit a
rated pipeline without re-deriving it from strings. It round-trips through `save` and `load`
because both pickle the whole object.

### 5.3 Defaults

`default_normalization_options(X)` now takes X and returns identity, `StandardScaler`, and
`log1p` only when `X.min() >= 0`. A warning names the omission when it happens.

`default_dim_reduction_options(X, subsample_ratio)` returns discrete grids, each filtered to
what the smallest subset can support (n_min = the P_test size, computed as now):

- identity
- PCA, `n_components` in {2, 5, 10, 20, 50} filtered to values below `min(n_min, p)`
- t-SNE, `n_components` 2, `perplexity` in {15, 30, 50} filtered to values below n_min
- UMAP (when installed, with the existing warning otherwise), `n_components` 2,
  `n_neighbors` in {15, 30} filtered to values below n_min, `min_dist` 0.1

An option whose filtered grid is empty is omitted with a warning rather than instantiated
with nothing.

### 5.4 Docstrings and output

`CARVE.fit`'s `randomize_preprocessing` entry states the per-subsample fit, the raw-feature
classifier, the memory formula, and that a user classifier sees unnormalized features. The
class Notes point out that the submitted S1 Text describes the old fit scope.
`_print_run_header` lists the pipeline option names when randomized. Warnings, not logging.

### 5.5 Plotting

`_plotting.plot_metric_by_pipeline` largely mirrors `plot_metric_over_n_clusters`, the
plot CARVE already has: the same signature shape and keyword names (`measure`, `rule`,
`not_two`, `ax`, `figsize`, `title`, `xlabel`, `ylabel`, `legend`, `legend_loc`,
`palette`, `show`, `save`, `dpi`, `**kwargs` forwarded to `ax.errorbar`), the same
drawing (one `errorbar` line per group at plus or minus one standard error, colors from
`plt.get_cmap(palette)`, a vertical dashed line at the sweep value the rule selects, the
x-axis label derived from the sweep column, `n_clusters` or `resolution`, the y-axis
label from the measure, legend, save-or-return), and the same errors on an empty table.
It differs in exactly two things: it reads `preprocessing_results_` rather than
`estimator_results_`, and it groups on `pipeline` within one estimator configuration
rather than on `method_id` across them. It takes that configuration as `method_id`; the
`CARVE` method resolves the default through `_select_row(measure, rule, not_two)` so the
plot with no arguments shows the pipelines behind the configuration CARVE selected. The
drawing code should be shared with `plot_metric_over_n_clusters` through a private
helper parameterized on the grouping column rather than copied, so the two plots cannot
drift apart. The function raises a clear error when the table is None, that is, when the
fit was not randomized.

`CARVE.plot_metric_by_pipeline(...)` forwards to it with `self.preprocessing_results_`.

`pl.metric_by_pipeline(adata, *, key="carve", ...)` reads the table from
`adata.uns[key]["preprocessing_results"]`, which `tl.carve` stores alongside
`"results"` when the fit was randomized and `store_results` is True, sanitized through
the same `results_to_uns` path so it survives an h5ad round trip, and read back through
`results_from_uns`. The `pl` function raises the same kind of `KeyError` the other
wrappers raise when the entry is absent, naming the two reasons it can be: the fit was
not randomized, or `store_results=False`.

## 6. Invariants

- `config_id` is a join key. The precompute pass produces nothing per configuration, so it
  cannot disturb the alignment `fit` asserts.
- Seeds are derived arithmetically (section 4.3). No shared RNG state crosses a joblib
  boundary.
- Selection rules operate on `sweep_rank`. The per-pipeline table carries the sweep column
  for reporting only; selection reads `estimator_results_` as before.
- Noise policy unchanged. Under `"drop"` the index arrays shrink and the classifier reads
  raw slices by the shrunken indices.
- Parallelism stays joblib over resamples inside a serial loop over configurations. The
  precompute pass is a second joblib-over-resamples stage before that loop, under the same
  core budget.
- No `logging`.

## 7. Verification

Every test must be able to fail; each is settled by naming the mutation that turns it red.

Regression gate, non-randomized path. Fit a fixed dataset with a fixed seed before and after
the change and assert `estimator_results_`, the consensus matrices and the per-sample score
arrays are equal to within floating-point tolerance. The before-values are captured into a
fixture at the start of the plan, from the commit the branch starts from. Mutation: passing
`randomize_preprocessing=True` in the after-fit must turn it red.

Per-subsample fit. A recording transformer stores the row count of every `fit`. After a
randomized fit with it as the only option, the recorded counts are exactly the sizes of P1,
P2 and P_test, and n itself never appears. Mutation: fitting on X once per resample turns it
red.

Once per resample, not per configuration. The same recorder, over a two-configuration grid,
reports `n_resamples x n_subsets` fits, not twice that.

Classifier on raw features. A transformer that maps every row to a constant vector makes the
embedding useless for prediction. With it as the only option, generalizability must still be
finite, and a spy classifier must report `n_features_in_ == p`. Mutation: training the
classifier on the embedding turns the second assertion red.

Seeding. With a transformer that draws from its `random_state` (PCA with
`svd_solver="randomized"` on data wide enough to make the solver stochastic, or a custom
transformer), two fits with the same seed give equal `preprocessing_results_` and two fits
with different seeds do not. The current test only has the first half, which cannot fail
for an unseeded transformer that happens to be deterministic.

Stratified allocation. Over the option product, per-combination counts differ by at most
one, and the allocation for a given seed is identical across two calls.

Summary schema. PCA with 2 components and PCA with 3 components appear as distinct rows;
two estimators in the grid appear as distinct `method_id` rows; `n_resamples` sums to the
run's `n_resamples` within each (configuration, sweep value); each row's SE is computed from
that row's resamples only.

Defaults. `default_normalization_options` omits `log1p` for an X with negatives and includes
it otherwise. `default_dim_reduction_options` filters perplexity and `n_neighbors` by the
P_test size and drops an option whose grid empties.

End to end with t-SNE. One randomized fit at n around 80 with `TSNE(perplexity=15)` among the
options and two KMeans configurations completes, populates both tables, and `get_labels`
returns a full-length label vector. This is the smoke test for the transformer-without-
`transform` path that the case study relies on.

Plotting. `plot_metric_by_pipeline` on a synthetic table draws one line per pipeline and no
line for a pipeline that only appears under another `method_id`; the `CARVE` method
defaults to the selected configuration (a decoy configuration with different pipelines
turns it red); `pl.metric_by_pipeline` reads the table back after an h5ad round trip and
raises with the two-reason message when it is absent. Mirrors go in `tests/test_plotting.py`,
`tests/test_api.py`, `tests/test_pl.py`, `tests/test_tl.py` and `tests/test_anndata.py` as the existing plots do.

Existing tests in `tests/test_pipeline.py`, `tests/test_runner.py`, `tests/test_api.py` and
`tests/test_utils.py` that pin the old return shapes are updated to the new ones, not
deleted; the ones that pin `norm__func` and `dr__method` are rewritten against section 5.1.
`filterwarnings = error` is on, so the new warnings in section 5.3 need `match=` assertions
where they fire.

## 8. Follow-ups this design creates

Tracked, not implicit:

- R port. `carve-r/R/pipeline.R` and `runner.R` still fit the pipeline on the full X and
  train the classifier on the embedding, and `plotting.R` has no per-pipeline plot. Port
  sections 4 and 5 to R; `r-ci.yml` checks it separately. Same status as the deferred
  anchored-consensus port.
- Manuscript, to be edited by the author in Overleaf; `overleaf/` is read-only here. S1 Text:
  "The pipeline T(b) is fit on X yielding X~(b) = T(b)(X)" becomes a fit per subsample, and
  the classifier trains on X[P1] rather than X~(b)[P1]. Output paragraph: `pipeline_df_` is
  `preprocessing_results_`. S1 Table: the default preprocessing grids, if listed, become the
  ones in section 5.3. Suggested replacement text is drafted in the case-study spec, section
  9, so both edits land together.
- Chang-style independent draws per subsample, as an option, if a pooled cross-pipeline
  stability is ever wanted on the ARI axis rather than the consensus axis.
