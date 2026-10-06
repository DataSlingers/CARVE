# Noise labels for ambiguous samples in get_labels: design

Date: 2026-10-06
Status: awaiting author review, then planning with /superpowers:writing-plans
Supersedes: nothing
Related: nothing

## 1. Motivation

CARVE already measures how ambiguous each sample's cluster membership is, through per-sample Gini
and cross-entropy (CE) stability and per-sample out-of-sample accuracy. The scatter plots encode
those scores as size and opacity, but `CARVE.get_labels()` always assigns every sample to a
cluster. This design lets the caller leave the least stable samples unlabeled, marked `-1`, the
label scikit-learn and HDBSCAN use for noise.

Python only. The R port is a later piece of work.

## 2. What already exists

Read on 2026-10-06 at `5ed30fc`.

- `stability_gini_scores_` and `stability_ce_scores_` are arrays of shape (n_configs, n_samples);
  `generalizability_scores_` is a list with one array of length n_samples per configuration, or
  None per entry for a run that skipped generalizability. All three are indexed by `config_id`.
- All three are on [0, 1], higher meaning more stable or more accurate
  (`_consensus.stability_from_consensus`, `_accuracy.compute_generalizability_scores`).
- Gini and CE are NaN for a sample that was never co-sampled with any partner
  (`_consensus._row_nanmean`). This happens at small `n_resamples`, and under HDBSCAN with
  `noise_policy="drop"` for a sample dropped as noise in every resample.
- Accuracy is 0, not NaN, for a sample that was never held out.
- On an anchored run, Gini and CE are still computed for all n samples
  (`_consensus.stability_from_runs_anchored`), and `get_labels` extends the anchor cut to all n
  samples (`CARVE._extend_anchor_labels`). Scores and labels therefore have the same length on
  both paths.
- `get_labels` stores its labels in `self.reference_labels` when the cluster count differs from
  the stored reference, and otherwise aligns its labels to the reference
  (`_utils.align_cluster_labels`). This keeps cluster numbers stable from call to call.
- `plot_cluster_boxplot`, `plot_cluster_violin`, `plot_cluster_scatter` and
  `plot_diagnostic_scatter` each contain the same lookup from `source` in {"gini", "ce",
  "accuracy"} to the per-sample array at `config_id`, with the same three `RuntimeError` messages
  and the same `ValueError("source must be one of: 'accuracy', 'gini', 'ce'.")`. They differ only
  in the axis label each assigns.
- Every call to `get_labels` inside the code base uses the defaults for the new arguments: the
  five plotting methods in `api.py`, `tl.carve` and `tl.attach_results` (`tl/_carve.py`), and the
  benchmarks (`_run`, `_ablation`, `_case_study`, `_heca_report`, `_cusanovich_compare`).

## 3. Decisions

Taken in the 2026-10-06 brainstorming session.

Names. Three keyword-only arguments: `noise_labels: bool = False`, `noise_quantile: float = 0.05`,
`noise_score: Literal["gini", "ce", "accuracy"] = "gini"`. The author's `noise_threshold` was
renamed to `noise_quantile` because the value is a fraction of samples, not a score cutoff. The
score values match the plotting methods' `source=`. Collapsing the flag into
`noise_quantile=None` was rejected: it hides the 0.05 default from the signature.

Cutoff. Strict comparison. Samples tied with the quantile value stay labeled. A `<=` rule would
label every sample as noise when 95 percent or more tie at the top (accuracy at 1.0 is the common
case), and it would label every sample when all scores are equal.

Guard. A per-sample margin rather than a whole-dataset spread check. A sample is noise only if it
is in the lowest quantile and also scores at least the margin below the median sample. A spread
check (max minus min below a tolerance) was rejected because one low outlier makes the range wide,
after which the check passes and about 5 percent of near-perfect samples are flagged. Raising an
error was also rejected.

Margin. A fixed module constant of 0.05 on the [0, 1] score scale, not an argument. For accuracy at
B=100 (about 38 held-out evaluations per sample), a sample needs about two misses to fall 0.05
below a median of 1.0. The quantile is the only setting users tune.

NaN scores. A sample with a NaN score is noise, and it is left out of the quantile and the median.
Without a score there is no evidence that its membership is stable.

Zero accuracy. Never-evaluated samples keep their accuracy of 0 and can be flagged. Changing it to
NaN would change `accuracy_generalizability` in `estimator_results_`. The docstring states the
caveat.

Warning. One `UserWarning` whenever fewer samples are flagged than the quantile asks for, whether
because of ties at the cutoff, the margin, or both. This refines the in-session wording, which
named only the margin. The refinement makes the all-scores-equal case warn instead of returning
no noise silently.

Shared score lookup. The four copies of the source-to-scores lookup move into one private method
that `get_labels` also uses, so there is no fifth copy.

## 4. Behavior

### 4.1 Signature

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

With `noise_labels=False` the result is identical to the current result, element for element.

### 4.2 The rule

After the existing flow has run up to and including the alignment and the `reference_labels`
update:

1. `scores = self._sample_scores(noise_score, config_id)`, where `config_id` is the value
   `_select_row` returned for this call. It is a join key, never a position. The scores belong to
   the selected configuration and do not depend on `consensus_k` or `estimator`.
2. Over the finite scores, `cutoff = np.quantile(finite, noise_quantile)` (the default linear
   method) and `median = np.median(finite)`.
3. A sample is noise if its score is NaN, or if `score < cutoff` and `score < median - 0.05`.
4. Noise samples are set to `-1` in a copy of the labels. The labels stored in
   `reference_labels` never contain `-1`, so calling with and without noise gives the same
   cluster numbers for every non-noise sample.
5. The return dtype stays `int32`.

The target count is the number of samples the rule would flag below the cutoff if all finite scores
were distinct: `n_target = int((ranks < np.quantile(ranks, noise_quantile)).sum())` with
`ranks = np.arange(n_finite)`. Computing it through `np.quantile` keeps it consistent with the
cutoff by construction; it equals `ceil(noise_quantile * (n_finite - 1))`. If fewer finite samples
than the target are flagged, `get_labels` warns once, with `stacklevel=2`:

```
noise_quantile=0.05 asks for about 120 of 2400 samples by gini; 31 were flagged. Samples tied at
the cutoff (0.981) or within 0.05 of the median (0.993) stay labeled.
```

NaN samples are counted neither in the target nor in the flagged count of the warning.

A cluster can lose every member to noise. Nothing prevents it; the docstring says so.

`mode` and `noise_score` are independent: `mode="generalizability"` with `noise_score="gini"`
works whenever the fit computed Gini scores.

### 4.3 Errors

- `noise_score` not in {"gini", "ce", "accuracy"}: `ValueError("noise_score must be one of:
  'gini', 'ce', 'accuracy'.")`.
- `noise_quantile` not strictly between 0 and 1: `ValueError` naming the value.
- Both checks run on every call, with the flag on or off, before configuration selection.
- `noise_labels=True` and the requested score was not computed by this fit: the `RuntimeError`
  from `_sample_scores`, with the wording the plotting methods use today.
- Every score of the selected configuration is NaN: `ValueError` from `noise_mask`.

### 4.4 Docstring

The `get_labels` docstring documents the three arguments, the rule with the 0.05 margin, NaN scores
becoming noise, the zero-accuracy caveat, that a cluster can lose all members, that the scores do
not depend on `consensus_k` or `estimator`, the `-1` convention in Returns, the new errors in
Raises, and the warning under a Warns section.

## 5. Implementation shape

### 5.1 `src/carve/_utils.py`

A module constant `NOISE_MARGIN = 0.05`, a `NamedTuple` `NoiseCut` with fields `mask`, `n_target`,
`cutoff` and `median`, and a pure function:

```python
def noise_mask(
    scores: np.ndarray, *, quantile: float, margin: float = NOISE_MARGIN
) -> NoiseCut:
```

It computes the rule in 4.2 and the target count, and raises `ValueError` when no score is finite.
It emits no warning. `_utils.py` stays a leaf; the function needs only NumPy.

### 5.2 `src/carve/api.py`

- `CARVE._sample_scores(source, config_id) -> np.ndarray` holds the lookup now repeated in the four
  plotting methods, with the same three `RuntimeError` messages and the same `ValueError` for an
  unknown source. The four methods call it and keep assigning their own axis labels.
- `get_labels` validates the two noise arguments at the top, and after the alignment block calls
  `_sample_scores` and `noise_mask`, applies the mask to a copy of the labels, and warns as in 4.2.

No other module changes. `tl`, `pl`, `_plotting` and `benchmarks` are untouched.

## 6. Verification

### 6.1 Tests

In `tests/test_utils.py`, a `TestNoiseMask` class on hand-built arrays:

- Wide spread: `np.linspace(0, 1, 101)` at quantile 0.05 gives cutoff 0.05 exactly; the mask is
  indices 0 to 4 (five samples), and `n_target` is 5.
- Equal scores: `np.full(50, 0.8)` flags nothing; `n_target` is 3.
- Bulk near the top: 95 scores spread over [0.995, 1.0] plus 5 scores at 0.4 and 0.6, at quantile
  0.10. Only the five low samples are flagged; `n_target` is 10.
- Tight bulk below one high outlier: 99 scores spread over [0.70, 0.72] plus one at 1.0, at
  quantile 0.05. Nothing is flagged; a margin anchored at the maximum would flag five.
- Ties at the cutoff: 90 scores at 1.0, 6 at 0.9, 4 at 0.5, at quantile 0.05. The cutoff falls in
  the 0.9 block, so only the four samples at 0.5 are flagged; `n_target` is 5.
- NaN: NaN entries are in the mask; `cutoff`, `median` and `n_target` equal those computed with
  the NaN entries removed.
- All NaN raises `ValueError`.

In `tests/test_api.py`, a `TestNoiseLabels` class. Tests that overwrite score arrays work on a
`copy.deepcopy` of the module-scoped `fitted_carve`, so other tests see the original. At the
`config_id` from `_select_row` with the call's arguments, the test writes a known score pattern;
every other row receives the reversed pattern, so reading the wrong configuration changes the
result.

- `noise_labels=False` returns an array equal to a plain call.
- For each of "gini", "ce" and "accuracy", the `-1` positions equal the known low indices, and
  every other position equals the plain call's label.
- The same holds after `estimator_results_` is reindexed with `TestRowIdentity._reindexed`, so the
  row position of the selected configuration no longer equals its `config_id`.
- Calling with noise, then without, returns identical labels at non-noise positions, and
  `reference_labels` contains no `-1`.
- A score pattern with ties at the cutoff triggers the warning, checked with
  `pytest.warns(UserWarning, match=r"noise_quantile=0\.05 asks for about \d+ of \d+ samples by
  gini; \d+ were flagged")`. The suite runs with `filterwarnings = error`, so every test whose
  pattern falls short of the target asserts the warning.
- An unknown `noise_score` and quantiles 0, 1 and 1.5 raise `ValueError`, with the flag on and off.
- `noise_score="gini"` on the generalizability-only fit from `split_mode_fits` raises the
  `RuntimeError`.
- An anchored fit (the `TestAnchoredLabels` setup: n=60, `anchor_threshold=30`) with a known
  pattern that puts low scores on non-anchor samples returns 60 labels with `-1` exactly there.

The existing plotting and error-contract tests cover the `_sample_scores` refactor unchanged.

### 6.2 Mutation checks

Each of these edits must make at least one new test fail:

- `<` replaced by `<=` in the cutoff comparison.
- The margin condition removed, or the median replaced by the maximum.
- NaN samples left labeled.
- The mask applied before the alignment block, so `reference_labels` holds `-1`.
- Scores read from a position instead of `config_id`.
- The warning condition compared against `n_target` with `>` instead of `<`.

### 6.3 Commands

From `code/`, with the pinned toolchain in `.venv`:

```bash
pytest tests/test_utils.py tests/test_api.py
pytest
ruff check src/
ruff format --check src/
```

## 7. Out of scope

- The R port.
- `tl.carve` and `tl.attach_results`. Scanpy marks unassigned cells as missing values in the
  categorical rather than as a `"-1"` category, so this is its own decision. The per-cell scores
  are already written to `adata.obs`.
- A plotting option that shows noise samples in grey.
- The manuscript and S2 Text.
