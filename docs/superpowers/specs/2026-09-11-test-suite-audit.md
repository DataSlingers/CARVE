# CARVE Python test suite audit

Date: 2026-09-11. Scope: `code/tests/` (carve and benchmarks), 52 files, 15.7k lines.
The R package is out of scope.

Method: every test file was read in full. Both suites were run locally
(`.venv`, Python 3.12.11, pytest 9.1.1, pandas 3.0.5, numpy 2.5.2, sklearn 1.7.2).
Claims about vacuous guards were settled by mutation, not by reading.

Baseline (`OMP_NUM_THREADS=1`, `-W default`):

- carve suite (`pytest tests --ignore=tests/benchmarks`): 686 passed, 0 skipped,
  780 warnings, 181 s. Slowest: `test_anchored_accuracy.py` (about 45 s for four tests),
  then the `tl.carve` fits in `test_tl.py` at 2.3 s each.
- benchmarks suite (`pytest tests/benchmarks`): 753 passed, 10 skipped (8 regression
  gate, 2 rpy2 gate), 117 warnings, 419 s. `test_cli.py` (2 tests) takes 83 s and
  `test_run.py` about 240 s; together they are roughly 5.5 of the 7 minutes.

Nothing fails. The findings below are about tests that pass for the wrong reason, behavior
that is not tested, and structure that will not hold up as the suite grows.

Line numbers refer to the current working tree (HEAD db4eca7).

---

## Part 1. Tests that do not make sense, or do not test what they claim

### 1.1 Copy-paste defects and dead code

| Where | Problem | Fix |
|---|---|---|
| `tests/test_api.py:133-135` | `test_generalizability_scores_populated` asserts `generalizability_scores_ is not None` twice. The second line was almost certainly meant to be `consensus_generalizability_matrices_`. | Replace the duplicate with the intended attribute. |
| `tests/test_types.py:57-63` | `test_invalid_mode` and `test_invalid_mode_type` are the same test (both pass a string). | Delete one, or make the second pass a non-string (`None`, `1`) if that path exists. |
| `tests/benchmarks/test_benchmarks_types.py:232` | `test_unknown_estimator_still_raises` duplicates `TestEstimatorSpec.test_raises_on_an_unknown_name`. | Delete. |
| `tests/benchmarks/test_studies.py:342` | `test_both_case_studies_are_registered` asserts four studies; "both" is stale, and `TestNewStudies.test_both_new_studies_are_registered` covers the same fact. | Rename to `test_every_case_study_is_registered` and drop the duplicate. |
| `tests/conftest.py:231` | `fitted_adata` (module-scoped, runs a fit) is defined and never used anywhere. | Either delete it or use it for the missing `test_pl.py` (see Part 2). |
| `tests/benchmarks/test_theme.py:13` | `CLUSTER_CMAP_NAME` imported and unused (ruff F401). | Remove. |
| `tests/test_grids.py:90` | `ParameterGrid` re-imported inside a test that already has it at module level. | Remove the inner import. |

### 1.2 Tests whose name or docstring does not match what they check

| Where | Problem | Fix |
|---|---|---|
| `tests/test_api.py:327` `test_reference_labels_alignment` | Docstring: "successive calls should produce consistent labels". It calls `get_labels()` twice with identical arguments and never touches reference labels. It cannot see the actual alignment behavior (see Part 2, item 9). | Rename to `test_get_labels_is_idempotent` and add a real alignment test. |
| `tests/test_api.py:208` `test_per_call_random_state` | Passes `random_state=99` to `fit()` and asserts only that results exist. Nothing checks that the seed was honored. | Fit twice with `random_state=99` and once with `random_state=0`; assert the first two frames are equal and differ from the third. |
| `tests/test_api.py:188` `test_reproducibility` | Compares a single scalar (`ari_stability.iloc[0]`) between two fits. | Compare whole `estimator_results_` frames and the consensus matrices. |
| `tests/benchmarks/test_calibrate.py:41` `test_returns_a_value_in_the_unit_interval` | Asserts `-1.0 <= ari <= 1.0`, which is not the unit interval. | Either rename to `..._in_ari_range` or tighten the bound. |
| `tests/benchmarks/figures/test_scaling.py:109` `test_no_element_scale_fudge_is_applied_by_default` | Asserts the literal `"0.47"` does not appear in the module source, and requests a `results` fixture it never uses. A negative source-text search is not a behavioral assertion; the number is inscrutable to a reader, and any other literal (0.5, 0.48) would pass. | Delete, or replace with the property it stands for (marker/line sizes on this figure equal those on Fig 4 under the same theme). |
| `tests/test_plotting.py:253` `test_no_legend` | Passes `legend=False` and asserts `ax is not None`. | Assert `ax.get_legend() is None`. |
| `tests/test_plotting.py:584` `test_plot_consensus_matrix_under_anchoring` | The comment says "The rendered image is the anchor block", but the assertion reads `c.consensus_matrices_[0].shape`, which is model state, not what was drawn. | Assert `ax.images[0].get_array().shape == (30, 30)`. |

### 1.3 Guards that pass regardless (verified by mutation)

The anchoring `RuntimeWarning` in `api.py:399` was removed and the suite re-run.

- `tests/test_tl.py:290` and `:296` (`TestAnchoringProvenance`): both bare
  `pytest.warns(RuntimeWarning)` blocks stayed green. They are satisfied by the unrelated
  `RuntimeWarning: Mean of empty slice` that `_consensus.py:375` emits on this fixture. As
  written they neither assert the anchoring warning nor silence it cleanly.
- `tests/test_api.py` (11 sites) and `tests/test_plotting.py` (2 sites) use the same bare
  pattern. Those 18 tests did go red under the mutation, so they currently work, but only
  because `_blobs(60)` with six resamples happens to co-sample every pair. Change the fixture
  size or resample count and they become vacuous the same way.

Fix: add `match="anchored consensus"` everywhere the intent is to assert the warning, or wrap
in `warnings.catch_warnings()` where the intent is only to keep the log quiet. Never use bare
`pytest.warns(RuntimeWarning)` in this code base while the library emits nanmean warnings.

### 1.4 Assertions too weak to catch a wrong answer

These pass with a same-shape wrong value.

- `tests/test_selection.py`: `TestSelectBestRow1se.test_return_idx`,
  `TestSelectBestRowQuantile.test_return_idx`, `TestSelectBestRowByRule.test_return_idx`
  assert only `isinstance(idx, int)`. The `max` variant asserts `== 0`; do the same.
  `TestSelectBestRowByRule.test_1se_rule` and `test_quantile_rule` assert only
  `isinstance(row, pd.Series)`.
- `tests/test_consensus.py::TestReorderConsensusMatrix`: three tests, none checks that the
  reordering actually groups the two interleaved blocks `{0,2,4}` and `{1,3,5}` (the function's
  purpose). `test_preserves_values` uses a 2x2 matrix where nothing can move.
- `tests/test_plotting.py`: 23 `assert ax is not None` smoke tests. The ones where a real
  property is cheap to assert: `test_custom_order` (tick labels in the requested order),
  `test_alpha_range` (facecolor alphas within the range), `test_high_dim_uses_pca` (axis
  labels read PC1/PC2), `test_violin_no_clip_when_ylim_none` (some vertex outside [0,1]),
  `test_nan_handling` (image array contains the NaN fill), `test_with_existing_ax`.
- `tests/test_api.py::TestGetLabels::test_different_measures` calls
  `get_labels(measure="pac"|"gini"|"ce")` with the default `rule="1se"`, which has no `_se`
  column and silently falls back to `max` with a `RuntimeWarning` (visible in the run log).
  The test asserts only a shape. Either pass `rule="max"` for those measures or assert the
  fallback warning explicitly; as is, a regression in the fallback is invisible.
- `tests/test_runner.py::TestComputeStabilityAri::test_basic` asserts `-0.5 <= ari <= 1.0`.
  The overlap is two points in two singleton clusters, so sklearn returns exactly 1.0; assert
  that.
- `tests/test_runner.py::TestResampleResult::test_field_count` pins `len(_fields) == 17`.
  A field count is a change detector, not a contract; the named-field test above it is
  sufficient.
- `tests/test_accuracy.py::test_wrong_predictions` asserts `np.any(scores < 1.0)`; the exact
  vector is computable from the fixture.
- `tests/test_utils.py::TestEnsure2dArray::test_invalid_type` and
  `tests/test_utils.py:499` (`test_rejects_degenerate_counts`, six parametrized cases) use `pytest.raises(ValueError)` with no `match`; the latter cannot tell a rejected count from an unrelated `ValueError`.
- `tests/test_tl.py:148` `pytest.raises((ValueError, RuntimeError, KeyError))` accepts three
  exception types. Pin the one `tl.carve` actually raises and its message.

### 1.5 Redundant or misplaced work

- `tests/test_anchored_accuracy.py::test_accuracy_improves_with_more_anchors` (13 s, the
  slowest test in the carve suite) refits `m=500` and `m=2000`, both of which the parametrized
  test above it already fitted. A module-scoped, parametrized fixture caching the anchored
  fits would remove roughly 15 s. The `print()` calls in this file are invisible without `-s`;
  if the numbers back a supplementary table, write them to a file or drop them.
- `tests/test_plotting.py:584-625`: the two anchoring tests fit the identical model twice.
  Share a module-scoped fixture.
- `tests/benchmarks/test_run.py`: the identical `run_cell(...)` call (120 samples, 3 ks,
  20 resamples) is repeated verbatim in 10 tests, and `run_scenario(...)` (2 axes x 2 seeds)
  is called 16 times. Every one is a full CARVE fit. This file is where the benchmarks suite
  spends most of its time; a module-scoped `rows, runtime = run_cell(...)` fixture and a
  module-scoped completed run directory (copied with `shutil.copytree` for the tests that
  delete checkpoints) would cut it substantially without weakening any assertion.
- `tests/benchmarks/test_cli.py`: `test_runs_a_scenario_end_to_end` and
  `test_promote_publishes_a_finished_run` each run the full `gaussians` scenario at
  `--n-seeds 1 --n-resamples 20`. The second can promote the first's output.
- `tests/test_anndata.py::test_1d_sparse_rejected` carries a skip for "older scipy" that
  cannot trigger under the `scipy>=1.16` floor. `tests/test_h5ad_roundtrip.py::TestZarr`
  guards with `importorskip("zarr")`, but zarr is a hard dependency of anndata>=0.13.
  Both guards are dead; remove them so the tests are unconditional.
- `tests/benchmarks/test_regression.py` and `tests/test_anchored_accuracy.py` are marked
  `slow`, but nothing deselects the marker: no `addopts`, no `-m` in CI. The anchored-accuracy
  file therefore runs in every CI job (about 45 s), while the regression file is gated by an
  environment variable instead of the marker. The marker currently does nothing.

---

## Part 2. Missing tests

Ordered by how much untested public behavior each item represents.

1. `carve.pl` has no test file. `src/carve/pl/_plots.py` is 775 lines and part of the public
   surface named in CLAUDE.md. Its only coverage is six parametrized
   `isinstance(fn(reloaded), Axes)` smoke calls in `test_h5ad_roundtrip.py`. Nothing tests:
   `key=` with a non-default key; every error path in `_entry`, `_labels`, `_scores`,
   `_results` (missing key, malformed uns entry, bad `source`, `store_results=False`,
   `store_consensus=False`); `_annotation_text` resolving `annotation=True|False|str`;
   `_representation` reconstructing `use_rep`/`layer`/`n_pcs` from params; that each
   `pl.*` function draws the same thing as the corresponding `CARVE.plot_*` method on the
   fitted model. The unused `fitted_adata` fixture in `conftest.py` is the natural base for
   `tests/test_pl.py`, which the "tests mirror modules 1:1" rule calls for.

2. `plot_diagnostic_scatter` (`_plotting.py:1201`, about 300 lines) has no unit test in
   `test_plotting.py`; its siblings have 6 to 10 each. It is exercised only by one smoke line
   under anchoring and via `pl` from disk. At minimum: shape-per-cluster encoding (one
   collection per cluster), colorbar label from `source`, `markers=` override, `sort_order`,
   `annotation_style="box"`, save path, and the "no finite scores" error.

3. `SpectralClustering` sparse eigensolver branch (`cluster.py:255-270`, `n >= 1000`,
   `eigsh` plus the `ArpackNoConvergence` fallback) has zero coverage. Every cluster fixture
   is `n <= 500`, so only the dense `eigh` path runs. `test_regression.py` documents a
   suspected nondeterminism in exactly this branch as the reason circles/moons are excluded
   from the regression gate. Add a reproducibility test at `n=1500` (two fits with the same
   `random_state` agree) and a correctness test (moons at `n=1500`, ARI > 0.9). If the
   reproducibility test fails, mark it `xfail(strict=True)` so the fix flips it visibly.
   (Locally, with scipy 1.16, four repeated fits at n=1500 agreed exactly, so the
   documented failure may be version-dependent; the test settles it either way.)

4. `CARVE.fit(randomize_preprocessing=True)` end to end. Only the pipeline builders
   (`test_pipeline.py`) and the summarizer on synthetic records (`test_utils.py`) are
   tested. Nothing fits with randomized preprocessing and checks that
   `preprocessing_results_` (api.py:262) is populated, that it groups on the sweep param,
   that the run is reproducible under a seed, or that `validation_iter` records the chosen
   normalization and reduction names.

5. `show_progress=True` (the tqdm path in `_output.py` / `_runner.py`) is never exercised.

6. Error contracts on the fitted model. None of these `api.py` messages has a test:
   `Unknown estimator_param_grids preset` (api.py:454), `Cannot cut the consensus matrix`
   (753), `Mode must be 'default' or 'generalizability'` (768), `Consensus matrix not
   available for mode=` (771), `source must be one of` and `mode must be one of` on all four
   score plots (1361, 1370, 1560, 1569, 1767, 1776, 1996, 2005), and the three
   `... scores are not available for this run` errors that fire after a `mode="stability"`
   or `mode="generalizability"` fit. The split modes are the ones `fit()` warns are
   experimental; their failure surface is exactly what should be pinned.

7. `get_labels(not_two=True)`, `get_k(not_two=True)`, `get_estimator(k=...)`, and
   `get_estimator()` on a `min_cluster_size` run (reconstructing an `HDBSCAN` from a row)
   are not tested at the `CARVE` level. `not_two` is covered only through `tl` and the
   selection units.

8. Re-fit semantics. Nothing calls `fit()` twice on one instance. It is not established that
   a second fit resets `consensus_anchors_`, `sweep_`, `preprocessing_results_`, and
   `reference_labels`, or that a k-mode fit after a resolution-mode fit leaves no stale
   columns.

9. `get_labels()` mutates the model: when `reference_labels` is `None` or its cluster count
   differs from the cut, it overwrites `self.reference_labels` with the returned labels
   (api.py:805-811). So a user who passed `reference_labels` with k=3 and then calls
   `get_labels(k=2)` silently loses their reference. No test observes this, and
   `test_reference_labels_alignment` cannot. Add a test that pins the intended behavior
   (whichever it is) for the sequence `get_labels(k=2)`, `get_labels(k=3)`,
   `get_labels(k=2)` and for a user-supplied reference with a different k.

10. Constructor and fit validation. There is no validation of `subsample_ratio` outside
    `(0, 1)`, `n_resamples < 1`, or `n_trees < 1`, so there are no tests either. Adding the
    checks and their tests together is the right unit of work; `subsample_ratio=1.0` in
    particular yields an empty test split and should fail loudly.

11. `SweepSpec` round trip through `save()`/`load()` is tested for resolution mode only; the
    `min_cluster_size` (inverted rank) case is not, and it is the one where a rank bug would
    invert selection after reload.

12. `_summarize_ari_scores`: `q95`/`q05` are asserted only as ordered, never against the
    known quantiles of the input. `test_single_value` asserts `se` is NaN but not the
    quantiles.

13. Benchmarks:
    - `tests/benchmarks/test_notebooks.py` checks six notebooks but not `Tutorial.ipynb` or
      `Resolution_Tutorial.ipynb`, which the nightly workflow executes.
    - `benchmarks.tables` `--tables` CLI path warns on multiple run directories; nothing tests
      which one is chosen.
    - `_studies.study_scaling_sweep` is tested for `sizes` but not for `consensus_anchors`
      forwarding, unlike `fit_or_load_carve` which has a spy for it.

---

## Part 3. Best practices not yet followed

### 3.1 Warning hygiene (the largest single issue)

The carve run emits 780 warnings; 738 are `RuntimeWarning: Mean of empty slice` from
`_consensus.py:169-170` and `:375-376` (`np.nanmean` over a consensus row that is entirely
NaN because that sample was never co-sampled with anyone). This is normal at small
`n_resamples` and the library should handle it explicitly (count non-NaN entries per row and
return NaN where the count is zero, or wrap the two calls in `warnings.catch_warnings`).
Until it does:

- the suite cannot adopt `filterwarnings = ["error"]`, which is the single most effective
  best practice available here (a memory note records that `-W error` currently produces
  about a dozen failures for this reason);
- real warnings are buried (the joblib `DeprecationWarning` under numpy 2.5 at
  `numpy_pickle.py:207` and the umap `ImportWarning` appear once each in 780 lines);
- bare `pytest.warns(RuntimeWarning)` is vacuous on any fixture that triggers it (Part 1.3).

Recommended sequence: fix the two nanmean sites, then add to `pyproject.toml`

```toml
[tool.pytest.ini_options]
filterwarnings = [
  "error",
  "ignore:Tensorflow not installed:ImportWarning",
  "ignore:Setting the shape on a NumPy array:DeprecationWarning:joblib",
]
```

and let every remaining warning in a test be either asserted with `pytest.warns(..., match=)`
or scoped with `@pytest.mark.filterwarnings`. Order matters: pytest applies the last matching
filter, so `error` must come first and the ignores after it. Marks on a test take precedence
over the ini list, so a test that legitimately needs a warning can opt out locally.

### 3.2 Global state leaks between tests

- `tests/benchmarks/test_theme.py::TestRcParams::test_apply_theme_actually_mutates_rcparams`
  and `test_theme_context_restores_previous_state` set `plt.rcParams["pdf.fonttype"]` and
  call `apply_theme()` without restoring. Verified: running `TestRcParams` followed by a
  probe test that asserts `plt.rcParams["axes.spines.top"] is True` (matplotlib's default)
  fails; `apply_theme()` leaves all of `RC_PARAMS` in effect for every later test in the
  process, including `test_panels.py` and the figures tests. Nothing currently breaks only
  because no later test asserts a default-rcParams property, which is luck, not design.
  Matplotlib's own guidance is `matplotlib.rc_context()`, which resets every rcParam changed
  inside the block on exit; the cheapest fix is to extend the existing autouse fixture in
  `tests/conftest.py` to `with matplotlib.rc_context(): yield`, which protects every test at
  once. Running the suite under `pytest-randomly` (or `-p random_order`) once in a while is
  the standard way to surface this class of bug.
- `tests/test_api.py::TestRowIdentity` and `test_misaligned_config_id_raises` patch
  attributes by hand with `try/finally` (module-scoped fixture attributes at 1016, 1033,
  1044, 1054, 1081; module globals at 1122-1129). Use the `monkeypatch` fixture, which is
  exception-safe and already used elsewhere in the same file.
- `_SeedSpy.seen`, `_NJobsSpy.seen`, `_ParallelSpy.seen`, `_SpyCARVE.captured_kwargs` are
  mutable class attributes shared across tests. They are cleared in `try/finally`, but an
  instance-level list handed in via a fixture would remove the need.

### 3.3 Duplicated test infrastructure

- `_NJobsSpy` is defined identically in `test_api.py:1170` and `test_runner.py:65`.
- A stub CARVE (`estimator_results_` + `_select_row` + `get_k`) is defined five times:
  `test_panels.py:398`, `figures/test_case_study.py:31`, `figures/test_heca_results.py:41`
  and `:148`, `figures/test_cusanovich_results.py:39`. One of them says in its docstring that
  it is "duplicated rather than shared ... a Minor finding already deferred to final
  review". That review is this one: move it to `tests/benchmarks/conftest.py`.
- `matplotlib.use("Agg")` appears at module level in 15 test files, and `close_figures`
  autouse fixtures in `test_api.py:90` and `test_plotting.py:95` duplicate the one in
  `conftest.py`. The conftest fixture is itself autouse per test and calls
  `matplotlib.use("Agg", force=True)` 686 times; make it session-scoped (or set
  `MPLBACKEND=Agg` once at conftest import) and delete every per-file copy.
- `test_panels.py` writes `fig, ax = plt.subplots()` ... `plt.close(fig)` by hand in every
  test (about 60 times) even though the autouse fixture closes all figures. An `ax` fixture
  removes roughly 120 lines and the closes that are skipped whenever an assertion fails.
- `RecordingCarve` subclass-and-monkeypatch pattern is repeated three times in
  `test_run.py`; a small fixture would do.

### 3.4 Import layout and collection

- `tests/` has no `__init__.py` files and pytest runs in the default `prepend` import mode,
  so every test basename must be unique across the whole tree. `test_benchmarks_types.py`
  carries a docstring explaining it was named to avoid colliding with `tests/test_types.py`.
  This is a trap for the next file. The pytest project's current recommendation for new
  projects is `--import-mode=importlib` (`addopts = ["--import-mode=importlib"]`), which
  leaves `sys.path` alone and allows duplicate basenames without `__init__.py`. The
  alternative, adding `__init__.py` to `tests/` and its three subdirectories, also works
  with prepend mode.
- `from conftest import requires_graph, with_sweep_cols` (in `test_api.py`,
  `test_cluster.py`, `test_selection.py`, `test_plotting.py`) imports `conftest` as a plain
  module. This works only because prepend mode puts `tests/` on `sys.path`. It is exactly
  the case the pytest docs call out as unsupported under `importlib` mode ("test modules
  can't import each other" and "testing utility modules in the tests directories ... are
  not importable"), and it also breaks under the `__init__.py` layout (where the module
  becomes `tests.conftest`). Either fix requires the same change: make `with_sweep_cols` a
  fixture (a fixture that returns the function is fine), and turn `requires_graph` into a
  registered marker handled in `conftest.py` by `pytest_collection_modifyitems`, which
  adds `pytest.mark.skip` to every item carrying it when `igraph`/`leidenalg` are absent.
  The pytest docs' other option, shipping helpers as `carve.testing`, is wrong for this
  package because the public surface is deliberately small.
- Mid-file imports: `test_artifacts.py:259`, `test_benchmarks_types.py:56,145,147` (ruff
  E402). Each marks a place where a later task appended to the file. Hoist them.
- Local imports inside test bodies that duplicate module-level ones
  (`test_studies.py:558-567`, `test_grids.py:90`, several `from benchmarks._theme import`
  in figure tests) should be hoisted.
- `tests/` is not linted by CI (by design), but running `ruff check tests/ --select
  E402,F401,F811,I` locally once would clear the 23 findings that exist today.

### 3.5 Dependency declarations

- `tests/benchmarks/test_ci_config.py` imports `yaml`. PyYAML is not declared in any extra;
  it reaches the CI benchmarks job only through `anndata -> zarr -> donfig -> pyyaml`.
  Declare `pyyaml` in the `dev` extra or parse the workflow with a regex as the same file
  already does for `--n-resamples`.
- The local `.venv` is Python 3.12.11 while `requires-python = ">=3.13"` and CI runs 3.13.
  Not a test defect, but local green does not certify the CI interpreter.

### 3.6 pytest configuration

`[tool.pytest.ini_options]` sets only `testpaths` and the `slow` marker. Worth adding:

- `addopts = ["-ra", "--import-mode=importlib"]` so skips and xfails are always listed and
  the import trap in 3.4 is closed (CLAUDE.md's note that "there are no pytest addopts"
  would need updating; `-ra` and the import mode do not conflict with the CI flags);
- `xfail_strict = true` so a fixed bug cannot leave a stale xfail (relevant to the sparse
  spectral test proposed in Part 2, item 3);
- `filterwarnings` as in 3.1;
- a registered `requires_graph` marker (3.4);
- either honor the `slow` marker (`-m "not slow"` locally, opt in on CI) or remove it, since
  today it is decorative.

Two optional dev dependencies would pay for themselves: `pytest-randomly` (order
independence, which would have caught 3.2) and `pytest-xdist` (`-n auto` would bring the
7-minute benchmarks job well under half that even before the fixture caching in 1.5,
since every fit already runs with `n_jobs=1`).

### 3.7 Small things

- `test_api.py:11-17` sets the backend at import, then imports `warnings` twice
  (`import warnings` and `import warnings as _w`).
- `test_anchored_accuracy.py:47` imports `warnings` inside a helper.
- `tests/benchmarks/figures/test_panels.py` and `figures/test_benchmarking_results.py` read
  `legend._ncols`, a private attribute that was `_ncol` before matplotlib 3.6. Acceptable
  given the `matplotlib>=3.9.4` floor, but worth a comment.
- `tests/benchmarks/test_artifacts.py::TestConfigHash::test_changes_when_an_anchor_changes`
  rebuilds a `Scenario` with `type(s)(...)` and drops `k_star`, `candidate_k`, `n_seeds`,
  `n_trees`; `dataclasses.replace(s, anchors=...)` (used two tests later) is the intended
  idiom.

---

## What is in good shape

For balance: the sweep abstraction (`test_sweep.py`), the anchored consensus math
(`test_consensus.py::TestConsensusAnchorBlock`, `TestStabilityFromRunsAnchored`), the
config_id join tests (`test_api.py::TestRowIdentity`, with a non-vacuity guard), the core
budget tests (`test_utils.py::TestResolveCoreBudget`, `test_runner.py::TestRunValidationCoreBudget`),
the panel primitives (`test_panels.py`), and the composed figure tests
(`figures/test_carve_output.py`, `test_case_study.py`) are strong: assertions are on values,
colors and drawn data, fixtures are built so a wrong answer differs from a right one, and
several docstrings record the mutation that was used to prove the test can fail. The
benchmarks side is, on average, better engineered than the older carve-side tests.

---

## Suggested order of work

1. Fix the nanmean warning sites in `_consensus.py`, then turn on `filterwarnings = error`
   and add `match=` to every `pytest.warns`. This is the one change that hardens everything
   else.
2. Write `tests/test_pl.py` and `plot_diagnostic_scatter` unit tests (Part 2, items 1-2).
3. Add the sparse-path spectral tests (Part 2, item 3).
4. Fix the Part 1.1 and 1.2 defects (an hour of mechanical edits).
5. Consolidate the stub CARVE and spies into `tests/benchmarks/conftest.py` and a
   `tests/_helpers.py`; add `__init__.py` or `importlib` mode; restore `rcParams` in
   `test_theme.py`.
6. Cache the expensive fits in `test_run.py`, `test_cli.py`, `test_anchored_accuracy.py`.
7. Cover the remaining Part 2 items as time allows, starting with the split-mode error
   contracts and re-fit semantics.

---

## Sources consulted for Part 3

- pytest, Good Integration Practices (import modes, layouts, unique basenames):
  https://docs.pytest.org/en/stable/explanation/goodpractices.html
- pytest, import mechanisms and sys.path (importlib mode limitations on test utility
  modules): https://docs.pytest.org/en/stable/explanation/pythonpath.html
- pytest, How to capture warnings (`filterwarnings`, ordering, mark precedence):
  https://docs.pytest.org/en/stable/how-to/capture-warnings.html
- matplotlib, Testing (rcParams and figure cleanup via fixtures):
  https://matplotlib.org/stable/devel/testing.html
- matplotlib, `rc_context`:
  https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc_context.html
