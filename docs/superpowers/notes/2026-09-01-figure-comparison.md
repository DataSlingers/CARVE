# Figure comparison: regenerated vs. published (task 15, step 8)

Date: 2026-09-04

## Scope of this comparison

The regenerated artifacts under `vis/` come from a reduced-scale benchmark run
(`--n-resamples 20 --n-seeds 5`), against the published `n_resamples=100`,
`n_seeds=20`. Every numeric value in these figures (curve heights, error bars,
point clouds) is therefore expected to differ from the published version, and
none of that is evaluated here. What this note verifies is structure: panel
layout and count, which series are present, axis labels and scale type, and
ordering. A numeric difference is never reported below as a finding. A
structural difference — a missing panel, a missing series, a wrong axis
label, a different panel count or ordering — is reported plainly, because
that is what this comparison exists to catch.

Each pair was opened and inspected by eye with the Read tool; findings below
describe what was actually seen, not inferred from file size or timestamp.

Known, pre-cleared differences (from the task brief, not re-litigated below):
the four screenshot-derived figures differ in resolution and rendering
style by design; Fig 4's CARVE-stability green is `#009E73`, replacing the
old `#00CD6C` (confirmed in `src/benchmarks/_theme.py:25-27`), which is the
palette fix landing, not a regression.

A separate, newly observed structural pattern recurs across three of the six
verified figures and is described once here rather than three times: a
"Baseline (Oracle)" reference series, present in every relevant published
panel, does not appear anywhere in the regenerated figures. `grep -rn
"oracle" src/benchmarks/figures/*.py` returns nothing — no figure module
plots it, even though the run pipeline computes it (`src/benchmarks/_run.py`
fits an oracle estimator) and the registry carries a display name and theme
color for it (`src/benchmarks/_registry.py:63`, `_theme.py:42`). This is
independent of run scale; it would reproduce at full scale too.

The two Klein case-study figures below are a different kind of artifact from
the rest of this comparison and are held to a different standard. They are
not read from the reduced-scale benchmark run described above; the Klein
CARVE cache is fit directly against the real Klein dataset through the same
code path used for the manuscript. Where the estimator, subsample, measure,
and rule now match the manuscript's methods, a matching number below is
reported as genuine reproduction, not as structural agreement — unlike every
other numeric value in this note, which is expected to differ and is not
evaluated.

## Figures verified

### benchmarking_examples.png (S1 Fig, `fig:benchmarking_examples`)

Differs structurally. Published (`overleaf/vis/benchmarking_examples.png`)
is a 2x3 grid, one panel per scenario, showing only the hard-SNR anchor
(caption: "at a hard SNR setting"). Regenerated
(`vis/benchmarking/benchmarking_examples.png`) is a 6x3 grid: one row per
scenario, one column per difficulty (easy/medium/hard) — 18 panels instead
of 6. This is a deliberate design choice, stated in the generating module's
own docstring (`src/benchmarks/figures/_benchmarking_examples.py:1-4`: "one
row per scenario, one column per difficulty anchor"), not a rendering bug,
but it does not match the published S1 Fig's panel count or layout.
Comparing the regenerated "hard" column against the published panels, the
cluster geometries are the right kind per scenario (Gaussian blobs,
t-distributed clumps, RFF loops for circles/moons/swiss-rolls); the
categorical cluster-label palette (5 colors) differs between the two but
that is a cosmetic/theme choice, not a content problem.

### benchmarking_results.png (main-text figure, `fig:benchmarking_results`)

Differs structurally. Both are 2x3 grids covering the same six scenarios,
and the CARVE-stability green matches the expected `#009E73` palette fix.
Three findings beyond that:

- Missing series: the grey dashed "Baseline (Oracle k*)" line, present in
  every published panel, is absent from all six regenerated panels (see the
  cross-cutting note above).
- Wrong axis label: the regenerated x-axis is labeled "Difficulty" in every
  panel; the published x-axis is labeled "SNR" in every panel. Same
  easy/medium/hard categories, different label text.
- Panel ordering: the bottom-row scenario order differs. Published:
  Swiss Rolls, Circles, Moons. Regenerated: Circles, Moons, Swiss Rolls (the
  top row, Gaussians / t-Distributed / t-Distributed + Nuisance Dims, is in
  the same order in both).

Cosmetic-only differences: regenerated panels carry A-F letter labels the
published ones do not; the published CVI lines are dashed and the CARVE
lines solid, while the regenerated figure uses solid lines throughout;
panel titles differ slightly in wording ("Gaussians" vs. "Gaussian
Mixtures").

### paper_fig_scaling_ari_k5.png (S2 Fig, `fig:scaling_main_k5`)

Differs structurally, in one respect: the same missing "Baseline (Oracle
k*)" dashed series described above is absent from both panels here too.
Everything else matches: 1x2 panel layout, x-axis labels "Number of samples
(n)" and "Number of features (p)", y-axis label "ARI (selected k̂ vs. true
labels)", CARVE Stability and CARVE Generalizability series both present
with the expected palette, error bars present in both.

### paper_fig_scaling_runtime_k5.png (S3 Fig, `fig:scaling_runtime_k5`)

Matches structurally. 1x2 panel layout, log-scale y-axis labeled "Runtime
(seconds)" in both, x-axis labels "Number of samples (n)" and "Number of
features (p)" match, both series (CARVE Stability, CARVE Generalizability)
present with legend and error bars in both. No baseline series is expected
here (the published version does not carry one either, since an oracle
runtime is not a meaningful comparison point). Minor cosmetic difference:
the regenerated right-panel x-axis starts near 0 with a "0" tick; the
published one starts at the data minimum. Not a structural issue.

### CARVE_output_klein.png (`fig:carve_output_klein`)

Status update: the estimator-mismatch finding this note previously reported
here is fixed. The finding was that the regenerated legend read "KMeans,
n_init=10" where the manuscript specifies Ward agglomerative clustering
(`CARVE_manuscript.tex:623`), traced to `src/benchmarks/_studies.py`
hardcoding KMeans in `study_model_grids()` regardless of `study.estimator`.
Commits 3b5f793 and 137dc8c fix it: `study_model_grids` now sweeps
`study.estimator` plus spectral, `STUDIES["klein"].estimator` is
`EstimatorSpec(name="agglomerative")`, and both the loader and the notebook's
own `load_klein` call were corrected to the manuscript's `subsample=0.5`. The
Klein CARVE cache was refit under this corrected configuration and the
notebook re-executed; the loader now produces exactly 1,358 cells across
2,000 genes, the count the manuscript reports at line 606.

Panel layout, letters, and per-panel content type still match the caption
(A: stability ARI over k, B: consensus matrix, C: generalizability ARI over
k, D: per-cluster stability violin, E: consensus-label scatter, F:
diagnostic scatter). Panel A and C legends now read "AgglomerativeClustering,
linkage=ward" and "SpectralClustering, affinity=self_tuning", matching the
published legend's estimator identities (the published legend names the
spectral series "SpectralClusteringCARVE", a cosmetic wrapper-name
difference that predates this fix and is not new).

Panel C is a genuine reproduction of the manuscript's headline result, not a
structural echo of it: `carve.get_k(measure="generalizability", rule="1se",
not_two=True)` returns 4, and the top generalizability row is
`AgglomerativeClustering` at `n_clusters=4` with `ari_generalizability =
0.936`. This matches the manuscript's own statement at line 624 ("The
generalizability ARI with the 1-SE rule selected Ward agglomerative
clustering at k=4") and tracks the published panel's peak closely (~0.945 at
k=4, same estimator, same k). Because this case study is fit directly on the
real dataset rather than read from the reduced-scale benchmark run described
above, this match is a real result, not a coincidence of comparable run
scale.

Two differences remain, both distinct from the fixed estimator bug (confirmed
distinct because the estimator identities are now correct throughout both
figures):

- Panel A's own stability selection differs: the regenerated 1-SE rule picks
  k=2 (SpectralClustering, ARI 0.98 at k=2, the single highest stability
  score in the whole grid); the published panel picks k=3. This is
  consistent with the manuscript's own caveat at line 624 ("While the
  stability ARI did not indicate a clear preference...") — stability alone
  is not expected to cleanly separate k here, in either run.
- Panels B, D, E, and F are all keyed to the stability selection rather than
  the generalizability one (`src/benchmarks/figures/_carve_output.py:52`,
  `carve.get_k(measure="stability", rule="1se")`, documented in that file as
  the manuscript's recommended default view). Because stability now picks
  k=2, those four panels show a 2-cluster consensus matrix, violin, and
  scatter (SpectralClustering) instead of the published figure's 4-cluster
  Ward view. This is not the estimator bug re-appearing — the panel D/E/F
  captions correctly name "SpectralClustering, affinity=self_tuning" — it is
  a downstream consequence of the differing stability selection above, on a
  panel set the code has always driven by stability rather than by the
  manuscript's manually preferred k=4 solution.

Net effect: the defect this note originally reported here — the wrong
estimator swept entirely — is fixed, and the one panel that carries the
manuscript's headline claim (C) now reproduces it. The stability-driven
detail panels (B, D, E, F) still diverge from the published figure, for
reasons unrelated to that defect.

### klein_results.png (`fig:klein_results`)

Same root fix as the figure above (same corrected `study_model_grids` and
`STUDIES["klein"].estimator`); this figure's estimator identification is
also threaded explicitly from the notebook
(`prepare_composite(..., measure="generalizability", rule="1se",
not_two=True, ...)` in `notebooks/case_studies/Klein.ipynb`). Panel lettering
and titles still match (A: Reported Labels, B: CARVE clustering, C: CVI
(Silhouette, k=2), D: CARVE ARI over k, E: CVIs over k, F: alluvial/Sankey).
Re-checked panel by panel against the published figure:

- Panel A and B: both now show 4 groups in the regenerated figure, matching
  the published figure's 4 groups, with panel B tracking panel A's reported
  time-point groups the same way in both — consistent with the manuscript's
  claim that the k=4 solution recovers the reported differentiation stages.
- Panel C, which uses only Silhouette-selected k=2 in both cases, is
  unaffected by any of this and continues to match well in shape.
- Panel D: both figures now show two dashed selection lines, one per rule.
  The generalizability line lands at k=4 in both (regenerated ARI ≈0.936,
  published ≈0.945 — the same headline result as `CARVE_output_klein.png`
  panel C). The stability line lands at k=2 in the regenerated figure versus
  k=3 in the published one — the same difference as, and for the same
  reason as, panel A of `CARVE_output_klein.png` (the manuscript's own text
  that stability did not give a clear preference here). The regenerated
  legend also still does not attach an estimator name to either curve
  ("CARVE Generalizability (1SE)" / "CARVE Stability (1SE)"), where the
  published legend does ("Generalizability (1se) —
  AgglomerativeClustering, linkage=ward" / "Stability (1se) —
  SpectralClusteringCARVE, affinity=self_tuning"); a cosmetic completeness
  gap, not an identity error — the curve colors and selected k's already
  correctly track which estimator wins each measure.
- Panel E's CVI curves are still jagged and non-monotonic in the regenerated
  version, smooth in the published one. The previous version of this note
  attributed that to "interleaving two different estimators' scores per k
  instead of one," as a consequence of the KMeans/Ward mismatch — with the
  estimator bug now fixed, the curves are still jagged, so that explanation
  needs correcting rather than just retiring: `cvi_lines`
  (`src/benchmarks/_panels.py:260`) groups `curves_df` by metric only and
  plots every `(model, k)` row on one line, so with both
  `AgglomerativeClustering` and `SpectralClustering` now correctly in the
  swept grid, each metric's line still zig-zags between the two models'
  scores at every k. The published figure's CVI lines are each labeled "—
  Agglomerative (linkage=ward)" and show only Ward's scores. This is a real,
  separate, still-open gap in the plotting function itself — it was never
  restricted to a single estimator per line — and fixing the estimator
  identity bug did not fix it.
- Panel F's Sankey: the CARVE-column block count now matches (4 regenerated,
  4 published), where it previously differed (3 vs 4) tracking the old
  k=2-vs-k=4 selection gap — that specific discrepancy is resolved. The
  published diagram still carries percentage annotations on every block and
  day labels (d0/d2/d4/d7) on the "Reported Label" column that the
  regenerated diagram does not render at all; that rendering-completeness
  gap is unrelated to the estimator fix and persists.

### CARVE_output_levine.png — not verified

Not regenerated this session. The Levine dataset is absent from this
machine, so the Levine case-study notebook could not execute, and no file
exists at `vis/case_studies/CARVE_output_levine.png` to compare.

### levine_results.png — not verified

Not regenerated this session, for the same reason (Levine dataset absent,
Levine notebook could not run). A file exists at
`vis/case_studies/levine_results.png`, but it predates this session (mtime
2026-05-09, versus the 2026-09-04 09:35 mtimes on the two Klein outputs,
current as of the Klein re-run under the corrected configuration) and is
leftover from an earlier run, not an artifact this comparison can vouch for.
It was not opened or compared.
