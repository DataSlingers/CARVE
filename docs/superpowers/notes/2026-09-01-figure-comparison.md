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

Differs structurally, and not for a reduced-scale reason. Panel layout,
letters, and per-panel content type all match the caption (A: stability ARI
over k, B: consensus matrix, C: generalizability ARI over k, D: per-cluster
stability violin, E: consensus-label scatter, F: diagnostic scatter). The
finding is an estimator mismatch: the regenerated legend reads "KMeans,
n_init=10" and "SpectralClustering, affinity=self_tuning"; the published
legend reads "AgglomerativeClustering, linkage=ward" and
"SpectralClusteringCARVE, affinity=self_tuning". The manuscript's methods
text is explicit that the Klein case study ran "Ward agglomerative and
spectral clustering with self-tuning affinity" (`CARVE_manuscript.tex:623`).
The source of the mismatch is in code, not data volume:
`src/benchmarks/_studies.py:206-211` registers `STUDIES["klein"]` with
`estimator=EstimatorSpec(name="kmeans")`, and `study_model_grids()`
(`_studies.py:221-231`) sweeps `EstimatorSpec(name="kmeans")` and
`EstimatorSpec(name="spectral")` — KMeans stands in for Ward agglomerative
clustering everywhere in this case study. Downstream consequences follow
directly from that: the regenerated run selects k=2 for both stability and
generalizability (panel A/C dashed selection lines), against k=3 (stability)
and k=4 (generalizability) in the published figure, and panel B's consensus
matrix shows 2 blocks instead of 4. This would not resolve by increasing
`n_resamples`/`n_seeds`; the estimator being swept is different.

### klein_results.png (`fig:klein_results`)

Differs structurally, same root cause as the figure above. Panel lettering
and titles match (A: Reported Labels, B: CARVE clustering, C: CVI
(Silhouette, k=2), D: CARVE ARI over k, E: CVIs over k, F: alluvial/Sankey).
Panel C, which uses only Silhouette-selected k=2 in both cases, matches well
in shape. Panels D, E, and F show the same estimator-identity gap as
`CARVE_output_klein.png`: the regenerated panel D legend has no estimator
name attached to "CARVE Stability (1SE)" / "CARVE Generalizability (1SE)",
where the published legend ties each rule explicitly to an estimator
("Stability (1se) — SpectralClusteringCARVE" / "Generalizability (1se) —
AgglomerativeClustering, linkage=ward"); the regenerated panel D shows what
look like four overlapping curves rather than the published two, consistent
with both KMeans and Spectral results being drawn together instead of one
result per rule. Panel E's CVI curves are jagged/non-monotonic in the
regenerated version and smooth in the published one, again consistent with
interleaving two different estimators' scores per k instead of one
(Ward agglomerative only, as the published legend states). Panel F's Sankey
also differs in block count (3 CARVE-column blocks regenerated vs. 4
published, tracking the k=2-vs-k=4 selection gap) and the published diagram
carries percentage annotations on every block and day labels (d0/d2/d4/d7)
on the "Reported Label" column that the regenerated diagram does not render
at all.

### CARVE_output_levine.png — not verified

Not regenerated this session. The Levine dataset is absent from this
machine, so the Levine case-study notebook could not execute, and no file
exists at `vis/case_studies/CARVE_output_levine.png` to compare.

### levine_results.png — not verified

Not regenerated this session, for the same reason (Levine dataset absent,
Levine notebook could not run). A file exists at
`vis/case_studies/levine_results.png`, but it predates this session (mtime
2026-05-09, versus the same-session 2026-09-04 07:07 timestamps on the two
Klein outputs) and is leftover from an earlier run, not an artifact this
comparison can vouch for. It was not opened or compared.
