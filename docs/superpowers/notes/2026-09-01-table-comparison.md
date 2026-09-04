# Table comparison: regenerated vs. published (task 15, step 9)

Date: 2026-09-04

## Scope of this comparison

The regenerated fragments under `vis/tables/` come from a reduced-scale
benchmark run (`--n-resamples 20 --n-seeds 5`), against the published
`n_resamples=100`, `n_seeds=20`. Every numeric cell (ARI mean, sd,
k-recovery proportion) is therefore expected to differ from
`CARVE_manuscript.tex`, and no numeric difference is reported as a finding
below — that includes the `gap` metric and `ari_at_k` for the
generalizability metrics, which the task brief already calls out as
expected to differ even at full scale because plan 1 fixed bugs there. What
this note verifies is structure: which scenario each fragment came from,
which manuscript table it corresponds to, column headers, row count,
ordering, and caption shape.

Fragment-to-scenario-to-table mapping is `TABLE_NAMES` in
`src/benchmarks/tables.py:26-35`, read directly rather than inferred:

| Fragment    | Scenario                  | Manuscript table | Manuscript lines |
|-------------|----------------------------|-------------------|-------------------|
| S2_table.tex | gaussians                 | S2 Table          | 954-1009 |
| S3_table.tex | t_dist                    | S3 Table          | 1011-1066 |
| S4_table.tex | t_dist_noise              | S4 Table          | 1068-1123 |
| S5_table.tex | circles                   | S5 Table          | 1125-1180 |
| S6_table.tex | moons                     | S6 Table          | 1182-1237 |
| S7_table.tex | swiss_rolls               | S7 Table          | 1239-1297 |
| S8_table.tex | gaussians_samples         | S8 Table          | 1303-1359 |
| S9_table.tex | gaussians_dimensionality  | S9 Table          | 1361-1417 |

All eight map one to one; there is no ambiguity in the mapping.

Note on the Klein estimator fix (commits 3b5f793, 137dc8c): the eight
fragments below come from the simulated benchmark scenarios (gaussians
through gaussians_dimensionality, registered in `src/benchmarks/_registry.py`
— a separate registry from `STUDIES` in `_studies.py`, which holds only the
klein and levine case studies), so that fix does not touch them. Checked
directly rather than assumed: neither this note nor `S2_table.tex`-
`S9_table.tex` mentions "klein", "levine", "agglomerative", or "kmeans"
anywhere, and neither `src/benchmarks/tables.py` nor
`src/benchmarks/_tables.py` references Klein, Levine, or `_studies.py` at
all. No rows in this note change.

## A structural pattern common to all eight fragments

Rather than repeat this eight times below, it is stated once here and
referenced per fragment. Every regenerated fragment differs from its
manuscript table in the same four ways, none of which are about run scale:

1. Table shape. The manuscript renders each experiment as two side-by-side
   sub-tables (mean ARI, then k-recovery proportion, sharing one row order).
   The generated fragment is a single merged table with ARI and k-recovery
   as a paired two-column group per axis value
   (`render_grouped_tex`, `src/benchmarks/_tables.py:142-191`). The same two
   statistics are present for the same axis values in both; only the LaTeX
   layout differs.
2. Missing row. Every manuscript table's first data row is "Baseline
   (Oracle)" (blank k-recovery cells, since the oracle knows k by
   construction). No generated fragment has this row: manuscript tables have
   13 method rows, generated fragments have 12. `table_metrics()`
   (`src/benchmarks/tables.py:58-62`) draws only from `CARVE_METRICS_ALL` and
   `CVI_METRICS`; `baseline_oracle` is a third, separate metric
   (`src/benchmarks/_registry.py:63`) that neither tuple includes, so it
   never reaches the table writer even though the run pipeline computes it.
3. Row order. The manuscript groups rows by category (Baseline; CARVE
   headline scores; CVIs; supplementary ARI-rule variants) separated by
   `\midrule`s, and marks the best and second-best value per column in bold
   and underline (manuscript text, line 940: "Rows are ranked by
   ARI($\hat k$) mean; bold = best, underline = second best per column,
   excluding oracle"). The generated fragment is one flat, ungrouped list,
   in `table_metrics()`'s order (CARVE metrics sorted alphabetically by
   internal key, then the four CVIs in their declared tuple order), with no
   bold/underline highlighting anywhere.
4. Column order. For the six difficulty-axis tables (S2-S7), the manuscript
   orders columns easy, medium, hard (the natural difficulty progression);
   the generated fragment orders them alphabetically by label text — easy,
   hard, medium — in every one of the six. For the two scaling-axis tables
   (S8, S9), the manuscript uses the literal axis values as column headers
   in ascending order (1000/5500/10000 for S8; 50/525/1000 for S9); the
   generated fragment uses the word labels, alphabetically ordered (end,
   middle, start) rather than in the low-to-high progression the labels
   were defined to represent.

The caption difference the task brief anticipates — the generated
`$k^\star = 5$` (or `$k^\star = ...$`) clause, absent from the hand-edited
manuscript captions — is present in all eight fragments and is exactly the
expected drift; it is not repeated as a finding per fragment below. One
manuscript-side inconsistency worth flagging while on the subject: S7 Table
uniquely already carries a `$k^* = 5$` header row in the committed
manuscript (`CARVE_manuscript.tex:1250`, `:1274`), unlike S2-S6, S8, and S9,
where it was stripped by hand. That inconsistency predates this work and is
not something the regenerated fragment introduced.

## Fragments verified

### S2_table.tex — gaussians / S2 Table

Differs structurally per the common pattern above (merged table, 12 rows
vs. 13, flat row order, easy/hard/medium column order vs. easy/medium/hard).
Caption content otherwise matches ("Gaussian mixtures: ARI at the selected k
and k-recovery by difficulty"), and the same 12 methods appear in both
(Baseline aside).

### S3_table.tex — t_dist / S3 Table

Same common pattern. Caption content matches ("t-distributed clusters: ARI
at the selected k and k-recovery by difficulty").

### S4_table.tex — t_dist_noise / S4 Table

Same common pattern. Caption content matches ("t-distributed clusters with
nuisance dimensions...").

### S5_table.tex — circles / S5 Table

Same common pattern. Caption content matches ("RFF-embedded circles...").

### S6_table.tex — moons / S6 Table

Same common pattern. Caption content matches ("RFF-embedded moons...").

### S7_table.tex — swiss_rolls / S7 Table

Same common pattern, plus the pre-existing manuscript-side k-star
inconsistency noted above (S7 already has a header row the others do not).
Caption content matches ("RFF-embedded Swiss rolls...").

### S8_table.tex — gaussians_samples / S8 Table

Same common pattern, with the added column-header difference described in
point 4 above: manuscript columns are headed 1000/5500/10000, generated
columns are headed end/middle/start (alphabetical, not low-to-high). Caption
content matches ("Gaussian mixtures over sample size...").

### S9_table.tex — gaussians_dimensionality / S9 Table

Same common pattern and the same numeric-vs-label column-header difference
as S8 (manuscript: 50/525/1000; generated: end/middle/start). Caption
content matches ("Gaussian mixtures over embedding dimension...").

## Summary

All eight fragments map cleanly to their manuscript tables by scenario name,
and all eight carry the same four structural differences from the published
layout: a single merged ARI/k-recovery table instead of two side-by-side
sub-tables, a missing Baseline (Oracle) row, no row grouping or
best/second-best highlighting, and axis-value columns ordered alphabetically
by label rather than in the manuscript's natural (difficulty or magnitude)
progression. None of these four are explained by the reduced sample size;
they would reproduce identically at full scale. The one difference the task
brief anticipated — the generated k-star caption clause — is present and
correct in all eight and is not a new finding.
