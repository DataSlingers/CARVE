# SNR calibration: regenerated anchors versus the published set

Date: 2026-09-02
Produced by: `benchmarks._calibrate.calibrate_scenario`, run over the six
difficulty scenarios.
Raw output: `2026-09-01-calibration-comparison.json`

## What was run

Settings: `max_iter=12`, `n_seeds=20`, `random_state=42`. Total wall clock 1046
seconds, dominated by `circles` (432 s) and `moons` (502 s), whose spectral
oracle fits cost roughly 4.5 seconds each against 0.05 for k-means.

`n_seeds=20` matches the benchmark's seed count, so the achieved ARI is
estimated over the same datasets the benchmark uses. `max_iter=12` bisects
`cluster_scale` over [0.1, 12.0] to a resolution of about 0.003, far finer than
anything that could matter against ARI bands 0.1 wide; the cap binds only when
a band is unreachable. `random_state=42` is the published seed, which is what
makes the calibration and benchmark datasets coincide as the manuscript states.

This is a documented replacement, not a reproduction. The notebook that
produced the published anchors was lost, and those anchors move four parameters
at once and non-monotonically across difficulty levels, which is not the
signature of a single ordered sweep. The procedure here holds every anchor
parameter fixed except `cluster_scale` and bisects that one parameter.

## Which anchors landed in band

Sixteen of eighteen. The two failures are `circles/easy` and `moons/easy`, and
they fail in the same way: bisection drove `cluster_scale` to the bottom of the
search interval (0.103) and still reached only about 0.37 mean ARI against a
target band of 0.9 to 1.0.

That is informative rather than a defect in the search. For the two nonlinear
embedded scenarios, `cluster_scale` alone cannot buy separability at the easy
end — difficulty there is governed by the embedding and correlation parameters
(`embed_param`, `corr_strength`) that this single-parameter search deliberately
holds fixed. It is direct evidence for the plan's premise that the original
calibration moved several parameters together and cannot be recovered by
sweeping one.

## How far the regenerated values sit from the published ones

Not always commensurably. Every published anchor in these six scenarios sets
`cluster_scale` to a per-cluster list rather than a scalar, while the search
fits a single scalar. The JSON prints the published value as-is; no attempt is
made to reduce a list to a number for comparison, because any such reduction
would be an invented statistic rather than a measurement.

Taking the first element of each published list as a rough locator, the
regenerated scalars are in the same order of magnitude but not close:
`gaussians/easy` 3.08 against a published 4.0, `t_dist/easy` 1.59 against 3.5,
`swiss_rolls/medium` 1.22 against 2.0. The regenerated values are mostly lower,
which is expected — a single scalar has to do work that the published
per-cluster lists spread across five clusters.

A further caveat specific to `circles` and `moons`: both use spectral
clustering at n=1500, which in this codebase is not reproducible run to run.
`carve.cluster.SpectralClustering` accepts a `random_state` but never threads it
into the sparse ARPACK eigensolver, whose starting vector comes from the global
numpy RNG. Their achieved-ARI figures above are therefore single draws from an
unstable process, and repeating this sweep will not reproduce them exactly.

## Recommendation

Do not switch `ACTIVE_ANCHORS`. It remains `PUBLISHED_ANCHORS`, and this note
is not a reason to change that.

Three reasons. Switching would change every published number in the manuscript,
which is an author's decision and not a consequence of running a calibration
script. The regenerated set is also strictly less expressive than the published
one — a scalar per anchor against a per-cluster list — so adopting it would
discard information rather than restore it. And it does not even cover the
space: two of the eighteen anchors cannot be reached at all by this search.

The useful outcome here is the S3 Text update. The manuscript should describe
the procedure that exists and is reproducible, rather than the lost one. Note
that `_calibrate.py`'s module docstring currently states that S3 Text "is
updated to describe this procedure" in the present tense; that edit has not been
made, and this repository cannot make it, since the manuscript directory is
read-only here. Either the docstring needs a qualifier or the manuscript needs
the edit.
