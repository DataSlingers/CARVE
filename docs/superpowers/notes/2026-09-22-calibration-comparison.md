# SNR calibration: regenerated anchors versus the published set (round 2)

Date: 2026-09-23
Produced by: `benchmarks._calibrate.calibrate_scenario`, run over the six
difficulty scenarios, with the shift-invert spectral eigensolver
(09b810c), `n_trees=500` everywhere (2265c4a), `swiss_rolls` on spectral
(945593e), and the per-scenario calibration knobs of `_calibrate.py`
(7558d50) all in place.
Raw output: `2026-09-22-calibration-comparison.json`, keyed by scenario then
difficulty label, holding each cell's `calibrate_scenario` result exactly
as recorded (anchor, knob_value, achieved_ari, achieved_sd, target,
in_band).

## What changed since the 2026-09-01 note

That note ran a single-parameter `cluster_scale` bisection against the
unfixed eigensolver and left two anchors (`circles/easy`, `moons/easy`)
unreachable, recommending against adopting the result. Since then:

- The spectral eigensolver was fixed to shift-invert mode (09b810c);
  `carve.cluster.SpectralClustering` no longer depends on an
  unseeded ARPACK start vector, and circles/moons stopped saturating at
  ARI 1.000 at every anchor regardless of difficulty.
- `_calibrate.py` was rewritten to declare one knob per scenario
  (`CALIBRATION_KNOBS`, 7558d50): a `cluster_scale` multiplier for the four
  scale-driven scenarios (`gaussians`, `t_dist`, `t_dist_noise`,
  `swiss_rolls`), and an absolute `embed_param` (the random Fourier feature
  bandwidth) for `circles` and `moons`, which do not respond to
  `cluster_scale` at all once the solver is fixed.

This run regenerates all eighteen anchors under those two changes.

## Settings

`n_seeds=20` (matches the benchmark's own seed count, so calibration and
benchmark datasets coincide), `random_state=42`
(`PUBLISHED_RANDOM_STATE`, the benchmark's own seed derivation:
`benchmark_seed = seed + axis_idx * 10000 + random_state`), `max_iter=25`.
Each scenario ran its final estimator (`kmeans` for `gaussians`,
`agglomerative` for `t_dist` and `t_dist_noise`, `spectral` for `circles`,
`moons` and `swiss_rolls`) against the fixed eigensolver. The six
scenarios were calibrated in six separate process invocations (one per
scenario, well under the ten-minute-per-invocation budget this session
worked under); no scenario needed splitting by anchor label.

## Which anchors landed in band

All eighteen. Every scenario converged well inside `max_iter=25`
bisection steps; none exhausted the search interval or required a target
band to be widened.

## The eighteen cells

Knob is the multiplier applied to the published `cluster_scale` vector for
the four scale-driven scenarios, or the absolute `embed_param` for
`circles` and `moons`. Published/calibrated columns show the vector or
scalar the knob touched; every other anchor key (`cluster_size_dirichlet_alpha`,
`t_df`, `corr_strength`, `noise_dims`, and — for the scale-driven
scenarios — `embed_param`) is carried over from `PUBLISHED_ANCHORS`
unchanged.

| Scenario | Label | Knob | Published value | Calibrated value | ARI mean | ARI sd | In band |
|---|---|---|---|---|---|---|---|
| gaussians | easy | cluster_scale ×0.825 | [4.0]×5 | [3.300]×5 | 0.984 | 0.004 | yes |
| gaussians | medium | cluster_scale ×1.006 | [4.5]×5 | [4.528]×5 | 0.862 | 0.022 | yes |
| gaussians | hard | cluster_scale ×1.006 | [4.6]×5 | [4.629]×5 | 0.740 | 0.163 | yes |
| t_dist | easy | cluster_scale ×1.55 | [3.5, 1.0, 1.0, 1.0, 1.0] | [5.425, 1.55, 1.55, 1.55, 1.55] | 0.935 | 0.034 | yes |
| t_dist | medium | cluster_scale ×1.1875 | [3.0, 1.5, 1.0, 1.0, 1.0] | [3.563, 1.781, 1.188, 1.188, 1.188] | 0.811 | 0.114 | yes |
| t_dist | hard | cluster_scale ×1.006 | [4.0, 3.0, 2.0, 1.0, 1.0] | [4.025, 3.019, 2.013, 1.006, 1.006] | 0.726 | 0.137 | yes |
| t_dist_noise | easy | cluster_scale ×0.825 | [1.0]×5 | [0.825]×5 | 0.995 | 0.003 | yes |
| t_dist_noise | medium | cluster_scale ×1.006 | [1.1, 1.0, 1.0, 1.0, 1.0] | [1.107, 1.006, 1.006, 1.006, 1.006] | 0.846 | 0.037 | yes |
| t_dist_noise | hard | cluster_scale ×1.006 | [1.1, 1.0, 1.0, 1.0, 1.0] | [1.107, 1.006, 1.006, 1.006, 1.006] | 0.711 | 0.090 | yes |
| circles | easy | embed_param | 12.0 | 3.5 | 0.981 | 0.084 | yes |
| circles | medium | embed_param | 7.3 | 3.0 | 0.800 | 0.186 | yes |
| circles | hard | embed_param | 6.0 | 3.25 | 0.724 | 0.282 | yes |
| moons | easy | embed_param | 10.7 | 3.5 | 0.967 | 0.102 | yes |
| moons | medium | embed_param | 5.7 | 3.0 | 0.842 | 0.168 | yes |
| moons | hard | embed_param | 14.5 | 3.5 | 0.766 | 0.204 | yes |
| swiss_rolls | easy | cluster_scale ×1.55 | [1.0]×5 | [1.55]×5 | 1.000 | 0.000 | yes |
| swiss_rolls | medium | cluster_scale ×1.1875 | [2.0, 2.0, 1.0, 1.0, 1.0] | [2.375, 2.375, 1.188, 1.188, 1.188] | 0.889 | 0.144 | yes |
| swiss_rolls | hard | cluster_scale ×1.55 | [2.0, 2.0, 1.0, 1.0, 1.0] | [3.1, 3.1, 1.55, 1.55, 1.55] | 0.739 | 0.264 | yes |

Exact knob values and cluster_scale/embed_param entries, to full floating
point precision, are recorded in `src/benchmarks/_registry.py`'s
`CALIBRATED_ANCHORS` comments and in `2026-09-22-calibration-comparison.json`
cited above; the table rounds to three decimals for readability only.

## How far the regenerated values sit from the published ones

The scale-driven scenarios move by a small multiplier (0.825 to 1.55) on
top of the published per-cluster vector, so the unequal per-cluster shape
the published anchors used survives. `gaussians` barely moves (multipliers
within 0.825-1.006 of the published vector) because the published anchor
was already close to its band once the estimator and forest size were
fixed elsewhere on this branch; `t_dist`, `t_dist_noise` and `swiss_rolls`
move similarly modestly.

`circles` and `moons` move a great deal, because the published
`embed_param` values (6.0-14.5) were chosen against the unfixed
eigensolver, where spectral clustering's ARPACK call did not reliably
converge and both scenarios needed a large bandwidth to compensate. With
the fixed solver, `_calibrate.py`'s module-level comment documents that
ARI on these two scenarios is a threshold function of `embed_param`: near
zero below 2.0, exactly 1.000 above 4.0, and all three target bands fall
inside the narrow 2.7-3.5 transition. The regenerated values (3.0-3.5 on
both scenarios) sit inside that transition, as expected; the published
values (6.0-14.5) sit well past it, in the region where the fixed solver
now returns ARI 1.000 regardless of anchor label, which is exactly the
saturation the calibration was run to correct.

The calibrated `embed_param` values do not fall monotonically from easy to
hard: 3.5, 3.0 and 3.25 on `circles`, 3.5, 3.0 and 3.5 on `moons`. They
need not. `embed_param` is the only key the calibration moves; the others
(`cluster_scale`, `cluster_size_dirichlet_alpha`, `corr_strength`) are
carried over from `PUBLISHED_ANCHORS` and differ by level, so each level's
bandwidth is whatever puts its ARI in band given the rest of its anchor. On
`moons`, easy and hard share a bandwidth of 3.5 and still land at ARI 0.967
and 0.766; the rest of the hard anchor, among it a
`cluster_size_dirichlet_alpha` of 0.1 against easy's 0.67, makes the
difference. S3 Text
should present each level by its full anchor, not by `embed_param` alone,
so a reader does not take the non-monotone bandwidths for an error.

Achieved standard deviations are wide on `circles`, `moons` and
`swiss_rolls/hard` (0.14-0.28) because the transition band and the
few-cluster spectral fits are both sensitive to draw; the mean lands
inside the target band but individual datasets range further, as
`_calibrate.py`'s `oracle_ari_stats` docstring anticipates.

## Recommendation

Switch `ACTIVE_ANCHORS` to `CALIBRATED_ANCHORS`. Unlike the 2026-09-01
round, every anchor lands inside its documented band, the search covers
all eighteen cells with no unreachable ones, and the two knobs are
declared and reproducible rather than an unrecoverable four-parameter
notebook sweep. `PUBLISHED_ANCHORS` stays in the registry for comparison
and revert, and `ACTIVE_ANCHOR_SET_NAME` is recorded in every run
manifest so no artifact is ambiguous about which set produced it.
