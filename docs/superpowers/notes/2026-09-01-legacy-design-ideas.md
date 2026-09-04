# Two design ideas from the legacy benchmarking code, before it is deleted

Date: 2026-09-04

Source read:
`notebooks/benchmarking_code/legacy/benchmarking_simulation_helpers copy.py`
(147 lines total; the interpolation logic is lines 30-101, matching the task
brief's line range exactly)
`notebooks/benchmarking_code/legacy/benchmarking_runners copy.py`
(396 lines total; `true_cluster_counts` is declared at line 30, and the loop
that consumes it runs lines 69-86)

Both files are dead: zero importers anywhere in the tree, and each has a
relative import (`from .benchmarking_simulation_helpers import ...`) that no
longer resolves now that `benchmarking_code/legacy/` is not itself a package
member of the same name. `legacy/` has not been deleted yet — that is Step 3
of this task, gated on the regression harness (plan 1 Task 14), which has not
been run. This note exists so the two ideas below survive that eventual
deletion.

## 1. Interpolated continuous difficulty

`parse_difficulty_and_simulate` in `benchmarking_simulation_helpers copy.py`
took a `difficulty_index` in `[0, difficulty_levels - 1]` and a dict of three
named anchors (`settings_by_k['easy' | 'medium' | 'hard']`), each a mapping
from a `simulate_clusters` keyword (e.g. `cluster_scale`, `noise_dims`) to its
value at that anchor. At the three special indices (0, `difficulty_levels -
1`, and the middle index) it simulated directly from the matching anchor. At
every other index it linearly interpolated between the two neighboring
anchors, so difficulty was a continuous dial with `difficulty_levels`
resolution rather than three discrete points.

The interpolation formula, copied verbatim from lines 51-81:

```python
if difficulty_index > 0 and difficulty_index < int(round(difficulty_levels // 2)):
    # Linear interpolation between 'easy' and 'medium' settings
    frac = difficulty_index / int(round(difficulty_levels // 2))
    easy_settings = settings_by_k['easy']
    medium_settings = settings_by_k['medium']

    interpolated_settings = {}
    for key in easy_settings:
        v_easy = np.array(easy_settings[key])
        v_medium = np.array(medium_settings[key])
        interpolated = (1 - frac) * v_easy + frac * v_medium
        interpolated_settings[key] = interpolated.tolist()

elif (
    difficulty_index > int(round(difficulty_levels // 2))
    and difficulty_index < (difficulty_levels - 1)
):
    # Linear interpolation between 'medium' and 'difficult' settings
    frac = (
        (difficulty_index - int(round(difficulty_levels // 2)))
        / (difficulty_levels - 1 - int(round(difficulty_levels // 2)))
    )
    medium_settings = settings_by_k['medium']
    difficult_settings = settings_by_k['hard']

    interpolated_settings = {}
    for key in medium_settings:
        v_medium = np.array(medium_settings[key])
        v_difficult = np.array(difficult_settings[key])
        interpolated = (1 - frac) * v_medium + frac * v_difficult
        interpolated_settings[key] = interpolated.tolist()
```

`frac` is the fractional position between the two bracketing anchors, and
each setting is a plain convex combination `(1 - frac) * low + frac * high`,
applied element-wise via `np.array` so a setting that is itself a per-cluster
list (e.g. a `cluster_scale` list with one entry per cluster) interpolates
component-wise. `noise_dims`, being a dimension count, is rounded back to an
integer afterward (`int(np.floor(interpolated_settings["noise_dims"] +
0.5))`) since interpolating two integers can produce a fraction.

Where this landed in the rebuilt design: `benchmarks._types.Axis` (defined at
`src/benchmarks/_types.py:15`) already generalizes "a swept experimental
dimension" to arbitrary `(values, labels)` pairs, and
`benchmarks._registry.DIFFICULTY_AXIS` (defined at
`src/benchmarks/_registry.py:107-111`) currently instantiates it with exactly
three points:

```python
DIFFICULTY_AXIS = Axis(
    name="difficulty_level",
    values=(0, 1, 2),
    labels=("easy", "medium", "hard"),
)
```

`Scenario.sim_kwargs` (`src/benchmarks/_types.py:135-146`) resolves an axis
point by looking up `self.anchors[axis_label]` directly — it selects an
anchor, it does not interpolate between two. So the continuous scheme above
is genuinely absent from the rebuilt design today, not merely relocated.

Recovering it needs no new runner, though. `Axis` is already a plain
`(values, labels)` pair with no constraint that `values` be three points or
that consecutive points differ qualitatively; a continuous difficulty axis is
`Axis("difficulty", values=<continuous grid>, labels=<one label per value>)`
fed through a `Scenario` whose `sim_kwargs` interpolates between the
bracketing anchors instead of selecting one. Two real constraints apply: (a)
`Axis.__post_init__` (`src/benchmarks/_types.py:31-38`) requires
`len(values) == len(labels)` and forbids duplicate labels, so a continuous
grid needs one distinct label per grid point, not just three; and (b) the
interpolation itself would have to be added to `Scenario.sim_kwargs` (or a
sibling of it), since today that method only ever indexes `self.anchors`, it
never blends two of them. Both are additions to existing machinery, not a
new axis type or a new runner.

## 2. Multi-k-star sweep (varying the true cluster count within a run)

`benchmark_cluster_metrics` in `benchmarking_runners copy.py` took a
`true_cluster_counts: Sequence[int] = (3, 4, 5, 6)` parameter (declared at
line 30) and swept it as a third nested loop dimension, alongside difficulty
level and seed, so one call to the function produced results for every
combination of difficulty x true k x seed rather than a single ground-truth
cluster count.

The loop structure, copied verbatim from lines 69-86:

```python
total_steps = difficulty_levels * n_seeds_per_dataset * len(true_cluster_counts)
pbar = tqdm(total=total_steps, desc="Benchmarking", leave=True)
for difficulty_level in range(difficulty_levels):
    for true_k in true_cluster_counts:
        for seed in range(n_seeds_per_dataset): 
            # --- 0) Set seed ---
            benchmark_seed = seed + ((true_k - min(true_cluster_counts)) * 100) + (difficulty_level * 10000) + random_state
            plotting_dict = {}  # plotting
            
            # --- 1) Simulate data ---
            X, y = parse_difficulty_and_simulate(
                settings_by_k=settings_by_k[true_k],
                other_settings=other_settings,
                difficulty_levels=difficulty_levels,
                difficulty_index=difficulty_level,
                true_cluster_count=true_k,
                seed=benchmark_seed
            )
```

Two things are worth noting about how `true_k` is threaded through: it
indexes `settings_by_k[true_k]` to select a per-k family of anchor settings
(so different true cluster counts could use different anchor calibrations,
not just a different `k=` passed to the simulator), and it participates in
deterministic seed derivation (`(true_k - min(true_cluster_counts)) * 100`),
so results are reproducible per (difficulty, true_k, seed) cell without
sharing RNG state across cells.

Where this stands in the rebuilt design: `benchmarks._types.Scenario`
(`src/benchmarks/_types.py:78-93`) is a frozen dataclass with a single
`axis: Axis` field and a single `k_star: int = 5` field — one experimental
axis and one fixed ground-truth cluster count per scenario, not a sweep over
several. Producing results across multiple true k values today means running
several separate scenarios (one per k_star), not one run that sweeps k_star
internally the way the legacy loop did.

A correction to the task brief's stated reason for why this is hard: the
brief attributes the blocker to "`SweepSpec` being frozen to one parameter."
That description of `SweepSpec` is accurate as far as it goes —
`carve._sweep.SweepSpec` (`src/carve/_sweep.py:50-51`) is a
`@dataclass(frozen=True)` with a single `param: str` field, and its module
docstring states plainly: "A run uses exactly one sweep parameter. Comparing
k-based estimators with resolution-based estimators inside a single run is
not supported." That is CARVE's own one-hyperparameter-per-fit()-call
constraint (`n_clusters` versus `resolution`), and grepping the tree confirms
`SweepSpec` is used only inside `src/carve/` (`_grids.py`, `_output.py`,
`_runner.py`, `api.py`, `_sweep.py` itself) — `src/benchmarks/` never
imports it.

`true_k` (the ground-truth cluster count used to simulate data) is a
different quantity from what `SweepSpec` sweeps (the candidate `n_clusters`
or `resolution` values CARVE evaluates once handed a dataset). `SweepSpec`
does not gate a multi-k-star benchmarks sweep at all; a `CARVE.fit()` call
run against data with `true_k = 6` still internally sweeps its own
`candidate_k` exactly as it does for `true_k = 3`. The actual blocker is
`benchmarks._types.Scenario` holding one `axis: Axis` and one `k_star: int`,
both frozen dataclass fields, rather than something that varies true k
alongside (or instead of) the difficulty axis. Whether the fix is a second
axis field on `Scenario`, a family of scenarios generated from a k_star list,
or something else, it is a benchmarks-level `Scenario` design question, not
a small parameter addition, and not a change to `SweepSpec` or anything else
in `src/carve/`.
