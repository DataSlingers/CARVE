# hECA on Longleaf

The full-scale hECA run, 695,304 cells, as three SLURM jobs on UNC's Longleaf
cluster: embed, calibrate and fit. Design:
docs/superpowers/specs/2026-10-01-heca-longleaf-runtime-design.md. Commands
run from the repository root on Longleaf unless marked "on the laptop".

Longleaf facts used here, from https://help.rc.unc.edu/getting-started-on-longleaf/
and https://help.rc.unc.edu/slurm-guide (read 2026-10-05): SLURM, single-node
jobs only; the general partition, which is the default, allows jobs up to
11 days; the interact partition runs short interactive sessions, up to 8 hours;
the datamover partition is for transfers; hyperthreading is enabled; home is
capped at 50 GB and is not for heavy I/O; since 8/14/2026 the default user
space is /hickory/users/<o>/<n>/<onyen> (2 TB, 3 million files), and new
/work/users space is no longer created; sacct reports MaxRSS, and seff
summarizes a finished job. Code does not run on the login nodes: anything
beyond editing, git and job submission goes through SLURM, including the
quick checks below. Off campus, connect to the UNC VPN first.

Below, <o> and <n> are the first and second letters of your onyen. If you
already had /work/users space, it serves in place of /hickory/users.

## 0. Access

- Request a Longleaf account from UNC Research Computing (see the page above).
- On the laptop, push main, which the run uses: `git push origin main`.
- On the first login, look at what the partitions offer and what limits
  your account carries. Do not pass `--partition` for general, general_big,
  hov or spill: Slurm routes a job to one of them by what it requests, and
  the submit filter rejects a named one, as it does `--exclusive` (Research
  Computing, 2026-10-08). The general partition caps a job at 232,448 MB
  (MaxMemPerNode), so the fit, which asks for 1400G and all 384 CPUs of an
  AMD node, goes to general_big. A per-user core or memory cap below that
  request would keep the fit pending; if so, ask research@unc.edu.

  ```bash
  sinfo -p general,general_big -o "%12P %10D %6c %8m %20f"   # nodes, CPUs, memory (MB), features
  scontrol show partition general | grep MaxMemPerNode
  sacctmgr show association where user=$USER format=Account%40,GrpTRES%40,MaxTRES%40
  ```

## 1. Code and environment, once

```bash
ssh <onyen>@longleaf.unc.edu
cd /hickory/users/<o>/<n>/<onyen>
git clone git@github.com:DataSlingers/CARVE.git carve   # needs a GitHub SSH key on Longleaf
cd carve
curl -LsSf https://astral.sh/uv/install.sh | sh          # installs uv into ~/.local/bin
export UV_CACHE_DIR=/hickory/users/<o>/<n>/<onyen>/.uv-cache
uv venv --python 3.13 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[graph,benchmarks]"
```

## 2. Data, once

About 12 GB to download and 42 GB extracted, into data/hECA/ where the loader
looks.

```bash
mkdir -p data/hECA
sbatch -p datamover -t 12:00:00 -J heca-download --wrap '
  cd data/hECA &&
  for organ in Lung Brain Kidney Heart Thymus; do
    curl -fL --retry 5 -o "ATAC-$organ.h5ad.zip" \
      "https://zenodo.org/records/15627886/files/ATAC-$organ.h5ad.zip?download=1" &&
    unzip -o "ATAC-$organ.h5ad.zip" && rm "ATAC-$organ.h5ad.zip"
  done'
```

When it finishes, check that all five files open. A truncated extraction has
happened before; it fails here rather than hours into the embed job. If unzip
created a subdirectory, move the h5ad files up into data/hECA/ first.

```bash
srun -p interact -t 0:15:00 --mem=4g .venv/bin/python -c "
import glob, h5py
paths = sorted(glob.glob('data/hECA/ATAC-*.h5ad'))
for path in paths:
    h5py.File(path, 'r').close()
print(len(paths), 'files open')"
```

## 3. Embed and calibrate

```bash
RUN_DIR=/hickory/users/<o>/<n>/<onyen>/carve-runs/heca/$(date +%Y%m%d)-$(git rev-parse --short HEAD)
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > ~/.heca-run-dir
embed=$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/embed.sbatch)
sbatch --dependency=afterok:"$embed" --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/calibrate.sbatch
squeue -u "$USER"
```

The embed job requests 8 cores, 200 GB and one day; if it runs out of memory,
resubmit with a larger `--mem` on the command line. Calibration requests
4 cores, 64 GB and one day.

Every later stage must use this same run directory. Its name holds the date
and commit it was created with, so recomputing it after the gate's commit, or
on another day, gives a different path. In a new session, restore it from the
file written above:

```bash
RUN_DIR=$(cat ~/.heca-run-dir)
```

## 4. The gate

Read `$RUN_DIR/calibration.json`:

- `proposed_grid`, or `grid_error` when the rule failed. On a failure, widen
  `SCAN_RESOLUTIONS` in src/benchmarks/_heca_calibration.py as the message
  says, push, pull here, and rerun calibration with `--force`, which the
  script passes on to the stage:

  ```bash
  sbatch --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/calibrate.sbatch --force
  ```
- `projection.wall_clock_hours` at the fit node's physical core count, and
  `worker_peak_bytes`. The peak includes the forest fit on the scan's fine
  end (`forest_fine`, the most clusters any setting gives at or below the
  grid's top), the largest forest a fit worker trains, so it bounds one
  worker's memory. Calibrations written before 2026-10-08 measured a
  coarser fine end; for those, measure the top configuration separately
  before choosing `--n-jobs`. If the
  projection exceeds 8 days (192 h), decide before fitting: fewer resamples
  (100 to 50) first, then fewer resolutions.

On the laptop, set `STUDIES["heca"].resolutions` to the proposed values as a
literal tuple, with a comment naming the calibration run, run
`tests/benchmarks/test_studies.py`, commit and push. Here, `git pull`.

## 5. Fit

```bash
sbatch --test-only --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/fit.sbatch --n-jobs 100
sbatch --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/fit.sbatch --n-jobs 100
squeue -u "$USER" -o "%.10i %.12P %.8T %.10M %R"
srun -p interact -t 0:15:00 --mem=4g .venv/bin/python -m benchmarks.heca status --run-dir "$RUN_DIR"
```

`--test-only` submits nothing; it prints the partition, node and estimated
start the request would get, and the partition should be general_big.
`--n-jobs 100` runs all of a configuration's resamples at once. One worker
peaked at about 12 GB on the grid's top configuration (measured on an AMD
node, 2026-10-08), so 100 workers need about 1.2 TB of the 1400G the script
requests. Without `--n-jobs`, the stage picks one worker per physical core
the job holds, capped by the job's memory over calibration's
`worker_peak_bytes`.

A whole node can wait days in the queue. Half a node starts sooner and runs
each configuration's resamples in two rounds, about twice as long:

```bash
sbatch --test-only -c 100 --mem=700G --time=8-00:00:00 --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/fit.sbatch --n-jobs 50
sbatch -c 100 --mem=700G --time=8-00:00:00 --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/fit.sbatch --n-jobs 50
```

Options before the script's name override its `#SBATCH` lines. On 2026-10-08
the estimates were 2026-10-19 for the whole node and the same day for half.

`status` reads the fit's timing rows and can run at any time, as a short
job on the interact partition. The first configuration completes within hours. If
the abort rule triggers (projected total over 10 days), cancel with
`scancel <jobid>` and revisit the grid or the resample count. The fit saves
only when it finishes, so a killed fit leaves none of the outputs the stage
refuses to overwrite: resubmit it with the same `sbatch` command, without
`--force`, and it reruns from the start.

CARVE's own output goes to the job's log, heca-fit-<jobid>.out in the
repository root: the run's settings at the start, then one line per
configuration as it completes, with its stability and generalizability
(mean ARI and standard error). Configurations 1 to 15 are the 15-neighbor
setting, 16 to 30 the 50-neighbor setting, each coarsest first.

```bash
grep '\[CARVE\]' heca-fit-<jobid>.out
```

SLURM's accounting can lag the step's end: if `fit/sacct.txt` is empty or
lacks MaxRSS, rerun fit.sbatch's `sacct` line from a login node once the job
has finished, with the job id in place of `${SLURM_JOB_ID}`, writing to the
same file. `seff <jobid>` gives a one-screen summary of the job's CPU and
memory use.

## 6. Copy back, on the laptop

```bash
cd ~/GitHub/CARVE/code
rsync -av <onyen>@rc-dm.its.unc.edu:/hickory/users/<o>/<n>/<onyen>/carve-runs/heca/<run> results/runs/heca/
rsync -av "<onyen>@rc-dm.its.unc.edu:/hickory/users/<o>/<n>/<onyen>/carve/data/hECA/.heca_cache_*" data/hECA/
```

rc-dm.its.unc.edu is Research Computing's data mover, which UNC asks to be
used for transfers instead of the login nodes.

The second command brings the embedding cache, about 280 MB, so the notebook
loads the embedding the fit ran on. Then run
notebooks/case_studies/hECA.ipynb.
