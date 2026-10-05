# hECA on Longleaf

The full-scale hECA run, 695,304 cells, as three SLURM jobs on UNC's Longleaf
cluster: embed, calibrate and fit. Design:
docs/superpowers/specs/2026-10-01-heca-longleaf-runtime-design.md. Commands
run from the repository root on Longleaf unless marked "on the laptop".

Longleaf facts used here, from https://help.rc.unc.edu/getting-started-on-longleaf/
(read 2026-10-01): SLURM; the general partition, which is the default, allows
jobs up to 11 days; home is capped at 50 GB; /work/users holds 8 TB per user;
the datamover partition is for transfers.

## 0. Access

- Request a Longleaf account from UNC Research Computing (see the page above).
- On the laptop, push the branch the run uses: `git push origin heca-longleaf`.

## 1. Code and environment, once

```bash
ssh <onyen>@longleaf.unc.edu
cd /work/users/<o>/<n>/<onyen>
git clone git@github.com:DataSlingers/CARVE.git carve   # needs a GitHub SSH key on Longleaf
cd carve
git checkout heca-longleaf
curl -LsSf https://astral.sh/uv/install.sh | sh          # installs uv into ~/.local/bin
export UV_CACHE_DIR=/work/users/<o>/<n>/<onyen>/.uv-cache
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
.venv/bin/python -c "
import glob, h5py
paths = sorted(glob.glob('data/hECA/ATAC-*.h5ad'))
for path in paths:
    h5py.File(path, 'r').close()
print(len(paths), 'files open')"
```

## 3. Embed and calibrate

```bash
RUN_DIR=/work/users/<o>/<n>/<onyen>/carve-runs/heca/$(date +%Y%m%d)-$(git rev-parse --short HEAD)
mkdir -p "$RUN_DIR"
embed=$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/embed.sbatch)
sbatch --dependency=afterok:"$embed" --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/calibrate.sbatch
squeue -u "$USER"
```

The embed job requests 8 cores, 200 GB and one day; if it runs out of memory,
resubmit with a larger `--mem` on the command line. Calibration requests
4 cores, 64 GB and one day.

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
  end (`forest_fine`, the most clusters within the upper target), the largest
  forest a fit worker trains, so it bounds one worker's memory. If the
  projection exceeds 8 days (192 h), decide before fitting: fewer resamples
  (100 to 50) first, then fewer resolutions.

On the laptop, set `STUDIES["heca"].resolutions` to the proposed values as a
literal tuple, with a comment naming the calibration run, run
`tests/benchmarks/test_studies.py`, commit and push. Here, `git pull`.

## 5. Fit

```bash
sbatch --export=ALL,RUN_DIR="$RUN_DIR" slurm/heca/fit.sbatch
.venv/bin/python -m benchmarks.heca status --run-dir "$RUN_DIR"
```

`status` reads the fit's timing rows and can run at any time; it is light
enough for a login node. The first configuration completes within hours. If
the abort rule triggers (projected total over 10 days), cancel with
`scancel <jobid>` and revisit the grid or the resample count. The fit saves
only when it finishes, so a killed fit leaves none of the outputs the stage
refuses to overwrite: resubmit it with the same `sbatch` command, without
`--force`, and it reruns from the start.

SLURM's accounting can lag the step's end: if `fit/sacct.txt` is empty or
lacks MaxRSS, rerun fit.sbatch's `sacct` line from a login node once the job
has finished, with the job id in place of `${SLURM_JOB_ID}`, writing to the
same file.

## 6. Copy back, on the laptop

```bash
cd ~/GitHub/CARVE/code
rsync -av <onyen>@longleaf.unc.edu:/work/users/<o>/<n>/<onyen>/carve-runs/heca/<run> results/runs/heca/
rsync -av "<onyen>@longleaf.unc.edu:/work/users/<o>/<n>/<onyen>/carve/data/hECA/.heca_cache_*" data/hECA/
```

The second command brings the embedding cache, about 280 MB, so the notebook
loads the embedding the fit ran on. Then run
notebooks/case_studies/hECA.ipynb.
