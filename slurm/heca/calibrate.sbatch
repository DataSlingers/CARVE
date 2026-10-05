#!/bin/bash
#SBATCH --job-name=heca-calibrate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=heca-calibrate-%j.out
#
# Scans Leiden resolution for both settings, times every component the fit
# repeats, proposes the grid and projects the fit's runtime. Every timed
# component runs single-threaded, as each worker of the fit does. Exits 1
# when no grid satisfies the rule; calibration.json then says why.
# Arguments after the script's name, such as --force, are passed to the
# stage.

set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR to the run directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source .venv/bin/activate
python -m benchmarks.heca calibrate --run-dir "$RUN_DIR" "$@"
