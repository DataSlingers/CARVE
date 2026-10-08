#!/bin/bash
#SBATCH --job-name=heca-fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=amd
#SBATCH --cpus-per-task=384
#SBATCH --mem=1400G
#SBATCH --time=11-00:00:00
#SBATCH --output=heca-fit-%j.out
#
# The CARVE fit on every cell, on one of general_big's AMD nodes (192
# physical cores, 384 hardware threads, 1.5 TB). Longleaf's submit filter
# rejects --exclusive and a named partition, and Slurm routes a job by what
# it requests (Research Computing, 2026-10-08). 1400G is above the general
# partition's per-node cap of 232,448 MB, so the job goes to general_big,
# and it covers 100 workers at the 12 GB one worker peaked at on the grid's
# top configuration. Asking for all 384 CPUs keeps other jobs, and their
# memory traffic, off the node. Override --mem or --cpus-per-task on the
# command line for another node. The stage picks its worker count from the
# node's physical cores and calibration.json unless --n-jobs is given.
# srun makes the fit a job step, so SLURM's accounting for it is final when
# the step ends and is saved beside the run. srun does not inherit
# --cpus-per-task from sbatch, so the step asks for every CPU the job holds
# explicitly. Lower --time on the command line to the projection plus a
# margin to schedule sooner. Arguments after the script's name, such as
# --n-jobs 100, are passed to the stage.

set -uo pipefail
: "${RUN_DIR:?Set RUN_DIR to the run directory}"
source .venv/bin/activate
# --export=ALL carries the submitting shell's thread counts into the job,
# and joblib passes a parent's value on to every worker, oversubscribing
# them. Once they are unset, CARVE's core budget sets them per worker.
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
# Write CARVE's per-configuration lines to the log as they happen rather
# than when Python's output buffer fills.
export PYTHONUNBUFFERED=1

status=0
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_ON_NODE}" \
    python -m benchmarks.heca fit --run-dir "$RUN_DIR" "$@" || status=$?
mkdir -p "$RUN_DIR/fit"
sacct -j "${SLURM_JOB_ID}.0" \
    --format=JobID,Elapsed,TotalCPU,MaxRSS,AveRSS,NCPUS -P > "$RUN_DIR/fit/sacct.txt"
exit "$status"
