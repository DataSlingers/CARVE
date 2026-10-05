#!/bin/bash
#SBATCH --job-name=heca-fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=11-00:00:00
#SBATCH --output=heca-fit-%j.out
#
# The CARVE fit on every cell, alone on one node so no other job shares its
# memory bandwidth. The stage picks its worker count from the node's
# physical cores and calibration.json. srun makes the fit a job step, so
# SLURM's accounting for it is final when the step ends and is saved beside
# the run. srun does not inherit --cpus-per-task from sbatch, so the step
# asks for every CPU on the node explicitly. Lower --time on the command
# line to the projection plus a margin to schedule sooner. Arguments after
# the script's name, such as --n-jobs 32, are passed to the stage.

set -uo pipefail
: "${RUN_DIR:?Set RUN_DIR to the run directory}"
source .venv/bin/activate
# --export=ALL carries the submitting shell's thread counts into the job,
# and joblib passes a parent's value on to every worker, oversubscribing
# them. Once they are unset, CARVE's core budget sets them per worker.
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS

status=0
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_ON_NODE}" \
    python -m benchmarks.heca fit --run-dir "$RUN_DIR" "$@" || status=$?
mkdir -p "$RUN_DIR/fit"
sacct -j "${SLURM_JOB_ID}.0" \
    --format=JobID,Elapsed,TotalCPU,MaxRSS,AveRSS,NCPUS -P > "$RUN_DIR/fit/sacct.txt"
exit "$status"
