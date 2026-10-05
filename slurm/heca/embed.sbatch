#!/bin/bash
#SBATCH --job-name=heca-embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=heca-embed-%j.out
#
# The pooled embedding of every annotated hECA cell: two chunked passes over
# the organ files, then normalization, peak selection and PCA in memory. Its
# cache in data/hECA/ serves every later stage. Submit from the repository
# root with RUN_DIR set; see slurm/heca/README.md. Arguments after the
# script's name, such as --force, are passed to the stage.

set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR to the run directory}"
source .venv/bin/activate
python -m benchmarks.heca embed --run-dir "$RUN_DIR" "$@"
