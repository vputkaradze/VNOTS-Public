#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ( "$1" != "piston" && "$1" != "so3" ) ]]; then
    echo 'Usage: sbatch --array=0-5 scripts/train_array.sh piston (SO3: --array=0-3)' >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=2
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
fi
python -u run.py "$1" train --case "${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID}"
