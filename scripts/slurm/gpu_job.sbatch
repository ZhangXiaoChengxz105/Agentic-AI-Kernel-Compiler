#!/bin/bash
# Generic 1x A100 batch job: runs whatever command you pass after the script name.
#   sbatch scripts/slurm/gpu_job.sbatch python -m harness.cli check --op softmax
#   sbatch --time=00:10:00 scripts/slurm/gpu_job.sbatch python -m harness.cli bench --op gemm
# ASK BEFORE SUBMITTING (CLAUDE.md rule 3): this uses the shared bidr-delta-gpu allocation.
#SBATCH --job-name=cs521
#SBATCH --account=bidr-delta-gpu
# partition: interactive A100 (max 1 h). The batch partition gpuA100x4 had a ~13-day queue on 2026-11-15.
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
CS521="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$CS521"
module load cudatoolkit/26.5_13.2 2>/dev/null || true
source env.sh
source envs/tirx/bin/activate

echo "== job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [ "$#" -eq 0 ]; then echo "no command given"; exit 2; fi
echo "== running: $*"
"$@"
echo "== done at $(date -Is)"
