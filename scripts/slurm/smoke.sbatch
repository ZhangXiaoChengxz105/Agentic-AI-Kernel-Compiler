#!/bin/bash
# One-off GPU smoke test for Tasks 1.3-1.6 (~10-20 min of one A100):
#   1. record GPU name + driver version (missing piece of notes/env.md)
#   2. torch sees the GPU; driver-API kernel launch works
#   3. GPU unit tests (tests/test_harness.py GpuTests)
#   4. fixed suite on every variant of every op (config A baseline: which bugs does it miss?)
#   5. benchmark ref/opt vs PyTorch for every op
#   6. change-aware GPU tests (Task 2) for the 11 injected CUDA bugs (config C's test part)
#   7. GLM-written CUDA softmax kernels (pilot): all-category tests + benchmark vs our opt and PyTorch
#   8. tirx-lite softmax kernels (hand-written + GLM) on the GPU: correctness + speed
#   sbatch scripts/slurm/smoke.sbatch          # ASK BEFORE SUBMITTING
#SBATCH --job-name=cs521-smoke
#SBATCH --account=bidr-delta-gpu
# partition: interactive A100 (max 1 h). The batch partition gpuA100x4 had a ~13-day queue on 2026-11-15.
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
CS521="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$CS521"
module load cudatoolkit/26.5_13.2 2>/dev/null || true
source env.sh
source envs/tirx/bin/activate

echo "== [1] GPU / driver"
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv | tee logs/slurm/gpu_info.csv
nvcc --version | tail -1

echo "== [2] torch"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"

echo "== [3] GPU unit tests"
python -m unittest tests.test_harness -v 2>&1 | tail -25

echo "== [4] fixed suite (config A) on all variants"
for op in softmax gemm layernorm; do python -m harness.cli check --op "$op" --repeat 3; done

echo "== [5] benchmark"
for op in softmax gemm layernorm; do python -m harness.cli bench --op "$op"; done
echo "== [6] change-aware GPU tests for the injected bugs"
for f in results/change_aware/tests/*_bug_*.json; do
  bug=$(basename "$f" .json); op=${bug%%_bug_*}
  python -m harness.cli check --op "$op" --variants "$bug" --tests "$f"
done

echo "== [7] GLM CUDA kernels"
P=results/pilot
EXTRA=(--extra glm_A_r1_FLTMAX=$P/20261115-203841/softmax_glm_r1.cu --extra glm_A_r3=$P/20261115-203841/softmax_glm_r3.cu
       --extra glm_B1_r1_racy=$P/20261115-204303/softmax_glm_r1.cu --extra glm_B1_r3=$P/20261115-204303/softmax_glm_r3.cu
       --extra glm_B2_r1=$P/20261115-204348/softmax_glm_r1.cu --extra glm_B3_r1=$P/20261115-204412/softmax_glm_r1.cu)
python -m harness.cli check --op softmax --variants softmax_opt "${EXTRA[@]}" --tests results/change_aware/tests/softmax_all_categories.json
python -m harness.cli bench --op softmax "${EXTRA[@]}"

echo "== [8] tirx-lite kernels on the GPU"
RUSTUP_TOOLCHAIN=1.95.0 python scripts/gpu_tirx_check.py
echo "== done at $(date -Is)"
