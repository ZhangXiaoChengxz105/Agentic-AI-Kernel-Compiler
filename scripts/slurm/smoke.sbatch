#!/bin/bash
# One-off GPU smoke test for Tasks 1.3-1.6 (~10-20 min of one A100):
#   1. GPU name + driver version
#   2. torch sees the GPU
#   3. full unit-test suite (GPU tests included)
#   4. configurations A / B / C on the injected CUDA bugs and the tirx-lite bugs (akc eval-bugs)
#   5. benchmark reference/optimized kernels vs PyTorch for every workload
#   6. GLM-written CUDA softmax kernels (pilot): all-category tests + benchmark
#   7. tirx-lite softmax kernels (hand-written + GLM) on the GPU: correctness + speed
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
source env.sh                      # activates envs/akc

echo "== [1] GPU / driver"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv | tee logs/slurm/gpu_info.csv
nvcc --version | tail -1

echo "== [2] torch"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"

echo "== [3] unit tests (GPU included)"
python -m unittest discover -s tests 2>&1 | tail -6

echo "== [4] configurations A/B/C on injected bugs"
akc eval-bugs --device gpu
akc eval-bugs --device gpu --lang tirx

echo "== [5] benchmarks"
for w in softmax gemm layernorm; do akc bench --workload "$w"; done

echo "== [6] GLM CUDA kernels"
P=results/pilot
EXTRA=(--extra glm_A_r1_FLTMAX=$P/20261115-203841/softmax_glm_r1.cu --extra glm_A_r3=$P/20261115-203841/softmax_glm_r3.cu
       --extra glm_B1_r1_racy=$P/20261115-204303/softmax_glm_r1.cu --extra glm_B1_r3=$P/20261115-204303/softmax_glm_r3.cu
       --extra glm_B2_r1=$P/20261115-204348/softmax_glm_r1.cu --extra glm_B3_r1=$P/20261115-204412/softmax_glm_r1.cu)
akc check --workload softmax --variants softmax_opt "${EXTRA[@]}" --tests results/change_aware/tests/softmax_all_categories.json
akc bench --workload softmax --variants softmax_opt "${EXTRA[@]}"

echo "== [7] tirx-lite kernels on the GPU"
akc tirx-gpu
echo "== done at $(date -Is)"
