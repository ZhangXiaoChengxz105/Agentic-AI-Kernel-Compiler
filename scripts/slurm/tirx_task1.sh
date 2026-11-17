#!/bin/bash
# Task 1 baseline for the tirx-lite LayerNorm and GEMM kernels (GPU tests, NumSim, VOLTA, speed) + their GPU unit tests.
# Run on an A100 node through the generic job (one job at a time for the user; see CLAUDE.md):
#   sbatch --time=00:30:00 --job-name=tirx_task1 scripts/slurm/gpu_job.sbatch bash scripts/slurm/tirx_task1.sh
# Output: results/eval/tirx_task1.json, results/eval/volta_tirx_{layernorm,gemm,softmax_task1}.json,
#         results/perf/tirx_task1_<ts>.json, results/perf/bench_{layernorm,gemm}_tirx_<ts>.json
cd "${CS521:-$(cd "$(dirname "$0")/../.." && pwd)}"
echo "== tirx_task1.py"; python scripts/experiments/tirx_task1.py "$@"
echo "== akc bench --lang tirx"
akc bench --workload layernorm --lang tirx
akc bench --workload gemm --lang tirx
echo "== CUDA layernorm reference on the offset test that the plain serial tirx reference failed (job 24061076)"
mkdir -p build/tmp
python -c "import json; from akc.task.testcase import TestCase; print(json.dumps([TestCase('layernorm', {'rows': 64, 'N': 1000}, 'offset', seed=130, tag='numerical').to_dict()]))" > build/tmp/ln_offset_test.json
akc check --workload layernorm --variants layernorm_ref,layernorm_opt --tests build/tmp/ln_offset_test.json
echo "== unit tests (tirx kernels, GPU part included)"
python -m unittest tests.validation.test_tirx_kernels -v
