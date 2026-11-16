# First A100 run (2026-11-15, Slurm job 24032011)

`sbatch scripts/slurm/smoke.sbatch` on `gpuA100x4-interactive`, node gpua001, **6 min 22 s** (limit 45 min).
Full log: `results/gpu/smoke_24032011.log`; per-step JSON: `results/harness/{check,bench}_*_20261115-21*.json`,
`results/harness/tirx_gpu_20261115-211846.json`.

## Environment (confirmed on the compute node)
NVIDIA A100-SXM4-40GB, compute capability 8.0, **driver 595.71.05**, 40 GiB; torch 2.14.1+cu132 sees the GPU;
all 20 harness unit tests pass (incl. the 3 GPU tests) in 32 s.

## 1. Bug detection on the GPU vs VOLTA (11 injected CUDA bugs)
fixed = configuration A's fixed suite (power-of-two sizes, N(0,1), ×3 repeats); CA tests = Task 2's targeted GPU tests
(`results/change_aware/tests/<bug>.json`); VOLTA columns from `scripts/run_change_aware.py` (CPU).

| bug | fixed tests | VOLTA, default shape | VOLTA, change-aware shapes | CA GPU tests |
|---|---|---|---|---|
| softmax_nosync | ✅ 3/3 fail | ✅ RACE | ✅ RACE | ✅ 2/2 fail |
| softmax_tail | ❌ pass | ❌ EQUIV | ✅ NOT_EQUIV (N=200) | ✅ 12/12 fail |
| softmax_offbyone | ✅ | ✅ NOT_EQUIV | ✅ | ✅ 6/6 |
| softmax_nomax | ❌ pass | ❌ EQUIV (blind) | ❌ EQUIV (blind) | ✅ 3/4 (NaN/Inf for large, neg_large, mixed) |
| gemm_nosync | ⚠️ 1/3 fail (only 512³) | ✅ RACE | ✅ RACE | ✅ 2/2 (one via **nondeterminism**: repeats differ) |
| gemm_ktail | ❌ pass | ❌ EQUIV | ✅ NOT_EQUIV (K=40) | ✅ 8/10 |
| gemm_guard | ❌ pass | ❌ EQUIV | ✅ RACE (12×12) | ✅ 5/5 (**canary**: out-of-bounds store) |
| layernorm_nosync | ❌ pass | ✅ RACE | ✅ RACE | ❌ 2/2 pass (×5 repeats) |
| layernorm_tail | ❌ pass | ❌ EQUIV | ✅ NOT_EQUIV (N=100) | ✅ 12/12 |
| layernorm_offbyone | ✅ | ✅ NOT_EQUIV | ✅ | ✅ 5/6 |
| layernorm_onepass | ❌ pass | ❌ EQUIV (blind) | ❌ EQUIV (blind) | ✅ 1/4 (NaN for offset inputs) |

| configuration | detected |
|---|---|
| A: fixed tests | **4/11** |
| B: fixed tests + VOLTA (default shape) | **5/11** |
| C: fixed tests + VOLTA (change-aware shapes) + change-aware tests | **11/11** |
Complementarity: only VOLTA caught `layernorm_nosync`; only the tests caught the two numerical bugs.

## 2. GLM-written kernels on the GPU
CUDA (24 tests: fixed + all change-aware categories, `results/change_aware/tests/softmax_all_categories.json`),
speed at rows=4096, N=1024 (median of 50, L2 flushed; PyTorch 0.0370 ms):

| kernel | VOLTA verdict (pilot) | GPU tests | time | vs torch |
|---|---|---|---|---|
| our softmax_opt | EQUIV | 24/24 | 0.0363 ms | 1.02× |
| glm_A_r1 (−FLT_MAX) | UNSUPPORTED (i128 overflow: verifier false alarm) | **24/24** | **0.0343 ms** | **1.08×** |
| glm_A_r3 (accepted by the pilot) | EQUIV at N=256/200/100 | **12/24 FAIL** | — | — |
| glm_B1_r1 (racy draft) | **RACE** | **24/24 pass** (incl. ×5 repeats) | 0.0404 ms | 0.92× |
| glm_B1_r3 | EQUIV | 24/24 | 0.0407 ms | 0.91× |
| glm_B2_r1 | EQUIV | 24/24 | 0.0366 ms | 1.01× |
| glm_B3_r1 | EQUIV | 24/24 | 0.0394 ms | 0.94× |

tirx-lite (8 cases incl. n=1, 129, 1000, 4096, large/neg_large/constant inputs; `scripts/gpu_tirx_check.py`):
| kernel | GPU cases | time | vs torch |
|---|---|---|---|
| txl_ref (hand-written) | 8/8 | 0.0341 ms | 1.08× |
| glm_C, glm_D1, glm_D2, glm_D3 | 8/8 each | 0.0317–0.0320 ms | **1.15–1.16×** |

### Two naturally occurring agent bugs (not injected)
1. **glm_B1_r1: real shared-memory race.** VOLTA reported it (t0 write vs t32 read on Shared[0x0]); every GPU test
   passed, even ×5 repeats. VOLTA-only detection.
2. **glm_A_r3: dropped term in a 4×-unrolled sum** (`l = (e0 + e1) + (e2 + l)`, e3 computed but unused), introduced
   while GLM rewrote the kernel in round 2 to work around the VOLTA overflow. The unrolled loop only runs for
   N > 384, so VOLTA at N ≤ 384 said EQUIV (confirmed: N=384 EQUIV, N=385 NOT_EQUIV, 385/385 elements differ) and the
   pilot accepted it. The GPU tests with N = 1000/1024/4096 caught it. Fix applied: Task 2 now always adds a "deep"
   VOLTA shape (softmax N=584, gemm K=72) that exercises unrolled / multi-iteration paths.

## 3. Our kernels vs PyTorch (rows=4096, N=1024; GEMM 1024³)
| op | ref | opt | PyTorch |
|---|---|---|---|
| softmax | 0.645 ms | 0.038 ms (0.97×) | 0.037 ms |
| layernorm | 0.336 ms | 0.035 ms (**1.19×**) | 0.042 ms |
| gemm (fp32) | 0.890 ms (2.4 TFLOP/s) | 0.609 ms (3.5 TFLOP/s, 0.28×) | 0.172 ms (12.5 TFLOP/s) |
GEMM leaves the most room for an optimizing agent.

## Caveats
- The 11 bugs, the classifier and the test generator were all written by the same person; the 11/11 for C is an
  upper bound. Re-measure with bugs written by someone else and with agent-produced revisions (Task 5).
- One GPU run (no repetition across jobs); timings are medians of 50 launches on one node.
