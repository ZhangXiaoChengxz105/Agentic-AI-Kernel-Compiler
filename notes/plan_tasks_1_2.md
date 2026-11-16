# Tasks 1–2 (Xiaocheng): scope, status, plan, as of 2026-11-15

Decisions so far: agent model = GLM-5.3 (Lumen). CAKE has no public code; TIRx-harness works on sm_80 as a
CAKE-style substitute (not yet confirmed by the team). **Open decision: what the agent writes (TIRx vs CUDA)**, which Task 2 depends on.

Resources: Slurm account `bidr-delta-gpu` (749 of 3000 GPU-hours left, shared project balance);
partitions `gpuA100x4` (≤2 days) and `gpuA100x4-interactive` (≤1 h). Lumen: 10 coins, +2.4/day.

## Task 1: environment and baseline
| # | Must cover | Status | Remaining |
|---|---|---|---|
| 1.1 | CAKE + VOLTA source; arch / CUDA / Python / deps recorded | ✅ mostly | VOLTA built; CAKE unavailable → TIRx substitute installed. Still missing: GPU **driver version** on an A100 node (needs a short job) |
| 1.2 | Can VOLTA handle GEMM / softmax / LayerNorm *as we express them*? | 🟡 | Ran only VOLTA's bundled matmul / race cases. Need our own ref+opt kernels → PTX → `volta compare` for all 3; record EQUIV / unknown / unsupported per kernel |
| 1.3 | Isolated env inside workspace, env.sh extended | 🟡 | `envs/tirx` (Py 3.13) exists. Add CUDA PyTorch matching the node driver; one activation script |
| 1.4 | PyTorch refs in `ref/` incl. non-divisible shapes | ⬜ | gemm / softmax / layernorm; fp64 "golden" mode for numerical edge cases |
| 1.5 | Correctness harness in `harness/`, dtype-aware tolerances | ⬜ | Must accept test cases produced by Task 2.3 (shape, dtype, value distribution); also guard against all-zeros / partial-write false passes |
| 1.6 | Benchmark harness (CUDA events, warmup, median of N) | ⬜ | L2 flush between runs; report median + spread |
| 1.7 | Slurm scripts in `scripts/slurm/` (1 GPU, tight time) | ⬜ | `-A bidr-delta-gpu -p gpuA100x4[-interactive] --gpus=1`; **ask before submitting** |
| 1.8 | Lumen client wrapper (`$LUMEN_API_KEY`) | ⬜ | Default `glm-5.3`; dry-run mode; budget guard via `/v1/usage`; per-call log (prompt, reply, tokens, coins); retries. **Ask before any real call** |

## Task 2: change-aware test generation (depends on the TIRx-vs-CUDA decision)
| # | Must cover | Notes |
|---|---|---|
| 2.1 | Diff two agent revisions → list of changed ops | Proposed: extract a normalized *feature summary* per version (tile sizes, smem buffers, barriers and positions, index/access expressions, math ops, dtypes, launch config) and diff those. Extractor for the chosen language (TIRx IR or CUDA); optional PTX extractor works for both |
| 2.2 | Classify: tiling / memory access / synchronization / numerical | Rule-based on feature deltas; a change can carry several labels |
| 2.3 | Generate targeted tests per category | tiling → non-divisible / boundary dims (1, tile±1, prime sizes); memory → misaligned / strided; sync → varied sizes and pipeline stages; numerical → large / small magnitudes, softmax overflow, subnormals. Optionally also pick VOLTA shapes so CTA-0 hits the boundary |
| 2.4 | Unit tests for diff, classifier, generator | pytest, CPU-only; use hand-made revision pairs (incl. the injected bugs planned for Task 5) |

## Proposed order
1. (Xiaocheng) Acknowledge `glm-5.3` in the Lumen web UI.
2. 1.8 Lumen client, tested in dry-run (no coins). Then, **with approval**, a GLM pilot: same softmax task in TIRx (docs in prompt) and in CUDA (~0.1–0.2 coins) → decide what the agent writes.
3. 1.3 / 1.4 / 1.5 / 1.6 / 1.7 locally; then, **with approval**, one short interactive A100 job: driver version + harness/bench smoke test.
4. 1.2 VOLTA feasibility on our three kernels (CPU).
5. Task 2 (2.1 → 2.4), CPU-only, unit-tested.

LLM spend in Tasks 1–2 is tiny (pilot only). The real spend is Tasks 4–5. Rough estimate: ~0.74 coins per 10-iteration GLM run
(assumed token counts); e.g. 36 repair runs ≈ 27 coins ≈ 10+ days of refill → start early, keep prompts compact, or ask for a larger pool.
