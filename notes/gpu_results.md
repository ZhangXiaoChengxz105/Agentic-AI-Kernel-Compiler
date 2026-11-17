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

---
# Second A100 run (2026-11-16, Slurm job 24036388, 7 min 24 s): after the restructure + `akc/perf`
Log: `logs/slurm/cs521-smoke-24036388.out`. Unit tests on the GPU node: 101 run, all pass (2 skipped: the slow
NumSim test and one CPU-only test), including the two new GPU perf tests.

## Unchanged after the restructure
- CUDA injected bugs: **A 4/11, B 5/11, C 11/11** (identical to run 24032011).
- GLM CUDA kernels on the 24 all-category tests: only glm_A_r3 (dropped term) fails, as before.
- tirx-lite kernels (hand-written + 4 GLM): 8/8 tests each.

## Speed with roofline numbers (rows=4096, N=1024; GEMM 1024³; median of 50, L2 flushed)
| kernel | ms | × PyTorch | effective GB/s (% of 1555) | TFLOP/s (% of 19.5) | floor ratio |
|---|---|---|---|---|---|
| softmax_ref | 0.637 | 0.06 | 53 (3%) | | 20.0× |
| softmax_opt | 0.0376 | 0.96 | 893 (57%) | | 1.18× |
| softmax_txl_ref (tirx) | 0.0334 | 1.08 | 1006 (65%) | | 1.06× |
| glm_A_r1 (CUDA) | 0.0339 | 1.07 | 991 (64%) | | 1.04× |
| glm_B2_r1 / B3_r1 / B1_r3 | 0.0366 / 0.0392 / 0.0409 | 0.99 / 0.92 / 0.88 | 916 / 855 / 820 | | 1.15–1.28× |
| layernorm_ref | 0.344 | 0.12 | 98 (6%) | | 10.5× |
| layernorm_opt | 0.0377 | 1.12 | 889 (57%) | | 1.16× |
| gemm_ref | 0.910 | 0.19 | | 2.36 (12%) | 8.3× (vs roofline) |
| gemm_opt | 0.606 | 0.28 | | 3.54 (18%); PyTorch 12.57 (64%) | 5.5× (vs roofline) |
A device copy of the same 33.6 MB takes 0.031–0.033 ms (≈1020–1050 GB/s, 66–68% of the datasheet peak): the
practical floor for the memory-bound ops. The best softmax kernels are within 4–6% of it.
tirx-lite GLM softmax kernels (akc tirx-gpu): 1.11–1.12× PyTorch (1.15× in run 24032011: run-to-run noise ~5%).

## Problems this run found (fixed afterwards, see notes/build_log.md §7)
1. **False alarm in the change-aware numerical test (Task 2).** `akc validate --config C` on our correct
   layernorm_opt failed the "offset" test (mean 1e4, std 1): 2159/8000 outputs off by up to 1.4e-3 vs atol 1e-4.
   CPU emulation: an ideal float32 LayerNorm (exact mean rounded to float32) fails ~1100/8000 and PyTorch's own
   float32 layer_norm ~2200/8000: rounding the mean costs ~u·|mean|/std ≈ 6e-4, unavoidable. Fix: condition-aware
   tolerance (`OpSpec.magnitude` for layernorm). Every correct agent kernel would otherwise have been rejected.
2. **`akc eval-bugs --lang tirx` crashed** (StopIteration): it looked for a bug-free tirx "candidate" to compare the
   bugs with; tirx bugs are edits of the reference. Not covered by any test before; fixed + test added.
3. Duplicate "related change" lines in the feedback (two classifier rules labelled the same line). Deduplicated.
4. The perf path itself worked as designed: glm_A_r3 was not timed ("incorrect at the benchmark shape"), and the
   failing layernorm_opt observation contained no speed section (speed only for passing candidates).

# Third A100 run (2026-11-16, Slurm job 24036943, 9 min 12 s): the fixes, verified
- Unit tests on the GPU node: 105 run, all pass (2 skipped, as before).
- CUDA injected bugs: A 4/11, B 5/11, C 11/11 (unchanged; layernorm_bug_onepass is still caught: nonfinite).
- **tirx injected bugs now run** (`akc eval-bugs --lang tirx`): A 1/4, B 1/4, C 4/4. VOLTA is n/a for tirx (not wired
  up yet, Task 3: TIRx's generated CUDA calls `expf`, which compiles to `ex2.approx`; VOLTA needs `__symexpf` as in
  our CUDA kernels' EXP() macro. VOLTA has not been tried on tirx PTX), so B = A; the change-aware tests catch tail / off-by-one / nomax.
- **layernorm_opt passes configuration C** (fixed tests, change-aware incl. the offset test, VOLTA) and the
  observation now ends with its speed section (1.13x PyTorch, 57% of peak bandwidth, floor 1.17x).
- Speeds within ~2% of run 24036388 (e.g. softmax_opt 0.0377 ms, layernorm_opt 0.0381 ms, gemm_opt 0.6085 ms).

## Mutation detection, A vs C (2026-11-16, jobs 24051467 / 24052255 / 24052791)
`akc mutants --detect --per-op 2`: generated single edits of 8 tuning-set kernels and 5 correct held-out GLM
kernels. C = fixed + change-aware tests (no VOLTA).
- Tuning set: A 83/136, C 105/136.
- Held out: A 45/67, C 53/67.
- No-op edits: 0 false alarms.

VOLTA was run on C's 31 CUDA misses:
- It adds 14 catches: races and out-of-bounds shared access.
- It proves 11 equivalent over the reals. 6 of those 11 are not harmless in float: the max reduction sees only
  half the warp, and a CPU model gives NaN for a row with one large value.

Details: notes/build_log.md §8c–8e.

# Operator-agnostic checkers on the bugs our tests miss (2026-11-17, Slurm job 24059082, 4 min 17 s)
**Question.** Do general GPU checkers catch the bugs our tests miss, without false alarms on correct kernels? The
checkers are compute-sanitizer and GPU-FPX, and they know nothing about the operator.

**Files.**
- Script: `scripts/experiments/sanitizer_trial.py`.
- Results: `results/eval/sanitizer_trial.json`.
- Raw tool logs: `logs/sanitizer_trial/`.
- PTX: `build/sanitizer_trial/`.
- Command: `sbatch --time=00:20:00 scripts/slurm/gpu_job.sbatch python scripts/experiments/sanitizer_trial.py run`.
- GPU use: one job, 4 min 17 s on one A100 (driver 595.71.05, compute-sanitizer 2026.1.1).

## Setup
- **Kernels.** 38 distinct CUDA sources, standing for 41 kernels. The mutants were regenerated with
  `akc.evaluation.mutants`.
  - **Correct bases (7):** softmax_opt, layernorm_opt, gemm_opt, glm_A_r1, glm_B1_r3, glm_B2_r1, glm_B3_r1.
    Any report on these is a false alarm.
  - **glm_B1_r1:** the GLM draft with a known shared-memory race.
  - **The 11 hand-injected CUDA bugs.**
  - **The 14 race / out-of-bounds mutants that only VOLTA caught:** RACE or OUT_OF_BOUNDS in
    `mutants_volta_misses.json`, without glm_B1_r1.
  - **The 8 shift-only mutants.** They have 5 distinct sources, because const 16→8 and shfl_off 16→8 make the
    same edit on a line.
- **Launcher.** A small driver-API launcher in the script, not the test runner. There are two reasons:
  - each tensor gets its own exact-size `cuMemAlloc`. The runner pads outputs with canaries, which would hide
    out-of-bounds writes from memcheck;
  - no PyTorch GPU kernels run. All reports were still filtered by our entry names. The only messages not tied to
    a kernel were initcheck "error buffer overflow" warnings.

  Inputs, the float64 golden (computed on the CPU) and the tolerances come from `OpSpec`, so each run also records
  the usual value check.
- **Shapes.**
  - Row ops: rows×N = 64×256, 64×1000 and 4×8193. 8193 is GLM's large-row path.
  - GEMM: 33×33×40 and 64×64×64.
  - Inputs are N(0,1). The shift-only mutants also get the on-disk `spike` distribution (one outlier per row; 64
    rows, so every warp lane is hit).
- **Tools.**
  - memcheck with `--padding 256 --destroy-on-device-error kernel`;
  - racecheck with `--racecheck-detect-level info`. INFO hazards were recorded, but none occurred;
  - synccheck;
  - initcheck with `--initcheck-address-space all`, which covers global and shared memory.

  One process per kernel and tool, 6 in parallel. "Caught" means at least one error or warning that names our
  kernel.
- **GPU-FPX.**
  - GPU-FPX ships as patches over NVBit 1.5.5, whose README requires a driver ≤ 510.xx. We applied the patches to
    the 1.5.5 sample sources and built them against NVBit 1.8 (CUDA 13.2 headers), for sm_80.
  - That needed three small API changes in `third_party/gpu-fpx-build/` (gitignored):
    - `nvbit_get_func_addr(ctx, f)`;
    - CUDA allocations moved from `nvbit_at_ctx_init` into `nvbit_tool_init`;
    - the receiver thread registered with `nvbit_set_tool_pthread`.
  - NVBit 1.8's README says driver ≤ 575.xx, but it worked on 595:
    - the stock `instr_count` sample counted softmax_opt's instructions;
    - GPU-FPX's positive control (softmax_bug_nomax, `large` inputs) reported INF and NaN at lines 36, 40 and 46.
  - The total setup effort was about 20 minutes.

## Results, compute-sanitizer (caught / total; inputs N(0,1), plus `spike` for shift-only)
| group | memcheck | racecheck | synccheck | initcheck | any of the four |
|---|---|---|---|---|---|
| correct bases (false alarms) | 0/7 | 0/7 | 0/7 | 0/7 | **0/7** |
| glm_B1_r1 (known race) | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 |
| 11 injected bugs | 3/11 | 3/11 | 0/11 | 0/11 | 6/11 |
| 14 VOLTA-only race/OOB mutants | 0/14 | 4/14 | 0/14 | 2/14 | 6/14 |
| 8 shift-only mutants | 0/8 | 1/8* | 0/8 | 0/8 | 1/8* |

\* glm_B1_r1_shfl_off_L52. racecheck reports the **base's** race, at the same lines 66/94 as glm_B1_r1 itself. So
nothing here can be attributed to the shift edit.

**What each tool caught.**
- **memcheck:** the global out-of-bounds bugs.
  - softmax_bug_offbyone and layernorm_bug_offbyone, at their source lines. Both read past the last row.
  - gemm_bug_guard, which writes past C (line 36).
  - These need the code path to run, not particular values.
- **racecheck:** every deleted barrier and every real write collision.
  - Injected: softmax / layernorm / gemm `_bug_nosync`. **layernorm_bug_nosync** had been caught only by VOLTA
    before; its outputs are correct in every run here.
  - Mutants: softmax_opt / layernorm_opt with the barrier deleted, and glm_A_r1 L58 / glm_B3_r1 L18
    (`smax[4]` → `smax[2]`, so warps 2 and 3 write onto `ssum`).
  - glm_B1_r1's real race.
- **initcheck (shared):** softmax_opt L23 / L35, `BLOCK / 2` → `BLOCK / 1`. The tree reduction reads `red[tid + 128]`,
  past the 128-float array, which is uninitialized shared memory.
- **synccheck:** nothing. No kernel in the set has a divergent barrier.

**What none of the four caught.**
- **6 `if (lane == 0)` negations.** Lanes 1–31 all write the same value to the same slot. VOLTA calls this a race,
  but outputs never change, and racecheck reports nothing, not even at INFO level. We count these as harmless.
- **2 halved `ssum` arrays (glm_A_r1 L59, glm_B3_r1 L19).**
  - Warps 2 and 3 write `ssum[2..3]`, past the array, then read their own values back.
  - memcheck does not flag it: the address is still inside the block's shared-memory window. We saw the same for
    softmax_opt L23: initcheck reported reads at shared address 0x280, past the 512-byte array, and memcheck stayed
    silent. [assumed cause: shared memory is reserved in larger units than the array]
  - Only VOLTA's bounds check sees these two.
- **Value-only bugs:**
  - the 3 tail bugs and gemm_bug_ktail;
  - softmax_bug_nomax and layernorm_bug_onepass;
  - all 8 shift-only edits.

  These need the right inputs. The value check in the same runs caught the tails and ktail at N = 1000 / K = 40.

**Taken together.**
- Among the 14 VOLTA-only mutants, the sanitizers catch 6. Of the 8 they miss, 6 are harmless same-value writes and
  2 are out-of-bounds writes that stay inside the block's shared memory. So they catch 6 of the 8 harmful race/OOB
  edits.
- On the 11 injected bugs, fixed tests ∪ sanitizers = 6/11, against A's 4/11. They add gemm_bug_guard and
  layernorm_bug_nosync.

## Results, GPU-FPX (detector)
| kernels | input | INF or NaN reported | NaN reported | value check fails |
|---|---|---|---|---|
| 8 shift-only mutants | N(0,1) | 5/8 (INF only) | 0/8 | 0/8 |
| 8 shift-only mutants | spike | 8/8 | 8/8 | 8/8 |
| their 5 bases (4 correct + glm_B1_r1) | N(0,1) | 3/5 (INF only) | 0/5 | 0/5 |
| their 5 bases | spike | 3/5 (INF only) | 0/5 | 0/5 |
| softmax_bug_nomax / softmax_opt | large | INF+NaN / SUB only | yes / no | fails / passes |
| layernorm_bug_onepass / layernorm_opt | offset | NaN / nothing | yes / no | fails / passes |

- **INF alone is not a bug signal.** Correct online-softmax kernels produce −INF on purpose. In glm_A_r1, lines 42
  and 51 compute `m − mn` with `m = −INFINITY`. GPU-FPX reports these in correct glm_A_r1, glm_B1_r3 and glm_B2_r1.
- **NaN, or an exception kind the parent does not raise on the same input, separates them cleanly on this set:**
  - 8/8 shift-only mutants with spike;
  - 0/8 with N(0,1);
  - 0 false alarms on the bases;
  - both numerical injected bugs, with `large` / `offset` inputs.
- **GPU-FPX does not remove the need for the right input.** Without the outlier no NaN ever happens. With the
  outlier, the value check already fails all 8.
  - Its extra value is a source line for the first exception, and catching exceptions that the output hides (not
    seen in this set).
  - The line was missing ("/unknown_path") for every GLM-kernel NaN. We did not investigate why.

## Cost
- **Per run** (one child process: Python + torch import, CUDA context, the tool, 2–6 small launches): median
  memcheck 8.1 s, racecheck 7.5 s, synccheck 7.7 s, initcheck 7.8 s, GPU-FPX 6.6 s; at most 14.6 s.
- **Process start dominates.** The first run with no tool at all took 10.3 s, on a cold cache.
- **Whole trial:** 176 runs in 252 s of wall time, 6 in parallel.

## Problems
- **GPU-FPX needed the NVBit 1.8 port above.** The stock build, NVBit 1.5.5, was not tried on the GPU because its
  README requires a driver ≤ 510.
- **A parser bug in the first summary.** GPU-FPX prints "/unknown_path in [kernel]:0" with spaces when it has no
  line info, and my first regex missed those lines.
  - The summary printed by the job therefore under-counted GPU-FPX.
  - The fix was to re-parse the saved logs on the CPU (`sanitizer_trial.py reparse`). The JSON holds the corrected
    numbers; nothing was re-run.
- **initcheck output was truncated.** It printed 771 errors for softmax_opt L23 and dropped 16k records (error
  buffer overflow). This is harmless for a yes/no answer.
- **glm_B1_r1's race shows in values only sometimes.** Across the base and its shift mutant there were 10 runs on
  N(0,1) inputs. 2 of them gave a value mismatch: the base at N = 1000 under synccheck, and the mutant at N = 256.

**Takeaway for the design** (`notes/design_universal_tests.md`):
- racecheck, memcheck and shared-memory initcheck are cheap, operator-agnostic and quiet on correct kernels.
  They take over most of VOLTA's race/OOB role and should run on every revision.
- GPU-FPX is a localizer, not a detector. The inputs still have to be searched for.

## D3: outlier inputs on correct kernels (2026-11-17, job 24061002, 2 min 14 s)
- Every correct kernel passes every outlier test: CUDA ref / opt of softmax, LayerNorm and GEMM; 4 GLM CUDA softmax
  kernels; softmax_txl_ref and 4 GLM tirx kernels.
- Details are in `notes/build_log.md` (D3).

# Task 1: tirx-lite LayerNorm and GEMM on the A100 (2026-11-17, jobs 24061076 and 24061108)
Jobs:
- 24061076: 4 min 56 s, first run;
- 24061108: 5 min 19 s, rerun after the LayerNorm reference fix.

Both ran `scripts/slurm/tirx_task1.sh`. Logs are in `logs/slurm/tirx_task1-<job>.out`; data in
`results/eval/tirx_task1.json` (rerun), plus `results/perf/tirx_task1_20261117-033945.json` and
`results/perf/bench_{layernorm,gemm}_tirx_20261117-0341*.json`.

**Tests per kernel.** GPU: the fixed suite plus change-aware-style tests inside the domain:
- LayerNorm: 15 tests (N = 1, 127, 129, 200, 4097, 16384; repeats; offset, large, tiny, constant, outlier@128);
- GEMM: 17 tests (size 1, M / N / K % 16 != 0, (4096, 17, 33); repeats; large, mixed, tiny, outlier@16).

NumSim runs 2 tests: LayerNorm rows = 2, N = 200; GEMM 12 x 12 x 40. The first test also runs Racecheck.

| kernel | GPU tests | NumSim (CPU) | VOLTA | caught by |
|---|---|---|---|---|
| layernorm_txl_ref | 15/15 | 2/2 | EQUIV x3 | - |
| layernorm_txl_opt | 15/15 | 2/2 | EQUIV x3 | - (no false alarm) |
| layernorm_txl_bug_nosync | 15/15 (missed, like CUDA) | race | RACE x3 | NumSim, VOLTA |
| layernorm_txl_bug_tail | 5/15 (mismatch) | mismatch | NOT_EQUIV at N = 200, 100 | all three |
| layernorm_txl_bug_offbyone | 0/15 (mismatch, then illegal address at 64 x 4096) | out_of_bounds | NOT_EQUIV x3 | all three |
| layernorm_txl_bug_onepass | 14/15 (nonfinite on offset) | nonfinite on offset | EQUIV (blind spot) | GPU, NumSim |
| gemm_txl_ref | 17/17 | 2/2 | EQUIV x5 | - |
| gemm_txl_opt | 17/17 | 2/2 | EQUIV x5 | - (no false alarm) |
| gemm_txl_bug_nosync | 14/17 (mismatch; also nondeterministic in 24061076) | race | RACE x5 | all three |
| gemm_txl_bug_ktail | 8/17 (mismatch when K % 16 != 0) | mismatch | NOT_EQUIV when K % 16 != 0 | all three |
| gemm_txl_bug_guard | 9/17 (canary: write past C) | runtime_error | RACE at 12 x 12 x 40 | all three |

- Every bug is caught by at least one check, and the ref / opt kernels raise no alarm.
  - The verdicts match the CUDA twins' (`notes/gpu_results.md` §1).
  - LayerNorm nosync is again missed by the GPU tests; Racecheck and VOLTA catch it.
- NumSim limitations:
  - NumSim stops the gemm guard bug at the out-of-range store into C, with "integer_address_without_binding requires
    an unmodeled analysis contract". The runner reports this as runtime_error, not out_of_bounds.
  - The first NumSim run on a node builds the engine (about 115 s); later kernels take 7-11 s each.

**The LayerNorm reference fix (between the two jobs).**
- In job 24061076 the first tirx reference failed 1 of 15 tests: rows = 64, N = 1000, offset (x ~ 1e4 + N(0, 1)).
  - 3384 of 64000 outputs were out of tolerance (max abs err 0.0176).
  - Cause: the plain serial float32 sum of a row with mean 1e4 drifts by about 1e-2. The tolerance allows about
    (log2 N + 4) u |mean| / std, which assumes tree-like accumulation.
  - A CPU float32 emulation reproduces it: 3817 of 64000, max 0.0168.
- Fix: mean = x0 + sum(x - x0) / n, which is equal over the reals.
  - Emulation: max err 6.4e-4. Job 24061108: 15/15 pass. VOLTA: still EQUIV.
- **The legacy CUDA `layernorm_ref` has the same problem.** In job 24061108, `akc check` on that test gave:
  - layernorm_ref: 0/1, mismatch;
  - layernorm_opt: 1/1.
  - It was never tested on offset data before, because tests run on candidates and the reference is not a candidate.
  - The CUDA reference was left unchanged (legacy).

**Speed** (rows = 4096, N = 1024; GEMM 1024^3; median of 50, L2 flushed):

| kernel | job 24061076 | job 24061108 | x PyTorch (24061076 / 24061108) | throughput (24061108) |
|---|---|---|---|---|
| layernorm_txl_ref | 0.357 ms (before the fix) | 0.295 ms | 0.12 / 0.13 | 114 GB/s |
| layernorm_txl_opt | 0.0383 ms | 0.0348 ms | 1.12 / 1.07 | 965 GB/s (62% of peak) |
| gemm_txl_ref | 1.335 ms | 1.089 ms | 0.13 / 0.12 | 1.97 TFLOP/s |
| gemm_txl_opt | 0.608 ms | 0.471 ms | 0.28 / 0.29 | 4.56 TFLOP/s |

- The two nodes differ by up to 20% in absolute time, PyTorch included (GEMM 0.171 vs 0.135 ms). Ratios to PyTorch
  are stable.
- The tirx versions match the CUDA ones in the same position:
  - layernorm_opt 1.12-1.19x;
  - gemm_opt 0.28x, 3.5 TFLOP/s in the earlier runs above.
- `akc bench --lang tirx` works unchanged for both workloads.
- Unit tests `tests.validation.test_tirx_kernels` on the A100: 13 run, all pass (GPU tests included), in both jobs.

GPU-job time for Task 1: 2 jobs, about 10 min 15 s.
